/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonMemoryAccess/MemoryAccessTags.h"
#include "TritonToGraph/EntryArgPointerAliasAnalysis.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/IndependentRowReductionAnalysis.h"
#include "TritonToGraph/ProgramAxisDependenceAnalysis.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

constexpr llvm::StringLiteral kIndependentAxisTensorizeMarkerAttr =
    "hacc.independent_axis_tensorize";
constexpr llvm::StringLiteral kPersistentTaskStripMiningMarkerAttr =
    "hacc.persistent_task_strip_mining";
constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";

constexpr std::array<unsigned, 3> kMergeTensorizeFactors = {2, 4, 8};
constexpr std::array<unsigned, 2> kMergeLargeTensorizeFactors = {16, 32};
constexpr std::array<unsigned, 4> kNormTensorizeFactors = {2, 4, 8, 16};
constexpr std::array<unsigned, 7> kNormPersistentBlockTCandidates = {
    2, 4, 8, 16, 32, 64, 128};

constexpr uint64_t kMergeSplitMaxTensorizedPlanes = 16;

constexpr int64_t kMergeSplitD192SourceTile = 256;
constexpr int64_t kMergeSplitD192TargetTile = 192;
constexpr unsigned kDynamicSingleMomentBlockT = 4;

enum class TensorizeForm : uint8_t {
  MergeSplit,
  NormRope,
  GuardedBatch,
  IntegerRowReduction,
};

enum class IATCandidateScope : uint8_t {
  Default,
  MergeSplitLarge,
};

struct IATCandidate {
  triton::FuncOp function;
  Operation *anchor = nullptr;
  TensorizeForm form = TensorizeForm::MergeSplit;
  int32_t axis = 1;
  int64_t splitExtent = 0;
  int64_t dimExtent = 0;
  unsigned factor = 1;
  bool usesMergeSplitSubCorePolicy = false;
  bool usesMergeSplitD192Tile = false;
  bool requiresPersistentChaining = false;
  bool guardedBatch = false;
  ResourceSnapshot resources;
  CandidateEvaluation evaluation;
};

struct JointProgramMappingCandidate {
  IATCandidate iat;
  unsigned blockT = 0;
  std::array<uint64_t, 3> finalLogicalGrid = {1, 1, 1};
  std::array<uint64_t, 3> finalPhysicalGrid = {1, 1, 1};
  CandidateEvaluation evaluation;
};

struct TensorizeReductionShape {
  int64_t splitExtent = 0;
  int64_t dimExtent = 0;
};

struct TensorizeFormMatch {
  TensorizeForm form;
  TensorizeReductionShape reduction;
};

struct MappedValue {
  Value value;
  bool tensorized = false;
  int64_t laneAxis = 0;
};

std::optional<triton::GetProgramIdOp> findOnlyProgramId(triton::FuncOp function,
                                                        int32_t axis) {
  std::optional<triton::GetProgramIdOp> result;
  unsigned count = 0;
  function.walk([&](triton::GetProgramIdOp pid) {
    if (pid.getAxisAsInt() != axis)
      return;
    ++count;
    if (count == 1)
      result = pid;
  });
  return count == 1 ? result : std::nullopt;
}

std::optional<TensorizeFormMatch>
classifyTensorizeForm(triton::FuncOp function) {
  std::optional<TensorizeReductionShape> mergeReduction;
  std::optional<TensorizeReductionShape> normReduction;
  function.walk([&](triton::ReduceOp reduce) {
    if (reduce.getSrcs().size() != 1 || reduce.getResults().size() != 1 ||
        reduce.getAxis() != 0)
      return;
    auto source =
        dyn_cast<RankedTensorType>(reduce.getSrcs().front().getType());
    if (!source || !source.hasStaticShape())
      return;
    Type result = reduce.getResults().front().getType();
    auto rankedResult = dyn_cast<RankedTensorType>(result);
    if (source.getRank() == 2 && rankedResult && rankedResult.getRank() == 1 &&
        rankedResult.hasStaticShape() &&
        source.getShape()[1] == rankedResult.getShape()[0]) {
      if (!mergeReduction)
        mergeReduction =
            TensorizeReductionShape{source.getShape()[0], source.getShape()[1]};
    } else if (source.getRank() == 1 && !isa<RankedTensorType>(result)) {
      if (!normReduction)
        normReduction = TensorizeReductionShape{0, source.getShape()[0]};
    }
  });
  if (mergeReduction)
    return TensorizeFormMatch{TensorizeForm::MergeSplit, *mergeReduction};
  if (normReduction)
    return TensorizeFormMatch{TensorizeForm::NormRope, *normReduction};
  return std::nullopt;
}

bool hasSupportedControlFlow(triton::FuncOp function) {
  Region &body = function.getBody();
  if (!body.hasOneBlock())
    return false;
  for (Operation &operation : body.front()) {
    if (isa<CallOpInterface>(operation))
      return false;
    StringRef name = operation.getName().getStringRef();
    if (name.contains("atomic") || name.contains("barrier") ||
        name == "cf.br" || name == "cf.cond_br" || name.starts_with("scf."))
      return false;
    if (operation.getNumRegions() != 0 &&
        !isa<triton::ReduceOp, triton::ScanOp>(operation))
      return false;
  }
  return true;
}

bool hasConflictingLaunchContract(ModuleOp module) {
  return module->hasAttr(kIndependentAxisTensorizeMarkerAttr) ||
         module->hasAttr(kProgramGridTransformsAttr) ||
         module->hasAttr(kCoalesceFactorAttr) ||
         module->hasAttr(kCoalesceAxisAttr) ||
         module->hasAttr(kCoalesceGridCeilDivAttr);
}

bool hasDisjointWriteReadRoots(
    const ProgramAxisDependence &dependence,
    const EntryArgPointerAliasAnalysis &entryPointerAliases) {
  for (const StoreAddressDependence &store : dependence.stores) {
    auto storeOp = dyn_cast_or_null<triton::StoreOp>(store.store);
    if (!storeOp)
      return false;
    for (Operation *operation : dependence.dependenceClosure) {
      auto load = dyn_cast<triton::LoadOp>(operation);
      if (!load)
        continue;
      if (entryPointerAliases.classify(storeOp.getPtr(), load.getPtr()) !=
          EntryArgPointerRelation::DistinctEntryRoots)
        return false;
    }
  }
  return true;
}

bool hasTensorPointerInProgramAxisClosure(
    const ProgramAxisDependence &dependence) {
  // A tensor pointer requires a scalar base pointer. IAT turns the dependent
  // program id into a vector of lanes, and the current materializer cannot
  // reconstruct a legal tensor pointer from that vector. Reject this case
  // while discovering candidates instead of scheduling a plan that fails in
  // apply().
  return llvm::any_of(dependence.dependenceClosure, [](Operation *operation) {
    return isa<triton::MakeTensorPtrOp>(operation);
  });
}

int64_t getPreferredLaneAxis(Type originalType, const IATCandidate &candidate) {
  auto tensor = dyn_cast<RankedTensorType>(originalType);
  if (!tensor || candidate.form == TensorizeForm::NormRope)
    return 0;
  if (candidate.form == TensorizeForm::GuardedBatch ||
      candidate.form == TensorizeForm::IntegerRowReduction)
    return 0;
  const int64_t rank = tensor.getRank();
  if (rank == 0)
    return 0;
  if (rank == 1)
    return tensor.getShape().front() == candidate.splitExtent ? 1 : 0;
  if (tensor.getShape().front() == candidate.splitExtent ||
      tensor.getShape().front() == 1)
    return 1;
  return 0;
}

std::optional<RankedTensorType>
getLiftedTensorType(Type originalType, const IATCandidate &candidate,
                    int64_t laneAxis) {
  Type elementType = originalType;
  SmallVector<int64_t> shape;
  if (auto tensor = dyn_cast<RankedTensorType>(originalType)) {
    if (laneAxis < 0 || laneAxis > tensor.getRank())
      return std::nullopt;
    shape.assign(tensor.getShape().begin(), tensor.getShape().end());
    elementType = tensor.getElementType();
  } else if (laneAxis != 0) {
    return std::nullopt;
  }
  shape.insert(shape.begin() + laneAxis, candidate.factor);
  return RankedTensorType::get(shape, elementType);
}

std::optional<RankedTensorType>
getSameShapeTensorType(Type originalType, ArrayRef<int64_t> shape) {
  Type elementType = originalType;
  if (auto tensor = dyn_cast<RankedTensorType>(originalType))
    elementType = tensor.getElementType();
  return RankedTensorType::get(shape, elementType);
}

bool hasSameShape(RankedTensorType lhs, RankedTensorType rhs) {
  return lhs.getRank() == rhs.getRank() &&
         llvm::equal(lhs.getShape(), rhs.getShape());
}

Value makeZero(IRRewriter &rewriter, Location loc, Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  Type elementType = tensor ? tensor.getElementType() : type;
  if (auto integer = dyn_cast<IntegerType>(elementType)) {
    TypedAttr zero = IntegerAttr::get(integer, 0);
    return tensor ? rewriter.create<arith::ConstantOp>(
                        loc, type, DenseElementsAttr::get(tensor, zero))
                  : rewriter.create<arith::ConstantOp>(loc, type, zero);
  }
  if (auto floating = dyn_cast<FloatType>(elementType)) {
    TypedAttr zero = FloatAttr::get(floating, 0.0);
    return tensor ? rewriter.create<arith::ConstantOp>(
                        loc, type, DenseElementsAttr::get(tensor, zero))
                  : rewriter.create<arith::ConstantOp>(loc, type, zero);
  }
  return Value();
}

bool rebuildTensorizedFunction(triton::FuncOp function,
                               const IATCandidate &candidate,
                               IRRewriter &rewriter) {
  if (!hasSupportedControlFlow(function))
    return false;
  if (failed(addProgramGridHiddenExtentArguments(function)))
    return false;
  Value originalExtent =
      function.getArgument(function.getNumArguments() - 2 + candidate.axis);
  auto originalExtentType = dyn_cast<IntegerType>(originalExtent.getType());
  if (!originalExtentType || originalExtentType.getWidth() != 32)
    return false;
  std::optional<triton::GetProgramIdOp> targetPid =
      findOnlyProgramId(function, candidate.axis);
  if (!targetPid)
    return false;

  ProgramAxisDependenceAnalysis analysis(function);
  const ProgramAxisDependence &dependence = analysis.get(candidate.axis);
  if (!dependence.isProgramMappingTransformCandidate())
    return false;
  DenseSet<Operation *> closure;
  for (Operation *operation : dependence.dependenceClosure)
    closure.insert(operation);

  Block &block = function.getBody().front();
  SmallVector<Operation *> originals;
  originals.reserve(block.getOperations().size());
  for (Operation &operation : block)
    originals.push_back(&operation);

  DenseMap<Value, MappedValue> values;
  auto lookup = [&](Value value) -> MappedValue {
    auto it = values.find(value);
    return it == values.end() ? MappedValue{value, false, 0} : it->second;
  };

  auto broadcastTo = [&](Location loc, Value value,
                         RankedTensorType target) -> Value {
    auto source = dyn_cast<RankedTensorType>(value.getType());
    if (!source || source.getElementType() != target.getElementType() ||
        source.getRank() != target.getRank())
      return Value();
    for (auto [sourceDim, targetDim] :
         llvm::zip(source.getShape(), target.getShape()))
      if (sourceDim != targetDim && sourceDim != 1)
        return Value();
    return hasSameShape(source, target)
               ? value
               : rewriter.create<triton::BroadcastOp>(loc, target, value);
  };

  auto alignTensorized = [&](Location loc, MappedValue mapped,
                             RankedTensorType target,
                             int64_t targetLaneAxis) -> Value {
    auto source = dyn_cast<RankedTensorType>(mapped.value.getType());
    if (!source || mapped.laneAxis < 0 || mapped.laneAxis >= source.getRank() ||
        targetLaneAxis < 0 || targetLaneAxis >= target.getRank() ||
        source.getElementType() != target.getElementType())
      return Value();
    const int64_t sourceLaneAxis = mapped.laneAxis;
    const int64_t sourceTail = source.getRank() - sourceLaneAxis - 1;
    const int64_t targetTail = target.getRank() - targetLaneAxis - 1;
    if (sourceLaneAxis > targetLaneAxis || sourceTail > targetTail ||
        source.getShape()[sourceLaneAxis] != candidate.factor ||
        target.getShape()[targetLaneAxis] != candidate.factor)
      return Value();

    SmallVector<int64_t> sourceToTarget(source.getRank(), -1);
    for (int64_t index = 0; index < sourceLaneAxis; ++index)
      sourceToTarget[index] = index;
    sourceToTarget[sourceLaneAxis] = targetLaneAxis;
    for (int64_t index = 0; index < sourceTail; ++index)
      sourceToTarget[sourceLaneAxis + 1 + index] =
          target.getRank() - sourceTail + index;

    SmallVector<bool> targetMapped(target.getRank(), false);
    for (int64_t sourceIndex = 0; sourceIndex < source.getRank();
         ++sourceIndex) {
      const int64_t targetIndex = sourceToTarget[sourceIndex];
      if (targetIndex < 0 || targetIndex >= target.getRank() ||
          targetMapped[targetIndex])
        return Value();
      targetMapped[targetIndex] = true;
      const int64_t sourceDim = source.getShape()[sourceIndex];
      const int64_t targetDim = target.getShape()[targetIndex];
      if (sourceDim != targetDim && sourceDim != 1)
        return Value();
    }

    Value current = mapped.value;
    int64_t currentAxis = 0;
    for (int64_t targetIndex = 0; targetIndex < target.getRank();
         ++targetIndex) {
      if (targetMapped[targetIndex]) {
        ++currentAxis;
        continue;
      }
      current =
          rewriter.create<triton::ExpandDimsOp>(loc, current, currentAxis);
      ++currentAxis;
    }
    return broadcastTo(loc, current, target);
  };

  auto promoteUniform = [&](Location loc, Value value, Type oldType,
                            RankedTensorType target,
                            int64_t targetLaneAxis) -> Value {
    if (targetLaneAxis < 0 || targetLaneAxis >= target.getRank())
      return Value();
    if (auto oldTensor = dyn_cast<RankedTensorType>(oldType)) {
      if (oldTensor.getRank() != target.getRank() - 1 ||
          oldTensor.getElementType() != target.getElementType())
        return Value();
      SmallVector<int64_t> shape(target.getShape().begin(),
                                 target.getShape().end());
      shape.erase(shape.begin() + targetLaneAxis);
      auto unlaned = RankedTensorType::get(shape, target.getElementType());
      Value base = broadcastTo(loc, value, unlaned);
      if (!base)
        return Value();
      Value expanded =
          rewriter.create<triton::ExpandDimsOp>(loc, base, targetLaneAxis);
      return broadcastTo(loc, expanded, target);
    }
    return rewriter.create<triton::SplatOp>(loc, target, value);
  };

  auto liftOperandTo = [&](Value value, RankedTensorType target,
                           int64_t targetLaneAxis) -> Value {
    MappedValue mapped = lookup(value);
    return mapped.tensorized
               ? alignTensorized(value.getLoc(), mapped, target, targetLaneAxis)
               : promoteUniform(value.getLoc(), mapped.value, value.getType(),
                                target, targetLaneAxis);
  };

  auto copyMissingAttrs = [](Operation *from, Operation *to) {
    for (NamedAttribute attribute : from->getAttrs())
      if (!to->hasAttr(attribute.getName()))
        to->setAttr(attribute.getName(), attribute.getValue());
  };

  Value laneMask;
  auto maskForPointerType = [&](Location loc, RankedTensorType pointerType,
                                int64_t laneAxis) -> Value {
    if (!laneMask)
      return Value();
    auto maskType =
        RankedTensorType::get(pointerType.getShape(), rewriter.getI1Type());
    return alignTensorized(loc, MappedValue{laneMask, true, 0}, maskType,
                           laneAxis);
  };

  auto createUnchanged = [&](Operation *operation) -> bool {
    IRMapping mapping;
    for (Value operand : operation->getOperands()) {
      Value mapped = lookup(operand).value;
      if (!mapped)
        return false;
      mapping.map(operand, mapped);
    }
    Operation *replacement = rewriter.clone(*operation, mapping);
    if (replacement->getNumResults() != operation->getNumResults())
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip(operation->getResults(), replacement->getResults()))
      values[oldResult] = {newResult, false, 0};
    return true;
  };

  auto originalTypesHaveSameShape = [](Type lhs, Type rhs) {
    auto left = dyn_cast<RankedTensorType>(lhs);
    auto right = dyn_cast<RankedTensorType>(rhs);
    if (!left || !right)
      return !left && !right;
    return hasSameShape(left, right);
  };

  auto createElementwise = [&](Operation *operation) -> bool {
    if (operation->getNumResults() == 0)
      return false;
    const bool isReshape = isa<triton::ReshapeOp>(operation);
    if (!operation->hasTrait<OpTrait::Elementwise>() && !isReshape)
      return false;
    if (isReshape &&
        (!originalTypesHaveSameShape(operation->getOperand(0).getType(),
                                     operation->getResult(0).getType())))
      return false;

    int64_t laneAxis =
        getPreferredLaneAxis(operation->getResult(0).getType(), candidate);
    for (Value operand : operation->getOperands()) {
      MappedValue mapped = lookup(operand);
      if (!mapped.tensorized ||
          !originalTypesHaveSameShape(operand.getType(),
                                      operation->getResult(0).getType()))
        continue;
      laneAxis = mapped.laneAxis;
      break;
    }
    std::optional<RankedTensorType> firstResultType = getLiftedTensorType(
        operation->getResult(0).getType(), candidate, laneAxis);
    if (!firstResultType)
      return false;

    SmallVector<Type> resultTypes;
    resultTypes.reserve(operation->getNumResults());
    for (Type resultType : operation->getResultTypes()) {
      std::optional<RankedTensorType> lifted =
          getLiftedTensorType(resultType, candidate, laneAxis);
      if (!lifted || !hasSameShape(*lifted, *firstResultType))
        return false;
      resultTypes.push_back(*lifted);
    }

    SmallVector<Value> operands;
    operands.reserve(operation->getNumOperands());
    for (Value operand : operation->getOperands()) {
      std::optional<RankedTensorType> operandType = getSameShapeTensorType(
          operand.getType(), firstResultType->getShape());
      if (!operandType)
        return false;
      Value lifted = liftOperandTo(operand, *operandType, laneAxis);
      if (!lifted)
        return false;
      operands.push_back(lifted);
    }

    if (auto divide = dyn_cast<arith::DivSIOp>(operation)) {
      APInt divisor;
      if (matchPattern(divide.getRhs(), m_ConstantInt(&divisor)) &&
          divisor.isStrictlyPositive() && divisor.isPowerOf2() &&
          firstResultType->getElementType().isInteger(32)) {
        auto constant = [&](int64_t value) -> Value {
          auto attr =
              rewriter.getIntegerAttr(firstResultType->getElementType(), value);
          return rewriter.create<arith::ConstantOp>(
              operation->getLoc(), *firstResultType,
              DenseElementsAttr::get(*firstResultType, attr));
        };
        Value sign = rewriter.create<arith::ShRSIOp>(operation->getLoc(),
                                                     operands[0], constant(31));
        Value bias = rewriter.create<arith::AndIOp>(
            operation->getLoc(), sign, constant(divisor.getSExtValue() - 1));
        Value corrected = rewriter.create<arith::AddIOp>(operation->getLoc(),
                                                         operands[0], bias);
        Value quotient = rewriter.create<arith::ShRSIOp>(
            operation->getLoc(), corrected, constant(divisor.logBase2()));
        values[operation->getResult(0)] = {quotient, true, laneAxis};
        return true;
      }
    }

    Operation *replacement = rewriter.create(
        operation->getLoc(), operation->getName().getIdentifier(), operands,
        resultTypes, operation->getAttrs());
    if (replacement->getNumResults() != operation->getNumResults())
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip(operation->getResults(), replacement->getResults()))
      values[oldResult] = {newResult, true, laneAxis};
    return true;
  };

  for (Operation *operation : originals) {
    rewriter.setInsertionPoint(operation);

    if (operation == targetPid->getOperation()) {
      Operation *scalarPid = rewriter.clone(*operation);
      if (scalarPid->getNumResults() != 1)
        return false;
      Value scalar = scalarPid->getResult(0);
      auto integer = dyn_cast<IntegerType>(scalar.getType());
      if (!integer || integer.getWidth() != 32)
        return false;
      Value factor = rewriter.create<arith::ConstantIntOp>(
          operation->getLoc(), candidate.factor, integer.getWidth());
      Value base =
          rewriter.create<arith::MulIOp>(operation->getLoc(), scalar, factor);
      auto laneType =
          RankedTensorType::get({candidate.factor}, scalar.getType());
      Value lane = rewriter.create<triton::MakeRangeOp>(
          operation->getLoc(), laneType, 0, candidate.factor);
      Value baseSplat =
          rewriter.create<triton::SplatOp>(operation->getLoc(), laneType, base);
      Value logicalIds =
          rewriter.create<arith::AddIOp>(operation->getLoc(), baseSplat, lane);
      Type i64 = rewriter.getI64Type();
      auto laneI64Type = RankedTensorType::get({candidate.factor}, i64);
      Value logicalIdsI64 = rewriter.create<arith::ExtUIOp>(
          operation->getLoc(), laneI64Type, logicalIds);
      Value extentI64 = rewriter.create<arith::ExtUIOp>(operation->getLoc(),
                                                        i64, originalExtent);
      Value extentSplat = rewriter.create<triton::SplatOp>(
          operation->getLoc(), laneI64Type, extentI64);
      auto runtimeExtentMask = rewriter.create<arith::CmpIOp>(
          operation->getLoc(), arith::CmpIPredicate::ult, logicalIdsI64,
          extentSplat);
      runtimeExtentMask->setAttr(
          mlir::triton::memory_access::IATRuntimeExtentUnsignedMaskTAG,
          rewriter.getUnitAttr());
      laneMask = runtimeExtentMask;
      values[operation->getResult(0)] = {logicalIds, true, 0};
      continue;
    }

    bool tensorized = closure.contains(operation);
    for (Value operand : operation->getOperands())
      tensorized |= lookup(operand).tensorized;

    if (auto assertion = dyn_cast<triton::AssertOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      if (!assertion->hasAttr("tt.auto_overflow_assert"))
        return false;
      auto condition = lookup(assertion.getCondition());
      auto type = dyn_cast<RankedTensorType>(condition.value.getType());
      if (!type || !condition.tensorized)
        return false;
      Value mask =
          maskForPointerType(operation->getLoc(), type, condition.laneAxis);
      if (!mask)
        return false;
      Value inactive = rewriter.create<arith::CmpIOp>(
          operation->getLoc(), arith::CmpIPredicate::eq, mask,
          makeZero(rewriter, operation->getLoc(), mask.getType()));
      Value guarded = rewriter.create<arith::OrIOp>(operation->getLoc(),
                                                    condition.value, inactive);
      auto replacement = rewriter.create<triton::AssertOp>(
          operation->getLoc(), guarded, assertion.getMessage());
      copyMissingAttrs(operation, replacement.getOperation());
      continue;
    }

    if (auto splat = dyn_cast<triton::SplatOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      const int64_t laneAxis = getPreferredLaneAxis(splat.getType(), candidate);
      std::optional<RankedTensorType> resultType =
          getLiftedTensorType(splat.getType(), candidate, laneAxis);
      if (!resultType)
        return false;
      Value result = liftOperandTo(splat.getSrc(), *resultType, laneAxis);
      if (!result)
        return false;
      values[splat.getResult()] = {result, true, laneAxis};
      continue;
    }

    if (auto expand = dyn_cast<triton::ExpandDimsOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue sourceMapped = lookup(expand.getSrc());
      const int64_t oldLaneAxis =
          sourceMapped.tensorized
              ? sourceMapped.laneAxis
              : getPreferredLaneAxis(expand.getSrc().getType(), candidate);
      std::optional<RankedTensorType> sourceType = getLiftedTensorType(
          expand.getSrc().getType(), candidate, oldLaneAxis);
      if (!sourceType)
        return false;
      Value source = liftOperandTo(expand.getSrc(), *sourceType, oldLaneAxis);
      if (!source)
        return false;
      const int64_t newAxis =
          expand.getAxis() + (expand.getAxis() >= oldLaneAxis ? 1 : 0);
      const int64_t resultLaneAxis =
          oldLaneAxis + (expand.getAxis() < oldLaneAxis ? 1 : 0);
      Value result = rewriter.create<triton::ExpandDimsOp>(operation->getLoc(),
                                                           source, newAxis);
      std::optional<RankedTensorType> expected =
          getLiftedTensorType(expand.getType(), candidate, resultLaneAxis);
      if (!expected || result.getType() != *expected)
        return false;
      values[expand.getResult()] = {result, true, resultLaneAxis};
      continue;
    }

    if (auto broadcast = dyn_cast<triton::BroadcastOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue sourceMapped = lookup(broadcast.getSrc());
      const int64_t laneAxis =
          sourceMapped.tensorized
              ? sourceMapped.laneAxis
              : getPreferredLaneAxis(broadcast.getSrc().getType(), candidate);
      std::optional<RankedTensorType> resultType =
          getLiftedTensorType(broadcast.getType(), candidate, laneAxis);
      if (!resultType)
        return false;
      Value result = liftOperandTo(broadcast.getSrc(), *resultType, laneAxis);
      if (!result)
        return false;
      values[broadcast.getResult()] = {result, true, laneAxis};
      continue;
    }

    if (auto reduce = dyn_cast<triton::ReduceOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      if (reduce.getSrcs().empty())
        return false;
      auto oldInputType =
          dyn_cast<RankedTensorType>(reduce.getSrcs().front().getType());
      if (!oldInputType)
        return false;
      MappedValue firstSource = lookup(reduce.getSrcs().front());
      const int64_t laneAxis =
          firstSource.tensorized
              ? firstSource.laneAxis
              : getPreferredLaneAxis(oldInputType, candidate);
      std::optional<RankedTensorType> firstSourceType =
          getLiftedTensorType(oldInputType, candidate, laneAxis);
      if (!firstSourceType)
        return false;
      SmallVector<Value> sources;
      for (Value source : reduce.getSrcs()) {
        std::optional<RankedTensorType> sourceType =
            getLiftedTensorType(source.getType(), candidate, laneAxis);
        if (!sourceType || !hasSameShape(*sourceType, *firstSourceType))
          return false;
        Value lifted = liftOperandTo(source, *sourceType, laneAxis);
        if (!lifted)
          return false;
        sources.push_back(lifted);
      }
      const int64_t axis =
          reduce.getAxis() + (reduce.getAxis() >= laneAxis ? 1 : 0);
      const int64_t resultLaneAxis = laneAxis - (axis < laneAxis ? 1 : 0);
      auto replacement =
          rewriter.create<triton::ReduceOp>(operation->getLoc(), sources, axis);
      IRMapping regionMapping;
      for (auto &entry : values)
        if (!entry.second.tensorized)
          regionMapping.map(entry.first, entry.second.value);
      rewriter.cloneRegionBefore(
          reduce.getCombineOp(), replacement.getCombineOp(),
          replacement.getCombineOp().end(), regionMapping);
      copyMissingAttrs(operation, replacement.getOperation());
      for (auto [oldResult, newResult] :
           llvm::zip(reduce.getResults(), replacement.getResults())) {
        std::optional<RankedTensorType> expected =
            getLiftedTensorType(oldResult.getType(), candidate, resultLaneAxis);
        if (!expected || newResult.getType() != *expected)
          return false;
        values[oldResult] = {newResult, true, resultLaneAxis};
      }
      continue;
    }

    if (auto scan = dyn_cast<triton::ScanOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      if (scan.getSrcs().empty())
        return false;
      auto oldInputType =
          dyn_cast<RankedTensorType>(scan.getSrcs().front().getType());
      if (!oldInputType)
        return false;
      MappedValue firstSource = lookup(scan.getSrcs().front());
      const int64_t laneAxis =
          firstSource.tensorized
              ? firstSource.laneAxis
              : getPreferredLaneAxis(oldInputType, candidate);
      std::optional<RankedTensorType> firstSourceType =
          getLiftedTensorType(oldInputType, candidate, laneAxis);
      if (!firstSourceType)
        return false;
      SmallVector<Value> sources;
      for (Value source : scan.getSrcs()) {
        std::optional<RankedTensorType> sourceType =
            getLiftedTensorType(source.getType(), candidate, laneAxis);
        if (!sourceType || !hasSameShape(*sourceType, *firstSourceType))
          return false;
        Value lifted = liftOperandTo(source, *sourceType, laneAxis);
        if (!lifted)
          return false;
        sources.push_back(lifted);
      }
      const int64_t axis =
          scan.getAxis() + (scan.getAxis() >= laneAxis ? 1 : 0);
      auto replacement = rewriter.create<triton::ScanOp>(
          operation->getLoc(), sources, static_cast<uint32_t>(axis),
          scan.getReverse());
      rewriter.cloneRegionBefore(scan.getCombineOp(),
                                 replacement.getCombineOp(),
                                 replacement.getCombineOp().end());
      copyMissingAttrs(operation, replacement.getOperation());
      for (auto [oldResult, newResult] :
           llvm::zip(scan.getResults(), replacement.getResults())) {
        std::optional<RankedTensorType> expected =
            getLiftedTensorType(oldResult.getType(), candidate, laneAxis);
        if (!expected || newResult.getType() != *expected)
          return false;
        values[oldResult] = {newResult, true, laneAxis};
      }
      continue;
    }

    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue pointerMapped = lookup(load.getPtr());
      const int64_t laneAxis =
          pointerMapped.tensorized
              ? pointerMapped.laneAxis
              : getPreferredLaneAxis(load.getPtr().getType(), candidate);
      std::optional<RankedTensorType> pointerType =
          getLiftedTensorType(load.getPtr().getType(), candidate, laneAxis);
      if (!pointerType)
        return false;
      Value pointer = liftOperandTo(load.getPtr(), *pointerType, laneAxis);
      std::optional<RankedTensorType> maskType =
          getSameShapeTensorType(rewriter.getI1Type(), pointerType->getShape());
      Value mask = load.getMask()
                       ? liftOperandTo(load.getMask(), *maskType, laneAxis)
                       : Value();
      Value laneMaskForLoad =
          maskForPointerType(operation->getLoc(), *pointerType, laneAxis);
      if (laneMaskForLoad)
        mask = mask ? rewriter.create<arith::AndIOp>(operation->getLoc(), mask,
                                                     laneMaskForLoad)
                    : laneMaskForLoad;
      std::optional<RankedTensorType> otherType = getSameShapeTensorType(
          load.getResult().getType(), pointerType->getShape());
      Value other = load.getOther()
                        ? liftOperandTo(load.getOther(), *otherType, laneAxis)
                        : makeZero(rewriter, operation->getLoc(), *otherType);
      if (!pointer || !maskType || !otherType || !other)
        return false;
      auto replacement = rewriter.create<triton::LoadOp>(
          operation->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      copyMissingAttrs(operation, replacement.getOperation());
      if (replacement.getResult().getType() != *otherType)
        return false;
      values[load.getResult()] = {replacement.getResult(), true, laneAxis};
      continue;
    }

    if (auto store = dyn_cast<triton::StoreOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue pointerMapped = lookup(store.getPtr());
      const int64_t laneAxis =
          pointerMapped.tensorized
              ? pointerMapped.laneAxis
              : getPreferredLaneAxis(store.getPtr().getType(), candidate);
      std::optional<RankedTensorType> pointerType =
          getLiftedTensorType(store.getPtr().getType(), candidate, laneAxis);
      if (!pointerType)
        return false;
      Value pointer = liftOperandTo(store.getPtr(), *pointerType, laneAxis);
      std::optional<RankedTensorType> valueType = getSameShapeTensorType(
          store.getValue().getType(), pointerType->getShape());
      if (!valueType)
        return false;
      Value value = liftOperandTo(store.getValue(), *valueType, laneAxis);
      auto maskType =
          RankedTensorType::get(pointerType->getShape(), rewriter.getI1Type());
      Value mask = store.getMask()
                       ? liftOperandTo(store.getMask(), maskType, laneAxis)
                       : Value();
      Value laneMaskForStore =
          maskForPointerType(operation->getLoc(), *pointerType, laneAxis);
      if (laneMaskForStore)
        mask = mask ? rewriter.create<arith::AndIOp>(operation->getLoc(), mask,
                                                     laneMaskForStore)
                    : laneMaskForStore;
      if (!pointer || !value || !laneMaskForStore)
        return false;
      auto replacement = rewriter.create<triton::StoreOp>(
          operation->getLoc(), pointer, value, mask, store.getBoundaryCheck(),
          store.getCache(), store.getEvict());
      copyMissingAttrs(operation, replacement.getOperation());
      continue;
    }

    if (isa<triton::ReturnOp>(operation)) {
      for (Value operand : operation->getOperands())
        if (lookup(operand).tensorized)
          return false;
      if (!createUnchanged(operation))
        return false;
      continue;
    }

    if (!tensorized) {
      if (!createUnchanged(operation))
        return false;
      continue;
    }
    if (!createElementwise(operation))
      return false;
  }

  if (!laneMask)
    return false;
  for (Operation *operation : llvm::reverse(originals)) {
    operation->dropAllUses();
    rewriter.eraseOp(operation);
  }
  return true;
}

LogicalResult runProgramMappingStructuralCleanup(ModuleOp module) {
  PassManager cleanup(module.getContext(), module.getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  return cleanup.run(module);
}

LogicalResult materializeIATCandidateToSandbox(ModuleOp module,
                                               triton::FuncOp function,
                                               const IATCandidate &candidate) {
  if (!module || module->hasAttr(kIndependentAxisTensorizeMarkerAttr) ||
      module->hasAttr(kProgramGridTransformsAttr))
    return failure();
  IRRewriter rewriter(module.getContext());
  if (!rebuildTensorizedFunction(function, candidate, rewriter))
    return failure();
  ProgramGridTransformContract contract;
  contract.transforms.push_back(ProgramGridTransform{
      0, candidate.axis, static_cast<int64_t>(candidate.factor), false, false});
  if (failed(setProgramGridTransformContract(module, contract)))
    return failure();
  module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                  UnitAttr::get(module.getContext()));
  return mlir::verify(module.getOperation());
}

ModuleOp createProgramMappingSandbox(ModuleOp module, triton::FuncOp function) {
  (void)module;
  ModuleOp sandbox = ModuleOp::create(function.getLoc());
  sandbox.getBody()->push_back(function->clone());
  return sandbox;
}

std::optional<IATCandidate>
analyzeDynamicSingleMomentCandidate(GraphOptimizationContext &context) {
  triton::FuncOp function = context.getFunction();
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module || hasConflictingLaunchContract(module) ||
      !hasSupportedControlFlow(function) ||
      classifyIndependentRowReduction(function) !=
          IndependentRowReductionKind::SingleMoment ||
      !findOnlyProgramId(function, 0))
    return std::nullopt;

  std::optional<TensorizeFormMatch> formMatch = classifyTensorizeForm(function);
  if (!formMatch || formMatch->form != TensorizeForm::NormRope)
    return std::nullopt;
  std::optional<triton::GetProgramIdOp> headPid =
      findOnlyProgramId(function, 1);
  if (!headPid)
    return std::nullopt;

  const ProgramAxisDependence &dependence =
      context.getProgramAxisDependenceAnalysis().get(1);
  if (!dependence.isProgramMappingTransformCandidate() ||
      hasTensorPointerInProgramAxisClosure(dependence) ||
      !hasDisjointWriteReadRoots(dependence,
                                 context.getEntryArgPointerAliasAnalysis()))
    return std::nullopt;
  const LiveByteEstimate &liveBytes =
      context.getResourceCostAnalysis().getLiveByteEstimate();
  if (!liveBytes.known)
    return std::nullopt;
  uint64_t estimatedPeak = 0;
  if (__builtin_mul_overflow(liveBytes.peakLiveBytes, uint64_t{16},
                             &estimatedPeak))
    return std::nullopt;

  IATCandidate candidate;
  candidate.function = function;
  candidate.anchor = headPid->getOperation();
  candidate.form = TensorizeForm::NormRope;
  candidate.axis = 1;
  candidate.dimExtent = formMatch->reduction.dimExtent;
  candidate.factor = 16;
  candidate.requiresPersistentChaining = true;
  candidate.resources = context.getResourceCostAnalysis().getResourceSnapshot();

  CandidateCost cost;
  cost.plan.tensorizeFactor = candidate.factor;
  cost.plan.blockT = kDynamicSingleMomentBlockT;
  cost.plan.staticAxisFusionFactor = 1;
  cost.plan.stableId = "iat.ptsm.single-moment.dynamic.f16.b4";
  cost.storeCountBefore = dependence.stores.size();
  cost.storeCountAfter = dependence.stores.size();
  cost.addressCalculationsBefore = dependence.dependenceClosure.size();
  cost.addressCalculationsAfter = dependence.dependenceClosure.size();
  cost.hasPeakLiveBytes = true;
  cost.baselinePeakLiveBytes = liveBytes.peakLiveBytes;
  cost.estimatedPeakLiveBytes = estimatedPeak;
  candidate.evaluation = context.getResourceCostAnalysis().evaluate(cost);
  return candidate.evaluation.accepted ? std::optional<IATCandidate>(candidate)
                                       : std::nullopt;
}

bool sameDynamicSingleMomentCandidate(const IATCandidate &lhs,
                                      const IATCandidate &rhs);

class DynamicSingleMomentProgramMappingPlan final : public RewritePlan {
public:
  DynamicSingleMomentProgramMappingPlan(IATCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    std::optional<IATCandidate> current =
        analyzeDynamicSingleMomentCandidate(context);
    return current && sameDynamicSingleMomentCandidate(candidate, *current)
               ? success()
               : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module || hasConflictingLaunchContract(module))
      return failure();

    ModuleOp sandbox = createProgramMappingSandbox(module, candidate.function);
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    CandidateEvaluation persistentEvaluation;
    if (!clonedFunction ||
        failed(materializeIATCandidateToSandbox(sandbox, clonedFunction,
                                                candidate)) ||
        failed(materializePersistentTaskStripMiningCandidate(
            sandbox, clonedFunction, candidate.resources,
            kDynamicSingleMomentBlockT, &persistentEvaluation, true)) ||
        failed(runProgramMappingStructuralCleanup(sandbox)) ||
        failed(mlir::verify(sandbox.getOperation())))
      return failure();

    const LiveByteEstimate baseline =
        estimatePeakLiveBytes(candidate.function.getOperation());
    const LiveByteEstimate final =
        estimatePeakLiveBytes(clonedFunction.getOperation());
    if (!baseline.known || !final.known)
      return failure();
    CandidateCost finalCost;
    finalCost.plan.tensorizeFactor = candidate.factor;
    finalCost.plan.blockT = kDynamicSingleMomentBlockT;
    finalCost.plan.staticAxisFusionFactor = 1;
    finalCost.plan.stableId = "iat.ptsm.single-moment.dynamic.f16.b4";
    finalCost.hasPeakLiveBytes = true;
    finalCost.baselinePeakLiveBytes = baseline.peakLiveBytes;
    finalCost.estimatedPeakLiveBytes = final.peakLiveBytes;
    if (!evaluateCandidateCost(candidate.resources, finalCost).accepted)
      return failure();

    Attribute transforms = sandbox->getAttr(kProgramGridTransformsAttr);
    if (!transforms)
      return failure();
    FailureOr<ProgramGridTransformContract> parsed =
        parseProgramGridTransformContract(transforms);
    if (failed(parsed) || parsed->transforms.size() != 2 ||
        !hasProgramGridHiddenExtentArguments(clonedFunction))
      return failure();

    if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                    clonedFunction)))
      return failure();
    module->setAttr(kProgramGridTransformsAttr, transforms);
    module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                    UnitAttr::get(module.getContext()));
    module->setAttr(kPersistentTaskStripMiningMarkerAttr,
                    UnitAttr::get(module.getContext()));
    return success();
  }

private:
  IATCandidate candidate;
  unsigned epoch;
};

std::optional<IATCandidate>
analyzeDynamicMergeSplitCandidate(GraphOptimizationContext &context) {
  triton::FuncOp function = context.getFunction();
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module || hasConflictingLaunchContract(module) ||
      !hasSupportedControlFlow(function))
    return std::nullopt;

  std::optional<TensorizeFormMatch> formMatch = classifyTensorizeForm(function);
  if (!formMatch || formMatch->form != TensorizeForm::MergeSplit ||
      !findOnlyProgramId(function, 0))
    return std::nullopt;
  std::optional<triton::GetProgramIdOp> headPid =
      findOnlyProgramId(function, 1);
  if (!headPid)
    return std::nullopt;

  const ProgramAxisDependence &dependence =
      context.getProgramAxisDependenceAnalysis().get(1);
  if (!dependence.isProgramMappingTransformCandidate() ||
      hasTensorPointerInProgramAxisClosure(dependence) ||
      !hasDisjointWriteReadRoots(dependence,
                                 context.getEntryArgPointerAliasAnalysis()))
    return std::nullopt;
  const LiveByteEstimate &liveBytes =
      context.getResourceCostAnalysis().getLiveByteEstimate();
  if (!liveBytes.known)
    return std::nullopt;
  uint64_t estimatedPeak = 0;
  if (__builtin_mul_overflow(liveBytes.peakLiveBytes, uint64_t{16},
                             &estimatedPeak))
    return std::nullopt;

  IATCandidate candidate;
  candidate.function = function;
  candidate.anchor = headPid->getOperation();
  candidate.form = TensorizeForm::MergeSplit;
  candidate.axis = 1;
  candidate.splitExtent = formMatch->reduction.splitExtent;
  candidate.dimExtent = formMatch->reduction.dimExtent;
  candidate.factor = 16;
  candidate.resources = context.getResourceCostAnalysis().getResourceSnapshot();

  CandidateCost cost;
  cost.plan.tensorizeFactor = candidate.factor;
  cost.plan.blockT = 1;
  cost.plan.staticAxisFusionFactor = 1;
  cost.plan.stableId = "iat.merge.dynamic.f16";
  cost.storeCountBefore = dependence.stores.size();
  cost.storeCountAfter = dependence.stores.size();
  cost.addressCalculationsBefore = dependence.dependenceClosure.size();
  cost.addressCalculationsAfter = dependence.dependenceClosure.size();
  cost.hasPeakLiveBytes = true;
  cost.baselinePeakLiveBytes = liveBytes.peakLiveBytes;
  cost.estimatedPeakLiveBytes = estimatedPeak;
  candidate.evaluation = context.getResourceCostAnalysis().evaluate(cost);
  return candidate.evaluation.accepted ? std::optional<IATCandidate>(candidate)
                                       : std::nullopt;
}

// This lower bound survives row lifting, even when masked loads produce zero.
bool isKnownPositiveRowValue(Value value) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return constant.isStrictlyPositive();
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return isKnownPositiveRowValue(splat.getSrc());
  if (auto maximum = value.getDefiningOp<arith::MaxSIOp>())
    return isKnownPositiveRowValue(maximum.getLhs()) ||
           isKnownPositiveRowValue(maximum.getRhs());
  return false;
}

// Some MLIR versions mark remainder and floor division as always speculatable.
// They still require a nonzero divisor, including on newly added inactive rows.
bool isSafeRowArithmetic(Operation *operation) {
  if (isa<arith::DivSIOp, arith::DivUIOp, arith::CeilDivSIOp,
          arith::CeilDivUIOp, arith::FloorDivSIOp, arith::RemSIOp,
          arith::RemUIOp>(operation) &&
      isKnownPositiveRowValue(operation->getOperand(1)))
    return true;
  if (isa<arith::RemSIOp, arith::RemUIOp, arith::FloorDivSIOp>(operation)) {
    APInt divisor;
    if (!matchPattern(operation->getOperand(1), m_ConstantInt(&divisor)) ||
        divisor.isZero() ||
        (!isa<arith::RemUIOp>(operation) && divisor.isAllOnes()))
      return false;
  }
  return isSpeculatable(operation);
}

// Match the reduction structure independently of program mapping and cost.
std::optional<int64_t> matchIntegerRowReductions(triton::FuncOp function) {
  int64_t maxRowExtent = 0;
  for (Operation &operation : function.getBody().front()) {
    auto reduce = dyn_cast<triton::ReduceOp>(operation);
    if (!reduce)
      continue;
    if (reduce.getAxis() != 0)
      return std::nullopt;
    for (auto [source, result] :
         llvm::zip(reduce.getSrcs(), reduce.getResults())) {
      auto type = cast<RankedTensorType>(source.getType());
      if (!type.hasStaticShape() || type.getRank() != 1 ||
          type.getShape()[0] <= 0 || !isa<IntegerType>(type.getElementType()) ||
          result.getType() != type.getElementType())
        return std::nullopt;
      maxRowExtent = std::max(maxRowExtent, type.getShape()[0]);
    }
    for (Operation &combine : reduce.getCombineOp().front()) {
      const bool isOverflowCheck = isa<triton::AssertOp>(combine) &&
                                   combine.hasAttr("tt.auto_overflow_assert");
      if (!isOverflowCheck &&
          !isa<triton::ReduceReturnOp, arith::ConstantOp>(combine) &&
          (combine.getNumRegions() != 0 ||
           !combine.hasTrait<OpTrait::Elementwise>() ||
           !isMemoryEffectFree(&combine) || !isSafeRowArithmetic(&combine)))
        return std::nullopt;
      for (Value operand : combine.getOperands()) {
        Operation *owner = operand.getDefiningOp();
        if (auto argument = dyn_cast<BlockArgument>(operand))
          owner = argument.getOwner()->getParentOp();
        if (!owner ||
            (owner != reduce.getOperation() && !reduce->isAncestor(owner) &&
             !isa<arith::ConstantOp>(owner)))
          return std::nullopt;
      }
    }
  }
  return maxRowExtent > 0 ? std::optional<int64_t>(maxRowExtent) : std::nullopt;
}

bool hasLiftableIntegerRowOperations(triton::FuncOp function) {
  // The launch grid and hidden extent arguments belong to one public entry.
  auto module = function->getParentOfType<ModuleOp>();
  if (!module || !function.isPublic() ||
      llvm::count_if(module.getOps<triton::FuncOp>(),
                     [](triton::FuncOp entry) { return entry.isPublic(); }) !=
          1 ||
      !hasSupportedControlFlow(function))
    return false;
  for (Operation &operation : function.getBody().front()) {
    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      if (load.getIsVolatile())
        return false;
    }
    if (isa<triton::GetProgramIdOp, triton::MakeRangeOp, triton::AddPtrOp,
            triton::LoadOp, triton::StoreOp, triton::SplatOp,
            triton::ExpandDimsOp, triton::BroadcastOp, triton::ReduceOp,
            triton::ReturnOp, arith::ConstantOp>(operation))
      continue;
    if (isa<triton::AssertOp>(operation) &&
        operation.hasAttr("tt.auto_overflow_assert"))
      continue;
    // Load/store masks do not guard arithmetic on newly added inactive rows.
    if (operation.getNumRegions() == 0 && operation.getNumResults() != 0 &&
        operation.hasTrait<OpTrait::Elementwise>() &&
        isMemoryEffectFree(&operation) && isSafeRowArithmetic(&operation))
      continue;
    return false;
  }
  return true;
}

bool hasIndependentRowMemoryAccesses(
    triton::FuncOp function, const ProgramAxisDependence &dependence,
    const EntryArgPointerAliasAnalysis &aliases, bool allowInPlace) {
  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::StoreOp> stores;
  function.walk([&](triton::LoadOp load) { loads.push_back(load); });
  function.walk([&](triton::StoreOp store) { stores.push_back(store); });
  if (stores.size() != dependence.stores.size())
    return false;
  for (auto [index, store] : llvm::enumerate(stores)) {
    for (triton::LoadOp load : loads) {
      if (aliases.classify(store.getPtr(), load.getPtr()) ==
          EntryArgPointerRelation::DistinctEntryRoots)
        continue;
      if (!allowInPlace || store.getPtr() != load.getPtr() ||
          !load->isBeforeInBlock(store))
        return false;
    }
    for (triton::StoreOp other : llvm::drop_begin(stores, index + 1)) {
      if (store.getPtr() != other.getPtr() &&
          aliases.classify(store.getPtr(), other.getPtr()) !=
              EntryArgPointerRelation::DistinctEntryRoots)
        return false;
    }
  }
  return dependence.isProgramMappingTransformCandidate();
}

std::optional<uint64_t> estimateIntegerRowStorage(triton::FuncOp function) {
  uint64_t baseBytes = 0;
  for (Operation &operation : function.getBody().front()) {
    for (Type type : operation.getResultTypes()) {
      uint64_t elements = 1;
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (!tensor.hasStaticShape())
          return std::nullopt;
        elements = tensor.getNumElements();
        type = tensor.getElementType();
      }
      unsigned bytes = 0;
      if (auto integer = dyn_cast<IntegerType>(type))
        bytes = (integer.getWidth() + 7) / 8;
      else if (isa<triton::PointerType>(type))
        bytes = 8;
      else
        return std::nullopt;
      uint64_t storage;
      if (__builtin_mul_overflow(elements, uint64_t{bytes}, &storage) ||
          __builtin_add_overflow(baseBytes, storage, &baseBytes))
        return std::nullopt;
    }
  }
  return baseBytes;
}

std::optional<IATCandidate>
analyzeIntegerRowReductionCandidate(GraphOptimizationContext &context) {
  triton::FuncOp function = context.getFunction();
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::isAscend950(module) ||
      hasConflictingLaunchContract(module) ||
      !hasLiftableIntegerRowOperations(function))
    return std::nullopt;
  auto rowExtent = matchIntegerRowReductions(function);
  auto baseBytes = estimateIntegerRowStorage(function);
  if (!rowExtent || !baseBytes)
    return std::nullopt;

  const auto &aliases = context.getEntryArgPointerAliasAnalysis();
  for (unsigned factor : {64u, 32u, 16u, 8u, 4u, 2u}) {
    for (int32_t axis : {0, 1}) {
      ProgramGridTransformContract contract;
      contract.transforms.push_back(
          ProgramGridTransform{0, axis, factor, false, false});
      if (failed(parseProgramGridTransformContract(
              serializeProgramGridTransformContract(module.getContext(),
                                                    contract))))
        continue;
      auto pid = findOnlyProgramId(function, axis);
      if (!pid)
        continue;
      const auto &dependence =
          context.getProgramAxisDependenceAnalysis().get(axis);
      if (!hasIndependentRowMemoryAccesses(function, dependence, aliases,
                                           /*allowInPlace=*/true))
        continue;
      // Batching an unrelated program axis cannot amortize the row reduction.
      if (!llvm::any_of(dependence.dependenceClosure, [](Operation *op) {
            return isa<triton::ReduceOp>(op);
          }))
        continue;
      uint64_t peak;
      if (__builtin_mul_overflow(*baseBytes, uint64_t{factor}, &peak))
        continue;
      IATCandidate candidate;
      candidate.function = function;
      candidate.anchor = pid->getOperation();
      candidate.form = TensorizeForm::IntegerRowReduction;
      candidate.axis = axis;
      candidate.dimExtent = *rowExtent;
      candidate.factor = factor;
      candidate.resources =
          context.getResourceCostAnalysis().getResourceSnapshot();
      CandidateCost cost;
      cost.plan.tensorizeFactor = factor;
      cost.plan.blockT = 1;
      cost.plan.staticAxisFusionFactor = 1;
      cost.plan.stableId = "iat.integer-row-reduction.dynamic";
      cost.storeCountBefore = dependence.stores.size();
      cost.storeCountAfter = dependence.stores.size();
      cost.hasPeakLiveBytes = true;
      cost.baselinePeakLiveBytes = *baseBytes;
      cost.estimatedPeakLiveBytes = peak;
      candidate.evaluation = context.getResourceCostAnalysis().evaluate(cost);
      if (candidate.evaluation.accepted)
        return candidate;
    }
  }
  return std::nullopt;
}

struct GuardedRowBound {
  Value count;
  int64_t minimum;
};

std::optional<GuardedRowBound> matchGuardedRowBound(arith::CmpIOp cmp) {
  if (!cmp)
    return std::nullopt;
  Value count = cmp.getLhs(), constant = cmp.getRhs();
  auto predicate = cmp.getPredicate();
  if (predicate == arith::CmpIPredicate::slt ||
      predicate == arith::CmpIPredicate::sle) {
    std::swap(count, constant);
    predicate = predicate == arith::CmpIPredicate::slt
                    ? arith::CmpIPredicate::sgt
                    : arith::CmpIPredicate::sge;
  }
  if (predicate != arith::CmpIPredicate::sgt &&
      predicate != arith::CmpIPredicate::sge)
    return std::nullopt;
  Type type = count.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (tensor.getRank() != 1 || !tensor.hasStaticShape())
      return std::nullopt;
    type = tensor.getElementType();
  }
  APInt value;
  if (!type.isInteger(32) || !matchPattern(constant, m_ConstantInt(&value)))
    return std::nullopt;
  int64_t minimum = value.getSExtValue();
  if (predicate == arith::CmpIPredicate::sgt)
    ++minimum;
  // count > INT_MAX is never true; do not wrap it to INT_MIN.
  if (minimum > std::numeric_limits<int32_t>::max())
    return std::nullopt;
  return GuardedRowBound{count, minimum};
}

bool hasContiguousGuardedScalarAddress(
    Value pointer, const ProgramAxisDependence &dependence) {
  auto dependsOnAxis = [&](Value value) {
    Operation *definition = value.getDefiningOp();
    if (auto pid = dyn_cast_or_null<triton::GetProgramIdOp>(definition))
      return pid.getAxisAsInt() == dependence.axis;
    return definition &&
           llvm::is_contained(dependence.dependenceClosure, definition);
  };
  bool foundPid = false;
  while (auto addPtr = pointer.getDefiningOp<triton::AddPtrOp>()) {
    Value offset = addPtr.getOffset();
    auto pid = offset.getDefiningOp<triton::GetProgramIdOp>();
    if (pid && pid.getAxisAsInt() == dependence.axis) {
      if (foundPid)
        return false;
      foundPid = true;
    } else if (dependsOnAxis(offset)) {
      return false;
    }
    pointer = addPtr.getPtr();
  }
  return foundPid && !dependsOnAxis(pointer);
}

// Find guarded scalar loads with proven forward-contiguous lane addresses.
std::optional<IATCandidate>
analyzeGeneralizedGuardedBatchCandidate(GraphOptimizationContext &context) {
  auto function = context.getFunction();
  auto module = function->getParentOfType<ModuleOp>();
  if (!module || function.getFunctionType().getNumResults() != 0 ||
      !hacc::utils::isAscend950(module) ||
      hasConflictingLaunchContract(module) ||
      !hasLiftableIntegerRowOperations(function))
    return std::nullopt;

  int64_t rowWidth = 1;
  SmallVector<triton::LoadOp> loads;
  for (Operation &op : function.getBody().front()) {
    if (op.getNumRegions())
      return std::nullopt;
    for (Type type : op.getResultTypes()) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (!tensor.hasStaticShape() || tensor.getRank() != 1)
          return std::nullopt;
        rowWidth = std::max(rowWidth, tensor.getNumElements());
      }
    }
    if (auto load = dyn_cast<triton::LoadOp>(op))
      loads.push_back(load);
  }
  const auto &aliases = context.getEntryArgPointerAliasAnalysis();
  const auto &live = context.getResourceCostAnalysis().getLiveByteEstimate();
  if (!live.known || !live.peakLiveBytes)
    return std::nullopt;

  for (int32_t axis : {0, 1}) {
    auto pid = findOnlyProgramId(function, axis);
    if (!pid)
      continue;
    const auto &dep = context.getProgramAxisDependenceAnalysis().get(axis);
    if (!hasIndependentRowMemoryAccesses(function, dep, aliases,
                                         /*allowInPlace=*/false))
      continue;
    bool hasGuardedLoad = false;
    bool hasNonContiguousScalarLoad = false;
    for (triton::LoadOp load : loads) {
      if (isa<RankedTensorType>(load.getType()) ||
          !llvm::is_contained(dep.dependenceClosure, load.getOperation()))
        continue;
      if (!hasContiguousGuardedScalarAddress(load.getPtr(), dep)) {
        hasNonContiguousScalarLoad = true;
        break;
      }
      if (!load.getMask())
        continue;
      auto bound =
          matchGuardedRowBound(load.getMask().getDefiningOp<arith::CmpIOp>());
      hasGuardedLoad |= bound && bound->count.getDefiningOp() &&
                        (bound->count.getDefiningOp() == pid->getOperation() ||
                         llvm::is_contained(dep.dependenceClosure,
                                            bound->count.getDefiningOp()));
    }
    if (!hasGuardedLoad || hasNonContiguousScalarLoad)
      continue;

    for (unsigned factor : {32u, 16u, 8u, 4u, 2u}) {
      if (rowWidth > 4096 / factor)
        continue;
      ProgramGridTransformContract contract;
      contract.transforms.push_back(
          ProgramGridTransform{0, axis, factor, false, false});
      if (failed(parseProgramGridTransformContract(
              serializeProgramGridTransformContract(module.getContext(),
                                                    contract))))
        continue;
      uint64_t peak;
      if (__builtin_mul_overflow(live.peakLiveBytes, uint64_t{2} * factor,
                                 &peak))
        continue;
      IATCandidate candidate;
      candidate.function = function;
      candidate.anchor = pid->getOperation();
      candidate.form = TensorizeForm::GuardedBatch;
      candidate.axis = axis;
      candidate.factor = factor;
      candidate.dimExtent = rowWidth;
      candidate.guardedBatch = true;
      candidate.resources =
          context.getResourceCostAnalysis().getResourceSnapshot();
      CandidateCost cost;
      cost.plan.tensorizeFactor = factor;
      cost.plan.stableId =
          "iat.guarded-batch.generalized.f" + std::to_string(factor);
      cost.hasPeakLiveBytes = true;
      cost.baselinePeakLiveBytes = live.peakLiveBytes;
      cost.estimatedPeakLiveBytes = peak;
      candidate.evaluation = context.getResourceCostAnalysis().evaluate(cost);
      if (candidate.evaluation.accepted)
        return candidate;
    }
  }
  return std::nullopt;
}

bool sameDynamicRowCandidate(const IATCandidate &lhs, const IATCandidate &rhs) {
  return lhs.function == rhs.function && lhs.anchor == rhs.anchor &&
         lhs.form == rhs.form && lhs.axis == rhs.axis &&
         lhs.splitExtent == rhs.splitExtent && lhs.dimExtent == rhs.dimExtent &&
         lhs.factor == rhs.factor && lhs.guardedBatch == rhs.guardedBatch;
}

bool sameDynamicSingleMomentCandidate(const IATCandidate &lhs,
                                      const IATCandidate &rhs) {
  return lhs.function == rhs.function && lhs.anchor == rhs.anchor &&
         lhs.form == rhs.form && lhs.axis == rhs.axis &&
         lhs.dimExtent == rhs.dimExtent && lhs.factor == rhs.factor &&
         lhs.requiresPersistentChaining == rhs.requiresPersistentChaining;
}

LogicalResult
addGeneralizedGuardedFullInputRowsFastPath(triton::FuncOp function) {
  if (!llvm::hasSingleElement(function.getBody()) ||
      function.getFunctionType().getNumResults() != 0)
    return failure();
  Block &block = function.getBody().front();
  arith::CmpIOp boundary;
  for (Operation &op : block) {
    if (op.hasAttr("IATRuntimeExtentUnsignedMask")) {
      if (boundary || !isa<arith::CmpIOp>(op))
        return failure();
      boundary = cast<arith::CmpIOp>(op);
    }
  }
  if (!boundary)
    return failure();
  auto boolType = dyn_cast<RankedTensorType>(boundary.getType());
  if (!boolType || boolType.getRank() != 1)
    return failure();

  Value count;
  for (Operation &op : block) {
    auto load = dyn_cast<triton::LoadOp>(op);
    if (!load || !load.getMask() || load.getMask().getType() != boolType)
      continue;
    SmallVector<Value> masks{load.getMask()};
    while (!masks.empty()) {
      Value mask = masks.pop_back_val();
      if (auto conjunction = mask.getDefiningOp<arith::AndIOp>()) {
        masks.push_back(conjunction.getLhs());
        masks.push_back(conjunction.getRhs());
        continue;
      }
      auto bound = matchGuardedRowBound(mask.getDefiningOp<arith::CmpIOp>());
      if (!bound)
        continue;
      Operation *definition = bound->count.getDefiningOp();
      if (definition && definition->getBlock() == &block) {
        count = bound->count;
        break;
      }
    }
    if (count)
      break;
  }
  if (!count)
    return failure();
  Operation *anchor = count.getDefiningOp();
  if (anchor->isBeforeInBlock(boundary))
    anchor = boundary;
  auto type = cast<RankedTensorType>(count.getType());
  if (type.getShape() != boolType.getShape())
    return failure();
  auto terminator = dyn_cast<triton::ReturnOp>(block.getTerminator());
  if (!terminator || terminator->getNumOperands())
    return failure();

  SmallVector<Operation *> suffix;
  DenseSet<Operation *> provenComparisons;
  int64_t bound = std::numeric_limits<int32_t>::min();
  for (Operation *op = anchor->getNextNode(); op != terminator.getOperation();
       op = op->getNextNode()) {
    if (!op || op->getNumRegions())
      return failure();
    suffix.push_back(op);
    auto lowerBound = matchGuardedRowBound(dyn_cast<arith::CmpIOp>(op));
    if (!lowerBound || lowerBound->count != count)
      continue;
    bound = std::max(bound, lowerBound->minimum);
    provenComparisons.insert(op);
  }
  if (provenComparisons.empty())
    return failure();

  SmallVector<std::tuple<arith::CmpIOp, Value, int64_t>> columnGuards;
  DenseSet<Operation *> provenColumnComparisons;
  for (Operation *op : suffix) {
    auto cmp = dyn_cast<arith::CmpIOp>(op);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
      continue;
    auto range = cmp.getLhs().getDefiningOp<triton::MakeRangeOp>();
    auto splat = cmp.getRhs().getDefiningOp<triton::SplatOp>();
    if (!range || !splat || !splat.getSrc().getType().isInteger(32))
      continue;
    Value scalar = splat.getSrc();
    Operation *definition = scalar.getDefiningOp();
    if (definition && (definition->getBlock() != &block ||
                       !definition->isBeforeInBlock(anchor)))
      continue;
    columnGuards.emplace_back(cmp, scalar, range.getEnd());
    provenColumnComparisons.insert(op);
  }

  OpBuilder builder(anchor);
  builder.setInsertionPointAfter(anchor);
  Location loc = anchor->getLoc();
  Value limit = builder.create<arith::ConstantOp>(
      loc, type,
      DenseIntElementsAttr::get(type, APInt(32, bound, /*isSigned=*/true)));
  Value valid = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              count, limit);
  Value safe = builder.create<arith::AndIOp>(loc, valid, boundary.getResult());
  Value integerMask = builder.create<arith::ExtUIOp>(loc, type, safe);
  auto minimum =
      builder.create<triton::ReduceOp>(loc, ValueRange{integerMask}, 0);
  Block *combine = new Block;
  minimum.getCombineOp().push_back(combine);
  combine->addArguments({builder.getI32Type(), builder.getI32Type()},
                        {loc, loc});
  OpBuilder reduceBuilder = OpBuilder::atBlockBegin(combine);
  Value result = reduceBuilder.create<arith::MinSIOp>(
      loc, combine->getArgument(0), combine->getArgument(1));
  reduceBuilder.create<triton::ReduceReturnOp>(loc, ValueRange{result});
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  minimum->getResult(0), one);
  for (auto &[cmp, scalar, end] : columnGuards) {
    Value upper = builder.create<arith::ConstantIntOp>(loc, end, 32);
    Value fullColumns = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, scalar, upper);
    condition = builder.create<arith::AndIOp>(loc, condition, fullColumns);
  }
  auto versioned = builder.create<scf::IfOp>(loc, condition, true);
  OpBuilder fastBuilder = versioned.getThenBodyBuilder();
  Value allTrue = fastBuilder.create<arith::ConstantOp>(
      loc, boolType, DenseIntElementsAttr::get(boolType, true));
  IRMapping fastMapping;
  fastMapping.map(boundary.getResult(), allTrue);
  for (auto &[cmp, scalar, end] : columnGuards) {
    auto predicateType = cast<RankedTensorType>(cmp.getType());
    Value trueColumns = fastBuilder.create<arith::ConstantOp>(
        loc, predicateType, DenseIntElementsAttr::get(predicateType, true));
    fastMapping.map(cmp.getResult(), trueColumns);
  }
  for (Operation *op : suffix) {
    if (provenColumnComparisons.contains(op))
      continue;
    if (provenComparisons.contains(op)) {
      fastMapping.map(op->getResult(0), allTrue);
      continue;
    }
    fastBuilder.clone(*op, fastMapping);
  }
  OpBuilder slowBuilder = versioned.getElseBodyBuilder();
  IRMapping slowMapping;
  for (Operation *op : suffix)
    slowBuilder.clone(*op, slowMapping);
  for (Operation *op : llvm::reverse(suffix))
    op->erase();
  return success();
}

class DynamicRowIATPlan final : public RewritePlan {
public:
  DynamicRowIATPlan(IATCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    auto current = candidate.form == TensorizeForm::IntegerRowReduction
                       ? analyzeIntegerRowReductionCandidate(context)
                       : analyzeDynamicMergeSplitCandidate(context);
    return current && sameDynamicRowCandidate(candidate, *current) ? success()
                                                                   : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module || hasConflictingLaunchContract(module))
      return failure();
    ModuleOp sandbox = createProgramMappingSandbox(module, candidate.function);
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    if (!clonedFunction ||
        failed(materializeIATCandidateToSandbox(sandbox, clonedFunction,
                                                candidate)) ||
        failed(runProgramMappingStructuralCleanup(sandbox)) ||
        failed(mlir::verify(sandbox.getOperation())))
      return failure();
    Attribute transforms = sandbox->getAttr(kProgramGridTransformsAttr);
    if (!transforms || failed(parseProgramGridTransformContract(transforms)) ||
        !hasProgramGridHiddenExtentArguments(clonedFunction))
      return failure();

    if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                    clonedFunction)))
      return failure();
    module->setAttr(kProgramGridTransformsAttr, transforms);
    module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                    UnitAttr::get(module.getContext()));
    return success();
  }

private:
  IATCandidate candidate;
  unsigned epoch;
};

LogicalResult materializeGuardedBatchCandidate(ModuleOp sandbox,
                                               triton::FuncOp function,
                                               const IATCandidate &candidate) {
  if (!function ||
      failed(materializeIATCandidateToSandbox(sandbox, function, candidate)) ||
      failed(runProgramMappingStructuralCleanup(sandbox)) ||
      failed(mlir::verify(sandbox.getOperation())))
    return failure();
  LogicalResult guarded = addGeneralizedGuardedFullInputRowsFastPath(function);
  if (failed(guarded) || failed(runProgramMappingStructuralCleanup(sandbox)) ||
      failed(mlir::verify(sandbox.getOperation())))
    return failure();
  Attribute transforms = sandbox->getAttr(kProgramGridTransformsAttr);
  return transforms &&
                 succeeded(parseProgramGridTransformContract(transforms)) &&
                 hasProgramGridHiddenExtentArguments(function)
             ? success()
             : failure();
}

bool canMaterializeGuardedBatchCandidate(const IATCandidate &candidate) {
  auto module = candidate.function->getParentOfType<ModuleOp>();
  OwningOpRef<ModuleOp> sandbox(
      createProgramMappingSandbox(module, candidate.function));
  auto function = dyn_cast<triton::FuncOp>(&sandbox->getBody()->front());
  return succeeded(
      materializeGuardedBatchCandidate(*sandbox, function, candidate));
}

class GuardedBatchIATPlan final : public RewritePlan {
public:
  GuardedBatchIATPlan(IATCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    auto current = analyzeGeneralizedGuardedBatchCandidate(context);
    return current && sameDynamicRowCandidate(candidate, *current) ? success()
                                                                   : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module || hasConflictingLaunchContract(module))
      return failure();
    OwningOpRef<ModuleOp> sandbox(
        createProgramMappingSandbox(module, candidate.function));
    auto clonedFunction =
        dyn_cast<triton::FuncOp>(&sandbox->getBody()->front());
    if (failed(materializeGuardedBatchCandidate(*sandbox, clonedFunction,
                                                candidate)))
      return failure();
    Attribute transforms = (*sandbox)->getAttr(kProgramGridTransformsAttr);
    if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                    clonedFunction)))
      return failure();
    module->setAttr(kProgramGridTransformsAttr, transforms);
    module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                    UnitAttr::get(module.getContext()));
    return success();
  }

private:
  IATCandidate candidate;
  unsigned epoch;
};

class DynamicIndependentAxisTensorizeRule final : public GraphOptimizationRule {
public:
  DynamicIndependentAxisTensorizeRule(bool enabledForCompileMode,
                                      bool iatAndPtsmEnabled)
      : enabledForCompileMode(enabledForCompileMode),
        iatAndPtsmEnabled(iatAndPtsmEnabled) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::EntryArgPointerAlias |
           AnalysisRequirement::ProgramAxisDependence |
           AnalysisRequirement::ResourceCost;
  }
  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    if (!enabledForCompileMode)
      return success();
    std::optional<IATCandidate> candidate =
        analyzeDynamicMergeSplitCandidate(context);
    if (candidate) {
      emitCandidateRemark(candidate->anchor, candidate->evaluation);
      plans.push_back(std::make_unique<DynamicRowIATPlan>(std::move(*candidate),
                                                          context.getEpoch()));
      return success();
    }
    candidate = analyzeGeneralizedGuardedBatchCandidate(context);
    if (candidate && canMaterializeGuardedBatchCandidate(*candidate)) {
      emitCandidateRemark(candidate->anchor, candidate->evaluation);
      plans.push_back(std::make_unique<GuardedBatchIATPlan>(
          std::move(*candidate), context.getEpoch()));
      return success();
    }
    candidate = analyzeIntegerRowReductionCandidate(context);
    if (candidate) {
      emitCandidateRemark(candidate->anchor, candidate->evaluation);
      plans.push_back(std::make_unique<DynamicRowIATPlan>(std::move(*candidate),
                                                          context.getEpoch()));
      return success();
    }
    if (!iatAndPtsmEnabled)
      return success();
    candidate = analyzeDynamicSingleMomentCandidate(context);
    if (candidate) {
      emitCandidateRemark(candidate->anchor, candidate->evaluation);
      plans.push_back(std::make_unique<DynamicSingleMomentProgramMappingPlan>(
          std::move(*candidate), context.getEpoch()));
    }
    return success();
  }

private:
  bool enabledForCompileMode;
  bool iatAndPtsmEnabled;
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createIndependentAxisTensorizeRule(
    const IndependentAxisTensorizeRuleOptions &options) {
  return std::make_unique<DynamicIndependentAxisTensorizeRule>(
      options.enabledForCompileMode, options.iatAndPtsmEnabled);
}
