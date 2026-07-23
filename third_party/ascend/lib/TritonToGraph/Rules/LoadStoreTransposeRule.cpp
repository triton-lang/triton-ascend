/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
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

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/PermutationAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

constexpr std::array<int32_t, 2> kSwapPermutation = {1, 0};

using UnaryChain = SmallVector<Operation *, 8>;

struct LoadStoreCandidate {
  triton::LoadOp load;
  Operation *loadOperation;
  UnaryChain unaryChain;
  triton::StoreOp store;
  Operation *storeOperation;
  StaticAccess loadAccess;
  StaticAccess storeAccess;
};

struct RebuiltPointer {
  Value pointer;
  SmallVector<Operation *, 16> created;
};

bool hasOnlyExpectedUse(Value value, Operation *expectedUser) {
  return value && expectedUser && value.hasOneUse() &&
         value.use_begin()->getOwner() == expectedUser;
}

bool isUnencodedStaticTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && !type.getEncoding();
}

bool isWhitelistedUnary(Operation *operation, Value input) {
  if (!operation || !input ||
      !isa<arith::NegFOp, math::AbsFOp, math::AbsIOp, math::CeilOp,
           math::FloorOp, math::CosOp, math::SinOp, math::ErfOp, math::ExpOp,
           math::Exp2Op, math::LogOp, math::Log2Op, math::SqrtOp,
           math::RsqrtOp, math::TanhOp, triton::PreciseSqrtOp>(operation) ||
      operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 0 || operation->getOperand(0) != input ||
      !isMemoryEffectFree(operation))
    return false;

  Value result = operation->getResult(0);
  return isUnencodedStaticTensor(input) && isUnencodedStaticTensor(result) &&
         input.getType() == result.getType();
}

bool containsExplicitTranspose(Operation *first, Operation *last) {
  if (!first || !last || first->getBlock() != last->getBlock())
    return true;

  for (Operation *operation = first->getNextNode(); operation;
       operation = operation->getNextNode()) {
    if (operation == last)
      return false;
    if (isa<triton::TransOp>(operation))
      return true;
  }
  return true;
}

bool isAllowedAccessProducer(Operation *operation) {
  return isa<triton::AddPtrOp, triton::SplatOp, triton::BroadcastOp,
             triton::ExpandDimsOp, triton::MakeRangeOp, arith::AddIOp,
             arith::MulIOp, arith::ConstantOp>(operation);
}

// We clone the old pointer DAG instead of erasing or mutating it.  That is
// safe for shared producers, but V1 still requires each analyzed structural
// producer to be local to the access block and effect-free.
bool hasLocalPureAccessDag(const StaticAccess &access, Block *block) {
  if (!block || !access.pointer || !access.base)
    return false;

  SmallVector<Value, 16> worklist = {access.pointer};
  llvm::DenseSet<Operation *> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (value == access.base)
      continue;

    Operation *operation = value.getDefiningOp();
    if (!operation)
      return false;
    if (!visited.insert(operation).second)
      continue;
    if (operation->getBlock() != block || operation->getNumRegions() != 0 ||
        !isAllowedAccessProducer(operation) || !isMemoryEffectFree(operation))
      return false;

    for (Value operand : operation->getOperands())
      worklist.push_back(operand);
  }
  return true;
}

bool hasCanonicalPreReindexAccess(const StaticAccess &access) {
  if (!access.lanesInjective || access.shape.size() != 2 ||
      access.strides.size() != 2 || access.axisProvenance.size() != 2 ||
      access.axes.size() != 2)
    return false;

  const int64_t extent = access.shape[0];
  if (extent <= 1 || access.shape[1] != extent ||
      access.strides[0] != extent || access.strides[1] != 1 ||
      access.axisProvenance[0] != 0 || access.axisProvenance[1] != 1 ||
      access.firstOffset != 0)
    return false;

  for (unsigned axis = 0; axis < 2; ++axis) {
    const StaticAccessAxis &term = access.axes[axis];
    if (!term.range || term.outputAxis != axis || term.rangeStart != 0 ||
        term.rangeEnd != extent || term.stride != access.strides[axis])
      return false;
  }
  return true;
}

bool haveIndependentPointerDags(const StaticAccess &loadAccess,
                                const StaticAccess &storeAccess) {
  if (!loadAccess.pointer || !storeAccess.pointer ||
      loadAccess.pointer == storeAccess.pointer ||
      loadAccess.offset == storeAccess.offset ||
      loadAccess.axes.size() != 2 || storeAccess.axes.size() != 2)
    return false;

  for (const StaticAccessAxis &loadAxis : loadAccess.axes) {
    for (const StaticAccessAxis &storeAxis : storeAccess.axes) {
      triton::MakeRangeOp loadRange = loadAxis.range;
      triton::MakeRangeOp storeRange = storeAxis.range;
      if (!loadRange || !storeRange ||
          loadRange.getOperation() == storeRange.getOperation())
        return false;
    }
  }
  return true;
}

std::optional<LoadStoreCandidate> matchCandidate(triton::LoadOp load) {
  if (!load || load->getNumResults() != 1 || load->getNumRegions() != 0)
    return std::nullopt;

  Block *block = load->getBlock();
  Value loadedValue = load->getResult(0);
  if (!block || !loadedValue.hasOneUse())
    return std::nullopt;

  UnaryChain unaryChain;
  Value currentValue = loadedValue;
  Operation *storeOperation = nullptr;
  triton::StoreOp store;
  while (true) {
    if (!currentValue.hasOneUse())
      return std::nullopt;

    Operation *user = currentValue.use_begin()->getOwner();
    if (!hasOnlyExpectedUse(currentValue, user) || user->getBlock() != block)
      return std::nullopt;

    if (auto candidateStore = dyn_cast<triton::StoreOp>(user)) {
      if (unaryChain.empty() || candidateStore->getNumRegions() != 0 ||
          candidateStore.getValue() != currentValue)
        return std::nullopt;
      store = candidateStore;
      storeOperation = user;
      break;
    }

    if (!isWhitelistedUnary(user, currentValue))
      return std::nullopt;
    unaryChain.push_back(user);
    currentValue = user->getResult(0);
  }

  StaticAccessAnalysis accessAnalysis;
  StaticAccessProof loadProof = accessAnalysis.analyzeLoad(load);
  StaticAccessProof storeProof = accessAnalysis.analyzeStore(store);
  if (!loadProof.isProven() || !storeProof.isProven() ||
      !hasCanonicalPreReindexAccess(*loadProof.access) ||
      !hasCanonicalPreReindexAccess(*storeProof.access) ||
      !haveIndependentPointerDags(*loadProof.access, *storeProof.access) ||
      !hasLocalPureAccessDag(*loadProof.access, block) ||
      !hasLocalPureAccessDag(*storeProof.access, block))
    return std::nullopt;

  if (loadProof.access->shape != storeProof.access->shape ||
      load.getPtr().getType() != store.getPtr().getType() ||
      !isUnencodedStaticTensor(load.getPtr()) ||
      !isUnencodedStaticTensor(store.getPtr()))
    return std::nullopt;

  ProtectedIntervalAnalysis intervalAnalysis;
  std::array<StaticAccess, 2> protectedAccesses = {
      *loadProof.access,
      *storeProof.access,
  };
  if (!intervalAnalysis
           .proveNoConflictingLoadStoreEffects(load.getOperation(),
                                               store.getOperation(),
                                               protectedAccesses)
           .isProven() ||
      containsExplicitTranspose(load.getOperation(), store.getOperation()))
    return std::nullopt;

  return LoadStoreCandidate{load,
                            load.getOperation(),
                            std::move(unaryChain),
                            store,
                            storeOperation,
                            std::move(*loadProof.access),
                            std::move(*storeProof.access)};
}

bool fitsI32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}

void eraseCreatedOperations(IRRewriter &rewriter,
                            ArrayRef<Operation *> created) {
  for (Operation *operation : llvm::reverse(created))
    rewriter.eraseOp(operation);
}

std::optional<RebuiltPointer>
buildReindexedPointer(IRRewriter &rewriter, const StaticAccess &access,
                      const Permutation &permutation) {
  if (access.shape.size() != 2 || access.axes.size() != 2 ||
      permutation.rank() != 2)
    return std::nullopt;

  FailureOr<SmallVector<int64_t>> permutedShape =
      permutation.permuteShape(access.shape);
  if (failed(permutedShape) || *permutedShape != access.shape)
    return std::nullopt;

  auto pointerType = dyn_cast<RankedTensorType>(access.pointer.getType());
  auto offsetType = dyn_cast<RankedTensorType>(access.offset.getType());
  if (!pointerType || !offsetType || pointerType.getEncoding() ||
      offsetType.getEncoding())
    return std::nullopt;

  RankedTensorType newPointerType = pointerType.clone(*permutedShape);
  RankedTensorType newOffsetType = offsetType.clone(*permutedShape);
  // Square P=[1, 0] must preserve every externally visible tensor type.
  if (newPointerType != pointerType || newOffsetType != offsetType)
    return std::nullopt;

  RebuiltPointer rebuilt;
  SmallVector<Value, 2> broadcastTerms;
  for (unsigned newAxis = 0; newAxis < 2; ++newAxis) {
    const int32_t oldAxis = permutation.getNewToOld()[newAxis];
    if (oldAxis < 0 || oldAxis >= 2)
      return std::nullopt;
    const StaticAccessAxis &oldTerm = access.axes[oldAxis];
    if (!oldTerm.range || !fitsI32(oldTerm.rangeStart) ||
        !fitsI32(oldTerm.rangeEnd) || !fitsI32(oldTerm.stride))
      return std::nullopt;

    triton::MakeRangeOp oldRange = oldTerm.range;
    auto oldRangeType =
        dyn_cast<RankedTensorType>(oldRange.getResult().getType());
    if (!oldRangeType || oldRangeType.getEncoding())
      return std::nullopt;

    SmallVector<int64_t, 1> rangeShape = {(*permutedShape)[newAxis]};
    RankedTensorType newRangeType = oldRangeType.clone(rangeShape);
    SmallVector<int64_t, 2> expandedShape;
    expandedShape.append(permutedShape->begin(), permutedShape->end());
    const unsigned insertedAxis = 1 - newAxis;
    expandedShape[insertedAxis] = 1;
    RankedTensorType expandedType = newOffsetType.clone(expandedShape);

    Location location = access.pointer.getLoc();
    auto range = rewriter.create<triton::MakeRangeOp>(
        location, newRangeType,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(oldTerm.rangeStart)),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(oldTerm.rangeEnd)));
    rebuilt.created.push_back(range.getOperation());
    auto expanded = rewriter.create<triton::ExpandDimsOp>(
        location, expandedType, range.getResult(), insertedAxis);
    rebuilt.created.push_back(expanded.getOperation());
    auto stride = rewriter.create<arith::ConstantOp>(
        location,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(oldTerm.stride)));
    rebuilt.created.push_back(stride.getOperation());
    auto strideSplat = rewriter.create<triton::SplatOp>(
        location, expandedType, stride.getResult());
    rebuilt.created.push_back(strideSplat.getOperation());
    auto multiplied = rewriter.create<arith::MulIOp>(
        location, expanded.getResult(), strideSplat.getResult());
    rebuilt.created.push_back(multiplied.getOperation());
    auto broadcast = rewriter.create<triton::BroadcastOp>(
        location, newOffsetType, multiplied.getResult());
    rebuilt.created.push_back(broadcast.getOperation());
    broadcastTerms.push_back(broadcast.getResult());
  }

  auto offsets = rewriter.create<arith::AddIOp>(
      access.pointer.getLoc(), broadcastTerms[0], broadcastTerms[1]);
  rebuilt.created.push_back(offsets.getOperation());
  auto baseSplat = rewriter.create<triton::SplatOp>(
      access.pointer.getLoc(), newPointerType, access.base);
  rebuilt.created.push_back(baseSplat.getOperation());
  auto pointer = rewriter.create<triton::AddPtrOp>(
      access.pointer.getLoc(), newPointerType, baseSplat.getResult(),
      offsets.getResult());
  rebuilt.created.push_back(pointer.getOperation());
  rebuilt.pointer = pointer.getResult();
  return rebuilt;
}

bool provesReindexedAccess(const StaticAccess &original,
                           const StaticAccess &rebuilt,
                           const Permutation &permutation) {
  if (!original.lanesInjective || !rebuilt.lanesInjective ||
      original.base != rebuilt.base || original.shape.size() != 2 ||
      rebuilt.shape.size() != 2 || original.axes.size() != 2 ||
      rebuilt.axes.size() != 2)
    return false;

  FailureOr<SmallVector<int64_t>> expectedShape =
      permutation.permuteShape(original.shape);
  if (failed(expectedShape) || *expectedShape != rebuilt.shape ||
      original.firstOffset != rebuilt.firstOffset ||
      original.lastOffset != rebuilt.lastOffset)
    return false;

  for (unsigned newAxis = 0; newAxis < 2; ++newAxis) {
    const int32_t oldAxis = permutation.getNewToOld()[newAxis];
    if (oldAxis < 0 || oldAxis >= 2)
      return false;
    const StaticAccessAxis &oldTerm = original.axes[oldAxis];
    const StaticAccessAxis &newTerm = rebuilt.axes[newAxis];
    if (!newTerm.range || newTerm.outputAxis != newAxis ||
        newTerm.rangeStart != oldTerm.rangeStart ||
        newTerm.rangeEnd != oldTerm.rangeEnd ||
        newTerm.stride != oldTerm.stride ||
        rebuilt.strides[newAxis] != original.strides[oldAxis])
      return false;
  }

  // The two terms are the same static ranges and strides in a bijective axis
  // order, so this proves the same active-address set, not merely matching
  // extrema.  The analysis has independently checked injectivity for both.
  return true;
}

class LoadStoreTransposePlan final : public RewritePlan {
public:
  LoadStoreTransposePlan(LoadStoreCandidate candidate, unsigned epoch)
      : load(candidate.load), loadOperation(candidate.loadOperation),
        unaryChain(std::move(candidate.unaryChain)), store(candidate.store),
        storeOperation(candidate.storeOperation), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::LoadStoreTranspose;
  }

  unsigned getBenefit() const override { return 1; }

  Operation *getAnchor() const override { return loadOperation; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getEpoch() != epoch || !load || !store)
      return failure();

    triton::FuncOp function = load->getParentOfType<triton::FuncOp>();
    if (!function ||
        function.getOperation() != context.getFunction().getOperation())
      return failure();

    std::optional<LoadStoreCandidate> current = matchCandidate(load);
    if (!current || current->loadOperation != loadOperation ||
        !llvm::equal(current->unaryChain, unaryChain) ||
        current->storeOperation != storeOperation)
      return failure();
    return success();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    std::optional<LoadStoreCandidate> current = matchCandidate(load);
    if (!current || current->loadOperation != loadOperation ||
        !llvm::equal(current->unaryChain, unaryChain) ||
        current->storeOperation != storeOperation)
      return failure();

    FailureOr<Permutation> permutation =
        Permutation::create(kSwapPermutation);
    if (failed(permutation))
      return failure();

    // The two pointer DAGs can depend on independent scalar bases.  In
    // particular, a legal store base may be defined after the load, so each
    // replacement must be inserted at the use it will dominate.
    rewriter.setInsertionPoint(loadOperation);
    std::optional<RebuiltPointer> rebuiltLoad =
        buildReindexedPointer(rewriter, current->loadAccess, *permutation);
    if (!rebuiltLoad)
      return failure();

    rewriter.setInsertionPoint(storeOperation);
    std::optional<RebuiltPointer> rebuiltStore =
        buildReindexedPointer(rewriter, current->storeAccess, *permutation);
    if (!rebuiltStore) {
      eraseCreatedOperations(rewriter, rebuiltLoad->created);
      return failure();
    }

    SmallVector<Operation *, 32> created;
    created.append(rebuiltLoad->created.begin(), rebuiltLoad->created.end());
    created.append(rebuiltStore->created.begin(), rebuiltStore->created.end());
    for (Operation *operation : created) {
      if (failed(mlir::verify(operation))) {
        eraseCreatedOperations(rewriter, created);
        return failure();
      }
    }

    StaticAccessAnalysis accessAnalysis;
    StaticAccessProof rebuiltLoadProof =
        accessAnalysis.analyzePointer(rebuiltLoad->pointer);
    StaticAccessProof rebuiltStoreProof =
        accessAnalysis.analyzePointer(rebuiltStore->pointer);
    if (!rebuiltLoadProof.isProven() || !rebuiltStoreProof.isProven() ||
        !provesReindexedAccess(current->loadAccess, *rebuiltLoadProof.access,
                               *permutation) ||
        !provesReindexedAccess(current->storeAccess, *rebuiltStoreProof.access,
                               *permutation)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }

    // No original op is changed until every replacement has verified and both
    // rebuilt roots have independently re-established the static proof.
    rewriter.modifyOpInPlace(loadOperation, [&] {
      loadOperation->setOperand(0, rebuiltLoad->pointer);
    });
    rewriter.modifyOpInPlace(storeOperation, [&] {
      storeOperation->setOperand(0, rebuiltStore->pointer);
    });
    return success();
  }

private:
  triton::LoadOp load;
  Operation *loadOperation;
  UnaryChain unaryChain;
  triton::StoreOp store;
  Operation *storeOperation;
  unsigned epoch;
};

class LoadStoreTransposeRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::LoadStoreTranspose;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](triton::LoadOp load) {
      std::optional<LoadStoreCandidate> candidate = matchCandidate(load);
      if (candidate) {
        plans.push_back(std::make_unique<LoadStoreTransposePlan>(
            std::move(*candidate), context.getEpoch()));
      }
    });
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule>
cfg::createLoadStoreTransposeRule() {
  return std::make_unique<LoadStoreTransposeRule>();
}
