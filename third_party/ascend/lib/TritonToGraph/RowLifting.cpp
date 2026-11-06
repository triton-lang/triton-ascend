/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "TritonToGraph/RowLifting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
namespace cfg {

//===----------------------------------------------------------------------===//
// Common analysis, shape and memory-mask helpers
//===----------------------------------------------------------------------===//

bool dependsOnValue(Value v, Value pid) {
  if (v == pid)
    return true;

  Operation *op = v.getDefiningOp();
  return op && llvm::any_of(op->getOperands(),
                            [&](Value a) { return dependsOnValue(a, pid); });
}

static bool dependsOnValueThroughRegions(Value root, Value target,
                                         llvm::DenseSet<Value> &visited) {
  if (root == target)
    return true;
  if (!visited.insert(root).second)
    return false;

  Operation *op = root.getDefiningOp();
  if (!op)
    return false;

  bool found = false;
  op->walk([&](Operation *nested) {
    if (found)
      return;
    for (Value operand : nested->getOperands())
      if (dependsOnValueThroughRegions(operand, target, visited)) {
        found = true;
        return;
      }
  });
  return found;
}

static bool dependsOnValueThroughRegions(Value root, Value target) {
  llvm::DenseSet<Value> visited;
  return dependsOnValueThroughRegions(root, target, visited);
}

bool isPidInvariant(Value value, Value pid) {
  return !dependsOnValueThroughRegions(value, pid);
}

std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value)) && value.isSignedIntN(64))
    return value.getSExtValue();

  return std::nullopt;
}

struct IntegerBounds {
  int64_t lo;
  int64_t hi;
};

static std::optional<IntegerBounds> inferMonotonicBounds(Value value, Value pid,
                                                         int64_t grid) {
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (!integer || integer.getWidth() > 64)
    return std::nullopt;
  unsigned width = integer.getWidth();
  IntegerBounds domain{APInt::getSignedMinValue(width).getSExtValue(),
                       APInt::getSignedMaxValue(width).getSExtValue()};

  auto finish = [&](__int128 lo, __int128 hi) -> IntegerBounds {
    return {
        static_cast<int64_t>(std::clamp<__int128>(lo, domain.lo, domain.hi)),
        static_cast<int64_t>(std::clamp<__int128>(hi, domain.lo, domain.hi))};
  };
  if (value == pid)
    return finish(0, (grid > 0 ? grid : INT32_MAX) - 1);
  if (auto constant = getConstantIntValue(value))
    return IntegerBounds{*constant, *constant};
  if (isPidInvariant(value, pid))
    return domain;
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return inferMonotonicBounds(ext.getIn(), pid, grid);

  Operation *op = value.getDefiningOp();
  if (!op || op->getNumOperands() != 2)
    return std::nullopt;
  auto lhs = inferMonotonicBounds(op->getOperand(0), pid, grid);
  auto rhs = inferMonotonicBounds(op->getOperand(1), pid, grid);
  if (!lhs || !rhs)
    return std::nullopt;
  if (isa<arith::AddIOp>(op))
    return finish(static_cast<__int128>(lhs->lo) + rhs->lo,
                  static_cast<__int128>(lhs->hi) + rhs->hi);
  if (isa<arith::SubIOp>(op) && isPidInvariant(op->getOperand(1), pid))
    return finish(static_cast<__int128>(lhs->lo) - rhs->hi,
                  static_cast<__int128>(lhs->hi) - rhs->lo);
  if (isa<arith::MulIOp>(op)) {

    if (rhs->lo == rhs->hi && rhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->hi) * rhs->lo);
    if (lhs->lo == lhs->hi && lhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->lo) * rhs->hi);
    if (lhs->lo >= 0 && rhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->hi) * rhs->hi);
  }
  if (isa<arith::DivSIOp>(op) && rhs->lo == rhs->hi && rhs->lo > 0)
    return finish(lhs->lo / rhs->lo, lhs->hi / rhs->lo);
  return std::nullopt;
}

bool canUseLastRowLoopBound(scf::ForOp loop, Value pid, int64_t grid) {
  Value bound = loop.getUpperBound();
  auto unsafeArithmetic = [](Operation *op) {
    StringRef ns = op->getName().getDialectNamespace();
    return (ns == "arith" || ns == "math") && !isSpeculatable(op);
  };
  if (grid <= 0) {

    auto function = loop->getParentOfType<triton::FuncOp>();
    if (!function)
      return false;
    auto result = function.walk([&](Operation *op) {
      bool unsafe = unsafeArithmetic(op) &&
                    llvm::any_of(op->getOperands(), [&](Value operand) {
                      return !isPidInvariant(operand, pid);
                    });
      return unsafe ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return false;
  }
  if (!isPidInvariant(bound, pid)) {

    auto result = loop.getBody()->walk([&](Operation *op) {
      return unsafeArithmetic(op) ? WalkResult::interrupt()
                                  : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return false;
  }
  return inferMonotonicBounds(bound, pid, grid).has_value();
}

static bool isScalarBoundSlice(Operation *op) {
  StringRef ns = op->getName().getDialectNamespace();
  return (ns == "arith" || ns == "math") && op->getNumRegions() == 0;
}

static bool isTensorPointer(Type type) {
  auto pointer = dyn_cast<triton::PointerType>(type);
  return pointer && isa<RankedTensorType>(pointer.getPointeeType());
}

int64_t selectRowsPerProgram(int64_t grid, int64_t maxDotRows,
                             int64_t maxTensorBytes, int64_t peakLiveBytes,
                             const RowCountPolicy &policy) {

  const int64_t budget = policy.splitWideAccumulator
                             ? 2 * policy.maxLiftedValueBytes
                             : policy.maxLiftedValueBytes;

  std::optional<int64_t> limit;
  auto tighten = [&](int64_t bound) {
    limit = limit ? std::min(*limit, bound) : bound;
  };

  if (policy.maxL0ABytes > 0 && policy.l0ABudgetBytes > 0)
    tighten(static_cast<int64_t>(policy.l0ABudgetBytes /
                                 static_cast<uint64_t>(policy.maxL0ABytes)));
  if (policy.maxL0CBytes > 0 && policy.l0CBudgetBytes > 0)
    tighten(static_cast<int64_t>(policy.l0CBudgetBytes /
                                 static_cast<uint64_t>(policy.maxL0CBytes)));

  if (maxTensorBytes > 0)
    tighten(budget / maxTensorBytes);

  if (policy.ubBudgetBytes > 0 && peakLiveBytes > 0) {
    const uint64_t perFactorDemand =
        static_cast<uint64_t>(peakLiveBytes) * kUBBufferReplication;
    tighten(static_cast<int64_t>(policy.ubBudgetBytes / perFactorDemand));
  }

  if (policy.targetDotRows > 0 && maxDotRows > 0)
    tighten(policy.targetDotRows / maxDotRows);

  if (grid > 0 && policy.coreNum > 0)
    tighten(grid / static_cast<int64_t>(policy.coreNum));

  if (!limit)
    return 1;

  int64_t factor = 1;
  while (factor <= *limit / 2)
    factor *= 2;

  while (factor >= 2 && grid && grid % factor != 0)
    factor /= 2;

  if (::getenv("TRITON_GRAPH_UB_DEBUG"))
    llvm::errs() << "graph-ub[rowcount]: unLiftedPeak=" << peakLiveBytes
                 << " ubBudget=" << policy.ubBudgetBytes << " limit=" << *limit
                 << " R=" << factor << "\n";

  return factor >= 2 ? factor : 1;
}

int64_t getElementStorageBytes(Type element) {
  if (auto integer = dyn_cast<IntegerType>(element))
    return (integer.getWidth() + 7) / 8;
  if (auto floatType = dyn_cast<FloatType>(element))
    return (floatType.getWidth() + 7) / 8;
  if (isa<triton::PointerType>(element))
    return 8;
  return 0;
}

static Value getRuntimeGridExtent(Block *body) {
  return body->getArgument(body->getNumArguments() - 2);
}

static RankedTensorType halvedType(RankedTensorType full) {
  if (!full || full.getRank() < 2)
    return RankedTensorType();
  int64_t width = full.getShape().back();
  if (width < 2 || width % 2 != 0)
    return RankedTensorType();
  SmallVector<int64_t> shape(full.getShape());
  shape.back() /= 2;
  return RankedTensorType::get(shape, full.getElementType());
}

static bool boundaryAxesInTile(ArrayRef<int32_t> boundaryCheck,
                               ArrayRef<int64_t> tileShape) {
  return llvm::all_of(boundaryCheck, [&](int32_t axis) {
    return axis >= 0 && axis < static_cast<int32_t>(tileShape.size());
  });
}

RankedTensorType RowLiftBuilder::getLiftedType(Type old, bool flat) {
  SmallVector<int64_t> shape{candidate.rowsPerProgram};
  Type elem = old;
  if (auto tensor = dyn_cast<RankedTensorType>(old)) {
    shape.append(tensor.getShape().begin(), tensor.getShape().end());
    elem = tensor.getElementType();
  }
  auto logical = RankedTensorType::get(shape, elem);
  return flat ? cast<RankedTensorType>(flatType(logical)) : logical;
}

Value RowLiftBuilder::getMappedValue(Value old) {
  return valueMapping.lookup(old);
}

bool RowLiftBuilder::hasRowDimension(Value old) {

  if (!valueMapping.contains(old))
    return false;
  return valueMapping.lookup(old).getType() != old.getType();
}

Value RowLiftBuilder::getRowValue(Value old, bool flat) {
  Value value = valueMapping.lookupOrNull(old);
  if (isTensorPointer(old.getType()))
    return value;
  RankedTensorType target = getLiftedType(old.getType(), flat);
  if (value.getType() == target)
    return value;

  Operation *producer = old.getDefiningOp();
  bool capture =
      flat && computeLoop && computeLoop.isDefinedOutsideOfLoop(old) &&
      (!producer || (producer->getBlock() == computeLoop->getBlock() &&
                     producer->isBeforeInBlock(computeLoop)));
  auto key = std::make_pair(old, Type(target));
  if (capture) {
    auto found = boundaryViews.find(key);
    if (found != boundaryViews.end())
      return found->second;
  }
  OpBuilder::InsertionGuard guard(rewriter);
  if (capture && rewrittenLoop)
    rewriter.setInsertionPoint(rewrittenLoop);
  if (!hasRowDimension(old)) {
    auto logical = getLiftedType(old.getType());
    if (!isa<RankedTensorType>(old.getType())) {
      value = rewriter.create<triton::SplatOp>(old.getLoc(), logical, value);
    } else {
      value = rewriter.create<triton::ExpandDimsOp>(old.getLoc(), value, 0);
      value =
          rewriter.create<triton::BroadcastOp>(old.getLoc(), logical, value);
    }
  }
  value = view(value, target);
  if (capture)
    boundaryViews[key] = value;
  return value;
}

Value RowLiftBuilder::reshape(Value v, ArrayRef<int64_t> shape) {
  auto t = cast<RankedTensorType>(v.getType());
  return rewriter.create<triton::ReshapeOp>(
      v.getLoc(), RankedTensorType::get(shape, t.getElementType()), v, false);
}

Value RowLiftBuilder::flattenDotRows(Value v) {
  auto t = cast<RankedTensorType>(v.getType());
  if (t.getRank() == 2)
    return v;
  return reshape(v, {t.getShape()[0] * t.getShape()[1], t.getShape()[2]});
}

Value RowLiftBuilder::broadcastRowValue(Value v, RankedTensorType target) {
  while (cast<RankedTensorType>(v.getType()).getRank() < target.getRank())
    v = rewriter.create<triton::ExpandDimsOp>(
        v.getLoc(), v, cast<RankedTensorType>(v.getType()).getRank());
  return rewriter.create<triton::BroadcastOp>(v.getLoc(), target, v);
}

Value RowLiftBuilder::broadcastFullValue(Value v, RankedTensorType target) {

  if (!isa<RankedTensorType>(v.getType()))
    return rewriter.create<triton::SplatOp>(v.getLoc(), target, v);
  if (v.getType() == target)
    return v;
  return broadcastRowValue(v, target);
}

Value RowLiftBuilder::combineMemoryMask(Value pointer, Value originalMask,
                                        Value extraMask, bool &ok) {

  Value mask =
      mergeMasks(pointer, Value(),
                 {extraMask, candidate.runtimeGrid ? validRows : Value()}, ok);
  if (!ok)
    return Value();
  return mergeMasks(pointer, mask,
                    {originalMask ? getRowValue(originalMask) : Value()}, ok);
}

Value RowLiftBuilder::mergeMasks(Value pointer, Value base,
                                 ArrayRef<Value> parts, bool &ok) {
  ok = true;

  auto pointerType = cast<RankedTensorType>(pointer.getType());
  auto maskType =
      RankedTensorType::get(pointerType.getShape(), rewriter.getI1Type());
  Value mask = base;

  auto merge = [&](Value part) {
    if (!part || !ok)
      return;
    if (part.getType() != maskType) {
      auto partType = dyn_cast<RankedTensorType>(part.getType());

      if (!partType || partType.getElementType() != rewriter.getI1Type() ||
          partType.getRank() >= maskType.getRank()) {
        ok = false;
        return;
      }
      part = broadcastRowValue(part, maskType);
    }
    mask =
        mask ? Value(rewriter.create<arith::AndIOp>(part.getLoc(), mask, part))
             : part;
  };
  for (Value part : parts)
    merge(part);
  return mask;
}

static Value buildMaskPadding(IRRewriter &rewriter, Location loc, Type type,
                              std::optional<triton::PaddingOption> padding) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  Type element = tensor ? tensor.getElementType() : type;
  if (padding != triton::PaddingOption::PAD_NAN)
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(type));
  auto floatType = dyn_cast<FloatType>(element);
  if (!floatType)
    return Value();
  Value nan = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(
               floatType, APFloat::getNaN(floatType.getFloatSemantics())));
  return tensor ? Value(rewriter.create<triton::SplatOp>(loc, type, nan)) : nan;
}

//===----------------------------------------------------------------------===//
// 1. tt.reduce and row-invariant scf.if
//===----------------------------------------------------------------------===//

bool RowLiftBuilder::rewriteReduce(triton::ReduceOp reduce) {

  if (!features.reduce || reduce.getSrcs().empty())
    return false;

  bool flat = flatOperations.contains(reduce);
  SmallVector<Value> sources;
  for (Value src : reduce.getSrcs())
    sources.push_back(getRowValue(src, flat));

  auto replacement = rewriter.create<triton::ReduceOp>(
      reduce.getLoc(), sources, reduce.getAxis() + (flat ? 0 : 1));

  rewriter.cloneRegionBefore(reduce.getCombineOp(), replacement.getCombineOp(),
                             replacement.getCombineOp().end());
  valueMapping.map(reduce.getResults(), replacement.getResults());
  return true;
}

bool liftsRowInvariantIf(scf::IfOp ifOp, Value pid) {

  Value condition = ifOp.getCondition();
  if (!condition.getType().isInteger(1) || !isPidInvariant(condition, pid))
    return false;

  if (!llvm::hasSingleElement(ifOp.getThenRegion().getBlocks()) ||
      (!ifOp.getElseRegion().empty() &&
       !llvm::hasSingleElement(ifOp.getElseRegion().getBlocks())))
    return false;

  return llvm::none_of(ifOp.getResultTypes(), isTensorPointer);
}

bool RowLiftBuilder::rewriteIfOp(scf::IfOp old) {

  if (!features.loopInvariantIf)
    return false;

  Value condition = old.getCondition();
  if (!condition.getType().isInteger(1) || hasRowDimension(condition))
    return false;

  if (!llvm::hasSingleElement(old.getThenRegion().getBlocks()) ||
      (!old.getElseRegion().empty() &&
       !llvm::hasSingleElement(old.getElseRegion().getBlocks())))
    return false;

  bool flat = flatOperations.contains(old);
  SmallVector<Type> types;
  SmallVector<int64_t> starts;
  SmallVector<bool> split;
  for (auto [index, t] : llvm::enumerate(old.getResultTypes())) {

    if (isTensorPointer(t))
      return false;
    starts.push_back(static_cast<int64_t>(types.size()));
    auto lifted = getLiftedType(t, flat);
    bool halve = features.splitWideAccumulator &&
                 plannedHalfWidths.contains(old.getResult(index));
    if (halve) {
      auto halfType = halvedType(lifted);
      if (!halfType)
        return false;
      types.push_back(halfType);
      types.push_back(halfType);
    } else {
      types.push_back(lifted);
    }
    split.push_back(halve);
  }

  auto replacement = rewriter.create<scf::IfOp>(
      old.getLoc(), types, valueMapping.lookupOrDefault(condition),
      !old.getElseRegion().empty());
  if (!rewriteIfBody(old.getThenRegion(), replacement.getThenRegion(), split,
                     flat) ||
      !rewriteIfBody(old.getElseRegion(), replacement.getElseRegion(), split,
                     flat))
    return false;

  rewriter.setInsertionPointAfter(replacement);
  for (auto [index, result] : llvm::enumerate(old.getResults())) {
    int64_t start = starts[index];
    if (split[index]) {
      splitValues[result] = {replacement.getResult(start),
                             replacement.getResult(start + 1)};
      continue;
    }
    valueMapping.map(result, replacement.getResult(start));
  }
  return true;
}

bool RowLiftBuilder::rewriteIfBody(Region &from, Region &to,
                                   ArrayRef<bool> split, bool flat) {
  if (from.empty())
    return true;

  Block *body = &to.front();
  body->clear();
  rewriter.setInsertionPointToStart(body);
  for (Operation &op : from.front().without_terminator())
    if (!rewrite(&op))
      return false;

  auto yield = cast<scf::YieldOp>(from.front().getTerminator());
  SmallVector<Value> values;
  for (auto [index, v] : llvm::enumerate(yield.getOperands())) {

    if (liftedDescriptors.contains(v))
      return false;
    if (index < split.size() && split[index]) {
      auto it = splitValues.find(v);
      if (it == splitValues.end())
        return false;
      values.push_back(it->second.first);
      values.push_back(it->second.second);
      continue;
    }
    values.push_back(getRowValue(v, flat));
  }
  rewriter.create<scf::YieldOp>(yield.getLoc(), values);
  return true;
}

//===----------------------------------------------------------------------===//
// 2. Q/O tensor pointers expanded into pointer tensors
//===----------------------------------------------------------------------===//

Value RowLiftBuilder::expandTileRange(Location loc, int64_t axis,
                                      int64_t extent, RankedTensorType target) {
  auto elementType = target.getElementType();
  auto integer = dyn_cast<IntegerType>(elementType);
  if (!integer)
    return Value();
  auto rangeType = RankedTensorType::get({extent}, rewriter.getI32Type());
  Value range = rewriter.create<triton::MakeRangeOp>(
      loc, rangeType, 0, static_cast<int32_t>(extent));
  if (integer.getWidth() > 32)
    range = rewriter.create<arith::ExtSIOp>(
        loc, RankedTensorType::get({extent}, elementType), range);

  int64_t kept = axis + 1;
  for (int64_t position = 0; position < target.getRank(); ++position)
    if (position != kept)
      range = rewriter.create<triton::ExpandDimsOp>(loc, range, position);
  return rewriter.create<triton::BroadcastOp>(loc, target, range);
}

bool canExpandTensorPointer(triton::MakeTensorPtrOp makePtr, bool enabled) {

  if (!enabled)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(makePtr.getType());
  if (!pointerType)
    return false;
  auto blockType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!blockType || blockType.getRank() == 0)
    return false;

  if (llvm::any_of(blockType.getShape(),
                   [](int64_t extent) { return extent <= 0; }))
    return false;

  auto isInteger = [](Value value) {
    Type type = value.getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    return isa<IntegerType>(type);
  };
  return llvm::all_of(makePtr.getOffsets(), isInteger) &&
         llvm::all_of(makePtr.getShape(), isInteger) &&
         llvm::all_of(makePtr.getStrides(), isInteger);
}

bool RowLiftBuilder::rewriteTensorPointer(triton::MakeTensorPtrOp makePtr) {

  if (!features.tensorPointer)
    return false;

  auto pointerType = cast<triton::PointerType>(makePtr.getType());
  auto blockType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!blockType || blockType.getRank() == 0)
    return false;

  LiftedDescriptor descriptor;
  if (!expandTensorPointer(makePtr, blockType.getShape().back(), Value(),
                           descriptor))
    return false;
  liftedDescriptors[makePtr.getResult()] = descriptor;
  return true;
}

bool RowLiftBuilder::expandTensorPointer(triton::MakeTensorPtrOp makePtr,
                                         int64_t lastExtent,
                                         Value lastOffsetOverride,
                                         LiftedDescriptor &descriptor) {
  auto pointerType = cast<triton::PointerType>(makePtr.getType());
  auto blockType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!blockType || blockType.getRank() == 0)
    return false;
  SmallVector<int64_t> tileShape(blockType.getShape());
  if (lastExtent > 0)
    tileShape.back() = lastExtent;

  if (llvm::any_of(tileShape, [](int64_t extent) { return extent <= 0; }))
    return false;

  Location loc = makePtr.getLoc();

  auto addressType = rewriter.getI32Type();
  auto elementPointer = triton::PointerType::get(blockType.getElementType(),
                                                 pointerType.getAddressSpace());
  SmallVector<int64_t> fullShape{candidate.rowsPerProgram};
  fullShape.append(tileShape.begin(), tileShape.end());
  auto indexType = RankedTensorType::get(fullShape, addressType);
  auto pointerTensorType = RankedTensorType::get(fullShape, elementPointer);

  auto toAddressType = [&](Value v) -> Value {
    Type type = v.getType();
    Type element = type;
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (tensor)
      element = tensor.getElementType();
    auto integer = dyn_cast<IntegerType>(element);
    if (!integer)
      return Value();
    if (integer.getWidth() == 32)
      return v;
    Type converted =
        tensor ? Type(RankedTensorType::get(tensor.getShape(), addressType))
               : Type(addressType);
    if (integer.getWidth() > 32)
      return rewriter.create<arith::TruncIOp>(loc, converted, v);
    return rewriter.create<arith::ExtSIOp>(loc, converted, v);
  };

  SmallVector<Value> offsets, shapes, strides;
  offsets.reserve(tileShape.size());
  shapes.reserve(tileShape.size());
  strides.reserve(tileShape.size());
  const int64_t lastDim = static_cast<int64_t>(tileShape.size()) - 1;
  for (int64_t dim = 0; dim < static_cast<int64_t>(tileShape.size()); ++dim) {
    Value offset =
        (lastOffsetOverride && dim == lastDim)
            ? toAddressType(lastOffsetOverride)
            : toAddressType(getMappedValue(makePtr.getOffsets()[dim]));
    Value shape = toAddressType(getMappedValue(makePtr.getShape()[dim]));
    Value stride = toAddressType(getMappedValue(makePtr.getStrides()[dim]));
    if (!offset || !shape || !stride)
      return false;
    offsets.push_back(offset);
    shapes.push_back(shape);
    strides.push_back(stride);
  }

  SmallVector<int64_t> baseShape(fullShape.begin(), fullShape.end());
  baseShape.back() = 1;
  auto baseIndexType = RankedTensorType::get(baseShape, addressType);
  auto basePointerType = RankedTensorType::get(baseShape, elementPointer);
  Value baseAddress;
  for (int64_t dim = 0; dim + 1 < static_cast<int64_t>(tileShape.size());
       ++dim) {
    Value index = rewriter.create<arith::AddIOp>(
        loc, broadcastFullValue(offsets[dim], baseIndexType),
        expandTileRange(loc, dim, tileShape[dim], baseIndexType));
    Value term = rewriter.create<arith::MulIOp>(
        loc, index, broadcastFullValue(strides[dim], baseIndexType));
    baseAddress =
        baseAddress
            ? Value(rewriter.create<arith::AddIOp>(loc, baseAddress, term))
            : term;
  }

  Value base =
      broadcastFullValue(getMappedValue(makePtr.getBase()), basePointerType);
  Value basePointer = baseAddress
                          ? Value(rewriter.create<triton::AddPtrOp>(
                                loc, basePointerType, base, baseAddress))
                          : base;
  Value pointer = broadcastFullValue(basePointer, pointerTensorType);

  Value lastIndex = rewriter.create<arith::AddIOp>(
      loc, broadcastFullValue(offsets.back(), indexType),
      expandTileRange(loc, lastDim, tileShape.back(), indexType));
  Value lastTerm = rewriter.create<arith::MulIOp>(
      loc, lastIndex, broadcastFullValue(strides.back(), indexType));

  descriptor.pointer = rewriter.create<triton::AddPtrOp>(loc, pointerTensorType,
                                                         pointer, lastTerm);
  descriptor.shapes = shapes;
  descriptor.offsets = offsets;
  descriptor.tileShape = tileShape;
  return true;
}

Value RowLiftBuilder::buildDescriptorMask(const LiftedDescriptor &descriptor,
                                          ArrayRef<int32_t> boundaryCheck,
                                          RankedTensorType indexType) {
  if (boundaryCheck.empty())
    return Value();

  Location loc = descriptor.pointer.getLoc();
  Value mask;
  for (int32_t axis : boundaryCheck) {

    const int64_t rank = static_cast<int64_t>(indexType.getRank());
    if (axis + 1 >= rank)
      return Value();

    auto narrowable = [&](Value v) {
      auto tensor = dyn_cast<RankedTensorType>(v.getType());
      if (!tensor)
        return true;
      for (int64_t dim = 0; dim < tensor.getRank(); ++dim)
        if (dim != axis + 1 && tensor.getShape()[dim] != 1)
          return false;
      return true;
    };
    SmallVector<int64_t> narrow(indexType.getShape().begin(),
                                indexType.getShape().end());
    if (descriptor.tileShape[axis] > 1 &&
        narrowable(descriptor.offsets[axis]) &&
        narrowable(descriptor.shapes[axis])) {
      narrow.assign(rank, 1);
    } else {
      for (int64_t dim = 1; dim < rank; ++dim)
        narrow[dim] = 1;
    }
    narrow[axis + 1] = descriptor.tileShape[axis];
    auto narrowType = RankedTensorType::get(narrow, indexType.getElementType());

    Value index = rewriter.create<arith::AddIOp>(
        loc, broadcastFullValue(descriptor.offsets[axis], narrowType),
        expandTileRange(loc, axis, descriptor.tileShape[axis], narrowType));
    Value zero = rewriter.create<triton::SplatOp>(
        loc, narrowType,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(
                     cast<IntegerType>(indexType.getElementType()), 0)));

    Value lower = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 index, zero);
    Value upper = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, index,
        broadcastFullValue(descriptor.shapes[axis], narrowType));
    Value axisNarrow = rewriter.create<arith::AndIOp>(loc, lower, upper);
    auto maskType =
        RankedTensorType::get(indexType.getShape(), rewriter.getI1Type());
    Value axisMask =
        rewriter.create<triton::BroadcastOp>(loc, maskType, axisNarrow);

    mask = mask ? Value(rewriter.create<arith::AndIOp>(loc, mask, axisMask))
                : axisMask;
  }
  return mask;
}

bool RowLiftBuilder::rewriteDescriptorLoad(triton::LoadOp load,
                                           const LiftedDescriptor &descriptor) {
  Value result;
  if (!emitDescriptorLoad(load, descriptor, result))
    return false;
  valueMapping.map(load.getResult(), result);
  return true;
}

bool RowLiftBuilder::emitDescriptorLoad(triton::LoadOp load,
                                        const LiftedDescriptor &descriptor,
                                        Value &result) {

  if (load.getMask() && !features.maskedStore)
    return false;

  ArrayRef<int32_t> boundaryCheck = load.getBoundaryCheck();
  if (!boundaryAxesInTile(boundaryCheck, descriptor.tileShape))
    return false;

  Location loc = load.getLoc();
  auto pointerType = cast<RankedTensorType>(descriptor.pointer.getType());
  auto resultType = RankedTensorType::get(
      pointerType.getShape(),
      cast<RankedTensorType>(load.getType()).getElementType());
  auto indexType =
      RankedTensorType::get(pointerType.getShape(), rewriter.getI32Type());

  bool ok = true;
  Value mask = combineMemoryMask(
      descriptor.pointer, load.getMask(),
      buildDescriptorMask(descriptor, boundaryCheck, indexType), ok);
  if (!ok)
    return false;

  Value other;
  if (mask) {
    if (load.getOther()) {

      if (getLiftedType(load.getOther().getType()) == resultType) {
        other = broadcastRowValue(getRowValue(load.getOther()), resultType);
      } else {
        other = rebuildHalfOperand(load.getOther(), resultType);
        if (!other) {
          Value lifted = getRowValue(load.getOther());
          if (lifted.getType() == resultType)
            other = lifted;
        }
      }
      if (!other)
        return false;
    } else {
      Value padding;
      if (load.getPadding() == triton::PaddingOption::PAD_NAN) {
        auto floatType = dyn_cast<FloatType>(resultType.getElementType());
        if (!floatType)
          return false;
        padding = rewriter.create<arith::ConstantOp>(
            loc,
            rewriter.getFloatAttr(
                floatType, APFloat::getNaN(floatType.getFloatSemantics())));
      } else {
        padding = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(resultType.getElementType()));
      }
      other = rewriter.create<triton::SplatOp>(loc, resultType, padding);
    }
  }

  result = rewriter
               .create<triton::LoadOp>(loc, descriptor.pointer, mask, other,
                                       ArrayRef<int32_t>{}, load.getPadding(),
                                       load.getCache(), load.getEvict(),
                                       load.getIsVolatile())
               .getResult();
  return true;
}

//===----------------------------------------------------------------------===//
// 3. Per-row reads, including the selected-block index table
//===----------------------------------------------------------------------===//

bool RowLiftBuilder::rewriteRowLoad(triton::LoadOp load) {
  bool insideLoop = static_cast<bool>(load->getParentOfType<scf::ForOp>());

  if (insideLoop && (!features.maskedLoad || !rowActiveMask))
    return false;
  Value pointer = getRowValue(load.getPtr());
  if (!isa<RankedTensorType>(pointer.getType()))
    return false;
  bool ok = true;
  Value mask = combineMemoryMask(pointer, load.getMask(),
                                 insideLoop ? rowActiveMask : Value(), ok);
  if (!ok || (insideLoop && !mask))
    return false;

  auto type = getLiftedType(load.getType());
  Value other;
  if (load.getOther()) {
    other = getRowValue(load.getOther());
    if (insideLoop)
      other = broadcastFullValue(other, type);
  } else {
    other = buildMaskPadding(rewriter, load.getLoc(), type, load.getPadding());
  }
  if (!other)
    return false;
  auto replacement = rewriter.create<triton::LoadOp>(
      load.getLoc(), pointer, mask, other, load.getBoundaryCheck(),
      load.getPadding(), load.getCache(), load.getEvict(),
      load.getIsVolatile());
  valueMapping.map(load.getResult(), replacement.getResult());
  return true;
}

//===----------------------------------------------------------------------===//
// 4. Masked O/lse stores and flat loop state/computation
//===----------------------------------------------------------------------===//

bool RowLiftBuilder::rewriteMaskedStore(triton::StoreOp store) {
  Value pointer = getRowValue(store.getPtr());
  if (!isa<RankedTensorType>(pointer.getType()))
    return false;
  bool ok = true;
  Value mask = combineMemoryMask(pointer, store.getMask(), Value(), ok);
  if (!ok)
    return false;
  rewriter.create<triton::StoreOp>(
      store.getLoc(), pointer, getRowValue(store.getValue()), mask,
      store.getBoundaryCheck(), store.getCache(), store.getEvict());
  return true;
}

bool RowLiftBuilder::rewriteDescriptorStore(
    triton::StoreOp store, const LiftedDescriptor &descriptor) {

  if (store.getMask() && !features.maskedStore)
    return false;

  ArrayRef<int32_t> boundaryCheck = store.getBoundaryCheck();
  if (!boundaryAxesInTile(boundaryCheck, descriptor.tileShape))
    return false;

  Location loc = store.getLoc();
  auto pointerType = cast<RankedTensorType>(descriptor.pointer.getType());
  auto indexType =
      RankedTensorType::get(pointerType.getShape(), rewriter.getI32Type());

  bool ok = true;
  Value mask = combineMemoryMask(
      descriptor.pointer, store.getMask(),
      buildDescriptorMask(descriptor, boundaryCheck, indexType), ok);
  if (!ok)
    return false;

  Value value = getRowValue(store.getValue());
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!valueType || valueType.getShape() != pointerType.getShape())
    return false;

  rewriter.create<triton::StoreOp>(loc, descriptor.pointer, value, mask,
                                   rewriter.getDenseI32ArrayAttr({}),
                                   store.getCache(), store.getEvict());
  return true;
}

static bool hasUniformGroupedTripCount(scf::ForOp loop, Value pid, int64_t rows,
                                       int64_t grid) {
  Value bound = loop.getUpperBound();
  if (isPidInvariant(bound, pid))
    return true;
  if (rows <= 0 || (rows & (rows - 1)) ||
      getConstantIntValue(loop.getLowerBound()) != std::optional<int64_t>(0))
    return false;
  auto step = getConstantIntValue(loop.getStep());
  if (!step || *step <= 0)
    return false;
  auto stripExtension = [](Value value) {
    while (auto ext = value.getDefiningOp<arith::ExtSIOp>())
      value = ext.getIn();
    return value;
  };
  bound = stripExtension(bound);
  if (auto add = bound.getDefiningOp<arith::AddIOp>()) {
    if (getConstantIntValue(add.getRhs()))
      bound = stripExtension(add.getLhs());
    else if (getConstantIntValue(add.getLhs()))
      bound = stripExtension(add.getRhs());
    else
      return false;
  }
  auto div = bound.getDefiningOp<arith::DivSIOp>();
  if (!div)
    return false;
  auto divisor = getConstantIntValue(div.getRhs());
  if (!divisor || *divisor <= 0 || *divisor % rows)
    return false;
  Value numerator = div.getLhs();

  auto bounds = inferMonotonicBounds(numerator, pid, grid);
  if (!bounds || bounds->lo < 0)
    return false;
  if (auto add = numerator.getDefiningOp<arith::AddIOp>()) {
    Value source = add.getLhs();
    auto offset = getConstantIntValue(add.getRhs());
    if (!offset) {
      source = add.getRhs();
      offset = getConstantIntValue(add.getLhs());
    }
    if (!offset || *offset % rows)
      return false;
    numerator = source;
  }
  return stripExtension(numerator) == pid;
}

bool RowLiftBuilder::rewriteLoop(scf::ForOp old) {

  for (Value arg : old.getInitArgs())
    if (liftedDescriptors.contains(arg))
      return false;

  rowActiveMask = Value();
  const bool uniformTripCount =
      features.groupedLoopBounds &&
      hasUniformGroupedTripCount(old, candidate.anchor->getResult(0),
                                 candidate.rowsPerProgram, candidate.grid);

  bool flat = computeLoop == old;
  SmallVector<Value> loopInit;
  SmallVector<std::pair<int64_t, int64_t>> ranges;
  for (auto [index, arg] : llvm::enumerate(old.getInitArgs())) {
    if (llvm::is_contained(plannedSplitIndices, static_cast<int64_t>(index))) {
      SplitValue halves;
      if (!ensureRowHalves(arg, halves))
        return false;
      ranges.push_back({loopInit.size(), 2});
      auto type = halvedType(getLiftedType(arg.getType(), flat));
      loopInit.push_back(view(halves.first, type));
      loopInit.push_back(view(halves.second, type));
    } else {
      ranges.push_back({loopInit.size(), 1});
      loopInit.push_back(getRowValue(arg, flat));
    }
  }

  Value sharedBound = lastRowMapping.lookupOrNull(old.getUpperBound());
  if (!sharedBound)
    return false;

  auto loop = rewriter.create<scf::ForOp>(
      old.getLoc(), getMappedValue(old.getLowerBound()), sharedBound,
      getMappedValue(old.getStep()), loopInit);
  loop->setAttrs(old->getAttrs());
  if (flat)
    rewrittenLoop = loop;

  valueMapping.map(old.getInductionVar(), loop.getInductionVar());
  for (auto [index, arg] : llvm::enumerate(old.getRegionIterArgs())) {
    auto [start, count] = ranges[index];
    if (count == 2)
      splitValues[arg] = {loop.getRegionIterArgs()[start],
                          loop.getRegionIterArgs()[start + 1]};
    else
      valueMapping.map(arg, loop.getRegionIterArgs()[start]);
  }

  rewriter.setInsertionPointToStart(loop.getBody());

  if (uniformTripCount) {
    auto type = getLiftedType(rewriter.getI1Type());
    rowActiveMask = rewriter.create<arith::ConstantOp>(
        old.getLoc(), DenseElementsAttr::get(type, true));
    LLVM_DEBUG(llvm::dbgs() << "row-lift: uniform grouped trip count\n");
  } else if (features.maskedLoad)
    rowActiveMask = createLoopActiveMask(old, loop);
  for (Operation &op : old.getBody()->without_terminator())
    if (!rewrite(&op))
      return false;

  auto yield = cast<scf::YieldOp>(old.getBody()->getTerminator());

  if (!rowActiveMask)
    rowActiveMask = createLoopActiveMask(old, loop);

  auto keep = [&](Value incoming, Value previous, RankedTensorType type) {
    if (uniformTripCount)
      return incoming;
    Value mask = broadcastRowValue(
        rowActiveMask,
        RankedTensorType::get(type.getShape(), rewriter.getI1Type()));
    auto previousType = cast<RankedTensorType>(previous.getType());
    mask = view(mask, RankedTensorType::get(previousType.getShape(),
                                            rewriter.getI1Type()));
    return Value(rewriter.create<arith::SelectOp>(old.getLoc(), mask, incoming,
                                                  previous));
  };

  SmallVector<Value> output;
  for (auto [index, value] : llvm::enumerate(yield.getOperands())) {
    auto [start, count] = ranges[index];
    auto logical = getLiftedType(value.getType());
    if (count == 2) {
      auto it = splitValues.find(value);
      if (it == splitValues.end())
        return false;
      output.push_back(keep(it->second.first, loop.getRegionIterArgs()[start],
                            halvedType(logical)));
      output.push_back(keep(it->second.second,
                            loop.getRegionIterArgs()[start + 1],
                            halvedType(logical)));
      continue;
    }
    output.push_back(keep(getRowValue(value, flat),
                          loop.getRegionIterArgs()[start], logical));
  }

  if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back()))
    rewriter.eraseOp(loop.getBody()->getTerminator());
  rewriter.setInsertionPointToEnd(loop.getBody());
  rewriter.create<scf::YieldOp>(old.getLoc(), output);

  rewriter.setInsertionPointAfter(loop);

  for (auto [index, result] : llvm::enumerate(old.getResults())) {
    auto [start, count] = ranges[index];
    if (count == 2) {

      auto type = halvedType(getLiftedType(result.getType()));
      splitValues[result] = {view(loop.getResults()[start], type),
                             view(loop.getResults()[start + 1], type)};
      continue;
    }
    valueMapping.map(result, view(loop.getResults()[start],
                                  getLiftedType(result.getType())));
  }

  return true;
}

Value RowLiftBuilder::createLoopActiveMask(scf::ForOp old, scf::ForOp loop) {
  Value ivSplat = rewriter.create<triton::SplatOp>(
      old.getLoc(), getLiftedType(old.getInductionVar().getType()),
      loop.getInductionVar());
  Value active =
      rewriter.create<arith::CmpIOp>(old.getLoc(), arith::CmpIPredicate::slt,
                                     ivSplat, getRowValue(old.getUpperBound()));

  bool hasAssert =
      candidate.function
          .walk([](triton::AssertOp) { return WalkResult::interrupt(); })
          .wasInterrupted();
  if (validRows && hasAssert)
    active = rewriter.create<arith::AndIOp>(old.getLoc(), active, validRows);
  return active;
}

void RowLiftBuilder::planFlatCompute(scf::ForOp loop) {
  if (!features.flatCompute)
    return;
  bool consistent = true;
  loop.walk([&](triton::DotOp dot) {
    int64_t m = cast<RankedTensorType>(dot.getType()).getShape()[0];
    consistent &= !dotRows || dotRows == m;
    dotRows = m;
  });
  if (!consistent || !dotRows) {
    dotRows = 0;
    return;
  }
  computeLoop = loop;
  auto collapsible = [&](Type type) {
    return getLiftedType(type, true) != getLiftedType(type);
  };
  auto canFlatten = [&](Operation *op) {
    if (op->getNumResults() == 0 ||
        !llvm::all_of(op->getResultTypes(), collapsible))
      return false;
    if (isa<triton::DotOp, triton::ReshapeOp>(op))
      return true;
    if (auto reduce = dyn_cast<triton::ReduceOp>(op))
      return reduce.getAxis() > 0 &&
             llvm::all_of(reduce.getOperandTypes(), collapsible);
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(op))
      return expand.getAxis() > 0 && collapsible(expand.getSrc().getType());
    return ((op->hasTrait<OpTrait::Elementwise>() && !op->getNumRegions()) ||
            isa<triton::BroadcastOp>(op)) &&
           llvm::all_of(op->getOperandTypes(), [&](Type type) {
             return !isa<RankedTensorType>(type) || collapsible(type);
           });
  };
  SmallVector<Value> captures;
  loop.walk([&](Operation *op) {
    if (isa<scf::IfOp>(op) || canFlatten(op)) {
      flatOperations.insert(op);
      llvm::append_range(captures, op->getOperands());
    }
  });

  while (!captures.empty()) {
    Value value = captures.pop_back_val();
    Operation *op = value.getDefiningOp();
    if (!op || !loop.isDefinedOutsideOfLoop(value) || isa<triton::DotOp>(op) ||
        !canFlatten(op) || !flatOperations.insert(op).second)
      continue;
    llvm::append_range(captures, op->getOperands());
  }
}

Type RowLiftBuilder::flatType(Type type) const {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!dotRows || !tensor || tensor.getRank() < 2 || tensor.getEncoding() ||
      tensor.getShape()[0] != candidate.rowsPerProgram ||
      tensor.getShape()[1] != dotRows ||
      !isa<FloatType, IntegerType>(tensor.getElementType()))
    return type;
  SmallVector<int64_t> shape{candidate.rowsPerProgram * dotRows};
  llvm::append_range(shape, tensor.getShape().drop_front(2));
  return RankedTensorType::get(shape, tensor.getElementType());
}

Value RowLiftBuilder::view(Value value, Type type) {
  return value.getType() == type
             ? value
             : reshape(value, cast<RankedTensorType>(type).getShape());
}

//===----------------------------------------------------------------------===//
// 5. Split wide output accumulators and descriptors along the width
//===----------------------------------------------------------------------===//

static Value ifResultForYield(OpOperand &use) {
  auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
  if (!yield)
    return Value();
  auto ifOp = dyn_cast<scf::IfOp>(yield->getParentOp());
  if (!ifOp)
    return Value();
  return ifOp.getResult(use.getOperandNumber());
}

static bool descriptorHalvesAtWidth(Value pointer, int64_t width) {
  auto makePtr = pointer.getDefiningOp<triton::MakeTensorPtrOp>();
  if (!makePtr)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(makePtr.getType());
  auto blockType =
      pointerType ? dyn_cast<RankedTensorType>(pointerType.getPointeeType())
                  : RankedTensorType();
  return blockType && blockType.getRank() > 0 &&
         blockType.getShape().back() == width && width >= 2 && width % 2 == 0;
}

static bool inspectSplitIsland(Value root, scf::ForOp loop, int64_t width,
                               SmallVectorImpl<Value> &descriptors,
                               DenseSet<Value> &island) {
  SmallVector<Value> work{root};
  bool sawLoopDot = false;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!island.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto dot = dyn_cast<triton::DotOp>(user)) {

        auto load = dot.getB().getDefiningOp<triton::LoadOp>();
        if (use.get() != dot.getC() || !load ||
            !descriptorHalvesAtWidth(load.getPtr(), width))
          return false;
        descriptors.push_back(load.getPtr());
        sawLoopDot |= loop->isProperAncestor(dot.getOperation());
        work.push_back(dot.getResult());
        continue;
      }
      if (auto store = dyn_cast<triton::StoreOp>(user)) {
        if (use.get() != store.getValue() ||
            !descriptorHalvesAtWidth(store.getPtr(), width))
          return false;
        descriptors.push_back(store.getPtr());
        continue;
      }
      if (isa<scf::YieldOp>(user)) {

        Value result = user->getParentOp() == loop.getOperation()
                           ? loop.getResult(use.getOperandNumber())
                           : ifResultForYield(use);
        if (!result)
          return false;
        work.push_back(result);
        continue;
      }
      StringRef dialect = user->getName().getDialectNamespace();
      if (user->getNumResults() != 1 || user->getNumRegions() != 0 ||
          (dialect != "arith" && dialect != "math"))
        return false;
      work.push_back(user->getResult(0));
    }
  }
  return sawLoopDot;
}

struct WideAccumulatorSplitPlan {
  SmallVector<int64_t> indices;
  DenseMap<Value, int64_t> halfWidths;
  SmallVector<Value> descriptors;
};

static bool analyzeWideAccumulatorSplits(scf::ForOp loop, int64_t rows,
                                         int64_t budget,
                                         WideAccumulatorSplitPlan &plan) {
  if (rows <= 0 || budget < 0)
    return false;
  for (auto [index, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    auto tensor = dyn_cast<RankedTensorType>(arg.getType());
    if (!tensor || tensor.getRank() < 2)
      continue;
    int64_t bytes = getElementStorageBytes(tensor.getElementType());
    if (!tensor.hasStaticShape() || bytes <= 0)
      return false;
    for (int64_t extent : tensor.getShape())
      if (__builtin_mul_overflow(bytes, extent, &bytes))
        return false;
    if (__builtin_mul_overflow(bytes, rows, &bytes))
      return false;
    if (bytes <= budget)
      continue;
    int64_t width = tensor.getShape().back();
    if (bytes - budget > budget || width < 2 || width % 2 != 0)
      return false;
    DenseSet<Value> island;
    if (!inspectSplitIsland(arg, loop, width, plan.descriptors, island))
      return false;

    island.insert(loop.getInitArgs()[index]);
    for (Value value : island) {
      auto [it, inserted] = plan.halfWidths.try_emplace(value, width / 2);
      if (!inserted && it->second != width / 2)
        return false;
    }
    plan.indices.push_back(static_cast<int64_t>(index));
  }
  return true;
}

bool canSplitWideAccumulators(scf::ForOp loop, int64_t rowsPerProgram,
                              bool enabled, int64_t budgetBytes) {
  WideAccumulatorSplitPlan plan;
  return !enabled ||
         analyzeWideAccumulatorSplits(loop, rowsPerProgram, budgetBytes, plan);
}

bool RowLiftBuilder::planWideAccumulatorSplits(scf::ForOp loop) {
  if (!features.splitWideAccumulator)
    return true;
  WideAccumulatorSplitPlan plan;
  if (!analyzeWideAccumulatorSplits(loop, candidate.rowsPerProgram,
                                    candidate.splitValueBudgetBytes, plan))
    return false;
  plannedSplitIndices = std::move(plan.indices);
  plannedHalfWidths = std::move(plan.halfWidths);
  for (Value descriptor : plan.descriptors)
    splitDescriptors.try_emplace(descriptor, SplitDescriptor{});
  LLVM_DEBUG(llvm::dbgs() << "row-lift: planned " << plannedSplitIndices.size()
                          << " split indices, " << splitDescriptors.size()
                          << " descriptors\n");
  return true;
}

static bool collapseAcrossLastDim(Value value, RankedTensorType target) {
  auto src = dyn_cast<RankedTensorType>(value.getType());
  if (!src || src.getRank() > target.getRank())
    return false;
  if (!(src.getRank() < target.getRank() || src.getShape().back() == 1))
    return false;
  for (int64_t dim = 0; dim < src.getRank(); ++dim)
    if (src.getShape()[dim] != 1 &&
        src.getShape()[dim] != target.getShape()[dim])
      return false;
  return true;
}

bool RowLiftBuilder::ensureRowHalves(Value value, SplitValue &halves) {
  if (auto it = splitValues.find(value); it != splitValues.end()) {
    halves = it->second;
    return true;
  }
  auto plan = plannedHalfWidths.find(value);
  if (plan == plannedHalfWidths.end())
    return false;
  auto original = dyn_cast<RankedTensorType>(value.getType());
  if (!original || original.getRank() < 2)
    return false;

  auto halfType = halvedType(getLiftedType(original));
  if (!halfType || halfType.getShape().back() != plan->second)
    return false;

  if (Value half = rebuildHalfOperand(value, halfType)) {
    halves = {half, half};
    return true;
  }

  return false;
}

Value RowLiftBuilder::rebuildHalfOperand(Value original,
                                         RankedTensorType halfType) {

  if (splitValues.contains(original))
    return Value();

  if (valueMapping.contains(original) &&
      valueMapping.lookup(original).getType() == halfType)
    return valueMapping.lookup(original);

  Location loc = original.getLoc();

  if (auto cst = original.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat())
      return Value();
    Type element = halfType.getElementType();
    if (isa<FloatType>(element))
      return rewriter.create<arith::ConstantOp>(
          loc,
          DenseElementsAttr::get(halfType, dense.getSplatValue<APFloat>()));
    if (isa<IntegerType>(element))
      return rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(halfType, dense.getSplatValue<APInt>()));
    return Value();
  }

  Operation *source = original.getDefiningOp();
  if (!isa_and_nonnull<triton::SplatOp, triton::BroadcastOp,
                       triton::ExpandDimsOp>(source))
    return Value();
  Value src = source->getOperand(0);
  auto originalType = dyn_cast<RankedTensorType>(original.getType());
  bool flat = originalType && halfType.getRank() == originalType.getRank();
  src = getRowValue(src, flat);
  return collapseAcrossLastDim(src, halfType) ? broadcastRowValue(src, halfType)
                                              : Value();
}

static Value shiftIntegerValue(IRRewriter &rewriter, Value base,
                               int64_t amount) {
  if (!base)
    return Value();
  Location loc = base.getLoc();
  Type type = base.getType();
  Type element = type;
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    element = tensor.getElementType();
  auto integer = dyn_cast<IntegerType>(element);
  if (!integer)
    return Value();
  Value shift = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(integer, amount));
  if (isa<RankedTensorType>(type))
    shift = rewriter.create<triton::SplatOp>(loc, type, shift);
  return rewriter.create<arith::AddIOp>(loc, base, shift);
}

bool RowLiftBuilder::rewriteSplitTensorPointer(triton::MakeTensorPtrOp makePtr,
                                               SplitDescriptor &descriptor) {
  if (!features.tensorPointer)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(makePtr.getType());
  auto blockType =
      pointerType ? dyn_cast<RankedTensorType>(pointerType.getPointeeType())
                  : RankedTensorType();
  if (!blockType || blockType.getRank() == 0)
    return false;
  auto tileShape = blockType.getShape();
  int64_t width = tileShape.back();
  if (width < 2 || width % 2 != 0)
    return false;
  int64_t halfWidth = width / 2;

  Value lastOffset = getMappedValue(makePtr.getOffsets().back());
  Value shifted = shiftIntegerValue(rewriter, lastOffset, halfWidth);
  if (!shifted)
    return false;

  bool rowDependent = llvm::any_of(makePtr->getOperands(),
                                   [&](Value v) { return hasRowDimension(v); });
  if (rowDependent) {
    if (!expandTensorPointer(makePtr, halfWidth, Value(), descriptor.first) ||
        !expandTensorPointer(makePtr, halfWidth, shifted, descriptor.second))
      return false;
    descriptor.expanded = true;
    return true;
  }

  SmallVector<int64_t> halfTile(tileShape.begin(), tileShape.end());
  halfTile.back() = halfWidth;
  auto halfBlock = RankedTensorType::get(halfTile, blockType.getElementType());
  auto halfPointerType =
      triton::PointerType::get(halfBlock, pointerType.getAddressSpace());

  SmallVector<Value> operands;
  for (Value v : makePtr->getOperands())
    operands.push_back(getMappedValue(v));
  auto build = [&](ArrayRef<Value> ops) {
    return rewriter
        .create(makePtr.getLoc(), makePtr->getName().getIdentifier(), ops,
                TypeRange{halfPointerType}, makePtr->getAttrs())
        ->getResult(0);
  };
  descriptor.firstPointer = build(operands);

  operands.back() = shifted;
  descriptor.secondPointer = build(operands);
  descriptor.expanded = false;
  return true;
}

bool RowLiftBuilder::rewriteSplitDescriptorLoad(triton::LoadOp load,
                                                SplitDescriptor &descriptor) {
  if (descriptor.expanded) {
    Value first, second;
    if (!emitDescriptorLoad(load, descriptor.first, first) ||
        !emitDescriptorLoad(load, descriptor.second, second))
      return false;
    splitValues[load.getResult()] = {first, second};
    return true;
  }

  if (!descriptor.firstPointer || !descriptor.secondPointer || load.getMask())
    return false;
  SmallVector<Value> halves;
  for (Value pointer : {descriptor.firstPointer, descriptor.secondPointer}) {
    halves.push_back(
        rewriter
            .create<triton::LoadOp>(load.getLoc(), pointer, Value(), Value(),
                                    load.getBoundaryCheck(), load.getPadding(),
                                    load.getCache(), load.getEvict(),
                                    load.getIsVolatile())
            .getResult());
  }
  splitValues[load.getResult()] = {halves[0], halves[1]};
  return true;
}

bool RowLiftBuilder::rewriteSplitOp(Operation *op) {

  if (!features.splitWideAccumulator)
    return false;
  if (auto dot = dyn_cast<triton::DotOp>(op))
    return rewriteSplitDot(dot);
  if (auto store = dyn_cast<triton::StoreOp>(op))
    return rewriteSplitStore(store);
  return rewriteSplitElementwise(op);
}

bool RowLiftBuilder::rewriteSplitDot(triton::DotOp dot) {
  auto accumulator = splitValues.find(dot.getC());
  if (accumulator == splitValues.end())
    return false;

  if (splitValues.contains(dot.getA()))
    return false;

  auto b = splitValues.find(dot.getB());
  if (b == splitValues.end())
    return false;
  if (!hasRowDimension(dot.getA()) || hasRowDimension(dot.getB()))
    return false;

  bool flat = flatOperations.contains(dot);
  auto accumulatorType =
      dyn_cast<RankedTensorType>(accumulator->second.first.getType());
  if (!accumulatorType || accumulatorType.getRank() != (flat ? 2 : 3))
    return false;
  if (accumulator->second.second.getType() != accumulatorType)
    return false;

  Value a = flattenDotRows(getRowValue(dot.getA(), flat));
  auto bType = dyn_cast<RankedTensorType>(b->second.first.getType());
  if (!bType || bType.getRank() != 2 || b->second.second.getType() != bType ||
      bType.getShape().back() != accumulatorType.getShape().back())
    return false;

  int64_t flatRows = candidate.rowsPerProgram *
                     cast<RankedTensorType>(dot.getType()).getShape()[0];
  auto dotType =
      RankedTensorType::get({flatRows, accumulatorType.getShape().back()},
                            accumulatorType.getElementType());

  SmallVector<Value> operands0{a, b->second.first,
                               flattenDotRows(accumulator->second.first)};
  SmallVector<Value> operands1{a, b->second.second,
                               flattenDotRows(accumulator->second.second)};
  SmallVector<Value> halves;
  for (const SmallVector<Value> &operands : {operands0, operands1}) {
    Operation *created =
        rewriter.create(dot.getLoc(), dot->getName().getIdentifier(), operands,
                        TypeRange{dotType}, dot->getAttrs());
    halves.push_back(
        flat ? created->getResult(0)
             : reshape(created->getResult(0), accumulatorType.getShape()));
  }

  splitValues[dot.getResult()] = {halves[0], halves[1]};
  return true;
}

bool RowLiftBuilder::rewriteSplitElementwise(Operation *op) {

  if (op->getNumResults() != 1 || op->getNumRegions() != 0)
    return false;
  StringRef dialect = op->getName().getDialectNamespace();
  if (dialect != "arith" && dialect != "math")
    return false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.getRank() < 2)
    return false;

  bool flat = flatOperations.contains(op);
  auto halfType = halvedType(getLiftedType(resultType, flat));
  if (!halfType)
    return false;

  SmallVector<std::pair<Value, Value>> operands;
  for (Value operand : op->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType())) {
      operands.push_back({operand, operand});
      continue;
    }
    auto operandHalfType = halvedType(getLiftedType(operand.getType(), flat));
    if (!operandHalfType)
      return false;
    auto split = splitValues.find(operand);
    if (split != splitValues.end()) {
      if (split->second.first.getType() != operandHalfType ||
          split->second.second.getType() != operandHalfType)
        return false;
      operands.push_back({split->second.first, split->second.second});
      continue;
    }

    Value half = rebuildHalfOperand(operand, operandHalfType);
    if (!half)
      return false;
    operands.push_back({half, half});
  }

  SmallVector<Value> results;
  for (bool first : {true, false}) {
    SmallVector<Value> halfOperands;
    for (auto &pair : operands)
      halfOperands.push_back(first ? pair.first : pair.second);
    results.push_back(rewriter
                          .create(op->getLoc(), op->getName().getIdentifier(),
                                  halfOperands, TypeRange{halfType},
                                  op->getAttrs())
                          ->getResult(0));
  }
  splitValues[op->getResult(0)] = {results[0], results[1]};
  return true;
}

bool RowLiftBuilder::rewriteSplitStore(triton::StoreOp store) {
  auto fail = [](const char *why) {
    LLVM_DEBUG(llvm::dbgs() << "split-store: fail " << why << "\n");
    return false;
  };
  auto valueIt = splitValues.find(store.getValue());
  if (valueIt == splitValues.end())
    return fail("value not split");
  auto descriptorIt = splitDescriptors.find(store.getPtr());
  if (descriptorIt == splitDescriptors.end())
    return fail("descriptor not planned");
  SplitDescriptor &descriptor = descriptorIt->second;

  auto valueType = dyn_cast<RankedTensorType>(valueIt->second.first.getType());
  if (!valueType || valueIt->second.second.getType() != valueType)
    return fail("value halves not same type");
  const Value halfValues[] = {valueIt->second.first, valueIt->second.second};

  Location loc = store.getLoc();

  if (store.getMask() && !features.maskedStore)
    return fail("masked store unsupported");

  if (descriptor.expanded) {

    ArrayRef<int32_t> boundaryCheck = store.getBoundaryCheck();
    const LiftedDescriptor *halves[] = {&descriptor.first, &descriptor.second};
    for (const LiftedDescriptor *half : halves) {
      if (!boundaryAxesInTile(boundaryCheck, half->tileShape))
        return fail("boundary axis out of tile rank");
    }
    for (auto [index, half] : llvm::enumerate(halves)) {
      auto pointer = half->pointer;
      auto pointerType = cast<RankedTensorType>(pointer.getType());
      if (pointerType.getShape() != valueType.getShape())
        return fail("expanded pointer shape != value shape");
      auto element =
          dyn_cast<triton::PointerType>(pointerType.getElementType());
      if (!element || element.getPointeeType() != valueType.getElementType())
        return fail("expanded pointer element mismatch");
      auto indexType =
          RankedTensorType::get(pointerType.getShape(), rewriter.getI32Type());
      Value boundary =
          buildDescriptorMask(*half, store.getBoundaryCheck(), indexType);

      Value original;
      if (store.getMask()) {
        auto maskType =
            RankedTensorType::get(pointerType.getShape(), rewriter.getI1Type());
        original = rebuildHalfOperand(store.getMask(), maskType);
        if (!original)
          return fail("mask cannot fold to half width");
      }
      bool ok = true;
      Value mask = mergeMasks(
          pointer, Value(),
          {boundary, candidate.runtimeGrid ? validRows : Value(), original},
          ok);
      if (!ok)
        return fail("mask merge failed");
      Value value = halfValues[index];
      rewriter.create<triton::StoreOp>(loc, pointer, value, mask,
                                       ArrayRef<int32_t>{}, store.getCache(),
                                       store.getEvict());
    }
    return true;
  }

  if (!descriptor.firstPointer || !descriptor.secondPointer)
    return fail("block pointers missing");
  if (store.getMask())
    return fail("masked store unsupported");
  Value pointers[] = {descriptor.firstPointer, descriptor.secondPointer};
  for (auto [index, pointer] : llvm::enumerate(pointers)) {
    auto pointerType = dyn_cast<triton::PointerType>(pointer.getType());
    if (!pointerType || pointerType.getPointeeType() != valueType)
      return fail("block pointer pointee != value type");
    Value value = halfValues[index];
    rewriter.create<triton::StoreOp>(loc, pointer, value, Value(),
                                     store.getBoundaryCheck(), store.getCache(),
                                     store.getEvict());
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Builder entry and operation dispatch
//===----------------------------------------------------------------------===//

RowLiftBuilder::RowLiftBuilder(IRRewriter &rewriter, RowLiftCandidate candidate)
    : rewriter(rewriter), candidate(candidate), features(candidate.features) {}

bool RowLiftBuilder::run(triton::FuncOp fn) {
  Block *old = &fn.getBody().front();

  auto loops = old->getOps<scf::ForOp>();
  if (loops.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "row-lift: no loop to lift\n");
    return false;
  }
  scf::ForOp loop = *loops.begin();
  Value upperBound = loop.getUpperBound();
  if (!canUseLastRowLoopBound(loop, candidate.anchor->getResult(0),
                              candidate.grid))
    return false;

  plannedSplitIndices.clear();
  plannedHalfWidths.clear();
  if (!planWideAccumulatorSplits(loop))
    return false;
  planFlatCompute(loop);

  auto *fresh = new Block();
  fn.getBody().push_back(fresh);

  for (BlockArgument arg : old->getArguments()) {
    Value newArg = fresh->addArgument(arg.getType(), arg.getLoc());
    valueMapping.map(arg, newArg);
    lastRowMapping.map(arg, newArg);
  }

  rewriter.setInsertionPointToStart(fresh);
  SmallVector<Operation *> captures;
  for (Operation &op : old->without_terminator()) {

    if (flatOperations.contains(&op) &&
        !isPidInvariant(op.getResult(0), candidate.anchor->getResult(0))) {
      captures.push_back(&op);
      continue;
    }
    if (&op == computeLoop || llvm::any_of(captures, [&](Operation *capture) {
          return llvm::any_of(capture->getUsers(), [&](Operation *user) {
            return &op == user || op.isAncestor(user);
          });
        })) {
      for (Operation *capture : captures)
        if (!rewrite(capture))
          return false;
      captures.clear();
    }

    if (&op == candidate.anchor) {

      auto pid = cast<triton::GetProgramIdOp>(rewriter.clone(op));
      Value base = rewriter.create<arith::MulIOp>(
          op.getLoc(), pid.getResult(),
          rewriter.create<arith::ConstantIntOp>(op.getLoc(),
                                                candidate.rowsPerProgram, 32));

      Value last = rewriter.create<arith::AddIOp>(
          op.getLoc(), base,
          rewriter.create<arith::ConstantIntOp>(
              op.getLoc(), candidate.rowsPerProgram - 1, 32));

      if (candidate.runtimeGrid) {
        Value extent = getRuntimeGridExtent(fresh);
        Value lastValid = rewriter.create<arith::SubIOp>(
            op.getLoc(), extent,
            rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 32));
        last = rewriter.create<arith::MinSIOp>(op.getLoc(), last, lastValid);
      }
      lastRowMapping.map(op.getResult(0), last);

      auto t = getLiftedType(op.getResult(0).getType());
      Value lanes = rewriter.create<triton::MakeRangeOp>(
          op.getLoc(), t, 0, candidate.rowsPerProgram);
      Value start = rewriter.create<triton::SplatOp>(op.getLoc(), t, base);
      valueMapping.map(op.getResult(0),
                       rewriter.create<arith::AddIOp>(op.getLoc(), start, lanes)
                           .getResult());

      if (candidate.runtimeGrid) {
        Value extent = getRuntimeGridExtent(fresh);

        Value rowsLeft =
            rewriter.create<arith::SubIOp>(op.getLoc(), extent, base);
        Value limit =
            rewriter.create<triton::SplatOp>(op.getLoc(), t, rowsLeft);
        validRows = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::slt, lanes, limit);
      }
      continue;
    }

    if (op.getNumResults() == 1 &&
        !isa<RankedTensorType>(op.getResult(0).getType()) &&
        isScalarBoundSlice(&op) &&
        dependsOnValueThroughRegions(upperBound, op.getResult(0)))
      rewriter.clone(op, lastRowMapping);

    if (!rewrite(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "row-lift: rewrite failed at " << op << "\n");
      return false;
    }

    for (Value v : op.getResults()) {

      if (!valueMapping.contains(v))
        continue;
      Value mapped = valueMapping.lookup(v);
      if (!hasRowDimension(v) && !lastRowMapping.contains(v))
        lastRowMapping.map(v, mapped);
    }
  }

  rewriter.create<triton::ReturnOp>(fn.getLoc());
  old->dropAllDefinedValueUses();
  old->erase();
  return true;
}

bool RowLiftBuilder::rewrite(Operation *op) {
  bool flat = flatOperations.contains(op);

  if (auto loop = dyn_cast<scf::ForOp>(op))
    return rewriteLoop(loop);

  if (auto makePtr = dyn_cast<triton::MakeTensorPtrOp>(op)) {
    auto split = splitDescriptors.find(makePtr.getResult());
    if (split != splitDescriptors.end())
      return rewriteSplitTensorPointer(makePtr, split->second);
  }
  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    auto split = splitDescriptors.find(load.getPtr());
    if (split != splitDescriptors.end())
      return rewriteSplitDescriptorLoad(load, split->second);
  }

  if (llvm::any_of(op->getOperands(),
                   [&](Value v) { return splitValues.contains(v); }))
    return rewriteSplitOp(op);

  bool descriptor = llvm::any_of(op->getOperands(), [&](Value v) {
    return liftedDescriptors.contains(v);
  });
  bool dependent = descriptor || llvm::any_of(op->getOperands(), [&](Value v) {
                     return hasRowDimension(v);
                   });

  if (!dependent)
    op->walk([&](Operation *nested) {
      for (Value v : nested->getOperands())
        if (!dependent && valueMapping.contains(v) && hasRowDimension(v))
          dependent = true;
    });

  if (!dependent) {
    rewriter.clone(*op, valueMapping);
    return true;
  }

  if (descriptor && !isa<triton::LoadOp, triton::StoreOp>(op))
    return false;

  if (auto reduce = dyn_cast<triton::ReduceOp>(op))
    return rewriteReduce(reduce);
  if (auto makePtr = dyn_cast<triton::MakeTensorPtrOp>(op))
    return rewriteTensorPointer(makePtr);

  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp);
  if (op->getNumRegions() != 0)
    return false;

  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    auto it = liftedDescriptors.find(load.getPtr());
    if (it != liftedDescriptors.end())
      return rewriteDescriptorLoad(load, it->second);
  }
  if (auto store = dyn_cast<triton::StoreOp>(op)) {
    auto it = liftedDescriptors.find(store.getPtr());
    if (it != liftedDescriptors.end())
      return rewriteDescriptorStore(store, it->second);
  }

  if (auto load = dyn_cast<triton::LoadOp>(op))
    if (op->getParentOfType<scf::ForOp>() || candidate.runtimeGrid)
      return rewriteRowLoad(load);

  if (auto dot = dyn_cast<triton::DotOp>(op)) {
    if (!hasRowDimension(dot.getA()) || hasRowDimension(dot.getB()))
      return false;

    auto result = cast<RankedTensorType>(dot.getType());
    SmallVector<Value> operands{flattenDotRows(getRowValue(dot.getA(), flat)),
                                getMappedValue(dot.getB()),
                                flattenDotRows(getRowValue(dot.getC(), flat))};
    Type flatType = RankedTensorType::get(
        {candidate.rowsPerProgram * result.getShape()[0], result.getShape()[1]},
        result.getElementType());
    Operation *d =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                        TypeRange{flatType}, op->getAttrs());
    valueMapping.map(
        dot.getResult(),
        flat ? d->getResult(0)
             : reshape(d->getResult(0), getLiftedType(result).getShape()));
    return true;
  }

  if (auto splat = dyn_cast<triton::SplatOp>(op)) {
    valueMapping.map(splat.getResult(),
                     broadcastRowValue(getRowValue(splat.getSrc()),
                                       getLiftedType(splat.getType())));
    return true;
  }

  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
    valueMapping.map(expand.getResult(),
                     rewriter
                         .create<triton::ExpandDimsOp>(
                             op->getLoc(), getRowValue(expand.getSrc(), flat),
                             expand.getAxis() + (flat ? 0 : 1))
                         .getResult());
    return true;
  }

  if (auto view = dyn_cast<triton::ReshapeOp>(op)) {

    valueMapping.map(view.getResult(),
                     reshape(getRowValue(view.getSrc(), flat),
                             getLiftedType(view.getType(), flat).getShape()));
    return true;
  }

  if (auto trans = dyn_cast<triton::TransOp>(op)) {
    SmallVector<int32_t> order{0};
    for (int32_t axis : trans.getOrder())
      order.push_back(axis + 1);
    valueMapping.map(trans.getResult(),
                     rewriter
                         .create<triton::TransOp>(
                             op->getLoc(), getRowValue(trans.getSrc()), order)
                         .getResult());
    return true;
  }

  if (auto store = dyn_cast<triton::StoreOp>(op)) {
    if (candidate.runtimeGrid || (store.getMask() && features.maskedStore)) {
      return rewriteMaskedStore(store);
    }

    if (op->getParentOfType<scf::ForOp>() && !store.getMask())
      return false;
  }

  SmallVector<Value> operands;
  for (Value v : op->getOperands())
    operands.push_back(getRowValue(v, flat));

  SmallVector<Type> types;
  for (Type t : op->getResultTypes())
    types.push_back(getLiftedType(t, flat));

  Operation *cloned =
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                      types, op->getAttrs());

  if (auto flags = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(cloned))
    cloned->setAttr(
        flags.getIntegerOverflowAttrName(),
        arith::IntegerOverflowFlagsAttr::get(
            rewriter.getContext(), arith::IntegerOverflowFlags::none));
  valueMapping.map(op->getResults(), cloned->getResults());
  return true;
}

} // namespace cfg
} // namespace triton
} // namespace mlir
