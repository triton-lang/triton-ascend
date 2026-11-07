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

#include "TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::triton::ptr {
namespace {
using Analyze = CommonPtrRules::Analyze;
using WithBindings = CommonPtrRules::WithBindings;
using Binding = CommonPtrRules::Binding;
using AxisKinds = SmallVector<AxisKind>;

// Scalars and singleton tensors already have exact uniform facts. Dynamic or
// empty tensors cannot supply the coordinates used by this representation.
bool isNumericCarrier(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getNumElements() > 1 &&
         succeeded(detail::getIntegerWidth(type));
}

AxisKinds unknownAxes(Value value) {
  AxisKinds axes;
  for (int64_t size : cast<RankedTensorType>(value.getType()).getShape())
    axes.push_back(size == 1 ? AxisKind::Invariant : AxisKind::Unknown);
  return axes;
}

AxisKinds analyzeAxes(Value value, Analyze analyze) {
  auto info = analyze(value);
  if (failed(info))
    return unknownAxes(value);
  AxisKinds axes;
  for (auto [size, axis] :
       llvm::zip_equal(info->offsets.getShape(), info->offsets.axes))
    axes.push_back(size == 1 ? AxisKind::Invariant : axis.kind);
  return axes;
}

// Invariant is the zero-stride subset of Structured. Each axis can weaken at
// most twice, independently of the other axes and other carried values.
bool joinAxes(AxisKinds &current, ArrayRef<AxisKind> incoming) {
  bool changed = false;
  for (auto [axis, next] : llvm::zip_equal(current, incoming)) {
    AxisKind joined = AxisKind::Invariant;
    if (axis == AxisKind::Unknown || next == AxisKind::Unknown)
      joined = AxisKind::Unknown;
    else if (axis == AxisKind::Structured || next == AxisKind::Structured)
      joined = AxisKind::Structured;
    changed |= axis != joined;
    axis = joined;
  }
  return changed;
}

// These samples represent an assumed or proved axis class; they never prove
// that an arbitrary tensor is affine. Keep the current SSA as the exact value,
// so origins and strides may change on every iteration.
PtrInfo carrierInfo(Value value, ArrayRef<AxisKind> kinds, OpBuilder &builder) {
  auto offsets = buildOffsetInfo(value, builder);
  assert(succeeded(offsets) && "expected a supported numeric carrier");
  bool needsOrigin =
      llvm::any_of(
          kinds, [](AxisKind kind) { return kind == AxisKind::Structured; }) ||
      llvm::all_of(kinds,
                   [](AxisKind kind) { return kind == AxisKind::Invariant; });
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(value);
  Location loc = value.getLoc();
  SmallVector<Value> indices;
  Value origin;
  auto extract = [&]() -> Value {
    Value scalar = builder.create<tensor::ExtractOp>(loc, value, indices);
    if (!scalar.getType().isIndex())
      scalar = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                  scalar);
    return scalar;
  };
  if (needsOrigin) {
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    indices.assign(kinds.size(), zero);
    origin = extract();
    offsets->scalarOffset = origin;
  }
  for (auto [i, kind] : llvm::enumerate(kinds)) {
    OpFoldResult stride = builder.getIndexAttr(0);
    if (kind == AxisKind::Structured) {
      Value saved = indices[i];
      indices[i] = builder.create<arith::ConstantIndexOp>(loc, 1);
      stride =
          builder.create<arith::SubIOp>(loc, extract(), origin).getResult();
      indices[i] = saved;
    }
    offsets->axes[i] = AxisInfo(kind, stride);
  }
  return PtrInfo{{}, std::move(*offsets)};
}

struct Carrier {
  Value argument;
  Value incoming;
  AxisKinds axes;
};

SmallVector<Binding, 0> assumptions(ArrayRef<Carrier> first,
                                    ArrayRef<Carrier> second,
                                    OpBuilder &builder) {
  SmallVector<Binding, 0> bindings;
  for (ArrayRef<Carrier> carriers : {first, second})
    for (const Carrier &carrier : carriers)
      bindings.emplace_back(
          carrier.argument,
          carrierInfo(carrier.argument, carrier.axes, builder));
  return bindings;
}

bool joinIncoming(SmallVectorImpl<Carrier> &carriers, Analyze analyze) {
  bool changed = false;
  for (Carrier &carrier : carriers)
    changed |= joinAxes(carrier.axes, analyzeAxes(carrier.incoming, analyze));
  return changed;
}

FailureOr<PtrInfo> analyzeFor(Value value, scf::ForOp loop, Analyze analyze,
                              WithBindings withBindings, OpBuilder &builder) {
  SmallVector<Carrier> carriers;
  auto yields =
      cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands();
  for (auto [arg, init, yielded] :
       llvm::zip_equal(loop.getRegionIterArgs(), loop.getInitArgs(), yields))
    if (isNumericCarrier(arg))
      carriers.push_back({arg, yielded, analyzeAxes(init, analyze)});

  // A_{n+1} = join(A_n, transfer(A_n)); A_0 already contains the initial
  // values, including the zero-trip result. Every pass has a fresh cache.
  bool changed;
  do {
    changed = false;
    auto bindings = assumptions(carriers, {}, builder);
    if (failed(withBindings(bindings, [&](Analyze nested) {
          changed = joinIncoming(carriers, nested);
          return success();
        })))
      return failure();
  } while (changed);

  Value argument = value;
  if (auto result = dyn_cast<OpResult>(value))
    argument = loop.getRegionIterArgs()[result.getResultNumber()];
  for (const Carrier &carrier : carriers)
    if (carrier.argument == argument)
      return carrierInfo(value, carrier.axes, builder);
  llvm_unreachable("queried for value must be a numeric carrier");
}

FailureOr<PtrInfo> analyzeWhile(Value value, scf::WhileOp loop, Analyze analyze,
                                WithBindings withBindings, OpBuilder &builder) {
  SmallVector<Carrier> before, after;
  for (auto [arg, init, yielded] :
       llvm::zip_equal(loop.getBeforeArguments(), loop.getInits(),
                       loop.getYieldOp().getOperands()))
    if (isNumericCarrier(arg))
      before.push_back({arg, yielded, analyzeAxes(init, analyze)});

  // condition arguments feed both the after region and the loop results.
  // Their count, order and types need not match the before region.
  auto initialBindings = assumptions(before, {}, builder);
  if (failed(withBindings(initialBindings, [&](Analyze nested) {
        for (auto [arg, condition] : llvm::zip_equal(
                 loop.getAfterArguments(), loop.getConditionOp().getArgs()))
          if (isNumericCarrier(arg))
            after.push_back({arg, condition, analyzeAxes(condition, nested)});
        return success();
      })))
    return failure();

  bool changed;
  do {
    changed = false;
    auto bindings = assumptions(before, after, builder);
    if (failed(withBindings(bindings, [&](Analyze nested) {
          changed = joinIncoming(before, nested);
          changed |= joinIncoming(after, nested);
          return success();
        })))
      return failure();
  } while (changed);

  Value argument = value;
  if (auto result = dyn_cast<OpResult>(value))
    argument = loop.getAfterArguments()[result.getResultNumber()];
  for (ArrayRef<Carrier> carriers :
       {ArrayRef<Carrier>(before), ArrayRef<Carrier>(after)})
    for (const Carrier &carrier : carriers)
      if (carrier.argument == argument)
        return carrierInfo(value, carrier.axes, builder);
  llvm_unreachable("queried while value must be a numeric carrier");
}

FailureOr<PtrInfo> analyzeIf(OpResult value, scf::IfOp branch, Analyze analyze,
                             OpBuilder &builder) {
  unsigned i = value.getResultNumber();
  AxisKinds axes = analyzeAxes(branch.thenYield().getOperand(i), analyze);
  joinAxes(axes, analyzeAxes(branch.elseYield().getOperand(i), analyze));
  // A scalar condition chooses a whole tensor. Never export branch-local
  // numerical fields into the parent region.
  return carrierInfo(value, axes, builder);
}
} // namespace

bool CommonPtrRules::isControlFlowValue(Value value) {
  if (!value || !isNumericCarrier(value))
    return false;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isa<scf::ForOp, scf::WhileOp>(arg.getOwner()->getParentOp());
  return isa<scf::ForOp, scf::WhileOp, scf::IfOp>(value.getDefiningOp());
}

FailureOr<PtrInfo> CommonPtrRules::parseControlFlow(Value value,
                                                    Analyze analyze,
                                                    WithBindings withBindings,
                                                    OpBuilder &builder) const {
  Operation *op = value.getDefiningOp();
  if (auto arg = dyn_cast<BlockArgument>(value))
    op = arg.getOwner()->getParentOp();
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return analyzeFor(value, loop, analyze, withBindings, builder);
  if (auto loop = dyn_cast<scf::WhileOp>(op))
    return analyzeWhile(value, loop, analyze, withBindings, builder);
  return analyzeIf(cast<OpResult>(value), cast<scf::IfOp>(op), analyze,
                   builder);
}
} // namespace mlir::triton::ptr
