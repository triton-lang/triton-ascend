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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::triton::ptr {
using namespace detail;
using Result = CommonPtrRules::Result;

// Local expression queries used by the recursive rules.
namespace {
// fptosi and integer reduction results need only axis invariance from float
// intermediates. Keep these queries local to preserve OffsetInfo's integer
// origin/stride contract; broader reuse can justify integration into parse.
// Deterministic arithmetic preserves input axis invariance, including in
// reduction combiners.
bool isInvariancePreservingArithmetic(Operation &op) {
  return isa<arith::ConstantOp>(op) ||
         (op.getName().getDialectNamespace() == "arith" &&
          op.hasTrait<OpTrait::Elementwise>() && op.getNumRegions() == 0);
}

// Each combiner operation must preserve the inputs' axis invariance.
bool hasInvariancePreservingCombiner(triton::ReduceOp op) {
  return llvm::all_of(op.getCombineOp().front().without_terminator(),
                      isInvariancePreservingArithmetic);
}

// Float intermediates need only per-axis equality facts. Integer inputs still
// go through the caller's Rules. This local memo avoids rewalking shared float
// expressions; it owns no SSA helpers and has no session lifecycle.
// An unanalyzable input contributes no invariant axes.
SmallVector<bool>
inferInvariantAxes(Value value, CommonPtrRules::Analyze analyze,
                   llvm::DenseMap<Value, SmallVector<bool>> &invarianceCache) {
  if (auto it = invarianceCache.find(value); it != invarianceCache.end())
    return it->second;
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  SmallVector<bool> invariantAxes(tensorType ? tensorType.getRank() : 0, false);
  if (!tensorType)
    return invariantAxes;
  auto remember = [&]() {
    invarianceCache.try_emplace(value, invariantAxes);
    return invariantAxes;
  };
  if (!isa<FloatType>(tensorType.getElementType())) {
    auto info = analyze(value);
    if (succeeded(info))
      for (unsigned i = 0; i < invariantAxes.size(); ++i)
        invariantAxes[i] = info->offsets.axes[i].kind == AxisKind::Invariant;
    return remember();
  }
  if (value.getDefiningOp<triton::SplatOp>()) {
    std::fill(invariantAxes.begin(), invariantAxes.end(), true);
  } else if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue()))
      if (dense.isSplat())
        std::fill(invariantAxes.begin(), invariantAxes.end(), true);
  } else if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
    auto sourceInvariantAxes =
        inferInvariantAxes(broadcast.getSrc(), analyze, invarianceCache);
    auto sourceShape =
        cast<RankedTensorType>(broadcast.getSrc().getType()).getShape();
    for (unsigned i = 0; i < invariantAxes.size(); ++i)
      invariantAxes[i] = sourceShape[i] == 1 || sourceInvariantAxes[i];
  } else if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
    invariantAxes =
        inferInvariantAxes(expand.getSrc(), analyze, invarianceCache);
    invariantAxes.insert(invariantAxes.begin() + expand.getAxis(), true);
  } else if (Operation *op = value.getDefiningOp();
             op && (isInvariancePreservingArithmetic(*op) ||
                    isa<triton::ClampFOp>(op))) {
    std::fill(invariantAxes.begin(), invariantAxes.end(), true);
    for (Value operand : op->getOperands()) {
      auto sourceInvariantAxes =
          inferInvariantAxes(operand, analyze, invarianceCache);
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      if (sourceInvariantAxes.size() != invariantAxes.size())
        return SmallVector<bool>(invariantAxes.size(), false);
      for (unsigned i = 0; i < invariantAxes.size(); ++i)
        invariantAxes[i] = invariantAxes[i] && sourceInvariantAxes[i];
    }
  }
  return remember();
}

} // namespace

// Rule entry points.
LogicalResult CommonPtrRules::validateBinding(Value value,
                                              const Result &info) const {
  if (!value || info.isEmpty())
    return failure();
  if (info.base)
    return success(validPtrBinding(value, info));
  return success(!info.block && validOffsetBinding(value, info.offsets));
}

FailureOr<Result> CommonPtrRules::parseOperation(Value value, Analyze analyze,
                                                 OpBuilder &builder) const {
  // Validate the supported integer and pointer types before dispatching.
  if (!value)
    return failure();
  bool integer = succeeded(detail::getIntegerWidth(value.getType()));
  bool ordinary = detail::isOrdinaryPtrType(value.getType());
  bool block = detail::isBlockPtrType(value.getType());
  if (!integer && !ordinary && !block) {
    emitError(value.getLoc())
        << "unsupported type passed to pointer analysis: " << value.getType();
    return failure();
  }
  // Analyze splat sources even for singleton results, preserving bound origins.
  if (auto op = value.getDefiningOp<triton::SplatOp>())
    return parseSplat(op, analyze, builder);
  // A scalar or singleton tensor is already an exact dynamic origin/stride.
  // Reuse its final SSA instead of rebuilding its arithmetic in index;
  // larger tensors need axis analysis below.
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (integer &&
      (isa<IntegerType, IndexType>(value.getType()) ||
       (tensor && tensor.hasStaticShape() && tensor.getNumElements() == 1))) {
    builder.setInsertionPointAfterValue(value);
    return buildIntegerInfo(value, builder);
  }
  // Block pointers derive coordinates and strides from their descriptors.
  if (block) {
    if (auto op = value.getDefiningOp<triton::MakeTensorPtrOp>())
      return parseMakeTensorPtr(op, analyze, builder);
    if (auto op = value.getDefiningOp<triton::AdvanceOp>())
      return parseAdvance(op, analyze, builder);
    return failure();
  }
  // Shape and selection rules apply to both offsets and ordinary pointers.
  if (auto op = value.getDefiningOp<triton::BroadcastOp>())
    return parseBroadcast(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::ExpandDimsOp>())
    return parseExpandDims(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::ReshapeOp>())
    return parseReshape(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::ExtractSliceOp>())
    return parseExtractSlice(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::InsertOp>())
    return parseInsert(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::InsertSliceOp>())
    return parseInsertSlice(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::CastOp>())
    return parseTensorCast(op, analyze, builder);
  if (auto op = value.getDefiningOp<arith::SelectOp>())
    return parseSelect(op, analyze, builder);
  // Ordinary pointers combine a scalar base with element offsets.
  if (ordinary) {
    if (auto op = value.getDefiningOp<triton::AddPtrOp>())
      return parseAddPtr(op, analyze, builder);
    if (auto op = value.getDefiningOp<triton::BitcastOp>())
      return parseBitcast(op, analyze, builder);
    return buildScalarPtrInfo(value, builder);
  }
  // Integer tensor rules infer origins and relationships between lanes.
  if (auto op = value.getDefiningOp<triton::MakeRangeOp>()) {
    return parseMakeRange(op, builder);
  }
  if (auto *op = value.getDefiningOp()) {
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
            arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::MinSIOp,
            arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::AndIOp,
            arith::OrIOp, arith::XOrIOp, arith::CmpIOp>(op))
      return parseBinaryArithmetic(value, analyze, builder);
    // Integer casts share width checks; distinguish unsigned extension
    // semantics.
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
            arith::IndexCastUIOp>(op)) {
      bool unsignedCast = isa<arith::ExtUIOp, arith::IndexCastUIOp>(op);
      return parseCast(value, op->getOperand(0), unsignedCast, analyze,
                       builder);
    }
  }
  if (auto op = value.getDefiningOp<arith::FPToSIOp>())
    return parseFloatToInt(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::ReduceOp>())
    return parseReduce(value, op, analyze, builder);
  // Other integer producers may still provide facts through the value itself.
  builder.setInsertionPointAfterValue(value);
  return buildIntegerInfo(value, builder);
}

// Shape and selection rules.
FailureOr<Result> CommonPtrRules::parseSplat(triton::SplatOp op,
                                             Analyze analyze,
                                             OpBuilder &builder) const {
  // Analyze the scalar source so its bindings and origin are retained.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Integer splats use the source origin for every result lane.
  if (!sourceInfo->base) {
    if (!sourceInfo->offsets.getShape().empty())
      return failure();
    OpFoldResult scalar = sourceInfo->offsets.scalarOffset;
    if (!scalar)
      scalar = op.getSrc();
    return Result{{}, buildUniformOffsetInfo(op.getResult(), builder, scalar)};
  }

  // Pointer splats broadcast the element offset and retain the scalar base.
  builder.setInsertionPoint(op);
  PtrInfo info = *sourceInfo;
  assert(info.offsets.valueType.isIndex() &&
         "scalar pointer offsets must have index type");
  auto targetType =
      cast<RankedTensorType>(addressValueType(op.getType(), builder));
  auto full = materializeFullOffset(info.offsets,
                                    addressStorageType(targetType, builder),
                                    builder, op.getLoc());
  if (failed(full))
    return failure();

  // Every new tensor axis is invariant because all pointers came from one SSA.
  info.offsets.valueType = targetType;
  info.offsets.fullOffsets = *full;
  info.offsets.axes.assign(targetType.getRank(),
                           makeInvariantAxisInfo(builder));
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseBroadcast(triton::BroadcastOp op,
                                                 Analyze analyze,
                                                 OpBuilder &builder) const {
  // Obtain the source facts before expanding its singleton dimensions.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Reuse the exact integer result and retain the source base.
  PtrInfo info = *sourceInfo;
  Value full = op.getResult();
  auto targetType = op.getType();
  if (info.base) {
    // Broadcast the pointer tensor's complete numeric offsets.
    builder.setInsertionPoint(op);
    targetType = cast<RankedTensorType>(addressValueType(targetType, builder));
    auto inputOffset = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(inputOffset))
      return failure();
    full = builder.create<triton::BroadcastOp>(
        op.getLoc(), addressStorageType(targetType, builder), *inputOffset);
  }

  // Broadcast the copied axes, then record the result type and complete value.
  broadcastAxes(info.offsets.axes, sourceInfo->offsets.getShape(),
                targetType.getShape(), makeInvariantAxisInfo(builder));
  info.offsets.valueType = targetType;
  info.offsets.fullOffsets = full;
  setUniformOrigin(info.offsets, full, builder);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseExpandDims(triton::ExpandDimsOp op,
                                                  Analyze analyze,
                                                  OpBuilder &builder) const {
  // Obtain the source facts before inserting the singleton dimension.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Reuse the exact integer result and retain the source base.
  PtrInfo info = *sourceInfo;
  Value full = op.getResult();
  auto targetType = op.getType();
  if (info.base) {
    // Insert the same dimension into the pointer tensor's numeric offsets.
    builder.setInsertionPoint(op);
    targetType = cast<RankedTensorType>(addressValueType(targetType, builder));
    auto inputOffset = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(inputOffset))
      return failure();
    full = builder.create<triton::ExpandDimsOp>(
        op.getLoc(), addressStorageType(targetType, builder), *inputOffset,
        op.getAxis());
  }

  // Insert the corresponding invariant axis into either descriptor.
  unsigned axis = op.getAxis();
  info.offsets.axes.insert(info.offsets.axes.begin() + axis,
                           makeInvariantAxisInfo(builder));
  info.offsets.valueType = targetType;
  info.offsets.fullOffsets = full;
  setUniformOrigin(info.offsets, full, builder);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseReshape(triton::ReshapeOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  // allow_reorder has no fixed lane correspondence. Normalization is a caller
  // decision; analysis must not change the source op or choose two
  // permutations.
  if (op.getAllowReorder())
    return failure();

  // Obtain the source facts under the fixed lane ordering.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Reuse the exact integer result and retain the source base.
  PtrInfo info = *sourceInfo;
  Value full = op.getResult();
  auto targetType = op.getType();
  if (info.base) {
    // Reshape the pointer tensor's offsets with the original lane ordering.
    builder.setInsertionPoint(op);
    targetType = cast<RankedTensorType>(addressValueType(targetType, builder));
    auto inputOffset = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(inputOffset))
      return failure();
    full = builder.create<triton::ReshapeOp>(
        op.getLoc(), addressStorageType(targetType, builder), *inputOffset,
        false, op.getEfficientLayout());
  }

  // Preserve uniformity and axes whose coordinates only gain or lose unit dims.
  info.offsets = reshapeOffsetInfo(full, targetType, info.offsets, builder);
  setUniformOrigin(info.offsets, full, builder);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseExtractSlice(tensor::ExtractSliceOp op,
                                                    Analyze analyze,
                                                    OpBuilder &builder) const {
  // Analyze the source; dynamic result extents still admit complete offsets.
  auto targetType = op.getType();
  auto sourceInfo = analyze(op.getSource());
  if (failed(sourceInfo))
    return failure();
  builder.setInsertionPointAfter(op);

  // Record which source dimensions disappear in a rank-reducing slice.
  auto mask = op.getDroppedDims();
  SmallVector<bool> dropped;
  for (unsigned i = 0; i < op.getSourceType().getRank(); ++i)
    dropped.push_back(mask.test(i));

  // Reuse the exact integer result and retain the source base.
  PtrInfo info = *sourceInfo;
  Value full = op.getResult();
  if (info.base) {
    // Apply the same slice to the pointer tensor's numeric offsets.
    builder.setInsertionPoint(op);
    targetType = cast<RankedTensorType>(addressValueType(targetType, builder));
    auto inputOffset = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(inputOffset))
      return failure();
    full = builder.create<tensor::ExtractSliceOp>(
        op.getLoc(),
        cast<RankedTensorType>(addressStorageType(targetType, builder)),
        *inputOffset, op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
  }

  // Update the origin and axes using the same starts, steps and dropped axes.
  auto offsets =
      sliceOffsetInfo(full, targetType, info.offsets, op.getMixedOffsets(),
                      op.getMixedStrides(), dropped, builder, op.getLoc());
  if (failed(offsets))
    return failure();
  info.offsets = std::move(*offsets);
  setUniformOrigin(info.offsets, full, builder);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseInsert(tensor::InsertOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  // Integer insertion keeps facts obtainable from the result value itself.
  if (!isOrdinaryPtrType(op.getType()))
    return buildIntegerInfo(op.getResult(), builder);

  // Analyze both pointer inputs; insertion reuses the destination's shape.
  auto source = analyze(op.getScalar()), dest = analyze(op.getDest());
  if (failed(source) || failed(dest))
    return failure();

  // Insert the source offset into the destination offsets under a common base.
  builder.setInsertionPoint(op);
  return insertPtrOffsets(
      op.getResult(), *source, *dest, op.getScalar().getType(),
      [&](Value src, Value dst) -> Value {
        return builder.create<tensor::InsertOp>(op.getLoc(), src, dst,
                                                op.getIndices());
      },
      builder);
}

FailureOr<Result> CommonPtrRules::parseInsertSlice(tensor::InsertSliceOp op,
                                                   Analyze analyze,
                                                   OpBuilder &builder) const {
  // Integer insertion keeps facts obtainable from the result value itself.
  if (!isOrdinaryPtrType(op.getType()))
    return buildIntegerInfo(op.getResult(), builder);

  // Analyze both pointer tensors; the original slice carries dynamic extents.
  auto source = analyze(op.getSource()), dest = analyze(op.getDest());
  if (failed(source) || failed(dest))
    return failure();

  // Replace the same slice in the complete offsets under a common base.
  builder.setInsertionPoint(op);
  return insertPtrOffsets(
      op.getResult(), *source, *dest, op.getSourceType(),
      [&](Value src, Value dst) -> Value {
        return builder.create<tensor::InsertSliceOp>(
            op.getLoc(), src, dst, op.getMixedOffsets(), op.getMixedSizes(),
            op.getMixedStrides());
      },
      builder);
}

FailureOr<Result> CommonPtrRules::parseTensorCast(tensor::CastOp op,
                                                  Analyze analyze,
                                                  OpBuilder &builder) const {
  // Axis facts require a known rank; verified casts preserve all coordinates.
  auto sourceType = dyn_cast<RankedTensorType>(op.getSource().getType());
  auto targetType = dyn_cast<RankedTensorType>(op.getType());
  if (!sourceType || !targetType)
    return failure();

  // Reuse the source base, origin and axis strides.
  auto sourceInfo = analyze(op.getSource());
  if (failed(sourceInfo))
    return failure();
  PtrInfo info = *sourceInfo;

  // Integer results already carry the cast. Pointer offsets need the same
  // shape cast, preserving their numeric element type.
  if (!info.base) {
    info.offsets.fullOffsets = op.getResult();
  } else if (sourceType != targetType) {
    builder.setInsertionPoint(op);
    Value full = dyn_cast<Value>(info.offsets.fullOffsets);
    if (!full)
      full = builder.create<arith::ConstantOp>(
          op.getLoc(),
          cast<TypedAttr>(cast<Attribute>(info.offsets.fullOffsets)));
    Type storageType = targetType.clone(
        cast<RankedTensorType>(full.getType()).getElementType());
    info.offsets.fullOffsets =
        builder.create<tensor::CastOp>(op.getLoc(), storageType, full)
            .getResult();
  }
  info.offsets.valueType =
      info.base ? addressValueType(targetType, builder) : targetType;

  // Fixed-size sub-axis partitions require a static extent in the result.
  for (auto [axis, size] : llvm::zip(info.offsets.axes, targetType.getShape()))
    if (ShapedType::isDynamic(size))
      axis.subAxes.clear();
  if (auto full = dyn_cast<Value>(info.offsets.fullOffsets)) {
    setUniformOrigin(info.offsets, full, builder);
  } else if (auto origin = getIntegerConstant(info.offsets.fullOffsets, 64);
             succeeded(origin)) {
    setUniformOrigin(info.offsets, op.getResult(), builder,
                     builder.getIndexAttr(origin->getSExtValue()));
  }
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseSelect(arith::SelectOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  // A scalar pointer selection can serve directly as the resulting base.
  if (isa<triton::PointerType>(op.getType()))
    return buildScalarPtrInfo(op.getResult(), builder);

  // Obtain both branch results before combining their offset facts.
  auto lhsInfo = analyze(op.getTrueValue());
  auto rhsInfo = analyze(op.getFalseValue());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();

  // Combine branch facts; return pointer or fully structured results directly.
  if (!lhsInfo->base)
    builder.setInsertionPointAfter(op);
  else
    builder.setInsertionPoint(op);
  auto result = combineSelectInfo(op, *lhsInfo, *rhsInfo, builder);
  if (failed(result) || result->base || result->offsets.hasStructuredForm())
    return result;

  // Condition invariance may refine the integer result; unavailable facts
  // leave the already known branch result intact.
  auto conditionInfo = analyze(op.getCondition());
  if (failed(conditionInfo))
    return result;
  // A tensor condition can be uniform on some axes too. Keep only equality
  // facts shared by all three inputs, then extract the original result if
  // every axis is invariant.
  bool scalarCondition = conditionInfo->offsets.getShape().empty();
  for (unsigned i = 0; i < result->offsets.axes.size(); ++i) {
    bool conditionInvariantOnAxis =
        scalarCondition ||
        conditionInfo->offsets.axes[i].kind == AxisKind::Invariant;
    if (conditionInvariantOnAxis &&
        lhsInfo->offsets.axes[i].kind == AxisKind::Invariant &&
        rhsInfo->offsets.axes[i].kind == AxisKind::Invariant)
      result->offsets.axes[i].kind = AxisKind::Invariant;
  }

  // Recover a scalar origin when every result axis is invariant.
  setUniformOrigin(result->offsets, op.getResult(), builder);
  return result;
}

// Integer rules.
Result CommonPtrRules::parseMakeRange(triton::MakeRangeOp op,
                                      OpBuilder &builder) const {
  // The verifier guarantees a nonempty 1D i32 range matching the signed bounds.
  int64_t start = op.getStartAttr().getInt();
  // Represent the range as its starting offset plus a unit-stride axis.
  OffsetInfo info;
  info.valueType = op.getType();
  info.fullOffsets = op.getResult();
  info.axes = {AxisInfo(AxisKind::Structured, builder.getIndexAttr(1))};
  info.scalarOffset = builder.getIndexAttr(start);
  return Result{{}, std::move(info)};
}

FailureOr<Result>
CommonPtrRules::parseBinaryArithmetic(Value value, Analyze analyze,
                                      OpBuilder &builder) const {
  // Obtain both operands' origins and axis facts through the current session.
  Operation *op = value.getDefiningOp();
  auto lhsInfo = analyze(op->getOperand(0));
  auto rhsInfo = analyze(op->getOperand(1));
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  const auto &lhs = *lhsInfo, &rhs = *rhsInfo;
  builder.setInsertionPointAfter(op);

  // Add or subtract the origins and corresponding axis contributions.
  // Preserve the legacy no-overflow assumption for index descriptors. Dynamic
  // origins and strides do not need a separate propagated range proof.
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    auto offsets = addOffsetInfo(value, lhs.offsets, rhs.offsets,
                                 isa<arith::SubIOp>(op), builder, op->getLoc());
    if (failed(offsets))
      return failure();
    return Result{{}, std::move(*offsets)};
  }

  // A uniform multiplier scales the other operand's origin and axis strides.
  if (isa<arith::MulIOp>(op)) {
    bool lhsUniform = lhs.offsets.isUniform();
    if (lhsUniform || rhs.offsets.isUniform()) {
      const auto &varying = lhsUniform ? rhs.offsets : lhs.offsets;
      const auto &scale = lhsUniform ? lhs.offsets : rhs.offsets;
      auto offsets = scaleOffsetInfo(value, varying, scale.scalarOffset,
                                     builder, op->getLoc());
      if (failed(offsets))
        return failure();
      return Result{{}, std::move(*offsets)};
    }
  }

  // Other cases retain the complete value and common invariant axes.
  auto fallback = combineElementwise(value, lhs, rhs, builder);
  if (failed(fallback))
    return failure();

  // Division and remainder may recover a stronger decomposition.
  if (isa<arith::DivSIOp, arith::DivUIOp>(op))
    return refineDivRem(value, lhs, rhs, *fallback, true, builder);
  if (isa<arith::RemSIOp, arith::RemUIOp>(op))
    return refineDivRem(value, lhs, rhs, *fallback, false, builder);
  return fallback;
}

FailureOr<Result> CommonPtrRules::parseCast(Value value, Value input,
                                            bool unsignedCast, Analyze analyze,
                                            OpBuilder &builder) const {
  // Obtain the source facts and compare the integer widths of the cast.
  auto sourceInfo = analyze(input);
  if (failed(sourceInfo))
    return failure();
  const auto &sourceOffsets = sourceInfo->offsets;
  builder.setInsertionPointAfterValue(value);
  auto targetWidth = getIntegerWidth(value.getType());
  auto sourceWidth = getIntegerWidth(sourceOffsets.valueType);
  if (failed(targetWidth) || failed(sourceWidth))
    return failure();

  // Check whether every represented lane keeps the same mathematical value.
  bool preservesValue = *targetWidth == *sourceWidth ||
                        (*targetWidth > *sourceWidth && !unsignedCast);
  if (!preservesValue) {
    auto bounds = constantDecompositionRange(sourceOffsets, *sourceWidth);
    preservesValue =
        bounds && (*targetWidth > *sourceWidth
                       ? bounds->isAllNonNegative()
                       : bounds->getSignedMin().isSignedIntN(*targetWidth) &&
                             bounds->getSignedMax().isSignedIntN(*targetWidth));
  }
  if (preservesValue) {
    // Signed widening keeps index descriptors. Other casts need the local
    // check above because they can change the value represented by a lane.
    auto offsets = sourceOffsets;
    offsets.valueType = value.getType();
    offsets.fullOffsets = value;
    setUniformOrigin(offsets, value, builder);
    return Result{{}, std::move(offsets)};
  }

  // Otherwise retain the exact cast result and only the invariant axis facts.
  auto offsets = buildOffsetInfo(value, builder);
  if (failed(offsets))
    return failure();
  for (unsigned i = 0; i < sourceOffsets.axes.size(); ++i)
    if (sourceOffsets.axes[i].kind == AxisKind::Invariant)
      offsets->axes[i].kind = AxisKind::Invariant;
  setUniformOrigin(*offsets, value, builder);
  return Result{{}, std::move(*offsets)};
}

FailureOr<Result> CommonPtrRules::parseFloatToInt(arith::FPToSIOp op,
                                                  Analyze analyze,
                                                  OpBuilder &builder) const {
  // First use any constant or uniform facts available from the integer result.
  auto info = buildIntegerInfo(op.getResult(), builder);
  if (failed(info) || info->offsets.getShape().empty())
    return info;

  // Floating arithmetic contributes axis equality, without integer strides.
  llvm::DenseMap<Value, SmallVector<bool>> invarianceCache;
  auto invariantAxes = inferInvariantAxes(op.getIn(), analyze, invarianceCache);
  for (unsigned i = 0; i < invariantAxes.size(); ++i)
    if (invariantAxes[i])
      info->offsets.axes[i].kind = AxisKind::Invariant;
  // Extract the integer result, preserving the actual conversion semantics.
  setUniformOrigin(info->offsets, op.getResult(), builder);
  return info;
}

FailureOr<Result> CommonPtrRules::parseReduce(Value value, triton::ReduceOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  // Seed result facts; tensor axes require a supported arithmetic combiner.
  auto info = buildIntegerInfo(value, builder);
  if (failed(info) || info->offsets.getShape().empty() ||
      !hasInvariancePreservingCombiner(op))
    return info;

  llvm::DenseMap<Value, SmallVector<bool>> invarianceCache;
  SmallVector<bool> invariantAxes(info->offsets.axes.size(), true);
  // Results can depend on every input (e.g. argmin). Intersect input facts,
  // then remove the reduced dimension; never copy a pre-reduction stride.
  for (Value operand : op.getSrcs()) {
    auto sourceInvariantAxes =
        inferInvariantAxes(operand, analyze, invarianceCache);
    sourceInvariantAxes.erase(sourceInvariantAxes.begin() + op.getAxis());
    for (unsigned i = 0; i < invariantAxes.size(); ++i)
      invariantAxes[i] = invariantAxes[i] && sourceInvariantAxes[i];
  }

  // Keep common invariant axes and singleton dimensions, then recover an
  // origin.
  for (unsigned i = 0; i < invariantAxes.size(); ++i)
    if (invariantAxes[i] || info->offsets.getShape()[i] == 1)
      info->offsets.axes[i].kind = AxisKind::Invariant;
  setUniformOrigin(info->offsets, value, builder);
  return info;
}

// Ordinary pointer rules.
FailureOr<Result> CommonPtrRules::parseAddPtr(triton::AddPtrOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  // Analyze the pointer and its integer increment independently.
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  auto sourceInfo = analyze(op.getOffset());
  if (failed(sourceInfo))
    return failure();

  // Descriptor fields are already index. The complete offset is widened
  // separately before accumulation, preserving each addptr increment.
  builder.setInsertionPoint(op);
  auto info =
      addPtrOffsets(op.getResult(), *parentInfo, sourceInfo->offsets, builder);
  if (failed(info))
    return failure();
  return Result(std::move(*info));
}

FailureOr<Result> CommonPtrRules::parseBitcast(triton::BitcastOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  // A scalar cast is already an exact base, including across element widths.
  // Do not reinterpret offsets accumulated before that scalar boundary.
  if (isa<triton::PointerType>(op.getType()))
    return buildScalarPtrInfo(op.getResult(), builder);

  // Reusing offsets requires identical shapes and encodings. The verifier
  // permits compatible dynamic shapes and an encoding on only one side.
  auto sourceType = cast<RankedTensorType>(op.getSrc().getType());
  auto targetType = cast<RankedTensorType>(op.getType());
  if (sourceType.getShape() != targetType.getShape() ||
      sourceType.getEncoding() != targetType.getEncoding())
    return failure();
  // The verifier guarantees pointer elements with the same address space.
  auto sourcePtrType = cast<triton::PointerType>(sourceType.getElementType());
  auto targetPtrType = cast<triton::PointerType>(targetType.getElementType());

  // Restrict element-unit conversion to the supported storage widths.
  auto storageWidth = [](Type type) -> unsigned {
    if (!isa<IntegerType, FloatType>(type))
      return 0;
    unsigned width = type.getIntOrFloatBitWidth();
    return width == 8 || width == 16 || width == 32 || width == 64 ? width : 0;
  };
  unsigned sourceElementWidth = storageWidth(sourcePtrType.getPointeeType());
  unsigned targetElementWidth = storageWidth(targetPtrType.getPointeeType());
  if (!sourceElementWidth || !targetElementWidth)
    return failure();

  // Recover the scalar base and element offsets of the source pointer tensor.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  assert(!sourceInfo->block && "a pointer tensor cannot carry block metadata");
  PtrInfo info = *sourceInfo;
  builder.setInsertionPoint(op);

  // Equal-width casts reuse offsets; other casts must change their units.
  if (sourceElementWidth != targetElementWidth) {
    // Rescaling uses a dense factor tensor and therefore needs static extents.
    if (!targetType.hasStaticShape())
      return failure();
    uint64_t ratio = std::max(sourceElementWidth, targetElementWidth) /
                     std::min(sourceElementWidth, targetElementWidth);
    bool widenElement = sourceElementWidth < targetElementWidth;
    // Widening needs exact division: offset_new * sizeof(new) must equal
    // offset_old * sizeof(old). Test the complete offset, never round terms.
    if (widenElement && !canDivideDecomposition(info.offsets, ratio) &&
        !isKnownMultiple(info.offsets.fullOffsets, ratio))
      return failure();

    // Materialize the complete offset and the element-size conversion factor.
    auto full = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(full))
      return failure();
    auto fullType = cast<RankedTensorType>((*full).getType());
    Value factor = builder.create<arith::ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get(fullType, int64_t(ratio)));
    if (widenElement) {
      // Divide the full offset and retain exactly divisible descriptor fields.
      info.offsets.fullOffsets =
          builder.create<arith::DivSIOp>(op.getLoc(), *full, factor)
              .getResult();
      divideDecomposition(info.offsets, ratio, builder, op.getLoc());
    } else {
      // Smaller elements scale up both the complete offset and its descriptor.
      Value scaled = builder.create<arith::MulIOp>(op.getLoc(), *full, factor);
      auto offsets =
          scaleOffsetInfo(scaled, info.offsets, builder.getIndexAttr(ratio),
                          builder, op.getLoc());
      if (failed(offsets))
        return failure();
      info.offsets = std::move(*offsets);
    }
  }

  // Express the scalar base in the destination pointer type.
  if (info.base.getType() != targetPtrType)
    info.base = builder.create<triton::BitcastOp>(op.getLoc(), targetPtrType,
                                                  info.base);
  return Result(std::move(info));
}

// Block pointer rules.
FailureOr<Result> CommonPtrRules::parseMakeTensorPtr(triton::MakeTensorPtrOp op,
                                                     Analyze analyze,
                                                     OpBuilder &builder) const {
  // Start from an ordinary scalar pointer with an index-typed base offset.
  auto baseInfo = analyze(op.getBase());
  if (failed(baseInfo))
    return failure();
  if (baseInfo->block || !baseInfo->offsets.valueType.isIndex())
    return failure();

  // Resolve each starting coordinate to its uniform index origin.
  SmallVector<OpFoldResult> coordinates;
  for (Value coordinate : op.getOffsets()) {
    auto sourceInfo = analyze(coordinate);
    if (failed(sourceInfo))
      return failure();
    if (!sourceInfo->offsets.isUniform())
      return failure();
    coordinates.push_back(sourceInfo->offsets.scalarOffset);
  }

  // Combine the base, coordinates and strides into the block descriptor.
  builder.setInsertionPoint(op);
  return makeBlockInfo(op, *baseInfo, coordinates, builder);
}

FailureOr<Result> CommonPtrRules::parseAdvance(triton::AdvanceOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  // An advance needs the existing block coordinates and strides.
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  if (!parentInfo->block)
    return failure();

  // Resolve each coordinate increment to its uniform index origin.
  SmallVector<OpFoldResult> coordinateDeltas;
  for (Value delta : op.getOffsets()) {
    auto deltaInfo = analyze(delta);
    if (failed(deltaInfo))
      return failure();
    if (!deltaInfo->offsets.isUniform())
      return failure();
    coordinateDeltas.push_back(deltaInfo->offsets.scalarOffset);
  }

  // Update the coordinates and add their stride-weighted offset contribution.
  builder.setInsertionPoint(op);
  return advanceBlockInfo(op, *parentInfo, coordinateDeltas, builder);
}

} // namespace mlir::triton::ptr
