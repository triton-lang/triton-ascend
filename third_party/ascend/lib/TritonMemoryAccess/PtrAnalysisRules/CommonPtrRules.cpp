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

namespace mlir::triton::ptr {
using namespace detail;
using Result = CommonPtrResult;

namespace {
// Index arithmetic describes the signed source value. A narrower source op
// must not wrap before its result is interpreted as an address component.
bool canCombineInIndex(BinaryKind kind, const llvm::ConstantRange &lhs,
                       const llvm::ConstantRange &rhs,
                       AnalysisOptions options) {
  unsigned width = lhs.getBitWidth();
  if (!options.indexBitWidth || width != rhs.getBitWidth() ||
      width > *options.indexBitWidth || width > 64 || lhs.isEmptySet() ||
      rhs.isEmptySet())
    return false;
  if (width == *options.indexBitWidth)
    return true;
  __int128 l0 = lhs.getSignedMin().getSExtValue();
  __int128 l1 = lhs.getSignedMax().getSExtValue();
  __int128 r0 = rhs.getSignedMin().getSExtValue();
  __int128 r1 = rhs.getSignedMax().getSExtValue();
  __int128 low, high;
  if (kind == BinaryKind::Add) {
    low = l0 + r0;
    high = l1 + r1;
  } else if (kind == BinaryKind::Subtract) {
    low = l0 - r1;
    high = l1 - r0;
  } else {
    low = std::min({l0 * r0, l0 * r1, l1 * r0, l1 * r1});
    high = std::max({l0 * r0, l0 * r1, l1 * r0, l1 * r1});
  }
  __int128 limit = __int128(1) << (width - 1);
  return low >= -limit && high < limit;
}

void setUniformOrigin(OffsetInfo &info, Value value, AnalysisOptions options,
                      OpBuilder &builder) {
  auto width = getIntegerWidth(value.getType(), options);
  if (info.scalarOffset || failed(width) || !options.indexBitWidth ||
      *width > *options.indexBitWidth ||
      !llvm::all_of(info.axes, [](const AxisInfo &axis) {
        return axis.kind == AxisKind::Invariant;
      }))
    return;
  // Read the result itself: in particular, do not re-execute div/rem on a
  // scalar or replace a possibly poisonous source operation by a constant.
  builder.setInsertionPointAfterValue(value);
  Value scalar = value;
  if (isa<RankedTensorType>(value.getType())) {
    SmallVector<Value> indices;
    for (unsigned i = 0; i < info.shape.size(); ++i)
      indices.push_back(
          builder.create<arith::ConstantIndexOp>(value.getLoc(), 0));
    scalar = builder.create<tensor::ExtractOp>(value.getLoc(), value, indices);
  }
  auto origin =
      foldOrCreateCast(scalar, builder.getIndexType(), CastKind::Signed,
                       options, builder, value.getLoc());
  if (succeeded(origin))
    info.scalarOffset = *origin;
}

Type addressValueType(Type source, OpBuilder &builder) {
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    return RankedTensorType::get(tensor.getShape(), builder.getIndexType(),
                                 tensor.getEncoding());
  return builder.getIndexType();
}
OffsetInfo invariantOffset(Value value, OpFoldResult scalar,
                           OpBuilder &builder) {
  OffsetInfo info;
  info.valueType = value.getType();
  info.shape = getLogicalShape(value.getType());
  info.fullOffsets = value;
  info.scalarOffset = scalar;
  info.axes.assign(info.shape.size(),
                   AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)));
  return info;
}
FailureOr<Value> asTensorOffset(OpFoldResult value, Type from,
                                RankedTensorType target, OpBuilder &builder,
                                Location loc) {
  if (isa<RankedTensorType>(from)) {
    if (from != target)
      return failure();
    return materialize(value, target, builder, loc);
  }
  auto scalar = materialize(value, target.getElementType(), builder, loc);
  if (failed(scalar))
    return failure();
  return splatTo(*scalar, target, builder, loc);
}
FailureOr<OpFoldResult> addScaledCoordinates(OpFoldResult origin,
                                             ArrayRef<OpFoldResult> coordinates,
                                             ArrayRef<OpFoldResult> strides,
                                             OpBuilder &builder, Location loc) {
  detail::ConcreteArithmetic arithmetic{
      builder.getIndexType(), {}, builder, loc};
  return addScaledCoordinateTerms(origin, coordinates, strides, arithmetic);
}

FailureOr<Result> ptrLeaf(Value value, OpBuilder &builder) {
  if (!isa<triton::PointerType>(value.getType()) ||
      !isOrdinaryPtrType(value.getType()))
    return Result{};
  PtrInfo info;
  info.base = value;
  info.offsets.valueType = builder.getIndexType();
  info.offsets.fullOffsets = builder.getIndexAttr(0);
  info.offsets.scalarOffset = builder.getIndexAttr(0);
  return Result(std::move(info));
}

FailureOr<Result> rangeInfo(triton::MakeRangeOp op, AnalysisOptions options,
                            OpBuilder &builder) {
  auto width = getIntegerWidth(op.getType(), options);
  if (failed(width) || *width != 32)
    return Result{};
  auto info = makeOpaqueOffset(op.getResult(), builder, options);
  if (failed(info) || info->shape.size() != 1 || info->shape[0] <= 0)
    return Result{};
  int64_t start = op.getStart(), end = op.getEnd();
  if (end <= start || end - start != info->shape[0])
    return Result{};
  if (options.indexBitWidth && *options.indexBitWidth >= *width) {
    info->axes = {AxisInfo(AxisKind::Structured, builder.getIndexAttr(1))};
    info->scalarOffset = builder.getIndexAttr(start);
  }
  return Result(
      IntegerInfo{std::move(*info), llvm::ConstantRange(llvm::APInt(32, start),
                                                        llvm::APInt(32, end))});
}

Type fieldType(OpFoldResult field) {
  if (auto value = dyn_cast<Value>(field))
    return value.getType();
  return cast<TypedAttr>(cast<Attribute>(field)).getType();
}
SmallVector<OpFoldResult> exactTerms(const OffsetInfo &info) {
  if (info.fullOffsets)
    return {info.fullOffsets};
  if (!info.addressTerms.empty())
    return info.addressTerms;
  if (info.isUniform())
    return {*info.scalarOffset};
  return {};
}
bool hasIndexTensorTerm(const OffsetInfo &info) {
  return llvm::any_of(exactTerms(info), [](OpFoldResult term) {
    auto type = dyn_cast<RankedTensorType>(fieldType(term));
    return type && type.getElementType().isIndex();
  });
}
// Apply a shape operation to tensor terms at their existing storage width.
// Scalar terms remain broadcast and need no splat helper.
LogicalResult reshapeTerms(OffsetInfo &info,
                           llvm::function_ref<Value(Value)> transform,
                           OpBuilder &builder, Location loc) {
  auto terms = exactTerms(info);
  if (terms.empty())
    return failure();
  for (auto &term : terms) {
    auto type = dyn_cast<RankedTensorType>(fieldType(term));
    if (!type)
      continue;
    auto value = materialize(term, type, builder, loc);
    if (failed(value))
      return failure();
    term = transform(*value);
  }
  info.fullOffsets = {};
  info.addressTerms = std::move(terms);
  return success();
}

FailureOr<Result> selectInfo(arith::SelectOp op, const Result &lhsInfo,
                             const Result &rhsInfo, AnalysisOptions options,
                             OpBuilder &builder) {
  Value value = op.getResult();
  if (auto *lhs = getIntegerInfo(lhsInfo)) {
    auto &rhs = std::get<IntegerInfo>(rhsInfo);
    auto info = makeOpaqueOffset(value, builder, options);
    if (failed(info))
      return Result{};
    if (op.getTrueValue() == op.getFalseValue()) {
      *info = lhs->offsets;
      info->fullOffsets = value;
    } else if (op.getCondition().getType().isInteger(1) &&
               lhs->offsets.hasStructuredForm() &&
               rhs.offsets.hasStructuredForm()) {
      Type scalar = builder.getIndexType();
      auto choose = [&](OpFoldResult l,
                        OpFoldResult r) -> FailureOr<OpFoldResult> {
        if (l == r)
          return l;
        auto left = materialize(l, scalar, builder, value.getLoc()),
             right = materialize(r, scalar, builder, value.getLoc());
        if (failed(left) || failed(right))
          return failure();
        return OpFoldResult(builder
                                .create<arith::SelectOp>(value.getLoc(),
                                                         op.getCondition(),
                                                         *left, *right)
                                .getResult());
      };
      auto origin =
          choose(*lhs->offsets.scalarOffset, *rhs.offsets.scalarOffset);
      if (failed(origin))
        return Result{};
      info->scalarOffset = *origin;
      for (unsigned i = 0; i < info->axes.size(); ++i) {
        auto stride =
            choose(lhs->offsets.axes[i].stride, rhs.offsets.axes[i].stride);
        if (failed(stride))
          return Result{};
        info->axes[i] =
            AxisInfo(lhs->offsets.axes[i].kind == AxisKind::Invariant &&
                             rhs.offsets.axes[i].kind == AxisKind::Invariant
                         ? AxisKind::Invariant
                         : AxisKind::Structured,
                     *stride);
      }
    }
    return Result(
        IntegerInfo{std::move(*info), lhs->range.unionWith(rhs.range)});
  }
  const auto &lhs = std::get<PtrInfo>(lhsInfo),
             &rhs = std::get<PtrInfo>(rhsInfo);
  if (lhs.base != rhs.base)
    return Result{};
  auto left = exactTerms(lhs.offsets), right = exactTerms(rhs.offsets);
  // Nonuniform affine-only bindings have no original tensor to select. They
  // remain a consumer fallback rather than forcing a wide materialization.
  if (left.empty() || right.empty())
    return Result{};
  auto isZero = [](OpFoldResult term) {
    auto zero = getIntegerConstant(term, 64);
    return succeeded(zero) && zero->isZero();
  };
  if (isa<RankedTensorType>(op.getCondition().getType()))
    for (auto terms :
         {ArrayRef<OpFoldResult>(left), ArrayRef<OpFoldResult>(right)})
      for (OpFoldResult term : terms)
        // A tensor predicate would require broadcasting this scalar address
        // into a new wide tensor. Leave that case to the consumer fallback.
        if ((fieldType(term).isIndex() || fieldType(term).isInteger(64)) &&
            !isZero(term))
          return Result{};
  auto target =
      cast<RankedTensorType>(addressValueType(value.getType(), builder));
  PtrInfo info = lhs;
  info.offsets.valueType = target;
  info.offsets.shape = llvm::to_vector(target.getShape());
  info.offsets.fullOffsets = {};
  info.offsets.addressTerms.clear();
  info.offsets.scalarOffset.reset();
  info.offsets.axes.assign(
      target.getRank(), AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  auto append = [&](ArrayRef<OpFoldResult> terms, bool yes) -> LogicalResult {
    for (auto term : terms) {
      if (isZero(term))
        continue;
      Type type = fieldType(term);
      Type termType = !isa<RankedTensorType>(type) &&
                              op.getCondition().getType().isInteger(1)
                          ? type
                          : Type(target.clone(getIntegerElementType(type)));
      auto tensor =
          isa<RankedTensorType>(termType)
              ? asTensorOffset(term, type, cast<RankedTensorType>(termType),
                               builder, op.getLoc())
              : materialize(term, type, builder, op.getLoc());
      if (failed(tensor))
        return failure();
      auto zero = builder.getZeroAttr(termType);
      Value zeroValue =
          builder.create<arith::ConstantOp>(op.getLoc(), termType, zero);
      Value chosen = builder.create<arith::SelectOp>(
          op.getLoc(), op.getCondition(), yes ? *tensor : zeroValue,
          yes ? zeroValue : *tensor);
      info.offsets.addressTerms.push_back(chosen);
    }
    return success();
  };
  if (failed(append(left, true)) || failed(append(right, false)))
    return Result{};
  if (info.offsets.addressTerms.empty()) {
    info.offsets.addressTerms.push_back(builder.getIndexAttr(0));
    info.offsets.scalarOffset = builder.getIndexAttr(0);
    info.offsets.axes.assign(
        target.getRank(),
        AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)));
  }
  return Result(std::move(info));
}

FailureOr<PtrInfo> addPtrOffsets(Value value, const PtrInfo &parent,
                                 const OffsetInfo &delta,
                                 AnalysisOptions options, OpBuilder &builder) {
  if (parent.getElementType() !=
      cast<triton::PointerType>(getIntegerElementType(value.getType()))
          .getPointeeType())
    return failure();
  Type type = addressValueType(value.getType(), builder);
  OffsetInfo lhs = parent.offsets, rhs = delta;
  OpFoldResult complete;
  SmallVector<OpFoldResult> terms;
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    auto left = exactTerms(lhs), right = exactTerms(rhs);
    if (left.empty() || right.empty())
      return failure();
    llvm::append_range(terms, left);
    llvm::append_range(terms, right);
    for (auto *info : {&lhs, &rhs}) {
      if (!isa<RankedTensorType>(info->valueType))
        info->axes.assign(tensor.getRank(), AxisInfo(AxisKind::Invariant,
                                                     builder.getIndexAttr(0)));
      info->valueType = type;
      info->shape = llvm::to_vector(tensor.getShape());
    }
  } else {
    auto left = materializeOffsetAt(lhs, {}, builder, value.getLoc(), options);
    auto right = materializeOffsetAt(rhs, {}, builder, value.getLoc(), options);
    if (failed(left) || failed(right))
      return failure();
    lhs.scalarOffset = *left;
    rhs.scalarOffset = *right;
    lhs.valueType = rhs.valueType = type;
    auto sum = foldOrCreateBinary(BinaryKind::Add, *left, *right, type, options,
                                  builder, value.getLoc());
    if (failed(sum))
      return failure();
    complete = *sum;
  }
  auto offsets =
      addOffsets(complete, lhs, rhs, false, options, builder, value.getLoc());
  if (failed(offsets))
    return failure();
  offsets->addressTerms = std::move(terms);
  PtrInfo info = parent;
  info.offsets = std::move(*offsets);
  return info;
}

FailureOr<Result> makeBlockInfo(triton::MakeTensorPtrOp op, const PtrInfo &base,
                                ArrayRef<OpFoldResult> coordinates,
                                AnalysisOptions options, OpBuilder &builder) {
  PtrInfo ptr = base;
  auto baseOffset =
      materializeOffsetAt(base.offsets, {}, builder, op.getLoc(), options);
  if (failed(baseOffset))
    return failure();
  auto tensorType = cast<RankedTensorType>(
      cast<triton::PointerType>(op.getType()).getPointeeType());
  auto &offsets = ptr.offsets;
  offsets.valueType = RankedTensorType::get(
      tensorType.getShape(), builder.getIndexType(), tensorType.getEncoding());
  offsets.shape = llvm::to_vector(tensorType.getShape());
  offsets.fullOffsets = {};
  offsets.addressTerms.clear();
  offsets.axes.clear();
  for (Value stride : op.getStrides()) {
    auto constant = detail::getIntegerConstant(stride, 64);
    auto indexStride =
        foldOrCreateCast(stride, builder.getIndexType(), CastKind::Signed,
                         options, builder, op.getLoc());
    if (failed(indexStride))
      return Result{};
    offsets.axes.emplace_back(succeeded(constant) && constant->isZero()
                                  ? AxisKind::Invariant
                                  : AxisKind::Structured,
                              *indexStride);
  }
  auto origin =
      addScaledCoordinates(*baseOffset, coordinates,
                           detail::axisStrides(offsets), builder, op.getLoc());
  if (failed(origin))
    return Result{};
  offsets.scalarOffset = *origin;
  ptr.block.emplace();
  for (Value shape : op.getShape()) {
    auto indexShape =
        foldOrCreateCast(shape, builder.getIndexType(), CastKind::Signed,
                         options, builder, op.getLoc());
    if (failed(indexShape))
      return Result{};
    ptr.block->parentShape.push_back(*indexShape);
  }
  ptr.block->coordinateOffsets.assign(coordinates.begin(), coordinates.end());
  ptr.block->order = llvm::to_vector(op.getOrderAttr().asArrayRef());
  if (!detail::validPtrBinding(op.getResult(), ptr, options))
    return failure();
  return Result(std::move(ptr));
}

FailureOr<Result> advanceBlockInfo(triton::AdvanceOp op, const PtrInfo &parent,
                                   ArrayRef<OpFoldResult> delta,
                                   AnalysisOptions options,
                                   OpBuilder &builder) {
  PtrInfo ptr = parent;
  detail::ConcreteArithmetic arithmetic{builder.getIndexType(), options,
                                        builder, op.getLoc()};
  auto updated = advanceCoordinateTerms<OpFoldResult>(
      ptr.block->coordinateOffsets, delta, arithmetic);
  if (failed(updated))
    return Result{};
  // Add only this advance, retaining the existing scalar base contribution.
  auto origin = addScaledCoordinates(*ptr.offsets.scalarOffset, delta,
                                     detail::axisStrides(ptr.offsets), builder,
                                     op.getLoc());
  if (failed(origin))
    return Result{};
  ptr.offsets.scalarOffset = *origin;
  ptr.offsets.fullOffsets = {};
  ptr.offsets.addressTerms.clear();
  ptr.block->coordinateOffsets = std::move(*updated);
  if (!detail::validPtrBinding(op.getResult(), ptr, options))
    return failure();
  return Result(std::move(ptr));
}

FailureOr<Result> integerLeaf(Value value, AnalysisOptions options,
                              OpBuilder &builder) {
  auto width = getIntegerWidth(value.getType(), options);
  if (failed(width))
    return Result{};
  auto info = makeOpaqueOffset(value, builder, options);
  if (failed(info))
    return Result{};
  auto constant = getIntegerConstant(value, *width);
  if (succeeded(constant) && options.indexBitWidth &&
      *width <= *options.indexBitWidth)
    *info = invariantOffset(
        value,
        scalarIntegerAttr(builder.getIndexType(),
                          constant->sextOrTrunc(*options.indexBitWidth),
                          builder),
        builder);
  return Result(IntegerInfo{std::move(*info), getValueRange(value, *width)});
}

} // namespace

FailureOr<Result> CommonPtrRules::parseReshape(triton::ReshapeOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  // allow_reorder has no fixed lane correspondence. Normalization is a caller
  // decision; analysis must not change the source op or choose two
  // permutations.
  if (op.getAllowReorder())
    return Result{};
  auto input = analyze(op.getSrc());
  if (failed(input))
    return failure();
  if (std::holds_alternative<std::monostate>(*input))
    return Result{};
  builder.setInsertionPointAfter(op);
  if (auto *integer = getIntegerInfo(*input)) {
    auto offsets =
        reshapeOffsets(op.getResult(), op.getType(), integer->offsets, options,
                       builder, op.getLoc());
    if (failed(offsets))
      return Result{};
    Result result = IntegerInfo{std::move(*offsets), integer->range};
    return result;
  }
  PtrInfo info = *getPtrInfo(*input);
  // Triton shape ops cannot consume index tensors supplied by a binding.
  if (hasIndexTensorTerm(info.offsets))
    return Result{};
  builder.setInsertionPoint(op);
  auto type = cast<RankedTensorType>(addressValueType(op.getType(), builder));
  if (failed(reshapeTerms(
          info.offsets,
          [&](Value term) -> Value {
            auto target = type.clone(getIntegerElementType(term.getType()));
            return builder.create<triton::ReshapeOp>(
                op.getLoc(), target, term, false, op.getEfficientLayout());
          },
          builder, op.getLoc())))
    return Result{};
  auto offsets =
      reshapeOffsets({}, type, info.offsets, options, builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  info.offsets = std::move(*offsets);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseExtractSlice(tensor::ExtractSliceOp op,
                                                    Analyze analyze,
                                                    OpBuilder &builder) const {
  auto target = op.getType();
  if (!target.hasStaticShape())
    return Result{};
  auto input = analyze(op.getSource());
  if (failed(input))
    return failure();
  if (std::holds_alternative<std::monostate>(*input))
    return Result{};
  builder.setInsertionPointAfter(op);
  auto mask = op.getDroppedDims();
  SmallVector<bool> dropped;
  for (unsigned i = 0; i < op.getSourceType().getRank(); ++i)
    dropped.push_back(mask.test(i));
  if (auto *integer = getIntegerInfo(*input)) {
    auto offsets = sliceOffsets(op.getResult(), target, integer->offsets,
                                op.getMixedOffsets(), op.getMixedStrides(),
                                dropped, options, builder, op.getLoc());
    if (failed(offsets))
      return Result{};
    Result result = IntegerInfo{std::move(*offsets), integer->range};
    return result;
  }
  builder.setInsertionPoint(op);
  PtrInfo info = *getPtrInfo(*input);
  auto type = cast<RankedTensorType>(addressValueType(target, builder));
  if (failed(reshapeTerms(
          info.offsets,
          [&](Value term) -> Value {
            auto target = type.clone(getIntegerElementType(term.getType()));
            return builder.create<tensor::ExtractSliceOp>(
                op.getLoc(), target, term, op.getMixedOffsets(),
                op.getMixedSizes(), op.getMixedStrides());
          },
          builder, op.getLoc())))
    return Result{};
  auto offsets = sliceOffsets({}, type, info.offsets, op.getMixedOffsets(),
                              op.getMixedStrides(), dropped, options, builder,
                              op.getLoc());
  if (failed(offsets))
    return Result{};
  info.offsets = std::move(*offsets);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseTensorCast(tensor::CastOp op,
                                                  Analyze analyze,
                                                  OpBuilder &builder) const {
  auto from = dyn_cast<RankedTensorType>(op.getSource().getType());
  auto to = dyn_cast<RankedTensorType>(op.getType());
  if (!from || !to || !from.hasStaticShape() || from != to)
    return Result{};
  auto input = analyze(op.getSource());
  if (failed(input))
    return failure();
  builder.setInsertionPointAfter(op);
  if (auto *integer = getIntegerInfo(*input)) {
    std::get<IntegerInfo>(*input).offsets.fullOffsets = op.getResult();
  }
  // Identical pointer tensor types need no extra helper: offsets already have
  // the same shape/encoding, and the cast preserves every lane.
  return *input;
}

FailureOr<Result> CommonPtrRules::parseElementwise(Value value, Analyze analyze,
                                                   OpBuilder &builder) const {
  Operation *op = value.getDefiningOp();
  auto lhs = analyze(op->getOperand(0));
  auto rhs = analyze(op->getOperand(1));
  if (failed(lhs) || failed(rhs))
    return failure();
  auto *left = getIntegerInfo(*lhs), *right = getIntegerInfo(*rhs);
  if (!left || !right)
    return Result{};
  builder.setInsertionPointAfterValue(value);
  return combineElementwise(value, *left, *right, builder);
}

FailureOr<Result> CommonPtrRules::combineElementwise(Value value,
                                                     const IntegerInfo &lhs,
                                                     const IntegerInfo &rhs,
                                                     OpBuilder &builder) const {
  const auto *left = &lhs, *right = &rhs;
  Operation *op = value.getDefiningOp();
  auto offsets = makeOpaqueOffset(value, builder, options);
  if (failed(offsets) || left->offsets.shape != offsets->shape ||
      right->offsets.shape != offsets->shape)
    return Result{};
  Type scalar = builder.getIndexType();
  for (unsigned i = 0; i < offsets->axes.size(); ++i) {
    bool invariant = left->offsets.axes[i].kind == AxisKind::Invariant &&
                     right->offsets.axes[i].kind == AxisKind::Invariant;
    offsets->axes[i] =
        AxisInfo(invariant ? AxisKind::Invariant : AxisKind::Unknown,
                 builder.getIntegerAttr(scalar, 0));
  }
  setUniformOrigin(*offsets, value, options, builder);
  auto range =
      llvm::ConstantRange::getFull(*getIntegerWidth(value.getType(), options));
  const auto &l = left->range, &r = right->range;
  if (isa<arith::AddIOp>(op))
    range = l.add(r);
  else if (isa<arith::SubIOp>(op))
    range = l.sub(r);
  else if (isa<arith::MulIOp>(op))
    range = l.multiply(r);
  else if (isa<arith::AndIOp>(op))
    range = l.binaryAnd(r);
  else if (isa<arith::OrIOp>(op))
    range = l.binaryOr(r);
  else if (isa<arith::XOrIOp>(op))
    range = l.binaryXor(r);
  else if (isa<arith::MinSIOp>(op))
    range = l.smin(r);
  else if (isa<arith::MaxSIOp>(op))
    range = l.smax(r);
  else if (isa<arith::MinUIOp>(op))
    range = l.umin(r);
  else if (isa<arith::MaxUIOp>(op))
    range = l.umax(r);
  else if (isa<arith::DivUIOp>(op))
    range = l.udiv(r);
  else if (isa<arith::RemUIOp>(op))
    range = l.urem(r);
  else if (isa<arith::DivSIOp>(op))
    range = l.sdiv(r);
  else if (isa<arith::RemSIOp>(op))
    range = l.srem(r);
  // Range utilities can produce the empty set when every input is poison.
  // Keep the source SSA and use no proof in that case.
  if (range.isEmptySet())
    range = llvm::ConstantRange::getFull(range.getBitWidth());
  return Result(IntegerInfo{std::move(*offsets), std::move(range)});
}

LogicalResult CommonPtrRules::validateBinding(Value value,
                                              const Result &info) const {
  if (!value)
    return failure();
  if (auto *integer = getIntegerInfo(info)) {
    auto width = getIntegerWidth(value.getType(), options);
    return success(succeeded(width) && integer->range.getBitWidth() == *width &&
                   validOffsetBinding(value, integer->offsets, options));
  }
  if (auto *ptr = getPtrInfo(info))
    return success(validPtrBinding(value, *ptr, options));
  return failure();
}

FailureOr<Result> CommonPtrRules::parseOperation(Value value, Analyze analyze,
                                                 OpBuilder &builder) const {
  if (!value)
    return failure();
  bool integer = succeeded(detail::getIntegerWidth(value.getType(), options));
  bool ordinary = detail::isOrdinaryPtrType(value.getType());
  bool block = detail::isBlockPtrType(value.getType());
  if (!integer &&
      (!(ordinary || block) || !detail::supportsAddressAnalysis(options) ||
       options.indexBitWidth != options.addressBitWidth))
    return Result{};
  if (block) {
    if (auto op = value.getDefiningOp<triton::MakeTensorPtrOp>())
      return parseMakeTensorPtr(op, analyze, builder);
    if (auto op = value.getDefiningOp<triton::AdvanceOp>())
      return parseAdvance(op, analyze, builder);
    return Result{};
  }
  if (auto op = value.getDefiningOp<triton::SplatOp>())
    return parseSplat(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::BroadcastOp>())
    return parseBroadcast(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::ExpandDimsOp>())
    return parseExpandDims(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::ReshapeOp>())
    return parseReshape(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::ExtractSliceOp>())
    return parseExtractSlice(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::CastOp>())
    return parseTensorCast(op, analyze, builder);
  if (auto op = value.getDefiningOp<arith::SelectOp>())
    return parseSelect(op, analyze, builder);
  if (ordinary) {
    if (auto op = value.getDefiningOp<triton::AddPtrOp>())
      return parseAddPtr(op, analyze, builder);
    if (auto op = value.getDefiningOp<triton::BitcastOp>())
      return parseBitcast(op, analyze, builder);
    return ptrLeaf(value, builder);
  }
  if (auto op = value.getDefiningOp<triton::MakeRangeOp>()) {
    return rangeInfo(op, options, builder);
  }
  if (auto op = value.getDefiningOp<arith::AddIOp>())
    return parseAdd(op, analyze, builder);
  if (auto op = value.getDefiningOp<arith::SubIOp>())
    return parseSub(op, analyze, builder);
  if (auto op = value.getDefiningOp<arith::MulIOp>())
    return parseMul(op, analyze, builder);
  if (auto *op = value.getDefiningOp())
    if (isa<arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
            arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
            arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::CmpIOp>(op))
      return parseElementwise(value, analyze, builder);
  if (auto op = value.getDefiningOp<arith::ExtSIOp>())
    return parseCast(value, op.getIn(), false, analyze, builder);
  if (auto op = value.getDefiningOp<arith::ExtUIOp>())
    return parseCast(value, op.getIn(), true, analyze, builder);
  if (auto op = value.getDefiningOp<arith::TruncIOp>())
    return parseCast(value, op.getIn(), false, analyze, builder);
  if (auto op = value.getDefiningOp<arith::IndexCastOp>())
    return parseCast(value, op.getIn(), false, analyze, builder);
  if (auto op = value.getDefiningOp<arith::IndexCastUIOp>())
    return parseCast(value, op.getIn(), true, analyze, builder);
  builder.setInsertionPointAfterValue(value);
  return integerLeaf(value, options, builder);
}

FailureOr<Result> CommonPtrRules::parseAdd(arith::AddIOp op, Analyze analyze,
                                           OpBuilder &builder) const {
  auto lhsInfo = analyze(op.getLhs());
  auto rhsInfo = analyze(op.getRhs());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  auto *lhs = getIntegerInfo(*lhsInfo), *rhs = getIntegerInfo(*rhsInfo);
  if (!lhs || !rhs)
    return Result{};
  builder.setInsertionPointAfter(op);
  if (!canCombineInIndex(BinaryKind::Add, lhs->range, rhs->range, options))
    return combineElementwise(op, *lhs, *rhs, builder);
  auto offsets = addOffsets(op.getResult(), lhs->offsets, rhs->offsets, false,
                            options, builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  return Result(IntegerInfo{std::move(*offsets), lhs->range.add(rhs->range)});
}

FailureOr<Result> CommonPtrRules::parseSub(arith::SubIOp op, Analyze analyze,
                                           OpBuilder &builder) const {
  auto lhsInfo = analyze(op.getLhs());
  auto rhsInfo = analyze(op.getRhs());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  auto *lhs = getIntegerInfo(*lhsInfo), *rhs = getIntegerInfo(*rhsInfo);
  if (!lhs || !rhs)
    return Result{};
  builder.setInsertionPointAfter(op);
  if (!canCombineInIndex(BinaryKind::Subtract, lhs->range, rhs->range, options))
    return combineElementwise(op, *lhs, *rhs, builder);
  auto offsets = addOffsets(op.getResult(), lhs->offsets, rhs->offsets, true,
                            options, builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  return Result(IntegerInfo{std::move(*offsets), lhs->range.sub(rhs->range)});
}

FailureOr<Result> CommonPtrRules::parseMul(arith::MulIOp op, Analyze analyze,
                                           OpBuilder &builder) const {
  auto lhsInfo = analyze(op.getLhs());
  auto rhsInfo = analyze(op.getRhs());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  auto *lhs = getIntegerInfo(*lhsInfo), *rhs = getIntegerInfo(*rhsInfo);
  if (!lhs || !rhs)
    return Result{};
  builder.setInsertionPointAfter(op);
  bool uniform = lhs->offsets.isUniform() || rhs->offsets.isUniform();
  if (!uniform ||
      !canCombineInIndex(BinaryKind::Multiply, lhs->range, rhs->range, options))
    return combineElementwise(op, *lhs, *rhs, builder);
  const auto &varying = lhs->offsets.isUniform() ? rhs->offsets : lhs->offsets;
  const auto &scale = lhs->offsets.isUniform() ? lhs->offsets : rhs->offsets;
  auto offsets = scaleOffsets(op.getResult(), varying, *scale.scalarOffset,
                              options, builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  return Result(
      IntegerInfo{std::move(*offsets), lhs->range.multiply(rhs->range)});
}

FailureOr<Result> CommonPtrRules::parseSplat(triton::SplatOp op,
                                             Analyze analyze,
                                             OpBuilder &builder) const {
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  if (std::holds_alternative<std::monostate>(*sourceInfo))
    return Result{};
  builder.setInsertionPointAfter(op);
  if (auto *integer = getIntegerInfo(*sourceInfo)) {
    if (!integer->offsets.shape.empty())
      return Result{};
    if (!integer->offsets.scalarOffset) {
      auto offsets = makeOpaqueOffset(op, builder, options);
      if (failed(offsets))
        return Result{};
      for (auto &axis : offsets->axes)
        axis.kind = AxisKind::Invariant;
      return Result(IntegerInfo{std::move(*offsets), integer->range});
    }
    Result result =
        IntegerInfo{invariantOffset(op.getResult(),
                                    *integer->offsets.scalarOffset, builder),
                    integer->range};
    return result;
  }
  builder.setInsertionPoint(op);
  PtrInfo info = std::get<PtrInfo>(*sourceInfo);
  if (isa<RankedTensorType>(info.offsets.valueType))
    return Result{};
  auto target = cast<RankedTensorType>(addressValueType(op.getType(), builder));
  auto terms = exactTerms(info.offsets);
  if (terms.empty())
    return Result{};
  info.offsets.valueType = target;
  info.offsets.shape = llvm::to_vector(target.getShape());
  info.offsets.fullOffsets = {};
  info.offsets.addressTerms = std::move(terms);
  info.offsets.axes.assign(
      target.getRank(), AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)));
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseBroadcast(triton::BroadcastOp op,
                                                 Analyze analyze,
                                                 OpBuilder &builder) const {
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  if (std::holds_alternative<std::monostate>(*sourceInfo))
    return Result{};
  builder.setInsertionPointAfter(op);
  if (auto *integer = getIntegerInfo(*sourceInfo)) {
    auto info = broadcastOffsets(op.getResult(), op.getType(), integer->offsets,
                                 builder);
    if (failed(info))
      return Result{};
    Result result = IntegerInfo{std::move(*info), integer->range};
    return result;
  }
  PtrInfo info = std::get<PtrInfo>(*sourceInfo);
  // Triton shape ops cannot consume index tensors supplied by a binding.
  if (hasIndexTensorTerm(info.offsets))
    return Result{};
  builder.setInsertionPoint(op);
  auto target = cast<RankedTensorType>(addressValueType(op.getType(), builder));
  if (failed(reshapeTerms(
          info.offsets,
          [&](Value term) -> Value {
            auto type = target.clone(getIntegerElementType(term.getType()));
            return builder.create<triton::BroadcastOp>(op.getLoc(), type, term);
          },
          builder, op.getLoc())))
    return Result{};
  auto offsets = broadcastOffsets({}, target, info.offsets, builder);
  if (failed(offsets))
    return Result{};
  info.offsets = std::move(*offsets);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseExpandDims(triton::ExpandDimsOp op,
                                                  Analyze analyze,
                                                  OpBuilder &builder) const {
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  if (std::holds_alternative<std::monostate>(*sourceInfo))
    return Result{};
  builder.setInsertionPointAfter(op);
  if (auto *integer = getIntegerInfo(*sourceInfo)) {
    auto info = expandDimsOffsets(op.getResult(), op.getType(),
                                  integer->offsets, op.getAxis(), builder);
    if (failed(info))
      return Result{};
    Result result = IntegerInfo{std::move(*info), integer->range};
    return result;
  }
  PtrInfo info = std::get<PtrInfo>(*sourceInfo);
  // Triton shape ops cannot consume index tensors supplied by a binding.
  if (hasIndexTensorTerm(info.offsets))
    return Result{};
  builder.setInsertionPoint(op);
  auto target = cast<RankedTensorType>(addressValueType(op.getType(), builder));
  if (failed(reshapeTerms(
          info.offsets,
          [&](Value term) -> Value {
            auto type = target.clone(getIntegerElementType(term.getType()));
            return builder.create<triton::ExpandDimsOp>(op.getLoc(), type, term,
                                                        op.getAxis());
          },
          builder, op.getLoc())))
    return Result{};
  auto offsets =
      expandDimsOffsets({}, target, info.offsets, op.getAxis(), builder);
  if (failed(offsets))
    return Result{};
  info.offsets = std::move(*offsets);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseCast(Value value, Value input,
                                            bool unsignedCast, Analyze analyze,
                                            OpBuilder &builder) const {
  auto sourceInfo = analyze(input);
  if (failed(sourceInfo))
    return failure();
  auto *source = getIntegerInfo(*sourceInfo);
  if (!source)
    return Result{};
  builder.setInsertionPointAfterValue(value);
  auto width = getIntegerWidth(value.getType(), options);
  if (failed(width))
    return Result{};
  unsigned oldWidth = source->range.getBitWidth();
  bool representable = options.indexBitWidth &&
                       std::max(*width, oldWidth) <= *options.indexBitWidth;
  bool keep =
      representable && (*width == oldWidth ||
                        (*width > oldWidth &&
                         (!unsignedCast || source->range.isAllNonNegative())) ||
                        (*width < oldWidth &&
                         source->range.getSignedMin().isSignedIntN(*width) &&
                         source->range.getSignedMax().isSignedIntN(*width)));
  auto offsets = makeOpaqueOffset(value, builder, options);
  if (failed(offsets))
    return Result{};
  if (keep) {
    // Metadata already uses index; only the exact source value changes type.
    *offsets = source->offsets;
    offsets->valueType = value.getType();
    offsets->fullOffsets = value;
  } else {
    for (unsigned i = 0; i < offsets->axes.size(); ++i)
      if (source->offsets.axes[i].kind == AxisKind::Invariant)
        offsets->axes[i].kind = AxisKind::Invariant;
    setUniformOrigin(*offsets, value, options, builder);
  }
  auto range = *width < oldWidth    ? source->range.truncate(*width)
               : *width == oldWidth ? source->range
               : unsignedCast       ? source->range.zeroExtend(*width)
                                    : source->range.signExtend(*width);
  return Result(IntegerInfo{std::move(*offsets), std::move(range)});
}

FailureOr<Result> CommonPtrRules::parseBitcast(triton::BitcastOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  // A scalar cast is already an exact base, including across element widths.
  // Do not reinterpret offsets accumulated before that scalar boundary.
  if (isa<triton::PointerType>(op.getType()))
    return ptrLeaf(op.getResult(), builder);
  auto target = dyn_cast<RankedTensorType>(op.getType());
  auto source = dyn_cast<RankedTensorType>(op.getSrc().getType());
  if (!source || !target || source.getShape() != target.getShape() ||
      source.getEncoding() != target.getEncoding())
    return Result{};
  auto from = dyn_cast<triton::PointerType>(source.getElementType());
  auto to = dyn_cast<triton::PointerType>(target.getElementType());
  if (!from || !to || from.getAddressSpace() != to.getAddressSpace())
    return Result{};
  auto storageWidth = [](Type type) -> unsigned {
    if (!isa<IntegerType, FloatType>(type))
      return 0;
    unsigned width = type.getIntOrFloatBitWidth();
    return width == 8 || width == 16 || width == 32 || width == 64 ? width : 0;
  };
  unsigned width = storageWidth(from.getPointeeType());
  if (!width || width != storageWidth(to.getPointeeType()))
    return Result{};

  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  auto *parent = getPtrInfo(*sourceInfo);
  if (!parent || parent->block)
    return Result{};
  // Equal storage units preserve every exact term and decomposition. Only
  // the scalar base needs a new type; no offset tensor is cast or widened.
  PtrInfo info = *parent;
  builder.setInsertionPoint(op);
  if (info.base.getType() != to)
    info.base = builder.create<triton::BitcastOp>(op.getLoc(), to, info.base);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseSelect(arith::SelectOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  if (isa<triton::PointerType>(op.getType()))
    return ptrLeaf(op.getResult(), builder);
  auto lhsInfo = analyze(op.getTrueValue());
  auto rhsInfo = analyze(op.getFalseValue());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  if (lhsInfo->index() != rhsInfo->index() ||
      std::holds_alternative<std::monostate>(*lhsInfo))
    return Result{};
  if (getIntegerInfo(*lhsInfo))
    builder.setInsertionPointAfter(op);
  else
    builder.setInsertionPoint(op);
  return selectInfo(op, *lhsInfo, *rhsInfo, options, builder);
}

FailureOr<Result> CommonPtrRules::parseAddPtr(triton::AddPtrOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  auto *parent = getPtrInfo(*parentInfo);
  if (!parent)
    return Result{};
  auto sourceInfo = analyze(op.getOffset());
  if (failed(sourceInfo))
    return failure();
  auto *source = getIntegerInfo(*sourceInfo);
  if (!source)
    return Result{};
  auto sourceWidth = getIntegerWidth(source->offsets.valueType, options);
  if (failed(sourceWidth) || *sourceWidth > *options.indexBitWidth)
    return Result{};
  // The integer result already carries index descriptor fields. Keep its
  // original exact term so consecutive addptr operations sign-extend
  // separately.
  builder.setInsertionPoint(op);
  auto info =
      addPtrOffsets(op.getResult(), *parent, source->offsets, options, builder);
  if (failed(info))
    return Result{};
  return Result(std::move(*info));
}

FailureOr<Result> CommonPtrRules::parseMakeTensorPtr(triton::MakeTensorPtrOp op,
                                                     Analyze analyze,
                                                     OpBuilder &builder) const {
  auto baseInfo = analyze(op.getBase());
  if (failed(baseInfo))
    return failure();
  auto *base = getPtrInfo(*baseInfo);
  if (!base || base->block || !base->offsets.valueType.isIndex())
    return Result{};
  SmallVector<OpFoldResult> coordinates;
  for (Value coordinate : op.getOffsets()) {
    auto sourceInfo = analyze(coordinate);
    if (failed(sourceInfo))
      return failure();
    auto *source = getIntegerInfo(*sourceInfo);
    if (!source || !source->offsets.isUniform())
      return Result{};
    coordinates.push_back(*source->offsets.scalarOffset);
  }
  builder.setInsertionPoint(op);
  return makeBlockInfo(op, *base, coordinates, options, builder);
}

FailureOr<Result> CommonPtrRules::parseAdvance(triton::AdvanceOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  auto *parent = getPtrInfo(*parentInfo);
  if (!parent || !parent->block)
    return Result{};
  SmallVector<OpFoldResult> coordinates;
  for (Value coordinate : op.getOffsets()) {
    auto sourceInfo = analyze(coordinate);
    if (failed(sourceInfo))
      return failure();
    auto *source = getIntegerInfo(*sourceInfo);
    if (!source || !source->offsets.isUniform())
      return Result{};
    coordinates.push_back(*source->offsets.scalarOffset);
  }
  builder.setInsertionPoint(op);
  return advanceBlockInfo(op, *parent, coordinates, options, builder);
}

} // namespace mlir::triton::ptr
