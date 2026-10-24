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

#include "TritonMemoryAccess/PointerAnalysisTransfer.h"
#include "PointerAnalysisInternal.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <algorithm>
#include <limits>

namespace mlir::triton::pointer::detail {
namespace {

bool isIntegerValue(Value value, AnalysisOptions options) {
  return succeeded(getIntegerWidth(value.getType(), options));
}

bool isOrdinaryPointerType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  auto ptr = dyn_cast<triton::PointerType>(type);
  return ptr && !isa<ShapedType>(ptr.getPointeeType());
}

bool isScalarPointerType(Type type) {
  return isa<triton::PointerType>(type) && isOrdinaryPointerType(type);
}

Type addressValueType(Type source, OpBuilder &builder) {
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    return RankedTensorType::get(tensor.getShape(), builder.getI64Type(),
                                 tensor.getEncoding());
  return builder.getI64Type();
}

OpFoldResult integerAttr(Type type, const llvm::APInt &value,
                         OpBuilder &builder) {
  return scalarIntegerAttr(type, value, builder);
}

OffsetComponents invariantOffset(Value value, OpFoldResult scalar,
                                 OpBuilder &builder) {
  OffsetComponents result;
  result.valueType = value.getType();
  result.shape = getLogicalShape(value.getType());
  result.completeOffset = value;
  result.uniformOffset = scalar;
  result.axes.assign(result.shape.size(), AxisKind::Invariant);
  Type element = getIntegerElementType(value.getType());
  result.strides.assign(result.shape.size(),
                        builder.getIntegerAttr(element, 0));
  return result;
}

FailureOr<OpFoldResult> combine(BinaryKind kind, OpFoldResult lhs,
                                OpFoldResult rhs, Type type,
                                AnalysisOptions options, OpBuilder &builder,
                                Location loc) {
  return foldOrCreateBinary(kind, lhs, rhs, type, options, builder, loc);
}

/// c/s are scalar. Unknown axes keep a zero slot, never a claimed stride.
FailureOr<OffsetComponents>
combineOffsets(Value complete, const OffsetComponents &lhs,
               const OffsetComponents &rhs, BinaryKind kind,
               AnalysisOptions options, OpBuilder &builder) {
  if (lhs.valueType != rhs.valueType || lhs.domain != rhs.domain ||
      lhs.shape != rhs.shape)
    return failure();
  Type scalar = getIntegerElementType(complete.getType());
  auto opaque = makeOpaqueOffset(complete, builder);
  if (failed(opaque))
    return failure();
  OffsetComponents result = std::move(*opaque);
  result.domain = lhs.domain;
  result.axes.clear();
  result.strides.clear();
  Location loc = complete.getLoc();
  for (unsigned axis = 0; axis < lhs.shape.size(); ++axis) {
    if (lhs.axes[axis] == AxisKind::Unknown ||
        rhs.axes[axis] == AxisKind::Unknown) {
      result.axes.push_back(AxisKind::Unknown);
      result.strides.push_back(builder.getIntegerAttr(scalar, 0));
      continue;
    }
    FailureOr<OpFoldResult> stride =
        combine(kind, lhs.strides[axis], rhs.strides[axis], scalar, options,
                builder, loc);
    if (failed(stride))
      return failure();
    result.axes.push_back(lhs.axes[axis] == AxisKind::Invariant &&
                                  rhs.axes[axis] == AxisKind::Invariant
                              ? AxisKind::Invariant
                              : AxisKind::Structured);
    result.strides.push_back(*stride);
  }
  if (result.hasAffineForm()) {
    FailureOr<OpFoldResult> uniform =
        combine(kind, lhs.uniformOffset, rhs.uniformOffset, scalar, options,
                builder, loc);
    if (failed(uniform))
      return failure();
    result.uniformOffset = *uniform;
  }
  return result;
}

/// Check the lifted scalar affine expression before sign extension. Unknown
/// scalar bounds cannot justify moving a modular iN formula into i64.
bool affineFitsSigned(const OffsetComponents &source, unsigned width) {
  if (!source.hasAffineForm())
    return false;
  if (source.isUniform())
    return true;
  if (width >= 64)
    return false;
  auto origin = getIntegerConstant(source.uniformOffset, width);
  if (failed(origin))
    return false;
  __int128 minimum = origin->getSExtValue();
  __int128 maximum = minimum;
  for (unsigned axis = 0; axis < source.shape.size(); ++axis) {
    if (source.shape[axis] <= 0)
      return false;
    auto step = getIntegerConstant(source.strides[axis], width);
    if (failed(step))
      return false;
    __int128 extent = source.shape[axis] - 1;
    __int128 contribution;
    if (__builtin_mul_overflow(extent, __int128(step->getSExtValue()),
                               &contribution) ||
        __builtin_add_overflow(minimum, std::min<__int128>(0, contribution),
                               &minimum) ||
        __builtin_add_overflow(maximum, std::max<__int128>(0, contribution),
                               &maximum))
      return false;
  }
  __int128 signedMin = -(__int128(1) << (width - 1));
  __int128 signedMax = (__int128(1) << (width - 1)) - 1;
  return minimum >= signedMin && maximum <= signedMax;
}

FailureOr<OffsetComponents> projectAddress(Value value,
                                           const AnalysisResult &source,
                                           AnalysisOptions options,
                                           OpBuilder &builder) {
  if (!source.offset || !source.integerRange ||
      !supportsAddressAnalysis(options))
    return failure();
  FailureOr<unsigned> width = getIntegerWidth(value.getType(), options);
  if (failed(width) || *width > 64)
    return failure();
  Type targetType = addressValueType(value.getType(), builder);
  FailureOr<OpFoldResult> complete =
      foldOrCreateCast(source.offset->completeOffset, targetType,
                       CastKind::Signed, options, builder, value.getLoc());
  if (failed(complete))
    return failure();
  OffsetComponents result;
  result.valueType = targetType;
  result.domain = ArithmeticDomain::ElementAddress;
  result.shape = source.offset->shape;
  result.completeOffset = *complete;
  Type scalarType = builder.getI64Type();
  bool lift = *width == 64 || affineFitsSigned(*source.offset, *width);
  for (AxisKind axis : source.offset->axes) {
    if (axis == AxisKind::Invariant || (lift && axis == AxisKind::Structured))
      result.axes.push_back(axis);
    else
      result.axes.push_back(AxisKind::Unknown);
  }
  result.strides.assign(result.shape.size(),
                        builder.getIntegerAttr(scalarType, 0));
  for (unsigned axis = 0; axis < result.shape.size(); ++axis) {
    if (result.axes[axis] != AxisKind::Structured)
      continue;
    FailureOr<OpFoldResult> stride =
        foldOrCreateCast(source.offset->strides[axis], scalarType,
                         CastKind::Signed, options, builder, value.getLoc());
    if (failed(stride))
      return failure();
    result.strides[axis] = *stride;
  }
  result.uniformOffset = builder.getIntegerAttr(scalarType, 0);
  if (result.hasAffineForm()) {
    if (!isa<RankedTensorType>(result.valueType)) {
      // A scalar's complete value is its uniform value. Reuse the same cast
      // rather than inserting two identical extension helpers.
      result.uniformOffset = *complete;
    } else {
      FailureOr<OpFoldResult> uniform =
          foldOrCreateCast(source.offset->uniformOffset, scalarType,
                           CastKind::Signed, options, builder, value.getLoc());
      if (failed(uniform))
        return failure();
      result.uniformOffset = *uniform;
    }
  }
  return result;
}

FailureOr<Value> asTensorOffset(OpFoldResult value, Type fromType,
                                RankedTensorType target, OpBuilder &builder,
                                Location loc) {
  if (auto tensor = dyn_cast<RankedTensorType>(fromType)) {
    if (tensor != target)
      return failure();
    return materialize(value, target, builder, loc);
  }
  FailureOr<Value> scalar =
      materialize(value, target.getElementType(), builder, loc);
  if (failed(scalar))
    return failure();
  return splatTo(*scalar, target, builder, loc);
}

FailureOr<PointerComponents> addPointerOffsets(Value resultValue,
                                               const PointerComponents &parent,
                                               const OffsetComponents &delta,
                                               AnalysisOptions options,
                                               OpBuilder &builder) {
  if (parent.offsets.domain != ArithmeticDomain::ElementAddress ||
      delta.domain != ArithmeticDomain::ElementAddress ||
      parent.elementType != cast<triton::PointerType>(
                                getIntegerElementType(resultValue.getType()))
                                .getPointeeType())
    return failure();
  Type numericType = addressValueType(resultValue.getType(), builder);
  SmallVector<int64_t> shape = getLogicalShape(numericType);
  if (isa<RankedTensorType>(parent.offsets.valueType) &&
      parent.offsets.shape != shape)
    return failure();
  if (isa<RankedTensorType>(delta.valueType) && delta.shape != shape)
    return failure();
  OpFoldResult zero = builder.getI64IntegerAttr(0);
  FailureOr<OpFoldResult> newO = failure();
  Location loc = resultValue.getLoc();
  if (auto tensor = dyn_cast<RankedTensorType>(numericType)) {
    FailureOr<Value> left =
        asTensorOffset(parent.offsets.completeOffset, parent.offsets.valueType,
                       tensor, builder, loc);
    FailureOr<Value> right = asTensorOffset(
        delta.completeOffset, delta.valueType, tensor, builder, loc);
    if (failed(left) || failed(right))
      return failure();
    newO = OpFoldResult(
        builder.create<arith::AddIOp>(loc, *left, *right).getResult());
  } else {
    newO = combine(BinaryKind::Add, parent.offsets.completeOffset,
                   delta.completeOffset, builder.getI64Type(), options, builder,
                   loc);
    if (failed(newO))
      return failure();
  }
  OffsetComponents offsets;
  offsets.valueType = numericType;
  offsets.domain = ArithmeticDomain::ElementAddress;
  offsets.shape = shape;
  offsets.completeOffset = *newO;
  offsets.uniformOffset = zero;
  for (unsigned axis = 0; axis < shape.size(); ++axis) {
    AxisKind left = !isa<RankedTensorType>(parent.offsets.valueType)
                        ? AxisKind::Invariant
                        : parent.offsets.axes[axis];
    AxisKind right = !isa<RankedTensorType>(delta.valueType)
                         ? AxisKind::Invariant
                         : delta.axes[axis];
    if (left == AxisKind::Unknown || right == AxisKind::Unknown) {
      offsets.axes.push_back(AxisKind::Unknown);
      offsets.strides.push_back(zero);
      continue;
    }
    OpFoldResult l = !isa<RankedTensorType>(parent.offsets.valueType)
                         ? zero
                         : parent.offsets.strides[axis];
    OpFoldResult r =
        !isa<RankedTensorType>(delta.valueType) ? zero : delta.strides[axis];
    FailureOr<OpFoldResult> stride = combine(
        BinaryKind::Add, l, r, builder.getI64Type(), options, builder, loc);
    if (failed(stride))
      return failure();
    offsets.axes.push_back(left == AxisKind::Invariant &&
                                   right == AxisKind::Invariant
                               ? AxisKind::Invariant
                               : AxisKind::Structured);
    offsets.strides.push_back(*stride);
  }
  if (offsets.hasAffineForm()) {
    FailureOr<OpFoldResult> uniform = combine(
        BinaryKind::Add, parent.offsets.uniformOffset, delta.uniformOffset,
        builder.getI64Type(), options, builder, loc);
    if (failed(uniform))
      return failure();
    offsets.uniformOffset = *uniform;
  }
  PointerComponents result = parent;
  result.offsets = std::move(offsets);
  return result;
}

} // namespace

FailureOr<SmallVector<AnalysisRequest>>
PointerRules::collectInputs(AnalysisRequest request) const {
  if (request.kind != SourceInteger && !supportsAddressAnalysis(options))
    return failure();
  SmallVector<AnalysisRequest> result;
  Value value = request.value;
  if (request.kind == AdvanceDeltaRequest ||
      (request.kind == PointerValue && isBlockPointerType(value.getType())))
    return collectBlockPointerInputs(request);
  if (request.kind == AddressDelta) {
    result.push_back({value, SourceInteger});
    return result;
  }
  if (request.kind == SourceInteger) {
    if (!isIntegerValue(value, options))
      return failure();
    if (auto op = value.getDefiningOp<arith::AddIOp>())
      result = {{op.getLhs(), SourceInteger}, {op.getRhs(), SourceInteger}};
    else if (auto op = value.getDefiningOp<arith::SubIOp>())
      result = {{op.getLhs(), SourceInteger}, {op.getRhs(), SourceInteger}};
    else if (auto op = value.getDefiningOp<arith::MulIOp>())
      result = {{op.getLhs(), SourceInteger}, {op.getRhs(), SourceInteger}};
    else if (auto op = value.getDefiningOp<arith::SelectOp>())
      result = {{op.getTrueValue(), SourceInteger},
                {op.getFalseValue(), SourceInteger}};
    else if (auto op = value.getDefiningOp<arith::ExtSIOp>())
      result.push_back({op.getIn(), SourceInteger});
    else if (auto op = value.getDefiningOp<arith::ExtUIOp>())
      result.push_back({op.getIn(), SourceInteger});
    else if (auto op = value.getDefiningOp<arith::TruncIOp>())
      result.push_back({op.getIn(), SourceInteger});
    else if (auto op = value.getDefiningOp<arith::IndexCastOp>())
      result.push_back({op.getIn(), SourceInteger});
    else if (auto op = value.getDefiningOp<arith::IndexCastUIOp>())
      result.push_back({op.getIn(), SourceInteger});
    else if (auto op = value.getDefiningOp<triton::SplatOp>())
      result.push_back({op.getSrc(), SourceInteger});
    else if (auto op = value.getDefiningOp<triton::BroadcastOp>())
      result.push_back({op.getSrc(), SourceInteger});
    else if (auto op = value.getDefiningOp<triton::ExpandDimsOp>())
      result.push_back({op.getSrc(), SourceInteger});
    return result;
  }
  if (request.kind != PointerValue || !isOrdinaryPointerType(value.getType()))
    return failure();
  if (auto op = value.getDefiningOp<triton::AddPtrOp>()) {
    result = {{op.getPtr(), PointerValue}, {op.getOffset(), AddressDelta}};
  } else if (auto op = value.getDefiningOp<triton::SplatOp>()) {
    result.push_back({op.getSrc(), PointerValue});
  } else if (auto op = value.getDefiningOp<triton::BroadcastOp>()) {
    result.push_back({op.getSrc(), PointerValue});
  } else if (auto op = value.getDefiningOp<triton::ExpandDimsOp>()) {
    result.push_back({op.getSrc(), PointerValue});
  } else if (auto op = value.getDefiningOp<arith::SelectOp>()) {
    if (isa<RankedTensorType>(value.getType()))
      result = {{op.getTrueValue(), PointerValue},
                {op.getFalseValue(), PointerValue}};
  }
  return result;
}

namespace {

FailureOr<AnalysisResult> transferAddressDelta(Value value,
                                               ArrayRef<AnalysisResult> inputs,
                                               AnalysisOptions options,
                                               OpBuilder &builder) {
  if (inputs.size() != 1)
    return failure();
  FailureOr<OffsetComponents> projected =
      projectAddress(value, inputs.front(), options, builder);
  if (failed(projected))
    return failure();
  AnalysisResult result;
  result.offset = std::move(*projected);
  const auto &range = *inputs.front().integerRange;
  result.integerRange = range.getBitWidth() < 64 ? range.signExtend(64) : range;
  return result;
}

FailureOr<AnalysisResult> transferSourceInteger(Value value,
                                                ArrayRef<AnalysisResult> inputs,
                                                AnalysisOptions options,
                                                OpBuilder &builder) {
  // Extensions may omit the basic view. Reject it before using any source
  // operand; shape operations need the range just as arithmetic operations do.
  for (const AnalysisResult &input : inputs)
    if (!input.offset || !input.integerRange)
      return failure();
  FailureOr<unsigned> width = getIntegerWidth(value.getType(), options);
  if (failed(width))
    return failure();
  Type scalar = getIntegerElementType(value.getType());
  auto opaque = makeOpaqueOffset(value, builder);
  if (failed(opaque))
    return failure();
  OffsetComponents state = std::move(*opaque);
  llvm::ConstantRange range = getValueRange(value, *width);
  if (auto makeRange = value.getDefiningOp<triton::MakeRangeOp>()) {
    if (state.shape.size() != 1 || state.shape[0] <= 0 || *width != 32)
      return failure();
    int64_t start = makeRange.getStartAttr().getInt();
    int64_t end = makeRange.getEndAttr().getInt();
    if (end <= start || end - start != state.shape[0])
      return failure();
    state.axes = {AxisKind::Structured};
    state.uniformOffset = integerAttr(scalar, llvm::APInt(32, start), builder);
    state.strides = {integerAttr(scalar, llvm::APInt(32, 1), builder)};
    range = llvm::ConstantRange(llvm::APInt(32, start), llvm::APInt(32, end));
  } else if (FailureOr<llvm::APInt> constant =
                 getIntegerConstant(value, *width);
             succeeded(constant)) {
    state = invariantOffset(value, integerAttr(scalar, *constant, builder),
                            builder);
  } else if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    if (inputs.size() != 2)
      return failure();
    const auto &left = *inputs[0].offset;
    const auto &right = *inputs[1].offset;
    range = inputs[0].integerRange->unionWith(*inputs[1].integerRange);
    if (select.getTrueValue() == select.getFalseValue()) {
      state = left;
      state.completeOffset = value;
    } else if (select.getCondition().getType().isInteger(1) &&
               left.valueType == right.valueType && left.hasAffineForm() &&
               right.hasAffineForm()) {
      auto choose = [&](OpFoldResult lhs,
                        OpFoldResult rhs) -> FailureOr<OpFoldResult> {
        if (lhs == rhs)
          return lhs;
        auto l = materialize(lhs, scalar, builder, value.getLoc());
        auto r = materialize(rhs, scalar, builder, value.getLoc());
        if (failed(l) || failed(r))
          return failure();
        return OpFoldResult(builder
                                .create<arith::SelectOp>(value.getLoc(),
                                                         select.getCondition(),
                                                         *l, *r)
                                .getResult());
      };
      auto uniform = choose(left.uniformOffset, right.uniformOffset);
      if (failed(uniform))
        return failure();
      state.uniformOffset = *uniform;
      for (unsigned axis = 0; axis < state.shape.size(); ++axis) {
        auto stride = choose(left.strides[axis], right.strides[axis]);
        if (failed(stride))
          return failure();
        state.strides[axis] = *stride;
        state.axes[axis] = left.axes[axis] == AxisKind::Invariant &&
                                   right.axes[axis] == AxisKind::Invariant
                               ? AxisKind::Invariant
                               : AxisKind::Structured;
      }
    }
  } else if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    if (inputs.size() != 1 ||
        isa<RankedTensorType>(inputs.front().offset->valueType))
      return failure();
    state =
        invariantOffset(value, inputs.front().offset->uniformOffset, builder);
    range = *inputs.front().integerRange;
  } else if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
    if (inputs.size() != 1)
      return failure();
    const auto &src = *inputs.front().offset;
    if (src.shape.size() != state.shape.size())
      return failure();
    state.axes = src.axes;
    state.strides = src.strides;
    for (unsigned axis = 0; axis < state.shape.size(); ++axis) {
      if (src.shape[axis] != state.shape[axis]) {
        if (src.shape[axis] != 1)
          return failure();
        // A one-element opaque tensor is constant along this dimension,
        // but its scalar value is not known. Keep the Unknown carrier.
        if (src.axes[axis] != AxisKind::Unknown)
          state.axes[axis] = AxisKind::Invariant;
        state.strides[axis] = builder.getIntegerAttr(scalar, 0);
      }
    }
    if (state.hasAffineForm())
      state.uniformOffset = src.uniformOffset;
    range = *inputs.front().integerRange;
  } else if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
    if (inputs.size() != 1)
      return failure();
    const auto &src = *inputs.front().offset;
    unsigned axis = expand.getAxis();
    if (axis > src.shape.size() || state.shape.size() != src.shape.size() + 1)
      return failure();
    state.axes = src.axes;
    state.axes.insert(state.axes.begin() + axis, AxisKind::Invariant);
    state.strides = src.strides;
    state.strides.insert(state.strides.begin() + axis,
                         builder.getIntegerAttr(scalar, 0));
    if (state.hasAffineForm())
      state.uniformOffset = src.uniformOffset;
    range = *inputs.front().integerRange;
  } else if (value.getDefiningOp<arith::AddIOp>() ||
             value.getDefiningOp<arith::SubIOp>()) {
    if (inputs.size() != 2)
      return failure();
    BinaryKind kind = value.getDefiningOp<arith::AddIOp>()
                          ? BinaryKind::Add
                          : BinaryKind::Subtract;
    FailureOr<OffsetComponents> joined = combineOffsets(
        value, *inputs[0].offset, *inputs[1].offset, kind, options, builder);
    if (failed(joined))
      return failure();
    state = std::move(*joined);
    range = kind == BinaryKind::Add
                ? inputs[0].integerRange->add(*inputs[1].integerRange)
                : inputs[0].integerRange->sub(*inputs[1].integerRange);
  } else if (value.getDefiningOp<arith::MulIOp>()) {
    if (inputs.size() != 2)
      return failure();
    const OffsetComponents &left = *inputs[0].offset;
    const OffsetComponents &right = *inputs[1].offset;
    if (left.shape != right.shape || left.valueType != right.valueType)
      return failure();
    bool leftUniform = left.isUniform();
    bool rightUniform = right.isUniform();
    if (leftUniform || rightUniform) {
      const auto &varying = leftUniform ? right : left;
      const auto &scale = leftUniform ? left : right;
      state.axes = varying.axes;
      state.strides.clear();
      for (unsigned axis = 0; axis < state.shape.size(); ++axis) {
        if (state.axes[axis] == AxisKind::Unknown) {
          state.strides.push_back(builder.getIntegerAttr(scalar, 0));
          continue;
        }
        FailureOr<OpFoldResult> stride = combine(
            BinaryKind::Multiply, varying.strides[axis], scale.uniformOffset,
            scalar, options, builder, value.getLoc());
        if (failed(stride))
          return failure();
        state.strides.push_back(*stride);
      }
      if (state.hasAffineForm()) {
        FailureOr<OpFoldResult> uniform = combine(
            BinaryKind::Multiply, varying.uniformOffset, scale.uniformOffset,
            scalar, options, builder, value.getLoc());
        if (failed(uniform))
          return failure();
        state.uniformOffset = *uniform;
      }
    }
    range = inputs[0].integerRange->multiply(*inputs[1].integerRange);
  } else if (value.getDefiningOp<arith::ExtSIOp>() ||
             value.getDefiningOp<arith::ExtUIOp>() ||
             value.getDefiningOp<arith::TruncIOp>() ||
             value.getDefiningOp<arith::IndexCastOp>() ||
             value.getDefiningOp<arith::IndexCastUIOp>()) {
    if (inputs.size() != 1)
      return failure();
    const auto &src = *inputs[0].offset;
    unsigned sourceWidth = inputs[0].integerRange->getBitWidth();
    bool unsignedCast = value.getDefiningOp<arith::ExtUIOp>() ||
                        value.getDefiningOp<arith::IndexCastUIOp>();
    bool narrowing = *width < sourceWidth;
    bool safeLift =
        narrowing || *width == sourceWidth ||
        (affineFitsSigned(src, sourceWidth) &&
         (!unsignedCast || inputs[0].integerRange->isAllNonNegative()));
    state.axes.clear();
    state.strides.clear();
    for (unsigned axis = 0; axis < src.shape.size(); ++axis) {
      AxisKind candidate = src.axes[axis];
      if (candidate == AxisKind::Structured && !safeLift)
        candidate = AxisKind::Unknown;
      state.axes.push_back(candidate);
      if (candidate != AxisKind::Structured) {
        state.strides.push_back(builder.getIntegerAttr(scalar, 0));
        continue;
      }
      // A proven nonnegative affine value can still have a negative stride
      // (e.g. 3 - range). Extend the coefficient as signed, not as extui.
      FailureOr<OpFoldResult> converted =
          foldOrCreateCast(src.strides[axis], scalar, CastKind::Signed, options,
                           builder, value.getLoc());
      if (failed(converted))
        return failure();
      state.strides.push_back(*converted);
    }
    if (state.hasAffineForm()) {
      FailureOr<OpFoldResult> converted =
          foldOrCreateCast(src.uniformOffset, scalar,
                           unsignedCast ? CastKind::Unsigned : CastKind::Signed,
                           options, builder, value.getLoc());
      if (failed(converted))
        return failure();
      state.uniformOffset = *converted;
    }
    if (*width < sourceWidth)
      range = inputs[0].integerRange->truncate(*width);
    else if (*width == sourceWidth)
      range = *inputs[0].integerRange;
    else if (unsignedCast)
      range = inputs[0].integerRange->zeroExtend(*width);
    else
      range = inputs[0].integerRange->signExtend(*width);
  }
  AnalysisResult result;
  result.offset = std::move(state);
  result.integerRange = std::move(range);
  return result;
}

FailureOr<AnalysisResult>
transferOrdinaryPointer(Value value, ArrayRef<AnalysisResult> inputs,
                        AnalysisOptions options, OpBuilder &builder) {
  if (!isOrdinaryPointerType(value.getType()))
    return failure();
  Type pointerElement = getIntegerElementType(value.getType());
  auto pointerType = cast<triton::PointerType>(pointerElement);
  if (auto add = value.getDefiningOp<triton::AddPtrOp>()) {
    if (inputs.size() != 2 || !inputs[0].pointer || !inputs[1].offset)
      return failure();
    FailureOr<PointerComponents> updated = addPointerOffsets(
        value, *inputs[0].pointer, *inputs[1].offset, options, builder);
    if (failed(updated))
      return failure();
    AnalysisResult result;
    result.pointer = std::move(*updated);
    return result;
  }
  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    if (inputs.size() != 1 || !inputs[0].pointer ||
        isa<RankedTensorType>(inputs[0].pointer->offsets.valueType))
      return failure();
    auto tensor = cast<RankedTensorType>(value.getType());
    auto numeric = cast<RankedTensorType>(addressValueType(tensor, builder));
    FailureOr<Value> offsets =
        asTensorOffset(inputs[0].pointer->offsets.completeOffset,
                       builder.getI64Type(), numeric, builder, value.getLoc());
    if (failed(offsets))
      return failure();
    PointerComponents result = *inputs[0].pointer;
    result.offsets.valueType = numeric;
    result.offsets.shape = llvm::to_vector(tensor.getShape());
    result.offsets.completeOffset = *offsets;
    result.offsets.axes.assign(tensor.getRank(), AxisKind::Invariant);
    result.offsets.strides.assign(tensor.getRank(),
                                  builder.getI64IntegerAttr(0));
    AnalysisResult wrapped;
    wrapped.pointer = std::move(result);
    return wrapped;
  }
  if (value.getDefiningOp<triton::BroadcastOp>() ||
      value.getDefiningOp<triton::ExpandDimsOp>()) {
    if (inputs.size() != 1 || !inputs[0].pointer)
      return failure();
    PointerComponents result = *inputs[0].pointer;
    auto target =
        cast<RankedTensorType>(addressValueType(value.getType(), builder));
    FailureOr<Value> oldOffset =
        materialize(result.offsets.completeOffset, result.offsets.valueType,
                    builder, value.getLoc());
    if (failed(oldOffset))
      return failure();
    Value complete;
    if (value.getDefiningOp<triton::BroadcastOp>()) {
      auto oldType = dyn_cast<RankedTensorType>(result.offsets.valueType);
      if (!oldType || oldType.getRank() != target.getRank())
        return failure();
      complete = builder.create<triton::BroadcastOp>(value.getLoc(), target,
                                                     *oldOffset);
      for (unsigned i = 0; i < target.getRank(); ++i) {
        if (oldType.getShape()[i] != target.getShape()[i]) {
          // A single opaque source lane cannot supply a proven scalar c.
          // Retain that unknown carrier even when the target repeats it.
          if (result.offsets.axes[i] != AxisKind::Unknown)
            result.offsets.axes[i] = AxisKind::Invariant;
          result.offsets.strides[i] = builder.getI64IntegerAttr(0);
        }
      }
    } else {
      unsigned axis = value.getDefiningOp<triton::ExpandDimsOp>().getAxis();
      if (axis > result.offsets.shape.size())
        return failure();
      complete = builder.create<triton::ExpandDimsOp>(value.getLoc(), target,
                                                      *oldOffset, axis);
      result.offsets.axes.insert(result.offsets.axes.begin() + axis,
                                 AxisKind::Invariant);
      result.offsets.strides.insert(result.offsets.strides.begin() + axis,
                                    builder.getI64IntegerAttr(0));
    }
    result.offsets.valueType = target;
    result.offsets.shape = llvm::to_vector(target.getShape());
    result.offsets.completeOffset = complete;
    AnalysisResult wrapped;
    wrapped.pointer = std::move(result);
    return wrapped;
  }
  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    if (isa<RankedTensorType>(value.getType())) {
      if (inputs.size() != 2 || !inputs[0].pointer || !inputs[1].pointer ||
          inputs[0].pointer->base != inputs[1].pointer->base)
        return failure();
      PointerComponents result = *inputs[0].pointer;
      auto target =
          cast<RankedTensorType>(addressValueType(value.getType(), builder));
      FailureOr<Value> yes =
          asTensorOffset(inputs[0].pointer->offsets.completeOffset,
                         inputs[0].pointer->offsets.valueType, target, builder,
                         value.getLoc());
      FailureOr<Value> no =
          asTensorOffset(inputs[1].pointer->offsets.completeOffset,
                         inputs[1].pointer->offsets.valueType, target, builder,
                         value.getLoc());
      if (failed(yes) || failed(no))
        return failure();
      Value full = builder.create<arith::SelectOp>(
          value.getLoc(), select.getCondition(), *yes, *no);
      auto opaque =
          makeOpaqueOffset(full, builder, ArithmeticDomain::ElementAddress);
      if (failed(opaque))
        return failure();
      result.offsets = std::move(*opaque);
      AnalysisResult wrapped;
      wrapped.pointer = std::move(result);
      return wrapped;
    }
  }
  if (!isScalarPointerType(value.getType()))
    return failure();
  PointerComponents leaf;
  leaf.base = value;
  leaf.elementType = pointerType.getPointeeType();
  leaf.addressSpace = pointerType.getAddressSpace();
  leaf.offsets.valueType = builder.getI64Type();
  leaf.offsets.domain = ArithmeticDomain::ElementAddress;
  leaf.offsets.completeOffset = builder.getI64IntegerAttr(0);
  leaf.offsets.uniformOffset = builder.getI64IntegerAttr(0);
  AnalysisResult result;
  result.pointer = std::move(leaf);
  return result;
}

} // namespace

FailureOr<PointerRules::Result>
PointerRules::transfer(AnalysisRequest request, ArrayRef<Result> inputs,
                       OpBuilder &builder) const {
  switch (request.kind) {
  case SourceInteger:
    return transferSourceInteger(request.value, inputs, options, builder);
  case AddressDelta:
    return transferAddressDelta(request.value, inputs, options, builder);
  case PointerValue:
    if (isBlockPointerType(request.value.getType()))
      return transferBlockPointer(request, inputs, builder);
    return transferOrdinaryPointer(request.value, inputs, options, builder);
  case AdvanceDeltaRequest:
    return transferBlockPointer(request, inputs, builder);
  default:
    return failure();
  }
}

LogicalResult PointerRules::mapResultValues(
    Result &result, llvm::function_ref<FailureOr<Value>(Value)> mapper) const {
  if (result.offset && failed(mapOffsetValues(*result.offset, mapper)))
    return failure();
  if (result.pointer) {
    FailureOr<Value> base = mapper(result.pointer->base);
    if (failed(base) ||
        failed(mapOffsetValues(result.pointer->offsets, mapper)))
      return failure();
    result.pointer->base = *base;
    if (result.pointer->block) {
      auto &block = *result.pointer->block;
      for (auto *fields : {&block.parentShape, &block.coordinateOffsets})
        for (OpFoldResult &field : *fields)
          if (failed(mapOfr(field, mapper)))
            return failure();
    }
  }
  if (result.advanceDelta)
    for (OpFoldResult &field : result.advanceDelta->coordinateDeltas)
      if (failed(mapOfr(field, mapper)))
        return failure();
  return success();
}

const OffsetComponents *getOffset(const AnalysisResult &result) {
  return result.offset ? &*result.offset : nullptr;
}

} // namespace mlir::triton::pointer::detail
