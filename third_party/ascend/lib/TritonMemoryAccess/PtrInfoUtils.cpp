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

#include "TritonMemoryAccess/PtrInfoUtils.h"
#include "TritonMemoryAccess/PtrAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::ptr::detail {
// Comparisons of index-domain descriptor coefficients.
bool equalConstant(OpFoldResult lhs, OpFoldResult rhs) {
  if (lhs == rhs)
    return true;
  auto l = getIntegerConstant(lhs, 64), r = getIntegerConstant(rhs, 64);
  return succeeded(l) && succeeded(r) && *l == *r;
}
bool zero(OpFoldResult value) {
  auto integer = getIntegerConstant(value, 64);
  return succeeded(integer) && integer->isZero();
}

namespace {
LogicalResult normalizeAxis(AxisInfo &axis, int64_t size, Type scalar,
                            OpBuilder &builder) {
  if (axis.subAxes.empty()) {
    if (axis.kind != AxisKind::Unknown && !axis.stride)
      return failure();
    if (axis.kind != AxisKind::Unknown)
      axis.kind =
          zero(axis.stride) ? AxisKind::Invariant : AxisKind::Structured;
    else
      axis.stride = builder.getIntegerAttr(scalar, 0);
    return success();
  }
  int64_t product = 1;
  for (const auto &sub : axis.subAxes) {
    if (sub.shape <= 0 || !sub.stride)
      return failure();
    if (__builtin_mul_overflow(product, sub.shape, &product))
      return failure();
  }
  if (product != size)
    return failure();
  axis.kind = AxisKind::Unknown;
  axis.stride = builder.getIntegerAttr(scalar, 0);
  if (llvm::all_of(axis.subAxes,
                   [&](const SubAxisInfo &sub) { return zero(sub.stride); })) {
    axis.kind = AxisKind::Invariant;
    return success();
  }
  // Prove that each outer coefficient equals the innermost coefficient times
  // all inner lengths. Do not emit IR just to test a symbolic equality.
  OpFoldResult stride = builder.getIntegerAttr(scalar, 0);
  for (const auto &sub : llvm::reverse(axis.subAxes))
    if (sub.shape != 1) {
      stride = sub.stride;
      break;
    }
  auto inner = getIntegerConstant(stride, 64);
  llvm::APInt scale(64, 1);
  bool linear = true;
  for (const auto &sub : llvm::reverse(axis.subAxes)) {
    int64_t n = sub.shape;
    if (n != 1) {
      if (scale.isOne())
        linear &= equalConstant(sub.stride, stride);
      else {
        auto coefficient = getIntegerConstant(sub.stride, 64);
        linear &= succeeded(inner) && succeeded(coefficient) &&
                  *coefficient == *inner * scale;
      }
    }
    scale *= llvm::APInt(64, n);
  }
  if (linear) {
    axis.stride = stride;
    axis.kind = zero(stride) ? AxisKind::Invariant : AxisKind::Structured;
  }
  return success();
}

FailureOr<SmallVector<SubAxisInfo>>
expandLinear(const AxisInfo &axis, ArrayRef<SubAxisInfo> partition,
             ConcreteArithmetic &arithmetic) {
  if (axis.kind == AxisKind::Unknown || !axis.stride)
    return failure();
  SmallVector<SubAxisInfo> parts;
  OpFoldResult coefficient = axis.stride;
  for (unsigned i = partition.size(); i > 0; --i) {
    const auto &sub = partition[i - 1];
    parts.push_back({sub.shape, coefficient});
    // Scale only when another outer sub-axis needs a coefficient.
    if (i == 1)
      break;
    auto scaled = arithmetic.multiply(
        coefficient,
        arithmetic.builder.getIntegerAttr(arithmetic.type, sub.shape));
    if (failed(scaled))
      return failure();
    coefficient = *scaled;
  }
  std::reverse(parts.begin(), parts.end());
  return parts;
}

// Exact division of descriptor fields.
bool canDivideAxis(const AxisInfo &axis, uint64_t divisor) {
  if (!axis.subAxes.empty())
    return llvm::all_of(axis.subAxes, [&](const SubAxisInfo &sub) {
      return isKnownMultiple(sub.stride, divisor);
    });
  return axis.kind != AxisKind::Unknown &&
         isKnownMultiple(axis.stride, divisor);
}

OpFoldResult divideIndex(OpFoldResult value, uint64_t divisor,
                         OpBuilder &builder, Location loc) {
  if (divisor == 1)
    return value;
  auto constant = getIntegerConstant(value, 64);
  if (succeeded(constant))
    return builder.getIndexAttr(
        constant->sdiv(llvm::APInt(64, divisor)).getSExtValue());
  Value d = builder.create<arith::ConstantIndexOp>(loc, divisor);
  return builder.create<arith::DivSIOp>(loc, cast<Value>(value), d).getResult();
}

} // namespace

// Uniform origin materialization.
void setUniformOrigin(OffsetInfo &info, Value value, OpBuilder &builder,
                      OpFoldResult scalar) {
  if (info.scalarOffset || !llvm::all_of(info.axes, [](const AxisInfo &axis) {
        return axis.kind == AxisKind::Invariant;
      }))
    return;
  // Reuse the supplied scalar; otherwise obtain a representative value.
  if (!scalar) {
    if (auto constant = getIntegerConstant(value, 64); succeeded(constant)) {
      info.scalarOffset = builder.getIndexAttr(constant->getSExtValue());
      return;
    }
    // Read the result itself without rebuilding its arithmetic.
    scalar = value;
    if (isa<RankedTensorType>(value.getType())) {
      if (llvm::any_of(info.getShape(), [](int64_t size) { return size <= 0; }))
        return;
      builder.setInsertionPointAfterValue(value);
      SmallVector<Value> indices;
      for (unsigned i = 0; i < info.getShape().size(); ++i)
        indices.push_back(
            builder.create<arith::ConstantIndexOp>(value.getLoc(), 0));
      scalar = builder.create<tensor::ExtractOp>(value.getLoc(), value, indices)
                   .getResult();
    }
  }
  if (auto source = dyn_cast<Value>(scalar))
    builder.setInsertionPointAfterValue(source);
  auto origin =
      foldOrCreateCast(scalar, builder.getIndexType(),
                       IntegerExtensionKind::Signed, builder, value.getLoc());
  if (succeeded(origin))
    info.scalarOffset = *origin;
}

// Address types and complete offset values.
Type addressValueType(Type source, OpBuilder &builder) {
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    return RankedTensorType::get(tensor.getShape(), builder.getIndexType(),
                                 tensor.getEncoding());
  return builder.getIndexType();
}

Type addressStorageType(Type source, OpBuilder &builder) {
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    return tensor.clone(builder.getI64Type());
  return builder.getIndexType();
}

FailureOr<Value> materializeFullOffset(const OffsetInfo &info, Type target,
                                       OpBuilder &builder, Location loc) {
  // Prefer the exact full value; a uniform origin can supply every lane.
  OpFoldResult offset = info.fullOffsets;
  if (!offset && info.isUniform())
    offset = info.scalarOffset;
  if (!offset)
    return failure();

  // Preserve source values by allowing only equal-width or widening casts.
  Type source = isa<Value>(offset)
                    ? cast<Value>(offset).getType()
                    : cast<TypedAttr>(cast<Attribute>(offset)).getType();
  auto width = getIntegerWidth(source);
  auto targetWidth = getIntegerWidth(target);
  if (failed(width) || failed(targetWidth) || *width > *targetWidth)
    return failure();

  // Convert the value, then broadcast a scalar if the target is a tensor.
  auto tensor = dyn_cast<RankedTensorType>(target);
  bool broadcast = tensor && !isa<RankedTensorType>(source);
  Type castType = broadcast ? tensor.getElementType() : target;
  auto converted = foldOrCreateCast(offset, castType,
                                    IntegerExtensionKind::Signed, builder, loc);
  if (failed(converted))
    return failure();
  auto value = materialize(*converted, castType, builder, loc);
  if (failed(value))
    return failure();
  return broadcast ? splatTo(*value, tensor, builder, loc) : value;
}

// Axis alignment, queries and normalization.
// Only proven equal partitions are merged. A linear axis can be expressed in
// either partition. Incompatible decompositions lose that axis proof, while
// the exact fullOffsets and independent origin remain available.
LogicalResult alignAxes(AxisInfo &lhs, AxisInfo &rhs,
                        ConcreteArithmetic &arithmetic) {
  if (lhs.subAxes.empty() && rhs.subAxes.empty())
    return success();
  if (lhs.subAxes.empty()) {
    auto expanded = expandLinear(lhs, rhs.subAxes, arithmetic);
    if (failed(expanded))
      return failure();
    lhs.subAxes = std::move(*expanded);
  }
  if (rhs.subAxes.empty()) {
    auto expanded = expandLinear(rhs, lhs.subAxes, arithmetic);
    if (failed(expanded))
      return failure();
    rhs.subAxes = std::move(*expanded);
  }
  if (lhs.subAxes.size() != rhs.subAxes.size())
    return failure();
  for (auto [l, r] : llvm::zip(lhs.subAxes, rhs.subAxes))
    if (l.shape != r.shape)
      return failure();
  return success();
}

SmallVector<OpFoldResult> axisStrides(const OffsetInfo &info) {
  SmallVector<OpFoldResult> strides;
  for (const auto &axis : info.axes)
    strides.push_back(axis.stride);
  return strides;
}

LogicalResult normalizeAxes(OffsetInfo &info, OpBuilder &builder) {
  if (info.axes.size() != info.getShape().size())
    return failure();
  Type scalar = builder.getIndexType();
  for (unsigned i = 0; i < info.axes.size(); ++i)
    if (failed(
            normalizeAxis(info.axes[i], info.getShape()[i], scalar, builder)))
      return failure();
  return success();
}

// Divisibility proofs and exact unit conversion.
bool canDivideDecomposition(const OffsetInfo &info, uint64_t divisor) {
  return info.hasStructuredDecomposition() &&
         isKnownMultiple(info.scalarOffset, divisor) &&
         llvm::all_of(info.axes, [&](const AxisInfo &axis) {
           return canDivideAxis(axis, divisor);
         });
}

// The caller has proved that the complete value can be divided exactly.
// Retain only descriptor fields whose units can also be converted exactly.
void divideDecomposition(OffsetInfo &info, uint64_t divisor, OpBuilder &builder,
                         Location loc) {
  if (info.scalarOffset) {
    if (isKnownMultiple(info.scalarOffset, divisor))
      info.scalarOffset = divideIndex(info.scalarOffset, divisor, builder, loc);
    else
      info.scalarOffset = {};
  }
  for (auto &axis : info.axes) {
    if (!canDivideAxis(axis, divisor)) {
      axis = AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0));
    } else if (!axis.subAxes.empty()) {
      for (auto &sub : axis.subAxes)
        sub.stride = divideIndex(sub.stride, divisor, builder, loc);
    } else {
      axis.stride = divideIndex(axis.stride, divisor, builder, loc);
    }
  }
}

// Range proofs for an existing decomposition.
bool decompositionFitsSigned(const OffsetInfo &info, unsigned width) {
  return info.isUniform() ||
         constantDecompositionRange(info, width).has_value();
}

std::optional<llvm::ConstantRange>
constantDecompositionRange(const OffsetInfo &info, unsigned width) {
  if (!info.hasStructuredDecomposition() || width == 0 || width > 64)
    return std::nullopt;
  // Descriptor coefficients are index values. Do not truncate them to the
  // source width: an i8 tensor can legitimately have index stride 200.
  auto origin = getIntegerConstant(info.scalarOffset, 64);
  if (failed(origin))
    return std::nullopt;
  __int128 minimum = origin->getSExtValue(), maximum = minimum;
  auto account = [&](OpFoldResult coefficient, int64_t size) {
    auto stride = getIntegerConstant(coefficient, 64);
    if (failed(stride) || size <= 0)
      return false;
    __int128 term = __int128(size - 1) * stride->getSExtValue();
    return !__builtin_add_overflow(minimum, std::min<__int128>(0, term),
                                   &minimum) &&
           !__builtin_add_overflow(maximum, std::max<__int128>(0, term),
                                   &maximum);
  };
  for (unsigned i = 0; i < info.axes.size(); ++i) {
    const auto &axis = info.axes[i];
    if (!axis.subAxes.empty()) {
      for (const auto &sub : axis.subAxes) {
        if (!account(sub.stride, sub.shape))
          return std::nullopt;
      }
    } else if (!account(axis.stride, info.getShape()[i])) {
      return std::nullopt;
    }
  }
  if (minimum < -(__int128(1) << (width - 1)) ||
      maximum > (__int128(1) << (width - 1)) - 1)
    return std::nullopt;
  return llvm::ConstantRange::getNonEmpty(
      llvm::APInt(width, uint64_t(minimum), true),
      llvm::APInt(width, uint64_t(maximum), true) + 1);
}
} // namespace mlir::triton::ptr::detail

namespace mlir::triton::ptr {
FailureOr<OpFoldResult> materializeOffsetAt(const OffsetInfo &info,
                                            ValueRange indices,
                                            OpBuilder &builder, Location loc) {
  OpBuilder::InsertionGuard guard(builder);
  if (!info.valueType || indices.size() != info.getShape().size() ||
      llvm::any_of(indices, [](Value v) { return !v.getType().isIndex(); }) ||
      (!info.fullOffsets && !info.hasStructuredDecomposition()))
    return failure();
  SmallVector<Value> dependencies(indices);
  if (info.fullOffsets)
    detail::collectValues(ArrayRef<OpFoldResult>{info.fullOffsets},
                          dependencies);
  else {
    detail::collectValues(ArrayRef<OpFoldResult>{info.scalarOffset},
                          dependencies);
    for (const auto &axis : info.axes) {
      if (!axis.subAxes.empty()) {
        for (const auto &sub : axis.subAxes)
          detail::collectValues(ArrayRef<OpFoldResult>{sub.stride},
                                dependencies);
      } else
        detail::collectValues(ArrayRef<OpFoldResult>{axis.stride},
                              dependencies);
    }
  }
  if (failed(checkInsertionPoint(builder, dependencies)))
    return failure();
  auto typeOf = [](OpFoldResult field) -> Type {
    if (!field)
      return {};
    if (auto value = dyn_cast<Value>(field))
      return value.getType();
    if (auto attr = dyn_cast<TypedAttr>(dyn_cast<Attribute>(field)))
      return attr.getType();
    return {};
  };
  if (info.fullOffsets) {
    OpFoldResult offset = info.fullOffsets;
    Type type = typeOf(offset);
    if (!type)
      return failure();
    if (failed(detail::getIntegerWidth(type)))
      return failure();
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      auto target = dyn_cast<RankedTensorType>(info.valueType);
      if (!target || tensor.getShape() != target.getShape() ||
          tensor.getEncoding() != target.getEncoding())
        return failure();
      auto value = detail::materialize(offset, type, builder, loc);
      if (failed(value))
        return failure();
      offset =
          builder.create<tensor::ExtractOp>(loc, *value, indices).getResult();
    }
    return detail::foldOrCreateCast(offset, builder.getIndexType(),
                                    IntegerExtensionKind::Signed, builder, loc);
  }
  detail::ConcreteArithmetic math{builder.getIndexType(), builder, loc};
  // Only a complete decomposition permits coordinate reconstruction. In
  // particular, an Unknown axis's placeholder never contributes a zero term.
  OpFoldResult sum = info.scalarOffset;
  for (unsigned i = 0; i < info.axes.size(); ++i) {
    const auto &axis = info.axes[i];
    Value coordinate = indices[i];
    auto addTerm = [&](OpFoldResult k, OpFoldResult stride) -> LogicalResult {
      auto product = math.multiply(k, stride);
      if (failed(product))
        return failure();
      auto next = math.add(sum, *product, false);
      if (failed(next))
        return failure();
      sum = *next;
      return success();
    };
    if (axis.subAxes.empty()) {
      if (failed(addTerm(coordinate, axis.stride)))
        return failure();
      continue;
    }
    // Mixed-radix coordinates are nonnegative; decompose inner to outer.
    Value remaining = coordinate;
    for (unsigned j = axis.subAxes.size(); j > 0; --j) {
      const auto &sub = axis.subAxes[j - 1];
      Value size = builder.create<arith::ConstantIndexOp>(loc, sub.shape);
      Value digit = builder.create<arith::RemUIOp>(loc, remaining, size);
      if (failed(addTerm(digit, sub.stride)))
        return failure();
      // The quotient is needed only for the remaining outer digits.
      if (j == 1)
        break;
      remaining =
          builder.create<arith::DivUIOp>(loc, remaining, size).getResult();
    }
  }
  return sum;
}
} // namespace mlir::triton::ptr
