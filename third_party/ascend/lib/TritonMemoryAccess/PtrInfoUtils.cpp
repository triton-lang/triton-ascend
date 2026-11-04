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
#include <limits>

namespace mlir::triton::ptr::detail {
namespace {
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
} // namespace

// Axis queries and normalization.
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

// Arithmetic combinations.
FailureOr<OffsetInfo> addOffsets(OpFoldResult complete, const OffsetInfo &lhs,
                                 const OffsetInfo &rhs, bool subtract,
                                 OpBuilder &builder, Location loc) {
  if (lhs.valueType != rhs.valueType || lhs.axes.size() != rhs.axes.size())
    return failure();
  OffsetInfo info = lhs;
  info.fullOffsets = complete;
  Type scalar = builder.getIndexType();
  ConcreteArithmetic arithmetic{scalar, builder, loc};
  // The exact value is supplied independently; combine only proven fields.
  info.scalarOffset = {};
  if (lhs.scalarOffset && rhs.scalarOffset) {
    auto origin = arithmetic.add(lhs.scalarOffset, rhs.scalarOffset, subtract);
    if (failed(origin))
      return failure();
    info.scalarOffset = *origin;
  }
  for (unsigned i = 0; i < lhs.axes.size(); ++i) {
    AxisInfo left = lhs.axes[i], right = rhs.axes[i];
    // Align sub-axis partitions in place instead of flattening and rebuilding
    // them. An incompatible or unknown axis loses only its own proof.
    if (failed(alignAxes(left, right, arithmetic)) ||
        (left.subAxes.empty() &&
         (left.kind == AxisKind::Unknown || right.kind == AxisKind::Unknown))) {
      info.axes[i] =
          AxisInfo(AxisKind::Unknown, builder.getIntegerAttr(scalar, 0));
      continue;
    }
    if (!left.subAxes.empty()) {
      for (auto [l, r] : llvm::zip(left.subAxes, right.subAxes)) {
        auto stride = arithmetic.add(l.stride, r.stride, subtract);
        if (failed(stride))
          return failure();
        l.stride = *stride;
      }
    } else {
      auto stride = arithmetic.add(left.stride, right.stride, subtract);
      if (failed(stride))
        return failure();
      left.stride = *stride;
    }
    info.axes[i] = std::move(left);
  }
  if (failed(normalizeAxes(info, builder)))
    return failure();
  return info;
}

FailureOr<OffsetInfo> scaleOffsets(OpFoldResult complete,
                                   const OffsetInfo &source, OpFoldResult scale,
                                   OpBuilder &builder, Location loc) {
  OffsetInfo info = source;
  info.fullOffsets = complete;
  Type scalar = builder.getIndexType();
  ConcreteArithmetic arithmetic{scalar, builder, loc};
  if (info.scalarOffset) {
    auto origin = arithmetic.multiply(info.scalarOffset, scale);
    if (failed(origin))
      return failure();
    info.scalarOffset = *origin;
  }
  for (auto &axis : info.axes) {
    if (!axis.subAxes.empty()) {
      for (auto &sub : axis.subAxes) {
        auto stride = arithmetic.multiply(sub.stride, scale);
        if (failed(stride))
          return failure();
        sub.stride = *stride;
      }
    } else if (axis.kind != AxisKind::Unknown) {
      auto stride = arithmetic.multiply(axis.stride, scale);
      if (failed(stride))
        return failure();
      axis.stride = *stride;
    }
  }
  if (failed(normalizeAxes(info, builder)))
    return failure();
  return info;
}

// Shape transformations.
FailureOr<OffsetInfo> broadcastOffsets(OpFoldResult complete, Type target,
                                       const OffsetInfo &source,
                                       OpBuilder &builder) {
  OffsetInfo info = source;
  info.valueType = target;
  info.fullOffsets = complete;
  auto axes = broadcastAxisFacts(
      ArrayRef<AxisInfo>(source.axes), source.getShape(), info.getShape(),
      [&] { return AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)); });
  if (failed(axes))
    return failure();
  info.axes = std::move(*axes);
  return info;
}
FailureOr<OffsetInfo> expandDimsOffsets(OpFoldResult complete, Type target,
                                        const OffsetInfo &source, unsigned axis,
                                        OpBuilder &builder) {
  OffsetInfo info = source;
  info.valueType = target;
  info.fullOffsets = complete;
  if (info.getShape().size() != source.getShape().size() + 1)
    return failure();
  auto axes = expandAxisFacts(ArrayRef<AxisInfo>(source.axes), axis, [&] {
    return AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0));
  });
  if (failed(axes))
    return failure();
  info.axes = std::move(*axes);
  return info;
}

FailureOr<OffsetInfo> reshapeOffsets(OpFoldResult complete,
                                     RankedTensorType target,
                                     const OffsetInfo &source,
                                     OpBuilder &builder, Location loc) {
  auto from = dyn_cast<RankedTensorType>(source.valueType);
  if (!from || !from.hasStaticShape() || !target.hasStaticShape() ||
      from.getNumElements() != target.getNumElements() ||
      from.getElementType() != target.getElementType() ||
      llvm::any_of(from.getShape(), [](int64_t size) { return size <= 0; }) ||
      llvm::any_of(target.getShape(), [](int64_t size) { return size <= 0; }))
    return failure();
  OffsetInfo info = source;
  info.valueType = target;
  info.fullOffsets = complete;
  Type scalar = builder.getIndexType();
  auto zero = builder.getIntegerAttr(scalar, 0);
  info.axes.assign(target.getRank(), AxisInfo(AxisKind::Unknown, zero));
  for (unsigned i = 0; i < info.getShape().size(); ++i)
    if (info.getShape()[i] == 1)
      info.axes[i] = AxisInfo(AxisKind::Invariant, zero);
  ConcreteArithmetic arithmetic{scalar, builder, loc};
  // Flatten to ordered digits, then repartition them from the minor end.
  // Unknown digits remain unknown; a shape change never invents an origin.
  struct Digit {
    int64_t size;
    OpFoldResult stride;
  };
  SmallVector<Digit> digits;
  for (unsigned i = 0; i < source.axes.size(); ++i) {
    const auto &axis = source.axes[i];
    if (source.getShape()[i] == 1)
      continue;
    if (axis.kind != AxisKind::Unknown) {
      digits.push_back({source.getShape()[i], axis.stride});
    } else if (!axis.subAxes.empty()) {
      for (const auto &sub : axis.subAxes)
        if (sub.shape != 1)
          digits.push_back({sub.shape, sub.stride});
    } else {
      digits.push_back({source.getShape()[i], {}});
    }
  }
  // Coalesce proven contiguous digits before introducing new boundaries.
  // For example [2, 6] with coefficients [6, 1] is one length-12 digit,
  // which can be repartitioned as [3, 4]. Zero coefficients also coalesce,
  // preserving uniform tensors. Compare in the source arithmetic width and
  // do not generate helpers merely to guess a symbolic stride equality.
  SmallVector<Digit> contiguous;
  for (const auto &digit : digits) {
    if (!contiguous.empty() && contiguous.back().stride && digit.stride) {
      auto outer = getIntegerConstant(contiguous.back().stride, 64);
      auto inner = getIntegerConstant(digit.stride, 64);
      int64_t size;
      if (succeeded(outer) && succeeded(inner) &&
          *outer == *inner * llvm::APInt(64, digit.size) &&
          !__builtin_mul_overflow(contiguous.back().size, digit.size, &size)) {
        contiguous.back() = {size, digit.stride};
        continue;
      }
    }
    contiguous.push_back(digit);
  }
  digits = std::move(contiguous);
  for (int i = target.getRank() - 1; i >= 0; --i) {
    int64_t remaining = target.getDimSize(i);
    if (remaining <= 0)
      return failure();
    SmallVector<SubAxisInfo> parts;
    bool known = true;
    while (remaining > 1) {
      if (digits.empty())
        return failure();
      auto &digit = digits.back();
      int64_t take = std::min(remaining, digit.size);
      if (remaining % take || digit.size % take) {
        // Only this and the remaining outer axes are unproved. Completed
        // inner axes retain their independent coordinate relationships.
        if (failed(normalizeAxes(info, builder)))
          return failure();
        return info;
      }
      known &= bool(digit.stride);
      parts.push_back({take, digit.stride});
      remaining /= take;
      digit.size /= take;
      if (digit.size == 1) {
        digits.pop_back();
      } else if (digit.stride) {
        auto scaled = arithmetic.multiply(digit.stride,
                                          builder.getIntegerAttr(scalar, take));
        if (failed(scaled))
          return failure();
        digit.stride = *scaled;
      }
    }
    if (!known)
      continue;
    if (parts.empty()) {
      info.axes[i] = AxisInfo(AxisKind::Invariant, zero);
    } else {
      std::reverse(parts.begin(), parts.end());
      info.axes[i].subAxes = std::move(parts);
    }
  }
  if (failed(normalizeAxes(info, builder)))
    return failure();
  return info;
}

FailureOr<OffsetInfo>
sliceOffsets(OpFoldResult complete, RankedTensorType target,
             const OffsetInfo &source, ArrayRef<OpFoldResult> starts,
             ArrayRef<OpFoldResult> steps, ArrayRef<bool> dropped,
             OpBuilder &builder, Location loc) {
  if (!target.hasStaticShape() ||
      target.getElementType() !=
          detail::getElementTypeOrSelf(source.valueType) ||
      llvm::count(dropped, false) != target.getRank() ||
      starts.size() != source.axes.size() || steps.size() != starts.size() ||
      dropped.size() != starts.size())
    return failure();
  OffsetInfo info = source;
  info.valueType = target;
  info.fullOffsets = complete;
  info.axes.clear();
  Type scalar = builder.getIndexType();
  ConcreteArithmetic arithmetic{scalar, builder, loc};
  auto convert = [&](OpFoldResult index) -> FailureOr<OpFoldResult> {
    auto constant = getIntegerConstant(index, 64);
    if (succeeded(constant))
      return OpFoldResult(builder.getIntegerAttr(scalar, *constant));
    return foldOrCreateCast(index, scalar, IntegerExtensionKind::Signed,
                            builder, loc);
  };
  unsigned next = 0;
  for (unsigned i = 0; i < starts.size(); ++i) {
    AxisInfo axis = source.axes[i];
    auto start = convert(starts[i]);
    auto step = convert(steps[i]);
    bool unchanged =
        zero(starts[i]) && equalConstant(steps[i], builder.getIndexAttr(1)) &&
        !dropped[i] && target.getDimSize(next) == source.getShape()[i];
    if (info.scalarOffset && !zero(starts[i])) {
      if (axis.kind == AxisKind::Unknown || failed(start)) {
        info.scalarOffset = {};
      } else {
        auto shift = arithmetic.multiply(*start, axis.stride);
        if (failed(shift))
          return failure();
        auto origin = arithmetic.add(info.scalarOffset, *shift, false);
        if (failed(origin))
          return failure();
        info.scalarOffset = *origin;
      }
    }
    if (dropped[i])
      continue;
    if (!unchanged) {
      axis.subAxes.clear();
      if (axis.kind != AxisKind::Unknown && succeeded(step)) {
        auto stride = arithmetic.multiply(axis.stride, *step);
        if (failed(stride))
          return failure();
        axis.stride = *stride;
      } else {
        axis = AxisInfo(AxisKind::Unknown, builder.getIntegerAttr(scalar, 0));
      }
    }
    if (target.getDimSize(next++) == 1)
      axis = AxisInfo(AxisKind::Invariant, builder.getIntegerAttr(scalar, 0));
    info.axes.push_back(std::move(axis));
  }
  if (failed(normalizeAxes(info, builder)))
    return failure();
  return info;
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
