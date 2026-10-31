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
std::optional<int64_t> extent(OpFoldResult shape) {
  auto value = getIntegerConstant(shape, 64);
  if (failed(value) || value->isNegative() || value->isZero() ||
      value->getActiveBits() > 63)
    return std::nullopt;
  return value->getSExtValue();
}
bool equalConstant(OpFoldResult lhs, OpFoldResult rhs, unsigned width) {
  if (lhs == rhs)
    return true;
  auto l = getIntegerConstant(lhs, width), r = getIntegerConstant(rhs, width);
  return succeeded(l) && succeeded(r) && *l == *r;
}
bool zero(OpFoldResult value, unsigned width) {
  auto integer = getIntegerConstant(value, width);
  return succeeded(integer) && integer->isZero();
}
LogicalResult normalizeAxis(AxisInfo &axis, int64_t size, Type scalar,
                            unsigned width, OpBuilder &builder) {
  if (!axis.subAxes) {
    if (axis.kind != AxisKind::Unknown && !axis.stride)
      return failure();
    if (axis.kind != AxisKind::Unknown)
      axis.kind =
          zero(axis.stride, width) ? AxisKind::Invariant : AxisKind::Structured;
    else
      axis.stride = builder.getIntegerAttr(scalar, 0);
    return success();
  }
  if (axis.subAxes->empty())
    return failure();
  int64_t product = 1;
  bool staticShape = true;
  for (const auto &sub : *axis.subAxes) {
    if (!sub.shape || !sub.stride)
      return failure();
    auto constant = getIntegerConstant(sub.shape, 64);
    if (succeeded(constant) && (constant->isNegative() || constant->isZero()))
      return failure();
    auto n = extent(sub.shape);
    if (!n) {
      staticShape = false;
      continue;
    }
    if (__builtin_mul_overflow(product, *n, &product))
      return failure();
  }
  if (staticShape && product != size)
    return failure();
  axis.kind = AxisKind::Unknown;
  axis.stride = builder.getIntegerAttr(scalar, 0);
  if (llvm::all_of(*axis.subAxes, [&](const SubAxisInfo &sub) {
        return zero(sub.stride, width);
      })) {
    axis.kind = AxisKind::Invariant;
    return success();
  }
  if (!staticShape)
    return success();
  // Prove that each outer coefficient equals the innermost coefficient times
  // all inner lengths. Do not emit IR just to test a symbolic equality.
  OpFoldResult stride = builder.getIntegerAttr(scalar, 0);
  for (const auto &sub : llvm::reverse(*axis.subAxes))
    if (*extent(sub.shape) != 1) {
      stride = sub.stride;
      break;
    }
  auto inner = getIntegerConstant(stride, width);
  llvm::APInt scale(width, 1);
  bool linear = true;
  for (const auto &sub : llvm::reverse(*axis.subAxes)) {
    int64_t n = *extent(sub.shape);
    if (n != 1) {
      if (scale.isOne())
        linear &= equalConstant(sub.stride, stride, width);
      else {
        auto coefficient = getIntegerConstant(sub.stride, width);
        linear &= succeeded(inner) && succeeded(coefficient) &&
                  *coefficient == *inner * scale;
      }
    }
    scale *= llvm::APInt(width, n);
  }
  if (linear) {
    axis.stride = stride;
    axis.kind =
        zero(stride, width) ? AxisKind::Invariant : AxisKind::Structured;
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
  for (const auto &sub : llvm::reverse(partition)) {
    parts.push_back({sub.shape, coefficient});
    auto n = extent(sub.shape);
    if (!n)
      return failure();
    auto scaled = arithmetic.multiply(
        coefficient, arithmetic.builder.getIntegerAttr(arithmetic.type, *n));
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
  if (!lhs.subAxes && !rhs.subAxes)
    return success();
  if (!lhs.subAxes) {
    auto expanded = expandLinear(lhs, *rhs.subAxes, arithmetic);
    if (failed(expanded))
      return failure();
    lhs.subAxes = std::move(*expanded);
  }
  if (!rhs.subAxes) {
    auto expanded = expandLinear(rhs, *lhs.subAxes, arithmetic);
    if (failed(expanded))
      return failure();
    rhs.subAxes = std::move(*expanded);
  }
  if (lhs.subAxes->size() != rhs.subAxes->size())
    return failure();
  for (auto [l, r] : llvm::zip(*lhs.subAxes, *rhs.subAxes))
    if (!equalConstant(l.shape, r.shape, 64))
      return failure();
  return success();
}
} // namespace

SmallVector<OpFoldResult> axisStrides(const OffsetInfo &info) {
  SmallVector<OpFoldResult> strides;
  for (const auto &axis : info.axes)
    strides.push_back(axis.stride);
  return strides;
}

LogicalResult normalizeAxes(OffsetInfo &info, AnalysisOptions options,
                            OpBuilder &builder) {
  auto width = getIntegerWidth(builder.getIndexType(), options);
  if (failed(width) || info.axes.size() != info.shape.size())
    return failure();
  Type scalar = builder.getIndexType();
  for (unsigned i = 0; i < info.axes.size(); ++i)
    if (failed(normalizeAxis(info.axes[i], info.shape[i], scalar, *width,
                             builder)))
      return failure();
  return success();
}

FailureOr<OffsetInfo> addOffsets(OpFoldResult complete, const OffsetInfo &lhs,
                                 const OffsetInfo &rhs, bool subtract,
                                 AnalysisOptions options, OpBuilder &builder,
                                 Location loc) {
  if (lhs.valueType != rhs.valueType || lhs.shape != rhs.shape ||
      lhs.axes.size() != rhs.axes.size())
    return failure();
  OffsetInfo info = lhs;
  info.fullOffsets = complete;
  Type scalar = builder.getIndexType();
  ConcreteArithmetic arithmetic{scalar, options, builder, loc};
  // The exact value is supplied independently; combine only proven fields.
  info.scalarOffset.reset();
  if (lhs.scalarOffset && rhs.scalarOffset) {
    auto origin =
        arithmetic.add(*lhs.scalarOffset, *rhs.scalarOffset, subtract);
    if (failed(origin))
      return failure();
    info.scalarOffset = *origin;
  }
  for (unsigned i = 0; i < lhs.axes.size(); ++i) {
    AxisInfo left = lhs.axes[i], right = rhs.axes[i];
    // Align sub-axis partitions in place instead of flattening and rebuilding
    // them. An incompatible or unknown axis loses only its own proof.
    if (failed(alignAxes(left, right, arithmetic)) ||
        (!left.subAxes &&
         (left.kind == AxisKind::Unknown || right.kind == AxisKind::Unknown))) {
      info.axes[i] =
          AxisInfo(AxisKind::Unknown, builder.getIntegerAttr(scalar, 0));
      continue;
    }
    if (left.subAxes) {
      for (auto [l, r] : llvm::zip(*left.subAxes, *right.subAxes)) {
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
  if (failed(normalizeAxes(info, options, builder)))
    return failure();
  return info;
}

FailureOr<OffsetInfo> scaleOffsets(OpFoldResult complete,
                                   const OffsetInfo &source, OpFoldResult scale,
                                   AnalysisOptions options, OpBuilder &builder,
                                   Location loc) {
  OffsetInfo info = source;
  info.fullOffsets = complete;
  Type scalar = builder.getIndexType();
  ConcreteArithmetic arithmetic{scalar, options, builder, loc};
  if (info.scalarOffset) {
    auto origin = arithmetic.multiply(*info.scalarOffset, scale);
    if (failed(origin))
      return failure();
    info.scalarOffset = *origin;
  }
  for (auto &axis : info.axes) {
    if (axis.subAxes) {
      for (auto &sub : *axis.subAxes) {
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
  if (failed(normalizeAxes(info, options, builder)))
    return failure();
  return info;
}

FailureOr<OffsetInfo> broadcastOffsets(OpFoldResult complete, Type target,
                                       const OffsetInfo &source,
                                       OpBuilder &builder) {
  OffsetInfo info = source;
  info.valueType = target;
  info.shape = getLogicalShape(target);
  info.fullOffsets = complete;
  auto axes = broadcastAxisFacts(
      ArrayRef<AxisInfo>(source.axes), ArrayRef<int64_t>(source.shape),
      ArrayRef<int64_t>(info.shape),
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
  info.shape = getLogicalShape(target);
  info.fullOffsets = complete;
  if (info.shape.size() != source.shape.size() + 1)
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
                                     AnalysisOptions options,
                                     OpBuilder &builder, Location loc) {
  auto from = dyn_cast<RankedTensorType>(source.valueType);
  auto width = getIntegerWidth(builder.getIndexType(), options);
  if (!from || !from.hasStaticShape() || !target.hasStaticShape() ||
      failed(width) || from.getNumElements() != target.getNumElements() ||
      from.getElementType() != target.getElementType() ||
      llvm::any_of(from.getShape(), [](int64_t size) { return size <= 0; }) ||
      llvm::any_of(target.getShape(), [](int64_t size) { return size <= 0; }))
    return failure();
  OffsetInfo info = source;
  info.valueType = target;
  info.shape = llvm::to_vector(target.getShape());
  info.fullOffsets = complete;
  Type scalar = builder.getIndexType();
  auto zero = builder.getIntegerAttr(scalar, 0);
  info.axes.assign(target.getRank(), AxisInfo(AxisKind::Unknown, zero));
  for (unsigned i = 0; i < info.shape.size(); ++i)
    if (info.shape[i] == 1)
      info.axes[i] = AxisInfo(AxisKind::Invariant, zero);
  ConcreteArithmetic arithmetic{scalar, options, builder, loc};
  // Flatten to ordered digits, then repartition them from the minor end.
  // Unknown digits remain unknown; a shape change never invents an origin.
  struct Digit {
    int64_t size;
    OpFoldResult stride;
  };
  SmallVector<Digit> digits;
  for (unsigned i = 0; i < source.axes.size(); ++i) {
    const auto &axis = source.axes[i];
    if (source.shape[i] == 1)
      continue;
    if (axis.kind != AxisKind::Unknown) {
      digits.push_back({source.shape[i], axis.stride});
    } else if (axis.subAxes && llvm::all_of(*axis.subAxes, [](const auto &s) {
                 return extent(s.shape).has_value();
               })) {
      for (const auto &sub : *axis.subAxes)
        if (*extent(sub.shape) != 1)
          digits.push_back({*extent(sub.shape), sub.stride});
    } else {
      digits.push_back({source.shape[i], {}});
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
      auto outer = getIntegerConstant(contiguous.back().stride, *width);
      auto inner = getIntegerConstant(digit.stride, *width);
      int64_t size;
      if (succeeded(outer) && succeeded(inner) &&
          *outer == *inner * llvm::APInt(64, digit.size).zextOrTrunc(*width) &&
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
        if (failed(normalizeAxes(info, options, builder)))
          return failure();
        return info;
      }
      known &= bool(digit.stride);
      parts.push_back({builder.getIndexAttr(take), digit.stride});
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
  if (failed(normalizeAxes(info, options, builder)))
    return failure();
  return info;
}

FailureOr<OffsetInfo>
sliceOffsets(OpFoldResult complete, RankedTensorType target,
             const OffsetInfo &source, ArrayRef<OpFoldResult> starts,
             ArrayRef<OpFoldResult> steps, ArrayRef<bool> dropped,
             AnalysisOptions options, OpBuilder &builder, Location loc) {
  auto integerWidth = getIntegerWidth(builder.getIndexType(), options);
  if (failed(integerWidth) || !target.hasStaticShape() ||
      target.getElementType() != getIntegerElementType(source.valueType) ||
      llvm::count(dropped, false) != target.getRank() ||
      starts.size() != source.axes.size() || steps.size() != starts.size() ||
      dropped.size() != starts.size())
    return failure();
  OffsetInfo info = source;
  info.valueType = target;
  info.shape = llvm::to_vector(target.getShape());
  info.fullOffsets = complete;
  info.axes.clear();
  Type scalar = builder.getIndexType();
  unsigned width = *integerWidth;
  ConcreteArithmetic arithmetic{scalar, options, builder, loc};
  auto convert = [&](OpFoldResult index) -> FailureOr<OpFoldResult> {
    auto constant = getIntegerConstant(index, 64);
    if (succeeded(constant))
      return OpFoldResult(
          scalarIntegerAttr(scalar, constant->sextOrTrunc(width), builder));
    return foldOrCreateCast(index, scalar, CastKind::Signed, options, builder,
                            loc);
  };
  unsigned next = 0;
  for (unsigned i = 0; i < starts.size(); ++i) {
    AxisInfo axis = source.axes[i];
    auto start = convert(starts[i]);
    auto step = convert(steps[i]);
    bool unchanged = zero(starts[i], 64) &&
                     equalConstant(steps[i], builder.getIndexAttr(1), 64) &&
                     !dropped[i] && target.getDimSize(next) == source.shape[i];
    if (info.scalarOffset && !zero(starts[i], 64)) {
      if (axis.kind == AxisKind::Unknown || failed(start)) {
        info.scalarOffset.reset();
      } else {
        auto shift = arithmetic.multiply(*start, axis.stride);
        if (failed(shift))
          return failure();
        auto origin = arithmetic.add(*info.scalarOffset, *shift, false);
        if (failed(origin))
          return failure();
        info.scalarOffset = *origin;
      }
    }
    if (dropped[i])
      continue;
    if (!unchanged) {
      axis.subAxes.reset();
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
  if (failed(normalizeAxes(info, options, builder)))
    return failure();
  return info;
}

bool decompositionFitsSigned(const OffsetInfo &info, unsigned width) {
  return info.isUniform() ||
         constantDecompositionRange(info, width).has_value();
}

std::optional<llvm::ConstantRange>
constantDecompositionRange(const OffsetInfo &info, unsigned width) {
  if (!info.hasCompleteForm() || width == 0 || width > 64)
    return std::nullopt;
  // Descriptor coefficients are index values. Do not truncate them to the
  // source width: an i8 tensor can legitimately have index stride 200.
  auto origin = getIntegerConstant(*info.scalarOffset, 64);
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
    if (axis.subAxes) {
      for (const auto &sub : *axis.subAxes) {
        auto size = extent(sub.shape);
        if (!size || !account(sub.stride, *size))
          return std::nullopt;
      }
    } else if (!account(axis.stride, info.shape[i])) {
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
                                            OpBuilder &builder, Location loc,
                                            AnalysisOptions options) {
  OpBuilder::InsertionGuard guard(builder);
  if (!options.indexBitWidth ||
      failed(detail::getIntegerWidth(builder.getIndexType(), options)) ||
      !info.valueType || indices.size() != info.shape.size() ||
      info.shape != detail::getLogicalShape(info.valueType) ||
      llvm::any_of(indices, [](Value v) { return !v.getType().isIndex(); }) ||
      (!info.fullOffsets && info.addressTerms.empty() &&
       !info.hasCompleteForm()))
    return failure();
  SmallVector<Value> dependencies(indices);
  if (info.fullOffsets)
    detail::collectValues(ArrayRef<OpFoldResult>{info.fullOffsets},
                          dependencies);
  else if (!info.addressTerms.empty())
    detail::collectValues(info.addressTerms, dependencies);
  else {
    detail::collectValues(ArrayRef<OpFoldResult>{*info.scalarOffset},
                          dependencies);
    for (const auto &axis : info.axes) {
      if (axis.subAxes) {
        for (const auto &sub : *axis.subAxes)
          detail::collectValues(ArrayRef<OpFoldResult>{sub.shape, sub.stride},
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
  auto terms = info.fullOffsets ? SmallVector<OpFoldResult>{info.fullOffsets}
                                : info.addressTerms;
  // Validate every term before emitting extraction or conversion helpers.
  for (OpFoldResult term : terms) {
    Type type = typeOf(term);
    if (!type)
      return failure();
    auto width = detail::getIntegerWidth(type, options);
    if (failed(width) || *width > *options.indexBitWidth)
      return failure();
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      auto target = dyn_cast<RankedTensorType>(info.valueType);
      if (!target || tensor.getShape() != target.getShape() ||
          tensor.getEncoding() != target.getEncoding())
        return failure();
    }
  }
  detail::ConcreteArithmetic math{builder.getIndexType(), options, builder,
                                  loc};
  OpFoldResult sum = builder.getIndexAttr(0);
  if (!terms.empty()) {
    for (OpFoldResult term : terms) {
      Type type = typeOf(term);
      if (isa<RankedTensorType>(type)) {
        auto tensor = detail::materialize(term, type, builder, loc);
        if (failed(tensor))
          return failure();
        term = builder.create<tensor::ExtractOp>(loc, *tensor, indices)
                   .getResult();
      }
      auto scalar = detail::foldOrCreateCast(term, builder.getIndexType(),
                                             detail::CastKind::Signed, options,
                                             builder, loc);
      if (failed(scalar))
        return failure();
      auto next = math.add(sum, *scalar, false);
      if (failed(next))
        return failure();
      sum = *next;
    }
    return sum;
  }
  // Only a complete decomposition permits coordinate reconstruction. In
  // particular, an Unknown axis's placeholder never contributes a zero term.
  sum = *info.scalarOffset;
  for (unsigned i = 0; i < info.axes.size(); ++i) {
    const auto &axis = info.axes[i];
    auto coordinate = detail::foldOrCreateCast(
        indices[i], builder.getIndexType(), detail::CastKind::Signed, options,
        builder, loc);
    if (failed(coordinate))
      return failure();
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
    if (!axis.subAxes) {
      if (failed(addTerm(*coordinate, axis.stride)))
        return failure();
      continue;
    }
    // Mixed-radix coordinates are nonnegative; decompose inner to outer.
    auto remaining =
        detail::materialize(*coordinate, builder.getIndexType(), builder, loc);
    if (failed(remaining))
      return failure();
    for (const auto &sub : llvm::reverse(*axis.subAxes)) {
      auto extent = detail::foldOrCreateCast(sub.shape, builder.getIndexType(),
                                             detail::CastKind::Signed, options,
                                             builder, loc);
      if (failed(extent))
        return failure();
      auto size =
          detail::materialize(*extent, builder.getIndexType(), builder, loc);
      if (failed(size))
        return failure();
      Value digit = builder.create<arith::RemUIOp>(loc, *remaining, *size);
      if (failed(addTerm(digit, sub.stride)))
        return failure();
      remaining =
          builder.create<arith::DivUIOp>(loc, *remaining, *size).getResult();
    }
  }
  return sum;
}
} // namespace mlir::triton::ptr
