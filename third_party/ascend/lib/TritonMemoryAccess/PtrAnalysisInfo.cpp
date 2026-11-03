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

#include "TritonMemoryAccess/PtrAnalysisInfo.h"
#include "TritonMemoryAccess/PtrInfoUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::ptr {

// Result queries and construction.
bool OffsetInfo::hasStructuredForm() const {
  return scalarOffset && llvm::all_of(axes, [](const AxisInfo &axis) {
           return axis.kind != AxisKind::Unknown;
         });
}

bool OffsetInfo::hasStructuredDecomposition() const {
  return scalarOffset && llvm::all_of(axes, [](const AxisInfo &axis) {
           return axis.kind != AxisKind::Unknown || !axis.subAxes.empty();
         });
}

bool OffsetInfo::isUniform() const {
  return hasStructuredForm() && llvm::all_of(axes, [](const AxisInfo &axis) {
           return axis.kind == AxisKind::Invariant;
         });
}

Type PtrInfo::getElementType() const {
  return cast<triton::PointerType>(base.getType()).getPointeeType();
}

unsigned PtrInfo::getAddressSpace() const {
  return cast<triton::PointerType>(base.getType()).getAddressSpace();
}

FailureOr<OffsetInfo> buildOffsetInfo(Value value, OpBuilder &builder) {
  if (!value)
    return failure();
  Type type = value.getType();
  // Getting the width also checks for a supported integer/index element type.
  if (failed(detail::getIntegerWidth(type)))
    return failure();
  OffsetInfo result;
  result.valueType = type;
  result.fullOffsets = value;
  result.axes.assign(result.getShape().size(),
                     AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  return result;
}

namespace {

// Block descriptor calculations reuse the shared coordinate formulas.
FailureOr<OpFoldResult> addScaledCoordinates(OpFoldResult origin,
                                             ArrayRef<OpFoldResult> coordinates,
                                             ArrayRef<OpFoldResult> strides,
                                             OpBuilder &builder, Location loc) {
  detail::ConcreteArithmetic arithmetic{builder.getIndexType(), builder, loc};
  return addScaledCoordinateTerms(origin, coordinates, strides, arithmetic);
}

// Local checks shared by binding validation.
bool validOfrType(OpFoldResult value, Type expected) {
  if (!value)
    return false;
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType() == expected;
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) && cast<TypedAttr>(attr).getType() == expected;
  return false;
}

bool isIndex(OpFoldResult value) {
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType().isIndex();
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) && cast<TypedAttr>(attr).getType().isIndex();
  return false;
}

bool validOffset(Type resultType, const OffsetInfo &components,
                 bool pointer = false) {
  Type scalarType = IndexType::get(resultType.getContext());
  if (components.valueType != resultType ||
      components.axes.size() != components.getShape().size() ||
      failed(detail::getIntegerWidth(resultType)) ||
      (!components.fullOffsets &&
       (!pointer || !components.hasStructuredDecomposition())) ||
      (components.scalarOffset &&
       !validOfrType(components.scalarOffset, scalarType)))
    return false;
  if (components.fullOffsets) {
    // Integer results retain their source type. Pointer descriptors use index,
    // but their complete offset tensor can use an integer storage type.
    Type type;
    if (auto value = dyn_cast<Value>(components.fullOffsets))
      type = value.getType();
    else if (auto attr = dyn_cast<TypedAttr>(
                 dyn_cast<Attribute>(components.fullOffsets)))
      type = attr.getType();
    if (!type)
      return false;
    if (failed(detail::getIntegerWidth(type)) ||
        (!pointer && type != resultType))
      return false;
    auto tensor = dyn_cast<RankedTensorType>(type);
    auto target = dyn_cast<RankedTensorType>(resultType);
    if (bool(tensor) != bool(target) ||
        (tensor && (tensor.getShape() != target.getShape() ||
                    tensor.getEncoding() != target.getEncoding())))
      return false;
  }
  for (const auto &axis : components.axes) {
    if (!validOfrType(axis.stride, scalarType))
      return false;
    for (const auto &sub : axis.subAxes)
      if (!validOfrType(sub.stride, scalarType))
        return false;
  }
  OpBuilder builder(resultType.getContext());
  OffsetInfo normalized = components;
  if (failed(detail::normalizeAxes(normalized, builder)))
    return false;
  for (unsigned i = 0; i < components.axes.size(); ++i) {
    if (normalized.axes[i].kind != components.axes[i].kind)
      return false;
    if (components.axes[i].kind != AxisKind::Unknown &&
        normalized.axes[i].stride != components.axes[i].stride) {
      auto a = detail::getIntegerConstant(normalized.axes[i].stride, 64);
      auto b = detail::getIntegerConstant(components.axes[i].stride, 64);
      if (failed(a) || failed(b) || *a != *b)
        return false;
    }
  }
  return true;
}

} // namespace

namespace detail {

// Pointer type queries.
bool isOrdinaryPtrType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  auto ptr = dyn_cast<triton::PointerType>(type);
  return ptr && !isa<ShapedType>(ptr.getPointeeType());
}

bool isBlockPtrType(Type type) {
  auto ptr = dyn_cast<triton::PointerType>(type);
  return ptr && isa<RankedTensorType>(ptr.getPointeeType());
}

// Initial result construction.
OffsetInfo buildUniformOffsetInfo(Value value, OpBuilder &builder,
                                  OpFoldResult scalar) {
  OffsetInfo info;
  info.valueType = value.getType();
  info.fullOffsets = value;
  info.axes.assign(info.getShape().size(),
                   AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)));
  setUniformOrigin(info, value, builder, scalar);
  return info;
}

// Build integer facts from the value itself, without analyzing its operands.
FailureOr<PtrInfo> buildIntegerInfo(Value value, OpBuilder &builder) {
  Type type = value.getType();
  // Require a supported integer/index type, including tensor element types.
  if (failed(getIntegerWidth(type)))
    return failure();

  // Scalars, singleton tensors and splat constants are uniform.
  OpFoldResult scalar;
  if (auto constant = getIntegerConstant(value, 64); succeeded(constant))
    scalar = builder.getIndexAttr(constant->getSExtValue());
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (scalar || !tensor ||
      (tensor.hasStaticShape() && tensor.getNumElements() == 1)) {
    // Scalars already supply an origin. Tensor extraction requires an active
    // memory lane or an `other` value for masked-out load results.
    Value mask;
    if (tensor) {
      if (auto load = value.getDefiningOp<triton::LoadOp>();
          load && !load.getOther())
        mask = load.getMask();
      else if (auto atomic = value.getDefiningOp<triton::AtomicRMWOp>())
        mask = atomic.getMask();
    }
    bool extractSingleton = true;
    if (mask) {
      auto constantMask = getIntegerConstant(mask, 1);
      extractSingleton = succeeded(constantMask) && constantMask->isAllOnes();
    }
    if (extractSingleton)
      return PtrInfo{{}, buildUniformOffsetInfo(value, builder, scalar)};
  }

  // Otherwise retain the full value without assuming an origin or axis strides.
  auto info = buildOffsetInfo(value, builder);
  if (failed(info))
    return failure();
  return PtrInfo{{}, std::move(*info)};
}

PtrInfo buildScalarPtrInfo(Value value, OpBuilder &builder) {
  if (!isa<triton::PointerType>(value.getType()) ||
      !isOrdinaryPtrType(value.getType()))
    return PtrInfo{};
  PtrInfo info;
  info.base = value;
  info.offsets.valueType = builder.getIndexType();
  info.offsets.fullOffsets = builder.getIndexAttr(0);
  info.offsets.scalarOffset = builder.getIndexAttr(0);
  return PtrInfo(std::move(info));
}

// Integer offset combinations.
PtrInfo combineElementwise(Value value, const PtrInfo &lhs, const PtrInfo &rhs,
                           OpBuilder &builder) {
  // Require matching shapes before combining the two sets of axis facts.
  const auto *left = &lhs, *right = &rhs;
  auto offsets = buildOffsetInfo(value, builder);
  if (failed(offsets) || left->offsets.getShape() != offsets->getShape() ||
      right->offsets.getShape() != offsets->getShape())
    return PtrInfo{};
  // Only axes invariant in both inputs remain invariant in the result.
  Type scalar = builder.getIndexType();
  for (unsigned i = 0; i < offsets->axes.size(); ++i) {
    bool invariant = left->offsets.axes[i].kind == AxisKind::Invariant &&
                     right->offsets.axes[i].kind == AxisKind::Invariant;
    offsets->axes[i] =
        AxisInfo(invariant ? AxisKind::Invariant : AxisKind::Unknown,
                 builder.getIntegerAttr(scalar, 0));
  }
  // Recover an origin when every result axis is invariant.
  setUniformOrigin(*offsets, value, builder);
  return PtrInfo{{}, std::move(*offsets)};
}

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
  // Scale the known origin before updating the axis contributions.
  if (info.scalarOffset) {
    auto origin = arithmetic.multiply(info.scalarOffset, scale);
    if (failed(origin))
      return failure();
    info.scalarOffset = *origin;
  }
  // Preserve unknown axes while scaling each known stride or sub-axis.
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

PtrInfo refineDivRem(Value value, const PtrInfo &lhsInfo,
                     const PtrInfo &rhsInfo, const PtrInfo &fallback,
                     bool divide, OpBuilder &builder) {
  // Require a positive uniform divisor and determine the source range.
  if (!rhsInfo.offsets.isUniform())
    return fallback;
  auto width = detail::getIntegerWidth(lhsInfo.offsets.valueType);
  if (failed(width))
    return fallback;
  auto constant = detail::getIntegerConstant(rhsInfo.offsets.scalarOffset, 64);
  if (failed(constant) || constant->isZero() || constant->isNegative())
    return fallback;
  auto divisor = *constant;
  uint64_t d = divisor.getZExtValue();
  auto bounds = detail::constantDecompositionRange(lhsInfo.offsets, *width);
  bool nonnegative = bounds && bounds->isAllNonNegative();
  bool signedOp = isa<arith::DivSIOp, arith::RemSIOp>(value.getDefiningOp());
  // Exact division needs no residual splitting: negative or dynamic origins
  // work too. Unsigned division can reuse signed descriptors only above zero.
  if ((signedOp || nonnegative) && canDivideDecomposition(lhsInfo.offsets, d)) {
    OffsetInfo info = lhsInfo.offsets;
    info.fullOffsets = value;
    if (divide) {
      divideDecomposition(info, d, builder, value.getLoc());
    } else {
      info.scalarOffset = builder.getIndexAttr(0);
      info.axes.assign(info.getShape().size(),
                       AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)));
    }
    if (succeeded(detail::normalizeAxes(info, builder)))
      return PtrInfo{{}, std::move(info)};
    return fallback;
  }
  // Splitting residual digits still needs a nonnegative constant
  // decomposition; otherwise signed division may change across zero.
  if (!nonnegative)
    return fallback;
  auto origin = detail::getIntegerConstant(lhsInfo.offsets.scalarOffset, 64);
  if (failed(origin) || !origin->urem(divisor).isZero())
    return fallback;
  OffsetInfo info = lhsInfo.offsets;
  info.fullOffsets = value;
  Type scalar = builder.getIndexType();
  info.scalarOffset = builder.getIntegerAttr(
      scalar, divide ? origin->udiv(divisor) : llvm::APInt(64, 0));
  __int128 residualMaximum = 0;
  for (unsigned i = 0; i < info.axes.size(); ++i) {
    auto &axis = info.axes[i];
    SmallVector<SubAxisInfo> input = axis.subAxes;
    if (input.empty())
      input.push_back({info.getShape()[i], axis.stride});
    SmallVector<SubAxisInfo> output;
    for (const auto &sub : input) {
      auto stride = detail::getIntegerConstant(sub.stride, 64);
      if (sub.shape <= 0 || failed(stride) || stride->isNegative())
        return fallback;
      uint64_t size = sub.shape, c = stride->getZExtValue();
      auto push = [&](uint64_t length, uint64_t coefficient) {
        output.push_back(
            {static_cast<int64_t>(length),
             builder.getIntegerAttr(scalar, llvm::APInt(64, coefficient))});
      };
      if (c % d == 0) {
        push(size, divide ? c / d : 0);
      } else if (d % c == 0 && size % (d / c) == 0) {
        uint64_t inner = d / c;
        push(size / inner, divide ? 1 : 0);
        push(inner, divide ? 0 : c);
        residualMaximum += d - c;
      } else if (__int128(size - 1) * c < d) {
        push(size, divide ? 0 : c);
        residualMaximum += __int128(size - 1) * c;
      } else {
        return fallback;
      }
    }
    axis.subAxes = std::move(output);
  }
  // Multiple residual digits may carry into the quotient. Only accept their
  // independent decomposition when their total stays below the divisor.
  if (residualMaximum >= d || failed(detail::normalizeAxes(info, builder)))
    return fallback;
  return PtrInfo{{}, std::move(info)};
}

// Offset shape transformations.
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
  // Require matching slice metadata and a statically shaped result.
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
  // Convert slice coordinates to the descriptor index domain.
  auto convert = [&](OpFoldResult index) -> FailureOr<OpFoldResult> {
    auto constant = getIntegerConstant(index, 64);
    if (succeeded(constant))
      return OpFoldResult(builder.getIntegerAttr(scalar, *constant));
    return foldOrCreateCast(index, scalar, IntegerExtensionKind::Signed,
                            builder, loc);
  };
  // Shift the origin and retain or scale each surviving axis.
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

// Ordinary pointer offset accumulation.
FailureOr<PtrInfo> addPtrOffsets(Value value, const PtrInfo &parent,
                                 const OffsetInfo &delta, OpBuilder &builder) {
  // Pointer increments preserve the pointee type and use address-domain
  // offsets.
  if (parent.getElementType() !=
      cast<triton::PointerType>(detail::getElementTypeOrSelf(value.getType()))
          .getPointeeType())
    return failure();
  Type type = addressValueType(value.getType(), builder);
  OffsetInfo lhs = parent.offsets, rhs = delta;
  OpFoldResult complete;
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    Type storage = addressStorageType(tensor, builder);
    // Widen each addptr increment before summing: two address increments do
    // not have the wrapping semantics of a single narrow integer addition.
    auto left = materializeFullOffset(lhs, storage, builder, value.getLoc());
    auto right = materializeFullOffset(rhs, storage, builder, value.getLoc());
    if (failed(left) || failed(right))
      return failure();
    complete = builder.create<arith::AddIOp>(value.getLoc(), *left, *right)
                   .getResult();
    for (auto *info : {&lhs, &rhs}) {
      if (!isa<RankedTensorType>(info->valueType))
        info->axes.assign(tensor.getRank(), AxisInfo(AxisKind::Invariant,
                                                     builder.getIndexAttr(0)));
      info->valueType = type;
    }
  } else {
    // Scalar pointers need only the sum of their two scalar offsets.
    auto left = materializeOffsetAt(lhs, {}, builder, value.getLoc());
    auto right = materializeOffsetAt(rhs, {}, builder, value.getLoc());
    if (failed(left) || failed(right))
      return failure();
    auto sum = foldOrCreateBinary(BinaryKind::Add, *left, *right, type, builder,
                                  value.getLoc());
    if (failed(sum))
      return failure();
    PtrInfo info = parent;
    info.offsets.valueType = type;
    info.offsets.fullOffsets = *sum;
    info.offsets.scalarOffset = *sum;
    return info;
  }
  // Combine the descriptor facts alongside the independently widened full sum.
  auto offsets = addOffsets(complete, lhs, rhs, false, builder, value.getLoc());
  if (failed(offsets))
    return failure();
  PtrInfo info = parent;
  info.offsets = std::move(*offsets);
  return info;
}

// Selection and insertion of analyzed results.
PtrInfo combineSelectInfo(arith::SelectOp op, const PtrInfo &lhsInfo,
                          const PtrInfo &rhsInfo, OpBuilder &builder) {
  Value value = op.getResult();
  // Integer selections can retain origins and strides from their branches.
  if (!lhsInfo.base) {
    const auto &lhs = lhsInfo, &rhs = rhsInfo;
    // Identical branches preserve all known offset facts.
    if (op.getTrueValue() == op.getFalseValue()) {
      auto info = lhs.offsets;
      info.fullOffsets = value;
      return PtrInfo{{}, std::move(info)};
    }
    auto info = buildOffsetInfo(value, builder);
    if (failed(info))
      return PtrInfo{};
    // A scalar condition selects one entire descriptor for all tensor lanes.
    if (op.getCondition().getType().isInteger(1) &&
        lhs.offsets.hasStructuredForm() && rhs.offsets.hasStructuredForm()) {
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
      auto origin = choose(lhs.offsets.scalarOffset, rhs.offsets.scalarOffset);
      if (failed(origin))
        return PtrInfo{};
      info->scalarOffset = *origin;
      for (unsigned i = 0; i < info->axes.size(); ++i) {
        auto stride =
            choose(lhs.offsets.axes[i].stride, rhs.offsets.axes[i].stride);
        if (failed(stride))
          return PtrInfo{};
        info->axes[i] =
            AxisInfo(lhs.offsets.axes[i].kind == AxisKind::Invariant &&
                             rhs.offsets.axes[i].kind == AxisKind::Invariant
                         ? AxisKind::Invariant
                         : AxisKind::Structured,
                     *stride);
      }
    }
    return PtrInfo{{}, std::move(*info)};
  }
  // Pointer selections need a common base and select the complete offsets.
  const auto &lhs = lhsInfo, &rhs = rhsInfo;
  if (lhs.base != rhs.base)
    return PtrInfo{};
  auto target =
      cast<RankedTensorType>(addressValueType(value.getType(), builder));
  Type storage = addressStorageType(target, builder);
  auto left = materializeFullOffset(lhs.offsets, storage, builder, op.getLoc());
  auto right =
      materializeFullOffset(rhs.offsets, storage, builder, op.getLoc());
  if (failed(left) || failed(right))
    return PtrInfo{};
  PtrInfo info = lhs;
  info.offsets.fullOffsets =
      builder
          .create<arith::SelectOp>(op.getLoc(), op.getCondition(), *left,
                                   *right)
          .getResult();
  info.offsets.scalarOffset = {};
  info.offsets.axes.assign(
      target.getRank(), AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  // Equal uniform offsets remain independent of the selected branch.
  if (lhs.offsets.isUniform() && rhs.offsets.isUniform() &&
      lhs.offsets.scalarOffset == rhs.offsets.scalarOffset) {
    info.offsets.scalarOffset = lhs.offsets.scalarOffset;
    for (auto &axis : info.offsets.axes)
      axis.kind = AxisKind::Invariant;
  }
  return PtrInfo(std::move(info));
}

FailureOr<PtrInfo> insertPtrOffsets(
    Value value, const PtrInfo &source, const PtrInfo &dest, Type sourceType,
    llvm::function_ref<Value(Value, Value)> insert, OpBuilder &builder) {
  // Both inputs must describe ordinary pointers relative to the same base.
  if (!source.base || source.base != dest.base || source.block || dest.block)
    return PtrInfo{};
  // Materialize the source and destination in the pointer offset storage type.
  Type insertedType = builder.getI64Type();
  if (auto tensor = dyn_cast<RankedTensorType>(sourceType))
    insertedType = tensor.clone(builder.getI64Type());
  auto src = materializeFullOffset(source.offsets, insertedType, builder,
                                   value.getLoc());
  auto dst = materializeFullOffset(dest.offsets,
                                   addressStorageType(value.getType(), builder),
                                   builder, value.getLoc());
  if (failed(src) || failed(dst))
    return PtrInfo{};
  // Apply the caller's insertion and retain only facts valid for both inputs.
  PtrInfo info = dest;
  info.offsets.fullOffsets = insert(*src, *dst);
  info.offsets.scalarOffset = {};
  info.offsets.axes.assign(
      info.offsets.getShape().size(),
      AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  if (source.offsets.isUniform() && dest.offsets.isUniform() &&
      source.offsets.scalarOffset == dest.offsets.scalarOffset) {
    info.offsets.scalarOffset = dest.offsets.scalarOffset;
    for (auto &axis : info.offsets.axes)
      axis.kind = AxisKind::Invariant;
  }
  return info;
}

// Block descriptor construction and coordinate updates.
FailureOr<PtrInfo> makeBlockInfo(triton::MakeTensorPtrOp op,
                                 const PtrInfo &base,
                                 ArrayRef<OpFoldResult> coordinates,
                                 OpBuilder &builder) {
  // Preserve the analyzed base and its existing element offset.
  PtrInfo ptr = base;
  auto baseOffset = materializeOffsetAt(base.offsets, {}, builder, op.getLoc());
  if (failed(baseOffset))
    return failure();
  // Build block axes from the result shape and index-converted strides.
  auto tensorType = cast<RankedTensorType>(
      cast<triton::PointerType>(op.getType()).getPointeeType());
  auto &offsets = ptr.offsets;
  offsets.valueType = RankedTensorType::get(
      tensorType.getShape(), builder.getIndexType(), tensorType.getEncoding());
  offsets.fullOffsets = {};
  offsets.axes.clear();
  for (Value stride : op.getStrides()) {
    auto constant = detail::getIntegerConstant(stride, 64);
    auto indexStride =
        foldOrCreateCast(stride, builder.getIndexType(),
                         IntegerExtensionKind::Signed, builder, op.getLoc());
    if (failed(indexStride))
      return PtrInfo{};
    offsets.axes.emplace_back(succeeded(constant) && constant->isZero()
                                  ? AxisKind::Invariant
                                  : AxisKind::Structured,
                              *indexStride);
  }
  // Add each starting coordinate contribution to the base offset.
  auto origin =
      addScaledCoordinates(*baseOffset, coordinates,
                           detail::axisStrides(offsets), builder, op.getLoc());
  if (failed(origin))
    return PtrInfo{};
  offsets.scalarOffset = *origin;
  // Keep the parent shape, coordinates and order for block boundary checks.
  ptr.block.emplace();
  for (Value shape : op.getShape()) {
    auto indexShape =
        foldOrCreateCast(shape, builder.getIndexType(),
                         IntegerExtensionKind::Signed, builder, op.getLoc());
    if (failed(indexShape))
      return PtrInfo{};
    ptr.block->parentShape.push_back(*indexShape);
  }
  ptr.block->coordinateOffsets.assign(coordinates.begin(), coordinates.end());
  ptr.block->order = llvm::to_vector(op.getOrderAttr().asArrayRef());
  // Validate the completed descriptor against the operation result.
  if (!detail::validPtrBinding(op.getResult(), ptr))
    return failure();
  return PtrInfo(std::move(ptr));
}

FailureOr<PtrInfo> advanceBlockInfo(triton::AdvanceOp op, const PtrInfo &parent,
                                    ArrayRef<OpFoldResult> delta,
                                    OpBuilder &builder) {
  // Apply the analyzed increments to the existing per-axis coordinates.
  PtrInfo ptr = parent;
  detail::ConcreteArithmetic arithmetic{builder.getIndexType(), builder,
                                        op.getLoc()};
  auto updated = advanceCoordinateTerms<OpFoldResult>(
      ptr.block->coordinateOffsets, delta, arithmetic);
  if (failed(updated))
    return PtrInfo{};
  // Add only this advance, retaining the existing scalar base contribution.
  auto origin = addScaledCoordinates(ptr.offsets.scalarOffset, delta,
                                     detail::axisStrides(ptr.offsets), builder,
                                     op.getLoc());
  if (failed(origin))
    return PtrInfo{};
  // Keep the updated origin and coordinates as one validated descriptor.
  ptr.offsets.scalarOffset = *origin;
  ptr.offsets.fullOffsets = {};
  ptr.block->coordinateOffsets = std::move(*updated);
  if (!detail::validPtrBinding(op.getResult(), ptr))
    return failure();
  return PtrInfo(std::move(ptr));
}

// Binding validation and dependency collection.
bool validOffsetBinding(Value boundary, const OffsetInfo &components) {
  return boundary && validOffset(boundary.getType(), components);
}

bool validBlockPtrBinding(Value value, const PtrInfo &components) {
  if (!value || !isBlockPtrType(value.getType()) || !components.block ||
      !components.offsets.hasStructuredForm())
    return false;
  const auto &block = *components.block;
  unsigned rank = components.offsets.getShape().size();
  if (block.parentShape.size() != rank ||
      block.coordinateOffsets.size() != rank || block.order.size() != rank)
    return false;
  SmallVector<bool> seen(rank, false);
  for (int32_t dim : block.order) {
    if (dim < 0 || static_cast<unsigned>(dim) >= rank || seen[dim])
      return false;
    seen[dim] = true;
  }
  for (unsigned i = 0; i < rank; ++i)
    if (!isIndex(block.parentShape[i]) || !isIndex(block.coordinateOffsets[i]))
      return false;
  return true;
}

bool validPtrBinding(Value boundary, const PtrInfo &components) {
  if (!boundary || !components.base ||
      !isa<triton::PointerType>(components.base.getType()))
    return false;
  Type ptrElement = boundary.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(ptrElement))
    ptrElement = tensor.getElementType();
  auto ptrType = dyn_cast<triton::PointerType>(ptrElement);
  auto baseType = dyn_cast<triton::PointerType>(components.base.getType());
  if (!ptrType || !baseType)
    return false;
  auto blockType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
  if (!blockType && isa<ShapedType>(ptrType.getPointeeType()))
    return false;
  Type elementType =
      blockType ? blockType.getElementType() : ptrType.getPointeeType();
  if (baseType.getPointeeType() != elementType ||
      baseType.getAddressSpace() != ptrType.getAddressSpace() ||
      (blockType ? !detail::validBlockPtrBinding(boundary, components)
                 : components.block.has_value()))
    return false;
  Type source = blockType ? Type(blockType) : boundary.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    source = RankedTensorType::get(tensor.getShape(),
                                   IndexType::get(source.getContext()),
                                   tensor.getEncoding());
  else
    source = IndexType::get(source.getContext());
  return validOffset(source, components.offsets,
                     /*pointer=*/true);
}

// SSA dependencies used by materialization.
void collectValues(ArrayRef<OpFoldResult> fields,
                   SmallVectorImpl<Value> &values) {
  for (OpFoldResult field : fields) {
    if (!field)
      continue;
    if (auto value = dyn_cast<Value>(field))
      values.push_back(value);
  }
}

} // namespace detail
} // namespace mlir::triton::ptr
