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
#include "TritonMemoryAccess/Utils.h"
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
  return scalarOffset && llvm::all_of(axes, [](const AxisInfo &axis) {
           return axis.kind == AxisKind::Invariant;
         });
}

Type PtrInfo::getElementType() const {
  Type type = base.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return cast<triton::PointerType>(type).getPointeeType();
}

unsigned PtrInfo::getAddressSpace() const {
  Type type = base.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return cast<triton::PointerType>(type).getAddressSpace();
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

// Index-domain address contributions; source operations keep their widths.
FailureOr<OpFoldResult> addScaledCoordinates(OpFoldResult origin,
                                             ArrayRef<OpFoldResult> coordinates,
                                             ArrayRef<OpFoldResult> strides,
                                             OpBuilder &builder, Location loc) {
  detail::ConcreteArithmetic arithmetic{builder.getIndexType(), builder, loc};
  assert(coordinates.size() == strides.size());
  for (auto [coordinate, stride] : llvm::zip_equal(coordinates, strides)) {
    auto term = arithmetic.multiply(coordinate, stride);
    if (failed(term))
      return failure();
    auto sum = arithmetic.add(origin, *term, false);
    if (failed(sum))
      return failure();
    origin = *sum;
  }
  return origin;
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
  // Bindings supply complete facts: check their types and consistency without
  // rewriting them. Internal analysis constructs these fields together.
  OpBuilder builder(resultType.getContext());
  for (auto [axis, size] : llvm::zip(components.axes, components.getShape())) {
    if (!validOfrType(axis.stride, scalarType))
      return false;
    if (axis.subAxes.empty()) {
      if (axis.kind != AxisKind::Unknown &&
          axis.kind != (detail::zero(axis.stride) ? AxisKind::Invariant
                                                  : AxisKind::Structured))
        return false;
      continue;
    }

    // A sub-axis partition must cover the axis exactly.
    int64_t product = 1;
    for (const auto &sub : axis.subAxes) {
      if (!validOfrType(sub.stride, scalarType) || sub.shape <= 0 ||
          __builtin_mul_overflow(product, sub.shape, &product))
        return false;
    }
    if (product != size)
      return false;
    auto stride = detail::inferAxisStride(axis.subAxes, builder);
    if (!stride) {
      if (axis.kind != AxisKind::Unknown)
        return false;
      continue;
    }
    AxisKind kind =
        detail::zero(stride) ? AxisKind::Invariant : AxisKind::Structured;
    if (axis.kind != kind || !detail::equalConstant(axis.stride, stride))
      return false;
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
AxisInfo buildAxisInfo(OpFoldResult stride) {
  return AxisInfo(zero(stride) ? AxisKind::Invariant : AxisKind::Structured,
                  stride);
}

AxisInfo buildAxisInfo(SmallVector<SubAxisInfo> subAxes, OpBuilder &builder) {
  normalizeSubAxes(subAxes, builder);
  auto stride = inferAxisStride(subAxes, builder);
  AxisInfo axis = stride ? buildAxisInfo(stride)
                         : AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0));
  axis.subAxes = std::move(subAxes);
  return axis;
}

AxisInfo makeInvariantAxisInfo(OpBuilder &builder) {
  return AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0));
}

OffsetInfo buildUniformOffsetInfo(Value value, OpBuilder &builder,
                                  OpFoldResult scalar) {
  OffsetInfo info;
  info.valueType = value.getType();
  info.fullOffsets = value;
  info.axes.assign(info.getShape().size(), makeInvariantAxisInfo(builder));
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

FailureOr<PtrInfo> buildPointerInfo(Value value, OpBuilder &builder) {
  if (!value || !isOrdinaryPtrType(value.getType()))
    return failure();
  PtrInfo info;
  info.base = value;
  info.offsets.valueType = addressValueType(value.getType(), builder);
  info.offsets.fullOffsets = builder.getIndexAttr(0);
  info.offsets.scalarOffset = builder.getIndexAttr(0);
  if (auto tensor = dyn_cast<RankedTensorType>(value.getType())) {
    auto storage = tensor.clone(builder.getI64Type());
    info.offsets.axes.assign(tensor.getRank(), makeInvariantAxisInfo(builder));
    if (tensor.hasStaticShape()) {
      info.offsets.fullOffsets = DenseIntElementsAttr::get(storage, int64_t(0));
    } else {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(value);
      SmallVector<Value> sizes;
      for (int64_t dim = 0; dim < tensor.getRank(); ++dim)
        if (tensor.isDynamicDim(dim))
          sizes.push_back(
              builder.create<tensor::DimOp>(value.getLoc(), value, dim));
      info.offsets.fullOffsets =
          builder
              .create<tensor::GenerateOp>(
                  value.getLoc(), storage, sizes,
                  [&](OpBuilder &nested, Location loc, ValueRange) {
                    Value zero =
                        nested.create<arith::ConstantIntOp>(loc, 0, 64);
                    nested.create<tensor::YieldOp>(loc, zero);
                  })
              .getResult();
    }
  }
  return PtrInfo(std::move(info));
}

// Integer offset combinations.
FailureOr<PtrInfo> combineElementwise(Value value, const PtrInfo &lhs,
                                      const PtrInfo &rhs, OpBuilder &builder) {
  // Elementwise operands and results have matching shapes in verified IR.
  auto offsets = buildOffsetInfo(value, builder);
  if (failed(offsets))
    return failure();
  assert(lhs.offsets.getShape() == offsets->getShape() &&
         rhs.offsets.getShape() == offsets->getShape() &&
         "elementwise facts must have matching shapes");
  // Only axes invariant in both inputs remain invariant in the result.
  for (unsigned i = 0; i < offsets->axes.size(); ++i)
    if (lhs.offsets.axes[i].kind == AxisKind::Invariant &&
        rhs.offsets.axes[i].kind == AxisKind::Invariant)
      offsets->axes[i].kind = AxisKind::Invariant;
  // Recover an origin when every result axis is invariant.
  setUniformOrigin(*offsets, value, builder);
  return PtrInfo{{}, std::move(*offsets)};
}

FailureOr<OffsetInfo> addOffsetInfo(OpFoldResult complete,
                                    const OffsetInfo &lhs,
                                    const OffsetInfo &rhs, bool subtract,
                                    OpBuilder &builder, Location loc) {
  assert(lhs.valueType == rhs.valueType && lhs.axes.size() == rhs.axes.size() &&
         "offset addition requires matching types and axes");
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
      info.axes[i] = buildAxisInfo(std::move(left.subAxes), builder);
    } else {
      auto stride = arithmetic.add(left.stride, right.stride, subtract);
      if (failed(stride))
        return failure();
      info.axes[i] = buildAxisInfo(*stride);
    }
  }
  return info;
}

FailureOr<OffsetInfo> scaleOffsetInfo(OpFoldResult complete,
                                      const OffsetInfo &source,
                                      OpFoldResult scale, OpBuilder &builder,
                                      Location loc) {
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
      axis = buildAxisInfo(std::move(axis.subAxes), builder);
    } else if (axis.kind != AxisKind::Unknown) {
      auto stride = arithmetic.multiply(axis.stride, scale);
      if (failed(stride))
        return failure();
      axis = buildAxisInfo(*stride);
    }
  }
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
  uint64_t divisor = constant->getZExtValue();
  auto bounds = detail::constantDecompositionRange(lhsInfo.offsets, *width);
  bool nonnegative = bounds && bounds->isAllNonNegative();
  bool signedOp = isa<arith::DivSIOp, arith::RemSIOp>(value.getDefiningOp());
  // Exact division needs no residual splitting: negative or dynamic origins
  // work too. Unsigned division can reuse signed descriptors only above zero.
  if ((signedOp || nonnegative) &&
      canDivideDecomposition(lhsInfo.offsets, divisor)) {
    OffsetInfo info = lhsInfo.offsets;
    info.fullOffsets = value;
    if (divide) {
      divideDecomposition(info, divisor, builder, value.getLoc());
    } else {
      info.scalarOffset = builder.getIndexAttr(0);
      info.axes.assign(info.getShape().size(), makeInvariantAxisInfo(builder));
    }
    return PtrInfo{{}, std::move(info)};
  }
  // Split only complete axes with an aligned, nonnegative origin. Signed
  // division across zero would change the residual digits' rounding.
  if (!lhsInfo.offsets.hasStructuredDecomposition() ||
      !isKnownMultiple(lhsInfo.offsets.scalarOffset, divisor))
    return fallback;
  auto origin = detail::getIntegerConstant(lhsInfo.offsets.scalarOffset, 64);
  // Keep the source-width range check for constant decompositions. Dynamic
  // origins use local sign facts under the existing no-overflow contract.
  if (succeeded(origin) ? !nonnegative
                        : !isKnownNonNegative(lhsInfo.offsets.scalarOffset))
    return fallback;
  OffsetInfo info = lhsInfo.offsets;
  info.fullOffsets = value;
  Type scalarType = builder.getIndexType();
  // Split the static digits first, before creating a dynamic origin quotient.
  __int128 residualMaximum = 0;
  for (unsigned i = 0; i < info.axes.size(); ++i) {
    auto &axis = info.axes[i];
    SmallVector<SubAxisInfo> sourceSubAxes = axis.subAxes;
    if (sourceSubAxes.empty())
      sourceSubAxes.push_back({info.getShape()[i], axis.stride});
    SmallVector<SubAxisInfo> resultSubAxes;
    for (const auto &sub : sourceSubAxes) {
      auto stride = detail::getIntegerConstant(sub.stride, 64);
      if (sub.shape <= 0 || failed(stride) || stride->isNegative())
        return fallback;
      uint64_t subAxisSize = sub.shape, strideValue = stride->getZExtValue();
      auto appendSubAxis = [&](uint64_t length, uint64_t coefficient) {
        resultSubAxes.push_back(
            {static_cast<int64_t>(length),
             builder.getIntegerAttr(scalarType, llvm::APInt(64, coefficient))});
      };
      if (strideValue % divisor == 0) {
        // Whole divisor multiples contribute only to the quotient.
        appendSubAxis(subAxisSize, divide ? strideValue / divisor : 0);
      } else if (divisor % strideValue == 0 &&
                 subAxisSize % (divisor / strideValue) == 0) {
        // Split complete periods into quotient and remainder sub-axes.
        uint64_t innerSize = divisor / strideValue;
        appendSubAxis(subAxisSize / innerSize, divide ? 1 : 0);
        appendSubAxis(innerSize, divide ? 0 : strideValue);
        residualMaximum += divisor - strideValue;
      } else if (__int128(subAxisSize - 1) * strideValue < divisor) {
        // A digit entirely below the divisor contributes only to the remainder.
        appendSubAxis(subAxisSize, divide ? 0 : strideValue);
        residualMaximum += __int128(subAxisSize - 1) * strideValue;
      } else {
        return fallback;
      }
    }
    axis = buildAxisInfo(std::move(resultSubAxes), builder);
  }
  // Multiple residual digits may carry into the quotient. Only accept their
  // independent decomposition when their total stays below the divisor.
  if (residualMaximum >= divisor)
    return fallback;
  info.scalarOffset =
      divide ? divideIndex(info.scalarOffset, divisor, builder, value.getLoc())
             : builder.getIndexAttr(0);
  return PtrInfo{{}, std::move(info)};
}

// Offset shape transformations.
OffsetInfo reshapeOffsetInfo(OpFoldResult complete, RankedTensorType targetType,
                             const OffsetInfo &sourceInfo, OpBuilder &builder) {
  auto fromType = cast<RankedTensorType>(sourceInfo.valueType);

  // Ordered reshape keeps the complete value and the known first-lane origin.
  OffsetInfo info = sourceInfo;
  info.valueType = targetType;
  info.fullOffsets = complete;
  info.axes.clear();

  // Uniform values remain invariant along every target axis.
  if (llvm::all_of(sourceInfo.axes, [](const AxisInfo &axis) {
        return axis.kind == AxisKind::Invariant;
      })) {
    info.axes.assign(targetType.getRank(), makeInvariantAxisInfo(builder));
    return info;
  }

  // Normalize shapes by removing unit dimensions. Matching shapes preserve
  // the other axes' coordinates.
  SmallVector<int64_t> sourceNormShape, targetNormShape;
  for (int64_t size : fromType.getShape())
    if (size != 1)
      sourceNormShape.push_back(size);
  for (int64_t size : targetType.getShape())
    if (size != 1)
      targetNormShape.push_back(size);
  if (sourceNormShape == targetNormShape) {
    unsigned sourceAxis = 0;
    for (int64_t size : targetType.getShape()) {
      if (size == 1) {
        info.axes.push_back(makeInvariantAxisInfo(builder));
      } else {
        while (fromType.getDimSize(sourceAxis) == 1)
          ++sourceAxis;
        info.axes.push_back(sourceInfo.axes[sourceAxis++]);
      }
    }
    return info;
  }

  // Other reshapes retain invariance only on target unit dimensions.
  for (int64_t size : targetType.getShape()) {
    if (size == 1) {
      info.axes.push_back(makeInvariantAxisInfo(builder));
    } else {
      info.axes.emplace_back(AxisKind::Unknown, builder.getIndexAttr(0));
    }
  }
  return info;
}

FailureOr<OffsetInfo>
sliceOffsetInfo(OpFoldResult complete, RankedTensorType targetType,
                const OffsetInfo &sourceInfo, ArrayRef<OpFoldResult> starts,
                ArrayRef<OpFoldResult> steps, ArrayRef<bool> dropped,
                OpBuilder &builder, Location loc) {
  // Verified slice metadata follows the source axes, with one result axis for
  // each retained dimension. Origin and stride updates do not need its extent.
  OffsetInfo info = sourceInfo;
  info.valueType = targetType;
  info.fullOffsets = complete;
  info.axes.clear();
  Type scalarType = builder.getIndexType();
  ConcreteArithmetic arithmetic{scalarType, builder, loc};
  // Dynamic slice coordinates are already index values; normalize constants.
  auto toIndex = [&](OpFoldResult coordinate) -> OpFoldResult {
    auto constant = getIntegerConstant(coordinate, 64);
    if (succeeded(constant))
      return OpFoldResult(builder.getIntegerAttr(scalarType, *constant));
    return coordinate;
  };

  // Moving along an unknown axis prevents deriving the new origin. Check this
  // before emitting any origin arithmetic that would later be discarded.
  for (unsigned sourceAxis = 0; sourceAxis < starts.size(); ++sourceAxis) {
    if (sourceInfo.axes[sourceAxis].kind == AxisKind::Unknown &&
        !zero(starts[sourceAxis])) {
      info.scalarOffset = {};
      break;
    }
  }

  // Shift a known origin by start * stride, including dropped and unit axes.
  if (info.scalarOffset) {
    for (unsigned sourceAxis = 0; sourceAxis < starts.size(); ++sourceAxis) {
      const AxisInfo &axis = sourceInfo.axes[sourceAxis];
      if (axis.kind == AxisKind::Invariant || zero(starts[sourceAxis]))
        continue;
      auto shift =
          arithmetic.multiply(toIndex(starts[sourceAxis]), axis.stride);
      if (failed(shift))
        return failure();
      auto origin = arithmetic.add(info.scalarOffset, *shift, false);
      if (failed(origin))
        return failure();
      info.scalarOffset = *origin;
    }
  }

  // Build result axes after the origin is settled.
  unsigned targetAxis = 0;
  for (unsigned sourceAxis = 0; sourceAxis < starts.size(); ++sourceAxis) {
    // Dropped axes have no result axis; unit axes need no stride calculation.
    if (dropped[sourceAxis])
      continue;
    int64_t targetSize = targetType.getDimSize(targetAxis++);
    if (targetSize == 1) {
      info.axes.push_back(makeInvariantAxisInfo(builder));
      continue;
    }

    AxisInfo axis = sourceInfo.axes[sourceAxis];
    // An in-bounds, nonempty full-length slice with unit step must start at
    // zero. Positive lengths exclude unknown dynamic extents and empty axes.
    bool fullAxis = targetSize > 0 &&
                    targetSize == sourceInfo.getShape()[sourceAxis] &&
                    equalConstant(steps[sourceAxis], builder.getIndexAttr(1));
    if (fullAxis) {
      info.axes.push_back(std::move(axis));
      continue;
    }

    // Other surviving axes keep only a stride scaled by the slice step.
    axis.subAxes.clear();
    if (axis.kind != AxisKind::Unknown) {
      auto stride =
          arithmetic.multiply(axis.stride, toIndex(steps[sourceAxis]));
      if (failed(stride))
        return failure();
      axis = buildAxisInfo(*stride);
    }
    info.axes.push_back(std::move(axis));
  }
  return info;
}

// Ordinary pointer offset accumulation.
FailureOr<PtrInfo> addPtrOffsets(Value value, const PtrInfo &parent,
                                 const OffsetInfo &delta, OpBuilder &builder) {
  // Pointer increments preserve the pointee type and use address-domain
  // offsets.
  assert(parent.getElementType() ==
             cast<triton::PointerType>(
                 detail::getElementTypeOrSelf(value.getType()))
                 .getPointeeType() &&
         "addptr must preserve the pointee type");
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
        info->axes.assign(tensor.getRank(), makeInvariantAxisInfo(builder));
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
  auto offsets =
      addOffsetInfo(complete, lhs, rhs, false, builder, value.getLoc());
  if (failed(offsets))
    return failure();
  PtrInfo info = parent;
  info.offsets = std::move(*offsets);
  return info;
}

// Selection and insertion of analyzed results.
FailureOr<PtrInfo> combineSelectInfo(arith::SelectOp op, const PtrInfo &lhsInfo,
                                     const PtrInfo &rhsInfo,
                                     OpBuilder &builder) {
  Value value = op.getResult();
  // Integer selections reuse identical branches or retain the exact result.
  if (!lhsInfo.base) {
    // Identical branches preserve all known offset facts.
    if (op.getTrueValue() == op.getFalseValue()) {
      auto info = lhsInfo.offsets;
      info.fullOffsets = value;
      return PtrInfo{{}, std::move(info)};
    }
    auto info = buildOffsetInfo(value, builder);
    if (failed(info))
      return failure();
    return PtrInfo{{}, std::move(*info)};
  }
  // Pointer selections need a common base and select the complete offsets.
  const auto &lhs = lhsInfo, &rhs = rhsInfo;
  if (lhs.base != rhs.base)
    return buildPointerInfo(value, builder);
  auto target =
      cast<RankedTensorType>(addressValueType(value.getType(), builder));
  Type storage = addressStorageType(target, builder);
  auto left = materializeFullOffset(lhs.offsets, storage, builder, op.getLoc());
  auto right =
      materializeFullOffset(rhs.offsets, storage, builder, op.getLoc());
  if (failed(left) || failed(right))
    return failure();
  PtrInfo info = lhs;
  info.offsets.fullOffsets =
      builder
          .create<arith::SelectOp>(op.getLoc(), op.getCondition(), *left,
                                   *right)
          .getResult();
  // Equal uniform offsets remain independent of the selected branch.
  if (lhs.offsets.isUniform() && rhs.offsets.isUniform() &&
      lhs.offsets.scalarOffset == rhs.offsets.scalarOffset) {
    info.offsets.scalarOffset = lhs.offsets.scalarOffset;
    info.offsets.axes.assign(target.getRank(), makeInvariantAxisInfo(builder));
  } else {
    info.offsets.scalarOffset = {};
    info.offsets.axes.assign(
        target.getRank(), AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  }
  return PtrInfo(std::move(info));
}

FailureOr<PtrInfo> insertPtrOffsets(
    Value value, const PtrInfo &source, const PtrInfo &dest, Type sourceType,
    llvm::function_ref<Value(Value, Value)> insert, OpBuilder &builder) {
  // Both inputs must describe ordinary pointers relative to the same base.
  if (!source.base || source.base != dest.base || source.block || dest.block)
    return buildPointerInfo(value, builder);
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
    return failure();
  // Apply the caller's insertion and retain only facts valid for both inputs.
  PtrInfo info = dest;
  info.offsets.fullOffsets = insert(*src, *dst);
  // Equal uniform offsets survive insertion; other cases keep only fullOffsets.
  if (source.offsets.isUniform() && dest.offsets.isUniform() &&
      source.offsets.scalarOffset == dest.offsets.scalarOffset) {
    info.offsets.scalarOffset = dest.offsets.scalarOffset;
    info.offsets.axes.assign(info.offsets.getShape().size(),
                             makeInvariantAxisInfo(builder));
  } else {
    info.offsets.scalarOffset = {};
    info.offsets.axes.assign(
        info.offsets.getShape().size(),
        AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
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
      return failure();
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
    return failure();
  offsets.scalarOffset = *origin;
  // Keep the parent shape, coordinates and order for block boundary checks.
  ptr.block.emplace();
  ptr.block->descriptorBase = op.getBase();
  for (Value shape : op.getShape()) {
    auto indexShape =
        foldOrCreateCast(shape, builder.getIndexType(),
                         IntegerExtensionKind::Signed, builder, op.getLoc());
    if (failed(indexShape))
      return failure();
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
  SmallVector<OpFoldResult> updated;
  for (auto [coordinate, increment] :
       llvm::zip_equal(ptr.block->coordinateOffsets, delta)) {
    auto next = arithmetic.add(coordinate, increment, false);
    if (failed(next))
      return failure();
    updated.push_back(*next);
  }
  // Add only this advance, retaining the existing scalar base contribution.
  auto origin = addScaledCoordinates(ptr.offsets.scalarOffset, delta,
                                     detail::axisStrides(ptr.offsets), builder,
                                     op.getLoc());
  if (failed(origin))
    return failure();
  // Keep the updated origin and coordinates as one validated descriptor.
  ptr.offsets.scalarOffset = *origin;
  ptr.offsets.fullOffsets = {};
  ptr.block->coordinateOffsets = std::move(updated);
  if (!detail::validPtrBinding(op.getResult(), ptr))
    return failure();
  return PtrInfo(std::move(ptr));
}

// Binding validation and dependency collection.
LogicalResult validateInfo(Value value, const PtrInfo &info) {
  if (!value || info.isEmpty())
    return failure();
  if (info.base)
    return success(validPtrBinding(value, info));
  return success(!info.block && validOffsetBinding(value, info.offsets));
}

bool validOffsetBinding(Value boundary, const OffsetInfo &components) {
  return boundary && validOffset(boundary.getType(), components);
}

bool validBlockPtrBinding(Value value, const PtrInfo &components) {
  if (!value || !isBlockPtrType(value.getType()) || !components.block ||
      !components.offsets.hasStructuredForm())
    return false;
  const auto &block = *components.block;
  if (!block.descriptorBase ||
      block.descriptorBase.getType() != components.base.getType())
    return false;
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
  if (!boundary || !components.base)
    return false;
  Type baseElement = components.base.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(baseElement)) {
    auto target = dyn_cast<RankedTensorType>(boundary.getType());
    if (!target || tensor.getShape() != target.getShape() ||
        tensor.getEncoding() != target.getEncoding())
      return false;
    baseElement = tensor.getElementType();
  }
  Type ptrElement = boundary.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(ptrElement))
    ptrElement = tensor.getElementType();
  auto ptrType = dyn_cast<triton::PointerType>(ptrElement);
  auto baseType = dyn_cast<triton::PointerType>(baseElement);
  if (!ptrType || !baseType)
    return false;
  auto blockType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
  if (!blockType && isa<ShapedType>(ptrType.getPointeeType()))
    return false;
  // Ordinary pointers retain complete offsets even with a full decomposition.
  if (!blockType && !components.offsets.fullOffsets)
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
