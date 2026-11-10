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

#include "TritonMemoryAccess/Utils.h"
#include "TritonMemoryAccess/PtrAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::triton::ptr::detail {

// Integer types, constants, and source-width arithmetic.
Type getElementTypeOrSelf(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}

FailureOr<unsigned> getIntegerWidth(Type type) {
  type = detail::getElementTypeOrSelf(type);
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (!integer.isSignless() || integer.getWidth() > 64)
      return failure();
    return integer.getWidth();
  }
  if (type.isIndex())
    return 64;
  return failure();
}

FailureOr<OpFoldResult> typedZero(Type scalarType, OpBuilder &builder) {
  if (scalarType.isIndex())
    return OpFoldResult(builder.getIndexAttr(0));
  auto integer = dyn_cast<IntegerType>(scalarType);
  if (!integer || failed(getIntegerWidth(integer)))
    return failure();
  return OpFoldResult(builder.getIntegerAttr(scalarType, 0));
}

FailureOr<llvm::APInt> getIntegerConstant(OpFoldResult value, unsigned width) {
  Attribute attr = dyn_cast<Attribute>(value);
  if (!attr) {
    Value ssa = dyn_cast<Value>(value);
    auto constant = ssa ? ssa.getDefiningOp<arith::ConstantOp>() : nullptr;
    if (!constant)
      return failure();
    attr = constant.getValue();
  }
  if (auto integer = dyn_cast<IntegerAttr>(attr))
    return integer.getValue().sextOrTrunc(width);
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (dense.isSplat())
      return dense.getSplatValue<llvm::APInt>().sextOrTrunc(width);
  }
  return failure();
}

// Inspect only local divisibility facts, without rebuilding scalar arithmetic
// or adding a second analysis cache. Bound the local walk; expressions beyond
// it simply need an explicit fact. Power-of-two factors survive integer wrap.
bool isKnownMultiple(OpFoldResult operand, uint64_t divisor, unsigned depth) {
  if (divisor == 1)
    return true;
  auto constant = getIntegerConstant(operand, 64);
  if (succeeded(constant))
    return constant->srem(llvm::APInt(64, divisor)).isZero();
  auto value = dyn_cast<Value>(operand);
  auto *op = value ? value.getDefiningOp() : nullptr;
  if (!op || !depth)
    return false;
  if (!isa<ShapedType>(value.getType()))
    if (auto hint = op->getAttrOfType<DenseIntElementsAttr>("tt.divisibility"))
      if (hint.getNumElements() == 1) {
        auto multiple = hint.getValues<llvm::APInt>()[0].getSExtValue();
        if (multiple > 0 && uint64_t(multiple) % divisor == 0)
          return true;
      }
  auto multiple = [&](Value input, uint64_t d) {
    return isKnownMultiple(input, d, depth - 1);
  };
  if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
          arith::ExtSIOp>(op))
    return multiple(op->getOperand(0), divisor);
  if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
    auto from = getIntegerWidth(cast.getIn().getType());
    auto to = getIntegerWidth(value.getType());
    if (succeeded(from) && succeeded(to) && *from <= *to)
      return multiple(cast.getIn(), divisor);
  }
  // For other integer arithmetic, a power-of-two modulus also divides the
  // source width's wrap modulus. General factors need an explicit fact above.
  auto width = getIntegerWidth(value.getType());
  if (!llvm::isPowerOf2_64(divisor) || failed(width) ||
      llvm::Log2_64(divisor) >= *width)
    return false;
  if (isa<arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
          arith::IndexCastUIOp>(op)) {
    auto from = getIntegerWidth(op->getOperand(0).getType());
    return succeeded(from) && llvm::Log2_64(divisor) < *from &&
           multiple(op->getOperand(0), divisor);
  }
  if (isa<arith::AddIOp, arith::SubIOp>(op))
    return multiple(op->getOperand(0), divisor) &&
           multiple(op->getOperand(1), divisor);
  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    for (unsigned i = 0; i < 2; ++i) {
      auto factor = getIntegerConstant(mul->getOperand(i), 64);
      uint64_t remaining = divisor;
      if (succeeded(factor))
        remaining >>=
            std::min<unsigned>(factor->countr_zero(), llvm::Log2_64(divisor));
      if (multiple(mul->getOperand(1 - i), remaining))
        return true;
    }
  }
  return false;
}

bool isKnownNonNegative(OpFoldResult operand, unsigned depth) {
  auto constant = getIntegerConstant(operand, 64);
  if (succeeded(constant))
    return !constant->isNegative();
  auto value = dyn_cast<Value>(operand);
  auto *op = value ? value.getDefiningOp() : nullptr;
  if (!op || !depth)
    return false;
  auto nonnegative = [&](Value input) {
    return isKnownNonNegative(input, depth - 1);
  };

  // Program coordinates and zero-extended scalars supply nonnegative origins.
  if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(op))
    return true;
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::IndexCastOp,
          arith::IndexCastUIOp>(op)) {
    auto from = getIntegerWidth(op->getOperand(0).getType());
    auto to = getIntegerWidth(value.getType());
    if (failed(from) || failed(to) || *from > *to)
      return false;
    if (*from < *to && isa<arith::ExtUIOp, arith::IndexCastUIOp>(op))
      return true;
    return nonnegative(op->getOperand(0));
  }

  // Address arithmetic follows the same no-overflow contract as its axes.
  if (isa<arith::AddIOp, arith::MulIOp>(op))
    return nonnegative(op->getOperand(0)) && nonnegative(op->getOperand(1));
  if (isa<arith::MaxSIOp>(op))
    return nonnegative(op->getOperand(0)) || nonnegative(op->getOperand(1));
  if (auto div = dyn_cast<arith::DivSIOp>(op)) {
    auto divisor = getIntegerConstant(div.getRhs(), 64);
    return succeeded(divisor) && divisor->isStrictlyPositive() &&
           nonnegative(div.getLhs());
  }
  return false;
}

FailureOr<Value> materialize(OpFoldResult value, Type expectedType,
                             OpBuilder &builder, Location loc) {
  if (auto ssa = dyn_cast<Value>(value)) {
    if (ssa.getType() != expectedType)
      return failure();
    return ssa;
  }
  auto attr = dyn_cast<Attribute>(value);
  auto typed = dyn_cast_or_null<TypedAttr>(attr);
  if (!typed || typed.getType() != expectedType)
    return failure();
  return builder.create<arith::ConstantOp>(loc, typed).getResult();
}

FailureOr<OpFoldResult> foldOrCreateBinary(BinaryKind kind, OpFoldResult lhs,
                                           OpFoldResult rhs, Type resultType,
                                           OpBuilder &builder, Location loc) {
  if (!resultType ||
      (isa<ShapedType>(resultType) && !isa<RankedTensorType>(resultType)))
    return failure();
  if (auto tensor = dyn_cast<RankedTensorType>(resultType))
    if (!tensor.hasStaticShape())
      return failure();
  FailureOr<unsigned> width = getIntegerWidth(resultType);
  if (failed(width))
    return failure();
  auto leftConstant = getIntegerConstant(lhs, *width);
  auto rightConstant = getIntegerConstant(rhs, *width);
  if (succeeded(leftConstant) && succeeded(rightConstant)) {
    llvm::APInt result = *leftConstant;
    switch (kind) {
    case BinaryKind::Add:
      result += *rightConstant;
      break;
    case BinaryKind::Subtract:
      result -= *rightConstant;
      break;
    case BinaryKind::Multiply:
      result *= *rightConstant;
      break;
    }
    if (auto tensor = dyn_cast<RankedTensorType>(resultType))
      return OpFoldResult(DenseIntElementsAttr::get(tensor, result));
    return OpFoldResult(builder.getIntegerAttr(resultType, result));
  }
  if (auto value = dyn_cast<Value>(lhs)) {
    if (value.getType() == resultType && succeeded(rightConstant)) {
      if ((kind != BinaryKind::Multiply && rightConstant->isZero()) ||
          (kind == BinaryKind::Multiply && rightConstant->isOne()))
        return lhs;
    }
  }
  if (auto value = dyn_cast<Value>(rhs)) {
    if (value.getType() == resultType && succeeded(leftConstant)) {
      if ((kind == BinaryKind::Add && leftConstant->isZero()) ||
          (kind == BinaryKind::Multiply && leftConstant->isOne()))
        return rhs;
    }
  }
  if (kind == BinaryKind::Multiply &&
      ((succeeded(leftConstant) && leftConstant->isZero()) ||
       (succeeded(rightConstant) && rightConstant->isZero()))) {
    if (auto tensor = dyn_cast<RankedTensorType>(resultType))
      return OpFoldResult(
          DenseIntElementsAttr::get(tensor, llvm::APInt(*width, 0)));
    return typedZero(resultType, builder);
  }
  FailureOr<Value> left = materialize(lhs, resultType, builder, loc);
  FailureOr<Value> right = materialize(rhs, resultType, builder, loc);
  if (failed(left) || failed(right))
    return failure();
  switch (kind) {
  case BinaryKind::Add:
    return OpFoldResult(
        builder.create<arith::AddIOp>(loc, *left, *right).getResult());
  case BinaryKind::Subtract:
    return OpFoldResult(
        builder.create<arith::SubIOp>(loc, *left, *right).getResult());
  case BinaryKind::Multiply:
    return OpFoldResult(
        builder.create<arith::MulIOp>(loc, *left, *right).getResult());
  }
  llvm_unreachable("exhaustive binary kind");
}

FailureOr<OpFoldResult> foldOrCreateCast(OpFoldResult source, Type targetType,
                                         IntegerExtensionKind kind,
                                         OpBuilder &builder, Location loc) {
  auto sourceType = dyn_cast<Value>(source)
                        ? cast<Value>(source).getType()
                        : cast<TypedAttr>(cast<Attribute>(source)).getType();
  FailureOr<unsigned> oldWidth = getIntegerWidth(sourceType);
  FailureOr<unsigned> newWidth = getIntegerWidth(targetType);
  if (failed(oldWidth) || failed(newWidth))
    return failure();
  if (sourceType == targetType)
    return source;
  FailureOr<llvm::APInt> constant = getIntegerConstant(source, *oldWidth);
  if (succeeded(constant)) {
    llvm::APInt converted = *constant;
    if (*newWidth < *oldWidth)
      converted = converted.trunc(*newWidth);
    else if (kind == IntegerExtensionKind::Unsigned)
      converted = converted.zextOrTrunc(*newWidth);
    else
      converted = converted.sextOrTrunc(*newWidth);
    if (auto tensor = dyn_cast<RankedTensorType>(targetType))
      return OpFoldResult(DenseIntElementsAttr::get(tensor, converted));
    return OpFoldResult(builder.getIntegerAttr(targetType, converted));
  }
  // Non-splat attributes are legal complete offsets supplied by bindings.
  // Materialize them before casting instead of assuming an SSA operand.
  auto value = materialize(source, sourceType, builder, loc);
  if (failed(value))
    return failure();
  // The shared integer helper handles scalar index, but only IntegerType
  // elements for tensors. Preserve tensor shape/encoding for index casts here.
  if (auto sourceTensor = dyn_cast<RankedTensorType>(sourceType)) {
    auto targetTensor = dyn_cast<RankedTensorType>(targetType);
    if (!targetTensor || sourceTensor.getShape() != targetTensor.getShape() ||
        sourceTensor.getEncoding() != targetTensor.getEncoding())
      return failure();
    if (sourceTensor.getElementType().isIndex() ||
        targetTensor.getElementType().isIndex()) {
      if (kind == IntegerExtensionKind::Unsigned)
        return OpFoldResult(
            builder.create<arith::IndexCastUIOp>(loc, targetType, *value)
                .getResult());
      return OpFoldResult(
          builder.create<arith::IndexCastOp>(loc, targetType, *value)
              .getResult());
    }
  }
  FailureOr<Value> result =
      castIntegerLike(builder, loc, *value, targetType, kind);
  if (failed(result))
    return failure();
  return OpFoldResult(*result);
}

FailureOr<Value> splatTo(Value scalar, RankedTensorType target,
                         OpBuilder &builder, Location loc) {
  if (!scalar || scalar.getType() != target.getElementType())
    return failure();
  return builder.create<triton::SplatOp>(loc, target, scalar).getResult();
}

// Address decompositions and axis proofs.
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
  OpFoldResult offset = info.fullOffsets;
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
  // A scalar origin alone does not supply runtime tensor extents.
  if (broadcast && !tensor.hasStaticShape())
    return failure();
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

// Axis alignment and stride queries.
void normalizeSubAxes(SmallVectorImpl<SubAxisInfo> &subAxes,
                      OpBuilder &builder) {
  if (subAxes.empty())
    return;
  SmallVector<SubAxisInfo> normalized;
  for (const SubAxisInfo &sub : subAxes) {
    if (sub.shape == 1)
      continue;
    if (!normalized.empty() && zero(normalized.back().stride) &&
        zero(sub.stride)) {
      // Valid partitions multiply to a representable original dimension.
      normalized.back().shape *= sub.shape;
    } else {
      normalized.push_back(sub);
    }
  }
  if (normalized.empty())
    normalized.push_back({1, builder.getIndexAttr(0)});
  subAxes.assign(normalized.begin(), normalized.end());
}

// A linear axis can use either partition. Two mixed-radix partitions can
// additionally split a longer zero-stride digit into integral factors: the
// split contributes zero in both digits and preserves every lane's value.
LogicalResult alignAxes(AxisInfo &lhs, AxisInfo &rhs,
                        ConcreteArithmetic &arithmetic) {
  normalizeSubAxes(lhs.subAxes, arithmetic.builder);
  normalizeSubAxes(rhs.subAxes, arithmetic.builder);
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
  SmallVector<SubAxisInfo> left, right;
  unsigned i = 0, j = 0;
  SubAxisInfo l = lhs.subAxes.front(), r = rhs.subAxes.front();
  while (true) {
    if (l.shape == r.shape) {
      left.push_back(l);
      right.push_back(r);
      ++i;
      ++j;
      if (i == lhs.subAxes.size() || j == rhs.subAxes.size())
        break;
      l = lhs.subAxes[i];
      r = rhs.subAxes[j];
    } else if (l.shape > r.shape && l.shape % r.shape == 0 && zero(l.stride)) {
      left.push_back({r.shape, l.stride});
      right.push_back(r);
      l.shape /= r.shape;
      if (++j == rhs.subAxes.size())
        return failure();
      r = rhs.subAxes[j];
    } else if (r.shape > l.shape && r.shape % l.shape == 0 && zero(r.stride)) {
      left.push_back(l);
      right.push_back({l.shape, r.stride});
      r.shape /= l.shape;
      if (++i == lhs.subAxes.size())
        return failure();
      l = lhs.subAxes[i];
    } else {
      return failure();
    }
  }
  if (i != lhs.subAxes.size() || j != rhs.subAxes.size())
    return failure();
  lhs.subAxes = std::move(left);
  rhs.subAxes = std::move(right);
  return success();
}

SmallVector<OpFoldResult> axisStrides(const OffsetInfo &info) {
  SmallVector<OpFoldResult> strides;
  for (const auto &axis : info.axes)
    strides.push_back(axis.stride);
  return strides;
}

OpFoldResult inferAxisStride(ArrayRef<SubAxisInfo> subAxes,
                             OpBuilder &builder) {
  // Unit sub-axes never change the coordinate; take the innermost varying
  // sub-axis as the candidate stride. An all-unit partition is invariant.
  OpFoldResult stride = builder.getIndexAttr(0);
  for (const auto &sub : llvm::reverse(subAxes))
    if (sub.shape != 1) {
      stride = sub.stride;
      break;
    }

  // Each outer coefficient must equal that stride times all inner lengths.
  // Symbolic products remain unproved; no helper IR is needed for this query.
  auto inner = getIntegerConstant(stride, 64);
  llvm::APInt scale(64, 1);
  for (const auto &sub : llvm::reverse(subAxes)) {
    if (sub.shape != 1 && !scale.isOne()) {
      auto coefficient = getIntegerConstant(sub.stride, 64);
      if (failed(inner) || failed(coefficient) ||
          *coefficient != *inner * scale)
        return {};
    }
    scale *= llvm::APInt(64, sub.shape);
  }
  return zero(stride) ? builder.getIndexAttr(0) : stride;
}

// Divisibility proofs and exact unit conversion.
bool canDivideDecomposition(const OffsetInfo &info, uint64_t divisor) {
  return info.hasStructuredDecomposition() &&
         isKnownMultiple(info.scalarOffset, divisor) &&
         llvm::all_of(info.axes, [&](const AxisInfo &axis) {
           return canDivideAxis(axis, divisor);
         });
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
      axis = buildAxisInfo(std::move(axis.subAxes), builder);
    } else {
      axis = buildAxisInfo(divideIndex(axis.stride, divisor, builder, loc));
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

void broadcastAxes(SmallVectorImpl<AxisInfo> &axes,
                   ArrayRef<int64_t> sourceShape, ArrayRef<int64_t> targetShape,
                   const AxisInfo &invariant) {
  assert(axes.size() == sourceShape.size() &&
         axes.size() == targetShape.size());
  for (unsigned i = 0; i < axes.size(); ++i) {
    if (sourceShape[i] == targetShape[i])
      continue;
    assert(sourceShape[i] == 1 && "only singleton dimensions can broadcast");
    axes[i] = invariant;
  }
}

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
