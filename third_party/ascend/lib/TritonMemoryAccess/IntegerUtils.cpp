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

#include "TritonMemoryAccess/IntegerUtils.h"
#include "TritonMemoryAccess/OpFoldResultUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

namespace mlir::triton::ptr::detail {

Type getIntegerElementType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}

SmallVector<int64_t> getLogicalShape(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return llvm::to_vector(tensor.getShape());
  return {};
}

FailureOr<unsigned> getIntegerWidth(Type type, AnalysisOptions options) {
  type = getIntegerElementType(type);
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (!integer.isSignless())
      return failure();
    return integer.getWidth();
  }
  if (!type.isIndex() || !options.indexBitWidth ||
      (*options.indexBitWidth != 32 && *options.indexBitWidth != 64))
    return failure();
  return *options.indexBitWidth;
}

IntegerAttr scalarIntegerAttr(Type type, const llvm::APInt &value,
                              OpBuilder &builder) {
  // Index arithmetic uses the configured target width, but MLIR attributes
  // always store index values in 64 bits.
  return builder.getIntegerAttr(
      type, type.isIndex()
                ? value.sextOrTrunc(IndexType::kInternalStorageBitWidth)
                : value);
}

FailureOr<OpFoldResult> typedZero(Type scalarType, OpBuilder &builder) {
  if (scalarType.isIndex())
    return OpFoldResult(builder.getIndexAttr(0));
  auto integer = dyn_cast<IntegerType>(scalarType);
  if (!integer || !integer.isSignless())
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

llvm::ConstantRange getValueRange(Value value, unsigned width) {
  FailureOr<llvm::APInt> constant =
      getIntegerConstant(OpFoldResult(value), width);
  if (succeeded(constant))
    return llvm::ConstantRange(*constant);
  return llvm::ConstantRange::getFull(width);
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
                                           AnalysisOptions options,
                                           OpBuilder &builder, Location loc) {
  if (!resultType ||
      (isa<ShapedType>(resultType) && !isa<RankedTensorType>(resultType)))
    return failure();
  if (auto tensor = dyn_cast<RankedTensorType>(resultType))
    if (!tensor.hasStaticShape())
      return failure();
  FailureOr<unsigned> width = getIntegerWidth(resultType, options);
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
    if (auto tensor = dyn_cast<RankedTensorType>(resultType)) {
      if (tensor.getElementType().isIndex())
        result = result.sextOrTrunc(IndexType::kInternalStorageBitWidth);
      return OpFoldResult(DenseIntElementsAttr::get(tensor, result));
    }
    return OpFoldResult(scalarIntegerAttr(resultType, result, builder));
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
      return OpFoldResult(DenseIntElementsAttr::get(
          tensor, llvm::APInt(tensor.getElementType().isIndex()
                                  ? IndexType::kInternalStorageBitWidth
                                  : *width,
                              0)));
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
                                         CastKind kind, AnalysisOptions options,
                                         OpBuilder &builder, Location loc) {
  auto sourceType = dyn_cast<Value>(source)
                        ? cast<Value>(source).getType()
                        : cast<TypedAttr>(cast<Attribute>(source)).getType();
  FailureOr<unsigned> oldWidth = getIntegerWidth(sourceType, options);
  FailureOr<unsigned> newWidth = getIntegerWidth(targetType, options);
  if (failed(oldWidth) || failed(newWidth))
    return failure();
  if (sourceType == targetType)
    return source;
  FailureOr<llvm::APInt> constant = getIntegerConstant(source, *oldWidth);
  if (succeeded(constant)) {
    llvm::APInt converted = *constant;
    if (*newWidth < *oldWidth)
      converted = converted.trunc(*newWidth);
    else if (kind == CastKind::Unsigned)
      converted = converted.zextOrTrunc(*newWidth);
    else
      converted = converted.sextOrTrunc(*newWidth);
    if (auto tensor = dyn_cast<RankedTensorType>(targetType)) {
      if (tensor.getElementType().isIndex())
        converted = converted.sextOrTrunc(IndexType::kInternalStorageBitWidth);
      return OpFoldResult(DenseIntElementsAttr::get(tensor, converted));
    }
    return OpFoldResult(scalarIntegerAttr(targetType, converted, builder));
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
      if (kind == CastKind::Unsigned)
        return OpFoldResult(
            builder.create<arith::IndexCastUIOp>(loc, targetType, *value)
                .getResult());
      return OpFoldResult(
          builder.create<arith::IndexCastOp>(loc, targetType, *value)
              .getResult());
    }
  }
  IntegerExtensionKind extension = kind == CastKind::Unsigned
                                       ? IntegerExtensionKind::Unsigned
                                       : IntegerExtensionKind::Signed;
  FailureOr<Value> result =
      castIntegerLike(builder, loc, *value, targetType, extension);
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

} // namespace mlir::triton::ptr::detail
