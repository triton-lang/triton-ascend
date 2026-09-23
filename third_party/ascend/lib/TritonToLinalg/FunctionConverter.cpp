/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "ascend/include/TritonToLinalg/FunctionConverter.h"
#include "ascend/include/TritonToLinalg/BlockPtrAnalysis.h"
#include "ascend/include/TritonToLinalg/TritonOpConverter.h"
#include "ascend/include/Utils/DebugUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace FunctionConverter {
using namespace mlir;
using namespace triton;

static bool isScalarPointer(Type type) {
  auto pointerType = dyn_cast<triton::PointerType>(type);
  return pointerType && !isa<ShapedType>(pointerType.getPointeeType());
}

static LogicalResult
convertFunctionResultTypes(TypeRange types, const TypeConverter &typeConverter,
                           SmallVectorImpl<Type> &convertedTypes,
                           Builder &builder) {
  for (Type type : types) {
    // A returned pointer carries an address, not a buffer to allocate or copy.
    // Keep it scalar across the call and reconstruct its memref at the use
    // site.
    if (isScalarPointer(type))
      convertedTypes.push_back(builder.getI64Type());
    else if (failed(typeConverter.convertType(type, convertedTypes)))
      return failure();
  }
  return success();
}

LogicalResult
FuncOpConverter::matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  FunctionType type = op.getFunctionType();
  TypeConverter::SignatureConversion inputs(type.getNumInputs());
  SmallVector<Type> results;
  const TypeConverter &converter = *getTypeConverter();
  if (failed(converter.convertSignatureArgs(type.getInputs(), inputs)) ||
      failed(convertFunctionResultTypes(type.getResults(), converter, results,
                                        rewriter)) ||
      failed(rewriter.convertRegionTypes(&op.getBody(), converter, &inputs)))
    return failure();

  rewriter.modifyOpInPlace(op, [&] {
    op.setType(FunctionType::get(op.getContext(), inputs.getConvertedTypes(),
                                 results));
  });
  return success();
}

static FailureOr<Value>
convertFunctionOperand(Value value, Type originalType,
                       const TypeConverter &typeConverter, Location loc,
                       ConversionPatternRewriter &rewriter) {
  Type expectedType = typeConverter.convertType(originalType);
  if (value.getType() == expectedType &&
      !value.getDefiningOp<UnrealizedConversionCastOp>())
    return value;

  auto pointerType = dyn_cast<triton::PointerType>(originalType);
  auto memrefType = dyn_cast_or_null<MemRefType>(expectedType);
  if (!pointerType || isa<ShapedType>(pointerType.getPointeeType()) ||
      !memrefType)
    return failure();

  // A view such as ptr + shift carries its displacement in the memref layout.
  // Rebase the full address before matching the function's identity memref
  // type.
  FailureOr<Value> address =
      TTOpConverters::materializePointerAddress(value, loc, rewriter);
  if (failed(address))
    return failure();
  return createScalarPointerCast(rewriter, loc, memrefType, *address)
      .getResult();
}

LogicalResult
CallOpConverter::matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  SmallVector<Type> resultTypes;
  if (failed(convertFunctionResultTypes(
          op.getResultTypes(), *getTypeConverter(), resultTypes, rewriter)))
    return failure();

  auto caller = op->getParentOfType<FunctionOpInterface>();
  constexpr unsigned programInfoArgCount =
      2 * (getMaxEnumValForProgramIDDim() + 1);
  SmallVector<Value> operands;
  for (auto [value, originalType] :
       llvm::zip_equal(adaptor.getOperands(), op.getOperandTypes())) {
    FailureOr<Value> converted = convertFunctionOperand(
        value, originalType, *getTypeConverter(), op.getLoc(), rewriter);
    if (failed(converted))
      return rewriter.notifyMatchFailure(op, "could not convert call operand");
    operands.push_back(*converted);
  }
  // addProgramInfo appends grid sizes and program IDs to every Triton function.
  llvm::append_range(operands,
                     caller.getArguments().take_back(programInfoArgCount));
  auto call = rewriter.create<func::CallOp>(op.getLoc(), op.getCallee(),
                                            resultTypes, operands);
  SmallVector<Value> results;
  for (auto [value, originalType] :
       llvm::zip_equal(call.getResults(), op.getResultTypes())) {
    Value replacement = value;
    if (isScalarPointer(originalType)) {
      auto memrefType =
          cast<MemRefType>(getTypeConverter()->convertType(originalType));
      replacement =
          createScalarPointerCast(rewriter, op.getLoc(), memrefType, value)
              .getResult();
    }
    results.push_back(replacement);
  }
  rewriter.replaceOp(op, results);
  return success();
}

LogicalResult
ReturnOpConverter::matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  SmallVector<Value> operands;
  for (auto [value, originalType] :
       llvm::zip_equal(adaptor.getOperands(), op.getOperandTypes())) {
    FailureOr<Value> converted =
        isScalarPointer(originalType)
            ? TTOpConverters::materializePointerAddress(value, op.getLoc(),
                                                        rewriter)
            : convertFunctionOperand(value, originalType, *getTypeConverter(),
                                     op.getLoc(), rewriter);
    if (failed(converted))
      return rewriter.notifyMatchFailure(op,
                                         "could not convert returned value");
    operands.push_back(*converted);
  }
  rewriter.replaceOpWithNewOp<triton::ReturnOp>(op, operands);
  return success();
}

LogicalResult GetProgramIDConverter::matchAndRewrite(
    triton::GetProgramIdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto axis = (uint32_t)op.getAxis();
  assert(axis < GetProgramIDConverter::LAUNCH_GRID_RANK &&
         "Invalid axis for GetProgramIdOp");
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto numArgs = func.getNumArguments();
  auto id = func.getArgument(numArgs - GetProgramIDConverter::LAUNCH_GRID_RANK +
                             axis);

  Location pidLoc = op.getLoc();
  insertDebugNop(pidLoc, rewriter);
  rewriter.replaceOp(op, id);
  return success();
}

LogicalResult GetNumProgramsConverter::matchAndRewrite(
    triton::GetNumProgramsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto axis = (uint32_t)op.getAxis();
  assert(axis < GetNumProgramsConverter::LAUNCH_GRID_RANK &&
         "Invalid axis for GetNumProgramsOp");
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto numArgs = func.getNumArguments();
  auto id = func.getArgument(
      numArgs - GetNumProgramsConverter::LAUNCH_GRID_RANK * 2 + axis);

  Location numProgsLoc = op.getLoc();
  insertDebugNop(numProgsLoc, rewriter);
  rewriter.replaceOp(op, id);
  return success();
}
} // namespace FunctionConverter
