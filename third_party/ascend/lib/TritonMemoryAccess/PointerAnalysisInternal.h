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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_INTERNAL_H
#define TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_INTERNAL_H

#include "TritonMemoryAccess/PointerAnalysis.h"
#include "TritonMemoryAccess/PointerAnalysisTransfer.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/IR/ConstantRange.h"

namespace mlir::triton::pointer::detail {

inline bool supportsAddressAnalysis(AnalysisOptions options) {
  return options.addressBitWidth == 64;
}

// Constraints of the current tt.make_range/expand/broadcast materialization.
bool canMaterializeAddressTensor(RankedTensorType type);

bool isBlockPointerType(Type type);
bool validBlockPointerBinding(Value value, const PointerComponents &components);
FailureOr<SmallVector<AnalysisRequest>>
collectBlockPointerInputs(AnalysisRequest request);
FailureOr<AnalysisResult> transferBlockPointer(AnalysisRequest request,
                                               ArrayRef<AnalysisResult> inputs,
                                               OpBuilder &builder);

enum class BinaryKind { Add, Subtract, Multiply };
enum class CastKind { Signed, Unsigned, Truncate };

FailureOr<unsigned> getIntegerWidth(Type type, AnalysisOptions options);
Type getIntegerElementType(Type type);
SmallVector<int64_t> getLogicalShape(Type type);
IntegerAttr scalarIntegerAttr(Type type, const llvm::APInt &value,
                              OpBuilder &builder);
FailureOr<OpFoldResult> typedZero(Type scalarType, OpBuilder &builder);
FailureOr<llvm::APInt> getIntegerConstant(OpFoldResult value, unsigned width);
llvm::ConstantRange getValueRange(Value value, unsigned width);
FailureOr<OpFoldResult> foldOrCreateBinary(BinaryKind kind, OpFoldResult lhs,
                                           OpFoldResult rhs, Type resultType,
                                           AnalysisOptions options,
                                           OpBuilder &builder, Location loc);
FailureOr<OpFoldResult> foldOrCreateCast(OpFoldResult source, Type targetType,
                                         CastKind kind, AnalysisOptions options,
                                         OpBuilder &builder, Location loc);
FailureOr<Value> materialize(OpFoldResult value, Type expectedType,
                             OpBuilder &builder, Location loc);
FailureOr<Value> splatTo(Value scalar, RankedTensorType target,
                         OpBuilder &builder, Location loc);
FailureOr<Value> materializeAxisRange(RankedTensorType target, unsigned axis,
                                      OpBuilder &builder, Location loc);
LogicalResult
mapOffsetValues(OffsetComponents &components,
                llvm::function_ref<FailureOr<Value>(Value)> mapper);
LogicalResult mapOfr(OpFoldResult &value,
                     llvm::function_ref<FailureOr<Value>(Value)> mapper);

} // namespace mlir::triton::pointer::detail

#endif // TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_INTERNAL_H
