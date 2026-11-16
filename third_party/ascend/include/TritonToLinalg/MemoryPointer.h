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

#ifndef TRITON_TO_LINALG_MEMORY_POINTER_H
#define TRITON_TO_LINALG_MEMORY_POINTER_H

#include "TritonMemoryAccess/PtrAnalysisInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::hivm {
class PointerCastOp;
}

namespace mlir::triton {
inline constexpr llvm::StringLiteral kScalarPointerCarrierAttr =
    "ScalarPointerCarrier";
hivm::PointerCastOp createScalarPointerCast(OpBuilder &builder, Location loc,
                                            MemRefType resultType,
                                            Value address);

// Facts belong to the complete source IR of one memory-conversion stage.
// Conversion consumes remapped SSA fields; it never resumes the analysis.
using MemoryPointerInfos = llvm::DenseMap<Value, ptr::PtrInfo>;
LogicalResult collectMemoryPointerInfos(ModuleOp module,
                                        MemoryPointerInfos &infos);
FailureOr<Value> materializeMemoryPointer(Value pointer, Value converted,
                                          const MemoryPointerInfos &infos,
                                          ConversionPatternRewriter &rewriter);
FailureOr<Value> materializePointerAddress(Value memref, Location loc,
                                           ConversionPatternRewriter &rewriter);
// Checked coordinates can be clamped in the GM view while padding remains
// relative to the original tile. Keep that choice local to this access.
struct BoundaryCheckedPointer {
  Value view;
  SmallVector<int32_t> zeroBasedAxes;
};
FailureOr<BoundaryCheckedPointer> materializeBoundaryCheckedMemoryPointer(
    Value pointer, Value converted, ArrayRef<int32_t> checks,
    const MemoryPointerInfos &infos, ConversionPatternRewriter &rewriter);

struct MemoryBoundary {
  SmallVector<OpFoldResult> gmStarts;
  SmallVector<OpFoldResult> tileStarts;
  SmallVector<OpFoldResult> sizes;
};
FailureOr<MemoryBoundary> getMemoryBoundary(
    Value pointer, ArrayRef<int32_t> checks, ArrayRef<int32_t> zeroBasedAxes,
    const MemoryPointerInfos &infos, ConversionPatternRewriter &rewriter);
} // namespace mlir::triton
#endif
