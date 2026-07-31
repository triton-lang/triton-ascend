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

#ifndef TRITON_ADAPTER_GATHER_OPTIMIZATION_PASSES_H
#define TRITON_ADAPTER_GATHER_OPTIMIZATION_PASSES_H

#include "ascend/include/TritonToUnstructure/OffsetAnalysis.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass to perform optimization for gather-like workloads.
/// This pass identifies gather patterns in load operations and rewrites
/// them using efficient gather operations with runtime bounds checking.
std::unique_ptr<OperationPass<ModuleOp>> createGatherOptimizationPass();

#define GEN_PASS_REGISTRATION
#include "ascend/include/GatherOptimization/Passes.h.inc"

struct GatherOptimizationConversionPattern
    : OpRewritePattern<mlir::triton::LoadOp> {
  using OpRewritePattern<mlir::triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const final;

  bool isIntegerTensorType(mlir::Type type, int &numDimensionsOut) const;
  bool tryOptimise(triton::LoadOp loadOp, PatternRewriter &rewriter) const;
  /// @brief Analyze operation tree and retrieve all the required variables.
  /// @param offsetMap Map of PtrOffsetInfo for each operation.
  /// @param analyzedOp Initial analyzed operation.
  /// @param ourIndices Retrieved tensor of indices.
  /// @param indexRank Retrieved rank of index tensor.
  /// @param shape Determined source tensor shape.
  /// @param rowOffset Retrieved tensor defining row offsets.
  /// @param srcPtr Retrieved pointer to source tensor.
  /// @param gatherAxis Determined axis on which gather op is performed.
  /// @return true if all the required variables are defined.
  bool analyze(llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
               Operation *analyzedOp, mlir::Value &ourIndices, int &indexRank,
               llvm::SmallVector<int64_t> &shape, mlir::Value &rowOffset,
               mlir::Value &srcPtr, int &gatherAxis) const;
  ArrayRef<int64_t> getShapeFromType(mlir::Type type) const;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_GATHER_OPTIMIZATION_PASSES_H
