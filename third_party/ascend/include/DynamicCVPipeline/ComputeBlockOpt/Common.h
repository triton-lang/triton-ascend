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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_OPT_COMMON_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_OPT_COMMON_H

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace CVPipeline {

/**
 * @brief Detect if unifying a list of operations to target block_id would create a cycle
 *
 * This helper temporarily assigns every op in @p opsToUnify to @p targetBlockId,
 * walks the SSA + memory dependency edges, and reports whether the resulting
 * block-level dependency graph would contain a cycle. The temporary block_id
 * assignments are always rolled back before returning, so the function leaves
 * @p bm in its original state regardless of the result.
 *
 * Shared by the ComputeBlockOpt passes (e.g. UnifyAllocBlockPass and
 * MergeVectorIfBlockPass) that merge operations into a common block_id.
 *
 * @param opsToUnify Block-level operations to add to the safe set (okSet)
 * @param memGraph Memory dependence graph for RAW/WAW/WAR dependency analysis
 * @param targetBlockId Target block_id after unification
 * @param bm Block-id manager used to query/temporarily mutate block ids
 * @return bool Returns true if unification would create a cycle, false otherwise
 */
bool willCreateCycle(llvm::ArrayRef<Operation *> opsToUnify,
                     const MemoryDependenceGraph &memGraph, int targetBlockId,
                     ComputeBlockIdManager &bm);

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_OPT_COMMON_H
