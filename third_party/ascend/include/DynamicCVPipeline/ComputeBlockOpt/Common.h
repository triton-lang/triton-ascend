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

// ============================================================================
// Function Name: mlir::CVPipeline::tryUpdate
// ============================================================================
/**
 * @brief Safely updates the block ID for a collection of operations after cycle verification.
 *
 * **Purpose**:
 * To assign a new scheduling block ID to a set of operations, ensuring that the assignment
 * does not introduce any invalid cyclical dependencies in the block-level execution graph.
 *
 * **Inputs & Assumptions**:
 * - `ops` (llvm::ArrayRef<Operation *>): The operations to be updated.
 * - `memGraph` (const MemoryDependenceGraph &): The memory dependence graph for safety verification.
 * - `targetBlockId` (int64_t): The destination block ID.
 * - `bm` (ComputeBlockIdManager &): The block ID manager handling the state updates.
 *
 * **Outputs & Guarantees**:
 * - Returns `llvm::success()` if the updates were verified as cycle-free and successfully applied.
 * - Returns `llvm::failure()` if the update would introduce a cycle.
 * - Guarantees transactional behavior: if validation fails, the system state remains unmodified.
 */
llvm::LogicalResult tryUpdate(llvm::ArrayRef<Operation *> newOpsForGroup,
                              const MemoryDependenceGraph &memGraph,
                              int targetBlockId,
                              ComputeBlockIdManager &bm);

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_OPT_COMMON_H
