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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

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

/**
 * @brief Collect predecessor operations in a block for the given values
 *
 * This function traces back through SSA dependencies to find all operations
 * that affect the given startValue within a specific block.
 *
 * Special handling for scf.for loop-carried dependencies:
 * When an operand is a BlockArgument from scf.for iter_arg, we also trace
 * through the yieldOp to find the operation that provides the yielded value.
 * This ensures we capture operations that update loop-carried variables
 * (e.g., %79 that updates %arg19).
 *
 * @param startValue The value to trace back from
 * @param block The block to search within
 * @return SmallVector<Operation*> List of predecessor operations
 */
SmallVector<Operation *> collectBlockPredecessors(ValueRange startValues, Block *block);

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
 *
 * **Safety Boundaries & Constraints**:
 * - Relies on `willCreateCycle` to perform speculative validation and roll back changes on failure.
 */
llvm::LogicalResult tryUpdate(llvm::ArrayRef<Operation *> ops,
                              const MemoryDependenceGraph &memGraph,
                              int64_t targetBlockId,
                              ComputeBlockIdManager &bm);

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_OPT_COMMON_H
