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

#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

namespace mlir {
namespace CVPipeline {

void DependencyHelper::forEachUser(Operation *op, DependencyHelper::PredFn pred) const
{
    for (auto *user : op->getUsers()) {
        pred(user);
    }
    for (auto *user : memGraph.getExecAfter(op)) {
        pred(user);
    }
}

template <bool AcrossIterArg> void DependencyHelper::forEachSource(Operation *op, DependencyHelper::PredFn pred) const
{
    auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(op->getBlock()->getParentOp());
    op->walk([&, this](Operation *subOp) {
        for (auto operand : subOp->getOperands()) {
            if (auto *defOp = operand.getDefiningOp()) {
                pred(defOp);
                continue;
            }

            if constexpr (AcrossIterArg) {
                if (!forOp) {
                    continue;
                }
                auto blockArg = llvm::dyn_cast<BlockArgument>(operand);
                auto *yieldedVal = forOp.getTiedLoopYieldedValue(blockArg);
                // this also ensures blockArg.getOwner() == op->getBlock()
                if (!yieldedVal) {
                    continue;
                }
                if (auto *defOp = yieldedVal->get().getDefiningOp()) {
                    pred(defOp);
                }
            }
        }
        for (auto *source : memGraph.getExecBefore(subOp)) {
            pred(source);
        }
    });
}

// Instantiate concrete functions for linking
template void mlir::CVPipeline::DependencyHelper::forEachSource<true>(
    mlir::Operation *op, llvm::function_ref<void(mlir::Operation *)> callback) const;

template void mlir::CVPipeline::DependencyHelper::forEachSource<false>(
    mlir::Operation *op, llvm::function_ref<void(mlir::Operation *)> callback) const;

void initializeIndegreeForBlock(Block *block,
                                llvm::DenseMap<Operation *, int> &indegree,
                                const DependencyHelper &depHelper,
                                ComputeBlockIdManager &bm)
{
    for (auto *op : llvm::make_pointer_range(block->getOperations())) {
        indegree[op] = 0;
        depHelper.forEachSource<false>(op, [&](Operation *source) {
            if (source->getBlock() == block && !bm.isSameBlock(source, op)) {
                indegree[op]++;
            }
        });
    }
}

Operation *getAncestorInBlock(Operation *inner, Block *block)
{
    Operation *cur = inner;
    while (cur) {
        if (cur->getBlock() == block) {
            return cur;
        }
        cur = cur->getParentOp();
    }
    return nullptr;
}

} // namespace CVPipeline
} // namespace mlir
