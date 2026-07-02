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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"

static constexpr const char *DEBUG_TYPE = "compute-block-opt-common";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;

namespace mlir {
namespace CVPipeline {

llvm::LogicalResult tryUpdate(llvm::ArrayRef<Operation *> newOpsForGroup,
                              const MemoryDependenceGraph &memGraph,
                              int targetBlockId,
                              ComputeBlockIdManager &bm)
{
    if (newOpsForGroup.empty()) {
        return success();
    }

    auto *block = newOpsForGroup.front()->getBlock();

    llvm::DenseSet<Operation *> newGroup {newOpsForGroup.begin(), newOpsForGroup.end()};
    for (auto *op : bm.getOpsByBlockId(targetBlockId)) {
        newGroup.insert(op);
    }

    DenseMap<Operation *, int> origBlockIdMap;
    for (auto *op : newOpsForGroup) {
        origBlockIdMap[op] = getOpBlockId(op).value_or(-1);
        bm.updateBlockId(op, targetBlockId);
    }

    // Initialize DFS detector
    DependencyCycleDetector dfs {block, DependencyHelper {memGraph}, newGroup, bm};
    bool hasCycle = dfs.detectCycle();

    if (!hasCycle) {
        // we have already updated block id, safely return success;
        return success();
    }

    // will create cycle, restore original block id
    for (auto &[op, origBlockId] : origBlockIdMap) {
        bm.updateBlockId(op, origBlockId);
    }
    return llvm::failure();
}

} // namespace CVPipeline
} // namespace mlir
