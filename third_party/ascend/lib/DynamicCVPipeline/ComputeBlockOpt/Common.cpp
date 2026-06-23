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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "compute-block-opt-common";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;

namespace mlir {
namespace CVPipeline {

namespace {

//===----------------------------------------------------------------------===//
// Cycle detection
//
// Merging operations into a single block_id is only valid if the resulting
// block-level dependency graph stays acyclic. The DFS below walks the SSA +
// memory dependency edges, expanding any visited op into every op sharing its
// block_id, and reports whether a cycle (a path back into the safe set) exists.
//===----------------------------------------------------------------------===//

struct CycleDfs {
    llvm::DenseSet<mlir::Operation *> &okSet;
    llvm::DenseSet<mlir::Operation *> visited;
    const MemoryDependenceGraph &memGraph;
    ComputeBlockIdManager &bm;
    Block *block;
    void clear() { visited.clear(); }
    bool operator()(Operation *cur);
    bool dfs(Operation *cur) { return (*this)(cur); };
    CycleDfs(Block *block, const MemoryDependenceGraph &memGraph, llvm::DenseSet<mlir::Operation *> &okSet,
             ComputeBlockIdManager &bm)
        : okSet(okSet), memGraph(memGraph), bm(bm), block(block)
    {
    }
};

bool CycleDfs::operator()(Operation *cur)
{
    if (okSet.contains(cur)) {
        return true;
    }
    if (!visited.insert(cur).second) {
        return false;
    }

    SmallVector<Operation *> allusers;
    allusers.append(cur->getUsers().begin(), cur->getUsers().end());
    for (auto *memUser : memGraph.getExecAfter(cur)) {
        allusers.push_back(memUser);
    }
    for (auto *user : allusers) {
        auto *userInBlock = getAncestorInBlock(user, block);
        if (!userInBlock)
            continue;
        if (okSet.contains(userInBlock)) {
            LOG_DEBUG("[CycleDfs] Cycle found, userInBlock in okSet: " << *userInBlock);
            return true;
        }
        int userBlockId = bm.getBlockIdByOp(userInBlock);
        if (userBlockId == -1) {
            if (dfs(userInBlock)) {
                return true;
            }
        } else {
            for (auto *nx : bm.getOpsByBlockId(userBlockId)) {
                if (dfs(nx)) {
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace

bool willCreateCycle(llvm::ArrayRef<Operation *> opsToUnify, const MemoryDependenceGraph &memGraph, int targetBlockId,
                     ComputeBlockIdManager &bm)
{
    if (opsToUnify.empty()) {
        return false;
    }

    auto *block = opsToUnify.front()->getBlock();

    llvm::DenseSet<Operation *> okSet;
    for (auto *op : bm.getOpsByBlockId(targetBlockId)) {
        okSet.insert(op);
    }
    for (auto *op : opsToUnify) {
        okSet.insert(op);
    }

    DenseMap<Operation *, int> origBlockIdMap;
    for (auto *op : opsToUnify) {
        auto optBlockId = getOpBlockId(op);
        origBlockIdMap[op] = optBlockId ? static_cast<int>(*optBlockId) : -1;
        bm.updateBlockId(op, targetBlockId);
    }

    // Initialize DFS detector
    CycleDfs dfs(block, memGraph, okSet, bm);
    bool hasCycle = false;

    for (mlir::Operation *okOp : okSet) {
        SmallVector<Operation *> allusers;
        allusers.append(okOp->getUsers().begin(), okOp->getUsers().end());
        for (auto *memUser : memGraph.getExecAfter(okOp)) {
            allusers.push_back(memUser);
        }
        for (auto *user : allusers) {
            auto *userInBlock = getAncestorInBlock(user, block);
            if (!userInBlock)
                continue;
            if (okSet.contains(userInBlock)) {
                continue;
            }
            int userBlockId = bm.getBlockIdByOp(userInBlock);
            if (userBlockId == -1) {
                dfs.clear();
                if (dfs(userInBlock)) {
                    hasCycle = true;
                    break;
                }
            } else {
                LOG_DEBUG("userInBlock: " << *userInBlock);
                LOG_DEBUG("okOp: " << *okOp);
                for (auto *userOp : bm.getOpsByBlockId(userBlockId)) {
                    dfs.clear();
                    LOG_DEBUG("userOp: " << *userOp);
                    if (dfs(userOp)) {
                        hasCycle = true;
                        break;
                    }
                }
            }
        }
        if (hasCycle) {
            break;
        }
    }

    for (auto &[op, origBlockId] : origBlockIdMap) {
        bm.updateBlockId(op, origBlockId);
    }

    return hasCycle;
}

} // namespace CVPipeline
} // namespace mlir
