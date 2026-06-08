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

#include <memory>
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

static constexpr const char *DEBUG_TYPE = "plan-vector-block";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

static bool isFusableOp(Operation *op)
{
    if (isVectorSimpleOpOrCf(op)) {
        // skip terminators
        if (op->getBlock()->mightHaveTerminator() && op == op->getBlock()->getTerminator()) {
            return false;
        }
        return true;
    }
    return false;
}

static void passAndCollectCandidates(Operation *nowOp,
                                     DenseMap<Operation *, int> &indegree,
                                     SmallVector<Operation *> &candidates,
                                     DenseMap<Operation *, bool> &visited,
                                     const CVPipeline::MemoryDependenceGraph &memGraph,
                                     ComputeBlockIdManager &bm)
{
    LOG_DEBUG("Bypassing non-fusable op " << *nowOp << "\nnow candidates size: " << candidates.size() << "\n");
    DependencyHelper depHelper {memGraph};
    depHelper.forEachUserInSameBlock(nowOp, [&](Operation *user) {
        if (!bm.isSameBlock(user, nowOp)) {
            indegree[user]--;
        }

        if (!bm.isWholeCubeReady(user, indegree)) {
            return;
        }
        if (!isFusableOp(user) && !visited[user]) {
            if (bm.getBlockIdByOp(user) == -1) {
                visited[user] = true; // mark as fused to avoid duplicate bypass
                passAndCollectCandidates(user, indegree, candidates, visited, memGraph, bm);
            } else {
                for (auto cubeop : bm.getOpsByBlockId(bm.getBlockIdByOp(user))) {
                    if (!visited[cubeop]) {
                        visited[cubeop] = true;
                        passAndCollectCandidates(cubeop, indegree, candidates, visited, memGraph, bm);
                    }
                }
            }
        } else if (isFusableOp(user) && !visited[user]) {
            visited[user] = true;
            candidates.push_back(user);
        }
    });

    LOG_DEBUG("After bypassing, candidates size: " << candidates.size() << "\n");
}

static void byPassNonFusable(DenseMap<Operation *, int> &indegree,
                             SmallVector<Operation *> &candidates,
                             DenseMap<Operation *, bool> &visited,
                             const CVPipeline::MemoryDependenceGraph &memGraph,
                             ComputeBlockIdManager &bm)
{
    // for every non-fusable candidates, bypass it.
    for (auto &[op, degree] : indegree) {
        if (bm.isWholeCubeReady(op, indegree) && !isFusableOp(op) && !visited[op]) {
            if (bm.getBlockIdByOp(op) == -1) {
                visited[op] = true; // mark as fused to avoid duplicate bypass
                passAndCollectCandidates(op, indegree, candidates, visited, memGraph, bm);
            } else {
                for (auto cubeop : bm.getOpsByBlockId(bm.getBlockIdByOp(op))) {
                    if (!visited[cubeop]) {
                        visited[cubeop] = true;
                        passAndCollectCandidates(cubeop, indegree, candidates, visited, memGraph, bm);
                    }
                }
            }
        }
    }
}

static void updateCandidates(Operation *nextFused,
                             SmallVector<Operation *> &candidates,
                             DenseMap<Operation *, int> &indegree,
                             DenseMap<Operation *, bool> &visited,
                             const CVPipeline::MemoryDependenceGraph &memGraph)
{
    // 1. Already fuse with nextFused, so remove it from candidates
    for (auto it = candidates.begin(); it != candidates.end(); it++) {
        if (*it == nextFused) {
            it = candidates.erase(it);
            break;
        }
    }

    // 2. Add new candidates whose indegree becomes 0 after fusing nextFused.
    DependencyHelper depHelper {memGraph};
    depHelper.forEachUserInSameBlock(nextFused, [&](Operation *user) {
        if (!visited[user]) {
            indegree[user]--;
            if (indegree[user] == 0 && isFusableOp(user)) {
                visited[user] = true;
                candidates.push_back(user);
            }
        }
    });
}

static void findCandidates(DenseMap<Operation *, int> &indegree,
                           SmallVector<Operation *> &candidates,
                           DenseMap<Operation *, bool> &visited,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           ComputeBlockIdManager &bm)
{
    // 1. if no candidate, try to bypass non-fusable
    LOG_DEBUG("Finding source ops............\n");
    if (candidates.empty()) {
        LOG_DEBUG("No candidates available, try bypass\n");
        byPassNonFusable(indegree, candidates, visited, memGraph, bm);
    }
    // 2. find candidates whose indegree is 0 and not visited, add them to candidates and mark visited
    for (auto &[op, degree] : indegree) {
        if (degree == 0 && isFusableOp(op) && !visited[op]) {
            visited[op] = true;
            candidates.push_back(op);
        }
    }
    LOG_DEBUG("end finding source ops............\n");
}

static SmallVector<Operation *> findOpsAdjacentToCube(Block *block, const SmallVector<Operation *> &fuseGroup,
                                                      DenseMap<Operation *, bool> &visited,
                                                      const CVPipeline::MemoryDependenceGraph &memGraph)
{
    SmallVector<Operation *> toProcess;
    DependencyHelper depHelper {memGraph};
    for (Operation *op : fuseGroup) {
        depHelper.forEachUserInSameBlock(op, [&](Operation *user) {
            if (block->mightHaveTerminator() && user == block->getTerminator()) {
                return;
            }
            if (!isFusableOp(user) && !visited[user]) {
                toProcess.push_back(op);
            }
        });
    }
    return toProcess;
}

static int getLoopCarriedArgIndex(Value operand, Block *block)
{
    auto barg = dyn_cast<BlockArgument>(operand);
    if (!barg || barg.getOwner() != block || !isa<scf::ForOp>(block->getParentOp())) {
        return -1;
    }
    unsigned argIdx = barg.getArgNumber();
    if (argIdx == 0) {
        return -1;
    }
    return argIdx;
}

static SetVector<Operation *> collectKeepOps(Block *block, SmallVector<Operation *> toProcess,
                                             const SmallVector<Operation *> &fuseGroup,
                                             const CVPipeline::MemoryDependenceGraph &memGraph)
{
    SetVector<Operation *> keepOps;
    DependencyHelper depHelper {memGraph};
    while (!toProcess.empty()) {
        Operation *op = toProcess.front();
        toProcess.erase(toProcess.begin());
        if (keepOps.contains(op)) {
            continue;
        }
        keepOps.insert(op);

        depHelper.forEachSource<true>(op, [&](Operation *source) {
            if (!keepOps.contains(source) && llvm::is_contained(fuseGroup, source)) {
                toProcess.push_back(source);
            }
        });
    }

    // special case: annotation.mark always follows the defining op
    for (auto *op : fuseGroup) {
        auto markOp = llvm::dyn_cast<annotation::MarkOp>(op);
        if (!markOp) {
            continue;
        }

        auto src = markOp.getSrc();
        auto *definingOp = src.getDefiningOp();
        if (definingOp && keepOps.contains(definingOp)) {
            keepOps.insert(markOp);
        }
    }

    return keepOps;
}

static void evictAndRestoreState(Block *block, const SetVector<Operation *> &keepOps,
                                 SmallVector<Operation *> &fuseGroup, DenseMap<Operation *, bool> &visited,
                                 SmallVector<Operation *> &candidates, DenseMap<Operation *, int> &indegree,
                                 const CVPipeline::MemoryDependenceGraph &memGraph)
{
    // 1. Collect ops to remove
    SmallVector<Operation *> toRemove;
    for (Operation *op : fuseGroup) {
        if (!keepOps.contains(op)) {
            toRemove.push_back(op);
        }
    }
    fuseGroup.assign(keepOps.begin(), keepOps.end());

    DependencyHelper depHelper {memGraph};
    // 2. Restore indegree for successors and reset visited for removed ops
    for (Operation *op : toRemove) {
        depHelper.forEachUserInSameBlock(op, [&](Operation *user) { indegree[user]++; });
        visited[op] = false;
    }

    // 3. Reset visited for current candidates so they can re-enter
    for (auto cand : candidates) {
        visited[cand] = false;
    }

    // 4. Rebuild candidates from (old candidates + removed ops), zero indegree only
    SmallVector<Operation *> pool(candidates.begin(), candidates.end());
    pool.append(toRemove.begin(), toRemove.end());
    candidates.clear();
    // Need to find source again
    for (Operation *op : pool) {
        if (indegree[op] == 0 && !visited[op]) {
            visited[op] = true;
            candidates.push_back(op);
        }
    }
}

static void refineFuseGroup(Block *block,
                            SmallVector<Operation *> &nowFuseGroup,
                            DenseMap<Operation *, bool> &visited,
                            SmallVector<Operation *> &candidates,
                            DenseMap<Operation *, int> &indegree,
                            const CVPipeline::MemoryDependenceGraph &memGraph,
                            ComputeBlockIdManager &bm)
{
    // 1.Find ops in fuse group whose next node is a non-fusable (CUBE-only) op
    auto toProcess = findOpsAdjacentToCube(block, nowFuseGroup, visited, memGraph);
    // 2. early return: if cannot find one node which next node is CUBE, need to check if return or not;
    if (toProcess.empty()) {
        findCandidates(indegree, candidates, visited, memGraph, bm);
        if (candidates.empty()) {
            return;
        }
    }

    // 3. Collect keepOps transitively (data + memory + loop-carried deps)
    auto keepOps = collectKeepOps(block, toProcess, nowFuseGroup, memGraph);

    // 4. Remove non-kept ops from fuseGroup and restore BFS state
    evictAndRestoreState(block, keepOps, nowFuseGroup, visited, candidates, indegree, memGraph);
    LOG_DEBUG("After cutting, kept " << keepOps.size() << "\n");
}

// Main function to plan vector block id for one block
static llvm::LogicalResult planVectorBlockId(Block *block,
                                             const CVPipeline::MemoryDependenceGraph &memGraph,
                                             ComputeBlockIdManager &bm)
{
    // 1. topo initialize
    llvm::DenseMap<Operation *, int> indegree;
    llvm::SmallVector<Operation *> queue;
    llvm::DenseMap<Operation *, bool> visited; // has been visited in search
    initializeIndegreeForBlock(block, indegree, DependencyHelper {memGraph}, bm);

    // 2. initialize visited and find initial candidates
    block->walk([&](Operation *op) {
        if (op->getBlock() == block) {
            visited[op] = false;
            if (isFusableOp(op) && indegree[op] == 0) {
                visited[op] = true;
                queue.push_back(op);
            }
        }
    });
    findCandidates(indegree, queue, visited, memGraph, bm);

    // 3. find fuse group follow topo order
    llvm::SmallVector<Operation *> nowFuseGroup;
    while (!queue.empty()) {
        auto nextFused = queue.front();
        if (nextFused) {
            // fused one && update candidates
            nowFuseGroup.push_back(nextFused);
            updateCandidates(nextFused, queue, indegree, visited, memGraph);
        }
        if (queue.empty() || nextFused == nullptr) {
            // finish one group, assign block id and start next iteration
            // Cut error operations before assigning block id
            refineFuseGroup(block, nowFuseGroup, visited, queue, indegree, memGraph, bm);

            if (llvm::failed(bm.markOpsWithNewId(nowFuseGroup))) {
                return llvm::failure();
            }
            nowFuseGroup.clear();
            // reset queue
            findCandidates(indegree, queue, visited, memGraph, bm);
        }
    }
    return llvm::success();
}

namespace {

class PlanVectorBlockPass : public PassWrapper<PlanVectorBlockPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanVectorBlockPass)

    PlanVectorBlockPass() = default;
    void runOnOperation() override;

    llvm::StringRef getArgument() const final { return "plan-vector-block"; }
};

void PlanVectorBlockPass::runOnOperation()
{
    // 1. Build memory dependence graph
    auto moduleOp = getOperation();
    auto &aa = getAnalysis<AliasAnalysis>();
    auto memDepGraph = MemoryDependenceGraph(moduleOp, aa);
    auto bm = ComputeBlockIdManager(moduleOp);

    // 2. search blocks in topo order and assign block id for each block
    llvm::SmallVector<Block *> blocks;
    moduleOp.walk([&](Block *block) {
        if (llvm::failed(planVectorBlockId(block, memDepGraph, bm))) {
            moduleOp.emitError() << "[" << DEBUG_TYPE << "] Failed to plan vector block id for block";
            signalPassFailure();
            return;
        }
    });
}

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPlanVectorBlockPass()
{
    return std::make_unique<PlanVectorBlockPass>();
}

} // namespace triton
} // namespace mlir
