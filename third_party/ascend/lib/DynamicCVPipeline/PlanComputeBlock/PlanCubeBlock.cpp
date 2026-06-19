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

#include <functional>
#include <queue>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/PlanCubeBlockPass.h"

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "PlanCubeBlock";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

static bool isMatmulOp(Operation *op)
{
    return isa<linalg::MatmulOp>(op);
}

namespace {

class SeedRegionPlanner {
    SmallVector<Operation*> seeds;
    Block *block;
    const MemoryDependenceGraph &memGraph;
    ComputeBlockIdManager &bm;
    llvm::DenseSet<Operation *> &assigned;
    llvm::SmallVectorImpl<Operation *> &group;
    bool willCreateCycle(Operation *op);
    bool isEligible(Operation *op);
    bool tryAddToGroup(Operation *op);

public:
    SeedRegionPlanner(SmallVector<Operation*> seeds,
                      Block *block,
                      const MemoryDependenceGraph &memGraph,
                      llvm::DenseSet<Operation *> &assigned,
                      llvm::SmallVectorImpl<Operation *> &group,
                      ComputeBlockIdManager &bm)
        : seeds(seeds), block(block), memGraph(memGraph), assigned(assigned), group(group), bm(bm)
    {
        for(auto sd: seeds) {
            group.push_back(sd);
        }
    }

    void run();
};

} // namespace


namespace {

class DependencyCycleDetector {
    const llvm::DenseSet<mlir::Operation *> &group;
    llvm::DenseSet<mlir::Operation *> visited;
    const MemoryDependenceGraph &memGraph;
    ComputeBlockIdManager &bm;
    Block *const block;

    bool detectCycleFrom(Operation *cur);

  public:
    DependencyCycleDetector(Block *block,
                            const MemoryDependenceGraph &memGraph,
                            llvm::DenseSet<mlir::Operation *> &group,
                            ComputeBlockIdManager &bm)
        : block(block), memGraph(memGraph), group(group), bm(bm)
    {}

    bool detectCycle();
};

} // namespace

bool DependencyCycleDetector::detectCycleFrom(Operation *cur)
{
    if (group.contains(cur)) {
        return true;
    }
    if (!visited.insert(cur).second) {
        return false;
    }

    auto userCreatesCycle = [this, cur](Operation *user) {
        auto *userInBlock = getAncestorInBlock(user, block);
        if (!userInBlock) {
            return false;
        }
        auto userBlockId = bm.getBlockIdByOp(userInBlock);
        if (userBlockId == -1) {
            return detectCycleFrom(userInBlock);
        }

        return llvm::any_of(bm.getOpsByBlockId(userBlockId), [this](Operation *user) { return detectCycleFrom(user); });
    };

    return llvm::any_of(cur->getUsers(), userCreatesCycle) ||
           llvm::any_of(memGraph.getExecAfter(cur), userCreatesCycle);
}

static void forEachUser(Operation *op,
                        const MemoryDependenceGraph &memGraph,
                        const std::function<void(Operation *op)> &pred)
{
    for (auto *user : op->getUsers()) {
        pred(user);
    }
    for (auto *user : memGraph.getExecAfter(op)) {
        pred(user);
    }
}

bool DependencyCycleDetector::detectCycle()
{
    llvm::DenseSet<Operation *> externalUsers;
    for (auto *op : group) {
        forEachUser(op, memGraph, [&](Operation *user) {
            auto *userInBlock = getAncestorInBlock(user, block);
            if (userInBlock && !group.contains(userInBlock)) {
                externalUsers.insert(userInBlock);
            }
        });
    }
    return llvm::any_of(externalUsers, [this](Operation *op) { return this->detectCycleFrom(op); });
}

bool SeedRegionPlanner::willCreateCycle(Operation *op)
{
    auto *block = op->getBlock();
    llvm::DenseSet<mlir::Operation *> okSet(group.begin(), group.end());
    okSet.insert(op);

    DependencyCycleDetector dfs = {block, memGraph, okSet, bm};
    return dfs.detectCycle();
}

/**
 * Checks if an operation is eligible to be added to a Cube group.
 * Eligibility depends on it being a cube op, not yet assigned, not a compute op,
 * and not creating a cycle in the dependence graph.
 */
bool SeedRegionPlanner::isEligible(Operation *op)
{
    if (!isCubeOp(op) || assigned.contains(op) || isMatmulOp(op)) {
        return false;
    }
    return !willCreateCycle(op);
}

bool SeedRegionPlanner::tryAddToGroup(Operation *op)
{
    if (!op || llvm::is_contained(group, op) || op->getBlock() != block || !isEligible(op)) {
        return false;
    }
    group.push_back(op);
    return true;
}

void SeedRegionPlanner::run()
{
    size_t head = 0;
    while (head < group.size()) {
        Operation *currOp = group[head++];

        // Check data operands
        for (Value iop : currOp->getOperands()) {
            if (auto *def = iop.getDefiningOp()) {
                tryAddToGroup(def);
            }
            // Check loop-carried dependencies (SCF ForOp iter_args)
            if (auto barg = dyn_cast<BlockArgument>(iop)) {
                if (barg.getOwner() == block && isa<scf::ForOp>(block->getParentOp()) && barg.getArgNumber() > 0) {
                    auto *yieldOp = barg.getOwner()->getTerminator();
                    if (auto *yieldedValDef = yieldOp->getOperand(barg.getArgNumber() - 1).getDefiningOp()) {
                        tryAddToGroup(yieldedValDef);
                    }
                }
            }
        }

        // Check memory dependencies (RAW/WAW/WAR)
        for (auto *def : memGraph.getMemDefs(currOp)) {
            tryAddToGroup(def);
        }
    }
}


namespace {

/**
 * Processes remaining unassigned cube operations by following the topological order.
 */
class TopologicalPartitionPlanner {
    Block *block;
    unsigned nonAssignedCubeCnt = 0;
    llvm::DenseMap<Operation *, int> indegree;
    llvm::DenseSet<Operation *> &assigned;
    const MemoryDependenceGraph &memGraph;
    ComputeBlockIdManager &bm;
    llvm::DenseSet<Operation *> newassigned;
    llvm::DenseSet<Operation *> bypassVisited;
    std::queue<Operation *> queue;

    void removeNonCubeOpsRecursively(Operation *op);
    llvm::LogicalResult removeReadyNonCubeOps();
    bool shouldSkip(Operation *op) { return !isCubeOp(op) || assigned.contains(op); };
    bool canExpandTo(Operation *op);
    void dumpQueueAndIndegreeInfo();
    llvm::LogicalResult populateQueueWithReadyOps();
    llvm::SmallVector<Operation *> createNewGroupFromQueue();

public:
    TopologicalPartitionPlanner(Block *block,
                                llvm::DenseSet<Operation *> &assigned,
                                const MemoryDependenceGraph &memGraph, ComputeBlockIdManager &bm)
        : block(block), assigned(assigned), memGraph(memGraph), bm(bm)
    {
        initializeIndegreeForBlock(block, indegree, memGraph, bm);

        block->walk([&](Operation *op) {
            if (op->getBlock() == block && isCubeOp(op) && !assigned.contains(op)) {
                nonAssignedCubeCnt++;
            }
        });
    }

    llvm::LogicalResult run();
};

} // namespace

// Recursively bypass non-cube ops: decrement indegree of users; collect newly-exposed cube ops
void TopologicalPartitionPlanner::removeNonCubeOpsRecursively(Operation *op)
{
    LOG_DEBUG("\tRemoved non-cube:" << *op << "\n");
    bypassVisited.insert(op);
    auto *block = op->getBlock();
    SmallVector<Operation *> allusers;
    allusers.append(op->getUsers().begin(), op->getUsers().end());
    for (auto *memUser : memGraph.getExecAfter(op)) {
        allusers.push_back(memUser);
    }
    for (auto *user : allusers) {
        auto *userInBlock = getAncestorInBlock(user, block);
        if (!userInBlock || !indegree.contains(userInBlock) ||
            bm.isSameBlock(userInBlock, op)) {
            continue;
        }
        LOG_DEBUG("Sub indegree to " << *userInBlock << " from " << *op << "new degree =  " << indegree[userInBlock] - 1
                                     << "\n");
        indegree[userInBlock]--;
        if (!bm.isWholeCubeReady(userInBlock, indegree) || bypassVisited.contains(userInBlock) ||
            !shouldSkip(userInBlock)) {
            continue;
        }
        auto blockId = bm.getBlockIdByOp(userInBlock);
        if (blockId == -1) {
            removeNonCubeOpsRecursively(userInBlock);
            continue;
        }
        for (auto *passop : bm.getOpsByBlockId(blockId)) {
            if (!bypassVisited.contains(passop)) {
                removeNonCubeOpsRecursively(passop);
            }
        }
    }
}

static bool mapsAreDiff(const llvm::DenseMap<Operation *, int> &a, const llvm::DenseMap<Operation *, int> &b)
{
    if (a.size() != b.size()) {
        return true;
    }
    return llvm::any_of(a, [&b](std::pair<Operation *, int> aIter) {
        auto bIter = b.find(aIter.first);
        return bIter == b.end() || bIter->second != aIter.second;
    });
}

/**
 * Logic to bypass non-cube operations that are ready to be executed.
 * This unblocks downstream cube operations in the topological sort.
 */
llvm::LogicalResult TopologicalPartitionPlanner::removeReadyNonCubeOps()
{
    auto indegreeBefore = indegree;
    size_t beforeVisitedSize = bypassVisited.size();
    for (auto &p : indegree) {
        Operation *op = p.first;
        if (shouldSkip(op) && bm.isWholeCubeReady(op, indegree) && !bypassVisited.contains(op)) {
            int blockId = bm.getBlockIdByOp(op);
            if (blockId == -1) {
                removeNonCubeOpsRecursively(op);
            } else {
                for (auto *passOp : bm.getOpsByBlockId(blockId)) {
                    if (!bypassVisited.contains(passOp)) {
                        removeNonCubeOpsRecursively(passOp);
                    }
                }
            }
        }
    }
    if (!mapsAreDiff(indegreeBefore, indegree) && beforeVisitedSize == bypassVisited.size()) {
        if (Operation *parentOp = block->getParentOp()) {
            parentOp->emitError("PlanCubeBlock cannot make progress while scheduling cube operations");
        }
        dumpQueueAndIndegreeInfo();
        return llvm::failure();
    }
    return llvm::success();
}

// Expansion condition: op must be CUBE_ONLY, indegree == 0 and all its dependency ops are CUBE_ONLY
bool TopologicalPartitionPlanner::canExpandTo(Operation *op)
{
    if (!isCubeOp(op) || assigned.contains(op)) {
        return false;
    }
    auto it = indegree.find(op);
    return it->second == 0;
}

// Encountered error. Need to print failure reason, so no need for LLVM_DEBUG
void TopologicalPartitionPlanner::dumpQueueAndIndegreeInfo()
{
    // simply print a debug header in a new line
    auto errs = []() -> llvm::raw_ostream& {
        return llvm::errs() << "\n[" << DEBUG_TYPE << "] ";
    };

    errs() << "failed to make progress while planning cube blocks.";
    errs() << "remaining cube count: " << nonAssignedCubeCnt;

    errs() << "ready queue";
    if (queue.empty()) {
        llvm::errs() << " is empty.";
    } else {
        llvm::errs() << ":\n";
        while (!queue.empty()) {
            Operation *op = queue.front();
            queue.pop();
            llvm::errs() << "  " << *op << "\n";
        }
    }

    errs() << "remaining unassigned cube ops:\n";
    bool foundRemainingCube = false;
    for (auto &p : indegree) {
        Operation *op = p.first;
        if (!op || op->getBlock() != block || !CVPipeline::isCubeOp(op) || assigned.contains(op) ||
            newassigned.contains(op)) {
            continue;
        }
        foundRemainingCube = true;
        llvm::errs() << "  indegree=" << p.second << " op=" << *op << "\n";
    }
    if (!foundRemainingCube) {
        llvm::errs() << "  <none>\n";
    }
}

llvm::LogicalResult TopologicalPartitionPlanner::populateQueueWithReadyOps()
{
    for (auto [op, indegree] : indegree) {
        if (indegree < 0) {
            op->emitError("Indegree cannot be negative");
            return llvm::failure();
        }
        if (indegree == 0 && !newassigned.contains(op) && isCubeOp(op) && !assigned.contains(op)) {
            queue.push(op);
        }
    }
    return llvm::success();
}

llvm::SmallVector<Operation *> TopologicalPartitionPlanner::createNewGroupFromQueue()
{
    llvm::SmallVector<Operation *> group;
    while (!queue.empty()) {
        auto *currOp = queue.front();
        queue.pop();

        newassigned.insert(currOp);
        group.push_back(currOp);
        nonAssignedCubeCnt--;

        llvm::SmallVector<Operation *> allUsers;
        for (auto *user : currOp->getUsers())
          allUsers.push_back(user);
        for (auto *user : memGraph.getExecAfter(currOp))
          allUsers.push_back(user);
        for (auto *user : allUsers) {
            auto *userInBlock = getAncestorInBlock(user, block);
            if (userInBlock && !newassigned.contains(userInBlock)) {
                auto &userInDegree = indegree[userInBlock];
                userInDegree--;
                LOG_DEBUG("Sub indegree to " << *userInBlock << " from " << *currOp << "new degree = " << userInDegree
                                             << "\n");
                if (canExpandTo(userInBlock)) {
                    queue.push(userInBlock);
                }
            }
        }
    }
    return group;
}

llvm::LogicalResult TopologicalPartitionPlanner::run()
{
    while (nonAssignedCubeCnt > 0) {
        if (failed(populateQueueWithReadyOps())) {
            return llvm::failure();
        }

        if (queue.empty()) {
            if (failed(removeReadyNonCubeOps())) {
                return llvm::failure();
            }
            continue;
        }

        auto group = createNewGroupFromQueue();
        if (llvm::failed(bm.markOpsWithNewId(group))) {
            return llvm::failure();
        }
    }

    return llvm::success();
}

static SmallVector<Operation *> collectMatmulOps(Block *block)
{
    SmallVector<Operation *> computeOps;
    for (Operation &op : *block) {
        if (isMatmulOp(&op)) {
            computeOps.push_back(&op);
        }
    }
    return computeOps;
}

static void fuseMarkOpToDef(Block *block, ComputeBlockIdManager &bm, const MemoryDependenceGraph &memGraph)
{
    for (auto *op : llvm::make_pointer_range(block->getOperations())) {
        if (getOpCoreType(op) != CUBE_ONLY) {
            continue;
        }
        auto markOp = llvm::dyn_cast<annotation::MarkOp>(op);
        if (!markOp) {
            continue;
        }
        auto *defOp = markOp.getSrc().getDefiningOp();
        if (!defOp) {
            continue;
        }

        auto defBlockId = bm.getBlockIdByOp(defOp);
        if (defBlockId == -1) {
            continue;
        }

        auto currGroup = bm.getOpsByBlockId(defBlockId);
        llvm::DenseSet<Operation *> newGroup {currGroup.begin(), currGroup.end()};

        if (newGroup.contains(markOp)) {
            continue;
        }
        newGroup.insert(markOp);

        DependencyCycleDetector dfs {block, memGraph, newGroup, bm};
        if (!dfs.detectCycle()) {
            bm.updateBlockId(markOp, defBlockId);
        }
    }
}


static bool checkValidInputSeed(Operation* op) {
    // keep unify to OpClassifer
    return isa<linalg::TransposeOp, bufferization::ToTensorOp, linalg::FillOp, tensor::EmptyOp>(op);
}
static bool checkValidUserSeed(Operation* op) {
    // keep unify to OpClassifer
    return isa<hivm::StoreOp, bufferization::MaterializeInDestinationOp, ViewLikeOpInterface, tensor::ExtractSliceOp>(op);
}
SmallVector<Operation*> PlanCubeBlockPass::matchSeed(Operation* dotOp, ComputeBlockIdManager &bm)
{
    // match inputs
    SmallVector<Operation*> ret;
    ret.push_back(dotOp);
    for (Value operand : dotOp->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
            continue;
        if (checkValidInputSeed(def) && isCubeOp(def) && dotOp->getBlock() == def->getBlock() && bm.getBlockIdByOp(def) == -1) {
            ret.push_back(def);
        }
    }
    // match outputs
    Operation* nowOp = dotOp;
    while (nowOp->hasOneUse()) {
        auto user = *nowOp->getUsers().begin();
        if(user->getBlock() != dotOp->getBlock() || !isCubeOp(user) || bm.getBlockIdByOp(user) != -1){
            break;
        }
        if (checkValidUserSeed(user)) {
            nowOp = user;
            ret.push_back(user);
        } else {
            break;
        }
    }
    return ret;
}

/**
 * Main entry point: Process a single block by grouping operations into
 * execution blocks using BFS and topological traversal.
 */
llvm::LogicalResult PlanCubeBlockPass::processBlockWithCubeBFS(Block *block, const MemoryDependenceGraph &memGraph, ComputeBlockIdManager &bm)
{
    llvm::DenseSet<Operation *> assigned;
    auto allDots = collectMatmulOps(block);

    // Phase 1: Add helper ops (transpose, load/store, ptr etc.) to cube block of related matmul
    for (auto *dot : allDots) {
        if (assigned.contains(dot)) {
            continue;
        }
        auto temBlockId = bm.getNextId();
        llvm::SmallVector<Operation*> dotSeeds = matchSeed(dot, bm);
        if (willCreateCycle(dotSeeds, memGraph, temBlockId, bm)) {
            LOG_DEBUG("Cube Seed already have a cycle!!");
            return llvm::failure();
        }
        llvm::SmallVector<Operation *> newGroup;
        SeedRegionPlanner regionPlanner {dotSeeds, block, memGraph, assigned, newGroup, bm};
        regionPlanner.run();

        for (auto *op : newGroup) {
            assigned.insert(op);
        }
        if (llvm::failed(bm.markOpsWithNewId(newGroup))) {
          return llvm::failure();
        }
    }

    // Phase 2: Handle remaining Cube Ops following Topo order
    TopologicalPartitionPlanner topoPlanner {block, assigned, memGraph, bm};
    if (failed(topoPlanner.run())) {
        return failure();
    }
    fuseMarkOpToDef(block, bm, memGraph);
    return llvm::success();
}

void mlir::triton::PlanCubeBlockPass::runOnOperation()
{
    LOG_DEBUG("\n--- Step 2: Partitioning compute blocks for cube operations --->\n");
    auto moduleOp = getOperation();
    auto &aa = getAnalysis<AliasAnalysis>();
    auto memGraph = MemoryDependenceGraph(moduleOp, aa);
    auto bm = ComputeBlockIdManager(moduleOp);

    // We do not need to skip linalg blocks since they do not have core types and do not contain matmul
    auto result = moduleOp.walk([&](Block *block) {
      if (llvm::failed(processBlockWithCubeBFS(block, memGraph, bm))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
    LOG_DEBUG("\n--- Step 2: end --->\n");
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createPlanCubeBlockPass()
{
    return std::make_unique<PlanCubeBlockPass>();
}
