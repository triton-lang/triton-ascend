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

#include "ascend/include/CVSplitScheduling/DependencyScheduler.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

// ============================================================================
// Stage 4: Dependency graph
// ============================================================================
static void buildDependencyGraph(Block *body, DenseMap<Operation *, SmallVector<Operation *>> &predecessors)
{
    for (Operation &op : *body) {
        if (isa<scf::YieldOp>(&op))
            continue;
        for (Value operand : op.getOperands()) {
            auto *defOp = operand.getDefiningOp();
            if (!defOp || defOp->getBlock() != body || isa<scf::YieldOp>(defOp))
                continue;
            predecessors[&op].push_back(defOp);
        }
    }

    // Add memory dependency edges: memref.copy → bufferization.to_tensor
    // memref.copy writes to an alloc via side effect (no SSA result).
    // to_tensor reads from the same alloc. Without this edge, dependency leveling
    // can place to_tensor BEFORE copy, causing reads of uninitialized data.
    for (Operation &op : *body) {
        auto copyOp = dyn_cast<memref::CopyOp>(&op);
        if (!copyOp)
            continue;
        Value dst = copyOp.getTarget();
        for (Operation *user : dst.getUsers()) {
            if (user == &op || user->getBlock() != body)
                continue;
            if (isa<bufferization::ToTensorOp>(user)) {
                predecessors[user].push_back(&op);
            }
        }
    }

    // Similarly: memref.copy also depends on reinterpret_cast of its SOURCE.
    // The source reinterpret_cast depends on address arithmetic that computes
    // the GM offset. Ensure copy is ordered after its source address is ready.
    // (This is already captured by SSA edges since copy USES the source value.)
}

static LogicalResult validateDependencyGraph(const DenseMap<Operation *, SmallVector<Operation *>> &predecessors)
{
    for (const auto &entry : predecessors) {
        if (entry.second.empty()) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Invalid empty predecessor entry for " << entry.first->getName()
                                    << ", bail\n");
            return failure();
        }
    }
    return success();
}

// ============================================================================
// Stage 5: Dependency level assignment
// ============================================================================
static SmallVector<Operation *> collectRoots(Block *body,
                                             const DenseMap<Operation *, SmallVector<Operation *>> &predecessors)
{
    SmallVector<Operation *> roots;
    for (Operation &op : *body) {
        if (isa<scf::YieldOp>(&op))
            continue;
        auto it = predecessors.find(&op);
        if (it == predecessors.end()) {
            roots.push_back(&op);
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Root: " << op.getName() << "\n");
        }
    }
    return roots;
}

static int assignDependencyLevels(Block *body, const DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
                                  const SmallVector<Operation *> &roots, DenseMap<Operation *, int> &levels)
{
    for (auto *root : roots)
        levels[root] = 0;

    int maxLevel = 0;

    bool changed = true;
    while (changed) {
        changed = false;
        for (Operation &op : *body) {
            if (isa<scf::YieldOp>(&op))
                continue;

            auto it = predecessors.find(&op);
            if (it == predecessors.end())
                continue;

            int requiredLevel = 0;
            bool allPredecessorsReady = true;
            for (auto *pred : it->second) {
                auto predIt = levels.find(pred);
                if (predIt == levels.end()) {
                    allPredecessorsReady = false;
                    break;
                }
                requiredLevel = std::max(requiredLevel, predIt->second + 1);
            }
            if (!allPredecessorsReady)
                continue;

            auto lvlIt = levels.find(&op);
            if (lvlIt == levels.end()) {
                levels[&op] = requiredLevel;
                maxLevel = std::max(maxLevel, requiredLevel);
                changed = true;
            }
        }
    }

    return maxLevel;
}

static LogicalResult verifyDependencyLevels(Block *body, const DenseMap<Operation *, int> &levels)
{
    for (Operation &op : *body) {
        if (isa<scf::YieldOp>(&op))
            continue;
        if (!levels.count(&op)) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] No dependency level assigned to " << op.getName()
                                    << "; dependency graph may contain a cycle, bail\n");
            return failure();
        }
    }
    return success();
}

// ============================================================================
// Stage 6: Level diagnostics
// ============================================================================
static void logLevelHistogram(Block *body, const DenseMap<Operation *, int> &levels,
                              const DenseMap<Operation *, EngineType> &classification, int maxLevel)
{
    for (int lvl = 0; lvl <= maxLevel; ++lvl) {
        SmallVector<Operation *> cubeOps, vectorOps;
        for (Operation &op : *body) {
            if (isa<scf::YieldOp>(&op))
                continue;
            auto lvlIt = levels.find(&op);
            if (lvlIt->second != lvl)
                continue;
            auto classIt = classification.find(&op);
            if (classIt->second == EngineType::CUBE)
                cubeOps.push_back(&op);
            else
                vectorOps.push_back(&op);
        }

        LLVM_DEBUG(llvm::dbgs() << "[cv-split]   L" << lvl << ": " << cubeOps.size() << "C " << vectorOps.size()
                                << "V\n");
    }
}

// ============================================================================
// Stage 7: Reorder by dependency level
// ============================================================================
static void reorderByLevel(Block *body, const DenseMap<Operation *, int> &levels)
{
    SmallVector<Operation *> ops;
    for (Operation &op : *body) {
        if (!isa<scf::YieldOp>(&op))
            ops.push_back(&op);
    }

    llvm::stable_sort(ops, [&](Operation *a, Operation *b) { return levels.lookup(a) < levels.lookup(b); });

    Operation *yield = body->getTerminator();
    for (auto *op : ops)
        op->moveBefore(yield);
}

// ============================================================================
// Dependency-level scheduler
// ----------------------------------------------------------------------------
// Orchestrates stages 4-7 over an (already unrolled) loop body:
//   1. build a def->use dependency graph (SSA edges + memref.copy->to_tensor
//      memory edges),
//   2. assign every op a level = longest dependency depth from a root,
//   3. report the per-level CUBE/VECTOR distribution,
//   4. reorder the body by level, ready to be
//      split into a CUBE scope and a VECTOR scope.
// run() fails when dependency levels cannot be assigned to every op.
// ============================================================================
LogicalResult DependencyScheduler::run(Block *body, const Classification &classification)
{
    DenseMap<Operation *, SmallVector<Operation *>> predecessors;
    DenseMap<Operation *, int> levels;

    buildDependencyGraph(body, predecessors);
    if (failed(validateDependencyGraph(predecessors)))
        return failure();

    SmallVector<Operation *> roots = collectRoots(body, predecessors);
    LLVM_DEBUG(llvm::dbgs() << "[cv-split] " << roots.size() << " roots\n");

    int maxLevel = assignDependencyLevels(body, predecessors, roots, levels);
    if (failed(verifyDependencyLevels(body, levels)))
        return failure();
    LLVM_DEBUG(llvm::dbgs() << "[cv-split] " << (maxLevel + 1) << " dependency levels\n");

    logLevelHistogram(body, levels, classification, maxLevel);

    reorderByLevel(body, levels);
    LLVM_DEBUG(llvm::dbgs() << "[cv-split] Reordered by level\n");
    return success();
}

} // namespace mlir::triton::cv_split
