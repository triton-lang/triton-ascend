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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
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
            if (levels.at(&op) != lvl)
                continue;
            auto classIt = classification.find(&op);
            if (classIt == classification.end())
                continue;
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
// Stage 7: Reorder around VECTOR -> CUBE producer boundaries
// ============================================================================
// A breadth-first dependency-level order starts every independent unrolled
// chain before completing any one of them.  For attention this leaves four
// full score tiles and their softmax temporaries live at once, fragments the
// vector functions, and can overflow UB.  The useful cross-core schedule is
// instead organized around values that VECTOR must produce before CUBE can
// continue (for example, each softmax P tile consumed by a PV matmul).
//
// Complete the dependency slice of each such producer in original program
// order, then append the remaining operations.  The input block is already a
// valid topological order, so filtering that order by a dependency-closed
// slice preserves SSA order.  After scope separation this yields the intended
// phase order without matching a particular kernel:
//
//   CUBE:   producer work for all boundaries, then consuming work
//   VECTOR: complete boundary 0, boundary 1, ..., then remaining work
//
// If a loop has no VECTOR -> CUBE boundary, retain the existing level order.
static void collectPredecessorSlice(Operation *op,
                                    const DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
                                    DenseSet<Operation *> &slice)
{
    if (!slice.insert(op).second)
        return;
    auto it = predecessors.find(op);
    if (it == predecessors.end())
        return;
    for (Operation *pred : it->second)
        collectPredecessorSlice(pred, predecessors, slice);
}

// A boundary tensor can have state-maintenance users that are not predecessors
// of the value sent to CUBE.  Softmax is the important example: exp(score) is
// both truncated into P (the boundary value) and reduced into the running
// denominator.  Leaving that rank-1 reduction branch for the later PV phase
// keeps the full score tile live across all unrolled stages.  Close a producer
// slice over ready scalar/rank-1 VECTOR operations so this state is completed
// with the boundary, while deliberately excluding rank-2 work such as the PV
// accumulator update.
static void extendWithReadyVectorState(
    Block *body, const DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
    const Classification &classification, const DenseSet<Operation *> &alreadyScheduled,
    DenseSet<Operation *> &slice)
{
    bool changed = true;
    while (changed) {
        changed = false;
        for (Operation &op : *body) {
            if (slice.contains(&op) || isa<scf::YieldOp>(&op))
                continue;
            auto classIt = classification.find(&op);
            if (classIt == classification.end() || classIt->second != EngineType::VECTOR || op.getNumResults() == 0)
                continue;

            bool isScalarOrRankOne = llvm::all_of(op.getResultTypes(), [](Type type) {
                if (auto shaped = dyn_cast<ShapedType>(type))
                    return shaped.hasRank() && shaped.getRank() <= 1;
                return true;
            });
            if (!isScalarOrRankOne)
                continue;

            auto predIt = predecessors.find(&op);
            if (predIt == predecessors.end())
                continue;
            bool touchesSlice = false;
            for (Operation *pred : predIt->second)
                touchesSlice |= slice.contains(pred);
            if (!touchesSlice)
                continue;

            // Pull in private scalar/rank-1 initializers as well.  For example,
            // a linalg.reduce commonly has a fresh rank-1 linalg.fill init that
            // is not yet in the boundary slice.  Reject the candidate if its
            // newly required dependency closure contains any other rank-2
            // value; that would absorb the next tile or PV accumulator phase.
            DenseSet<Operation *> candidateSlice;
            collectPredecessorSlice(&op, predecessors, candidateSlice);
            bool safe = true;
            for (Operation *candidate : candidateSlice) {
                if (slice.contains(candidate) || alreadyScheduled.contains(candidate))
                    continue;
                auto candidateClassIt = classification.find(candidate);
                if (candidateClassIt == classification.end() || candidateClassIt->second != EngineType::VECTOR ||
                    !llvm::all_of(candidate->getResultTypes(), [](Type type) {
                        if (auto shaped = dyn_cast<ShapedType>(type))
                            return shaped.hasRank() && shaped.getRank() <= 1;
                        return true;
                    })) {
                    safe = false;
                    break;
                }
            }
            if (!safe)
                continue;
            for (Operation *candidate : candidateSlice)
                slice.insert(candidate);
            changed = true;
        }
    }
}

static SmallVector<Operation *> collectVectorToCubeProducers(Block *body, const Classification &classification)
{
    SmallVector<Operation *> producers;
    for (Operation &op : *body) {
        auto classIt = classification.find(&op);
        if (classIt == classification.end() || classIt->second != EngineType::VECTOR)
            continue;

        bool feedsCubeMatmul = false;
        for (Value result : op.getResults()) {
            for (Operation *user : result.getUsers()) {
                if (user->getBlock() != body || !isa<linalg::MatmulOp>(user))
                    continue;
                auto userClassIt = classification.find(user);
                if (userClassIt == classification.end() || userClassIt->second != EngineType::CUBE)
                    continue;
                if (user->getOperand(0) == result || user->getOperand(1) == result) {
                    feedsCubeMatmul = true;
                    break;
                }
            }
            if (feedsCubeMatmul)
                break;
        }
        if (feedsCubeMatmul)
            producers.push_back(&op);
    }
    return producers;
}

static void reorderForCrossScopeProducerPhases(
    Block *body, const DenseMap<Operation *, SmallVector<Operation *>> &predecessors,
    const DenseMap<Operation *, int> &levels, const Classification &classification,
    DenseMap<Operation *, Operation *> &transferPhaseEnds)
{
    SmallVector<Operation *> originalOrder;
    for (Operation &op : *body) {
        if (!isa<scf::YieldOp>(&op))
            originalOrder.push_back(&op);
    }

    SmallVector<Operation *> boundaryProducers = collectVectorToCubeProducers(body, classification);
    SmallVector<Operation *> scheduled;
    DenseSet<Operation *> alreadyScheduled;

    if (boundaryProducers.empty()) {
        scheduled = originalOrder;
        llvm::stable_sort(scheduled, [&](Operation *a, Operation *b) { return levels.at(a) < levels.at(b); });
    } else {
        for (Operation *producer : boundaryProducers) {
            DenseSet<Operation *> slice;
            collectPredecessorSlice(producer, predecessors, slice);
            extendWithReadyVectorState(body, predecessors, classification, alreadyScheduled, slice);

            // Keep the original topological order of the complete producer
            // phase. The actual cross-core handoff is inserted after the last
            // operation in this phase, so independent recurrent state can be
            // updated in the same vector function after the P value is formed.
            Operation *phaseEnd = producer;
            for (Operation *op : originalOrder) {
                if (!slice.contains(op))
                    continue;
                if (alreadyScheduled.insert(op).second) {
                    scheduled.push_back(op);
                    phaseEnd = op;
                }
            }
            transferPhaseEnds[producer] = phaseEnd;
        }
        for (Operation *op : originalOrder)
            if (alreadyScheduled.insert(op).second)
                scheduled.push_back(op);
    }

    Operation *yield = body->getTerminator();
    for (Operation *op : scheduled)
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
//   4. complete each VECTOR -> CUBE producer slice before starting unrelated
//      work, then split into a CUBE scope and a VECTOR scope.
// run() fails when dependency levels cannot be assigned to every op.
// ============================================================================
LogicalResult DependencyScheduler::run(Block *body, const Classification &classification,
                                       DenseMap<Operation *, Operation *> &transferPhaseEnds)
{
    DenseMap<Operation *, SmallVector<Operation *>> predecessors;
    DenseMap<Operation *, int> levels;

    buildDependencyGraph(body, predecessors);

    SmallVector<Operation *> roots = collectRoots(body, predecessors);
    LLVM_DEBUG(llvm::dbgs() << "[cv-split] " << roots.size() << " roots\n");

    int maxLevel = assignDependencyLevels(body, predecessors, roots, levels);
    if (failed(verifyDependencyLevels(body, levels)))
        return failure();
    LLVM_DEBUG(llvm::dbgs() << "[cv-split] " << (maxLevel + 1) << " dependency levels\n");

    logLevelHistogram(body, levels, classification, maxLevel);

    reorderForCrossScopeProducerPhases(body, predecessors, levels, classification, transferPhaseEnds);
    LLVM_DEBUG(llvm::dbgs() << "[cv-split] Reordered by cross-scope producer phase\n");
    return success();
}

} // namespace mlir::triton::cv_split
