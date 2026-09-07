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

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"
#include "ascend/include/DynamicCVPipeline/Common/DependencyHelper.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "DynamicCVPipeline/Common/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <optional>

static constexpr const char *DEBUG_TYPE = "merge-compute-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

/// Represents a ComputeBlock: a group of ops sharing the same block_id.
/// Also carries block-level dependency edges: predecessor/successor blocks
/// and the source ops on incoming cross-block edges.
struct ComputeBlock {
  int id;                            // block_id value
  CVPipeline::CoreType coreType;     // CUBE_ONLY / VECTOR_ONLY
  SmallVector<Operation *> ops;      // all ops in the group (in IR order)
  SmallVector<ComputeBlock *> preds; // predecessor ComputeBlocks
  SmallVector<ComputeBlock *> succs; // successor ComputeBlocks
  /// Source ops of incoming cross-block edges
  DenseMap<ComputeBlock *, SmallVector<Operation *>> inEdgeSrcOps{};
};

static bool hasTensorResult(const ComputeBlock &blk) {
  for (Operation *op : blk.ops) {
    for (Value result : op->getResults()) {
      if (isa<TensorType>(result.getType())) {
        return true;
      }
    }
  }
  return false;
}

/// Step 1: Group ops and build block-level dependency graph.
/// computeBlocks: block_id → ComputeBlock
static void groupAndBuildGraph(
    Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
    DenseMap<int, std::unique_ptr<ComputeBlock>> &computeBlocks) {
  // Walk all ops in the body block and its nested regions, group by
  // block_id
  block->walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return;
    }
    auto optId = CVPipeline::getOpBlockId(op);
    if (!optId.has_value()) {
      return;
    }
    int bid = *optId;
    if (!computeBlocks.contains(bid)) {
      // find new block_id, create new ComputeBlock
      computeBlocks[bid] = std::make_unique<ComputeBlock>(
          ComputeBlock{bid, CVPipeline::getOpCoreType(op), {}, {}});
    }
    computeBlocks[bid]->ops.push_back(op);
  });

  if (computeBlocks.empty()) {
    return;
  }

  // Build dependency graph between ComputeBlocks via DependencyHelper,
  // covering both SSA operands and memory execution dependencies.
  CVPipeline::DependencyHelper helper(memGraph);
  DenseSet<std::pair<int, int>> seenEdges;
  DenseSet<std::tuple<int, int, Operation *>> seenDeps;
  for (auto &kv : computeBlocks) {
    ComputeBlock *curBlk = kv.second.get();
    for (Operation *op : curBlk->ops) {
      helper.forEachSource(op, [&](Operation *src) {
        Operation *ancestor = CVPipeline::getAncestorInBlock(src, block);
        if (!ancestor) {
          return WalkResult::advance(); // not in this block
        }
        auto ancIdOpt = CVPipeline::getOpBlockId(ancestor);
        if (!ancIdOpt.has_value()) {
          return WalkResult::advance();
        }
        auto ancIt = computeBlocks.find(*ancIdOpt);
        if (ancIt == computeBlocks.end()) {
          return WalkResult::advance();
        }
        ComputeBlock *ancBlk = ancIt->second.get();
        if (ancBlk == curBlk) {
          return WalkResult::advance(); // same ComputeBlock internal edge
        }

        // Record the source op that triggers this cross-block edge
        if (seenDeps.insert({*ancIdOpt, curBlk->id, src}).second) {
          curBlk->inEdgeSrcOps[ancBlk].push_back(src);
        }

        if (seenEdges.insert({*ancIdOpt, curBlk->id}).second) {
          ancBlk->succs.push_back(curBlk);
          curBlk->preds.push_back(ancBlk);
        }
        return WalkResult::advance();
      });
    }
  }
}

/// Find the first CUBE predecessor of a given ComputeBlock.
static ComputeBlock *findCubePred(const ComputeBlock &blk) {
  for (ComputeBlock *p : blk.preds) {
    if (p->coreType == CVPipeline::CoreType::CUBE_ONLY) {
      return p;
    }
  }
  return nullptr;
}

/// Collect all transitive dependencies of \p startOp via SSA use-def chains,
/// memory execution predecessors, and ops nested in regions. Results are
/// accumulated into \p collected (including \p startOp itself).
static void collectAllDeps(Operation *startOp,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           SmallPtrSetImpl<Operation *> &collected) {
  collected.insert(startOp);
  CVPipeline::DependencyHelper helper(memGraph);
  SmallVector<Operation *> worklist = {startOp};
  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    helper.forEachSource(cur, [&](Operation *src) {
      if (collected.insert(src).second)
        worklist.push_back(src);
      return WalkResult::advance();
    });
  }
}

/// Check if the memref.copy feeding \p toTensor (the copy writing the buffer
/// consumed by to_tensor) loads its data from global memory (external func
/// argument).
static bool
isToTensorFedByGlobalCopy(bufferization::ToTensorOp toTensor,
                          const CVPipeline::MemoryDependenceGraph &memGraph) {
  bool foundCopy = false;
  for (Operation *def : memGraph.getMemDefs(toTensor.getOperation())) {
    auto copyOp = dyn_cast<memref::CopyOp>(def);
    if (!copyOp) {
      continue;
    }
    foundCopy = true;
    SetVector<Operation *> viewOps;
    if (!CVPipeline::collectViewOpsAndCheckGlobalMemory(copyOp.getSource(),
                                                        viewOps)) {
      return false;
    }
  }
  if (!foundCopy) {
    return false;
  }
  return true;
}

/// Returns the bufferization::ToTensorOp in CubePre that Cube depends on,
/// only to_tensor ops whose feeding memref.copy loads data from GM (external
/// func argument).
static SmallVector<Operation *>
findToTensorDeps(const ComputeBlock *cubePre, ArrayRef<Operation *> edgeSrcOps,
                 const CVPipeline::MemoryDependenceGraph &memGraph) {
  SmallVector<Operation *> result;
  for (Operation *srcOp : edgeSrcOps) {
    auto toTensor = dyn_cast<bufferization::ToTensorOp>(srcOp);
    if (toTensor && llvm::is_contained(cubePre->ops, srcOp) &&
        isToTensorFedByGlobalCopy(toTensor, memGraph)) {
      result.push_back(srcOp);
    }
  }
  return result;
}

/// Records the IR mutations performed by cloneOpCrossCubeDep so they can be
/// rolled back if the subsequent merge attempt fails.
struct CrossCubeCloneRecord {
  SmallVector<Operation *> clonedOps;
  /// (operand, original value) pairs for Cube's original ops whose operands
  /// were remapped to cloned values.
  SmallVector<std::pair<OpOperand *, Value>> remappedOperands;

  void rollback(CVPipeline::ComputeBlockIdManager &bm) {
    for (auto &[operand, origValue] : remappedOperands) {
      operand->set(origValue);
    }
    for (Operation *op : llvm::reverse(clonedOps)) {
      op->walk([&](Operation *innerOp) { bm.eraseOp(innerOp); });
      op->erase();
    }
    clonedOps.clear();
    remappedOperands.clear();
  }
};

/// Clone ops from CubePre into Cube at the front of Cube's ops.
/// Selected ops must be in CubePre and are in `toClone`.
static void cloneOpCrossCubeDep(
    int cubePreId, int cubeId, const SmallPtrSet<Operation *, 16> &toClone,
    const DenseMap<int, std::unique_ptr<ComputeBlock>> &computeBlocks,
    CVPipeline::ComputeBlockIdManager &bm, CrossCubeCloneRecord &record) {
  auto cubePreIt = computeBlocks.find(cubePreId);
  auto cubeIt = computeBlocks.find(cubeId);
  if (cubePreIt == computeBlocks.end() || cubeIt == computeBlocks.end()) {
    return;
  }
  ComputeBlock *cubePreBlock = cubePreIt->second.get();
  ComputeBlock *cubeBlock = cubeIt->second.get();

  if (cubeBlock->ops.empty()) {
    return;
  }

  Operation *insertBefore = cubeBlock->ops.front();
  OpBuilder builder(insertBefore);
  IRMapping mapper;
  for (Operation *op : cubePreBlock->ops) {
    if (!toClone.contains(op)) {
      continue;
    }
    Operation *cloned = builder.clone(*op, mapper);
    record.clonedOps.push_back(cloned);
    cloned->walk(
        [&](Operation *innerOp) { bm.updateBlockId(innerOp, cubeId); });
  }

  // Remap Cube's original ops' operands: replace references to old CubePre
  // values with the corresponding cloned values now in Cube.
  for (Operation *op : cubeBlock->ops) {
    for (auto &operand : op->getOpOperands()) {
      if (Value mapped = mapper.lookupOrNull(operand.get())) {
        record.remappedOperands.emplace_back(&operand, operand.get());
        operand.set(mapped);
      }
    }
  }
}

/// Step 2: Collect merge candidate edges (predV → succV) between
/// VECTOR_ONLY blocks. A candidate block must have tensor results, both a
/// CUBE predecessor and a CUBE successor. Edges are sorted by (predV.id,
/// succV.id) for deterministic processing order.
static SmallVector<std::pair<ComputeBlock *, ComputeBlock *>>
collectVectorMergeEdges(
    const DenseMap<int, std::unique_ptr<ComputeBlock>> &computeBlocks) {
  auto hasCubeNeighbor = [](ArrayRef<ComputeBlock *> neighbors) {
    return llvm::any_of(neighbors, [](const ComputeBlock *nb) {
      return nb->coreType == CVPipeline::CoreType::CUBE_ONLY;
    });
  };

  // Collect VECTOR_ONLY candidate blocks
  SmallVector<ComputeBlock *> vecCandidates;
  for (auto &kv : computeBlocks) {
    ComputeBlock *blk = kv.second.get();
    if (blk->coreType != CVPipeline::CoreType::VECTOR_ONLY)
      continue;
    if (!hasTensorResult(*blk))
      continue;

    // Must have both a CUBE predecessor and a CUBE successor
    if (!hasCubeNeighbor(blk->succs) || !hasCubeNeighbor(blk->preds))
      continue;

    if (!llvm::is_contained(vecCandidates, blk))
      vecCandidates.push_back(blk);
  }

  // Collect ALL adjacent edges between candidates as merge candidates
  SmallVector<std::pair<ComputeBlock *, ComputeBlock *>> edges;
  for (ComputeBlock *cand : vecCandidates) {
    for (ComputeBlock *succ : cand->succs) {
      if (llvm::is_contained(vecCandidates, succ)) {
        edges.emplace_back(cand, succ);
      }
    }
  }
  llvm::sort(edges, [](const std::pair<ComputeBlock *, ComputeBlock *> &a,
                       const std::pair<ComputeBlock *, ComputeBlock *> &b) {
    if (a.first->id != b.first->id) {
      return a.first->id < b.first->id;
    }
    return a.second->id < b.second->id;
  });
  if (edges.empty()) {
    LOG_DEBUG("No adjacent VECTOR edge found, skipping");
  }
  return edges;
}

static void markSubBlock(const ComputeBlock &predV, const ComputeBlock &succV) {
  OpBuilder builder(predV.ops.front());
  for (const ComputeBlock *blk : {&predV, &succV}) {
    for (Operation *op : blk->ops) {
      int curId = CVPipeline::getOpBlockId(op).value_or(blk->id);
      op->setAttr(CVPipeline::kSubBlock, builder.getI32IntegerAttr(curId));
    }
  }
}

/// Step 3: Try to directly merge succV into predV.
/// Returns true if merge succeeded.
static bool tryDirectMerge(const CVPipeline::MemoryDependenceGraph &memGraph,
                           const ComputeBlock &predV, const ComputeBlock &succV,
                           CVPipeline::ComputeBlockIdManager &bm) {
  if (willCreateCycle(succV.ops, memGraph, predV.id, bm))
    return false;
  LOG_DEBUG("Successfully direct merge: " << succV.id << " -> " << predV.id);

  // Mark sub-block for insert sync in splitDataFlow.
  markSubBlock(predV, succV);
  for (Operation *op : succV.ops)
    bm.updateBlockId(op, predV.id);
  return true;
}

/// Collect the ops in CubePre that the given to_tensor ops transitively
/// depend on (via SSA use-def chains, memory execution predecessors, and
/// ops nested in regions). These are the ops that need to be cloned to Cube
/// to break the cycle.
static SmallPtrSet<Operation *, 16>
collectDepOpsInCubePre(const ComputeBlock &cubePre,
                       ArrayRef<Operation *> toTensorOps,
                       const CVPipeline::MemoryDependenceGraph &memGraph) {
  SmallPtrSet<Operation *, 16> opsToClone;
  SmallPtrSet<Operation *, 16> visited;
  for (Operation *toTensor : toTensorOps) {
    collectAllDeps(toTensor, memGraph, visited);
  }
  for (Operation *op : visited) {
    if (llvm::is_contained(cubePre.ops, op))
      opsToClone.insert(op);
  }
  return opsToClone;
}

/// Step 4: Try cross-Cube clone to break the cycle, then merge.
/// Returns true if merge succeeded after clone.
static bool tryCrossCubeCloneMerge(
    Block *block,
    const DenseMap<int, std::unique_ptr<ComputeBlock>> &computeBlocks,
    const ComputeBlock &predV, const ComputeBlock &succV,
    const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  // 4a. Find succV's CUBE predecessor (Cube)
  ComputeBlock *cube = findCubePred(succV);
  if (!cube) {
    LOG_DEBUG("succV " << succV.id << " has no CUBE predecessor");
    return false;
  }

  // 4b. Find Cube's CUBE predecessor (CubePre)
  ComputeBlock *cubePre = findCubePred(*cube);
  if (!cubePre) {
    LOG_DEBUG("Cube " << cube->id << " has no CUBE predecessor");
    return false;
  }

  // 4c. Check if Cube depends on CubePre via to_tensor whose feeding
  // memref.copy loads data from GM (external func argument).
  SmallVector<Operation *> toTensorOps;
  auto edgeIt = cube->inEdgeSrcOps.find(cubePre);
  if (edgeIt != cube->inEdgeSrcOps.end()) {
    toTensorOps = findToTensorDeps(cubePre, edgeIt->second, memGraph);
  }
  if (toTensorOps.empty()) {
    LOG_DEBUG("Cube(" << cube->id << ") depends on CubePre(" << cubePre->id
                      << ") not for load data, skipping");
    return false;
  }

  // 4d. Trace back all transitive dependencies of the to_tensor ops and
  // keep the ones in CubePre; they will be cloned to Cube.
  SmallPtrSet<Operation *, 16> opsToClone =
      collectDepOpsInCubePre(*cubePre, toTensorOps, memGraph);

  CrossCubeCloneRecord cloneRecord;
  cloneOpCrossCubeDep(cubePre->id, cube->id, opsToClone, computeBlocks, bm,
                      cloneRecord);

  // 4e. Re-check cycle after cloning; roll back the clone on failure.
  if (willCreateCycle(succV.ops, memGraph, predV.id, bm)) {
    cloneRecord.rollback(bm);
    LOG_DEBUG("Still creates cycle after cross-Cube clone "
              "for VECTOR "
              << predV.id << " -> " << succV.id);
    return false;
  }

  LOG_DEBUG("Successfully merge after cross-Cube clone: " << succV.id << " -> "
                                                          << predV.id);

  // Mark sub-block for insert sync in splitDataFlow.
  markSubBlock(predV, succV);
  for (Operation *op : succV.ops)
    bm.updateBlockId(op, predV.id);
  return true;
}

static void tryMergeInBlock(Block *mainLoop,
                            CVPipeline::ComputeBlockIdManager &bm,
                            const CVPipeline::MemoryDependenceGraph &memGraph,
                            bool &anyMerged) {
  // Step 1: Group and build dependency graph (block_id → ComputeBlock)
  DenseMap<int, std::unique_ptr<ComputeBlock>> computeBlocks;
  groupAndBuildGraph(mainLoop, memGraph, computeBlocks);
  if (computeBlocks.empty())
    return;

  // Step 2: Collect merge candidate edges between VECTOR blocks
  SmallVector<std::pair<ComputeBlock *, ComputeBlock *>> edgeCandidates =
      collectVectorMergeEdges(computeBlocks);

  while (!edgeCandidates.empty()) {
    // Pop the next VECTOR pair to process from the candidate set.
    ComputeBlock *predV = edgeCandidates.front().first;
    ComputeBlock *succV = edgeCandidates.front().second;
    edgeCandidates.erase(edgeCandidates.begin());

    LOG_DEBUG("Trying VECTOR edge: predV=" << predV->id
                                           << ", succV=" << succV->id);

    // Step 3: Try direct merge; Step 4: try cross-Cube clone merge.
    // Failure of a single edge only skips that edge.
    bool merged = tryDirectMerge(memGraph, *predV, *succV, bm) ||
                  tryCrossCubeCloneMerge(mainLoop, computeBlocks, *predV,
                                         *succV, memGraph, bm);
    if (merged) {
      anyMerged = true;
      // Remove all remaining edges touching the merged blocks: they can
      // no longer participate in further merges.
      llvm::erase_if(edgeCandidates,
                     [&](const std::pair<ComputeBlock *, ComputeBlock *> &e) {
                       return e.first == predV || e.second == predV ||
                              e.first == succV || e.second == succV;
                     });
    }
  }
}

namespace {

class MergeComputeBlockPass
    : public PassWrapper<MergeComputeBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeComputeBlockPass)

  MergeComputeBlockPass() = default;

  StringRef getArgument() const override { return "merge-compute-block"; }

  StringRef getDescription() const override {
    return "Merge adjacent vector compute blocks between/around CUBE blocks";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    auto intraBufCount =
        module->getAttrOfType<IntegerAttr>(CVPipeline::kIntraBufCount);
    auto interCoreBufCount =
        module->getAttrOfType<IntegerAttr>(CVPipeline::kInterCoreBufCount);
    if (!intraBufCount || !interCoreBufCount || intraBufCount.getInt() < 3 ||
        interCoreBufCount.getInt() < 2) {
      LOG_DEBUG("MergeComputeBlock disabled: intraBufCount < 3 or "
                "interCoreBufCount < 2");
      CVPipeline::setSkipExtraReorder(module, true);
      return;
    }

    LOG_DEBUG("Before MergeComputeBlockPass: " << *module);

    SmallVector<Block *> mainLoops;
    llvm::LogicalResult walkResult =
        CVPipeline::SplitIf::walkMainLoop(module, [&](Operation *loop) {
          mainLoops.push_back(&loop->getRegion(0).front());
          return success();
        });
    if (failed(walkResult)) {
      CVPipeline::setSkipExtraReorder(module, true);
      return;
    }

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);
    bool anyMerged = false;
    for (Block *mainLoop : mainLoops) {
      LOG_DEBUG("try merge in LoopBlock");
      tryMergeInBlock(mainLoop, bm, memGraph, anyMerged);
    }

    // Request ReorderOpsByBlockIdPass to skip the extra reorder when this
    // pass ran but merged nothing.
    CVPipeline::setSkipExtraReorder(module, !anyMerged);

    LOG_DEBUG("After MergeComputeBlockPass: " << *module);
  }
};

} // namespace

// ============================================================================
// Pass Registration
// ============================================================================

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeComputeBlockPass() {
  return std::make_unique<MergeComputeBlockPass>();
}

void registerMergeComputeBlockPass() {
  PassRegistration<MergeComputeBlockPass> reg;
}

} // namespace triton
} // namespace mlir
