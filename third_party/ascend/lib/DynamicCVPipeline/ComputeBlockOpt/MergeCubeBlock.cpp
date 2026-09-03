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
#include "DynamicCVPipeline/Common/BlockDependencyGraph.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/MergeCubeBlockPass.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "merge-cube-block";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

static const llvm::DenseSet<llvm::StringRef> kDisableMergeCubeKernel = {
    "flex_attention_backward_dq_kernel",
    "parallel_deltaformer_bwd_kernel_qk",
    "_parallel_hstu_attn_bwd",
};

void MergeCubeBlockPass::runOnOperation() {
  auto moduleOp = getOperation();

  if (hasFallbackAttr(moduleOp)) {
    return;
  }

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (kDisableMergeCubeKernel.contains(funcOp.getSymName())) {
      LDBG("Found unsupport kernel: " << funcOp.getSymName());
      return;
    }
  }

  auto &aa = getAnalysis<AliasAnalysis>();
  auto memGraph = MemoryDependenceGraph(moduleOp, aa);
  auto bm = ComputeBlockIdManager(moduleOp);

  LDBG("Starting MergeCubeBlockPass\n" << moduleOp);

  // Collect main loop body blocks via the shared walkMainLoop helper.
  llvm::SmallVector<Block *> mainLoopBlocks;
  if (failed(CVPipeline::SplitIf::walkMainLoop(
          moduleOp, [&](Operation *loop) -> llvm::LogicalResult {
            mainLoopBlocks.push_back(&loop->getRegion(0).front());
            return llvm::success();
          }))) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LDBG("Found " << mainLoopBlocks.size() << " main loop blocks\n");

  // Process each main loop block
  for (Block *block : mainLoopBlocks) {
    if (failed(processBlock(block, memGraph, bm))) {
      CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
      return;
    }
  }

  LDBG("MergeCubeBlockPass completed\n" << moduleOp);
}

llvm::LogicalResult
MergeCubeBlockPass::processBlock(Block *block,
                                 const MemoryDependenceGraph &memGraph,
                                 ComputeBlockIdManager &bm) {
  LDBG("Processing Block: " << *block);
  if (Operation *parentOp = block->getParentOp()) {
    LDBG("  Parent operation: " << parentOp->getName().getStringRef());
  }

  // Step 1: Build Block dependency graph
  BlockDependencyGraph graph(block, memGraph, bm);
  if (failed(graph.buildGraph())) {
    LDBG("Failed to build dependency graph");
    return llvm::failure();
  }

  LDBG("Built dependency graph with " << graph.blockNodes.size()
                                      << " blocks\n");

  // Step 2: Find merge candidates
  llvm::SmallVector<std::pair<BlockNode *, BlockNode *>> candidates;
  if (failed(findMergeCandidates(graph, candidates))) {
    LDBG("Failed to find merge candidates");
    return llvm::failure();
  }
  LDBG("Found " << candidates.size() << " merge candidates\n");

  // If merge candidates are found, print graph structure
  if (!candidates.empty()) {
    LDBG("=== Printing dependency graph for current region ===");
    LLVM_DEBUG(printGraph(graph));
  }

  // Step 3: Perform iterative merging
  if (failed(performMerging(graph, memGraph, bm, candidates))) {
    LDBG("Failed to perform merging");
    return llvm::failure();
  }

  return llvm::success();
}

llvm::LogicalResult MergeCubeBlockPass::performMerging(
    BlockDependencyGraph &graph, const MemoryDependenceGraph &memGraph,
    ComputeBlockIdManager &bm,
    llvm::SmallVectorImpl<std::pair<BlockNode *, BlockNode *>> &candidates) {

  bool merged = true;
  int mergeCount = 0;

  while (merged) {
    merged = false;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (canMergeBlocks(candidates[i], graph, memGraph, bm)) {
        BlockNode *target = candidates[i].first;
        BlockNode *source = candidates[i].second;
        LDBG("Merging block " << source->blockId << " into "
                              << target->blockId);

        // Execute merge
        if (failed(mergeBlocks(target, source, bm))) {
          LDBG("Failed to merge blocks");
          return llvm::failure();
        }
        merged = true;
        mergeCount++;

        // Update graph structure. After this call, the BlockNode pointed to
        // by `source` has been destroyed; the pointer value itself remains
        // usable for equality checks, but any dereference is unsafe. The
        // pointers for `target` and unrelated nodes stay valid because
        // blockNodes.erase() does not rehash or relocate other entries.
        //
        // Note: mergeBlocks above has already re-tagged source's ops to
        // target via the ComputeBlockIdManager, which is not transactional.
        // A rebuildAfterMerge failure therefore leaves the graph and the
        // blockIdManager in an inconsistent state, so we abort the whole
        // merging pass and surface the error to processBlock.
        if (failed(graph.rebuildAfterMerge(target, source))) {
          LDBG("Failed to rebuild graph after merging "
               << source->blockId << " into " << target->blockId);
          return llvm::failure();
        }

        // Incrementally update candidates in place:
        //   * drop every pair that still references the merged-out `source`
        //   * keep pairs that touch `target` (its neighbours may have
        //     changed, but the next iteration will re-evaluate them via
        //     canMergeBlocks).
        // This avoids the O(N^2) full rebuild that was previously done on
        // every merge.
        llvm::SmallVector<std::pair<BlockNode *, BlockNode *>, 16> kept;
        kept.reserve(candidates.size());
        for (auto &p : candidates) {
          if (p.first == source || p.second == source)
            continue;
          kept.push_back(p);
        }
        candidates = std::move(kept);
        break;
      }
    }
  }

  LDBG("Merged " << mergeCount << " blocks\n");

  return llvm::success();
}

llvm::LogicalResult MergeCubeBlockPass::findMergeCandidates(
    BlockDependencyGraph &graph,
    llvm::SmallVectorImpl<std::pair<BlockNode *, BlockNode *>> &candidates) {

  // Only get cube blocks (filtered from complete dependency graph)
  llvm::SmallVector<BlockNode *> cubeBlocks;
  for (auto &entry : graph.blockNodes) {
    if (entry.second.isCube) {
      cubeBlocks.push_back(&entry.second);
    }
  }

  // Check all pairs of cube blocks for merge possibility
  for (size_t i = 0; i < cubeBlocks.size(); ++i) {
    for (size_t j = i + 1; j < cubeBlocks.size(); ++j) {
      candidates.push_back({cubeBlocks[i], cubeBlocks[j]});
    }
  }

  return llvm::success();
}

bool MergeCubeBlockPass::canMergeBlocks(
    std::pair<BlockNode *, BlockNode *> pair, BlockDependencyGraph &graph,
    const MemoryDependenceGraph &memGraph, ComputeBlockIdManager &bm) {
  BlockNode *node1 = pair.first;
  BlockNode *node2 = pair.second;

  // Precondition: both blocks must be cube
  if (!node1 || !node2 || !node1->isCube || !node2->isCube) {
    return false;
  }

  // Step 1: Check if they have common input or output nodes
  if (!hasCommonInputOrOutput(node1, node2, graph)) {
    LDBG("Blocks " << node1->blockId << " and " << node2->blockId
                   << " cannot merge: no common input or output nodes\n");
    return false;
  }

  // Step 2: Check if merging would create a cycle
  if (!checkNoCycle(node1, node2, graph, memGraph, bm)) {
    LDBG("Blocks " << node1->blockId << " and " << node2->blockId
                   << " cannot merge: would create cycle\n");
    return false;
  }

  // Step 3: cube blocks at the same depth (with matching successor depths).
  if (hasSameDepth(node1, node2, graph)) {
    LDBG("Blocks " << node1->blockId << " and " << node2->blockId
                   << " can merge: cube blocks have same depth\n");
    return true;
  }

  LDBG("Blocks " << node1->blockId << " and " << node2->blockId
                 << " cannot merge: unsupport scenario\n");
  return false;
}

bool MergeCubeBlockPass::hasSameDepth(BlockNode *node1, BlockNode *node2,
                                      BlockDependencyGraph &graph) {

  if (!node1 || !node2) {
    return false;
  }

  if (node1->depth != node2->depth) {
    return false;
  }

  // Check that the max depth among successors of both blocks is the same
  auto getMaxSuccDepth = [&](BlockNode *node) {
    int maxDepth = -1;
    for (BlockNode *succNode : graph.getSuccessors(node)) {
      if (succNode && succNode->depth > maxDepth) {
        maxDepth = succNode->depth;
      }
    }
    return maxDepth;
  };

  int maxSuccDepth1 = getMaxSuccDepth(node1);
  int maxSuccDepth2 = getMaxSuccDepth(node2);

  // If either block has no successors, still mergeable
  if (maxSuccDepth1 == -1 || maxSuccDepth2 == -1) {
    return true;
  }

  return maxSuccDepth1 == maxSuccDepth2;
}

llvm::SmallVector<BlockNode *> MergeCubeBlockPass::filterBlocksByType(
    BlockNode *currentNode, llvm::SmallVector<BlockNode *> blocks) {

  if (!currentNode) {
    return {};
  }

  // Filter blocks: only keep blocks with different type from current block
  llvm::SmallVector<BlockNode *> filteredBlocks;
  for (BlockNode *blockNode : blocks) {
    if (blockNode && blockNode->isCube != currentNode->isCube) {
      filteredBlocks.push_back(blockNode);
    }
  }

  return filteredBlocks;
}

bool MergeCubeBlockPass::checkNoCycle(BlockNode *node1, BlockNode *node2,
                                      BlockDependencyGraph &graph,
                                      const MemoryDependenceGraph &memGraph,
                                      ComputeBlockIdManager &bm) {

  // Get all operations from node2 to be merged into node1
  auto opsToUnify = bm.getOpsByBlockId(node2->blockId);

  // Use willCreateCycle to check if merging would create a cycle
  return !willCreateCycle(opsToUnify, memGraph, node1->blockId, bm);
}

bool MergeCubeBlockPass::hasCommonInputOrOutput(BlockNode *node1,
                                                BlockNode *node2,
                                                BlockDependencyGraph &graph) {

  // Get predecessors and successors
  auto preds1 = graph.getPredecessors(node1);
  auto preds2 = graph.getPredecessors(node2);

  auto succs1 = graph.getSuccessors(node1);
  auto succs2 = graph.getSuccessors(node2);

  // Filter blocks: only keep blocks with different type from current block
  llvm::SmallVector<BlockNode *> filteredPreds1 =
      filterBlocksByType(node1, preds1);
  llvm::SmallVector<BlockNode *> filteredPreds2 =
      filterBlocksByType(node2, preds2);
  llvm::SmallVector<BlockNode *> filteredSuccs1 =
      filterBlocksByType(node1, succs1);
  llvm::SmallVector<BlockNode *> filteredSuccs2 =
      filterBlocksByType(node2, succs2);

  // Check if there are common input nodes (predecessors)
  llvm::DenseSet<BlockNode *> predSet1(filteredPreds1.begin(),
                                       filteredPreds1.end());
  bool hasCommonPred =
      llvm::any_of(filteredPreds2,
                   [&](BlockNode *p) { return predSet1.count(p); });

  // Check if there are common output nodes (successors)
  llvm::DenseSet<BlockNode *> succSet1(filteredSuccs1.begin(),
                                       filteredSuccs1.end());
  bool hasCommonSucc =
      llvm::any_of(filteredSuccs2,
                   [&](BlockNode *s) { return succSet1.count(s); });

  // Return true if they have common input OR output nodes
  return hasCommonPred || hasCommonSucc;
}

llvm::LogicalResult MergeCubeBlockPass::mergeBlocks(BlockNode *target,
                                                    BlockNode *source,
                                                    ComputeBlockIdManager &bm) {

  // Merge all ops of source into target
  auto sourceOps = bm.getOpsByBlockId(source->blockId);
  for (Operation *op : sourceOps) {
    bm.updateBlockId(op, target->blockId);
  }

  return llvm::success();
}

void MergeCubeBlockPass::printGraph(BlockDependencyGraph &graph) {

  LDBG("Total blocks in graph: " << graph.blockNodes.size());

  // Print all block nodes
  for (auto &entry : graph.blockNodes) {
    int blockId = entry.first;
    BlockNode *node = &entry.second;

    LDBG("Block " << blockId << ":");
    LDBG("  Type: " << (node->isCube ? "CUBE" : "VECTOR"));
    LDBG("  Depth: " << node->depth);
    LDBG("  Num ops: " << node->ops.size());

    // Print predecessors
    auto preds = graph.getPredecessors(node);
    if (!preds.empty()) {
      llvm::SmallVector<std::string> predStrs;
      for (BlockNode *p : preds) {
        predStrs.push_back(std::to_string(p->blockId));
      }
      LDBG("  Predecessors: [" << llvm::join(predStrs, ", ") << "]");
    } else {
      LDBG("  Predecessors: []");
    }

    // Print successors
    auto succs = graph.getSuccessors(node);
    if (!succs.empty()) {
      llvm::SmallVector<std::string> succStrs;
      for (BlockNode *s : succs) {
        succStrs.push_back(std::to_string(s->blockId));
      }
      LDBG("  Successors: [" << llvm::join(succStrs, ", ") << "]");
    } else {
      LDBG("  Successors: []");
    }
  }

  LDBG("=== End of graph ===");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createMergeCubeBlockPass() {
  return std::make_unique<MergeCubeBlockPass>();
}