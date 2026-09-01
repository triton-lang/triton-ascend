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

static const llvm::DenseSet<llvm::StringRef> kDisnbleMergeCubeKernel = {
    "flex_attention_backward_dq_kernel",
    "parallel_deltaformer_bwd_kernel_qk",
    "_parallel_hstu_attn_bwd",
};

void MergeCubeBlockPass::findInnermostLoopBlocksWithMatmul(
    ModuleOp moduleOp, llvm::SmallVectorImpl<Block *> &innermostBlocks) {

  // Find all matmul operations
  llvm::SmallVector<linalg::MatmulOp> matmulOps;
  moduleOp.walk([&](linalg::MatmulOp matmulOp) {
    matmulOps.push_back(matmulOp);
    return WalkResult::advance();
  });

  if (matmulOps.empty()) {
    LDBG("No matmul operations found");
    return;
  }

  // For each matmul, find its innermost loop and calculate depth
  llvm::DenseMap<Block *, int> blockDepthMap;

  for (auto matmulOp : matmulOps) {
    // Walk up the parent chain to find all loops
    llvm::SmallVector<Operation *> loops;
    Operation *currentOp = matmulOp;

    while (currentOp) {
      if (auto forOp = dyn_cast<scf::ForOp>(currentOp)) {
        loops.push_back(forOp);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(currentOp)) {
        loops.push_back(whileOp);
      }
      currentOp = currentOp->getParentOp();
    }

    // If this matmul is inside loops, record the innermost loop's block
    if (!loops.empty()) {
      Operation *innermostLoop =
          loops.front(); // The first one is the innermost
      int depth = loops.size();

      // Get the block from the innermost loop
      Block *block = nullptr;
      if (auto forOp = dyn_cast<scf::ForOp>(innermostLoop)) {
        block = forOp.getBody();
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(innermostLoop)) {
        // WhileOp has a region, get the first block
        block = whileOp.getAfterBody();
      }

      if (block) {
        // Update depth map (keep the maximum depth for each block)
        if (blockDepthMap.find(block) == blockDepthMap.end() ||
            depth > blockDepthMap[block]) {
          blockDepthMap[block] = depth;
        }

        LDBG("Matmul at " << matmulOp << " is in loop at depth " << depth);
      }
    }
  }

  // Find the maximum depth
  int maxDepth = 0;
  for (auto &entry : blockDepthMap) {
    if (entry.second > maxDepth) {
      maxDepth = entry.second;
    }
  }

  LDBG("Maximum loop depth: " << maxDepth);

  // Collect all blocks with maximum depth
  for (auto &entry : blockDepthMap) {
    if (entry.second == maxDepth) {
      LDBG("Found innermost loop block with matmul");
      innermostBlocks.push_back(entry.first);
    }
  }
}

void MergeCubeBlockPass::runOnOperation() {
  auto moduleOp = getOperation();

  if (hasFallbackAttr(moduleOp)) {
    return;
  }

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (kDisnbleMergeCubeKernel.contains(funcOp.getSymName())) {
      LDBG("Found unsupport kernel: " << funcOp.getSymName());
      return;
    }
  }

  auto &aa = getAnalysis<AliasAnalysis>();
  auto memGraph = MemoryDependenceGraph(moduleOp, aa);
  auto bm = ComputeBlockIdManager(moduleOp);

  LDBG("Starting MergeCubeBlockPass\n" << moduleOp);

  // Find innermost loop blocks with matmul operations
  llvm::SmallVector<Block *> innermostBlocks;
  findInnermostLoopBlocksWithMatmul(moduleOp, innermostBlocks);

  LDBG("Found " << innermostBlocks.size()
                << " innermost loop blocks with matmul\n");

  // Process each innermost loop block
  for (Block *block : innermostBlocks) {
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

  // Print block information before processing
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

  llvm::DenseMap<int, llvm::DenseSet<Value>> blockLoadedValues;

  // Step 2: Find merge candidates
  llvm::SmallVector<std::pair<int, int>> candidates;
  if (failed(findMergeCandidates(graph, blockLoadedValues, candidates))) {
    LDBG("Failed to find merge candidates");
    return llvm::failure();
  }
  LDBG("Found " << candidates.size() << " merge candidates\n");

  // If merge candidates are found, print graph structure
  if (!candidates.empty()) {
    LDBG("=== Printing dependency graph for current region ===");
    printGraph(graph, blockLoadedValues);
  }

  // Step 3: Perform iterative merging
  if (failed(
          performMerging(graph, memGraph, bm, blockLoadedValues, candidates))) {
    LDBG("Failed to perform merging");
    return llvm::failure();
  }

  return llvm::success();
}

llvm::LogicalResult MergeCubeBlockPass::performMerging(
    BlockDependencyGraph &graph, const MemoryDependenceGraph &memGraph,
    ComputeBlockIdManager &bm,
    llvm::DenseMap<int, llvm::DenseSet<Value>> &blockLoadedValues,
    llvm::SmallVectorImpl<std::pair<int, int>> &candidates) {

  bool merged = true;
  int mergeCount = 0;

  while (merged) {
    merged = false;
    for (auto &pair : candidates) {
      if (canMergeBlocks(pair.first, pair.second, graph, memGraph, bm,
                         blockLoadedValues)) {
        LDBG("Merging block " << pair.second << " into " << pair.first);

        // Execute merge
        if (failed(mergeBlocks(pair.first, pair.second, bm))) {
          LDBG("Failed to merge blocks");
          return llvm::failure();
        }
        merged = true;
        mergeCount++;

        // Update graph structure
        graph.rebuildAfterMerge(pair.first, pair.second);

        // Re-find candidates
        candidates.clear();
        if (failed(findMergeCandidates(graph, blockLoadedValues, candidates))) {
          LDBG("Failed to re-find merge candidates");
          return llvm::failure();
        }
        break;
      }
    }
  }

  LDBG("Merged " << mergeCount << " blocks\n");

  return llvm::success();
}

llvm::LogicalResult MergeCubeBlockPass::findMergeCandidates(
    BlockDependencyGraph &graph,
    llvm::DenseMap<int, llvm::DenseSet<Value>> &blockLoadedValues,
    llvm::SmallVectorImpl<std::pair<int, int>> &candidates) {

  // Only get cube blocks (filtered from complete dependency graph)
  llvm::SmallVector<int> cubeBlocks;
  for (auto &entry : graph.blockNodes) {
    if (entry.second.isCube) {
      cubeBlocks.push_back(entry.first);
    }
  }

  // Check all pairs of cube blocks for merge possibility
  for (size_t i = 0; i < cubeBlocks.size(); ++i) {
    for (size_t j = i + 1; j < cubeBlocks.size(); ++j) {
      int blockId1 = cubeBlocks[i];
      int blockId2 = cubeBlocks[j];

      candidates.push_back({blockId1, blockId2});
    }
  }

  return llvm::success();
}

bool MergeCubeBlockPass::canMergeBlocks(
    int blockId1, int blockId2, BlockDependencyGraph &graph,
    const MemoryDependenceGraph &memGraph, ComputeBlockIdManager &bm,
    llvm::DenseMap<int, llvm::DenseSet<Value>> &blockLoadedValues) {

  // Precondition: both blocks must be cube
  BlockNode *node1 = graph.getBlockNode(blockId1);
  BlockNode *node2 = graph.getBlockNode(blockId2);

  if (!node1 || !node2 || !node1->isCube || !node2->isCube) {
    return false;
  }

  // Step 1: Check if they have common input or output nodes
  if (!hasCommonInputOrOutput(blockId1, blockId2, graph)) {
    LDBG("Blocks " << blockId1 << " and " << blockId2
                   << " cannot merge: no common input or output nodes\n");
    return false;
  }

  // Step 2: Check if they have same source and sink with no other nodes
  if (!checkSameSourceAndSink(blockId1, blockId2, graph)) {
    LDBG("Blocks " << blockId1 << " and " << blockId2
                   << " cannot merge: different source or sink\n");
    return false;
  }

  // Step 3: Check if merging would create a cycle
  if (checkNoCycle(blockId1, blockId2, graph, memGraph, bm)) {
    LDBG("Blocks " << blockId1 << " and " << blockId2
                   << " can merge: no cycle detected\n");
    return true;
  }

  LDBG("Blocks " << blockId1 << " and " << blockId2
                 << " cannot merge: would create cycle\n");
  return false;
}

bool MergeCubeBlockPass::checkSameSourceAndSink(int blockId1, int blockId2,
                                                BlockDependencyGraph &graph) {

  // Get block nodes
  BlockNode *node1 = graph.getBlockNode(blockId1);
  BlockNode *node2 = graph.getBlockNode(blockId2);

  if (!node1 || !node2) {
    return false;
  }

  // Get predecessors and successors
  auto preds1 = graph.getPredecessors(blockId1);
  auto preds2 = graph.getPredecessors(blockId2);

  auto succs1 = graph.getSuccessors(blockId1);
  auto succs2 = graph.getSuccessors(blockId2);

  // Filter blocks: only keep blocks with different type from current block
  llvm::SmallVector<int> filteredPreds1 =
      filterBlocksByType(blockId1, preds1, graph);
  llvm::SmallVector<int> filteredPreds2 =
      filterBlocksByType(blockId2, preds2, graph);
  llvm::SmallVector<int> filteredSuccs1 =
      filterBlocksByType(blockId1, succs1, graph);
  llvm::SmallVector<int> filteredSuccs2 =
      filterBlocksByType(blockId2, succs2, graph);

  // Check if filtered predecessors are exactly the same
  llvm::DenseSet<int> predSet1(filteredPreds1.begin(), filteredPreds1.end());
  llvm::DenseSet<int> predSet2(filteredPreds2.begin(), filteredPreds2.end());

  if (predSet1 != predSet2) {
    return false;
  }

  // Check if filtered successors are exactly the same
  llvm::DenseSet<int> succSet1(filteredSuccs1.begin(), filteredSuccs1.end());
  llvm::DenseSet<int> succSet2(filteredSuccs2.begin(), filteredSuccs2.end());

  if (succSet1 != succSet2) {
    return false;
  }

  return true;
}

llvm::SmallVector<int> MergeCubeBlockPass::filterBlocksByType(
    int blockId, llvm::SmallVector<int> blocks, BlockDependencyGraph &graph) {

  // Get current block node
  BlockNode *currentNode = graph.getBlockNode(blockId);
  if (!currentNode) {
    return {};
  }

  // Filter blocks: only keep blocks with different type from current block
  llvm::SmallVector<int> filteredBlocks;
  for (int blockId : blocks) {
    BlockNode *blockNode = graph.getBlockNode(blockId);
    if (blockNode && blockNode->isCube != currentNode->isCube) {
      filteredBlocks.push_back(blockId);
    }
  }

  return filteredBlocks;
}

bool MergeCubeBlockPass::checkNoCycle(int blockId1, int blockId2,
                                      BlockDependencyGraph &graph,
                                      const MemoryDependenceGraph &memGraph,
                                      ComputeBlockIdManager &bm) {

  // Get all operations from blockId2 to be merged into blockId1
  auto opsToUnify = bm.getOpsByBlockId(blockId2);

  // Use willCreateCycle to check if merging would create a cycle
  return !willCreateCycle(opsToUnify, memGraph, blockId1, bm);
}

bool MergeCubeBlockPass::hasCommonInputOrOutput(int blockId1, int blockId2,
                                                BlockDependencyGraph &graph) {

  // Get predecessors and successors
  auto preds1 = graph.getPredecessors(blockId1);
  auto preds2 = graph.getPredecessors(blockId2);

  auto succs1 = graph.getSuccessors(blockId1);
  auto succs2 = graph.getSuccessors(blockId2);

  // Filter blocks: only keep blocks with different type from current block
  llvm::SmallVector<int> filteredPreds1 =
      filterBlocksByType(blockId1, preds1, graph);
  llvm::SmallVector<int> filteredPreds2 =
      filterBlocksByType(blockId2, preds2, graph);
  llvm::SmallVector<int> filteredSuccs1 =
      filterBlocksByType(blockId1, succs1, graph);
  llvm::SmallVector<int> filteredSuccs2 =
      filterBlocksByType(blockId2, succs2, graph);

  // Check if there are common input nodes (predecessors)
  llvm::DenseSet<int> predSet1(filteredPreds1.begin(), filteredPreds1.end());
  bool hasCommonPred =
      llvm::any_of(filteredPreds2, [&](int p) { return predSet1.count(p); });

  // Check if there are common output nodes (successors)
  llvm::DenseSet<int> succSet1(filteredSuccs1.begin(), filteredSuccs1.end());
  bool hasCommonSucc =
      llvm::any_of(filteredSuccs2, [&](int s) { return succSet1.count(s); });

  // Return true if they have common input OR output nodes
  return hasCommonPred || hasCommonSucc;
}

llvm::LogicalResult MergeCubeBlockPass::mergeBlocks(int targetBlockId,
                                                    int sourceBlockId,
                                                    ComputeBlockIdManager &bm) {

  // Merge all ops of sourceBlockId into targetBlockId
  auto sourceOps = bm.getOpsByBlockId(sourceBlockId);
  for (Operation *op : sourceOps) {
    bm.updateBlockId(op, targetBlockId);
  }

  return llvm::success();
}

void MergeCubeBlockPass::printGraph(
    BlockDependencyGraph &graph,
    llvm::DenseMap<int, llvm::DenseSet<Value>> &blockLoadedValues) {

  LDBG("Total blocks in graph: " << graph.blockNodes.size());

  // Print all block nodes
  for (auto &entry : graph.blockNodes) {
    int blockId = entry.first;
    auto &node = entry.second;

    LDBG("Block " << blockId << ":");
    LDBG("  Type: " << (node.isCube ? "CUBE" : "VECTOR"));
    LDBG("  Depth: " << node.depth);
    LDBG("  Num ops: " << node.ops.size());

    // Print predecessors
    auto preds = graph.getPredecessors(blockId);
    if (!preds.empty()) {
      llvm::SmallVector<std::string> predStrs;
      for (int p : preds) {
        predStrs.push_back(std::to_string(p));
      }
      LDBG("  Predecessors: [" << llvm::join(predStrs, ", ") << "]");
    } else {
      LDBG("  Predecessors: []");
    }

    // Print successors
    auto succs = graph.getSuccessors(blockId);
    if (!succs.empty()) {
      llvm::SmallVector<std::string> succStrs;
      for (int s : succs) {
        succStrs.push_back(std::to_string(s));
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
