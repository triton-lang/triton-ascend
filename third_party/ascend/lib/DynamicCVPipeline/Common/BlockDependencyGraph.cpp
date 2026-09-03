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
#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "block-dependency-graph";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

BlockDependencyGraph::BlockDependencyGraph(
    Block *block, const MemoryDependenceGraph &memGraph,
    ComputeBlockIdManager &bm)
    : block(block), memGraph(memGraph), bm(bm) {}

llvm::LogicalResult BlockDependencyGraph::buildGraph() {
  // Step 1: Collect all compute blocks (including cube and vector) in current
  // MLIR Block
  for (Operation &op : *block) {
    int blockId = bm.getBlockIdByOp(&op);
    if (blockId == -1)
      continue;

    if (!blockNodes.contains(blockId)) {
      BlockNode node;
      node.blockId = blockId;
      node.isCube = isCubeSimpleOpOrCf(&op);
      node.depth = 0;
      blockNodes[blockId] = node;
    }

    blockNodes[blockId].ops.push_back(&op);
  }

  LDBG("Collected " << blockNodes.size() << " blocks");

  // Step 2: Build dependency relationships between blocks (only in current MLIR
  // Block)
  DependencyHelper depHelper(memGraph);
  for (auto &entry : blockNodes) {
    BlockNode *consumerNode = &entry.second;

    for (Operation *op : consumerNode->ops) {
      // Use DependencyHelper to iterate all source ops (SSA defs + memory
      // exec-before) in one pass.
      depHelper.forEachSource<DependencyHelper::SourceMode::Default>(
          op, [&](Operation *source) {
            if (!isInCurrentBlock(source))
              return;

            int sourceBlockId = bm.getBlockIdByOp(source);
            if (sourceBlockId == -1 || sourceBlockId == consumerNode->blockId)
              return;

            BlockNode *producerNode = getBlockNode(sourceBlockId);
            if (producerNode)
              addEdge(producerNode, consumerNode);
          });
    }
  }

  // Step 3: Compute depths (for all blocks)
  computeDepths();

  return llvm::success();
}

bool BlockDependencyGraph::isInCurrentBlock(Operation *op) {
  if (!op)
    return false;

  // Get the MLIR Block where op is located
  Block *opBlock = op->getBlock();

  // Check if it's in current MLIR Block
  return opBlock == block;
}

void BlockDependencyGraph::addEdge(BlockNode *from, BlockNode *to) {
  predecessors[to].insert(from);
  successors[from].insert(to);
}

void BlockDependencyGraph::computeDepths() {
  // Use topological sort to compute depth for each node
  // Depth definition: maximum steps from nodes with zero in-degree
  // Special rule: depth only increases when cube and vector type conversion
  // occurs

  llvm::DenseMap<BlockNode *, int> indegree;
  std::queue<BlockNode *> queue;

  // Initialize in-degree. Use find() to avoid accidentally inserting
  // zero-sized sets into predecessors/successors for nodes with no edges.
  for (auto &entry : blockNodes) {
    BlockNode *node = &entry.second;
    auto predIt = predecessors.find(node);
    indegree[node] = (predIt != predecessors.end()) ? predIt->second.size() : 0;
    if (indegree[node] == 0) {
      queue.push(node);
      node->depth = 0;
    }
  }

  // BFS to compute depths
  while (!queue.empty()) {
    BlockNode *current = queue.front();
    queue.pop();

    auto succIt = successors.find(current);
    if (succIt == successors.end())
      continue;
    for (BlockNode *succ : succIt->second) {
      // Compute depth from current to succ
      // If types are different (cube->vector or vector->cube), depth+1
      // If types are the same, depth remains unchanged
      int newDepth = current->depth;
      if (current->isCube != succ->isCube) {
        newDepth++;
      }

      // Update successor's depth (take maximum)
      succ->depth = std::max(succ->depth, newDepth);

      indegree[succ]--;
      if (indegree[succ] == 0) {
        queue.push(succ);
      }
    }
  }

  LDBG("Computed depths for all blocks");
}

int BlockDependencyGraph::getDepth(int blockId) {
  auto it = blockNodes.find(blockId);
  if (it != blockNodes.end()) {
    return it->second.depth;
  }
  return -1;
}

BlockNode *BlockDependencyGraph::getBlockNode(int blockId) {
  auto it = blockNodes.find(blockId);
  if (it != blockNodes.end()) {
    return &it->second;
  }
  return nullptr;
}

llvm::SmallVector<BlockNode *>
BlockDependencyGraph::getPredecessors(BlockNode *node) {
  auto it = predecessors.find(node);
  if (it != predecessors.end()) {
    return llvm::SmallVector<BlockNode *>(it->second.begin(), it->second.end());
  }
  return {};
}

llvm::SmallVector<BlockNode *>
BlockDependencyGraph::getSuccessors(BlockNode *node) {
  auto it = successors.find(node);
  if (it != successors.end()) {
    return llvm::SmallVector<BlockNode *>(it->second.begin(), it->second.end());
  }
  return {};
}

void BlockDependencyGraph::recomputeDepthsFrom(BlockNode *start) {
  // Walk forward through the successor chain starting at `start`. For each
  // visited node, the new depth is propagated to its successors; a successor
  // is enqueued again only when its depth actually grows. Since depth is
  // monotonically non-decreasing, this terminates.
  std::queue<BlockNode *> queue;
  queue.push(start);
  while (!queue.empty()) {
    BlockNode *current = queue.front();
    queue.pop();

    auto succIt = successors.find(current);
    if (succIt == successors.end())
      continue;
    for (BlockNode *succ : succIt->second) {
      int newDepth = current->depth;
      if (current->isCube != succ->isCube)
        newDepth++;
      if (succ->depth < newDepth) {
        succ->depth = newDepth;
        queue.push(succ);
      }
    }
  }

  LDBG("Recomputed depths from block " << start->blockId);
}

llvm::LogicalResult BlockDependencyGraph::rebuildAfterMerge(BlockNode *target,
                                                            BlockNode *source) {
  // Merge node information: append source ops into target.
  target->ops.append(source->ops.begin(), source->ops.end());

  // Sanity check: source must be present in both predecessors and
  // successors. A node that is not registered in either map has no graph
  // edges at all; merging such a node is not a supported scenario, so
  // signal failure and let the caller abort the merge pass.
  auto predIt = predecessors.find(source);
  auto succIt = successors.find(source);
  if (predIt == predecessors.end() && succIt == successors.end()) {
    LDBG("Cannot merge block " << source->blockId
                               << ": not found in predecessors or successors\n");
    return llvm::failure();
  }

  // Snapshot the source adjacency. Both insertions into predecessors/
  // successors and the subsequent erase can rehash the underlying maps,
  // so we work off local copies.
  llvm::SmallVector<BlockNode *> sourcePreds;
  if (predIt != predecessors.end()) {
    sourcePreds.assign(predIt->second.begin(), predIt->second.end());
  }
  llvm::SmallVector<BlockNode *> sourceSuccs;
  if (succIt != successors.end()) {
    sourceSuccs.assign(succIt->second.begin(), succIt->second.end());
  }

  // Union the source predecessor/successor sets into target. The pred/succ
  // of the merged-out source naturally become target's neighbours; existing
  // edges (target <-> target) are de-duplicated by DenseSet::insert.
  predecessors[target].insert(sourcePreds.begin(), sourcePreds.end());
  predecessors.erase(source);
  for (BlockNode *pred : sourcePreds) {
    successors[pred].erase(source);
    successors[pred].insert(target);
  }

  successors[target].insert(sourceSuccs.begin(), sourceSuccs.end());
  successors.erase(source);
  for (BlockNode *succ : sourceSuccs) {
    predecessors[succ].erase(source);
    predecessors[succ].insert(target);
  }

  // Drop the source node from blockNodes. This invalidates `source` itself,
  // but no live edge/key still references it at this point.
  int sourceId = source->blockId;
  blockNodes.erase(source->blockId);

  // Only the successor chain reachable from target is affected by the merge;
  // recompute depth locally rather than walking the entire graph.
  recomputeDepthsFrom(target);

  LDBG("Rebuilt graph after merging block " << sourceId << " into "
                                            << target->blockId);
  return llvm::success();
}
