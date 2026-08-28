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
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "block-dependency-graph";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

static bool isCubeOp(Operation *op) {
  if (!op)
    return false;
  auto coreTypeAttr = op->getAttrOfType<StringAttr>(CVPipeline::kCoreType);
  if (!coreTypeAttr)
    return false;

  return coreTypeAttr.getValue() == "CUBE";
}

BlockDependencyGraph::BlockDependencyGraph(
    Block *block, const MemoryDependenceGraph &memGraph,
    ComputeBlockIdManager &bm)
    : block(block), memGraph(memGraph), bm(bm) {}

llvm::LogicalResult BlockDependencyGraph::buildGraph() {
  // Step 1: Collect all compute blocks (including cube and vector) in current
  // MLIR Block
  block->walk([&](Operation *op) {
    int blockId = bm.getBlockIdByOp(op);
    if (blockId == -1)
      return;

    // Only process operations in current MLIR Block
    if (op->getBlock() != block)
      return;

    if (!blockNodes.count(blockId)) {
      BlockNode node;
      node.blockId = blockId;
      node.isCube = isCubeOp(op);
      node.depth = 0;
      blockNodes[blockId] = node;
    }

    blockNodes[blockId].ops.push_back(op);
  });

  LDBG("Collected " << blockNodes.size() << " blocks");

  // Step 2: Build dependency relationships between blocks (only in current MLIR
  // Block)
  for (auto &entry : blockNodes) {
    int blockId = entry.first;
    auto &node = entry.second;

    for (Operation *op : node.ops) {
      // Process SSA data flow dependencies
      for (Value operand : op->getOperands()) {
        if (Operation *def = operand.getDefiningOp()) {
          // Key: Check if def is in current MLIR Block
          if (!isInCurrentBlock(def))
            continue;

          int defBlockId = bm.getBlockIdByOp(def);
          if (defBlockId != -1 && defBlockId != blockId) {
            addEdge(defBlockId, blockId);
          }
        }
      }

      // Process memory dependencies
      for (Operation *memDef : memGraph.getMemDefs(op)) {
        // Key: Check if memDef is in current MLIR Block
        if (!isInCurrentBlock(memDef))
          continue;

        int defBlockId = bm.getBlockIdByOp(memDef);
        if (defBlockId != -1 && defBlockId != blockId) {
          addEdge(defBlockId, blockId);
        }
      }
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

void BlockDependencyGraph::addEdge(int from, int to) {
  // Avoid duplicate additions
  if (!llvm::is_contained(predecessors[to], from)) {
    predecessors[to].push_back(from);
  }
  if (!llvm::is_contained(successors[from], to)) {
    successors[from].push_back(to);
  }
}

void BlockDependencyGraph::computeDepths() {
  // Use topological sort to compute depth for each node
  // Depth definition: maximum steps from nodes with zero in-degree
  // Special rule: depth only increases when cube and vector type conversion
  // occurs

  llvm::DenseMap<int, int> indegree;
  std::queue<int> queue;

  // Initialize in-degree
  for (auto &entry : blockNodes) {
    indegree[entry.first] = predecessors[entry.first].size();
    if (indegree[entry.first] == 0) {
      queue.push(entry.first);
      blockNodes[entry.first].depth = 0;
    }
  }

  // BFS to compute depths
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();

    for (int succ : successors[current]) {
      // Compute depth from current to succ
      // If types are different (cube->vector or vector->cube), depth+1
      // If types are the same, depth remains unchanged
      int newDepth = blockNodes[current].depth;
      if (blockNodes[current].isCube != blockNodes[succ].isCube) {
        newDepth++;
      }

      // Update successor's depth (take maximum)
      blockNodes[succ].depth = std::max(blockNodes[succ].depth, newDepth);

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

llvm::SmallVector<int> BlockDependencyGraph::getPredecessors(int blockId) {
  auto it = predecessors.find(blockId);
  if (it != predecessors.end()) {
    return it->second;
  }
  return {};
}

llvm::SmallVector<int> BlockDependencyGraph::getSuccessors(int blockId) {
  auto it = successors.find(blockId);
  if (it != successors.end()) {
    return it->second;
  }
  return {};
}

bool BlockDependencyGraph::wouldCreateCycle(int blockId1, int blockId2) {
  // Use DFS to check reachability
  llvm::DenseSet<int> visited;

  std::function<bool(int, int)> hasPath = [&](int from, int to) -> bool {
    if (from == to)
      return true;
    if (visited.count(from))
      return false;
    visited.insert(from);

    for (int succ : successors[from]) {
      if (hasPath(succ, to))
        return true;
    }
    return false;
  };

  // If there are bidirectional paths, merging would create a cycle
  bool path1to2 = hasPath(blockId1, blockId2);
  visited.clear();
  bool path2to1 = hasPath(blockId2, blockId1);

  return path1to2 && path2to1;
}

void BlockDependencyGraph::rebuildAfterMerge(int targetBlockId,
                                             int sourceBlockId) {
  // Merge node information of two blocks
  auto sourceIt = blockNodes.find(sourceBlockId);
  if (sourceIt != blockNodes.end()) {
    auto &targetNode = blockNodes[targetBlockId];
    auto &sourceNode = sourceIt->second;

    // Merge ops
    targetNode.ops.append(sourceNode.ops.begin(), sourceNode.ops.end());

    // Delete source node
    blockNodes.erase(sourceIt);
  }

  // Merge predecessors and successors
  auto sourcePredIt = predecessors.find(sourceBlockId);
  if (sourcePredIt != predecessors.end()) {
    for (int pred : sourcePredIt->second) {
      if (!llvm::is_contained(predecessors[targetBlockId], pred)) {
        predecessors[targetBlockId].push_back(pred);
      }
      // Update successors
      auto &predSuccs = successors[pred];
      predSuccs.erase(llvm::find(predSuccs, sourceBlockId));
      if (!llvm::is_contained(predSuccs, targetBlockId)) {
        predSuccs.push_back(targetBlockId);
      }
    }
    predecessors.erase(sourcePredIt);
  }

  auto sourceSuccIt = successors.find(sourceBlockId);
  if (sourceSuccIt != successors.end()) {
    for (int succ : sourceSuccIt->second) {
      if (!llvm::is_contained(successors[targetBlockId], succ)) {
        successors[targetBlockId].push_back(succ);
      }
      // Update predecessors
      auto &succPreds = predecessors[succ];
      succPreds.erase(llvm::find(succPreds, sourceBlockId));
      if (!llvm::is_contained(succPreds, targetBlockId)) {
        succPreds.push_back(targetBlockId);
      }
    }
    successors.erase(sourceSuccIt);
  }

  // Recompute depths
  computeDepths();

  LDBG("Rebuilt graph after merging block " << sourceBlockId << " into "
                                            << targetBlockId);
}
