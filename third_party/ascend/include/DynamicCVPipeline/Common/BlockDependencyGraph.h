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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_BLOCK_DEPENDENCY_GRAPH_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_BLOCK_DEPENDENCY_GRAPH_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace CVPipeline {

// Forward declarations
class MemoryDependenceGraph;
class ComputeBlockIdManager;

struct BlockNode {
  int blockId;
  llvm::SmallVector<Operation *> ops;
  bool isCube;
  int depth;
};

class BlockDependencyGraph {
public:
  BlockDependencyGraph(Block *block, const MemoryDependenceGraph &memGraph,
                       ComputeBlockIdManager &bm);

  // Build block-level dependency graph (only includes dependencies within
  // current MLIR Block)
  llvm::LogicalResult buildGraph();

  BlockNode *getBlockNode(int blockId);
  llvm::SmallVector<int> getPredecessors(int blockId);
  llvm::SmallVector<int> getSuccessors(int blockId);

  // computeDepth
  void computeDepths();
  int getDepth(int blockId);

  // detect cycle
  bool wouldCreateCycle(int blockId1, int blockId2);

  // rebuildgraph
  void rebuildAfterMerge(int targetBlockId, int sourceBlockId);

  llvm::DenseMap<int, BlockNode> blockNodes;

private:
  Block *block;
  const MemoryDependenceGraph &memGraph;
  ComputeBlockIdManager &bm;

  llvm::DenseMap<int, llvm::SmallVector<int>> predecessors;
  llvm::DenseMap<int, llvm::SmallVector<int>> successors;

  void addEdge(int from, int to);

  bool isInCurrentBlock(Operation *op);
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_BLOCK_DEPENDENCY_GRAPH_H
