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
  llvm::SmallVector<BlockNode *> getPredecessors(BlockNode *node);
  llvm::SmallVector<BlockNode *> getSuccessors(BlockNode *node);

  // computeDepth
  void computeDepths();
  int getDepth(int blockId);

  // rebuildgraph
  llvm::LogicalResult rebuildAfterMerge(BlockNode *target, BlockNode *source);

  llvm::DenseMap<int, BlockNode> blockNodes;

private:
  Block *block;
  const MemoryDependenceGraph &memGraph;
  ComputeBlockIdManager &bm;

  llvm::DenseMap<BlockNode *, llvm::DenseSet<BlockNode *>> predecessors;
  llvm::DenseMap<BlockNode *, llvm::DenseSet<BlockNode *>> successors;

  void addEdge(BlockNode *from, BlockNode *to);

  bool isInCurrentBlock(Operation *op);

  // Recompute depths starting from `start` and propagating forward through
  // its successor chain. Assumes predecessors of `start` already hold
  // correct depths.
  void recomputeDepthsFrom(BlockNode *start);
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_BLOCK_DEPENDENCY_GRAPH_H
