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

/**
 * Block节点信息（通用定义）
 * 注意：包含所有类型的block（cube和vector），用于构建完整的依赖图
 */
struct BlockNode {
  int blockId;
  llvm::SmallVector<Operation *> ops; // 该block包含的所有op

  // 辅助信息
  bool isCube; // 标识是否为cube block（用于合并时筛选）
  int depth;   // 在依赖图中的深度

  // 特殊情况说明：
  // ops可能包含以下类型的操作：
  // 1. 普通计算操作（linalg.matmul, linalg.fill等）
  // 2. 特殊结构操作（scf.if, scf.for等），这些操作可能被单独分配blockid
  //    例如：一个scf.if操作本身被标记为cube或vector，作为一个独立的block
};

/**
 * Block级依赖图
 *
 * 作用域：针对单个MLIR Block（基本块）构建依赖图
 *
 * 重要说明：
 * 1. 只处理同一个MLIR Block内的依赖关系，不处理跨Block的依赖
 * 2. 不同MLIR
 * Block可能有相同的block_id，因此BlockDependencyGraph实例不能跨Block使用
 * 3. 在构建依赖边时，会检查操作是否在同一个MLIR Block内
 */
class BlockDependencyGraph {
public:
  BlockDependencyGraph(Block *block, const MemoryDependenceGraph &memGraph,
                       ComputeBlockIdManager &bm);

  // Build block-level dependency graph (only includes dependencies within
  // current MLIR Block)
  llvm::LogicalResult buildGraph();

  // 查询接口
  BlockNode *getBlockNode(int blockId);
  llvm::SmallVector<int> getPredecessors(int blockId);
  llvm::SmallVector<int> getSuccessors(int blockId);

  // 深度计算
  void computeDepths();
  int getDepth(int blockId);

  // 环检测
  bool wouldCreateCycle(int blockId1, int blockId2);

  // 重建图（合并后）
  void rebuildAfterMerge(int targetBlockId, int sourceBlockId);

  // 公共数据成员（供外部访问）
  llvm::DenseMap<int, BlockNode> blockNodes;

private:
  Block *block; // 当前MLIR Block（作用域）
  const MemoryDependenceGraph &memGraph;
  ComputeBlockIdManager &bm;

  // Block级图结构（包含所有block）
  llvm::DenseMap<int, llvm::SmallVector<int>> predecessors;
  llvm::DenseMap<int, llvm::SmallVector<int>> successors;

  // 辅助方法
  void addEdge(int from, int to); // 简化：只需要建立依赖关系

  // 检查操作是否在当前MLIR Block内
  bool isInCurrentBlock(Operation *op);
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_BLOCK_DEPENDENCY_GRAPH_H
