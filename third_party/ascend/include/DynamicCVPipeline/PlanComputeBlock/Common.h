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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMMON_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMMON_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"

namespace mlir {
namespace CVPipeline {

class DependencyHelper {
  using PredFn = llvm::function_ref<void(Operation *)>;

  template <typename Fn>
  static auto mapToAncestorInBlock(Block *block, Fn &&pred) {
    return [block, pred = std::forward<Fn>(pred)](Operation *op) {
      if (auto *ancestor = block->findAncestorOpInBlock(*op)) {
        return pred(ancestor);
      }
    };
  }

public:
  const MemoryDependenceGraph &memGraph;

  explicit DependencyHelper(const MemoryDependenceGraph &memGraph)
      : memGraph(memGraph) {}

  void forEachUser(Operation *op, PredFn pred) const;

  enum class SourceMode { Default, AcrossIterArg };
  template <SourceMode SM = SourceMode::Default>
  void forEachSource(Operation *op, PredFn pred) const;

  void forEachUserInSameBlock(Operation *op, PredFn pred) const {
    forEachUser(op, mapToAncestorInBlock(op->getBlock(), pred));
  }

  template <SourceMode ST>
  void forEachSourceInSameBlock(Operation *op, PredFn pred) const {
    forEachSource<ST>(op, mapToAncestorInBlock(op->getBlock(), pred));
  }
};

Operation *getAncestorInBlock(Operation *inner, Block *block);
void initializeIndegreeForBlock(Block *block,
                                llvm::DenseMap<Operation *, int> &indegree,
                                const DependencyHelper &depHelper,
                                ComputeBlockIdManager &bm);

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMMON_H
