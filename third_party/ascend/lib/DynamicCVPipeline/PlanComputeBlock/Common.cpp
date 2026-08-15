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

#include "llvm/ADT/iterator.h"

#include "mlir/IR/Value.h"

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

namespace mlir {
namespace CVPipeline {

void DependencyHelper::forEachUser(Operation *op,
                                   DependencyHelper::PredFn pred) const {
  for (auto *user : op->getUsers()) {
    pred(user);
  }
  for (auto *user : memGraph.getExecAfter(op)) {
    pred(user);
  }
}

namespace {
using SourceMode = DependencyHelper::SourceMode;
}

template <SourceMode SM>
void DependencyHelper::forEachSource(Operation *op,
                                     DependencyHelper::PredFn pred) const {
  op->walk([&, this, op](Operation *subOp) {
    for (auto operand : subOp->getOperands()) {
      if (auto *defOp = operand.getDefiningOp(); defOp) {
        if (!op->isAncestor(defOp)) {
          pred(defOp);
        }
        continue;
      }

      if constexpr (SM == SourceMode::AcrossIterArg) {
        if (auto *defOp = getLoopCarriedDefOp(operand, op->getBlock())) {
          pred(defOp);
        }
      }
    }
    for (auto *source : memGraph.getExecBefore(subOp)) {
      if (!op->isAncestor(
              source)) { // this filters only the outer mem dependencies
        pred(source);
      }
    }
  });
}

// Instantiate concrete functions for linking
template void
mlir::CVPipeline::DependencyHelper::forEachSource<SourceMode::Default>(
    mlir::Operation *op,
    llvm::function_ref<void(mlir::Operation *)> callback) const;

template void
mlir::CVPipeline::DependencyHelper::forEachSource<SourceMode::AcrossIterArg>(
    mlir::Operation *op,
    llvm::function_ref<void(mlir::Operation *)> callback) const;

void initializeIndegreeForBlock(Block *block,
                                llvm::DenseMap<Operation *, int> &indegree,
                                const DependencyHelper &depHelper,
                                ComputeBlockIdManager &bm) {
  for (auto *op : llvm::make_pointer_range(block->getOperations())) {
    indegree[op] = 0;
    depHelper.forEachSource(op, [&](Operation *source) {
      if (source->getBlock() == block && !bm.isSameBlock(source, op)) {
        indegree[op]++;
      }
    });
  }
}

Operation *getAncestorInBlock(Operation *inner, Block *block) {
  Operation *cur = inner;
  while (cur) {
    if (cur->getBlock() == block) {
      return cur;
    }
    cur = cur->getParentOp();
  }
  return nullptr;
}

} // namespace CVPipeline
} // namespace mlir
