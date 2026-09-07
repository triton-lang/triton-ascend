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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SYNC_EXEC_EDGES_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SYNC_EXEC_EDGES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Operation.h"

namespace mlir {
namespace CVPipeline {

// Independent store for execution-order edges introduced by synchronization
// ops. Kept separate from MemoryDependenceGraph's execBefore/execAfter so the
// sync-edge builder never touches the memory-dependence tables; the graph
// merges both views in getExecBefore / getExecAfter. This is the decoupling
// seam: buildSyncEdges only fills a SyncExecEdges, and the graph only reads it.
class SyncExecEdges {
public:
  // Record an edge from -> to, deduped per (from, to).
  void addEdge(Operation *from, Operation *to) {
    auto &before = mapBefore[to];
    if (!llvm::is_contained(before, from)) {
      before.push_back(from);
    }
    auto &after = mapAfter[from];
    if (!llvm::is_contained(after, to)) {
      after.push_back(to);
    }
  }

  // Ops that must execute before @p op via sync edges (from -> op).
  ArrayRef<Operation *> getBefore(Operation *op) const {
    auto it = mapBefore.find(op);
    if (it == mapBefore.end()) {
      return {};
    }
    return it->second;
  }

  // Ops that must execute after @p op via sync edges (op -> to).
  ArrayRef<Operation *> getAfter(Operation *op) const {
    auto it = mapAfter.find(op);
    if (it == mapAfter.end()) {
      return {};
    }
    return it->second;
  }

private:
  DenseMap<Operation *, SmallVector<Operation *>> mapBefore; // [to]   = {from}
  DenseMap<Operation *, SmallVector<Operation *>> mapAfter;  // [from] = {to}
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SYNC_EXEC_EDGES_H
