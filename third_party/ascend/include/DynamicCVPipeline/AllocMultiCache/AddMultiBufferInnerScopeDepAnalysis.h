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

#ifndef TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_INNER_SCOPE_DEP_ANALYSIS_H
#define TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_INNER_SCOPE_DEP_ANALYSIS_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

namespace mlir {
namespace triton {

// Per-block dependency metadata built during the Phase-1 prep pass. Each
// entry is keyed by the block_id-typed Value that represents the block, and
// carries the ops that live in that block.
struct InnerBlockInfo {
  Value blockId;
  SmallVector<Operation *> ops;
};

// Return the outermost ssbuffer.id visible on `op`'s ancestor chain. -1 is
// returned when an ancestor carries kMainLoop and the innermost id seen on
// the walk is empty. Used by both Phase 1 (dep analysis) and Phase 2
// (cross-block consumer detection), so kept inline here.
inline std::optional<int64_t> getOutermostSsbufferId(Operation *op) {
  std::optional<int64_t> result;
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (current->hasAttr(CVPipeline::kMainLoop))
      return result.has_value() ? result : -1;

    if (current->getNumRegions() >= 2)
      return CVPipeline::getOpBlockId(current);

    // Otherwise remember the deepest id seen; the parent walk will
    // overwrite it if a closer-to-boundary op carries one.
    if (auto curId = CVPipeline::getOpBlockId(current); curId.has_value())
      result = curId;
  }
  return result;
}

// Recursively collect every Operation in `block` and its nested regions into
// `ops`, in pre-order. Shared between Phase 1 helpers (collectInnerBlockInfo,
// rematerializeTensorRootedScalarDeps) and the main file's
// findNestedMainloop / collectMainLoopsRecursively.
inline void collectNestedOps(Block *block, SmallVector<Operation *> &ops) {
  for (auto &op : *block) {
    ops.push_back(&op);
    for (auto &region : op.getRegions()) {
      for (auto &innerBlock : region) {
        collectNestedOps(&innerBlock, ops);
      }
    }
  }
}

// Walk the main loop body, group ops by their ssbuffer.block_id, build the
// per-block InnerBlockInfo map, and collect cross-block tensor deps into
// `depValueMap`. `i1Found` is set when any collected tensor dep has element
// type i1; the caller is expected to abort and trigger fallback.
//
// Run once per phase: Phase 1 feeds the empty/fill clone + scalar
// rematerialize; Phase 2 picks up the new cross-block refs the clone
// introduced. Returns 0 on success, -1 when an invalid (negative) block id
// surfaces from an upstream pass.
int collectInnerBlockInfo(const CVPipeline::MainLoop &loop,
                          DenseMap<Value, InnerBlockInfo> &blocks,
                          DenseMap<Value, SmallVector<Value>> &depValueMap,
                          SmallVector<Operation *> &allOps, bool &i1Found);

// Build the dependency-user map by scanning block ops and the yield operands
// of any multi-region op (scf.if, scf.while, ...) whose depVal is not a direct
// operand. Called once per phase: initial map feeds the Phase-1 empty/fill
// clone; the rebuilt map feeds Phase-2's processTensorDependencies.
DenseMap<Value, SmallVector<Operation *>>
buildDepUserMap(DenseMap<Value, InnerBlockInfo> &blocks,
                SmallVector<Operation *> &allOps,
                DenseMap<Value, SmallVector<Value>> &depValueMap);

// Clone each bufferization.alloc_tensor dep value into its consumer block so
// the cloned tensor is local to the consumer and the normal multi-buffer
// pipeline can wrap it. Called from Phase-2 orchestration in
// AddMultiBufferInnerScope.cpp.
int cloneAllocTensorsInBlocks(
    const CVPipeline::MainLoop &loop, DenseMap<Value, InnerBlockInfo> &blocks,
    DenseMap<Value, SmallVector<Value>> &depValueMap,
    DenseMap<Value, SmallVector<Operation *>> &depUserMap,
    OpBuilder &globalBuilder);

// Phase 1 driver: run the initial dep collection, surface memref / i1
// fallbacks, build the initial dep-user map, clone the empty+fill pattern
// into consumer blocks, and rematerialize scalar chains rooted in a tensor.
// On success populates `blocks` / `depValueMap` / `allOps` /
// `phase1ClonedDepVals` for Phase 2 to consume (Phase 2 re-clears and
// re-collects these).
//
// Returns 0 on success, -1 on memref fallback or any failure. `i1Found` is
// set whenever an i1 tensor dep surfaces; mirroring the original driver,
// callers check `i1Found` after Phase 2's second dep collection rather than
// here.
int runDepAnalysisAndClone(CVPipeline::MainLoop &mainLoop,
                           OpBuilder &globalBuilder, bool &i1Found,
                           DenseMap<Value, InnerBlockInfo> &blocks,
                           DenseMap<Value, SmallVector<Value>> &depValueMap,
                           SmallVector<Operation *> &allOps,
                           DenseSet<Value> &phase1ClonedDepVals);

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_ALLOC_MULTI_CACHE_ADD_MULTI_BUFFER_INNER_SCOPE_DEP_ANALYSIS_H
