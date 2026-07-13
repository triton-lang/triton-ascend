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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CloneOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

static constexpr const char *DEBUG_TYPE = "CloneOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::outs() << __VA_ARGS__; \
  llvm::outs() << "\n"; \
})

using namespace mlir;
using namespace triton;
using namespace hivm;

using MemDepGraph = std::unique_ptr<CVPipeline::MemoryDependenceGraph>;
using MemDepGraphT = CVPipeline::MemoryDependenceGraph;

// Update op operands using value mapping, skip yield values of forOp
static LogicalResult updateCloneMapping(Operation *op,
    llvm::DenseMap<Value, Value> &valueMap,
    const llvm::DenseSet<Value> &yieldValues)
{
  if (!op) {
    return failure();
  }

  for (OpOperand &operand : op->getOpOperands()) {
    // Only skip if this operand is a yield value from the main_loop forOp
    // Nested yield ops (in scf.if/scf.for) should have their operands updated
    Value v = operand.get();
    if (yieldValues.contains(v)) {
      continue;
    }

    auto it = valueMap.find(v);
    if (it != valueMap.end()) {
      if (it->second.getType() != v.getType()) {
        LDBG("[Error]: type mismatch in value mapping: " << v.getType() << " vs " << it->second.getType() << "\n");
        return failure();
      }
      operand.set(it->second);
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(updateCloneMapping(&nestedOp, valueMap, yieldValues))) {
          return failure();
        }
      }
    }
  }

  return success();
}

// Clone a single op with IRMapping
static Operation *cloneOpWithMapping(Operation *op,
    OpBuilder &builder,
    llvm::DenseMap<Value, Value> &valueMap)
{
  IRMapping mapper;
  // Populate mapper with ALL previously cloned values (not just the current op's results). 
  for (const auto &entry : valueMap) {
    mapper.map(entry.first, entry.second);
  }

  Operation *cloned = builder.clone(*op, mapper);
  for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
    valueMap[std::get<0>(it)] = std::get<1>(it);
  }

  return cloned;
}

// Clone ops for a single block in vector/cube mode
static LogicalResult cloneOpsForBlock(int curId, SmallVector<Operation *> &curOps,
    const SmallVector<int> &earlierIds,
    const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    scf::ForOp forOp)
{
  if (curOps.empty() || earlierIds.empty()) {
    return success();
  }

  // Collect all ops from earlier blocks to clone
  SmallVector<Operation *> toClone;
  for (int eid : earlierIds) {
    llvm::append_range(toClone, blockOps.lookup(eid));
  }
  if (toClone.empty()) {
    return success();
  }

  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<Operation *> clonedOps;
  OpBuilder builder(curOps.front());

  for (Operation *op : toClone) {
    Operation *cloned = cloneOpWithMapping(op, builder, valueMap);
    cloned->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(curId));
    if (auto origBlockIdOpt = CVPipeline::getOpBlockId(op)) {
      cloned->setAttr(CVPipeline::kClone, builder.getI32IntegerAttr(static_cast<int32_t>(*origBlockIdOpt)));
    }
    clonedOps.push_back(cloned);
  }

  curOps.insert(curOps.begin(), clonedOps.begin(), clonedOps.end());

  llvm::DenseSet<Value> yieldValues;
  if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
    for (Value operand : yieldOp.getOperands()) {
      yieldValues.insert(operand);
    }
  }

  for (Operation *op : curOps) {
    if (failed(updateCloneMapping(op, valueMap, yieldValues))) {
      return failure();
    }
  }

  if (failed(topologicalSort(curOps))) {
    return failure();
  }

  return success();
}

// Clone ops for all blocks in main loop
LogicalResult CloneOpsPass::cloneOpsInMainLoop(scf::ForOp forOp)
{
  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(forOp, blockOps))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);

  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    int curId = idsInOrder[i];
    SmallVector<int> earlierIds(idsInOrder.begin(), idsInOrder.begin() + i);

    if (failed(cloneOpsForBlock(curId, blockOps[curId], earlierIds, blockOps, forOp))) {
      return failure();
    }
  }

  return success();
}

// Check if an op should be erased during cleanup (for cube)
// sameBlockIdExecAfter: precomputed map from cloned op to its same-block-id exec-after ops
//   (already filtered to skip SyncBlockWaitOp/SyncBlockSetOp), built once per block from a
//   single MemoryDependenceGraph.
// erasedOps: set of cloned ops already erased in this cleanup pass; an exec-after entry
//   that has been erased no longer pins the current op (equivalent to the original
//   per-op graph rebuild reflecting the post-erasure IR).
static bool shouldEraseOpForCube(Operation *op,
    const llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>> &sameBlockIdExecAfter,
    const llvm::DenseSet<Operation *> &erasedOps)
{
  // Rule 1: SyncBlockWaitOp, SyncBlockSetOp, FixpipeOp -> directly erase
  if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
      isa<hivm::FixpipeOp>(op)) {
    return true;
  }

  auto opBlockId = getForDirectChildBlockId(op);

  // Rule 2: If op has results, check via SSA if result is used by later ops in same block_id
  // Use for-direct-child block_id (the immediate child of scf.for) for comparison
  if (op->getNumResults() > 0) {
    for (auto result : op->getResults()) {
      if (result.use_empty()) {
        // Result not used by anyone, can erase
        continue;
      }
      if (!opBlockId) {
        // No block_id but result is used, be conservative and keep
        return false;
      }
      // Check if any user is in the same for-direct-child block_id
      bool usedInSameBlockId = llvm::any_of(result.getUsers(), [&](Operation *user) {
        auto userBlockId = getForDirectChildBlockId(user);
        return userBlockId && *userBlockId == *opBlockId;
      });
      if (usedInSameBlockId) {
        // Result used in same for-direct-child block, cannot erase
        return false;
      }
    }
    // All results are either unused or not used in same for-direct-child block, can erase
    return true;
  }

  // Rule 3: If op has no results, consult the precomputed same-block-id exec-after set.
  // An exec-after op that has already been erased in this pass is no longer live and
  // does not require keeping this op. This mirrors the original per-iteration
  // memGraph rebuild, where already-erased ops simply disappear from getExecAfter().
  auto it = sameBlockIdExecAfter.find(op);
  if (it != sameBlockIdExecAfter.end()) {
    for (Operation *execOp : it->second) {
      // Already erased in this pass → no longer pins the current op.
      if (erasedOps.contains(execOp)) {
        continue;
      }
      // scf.if whose body only contains sync_block_wait/sync_block_set ops will be cleaned up
      // in its own turn; skip it here so it does not block the current op's erasure.
      if (isIfOpWithOnlySyncOps(execOp)) {
        continue;
      }
      auto execBlockId = CVPipeline::getOpBlockId(execOp);
      if (execBlockId && opBlockId && *execBlockId == *opBlockId) {
        return false;
      }
    }
  }

  // Can erase if no results and no live exec-after dependencies in same block
  return true;
}

// Check if an op should be erased (for vector)
static bool shouldEraseOpForVector(Operation *op)
{
  return llvm::none_of(op->getResults(),
                       [](auto result) { return !result.use_empty(); });
}

// Cleanup for cloned ops in a forOp
// memGraphFactory: callable that rebuilds MemoryDependenceGraph for current IR state
static LogicalResult cleanupClonedOps(scf::ForOp forOp,
    llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    const SmallVector<int> &idsInOrder,
    bool isCube,
    std::function<MemDepGraph(scf::ForOp)> memGraphFactory)
{
  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    auto &curOps = blockOps[idsInOrder[i]];
    if (curOps.empty()) {
      continue;
    }

    // Find last index of cloned ops
    int startIdx = -1;
    for (int j = curOps.size() - 1; j >= 0; --j) {
      if (curOps[j]->hasAttr(CVPipeline::kClone)) {
        startIdx = j;
        break;
      }
    }
    if (startIdx < 0) {
      continue;
    }

    // Locate the start of the contiguous cloned-op suffix. The original cleanup
    // loop broke on the first non-cloned op; preserve that behavior so we only
    // consider ops that the previous implementation would have considered,
    // regardless of how topologicalSort interleaves cloned and non-cloned ops.
    int firstClonedIdx = 0;
    for (int j = startIdx - 1; j >= 0; --j) {
      if (!curOps[j]->hasAttr(CVPipeline::kClone)) {
        firstClonedIdx = j + 1;
        break;
      }
    }

    // Collect the cloned ops in the contiguous suffix (in execution order).
    SmallVector<Operation *> clonedOps;
    clonedOps.reserve(startIdx - firstClonedIdx + 1);
    for (int j = firstClonedIdx; j <= startIdx; ++j) {
      clonedOps.push_back(curOps[j]);
    }

    // Build the MemoryDependenceGraph at most once per block, and only if at
    // least one cloned op has no results (Rule 3 is the only path that needs it;
    // Rule 2 is a pure SSA check and Rule 1 is type-based).
    std::unique_ptr<MemDepGraphT> memGraph;
    if (isCube) {
      bool needsMemGraph = llvm::any_of(clonedOps, [](Operation *o) {
        return o->getNumResults() == 0;
      });
      if (needsMemGraph) {
        memGraph = memGraphFactory(forOp);
      }
    }

    // Precompute, for each cloned op, the set of its exec-after ops that share
    // its block_id. Building the memGraph was the expensive part; we now do
    // getExecAfter once per op up front and reuse the result during cleanup.
    // An erased op is filtered out at check time (see shouldEraseOpForCube),
    // which reproduces the original "rebuild after each erasure" effect.
    llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>> sameBlockIdExecAfter;
    if (memGraph) {
      for (Operation *op : clonedOps) {
        if (op->getNumResults() > 0) {
          // Rule 2 path: memgraph is not consulted.
          continue;
        }
        auto opBlockId = getForDirectChildBlockId(op);
        if (!opBlockId) {
          continue;
        }
        for (Operation *execOp : memGraph->getExecAfter(op)) {
          // sync_block_wait/sync_block_set ops are not memory side effects in
          // analyzing cleanup ops, therefore we skip them.
          if (isa<SyncBlockWaitOp>(execOp) || isa<SyncBlockSetOp>(execOp)) {
            continue;
          }
          auto execBlockId = CVPipeline::getOpBlockId(execOp);
          if (execBlockId && *execBlockId == *opBlockId) {
            sameBlockIdExecAfter[op].insert(execOp);
          }
        }
      }
    }

    // Erase cloned ops from bottom to top, matching the original ordering.
    // erasedOps mirrors the effect of the per-iteration graph rebuild: an op
    // already erased in this pass is treated as absent from the precomputed
    // exec-after sets.
    llvm::DenseSet<Operation *> erasedOps;
    for (int j = startIdx; j >= firstClonedIdx; --j) {
      Operation *op = curOps[j];
      bool shouldErase = isCube ? shouldEraseOpForCube(op, sameBlockIdExecAfter, erasedOps)
                                : shouldEraseOpForVector(op);
      if (shouldErase) {
        op->erase();
        erasedOps.insert(op);
      }
    }
  }

  // Check whether new forOp is valid after cleanup
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (op.hasAttr(CVPipeline::kClone)) {
      if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
          isa<hivm::FixpipeOp>(op)) {
        LDBG("[ERROR]: Cloned sync/fixpipe op should have been erased: " << op.getName() << "\n");
        return failure();
      }
    }
  }

  return success();
}

// Cleanup cloned ops for a single main loop
LogicalResult CloneOpsPass::cleanupClonedOpsInMainLoop(scf::ForOp forOp)
{
  ModuleOp module = getOperation();
  scope::ScopeOp scopeOp = forOp->getParentOfType<scope::ScopeOp>();
  if (!scopeOp) {
    return success();
  }

  auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
  if (!attr) {
    return success();
  }

  bool isCube = (attr == hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE));

  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(forOp, blockOps))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);
  if (failed(cleanupClonedOps(forOp, blockOps, idsInOrder, isCube, [&](scf::ForOp forOp) -> MemDepGraph {
    if (!isCube) {
      return nullptr;
    }
    auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    return std::make_unique<MemDepGraphT>(forOp, aliasAnalysis);
  }))) {
    return failure();
  }

  return success();
}

// Validate that each block_id's ops form contiguous ranges (not interleaved with other ids)
// e.g., [1,1,2,2] is valid, but [1,2,1,2] is invalid
static bool areBlockIdsConsecutive(scf::ForOp forOp)
{
  SmallVector<int> idsInOrder;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto blockIdOpt = CVPipeline::getOpBlockId(&op);
    if (!blockIdOpt) {
      LDBG("[ERROR]: Op missing ssbuffer.block_id: " << op.getName() << "\n");
      return false;
    }

    idsInOrder.push_back(static_cast<int>(*blockIdOpt));
  }

  // Check that each block_id forms a contiguous range
  for (size_t i = 0; i < idsInOrder.size();) {
    int currentId = idsInOrder[i];
    size_t j = i;

    while (j < idsInOrder.size() && idsInOrder[j] == currentId) {
      ++j;
    }

    for (size_t k = j; k < idsInOrder.size(); ++k) {
      if (idsInOrder[k] == currentId) {
        LDBG("[ERROR]: block_id: " << currentId << " is interleaved\n");
        return false;
      }
    }

    i = j;
  }

  return true;
}

LogicalResult CloneOpsPass::validateBlockIdsConsecutive(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }
    if (!areBlockIdsConsecutive(forOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  return success();
}

// Check that no op in a VECTOR scope's main_loop forOp has a tensor result \
// carrying the ssbuffer.clone attribute.
LogicalResult CloneOpsPass::validateClonedOpsInVector(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }

    scope::ScopeOp scopeOp = forOp->getParentOfType<scope::ScopeOp>();
    if (!scopeOp) {
      return WalkResult::advance();
    }

    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
    if (!attr) {
      return WalkResult::advance();
    }

    if (attr != hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR)) {
      return WalkResult::advance();
    }

    for (Operation &bodyOp : forOp.getBody()->without_terminator()) {
      if (!bodyOp.hasAttr(CVPipeline::kClone)) {
        continue;
      }
      if (isa<tensor::EmptyOp>(bodyOp)) {
        continue;
      }
      bool hasTensorDep = llvm::any_of(bodyOp.getResults(), [](Value result) {
        return isa<RankedTensorType>(result.getType());
      });
      if (hasTensorDep) {
        LDBG("[Error]: VECTOR main_loop contains cloned op with tensor type: "
             << bodyOp.getName() << "\n");
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

void CloneOpsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before cloneOps:\n" << module << "\n");

  // Validate block_ids are consecutive before cloning
  if (failed(validateBlockIdsConsecutive(module))) {
    signalPassFailure();
    return;
  }

  // Clone ops in vector/cube to ensure that each block_id has its own
  // ops without sharing
  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (failed(cloneOpsInMainLoop(forOp))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (failed(cleanupClonedOpsInMainLoop(forOp))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  LDBG("after cloneOps:\n" << module << "\n");

  // Validate no cloned tensor ops remaining in VECTOR main_loop forOp
  if (failed(validateClonedOpsInVector(module))) {
    signalPassFailure();
    return;
  }

}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCloneOpsPass()
{
  return std::make_unique<CloneOpsPass>();
}

} // namespace triton
} // namespace mlir
