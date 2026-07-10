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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include <optional>

using namespace mlir;
using namespace llvm;

// Collect all nested ops within an operation's regions
LogicalResult
triton::collectAllNestedOps(Operation *op,
                            llvm::DenseSet<Operation *> &regionOps) {
  if (!op) {
    return failure();
  }

  if (regionOps.contains(op)) {
    return success();
  }

  regionOps.insert(op);
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(triton::collectAllNestedOps(&nestedOp, regionOps))) {
          return failure();
        }
      }
    }
  }

  return success();
}

// Group operations by their block_id attribute
LogicalResult triton::collectOpsByBlockId(
    scf::ForOp forOp, llvm::DenseMap<int, SmallVector<Operation *>> &blockOps) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto attr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId)) {
      blockOps[attr.getInt()].push_back(&op);
    } else {
      return failure();
    }
  }

  return success();
}

// DFS for topological sort - returns failure if cycle detected
static LogicalResult
dfsTopologicalSort(Operation *op, llvm::DenseSet<Operation *> &visited,
                   llvm::DenseSet<Operation *> &inStack,
                   const llvm::DenseSet<Operation *> &ops,
                   llvm::DenseMap<Operation *, int> *opOrder,
                   SmallVectorImpl<Operation *> &sorted) {
  if (!op) {
    return success();
  }
  if (inStack.contains(op)) {
    return failure();
  }
  if (visited.contains(op)) {
    return success();
  }

  visited.insert(op);
  inStack.insert(op);

  SmallVector<Operation *> deps;
  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) {
      if (ops.contains(def)) {
        deps.push_back(def);
      }
    }
  }

  if (opOrder && !opOrder->empty()) {
    llvm::sort(deps, [&](Operation *a, Operation *b) {
      auto itA = opOrder->find(a);
      auto itB = opOrder->find(b);
      if (itA == opOrder->end() || itB == opOrder->end()) {
        return false;
      }
      return itA->second < itB->second;
    });
  }

  for (Operation *dep : deps) {
    if (failed(
            dfsTopologicalSort(dep, visited, inStack, ops, opOrder, sorted))) {
      return failure();
    }
  }

  inStack.erase(op);
  sorted.push_back(op);
  return success();
}

// Topological sort of operations based on operand dependencies
LogicalResult triton::topologicalSort(llvm::DenseSet<Operation *> &ops,
                                      llvm::DenseMap<Operation *, int> *opOrder,
                                      SmallVectorImpl<Operation *> &sorted) {
  llvm::DenseSet<Operation *> visited;
  llvm::DenseSet<Operation *> inStack;

  SmallVector<Operation *> opList(ops.begin(), ops.end());
  if (opOrder && !opOrder->empty()) {
    llvm::sort(opList, [&](Operation *a, Operation *b) {
      return (*opOrder)[a] < (*opOrder)[b];
    });
  }

  for (Operation *op : opList) {
    if (failed(
            dfsTopologicalSort(op, visited, inStack, ops, opOrder, sorted))) {
      return failure();
    }
  }
  return success();
}

LogicalResult triton::topologicalSort(SmallVector<Operation *> &ops) {
  llvm::DenseSet<Operation *> opSet(ops.begin(), ops.end());
  SmallVector<Operation *> sorted;

  if (failed(triton::topologicalSort(opSet, nullptr, sorted))) {
    return failure();
  }
  ops.assign(sorted.begin(), sorted.end());
  return success();
}

// Get block_ids in order of appearance in for loop body
SmallVector<int> triton::getBlockIdsInOrder(scf::ForOp forOp) {
  SmallVector<int> idsInOrder;
  llvm::DenseSet<int> seenIds;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto blockIdAttr =
            op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId)) {
      int id = blockIdAttr.getInt();
      if (seenIds.insert(id).second) {
        idsInOrder.push_back(id);
      }
    }
  }
  return idsInOrder;
}

// Get the block_id of the immediate child of scf.for that contains op
// For nested ops inside scf.if/scf.for, returns the block_id of the immediate
// child of scf.for Only considers scf.for ops that have ssbuffer.main_loop
// attribute
std::optional<int> triton::getForDirectChildBlockId(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  Operation *parent = op->getParentOp();
  while (parent) {
    // Found the main_loop forOp, op is its direct child
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(CVPipeline::kMainLoop)) {
        return CVPipeline::getOpBlockId(op);
      }
    }
    op = parent;
    parent = parent->getParentOp();
  }
  return std::nullopt;
}

// Find the tcb group id that contains value v
int triton::findTcbGroupId(
    Value v,
    llvm::DenseMap<int, SmallVector<Value>> &tightlyCoupledBufferGroups) {
  for (auto &tcbEntry : tightlyCoupledBufferGroups) {
    if (llvm::is_contained(tcbEntry.second, v)) {
      return tcbEntry.first;
    }
  }
  return -1;
}

// Get isCube/isVector based on the scope's tcore_type attribute
// Returns failure if scopeOp does not have tcore_type attribute or it's not
// CUBE/VECTOR
LogicalResult triton::getScopeType(Operation *scopeOp, bool &isCube,
                                   bool &isVector) {
  isCube = false;
  isVector = false;

  if (!scopeOp->hasAttr(CVPipeline::kTcoreType)) {
    return failure();
  }

  auto attr = scopeOp->getAttr(CVPipeline::kTcoreType);
  auto aiCAttr =
      hivm::TCoreTypeAttr::get(scopeOp->getContext(), hivm::TCoreType::CUBE);
  auto aiVAttr =
      hivm::TCoreTypeAttr::get(scopeOp->getContext(), hivm::TCoreType::VECTOR);
  if (attr == aiCAttr) {
    isCube = true;
  } else if (attr == aiVAttr) {
    isVector = true;
  }

  if (!isCube && !isVector) {
    return failure();
  }

  return success();
}

// Check if op is a scf.if whose body only contains hivm.hir.sync_block_wait,
// hivm.hir.sync_block_set and hivm.fixpipe ops (excluding terminators).
bool triton::isIfOpWithOnlySyncOps(Operation *op) {
  auto ifOp = dyn_cast<scf::IfOp>(op);
  if (!ifOp) {
    return false;
  }

  WalkResult result = ifOp->walk([&](Operation *innerOp) -> WalkResult {
    if (innerOp == op) {
      return WalkResult::advance();
    }
    if (innerOp->hasTrait<OpTrait::IsTerminator>()) {
      return WalkResult::advance();
    }
    if (!isa<hivm::SyncBlockWaitOp>(innerOp) &&
        !isa<hivm::SyncBlockSetOp>(innerOp) && !isa<hivm::FixpipeOp>(innerOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return !result.wasInterrupted();
}
