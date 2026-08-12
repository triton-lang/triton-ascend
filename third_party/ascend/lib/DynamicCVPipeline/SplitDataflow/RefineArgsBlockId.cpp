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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/RefineArgsBlockId.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "refine-args-block-id";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;

void eraseOpsWithUnusedUsers(Operation *op, Block *forBlock) {
  llvm::SetVector<Operation *> toErase;
  llvm::SetVector<Operation *> visited;
  SmallVector<Operation *> worklist;

  worklist.push_back(op);

  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    for (Value operand : cur->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (defOp->getResult(0).getNumUses() == 1) {
          worklist.push_back(defOp);
        }
      }
    }
    cur->erase();
  }
}

bool cloneDepSubgraph(Operation *yieldDefOp, Block *forBlock,
                      SmallVector<Operation *> &clonedOps, IRMapping &mapping,
                      int targetBlockId,
                      CVPipeline::ComputeBlockIdManager &bm) {
  SmallVector<Operation *> worklist;
  llvm::SetVector<Operation *> visited;
  SmallVector<Operation *> toClone;

  // First pass: BFS to collect ops to clone
  visited.insert(yieldDefOp);
  toClone.push_back(yieldDefOp);
  worklist.push_back(yieldDefOp);

  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();

    for (Value operand : cur->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != forBlock || visited.count(defOp)) {
        continue;
      }
      if (defOp != yieldDefOp &&
          (defOp->getNumResults() != 1 ||
           !CVPipeline::isScalarLike(defOp->getResult(0)))) {
        return false;
      }
      visited.insert(defOp);
      toClone.push_back(defOp);
      worklist.push_back(defOp);
    }
  }

  // Second pass: clone in code order and update mapping
  llvm::sort(toClone,
             [](Operation *l, Operation *r) { return l->isBeforeInBlock(r); });

  for (Operation *cur : toClone) {
    Operation *cloned = cur->clone(mapping);
    forBlock->push_back(cloned);
    cloned->moveAfter(cur);
    bm.updateBlockId(cloned, targetBlockId);
    clonedOps.push_back(cloned);
    mapping.map(cur->getResult(0), cloned->getResult(0));
  }

  // Move cloned ops to after the last op in target block
  auto targetOps = bm.getOpsByBlockId(targetBlockId);
  Operation *lastOpInTargetBlock = nullptr;
  for (Operation *op : targetOps) {
    if (op->getBlock() != forBlock)
      continue;
    if (!lastOpInTargetBlock || op->isBeforeInBlock(lastOpInTargetBlock)) {
      lastOpInTargetBlock = op;
    }
  }
  if (lastOpInTargetBlock) {
    for (Operation *cloned : llvm::reverse(clonedOps)) {
      cloned->moveAfter(lastOpInTargetBlock);
    }
  }

  return true;
}

int getLoopCarriedArgIndex(Value operand, Block *block) {
  auto barg = dyn_cast<BlockArgument>(operand);
  if (!barg || barg.getOwner() != block ||
      !isa<scf::ForOp>(block->getParentOp())) {
    return -1;
  }
  unsigned argIdx = barg.getArgNumber();
  if (argIdx == 0) {
    return -1;
  }
  return argIdx;
}

int findFirstUser(BlockArgument iterArg, Block *forBlock,
                  CVPipeline::ComputeBlockIdManager &bm) {
  llvm::SetVector<Value> visited;
  SmallVector<std::pair<Value, int>> worklist;
  int firstUserBlockId = -1;
  Operation *firstUserOp = nullptr;

  for (OpOperand &use : iterArg.getUses()) {
    Operation *user = use.getOwner();
    auto userInblock = CVPipeline::getAncestorInBlock(user, forBlock);
    if (!userInblock) {
      continue;
    }
    if (isa<scf::YieldOp>(userInblock)) {
      continue;
    }
    if (firstUserOp == nullptr || userInblock->isBeforeInBlock(firstUserOp)) {
      firstUserOp = userInblock;
      firstUserBlockId = bm.getBlockIdByOp(userInblock);
    }
  }
  return firstUserBlockId;
}

bool isDependenceOther(Operation *yieldDefOp, Block *forBlock, int argsId,
                       const CVPipeline::MemoryDependenceGraph &memGraph) {
  // Now we only filter the different iter_arg dependency.
  for (Value operand : yieldDefOp->getOperands()) {
    if (operand.getDefiningOp())
      continue;
    // if have block argument from for block. Skip;
    if (getLoopCarriedArgIndex(operand, forBlock) != argsId + 1) {
      LOG_DEBUG("Yield def op depends on other arg:"
                << getLoopCarriedArgIndex(operand, forBlock) << "\n");
      return true;
    }
  }

  for (auto memDep : memGraph.getExecBefore(yieldDefOp)) {
    auto userInBlock = CVPipeline::getAncestorInBlock(memDep, forBlock);
    if (userInBlock) {
      LOG_DEBUG("Yield def op depends on other memory in for block: " << *memDep
                                                                      << "\n");
      return true;
    }
  }
  return false;
}

void processOnefor(scf::ForOp forOp, CVPipeline::ComputeBlockIdManager &bm,
                   const CVPipeline::MemoryDependenceGraph &memGraph) {

  Block *forBlock = &forOp.getRegion().front();
  auto yieldOp = dyn_cast<scf::YieldOp>(forBlock->getTerminator());
  if (!yieldOp) {
    LOG_DEBUG("No yield op found in for block\n");
    return;
  }
  ArrayRef<BlockArgument> iterArgs = forOp.getRegionIterArgs();

  for (size_t i = 0; i < iterArgs.size(); ++i) {
    BlockArgument argsi = iterArgs[i];

    Value yieldOperand = yieldOp.getOperand(i);
    Operation *yieldDefOp = yieldOperand.getDefiningOp();
    if (!yieldDefOp) {
      LOG_DEBUG("Yield operand is a block argument, skip. Operand: "
                << yieldOperand << "\n");
      continue;
    }
    LOG_DEBUG("yieldDefOp: " << *yieldDefOp << "\n"
                             << "idx: " << i << "\n");
    if (isDependenceOther(yieldDefOp, forBlock, i, memGraph)) {
      continue;
    }

    int updateBlockId = bm.getBlockIdByOp(yieldDefOp);
    LOG_DEBUG("Update block id for yield def op: " << updateBlockId << "\n");

    int firstUserBlockId = findFirstUser(argsi, forBlock, bm);
    LOG_DEBUG("First user block id: " << firstUserBlockId << "\n");

    if (firstUserBlockId == -1 || updateBlockId == firstUserBlockId) {
      continue;
    }

    IRMapping mapping;
    SmallVector<Operation *> clonedOps;

    // Clone entire depSubgraph (yieldDefOp + upstream deps) to target block
    if (!cloneDepSubgraph(yieldDefOp, forBlock, clonedOps, mapping,
                          firstUserBlockId, bm)) {
      LOG_DEBUG("Skip iter_arg " << i << " due to non-scalar dependency\n");
      continue;
    }

    // Check if clonedOps would create a cycle
    if (willCreateCycle(clonedOps, memGraph, firstUserBlockId, bm)) {
      LOG_DEBUG("Skip iter_arg " << i << " due to cycle detected\n");
      continue;
    }

    // Update yield to use cloned yieldDefOp result
    yieldOp.setOperand(i, mapping.lookup(yieldDefOp->getResult(0)));

    // Erase original yieldDefOp and its upstream ops that have no more users
    eraseOpsWithUnusedUsers(yieldDefOp, forBlock);
    LOG_DEBUG("Successfully moved iter_arg " << i << " from block "
                                             << updateBlockId << " to block "
                                             << firstUserBlockId << "\n");
  }
}

void RefineArgsBlockIdPass::runOnOperation() {
  LOG_DEBUG("\n--- enter RefineArgsBlockIdPass --->\n");
  ModuleOp moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  CVPipeline::ComputeBlockIdManager bm(moduleOp);
  auto &aa = getAnalysis<AliasAnalysis>();
  LOG_DEBUG(*moduleOp);
  moduleOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("ssbuffer.main_loop")) {
      auto memDepGraph = CVPipeline::MemoryDependenceGraph(forOp, aa);
      processOnefor(forOp, bm, memDepGraph);
    }
  });

  LOG_DEBUG("--- exit RefineArgsBlockIdPass --->\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createRefineArgsBlockIdPass() {
  return std::make_unique<RefineArgsBlockIdPass>();
}

void registerRefineArgsBlockIdPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRefineArgsBlockIdPass();
  });
}

} // namespace triton
} // namespace mlir
