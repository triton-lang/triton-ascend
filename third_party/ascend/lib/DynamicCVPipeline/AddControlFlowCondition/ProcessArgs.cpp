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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

static constexpr const char *DEBUG_TYPE = "ProcessArgs";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;

// For each shared iter_arg, we need to track:
// - Which block_ids use it
// - Who is the owner (first block_id in order)
// - For each non-owner block, what new iter_arg index to use
struct SharedArgInfo {
  int argIndex;           // original iter_arg index (0, 1, 2...)
  Value iterArg;          // the original iter_arg value
  int ownerBlockId;       // block_id that "owns" this arg (first in order)
  int newArgIndex;        // new iter_arg index in the new for op (for this specific block)
  int nonOwnerBlockId;    // the non-owner block that needs a clone
};

static LogicalResult processSharedIterArgsInForOp(scf::ForOp forOp)
{
  Block *body = forOp.getBody();
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: forOp body is invalid or has no terminator\n");
    return failure();
  }

  // Step 1: Find which block_ids use which iter_args
  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;  // argIndex -> set of blockIds
  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr) continue;
    int blockId = blockIdAttr.getInt();

    for (OpOperand &operand : op.getOpOperands()) {
      Value v = operand.get();
      for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
        if (v == forOp.getRegionIterArgs()[i]) {
          argIndexToBlockIds[i].insert(blockId);
        }
      }
    }
  }

  // Step 2: Find iter_args used by multiple block_ids (shared args)
  // Get block_ids in order
  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);

  // For each shared arg, create a SharedArgInfo for each non-owner block
  SmallVector<SharedArgInfo> sharedArgsInfo;
  llvm::DenseMap<int, int> oldArgIndexToNewArgIndexBase;  // first new arg index per shared arg

  int extraArgCount = 0;
  for (auto &p : argIndexToBlockIds) {
    int argIndex = p.first;
    const llvm::DenseSet<int> &blockIds = p.second;

    if (blockIds.size() > 1) {
      // Find owner block (first in order)
      int ownerBlockId = -1;
      for (int id : idsInOrder) {
        if (blockIds.contains(id)) {
          ownerBlockId = id;
          break;
        }
      }
      if (ownerBlockId == -1) continue;

      // Record the base new arg index for this shared arg
      oldArgIndexToNewArgIndexBase[argIndex] = forOp.getNumRegionIterArgs() + extraArgCount;
      extraArgCount++;

      // For each non-owner block, create a SharedArgInfo
      for (int bid : blockIds) {
        if (bid != ownerBlockId) {
          SharedArgInfo info;
          info.argIndex = argIndex;
          info.iterArg = forOp.getRegionIterArgs()[argIndex];
          info.ownerBlockId = ownerBlockId;
          info.newArgIndex = oldArgIndexToNewArgIndexBase[argIndex];
          info.nonOwnerBlockId = bid;
          sharedArgsInfo.push_back(info);
        }
      }
    }
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  LDBG("Found " << sharedArgsInfo.size() << " shared iter_args to process\n");

  // Step 3: Find the computation chain for each shared arg
  // The chain starts from the owner block's operation that produces the yield input
  llvm::DenseMap<int, Operation*> sharedArgToCompOp;  // argIndex -> comp op
  llvm::DenseMap<int, llvm::DenseSet<Operation*>> sharedArgToChainOps;  // argIndex -> chain ops

  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    if (sharedArgToCompOp.contains(argIndex)) continue;

    Value iterArg = forOp.getRegionIterArgs()[argIndex];

    // Find the operation in owner block that uses iterArg and whose result goes to yield
    Operation *compOp = nullptr;
    for (Operation &op : body->without_terminator()) {
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
      if (!blockIdAttr || blockIdAttr.getInt() != info.ownerBlockId) continue;

      // Check if op uses iterArg
      bool usesIterArg = false;
      for (Value operand : op.getOperands()) {
        if (operand == iterArg) {
          usesIterArg = true;
          break;
        }
      }
      if (!usesIterArg) continue;

      // Check if result is used by yield
      for (Value result : op.getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (isa<scf::YieldOp>(use.getOwner())) {
            compOp = &op;
            break;
          }
        }
        if (compOp) break;
      }
      if (compOp) break;
    }

    if (!compOp) {
      LDBG("Could not find comp op for arg index " << argIndex);
      continue;
    }

    sharedArgToCompOp[argIndex] = compOp;

    // Collect the computation chain (backward traversal)
    llvm::DenseSet<Operation*> chainOps;
    SmallVector<Operation*> worklist;
    worklist.push_back(compOp);

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (chainOps.contains(op)) continue;
      chainOps.insert(op);

      for (Value operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (defOp->getParentOp() == forOp && !chainOps.contains(defOp)) {
            worklist.push_back(defOp);
          }
        }
      }
    }

    sharedArgToChainOps[argIndex] = chainOps;
  }

  // Step 4: Create new for op with extra iter_args
  OpBuilder builder(forOp);
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(), forOp.getInitArgs().end());

  // Add new init args (one per shared arg)
  for (auto &p : oldArgIndexToNewArgIndexBase) {
    int oldArgIndex = p.first;
    newInitArgs.push_back(forOp.getInitArgs()[oldArgIndex]);
  }

  scf::ForOp newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);

  // Copy attributes
  for (auto &attr : forOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }

  // Step 5: Migrate body - redirect old block args to new block args, move ops
  Block *oldBlock = forOp.getBody();
  Block *newBlock = newForOp.getBody();

  // Save old block arguments BEFORE replaceAllUsesWith
  // Note: oldBlock has (1 induction var + N iter_args) arguments
  // newBlock has (1 induction var + N + M iter_args) arguments where M = num shared args
  SmallVector<Value> oldBlockArgs;
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlockArgs.push_back(oldBlock->getArgument(i));
  }

  // Map from iter_arg index to block arg index (offset by 1 for induction var)
  auto getOldBlockArgIdx = [&](int iterArgIdx) { return iterArgIdx + 1; };
  auto getNewBlockArgIdx = [&](int iterArgIdx) { return iterArgIdx + 1; };

  // Redirect block arguments
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  // Move all operations
  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
    op.moveBefore(newBlock, newBlock->end());
  }

  // Step 6: For each shared arg info, clone the chain into the non-owner block
  // and replace uses of the old iter_arg with the new one
  SmallVector<Value> clonedResults;  // one per SharedArgInfo, in order

  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    Operation *compOp = sharedArgToCompOp[argIndex];
    if (!compOp) continue;

    llvm::DenseSet<Operation*> &chainOps = sharedArgToChainOps[argIndex];
    if (chainOps.empty()) continue;

    // Find the last op in the non-owner block
    Operation *lastOpInBlock = nullptr;
    for (Operation &op : newBlock->without_terminator()) {
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
      if (blockIdAttr && blockIdAttr.getInt() == info.nonOwnerBlockId) {
        lastOpInBlock = &op;
      }
    }

    // Create mapper: new iter_arg (at argIndex) -> cloned iter_arg (at newArgIndex)
    // After replaceAllUsesWith, operations use newBlock->getArgument(argIndex+1)
    // We need to map this to the clone at newArgIndex+1
    IRMapping argMapper;
    Value oldBlockArg = oldBlockArgs[getOldBlockArgIdx(info.argIndex)];  // Save for Step 7 replacement
    Value newBlockArg = newBlock->getArgument(getNewBlockArgIdx(info.newArgIndex));  // Clone
    // Map the NEW block arg (which is what operations use after replaceAllUsesWith) to the clone
    argMapper.map(newBlock->getArgument(getNewBlockArgIdx(info.argIndex)), newBlockArg);

    // Topological sort the chain
    SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
    if (failed(topologicalSort(sortedChain))) continue;

    // Clone the chain after lastOpInBlock
    OpBuilder cloneBuilder(newBlock, newBlock->end());
    if (lastOpInBlock) {
      cloneBuilder.setInsertionPointAfter(lastOpInBlock);
    }

    IRMapping resultMapper;
    for (Operation *op : sortedChain) {
      IRMapping opMapper;
      // Map operands
      for (OpOperand &operand : op->getOpOperands()) {
        Value oldVal = operand.get();
        Value newVal = oldVal;
        if (argMapper.contains(oldVal)) {
          newVal = argMapper.lookup(oldVal);
        }
        opMapper.map(oldVal, newVal);
      }

      // Skip if already cloned (shouldn't happen with topological sort)
      if (resultMapper.contains(op->getResult(0))) continue;

      // Clone the op
      Operation *cloned = cloneBuilder.clone(*op, opMapper);
      cloned->setAttr("ssbuffer.block_id", cloneBuilder.getI32IntegerAttr(info.nonOwnerBlockId));
      cloned->setAttr("ssbuffer.arg", cloneBuilder.getI32IntegerAttr(info.argIndex));

      resultMapper.map(op->getResult(0), cloned->getResult(0));
      cloneBuilder.setInsertionPointAfter(cloned);
    }

    // Record the cloned result
    Value clonedResult = resultMapper.lookup(compOp->getResult(0));
    clonedResults.push_back(clonedResult);

    // Step 7: In the non-owner block, replace the iter_arg with the cloned iter_arg
    // After replaceAllUsesWith, operands already use newBlock->getArgument(argIndex+1)
    // We need to replace them with newBlock->getArgument(newArgIndex+1) which is the clone
    Value originalArg = newBlock->getArgument(getNewBlockArgIdx(info.argIndex));
    for (Operation &op : newBlock->without_terminator()) {
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
      if (!blockIdAttr || blockIdAttr.getInt() != info.nonOwnerBlockId) continue;

      // Check each operand of this op - replace if it matches original arg
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value operandVal = op.getOperand(i);
        if (operandVal == originalArg) {
          op.setOperand(i, newBlockArg);
          op.setAttr("ssbuffer.arg", cloneBuilder.getI32IntegerAttr(info.argIndex));
        }
      }
    }
  }

  // Step 8: Build yield with original operands + cloned results
  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> yieldOperands;
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    yieldOperands.push_back(oldYield.getOperand(i));
  }
  for (auto &result : clonedResults) {
    yieldOperands.push_back(result);
  }

  builder.setInsertionPointToEnd(newBlock);
  builder.create<scf::YieldOp>(newForOp.getLoc(), yieldOperands);
  oldYield.erase();

  // Step 9: Replace uses and erase old for op
  if (forOp.getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      newResults.push_back(newForOp.getResult(i));
    }
    forOp.replaceAllUsesWith(newResults);
  }
  forOp.erase();

  return success();
}

LogicalResult ProcessArgsPass::processSharedIterArgs(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }

    if (failed(processSharedIterArgsInForOp(forOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}

void ProcessArgsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before processArgs:\n" << module << "\n");

  if (failed(processSharedIterArgs(module))) {
    signalPassFailure();
    return;
  }

  LDBG("after processArgs:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass()
{
  return std::make_unique<ProcessArgsPass>();
}

} // namespace triton
} // namespace mlir