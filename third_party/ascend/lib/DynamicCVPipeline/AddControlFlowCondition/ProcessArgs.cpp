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

// Collects mapping from iter_arg index to block_ids that use it.
// For each iter_arg, tracks which block_ids reference it in their operations.
static LogicalResult collectArgIndexToBlockIds(
    scf::ForOp forOp,
    llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds)
{
  Block *body = forOp.getBody();
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: forOp body is invalid or has no terminator\n");
    return failure();
  }

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
  return success();
}

// Finds iter_args used by multiple block_ids (shared args).
// Determines owner block (first in order) and creates SharedArgInfo for each non-owner.
// Each non-owner block gets its own extra iter_arg.
static LogicalResult findSharedArgs(
    const llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds,
    const SmallVector<int> &idsInOrder,
    SmallVector<SharedArgInfo> &sharedArgsInfo)
{
  int extraArgCount = 0;
  for (auto &p : argIndexToBlockIds) {
    int argIndex = p.first;
    const llvm::DenseSet<int> &blockIds = p.second;

    if (blockIds.size() <= 1) continue;

    int ownerBlockId = -1;
    for (int id : idsInOrder) {
      if (blockIds.contains(id)) {
        ownerBlockId = id;
        break;
      }
    }
    if (ownerBlockId == -1) continue;

    // Each non-owner block for this argIndex gets its own extra iter_arg
    for (int bid : blockIds) {
      if (bid != ownerBlockId) {
        sharedArgsInfo.push_back(
            SharedArgInfo(argIndex, ownerBlockId, extraArgCount, bid));
        extraArgCount++;
      }
    }
  }
  return success();
}

// Finds the computation operation in owner block that produces the iter_arg value.
// compOp is the defining op of the iter_arg in the scf.yield operand list.
static LogicalResult findCompOpInOwnerBlock(
    scf::ForOp forOp,
    Block *body,
    const SharedArgInfo &info,
    Operation *&compOp)
{
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  Value yieldArg = yieldOp.getOperand(info.argIndex);

  if (auto *defOp = yieldArg.getDefiningOp()) {
    compOp = defOp;
    return success();
  }

  return failure();
}

// Collects all operations in the computation chain by backward traversal from compOp.
// Builds the dependency graph needed to clone the computation for non-owner blocks.
static void collectChainOps(
    scf::ForOp forOp,
    Operation *compOp,
    llvm::DenseSet<Operation*> &chainOps)
{
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
}

// Builds computation info (compOp and chainOps) for each shared arg.
static LogicalResult buildCompInfoForSharedArgs(
    scf::ForOp forOp,
    Block *body,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps)
{
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    if (sharedArgToCompOp.contains(argIndex)) continue;

    Operation *compOp = nullptr;
    if (failed(findCompOpInOwnerBlock(forOp, body, info, compOp))) {
      continue;
    }

    sharedArgToCompOp[argIndex] = compOp;

    llvm::DenseSet<Operation*> chainOps;
    collectChainOps(forOp, compOp, chainOps);
    sharedArgToChainOps[argIndex] = chainOps;
  }
  return success();
}

// Creates a new scf.for op with extra iter_args for shared arguments.
// Copies attributes from the original for op.
// Each SharedArgInfo entry (non-owner block) gets its own extra iter_arg.
static scf::ForOp createNewForOp(
    scf::ForOp forOp,
    const SmallVector<SharedArgInfo> &sharedArgsInfo)
{
  OpBuilder builder(forOp);
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(), forOp.getInitArgs().end());

  // Each non-owner block gets its own extra iter_arg
  for (auto &info : sharedArgsInfo) {
    newInitArgs.push_back(forOp.getInitArgs()[info.argIndex]);
  }

  scf::ForOp newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);

  for (auto &attr : forOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }
  return newForOp;
}

// Migrates operations from old block to new block.
// Redirects block arguments to new block arguments and moves all ops.
static void migrateBody(Block *oldBlock, Block *newBlock)
{
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
    op.moveBefore(newBlock, newBlock->end());
  }
}

// Clones the computation chain for a non-owner block.
// Topologically sorts the chain and clones each op with remapped operands.
// argRemapping: maps migrated iter_arg Value -> new extra iter_arg Value.
// resultMapper: maps original op results -> cloned op results.
// clonedArgIdx: unique index for this non-owner block's clone (used as ssbuffer.arg).
static LogicalResult cloneChainForBlock(
    SharedArgInfo &info,
    Operation *compOp,
    const llvm::DenseSet<Operation*> &chainOps,
    Block *newBlock,
    IRMapping &argRemapping,
    OpBuilder &cloneBuilder,
    IRMapping &resultMapper,
    int clonedArgIdx)
{
  if (!compOp || chainOps.empty()) {
    return failure();
  }

  SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
  if (failed(topologicalSort(sortedChain))) {
    return failure();
  }

  for (Operation *op : sortedChain) {
    IRMapping opMapper;
    for (OpOperand &operand : op->getOpOperands()) {
      Value oldVal = operand.get();
      Value newVal = oldVal;
      if (argRemapping.contains(oldVal)) {
        newVal = argRemapping.lookup(oldVal);
      } else if (resultMapper.contains(oldVal)) {
        // Operand is a result from earlier in the owner chain, use cloned result
        newVal = resultMapper.lookup(oldVal);
      }
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0))) continue;

    Operation *cloned = cloneBuilder.clone(*op, opMapper);
    cloned->setAttr("ssbuffer.block_id", cloneBuilder.getI32IntegerAttr(info.nonOwnerBlockId));
    cloned->setAttr("ssbuffer.arg", cloneBuilder.getI32IntegerAttr(info.argIndex));

    resultMapper.map(op->getResult(0), cloned->getResult(0));
    cloneBuilder.setInsertionPointAfter(cloned);
  }
  return success();
}

// Replaces iter_arg uses in non-owner block with the cloned iter_arg.
// argRemapping maps migrated iter_arg Value -> new extra iter_arg Value.
static LogicalResult replaceIterArgsInBlock(
    SharedArgInfo &info,
    Block *newBlock,
    IRMapping &argRemapping,
    OpBuilder &cloneBuilder)
{
  for (Operation &op : newBlock->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr || blockIdAttr.getInt() != info.nonOwnerBlockId) continue;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value operand = op.getOperand(i);
      if (argRemapping.contains(operand)) {
        Value newVal = argRemapping.lookup(operand);
        op.setOperand(i, newVal);
        op.setAttr("ssbuffer.arg", cloneBuilder.getI32IntegerAttr(info.argIndex));
      }
    }
  }
  return success();
}

// Processes each shared arg: finds insertion point, clones chain, replaces iter_args.
// Collects cloned results for building new yield operands.
static LogicalResult processSharedArgsIteration(
    scf::ForOp forOp,
    Block *newBlock,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    const llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    const llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps,
    const SmallVector<Value> &oldBlockArgs,
    SmallVector<Value> &clonedResults)
{
  unsigned numOriginalIterArgs = forOp.getNumRegionIterArgs();
  unsigned extraIterArgsBase = 1 + numOriginalIterArgs; // block arg index where extra iter_args start

  int clonedArgIdx = clonedResults.size();
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    info.iterArg = forOp.getRegionIterArgs()[argIndex];

    // The migrated iter_arg (original iter_arg moved to new block)
    Value migratedIterArg = newBlock->getArgument(argIndex + 1);
    // The new extra iter_arg added for this shared arg
    unsigned newExtraBlockArgIdx = extraIterArgsBase + info.newArgIndex;
    Value newExtraIterArg = newBlock->getArgument(newExtraBlockArgIdx);

    // Build argRemapping: migratedIterArg -> newExtraIterArg
    IRMapping argRemapping;
    argRemapping.map(migratedIterArg, newExtraIterArg);

    Operation *lastOpInBlock = nullptr;
    for (Operation &op : newBlock->without_terminator()) {
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
      if (blockIdAttr && blockIdAttr.getInt() == info.nonOwnerBlockId) {
        lastOpInBlock = &op;
      }
    }

    OpBuilder cloneBuilder(newBlock, newBlock->end());
    if (lastOpInBlock) {
      cloneBuilder.setInsertionPointAfter(lastOpInBlock);
    }

    IRMapping resultMapper;
    if (failed(cloneChainForBlock(info, sharedArgToCompOp.lookup(argIndex),
                                  sharedArgToChainOps.lookup(argIndex),
                                  newBlock, argRemapping,
                                  cloneBuilder, resultMapper, clonedArgIdx))) {
      continue;
    }

    if (failed(replaceIterArgsInBlock(info, newBlock, argRemapping, cloneBuilder))) {
      continue;
    }

    Value clonedResult = resultMapper.lookup(
        sharedArgToCompOp.lookup(argIndex)->getResult(0));
    clonedResults.push_back(clonedResult);
    clonedArgIdx++;
  }
  return success();
}

// Prepares all shared args data: collects arg->blockId mapping, finds shared args,
// and builds computation info for each shared arg.
static LogicalResult prepareSharedArgsData(
    scf::ForOp forOp,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps)
{
  Block *body = forOp.getBody();
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: forOp body is invalid or has no terminator\n");
    return failure();
  }

  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;
  if (failed(collectArgIndexToBlockIds(forOp, argIndexToBlockIds))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);
  if (failed(findSharedArgs(argIndexToBlockIds, idsInOrder, sharedArgsInfo))) {
    return failure();
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  LDBG("[INFO]: Found " << sharedArgsInfo.size() << " shared iter_args to process\n");

  if (failed(buildCompInfoForSharedArgs(forOp, body, sharedArgsInfo,
                                        sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }

  return success();
}

// Builds new yield op with original operands plus cloned results.
static LogicalResult buildNewYieldOp(
    Block *oldBlock, Block *newBlock, scf::ForOp newForOp,
    const SmallVector<Value> &clonedResults)
{
  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> yieldOperands;

  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    yieldOperands.push_back(oldYield.getOperand(i));
  }
  for (auto &result : clonedResults) {
    yieldOperands.push_back(result);
  }

  OpBuilder builder = OpBuilder::atBlockEnd(newBlock);
  builder.create<scf::YieldOp>(newForOp.getLoc(), yieldOperands);
  oldYield.erase();
  return success();
}

// Replaces all uses of old for op with new for op results and erases old for op.
static LogicalResult replaceForOpAndErase(scf::ForOp oldForOp, scf::ForOp newForOp)
{
  if (oldForOp.getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < oldForOp.getNumResults(); ++i) {
      newResults.push_back(newForOp.getResult(i));
    }
    oldForOp.replaceAllUsesWith(newResults);
  }
  oldForOp.erase();
  return success();
}

// Main entry point for processing shared iter_args in a single for op.
// Orchestrates data preparation, new for op creation, body migration, and cloning.
static LogicalResult processSharedIterArgsInForOp(scf::ForOp forOp)
{
  SmallVector<SharedArgInfo> sharedArgsInfo;
  llvm::DenseMap<int, Operation*> sharedArgToCompOp;
  llvm::DenseMap<int, llvm::DenseSet<Operation*>> sharedArgToChainOps;

  if (failed(prepareSharedArgsData(forOp, sharedArgsInfo,
                                   sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  scf::ForOp newForOp = createNewForOp(forOp, sharedArgsInfo);
  Block *oldBlock = forOp.getBody();
  Block *newBlock = newForOp.getBody();

  SmallVector<Value> oldBlockArgs;
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlockArgs.push_back(oldBlock->getArgument(i));
  }

  migrateBody(oldBlock, newBlock);

  SmallVector<Value> clonedResults;
  if (failed(processSharedArgsIteration(forOp, newBlock, sharedArgsInfo,
                                       sharedArgToCompOp, sharedArgToChainOps,
                                       oldBlockArgs, clonedResults))) {
    return failure();
  }

  if (failed(buildNewYieldOp(oldBlock, newBlock, newForOp, clonedResults))) {
    return failure();
  }

  if (failed(replaceForOpAndErase(forOp, newForOp))) {
    return failure();
  }

  return success();
}

// Walks module to find for ops with ssbuffer.main_loop attribute.
// Processes each main loop to handle shared iter_args.
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
