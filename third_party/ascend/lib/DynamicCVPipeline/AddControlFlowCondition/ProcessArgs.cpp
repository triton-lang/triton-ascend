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
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"

static constexpr const char *DEBUG_TYPE = "ProcessArgs";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace mlir;
using namespace triton;

// Collects mapping from iter_arg index to block_ids that use it.
// For each iter_arg, tracks which block_ids reference it in their operations.
// `body` is the loop body (scf.for body or scf.while after-region body).
// `ivOffset` is 1 for scf.for (the induction variable sits at block arg 0;
// iter_args start at 1) and 0 for scf.while (no IV; iter_args start at 0).
// We index by the iter_arg's position in the iter_args list — i.e.
// argNumber - ivOffset — so callers can address entries by iter_arg index
// rather than absolute block-arg position. This matters for scf.for, where
// the iter_arg index is what yieldOp.getOperand(i) expects. We compare
// against body->getArguments() directly (rather than loopOp.getRegionIterArgs())
// because for scf.while those are different blocks' args (before vs after) and
// SSA equality is block-local.
static LogicalResult collectArgIndexToBlockIds(
    Block *body,
    unsigned ivOffset,
    llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds)
{
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: loop body is invalid or has no terminator\n");
    return failure();
  }

  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr) continue;
    int blockId = blockIdAttr.getInt();

    for (OpOperand &operand : op.getOpOperands()) {
      Value v = operand.get();
      for (BlockArgument iterArg : body->getArguments()) {
        int argIdx = iterArg.getArgNumber();
        if (argIdx < (int)ivOffset) {
          // scf.for's IV at block arg 0 — never an iter_arg.
          continue;
        }
        // Skip tensor-type iter_args, only process scalar and index types
        if (mlir::isa<TensorType>(iterArg.getType())) {
          continue;
        }
        if (v == iterArg) {
          argIndexToBlockIds[argIdx - (int)ivOffset].insert(blockId);
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
// compOp is the defining op of the iter_arg in the scf.yield operand list. The
// caller-provided body is the loop body (forOp body or whileOp after-body);
// the iter_arg's position in the yield matches the iter_arg's position in the
// region argument list, so this is op-agnostic.
static LogicalResult findCompOpInOwnerBlock(
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
// `loopOp` is the main-loop op (scf.for or scf.while) and scopes the walk to
// ops inside its body.
static void collectChainOps(
    Operation *loopOp,
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
        if (defOp->getParentOp() == loopOp && !chainOps.contains(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }
  }
}

// Builds computation info (compOp and chainOps) for each shared arg. `loopOp`
// is the main-loop op (scf.for or scf.while) and scopes the chain walk;
// `body` is the loop body (forOp body or whileOp after-body) and is used to
// locate the scf.yield terminator for finding the compOp.
static LogicalResult buildCompInfoForSharedArgs(
    Operation *loopOp,
    Block *body,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps)
{
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    if (sharedArgToCompOp.contains(argIndex)) continue;

    Operation *compOp = nullptr;
    if (failed(findCompOpInOwnerBlock(body, info, compOp))) {
      continue;
    }

    sharedArgToCompOp[argIndex] = compOp;

    llvm::DenseSet<Operation*> chainOps;
    collectChainOps(loopOp, compOp, chainOps);
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
//
// `iterArgs` is the list of iter_args of the loop op (forOp.getRegionIterArgs()
// or whileOp.getRegionIterArgs() — same API). `ivOffset` is 1 for scf.for (the
// induction variable sits at block arg 0) and 0 for scf.while (no induction
// variable, iter_args start at 0). `oldBlockArgs` is kept in the signature for
// source compatibility but is no longer used; clonedResults is the sole output.
static LogicalResult processSharedArgsIteration(
    Block *newBlock,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    const llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    const llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps,
    ValueRange iterArgs,
    unsigned ivOffset,
    SmallVector<Value> &clonedResults)
{
  unsigned numOriginalIterArgs = iterArgs.size();
  unsigned extraIterArgsBase = ivOffset + numOriginalIterArgs;

  int clonedArgIdx = clonedResults.size();
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    info.iterArg = iterArgs[argIndex];

    // The migrated iter_arg (original iter_arg moved to new block)
    Value migratedIterArg = newBlock->getArgument(argIndex + ivOffset);
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
// and builds computation info for each shared arg. `loopOp` is the main-loop op
// (scf.for or scf.while); `body` is its loop body (forOp body or whileOp
// after-body). The function dispatches on op type only for getBlockIdsInOrder,
// which has separate forOp/whileOp overloads.
static LogicalResult prepareSharedArgsData(
    Operation *loopOp,
    Block *body,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation*> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation*>> &sharedArgToChainOps)
{
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: loop body is invalid or has no terminator\n");
    return failure();
  }

  // ivOffset: 1 for scf.for (IV at block arg 0; iter_args start at 1), 0 for
  // scf.while (no IV; iter_args start at 0). collectArgIndexToBlockIds indexes
  // its result map by iter_arg index (0-based), so we subtract the offset
  // when storing.
  unsigned ivOffset = isa<scf::ForOp>(loopOp) ? 1 : 0;

  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;
  if (failed(collectArgIndexToBlockIds(body, ivOffset, argIndexToBlockIds))) {
    return failure();
  }

  SmallVector<int> idsInOrder;
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    idsInOrder = getBlockIdsInOrder(forOp);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    idsInOrder = getBlockIdsInOrder(whileOp);
  } else {
    LDBG("[Error]: loopOp is neither scf::ForOp nor scf::WhileOp\n");
    return failure();
  }
  if (failed(findSharedArgs(argIndexToBlockIds, idsInOrder, sharedArgsInfo))) {
    return failure();
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  LDBG("[INFO]: Found " << sharedArgsInfo.size() << " shared iter_args to process\n");

  if (failed(buildCompInfoForSharedArgs(loopOp, body, sharedArgsInfo,
                                        sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }

  return success();
}

// Builds new yield op with original operands plus cloned results. Generic over
// the loop op (scf.for or scf.while): uses oldBlock's scf.yield terminator as
// the source of original operands and newBlock (forOp body or whileOp
// after-body) as the destination. Location is taken from newOp.
static LogicalResult buildNewYieldOp(
    Block *oldBlock, Block *newBlock, Operation *newOp,
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
  builder.create<scf::YieldOp>(newOp->getLoc(), yieldOperands);
  oldYield.erase();
  return success();
}

// Replaces all uses of the old main-loop op (scf.for or scf.while) with the
// new op's results and erases the old op. Also transfers the
// intraCoreDependentMap entry from the old main-loop op to the new one. The
// map is keyed on Operation* (the main-loop op itself), so this works for
// both scf.for and scf.while.
static LogicalResult replaceForOpAndErase(Operation *oldOp, Operation *newOp,
                                          ControlFlowConditionInfo *info)
{
  if (oldOp->getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < oldOp->getNumResults(); ++i) {
      newResults.push_back(newOp->getResult(i));
    }
    oldOp->replaceAllUsesWith(newResults);
  }

  // Transfer intraCoreDependentMap entry from oldOp to newOp.
  if (info) {
    if (info->intraCoreDependentMap.count(oldOp)) {
      info->intraCoreDependentMap[newOp] = info->intraCoreDependentMap[oldOp];
      info->intraCoreDependentMap.erase(oldOp);
    }
  }

  oldOp->erase();
  return success();
}

// ============================================================
// scf.while support
// ============================================================
//
// The transformations below reuse the generic helpers defined for scf.for
// (collectArgIndexToBlockIds, findCompOpInOwnerBlock, collectChainOps,
// buildCompInfoForSharedArgs, prepareSharedArgsData, processSharedArgsIteration,
// buildNewYieldOp, replaceForOpAndErase) and only add what's truly new for
// scf.while: the scf.while-specific op construction, before+after body
// migration, and the before-region condition extension.
//
// Notes:
//   - scf.while has no induction variable, so before/after block args start at
//     0 with iter_args (processSharedArgsIteration takes ivOffset=0).
//   - scf.while's condition is the loop's continuation check. When we add a
//     clone iter_arg for a non-owner block, the loop's continuation must also
//     depend on the clone — otherwise a divergent clone chain would never
//     cause the loop to terminate. extendConditionWithClonedHandles that.

// Creates a new scf.while op with extra init args for shared arguments and
// matching extra result types. Empty before/after regions are populated with
// single blocks whose arg types match the new init args (to mirror the count
// carried by init args). The blocks are empty; migration and terminator
// insertion (buildNewConditionAndYieldOpsWhile) finish them.
static scf::WhileOp createNewWhileOp(scf::WhileOp whileOp,
                                     const SmallVector<SharedArgInfo> &sharedArgsInfo)
{
  OpBuilder builder(whileOp);

  SmallVector<Value> newInits(whileOp.getInits().begin(), whileOp.getInits().end());
  SmallVector<Type> newResultTypes(whileOp->getResultTypes().begin(),
                                   whileOp->getResultTypes().end());
  // Each non-owner block gets its own extra iter_arg whose init value is the
  // shared-arg's init. The result type grows in lockstep so the loop returns
  // the new iter_args at exit.
  for (const auto &info : sharedArgsInfo) {
    Value init = whileOp.getInits()[info.argIndex];
    newInits.push_back(init);
    newResultTypes.push_back(init.getType());
  }

  scf::WhileOp newWhileOp =
      builder.create<scf::WhileOp>(whileOp.getLoc(), newResultTypes, newInits);

  SmallVector<Type> argTypes;
  argTypes.reserve(newInits.size());
  for (Value v : newInits) {
    argTypes.push_back(v.getType());
  }

  SmallVector<Location> argLocs(newInits.size(), whileOp.getLoc());

  builder.createBlock(&newWhileOp.getBefore(), /*insertBefore=*/{}, argTypes,
                      argLocs);
  builder.createBlock(&newWhileOp.getAfter(), /*insertBefore=*/{}, argTypes,
                      argLocs);

  for (auto &attr : whileOp->getAttrs()) {
    newWhileOp->setAttr(attr.getName(), attr.getValue());
  }
  return newWhileOp;
}

// Extends the before-region condition by cloning the atomic check(s) that
// read each shared iter_arg, substituting the original iter_arg with its
// clone iter_arg. The cloned checks are appended to the new before block.
//
// Why: scf.while's condition is the loop's continuation check. When we add a
// clone iter_arg for a non-owner block, the loop's continuation must also
// depend on the clone — otherwise a divergent clone chain would never cause
// the loop to terminate. The new condition is the OR of the original cond
// and all the cloned-check results, which (after associativity) gives:
//
//   new_cond = original_cond
//            | cloned_check_for_each_shared_arg_in_each_atomic_check
//
// where each cloned_check uses the SAME cmp op (predicate, type, attributes)
// as the original — only the iter_arg operand is substituted with its clone.
//
// In the simple test case (one shared iter_arg, one atomic cmpi), this adds
// a single cloned cmpi and one arith.ori to combine them.
static LogicalResult extendConditionWithClonedChecks(
    scf::WhileOp oldWhileOp,
    scf::WhileOp newWhileOp,
    const SmallVector<SharedArgInfo> &sharedArgsInfo,
    SmallVectorImpl<Value> &clonedCondValues)
{
  if (sharedArgsInfo.empty()) {
    return success();
  }

  // The original cond value is the operand of the old scf.condition, which
  // still lives in the old before block (terminators are not moved by the
  // migrateBody calls in processSharedIterArgsInLoop).
  auto oldCond = oldWhileOp.getConditionOp();
  Value origCond = oldCond.getCondition();

  Block *newBefore = newWhileOp.getBeforeBody();
  unsigned numOriginalIterArgs = oldWhileOp.getInits().size();

  OpBuilder builder(newBefore, newBefore->end());

  for (const auto &info : sharedArgsInfo) {
    // After migration, the atomic check that used info.iterArg now reads
    // the new before block's argument at the same index (migrateBody
    // re-wires old args -> new args before moving ops).
    BlockArgument origIterArgInNew = newBefore->getArgument(info.argIndex);
    BlockArgument cloneIterArg =
        newBefore->getArgument(numOriginalIterArgs + info.newArgIndex);

    // Walk back from origCond inside the new before block. When we hit an op
    // that directly uses origIterArgInNew, it's a leaf atomic check: clone
    // it with the clone iter_arg, then stop descending through that op.
    llvm::DenseSet<Operation *> visited;
    SmallVector<Operation *> worklist;
    if (Operation *defOp = origCond.getDefiningOp()) {
      worklist.push_back(defOp);
    }

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (!visited.insert(op).second) {
        continue;
      }

      if (llvm::is_contained(op->getOperands(), origIterArgInNew)) {
        // Leaf atomic check. Clone the op (preserves cmp predicate and all
        // other attributes), substituting origIterArgInNew -> cloneIterArg.
        IRMapping mapper;
        mapper.map(origIterArgInNew, cloneIterArg);
        Operation *cloned = builder.clone(*op, mapper);
        // Mark the clone as belonging to the non-owner block. The
        // ssbuffer.clone attribute is left to CloneOps to fill in if it
        // chooses to (this is the same convention used elsewhere).
        cloned->setAttr("ssbuffer.block_id",
                        builder.getI32IntegerAttr(info.nonOwnerBlockId));
        clonedCondValues.push_back(cloned->getResult(0));
        continue;
      }

      // Not a leaf for this iter_arg — recurse into operands within the same
      // region to find leaves deeper in the expression.
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (defOp && defOp->getParentOp() == newWhileOp && !visited.contains(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }
  }

  return success();
}

// Builds the new scf.condition (in the before region of the new whileOp) and
// reuses buildNewYieldOp for the new scf.yield (in the after region). The
// yield operands are oldYield's operands + clonedResults; the cond is the
// original cond OR'd with each cloned check produced by
// extendConditionWithClonedChecks. The OR'd cond inherits the original cond
// defining op's ssbuffer.block_id (defaulting to the first shared arg's
// non-owner block id when no defining op carries a block_id).
static LogicalResult buildNewConditionAndYieldOpsWhile(
    scf::WhileOp oldWhileOp,
    scf::WhileOp newWhileOp,
    const SmallVector<SharedArgInfo> &sharedArgsInfo,
    const SmallVector<Value> &clonedCondValues,
    const SmallVector<Value> &clonedResults)
{
  // scf.condition: orig cond OR'd with all cloned cond values.
  auto oldCond = oldWhileOp.getConditionOp();
  Value origCond = oldCond.getCondition();

  // Inherit the block_id from the original cond's defining op so downstream
  // passes see the OR'd cond as part of the same block.
  int condBlockId = -1;
  if (Operation *defOp = origCond.getDefiningOp()) {
    if (auto attr = defOp->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      condBlockId = attr.getInt();
    }
  }
  if (condBlockId == -1 && !sharedArgsInfo.empty()) {
    condBlockId = sharedArgsInfo.front().nonOwnerBlockId;
  }

  Block *newBefore = newWhileOp.getBeforeBody();
  Value newCond = origCond;
  OpBuilder beforeBuilder(newBefore, newBefore->end());
  for (Value clonedCond : clonedCondValues) {
    // arith.cmpi returns i1, so the OR of two cmpi results is also i1. We
    // pass newCond.getType() explicitly so the result type matches the
    // existing cond (i1), keeping the signature consistent with
    // scf.condition's i1 cond slot.
    newCond = beforeBuilder.create<arith::OrIOp>(oldWhileOp.getLoc(),
                                                  newCond.getType(), newCond,
                                                  clonedCond);
    if (condBlockId != -1) {
      // Stamp the OR op (most recent) with the inherited block_id. We
      // walk back to the most recent builder insertion for the OR op.
      if (Operation *orOp = newCond.getDefiningOp()) {
        orOp->setAttr("ssbuffer.block_id",
                      beforeBuilder.getI32IntegerAttr(condBlockId));
      }
    }
  }

  SmallVector<Value> forwardedValues;
  for (BlockArgument arg : newWhileOp.getBeforeArguments()) {
    forwardedValues.push_back(arg);
  }
  beforeBuilder.create<scf::ConditionOp>(oldWhileOp.getLoc(), newCond,
                                          forwardedValues);
  oldCond.erase();

  // scf.yield reuses the generic helper.
  if (failed(buildNewYieldOp(oldWhileOp.getAfterBody(), newWhileOp.getAfterBody(),
                             newWhileOp, clonedResults))) {
    return failure();
  }
  return success();
}

// Main entry point for processing shared iter_args in a single main-loop op
// (scf.for or scf.while). Orchestrates data preparation, new op construction,
// body migration, and cloning. Dispatches on op type at each step so the
// forOp and whileOp pipelines share as much code as possible.
//
// Pipeline (common to both forOp and whileOp):
//   1. prepareSharedArgsData     — collect shared args & build chain info
//   2. createNew*Op              — construct a new loop op with extra iter_args
//   3. migrateBody               — move ops from old block(s) into new block(s)
//   4. processSharedArgsIteration — clone the chain for each non-owner block,
//                                   redirected to the new extra iter_arg
//   5. build terminator(s)       — forOp: scf.yield; whileOp: scf.condition
//                                   (OR'd with cloned checks) + scf.yield
//   6. replaceForOpAndErase      — splice new op in place of old, transfer any
//                                   intraCoreDependentMap entry to newOp (the
//                                   map is keyed on Operation* now, so the
//                                   same transfer applies to scf.for and
//                                   scf.while).
//
// Differences from forOp to whileOp:
//   - scf.while has no induction variable, so before/after block args start
//     at 0 with iter_args (processSharedArgsIteration takes ivOffset=0).
//   - scf.while has two regions (before + after) instead of one.
//   - scf.while's condition is the loop's continuation check. When we add a
//     clone iter_arg for a non-owner block, the loop's continuation must
//     also depend on the clone — otherwise a divergent clone chain would
//     never cause the loop to terminate. extendConditionWithClonedChecks
//     handles that.
static LogicalResult processSharedIterArgsInLoop(Operation *op,
                                                 ControlFlowConditionInfo *info)
{
  // Dispatch on op type once for the steps that need different inputs
  // (inspect body + ivOffset). Both forOp and whileOp are handled uniformly
  // below the type-specific work.
  Block *inspectBody = nullptr;
  unsigned ivOffset = 0;
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    inspectBody = forOp.getBody();
    ivOffset = 1; // scf.for has the IV at block arg 0; iter_args start at 1.
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    inspectBody = whileOp.getAfterBody();
    ivOffset = 0; // scf.while has no IV; iter_args start at 0.
  } else {
    LDBG("[Error]: op with ssbuffer.main_loop is neither scf::ForOp nor scf::WhileOp\n");
    return failure();
  }

  // 1. Prepare shared args data (dispatches on op type internally for
  //    getBlockIdsInOrder).
  SmallVector<SharedArgInfo> sharedArgsInfo;
  llvm::DenseMap<int, Operation*> sharedArgToCompOp;
  llvm::DenseMap<int, llvm::DenseSet<Operation*>> sharedArgToChainOps;
  if (failed(prepareSharedArgsData(op, inspectBody, sharedArgsInfo,
                                   sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  SmallVector<Value> clonedResults;

  // 2-6. Per-op-type pipeline. forOp and whileOp share the call sites
  // (processSharedArgsIteration, replaceForOpAndErase), but the create step
  // (different op factory), the migrate step (1 vs 2 bodies), the iteration
  // target (body + ivOffset), and the final terminator(s) differ.
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    scf::ForOp newForOp = createNewForOp(forOp, sharedArgsInfo);
    Block *oldBlock = forOp.getBody();
    Block *newBlock = newForOp.getBody();
    migrateBody(oldBlock, newBlock);

    if (failed(processSharedArgsIteration(newBlock, sharedArgsInfo,
                                          sharedArgToCompOp, sharedArgToChainOps,
                                          forOp.getRegionIterArgs(), 1,
                                          clonedResults))) {
      return failure();
    }
    if (failed(buildNewYieldOp(oldBlock, newBlock, newForOp, clonedResults))) {
      return failure();
    }
    return replaceForOpAndErase(forOp, newForOp, info);
  }

  // whileOp
  auto whileOp = cast<scf::WhileOp>(op);
  scf::WhileOp newWhileOp = createNewWhileOp(whileOp, sharedArgsInfo);
  migrateBody(whileOp.getBeforeBody(), newWhileOp.getBeforeBody());
  migrateBody(whileOp.getAfterBody(), newWhileOp.getAfterBody());

  if (failed(processSharedArgsIteration(newWhileOp.getAfterBody(), sharedArgsInfo,
                                        sharedArgToCompOp, sharedArgToChainOps,
                                        whileOp.getRegionIterArgs(), 0,
                                        clonedResults))) {
    return failure();
  }

  SmallVector<Value> clonedCondValues;
  if (failed(extendConditionWithClonedChecks(whileOp, newWhileOp, sharedArgsInfo,
                                             clonedCondValues))) {
    return failure();
  }
  if (failed(buildNewConditionAndYieldOpsWhile(whileOp, newWhileOp, sharedArgsInfo,
                                               clonedCondValues, clonedResults))) {
    return failure();
  }
  return replaceForOpAndErase(whileOp, newWhileOp, info);
}

// Walks module to find for/while ops with ssbuffer.main_loop attribute.
// Processes each main loop to handle shared iter_args. The per-op pipeline
// is shared (see processSharedIterArgsInLoop); this function only handles
// the type-agnostic walk + dispatch into the unified entry point.
LogicalResult ProcessArgsPass::processSharedIterArgs(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    if (failed(processSharedIterArgsInLoop(op, info))) {
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
