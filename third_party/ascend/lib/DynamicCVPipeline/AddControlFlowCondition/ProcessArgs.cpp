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
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/OuterEventLoop.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "ProcessArgs";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace mlir;
using namespace CVPipeline;
using namespace triton;

// Collects iter_arg index -> block_ids that use it. `ivOffset` is 1 for
// scf.for (IV at arg 0) or 0 for scf.while. Uses body args directly
// (block-local).
static LogicalResult collectArgIndexToBlockIds(
    Block *body, unsigned ivOffset,
    llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds) {
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: loop body is invalid or has no terminator");
    return failure();
  }

  for (auto iterArg : body->getArguments()) {
    int argIdx = iterArg.getArgNumber();
    if (argIdx < (int)ivOffset) {
      // scf.for's IV at block arg 0 — never an iter_arg.
      continue;
    }
    // Skip tensor-type iter_args, only process scalar and index types
    if (mlir::isa<TensorType>(iterArg.getType())) {
      continue;
    }
    for (auto &use : iterArg.getUses()) {
      Operation *owner = use.getOwner();
      if (auto blockId = getLoopDirectChildBlockId(owner)) {
        argIndexToBlockIds[argIdx - (int)ivOffset].insert(*blockId);
      }
    }
  }
  return success();
}

// Finds iter_args used by multiple block_ids. Owner = first block in order;
// each non-owner gets its own extra iter_arg via SharedArgInfo.
static LogicalResult findSharedArgs(
    const llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds,
    const SmallVector<int> &idsInOrder,
    SmallVector<SharedArgInfo> &sharedArgsInfo) {
  int extraArgCount = 0;
  for (auto &p : argIndexToBlockIds) {
    int argIndex = p.first;
    const llvm::DenseSet<int> &blockIds = p.second;

    if (blockIds.size() <= 1)
      continue;

    int ownerBlockId = -1;
    for (int id : idsInOrder) {
      if (blockIds.contains(id)) {
        ownerBlockId = id;
        break;
      }
    }
    if (ownerBlockId == -1)
      continue;

    // Each non-owner block for this argIndex gets its own extra iter_arg
    for (int bid : blockIds) {
      if (bid == ownerBlockId)
        continue;
      sharedArgsInfo.push_back(
          SharedArgInfo(argIndex, ownerBlockId, extraArgCount, bid));
      extraArgCount++;
    }
  }
  return success();
}

// Returns the op defining iter_arg at `argIndex` in `body`'s scf.yield (top
// of update chain), or nullptr if out of bounds. Shared by both paths.
static Operation *findYieldDefiningOp(Block *body, unsigned argIndex) {
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  if (argIndex >= yieldOp.getNumOperands()) {
    return nullptr;
  }
  return yieldOp.getOperand(argIndex).getDefiningOp();
}

// Collects all operations in the computation chain by backward traversal
// from compOp, scoped to ops inside `loopOp`'s body.
static void collectChainOps(Operation *loopOp, Operation *compOp,
                            llvm::DenseSet<Operation *> &chainOps) {
  SmallVector<Operation *> worklist;
  worklist.push_back(compOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (chainOps.contains(op))
      continue;
    chainOps.insert(op);

    op->walk([&](Operation *nestOp) {
      for (Value operand : nestOp->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (defOp->getParentOp() == loopOp && !chainOps.contains(defOp)) {
            worklist.push_back(defOp);
          }
        }
      }
    });
  }
}

// Builds compOp + chainOps for each shared arg. `body` is the loop body
// (forOp body or whileOp after-body) used to locate scf.yield.
static LogicalResult buildCompInfoForSharedArgs(
    Operation *loopOp, Block *body, SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps) {
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    if (sharedArgToCompOp.contains(argIndex))
      continue;

    Operation *compOp = findYieldDefiningOp(body, argIndex);
    if (!compOp) {
      continue;
    }

    sharedArgToCompOp[argIndex] = compOp;

    llvm::DenseSet<Operation *> chainOps;
    collectChainOps(loopOp, compOp, chainOps);
    sharedArgToChainOps[argIndex] = chainOps;
  }
  return success();
}

// Clones the computation chain for a non-owner block (topo-sorted, operands
// remapped). argRemapping: iter_arg -> new extra. clonedArgIdx: this block's
// arg.
static LogicalResult
cloneChainForBlock(SharedArgInfo &info, Operation *compOp,
                   const llvm::DenseSet<Operation *> &chainOps, Block *newBlock,
                   IRMapping &argRemapping, OpBuilder &cloneBuilder,
                   IRMapping &resultMapper, int clonedArgIdx) {
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
        // Operand is a result from earlier in the owner chain, use cloned
        // result
        newVal = resultMapper.lookup(oldVal);
      }
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0)))
      continue;

    Operation *cloned = cloneBuilder.clone(*op, opMapper);
    if (!cloned) {
      LDBG("Clone chain failed when cloning op " << *op);
      return failure();
    }
    cloned->setAttr(CVPipeline::kBlockId,
                    cloneBuilder.getI32IntegerAttr(info.nonOwnerBlockId));
    cloned->setAttr(CVPipeline::kArg,
                    cloneBuilder.getI32IntegerAttr(info.argIndex));

    resultMapper.map(op->getResult(0), cloned->getResult(0));
    cloneBuilder.setInsertionPointAfter(cloned);
  }
  return success();
}

// Replaces iter_arg uses in non-owner block with the cloned iter_arg.
// argRemapping maps migrated iter_arg Value -> new extra iter_arg Value.
static LogicalResult replaceIterArgsInBlock(SharedArgInfo &info,
                                            Block *newBlock,
                                            IRMapping &argRemapping,
                                            OpBuilder &cloneBuilder) {
  for (auto arg : newBlock->getArguments()) {
    if (!argRemapping.contains(arg)) {
      continue;
    }
    Value newVal = argRemapping.lookup(arg);

    SmallVector<OpOperand *, 8> usesToReplace;
    for (OpOperand &use : arg.getUses()) {
      auto blockId = getLoopDirectChildBlockId(use.getOwner());
      if (blockId && *blockId == info.nonOwnerBlockId) {
        usesToReplace.push_back(&use);
      }
    }
    for (OpOperand *use : usesToReplace) {
      use->set(newVal);
      use->getOwner()->setAttr(CVPipeline::kArg,
                               cloneBuilder.getI32IntegerAttr(info.argIndex));
    }
  }

  return success();
}

// Processes each shared arg: finds insertion point, clones chain, replaces
// iter_args. `ivOffset` is 1 for scf.for (IV at arg 0), 0 for scf.while.
static LogicalResult processSharedArgsIteration(
    Block *newBlock, SmallVector<SharedArgInfo> &sharedArgsInfo,
    const llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    const llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps,
    ValueRange iterArgs, unsigned ivOffset, SmallVector<Value> &clonedResults) {
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
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
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
                                  newBlock, argRemapping, cloneBuilder,
                                  resultMapper, clonedArgIdx))) {
      continue;
    }

    if (failed(replaceIterArgsInBlock(info, newBlock, argRemapping,
                                      cloneBuilder))) {
      continue;
    }

    Value clonedResult =
        resultMapper.lookup(sharedArgToCompOp.lookup(argIndex)->getResult(0));
    clonedResults.push_back(clonedResult);
    clonedArgIdx++;
  }
  return success();
}

// Prepares all shared args data: collects arg->blockId mapping, finds shared
// args, and builds computation info for each shared arg.
static LogicalResult prepareSharedArgsData(
    Operation *loopOp, Block *body, SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps) {
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: loop body is invalid or has no terminator");
    return failure();
  }

  // ivOffset: 1 for scf.for (IV at block arg 0), 0 for scf.while (no IV).
  unsigned ivOffset = isa<scf::ForOp>(loopOp) ? 1 : 0;

  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;
  if (failed(collectArgIndexToBlockIds(body, ivOffset, argIndexToBlockIds))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(loopOp);
  if (idsInOrder.empty() && !MainLoop(loopOp).getBody()) {
    LDBG("[Error]: loopOp is neither scf::ForOp nor scf::WhileOp");
    return failure();
  }
  if (failed(findSharedArgs(argIndexToBlockIds, idsInOrder, sharedArgsInfo))) {
    return failure();
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  LDBG("[INFO]: Found " << sharedArgsInfo.size()
                        << " shared iter_args to process");

  if (failed(buildCompInfoForSharedArgs(loopOp, body, sharedArgsInfo,
                                        sharedArgToCompOp,
                                        sharedArgToChainOps))) {
    return failure();
  }

  return success();
}

// Replaces all uses of the old main-loop op with the new op's results, erases
// the old op, and transfers its intraCoreDependentMap entry to the new op.
static LogicalResult replaceMainLoopOpAndErase(Operation *oldOp,
                                               Operation *newOp,
                                               ControlFlowConditionInfo *info) {
  replaceOpResultUses(oldOp, newOp);

  // Transfer intraCoreDependentMap entry from oldOp to newOp.
  if (info && info->intraCoreDependentMap.count(oldOp)) {
    info->intraCoreDependentMap[newOp] = info->intraCoreDependentMap[oldOp];
    info->intraCoreDependentMap.erase(oldOp);
  }

  oldOp->erase();
  return success();
}

// Derives the body block to inspect for shared-iter_arg analysis and the IV
// offset (1 for scf.for, 0 for scf.while). Returns false if `op` is neither.
static bool getOpIterParams(Operation *op, Block *&inspectBody,
                            unsigned &ivOffset) {
  MainLoop ml = MainLoop(op);
  if (ml.getBody()) {
    inspectBody = ml.getBody();
    ivOffset = ml.isWhile() ? 0 : 1;
    return true;
  }
  return false;
}

// Completes the scf.for path: migrate body, clone per-block chains, rebuild
// scf.yield, swap old op out.
static LogicalResult processSharedArgsInForOp(
    scf::ForOp forOp, scf::ForOp newForOp,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    const llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    const llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps,
    ControlFlowConditionInfo *info) {
  Block *oldBlock = forOp.getBody();
  Block *newBlock = newForOp.getBody();
  migrateBody(oldBlock, newBlock);

  SmallVector<Value> clonedResults;
  if (failed(processSharedArgsIteration(
          newBlock, sharedArgsInfo, sharedArgToCompOp, sharedArgToChainOps,
          MainLoop(forOp).getIterArgs(), 1, clonedResults))) {
    return failure();
  }
  if (failed(buildNewYieldOp(oldBlock, newBlock, newForOp, clonedResults))) {
    return failure();
  }
  return replaceMainLoopOpAndErase(forOp, newForOp, info);
}

// Completes the scf.while path: migrate before/after bodies, clone per-block
// chains, rebuild scf.yield/condition, transfer maps, swap old op out.
LogicalResult ProcessArgsPass::processSharedArgsInWhileOp(
    scf::WhileOp whileOp, scf::WhileOp newWhileOp,
    SmallVector<SharedArgInfo> &sharedArgsInfo,
    const llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    const llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps,
    ControlFlowConditionInfo *info) {
  migrateWhileBodies(whileOp, newWhileOp);

  SmallVector<Value> clonedResults;
  if (failed(processSharedArgsIteration(
          newWhileOp.getAfterBody(), sharedArgsInfo, sharedArgToCompOp,
          sharedArgToChainOps, MainLoop(whileOp).getIterArgs(), 0,
          clonedResults))) {
    return failure();
  }
  buildNewWhileCondition(whileOp, newWhileOp);
  if (failed(buildNewYieldOp(whileOp.getAfterBody(), newWhileOp.getAfterBody(),
                             newWhileOp, clonedResults))) {
    return failure();
  }
  // Transfer originalWhileIterArgIndices so downstream still resolves
  // newWhileOp->old arg indices.
  if (originalWhileIterArgIndices.count(whileOp)) {
    originalWhileIterArgIndices[newWhileOp] =
        originalWhileIterArgIndices[whileOp];
    originalWhileIterArgIndices.erase(whileOp);
  }
  // Transfer whileBlockArgMap entries (local + info mirror) so the (block_id,
  // new_arg_idx) -> old_arg_idx mapping survives the old-op erase.
  if (localWhileBlockArgMap.count(whileOp)) {
    localWhileBlockArgMap[newWhileOp] =
        std::move(localWhileBlockArgMap[whileOp]);
    localWhileBlockArgMap.erase(whileOp);
  }
  if (info && info->whileBlockArgMap.count(whileOp)) {
    info->whileBlockArgMap[newWhileOp] =
        std::move(info->whileBlockArgMap[whileOp]);
    info->whileBlockArgMap.erase(whileOp);
  }
  return replaceMainLoopOpAndErase(whileOp, newWhileOp, info);
}

// Single entry point for processing shared iter_args in a main-loop op
// (scf.for or scf.while). whileOp builds new scf.condition, transfers indices.
static LogicalResult processOuterEventControl(scf::ForOp loopOp,
                                              ControlFlowConditionInfo *info);

LogicalResult
ProcessArgsPass::processSharedIterArgsInLoop(Operation *op,
                                             ControlFlowConditionInfo *info) {
  if (isa<scf::ForOp>(op) && op->hasAttr(kOuterEventControlCount))
    return processOuterEventControl(cast<scf::ForOp>(op), info);
  Block *inspectBody = nullptr;
  unsigned ivOffset = 0;
  if (!getOpIterParams(op, inspectBody, ivOffset)) {
    LDBG("[Error]: op with ssbuffer.main_loop is neither scf::ForOp nor "
         "scf::WhileOp");
    return failure();
  }

  SmallVector<SharedArgInfo> sharedArgsInfo;
  llvm::DenseMap<int, Operation *> sharedArgToCompOp;
  llvm::DenseMap<int, llvm::DenseSet<Operation *>> sharedArgToChainOps;
  if (failed(prepareSharedArgsData(op, inspectBody, sharedArgsInfo,
                                   sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }
  if (sharedArgsInfo.empty()) {
    return success();
  }

  // Build extra init values from original arg values at shared indices. Each
  // new iter_arg shadows the existing one at the same index.
  SmallVector<Value> extraInitArgs;
  extraInitArgs.reserve(sharedArgsInfo.size());
  ValueRange origInits;
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    origInits = forOp.getInitArgs();
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    origInits = whileOp.getInits();
  }
  for (const auto &info : sharedArgsInfo) {
    extraInitArgs.push_back(origInits[info.argIndex]);
  }

  Operation *newOp = createMainLoopOpWithExtras(op, extraInitArgs);
  if (!newOp) {
    return failure();
  }

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return processSharedArgsInForOp(forOp, cast<scf::ForOp>(newOp),
                                    sharedArgsInfo, sharedArgToCompOp,
                                    sharedArgToChainOps, info);
  }
  return processSharedArgsInWhileOp(
      cast<scf::WhileOp>(op), cast<scf::WhileOp>(newOp), sharedArgsInfo,
      sharedArgToCompOp, sharedArgToChainOps, info);
}

// Walks module to find for/while ops with ssbuffer.main_loop attribute and
// dispatches each into processSharedIterArgsInLoop.
LogicalResult ProcessArgsPass::processSharedIterArgs(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
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

// ----- Per-block update chain for scf.while iter_args used in scf.condition
// For each scf.while op with main_loop, every block_id's run keeps its own
// iter_arg value; clones update chain into each block, extends scf.while with
// one extra iter_arg per (block_id, original iter_arg), and records
// (new_arg_idx -> old_arg_idx) in info->whileBlockArgMap.

// Snapshots original iter_args of every scf.while with main_loop; per whileOp
// clones cond-used update chains per block_id. Recorded in
// info->whileBlockArgMap.
LogicalResult
ProcessArgsPass::updateIndependentCondsInWhileBlocks(ModuleOp module) {
  // Collect whileOps in a worklist (must NOT mutate IR during the walk —
  // processWhileIterArgsInWhileOp replaces the old whileOp). Snapshot
  // original iter_args in the same pass so we can later identify which
  // iter_args were referenced by scf.condition in the input.
  SmallVector<scf::WhileOp> worklist;
  module.walk([&](scf::WhileOp whileOp) {
    if (!isMainLoopOp(whileOp))
      return;
    SmallVector<unsigned> indices;
    for (unsigned i = 0; i < whileOp.getNumOperands(); ++i)
      indices.push_back(i);
    originalWhileIterArgIndices[whileOp] = indices;
    worklist.push_back(whileOp);
  });
  for (scf::WhileOp whileOp : worklist) {
    if (failed(processWhileIterArgsInWhileOp(whileOp, info)))
      return failure();
  }

  // Dump whileBlockArgMap (new_whileop -> block_id -> (new_arg_idx ->
  // old_arg_idx)); pass-local map so observable when --process-args runs
  // standalone.
  dumpWhileBlockArgMap(localWhileBlockArgMap,
                       "whileBlockArgMap contents (new_whileop -> block_id -> "
                       "(new_arg_idx -> old_arg_idx))");
  return success();
}

// Returns original iter_arg indices contributing to cond value of scf.condition
// for `whileOp`. Walks def-chain to before-block BlockArguments (cond-used).
static llvm::DenseSet<unsigned> collectConditionUsedIterArgIndices(
    scf::WhileOp whileOp, const SmallVector<unsigned> &originalIndices) {
  llvm::DenseSet<unsigned> used;
  auto cond = whileOp.getConditionOp();
  Value condValue = cond.getCondition();
  Block *beforeBlock = whileOp.getBeforeBody();

  // BFS over def-chain of condValue; stops at before-block BlockArgument
  // (collected) or no defining op in region (parent BlockArgument — ignored).
  llvm::SmallPtrSet<Value, 16> visited;
  SmallVector<Value> worklist;
  worklist.push_back(condValue);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      if (blockArg.getOwner() == beforeBlock) {
        unsigned idx = blockArg.getArgNumber();
        if (idx < originalIndices.size()) {
          used.insert(idx);
        }
      }
      continue;
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      continue;
    for (Value operand : defOp->getOperands()) {
      worklist.push_back(operand);
    }
  }
  return used;
}

// (findYieldCompOpForWhile folded into findYieldDefiningOp — see top of
// file. Callers now pass `whileOp.getAfterBody()` directly.)

// Returns the last op in `body` whose `ssbuffer.block_id` matches `blockId`
// (i.e., end of the block's run of consecutive ops).
static Operation *findLastOpWithBlockId(Block *body, int blockId) {
  Operation *last = nullptr;
  for (Operation &op : body->without_terminator()) {
    auto attr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (attr && attr.getInt() == blockId) {
      last = &op;
    }
  }
  return last;
}

// Builds a clone of the update chain after the last op with `blockId` in
// `body`. Annotates with ssbuffer.while_arg/block_id; remaps operands.
static LogicalResult cloneUpdateChainForWhileBlock(
    scf::WhileOp whileOp, Block *body, Operation *compOp, int blockId,
    unsigned originalArgIndex, unsigned newArgIndex,
    const llvm::DenseSet<Operation *> &chainOps, Value &cloned) {
  SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
  if (failed(topologicalSort(sortedChain))) {
    cloned = Value();
    return failure();
  }

  // Find insertion point: after the last op with this block_id.
  Operation *lastOp = findLastOpWithBlockId(body, blockId);
  OpBuilder builder(body, body->end());
  if (lastOp) {
    builder.setInsertionPointAfter(lastOp);
  }
  // If no op has this block_id, the builder is at body->end() (set by ctor);
  // avoid setInsertionPoint(terminator) since it may not exist yet.

  IRMapping resultMapper;
  for (Operation *op : sortedChain) {
    IRMapping opMapper;
    for (OpOperand &operand : op->getOpOperands()) {
      Value oldVal = operand.get();
      Value newVal = oldVal;
      // Remap the original iter_arg to the new iter_arg for this block;
      // chain-internal operands remap to the corresponding cloned value.
      auto blockArg = dyn_cast<BlockArgument>(oldVal);
      if (blockArg && blockArg.getOwner() == body &&
          (unsigned)blockArg.getArgNumber() == originalArgIndex) {
        newVal = body->getArgument(newArgIndex);
      } else if (resultMapper.contains(oldVal)) {
        newVal = resultMapper.lookup(oldVal);
      }
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0)))
      continue;

    Operation *cloned = builder.clone(*op, opMapper);
    cloned->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
    cloned->setAttr(CVPipeline::kWhileArg,
                    builder.getI32IntegerAttr(originalArgIndex));
    resultMapper.map(op->getResult(0), cloned->getResult(0));
    builder.setInsertionPointAfter(cloned);
  }

  cloned = resultMapper.lookup(compOp->getResult(0));
  return success();
}

// For each cond-used iter_arg, plans one new iter_arg per (block_id, origIdx).
// compOp + def-chain must be computed BEFORE migrateBody (caused lookup crash).
static WhileIterArgClonePlan
planWhileIterArgDescriptors(scf::WhileOp whileOp,
                            const SmallVector<unsigned> &originalIndices,
                            const llvm::DenseSet<unsigned> &condUsed) {
  WhileIterArgClonePlan plan;
  SmallVector<int> blockIdsInOrder = getBlockIdsInOrder(whileOp);
  if (blockIdsInOrder.empty())
    return plan;
  unsigned nextNewArgIdx = whileOp.getNumOperands();
  unsigned descIdx = 0;
  for (unsigned origIdx : originalIndices) {
    if (!condUsed.contains(origIdx))
      continue;
    plan.posInClonedVec[origIdx] = descIdx++;

    Operation *compOp = findYieldDefiningOp(whileOp.getAfterBody(), origIdx);
    if (!compOp) {
      LDBG("[WARN]: no compOp for while iter_arg idx=" << origIdx);
      continue;
    }

    llvm::DenseSet<Operation *> chainOps;
    collectChainOps(whileOp, compOp, chainOps);
    plan.compOp[origIdx] = compOp;
    plan.chainOps[origIdx] = chainOps;

    for (int blockId : blockIdsInOrder) {
      plan.newArgDescriptors.push_back({blockId, nextNewArgIdx++, origIdx});
    }
  }
  // Pre-size clonedPerBlock so cloneWhileBlockChains can assign by index
  // without a placeholder. Slots for cond-used origIdx with no compOp stay
  // null and are never read (newArgDescriptors skips them).
  unsigned numOrigIds = descIdx;
  for (int blockId : blockIdsInOrder) {
    plan.clonedPerBlock[blockId].resize(numOrigIds);
  }
  return plan;
}

// Clones update chains into newAfter per (block_id, origIdx); reuses compOp/
// chainOps from before migrateBody; records into plan.clonedPerBlock.
static LogicalResult cloneWhileBlockChains(scf::WhileOp newWhileOp,
                                           WhileIterArgClonePlan &plan) {
  Block *newAfter = newWhileOp.getAfterBody();
  // Derive (blockId, origIdx) -> newArgIdx and the ordered blockId list from
  // newArgDescriptors (first-appearance order matches getBlockIdsInOrder).
  llvm::DenseMap<std::pair<int, unsigned>, unsigned> blockOrigToNewArg;
  llvm::DenseSet<int> seenBlocks;
  SmallVector<int> blockIdsInOrder;
  for (auto &desc : plan.newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    blockOrigToNewArg[{blockId, origIdx}] = newArgIdx;
    if (seenBlocks.insert(blockId).second)
      blockIdsInOrder.push_back(blockId);
  }

  // Iterate unique origIdx in first-appearance order;
  // planWhileIterArgDescriptors pushes (blockId, origIdx) in originalIndices x
  // blockIdsInOrder order.
  llvm::DenseSet<unsigned> seenOrig;
  for (auto &desc : plan.newArgDescriptors) {
    unsigned origIdx;
    std::tie(std::ignore, std::ignore, origIdx) = desc;
    if (!seenOrig.insert(origIdx).second)
      continue;
    Operation *compOp = plan.compOp.lookup(origIdx);
    if (!compOp)
      continue;
    const llvm::DenseSet<Operation *> &chainOps = plan.chainOps.lookup(origIdx);

    for (int blockId : blockIdsInOrder) {
      unsigned newArgIdx = blockOrigToNewArg.lookup({blockId, origIdx});
      Value cloned;
      if (failed(cloneUpdateChainForWhileBlock(newWhileOp, newAfter, compOp,
                                               blockId, origIdx, newArgIdx,
                                               chainOps, cloned))) {
        return failure();
      }
      plan.clonedPerBlock[blockId][plan.posInClonedVec.lookup(origIdx)] =
          cloned;
    }
  }
  return success();
}

// Per (block_id, cond-used iter_arg) pair: clone the update chain into the
// new after body, extend the scf.while, and record the new arg mapping.

// Event control is a coupled state machine: updating task may read phase,
// while updating phase may read j and the current task's bounds. Cloning each
// argument independently would mix different compute blocks' event positions.
// Collect the complete scalar update graph before changing any of its uses.
static LogicalResult
collectOuterEventControlChain(scf::ForOp loopOp, unsigned stateCount,
                              SmallVectorImpl<Operation *> &orderedOps) {
  Block *body = loopOp.getBody();
  llvm::DenseSet<Value> visited;
  llvm::DenseSet<Operation *> roots;
  SmallVector<Value> worklist(loopOp.getYieldedValues().take_front(stateCount));

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() == body && arg.getArgNumber() > stateCount)
        return loopOp->emitError("outer event control update captures a "
                                 "payload loop argument");
      continue;
    }
    Operation *def = value.getDefiningOp();
    if (!loopOp->isProperAncestor(def))
      continue;
    Operation *root = body->findAncestorOpInBlock(*def);
    if (!root)
      return failure();
    if (!roots.insert(root).second)
      continue;
    bool readsMemory;
    if (!isOuterEventScalarControl(root, readsMemory))
      return root->emitError("outer event control must have a separate scalar "
                             "update graph containing only pure operations "
                             "and metadata loads");
    llvm::append_range(worklist, root->getOperands());
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(root->getRegions(), captures);
    llvm::append_range(worklist, captures);
  }
  for (Operation &op : body->without_terminator())
    if (roots.contains(&op))
      orderedOps.push_back(&op);
  return success();
}

static bool isReusableEventControl(Operation *root, bool &readsMemory) {
  return (root->hasAttr(kOuterEventControl) || root->hasAttr(kArg) ||
          root->hasAttr(kWhileArg)) &&
         isOuterEventScalarControl(root, readsMemory);
}

// SeparateCVScope preserves the event-control template on both cores. After
// CloneOps has made each payload group independent, that template can remain
// as an otherwise empty group. Keep it as the source for state cloning, but do
// not give it an actor of its own. This first version applies only to counted
// event loops; the legacy paths and while termination contract are unchanged.
static llvm::DenseSet<int> findControlOnlyEventGroups(scf::ForOp loopOp,
                                                      ArrayRef<int> blockIds,
                                                      unsigned stateCount) {
  llvm::DenseSet<int> removable;
  if (blockIds.size() < 2 ||
      llvm::any_of(loopOp.getResults().take_front(stateCount),
                   [](Value result) { return !result.use_empty(); }))
    return removable;

  Block *body = loopOp.getBody();
  Operation *yield = body->getTerminator();
  // CloneOps can leave the original slot-selection arithmetic behind after
  // cloning it into every real consumer. A transfer_id on this dead integer
  // chain is provenance, but a transfer/dependency tag on live control must
  // still reject removal. Recognize only a same-group, wholly unused arith
  // closure; do not erase or ignore any load, region or protocol operation.
  llvm::DenseSet<Operation *> deadTransferArithmetic;
  for (Operation &op : llvm::reverse(body->without_terminator())) {
    auto blockId = getOpBlockId(&op);
    auto integer = [](Type type) { return type.isIntOrIndex(); };
    if (!blockId || !op.hasAttr(kTransferId) ||
        !isa<arith::ArithDialect>(op.getDialect()) || op.getNumRegions() ||
        !isMemoryEffectFree(&op) ||
        !llvm::all_of(op.getOperandTypes(), integer) ||
        !llvm::all_of(op.getResultTypes(), integer))
      continue;
    bool knownAttrs = llvm::all_of(op.getAttrs(), [](NamedAttribute attr) {
      StringRef name = attr.getName().getValue();
      return !name.starts_with("ssbuffer.") || name == kBlockId ||
             name == kCoreType || name == kClone || name == kTransferId;
    });
    if (knownAttrs && llvm::all_of(op.getUsers(), [&](Operation *user) {
          return getOpBlockId(user) == blockId &&
                 deadTransferArithmetic.contains(user);
        }))
      deadTransferArithmetic.insert(&op);
  }
  for (int blockId : blockIds) {
    bool safe = true, hasControlRoot = false;
    for (Operation &op : body->without_terminator()) {
      if (getOpBlockId(&op) != blockId)
        continue;
      if (deadTransferArithmetic.contains(&op))
        continue;
      hasControlRoot |= op.hasAttr(kOuterEventControl);
      bool readsMemory = false;
      if (!isOuterEventScalarControl(&op, readsMemory)) {
        safe = false;
        break;
      }
      // Reject any protocol annotation, including on nested operations. Only
      // known scalar-control provenance is allowed; unknown ssbuffer tags keep
      // the group rather than risking a dropped dependency or flag update.
      // dep_mark is diagnostic provenance written by AddMultiBufferInnerScope;
      // its generated channels use intraDeps, which remains disallowed here.
      auto protocol = op.walk([](Operation *nested) -> WalkResult {
        for (NamedAttribute attr : nested->getAttrs()) {
          StringRef name = attr.getName().getValue();
          if (name.starts_with("ssbuffer.") && name != kBlockId &&
              name != kCoreType && name != kOuterEventControl &&
              name != kClone && name != kArg && name != kWhileArg &&
              name != kDepMark)
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (protocol.wasInterrupted()) {
        safe = false;
        break;
      }
      for (Value result : op.getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (use.getOwner() == yield && use.getOperandNumber() < stateCount)
            continue;
          Operation *user = body->findAncestorOpInBlock(*use.getOwner());
          if (!user || getOpBlockId(user) != blockId) {
            safe = false;
            break;
          }
        }
        if (!safe)
          break;
      }
      if (!safe)
        break;
    }
    if (safe && hasControlRoot)
      removable.insert(blockId);
  }
  // A control-only function still needs an owner for its state transition.
  if (removable.size() == blockIds.size())
    removable.clear();
  return removable;
}

static bool isEventControlReadBarrier(Operation *root) {
  bool syncOrVolatile = false;
  root->walk([&](Operation *op) {
    syncOrVolatile |= isSyncOp(op) || op->hasAttr(kMemrefExtVolatile);
  });
  if (syncOrVolatile)
    return true;
  auto effects = getEffectsRecursively(root);
  return !effects || llvm::any_of(*effects, [](const auto &effect) {
    return !isa<MemoryEffects::Read>(effect.getEffect());
  });
}

static bool equivalentEventControl(Operation *lhs, Operation *rhs) {
  if (lhs->getName() != rhs->getName() ||
      !llvm::equal(lhs->getOperands(), rhs->getOperands()) ||
      !llvm::equal(lhs->getResultTypes(), rhs->getResultTypes()))
    return false;

  // Scheduling provenance can differ between the payload and update copies.
  // Compare detached copies so no live scheduling annotations are changed.
  auto normalizedClone = [](Operation *op) {
    Operation *clone = op->clone();
    clone->walk([](Operation *nested) {
      for (StringRef attr :
           {kBlockId, kClone, kArg, kWhileArg, kOuterEventControl})
        nested->removeAttr(attr);
    });
    return clone;
  };
  Operation *left = normalizedClone(lhs);
  Operation *right = normalizedClone(rhs);
  IRMapping values;
  bool equivalent = OperationEquivalence::isEquivalentTo(
      left, right,
      [&](Value a, Value b) {
        return success(a == b || (values.contains(a) && values.lookup(a) == b));
      },
      [&](Value a, Value b) { values.map(a, b); },
      OperationEquivalence::IgnoreLocations);
  left->destroy();
  right->destroy();
  return equivalent;
}

// Run after all block-local old-state arguments have been remapped. Reuse only
// a dominating result in the same stage; never substitute a different stage's
// cursor or a next-state value. Writes, sync and unknown effects invalidate all
// available reads, including reads nested in a scalar if.
static void deduplicateOuterEventControl(Block *body) {
  struct Available {
    Operation *op;
    int blockId;
    bool readsMemory;
  };
  SmallVector<Available> available;
  for (Operation &op : llvm::make_early_inc_range(body->without_terminator())) {
    if (isEventControlReadBarrier(&op))
      llvm::erase_if(available,
                     [](const Available &entry) { return entry.readsMemory; });
    auto blockId = getOpBlockId(&op);
    bool readsMemory = false;
    if (!blockId || !isReusableEventControl(&op, readsMemory))
      continue;
    auto same = llvm::find_if(available, [&](const Available &entry) {
      return entry.blockId == *blockId && equivalentEventControl(entry.op, &op);
    });
    if (same == available.end()) {
      available.push_back({&op, *blockId, readsMemory});
      continue;
    }
    op.replaceAllUsesWith(same->op->getResults());
    op.erase();
  }
}

static LogicalResult processOuterEventControl(scf::ForOp loopOp,
                                              ControlFlowConditionInfo *info) {
  Block *oldBody = loopOp.getBody();
  ValueRange inits = loopOp.getInitArgs();
  auto iterArgs = loopOp.getRegionIterArgs();
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  auto countAttr = loopOp->getAttrOfType<IntegerAttr>(kOuterEventControlCount);
  int64_t count = countAttr ? countAttr.getInt() : 0;
  if (count <= 0 || count > static_cast<int64_t>(inits.size()))
    return loopOp->emitError("invalid outer event control argument prefix");

  unsigned stateCount = count;
  auto stateIndices = llvm::seq<unsigned>(0, stateCount);
  if (!llvm::all_of(iterArgs.take_front(stateCount),
                    [](Value arg) { return arg.getType().isIntOrIndex(); }))
    return loopOp->emitError("outer event control arguments must be integer "
                             "or index values");

  SmallVector<int> blockIds = getBlockIdsInOrder(loopOp);
  if (blockIds.empty())
    return loopOp->emitError("outer event loop has no compute blocks");
  SmallVector<Operation *> chain;
  if (failed(collectOuterEventControlChain(loopOp, stateCount, chain)))
    return failure();

  // Other scalar carries must already belong to one block. The generic
  // shared-argument path cannot clone a mixed scalar/tensor payload if as a
  // scalar state update, nor may it reuse another block's task cursor.
  llvm::DenseMap<int, llvm::DenseSet<int>> users;
  if (failed(collectArgIndexToBlockIds(oldBody, 1, users)))
    return failure();
  for (auto &[index, ids] : users)
    if (static_cast<unsigned>(index) >= stateCount && ids.size() > 1)
      return loopOp->emitError("outer event shared scalar payload must be "
                               "rematerialized from the block's control state");

  llvm::DenseSet<int> controlOnlyGroups =
      findControlOnlyEventGroups(loopOp, blockIds, stateCount);
  llvm::erase_if(blockIds,
                 [&](int id) { return controlOnlyGroups.contains(id); });

  SmallVector<Value> extraInits;
  for (unsigned i = 0; i < blockIds.size(); ++i)
    llvm::append_range(extraInits, inits.take_front(stateCount));
  unsigned originalCount = inits.size();
  Operation *newLoop = createMainLoopOpWithExtras(loopOp, extraInits);
  Block *body = MainLoop(newLoop).getBody();
  migrateBody(oldBody, body);
  SmallVector<Value> originalYields(oldYield.getOperands());
  SmallVector<Value> extraYields;
  llvm::DenseMap<int, IRMapping> blockMappings;

  for (auto [blockPosition, blockId] : llvm::enumerate(blockIds)) {
    IRMapping &mapping = blockMappings[blockId];
    for (unsigned index : stateIndices) {
      unsigned newIndex = originalCount + blockPosition * stateCount + index;
      mapping.map(body->getArgument(index + 1),
                  body->getArgument(newIndex + 1));
    }
    OpBuilder builder(body, body->end());
    if (Operation *last = findLastOpWithBlockId(body, blockId))
      builder.setInsertionPointAfter(last);
    for (Operation *op : chain) {
      // Pass the complete mapping, including captures inside scalar scf.if
      // regions. OpBuilder::clone records every result, not just result(0).
      Operation *cloned = builder.clone(*op, mapping);
      cloned->setAttr(kBlockId, builder.getI32IntegerAttr(blockId));
      cloned->setAttr(kArg, builder.getI32IntegerAttr(0));
    }
    for (unsigned index : stateIndices)
      extraYields.push_back(mapping.lookupOrDefault(originalYields[index]));
  }

  // Payload guards, addresses and slot calculations must read their own
  // block's task/j/phase, including operands captured by nested ops.
  for (unsigned index : stateIndices) {
    Value arg = body->getArgument(index + 1);
    for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
      auto blockId = getLoopDirectChildBlockId(use.getOwner());
      if (blockId && blockMappings.count(*blockId))
        use.set(blockMappings[*blockId].lookup(arg));
    }
  }
  // Keep the original result slots coherent with the first block. They are
  // not a global scheduling cursor: conditions and buffer slots use the
  // existing per-block for counter.
  // The for IV is deliberately not part of the state group. Its uses will be
  // remapped by replaceForOpCounterInIfOps after scheduling is constructed.
  oldYield->setOperands(0, stateCount,
                        ArrayRef<Value>(extraYields).take_front(stateCount));
  if (failed(buildNewYieldOp(oldBody, body, newLoop, extraYields)))
    return failure();
  for (Operation *op : llvm::reverse(chain))
    if (isOpTriviallyDead(op))
      op->erase();
  // The original prefix now yields the first real actor's state. The strict
  // use check above guarantees that the template has no remaining observers.
  // Do not move or erase any write or synchronization from another group.
  SmallVector<Operation *> controlOnlyOps;
  for (Operation &op : body->without_terminator()) {
    auto blockId = getOpBlockId(&op);
    if (blockId && controlOnlyGroups.contains(*blockId))
      controlOnlyOps.push_back(&op);
  }
  for (Operation *op : llvm::reverse(controlOnlyOps)) {
    if (!op->use_empty())
      return op->emitError("control-only event group still has a live user");
    op->erase();
  }
  deduplicateOuterEventControl(body);
  return replaceMainLoopOpAndErase(loopOp, newLoop, info);
}

LogicalResult
ProcessArgsPass::processWhileIterArgsInWhileOp(scf::WhileOp whileOp,
                                               ControlFlowConditionInfo *info) {
  auto it = originalWhileIterArgIndices.find(whileOp);
  if (it == originalWhileIterArgIndices.end())
    return success();
  const SmallVector<unsigned> &originalIndices = it->second;
  llvm::DenseSet<unsigned> condUsed =
      collectConditionUsedIterArgIndices(whileOp, originalIndices);
  if (condUsed.empty())
    return success();

  WhileIterArgClonePlan plan =
      planWhileIterArgDescriptors(whileOp, originalIndices, condUsed);
  if (plan.newArgDescriptors.empty()) {
    return success();
  }

  // Gather origIdx from plan.newArgDescriptors in order — same shadowing rule
  // as shared-iter-args (one new init per orig to shadow), routes through
  // createMainLoopOpWithExtras.
  SmallVector<Value> extraInitArgs;
  extraInitArgs.reserve(plan.newArgDescriptors.size());
  ValueRange origInits = whileOp.getInits();
  for (auto &desc : plan.newArgDescriptors) {
    unsigned origIdx;
    std::tie(std::ignore, std::ignore, origIdx) = desc;
    extraInitArgs.push_back(origInits[origIdx]);
  }
  auto newWhileOp =
      cast<scf::WhileOp>(createMainLoopOpWithExtras(whileOp, extraInitArgs));
  migrateWhileBodies(whileOp, newWhileOp);
  if (failed(cloneWhileBlockChains(newWhileOp, plan))) {
    return failure();
  }
  buildNewWhileCondition(whileOp, newWhileOp);
  // Pre-compute extra yield values from plan.clonedPerBlock in
  // newArgDescriptors order; appended to new scf.yield operands by
  // buildNewYieldOp.
  SmallVector<Value> extraYieldValues;
  extraYieldValues.reserve(plan.newArgDescriptors.size());
  for (auto &desc : plan.newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    extraYieldValues.push_back(plan.clonedPerBlock.lookup(
        blockId)[plan.posInClonedVec.lookup(origIdx)]);
  }
  if (failed(buildNewYieldOp(whileOp.getAfterBody(), newWhileOp.getAfterBody(),
                             newWhileOp, extraYieldValues))) {
    return failure();
  }
  // Mirror (block_id, new_arg_idx) -> orig_idx into localWhileBlockArgMap and
  // info->whileBlockArgMap so downstream passes can resolve newWhileOp's args.
  for (auto &desc : plan.newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    localWhileBlockArgMap[newWhileOp][blockId][newArgIdx] = (int)origIdx;
    if (info) {
      info->whileBlockArgMap[newWhileOp][blockId][newArgIdx] = (int)origIdx;
    }
  }
  if (failed(replaceMainLoopOpAndErase(whileOp, newWhileOp, info))) {
    return failure();
  }
  return success();
}

void ProcessArgsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("before processArgs:\n" << module);

  // 1. While-specific decoupling: snapshot original iter_args; per scf.while,
  //    clone cond-used iter_arg update chains per block_id; record in
  //    info->whileBlockArgMap.
  if (failed(updateIndependentCondsInWhileBlocks(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // 2. Process shared iter_args (adds per-block clones for args shared
  //    across block_ids). Uses originalWhileIterArgIndices captured above.
  if (failed(processSharedIterArgs(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LDBG("after processArgs:\n" << module);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass() {
  return std::make_unique<ProcessArgsPass>();
}

} // namespace triton
} // namespace mlir
