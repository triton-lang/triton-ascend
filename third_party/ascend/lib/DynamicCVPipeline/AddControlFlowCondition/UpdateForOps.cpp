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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateForOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

static constexpr const char *DEBUG_TYPE = "UpdateForOps";
static constexpr int kPipeSFlagId = 15;
static constexpr const char *kSsbufferMainLoop = "ssbuffer.main_loop";
static constexpr const char *kSsbufferIf = "ssbuffer.if";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::outs() << __VA_ARGS__; \
  llvm::outs() << "\n"; \
})

using llvm::SmallVector;
using namespace mlir;
using namespace triton;
using namespace hivm;

// Replace old block arguments with new ones
static LogicalResult replaceBlockArguments(Block *oldBlock, Block *newBlock)
{
  if (!oldBlock || !newBlock) {
    LDBG("[Error]: oldBlock or newBlock is null\n");
    return failure();
  }

  unsigned totalArgs = oldBlock->getNumArguments();

  for (unsigned i = 0; i < totalArgs; ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  return success();
}

// Create new yield operands: original yield ops + extra args from new block
static SmallVector<Value> createNewYieldOperands(
    scf::YieldOp oldYield, unsigned oldNumArgs,
    Block *newBlock, int numExtraArgs)
{
  SmallVector<Value> newYieldOperands;

  for (unsigned i = 0; i < oldNumArgs; ++i) {
    newYieldOperands.push_back(oldYield.getOperand(i));
  }

  for (int i = 0; i < numExtraArgs; ++i) {
    newYieldOperands.push_back(newBlock->getArgument(1 + oldNumArgs + i));
  }

  return newYieldOperands;
}

// Derive block counters from ssbuffer.if attributes when info is not pre-populated.
// `op` is the main-loop op (scf.for or scf.while) carrying ssbuffer.main_loop.
// blockCounterNums is keyed on Operation*, so we record the count for both
// scf.for and scf.while main-loop ops. UpdateForOps's extend* helpers consume
// this map for iter_arg extension.
LogicalResult UpdateForOpsPass::deriveBlockCountersFromIfOps(ModuleOp module, ControlFlowConditionInfo *info)
{
  if (!info) {
    LDBG("[Error]: info is null\n");
    return failure();
  }

  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    if (!isa<scf::ForOp>(op) && !isa<scf::WhileOp>(op)) {
      LDBG("[Error]: op with ssbuffer.main_loop is neither scf::ForOp nor scf::WhileOp\n");
      return WalkResult::interrupt();
    }

    int numIfBlockIds = countUniqueIfBlockIds(op);
    if (numIfBlockIds > 0) {
      info->blockCounterNums[op] = numIfBlockIds;
    }
    return WalkResult::advance();
  });

  return success();
}

// Create new for op with extra iter args and migrate body
static scf::ForOp createForOpAndMigrateBody(
    scf::ForOp oldForOp, int numExtraArgs,
    const SmallVector<Value> &extraInitArgs)
{
  if (numExtraArgs < 0) {
    LDBG("[Error]: invalid numExtraArgs " << numExtraArgs << "\n");
    return scf::ForOp();
  }
  if (numExtraArgs == 0)
    return oldForOp;
  if ((int)extraInitArgs.size() != numExtraArgs) {
    LDBG("[Error]: extraInitArgs size " << extraInitArgs.size() << " != numExtraArgs " << numExtraArgs << "\n");
    return scf::ForOp();
  }

  OpBuilder builder(oldForOp);
  // Create new for op with extra iter args
  SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(),
                                 oldForOp.getInitArgs().end());
  llvm::append_range(newInitArgs, extraInitArgs);

  scf::ForOp newForOp = builder.create<scf::ForOp>(
      oldForOp.getLoc(), oldForOp.getLowerBound(), oldForOp.getUpperBound(),
      oldForOp.getStep(), newInitArgs);

  for (auto &attr : oldForOp->getAttrs())
    newForOp->setAttr(attr.getName(), attr.getValue());

  // Migrate body
  Block *oldBlock = oldForOp.getBody();
  Block *newBlock = newForOp.getBody();

  if (failed(replaceBlockArguments(oldBlock, newBlock))) {
    newForOp.erase();
    return scf::ForOp();
  }

  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator()))
    op.moveBefore(newBlock, newBlock->end());

  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> newYieldOperands = createNewYieldOperands(
      oldYield, oldForOp.getNumRegionIterArgs(), newBlock, numExtraArgs);

  builder.setInsertionPointToEnd(newBlock);
  builder.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
  oldYield.erase();

  return newForOp;
}

// Splice the new main-loop op (scf.for or scf.while) in place of the old one:
// redirect external uses to the new results and erase the old op. Result types
// are checked before the splice so a mismatch surfaces here rather than as a
// silent type change downstream.
static LogicalResult replaceMainLoopOpUsesAndErase(Operation *oldOp, Operation *newOp)
{
  if (oldOp->getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < oldOp->getNumResults(); ++i) {
      if (oldOp->getResult(i).getType() != newOp->getResult(i).getType()) {
        LDBG("[Error]: main_loop op result type mismatch at index " << i << "\n");
        return failure();
      }
      newResults.push_back(newOp->getResult(i));
    }
    oldOp->replaceAllUsesWith(newResults);
  }

  oldOp->erase();
  return success();
}

// Creates a new scf::WhileOp whose init args / result types are the originals
// plus the supplied extra init args. Empty before/after blocks (with arg types
// matching the new inits) are added to the new op; bodies are populated by the
// migrate steps. Attributes are copied from oldWhileOp.
static scf::WhileOp createNewWhileOpWithExtras(
    scf::WhileOp oldWhileOp, const SmallVector<Value> &extraInitArgs)
{
  OpBuilder builder(oldWhileOp);

  SmallVector<Value> newInits(oldWhileOp.getInits().begin(),
                              oldWhileOp.getInits().end());
  llvm::append_range(newInits, extraInitArgs);

  SmallVector<Type> newResultTypes(oldWhileOp->getResultTypes().begin(),
                                   oldWhileOp->getResultTypes().end());
  for (Value v : extraInitArgs) {
    newResultTypes.push_back(v.getType());
  }

  scf::WhileOp newWhileOp =
      builder.create<scf::WhileOp>(oldWhileOp.getLoc(), newResultTypes, newInits);

  for (auto &attr : oldWhileOp->getAttrs()) {
    newWhileOp->setAttr(attr.getName(), attr.getValue());
  }

  SmallVector<Type> argTypes;
  argTypes.reserve(newInits.size());
  for (Value v : newInits) {
    argTypes.push_back(v.getType());
  }
  SmallVector<Location> argLocs(newInits.size(), oldWhileOp.getLoc());

  builder.createBlock(&newWhileOp.getBefore(), /*insertBefore=*/{}, argTypes,
                      argLocs);
  builder.createBlock(&newWhileOp.getAfter(), /*insertBefore=*/{}, argTypes,
                      argLocs);

  return newWhileOp;
}

// Append the new iter_args (new before-block args at positions
// [numOriginal..numOriginal+numExtra-1]) to the scf.condition's forwarded-value
// list.
static void extendWhileCondition(scf::ConditionOp oldCond, Block *newBefore,
                                 unsigned numOriginalIterArgs, unsigned numExtraArgs)
{
  SmallVector<Value> newCondValues(oldCond.getArgs().begin(),
                                   oldCond.getArgs().end());
  for (unsigned i = 0; i < numExtraArgs; ++i) {
    newCondValues.push_back(newBefore->getArgument(numOriginalIterArgs + i));
  }
  oldCond.getArgsMutable().assign(newCondValues);
}

// Append the new iter_args (new after-block args at positions
// [numOriginal..numOriginal+numExtra-1]) to the scf.yield's operand list.
// Block counters are carry-only, so each new iter_arg just forwards itself.
static void extendWhileYield(scf::YieldOp oldYield, Block *newAfter,
                             unsigned numOriginalIterArgs, unsigned numExtraArgs)
{
  SmallVector<Value> newYieldOperands(oldYield.getOperands().begin(),
                                      oldYield.getOperands().end());
  for (unsigned i = 0; i < numExtraArgs; ++i) {
    newYieldOperands.push_back(newAfter->getArgument(numOriginalIterArgs + i));
  }
  oldYield.getResultsMutable().assign(newYieldOperands);
}

// Migrates the before region of a whileOp: rewire old before-block args to
// the new before-block args, move body ops, then move and extend the
// scf.condition terminator.
static LogicalResult migrateWhileBeforeRegion(scf::WhileOp oldWhileOp,
                                              scf::WhileOp newWhileOp,
                                              unsigned numOriginalIterArgs,
                                              unsigned numExtraArgs)
{
  Block *oldBefore = oldWhileOp.getBeforeBody();
  Block *newBefore = newWhileOp.getBeforeBody();
  if (failed(replaceBlockArguments(oldBefore, newBefore))) {
    return failure();
  }
  for (Operation &op : llvm::make_early_inc_range(oldBefore->without_terminator()))
    op.moveBefore(newBefore, newBefore->end());
  auto oldCond = cast<scf::ConditionOp>(oldBefore->getTerminator());
  oldCond->moveBefore(newBefore, newBefore->end());
  extendWhileCondition(oldCond, newBefore, numOriginalIterArgs, numExtraArgs);
  return success();
}

// Migrates the after region of a whileOp: rewire old after-block args to the
// new after-block args, move body ops, then move and extend the scf.yield
// terminator.
static LogicalResult migrateWhileAfterRegion(scf::WhileOp oldWhileOp,
                                             scf::WhileOp newWhileOp,
                                             unsigned numOriginalIterArgs,
                                             unsigned numExtraArgs)
{
  Block *oldAfter = oldWhileOp.getAfterBody();
  Block *newAfter = newWhileOp.getAfterBody();
  if (failed(replaceBlockArguments(oldAfter, newAfter))) {
    return failure();
  }
  for (Operation &op : llvm::make_early_inc_range(oldAfter->without_terminator()))
    op.moveBefore(newAfter, newAfter->end());
  auto oldYield = cast<scf::YieldOp>(oldAfter->getTerminator());
  oldYield->moveBefore(newAfter, newAfter->end());
  extendWhileYield(oldYield, newAfter, numOriginalIterArgs, numExtraArgs);
  return success();
}

// Creates a new scf::WhileOp with extra init args and migrates the body from
// oldWhileOp. The before-region terminator (scf.condition) and after-region
// terminator (scf.yield) are updated to forward the new iter_args.
//
// Notes on whileOp structure:
//   - scf.while has no induction variable; before/after block args start at 0
//     with iter_args (whereas scf.for has its IV at block arg 0 and iter_args
//     start at 1).
//   - The new iter_args are appended after the originals in both the before and
//     after block arg lists, mirroring forOp's layout (modulo the IV offset).
static scf::WhileOp createWhileOpAndMigrateBody(
    scf::WhileOp oldWhileOp, int numExtraArgs,
    const SmallVector<Value> &extraInitArgs)
{
  if (numExtraArgs < 0) {
    LDBG("[Error]: invalid numExtraArgs " << numExtraArgs << "\n");
    return scf::WhileOp();
  }
  if (numExtraArgs == 0) {
    return oldWhileOp;
  }
  if ((int)extraInitArgs.size() != numExtraArgs) {
    LDBG("[Error]: extraInitArgs size " << extraInitArgs.size()
          << " != numExtraArgs " << numExtraArgs << "\n");
    return scf::WhileOp();
  }

  scf::WhileOp newWhileOp = createNewWhileOpWithExtras(oldWhileOp, extraInitArgs);

  unsigned numOriginalIterArgs = oldWhileOp.getInits().size();
  if (failed(migrateWhileBeforeRegion(oldWhileOp, newWhileOp,
                                      numOriginalIterArgs, numExtraArgs))) {
    newWhileOp.erase();
    return scf::WhileOp();
  }
  if (failed(migrateWhileAfterRegion(oldWhileOp, newWhileOp,
                                     numOriginalIterArgs, numExtraArgs))) {
    newWhileOp.erase();
    return scf::WhileOp();
  }

  return newWhileOp;
}

// Extends a scf::WhileOp with iter_args for block counters, inner dependency
// conditions, and tensor iter_args — mirroring extendForOpWithExtraArgs so
// the whileOp path goes through the same info map writes as forOp.
//
// Block counter init is 0 (whileOp has no lower bound; 0 is the natural "no
// block executed yet" sentinel and matches the existing UpdateConditionInfo
// expectations). Inner-dep-conds and tensor iter_args are seeded from
// intraCoreDependentMap / tensorIterArgDepsMap and mirrored under newWhileOp.
LogicalResult extendWhileOpWithExtraArgs(scf::WhileOp oldWhileOp, ControlFlowConditionInfo *info)
{
  int numBlockCounters = info->blockCounterNums[oldWhileOp];
  int numInnerDepConds = info->intraCoreDependentMap[oldWhileOp].size();
  int totalExtraArgs = numBlockCounters + numInnerDepConds;

  int numTensorIterArgs = 0;
  llvm::DenseMap<Value, int> tensorIterArgNumConsumers;
  llvm::SmallVector<TensorIterArgIfOpRelation> depsVecCopy;
  auto tensorIterArgDepsIt = info->tensorIterArgDepsMap.find(oldWhileOp);
  if (tensorIterArgDepsIt != info->tensorIterArgDepsMap.end()) {
    depsVecCopy = tensorIterArgDepsIt->second;
    for (auto &entry : depsVecCopy) {
      Value iterArg = entry.iterArg;
      int numConsumers = entry.consumers.size();
      tensorIterArgNumConsumers[iterArg] = numConsumers;
      numTensorIterArgs += numConsumers;
    }
  }

  totalExtraArgs += numTensorIterArgs;
  if (totalExtraArgs == 0) {
    return success();
  }

  OpBuilder builder(oldWhileOp);
  SmallVector<Value> extraInitArgs;
  for (int i = 0; i < numBlockCounters; ++i) {
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        oldWhileOp.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0)));
  }
  for (int i = 0; i < numInnerDepConds; ++i) {
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        oldWhileOp.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0)));
  }
  for (int i = 0; i < numTensorIterArgs; ++i) {
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        oldWhileOp.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1)));
  }

  scf::WhileOp newWhileOp = createWhileOpAndMigrateBody(oldWhileOp, totalExtraArgs, extraInitArgs);
  if (!newWhileOp) {
    return failure();
  }

  // whileOp has no IV, so iter_args start at index 0 in the before/after
  // block arg lists (whereas forOp's IV sits at index 0 and iter_args start
  // at 1). baseIdx=0 here mirrors that.
  unsigned baseIdx = 0;
  if (numBlockCounters > 0) {
    SmallVector<int> indices;
    for (int j = 0; j < numBlockCounters; ++j)
      indices.push_back(baseIdx + j);
    info->blockCounters.erase(oldWhileOp);
    info->blockCounters[newWhileOp] = indices;
  }

  if (numInnerDepConds > 0) {
    SmallVector<int> indices;
    for (int j = 0; j < numInnerDepConds; ++j)
      indices.push_back(baseIdx + numBlockCounters + j);
    info->innerDepConds.erase(oldWhileOp);
    info->innerDepConds[newWhileOp] = indices;
  }

  if (numTensorIterArgs > 0) {
    unsigned tensorBaseIdx = baseIdx + numBlockCounters + numInnerDepConds;
    auto &newIndicesMap = info->tensorIterArgIndicesMap[newWhileOp];

    unsigned currentIdx = tensorBaseIdx;
    for (auto &entry : depsVecCopy) {
      Value iterArg = entry.iterArg;
      int numConsumers = entry.consumers.size();
      SmallVector<int> indices;
      for (int j = 0; j < numConsumers; ++j) {
        indices.push_back(currentIdx++);
      }
      newIndicesMap[iterArg] = indices;
    }

    info->tensorIterArgIndicesMap.erase(oldWhileOp);
    info->tensorIterArgDepsMap[newWhileOp] = std::move(depsVecCopy);
    info->tensorIterArgDepsMap.erase(oldWhileOp);
  }

  if (info->intraCoreDependentMap.count(oldWhileOp)) {
    info->intraCoreDependentMap[newWhileOp] = info->intraCoreDependentMap[oldWhileOp];
    info->intraCoreDependentMap.erase(oldWhileOp);
  }

  return replaceMainLoopOpUsesAndErase(oldWhileOp, newWhileOp);
}

LogicalResult extendForOpWithExtraArgs(scf::ForOp oldForOp, ControlFlowConditionInfo *info)
{
  int numBlockCounters = info->blockCounterNums[oldForOp];
  int numInnerDepConds = info->intraCoreDependentMap[oldForOp].size();
  int totalExtraArgs = numBlockCounters + numInnerDepConds;

  int numTensorIterArgs = 0;
  // Record the number of consumers for each tensor iter_args (the number of parameters to be created)
  llvm::DenseMap<Value, int> tensorIterArgNumConsumers;
  // First, copy the depsVec out to avoid iterator invalidation later
  llvm::SmallVector<TensorIterArgIfOpRelation> depsVecCopy;
  auto tensorIterArgDepsIt = info->tensorIterArgDepsMap.find(oldForOp);
  if (tensorIterArgDepsIt != info->tensorIterArgDepsMap.end()) {
    depsVecCopy = tensorIterArgDepsIt->second;  // Make a copy
    for (auto &entry : depsVecCopy) {
      Value iterArg = entry.iterArg;
      int numConsumers = entry.consumers.size();
      tensorIterArgNumConsumers[iterArg] = numConsumers;
      numTensorIterArgs += numConsumers;
    }
  }
  
  totalExtraArgs += numTensorIterArgs;
  if (totalExtraArgs == 0) {
    return success();
  }

  OpBuilder builder(oldForOp);
  SmallVector<Value> extraInitArgs;
  for (int i = 0; i < numBlockCounters; ++i)
    extraInitArgs.push_back(oldForOp.getLowerBound());
  for (int i = 0; i < numInnerDepConds; ++i)
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        oldForOp.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0)));
  // Add an initial value (1) for the new parameter iter_arg of tensor
  for (int i = 0; i < numTensorIterArgs; ++i)
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        oldForOp.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1)));

  scf::ForOp newForOp = createForOpAndMigrateBody(oldForOp, totalExtraArgs, extraInitArgs);
  if (!newForOp) {
    return failure();
  }

  unsigned baseIdx = oldForOp.getNumRegionIterArgs();
  if (numBlockCounters > 0) {
    SmallVector<int> indices;
    for (int j = 0; j < numBlockCounters; ++j)
      indices.push_back(baseIdx + j);
    info->blockCounters.erase(oldForOp);
    info->blockCounters[newForOp] = indices;
  }

  if (numInnerDepConds > 0) {
    SmallVector<int> indices;
    for (int j = 0; j < numInnerDepConds; ++j)
      indices.push_back(baseIdx + numBlockCounters + j);
    info->innerDepConds.erase(oldForOp);
    info->innerDepConds[newForOp] = indices;
  }

  // Record the index of the new parameter iter_arg for the tensor and update the corresponding map
  if (numTensorIterArgs > 0) {
    unsigned tensorBaseIdx = baseIdx + numBlockCounters + numInnerDepConds;
    auto &newIndicesMap = info->tensorIterArgIndicesMap[newForOp];
    
    unsigned currentIdx = tensorBaseIdx;
    for (auto &entry : depsVecCopy) {
      Value iterArg = entry.iterArg;
      int numConsumers = entry.consumers.size();
      SmallVector<int> indices;
      for (int j = 0; j < numConsumers; ++j) {
        indices.push_back(currentIdx++);
      }
      newIndicesMap[iterArg] = indices;
    }
    
    info->tensorIterArgIndicesMap.erase(oldForOp);
    info->tensorIterArgDepsMap[newForOp] = std::move(depsVecCopy);
    info->tensorIterArgDepsMap.erase(oldForOp);
  }

  if (info->intraCoreDependentMap.count(oldForOp)) {
    info->intraCoreDependentMap[newForOp] = info->intraCoreDependentMap[oldForOp];
    info->intraCoreDependentMap.erase(oldForOp);
  }

  return replaceMainLoopOpUsesAndErase(oldForOp, newForOp);
}

// Add block counter and inner dependency condition iter args to for ops
// (and whileOps — see below).
//
// Main-loop ops are processed from the info-driven `mainLoopOpsToProcess`
// set, built from `blockCounterNums` and `tensorIterArgDepsMap`. Both forOp
// and whileOp entries participate: CreateIfOps / InitDependentMap / ProcessArgs
// populate these maps for both op kinds (the maps are keyed on Operation*),
// so we don't need a separate walk for whileOps.
LogicalResult UpdateForOpsPass::addBlockCountersAndInnerDepConds(ModuleOp module, ControlFlowConditionInfo *info)
{
  llvm::DenseSet<Operation *> mainLoopOpsToProcess;

  for (auto &p : info->blockCounterNums) {
    if (p.second < 0) {
      LDBG("[Error]: invalid blockCounterNum " << p.second << "\n");
      return failure();
    }
    mainLoopOpsToProcess.insert(p.first);
  }
  for (auto &p : info->tensorIterArgDepsMap) {
    mainLoopOpsToProcess.insert(p.first);
  }

  for (Operation *loopOp : mainLoopOpsToProcess) {
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
      if (failed(extendForOpWithExtraArgs(forOp, info)))
        return failure();
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
      if (failed(extendWhileOpWithExtraArgs(whileOp, info)))
        return failure();
    } else {
      LDBG("[Error]: main_loop op is neither scf::ForOp nor scf::WhileOp\n");
      return failure();
    }
  }

  return success();
}

// Insert sync ops inside a main-loop body (scf.for or scf.while after-region):
// wait at start, set before yield. Body block is resolved by the caller via
// getMainLoopBody.
static LogicalResult insertSyncOpsInsideMainLoop(Block *loopBody, Location loc,
                                               hivm::TCoreTypeAttr coreType,
                                               PipeAttr setPipe, PipeAttr waitPipe,
                                               int waitFlagId, int setFlagId)
{
  Operation *forTerminator = loopBody->getTerminator();
  if (!forTerminator) {
    return failure();
  }

  // Insert wait at loop body start
  OpBuilder insertionBuilder(&loopBody->front());
  auto waitFlagAttr = insertionBuilder.getIntegerAttr(insertionBuilder.getI64Type(), waitFlagId);
  insertionBuilder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe, waitFlagAttr);

  // Insert set before yield
  OpBuilder setBuilder(forTerminator);
  auto setFlagAttr = setBuilder.getIntegerAttr(setBuilder.getI64Type(), setFlagId);
  setBuilder.setInsertionPoint(forTerminator);
  setBuilder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe, setFlagAttr);

  return success();
}

// Insert a sync op (SET before, or WAIT after) outside a main-loop op
// (scf.for or scf.while). OpBuilder(Operation*) inserts at the end of the
// parent block before the op, and setInsertionPointAfter(Operation*) inserts
// after the op — both APIs take Operation*, so this function is op-agnostic.
// `isBefore` refers to the insertion point relative to the loop op (true →
// before → SET, false → after → WAIT), not the semantic of the sync op.
static LogicalResult insertSyncOpsOutsideMainLoop(Operation *loopOp, Location loc,
                                             hivm::TCoreTypeAttr coreType,
                                             PipeAttr setPipe, PipeAttr waitPipe,
                                             int flagId, bool isBefore)
{
  OpBuilder builder(loopOp);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);
  if (isBefore) {
    builder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe, flagAttr);
  } else {
    builder.setInsertionPointAfter(loopOp);
    builder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe, flagAttr);
  }
  return success();
}

// Returns the loop body block for a main-loop op (scf.for or scf.while carrying
// ssbuffer.main_loop). For scf.for this is the single region; for scf.while
// this is the after region (where the actual loop body lives).
static Block *getMainLoopBody(Operation *loopOp)
{
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    return &whileOp.getAfter().front();
  }
  return &loopOp->getRegion(0).front();
}

// Insert PIPE_S for a main_loop op (scf.for or scf.while) based on loop type
// and scope type. Body block is resolved via getMainLoopBody — for scf.while
// we use the after region block, which is where the actual loop body lives.
static LogicalResult insertPipeSForMainLoopOp(Operation *loopOp, scope::ScopeOp scopeOp,
                                                  bool isScopeCube, bool isScopeVector,
                                                  PipeAttr setPipe, PipeAttr waitPipe,
                                                  int flagId)
{
  Block *loopBody = getMainLoopBody(loopOp);
  Location loc = loopOp->getLoc();
  bool isVectorFirst = loopOp->hasAttr("ssbuffer.vector_first");
  auto cubeType = hivm::TCoreTypeAttr::get(loopOp->getContext(), hivm::TCoreType::CUBE);
  auto vectorType = hivm::TCoreTypeAttr::get(loopOp->getContext(), hivm::TCoreType::VECTOR);

  if (isVectorFirst) {
    if (isScopeCube) {
      // vector_first + CUBE: before loop op (SET), inside (WAIT/SET)
      if (failed(insertSyncOpsOutsideMainLoop(loopOp, loc, cubeType, setPipe, waitPipe, flagId, true))) {
        return failure();
      }
      if (failed(insertSyncOpsInsideMainLoop(loopBody, loc, cubeType, setPipe, waitPipe, flagId, flagId))) {
        return failure();
      }
    } else if (isScopeVector) {
      // vector_first + VECTOR: inside (WAIT/SET), after loop op (WAIT)
      if (failed(insertSyncOpsInsideMainLoop(loopBody, loc, vectorType, setPipe, waitPipe, flagId, flagId))) {
        return failure();
      }
      if (failed(insertSyncOpsOutsideMainLoop(loopOp, loc, vectorType, setPipe, waitPipe, flagId, false))) {
        return failure();
      }
    }
  } else {
    // cube_first (including default when neither attribute is present)
    if (isScopeCube) {
      // cube_first + CUBE: inside (WAIT/SET), after loop op (WAIT)
      if (failed(insertSyncOpsInsideMainLoop(loopBody, loc, cubeType, setPipe, waitPipe, flagId, flagId))) {
        return failure();
      }
      if (failed(insertSyncOpsOutsideMainLoop(loopOp, loc, cubeType, setPipe, waitPipe, flagId, false))) {
        return failure();
      }
    } else if (isScopeVector) {
      // cube_first + VECTOR: before loop op (SET), inside (WAIT/SET)
      if (failed(insertSyncOpsOutsideMainLoop(loopOp, loc, vectorType, setPipe, waitPipe, flagId, true))) {
        return failure();
      }
      if (failed(insertSyncOpsInsideMainLoop(loopBody, loc, vectorType, setPipe, waitPipe, flagId, flagId))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult UpdateForOpsPass::insertInterCorePipeS(ModuleOp module)
{
  auto cubeCoreType = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  auto vectorCoreType = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto setPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  auto waitPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);

  WalkResult result = module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto scopeTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!scopeTypeAttr) {
      return WalkResult::advance();
    }

    bool isScopeCube = (scopeTypeAttr == cubeCoreType);
    bool isScopeVector = (scopeTypeAttr == vectorCoreType);

    // Walk both scf.for and scf.while carrying ssbuffer.main_loop. Both need
    // the same PIPE_S inter-core sync insertion; insertPipeSForMainLoopOp
    // dispatches internally on op type via getMainLoopBody.
    WalkResult innerResult = scopeOp.walk([&](Operation *op) -> WalkResult {
      if (!op->hasAttr("ssbuffer.main_loop")) {
        return WalkResult::advance();
      }
      if (!isa<scf::ForOp>(op) && !isa<scf::WhileOp>(op)) {
        return WalkResult::advance();
      }
      if (failed(insertPipeSForMainLoopOp(op, scopeOp, isScopeCube, isScopeVector,
                                          setPipeType, waitPipeType, kPipeSFlagId))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (innerResult.wasInterrupted()) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

// Analyze the producer/consumer relationship between the tensor type iter_args
// in the main_loop and ssbuffer.if. Supports both scf.for and scf.while
// (both carry ssbuffer.main_loop). tensorIterArgDepsMap is keyed on
// Operation*, so we record the dependency for both op types. Downstream
// consumers (extendForOpWithExtraArgs) currently only act on forOp — the
// whileOp path is recorded for future use but not yet consumed.
LogicalResult UpdateForOpsPass::analyzeTensorIterArgDependencies(ModuleOp module, ControlFlowConditionInfo *info)
{
  bool failed = false;
  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(kSsbufferMainLoop)) {
      return WalkResult::advance();
    }
    if (!isa<scf::ForOp>(op) && !isa<scf::WhileOp>(op)) {
      LDBG("[Error]: op with ssbuffer.main_loop is neither scf::ForOp nor scf::WhileOp\n");
      failed = true;
      return WalkResult::interrupt();
    }

    LDBG("Analyzing main_loop op: " << op << "\n");

    // Get iter_args to analyze. For scf.for this is getRegionIterArgs() (the
    // body block args minus the IV); for scf.while this is getAfterArguments()
    // (the after-block args, which carry the iter_args visible inside the
    // body — the before-block args are not visible here).
    ValueRange iterArgsRange;
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      iterArgsRange = forOp.getRegionIterArgs();
    } else {
      iterArgsRange = cast<scf::WhileOp>(op).getAfterArguments();
    }
    SmallVector<Value> iterArgsVec(iterArgsRange.begin(), iterArgsRange.end());

    for (auto iterArg : iterArgsVec) {
      if (!mlir::isa<TensorType>(iterArg.getType())) {
        continue;
      }

      LDBG("Found tensor type iter_arg: " << iterArg << "\n");
      scf::IfOp producerIfOp = nullptr;
      llvm::SmallVector<scf::IfOp> consumerIfOps;

      for (auto &use : iterArg.getUses()) {
        Operation *user = use.getOwner();
        scf::IfOp ifOp = nullptr;
        Operation *curr = user;
        while (curr && curr != op) {
          if (auto currIf = dyn_cast<scf::IfOp>(curr)) {
            if (currIf->hasAttr(kSsbufferIf)) {
              ifOp = currIf;
              break;
            }
          }
          curr = curr->getParentOp();
        }

        if (!ifOp) {
          LDBG("Use of tensor iter_arg " << iterArg << " is not inside any ssbuffer.if op." << "\n");
          continue;
        }

        bool isProducer = false;
        if (isa<scf::YieldOp>(user)) {
          auto yieldOp = cast<scf::YieldOp>(user);
          Operation *parentOp = yieldOp->getParentOp();
          while (parentOp && parentOp != ifOp.getOperation()) {
            parentOp = parentOp->getParentOp();
          }
          if (parentOp == ifOp.getOperation()) {
            isProducer = true;
          }
        }

        // Check and update status of ifOp
        if (isProducer) {
          if (producerIfOp && producerIfOp != ifOp) {
            // Found a different producer ifOp! This is an error.
            LDBG("[Error]: tensor iter_arg " << iterArg << " has multiple different producers!\n");
            LDBG("Existing producer: " << producerIfOp << "\n");
            LDBG("New producer: " << ifOp << "\n");
            failed = true;
            return WalkResult::interrupt();
          }
          if (!producerIfOp) {
            // First producer, or upgrade from consumer
            auto it = llvm::find(consumerIfOps, ifOp);
            if (it != consumerIfOps.end()) {
              consumerIfOps.erase(it);
              LDBG("This ifOp was consumer, now updated to producer: " << ifOp << "\n");
            } else {
              LDBG("Found producer ifOp (first time): " << ifOp << "\n");
            }
            producerIfOp = ifOp;
          }
          // Else: already is this producer, do nothing
        } else {
          // isConsumer
          if (producerIfOp == ifOp) {
            // Already a producer, even if current use is consumer, do nothing
            continue;
          }
          if (!llvm::is_contained(consumerIfOps, ifOp)) {
            consumerIfOps.push_back(ifOp);
            LDBG("Found consumer ifOp (first time): " << ifOp << "\n");
          }
          // Else: already a consumer, do nothing
        }
      }
      // Check: must have both producers AND consumers
      if (!producerIfOp || consumerIfOps.empty()) {
        LDBG("tensor iter_arg " << iterArg << " has only "
                                           << (!producerIfOp ? "consumers" : "producers") << ", skipped\n");
        continue;
      }
      TensorIterArgIfOpRelation relation;
      relation.iterArg = iterArg;
      relation.producer = producerIfOp;
      relation.consumers = consumerIfOps;

      // Only record for main-loop ops (scf.for or scf.while). The map is keyed
      // on Operation* now, so both op types participate.
      if (isa<scf::ForOp>(op) || isa<scf::WhileOp>(op)) {
        info->tensorIterArgDepsMap[op].push_back(relation);
      }
      LDBG("Recorded tensor iter_arg dependency: " << iterArg << " has 1 producer, "
                                                   << relation.consumers.size() << " consumers\n");
    }

    return WalkResult::advance();
  });

  return failed ? failure() : success();
}

void UpdateForOpsPass::runOnOperation() {
  ModuleOp module = getOperation();

  LDBG("before updateForOps:\n" << module << "\n");

  // Use provided info, or create a local one if not available
  ControlFlowConditionInfo localInfo;
  ControlFlowConditionInfo *infoToUse = info ? info : &localInfo;

  // Analyze the dependencies of the tensor type iter_args in the main_loop with the ssbuffer.if ops
  if (failed(analyzeTensorIterArgDependencies(module, infoToUse))) {
    signalPassFailure();
    return;
  }

  // Derive block counters from ssbuffer.if if blockCounterNums is empty
  if (infoToUse->blockCounterNums.empty()) {
    if (failed(deriveBlockCountersFromIfOps(module, infoToUse))) {
      signalPassFailure();
      return;
    }
  }

  // Update for/while ops iter_args for block counters and inner dependency
  // conditions. Always invoked: forOp is processed from info-driven sets
  // (no-ops when those sets are empty), whileOp is found by walking the module
  // directly (info->blockCounterNums still only holds forOp entries, so we
  // can't gate on it).
  if (infoToUse && (failed(addBlockCountersAndInnerDepConds(module, infoToUse)))) {
    signalPassFailure();
    return;
  }

  // Insert PIPE_S inter-core synchronization
  if (failed(insertInterCorePipeS(module))) {
    signalPassFailure();
    return;
  }

  LDBG("after updateForOps:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUpdateForOpsPass()
{
  return std::make_unique<UpdateForOpsPass>();
}

} // namespace triton
} // namespace mlir
