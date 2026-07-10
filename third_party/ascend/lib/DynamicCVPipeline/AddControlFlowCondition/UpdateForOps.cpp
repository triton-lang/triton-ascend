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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "UpdateForOps";
static constexpr int kPipeSFlagId = 15;
static constexpr const char *kSsbufferMainLoop = "ssbuffer.main_loop";
static constexpr const char *kSsbufferIf = "ssbuffer.if";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::outs() << __VA_ARGS__;                                               \
    llvm::outs() << "\n";                                                      \
  })

using llvm::SmallVector;
using namespace mlir;
using namespace triton;
using namespace hivm;

// Replace old block arguments with new ones
static LogicalResult replaceBlockArguments(Block *oldBlock, Block *newBlock) {
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

// Collect forOps that need processing based on info
static SmallVector<scf::ForOp>
collectForOpsToProcess(ModuleOp module,
                       const llvm::DenseMap<scf::ForOp, int> &numInfo) {
  SmallVector<scf::ForOp> forOps;

  module.walk([&](scf::ForOp forOp) {
    if (numInfo.count(forOp)) {
      forOps.push_back(forOp);
    }
  });

  return forOps;
}

// Create new yield operands: original yield ops + extra args from new block
static SmallVector<Value> createNewYieldOperands(scf::YieldOp oldYield,
                                                 unsigned oldNumArgs,
                                                 Block *newBlock,
                                                 int numExtraArgs) {
  SmallVector<Value> newYieldOperands;

  for (unsigned i = 0; i < oldNumArgs; ++i) {
    newYieldOperands.push_back(oldYield.getOperand(i));
  }

  for (int i = 0; i < numExtraArgs; ++i) {
    newYieldOperands.push_back(newBlock->getArgument(1 + oldNumArgs + i));
  }

  return newYieldOperands;
}

// Derive block counters from ssbuffer.if attributes when info is not
// pre-populated
LogicalResult
UpdateForOpsPass::deriveBlockCountersFromIfOps(ModuleOp module,
                                               ControlFlowConditionInfo *info) {
  if (!info) {
    LDBG("[Error]: info is null\n");
    return failure();
  }

  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }

    // Collect unique ssbuffer.if values in this for op's body
    llvm::DenseSet<int> ifBlockIds;

    forOp.walk([&](Operation *innerOp) {
      if (auto ifAttr = innerOp->getAttrOfType<IntegerAttr>("ssbuffer.if")) {
        ifBlockIds.insert(ifAttr.getInt());
      }
    });

    if (!ifBlockIds.empty()) {
      info->blockCounterNums[forOp] = ifBlockIds.size();
    }
    return WalkResult::advance();
  });

  return success();
}

// Create new for op with extra iter args and migrate body
static scf::ForOp
createForOpAndMigrateBody(scf::ForOp oldForOp, int numExtraArgs,
                          const SmallVector<Value> &extraInitArgs) {
  if (numExtraArgs < 0) {
    LDBG("[Error]: invalid numExtraArgs " << numExtraArgs << "\n");
    return scf::ForOp();
  }
  if (numExtraArgs == 0)
    return oldForOp;
  if ((int)extraInitArgs.size() != numExtraArgs) {
    LDBG("[Error]: extraInitArgs size " << extraInitArgs.size()
                                        << " != numExtraArgs " << numExtraArgs
                                        << "\n");
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

  for (Operation &op :
       llvm::make_early_inc_range(oldBlock->without_terminator()))
    op.moveBefore(newBlock, newBlock->end());

  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> newYieldOperands = createNewYieldOperands(
      oldYield, oldForOp.getNumRegionIterArgs(), newBlock, numExtraArgs);

  builder.setInsertionPointToEnd(newBlock);
  builder.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
  oldYield.erase();

  return newForOp;
}

static LogicalResult replaceForOpUsesAndErase(scf::ForOp oldForOp,
                                              scf::ForOp newForOp) {
  if (oldForOp.getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < oldForOp.getNumResults(); ++i) {
      if (oldForOp.getResult(i).getType() != newForOp.getResult(i).getType()) {
        LDBG("[Error]: result type mismatch at index " << i << "\n");
        return failure();
      }
      newResults.push_back(newForOp.getResult(i));
    }
    oldForOp.replaceAllUsesWith(newResults);
  }

  oldForOp.erase();
  return success();
}

LogicalResult extendForOpWithExtraArgs(scf::ForOp oldForOp,
                                       ControlFlowConditionInfo *info) {
  int numBlockCounters = info->blockCounterNums[oldForOp];
  int numInnerDepConds = info->intraCoreDependentMap[oldForOp].size();
  int totalExtraArgs = numBlockCounters + numInnerDepConds;

  int numTensorIterArgs = 0;
  // Record the number of consumers for each tensor iter_args (the number of
  // parameters to be created)
  llvm::DenseMap<Value, int> tensorIterArgNumConsumers;
  // First, copy the depsVec out to avoid iterator invalidation later
  llvm::SmallVector<TensorIterArgIfOpRelation> depsVecCopy;
  auto tensorIterArgDepsIt = info->tensorIterArgDepsMap.find(oldForOp);
  if (tensorIterArgDepsIt != info->tensorIterArgDepsMap.end()) {
    depsVecCopy = tensorIterArgDepsIt->second; // Make a copy
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

  scf::ForOp newForOp =
      createForOpAndMigrateBody(oldForOp, totalExtraArgs, extraInitArgs);
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

  // Record the index of the new parameter iter_arg for the tensor and update
  // the corresponding map
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
    info->intraCoreDependentMap[newForOp] =
        info->intraCoreDependentMap[oldForOp];
    info->intraCoreDependentMap.erase(oldForOp);
  }

  return replaceForOpUsesAndErase(oldForOp, newForOp);
}

// Add block counter and inner dependency condition iter args to for ops
LogicalResult UpdateForOpsPass::addBlockCountersAndInnerDepConds(
    ModuleOp module, ControlFlowConditionInfo *info) {
  llvm::DenseSet<scf::ForOp> forOpsToProcess;

  for (auto &p : info->blockCounterNums) {
    if (p.second < 0) {
      LDBG("[Error]: invalid blockCounterNum " << p.second << "\n");
      return failure();
    }
    forOpsToProcess.insert(p.first);
  }
  for (auto &p : info->tensorIterArgDepsMap) {
    forOpsToProcess.insert(p.first);
  }

  for (scf::ForOp forOp : forOpsToProcess) {
    if (failed(extendForOpWithExtraArgs(forOp, info)))
      return failure();
  }

  return success();
}

// Insert sync ops inside a forOp: wait at start, set before yield
static LogicalResult insertSyncOpsInsideForOp(Block *forBody, Location loc,
                                              hivm::TCoreTypeAttr coreType,
                                              PipeAttr setPipe,
                                              PipeAttr waitPipe, int waitFlagId,
                                              int setFlagId) {
  Operation *forTerminator = forBody->getTerminator();
  if (!forTerminator) {
    return failure();
  }

  // Insert wait at for loop start
  OpBuilder insertionBuilder(&forBody->front());
  auto waitFlagAttr = insertionBuilder.getIntegerAttr(
      insertionBuilder.getI64Type(), waitFlagId);
  insertionBuilder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe,
                                           waitFlagAttr);

  // Insert set before yield
  OpBuilder setBuilder(forTerminator);
  auto setFlagAttr =
      setBuilder.getIntegerAttr(setBuilder.getI64Type(), setFlagId);
  setBuilder.setInsertionPoint(forTerminator);
  setBuilder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe,
                                    setFlagAttr);

  return success();
}

// Insert set before or wait after a forOp
static LogicalResult insertSetOrWaitForForOp(scf::ForOp forOp, Location loc,
                                             hivm::TCoreTypeAttr coreType,
                                             PipeAttr setPipe,
                                             PipeAttr waitPipe, int flagId,
                                             bool isBefore) {
  OpBuilder builder(forOp);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);
  if (isBefore) {
    builder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe, flagAttr);
  } else {
    builder.setInsertionPointAfter(forOp);
    builder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe, flagAttr);
  }
  return success();
}

// Insert PIPE_S for a main_loop forOp based on forOp type and scope type
static LogicalResult
insertPipeSForMainLoopForOp(scf::ForOp forOp, scope::ScopeOp scopeOp,
                            bool isScopeCube, bool isScopeVector,
                            PipeAttr setPipe, PipeAttr waitPipe, int flagId) {
  Block *forBody = &forOp.getRegion().front();
  Location loc = forOp.getLoc();
  bool isVectorFirst = forOp->hasAttr("ssbuffer.vector_first");
  auto cubeType =
      hivm::TCoreTypeAttr::get(forOp.getContext(), hivm::TCoreType::CUBE);
  auto vectorType =
      hivm::TCoreTypeAttr::get(forOp.getContext(), hivm::TCoreType::VECTOR);

  if (isVectorFirst) {
    if (isScopeCube) {
      // vector_first + CUBE: before forop (SET), inside (WAIT/SET)
      if (failed(insertSetOrWaitForForOp(forOp, loc, cubeType, setPipe,
                                         waitPipe, flagId, true))) {
        return failure();
      }
      if (failed(insertSyncOpsInsideForOp(forBody, loc, cubeType, setPipe,
                                          waitPipe, flagId, flagId))) {
        return failure();
      }
    } else if (isScopeVector) {
      // vector_first + VECTOR: inside (WAIT/SET), after forop (WAIT)
      if (failed(insertSyncOpsInsideForOp(forBody, loc, vectorType, setPipe,
                                          waitPipe, flagId, flagId))) {
        return failure();
      }
      if (failed(insertSetOrWaitForForOp(forOp, loc, vectorType, setPipe,
                                         waitPipe, flagId, false))) {
        return failure();
      }
    }
  } else {
    // cube_first (including default when neither attribute is present)
    if (isScopeCube) {
      // cube_first + CUBE: inside (WAIT/SET), after forop (WAIT)
      if (failed(insertSyncOpsInsideForOp(forBody, loc, cubeType, setPipe,
                                          waitPipe, flagId, flagId))) {
        return failure();
      }
      if (failed(insertSetOrWaitForForOp(forOp, loc, cubeType, setPipe,
                                         waitPipe, flagId, false))) {
        return failure();
      }
    } else if (isScopeVector) {
      // cube_first + VECTOR: before forop (SET), inside (WAIT/SET)
      if (failed(insertSetOrWaitForForOp(forOp, loc, vectorType, setPipe,
                                         waitPipe, flagId, true))) {
        return failure();
      }
      if (failed(insertSyncOpsInsideForOp(forBody, loc, vectorType, setPipe,
                                          waitPipe, flagId, flagId))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult UpdateForOpsPass::insertInterCorePipeS(ModuleOp module) {
  auto cubeCoreType =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  auto vectorCoreType =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto setPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  auto waitPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);

  WalkResult result = module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto scopeTypeAttr =
        scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!scopeTypeAttr) {
      return WalkResult::advance();
    }

    bool isScopeCube = (scopeTypeAttr == cubeCoreType);
    bool isScopeVector = (scopeTypeAttr == vectorCoreType);

    WalkResult innerResult = scopeOp.walk([&](scf::ForOp forOp) -> WalkResult {
      if (!forOp->hasAttr("ssbuffer.main_loop")) {
        return WalkResult::advance();
      }
      if (failed(insertPipeSForMainLoopForOp(forOp, scopeOp, isScopeCube,
                                             isScopeVector, setPipeType,
                                             waitPipeType, kPipeSFlagId))) {
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
// in the main_loop and ssbuffer.if
LogicalResult UpdateForOpsPass::analyzeTensorIterArgDependencies(
    ModuleOp module, ControlFlowConditionInfo *info) {
  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(kSsbufferMainLoop)) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }

    LDBG("Analyzing main_loop forOp: " << forOp << "\n");

    for (auto iterArg : forOp.getRegionIterArgs()) {
      if (!mlir::isa<TensorType>(iterArg.getType())) {
        continue;
      }

      LDBG("Found tensor type iter_arg: " << iterArg << "\n");
      llvm::SmallVector<scf::IfOp> producerIfOps;
      llvm::SmallVector<scf::IfOp> consumerIfOps;

      for (auto &use : iterArg.getUses()) {
        Operation *user = use.getOwner();
        scf::IfOp ifOp = nullptr;
        Operation *curr = user;
        while (curr && curr != forOp.getOperation()) {
          if (auto currIf = dyn_cast<scf::IfOp>(curr)) {
            if (currIf->hasAttr(kSsbufferIf)) {
              ifOp = currIf;
              break;
            }
          }
          curr = curr->getParentOp();
        }

        if (!ifOp) {
          LDBG("Use of tensor iter_arg "
               << iterArg << " is not inside any ssbuffer.if op." << "\n");
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

        // Check current status of ifOp
        bool inProducer = llvm::is_contained(producerIfOps, ifOp);
        bool inConsumer = llvm::is_contained(consumerIfOps, ifOp);

        if (!inProducer && !inConsumer) {
          // First time seeing this ifOp
          if (isProducer) {
            producerIfOps.push_back(ifOp);
            LDBG("  Found producer ifOp (first time): " << ifOp << "\n");
          } else {
            consumerIfOps.push_back(ifOp);
            LDBG("  Found consumer ifOp (first time): " << ifOp << "\n");
          }
        } else if (inConsumer && isProducer) {
          // Was consumer, now need to upgrade to producer (this is the only
          // update case)
          consumerIfOps.erase(llvm::find(consumerIfOps, ifOp));
          producerIfOps.push_back(ifOp);
          LDBG("  ifOp was consumer, now updated to producer: " << ifOp
                                                                << "\n");
        }
        // Note: if already in producer, do nothing even if current use is
        // consumer
      }
      // Check: must have both producers AND consumers
      if (producerIfOps.empty() || consumerIfOps.empty()) {
        LDBG("[Warning]: tensor iter_arg "
             << iterArg << " has only "
             << (producerIfOps.empty() ? "consumers" : "producers")
             << ", skipped\n");
        continue;
      }
      TensorIterArgIfOpRelation relation;
      relation.iterArg = iterArg;
      relation.producers = producerIfOps;
      relation.consumers = consumerIfOps;

      info->tensorIterArgDepsMap[forOp].push_back(relation);
      LDBG("Recorded tensor iter_arg dependency: "
           << iterArg << " has " << relation.producers.size() << " producers, "
           << relation.consumers.size() << " consumers\n");
    }

    return WalkResult::advance();
  });

  return success();
}

void UpdateForOpsPass::runOnOperation() {
  ModuleOp module = getOperation();

  LDBG("before updateForOps:\n" << module << "\n");

  // Use provided info, or create a local one if not available
  ControlFlowConditionInfo localInfo;
  ControlFlowConditionInfo *infoToUse = info ? info : &localInfo;

  // Analyze the dependencies of the tensor type iter_args in the main_loop with
  // the ssbuffer.if ops
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

  // Update for ops iter_args for block counters and inner dependency conditions
  if (infoToUse && (!infoToUse->blockCounterNums.empty() ||
                    !infoToUse->intraCoreDependentMap.empty() ||
                    !infoToUse->tensorIterArgDepsMap.empty()))
    if (failed(addBlockCountersAndInnerDepConds(module, infoToUse))) {
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

std::unique_ptr<OperationPass<ModuleOp>> createUpdateForOpsPass() {
  return std::make_unique<UpdateForOpsPass>();
}

} // namespace triton
} // namespace mlir
