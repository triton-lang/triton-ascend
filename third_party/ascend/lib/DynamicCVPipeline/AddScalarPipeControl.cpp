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

#include "ascend/include/DynamicCVPipeline/AddScalarPipeControl.h"

#include <memory>
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::hivm;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "AddScalarPipeControl";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

// Flag id reserved for the PIPE_S scalar synchronization. Kept in sync with
// kPipeSFlagId in AddControlFlowCondition/UpdateLoopOps.cpp.
static constexpr int kPipeSFlagId = 15;

// Offset of the second counter observed by a CUBE side wait. A CUBE waits for
// two VECTOR sub-blocks, which own the counters `flag_id` and `flag_id + 16`.
static constexpr int kPipeSSecondCounterOffset = 16;

// True if `op` synchronizes through PIPE_S on `flagId`, i.e. it is one of the
// scalar pipe syncs emitted around a main loop.
static bool isPipeSSyncOp(Operation *op, int flagId) {
  auto isTarget = [&](PipeAttr tpipe, PipeAttr pipe,
                      std::optional<IntegerAttr> staticFlagId) {
    if (!staticFlagId || staticFlagId->getInt() != flagId) {
      return false;
    }
    return tpipe.getPipe() == PIPE::PIPE_S && pipe.getPipe() == PIPE::PIPE_S;
  };

  if (auto waitOp = dyn_cast<SyncBlockWaitOp>(op)) {
    return isTarget(waitOp.getTpipe(), waitOp.getPipe(),
                    waitOp.getStaticFlagId());
  }
  if (auto setOp = dyn_cast<SyncBlockSetOp>(op)) {
    return isTarget(setOp.getTpipe(), setOp.getPipe(), setOp.getStaticFlagId());
  }
  return false;
}

// True when `op` contains another main_loop op inside its body. Only the
// innermost main_loop is handled: an outer loop that wraps an inner main_loop
// is left alone, because the inner one owns the PIPE_S syncs.
//
// This mirrors the invariant enforced by MarkMainLoopPass, which strips the
// `ssbuffer.main_loop` attribute from outer loops when a nested loop also
// carries it. So in a pipeline-produced IR there is exactly one main_loop.
static bool hasNestedMainLoop(Operation *op) {
  bool nested = false;
  op->walk([&](Operation *inner) {
    if (inner != op && isMainLoopOp(inner)) {
      nested = true;
    }
  });
  return nested;
}

// Build the readiness condition of the guard.
//
// The condition queries the PIPE_S counters that the blocking
// `sync_block_wait` inside the guard would consume, without consuming them:
// `try_wait` only observes the hardware state. The blocking wait therefore
// stays inside the guard and still consumes the token once the condition
// holds.
//
// A VECTOR side wait observes a single counter. A CUBE side wait observes the
// counters of the two VECTOR sub-blocks, i.e. `flag_id` and `flag_id + 16`, so
// both of them have to be queried.
//
// The queries are emitted before the guard, they must not be part of it:
// otherwise the condition could never be evaluated.
static Value buildReadinessCondition(OpBuilder &builder, Location loc,
                                     Operation *waitOp) {
  auto wait = cast<SyncBlockWaitOp>(waitOp);

  SmallVector<int> flagIds = {kPipeSFlagId};
  if (wait.getTcoreTypeAttr().getTcoretype() == TCoreType::CUBE) {
    flagIds.push_back(kPipeSFlagId + kPipeSSecondCounterOffset);
  }

  Value ready;
  for (int flagId : flagIds) {
    auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);
    auto tryWait =
        builder.create<TryWaitOp>(loc, flagAttr, TryWaitSyncMode::INTRA_BLOCK);
    Value zero = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
    // The counter holds the number of pending tokens, a non-zero value means
    // the peer already signalled.
    Value isReady = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  tryWait.getCounter(), zero);
    ready =
        ready ? builder.create<arith::AndIOp>(loc, ready, isReady) : isReady;
  }
  return ready;
}

// Guard the whole main loop body with a single `scf.if` on the readiness
// condition of the PIPE_S synchronization.
//
// The guarded region spans from the PIPE_S wait at the head of the body up to
// the matching PIPE_S set right before the terminator, so every op in between
// ends up inside the guard.
//
// The loop body may carry values across iterations. Those values are produced
// inside the guarded region, so the guard has to forward them: the then
// branch yields the values the loop used to yield, and the else branch yields
// the carried-over iter args, i.e. skipping the body keeps the loop state
// unchanged.
static LogicalResult guardMainLoopBody(Operation *loopOp) {
  // ForOp body / WhileOp after-body.
  Block *body = MainLoop(loopOp).getBody();
  if (!body) {
    return failure();
  }

  // The PIPE_S wait and the PIPE_S set that close the guarded region.
  Operation *waitOp = nullptr;
  Operation *setOp = nullptr;
  for (Operation &op : *body) {
    if (!waitOp) {
      if (isa<SyncBlockWaitOp>(&op) && isPipeSSyncOp(&op, kPipeSFlagId)) {
        waitOp = &op;
      }
      continue;
    }
    if (isa<SyncBlockSetOp>(&op) && isPipeSSyncOp(&op, kPipeSFlagId)) {
      setOp = &op;
      break;
    }
  }
  if (!waitOp || !setOp) {
    return failure();
  }

  // Every op between the wait and the set is guarded.
  SmallVector<Operation *> guarded;
  bool collecting = false;
  for (Operation &op : *body) {
    if (&op == waitOp) {
      collecting = true;
    }
    if (collecting) {
      guarded.push_back(&op);
    }
    if (&op == setOp) {
      break;
    }
  }

  auto yieldOp = dyn_cast<scf::YieldOp>(body->getTerminator());
  if (!yieldOp) {
    return failure();
  }

  // The values the loop body yields are computed inside the guarded region,
  // hence they become the results of the guard.
  SmallVector<Value> thenValues(yieldOp->getOperands());
  SmallVector<Value> elseValues;
  if (!thenValues.empty()) {
    elseValues = MainLoop(loopOp).getIterArgs();
    if (elseValues.size() != thenValues.size()) {
      return failure();
    }
    for (auto [thenValue, elseValue] : llvm::zip(thenValues, elseValues)) {
      if (thenValue.getType() != elseValue.getType()) {
        return failure();
      }
    }
  }

  // A value computed inside the guarded region must not be used outside of it,
  // otherwise moving the computation into the guard would break those uses.
  // A user counts as inside when it is nested in one of the guarded ops.
  DenseSet<Operation *> guardedSet(guarded.begin(), guarded.end());
  auto isInsideGuarded = [&](Operation *user) {
    for (Operation *parent = user; parent; parent = parent->getParentOp()) {
      if (guardedSet.contains(parent)) {
        return true;
      }
    }
    return false;
  };
  for (Operation *op : guarded) {
    for (Value result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        if (user == yieldOp.getOperation()) {
          continue;
        }
        if (!isInsideGuarded(user)) {
          return failure();
        }
      }
    }
  }

  OpBuilder builder(waitOp);
  Location loc = waitOp->getLoc();

  Value condition = buildReadinessCondition(builder, loc, waitOp);

  SmallVector<Type> resultTypes;
  for (Value value : thenValues) {
    resultTypes.push_back(value.getType());
  }

  scf::IfOp ifOp =
      builder.create<scf::IfOp>(loc, resultTypes, condition,
                                /*withElseRegion=*/!thenValues.empty());

  Region &thenRegion = ifOp.getThenRegion();
  if (thenRegion.empty()) {
    thenRegion.emplaceBlock();
  }
  Block &thenBlock = thenRegion.front();
  if (thenBlock.mightHaveTerminator()) {
    thenBlock.getTerminator()->erase();
  }

  for (Operation *op : guarded) {
    op->moveBefore(&thenBlock, thenBlock.end());
  }

  OpBuilder thenBuilder(&thenBlock, thenBlock.end());
  thenBuilder.create<scf::YieldOp>(loc, thenValues);

  if (!thenValues.empty()) {
    Region &elseRegion = ifOp.getElseRegion();
    if (elseRegion.empty()) {
      elseRegion.emplaceBlock();
    }
    Block &elseBlock = elseRegion.front();
    if (elseBlock.mightHaveTerminator()) {
      elseBlock.getTerminator()->erase();
    }

    OpBuilder elseBuilder(&elseBlock, elseBlock.end());
    elseBuilder.create<scf::YieldOp>(loc, elseValues);

    // The loop now yields the results of the guard.
    yieldOp->setOperands(ifOp.getResults());
  }

  LOG_DEBUG("Guarded main loop body with scf.if: " << *waitOp);
  return success();
}

void AddScalarPipeControlPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // The guard condition is built on `try_wait`, which is only available on the
  // register based architectures. On the memory based ones (A3) the LLVM
  // export rejects the op, so the main loop is left untouched there.
  if (hacc::utils::isMemBasedArch(module)) {
    LOG_DEBUG("mem-based arch, try_wait is unsupported: keep main loop as is");
    return;
  }

  SmallVector<Operation *> mainLoops;
  module.walk([&](Operation *op) {
    if (isMainLoopOp(op)) {
      mainLoops.push_back(op);
    }
  });

  for (Operation *loopOp : mainLoops) {
    // Only the innermost main_loop is handled: an outer loop that wraps
    // another main_loop is left alone, the inner one owns the PIPE_S syncs.
    if (hasNestedMainLoop(loopOp)) {
      continue;
    }

    if (failed(guardMainLoopBody(loopOp))) {
      LOG_DEBUG("No pipe_s wait/set pair to guard in main loop");
    }
  }
}

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddScalarPipeControlPass() {
  return std::make_unique<AddScalarPipeControlPass>();
}

void registerAddScalarPipeControlPasses() {
  registerPass(createAddScalarPipeControlPass);
}

} // namespace mlir::triton
