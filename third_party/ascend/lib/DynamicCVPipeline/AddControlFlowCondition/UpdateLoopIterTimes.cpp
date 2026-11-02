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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/InitDependentMap.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "UpdateLoopIterTimes";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

using llvm::DenseMap;
using llvm::SmallVector;
using namespace mlir;
using namespace triton;

// Type definition for IfOp filter function
// Returns true if the IfOp should be collected, false otherwise
using IfOpFilter = std::function<bool(scf::IfOp)>;

// Default filter that accepts all IfOps
static bool defaultIfOpFilter(scf::IfOp ifOp) { return true; }

// Collect ifOps from forOp based on the provided filter
// Fills ifOps and ifOpIndexMap with the collected ifOps
// Returns true on success, false on error or if no ifOps found
bool collectIfOps(scf::ForOp forOp, SmallVector<scf::IfOp> &ifOps,
                  llvm::DenseMap<Operation *, int> &ifOpIndexMap,
                  IfOpFilter filter = defaultIfOpFilter) {
  ifOps.clear();
  ifOpIndexMap.clear();
  int index = 1;
  int ret = 0;

  forOp.walk([&](Operation *op) {
    if (op->hasAttr(CVPipeline::kIf)) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp) {
        ret = -1;
        LDBG("ssbuffer.if attribute is not allocated on ifOp!");
        return WalkResult::interrupt();
      }
      // Apply the filter to determine if this IfOp should be collected
      if (filter(ifOp)) {
        ifOps.push_back(ifOp);
        ifOpIndexMap[ifOp.getOperation()] = index++;
      }
    }
    return WalkResult::advance();
  });

  if (ret == -1) {
    LDBG("Failed to collect ifOps!");
    return false;
  }

  if (ifOps.empty()) {
    LDBG("mainloop does not contain any ifblocks!");
    return false;
  }

  return true;
}

// Update WhileOp condition based on ifblock conditions
// Uses info->whileBlockArgMap: {WhileOp: {block_id: {new_arg_idx:
// old_arg_idx}}} For each IfOp in WhileOp, get its block_id Get new args and
// mapping from info->whileBlockArgMap Copy entire beforeRegion and replace old
// args with new args Combine all ifblock conditions with OR operation
int UpdateLoopIterTimesPass::UpdateWhileLoopCondition(
    DenseMap<int, SmallVector<Operation *>> &mainLoopIdMap) {
  int ret = 0;

  for (auto &entry : mainLoopIdMap) {
    for (Operation *loopOp : entry.second) {
      if (!isa<scf::WhileOp>(loopOp)) {
        continue;
      }

      auto whileOp = dyn_cast<scf::WhileOp>(loopOp);
      if (!info->whileBlockArgMap.count(whileOp)) {
        LDBG("WhileOp not found in whileBlockArgMap!");
        return -1;
      }
      auto &blockArgMap = info->whileBlockArgMap[whileOp];
      if (blockArgMap.empty()) {
        LDBG("blockArgMap is empty for WhileOp!");
        return -1;
      }

      // Get the 'before' region which contains the condition
      Region &beforeRegion = whileOp.getBefore();
      Block &beforeBlock = beforeRegion.front();
      Operation *terminator = beforeBlock.getTerminator();
      if (!terminator || !isa<scf::ConditionOp>(terminator)) {
        LDBG("Before block has no valid ConditionOp terminator!");
        return -1;
      }
      auto conditionOp = dyn_cast<scf::ConditionOp>(terminator);
      Value originalCondition = conditionOp.getCondition();
      OpBuilder builder(conditionOp);
      Location loc = whileOp.getLoc();

      // Collect all operations to clone
      SmallVector<Operation *> opsToClone;
      for (Operation &op : beforeBlock.without_terminator()) {
        opsToClone.push_back(&op);
      }

      // Traverse all IfOps with ssbuffer.if and
      // build condition for each IfOp directly
      Value combinedCondition;
      whileOp.walk([&](scf::IfOp ifOp) {
        if (!ifOp->hasAttr(CVPipeline::kIf)) {
          return WalkResult::advance();
        }

        auto blockIdAttr = ifOp->getAttrOfType<IntegerAttr>(CVPipeline::kIf);
        if (!blockIdAttr) {
          ret = -1;
          return WalkResult::interrupt();
        }
        int blockId = blockIdAttr.getInt();
        if (!blockArgMap.count(blockId)) {
          ret = -1;
          LDBG("blockId " << blockId << " not found in blockArgMap!");
          return WalkResult::interrupt();
        }

        auto &argMap = blockArgMap[blockId]; // {new_arg_idx: old_arg_idx}

        // Step 1: Build arg replacement mapping
        IRMapping mapper;
        for (auto &argEntry : argMap) {
          int newArgIdx = argEntry.first;
          int oldArgIdx = argEntry.second;

          BlockArgument oldArg = beforeBlock.getArgument(oldArgIdx);
          BlockArgument newArg = beforeBlock.getArgument(newArgIdx);

          mapper.map(oldArg, newArg);
        }

        // Step 2: Clone entire beforeRegion
        // Set insertion point before conditionOp
        builder.setInsertionPoint(conditionOp);

        // Clone all operations
        Value newCondition;
        for (Operation *op : opsToClone) {
          Operation *clonedOp = builder.clone(*op, mapper);

          // If this op defines the original condition, get the corresponding
          // result
          if (op->getResultTypes().size() > 0) {
            for (unsigned i = 0; i < op->getNumResults(); ++i) {
              if (op->getResult(i) == originalCondition) {
                newCondition = clonedOp->getResult(i);
                break;
              }
            }
          }
        }

        if (!newCondition) {
          ret = -1;
          LDBG("Failed to create new condition for blockId: " << blockId);
          return WalkResult::interrupt();
        }

        // Step 3: Combine with existing conditions using OR
        if (!combinedCondition) {
          combinedCondition = newCondition;
        } else {
          combinedCondition = builder.create<arith::OrIOp>(
              loc, combinedCondition, newCondition);
        }

        return WalkResult::advance();
      });

      // Step 4: Update the condition in the before block
      if (combinedCondition) {
        conditionOp.getConditionMutable().assign(combinedCondition);
        LDBG("Updated WhileOp condition for: " << whileOp);
      } else {
        LDBG("Failed to generate valid condition!");
        return -1;
      }
    }
  }

  return ret;
}

// Traverse each mainloop, find ifOp with ssbuffer.if attribute inside,
// and replace the mainloop's induction variable with the counter in cntArgs
int UpdateLoopIterTimesPass::replaceForOpCounterInIfOps() {
  int ret = 0;
  // Traverse all mainloops in the module
  getOperation().walk([&](Operation *op) {
    if (op->hasAttr(CVPipeline::kMainLoop)) {
      // Skip WhileOp, only process ForOp in this step
      if (isa<scf::WhileOp>(op)) {
        LDBG("Skip WhileOp in replaceForOpCounterInIfOps.");
        return WalkResult::advance();
      }

      auto forOp = dyn_cast<scf::ForOp>(op);
      if (!forOp) {
        ret = -1;
        LDBG("Do not support other mainloop except forOp!");
        return WalkResult::interrupt();
      }

      Value indVar = forOp.getInductionVar();

      // Find all ifOps with ssbuffer.if attribute inside this mainloop
      forOp.walk([&](scf::IfOp ifOp) {
        if (ifOp->hasAttr(CVPipeline::kIf)) {
          if (!info->cntArgs.count(ifOp)) {
            LDBG("ifblock has no counter in cntArgs");
            ret = -1;
            return WalkResult::interrupt();
          }

          Value cntVal = info->cntArgs[ifOp];

          // Replace all uses of induction variable in the ifOp with cntVal
          ifOp.walk([&](Operation *op) {
            for (OpOperand &operand : op->getOpOperands()) {
              if (operand.get() == indVar) {
                operand.set(cntVal);
              }
            }
          });
        }
        return mlir::WalkResult::advance();
      });
    }
    return mlir::WalkResult::advance();
  });
  return ret;
}

// step1: Get mainloop id to loop operation mapping, separated into cube and
// vector
int UpdateLoopIterTimesPass::GetMainLoopIdToLoopOpMap(
    ModuleOp module, DenseMap<int, SmallVector<Operation *>> &cmap,
    DenseMap<int, SmallVector<Operation *>> &vmap) {
  cmap.clear();
  vmap.clear();
  int ret = 0;

  module.walk([&](scope::ScopeOp scopeOp) {
    // Determine if it's CUBE or VECTOR
    bool isCube = false;
    bool isVector = false;
    if (failed(getScopeType(scopeOp, isCube, isVector))) {
      ret = -1;
      LDBG("mlir do not processed by split mix kernel!");
      return mlir::WalkResult::interrupt();
    }

    // Walk for loops inside the scope
    scopeOp.walk([&](Operation *op) {
      if (op->hasAttr(CVPipeline::kMainLoop)) {
        // Support both ForOp and WhileOp for mainloop
        bool isValidLoop = isa<scf::ForOp>(op) || isa<scf::WhileOp>(op);
        if (!isValidLoop) {
          ret = -1;
          LDBG("mainloop must be ForOp or WhileOp!");
          return mlir::WalkResult::interrupt();
        }

        auto mainLoopId = op->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
        if (mainLoopId) {
          int id = mainLoopId.getInt();
          if (isCube) {
            cmap[id].push_back(op);
          } else if (isVector) {
            vmap[id].push_back(op);
          }
        }
      }
      return mlir::WalkResult::advance();
    });
    return mlir::WalkResult::advance();
  });

  return ret;
}


// Returns the number of the main loop's body argument that holds the counter
// of `ifOp`, or -1 when the counter of the block is unknown.
static int getCounterArgNumber(scf::ForOp forOp, scf::IfOp ifOp,
                               const llvm::DenseMap<scf::IfOp, Value> &cntArgs) {
  Value counter = cntArgs.lookup(ifOp);
  if (!counter) {
    return -1;
  }
  auto arg = dyn_cast<BlockArgument>(counter);
  if (!arg || arg.getOwner() != forOp.getBody()) {
    return -1;
  }
  // The body of the while holds the same arguments as the body of the for:
  // the induction variable comes first, the iter args after it.
  return static_cast<int>(arg.getArgNumber());
}

// Builds the condition of the while loop.
//
// The loop keeps running while at least one of the ssbuffer.if computation
// blocks has not reached the loop upper bound yet, i.e. the condition is the
// OR of "counter < upperBound" over all the blocks. The loop therefore exits
// only when every block ran as often as the upper bound asks for.
//
// The counters are the loop carried values of the main loop, so the condition
// has to compare the arguments of the 'before' region.
static Value buildWhileCondition(scf::ForOp forOp, Block *beforeBlock,
                                  const llvm::DenseMap<scf::IfOp, Value> &cntArgs,
                                  OpBuilder &builder, Location loc) {
  Value upperBound = forOp.getUpperBound();
  Value condition;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp || !ifOp->hasAttr(CVPipeline::kIf)) {
      continue;
    }
    int argNumber = getCounterArgNumber(forOp, ifOp, cntArgs);
    if (argNumber < 0) {
      LDBG("no counter known for the ssbuffer.if, it does not take part in "
           "the loop condition");
      continue;
    }
    Value counter = beforeBlock->getArgument(argNumber);
    if (counter.getType() != upperBound.getType()) {
      LDBG("the counter of the ssbuffer.if and the loop upper bound have "
           "different types");
      continue;
    }
    Value cmp = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, counter, upperBound);
    if (!condition) {
      condition = cmp;
      continue;
    }
    condition = builder.create<arith::OrIOp>(loc, condition, cmp);
  }
  return condition;
}

// Rewrite a main-loop scf.for into an equivalent scf.while.
//
// This mirrors upstream's `ForLoopLoweringPattern`
// (mlir/lib/Dialect/SCF/Transforms/ForToWhile.cpp) with one extra guard: only
// loops carrying the `ssbuffer.main_loop` tag are rewritten, every other
// scf.for is left untouched.
//
// The upstream pattern is internal to its translation unit, so it cannot be
// reused with a filter; the conversion body is copied here instead.
static LogicalResult convertMainLoopForToWhile(
    scf::ForOp forOp, const llvm::DenseMap<scf::IfOp, Value> &cntArgs,
    RewriterBase &rewriter) {
  Location loc = forOp.getLoc();
  // Loop-carried values of the while: the induction variable comes first,
  // followed by the for loop's iter args.
  SmallVector<Type> lcvTypes;
  SmallVector<Location> lcvLocs;
  lcvTypes.push_back(forOp.getInductionVar().getType());
  lcvLocs.push_back(forOp.getInductionVar().getLoc());
  for (Value value : forOp.getInitArgs()) {
    lcvTypes.push_back(value.getType());
    lcvLocs.push_back(value.getLoc());
  }

  SmallVector<Value> initArgs;
  initArgs.push_back(forOp.getLowerBound());
  llvm::append_range(initArgs, forOp.getInitArgs());

  // Carry over the for loop's discardable attributes (ssbuffer.main_loop,
  // DataUse, ...) so the while loop keeps the same tagging.
  auto whileOp = rewriter.create<scf::WhileOp>(forOp.getLoc(), lcvTypes,
                                               initArgs, forOp->getAttrs());

  // 'before' region holds the loop condition and forwards the iteration
  // arguments to the 'after' region.
  auto *beforeBlock = rewriter.createBlock(
      &whileOp.getBefore(), whileOp.getBefore().begin(), lcvTypes, lcvLocs);
  rewriter.setInsertionPointToStart(whileOp.getBeforeBody());
  Value condition = buildWhileCondition(forOp, beforeBlock, cntArgs, rewriter,
                                        whileOp.getLoc());
  if (!condition) {
    // The main loop holds no ssbuffer.if block to drive it, fall back to the
    // induction variable.
    // `unsignedCmp` is an operation property, so it is not carried over to
    // the while op; the comparison predicate has to be selected explicitly.
    arith::CmpIPredicate predicate = forOp.getUnsignedCmp()
                                         ? arith::CmpIPredicate::ult
                                         : arith::CmpIPredicate::slt;
    condition = rewriter.create<arith::CmpIOp>(
        whileOp.getLoc(), predicate, beforeBlock->getArgument(0),
        forOp.getUpperBound());
  }
  rewriter.create<scf::ConditionOp>(whileOp.getLoc(), condition,
                                    beforeBlock->getArguments());

  // 'after' region holds the induction variable increment and the loop body.
  // The while op returns an extra value (the induction variable), so the
  // yields of the source for loop contain only the iter args.
  auto *afterBlock = rewriter.createBlock(
      &whileOp.getAfter(), whileOp.getAfter().begin(), lcvTypes, lcvLocs);

  rewriter.setInsertionPointToEnd(afterBlock);
  auto ivIncOp = rewriter.create<arith::AddIOp>(
      whileOp.getLoc(), afterBlock->getArgument(0), forOp.getStep());

  // Rewrite uses of the for loop's block arguments to the while loop's
  // 'after' arguments.
  for (const auto &barg : llvm::enumerate(forOp.getBody(0)->getArguments())) {
    rewriter.replaceAllUsesWith(barg.value(),
                                afterBlock->getArgument(barg.index()));
  }

  // Inline the for loop body into the 'after' region.
  for (auto &arg : llvm::make_early_inc_range(*forOp.getBody())) {
    rewriter.moveOpBefore(&arg, afterBlock, afterBlock->end());
  }

  // Prepend the incremented induction variable to the yielded values.
  for (auto yieldOp : afterBlock->getOps<scf::YieldOp>()) {
    SmallVector<Value> yieldOperands = yieldOp.getOperands();
    yieldOperands.insert(yieldOperands.begin(), ivIncOp.getResult());
    rewriter.modifyOpInPlace(yieldOp,
                             [&]() { yieldOp->setOperands(yieldOperands); });
  }

  // The while op returns one extra value, so the for loop's results are
  // remapped to the while op's results starting from index 1.
  for (const auto &arg : llvm::enumerate(forOp.getResults())) {
    rewriter.replaceAllUsesWith(arg.value(),
                                whileOp.getResult(arg.index() + 1));
  }

  rewriter.eraseOp(forOp);
  // Note: the while op must not be printed before its regions are filled in,
  // the asm printer verifies every op it prints.
  LDBG("converted main loop scf.for at " << loc << " into scf.while");
  return success();
}

// Convert every main loop tagged with `ssbuffer.main_loop` from scf.for into
// scf.while.
static LogicalResult convertMainLoopsToWhile(
    ModuleOp module, const llvm::DenseMap<scf::IfOp, Value> &cntArgs) {
  // Collect the tagged main loops up front: the conversion rewrites the loop
  // body, so walking the IR while modifying it is not safe.
  SmallVector<scf::ForOp> mainLoops;
  module.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(CVPipeline::kMainLoop)) {
      mainLoops.push_back(forOp);
    }
  });

  if (mainLoops.empty()) {
    LDBG("no main loop found, nothing to convert");
    return success();
  }

  // The rewrite is driven manually instead of through the greedy pattern
  // driver: the driver also drops "trivially dead" ops, which would delete
  // loops whose body has no side effect.
  IRRewriter rewriter(module.getContext());
  for (scf::ForOp forOp : mainLoops) {
    rewriter.setInsertionPoint(forOp);
    if (failed(convertMainLoopForToWhile(forOp, cntArgs, rewriter))) {
      module.emitError() << "Failed to convert main loop scf.for into "
                            "scf.while";
      return failure();
    }
  }
  return success();
}

// ---------------------------------------------------------------------------
// Standalone mode: rebuild the ControlFlowConditionInfo from the IR.
//
// UpdateLoopIterTimesPass is normally driven by AddControlFlowConditionPass,
// which shares its ControlFlowConditionInfo with the preceding sub-passes.
// When the pass runs on its own (`triton-opt --update-loop-iter-times`) there
// is no such info, but everything the info holds has already been
// materialised in the IR by the preceding sub-passes:
//   * the dependency maps are derived from the ssbuffer.intraDeps and
//     ssbuffer.crossCoreDeps attributes (InitDependentMapPass),
//   * the tensor iter_arg relations are derived from the uses of the tensor
//     iter_args of the main loops,
//   * the counter of an ssbuffer.if is the main loop iter_arg that the if's
//     condition compares against the main loop upper bound.
// ---------------------------------------------------------------------------

// Returns true if `value` is one of the main loop's region iter args.
static bool isMainLoopIterArg(scf::ForOp forOp, Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  return arg && arg.getOwner() == forOp.getBody();
}

// Traces `value` back to the main loop iter_arg it originates from. A counter
// can be used through its latest value, i.e. the result of the if op that
// updated it, so the trace follows the arithmetic and the if results as well.
static Value traceToMainLoopIterArg(scf::ForOp forOp, Value value,
                                    unsigned depth = 0) {
  constexpr unsigned maxDepth = 8;
  if (depth > maxDepth) {
    return nullptr;
  }
  if (isMainLoopIterArg(forOp, value)) {
    return value;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return nullptr;
  }

  // counter + step still points at the counter iter arg.
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    if (Value lhs = traceToMainLoopIterArg(forOp, addOp.getLhs(), depth + 1))
      return lhs;
    return traceToMainLoopIterArg(forOp, addOp.getRhs(), depth + 1);
  }

  // The latest value of a counter is a result of the if op that updated it.
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return nullptr;
  }
  auto ifOp = dyn_cast<scf::IfOp>(result.getOwner());
  if (!ifOp) {
    return nullptr;
  }
  unsigned index = result.getResultNumber();
  for (Region *region : ifOp.getRegions()) {
    auto yieldOp =
        dyn_cast<scf::YieldOp>(region->getBlocks().front().getTerminator());
    if (!yieldOp || index >= yieldOp.getNumOperands()) {
      continue;
    }
    if (Value operand =
            traceToMainLoopIterArg(forOp, yieldOp.getOperand(index), depth + 1))
      return operand;
  }
  return nullptr;
}

// Finds the counter of `ifOp`: the main loop iter_arg that the if's condition
// compares against the main loop upper bound.
static Value findIfOpCounter(scf::ForOp forOp, scf::IfOp ifOp) {
  Value upperBound = forOp.getUpperBound();
  SmallVector<Value> worklist;
  worklist.push_back(ifOp.getCondition());
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *defOp = value.getDefiningOp();
    if (!defOp) {
      continue;
    }
    if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
      if (cmpOp.getPredicate() != arith::CmpIPredicate::slt ||
          cmpOp.getRhs() != upperBound) {
        continue;
      }
      if (Value counter = traceToMainLoopIterArg(forOp, cmpOp.getLhs()))
        return counter;
      continue;
    }
    // The conditions of an if op are combined with arith.andi.
    if (auto andOp = dyn_cast<arith::AndIOp>(defOp)) {
      worklist.push_back(andOp.getLhs());
      worklist.push_back(andOp.getRhs());
    }
  }
  return nullptr;
}

// Collects the counter of every ssbuffer.if of the main loops.
static void collectIfOpCounters(ModuleOp module,
                                ControlFlowConditionInfo &info) {
  module.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr(CVPipeline::kMainLoop)) {
      return;
    }
    SmallVector<scf::IfOp> ifOps;
    llvm::DenseMap<Operation *, int> ifOpIndex;
    if (!collectIfOps(forOp, ifOps, ifOpIndex)) {
      return;
    }
    for (scf::IfOp ifOp : ifOps) {
      Value counter = findIfOpCounter(forOp, ifOp);
      if (!counter) {
        LDBG("Can not find the counter of the if op: " << ifOp);
        continue;
      }
      info.cntArgs[ifOp] = counter;
    }
  });
}

// Rebuilds the tensor iter_arg dependency relations of the main loops: for
// every tensor iter_arg, the ssbuffer.if that produces it and the ssbuffer.if
// ops that consume it.
static void collectTensorIterArgDeps(ModuleOp module,
                                     ControlFlowConditionInfo &info) {
  module.walk([&](Operation *op) {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
      return;
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      return;
    }

    for (BlockArgument iterArg : forOp.getRegionIterArgs()) {
      if (!isa<TensorType>(iterArg.getType())) {
        continue;
      }

      scf::IfOp producerIfOp;
      SmallVector<scf::IfOp> consumerIfOps;
      for (OpOperand &use : iterArg.getUses()) {
        Operation *user = use.getOwner();

        // The ssbuffer.if the use belongs to.
        scf::IfOp ifOp;
        for (Operation *curr = user; curr && curr != op;
             curr = curr->getParentOp()) {
          auto currIf = dyn_cast<scf::IfOp>(curr);
          if (currIf && currIf->hasAttr(CVPipeline::kIf)) {
            ifOp = currIf;
            break;
          }
        }
        if (!ifOp) {
          continue;
        }

        // Only the terminator of the ssbuffer.if produces the iter_arg.
        bool isProducer = isa<scf::YieldOp>(user) &&
                          user->getParentOp() == ifOp.getOperation();
        if (isProducer) {
          if (producerIfOp && producerIfOp != ifOp) {
            // Ambiguous producer, drop the relation.
            producerIfOp = nullptr;
            break;
          }
          producerIfOp = ifOp;
          continue;
        }
        if (producerIfOp == ifOp) {
          continue;
        }
        if (!llvm::is_contained(consumerIfOps, ifOp)) {
          consumerIfOps.push_back(ifOp);
        }
      }

      if (!producerIfOp || consumerIfOps.empty()) {
        continue;
      }
      TensorIterArgIfOpRelation relation;
      relation.iterArg = iterArg;
      relation.producer = producerIfOp;
      relation.consumers = consumerIfOps;
      info.tensorIterArgDepsMap[op].push_back(relation);
    }
  });
}

// Rebuilds the whole ControlFlowConditionInfo from the IR.
LogicalResult UpdateLoopIterTimesPass::buildConditionInfoFromIR(
    ModuleOp module, ControlFlowConditionInfo &info) {
  // The dependency maps are derived from the ssbuffer.intraDeps and
  // ssbuffer.crossCoreDeps attributes, exactly like InitDependentMapPass does
  // inside the AddControlFlowCondition pipeline.
  {
    PassManager pm(module.getContext(), module.getOperationName());
    std::unique_ptr<InitDependentMapPass> initDependentMapPass(
        new InitDependentMapPass());
    initDependentMapPass->setConditionInfo(&info);
    pm.addPass(std::move(initDependentMapPass));
    if (failed(runPipeline(pm, module))) {
      LDBG("InitDependentMapPass failed while rebuilding the condition info!");
      return failure();
    }
  }
  if (CVPipeline::hasFallbackAttr(module)) {
    LDBG("InitDependentMapPass failed while rebuilding the condition info!");
    return failure();
  }

  collectTensorIterArgDeps(module, info);
  collectIfOpCounters(module, info);
  return success();
}

void UpdateLoopIterTimesPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("before updateloopitertimes:\n" << module);
  LDBG("\nEnter UpdateLoopIterTimesPass!");

  // The pass is normally driven by AddControlFlowConditionPass, which shares
  // its ControlFlowConditionInfo with the preceding sub-passes. Running the
  // pass on its own leaves the info unset, in that case it is rebuilt from
  // the IR.
  ControlFlowConditionInfo localInfo;
  if (!info) {
    LDBG("no condition info provided, rebuild it from the IR");
    if (failed(buildConditionInfoFromIR(module, localInfo))) {
      LDBG("Failed to rebuild the condition info from the IR!");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
    info = &localInfo;
  }

  int ret = 0;
  // step1: Get mainloop id to loop operation mapping, separated into cube and
  // vector
  DenseMap<int, SmallVector<Operation *>> cmap;
  DenseMap<int, SmallVector<Operation *>> vmap;
  ret = GetMainLoopIdToLoopOpMap(module, cmap, vmap);
  if (ret != 0) {
    LDBG("GetMainLoopIdToLoopOpMap Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  // step2: Replace loop counter by if blocks' counter
  ret = replaceForOpCounterInIfOps();
  if (ret != 0) {
    LDBG("replaceForOpCounterInIfOps Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  // step3: Update WhileOp condition based on ifblock conditions
  // The whileBlockArgMap is only filled by the preceding sub-passes, so the
  // step is skipped when the pass runs on its own and the map is empty.
  if (!info->whileBlockArgMap.empty()) {
    ret = UpdateWhileLoopCondition(cmap);
    if (ret != 0) {
      LDBG("UpdateWhileLoopCondition from cube Failed!");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    ret = UpdateWhileLoopCondition(vmap);
    if (ret != 0) {
      LDBG("UpdateWhileLoopCondition from vector Failed!");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
  } else {
    LDBG("no whileBlockArgMap, skip updating the while op conditions");
  }

  // step4: Convert the main loops from scf.for into scf.while. The while loop
  // keeps the upper bound of the for loop and its condition is built from the
  // counters of the ssbuffer.if blocks.
  if (failed(convertMainLoopsToWhile(module, info->cntArgs))) {
    LDBG("convert main loops to while Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  LDBG("after updateloopitertimes:\n" << module);
  LDBG("\nExit UpdateLoopIterTimes pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createUpdateLoopIterTimesPass() {
  return std::make_unique<UpdateLoopIterTimesPass>();
}
} // namespace triton
} // namespace mlir
