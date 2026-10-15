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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/WrapSplittedIf.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "WrapSplittedIfPass";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
  })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

// Same shape as UpdateConditionInfo: 0 success, -1 unexpected failure.
// IGNORED is 2, the same value as ERRCODE_IGNORED.
// A per-if or per-loop IGNORED is consumed by the caller: that if or loop
// stays unwrapped, and the module attribute is not set. Setting it would
// restore the snapshot taken before the whole CV pipeline.
// A walk that stops with IGNORED is an expected pipeline rollback
// (ERRCODE_IGNORED). A walk that stops with FAILED is a bug (ERRCODE_FAILED).
static constexpr int WRAP_SPLITTED_IF_SUCCESS = 0;
static constexpr int WRAP_SPLITTED_IF_FAILED = -1;
static constexpr int WRAP_SPLITTED_IF_IGNORED = 2;

// First-level then-block search for the one ssbuffer.splitted_if.
// Nested ifs and sibling ifs without the attr are ignored.
// Returns null when there are zero or more than one.
static scf::IfOp getUniqueFirstLevelSplittedIf(scf::IfOp ssbufIf) {
  if (!ssbufIf || !ssbufIf.thenBlock())
    return nullptr;
  scf::IfOp found;
  for (Operation &op : *ssbufIf.thenBlock()) {
    auto innerIf = dyn_cast<scf::IfOp>(&op);
    if (!innerIf || !innerIf->hasAttr(kSplittedIf))
      continue;
    if (found)
      return nullptr;
    found = innerIf;
  }
  return found;
}

// `splitIf` is the only ssbuffer.splitted_if anywhere under `ssbufIf`.
// getUniqueFirstLevelSplittedIf only looks at the then block's first level.
static bool isOnlySplittedIf(scf::IfOp ssbufIf, scf::IfOp splitIf) {
  bool extra = false;
  ssbufIf.walk([&](scf::IfOp ifOp) {
    if (ifOp == ssbufIf || !ifOp->hasAttr(kSplittedIf))
      return WalkResult::advance();
    if (ifOp == splitIf)
      return WalkResult::advance();
    extra = true;
    return WalkResult::interrupt();
  });
  return !extra;
}

// True when \p op lives in splitIf's then block, or is nested under an op there.
// The check runs before the condition swap, so that block is still the
// original splitted_if then.
static bool opIsInSplitThen(Operation *op, scf::IfOp splitIf) {
  Block *thenBlock = splitIf.thenBlock();
  if (!thenBlock)
    return false;
  return thenBlock->findAncestorOpInBlock(*op) != nullptr;
}

// Real tensor compute is allowed only in splitIf's then. The else is dropped
// by the wrap, and every other op in ssbufIf moves under the split cond.
// Walk from ssbufIf and ignore splitIf's then block. Do not skip the whole
// splitIf: its else, and ops after it, still have to be checked.
static bool computeOnlyInSplitThen(scf::IfOp ssbufIf, scf::IfOp splitIf) {
  WalkResult walk = ssbufIf->walk([&](Operation *op) {
    if (opIsInSplitThen(op, splitIf))
      return WalkResult::skip();
    if (isRealTensorComputeOp(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !walk.wasInterrupted();
}

// Walk the def-chain of a condition value and keep ops defined inside ssbufIf.
// SUCCESS when the chain is collected. FAILED when the walk cannot finish.
static int collectConditionDefOpsInside(Value value, scf::IfOp ssbufIf,
                                        DenseSet<Operation *> &ops) {
  if (!value)
    return WRAP_SPLITTED_IF_SUCCESS;
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return WRAP_SPLITTED_IF_SUCCESS;
  if (!ssbufIf->isProperAncestor(defOp))
    return WRAP_SPLITTED_IF_SUCCESS;
  if (!ops.insert(defOp).second)
    return WRAP_SPLITTED_IF_SUCCESS;
  for (Value operand : defOp->getOperands()) {
    if (collectConditionDefOpsInside(operand, ssbufIf, ops) !=
        WRAP_SPLITTED_IF_SUCCESS)
      return WRAP_SPLITTED_IF_FAILED;
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// True when \p def properly dominates \p user.
static bool opDominates(Operation *def, Operation *user) {
  if (!def || !user || def == user)
    return false;
  Block *defBlock = def->getBlock();
  Block *userBlock = user->getBlock();
  if (!defBlock || !userBlock)
    return false;
  if (defBlock == userBlock)
    return def->isBeforeInBlock(user);
  Operation *cur = user;
  while (Operation *parent = cur->getParentOp()) {
    if (parent == def)
      return false;
    if (parent->getBlock() == defBlock)
      return def->isBeforeInBlock(parent);
    cur = parent;
  }
  return false;
}

// True when every out-of-chain operand already dominates ssbufIf, so the
// chain can move above it. A def inside a sibling ssbuffer.if does not.
static bool canHoistConditionOps(scf::IfOp ssbufIf,
                                 const DenseSet<Operation *> &ops) {
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      if (ops.contains(def))
        continue;
      // Operand stays put: it must already dominate the hoist point. A def
      // inside a sibling ssbuffer.if (e.g. the pre-clone original) does not.
      if (!opDominates(def, ssbufIf))
        return false;
    }
  }
  return true;
}

// Move the splitted_if condition def-chain out of ssbufIf, above ssbufIf.
// No IR change until every def is in the then block and can be hoisted.
// IGNORED: the chain is not hoistable. FAILED: the if is missing structure.
// SUCCESS: nothing to move, or the chain now sits above ssbufIf.
static int hoistSplittedIfCondition(scf::IfOp ssbufIf, scf::IfOp splittedIf) {
  Value cond = splittedIf.getCondition();
  if (!cond)
    return WRAP_SPLITTED_IF_FAILED;

  DenseSet<Operation *> condOps;
  if (collectConditionDefOpsInside(cond, ssbufIf, condOps) !=
      WRAP_SPLITTED_IF_SUCCESS)
    return WRAP_SPLITTED_IF_FAILED;
  if (condOps.empty())
    return WRAP_SPLITTED_IF_SUCCESS;
  if (!canHoistConditionOps(ssbufIf, condOps))
    return WRAP_SPLITTED_IF_IGNORED;

  Block *thenBlock = ssbufIf.thenBlock();
  if (!thenBlock)
    return WRAP_SPLITTED_IF_FAILED;
  SmallVector<Operation *> sorted;
  for (Operation &op : *thenBlock) {
    if (condOps.contains(&op))
      sorted.push_back(&op);
  }
  // A def nested deeper than the then block cannot be lifted in order.
  if (sorted.size() != condOps.size())
    return WRAP_SPLITTED_IF_IGNORED;
  for (Operation *op : sorted)
    op->moveBefore(ssbufIf);
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Def-chain of \p value that already sits in \p anchor's block, before anchor.
// SUCCESS, or FAILED when the recursive walk cannot finish.
static int collectConditionDefOpsBefore(Value value, Operation *anchor,
                                        DenseSet<Operation *> &ops) {
  if (!value || !anchor)
    return WRAP_SPLITTED_IF_SUCCESS;
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return WRAP_SPLITTED_IF_SUCCESS;
  if (defOp->getBlock() != anchor->getBlock() ||
      !defOp->isBeforeInBlock(anchor))
    return WRAP_SPLITTED_IF_SUCCESS;
  if (!ops.insert(defOp).second)
    return WRAP_SPLITTED_IF_SUCCESS;
  for (Value operand : defOp->getOperands()) {
    if (collectConditionDefOpsBefore(operand, anchor, ops) !=
        WRAP_SPLITTED_IF_SUCCESS)
      return WRAP_SPLITTED_IF_FAILED;
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// annotation.mark on a cond value is not in the def-chain but must move with it
// (otherwise the load looks like it has an extra use and stays outside).
static void addSatelliteMarkOps(DenseSet<Operation *> &ops, Operation *anchor) {
  for (Operation &op : *anchor->getBlock()) {
    if (&op == anchor)
      break;
    if (!isa<annotation::MarkOp>(&op))
      continue;
    for (Value operand : op.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && ops.contains(def)) {
        ops.insert(&op);
        break;
      }
    }
  }
}

// True when every use of \p op moves with the chain or already lives inside
// the if we are about to wrap.
static bool opHasOnlyLocalUses(Operation *op, const DenseSet<Operation *> &ops,
                               Operation *insertBefore) {
  Block *destBlock = insertBefore->getBlock();
  for (OpOperand &use : op->getUses()) {
    Operation *user = use.getOwner();
    if (ops.contains(user) || user == insertBefore)
      continue;
    // Uses inside ssbuffer.if are dominated once the def sits above it.
    if (insertBefore->isProperAncestor(user))
      continue;
    // Cond pieces already moved into the dest block (wrapper then).
    if (op->getBlock() != destBlock && user->getBlock() == destBlock &&
        user->isBeforeInBlock(insertBefore))
      continue;
    return false;
  }
  return true;
}

// Insertion point that keeps an already-moved cond op ahead of \p before.
static Operation *insertPointForCondOps(Operation *before,
                                        const DenseSet<Operation *> &condOps) {
  for (Operation &op : *before->getBlock()) {
    if (&op == before)
      break;
    if (condOps.contains(&op))
      return &op;
  }
  return before;
}

// Move the ssbuf cond def-chain (loads, marks, cmp, andi) to sit immediately
// above \p before. Never sinks into \p before's regions.
// FAILED: the def-chain walk broke. SUCCESS: nothing to move, or it moved.
static int moveDefChainBefore(Value cond, Operation *fromAnchor,
                              Operation *before,
                              const DenseSet<Operation *> &stayOutside) {
  DenseSet<Operation *> condOps;
  if (collectConditionDefOpsBefore(cond, fromAnchor, condOps) !=
      WRAP_SPLITTED_IF_SUCCESS)
    return WRAP_SPLITTED_IF_FAILED;
  if (condOps.empty())
    return WRAP_SPLITTED_IF_SUCCESS;
  addSatelliteMarkOps(condOps, fromAnchor);

  SmallVector<Operation *> sorted;
  for (Operation &op : *fromAnchor->getBlock()) {
    if (condOps.contains(&op))
      sorted.push_back(&op);
    if (&op == fromAnchor)
      break;
  }

  DenseSet<Operation *> moveSet;
  for (Operation *op : sorted) {
    if (stayOutside.contains(op))
      continue;
    if (opHasOnlyLocalUses(op, condOps, before))
      moveSet.insert(op);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : sorted) {
      if (!moveSet.contains(op))
        continue;
      for (OpOperand &use : op->getUses()) {
        Operation *user = use.getOwner();
        if (!condOps.contains(user) || moveSet.contains(user))
          continue;
        moveSet.erase(op);
        changed = true;
        break;
      }
    }
  }

  Operation *insertPt = before;
  if (fromAnchor->getBlock() != before->getBlock())
    insertPt = insertPointForCondOps(before, condOps);

  for (Operation *op : sorted) {
    if (!moveSet.contains(op) || op == insertPt)
      continue;
    op->moveBefore(insertPt);
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Outer wrapper already gates on the split cond, so the inner then always
// runs. Splice the then-body in front of splitIf, retarget its results to the
// then-yield values, and drop the else. The then terminator stays put: a
// yield in the parent block would cut off the copies and the counter update.
// FAILED: the then yield cannot be spliced. The caller keeps the inner if.
static int unwrapSplittedIf(scf::IfOp splitIf) {
  Block *thenBlock = splitIf.thenBlock();
  if (!thenBlock || !thenBlock->mightHaveTerminator())
    return WRAP_SPLITTED_IF_FAILED;
  auto thenYield = dyn_cast<scf::YieldOp>(thenBlock->getTerminator());
  if (!thenYield)
    return WRAP_SPLITTED_IF_FAILED;
  SmallVector<Value> thenVals(thenYield.getOperands().begin(),
                              thenYield.getOperands().end());
  if (splitIf.getNumResults() != thenVals.size())
    return WRAP_SPLITTED_IF_FAILED;
  for (Value val : thenVals) {
    if (val.getDefiningOp() == splitIf.getOperation())
      return WRAP_SPLITTED_IF_FAILED;
  }

  SmallVector<Operation *> thenOps;
  for (Operation &op : *thenBlock) {
    if (!op.hasTrait<OpTrait::IsTerminator>())
      thenOps.push_back(&op);
  }
  for (Operation *op : thenOps)
    op->moveBefore(splitIf);

  splitIf.replaceAllUsesWith(thenVals);
  splitIf.erase();
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Replace the block terminator with scf.yield of \p operands.
static void replaceTerminator(Block *block, Location loc, ValueRange operands) {
  if (block->mightHaveTerminator())
    block->getTerminator()->erase();
  OpBuilder builder(block, block->end());
  builder.create<scf::YieldOp>(loc, operands);
}

// Nearest ancestor with ssbuffer.main_loop. Used to keep a nested main
// loop's splitted_if ids out of the outer loop's kind check.
static Operation *getEnclosingMainLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isMainLoopOp(parent))
      return parent;
  }
  return nullptr;
}

// Kind is the integer ssbuffer.splitted_if id. Nested main loops are excluded.
// SUCCESS: zero ids or one shared id.
// IGNORED: two different ids. This loop is left unwrapped.
// FAILED: the attr is present but not an integer.
static int onlyOneSplittedIfKind(Operation *mainLoop) {
  IntegerAttr firstSplittedIfId;
  int result = WRAP_SPLITTED_IF_SUCCESS;
  mainLoop->walk([&](scf::IfOp ifOp) {
    if (getEnclosingMainLoop(ifOp) != mainLoop)
      return WalkResult::advance();
    if (!ifOp->hasAttr(kSplittedIf))
      return WalkResult::advance();
    // ssbuffer.splitted_if is printed as `N : i32`.
    auto splittedIfId = ifOp->getAttrOfType<IntegerAttr>(kSplittedIf);
    if (!splittedIfId) {
      LDBG("ssbuffer.splitted_if is not an integer id.\n");
      result = WRAP_SPLITTED_IF_FAILED;
      return WalkResult::interrupt();
    }
    if (!firstSplittedIfId) {
      firstSplittedIfId = splittedIfId;
      return WalkResult::advance();
    }
    if (firstSplittedIfId.getInt() == splittedIfId.getInt())
      return WalkResult::advance();
    LDBG("Skip wrap: main loop has more than one splitted_if id.\n");
    result = WRAP_SPLITTED_IF_IGNORED;
    return WalkResult::interrupt();
  });
  return result;
}

// ssbuffer.while_arg ops update the values tested by scf.condition.
// The wrapper else can clone that chain only when every in-if operand is
// itself a while_arg. A value produced by the splitted compute has no def
// on the split-false path.
// IGNORED: the chain is not cloneable. SUCCESS: it is.
static int whileArgChainCanClone(scf::IfOp ssbufIf) {
  if (ssbufIf.getThenRegion().empty())
    return WRAP_SPLITTED_IF_IGNORED;
  int result = WRAP_SPLITTED_IF_SUCCESS;
  ssbufIf.getThenRegion().walk([&](Operation *op) {
    if (!op->hasAttr(kWhileArg))
      return WalkResult::advance();
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !ssbufIf->isProperAncestor(def))
        continue;
      if (!def->hasAttr(kWhileArg)) {
        result = WRAP_SPLITTED_IF_IGNORED;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

// Topological order of the while_arg chain. FAILED when the chain has a cycle.
static int orderWhileArgOps(ArrayRef<Operation *> chain,
                            SmallVector<Operation *> &ordered) {
  DenseSet<Operation *> pending(chain.begin(), chain.end());
  while (!pending.empty()) {
    bool progress = false;
    for (Operation *op : chain) {
      if (!pending.contains(op))
        continue;
      bool ready = true;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (def && pending.contains(def)) {
          ready = false;
          break;
        }
      }
      if (!ready)
        continue;
      ordered.push_back(op);
      pending.erase(op);
      progress = true;
    }
    if (!progress)
      return WRAP_SPLITTED_IF_FAILED;
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Clone the while_arg chain into the wrapper else. Yield cloned results in
// the slots the ssbuffer.if then uses for those values. Other slots keep the
// ssbuffer.if else passthrough.
// FAILED: the then yield does not match, or the chain cannot be ordered.
static int buildWhileWrapperElse(scf::IfOp ssbufIf, Block *elseBlock,
                                 ArrayRef<Value> elsePassthrough,
                                 Location loc) {
  auto thenYield =
      dyn_cast<scf::YieldOp>(ssbufIf.thenBlock()->getTerminator());
  if (!thenYield || thenYield.getNumOperands() != elsePassthrough.size() ||
      thenYield.getNumOperands() != ssbufIf.getNumResults())
    return WRAP_SPLITTED_IF_FAILED;

  SmallVector<Operation *> chain;
  ssbufIf.getThenRegion().walk([&](Operation *op) {
    if (op->hasAttr(kWhileArg))
      chain.push_back(op);
  });
  SmallVector<Operation *> ordered;
  if (orderWhileArgOps(chain, ordered) != WRAP_SPLITTED_IF_SUCCESS)
    return WRAP_SPLITTED_IF_FAILED;

  OpBuilder elseBuilder(elseBlock, elseBlock->end());
  if (elseBlock->mightHaveTerminator())
    elseBuilder.setInsertionPoint(elseBlock->getTerminator());
  IRMapping mapping;
  for (Operation *op : ordered) {
    Operation *cloned = elseBuilder.clone(*op, mapping);
    elseBuilder.setInsertionPointAfter(cloned);
  }

  SmallVector<Value> elseVals(elsePassthrough.begin(), elsePassthrough.end());
  for (auto [idx, thenVal] : llvm::enumerate(thenYield.getOperands())) {
    if (mapping.contains(thenVal))
      elseVals[idx] = mapping.lookup(thenVal);
  }
  replaceTerminator(elseBlock, loc, elseVals);
  LDBG("Cloned " << ordered.size()
                 << " while_arg ops into the wrapper else.\n");
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Wrapper else for scf.for: yield the ssbuffer.if else values, with the
// block counter replaced by counter+step so a false split cond still
// consumes the iteration.
// FAILED: the counter recorded for this if is missing or has no step.
static int buildForWrapperElse(scf::ForOp forOp,
                               ControlFlowConditionInfo *info,
                               scf::IfOp ssbufIf, Block *elseBlock,
                               ArrayRef<Value> elsePassthrough, Location loc) {
  if (!info->cntArgs.count(ssbufIf))
    return WRAP_SPLITTED_IF_FAILED;
  Value counter = info->cntArgs[ssbufIf];
  Value step = forOp.getStep();
  if (!counter || !step)
    return WRAP_SPLITTED_IF_FAILED;

  OpBuilder elseBuilder(elseBlock, elseBlock->end());
  if (elseBlock->mightHaveTerminator())
    elseBuilder.setInsertionPoint(elseBlock->getTerminator());
  Value inc = elseBuilder.create<arith::AddIOp>(loc, counter, step);
  SmallVector<Value> elseVals(elsePassthrough.begin(), elsePassthrough.end());
  bool replacedCounter = false;
  for (Value &val : elseVals) {
    if (val == counter) {
      val = inc;
      replacedCounter = true;
    }
  }
  if (!replacedCounter && !elseVals.empty() &&
      elseVals.back().getType() == inc.getType())
    elseVals.back() = inc;
  if (elseVals.size() != ssbufIf.getNumResults())
    return WRAP_SPLITTED_IF_FAILED;
  replaceTerminator(elseBlock, loc, elseVals);
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Fill the wrapper else. for adds counter+step; while clones the while_arg
// chain. FAILED when \p loopOp is neither, or the chosen builder fails.
static int buildWrapperElse(Operation *loopOp, ControlFlowConditionInfo *info,
                            scf::IfOp ssbufIf, Block *elseBlock,
                            ArrayRef<Value> elsePassthrough, Location loc) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
    return buildForWrapperElse(forOp, info, ssbufIf, elseBlock, elsePassthrough,
                               loc);
  if (isa<scf::WhileOp>(loopOp))
    return buildWhileWrapperElse(ssbufIf, elseBlock, elsePassthrough, loc);
  return WRAP_SPLITTED_IF_FAILED;
}

// The wrapper else has to advance the loop when the split cond is false.
// for needs the block counter. while needs a cloneable while_arg chain.
// IGNORED: this if has no counter, or the while chain is not cloneable.
// FAILED: a for that claims a counter has a null counter or no step.
static int checkLoopCanAdvance(scf::IfOp ssbufIf, Operation *loopOp,
                               ControlFlowConditionInfo *info) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    if (!info->cntArgs.count(ssbufIf)) {
      LDBG("Skip wrap: ssbuffer.if has no cntArgs.\n");
      return WRAP_SPLITTED_IF_IGNORED;
    }
    if (!info->cntArgs[ssbufIf] || !forOp.getStep())
      return WRAP_SPLITTED_IF_FAILED;
    return WRAP_SPLITTED_IF_SUCCESS;
  }
  if (isa<scf::WhileOp>(loopOp)) {
    int cloneRc = whileArgChainCanClone(ssbufIf);
    if (cloneRc != WRAP_SPLITTED_IF_SUCCESS) {
      LDBG("Skip wrap: while update chain uses a value inside ssbuffer.if.\n");
      return cloneRc;
    }
    return WRAP_SPLITTED_IF_SUCCESS;
  }
  return WRAP_SPLITTED_IF_FAILED;
}

// Gates that run before any IR change. On SUCCESS, \p elsePassthrough holds
// the ssbuffer.if else yield operands for the wrapper else.
// IGNORED: this if is not a wrap candidate. FAILED: the if is malformed
// (no condition) or the loop-advance check hit an unexpected case.
static int prepareSsbufWrap(scf::IfOp ssbufIf, Operation *loopOp,
                            ControlFlowConditionInfo *info, scf::IfOp &splitIf,
                            SmallVector<Value> &elsePassthrough) {
  if (!isa<scf::ForOp, scf::WhileOp>(loopOp))
    return WRAP_SPLITTED_IF_IGNORED;
  splitIf = getUniqueFirstLevelSplittedIf(ssbufIf);
  if (!splitIf)
    return WRAP_SPLITTED_IF_IGNORED;
  if (!isOnlySplittedIf(ssbufIf, splitIf)) {
    LDBG("Skip wrap: ssbuffer.if contains more than one splitted_if.\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  if (!computeOnlyInSplitThen(ssbufIf, splitIf)) {
    LDBG("Skip wrap: real tensor compute outside the splitted_if then.\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  Block *ssbufElse = ssbufIf.elseBlock();
  if (!ssbufElse || !ssbufElse->mightHaveTerminator()) {
    LDBG("Skip wrap: ssbuffer.if has no else yield.\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  auto ssbufElseYield = dyn_cast<scf::YieldOp>(ssbufElse->getTerminator());
  if (!ssbufElseYield)
    return WRAP_SPLITTED_IF_IGNORED;
  if (!splitIf.getCondition() || !ssbufIf.getCondition())
    return WRAP_SPLITTED_IF_FAILED;
  int advanceRc = checkLoopCanAdvance(ssbufIf, loopOp, info);
  if (advanceRc != WRAP_SPLITTED_IF_SUCCESS)
    return advanceRc;
  elsePassthrough.assign(ssbufElseYield.getOperands().begin(),
                         ssbufElseYield.getOperands().end());
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Hoist the split cond above ssbufIf, then pack the ssbuf cond against the if
// so both move together into the wrapper. Split-cond ops stay outside.
// IGNORED: the split cond cannot be hoisted. Nothing has moved yet.
// FAILED: a def-chain walk broke. A pack failure after a successful hoist
// has already moved the split cond, so the caller rolls the pipeline back.
static int hoistSplitCondAndPackSsbufCond(scf::IfOp ssbufIf,
                                          scf::IfOp splitIf) {
  int hoistRc = hoistSplittedIfCondition(ssbufIf, splitIf);
  if (hoistRc != WRAP_SPLITTED_IF_SUCCESS) {
    if (hoistRc == WRAP_SPLITTED_IF_IGNORED)
      LDBG("Skip wrap: cannot hoist splitted_if condition.\n");
    return hoistRc;
  }
  DenseSet<Operation *> splitCondOps;
  if (collectConditionDefOpsBefore(splitIf.getCondition(),
                                   ssbufIf.getOperation(),
                                   splitCondOps) != WRAP_SPLITTED_IF_SUCCESS)
    return WRAP_SPLITTED_IF_FAILED;
  addSatelliteMarkOps(splitCondOps, ssbufIf.getOperation());
  int moveRc =
      moveDefChainBefore(ssbufIf.getCondition(), ssbufIf.getOperation(),
                         ssbufIf.getOperation(), splitCondOps);
  if (moveRc != WRAP_SPLITTED_IF_SUCCESS) {
    LDBG("Skip wrap: cannot pack ssbuffer.if cond against the if.\n");
    return moveRc;
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Build the outer scf.if on the split cond, tag it ssbuffer.splitted_if, and
// move ssbufIf into its then. Packing the ssbuf cond into that then is
// best-effort: a failure there keeps the cond ops outside and still returns
// SUCCESS. FAILED only when the new if has no then or else block.
static int createSplitCondWrapper(scf::IfOp ssbufIf, scf::IfOp splitIf,
                                  scf::IfOp &wrapper) {
  Location loc = ssbufIf.getLoc();
  OpBuilder builder(ssbufIf);
  SmallVector<Type> resultTypes(ssbufIf->getResultTypes());
  wrapper = builder.create<scf::IfOp>(loc, resultTypes, splitIf.getCondition(),
                                      /*withElse=*/true);
  if (Attribute splitAttr = splitIf->getAttr(kSplittedIf))
    wrapper->setAttr(kSplittedIf, splitAttr);

  Block *thenBlock = wrapper.thenBlock();
  Block *elseBlock = wrapper.elseBlock();
  if (!thenBlock || !elseBlock)
    return WRAP_SPLITTED_IF_FAILED;
  if (thenBlock->mightHaveTerminator())
    ssbufIf->moveBefore(thenBlock->getTerminator());
  else
    ssbufIf->moveBefore(thenBlock, thenBlock->end());

  DenseSet<Operation *> splitCondOps;
  if (collectConditionDefOpsBefore(splitIf.getCondition(),
                                   wrapper.getOperation(),
                                   splitCondOps) != WRAP_SPLITTED_IF_SUCCESS) {
    LDBG("Keep ssbuffer.if cond ops outside wrapper then.\n");
    return WRAP_SPLITTED_IF_SUCCESS;
  }
  if (moveDefChainBefore(ssbufIf.getCondition(), wrapper.getOperation(),
                         ssbufIf.getOperation(),
                         splitCondOps) != WRAP_SPLITTED_IF_SUCCESS)
    LDBG("Keep ssbuffer.if cond ops outside wrapper then.\n");
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Uses of ssbufIf results that live outside the wrapper read the wrapper
// results. Uses inside the wrapper stay on the inner results.
static void replaceUsesOutsideWrapper(scf::IfOp ssbufIf, scf::IfOp wrapper) {
  for (auto [innerRes, outerRes] :
       llvm::zip(ssbufIf.getResults(), wrapper.getResults())) {
    innerRes.replaceUsesWithIf(outerRes, [&](OpOperand &use) {
      return !wrapper->isProperAncestor(use.getOwner());
    });
  }
}

// Wrap one ssbuffer.if.
// IGNORED: it is not a wrap candidate, so the caller leaves it and continues.
// FAILED: an unexpected error. The caller stops the walk and falls back.
static int wrapSsbufIf(scf::IfOp ssbufIf, Operation *loopOp,
                       ControlFlowConditionInfo *info) {
  scf::IfOp splitIf;
  SmallVector<Value> elsePassthrough;
  int prepareRc =
      prepareSsbufWrap(ssbufIf, loopOp, info, splitIf, elsePassthrough);
  if (prepareRc != WRAP_SPLITTED_IF_SUCCESS)
    return prepareRc;
  int hoistRc = hoistSplitCondAndPackSsbufCond(ssbufIf, splitIf);
  if (hoistRc != WRAP_SPLITTED_IF_SUCCESS)
    return hoistRc;

  scf::IfOp wrapper;
  if (createSplitCondWrapper(ssbufIf, splitIf, wrapper) !=
      WRAP_SPLITTED_IF_SUCCESS)
    return WRAP_SPLITTED_IF_FAILED;
  // Splice failure keeps the inner if. The wrapper is already in place, so
  // this is not a pass failure.
  if (unwrapSplittedIf(splitIf) != WRAP_SPLITTED_IF_SUCCESS)
    LDBG("Keep inner splitted_if; cannot splice then-body.\n");

  Location loc = ssbufIf.getLoc();
  replaceTerminator(wrapper.thenBlock(), loc, ssbufIf.getResults());
  if (buildWrapperElse(loopOp, info, ssbufIf, wrapper.elseBlock(),
                       elsePassthrough, loc) != WRAP_SPLITTED_IF_SUCCESS)
    return WRAP_SPLITTED_IF_FAILED;
  replaceUsesOutsideWrapper(ssbufIf, wrapper);
  LDBG("Wrapped ssbuffer.if with hoisted splitted_if.\n");
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Walk main loops. An ssbuffer.if is recognized only by kIf ("ssbuffer.if").
// Per-if IGNORED and a loop with more than one splitted_if kind are expected
// and do not stop the walk. Stopping the walk records the return code:
// IGNORED selects ERRCODE_IGNORED, anything else selects ERRCODE_FAILED.
void WrapSplittedIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  LDBG("Before WrapSplittedIf:\n" << module << "\n");
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  int fallbackRc = WRAP_SPLITTED_IF_SUCCESS;
  WalkResult walkResult = module.walk([&](Operation *loopOp) -> WalkResult {
    if (!isMainLoopOp(loopOp))
      return WalkResult::advance();

    int kindRc = onlyOneSplittedIfKind(loopOp);
    // Expected: more than one splitted_if kind. Leave this loop unwrapped.
    if (kindRc == WRAP_SPLITTED_IF_IGNORED)
      return WalkResult::advance();
    if (kindRc != WRAP_SPLITTED_IF_SUCCESS) {
      fallbackRc = kindRc;
      return WalkResult::interrupt();
    }

    SmallVector<scf::IfOp> ssbufIfs;
    loopOp->walk([&](scf::IfOp ifOp) {
      if (!ifOp->hasAttr(kIf))
        return;
      ssbufIfs.push_back(ifOp);
    });
    for (scf::IfOp ssbufIf : ssbufIfs) {
      int wrapRc = wrapSsbufIf(ssbufIf, loopOp, info);
      // Expected: this if is not a wrap candidate. Leave it unchanged.
      if (wrapRc == WRAP_SPLITTED_IF_IGNORED ||
          wrapRc == WRAP_SPLITTED_IF_SUCCESS)
        continue;
      LDBG("wrapSsbufIf failed.\n");
      fallbackRc = wrapRc;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    if (fallbackRc == WRAP_SPLITTED_IF_IGNORED)
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    else
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }
  LDBG("After WrapSplittedIf:\n" << module << "\n");
}

namespace mlir {
namespace triton {
// Pipeline code constructs WrapSplittedIfPass directly so it can call
// setConditionInfo. This factory stays for the same shape as the other passes.
std::unique_ptr<OperationPass<ModuleOp>> createWrapSplittedIfPass() {
  return std::make_unique<WrapSplittedIfPass>();
}
} // namespace triton
} // namespace mlir
