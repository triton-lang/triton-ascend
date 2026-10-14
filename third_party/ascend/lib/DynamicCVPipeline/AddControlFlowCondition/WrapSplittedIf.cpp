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

class WrapSplittedIfPass
    : public PassWrapper<WrapSplittedIfPass, OperationPass<ModuleOp>> {
public:
  WrapSplittedIfPass() = default;

  void runOnOperation() override;

  void setConditionInfo(ControlFlowConditionInfo *info) { this->info = info; }

  llvm::StringRef getArgument() const override { return "wrap-splitted-if"; }

private:
  ControlFlowConditionInfo *info = nullptr;
};

// Exactly one first-level then-block ssbuffer.splitted_if. Nested ifs and
// sibling ifs without the attr are ignored. Two or more: skip.
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

static int collectConditionDefOpsInside(Value value, scf::IfOp ssbufIf,
                                        DenseSet<Operation *> &ops) {
  if (!value)
    return 0;
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return 0;
  if (!ssbufIf->isProperAncestor(defOp))
    return 0;
  if (!ops.insert(defOp).second)
    return 0;
  for (Value operand : defOp->getOperands()) {
    if (collectConditionDefOpsInside(operand, ssbufIf, ops) != 0)
      return -1;
  }
  return 0;
}

// True if \p def properly dominates \p user.
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

static LogicalResult hoistSplittedIfCondition(scf::IfOp ssbufIf,
                                              scf::IfOp splittedIf) {
  Value cond = splittedIf.getCondition();
  if (!cond)
    return failure();

  DenseSet<Operation *> condOps;
  if (collectConditionDefOpsInside(cond, ssbufIf, condOps) != 0)
    return failure();
  if (condOps.empty())
    return success();
  if (!canHoistConditionOps(ssbufIf, condOps))
    return failure();

  Block *thenBlock = ssbufIf.thenBlock();
  if (!thenBlock)
    return failure();
  SmallVector<Operation *> sorted;
  for (Operation &op : *thenBlock) {
    if (condOps.contains(&op))
      sorted.push_back(&op);
  }
  if (sorted.size() != condOps.size())
    return failure();
  for (Operation *op : sorted)
    op->moveBefore(ssbufIf);
  return success();
}

static int collectConditionDefOpsBefore(Value value, Operation *anchor,
                                        DenseSet<Operation *> &ops) {
  if (!value || !anchor)
    return 0;
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return 0;
  if (defOp->getBlock() != anchor->getBlock() ||
      !defOp->isBeforeInBlock(anchor))
    return 0;
  if (!ops.insert(defOp).second)
    return 0;
  for (Value operand : defOp->getOperands()) {
    if (collectConditionDefOpsBefore(operand, anchor, ops) != 0)
      return -1;
  }
  return 0;
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

// Move ssbuf cond def-chain (loads + marks + cmp/andi) to sit immediately
// above \p before. Never sinks into \p before's regions.
static LogicalResult
moveDefChainBefore(Value cond, Operation *fromAnchor, Operation *before,
                   const DenseSet<Operation *> &stayOutside) {
  DenseSet<Operation *> condOps;
  if (collectConditionDefOpsBefore(cond, fromAnchor, condOps) != 0)
    return failure();
  if (condOps.empty())
    return success();
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
  return success();
}

// Outer wrapper already gates on the split cond, so the inner then always
// runs. Splice then-body into the parent; replace %if#N with then-yield
// values; drop else. Never move the then terminator — a yield in the
// parent would cut off copies / counter updates and hang.
static LogicalResult unwrapSplittedIf(scf::IfOp splitIf) {
  Block *thenBlock = splitIf.thenBlock();
  if (!thenBlock || !thenBlock->mightHaveTerminator())
    return failure();
  auto thenYield = dyn_cast<scf::YieldOp>(thenBlock->getTerminator());
  if (!thenYield)
    return failure();
  SmallVector<Value> thenVals(thenYield.getOperands().begin(),
                              thenYield.getOperands().end());
  if (splitIf.getNumResults() != thenVals.size())
    return failure();
  for (Value val : thenVals) {
    if (val.getDefiningOp() == splitIf.getOperation())
      return failure();
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
  return success();
}

static void replaceTerminator(Block *block, Location loc, ValueRange operands) {
  if (block->mightHaveTerminator())
    block->getTerminator()->erase();
  OpBuilder builder(block, block->end());
  builder.create<scf::YieldOp>(loc, operands);
}

static Operation *getEnclosingMainLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isMainLoopOp(parent))
      return parent;
  }
  return nullptr;
}

// ssbuffer.while_arg ops are the per-block update of scf.condition args.
// A copy in the wrapper else is valid when every in-if operand is itself
// part of that chain. A value produced by the splitted compute has no def
// on the split-false path.
static LogicalResult whileArgChainCanClone(scf::IfOp ssbufIf) {
  if (ssbufIf.getThenRegion().empty())
    return failure();
  LogicalResult result = success();
  ssbufIf.getThenRegion().walk([&](Operation *op) {
    if (!op->hasAttr(kWhileArg))
      return WalkResult::advance();
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !ssbufIf->isProperAncestor(def))
        continue;
      if (!def->hasAttr(kWhileArg)) {
        result = failure();
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

static LogicalResult orderWhileArgOps(ArrayRef<Operation *> chain,
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
      return failure();
  }
  return success();
}

// Clone the while_arg chain into the wrapper else and yield its results in
// the slots the ssbuffer.if then uses for those values. Other slots keep the
// ssbuffer.if else passthrough.
static LogicalResult buildWhileWrapperElse(scf::IfOp ssbufIf, Block *elseBlock,
                                           ArrayRef<Value> elsePassthrough,
                                           Location loc) {
  auto thenYield =
      dyn_cast<scf::YieldOp>(ssbufIf.thenBlock()->getTerminator());
  if (!thenYield || thenYield.getNumOperands() != elsePassthrough.size() ||
      thenYield.getNumOperands() != ssbufIf.getNumResults())
    return failure();

  SmallVector<Operation *> chain;
  ssbufIf.getThenRegion().walk([&](Operation *op) {
    if (op->hasAttr(kWhileArg))
      chain.push_back(op);
  });
  SmallVector<Operation *> ordered;
  if (failed(orderWhileArgOps(chain, ordered)))
    return failure();

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
  return success();
}

static LogicalResult buildForWrapperElse(scf::ForOp forOp,
                                         ControlFlowConditionInfo *info,
                                         scf::IfOp ssbufIf, Block *elseBlock,
                                         ArrayRef<Value> elsePassthrough,
                                         Location loc) {
  if (!info->cntArgs.count(ssbufIf))
    return failure();
  Value counter = info->cntArgs[ssbufIf];
  Value step = forOp.getStep();
  if (!counter || !step)
    return failure();

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
    return failure();
  replaceTerminator(elseBlock, loc, elseVals);
  return success();
}

static LogicalResult buildWrapperElse(Operation *loopOp,
                                      ControlFlowConditionInfo *info,
                                      scf::IfOp ssbufIf, Block *elseBlock,
                                      ArrayRef<Value> elsePassthrough,
                                      Location loc) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
    return buildForWrapperElse(forOp, info, ssbufIf, elseBlock, elsePassthrough,
                               loc);
  if (isa<scf::WhileOp>(loopOp))
    return buildWhileWrapperElse(ssbufIf, elseBlock, elsePassthrough, loc);
  return failure();
}

// For needs a block counter. While needs a while_arg chain whose operands
// are available on the split-false path. failure means leave this if alone.
static LogicalResult checkLoopCanAdvance(scf::IfOp ssbufIf, Operation *loopOp,
                                         ControlFlowConditionInfo *info) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    if (!info->cntArgs.count(ssbufIf)) {
      LDBG("Skip wrap: ssbuffer.if has no cntArgs.\n");
      return failure();
    }
    if (!info->cntArgs[ssbufIf] || !forOp.getStep())
      return failure();
    return success();
  }
  if (isa<scf::WhileOp>(loopOp)) {
    if (failed(whileArgChainCanClone(ssbufIf))) {
      LDBG("Skip wrap: while update chain uses a value inside ssbuffer.if.\n");
      return failure();
    }
    return success();
  }
  return failure();
}

// failure means skip. No IR change yet.
static LogicalResult prepareSsbufWrap(scf::IfOp ssbufIf, Operation *loopOp,
                                      ControlFlowConditionInfo *info,
                                      scf::IfOp &splitIf,
                                      SmallVector<Value> &elsePassthrough) {
  if (!isa<scf::ForOp, scf::WhileOp>(loopOp))
    return failure();
  splitIf = getUniqueFirstLevelSplittedIf(ssbufIf);
  if (!splitIf)
    return failure();
  if (!isOnlySplittedIf(ssbufIf, splitIf)) {
    LDBG("Skip wrap: ssbuffer.if contains more than one splitted_if.\n");
    return failure();
  }
  if (!computeOnlyInSplitThen(ssbufIf, splitIf)) {
    LDBG("Skip wrap: real tensor compute outside the splitted_if then.\n");
    return failure();
  }
  Block *ssbufElse = ssbufIf.elseBlock();
  if (!ssbufElse || !ssbufElse->mightHaveTerminator()) {
    LDBG("Skip wrap: ssbuffer.if has no else yield.\n");
    return failure();
  }
  auto ssbufElseYield = dyn_cast<scf::YieldOp>(ssbufElse->getTerminator());
  if (!ssbufElseYield)
    return failure();
  if (!splitIf.getCondition() || !ssbufIf.getCondition())
    return failure();
  if (failed(checkLoopCanAdvance(ssbufIf, loopOp, info)))
    return failure();
  elsePassthrough.assign(ssbufElseYield.getOperands().begin(),
                         ssbufElseYield.getOperands().end());
  return success();
}

static LogicalResult hoistSplitCondAndPackSsbufCond(scf::IfOp ssbufIf,
                                                    scf::IfOp splitIf) {
  if (failed(hoistSplittedIfCondition(ssbufIf, splitIf))) {
    LDBG("Skip wrap: cannot hoist splitted_if condition.\n");
    return failure();
  }
  DenseSet<Operation *> splitCondOps;
  if (collectConditionDefOpsBefore(splitIf.getCondition(),
                                   ssbufIf.getOperation(), splitCondOps) != 0)
    return failure();
  addSatelliteMarkOps(splitCondOps, ssbufIf.getOperation());
  if (failed(moveDefChainBefore(ssbufIf.getCondition(), ssbufIf.getOperation(),
                                ssbufIf.getOperation(), splitCondOps))) {
    LDBG("Skip wrap: cannot pack ssbuffer.if cond against the if.\n");
    return failure();
  }
  return success();
}

static LogicalResult createSplitCondWrapper(scf::IfOp ssbufIf,
                                            scf::IfOp splitIf,
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
    return failure();
  if (thenBlock->mightHaveTerminator())
    ssbufIf->moveBefore(thenBlock->getTerminator());
  else
    ssbufIf->moveBefore(thenBlock, thenBlock->end());

  DenseSet<Operation *> splitCondOps;
  if (collectConditionDefOpsBefore(splitIf.getCondition(),
                                   wrapper.getOperation(), splitCondOps) != 0) {
    LDBG("Keep ssbuffer.if cond ops outside wrapper then.\n");
    return success();
  }
  if (failed(moveDefChainBefore(ssbufIf.getCondition(), wrapper.getOperation(),
                                ssbufIf.getOperation(), splitCondOps)))
    LDBG("Keep ssbuffer.if cond ops outside wrapper then.\n");
  return success();
}

static void replaceUsesOutsideWrapper(scf::IfOp ssbufIf, scf::IfOp wrapper) {
  for (auto [innerRes, outerRes] :
       llvm::zip(ssbufIf.getResults(), wrapper.getResults())) {
    innerRes.replaceUsesWithIf(outerRes, [&](OpOperand &use) {
      return !wrapper->isProperAncestor(use.getOwner());
    });
  }
}

static LogicalResult wrapSsbufIf(scf::IfOp ssbufIf, Operation *loopOp,
                                 ControlFlowConditionInfo *info) {
  scf::IfOp splitIf;
  SmallVector<Value> elsePassthrough;
  if (failed(prepareSsbufWrap(ssbufIf, loopOp, info, splitIf, elsePassthrough)))
    return success();
  if (failed(hoistSplitCondAndPackSsbufCond(ssbufIf, splitIf)))
    return success();

  scf::IfOp wrapper;
  if (failed(createSplitCondWrapper(ssbufIf, splitIf, wrapper)))
    return failure();
  if (failed(unwrapSplittedIf(splitIf)))
    LDBG("Keep inner splitted_if; cannot splice then-body.\n");

  Location loc = ssbufIf.getLoc();
  replaceTerminator(wrapper.thenBlock(), loc, ssbufIf.getResults());
  if (failed(buildWrapperElse(loopOp, info, ssbufIf, wrapper.elseBlock(),
                              elsePassthrough, loc)))
    return failure();
  replaceUsesOutsideWrapper(ssbufIf, wrapper);
  LDBG("Wrapped ssbuffer.if with hoisted splitted_if.\n");
  return success();
}

void WrapSplittedIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  LDBG("Before WrapSplittedIf:\n" << module << "\n");
  if (CVPipeline::hasFallbackAttr(module) || !info) {
    return;
  }

  WalkResult walkResult = module.walk([&](Operation *loopOp) -> WalkResult {
    if (!isMainLoopOp(loopOp))
      return WalkResult::advance();

    SmallVector<scf::IfOp> ssbufIfs;
    loopOp->walk([&](scf::IfOp ifOp) {
      if (!ifOp->hasAttr(kIf))
        return;
      if (getEnclosingMainLoop(ifOp) != loopOp)
        return;
      ssbufIfs.push_back(ifOp);
    });
    for (scf::IfOp ssbufIf : ssbufIfs) {
      if (failed(wrapSsbufIf(ssbufIf, loopOp, info))) {
        LDBG("wrapSsbufIf failed.\n");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  LDBG("After WrapSplittedIf:\n" << module << "\n");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>>
createWrapSplittedIfPass(ControlFlowConditionInfo *info) {
  auto pass = std::make_unique<WrapSplittedIfPass>();
  pass->setConditionInfo(info);
  return pass;
}
} // namespace triton
} // namespace mlir
