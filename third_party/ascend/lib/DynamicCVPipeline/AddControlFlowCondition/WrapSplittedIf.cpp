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
  if (defOp->getBlock() != anchor->getBlock() || !defOp->isBeforeInBlock(anchor))
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

static LogicalResult wrapSsbufIf(scf::IfOp ssbufIf, scf::ForOp forOp,
                                 ControlFlowConditionInfo *info) {
  scf::IfOp splitIf = getUniqueFirstLevelSplittedIf(ssbufIf);
  if (!splitIf)
    return success();
  if (!info->cntArgs.count(ssbufIf)) {
    LDBG("Skip wrap: ssbuffer.if has no cntArgs.\n");
    return success();
  }
  Block *ssbufElse = ssbufIf.elseBlock();
  if (!ssbufElse || !ssbufElse->mightHaveTerminator()) {
    LDBG("Skip wrap: ssbuffer.if has no else yield.\n");
    return success();
  }
  auto ssbufElseYield = dyn_cast<scf::YieldOp>(ssbufElse->getTerminator());
  if (!ssbufElseYield)
    return success();

  Value splitCond = splitIf.getCondition();
  Value ssbufCond = ssbufIf.getCondition();
  Value counter = info->cntArgs[ssbufIf];
  Value step = forOp.getStep();
  Attribute splitAttr = splitIf->getAttr(kSplittedIf);
  if (!splitCond || !ssbufCond || !counter || !step)
    return success();

  SmallVector<Value> elsePassthrough(ssbufElseYield.getOperands().begin(),
                                     ssbufElseYield.getOperands().end());

  if (failed(hoistSplittedIfCondition(ssbufIf, splitIf))) {
    LDBG("Skip wrap: cannot hoist splitted_if condition.\n");
    return success();
  }

  // Split cond now sits immediately before ssbuffer.if. Pack the ssbuf cond
  // (loads + marks + cmp/andi) against ssbuffer.if so it becomes the if's
  // condition chain, not leftover ops above the split cond.
  DenseSet<Operation *> splitCondOps;
  if (collectConditionDefOpsBefore(splitCond, ssbufIf.getOperation(),
                                   splitCondOps) != 0)
    return success();
  addSatelliteMarkOps(splitCondOps, ssbufIf.getOperation());
  if (failed(moveDefChainBefore(ssbufCond, ssbufIf.getOperation(),
                                ssbufIf.getOperation(), splitCondOps))) {
    LDBG("Skip wrap: cannot pack ssbuffer.if cond against the if.\n");
    return success();
  }

  Location loc = ssbufIf.getLoc();
  OpBuilder builder(ssbufIf);
  SmallVector<Type> resultTypes(ssbufIf->getResultTypes());
  scf::IfOp wrapper =
      builder.create<scf::IfOp>(loc, resultTypes, splitCond, /*withElse=*/true);
  if (splitAttr)
    wrapper->setAttr(kSplittedIf, splitAttr);

  Block *thenBlock = wrapper.thenBlock();
  Block *elseBlock = wrapper.elseBlock();
  if (!thenBlock || !elseBlock)
    return failure();

  if (thenBlock->mightHaveTerminator())
    ssbufIf->moveBefore(thenBlock->getTerminator());
  else
    ssbufIf->moveBefore(thenBlock, thenBlock->end());

  splitCondOps.clear();
  if (collectConditionDefOpsBefore(splitCond, wrapper.getOperation(),
                                   splitCondOps) != 0) {
    LDBG("Keep ssbuffer.if cond ops outside wrapper then.\n");
  } else if (failed(moveDefChainBefore(ssbufCond, wrapper.getOperation(),
                                       ssbufIf.getOperation(), splitCondOps))) {
    LDBG("Keep ssbuffer.if cond ops outside wrapper then.\n");
  }

  if (failed(unwrapSplittedIf(splitIf))) {
    LDBG("Keep inner splitted_if; cannot splice then-body.\n");
  }

  replaceTerminator(thenBlock, loc, ssbufIf.getResults());

  OpBuilder elseBuilder(elseBlock, elseBlock->end());
  if (elseBlock->mightHaveTerminator())
    elseBuilder.setInsertionPoint(elseBlock->getTerminator());
  Value inc = elseBuilder.create<arith::AddIOp>(loc, counter, step);
  SmallVector<Value> elseVals = elsePassthrough;
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
  if (elseVals.size() != resultTypes.size())
    return failure();
  replaceTerminator(elseBlock, loc, elseVals);

  for (auto [innerRes, outerRes] :
       llvm::zip(ssbufIf.getResults(), wrapper.getResults())) {
    innerRes.replaceUsesWithIf(outerRes, [&](OpOperand &use) {
      return !wrapper->isProperAncestor(use.getOwner());
    });
  }

  LDBG("Wrapped ssbuffer.if with hoisted splitted_if.\n");
  return success();
}

void WrapSplittedIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module) || !info)
    return;

  LDBG("Enter WrapSplittedIf pass.\n");
  WalkResult walkResult = module.walk([&](scf::ForOp forOp) -> WalkResult {
    if (!forOp->hasAttr(kMainLoop))
      return WalkResult::advance();

    SmallVector<scf::IfOp> ssbufIfs;
    forOp.walk([&](scf::IfOp ifOp) {
      if (ifOp->hasAttr(kIf))
        ssbufIfs.push_back(ifOp);
    });
    for (scf::IfOp ssbufIf : ssbufIfs) {
      if (failed(wrapSsbufIf(ssbufIf, forOp, info))) {
        LDBG("wrapSsbufIf failed.\n");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  LDBG("Exit WrapSplittedIf pass.\n");
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
