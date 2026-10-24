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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/SplittedIfWrapSsbufIf.h"
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

static constexpr const char *DEBUG_TYPE = "SplittedIfWrapSsbufIfPass";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
  })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

// 0 success, -1 failure, 2 ignored.
static constexpr int WRAP_SPLITTED_IF_SUCCESS = 0;
static constexpr int WRAP_SPLITTED_IF_FAILED = -1;
static constexpr int WRAP_SPLITTED_IF_IGNORED = 2;

// The one splitted_if directly in the then block. Null if there is not exactly
// one.
static scf::IfOp getUniqueFirstLevelSplittedIf(scf::IfOp ssbufIf) {
  if (!ssbufIf || !ssbufIf.thenBlock()) {
    return nullptr;
  }
  scf::IfOp found;
  for (Operation &op : *ssbufIf.thenBlock()) {
    auto innerIf = dyn_cast<scf::IfOp>(&op);
    if (!innerIf || !innerIf->hasAttr(kSplittedIf)) {
      continue;
    }
    if (found) {
      return nullptr;
    }
    found = innerIf;
  }
  return found;
}

// True when splittedIf is the only splitted_if under ssbufIf.
static bool isOnlySplittedIf(scf::IfOp ssbufIf, scf::IfOp splittedIf) {
  bool extra = false;
  ssbufIf.walk([&](scf::IfOp ifOp) {
    if (ifOp == ssbufIf || !ifOp->hasAttr(kSplittedIf)) {
      return WalkResult::advance();
    }
    if (ifOp == splittedIf) {
      return WalkResult::advance();
    }
    extra = true;
    return WalkResult::interrupt();
  });
  return !extra;
}

// True when op is in splittedIf's then block.
static bool opIsInSplitThen(Operation *op, scf::IfOp splittedIf) {
  Block *thenBlock = splittedIf.thenBlock();
  if (!thenBlock) {
    return false;
  }
  return thenBlock->findAncestorOpInBlock(*op) != nullptr;
}

// True when ssbufIf contains a splitted_if.
static bool containsSplittedIf(scf::IfOp ssbufIf) {
  bool found = false;
  ssbufIf.walk([&](scf::IfOp ifOp) {
    if (ifOp == ssbufIf || !ifOp->hasAttr(kSplittedIf)) {
      return WalkResult::advance();
    }
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

// True when ssbufIf contains a real tensor compute op.
static bool ssbufIfHasTensorCompute(scf::IfOp ssbufIf) {
  WalkResult walk = ssbufIf->walk([&](Operation *op) {
    if (isRealTensorComputeOp(op)) {
      LDBG("Tensor compute in ssbuffer.if without splitted_if: " << *op
                                                                 << "\n");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walk.wasInterrupted();
}

// True when every ssbuffer.if without a splitted_if is scalar-only.
static bool scalarSsbufIfsHaveNoTensorCompute(ModuleOp module) {
  bool allScalar = true;
  module.walk([&](scf::IfOp ifOp) {
    if (!ifOp->hasAttr(kIf) || containsSplittedIf(ifOp)) {
      return WalkResult::advance();
    }
    if (ssbufIfHasTensorCompute(ifOp)) {
      LDBG("Skip wrap: ssbuffer.if without splitted_if has tensor compute. if: "
           << ifOp << "\n");
      allScalar = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return allScalar;
}

// True when real tensor compute under ssbufIf stays in splittedIf's then.
static bool computeOnlyInSplitThen(scf::IfOp ssbufIf, scf::IfOp splittedIf) {
  WalkResult walk = ssbufIf->walk([&](Operation *op) {
    if (opIsInSplitThen(op, splittedIf)) {
      return WalkResult::skip();
    }
    if (isRealTensorComputeOp(op)) {
      LDBG("Tensor compute outside the splitted_if then: " << *op << "\n");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !walk.wasInterrupted();
}

// Collect the condition def-chain defined inside ssbufIf.
static int collectConditionDefOpsInside(Value value, scf::IfOp ssbufIf,
                                        DenseSet<Operation *> &ops) {
  Operation *defOp = value ? value.getDefiningOp() : nullptr;
  if (defOp && ssbufIf->isProperAncestor(defOp) && ops.insert(defOp).second) {
    for (Value operand : defOp->getOperands()) {
      if (collectConditionDefOpsInside(operand, ssbufIf, ops) !=
          WRAP_SPLITTED_IF_SUCCESS) {
        return WRAP_SPLITTED_IF_FAILED;
      }
    }
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// True when def is before user in the IR.
static bool opDominates(Operation *def, Operation *user) {
  if (!def || !user || def == user) {
    return false;
  }
  Block *defBlock = def->getBlock();
  Block *userBlock = user->getBlock();
  if (!defBlock || !userBlock) {
    return false;
  }
  if (defBlock == userBlock) {
    return def->isBeforeInBlock(user);
  }
  Operation *cur = user;
  while (Operation *parent = cur->getParentOp()) {
    if (parent == def) {
      return false;
    }
    if (parent->getBlock() == defBlock) {
      return def->isBeforeInBlock(parent);
    }
    cur = parent;
  }
  return false;
}

// True when the condition def-chain can be hoisted above ssbufIf.
static bool canHoistConditionOps(scf::IfOp ssbufIf,
                                 const DenseSet<Operation *> &ops) {
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || ops.contains(def)) {
        continue;
      }
      if (!opDominates(def, ssbufIf)) {
        return false;
      }
    }
  }
  return true;
}

// Hoist the splitted_if condition above ssbufIf.
static int hoistSplittedIfCondition(scf::IfOp ssbufIf, scf::IfOp splittedIf) {
  Value cond = splittedIf.getCondition();
  if (!cond) {
    LDBG("Failed to hoist splitted_if condition: missing condition. if: "
         << splittedIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }

  DenseSet<Operation *> condOps;
  if (collectConditionDefOpsInside(cond, ssbufIf, condOps) !=
      WRAP_SPLITTED_IF_SUCCESS) {
    LDBG("Failed to hoist splitted_if condition: def-chain walk failed. if: "
         << splittedIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  if (condOps.empty()) {
    return WRAP_SPLITTED_IF_SUCCESS;
  }
  if (!canHoistConditionOps(ssbufIf, condOps)) {
    LDBG("Skip wrap: splitted_if condition operand does not dominate "
         "ssbuffer.if. splitted_if: "
         << splittedIf << "\n ssbuffer.if: " << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }

  Block *thenBlock = ssbufIf.thenBlock();
  if (!thenBlock) {
    LDBG("Failed to hoist splitted_if condition: missing then block. "
         "ssbuffer.if: "
         << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  SmallVector<Operation *> sorted;
  for (Operation &op : *thenBlock) {
    if (condOps.contains(&op)) {
      sorted.push_back(&op);
    }
  }
  if (sorted.size() != condOps.size()) {
    LDBG("Skip wrap: splitted_if condition def is nested below the then "
         "block. splitted_if: "
         << splittedIf << "\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  for (Operation *op : sorted) {
    op->moveBefore(ssbufIf);
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Collect the condition def-chain that already sits before anchor.
static int collectConditionDefOpsBefore(Value value, Operation *anchor,
                                        DenseSet<Operation *> &ops) {
  Operation *defOp = value ? value.getDefiningOp() : nullptr;
  if (anchor && defOp && defOp->getBlock() == anchor->getBlock() &&
      defOp->isBeforeInBlock(anchor) && ops.insert(defOp).second) {
    for (Value operand : defOp->getOperands()) {
      if (collectConditionDefOpsBefore(operand, anchor, ops) !=
          WRAP_SPLITTED_IF_SUCCESS) {
        return WRAP_SPLITTED_IF_FAILED;
      }
    }
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Include annotation.mark ops that use a condition value.
static void addSatelliteMarkOps(DenseSet<Operation *> &ops, Operation *anchor) {
  for (Operation &op : *anchor->getBlock()) {
    if (&op == anchor) {
      break;
    }
    if (!isa<annotation::MarkOp>(&op)) {
      continue;
    }
    for (Value operand : op.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && ops.contains(def)) {
        ops.insert(&op);
        break;
      }
    }
  }
}

// True when every use of op moves with the chain or already sits inside the if.
static bool opHasOnlyLocalUses(Operation *op, const DenseSet<Operation *> &ops,
                               Operation *insertBefore) {
  Block *destBlock = insertBefore->getBlock();
  for (OpOperand &use : op->getUses()) {
    Operation *user = use.getOwner();
    if (ops.contains(user) || user == insertBefore) {
      continue;
    }
    if (insertBefore->isProperAncestor(user)) {
      continue;
    }
    if (op->getBlock() != destBlock && user->getBlock() == destBlock &&
        user->isBeforeInBlock(insertBefore)) {
      continue;
    }
    return false;
  }
  return true;
}

// First condition op already sitting before `before`, otherwise `before`.
static Operation *insertPointForCondOps(Operation *before,
                                        const DenseSet<Operation *> &condOps) {
  for (Operation &op : *before->getBlock()) {
    if (&op == before) {
      break;
    }
    if (condOps.contains(&op)) {
      return &op;
    }
  }
  return before;
}

// Move the ssbuffer.if condition def-chain to just before `before`.
static int moveDefChainBefore(Value cond, Operation *fromAnchor,
                              Operation *before,
                              const DenseSet<Operation *> &stayOutside) {
  DenseSet<Operation *> condOps;
  if (collectConditionDefOpsBefore(cond, fromAnchor, condOps) !=
      WRAP_SPLITTED_IF_SUCCESS) {
    return WRAP_SPLITTED_IF_FAILED;
  }
  if (condOps.empty()) {
    return WRAP_SPLITTED_IF_SUCCESS;
  }
  addSatelliteMarkOps(condOps, fromAnchor);

  SmallVector<Operation *> sorted;
  for (Operation &op : *fromAnchor->getBlock()) {
    if (condOps.contains(&op)) {
      sorted.push_back(&op);
    }
    if (&op == fromAnchor) {
      break;
    }
  }

  DenseSet<Operation *> moveSet;
  for (Operation *op : sorted) {
    if (stayOutside.contains(op)) {
      continue;
    }
    if (opHasOnlyLocalUses(op, condOps, before)) {
      moveSet.insert(op);
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : sorted) {
      if (!moveSet.contains(op)) {
        continue;
      }
      for (OpOperand &use : op->getUses()) {
        Operation *user = use.getOwner();
        if (!condOps.contains(user) || moveSet.contains(user)) {
          continue;
        }
        moveSet.erase(op);
        changed = true;
        break;
      }
    }
  }

  Operation *insertPt = before;
  if (fromAnchor->getBlock() != before->getBlock()) {
    insertPt = insertPointForCondOps(before, condOps);
  }

  for (Operation *op : sorted) {
    if (!moveSet.contains(op) || op == insertPt) {
      continue;
    }
    op->moveBefore(insertPt);
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Splice the splitted_if then-body in front of it and erase the if.
static int unwrapSplittedIf(scf::IfOp splittedIf) {
  Block *thenBlock = splittedIf.thenBlock();
  if (!thenBlock || !thenBlock->mightHaveTerminator()) {
    LDBG("Failed to unwrap splitted_if: missing then terminator. if: "
         << splittedIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  auto thenYield = dyn_cast<scf::YieldOp>(thenBlock->getTerminator());
  if (!thenYield) {
    LDBG("Failed to unwrap splitted_if: then terminator is not scf.yield. if: "
         << splittedIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  SmallVector<Value> thenVals(thenYield.getOperands().begin(),
                              thenYield.getOperands().end());
  if (splittedIf.getNumResults() != thenVals.size()) {
    LDBG("Failed to unwrap splitted_if: yield operand count "
         << thenVals.size() << ", results " << splittedIf.getNumResults()
         << ". if: " << splittedIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  for (Value val : thenVals) {
    if (val.getDefiningOp() == splittedIf.getOperation()) {
      LDBG("Failed to unwrap splitted_if: yield uses its own result. if: "
           << splittedIf << "\n");
      return WRAP_SPLITTED_IF_FAILED;
    }
  }

  SmallVector<Operation *> thenOps;
  for (Operation &op : *thenBlock) {
    if (!op.hasTrait<OpTrait::IsTerminator>()) {
      thenOps.push_back(&op);
    }
  }
  for (Operation *op : thenOps) {
    op->moveBefore(splittedIf);
  }

  splittedIf.replaceAllUsesWith(thenVals);
  splittedIf.erase();
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Replace the block terminator with scf.yield.
static void replaceTerminator(Block *block, Location loc, ValueRange operands) {
  if (block->mightHaveTerminator()) {
    block->getTerminator()->erase();
  }
  OpBuilder builder(block, block->end());
  builder.create<scf::YieldOp>(loc, operands);
}

// Nearest ancestor main loop.
static Operation *getEnclosingMainLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isMainLoopOp(parent)) {
      return parent;
    }
  }
  return nullptr;
}

// Check that this main loop has at most one splitted_if id.
static int onlyOneSplittedIfKind(Operation *mainLoop) {
  IntegerAttr firstSplittedIfId;
  int result = WRAP_SPLITTED_IF_SUCCESS;
  mainLoop->walk([&](scf::IfOp ifOp) {
    if (getEnclosingMainLoop(ifOp) != mainLoop) {
      return WalkResult::advance();
    }
    if (!ifOp->hasAttr(kSplittedIf)) {
      return WalkResult::advance();
    }
    auto splittedIfId = ifOp->getAttrOfType<IntegerAttr>(kSplittedIf);
    if (!splittedIfId) {
      LDBG("ssbuffer.splitted_if is not an integer id. if: " << ifOp << "\n");
      result = WRAP_SPLITTED_IF_FAILED;
      return WalkResult::interrupt();
    }
    if (!firstSplittedIfId) {
      firstSplittedIfId = splittedIfId;
      return WalkResult::advance();
    }
    if (firstSplittedIfId.getInt() == splittedIfId.getInt()) {
      return WalkResult::advance();
    }
    LDBG("Skip wrap: main loop has more than one splitted_if id. if: " << ifOp
                                                                       << "\n");
    result = WRAP_SPLITTED_IF_IGNORED;
    return WalkResult::interrupt();
  });
  return result;
}

// True when the while_arg chain can be cloned into the new if's else.
static int whileArgChainCanClone(scf::IfOp ssbufIf) {
  if (ssbufIf.getThenRegion().empty()) {
    return WRAP_SPLITTED_IF_IGNORED;
  }
  int result = WRAP_SPLITTED_IF_SUCCESS;
  ssbufIf.getThenRegion().walk([&](Operation *op) {
    if (!op->hasAttr(kWhileArg)) {
      return WalkResult::advance();
    }
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !ssbufIf->isProperAncestor(def)) {
        continue;
      }
      if (!def->hasAttr(kWhileArg)) {
        LDBG("Skip wrap: while_arg operand is not a while_arg. op: "
             << *op << "\n def: " << *def << "\n");
        result = WRAP_SPLITTED_IF_IGNORED;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

// Order while_arg ops so defs come before uses.
static int orderWhileArgOps(ArrayRef<Operation *> chain,
                            SmallVector<Operation *> &ordered) {
  DenseSet<Operation *> pending(chain.begin(), chain.end());
  while (!pending.empty()) {
    bool progress = false;
    for (Operation *op : chain) {
      if (!pending.contains(op)) {
        continue;
      }
      bool ready = true;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (def && pending.contains(def)) {
          ready = false;
          break;
        }
      }
      if (!ready) {
        continue;
      }
      ordered.push_back(op);
      pending.erase(op);
      progress = true;
    }
    if (!progress) {
      LDBG("Failed to order while_arg ops: chain has a cycle.\n");
      for (Operation *op : pending) {
        LDBG("  " << *op << "\n");
      }
      return WRAP_SPLITTED_IF_FAILED;
    }
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Clone the while_arg chain into the new if's else.
static int buildWhileNewIfElse(scf::IfOp ssbufIf, Block *elseBlock,
                               ArrayRef<Value> elsePassthrough, Location loc) {
  auto thenYield = dyn_cast<scf::YieldOp>(ssbufIf.thenBlock()->getTerminator());
  if (!thenYield || thenYield.getNumOperands() != elsePassthrough.size() ||
      thenYield.getNumOperands() != ssbufIf.getNumResults()) {
    LDBG("Failed to build while new-if else: then yield does not match. if: "
         << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }

  SmallVector<Operation *> chain;
  ssbufIf.getThenRegion().walk([&](Operation *op) {
    if (op->hasAttr(kWhileArg)) {
      chain.push_back(op);
    }
  });
  SmallVector<Operation *> ordered;
  if (orderWhileArgOps(chain, ordered) != WRAP_SPLITTED_IF_SUCCESS) {
    return WRAP_SPLITTED_IF_FAILED;
  }

  OpBuilder elseBuilder(elseBlock, elseBlock->end());
  if (elseBlock->mightHaveTerminator()) {
    elseBuilder.setInsertionPoint(elseBlock->getTerminator());
  }
  IRMapping mapping;
  for (Operation *op : ordered) {
    Operation *cloned = elseBuilder.clone(*op, mapping);
    elseBuilder.setInsertionPointAfter(cloned);
  }

  SmallVector<Value> elseVals(elsePassthrough.begin(), elsePassthrough.end());
  for (auto [idx, thenVal] : llvm::enumerate(thenYield.getOperands())) {
    if (mapping.contains(thenVal)) {
      elseVals[idx] = mapping.lookup(thenVal);
    }
  }
  replaceTerminator(elseBlock, loc, elseVals);
  LDBG("Cloned " << ordered.size() << " while_arg ops into the new if else.\n");
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Build the scf.for new-if else, replacing the block counter with counter+step.
static int buildForNewIfElse(scf::ForOp forOp, ControlFlowConditionInfo *info,
                             scf::IfOp ssbufIf, Block *elseBlock,
                             ArrayRef<Value> elsePassthrough, Location loc) {
  if (!info->cntArgs.count(ssbufIf)) {
    LDBG("Failed to build for new-if else: ssbuffer.if has no cntArgs. if: "
         << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  Value counter = info->cntArgs[ssbufIf];
  Value step = forOp.getStep();
  if (!counter || !step) {
    LDBG("Failed to build for new-if else: counter or step is null. if: "
         << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }

  OpBuilder elseBuilder(elseBlock, elseBlock->end());
  if (elseBlock->mightHaveTerminator()) {
    elseBuilder.setInsertionPoint(elseBlock->getTerminator());
  }
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
      elseVals.back().getType() == inc.getType()) {
    elseVals.back() = inc;
  }
  if (elseVals.size() != ssbufIf.getNumResults()) {
    LDBG("Failed to build for new-if else: else value count "
         << elseVals.size() << ", results " << ssbufIf.getNumResults()
         << ". if: " << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  replaceTerminator(elseBlock, loc, elseVals);
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Build the new if's else. for adds counter+step; while clones the while_arg
// chain.
static int buildNewIfElse(Operation *loopOp, ControlFlowConditionInfo *info,
                          scf::IfOp ssbufIf, Block *elseBlock,
                          ArrayRef<Value> elsePassthrough, Location loc) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    return buildForNewIfElse(forOp, info, ssbufIf, elseBlock, elsePassthrough,
                             loc);
  }
  if (isa<scf::WhileOp>(loopOp)) {
    return buildWhileNewIfElse(ssbufIf, elseBlock, elsePassthrough, loc);
  }
  LDBG("Failed to build new-if else: loop is neither scf.for nor scf.while. "
       "loop: "
       << *loopOp << "\n");
  return WRAP_SPLITTED_IF_FAILED;
}

// Check that the new if's else can advance the loop.
static int checkLoopCanAdvance(scf::IfOp ssbufIf, Operation *loopOp,
                               ControlFlowConditionInfo *info) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    if (!info->cntArgs.count(ssbufIf)) {
      LDBG("Skip wrap: ssbuffer.if has no cntArgs. if: " << ssbufIf << "\n");
      return WRAP_SPLITTED_IF_IGNORED;
    }
    if (!info->cntArgs[ssbufIf] || !forOp.getStep()) {
      LDBG("Failed to wrap: counter or step is null. if: " << ssbufIf << "\n");
      return WRAP_SPLITTED_IF_FAILED;
    }
    return WRAP_SPLITTED_IF_SUCCESS;
  }
  if (isa<scf::WhileOp>(loopOp)) {
    if (int returnCode = whileArgChainCanClone(ssbufIf);
        returnCode != WRAP_SPLITTED_IF_SUCCESS) {
      LDBG("Skip wrap: while update chain uses a value inside ssbuffer.if. if: "
           << ssbufIf << "\n");
      return returnCode;
    }
    return WRAP_SPLITTED_IF_SUCCESS;
  }
  return WRAP_SPLITTED_IF_FAILED;
}

// Check that ssbufIf can be wrapped. On success, elsePassthrough holds the else
// yield.
static int prepareSsbufWrap(scf::IfOp ssbufIf, Operation *loopOp,
                            ControlFlowConditionInfo *info,
                            scf::IfOp &splittedIf,
                            SmallVector<Value> &elsePassthrough) {
  if (!isa<scf::ForOp, scf::WhileOp>(loopOp)) {
    LDBG("Skip wrap: main loop is neither scf.for nor scf.while. loop: "
         << *loopOp << "\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  splittedIf = getUniqueFirstLevelSplittedIf(ssbufIf);
  if (!splittedIf) {
    return WRAP_SPLITTED_IF_IGNORED;
  }
  if (!isOnlySplittedIf(ssbufIf, splittedIf)) {
    LDBG("Skip wrap: ssbuffer.if contains more than one splitted_if. if: "
         << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  if (!computeOnlyInSplitThen(ssbufIf, splittedIf)) {
    LDBG("Skip wrap: real tensor compute outside the splitted_if then. if: "
         << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  Block *ssbufElse = ssbufIf.elseBlock();
  if (!ssbufElse || !ssbufElse->mightHaveTerminator()) {
    LDBG("Skip wrap: ssbuffer.if has no else yield. if: " << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  auto ssbufElseYield = dyn_cast<scf::YieldOp>(ssbufElse->getTerminator());
  if (!ssbufElseYield) {
    LDBG("Skip wrap: ssbuffer.if else terminator is not scf.yield. if: "
         << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_IGNORED;
  }
  if (!splittedIf.getCondition() || !ssbufIf.getCondition()) {
    LDBG("Failed to wrap: splitted_if or ssbuffer.if has no condition. "
         "splitted_if: "
         << splittedIf << "\n ssbuffer.if: " << ssbufIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  if (int returnCode = checkLoopCanAdvance(ssbufIf, loopOp, info);
      returnCode != WRAP_SPLITTED_IF_SUCCESS) {
    return returnCode;
  }
  elsePassthrough.assign(ssbufElseYield.getOperands().begin(),
                         ssbufElseYield.getOperands().end());
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Hoist the split condition and pack the ssbuffer.if condition against the if.
static int hoistSplitCondAndPackSsbufCond(scf::IfOp ssbufIf,
                                          scf::IfOp splittedIf) {
  if (int returnCode = hoistSplittedIfCondition(ssbufIf, splittedIf);
      returnCode != WRAP_SPLITTED_IF_SUCCESS) {
    return returnCode;
  }
  DenseSet<Operation *> splitCondOps;
  if (collectConditionDefOpsBefore(splittedIf.getCondition(),
                                   ssbufIf.getOperation(),
                                   splitCondOps) != WRAP_SPLITTED_IF_SUCCESS) {
    LDBG("Failed to pack conditions: splitted_if def-chain walk failed. if: "
         << splittedIf << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  addSatelliteMarkOps(splitCondOps, ssbufIf.getOperation());
  if (int returnCode =
          moveDefChainBefore(ssbufIf.getCondition(), ssbufIf.getOperation(),
                             ssbufIf.getOperation(), splitCondOps);
      returnCode != WRAP_SPLITTED_IF_SUCCESS) {
    LDBG("Skip wrap: cannot pack ssbuffer.if cond against the if. if: "
         << ssbufIf << "\n");
    return returnCode;
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Create the outer if on the split condition and move ssbufIf into its then.
static int createNewIfOp(scf::IfOp ssbufIf, scf::IfOp splittedIf,
                         scf::IfOp &newIfOp) {
  Location loc = ssbufIf.getLoc();
  OpBuilder builder(ssbufIf);
  SmallVector<Type> resultTypes(ssbufIf->getResultTypes());
  newIfOp =
      builder.create<scf::IfOp>(loc, resultTypes, splittedIf.getCondition(),
                                /*withElse=*/true);
  if (Attribute splitAttr = splittedIf->getAttr(kSplittedIf)) {
    newIfOp->setAttr(kSplittedIf, splitAttr);
  }

  Block *thenBlock = newIfOp.thenBlock();
  Block *elseBlock = newIfOp.elseBlock();
  if (!thenBlock || !elseBlock) {
    LDBG("Failed to create new if: missing then or else block. if: " << ssbufIf
                                                                     << "\n");
    return WRAP_SPLITTED_IF_FAILED;
  }
  if (thenBlock->mightHaveTerminator()) {
    ssbufIf->moveBefore(thenBlock->getTerminator());
  } else {
    ssbufIf->moveBefore(thenBlock, thenBlock->end());
  }

  DenseSet<Operation *> splitCondOps;
  if (collectConditionDefOpsBefore(splittedIf.getCondition(),
                                   newIfOp.getOperation(),
                                   splitCondOps) != WRAP_SPLITTED_IF_SUCCESS) {
    LDBG("Keep ssbuffer.if cond ops outside the new if then. if: " << ssbufIf
                                                                   << "\n");
    return WRAP_SPLITTED_IF_SUCCESS;
  }
  if (moveDefChainBefore(ssbufIf.getCondition(), newIfOp.getOperation(),
                         ssbufIf.getOperation(),
                         splitCondOps) != WRAP_SPLITTED_IF_SUCCESS) {
    LDBG("Keep ssbuffer.if cond ops outside the new if then. if: " << ssbufIf
                                                                   << "\n");
  }
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Retarget uses of ssbufIf results outside the new if to the new if's results.
static void replaceUsesOutsideNewIfOp(scf::IfOp ssbufIf, scf::IfOp newIfOp) {
  for (auto [innerRes, outerRes] :
       llvm::zip(ssbufIf.getResults(), newIfOp.getResults())) {
    innerRes.replaceUsesWithIf(outerRes, [&](OpOperand &use) {
      return !newIfOp->isProperAncestor(use.getOwner());
    });
  }
}

// Wrap one ssbuffer.if.
static int splittedIfWrapSsbufIf(scf::IfOp ssbufIf, Operation *loopOp,
                                 ControlFlowConditionInfo *info) {
  scf::IfOp splittedIf;
  SmallVector<Value> elsePassthrough;
  if (int returnCode =
          prepareSsbufWrap(ssbufIf, loopOp, info, splittedIf, elsePassthrough);
      returnCode != WRAP_SPLITTED_IF_SUCCESS) {
    return returnCode;
  }
  if (int returnCode = hoistSplitCondAndPackSsbufCond(ssbufIf, splittedIf);
      returnCode != WRAP_SPLITTED_IF_SUCCESS) {
    return returnCode;
  }

  scf::IfOp newIfOp;
  if (createNewIfOp(ssbufIf, splittedIf, newIfOp) != WRAP_SPLITTED_IF_SUCCESS) {
    return WRAP_SPLITTED_IF_FAILED;
  }
  if (unwrapSplittedIf(splittedIf) != WRAP_SPLITTED_IF_SUCCESS) {
    LDBG("Keep inner splitted_if; cannot splice then-body. if: " << splittedIf
                                                                 << "\n");
  }

  Location loc = ssbufIf.getLoc();
  replaceTerminator(newIfOp.thenBlock(), loc, ssbufIf.getResults());
  if (buildNewIfElse(loopOp, info, ssbufIf, newIfOp.elseBlock(),
                     elsePassthrough, loc) != WRAP_SPLITTED_IF_SUCCESS) {
    return WRAP_SPLITTED_IF_FAILED;
  }
  replaceUsesOutsideNewIfOp(ssbufIf, newIfOp);
  LDBG("Wrapped ssbuffer.if with hoisted splitted_if. if: " << ssbufIf << "\n");
  return WRAP_SPLITTED_IF_SUCCESS;
}

// Walk main loops and wrap each eligible ssbuffer.if.
void SplittedIfWrapSsbufIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  LDBG("Before SplittedIfWrapSsbufIf:\n" << module << "\n");
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  if (!scalarSsbufIfsHaveNoTensorCompute(module)) {
    return;
  }

  int fallbackReturnCode = WRAP_SPLITTED_IF_SUCCESS;
  WalkResult walkResult = module.walk([&](Operation *loopOp) -> WalkResult {
    if (!isMainLoopOp(loopOp)) {
      return WalkResult::advance();
    }

    int splittedIfKindReturnCode = onlyOneSplittedIfKind(loopOp);
    if (splittedIfKindReturnCode == WRAP_SPLITTED_IF_IGNORED) {
      return WalkResult::advance();
    }
    if (splittedIfKindReturnCode != WRAP_SPLITTED_IF_SUCCESS) {
      fallbackReturnCode = splittedIfKindReturnCode;
      return WalkResult::interrupt();
    }

    SmallVector<scf::IfOp> ssbufIfs;
    loopOp->walk([&](scf::IfOp ifOp) {
      if (!ifOp->hasAttr(kIf)) {
        return;
      }
      ssbufIfs.push_back(ifOp);
    });
    for (scf::IfOp ssbufIf : ssbufIfs) {
      int wrapReturnCode = splittedIfWrapSsbufIf(ssbufIf, loopOp, info);
      if (wrapReturnCode == WRAP_SPLITTED_IF_IGNORED ||
          wrapReturnCode == WRAP_SPLITTED_IF_SUCCESS) {
        continue;
      }
      LDBG("splittedIfWrapSsbufIf failed. if: " << ssbufIf << "\n");
      fallbackReturnCode = wrapReturnCode;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    if (fallbackReturnCode == WRAP_SPLITTED_IF_IGNORED) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    } else {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
  }
  LDBG("After SplittedIfWrapSsbufIf:\n" << module << "\n");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createSplittedIfWrapSsbufIfPass() {
  return std::make_unique<SplittedIfWrapSsbufIfPass>();
}
} // namespace triton
} // namespace mlir
