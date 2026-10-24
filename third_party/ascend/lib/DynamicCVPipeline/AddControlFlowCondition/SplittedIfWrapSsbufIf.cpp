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
#include "llvm/ADT/MapVector.h"
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

// 0 success, -1 failure.
static constexpr int WRAP_SSBUF_IF_SUCCESS = 0;
static constexpr int WRAP_SSBUF_IF_FAILED = -1;

// Retarget uses of ssbufIf's results that sit outside newIfOp to newIfOp's
// results.
static void replaceUsesOutsideNewIfOp(scf::IfOp ssbufIf, scf::IfOp newIfOp) {
  for (auto [innerRes, outerRes] :
       llvm::zip(ssbufIf.getResults(), newIfOp.getResults())) {
    innerRes.replaceUsesWithIf(outerRes, [&](OpOperand &use) {
      return !newIfOp->isProperAncestor(use.getOwner());
    });
  }
}

// Replace block's terminator with scf.yield of operands.
static void replaceTerminator(Block *block, Location loc, ValueRange operands) {
  if (block->mightHaveTerminator()) {
    block->getTerminator()->erase();
  }
  OpBuilder builder(block, block->end());
  builder.create<scf::YieldOp>(loc, operands);
}

// Move splittedIf's then body in front of splittedIf.
// Retarget splittedIf's results to the then yield operands, then erase
// splittedIf.
static void unwrapSplittedIf(scf::IfOp splittedIf) {
  Block *thenBlock = splittedIf.thenBlock();
  auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
  SmallVector<Value> thenVals(thenYield.getOperands().begin(),
                              thenYield.getOperands().end());

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
}

// Order chain so each def precedes its uses, and append that order to ordered.
// Return FAILED when chain contains a cycle.
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
      return WRAP_SSBUF_IF_FAILED;
    }
  }
  return WRAP_SSBUF_IF_SUCCESS;
}

// Clone ssbufIf's while_arg ops that sit outside splittedIf into elseBlock.
// Yield elsePassthrough, replacing a slot when the cloned ops produce that
// then-yield value.
static int buildWhileNewIfElse(scf::IfOp ssbufIf, scf::IfOp splittedIf,
                               Block *elseBlock,
                               ArrayRef<Value> elsePassthrough, Location loc) {
  auto thenYield = dyn_cast<scf::YieldOp>(ssbufIf.thenBlock()->getTerminator());
  if (!thenYield || thenYield.getNumOperands() != elsePassthrough.size() ||
      thenYield.getNumOperands() != ssbufIf.getNumResults()) {
    LDBG("Failed to build while new-if else: then yield does not match. if: "
         << ssbufIf << "\n");
    return WRAP_SSBUF_IF_FAILED;
  }

  SmallVector<Operation *> chain;
  ssbufIf.getThenRegion().walk([&](Operation *op) {
    if (op->hasAttr(kWhileArg) && !splittedIf->isProperAncestor(op)) {
      chain.push_back(op);
    }
  });
  SmallVector<Operation *> ordered;
  if (orderWhileArgOps(chain, ordered) != WRAP_SSBUF_IF_SUCCESS) {
    return WRAP_SSBUF_IF_FAILED;
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
  return WRAP_SSBUF_IF_SUCCESS;
}

// Yield elsePassthrough from elseBlock.
// Replace ssbufIf's recorded block counter with that counter plus forOp's step.
static int buildForNewIfElse(scf::ForOp forOp, ControlFlowConditionInfo *info,
                             scf::IfOp ssbufIf, Block *elseBlock,
                             ArrayRef<Value> elsePassthrough, Location loc) {
  if (!info->cntArgs.count(ssbufIf)) {
    LDBG("Failed to build for new-if else: ssbuffer.if has no cntArgs. if: "
         << ssbufIf << "\n");
    return WRAP_SSBUF_IF_FAILED;
  }
  Value counter = info->cntArgs[ssbufIf];
  Value step = forOp.getStep();
  if (!counter || !step) {
    LDBG("Failed to build for new-if else: counter or step is null. if: "
         << ssbufIf << "\n");
    return WRAP_SSBUF_IF_FAILED;
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
    return WRAP_SSBUF_IF_FAILED;
  }
  replaceTerminator(elseBlock, loc, elseVals);
  return WRAP_SSBUF_IF_SUCCESS;
}

// Fill elseBlock for loopOp from elsePassthrough.
// for replaces ssbufIf's block counter. while clones while_arg ops outside
// splittedIf.
static int buildNewIfElse(Operation *loopOp, ControlFlowConditionInfo *info,
                          scf::IfOp ssbufIf, scf::IfOp splittedIf,
                          Block *elseBlock, ArrayRef<Value> elsePassthrough,
                          Location loc) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    return buildForNewIfElse(forOp, info, ssbufIf, elseBlock, elsePassthrough,
                             loc);
  }
  return buildWhileNewIfElse(ssbufIf, splittedIf, elseBlock, elsePassthrough,
                             loc);
}

// Return the first op in condOps that already sits ahead of before. Otherwise
// return before.
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

// Return true when every use of op is in ops or inside insertBefore.
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

// Insert into ops each annotation.mark before anchor that uses a value defined
// by ops.
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

// Insert into ops the def-chain of value that already sits in anchor's block,
// before anchor.
static void collectConditionDefOpsBefore(Value value, Operation *anchor,
                                         DenseSet<Operation *> &ops) {
  SmallVector<Value> worklist;
  if (value) {
    worklist.push_back(value);
  }
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    Operation *defOp = current.getDefiningOp();
    if (!defOp || defOp->getBlock() != anchor->getBlock() ||
        !defOp->isBeforeInBlock(anchor) || !ops.insert(defOp).second) {
      continue;
    }
    worklist.append(defOp->operand_begin(), defOp->operand_end());
  }
}

// Move the def-chain of cond that sits before fromAnchor to just ahead of
// before. Leave ops in stayOutside where they are.
static void moveDefChainBefore(Value cond, Operation *fromAnchor,
                               Operation *before,
                               const DenseSet<Operation *> &stayOutside) {
  DenseSet<Operation *> condOps;
  collectConditionDefOpsBefore(cond, fromAnchor, condOps);
  if (condOps.empty()) {
    return;
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
}

// Create newIfOp on splittedIf's condition, move ssbufIf into its then, and
// pack ssbufIf's condition there.
static void wrapSsbufIf(scf::IfOp ssbufIf, scf::IfOp splittedIf,
                        scf::IfOp &newIfOp) {
  Location loc = ssbufIf.getLoc();
  OpBuilder builder(ssbufIf);
  SmallVector<Type> resultTypes(ssbufIf->getResultTypes());
  newIfOp =
      builder.create<scf::IfOp>(loc, resultTypes, splittedIf.getCondition(),
                                /*withElse=*/true);
  newIfOp->setAttr(kSplittedIf, splittedIf->getAttr(kSplittedIf));

  // An scf.if with results is created without a yield. Put ssbufIf before the
  // yield when one exists, otherwise at the end of the then block.
  Block *thenBlock = newIfOp.thenBlock();
  if (thenBlock->mightHaveTerminator()) {
    ssbufIf->moveBefore(thenBlock->getTerminator());
  } else {
    ssbufIf->moveBefore(thenBlock, thenBlock->end());
  }

  DenseSet<Operation *> splitCondOps;
  collectConditionDefOpsBefore(splittedIf.getCondition(),
                               newIfOp.getOperation(), splitCondOps);
  moveDefChainBefore(ssbufIf.getCondition(), newIfOp.getOperation(),
                     ssbufIf.getOperation(), splitCondOps);
}

// Insert into ops the def-chain of value that is defined inside ssbufIf.
static void collectConditionDefOpsInside(Value value, scf::IfOp ssbufIf,
                                         DenseSet<Operation *> &ops) {
  SmallVector<Value> worklist;
  if (value) {
    worklist.push_back(value);
  }
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    Operation *defOp = current.getDefiningOp();
    if (!defOp || !ssbufIf->isProperAncestor(defOp) ||
        !ops.insert(defOp).second) {
      continue;
    }
    worklist.append(defOp->operand_begin(), defOp->operand_end());
  }
}

// Move the def-chain of splittedIf's condition that sits in ssbufIf's then to
// just before ssbufIf.
static void hoistSplittedIfCondition(scf::IfOp ssbufIf, scf::IfOp splittedIf) {
  DenseSet<Operation *> condOps;
  collectConditionDefOpsInside(splittedIf.getCondition(), ssbufIf, condOps);
  if (condOps.empty()) {
    return;
  }
  SmallVector<Operation *> sorted;
  for (Operation &op : *ssbufIf.thenBlock()) {
    if (condOps.contains(&op)) {
      sorted.push_back(&op);
    }
  }
  for (Operation *op : sorted) {
    op->moveBefore(ssbufIf);
  }
}

// Hoist splittedIf's condition above ssbufIf, then pack ssbufIf's condition
// against ssbufIf.
static void hoistSplitCondAndPackSsbufCond(scf::IfOp ssbufIf,
                                           scf::IfOp splittedIf) {
  hoistSplittedIfCondition(ssbufIf, splittedIf);
  DenseSet<Operation *> splitCondOps;
  collectConditionDefOpsBefore(splittedIf.getCondition(),
                               ssbufIf.getOperation(), splitCondOps);
  addSatelliteMarkOps(splitCondOps, ssbufIf.getOperation());
  moveDefChainBefore(ssbufIf.getCondition(), ssbufIf.getOperation(),
                     ssbufIf.getOperation(), splitCondOps);
}

// Wrap ssbufIf in an if on splittedIf's condition inside loopOp.
// for reads the block counter recorded for ssbufIf.
static int splittedIfWrapSsbufIf(scf::IfOp ssbufIf, scf::IfOp splittedIf,
                                 Operation *loopOp,
                                 ControlFlowConditionInfo *info) {
  auto elseYield = cast<scf::YieldOp>(ssbufIf.elseBlock()->getTerminator());
  SmallVector<Value> elsePassthrough(elseYield.getOperands().begin(),
                                     elseYield.getOperands().end());
  hoistSplitCondAndPackSsbufCond(ssbufIf, splittedIf);

  scf::IfOp newIfOp;
  wrapSsbufIf(ssbufIf, splittedIf, newIfOp);

  Location loc = ssbufIf.getLoc();
  if (buildNewIfElse(loopOp, info, ssbufIf, splittedIf, newIfOp.elseBlock(),
                     elsePassthrough, loc) != WRAP_SSBUF_IF_SUCCESS) {
    return WRAP_SSBUF_IF_FAILED;
  }
  unwrapSplittedIf(splittedIf);
  replaceTerminator(newIfOp.thenBlock(), loc, ssbufIf.getResults());
  replaceUsesOutsideNewIfOp(ssbufIf, newIfOp);
  LDBG("Wrapped ssbuffer.if with hoisted splitted_if. if: " << ssbufIf << "\n");
  return WRAP_SSBUF_IF_SUCCESS;
}

// Return true when ssbufIf contains a real tensor compute op.
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

// Return true when every real tensor compute op under ssbufIf sits in
// splittedIf's then.
static bool computeOnlyInSplitThen(scf::IfOp ssbufIf, scf::IfOp splittedIf) {
  Block *thenBlock = splittedIf.thenBlock();
  WalkResult walk = ssbufIf->walk([&](Operation *op) {
    if (thenBlock->findAncestorOpInBlock(*op)) {
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

// Return how many splitted_if ops sit under ssbufIf. Stop after the second one.
// Set splittedIf when the count is one. Otherwise set splittedIf to null.
static unsigned countSplittedIfs(scf::IfOp ssbufIf, scf::IfOp &splittedIf) {
  unsigned count = 0;
  scf::IfOp found;
  ssbufIf.walk([&](scf::IfOp nested) {
    if (nested == ssbufIf || !nested->hasAttr(kSplittedIf)) {
      return WalkResult::advance();
    }
    if (count == 0) {
      found = nested;
    }
    ++count;
    if (count > 1) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  splittedIf = count == 1 ? found : scf::IfOp();
  return count;
}

// Fill loopToSsbufSplittedIf with each main loop's ssbuffer.if and its
// splitted_if. Return false when a loop fails a wrap restriction.
static bool checkCanWrap(
    ModuleOp module,
    llvm::MapVector<Operation *, SmallVector<std::pair<scf::IfOp, scf::IfOp>>>
        &loopToSsbufSplittedIf) {
  IntegerAttr splittedIfId;
  WalkResult loopWalk = module.walk([&](Operation *loopOp) -> WalkResult {
    if (!isMainLoopOp(loopOp)) {
      return WalkResult::advance();
    }
    // ssbuffer.if sits directly in the main loop body. for uses getBody();
    // while uses the after body. Nested ifs are not ops of this block.
    Block *body = MainLoop(loopOp).getBody();
    SmallVector<std::pair<scf::IfOp, scf::IfOp>> ssbufIfs;
    for (Operation &op : body->without_terminator()) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp || !ifOp->hasAttr(kIf)) {
        continue;
      }
      scf::IfOp splittedIf;
      unsigned splittedIfCount = countSplittedIfs(ifOp, splittedIf);
      if (splittedIfCount > 1) {
        LDBG("Skip wrap: ssbuffer.if contains more than one splitted_if. if: "
             << ifOp << "\n");
        return WalkResult::interrupt();
      }
      if (splittedIfCount == 1) {
        auto id = splittedIf->getAttrOfType<IntegerAttr>(kSplittedIf);
        if (!splittedIfId) {
          splittedIfId = id;
        } else if (splittedIfId.getInt() != id.getInt()) {
          LDBG("Skip wrap: module has more than one splitted_if id. if: "
               << splittedIf << "\n");
          return WalkResult::interrupt();
        }
        if (!computeOnlyInSplitThen(ifOp, splittedIf)) {
          return WalkResult::interrupt();
        }
        ssbufIfs.push_back({ifOp, splittedIf});
        continue;
      }
      if (ssbufIfHasTensorCompute(ifOp)) {
        LDBG("Skip wrap: ssbuffer.if without splitted_if has tensor compute. "
             "if: "
             << ifOp << "\n");
        return WalkResult::interrupt();
      }
    }
    loopToSsbufSplittedIf.insert({loopOp, std::move(ssbufIfs)});
    return WalkResult::advance();
  });
  if (loopWalk.wasInterrupted()) {
    return false;
  }
  if (!splittedIfId) {
    LDBG("Skip wrap: module has no splitted_if.\n");
    return false;
  }
  return true;
}

// Wrap each ssbuffer.if that checkCanWrap accepts. A wrap failure sets
// ERRCODE_FAILED.
void SplittedIfWrapSsbufIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  LDBG("Before SplittedIfWrapSsbufIf:\n" << module << "\n");
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // Main loop -> (ssbuffer.if, its one splitted_if).
  llvm::MapVector<Operation *, SmallVector<std::pair<scf::IfOp, scf::IfOp>>>
      loopToSsbufSplittedIf;
  if (!checkCanWrap(module, loopToSsbufSplittedIf)) {
    return;
  }

  for (auto &[loopOp, ssbufIfs] : loopToSsbufSplittedIf) {
    for (auto [ssbufIf, splittedIf] : ssbufIfs) {
      int wrapReturnCode =
          splittedIfWrapSsbufIf(ssbufIf, splittedIf, loopOp, info);
      if (wrapReturnCode == WRAP_SSBUF_IF_SUCCESS) {
        continue;
      }
      LDBG("splittedIfWrapSsbufIf failed. if: " << ssbufIf << "\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
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
