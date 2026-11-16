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

// Record else-yield values that must be updated.
// for maps the block counter to counter+step. while clones ssbuffer.while_arg
// ops and maps each else operand to the clone of the then value in that slot.
static int buildElseUpdate(Operation *loopOp, ControlFlowConditionInfo *info,
                           scf::IfOp ssbufIf, OpBuilder &elseBuilder,
                           ArrayRef<Value> elsePassthrough, Location loc,
                           IRMapping &mapping) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    if (!info->cntArgs.count(ssbufIf)) {
      LDBG("Failed to build for new-if else: ssbuffer.if has no cntArgs. if: "
           << ssbufIf << "\n");
      return WRAP_SSBUF_IF_FAILED;
    }
    Value counter = info->cntArgs[ssbufIf];
    if (!counter) {
      LDBG("Failed to build for new-if else: counter is null. if: " << ssbufIf
                                                                    << "\n");
      return WRAP_SSBUF_IF_FAILED;
    }
    Value inc =
        elseBuilder.create<arith::AddIOp>(loc, counter, forOp.getStep());
    mapping.map(counter, inc);
    return WRAP_SSBUF_IF_SUCCESS;
  }

  auto thenYield = cast<scf::YieldOp>(ssbufIf.thenBlock()->getTerminator());
  for (Operation &op : *ssbufIf.thenBlock()) {
    if (!op.hasAttr(kWhileArg)) {
      continue;
    }
    Operation *cloned = elseBuilder.clone(op, mapping);
    elseBuilder.setInsertionPointAfter(cloned);
  }
  for (auto [idx, thenVal] : llvm::enumerate(thenYield.getOperands())) {
    if (!mapping.contains(thenVal)) {
      continue;
    }
    mapping.map(elsePassthrough[idx], mapping.lookup(thenVal));
  }
  return WRAP_SSBUF_IF_SUCCESS;
}

// Fill elseBlock from elsePassthrough. An else operand in mapping is replaced
// by the value just built in elseBlock.
static int buildNewIfElse(Operation *loopOp, ControlFlowConditionInfo *info,
                          scf::IfOp ssbufIf, Block *elseBlock,
                          ArrayRef<Value> elsePassthrough, Location loc) {
  OpBuilder elseBuilder(elseBlock, elseBlock->end());
  if (elseBlock->mightHaveTerminator()) {
    elseBuilder.setInsertionPoint(elseBlock->getTerminator());
  }

  IRMapping mapping;
  if (buildElseUpdate(loopOp, info, ssbufIf, elseBuilder, elsePassthrough, loc,
                      mapping) != WRAP_SSBUF_IF_SUCCESS) {
    return WRAP_SSBUF_IF_FAILED;
  }

  SmallVector<Value> elseVals(elsePassthrough.begin(), elsePassthrough.end());
  for (Value &val : elseVals) {
    if (!mapping.contains(val)) {
      continue;
    }
    val = mapping.lookup(val);
  }
  if (elseVals.size() != ssbufIf.getNumResults()) {
    LDBG("Failed to build new-if else: else value count "
         << elseVals.size() << ", results " << ssbufIf.getNumResults()
         << ". if: " << ssbufIf << "\n");
    return WRAP_SSBUF_IF_FAILED;
  }
  if (elseBlock->mightHaveTerminator()) {
    elseBlock->getTerminator()->erase();
  }
  OpBuilder yieldBuilder(elseBlock, elseBlock->end());
  yieldBuilder.create<scf::YieldOp>(loc, elseVals);
  return WRAP_SSBUF_IF_SUCCESS;
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

  // Defs precede uses in this block, so one reverse pass propagates a staying
  // user back to its producers.
  for (Operation *op : llvm::reverse(sorted)) {
    if (!moveSet.contains(op)) {
      continue;
    }
    for (OpOperand &use : op->getUses()) {
      Operation *user = use.getOwner();
      if (condOps.contains(user) && !moveSet.contains(user)) {
        moveSet.erase(op);
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
  if (buildNewIfElse(loopOp, info, ssbufIf, newIfOp.elseBlock(),
                     elsePassthrough, loc) != WRAP_SSBUF_IF_SUCCESS) {
    return WRAP_SSBUF_IF_FAILED;
  }
  unwrapSplittedIf(splittedIf);
  Block *thenBlock = newIfOp.thenBlock();
  if (thenBlock->mightHaveTerminator()) {
    thenBlock->getTerminator()->erase();
  }
  OpBuilder thenBuilder(thenBlock, thenBlock->end());
  thenBuilder.create<scf::YieldOp>(loc, ssbufIf.getResults());
  // Retarget uses of ssbufIf's results that sit outside newIfOp to newIfOp's
  // results.
  for (auto [innerRes, outerRes] :
       llvm::zip(ssbufIf.getResults(), newIfOp.getResults())) {
    innerRes.replaceUsesWithIf(outerRes, [&](OpOperand &use) {
      return !newIfOp->isProperAncestor(use.getOwner());
    });
  }
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
