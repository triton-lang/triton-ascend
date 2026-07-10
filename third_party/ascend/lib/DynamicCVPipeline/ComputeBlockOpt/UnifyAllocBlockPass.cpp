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

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeSize.h"

static constexpr const char *DEBUG_TYPE = "unify-alloc-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

struct FillInfo {
  linalg::FillOp fillOp;
  scf::IfOp parentIf;
  bool needsSplit;
};

/**
 * @brief Collect direct users of alloc result
 *
 * This function collects all operations that directly use the alloc result,
 * excluding linalg.fill operations because linalg.fill uses BlockArgument
 * (i.e., the outs parameter) rather than SSA value dependency.
 *
 * @param allocResult The result value of memref.alloc
 * @return SmallVector<Operation*> List of direct user operations
 *
 * @note linalg.fill is a DestinationStyleOp where:
 *       - ins(%v : f16) is the input value
 *       - outs(%alloc : memref) is the target memory location (BlockArgument)
 *       Therefore, linalg.fill does not appear in allocResult.getUsers().
 */
static SmallVector<Operation *> collectDirectUsers(Value allocResult) {
  SmallVector<Operation *> directUsers;
  for (Operation *user : allocResult.getUsers()) {
    if (!isa<linalg::FillOp>(user)) {
      directUsers.push_back(user);
    }
  }
  return directUsers;
}

/**
 * @brief Get common block_id from a list of operations
 *
 * Checks whether all operations have the same block_id.
 * If they are the same, returns that block_id; otherwise returns std::nullopt.
 *
 * Special handling for memref.subview: if direct users include memref.subview,
 * we look through to memref.copy. If there's exactly one memref.copy reachable,
 * return its block_id (not the subview's). If multiple copies exist, return
 * error.
 *
 * @param ops List of operations to check
 * @return std::optional<int> Returns common block_id if all are the same,
 *         otherwise returns std::nullopt
 */
static LogicalResult getCommonBlockId(ArrayRef<Operation *> ops, int &blockId) {
  if (ops.empty()) {
    return failure();
  }

  llvm::SmallDenseSet<int, 4> copyBlockIds;
  for (Operation *op : ops) {
    if (isa<ViewLikeOpInterface>(op)) {
      for (auto *user : op->getUsers()) {
        if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
          if (auto blockId = CVPipeline::getOpBlockId(copyOp)) {
            copyBlockIds.insert(*blockId);
          }
        }
      }
    }
  }

  if (copyBlockIds.size() > 1) {
    LOG_DEBUG("[getCommonBlockId] Multiple block_ids found, ops: ");
    for (Operation *op : ops) {
      LOG_DEBUG("  " << *op);
    }
    return failure();
  }

  if (copyBlockIds.empty()) {
    LOG_DEBUG("[getCommonBlockId] There are not CopyOp!");
    for (Operation *op : ops) {
      LOG_DEBUG("  " << *op);
    }
    return failure();
  }

  blockId = *copyBlockIds.begin();
  return success();
}

/**
 * @brief Collect predecessor operations in a block for a given value
 *
 * This function traces back through SSA dependencies to find all operations
 * that affect the given startValue within a specific block.
 *
 * Special handling for scf.for loop-carried dependencies:
 * When an operand is a BlockArgument from scf.for iter_arg, we also trace
 * through the yieldOp to find the operation that provides the yielded value.
 * This ensures we capture operations that update loop-carried variables
 * (e.g., %79 that updates %arg19).
 *
 * @param startValue The value to trace back from
 * @param block The block to search within
 * @return SmallVector<Operation*> List of predecessor operations
 */
static SmallVector<Operation *> collectBlockPredecessors(Value startValue,
                                                         Block *block) {
  SmallVector<Operation *> result;
  SmallVector<Operation *> toProcess;

  auto addToProcess = [&](Operation *op) {
    if (auto *ancestorInBlock = CVPipeline::getAncestorInBlock(op, block)) {
      if (!llvm::is_contained(result, ancestorInBlock)) {
        toProcess.push_back(ancestorInBlock);
      }
    }
  };

  if (auto *condDefOp = startValue.getDefiningOp()) {
    addToProcess(condDefOp);
  }

  while (!toProcess.empty()) {
    auto *op = toProcess.pop_back_val();
    if (llvm::is_contained(result, op)) {
      continue;
    }
    result.push_back(op);

    for (auto operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        // SSA operand: trace to its defining operation
        addToProcess(defOp);
      } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        // Block argument: check if it's from scf.for iter_arg
        Operation *parentOp = blockArg.getOwner()->getParentOp();
        if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
          unsigned argIdx = blockArg.getArgNumber();
          // argIdx == 0 is the loop variable, not iter_arg; skip invalid
          // indices
          if (argIdx == 0 || argIdx > forOp.getInitArgs().size()) {
            continue;
          }
          Operation *yieldOp = forOp.getBody()->getTerminator();
          if (!isa<scf::YieldOp>(yieldOp) ||
              argIdx > yieldOp->getNumOperands()) {
            continue;
          }
          // Get the value yielded for this iter_arg and trace its defining op
          Value yieldedValue = yieldOp->getOperand(argIdx - 1);
          if (auto *yieldedDef = yieldedValue.getDefiningOp()) {
            addToProcess(yieldedDef);
          }
        }
      }
    }
  }
  return result;
}

/**
 * @brief Find linalg.fill operation that uses alloc as outs inside scf.if
 *
 * This function searches for linalg.fill operations that satisfy:
 * 1. Use the given alloc result as its outs parameter
 * 2. Located inside an scf.if operation (then branch only)
 * 3. The scf.if has no else region (withElseRegion=false)
 *
 * @param allocResult The alloc result value to search for
 * @return FillInfo Structure containing fillOp and parentIf if found
 */
static FillInfo findFillOpInSCFIf(Value allocResult) {
  FillInfo info;
  for (Operation *user : allocResult.getUsers()) {
    auto fillOp = dyn_cast<linalg::FillOp>(user);
    if (!fillOp) {
      continue;
    }

    auto parentIf = fillOp->getParentOfType<scf::IfOp>();
    if (!parentIf) {
      continue;
    }

    if (!parentIf.getElseRegion().empty()) {
      continue;
    }

    Block *parentBlock = fillOp->getBlock();
    if (parentBlock != &parentIf.getThenRegion().front()) {
      continue;
    }

    if (fillOp.getDpsInits()[0] == allocResult) {
      info.fillOp = fillOp;
      info.parentIf = parentIf;
      return info;
    }
  }
  return info;
}

/**
 * @brief Check if scf.if needs to be split
 *
 * Determines whether the scf.if operation containing linalg.fill needs to be
 * split. Split is needed when the if branch contains multiple operations (not
 * just linalg.fill).
 *
 * @param info FillInfo structure containing fillOp and parentIf
 * @return bool Returns true if split is needed, false otherwise
 *
 * @note Split logic:
 *       - If branch only has linalg.fill (+ scf.yield terminator), no split
 * needed
 *       - If branch has other operations besides linalg.fill, split needed
 */
static bool needsSplitIf(const FillInfo &info) {
  if (!info.fillOp || !info.parentIf) {
    return false;
  }

  Block *fillBlock = info.fillOp->getBlock();
  int opCount = 0;
  for (auto &op : fillBlock->without_terminator()) {
    (void)op;
    opCount++;
  }
  return opCount > 1;
}

/**
 * @brief Split scf.if into two separate scf.if blocks
 *
 * When an scf.if branch contains multiple operations (linalg.fill + other ops),
 * this function splits it into two scf.if blocks:
 * - One containing only linalg.fill (will be unified)
 * - One containing other operations (keeps original block_id)
 *
 * @param info FillInfo structure containing fillOp and parentIf
 * @return FillInfo Updated FillInfo pointing to the new fill-only scf.if
 *
 * @note Split pattern:
 *       Before:
 *         scf.if %cond {
 *           linalg.fill {block_id=8} ins(%v) outs(%alloc)  // keep
 *           arith.addf {block_id=12} %x, %y               // move to new scf.if
 *         } {hivm.unlikely_condition}
 *
 *       After:
 *         scf.if %cond {
 *           linalg.fill {block_id=8} ins(%v) outs(%alloc)  // keep
 *         } {hivm.unlikely_condition}
 *
 *         scf.if %cond {
 *           arith.addf {block_id=12} %x, %y               // new scf.if
 *         } {hivm.unlikely_condition}
 */
static FillInfo splitSCFIfIfNeeded(FillInfo &info) {
  Block *originalBlock = info.fillOp->getBlock();
  Operation *fillOp = info.fillOp.getOperation();
  scf::IfOp originalIf = info.parentIf;
  Value cond = originalIf.getCondition();
  Location loc = originalIf.getLoc();

  SmallVector<Operation *> otherOps;
  for (auto &op : originalBlock->without_terminator()) {
    if (&op != fillOp) {
      otherOps.push_back(&op);
    }
  }

  if (otherOps.empty()) {
    return info;
  }

  DictionaryAttr originalAttrs = originalIf->getAttrDictionary();

  OpBuilder builder(originalIf);

  fillOp->moveBefore(originalIf.getOperation()->getNextNode());
  builder.setInsertionPointAfter(fillOp);

  auto newFillIf =
      builder.create<scf::IfOp>(loc, cond, /*withElseRegion=*/false);
  if (originalAttrs) {
    for (auto attr : originalAttrs) {
      newFillIf->setAttr(attr.getName(), attr.getValue());
    }
  }

  fillOp->moveBefore(newFillIf.getThenRegion().front().getTerminator());

  info.parentIf = newFillIf;
  return info;
}

/**
 * @brief Try to unify block_id for a single alloc operation
 *
 * @param allocOp The memref.alloc operation to process
 * @param memGraph Memory dependence graph for cycle detection
 * @return LogicalResult Returns success if unification was performed, failure
 * otherwise
 */
static LogicalResult
tryUnifyForAlloc(memref::AllocOp allocOp,
                 const CVPipeline::MemoryDependenceGraph &memGraph,
                 CVPipeline::ComputeBlockIdManager &bm) {
  // Step1: Collect direct users (excluding linalg.fill)
  Value allocResult = allocOp.getResult();
  LOG_DEBUG("[tryUnifyForAlloc] start from allocOp: " << *allocOp);
  SmallVector<Operation *> directUsers = collectDirectUsers(allocResult);
  if (directUsers.empty()) {
    return success();
  }

  // Step2: Find linalg.fill inside scf.if that uses this alloc
  FillInfo fillInfo = findFillOpInSCFIf(allocResult);
  if (!fillInfo.fillOp) {
    return success();
  }
  LOG_DEBUG(
      "[tryUnifyForAlloc] Found fillOp in scf.if: " << *fillInfo.parentIf);

  // Step3: Check if all direct users have the same block_id
  int targetBlockId;
  if (failed(getCommonBlockId(directUsers, targetBlockId))) {
    LOG_DEBUG("allocOp has copyOp from different Block");
    return failure();
  }
  LOG_DEBUG("[getSameBlockId] GetSameBlockId: " << targetBlockId);

  // Step4: Split if scf.if contains multiple operations
  if (needsSplitIf(fillInfo)) {
    LOG_DEBUG("[needsSplitIf] SCF.IF need split ");
    fillInfo = splitSCFIfIfNeeded(fillInfo);
  }

  // Step5: Collect predecessor_ops for scf.if condition
  SmallVector<Operation *> conditionOps = collectBlockPredecessors(
      fillInfo.parentIf.getCondition(), fillInfo.parentIf->getBlock());

  // Step6: Cycle detection and block_id assignment with fallback
  SmallVector<Operation *> coreOps = {
      allocOp.getOperation(),
      fillInfo.fillOp.getOperation(),
      fillInfo.parentIf.getOperation(),
  };
  coreOps.append(directUsers);
  SmallVector<Operation *> allOps = coreOps;
  // allOps include coreOps and scf.if_condition's predecessor_ops
  allOps.append(conditionOps);

  if (CVPipeline::willCreateCycle(allOps, memGraph, targetBlockId, bm)) {
    LOG_DEBUG("[Cycle detection] First time: Find cycle with conditionOps, "
              "retry without conditionOps");
    if (CVPipeline::willCreateCycle(coreOps, memGraph, targetBlockId, bm)) {
      LOG_DEBUG("[Cycle detection] Second time: Find Cycle, have unsupport IR! "
                "Should Check!!");
      return success();
    }
    for (auto *op : coreOps) {
      bm.updateBlockId(op, targetBlockId);
    }
  } else {
    for (auto *op : allOps) {
      bm.updateBlockId(op, targetBlockId);
    }
  }
  return success();
}

} // anonymous namespace

class UnifyAllocBlockPass
    : public PassWrapper<UnifyAllocBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnifyAllocBlockPass)

  UnifyAllocBlockPass() = default;

  StringRef getArgument() const override { return "unify-alloc-block"; }

  StringRef getDescription() const override {
    return "Unify block_id for memref.alloc, scf.if with linalg.fill, and "
           "memref.subview operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LOG_DEBUG("Before: " << *module << "\n");
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    auto bm = CVPipeline::ComputeBlockIdManager(module);

    llvm::SmallVector<memref::AllocOp> allocOps;

    module.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });

    for (memref::AllocOp allocOp : allocOps) {
      if (failed(tryUnifyForAlloc(allocOp, memGraph, bm))) {
        signalPassFailure();
      }
    }

    LOG_DEBUG("After: " << *module << "\n");
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnifyAllocBlockPass() {
  return std::make_unique<UnifyAllocBlockPass>();
}

void registerUnifyAllocBlockPass() {
  PassRegistration<UnifyAllocBlockPass> reg;
}

} // namespace triton
} // namespace mlir
