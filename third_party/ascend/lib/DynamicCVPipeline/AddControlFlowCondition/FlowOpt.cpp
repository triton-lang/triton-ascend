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

#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/FlowOpt.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "third_party/ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "FlowOptPass";
static constexpr int FLOW_OPT_SUCCESS = 0;
static constexpr int FLOW_OPT_FAILED = -1;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
  })

using namespace mlir;
using namespace triton;

// Check if IfOp contains sync_block_wait or sync_block_set op
static bool containsSyncBlockOp(scf::IfOp ifOp) {
  bool containsSyncBlock = false;
  ifOp.walk([&](Operation *innerOp) {
    if (isa<hivm::SyncBlockWaitOp>(innerOp) ||
        isa<hivm::SyncBlockSetOp>(innerOp)) {
      containsSyncBlock = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return containsSyncBlock;
}

// Find ssbuffer.if IfOps in a forOp (non-recursive)
// Only collects IfOps that contain sync_block_wait or sync_block_set
// Returns first and second IfOp found in one traversal
static int findIfOpsInForOp(scf::ForOp forOp, scf::IfOp &firstIfOp,
                            scf::IfOp &secondIfOp) {
  Block *body = forOp.getBody();
  if (!body) {
    LDBG("ForOp body is null.\n");
    return FLOW_OPT_FAILED;
  }

  int count = 0;
  for (Operation &op : *body) {
    // Skip nested operations - only check direct children
    if (op.getBlock() != body) {
      continue;
    }

    if (op.hasAttr(CVPipeline::kIf)) {
      auto ifOp = dyn_cast<scf::IfOp>(&op);
      if (!ifOp) {
        LDBG("Op with ssbuffer.if is not an IfOp: " << op << "\n");
        return FLOW_OPT_FAILED;
      }

      // Only collect IfOps that contain sync_block_wait or sync_block_set
      if (!containsSyncBlockOp(ifOp)) {
        continue;
      }

      count++;
      if (count == 1) {
        firstIfOp = ifOp;
      } else if (count == 2) {
        secondIfOp = ifOp;
        return FLOW_OPT_SUCCESS;
      }
    }
  }

  if (count == 0) {
    LDBG("No ssbuffer.if IfOp found.\n");
    return FLOW_OPT_FAILED;
  } else if (count == 1) {
    LDBG("Only one ssbuffer.if IfOp found.\n");
    return FLOW_OPT_FAILED;
  }

  return FLOW_OPT_SUCCESS;
}

// Build the flow optimization condition
Value FlowOptPass::buildFlowOptCondition(OpBuilder &builder, Location loc,
                                         scf::IfOp firstIfOp, scf::ForOp forOp,
                                         Value originalCondition) {
  // Get counter and loop bounds
  if (!info || !info->cntArgs.count(firstIfOp)) {
    LDBG("No counter found for first IfOp in cntArgs.\n");
    return nullptr;
  }

  Value counter = info->cntArgs[firstIfOp];
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // Build condition: counter >= upperBound
  Value cond1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              counter, upperBound);

  // Build condition: counter >= lowerBound + step*(opt_num)
  // opt_num = min(intra-buffer size - 1, cross-buffer size)
  using BufferCountManager = mlir::triton::BufferCountManager;
  int intraBufNum = BufferCountManager::getInstance().getBufferCountByType(
      BufferCountManager::DepType::IntraCore);
  int crossBufNum = BufferCountManager::getInstance().getBufferCountByType(
      BufferCountManager::DepType::InterCore);
  int optInt = std::min(intraBufNum - 1, crossBufNum);

  Value optNum =
      builder.create<arith::ConstantIntOp>(loc, step.getType(), optInt);
  Value optOffset = builder.create<arith::MulIOp>(loc, step, optNum);
  Value lowerPlusOffset =
      builder.create<arith::AddIOp>(loc, lowerBound, optOffset);
  Value cond2 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              counter, lowerPlusOffset);

  // Combine: original AND (cond1 OR cond2)
  Value orCond = builder.create<arith::OrIOp>(loc, cond1, cond2);
  Value finalCond =
      builder.create<arith::AndIOp>(loc, originalCondition, orCond);

  return finalCond;
}

// Create new IfOp with updated condition
scf::IfOp FlowOptPass::createNewIfOpWithFlowOptCondition(scf::IfOp oldIfOp,
                                                         Value newCondition) {
  Location loc = oldIfOp.getLoc();
  OpBuilder builder(oldIfOp);

  // Get result types and create new IfOp
  SmallVector<Type> resultTypes;
  for (Value result : oldIfOp.getResults()) {
    resultTypes.push_back(result.getType());
  }

  bool hasElse = oldIfOp.getElseRegion().hasOneBlock();
  scf::IfOp newIfOp =
      builder.create<scf::IfOp>(loc, resultTypes, newCondition, hasElse);

  // Copy attributes
  for (auto &attr : oldIfOp->getAttrs()) {
    newIfOp->setAttr(attr.getName(), attr.getValue());
  }

  // Move then block operations
  Block &oldThenBlock = oldIfOp.getThenRegion().front();
  Block &newThenBlock = newIfOp.getThenRegion().front();

  Operation *oldThenYield = nullptr;
  SmallVector<Value> thenYieldOperands;
  for (Operation &op : oldThenBlock) {
    if (isa<scf::YieldOp>(op)) {
      oldThenYield = &op;
      auto yieldOp = cast<scf::YieldOp>(op);
      thenYieldOperands.assign(yieldOp.getOperands().begin(),
                               yieldOp.getOperands().end());
      break;
    }
  }

  for (Operation &op : llvm::make_early_inc_range(oldThenBlock)) {
    if (&op != oldThenYield) {
      op.moveBefore(&newThenBlock, newThenBlock.end());
    }
  }

  OpBuilder thenBuilder(&newThenBlock, newThenBlock.end());
  thenBuilder.create<scf::YieldOp>(loc, thenYieldOperands);

  // Move else block operations if exists
  if (hasElse) {
    Block &oldElseBlock = oldIfOp.getElseRegion().front();
    Block &newElseBlock = newIfOp.getElseRegion().front();

    Operation *oldElseYield = nullptr;
    SmallVector<Value> elseYieldOperands;
    for (Operation &op : oldElseBlock) {
      if (isa<scf::YieldOp>(op)) {
        oldElseYield = &op;
        auto yieldOp = cast<scf::YieldOp>(op);
        elseYieldOperands.assign(yieldOp.getOperands().begin(),
                                 yieldOp.getOperands().end());
        break;
      }
    }

    for (Operation &op : llvm::make_early_inc_range(oldElseBlock)) {
      if (&op != oldElseYield) {
        op.moveBefore(&newElseBlock, newElseBlock.end());
      }
    }

    OpBuilder elseBuilder(&newElseBlock, newElseBlock.end());
    elseBuilder.create<scf::YieldOp>(loc, elseYieldOperands);
  }

  // Replace all uses of old IfOp results with new IfOp results
  for (size_t i = 0; i < oldIfOp.getNumResults(); ++i) {
    oldIfOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));
  }

  return newIfOp;
}

void FlowOptPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter FlowOpt pass.\n");
  SmallVector<scf::ForOp> mainLoopForOps;
  module.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(CVPipeline::kMainLoop)) {
      mainLoopForOps.push_back(forOp);
    }
  });

  for (scf::ForOp forOp : mainLoopForOps) {
    // Step 1: Check if this forOp has ssbuffer.flowOpt attribute
    if (!forOp->hasAttr(CVPipeline::kFlowOpt)) {
      continue;
    }
    // Step 2: Find first and second ssbuffer.if IfOps
    scf::IfOp firstIfOp = nullptr;
    scf::IfOp secondIfOp = nullptr;
    if (findIfOpsInForOp(forOp, firstIfOp, secondIfOp) != FLOW_OPT_SUCCESS) {
      LDBG("Failed to find first and second ssbuffer.if IfOps.\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }

    // Step 3: Get the original condition of the second IfOp
    Value originalCondition = secondIfOp.getCondition();
    if (!originalCondition) {
      LDBG("Second IfOp has no condition.\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }

    // Step 4: Build the new flow optimization condition
    OpBuilder builder(secondIfOp);
    Location loc = secondIfOp.getLoc();
    Value newCondition = buildFlowOptCondition(builder, loc, firstIfOp, forOp,
                                               originalCondition);
    if (!newCondition) {
      LDBG("Failed to build flow optimization condition.\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }

    // Step 5: Create new IfOp with the updated condition
    scf::IfOp newIfOp =
        createNewIfOpWithFlowOptCondition(secondIfOp, newCondition);
    if (!newIfOp) {
      LDBG("Failed to create new IfOp.\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }

    // Step 6: Update cntArgs mapping if the second IfOp has a counter
    if (info->cntArgs.count(secondIfOp)) {
      Value counter = info->cntArgs[secondIfOp];
      info->cntArgs.erase(secondIfOp);
      info->cntArgs[newIfOp] = counter;
    }

    secondIfOp.erase();
  }

  LDBG("Exit FlowOpt pass.\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createFlowOptPass() {
  return std::make_unique<FlowOptPass>();
}

} // namespace triton
} // namespace mlir
