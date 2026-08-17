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

#include "ascend/include/DynamicCVPipeline/AutoBlockify/AutoBlockifyParallelLoop.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::hivm;

static constexpr llvm::StringLiteral BlockifyLoopAttrName = "autoblockify.subloop";

static constexpr const char *DEBUG_TYPE = "auto-blockify-parallel-loop";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

namespace {

void traceExceptions(Value input, SmallPtrSet<Operation *, 4> &exceptions) {
  if (isa<BlockArgument>(input)) {
    return;
  }
  Operation *curOp = input.getDefiningOp();
  if (!curOp)
    return;
  exceptions.insert(curOp);
  for (auto opr : curOp->getOperands()) {
    traceExceptions(opr, exceptions);
  }
}

void replaceBlockIdUsers(IRRewriter &rewriter,
                         hivm::GetBlockIdxOp getBlockIdxOp, Value iv,
                         Value logicBlockNum, Operation *castedBlockID, Value blockifyV2) {
  rewriter.setInsertionPointAfterValue(iv);
  auto loc = getBlockIdxOp->getLoc();
  auto mulOp = rewriter.create<arith::MulIOp>(loc, iv, blockifyV2);
  auto castedMulOp =
      rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), mulOp);
  rewriter.replaceAllUsesExcept(getBlockIdxOp, castedMulOp, castedBlockID);
}

LogicalResult loopOnLogicBlock(func::FuncOp funcOp, IRRewriter &rewriter, int aicoreNum) {
  LDBG("aicoreNum : " << aicoreNum);
  auto &entryBlock = funcOp.getBody().front();
  mlir::Location loc = entryBlock.front().getLoc();
  hivm::GetBlockIdxOp getBlockIdxOp;
  Value logicBlockNum;
  SmallPtrSet<Operation *, 4> exceptions;
  SmallVector<Operation *> opsToMove;
  for (auto &op : entryBlock) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(op)) {
      if (markOp->hasAttr(kLogicalBlockNumAttr)) {
        logicBlockNum = markOp->getOperand(0);
        continue;
      }
    }
    if (!isa<func::ReturnOp>(op)) {
      opsToMove.push_back(&op);
    }
    if (isa<hivm::GetBlockIdxOp>(op)) {
      getBlockIdxOp = dyn_cast<hivm::GetBlockIdxOp>(op);
      rewriter.setInsertionPointAfter(getBlockIdxOp);
    }
  }
  if (!logicBlockNum)
    return funcOp->emitError("Logical Block number not found");
  if (!getBlockIdxOp) {
    return success();
  }

  traceExceptions(logicBlockNum, exceptions);
  exceptions.insert(getBlockIdxOp);
  const int intBits = 32;
  
  int physicalBlockNum = aicoreNum;
  if (physicalBlockNum <= 0) {
    return funcOp->emitError("Physical block num cannot be inferred");
  }
  
  Value physicalBlockNumValue = rewriter.create<arith::ConstantIntOp>(
      loc, physicalBlockNum, intBits);
  Value upperBound = logicBlockNum;
  int blockifyNum = 1;
  if (auto blockifyAttr = funcOp->getAttr("auto_blockify_size"))
    blockifyNum = cast<IntegerAttr>(blockifyAttr).getInt();
  Value blockifyV2 =
      rewriter.create<arith::ConstantIntOp>(loc, blockifyNum, intBits);
  if (blockifyNum > 1)
    upperBound =
        rewriter.create<arith::CeilDivSIOp>(loc, upperBound, blockifyV2);
  auto blockID = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(),
                                                  getBlockIdxOp);
  auto forOp = rewriter.create<scf::ForOp>(loc, blockID, upperBound,
                                           physicalBlockNumValue);

  Block *loopBody = forOp.getBody();
  Operation *yieldOp = loopBody->getTerminator();
  for (Operation *op : opsToMove) {
    if (exceptions.count(op) == 0) {
      op->moveBefore(yieldOp);
    }
  }
  replaceBlockIdUsers(rewriter, getBlockIdxOp, forOp.getInductionVar(),
                      logicBlockNum, blockID, blockifyV2);
  auto unit = UnitAttr::get(forOp->getContext());
  forOp->setAttr(BlockifyLoopAttrName, unit);
  return success();
}

} // namespace

AutoBlockifyParallelLoopPass::AutoBlockifyParallelLoopPass(
    const AutoBlockifyParallelLoopPassOptions &options)
    : AutoBlockifyParallelLoopPassBase(options) {}

void AutoBlockifyParallelLoopPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = moduleOp->getContext();
  IRRewriter rewriter(ctx);
  
  moduleOp.walk([&](func::FuncOp funOp) {
    if (this->aicoreNum <= 0) {
      LDBG("aicoreNum is illegal: " << this->aicoreNum);
      return WalkResult::interrupt();
    }

    if (!funOp->hasAttr("global_kernel")) {
      llvm::outs() << "no global_kernel\n";
      return WalkResult::advance();
    }

    llvm::outs() << "has global_kernel\n";
    if (failed(loopOnLogicBlock(funOp, rewriter, this->aicoreNum))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<Pass> mlir::triton::createAutoBlockifyParallelLoopPass(
    const AutoBlockifyParallelLoopPassOptions &options)
{
    return std::make_unique<AutoBlockifyParallelLoopPass>(options);
}