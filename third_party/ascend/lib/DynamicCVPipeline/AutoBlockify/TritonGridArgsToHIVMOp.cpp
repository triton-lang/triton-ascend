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

#include "ascend/include/DynamicCVPipeline/AutoBlockify/TritonGridArgsToHIVMOp.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "triton-grid-args-to-hivm-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// TritonGridArgsToHIVMOp
//===----------------------------------------------------------------------===//
static inline constexpr int kProgramNumArgsNum = 3;
static inline constexpr int kProgramIdArgsNum = 3;

void TritonGridArgsToHIVMOpPass::getDependentDialects(DialectRegistry &registry) const {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<hivm::HIVMDialect>();
    registry.insert<annotation::AnnotationDialect>();
}

/// This pass convert global kernel function arguments to hivm op

// The launch grid of triton is always 3D while hivm::get_block_idx is just 1D.
// So the following wanna transform 1D index to 3D.
//
// Currently, shape of triton launch grid, like [x, y, z], will be really passed
// as final three i32 args of global kernel.
// And before this pass, final six i32 args of global kernel represent orderly
// three PROGRAM_NUM_ARGS and three PROGRAM_ID_ARGS. Therefore PROGRAM_NUM_ARGS
// is equivalent to the 3 actual args, [x, y, z], and PROGRAM_ID_ARGS will
// later be erased from func args.
//
// The program_id decode order follows the launch order of the target arch.
// idx = hivm::get_block_idx
//
// Reg-based (A5) keeps Triton's x-fastest launch order:
// idx = program_id_0
//     + program_id_1 * program_num_0(x)
//     + program_id_2 * program_num_0(x) * program_num_1(y)
// so,
// program_id_0 = idx // (1)     mod x
// program_id_1 = idx // (x)     mod y
// program_id_2 = idx // (x * y) mod z
//
// Mem-based (A3) keeps the legacy z-fastest launch order:
// idx = program_id_0 * program_num_1(y) * program_num_2(z)
//     + program_id_1 * program_num_2(z)
//     + program_id_2
// so,
// program_id_2 = idx // (1)     mod z
// program_id_1 = idx // (z)     mod y
// program_id_0 = idx // (y * z) mod x
//
// FixMe: How to take advantage of hivm::get_block_num?
LogicalResult replaceProgramID(func::FuncOp funOp, IRRewriter &rewriter) {
  auto args = funOp.getArguments();
  auto argNum = funOp.getNumArguments();
  // Verify whether there exist final 6 args to express BLOCK info
  if (argNum < kProgramIdArgsNum + kProgramNumArgsNum) {
    funOp.emitError("arguments program id or program num are missing");
    return failure();
  }

  // Verify type of final 6 args.
  for (auto itr = (args.end() - (kProgramIdArgsNum + kProgramNumArgsNum));
       itr != args.end(); itr++) {
    if ((*itr).getType() != rewriter.getI32Type()) {
      funOp.emitError(
          "incompatible types of arguments program id or program num");
      return failure();
    }
  }

  Block &block = funOp.getBody().front();
  rewriter.setInsertionPointToStart(&block);
  mlir::Location loc = block.front().getLoc();
  auto *argEnd = args.end();
  auto progID0 = argEnd[-(kProgramNumArgsNum + kProgramIdArgsNum)];
  auto progID1 = argEnd[-(kProgramNumArgsNum + kProgramIdArgsNum) + 1];
  auto progID2 = argEnd[-(kProgramNumArgsNum + kProgramIdArgsNum) + 2];
  auto tempMul = rewriter.create<arith::MulIOp>(loc, progID0, progID1);
  auto logicBlockNum = rewriter.create<arith::MulIOp>(loc, tempMul, progID2);
  auto mark = rewriter.create<annotation::MarkOp>(loc, logicBlockNum);
  mark->setAttr(kLogicalBlockNumAttr, rewriter.getUnitAttr());
  // Replace used program_id args
  auto hivmOp =
      rewriter.create<hivm::GetBlockIdxOp>(loc, rewriter.getI64Type());
  Value castedBlockID = rewriter.create<arith::TruncIOp>(
      loc, rewriter.getI32Type(), hivmOp.getResult());
  Value accumulateShape = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
  auto argProgNumAxis0 =
      (args.end() - (kProgramNumArgsNum + kProgramIdArgsNum));
  for (int i = 0; i < kProgramIdArgsNum; ++i) {
    auto curProgID = args.end() - (kProgramIdArgsNum) + i;

    auto indexAlongCurAxis =
        rewriter.create<arith::DivSIOp>(loc, castedBlockID, accumulateShape);
    auto realIndexAlongCurAxis = rewriter.create<arith::RemSIOp>(
        loc, indexAlongCurAxis, *(argProgNumAxis0 + i));
    rewriter.replaceAllUsesWith(*curProgID, realIndexAlongCurAxis);
    if (i != kProgramIdArgsNum - 1) {
      accumulateShape = rewriter.create<arith::MulIOp>(loc, accumulateShape,
                                                       *(argProgNumAxis0 + i));
    }
  }

  return success();
}

void TritonGridArgsToHIVMOpPass::runOnOperation() {
  ModuleOp module = getOperation();
  LDBG("Enter TritonGridArgsToHIVMOp pass\n" << module);
  module.walk([&](func::FuncOp funOp) {
    if (!funOp->hasAttr("global_kernel")) {
      return WalkResult::advance();
    }

    MLIRContext *ctx = funOp->getContext();
    IRRewriter rewriter(ctx);
    if (failed(replaceProgramID(funOp, rewriter))) {
      LDBG("failed to replaceProgramID");
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  LDBG("After TritonGridArgsToHIVMOp pass\n" << module);
}

std::unique_ptr<Pass> mlir::triton::createTritonGridArgsToHIVMOpPass() {
  return std::make_unique<TritonGridArgsToHIVMOpPass>();
}