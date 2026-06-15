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

#include "AnnotateUserCasts/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ANNOTATEUSERCASTS
#include "ascend/include/AnnotateUserCasts/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {
struct AnnotateUserCastsPass
    : public mlir::triton::impl::AnnotateUserCastsBase<
          AnnotateUserCastsPass> {
  void runOnOperation() override;
};
} // namespace

static bool isCastOp(Operation *op) {
  return isa<triton::FpToFpOp, triton::BitcastOp,
             arith::TruncIOp, arith::TruncFOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp,
             arith::SIToFPOp, arith::FPToSIOp, 
             arith::FPToUIOp, arith::UIToFPOp>(op);
}

static bool hasExistingCastAnnotation(Operation *op) {
  assert(isCastOp(op));
  if (op->getAttrOfType<StringAttr>("cast.source")) {
    return true;
  }
  return false;
}

void AnnotateUserCastsPass::runOnOperation() {
  auto module = getOperation();
  IRRewriter rewriter(module.getContext()); 

  module.walk([&rewriter](triton::FuncOp func) {
    func.walk([&rewriter](Operation *op) {
      if (!isCastOp(op)) {
        return WalkResult::advance();
      }
      if (hasExistingCastAnnotation(op)) {
        return WalkResult::advance();
      }

      op->setAttr("cast.source", rewriter.getStringAttr("user"));
      return WalkResult::advance();
    });
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createAnnotateUserCastsPass() {
  return std::make_unique<AnnotateUserCastsPass>();
}