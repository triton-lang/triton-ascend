//===- UpliftWhileToFor.cpp - Uplift scf.while to scf.for ----------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Thin downstream wrapper around upstream
// `mlir::scf::populateUpliftWhileToForPatterns`, for DynamicCVPipeline.
// Mirrors bishengir HFusion's `hfusion-uplift-while-to-for` pass
// (AscendNPU-IR/.../HFusion/Transforms/UpliftWhileToFor.cpp).
//
// The upstream pattern only fires when the `scf.while` matches a
// canonical for-shape:
//   * `before` block: a single `arith.cmpi` feeding `scf.condition`
//   * `after`  block: a linear `arith.addi` on the induction variable
// All other while-loops (e.g. data-driven exit conditions) are left
// untouched and continue to use the existing DynamicCVPipeline whileop
// path from 3768ccd4.
//
//===----------------------------------------------------------------------===//

#include "ascend/include/DynamicCVPipeline/UpliftWhileToFor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_UPLIFTWHILETOFOR
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

struct UpliftWhileToForPass
    : public impl::UpliftWhileToForBase<UpliftWhileToForPass> {
  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    scf::populateUpliftWhileToForPatterns(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::triton::createUpliftWhileToForPass() {
  return std::make_unique<UpliftWhileToForPass>();
}
