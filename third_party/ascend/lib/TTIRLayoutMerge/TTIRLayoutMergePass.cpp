//===- TTIRLayoutMergePass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/TTIRLayoutMerge/TTIRLayoutMergePass.h"

#include "ascend/include/TTIRLayoutMerge/RowCoalescing.h"
#include "ascend/include/TritonToLinalg/ImplicitPermute.h"
#include "ascend/include/TritonToLinalg/StridedAxisCoalescing.h"
#include "ascend/include/TritonToLinalg/TileChunkCoalescing.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir::triton {
namespace {

constexpr llvm::StringLiteral kLayoutMergeAppliedAttr =
    "ta.ttir_layout_merge.applied";

class TTIRLayoutMergePass
    : public ::impl::TTIRLayoutMergeBase<TTIRLayoutMergePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(kLayoutMergeAppliedAttr))
      return;

    RewritePatternSet implicitPermutePatterns(&getContext());
    implicitPermutePatterns
        .add<ImplicitPermute::LoadConverter, ImplicitPermute::StoreConverter,
             ImplicitPermute::AtomicRMWConverter,
             ImplicitPermute::AtomicCASConverter>(&getContext());
    // These converters intentionally return failure for memory operations
    // whose pointer graph does not describe an implicit permutation.  Match
    // the historical TritonToLinalg behavior: keep any successful rewrites and
    // continue with the independent coalescing analyses.
    (void)applyPatternsGreedily(module, std::move(implicitPermutePatterns));

    // Coalescing must precede AutoBlockify: both analyses recognize the
    // original tt.get_program_id graph, while AutoBlockify replaces it with a
    // loop induction variable and reconstructed logical program ids.
    StridedAxisCoalescing::rewriteStridedAxisCoalesce(module);
    TileChunkCoalescing::rewriteTileChunkCoalesce(module);
    // RowCoalescing is the general row-wise fallback.  Keep it after the two
    // more specific layout rewrites so it cannot steal their proven access
    // patterns.  All three transformations share hacc.coalesce_factor/axis,
    // therefore a successful earlier rewrite makes this call a no-op.
    RowCoalescing::rewriteRowCoalesceForLayoutMerge(module);

    PassManager cleanup(&getContext(), module.getOperationName());
    cleanup.addPass(createCSEPass());
    cleanup.addPass(createCanonicalizerPass());
    if (failed(runPipeline(cleanup, module))) {
      module.emitError("failed to canonicalize post-layout TTIR");
      return signalPassFailure();
    }

    module->setAttr(kLayoutMergeAppliedAttr, UnitAttr::get(&getContext()));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTTIRLayoutMergePass() {
  return std::make_unique<TTIRLayoutMergePass>();
}

} // namespace mlir::triton
