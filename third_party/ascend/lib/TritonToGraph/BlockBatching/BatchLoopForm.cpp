// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopForm.h"

#include "TritonToGraph/BlockBatching/BatchRequest.h"

namespace mlir::triton::cfg {

BatchResult<scf::ForOp> checkBatchLoopForm(Operation *function,
                                           BatchVaryingPredicate isVarying) {
  scf::ForOp programLoop;
  BatchDiagnostic first;
  auto walk = function->walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
    scf::ForOp outer = programLoop;
    // A loop nested inside a program-axis loop must have its bounds and step
    // defined outside that outer loop. Otherwise the F lanes could require
    // different inner-loop lengths. This checks definition sites, not dataflow,
    // so it conservatively rejects even invariant constants defined inside the
    // outer body until canonicalization hoists them.
    if (outer && outer->isProperAncestor(loop)) {
      for (Value bound :
           {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()})
        if (outer->isAncestor(bound.getParentBlock()->getParentOp())) {
          first = snapshotBatchDiagnostic(loop, "loop_form",
                                          "INNER_TRIP_COUNT_NOT_INVARIANT");
          return WalkResult::interrupt();
        }
      // Bounds defined outside the outer loop but varying with pid form a
      // nested program-axis loop; reject it below.
    }
    // Any varying bound or step makes this a program-axis loop. Checking only
    // the lower bound would miss a constant-lower/varying-upper loop whose
    // lanes have different trip counts. Coordinate normalization must reject
    // unsupported forms; the control-flow model delegates bound checks here.
    if (!isVarying(loop.getLowerBound()) && !isVarying(loop.getUpperBound()) &&
        !isVarying(loop.getStep()))
      return WalkResult::advance();
    // Allow one program-axis loop directly at function top level, not inside
    // scf.if or another loop, and not alongside a second program-axis loop.
    // Retiling combines the program ID and one induction variable into a row
    // coordinate; multiple such loops require more than this mapping supports.
    if (outer || loop->getParentOp() != function) {
      first = snapshotBatchDiagnostic(loop, "loop_form",
                                      "UNPROVEN_LOOP_COALESCING");
      return WalkResult::interrupt();
    }
    programLoop = loop;
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return BatchResult<scf::ForOp>::decline(std::move(first));
  // A null handle is a valid straight-line program.
  return BatchResult<scf::ForOp>::success(programLoop);
}

bool batchLoopCarriesValues(scf::ForOp loop) {
  // Count iter_args without classifying reductions. Zero proves there are
  // no values carried across rows. Distinguishing reductions from passthrough
  // values would require following yields without admitting additional
  // currently supported cases.
  return !loop.getInitArgs().empty();
}

} // namespace mlir::triton::cfg
