// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_FORM_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_FORM_H
#include "TritonToGraph/BlockBatching/BatchResult.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::cfg {

// Gate 0 belongs to the fact layer because it walks IR. The legality layer
// (BatchLoopLegality.h) consumes its result without depending on MLIR.
//
// Whether a value varies along the axis being merged. Injected rather than
// passed a full analysis workspace: that one bit is all the form check needs
// from the plan, and taking the plan would drag the Triton dialect into this
// header for nothing.
using BatchVaryingPredicate = llvm::function_ref<bool(Value)>;

// A loop is a program-axis loop when its lower bound varies along the batched
// axis (for r in range(pid, N, num_programs)). Every other loop is an
// in-program loop (for off in range(0, N, BLOCK)) and is left as is.
BatchResult<scf::ForOp> checkBatchLoopForm(Operation *function,
                                           BatchVaryingPredicate isVarying);

// Detect values carried between iterations for BatchLoopFact::carriesValues.
// Cross-row reductions, such as dW_row += ... followed by one final store,
// appear as iter_args in the IR. Legality checks consume this fact.
//
// Keep this helper separate from Triton-dependent fact collection so its
// tests need only the scf dialect, not the full Triton/GPU dependency chain.
bool batchLoopCarriesValues(scf::ForOp loop);

} // namespace mlir::triton::cfg
#endif
