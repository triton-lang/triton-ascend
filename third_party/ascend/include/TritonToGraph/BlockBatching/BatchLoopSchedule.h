// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_SCHEDULE_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_SCHEDULE_H
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include "TritonToGraph/BlockBatching/BatchRangeAnalysis.h"
#include "TritonToGraph/BlockBatching/BatchResult.h"
#include <cstdint>
#include <optional>
#include <string>
namespace mlir::triton::cfg {
// Layer 2 contract: every entry point in this header is a pure predicate.
// It takes facts and returns a verdict. It must not take a ModuleOp, must not
// create or erase any operation, and must be callable before any rewrite has
// happened. Mirrors llvm::isSafeToUnrollAndJam.

// Original program coordinates, independent of the selected lowering.
// Cyclic: lower = p*R, step = G*R. Program-lane batching retains each
// original program's rows and stride under a folded launch grid.
// Blocked: constant positive step and a proved uniform or per-program segment.
// SequentialIterations uses these same facts but retains the original grid,
// tile width, recurrence ownership and order within each program.
struct BatchLogicalAxis {
  enum Distribution : uint8_t { Cyclic, Blocked };
  Distribution distribution = Cyclic;
  unsigned loopID = 0;
  int gridAxis = -1;
  // Rows covered by one iteration: R for cyclic distribution, always 1 for
  // blocked distribution, which advances by its step rather than chunk width.
  int64_t chunk = 1;
};

// The existing coordinate proof also identifies program-owned recurrences.
// Group their consecutive iterations without moving rows between programs or
// reassociating carried values. The runtime proof checks the new arithmetic.
enum class BatchSchedule { ProgramLanes, SequentialIterations };
BatchSchedule selectBatchLoopLowering(const BatchLoopFact *loop,
                                      const BatchLogicalAxis *axis,
                                      int64_t factor);

// Normalize one loop, not a set of accesses: coordinates are a loop property.
// The form check used by BatchLoopLegality enforces a single program-axis loop
// at function top level; that check does not belong here.
//
// No factor argument is needed. Factor-dependent checks such as T >= F and
// F <= R belong to the main legality predicate.
BatchResult<BatchLogicalAxis>
normalizeBatchLoopCoordinates(const BatchLoopFact &loop, int gridAxis);

// Replace this loop's induction variable in an access expression with its
// lower bound plus an iteration offset:
//
//     Induction(id) = loop.lower + Induction(id)'
//
// The primed term denotes 0, step, 2*step, ... and is uniform across lanes.
// Lane variation resides in lower: both cyclic and blocked segments start at
// p*R and use a uniform step. Substitution exposes the program ID in the
// address while keeping the iteration offset uniform.
//
// Layout analysis looks for K::Program in byteOffset. Without substitution,
// the program ID remains hidden in the loop's lower bound, so a row access
// would incorrectly appear to be SHARED_ACCESS.
//
// Return null if substitution cannot be proved, including a missing lower
// bound. Callers must not fall back to the original expression. Expressions
// independent of this induction variable retain their original pointer.
BatchExprRef normalizeBatchAccessOnLogicalAxis(const BatchExprRef &e,
                                               const BatchLoopFact &loop);
} // namespace mlir::triton::cfg
#endif
