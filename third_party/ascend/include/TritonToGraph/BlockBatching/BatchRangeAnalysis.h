// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_RANGE_ANALYSIS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_RANGE_ANALYSIS_H
#include "TritonToGraph/BlockBatching/BatchExpr.h"
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::triton::cfg {

// Narrow-scope value facts over BatchExpr DAGs. Deliberately not a dataflow
// lattice over MLIR values: it answers only the four questions the loop gates
// ask, on expressions the access analysis already built. Every query returns
// "unproven" rather than a guess, and unproven never means "unconstrained".
//
// The only context it needs is which program axis the merge happens on, so it
// stays free of any MLIR dependency: lane uniformity is a structural property
// of the expression (spec 2.0 -- a value differs across lanes exactly when it
// derives from the merged axis's program id), not a dataflow fact.
class BatchRangeFacts {
public:
  explicit BatchRangeFacts(unsigned mergeAxis) : mergeAxis(mergeAxis) {}

  std::optional<int64_t> constant(const BatchExprRef &e) const;
  std::optional<std::pair<int64_t, int64_t>> range(const BatchExprRef &e) const;
  // Largest d dividing every value of e. 1 when nothing is proven.
  int64_t largestKnownDivisor(const BatchExprRef &e) const;
  // True when e takes the same value in all F lanes of the merged program.
  bool isLaneUniform(const BatchExprRef &e) const;

private:
  unsigned mergeAxis;
};

// Loop-level questions the gates ask. Free functions, not cached fields: a
// BatchLoopFact carries only the loop's three bound expressions, so there is
// no state here that can go stale.
std::optional<int64_t> batchLoopConstantStep(const BatchLoopFact &loop,
                                             const BatchRangeFacts &facts);
bool batchLoopHasProvenPositiveStep(const BatchLoopFact &loop,
                                    const BatchRangeFacts &facts);
bool batchLoopBoundsAreLaneUniform(const BatchLoopFact &loop,
                                   const BatchRangeFacts &facts);
std::optional<int64_t> batchLoopConstantTripCount(const BatchLoopFact &loop,
                                                  const BatchRangeFacts &facts);
// Unknown scalar inputs are checked by the existing per-launch guard. Known
// unsafe arithmetic must never reach the SCF unroller, even in a dead
// candidate.
enum class BatchLoopArithmetic { StaticSafe, NeedsRuntimeGuard, Unsafe };
BatchLoopArithmetic batchLoopIterationArithmetic(const BatchLoopFact &loop,
                                                 const BatchRangeFacts &facts,
                                                 int64_t factor);
// Largest d dividing every possible trip count. 1 when nothing is proven;
// d % factor == 0 is what lets the rewrite skip the tail mask.
int64_t batchLoopTripCountDivisor(const BatchLoopFact &loop,
                                  const BatchRangeFacts &facts);

} // namespace mlir::triton::cfg
#endif
