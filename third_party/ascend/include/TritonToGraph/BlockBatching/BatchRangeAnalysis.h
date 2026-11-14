// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_RANGE_ANALYSIS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_RANGE_ANALYSIS_H
#include "TritonToGraph/BlockBatching/BatchExpr.h"
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::triton::cfg {

// Loop-level questions the gates ask. Free functions, not cached fields: a
// BatchLoopFact carries only the loop's three bound expressions, so there is
// no state here that can go stale.
std::optional<int64_t> batchLoopConstantStep(const BatchLoopFact &loop);
bool batchLoopHasProvenPositiveStep(const BatchLoopFact &loop);
bool batchLoopBoundsAreLaneUniform(const BatchLoopFact &loop,
                                   unsigned mergeAxis);
std::optional<int64_t> batchLoopConstantTripCount(const BatchLoopFact &loop);
// Unknown scalar inputs are checked by the existing per-launch guard. Known
// unsafe arithmetic must never reach the SCF unroller, even in a dead
// candidate.
enum class BatchLoopArithmetic { StaticSafe, NeedsRuntimeGuard, Unsafe };
BatchLoopArithmetic batchLoopIterationArithmetic(const BatchLoopFact &loop,
                                                 int64_t factor);
// Largest d dividing every possible trip count. 1 when nothing is proven;
// d % factor == 0 is what lets the rewrite skip the tail mask.
int64_t batchLoopTripCountDivisor(const BatchLoopFact &loop);

} // namespace mlir::triton::cfg
#endif
