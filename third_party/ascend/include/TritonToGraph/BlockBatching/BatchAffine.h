// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_AFFINE_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_AFFINE_H
#include "TritonToGraph/BlockBatching/BatchExpr.h"
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
namespace mlir::triton::cfg {
// Deliberately bounded exact integer algebra: constant + sum(coefficient*leaf).
// Every coefficient is proven by exact integer arithmetic or the whole
// decomposition fails. No sampled value and no equality of byte strides can
// establish a term here.
struct BatchLinearForm {
  using Key = std::pair<BatchExpr::Kind, int64_t>;
  int64_t constant = 0;
  std::map<Key, int64_t> terms;
};
// Which leaf kinds count as free variables. Every other leaf fails the
// decomposition, so widening this set can only turn failures into successes,
// never the reverse. That asymmetry is why each caller names its own set rather
// than sharing one default: a shared default would silently let the loop layer
// change what the layout layer proves.
using BatchLeafMask = uint32_t;
constexpr BatchLeafMask batchLeaf(BatchExpr::Kind kind) {
  return BatchLeafMask(1) << static_cast<unsigned>(kind);
}
// The set the layout analysis has used since before this header existed.
inline constexpr BatchLeafMask kBatchLayoutLeaves =
    batchLeaf(BatchExpr::Coordinate) | batchLeaf(BatchExpr::Program) |
    batchLeaf(BatchExpr::Argument) | batchLeaf(BatchExpr::Grid);
// The loop judgment layer additionally resolves induction variables, which is
// the whole point of normalizing (program id, induction variable) into one
// logical axis.
inline constexpr BatchLeafMask kBatchLoopLeaves =
    kBatchLayoutLeaves | batchLeaf(BatchExpr::Induction);
// Returns nullopt when the expression is not affine over `leaves`, when any
// coefficient overflows int64, or when `budget` runs out. `budget` is consumed
// across the whole traversal and is deliberately by reference, so a caller
// cannot be amplified by a deeply shared subexpression.
std::optional<BatchLinearForm>
batchLinearForm(const BatchExprRef &e, unsigned &budget, BatchLeafMask leaves);

// Structural equality: same kind, same payload, same children, recursively.
// Deliberately no canonicalization -- it does not see that 2*R and R*2 are the
// same value, and must not be relied on to. Its one caller asks whether two
// loop bounds were built from the same SSA value (a segment's start and end
// both scale by rows_per_program), and in that case the two DAGs are literally
// the same shape. A false negative there costs a rejection, which is the safe
// direction; a canonicalizing version would need its own proof of soundness.
bool batchExprEquivalent(const BatchExprRef &a, const BatchExprRef &b);
} // namespace mlir::triton::cfg
#endif
