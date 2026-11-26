// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_AFFINE_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_AFFINE_H
#include "TritonToGraph/BlockBatching/BatchExpr.h"
#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
namespace mlir::triton::cfg {
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
// Parameters retain their bit width. Coordinate keys retain the quotient or
// remainder divisor.
struct BatchAffineForm {
  enum class Profile { Integer, Parameter };
  using Key = std::tuple<BatchExpr::Kind, int64_t, int64_t>;
  struct Coefficient {
    using Atom = std::tuple<BatchExpr::Kind, int64_t, unsigned>;
    int64_t constant = 0;
    std::map<Atom, int64_t> parameters;
    bool isZero() const { return constant == 0 && parameters.empty(); }
    bool operator==(const Coefficient &other) const {
      return constant == other.constant && parameters == other.parameters;
    }
    bool addScaled(const Coefficient &other, int64_t scale);
    std::optional<Coefficient> multiply(const Coefficient &other) const;
    std::optional<Coefficient> divide(int64_t divisor) const;
    BatchExprRef expression() const;
  };
  Coefficient offset;
  std::map<Key, Coefficient> terms;
  bool symbolic() const;
  static std::optional<BatchAffineForm> fromExpr(const BatchExprRef &e,
                                                 unsigned &budget,
                                                 BatchLeafMask leaves,
                                                 Profile profile, int axis);
};

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
