// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchRangeAnalysis.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace mlir::triton::cfg {
namespace {

using K = BatchExpr;
using Interval = std::pair<int64_t, int64_t>;

// i32 is the widest a program id or a grid extent can be. Triton's launcher
// rejects anything larger, and the runtime guard checks it again per launch.
constexpr int64_t kGridMax = 2147483647;

bool isBoolean(K::Kind kind) {
  switch (kind) {
  case K::True:
  case K::False:
  case K::And:
  case K::SLT:
  case K::ULT:
  case K::EQ:
    return true;
  default:
    return false;
  }
}

// Reinterpret the low `sourceBits` of a sign-stored value as unsigned. Done in
// unsigned arithmetic throughout: 1 << 63 and INT64_MIN + x are both undefined.
std::optional<int64_t> zeroExtend(int64_t value, unsigned sourceBits) {
  if (value >= 0)
    return value;
  if (!sourceBits || sourceBits >= 64)
    return std::nullopt;
  // sourceBits is 1..63 here, so the mask is at most 2^63 - 1 == INT64_MAX and
  // the result always fits.
  return static_cast<int64_t>(static_cast<uint64_t>(value) &
                              ((uint64_t(1) << sourceBits) - 1));
}

// Does [lo, hi] survive a truncation to `bits` unchanged?
bool fitsInBits(const Interval &span, unsigned bits) {
  if (!bits || bits >= 64)
    return bits >= 64;
  const int64_t limit = int64_t(1) << (bits - 1);
  return span.first >= -limit && span.second <= limit - 1;
}

} // namespace

std::optional<int64_t> BatchRangeFacts::constant(const BatchExprRef &e) const {
  if (!e || isBoolean(e->kind))
    return std::nullopt;
  switch (e->kind) {
  case K::Constant:
    return e->value;
  case K::Add:
  case K::Sub:
  case K::Mul:
  case K::MinSI:
  case K::MaxSI: {
    auto lhs = constant(e->lhs);
    auto rhs = constant(e->rhs);
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t result;
    switch (e->kind) {
    case K::Add:
      return llvm::AddOverflow(*lhs, *rhs, result) ? std::nullopt
                                                   : std::optional(result);
    case K::Sub:
      return llvm::SubOverflow(*lhs, *rhs, result) ? std::nullopt
                                                   : std::optional(result);
    case K::Mul:
      return llvm::MulOverflow(*lhs, *rhs, result) ? std::nullopt
                                                   : std::optional(result);
    case K::MinSI:
      return std::min(*lhs, *rhs);
    default:
      return std::max(*lhs, *rhs);
    }
  }
  case K::ExtSI:
    // The value is already stored sign-extended, so widening is a no-op.
    return constant(e->lhs);
  case K::ExtUI:
    if (auto sub = constant(e->lhs))
      return zeroExtend(*sub, static_cast<unsigned>(e->extent));
    return std::nullopt;
  case K::Trunc:
    if (auto sub = constant(e->lhs)) {
      if (!e->bits || e->bits >= 64)
        return sub;
      return llvm::SignExtend64(static_cast<uint64_t>(*sub), e->bits);
    }
    return std::nullopt;
  default:
    // Argument, Program, Grid, Induction, Coordinate, Unknown.
    return std::nullopt;
  }
}

std::optional<Interval> BatchRangeFacts::range(const BatchExprRef &e) const {
  if (!e || isBoolean(e->kind))
    return std::nullopt;
  switch (e->kind) {
  case K::Constant:
    return Interval(e->value, e->value);
  case K::Program:
    return Interval(0, kGridMax);
  case K::Grid:
    // num_programs >= 1. This is what proves a program-axis loop's step
    // positive: the step is num_programs * c, never constant but always > 0.
    return Interval(1, kGridMax);
  case K::Coordinate:
    if (e->extent <= 0)
      return std::nullopt;
    return Interval(0, e->extent - 1);
  case K::Add:
  case K::Sub:
  case K::Mul:
  case K::MinSI:
  case K::MaxSI: {
    auto lhs = range(e->lhs);
    auto rhs = range(e->rhs);
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t lo, hi;
    switch (e->kind) {
    case K::Add:
      if (llvm::AddOverflow(lhs->first, rhs->first, lo) ||
          llvm::AddOverflow(lhs->second, rhs->second, hi))
        return std::nullopt;
      return Interval(lo, hi);
    case K::Sub:
      if (llvm::SubOverflow(lhs->first, rhs->second, lo) ||
          llvm::SubOverflow(lhs->second, rhs->first, hi))
        return std::nullopt;
      return Interval(lo, hi);
    case K::Mul: {
      // A negative factor swaps the ends, so all four corners matter.
      int64_t corners[4];
      const int64_t ends[2][2] = {{lhs->first, lhs->second},
                                  {rhs->first, rhs->second}};
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          if (llvm::MulOverflow(ends[0][i], ends[1][j], corners[i * 2 + j]))
            return std::nullopt;
      return Interval(*std::min_element(corners, corners + 4),
                      *std::max_element(corners, corners + 4));
    }
    case K::MinSI:
      return Interval(std::min(lhs->first, rhs->first),
                      std::min(lhs->second, rhs->second));
    default:
      return Interval(std::max(lhs->first, rhs->first),
                      std::max(lhs->second, rhs->second));
    }
  }
  case K::ExtSI:
    return range(e->lhs);
  case K::ExtUI: {
    auto sub = range(e->lhs);
    if (!sub)
      return std::nullopt;
    if (sub->first >= 0)
      return sub;
    // Possibly negative: all we can say is the source width's unsigned span.
    const unsigned sourceBits = static_cast<unsigned>(e->extent);
    if (!sourceBits || sourceBits >= 64)
      return std::nullopt;
    return Interval(0, static_cast<int64_t>((uint64_t(1) << sourceBits) - 1));
  }
  case K::Trunc: {
    auto sub = range(e->lhs);
    if (!sub || !fitsInBits(*sub, e->bits))
      return std::nullopt;
    return sub;
  }
  default:
    // Argument, Induction, Unknown.
    return std::nullopt;
  }
}

int64_t BatchRangeFacts::largestKnownDivisor(const BatchExprRef &e) const {
  if (!e)
    return 1;
  // Mirrors MLIR's AffineExpr::getLargestKnownDivisor: a constant contributes
  // its absolute value, so 0 contributes 0 -- the sentinel for "divisible by
  // everything", which is exactly what std::gcd(0, n) == n needs.
  if (auto value = constant(e)) {
    // |INT64_MIN| is not representable, and std::abs on it is undefined. A
    // negative divisor leaking out would make every trip-count quotient
    // negative, so refuse to name it instead.
    if (*value == std::numeric_limits<int64_t>::min())
      return 1;
    return std::abs(*value);
  }
  switch (e->kind) {
  case K::Mul: {
    const int64_t lhs = largestKnownDivisor(e->lhs);
    const int64_t rhs = largestKnownDivisor(e->rhs);
    int64_t product;
    // Overflow means we cannot name the product; fall back to the larger
    // factor, which still divides.
    return llvm::MulOverflow(lhs, rhs, product) ? std::max(lhs, rhs) : product;
  }
  case K::Add:
  case K::Sub:
  // min and max of two multiples of d are multiples of d. Needed because a
  // blocked loop's upper bound is written min((p + 1) * R, N).
  case K::MinSI:
  case K::MaxSI:
    return std::gcd(largestKnownDivisor(e->lhs), largestKnownDivisor(e->rhs));
  case K::ExtSI:
    return largestKnownDivisor(e->lhs);
  default:
    return 1;
  }
}

bool BatchRangeFacts::isLaneUniform(const BatchExprRef &e) const {
  if (!e)
    return false;
  switch (e->kind) {
  case K::Program:
    // Spec 2.0: lanes differ only in their coordinate along the merged axis.
    return static_cast<unsigned>(e->value) != mergeAxis;
  case K::Constant:
  case K::Argument:
  case K::Grid:
    return true;
  case K::Induction:
    // After unroll-and-jam the F lanes share one loop, so the induction
    // variable holds a single value.
    return true;
  case K::Coordinate:
  case K::Unknown:
    // A coordinate is an in-block element index, not a scalar; asking whether
    // it is "the same in every lane" is the wrong question, and it has no
    // business appearing in a scalar loop bound. Answering no rejects, which
    // is the safe direction.
    return false;
  default:
    break;
  }
  if (!e->lhs)
    return false;
  if (!isLaneUniform(e->lhs))
    return false;
  return !e->rhs || isLaneUniform(e->rhs);
}

std::optional<int64_t> batchLoopConstantStep(const BatchLoopFact &loop,
                                             const BatchRangeFacts &facts) {
  return facts.constant(loop.step);
}

bool batchLoopHasProvenPositiveStep(const BatchLoopFact &loop,
                                    const BatchRangeFacts &facts) {
  auto stepRange = facts.range(loop.step);
  return stepRange && stepRange->first > 0;
}

bool batchLoopBoundsAreLaneUniform(const BatchLoopFact &loop,
                                   const BatchRangeFacts &facts) {
  return facts.isLaneUniform(loop.lower) && facts.isLaneUniform(loop.upper) &&
         facts.isLaneUniform(loop.step);
}

std::optional<int64_t>
batchLoopConstantTripCount(const BatchLoopFact &loop,
                           const BatchRangeFacts &facts) {
  auto step = batchLoopConstantStep(loop, facts);
  if (!step || *step <= 0)
    return std::nullopt;
  auto lower = facts.constant(loop.lower);
  auto upper = facts.constant(loop.upper);
  if (!lower || !upper)
    return std::nullopt;
  if (*upper <= *lower)
    return 0;
  // Two signed i64 bounds can span UINT64_MAX. Keep both the subtraction and
  // ceil-div numerator wide; widening an already evaluated i64 sum is too late.
  const llvm::APInt lo(128, *lower, true), hi(128, *upper, true);
  const llvm::APInt stride(128, *step, true);
  const llvm::APInt trips = (hi - lo + stride - 1).udiv(stride);
  if (!trips.isSignedIntN(64))
    return std::nullopt;
  return trips.getSExtValue();
}

BatchLoopArithmetic batchLoopIterationArithmetic(const BatchLoopFact &loop,
                                                 const BatchRangeFacts &facts,
                                                 int64_t factor) {
  using Result = BatchLoopArithmetic;
  if (factor <= 0 || !loop.bits || loop.bits > 64)
    return Result::Unsafe;
  auto wide = [](int64_t value) { return llvm::APInt(128, value, true); };
  auto fits = [&](const llvm::APInt &value) {
    return value.isSignedIntN(loop.bits);
  };
  auto lower = facts.constant(loop.lower);
  auto upper = facts.constant(loop.upper);
  auto step = facts.constant(loop.step);
  if (!fits(wide(factor)) || (lower && !fits(wide(*lower))) ||
      (upper && !fits(wide(*upper))) ||
      (step &&
       (*step <= 0 || !fits(wide(*step)) || !fits(wide(*step) * wide(factor)))))
    return Result::Unsafe;

  // These checks also apply when another bound is dynamic. The guard cannot
  // repair a statically overflowing step, difference or endpoint.
  if (lower && upper && *upper > *lower && !fits(wide(*upper) - wide(*lower)))
    return Result::Unsafe;
  if (upper && step && !fits(wide(*upper) + wide(*step) * wide(factor)))
    return Result::Unsafe;
  if (!lower || !upper || !step)
    return Result::NeedsRuntimeGuard;

  auto trips = batchLoopConstantTripCount(loop, facts);
  if (!trips)
    return Result::Unsafe;
  if (*trips == 0)
    return Result::StaticSafe;
  const llvm::APInt stride = wide(*step);
  const llvm::APInt span = wide(*upper) - wide(*lower);
  // SCF may use a dynamic ceil-div if constants are not materialized as
  // ConstantOps. Its static path also forms groupedTrips * step in int64_t
  // before adding lower. Check the intermediates, not only the final bound.
  const llvm::APInt groupedTrips = wide(*trips - *trips % factor);
  const llvm::APInt offset = groupedTrips * stride;
  if (!fits(span + stride - 1) || !fits(offset) || !fits(wide(*lower) + offset))
    return Result::Unsafe;
  return Result::StaticSafe;
}

int64_t batchLoopTripCountDivisor(const BatchLoopFact &loop,
                                  const BatchRangeFacts &facts) {
  auto step = batchLoopConstantStep(loop, facts);
  if (!step || *step <= 0)
    return 1;
  // trip = ceildiv(upper - lower, step). When step divides a factor common to
  // both ends, every trip count is a multiple of that factor over step.
  const int64_t spanDivisor = std::gcd(facts.largestKnownDivisor(loop.lower),
                                       facts.largestKnownDivisor(loop.upper));
  if (spanDivisor <= 0 || spanDivisor % *step != 0)
    return 1;
  return std::max<int64_t>(1, spanDivisor / *step);
}

} // namespace mlir::triton::cfg
