// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Standalone regression: link the production helpers and run under UBSan.
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace mlir::triton::cfg;

static void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

static BatchLoopFact loop(int64_t lower, int64_t upper, int64_t step,
                          unsigned bits = 64) {
  BatchLoopFact fact;
  fact.bits = bits;
  fact.lower = BatchExpr::leaf(BatchExpr::Constant, lower, bits);
  fact.upper = BatchExpr::leaf(BatchExpr::Constant, upper, bits);
  fact.step = BatchExpr::leaf(BatchExpr::Constant, step, bits);
  return fact;
}

int main() {
  BatchRangeFacts facts(0);
  check(batchLoopConstantTripCount(
            loop(0, 8000000000000000000LL, 4000000000000000000LL), facts) == 2,
        "ceildiv intermediate overflow must not change trip count 2");
  constexpr int64_t min = std::numeric_limits<int64_t>::min();
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  check(batchLoopConstantTripCount(loop(-8, 8, 3), facts) == 6,
        "negative lower bound and zero crossing");
  check(batchLoopConstantTripCount(loop(min, max, max), facts) == 3,
        "full i64 span must be computed before narrowing");
  check(!batchLoopConstantTripCount(loop(min, max, 1), facts),
        "unrepresentable trip count must be unproven");
  for (auto bounds : {loop(8, 8, 1), loop(9, 8, 1)})
    check(batchLoopConstantTripCount(bounds, facts) == 0, "empty loop");
  for (auto step : {0, -1})
    check(!batchLoopConstantTripCount(loop(0, 8, step), facts),
          "nonpositive step must not prove a trip count");

  BatchLogicalAxis axis{BatchLogicalAxis::Blocked, 0, 0, 1};
  using Arithmetic = BatchLoopArithmetic;
  using Lowering = BatchLoopLowering;
  for (int64_t upper : {0, 1, 2, 3, 4, 5}) {
    auto fact = loop(0, upper, 1, 32);
    check(batchLoopConstantTripCount(fact, facts) == upper,
          "empty, singleton, complete groups and tail counts");
    check(selectBatchLoopLowering(&fact, &axis, facts, 2) ==
              (upper < 2 ? Lowering::ProgramLanes
                         : Lowering::SequentialIterations),
          "short loops retain program lanes; groups and tails are admitted");
  }
  for (auto fact : {loop(0, 8000000000000000000LL, 4000000000000000000LL),
                    loop(min, max, 1), loop(0, max, max),
                    loop(0, 2147483647, 1073741824, 32), loop(0, 1, 0),
                    loop(0, 1, -1), loop(max - 3, max, 1)}) {
    check(batchLoopIterationArithmetic(fact, facts, 2) == Arithmetic::Unsafe,
          "known overflowing or invalid unroll arithmetic must be unsafe");
    check(selectBatchLoopLowering(&fact, &axis, facts, 2) ==
              Lowering::ProgramLanes,
          "nullopt on static unsafe bounds must not select iteration groups");
  }
  auto dynamic = loop(0, 10, 1, 32);
  dynamic.upper = BatchExpr::leaf(BatchExpr::Argument, 0, 32);
  check(!batchLoopConstantTripCount(dynamic, facts), "dynamic trip count");
  check(batchLoopIterationArithmetic(dynamic, facts, 2) ==
            Arithmetic::NeedsRuntimeGuard,
        "dynamic bounds retain per-launch guards");
  check(selectBatchLoopLowering(&dynamic, &axis, facts, 2) ==
            Lowering::SequentialIterations,
        "dynamic bounds must not be confused with static overflow");
  dynamic.step = BatchExpr::leaf(BatchExpr::Constant, 1073741824, 32);
  check(batchLoopIterationArithmetic(dynamic, facts, 2) == Arithmetic::Unsafe,
        "known step overflow is unsafe even with a dynamic upper bound");
  check(selectBatchLoopLowering(&dynamic, &axis, facts, 2) ==
            Lowering::ProgramLanes,
        "partially dynamic bounds must not hide unsafe static arithmetic");
  for (int64_t factor : {0, -1})
    check(batchLoopIterationArithmetic(loop(0, 8, 1), facts, factor) ==
              Arithmetic::Unsafe,
          "invalid factors must not reach the unroller");
  check(batchLoopIterationArithmetic(loop(-8, 8, 3), facts, 2) ==
            Arithmetic::StaticSafe,
        "representable cross-zero arithmetic remains computable");
  std::cout << "loop arithmetic and schedule regressions passed\n";
  return 0;
}
