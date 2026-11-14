// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchAffine.h"
#include "TritonToGraph/BlockBatching/BatchLoopLegality.h"
#include <cassert>
#include <limits>

using namespace mlir::triton::cfg;
using K = BatchExpr;
using Profile = BatchAffineForm::Profile;

int main() {
  auto value = [](int64_t n) { return K::leaf(K::Constant, n, 64); };
  auto arg = K::leaf(K::Argument, 1, 32);
  auto pid = K::leaf(K::Program, 0, 32);
  auto sub = K::node(K::Sub, 64, pid, pid);
  auto prove = [](const BatchExprRef &expr, Profile profile) {
    unsigned budget = 256;
    return BatchAffineForm::fromExpr(expr, budget,
                                     kBatchLoopLeaves |
                                         batchLeaf(K::ProgramQuotient) |
                                         batchLeaf(K::ProgramRemainder),
                                     profile, 0);
  };
  // At zero weight, integer loop proofs still observe the key. Parameter
  // layout proofs cancel it before multiplying sparse coordinate expressions.
  assert(prove(sub, Profile::Integer)->terms.size() == 1);
  assert(prove(sub, Profile::Parameter)->terms.empty());
  auto scaled = K::node(K::Mul, 64, pid, arg);
  auto symbolic = prove(scaled, Profile::Parameter);
  assert(symbolic && symbolic->symbolic() && symbolic->terms.size() == 1);
  assert(!prove(scaled, Profile::Integer));
  assert(!prove(K::node(K::Mul, 64, arg, arg), Profile::Parameter));
  assert(!prove(K::leaf(K::Argument, 1, 16), Profile::Parameter));
  assert(prove(K::leaf(K::Argument, 1, 16), Profile::Integer));
  auto quotient = K::leaf(K::ProgramQuotient, 1, 32, 4);
  assert(prove(quotient, Profile::Parameter)
             ->terms.count({K::ProgramQuotient, 1, 4}) == 1);
  // Integer callers specify the complete allowed-leaf set. Production loop
  // masks exclude divisions, while the generic decomposition retains its
  // contract.
  assert(prove(quotient, Profile::Integer)
             ->terms.count({K::ProgramQuotient, 1, 0}) == 1);
  assert(!prove(K::leaf(K::ProgramQuotient, 0, 32, 4), Profile::Parameter));
  assert(!prove(K::leaf(K::ProgramRemainder, 1, 32, 0), Profile::Parameter));
  for (auto profile : {Profile::Integer, Profile::Parameter}) {
    unsigned budget = 2;
    assert(
        !BatchAffineForm::fromExpr(sub, budget, kBatchLoopLeaves, profile, 0));
    assert(budget == 0);
    assert(!prove(K::node(K::Sub, 64, value(-1), value(INT64_MIN)), profile));
    assert(!prove(K::node(K::Mul, 64, value(INT64_MAX), value(2)), profile));
  }
  assert(!K::isLaneUniform(scaled, 0));
  assert(K::isLaneUniform(scaled, 1));
  assert(!K::constant(K::node(K::Add, 64, value(INT64_MAX), value(1))));
  assert(K::largestKnownDivisor(value(-24)) == 24);
  assert(K::rangeOf(K::leaf(K::Coordinate, 0, 32, 8)) ==
         std::make_optional(std::pair<int64_t, int64_t>{0, 7}));
  auto straight = isSafeToBatchLoop(nullptr, 0);
  assert(straight.hasValue() && !std::move(straight).takeValue());
  BatchLoopFact invalid;
  auto refusal = isSafeToBatchLoop(&invalid, 0);
  assert(!refusal.hasValue() &&
         refusal.diagnostic().reason == "UNPROVEN_LOOP_COALESCING");
}
