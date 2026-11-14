// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_FACTS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_FACTS_H
#include "TritonToGraph/BlockBatching/BatchExpr.h"

namespace mlir::triton::cfg {

// One enclosing scf::ForOp of an access, innermost first. A pure value type:
// the owned BatchExpr DAGs outlive the original body, nothing else does.
//
// Only what the loop literally is. Everything derivable from these three
// expressions -- constant step, trip count, divisibility, lane uniformity --
// is answered on demand on the BatchExpr DAG, so there is no field here that
// can be stale or unfilled.
struct BatchLoopFact {
  unsigned id = 0;
  unsigned bits = 0;
  BatchExprRef lower, upper, step;
  // Whether the loop carries any value from one iteration to the next, i.e.
  // !scf::ForOp::getInitArgs().empty(). This is the one thing about the loop
  // that the three expressions above cannot answer, and the gates do ask it:
  // a cross-row accumulation (dW_row += ...) is exactly an iter_arg, and
  // re-grouping rows across lanes would re-associate it.
  //
  // Counting iter_args rather than recognising accumulations on purpose: zero
  // means "provably no cross-row carry", which is all a gate needs to let a
  // loop through. Telling a real accumulation from a pass-through would need
  // the yield chain, and today it would not admit one extra operator.
  //
  // Defaults to the restrictive answer so a producer that forgets to fill it
  // makes gates reject rather than silently allow.
  bool carriesValues = true;
};

// 序列化不在这里。它是输出格式，不是数据载体，见 BatchFactsJSON.h。

} // namespace mlir::triton::cfg
#endif
