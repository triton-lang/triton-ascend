// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_FORM_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_FORM_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <string>

namespace mlir::triton::cfg {

// 关卡 0 住在事实层而不是判定层，因为它必须走 IR。判定层（BatchLoopLegality.h）
// 只吃这里的结论，所以能保持零 MLIR 依赖。
//
// Whether a value varies along the axis being merged. Injected rather than
// taken as a BatchControlFlowPlan: that one bit is all the form check needs
// from the plan, and taking the plan would drag the Triton dialect into this
// header for nothing.
using BatchVaryingPredicate = llvm::function_ref<bool(Value)>;

struct BatchLoopForm {
  bool ok = false;
  std::string reason;
  Operation *operation = nullptr;
  // The single loop whose lower bound varies along the batched program axis.
  // Null is a valid form, not a failure: the function has no program-axis loop,
  // and the existing program-level widening applies to it unchanged.
  scf::ForOp programLoop;
};

// A loop is a program-axis loop when its lower bound varies along the batched
// axis (for r in range(pid, N, num_programs)). Every other loop is an
// in-program loop (for off in range(0, N, BLOCK)) and is left as is.
BatchLoopForm checkBatchLoopForm(Operation *function,
                                 BatchVaryingPredicate isVarying);

// 循环会不会把值从这一圈带到下一圈，用来填 BatchLoopFact::carriesValues。
// 跨行归约（dW_row += …，循环结束才写一次）在 IR 里的对应物就是 iter_args，
// 判定层的关卡 1.5a 查的是这一位。
//
// 放在这里而不是直接写在采事实的地方：那边链着 Triton dialect，给它写单测会
// 把整条 GPU 依赖链拖进测试目标（C1/C2）。这里只认 scf，测得起。
bool batchLoopCarriesValues(scf::ForOp loop);

} // namespace mlir::triton::cfg
#endif
