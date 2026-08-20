/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_CV_SPLIT_SCHEDULING_PASSES
#define TRITON_CV_SPLIT_SCHEDULING_PASSES

include "mlir/Pass/PassBase.td"

def CVSplitScheduling : Pass<"cv_split_scheduling", "mlir::ModuleOp"> {
  let summary = "Unroll innermost loop and split into concurrent Cube/Vector scopes";
  let constructor = "triton::createCVSplitSchedulingPass()";
  let options = [
    Option<"compileOn91095", "compile-on-910-95",
           "bool", /*default*/"false",
           "compile on 910_95">,
    Option<"unrollFactor", "unroll-factor",
           "int", /*default*/"2",
           "loop unroll factor (2, 4, or 8)">,
    Option<"promoteFullyUnrolled", "promote-fully-unrolled",
           "bool", /*default*/"true",
           "promote the scope loops left by a candidate whose unroll consumed "
           "its whole trip count; disable to keep them for inspection">,
    Option<"pipelineDistance", "pipeline-distance",
           "int", /*default*/"0",
           "how many transfer boundaries apart to software-pipeline a "
           "boundary's consuming work from the boundary that reuses its slot. "
           "Zero follows the inter-core buffer depth, which is what the shared "
           "slots need. A distance at or above the unroll factor emits every "
           "boundary's own work first and drains the consumers after, which is "
           "right when no slot is reused and the interleaving only shortens "
           "the producing core's run-ahead.">,
    Option<"privateBufferUbBudgetBytes", "private-buffer-ub-budget-bytes",
           "int64_t", /*default*/"-1",
           "spare UB, in bytes, that cross-scope pools may spend to stop "
           "reusing a slot between lanes. Negative means the pass does not "
           "know the headroom and spends what the schedule asks for; the "
           "backend's memory planner is what actually decides whether it "
           "fits, and it reports an overflow with the exact requirement when "
           "it does not. Zero declines any spend, which keeps the rotating "
           "pools and the interleaved schedule that orders them.">,
    Option<"promotePrivateBufferPools", "promote-private-buffer-pools",
           "bool", /*default*/"false",
           "additionally give individual pools one buffer per lane instead of "
           "a depth-sized rotating set, cheapest-first against the same UB "
           "budget. Off by default: at the unroll factors in use the reuse it "
           "removes is already ordered by a flag the schedule needs anyway, so "
           "it costs buffers and flags without removing a stall. Turn it on to "
           "stop depending on that ordering being emergent.">
  ];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::math::MathDialect",
    // The pass creates ops/attrs from these BiShengIR dialects (scopes,
    // tcore_type attrs, fixpipe/copy/sync transfers). They must be declared
    // so the dialect storage uniquers are initialized even when the input IR
    // contains no such ops yet (e.g. when run standalone via triton-opt).
    "mlir::hivm::HIVMDialect",
    "mlir::scope::ScopeDialect",
    "mlir::annotation::AnnotationDialect",
    "mlir::hacc::HACCDialect"
  ];
}

#endif // TRITON_CV_SPLIT_SCHEDULING_PASSES
