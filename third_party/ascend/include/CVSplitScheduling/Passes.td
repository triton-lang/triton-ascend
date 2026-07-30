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
           "loop unroll factor (2, 4, or 8)">
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
