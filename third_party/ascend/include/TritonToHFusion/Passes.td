/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_TO_HFUSION_CONVERSION_PASSES
#define TRITON_TO_HFUSION_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToHFusion : Pass<"triton-to-hfusion", "mlir::ModuleOp"> {
  let summary = "Convert Triton to HFusion dialect";
  let constructor = "triton::createTritonToHFusionPass()";
  let dependentDialects = ["hfusion::HFusionDialect"];
  let options = [
    Option<"compileOn91095", "compile-on-910-95", "bool", "false",
           "Compile for register-based Ascend targets">
  ];
}

#endif // TRITON_TO_HFUSION_CONVERSION_PASSES
