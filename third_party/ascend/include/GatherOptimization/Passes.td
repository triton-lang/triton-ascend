/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_GATHER_OPTIMIZATION_PASSES
#define TRITON_GATHER_OPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"

def GatherOptimization : Pass<"gather-optimization", "mlir::ModuleOp"> {
  let summary = "Optimization for gather-like workloads";
  let constructor = "triton::createGatherOptimizationPass()";
}

#endif // TRITON_GATHER_OPTIMIZATION_PASSES
