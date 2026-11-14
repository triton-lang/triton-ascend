// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_CONTROL_FLOW_MODELS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_CONTROL_FLOW_MODELS_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <string>
namespace mlir::triton::cfg {
// A true dot-accumulator/pointer recurrence keeps its original scf.for
// zero-trip semantics. Retain the consumer workaround for invariant or
// unclassified yielded tensors rather than wrapping every CV loop in an if.
bool cvLoopNeedsZeroTripGuard(scf::ForOp loop);
} // namespace mlir::triton::cfg
#endif
