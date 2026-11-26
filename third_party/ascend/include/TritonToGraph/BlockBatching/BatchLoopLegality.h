// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_LEGALITY_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_LEGALITY_H
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"

namespace mlir::triton::cfg {

// Layer 2 contract, same as BatchLoopSchedule.h: a pure predicate over facts.
// No MLIR, IR walking or diagnostics. Gate 0 walks IR in BatchLoopForm.h;
// this layer consumes its result. A null programLoop means there is no
// program-axis loop, which is a valid form rather than a failure.
BatchResult<std::optional<BatchLogicalAxis>>
isSafeToBatchLoop(const BatchLoopFact *programLoop, int gridAxis);

} // namespace mlir::triton::cfg
#endif
