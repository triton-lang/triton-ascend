// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_LEGALITY_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_LEGALITY_H
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"

namespace mlir::triton::cfg {

// Layer 2 contract, same as BatchLoopSchedule.h: a pure predicate over facts.
// No MLIR, no IR walking, no diagnostics. 关卡 0 的形态检查必须走 IR，所以它住
// 在 BatchLoopForm.h；这里只吃它的结论。
// programLoop 为空表示关卡 0 判定"没有程序轴循环"，那是合法形态不是失败。
BatchResult<std::optional<BatchLogicalAxis>>
isSafeToBatchLoop(const BatchLoopFact *programLoop, int gridAxis);

} // namespace mlir::triton::cfg
#endif
