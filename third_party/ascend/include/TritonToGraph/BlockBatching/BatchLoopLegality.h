// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_LEGALITY_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_LEGALITY_H
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"

namespace mlir::triton::cfg {

// Layer 2 contract, same as BatchLoopSchedule.h: a pure predicate over facts.
// No MLIR, no IR walking, no diagnostics. 关卡 0 的形态检查必须走 IR，所以它住
// 在 BatchLoopForm.h；这里只吃它的结论。
struct BatchLoopLegality {
  bool safe = false;
  std::string reason; // 空表示放行
  // 有值 当且仅当 safe 且函数里确实有一条程序轴循环。safe 而 axis 为空，说明
  // 这是直线 kernel 或只含块内循环，交回既有的程序级加宽路径，不重分块。
  std::optional<BatchLogicalAxis> axis;
};

// programLoop 为空表示关卡 0 判定"没有程序轴循环"，那是合法形态不是失败。
BatchLoopLegality isSafeToBatchLoop(const BatchLoopFact *programLoop,
                                    const BatchRangeFacts &ranges,
                                    int gridAxis);

} // namespace mlir::triton::cfg
#endif
