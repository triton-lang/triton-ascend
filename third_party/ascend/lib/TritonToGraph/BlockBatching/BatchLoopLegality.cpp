// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopLegality.h"

namespace mlir::triton::cfg {

BatchLoopLegality isSafeToBatchLoop(const BatchLoopFact *programLoop,
                                    const BatchRangeFacts &ranges,
                                    int gridAxis) {
  // Straight-line kernels and block-local loops retain program-level grouping.
  // There is no program-axis loop to normalize in that case.
  if (!programLoop)
    return {true, "", std::nullopt};

  auto schedule = normalizeBatchLoopCoordinates(*programLoop, ranges, gridAxis);
  if (!schedule.axis)
    return {false, schedule.reason, std::nullopt};

  // Each lane owns one original program and its complete row segment. The
  // normalized segment bounds preserve its recurrence order and output slots;
  // grouping must not repartition that segment across lanes.
  return {true, "", schedule.axis};
}

} // namespace mlir::triton::cfg
