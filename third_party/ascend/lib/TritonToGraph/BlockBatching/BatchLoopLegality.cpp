// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopLegality.h"

namespace mlir::triton::cfg {

BatchResult<std::optional<BatchLogicalAxis>>
isSafeToBatchLoop(const BatchLoopFact *programLoop, int gridAxis) {
  // Straight-line kernels and block-local loops retain program-level grouping.
  using Result = BatchResult<std::optional<BatchLogicalAxis>>;
  if (!programLoop)
    return Result::success(std::nullopt);

  auto schedule = normalizeBatchLoopCoordinates(*programLoop, gridAxis);
  if (!schedule.hasValue())
    return Result::decline(schedule.diagnostic());

  // Each lane owns one original program and its complete row segment. The
  // normalized bounds preserve its recurrence order and output slots.
  return Result::success(std::move(schedule).takeValue());
}

} // namespace mlir::triton::cfg
