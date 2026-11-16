// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_PLAN_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_PLAN_H
#include "TritonToGraph/BlockBatching/BatchAnalyzer.h"
#include "TritonToGraph/BlockBatching/BatchLayoutFacts.h"
namespace mlir::triton::cfg {
SmallVector<BatchLayoutResult, 0>
initialBatchLayoutResults(ArrayRef<BatchLayoutPlan> plans);
std::string serializeBatchLayoutResults(ArrayRef<BatchLayoutPlan> plans,
                                        ArrayRef<BatchLayoutResult> results);

SmallVector<BatchLayoutPlan, 0>
planBatchLayouts(const BatchRewriteWorkspace &scope, int axis, int64_t factor);
std::string serializeBatchLayouts(ArrayRef<BatchLayoutPlan> plans);
} // namespace mlir::triton::cfg
#endif
