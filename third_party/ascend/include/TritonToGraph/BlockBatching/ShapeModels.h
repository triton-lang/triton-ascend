// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_SHAPE_MODELS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_SHAPE_MODELS_H
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
namespace mlir::triton::cfg {
// All input operands are already mapped to their batch representation. The
// caller handles U/V broadcasting for the two operands of join.
Operation *rewriteBatchShape(Operation *op, TypeRange results,
                             IRMapping &operands, IRRewriter &rewriter);
} // namespace mlir::triton::cfg
#endif
