// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_CV_SPATIAL_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_CV_SPATIAL_H
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <string>

namespace mlir::triton::cfg {
// Called only inside the transactional batching sandbox, after the original
// memory-domain proof. Failure may have changed the sandbox; never commit it.
LogicalResult rewriteSpatialCV(ModuleOp module, triton::FuncOp function,
                               int axis, int64_t factor, std::string &lowering,
                               std::string &reason);
} // namespace mlir::triton::cfg
#endif
