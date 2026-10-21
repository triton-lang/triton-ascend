/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ASCEND_STORE_OWNERSHIP_ANALYSIS_H
#define TRITON_ASCEND_STORE_OWNERSHIP_ANALYSIS_H

#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

// Failure means that the access pattern is unsupported, rather than disjoint.
// The returned predicate is uniform across the logical launch grid.
FailureOr<Value> buildStoreOwnershipGuard(StoreOp store,
                                          PatternRewriter &rewriter);

} // namespace mlir::triton

#endif
