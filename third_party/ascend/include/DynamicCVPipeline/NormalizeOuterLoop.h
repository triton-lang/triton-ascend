/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ASCEND_NORMALIZE_OUTER_LOOP_H
#define TRITON_ASCEND_NORMALIZE_OUTER_LOOP_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton {

// Expose bounded inner iterations before planning compute blocks.
std::unique_ptr<OperationPass<ModuleOp>>
createNormalizeOuterLoopPass(bool discoverMainLoop = false);

} // namespace mlir::triton

#endif // TRITON_ASCEND_NORMALIZE_OUTER_LOOP_H
