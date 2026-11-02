// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef TRITON_ASCEND_PROVEN_OVERFLOW_ASSERT_H
#define TRITON_ASCEND_PROVEN_OVERFLOW_ASSERT_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton {

// Run while the condition producers are still TTIR. Unknown ranges preserve
// their assertions; ordinary dead-code cleanup can retire proven checks'
// inputs.
void eliminateProvenOverflowAssertions(ModuleOp module);

} // namespace mlir::triton

#endif
