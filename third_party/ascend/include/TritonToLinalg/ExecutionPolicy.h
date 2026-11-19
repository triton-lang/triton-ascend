// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef TRITON_ASCEND_EXECUTION_POLICY_H
#define TRITON_ASCEND_EXECUTION_POLICY_H

#include "mlir/IR/Operation.h"

namespace mlir::triton {

enum class ExecutionKind { Vector, Cube, Scalar, Unknown };
enum class ExecutionMode { SIMD, SIMT, Unknown };

/// Check local motion legality, including structured tensor payloads.
/// Operand availability and the carried state size are checked by the caller.
bool canLookAhead(Operation *op);

/// Classify computations under the TA/backend SIMD contract.
/// Other execution modes and unsupported contexts return Unknown.
ExecutionKind classifyExecution(Operation *op, ExecutionMode mode);

} // namespace mlir::triton
#endif
