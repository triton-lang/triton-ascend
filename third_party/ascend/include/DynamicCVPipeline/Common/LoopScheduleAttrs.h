/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ASCEND_LOOP_SCHEDULE_ATTRS_H
#define TRITON_ASCEND_LOOP_SCHEDULE_ATTRS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::CVPipeline {

// Module request consumed by AddDynamicCVPipeline.
inline constexpr llvm::StringLiteral kEnableOuterMainLoop =
    "ssbuffer.enable_outer_main_loop";
// Internal loop selection and the module flag for outer-only passes.
inline constexpr llvm::StringLiteral kOuterMainLoopHint =
    "ssbuffer.outer_main_loop";
// Set only after normalization has established a supported scheduling domain.
inline constexpr llvm::StringLiteral kExplicitMainLoop =
    "ssbuffer.explicit_main_loop";
// Number of leading scalar control arguments of a normalized event loop.
inline constexpr llvm::StringLiteral kOuterEventControlCount =
    "ssbuffer.outer_event_control_count";
// A scalar event transition must not be fused with a tensor payload guard.
inline constexpr llvm::StringLiteral kOuterEventControl =
    "ssbuffer.outer_event_control";
inline constexpr llvm::StringLiteral kOuterEventFallback =
    "ssbuffer.outer_event_fallback";

inline bool hasOuterMainLoop(ModuleOp module) {
  return module->hasAttr(kOuterMainLoopHint);
}

} // namespace mlir::CVPipeline

#endif // TRITON_ASCEND_LOOP_SCHEDULE_ATTRS_H
