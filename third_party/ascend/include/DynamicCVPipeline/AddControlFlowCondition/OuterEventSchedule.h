/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */
#ifndef TRITON_ASCEND_OUTER_EVENT_SCHEDULE_H
#define TRITON_ASCEND_OUTER_EVENT_SCHEDULE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::triton {
struct ControlFlowConditionInfo;

struct OuterEventStaticSchedule {
  llvm::DenseMap<Operation *, int> offsets;
  int drain;
  bool groupSweepGrants = false;
};

// Prove a one-event-per-sweep schedule before adding scalar readiness checks.
FailureOr<OuterEventStaticSchedule>
inferOuterEventStaticSchedule(ArrayRef<Operation *> loops,
                              const ControlFlowConditionInfo &info);

LogicalResult groupOuterEventSweepGrants(scf::ForOp loop);
void issueOuterProducerBeforeWait(ModuleOp module);
} // namespace mlir::triton

#endif
