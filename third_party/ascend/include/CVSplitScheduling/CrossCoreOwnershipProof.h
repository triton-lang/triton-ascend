/* Copyright (c) Huawei Technologies Co., Ltd. 2026. */
#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_OWNERSHIP_PROOF_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_OWNERSHIP_PROOF_H
#include "ascend/include/CVSplitScheduling/CrossCorePipelinePlan.h"
#include "ascend/include/CVSplitScheduling/CrossCoreResourcePlan.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
namespace mlir::triton::cv_split {
LogicalResult proveCrossCoreResourceOwnership(
    const CrossCorePipelinePlan &materializedPlan,
    llvm::ArrayRef<int64_t> delayedReleaseGroups,
    CrossCoreResourcePlan &resourcePlan);
} // namespace mlir::triton::cv_split
#endif
