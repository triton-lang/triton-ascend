/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_SCHEDULE_BINDING_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_SCHEDULE_BINDING_H

#include "ascend/include/CVSplitScheduling/CrossCorePipelinePlan.h"
#include "ascend/include/CVSplitScheduling/PostCVSplitDetachedSchedule.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::cv_split {

enum class PostCVSplitScheduleBindingStatus {
  Ready,
  DetachedScheduleNotReady,
  InvalidLineageSet,
  MissingLaneBoundary,
  AmbiguousLineageRole,
  InvalidDependencyChain,
  InvalidEngineOwnership,
  InvalidOperationPlacement,
};

struct PostCVSplitLaneAnchorBinding {
  unsigned lane = 0;
  int64_t scoreOriginId = 0;
  int64_t probabilityOriginId = 0;
  int64_t productOriginId = 0;
  Operation *scoreProducer = nullptr;
  Operation *probabilityProducer = nullptr;
  Operation *productProducer = nullptr;
  llvm::SmallVector<Operation *> scoreConsumers;
  llvm::SmallVector<Operation *> probabilityConsumers;
  llvm::SmallVector<Operation *> productConsumers;
};

struct PostCVSplitScheduleBinding {
  PostCVSplitScheduleBindingStatus status =
      PostCVSplitScheduleBindingStatus::DetachedScheduleNotReady;
  unsigned logicalLaneCount = 0;
  unsigned scoreConsumerCount = 0;
  unsigned probabilityConsumerCount = 0;
  unsigned productConsumerCount = 0;
  bool verified = false;
  bool publicationEligible = false;
  bool mutationPerformed = false;
  llvm::SmallVector<PostCVSplitLaneAnchorBinding> lanes;
};

/// Binds a verified detached schedule to semantic producer/consumer anchors in
/// the current candidate body. Returned handles are non-owning and valid only
/// while that body remains unchanged. This function never mutates IR.
PostCVSplitScheduleBinding bindPostCVSplitScheduleAnchors(
    Block *body, const Classification &classification,
    const CrossCorePipelinePlan &pipelinePlan,
    const PostCVSplitDetachedSchedule &detachedSchedule);

void logPostCVSplitScheduleBinding(
    const PostCVSplitScheduleBinding &binding);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_SCHEDULE_BINDING_H
