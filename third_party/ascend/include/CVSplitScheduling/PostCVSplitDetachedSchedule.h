/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_DETACHED_SCHEDULE_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_DETACHED_SCHEDULE_H

#include "ascend/include/CVSplitScheduling/PostCVSplitSchedulePlan.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::triton::cv_split {

enum class PostCVSplitDetachedScheduleStatus {
  Ready,
  PlanNotBuildable,
  MissingSlotAssignment,
  MissingEvent,
  EventContractMismatch,
  InvalidCommandOrder,
  LiveDepthExceeded,
  ReductionMismatch,
  BackendMismatch,
};

enum class PostCVSplitDetachedSide { Cube, Vector };

enum class PostCVSplitDetachedCommandKind {
  ScoreMatmul,
  ScoreReleaseWait,
  ScorePublish,
  ProbabilityWait,
  ProductMatmul,
  ProductReleaseWait,
  ProductPublish,
  ScoreWait,
  RowwiseSoftmax,
  DirectNzPack,
  ProbabilityPublish,
  ScoreRelease,
  ProductWait,
  ProductAccumulate,
  ProductRelease,
  AffineReduce,
};

struct PostCVSplitDetachedCommand {
  PostCVSplitDetachedSide side = PostCVSplitDetachedSide::Cube;
  PostCVSplitDetachedCommandKind kind =
      PostCVSplitDetachedCommandKind::ScoreMatmul;
  PostCVSplitLineageRole role = PostCVSplitLineageRole::Score;
  unsigned lane = 0;
  unsigned slot = 0;
  PrincipalResource resource = PrincipalResource::ScalarControl;
  uint32_t rows = 0;
  uint32_t chunkWidth = 0;
  uint32_t chunkCount = 0;
  unsigned reductionLevel = 0;
  unsigned leftValue = 0;
  unsigned rightValue = 0;
  unsigned resultValue = 0;
  unsigned logicalFlagId = 0;
  bool hasRole = true;
  bool hasSlot = true;
  bool hasEvent = false;
  bool signalsEvent = false;
  bool directNzPacking = false;
};

struct PostCVSplitDetachedEventUse {
  unsigned logicalFlagId = 0;
  PostCVSplitLineageRole role = PostCVSplitLineageRole::Score;
  PostCVSplitEventKind kind = PostCVSplitEventKind::Forward;
  unsigned expectedUses = 0;
  unsigned signalUses = 0;
  unsigned waitUses = 0;
  bool paired = false;
};

struct PostCVSplitDetachedSchedule {
  PostCVSplitDetachedScheduleStatus status =
      PostCVSplitDetachedScheduleStatus::PlanNotBuildable;
  unsigned logicalLaneCount = 0;
  unsigned observedMaxScoreLive = 0;
  unsigned observedMaxProductLive = 0;
  unsigned reductionRootValue = 0;
  bool verified = false;
  bool publicationEligible = false;
  bool mutationPerformed = false;
  PostCVSplitBackendRequirements backend;
  llvm::SmallVector<PostCVSplitDetachedCommand> cubeCommands;
  llvm::SmallVector<PostCVSplitDetachedCommand> vectorCommands;
  llvm::SmallVector<PostCVSplitDetachedEventUse> eventUses;
};

/// Lowers a verified numeric post-CVSplit plan into complete detached CUBE and
/// VECTOR command streams. The result owns no MLIR handles and never mutates
/// or publishes live IR.
PostCVSplitDetachedSchedule
buildPostCVSplitDetachedSchedule(const PostCVSplitSchedulePlan &plan);

void logPostCVSplitDetachedSchedule(
    const PostCVSplitDetachedSchedule &schedule);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_DETACHED_SCHEDULE_H
