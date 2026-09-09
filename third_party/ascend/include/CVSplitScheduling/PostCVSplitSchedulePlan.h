/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_SCHEDULE_PLAN_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_SCHEDULE_PLAN_H

#include "ascend/include/CVSplitScheduling/PostCVSplitRequestExtraction.h"
#include "ascend/include/CVSplitScheduling/CrossCoreResourcePlan.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::triton::cv_split {

enum class PostCVSplitSchedulePlanStatus {
  ValidKnownCapacity,
  ValidUnknownCapacity,
  UnsupportedRecurrence,
  UnsupportedGeometry,
  FlagOverflow,
  MemoryBudgetExceeded,
  ArithmeticOverflow,
  InvalidInput,
};

enum class PostCVSplitLineageRole { Score, Probability, Product };

enum class PostCVSplitEventKind { Forward, Release };

struct AttentionRecurrenceDescriptor {
  unsigned logicalLaneCount = 0;
  unsigned matrixLineageCount = 0;
  unsigned transferLineageCount = 0;
  uint32_t vectorRows = 0;
  uint32_t scoreWidth = 0;
  uint32_t headDimension = 0;
  bool hasMaximum = false;
  bool hasRowReduceMax = false;
  bool hasExp = false;
  bool hasRowReduceSum = false;
  bool hasNormalizationArithmetic = false;
  bool hasCast = false;
  bool hasPermute = false;
};

struct PostCVSplitSlotAssignment {
  PostCVSplitLineageRole role = PostCVSplitLineageRole::Score;
  unsigned lane = 0;
  unsigned slot = 0;
};

struct PostCVSplitEventPlan {
  unsigned logicalFlagId = 0;
  PostCVSplitLineageRole role = PostCVSplitLineageRole::Score;
  PostCVSplitEventKind kind = PostCVSplitEventKind::Forward;
  unsigned laneOrSlot = 0;
  PrincipalResource signalingResource = PrincipalResource::ScalarControl;
  PrincipalResource waitingResource = PrincipalResource::ScalarControl;
  bool loopCarried = false;
};

struct PostCVSplitVectorLanePlan {
  unsigned lane = 0;
  uint32_t rows = 0;
  uint32_t chunkWidth = 0;
  uint32_t chunkCount = 0;
  bool directNzPacking = false;
};

struct PostCVSplitReductionStep {
  unsigned level = 0;
  unsigned leftValue = 0;
  unsigned rightValue = 0;
  unsigned resultValue = 0;
};

struct PostCVSplitBackendRequirements {
  bool disableAutoBindSubBlock = true;
  bool enableGraphSync = true;
  unsigned vfMergeLevel = 1;
};

struct PostCVSplitSchedulePlan {
  PostCVSplitSchedulePlanStatus status =
      PostCVSplitSchedulePlanStatus::InvalidInput;
  AttentionRecurrenceDescriptor recurrence;
  unsigned scoreLiveDepth = 0;
  unsigned probabilitySlotCount = 0;
  unsigned productLiveDepth = 0;
  uint64_t scoreBytesPerSlot = 0;
  uint64_t probabilityBytesPerSlot = 0;
  uint64_t productBytesPerSlot = 0;
  uint64_t allocatedUbBytes = 0;
  uint64_t incrementalUbBytes = 0;
  uint64_t allocatedL1Bytes = 0;
  unsigned firstLogicalFlagId = 0;
  unsigned maximumLogicalFlagId = 0;
  unsigned forwardEventCount = 0;
  unsigned releaseEventCount = 0;
  unsigned requiredEventCount = 0;
  unsigned reductionRootValue = 0;
  bool affineTreeRequired = false;
  bool selectionEligible = false;
  PostCVSplitBackendRequirements backend;
  llvm::SmallVector<PostCVSplitSlotAssignment> slots;
  llvm::SmallVector<PostCVSplitEventPlan> events;
  llvm::SmallVector<PostCVSplitVectorLanePlan> vectorLanes;
  llvm::SmallVector<PostCVSplitReductionStep> reductionSteps;
};

/// Builds and verifies a numeric, parameterized post-CVSplit schedule plan without
/// retaining MLIR handles or mutating IR.
PostCVSplitSchedulePlan buildPostCVSplitSchedulePlan(
    const PostCVSplitRequestSet &requests,
    const CrossCoreResourcePlan &currentResources,
    const CrossCoreResourceLimits &limits);

void logPostCVSplitSchedulePlan(const PostCVSplitSchedulePlan &plan);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_SCHEDULE_PLAN_H
