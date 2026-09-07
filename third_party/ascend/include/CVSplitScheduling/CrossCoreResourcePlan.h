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

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_RESOURCE_PLAN_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_RESOURCE_PLAN_H

#include "ascend/include/CVSplitScheduling/CrossCorePipelinePlan.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::triton::cv_split {

enum class ResourcePlanStatus {
  ValidKnownCapacity,
  ValidUnknownCapacity,
  FlagOverflow,
  MemoryBudgetExceeded,
  IncompleteBoundarySet,
  PendingMaterialization,
  UnresolvedOwnership
};

enum class ResourceOwnershipOrdering {
  SameEngineOrder,
  ExistingCrossCorePath,
  ExplicitReleaseRequired,
  Unresolved
};

struct CrossCoreResourceLimits {
  unsigned interCoreBufferDepth = 0;
  std::optional<uint64_t> extraUbBudgetBytes;
  std::optional<uint64_t> l1BudgetBytes;
  unsigned firstAvailableFlagId = 0;
  unsigned maximumFlagId = 0;
  unsigned vectorToCubeSlotOverride = 0;
  bool legacyUnknownUbBudgetMaySelect = true;
};

struct ResourceLineagePlan {
  int64_t originId;
  CrossCoreDirection direction;
  llvm::SmallVector<unsigned> boundaryIndices;
  unsigned laneCount;
  unsigned slotCount;
  int64_t physicalGroup;
  uint64_t bytesPerSlot;
  uint64_t allocatedBytes;
  PipelineMemorySpace memorySpace;
  Type elementType;
  bool lanePrivate;
};

struct ResourcePhysicalGroup {
  int64_t groupId;
  llvm::SmallVector<unsigned> lineageIndices;
  unsigned slotCount;
  uint64_t bytesPerSlot;
  uint64_t allocatedBytes;
  PipelineMemorySpace memorySpace;
  bool unionStorage;
  std::optional<unsigned> releaseFlagId;
};

struct ResourceSlotAssignment {
  unsigned boundaryIndex;
  unsigned lineageIndex;
  unsigned lane;
  int64_t physicalGroup;
  unsigned slot;
  unsigned forwardFlagId;
};

struct ResourceOwnershipEdge {
  int64_t physicalGroup;
  unsigned slot;
  unsigned fromBoundaryIndex;
  unsigned toBoundaryIndex;
  Operation *lastReader;
  Operation *nextWriter;
  bool loopCarried;
  bool needsSeed;
  bool usesCrossCorePath;
  ResourceOwnershipOrdering ordering;
};

struct CrossCoreResourcePlan {
  ResourcePlanStatus status = ResourcePlanStatus::ValidUnknownCapacity;
  bool ubCapacityKnown = false;
  bool l1CapacityKnown = false;
  bool flagCapacityProven = false;
  bool ownershipResolved = false;
  bool completeLaneCoverage = false;
  bool anchorsComplete = false;
  bool selectionEligible = false;
  unsigned firstAvailableFlagId = 0;
  unsigned maximumFlagId = 0;
  unsigned forwardFlags = 0;
  unsigned releaseFlags = 0;
  unsigned requiredFlags = 0;
  uint64_t allocatedUbBytes = 0;
  uint64_t allocatedL1Bytes = 0;
  uint64_t baselineUbBytes = 0;
  uint64_t incrementalUbBytes = 0;
  unsigned sameEngineOwnershipEdges = 0;
  unsigned crossCoreOwnershipEdges = 0;
  unsigned explicitReleaseOwnershipEdges = 0;
  unsigned unresolvedOwnershipEdges = 0;
  unsigned loopCarriedOwnershipEdges = 0;
  unsigned seedRequirements = 0;
  llvm::SmallVector<ResourceLineagePlan> lineages;
  llvm::SmallVector<ResourcePhysicalGroup> groups;
  llvm::SmallVector<ResourceSlotAssignment> assignments;
  llvm::SmallVector<ResourceOwnershipEdge> ownershipEdges;
};

FailureOr<CrossCoreResourcePlan>
buildCrossCoreResourcePlan(const CrossCorePipelinePlan &pipelinePlan,
                           const CrossCoreResourceLimits &limits);

void logCrossCoreResourcePlan(const CrossCoreResourcePlan &plan);

void logMaterializedCrossCoreResourcePlan(const CrossCoreResourcePlan &plan);

} // namespace mlir::triton::cv_split

#endif
