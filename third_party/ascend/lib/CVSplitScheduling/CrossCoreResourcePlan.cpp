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

#include "ascend/include/CVSplitScheduling/CrossCoreResourcePlan.h"
#include "ascend/include/CVSplitScheduling/CrossCoreOwnershipProof.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

static llvm::StringRef directionName(CrossCoreDirection direction) {
  return direction == CrossCoreDirection::CubeToVector ? "C2V" : "V2C";
}

static llvm::StringRef memoryName(PipelineMemorySpace memorySpace) {
  return memorySpace == PipelineMemorySpace::UB ? "UB" : "L1";
}

static llvm::StringRef statusName(ResourcePlanStatus status) {
  switch (status) {
  case ResourcePlanStatus::ValidKnownCapacity:
    return "valid-known-capacity";
  case ResourcePlanStatus::ValidUnknownCapacity:
    return "valid-unknown-capacity";
  case ResourcePlanStatus::FlagOverflow:
    return "flag-overflow";
  case ResourcePlanStatus::MemoryBudgetExceeded:
    return "memory-budget-exceeded";
  case ResourcePlanStatus::IncompleteBoundarySet:
    return "incomplete-boundary-set";
  case ResourcePlanStatus::PendingMaterialization:
    return "pending-materialization";
  case ResourcePlanStatus::UnresolvedOwnership:
    return "unresolved-ownership";
  }
  llvm_unreachable("unknown resource-plan status");
}

static llvm::StringRef
orderingName(ResourceOwnershipOrdering ordering) {
  switch (ordering) {
  case ResourceOwnershipOrdering::SameEngineOrder:
    return "same-engine-order";
  case ResourceOwnershipOrdering::ExistingCrossCorePath:
    return "existing-cross-core-path";
  case ResourceOwnershipOrdering::ExplicitReleaseRequired:
    return "explicit-release";
  case ResourceOwnershipOrdering::Unresolved:
    return "unresolved";
  }
  llvm_unreachable("unknown ownership ordering");
}

static bool hasFlagCapacity(const CrossCoreResourceLimits &limits,
                            unsigned requiredFlags) {
  if (limits.firstAvailableFlagId > limits.maximumFlagId)
    return requiredFlags == 0;
  return requiredFlags <=
         limits.maximumFlagId - limits.firstAvailableFlagId + 1;
}

static uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return std::numeric_limits<uint64_t>::max();
  return lhs * rhs;
}

static FailureOr<ResourceLineagePlan>
buildLineagePlan(const CrossCorePipelinePlan &pipelinePlan,
                 const CrossCorePhaseLineage &lineage,
                 unsigned interCoreBufferDepth) {
  if (lineage.boundaryIndices.empty() || interCoreBufferDepth == 0)
    return failure();

  SmallVector<unsigned> indices(lineage.boundaryIndices.begin(),
                                lineage.boundaryIndices.end());
  if (llvm::any_of(indices, [&](unsigned index) {
        return index >= pipelinePlan.boundaries.size();
      }))
    return failure();
  llvm::sort(indices, [&](unsigned lhs, unsigned rhs) {
    const CrossCoreBoundary &left = pipelinePlan.boundaries[lhs];
    const CrossCoreBoundary &right = pipelinePlan.boundaries[rhs];
    return left.key.lane != right.key.lane
               ? left.key.lane < right.key.lane
               : lhs < rhs;
  });

  const CrossCoreBoundary &first = pipelinePlan.boundaries[indices.front()];
  for (auto [expectedLane, boundaryIndex] : llvm::enumerate(indices)) {
    const CrossCoreBoundary &boundary =
        pipelinePlan.boundaries[boundaryIndex];
    if (boundary.key.originId != lineage.originId ||
        boundary.key.direction != lineage.direction ||
        boundary.key.lane != expectedLane ||
        boundary.footprintBytes != first.footprintBytes ||
        boundary.memorySpace != first.memorySpace ||
        boundary.elementType != first.elementType || !boundary.producer ||
        !boundary.earliestPublishAnchor)
      return failure();
  }

  const unsigned lanes = indices.size();
  const unsigned slots = std::min(lanes, interCoreBufferDepth);
  return ResourceLineagePlan{
      lineage.originId,
      lineage.direction,
      std::move(indices),
      lanes,
      slots,
      lineage.originId,
      first.footprintBytes,
      saturatingMultiply(slots, first.footprintBytes),
      first.memorySpace,
      first.elementType,
      slots == lanes};
}

static FailureOr<SmallVector<unsigned>>
buildEmissionLineageOrder(const CrossCorePipelinePlan &pipelinePlan) {
  DenseMap<int64_t, unsigned> lineageIndexByOrigin;
  for (auto [index, lineage] : llvm::enumerate(pipelinePlan.lineages))
    if (!lineageIndexByOrigin
             .try_emplace(lineage.originId, static_cast<unsigned>(index))
             .second)
      return failure();

  std::optional<unsigned> lastVectorToCubeOrder;
  for (const CrossCoreBoundary &boundary : pipelinePlan.boundaries)
    if (boundary.key.direction == CrossCoreDirection::VectorToCube)
      lastVectorToCubeOrder =
          std::max(lastVectorToCubeOrder.value_or(0),
                   boundary.producerOrder);

  SmallVector<unsigned> boundaries(pipelinePlan.boundaries.size());
  for (unsigned i = 0; i < boundaries.size(); ++i)
    boundaries[i] = i;
  llvm::sort(boundaries, [&](unsigned lhs, unsigned rhs) {
    return pipelinePlan.boundaries[lhs].producerOrder <
           pipelinePlan.boundaries[rhs].producerOrder;
  });
  llvm::stable_sort(boundaries, [&](unsigned lhs, unsigned rhs) {
    auto phase = [&](unsigned boundaryIndex) {
      const CrossCoreBoundary &boundary =
          pipelinePlan.boundaries[boundaryIndex];
      if (boundary.key.direction == CrossCoreDirection::VectorToCube)
        return 1;
      return lastVectorToCubeOrder &&
                     boundary.producerOrder < *lastVectorToCubeOrder
                 ? 0
                 : 2;
    };
    return phase(lhs) < phase(rhs);
  });

  DenseSet<unsigned> seen;
  SmallVector<unsigned> lineageOrder;
  for (unsigned boundaryIndex : boundaries) {
    const int64_t originId =
        pipelinePlan.boundaries[boundaryIndex].key.originId;
    auto lineageIt = lineageIndexByOrigin.find(originId);
    if (lineageIt == lineageIndexByOrigin.end())
      return failure();
    if (seen.insert(lineageIt->second).second)
      lineageOrder.push_back(lineageIt->second);
  }
  if (lineageOrder.size() != pipelinePlan.lineages.size())
    return failure();
  return lineageOrder;
}

} // namespace

FailureOr<CrossCoreResourcePlan>
buildCrossCoreResourcePlan(const CrossCorePipelinePlan &pipelinePlan,
                           const CrossCoreResourceLimits &limits) {
  if (limits.interCoreBufferDepth == 0 || pipelinePlan.lineages.empty())
    return failure();

  CrossCoreResourcePlan plan;
  plan.firstAvailableFlagId = limits.firstAvailableFlagId;
  plan.maximumFlagId = limits.maximumFlagId;
  plan.ubCapacityKnown = limits.extraUbBudgetBytes.has_value();
  plan.l1CapacityKnown = limits.l1BudgetBytes.has_value();

  for (const CrossCorePhaseLineage &lineage : pipelinePlan.lineages) {
    FailureOr<ResourceLineagePlan> lineagePlan =
        buildLineagePlan(pipelinePlan, lineage,
                         limits.interCoreBufferDepth);
    if (failed(lineagePlan))
      return failure();
    plan.lineages.push_back(std::move(*lineagePlan));
  }
  plan.completeLaneCoverage =
      pipelinePlan.laneCount > 0 &&
      llvm::all_of(plan.lineages, [&](const ResourceLineagePlan &lineage) {
        return lineage.laneCount == pipelinePlan.laneCount;
      });
  plan.anchorsComplete =
      llvm::all_of(plan.lineages, [&](const ResourceLineagePlan &lineage) {
        return llvm::all_of(lineage.boundaryIndices,
                            [&](unsigned boundaryIndex) {
          const CrossCoreBoundary &boundary =
              pipelinePlan.boundaries[boundaryIndex];
          return !boundary.consumers.empty() && boundary.lastReader &&
                 boundary.lastReaderOrder.has_value();
        });
      });

  for (ResourceLineagePlan &lineage : plan.lineages) {
    if (lineage.memorySpace == PipelineMemorySpace::UB)
      plan.baselineUbBytes += lineage.allocatedBytes;
    if (lineage.direction != CrossCoreDirection::VectorToCube ||
        limits.vectorToCubeSlotOverride == 0)
      continue;
    if (limits.vectorToCubeSlotOverride > lineage.laneCount)
      return failure();
    lineage.slotCount = limits.vectorToCubeSlotOverride;
    lineage.allocatedBytes =
        saturatingMultiply(lineage.slotCount, lineage.bytesPerSlot);
    lineage.lanePrivate = lineage.slotCount == lineage.laneCount;
  }

  unsigned flagsUsed = 0;
  for (const ResourceLineagePlan &lineage : plan.lineages)
    flagsUsed += lineage.slotCount;

  uint64_t ubBudgetLeft =
      limits.extraUbBudgetBytes.value_or(
          std::numeric_limits<uint64_t>::max());
  const bool maySpendUnknownUb =
      limits.extraUbBudgetBytes.has_value() ||
      limits.legacyUnknownUbBudgetMaySelect;
  DenseSet<int64_t> mergedGroupIds;

  llvm::MapVector<Type, SmallVector<unsigned>> c2vByElementType;
  for (auto [lineageIndex, lineage] : llvm::enumerate(plan.lineages))
    if (lineage.direction == CrossCoreDirection::CubeToVector)
      c2vByElementType[lineage.elementType].push_back(lineageIndex);

  for (auto &entry : c2vByElementType) {
    SmallVector<unsigned> &lineageIndices = entry.second;
    llvm::sort(lineageIndices, [&](unsigned lhs, unsigned rhs) {
      return plan.lineages[lhs].originId < plan.lineages[rhs].originId;
    });
    if (lineageIndices.size() < 2 || limits.interCoreBufferDepth < 2)
      continue;

    const unsigned lanes = plan.lineages[lineageIndices.front()].laneCount;
    const CrossCoreBoundary &shapeReference =
        pipelinePlan.boundaries[
            plan.lineages[lineageIndices.front()].boundaryIndices.front()];
    bool compatible = lanes > 0 && shapeReference.logicalShape.size() == 2;
    uint64_t pooledBytes = 0;
    uint64_t maxBytes = 0;
    unsigned extraFlags = 1;
    for (unsigned lineageIndex : lineageIndices) {
      ResourceLineagePlan &lineage = plan.lineages[lineageIndex];
      const CrossCoreBoundary &boundary =
          pipelinePlan.boundaries[lineage.boundaryIndices.front()];
      compatible &=
          lineage.laneCount == lanes &&
          lineage.memorySpace == PipelineMemorySpace::UB &&
          boundary.logicalShape.size() == 2 &&
          boundary.logicalShape.front() ==
              shapeReference.logicalShape.front();
      pooledBytes +=
          saturatingMultiply(lineage.slotCount, lineage.bytesPerSlot);
      maxBytes = std::max(maxBytes, lineage.bytesPerSlot);
      extraFlags += lanes - lineage.slotCount;
    }
    if (!compatible)
      continue;

    const uint64_t unionBytes = saturatingMultiply(lanes, maxBytes);
    const uint64_t extraBytes =
        unionBytes > pooledBytes ? unionBytes - pooledBytes : 0;
    if (!maySpendUnknownUb || extraBytes > ubBudgetLeft ||
        !hasFlagCapacity(limits, flagsUsed + extraFlags))
      continue;

    const int64_t groupId =
        plan.lineages[lineageIndices.front()].originId;
    for (unsigned lineageIndex : lineageIndices) {
      ResourceLineagePlan &lineage = plan.lineages[lineageIndex];
      lineage.slotCount = lanes;
      lineage.physicalGroup = groupId;
      lineage.allocatedBytes =
          saturatingMultiply(lanes, lineage.bytesPerSlot);
      lineage.lanePrivate = true;
    }
    if (limits.extraUbBudgetBytes)
      ubBudgetLeft -= extraBytes;
    flagsUsed += extraFlags;
    mergedGroupIds.insert(groupId);
  }

  DenseMap<int64_t, unsigned> groupIndexById;
  for (auto [lineageIndex, lineage] : llvm::enumerate(plan.lineages)) {
    auto groupIt = groupIndexById.find(lineage.physicalGroup);
    if (groupIt == groupIndexById.end()) {
      const unsigned groupIndex = plan.groups.size();
      groupIndexById[lineage.physicalGroup] = groupIndex;
      plan.groups.push_back(ResourcePhysicalGroup{
          lineage.physicalGroup, {static_cast<unsigned>(lineageIndex)},
          lineage.slotCount, lineage.bytesPerSlot, 0, lineage.memorySpace,
          false, std::nullopt});
      continue;
    }
    ResourcePhysicalGroup &group = plan.groups[groupIt->second];
    if (group.memorySpace != lineage.memorySpace ||
        group.slotCount != lineage.slotCount)
      return failure();
    group.lineageIndices.push_back(lineageIndex);
    group.bytesPerSlot = std::max(group.bytesPerSlot, lineage.bytesPerSlot);
    group.unionStorage = true;
  }

  for (ResourcePhysicalGroup &group : plan.groups) {
    group.allocatedBytes =
        saturatingMultiply(group.slotCount, group.bytesPerSlot);
    if (group.memorySpace == PipelineMemorySpace::UB)
      plan.allocatedUbBytes += group.allocatedBytes;
    else
      plan.allocatedL1Bytes += group.allocatedBytes;
  }
  plan.incrementalUbBytes =
      plan.allocatedUbBytes > plan.baselineUbBytes
          ? plan.allocatedUbBytes - plan.baselineUbBytes
          : 0;

  plan.forwardFlags = 0;
  for (const ResourceLineagePlan &lineage : plan.lineages)
    plan.forwardFlags += lineage.slotCount;
  plan.releaseFlags = mergedGroupIds.size();
  plan.requiredFlags = plan.forwardFlags + plan.releaseFlags;
  plan.flagCapacityProven = hasFlagCapacity(limits, plan.requiredFlags);

  FailureOr<SmallVector<unsigned>> emissionLineageOrder =
      buildEmissionLineageOrder(pipelinePlan);
  if (failed(emissionLineageOrder))
    return failure();
  SmallVector<unsigned> lineageFlagBase(plan.lineages.size());
  unsigned nextFlag = plan.firstAvailableFlagId;
  for (unsigned lineageIndex : *emissionLineageOrder) {
    lineageFlagBase[lineageIndex] = nextFlag;
    const ResourceLineagePlan &lineage = plan.lineages[lineageIndex];
    nextFlag += lineage.slotCount;
  }
  SmallVector<int64_t> delayedReleaseGroups(mergedGroupIds.begin(),
                                             mergedGroupIds.end());
  llvm::sort(delayedReleaseGroups);
  for (int64_t groupId : delayedReleaseGroups) {
    auto groupIt = groupIndexById.find(groupId);
    if (groupIt == groupIndexById.end())
      return failure();
    plan.groups[groupIt->second].releaseFlagId = nextFlag++;
  }
  if (nextFlag - plan.firstAvailableFlagId != plan.requiredFlags)
    return failure();

  for (auto [lineageIndex, lineage] : llvm::enumerate(plan.lineages))
    for (unsigned boundaryIndex : lineage.boundaryIndices) {
      const CrossCoreBoundary &boundary =
          pipelinePlan.boundaries[boundaryIndex];
      plan.assignments.push_back(ResourceSlotAssignment{
          boundaryIndex, static_cast<unsigned>(lineageIndex),
          boundary.key.lane, lineage.physicalGroup,
          boundary.key.lane % lineage.slotCount,
          lineageFlagBase[lineageIndex] +
              boundary.key.lane % lineage.slotCount});
    }

  for (const ResourcePhysicalGroup &group : plan.groups) {
    for (unsigned slot = 0; slot < group.slotCount; ++slot) {
      SmallVector<unsigned> uses;
      for (auto [assignmentIndex, assignment] :
           llvm::enumerate(plan.assignments))
        if (assignment.physicalGroup == group.groupId &&
            assignment.slot == slot)
          uses.push_back(assignmentIndex);
      if (uses.empty())
        continue;
      llvm::sort(uses, [&](unsigned lhs, unsigned rhs) {
        const ResourceSlotAssignment &left = plan.assignments[lhs];
        const ResourceSlotAssignment &right = plan.assignments[rhs];
        const CrossCoreBoundary &leftBoundary =
            pipelinePlan.boundaries[left.boundaryIndex];
        const CrossCoreBoundary &rightBoundary =
            pipelinePlan.boundaries[right.boundaryIndex];
        if (leftBoundary.producerOrder != rightBoundary.producerOrder)
          return leftBoundary.producerOrder < rightBoundary.producerOrder;
        if (left.lineageIndex != right.lineageIndex)
          return left.lineageIndex < right.lineageIndex;
        return left.lane < right.lane;
      });

      auto appendEdge = [&](unsigned fromUse, unsigned toUse,
                            bool loopCarried) {
        const ResourceSlotAssignment &from = plan.assignments[fromUse];
        const ResourceSlotAssignment &to = plan.assignments[toUse];
        const CrossCoreBoundary &fromBoundary =
            pipelinePlan.boundaries[from.boundaryIndex];
        const CrossCoreBoundary &toBoundary =
            pipelinePlan.boundaries[to.boundaryIndex];
        const bool explicitRelease =
            loopCarried && mergedGroupIds.contains(group.groupId);
        plan.ownershipEdges.push_back(ResourceOwnershipEdge{
            group.groupId, slot, from.boundaryIndex, to.boundaryIndex,
            fromBoundary.lastReader, toBoundary.producer, loopCarried,
            /*needsSeed=*/false,
            /*usesCrossCorePath=*/false,
            explicitRelease
                ? ResourceOwnershipOrdering::ExplicitReleaseRequired
                : ResourceOwnershipOrdering::Unresolved});
      };

      for (unsigned i = 1; i < uses.size(); ++i)
        appendEdge(uses[i - 1], uses[i], /*loopCarried=*/false);
      appendEdge(uses.back(), uses.front(), /*loopCarried=*/true);
    }
  }

  if (plan.anchorsComplete) {
    if (failed(proveCrossCoreResourceOwnership(
            pipelinePlan, delayedReleaseGroups, plan)))
      return failure();
  } else {
    plan.ownershipResolved =
        llvm::none_of(plan.ownershipEdges,
                      [](const ResourceOwnershipEdge &edge) {
      return edge.ordering == ResourceOwnershipOrdering::Unresolved;
    });
  }
  const bool ubWithinBudget =
      !limits.extraUbBudgetBytes ||
      plan.incrementalUbBytes <= *limits.extraUbBudgetBytes;
  const bool l1WithinBudget =
      !limits.l1BudgetBytes ||
      plan.allocatedL1Bytes <= *limits.l1BudgetBytes;

  if (!plan.completeLaneCoverage)
    plan.status = ResourcePlanStatus::IncompleteBoundarySet;
  else if (!plan.anchorsComplete)
    plan.status = ResourcePlanStatus::PendingMaterialization;
  else if (!plan.flagCapacityProven)
    plan.status = ResourcePlanStatus::FlagOverflow;
  else if (!ubWithinBudget || !l1WithinBudget)
    plan.status = ResourcePlanStatus::MemoryBudgetExceeded;
  else if (!plan.ownershipResolved)
    plan.status = ResourcePlanStatus::UnresolvedOwnership;
  else if (plan.ubCapacityKnown && plan.l1CapacityKnown)
    plan.status = ResourcePlanStatus::ValidKnownCapacity;
  else
    plan.status = ResourcePlanStatus::ValidUnknownCapacity;

  plan.selectionEligible =
      plan.status == ResourcePlanStatus::ValidKnownCapacity &&
      plan.ownershipResolved;
  return plan;
}

static void logResourcePlan(const CrossCoreResourcePlan &plan,
                            llvm::StringRef label) {
  LLVM_DEBUG({
    llvm::dbgs() << "[cv-split] " << label << "-plan status="
                 << statusName(plan.status)
                 << " lineages=" << plan.lineages.size()
                 << " groups=" << plan.groups.size()
                 << " complete-lanes="
                 << (plan.completeLaneCoverage ? "yes" : "no")
                 << " anchors-complete="
                 << (plan.anchorsComplete ? "yes" : "no")
                 << " selection-eligible="
                 << (plan.selectionEligible ? "yes" : "no") << "\n";

    for (const ResourceLineagePlan &lineage : plan.lineages)
      llvm::dbgs() << "[cv-split] " << label << "-lineage origin="
                   << lineage.originId
                   << " direction=" << directionName(lineage.direction)
                   << " lanes=" << lineage.laneCount
                   << " slots=" << lineage.slotCount
                   << " group=" << lineage.physicalGroup
                   << " bytes-per-slot=" << lineage.bytesPerSlot
                   << " memory=" << memoryName(lineage.memorySpace)
                   << " private=" << (lineage.lanePrivate ? "yes" : "no")
                   << "\n";

    for (const ResourcePhysicalGroup &group : plan.groups) {
      llvm::dbgs() << "[cv-split] " << label << "-group id="
                   << group.groupId
                   << " roles=" << group.lineageIndices.size()
                   << " slots=" << group.slotCount
                   << " bytes-per-slot=" << group.bytesPerSlot
                   << " allocated=" << group.allocatedBytes
                   << " memory=" << memoryName(group.memorySpace)
                   << " union=" << (group.unionStorage ? "yes" : "no")
                   << " release-flag=";
      if (group.releaseFlagId)
        llvm::dbgs() << *group.releaseFlagId;
      else
        llvm::dbgs() << "none";
      llvm::dbgs() << "\n";
    }

    for (const ResourceOwnershipEdge &edge : plan.ownershipEdges)
      llvm::dbgs() << "[cv-split] " << label << "-edge group="
                   << edge.physicalGroup << " slot=" << edge.slot
                   << " from=" << edge.fromBoundaryIndex
                   << " to=" << edge.toBoundaryIndex
                   << " loop-carried=" << (edge.loopCarried ? "yes" : "no")
                   << " ordering=" << orderingName(edge.ordering)
                   << " cross-core="
                   << (edge.usesCrossCorePath ? "yes" : "no")
                   << " seed=" << (edge.needsSeed ? "yes" : "no") << "\n";

    llvm::dbgs() << "[cv-split] " << label << "-flags first="
                 << plan.firstAvailableFlagId
                 << " required=" << plan.requiredFlags
                 << " forward=" << plan.forwardFlags
                 << " release=" << plan.releaseFlags
                 << " maximum=" << plan.maximumFlagId
                 << " proven=" << (plan.flagCapacityProven ? "yes" : "no")
                 << "\n";
    llvm::dbgs() << "[cv-split] " << label << "-memory UB="
                 << plan.allocatedUbBytes
                 << " L1=" << plan.allocatedL1Bytes
                 << " baseline-UB=" << plan.baselineUbBytes
                 << " incremental-UB=" << plan.incrementalUbBytes
                 << " UB-known=" << (plan.ubCapacityKnown ? "yes" : "no")
                 << " L1-known=" << (plan.l1CapacityKnown ? "yes" : "no")
                 << "\n";
    llvm::dbgs() << "[cv-split] ownership-summary same-resource="
                 << plan.sameEngineOwnershipEdges
                 << " cross-core=" << plan.crossCoreOwnershipEdges
                 << " explicit-release="
                 << plan.explicitReleaseOwnershipEdges
                 << " unresolved=" << plan.unresolvedOwnershipEdges
                 << " loop-carried=" << plan.loopCarriedOwnershipEdges
                 << " seeds=" << plan.seedRequirements << "\n";
  });
}

void logCrossCoreResourcePlan(const CrossCoreResourcePlan &plan) {
  logResourcePlan(plan, "resource");
}

void logMaterializedCrossCoreResourcePlan(const CrossCoreResourcePlan &plan) {
  logResourcePlan(plan, "bound-resource");
}

} // namespace mlir::triton::cv_split
