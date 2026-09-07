/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "ascend/include/CVSplitScheduling/PostCVSplitScheduleBinding.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

static llvm::StringRef statusName(PostCVSplitScheduleBindingStatus status) {
  switch (status) {
  case PostCVSplitScheduleBindingStatus::Ready:
    return "ready";
  case PostCVSplitScheduleBindingStatus::DetachedScheduleNotReady:
    return "detached-schedule-not-ready";
  case PostCVSplitScheduleBindingStatus::InvalidLineageSet:
    return "invalid-lineage-set";
  case PostCVSplitScheduleBindingStatus::MissingLaneBoundary:
    return "missing-lane-boundary";
  case PostCVSplitScheduleBindingStatus::AmbiguousLineageRole:
    return "ambiguous-lineage-role";
  case PostCVSplitScheduleBindingStatus::InvalidDependencyChain:
    return "invalid-dependency-chain";
  case PostCVSplitScheduleBindingStatus::InvalidEngineOwnership:
    return "invalid-engine-ownership";
  case PostCVSplitScheduleBindingStatus::InvalidOperationPlacement:
    return "invalid-operation-placement";
  }
  return "unknown";
}

static const CrossCoreBoundary *
findLaneBoundary(const CrossCorePipelinePlan &plan,
                 const CrossCorePhaseLineage &lineage, unsigned lane) {
  for (unsigned boundaryIndex : lineage.boundaryIndices) {
    if (boundaryIndex >= plan.boundaries.size())
      return nullptr;
    const CrossCoreBoundary &boundary = plan.boundaries[boundaryIndex];
    if (boundary.key.lane == lane)
      return &boundary;
  }
  return nullptr;
}

static bool belongsToCandidateBody(Operation *operation, Block *body) {
  if (!operation || !body)
    return false;
  if (operation->getBlock() == body)
    return true;
  Operation *container = body->getParentOp();
  return container && container->isAncestor(operation);
}

static bool operationDependsOnValue(Operation *operation, Value target,
                                    Block *body) {
  if (!operation || !target)
    return false;
  llvm::SmallVector<Value> worklist(operation->getOperands());
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (value == target)
      return true;
    if (!visited.insert(value).second)
      continue;
    Operation *definition = value.getDefiningOp();
    if (!belongsToCandidateBody(definition, body))
      continue;
    llvm::append_range(worklist, definition->getOperands());
  }
  return false;
}

static bool hasEngine(const Classification &classification,
                      Operation *operation, EngineType expected) {
  auto found = classification.find(operation);
  return found != classification.end() && found->second == expected;
}

static bool allConsumersHaveEngine(const Classification &classification,
                                   ArrayRef<Operation *> consumers,
                                   EngineType expected) {
  return !consumers.empty() &&
         llvm::all_of(consumers, [&](Operation *consumer) {
           return hasEngine(classification, consumer, expected);
         });
}

static void reject(PostCVSplitScheduleBinding &binding,
                   PostCVSplitScheduleBindingStatus status) {
  binding.status = status;
  binding.lanes.clear();
  binding.scoreConsumerCount = 0;
  binding.probabilityConsumerCount = 0;
  binding.productConsumerCount = 0;
  binding.verified = false;
  binding.publicationEligible = false;
  binding.mutationPerformed = false;
}

} // namespace

PostCVSplitScheduleBinding bindPostCVSplitScheduleAnchors(
    Block *body, const Classification &classification,
    const CrossCorePipelinePlan &pipelinePlan,
    const PostCVSplitDetachedSchedule &detachedSchedule) {
  PostCVSplitScheduleBinding binding;
  binding.logicalLaneCount = detachedSchedule.logicalLaneCount;
  if (!body || detachedSchedule.status !=
                   PostCVSplitDetachedScheduleStatus::Ready ||
      !detachedSchedule.verified || detachedSchedule.publicationEligible ||
      detachedSchedule.mutationPerformed)
    return binding;
  if (pipelinePlan.laneCount != detachedSchedule.logicalLaneCount ||
      pipelinePlan.lineages.size() != 3) {
    reject(binding, PostCVSplitScheduleBindingStatus::InvalidLineageSet);
    return binding;
  }

  const CrossCorePhaseLineage *probabilityLineage = nullptr;
  llvm::SmallVector<const CrossCorePhaseLineage *, 2> cubeToVectorLineages;
  for (const CrossCorePhaseLineage &lineage : pipelinePlan.lineages) {
    if (lineage.boundaryIndices.size() != pipelinePlan.laneCount) {
      reject(binding, PostCVSplitScheduleBindingStatus::InvalidLineageSet);
      return binding;
    }
    if (lineage.direction == CrossCoreDirection::VectorToCube) {
      if (probabilityLineage) {
        reject(binding, PostCVSplitScheduleBindingStatus::InvalidLineageSet);
        return binding;
      }
      probabilityLineage = &lineage;
    } else {
      cubeToVectorLineages.push_back(&lineage);
    }
  }
  if (!probabilityLineage || cubeToVectorLineages.size() != 2) {
    reject(binding, PostCVSplitScheduleBindingStatus::InvalidLineageSet);
    return binding;
  }

  const CrossCorePhaseLineage *scoreLineage = nullptr;
  const CrossCorePhaseLineage *productLineage = nullptr;
  for (const CrossCorePhaseLineage *candidate : cubeToVectorLineages) {
    unsigned dependentLanes = 0;
    for (unsigned lane = 0; lane < pipelinePlan.laneCount; ++lane) {
      const CrossCoreBoundary *candidateBoundary =
          findLaneBoundary(pipelinePlan, *candidate, lane);
      const CrossCoreBoundary *probabilityBoundary =
          findLaneBoundary(pipelinePlan, *probabilityLineage, lane);
      if (!candidateBoundary || !probabilityBoundary) {
        reject(binding,
               PostCVSplitScheduleBindingStatus::MissingLaneBoundary);
        return binding;
      }
      if (operationDependsOnValue(candidateBoundary->producer,
                                  probabilityBoundary->value, body))
        ++dependentLanes;
    }
    if (dependentLanes == pipelinePlan.laneCount) {
      if (productLineage) {
        reject(binding,
               PostCVSplitScheduleBindingStatus::AmbiguousLineageRole);
        return binding;
      }
      productLineage = candidate;
    } else if (dependentLanes == 0) {
      if (scoreLineage) {
        reject(binding,
               PostCVSplitScheduleBindingStatus::AmbiguousLineageRole);
        return binding;
      }
      scoreLineage = candidate;
    } else {
      reject(binding, PostCVSplitScheduleBindingStatus::AmbiguousLineageRole);
      return binding;
    }
  }
  if (!scoreLineage || !productLineage) {
    reject(binding, PostCVSplitScheduleBindingStatus::AmbiguousLineageRole);
    return binding;
  }

  for (unsigned lane = 0; lane < pipelinePlan.laneCount; ++lane) {
    const CrossCoreBoundary *score =
        findLaneBoundary(pipelinePlan, *scoreLineage, lane);
    const CrossCoreBoundary *probability =
        findLaneBoundary(pipelinePlan, *probabilityLineage, lane);
    const CrossCoreBoundary *product =
        findLaneBoundary(pipelinePlan, *productLineage, lane);
    if (!score || !probability || !product) {
      reject(binding, PostCVSplitScheduleBindingStatus::MissingLaneBoundary);
      return binding;
    }
    if (!belongsToCandidateBody(score->producer, body) ||
        !belongsToCandidateBody(probability->producer, body) ||
        !belongsToCandidateBody(product->producer, body) ||
        llvm::any_of(score->consumers, [&](Operation *operation) {
          return !belongsToCandidateBody(operation, body);
        }) ||
        llvm::any_of(probability->consumers, [&](Operation *operation) {
          return !belongsToCandidateBody(operation, body);
        }) ||
        llvm::any_of(product->consumers, [&](Operation *operation) {
          return !belongsToCandidateBody(operation, body);
        })) {
      reject(binding,
             PostCVSplitScheduleBindingStatus::InvalidOperationPlacement);
      return binding;
    }
    if (!hasEngine(classification, score->producer, EngineType::CUBE) ||
        !hasEngine(classification, probability->producer,
                   EngineType::VECTOR) ||
        !hasEngine(classification, product->producer, EngineType::CUBE) ||
        !allConsumersHaveEngine(classification, score->consumers,
                                EngineType::VECTOR) ||
        !allConsumersHaveEngine(classification, probability->consumers,
                                EngineType::CUBE) ||
        !allConsumersHaveEngine(classification, product->consumers,
                                EngineType::VECTOR)) {
      reject(binding,
             PostCVSplitScheduleBindingStatus::InvalidEngineOwnership);
      return binding;
    }
    if (!operationDependsOnValue(probability->producer, score->value, body) ||
        !operationDependsOnValue(product->producer, probability->value,
                                 body)) {
      reject(binding,
             PostCVSplitScheduleBindingStatus::InvalidDependencyChain);
      return binding;
    }

    PostCVSplitLaneAnchorBinding laneBinding;
    laneBinding.lane = lane;
    laneBinding.scoreOriginId = score->key.originId;
    laneBinding.probabilityOriginId = probability->key.originId;
    laneBinding.productOriginId = product->key.originId;
    laneBinding.scoreProducer = score->producer;
    laneBinding.probabilityProducer = probability->producer;
    laneBinding.productProducer = product->producer;
    laneBinding.scoreConsumers = score->consumers;
    laneBinding.probabilityConsumers = probability->consumers;
    laneBinding.productConsumers = product->consumers;
    binding.scoreConsumerCount += laneBinding.scoreConsumers.size();
    binding.probabilityConsumerCount +=
        laneBinding.probabilityConsumers.size();
    binding.productConsumerCount += laneBinding.productConsumers.size();
    binding.lanes.push_back(std::move(laneBinding));
  }

  binding.status = PostCVSplitScheduleBindingStatus::Ready;
  binding.verified = true;
  binding.publicationEligible = false;
  binding.mutationPerformed = false;
  return binding;
}

void logPostCVSplitScheduleBinding(
    const PostCVSplitScheduleBinding &binding) {
  LLVM_DEBUG({
    llvm::dbgs() << "[cv-split] schedule-binding status="
                 << statusName(binding.status)
                 << " lanes=" << binding.logicalLaneCount
                 << " bound-lanes=" << binding.lanes.size()
                 << " score-consumers=" << binding.scoreConsumerCount
                 << " probability-consumers="
                 << binding.probabilityConsumerCount
                 << " product-consumers=" << binding.productConsumerCount
                 << " verified=" << (binding.verified ? "yes" : "no")
                 << " publication-eligible=no mutation=no\n";
    for (const PostCVSplitLaneAnchorBinding &lane : binding.lanes)
      llvm::dbgs() << "[cv-split] schedule-binding-lane lane=" << lane.lane
                   << " score-origin=" << lane.scoreOriginId
                   << " probability-origin=" << lane.probabilityOriginId
                   << " product-origin=" << lane.productOriginId
                   << " score-consumers=" << lane.scoreConsumers.size()
                   << " probability-consumers="
                   << lane.probabilityConsumers.size()
                   << " product-consumers=" << lane.productConsumers.size()
                   << "\n";
  });
}

} // namespace mlir::triton::cv_split
