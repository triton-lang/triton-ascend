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

#include "ascend/include/CVSplitScheduling/CrossCoreScheduleCandidate.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

struct MatrixLineageFacts {
  unsigned pipelineLineageIndex;
  int64_t originId;
  unsigned firstProducerOrder;
  unsigned laneCount;
  unsigned transferSlotCount;
};

static llvm::StringRef directionName(CrossCoreDirection direction) {
  return direction == CrossCoreDirection::CubeToVector ? "C2V" : "V2C";
}

static FailureOr<const ResourceLineagePlan *>
findResourceLineage(const CrossCorePhaseLineage &pipelineLineage,
                    const CrossCoreResourcePlan &resourcePlan) {
  const ResourceLineagePlan *match = nullptr;
  for (const ResourceLineagePlan &resourceLineage : resourcePlan.lineages) {
    if (resourceLineage.originId != pipelineLineage.originId ||
        resourceLineage.direction != pipelineLineage.direction)
      continue;
    if (match)
      return failure();
    match = &resourceLineage;
  }
  if (!match)
    return failure();
  return match;
}

static FailureOr<unsigned>
firstProducerOrder(const CrossCorePhaseLineage &lineage,
                   const CrossCorePipelinePlan &pipelinePlan) {
  if (lineage.boundaryIndices.empty())
    return failure();
  std::optional<unsigned> first;
  for (unsigned boundaryIndex : lineage.boundaryIndices) {
    if (boundaryIndex >= pipelinePlan.boundaries.size())
      return failure();
    const CrossCoreBoundary &boundary = pipelinePlan.boundaries[boundaryIndex];
    if (boundary.key.originId != lineage.originId ||
        boundary.key.direction != lineage.direction || !boundary.producer)
      return failure();
    first = std::min(first.value_or(boundary.producerOrder),
                     boundary.producerOrder);
  }
  return *first;
}

} // namespace

FailureOr<CrossCoreScheduleCandidateSet>
buildCrossCoreScheduleCandidates(const CrossCorePipelinePlan &pipelinePlan,
                                 const CrossCoreResourcePlan &resourcePlan) {
  if (pipelinePlan.laneCount == 0 || pipelinePlan.lineages.empty() ||
      pipelinePlan.lineages.size() != resourcePlan.lineages.size() ||
      !resourcePlan.completeLaneCoverage)
    return failure();

  SmallVector<MatrixLineageFacts> matrixLineages;
  for (auto [lineageIndex, pipelineLineage] :
       llvm::enumerate(pipelinePlan.lineages)) {
    FailureOr<const ResourceLineagePlan *> resourceLineage =
        findResourceLineage(pipelineLineage, resourcePlan);
    if (failed(resourceLineage) ||
        (*resourceLineage)->laneCount != pipelinePlan.laneCount ||
        (*resourceLineage)->slotCount == 0 ||
        (*resourceLineage)->slotCount > (*resourceLineage)->laneCount)
      return failure();

    FailureOr<unsigned> producerOrder =
        firstProducerOrder(pipelineLineage, pipelinePlan);
    if (failed(producerOrder))
      return failure();

    if (pipelineLineage.direction != CrossCoreDirection::CubeToVector)
      continue;
    matrixLineages.push_back({static_cast<unsigned>(lineageIndex),
                              pipelineLineage.originId, *producerOrder,
                              (*resourceLineage)->laneCount,
                              (*resourceLineage)->slotCount});
  }
  if (matrixLineages.empty())
    return failure();

  llvm::sort(matrixLineages,
             [](const MatrixLineageFacts &lhs, const MatrixLineageFacts &rhs) {
               return lhs.firstProducerOrder != rhs.firstProducerOrder
                          ? lhs.firstProducerOrder < rhs.firstProducerOrder
                          : lhs.originId < rhs.originId;
             });

  CrossCoreScheduleCandidateSet candidateSet;
  candidateSet.logicalLaneCount = pipelinePlan.laneCount;
  candidateSet.matrixLineageCount = matrixLineages.size();

  const unsigned maximumDepth = std::min(2u, pipelinePlan.laneCount);
  const unsigned candidateCount =
      maximumDepth == 1 ? 1 : matrixLineages.size() + 1;
  for (unsigned candidateId = 0; candidateId < candidateCount; ++candidateId) {
    CrossCoreScheduleCandidate candidate;
    candidate.candidateId = candidateId;
    candidate.logicalLaneCount = pipelinePlan.laneCount;
    candidate.waveWidth = pipelinePlan.laneCount;
    candidate.maximumLiveMatrixResultsPerLineage = 1;
    candidate.prefetchLimit = 1;

    const unsigned widenedPrefix = maximumDepth == 1 ? 0 : candidateId;
    for (auto [phaseOrdinal, lineage] : llvm::enumerate(matrixLineages)) {
      const unsigned inFlightLimit =
          phaseOrdinal < widenedPrefix ? maximumDepth : 1;
      if (inFlightLimit == 0 || inFlightLimit > lineage.laneCount)
        return failure();
      candidate.maximumLiveMatrixResultsPerLineage =
          std::max(candidate.maximumLiveMatrixResultsPerLineage, inFlightLimit);
      candidate.prefetchLimit =
          std::max(candidate.prefetchLimit, inFlightLimit);
      candidate.matrixLineageLimits.push_back(
          {lineage.pipelineLineageIndex, lineage.originId,
           CrossCoreDirection::CubeToVector,
           static_cast<unsigned>(phaseOrdinal), inFlightLimit,
           lineage.transferSlotCount});
    }
    candidateSet.candidates.push_back(std::move(candidate));
  }

  return candidateSet;
}

void logCrossCoreScheduleCandidates(
    const CrossCoreScheduleCandidateSet &candidateSet) {
  LLVM_DEBUG({
    llvm::dbgs() << "[cv-split] schedule-candidates mode=diagnose lanes="
                 << candidateSet.logicalLaneCount
                 << " matrix-lineages=" << candidateSet.matrixLineageCount
                 << " count=" << candidateSet.candidates.size() << "\n";
    for (const CrossCoreScheduleCandidate &candidate :
         candidateSet.candidates) {
      llvm::dbgs() << "[cv-split] schedule-candidate id="
                   << candidate.candidateId << " wave=" << candidate.waveWidth
                   << " live-matrix="
                   << candidate.maximumLiveMatrixResultsPerLineage
                   << " prefetch=" << candidate.prefetchLimit << " selectable="
                   << (candidate.selectionEligible ? "yes" : "no") << "\n";
      for (const ScheduleMatrixLineageLimit &lineage :
           candidate.matrixLineageLimits)
        llvm::dbgs() << "[cv-split] schedule-lineage candidate="
                     << candidate.candidateId
                     << " phase=" << lineage.phaseOrdinal
                     << " origin=" << lineage.originId
                     << " direction=" << directionName(lineage.direction)
                     << " in-flight=" << lineage.inFlightLimit
                     << " transfer-slots=" << lineage.transferSlotCount << "\n";
    }
  });
}

} // namespace mlir::triton::cv_split
