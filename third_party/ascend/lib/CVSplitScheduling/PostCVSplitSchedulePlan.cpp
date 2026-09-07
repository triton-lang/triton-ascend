/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "ascend/include/CVSplitScheduling/PostCVSplitSchedulePlan.h"
#include "ascend/include/CVSplitScheduling/HardwareConstants.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

constexpr uint32_t kVectorChunkElements = 4 * kNzTileSize;
constexpr unsigned kMaximumLogicalFlagId = 15;

static bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool isFloatingInput(CVSplitElementType type) {
  return type == CVSplitElementType::F16 ||
         type == CVSplitElementType::BF16;
}

static bool sameCubeRequest(const CVSplitCubeRequest &lhs,
                            const CVSplitCubeRequest &rhs) {
  return lhs.kind == rhs.kind && lhs.m == rhs.m && lhs.n == rhs.n &&
         lhs.k == rhs.k && lhs.lhsType == rhs.lhsType &&
         lhs.rhsType == rhs.rhsType &&
         lhs.accumulatorType == rhs.accumulatorType &&
         lhs.lhsLayout == rhs.lhsLayout && lhs.rhsLayout == rhs.rhsLayout &&
         lhs.outputLayout == rhs.outputLayout &&
         lhs.transposeLhs == rhs.transposeLhs &&
         lhs.transposeRhs == rhs.transposeRhs &&
         lhs.lhsBytes == rhs.lhsBytes && lhs.rhsBytes == rhs.rhsBytes &&
         lhs.resultBytes == rhs.resultBytes && lhs.lhsPath == rhs.lhsPath &&
         lhs.rhsPath == rhs.rhsPath && lhs.drainKind == rhs.drainKind;
}

static bool pathStartsInUb(CVSplitMemoryPath path) {
  return path == CVSplitMemoryPath::UBToL1ToL0A ||
         path == CVSplitMemoryPath::UBToL1ToL0B;
}

static bool sameTransferRequest(const CVSplitTransferRequest &lhs,
                                const CVSplitTransferRequest &rhs) {
  return lhs.kind == rhs.kind && lhs.source == rhs.source &&
         lhs.destination == rhs.destination &&
         lhs.sourceLayout == rhs.sourceLayout &&
         lhs.destinationLayout == rhs.destinationLayout &&
         lhs.elementType == rhs.elementType && lhs.bytes == rhs.bytes &&
         lhs.rows == rhs.rows && lhs.columns == rhs.columns &&
         lhs.rowSplit == rhs.rowSplit;
}

static llvm::StringRef statusName(PostCVSplitSchedulePlanStatus status) {
  switch (status) {
  case PostCVSplitSchedulePlanStatus::ValidKnownCapacity:
    return "valid-known-capacity";
  case PostCVSplitSchedulePlanStatus::ValidUnknownCapacity:
    return "valid-unknown-capacity";
  case PostCVSplitSchedulePlanStatus::UnsupportedRecurrence:
    return "unsupported-recurrence";
  case PostCVSplitSchedulePlanStatus::UnsupportedGeometry:
    return "unsupported-geometry";
  case PostCVSplitSchedulePlanStatus::FlagOverflow:
    return "flag-overflow";
  case PostCVSplitSchedulePlanStatus::MemoryBudgetExceeded:
    return "memory-budget-exceeded";
  case PostCVSplitSchedulePlanStatus::ArithmeticOverflow:
    return "arithmetic-overflow";
  case PostCVSplitSchedulePlanStatus::InvalidInput:
    return "invalid-input";
  }
  return "unknown";
}

static llvm::StringRef roleName(PostCVSplitLineageRole role) {
  switch (role) {
  case PostCVSplitLineageRole::Score:
    return "score";
  case PostCVSplitLineageRole::Probability:
    return "probability";
  case PostCVSplitLineageRole::Product:
    return "product";
  }
  return "unknown";
}

static llvm::StringRef eventName(PostCVSplitEventKind kind) {
  return kind == PostCVSplitEventKind::Forward ? "forward" : "release";
}

static llvm::StringRef resourceName(PrincipalResource resource) {
  switch (resource) {
  case PrincipalResource::Matrix:
    return "matrix";
  case PrincipalResource::Fixpipe:
    return "fixpipe";
  case PrincipalResource::Vector:
    return "vector";
  case PrincipalResource::Mte1:
    return "mte1";
  case PrincipalResource::Mte2:
    return "mte2";
  case PrincipalResource::Mte3:
    return "mte3";
  case PrincipalResource::ScalarControl:
    return "scalar";
  }
  return "unknown";
}

static bool verifyPlan(const PostCVSplitSchedulePlan &plan) {
  const unsigned lanes = plan.recurrence.logicalLaneCount;
  if (lanes < 2 || plan.scoreLiveDepth == 0 ||
      plan.productLiveDepth == 0 || plan.probabilitySlotCount != lanes ||
      plan.slots.size() != 3 * lanes || plan.vectorLanes.size() != lanes ||
      plan.requiredEventCount != plan.events.size() ||
      plan.forwardEventCount != 3 * lanes ||
      plan.releaseEventCount !=
          plan.scoreLiveDepth + plan.productLiveDepth ||
      plan.reductionSteps.size() != lanes - 1 || !plan.affineTreeRequired ||
      plan.backend.vfMergeLevel != 1 ||
      !plan.backend.disableAutoBindSubBlock ||
      !plan.backend.enableGraphSync)
    return false;

  for (const PostCVSplitSlotAssignment &slot : plan.slots) {
    unsigned limit = plan.probabilitySlotCount;
    if (slot.role == PostCVSplitLineageRole::Score)
      limit = plan.scoreLiveDepth;
    else if (slot.role == PostCVSplitLineageRole::Product)
      limit = plan.productLiveDepth;
    if (slot.lane >= lanes || slot.slot >= limit)
      return false;
  }
  for (const PostCVSplitVectorLanePlan &lane : plan.vectorLanes) {
    uint64_t covered;
    if (lane.lane >= lanes || lane.rows != plan.recurrence.vectorRows ||
        lane.chunkWidth != kVectorChunkElements ||
        !lane.directNzPacking ||
        !checkedMul(lane.chunkWidth, lane.chunkCount, covered) ||
        covered != plan.recurrence.scoreWidth)
      return false;
  }

  llvm::DenseSet<unsigned> flags;
  for (const PostCVSplitEventPlan &event : plan.events)
    if (event.logicalFlagId < plan.firstLogicalFlagId ||
        event.logicalFlagId > plan.maximumLogicalFlagId ||
        !flags.insert(event.logicalFlagId).second)
      return false;
  return flags.size() == plan.requiredEventCount;
}

} // namespace

PostCVSplitSchedulePlan buildPostCVSplitSchedulePlan(
  const PostCVSplitRequestSet &requests,
  const CrossCoreResourcePlan &currentResources,
  const CrossCoreResourceLimits &limits) {
  PostCVSplitSchedulePlan plan;
  plan.firstLogicalFlagId = limits.firstAvailableFlagId;
  plan.maximumLogicalFlagId = kMaximumLogicalFlagId;
  if (requests.target.archFamily != CVSplitArchFamily::A5 ||
      requests.candidates.empty() || !currentResources.completeLaneCoverage ||
      !currentResources.anchorsComplete || !currentResources.ownershipResolved)
    return plan;

  const unsigned lanes = requests.candidates.front().logicalLaneCount;
  if (lanes < 2 || requests.transferRequests.size() != 3 * lanes ||
      requests.cubeRequests.size() != 2 * lanes ||
      requests.unsupportedVectorOperations != 0)
    return plan;
  for (const PostCVSplitCandidateSummary &candidate :
       requests.candidates)
    if (candidate.logicalLaneCount != lanes ||
        candidate.matrixLineages.size() != 2)
      return plan;

  const std::array<CVSplitTransferKind, 3> expectedKinds{
      CVSplitTransferKind::FixpipeDrain,
      CVSplitTransferKind::CopyAndLayoutConversion,
      CVSplitTransferKind::FixpipeDrain};
  for (unsigned lane = 0; lane < lanes; ++lane)
    for (unsigned phase = 0; phase < expectedKinds.size(); ++phase) {
      const CVSplitTransferRequest &request =
          requests.transferRequests[lane * expectedKinds.size() + phase];
      if (request.kind != expectedKinds[phase] ||
          !sameTransferRequest(request, requests.transferRequests[phase])) {
        plan.status = PostCVSplitSchedulePlanStatus::UnsupportedRecurrence;
        return plan;
      }
    }

  llvm::SmallVector<std::pair<const CVSplitCubeRequest *, unsigned>, 2>
      cubeLineages;
  for (const CVSplitCubeRequest &request : requests.cubeRequests) {
    auto lineage = llvm::find_if(cubeLineages, [&](const auto &candidate) {
      return sameCubeRequest(*candidate.first, request);
    });
    if (lineage == cubeLineages.end())
      cubeLineages.push_back({&request, 1});
    else
      ++lineage->second;
  }
  if (cubeLineages.size() != 2 ||
      llvm::any_of(cubeLineages,
                   [&](const auto &lineage) { return lineage.second != lanes; })) {
    plan.status = PostCVSplitSchedulePlanStatus::UnsupportedRecurrence;
    return plan;
  }
  const CVSplitCubeRequest *scoreCube = nullptr;
  const CVSplitCubeRequest *productCube = nullptr;
  for (const auto &lineage : cubeLineages) {
    const bool consumesUb = pathStartsInUb(lineage.first->lhsPath) ||
                            pathStartsInUb(lineage.first->rhsPath);
    const CVSplitCubeRequest **role = consumesUb ? &productCube : &scoreCube;
    if (*role) {
      plan.status = PostCVSplitSchedulePlanStatus::UnsupportedRecurrence;
      return plan;
    }
    *role = lineage.first;
  }
  if (!scoreCube || !productCube) {
    plan.status = PostCVSplitSchedulePlanStatus::UnsupportedRecurrence;
    return plan;
  }

  const CVSplitTransferRequest &scoreTransfer = requests.transferRequests[0];
  const CVSplitTransferRequest &probabilityTransfer =
      requests.transferRequests[1];
  const CVSplitTransferRequest &productTransfer = requests.transferRequests[2];
  if (scoreCube->kind != CVSplitMatrixKind::Matmul ||
      productCube->kind != CVSplitMatrixKind::Matmul ||
      !isFloatingInput(scoreCube->lhsType) ||
      !isFloatingInput(scoreCube->rhsType) ||
      !isFloatingInput(productCube->lhsType) ||
      !isFloatingInput(productCube->rhsType) ||
      scoreCube->accumulatorType != CVSplitElementType::F32 ||
      productCube->accumulatorType != CVSplitElementType::F32 ||
      scoreTransfer.source != CVSplitMemorySpace::L0C ||
      scoreTransfer.destination != CVSplitMemorySpace::UB ||
      productTransfer.source != CVSplitMemorySpace::L0C ||
      productTransfer.destination != CVSplitMemorySpace::UB ||
      probabilityTransfer.source != CVSplitMemorySpace::UB ||
      probabilityTransfer.destination != CVSplitMemorySpace::L1 ||
      probabilityTransfer.destinationLayout != CVSplitLayout::NZ ||
      !isFloatingInput(probabilityTransfer.elementType)) {
    plan.status = PostCVSplitSchedulePlanStatus::UnsupportedRecurrence;
    return plan;
  }

  std::array<unsigned,
             static_cast<unsigned>(CVSplitVectorOpClass::OtherCalibrated) + 1>
      operationCounts{};
  bool reductionGeometryFound = false;
  for (const PostCVSplitOwnedVectorRegionRequest &region :
       requests.vectorRegionRequests) {
    if (region.reductionRows != 0 || region.reductionWidth != 0)
      reductionGeometryFound |=
          region.reductionRows == scoreTransfer.rows &&
          region.reductionWidth == scoreTransfer.columns;
    for (const CVSplitVectorOpSummary &operation : region.operations)
      ++operationCounts[static_cast<unsigned>(operation.operationClass)];
  }
  auto has = [&](CVSplitVectorOpClass operationClass) {
    return operationCounts[static_cast<unsigned>(operationClass)] != 0;
  };
  plan.recurrence.logicalLaneCount = lanes;
  plan.recurrence.matrixLineageCount = 2;
  plan.recurrence.transferLineageCount = expectedKinds.size();
  plan.recurrence.vectorRows = scoreTransfer.rows;
  plan.recurrence.scoreWidth = scoreTransfer.columns;
  plan.recurrence.headDimension = productCube->n;
  plan.recurrence.hasMaximum = has(CVSplitVectorOpClass::Maximum);
  plan.recurrence.hasRowReduceMax = has(CVSplitVectorOpClass::RowReduceMax);
  plan.recurrence.hasExp = has(CVSplitVectorOpClass::Exp);
  plan.recurrence.hasRowReduceSum = has(CVSplitVectorOpClass::RowReduceSum);
  plan.recurrence.hasNormalizationArithmetic =
      has(CVSplitVectorOpClass::ElementwiseAdd) &&
      has(CVSplitVectorOpClass::ElementwiseSub) &&
      has(CVSplitVectorOpClass::ElementwiseMul);
  plan.recurrence.hasCast = has(CVSplitVectorOpClass::Cast);
  plan.recurrence.hasPermute = has(CVSplitVectorOpClass::Permute);
  if (!reductionGeometryFound || !plan.recurrence.hasMaximum ||
      !plan.recurrence.hasRowReduceMax || !plan.recurrence.hasExp ||
      !plan.recurrence.hasRowReduceSum ||
      !plan.recurrence.hasNormalizationArithmetic ||
      !plan.recurrence.hasCast) {
    plan.status = PostCVSplitSchedulePlanStatus::UnsupportedRecurrence;
    return plan;
  }

  if (scoreTransfer.rows == 0 || scoreTransfer.columns == 0 ||
      scoreTransfer.columns % kVectorChunkElements != 0 ||
      productCube->n == 0 || productTransfer.columns != productCube->n) {
    plan.status = PostCVSplitSchedulePlanStatus::UnsupportedGeometry;
    return plan;
  }

  plan.scoreLiveDepth = std::min(2u, lanes);
  plan.productLiveDepth = std::min(2u, lanes);
  plan.probabilitySlotCount = lanes;
  plan.scoreBytesPerSlot = scoreTransfer.bytes;
  plan.probabilityBytesPerSlot = probabilityTransfer.bytes;
  plan.productBytesPerSlot = productTransfer.bytes;
  uint64_t scoreBytes, productBytes;
  if (!checkedMul(plan.scoreLiveDepth, plan.scoreBytesPerSlot, scoreBytes) ||
      !checkedMul(plan.productLiveDepth, plan.productBytesPerSlot,
                  productBytes) ||
      !checkedAdd(scoreBytes, productBytes, plan.allocatedUbBytes) ||
      !checkedMul(plan.probabilitySlotCount, plan.probabilityBytesPerSlot,
                  plan.allocatedL1Bytes)) {
    plan.status = PostCVSplitSchedulePlanStatus::ArithmeticOverflow;
    return plan;
  }
  plan.incrementalUbBytes =
      plan.allocatedUbBytes > currentResources.baselineUbBytes
          ? plan.allocatedUbBytes - currentResources.baselineUbBytes
          : 0;

  for (unsigned lane = 0; lane < lanes; ++lane) {
    plan.slots.push_back({PostCVSplitLineageRole::Score, lane,
                          lane % plan.scoreLiveDepth});
    plan.slots.push_back(
        {PostCVSplitLineageRole::Probability, lane, lane});
    plan.slots.push_back({PostCVSplitLineageRole::Product, lane,
                          lane % plan.productLiveDepth});
    plan.vectorLanes.push_back(
        {lane, scoreTransfer.rows, kVectorChunkElements,
         scoreTransfer.columns / kVectorChunkElements, true});
  }

  unsigned nextFlag = plan.firstLogicalFlagId;
  auto addEvent = [&](PostCVSplitLineageRole role, PostCVSplitEventKind kind,
                      unsigned laneOrSlot, PrincipalResource signal,
                      PrincipalResource wait, bool loopCarried) {
    plan.events.push_back(
        {nextFlag++, role, kind, laneOrSlot, signal, wait, loopCarried});
  };
  for (PostCVSplitLineageRole role :
       {PostCVSplitLineageRole::Score,
        PostCVSplitLineageRole::Probability,
        PostCVSplitLineageRole::Product})
    for (unsigned lane = 0; lane < lanes; ++lane) {
      if (role == PostCVSplitLineageRole::Probability)
        addEvent(role, PostCVSplitEventKind::Forward, lane,
                 PrincipalResource::Mte3, PrincipalResource::Mte1, false);
      else
        addEvent(role, PostCVSplitEventKind::Forward, lane,
                 PrincipalResource::Fixpipe, PrincipalResource::Vector,
                 false);
    }
  for (unsigned slot = 0; slot < plan.scoreLiveDepth; ++slot)
    addEvent(PostCVSplitLineageRole::Score,
             PostCVSplitEventKind::Release, slot, PrincipalResource::Mte3,
             PrincipalResource::Fixpipe, true);
  for (unsigned slot = 0; slot < plan.productLiveDepth; ++slot)
    addEvent(PostCVSplitLineageRole::Product,
             PostCVSplitEventKind::Release, slot, PrincipalResource::Vector,
             PrincipalResource::Fixpipe, true);
  plan.forwardEventCount = 3 * lanes;
  plan.releaseEventCount = plan.scoreLiveDepth + plan.productLiveDepth;
  plan.requiredEventCount = plan.events.size();
  const bool flagOverflow = plan.events.empty() || nextFlag == 0 ||
                            nextFlag - 1 > plan.maximumLogicalFlagId;

  llvm::SmallVector<unsigned> activeValues;
  for (unsigned lane = 0; lane < lanes; ++lane)
    activeValues.push_back(lane);
  unsigned nextValue = lanes;
  unsigned level = 0;
  while (activeValues.size() > 1) {
    llvm::SmallVector<unsigned> nextLevel;
    for (unsigned index = 0; index < activeValues.size(); index += 2) {
      if (index + 1 == activeValues.size()) {
        nextLevel.push_back(activeValues[index]);
        continue;
      }
      plan.reductionSteps.push_back({level, activeValues[index],
                                     activeValues[index + 1], nextValue});
      nextLevel.push_back(nextValue++);
    }
    activeValues = std::move(nextLevel);
    ++level;
  }
  plan.reductionRootValue = activeValues.front();
  plan.affineTreeRequired = true;

  if (flagOverflow) {
    plan.status = PostCVSplitSchedulePlanStatus::FlagOverflow;
    return plan;
  }
  const bool ubWithinBudget =
      !limits.extraUbBudgetBytes ||
      plan.incrementalUbBytes <= *limits.extraUbBudgetBytes;
  const bool l1WithinBudget =
      !limits.l1BudgetBytes || plan.allocatedL1Bytes <= *limits.l1BudgetBytes;
  if (!ubWithinBudget || !l1WithinBudget) {
    plan.status = PostCVSplitSchedulePlanStatus::MemoryBudgetExceeded;
    return plan;
  }
  if (!verifyPlan(plan)) {
    plan.status = PostCVSplitSchedulePlanStatus::InvalidInput;
    return plan;
  }
  plan.status = limits.extraUbBudgetBytes && limits.l1BudgetBytes
                    ? PostCVSplitSchedulePlanStatus::ValidKnownCapacity
                    : PostCVSplitSchedulePlanStatus::ValidUnknownCapacity;
  plan.selectionEligible = false;
  return plan;
}

void logPostCVSplitSchedulePlan(const PostCVSplitSchedulePlan &plan) {
  LLVM_DEBUG({
    llvm::dbgs() << "[cv-split] post-split-plan status="
                 << statusName(plan.status)
                 << " lanes=" << plan.recurrence.logicalLaneCount
                 << " matrix-lineages="
                 << plan.recurrence.matrixLineageCount
                 << " transfer-lineages="
                 << plan.recurrence.transferLineageCount
                 << " selection-eligible="
                 << (plan.selectionEligible ? "yes" : "no")
                 << " mutation=no\n";
    llvm::dbgs() << "[cv-split] post-split-geometry rows="
                 << plan.recurrence.vectorRows
                 << " score-width=" << plan.recurrence.scoreWidth
                 << " head-dim=" << plan.recurrence.headDimension
                 << " chunk-width=" << kVectorChunkElements
                 << " chunks="
                 << (plan.vectorLanes.empty()
                         ? 0
                         : plan.vectorLanes.front().chunkCount)
                 << " direct-nz="
                 << (!plan.vectorLanes.empty() &&
                             plan.vectorLanes.front().directNzPacking
                         ? "yes"
                         : "no")
                 << "\n";
    llvm::dbgs() << "[cv-split] post-split-memory score-depth="
                 << plan.scoreLiveDepth
                 << " probability-slots=" << plan.probabilitySlotCount
                 << " product-depth=" << plan.productLiveDepth
                 << " score-bytes=" << plan.scoreBytesPerSlot
                 << " probability-bytes=" << plan.probabilityBytesPerSlot
                 << " product-bytes=" << plan.productBytesPerSlot
                 << " UB=" << plan.allocatedUbBytes
                 << " incremental-UB=" << plan.incrementalUbBytes
                 << " L1=" << plan.allocatedL1Bytes << "\n";
    llvm::dbgs() << "[cv-split] post-split-flags first="
                 << plan.firstLogicalFlagId
                 << " maximum=" << plan.maximumLogicalFlagId
                 << " forward=" << plan.forwardEventCount
                 << " release=" << plan.releaseEventCount
                 << " required=" << plan.requiredEventCount << "\n";
    llvm::dbgs() << "[cv-split] post-split-backend auto-bind=off graph-sync=on"
                    " vf-merge="
                 << plan.backend.vfMergeLevel << " attribute-emitted=no\n";
    for (const PostCVSplitSlotAssignment &slot : plan.slots)
      llvm::dbgs() << "[cv-split] post-split-slot role=" << roleName(slot.role)
                   << " lane=" << slot.lane << " slot=" << slot.slot
                   << "\n";
    for (const PostCVSplitEventPlan &event : plan.events)
      llvm::dbgs() << "[cv-split] post-split-event flag="
                   << event.logicalFlagId
                   << " role=" << roleName(event.role)
                   << " kind=" << eventName(event.kind)
                   << " lane-or-slot=" << event.laneOrSlot
                   << " signal=" << resourceName(event.signalingResource)
                   << " wait=" << resourceName(event.waitingResource)
                   << " loop-carried="
                   << (event.loopCarried ? "yes" : "no") << "\n";
    for (const PostCVSplitReductionStep &step : plan.reductionSteps)
      llvm::dbgs() << "[cv-split] post-split-reduction level=" << step.level
                   << " left=" << step.leftValue
                   << " right=" << step.rightValue
                   << " result=" << step.resultValue << "\n";
    llvm::dbgs() << "[cv-split] post-split-recurrence maximum="
                 << (plan.recurrence.hasMaximum ? "yes" : "no")
                 << " reduce-max="
                 << (plan.recurrence.hasRowReduceMax ? "yes" : "no")
                 << " exp=" << (plan.recurrence.hasExp ? "yes" : "no")
                 << " reduce-sum="
                 << (plan.recurrence.hasRowReduceSum ? "yes" : "no")
                 << " arithmetic="
                 << (plan.recurrence.hasNormalizationArithmetic ? "yes"
                                                                : "no")
                 << " cast=" << (plan.recurrence.hasCast ? "yes" : "no")
                 << " permute="
                 << (plan.recurrence.hasPermute ? "yes" : "no")
                 << " affine-tree="
                 << (plan.affineTreeRequired ? "yes" : "no") << "\n";
  });
}

} // namespace mlir::triton::cv_split
