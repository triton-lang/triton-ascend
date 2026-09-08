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

#include "ascend/include/CVSplitScheduling/CrossCorePipelinePlan.h"
#include "ascend/include/CVSplitScheduling/UnfusePVMatmuls.h"
#include "ascend/include/CVSplitScheduling/UnrollOrigin.h"
#include "ascend/include/CVSplitScheduling/VectorAccumulatorMatmul.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

static llvm::StringRef directionName(CrossCoreDirection direction) {
  return direction == CrossCoreDirection::CubeToVector ? "C2V" : "V2C";
}

static llvm::StringRef
materializationName(BoundaryMaterialization materialization) {
  return materialization == BoundaryMaterialization::Observed
             ? "observed"
             : "post-unfuse-dps-join";
}

static llvm::StringRef memorySpaceName(PipelineMemorySpace memorySpace) {
  return memorySpace == PipelineMemorySpace::UB ? "UB" : "L1";
}

static llvm::StringRef resourceName(PrincipalResource resource) {
  switch (resource) {
  case PrincipalResource::Matrix:
    return "M";
  case PrincipalResource::Fixpipe:
    return "FIX";
  case PrincipalResource::Vector:
    return "VECTOR";
  case PrincipalResource::Mte1:
    return "MTE1";
  case PrincipalResource::Mte2:
    return "MTE2";
  case PrincipalResource::Mte3:
    return "MTE3";
  case PrincipalResource::ScalarControl:
    return "SCALAR";
  }
  llvm_unreachable("unknown principal resource");
}

static PrincipalResource
inferPrincipalResource(Operation *op, EngineType engine) {
  if (isa<linalg::MatmulOp>(op))
    return PrincipalResource::Matrix;
  if (isa<hivm::FixpipeOp>(op))
    return PrincipalResource::Fixpipe;
  if (isa<hivm::CopyOp, memref::CopyOp>(op))
    return engine == EngineType::VECTOR ? PrincipalResource::Mte3
                                        : PrincipalResource::Mte2;
  return engine == EngineType::VECTOR ? PrincipalResource::Vector
                                      : PrincipalResource::ScalarControl;
}

static FailureOr<uint64_t>
getFootprintBytes(RankedTensorType tensorType,
                  CrossCoreDirection direction) {
  if (!tensorType.hasStaticShape() || tensorType.getRank() != 2)
    return failure();

  uint64_t elements = 1;
  for (int64_t dim : tensorType.getShape()) {
    if (dim < 0 ||
        (dim != 0 &&
         elements > std::numeric_limits<uint64_t>::max() /
                        static_cast<uint64_t>(dim)))
      return failure();
    elements *= static_cast<uint64_t>(dim);
  }

  if (direction == CrossCoreDirection::CubeToVector) {
    if (tensorType.getDimSize(0) % 2 != 0)
      return failure();
    elements /= 2;
  }

  const uint64_t elementBits = tensorType.getElementTypeBitWidth();
  if (elementBits != 0 &&
      elements > std::numeric_limits<uint64_t>::max() / elementBits)
    return failure();
  return (elements * elementBits + 7) / 8;
}

static SmallVector<Operation *>
collectCrossCoreConsumers(Value value, CrossCoreDirection direction,
                          Block *body,
                          const Classification &classification) {
  SmallVector<Operation *> consumers;
  for (Operation *user : value.getUsers()) {
    if (user->getBlock() != body || isa<scf::YieldOp>(user))
      continue;
    auto classIt = classification.find(user);
    if (classIt == classification.end())
      continue;

    if (direction == CrossCoreDirection::CubeToVector) {
      if (classIt->second == EngineType::VECTOR)
        consumers.push_back(user);
      continue;
    }

    auto matmul = dyn_cast<linalg::MatmulOp>(user);
    if (!matmul || classIt->second != EngineType::CUBE)
      continue;
    if (matmul->getOperand(0) == value || matmul->getOperand(1) == value)
      consumers.push_back(user);
  }

  llvm::sort(consumers, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return consumers;
}

} // namespace

FailureOr<CrossCorePipelinePlan>
buildCrossCorePipelinePlan(Block *body,
                           const Classification &classification) {
  if (!body)
    return failure();

  CrossCorePipelinePlan plan;
  DenseMap<Operation *, unsigned> operationOrder;
  unsigned nextOrder = 0;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(op))
      continue;
    operationOrder[&op] = nextOrder++;
    auto classIt = classification.find(&op);
    if (classIt == classification.end())
      return failure();
    plan.resourceUses.push_back(
        {&op, classIt->second,
         inferPrincipalResource(&op, classIt->second),
         operationOrder.lookup(&op)});
  }

  llvm::MapVector<int64_t, unsigned> lineageIndexByOrigin;
  DenseMap<int64_t, unsigned> nextLaneByOrigin;
  DenseMap<Operation *, unsigned> laneByProducer;

  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(op))
      continue;

    auto classIt = classification.find(&op);
    if (classIt == classification.end())
      return failure();

    std::optional<CrossCoreDirection> direction;
    if (classIt->second == EngineType::CUBE && isa<linalg::MatmulOp>(op))
      direction = CrossCoreDirection::CubeToVector;
    else if (classIt->second == EngineType::VECTOR)
      direction = CrossCoreDirection::VectorToCube;
    else
      continue;

    bool projectsAccumulatorJoin = false;
    if (auto matmul = dyn_cast<linalg::MatmulOp>(&op)) {
      FailureOr<bool> projected =
          isVectorAccumulatorMatmul(matmul, body, classification);
      if (failed(projected))
        return failure();
      projectsAccumulatorJoin = *projected;
    }

    for (auto [resultNumber, value] : llvm::enumerate(op.getResults())) {
      auto tensorType = dyn_cast<RankedTensorType>(value.getType());
      if (!tensorType)
        continue;

      SmallVector<Operation *> consumers;
      if (!projectsAccumulatorJoin)
        consumers = collectCrossCoreConsumers(value, *direction, body,
                                              classification);
      if (!projectsAccumulatorJoin && consumers.empty())
        continue;

      auto originAttr =
          op.getAttrOfType<IntegerAttr>(kUnrollOriginIdAttrName);
      if (!originAttr)
        return failure();
      const int64_t originId = originAttr.getInt();

      FailureOr<uint64_t> footprint =
          getFootprintBytes(tensorType, *direction);
      if (failed(footprint))
        return failure();

      auto laneIt = laneByProducer.find(&op);
      unsigned lane = 0;
      if (laneIt == laneByProducer.end()) {
        lane = nextLaneByOrigin[originId]++;
        laneByProducer[&op] = lane;
      } else {
        lane = laneIt->second;
      }

      Operation *lastReader =
          projectsAccumulatorJoin ? nullptr : consumers.back();
      std::optional<unsigned> lastReaderOrder;
      if (lastReader)
        lastReaderOrder = operationOrder.lookup(lastReader);
      const unsigned boundaryIndex = plan.boundaries.size();
      plan.boundaries.push_back(
          {{originId, lane, *direction,
            static_cast<unsigned>(resultNumber)},
           projectsAccumulatorJoin
               ? BoundaryMaterialization::PostUnfuseDpsJoin
               : BoundaryMaterialization::Observed,
           value,
           &op,
           &op,
           std::move(consumers),
           lastReader,
           *footprint,
           *direction == CrossCoreDirection::CubeToVector
               ? PipelineMemorySpace::UB
               : PipelineMemorySpace::L1,
           tensorType.getElementType(),
           SmallVector<int64_t, 4>(tensorType.getShape().begin(),
                                   tensorType.getShape().end()),
           *direction == CrossCoreDirection::CubeToVector
               ? PrincipalResource::Fixpipe
               : PrincipalResource::Mte3,
           *direction == CrossCoreDirection::CubeToVector
               ? PrincipalResource::Vector
               : PrincipalResource::Mte1,
           operationOrder.lookup(&op),
           operationOrder.lookup(&op),
           lastReaderOrder});

      auto lineageIt = lineageIndexByOrigin.find(originId);
      if (lineageIt == lineageIndexByOrigin.end()) {
        const unsigned lineageIndex = plan.lineages.size();
        lineageIndexByOrigin[originId] = lineageIndex;
        plan.lineages.push_back({originId, *direction, {boundaryIndex}});
      } else {
        CrossCorePhaseLineage &lineage = plan.lineages[lineageIt->second];
        if (lineage.direction != *direction)
          return failure();
        lineage.boundaryIndices.push_back(boundaryIndex);
      }

      plan.laneCount = std::max(plan.laneCount, lane + 1);
    }
  }

  if (plan.boundaries.empty() || plan.lineages.empty())
    return failure();
  return plan;
}

FailureOr<CrossCorePipelinePlan> bindCrossCorePipelinePlan(
    const CrossCorePipelinePlan &logicalPlan,
    const AccumulatorJoinRewriteResult &rewriteResult, Block *body,
    const Classification &classification) {
  if (!body)
    return failure();

  DenseMap<Operation *, Operation *> joinByProducer;
  DenseSet<Operation *> uniqueJoins;
  for (const AccumulatorJoinBinding &binding : rewriteResult.bindings) {
    if (!binding.matmulProducer || !binding.vectorJoin ||
        !joinByProducer.try_emplace(binding.matmulProducer, binding.vectorJoin)
             .second ||
        !uniqueJoins.insert(binding.vectorJoin).second)
      return failure();
  }

  DenseMap<Operation *, unsigned> operationOrder;
  unsigned nextOrder = 0;
  CrossCorePipelinePlan materialized = logicalPlan;
  materialized.resourceUses.clear();
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(op))
      continue;
    auto classIt = classification.find(&op);
    if (classIt == classification.end())
      return failure();
    operationOrder[&op] = nextOrder++;
    materialized.resourceUses.push_back(
        {&op, classIt->second,
         inferPrincipalResource(&op, classIt->second),
         operationOrder.lookup(&op)});
  }

  DenseSet<Operation *> usedBindings;
  for (CrossCoreBoundary &boundary : materialized.boundaries) {
    if (!boundary.producer || boundary.producer->getBlock() != body ||
        !boundary.earliestPublishAnchor ||
        boundary.earliestPublishAnchor->getBlock() != body ||
        boundary.key.resultNumber >= boundary.producer->getNumResults() ||
        boundary.producer->getResult(boundary.key.resultNumber) !=
            boundary.value)
      return failure();

    if (boundary.materialization ==
        BoundaryMaterialization::PostUnfuseDpsJoin) {
      auto joinIt = joinByProducer.find(boundary.producer);
      if (joinIt == joinByProducer.end())
        return failure();
      Operation *join = joinIt->second;
      if (!usedBindings.insert(boundary.producer).second ||
          join->getBlock() != body)
        return failure();
      auto classIt = classification.find(join);
      if (classIt == classification.end() ||
          classIt->second != EngineType::VECTOR ||
          !llvm::is_contained(join->getOperands(), boundary.value))
        return failure();

      SmallVector<Operation *> directConsumers;
      for (Operation *user : boundary.value.getUsers())
        if (user->getBlock() == body && !isa<scf::YieldOp>(user))
          directConsumers.push_back(user);
      if (directConsumers.size() != 1 || directConsumers.front() != join)
        return failure();
      boundary.consumers = {join};
      boundary.lastReader = join;
    } else {
      if (boundary.consumers.empty())
        return failure();
      for (Operation *consumer : boundary.consumers)
        if (!consumer || consumer->getBlock() != body ||
            !llvm::is_contained(consumer->getOperands(), boundary.value))
          return failure();
      llvm::sort(boundary.consumers, [&](Operation *lhs, Operation *rhs) {
        return operationOrder.lookup(lhs) < operationOrder.lookup(rhs);
      });
      boundary.lastReader = boundary.consumers.back();
    }

    if (!operationOrder.contains(boundary.producer) ||
        !operationOrder.contains(boundary.earliestPublishAnchor) ||
        !operationOrder.contains(boundary.lastReader))
      return failure();
    boundary.producerOrder = operationOrder.lookup(boundary.producer);
    boundary.earliestPublishOrder =
        operationOrder.lookup(boundary.earliestPublishAnchor);
    boundary.lastReaderOrder = operationOrder.lookup(boundary.lastReader);
  }

  if (usedBindings.size() != joinByProducer.size())
    return failure();
  return materialized;
}

static void logPipelinePlan(const CrossCorePipelinePlan &plan,
                            llvm::StringRef label) {
  LLVM_DEBUG({
    llvm::dbgs() << "[cv-split] " << label << " plan: lanes=" << plan.laneCount
                 << " boundaries=" << plan.boundaries.size()
                 << " lineages=" << plan.lineages.size() << "\n";

    for (const CrossCorePhaseLineage &lineage : plan.lineages) {
      const CrossCoreBoundary &first =
          plan.boundaries[lineage.boundaryIndices.front()];
      llvm::dbgs() << "[cv-split] " << label << " lineage "
                   << lineage.originId << ": "
                   << directionName(lineage.direction) << ", "
                   << lineage.boundaryIndices.size() << " lane(s), "
                   << first.footprintBytes << " bytes/boundary, "
                   << memorySpaceName(first.memorySpace)
                   << ", publish=" << resourceName(first.publishResource)
                   << ", consume=" << resourceName(first.consumeResource)
                   << "\n";
    }

    for (const CrossCoreBoundary &boundary : plan.boundaries) {
      llvm::dbgs() << "[cv-split] " << label << " boundary origin="
                   << boundary.key.originId
                   << " lane=" << boundary.key.lane << " "
                   << directionName(boundary.key.direction)
                   << " materialization="
                   << materializationName(boundary.materialization)
                   << " publish@" << boundary.earliestPublishOrder
                   << " last-read@";
      if (boundary.lastReaderOrder)
        llvm::dbgs() << *boundary.lastReaderOrder << " lifetime="
                     << (*boundary.lastReaderOrder -
                         boundary.earliestPublishOrder);
      else
        llvm::dbgs() << "pending lifetime=pending";
      llvm::dbgs() << "\n";
    }

    constexpr unsigned resourceCount =
        static_cast<unsigned>(PrincipalResource::ScalarControl) + 1;
    unsigned counts[resourceCount] = {};
    for (const PipelineResourceUse &use : plan.resourceUses)
      ++counts[static_cast<unsigned>(use.resource)];
    llvm::dbgs() << "[cv-split] " << label << " resources:";
    for (unsigned i = 0; i < resourceCount; ++i)
      llvm::dbgs() << " "
                   << resourceName(static_cast<PrincipalResource>(i)) << "="
                   << counts[i];
    llvm::dbgs() << "\n";
  });
}

void logCrossCorePipelinePlan(const CrossCorePipelinePlan &plan) {
  logPipelinePlan(plan, "analysis");
}

void logMaterializedCrossCorePipelinePlan(const CrossCorePipelinePlan &plan) {
  logPipelinePlan(plan, "materialized");
}

} // namespace mlir::triton::cv_split
