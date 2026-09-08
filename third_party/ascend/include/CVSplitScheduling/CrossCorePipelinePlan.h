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

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_PIPELINE_PLAN_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_PIPELINE_PLAN_H

#include "ascend/include/CVSplitScheduling/CVSplitTypes.h"
#include "ascend/include/CVSplitScheduling/classifyAllOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::triton::cv_split {

struct AccumulatorJoinRewriteResult;

enum class CrossCoreDirection { CubeToVector, VectorToCube };

enum class BoundaryMaterialization { Observed, PostUnfuseDpsJoin };

enum class PipelineMemorySpace { UB, L1 };

/// One physical-resource annotation in the unrolled candidate body.
struct PipelineResourceUse {
  Operation *operation;
  EngineType engine;
  PrincipalResource resource;
  unsigned order;
};

/// One value crossing between the two execution engines.
///
/// Stable identity for one logical boundary across scheduling and the
/// accumulator-join rewrite.
struct CrossCoreBoundaryKey {
  int64_t originId;
  unsigned lane;
  CrossCoreDirection direction;
  unsigned resultNumber;
};

/// One logical value crossing between the two execution engines.
///
/// Observed boundaries have concrete consumers and a last reader. A projected
/// boundary describes the consumer that the accumulator-join rewrite will
/// create; those anchors stay absent until the later binding phase. Operation and
/// value handles are non-owning.
struct CrossCoreBoundary {
  CrossCoreBoundaryKey key;
  BoundaryMaterialization materialization;
  Value value;
  Operation *producer;
  Operation *earliestPublishAnchor;
  llvm::SmallVector<Operation *> consumers;
  Operation *lastReader = nullptr;
  uint64_t footprintBytes;
  PipelineMemorySpace memorySpace;
  Type elementType;
  llvm::SmallVector<int64_t, 4> logicalShape;
  PrincipalResource publishResource;
  PrincipalResource consumeResource;
  unsigned producerOrder;
  unsigned earliestPublishOrder;
  std::optional<unsigned> lastReaderOrder;
};

/// Every unrolled clone of one original cross-core producer.
struct CrossCorePhaseLineage {
  int64_t originId;
  CrossCoreDirection direction;
  llvm::SmallVector<unsigned> boundaryIndices;
};

/// Read-only description built before scheduling or transfer emission.
struct CrossCorePipelinePlan {
  unsigned laneCount = 0;
  llvm::SmallVector<CrossCoreBoundary> boundaries;
  llvm::SmallVector<CrossCorePhaseLineage> lineages;
  llvm::SmallVector<PipelineResourceUse> resourceUses;
};

/// Builds the lane, boundary, lifetime, memory-footprint, and principal-resource
/// model without mutating body.
FailureOr<CrossCorePipelinePlan>
buildCrossCorePipelinePlan(Block *body, const Classification &classification);

/// Binds projected boundaries to real post-rewrite consumers and refreshes all
/// operation-order fields after dependency scheduling.
FailureOr<CrossCorePipelinePlan> bindCrossCorePipelinePlan(
    const CrossCorePipelinePlan &logicalPlan,
    const AccumulatorJoinRewriteResult &rewriteResult, Block *body,
    const Classification &classification);

/// Emits deterministic LLVM_DEBUG diagnostics for an analysis plan.
void logCrossCorePipelinePlan(const CrossCorePipelinePlan &plan);

void logMaterializedCrossCorePipelinePlan(const CrossCorePipelinePlan &plan);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_PIPELINE_PLAN_H
