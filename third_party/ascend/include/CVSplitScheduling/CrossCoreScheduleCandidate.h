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

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_SCHEDULE_CANDIDATE_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_SCHEDULE_CANDIDATE_H

#include "ascend/include/CVSplitScheduling/CrossCoreResourcePlan.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::triton::cv_split {

/// One matrix-publication lineage's structural scheduling limit.
///
/// `inFlightLimit` is independent of the logical lane count and the transfer
/// pool's physical slot count. Candidate construction records this distinction but does not
/// yet change either operation order or transfer allocation.
struct ScheduleMatrixLineageLimit {
  unsigned pipelineLineageIndex = 0;
  int64_t originId = 0;
  CrossCoreDirection direction = CrossCoreDirection::CubeToVector;
  unsigned phaseOrdinal = 0;
  unsigned inFlightLimit = 0;
  unsigned transferSlotCount = 0;
};

/// Immutable topology for one resource-aware scheduling experiment.
///
/// Structural candidates are diagnostic-only. A forced override may consume
/// one only after separate legality/resource checks; The cost model adds calibrated
/// profitability and automatic selection.
struct CrossCoreScheduleCandidate {
  unsigned candidateId = 0;
  unsigned logicalLaneCount = 0;
  unsigned waveWidth = 0;
  unsigned maximumLiveMatrixResultsPerLineage = 0;
  unsigned prefetchLimit = 0;
  llvm::SmallVector<ScheduleMatrixLineageLimit> matrixLineageLimits;
  bool diagnosticOnly = true;
  bool selectionEligible = false;
};

struct CrossCoreScheduleCandidateSet {
  unsigned logicalLaneCount = 0;
  unsigned matrixLineageCount = 0;
  llvm::SmallVector<CrossCoreScheduleCandidate> candidates;
};

/// Build the initial monotone depth family without mutating IR.
///
/// Matrix-publication lineages are CUBE-to-VECTOR lineages ordered by their
/// first producer. For at least two logical lanes, the initial depth alphabet
/// is {1, 2}; widened prefixes produce [1,1], [2,1], [2,2] for a two-lineage
/// topology. The implementation is generic in lane and lineage count.
FailureOr<CrossCoreScheduleCandidateSet>
buildCrossCoreScheduleCandidates(const CrossCorePipelinePlan &pipelinePlan,
                                 const CrossCoreResourcePlan &resourcePlan);

void logCrossCoreScheduleCandidates(
    const CrossCoreScheduleCandidateSet &candidateSet);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_CROSS_CORE_SCHEDULE_CANDIDATE_H
