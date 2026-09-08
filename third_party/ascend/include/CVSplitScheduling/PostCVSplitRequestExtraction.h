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

#ifndef TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_REQUEST_EXTRACTION_H
#define TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_REQUEST_EXTRACTION_H

#include "ascend/include/CVSplitScheduling/PostCVSplitRequestTypes.h"
#include "ascend/include/CVSplitScheduling/CrossCorePipelinePlan.h"
#include "ascend/include/CVSplitScheduling/CrossCoreResourcePlan.h"
#include "ascend/include/CVSplitScheduling/CrossCoreScheduleCandidate.h"
#include "ascend/include/CVSplitScheduling/classifyAllOps.h"

#include "mlir/IR/Block.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::triton::cv_split {

/// Owns the arrays referenced by a v4 VECTOR-region request.
struct PostCVSplitOwnedVectorRegionRequest {
  CVSplitTargetIdentity target;
  llvm::SmallVector<CVSplitVectorOpSummary> operations;
  llvm::SmallVector<CVSplitNumericDependency> dependencies;
  uint64_t externalBytesRead = 0;
  uint64_t externalBytesWritten = 0;
  uint64_t temporaryUbBytes = 0;
  uint32_t reductionRows = 0;
  uint32_t reductionWidth = 0;
  CVSplitLayout inputLayout = CVSplitLayout::ND;
  CVSplitLayout outputLayout = CVSplitLayout::ND;
  bool oneOutlinedRegion = false;

  CVSplitVectorRegionRequest getRequest() const;
};

struct PostCVSplitLineageSummary {
  unsigned phaseOrdinal = 0;
  int64_t originId = 0;
  unsigned inFlightLimit = 0;
  unsigned transferSlotCount = 0;
};

struct PostCVSplitCandidateSummary {
  unsigned candidateId = 0;
  unsigned logicalLaneCount = 0;
  unsigned waveWidth = 0;
  unsigned maximumLiveMatrixResultsPerLineage = 0;
  unsigned prefetchLimit = 0;
  llvm::SmallVector<PostCVSplitLineageSummary> matrixLineages;
};
/// Owned, read-only compiler facts consumed by schedule planning.
struct PostCVSplitRequestSet {
  CVSplitTargetIdentity target;
  llvm::SmallVector<CVSplitCubeRequest> cubeRequests;
  llvm::SmallVector<PostCVSplitOwnedVectorRegionRequest> vectorRegionRequests;
  llvm::SmallVector<CVSplitTransferRequest> transferRequests;
  llvm::SmallVector<CVSplitSynchronizationRequest> synchronizationRequests;
  llvm::SmallVector<PostCVSplitCandidateSummary> candidates;
  unsigned unsupportedVectorOperations = 0;
};

/// Extracts schedule facts without mutating IR.
FailureOr<PostCVSplitRequestSet>
extractPostCVSplitRequests(Block *body, const Classification &classification,
                         const CrossCorePipelinePlan &pipelinePlan,
                         const CrossCoreResourcePlan &resourcePlan,
                         const CrossCoreScheduleCandidateSet &candidateSet);

} // namespace mlir::triton::cv_split

#endif // TRITON_ASCEND_CV_SPLIT_SCHEDULING_POST_CV_SPLIT_REQUEST_EXTRACTION_H
