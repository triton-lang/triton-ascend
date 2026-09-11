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

#ifndef TRITON_TO_GRAPH_RESOURCE_COST_MODEL_H
#define TRITON_TO_GRAPH_RESOURCE_COST_MODEL_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace ascend {
class HardwareConfig;
} // namespace ascend

namespace triton {
namespace cfg {

// Every unproven resource condition is a hard rejection.  Keep the reason
// machine-readable so callers can surface it in compiler diagnostics and
// operator acceptance reports without trying to parse free-form text.
enum class ResourceCostRejectReason : uint8_t {
  None,
  UnknownResource,
  DynamicShape,
  UnknownElementType,
  Overflow,
  UBOverflow,
  InvalidCandidate,
};

const char *getResourceCostRejectReasonName(ResourceCostRejectReason reason);

// A per-function snapshot shared by every mapping rule. It carries only the
// static UB/liveness facts that are available during compilation.
struct ResourceSnapshot {
  uint64_t ubCapacityBytes = 0;
  uint64_t reservedUBBytes = 0;
  unsigned ubSafetyPercent = 80;

  bool isKnown() const;
  std::optional<uint64_t> getSafeUBBudget() const;

  static ResourceSnapshot fromExplicit(uint64_t ubCapacityBytes,
                                       unsigned ubSafetyPercent = 80,
                                       uint64_t reservedUBBytes = 0);

  // Reads the vendored hardware-information interface when it is enabled by
  // the build. A missing UB field (or a build without that interface) returns
  // an unknown snapshot instead of guessing.
  static ResourceSnapshot fromHardwareConfig(
      const ascend::HardwareConfig &hardware, unsigned ubSafetyPercent = 80,
      uint64_t reservedUBBytes = 0);
};

struct LiveTensorInterval {
  Value value;
  uint64_t begin = 0;
  uint64_t end = 0;
  uint64_t bytes = 0;
};

struct LiveByteEstimate {
  bool known = false;
  bool overflow = false;
  ResourceCostRejectReason reason = ResourceCostRejectReason::UnknownResource;
  uint64_t peakLiveBytes = 0;
  llvm::SmallVector<LiveTensorInterval, 16> intervals;
};

// Returns static resident bytes for ranked tensor values.  Dynamic shapes and
// element types with no conservative byte width return std::nullopt.
std::optional<uint64_t> getStaticTensorBytes(Type type);

// Estimates peak resident tensor bytes in one dynamic execution of root.  The
// walk is region-aware: loop bodies are considered once (never multiplied by a
// trip/fusion extent), loop-carried block arguments alias their incoming
// carrier, and values captured by a loop stay live through its final body op.
LiveByteEstimate estimatePeakLiveBytes(Operation *root);

struct CandidatePlan {
  uint64_t tensorizeFactor = 1;
  uint64_t blockT = 1;
  uint64_t staticAxisFusionFactor = 1;
  std::string stableId;
  uint64_t sourceOrdinal = 0;
};

// Counters are before -> after.  ``logicalTasks`` is the exact logical grid
// product. ``actualPrograms`` is the launcher-visible physical Block Num,
// after legacy auto-map/persistent capping has been applied.  Keep them
// separate: a nonpersistent IAT transform may reduce logical work while
// increasing Block Num because it is no longer eligible for legacy auto-map.
struct CandidateCost {
  CandidatePlan plan;
  uint64_t logicalTasksBefore = 0;
  uint64_t logicalTasksAfter = 0;
  uint64_t actualProgramsBefore = 0;
  uint64_t actualProgramsAfter = 0;
  uint64_t physicalWavesBefore = 0;
  uint64_t physicalWavesAfter = 0;
  uint64_t launchesBefore = 0;
  uint64_t launchesAfter = 0;
  uint64_t gmReadBytesBefore = 0;
  uint64_t gmReadBytesAfter = 0;
  uint64_t gmWriteBytesBefore = 0;
  uint64_t gmWriteBytesAfter = 0;
  uint64_t storeCountBefore = 0;
  uint64_t storeCountAfter = 0;
  uint64_t addressCalculationsBefore = 0;
  uint64_t addressCalculationsAfter = 0;
  uint64_t baselinePeakLiveBytes = 0;
  uint64_t estimatedPeakLiveBytes = 0;
  // Persistent plans carry their final grid-stride work distribution
  // explicitly.  These values are derived from logical tasks and physical
  // programs when callers leave them zero, but recording them makes the
  // selected plan auditable in cache/candidate manifests.
  uint64_t persistentLoopTripsBefore = 0;
  uint64_t persistentLoopTripsAfter = 0;
  // Bytes of token-only values that must be redundantly processed by separate
  // head groups.  A fully packed head tile therefore has a smaller "after"
  // value even when its row count matches a partially packed alternative.
  uint64_t tokenOnlyRepeatedBytesBefore = 0;
  uint64_t tokenOnlyRepeatedBytesAfter = 0;
  uint64_t workPerProgramBefore = 0;
  uint64_t workPerProgramAfter = 0;
  bool hasPeakLiveBytes = false;
  bool hasDynamicShape = false;
  bool hasUnknownResource = false;
  bool legacyAutoMapBefore = false;
  bool legacyAutoMapAfter = false;
  bool persistent = false;
};

struct CostModelWeights {
  uint64_t logicalProgramReduction = 64;
  // Kept under its established spelling for source compatibility. It now
  // prices launcher-visible physical Block Num, never a logical grid count.
  uint64_t programReduction = 64;
  uint64_t launchReduction = 256;
  uint64_t gmByte = 1;
  uint64_t store = 16;
  uint64_t addressCalculation = 1;
  uint64_t liveByte = 1;
  // Increasing a persistent tile does not create more total work; price the
  // resulting loop trips instead of a linear block_t/work-item penalty.
  uint64_t persistentLoopTrip = 512;
  uint64_t tokenOnlyRepeatedByte = 1;
};

struct CandidateEvaluation {
  CandidateCost candidate;
  bool accepted = false;
  ResourceCostRejectReason reason = ResourceCostRejectReason::UnknownResource;
  int64_t benefitScore = 0;
  uint64_t safeUBBudgetBytes = 0;
  uint64_t requiredParallelPrograms = 0;
  uint64_t effectiveWorkPerProgram = 0;
  uint64_t physicalWavesBefore = 0;
  uint64_t physicalWavesAfter = 0;
  uint64_t persistentLoopTripsBefore = 0;
  uint64_t persistentLoopTripsAfter = 0;
  uint64_t tokenOnlyRepeatedBytesBefore = 0;
  uint64_t tokenOnlyRepeatedBytesAfter = 0;
  std::string remark;
};

CandidateEvaluation evaluateCandidateCost(const ResourceSnapshot &resources,
                                          const CandidateCost &candidate,
                                          const CostModelWeights &weights = {});

// Canonicalizes the requested factors before producing their Cartesian product,
// so caller container order cannot make candidate discovery nondeterministic.
llvm::SmallVector<CandidatePlan>
enumerateCandidatePlans(llvm::ArrayRef<unsigned> tensorizeFactors,
                        llvm::ArrayRef<unsigned> blockTFactors,
                        llvm::ArrayRef<unsigned> staticAxisFusionFactors,
                        llvm::StringRef stableIdPrefix = "resource");

// Accepted candidates come first, then descending score.  Ties use the factor
// tuple, stable id, and original ordinal, making selection reproducible.
void sortCandidateEvaluations(
    llvm::MutableArrayRef<CandidateEvaluation> values);

std::string formatCandidateRemark(const CandidateEvaluation &evaluation);
void emitCandidateRemark(Operation *anchor,
                         const CandidateEvaluation &evaluation);

// Stable profiler-calibration record schema.  The time fields are reserved for
// target-kernel profiler data (not pytest/process wall time) so future tuning
// can compare measured latency with the exact model inputs that selected a
// factor.
struct ProfilerCalibrationRecord {
  std::string kernelName;
  std::string profilerArtifact;
  uint64_t sampleCount = 0;
  uint64_t medianNanoseconds = 0;
  uint64_t p90Nanoseconds = 0;
  CandidateEvaluation evaluation;
};

std::string
formatProfilerCalibrationRecord(const ProfilerCalibrationRecord &record);

// Cached by GraphOptimizationContext for one immutable IR epoch.  Rules can
// use it directly or provide a candidate-specific peak estimate; absent peak
// estimates fall back to the current function estimate only when it is known.
class ResourceCostAnalysis {
public:
  ResourceCostAnalysis(Operation *root, ResourceSnapshot resources);

  const ResourceSnapshot &getResourceSnapshot() const { return resources; }
  const LiveByteEstimate &getLiveByteEstimate() const { return liveBytes; }
  CandidateEvaluation evaluate(const CandidateCost &candidate) const;

private:
  ResourceSnapshot resources;
  LiveByteEstimate liveBytes;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_RESOURCE_COST_MODEL_H
