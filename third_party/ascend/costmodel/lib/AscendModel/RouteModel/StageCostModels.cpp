//===- StageCostModels.cpp - Per-stage analytical models -----------------===//

#include "AscendModel/RouteModel/StageCostModels.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static double iterations(const LogicalStage &stage) {
  return static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
}

static std::vector<std::string>
collectSourceLocations(const LogicalStage &stage) {
  std::vector<std::string> result;
  llvm::StringSet<> seen;
  for (Operation *operation : stage.operations) {
    std::string location;
    llvm::raw_string_ostream stream(location);
    operation->getLoc().print(stream);
    stream.flush();
    if (location.empty() || !seen.insert(location).second)
      continue;
    result.push_back(std::move(location));
  }
  return result;
}

static double controlBody(const StageResourceCycles &resources) {
  return resources.loopControl + resources.branchControl +
         resources.divergence + resources.synchronization;
}

static double serialBody(const StageResourceCycles &resources) {
  const double execution =
      resources.scalar + resources.load + resources.store + resources.atomic +
      resources.compute + resources.predicate + resources.shuffle +
      resources.dot + controlBody(resources) + resources.spill;
  // Issue is a shared front-end throughput bound, not an extra instruction
  // stream.  Adding it to execution double-counts every instruction.
  return std::max(execution, resources.issue);
}

static bool permitsSimdOverlap(const LogicalStage &stage) {
  return stage.scheduleKind == StageScheduleKind::IndependentPipelined &&
         stage.features.permitsSimdRoofline();
}

// Scalar white-box terms, already in the profile's SYS_CNT cycle domain.
static double mainScalarLoadCycles(double count,
                                   const StageModeProfile &profile) {
  const double k = std::max(1.0, count);
  const double perLine = k <= profile.mainScalarLoadExtraLineHighThreshold
                             ? profile.mainScalarLoadExtraLineLowCycles
                             : profile.mainScalarLoadExtraLineHighCycles;
  return profile.mainScalarLoadPrepCycles + profile.mainScalarLoadFillCycles +
         std::max(0.0, k - profile.mainScalarLoadOutstandingLines) * perLine +
         (k - 1.0) * profile.mainScalarLoadIssueCycles;
}

static double simtUniformLoadCycles(double count,
                                    const StageModeProfile &profile) {
  return profile.simtUniformLoadPrepCycles + profile.simtUniformLoadFillCycles +
         (std::max(1.0, count) - 1.0) *
             profile.simtUniformLoadDiffLineIssueCycles;
}

static double mte3StoreCycles(const StageModeProfile &profile) {
  return profile.mte3StorePrepCycles + profile.mte3StoreFillCycles;
}

static double simtUniformStoreCycles(const StageModeProfile &profile) {
  return profile.simtUniformStoreBaseCycles;
}

static StageResourceCycles
materializeControlFlow(const LogicalStage &stage, StageMode mode,
                       StageResourceCycles resources,
                       const StageControlFlowRates &rates) {
  resources.loopControl +=
      static_cast<double>(stage.features.loopBackedgeCount) *
      rates.loopBackedgeCycles;
  resources.branchControl +=
      static_cast<double>(stage.features.conditionalBranchCount) *
      rates.conditionalBranchCycles;
  resources.synchronization +=
      static_cast<double>(stage.features.synchronizationCount) *
      rates.synchronizationCycles;
  if (mode == StageMode::SIMT) {
    resources.divergence +=
        static_cast<double>(stage.features.divergentBranchCount) *
        (1.0 - stage.features.activeLaneRatio) *
        rates.divergentBranchPenaltyCycles;
  }
  return resources;
}

static StageResourceCycles mapWorkload(const LogicalStage &stage,
                                       const StageModeProfile &profile,
                                       StageMode mode) {
  StageResourceCycles resources;
  const StageWorkload &work = stage.workload;
  const bool simd = mode == StageMode::SIMD;
  resources.setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
  llvm::StringMap<double> describedElements;
  llvm::StringMap<double> describedVectorInstructions;
  double describedIssueElements = 0.0;
  double describedIssueInstructions = 0.0;
  if (simd) {
    for (const TensorOperationWorkload &tensor :
         work.tensorOperationWorkloads) {
      describedElements[tensor.operation] += tensor.logicalElements;
      describedIssueElements += tensor.logicalElements;
      const double segmentBits =
          static_cast<double>(tensor.contiguousElementsPerSegment) *
          static_cast<double>(tensor.elementBitWidth);
      const double vectorInstructions =
          tensor.segmentCount *
          std::ceil(segmentBits / static_cast<double>(profile.vectorWidthBits));
      describedVectorInstructions[tensor.operation] += vectorInstructions;
      describedIssueInstructions += vectorInstructions;
    }
  }
  for (const auto &[name, elements] : work.operationElements) {
    auto rate = profile.operationRates.find(name);
    if (rate == profile.operationRates.end() || rate->second.throughput <= 0.0)
      continue;
    double instructions = elements;
    if (simd) {
      const double described = describedElements.lookup(name);
      const double tolerance = 1e-9 * std::max(1.0, elements);
      instructions =
          std::abs(described - elements) <= tolerance
              ? describedVectorInstructions.lookup(name)
              : std::ceil(elements / static_cast<double>(profile.vectorWidth));
    }
    resources.compute +=
        instructions / rate->second.throughput * rate->second.factor;
  }
  resources.scalar += work.scalarOperations / profile.scalarOperationsPerCycle;
  const double directLoadBytes = work.loadBytes - work.indirectLoadBytes;
  const double directStoreBytes = work.storeBytes - work.indirectStoreBytes;
  const double directLoadInstructions =
      work.loadWarpInstructions - work.indirectLoadTransactions;
  const double directStoreInstructions =
      work.storeWarpInstructions - work.indirectStoreTransactions;
  if (simd) {
    resources.load = directLoadBytes / profile.loadBytesPerCycle;
    resources.store = directStoreBytes / profile.storeBytesPerCycle;
  } else {
    resources.load =
        directLoadInstructions / profile.loadWarpInstructionsPerCycle;
    resources.store =
        directStoreInstructions / profile.storeWarpInstructionsPerCycle;
  }
  resources.load +=
      work.indirectLoadTransactions / profile.indirectLoadTransactionsPerCycle;
  resources.store += work.indirectStoreTransactions /
                     profile.indirectStoreTransactionsPerCycle;
  // Preserve one uncovered loaded-index dependency latency per Stage
  // iteration, but charge it only when an actual indirect access exists.
  if (work.indirectLoadTransactions > 0.0)
    resources.load += profile.indirectDependencyLatencyCycles;
  else if (work.indirectStoreTransactions > 0.0)
    resources.store += profile.indirectDependencyLatencyCycles;

  for (const AtomicWorkload &atomic : work.atomicWorkloads) {
    auto rate = profile.atomicRates.find(atomic.profileKey());
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find("default");
    if (rate == profile.atomicRates.end())
      continue;
    const StageAtomicRate &atomicRate = rate->second;
    const double base =
        atomic.logicalOperationInstances * atomicRate.operationStartupCycles +
        atomic.logicalElements / atomicRate.logicalElementsPerCycle;
    const double contention =
        atomic.contentionUnknown ? atomicRate.unknownContentionMultiplier : 1.0;
    resources.atomic += base * contention;
    if (atomic.resultUsed)
      resources.atomic +=
          atomic.logicalOperationInstances * atomicRate.resultDependencyCycles;
  }
  if (work.scalarLoadCount > 0.0) {
    resources.load +=
        simd ? mainScalarLoadCycles(work.scalarLoadCount, profile)
             : simtUniformLoadCycles(work.scalarLoadCount, profile);
  }
  if (work.scalarStoreCount > 0.0) {
    resources.store +=
        simd ? mte3StoreCycles(profile) : simtUniformStoreCycles(profile);
  }
  double predicateInstructions = work.predicateElements;
  if (simd) {
    const double described = describedElements.lookup("predicate.cmp");
    const double tolerance = 1e-9 * std::max(1.0, work.predicateElements);
    predicateInstructions =
        std::abs(described - work.predicateElements) <= tolerance
            ? describedVectorInstructions.lookup("predicate.cmp")
            : std::ceil(work.predicateElements /
                        static_cast<double>(profile.vectorWidth));
  }
  resources.predicate =
      predicateInstructions / profile.predicateOperationsPerCycle;
  // The scan pool is recorded in the unit each mode is priced in.  SIMD prices
  // the 1D scans by their regime features (element count for extent <= 64,
  // Sklansky element-rounds above it - exactly the bucket fields below) and the
  // multi-dim part by its column-parallel count S = extent * ceil(columns /
  // BpE), BpE being the dtype's vector lane count (256 / element size; 64 for
  // f32); the shared workload holds that sum.  SIMT prices the multi-dim part
  // by the total element count N = extent * columns (per-thread lane work) and
  // the 1D part by its element count (its warp segment table is keyed on N), so
  // it swaps in the element-unit totals instead.  StagePartitioner records both
  // views; the refund terms below back out exactly the units charged here.
  double scanSteps = work.scanShuffleLaneSteps;
  if (!simd)
    scanSteps =
        work.scanShuffleLaneSteps1d + work.scanShuffleLaneStepsSimtMulti;
  // Tree reductions are priced by their element-rounds, the unit the SIMD
  // Sklansky network is shaped by (shuffleLaneSteps minus the scan pool).
  const double reduceSteps = work.shuffleLaneSteps - work.scanShuffleLaneSteps;
  resources.shuffle = (reduceSteps + scanSteps) / profile.shuffleLanesPerCycle;
  resources.scanShuffle = scanSteps / profile.shuffleLanesPerCycle;
  resources.scanShuffle1d =
      work.scanShuffleLaneSteps1d / profile.shuffleLanesPerCycle;
  resources.scanShuffle1dSmall =
      work.scanShuffleLaneSteps1dSmall / profile.shuffleLanesPerCycle;
  resources.scanShuffle1dMidWork =
      work.scanShuffleLaneSteps1dMidWork / profile.shuffleLanesPerCycle;
  resources.scanShuffle1dTiledWork =
      work.scanShuffleLaneSteps1dTiledWork / profile.shuffleLanesPerCycle;
  resources.scanShuffleMultiTail =
      work.scanShuffleLaneStepsMultiTail / profile.shuffleLanesPerCycle;
  // Non-leading-axis multi-dim scans pay two transposes (Cumsum.cpp).  Each
  // body has its own fixed part plus byte rate; the fixed part is charged once
  // per present transpose kind, mirroring the per-stage startup above.  A
  // profile with no transpose rate charges nothing and keeps its previous
  // price.
  auto transposeCycles = [](double bytes, double fixed, double rate) {
    return rate > 0.0 ? fixed + bytes / rate : 0.0;
  };
  const double rank2Bytes = work.scanTransposeBytesRank2;
  const double rank3Bytes = work.scanTransposeBytes - rank2Bytes;
  if (rank3Bytes > 0.0)
    resources.scanTranspose +=
        transposeCycles(rank3Bytes, profile.prefixScanTransposeDim01FixedCycles,
                        profile.prefixScanTransposeDim01BytesPerCycle);
  if (rank2Bytes > 0.0)
    resources.scanTranspose +=
        transposeCycles(rank2Bytes, profile.prefixScanTransposeAr2raFixedCycles,
                        profile.prefixScanTransposeAr2raBytesPerCycle);
  // Fixed per-scan-execution cost (thread barrier, fixed launch shape, UB
  // round-trip, library-call setup), charged once per present scan segment.
  // The two 1D regimes each carry their own: the scalar-register segment
  // (extent <= 64) and the rvec segment (extent > 64), the latter shared by
  // both rvec buckets through the block-invocation setup.  The multi-dim
  // segment has its own too: SIMT pays a barrier and a UB round-trip, and the
  // SIMD 2D library template (Cumsum.cpp, column parallel) pays a fixed
  // issue/setup cost.  Profiles that provide no such value keep 0 and the
  // segments stay proportional.
  if (work.scanShuffleLaneSteps1dSmall > 0.0)
    resources.scanStartup += profile.prefixScanStartupCycles1dSmall;
  if (work.scanShuffleLaneSteps1dMidWork +
          work.scanShuffleLaneSteps1dTiledWork >
      0.0)
    resources.scanStartup += profile.prefixScanStartupCycles1d;
  if (work.scanShuffleLaneStepsSimtMulti > 0.0)
    resources.scanStartup += profile.prefixScanStartupCycles;
  if (work.dotFlops > 0.0) {
    resources.setup += profile.dotSetupCycles;
    resources.dot = work.dotFlops / profile.dotFlopsPerCycle;
  }
  double issueInstructions =
      std::ceil(work.issueElements / static_cast<double>(profile.issueWidth));
  if (simd) {
    // Keep the issue floor consistent with operation pricing.  Aggregating
    // unrelated short rows before dividing by the vector width makes several
    // independently issued instructions look like one full-width operation.
    const double undescribedIssueElements =
        std::max(0.0, work.issueElements - describedIssueElements);
    issueInstructions = describedIssueInstructions +
                        std::ceil(undescribedIssueElements /
                                  static_cast<double>(profile.issueWidth));
  }
  resources.issue = issueInstructions / profile.issueOperationsPerCycle;
  resources.spill =
      work.estimatedSpillTransactions / profile.spillTransactionsPerCycle;
  if (stage.features.hasLoopCarriedDataDependency)
    resources.criticalPath = resources.scalar + resources.compute +
                             resources.predicate + resources.shuffle +
                             resources.dot;
  else if (stage.features.hasReduction)
    resources.criticalPath =
        resources.compute + resources.predicate + resources.shuffle;
  return materializeControlFlow(stage, mode, resources, profile.controlFlow);
}

// v3.1 warp-shape-aware SIMT 1D scan cost.  The ScanOpToLLVM lowering has
// three structural regimes (single-warp fast path / cross-warp Sklansky merge
// / general path) keyed off N and the kernel warp count; the segment cost
// replaces the legacy factor * laneSteps pricing for the whole 1D pool,
// including its startups.
static double simtPrefixScan1dSegmentCost(const StageModeProfile &simt,
                                          int64_t numWarps, double n) {
  const auto &segments = simt.prefixScan1dWarpSegments;
  auto it = segments.lower_bound(
      static_cast<unsigned>(std::max<int64_t>(1, numWarps)));
  if (it == segments.end())
    it = std::prev(segments.end());
  const auto &segment = it->second;
  const double warps = static_cast<double>(it->first);
  const double threads = 32.0 * warps;
  if (n <= 32.0)
    return segment.cLocal;
  // warpsPerCTA[axis] = ceil(N/32), capped by the kernel warp count.
  const double chainWarps = std::min(warps, std::ceil(n / 32.0));
  // Cross-warp Sklansky merge.  Both gates mirror canUseMultiWarpSklansky:
  // one element per thread (N <= 32*w) and at most warpSize axis warps
  // (k <= 32, i.e. N <= 1024).  A kernel with more warps than that
  // (num_warps = 64) drops to the general path already at N = 2048 although
  // every thread still owns one element, so pricing it as Sklansky would
  // under-predict it.  ceil(log2 k) is the trip count of the Sklansky
  // `for (h = 1; h < k; h <<= 1)` loop, so the pricing tracks the real round
  // count instead of a linear proxy.
  if (n <= threads && chainWarps <= 32.0) {
    const double rounds = std::ceil(std::log2(std::max(1.0, chainWarps)));
    return segment.cFixed + segment.cRound * rounds;
  }
  // General path.  The per-element serial slope is charged from N = 0; the
  // 32*w anchor of the (N - 32*w) form is folded into cGen, so cGen is not
  // directly comparable with the cross-warp segment constant.
  return segment.cGen + segment.serialRate * n;
}

// v3.2 warp-shape-aware SIMT multi-dim (2D) scan cost.  The generic scan
// lowering (AddPartialReduce) distributes the scan across warps with a serial
// UB load chain whose cost is additive in the total element count N, plus a
// per-thread-element hinge that prices register pressure beyond r0.  E is the
// aggregate scan extent (exact for a single multi-dim scan per Stage) and
// determines k = warpsPerCTA[axis], the UB chain length.  Extents below one
// warp (E < 32) take the separate linear regime described inline below.
static double simtPrefixScan2dSegmentCost(const StageModeProfile &simt,
                                          int64_t numWarps, double extentSum,
                                          double n) {
  const auto &segments = simt.prefixScan2dWarpSegments;
  auto it = segments.lower_bound(
      static_cast<unsigned>(std::max<int64_t>(1, numWarps)));
  if (it == segments.end())
    it = std::prev(segments.end());
  const auto &segment = it->second;
  const double extent = std::max(1.0, extentSum);
  // Sub-warp extent: the warp owning the axis holds fewer than 32 valid lanes,
  // so the axis warp never needs a cross-warp chain (k = 1) and the cost
  // collapses onto a single linear-in-N regime.  Layout-coalesced small scans
  // (e.g. a 16-element cumsum promoted to an (8,16,16) scan) land here.
  if (extent < 32.0 && (segment.aSubwarp > 0.0 || segment.bSubwarp > 0.0))
    return segment.aSubwarp + segment.bSubwarp * n;
  const double threads = 32.0 * static_cast<double>(it->first);
  // warpsPerCTA[axis]: warps holding unique data along the scan axis.
  const double chainWarps =
      std::min(static_cast<double>(it->first), std::ceil(extent / 32.0));
  const double chainLoads = n * std::max(1.0, chainWarps) / threads;
  const double hinge = std::max(0.0, n / threads - segment.r0);
  return segment.a + segment.b * n + segment.c * chainLoads + segment.d * hinge;
}

static double applySuperBlock(const LogicalStage &stage,
                              const StageResourceCycles &resources,
                              const StageImplementation &implementation,
                              const HardwareProfile &profile,
                              double stageCycles) {
  if (implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor == 1)
    return stageCycles;

  const double factor = static_cast<double>(implementation.superblockFactor);
  const double effectiveFactor = std::min(
      factor, static_cast<double>(profile.superblockUsefulFactorLimit));
  const double latencySensitivePerIteration =
      resources.load + resources.store + resources.atomic + resources.shuffle +
      resources.divergence;
  const double latencySensitive =
      iterations(stage) * latencySensitivePerIteration;
  // SuperBlock creates `factor` independent logical-program groups on one
  // physical core.  It can hide latency across those groups, but it cannot
  // divide dependent arithmetic, loop control, or synchronization.
  const double pressure =
      iterations(stage) * resources.spill * std::max(0.0, factor - 1.0);
  // Live-out bytes alone do not prove register pressure: they describe the
  // Stage ABI, not the allocator's simultaneously-live set.  Charge replicated
  // persistent state only when workload analysis has independently predicted
  // spill traffic.  This keeps the penalty evidence based and lets independent
  // recurrence groups use F4 when the generated SIMT VF has no STK/LDK.
  const double persistentStatePressure =
      stage.features.hasLoopCarriedDataDependency && resources.spill > 0.0
          ? std::max(
                0.0,
                factor -
                    static_cast<double>(
                        profile.superblockPersistentStatePressureFreeFactor)) *
                static_cast<double>(stage.liveOutBytes) /
                profile.superblockPersistentStateBytesPerCycle
          : 0.0;
  const double fixed = resources.setup;
  const double issueFloor =
      fixed + factor * iterations(stage) * resources.issue;
  // A recurrence is serial inside one logical program.  SuperBlock contributes
  // F independent logical programs to the same physical program, allowing the
  // scheduler to cover one program's dependency stalls with another program.
  // Normalize the critical-path portion per logical program, but retain the
  // aggregate issue floor: a larger factor cannot create additional issue
  // bandwidth.
  // This applies equally to whole-kernel and scope-local SuperBlock because
  // both materializers batch complete logical programs around the Stage.
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence) {
    const double recurrenceBody = std::max(0.0, stageCycles - fixed);
    return std::max(issueFloor, fixed + recurrenceBody + pressure) +
           persistentStatePressure;
  }
  // Proven persistent-state pressure is additional register/stack work and
  // cannot disappear behind the ordinary issue floor.
  const double body = std::max(0.0, stageCycles - fixed);
  const double groupedBody = factor * std::max(0.0, body - latencySensitive) +
                             factor * latencySensitive / effectiveFactor;
  return std::max(issueFloor, fixed + groupedBody + pressure) +
         persistentStatePressure;
}

static double estimateStage(const LogicalStage &stage,
                            const HardwareProfile &profile, StageMode mode,
                            StageResourceCycles &r) {
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
  // Per-pipe split for diagnostics (see StageResourceCycles::pipe*): load/
  // store/atomic execute on MTE, scalar/control/spill on SI, and compute/
  // predicate/shuffle/dot on VEC.  A loop-carried recurrence is one dependency
  // chain whose body cannot be split across pipes, so its pipe terms are
  // cleared and its body is reported as pipeSerial instead.
  r.pipeMte = r.load + r.store + r.atomic;
  r.pipeSi = r.scalar + controlBody(r) + r.spill;
  r.pipeVec = r.compute + r.predicate + r.shuffle + r.dot;
  r.pipeSerial = 0.0;
  switch (stage.costModelKind) {
  case StageCostModelKind::AutoBlockifyDispatch:
  case StageCostModelKind::AutoBlockifyLoop: {
    const double dispatchCount =
        stage.costModelKind == StageCostModelKind::AutoBlockifyLoop ? count
                                                                    : 1.0;
    return r.setup +
           dispatchCount * std::max(r.scalar + controlBody(r), r.issue);
  }
  case StageCostModelKind::ContinuousTileMemory:
  case StageCostModelKind::ContinuousTileStore:
  case StageCostModelKind::ContinuousShortLoad:
  case StageCostModelKind::CachePolicyStore:
  case StageCostModelKind::AtomicMemory:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup +
             count * (r.scalar + r.predicate + controlBody(r) + r.spill +
                      std::max({r.load, r.store, r.atomic, r.issue}));
    return serial;
  case StageCostModelKind::IndependentPipelinedLoop:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup +
             count *
                 (std::max({r.load, r.store, r.atomic,
                            r.compute + r.dot + r.shuffle,
                            r.scalar + r.predicate + controlBody(r), r.issue}) +
                  r.spill);
    return serial;
  case StageCostModelKind::LoopCarriedRecurrence: {
    // A prefix scan nested inside a recurrence keeps its lane dependency
    // chain: each scan level must complete before the next starts, so the
    // scan's shuffle traffic cannot reach the ideal vector throughput.
    // Scale only the tt.scan-contributed portion of the shuffle critical
    // path with the same mode-specific dependency factor the standalone
    // PrefixScan model uses (identity for SIMT).  tt.reduce-contributed
    // shuffle keeps the ideal rate: tree reductions halve their active
    // lanes per level, so the N*log2(N) lane-step billing already carries
    // enough slack to absorb per-level inefficiency.
    double critical = r.criticalPath > 0.0
                          ? std::max(r.criticalPath + r.load + r.store +
                                         r.atomic + controlBody(r) + r.spill,
                                     r.issue)
                          : serialBody(r);
    if (r.criticalPath > 0.0 && stage.features.hasPrefixScan) {
      const double dependencyFactor =
          mode == StageMode::SIMD ? profile.simd.prefixScanDependencyFactor
                                  : profile.simt.prefixScanDependencyFactor;
      const double dependencyFactor1d =
          mode == StageMode::SIMD ? profile.simd.prefixScanDependencyFactor1d
                                  : profile.simt.prefixScanDependencyFactor1d;
      const double dependencyFactor1dSmall =
          mode == StageMode::SIMD
              ? profile.simd.prefixScanDependencyFactor1dSmall
              : profile.simt.prefixScanDependencyFactor1dSmall;
      // Back out the 1D portion of the base charge in the units that base
      // charge used.  SIMD stores the 1D contribution to r.scanShuffle in the
      // per-regime features StagePartitioner recorded - element count for the
      // extent <= 64 scalar-register segment, Sklansky element-rounds
      // (N * log2(N), shared by the two rvec segments) above it - so it refunds
      // the same three buckets.  SIMT's scan total was swapped for the
      // element-unit 1D + multi-dim counts, so it keeps the element-unit
      // refund.
      double oneDimRefund = 0.0;
      double multiTailPremium = 0.0;
      if (mode == StageMode::SIMD) {
        oneDimRefund = (dependencyFactor - dependencyFactor1dSmall) *
                           r.scanShuffle1dSmall +
                       (dependencyFactor -
                        profile.simd.prefixScanDependencyFactor1dMidWork) *
                           r.scanShuffle1dMidWork +
                       (dependencyFactor -
                        profile.simd.prefixScanDependencyFactor1dTiledWork) *
                           r.scanShuffle1dTiledWork;
        // Same multi-dim hinge as the standalone PrefixScan model.
        multiTailPremium = r.scanShuffleMultiTail *
                           profile.simd.prefixScanDependencyFactorMultiTail;
      } else {
        oneDimRefund =
            (dependencyFactor - dependencyFactor1d) *
                (r.scanShuffle1d - r.scanShuffle1dSmall) +
            (dependencyFactor - dependencyFactor1dSmall) * r.scanShuffle1dSmall;
      }
      critical =
          std::max(r.criticalPath + (dependencyFactor - 1.0) * r.scanShuffle +
                       r.scanStartup + r.scanTranspose - oneDimRefund +
                       multiTailPremium + r.load + r.store + r.atomic +
                       controlBody(r) + r.spill,
                   r.issue);
    }
    if (mode == StageMode::SIMD) {
      // A loop-carried tensor is not ordinary embarrassingly-parallel vector
      // work: the updated state must remain live until the next recurrence
      // step.  The operation-throughput terms above account for arithmetic,
      // but not this persistent register/stack traffic.  Charge the exact
      // SSA live-out footprint once per Stage invocation using the target
      // profile's persistent-state byte rate.
      const double persistentState =
          static_cast<double>(stage.liveOutBytes) /
          profile.superblockPersistentStateBytesPerCycle;
      r.pipeSerial = critical;
      r.pipeVec = 0.0;
      r.pipeMte = 0.0;
      r.pipeSi = 0.0;
      return r.setup + count * critical + persistentState;
    }
    const int64_t groups = std::max<int64_t>(
        1, std::min(stage.features.parallelRecurrenceGroupCount,
                    profile.logicalWarpGroupCount));
    r.pipeSerial = critical;
    r.pipeVec = 0.0;
    r.pipeMte = 0.0;
    r.pipeSi = 0.0;
    return r.setup +
           std::max(std::ceil(count / static_cast<double>(groups)) * critical,
                    count * r.issue);
  }
  case StageCostModelKind::RowwiseReduction:
    return r.setup +
           count * std::max(r.scalar + r.load + r.store + r.atomic +
                                r.criticalPath + controlBody(r) + r.spill,
                            r.issue);
  case StageCostModelKind::PrefixScan: {
    const double dependencyFactor =
        mode == StageMode::SIMD ? profile.simd.prefixScanDependencyFactor
                                : profile.simt.prefixScanDependencyFactor;
    const double dependencyFactor1d =
        mode == StageMode::SIMD ? profile.simd.prefixScanDependencyFactor1d
                                : profile.simt.prefixScanDependencyFactor1d;
    const double dependencyFactor1dSmall =
        mode == StageMode::SIMD
            ? profile.simd.prefixScanDependencyFactor1dSmall
            : profile.simt.prefixScanDependencyFactor1dSmall;
    // Charge the multi-dim factor on all scan shuffle, then refund the 1D
    // portion at the regime factors.  Each mode backs out exactly the cost unit
    // its base charge used:
    //   SIMD - the per-regime features StagePartitioner recorded (element count
    //          for the extent <= 64 scalar-register regime, Sklansky
    //          element-rounds for the two rvec regimes), because the 1D
    //          contribution to the scan pool is stored in those units;
    //   SIMT - the element count, because mapWorkload swapped the scan total
    //          for the element-unit 1D + multi-dim counts.  Its 1D price is
    //          then replaced wholesale by the warp segment table below, so the
    //          refund only has to cancel this same expression.
    const double scanShuffle1dLarge = r.scanShuffle1d - r.scanShuffle1dSmall;
    double scanCritical = r.compute + r.predicate +
                          r.shuffle * dependencyFactor + r.scanStartup +
                          r.scanTranspose;
    if (mode == StageMode::SIMD) {
      scanCritical -=
          r.scanShuffle1dSmall * (dependencyFactor - dependencyFactor1dSmall) +
          r.scanShuffle1dMidWork *
              (dependencyFactor -
               profile.simd.prefixScanDependencyFactor1dMidWork) +
          r.scanShuffle1dTiledWork *
              (dependencyFactor -
               profile.simd.prefixScanDependencyFactor1dTiledWork);
      // Multi-dim hinge: the column-parallel steps above S = 256 cost more per
      // step than the ones below it, so the shared base factor above is not
      // enough for them.  They are already part of r.scanShuffle at
      // dependencyFactor, so only the extra factor is added here.  SIMT has
      // no such term: its multi-dim price is replaced by the warp segment
      // table below.
      scanCritical += r.scanShuffleMultiTail *
                      profile.simd.prefixScanDependencyFactorMultiTail;
    } else {
      scanCritical -=
          scanShuffle1dLarge * (dependencyFactor - dependencyFactor1d) +
          r.scanShuffle1dSmall * (dependencyFactor - dependencyFactor1dSmall);
    }
    if (mode == StageMode::SIMT && r.scanShuffle1d > 0.0 &&
        !profile.simt.prefixScan1dWarpSegments.empty()) {
      // v3.1: the SIMT 1D scan cost is set by the ScanOpToLLVM lowering's
      // structural regime (single-warp fast path / cross-warp Sklansky /
      // general path), keyed off N and the kernel warp count.  Replace the
      // legacy 1D factor pricing and its startups with the segment cost;
      // multi-dim and non-scan shuffle keep the factor formula.  N is the 1D
      // scan element count (S_1d = N, and
      // scanShuffle1d = N / shuffleLanesPerCycle).
      const double n = r.scanShuffle1d * profile.simt.shuffleLanesPerCycle;
      const double segmentCost = simtPrefixScan1dSegmentCost(
          profile.simt, profile.logicalWarpGroupCount, n);
      const double legacy1d = scanShuffle1dLarge * dependencyFactor1d +
                              r.scanShuffle1dSmall * dependencyFactor1dSmall;
      // Match mapWorkload's startup conditions exactly (scalar-register bucket
      // present; either rvec bucket present) so the replacement cancels the
      // charge it is replacing.
      const double legacy1dStartup =
          (r.scanShuffle1dSmall > 0.0
               ? profile.simt.prefixScanStartupCycles1dSmall
               : 0.0) +
          (r.scanShuffle1dMidWork + r.scanShuffle1dTiledWork > 0.0
               ? profile.simt.prefixScanStartupCycles1d
               : 0.0);
      scanCritical += segmentCost - legacy1d - legacy1dStartup;
    }
    if (mode == StageMode::SIMT &&
        stage.workload.scanShuffleLaneStepsSimtMulti > 0.0 &&
        !profile.simt.prefixScan2dWarpSegments.empty()) {
      // v3.2: the SIMT multi-dim scan cost is a warp-shape-aware additive
      // formula in N (total elements) and k (UB chain length), with a
      // per-thread-element hinge (register pressure).  Replace the legacy
      // multi-dim factor pricing and its startup; the 1D pool and non-scan
      // shuffle keep their formulas.  The multi-dim total N is the
      // per-iteration workload count (the same value mapWorkload swapped
      // into r.scanShuffle), and the legacy price is that count's share of
      // r.scanShuffle at the factor rate.
      const double lanes = profile.simt.shuffleLanesPerCycle;
      const double multiN = stage.workload.scanShuffleLaneStepsSimtMulti;
      const double segmentCost = simtPrefixScan2dSegmentCost(
          profile.simt, profile.logicalWarpGroupCount,
          stage.workload.scanSimtMultiExtentSum, multiN);
      const double legacyMulti = multiN / lanes * dependencyFactor;
      const double legacyMultiStartup = profile.simt.prefixScanStartupCycles;
      scanCritical += segmentCost - legacyMulti - legacyMultiStartup;
    }
    // The base charge records the un-amplified scan step count; the price the
    // model actually charges multiplies it by the dependency factor and, for
    // SIMT, replaces the 1D and multi-dim pools with the warp segment tables.
    // Report the priced value as the scan's VEC load so the dumped resources
    // match the charged cost and route aggregation sees the scan's real VEC
    // occupancy instead of the un-amplified base.
    r.pipeVec = scanCritical;
    r.pipeMte = r.load + r.store + r.atomic;
    r.pipeSi = r.scalar + controlBody(r) + r.spill;
    return r.setup +
           count * std::max(r.scalar + r.load + r.store + r.atomic +
                                scanCritical + controlBody(r) + r.spill,
                            r.issue);
  }
  case StageCostModelKind::CubeRoofline:
  case StageCostModelKind::TinyCubeRoofline:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup + count * (r.scalar + r.predicate + controlBody(r) +
                                r.shuffle + r.spill +
                                std::max({r.load, r.compute + r.dot, r.store,
                                          r.atomic, r.issue}));
    return serial;
  case StageCostModelKind::ConversionPack:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup + count * (r.predicate + controlBody(r) + r.spill +
                                std::max({r.scalar + r.compute, r.load, r.store,
                                          r.atomic, r.issue}));
    return serial;
  default:
    if (mode == StageMode::SIMD)
      return r.setup +
             count *
                 (std::max({r.load, r.store, r.atomic,
                            r.compute + r.dot + r.shuffle,
                            r.scalar + r.predicate + controlBody(r), r.issue}) +
                  r.spill);
    return serial;
  }
}
static bool isDeclaredLegal(const LogicalStage &stage,
                            const StageImplementation &implementation) {
  if (!implementation.isValid())
    return false;
  if (implementation.mode == StageMode::SIMD)
    return stage.simdLegal && implementation.superblockFactor == 1 &&
           !implementation.localScope;
  if (!stage.simtLegal)
    return false;
  if (implementation.localScope)
    return stage.localSimtMaterializable &&
           llvm::is_contained(stage.localSimtFactors,
                              implementation.superblockFactor);
  return llvm::is_contained(stage.legalSimtFactors,
                            implementation.superblockFactor);
}

} // namespace

llvm::StringRef mlir::ascend::stringifyStageCostModel(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::AutoBlockifyDispatch:
    return "auto_blockify_dispatch";
  case StageCostModelKind::AutoBlockifyLoop:
    return "auto_blockify_loop";
  case StageCostModelKind::ScalarIssue:
    return "scalar_issue";
  case StageCostModelKind::ScalarControl:
    return "scalar_control";
  case StageCostModelKind::ScalarMath:
    return "scalar_math";
  case StageCostModelKind::ScalarLoad:
    return "scalar_load";
  case StageCostModelKind::ScalarStore:
    return "scalar_store";
  case StageCostModelKind::IndexGeneration:
    return "index_generation";
  case StageCostModelKind::PredicateMask:
    return "predicate_mask";
  case StageCostModelKind::LoopPredicate:
    return "loop_predicate";
  case StageCostModelKind::ContinuousTileMemory:
    return "continuous_tile_memory";
  case StageCostModelKind::ContinuousTileStore:
    return "continuous_tile_store";
  case StageCostModelKind::ContinuousShortLoad:
    return "continuous_short_load";
  case StageCostModelKind::CachePolicyStore:
    return "cache_policy_store";
  case StageCostModelKind::IndirectScalarMemory:
    return "indirect_scalar_memory";
  case StageCostModelKind::IndirectGatherMemory:
    return "indirect_gather_memory";
  case StageCostModelKind::AtomicMemory:
    return "atomic_memory";
  case StageCostModelKind::IndependentPipelinedLoop:
    return "independent_pipelined_loop";
  case StageCostModelKind::LoopCarriedRecurrence:
    return "loop_carried_recurrence";
  case StageCostModelKind::RowwiseReduction:
    return "rowwise_reduction";
  case StageCostModelKind::PrefixScan:
    return "prefix_scan";
  case StageCostModelKind::CubeRoofline:
    return "cube_roofline";
  case StageCostModelKind::TinyCubeRoofline:
    return "tiny_cube_roofline";
  case StageCostModelKind::ConversionPack:
    return "conversion_pack";
  }
  llvm_unreachable("unknown StageCostModelKind");
}

bool StageControlFlowRates::isFiniteAndNonNegative() const {
  const std::array<double, 4> values = {
      loopBackedgeCycles, conditionalBranchCycles, divergentBranchPenaltyCycles,
      synchronizationCycles};
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

bool StageAtomicRate::isValid() const {
  return std::isfinite(logicalElementsPerCycle) &&
         logicalElementsPerCycle > 0.0 &&
         std::isfinite(operationStartupCycles) &&
         operationStartupCycles >= 0.0 &&
         std::isfinite(resultDependencyCycles) &&
         resultDependencyCycles >= 0.0 &&
         std::isfinite(unknownContentionMultiplier) &&
         unknownContentionMultiplier >= 1.0;
}

bool StageModeProfile::isValid(StageMode mode) const {
  const std::array<double, 18> common = {setupCycles,
                                         predicateOperationsPerCycle,
                                         shuffleLanesPerCycle,
                                         dotSetupCycles,
                                         dotFlopsPerCycle,
                                         scalarOperationsPerCycle,
                                         issueOperationsPerCycle,
                                         spillTransactionsPerCycle,
                                         indirectLoadTransactionsPerCycle,
                                         indirectStoreTransactionsPerCycle,
                                         prefixScanDependencyFactor,
                                         prefixScanDependencyFactor1d,
                                         prefixScanDependencyFactor1dSmall,
                                         prefixScanDependencyFactor1dMidWork,
                                         prefixScanDependencyFactor1dTiledWork,
                                         static_cast<double>(vectorWidthBits),
                                         static_cast<double>(vectorWidth),
                                         static_cast<double>(issueWidth)};
  if (!std::all_of(
          common.begin(), common.end(),
          [](double value) { return std::isfinite(value) && value > 0.0; }) ||
      !std::isfinite(indirectDependencyLatencyCycles) ||
      indirectDependencyLatencyCycles < 0.0 ||
      !std::isfinite(prefixScanDependencyFactorMultiTail) ||
      prefixScanDependencyFactorMultiTail < 0.0 ||
      !controlFlow.isFiniteAndNonNegative())
    return false;
  if (mode == StageMode::SIMD) {
    if (!(loadBytesPerCycle > 0.0 && storeBytesPerCycle > 0.0))
      return false;
  } else if (!(loadWarpInstructionsPerCycle > 0.0 &&
               storeWarpInstructionsPerCycle > 0.0)) {
    return false;
  }
  return llvm::all_of(operationRates,
                      [](const auto &entry) {
                        return std::isfinite(entry.second.throughput) &&
                               entry.second.throughput > 0.0 &&
                               std::isfinite(entry.second.factor) &&
                               entry.second.factor > 0.0;
                      }) &&
         atomicRates.contains("default") &&
         llvm::all_of(atomicRates,
                      [](const auto &entry) { return entry.second.isValid(); });
}

bool HardwareProfile::isValid() const {
  return !profileVersion.empty() && !target.empty() &&
         logicalWarpGroupCount > 0 && superblockUsefulFactorLimit > 0 &&
         superblockPersistentStatePressureFreeFactor > 0 &&
         superblockPersistentStatePressureFreeFactor <=
             superblockUsefulFactorLimit &&
         std::isfinite(superblockPersistentStateBytesPerCycle) &&
         superblockPersistentStateBytesPerCycle > 0.0 &&
         simd.isValid(StageMode::SIMD) && simt.isValid(StageMode::SIMT) &&
         transition.isValid();
}

llvm::Expected<StageCostTable>
StageCostEvaluator::evaluate(const StagePartition &partition,
                             const HardwareProfile &profile) const {
  if (partition.stages.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition requires at least one Stage");
  if (!profile.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "HardwareProfile is invalid");
  StageCostTable table;
  table.operationOwnershipComplete = partition.operationOwnershipComplete;
  table.modeledOperationCount = partition.modeledOperationCount;
  table.profileVersion = profile.profileVersion;
  llvm::StringSet<> stageIds;

  for (const LogicalStage &stage : partition.stages) {
    if (stage.id.empty() || !stageIds.insert(stage.id).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage ids must be non-empty and unique: '%s'", stage.id.c_str());
    if (stage.iterationCount <= 0 || !stage.features.isValid() ||
        !stage.workload.isFiniteAndNonNegative())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage '%s' has invalid iteration/features", stage.id.c_str());
    if (!stage.simdLegal && !stage.simtLegal)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Stage '%s' has no legal StageMode",
                                     stage.id.c_str());
    if (stage.simtLegal && stage.legalSimtFactors.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "SIMT Stage '%s' has no legal SuperBlock factor", stage.id.c_str());

    LogicalStageCost logicalCost;
    logicalCost.id = stage.id;
    logicalCost.model = stringifyStageCostModel(stage.costModelKind).str();
    logicalCost.schedule = stage.scheduleKind;
    logicalCost.iterationCount = stage.iterationCount;
    logicalCost.features = stage.features;
    logicalCost.workload = stage.workload;
    logicalCost.ownedOperationCount =
        static_cast<int64_t>(stage.operations.size());
    logicalCost.sourceLocations = collectSourceLocations(stage);
    logicalCost.liveInCount = static_cast<int64_t>(stage.liveIns.size());
    logicalCost.liveOutCount = static_cast<int64_t>(stage.liveOuts.size());
    logicalCost.liveInBytes = stage.liveInBytes;
    logicalCost.liveOutBytes = stage.liveOutBytes;
    logicalCost.localSimtScopeCount = stage.localSimtScopeCount;
    logicalCost.scopeInputTensorBytes = stage.scopeInputTensorBytes;
    logicalCost.scopeOutputTensorBytes = stage.scopeOutputTensorBytes;
    logicalCost.simtAnchorIndices = stage.simtAnchorIndices;
    logicalCost.localSimtMaterializable = stage.localSimtMaterializable;
    logicalCost.localSuperblockMaterializable =
        stage.localSuperblockMaterializable;
    logicalCost.legalSimtFactors = stage.legalSimtFactors;
    logicalCost.localSimtFactors = stage.localSimtFactors;

    llvm::SmallVector<StageImplementation> implementations;
    if (stage.simdLegal)
      implementations.push_back({StageMode::SIMD, 1, false});
    if (stage.simtLegal)
      for (int64_t factor : stage.legalSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, false});
    if (stage.simtLegal && stage.localSimtMaterializable)
      for (int64_t factor : stage.localSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, true});

    for (const StageImplementation &implementation : implementations) {
      if (!isDeclaredLegal(stage, implementation))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has an illegal candidate",
                                       stage.id.c_str());
      StageResourceCycles resources = mapWorkload(
          stage,
          implementation.mode == StageMode::SIMD ? profile.simd : profile.simt,
          implementation.mode);
      // estimateStage also records the per-pipe split of the priced body, so it
      // must run before the resources are copied into the result.
      const double stageCycles =
          estimateStage(stage, profile, implementation.mode, resources);
      StageImplementationCost cost;
      cost.implementation = implementation;
      cost.resources = resources;
      cost.totalCycles = applySuperBlock(stage, resources, implementation,
                                         profile, stageCycles);
      if (!cost.isValid())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' produced an invalid cost",
                                       stage.id.c_str());
      logicalCost.implementations.push_back(std::move(cost));
    }

    table.stages.push_back(std::move(logicalCost));
  }
  return table;
}
