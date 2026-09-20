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

/// Structured CAModel active-cycle model for `K` MainScalar scalar loads that
/// touch `U` distinct 64B AIV DCache lines (round 1-6 calibration):
///   T = prep + fill
///       + max(0, U - outstanding) * perLineCost(U)
///       + (K - U) * hit
///       + (K - 1) * issue
/// `sameLine` forces U = 1 (white-box same-line branch).
static double mainScalarLoadCycles(double opCount, double uniqueLines,
                                   bool sameLine,
                                   const StageModeProfile &profile) {
  const double k = std::max(1.0, opCount);
  const double u = sameLine ? 1.0 : std::clamp(uniqueLines, 1.0, k);
  const double perLine = u <= profile.mainScalarLoadExtraLineHighThreshold
                             ? profile.mainScalarLoadExtraLineLowCycles
                             : profile.mainScalarLoadExtraLineHighCycles;
  return profile.mainScalarLoadPrepCycles + profile.mainScalarLoadFillCycles +
         std::max(0.0, u - profile.mainScalarLoadOutstandingLines) * perLine +
         (k - u) * profile.mainScalarLoadHitCycles +
         (k - 1.0) * profile.mainScalarLoadIssueCycles;
}

/// Structured CAModel active-cycle model for `K` SIMT warp-uniform scalar
/// loads.  Same-line reuse serializes through the (CAModel) SIMT DCache;
/// distinct 128B lines can overlap and only pay the LSU issue floor.
static double simtUniformLoadCycles(double opCount, bool sameLine,
                                    const StageModeProfile &profile) {
  const double k = std::max(1.0, opCount);
  const double marginal = sameLine ? profile.simtUniformLoadSameLineSerialCycles
                                   : profile.simtUniformLoadDiffLineIssueCycles;
  return profile.simtUniformLoadPrepCycles + profile.simtUniformLoadFillCycles +
         (k - 1.0) * marginal;
}

/// White-box CAModel active-cycle model for `K` Triton SIMD scalar stores.
/// Triton lowers a scalar `tt.store` to `SCALAR ST_XD_XN_IMM -> UB staging`
/// followed by MTE3 `MOV_SRC_TO_DST_ALIGNv2 UB -> OUT`; the CCE MainScalar
/// `ST_XD_XN_IMM -> GM` write-allocate path is not used.  The MTE3 window is
/// modeled as `prep + fill + (K - 1) * serial`.
static double mte3StoreCycles(double opCount, const StageModeProfile &profile) {
  const double k = std::max(1.0, opCount);
  return profile.mte3StorePrepCycles + profile.mte3StoreFillCycles +
         (k - 1.0) * profile.mte3StoreSerialCycles;
}

/// Structured CAModel active-cycle model for `K` SIMT warp-uniform scalar
/// stores (`SCALAR-MODEL.md` section 3.4/4.6).  The same-line serial branch
/// is only meaningful for K >= 2 stores that land on one line; a single
/// scalar store uses the first-store (diff-line) preparation.
static double simtUniformStoreCycles(double opCount, bool sameLine,
                                     const StageModeProfile &profile) {
  const double k = std::max(1.0, opCount);
  if (sameLine && k > 1.0)
    return profile.simtUniformStoreSameLineBaseCycles +
           (k - 1.0) * profile.simtUniformStoreSameLineSerialCycles;
  return profile.simtUniformStoreDiffLineBaseCycles +
         (k - 1.0) * profile.simtUniformStoreDiffLineIssueCycles;
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

/// Legacy `scalar_ldst` cheap-hash fit:
///   cycles = a + b*warps + c*ops + d*warps*ops
/// It models warm/runtime-loop throughput rather than a cold Stage active
/// window, so it is only used when the structured white-box fields are absent.
static double lookupThroughputByCyclesFit(const std::vector<double> &fit,
                                          double opCount, double warpCount) {
  if (fit.size() != 4 || opCount <= 0.0 || warpCount <= 0.0)
    return 0.0;
  const double cycles = fit[0] + fit[1] * warpCount + fit[2] * opCount +
                        fit[3] * warpCount * opCount;
  if (!(cycles > 0.0))
    return 0.0;
  return warpCount * opCount / cycles;
}

static StageResourceCycles mapWorkload(const LogicalStage &stage,
                                       const StageModeProfile &profile,
                                       StageMode mode, int64_t effectiveWarps) {
  StageResourceCycles resources;
  const StageWorkload &work = stage.workload;
  const bool simd = mode == StageMode::SIMD;
  resources.setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
  for (const auto &[name, elements] : work.operationElements) {
    auto rate = profile.operationRates.find(name);
    if (rate == profile.operationRates.end() || rate->second.throughput <= 0.0)
      continue;
    const double instructions =
        simd ? std::ceil(elements / static_cast<double>(profile.vectorWidth))
             : elements;
    resources.compute +=
        instructions / rate->second.throughput * rate->second.factor;
  }
  resources.scalar = work.scalarOperations / profile.scalarOperationsPerCycle;
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
  // Scalar loads/stores execute on the scalar pipe, not on vector MTE.
  // Prefer structured CAModel active-cycle formulas when present.  The
  // conservative default is the diff-line branch: scalarLoadsShareLine is
  // only true when the IR proves all scalar loads fall inside one 64B line.
  if (work.scalarLoadCount > 0.0) {
    const double k = work.scalarLoadCount;
    const double inferredLines =
        work.scalarLoadUniqueLines > 0.0 ? work.scalarLoadUniqueLines : k;
    const double uniqueLines = stage.features.scalarLoadsShareLine
                                   ? 1.0
                                   : std::max(1.0, std::min(k, inferredLines));
    const bool hasMainScalarDiffFields =
        profile.mainScalarLoadOutstandingLines > 0.0 &&
        profile.mainScalarLoadExtraLineLowCycles > 0.0;
    const bool useMainScalarStructured =
        simd && profile.mainScalarLoadFillCycles > 0.0 &&
        (stage.features.scalarLoadsShareLine || hasMainScalarDiffFields);
    const bool useSimtStructured =
        !simd && profile.simtUniformLoadFillCycles > 0.0 &&
        (stage.features.scalarLoadsShareLine ||
         profile.simtUniformLoadDiffLineIssueCycles > 0.0);
    if (useMainScalarStructured) {
      resources.load += mainScalarLoadCycles(
          k, uniqueLines, stage.features.scalarLoadsShareLine, profile);
    } else if (useSimtStructured) {
      resources.load += simtUniformLoadCycles(
          k, stage.features.scalarLoadsShareLine, profile);
    } else {
      double throughput = profile.scalarLoadInstructionsPerCycle;
      if (!simd) {
        const double fitted = lookupThroughputByCyclesFit(
            profile.scalarLoadCyclesFit, k, effectiveWarps);
        if (fitted > 0.0)
          throughput = fitted;
      }
      if (throughput > 0.0)
        resources.load += k / throughput + profile.scalarLoadLatencyCycles;
      else
        resources.load += profile.scalarLoadLatencyCycles;
    }
    // Charge legacy serial-chain load-to-use latency only for edges beyond
    // the first exposure.  A single shallow edge (one producer feeding one
    // consumer, or one producer fanning out) is already covered by the
    // consumer's own white-box line cost; charging the full legacy 65.4 cyc
    // on this first edge over-predicted the binned SIMT indirect load.
    // Deeper serial chains still pay one extra latency per additional edge.
    const double indirectLoadExposures =
        work.indirectScalarLoadExposureCount > 0.0
            ? work.indirectScalarLoadExposureCount
            : work.indirectScalarLoadCount;
    if (stage.features.hasScalarIndirectLoad && indirectLoadExposures > 1.0)
      resources.load += (indirectLoadExposures - 1.0) *
                        profile.scalarIndirectDependencyLatencyCycles;
  }
  if (work.scalarStoreCount > 0.0) {
    const double k = std::max(1.0, work.scalarStoreCount);
    const bool useMte3StoreStructured =
        simd && profile.mte3StoreFillCycles > 0.0;
    const bool useSimtStoreStructured =
        !simd && profile.simtUniformStoreDiffLineBaseCycles > 0.0;
    if (useMte3StoreStructured) {
      // White-box Triton SIMD scalar store lowering: scalar ST -> UB staging
      // followed by MTE3 MOV UB -> OUT.
      resources.store += mte3StoreCycles(k, profile);
    } else if (useSimtStoreStructured) {
      // White-box SIMT warp-uniform store: same-line serial branch only for
      // K >= 2; a single scalar store uses the first-store preparation.
      resources.store += simtUniformStoreCycles(
          k, stage.features.scalarStoresShareLine, profile);
    } else {
      // Legacy scalar_ldst fallback: SIMT uses the `store_cycles_fit`
      // aggregate fit (warm/runtime-loop throughput); SIMD keeps the
      // provisional scalar-pipe throughput.
      double throughput = profile.scalarStoreInstructionsPerCycle;
      if (!simd) {
        const double fitted =
            lookupThroughputByCyclesFit(profile.scalarStoreCyclesFit,
                                        work.scalarStoreCount, effectiveWarps);
        if (fitted > 0.0)
          throughput = fitted;
      }
      if (throughput > 0.0)
        resources.store += work.scalarStoreCount / throughput +
                           profile.scalarStoreLatencyCycles;
      else
        resources.store += profile.scalarStoreLatencyCycles;
    }
    const double indirectStoreExposures =
        work.indirectScalarStoreExposureCount > 0.0
            ? work.indirectScalarStoreExposureCount
            : work.indirectScalarStoreCount;
    if (stage.features.hasScalarIndirectStore && indirectStoreExposures > 1.0)
      resources.store += (indirectStoreExposures - 1.0) *
                         profile.scalarIndirectDependencyLatencyCycles;
  }
  resources.predicate =
      (simd ? std::ceil(work.predicateElements /
                        static_cast<double>(profile.vectorWidth))
            : work.predicateElements) /
      profile.predicateOperationsPerCycle;
  resources.shuffle = work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
  resources.scanShuffle =
      work.scanShuffleLaneSteps / profile.shuffleLanesPerCycle;
  if (work.dotFlops > 0.0) {
    resources.setup += profile.dotSetupCycles;
    resources.dot = work.dotFlops / profile.dotFlopsPerCycle;
  }
  resources.issue =
      std::ceil(work.issueElements / static_cast<double>(profile.issueWidth)) /
      profile.issueOperationsPerCycle;
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
                            const StageResourceCycles &r) {
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
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
      critical =
          std::max(r.criticalPath + (dependencyFactor - 1.0) * r.scanShuffle +
                       r.load + r.store + r.atomic + controlBody(r) + r.spill,
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
      return r.setup + count * critical + persistentState;
    }
    const int64_t groups = std::max<int64_t>(
        1, std::min(stage.features.parallelRecurrenceGroupCount,
                    profile.logicalWarpGroupCount));
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
    const double scanCritical =
        r.compute + r.predicate +
        r.shuffle * (mode == StageMode::SIMD
                         ? profile.simd.prefixScanDependencyFactor
                         : profile.simt.prefixScanDependencyFactor);
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
  const std::array<double, 13> common = {setupCycles,
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
                                         static_cast<double>(vectorWidth),
                                         static_cast<double>(issueWidth)};
  const bool hasWhiteboxLoad = mode == StageMode::SIMD
                                   ? mainScalarLoadFillCycles > 0.0
                                   : simtUniformLoadFillCycles > 0.0;
  if (!std::all_of(
          common.begin(), common.end(),
          [](double value) { return std::isfinite(value) && value > 0.0; }) ||
      !std::isfinite(scalarLoadInstructionsPerCycle) ||
      scalarLoadInstructionsPerCycle < 0.0 ||
      (!hasWhiteboxLoad && scalarLoadInstructionsPerCycle == 0.0) ||
      !std::isfinite(scalarStoreInstructionsPerCycle) ||
      scalarStoreInstructionsPerCycle <= 0.0 ||
      !std::isfinite(indirectDependencyLatencyCycles) ||
      indirectDependencyLatencyCycles < 0.0 ||
      !std::isfinite(scalarLoadLatencyCycles) ||
      scalarLoadLatencyCycles < 0.0 ||
      !std::isfinite(scalarStoreLatencyCycles) ||
      scalarStoreLatencyCycles < 0.0 ||
      !std::isfinite(scalarIndirectDependencyLatencyCycles) ||
      scalarIndirectDependencyLatencyCycles < 0.0 ||
      !std::isfinite(mainScalarLoadPrepCycles) ||
      mainScalarLoadPrepCycles < 0.0 ||
      !std::isfinite(mainScalarLoadFillCycles) ||
      mainScalarLoadFillCycles < 0.0 ||
      !std::isfinite(mainScalarLoadHitCycles) ||
      mainScalarLoadHitCycles < 0.0 ||
      !std::isfinite(mainScalarLoadIssueCycles) ||
      mainScalarLoadIssueCycles < 0.0 ||
      !std::isfinite(simtUniformLoadPrepCycles) ||
      simtUniformLoadPrepCycles < 0.0 ||
      !std::isfinite(simtUniformLoadFillCycles) ||
      simtUniformLoadFillCycles < 0.0 ||
      !std::isfinite(simtUniformLoadSameLineSerialCycles) ||
      simtUniformLoadSameLineSerialCycles < 0.0 ||
      !std::isfinite(mainScalarLoadOutstandingLines) ||
      mainScalarLoadOutstandingLines < 0.0 ||
      !std::isfinite(mainScalarLoadExtraLineLowCycles) ||
      mainScalarLoadExtraLineLowCycles < 0.0 ||
      !std::isfinite(mainScalarLoadExtraLineHighCycles) ||
      mainScalarLoadExtraLineHighCycles < 0.0 ||
      !std::isfinite(mainScalarLoadExtraLineHighThreshold) ||
      mainScalarLoadExtraLineHighThreshold < 0.0 ||
      !std::isfinite(simtUniformLoadDiffLineIssueCycles) ||
      simtUniformLoadDiffLineIssueCycles < 0.0 ||
      !std::isfinite(mte3StorePrepCycles) || mte3StorePrepCycles < 0.0 ||
      !std::isfinite(mte3StoreFillCycles) || mte3StoreFillCycles < 0.0 ||
      !std::isfinite(mte3StoreSerialCycles) || mte3StoreSerialCycles < 0.0 ||
      !std::isfinite(simtUniformStoreSameLineBaseCycles) ||
      simtUniformStoreSameLineBaseCycles < 0.0 ||
      !std::isfinite(simtUniformStoreSameLineSerialCycles) ||
      simtUniformStoreSameLineSerialCycles < 0.0 ||
      !std::isfinite(simtUniformStoreDiffLineBaseCycles) ||
      simtUniformStoreDiffLineBaseCycles < 0.0 ||
      !std::isfinite(simtUniformStoreDiffLineIssueCycles) ||
      simtUniformStoreDiffLineIssueCycles < 0.0 ||
      !std::all_of(
          scalarLoadCyclesFit.begin(), scalarLoadCyclesFit.end(),
          [](double value) { return std::isfinite(value); }) ||
      !std::all_of(
          scalarStoreCyclesFit.begin(), scalarStoreCyclesFit.end(),
          [](double value) { return std::isfinite(value); }) ||
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
      const int64_t effectiveWarps =
          implementation.mode == StageMode::SIMT
              ? std::max<int64_t>(1, profile.logicalWarpGroupCount *
                                         implementation.superblockFactor)
              : 1;
      StageResourceCycles resources = mapWorkload(
          stage,
          implementation.mode == StageMode::SIMD ? profile.simd : profile.simt,
          implementation.mode, effectiveWarps);
      StageImplementationCost cost;
      cost.implementation = implementation;
      cost.resources = resources;
      cost.totalCycles = applySuperBlock(
          stage, resources, implementation, profile,
          estimateStage(stage, profile, implementation.mode, resources));
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
