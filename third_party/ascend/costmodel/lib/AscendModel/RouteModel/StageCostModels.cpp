//===- StageCostModels.cpp - Per-stage analytical models -----------------===//

#include "AscendModel/RouteModel/StageCostModels.h"
#include "AscendModel/Support/CostModelLogger.h"

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
  const double execution = resources.scalar + resources.load + resources.store +
                           resources.compute + resources.predicate +
                           resources.shuffle + resources.dot +
                           controlBody(resources) + resources.spill;
  // Issue is a shared front-end throughput bound, not an extra instruction
  // stream.  Adding it to execution double-counts every instruction.
  return std::max(execution, resources.issue);
}

static bool permitsSimdOverlap(const LogicalStage &stage) {
  return stage.scheduleKind == StageScheduleKind::IndependentPipelined &&
         stage.features.permitsSimdRoofline();
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
  if (stage.features.hasIndirectMemory) {
    const double loads =
        std::max(work.loadWarpInstructions, work.loadBytes > 0.0 ? 1.0 : 0.0);
    const double stores =
        std::max(work.storeWarpInstructions, work.storeBytes > 0.0 ? 1.0 : 0.0);
    resources.load = loads / profile.indirectLoadTransactionsPerCycle;
    resources.store = stores / profile.indirectStoreTransactionsPerCycle;
    if (loads + stores > 0.0)
      resources.load += profile.indirectDependencyLatencyCycles;
  } else if (simd) {
    resources.load = work.loadBytes / profile.loadBytesPerCycle;
    resources.store = work.storeBytes / profile.storeBytesPerCycle;
  } else {
    resources.load =
        work.loadWarpInstructions / profile.loadWarpInstructionsPerCycle;
    resources.store =
        work.storeWarpInstructions / profile.storeWarpInstructionsPerCycle;
  }
  resources.predicate =
      (simd ? std::ceil(work.predicateElements /
                        static_cast<double>(profile.vectorWidth))
            : work.predicateElements) /
      profile.predicateOperationsPerCycle;
  resources.shuffle = work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
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
                              double stageCycles,
                              std::string *formula = nullptr) {
  if (implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor == 1)
    return stageCycles;

  const double factor = static_cast<double>(implementation.superblockFactor);
  const double effectiveFactor = std::min(
      factor, static_cast<double>(profile.superblockUsefulFactorLimit));
  const double latencySensitivePerIteration = resources.load + resources.store +
                                              resources.shuffle +
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
  // aggregate issue floor: F2/F4 cannot create additional issue bandwidth.
  // This applies equally to whole-kernel and scope-local SuperBlock because
  // both materializers batch complete logical programs around the Stage.
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence) {
    const double recurrenceBody = std::max(0.0, stageCycles - fixed);
    if (formula)
      *formula = "superblock: max(setup + F*N_iter*issue, "
                 "setup + recurrenceBody + spillPressure) + "
                 "persistentStatePressure";
    return std::max(issueFloor, fixed + recurrenceBody + pressure) +
           persistentStatePressure;
  }
  // Proven persistent-state pressure is additional register/stack work and
  // cannot disappear behind the ordinary issue floor.
  const double body = std::max(0.0, stageCycles - fixed);
  const double groupedBody = factor * std::max(0.0, body - latencySensitive) +
                             factor * latencySensitive / effectiveFactor;
  if (formula)
    *formula = "superblock: max(setup + F*N_iter*issue, setup + F*(body - "
               "latencySensitive) + F*latencySensitive/F_eff + spillPressure) "
               "+ persistentStatePressure";
  return std::max(issueFloor, fixed + groupedBody + pressure) +
         persistentStatePressure;
}

static double estimateStage(const LogicalStage &stage,
                            const HardwareProfile &profile, StageMode mode,
                            const StageResourceCycles &r,
                            std::string *formula = nullptr) {
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
  if (formula)
    *formula = "total = setup + N_iter * max(execution, issue), execution = "
               "scalar + load + store + compute + predicate + shuffle + dot + "
               "control + spill";
  switch (stage.costModelKind) {
  case StageCostModelKind::AutoBlockifyDispatch:
  case StageCostModelKind::AutoBlockifyLoop: {
    const double dispatchCount =
        stage.costModelKind == StageCostModelKind::AutoBlockifyLoop ? count
                                                                    : 1.0;
    if (formula)
      *formula = "total = setup + D * max(scalar + control, issue)";
    return r.setup +
           dispatchCount * std::max(r.scalar + controlBody(r), r.issue);
  }
  case StageCostModelKind::ContinuousTileMemory:
  case StageCostModelKind::ContinuousTileStore:
  case StageCostModelKind::ContinuousShortLoad:
  case StageCostModelKind::CachePolicyStore:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      if (formula)
        *formula = "total = setup + N_iter * (scalar + predicate + control + "
                   "spill + max(load, store, issue))";
      return r.setup + count * (r.scalar + r.predicate + controlBody(r) +
                                r.spill + std::max({r.load, r.store, r.issue}));
    }
    return serial;
  case StageCostModelKind::IndependentPipelinedLoop:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      if (formula)
        *formula = "total = setup + N_iter * (max(load, store, compute + dot + "
                   "shuffle, scalar + predicate + control, issue) + spill)";
      return r.setup +
             count *
                 (std::max({r.load, r.store, r.compute + r.dot + r.shuffle,
                            r.scalar + r.predicate + controlBody(r), r.issue}) +
                  r.spill);
    }
    return serial;
  case StageCostModelKind::LoopCarriedRecurrence: {
    const double critical = r.criticalPath > 0.0
                                ? std::max(r.criticalPath + r.load + r.store +
                                               controlBody(r) + r.spill,
                                           r.issue)
                                : serialBody(r);
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
      if (formula)
        *formula = "total = setup + N_iter * max(criticalPath + load + store "
                   "+ control + spill, issue) + liveOutBytes / "
                   "persistentStateBytesPerCycle";
      return r.setup + count * critical + persistentState;
    }
    const int64_t groups = std::max<int64_t>(
        1, std::min(stage.features.parallelRecurrenceGroupCount,
                    profile.logicalWarpGroupCount));
    if (formula)
      *formula = "total = setup + max(ceil(N_iter / warpGroups) * critical, "
                 "N_iter * issue)";
    return r.setup +
           std::max(std::ceil(count / static_cast<double>(groups)) * critical,
                    count * r.issue);
  }
  case StageCostModelKind::RowwiseReduction:
    if (formula)
      *formula = "total = setup + N_iter * max(scalar + load + store + "
                 "criticalPath + control + spill, issue)";
    return r.setup +
           count * std::max(r.scalar + r.load + r.store + r.criticalPath +
                                controlBody(r) + r.spill,
                            r.issue);
  case StageCostModelKind::PrefixScan: {
    const double scanCritical =
        r.compute + r.predicate +
        r.shuffle * (mode == StageMode::SIMD
                         ? profile.simd.prefixScanDependencyFactor
                         : profile.simt.prefixScanDependencyFactor);
    if (formula)
      *formula = "total = setup + N_iter * max(scalar + load + store + compute "
                 "+ predicate + shuffle * scanDependencyFactor + control + "
                 "spill, issue)";
    return r.setup +
           count * std::max(r.scalar + r.load + r.store + scanCritical +
                                controlBody(r) + r.spill,
                            r.issue);
  }
  case StageCostModelKind::CubeRoofline:
  case StageCostModelKind::TinyCubeRoofline:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      if (formula)
        *formula = "total = setup + N_iter * (scalar + predicate + control + "
                   "shuffle + spill + max(load, compute + dot, store, issue))";
      return r.setup +
             count * (r.scalar + r.predicate + controlBody(r) + r.shuffle +
                      r.spill +
                      std::max({r.load, r.compute + r.dot, r.store, r.issue}));
    }
    return serial;
  case StageCostModelKind::ConversionPack:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      if (formula)
        *formula = "total = setup + N_iter * (predicate + control + spill + "
                   "max(scalar + compute, load, store, issue))";
      return r.setup + count * (r.predicate + controlBody(r) + r.spill +
                                std::max({r.scalar + r.compute, r.load, r.store,
                                          r.issue}));
    }
    return serial;
  default:
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

/// One log block per (Stage, implementation) candidate: mode, factor, the
/// mapped resource cycles, and the symbolic cost formula that produced the
/// total.
static void logImplementationCost(const LogicalStage &stage,
                                  const StageImplementation &implementation,
                                  const StageResourceCycles &r,
                                  double totalCycles,
                                  const std::string &formula,
                                  const std::string &superblockFormula) {
  llvm::raw_ostream &os = costModelLog();
  os << "cost stage '" << stage.id << "' ["
     << stringifyStageCostModel(stage.costModelKind) << "] "
     << (implementation.mode == StageMode::SIMD ? "SIMD" : "SIMT")
     << " factor=" << implementation.superblockFactor
     << (implementation.localScope ? " local_scope" : "")
     << ": total=" << totalCycles << " cycles\n";
  os << "  resources: setup=" << r.setup << " scalar=" << r.scalar
     << " load=" << r.load << " store=" << r.store << " compute=" << r.compute
     << " predicate=" << r.predicate << " shuffle=" << r.shuffle
     << " dot=" << r.dot << " loopCtl=" << r.loopControl
     << " branchCtl=" << r.branchControl << " divergence=" << r.divergence
     << " sync=" << r.synchronization << " spill=" << r.spill
     << " issue=" << r.issue << " criticalPath=" << r.criticalPath << "\n";
  os << "  formula: " << formula;
  if (!superblockFormula.empty())
    os << " | " << superblockFormula;
  os << "\n";
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
  if (!std::all_of(
          common.begin(), common.end(),
          [](double value) { return std::isfinite(value) && value > 0.0; }) ||
      !std::isfinite(indirectDependencyLatencyCycles) ||
      indirectDependencyLatencyCycles < 0.0 ||
      !controlFlow.isFiniteAndNonNegative())
    return false;
  if (mode == StageMode::SIMD) {
    if (!(loadBytesPerCycle > 0.0 && storeBytesPerCycle > 0.0))
      return false;
  } else if (!(loadWarpInstructionsPerCycle > 0.0 &&
               storeWarpInstructionsPerCycle > 0.0)) {
    return false;
  }
  return llvm::all_of(operationRates, [](const auto &entry) {
    return std::isfinite(entry.second.throughput) &&
           entry.second.throughput > 0.0 &&
           std::isfinite(entry.second.factor) && entry.second.factor > 0.0;
  });
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
  const bool trace = logEnabled();
  if (trace) {
    const StageModeProfile &simd = profile.simd;
    const StageModeProfile &simt = profile.simt;
    costModelLog() << "profile rates: SIMD(vectorWidth=" << simd.vectorWidth
                   << " issueWidth=" << simd.issueWidth
                   << " loadBytesPerCycle=" << simd.loadBytesPerCycle
                   << " storeBytesPerCycle=" << simd.storeBytesPerCycle
                   << " scalarOpsPerCycle=" << simd.scalarOperationsPerCycle
                   << " issueOpsPerCycle=" << simd.issueOperationsPerCycle
                   << ")\n";
    costModelLog() << "profile rates: SIMT(loadWarpsPerCycle="
                   << simt.loadWarpInstructionsPerCycle
                   << " storeWarpsPerCycle="
                   << simt.storeWarpInstructionsPerCycle
                   << " scalarOpsPerCycle=" << simt.scalarOperationsPerCycle
                   << " issueOpsPerCycle=" << simt.issueOperationsPerCycle
                   << " warpGroups=" << profile.logicalWarpGroupCount << ")\n";
  }

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
      std::string formula;
      std::string superblockFormula;
      const double baseCycles =
          estimateStage(stage, profile, implementation.mode, resources,
                        trace ? &formula : nullptr);
      StageImplementationCost cost;
      cost.implementation = implementation;
      cost.resources = resources;
      cost.totalCycles =
          applySuperBlock(stage, resources, implementation, profile,
                          baseCycles, trace ? &superblockFormula : nullptr);
      if (!cost.isValid())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' produced an invalid cost",
                                       stage.id.c_str());
      if (trace)
        logImplementationCost(stage, implementation, resources,
                              cost.totalCycles, formula, superblockFormula);
      logicalCost.implementations.push_back(std::move(cost));
    }

    table.stages.push_back(std::move(logicalCost));
  }
  return table;
}
