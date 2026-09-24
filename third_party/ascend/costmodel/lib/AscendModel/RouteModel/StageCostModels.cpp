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
#include <limits>
#include <optional>
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
                           resources.atomic + resources.compute +
                           resources.predicate + resources.shuffle +
                           resources.reduction + resources.dot +
                           controlBody(resources) + resources.spill;
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

static const std::vector<double> *
getReductionParameters(const ReductionCostProfile &profile, llvm::StringRef key,
                       size_t expectedSize) {
  auto found = profile.parameters.find(key);
  if (found == profile.parameters.end() || found->second.size() != expectedSize)
    return nullptr;
  return &found->second;
}

static std::string reductionRoute(const ReductionWorkload &work) {
  return work.kind + "_" + work.dataType;
}

static bool isSupportedReductionRoute(const ReductionWorkload &work) {
  const llvm::StringRef op(work.kind);
  const llvm::StringRef dtype(work.dataType);
  if (op == "sum" || op == "max" || op == "min")
    return dtype == "f16" || dtype == "bf16" || dtype == "f32" ||
           dtype == "i32";
  if (op == "prod")
    return dtype == "f16" || dtype == "bf16" || dtype == "f32" ||
           dtype == "i8" || dtype == "i16" || dtype == "i32" || dtype == "i64";
  if (op == "xor" || op == "or" || op == "and")
    return dtype == "i8" || dtype == "i16" || dtype == "i32" || dtype == "i64";
  return false;
}

static std::string canonicalStandardRoute(const ReductionWorkload &work) {
  const std::string route = reductionRoute(work);
  if (work.kind == "min")
    return "max_" +
           (work.dataType == "bf16" ? std::string("f16") : work.dataType);
  if (work.dataType == "bf16" && (work.kind == "sum" || work.kind == "max"))
    return work.kind + "_f16";
  return route;
}

static std::optional<int64_t> checkedProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

static int64_t nextPowerOfTwo(int64_t value) {
  int64_t result = 1;
  while (result < value && result <= 4096)
    result *= 2;
  return result;
}

static std::optional<double>
estimateSimdRank1(const ReductionWorkload &work,
                  const ReductionCostProfile &profile, int64_t k) {
  const std::string route = canonicalStandardRoute(work);
  if (const auto *p =
          getReductionParameters(profile, "r1_standard_" + route, 3)) {
    const double q = (*p)[0];
    return (*p)[1] + (*p)[2] * std::max(0.0, std::ceil(k / q) - 1.0);
  }
  const std::string raw = reductionRoute(work);
  if (const auto *p =
          getReductionParameters(profile, "r1_piecewise_" + raw, 6)) {
    if (k < 32) {
      if (k != 2 && k != 4 && k != 8 && k != 16)
        return std::nullopt;
      unsigned index = static_cast<unsigned>(std::log2(k)) - 1;
      return (*p)[index];
    }
    return (*p)[4] + (*p)[5] * static_cast<double>(k);
  }
  return std::nullopt;
}

static std::optional<double>
estimateSimdRankN(const ReductionWorkload &work,
                  const ReductionCostProfile &profile, int64_t g, int64_t k) {
  const std::string route = canonicalStandardRoute(work);
  if (route == "xor_i64" && k == 2) {
    if (const auto *p = getReductionParameters(profile, "rn_xor_i64_k2", 2))
      return (*p)[0] + (*p)[1] * static_cast<double>(g);
    return std::nullopt;
  }
  if (const auto *p =
          getReductionParameters(profile, "rn_standard_" + route, 5)) {
    const double h = std::max(0.0, std::ceil(k / (*p)[0]) - 1.0);
    const double shortK = k >= 2 && k <= 8 ? 1.0 : 0.0;
    return (*p)[1] + (*p)[2] * g + (*p)[3] * g * h + (*p)[4] * g * shortK;
  }
  const std::string raw = reductionRoute(work);
  if (const auto *p = getReductionParameters(profile, "rn_linear_" + raw, 2)) {
    const double combines = static_cast<double>(g) * (k - 1);
    return std::max((*p)[0], (*p)[1] * combines);
  }
  return std::nullopt;
}

static std::string simtRank1Class(const ReductionWorkload &work) {
  if (work.kind == "prod" &&
      (work.dataType == "f16" || work.dataType == "bf16"))
    return "prod16";
  if (work.dataType == "i64" && work.kind == "prod")
    return "prod_i64";
  if (work.dataType == "i64" &&
      (work.kind == "xor" || work.kind == "or" || work.kind == "and"))
    return "i64";
  return "standard";
}

static std::optional<double>
rank1WarpFactor(const ReductionCostProfile &profile, llvm::StringRef routeClass,
                int64_t warps, int64_t b) {
  const auto *p = getReductionParameters(
      profile, ("r1_w_" + routeClass + "_" + llvm::Twine(warps)).str(), 12);
  if (!p || b < 2 || b > 4096 || (b & (b - 1)) != 0)
    return std::nullopt;
  return (*p)[static_cast<size_t>(std::log2(b)) - 1];
}

static std::optional<double>
estimateSimtRank1(const ReductionWorkload &work,
                  const ReductionCostProfile &profile, int64_t warps,
                  int64_t k) {
  const int64_t b = nextPowerOfTwo(k);
  if (b < 2 || b > 4096)
    return std::nullopt;
  const std::string routeClass = simtRank1Class(work);
  const auto factor = rank1WarpFactor(profile, routeClass, warps, b);
  if (!factor)
    return std::nullopt;
  const double l = std::log2(static_cast<double>(b));
  double base = 0.0;
  if (routeClass == "prod16") {
    const auto *p = getReductionParameters(profile, "r1_base_prod16", 9);
    if (!p)
      return std::nullopt;
    base = b <= 128 ? (*p)[static_cast<size_t>(l) - 1]
                    : (*p)[7] + (*p)[8] * (l - 7.0);
  } else {
    const std::string baseClass =
        routeClass == "prod_i64" ? std::string("i64") : routeClass;
    const auto *p = getReductionParameters(profile, "r1_base_" + baseClass, 4);
    if (!p)
      return std::nullopt;
    base = (*p)[0] + (*p)[1] * l + (*p)[2] * (b >= 64 ? 1.0 : 0.0) +
           (*p)[3] * std::max(l - 6.0, 0.0);
  }
  return base * *factor;
}

static std::string simtRankNBaseRoute(const ReductionWorkload &work) {
  if (work.kind == "min")
    return "max_" +
           (work.dataType == "bf16" ? std::string("f16") : work.dataType);
  if (work.kind == "or" || work.kind == "and")
    return "xor_" + work.dataType;
  if (work.dataType == "bf16" &&
      (work.kind == "sum" || work.kind == "max" || work.kind == "prod"))
    return work.kind + "_f16";
  if (work.kind == "prod" && work.dataType == "i32")
    return "prod_f32";
  return reductionRoute(work);
}

static std::optional<double>
estimateSimtRankN(const ReductionWorkload &work,
                  const ReductionCostProfile &profile, int64_t warps, int64_t g,
                  int64_t k) {
  const std::string baseRoute = simtRankNBaseRoute(work);
  const auto *p = getReductionParameters(profile, "rn_base_" + baseRoute, 8);
  const auto *scale =
      getReductionParameters(profile, "rn_alias_" + reductionRoute(work), 1);
  const auto *warp = getReductionParameters(
      profile, "rn_w_" + baseRoute + "_" + std::to_string(warps), 1);
  if (!p || !scale || !warp)
    return std::nullopt;
  const double h = std::max(0.0, std::ceil(k / (*p)[0]) - 1.0);
  const double excess =
      std::max(static_cast<double>(g) * k - (*p)[1], 0.0) / 64.0;
  const double base = (*p)[2] + (*p)[3] * g + (*p)[4] * g * h +
                      (*p)[5] * excess + (*p)[6] * g * h * h +
                      (*p)[7] * excess * excess / 64.0;
  return (*scale)[0] * (*warp)[0] * base;
}

static std::optional<double>
estimateTailAxisReductionCycles(const ReductionWorkload &work,
                                const ReductionCostProfile &profile,
                                StageMode mode, int64_t numWarps) {
  if (!work.isFiniteAndNonNegative() || work.axis + 1 != work.shape.size() ||
      !isSupportedReductionRoute(work))
    return std::nullopt;
  const int64_t k = work.shape.back();
  if (k <= 1)
    return 0.0;
  if (k > 4096)
    return std::nullopt;
  std::optional<double> cycles;
  if (work.shape.size() == 1) {
    cycles = mode == StageMode::SIMD
                 ? estimateSimdRank1(work, profile, k)
                 : estimateSimtRank1(work, profile, numWarps, k);
  } else {
    auto g = checkedProduct(llvm::ArrayRef<int64_t>(work.shape).drop_back());
    if (!g || *g > 256)
      return std::nullopt;
    cycles = mode == StageMode::SIMD
                 ? estimateSimdRankN(work, profile, *g, k)
                 : estimateSimtRankN(work, profile, numWarps, *g, k);
  }
  if (!cycles || !std::isfinite(*cycles) || *cycles < 0.0)
    return std::nullopt;
  return cycles;
}

static double legacyReductionShuffleCycles(const ReductionWorkload &work,
                                           double shuffleLanesPerCycle) {
  if (!work.isFiniteAndNonNegative() || shuffleLanesPerCycle <= 0.0)
    return 0.0;
  auto elements = checkedProduct(work.shape);
  const int64_t extent = work.shape[work.axis];
  if (!elements || extent <= 1)
    return 0.0;
  return work.instances * static_cast<double>(*elements) *
         std::ceil(std::log2(static_cast<double>(extent))) /
         shuffleLanesPerCycle;
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
                                       StageMode mode, int64_t numWarps) {
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
  resources.shuffle = work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
  for (const ReductionWorkload &reduction : work.reductionWorkloads) {
    auto cycles = estimateTailAxisReductionCycles(
        reduction, profile.tailAxisReduction, mode, numWarps);
    if (cycles)
      resources.reduction += reduction.instances * *cycles;
    else
      resources.shuffle +=
          legacyReductionShuffleCycles(reduction, profile.shuffleLanesPerCycle);
  }
  resources.scanShuffle =
      work.scanShuffleLaneSteps / profile.shuffleLanesPerCycle;
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
                             resources.reduction + resources.dot;
  else if (stage.features.hasReduction)
    resources.criticalPath = resources.compute + resources.predicate +
                             resources.shuffle + resources.reduction;
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
      resources.reduction + resources.divergence;
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
                            r.compute + r.dot + r.shuffle + r.reduction,
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
        r.compute + r.predicate + r.reduction +
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
                                r.shuffle + r.reduction + r.spill +
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
                            r.compute + r.dot + r.shuffle + r.reduction,
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

bool ReductionCostProfile::isValid() const {
  // An empty profile is a supported compatibility state: every reduction
  // falls back to the legacy shuffle formula.  Production schema v13 still
  // requires non-empty calibrated parameters.
  return llvm::all_of(parameters, [](const auto &entry) {
    return !entry.second.empty() &&
           llvm::all_of(entry.second,
                        [](double value) { return std::isfinite(value); });
  });
}

bool StageModeProfile::isValid(StageMode mode) const {
  const std::array<double, 14> common = {setupCycles,
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
                                         static_cast<double>(vectorWidthBits),
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
  return llvm::all_of(operationRates,
                      [](const auto &entry) {
                        return std::isfinite(entry.second.throughput) &&
                               entry.second.throughput > 0.0 &&
                               std::isfinite(entry.second.factor) &&
                               entry.second.factor > 0.0;
                      }) &&
         atomicRates.contains("default") && tailAxisReduction.isValid() &&
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
          implementation.mode, profile.logicalWarpGroupCount);
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
