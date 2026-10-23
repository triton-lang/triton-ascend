//===- StageCostModels.cpp - Per-stage analytical models -----------------===//

#include "AscendModel/RouteModel/StageCostModels.h"

#include "AscendModel/RouteModel/Models/StridedShortCostModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <system_error>
#include <type_traits>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static double iterations(const LogicalStage &stage) {
  return static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
}

/// Resolve an operation-specific rate first.
static const StageOperationRate *
findOperationRate(const StageModeProfile &profile, llvm::StringRef name) {
  auto exact = profile.operationRates.find(name);
  if (exact != profile.operationRates.end())
    return &exact->second;
  if (!name.starts_with("fp32.saturated."))
    return nullptr;
  auto saturated = profile.operationRates.find("fp32.saturated");
  if (saturated != profile.operationRates.end())
    return &saturated->second;
  // Backward compatibility for hardware profiles created before the named
  // saturation alias existed.
  auto fp32Add = profile.operationRates.find("f32.add");
  return fp32Add == profile.operationRates.end() ? nullptr : &fp32Add->second;
}

static bool usesSaturatedComputeFallback(const StageModeProfile &profile,
                                         llvm::StringRef name) {
  return name.starts_with("fp32.saturated.") &&
         profile.operationRates.find(name) == profile.operationRates.end() &&
         (profile.operationRates.find("fp32.saturated") !=
              profile.operationRates.end() ||
          profile.operationRates.find("f32.add") !=
              profile.operationRates.end());
}

static double selectWarpCurveValue(const std::map<int64_t, double> &curve,
                                   double fallback, int64_t warpCount) {
  if (curve.empty())
    return fallback;
  auto selected = curve.lower_bound(warpCount);
  if (selected == curve.end())
    return std::prev(selected)->second;
  return selected->second;
}

/// Resolve occupancy-sensitive SIMT rates for one concrete implementation.
static StageModeProfile
resolveModeProfile(const HardwareProfile &hardware,
                   const StageImplementation &implementation,
                   bool dependentDotRecurrence) {
  StageModeProfile resolved =
      implementation.mode == StageMode::SIMD ? hardware.simd : hardware.simt;
  if (implementation.mode == StageMode::SIMD)
    return resolved;

  const int64_t activeWarps =
      std::max<int64_t>(1, hardware.baseSimtWarpCount) *
      std::max<int64_t>(1, implementation.superblockFactor);
  resolved.activeWarpCount = activeWarps;
  resolved.setupCycles = selectWarpCurveValue(
      resolved.setupCyclesByWarpCount, resolved.setupCycles, activeWarps);
  for (auto &entry : resolved.operationRates)
    entry.second.throughput =
        selectWarpCurveValue(entry.second.throughputByWarpCount,
                             entry.second.throughput, activeWarps);
  auto predicate = resolved.operationRates.find("predicate.cmp");
  if (predicate != resolved.operationRates.end() &&
      predicate->second.throughput > 0.0)
    resolved.predicateOperationsPerCycle =
        predicate->second.throughput / std::max(1.0, predicate->second.factor);
  resolved.shuffleLanesPerCycle =
      selectWarpCurveValue(resolved.shuffleLanesPerCycleByWarpCount,
                           resolved.shuffleLanesPerCycle, activeWarps);
  resolved.loadWarpInstructionsPerCycle =
      selectWarpCurveValue(resolved.loadWarpInstructionsPerCycleByWarpCount,
                           resolved.loadWarpInstructionsPerCycle, activeWarps);
  resolved.storeWarpInstructionsPerCycle =
      selectWarpCurveValue(resolved.storeWarpInstructionsPerCycleByWarpCount,
                           resolved.storeWarpInstructionsPerCycle, activeWarps);
  // The occupancy sweep executes an accumulator-feedback dot chain.
  if (dependentDotRecurrence)
    resolved.dotFlopsPerCycle =
        selectWarpCurveValue(resolved.dotFlopsPerCycleByWarpCount,
                             resolved.dotFlopsPerCycle, activeWarps);
  // Factor-specific group curves are measured at one explicit base warp count.
  if (resolved.recurrenceGroupBaseWarpCount != hardware.baseSimtWarpCount) {
    resolved.recurrenceReductionGroupCyclesByFactorAndLaneSteps.clear();
    resolved.recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps.clear();
  }
  if (resolved.continuousMemoryGroupBaseWarpCount !=
      hardware.baseSimtWarpCount) {
    resolved.continuousLoadGroupCyclesByFactorAndBytes.clear();
    resolved.continuousStoreGroupCyclesByFactorAndBytes.clear();
    resolved.continuousLoadGroupStartupCyclesByFactorAndBytes.clear();
    resolved.continuousStoreGroupStartupCyclesByFactorAndBytes.clear();
  }
  return resolved;
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
      resources.indirectLoadGroup + resources.sharedLayout + resources.compute +
      resources.predicate + resources.reduction + resources.shuffle +
      resources.dot + controlBody(resources) + resources.spill;
  // Issue is a shared front-end throughput bound, not an extra instruction
  // stream.  Adding it to execution double-counts every instruction.
  return std::max(execution, resources.issue);
}

static bool permitsSimdOverlap(const LogicalStage &stage) {
  return stage.scheduleKind == StageScheduleKind::IndependentPipelined &&
         stage.features.permitsSimdRoofline();
}

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

static double mte3ScalarStoreCycles(const StageModeProfile &profile) {
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

static bool isContinuousMemoryStage(StageCostModelKind kind) {
  return kind == StageCostModelKind::ContinuousTileMemory ||
         kind == StageCostModelKind::ContinuousTileStore ||
         kind == StageCostModelKind::ContinuousShortLoad ||
         kind == StageCostModelKind::CachePolicyStore;
}

struct LayoutCoverage {
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  double loadOperations = 0.0;
  double storeOperations = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  std::map<std::pair<int64_t, int64_t>, double> loadShapes;
  std::map<std::pair<int64_t, int64_t>, double> storeShapes;
};

static std::optional<int64_t> accessElements(const StageMemoryAccess &access) {
  if (access.shape.empty())
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t extent : access.shape) {
    if (extent <= 0 || elements > std::numeric_limits<int64_t>::max() / extent)
      return std::nullopt;
    elements *= extent;
  }
  return elements;
}

static std::optional<std::pair<int64_t, int64_t>>
accessShape(const StageMemoryAccess &access) {
  if (!access.hasStaticLayout() || access.elementBits % 8 != 0)
    return std::nullopt;
  const auto elements = accessElements(access);
  if (!elements)
    return std::nullopt;
  const int64_t elementBytes = access.elementBits / 8;
  if (elementBytes <= 0 ||
      *elements > std::numeric_limits<int64_t>::max() / elementBytes)
    return std::nullopt;
  return std::make_pair(*elements * elementBytes, *elements);
}

static int64_t stridedElementAddress(const StridedAddressStats &stats,
                                     int64_t flat) {
  if (stats.cols <= 0)
    return flat * stats.innerStrideBytes;
  const int64_t row = flat / stats.cols;
  const int64_t col = flat % stats.cols;
  return row * stats.rowPitchBytes + col * stats.innerStrideBytes;
}

StridedAddressStats
describeStridedShortAddresses(const StageMemoryAccess &access,
                              llvm::ArrayRef<int64_t> windows) {
  StridedAddressStats stats;
  auto shape = accessShape(access);
  if (!shape || access.elementBits <= 0 || access.elementBits % 8 != 0)
    return stats;
  stats.elementBytes = access.elementBits / 8;
  stats.elements = shape->second;
  // A rank-one access has exactly one shape axis; the rank-two geometry below
  // is the only part that needs a second axis.  Requiring two axes here would
  // reject the whole rank-one half of the calibrated domain.
  if (stats.elements <= 0 || access.shape.empty() ||
      access.strides.size() < access.shape.size())
    return stats;

  if (access.effectiveRank() <= 1) {
    stats.rows = 1;
    stats.cols = stats.elements;
    // A trailing singleton axis has stride 0, so take the pitch from the axis
    // that actually varies.
    int64_t innerStride = 0;
    for (size_t axis = 0; axis < access.shape.size(); ++axis)
      if (access.shape[axis] > 1 && access.strides[axis] &&
          (innerStride == 0 || *access.strides[axis] < innerStride))
        innerStride = *access.strides[axis];
    stats.innerStrideBytes =
        innerStride > 0 ? innerStride * stats.elementBytes : stats.elementBytes;
    stats.rowPitchBytes = 0;
  } else {
    if (access.shape.size() < 2)
      return stats;
    stats.rows = access.shape[0];
    stats.cols = access.shape[1];
    stats.innerStrideBytes = !access.strides[1]
                                 ? stats.elementBytes
                                 : *access.strides[1] * stats.elementBytes;
    stats.rowPitchBytes =
        !access.strides[0] ? 0 : *access.strides[0] * stats.elementBytes;
  }
  if (stats.innerStrideBytes <= 0 || stats.cols <= 0)
    return stats;
  stats.spanBytes =
      (stats.elements - 1) * stats.innerStrideBytes + stats.elementBytes;

  for (int64_t window : windows) {
    if (window <= 0)
      continue;
    std::map<int64_t, int64_t> counts;
    std::map<int64_t, std::set<int64_t>> blockRows;
    std::map<int64_t, std::map<int64_t, int64_t>> perRowCounts;
    for (int64_t flat = 0; flat < stats.elements; ++flat) {
      const int64_t address = stridedElementAddress(stats, flat);
      const int64_t block = address / window;
      const int64_t row = stats.cols > 0 ? flat / stats.cols : 0;
      ++counts[block];
      blockRows[block].insert(row);
      ++perRowCounts[block][row];
    }
    int64_t maximum = 0;
    int64_t squares = 0;
    int64_t rowBlocksTotal = 0;
    int64_t shared = 0;
    int64_t maxRow = 0;
    for (const auto &entry : counts) {
      maximum = std::max(maximum, entry.second);
      squares += entry.second * entry.second;
      rowBlocksTotal += static_cast<int64_t>(blockRows[entry.first].size());
      if (blockRows[entry.first].size() >= 2)
        ++shared;
      for (const auto &rowEntry : perRowCounts[entry.first])
        maxRow = std::max(maxRow, rowEntry.second);
    }
    stats.maxPerBlock[window] = maximum;
    stats.sumSquaresPerBlock[window] = squares;
    stats.distinctBlocks[window] = static_cast<int64_t>(counts.size());
    stats.rowBlocksSum[window] = rowBlocksTotal;
    stats.sharedBlocks[window] = shared;
    stats.maxPerRowBlock[window] = maxRow;
  }

  {
    std::map<int64_t, std::set<int64_t>> linesPerKilo;
    std::map<int64_t, std::set<int64_t>> rowsPerKilo;
    for (int64_t flat = 0; flat < stats.elements; ++flat) {
      const int64_t address = stridedElementAddress(stats, flat);
      linesPerKilo[address / 1024].insert(address / 128);
      rowsPerKilo[address / 1024].insert(stats.cols > 0 ? flat / stats.cols
                                                        : 0);
    }
    for (const auto &entry : linesPerKilo)
      stats.maxConc1k =
          std::max(stats.maxConc1k, static_cast<int64_t>(entry.second.size()));
    for (const auto &entry : rowsPerKilo)
      stats.maxRowsPerKilo = std::max(
          stats.maxRowsPerKilo, static_cast<int64_t>(entry.second.size()));
  }

  const int64_t waves = llvm::divideCeil(stats.elements, int64_t(32));
  if (waves > 0) {
    int64_t total = 0;
    for (int64_t wave = 0; wave < waves; ++wave) {
      std::set<int64_t> lines;
      for (int64_t lane = 0; lane < 32; ++lane) {
        const int64_t flat = wave * 32 + lane;
        if (flat >= stats.elements)
          break;
        lines.insert(stridedElementAddress(stats, flat) / 128);
      }
      total += static_cast<int64_t>(lines.size());
    }
    stats.meanWarpLines =
        static_cast<double>(total) / static_cast<double>(waves);
  }
  return stats;
}

/// Claim the StridedShort direct accesses of one Stage with the closed-form
/// completion formulas, recording the usual receipts so the analytical
/// fallback does not price the same bytes twice.
///
/// Guards:
///   * StridedShort - the calibrated layout class.
///   * masked       - admitted, except a SIMD store, which pays a destination
///                    read-back and therefore has its own two slots.
///   * no sync      - a synchronised Stage is not the measured pipeline.
///
/// A straight-line Stage is admissible too: the fits are per-iteration slopes,
/// which stay defined without a loop to amortise them over.
static void mapStridedShortMemoryCompletion(
    const LogicalStage &stage, const StageWorkload &work,
    const StageModeProfile &profile, StageResourceCycles &resources, bool simd,
    LayoutCoverage &layout, std::vector<const StageMemoryAccess *> &claimed) {
  if (profile.stridedShortCompletionFits.empty())
    return;
  if (stage.features.synchronizationCount != 0)
    return;
  // Static storage: binding an ArrayRef to a braced temporary is -Wdangling
  // and the tree builds with -Werror.
  static constexpr int64_t windowSizes[] = {64, 128, 256, 512, 1024};
  const llvm::ArrayRef<int64_t> windows = windowSizes;
  for (const StageMemoryAccess &access : work.memoryAccesses) {
    if (access.count <= 0.0)
      continue;
    if (access.layoutClass() != StageMemoryAccess::LayoutClass::StridedShort)
      continue;
    // Masked accesses are admitted: the load and SIMT store fits were measured
    // to be mask-invariant (the compaction form shows -0.6% masked versus
    // unmasked), and the two SIMD store fits are masked-only slots, so no
    // access can be priced by a fit that was swept under the other mask state.
    if (!access.hasStaticLayout())
      continue;
    if (access.addressDependsOnLoadedIndex ||
        access.addressHasNonAffineTensorComputation ||
        access.addressHasTensorControlFlowSelection)
      continue;
    if (access.effectiveRank() > 2)
      continue;
    if (access.elementBits / 8 <= 0)
      continue;
    // describeStridedShortAddresses owns the element pitch, so both the domain
    // check and the fit read the same number.
    const auto stats = describeStridedShortAddresses(access, windows);
    if (stats.elements <= 0)
      continue;
    // A unit inner stride is contiguous, not strided: that regime belongs to
    // the wide/contiguous path, which these formulas do not describe.
    if (stats.innerStrideBytes <= stats.elementBytes)
      continue;
    const auto shape = accessShape(access);
    if (!shape)
      continue;
    // Geometry decides the slot; the mode is the only non-geometry input and
    // travels as a bool, never as a name prefix.  This is the whole dispatch:
    // a slot that owns this geometry is priced, and a geometry that no slot
    // owns keeps the analytical fallback.
    const auto formula = selectStridedShortFormula(access, stats, simd);
    if (!formula)
      continue;
    auto fit = profile.stridedShortCompletionFits.find(
        stringifyStridedShortFormula(*formula).str());
    // Missing coefficients are missing data, not a reason to price the access
    // with a different formula: skip it and let the fallback own the bytes.
    if (fit == profile.stridedShortCompletionFits.end())
      continue;
    auto cycles = evaluateStridedShortCompletionFormula(
        *formula, fit->second, stats, profile.activeWarpCount);
    if (!cycles || *cycles <= 0.0)
      continue;
    const double total = *cycles * access.count;
    const double warpInstructions =
        access.count * llvm::divideCeil(shape->second, int64_t(32));
    // Receipt for the masked strided copy mechanism: it must yield on any
    // access this formula already priced, so the two mechanisms cannot both
    // charge the same bytes.
    claimed.push_back(&access);
    if (access.store) {
      resources.store += total;
      layout.storeBytes += access.count * shape->first;
      layout.storeOperations += access.count;
      layout.storeWarpInstructions += warpInstructions;
      resources.stridedShortStoreCompletion += total;
    } else {
      resources.load += total;
      layout.loadBytes += access.count * shape->first;
      layout.loadOperations += access.count;
      layout.loadWarpInstructions += warpInstructions;
      resources.stridedShortLoadCompletion += total;
    }
  }
}

/// The single string form of a slot: the profile JSON key.

} // namespace

namespace {

static void mapMaskedStridedCopyCompletion(
    const LogicalStage &stage, const StageWorkload &work,
    const StageModeProfile &profile, StageResourceCycles &resources,
    llvm::ArrayRef<const StageMemoryAccess *> claimed) {
  if (stage.costModelKind != StageCostModelKind::IndependentPipelinedLoop ||
      stage.features.synchronizationCount != 0 ||
      stage.features.conditionalBranchCount != 0 ||
      work.loadOperations != 1.0 || work.storeOperations != 1.0)
    return;

  const StageMemoryAccess *load = nullptr;
  const StageMemoryAccess *store = nullptr;
  for (const StageMemoryAccess &access : work.memoryAccesses) {
    if (access.count <= 0.0)
      continue;
    if (std::abs(access.count - 1.0) > 1.0e-9)
      return;
    const StageMemoryAccess **slot = access.store ? &store : &load;
    if (*slot)
      return;
    *slot = &access;
  }
  if (!load || !store || !load->masked || !store->masked ||
      load->elementBits != 32 || store->elementBits != 32 ||
      load->effectiveRank() != 1 || store->effectiveRank() != 1 ||
      !load->hasStaticLayout() || !store->hasStaticLayout() ||
      load->addressDependsOnLoadedIndex || store->addressDependsOnLoadedIndex ||
      load->addressHasNonAffineTensorComputation ||
      store->addressHasNonAffineTensorComputation ||
      load->addressHasTensorControlFlowSelection ||
      store->addressHasTensorControlFlowSelection ||
      store->maskedStoreHasMatchingLoadDependency)
    return;

  const auto loadShape = accessShape(*load);
  const auto storeShape = accessShape(*store);
  if (!loadShape || !storeShape || loadShape->second != storeShape->second)
    return;

  const auto contiguous = StageMemoryAccess::LayoutClass::ContiguousWide;
  const auto strided = StageMemoryAccess::LayoutClass::StridedShort;
  const StageMemoryAccess *stridedAccess = nullptr;
  const std::map<int64_t, MaskedStridedCopyCompletionCost> *curves = nullptr;
  double *resource = nullptr;
  if (load->layoutClass() == strided && store->layoutClass() == contiguous) {
    stridedAccess = load;
    curves = &profile.maskedStridedLoadCopyCompletionByStrideBytes;
    resource = &resources.load;
  } else if (load->layoutClass() == contiguous &&
             store->layoutClass() == strided) {
    stridedAccess = store;
    curves = &profile.maskedStridedStoreCopyCompletionByStrideBytes;
    resource = &resources.store;
  } else {
    return;
  }

  const auto curve = curves->find(stridedAccess->rowStrideBytes());
  if (curve == curves->end())
    return;
  // A strided access already priced by the closed-form strided-direct
  // mechanism yields this floor: otherwise both charge the same bytes, and the
  // measured absolute cost says the closed form is the more accurate of the
  // two.  Accesses it did not claim keep the floor exactly as before.
  if (llvm::is_contained(claimed, stridedAccess))
    return;
  const auto floor =
      curve->second.evaluate(stridedAccess->rowCount(), stage.iterationCount);
  if (floor)
    *resource = std::max(*resource, *floor);
}

static void mapSimdIrregularTileCompletion(const StageWorkload &work,
                                           const StageModeProfile &profile,
                                           StageResourceCycles &resources) {
  if (!profile.irregularTileCompletionEnabled || profile.vectorWidthBits <= 0)
    return;
  double maskedPhysicalElements = 0.0;
  double maskedActiveElements = 0.0;
  double tailInactiveElements = 0.0;
  double smallRowCompletionFloor = 0.0;
  int64_t maskedTileRows = 0;
  for (const StageMemoryAccess &access : work.memoryAccesses) {
    auto shape = accessShape(access);
    if (!shape || access.elementBits <= 0 ||
        profile.vectorWidthBits % access.elementBits != 0 ||
        access.shape.empty())
      continue;
    const int64_t lanes = profile.vectorWidthBits / access.elementBits;
    const int64_t inner = access.shape.back();
    if (lanes <= 0 || inner < lanes)
      continue;
    const double elements = static_cast<double>(shape->second) * access.count;
    if (access.masked && access.activeElementRatio < 1.0 - 1.0e-9) {
      if (elements >= maskedPhysicalElements) {
        maskedPhysicalElements = elements;
        maskedActiveElements =
            elements * std::clamp(access.activeElementRatio, 0.0, 1.0);
        maskedTileRows = access.rowCount();
      }
    }
    const int64_t paddedInner = llvm::alignTo(inner, lanes);
    const double outer = static_cast<double>(shape->second) / inner;
    tailInactiveElements = std::max(
        tailInactiveElements, access.count * outer * (paddedInner - inner));

    // Full aligned row groups expose a completion bound not visible in TTIR
    // byte counts.
    const int64_t rows = access.rowCount();
    const bool smallWideTile =
        profile.irregularTileRowGroupSize > 0 &&
        profile.irregularTileSmallRowMaximumRows > 0 && rows > 0 &&
        rows <= profile.irregularTileSmallRowMaximumRows && inner >= 2 * lanes;
    if (smallWideTile) {
      const bool unalignedInner = paddedInner != inner;
      const bool partialRowGroup =
          rows % profile.irregularTileRowGroupSize != 0;
      double accessFloor = 0.0;
      if (!partialRowGroup && !unalignedInner) {
        // Full aligned row groups have a completion envelope that is not
        // visible in TTIR GM bytes.
        const int64_t fullGroups = rows / profile.irregularTileRowGroupSize;
        const double elementsPerGroup =
            access.count * profile.irregularTileRowGroupSize * inner;
        accessFloor =
            elementsPerGroup *
            (profile.irregularTileAlignedFirstRowGroupCyclesPerElement +
             std::max<int64_t>(0, fullGroups - 1) *
                 profile
                     .irregularTileAlignedAdditionalRowGroupCyclesPerElement);
      }
      smallRowCompletionFloor = std::max(smallRowCompletionFloor, accessFloor);
    }
  }
  const double maskedFloorScale =
      maskedTileRows > profile.irregularTileSmallRowMaximumRows
          ? profile.irregularTileLargeRowMaskedFloorScale
          : 1.0;
  const double fullPhysicalMaterialization =
      maskedFloorScale * profile.irregularTilePhysicalExcessCyclesPerElement *
      std::max(0.0, maskedPhysicalElements -
                        profile.irregularTilePhysicalThresholdElements);
  if (maskedPhysicalElements > 0.0) {
    resources.irregularFullPhysicalMaterialization =
        fullPhysicalMaterialization;
    // The masked-tile completion floor always owns the complete physical
    // materialization.
    const double groupRows =
        static_cast<double>(profile.irregularTileRowGroupSize);
    double uncoveredFraction =
        groupRows > 0.0
            ? std::clamp((2.0 * groupRows - maskedTileRows) / groupRows, 0.0,
                         1.0)
            : 1.0;
    if (groupRows > 0.0 &&
        maskedTileRows == 2 * profile.irregularTileRowGroupSize)
      uncoveredFraction =
          profile.irregularTileAlignedTwoRowGroupPhysicalMaterializationScale;
    resources.irregularPhysicalMaterialization =
        fullPhysicalMaterialization * uncoveredFraction;
    resources.irregularMaskedTileFloor =
        fullPhysicalMaterialization +
        maskedFloorScale * profile.irregularTileActiveCyclesPerElement *
            maskedActiveElements;
  }
  resources.irregularMaskedTileFloor =
      std::max(resources.irregularMaskedTileFloor, smallRowCompletionFloor);
  resources.irregularTailMaterialization =
      profile.irregularTileTailCyclesPerInactiveElement * tailInactiveElements;
}

/// Only consume exact group evidence.
static std::optional<double> mapLayoutAccess(const StageMemoryAccess &access,
                                             const StageModeProfile &profile,
                                             bool simd, int64_t factor) {
  if (!accessShape(access) ||
      access.layoutClass() == StageMemoryAccess::LayoutClass::Unknown)
    return std::nullopt;
  const auto &profileCosts =
      access.store ? profile.storeLayoutCosts : profile.loadLayoutCosts;
  const std::string key = access.layoutKey() + ":warps=" +
                          std::to_string(simd ? 0 : profile.activeWarpCount) +
                          ":factor=" + std::to_string(simd ? 1 : factor);
  auto measured = profileCosts.find(key);
  return measured == profileCosts.end()
             ? std::nullopt
             : measured->second.evaluate(access.count);
}

static const StageMemoryLayoutCost *
dotStoreSharedCurve(const StageMemoryAccess &access,
                    const StageModeProfile &profile, int64_t factor) {
  if (!access.store || !access.fromDotAccumulator || factor <= 0 ||
      access.shape.size() != 2 || !access.hasStaticLayout() ||
      !access.strides.back() || *access.strides.back() != 1 ||
      profile.activeWarpCount % factor != 0)
    return nullptr;
  const std::string memoryKey =
      access.layoutKey() + ":warps=" + std::to_string(profile.activeWarpCount) +
      ":factor=" + std::to_string(factor);
  auto memory = profile.storeLayoutCosts.find(memoryKey);
  if (memory != profile.storeLayoutCosts.end() &&
      memory->second.includesSharedLayout)
    return nullptr;
  const std::string key =
      "fma_to_row_major:" + std::to_string(access.shape[0]) + "x" +
      std::to_string(access.shape[1]) +
      ":bits=" + std::to_string(access.elementBits) +
      ":base_warps=" + std::to_string(profile.activeWarpCount / factor) +
      ":factor=" + std::to_string(factor);
  auto found = profile.dotStoreSharedLayoutCosts.find(key);
  return found == profile.dotStoreSharedLayoutCosts.end() ? nullptr
                                                          : &found->second;
}

static double storeLayoutService(const StageWorkload &work,
                                 const StageModeProfile &profile,
                                 int64_t factor) {
  const int64_t unit = profile.storeFootprintUnitBytes;
  auto floor = profile.storeIssueFloorByWarpCount.find(profile.activeWarpCount);
  if (unit <= 0 || factor <= 0 ||
      floor == profile.storeIssueFloorByWarpCount.end())
    return 0.0;
  double groupService = 0.0;
  for (const auto &access : work.memoryAccesses) {
    auto shape = accessShape(access);
    if (!access.store || !shape || access.effectiveRank() > 2 ||
        access.isFlatContiguous() || !access.hasUnitInnerStride() ||
        static_cast<double>(shape->first) * factor !=
            static_cast<double>(profile.activeWarpCount) * 32 * 8)
      continue;
    int64_t units = 0;
    if (access.rowStrideBytes() >= access.rowBytes() &&
        access.rowStrideBytes() % unit == 0)
      units = access.rowCount() * llvm::divideCeil(access.rowBytes(), unit);
    if (units > 0)
      groupService +=
          access.count *
          std::max(floor->second, static_cast<double>(units) * factor *
                                      profile.storeCyclesPerFootprintUnit);
  }
  return groupService / factor;
}

struct SegmentedSimdInstructionSummary {
  llvm::StringMap<double> logicalElements;
  llvm::StringMap<double> instructions;
};

static SegmentedSimdInstructionSummary
summarizeSegmentedSimdInstructions(const StageWorkload &work,
                                   int64_t vectorWidthBits) {
  SegmentedSimdInstructionSummary summary;
  if (vectorWidthBits <= 0)
    return summary;
  for (const TensorOperationWorkload &tensor : work.tensorOperationWorkloads) {
    summary.logicalElements[tensor.operation] += tensor.logicalElements;
    const double segmentBits =
        static_cast<double>(tensor.contiguousElementsPerSegment) *
        static_cast<double>(tensor.elementBitWidth);
    summary.instructions[tensor.operation] +=
        tensor.segmentCount *
        std::ceil(segmentBits / static_cast<double>(vectorWidthBits));
  }
  return summary;
}

static bool fullyDescribed(double described, double total) {
  return std::abs(described - total) <= 1e-9 * std::max(1.0, total);
}

/// Count issue instructions without collapsing distinct short operations into
/// one vector/warp.
static double countIssueInstructions(const StageWorkload &work,
                                     int64_t issueWidth, bool simd,
                                     int64_t vectorWidthBits) {
  if (issueWidth <= 0)
    return 0.0;
  const double width = static_cast<double>(issueWidth);
  double instructions =
      work.scalarOperations + work.scalarLoadCount + work.scalarStoreCount;
  const SegmentedSimdInstructionSummary segmented =
      summarizeSegmentedSimdInstructions(work, vectorWidthBits);
  for (const auto &entry : work.operationCounts) {
    if (simd && entry.first() == "simd.folded_flat_address")
      continue;
    double totalElements = 0.0;
    for (const auto &[elements, count] : entry.second)
      totalElements += static_cast<double>(elements) * count;
    if (simd && fullyDescribed(segmented.logicalElements.lookup(entry.first()),
                               totalElements)) {
      instructions += segmented.instructions.lookup(entry.first());
      continue;
    }
    for (const auto &[elements, count] : entry.second)
      instructions += count * std::ceil(static_cast<double>(elements) / width);
  }
  double predicateElements = 0.0;
  for (const auto &[elements, count] : work.predicateCounts)
    predicateElements += static_cast<double>(elements) * count;
  if (simd && fullyDescribed(segmented.logicalElements.lookup("predicate.cmp"),
                             predicateElements)) {
    instructions += segmented.instructions.lookup("predicate.cmp");
  } else {
    for (const auto &[elements, count] : work.predicateCounts)
      instructions += count * std::ceil(static_cast<double>(elements) / width);
  }

  // These are normalized element-work units, not a count of final LDG/STG.
  // Without a lowering contract, row geometry cannot change that unit.
  auto countMemory = [&](const auto &shapes, double &coveredWarpInstructions) {
    for (const auto &[shape, count] : shapes) {
      const int64_t warpInstructions =
          llvm::divideCeil(shape.second, int64_t(32));
      instructions += count * std::ceil(32.0 * warpInstructions / width);
      coveredWarpInstructions += count * warpInstructions;
    }
  };
  double coveredLoadWarpInstructions = 0.0;
  double coveredStoreWarpInstructions = 0.0;
  countMemory(work.continuousLoadCounts, coveredLoadWarpInstructions);
  countMemory(work.indirectLoadCounts, coveredLoadWarpInstructions);
  for (const auto &[shape, count] : work.segmentedIndirectLoadCounts) {
    const int64_t elements = std::get<1>(shape);
    const int64_t warpInstructions = llvm::divideCeil(elements, int64_t(32));
    instructions += count * std::ceil(32.0 * warpInstructions / width);
    coveredLoadWarpInstructions += count * warpInstructions;
  }
  countMemory(work.continuousStoreCounts, coveredStoreWarpInstructions);
  countMemory(work.indirectStoreCounts, coveredStoreWarpInstructions);
  for (const auto &[shape, count] : work.segmentedIndirectStoreCounts) {
    const int64_t elements = std::get<1>(shape);
    const int64_t warpInstructions = llvm::divideCeil(elements, int64_t(32));
    instructions += count * std::ceil(32.0 * warpInstructions / width);
    coveredStoreWarpInstructions += count * warpInstructions;
  }
  instructions += std::ceil(
      32.0 *
      std::max(0.0, work.loadWarpInstructions - coveredLoadWarpInstructions) /
      width);
  instructions += std::ceil(
      32.0 *
      std::max(0.0, work.storeWarpInstructions - coveredStoreWarpInstructions) /
      width);
  instructions +=
      work.sharedLocalLoadInstructions + work.sharedLocalStoreInstructions +
      work.privateStackLoadInstructions + work.privateStackStoreInstructions;
  instructions +=
      work.loopBackedges + work.conditionalBranches + work.synchronizations;
  return instructions;
}

/// Interpolate an isolated per-operation cost curve.
static double interpolateOperationCost(const std::map<int64_t, double> &curve,
                                       int64_t workSize) {
  if (curve.empty() || workSize <= 0)
    return 0.0;
  auto upper = curve.lower_bound(workSize);
  if (upper == curve.begin())
    return upper->second * static_cast<double>(workSize) / upper->first;
  if (upper == curve.end()) {
    auto last = std::prev(curve.end());
    if (last == curve.begin())
      return last->second * static_cast<double>(workSize) / last->first;
    auto previous = std::prev(last);
    const double slope = (last->second - previous->second) /
                         static_cast<double>(last->first - previous->first);
    return std::max(0.0, last->second + (workSize - last->first) * slope);
  }
  auto lower = std::prev(upper);
  const double ratio = static_cast<double>(workSize - lower->first) /
                       static_cast<double>(upper->first - lower->first);
  return lower->second + ratio * (upper->second - lower->second);
}

/// Continuous-memory curves are only evidence inside their measured byte
/// interval.
static std::optional<double>
interpolateBoundedOperationCost(const std::map<int64_t, double> &curve,
                                int64_t workSize) {
  if (curve.empty() || workSize < curve.begin()->first ||
      workSize > curve.rbegin()->first)
    return std::nullopt;
  return interpolateOperationCost(curve, workSize);
}

/// SIMT row groups execute the complete backend template selected for the
/// enclosing row bucket.
static std::optional<double>
lookupBoundedCeilingGroupCost(const std::map<int64_t, double> &curve,
                              int64_t workSize) {
  if (curve.empty() || workSize < curve.begin()->first ||
      workSize > curve.rbegin()->first)
    return std::nullopt;
  auto upper = curve.lower_bound(workSize);
  if (upper == curve.end())
    return std::nullopt;
  return upper->second;
}

static llvm::StringRef multiResultReductionTailClass(int64_t extent) {
  if (extent <= 0)
    return {};
  if (llvm::isPowerOf2_64(static_cast<uint64_t>(extent)))
    return "power_of_two";
  if (extent % 4 == 0)
    return "non_power_divisible_by_4";
  if (extent % 2 == 0)
    return "tail_mod_4_eq_2";
  return "tail_odd";
}

static std::optional<double> interpolateBoundedRowExtentSurface(
    const std::map<int64_t, std::map<int64_t, double>> &surface, int64_t rows,
    int64_t extent) {
  if (surface.empty() || rows < surface.begin()->first ||
      rows > surface.rbegin()->first)
    return std::nullopt;
  auto upper = surface.lower_bound(rows);
  if (upper == surface.end())
    return std::nullopt;
  auto upperValue = interpolateBoundedOperationCost(upper->second, extent);
  if (!upperValue)
    return std::nullopt;
  if (upper->first == rows)
    return upperValue;
  if (upper == surface.begin())
    return std::nullopt;
  auto lower = std::prev(upper);
  auto lowerValue = interpolateBoundedOperationCost(lower->second, extent);
  if (!lowerValue)
    return std::nullopt;
  const double ratio = static_cast<double>(rows - lower->first) /
                       static_cast<double>(upper->first - lower->first);
  return *lowerValue + ratio * (*upperValue - *lowerValue);
}

static std::optional<int64_t> exactNonNegativeWorkCount(double value) {
  if (!std::isfinite(value) || value < 0.0 ||
      value > static_cast<double>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  const double rounded = std::round(value);
  if (std::abs(value - rounded) > 1.0e-9)
    return std::nullopt;
  return static_cast<int64_t>(rounded);
}

static std::optional<std::pair<int64_t, int64_t>>
rowwiseReductionGeometry(const LogicalStage &stage, const StageWorkload &work,
                         int64_t referenceReductionOperations) {
  if (stage.costModelKind != StageCostModelKind::RowwiseReduction ||
      !stage.features.hasReduction || stage.features.hasDot ||
      stage.features.hasIndirectMemory || referenceReductionOperations <= 0 ||
      std::abs(work.reductionOperations -
               static_cast<double>(referenceReductionOperations)) > 1.0e-9)
    return std::nullopt;
  if (work.reductionGeometryCounts.size() == 1) {
    const auto &[shape, count] = *work.reductionGeometryCounts.begin();
    if (std::get<2>(shape) == 32 &&
        std::abs(count - static_cast<double>(referenceReductionOperations)) <=
            1.0e-9)
      return std::make_pair(std::get<0>(shape), std::get<1>(shape));
  }
  const StageMemoryAccess *input = nullptr;
  for (const StageMemoryAccess &access : work.memoryAccesses) {
    if (access.store)
      continue;
    if (input || access.elementBits != 32 || access.effectiveRank() != 2 ||
        !access.hasUnitInnerStride() || access.count <= 0.0)
      return std::nullopt;
    input = &access;
  }
  if (!input || input->rowCount() <= 0 || input->rowBytes() <= 0 ||
      input->contiguousAxisBytes() != input->rowBytes() ||
      input->rowBytes() % 4 != 0)
    return std::nullopt;
  return std::make_pair(input->rowCount(), input->rowBytes() / 4);
}

static double interpolateEnvelopeCost(const std::map<int64_t, double> &curve,
                                      int64_t workSize) {
  if (curve.empty() || workSize <= 0)
    return 0.0;
  auto upper = curve.lower_bound(workSize);
  if (upper == curve.begin())
    return upper->second;
  if (upper == curve.end())
    return curve.rbegin()->second;
  auto lower = std::prev(upper);
  const double ratio = static_cast<double>(workSize - lower->first) /
                       static_cast<double>(upper->first - lower->first);
  return lower->second + ratio * (upper->second - lower->second);
}

struct SimtSourceFeatures {
  double segments = 0.0;
  double strideRatio = 0.0;
  double masked = 0.0;
  double unrollFactor = 1.0;
};

static SimtSourceFeatures sourceFeatures(const StageWorkload &work,
                                         const LogicalStage &stage, bool store,
                                         bool useWholeSimtPressure) {
  SimtSourceFeatures features;
  for (const StageMemoryAccess &access : work.memoryAccesses) {
    if (access.store != store)
      continue;
    features.segments +=
        access.count * std::max<int64_t>(1, access.descriptorCount(2));
    if (access.rowBytes() > 0 && access.rowStrideBytes() > access.rowBytes())
      features.strideRatio +=
          access.count *
          (static_cast<double>(access.rowStrideBytes()) / access.rowBytes() -
           1.0);
    if (access.masked)
      features.masked += access.count;
  }
  if (useWholeSimtPressure && stage.wholeSimtRegisterPressure)
    features.unrollFactor = std::max(
        1.0, static_cast<double>(
                 stage.wholeSimtRegisterPressure->backendUnrollFactor));
  return features;
}

static double derivedRegisterBudget(const StageModeProfile &profile) {
  if (profile.registerBudgetPerThread > 0.0)
    return profile.registerBudgetPerThread;
  if (profile.activeWarpCount <= 0)
    return 0.0;
  return std::min(
      127.0, std::floor(1024.0 / static_cast<double>(profile.activeWarpCount)));
}

static double peakLiveExcess(const LogicalStage &stage,
                             const StageModeProfile &profile) {
  if (!stage.wholeSimtRegisterPressure)
    return 0.0;
  const double effectivePeak = std::max(
      static_cast<double>(
          stage.wholeSimtRegisterPressure->peakLiveUnitsPerThread),
      static_cast<double>(
          stage.wholeSimtRegisterPressure->loweredPeakLiveUnitsPerThread));
  return std::max(0.0, effectivePeak - derivedRegisterBudget(profile));
}

static double sourceFitInstructions(const SimtSourceFit &fit,
                                    const StageWorkload &work,
                                    const LogicalStage &stage,
                                    const StageModeProfile &profile, bool store,
                                    double direct, bool useWholeSimtPressure) {
  SimtSourceFeatures features =
      sourceFeatures(work, stage, store, useWholeSimtPressure);
  const double excess =
      useWholeSimtPressure ? peakLiveExcess(stage, profile) : 0.0;
  if (!fit.enabled || (features.segments == 0.0 && direct == 0.0 &&
                       excess == 0.0 && features.unrollFactor <= 1.0))
    return 0.0;
  return fit.evaluate(direct, features.segments, features.strideRatio,
                      features.masked, excess, features.unrollFactor);
}

static const DotComponentPrivateStackFloor *
matchingDotComponentPrivateStackFloor(const LogicalStage &stage,
                                      const StageModeProfile &profile,
                                      int64_t superblockFactor,
                                      bool wholeSimt) {
  if (!wholeSimt || stage.iterationCount != 1 ||
      stage.workload.privateStackLoadInstructions > 0.0 ||
      stage.workload.privateStackStoreInstructions > 0.0 ||
      !stage.wholeSimtRegisterPressure)
    return nullptr;
  const SimtRegisterPressure &facts = *stage.wholeSimtRegisterPressure;
  if (facts.dotComponentPrivateStackChargeRegion ==
          SimtPrivateStackChargeRegion::None ||
      !facts.ttirMaximumWeakDotComponentGeometryKnown)
    return nullptr;
  for (const auto &floor : profile.dotComponentPrivateStackFloors) {
    if (!floor.enabled || profile.activeWarpCount != floor.activeWarpCount ||
        superblockFactor != floor.superblockFactor ||
        facts.ttirMaximumWeakDotComponentNodes < floor.minimumComponentNodes ||
        facts.ttirMaximumWeakDotComponentNodes > floor.maximumComponentNodes ||
        facts.ttirMaximumWeakDotComponentM != floor.m ||
        facts.ttirMaximumWeakDotComponentN != floor.n ||
        facts.ttirMaximumWeakDotComponentK != floor.k ||
        facts.ttirMaximumWeakDotComponentElementBits != floor.elementBits ||
        facts.ttirFunctionDotUseSignature != floor.dotUseSignature)
      continue;
    return &floor;
  }
  return nullptr;
}

static double
privateStackFloorFraction(const SimtRegisterPressure &facts,
                          const DotComponentPrivateStackFloor &floor) {
  switch (facts.dotComponentPrivateStackChargeRegion) {
  case SimtPrivateStackChargeRegion::WholeFunction:
    return 1.0;
  case SimtPrivateStackChargeRegion::BeforeRecurrence:
    return floor.preRecurrenceFraction;
  case SimtPrivateStackChargeRegion::DotRegion:
    return floor.dotRegionFraction;
  case SimtPrivateStackChargeRegion::AfterDot:
    return floor.postDotFraction;
  case SimtPrivateStackChargeRegion::None:
    return 0.0;
  }
  llvm_unreachable("unknown private-stack charge region");
}

static double generatedRate(double explicitRate, double fallback) {
  return explicitRate > 0.0 ? explicitRate : fallback;
}

static bool extrapolatesIndirectEvidence(const LogicalStage &stage,
                                         const StageModeProfile &profile) {
  if (!stage.features.hasIndirectMemory ||
      profile.indirectLoadSystemCyclesByWarpInstructions.empty())
    return false;
  const int64_t first =
      profile.indirectLoadSystemCyclesByWarpInstructions.begin()->first;
  const int64_t last =
      profile.indirectLoadSystemCyclesByWarpInstructions.rbegin()->first;
  double operationCount = 0.0;
  for (const auto &[shape, count] : stage.workload.indirectLoadCounts) {
    const int64_t warpInstructions =
        llvm::divideCeil(shape.second, int64_t(32));
    if (warpInstructions < first || warpInstructions > last)
      return true;
    operationCount += count;
  }
  // The checked-in count sweep is fitted on 1, 2 and 4 operations.  Count=8
  // is a held-out capacity check, not a calibrated linear domain.
  return operationCount > 4.0;
}

static const StageMemoryLayoutCost *
dotOperandCurve(const std::string &path, const StageModeProfile &profile,
                int64_t factor) {
  if (factor <= 0 || profile.activeWarpCount % factor != 0)
    return nullptr;
  const std::string key =
      path + ":base_warps=" + std::to_string(profile.activeWarpCount / factor) +
      ":factor=" + std::to_string(factor);
  auto found = profile.dotOperandPathCosts.find(key);
  const auto isLinear = [](const StageMemoryLayoutCost &curve) {
    return curve.minCount <= 1 && curve.firstCycles == curve.incrementalCycles;
  };
  if (found != profile.dotOperandPathCosts.end() && isLinear(found->second))
    return &found->second;

  // The measured dot composite describes a 16x16 operand tile and its producer
  // topology.
  const auto eraseOuterPitch = [](std::string value) {
    for (llvm::StringRef marker :
         {llvm::StringRef("stride="), llvm::StringRef("strides=")}) {
      size_t position = 0;
      while ((position = value.find(marker.str(), position)) !=
             std::string::npos) {
        const size_t begin = position + marker.size();
        size_t end = begin;
        if (end < value.size() && (value[end] == '-' || value[end] == '+'))
          ++end;
        while (end < value.size() &&
               std::isdigit(static_cast<unsigned char>(value[end])))
          ++end;
        if (end == begin || (end == begin + 1 &&
                             (value[begin] == '-' || value[begin] == '+'))) {
          position = begin;
          continue;
        }
        value.replace(begin, end - begin, "*");
        position = begin + 1;
      }
    }
    return value;
  };
  const std::string topology = eraseOuterPitch(key);
  const StageMemoryLayoutCost *topologyMatch = nullptr;
  for (const auto &[candidateKey, candidateCurve] :
       profile.dotOperandPathCosts) {
    if (!isLinear(candidateCurve) || eraseOuterPitch(candidateKey) != topology)
      continue;
    if (topologyMatch)
      return nullptr;
    topologyMatch = &candidateCurve;
  }
  return topologyMatch;
}

/// Transfer ownership, never add a composite dot cost on top of its inputs.
static void mapCompositeDotPaths(StageWorkload &work,
                                 const StageModeProfile &profile,
                                 int64_t factor,
                                 StageResourceCycles &resources) {
  bool matchedOperandPath = false;
  for (const auto &[path, entry] : work.dotOperandPaths) {
    const auto *curve = dotOperandCurve(path, profile, factor);
    if (!curve)
      continue;
    matchedOperandPath = true;
    // Linear per-dot coefficient; Stage grouping cannot change ownership.
    resources.dotOperandGroup += curve->firstCycles * entry.second;
    work.dotOperations = std::max(0.0, work.dotOperations - entry.second);
    work.dotFlops = std::max(0.0, work.dotFlops - entry.first * entry.second);
  }
  if (matchedOperandPath && (work.dotOperations > 0.0 || work.dotFlops > 0.0)) {
    work.dotOperations *= profile.dotPartialCompositeResidualRelativeScale;
    work.dotFlops *= profile.dotPartialCompositeResidualRelativeScale;
  }
  for (auto &access : work.memoryAccesses) {
    auto shape = accessShape(access);
    if (access.store || !shape || access.dotConsumerPaths.empty() ||
        !llvm::all_of(access.dotConsumerPaths, [&](const std::string &path) {
          const auto *curve = dotOperandCurve(path, profile, factor);
          // Load ownership must not depend on how many dots the partitioner
          // groups together. Only linear per-dot curves admit this transfer.
          return curve && curve->minCount <= 1 &&
                 curve->firstCycles == curve->incrementalCycles;
        }))
      continue;
    const double count = access.count;
    const double warps = llvm::divideCeil(shape->second, int64_t(32)) * count;
    work.loadBytes = std::max(0.0, work.loadBytes - shape->first * count);
    work.loadOperations = std::max(0.0, work.loadOperations - count);
    work.loadWarpInstructions =
        std::max(0.0, work.loadWarpInstructions - warps);
    work.issueElements = std::max(0.0, work.issueElements - 32 * warps);
    if (access.isFlatContiguous()) {
      auto found = work.continuousLoadCounts.find(*shape);
      if (found != work.continuousLoadCounts.end())
        found->second = std::max(0.0, found->second - count);
    }
    resources.dotInputLoadsCovered += count;
    access.count = 0.0;
  }
}

static std::optional<StageResourceCycles>
mapWorkload(const LogicalStage &stage, const StageModeProfile &profile,
            StageMode mode, int64_t superblockFactor, bool wholeSimt) {
  StageResourceCycles resources;
  // Accesses priced by the closed-form strided-direct mechanism.  The masked
  // strided copy floor yields on them and stays in force for everything else.
  std::vector<const StageMemoryAccess *> stridedShortAlreadyClaimed;
  StageWorkload work = stage.workload;
  if (wholeSimt)
    mapCompositeDotPaths(work, profile, superblockFactor, resources);
  const bool simd = mode == StageMode::SIMD;
  resources.setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
  if (simd)
    mapSimdIrregularTileCompletion(work, profile, resources);
  if (work.scalarLoadCount > 0.0)
    resources.load +=
        simd ? mainScalarLoadCycles(work.scalarLoadCount, profile)
             : simtUniformLoadCycles(work.scalarLoadCount, profile);
  if (work.scalarStoreCount > 0.0)
    resources.store +=
        simd ? mte3ScalarStoreCycles(profile) : simtUniformStoreCycles(profile);
  for (const AtomicWorkload &atomic : work.atomicWorkloads) {
    const llvm::StringRef addressClass =
        atomic.addressDependsOnLoadedIndex
            ? "loaded_index"
            : (atomic.provenContiguousRunWidth > 1.0 ? "contiguous"
                                                     : "unknown_address");
    auto rate = profile.atomicRates.find(atomic.profileKey() + "." +
                                         addressClass.str());
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find(addressClass);
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find(atomic.profileKey());
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find("default");
    if (rate == profile.atomicRates.end())
      continue;
    const StageAtomicRate &atomicRate = rate->second;
    const double base =
        atomic.logicalOperationInstances * atomicRate.operationStartupCycles +
        atomic.logicalElements / atomicRate.logicalElementsPerCycle;
    const double singleProgramRate =
        atomicRate.singlePhysicalProgramLogicalElementsPerCycle > 0.0
            ? atomicRate.singlePhysicalProgramLogicalElementsPerCycle
            : atomicRate.logicalElementsPerCycle;
    const double singleProgramBase =
        atomic.logicalOperationInstances * atomicRate.operationStartupCycles +
        atomic.logicalElements / singleProgramRate;
    const double contention =
        atomic.contentionUnknown ? atomicRate.unknownContentionMultiplier : 1.0;
    resources.atomic += base * contention;
    resources.singlePhysicalProgramAtomic += singleProgramBase * contention;
    resources.hasSinglePhysicalProgramAtomicRate |=
        atomicRate.singlePhysicalProgramLogicalElementsPerCycle > 0.0;
    if (atomic.resultUsed) {
      resources.atomic +=
          atomic.logicalOperationInstances * atomicRate.resultDependencyCycles;
      resources.singlePhysicalProgramAtomic +=
          atomic.logicalOperationInstances * atomicRate.resultDependencyCycles;
    }
  }
  const SegmentedSimdInstructionSummary segmented =
      summarizeSegmentedSimdInstructions(work, profile.vectorWidthBits);
  for (const auto &[name, counts] : work.operationCounts) {
    const StageOperationRate *rate = findOperationRate(profile, name);
    if (!rate || rate->throughput <= 0.0)
      continue;
    if (rate->gridStrideKernelStartupCycles > 0.0)
      resources.gridStrideStartup = std::max(
          resources.gridStrideStartup, rate->gridStrideKernelStartupCycles);
    double totalElements = 0.0;
    for (const auto &[elements, count] : counts)
      totalElements += static_cast<double>(elements) * count;
    if (simd && rate->startupCycles == 0.0 && rate->freeWorkUnits == 0.0 &&
        fullyDescribed(segmented.logicalElements.lookup(name), totalElements)) {
      resources.compute +=
          segmented.instructions.lookup(name) / rate->throughput * rate->factor;
      continue;
    }
    for (const auto &[elements, count] : counts) {
      const double workUnits =
          simd ? std::ceil(static_cast<double>(elements) / profile.vectorWidth)
               : static_cast<double>(elements);
      if (elements >= rate->minimumStartupElements)
        resources.operationCompletion += count * rate->startupCycles;
      resources.compute += count *
                           std::max(0.0, workUnits - rate->freeWorkUnits) /
                           rate->throughput * rate->factor;
    }
  }
  double prefixScanIssuedInstructions = 0.0;
  for (const auto &[shape, count] : work.prefixScanCounts) {
    const int64_t extent = std::get<0>(shape);
    const int64_t independentSequences = std::get<1>(shape);
    const int64_t axisStride = std::get<2>(shape);
    const int64_t resultCount = std::get<3>(shape);
    const bool reverse = std::get<4>(shape);
    const std::string &combineClass = std::get<5>(shape);
    if (extent <= 1 || independentSequences <= 0 || count <= 0.0)
      continue;
    auto addRate = profile.operationRates.find("f32.add");
    if (addRate == profile.operationRates.end() ||
        addRate->second.throughput <= 0.0)
      continue;
    if (stage.features.isGridStrideLoopShell ||
        stage.features.insideGridStrideLoopBody) {
      const auto &factorBodyCurves =
          reverse ? profile.gridStrideReverseScanCyclesByFactorAndExtent
                  : profile.gridStrideForwardScanCyclesByFactorAndExtent;
      const auto &factorStartupCurves =
          reverse ? profile.gridStrideReverseScanStartupCyclesByFactorAndExtent
                  : profile.gridStrideForwardScanStartupCyclesByFactorAndExtent;
      const auto factorBody = factorBodyCurves.find(superblockFactor);
      const auto factorStartup = factorStartupCurves.find(superblockFactor);
      const auto &bodyCurve =
          factorBody != factorBodyCurves.end()
              ? factorBody->second
              : (reverse ? profile.gridStrideReverseScanCyclesByExtent
                         : profile.gridStrideForwardScanCyclesByExtent);
      const auto &startupCurve =
          factorStartup != factorStartupCurves.end()
              ? factorStartup->second
              : (reverse ? profile.gridStrideReverseScanStartupCyclesByExtent
                         : profile.gridStrideForwardScanStartupCyclesByExtent);
      if (!bodyCurve.empty()) {
        resources.gridStridePrefixScanBody +=
            count * independentSequences *
            interpolateOperationCost(bodyCurve, extent);
        if (factorBody != factorBodyCurves.end())
          resources.gridStrideFactorSpecificResourceBody = true;
        if (!startupCurve.empty())
          resources.gridStrideStartup =
              std::max(resources.gridStrideStartup,
                       interpolateEnvelopeCost(startupCurve, extent));
      }
    }
    if (simd) {
      if (stage.scheduleKind == StageScheduleKind::StraightLine &&
          !stage.features.isGridStrideLoopShell &&
          !stage.features.insideGridStrideLoopBody) {
        auto helper =
            profile.straightLinePrefixScanCostsByCombine.find(combineClass);
        if (helper != profile.straightLinePrefixScanCostsByCombine.end()) {
          const int64_t semanticElementCount = extent * independentSequences;
          const double semanticElements =
              static_cast<double>(semanticElementCount);
          const bool useNonUnitStrideCompletion =
              axisStride > 1 && resultCount == 1 &&
              helper->second.nonUnitStrideElementsPerCycle > 0.0 &&
              extent >= helper->second.nonUnitStrideMinimumAxisExtent &&
              extent <= helper->second.nonUnitStrideMaximumAxisExtent &&
              axisStride >= helper->second.nonUnitStrideMinimumAxisStride &&
              axisStride <= helper->second.nonUnitStrideMaximumAxisStride &&
              semanticElementCount >=
                  helper->second.nonUnitStrideMinimumSemanticElements &&
              semanticElementCount <=
                  helper->second.nonUnitStrideMaximumSemanticElements;
          const bool useSingleSequenceCurve =
              independentSequences == 1 &&
              !helper->second.singleSequenceCompletionCyclesByExtent.empty();
          const double completion =
              useNonUnitStrideCompletion
                  ? helper->second.nonUnitStrideStartupCycles +
                        semanticElements /
                            helper->second.nonUnitStrideElementsPerCycle
              : useSingleSequenceCurve
                  ? interpolateOperationCost(
                        helper->second.singleSequenceCompletionCyclesByExtent,
                        extent)
                  : helper->second.startupCycles +
                        semanticElements / helper->second.elementsPerCycle +
                        static_cast<double>(extent - 1) *
                            helper->second.dependencyStepCycles;
          resources.prefixScanCriticalPath += count * completion;
          continue;
        }
      }
      // One SIMD vector lane follows one independent sequence. Every vector
      // group executes an R-1 dependent add chain.
      const double groups = std::ceil(
          static_cast<double>(independentSequences) / profile.vectorWidth);
      const double vectorAdds =
          count * groups * static_cast<double>(extent - 1);
      resources.compute +=
          vectorAdds / addRate->second.throughput * addRate->second.factor;
      prefixScanIssuedInstructions += vectorAdds;
      resources.prefixScanCriticalPath += count *
                                          static_cast<double>(extent - 1) *
                                          profile.prefixScanStepLatencyCycles;
      continue;
    }

    // One SIMT warp follows one independent sequence.
    int64_t depth = 0;
    int64_t activeAddsPerSequence = 0;
    for (int64_t offset = 1; offset < extent;) {
      ++depth;
      activeAddsPerSequence += extent - offset;
      if (offset > (extent - 1) / 2)
        break;
      offset <<= 1;
    }
    const double warpSteps = count * independentSequences * depth;
    const double activeAdds =
        count * independentSequences * activeAddsPerSequence;
    const double predicateOps = count * independentSequences * extent * depth;
    resources.compute +=
        activeAdds / addRate->second.throughput * addRate->second.factor;
    resources.predicate += predicateOps / profile.predicateOperationsPerCycle;
    const double scanShuffleCycles =
        warpSteps * profile.issueWidth / profile.shuffleLanesPerCycle;
    resources.shuffle += scanShuffleCycles;
    resources.scanShuffle += scanShuffleCycles;
    // Preserve operation boundaries at every scan level.
    constexpr double instructionsPerWarpStep = 3.0;
    prefixScanIssuedInstructions += instructionsPerWarpStep * warpSteps;
    resources.prefixScanCriticalPath +=
        count * depth * profile.prefixScanStepLatencyCycles;
  }
  // Unclassified scalar operations have no execution-throughput evidence.
  if (stage.features.hasIndirectMemory) {
    const double loads = work.loadWarpInstructions;
    const double stores = work.storeWarpInstructions;
    double curvedLoadBytes = 0.0;
    double curvedLoadOperations = 0.0;
    double curvedLoadWarpInstructions = 0.0;
    double curvedStoreBytes = 0.0;
    double curvedStoreOperations = 0.0;
    double curvedStoreWarpInstructions = 0.0;
    double loadStartup = 0.0;
    std::map<std::tuple<int64_t, int64_t, int64_t>, double>
        coveredSegmentedLoadCounts;
    const auto factorLoad =
        profile.indirectLoadGroupCyclesByFactorAndWarpInstructions.find(
            superblockFactor);
    const auto factorStartup =
        profile.indirectLoadGroupStartupCyclesByFactorAndWarpInstructions.find(
            superblockFactor);
    const bool hasGridStrideLoadCurve =
        !profile.gridStrideIndirectLoadCyclesByWarpInstructions.empty() ||
        !profile.gridStrideIndirectLoadCyclesByFactorAndWarpInstructions
             .empty();
    const bool dynamicGridStride = stage.features.insideGridStrideLoopBody &&
                                   (!simd || hasGridStrideLoadCurve);
    const auto factorGridStrideLoad =
        profile.gridStrideIndirectLoadCyclesByFactorAndWarpInstructions.find(
            superblockFactor);
    const auto factorGridStrideStartup =
        profile.gridStrideIndirectLoadStartupCyclesByFactorAndWarpInstructions
            .find(superblockFactor);
    const std::map<int64_t, double> &baseLoadCurve =
        dynamicGridStride &&
                factorGridStrideLoad !=
                    profile
                        .gridStrideIndirectLoadCyclesByFactorAndWarpInstructions
                        .end()
            ? factorGridStrideLoad->second
        : dynamicGridStride &&
                !profile.gridStrideIndirectLoadCyclesByWarpInstructions.empty()
            ? profile.gridStrideIndirectLoadCyclesByWarpInstructions
        : factorLoad !=
                profile.indirectLoadGroupCyclesByFactorAndWarpInstructions.end()
            ? factorLoad->second
            : profile.indirectLoadSystemCyclesByWarpInstructions;
    const std::map<int64_t, double> &startupCurve =
        dynamicGridStride &&
                factorGridStrideStartup !=
                    profile
                        .gridStrideIndirectLoadStartupCyclesByFactorAndWarpInstructions
                        .end()
            ? factorGridStrideStartup->second
        : dynamicGridStride &&
                !profile.gridStrideIndirectLoadStartupCyclesByWarpInstructions
                     .empty()
            ? profile.gridStrideIndirectLoadStartupCyclesByWarpInstructions
        : factorStartup !=
                profile
                    .indirectLoadGroupStartupCyclesByFactorAndWarpInstructions
                    .end()
            ? factorStartup->second
            : profile.indirectLoadStartupSystemCyclesByWarpInstructions;
    for (const auto &[shape, count] : work.segmentedIndirectLoadCounts) {
      const int64_t bytes = std::get<0>(shape);
      const int64_t elements = std::get<1>(shape);
      const int64_t contiguousElements = std::get<2>(shape);
      if (elements <= 0 || contiguousElements <= 0)
        continue;
      const auto *segmentedSurfaces =
          bytes == elements * 2
              ? &profile
                     .segmentedIndirectLoadGroupCyclesByFactorSegmentsAndContiguousElements
          : bytes == elements * 4
              ? &profile
                     .segmentedIndirectLoadFp32GroupCyclesByFactorSegmentsAndContiguousElements
              : nullptr;
      if (!segmentedSurfaces)
        continue;
      const auto segmentedFactor = segmentedSurfaces->find(superblockFactor);
      if (segmentedFactor == segmentedSurfaces->end())
        continue;
      const int64_t segments = llvm::divideCeil(elements, contiguousElements);
      auto body = interpolateBoundedRowExtentSurface(
          segmentedFactor->second, segments, contiguousElements);
      if (!body)
        continue;
      resources.indirectLoadGroup += count * *body;
      curvedLoadBytes += count * bytes;
      curvedLoadOperations += count;
      curvedLoadWarpInstructions +=
          count * llvm::divideCeil(elements, int64_t(32));
      coveredSegmentedLoadCounts[shape] += count;
    }
    for (const auto &[shape, count] : work.indirectLoadCounts) {
      const int64_t bytes = shape.first;
      const int64_t elements = shape.second;
      const int64_t warpInstructions = llvm::divideCeil(elements, int64_t(32));
      // SIMD lowers a non-uniform loaded-index tensor to discrete element
      // accesses.
      const bool useElementLoadCurve =
          !dynamicGridStride &&
          !profile.indirectLoadSystemCyclesByElements.empty() &&
          (simd || bytes == elements * 4);
      const std::map<int64_t, double> &loadCurve =
          useElementLoadCurve ? profile.indirectLoadSystemCyclesByElements
                              : baseLoadCurve;
      const bool supportedElementWidth =
          bytes == elements * 2 ||
          (useElementLoadCurve && bytes == elements * 4);
      if (!supportedElementWidth || loadCurve.empty() || startupCurve.empty())
        continue;
      const int64_t loadCurveWork =
          useElementLoadCurve ? elements : warpInstructions;
      resources.indirectLoadGroup +=
          count * interpolateOperationCost(loadCurve, loadCurveWork);
      loadStartup = std::max(
          loadStartup, interpolateEnvelopeCost(startupCurve, warpInstructions));
      curvedLoadBytes += count * bytes;
      curvedLoadOperations += count;
      curvedLoadWarpInstructions += count * warpInstructions;
    }
    if (curvedLoadOperations > 0.0) {
      if (dynamicGridStride) {
        resources.gridStrideStartup = loadStartup;
        if (factorGridStrideLoad !=
            profile.gridStrideIndirectLoadCyclesByFactorAndWarpInstructions
                .end())
          resources.gridStrideFactorSpecificResourceBody = true;
      } else
        resources.indirectLoadGroupStartup = loadStartup;
    }
    // Loaded-index stores are indexed by distinct destination segments.
    if (!profile.indirectStoreSystemCyclesBySegments.empty()) {
      auto priceStore = [&](int64_t bytes, int64_t elements,
                            int64_t contiguousElements, double count) {
        if (bytes != elements * 4 || elements <= 0 || contiguousElements <= 0)
          return;
        const int64_t segments = llvm::divideCeil(elements, contiguousElements);
        resources.store +=
            count * interpolateOperationCost(
                        profile.indirectStoreSystemCyclesBySegments, segments);
        curvedStoreBytes += count * bytes;
        curvedStoreOperations += count;
        curvedStoreWarpInstructions +=
            count * llvm::divideCeil(elements, int64_t(32));
      };
      for (const auto &[shape, count] : work.indirectStoreCounts)
        priceStore(shape.first, shape.second, 1, count);
      for (const auto &[shape, count] : work.segmentedIndirectStoreCounts)
        priceStore(std::get<0>(shape), std::get<1>(shape), std::get<2>(shape),
                   count);
    }

    // Exact dtype/shape curves are stronger evidence, but they are not an
    // admission policy.
    constexpr double coverageTolerance = 1.0e-9;
    const double uncoveredLoadBytes =
        std::max(0.0, work.loadBytes - curvedLoadBytes);
    const double uncoveredLoadOperations =
        std::max(0.0, work.loadOperations - curvedLoadOperations);
    const double uncoveredLoadWarpInstructions =
        std::max(0.0, loads - curvedLoadWarpInstructions);
    const bool hasFallbackLoad =
        uncoveredLoadBytes > coverageTolerance ||
        uncoveredLoadOperations > coverageTolerance ||
        uncoveredLoadWarpInstructions > coverageTolerance;
    const double uncoveredStoreBytes =
        std::max(0.0, work.storeBytes - curvedStoreBytes);
    const double uncoveredStoreOperations =
        std::max(0.0, work.storeOperations - curvedStoreOperations);
    const double uncoveredStoreWarpInstructions =
        std::max(0.0, stores - curvedStoreWarpInstructions);
    const bool hasFallbackStore =
        uncoveredStoreWarpInstructions > coverageTolerance ||
        uncoveredStoreBytes > coverageTolerance ||
        uncoveredStoreOperations > coverageTolerance;
    resources.indirectMemoryAnalyticalFallback =
        hasFallbackLoad || hasFallbackStore;
    if (simd) {
      resources.load +=
          uncoveredLoadOperations * profile.loadOperationSetupCycles +
          uncoveredLoadBytes / profile.loadBytesPerCycle;
      for (const auto &[shape, count] : work.segmentedIndirectLoadCounts) {
        const int64_t elements = std::get<1>(shape);
        const int64_t contiguousElements = std::get<2>(shape);
        const int64_t segments = llvm::divideCeil(elements, contiguousElements);
        const auto coveredIt = coveredSegmentedLoadCounts.find(shape);
        const double covered = coveredIt == coveredSegmentedLoadCounts.end()
                                   ? 0.0
                                   : coveredIt->second;
        const double uncoveredCount = std::max(0.0, count - covered);
        resources.load += uncoveredCount * std::max<int64_t>(0, segments - 1) *
                          profile.loadOperationSetupCycles;
      }
      resources.store +=
          uncoveredStoreOperations * profile.storeOperationSetupCycles +
          uncoveredStoreBytes / profile.storeBytesPerCycle;
    } else {
      resources.load +=
          uncoveredLoadWarpInstructions / profile.loadWarpInstructionsPerCycle;
      resources.store += uncoveredStoreWarpInstructions /
                         profile.storeWarpInstructionsPerCycle;
    }
  } else {
    LayoutCoverage layout;
    const bool pipelinedSimd = simd && permitsSimdOverlap(stage) &&
                               stage.features.synchronizationCount == 0 &&
                               stage.features.conditionalBranchCount == 0;
    // Runs before the load-group curve below, and before the masked-copy curve
    // further down, so both can see which accesses this pass already priced.
    mapStridedShortMemoryCompletion(stage, work, profile, resources, simd,
                                    layout, stridedShortAlreadyClaimed);
    // The store-group curve below must not price a store this pass already did.
    const bool storesClaimedByClosedForm = std::any_of(
        stridedShortAlreadyClaimed.begin(), stridedShortAlreadyClaimed.end(),
        [](const StageMemoryAccess *access) { return access->store; });
    if (pipelinedSimd && profile.independentLoadGroupBytesPerCycle > 0.0) {
      std::vector<const StageMemoryAccess *> flatPrefix;
      bool reachedFirstLoad = false;
      bool allLoadsRankOne = true;
      for (const StageMemoryAccess &access : work.memoryAccesses) {
        if (access.store) {
          if (reachedFirstLoad)
            break;
          continue;
        }
        reachedFirstLoad = true;
        auto shape = accessShape(access);
        if (access.count <= 0.0 || !shape || access.elementBits != 32 ||
            !access.isFlatContiguous())
          break;
        allLoadsRankOne &= access.effectiveRank() == 1;
        flatPrefix.push_back(&access);
      }
      constexpr double tolerance = 1.0e-9;
      int64_t groupOccurrences = 0;
      bool exactCounts = !flatPrefix.empty();
      for (const StageMemoryAccess *access : flatPrefix) {
        const int64_t count = static_cast<int64_t>(std::llround(access->count));
        exactCounts &=
            count > 0 &&
            std::abs(access->count - static_cast<double>(count)) <= tolerance;
        if (exactCounts)
          groupOccurrences =
              groupOccurrences == 0 ? count : std::gcd(groupOccurrences, count);
      }
      int64_t operationsPerGroup = 0;
      double bytesPerGroup = 0.0;
      double coveredOperations = 0.0;
      double coveredBytes = 0.0;
      double coveredWarpInstructions = 0.0;
      if (exactCounts && groupOccurrences > 0) {
        for (const StageMemoryAccess *access : flatPrefix) {
          const auto shape = *accessShape(*access);
          const int64_t count =
              static_cast<int64_t>(std::llround(access->count));
          const int64_t occurrencesPerGroup = count / groupOccurrences;
          operationsPerGroup += occurrencesPerGroup;
          bytesPerGroup += occurrencesPerGroup * shape.first;
          coveredOperations += access->count;
          coveredBytes += access->count * shape.first;
          coveredWarpInstructions +=
              access->count * llvm::divideCeil(shape.second, int64_t(32));
        }
      }
      const bool groupCovered =
          exactCounts && groupOccurrences > 0 &&
          operationsPerGroup >= profile.independentLoadGroupMinimumOperations &&
          operationsPerGroup <= profile.independentLoadGroupMaximumOperations &&
          // The one-stream evidence is a flat rank-one grid-stride transfer.
          (operationsPerGroup != 1 || allLoadsRankOne);
      if (groupCovered) {
        resources.setup += profile.independentLoadGroupKernelStartupCycles;
        resources.continuousLoadGroup +=
            groupOccurrences *
            (profile.independentLoadGroupIterationStartupCycles +
             bytesPerGroup / profile.independentLoadGroupBytesPerCycle);
        layout.loadBytes += coveredBytes;
        layout.loadOperations += coveredOperations;
        layout.loadWarpInstructions += coveredWarpInstructions;
      }
    }

    if (pipelinedSimd && !storesClaimedByClosedForm &&
        !profile.pipelinedStridedStoreGroupsByRowStrideBytes.empty()) {
      double operations = 0.0;
      double bytes = 0.0;
      double warpInstructions = 0.0;
      int64_t rowStrideBytes = 0;
      int64_t rows = 0;
      bool allStoresCovered = work.storeOperations > 0.0;
      for (const StageMemoryAccess &access : work.memoryAccesses) {
        if (!access.store || access.count <= 0.0)
          continue;
        auto shape = accessShape(access);
        const int64_t accessStride = access.rowStrideBytes();
        const int64_t accessRows = access.rowCount();
        if (!shape || access.elementBits != 32 || !access.masked ||
            access.layoutClass() !=
                StageMemoryAccess::LayoutClass::StridedShort ||
            accessStride <= 0 || accessRows <= 0 ||
            (rowStrideBytes != 0 && rowStrideBytes != accessStride) ||
            (rows != 0 && rows != accessRows)) {
          allStoresCovered = false;
          break;
        }
        rowStrideBytes = accessStride;
        rows = accessRows;
        operations += access.count;
        bytes += access.count * shape->first;
        warpInstructions +=
            access.count * llvm::divideCeil(shape->second, int64_t(32));
      }
      auto curve = profile.pipelinedStridedStoreGroupsByRowStrideBytes.find(
          rowStrideBytes);
      constexpr double tolerance = 1.0e-9;
      const int64_t roundedOperations =
          static_cast<int64_t>(std::llround(operations));
      allStoresCovered &=
          std::abs(operations - work.storeOperations) <= tolerance &&
          std::abs(operations - roundedOperations) <= tolerance &&
          curve != profile.pipelinedStridedStoreGroupsByRowStrideBytes.end();
      if (allStoresCovered) {
        const auto &fit = curve->second;
        allStoresCovered = rows >= fit.minimumRows && rows <= fit.maximumRows &&
                           roundedOperations >= fit.minimumStores &&
                           roundedOperations <= fit.maximumStores;
        if (allStoresCovered) {
          resources.setup += fit.kernelStartupCycles;
          resources.continuousStoreGroup +=
              fit.firstStoreCycles +
              (roundedOperations - 1) * fit.incrementalStoreCycles +
              std::max<int64_t>(0, rows - 32) * fit.extraLaneCycles;
          layout.storeBytes += bytes;
          layout.storeOperations += operations;
          layout.storeWarpInstructions += warpInstructions;
        }
      }
    }

    // GM count curves contain no FMA-to-store shared-memory exchange.
    std::map<const StageMemoryLayoutCost *, double> exchanges;
    if (!simd && isContinuousMemoryStage(stage.costModelKind))
      for (const auto &access : work.memoryAccesses)
        if (const auto *curve =
                dotStoreSharedCurve(access, profile, superblockFactor))
          exchanges[curve] += access.count;
    for (const auto &[curve, count] : exchanges)
      if (auto cost = curve->evaluate(count))
        resources.sharedLayout += *cost / superblockFactor;
    auto hasExchange = [&](const StageMemoryAccess &access) {
      const auto *curve =
          simd ? nullptr
               : dotStoreSharedCurve(access, profile, superblockFactor);
      return curve && exchanges.count(curve) &&
             curve->evaluate(exchanges.at(curve)).has_value();
    };
    // A count curve describes one uninterrupted sequence.
    std::vector<StageMemoryAccess> accessGroups;
    for (const StageMemoryAccess &access : work.memoryAccesses) {
      if (!accessGroups.empty() && accessGroups.back().store == access.store &&
          accessGroups.back().layoutKey() == access.layoutKey() &&
          !hasExchange(access) && !hasExchange(accessGroups.back()))
        accessGroups.back().count += access.count;
      else
        accessGroups.push_back(access);
    }
    for (const auto &access : accessGroups) {
      if (!isContinuousMemoryStage(stage.costModelKind) ||
          stage.features.synchronizationCount != 0 || stage.features.hasLoop ||
          stage.features.conditionalBranchCount != 0)
        break;
      auto measuredAccess = access;
      if (hasExchange(access))
        measuredAccess.count = 1;
      auto groupCost =
          mapLayoutAccess(measuredAccess, profile, simd, superblockFactor);
      if (groupCost && hasExchange(access))
        *groupCost *= access.count;
      const auto shape = accessShape(access);
      if (!groupCost || !shape)
        continue;
      // Subtract the covered work in its original normalized unit. The new
      // profile's group envelope is not itself an instruction count.
      const double normalizedWarps =
          llvm::divideCeil(shape->second, int64_t(32));
      if (access.store) {
        resources.continuousStoreGroup += *groupCost;
        layout.storeBytes += access.count * shape->first;
        layout.storeOperations += access.count;
        layout.storeWarpInstructions += access.count * normalizedWarps;
        layout.storeShapes[*shape] += access.count;
      } else {
        resources.continuousLoadGroup += *groupCost;
        layout.loadBytes += access.count * shape->first;
        layout.loadOperations += access.count;
        layout.loadWarpInstructions += access.count * normalizedWarps;
        layout.loadShapes[*shape] += access.count;
      }
    }
    struct CurvedMemoryCoverage {
      double operations = 0.0;
      double bytes = 0.0;
      double warpInstructions = 0.0;
      double startup = 0.0;
    } curvedLoad, curvedStore;
    auto mapContinuousCurves = [&](const auto &counts, const auto &curves,
                                   const auto &startupCurves,
                                   const auto &coveredShapes, double &resource,
                                   CurvedMemoryCoverage &coverage) {
      const auto curve = curves.find(superblockFactor);
      if (!isContinuousMemoryStage(stage.costModelKind) ||
          curve == curves.end())
        return;
      const auto startupCurve = startupCurves.find(superblockFactor);
      for (const auto &[shape, count] : counts) {
        const auto [bytes, elements] = shape;
        const auto covered = coveredShapes.find(shape);
        const double availableCount = std::max(
            0.0,
            count - (covered == coveredShapes.end() ? 0.0 : covered->second));
        if (availableCount <= 0.0 || bytes != elements * 4)
          continue;
        auto operationCost =
            interpolateBoundedOperationCost(curve->second, bytes);
        if (!operationCost)
          continue;
        resource += availableCount * *operationCost;
        if (startupCurve != startupCurves.end()) {
          auto startupCost =
              interpolateBoundedOperationCost(startupCurve->second, bytes);
          if (startupCost)
            coverage.startup = std::max(coverage.startup, *startupCost);
        }
        coverage.operations += availableCount;
        coverage.bytes += availableCount * bytes;
        coverage.warpInstructions +=
            availableCount * llvm::divideCeil(elements, int64_t(32));
      }
      if (coverage.operations > 0.0)
        resource += coverage.startup;
    };
    // These curves are complete FP32 contiguous-memory group envelopes.
    mapContinuousCurves(
        work.continuousLoadCounts,
        profile.continuousLoadGroupCyclesByFactorAndBytes,
        profile.continuousLoadGroupStartupCyclesByFactorAndBytes,
        layout.loadShapes, resources.continuousLoadGroup, curvedLoad);
    mapContinuousCurves(
        work.continuousStoreCounts,
        profile.continuousStoreGroupCyclesByFactorAndBytes,
        profile.continuousStoreGroupStartupCyclesByFactorAndBytes,
        layout.storeShapes, resources.continuousStoreGroup, curvedStore);
    if (simd) {
      const double uncoveredLoadOperations =
          std::max(0.0, work.loadOperations - layout.loadOperations -
                            curvedLoad.operations);
      const double uncoveredStoreOperations =
          std::max(0.0, work.storeOperations - layout.storeOperations -
                            curvedStore.operations);
      const double loadSetup =
          uncoveredLoadOperations * profile.loadOperationSetupCycles;
      const double storeSetup =
          uncoveredStoreOperations * profile.storeOperationSetupCycles;
      const bool amortizeAdditionalRowwiseLoadSetups = [&]() {
        if (stage.costModelKind != StageCostModelKind::RowwiseReduction ||
            stage.iterationCount <= 1 || uncoveredLoadOperations <= 1.0 ||
            std::abs(uncoveredLoadOperations - work.loadOperations) > 1.0e-9)
          return false;
        std::optional<std::string> commonLayout;
        double matchingOperations = 0.0;
        for (const StageMemoryAccess &access : work.memoryAccesses) {
          if (access.store || access.count <= 0.0)
            continue;
          if (access.masked || access.addressDependsOnLoadedIndex ||
              access.layoutClass() !=
                  StageMemoryAccess::LayoutClass::StridedWide)
            return false;
          const std::string layout = access.layoutKey();
          if (commonLayout && *commonLayout != layout)
            return false;
          commonLayout = layout;
          matchingOperations += access.count;
        }
        return commonLayout &&
               std::abs(matchingOperations - uncoveredLoadOperations) <= 1.0e-9;
      }();
      const bool amortizeLoopTransferSetup =
          stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence &&
          stage.iterationCount > 1;
      if (amortizeLoopTransferSetup)
        resources.setup += loadSetup + storeSetup;
      else if (amortizeAdditionalRowwiseLoadSetups) {
        resources.setup +=
            (uncoveredLoadOperations - 1.0) * profile.loadOperationSetupCycles;
        resources.load += profile.loadOperationSetupCycles;
        resources.store += storeSetup;
        resources.rowwiseAdditionalLoadSetupsOwnedOnce = true;
      } else {
        resources.load += loadSetup;
        resources.store += storeSetup;
      }
      resources.load +=
          std::max(0.0, work.loadBytes - layout.loadBytes - curvedLoad.bytes) /
          profile.loadBytesPerCycle;
      resources.store += std::max(0.0, work.storeBytes - layout.storeBytes -
                                           curvedStore.bytes) /
                         profile.storeBytesPerCycle;
      if (profile.computedAddressScalarizedLoadCyclesPerElement > 0.0) {
        for (const StageMemoryAccess &access : work.memoryAccesses) {
          if (access.store || access.count <= 0.0 ||
              access.addressDependsOnLoadedIndex ||
              !access.addressHasNonAffineTensorComputation ||
              access.layoutClass() != StageMemoryAccess::LayoutClass::Unknown ||
              access.effectiveRank() != 1 ||
              access.provenContiguousElements() != 1)
            continue;
          const auto elements = accessElements(access);
          if (!elements)
            continue;
          if (*elements <
                  profile.computedAddressScalarizedLoadMinimumElements ||
              *elements > profile.computedAddressScalarizedLoadMaximumElements)
            continue;
          const double completion =
              access.count * *elements *
              profile.computedAddressScalarizedLoadCyclesPerElement;
          resources.computedAddressScalarizedLoad += completion;
          resources.load += completion;
        }
      }
      if (!profile.controlFlowAddressScalarizedLoadResidualCyclesByElements
               .empty()) {
        for (const StageMemoryAccess &access : work.memoryAccesses) {
          if (access.store || access.count <= 0.0 || access.elementBits != 32 ||
              access.addressDependsOnLoadedIndex ||
              !access.addressHasTensorControlFlowSelection ||
              access.layoutClass() != StageMemoryAccess::LayoutClass::Unknown ||
              access.effectiveRank() != 1 ||
              access.provenContiguousElements() != 1)
            continue;
          const auto elements = accessElements(access);
          if (!elements)
            continue;
          auto completion = interpolateBoundedOperationCost(
              profile.controlFlowAddressScalarizedLoadResidualCyclesByElements,
              *elements);
          if (!completion)
            continue;
          const double total = access.count * *completion;
          resources.controlFlowAddressScalarizedLoad += total;
          resources.load += total;
        }
      }
      if (!profile.maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows
               .empty()) {
        for (const StageMemoryAccess &access : work.memoryAccesses) {
          if (!access.store || access.count <= 0.0 || !access.masked ||
              access.elementBits != 16 ||
              !access.maskedStoreHasMatchingLoadDependency ||
              access.effectiveRank() != 2 || !access.hasStaticLayout() ||
              !access.hasUnitInnerStride() || access.rowBytes() <= 0 ||
              access.rowCount() <= 0 || access.activeElementRatio <= 0.0 ||
              access.activeElementRatio >= 1.0 - 1.0e-9)
            continue;
          auto rowCurve =
              profile.maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows
                  .find(access.rowBytes());
          if (rowCurve ==
              profile.maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows
                  .end())
            continue;
          const double activeRowsValue =
              access.rowCount() * access.activeElementRatio;
          const int64_t activeRows =
              static_cast<int64_t>(std::llround(activeRowsValue));
          if (activeRows <= 0 ||
              std::abs(activeRowsValue - static_cast<double>(activeRows)) >
                  1.0e-6)
            continue;
          auto completion =
              interpolateBoundedOperationCost(rowCurve->second, activeRows);
          if (!completion)
            continue;
          const double total = access.count * *completion;
          resources.maskedLoadStoreReadModifyWrite += total;
          resources.store += total;
        }
      }
    } else {
      resources.load += std::max(0.0, work.loadWarpInstructions -
                                          layout.loadWarpInstructions -
                                          curvedLoad.warpInstructions) /
                        profile.loadWarpInstructionsPerCycle;
      resources.store += std::max(0.0, work.storeWarpInstructions -
                                           layout.storeWarpInstructions -
                                           curvedStore.warpInstructions) /
                         profile.storeWarpInstructionsPerCycle;
    }
  }
  if (!simd) {
    // Generated LDS/STS/LDK/STK streams are not semantic TTIR operations.
    const double fallbackLoadRate = profile.loadWarpInstructionsPerCycle;
    const double fallbackStoreRate = profile.storeWarpInstructionsPerCycle;
    if (work.sharedLocalLoadInstructions == 0.0)
      work.sharedLocalLoadInstructions =
          sourceFitInstructions(profile.sharedLoadSourceFit, work, stage,
                                profile, false, 0.0, wholeSimt);
    if (work.sharedLocalStoreInstructions == 0.0)
      work.sharedLocalStoreInstructions =
          sourceFitInstructions(profile.sharedStoreSourceFit, work, stage,
                                profile, true, 0.0, wholeSimt);
    // The register-pressure observation describes the prospective whole-SIMT
    // function.
    if (wholeSimt && work.privateStackLoadInstructions == 0.0)
      work.privateStackLoadInstructions = sourceFitInstructions(
          profile.privateLoadSourceFit, work, stage, profile, false, 0.0, true);
    if (wholeSimt && work.privateStackStoreInstructions == 0.0)
      work.privateStackStoreInstructions = sourceFitInstructions(
          profile.privateStoreSourceFit, work, stage, profile, true, 0.0, true);
    const double sharedLoadInstructions = work.sharedLocalLoadInstructions;
    const double sharedStoreInstructions = work.sharedLocalStoreInstructions;
    const double privateLoadInstructions = work.privateStackLoadInstructions;
    const double privateStoreInstructions = work.privateStackStoreInstructions;
    const double sharedLoadRate = generatedRate(
        profile.sharedLocalLoadInstructionsPerCycle, fallbackLoadRate);
    const double sharedStoreRate = generatedRate(
        profile.sharedLocalStoreInstructionsPerCycle, fallbackStoreRate);
    const double privateLoadRate = generatedRate(
        profile.privateStackLoadInstructionsPerCycle, fallbackLoadRate);
    const double privateStoreRate = generatedRate(
        profile.privateStackStoreInstructionsPerCycle, fallbackStoreRate);
    auto chargeGeneratedTraffic = [](double instructions, double rate,
                                     double scale, double &component,
                                     double &total) {
      if (instructions <= 0.0 || rate <= 0.0)
        return;
      component = instructions / rate * scale;
      total += component;
    };
    chargeGeneratedTraffic(sharedLoadInstructions, sharedLoadRate, 1.0,
                           resources.sharedLocalLoad, resources.load);
    chargeGeneratedTraffic(sharedStoreInstructions, sharedStoreRate, 1.0,
                           resources.sharedLocalStore, resources.store);
    chargeGeneratedTraffic(privateLoadInstructions, privateLoadRate,
                           profile.spillTransactionCoverage,
                           resources.privateStackLoad, resources.load);
    chargeGeneratedTraffic(privateStoreInstructions, privateStoreRate,
                           profile.spillTransactionCoverage,
                           resources.privateStackStore, resources.store);
    if (const auto *floor = matchingDotComponentPrivateStackFloor(
            stage, profile, superblockFactor, wholeSimt)) {
      // Each entry prices the combined private load/store stream for one
      // complete measured warp group.
      const double fraction =
          privateStackFloorFraction(*stage.wholeSimtRegisterPressure, *floor);
      const double combined =
          floor->combinedGroupCycles * fraction / superblockFactor;
      const double alreadyAttributed =
          resources.privateStackLoad + resources.privateStackStore;
      resources.privateStackCombinedFloor =
          std::max(0.0, combined - alreadyAttributed);
      resources.dotComponentPrivateStackFloorApplied = true;
    }

    if (wholeSimt &&
        stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence &&
        profile.spillTransactionsPerCycle > 0.0) {
      const double pressureFloor = peakLiveExcess(stage, profile) /
                                   profile.spillTransactionsPerCycle *
                                   profile.spillTransactionCoverage;
      const double alreadyOwned = resources.privateStackLoad +
                                  resources.privateStackStore +
                                  resources.privateStackCombinedFloor;
      resources.spill = std::max(resources.spill,
                                 std::max(0.0, pressureFloor - alreadyOwned));
    }

    if (work.indirectLoadCounts.empty() &&
        work.segmentedIndirectLoadCounts.empty() &&
        work.continuousLoadCounts.empty() &&
        profile.globalLoadSourceFit.enabled) {
      const double fitted = sourceFitInstructions(
          profile.globalLoadSourceFit, work, stage, profile, false,
          work.loadWarpInstructions, wholeSimt);
      if (fitted > 0.0 && fallbackLoadRate > 0.0) {
        work.loadWarpInstructions = fitted;
        resources.load = fitted / fallbackLoadRate + resources.sharedLocalLoad +
                         resources.privateStackLoad;
      }
    }
    if (work.indirectStoreCounts.empty() &&
        work.segmentedIndirectStoreCounts.empty() &&
        work.continuousStoreCounts.empty() &&
        profile.globalStoreSourceFit.enabled) {
      const double fitted = sourceFitInstructions(
          profile.globalStoreSourceFit, work, stage, profile, true,
          work.storeWarpInstructions, wholeSimt);
      if (fitted > 0.0 && fallbackStoreRate > 0.0) {
        work.storeWarpInstructions = fitted;
        resources.store = fitted / fallbackStoreRate +
                          resources.sharedLocalStore +
                          resources.privateStackStore;
      }
    }
  }
  double predicateElements = 0.0;
  for (const auto &[elements, count] : work.predicateCounts)
    predicateElements += static_cast<double>(elements) * count;
  if (simd && fullyDescribed(segmented.logicalElements.lookup("predicate.cmp"),
                             predicateElements)) {
    resources.predicate += segmented.instructions.lookup("predicate.cmp") /
                           profile.predicateOperationsPerCycle;
  } else {
    for (const auto &[elements, count] : work.predicateCounts) {
      const double instructions =
          simd ? std::ceil(static_cast<double>(elements) / profile.vectorWidth)
               : static_cast<double>(elements);
      resources.predicate +=
          count * instructions / profile.predicateOperationsPerCycle;
    }
  }
  const double simdReductionStartupOwners =
      std::min(1.0, std::max(0.0, work.reductionOperations));
  resources.reduction =
      simd && profile.reductionLaneStepsPerCycle > 0.0
          ? simdReductionStartupOwners * profile.reductionOperationSetupCycles +
                work.shuffleLaneSteps / profile.reductionLaneStepsPerCycle
          : work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
  if (simd && profile.scalarizedMultiResultReductionMinimumExtent > 0 &&
      profile.scalarizedMultiResultReductionReferenceResults > 0 &&
      profile.scalarizedMultiResultReductionElementsPerCycle > 0.0) {
    for (const auto &[shape, count] :
         work.scalarizedMultiResultReductionCounts) {
      const int64_t rows = std::get<0>(shape);
      const int64_t extent = std::get<1>(shape);
      const int64_t results = std::get<2>(shape);
      const int64_t bits = std::get<3>(shape);
      if (count <= 0.0 ||
          extent < profile.scalarizedMultiResultReductionMinimumExtent ||
          results < profile.scalarizedMultiResultReductionReferenceResults ||
          bits != 32)
        continue;
      const double resultScale =
          static_cast<double>(results) /
          profile.scalarizedMultiResultReductionReferenceResults;
      resources.scalarizedMultiResultReduction +=
          count * (profile.scalarizedMultiResultReductionSetupCycles +
                   static_cast<double>(rows) * extent * resultScale /
                       profile.scalarizedMultiResultReductionElementsPerCycle);
    }
    // The scalar loop is the concrete lowering of the same semantic reduction
    // already priced above.
    resources.reduction =
        std::max(resources.reduction, resources.scalarizedMultiResultReduction);
  }
  if (!simd && wholeSimt && superblockFactor == 1 &&
      profile.multiResultReductionGroupBaseWarpCount > 0 &&
      profile.activeWarpCount ==
          profile.multiResultReductionGroupBaseWarpCount * superblockFactor &&
      profile.multiResultReductionResultCount > 0 &&
      profile.multiResultReductionElementBits > 0 &&
      profile.multiResultReductionPaddedAxisExtent > 0) {
    for (const auto &[shape, count] :
         work.scalarizedMultiResultReductionCounts) {
      const int64_t rows = std::get<0>(shape);
      const int64_t extent = std::get<1>(shape);
      const int64_t results = std::get<2>(shape);
      const int64_t bits = std::get<3>(shape);
      if (count <= 0.0 || rows <= 0 || extent <= 0 ||
          results != profile.multiResultReductionResultCount ||
          bits != profile.multiResultReductionElementBits ||
          llvm::PowerOf2Ceil(static_cast<uint64_t>(extent)) !=
              static_cast<uint64_t>(
                  profile.multiResultReductionPaddedAxisExtent))
        continue;
      const llvm::StringRef tailClass = multiResultReductionTailClass(extent);
      auto curve =
          profile.multiResultReductionGroupCyclesByTailClassAndRows.find(
              tailClass.str());
      if (curve ==
          profile.multiResultReductionGroupCyclesByTailClassAndRows.end())
        continue;
      auto body = lookupBoundedCeilingGroupCost(curve->second, rows);
      if (body)
        resources.multiResultReductionGroupBody += count * *body;
    }
  }
  if (simd && profile.sortMinimumExtent > 0 &&
      profile.sortMaximumExtent >= profile.sortMinimumExtent &&
      profile.sortElementBits > 0 && profile.sortRowsPerCycle > 0.0) {
    for (const auto &[shape, count] : work.sortCounts) {
      const int64_t rows = std::get<0>(shape);
      const int64_t extent = std::get<1>(shape);
      const int64_t bits = std::get<2>(shape);
      if (count <= 0.0 || extent < profile.sortMinimumExtent ||
          extent > profile.sortMaximumExtent || bits != profile.sortElementBits)
        continue;
      resources.sort +=
          count * (profile.sortSetupCycles + rows / profile.sortRowsPerCycle);
    }
    // ascend.sort previously contributed only to the generic issue floor.
    resources.compute += resources.sort;
  }
  if (simd && !work.oneTripMaskedAccumulatorReductionExtents.empty()) {
    resources.maskedAccumulatorReductionCompletionSetup =
        profile.maskedAccumulatorReductionCompletionSetupCycles;
    resources.setup += resources.maskedAccumulatorReductionCompletionSetup;
  }
  if (!simd && wholeSimt && profile.rowwiseReductionGroupBaseWarpCount > 0 &&
      profile.activeWarpCount ==
          profile.rowwiseReductionGroupBaseWarpCount * superblockFactor) {
    const auto exactOperations =
        exactNonNegativeWorkCount(work.reductionOperations);
    const auto additionalSurface =
        exactOperations
            ? profile.rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent
                  .find(*exactOperations)
            : profile.rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent
                  .end();
    const auto *surfaces =
        additionalSurface !=
                profile
                    .rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent
                    .end()
            ? &additionalSurface->second
            : &profile.rowwiseReductionGroupCyclesByFactorRowsAndExtent;
    const int64_t referenceOperations =
        additionalSurface !=
                profile
                    .rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent
                    .end()
            ? additionalSurface->first
            : profile.rowwiseReductionGroupReferenceOperations;
    auto geometry = rowwiseReductionGeometry(stage, work, referenceOperations);
    auto factorSurface = surfaces->find(superblockFactor);
    if (geometry && factorSurface != surfaces->end()) {
      auto body = interpolateBoundedRowExtentSurface(
          factorSurface->second, geometry->first, geometry->second);
      if (body)
        resources.rowwiseReductionGroupBody = *body;
    }
  }
  if (!simd && stage.costModelKind == StageCostModelKind::RowwiseReduction &&
      resources.rowwiseReductionGroupBody == 0.0) {
    // The generic shuffleLaneSteps workload counts source lanes at every
    // reduction level.
    std::optional<std::pair<int64_t, int64_t>> geometry;
    if (work.reductionGeometryCounts.size() == 1) {
      const auto &[shape, count] = *work.reductionGeometryCounts.begin();
      if (std::get<2>(shape) == 32 && std::get<0>(shape) > 0 &&
          std::get<1>(shape) > 1 && count > 0.0)
        geometry = std::make_pair(std::get<0>(shape), std::get<1>(shape));
    }
    if (!geometry) {
      const auto exactOperations =
          exactNonNegativeWorkCount(work.reductionOperations);
      if (exactOperations)
        geometry = rowwiseReductionGeometry(stage, work, *exactOperations);
    }
    if (geometry) {
      const int64_t rows = geometry->first;
      const int64_t extent = geometry->second;
      const int64_t warpWidth = 32;
      auto add = profile.operationRates.find("f32.add");
      if (rows > 0 && extent > 1 && add != profile.operationRates.end() &&
          add->second.throughput > 0.0 && profile.shuffleLanesPerCycle > 0.0) {
        const double scalarReductionOps = work.reductionOperations *
                                          static_cast<double>(rows) *
                                          static_cast<double>(extent - 1);
        resources.reduction =
            scalarReductionOps / add->second.throughput * add->second.factor;

        const int64_t participatingLanes = std::min(extent, warpWidth);
        int64_t shuffleDepth = 0;
        for (int64_t lanes = participatingLanes; lanes > 1;
             lanes = llvm::divideCeil(lanes, int64_t(2)))
          ++shuffleDepth;
        const double shuffleIssue =
            work.reductionOperations * static_cast<double>(rows) *
            static_cast<double>(participatingLanes - 1) /
            profile.shuffleLanesPerCycle;
        const double shuffleDependency = work.reductionOperations *
                                         static_cast<double>(shuffleDepth) *
                                         profile.prefixScanStepLatencyCycles;
        resources.shuffle = std::max(shuffleIssue, shuffleDependency);
      }
    }
  }
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence &&
      stage.features.hasReduction && work.shuffleLaneSteps > 0.0) {
    auto factorCurve =
        profile.recurrenceReductionGroupCyclesByFactorAndLaneSteps.find(
            superblockFactor);
    if (factorCurve !=
        profile.recurrenceReductionGroupCyclesByFactorAndLaneSteps.end())
      resources.recurrenceGroupCriticalPath = interpolateOperationCost(
          factorCurve->second,
          static_cast<int64_t>(std::ceil(work.shuffleLaneSteps)));
    auto startupCurve =
        profile.recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps.find(
            superblockFactor);
    if (startupCurve !=
        profile.recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps.end())
      resources.recurrenceGroupStartup = interpolateOperationCost(
          startupCurve->second,
          static_cast<int64_t>(std::ceil(work.shuffleLaneSteps)));
  }
  if (work.dotFlops > 0.0) {
    // A Stage can own multiple tiny dots.
    resources.dot = work.dotOperations * profile.dotSetupCycles +
                    work.dotFlops / profile.dotFlopsPerCycle;
    if (stage.features.isGridStrideLoopShell ||
        stage.features.insideGridStrideLoopBody) {
      const auto factorBody =
          profile.gridStrideDotCyclesByFactorAndFlops.find(superblockFactor);
      const auto factorStartup =
          profile.gridStrideDotStartupCyclesByFactorAndFlops.find(
              superblockFactor);
      const auto &bodyCurve =
          factorBody != profile.gridStrideDotCyclesByFactorAndFlops.end()
              ? factorBody->second
              : profile.gridStrideDotCyclesByFlops;
      const auto &startupCurve =
          factorStartup !=
                  profile.gridStrideDotStartupCyclesByFactorAndFlops.end()
              ? factorStartup->second
              : profile.gridStrideDotStartupCyclesByFlops;
      double dynamicDot = 0.0;
      double dynamicStartup = 0.0;
      double matchedOperations = 0.0;
      for (const auto &[flops, count] : work.dotFlopCounts) {
        if (count <= 0.0 || bodyCurve.empty())
          continue;
        dynamicDot += count * interpolateOperationCost(bodyCurve, flops);
        if (!startupCurve.empty())
          dynamicStartup = std::max(
              dynamicStartup, interpolateEnvelopeCost(startupCurve, flops));
        matchedOperations += count;
      }
      if (matchedOperations > 0.0) {
        resources.dot = dynamicDot;
        resources.gridStrideStartup =
            std::max(resources.gridStrideStartup, dynamicStartup);
        if (factorBody != profile.gridStrideDotCyclesByFactorAndFlops.end())
          resources.gridStrideFactorSpecificResourceBody = true;
      }
    }
  }
  resources.issue = countIssueInstructions(work, profile.issueWidth, simd,
                                           profile.vectorWidthBits) /
                    profile.issueOperationsPerCycle;
  resources.dot += resources.dotOperandGroup / superblockFactor;
  resources.prefixScanIssue =
      prefixScanIssuedInstructions / profile.issueOperationsPerCycle;
  resources.issue += resources.prefixScanIssue;
  if (stage.features.hasLoopCarriedDataDependency)
    resources.criticalPath = resources.scalar + resources.compute +
                             resources.predicate + resources.reduction +
                             resources.shuffle + resources.dot;
  else if (stage.features.hasReduction)
    resources.criticalPath = resources.compute + resources.predicate +
                             resources.reduction + resources.shuffle;
  if (!simd && isContinuousMemoryStage(stage.costModelKind) &&
      resources.continuousStoreGroup == 0.0) {
    resources.storeLayoutService =
        storeLayoutService(work, profile, superblockFactor);
    // Both terms price the SAME transfers. Taking their max strengthens the
    // old flat-throughput bound without charging those bytes a second time.
    resources.store = std::max(resources.store, resources.storeLayoutService);
  }
  mapMaskedStridedCopyCompletion(stage, work, profile, resources,
                                 stridedShortAlreadyClaimed);
  return materializeControlFlow(stage, mode, resources, profile.controlFlow);
}

/// Balanced CV tensor partition on one AIV.
static LogicalStage perAivStage(const LogicalStage &logical, int64_t parts) {
  LogicalStage stage = logical;
  if (parts == 1)
    return stage;
  auto extent = [parts](int64_t n) { return llvm::divideCeil(n, parts); };
  StageWorkload &work = stage.workload;
  // The TTIR segment projection describes the unsliced tensor.
  work.tensorOperationWorkloads.clear();
  for (auto &entry : work.operationCounts) {
    std::map<int64_t, double> counts;
    for (const auto &[elements, count] : entry.second)
      counts[extent(elements)] += count;
    entry.second = std::move(counts);
  }
  std::map<int64_t, double> predicates;
  for (const auto &[elements, count] : work.predicateCounts)
    predicates[extent(elements)] += count;
  work.predicateCounts = std::move(predicates);
  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, bool, std::string>,
           double>
      scans;
  for (const auto &[shape, count] : work.prefixScanCounts)
    scans[{std::get<0>(shape), extent(std::get<1>(shape)), std::get<2>(shape),
           std::get<3>(shape), std::get<4>(shape), std::get<5>(shape)}] +=
        count;
  work.prefixScanCounts = std::move(scans);
  work.reductionOperations /= parts;
  work.shuffleLaneSteps /= parts;
  work.dotFlops /= parts;
  work.sharedLocalLoadInstructions /= parts;
  work.sharedLocalStoreInstructions /= parts;
  work.privateStackLoadInstructions /= parts;
  work.privateStackStoreInstructions /= parts;
  // Dot startup is still paid per call. Do not divide its reduction/K depth.
  std::map<int64_t, double> dots;
  for (const auto &[flops, count] : work.dotFlopCounts)
    dots[extent(flops)] += count;
  work.dotFlopCounts = std::move(dots);

  // Remap a one-dimensional, independent access BEFORE selecting size curves.
  for (auto &access : work.memoryAccesses) {
    const auto originalShape = accessShape(access);
    if (!originalShape || access.shape.size() != 1 ||
        stage.features.hasReduction)
      continue;
    const int64_t elements = originalShape->second;
    auto axis =
        llvm::find_if(access.shape, [](int64_t dim) { return dim > 1; });
    if (axis == access.shape.end())
      continue;
    const int64_t oldDim = *axis;
    *axis = extent(oldDim);
    const int64_t sliced = elements / oldDim * *axis;
    const int64_t bytes = originalShape->first;
    const int64_t slicedBytes =
        sliced * llvm::divideCeil(access.elementBits, int64_t(8));
    double &totalBytes = access.store ? work.storeBytes : work.loadBytes;
    double &warps =
        access.store ? work.storeWarpInstructions : work.loadWarpInstructions;
    totalBytes -= access.count * (bytes - slicedBytes);
    warps -= access.count * (llvm::divideCeil(elements, int64_t(32)) -
                             llvm::divideCeil(sliced, int64_t(32)));
    auto &shapes =
        access.store ? work.continuousStoreCounts : work.continuousLoadCounts;
    auto original = shapes.find({bytes, elements});
    if (original != shapes.end() && original->second >= access.count) {
      original->second -= access.count;
      shapes[{slicedBytes, sliced}] += access.count;
    }
  }
  // Preserve scalar/address/control work; only tensor lane work is shared.
  work.issueElements =
      work.scalarOperations +
      32 * (work.loadWarpInstructions + work.storeWarpInstructions +
            work.sharedLocalLoadInstructions +
            work.sharedLocalStoreInstructions +
            work.privateStackLoadInstructions +
            work.privateStackStoreInstructions);
  for (const auto &entry : work.operationCounts)
    for (const auto &[elements, count] : entry.second)
      work.issueElements += elements * count;
  for (const auto &[elements, count] : work.predicateCounts)
    work.issueElements += elements * count;
  return stage;
}

static double estimateStage(const LogicalStage &stage,
                            const HardwareProfile &profile, StageMode mode,
                            const StageResourceCycles &resources);

static double applySuperBlock(const LogicalStage &stage,
                              const StageResourceCycles &resources,
                              const StageImplementation &implementation,
                              const HardwareProfile &profile,
                              double stageCycles) {
  if (implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor == 1)
    return stageCycles;

  const double factor = static_cast<double>(implementation.superblockFactor);
  if (stage.costModelKind == StageCostModelKind::RowwiseReduction &&
      resources.rowwiseReductionGroupBody > 0.0)
    return stageCycles;
  // SuperBlock creates `factor` independent logical-program groups on one
  // physical core.
  const double fixed = resources.setup;
  const double issueFloor =
      fixed + factor * iterations(stage) * resources.issue;
  if (stage.costModelKind == StageCostModelKind::ScalarStore &&
      stage.workload.scalarStoreCount > 0.0) {
    // One uniform scalar store from each resident warp is a single physical
    // SuperBlock latency envelope.
    const double oneGroupLatency =
        iterations(stage) *
        std::max(resources.scalar + resources.store, resources.issue);
    return fixed + std::max(oneGroupLatency,
                            factor * iterations(stage) * resources.issue);
  }
  if (stage.costModelKind == StageCostModelKind::RowwiseReduction &&
      resources.rowwiseReductionGroupBody == 0.0) {
    int64_t dependencyCopies = implementation.superblockFactor;
    bool exactGeometry = !stage.workload.reductionGeometryCounts.empty();
    int64_t maximumLaneChunks = 1;
    for (const auto &[shape, count] : stage.workload.reductionGeometryCounts) {
      const int64_t extent = std::get<1>(shape);
      if (count <= 0.0 || extent <= 1) {
        exactGeometry = false;
        break;
      }
      const int64_t lanesCoveredByResidentGroup =
          std::max<int64_t>(1, profile.baseSimtWarpCount) *
          implementation.superblockFactor * 32;
      maximumLaneChunks = std::max(maximumLaneChunks,
                                   static_cast<int64_t>(llvm::divideCeil(
                                       extent, lanesCoveredByResidentGroup)));
    }
    if (exactGeometry)
      dependencyCopies =
          std::min<int64_t>(implementation.superblockFactor, maximumLaneChunks);
    const double oneGroupDependency =
        iterations(stage) *
        std::max(resources.scalar + resources.load + resources.store +
                     resources.criticalPath,
                 resources.issue);
    return fixed +
           std::max(static_cast<double>(dependencyCopies) * oneGroupDependency,
                    factor * iterations(stage) * resources.issue);
  }
  // A recurrence is serial inside one logical program.
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence) {
    const double oneProgramExecution =
        resources.scalar + resources.load + resources.store + resources.atomic +
        resources.sharedLayout + resources.compute + resources.predicate +
        resources.reduction + resources.shuffle + resources.dot +
        controlBody(resources) + resources.spill;
    // A measured indirect-load group replaces the corresponding analytical load
    // instructions.
    const double recurrenceMemoryService =
        resources.load + resources.store + resources.atomic +
        resources.continuousLoadGroup + resources.continuousStoreGroup +
        resources.indirectLoadGroup + resources.sharedLayout;
    const double oneProgramDependency =
        resources.criticalPath > 0.0
            ? std::max(resources.criticalPath + recurrenceMemoryService,
                       resources.issue)
            : std::max(oneProgramExecution, resources.issue);
    const double groupBody = std::max(
        {oneProgramDependency, factor * oneProgramExecution,
         factor * resources.issue, resources.recurrenceGroupCriticalPath});
    return fixed + resources.recurrenceGroupStartup +
           iterations(stage) *
               (groupBody + factor * resources.privateStackCombinedFloor);
  }
  if ((stage.costModelKind == StageCostModelKind::ContinuousTileMemory ||
       stage.costModelKind == StageCostModelKind::ContinuousTileStore ||
       stage.costModelKind == StageCostModelKind::ContinuousShortLoad ||
       stage.costModelKind == StageCostModelKind::CachePolicyStore ||
       stage.costModelKind == StageCostModelKind::IndirectScalarMemory ||
       stage.costModelKind == StageCostModelKind::IndirectGatherMemory) &&
      (resources.continuousLoadGroup > 0.0 ||
       resources.continuousStoreGroup > 0.0 ||
       resources.indirectLoadGroup > 0.0)) {
    // Measured envelopes already cover F programs.
    auto grouped = resources;
    for (double *term :
         {&grouped.scalar, &grouped.load, &grouped.store, &grouped.sharedLayout,
          &grouped.compute, &grouped.operationCompletion, &grouped.predicate,
          &grouped.reduction, &grouped.scalarizedMultiResultReduction,
          &grouped.sort, &grouped.shuffle, &grouped.dot,
          &grouped.privateStackCombinedFloor, &grouped.issue})
      *term *= factor;
    return estimateStage(stage, profile, implementation.mode, grouped);
  }
  const double body = std::max(0.0, stageCycles - fixed);
  // Resource rates describe aggregate throughput of the selected VF (already
  // resolved at baseWarps * F).
  return std::max(issueFloor, fixed + factor * body);
}

static double estimateStageBase(const LogicalStage &stage,
                                const HardwareProfile &profile, StageMode mode,
                                const StageResourceCycles &r) {
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
  switch (stage.costModelKind) {
  case StageCostModelKind::VectorIssue:
    return serial;
  case StageCostModelKind::AutoBlockifyDispatch:
  case StageCostModelKind::AutoBlockifyLoop: {
    const double dispatchCount =
        stage.costModelKind == StageCostModelKind::AutoBlockifyLoop ? count
                                                                    : 1.0;
    return r.setup + dispatchCount * std::max(r.scalar, r.issue);
  }
  case StageCostModelKind::ContinuousTileMemory:
  case StageCostModelKind::ContinuousTileStore:
  case StageCostModelKind::ContinuousShortLoad:
  case StageCostModelKind::CachePolicyStore:
    if (r.continuousLoadGroup > 0.0 || r.continuousStoreGroup > 0.0) {
      const double measuredMemory =
          r.continuousLoadGroup + r.continuousStoreGroup;
      const double otherExecution = r.scalar + r.load + r.store + r.compute +
                                    r.predicate + r.reduction + r.shuffle +
                                    r.dot + r.sharedLayout;
      // Covered and uncovered transfers are disjoint workloads.
      return r.setup + count * std::max({measuredMemory + r.load + r.store +
                                             r.sharedLayout,
                                         otherExecution, r.issue});
    }
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      // A memory-classified Stage may still own fused TTIR arithmetic.
      const double vectorExecution =
          r.compute + r.dot + r.reduction + r.shuffle;
      return r.setup +
             count * std::max(r.scalar + r.predicate +
                                  std::max({r.load, r.store, vectorExecution}),
                              r.issue);
    }
    return serial;
  case StageCostModelKind::AtomicMemory:
    return serial;
  case StageCostModelKind::IndependentPipelinedLoop:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      const double vectorService =
          std::max(r.compute + r.dot + r.reduction + r.shuffle, r.issue);
      const double loadService =
          r.load + r.continuousLoadGroup + r.indirectLoadGroup + r.sharedLayout;
      const double storeService = r.store + r.continuousStoreGroup;
      // Model one source iteration as competing engine resources.
      return r.setup +
             count * std::max({loadService, storeService, vectorService,
                               vectorService + r.irregularTailMaterialization,
                               r.irregularMaskedTileFloor, r.atomic,
                               r.scalar + r.predicate});
    }
    return serial;
  case StageCostModelKind::LoopCarriedRecurrence: {
    // Loads, stores and layout conversion feed the next recurrence value.
    const double recurrenceMemoryService =
        r.load + r.store + r.continuousLoadGroup + r.continuousStoreGroup +
        r.indirectLoadGroup + r.sharedLayout;
    const double dependencyFactor =
        mode == StageMode::SIMD ? profile.simd.prefixScanDependencyFactor
                                : profile.simt.prefixScanDependencyFactor;
    const double scanDependency =
        std::max(r.prefixScanCriticalPath,
                 r.scanShuffle * std::max(1.0, dependencyFactor));
    const double dependencyBody = std::max(r.criticalPath, scanDependency);
    const double analyticalCritical =
        dependencyBody > 0.0
            ? std::max(dependencyBody + recurrenceMemoryService + r.atomic,
                       r.issue)
            : serialBody(r);
    const double critical =
        std::max(analyticalCritical, r.recurrenceGroupCriticalPath);
    // Workload is already the aggregate work of all owned loops divided by
    // count.
    return r.setup + r.recurrenceGroupStartup + count * critical;
  }
  case StageCostModelKind::RowwiseReduction:
    if (mode == StageMode::SIMT && r.multiResultReductionGroupBody > 0.0)
      return r.setup +
             count * std::max(r.multiResultReductionGroupBody, r.issue);
    if (mode == StageMode::SIMT && r.rowwiseReductionGroupBody > 0.0)
      return r.setup + count * std::max(r.rowwiseReductionGroupBody, r.issue);
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage)) {
      // SIMD rowwise lowering exposes pointwise and reduction service as
      // competing vector-pipe bounds.
      const double vectorExecution =
          r.predicate + std::max(r.compute + r.dot, r.reduction + r.shuffle) +
          r.irregularPhysicalMaterialization + r.irregularTailMaterialization;
      return r.setup +
             count * std::max({r.load + r.continuousLoadGroup + vectorExecution,
                               r.store + r.continuousStoreGroup,
                               r.irregularMaskedTileFloor,
                               r.scalar + r.predicate, r.issue});
    }
    return r.setup +
           count *
               std::max(r.scalar + r.load + r.store + r.criticalPath, r.issue);
  case StageCostModelKind::PrefixScan: {
    if (r.gridStridePrefixScanBody > 0.0) {
      const double ownedMemory = r.load + r.store + r.continuousLoadGroup +
                                 r.continuousStoreGroup + r.indirectLoadGroup +
                                 r.sharedLayout;
      return r.setup +
             count * std::max(r.gridStridePrefixScanBody +
                                  (mode == StageMode::SIMT ? ownedMemory : 0.0),
                              r.issue);
    }
    if (mode == StageMode::SIMT &&
        stage.scheduleKind == StageScheduleKind::StraightLine &&
        !stage.features.isGridStrideLoopShell &&
        !stage.features.insideGridStrideLoopBody &&
        profile.simt.straightLinePrefixScanElementsPerCycle > 0.0) {
      double semanticElements = 0.0;
      for (const auto &[shape, occurrences] : stage.workload.prefixScanCounts) {
        semanticElements += static_cast<double>(std::get<0>(shape)) *
                            static_cast<double>(std::get<1>(shape)) *
                            occurrences;
      }
      const double ownedMemory = r.load + r.store + r.continuousLoadGroup +
                                 r.continuousStoreGroup + r.indirectLoadGroup +
                                 r.sharedLayout;
      return count * (ownedMemory +
                      semanticElements /
                          profile.simt.straightLinePrefixScanElementsPerCycle);
    }
    const double scanExecution =
        r.compute + r.predicate + r.reduction + r.shuffle;
    const double scanCritical =
        std::max(scanExecution, r.prefixScanCriticalPath);
    const double execution = r.scalar + r.load + r.store + scanCritical;
    return r.setup + count * std::max(execution, r.issue);
  }
  case StageCostModelKind::CubeRoofline:
  case StageCostModelKind::TinyCubeRoofline:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup +
             count *
                 std::max(r.scalar + r.predicate + r.reduction + r.shuffle +
                              std::max({r.load, r.compute + r.dot, r.store}),
                          r.issue);
    return serial;
  case StageCostModelKind::ConversionPack:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup +
             count * std::max(r.predicate + std::max({r.scalar + r.compute,
                                                      r.load, r.store}),
                              r.issue);
    return serial;
  default:
    if (mode == StageMode::SIMD) {
      // Measured group envelopes replace the analytical bytes/transactions that
      // they cover.
      const double loadService =
          r.load + r.continuousLoadGroup + r.indirectLoadGroup + r.sharedLayout;
      const double storeService = r.store + r.continuousStoreGroup;
      const double vectorService = r.compute + r.dot + r.reduction + r.shuffle;
      return r.setup +
             count *
                 (std::max({loadService, storeService, r.atomic, vectorService,
                            r.scalar + r.predicate + controlBody(r), r.issue}) +
                  r.spill);
    }
    return serial;
  }
}

static double estimateStage(const LogicalStage &stage,
                            const HardwareProfile &profile, StageMode mode,
                            const StageResourceCycles &r) {
  StageResourceCycles base = r;
  base.operationCompletion = 0.0;
  base.privateStackCombinedFloor = 0.0;
  base.loopControl = 0.0;
  base.branchControl = 0.0;
  base.divergence = 0.0;
  base.synchronization = 0.0;
  base.spill = 0.0;
  return estimateStageBase(stage, profile, mode, base) +
         iterations(stage) *
             (r.operationCompletion + r.privateStackCombinedFloor +
              controlBody(r) + r.spill);
}

static bool modeSupportsPrefixScan(const LogicalStage &stage,
                                   const StageModeProfile &profile,
                                   StageMode mode) {
  if (mode != StageMode::SIMD || profile.maximumReversePrefixScanExtent <= 0)
    return true;
  return llvm::all_of(stage.workload.prefixScanCounts,
                      [&](const auto &shapeCount) {
                        return !std::get<4>(shapeCount.first) ||
                               std::get<0>(shapeCount.first) <=
                                   profile.maximumReversePrefixScanExtent;
                      });
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
  case StageCostModelKind::VectorIssue:
    return "vector_issue";
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
         std::isfinite(singlePhysicalProgramLogicalElementsPerCycle) &&
         singlePhysicalProgramLogicalElementsPerCycle >= 0.0 &&
         std::isfinite(operationStartupCycles) &&
         operationStartupCycles >= 0.0 &&
         std::isfinite(resultDependencyCycles) &&
         resultDependencyCycles >= 0.0 &&
         std::isfinite(unknownContentionMultiplier) &&
         unknownContentionMultiplier >= 1.0;
}

bool StraightLinePrefixScanCost::isValid() const {
  const bool validSingleSequenceCurve =
      llvm::all_of(singleSequenceCompletionCyclesByExtent, [](const auto &it) {
        return it.first > 0 && std::isfinite(it.second) && it.second >= 0.0;
      });
  const bool hasNonUnitStrideCompletion =
      nonUnitStrideElementsPerCycle != 0.0 ||
      nonUnitStrideStartupCycles != 0.0 ||
      nonUnitStrideMinimumAxisExtent != 0 ||
      nonUnitStrideMaximumAxisExtent != 0 ||
      nonUnitStrideMinimumAxisStride != 0 ||
      nonUnitStrideMaximumAxisStride != 0 ||
      nonUnitStrideMinimumSemanticElements != 0 ||
      nonUnitStrideMaximumSemanticElements != 0 ||
      !nonUnitStrideEvidence.empty();
  const bool validNonUnitStrideCompletion =
      !hasNonUnitStrideCompletion ||
      (std::isfinite(nonUnitStrideStartupCycles) &&
       nonUnitStrideStartupCycles >= 0.0 &&
       std::isfinite(nonUnitStrideElementsPerCycle) &&
       nonUnitStrideElementsPerCycle > 0.0 &&
       nonUnitStrideMinimumAxisExtent > 1 &&
       nonUnitStrideMaximumAxisExtent >= nonUnitStrideMinimumAxisExtent &&
       nonUnitStrideMinimumAxisStride > 1 &&
       nonUnitStrideMaximumAxisStride >= nonUnitStrideMinimumAxisStride &&
       nonUnitStrideMinimumSemanticElements > 0 &&
       nonUnitStrideMaximumSemanticElements >=
           nonUnitStrideMinimumSemanticElements &&
       !nonUnitStrideEvidence.empty());
  return validSingleSequenceCurve && validNonUnitStrideCompletion &&
         std::isfinite(startupCycles) && startupCycles >= 0.0 &&
         std::isfinite(elementsPerCycle) && elementsPerCycle > 0.0 &&
         std::isfinite(dependencyStepCycles) && dependencyStepCycles >= 0.0 &&
         !evidence.empty();
}

bool StageMemoryLayoutCost::isValid() const {
  return std::isfinite(firstCycles) && firstCycles >= 0.0 &&
         std::isfinite(incrementalCycles) && incrementalCycles >= 0.0 &&
         minCount >= 1 && maxCount >= minCount && !evidence.empty();
}

bool PipelinedStridedStoreGroupCost::isValid() const {
  return std::isfinite(kernelStartupCycles) && kernelStartupCycles >= 0.0 &&
         std::isfinite(firstStoreCycles) && firstStoreCycles > 0.0 &&
         std::isfinite(incrementalStoreCycles) &&
         incrementalStoreCycles > 0.0 && std::isfinite(extraLaneCycles) &&
         extraLaneCycles >= 0.0 && minimumRows > 0 &&
         maximumRows >= minimumRows && minimumStores > 0 &&
         maximumStores >= minimumStores && !evidence.empty();
}

bool MaskedStridedCopyCompletionCost::isValid() const {
  return std::isfinite(baseSystemCycles) && baseSystemCycles > 0.0 &&
         std::isfinite(additionalRowSystemCycles) &&
         additionalRowSystemCycles >= 0.0 && minimumRows > 0 &&
         maximumRows >= minimumRows && minimumIterations > 0 &&
         maximumIterations >= minimumIterations && !evidence.empty();
}

std::optional<double>
MaskedStridedCopyCompletionCost::evaluate(int64_t rows,
                                          int64_t iterations) const {
  if (!isValid() || rows < minimumRows || rows > maximumRows ||
      iterations < minimumIterations || iterations > maximumIterations)
    return std::nullopt;
  return baseSystemCycles +
         static_cast<double>(rows - minimumRows) * additionalRowSystemCycles;
}

bool SimtSourceFit::isValid() const {
  const std::array<double, 7> coefficients = {
      intercept,         directInstructionWeight,
      segmentWeight,     strideRatioWeight,
      maskedWeight,      peakLiveRegisterWeight,
      unrollFactorWeight};
  return std::all_of(
      coefficients.begin(), coefficients.end(),
      [](double value) { return std::isfinite(value) && value >= 0.0; });
}

bool DotComponentPrivateStackFloor::isValid() const {
  if (!enabled)
    return true;
  return activeWarpCount > 0 && superblockFactor > 0 &&
         minimumComponentNodes > 0 &&
         maximumComponentNodes >= minimumComponentNodes && m > 0 && n > 0 &&
         k > 0 && elementBits > 0 && !dotUseSignature.empty() &&
         loadInstructionFloor > 0 && storeInstructionFloor > 0 &&
         std::isfinite(combinedGroupCycles) && combinedGroupCycles > 0.0 &&
         std::isfinite(preRecurrenceFraction) && preRecurrenceFraction >= 0.0 &&
         std::isfinite(dotRegionFraction) && dotRegionFraction >= 0.0 &&
         std::isfinite(postDotFraction) && postDotFraction >= 0.0 &&
         std::abs(preRecurrenceFraction + dotRegionFraction + postDotFraction -
                  1.0) < 1.0e-9 &&
         !evidence.empty();
}

bool WholeSimtCompletionCurve::isValid() const {
  if (!enabled)
    return true;
  const auto allFinite = [](const std::array<double, 8> &values) {
    return llvm::all_of(values,
                        [](double value) { return std::isfinite(value); });
  };
  return activeWarpCount > 0 && superblockFactor > 0 &&
         minimumComponentNodes > 0 &&
         maximumComponentNodes >= minimumComponentNodes && m > 0 && n > 0 &&
         k > 0 && elementBits > 0 && !dotUseSignature.empty() &&
         referenceRowStrideBytes > 0 &&
         std::isfinite(referencePreRecurrenceScalarOperations) &&
         referencePreRecurrenceScalarOperations >= 0.0 &&
         std::isfinite(recurrenceScale) && recurrenceScale > 0.0 &&
         std::isfinite(dotRegionScale) && dotRegionScale > 0.0 &&
         std::isfinite(entryGroupMidpoint) && entryGroupMidpoint > 0.0 &&
         std::isfinite(entryGroupPower) && entryGroupPower > 0.0 &&
         allFinite(entryAdditiveCoefficients) &&
         ((exitRampExponential && std::isfinite(exitGroupTimeConstant) &&
           exitGroupTimeConstant > 0.0) ||
          (!exitRampExponential && std::isfinite(exitGroupMidpoint) &&
           exitGroupMidpoint > 0.0 && std::isfinite(exitGroupPower) &&
           exitGroupPower > 0.0)) &&
         allFinite(exitAbsoluteCoefficients) &&
         std::isfinite(exitTailLog2ActiveFractionCoefficient) &&
         std::isfinite(kernelBoundarySystemCycles) &&
         kernelBoundarySystemCycles >= 0.0 && !evidence.empty();
}

bool WholeSimdCompletionCurve::isValid() const {
  if (!enabled)
    return true;
  const auto positiveFinite = [](double value) {
    return std::isfinite(value) && value > 0.0;
  };
  return minimumComponentNodes > 0 &&
         maximumComponentNodes >= minimumComponentNodes && m > 0 && n > 0 &&
         k > 0 && elementBits > 0 && !dotUseSignature.empty() &&
         minimumRowStrideBytes > 0 &&
         maximumRowStrideBytes >= minimumRowStrideBytes &&
         minimumGroupCount > 0 && maximumGroupCount >= minimumGroupCount &&
         positiveFinite(entryBaseScale) &&
         positiveFinite(entrySaturatedScale) &&
         positiveFinite(entryUnalignedGroupMidpoint) &&
         positiveFinite(entryGroupMidpoint) &&
         positiveFinite(entryGroupPower) &&
         positiveFinite(singleWaveEntryBaseSystemCycles) &&
         positiveFinite(singleWaveEntrySaturatedSystemCycles) &&
         positiveFinite(singleWaveEntrySlotMidpoint) &&
         positiveFinite(singleWaveEntrySlotPower) &&
         positiveFinite(twoWaveEntryBaseSystemCycles) &&
         positiveFinite(twoWaveEntryGaussianSystemCycles) &&
         std::isfinite(twoWaveEntryTailCenter) &&
         twoWaveEntryTailCenter >= 0.0 &&
         positiveFinite(twoWaveEntryTailWidth) &&
         std::isfinite(twoWaveEntryHighTailPenaltySystemCycles) &&
         twoWaveEntryHighTailPenaltySystemCycles >= 0.0 &&
         positiveFinite(twoWaveEntryHighTailMidpoint) &&
         positiveFinite(twoWaveEntryHighTailPower) &&
         std::isfinite(twoWaveEntryFullTailSystemCycles) &&
         twoWaveEntryFullTailSystemCycles >= 0.0 &&
         std::isfinite(recurrenceSetupSystemCyclesPerGroup) &&
         recurrenceSetupSystemCyclesPerGroup >= 0.0 &&
         positiveFinite(dotRegionScale) &&
         positiveFinite(singleWaveDotBaseSystemCycles) &&
         positiveFinite(singleWaveDotSaturatedSystemCycles) &&
         positiveFinite(singleWaveDotSlotMidpoint) &&
         positiveFinite(singleWaveDotSlotPower) &&
         positiveFinite(twoWaveDotTargetSystemCycles) &&
         positiveFinite(exitBaseScale) && positiveFinite(exitSaturatedScale) &&
         positiveFinite(exitUnalignedGroupMidpoint) &&
         positiveFinite(exitGroupMidpoint) && positiveFinite(exitGroupPower) &&
         positiveFinite(singleWaveExitBaseSystemCycles) &&
         positiveFinite(singleWaveExitSaturatedSystemCycles) &&
         positiveFinite(singleWaveExitSlotMidpoint) &&
         positiveFinite(singleWaveExitSlotPower) &&
         positiveFinite(twoWaveExitBaseSystemCycles) &&
         positiveFinite(twoWaveExitGaussianSystemCycles) &&
         std::isfinite(twoWaveExitTailCenter) && twoWaveExitTailCenter >= 0.0 &&
         positiveFinite(twoWaveExitTailWidth) &&
         std::isfinite(kernelBoundarySystemCycles) &&
         kernelBoundarySystemCycles >= 0.0 && !evidence.empty();
}

double SimtSourceFit::evaluate(double directInstructions, double segments,
                               double strideRatio, double maskedAccesses,
                               double peakLiveExcess,
                               double unrollFactor) const {
  if (!enabled || !isValid())
    return 0.0;
  const double result =
      intercept + directInstructionWeight * std::max(0.0, directInstructions) +
      segmentWeight * std::max(0.0, segments) +
      strideRatioWeight * std::max(0.0, strideRatio) +
      maskedWeight * std::max(0.0, maskedAccesses) +
      peakLiveRegisterWeight * std::max(0.0, peakLiveExcess) +
      unrollFactorWeight * std::max(0.0, unrollFactor - 1.0);
  return std::isfinite(result) ? std::max(0.0, result) : 0.0;
}

std::optional<double> StageMemoryLayoutCost::evaluate(double count) const {
  if (!isValid() || !std::isfinite(count) || count < 0.0)
    return std::nullopt;
  if (count == 0.0)
    return 0.0;
  if (count < minCount || count > maxCount || std::floor(count) != count)
    return std::nullopt;
  const double cost = firstCycles + (count - 1.0) * incrementalCycles;
  return std::isfinite(cost) ? std::optional<double>(cost) : std::nullopt;
}

namespace {

static bool finitePositive(std::initializer_list<double> values) {
  return llvm::all_of(
      values, [](double value) { return std::isfinite(value) && value > 0.0; });
}

static bool finiteNonNegative(std::initializer_list<double> values) {
  return llvm::all_of(values, [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

template <typename Map, typename LeafPredicate>
static bool validPositiveKeyTree(const Map &values,
                                 const LeafPredicate &validLeaf) {
  return llvm::all_of(values, [&](const auto &entry) {
    if (entry.first <= 0)
      return false;
    using Mapped = typename Map::mapped_type;
    if constexpr (std::is_arithmetic_v<Mapped>) {
      return validLeaf(entry.second);
    } else {
      return !entry.second.empty() &&
             validPositiveKeyTree(entry.second, validLeaf);
    }
  });
}

template <typename Range>
static bool allProfileEntriesValid(const Range &entries) {
  return llvm::all_of(entries,
                      [](const auto &entry) { return entry.isValid(); });
}

template <typename Curve>
static bool sameDotDomain(const Curve &lhs, const Curve &rhs) {
  return lhs.m == rhs.m && lhs.n == rhs.n && lhs.k == rhs.k &&
         lhs.elementBits == rhs.elementBits &&
         lhs.dotUseSignature == rhs.dotUseSignature;
}

template <typename Curve>
static bool overlappingNodeRanges(const std::vector<Curve> &curves,
                                  bool matchWarpDomain) {
  for (auto lhs = curves.begin(); lhs != curves.end(); ++lhs)
    for (auto rhs = std::next(lhs); rhs != curves.end(); ++rhs) {
      if (!lhs->enabled || !rhs->enabled)
        continue;
      const bool sameWarpDomain =
          !matchWarpDomain || (lhs->activeWarpCount == rhs->activeWarpCount &&
                               lhs->superblockFactor == rhs->superblockFactor);
      if (sameWarpDomain && sameDotDomain(*lhs, *rhs) &&
          std::max(lhs->minimumComponentNodes, rhs->minimumComponentNodes) <=
              std::min(lhs->maximumComponentNodes, rhs->maximumComponentNodes))
        return true;
    }
  return false;
}

} // namespace

bool StageModeProfile::isValid(StageMode mode) const {
  if (!finitePositive({setupCycles, predicateOperationsPerCycle,
                       shuffleLanesPerCycle, dotSetupCycles, dotFlopsPerCycle,
                       issueOperationsPerCycle, prefixScanStepLatencyCycles,
                       prefixScanDependencyFactor,
                       static_cast<double>(vectorWidthBits),
                       static_cast<double>(vectorWidth),
                       static_cast<double>(issueWidth)}) ||
      !finiteNonNegative({loadOperationSetupCycles, storeOperationSetupCycles,
                          reductionOperationSetupCycles,
                          reductionLaneStepsPerCycle,
                          maskedAccumulatorReductionCompletionSetupCycles}) ||
      maximumReversePrefixScanExtent < 0)
    return false;
  const bool hasScalarizedMultiResultReduction =
      scalarizedMultiResultReductionMinimumExtent != 0 ||
      scalarizedMultiResultReductionReferenceResults != 0 ||
      scalarizedMultiResultReductionSetupCycles != 0.0 ||
      scalarizedMultiResultReductionElementsPerCycle != 0.0 ||
      !scalarizedMultiResultReductionEvidence.empty();
  if (hasScalarizedMultiResultReduction &&
      (mode != StageMode::SIMD ||
       scalarizedMultiResultReductionMinimumExtent <= 1 ||
       scalarizedMultiResultReductionReferenceResults <= 1 ||
       !std::isfinite(scalarizedMultiResultReductionSetupCycles) ||
       scalarizedMultiResultReductionSetupCycles < 0.0 ||
       !std::isfinite(scalarizedMultiResultReductionElementsPerCycle) ||
       scalarizedMultiResultReductionElementsPerCycle <= 0.0 ||
       scalarizedMultiResultReductionEvidence.empty()))
    return false;
  const bool hasSort = sortMinimumExtent != 0 || sortMaximumExtent != 0 ||
                       sortElementBits != 0 || sortSetupCycles != 0.0 ||
                       sortRowsPerCycle != 0.0 || !sortEvidence.empty();
  if (hasSort && (mode != StageMode::SIMD || sortMinimumExtent <= 1 ||
                  sortMaximumExtent < sortMinimumExtent ||
                  sortElementBits <= 0 || !std::isfinite(sortSetupCycles) ||
                  sortSetupCycles < 0.0 || !std::isfinite(sortRowsPerCycle) ||
                  sortRowsPerCycle <= 0.0 || sortEvidence.empty()))
    return false;
  const bool hasStraightLinePrefixScanCompletion =
      straightLinePrefixScanElementsPerCycle != 0.0 ||
      !straightLinePrefixScanCompletionEvidence.empty();
  if (hasStraightLinePrefixScanCompletion &&
      (mode != StageMode::SIMT ||
       !std::isfinite(straightLinePrefixScanElementsPerCycle) ||
       straightLinePrefixScanElementsPerCycle <= 0.0 ||
       straightLinePrefixScanCompletionEvidence.empty()))
    return false;
  if (!globalLoadSourceFit.isValid() || !globalStoreSourceFit.isValid() ||
      !sharedLoadSourceFit.isValid() || !sharedStoreSourceFit.isValid() ||
      !privateLoadSourceFit.isValid() || !privateStoreSourceFit.isValid() ||
      !finiteNonNegative({registerBudgetPerThread, spillTransactionsPerCycle,
                          indirectLoadTransactionsPerCycle,
                          indirectStoreTransactionsPerCycle,
                          indirectDependencyLatencyCycles}) ||
      !finitePositive({spillTransactionCoverage}) || usableUbBytes < 0 ||
      !controlFlow.isFiniteAndNonNegative() ||
      (!atomicRates.empty() &&
       (!atomicRates.contains("default") ||
        !llvm::all_of(
            atomicRates,
            [](const auto &entry) { return entry.second.isValid(); }))))
    return false;
  if (irregularTileCompletionEnabled &&
      (irregularTilePhysicalThresholdElements < 0 ||
       !finiteNonNegative(
           {irregularTilePhysicalExcessCyclesPerElement,
            irregularTileActiveCyclesPerElement,
            irregularTileTailCyclesPerInactiveElement,
            irregularTileAlignedFirstRowGroupCyclesPerElement,
            irregularTileAlignedAdditionalRowGroupCyclesPerElement}) ||
       irregularTileRowGroupSize <= 0 ||
       irregularTileSmallRowMaximumRows <= 0 ||
       !std::isfinite(irregularTileLargeRowMaskedFloorScale) ||
       irregularTileLargeRowMaskedFloorScale <= 0.0 ||
       irregularTileLargeRowMaskedFloorScale > 1.0 ||
       !std::isfinite(
           irregularTileAlignedTwoRowGroupPhysicalMaterializationScale) ||
       irregularTileAlignedTwoRowGroupPhysicalMaterializationScale < 0.0 ||
       irregularTileAlignedTwoRowGroupPhysicalMaterializationScale > 1.0 ||
       irregularTileLargeWorkingSetFullPhysicalMinimumBytes <= 0 ||
       irregularTileCompletionEvidence.empty()))
    return false;
  if (mixedParentMemoryCompletionEnabled &&
      (mode != StageMode::SIMD || mixedParentMemoryMinimumIterations < 0 ||
       mixedParentMemoryReferenceBytesPerProgram <= 0 ||
       !std::isfinite(mixedParentMemoryCyclesPerExcessIteration) ||
       mixedParentMemoryCyclesPerExcessIteration <= 0.0 ||
       mixedParentMemoryCompletionEvidence.empty()))
    return false;
  const bool hasGridStrideResidentMemory =
      gridStrideResidentMemoryMaximumBytes != 0 ||
      gridStrideResidentLoadWarpInstructionsPerCycle != 0.0 ||
      gridStrideResidentStoreWarpInstructionsPerCycle != 0.0 ||
      !gridStrideResidentMemoryEvidence.empty();
  if (hasGridStrideResidentMemory &&
      (mode != StageMode::SIMT || gridStrideResidentMemoryMaximumBytes <= 0 ||
       !std::isfinite(gridStrideResidentLoadWarpInstructionsPerCycle) ||
       gridStrideResidentLoadWarpInstructionsPerCycle <= 0.0 ||
       !std::isfinite(gridStrideResidentStoreWarpInstructionsPerCycle) ||
       gridStrideResidentStoreWarpInstructionsPerCycle <= 0.0 ||
       gridStrideResidentMemoryEvidence.empty()))
    return false;
  const bool hasRowwiseReductionGroupSurface =
      rowwiseReductionGroupBaseWarpCount != 0 ||
      rowwiseReductionGroupReferenceOperations != 0 ||
      !rowwiseReductionGroupCyclesByFactorRowsAndExtent.empty() ||
      !rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent.empty() ||
      !rowwiseReductionGroupEvidence.empty();
  if (hasRowwiseReductionGroupSurface &&
      (mode != StageMode::SIMT || rowwiseReductionGroupBaseWarpCount <= 0 ||
       rowwiseReductionGroupReferenceOperations <= 0 ||
       rowwiseReductionGroupCyclesByFactorRowsAndExtent.empty() ||
       rowwiseReductionGroupEvidence.empty()))
    return false;
  if (!llvm::all_of(rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent,
                    [](const auto &operationEntry) {
                      return operationEntry.first > 0 &&
                             !operationEntry.second.empty();
                    }))
    return false;
  const bool hasMultiResultReductionGroupSurface =
      multiResultReductionGroupBaseWarpCount != 0 ||
      multiResultReductionResultCount != 0 ||
      multiResultReductionElementBits != 0 ||
      multiResultReductionPaddedAxisExtent != 0 ||
      !multiResultReductionGroupCyclesByTailClassAndRows.empty() ||
      !multiResultReductionGroupEvidence.empty();
  if (hasMultiResultReductionGroupSurface &&
      (mode != StageMode::SIMT || multiResultReductionGroupBaseWarpCount <= 0 ||
       multiResultReductionResultCount <= 1 ||
       multiResultReductionElementBits <= 0 ||
       multiResultReductionPaddedAxisExtent <= 1 ||
       !llvm::isPowerOf2_64(
           static_cast<uint64_t>(multiResultReductionPaddedAxisExtent)) ||
       multiResultReductionGroupCyclesByTailClassAndRows.empty() ||
       multiResultReductionGroupEvidence.empty()))
    return false;
  for (const auto &[tailClass, curve] :
       multiResultReductionGroupCyclesByTailClassAndRows) {
    if ((tailClass != "non_power_divisible_by_4" &&
         tailClass != "tail_mod_4_eq_2" && tailClass != "tail_odd") ||
        curve.empty() || !llvm::all_of(curve, [](const auto &point) {
          return point.first > 0 && std::isfinite(point.second) &&
                 point.second > 0.0;
        }))
      return false;
  }
  const bool hasTwoPassRowwiseCompletion =
      twoPassRowwiseOwnsSegmentedIndirectLoads ||
      !twoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot
           .empty() ||
      !twoPassRowwiseCompletionEvidence.empty();
  if (hasTwoPassRowwiseCompletion && twoPassRowwiseCompletionEvidence.empty())
    return false;
  if (twoPassRowwiseOwnsSegmentedIndirectLoads && mode != StageMode::SIMT)
    return false;
  if (!allProfileEntriesValid(dotComponentPrivateStackFloors) ||
      !allProfileEntriesValid(wholeSimtCompletionCurves) ||
      !allProfileEntriesValid(wholeSimdCompletionCurves) ||
      overlappingNodeRanges(dotComponentPrivateStackFloors, true) ||
      overlappingNodeRanges(wholeSimtCompletionCurves, true))
    return false;
  for (auto lhs = wholeSimdCompletionCurves.begin();
       lhs != wholeSimdCompletionCurves.end(); ++lhs)
    for (auto rhs = std::next(lhs); rhs != wholeSimdCompletionCurves.end();
         ++rhs) {
      if (!lhs->enabled || !rhs->enabled)
        continue;
      const bool sameStructuralDomain = sameDotDomain(*lhs, *rhs);
      const bool overlappingNodeRange =
          std::max(lhs->minimumComponentNodes, rhs->minimumComponentNodes) <=
          std::min(lhs->maximumComponentNodes, rhs->maximumComponentNodes);
      const bool overlappingStrideRange =
          std::max(lhs->minimumRowStrideBytes, rhs->minimumRowStrideBytes) <=
          std::min(lhs->maximumRowStrideBytes, rhs->maximumRowStrideBytes);
      const bool overlappingGroupRange =
          std::max(lhs->minimumGroupCount, rhs->minimumGroupCount) <=
          std::min(lhs->maximumGroupCount, rhs->maximumGroupCount);
      if (sameStructuralDomain && overlappingNodeRange &&
          overlappingStrideRange && overlappingGroupRange)
        return false;
    }
  if (!finiteNonNegative({sharedLocalLoadInstructionsPerCycle,
                          sharedLocalStoreInstructionsPerCycle,
                          privateStackLoadInstructionsPerCycle,
                          privateStackStoreInstructionsPerCycle}))
    return false;
  if (mode == StageMode::SIMD) {
    if (!(loadBytesPerCycle > 0.0 && storeBytesPerCycle > 0.0 &&
          reductionLaneStepsPerCycle > 0.0))
      return false;
    const bool hasIndependentLoadGroup =
        independentLoadGroupKernelStartupCycles != 0.0 ||
        independentLoadGroupIterationStartupCycles != 0.0 ||
        independentLoadGroupBytesPerCycle != 0.0 ||
        independentLoadGroupMinimumOperations != 0 ||
        independentLoadGroupMaximumOperations != 0;
    if (hasIndependentLoadGroup &&
        (!(std::isfinite(independentLoadGroupKernelStartupCycles) &&
           independentLoadGroupKernelStartupCycles >= 0.0 &&
           std::isfinite(independentLoadGroupIterationStartupCycles) &&
           independentLoadGroupIterationStartupCycles >= 0.0 &&
           std::isfinite(independentLoadGroupBytesPerCycle) &&
           independentLoadGroupBytesPerCycle > 0.0 &&
           independentLoadGroupMinimumOperations >= 1 &&
           independentLoadGroupMaximumOperations >=
               independentLoadGroupMinimumOperations)))
      return false;
    const bool hasComputedAddressScalarizedLoad =
        computedAddressScalarizedLoadMinimumElements != 0 ||
        computedAddressScalarizedLoadMaximumElements != 0 ||
        computedAddressScalarizedLoadCyclesPerElement != 0.0 ||
        !computedAddressScalarizedLoadEvidence.empty();
    if (hasComputedAddressScalarizedLoad &&
        (!(computedAddressScalarizedLoadMinimumElements >= 1 &&
           computedAddressScalarizedLoadMaximumElements >=
               computedAddressScalarizedLoadMinimumElements &&
           std::isfinite(computedAddressScalarizedLoadCyclesPerElement) &&
           computedAddressScalarizedLoadCyclesPerElement > 0.0 &&
           !computedAddressScalarizedLoadEvidence.empty())))
      return false;
    const bool hasControlFlowAddressScalarizedLoad =
        !controlFlowAddressScalarizedLoadResidualCyclesByElements.empty() ||
        !controlFlowAddressScalarizedLoadEvidence.empty();
    if (hasControlFlowAddressScalarizedLoad &&
        (controlFlowAddressScalarizedLoadResidualCyclesByElements.empty() ||
         controlFlowAddressScalarizedLoadEvidence.empty() ||
         !llvm::all_of(controlFlowAddressScalarizedLoadResidualCyclesByElements,
                       [](const auto &point) {
                         return point.first > 0 &&
                                std::isfinite(point.second) &&
                                point.second > 0.0;
                       })))
      return false;
    const bool hasMaskedLoadStoreRmw =
        !maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows.empty() ||
        !maskedLoadStoreRmwEvidence.empty();
    if (hasMaskedLoadStoreRmw &&
        (maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows.empty() ||
         maskedLoadStoreRmwEvidence.empty() ||
         !llvm::all_of(
             maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows,
             [](const auto &rowEntry) {
               return rowEntry.first > 0 && !rowEntry.second.empty() &&
                      llvm::all_of(rowEntry.second, [](const auto &point) {
                        return point.first > 0 && std::isfinite(point.second) &&
                               point.second > 0.0;
                      });
             })))
      return false;
    if (!llvm::all_of(pipelinedStridedStoreGroupsByRowStrideBytes,
                      [](const auto &entry) {
                        return entry.first > 0 && entry.second.isValid();
                      }))
      return false;
  } else if (!(loadWarpInstructionsPerCycle > 0.0 &&
               storeWarpInstructionsPerCycle > 0.0)) {
    return false;
  }
  auto validMaskedStridedCopyCurves = [](const auto &curves) {
    return llvm::all_of(curves, [](const auto &entry) {
      return entry.first > 0 && entry.second.isValid();
    });
  };
  if (!validMaskedStridedCopyCurves(
          maskedStridedLoadCopyCompletionByStrideBytes) ||
      !validMaskedStridedCopyCurves(
          maskedStridedStoreCopyCompletionByStrideBytes))
    return false;
  const bool hasRecurrenceGroupCurve =
      !recurrenceReductionGroupCyclesByFactorAndLaneSteps.empty() ||
      !recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps.empty();
  const bool hasSingleResultRowReductionRecurrenceCompletion =
      singleResultRowReductionRecurrenceCompletionEnabled ||
      singleResultRowReductionRecurrenceMinimumExtent != 0 ||
      singleResultRowReductionRecurrenceMaximumExtent != 0 ||
      singleResultRowReductionRecurrenceMinimumIterations != 0 ||
      singleResultRowReductionRecurrenceMaximumIterations != 0 ||
      singleResultRowReductionRecurrenceMinimumPrograms != 0 ||
      singleResultRowReductionRecurrenceMaximumPrograms != 0 ||
      singleResultRowReductionRecurrenceIterationCycles != 0.0 ||
      singleResultRowReductionRecurrenceStartupInterceptCycles != 0.0 ||
      singleResultRowReductionRecurrenceStartupSqrtProgramCycles != 0.0 ||
      !singleResultRowReductionRecurrenceEvidence.empty();
  const bool validSingleResultRowReductionRecurrenceCompletion =
      !hasSingleResultRowReductionRecurrenceCompletion ||
      (singleResultRowReductionRecurrenceCompletionEnabled &&
       singleResultRowReductionRecurrenceMinimumExtent > 0 &&
       singleResultRowReductionRecurrenceMaximumExtent >=
           singleResultRowReductionRecurrenceMinimumExtent &&
       singleResultRowReductionRecurrenceMinimumIterations > 0 &&
       singleResultRowReductionRecurrenceMaximumIterations >=
           singleResultRowReductionRecurrenceMinimumIterations &&
       singleResultRowReductionRecurrenceMinimumPrograms > 0 &&
       singleResultRowReductionRecurrenceMaximumPrograms >=
           singleResultRowReductionRecurrenceMinimumPrograms &&
       std::isfinite(singleResultRowReductionRecurrenceIterationCycles) &&
       singleResultRowReductionRecurrenceIterationCycles > 0.0 &&
       std::isfinite(
           singleResultRowReductionRecurrenceStartupInterceptCycles) &&
       std::isfinite(
           singleResultRowReductionRecurrenceStartupSqrtProgramCycles) &&
       singleResultRowReductionRecurrenceStartupSqrtProgramCycles > 0.0 &&
       singleResultRowReductionRecurrenceStartupInterceptCycles +
               singleResultRowReductionRecurrenceStartupSqrtProgramCycles *
                   std::sqrt(static_cast<double>(
                       singleResultRowReductionRecurrenceMinimumPrograms)) >=
           0.0 &&
       !singleResultRowReductionRecurrenceEvidence.empty());
  const bool hasContinuousMemoryGroupCurve =
      !continuousLoadGroupCyclesByFactorAndBytes.empty() ||
      !continuousStoreGroupCyclesByFactorAndBytes.empty() ||
      !continuousLoadGroupStartupCyclesByFactorAndBytes.empty() ||
      !continuousStoreGroupStartupCyclesByFactorAndBytes.empty();
  if (mode == StageMode::SIMT &&
      ((hasRecurrenceGroupCurve && recurrenceGroupBaseWarpCount <= 0) ||
       (hasContinuousMemoryGroupCurve &&
        continuousMemoryGroupBaseWarpCount <= 0)))
    return false;
  const auto positiveLeaf = [](double value) {
    return std::isfinite(value) && value > 0.0;
  };
  const auto finiteLeaf = [](double value) { return std::isfinite(value); };
  const auto validFactorCurves = [&](const auto &curves) {
    return validPositiveKeyTree(curves, positiveLeaf);
  };
  const auto validFactorSurfaces = [&](const auto &surfaces) {
    return validPositiveKeyTree(surfaces, finiteLeaf);
  };
  const bool validIndirectCurve = validPositiveKeyTree(
      indirectLoadSystemCyclesByWarpInstructions, positiveLeaf);
  const bool validIndirectElementCurve =
      validPositiveKeyTree(indirectLoadSystemCyclesByElements, positiveLeaf);
  const bool validIndirectStoreCurve =
      validPositiveKeyTree(indirectStoreSystemCyclesBySegments, positiveLeaf);
  const bool validDotWarpCurve =
      validPositiveKeyTree(dotFlopsPerCycleByWarpCount, positiveLeaf);
  const bool validIndirectStartupCurve = validPositiveKeyTree(
      indirectLoadStartupSystemCyclesByWarpInstructions,
      [](double value) { return std::isfinite(value) && value >= 0.0; });
  const bool hasGridStrideCompositionResidual =
      !gridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry
           .empty() ||
      !gridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry
           .empty() ||
      !gridStrideIndirectDotCompositionEvidence.empty();
  const bool validGridStrideCompositionResidual =
      !hasGridStrideCompositionResidual ||
      ((!gridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry
             .empty() ||
        !gridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry
             .empty()) &&
       !gridStrideIndirectDotCompositionEvidence.empty() &&
       validFactorSurfaces(
           gridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry) &&
       validFactorSurfaces(
           gridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry));
  const auto validLayouts = [](const auto &curves) {
    return llvm::all_of(curves, [](const auto &entry) {
      return !entry.first.empty() && entry.second.isValid();
    });
  };
  const bool validStoreService =
      storeFootprintUnitBytes == 0
          ? storeCyclesPerFootprintUnit == 0.0 &&
                storeIssueFloorByWarpCount.empty()
          : storeFootprintUnitBytes > 0 &&
                std::isfinite(storeCyclesPerFootprintUnit) &&
                storeCyclesPerFootprintUnit > 0.0 &&
                !storeIssueFloorByWarpCount.empty() &&
                llvm::all_of(storeIssueFloorByWarpCount, [](const auto &entry) {
                  return entry.first > 0 && std::isfinite(entry.second) &&
                         entry.second > 0.0;
                });
  return validDotWarpCurve && validIndirectCurve && validIndirectElementCurve &&
         validIndirectStoreCurve && validIndirectStartupCurve &&
         validStoreService && validGridStrideCompositionResidual &&
         validSingleResultRowReductionRecurrenceCompletion &&
         validFactorSurfaces(
             twoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot) &&
         validLayouts(loadLayoutCosts) && validLayouts(storeLayoutCosts) &&
         validLayouts(dotStoreSharedLayoutCosts) &&
         validLayouts(dotOperandPathCosts) &&
         validFactorCurves(
             recurrenceReductionGroupCyclesByFactorAndLaneSteps) &&
         validFactorCurves(
             recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps) &&
         validFactorCurves(continuousLoadGroupCyclesByFactorAndBytes) &&
         validFactorCurves(continuousStoreGroupCyclesByFactorAndBytes) &&
         validFactorCurves(continuousLoadGroupStartupCyclesByFactorAndBytes) &&
         validFactorCurves(continuousStoreGroupStartupCyclesByFactorAndBytes) &&
         validFactorCurves(gridStrideForwardScanCyclesByFactorAndExtent) &&
         validFactorCurves(
             gridStrideForwardScanStartupCyclesByFactorAndExtent) &&
         validFactorCurves(gridStrideReverseScanCyclesByFactorAndExtent) &&
         validFactorCurves(
             gridStrideReverseScanStartupCyclesByFactorAndExtent) &&
         validFactorCurves(
             gridStrideIndirectLoadCyclesByFactorAndWarpInstructions) &&
         validFactorCurves(
             gridStrideIndirectLoadStartupCyclesByFactorAndWarpInstructions) &&
         validFactorSurfaces(
             segmentedIndirectLoadGroupCyclesByFactorSegmentsAndContiguousElements) &&
         validFactorSurfaces(
             segmentedIndirectLoadFp32GroupCyclesByFactorSegmentsAndContiguousElements) &&
         validFactorCurves(gridStrideDotCyclesByFactorAndFlops) &&
         validFactorCurves(gridStrideDotStartupCyclesByFactorAndFlops) &&
         validPositiveKeyTree(rowwiseReductionGroupCyclesByFactorRowsAndExtent,
                              positiveLeaf) &&
         validPositiveKeyTree(
             rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent,
             positiveLeaf) &&
         llvm::all_of(
             operationRates,
             [](const auto &entry) {
               return std::isfinite(entry.second.throughput) &&
                      entry.second.throughput > 0.0 &&
                      std::isfinite(entry.second.factor) &&
                      entry.second.factor > 0.0 &&
                      std::isfinite(entry.second.startupCycles) &&
                      entry.second.startupCycles >= 0.0 &&
                      std::isfinite(entry.second.freeWorkUnits) &&
                      entry.second.freeWorkUnits >= 0.0 &&
                      entry.second.minimumStartupElements > 0 &&
                      std::isfinite(
                          entry.second.gridStrideKernelStartupCycles) &&
                      entry.second.gridStrideKernelStartupCycles >= 0.0;
             }) &&
         llvm::all_of(straightLinePrefixScanCostsByCombine,
                      [](const auto &entry) {
                        return !entry.first.empty() && entry.second.isValid();
                      });
}

bool HardwareProfile::isValid() const {
  return !profileVersion.empty() && !target.empty() && baseSimtWarpCount > 0 &&
         (cvAivSubblocks == 1 || cvAivSubblocks == 2) &&
         maximumSimtEstimationFactor >= 1 &&
         maximumSimtEstimationFactor <= 64 &&
         (maximumSimtEstimationFactor & (maximumSimtEstimationFactor - 1)) ==
             0 &&
         logicalWarpGroupCount > 0 && superblockUsefulFactorLimit > 0 &&
         superblockPersistentStatePressureFreeFactor > 0 &&
         superblockPersistentStatePressureFreeFactor <=
             superblockUsefulFactorLimit &&
         std::isfinite(superblockPersistentStateBytesPerCycle) &&
         superblockPersistentStateBytesPerCycle > 0.0 &&
         simd.isValid(StageMode::SIMD) && simt.isValid(StageMode::SIMT) &&
         scopeHandoff.isValid();
}

llvm::Expected<StageCostTable>
StageCostEvaluator::evaluate(const StagePartition &partition,
                             const HardwareProfile &profile,
                             bool collectCalibrationEstimates) const {
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
  table.cvAivSubblocks = profile.cvAivSubblocks;
  table.baseSimtWarpCount = profile.baseSimtWarpCount;
  table.simdLoadOperationStartupCycles = profile.simd.loadOperationSetupCycles;
  table.simdIndependentLoadGroupKernelStartupCycles =
      profile.simd.independentLoadGroupKernelStartupCycles;
  table.simtStraightLinePrefixScanElementsPerCycle =
      profile.simt.straightLinePrefixScanElementsPerCycle;
  table.simtStraightLinePrefixScanCompletionEvidence =
      profile.simt.straightLinePrefixScanCompletionEvidence;
  table.wholeSimdCompletionCurves = profile.simd.wholeSimdCompletionCurves;
  table.wholeSimtCompletionCurves = profile.simt.wholeSimtCompletionCurves;
  table.simdGridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry =
      profile.simd
          .gridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry;
  table.simdGridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry =
      profile.simd
          .gridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry;
  table.simdGridStrideIndirectDotCompositionEvidence =
      profile.simd.gridStrideIndirectDotCompositionEvidence;
  table.simdIrregularTileLargeWorkingSetFullPhysicalMinimumBytes =
      profile.simd.irregularTileLargeWorkingSetFullPhysicalMinimumBytes;
  table.simdIrregularTileCompletionEvidence =
      profile.simd.irregularTileCompletionEvidence;
  table.simtGridStrideResidentMemoryMaximumBytes =
      profile.simt.gridStrideResidentMemoryMaximumBytes;
  table.simtGridStrideResidentLoadWarpInstructionsPerCycle =
      profile.simt.gridStrideResidentLoadWarpInstructionsPerCycle;
  table.simtGridStrideResidentStoreWarpInstructionsPerCycle =
      profile.simt.gridStrideResidentStoreWarpInstructionsPerCycle;
  table.simtGridStrideResidentMemoryEvidence =
      profile.simt.gridStrideResidentMemoryEvidence;
  table.simtGridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry =
      profile.simt
          .gridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry;
  table.simtGridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry =
      profile.simt
          .gridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry;
  table.simtGridStrideIndirectDotCompositionEvidence =
      profile.simt.gridStrideIndirectDotCompositionEvidence;
  table.simtTwoPassRowwiseOwnsSegmentedIndirectLoads =
      profile.simt.twoPassRowwiseOwnsSegmentedIndirectLoads;
  table.simtTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot =
      profile.simt
          .twoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot;
  table.simtTwoPassRowwiseCompletionEvidence =
      profile.simt.twoPassRowwiseCompletionEvidence;
  table.simdTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot =
      profile.simd
          .twoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot;
  table.simdTwoPassRowwiseCompletionEvidence =
      profile.simd.twoPassRowwiseCompletionEvidence;
  const auto singleResultReduction = [](const StageModeProfile &mode) {
    return StageCostTable::SingleResultReductionCompletion{
        mode.singleResultRowReductionRecurrenceCompletionEnabled,
        mode.singleResultRowReductionRecurrenceMinimumExtent,
        mode.singleResultRowReductionRecurrenceMaximumExtent,
        mode.singleResultRowReductionRecurrenceMinimumIterations,
        mode.singleResultRowReductionRecurrenceMaximumIterations,
        mode.singleResultRowReductionRecurrenceMinimumPrograms,
        mode.singleResultRowReductionRecurrenceMaximumPrograms,
        mode.singleResultRowReductionRecurrenceIterationCycles,
        mode.singleResultRowReductionRecurrenceStartupInterceptCycles,
        mode.singleResultRowReductionRecurrenceStartupSqrtProgramCycles,
        mode.singleResultRowReductionRecurrenceEvidence};
  };
  table.simdSingleResultReduction = singleResultReduction(profile.simd);
  table.simtSingleResultReduction = singleResultReduction(profile.simt);
  table.mixedParentMemoryCompletionEnabled =
      profile.simd.mixedParentMemoryCompletionEnabled;
  table.mixedParentMemoryMinimumIterations =
      profile.simd.mixedParentMemoryMinimumIterations;
  table.mixedParentMemoryReferenceBytesPerProgram =
      profile.simd.mixedParentMemoryReferenceBytesPerProgram;
  table.mixedParentMemoryCyclesPerExcessIteration =
      profile.simd.mixedParentMemoryCyclesPerExcessIteration;
  table.mixedParentMemoryCompletionEvidence =
      profile.simd.mixedParentMemoryCompletionEvidence;
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
    if (stage.simtLegal && stage.legalSimtFactors.empty() &&
        (!stage.localSimtMaterializable || stage.localSimtFactors.empty()))
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
    logicalCost.wholeSimtRegisterPressure = stage.wholeSimtRegisterPressure;
    logicalCost.simtUbWorkingSetBytesPerProgram =
        stage.simtUbWorkingSetBytesPerProgram;
    logicalCost.simtUsableUbBytes = stage.simtUsableUbBytes;
    logicalCost.simtFactorsRejectedByUb = stage.simtFactorsRejectedByUb;
    logicalCost.scopeInputTensorBytes = stage.scopeInputTensorBytes;
    logicalCost.scopeOutputTensorBytes = stage.scopeOutputTensorBytes;
    logicalCost.simtAnchorIndices = stage.simtAnchorIndices;
    logicalCost.localSimtMaterializable = stage.localSimtMaterializable;
    logicalCost.localSuperblockMaterializable =
        stage.localSuperblockMaterializable;
    logicalCost.legalSimtFactors = stage.legalSimtFactors;
    logicalCost.localSimtFactors = stage.localSimtFactors;

    llvm::SmallVector<StageImplementation> implementations;
    if (stage.simdLegal &&
        modeSupportsPrefixScan(stage, profile.simd, StageMode::SIMD))
      implementations.push_back({StageMode::SIMD, 1, false});
    if (stage.simtLegal)
      for (int64_t factor : stage.legalSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, false});
    // Report mode estimates every Stage's SIMT body, whether or not an anchor
    // exists.
    for (int64_t factor : {1, 2, 4}) {
      if (factor > profile.maximumSimtEstimationFactor &&
          !llvm::is_contained(stage.localSimtFactors, factor))
        continue;
      StageImplementation local{StageMode::SIMT, factor, true};
      if (collectCalibrationEstimates || isDeclaredLegal(stage, local))
        implementations.push_back(local);
      if (profile.cvAivSubblocks > 1) {
        StageImplementation partitioned{StageMode::SIMT, factor, true,
                                        profile.cvAivSubblocks};
        if (collectCalibrationEstimates || isDeclaredLegal(stage, partitioned))
          implementations.push_back(partitioned);
      }
    }

    for (const StageImplementation &implementation : implementations) {
      const bool selectable = isDeclaredLegal(stage, implementation);
      if (!selectable && !implementation.localScope)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has an illegal candidate",
                                       stage.id.c_str());
      const StageModeProfile modeProfile = resolveModeProfile(
          profile, implementation,
          stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence &&
              stage.workload.dotFlops > 0.0);
      const LogicalStage executionStage =
          perAivStage(stage, implementation.aivSubblocks);
      std::optional<StageResourceCycles> resources = mapWorkload(
          executionStage, modeProfile, implementation.mode,
          implementation.superblockFactor,
          implementation.mode == StageMode::SIMT && !implementation.localScope);
      if (!resources)
        continue;
      StageImplementationCost cost;
      cost.implementation = implementation;
      cost.resources = *resources;
      if (implementation.aivSubblocks > 1)
        cost.perAivWorkload = executionStage.workload;
      if (resources->recurrenceGroupCriticalPath > 0.0 ||
          resources->recurrenceGroupStartup > 0.0 ||
          resources->multiResultReductionGroupBody > 0.0 ||
          resources->rowwiseReductionGroupBody > 0.0 ||
          resources->continuousLoadGroup > 0.0 ||
          resources->continuousStoreGroup > 0.0)
        cost.formulaEvidence = "factor_specific_measured_group_curve";
      else if (resources->storeLayoutService > 0.0)
        cost.formulaEvidence =
            "layout_store_service_floor_startup_uncalibrated";
      else if (stage.features.hasIndirectMemory) {
        if (resources->indirectMemoryAnalyticalFallback)
          cost.formulaEvidence = "analytical_indirect_memory_resource_fallback";
        else if (!stage.workload.indirectStoreCounts.empty() ||
                 !stage.workload.segmentedIndirectStoreCounts.empty())
          cost.formulaEvidence = "measured_indirect_store_segment_curve";
        else
          cost.formulaEvidence =
              extrapolatesIndirectEvidence(stage, modeProfile)
                  ? "measured_indirect_load_curve_extrapolated"
                  : "measured_indirect_load_curve";
      } else if (llvm::any_of(stage.workload.memoryAccesses,
                              [](const StageMemoryAccess &access) {
                                return access.layoutClass() !=
                                       StageMemoryAccess::LayoutClass::Unknown;
                              }))
        cost.formulaEvidence = "uncalibrated_layout_memory_fallback";
      else if (implementation.mode == StageMode::SIMT &&
               implementation.superblockFactor > 1)
        cost.formulaEvidence = "uncalibrated_superblock_fallback";
      else
        cost.formulaEvidence = "analytical_resource_formula";
      if (resources->indirectMemoryAnalyticalFallback &&
          cost.formulaEvidence.find(
              "analytical_indirect_memory_resource_fallback") ==
              std::string::npos)
        cost.formulaEvidence += "+analytical_indirect_memory_resource_fallback";
      if (resources->computedAddressScalarizedLoad > 0.0)
        cost.formulaEvidence +=
            "+simd_computed_address_scalarization_completion";
      if (resources->controlFlowAddressScalarizedLoad > 0.0)
        cost.formulaEvidence +=
            "+simd_control_flow_address_scalarization_completion";
      if (resources->maskedLoadStoreReadModifyWrite > 0.0)
        cost.formulaEvidence +=
            "+simd_masked_load_store_read_modify_write_completion";
      if (!stage.workload.segmentedIndirectLoadCounts.empty())
        cost.formulaEvidence +=
            resources->indirectMemoryAnalyticalFallback
                ? "+segmented_indirect_load_analytical_fallback"
                : "+measured_segmented_indirect_load_surface";
      if (!stage.workload.segmentedIndirectStoreCounts.empty() &&
          resources->indirectMemoryAnalyticalFallback)
        cost.formulaEvidence += "+segmented_indirect_store_analytical_fallback";
      if (resources->scalarizedMultiResultReduction > 0.0)
        cost.formulaEvidence += "+scalarized_multi_result_reduction_surface";
      if (resources->multiResultReductionGroupBody > 0.0)
        cost.formulaEvidence += "+simt_multi_result_reduction_tail_row_surface";
      if (resources->sort > 0.0)
        cost.formulaEvidence += "+native_sort_row_surface";
      if (stage.costModelKind == StageCostModelKind::VectorIssue)
        cost.formulaEvidence += "+elementwise_workload";

      bool usedSaturatedComputeFallback = false;
      bool hasUnprofiledExecution = false;
      for (const auto &entry : stage.workload.operationCounts) {
        // generic.issue is the intentional class for shape/view/layout
        // bookkeeping that consumes issue capacity but no FP32 ALU resource.
        if (entry.first() == "generic.issue")
          continue;
        // This reserved address class is deliberately issue-only for SIMT and
        // folded into the flat DMA descriptor for SIMD.
        if (entry.first() == "simd.folded_flat_address")
          continue;
        usedSaturatedComputeFallback |=
            usesSaturatedComputeFallback(modeProfile, entry.first());
        const StageOperationRate *rate =
            findOperationRate(modeProfile, entry.first());
        if (!rate || rate->throughput <= 0.0)
          hasUnprofiledExecution = true;
      }
      if (hasUnprofiledExecution)
        cost.formulaEvidence += "+unprofiled_vector_execution";
      if (usedSaturatedComputeFallback)
        cost.formulaEvidence += "+fp32_saturated_compute_fallback";
      if (implementation.mode == StageMode::SIMT && implementation.localScope)
        cost.localScopeCompletionInclusive = llvm::any_of(
            stage.workload.operationCounts, [&](const auto &entry) {
              const StageOperationRate *rate =
                  findOperationRate(modeProfile, entry.first());
              return rate && rate->includesLocalScopeHandoff;
            });
      if (cost.localScopeCompletionInclusive)
        cost.formulaEvidence += "+completion_inclusive_local_scope_operation";
      if (stage.workload.scalarOperations > 0.0)
        cost.formulaEvidence += "+unclassified_scalar_issue_floor";
      if (implementation.mode == StageMode::SIMD &&
          stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence &&
          stage.iterationCount > 1 &&
          (stage.workload.loadOperations > 0.0 ||
           stage.workload.storeOperations > 0.0))
        cost.formulaEvidence += "+resident_loop_transfer_setup_once";
      if (implementation.mode == StageMode::SIMD &&
          resources->rowwiseAdditionalLoadSetupsOwnedOnce)
        cost.formulaEvidence += "+resident_rowwise_additional_load_setups_once";
      if (implementation.mode == StageMode::SIMD &&
          stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop &&
          (resources->load + resources->continuousLoadGroup +
               resources->indirectLoadGroup + resources->sharedLayout >
           0.0) &&
          (resources->compute + resources->dot + resources->reduction +
               resources->shuffle >
           0.0))
        cost.formulaEvidence += "+serialized_iteration_resource_roofline";
      if (resources->sharedLayout > 0.0)
        cost.formulaEvidence += "+ttir_dot_store_shared_exchange";
      if (resources->irregularMaskedTileFloor > 0.0 ||
          resources->irregularTailMaterialization > 0.0)
        cost.formulaEvidence += "+generic_irregular_tile_completion";
      if (resources->dotOperandGroup > 0.0)
        cost.formulaEvidence += "+ttir_dot_operand_path_composite";
      if (resources->dotOperandGroup > 0.0 &&
          resources->dot * implementation.superblockFactor >
              resources->dotOperandGroup + 1e-9)
        cost.formulaEvidence += "+partial_composite_dot_residual";
      if (resources->dotInputLoadsCovered > 0.0)
        cost.formulaEvidence += "+gm_input_owned_by_dot_composite";
      if (resources->dotComponentPrivateStackFloorApplied)
        cost.formulaEvidence +=
            "+conditional_dot_component_private_stack_combined_group_floor";
      if (stage.workload.loopBackedges > 0.0 ||
          stage.workload.conditionalBranches > 0.0 ||
          stage.workload.divergentBranches > 0.0 ||
          stage.workload.synchronizations > 0.0)
        cost.formulaEvidence += "+control_issue_floor_only";
      if (!stage.workload.prefixScanCounts.empty()) {
        const bool usesStraightLineHelper =
            implementation.mode == StageMode::SIMD &&
            stage.scheduleKind == StageScheduleKind::StraightLine &&
            llvm::any_of(
                stage.workload.prefixScanCounts, [&](const auto &entry) {
                  return modeProfile.straightLinePrefixScanCostsByCombine.count(
                             std::get<5>(entry.first)) != 0;
                });
        cost.formulaEvidence +=
            usesStraightLineHelper ? "+straight_line_scan_structural_completion"
                                   : "+prefix_scan_per_level_issue";
      }
      if (resources->rowwiseReductionGroupBody > 0.0)
        cost.formulaEvidence += "+bounded_ttir_row_extent_surface";
      cost.totalCycles =
          applySuperBlock(executionStage, *resources, implementation, profile,
                          estimateStage(executionStage, profile,
                                        implementation.mode, *resources));
      if (implementation.aivSubblocks > 1) {
        cost.cvPartitionIdealTotalCycles = cost.totalCycles;
        cost.cvPartitionPayloadEfficiency =
            profile.scopeHandoff.cvPartitionPayloadEfficiency;
        // CV1:N is not guaranteed to realize an ideal N-way speedup.
        if (cost.cvPartitionPayloadEfficiency < 1.0) {
          auto fullResources =
              mapWorkload(stage, modeProfile, implementation.mode,
                          implementation.superblockFactor,
                          implementation.mode == StageMode::SIMT &&
                              !implementation.localScope);
          if (!fullResources)
            continue;
          const double unpartitionedTotal =
              applySuperBlock(stage, *fullResources, implementation, profile,
                              estimateStage(stage, profile, implementation.mode,
                                            *fullResources));
          cost.cvPartitionInefficiencyCorrectionCycles =
              (1.0 - cost.cvPartitionPayloadEfficiency) *
              std::max(0.0, unpartitionedTotal - cost.totalCycles);
          cost.totalCycles += cost.cvPartitionInefficiencyCorrectionCycles;
        }
        cost.formulaEvidence += "+cv_payload_partition_efficiency_envelope";
      }
      if (!cost.isValid())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' produced an invalid cost",
                                       stage.id.c_str());
      if (collectCalibrationEstimates && implementation.localScope)
        logicalCost.simtCalibrationEstimates.push_back(cost);
      if (selectable)
        logicalCost.implementations.push_back(std::move(cost));
    }

    if (logicalCost.implementations.empty())
      return llvm::createStringError(
          std::errc::not_supported,
          "Stage '%s' has no measured implementation for its workload",
          stage.id.c_str());

    table.stages.push_back(std::move(logicalCost));
  }
  return table;
}
