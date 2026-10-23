//===- SimdSimtCostModel.cpp - Ascend SIMD/SIMT candidate model ----------===//
//
// The numerical model in this file is the versioned C++ candidate model.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Analysis/LoopWorkloadAnalysis.h"
#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/Analysis/SimtRegisterPressureAnalysis.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/Profile/MicrobenchmarkProfile.h"
#include "AscendModel/RouteModel/StageCostModels.h"
#include "ascend/include/Utils/SuperBlockFactor.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

static bool finiteValues(std::initializer_list<double> values) {
  return llvm::all_of(values,
                      [](double value) { return std::isfinite(value); });
}

static bool finiteNonNegative(std::initializer_list<double> values) {
  return llvm::all_of(values, [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

static bool finitePositive(std::initializer_list<double> values) {
  return llvm::all_of(
      values, [](double value) { return std::isfinite(value) && value > 0.0; });
}

bool StageRouteCompletionProfile::MultiResultPrefixScan::isValid() const {
  if (!enabled)
    return true;
  return minimumPrograms > 0 && maximumPrograms >= minimumPrograms &&
         minimumPaddedAxis > 0 && maximumPaddedAxis >= minimumPaddedAxis &&
         requiredResultCount > 1 && requiredIndependentSequences > 0 &&
         minimumActiveRatio > 0.0 && minimumActiveRatio <= 1.0 &&
         maximumActiveRatio >= minimumActiveRatio &&
         maximumActiveRatio <= 1.0 &&
         finiteValues({minimumActiveRatio, maximumActiveRatio,
                       elapsedInterceptCycles, elapsedPaddedAxisCycles}) &&
         finiteNonNegative(
             {elapsedSqrtProgramCycles, elapsedInactiveRatioCycles}) &&
         !evidence.empty();
}

bool StageRouteCompletionProfile::PairwiseSelectionNetwork::isValid() const {
  if (!enabled)
    return true;
  return minimumInputElements > 0 &&
         maximumInputElements >= minimumInputElements &&
         minimumOutputColumns > 0 &&
         maximumOutputColumns >= minimumOutputColumns &&
         minimumReductionStages > 0 && finiteValues({interceptCycles}) &&
         finiteNonNegative({inputElementCycles, log2OutputColumnCycles}) &&
         !evidence.empty();
}

bool StageRouteCompletionProfile::LargeSequenceMinmaxScan::isValid() const {
  if (!enabled)
    return true;
  return minimumIndependentSequences > 1 &&
         maximumIndependentSequences >= minimumIndependentSequences &&
         minimumExtent > 1 && maximumExtent >= minimumExtent &&
         finiteNonNegative({perIndependentSequenceSystemCycles}) &&
         !evidence.empty();
}

bool StageRouteCompletionProfile::MultiSequenceMinmaxScan::isValid() const {
  if (!enabled)
    return true;
  const bool base =
      minimumIndependentSequences > 1 &&
      maximumIndependentSequences >= minimumIndependentSequences &&
      minimumExtent > 1 && maximumExtent >= minimumExtent &&
      finiteNonNegative({baseReductionSystemCycles,
                         perIndependentSequenceReductionSystemCycles,
                         maximumReductionSystemCycles}) &&
      maximumReductionSystemCycles >= baseReductionSystemCycles;
  const bool hasLarge =
      largeMinimumIndependentSequences != 0 ||
      largeMaximumIndependentSequences != 0 || largeMinimumExtent != 0 ||
      largeMaximumExtent != 0 || largeBaseReductionSystemCycles != 0.0 ||
      largePerIndependentSequenceReductionSystemCycles != 0.0 ||
      largeMaximumReductionSystemCycles != 0.0;
  const bool large =
      !hasLarge ||
      (largeMinimumIndependentSequences > maximumIndependentSequences &&
       largeMaximumIndependentSequences >= largeMinimumIndependentSequences &&
       largeMinimumExtent > 1 && largeMaximumExtent >= largeMinimumExtent &&
       finiteNonNegative({largeBaseReductionSystemCycles,
                          largePerIndependentSequenceReductionSystemCycles,
                          largeMaximumReductionSystemCycles}) &&
       largeMaximumReductionSystemCycles >= largeBaseReductionSystemCycles);
  return base && large && !evidence.empty();
}

bool StageRouteCompletionProfile::StraightLineAddReduction::isValid() const {
  if (!enabled)
    return true;
  return minimumRows > 0 && maximumRows >= minimumRows && minimumExtent > 1 &&
         maximumExtent >= minimumExtent &&
         finiteNonNegative({sharedCompletionSystemCycles}) && !evidence.empty();
}

bool StageRouteCompletionProfile::TwoLoopSingleReduction::isValid() const {
  if (!enabled)
    return true;
  return minimumSourcePrograms > 0 &&
         maximumSourcePrograms >= minimumSourcePrograms &&
         minimumRuntimeIterations > 0 &&
         maximumRuntimeIterations >= minimumRuntimeIterations &&
         requiredReductionExtent > 1 &&
         finiteNonNegative(
             {firstIterationSystemCycles, additionalIterationSystemCycles}) &&
         !evidence.empty();
}

bool StageRouteCompletionProfile::isValid() const {
  return multiResultPrefixScan.isValid() &&
         pairwiseSelectionNetwork.isValid() &&
         largeSequenceMinmaxScan.isValid() &&
         multiSequenceMinmaxScan.isValid() &&
         straightLineAddReduction.isValid() && twoLoopSingleReduction.isValid();
}

namespace {

constexpr llvm::StringLiteral kAllSimd = "all_simd";
constexpr llvm::StringLiteral kAllSimtOnly = "all_simt_only";
constexpr llvm::StringLiteral kMixedSimdSimt = "mixed_simd_simt";

static bool isKnownRoute(llvm::StringRef route) {
  return route == kAllSimd || route == kAllSimtOnly || route == kMixedSimdSimt;
}

static bool validNonNegativeCurve(const std::map<int64_t, double> &curve) {
  return llvm::all_of(curve, [](const auto &entry) {
    return entry.first > 0 && finiteNonNegative({entry.second});
  });
}

struct StructuralProfile {
  int64_t tinyDotFlopsMax = 0;
};

/// Route-independent scalar work outside the Stage graph.
struct KernelTimingProfile {
  struct RuntimeArgumentSetup {
    int64_t freeArgumentCount = 0;
    double firstExcessSystemCycles = 0.0;
    double incrementalExcessSystemCycles = 0.0;

    bool isValid() const {
      return freeArgumentCount >= 0 &&
             finiteNonNegative(
                 {firstExcessSystemCycles, incrementalExcessSystemCycles});
    }

    double cost(int64_t argumentCount) const {
      const int64_t excess = argumentCount - freeArgumentCount;
      if (excess <= 0)
        return 0.0;
      return firstExcessSystemCycles +
             static_cast<double>(excess - 1) * incrementalExcessSystemCycles;
    }
  };

  struct StageDependencyCompletion {
    bool enabled = false;
    std::string target;
    std::string loweringContract;
    double taskEnvelopeSystemCycles = 0.0;
    double dynamicCVFirstFillSystemCycles = 0.0;
    double dynamicCVPerIterationDeltaSystemCycles = 0.0;
    std::string scheduleContract;
    std::string fullKernelValidationSHA256;
    std::string evidence;

    bool isValid() const {
      return !enabled ||
             (target == "Ascend950PR_9579" && !loweringContract.empty() &&
              finitePositive(
                  {taskEnvelopeSystemCycles, dynamicCVFirstFillSystemCycles}) &&
              finiteValues({dynamicCVPerIterationDeltaSystemCycles}) &&
              !scheduleContract.empty() &&
              fullKernelValidationSHA256.size() == 64 && !evidence.empty());
    }
  };

  struct TinyStraightLineSimtMemoryCompletion {
    bool enabled = false;
    int64_t maximumElementsPerAccess = 0;
    double signedResidualSystemCycles = 0.0;
    std::string evidence;

    bool isValid() const {
      if (!enabled)
        return true;
      return maximumElementsPerAccess > 0 &&
             finiteValues({signedResidualSystemCycles}) && !evidence.empty();
    }
  };

  struct SimtNonAffineRank1MemoryCompletion {
    bool enabled = false;
    int64_t maximumElementsPerAccess = 0;
    std::map<int64_t, double> residualBySourceProgramCount;
    std::string evidence;

    bool isValid() const {
      if (!enabled)
        return true;
      return maximumElementsPerAccess > 0 &&
             !residualBySourceProgramCount.empty() && !evidence.empty() &&
             llvm::all_of(residualBySourceProgramCount, [](const auto &entry) {
               return entry.first > 0 && std::isfinite(entry.second);
             });
    }
  };

  struct Rank2StridedDirectLoadCompletion {
    int64_t minimumElements = 0;
    int64_t maximumElements = 0;
    double startupCycles = 0.0;
    double perElementCycles = 0.0;
    double maximumCycles = 0.0;
    std::string evidence;

    bool isPresent() const {
      return minimumElements != 0 || maximumElements != 0 ||
             startupCycles != 0.0 || perElementCycles != 0.0 ||
             maximumCycles != 0.0 || !evidence.empty();
    }

    bool isValid() const {
      return !isPresent() ||
             (minimumElements > 0 && maximumElements >= minimumElements &&
              finiteNonNegative({startupCycles, perElementCycles}) &&
              finitePositive({maximumCycles}) && !evidence.empty());
    }

    double cycles(int64_t elements) const {
      if (elements < minimumElements || elements > maximumElements)
        return 0.0;
      return std::min(maximumCycles,
                      startupCycles + perElementCycles * elements);
    }
  };

  struct Rank2InterleavedPrefixScanCompletion {
    bool enabled = false;
    int64_t minimumSourcePrograms = 0;
    int64_t maximumSourcePrograms = 0;
    int64_t minimumExtent = 0;
    int64_t maximumExtent = 0;
    int64_t minimumIndependentSequences = 0;
    int64_t maximumIndependentSequences = 0;
    double startupSystemCycles = 0.0;
    double perElementSystemCycles = 0.0;
    double factor2StartupSystemCycles = 0.0;
    double factor2PerElementSystemCycles = 0.0;
    int64_t factor2ReferencePhysicalProgramCount = 0;
    double factor2PerActivePhysicalProgramSystemCycles = 0.0;
    double factor2MinimumTargetSystemCycles = 0.0;
    double factor4StartupSystemCycles = 0.0;
    double factor4PerElementSystemCycles = 0.0;
    std::string evidence;

    bool isValid() const {
      if (!enabled)
        return true;
      return minimumSourcePrograms > 0 &&
             maximumSourcePrograms >= minimumSourcePrograms &&
             minimumExtent > 1 && maximumExtent >= minimumExtent &&
             minimumIndependentSequences > 1 &&
             maximumIndependentSequences >= minimumIndependentSequences &&
             factor2ReferencePhysicalProgramCount > 0 &&
             finiteNonNegative({startupSystemCycles, factor2StartupSystemCycles,
                                factor2PerActivePhysicalProgramSystemCycles,
                                factor4StartupSystemCycles}) &&
             finitePositive({perElementSystemCycles,
                             factor2PerElementSystemCycles,
                             factor2MinimumTargetSystemCycles,
                             factor4PerElementSystemCycles}) &&
             !evidence.empty();
    }
  };

  struct SimtStaticLoopDirectMemoryCompletion {
    bool enabled = false;
    int64_t minimumStaticIterations = 0;
    int64_t maximumStaticIterations = 0;
    int64_t requiredContiguousElements = 0;
    int64_t requiredElementBits = 0;
    int64_t minimumLoadCount = 0;
    int64_t maximumLoadCount = 0;
    int64_t requiredStoreCount = 0;
    int64_t maximumSuperblockFactor = 0;
    double startupSystemCycles = 0.0;
    std::map<int64_t, double> perIterationSystemCyclesByLoadCount;
    std::string evidence;

    bool isValid() const {
      if (!enabled)
        return true;
      if (minimumStaticIterations <= 0 ||
          maximumStaticIterations < minimumStaticIterations ||
          requiredContiguousElements <= 0 || requiredElementBits <= 0 ||
          minimumLoadCount <= 0 || maximumLoadCount < minimumLoadCount ||
          requiredStoreCount != 1 || maximumSuperblockFactor <= 0 ||
          !finiteNonNegative({startupSystemCycles}) || evidence.empty())
        return false;
      for (int64_t loads = minimumLoadCount; loads <= maximumLoadCount;
           ++loads) {
        const auto it = perIterationSystemCyclesByLoadCount.find(loads);
        if (it == perIterationSystemCyclesByLoadCount.end() ||
            !std::isfinite(it->second) || it->second <= 0.0)
          return false;
      }
      return true;
    }

    double targetSystemCycles(int64_t staticIterations,
                              int64_t loadCount) const {
      const auto it = perIterationSystemCyclesByLoadCount.find(loadCount);
      if (it == perIterationSystemCyclesByLoadCount.end())
        return 0.0;
      return startupSystemCycles +
             static_cast<double>(staticIterations) * it->second;
    }
  };

  double scalarPrologueSystemCycles = 0.0;
  double scalarEpilogueSystemCycles = 0.0;
  std::map<int64_t, double> boundaryResidualSystemCyclesByPhysicalProgramCount;
  std::map<std::string, std::map<int64_t, std::map<int64_t, double>>>
      boundaryResidualSystemCyclesByRouteFactorAndPhysicalProgramCount;
  /// A minimal observable boundary kernel contains a final global store but no
  /// global load.
  std::map<std::string, double> globalLoadBoundaryResidualSystemCyclesByRoute;
  std::map<std::string, RuntimeArgumentSetup> runtimeArgumentSetupByRoute;
  int64_t scalarizedRank2IndirectLoadMaximumElements = 0;
  std::map<int64_t, double>
      scalarizedRank2IndirectLoadResidualByPhysicalProgramCount;
  std::string scalarizedRank2IndirectLoadEvidence;
  Rank2StridedDirectLoadCompletion simdRank2StridedDirectLoad;
  Rank2StridedDirectLoadCompletion simtRank2StridedDirectLoad;
  /// Signed completion adjustment for the whole-SIMT lowering of a tiny rank-2
  /// loaded-index tile.
  int64_t simtRank2IndirectLoadMaximumElements = 0;
  std::map<int64_t, std::map<int64_t, double>>
      simtRank2IndirectLoadResidualByElementBitsAndElements;
  std::string simtRank2IndirectLoadEvidence;
  int64_t subwarpIndirectCompositeMaximumElements = 0;
  std::map<int64_t, double> subwarpIndirectCompositeResidualByElements;
  std::string subwarpIndirectCompositeEvidence;
  StageRouteCompletionProfile routeCompletions;
  TinyStraightLineSimtMemoryCompletion tinyStraightLineSimtMemoryCompletion;
  SimtNonAffineRank1MemoryCompletion simtNonAffineRank1MemoryCompletion;
  Rank2InterleavedPrefixScanCompletion rank2InterleavedPrefixScanCompletion;
  SimtStaticLoopDirectMemoryCompletion simtStaticLoopDirectMemoryCompletion;
  StageDependencyCompletion stageDependencyCompletion;
  double systemCounterFrequencyMHz = 0.0;
  std::string evidence;

  bool isValid() const {
    const bool validCurve = validNonNegativeCurve(
        boundaryResidualSystemCyclesByPhysicalProgramCount);
    const bool validRouteCurves = llvm::all_of(
        boundaryResidualSystemCyclesByRouteFactorAndPhysicalProgramCount,
        [](const auto &routeEntry) {
          if (!isKnownRoute(routeEntry.first))
            return false;
          return llvm::all_of(routeEntry.second, [](const auto &factorEntry) {
            return factorEntry.first > 0 && !factorEntry.second.empty() &&
                   validNonNegativeCurve(factorEntry.second);
          });
        });
    const bool validGlobalLoadResiduals = llvm::all_of(
        globalLoadBoundaryResidualSystemCyclesByRoute, [](const auto &entry) {
          const llvm::StringRef route(entry.first);
          return isKnownRoute(route) && finiteNonNegative({entry.second});
        });
    const bool validArgumentSetup =
        llvm::all_of(runtimeArgumentSetupByRoute, [](const auto &entry) {
          const llvm::StringRef route(entry.first);
          return isKnownRoute(route) && entry.second.isValid();
        });
    const bool hasScalarizedRank2Residual =
        scalarizedRank2IndirectLoadMaximumElements != 0 ||
        !scalarizedRank2IndirectLoadResidualByPhysicalProgramCount.empty() ||
        !scalarizedRank2IndirectLoadEvidence.empty();
    const bool validScalarizedRank2Residual =
        !hasScalarizedRank2Residual ||
        (scalarizedRank2IndirectLoadMaximumElements > 0 &&
         !scalarizedRank2IndirectLoadResidualByPhysicalProgramCount.empty() &&
         !scalarizedRank2IndirectLoadEvidence.empty() &&
         validNonNegativeCurve(
             scalarizedRank2IndirectLoadResidualByPhysicalProgramCount));
    const bool hasSubwarpIndirectComposite =
        subwarpIndirectCompositeMaximumElements != 0 ||
        !subwarpIndirectCompositeResidualByElements.empty() ||
        !subwarpIndirectCompositeEvidence.empty();
    const bool validSubwarpIndirectComposite =
        !hasSubwarpIndirectComposite ||
        (subwarpIndirectCompositeMaximumElements > 0 &&
         !subwarpIndirectCompositeResidualByElements.empty() &&
         !subwarpIndirectCompositeEvidence.empty() &&
         validNonNegativeCurve(subwarpIndirectCompositeResidualByElements));
    const bool hasSimtRank2Residual =
        simtRank2IndirectLoadMaximumElements != 0 ||
        !simtRank2IndirectLoadResidualByElementBitsAndElements.empty() ||
        !simtRank2IndirectLoadEvidence.empty();
    const bool validSimtRank2Residual =
        !hasSimtRank2Residual ||
        (simtRank2IndirectLoadMaximumElements > 0 &&
         !simtRank2IndirectLoadResidualByElementBitsAndElements.empty() &&
         !simtRank2IndirectLoadEvidence.empty() &&
         llvm::all_of(
             simtRank2IndirectLoadResidualByElementBitsAndElements,
             [](const auto &bitsEntry) {
               return bitsEntry.first > 0 && !bitsEntry.second.empty() &&
                      llvm::all_of(bitsEntry.second, [](const auto &entry) {
                        return entry.first > 0 && std::isfinite(entry.second);
                      });
             }));
    return validCurve && validRouteCurves && validGlobalLoadResiduals &&
           validArgumentSetup && validScalarizedRank2Residual &&
           simdRank2StridedDirectLoad.isValid() &&
           simtRank2StridedDirectLoad.isValid() &&
           validSubwarpIndirectComposite && validSimtRank2Residual &&
           routeCompletions.isValid() &&
           tinyStraightLineSimtMemoryCompletion.isValid() &&
           simtNonAffineRank1MemoryCompletion.isValid() &&
           rank2InterleavedPrefixScanCompletion.isValid() &&
           simtStaticLoopDirectMemoryCompletion.isValid() &&
           stageDependencyCompletion.isValid() &&
           finiteNonNegative(
               {scalarPrologueSystemCycles, scalarEpilogueSystemCycles}) &&
           finitePositive({systemCounterFrequencyMHz}) && !evidence.empty();
  }

  static double interpolateBoundaryCurve(const std::map<int64_t, double> &curve,
                                         int64_t physicalProgramCount,
                                         double fallback) {
    if (physicalProgramCount <= 0 || curve.empty())
      return fallback;
    auto upper = curve.lower_bound(physicalProgramCount);
    if (upper == curve.begin())
      return upper->second;
    if (upper == curve.end())
      return std::prev(upper)->second;
    if (upper->first == physicalProgramCount)
      return upper->second;
    const auto lower = std::prev(upper);
    const double fraction =
        static_cast<double>(physicalProgramCount - lower->first) /
        static_cast<double>(upper->first - lower->first);
    return lower->second + fraction * (upper->second - lower->second);
  }

  double boundarySystemCycles(llvm::StringRef route, int64_t factor,
                              int64_t physicalProgramCount) const {
    const double fallback =
        scalarPrologueSystemCycles + scalarEpilogueSystemCycles;
    const auto routeIt =
        boundaryResidualSystemCyclesByRouteFactorAndPhysicalProgramCount.find(
            route.str());
    if (routeIt !=
        boundaryResidualSystemCyclesByRouteFactorAndPhysicalProgramCount
            .end()) {
      const auto factorIt = routeIt->second.find(factor);
      if (factorIt != routeIt->second.end())
        return interpolateBoundaryCurve(factorIt->second, physicalProgramCount,
                                        fallback);
    }
    return interpolateBoundaryCurve(
        boundaryResidualSystemCyclesByPhysicalProgramCount,
        physicalProgramCount, fallback);
  }

  double globalLoadBoundarySystemCycles(llvm::StringRef route) const {
    const auto found =
        globalLoadBoundaryResidualSystemCyclesByRoute.find(route.str());
    return found == globalLoadBoundaryResidualSystemCyclesByRoute.end()
               ? 0.0
               : found->second;
  }

  double runtimeArgumentSetupSystemCycles(llvm::StringRef route,
                                          int64_t argumentCount) const {
    const auto found = runtimeArgumentSetupByRoute.find(route.str());
    return found == runtimeArgumentSetupByRoute.end()
               ? 0.0
               : found->second.cost(argumentCount);
  }

  double scalarizedRank2IndirectLoadResidualSystemCycles(
      int64_t physicalProgramCount) const {
    return interpolateBoundaryCurve(
        scalarizedRank2IndirectLoadResidualByPhysicalProgramCount,
        physicalProgramCount, 0.0);
  }

  double subwarpIndirectCompositeResidualSystemCycles(int64_t elements) const {
    if (subwarpIndirectCompositeResidualByElements.empty() ||
        elements < subwarpIndirectCompositeResidualByElements.begin()->first ||
        elements > subwarpIndirectCompositeResidualByElements.rbegin()->first)
      return 0.0;
    return interpolateBoundaryCurve(subwarpIndirectCompositeResidualByElements,
                                    elements, 0.0);
  }

  double simtRank2IndirectLoadResidualSystemCycles(int64_t elementBits,
                                                   int64_t elements) const {
    const auto found =
        simtRank2IndirectLoadResidualByElementBitsAndElements.find(elementBits);
    if (found == simtRank2IndirectLoadResidualByElementBitsAndElements.end() ||
        found->second.empty() || elements < found->second.begin()->first ||
        elements > found->second.rbegin()->first)
      return 0.0;
    return interpolateBoundaryCurve(found->second, elements, 0.0);
  }
};

struct CandidateProfile {
  HardwareProfile hardware;
  std::vector<std::string> compatibleTargets;
  std::string scoreUnit;
  std::string contentSha256;
  std::string selectionContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkContentSha256;
  StructuralProfile structural;
  KernelTimingProfile kernelTiming;
};

static bool wildcardMatchInsensitive(llvm::StringRef pattern,
                                     llvm::StringRef value) {
  const std::string loweredPattern = pattern.lower();
  const std::string loweredValue = value.lower();
  size_t patternIndex = 0;
  size_t valueIndex = 0;
  size_t starIndex = std::string::npos;
  size_t starValueIndex = 0;
  while (valueIndex < loweredValue.size()) {
    if (patternIndex < loweredPattern.size() &&
        (loweredPattern[patternIndex] == '?' ||
         loweredPattern[patternIndex] == loweredValue[valueIndex])) {
      ++patternIndex;
      ++valueIndex;
      continue;
    }
    if (patternIndex < loweredPattern.size() &&
        loweredPattern[patternIndex] == '*') {
      starIndex = patternIndex++;
      starValueIndex = valueIndex;
      continue;
    }
    if (starIndex != std::string::npos) {
      patternIndex = starIndex + 1;
      valueIndex = ++starValueIndex;
      continue;
    }
    return false;
  }
  while (patternIndex < loweredPattern.size() &&
         loweredPattern[patternIndex] == '*')
    ++patternIndex;
  return patternIndex == loweredPattern.size();
}

/// Small fail-fast facade around llvm::json.  It permits a readable profile
/// parser while retaining a single actionable error message.
class ProfileJSONReader {
public:
  const llvm::json::Object *object(const llvm::json::Object &parent,
                                   llvm::StringRef key,
                                   llvm::StringRef context) {
    if (failed())
      return nullptr;
    if (const auto *value = parent.getObject(key))
      return value;
    setError(context + "." + key + " must be an object");
    return nullptr;
  }

  double number(const llvm::json::Object &parent, llvm::StringRef key,
                llvm::StringRef context) {
    if (failed())
      return 0.0;
    if (auto value = parent.getNumber(key))
      return *value;
    setError(context + "." + key + " must be a number");
    return 0.0;
  }

  int64_t integer(const llvm::json::Object &parent, llvm::StringRef key,
                  llvm::StringRef context) {
    if (failed())
      return 0;
    if (auto value = parent.getInteger(key))
      return *value;
    setError(context + "." + key + " must be an integer");
    return 0;
  }

  std::string string(const llvm::json::Object &parent, llvm::StringRef key,
                     llvm::StringRef context) {
    if (failed())
      return {};
    if (auto value = parent.getString(key))
      return value->str();
    setError(context + "." + key + " must be a string");
    return {};
  }

  double optionalNumber(const llvm::json::Object &parent, llvm::StringRef key,
                        double defaultValue) {
    if (auto value = parent.getNumber(key))
      return *value;
    return defaultValue;
  }

  bool failed() const { return !error.empty(); }
  llvm::StringRef getError() const { return error; }

  void setError(const llvm::Twine &message) {
    if (error.empty())
      error = message.str();
  }

private:
  std::string error;
};

using NumberDestination = std::pair<llvm::StringRef, double *>;
using IntegerDestination = std::pair<llvm::StringRef, int64_t *>;
using IntegerCurve = std::map<int64_t, double>;
using FactorCurves = std::map<int64_t, IntegerCurve>;
using FactorSurfaces = std::map<int64_t, std::map<int64_t, IntegerCurve>>;
template <typename Curves>
using CurveDestination = std::pair<llvm::StringRef, Curves *>;

static void readNumbers(ProfileJSONReader &reader,
                        const llvm::json::Object &object,
                        llvm::StringRef context,
                        std::initializer_list<NumberDestination> fields) {
  for (const auto &[key, destination] : fields)
    *destination = reader.number(object, key, context);
}

static void
readOptionalNumbers(ProfileJSONReader &reader, const llvm::json::Object &object,
                    std::initializer_list<NumberDestination> fields) {
  for (const auto &[key, destination] : fields)
    *destination = reader.optionalNumber(object, key, *destination);
}

static void readIntegers(ProfileJSONReader &reader,
                         const llvm::json::Object &object,
                         llvm::StringRef context,
                         std::initializer_list<IntegerDestination> fields) {
  for (const auto &[key, destination] : fields)
    *destination = reader.integer(object, key, context);
}

static void
readOptionalIntegers(const llvm::json::Object &object,
                     std::initializer_list<IntegerDestination> fields) {
  for (const auto &[key, destination] : fields)
    *destination = object.getInteger(key).value_or(*destination);
}

static double resolveNumberOrMeasurement(
    const llvm::json::Object &object, llvm::StringRef numberKey,
    llvm::StringRef measurementKey, llvm::StringRef expectedUnit,
    const MicrobenchmarkProfile *microbench, ProfileJSONReader &reader,
    llvm::StringRef context) {
  if (auto reference = object.getString(measurementKey)) {
    if (!microbench) {
      reader.setError(context + "." + measurementKey +
                      " requires microbenchmark_profile");
      return 0.0;
    }
    llvm::StringRef expectedCycleDomain = "none";
    if (expectedUnit == "system_cycle" ||
        expectedUnit.ends_with("/system_cycle"))
      expectedCycleDomain = "SYS_CNT";
    auto value =
        microbench->requireValue(*reference, expectedUnit, expectedCycleDomain);
    if (!value) {
      reader.setError(llvm::toString(value.takeError()));
      return 0.0;
    }
    return *value;
  }
  return reader.number(object, numberKey, context);
}

static void readPositiveIntegerMeasurementCurve(
    const llvm::json::Object &owner, llvm::StringRef curveKey,
    llvm::StringRef expectedUnit, const MicrobenchmarkProfile *microbench,
    ProfileJSONReader &reader, llvm::StringRef context,
    std::map<int64_t, double> &curve) {
  const auto *object = owner.getObject(curveKey);
  if (!object)
    return;
  if (!microbench) {
    reader.setError(context + "." + curveKey +
                    " requires microbenchmark_profile");
    return;
  }
  llvm::StringRef expectedCycleDomain = "none";
  if (expectedUnit == "system_cycle" || expectedUnit.ends_with("/system_cycle"))
    expectedCycleDomain = "SYS_CNT";
  for (const auto &entry : *object) {
    int64_t point = 0;
    if (llvm::StringRef(entry.first).getAsInteger(10, point) || point <= 0) {
      reader.setError(context + "." + curveKey +
                      " keys must be positive integers");
      return;
    }
    const auto measurement = entry.second.getAsString();
    if (!measurement) {
      reader.setError(context + "." + curveKey +
                      " values must be measurement names");
      return;
    }
    auto value = microbench->requireValue(*measurement, expectedUnit,
                                          expectedCycleDomain);
    if (!value) {
      reader.setError(llvm::toString(value.takeError()));
      return;
    }
    curve[point] = *value;
  }
}

static void readPositiveIntegerNumberCurve(const llvm::json::Object &owner,
                                           llvm::StringRef curveKey,
                                           ProfileJSONReader &reader,
                                           llvm::StringRef context,
                                           std::map<int64_t, double> &curve) {
  const auto *object = owner.getObject(curveKey);
  if (!object)
    return;
  for (const auto &entry : *object) {
    int64_t point = 0;
    const auto value = entry.second.getAsNumber();
    if (llvm::StringRef(entry.first).getAsInteger(10, point) || point <= 0 ||
        !value || !std::isfinite(*value) || *value <= 0.0) {
      reader.setError(context + "." + curveKey +
                      " must map positive integer keys to positive numbers");
      return;
    }
    curve[point] = *value;
  }
}

static void readSignedPositiveIntegerNumberCurve(
    const llvm::json::Object &owner, llvm::StringRef curveKey,
    ProfileJSONReader &reader, llvm::StringRef context,
    std::map<int64_t, double> &curve) {
  const auto *object = owner.getObject(curveKey);
  if (!object)
    return;
  for (const auto &entry : *object) {
    int64_t point = 0;
    const auto value = entry.second.getAsNumber();
    if (llvm::StringRef(entry.first).getAsInteger(10, point) || point <= 0 ||
        !value || !std::isfinite(*value)) {
      reader.setError(context + "." + curveKey +
                      " must map positive integer keys to finite numbers");
      return;
    }
    curve[point] = *value;
  }
}

static void readSignedPositiveIntegerNumberCurves(
    const llvm::json::Object &owner, llvm::StringRef curvesKey,
    ProfileJSONReader &reader, llvm::StringRef context,
    std::map<int64_t, std::map<int64_t, double>> &curves) {
  const auto *outer = owner.getObject(curvesKey);
  if (!outer)
    return;
  for (const auto &outerEntry : *outer) {
    int64_t outerPoint = 0;
    const llvm::StringRef outerKey(outerEntry.first);
    if (outerKey.getAsInteger(10, outerPoint) || outerPoint <= 0) {
      reader.setError(context + "." + curvesKey +
                      " keys must be positive integers");
      return;
    }
    const auto *curveObject = outerEntry.second.getAsObject();
    if (!curveObject) {
      reader.setError(context + "." + curvesKey + "." + outerKey +
                      " must be an object");
      return;
    }
    auto &curve = curves[outerPoint];
    for (const auto &entry : *curveObject) {
      int64_t point = 0;
      const auto value = entry.second.getAsNumber();
      if (llvm::StringRef(entry.first).getAsInteger(10, point) || point <= 0 ||
          !value || !std::isfinite(*value)) {
        reader.setError(context + "." + curvesKey + "." + outerKey +
                        " must map positive integer keys to finite numbers");
        return;
      }
      curve[point] = *value;
    }
  }
}

static void readRouteFactorNumberCurves(
    const llvm::json::Object &owner, llvm::StringRef curveKey,
    ProfileJSONReader &reader, llvm::StringRef context,
    std::map<std::string, std::map<int64_t, std::map<int64_t, double>>>
        &curves) {
  const auto *routes = owner.getObject(curveKey);
  if (!routes)
    return;
  for (const auto &routeEntry : *routes) {
    const llvm::StringRef route(routeEntry.first);
    if (!isKnownRoute(route)) {
      reader.setError(context + "." + curveKey +
                      " contains an unknown route key '" + route + "'");
      return;
    }
    const auto *factors = routeEntry.second.getAsObject();
    if (!factors) {
      reader.setError(context + "." + curveKey + "." + route +
                      " must be an object");
      return;
    }
    for (const auto &factorEntry : *factors) {
      int64_t factor = 0;
      const llvm::StringRef factorKey(factorEntry.first);
      if (factorKey.getAsInteger(10, factor) || factor <= 0) {
        reader.setError(context + "." + curveKey + "." + route +
                        " factor keys must be positive integers");
        return;
      }
      const auto *points = factorEntry.second.getAsObject();
      if (!points) {
        reader.setError(context + "." + curveKey + "." + route + "." +
                        factorKey + " must be an object");
        return;
      }
      auto &curve = curves[route.str()][factor];
      for (const auto &pointEntry : *points) {
        int64_t programs = 0;
        const auto value = pointEntry.second.getAsNumber();
        if (llvm::StringRef(pointEntry.first).getAsInteger(10, programs) ||
            programs <= 0 || !value || !std::isfinite(*value) || *value < 0.0) {
          reader.setError(context + "." + curveKey + "." + route + "." +
                          factorKey +
                          " must map positive integer keys to "
                          "non-negative numbers");
          return;
        }
        curve[programs] = *value;
      }
    }
  }
}

static void readRouteNumberMap(const llvm::json::Object &owner,
                               llvm::StringRef mapKey,
                               ProfileJSONReader &reader,
                               llvm::StringRef context,
                               std::map<std::string, double> &values) {
  const auto *object = owner.getObject(mapKey);
  if (!object)
    return;
  for (const auto &entry : *object) {
    const llvm::StringRef route(entry.first);
    const auto value = entry.second.getAsNumber();
    if (!isKnownRoute(route) || !value || !std::isfinite(*value) ||
        *value < 0.0) {
      reader.setError(context + "." + mapKey +
                      " must map known route keys to non-negative numbers");
      return;
    }
    values[route.str()] = *value;
  }
}

static void readRuntimeArgumentSetupByRoute(
    const llvm::json::Object &owner, ProfileJSONReader &reader,
    llvm::StringRef context,
    std::map<std::string, KernelTimingProfile::RuntimeArgumentSetup> &values) {
  constexpr llvm::StringLiteral key = "runtime_argument_setup_by_route";
  const auto *routes = owner.getObject(key);
  if (!routes)
    return;
  const std::string fieldContext = (context + "." + key).str();
  for (const auto &entry : *routes) {
    const llvm::StringRef route(entry.first);
    const auto *object = entry.second.getAsObject();
    if (!isKnownRoute(route) || !object) {
      reader.setError(context + "." + key +
                      " must contain known route objects");
      return;
    }
    KernelTimingProfile::RuntimeArgumentSetup setup;
    setup.freeArgumentCount =
        reader.integer(*object, "free_argument_count", fieldContext);
    setup.firstExcessSystemCycles =
        reader.number(*object, "first_excess_system_cycles", fieldContext);
    setup.incrementalExcessSystemCycles = reader.number(
        *object, "incremental_excess_system_cycles", fieldContext);
    if (!setup.isValid()) {
      reader.setError(context + "." + key + "." + route +
                      " contains invalid argument-setup coefficients");
      return;
    }
    values[route.str()] = setup;
  }
}

static void readFactorMeasurementCurves(
    const llvm::json::Object &owner, llvm::StringRef curveKey,
    llvm::StringRef expectedUnit, const MicrobenchmarkProfile *microbench,
    ProfileJSONReader &reader, llvm::StringRef context,
    std::map<int64_t, std::map<int64_t, double>> &curves) {
  const auto *factors = owner.getObject(curveKey);
  if (!factors)
    return;
  if (!microbench) {
    reader.setError(context + "." + curveKey +
                    " requires microbenchmark_profile");
    return;
  }
  llvm::StringRef expectedCycleDomain = "none";
  if (expectedUnit == "system_cycle" || expectedUnit.ends_with("/system_cycle"))
    expectedCycleDomain = "SYS_CNT";
  for (const auto &factorEntry : *factors) {
    const llvm::StringRef factorKey(factorEntry.first);
    int64_t factor = 0;
    if (factorKey.getAsInteger(10, factor) || factor <= 0) {
      reader.setError(context + "." + curveKey +
                      " factor keys must be positive integers");
      return;
    }
    const std::string factorContext =
        (context + "." + curveKey + "." + factorKey).str();
    const auto *points = factorEntry.second.getAsObject();
    if (!points) {
      reader.setError(factorContext + " must be an object");
      return;
    }
    auto &curve = curves[factor];
    for (const auto &pointEntry : *points) {
      int64_t laneSteps = 0;
      if (llvm::StringRef(pointEntry.first).getAsInteger(10, laneSteps) ||
          laneSteps <= 0) {
        reader.setError(factorContext + " keys must be positive integers");
        return;
      }
      const auto measurement = pointEntry.second.getAsString();
      if (!measurement) {
        reader.setError(factorContext + " values must be measurement names");
        return;
      }
      auto value = microbench->requireValue(*measurement, expectedUnit,
                                            expectedCycleDomain);
      if (!value) {
        reader.setError(llvm::toString(value.takeError()));
        return;
      }
      curve[laneSteps] = *value;
    }
  }
}

static void readFactorRowExtentMeasurementSurfaces(
    const llvm::json::Object &owner, llvm::StringRef surfaceKey,
    const MicrobenchmarkProfile *microbench, ProfileJSONReader &reader,
    llvm::StringRef context,
    std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>> &surfaces) {
  const auto *factors = owner.getObject(surfaceKey);
  if (!factors)
    return;
  if (!microbench) {
    reader.setError(context + "." + surfaceKey +
                    " requires microbenchmark_profile");
    return;
  }
  for (const auto &factorEntry : *factors) {
    int64_t factor = 0;
    const llvm::StringRef factorKey(factorEntry.first);
    if (factorKey.getAsInteger(10, factor) || factor <= 0) {
      reader.setError(context + "." + surfaceKey +
                      " factor keys must be positive integers");
      return;
    }
    const auto *rowsObject = factorEntry.second.getAsObject();
    const std::string factorContext =
        (context + "." + surfaceKey + "." + factorKey).str();
    if (!rowsObject) {
      reader.setError(factorContext + " must be an object");
      return;
    }
    for (const auto &rowEntry : *rowsObject) {
      int64_t rows = 0;
      const llvm::StringRef rowKey(rowEntry.first);
      if (rowKey.getAsInteger(10, rows) || rows <= 0) {
        reader.setError(factorContext + " row keys must be positive integers");
        return;
      }
      const auto *extentObject = rowEntry.second.getAsObject();
      const std::string rowContext = factorContext + "." + rowKey.str();
      if (!extentObject) {
        reader.setError(rowContext + " must be an object");
        return;
      }
      for (const auto &extentEntry : *extentObject) {
        int64_t extent = 0;
        const llvm::StringRef extentKey(extentEntry.first);
        if (extentKey.getAsInteger(10, extent) || extent <= 0) {
          reader.setError(rowContext +
                          " extent keys must be positive integers");
          return;
        }
        const auto measurement = extentEntry.second.getAsString();
        if (!measurement) {
          reader.setError(rowContext + " values must be measurement names");
          return;
        }
        auto value =
            microbench->requireValue(*measurement, "system_cycle", "SYS_CNT");
        if (!value) {
          reader.setError(llvm::toString(value.takeError()));
          return;
        }
        surfaces[factor][rows][extent] = *value;
      }
    }
  }
}

static void readPositiveIntegerMeasurementCurves(
    const llvm::json::Object &owner, llvm::StringRef expectedUnit,
    const MicrobenchmarkProfile *microbench, ProfileJSONReader &reader,
    llvm::StringRef context,
    std::initializer_list<CurveDestination<IntegerCurve>> destinations) {
  for (const auto &[key, destination] : destinations)
    readPositiveIntegerMeasurementCurve(owner, key, expectedUnit, microbench,
                                        reader, context, *destination);
}

static void readFactorMeasurementCurveFields(
    const llvm::json::Object &owner, llvm::StringRef expectedUnit,
    const MicrobenchmarkProfile *microbench, ProfileJSONReader &reader,
    llvm::StringRef context,
    std::initializer_list<CurveDestination<FactorCurves>> destinations) {
  for (const auto &[key, destination] : destinations)
    readFactorMeasurementCurves(owner, key, expectedUnit, microbench, reader,
                                context, *destination);
}

static void readFactorRowExtentMeasurementSurfaceFields(
    const llvm::json::Object &owner, const MicrobenchmarkProfile *microbench,
    ProfileJSONReader &reader, llvm::StringRef context,
    std::initializer_list<CurveDestination<FactorSurfaces>> destinations) {
  for (const auto &[key, destination] : destinations)
    readFactorRowExtentMeasurementSurfaces(owner, key, microbench, reader,
                                           context, *destination);
}

static StageOperationRate
resolveOpProfile(const llvm::json::Object &ops, llvm::StringRef opName,
                 llvm::StringRef throughputKey, llvm::StringRef expectedUnit,
                 const MicrobenchmarkProfile *microbench,
                 ProfileJSONReader &reader) {
  StageOperationRate result;
  const llvm::json::Value *raw = ops.get(opName);
  if (!raw) {
    reader.setError("missing operation profile " + opName);
    return result;
  }
  const auto *op = raw->getAsObject();
  if (!op) {
    reader.setError("operation profile " + opName + " must be an object");
    return result;
  }
  if (auto relative = op->getString("relative_to")) {
    StageOperationRate base = resolveOpProfile(
        ops, *relative, throughputKey, expectedUnit, microbench, reader);
    result.throughput = base.throughput;
    result.throughputByWarpCount = std::move(base.throughputByWarpCount);
    // A relative operation without a measured factor is not calibrated.  In
    // particular, an explicit JSON null must not silently become unit cost.
    result.factor = reader.number(*op, "factor", opName);
  } else {
    result.throughput =
        resolveNumberOrMeasurement(*op, throughputKey, "throughput_measurement",
                                   expectedUnit, microbench, reader, opName);
    readPositiveIntegerMeasurementCurve(*op, "throughput_measurements",
                                        expectedUnit, microbench, reader,
                                        opName, result.throughputByWarpCount);
    result.factor = reader.optionalNumber(*op, "factor", 1.0);
  }
  result.startupCycles =
      reader.optionalNumber(*op, "startup_system_cycles", 0.0);
  result.freeWorkUnits = reader.optionalNumber(*op, "free_work_units", 0.0);
  if (auto minimum = op->getInteger("minimum_startup_elements"))
    result.minimumStartupElements = *minimum;
  result.gridStrideKernelStartupCycles = reader.optionalNumber(
      *op, "grid_stride_kernel_startup_system_cycles", 0.0);
  result.includesLocalScopeHandoff =
      op->getBoolean("includes_local_scope_handoff").value_or(false);
  return result;
}

static void readMaskedStridedCopyCompletion(
    const llvm::json::Object &memory, llvm::StringRef field,
    llvm::StringRef context, ProfileJSONReader &reader,
    std::map<int64_t, MaskedStridedCopyCompletionCost> &result) {
  const auto *curves = memory.getObject(field);
  if (!curves)
    return;
  for (const auto &entry : *curves) {
    int64_t strideBytes = 0;
    const llvm::StringRef strideKey = entry.first;
    const std::string path =
        context.str() + "." + field.str() + "." + entry.first.str();
    const auto *curve = entry.second.getAsObject();
    if (strideKey.getAsInteger(10, strideBytes) || strideBytes <= 0 || !curve) {
      reader.setError(path + " requires a positive byte stride and object");
      continue;
    }
    MaskedStridedCopyCompletionCost cost;
    cost.baseSystemCycles = reader.number(*curve, "base_system_cycles", path);
    cost.additionalRowSystemCycles =
        reader.number(*curve, "additional_row_system_cycles", path);
    cost.minimumRows = reader.integer(*curve, "minimum_rows", path);
    cost.maximumRows = reader.integer(*curve, "maximum_rows", path);
    cost.minimumIterations = reader.integer(*curve, "minimum_iterations", path);
    cost.maximumIterations = reader.integer(*curve, "maximum_iterations", path);
    cost.evidence = reader.string(*curve, "evidence", path);
    if (!cost.isValid())
      reader.setError(path + " contains an invalid completion fit");
    else
      result[strideBytes] = std::move(cost);
  }
}

/// Match Python's json.dumps(value, sort_keys=True) representation.
static void emitPythonCanonicalJSON(const llvm::json::Value &value,
                                    llvm::raw_ostream &os) {
  if (const auto *object = value.getAsObject()) {
    std::vector<llvm::StringRef> keys;
    keys.reserve(object->size());
    for (const auto &entry : *object)
      keys.push_back(entry.first);
    llvm::sort(keys);
    os << '{';
    bool first = true;
    for (llvm::StringRef key : keys) {
      if (!first)
        os << ", ";
      first = false;
      os << llvm::json::Value(key.str()) << ": ";
      emitPythonCanonicalJSON(*object->get(key), os);
    }
    os << '}';
    return;
  }
  if (const auto *array = value.getAsArray()) {
    os << '[';
    bool first = true;
    for (const llvm::json::Value &element : *array) {
      if (!first)
        os << ", ";
      first = false;
      emitPythonCanonicalJSON(element, os);
    }
    os << ']';
    return;
  }
  os << value;
}

static std::string resolveProfileReference(llvm::StringRef ownerPath,
                                           llvm::StringRef reference) {
  if (llvm::sys::path::is_absolute(reference))
    return reference.str();
  llvm::SmallString<256> resolved(ownerPath);
  llvm::sys::path::remove_filename(resolved);
  llvm::sys::path::append(resolved, reference);
  llvm::sys::path::remove_dots(resolved, true);
  return resolved.str().str();
}

static void
readLayoutCostMap(ProfileJSONReader &reader,
                  const llvm::json::Object &resources, llvm::StringRef key,
                  llvm::StringRef context,
                  std::map<std::string, StageMemoryLayoutCost> &out) {
  const auto *object = resources.getObject(key);
  if (!object)
    return;
  for (const auto &entry : *object) {
    const auto *value = entry.second.getAsObject();
    const llvm::StringRef entryKey = entry.first;
    const std::string path = (context + "." + key + "." + entryKey).str();
    if (!value) {
      reader.setError(path + " must be an object");
      return;
    }
    StageMemoryLayoutCost cost;
    cost.firstCycles = reader.number(*value, "first_system_cycles", path);
    cost.incrementalCycles =
        reader.number(*value, "incremental_system_cycles", path);
    cost.minCount = reader.integer(*value, "min_count", path);
    cost.maxCount = reader.integer(*value, "max_count", path);
    cost.evidence = reader.string(*value, "evidence", path);
    cost.includesSharedLayout =
        value->getBoolean("includes_shared_layout").value_or(false);
    if (!cost.isValid())
      reader.setError(path + " requires finite nonnegative costs, a valid "
                             "count domain and evidence");
    out[entryKey.str()] = cost;
  }
}

static void readSourceFit(ProfileJSONReader &reader,
                          const llvm::json::Object &resources,
                          llvm::StringRef key, llvm::StringRef context,
                          SimtSourceFit &fit) {
  const auto *object = resources.getObject(key);
  if (!object)
    return;
  const std::string path = (context + "." + key).str();
  fit.enabled = object->getBoolean("enabled").value_or(false);
  fit.intercept = reader.optionalNumber(*object, "intercept", 0.0);
  fit.directInstructionWeight =
      reader.optionalNumber(*object, "direct_instruction_weight", 0.0);
  fit.segmentWeight = reader.optionalNumber(*object, "segment_weight", 0.0);
  fit.strideRatioWeight =
      reader.optionalNumber(*object, "stride_ratio_weight", 0.0);
  fit.maskedWeight = reader.optionalNumber(*object, "masked_weight", 0.0);
  fit.peakLiveRegisterWeight =
      reader.optionalNumber(*object, "peak_live_register_weight", 0.0);
  fit.unrollFactorWeight =
      reader.optionalNumber(*object, "unroll_factor_weight", 0.0);
  if (!fit.isValid())
    reader.setError(path + " contains invalid nonnegative fit coefficients");
}

static void readStageResources(ProfileJSONReader &reader,
                               const llvm::json::Object &mode,
                               llvm::StringRef context,
                               const MicrobenchmarkProfile *microbench,
                               StageModeProfile &profile) {
  const auto *resources = reader.object(mode, "stage_resources", context);
  if (!resources)
    return;
  const std::string prefix = (context + ".stage_resources").str();
  profile.issueOperationsPerCycle = resolveNumberOrMeasurement(
      *resources, "issue_instructions_per_system_cycle",
      "issue_instructions_per_system_cycle_measurement",
      "instruction/system_cycle", microbench, reader, prefix);
  profile.spillTransactionsPerCycle =
      reader.optionalNumber(*resources, "spill_transactions_per_system_cycle",
                            profile.spillTransactionsPerCycle);
  if (auto usableUbBytes = resources->getInteger("usable_ub_bytes")) {
    if (*usableUbBytes <= 0)
      reader.setError(prefix + ".usable_ub_bytes must be positive");
    else
      profile.usableUbBytes = *usableUbBytes;
  }
  if (auto budget = resources->getNumber("register_budget_per_thread")) {
    if (*budget <= 0.0 || !std::isfinite(*budget))
      reader.setError(prefix + ".register_budget_per_thread must be positive");
    else
      profile.registerBudgetPerThread = *budget;
  }
  if (auto coverage = resources->getNumber("spill_transaction_coverage")) {
    if (*coverage <= 0.0 || !std::isfinite(*coverage))
      reader.setError(prefix + ".spill_transaction_coverage must be positive");
    else
      profile.spillTransactionCoverage = *coverage;
  }
  if (const auto *scalar = reader.object(*resources, "scalar_memory", prefix)) {
    readOptionalNumbers(
        reader, *scalar,
        {{"main_load_prep_system_cycles", &profile.mainScalarLoadPrepCycles},
         {"main_load_fill_system_cycles", &profile.mainScalarLoadFillCycles},
         {"main_load_issue_system_cycles", &profile.mainScalarLoadIssueCycles},
         {"main_load_outstanding_line_count",
          &profile.mainScalarLoadOutstandingLines},
         {"main_load_extra_line_low_system_cycles",
          &profile.mainScalarLoadExtraLineLowCycles},
         {"main_load_extra_line_high_system_cycles",
          &profile.mainScalarLoadExtraLineHighCycles},
         {"main_load_extra_line_high_threshold",
          &profile.mainScalarLoadExtraLineHighThreshold},
         {"uniform_load_prep_system_cycles",
          &profile.simtUniformLoadPrepCycles},
         {"uniform_load_fill_system_cycles",
          &profile.simtUniformLoadFillCycles},
         {"uniform_load_diff_line_issue_system_cycles",
          &profile.simtUniformLoadDiffLineIssueCycles},
         {"mte3_store_prep_system_cycles", &profile.mte3StorePrepCycles},
         {"mte3_store_fill_system_cycles", &profile.mte3StoreFillCycles},
         {"uniform_store_base_system_cycles",
          &profile.simtUniformStoreBaseCycles}});
  }
  if (const auto *generated =
          resources->getObject("generated_vector_traffic")) {
    const std::string path = prefix + ".generated_vector_traffic";
    readOptionalNumbers(reader, *generated,
                        {{"shared_local_load_instructions_per_system_cycle",
                          &profile.sharedLocalLoadInstructionsPerCycle},
                         {"shared_local_store_instructions_per_system_cycle",
                          &profile.sharedLocalStoreInstructionsPerCycle},
                         {"private_stack_load_instructions_per_system_cycle",
                          &profile.privateStackLoadInstructionsPerCycle},
                         {"private_stack_store_instructions_per_system_cycle",
                          &profile.privateStackStoreInstructionsPerCycle}});
    if (profile.sharedLocalLoadInstructionsPerCycle < 0.0 ||
        profile.sharedLocalStoreInstructionsPerCycle < 0.0 ||
        profile.privateStackLoadInstructionsPerCycle < 0.0 ||
        profile.privateStackStoreInstructionsPerCycle < 0.0)
      reader.setError(path + " rates must be nonnegative");
  }
  if (context == "simd")
    if (const auto *curves =
            resources->getArray("whole_simd_completion_curves")) {
      const std::string basePath = prefix + ".whole_simd_completion_curves";
      for (auto indexed : llvm::enumerate(*curves)) {
        const auto *curve = indexed.value().getAsObject();
        const std::string path =
            basePath + "[" + std::to_string(indexed.index()) + "]";
        if (!curve) {
          reader.setError(path + " must be an object");
          continue;
        }
        WholeSimdCompletionCurve result;
        result.enabled = curve->getBoolean("enabled").value_or(false);
        readIntegers(
            reader, *curve, path,
            {
                {"minimum_component_nodes", &result.minimumComponentNodes},
                {"maximum_component_nodes", &result.maximumComponentNodes},
                {"dot_m", &result.m},
                {"dot_n", &result.n},
                {"dot_k", &result.k},
                {"element_bits", &result.elementBits},
                {"minimum_row_stride_bytes", &result.minimumRowStrideBytes},
                {"maximum_row_stride_bytes", &result.maximumRowStrideBytes},
                {"minimum_group_count", &result.minimumGroupCount},
                {"maximum_group_count", &result.maximumGroupCount},
            });
        result.dotUseSignature =
            reader.string(*curve, "dot_use_signature", path);
        readNumbers(
            reader, *curve, path,
            {
                {"entry_base_scale", &result.entryBaseScale},
                {"entry_saturated_scale", &result.entrySaturatedScale},
                {"entry_unaligned_group_midpoint",
                 &result.entryUnalignedGroupMidpoint},
                {"entry_group_midpoint", &result.entryGroupMidpoint},
                {"entry_group_power", &result.entryGroupPower},
                {"single_wave_entry_base_system_cycles",
                 &result.singleWaveEntryBaseSystemCycles},
                {"single_wave_entry_saturated_system_cycles",
                 &result.singleWaveEntrySaturatedSystemCycles},
                {"single_wave_entry_slot_midpoint",
                 &result.singleWaveEntrySlotMidpoint},
                {"single_wave_entry_slot_power",
                 &result.singleWaveEntrySlotPower},
                {"two_wave_entry_base_system_cycles",
                 &result.twoWaveEntryBaseSystemCycles},
                {"two_wave_entry_gaussian_system_cycles",
                 &result.twoWaveEntryGaussianSystemCycles},
                {"two_wave_entry_tail_center", &result.twoWaveEntryTailCenter},
                {"two_wave_entry_tail_width", &result.twoWaveEntryTailWidth},
                {"two_wave_entry_high_tail_penalty_system_cycles",
                 &result.twoWaveEntryHighTailPenaltySystemCycles},
                {"two_wave_entry_high_tail_midpoint",
                 &result.twoWaveEntryHighTailMidpoint},
                {"two_wave_entry_high_tail_power",
                 &result.twoWaveEntryHighTailPower},
                {"two_wave_entry_full_tail_system_cycles",
                 &result.twoWaveEntryFullTailSystemCycles},
                {"recurrence_setup_system_cycles_per_group",
                 &result.recurrenceSetupSystemCyclesPerGroup},
                {"dot_region_scale", &result.dotRegionScale},
                {"single_wave_dot_base_system_cycles",
                 &result.singleWaveDotBaseSystemCycles},
                {"single_wave_dot_saturated_system_cycles",
                 &result.singleWaveDotSaturatedSystemCycles},
                {"single_wave_dot_slot_midpoint",
                 &result.singleWaveDotSlotMidpoint},
                {"single_wave_dot_slot_power", &result.singleWaveDotSlotPower},
                {"two_wave_dot_target_system_cycles",
                 &result.twoWaveDotTargetSystemCycles},
                {"exit_base_scale", &result.exitBaseScale},
                {"exit_saturated_scale", &result.exitSaturatedScale},
                {"exit_unaligned_group_midpoint",
                 &result.exitUnalignedGroupMidpoint},
                {"exit_group_midpoint", &result.exitGroupMidpoint},
                {"exit_group_power", &result.exitGroupPower},
                {"single_wave_exit_base_system_cycles",
                 &result.singleWaveExitBaseSystemCycles},
                {"single_wave_exit_saturated_system_cycles",
                 &result.singleWaveExitSaturatedSystemCycles},
                {"single_wave_exit_slot_midpoint",
                 &result.singleWaveExitSlotMidpoint},
                {"single_wave_exit_slot_power",
                 &result.singleWaveExitSlotPower},
                {"two_wave_exit_base_system_cycles",
                 &result.twoWaveExitBaseSystemCycles},
                {"two_wave_exit_gaussian_system_cycles",
                 &result.twoWaveExitGaussianSystemCycles},
                {"two_wave_exit_tail_center", &result.twoWaveExitTailCenter},
                {"two_wave_exit_tail_width", &result.twoWaveExitTailWidth},
                {"kernel_boundary_system_cycles",
                 &result.kernelBoundarySystemCycles},
            });
        result.evidence = reader.string(*curve, "evidence", path);
        if (!result.isValid())
          reader.setError(path + " contains an invalid completion curve");
        else
          profile.wholeSimdCompletionCurves.push_back(std::move(result));
      }
    }
  if (context == "simt")
    if (const auto *floors =
            resources->getArray("dot_component_private_stack_floors")) {
      const std::string basePath =
          prefix + ".dot_component_private_stack_floors";
      for (auto indexed : llvm::enumerate(*floors)) {
        const auto *floor = indexed.value().getAsObject();
        const std::string path =
            basePath + "[" + std::to_string(indexed.index()) + "]";
        if (!floor) {
          reader.setError(path + " must be an object");
          continue;
        }
        DotComponentPrivateStackFloor result;
        result.enabled = floor->getBoolean("enabled").value_or(false);
        readIntegers(
            reader, *floor, path,
            {
                {"active_warp_count", &result.activeWarpCount},
                {"superblock_factor", &result.superblockFactor},
                {"minimum_component_nodes", &result.minimumComponentNodes},
                {"maximum_component_nodes", &result.maximumComponentNodes},
                {"dot_m", &result.m},
                {"dot_n", &result.n},
                {"dot_k", &result.k},
                {"element_bits", &result.elementBits},
                {"load_instruction_floor", &result.loadInstructionFloor},
                {"store_instruction_floor", &result.storeInstructionFloor},
            });
        result.dotUseSignature =
            reader.string(*floor, "dot_use_signature", path);
        readNumbers(
            reader, *floor, path,
            {
                {"combined_group_system_cycles", &result.combinedGroupCycles},
                {"pre_recurrence_fraction", &result.preRecurrenceFraction},
                {"dot_region_fraction", &result.dotRegionFraction},
                {"post_dot_fraction", &result.postDotFraction},
            });
        result.evidence = reader.string(*floor, "evidence", path);
        if (!result.isValid())
          reader.setError(path + " contains an invalid conditional floor");
        else
          profile.dotComponentPrivateStackFloors.push_back(std::move(result));
      }
    }
  if (context == "simt")
    if (const auto *curves =
            resources->getArray("whole_simt_completion_curves")) {
      const std::string basePath = prefix + ".whole_simt_completion_curves";
      for (auto indexed : llvm::enumerate(*curves)) {
        const auto *curve = indexed.value().getAsObject();
        const std::string path =
            basePath + "[" + std::to_string(indexed.index()) + "]";
        if (!curve) {
          reader.setError(path + " must be an object");
          continue;
        }
        WholeSimtCompletionCurve result;
        result.enabled = curve->getBoolean("enabled").value_or(false);
        readIntegers(
            reader, *curve, path,
            {
                {"active_warp_count", &result.activeWarpCount},
                {"superblock_factor", &result.superblockFactor},
                {"minimum_component_nodes", &result.minimumComponentNodes},
                {"maximum_component_nodes", &result.maximumComponentNodes},
                {"dot_m", &result.m},
                {"dot_n", &result.n},
                {"dot_k", &result.k},
                {"element_bits", &result.elementBits},
                {"reference_row_stride_bytes", &result.referenceRowStrideBytes},
            });
        result.dotUseSignature =
            reader.string(*curve, "dot_use_signature", path);
        readNumbers(reader, *curve, path,
                    {
                        {"reference_pre_recurrence_scalar_operations",
                         &result.referencePreRecurrenceScalarOperations},
                        {"recurrence_scale", &result.recurrenceScale},
                        {"dot_region_scale", &result.dotRegionScale},
                        {"entry_group_midpoint", &result.entryGroupMidpoint},
                        {"entry_group_power", &result.entryGroupPower},
                    });
        result.entryAdditiveCoefficients = {
            reader.number(*curve, "entry_intercept_system_cycles_per_group",
                          path),
            reader.number(*curve, "entry_log_stride_system_cycles_per_group",
                          path),
            reader.number(*curve, "entry_ramp_system_cycles_per_group", path),
            reader.number(
                *curve, "entry_log_stride_ramp_system_cycles_per_group", path),
            reader.number(*curve, "entry_scalar_excess_system_cycles_per_group",
                          path),
            reader.number(
                *curve,
                "entry_log_stride_scalar_excess_system_cycles_per_group", path),
            reader.number(*curve, "entry_constant_system_cycles", path),
            reader.number(*curve, "entry_constant_log_stride_system_cycles",
                          path)};
        const std::string exitRampKind =
            reader.string(*curve, "exit_ramp_kind", path);
        if (exitRampKind == "exponential")
          result.exitRampExponential = true;
        else if (exitRampKind == "hill")
          result.exitRampExponential = false;
        else
          reader.setError(path + ".exit_ramp_kind must be exponential or hill");
        result.exitGroupTimeConstant =
            reader.number(*curve, "exit_group_time_constant", path);
        result.exitGroupMidpoint =
            reader.number(*curve, "exit_group_midpoint", path);
        result.exitGroupPower = reader.number(*curve, "exit_group_power", path);
        result.exitAbsoluteCoefficients = {
            reader.number(*curve, "exit_intercept_system_cycles_per_group",
                          path),
            reader.number(*curve, "exit_log_stride_system_cycles_per_group",
                          path),
            reader.number(*curve, "exit_ramp_system_cycles_per_group", path),
            reader.number(*curve,
                          "exit_log_stride_ramp_system_cycles_per_group", path),
            reader.number(*curve, "exit_scalar_excess_system_cycles_per_group",
                          path),
            reader.number(
                *curve, "exit_log_stride_scalar_excess_system_cycles_per_group",
                path),
            reader.number(*curve, "exit_constant_system_cycles", path),
            reader.number(*curve, "exit_constant_log_stride_system_cycles",
                          path)};
        readNumbers(reader, *curve, path,
                    {
                        {"exit_tail_log2_active_fraction_system_cycles",
                         &result.exitTailLog2ActiveFractionCoefficient},
                        {"kernel_boundary_system_cycles",
                         &result.kernelBoundarySystemCycles},
                    });
        result.evidence = reader.string(*curve, "evidence", path);
        if (!result.isValid())
          reader.setError(path + " contains an invalid completion curve");
        else
          profile.wholeSimtCompletionCurves.push_back(std::move(result));
      }
    }
  readSourceFit(reader, *resources, "global_load_source_fit", prefix,
                profile.globalLoadSourceFit);
  readSourceFit(reader, *resources, "global_store_source_fit", prefix,
                profile.globalStoreSourceFit);
  readSourceFit(reader, *resources, "shared_load_source_fit", prefix,
                profile.sharedLoadSourceFit);
  readSourceFit(reader, *resources, "shared_store_source_fit", prefix,
                profile.sharedStoreSourceFit);
  readSourceFit(reader, *resources, "private_load_source_fit", prefix,
                profile.privateLoadSourceFit);
  readSourceFit(reader, *resources, "private_store_source_fit", prefix,
                profile.privateStoreSourceFit);
  if (const auto *scan = resources->getObject("prefix_scan")) {
    profile.prefixScanDependencyFactor = reader.optionalNumber(
        *scan, "dependency_factor", profile.prefixScanDependencyFactor);
    profile.straightLinePrefixScanElementsPerCycle = reader.optionalNumber(
        *scan, "straight_line_completion_elements_per_system_cycle", 0.0);
    if (auto evidence = scan->getString("straight_line_completion_evidence"))
      profile.straightLinePrefixScanCompletionEvidence = evidence->str();
    if (const auto *helpers =
            scan->getObject("straight_line_completion_by_combine")) {
      for (const auto &entry : *helpers) {
        const llvm::StringRef combine = entry.first;
        const std::string path =
            prefix + ".prefix_scan.straight_line_completion_by_combine." +
            combine.str();
        const auto *helper = entry.second.getAsObject();
        if (!helper) {
          reader.setError(path + " must be an object");
          continue;
        }
        StraightLinePrefixScanCost cost;
        cost.startupCycles =
            reader.number(*helper, "startup_system_cycles", path);
        cost.elementsPerCycle =
            reader.number(*helper, "elements_per_system_cycle", path);
        cost.dependencyStepCycles =
            reader.number(*helper, "dependency_step_system_cycles", path);
        readPositiveIntegerMeasurementCurve(
            *helper, "single_sequence_completion_system_cycle_measurements",
            "system_cycle", microbench, reader, path,
            cost.singleSequenceCompletionCyclesByExtent);
        cost.evidence = reader.string(*helper, "evidence", path);
        if (const auto *strided =
                helper->getObject("non_unit_axis_stride_completion")) {
          const std::string stridedPath =
              path + ".non_unit_axis_stride_completion";
          cost.nonUnitStrideStartupCycles =
              reader.number(*strided, "startup_system_cycles", stridedPath);
          cost.nonUnitStrideElementsPerCycle =
              reader.number(*strided, "elements_per_system_cycle", stridedPath);
          cost.nonUnitStrideMinimumAxisExtent =
              reader.integer(*strided, "minimum_axis_extent", stridedPath);
          cost.nonUnitStrideMaximumAxisExtent =
              reader.integer(*strided, "maximum_axis_extent", stridedPath);
          cost.nonUnitStrideMinimumAxisStride =
              reader.integer(*strided, "minimum_axis_stride", stridedPath);
          cost.nonUnitStrideMaximumAxisStride =
              reader.integer(*strided, "maximum_axis_stride", stridedPath);
          cost.nonUnitStrideMinimumSemanticElements = reader.integer(
              *strided, "minimum_semantic_elements", stridedPath);
          cost.nonUnitStrideMaximumSemanticElements = reader.integer(
              *strided, "maximum_semantic_elements", stridedPath);
          cost.nonUnitStrideEvidence =
              reader.string(*strided, "evidence", stridedPath);
        }
        if (!cost.isValid())
          reader.setError(path + " requires finite nonnegative costs, a "
                                 "positive element rate and evidence");
        profile.straightLinePrefixScanCostsByCombine[combine.str()] =
            std::move(cost);
      }
    }
    profile.prefixScanStepLatencyCycles = resolveNumberOrMeasurement(
        *scan, "step_latency_system_cycles", "step_latency_measurement",
        "system_cycle", microbench, reader, prefix + ".prefix_scan");
    if (auto maximum = scan->getInteger("maximum_reverse_axis_extent")) {
      if (*maximum <= 0)
        reader.setError(
            prefix +
            ".prefix_scan.maximum_reverse_axis_extent must be positive");
      else
        profile.maximumReversePrefixScanExtent = *maximum;
    }
  }
  if (const auto *recurrence =
          resources->getObject("loop_carried_recurrence")) {
    if (auto baseWarpCount = recurrence->getInteger("base_warp_count")) {
      if (*baseWarpCount <= 0)
        reader.setError(prefix +
                        ".loop_carried_recurrence.base_warp_count must be "
                        "positive");
      else
        profile.recurrenceGroupBaseWarpCount = *baseWarpCount;
    }
    readFactorMeasurementCurveFields(
        *recurrence, "system_cycle", microbench, reader,
        prefix + ".loop_carried_recurrence",
        {{"reduction_group_iteration_system_cycle_measurements",
          &profile.recurrenceReductionGroupCyclesByFactorAndLaneSteps},
         {"reduction_group_startup_system_cycle_measurements",
          &profile.recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps}});
    if (const auto *completion =
            recurrence->getObject("single_result_row_reduction_completion")) {
      const std::string path =
          prefix +
          ".loop_carried_recurrence.single_result_row_reduction_completion";
      profile.singleResultRowReductionRecurrenceCompletionEnabled = true;
      profile.singleResultRowReductionRecurrenceMinimumExtent =
          reader.integer(*completion, "minimum_reduction_extent", path);
      profile.singleResultRowReductionRecurrenceMaximumExtent =
          reader.integer(*completion, "maximum_reduction_extent", path);
      profile.singleResultRowReductionRecurrenceMinimumIterations =
          reader.integer(*completion, "minimum_iteration_count", path);
      profile.singleResultRowReductionRecurrenceMaximumIterations =
          reader.integer(*completion, "maximum_iteration_count", path);
      profile.singleResultRowReductionRecurrenceMinimumPrograms =
          reader.integer(*completion, "minimum_source_program_count", path);
      profile.singleResultRowReductionRecurrenceMaximumPrograms =
          reader.integer(*completion, "maximum_source_program_count", path);
      profile.singleResultRowReductionRecurrenceIterationCycles =
          reader.number(*completion, "iteration_system_cycles", path);
      profile.singleResultRowReductionRecurrenceStartupInterceptCycles =
          reader.number(*completion, "startup_intercept_system_cycles", path);
      profile.singleResultRowReductionRecurrenceStartupSqrtProgramCycles =
          reader.number(*completion, "startup_sqrt_program_system_cycles",
                        path);
      profile.singleResultRowReductionRecurrenceEvidence =
          reader.string(*completion, "evidence", path);
    }
  }
  if (const auto *rowwise = resources->getObject("rowwise_reduction")) {
    const std::string path = prefix + ".rowwise_reduction";
    profile.rowwiseReductionGroupBaseWarpCount =
        reader.integer(*rowwise, "base_warp_count", path);
    profile.rowwiseReductionGroupReferenceOperations =
        reader.integer(*rowwise, "reference_reduction_operations", path);
    profile.rowwiseReductionGroupEvidence =
        reader.string(*rowwise, "evidence", path);
    readFactorRowExtentMeasurementSurfaces(
        *rowwise, "group_body_system_cycle_measurements", microbench, reader,
        path, profile.rowwiseReductionGroupCyclesByFactorRowsAndExtent);
    if (const auto *operationSurfaces =
            rowwise->getObject("operation_count_surfaces")) {
      for (const auto &operationEntry : *operationSurfaces) {
        int64_t operations = 0;
        const llvm::StringRef operationKey(operationEntry.first);
        if (operationKey.getAsInteger(10, operations) || operations <= 0) {
          reader.setError(path +
                          ".operation_count_surfaces keys must be positive "
                          "integers");
          break;
        }
        const auto *surface = operationEntry.second.getAsObject();
        if (!surface) {
          reader.setError(path + ".operation_count_surfaces." + operationKey +
                          " must be an object");
          break;
        }
        const std::string operationPath =
            path + ".operation_count_surfaces." + operationKey.str();
        readFactorRowExtentMeasurementSurfaces(
            *surface, "group_body_system_cycle_measurements", microbench,
            reader, operationPath,
            profile.rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent
                [operations]);
      }
    }
  }
  if (const auto *multiResult =
          resources->getObject("multi_result_reduction")) {
    const std::string path = prefix + ".multi_result_reduction";
    profile.multiResultReductionGroupBaseWarpCount =
        reader.integer(*multiResult, "base_warp_count", path);
    profile.multiResultReductionResultCount =
        reader.integer(*multiResult, "result_count", path);
    profile.multiResultReductionElementBits =
        reader.integer(*multiResult, "element_bit_width", path);
    profile.multiResultReductionPaddedAxisExtent =
        reader.integer(*multiResult, "padded_axis_extent", path);
    profile.multiResultReductionGroupEvidence =
        reader.string(*multiResult, "evidence", path);
    if (const auto *classes =
            multiResult->getObject("group_body_system_cycles_by_tail_class")) {
      for (const auto &classEntry : *classes) {
        const llvm::StringRef tailClass(classEntry.first);
        const auto *curveObject = classEntry.second.getAsObject();
        if (!curveObject) {
          reader.setError(path + ".group_body_system_cycles_by_tail_class." +
                          tailClass.str() + " must be an object");
          break;
        }
        auto &curve = profile.multiResultReductionGroupCyclesByTailClassAndRows
                          [tailClass.str()];
        for (const auto &pointEntry : *curveObject) {
          int64_t rows = 0;
          const llvm::StringRef rowKey(pointEntry.first);
          const auto value = pointEntry.second.getAsNumber();
          if (rowKey.getAsInteger(10, rows) || rows <= 0 || !value ||
              !std::isfinite(*value) || *value <= 0.0) {
            reader.setError(path + ".group_body_system_cycles_by_tail_class." +
                            tailClass.str() +
                            " must map positive row keys to positive numbers");
            break;
          }
          curve[rows] = *value;
        }
      }
    }
  }
  if (const auto *twoPass =
          resources->getObject("two_pass_rowwise_completion")) {
    const std::string path = prefix + ".two_pass_rowwise_completion";
    if (auto owns = twoPass->getBoolean("owns_segmented_indirect_loads"))
      profile.twoPassRowwiseOwnsSegmentedIndirectLoads = *owns;
    readFactorRowExtentMeasurementSurfaces(
        *twoPass, "residual_system_cycle_measurements", microbench, reader,
        path,
        profile
            .twoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot);
    profile.twoPassRowwiseCompletionEvidence =
        reader.string(*twoPass, "evidence", path);
  }
  if (const auto *continuous = resources->getObject("continuous_memory")) {
    if (auto baseWarpCount = continuous->getInteger("base_warp_count")) {
      if (*baseWarpCount <= 0)
        reader.setError(prefix +
                        ".continuous_memory.base_warp_count must be positive");
      else
        profile.continuousMemoryGroupBaseWarpCount = *baseWarpCount;
    }
    readFactorMeasurementCurveFields(
        *continuous, "system_cycle", microbench, reader,
        prefix + ".continuous_memory",
        {{"load_group_operation_system_cycle_measurements",
          &profile.continuousLoadGroupCyclesByFactorAndBytes},
         {"store_group_operation_system_cycle_measurements",
          &profile.continuousStoreGroupCyclesByFactorAndBytes},
         {"load_group_startup_system_cycle_measurements",
          &profile.continuousLoadGroupStartupCyclesByFactorAndBytes},
         {"store_group_startup_system_cycle_measurements",
          &profile.continuousStoreGroupStartupCyclesByFactorAndBytes}});
  }
  if (context == "simd")
    if (const auto *completion =
            resources->getObject("mixed_parent_memory_completion")) {
      const std::string path = prefix + ".mixed_parent_memory_completion";
      profile.mixedParentMemoryCompletionEnabled =
          completion->getBoolean("enabled").value_or(false);
      profile.mixedParentMemoryMinimumIterations =
          reader.integer(*completion, "minimum_parent_iterations", path);
      profile.mixedParentMemoryReferenceBytesPerProgram =
          reader.integer(*completion, "reference_bytes_per_program", path);
      profile.mixedParentMemoryCyclesPerExcessIteration = reader.number(
          *completion, "system_cycles_per_excess_parent_iteration", path);
      profile.mixedParentMemoryCompletionEvidence =
          reader.string(*completion, "evidence", path);
    }
  if (const auto *layout = resources->getObject("layout_memory")) {
    readLayoutCostMap(reader, *layout, "load", prefix + ".layout_memory",
                      profile.loadLayoutCosts);
    readLayoutCostMap(reader, *layout, "store", prefix + ".layout_memory",
                      profile.storeLayoutCosts);
  }
  if (context == "simt")
    readLayoutCostMap(reader, *resources, "dot_store_shared_layout", prefix,
                      profile.dotStoreSharedLayoutCosts);
  if (context == "simt")
    readLayoutCostMap(reader, *resources, "dot_operand_paths", prefix,
                      profile.dotOperandPathCosts);
  if (const auto *service = resources->getObject("store_footprint_service")) {
    const std::string path = prefix + ".store_footprint_service";
    if (reader.string(*service, "evidence", path).empty())
      reader.setError(path + ".evidence must identify the calibration data");
    profile.storeFootprintUnitBytes =
        reader.integer(*service, "unit_bytes", path);
    profile.storeCyclesPerFootprintUnit =
        reader.number(*service, "system_cycles_per_unit", path);
    if (const auto *floors =
            reader.object(*service, "issue_floor_by_warps", path))
      for (const auto &entry : *floors) {
        const llvm::StringRef key = entry.first;
        int64_t warps = 0;
        auto value = entry.second.getAsNumber();
        if (key.getAsInteger(10, warps) || warps <= 0 || !value || *value <= 0)
          reader.setError(path + " has an invalid warp-count floor");
        else
          profile.storeIssueFloorByWarpCount[warps] = *value;
      }
  }
  if (const auto *indirect =
          reader.object(*resources, "indirect_memory", prefix)) {
    const std::string path = prefix + ".indirect_memory";
    profile.indirectLoadTransactionsPerCycle =
        reader.optionalNumber(*indirect, "load_transactions_per_system_cycle",
                              profile.indirectLoadTransactionsPerCycle);
    profile.indirectStoreTransactionsPerCycle =
        reader.optionalNumber(*indirect, "store_transactions_per_system_cycle",
                              profile.indirectStoreTransactionsPerCycle);
    profile.indirectDependencyLatencyCycles =
        reader.optionalNumber(*indirect, "dependency_latency_system_cycles",
                              profile.indirectDependencyLatencyCycles);
    readPositiveIntegerMeasurementCurves(
        *indirect, "system_cycle", microbench, reader, path,
        {{"load_system_cycle_measurements",
          &profile.indirectLoadSystemCyclesByWarpInstructions},
         {"load_element_system_cycle_measurements",
          &profile.indirectLoadSystemCyclesByElements},
         {"load_startup_system_cycle_measurements",
          &profile.indirectLoadStartupSystemCyclesByWarpInstructions},
         {"store_system_cycle_measurements",
          &profile.indirectStoreSystemCyclesBySegments}});
    readFactorMeasurementCurveFields(
        *indirect, "system_cycle", microbench, reader, path,
        {{"load_group_system_cycle_measurements",
          &profile.indirectLoadGroupCyclesByFactorAndWarpInstructions},
         {"load_group_startup_system_cycle_measurements",
          &profile.indirectLoadGroupStartupCyclesByFactorAndWarpInstructions}});
    readFactorRowExtentMeasurementSurfaceFields(
        *indirect, microbench, reader, path,
        {{"segmented_load_group_system_cycle_measurements",
          &profile
               .segmentedIndirectLoadGroupCyclesByFactorSegmentsAndContiguousElements},
         {"segmented_load_fp32_group_system_cycle_measurements",
          &profile
               .segmentedIndirectLoadFp32GroupCyclesByFactorSegmentsAndContiguousElements}});
    readPositiveIntegerMeasurementCurves(
        *indirect, "system_cycle", microbench, reader, path,
        {{"grid_stride_load_system_cycle_measurements",
          &profile.gridStrideIndirectLoadCyclesByWarpInstructions},
         {"grid_stride_load_startup_system_cycle_measurements",
          &profile.gridStrideIndirectLoadStartupCyclesByWarpInstructions}});
    readFactorMeasurementCurveFields(
        *indirect, "system_cycle", microbench, reader, path,
        {{"grid_stride_load_factor_system_cycle_measurements",
          &profile.gridStrideIndirectLoadCyclesByFactorAndWarpInstructions},
         {"grid_stride_load_factor_startup_system_cycle_measurements",
          &profile
               .gridStrideIndirectLoadStartupCyclesByFactorAndWarpInstructions}});
  }
  if (const auto *atomic = resources->getObject("atomic_memory")) {
    const std::string path = prefix + ".atomic_memory";
    auto readRate = [&](const llvm::json::Object &object, llvm::StringRef key) {
      const std::string ratePath = path + "." + key.str();
      StageAtomicRate rate;
      rate.logicalElementsPerCycle =
          reader.number(object, "logical_elements_per_system_cycle", ratePath);
      rate.singlePhysicalProgramLogicalElementsPerCycle = reader.optionalNumber(
          object, "single_physical_program_logical_elements_per_system_cycle",
          0.0);
      rate.operationStartupCycles =
          reader.number(object, "operation_startup_system_cycles", ratePath);
      rate.resultDependencyCycles =
          reader.number(object, "result_dependency_system_cycles", ratePath);
      rate.unknownContentionMultiplier =
          reader.number(object, "unknown_contention_multiplier", ratePath);
      profile.atomicRates[key] = rate;
    };
    if (const auto *fallback = atomic->getObject("default"))
      readRate(*fallback, "default");
    if (const auto *operations = atomic->getObject("operations")) {
      for (const auto &[key, value] : *operations) {
        const auto *operation = value.getAsObject();
        if (!operation) {
          reader.setError(path + ".operations." + key.str() +
                          " must be an object");
          break;
        }
        readRate(*operation, key);
      }
    }
  }
  if (const auto *control = resources->getObject("control_flow")) {
    const std::string path = prefix + ".control_flow";
    profile.controlFlow.loopBackedgeCycles =
        reader.number(*control, "loop_backedge_system_cycles", path);
    profile.controlFlow.conditionalBranchCycles =
        reader.number(*control, "conditional_branch_system_cycles", path);
    profile.controlFlow.divergentBranchPenaltyCycles =
        reader.number(*control, "divergent_branch_penalty_system_cycles", path);
    profile.controlFlow.synchronizationCycles =
        reader.number(*control, "synchronization_system_cycles", path);
  }
  if (const auto *gridStride = resources->getObject("grid_stride_loop")) {
    const std::string path = prefix + ".grid_stride_loop";
    if (const auto *resident =
            gridStride->getObject("resident_direct_memory")) {
      const std::string residentPath = path + ".resident_direct_memory";
      profile.gridStrideResidentMemoryMaximumBytes =
          reader.integer(*resident, "maximum_total_bytes", residentPath);
      profile.gridStrideResidentLoadWarpInstructionsPerCycle =
          resolveNumberOrMeasurement(
              *resident, "load_warp_instructions_per_system_cycle",
              "load_warp_instructions_per_system_cycle_measurement",
              "warp_instruction/system_cycle", microbench, reader,
              residentPath);
      profile.gridStrideResidentStoreWarpInstructionsPerCycle =
          resolveNumberOrMeasurement(
              *resident, "store_warp_instructions_per_system_cycle",
              "store_warp_instructions_per_system_cycle_measurement",
              "warp_instruction/system_cycle", microbench, reader,
              residentPath);
      profile.gridStrideResidentMemoryEvidence =
          reader.string(*resident, "evidence", residentPath);
    }
    readPositiveIntegerMeasurementCurves(
        *gridStride, "system_cycle", microbench, reader, path,
        {{"dot_system_cycle_measurements", &profile.gridStrideDotCyclesByFlops},
         {"dot_startup_system_cycle_measurements",
          &profile.gridStrideDotStartupCyclesByFlops}});
    readFactorMeasurementCurveFields(
        *gridStride, "system_cycle", microbench, reader, path,
        {{"dot_factor_system_cycle_measurements",
          &profile.gridStrideDotCyclesByFactorAndFlops},
         {"dot_factor_startup_system_cycle_measurements",
          &profile.gridStrideDotStartupCyclesByFactorAndFlops}});
    readFactorRowExtentMeasurementSurfaceFields(
        *gridStride, microbench, reader, path,
        {{"indirect_dot_composition_body_residual_system_cycle_measurements",
          &profile
               .gridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry},
         {"indirect_dot_composition_startup_residual_system_cycle_measurements",
          &profile
               .gridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry}});
    if (gridStride->get(
            "indirect_dot_composition_body_residual_system_cycle_measurements"))
      profile.gridStrideIndirectDotCompositionEvidence =
          reader.string(*gridStride, "indirect_dot_composition_evidence", path);
    readPositiveIntegerMeasurementCurves(
        *gridStride, "system_cycle", microbench, reader, path,
        {{"forward_scan_system_cycle_measurements",
          &profile.gridStrideForwardScanCyclesByExtent},
         {"forward_scan_startup_system_cycle_measurements",
          &profile.gridStrideForwardScanStartupCyclesByExtent},
         {"reverse_scan_system_cycle_measurements",
          &profile.gridStrideReverseScanCyclesByExtent},
         {"reverse_scan_startup_system_cycle_measurements",
          &profile.gridStrideReverseScanStartupCyclesByExtent}});
    readFactorMeasurementCurveFields(
        *gridStride, "system_cycle", microbench, reader, path,
        {{"forward_scan_factor_system_cycle_measurements",
          &profile.gridStrideForwardScanCyclesByFactorAndExtent},
         {"forward_scan_factor_startup_system_cycle_measurements",
          &profile.gridStrideForwardScanStartupCyclesByFactorAndExtent},
         {"reverse_scan_factor_system_cycle_measurements",
          &profile.gridStrideReverseScanCyclesByFactorAndExtent},
         {"reverse_scan_factor_startup_system_cycle_measurements",
          &profile.gridStrideReverseScanStartupCyclesByFactorAndExtent}});
  }
}

static void readSetupCycleCurve(ProfileJSONReader &reader,
                                const llvm::json::Object &setup,
                                const MicrobenchmarkProfile *microbench,
                                StageModeProfile &profile) {
  const auto *curve = setup.getObject("empty_launch_measurements");
  if (!curve)
    return;
  if (!microbench) {
    reader.setError("SIMT setup curve requires microbenchmark_profile");
    return;
  }
  for (const auto &entry : *curve) {
    int64_t warpCount = 0;
    if (llvm::StringRef(entry.first).getAsInteger(10, warpCount) ||
        warpCount <= 0) {
      reader.setError("SIMT setup curve keys must be positive warp counts");
      return;
    }
    const auto measurement = entry.second.getAsString();
    if (!measurement) {
      reader.setError("SIMT setup curve values must be measurement names");
      return;
    }
    auto cycles = microbench->requireValue(*measurement, "system_cycle");
    if (!cycles) {
      reader.setError(llvm::toString(cycles.takeError()));
      return;
    }
    profile.setupCyclesByWarpCount[warpCount] = *cycles;
  }
}

static llvm::Expected<CandidateProfile>
loadCandidateProfileUncached(llvm::StringRef pathRef,
                             std::string *microbenchmarkPath) {
  std::string path = pathRef.str();
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read SIMD/SIMT profile '%s'",
                                   path.c_str());
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to parse SIMD/SIMT profile '%s': %s",
                                   path.c_str(),
                                   llvm::toString(parsed.takeError()).c_str());
  const auto *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile root must be an object");
  auto selectionSchemaVersion = root->getInteger("schema_version");
  if (!selectionSchemaVersion ||
      (*selectionSchemaVersion != 18 && *selectionSchemaVersion != 19))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile schema_version must be 18 or 19");
  const bool usesGenericResourceFormula = *selectionSchemaVersion == 19;

  CandidateProfile profile;
  ProfileJSONReader reader;
  std::optional<MicrobenchmarkProfile> microbenchmarkProfile;
  if (auto reference = root->getString("microbenchmark_profile")) {
    std::string resolved = resolveProfileReference(path, *reference);
    if (microbenchmarkPath)
      *microbenchmarkPath = resolved;
    auto loaded = MicrobenchmarkProfile::loadFromFile(resolved);
    if (!loaded)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to load shared microbenchmark profile referenced by '%s': %s",
          path.c_str(), llvm::toString(loaded.takeError()).c_str());
    microbenchmarkProfile.emplace(std::move(*loaded));
    profile.microbenchmarkProfileVersion =
        microbenchmarkProfile->getProfileVersion().str();
    profile.microbenchmarkProfileTarget =
        microbenchmarkProfile->getTarget().str();
    profile.microbenchmarkContentSha256 =
        microbenchmarkProfile->getContentSha256().str();
  }
  const MicrobenchmarkProfile *microbench =
      microbenchmarkProfile ? &*microbenchmarkProfile : nullptr;

  HardwareProfile &hardware = profile.hardware;
  hardware.profileVersion = reader.string(*root, "profile_version", "profile");
  hardware.target = reader.string(*root, "target", "profile");
  if (const auto *targets = root->getArray("compatible_targets")) {
    for (const llvm::json::Value &target : *targets) {
      auto pattern = target.getAsString();
      if (!pattern || pattern->empty()) {
        reader.setError(
            "profile.compatible_targets entries must be non-empty strings");
        break;
      }
      profile.compatibleTargets.push_back(pattern->str());
    }
    if (profile.compatibleTargets.empty())
      reader.setError("profile.compatible_targets must not be empty");
  } else {
    reader.setError("profile.compatible_targets must be an array");
  }
  if (microbench && llvm::StringRef(hardware.target) != microbench->getTarget())
    reader.setError("selection profile target '" + hardware.target +
                    "' does not match shared microbenchmark target '" +
                    microbench->getTarget().str() + "'");
  profile.scoreUnit = reader.string(*root, "score_unit", "profile");
  const auto *calibration =
      reader.object(*root, "selection_calibration", "profile");
  if (calibration)
    profile.structural.tinyDotFlopsMax = reader.integer(
        *calibration, "tiny_dot_flops_max", "profile.selection_calibration");

  const auto *kernelTiming = root->getObject("kernel_timing");
  if (usesGenericResourceFormula && !kernelTiming)
    reader.setError("profile.kernel_timing must be an object for schema 19");
  if (kernelTiming) {
    auto &timing = profile.kernelTiming;
    if (const auto *completion =
            kernelTiming->getObject("stage_dependency_completion")) {
      const std::string path =
          "profile.kernel_timing.stage_dependency_completion";
      auto &result = timing.stageDependencyCompletion;
      result.enabled = reader.string(*completion, "status", path) ==
                       "qualified_stage_dependency_completion_v1";
      if (!result.enabled)
        reader.setError(path + " has an unknown qualification status");
      result.target = reader.string(*completion, "target", path);
      result.loweringContract =
          reader.string(*completion, "lowering_contract", path);
      result.taskEnvelopeSystemCycles =
          reader.number(*completion, "task_envelope_system_cycles", path);
      result.dynamicCVFirstFillSystemCycles = reader.number(
          *completion, "dynamic_cv_first_fill_system_cycles", path);
      result.dynamicCVPerIterationDeltaSystemCycles = reader.number(
          *completion, "dynamic_cv_per_iteration_delta_system_cycles", path);
      result.scheduleContract =
          reader.string(*completion, "schedule_contract", path);
      result.fullKernelValidationSHA256 =
          reader.string(*completion, "full_kernel_validation_sha256", path);
      result.evidence = reader.string(*completion, "evidence", path);
    }
    timing.scalarPrologueSystemCycles =
        reader.number(*kernelTiming, "scalar_prologue_system_cycles",
                      "profile.kernel_timing");
    timing.scalarEpilogueSystemCycles =
        reader.number(*kernelTiming, "scalar_epilogue_system_cycles",
                      "profile.kernel_timing");
    readPositiveIntegerNumberCurve(
        *kernelTiming,
        "boundary_residual_system_cycles_by_physical_program_count", reader,
        "profile.kernel_timing",
        timing.boundaryResidualSystemCyclesByPhysicalProgramCount);
    readRouteFactorNumberCurves(
        *kernelTiming,
        "boundary_residual_system_cycles_by_route_factor_and_physical_program_"
        "count",
        reader, "profile.kernel_timing",
        timing
            .boundaryResidualSystemCyclesByRouteFactorAndPhysicalProgramCount);
    readRouteNumberMap(*kernelTiming,
                       "global_load_boundary_residual_system_cycles_by_route",
                       reader, "profile.kernel_timing",
                       timing.globalLoadBoundaryResidualSystemCyclesByRoute);
    readRuntimeArgumentSetupByRoute(*kernelTiming, reader,
                                    "profile.kernel_timing",
                                    timing.runtimeArgumentSetupByRoute);
    if (const auto *scalarizedRank2 = kernelTiming->getObject(
            "scalarized_rank2_indirect_load_completion")) {
      const std::string path =
          "profile.kernel_timing.scalarized_rank2_indirect_load_completion";
      timing.scalarizedRank2IndirectLoadMaximumElements =
          reader.integer(*scalarizedRank2, "maximum_elements", path);
      readPositiveIntegerNumberCurve(
          *scalarizedRank2, "residual_system_cycles_by_physical_program_count",
          reader, path,
          timing.scalarizedRank2IndirectLoadResidualByPhysicalProgramCount);
      timing.scalarizedRank2IndirectLoadEvidence =
          reader.string(*scalarizedRank2, "evidence", path);
    }
    if (const auto *rank2Direct = kernelTiming->getObject(
            "simd_rank2_strided_direct_load_completion")) {
      const std::string path =
          "profile.kernel_timing.simd_rank2_strided_direct_load_completion";
      auto &result = timing.simdRank2StridedDirectLoad;
      readIntegers(reader, *rank2Direct, path,
                   {{"minimum_elements", &result.minimumElements},
                    {"maximum_elements", &result.maximumElements}});
      readNumbers(reader, *rank2Direct, path,
                  {{"startup_system_cycles", &result.startupCycles},
                   {"per_element_system_cycles", &result.perElementCycles},
                   {"maximum_residual_system_cycles", &result.maximumCycles}});
      result.evidence = reader.string(*rank2Direct, "evidence", path);
    }
    if (const auto *rank2Direct = kernelTiming->getObject(
            "simt_rank2_strided_direct_load_completion")) {
      const std::string path =
          "profile.kernel_timing.simt_rank2_strided_direct_load_completion";
      auto &result = timing.simtRank2StridedDirectLoad;
      readIntegers(reader, *rank2Direct, path,
                   {{"minimum_elements", &result.minimumElements},
                    {"maximum_elements", &result.maximumElements}});
      readNumbers(reader, *rank2Direct, path,
                  {{"startup_system_cycles", &result.startupCycles},
                   {"per_element_system_cycles", &result.perElementCycles},
                   {"maximum_target_system_cycles", &result.maximumCycles}});
      result.evidence = reader.string(*rank2Direct, "evidence", path);
    }
    if (const auto *simtRank2 =
            kernelTiming->getObject("simt_rank2_indirect_load_completion")) {
      const std::string path =
          "profile.kernel_timing.simt_rank2_indirect_load_completion";
      timing.simtRank2IndirectLoadMaximumElements =
          reader.integer(*simtRank2, "maximum_elements", path);
      readSignedPositiveIntegerNumberCurves(
          *simtRank2, "residual_system_cycles_by_element_bits_and_elements",
          reader, path,
          timing.simtRank2IndirectLoadResidualByElementBitsAndElements);
      timing.simtRank2IndirectLoadEvidence =
          reader.string(*simtRank2, "evidence", path);
    }
    if (const auto *subwarpComposite =
            kernelTiming->getObject("subwarp_indirect_composite_completion")) {
      const std::string path =
          "profile.kernel_timing.subwarp_indirect_composite_completion";
      timing.subwarpIndirectCompositeMaximumElements =
          reader.integer(*subwarpComposite, "maximum_elements", path);
      readPositiveIntegerNumberCurve(
          *subwarpComposite, "residual_system_cycles_by_elements", reader, path,
          timing.subwarpIndirectCompositeResidualByElements);
      timing.subwarpIndirectCompositeEvidence =
          reader.string(*subwarpComposite, "evidence", path);
    }
    if (const auto *multiResultScan = kernelTiming->getObject(
            "simd_multi_result_prefix_scan_completion")) {
      const std::string path =
          "profile.kernel_timing.simd_multi_result_prefix_scan_completion";
      auto &result = timing.routeCompletions.multiResultPrefixScan;
      result.enabled = true;
      readIntegers(reader, *multiResultScan, path,
                   {{"minimum_source_programs", &result.minimumPrograms},
                    {"maximum_source_programs", &result.maximumPrograms},
                    {"minimum_padded_axis", &result.minimumPaddedAxis},
                    {"maximum_padded_axis", &result.maximumPaddedAxis},
                    {"required_result_count", &result.requiredResultCount},
                    {"required_independent_sequences",
                     &result.requiredIndependentSequences}});
      readNumbers(
          reader, *multiResultScan, path,
          {{"minimum_active_ratio", &result.minimumActiveRatio},
           {"maximum_active_ratio", &result.maximumActiveRatio},
           {"elapsed_intercept_system_cycles", &result.elapsedInterceptCycles},
           {"elapsed_sqrt_program_system_cycles",
            &result.elapsedSqrtProgramCycles},
           {"elapsed_inactive_ratio_system_cycles",
            &result.elapsedInactiveRatioCycles},
           {"elapsed_padded_axis_system_cycles",
            &result.elapsedPaddedAxisCycles}});
      result.evidence = reader.string(*multiResultScan, "evidence", path);
    }
    if (const auto *scan = kernelTiming->getObject(
            "rank2_interleaved_prefix_scan_completion")) {
      const std::string path =
          "profile.kernel_timing.rank2_interleaved_prefix_scan_completion";
      auto &result = timing.rank2InterleavedPrefixScanCompletion;
      result.enabled = true;
      readIntegers(reader, *scan, path,
                   {{"minimum_source_programs", &result.minimumSourcePrograms},
                    {"maximum_source_programs", &result.maximumSourcePrograms},
                    {"minimum_scan_extent", &result.minimumExtent},
                    {"maximum_scan_extent", &result.maximumExtent},
                    {"minimum_independent_sequences",
                     &result.minimumIndependentSequences},
                    {"maximum_independent_sequences",
                     &result.maximumIndependentSequences},
                    {"factor2_reference_physical_program_count",
                     &result.factor2ReferencePhysicalProgramCount}});
      readNumbers(
          reader, *scan, path,
          {{"startup_system_cycles", &result.startupSystemCycles},
           {"per_element_system_cycles", &result.perElementSystemCycles},
           {"factor2_startup_system_cycles",
            &result.factor2StartupSystemCycles},
           {"factor2_per_element_system_cycles",
            &result.factor2PerElementSystemCycles},
           {"factor2_per_active_physical_program_system_cycles",
            &result.factor2PerActivePhysicalProgramSystemCycles},
           {"factor2_minimum_target_system_cycles",
            &result.factor2MinimumTargetSystemCycles},
           {"factor4_startup_system_cycles",
            &result.factor4StartupSystemCycles},
           {"factor4_per_element_system_cycles",
            &result.factor4PerElementSystemCycles}});
      result.evidence = reader.string(*scan, "evidence", path);
    }
    if (const auto *staticLoop = kernelTiming->getObject(
            "simt_static_loop_direct_memory_completion")) {
      const std::string path =
          "profile.kernel_timing.simt_static_loop_direct_memory_completion";
      auto &result = timing.simtStaticLoopDirectMemoryCompletion;
      result.enabled = true;
      readIntegers(
          reader, *staticLoop, path,
          {{"minimum_static_iterations", &result.minimumStaticIterations},
           {"maximum_static_iterations", &result.maximumStaticIterations},
           {"required_contiguous_elements_per_access",
            &result.requiredContiguousElements},
           {"required_element_bits", &result.requiredElementBits},
           {"minimum_load_count", &result.minimumLoadCount},
           {"maximum_load_count", &result.maximumLoadCount},
           {"required_store_count", &result.requiredStoreCount},
           {"maximum_superblock_factor", &result.maximumSuperblockFactor}});
      readNumbers(reader, *staticLoop, path,
                  {{"startup_system_cycles", &result.startupSystemCycles}});
      readPositiveIntegerNumberCurve(
          *staticLoop, "per_iteration_system_cycles_by_load_count", reader,
          path, result.perIterationSystemCyclesByLoadCount);
      result.evidence = reader.string(*staticLoop, "evidence", path);
    }
    if (const auto *selectionNetwork = kernelTiming->getObject(
            "simt_pairwise_selection_network_completion")) {
      const std::string path =
          "profile.kernel_timing.simt_pairwise_selection_network_completion";
      auto &result = timing.routeCompletions.pairwiseSelectionNetwork;
      result.enabled = true;
      readIntegers(
          reader, *selectionNetwork, path,
          {{"minimum_input_elements", &result.minimumInputElements},
           {"maximum_input_elements", &result.maximumInputElements},
           {"minimum_output_columns", &result.minimumOutputColumns},
           {"maximum_output_columns", &result.maximumOutputColumns},
           {"minimum_reduction_stages", &result.minimumReductionStages}});
      readNumbers(reader, *selectionNetwork, path,
                  {{"intercept_system_cycles", &result.interceptCycles},
                   {"input_element_system_cycles", &result.inputElementCycles},
                   {"log2_output_column_system_cycles",
                    &result.log2OutputColumnCycles}});
      result.evidence = reader.string(*selectionNetwork, "evidence", path);
    }
    if (const auto *largeScan = kernelTiming->getObject(
            "simd_large_sequence_minmax_scan_completion")) {
      const std::string path =
          "profile.kernel_timing.simd_large_sequence_minmax_scan_completion";
      auto &result = timing.routeCompletions.largeSequenceMinmaxScan;
      result.enabled = true;
      readIntegers(reader, *largeScan, path,
                   {{"minimum_independent_sequences",
                     &result.minimumIndependentSequences},
                    {"maximum_independent_sequences",
                     &result.maximumIndependentSequences},
                    {"minimum_scan_extent", &result.minimumExtent},
                    {"maximum_scan_extent", &result.maximumExtent}});
      readNumbers(reader, *largeScan, path,
                  {{"per_independent_sequence_system_cycles",
                    &result.perIndependentSequenceSystemCycles}});
      result.evidence = reader.string(*largeScan, "evidence", path);
    }
    if (const auto *minmaxScan = kernelTiming->getObject(
            "simt_multi_sequence_minmax_scan_completion")) {
      const std::string path =
          "profile.kernel_timing.simt_multi_sequence_minmax_scan_completion";
      auto &result = timing.routeCompletions.multiSequenceMinmaxScan;
      result.enabled = true;
      readIntegers(reader, *minmaxScan, path,
                   {{"minimum_independent_sequences",
                     &result.minimumIndependentSequences},
                    {"maximum_independent_sequences",
                     &result.maximumIndependentSequences},
                    {"minimum_scan_extent", &result.minimumExtent},
                    {"maximum_scan_extent", &result.maximumExtent}});
      readNumbers(
          reader, *minmaxScan, path,
          {{"base_reduction_system_cycles", &result.baseReductionSystemCycles},
           {"per_independent_sequence_reduction_system_cycles",
            &result.perIndependentSequenceReductionSystemCycles},
           {"maximum_reduction_system_cycles",
            &result.maximumReductionSystemCycles}});
      readOptionalIntegers(
          *minmaxScan,
          {{"large_minimum_independent_sequences",
            &result.largeMinimumIndependentSequences},
           {"large_maximum_independent_sequences",
            &result.largeMaximumIndependentSequences},
           {"large_minimum_scan_extent", &result.largeMinimumExtent},
           {"large_maximum_scan_extent", &result.largeMaximumExtent}});
      readOptionalNumbers(
          reader, *minmaxScan,
          {{"large_base_reduction_system_cycles",
            &result.largeBaseReductionSystemCycles},
           {"large_per_independent_sequence_reduction_system_cycles",
            &result.largePerIndependentSequenceReductionSystemCycles},
           {"large_maximum_reduction_system_cycles",
            &result.largeMaximumReductionSystemCycles}});
      result.evidence = reader.string(*minmaxScan, "evidence", path);
    }
    if (const auto *addReduction = kernelTiming->getObject(
            "simd_straight_line_add_reduction_completion")) {
      const std::string path =
          "profile.kernel_timing.simd_straight_line_add_reduction_completion";
      auto &result = timing.routeCompletions.straightLineAddReduction;
      result.enabled = true;
      readIntegers(reader, *addReduction, path,
                   {{"minimum_rows", &result.minimumRows},
                    {"maximum_rows", &result.maximumRows},
                    {"minimum_extent", &result.minimumExtent},
                    {"maximum_extent", &result.maximumExtent}});
      readNumbers(reader, *addReduction, path,
                  {{"shared_completion_system_cycles",
                    &result.sharedCompletionSystemCycles}});
      result.evidence = reader.string(*addReduction, "evidence", path);
    }
    if (const auto *twoLoopReduction = kernelTiming->getObject(
            "simd_two_loop_single_reduction_completion")) {
      const std::string path =
          "profile.kernel_timing.simd_two_loop_single_reduction_completion";
      auto &result = timing.routeCompletions.twoLoopSingleReduction;
      result.enabled = true;
      readIntegers(
          reader, *twoLoopReduction, path,
          {{"minimum_source_programs", &result.minimumSourcePrograms},
           {"maximum_source_programs", &result.maximumSourcePrograms},
           {"minimum_runtime_iterations", &result.minimumRuntimeIterations},
           {"maximum_runtime_iterations", &result.maximumRuntimeIterations},
           {"required_reduction_extent", &result.requiredReductionExtent}});
      readNumbers(reader, *twoLoopReduction, path,
                  {{"first_iteration_system_cycles",
                    &result.firstIterationSystemCycles},
                   {"additional_iteration_system_cycles",
                    &result.additionalIterationSystemCycles}});
      result.evidence = reader.string(*twoLoopReduction, "evidence", path);
    }
    if (const auto *tinyMemory = kernelTiming->getObject(
            "tiny_straight_line_simt_memory_completion")) {
      const std::string path =
          "profile.kernel_timing.tiny_straight_line_simt_memory_completion";
      auto &result = timing.tinyStraightLineSimtMemoryCompletion;
      result.enabled = true;
      result.maximumElementsPerAccess =
          reader.integer(*tinyMemory, "maximum_elements_per_access", path);
      result.signedResidualSystemCycles =
          reader.number(*tinyMemory, "signed_residual_system_cycles", path);
      result.evidence = reader.string(*tinyMemory, "evidence", path);
    }
    if (const auto *nonAffine = kernelTiming->getObject(
            "simt_non_affine_rank1_memory_completion")) {
      const std::string path =
          "profile.kernel_timing.simt_non_affine_rank1_memory_completion";
      auto &result = timing.simtNonAffineRank1MemoryCompletion;
      result.enabled = true;
      result.maximumElementsPerAccess =
          reader.integer(*nonAffine, "maximum_elements_per_access", path);
      readSignedPositiveIntegerNumberCurve(
          *nonAffine, "residual_system_cycles_by_source_program_count", reader,
          path, result.residualBySourceProgramCount);
      result.evidence = reader.string(*nonAffine, "evidence", path);
    }
    timing.systemCounterFrequencyMHz = reader.number(
        *kernelTiming, "system_counter_frequency_mhz", "profile.kernel_timing");
    timing.evidence =
        reader.string(*kernelTiming, "evidence", "profile.kernel_timing");
    if (!timing.isValid())
      reader.setError("profile.kernel_timing contains invalid generic kernel "
                      "boundary parameters");
  }

  const auto *simd = reader.object(*root, "simd", "profile");
  if (simd) {
    auto &simdProfile = hardware.simd;
    int64_t vectorWidthBits = 0;
    if (simd->getString("vector_width_measurement")) {
      vectorWidthBits =
          static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
              *simd, "vector_width_bits", "vector_width_measurement", "bit",
              microbench, reader, "simd")));
    } else {
      vectorWidthBits = reader.integer(*simd, "vector_width_bits", "simd");
    }
    simdProfile.vectorWidthBits = std::max<int64_t>(1, vectorWidthBits);
    simdProfile.vectorWidth =
        std::max<int64_t>(1, simdProfile.vectorWidthBits / 32);
    simdProfile.issueWidth = simdProfile.vectorWidth;
    if (const auto *startup =
            reader.object(*simd, "startup_system_cycles", "simd"))
      simdProfile.setupCycles = resolveNumberOrMeasurement(
          *startup, "vector", "vector_measurement", "system_cycle", microbench,
          reader, "simd.startup_system_cycles");
    if (const auto *ops = reader.object(*simd, "ops", "simd")) {
      // The schema validates every entry.
      for (const auto &entry : *ops)
        simdProfile.operationRates[entry.first] = resolveOpProfile(
            *ops, entry.first,
            "throughput_vector_instructions_per_system_cycle",
            "vector_instruction/system_cycle", microbench, reader);
    }
    if (const auto *memory = reader.object(*simd, "memory", "simd")) {
      simdProfile.loadBytesPerCycle = resolveNumberOrMeasurement(
          *memory, "vector_mte2_bytes_per_system_cycle",
          "load_throughput_measurement", "byte/system_cycle", microbench,
          reader, "simd.memory");
      simdProfile.storeBytesPerCycle = resolveNumberOrMeasurement(
          *memory, "mte3_bytes_per_system_cycle",
          "store_throughput_measurement", "byte/system_cycle", microbench,
          reader, "simd.memory");
      if (memory->getString("load_startup_measurement"))
        simdProfile.loadOperationSetupCycles = resolveNumberOrMeasurement(
            *memory, "load_operation_startup_system_cycles",
            "load_startup_measurement", "system_cycle", microbench, reader,
            "simd.memory");
      else
        simdProfile.loadOperationSetupCycles = reader.optionalNumber(
            *memory, "load_operation_startup_system_cycles", 0.0);
      if (memory->getString("store_startup_measurement"))
        simdProfile.storeOperationSetupCycles = resolveNumberOrMeasurement(
            *memory, "store_operation_startup_system_cycles",
            "store_startup_measurement", "system_cycle", microbench, reader,
            "simd.memory");
      else
        simdProfile.storeOperationSetupCycles = reader.optionalNumber(
            *memory, "store_operation_startup_system_cycles", 0.0);
      if (const auto *group =
              memory->getObject("independent_loop_load_group")) {
        const std::string path = "simd.memory.independent_loop_load_group";
        simdProfile.independentLoadGroupKernelStartupCycles =
            resolveNumberOrMeasurement(*group, "kernel_startup_system_cycles",
                                       "kernel_startup_measurement",
                                       "system_cycle", microbench, reader,
                                       path);
        simdProfile.independentLoadGroupIterationStartupCycles =
            resolveNumberOrMeasurement(
                *group, "iteration_startup_system_cycles",
                "iteration_startup_measurement", "system_cycle", microbench,
                reader, path);
        simdProfile.independentLoadGroupBytesPerCycle =
            resolveNumberOrMeasurement(
                *group, "bytes_per_system_cycle", "throughput_measurement",
                "byte/system_cycle", microbench, reader, path);
        simdProfile.independentLoadGroupMinimumOperations =
            reader.integer(*group, "minimum_operations", path);
        simdProfile.independentLoadGroupMaximumOperations =
            reader.integer(*group, "maximum_operations", path);
        if (reader.string(*group, "evidence", path).empty())
          reader.setError(path + ".evidence must identify calibration data");
      }
      if (const auto *completion =
              memory->getObject("computed_address_scalarized_load")) {
        const std::string path = "simd.memory.computed_address_scalarized_load";
        simdProfile.computedAddressScalarizedLoadMinimumElements =
            reader.integer(*completion, "minimum_elements", path);
        simdProfile.computedAddressScalarizedLoadMaximumElements =
            reader.integer(*completion, "maximum_elements", path);
        simdProfile.computedAddressScalarizedLoadCyclesPerElement =
            reader.number(*completion, "system_cycles_per_element", path);
        simdProfile.computedAddressScalarizedLoadEvidence =
            reader.string(*completion, "evidence", path);
      }
      if (const auto *completion =
              memory->getObject("control_flow_address_scalarized_load")) {
        const std::string path =
            "simd.memory.control_flow_address_scalarized_load";
        readPositiveIntegerNumberCurve(
            *completion, "residual_system_cycles_by_elements", reader, path,
            simdProfile
                .controlFlowAddressScalarizedLoadResidualCyclesByElements);
        simdProfile.controlFlowAddressScalarizedLoadEvidence =
            reader.string(*completion, "evidence", path);
      }
      if (const auto *completion =
              memory->getObject("masked_load_store_read_modify_write")) {
        const std::string path =
            "simd.memory.masked_load_store_read_modify_write";
        readSignedPositiveIntegerNumberCurves(
            *completion, "residual_system_cycles_by_row_bytes_and_active_rows",
            reader, path,
            simdProfile
                .maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows);
        simdProfile.maskedLoadStoreRmwEvidence =
            reader.string(*completion, "evidence", path);
      }
      if (const auto *completion =
              memory->getObject("irregular_tile_completion")) {
        const std::string path = "simd.memory.irregular_tile_completion";
        simdProfile.irregularTileCompletionEnabled = true;
        readIntegers(
            reader, *completion, path,
            {{"physical_threshold_elements",
              &simdProfile.irregularTilePhysicalThresholdElements},
             {"row_group_size", &simdProfile.irregularTileRowGroupSize},
             {"small_row_maximum_rows",
              &simdProfile.irregularTileSmallRowMaximumRows},
             {"large_working_set_full_physical_minimum_bytes",
              &simdProfile
                   .irregularTileLargeWorkingSetFullPhysicalMinimumBytes}});
        readNumbers(
            reader, *completion, path,
            {{"physical_excess_system_cycles_per_element",
              &simdProfile.irregularTilePhysicalExcessCyclesPerElement},
             {"active_system_cycles_per_element",
              &simdProfile.irregularTileActiveCyclesPerElement},
             {"tail_system_cycles_per_inactive_element",
              &simdProfile.irregularTileTailCyclesPerInactiveElement},
             {"large_row_masked_floor_scale",
              &simdProfile.irregularTileLargeRowMaskedFloorScale},
             {"aligned_two_row_group_physical_materialization_scale",
              &simdProfile
                   .irregularTileAlignedTwoRowGroupPhysicalMaterializationScale},
             {"aligned_first_row_group_system_cycles_per_element",
              &simdProfile.irregularTileAlignedFirstRowGroupCyclesPerElement},
             {"aligned_additional_row_group_system_cycles_per_element",
              &simdProfile
                   .irregularTileAlignedAdditionalRowGroupCyclesPerElement}});
        simdProfile.irregularTileCompletionEvidence =
            reader.string(*completion, "evidence", path);
      }
      if (const auto *groups =
              memory->getObject("pipelined_strided_short_store_groups")) {
        for (const auto &entry : *groups) {
          int64_t rowStrideBytes = 0;
          const llvm::StringRef strideKey = entry.first;
          const std::string path =
              "simd.memory.pipelined_strided_short_store_groups." +
              entry.first.str();
          const auto *group = entry.second.getAsObject();
          if (strideKey.getAsInteger(10, rowStrideBytes) ||
              rowStrideBytes <= 0 || !group) {
            reader.setError(path + " requires a positive stride and object");
            continue;
          }
          PipelinedStridedStoreGroupCost cost;
          cost.kernelStartupCycles = resolveNumberOrMeasurement(
              *group, "kernel_startup_system_cycles",
              "kernel_startup_measurement", "system_cycle", microbench, reader,
              path);
          cost.firstStoreCycles = resolveNumberOrMeasurement(
              *group, "first_store_system_cycles", "first_store_measurement",
              "system_cycle", microbench, reader, path);
          cost.incrementalStoreCycles = resolveNumberOrMeasurement(
              *group, "incremental_store_system_cycles",
              "incremental_store_measurement", "system_cycle", microbench,
              reader, path);
          cost.extraLaneCycles = resolveNumberOrMeasurement(
              *group, "extra_lane_system_cycles", "extra_lane_measurement",
              "system_cycle/lane", microbench, reader, path);
          cost.minimumRows = reader.integer(*group, "minimum_rows", path);
          cost.maximumRows = reader.integer(*group, "maximum_rows", path);
          cost.minimumStores = reader.integer(*group, "minimum_stores", path);
          cost.maximumStores = reader.integer(*group, "maximum_stores", path);
          cost.evidence = reader.string(*group, "evidence", path);
          if (!cost.isValid())
            reader.setError(path + " contains an invalid store-group fit");
          else
            simdProfile
                .pipelinedStridedStoreGroupsByRowStrideBytes[rowStrideBytes] =
                std::move(cost);
        }
      }
      readMaskedStridedCopyCompletion(
          *memory, "masked_strided_load_copy_completion", "simd.memory", reader,
          simdProfile.maskedStridedLoadCopyCompletionByStrideBytes);
      readMaskedStridedCopyCompletion(
          *memory, "masked_strided_store_copy_completion", "simd.memory",
          reader, simdProfile.maskedStridedStoreCopyCompletionByStrideBytes);
    }
    if (const auto *dot = reader.object(*simd, "dot", "simd")) {
      simdProfile.dotSetupCycles = resolveNumberOrMeasurement(
          *dot, "startup_system_cycles", "startup_measurement", "system_cycle",
          microbench, reader, "simd.dot");
      simdProfile.dotFlopsPerCycle = resolveNumberOrMeasurement(
          *dot, "flops_per_system_cycle", "throughput_measurement",
          "flop/system_cycle", microbench, reader, "simd.dot");
      if (const auto *resource =
              dot->getObject("physical_tile_resource_formula")) {
        const std::string context = "simd.dot.physical_tile_resource_formula";
        const std::string resourceStatus =
            reader.string(*resource, "status", context);
        if (resourceStatus != "qualified_physical_resource_formula_v2")
          reader.setError(context + " is not a qualified formula");
        simdProfile.dotPhysicalResourceTarget =
            reader.string(*resource, "target", context);
        simdProfile.dotPhysicalResourceLoweringContract =
            reader.string(*resource, "lowering_contract", context);
        if (const auto *formula = resource->getObject("formula")) {
          const std::string formulaContext = context + ".formula";
          StageModeProfile::DotPhysicalResourceFormula value;
          value.tinyMacFloorCycles = reader.number(
              *formula, "tiny_mac_floor_system_cycles", formulaContext);
          value.cubeMacCyclesPerSqrtFlop = reader.number(
              *formula, "cube_mac_system_cycles_per_sqrt_flop", formulaContext);
          value.cubeMinimumFlops =
              reader.number(*formula, "cube_minimum_flops", formulaContext);
          value.mte1StartupCycles = reader.number(
              *formula, "mte1_startup_system_cycles", formulaContext);
          value.mte1BytesExponent =
              reader.number(*formula, "mte1_bytes_exponent", formulaContext);
          value.lhsMte1CyclesPerBytePower =
              reader.number(*formula, "lhs_mte1_system_cycles_per_byte_power",
                            formulaContext);
          value.rhsMte1CyclesPerBytePower =
              reader.number(*formula, "rhs_mte1_system_cycles_per_byte_power",
                            formulaContext);
          value.tileSize =
              reader.integer(*formula, "tile_size", formulaContext);
          value.narrowNReferenceTiles = reader.number(
              *formula, "narrow_n_reference_tiles", formulaContext);
          value.wideNReferenceTiles =
              reader.number(*formula, "wide_n_reference_tiles", formulaContext);
          value.narrowNOccupancyExponent = reader.number(
              *formula, "narrow_n_occupancy_exponent", formulaContext);
          value.wideNOccupancyExponent = reader.number(
              *formula, "wide_n_occupancy_exponent", formulaContext);
          value.evidence = reader.string(*formula, "evidence", formulaContext);
          if (!value.isValid())
            reader.setError(formulaContext + " is invalid");
          else
            simdProfile.dotPhysicalResourceFormula = std::move(value);
        }
        if (!simdProfile.dotPhysicalResourceFormula)
          reader.setError(context + " requires a valid formula");
      }
    }
    readStageResources(reader, *simd, "simd", microbench, simdProfile);
    const auto predicate = simdProfile.operationRates.lookup("predicate.cmp");
    simdProfile.predicateOperationsPerCycle =
        predicate.throughput / std::max(1.0, predicate.factor);
    if (const auto *reduction = simd->getObject("reduction")) {
      simdProfile.reductionOperationSetupCycles = resolveNumberOrMeasurement(
          *reduction, "operation_startup_system_cycles",
          "operation_startup_measurement", "system_cycle", microbench, reader,
          "simd.reduction");
      simdProfile.reductionLaneStepsPerCycle = resolveNumberOrMeasurement(
          *reduction, "lane_steps_per_system_cycle",
          "lane_step_throughput_measurement", "lane_step/system_cycle",
          microbench, reader, "simd.reduction");
      if (reduction->getString(
              "masked_accumulator_completion_setup_measurement") ||
          reduction->getNumber(
              "masked_accumulator_completion_setup_system_cycles"))
        simdProfile.maskedAccumulatorReductionCompletionSetupCycles =
            resolveNumberOrMeasurement(
                *reduction, "masked_accumulator_completion_setup_system_cycles",
                "masked_accumulator_completion_setup_measurement",
                "system_cycle", microbench, reader, "simd.reduction");
      if (const auto *scalarized =
              reduction->getObject("scalarized_multi_result")) {
        simdProfile.scalarizedMultiResultReductionMinimumExtent =
            reader.integer(*scalarized, "minimum_axis_extent",
                           "simd.reduction.scalarized_multi_result");
        simdProfile.scalarizedMultiResultReductionReferenceResults =
            reader.integer(*scalarized, "reference_result_count",
                           "simd.reduction.scalarized_multi_result");
        simdProfile.scalarizedMultiResultReductionSetupCycles =
            reader.number(*scalarized, "operation_startup_system_cycles",
                          "simd.reduction.scalarized_multi_result");
        simdProfile.scalarizedMultiResultReductionElementsPerCycle =
            reader.number(*scalarized, "logical_elements_per_system_cycle",
                          "simd.reduction.scalarized_multi_result");
        simdProfile.scalarizedMultiResultReductionEvidence = reader.string(
            *scalarized, "evidence", "simd.reduction.scalarized_multi_result");
      }
    }
    if (const auto *sort = simd->getObject("sort")) {
      simdProfile.sortMinimumExtent =
          reader.integer(*sort, "minimum_axis_extent", "simd.sort");
      simdProfile.sortMaximumExtent =
          reader.integer(*sort, "maximum_axis_extent", "simd.sort");
      simdProfile.sortElementBits =
          reader.integer(*sort, "element_bit_width", "simd.sort");
      simdProfile.sortSetupCycles =
          reader.number(*sort, "operation_startup_system_cycles", "simd.sort");
      simdProfile.sortRowsPerCycle =
          reader.number(*sort, "rows_per_system_cycle", "simd.sort");
      simdProfile.sortEvidence = reader.string(*sort, "evidence", "simd.sort");
    }
    if (simdProfile.reductionLaneStepsPerCycle <= 0.0)
      simdProfile.reductionLaneStepsPerCycle = simdProfile.vectorWidth;
    simdProfile.shuffleLanesPerCycle = simdProfile.vectorWidth;
  }

  const auto *simt = reader.object(*root, "simt", "profile");
  if (simt) {
    auto &simtProfile = hardware.simt;
    if (simt->getString("warp_size_measurement")) {
      simtProfile.issueWidth =
          static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
              *simt, "warp_size", "warp_size_measurement", "lane", microbench,
              reader, "simt")));
    } else {
      simtProfile.issueWidth = reader.integer(*simt, "warp_size", "simt");
    }
    simtProfile.vectorWidth = 1;
    simtProfile.vectorWidthBits = 1;
    if (const auto *setup =
            reader.object(*simt, "setup_system_cycles", "simt")) {
      simtProfile.setupCycles = resolveNumberOrMeasurement(
          *setup, "empty_launch", "empty_launch_measurement", "system_cycle",
          microbench, reader, "simt.setup_system_cycles");
      readSetupCycleCurve(reader, *setup, microbench, simtProfile);
    }
    if (const auto *ops = reader.object(*simt, "ops", "simt")) {
      // Keep SIMD and SIMT operation namespaces equally extensible.
      for (const auto &entry : *ops)
        simtProfile.operationRates[entry.first] = resolveOpProfile(
            *ops, entry.first, "throughput_scalar_ops_per_system_cycle",
            "scalar_op/system_cycle", microbench, reader);
    }
    if (const auto *dot = reader.object(*simt, "dot", "simt")) {
      simtProfile.dotSetupCycles = resolveNumberOrMeasurement(
          *dot, "startup_system_cycles", "startup_measurement", "system_cycle",
          microbench, reader, "simt.dot");
      simtProfile.dotFlopsPerCycle = resolveNumberOrMeasurement(
          *dot, "flops_per_system_cycle", "throughput_measurement",
          "flop/system_cycle", microbench, reader, "simt.dot");
      readPositiveIntegerNumberCurve(
          *dot, "flops_per_system_cycle_by_active_warp_count", reader,
          "simt.dot", simtProfile.dotFlopsPerCycleByWarpCount);
      const double genericCompletionScale =
          reader.optionalNumber(*dot, "generic_completion_scale", 1.0);
      const double partialCompositeResidualScale = reader.optionalNumber(
          *dot, "partial_composite_residual_scale", genericCompletionScale);
      simtProfile.dotPartialCompositeResidualRelativeScale =
          partialCompositeResidualScale / genericCompletionScale;
      simtProfile.dotSetupCycles *= genericCompletionScale;
      simtProfile.dotFlopsPerCycle /= genericCompletionScale;
      for (auto &[warpCount, rate] : simtProfile.dotFlopsPerCycleByWarpCount)
        rate /= genericCompletionScale;
    }
    const auto predicate = simtProfile.operationRates.lookup("predicate.cmp");
    simtProfile.predicateOperationsPerCycle =
        predicate.throughput / std::max(1.0, predicate.factor);
    if (const auto *shuffle = reader.object(*simt, "shuffle", "simt")) {
      simtProfile.shuffleLanesPerCycle =
          simtProfile.issueWidth *
          resolveNumberOrMeasurement(
              *shuffle, "warp_instructions_per_system_cycle",
              "throughput_measurement", "warp_instruction/system_cycle",
              microbench, reader, "simt.shuffle");
      std::map<int64_t, double> warpInstructionCurve;
      readPositiveIntegerMeasurementCurve(
          *shuffle, "throughput_measurements", "warp_instruction/system_cycle",
          microbench, reader, "simt.shuffle", warpInstructionCurve);
      for (const auto &[warpCount, rate] : warpInstructionCurve)
        simtProfile.shuffleLanesPerCycleByWarpCount[warpCount] =
            simtProfile.issueWidth * rate;
    }
    if (const auto *memory = reader.object(*simt, "memory", "simt")) {
      simtProfile.loadWarpInstructionsPerCycle = resolveNumberOrMeasurement(
          *memory, "load_warp_instructions_per_system_cycle",
          "load_throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.memory");
      simtProfile.storeWarpInstructionsPerCycle = resolveNumberOrMeasurement(
          *memory, "store_warp_instructions_per_system_cycle",
          "store_throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.memory");
      readPositiveIntegerMeasurementCurves(
          *memory, "warp_instruction/system_cycle", microbench, reader,
          "simt.memory",
          {{"load_throughput_measurements",
            &simtProfile.loadWarpInstructionsPerCycleByWarpCount},
           {"store_throughput_measurements",
            &simtProfile.storeWarpInstructionsPerCycleByWarpCount}});
      readMaskedStridedCopyCompletion(
          *memory, "masked_strided_load_copy_completion", "simt.memory", reader,
          simtProfile.maskedStridedLoadCopyCompletionByStrideBytes);
      readMaskedStridedCopyCompletion(
          *memory, "masked_strided_store_copy_completion", "simt.memory",
          reader, simtProfile.maskedStridedStoreCopyCompletionByStrideBytes);
    }
    readStageResources(reader, *simt, "simt", microbench, simtProfile);
    if (const auto *resources = simt->getObject("stage_resources")) {
      if (const auto *superblock = resources->getObject("superblock")) {
        hardware.superblockUsefulFactorLimit =
            reader.integer(*superblock, "useful_factor_limit", "superblock");
        hardware.superblockPersistentStatePressureFreeFactor = reader.integer(
            *superblock, "persistent_state_pressure_free_factor", "superblock");
        hardware.superblockPersistentStateBytesPerCycle = reader.number(
            *superblock, "persistent_state_bytes_per_system_cycle",
            "superblock");
      }
      if (const auto *handoff = resources->getObject("scope_handoff")) {
        hardware.scopeHandoff.fixedScopeCycles = resolveNumberOrMeasurement(
            *handoff, "fixed_scope_system_cycles", "fixed_scope_measurement",
            "system_cycle", microbench, reader, "scope_handoff");
        hardware.scopeHandoff.cvCrossCoreSynchronizationCycles =
            reader.optionalNumber(
                *handoff, "cv_cross_core_synchronization_system_cycles", 0.0);
        hardware.scopeHandoff.cvPartitionPayloadEfficiency =
            reader.optionalNumber(*handoff, "cv_partition_payload_efficiency",
                                  1.0);
        hardware.scopeHandoff.inputHandoffBytesPerCycle = reader.number(
            *handoff, "input_handoff_bytes_per_system_cycle", "scope_handoff");
        hardware.scopeHandoff.outputHandoffBytesPerCycle = reader.number(
            *handoff, "output_handoff_bytes_per_system_cycle", "scope_handoff");
      }
    }
  }

  if (reader.failed())
    return llvm::createStringError(
        std::errc::invalid_argument, "invalid SIMD/SIMT profile '%s': %s",
        path.c_str(), reader.getError().str().c_str());
  if (!microbench)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile must reference "
                                   "microbenchmark_profile");
  if (!hardware.isValid())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile contains invalid hardware rates");
  std::string canonicalProfile;
  llvm::raw_string_ostream canonicalStream(canonicalProfile);
  emitPythonCanonicalJSON(*parsed, canonicalStream);
  canonicalStream.flush();
  llvm::ArrayRef<uint8_t> byteArray(
      reinterpret_cast<const uint8_t *>(canonicalProfile.data()),
      canonicalProfile.size());
  auto hash = llvm::SHA256::hash(byteArray);
  profile.selectionContentSha256 =
      llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true);

  profile.contentSha256 = profile.selectionContentSha256;
  if (!profile.microbenchmarkContentSha256.empty()) {
    std::string combinedAssets =
        canonicalProfile +
        "\nshared_microbenchmark_sha256=" + profile.microbenchmarkContentSha256;
    llvm::ArrayRef<uint8_t> combinedBytes(
        reinterpret_cast<const uint8_t *>(combinedAssets.data()),
        combinedAssets.size());
    auto combinedHash = llvm::SHA256::hash(combinedBytes);
    profile.contentSha256 =
        llvm::toHex(llvm::ArrayRef<uint8_t>(combinedHash), true);
  }
  return profile;
}

struct ProfileFileStamp {
  uint64_t size = 0;
  llvm::sys::TimePoint<> modificationTime;

  bool operator==(const ProfileFileStamp &other) const {
    return size == other.size && modificationTime == other.modificationTime;
  }
};

static std::optional<ProfileFileStamp> profileFileStamp(llvm::StringRef path) {
  llvm::sys::fs::file_status status;
  if (llvm::sys::fs::status(path, status))
    return std::nullopt;
  return ProfileFileStamp{status.getSize(), status.getLastModificationTime()};
}

struct CachedCandidateProfile {
  ProfileFileStamp selectionStamp;
  std::string microbenchmarkPath;
  ProfileFileStamp microbenchmarkStamp;
  std::shared_ptr<const CandidateProfile> profile;
};

static llvm::Expected<std::shared_ptr<const CandidateProfile>>
loadCandidateProfile(llvm::StringRef requestedPath) {
  std::string path = requestedPath.empty() ? getDefaultSimdSimtProfilePath()
                                           : requestedPath.str();
  if (path.empty())
    return llvm::createStringError(std::errc::no_such_file_or_directory,
                                   "SIMD/SIMT profile path is empty; set "
                                   "TRITON_ASCEND_SIMD_SIMT_PROFILE");

  const auto selectionStamp = profileFileStamp(path);
  static std::mutex cacheMutex;
  static std::map<std::string, CachedCandidateProfile> cache;
  std::lock_guard<std::mutex> lock(cacheMutex);
  if (selectionStamp) {
    auto cached = cache.find(path);
    if (cached != cache.end() &&
        cached->second.selectionStamp == *selectionStamp) {
      const auto microbenchmarkStamp =
          profileFileStamp(cached->second.microbenchmarkPath);
      if (microbenchmarkStamp &&
          cached->second.microbenchmarkStamp == *microbenchmarkStamp)
        return cached->second.profile;
    }
  }

  std::string microbenchmarkPath;
  auto loaded = loadCandidateProfileUncached(path, &microbenchmarkPath);
  if (!loaded)
    return loaded.takeError();
  auto profile = std::make_shared<const CandidateProfile>(std::move(*loaded));
  const auto microbenchmarkStamp = profileFileStamp(microbenchmarkPath);
  if (selectionStamp && microbenchmarkStamp)
    cache[path] = CachedCandidateProfile{*selectionStamp, microbenchmarkPath,
                                         *microbenchmarkStamp, profile};
  return profile;
}

static double setRouteTotal(StageRoutePlan &plan, size_t owner,
                            double targetCycles) {
  targetCycles = std::max(0.0, targetCycles);
  const double adjustment = targetCycles - plan.totalCycles;
  plan.totalCycles = targetCycles;
  if (owner < plan.logicalStageCycles.size())
    plan.logicalStageCycles[owner] =
        std::max(0.0, plan.logicalStageCycles[owner] + adjustment);
  return adjustment;
}

static void addRouteResidual(StageRoutePlan &plan, size_t owner,
                             double residual) {
  setRouteTotal(plan, owner, plan.totalCycles + residual);
}

static llvm::Expected<StageCostModelSummary> evaluateStageModel(
    const SimdSimtFeatureSummary &features, const CandidateProfile &profile,
    unsigned numWarps, bool wholeKernelSuperblockMaterializable,
    bool scopeSuperblockMaterializable, int64_t logicalProgramCountHint,
    int64_t physicalCoreCountHint, int64_t physicalAiCoreCountHint,
    int64_t kernelArgumentCount, ModuleOp module,
    const SimtAnchorPlan *anchorPlan, const SimdSimtCostModelOptions &options,
    const LoopTripCountMap *loopTripCounts = nullptr) {
  StagePartitionerOptions partitionerOptions;
  partitionerOptions.tinyDotFlopsMax = profile.structural.tinyDotFlopsMax;
  const auto largestFactor = [](llvm::ArrayRef<int64_t> factors,
                                int64_t maximum) {
    int64_t largest = 1;
    for (int64_t factor : factors)
      if (factor >= 1 && factor <= maximum)
        largest = std::max(largest, factor);
    return largest;
  };
  const auto floorPowerOfTwo = [](int64_t value) {
    int64_t factor = 1;
    while (factor <= value / 2 && factor < 64)
      factor *= 2;
    return factor;
  };
  const int64_t wholeKernelMaximum =
      largestFactor(options.wholeKernelSuperblockFactors,
                    options.maximumWholeKernelSuperblockFactor);
  const int64_t scopeMaximum = largestFactor(
      options.scopeSuperblockFactors, options.maximumScopeSuperblockFactor);
  partitionerOptions.maximumSuperblockFactor = 1;
  if (wholeKernelSuperblockMaterializable || features.autoBlockifyV1Applied)
    partitionerOptions.maximumSuperblockFactor = wholeKernelMaximum;
  if (scopeSuperblockMaterializable)
    partitionerOptions.maximumSuperblockFactor =
        std::max(partitionerOptions.maximumSuperblockFactor, scopeMaximum);
  const int64_t warpLimitedMaximum = floorPowerOfTwo(
      64 / std::max<int64_t>(1, static_cast<int64_t>(numWarps)));
  partitionerOptions.maximumSuperblockFactor =
      std::min(partitionerOptions.maximumSuperblockFactor, warpLimitedMaximum);
  if (logicalProgramCountHint > 0) {
    const int64_t runtimeMaximum = floorPowerOfTwo(logicalProgramCountHint);
    partitionerOptions.maximumSuperblockFactor =
        std::min(partitionerOptions.maximumSuperblockFactor, runtimeMaximum);
  }
  partitionerOptions.maximumScopeSuperblockFactor =
      std::min(scopeMaximum, partitionerOptions.maximumSuperblockFactor);
  partitionerOptions.scopeSuperblockMaterializable =
      scopeSuperblockMaterializable;
  partitionerOptions.logicalProgramCountHint = logicalProgramCountHint;
  partitionerOptions.counterfactualLoopTripCounts = loopTripCounts;
  StagePartitioner partitioner;
  if (!module || !anchorPlan)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Stage model requires PreparedTTIR and "
                                   "its anchor plan");
  auto partition =
      partitioner.partition(module, *anchorPlan, partitionerOptions);
  if (!partition)
    return partition.takeError();
  if (llvm::Error error = SimtRegisterPressureAnalysis().analyze(
          module, *partition, std::max<unsigned>(1, numWarps)))
    return std::move(error);
  if (llvm::Error error = StageResourceLegalityAnalysis().applySimtUbCapacity(
          *partition, profile.hardware.simt.usableUbBytes))
    return std::move(error);
  for (LogicalStage &stage : partition->stages) {
    llvm::erase_if(stage.legalSimtFactors, [&](int64_t factor) {
      return !llvm::is_contained(options.wholeKernelSuperblockFactors, factor);
    });
    llvm::erase_if(stage.localSimtFactors, [&](int64_t factor) {
      return !llvm::is_contained(options.scopeSuperblockFactors, factor);
    });
    stage.simtLegal &=
        !stage.legalSimtFactors.empty() ||
        (stage.localSimtMaterializable && !stage.localSimtFactors.empty());
  }

  std::optional<size_t> scalarizedRank2IndirectOwner;
  const int64_t scalarizedRank2Maximum =
      profile.kernelTiming.scalarizedRank2IndirectLoadMaximumElements;
  if (scalarizedRank2Maximum > 0) {
    for (size_t stageIndex = 0; stageIndex < partition->stages.size();
         ++stageIndex) {
      const LogicalStage &stage = partition->stages[stageIndex];
      const bool matches = llvm::any_of(
          stage.workload.memoryAccesses, [&](const StageMemoryAccess &access) {
            if (access.store || !access.addressDependsOnLoadedIndex ||
                access.effectiveRank() != 2 ||
                access.provenContiguousElements() != 1 || access.shape.empty())
              return false;
            int64_t elements = 1;
            for (int64_t extent : access.shape) {
              if (extent <= 0 || elements > scalarizedRank2Maximum / extent)
                return false;
              elements *= extent;
            }
            return elements <= scalarizedRank2Maximum;
          });
      if (matches) {
        scalarizedRank2IndirectOwner = stageIndex;
        break;
      }
    }
  }

  std::optional<size_t> simtRank2IndirectOwner;
  int64_t simtRank2IndirectElements = 0;
  int64_t simtRank2IndirectElementBits = 0;
  const int64_t simtRank2Maximum =
      profile.kernelTiming.simtRank2IndirectLoadMaximumElements;
  if (simtRank2Maximum > 0) {
    for (size_t stageIndex = 0; stageIndex < partition->stages.size();
         ++stageIndex) {
      const LogicalStage &stage = partition->stages[stageIndex];
      for (const StageMemoryAccess &access : stage.workload.memoryAccesses) {
        if (access.store || !access.addressDependsOnLoadedIndex ||
            access.effectiveRank() != 2 ||
            access.provenContiguousElements() != 1 || access.shape.empty())
          continue;
        int64_t elements = 1;
        bool valid = true;
        for (int64_t extent : access.shape) {
          if (extent <= 0 || elements > simtRank2Maximum / extent) {
            valid = false;
            break;
          }
          elements *= extent;
        }
        if (!valid || elements > simtRank2Maximum)
          continue;
        if (elements >= simtRank2IndirectElements) {
          simtRank2IndirectOwner = stageIndex;
          simtRank2IndirectElements = elements;
          simtRank2IndirectElementBits = access.elementBits;
        }
      }
    }
  }

  std::optional<size_t> simdRank2StridedDirectOwner;
  int64_t simdRank2StridedDirectElements = 0;
  const auto &simdRank2Direct = profile.kernelTiming.simdRank2StridedDirectLoad;
  const auto &simtRank2Direct = profile.kernelTiming.simtRank2StridedDirectLoad;
  const int64_t simdRank2StridedDirectMaximum = simdRank2Direct.maximumElements;
  if (simdRank2StridedDirectMaximum > 0) {
    bool exactStructure = true;
    int64_t matchingLoads = 0;
    int64_t matchingStores = 0;
    for (size_t stageIndex = 0; stageIndex < partition->stages.size();
         ++stageIndex) {
      const LogicalStage &stage = partition->stages[stageIndex];
      if (stage.features.hasLoop || stage.features.hasDot ||
          stage.features.hasAtomicMemory || stage.features.hasPrefixScan ||
          stage.features.hasReduction || stage.iterationCount != 1)
        exactStructure = false;
      for (const StageMemoryAccess &access : stage.workload.memoryAccesses) {
        if (access.addressDependsOnLoadedIndex ||
            access.addressHasNonAffineTensorComputation ||
            access.addressHasTensorControlFlowSelection || access.masked ||
            access.shape.size() != 2 || access.effectiveRank() != 2 ||
            access.count != 1.0) {
          exactStructure = false;
          continue;
        }
        int64_t elements = 1;
        bool validShape = true;
        for (int64_t extent : access.shape) {
          if (extent <= 0 ||
              elements > simdRank2StridedDirectMaximum / extent) {
            validShape = false;
            break;
          }
          elements *= extent;
        }
        if (!validShape || elements < simdRank2Direct.minimumElements ||
            elements > simdRank2StridedDirectMaximum) {
          exactStructure = false;
          continue;
        }
        if (access.store) {
          if (!access.hasUnitInnerStride()) {
            exactStructure = false;
            continue;
          }
          ++matchingStores;
          continue;
        }
        const bool innerStrideIsNotProvenUnit =
            access.strides.size() == 2 &&
            (!access.strides[1] || *access.strides[1] != 1);
        if (access.provenContiguousElements() != 1 ||
            !innerStrideIsNotProvenUnit) {
          exactStructure = false;
          continue;
        }
        ++matchingLoads;
        simdRank2StridedDirectOwner = stageIndex;
        simdRank2StridedDirectElements = elements;
      }
    }
    if (!exactStructure || matchingLoads != 1 || matchingStores != 1) {
      simdRank2StridedDirectOwner.reset();
      simdRank2StridedDirectElements = 0;
    }
  }

  std::optional<size_t> rank2InterleavedPrefixScanOwner;
  int64_t rank2InterleavedPrefixScanExtent = 0;
  int64_t rank2InterleavedPrefixScanSequences = 0;
  const auto &rank2ScanFit =
      profile.kernelTiming.rank2InterleavedPrefixScanCompletion;
  if (rank2ScanFit.enabled) {
    const int64_t sourcePrograms = options.sourceLogicalProgramCountHint > 0
                                       ? options.sourceLogicalProgramCountHint
                                       : logicalProgramCountHint;
    bool exactStructure =
        sourcePrograms >= rank2ScanFit.minimumSourcePrograms &&
        sourcePrograms <= rank2ScanFit.maximumSourcePrograms;
    int64_t matchingScans = 0;
    for (size_t stageIndex = 0; stageIndex < partition->stages.size();
         ++stageIndex) {
      const LogicalStage &stage = partition->stages[stageIndex];
      if (stage.features.hasDot || stage.features.hasAtomicMemory ||
          (stage.features.hasReduction && !stage.features.hasPrefixScan) ||
          stage.features.hasLoopCarriedDataDependency ||
          stage.features.hasIndirectMemory)
        exactStructure = false;
      if (stage.workload.prefixScanCounts.empty())
        continue;
      if (stage.iterationCount != 1 ||
          stage.workload.prefixScanCounts.size() != 1) {
        exactStructure = false;
        continue;
      }
      const auto &[geometry, count] = *stage.workload.prefixScanCounts.begin();
      const auto &[extent, sequences, stride, results, reverse, combine] =
          geometry;
      if (!std::isfinite(count) || std::abs(count - 1.0) > 1.0e-12 || reverse ||
          combine != "add" || results != 1 || stride != sequences ||
          extent < rank2ScanFit.minimumExtent ||
          extent > rank2ScanFit.maximumExtent ||
          sequences < rank2ScanFit.minimumIndependentSequences ||
          sequences > rank2ScanFit.maximumIndependentSequences) {
        exactStructure = false;
        continue;
      }
      ++matchingScans;
      rank2InterleavedPrefixScanOwner = stageIndex;
      rank2InterleavedPrefixScanExtent = extent;
      rank2InterleavedPrefixScanSequences = sequences;
    }
    int64_t tensorAccesses = 0;
    int64_t matchingLoads = 0;
    int64_t matchingStores = 0;
    for (const LogicalStage &stage : partition->stages) {
      for (const StageMemoryAccess &access : stage.workload.memoryAccesses) {
        ++tensorAccesses;
        if (access.addressDependsOnLoadedIndex ||
            access.addressHasNonAffineTensorComputation ||
            access.addressHasTensorControlFlowSelection ||
            access.elementBits != 32 || access.count != 1.0 ||
            access.effectiveRank() != 2 || access.shape.size() != 2 ||
            access.strides.size() != 2 || !access.strides[0] ||
            !access.strides[1] ||
            access.shape[0] != rank2InterleavedPrefixScanExtent ||
            access.shape[1] != rank2InterleavedPrefixScanSequences ||
            *access.strides[0] != rank2InterleavedPrefixScanSequences ||
            *access.strides[1] != 1) {
          exactStructure = false;
          continue;
        }
        if (access.store)
          ++matchingStores;
        else
          ++matchingLoads;
      }
    }
    exactStructure &= matchingScans == 1 && tensorAccesses == 2 &&
                      matchingLoads == 1 && matchingStores == 1;
    if (!exactStructure) {
      rank2InterleavedPrefixScanOwner.reset();
      rank2InterleavedPrefixScanExtent = 0;
      rank2InterleavedPrefixScanSequences = 0;
    }
  }

  std::optional<size_t> simtStaticLoopDirectMemoryOwner;
  int64_t simtStaticLoopDirectMemoryIterations = 0;
  int64_t simtStaticLoopDirectMemoryLoadCount = 0;
  const auto &staticLoopFit =
      profile.kernelTiming.simtStaticLoopDirectMemoryCompletion;
  if (staticLoopFit.enabled) {
    bool exactStructure = true;
    int64_t matchingLoopStages = 0;
    int64_t matchingLoads = 0;
    int64_t matchingStores = 0;
    for (size_t stageIndex = 0; stageIndex < partition->stages.size();
         ++stageIndex) {
      const LogicalStage &stage = partition->stages[stageIndex];
      if (stage.features.hasDot || stage.features.hasAtomicMemory ||
          stage.features.hasPrefixScan || stage.features.hasReduction ||
          stage.features.hasLoopCarriedDataDependency ||
          stage.features.hasIndirectMemory ||
          stage.features.isGridStrideLoopShell ||
          stage.features.insideGridStrideLoopBody ||
          stage.features.unknownTripCountLoopCount > 0 ||
          stage.features.upperBoundLoopCount > 0 ||
          stage.features.synchronizationCount > 0 ||
          stage.workload.synchronizations > 0.0 ||
          stage.workload.sharedLocalLoadInstructions > 0.0 ||
          stage.workload.sharedLocalStoreInstructions > 0.0 ||
          stage.workload.privateStackLoadInstructions > 0.0 ||
          stage.workload.privateStackStoreInstructions > 0.0 ||
          !stage.workload.atomicWorkloads.empty())
        exactStructure = false;
      if (stage.workload.memoryAccesses.empty())
        continue;
      if (!stage.features.hasLoop || !stage.features.hasPointerInduction ||
          stage.iterationCount < staticLoopFit.minimumStaticIterations ||
          stage.iterationCount > staticLoopFit.maximumStaticIterations) {
        exactStructure = false;
        continue;
      }
      ++matchingLoopStages;
      simtStaticLoopDirectMemoryOwner = stageIndex;
      simtStaticLoopDirectMemoryIterations = stage.iterationCount;
      for (const StageMemoryAccess &access : stage.workload.memoryAccesses) {
        if (access.count != 1.0 || !access.masked ||
            access.elementBits != staticLoopFit.requiredElementBits ||
            access.effectiveRank() != 1 || access.shape.size() != 1 ||
            access.shape.front() != staticLoopFit.requiredContiguousElements ||
            !access.hasUnitInnerStride() ||
            access.provenContiguousElements() !=
                staticLoopFit.requiredContiguousElements ||
            access.addressDependsOnLoadedIndex ||
            access.addressHasNonAffineTensorComputation ||
            access.addressHasTensorControlFlowSelection) {
          exactStructure = false;
          continue;
        }
        if (access.store) {
          if (!access.maskedStoreHasMatchingLoadDependency)
            exactStructure = false;
          ++matchingStores;
        } else {
          ++matchingLoads;
        }
      }
    }
    exactStructure &= matchingLoopStages == 1 &&
                      matchingLoads >= staticLoopFit.minimumLoadCount &&
                      matchingLoads <= staticLoopFit.maximumLoadCount &&
                      matchingStores == staticLoopFit.requiredStoreCount;
    if (exactStructure) {
      simtStaticLoopDirectMemoryLoadCount = matchingLoads;
    } else {
      simtStaticLoopDirectMemoryOwner.reset();
      simtStaticLoopDirectMemoryIterations = 0;
      simtStaticLoopDirectMemoryLoadCount = 0;
    }
  }

  std::optional<size_t> subwarpIndirectCompositeOwner;
  int64_t subwarpIndirectCompositeElements = 0;
  bool hasProfiledTensorArithmetic = false;
  for (size_t stageIndex = 0; stageIndex < partition->stages.size();
       ++stageIndex) {
    const LogicalStage &stage = partition->stages[stageIndex];
    for (const auto &[shape, count] : stage.workload.indirectLoadCounts) {
      if (count <= 0.0 || shape.second <= 0 ||
          shape.second >
              profile.kernelTiming.subwarpIndirectCompositeMaximumElements)
        continue;
      if (!subwarpIndirectCompositeOwner)
        subwarpIndirectCompositeOwner = stageIndex;
      subwarpIndirectCompositeElements =
          std::max(subwarpIndirectCompositeElements, shape.second);
    }
    for (const auto &entry : stage.workload.operationCounts) {
      if (entry.first() == "generic.issue" || entry.first() == "predicate.cmp")
        continue;
      hasProfiledTensorArithmetic |=
          llvm::any_of(entry.second, [](const auto &shapeCount) {
            return shapeCount.second > 0.0;
          });
    }
  }
  if (!hasProfiledTensorArithmetic)
    subwarpIndirectCompositeOwner.reset();

  HardwareProfile hardwareProfile = profile.hardware;
  if (hardwareProfile.simd.dotPhysicalResourceTarget != options.actualTarget)
    hardwareProfile.simd.dotPhysicalResourceFormula.reset();
  hardwareProfile.baseSimtWarpCount = std::max<int64_t>(1, numWarps);
  hardwareProfile.maximumSimtEstimationFactor =
      partitionerOptions.maximumSuperblockFactor;
  hardwareProfile.cvAivSubblocks =
      options.cvAivSubblocks == 2 && physicalAiCoreCountHint > 0 &&
              physicalCoreCountHint / physicalAiCoreCountHint == 2 &&
              physicalCoreCountHint % physicalAiCoreCountHint == 0
          ? 2
          : 1;
  StageCostEvaluator evaluator;
  auto costTable = evaluator.evaluate(*partition, hardwareProfile,
                                      options.collectCalibrationEstimates);
  if (!costTable)
    return costTable.takeError();
  costTable->logicalProgramCountHint = logicalProgramCountHint;
  costTable->sourceLogicalProgramCountHint =
      options.sourceLogicalProgramCountHint;
  costTable->physicalCoreCountHint = physicalCoreCountHint;
  costTable->physicalAiCoreCountHint = physicalAiCoreCountHint;
  costTable->routeCompletions = profile.kernelTiming.routeCompletions;
  costTable->stageDependencyScheduleEnabled =
      profile.kernelTiming.stageDependencyCompletion.enabled;
  auto &multiResultScan = costTable->routeCompletions.multiResultPrefixScan;
  if (multiResultScan.enabled) {
    int64_t argumentCount = 0;
    module.walk([&](Operation *operation) {
      if (operation->getName().getStringRef() == "tt.func" &&
          operation->getNumRegions() != 0 && !operation->getRegion(0).empty())
        argumentCount = std::max<int64_t>(
            argumentCount, operation->getRegion(0).front().getNumArguments());
    });
    const int64_t physicalPrograms =
        logicalProgramCountHint > 0 && physicalCoreCountHint > 0
            ? std::min(logicalProgramCountHint, physicalCoreCountHint)
            : logicalProgramCountHint;
    multiResultScan.fixedBoundaryCycles =
        profile.kernelTiming.boundarySystemCycles(kAllSimd, 1,
                                                  physicalPrograms) +
        profile.kernelTiming.globalLoadBoundarySystemCycles(kAllSimd) +
        profile.kernelTiming.runtimeArgumentSetupSystemCycles(kAllSimd,
                                                              argumentCount);
  }
  auto routes = solveStageRoutes(*costTable, hardwareProfile.scopeHandoff);
  if (!routes)
    return routes.takeError();
  auto refreshBestAllSimt = [&]() {
    std::optional<StageRoutePlan> best;
    for (const auto &[factor, plan] : routes->allSimtFactorPlans) {
      if (plan.legal && (!best || plan.totalCycles < best->totalCycles))
        best = plan;
    }
    if (best)
      routes->allSimt = std::move(*best);
  };
  if (scalarizedRank2IndirectOwner && routes->allSimd.legal) {
    const double residual =
        profile.kernelTiming.scalarizedRank2IndirectLoadResidualSystemCycles(
            routes->allSimd.runtimePhysicalProgramCount);
    if (residual > 0.0)
      addRouteResidual(routes->allSimd, *scalarizedRank2IndirectOwner,
                       residual);
  }
  if (simdRank2StridedDirectOwner && simdRank2StridedDirectElements > 0 &&
      routes->allSimd.legal &&
      routes->allSimd.runtimePhysicalProgramCount == 1) {
    const double residual =
        simdRank2Direct.cycles(simdRank2StridedDirectElements);
    if (residual > 0.0) {
      addRouteResidual(routes->allSimd, *simdRank2StridedDirectOwner, residual);
      routes->allSimd.diagnostic<bool>(
          "simd_rank2_strided_direct_load_completion_applied") = true;
      routes->allSimd.diagnostic<int64_t>(
          "simd_rank2_strided_direct_load_elements") =
          simdRank2StridedDirectElements;
      routes->allSimd.diagnostic<double>(
          "simd_rank2_strided_direct_load_adjustment_system_cycles") = residual;
      routes->allSimd.diagnostic<std::string>(
          "simd_rank2_strided_direct_load_evidence") = simdRank2Direct.evidence;
    }
  }
  if (simdRank2StridedDirectOwner && simdRank2StridedDirectElements > 0) {
    const double target =
        simtRank2Direct.cycles(simdRank2StridedDirectElements);
    auto applyTarget = [&](StageRoutePlan &plan) {
      if (!plan.legal || plan.runtimePhysicalProgramCount != 1 ||
          target <= plan.totalCycles)
        return;
      const double adjustment =
          setRouteTotal(plan, *simdRank2StridedDirectOwner, target);
      plan.diagnostic<bool>(
          "simt_rank2_strided_direct_load_completion_applied") = true;
      plan.diagnostic<int64_t>("simt_rank2_strided_direct_load_elements") =
          simdRank2StridedDirectElements;
      plan.diagnostic<double>(
          "simt_rank2_strided_direct_load_target_system_cycles") = target;
      plan.diagnostic<double>(
          "simt_rank2_strided_direct_load_adjustment_system_cycles") =
          adjustment;
      plan.diagnostic<std::string>("simt_rank2_strided_direct_load_evidence") =
          simtRank2Direct.evidence;
    };
    for (auto &[factor, plan] : routes->allSimtFactorPlans)
      applyTarget(plan);
    refreshBestAllSimt();
  }
  if (simtStaticLoopDirectMemoryOwner &&
      simtStaticLoopDirectMemoryIterations > 0 &&
      simtStaticLoopDirectMemoryLoadCount > 0) {
    const double elapsedTarget =
        staticLoopFit.targetSystemCycles(simtStaticLoopDirectMemoryIterations,
                                         simtStaticLoopDirectMemoryLoadCount);
    auto applyTarget = [&](StageRoutePlan &plan) {
      if (!plan.legal || plan.routeSuperblockFactor <= 0 ||
          plan.routeSuperblockFactor > staticLoopFit.maximumSuperblockFactor)
        return;
      double fixedCycles = profile.kernelTiming.boundarySystemCycles(
          kAllSimtOnly, plan.routeSuperblockFactor,
          plan.runtimePhysicalProgramCount);
      fixedCycles +=
          profile.kernelTiming.globalLoadBoundarySystemCycles(kAllSimtOnly);
      fixedCycles += profile.kernelTiming.runtimeArgumentSetupSystemCycles(
          kAllSimtOnly, kernelArgumentCount);
      const double target = std::max(0.0, elapsedTarget - fixedCycles);
      const double adjustment =
          setRouteTotal(plan, *simtStaticLoopDirectMemoryOwner, target);
      plan.diagnostic<bool>(
          "simt_static_loop_direct_memory_completion_applied") = true;
      plan.diagnostic<int64_t>("simt_static_loop_direct_memory_iterations") =
          simtStaticLoopDirectMemoryIterations;
      plan.diagnostic<int64_t>("simt_static_loop_direct_memory_load_count") =
          simtStaticLoopDirectMemoryLoadCount;
      plan.diagnostic<double>(
          "simt_static_loop_direct_memory_elapsed_target_system_cycles") =
          elapsedTarget;
      plan.diagnostic<double>(
          "simt_static_loop_direct_memory_adjustment_system_cycles") =
          adjustment;
      plan.diagnostic<std::string>("simt_static_loop_direct_memory_evidence") =
          staticLoopFit.evidence;
    };
    for (auto &[factor, plan] : routes->allSimtFactorPlans)
      applyTarget(plan);
    refreshBestAllSimt();
  }
  if (rank2InterleavedPrefixScanOwner && rank2InterleavedPrefixScanExtent > 0 &&
      rank2InterleavedPrefixScanSequences > 0) {
    auto applyTarget = [&](StageRoutePlan &plan) {
      if (!plan.legal || plan.runtimeLoopIterationCount <= 0)
        return;
      const double semanticElements =
          static_cast<double>(rank2InterleavedPrefixScanExtent) *
          static_cast<double>(rank2InterleavedPrefixScanSequences) *
          static_cast<double>(plan.runtimeLoopIterationCount);
      double startup = rank2ScanFit.startupSystemCycles;
      double perElement = rank2ScanFit.perElementSystemCycles;
      if (plan.routeSuperblockFactor == 2) {
        startup = rank2ScanFit.factor2StartupSystemCycles;
        perElement = rank2ScanFit.factor2PerElementSystemCycles;
      } else if (plan.routeSuperblockFactor == 4) {
        startup = rank2ScanFit.factor4StartupSystemCycles;
        perElement = rank2ScanFit.factor4PerElementSystemCycles;
      }
      double target = startup + perElement * semanticElements;
      if (plan.routeSuperblockFactor == 2) {
        const double activeProgramDelta = static_cast<double>(
            plan.runtimePhysicalProgramCount -
            rank2ScanFit.factor2ReferencePhysicalProgramCount);
        target += rank2ScanFit.factor2PerActivePhysicalProgramSystemCycles *
                  activeProgramDelta;
        target =
            std::max(target, rank2ScanFit.factor2MinimumTargetSystemCycles);
      }
      target = std::max(0.0, target);
      if (target <= plan.totalCycles)
        return;
      const double adjustment =
          setRouteTotal(plan, *rank2InterleavedPrefixScanOwner, target);
      plan.diagnostic<bool>(
          "rank2_interleaved_prefix_scan_completion_applied") = true;
      plan.diagnostic<int64_t>("rank2_interleaved_prefix_scan_extent") =
          rank2InterleavedPrefixScanExtent;
      plan.diagnostic<int64_t>(
          "rank2_interleaved_prefix_scan_independent_sequences") =
          rank2InterleavedPrefixScanSequences;
      plan.diagnostic<double>(
          "rank2_interleaved_prefix_scan_target_system_cycles") = target;
      plan.diagnostic<double>(
          "rank2_interleaved_prefix_scan_adjustment_system_cycles") =
          adjustment;
      plan.diagnostic<std::string>("rank2_interleaved_prefix_scan_evidence") =
          rank2ScanFit.evidence;
    };
    applyTarget(routes->allSimd);
    applyTarget(routes->mixed);
    for (auto &[factor, plan] : routes->allSimtFactorPlans)
      applyTarget(plan);
    refreshBestAllSimt();
  }
  if (subwarpIndirectCompositeOwner && routes->allSimd.legal) {
    const double residual =
        profile.kernelTiming.subwarpIndirectCompositeResidualSystemCycles(
            subwarpIndirectCompositeElements);
    if (residual > 0.0)
      addRouteResidual(routes->allSimd, *subwarpIndirectCompositeOwner,
                       residual);
  }
  if (simtRank2IndirectOwner && simtRank2IndirectElements > 0) {
    const double residual =
        profile.kernelTiming.simtRank2IndirectLoadResidualSystemCycles(
            simtRank2IndirectElementBits, simtRank2IndirectElements);
    auto applyResidual = [&](StageRoutePlan &plan) {
      if (!plan.legal || plan.runtimePhysicalProgramCount != 1 ||
          std::abs(residual) <= 1.0e-12)
        return;
      plan.totalCycles = std::max(0.0, plan.totalCycles + residual);
      if (*simtRank2IndirectOwner < plan.logicalStageCycles.size())
        plan.logicalStageCycles[*simtRank2IndirectOwner] = std::max(
            0.0, plan.logicalStageCycles[*simtRank2IndirectOwner] + residual);
    };
    applyResidual(routes->allSimt);
    for (auto &[factor, plan] : routes->allSimtFactorPlans)
      applyResidual(plan);
  }
  return std::move(*routes);
}

static llvm::SmallVector<std::pair<double, SimdSimtCandidateKind>>
legalCandidates(const SimdSimtCandidateScores &scores, bool allSimdLegal,
                bool allSimtLegal, bool mixedLegal) {
  llvm::SmallVector<std::pair<double, SimdSimtCandidateKind>> candidates;
  if (allSimdLegal)
    candidates.push_back({scores.allSimd, SimdSimtCandidateKind::AllSIMD});
  if (allSimtLegal)
    candidates.push_back(
        {scores.allSimtOnly, SimdSimtCandidateKind::AllSIMTOnly});
  if (mixedLegal)
    candidates.push_back(
        {scores.mixedSimdSimt, SimdSimtCandidateKind::MixedSIMDSIMT});
  llvm::stable_sort(candidates, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  return candidates;
}

static SimdSimtCandidateKind chooseBest(const SimdSimtCandidateScores &scores,
                                        bool allSimdLegal, bool allSimtLegal,
                                        bool mixedLegal) {
  return legalCandidates(scores, allSimdLegal, allSimtLegal, mixedLegal)
      .front()
      .second;
}

/// PreparedTTIR may replace logical program IDs with scheduling IDs. Keep
/// workload expressions from the route-neutral module only if its algorithm
/// loops can be correlated with the prepared analysis view.
static std::string correlateLoopWorkloadExpressions(
    const std::vector<LoopWorkloadExpression> &source,
    const std::vector<LoopWorkloadExpression> &prepared) {
  if (source.size() != prepared.size())
    return "algorithm loop count differs between source and analysis modules";
  llvm::StringSet<> ids;
  auto algorithmParents = [](Operation *loop) {
    llvm::SmallVector<Operation *> parents;
    for (Operation *parent = loop->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto name = parent->getName().getStringRef();
      if ((name == "scf.for" || name == "scf.while") &&
          !parent->hasAttr("ta.auto_blockify_v1.loop"))
        parents.push_back(parent);
    }
    return parents;
  };
  for (size_t index = 0; index < source.size(); ++index) {
    const auto &before = source[index];
    const auto &after = prepared[index];
    if (!ids.insert(before.loopId).second)
      return "ambiguous algorithm loop ID: " + before.loopId;
    if (before.loopId != after.loopId ||
        before.operation->getName() != after.operation->getName())
      return "algorithm loop ID or kind differs: " + before.loopId;
    if (before.operation->getLoc() != after.operation->getLoc())
      return "algorithm loop source location differs: " + before.loopId;
    auto beforeParents = algorithmParents(before.operation);
    auto afterParents = algorithmParents(after.operation);
    if (beforeParents.size() != afterParents.size())
      return "algorithm loop nesting differs: " + before.loopId;
    for (size_t parentIndex = 0; parentIndex < beforeParents.size();
         ++parentIndex) {
      auto parent = llvm::find_if(source, [&](const auto &expression) {
        return expression.operation == beforeParents[parentIndex];
      });
      if (parent == source.end() ||
          prepared[std::distance(source.begin(), parent)].operation !=
              afterParents[parentIndex])
        return "algorithm loop parent identity differs: " + before.loopId;
    }
  }
  return {};
}

static bool hasGlobalLoad(const StageCostModelSummary &stageModel) {
  return llvm::any_of(stageModel.stages, [](const LogicalStageCost &stage) {
    return stage.workload.loadOperations > 0.0 ||
           stage.workload.loadBytes > 0.0 ||
           stage.workload.scalarLoadCount > 0.0;
  });
}

static std::optional<double> tinyStraightLineSimtMemoryAdjustment(
    const StageCostModelSummary &stageModel, const StageRoutePlan &plan,
    const KernelTimingProfile::TinyStraightLineSimtMemoryCompletion &fit) {
  if (!fit.enabled || !plan.legal || plan.routeSuperblockFactor != 1 ||
      plan.runtimePhysicalProgramCount != 1)
    return std::nullopt;

  int64_t loadAccesses = 0;
  int64_t storeAccesses = 0;
  for (const LogicalStageCost &stage : stageModel.stages) {
    if (stage.iterationCount != 1 || stage.features.hasLoop ||
        stage.features.hasAtomicMemory || stage.features.hasReduction ||
        stage.features.hasPrefixScan || stage.features.hasDot ||
        !stage.workload.atomicWorkloads.empty() ||
        !stage.workload.prefixScanCounts.empty() ||
        stage.workload.reductionOperations > 0.0 ||
        stage.workload.shuffleLaneSteps > 0.0 ||
        stage.workload.sharedLocalLoadInstructions > 0.0 ||
        stage.workload.sharedLocalStoreInstructions > 0.0 ||
        stage.workload.privateStackLoadInstructions > 0.0 ||
        stage.workload.privateStackStoreInstructions > 0.0)
      return std::nullopt;
    for (const StageMemoryAccess &access : stage.workload.memoryAccesses) {
      if (access.addressDependsOnLoadedIndex || access.count != 1.0 ||
          access.shape.empty())
        return std::nullopt;
      int64_t elements = 1;
      for (int64_t extent : access.shape) {
        if (extent <= 0 || elements > fit.maximumElementsPerAccess / extent)
          return std::nullopt;
        elements *= extent;
      }
      if (elements > fit.maximumElementsPerAccess)
        return std::nullopt;
      if (access.store)
        ++storeAccesses;
      else
        ++loadAccesses;
    }
  }
  if (loadAccesses != 1 || storeAccesses != 1)
    return std::nullopt;
  return fit.signedResidualSystemCycles;
}

static std::optional<double> simtNonAffineRank1MemoryAdjustment(
    const StageCostModelSummary &stageModel, const StageRoutePlan &plan,
    int64_t sourceProgramCount,
    const KernelTimingProfile::SimtNonAffineRank1MemoryCompletion &fit) {
  if (!fit.enabled || !plan.legal || sourceProgramCount <= 0 ||
      fit.residualBySourceProgramCount.empty() ||
      sourceProgramCount < fit.residualBySourceProgramCount.begin()->first ||
      sourceProgramCount > fit.residualBySourceProgramCount.rbegin()->first)
    return std::nullopt;

  int64_t nonAffineLoads = 0;
  int64_t stores = 0;
  for (const LogicalStageCost &stage : stageModel.stages) {
    const bool sourceLoop = stage.features.hasLoop &&
                            (stage.features.hasLoopCarriedDataDependency ||
                             stage.features.hasPointerInduction ||
                             stage.features.upperBoundLoopCount > 0 ||
                             stage.features.unknownTripCountLoopCount > 0 ||
                             stage.features.isGridStrideLoopShell ||
                             stage.features.insideGridStrideLoopBody);
    if (stage.iterationCount != 1 || sourceLoop ||
        stage.features.hasAtomicMemory || stage.features.hasReduction ||
        stage.features.hasPrefixScan || stage.features.hasDot ||
        !stage.workload.atomicWorkloads.empty() ||
        !stage.workload.prefixScanCounts.empty() ||
        stage.workload.reductionOperations > 0.0 ||
        stage.workload.shuffleLaneSteps > 0.0 ||
        stage.workload.sharedLocalLoadInstructions > 0.0 ||
        stage.workload.sharedLocalStoreInstructions > 0.0 ||
        stage.workload.privateStackLoadInstructions > 0.0 ||
        stage.workload.privateStackStoreInstructions > 0.0)
      return std::nullopt;
    for (const StageMemoryAccess &access : stage.workload.memoryAccesses) {
      if (access.count != 1.0 || access.effectiveRank() != 1 ||
          access.shape.size() != 1 || access.shape.front() <= 0 ||
          access.shape.front() > fit.maximumElementsPerAccess ||
          access.addressDependsOnLoadedIndex)
        return std::nullopt;
      if (access.store) {
        if (access.addressHasNonAffineTensorComputation)
          return std::nullopt;
        ++stores;
      } else {
        if (!access.addressHasNonAffineTensorComputation)
          return std::nullopt;
        ++nonAffineLoads;
      }
    }
  }
  if (nonAffineLoads != 1 || stores != 1)
    return std::nullopt;
  return KernelTimingProfile::interpolateBoundaryCurve(
      fit.residualBySourceProgramCount, sourceProgramCount, 0.0);
}

static double routeFixedSystemCycles(const CandidateProfile &profile,
                                     llvm::StringRef route,
                                     const StageRoutePlan &plan,
                                     int64_t kernelArgumentCount,
                                     bool routeHasGlobalLoad) {
  double cycles = profile.kernelTiming.boundarySystemCycles(
      route, plan.routeSuperblockFactor, plan.runtimePhysicalProgramCount);
  if (routeHasGlobalLoad)
    cycles += profile.kernelTiming.globalLoadBoundarySystemCycles(route);
  cycles += profile.kernelTiming.runtimeArgumentSetupSystemCycles(
      route, kernelArgumentCount);
  return cycles;
}

static SimdSimtCandidateScores
completeCandidateCosts(const SimdSimtCostReport &report,
                       const CandidateProfile &profile,
                       int64_t kernelArgumentCount) {
  SimdSimtCandidateScores scores = report.candidateCosts;
  const bool routeHasGlobalLoad = hasGlobalLoad(report.stageModel);
  scores.allSimd +=
      routeFixedSystemCycles(profile, kAllSimd, report.stageModel.allSimd,
                             kernelArgumentCount, routeHasGlobalLoad);
  scores.allSimtOnly +=
      routeFixedSystemCycles(profile, kAllSimtOnly, report.stageModel.allSimt,
                             kernelArgumentCount, routeHasGlobalLoad);
  if (auto adjustment = tinyStraightLineSimtMemoryAdjustment(
          report.stageModel, report.stageModel.allSimt,
          profile.kernelTiming.tinyStraightLineSimtMemoryCompletion))
    scores.allSimtOnly = std::max(0.0, scores.allSimtOnly + *adjustment);
  if (auto adjustment = simtNonAffineRank1MemoryAdjustment(
          report.stageModel, report.stageModel.allSimt,
          report.sourceLogicalProgramCountHint > 0
              ? report.sourceLogicalProgramCountHint
              : report.logicalProgramCountHint,
          profile.kernelTiming.simtNonAffineRank1MemoryCompletion))
    scores.allSimtOnly = std::max(0.0, scores.allSimtOnly + *adjustment);
  scores.mixedSimdSimt +=
      routeFixedSystemCycles(profile, kMixedSimdSimt, report.stageModel.mixed,
                             kernelArgumentCount, routeHasGlobalLoad);
  return scores;
}

static void
populateSelectedRouteElapsedPrediction(SimdSimtCostReport &report,
                                       const CandidateProfile &profile,
                                       int64_t kernelArgumentCount) {
  auto &prediction = report.selectedRouteElapsed;
  const StageRoutePlan *plan = nullptr;
  double baseScore = 0.0;
  switch (report.decision) {
  case SimdSimtCandidateKind::AllSIMD:
    plan = &report.stageModel.allSimd;
    baseScore = report.candidateCosts.allSimd;
    break;
  case SimdSimtCandidateKind::AllSIMTOnly:
    plan = &report.stageModel.allSimt;
    baseScore = report.candidateCosts.allSimtOnly;
    break;
  case SimdSimtCandidateKind::MixedSIMDSIMT:
    plan = &report.stageModel.mixed;
    baseScore = report.candidateCosts.mixedSimdSimt;
    break;
  }

  if (!plan || !profile.kernelTiming.isValid() || !std::isfinite(baseScore) ||
      baseScore < 0.0) {
    prediction.reason = "invalid_generic_resource_prediction_inputs";
    return;
  }

  prediction.applied = true;
  prediction.reason = "generic_stage_resource_formula";
  prediction.calibrationName = "generic_stage_resource_formula_v1";
  prediction.candidate = stringifySimdSimtCandidate(report.decision).str();
  prediction.formula = "(kernel_boundary_residual_system_cycles("
                       "candidate, superblock_factor, "
                       "runtime_physical_program_count) + "
                       "has_global_load * "
                       "global_load_boundary_residual_system_cycles(route) + "
                       "runtime_argument_setup_system_cycles(route, "
                       "kernel_argument_count) + "
                       "route_resource_system_cycles) / "
                       "system_counter_frequency_mhz";
  prediction.evidence = profile.kernelTiming.evidence;
  prediction.superblockFactor = plan->routeSuperblockFactor;
  prediction.baseWarpCount = profile.hardware.baseSimtWarpCount;
  prediction.runtimeSchedulingSlotCount = plan->runtimeSchedulingSlotCount;
  prediction.selectionScoreScale = 1.0;
  prediction.schedulingSlotScale = 0.0;
  const bool routeHasGlobalLoad = hasGlobalLoad(report.stageModel);
  prediction.runtimeKernelArgumentCount = kernelArgumentCount;
  prediction.runtimeArgumentSetupSystemCycles =
      profile.kernelTiming.runtimeArgumentSetupSystemCycles(
          prediction.candidate, kernelArgumentCount);
  prediction.fixedSystemCycles =
      routeFixedSystemCycles(profile, prediction.candidate, *plan,
                             kernelArgumentCount, routeHasGlobalLoad);
  if (report.decision == SimdSimtCandidateKind::AllSIMTOnly) {
    if (auto adjustment = tinyStraightLineSimtMemoryAdjustment(
            report.stageModel, *plan,
            profile.kernelTiming.tinyStraightLineSimtMemoryCompletion)) {
      prediction.reason = "tiny_straight_line_simt_memory_completion";
      prediction.calibrationName =
          "tiny_straight_line_simt_memory_completion_v1";
      prediction.formula =
          "(generic_route_elapsed_system_cycles + "
          "tiny_straight_line_simt_memory_signed_residual_system_cycles) / "
          "system_counter_frequency_mhz";
      prediction.evidence =
          profile.kernelTiming.tinyStraightLineSimtMemoryCompletion.evidence;
      prediction.fixedSystemCycles += *adjustment;
    }
    if (auto adjustment = simtNonAffineRank1MemoryAdjustment(
            report.stageModel, *plan,
            report.sourceLogicalProgramCountHint > 0
                ? report.sourceLogicalProgramCountHint
                : report.logicalProgramCountHint,
            profile.kernelTiming.simtNonAffineRank1MemoryCompletion)) {
      prediction.reason = "simt_non_affine_rank1_memory_completion";
      prediction.calibrationName = "simt_non_affine_rank1_memory_completion_v1";
      prediction.formula =
          "(generic_route_elapsed_system_cycles + "
          "interpolate(non_affine_rank1_residual, "
          "source_logical_program_count)) / system_counter_frequency_mhz";
      prediction.evidence =
          profile.kernelTiming.simtNonAffineRank1MemoryCompletion.evidence;
      prediction.fixedSystemCycles += *adjustment;
    }
  }
  // The profiler measures the complete route-independent residual.
  prediction.scalarPrologueSystemCycles = prediction.fixedSystemCycles / 2.0;
  prediction.scalarEpilogueSystemCycles = prediction.fixedSystemCycles / 2.0;
  prediction.systemCounterFrequencyMHz =
      profile.kernelTiming.systemCounterFrequencyMHz;
  prediction.baseSelectionScore = baseScore;
  prediction.predictedSystemCounterCycles =
      prediction.fixedSystemCycles + prediction.baseSelectionScore;
  prediction.predictedMicroseconds = prediction.predictedSystemCounterCycles /
                                     prediction.systemCounterFrequencyMHz;
}

/// Samples hypothetical uniform trip counts, never a runtime observation or
/// a feasible-range proof. The selected production route is not modified.
static llvm::json::Object analyzeLoopSensitivity(
    ModuleOp preparedModule, const SimtAnchorPlan *anchorPlan,
    const SimdSimtFeatureSummary &features, const CandidateProfile &profile,
    const SimdSimtCostModelOptions &options, const SimdSimtCostReport &baseline,
    const std::vector<LoopWorkloadExpression> &sourceExpressions,
    const std::vector<LoopWorkloadExpression> &preparedExpressions) {
  constexpr int64_t counts[] = {1, 2, 3, 4, 8, 16, 32};
  constexpr unsigned maximumAnalyzedLoops = 8;
  llvm::json::Object result;
  result["status"] = "sampled_only";
  result["global_invariance_proven"] = false;
  result["sample_feasibility_proven"] = false;
  result["binding_semantics"] = "counterfactual_uniform_per_invocation";
  result["maximum_analyzed_loops"] = maximumAnalyzedLoops;
  result["maximum_samples_per_loop"] = static_cast<int64_t>(std::size(counts));
  result["score_basis"] = "candidate_elapsed_costs";
  llvm::json::Array loops;
  unsigned analyzedLoops = 0;
  bool truncated = false;
  for (size_t index = 0; index < sourceExpressions.size(); ++index) {
    const auto &expression = sourceExpressions[index];
    llvm::json::Object loop;
    loop["loop_id"] = expression.loopId;
    if (expression.constantTripCount) {
      loop["status"] = "skipped_static";
      loops.push_back(std::move(loop));
      continue;
    }
    if (!expression.supported ||
        expression.operation->getName().getStringRef() != "scf.for") {
      loop["status"] = "skipped_unsupported";
      loop["reason"] = "requires a supported dynamic scf.for expression";
      loops.push_back(std::move(loop));
      continue;
    }
    if (analyzedLoops == maximumAnalyzedLoops) {
      loop["status"] = "skipped_limit";
      truncated = true;
      loops.push_back(std::move(loop));
      continue;
    }
    ++analyzedLoops;
    llvm::json::Array heldLoops;
    for (const auto &other : sourceExpressions) {
      if (other.loopId == expression.loopId || other.constantTripCount)
        continue;
      heldLoops.push_back(llvm::json::Object{
          {"loop_id", other.loopId}, {"source", "nominal_model_estimate"}});
    }
    loop["conditional_on_other_dynamic_loops"] = !heldLoops.empty();
    loop["held_dynamic_loops"] = std::move(heldLoops);
    llvm::json::Array samples;
    llvm::json::Array changes;
    llvm::SmallVector<std::pair<int64_t, double>> selectedElapsedSamples;
    std::optional<SimdSimtCandidateKind> previousWinner;
    int64_t previousCount = 0;
    bool failedSample = false;
    bool selectedRouteStable = true;
    for (int64_t count : counts) {
      LoopTripCountMap counterfactual;
      counterfactual.try_emplace(preparedExpressions[index].operation, count);
      auto model = evaluateStageModel(
          features, profile, options.numWarps,
          options.wholeKernelSuperblockMaterializable,
          options.scopeSuperblockMaterializable,
          options.logicalProgramCountHint, options.physicalVectorCoreCountHint,
          options.physicalAiCoreCountHint, baseline.kernelArgumentCount,
          preparedModule, anchorPlan, options, &counterfactual);
      llvm::json::Object sample;
      sample["trip_count"] = count;
      if (!model) {
        sample["status"] = "evaluation_failed";
        sample["reason"] = llvm::toString(model.takeError());
        failedSample = true;
        samples.push_back(std::move(sample));
        continue;
      }
      SimdSimtCostReport scenario;
      scenario.stageModel = std::move(*model);
      scenario.logicalProgramCountHint = baseline.logicalProgramCountHint;
      scenario.sourceLogicalProgramCountHint =
          baseline.sourceLogicalProgramCountHint;
      scenario.candidateCosts = {scenario.stageModel.allSimd.totalCycles,
                                 scenario.stageModel.allSimt.totalCycles,
                                 scenario.stageModel.mixed.totalCycles};
      scenario.candidateElapsedCosts = completeCandidateCosts(
          scenario, profile, baseline.kernelArgumentCount);
      const bool simdLegal = features.simtAnchors.kernelLowerability.allSimd &&
                             scenario.stageModel.allSimd.legal;
      const bool simtLegal =
          options.compileOn91095 && !features.hasExplicitScope &&
          features.simtAnchors.kernelLowerability.allSimtOnly &&
          scenario.stageModel.allSimt.legal;
      const bool mixedLegal = options.compileOn91095 &&
                              !features.hasExplicitScope &&
                              features.simtAnchors.count > 0 &&
                              features.simtAnchors.kernelLowerability.mixed &&
                              scenario.stageModel.mixed.legal;
      auto candidates = legalCandidates(scenario.candidateElapsedCosts,
                                        simdLegal, simtLegal, mixedLegal);
      sample["candidate_costs"] = scenario.candidateCosts.toJSON();
      sample["candidate_elapsed_costs"] =
          scenario.candidateElapsedCosts.toJSON();
      llvm::json::Array legal;
      for (const auto &candidate : candidates)
        legal.push_back(stringifySimdSimtCandidate(candidate.second));
      sample["legal_candidates"] = std::move(legal);
      sample["route_factors"] = llvm::json::Object{
          {kAllSimd, scenario.stageModel.allSimd.routeSuperblockFactor},
          {kAllSimtOnly, scenario.stageModel.allSimt.routeSuperblockFactor},
          {kMixedSimdSimt, scenario.stageModel.mixed.routeSuperblockFactor}};
      const StageRoutePlan &dependencyPlan = scenario.stageModel.allSimd;
      const std::string &dependencyStatus =
          dependencyPlan.diagnostic<std::string>(
              "stage_dependency_schedule_status");
      if (!dependencyStatus.empty()) {
        const double scheduledCore = dependencyPlan.diagnostic<double>(
            "stage_dependency_schedule_total_system_cycles");
        sample["all_simd_dependency_schedule"] = llvm::json::Object{
            {"status", dependencyStatus},
            {"loop_stage", dependencyPlan.diagnostic<std::string>(
                               "stage_dependency_schedule_loop_stage")},
            {"stage_count", dependencyPlan.diagnostic<int64_t>(
                                "stage_dependency_schedule_stage_count")},
            {"edge_count", dependencyPlan.diagnostic<int64_t>(
                               "stage_dependency_schedule_edge_count")},
            {"fill_system_cycles",
             dependencyPlan.diagnostic<double>(
                 "stage_dependency_schedule_fill_system_cycles")},
            {"initiation_interval_system_cycles",
             dependencyPlan.diagnostic<double>(
                 "stage_dependency_schedule_initiation_interval_system_"
                 "cycles")},
            {"scheduled_core_system_cycles", scheduledCore},
            {"route_system_cycles", scenario.candidateCosts.allSimd},
            {"elapsed_system_cycles", scenario.candidateElapsedCosts.allSimd},
            {"applied", dependencyPlan.diagnostic<bool>(
                            "stage_dependency_schedule_applied")}};
      }
      if (candidates.empty()) {
        sample["status"] = "no_legal_candidate";
        failedSample = true;
      } else {
        sample["status"] = "evaluated";
        const auto winner = candidates.front().second;
        sample["winner"] = stringifySimdSimtCandidate(winner);
        const int64_t selectedFactor =
            winner == SimdSimtCandidateKind::AllSIMD
                ? scenario.stageModel.allSimd.routeSuperblockFactor
            : winner == SimdSimtCandidateKind::AllSIMTOnly
                ? scenario.stageModel.allSimt.routeSuperblockFactor
                : scenario.stageModel.mixed.routeSuperblockFactor;
        if (winner != baseline.decision ||
            selectedFactor != baseline.selectedRouteElapsed.superblockFactor)
          selectedRouteStable = false;
        else
          selectedElapsedSamples.emplace_back(count, candidates.front().first);
        if (candidates.size() > 1)
          sample["runner_up_margin"] =
              candidates[1].first - candidates[0].first;
        if (previousWinner && *previousWinner != winner)
          changes.push_back(llvm::json::Object{
              {"previous_trip_count", previousCount},
              {"trip_count", count},
              {"previous_winner", stringifySimdSimtCandidate(*previousWinner)},
              {"winner", stringifySimdSimtCandidate(winner)}});
        previousWinner = winner;
        previousCount = count;
      }
      llvm::json::Array dotStages;
      for (const auto &stage : scenario.stageModel.stages)
        if (stage.workload.dotFlops > 0)
          dotStages.push_back(llvm::json::Object{
              {"stage_id", stage.id},
              {"iteration_count", stage.iterationCount},
              {"dot_flops_per_iteration", stage.workload.dotFlops}});
      sample["dot_stages"] = std::move(dotStages);
      if (options.includeDiagnosticStageDetailsInJSON) {
        const auto &stages = scenario.stageModel.stages;
        const auto &plan = scenario.stageModel.allSimd;
        if (plan.logicalStageCycles.size() != stages.size() ||
            plan.scopeHandoffCycles.size() != stages.size()) {
          sample["all_simd_stage_costs_status"] = "unavailable_size_mismatch";
        } else {
          llvm::json::Array stageCosts;
          for (size_t stageIndex = 0; stageIndex < stages.size(); ++stageIndex)
            stageCosts.push_back(llvm::json::Object{
                {"stage_id", stages[stageIndex].id},
                {"iteration_count", stages[stageIndex].iterationCount},
                {"logical_stage_system_cycles",
                 plan.logicalStageCycles[stageIndex]},
                {"scope_handoff_system_cycles",
                 plan.scopeHandoffCycles[stageIndex]}});
          sample["all_simd_stage_costs_status"] = "available";
          sample["all_simd_stage_costs"] = std::move(stageCosts);
        }
      }
      samples.push_back(std::move(sample));
    }
    loop["status"] = failedSample ? "incomplete_samples" : "sampled_only";
    loop["winner_stable_over_samples"] = !failedSample && changes.empty();
    loop["sampled_changes"] = std::move(changes);
    // This is a conditional report, not a runtime trip-count observation.
    // Only publish an affine expression when every sampled winner matches the
    // compiled route and the native complete-route scores agree on one line.
    if (baseline.selectedRouteElapsed.applied && !failedSample &&
        selectedRouteStable &&
        selectedElapsedSamples.size() == std::size(counts)) {
      const auto [firstCount, firstCost] = selectedElapsedSamples.front();
      const auto [lastCount, lastCost] = selectedElapsedSamples.back();
      const double perIteration =
          (lastCost - firstCost) / (lastCount - firstCount);
      const double offset = firstCost - firstCount * perIteration;
      const bool affine =
          llvm::all_of(selectedElapsedSamples, [&](const auto &point) {
            const double expected = offset + point.first * perIteration;
            return std::abs(point.second - expected) <=
                   1.0e-8 * std::max(1.0, std::abs(point.second));
          });
      if (affine && std::isfinite(offset) && std::isfinite(perIteration) &&
          perIteration >= 0.0) {
        llvm::json::Array verifiedCounts;
        for (const auto &[count, cost] : selectedElapsedSamples)
          verifiedCounts.push_back(count);
        loop["conditional_elapsed_prediction"] = llvm::json::Object{
            {"status", "affine_on_sampled_counts"},
            {"candidate", stringifySimdSimtCandidate(baseline.decision)},
            {"offset_system_cycles", offset},
            {"per_iteration_system_cycles", perIteration},
            {"system_counter_frequency_mhz",
             profile.kernelTiming.systemCounterFrequencyMHz},
            {"verified_trip_counts", std::move(verifiedCounts)},
            {"trip_count_source", "loop_workload_expressions"},
            {"runtime_trip_count_observed", false},
            {"selected_route_and_factor_unchanged", true}};
      }
    }
    loop["samples"] = std::move(samples);
    loops.push_back(std::move(loop));
  }
  result["truncated"] = truncated;
  result["analyzed_loop_count"] = analyzedLoops;
  result["loops"] = std::move(loops);
  return result;
}

static bool admitsStageDependencyCompletion(const SimdSimtCostReport &report,
                                            const CandidateProfile &profile) {
  const auto &completion = profile.kernelTiming.stageDependencyCompletion;
  const auto &model = report.stageModel;
  if (!completion.enabled || report.actualTarget != completion.target ||
      completion.loweringContract !=
          profile.hardware.simd.dotPhysicalResourceLoweringContract ||
      report.decision != SimdSimtCandidateKind::AllSIMD ||
      !report.features.autoBlockifyV1Applied ||
      !model.operationOwnershipComplete || !model.allSimd.legal ||
      model.allSimd.routeSuperblockFactor != 1 ||
      model.allSimd.runtimePhysicalProgramCount != 1 ||
      !model.allSimd.diagnostic<bool>("stage_dependency_schedule_applied") ||
      model.allSimd.diagnostic<std::string>(
          "stage_dependency_schedule_status") != "affine" ||
      report.loopWorkloadExpressions.size() != 1)
    return false;

  const auto *expression = report.loopWorkloadExpressions.front().getAsObject();
  return expression && expression->getBoolean("supported") == true;
}

static void
populateStageDependencyCompletion(SimdSimtCostReport &report,
                                  const CandidateProfile &profile,
                                  const SimdSimtCostModelOptions &options) {
  if (!admitsStageDependencyCompletion(report, profile))
    return;
  auto *loops = report.loopSensitivity.getArray("loops");
  if (!loops || loops->size() != 1)
    return;
  auto *loop = loops->front().getAsObject();
  auto *native =
      loop ? loop->getObject("conditional_elapsed_prediction") : nullptr;
  if (!native || native->getString("status") != "affine_on_sampled_counts" ||
      native->getString("candidate") != kAllSimd ||
      native->getBoolean("selected_route_and_factor_unchanged") != true)
    return;
  const auto offset = native->getNumber("offset_system_cycles");
  const auto perIteration = native->getNumber("per_iteration_system_cycles");
  const auto *verified = native->getArray("verified_trip_counts");
  if (!offset || !perIteration || !verified)
    return;
  for (int64_t count : {1, 2, 3, 4})
    if (!llvm::any_of(*verified, [&](const llvm::json::Value &value) {
          return value.getAsInteger() == count;
        }))
      return;
  const auto &completion = profile.kernelTiming.stageDependencyCompletion;
  const double correctedOffset = *offset -
                                 report.selectedRouteElapsed.fixedSystemCycles +
                                 completion.taskEnvelopeSystemCycles;
  const double nonrecurrentRouteBody =
      *offset - report.selectedRouteElapsed.fixedSystemCycles;
  llvm::json::Object prediction{
      {"prediction_scope", "complete_kernel_duration"},
      {"candidate", kAllSimd},
      {"offset_system_cycles", correctedOffset},
      {"per_iteration_system_cycles", *perIteration},
      {"system_counter_frequency_mhz",
       profile.kernelTiming.systemCounterFrequencyMHz},
      {"nonrecurrent_route_body_system_cycles", nonrecurrentRouteBody},
      {"independent_task_envelope_system_cycles",
       completion.taskEnvelopeSystemCycles},
      {"generic_fixed_system_cycles_replaced",
       report.selectedRouteElapsed.fixedSystemCycles},
      {"valid_trip_counts", llvm::json::Array{1, 2, 3, 4}},
      {"runtime_condition", "positive_trip_count_no_early_return"},
      {"dependency_schedule_applied", true},
      {"offline_schedule_calibration_certified", true},
      {"schedule_contract", completion.scheduleContract},
      {"full_kernel_validation_sha256", completion.fullKernelValidationSHA256},
      {"runtime_trip_count_observed", false},
      {"selected_route_unchanged", true},
      {"dynamic_cv_requested", options.dynamicCVPipelineRequested},
      {"evidence", completion.evidence}};
  if (options.dynamicCVPipelineRequested) {
    const double correctedPerIteration =
        *perIteration + completion.dynamicCVPerIterationDeltaSystemCycles;
    if (!std::isfinite(correctedPerIteration) || correctedPerIteration <= 0.0)
      return;
    prediction["status"] = "requested_on_requires_backend_confirmation";
    prediction["offset_system_cycles"] =
        correctedOffset + completion.dynamicCVFirstFillSystemCycles;
    prediction["per_iteration_system_cycles"] = correctedPerIteration;
    prediction["independent_dynamic_cv_first_fill_system_cycles"] =
        completion.dynamicCVFirstFillSystemCycles;
    prediction["independent_dynamic_cv_per_iteration_delta_system_cycles"] =
        completion.dynamicCVPerIterationDeltaSystemCycles;
    prediction["backend_effective_dynamic_cv_confirmed"] = false;
    prediction["formula"] =
        "(nonrecurrent_route_body + task_envelope + "
        "dynamic_cv_first_fill + q * (route_per_iteration + "
        "dynamic_cv_per_iteration_delta)) / system_counter_frequency";
  } else {
    prediction["status"] = "qualified_off_stage_dependency_prediction";
    prediction["formula"] = "(dependency_scheduled_route + task_envelope) / "
                            "system_counter_frequency";
  }
  (*loop)["stage_dependency_elapsed_prediction"] = std::move(prediction);
}

} // namespace

llvm::StringRef
mlir::ascend::stringifySimdSimtCandidate(SimdSimtCandidateKind candidate) {
  switch (candidate) {
  case SimdSimtCandidateKind::AllSIMD:
    return kAllSimd;
  case SimdSimtCandidateKind::AllSIMTOnly:
    return kAllSimtOnly;
  case SimdSimtCandidateKind::MixedSIMDSIMT:
    return kMixedSimdSimt;
  }
  llvm_unreachable("unknown SIMD/SIMT candidate");
}

llvm::json::Object SimdSimtCandidateScores::toJSON() const {
  return llvm::json::Object{{kAllSimd, allSimd},
                            {kAllSimtOnly, allSimtOnly},
                            {kMixedSimdSimt, mixedSimdSimt}};
}

llvm::json::Object
SimdSimtCostReport::SelectedRouteElapsedPrediction::toJSON() const {
  llvm::json::Object result;
  result["applied"] = applied;
  result["reason"] = reason;
  if (!applied)
    return result;
  result["calibration_name"] = calibrationName;
  result["candidate"] = candidate;
  result["superblock_factor"] = superblockFactor;
  result["base_warp_count"] = baseWarpCount;
  result["runtime_scheduling_slot_count"] = runtimeSchedulingSlotCount;
  if (minimumRuntimeSchedulingSlotCount > 0)
    result["minimum_runtime_scheduling_slot_count"] =
        minimumRuntimeSchedulingSlotCount;
  if (maximumRuntimeSchedulingSlotCount > 0)
    result["maximum_runtime_scheduling_slot_count"] =
        maximumRuntimeSchedulingSlotCount;
  if (minimumRuntimeLoopIterations > 0)
    result["minimum_runtime_loop_iterations"] = minimumRuntimeLoopIterations;
  if (maximumRuntimeLoopIterations > 0)
    result["maximum_runtime_loop_iterations"] = maximumRuntimeLoopIterations;
  if (maximumSelectionScore > 0.0) {
    result["minimum_selection_score"] = minimumSelectionScore;
    result["maximum_selection_score"] = maximumSelectionScore;
  }
  if (!dotUseSignature.empty())
    result["dot_use_signature"] = dotUseSignature;
  result["formula"] = formula;
  result["selection_score_scale"] = selectionScoreScale;
  result["scheduling_slot_scale"] = schedulingSlotScale;
  result["fixed_system_cycles"] = fixedSystemCycles;
  result["runtime_kernel_argument_count"] = runtimeKernelArgumentCount;
  result["runtime_argument_setup_system_cycles"] =
      runtimeArgumentSetupSystemCycles;
  result["scalar_prologue_system_cycles"] = scalarPrologueSystemCycles;
  result["scalar_epilogue_system_cycles"] = scalarEpilogueSystemCycles;
  result["system_counter_frequency_mhz"] = systemCounterFrequencyMHz;
  result["base_selection_score"] = baseSelectionScore;
  result["predicted_system_counter_cycles"] = predictedSystemCounterCycles;
  result["predicted_microseconds"] = predictedMicroseconds;
  result["evidence"] = evidence;
  return result;
}

static llvm::json::Object
toLowerabilityJSON(const CandidateLowerability &lowerability) {
  return llvm::json::Object{{kAllSimd, lowerability.allSimd},
                            {kAllSimtOnly, lowerability.allSimtOnly},
                            {kMixedSimdSimt, lowerability.mixed}};
}

static llvm::json::Object
toTriangularSolveFactsJSON(const TriangularSolveFacts &facts) {
  return llvm::json::Object{
      {"block_rows", facts.blockRows},
      {"block_columns", facts.blockColumns},
      {"accumulator_type", facts.accumulatorType},
      {"recurrence_start_row", facts.recurrenceStartRow},
      {"recurrence_loop_count", facts.recurrenceLoopCount},
      {"dense_dot_tail_ops", facts.denseDotTailOps},
      {"requires_cube_tail_partition", facts.requiresCubeTailPartition}};
}

llvm::json::Object SimtAnchorFeatureSummary::toJSON() const {
  llvm::json::Object result;
  result["count"] = count;
  llvm::json::Array triangularFacts;
  for (const TriangularSolveFacts &facts : triangularSolves)
    triangularFacts.push_back(toTriangularSolveFactsJSON(facts));
  result["triangular_solves"] = std::move(triangularFacts);
  result["kernel_lowerability"] = toLowerabilityJSON(kernelLowerability);
  return result;
}

llvm::json::Object SimdSimtFeatureSummary::toJSON() const {
  llvm::json::Object result;
  llvm::json::Object postTransform;
  postTransform["auto_blockify_v1_applied"] = autoBlockifyV1Applied;
  result["post_transform"] = std::move(postTransform);
  result["has_explicit_scope"] = hasExplicitScope;
  result["simt_anchors"] = simtAnchors.toJSON();
  return result;
}

llvm::json::Object SimdSimtCostReport::toJSON() const {
  llvm::json::Object result;
  result["schema_version"] = schemaVersion;
  result["model"] = model;
  result["profile_version"] = profileVersion;
  result["profile_target"] = profileTarget;
  result["actual_target"] = actualTarget;
  result["profile_content_sha256"] = profileContentSha256;
  result["selection_profile_content_sha256"] = selectionProfileContentSha256;
  llvm::json::Object sharedEvidence;
  sharedEvidence["profile_version"] = microbenchmarkProfileVersion;
  sharedEvidence["target"] = microbenchmarkProfileTarget;
  sharedEvidence["content_sha256"] = microbenchmarkProfileContentSha256;
  result["shared_microbenchmark_profile"] = std::move(sharedEvidence);
  result["unit"] = scoreUnit;
  result["kernel_argument_count"] = kernelArgumentCount;
  result["logical_program_count_hint"] = logicalProgramCountHint;
  result["source_logical_program_count_hint"] = sourceLogicalProgramCountHint;
  result["candidate_costs"] = candidateCosts.toJSON();
  result["candidate_elapsed_costs"] = candidateElapsedCosts.toJSON();
  result["selected_route_elapsed_prediction"] = selectedRouteElapsed.toJSON();
  result["decision_kind"] = stringifySimdSimtCandidate(decision);
  llvm::json::Array selectableCandidates;
  if (allSimdCandidateLegal)
    selectableCandidates.push_back(kAllSimd);
  if (allSimtOnlyCandidateLegal)
    selectableCandidates.push_back(kAllSimtOnly);
  if (mixedCandidateLegal)
    selectableCandidates.push_back(kMixedSimdSimt);
  result["selectable_candidates"] = std::move(selectableCandidates);
  llvm::json::Array unsupportedValues;
  for (const std::string &value : unsupported)
    unsupportedValues.push_back(value);
  result["unmodeled_cost_terms"] = std::move(unsupportedValues);
  result["stage_model"] =
      stageModel.toJSON(includeDiagnosticStageDetailsInJSON);
  result["stage_report_detail"] =
      includeDiagnosticStageDetailsInJSON ? "diagnostic" : "runtime_compact";
  result["loop_workload_expressions"] =
      llvm::json::Array(loopWorkloadExpressions);
  result["loop_workload_correlation"] =
      llvm::json::Object(loopWorkloadCorrelation);
  if (!loopSensitivity.empty())
    result["loop_sensitivity"] = llvm::json::Object(loopSensitivity);

  if (includeFeaturesInJSON)
    result["features"] = features.toJSON();
  return result;
}

void SimdSimtCostReport::printJSON(llvm::raw_ostream &os, bool pretty) const {
  llvm::json::Object object = toJSON();
  if (pretty)
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(object)));
  else
    os << llvm::json::Value(std::move(object));
}

std::string mlir::ascend::getDefaultSimdSimtProfilePath() {
  if (const char *environment = std::getenv("TRITON_ASCEND_SIMD_SIMT_PROFILE"))
    if (*environment)
      return environment;
#ifdef TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH
  return TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH;
#else
  return {};
#endif
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module, bool compileOn91095) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");
  SimtAnchorPlan anchorPlan = buildMixedSimtAnchorPlan(module, compileOn91095);
  return analyzeSimdSimtFeatures(module, anchorPlan);
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module,
                                      const SimtAnchorPlan &anchorPlan) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");

  SimdSimtFeatureSummary features;
  features.simtAnchors.count = llvm::count_if(
      anchorPlan.anchors,
      [](const SimtAnchorDescriptor &anchor) { return anchor.materializable; });
  features.simtAnchors.kernelLowerability = anchorPlan.kernelLowerability;

  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (anchor.triangularSolve)
      features.simtAnchors.triangularSolves.push_back(*anchor.triangularSolve);
  }

  module.walk([&](Operation *operation) {
    if (operation->hasAttr("ta.auto_blockify_v1") ||
        operation->hasAttr("ta.auto_blockify_v1.loop"))
      features.autoBlockifyV1Applied = true;
    features.hasExplicitScope |=
        operation->getName().getStringRef() == "scope.scope";
  });
  return features;
}

static llvm::Expected<SimdSimtCostReport>
estimateSimdSimtCandidatesImpl(const SimdSimtFeatureSummary &features,
                               const SimdSimtCostModelOptions &options,
                               ModuleOp module,
                               const SimtAnchorPlan *anchorPlan) {
  auto profileOrError = loadCandidateProfile(options.profilePath);
  if (!profileOrError)
    return profileOrError.takeError();
  const CandidateProfile &profile = **profileOrError;
  if (!options.actualTarget.empty() &&
      llvm::none_of(profile.compatibleTargets, [&](llvm::StringRef pattern) {
        return wildcardMatchInsensitive(pattern, options.actualTarget);
      }))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile target '%s' is not "
                                   "compatible with actual target '%s'",
                                   profile.hardware.target.c_str(),
                                   options.actualTarget.c_str());

  SimdSimtCostReport report;
  report.profileVersion = profile.hardware.profileVersion;
  report.profileTarget = profile.hardware.target;
  report.actualTarget = options.actualTarget;
  report.profileContentSha256 = profile.contentSha256;
  report.selectionProfileContentSha256 = profile.selectionContentSha256;
  report.microbenchmarkProfileVersion = profile.microbenchmarkProfileVersion;
  report.microbenchmarkProfileTarget = profile.microbenchmarkProfileTarget;
  report.microbenchmarkProfileContentSha256 =
      profile.microbenchmarkContentSha256;
  report.scoreUnit = profile.scoreUnit;
  report.logicalProgramCountHint = options.logicalProgramCountHint;
  report.sourceLogicalProgramCountHint =
      options.sourceLogicalProgramCountHint > 0
          ? options.sourceLogicalProgramCountHint
          : options.logicalProgramCountHint;
  module.walk([&](Operation *operation) {
    if (operation->getName().getStringRef() != "tt.func" ||
        operation->getNumRegions() == 0 || operation->getRegion(0).empty())
      return;
    report.kernelArgumentCount =
        std::max<int64_t>(report.kernelArgumentCount,
                          operation->getRegion(0).front().getNumArguments());
  });
  report.features = features;
  report.allSimdCandidateLegal =
      features.simtAnchors.kernelLowerability.allSimd;
  report.allSimtOnlyCandidateLegal =
      options.compileOn91095 && !features.hasExplicitScope &&
      features.simtAnchors.kernelLowerability.allSimtOnly;
  report.mixedCandidateLegal = !features.hasExplicitScope &&
                               options.compileOn91095 &&
                               features.simtAnchors.count > 0 &&
                               features.simtAnchors.kernelLowerability.mixed;
  report.includeFeaturesInJSON = options.includeFeaturesInJSON;
  report.includeDiagnosticStageDetailsInJSON =
      options.includeDiagnosticStageDetailsInJSON;

  const int64_t numWarps =
      std::max<int64_t>(1, static_cast<int64_t>(options.numWarps));
  auto stageModel = evaluateStageModel(
      features, profile, static_cast<unsigned>(numWarps),
      options.wholeKernelSuperblockMaterializable,
      options.scopeSuperblockMaterializable, options.logicalProgramCountHint,
      options.physicalVectorCoreCountHint, options.physicalAiCoreCountHint,
      report.kernelArgumentCount, module, anchorPlan, options);
  if (!stageModel)
    return stageModel.takeError();
  report.stageModel = std::move(*stageModel);
  report.candidateCosts.allSimd = report.stageModel.allSimd.totalCycles;
  report.candidateCosts.allSimtOnly = report.stageModel.allSimt.totalCycles;
  report.candidateCosts.mixedSimdSimt = report.stageModel.mixed.totalCycles;
  report.allSimdCandidateLegal &= report.stageModel.allSimd.legal;
  report.allSimtOnlyCandidateLegal &= report.stageModel.allSimt.legal;
  report.mixedCandidateLegal &= report.stageModel.mixed.legal;
  const unsigned legalCandidateCount =
      static_cast<unsigned>(report.allSimdCandidateLegal) +
      static_cast<unsigned>(report.allSimtOnlyCandidateLegal) +
      static_cast<unsigned>(report.mixedCandidateLegal);
  if (legalCandidateCount == 0)
    return llvm::createStringError(
        std::errc::not_supported,
        "Stage Route Model found no materializable candidate");
  report.candidateElapsedCosts =
      completeCandidateCosts(report, profile, report.kernelArgumentCount);
  report.decision =
      chooseBest(report.candidateElapsedCosts, report.allSimdCandidateLegal,
                 report.allSimtOnlyCandidateLegal, report.mixedCandidateLegal);
  if (options.backendForcedDecision) {
    const SimdSimtCandidateKind forced = *options.backendForcedDecision;
    const bool legal = forced == SimdSimtCandidateKind::AllSIMD
                           ? report.allSimdCandidateLegal
                       : forced == SimdSimtCandidateKind::AllSIMTOnly
                           ? report.allSimtOnlyCandidateLegal
                           : report.mixedCandidateLegal;
    if (!legal)
      return llvm::createStringError(
          std::errc::not_supported,
          "backend resource feedback selected an illegal SIMD/SIMT route: %s",
          stringifySimdSimtCandidate(forced).str().c_str());
    report.decision = forced;
  }
  populateSelectedRouteElapsedPrediction(report, profile,
                                         report.kernelArgumentCount);
  ModuleOp expressionModule =
      options.loopWorkloadModule ? options.loopWorkloadModule : module;
  auto expressions = analyzeLoopWorkloadExpressions(expressionModule);
  const bool separateSource = expressionModule != module;
  auto preparedExpressions =
      separateSource ? analyzeLoopWorkloadExpressions(module) : expressions;
  report.loopWorkloadCorrelation["source"] =
      separateSource ? "route_neutral_module" : "analysis_module";
  report.loopWorkloadCorrelation["status"] = "identity";
  std::string correlationFailure;
  if (separateSource) {
    correlationFailure =
        correlateLoopWorkloadExpressions(expressions, preparedExpressions);
    report.loopWorkloadCorrelation["status"] =
        correlationFailure.empty() ? "matched" : "mismatch";
    report.loopWorkloadCorrelation["checked_identity"] =
        "loop_id_kind_location_and_algorithm_nesting";
    if (!correlationFailure.empty()) {
      report.loopWorkloadCorrelation["reason"] = correlationFailure;
      report.unsupported.push_back("loop workload correlation: " +
                                   correlationFailure);
    }
  }
  for (const auto &expression : expressions)
    report.loopWorkloadExpressions.push_back(
        llvm::json::Object(expression.description));
  if (options.analyzeLoopSensitivity) {
    if (correlationFailure.empty())
      report.loopSensitivity =
          analyzeLoopSensitivity(module, anchorPlan, features, profile, options,
                                 report, expressions, preparedExpressions);
    else
      report.loopSensitivity =
          llvm::json::Object{{"status", "correlation_mismatch"},
                             {"global_invariance_proven", false},
                             {"sample_feasibility_proven", false},
                             {"analyzed_loop_count", 0},
                             {"reason", correlationFailure}};
  }
  populateStageDependencyCompletion(report, profile, options);
  return report;
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimdSimtCostModelOptions &options) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");
  SimtAnchorPlan anchorPlan =
      buildMixedSimtAnchorPlan(module, options.compileOn91095);
  return analyzeSimdSimtCandidates(module, anchorPlan, options);
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimtAnchorPlan &anchorPlan,
    const SimdSimtCostModelOptions &options) {
  auto features = analyzeSimdSimtFeatures(module, anchorPlan);
  if (!features)
    return features.takeError();
  return estimateSimdSimtCandidatesImpl(*features, options, module,
                                        &anchorPlan);
}
