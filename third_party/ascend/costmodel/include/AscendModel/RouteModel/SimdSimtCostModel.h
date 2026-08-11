//===- SimdSimtCostModel.h - Ascend SIMD/SIMT candidate model -*- C++ -*-===//
//
// This file exposes the target-profile-backed candidate cost model used to
// compare all-SIMD, all-SIMT, and mixed SIMD/SIMT execution for generic TTIR.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_SIMDSIMTCOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_SIMDSIMTCOSTMODEL_H

#include "AscendModel/RouteModel/SimtAnchorAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

enum class SimdSimtCandidateKind {
  AllSIMD,
  AllSIMTOnly,
  MixedSIMDSIMT,
};

llvm::StringRef stringifySimdSimtCandidate(SimdSimtCandidateKind candidate);

/// Resource summary for the exact non-overlapping TTIR operations that the
/// current materializer can wrap in local SIMT scopes.  It intentionally
/// contains no Operation pointers, so reports remain stable and serializable.
struct SimtAnchorFeatureSummary {
  /// All mechanism anchors recognized before target/lowering checks.
  int64_t recognizedCount = 0;
  /// Materializable anchors used by the mixed candidate.  `count` is retained
  /// as the serialized compatibility name for this value.
  int64_t count = 0;
  int64_t coveredOperationCount = 0;
  int64_t loadOps = 0;
  int64_t storeOps = 0;
  int64_t reduceOps = 0;
  int64_t scanOps = 0;
  int64_t gatherOps = 0;
  int64_t dotOps = 0;
  int64_t atomicOps = 0;
  int64_t histogramOps = 0;

  int64_t maxTensorNumel = 1;
  int64_t maxElementBits = 0;
  int64_t maskRankSum = 0;
  int64_t uniqueMaskValues = 0;
  int64_t uniqueMaskRankSum = 0;
  int64_t predicateElements = 0;
  /// Loop-weighted mask lanes on predicate-producing/consuming IR edges.
  int64_t predicateLaneEvaluations = 0;
  int64_t pointerTensorOps = 0;
  int64_t loadedIndexDependentMemoryOps = 0;
  int64_t laneDependentPointerOps = 0;
  int64_t maxReduceAxisExtent = 1;
  int64_t weightedReduceAxisElements = 0;
  /// Loop-weighted input lanes times log2(reduction/scan axis extent).
  int64_t shuffleLaneSteps = 0;
  int64_t staticLoopCount = 0;
  int64_t staticLoopTripCountSum = 0;
  int64_t modeledDynamicLoopCount = 0;
  int64_t modeledDynamicLoopTripCountSum = 0;
  bool hasControlFlow = false;

  llvm::StringMap<int64_t> weightedOps;
  llvm::StringMap<int64_t> opElements;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  int64_t loadWarpInstructions = 0;
  int64_t storeWarpInstructions = 0;
  int64_t dotFlops = 0;

  int64_t capturedTensorCount = 0;
  int64_t escapingTensorCount = 0;
  double capturedTensorBytes = 0.0;
  double escapingTensorBytes = 0.0;

  std::vector<std::string> mechanismKinds;
  std::vector<TensorAtomicFacts> tensorAtomics;
  std::vector<HistogramFacts> histograms;
  std::vector<PlainCumsumFacts> plainCumsums;
  CandidateLowerability kernelLowerability;

  llvm::json::Object toJSON() const;
};

/// Static, workload-name-independent properties extracted directly from a
/// generic TTIR ModuleOp.  Weighted fields include statically known scf.for
/// trip counts.
struct SimdSimtFeatureSummary {
  int64_t loadOps = 0;
  int64_t storeOps = 0;
  int64_t reduceOps = 0;
  int64_t scanOps = 0;
  int64_t gatherOps = 0;
  int64_t dotOps = 0;
  int64_t atomicOps = 0;
  int64_t histogramOps = 0;
  int64_t broadcastOps = 0;
  int64_t expandDimsOps = 0;
  int64_t splatOps = 0;
  int64_t addPtrOps = 0;

  int64_t arithOps = 0;
  int64_t mathOps = 0;
  int64_t addOps = 0;
  int64_t subOps = 0;
  int64_t mulOps = 0;
  int64_t divOps = 0;
  int64_t maxOps = 0;
  int64_t absOps = 0;
  int64_t expOps = 0;
  int64_t logOps = 0;
  int64_t cmpOps = 0;
  int64_t selectOps = 0;
  int64_t castOps = 0;
  int64_t clampOps = 0;
  int64_t scalarOps = 0;

  int64_t maxTensorRank = 0;
  int64_t maxTensorNumel = 1;
  int64_t maxElementBits = 0;
  int64_t maskTensorOps = 0;
  int64_t maskRankSum = 0;
  int64_t uniqueMaskValues = 0;
  int64_t uniqueMaskRankSum = 0;
  int64_t predicateElements = 0;
  /// Loop-weighted mask lanes on predicate-producing/consuming IR edges.
  int64_t predicateLaneEvaluations = 0;
  int64_t maskBroadcastOps = 0;
  int64_t pointerTensorOps = 0;
  int64_t pointerUnstructuredDims = 0;
  /// Real SSA provenance count.  Unlike laneDependentPointerOps, this field
  /// requires the address backward slice to reach a loaded/gathered index.
  int64_t loadedIndexDependentMemoryOps = 0;
  /// Legacy rank-based proxy retained for report compatibility.
  int64_t laneDependentPointerOps = 0;
  int64_t rowLocalReduceOps = 0;
  int64_t maxReduceAxisExtent = 1;
  int64_t weightedReduceAxisElements = 0;
  /// Loop-weighted input lanes times log2(reduction/scan axis extent).
  int64_t shuffleLaneSteps = 0;
  int64_t scalarLoadOps = 0;
  int64_t scalarStoreOps = 0;
  int64_t vectorPtrSplatOps = 0;
  int64_t vectorReduceToScalarOps = 0;
  bool rank1IndirectVectorReduce = false;

  llvm::StringMap<int64_t> weightedOps;
  llvm::StringMap<int64_t> opElements;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  int64_t loadWarpInstructions = 0;
  int64_t storeWarpInstructions = 0;
  int64_t dotFlops = 0;
  int64_t dotOutputElements = 0;
  std::vector<std::array<int64_t, 3>> dotMNK;
  int64_t staticLoopCount = 0;
  int64_t staticLoopTripCountSum = 0;
  int64_t staticLoopTripCountMax = 1;
  int64_t modeledDynamicLoopCount = 0;
  int64_t modeledDynamicLoopTripCountSum = 0;

  bool hasDot = false;
  bool hasGather = false;
  bool hasAtomic = false;
  bool hasHistogram = false;
  bool hasScan = false;
  bool hasExplicitScope = false;
  bool hasControlFlow = false;
  bool hasDynamicShape = false;
  bool hasUnknownTripCount = false;

  /// Pattern observations are diagnostic inputs to the model.  They do not
  /// create a mandatory online route: all three candidates stay selectable.
  std::vector<std::string> observedMixedKinds;

  SimtAnchorFeatureSummary simtAnchors;

  llvm::json::Object toJSON() const;
};

/// Applicability is a hardware/lowering statement, not a calibration-domain
/// statement.  It says whether TTIR contains a recognized SIMT mechanism and
/// whether the current target can materialize at least one corresponding
/// anchor.  Calibration coverage is reported separately.
struct SimtApplicabilityResult {
  bool mechanismDetected = false;
  bool targetSupported = false;
  bool materializable = false;
  int64_t recognizedAnchorCount = 0;
  int64_t materializableAnchorCount = 0;
  std::vector<std::string> mechanisms;
  std::vector<std::string> reasons;

  llvm::json::Object toJSON() const;
};

struct SimdSimtCandidateScores {
  double allSimd = 0.0;
  double allSimtOnly = 0.0;
  double mixedSimdSimt = 0.0;

  double get(SimdSimtCandidateKind candidate) const;
  llvm::json::Object toJSON() const;
};

/// Detailed values retained so the JSON report can explain every major term
/// in the versioned analytical and structural formulas.
struct SimdSimtCostBreakdown {
  llvm::StringMap<double> simdOpSystemCycles;
  llvm::StringMap<double> simtOpSystemCycles;
  llvm::StringMap<double> structuralComponents;

  double simdComputeCycles = 0.0;
  double simtComputeCycles = 0.0;
  double simdDotCycles = 0.0;
  double simtDotCycles = 0.0;
  double simdLoadCycles = 0.0;
  double simdStoreCycles = 0.0;
  double simdMemoryCycles = 0.0;
  double simtLoadCycles = 0.0;
  double simtStoreCycles = 0.0;
  double simtMemoryCycles = 0.0;
  double simtShuffleInstructions = 0.0;
  double simtShuffleCycles = 0.0;
  double simtPredicateInstructions = 0.0;
  double simtPredicateCycles = 0.0;
  double simdSetupCycles = 0.0;
  double simtSetupCycles = 0.0;
  double simdIssuePayloadCycles = 0.0;
  double simtIssuePayloadCycles = 0.0;
  double simdAnalyticalCycles = 0.0;
  double simtAnalyticalCycles = 0.0;
  double programIssueScale = 1.0;

  double mixedSimdRegularComputeCycles = 0.0;
  double mixedSimdRegularDotCycles = 0.0;
  double mixedSimdRegularMemoryCycles = 0.0;
  double mixedSimdRegularPayloadCycles = 0.0;
  double mixedSimtAnchorComputeCycles = 0.0;
  double mixedSimtAnchorDotCycles = 0.0;
  double mixedSimtAnchorMemoryCycles = 0.0;
  double mixedSimtAnchorShuffleCycles = 0.0;
  double mixedSimtAnchorPredicateCycles = 0.0;
  double mixedSimtAnchorPayloadCycles = 0.0;
  double mixedBoundaryCycles = 0.0;
  double mixedRemainingStructuralPenaltyRatio = 0.0;

  double irregularDensity = 0.0;
  double tinyDotUnderfill = 0.0;
  double structuralPenaltyRatio = 0.0;
  /// SIMD-only residual cost for structures omitted by its compute/memory
  /// roofline. It must not depend on the cost of another route candidate.
  double simdStructuralPenaltyCycles = 0.0;

  double mixedSimdFraction = 0.0;
  /// Conservative low-confidence setup fallback for a mixed plan.  The
  /// current source is a standalone empty-VF harness, not a measured
  /// directional SIMD/SIMT transition.
  double mixedSetupFallbackCycles = 0.0;
  double standaloneSimtSetupCycles = 0.0;
  double setupProxyDeltaCycles = 0.0;
  int64_t mixedSetupFallbackNumWarps = 0;
  std::string mixedCostSource;

  llvm::json::Object toJSON(const SimdSimtFeatureSummary &features) const;
};

struct SimdSimtCostModelOptions {
  /// Empty selects TRITON_ASCEND_SIMD_SIMT_PROFILE, then the source-tree
  /// profile compiled into AscendModelRouteModel.
  std::string profilePath;
  std::string actualTarget;
  unsigned numWarps = 32;
  /// Minimum relative advantage over the established all-SIMD baseline.
  /// The absolute floor is fixed at 64 selection-score cycles.
  double marginRatio = 0.10;
  bool includeFeaturesInJSON = true;
  /// Keep scoring outside the calibrated feature domain for offline/report
  /// diagnostics.  Production auto selection disables this so that coverage
  /// rejection happens before candidate-cost evaluation.
  bool scoreOutsideCalibrationCoverage = true;
  /// True only when the target backend can materialize the shared TTIR anchor
  /// plan.  Candidate costs remain reportable when false, but mixed is not
  /// eligible for selection.
  bool compileOn91095 = false;
};

struct SimdSimtCostReport {
  int64_t schemaVersion = 10;
  std::string model = "ascend_candidate_cost_v2_cpp";
  std::string profileVersion;
  std::string profileTarget;
  std::string actualTarget;
  std::string profileContentSha256;
  std::string selectionProfileContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkProfileContentSha256;
  std::string scoreUnit;
  std::string scoreScope = "per_program_ranking_proxy";

  /// False means coverage rejected the kernel before any candidate cost was
  /// evaluated.  In that state the numeric score members are placeholders and
  /// are serialized as null rather than as meaningful zeros.
  bool candidateCostsEvaluated = false;
  /// Resource/structural scores before the bounded Event calibration for the
  /// recognized feature domain is applied.
  SimdSimtCandidateScores uncalibratedCandidateCosts;
  SimdSimtCandidateScores candidateCosts;
  SimdSimtCandidateScores candidateRatiosToBest;
  SimdSimtCandidateScores eventRouteScoreMultipliers{1.0, 1.0, 1.0};
  bool eventRouteCalibrationApplied = false;
  bool eventAllSimtOnlyValidated = false;
  bool eventMixedSimdSimtValidated = false;
  std::string eventRouteCalibrationSource;
  std::string eventRouteCalibrationConfidence = "none";
  bool allSimdCandidateLegal = true;
  bool allSimtOnlyCandidateLegal = true;
  bool mixedCandidateLegal = false;
  SimdSimtCandidateKind decision = SimdSimtCandidateKind::AllSIMD;
  SimdSimtCandidateKind runnerUp = SimdSimtCandidateKind::AllSIMTOnly;
  double bestScore = 0.0;
  double runnerUpScore = 0.0;
  /// Alias of decisionAdvantage kept as the compact gate-facing gain field.
  double gainScore = 0.0;
  /// For non-SIMD decisions this is all_simd - best.  For an all-SIMD
  /// decision this is runner_up - best, so the baseline can win its own gate.
  double decisionAdvantage = 0.0;
  double requiredGainScore = 0.0;
  double marginRatio = 0.10;

  bool targetCompatible = true;
  bool selectionScoreValid = false;
  bool absoluteCostValid = false;
  std::string rankingConfidence = "none";
  std::string minimumConfidenceForDecision = "medium";
  std::string absoluteConfidence = "none";
  bool gatePassed = false;
  std::vector<std::string> gateReasons;
  std::vector<std::string> unsupported;
  bool calibrationCovered = false;
  std::string calibrationDomain = "out_of_calibration_domain";
  SimtApplicabilityResult applicability;

  SimdSimtFeatureSummary features;
  SimdSimtCostBreakdown breakdown;
  bool includeFeaturesInJSON = true;

  llvm::json::Object toJSON() const;
  void printJSON(llvm::raw_ostream &os, bool pretty = true) const;
};

/// Return the configured profile path.  The value is resolved in this
/// order: TRITON_ASCEND_SIMD_SIMT_PROFILE, compiled source-tree path.
std::string getDefaultSimdSimtProfilePath();

/// Analyze generic TTIR without depending on Triton C++ op classes.
llvm::Expected<SimdSimtFeatureSummary>
analyzeSimdSimtFeatures(mlir::ModuleOp module, bool compileOn91095 = true);

/// Analyze using a caller-owned immutable plan.  Selector uses this overload
/// so the exact operations charged by the mixed score are the operations later
/// marked for materialization.
llvm::Expected<SimdSimtFeatureSummary>
analyzeSimdSimtFeatures(mlir::ModuleOp module,
                        const SimtAnchorPlan &anchorPlan);

/// Run the versioned profile formula on an already materialized feature
/// summary.  By default this preserves full out-of-coverage scoring for
/// offline diagnostics; production callers can disable it in the options.
/// This overload is useful for golden feature tests and offline tools.
llvm::Expected<SimdSimtCostReport>
estimateSimdSimtCandidates(const SimdSimtFeatureSummary &features,
                           const SimdSimtCostModelOptions &options = {});

/// Analyze a ModuleOp and, when admitted for scoring, score all three
/// candidates in one call.
llvm::Expected<SimdSimtCostReport>
analyzeSimdSimtCandidates(mlir::ModuleOp module,
                          const SimdSimtCostModelOptions &options = {});

llvm::Expected<SimdSimtCostReport>
analyzeSimdSimtCandidates(mlir::ModuleOp module,
                          const SimtAnchorPlan &anchorPlan,
                          const SimdSimtCostModelOptions &options = {});

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ROUTEMODEL_SIMDSIMTCOSTMODEL_H
