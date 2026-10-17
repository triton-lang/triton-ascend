//===- SimdSimtCostModel.h - Ascend SIMD/SIMT candidate model -*- C++ -*-===//
//
// This file exposes the target-profile-backed candidate cost model used to
// compare all-SIMD, all-SIMT, and mixed SIMD/SIMT execution for generic TTIR.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_SIMDSIMTCOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_SIMDSIMTCOSTMODEL_H

#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/RouteModel/StageRouteCostModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
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

struct SimtAnchorFeatureSummary {
  int64_t count = 0;
  std::vector<TriangularSolveFacts> triangularSolves;
  CandidateLowerability kernelLowerability;

  llvm::json::Object toJSON() const;
};

/// Kernel-level facts needed before Stage partitioning.
struct SimdSimtFeatureSummary {
  /// Scheduling/layout facts read from the transformed TTIR consumed by this
  /// model.
  bool autoBlockifyV1Applied = false;
  bool hasExplicitScope = false;

  SimtAnchorFeatureSummary simtAnchors;

  llvm::json::Object toJSON() const;
};

struct SimdSimtCandidateScores {
  double allSimd = 0.0;
  double allSimtOnly = 0.0;
  double mixedSimdSimt = 0.0;

  llvm::json::Object toJSON() const;
};

struct SimdSimtCostModelOptions {
  /// Empty selects TRITON_ASCEND_SIMD_SIMT_PROFILE, then the source-tree
  /// profile compiled into AscendModelRouteModel.
  std::string profilePath;
  std::string actualTarget;
  unsigned numWarps = 32;
  unsigned numStages = 0;
  /// Unknown compile/launch facts must not admit a calibrated Cube domain.
  int dynamicCvPipeline = -1;
  int debug = -1;
  int disableFfts = -1;
  /// Route-neutral TTIR used to recognize whole-kernel Cube transactions.
  mlir::ModuleOp cubeCompositionModule;
  bool includeFeaturesInJSON = true;
  /// Full Stage implementation matrices are diagnostic-only.
  bool includeDiagnosticStageDetailsInJSON = true;
  bool collectCalibrationEstimates = true;
  /// True only when the target backend can materialize the shared TTIR anchor
  /// plan.
  bool compileOn91095 = false;
  /// True when backend integration can wrap a mixed local scope with the
  /// AutoBlockify V1 logical-program schedule for factors greater than one.
  bool scopeSuperblockMaterializable = false;
  /// True when backend integration can apply AutoBlockify V1 to a pure-SIMT
  /// kernel.  This is deliberately independent of local-scope batching.
  bool wholeKernelSuperblockMaterializable = false;
  /// Largest factor accepted by the whole-kernel backend contract.
  int64_t maximumWholeKernelSuperblockFactor = 64;
  /// Largest factor accepted by the outlined local-scope backend contract.
  int64_t maximumScopeSuperblockFactor = 4;
  /// Optional runtime launch count.
  int64_t logicalProgramCountHint = 0;
  /// Launch count before AutoBlockify V1 row coalescing.
  int64_t sourceLogicalProgramCountHint = 0;
  int64_t physicalVectorCoreCountHint = 0;
  int64_t physicalAiCoreCountHint = 0;
  /// Backend CV payload partition capability; one when binding is disabled.
  int64_t cvAivSubblocks = 1;
  /// Backend capability sets, intersected with structural/resource legality.
  /// Empty explicitly disables the corresponding SIMT implementations.
  std::vector<int64_t> wholeKernelSuperblockFactors = {1, 2, 4};
  std::vector<int64_t> scopeSuperblockFactors = {1, 2, 4};
  /// A backend resource pass can invalidate the pre-register-allocation winner.
  std::optional<SimdSimtCandidateKind> backendForcedDecision;
};

struct SimdSimtCostReport {
  int64_t schemaVersion = 21;
  std::string model = "ascend_stage_route_cost_v3_cpp";
  std::string profileVersion;
  std::string profileTarget;
  std::string actualTarget;
  std::string profileContentSha256;
  std::string selectionProfileContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkProfileContentSha256;
  std::string scoreUnit;
  int64_t kernelArgumentCount = 0;
  int64_t logicalProgramCountHint = 0;
  int64_t sourceLogicalProgramCountHint = 0;
  /// Compiler-derived Stage resources before the one-time route envelope.
  SimdSimtCandidateScores candidateCosts;
  SimdSimtCandidateScores candidateElapsedCosts;
  /// Absolute elapsed-time prediction for the selected route.
  struct SelectedRouteElapsedPrediction {
    bool applied = false;
    std::string reason = "no_structural_route_elapsed_calibration";
    std::string calibrationName;
    std::string candidate;
    std::string dotUseSignature;
    std::string formula;
    std::string evidence;
    int64_t superblockFactor = 1;
    int64_t baseWarpCount = 0;
    int64_t runtimeSchedulingSlotCount = 0;
    int64_t minimumRuntimeSchedulingSlotCount = 0;
    int64_t maximumRuntimeSchedulingSlotCount = 0;
    int64_t minimumRuntimeLoopIterations = 0;
    int64_t maximumRuntimeLoopIterations = 0;
    double minimumSelectionScore = 0.0;
    double maximumSelectionScore = 0.0;
    double selectionScoreScale = 0.0;
    double schedulingSlotScale = 0.0;
    double fixedSystemCycles = 0.0;
    int64_t runtimeKernelArgumentCount = 0;
    double runtimeArgumentSetupSystemCycles = 0.0;
    /// Accounting split of the measured route-independent kernel boundary.
    double scalarPrologueSystemCycles = 0.0;
    double scalarEpilogueSystemCycles = 0.0;
    double systemCounterFrequencyMHz = 0.0;
    double baseSelectionScore = 0.0;
    double predictedSystemCounterCycles = 0.0;
    double predictedMicroseconds = 0.0;

    llvm::json::Object toJSON() const;
  } selectedRouteElapsed;
  bool allSimdCandidateLegal = true;
  bool allSimtOnlyCandidateLegal = true;
  bool mixedCandidateLegal = false;
  SimdSimtCandidateKind decision = SimdSimtCandidateKind::AllSIMD;

  std::vector<std::string> unsupported;
  SimdSimtFeatureSummary features;
  StageCostModelSummary stageModel;
  /// Whole-completion calibration; Stage/body reference scores stay intact.
  llvm::json::Object cubeGridCost;
  bool includeFeaturesInJSON = true;
  bool includeDiagnosticStageDetailsInJSON = true;

  llvm::json::Object toJSON() const;
  void printJSON(llvm::raw_ostream &os, bool pretty = true) const;
};

/// Return the configured profile path.  The value is resolved in this
/// order: TRITON_ASCEND_SIMD_SIMT_PROFILE, compiled source-tree path.
std::string getDefaultSimdSimtProfilePath();

/// Analyze generic TTIR without depending on Triton C++ op classes.
llvm::Expected<SimdSimtFeatureSummary>
analyzeSimdSimtFeatures(mlir::ModuleOp module, bool compileOn91095 = true);

/// Analyze using a caller-owned immutable plan.
llvm::Expected<SimdSimtFeatureSummary>
analyzeSimdSimtFeatures(mlir::ModuleOp module,
                        const SimtAnchorPlan &anchorPlan);

/// Analyze a ModuleOp and score all three candidates in one call.
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
