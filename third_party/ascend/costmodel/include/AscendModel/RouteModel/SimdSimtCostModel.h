//===- SimdSimtCostModel.h - Ascend SIMD/SIMT candidate model -*- C++ -*-===//
//
// This file exposes the target-profile-backed candidate cost model used to
// compare all-SIMD, all-SIMT, and mixed SIMD/SIMT execution for generic TTIR.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_SIMDSIMTCOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_SIMDSIMTCOSTMODEL_H

#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/RouteModel/StageCostModels.h"
#include "AscendModel/RouteModel/StageRouteCostModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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
  int64_t count = 0;
  std::vector<TriangularSolveFacts> triangularSolves;
  CandidateLowerability kernelLowerability;

  llvm::json::Object toJSON() const;
};

/// Kernel-level facts needed before Stage partitioning.  Operation workload
/// belongs to StageFeatureAnalysis/StageWorkloadAnalysis and is intentionally
/// not duplicated here.
struct SimdSimtFeatureSummary {
  /// Scheduling/layout facts read from the transformed TTIR consumed by this
  /// model.  These make it explicit that layout merging and AutoBlockify V1
  /// ran before feature extraction rather than being guessed from source TTIR.
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
  bool includeFeaturesInJSON = true;
  /// True only when the target backend can materialize the shared TTIR anchor
  /// plan.  Candidate costs remain reportable when false, but mixed is not
  /// eligible for selection.
  bool compileOn91095 = false;
  /// True when backend integration can wrap a mixed local scope with the
  /// AutoBlockify V1 logical-program schedule for F2/F4.
  bool scopeSuperblockMaterializable = false;
  /// True when backend integration can apply AutoBlockify V1 to a pure-SIMT
  /// kernel.  This is deliberately independent of local-scope batching.
  bool wholeKernelSuperblockMaterializable = false;
  /// Optional runtime launch count. Zero means unknown; a positive value
  /// prevents the solver from pricing factors that cannot form one full
  /// logical-program group.
  int64_t logicalProgramCountHint = 0;
  int64_t physicalVectorCoreCountHint = 0;
};

struct SimdSimtCostReport {
  int64_t schemaVersion = 14;
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
  SimdSimtCandidateScores candidateCosts;
  bool allSimdCandidateLegal = true;
  bool allSimtOnlyCandidateLegal = true;
  bool mixedCandidateLegal = false;
  SimdSimtCandidateKind decision = SimdSimtCandidateKind::AllSIMD;

  std::vector<std::string> unsupported;
  SimdSimtFeatureSummary features;
  StageCostModelSummary stageModel;
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

/// Analyze a ModuleOp and score all three candidates in one call.
llvm::Expected<SimdSimtCostReport>
analyzeSimdSimtCandidates(mlir::ModuleOp module,
                          const SimdSimtCostModelOptions &options = {});

llvm::Expected<SimdSimtCostReport>
analyzeSimdSimtCandidates(mlir::ModuleOp module,
                          const SimtAnchorPlan &anchorPlan,
                          const SimdSimtCostModelOptions &options = {});

/// Partition `module` with exactly the options the candidate model uses for
/// scoring.  The selector calls this so a Mixed decision synthesized from
/// StageOwnedScope units (anchor-free Stages) wraps the same root ranges
/// that produced the scored route.
llvm::Expected<StagePartition>
partitionForSimdSimtSelection(mlir::ModuleOp module,
                              const SimtAnchorPlan &anchorPlan,
                              const SimdSimtCostModelOptions &options = {});

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ROUTEMODEL_SIMDSIMTCOSTMODEL_H
