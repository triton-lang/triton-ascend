//===- StageRouteCostModel.h - Logical-stage route model -------*- C++ -*-===//
//
// A kernel is represented as serial algorithm stages.  Every Stage is
// implemented entirely by SIMD or entirely by SIMT.  A mixed kernel is a
// route containing both modes; there is deliberately no mixed Stage.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::ascend {

enum class StageMode { SIMD, SIMT };
enum class StageKernelRouteKind { AllSIMD, AllSIMT, Mixed };
enum class StageScheduleKind {
  StraightLine,
  IndependentPipelined,
  LoopCarriedSerial,
  PartiallyDependent,
};

llvm::StringRef stringifyStageMode(StageMode mode);

struct StageImplementation {
  StageMode mode = StageMode::SIMD;
  /// SIMD always uses factor=1.  For a whole-kernel SIMT implementation this
  /// is the AutoBlockify V1 factor.  A local SIMT implementation identifies a
  /// mixed-kernel candidate whose selected Stage is materialized as a scope.
  /// The current backend still applies factor>1 through the surrounding V1
  /// kernel schedule; it is not an independently widened scope VF.
  int64_t superblockFactor = 1;
  bool localScope = false;

  bool isValid() const;
  llvm::json::Object toJSON() const;
};

/// Structural facts owned by one logical Stage.  Pointer induction is kept
/// separate from a true loop-carried data dependency because later address
/// lowering can remove it without serializing the Stage payload.
struct StageModelFeatures {
  bool hasLoop = false;
  bool hasLoopCarriedDataDependency = false;
  bool hasPointerInduction = false;
  bool hasContiguousMemory = false;
  bool hasIndirectMemory = false;
  bool hasReduction = false;
  bool hasPrefixScan = false;
  bool hasDot = false;
  bool hasConversionPack = false;
  /// True when the Stage is part of the logical-program body created by
  /// AutoBlockify V1.  A factor-F local SuperBlock executes SIMD Stages in
  /// this body once for every grouped logical program, while the selected
  /// local SIMT Stage consumes the factor through its SuperBlock model.
  bool replicatedByLocalSuperBlock = false;
  int64_t conditionalBranchCount = 0;
  int64_t divergentBranchCount = 0;
  int64_t loopBackedgeCount = 0;
  int64_t synchronizationCount = 0;
  /// Number of mutually independent loop-carried recurrence groups owned by
  /// this Stage.  Each group is serial internally, but SIMT may interleave
  /// different groups on independent warp groups.  Non-recurrence Stages and
  /// a single recurrence use one group.
  int64_t parallelRecurrenceGroupCount = 1;
  double activeLaneRatio = 1.0;

  bool isValid() const;
  bool permitsSimdRoofline() const;
  llvm::json::Object toJSON() const;
};

/// Mode-independent work owned exactly once by one Stage.  Values are
/// logical elements/bytes, not mode-specific instructions or cycles.
struct StageWorkload {
  llvm::StringMap<double> operationElements;
  double scalarOperations = 0.0;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  double predicateElements = 0.0;
  double shuffleLaneSteps = 0.0;
  /// Portion of shuffleLaneSteps contributed by tt.scan (prefix-scan class).
  /// The remainder is contributed by tt.reduce.  Only the scan portion
  /// consumes the prefix-scan dependency factor inside recurrence stages.
  double scanShuffleLaneSteps = 0.0;
  double dotFlops = 0.0;
  double issueElements = 0.0;
  double estimatedSpillTransactions = 0.0;
  bool paysKernelSetup = false;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// Resource costs for one iteration after raw Stage workload has been mapped
/// through the selected immutable hardware profile. Setup is paid once; all
/// other fields are per iteration.
struct StageResourceCycles {
  double setup = 0.0;
  double scalar = 0.0;
  double load = 0.0;
  double store = 0.0;
  double compute = 0.0;
  double predicate = 0.0;
  double shuffle = 0.0;
  /// Cycles for the tt.scan-contributed portion of shuffle at the ideal rate.
  double scanShuffle = 0.0;
  double dot = 0.0;
  double loopControl = 0.0;
  double branchControl = 0.0;
  double divergence = 0.0;
  double synchronization = 0.0;
  double spill = 0.0;
  double issue = 0.0;
  double criticalPath = 0.0;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

struct StageImplementationCost {
  StageImplementation implementation;
  double totalCycles = 0.0;
  StageResourceCycles resources;

  bool isValid() const;
  llvm::json::Object toJSON() const;
};

struct LogicalStageCost {
  std::string id;
  std::string model;
  StageScheduleKind schedule = StageScheduleKind::StraightLine;
  int64_t iterationCount = 1;
  StageModelFeatures features;
  StageWorkload workload;
  int64_t ownedOperationCount = 0;
  /// Unique source locations of the TTIR operations owned by this Stage.
  /// These are calibration provenance only: they let a debug-line-enabled
  /// CaModel artifact map binary PCs back to the immutable StagePartition
  /// without adding marker operations or attributes to production IR.
  std::vector<std::string> sourceLocations;
  int64_t liveInCount = 0;
  int64_t liveOutCount = 0;
  /// Static tensor footprint crossing the Stage boundary.  Counts alone are
  /// insufficient for a mixed route: returning tensor<8xf16> and
  /// tensor<8x1024xf16> are both one SSA value but have very different
  /// register/stack hand-off costs.
  int64_t liveInBytes = 0;
  int64_t liveOutBytes = 0;
  /// Number of primitive scope regions produced by the current immutable
  /// anchor plan when this Stage is selected as local SIMT.
  int64_t localSimtScopeCount = 0;
  int64_t scopeInputTensorBytes = 0;
  int64_t scopeOutputTensorBytes = 0;
  std::vector<unsigned> simtAnchorIndices;
  bool localSimtMaterializable = false;
  bool localSuperblockMaterializable = false;
  /// Factors legal for a whole-kernel pure-SIMT schedule.
  std::vector<int64_t> legalSimtFactors;
  /// Factors legal when this Stage alone is materialized as a local scope.
  std::vector<int64_t> localSimtFactors;
  std::vector<StageImplementationCost> implementations;

  llvm::json::Object toJSON() const;
};

struct StageCostTable {
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  int64_t logicalProgramCountHint = 0;
  int64_t physicalCoreCountHint = 0;
  std::vector<LogicalStageCost> stages;
};

struct StageTransitionCost {
  double simdToSimtCycles = 0.0;
  double simtToSimdCycles = 0.0;
  /// Local scope values cross the SIMD/SIMT register-file boundary through
  /// UB.  SIMD rates are aggregate vector-pipeline rates; SIMT rates are
  /// explicitly per active thread and are aggregated over one logical warp.
  double simdUbLoadBytesPerCycle = 1.0;
  double simdUbStoreBytesPerCycle = 1.0;
  double simtUbLoadBytesPerThreadPerCycle = 1.0;
  double simtUbStoreBytesPerThreadPerCycle = 1.0;
  int64_t simtWarpSize = 1;

  bool isValid() const;
  double get(StageMode from, StageMode to) const;
  llvm::json::Object toJSON() const;
};

struct StageRoutePlan {
  StageKernelRouteKind candidate = StageKernelRouteKind::AllSIMD;
  bool legal = false;
  std::vector<StageImplementation> implementations;
  std::vector<double> entryTransitionCycles;
  std::vector<double> logicalStageCycles;
  int64_t routeSuperblockFactor = 1;
  int64_t runtimePhysicalProgramCount = 0;
  int64_t runtimeWaveCount = 1;
  double totalCycles = 0.0;

  llvm::json::Object toJSON() const;
};

struct StageCostModelSummary {
  bool applied = false;
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  std::vector<LogicalStageCost> stages;
  StageTransitionCost transition;
  StageRoutePlan allSimd;
  StageRoutePlan allSimt;
  StageRoutePlan mixed;

  llvm::json::Object toJSON() const;
};

llvm::Expected<StageCostModelSummary>
solveStageRoutes(const StageCostTable &costTable,
                 const StageTransitionCost &transition);

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H
