//===- StageCostModels.h - Per-stage analytical models --------*- C++ -*-===//
//
// StagePartitioner, StageCostEvaluator, and KernelRouteSolver are separate
// components.  This file defines the immutable data passed between them and
// the mode-specific StageCostModel tree used by StageCostEvaluator.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H
#define ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H

#include "AscendModel/RouteModel/StageRouteCostModel.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::ascend {

enum class StageCostModelKind {
  AutoBlockifyDispatch,
  AutoBlockifyLoop,
  ScalarIssue,
  ScalarControl,
  ScalarMath,
  IndexGeneration,
  PredicateMask,
  LoopPredicate,
  ContinuousTileMemory,
  ContinuousTileStore,
  ContinuousShortLoad,
  CachePolicyStore,
  IndirectScalarMemory,
  IndirectGatherMemory,
  IndependentPipelinedLoop,
  LoopCarriedRecurrence,
  RowwiseReduction,
  PrefixScan,
  CubeRoofline,
  TinyCubeRoofline,
  ConversionPack,
};

llvm::StringRef stringifyStageCostModel(StageCostModelKind kind);

struct StageControlFlowRates {
  double loopBackedgeCycles = 0.0;
  double conditionalBranchCycles = 0.0;
  double divergentBranchPenaltyCycles = 0.0;
  double synchronizationCycles = 0.0;

  bool isFiniteAndNonNegative() const;
};

struct LogicalStage {
  std::string id;
  StageCostModelKind costModelKind = StageCostModelKind::ScalarIssue;
  StageScheduleKind scheduleKind = StageScheduleKind::StraightLine;
  int64_t iterationCount = 1;
  StageModelFeatures features;
  StageWorkload workload;
  /// Exact TTIR ownership when StagePartition was built from an operation
  /// graph.  Feature-summary fallback partitions deliberately leave this
  /// empty and must not be treated as materialization evidence.
  std::vector<Operation *> operations;
  /// SSA values crossing the Stage boundary.  These are derived from the
  /// same exact operation ownership as `operations`; they are the contract
  /// consumed by legality checks and the scope materializer.
  std::vector<Value> liveIns;
  std::vector<Value> liveOuts;
  int64_t liveInBytes = 0;
  int64_t liveOutBytes = 0;
  /// Exact tensor traffic at the local scope boundary.  Unlike Stage
  /// live-in/live-out, these fields mirror the SSA values captured by and
  /// returned from the materialized scope.scope regions.
  int64_t localSimtScopeCount = 0;
  int64_t scopeInputTensorBytes = 0;
  int64_t scopeOutputTensorBytes = 0;
  /// Indices into the immutable SimtAnchorPlan.  A mixed route may
  /// materialize only anchors owned by Stages that the solver selected as
  /// SIMT; consuming every materializable anchor would violate the route.
  std::vector<unsigned> simtAnchorIndices;
  /// Conjunction of all owned anchors' pure-SIMD lowering capabilities.
  bool allAnchorsSimdLowerable = true;
  /// Exact operations materialized inside the local SIMT scope.  This may be
  /// much smaller than `operations` for a hybrid Stage (for example one
  /// tt.scan inside a loop-carried Stage that also owns several tt.dot ops).
  std::vector<Operation *> localSimtOperations;
  /// Per-Stage-iteration workload of localSimtOperations.  Local mixed costing
  /// uses this for SIMT and charges the residual workload with the SIMD model.
  StageWorkload localSimtWorkload;
  bool simdLegal = false;
  bool simtLegal = false;
  /// True when this Stage has exact operation ownership/live-in/live-out and
  /// can therefore become a local SIMT scope inside a mixed kernel.
  bool localSimtMaterializable = false;
  /// True when the one selected local scope will be a direct operation of an
  /// AutoBlockify V1 loop body.  NPUIR's current scope-SuperBlock ABI requires
  /// this stronger condition for F2/F4; nested scopes remain legal at F1.
  bool localSuperblockMaterializable = false;
  std::vector<int64_t> legalSimtFactors;
  std::vector<int64_t> localSimtFactors;
};

struct StagePartition {
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::vector<LogicalStage> stages;
};

struct StageOperationRate {
  double throughput = 0.0;
  double factor = 1.0;
};

struct StageModeProfile {
  double setupCycles = 0.0;
  int64_t vectorWidth = 1;
  int64_t issueWidth = 1;
  llvm::StringMap<StageOperationRate> operationRates;
  double loadBytesPerCycle = 0.0;
  double storeBytesPerCycle = 0.0;
  double loadWarpInstructionsPerCycle = 0.0;
  double storeWarpInstructionsPerCycle = 0.0;
  double predicateOperationsPerCycle = 0.0;
  double shuffleLanesPerCycle = 0.0;
  double prefixScanDependencyFactor = 1.0;
  double dotSetupCycles = 0.0;
  double dotFlopsPerCycle = 0.0;
  double scalarOperationsPerCycle = 0.0;
  double issueOperationsPerCycle = 0.0;
  double spillTransactionsPerCycle = 0.0;
  /// Loaded-index memory cannot use the continuous MTE/LSU throughput model.
  /// These rates operate on logical warp/transaction counts and include one
  /// uncovered dependency latency per Stage iteration.
  double indirectLoadTransactionsPerCycle = 0.0;
  double indirectStoreTransactionsPerCycle = 0.0;
  double indirectDependencyLatencyCycles = 0.0;
  StageControlFlowRates controlFlow;

  bool isValid(StageMode mode) const;
};

struct HardwareProfile {
  std::string profileVersion;
  std::string target;
  /// Logical warp groups available to one SIMT program.  This is a compile
  /// option, not a hardware constant, and bounds cross-group interleaving in
  /// recurrence Stage models.
  int64_t logicalWarpGroupCount = 1;
  /// Long-lived recurrence state consumes finite register/stack bandwidth.
  /// The byte rate is shared by the SIMD recurrence-state term and the extra
  /// pressure created when a SIMT SuperBlock replicates that state; neither
  /// formula depends on a workload name.
  /// Largest factor that still gives proportional latency-hiding benefit.
  int64_t superblockUsefulFactorLimit = 1;
  /// Largest factor that may replicate loop-carried live state without an
  /// explicit persistent-state pressure charge.  This is intentionally
  /// independent from the latency-hiding limit: straight-line kernels may
  /// benefit through F4 while recurrence state becomes expensive above F2.
  int64_t superblockPersistentStatePressureFreeFactor = 1;
  double superblockPersistentStateBytesPerCycle = 1.0;
  StageModeProfile simd;
  StageModeProfile simt;
  StageTransitionCost transition;

  bool isValid() const;
};

class StageCostEvaluator {
public:
  llvm::Expected<StageCostTable> evaluate(const StagePartition &partition,
                                          const HardwareProfile &profile) const;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H
