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
  ScalarLoad,
  ScalarStore,
  IndexGeneration,
  PredicateMask,
  LoopPredicate,
  ContinuousTileMemory,
  ContinuousTileStore,
  ContinuousShortLoad,
  CachePolicyStore,
  IndirectScalarMemory,
  IndirectGatherMemory,
  AtomicMemory,
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

struct StageAtomicRate {
  double logicalElementsPerCycle = 0.0;
  double operationStartupCycles = 0.0;
  double resultDependencyCycles = 0.0;
  /// Relative penalty used when address collision is a runtime property.  It
  /// is a versioned selection-score policy, not a claim that absolute latency
  /// grows by this factor on every workload.
  double unknownContentionMultiplier = 1.0;

  bool isValid() const;
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
  /// Scalar pipe load/store throughput and latency.  Scalar loads/stores are
  /// executed by the scalar unit (MainScalar/AuxScalar/SIMT scalar path), not
  /// by the vector MTE pipes.
  double scalarLoadInstructionsPerCycle = 0.0;
  double scalarStoreInstructionsPerCycle = 0.0;
  double scalarLoadLatencyCycles = 0.0;
  double scalarStoreLatencyCycles = 0.0;
  /// Extra uncovered dependency latency for scalar load/store chains whose
  /// address is produced by another scalar load.  Restored from the legacy
  /// `scalar_ldst` model, but charged per **producer-side exposure** on top of
  /// the white-box/diff-line line cost: one producer feeding several consumer
  /// loads counts once, while a serial chain counts one per edge.
  double scalarIndirectDependencyLatencyCycles = 0.0;
  /// White-box CAModel load models.  When the corresponding `*FillCycles`
  /// is positive, ScalarLoad uses these first-line formulas instead of the
  /// provisional scalar-pipe throughput above.
  ///
  /// SIMD MainScalar same-64B-line load:
  ///   T = prep + fill + (K - 1) * (hit + issue)
  /// SIMT warp-uniform same-128B-line load:
  ///   T = prep + fill + (K - 1) * sameLineSerial
  /// Values are CAModel active cycles (first ISSUE to last RETIRE) for the
  /// the `load/scalar_o1` and `load/scalar_o4` probe shapes; see
  /// `data_provider/scalar_ldst_whitebox/README.md`.
  double mainScalarLoadPrepCycles = 0.0;
  double mainScalarLoadFillCycles = 0.0;
  double mainScalarLoadHitCycles = 0.0;
  double mainScalarLoadIssueCycles = 0.0;
  /// Structured diff-line MainScalar load model (CAModel round 6):
  ///   T = prep + fill
  ///       + max(0, U - outstanding) * perLineCost(U)
  ///       + (K - U) * hit + (K - 1) * issue
  /// where U is the number of distinct 64B lines and
  /// perLineCost(U) = low for U <= highThreshold, else high.
  double mainScalarLoadOutstandingLines = 0.0;
  double mainScalarLoadExtraLineLowCycles = 0.0;
  double mainScalarLoadExtraLineHighCycles = 0.0;
  double mainScalarLoadExtraLineHighThreshold = 0.0;
  double simtUniformLoadPrepCycles = 0.0;
  double simtUniformLoadFillCycles = 0.0;
  double simtUniformLoadSameLineSerialCycles = 0.0;
  /// Diff-line SIMT warp-uniform load model (CAModel round 6): distinct
  /// 128B lines can overlap, so only the LSU issue floor remains:
  ///   T = prep + fill + (K - 1) * diffLineIssue
  double simtUniformLoadDiffLineIssueCycles = 0.0;
  /// White-box CAModel store model for the Triton SIMD scalar-store lowering
  /// (MTE3 `MOV_SRC_TO_DST_ALIGNv2` UB -> OUT; this target does not use the
  /// CCE MainScalar `ST_XD_XN_IMM` GM path):
  ///   T = prep + fill + (K - 1) * serial
  /// `prep` covers scalar value -> UB staging / issue, `fill` is the first
  /// BIU write completion, `serial` is the extra per-store serialization when
  /// several MTE3 stores cannot overlap.
  double mte3StorePrepCycles = 0.0;
  double mte3StoreFillCycles = 0.0;
  double mte3StoreSerialCycles = 0.0;
  /// White-box SIMT warp-uniform store model.  The same-line branch models
  /// K >= 2 stores that land on one 128B line; a single scalar store has no
  /// repeated line access and uses the diff-line first-store preparation.
  double simtUniformStoreSameLineBaseCycles = 0.0;
  double simtUniformStoreSameLineSerialCycles = 0.0;
  double simtUniformStoreDiffLineBaseCycles = 0.0;
  double simtUniformStoreDiffLineIssueCycles = 0.0;
  /// Optional legacy scalar_ldst cheap-hash fit
  ///   cycles = a + b*warps + c*ops + d*warps*ops
  /// retained as a warm/runtime-throughput fallback for profiles that do
  /// not provide the structured white-box fields above.
  std::vector<double> scalarLoadCyclesFit;
  std::vector<double> scalarStoreCyclesFit;
  /// Loaded-index memory cannot use the continuous MTE/LSU throughput model.
  /// These rates operate on logical warp/transaction counts and include one
  /// uncovered dependency latency per Stage iteration.
  double indirectLoadTransactionsPerCycle = 0.0;
  double indirectStoreTransactionsPerCycle = 0.0;
  double indirectDependencyLatencyCycles = 0.0;
  llvm::StringMap<StageAtomicRate> atomicRates;
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
