//===- StageCostModels.h - Per-stage analytical models --------*- C++ -*-===//
//
// StagePartitioner, StageCostEvaluator, and KernelRouteSolver are separate
// components.
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
  /// Elementwise tensor/vector computation, not scalar dispatch. The mode
  /// selects SIMD vector throughput or SIMT lane throughput for its workload.
  VectorIssue,
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
  /// graph.
  std::vector<Operation *> operations;
  /// SSA values crossing the Stage boundary.
  std::vector<Value> liveIns;
  std::vector<Value> liveOuts;
  int64_t liveInBytes = 0;
  int64_t liveOutBytes = 0;
  std::optional<SimtRegisterPressure> wholeSimtRegisterPressure;
  /// Conservative pre-lowering UB footprint of one logical SIMT program.
  int64_t simtUbWorkingSetBytesPerProgram = 0;
  int64_t simtUsableUbBytes = 0;
  std::vector<int64_t> simtFactorsRejectedByUb;
  /// Exact tensor traffic at the local scope boundary.
  int64_t scopeInputTensorBytes = 0;
  int64_t scopeOutputTensorBytes = 0;
  /// Indices into the immutable SimtAnchorPlan.
  std::vector<unsigned> simtAnchorIndices;
  bool simdLegal = false;
  bool simtLegal = false;
  /// True when this Stage has exact operation ownership/live-in/live-out and
  /// can therefore become a local SIMT scope inside a mixed kernel.
  bool localSimtMaterializable = false;
  /// True when the one selected local scope will be a direct operation of an
  /// AutoBlockify V1 loop body.
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
  double startupCycles = 0.0;
  double freeWorkUnits = 0.0;
  /// Minimum logical element count for charging startupCycles.
  int64_t minimumStartupElements = 1;
  /// Optional one-time startup of this operation's dynamic grid-stride
  /// lowering.
  double gridStrideKernelStartupCycles = 0.0;
  bool includesLocalScopeHandoff = false;
  /// Aggregate throughput of one SIMT VF indexed by its launched warp count.
  std::map<int64_t, double> throughputByWarpCount;
};

/// Completion-inclusive cost of one straight-line prefix-scan helper, selected
/// by the TTIR combine class.
struct StraightLinePrefixScanCost {
  double startupCycles = 0.0;
  double elementsPerCycle = 0.0;
  double dependencyStepCycles = 0.0;
  std::map<int64_t, double> singleSequenceCompletionCyclesByExtent;
  std::string evidence;
  /// A non-unit scan axis lowers through a different backend helper.
  double nonUnitStrideStartupCycles = 0.0;
  double nonUnitStrideElementsPerCycle = 0.0;
  int64_t nonUnitStrideMinimumAxisExtent = 0;
  int64_t nonUnitStrideMaximumAxisExtent = 0;
  int64_t nonUnitStrideMinimumAxisStride = 0;
  int64_t nonUnitStrideMaximumAxisStride = 0;
  int64_t nonUnitStrideMinimumSemanticElements = 0;
  int64_t nonUnitStrideMaximumSemanticElements = 0;
  std::string nonUnitStrideEvidence;

  bool isValid() const;
};

/// Offline fit from TTIR-visible source facts to one generated SIMT vector
/// stream.
struct SimtSourceFit {
  bool enabled = false;
  double intercept = 0.0;
  double directInstructionWeight = 0.0;
  double segmentWeight = 0.0;
  double strideRatioWeight = 0.0;
  double maskedWeight = 0.0;
  double peakLiveRegisterWeight = 0.0;
  /// Weight for the effective backend unroll excess (unroll factor - 1).
  /// This is used only when a lowering supplies explicit unroll evidence.
  double unrollFactorWeight = 0.0;

  bool isValid() const;
  double evaluate(double directInstructions, double segments,
                  double strideRatio, double maskedAccesses,
                  double peakLiveExcess, double unrollFactor = 1.0) const;
};

/// Conditional one-time private-stack floor for a function-wide dot DAG.
struct DotComponentPrivateStackFloor {
  bool enabled = false;
  int64_t activeWarpCount = 0;
  int64_t superblockFactor = 0;
  int64_t minimumComponentNodes = 0;
  int64_t maximumComponentNodes = 0;
  int64_t m = 0, n = 0, k = 0;
  int64_t elementBits = 0;
  std::string dotUseSignature;
  int64_t loadInstructionFloor = 0;
  int64_t storeInstructionFloor = 0;
  /// End-to-end price of the load/store group in one compiler schedule.  This
  /// is a floor on the combined stream, not the sum of isolated opcode prices.
  double combinedGroupCycles = 0.0;
  /// Structural attribution of the whole-function group envelope.
  double preRecurrenceFraction = 0.0;
  double dotRegionFraction = 0.0;
  double postDotFraction = 0.0;
  std::string evidence;

  bool isValid() const;
};

/// Completed group fit after subtracting a matched nonempty control VF:
/// C(n>0)=first+(n-1)*increment.
struct StageMemoryLayoutCost {
  double firstCycles = 0.0;
  double incrementalCycles = 0.0;
  int64_t minCount = 1;
  int64_t maxCount = 0;
  std::string evidence;
  bool includesSharedLayout = false;
  bool isValid() const;
  std::optional<double> evaluate(double count) const;
};

struct PipelinedStridedStoreGroupCost {
  double kernelStartupCycles = 0.0;
  double firstStoreCycles = 0.0;
  double incrementalStoreCycles = 0.0;
  double extraLaneCycles = 0.0;
  int64_t minimumRows = 0;
  int64_t maximumRows = 0;
  int64_t minimumStores = 0;
  int64_t maximumStores = 0;
  std::string evidence;

  bool isValid() const;
};

struct MaskedStridedCopyCompletionCost {
  double baseSystemCycles = 0.0;
  double additionalRowSystemCycles = 0.0;
  int64_t minimumRows = 0;
  int64_t maximumRows = 0;
  int64_t minimumIterations = 0;
  int64_t maximumIterations = 0;
  std::string evidence;

  bool isValid() const;
  std::optional<double> evaluate(int64_t rows, int64_t iterations) const;
};

struct StageAtomicRate {
  double logicalElementsPerCycle = 0.0;
  double operationStartupCycles = 0.0;
  double resultDependencyCycles = 0.0;
  /// Relative penalty used when address collision is a runtime property.
  double unknownContentionMultiplier = 1.0;
  /// Optional rate for the one-physical-program hardware regime.
  double singlePhysicalProgramLogicalElementsPerCycle = 0.0;

  bool isValid() const;
};

struct DotInputDomainRate {
  double setupCycles = 0.0;
  double flopsPerCycle = 0.0;
  /// Effective critical path of a direct accumulator-feedback MMAD chain.
  double dependencyCyclesPerDot = 0.0;
  double dependencyFillCycles = 0.0;
};

struct StageModeProfile {
  /// Resolved by the evaluator; not an independently editable profile knob.
  int64_t activeWarpCount = 0;
  double setupCycles = 0.0;
  /// Completion-inclusive setup measurements indexed by the launched SIMT
  /// warp count.  The scalar setupCycles remains the documented fallback.
  std::map<int64_t, double> setupCyclesByWarpCount;
  /// Physical SIMD instruction width.
  int64_t vectorWidthBits = 1;
  int64_t vectorWidth = 1;
  int64_t issueWidth = 1;
  /// Scalar white-box terms in SYS_CNT cycles. These coefficients describe
  /// the target memory pipelines and are independent of any operator.
  double mainScalarLoadPrepCycles = 0.0;
  double mainScalarLoadFillCycles = 0.0;
  double mainScalarLoadIssueCycles = 0.0;
  double mainScalarLoadOutstandingLines = 0.0;
  double mainScalarLoadExtraLineLowCycles = 0.0;
  double mainScalarLoadExtraLineHighCycles = 0.0;
  double mainScalarLoadExtraLineHighThreshold = 0.0;
  double simtUniformLoadPrepCycles = 0.0;
  double simtUniformLoadFillCycles = 0.0;
  double simtUniformLoadDiffLineIssueCycles = 0.0;
  double mte3StorePrepCycles = 0.0;
  double mte3StoreFillCycles = 0.0;
  double simtUniformStoreBaseCycles = 0.0;
  /// Zero means no declared limit. A positive value is a compiler/target
  /// legality bound for reverse scans in this mode.
  int64_t maximumReversePrefixScanExtent = 0;
  llvm::StringMap<StageOperationRate> operationRates;
  double loadBytesPerCycle = 0.0;
  double storeBytesPerCycle = 0.0;
  /// Startup cost of one logical load/store operation.
  double loadOperationSetupCycles = 0.0;
  double storeOperationSetupCycles = 0.0;
  /// SIMD independent-loop MTE2 envelope for two or more uninterrupted FP32
  /// load streams.
  double independentLoadGroupKernelStartupCycles = 0.0;
  double independentLoadGroupIterationStartupCycles = 0.0;
  double independentLoadGroupBytesPerCycle = 0.0;
  int64_t independentLoadGroupMinimumOperations = 0;
  int64_t independentLoadGroupMaximumOperations = 0;
  int64_t computedAddressScalarizedLoadMinimumElements = 0;
  int64_t computedAddressScalarizedLoadMaximumElements = 0;
  double computedAddressScalarizedLoadCyclesPerElement = 0.0;
  std::string computedAddressScalarizedLoadEvidence;
  /// Residual SIMD completion curve for a rank-one FP32 load whose tensor
  /// address is selected by scalar control flow and lowered lane by lane.
  std::map<int64_t, double>
      controlFlowAddressScalarizedLoadResidualCyclesByElements;
  std::string controlFlowAddressScalarizedLoadEvidence;
  /// Residual completion of the rank-two FP16 SIMD masked load/store group.
  std::map<int64_t, std::map<int64_t, double>>
      maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows;
  std::string maskedLoadStoreRmwEvidence;
  /// SIMD scalar-row store groups indexed by row stride in bytes.
  std::map<int64_t, PipelinedStridedStoreGroupCost>
      pipelinedStridedStoreGroupsByRowStrideBytes;
  /// Directional completion floors for masked FP32 strided copies.  The key
  /// is the physical byte stride of the strided side.
  std::map<int64_t, MaskedStridedCopyCompletionCost>
      maskedStridedLoadCopyCompletionByStrideBytes;
  std::map<int64_t, MaskedStridedCopyCompletionCost>
      maskedStridedStoreCopyCompletionByStrideBytes;
  /// Generic SIMD completion floor for statically padded rectangular tiles.
  bool irregularTileCompletionEnabled = false;
  int64_t irregularTilePhysicalThresholdElements = 0;
  double irregularTilePhysicalExcessCyclesPerElement = 0.0;
  double irregularTileActiveCyclesPerElement = 0.0;
  double irregularTileTailCyclesPerInactiveElement = 0.0;
  int64_t irregularTileRowGroupSize = 0;
  int64_t irregularTileSmallRowMaximumRows = 0;
  double irregularTileLargeRowMaskedFloorScale = 1.0;
  double irregularTileAlignedTwoRowGroupPhysicalMaterializationScale = 0.0;
  int64_t irregularTileLargeWorkingSetFullPhysicalMinimumBytes = 0;
  double irregularTileAlignedFirstRowGroupCyclesPerElement = 0.0;
  double irregularTileAlignedAdditionalRowGroupCyclesPerElement = 0.0;
  std::string irregularTileCompletionEvidence;
  double loadWarpInstructionsPerCycle = 0.0;
  double storeWarpInstructionsPerCycle = 0.0;
  /// Optional rates for post-TTIR generated vector streams.  A zero rate
  /// falls back to the ordinary SIMT load/store rate.
  double sharedLocalLoadInstructionsPerCycle = 0.0;
  double sharedLocalStoreInstructionsPerCycle = 0.0;
  double privateStackLoadInstructionsPerCycle = 0.0;
  double privateStackStoreInstructionsPerCycle = 0.0;
  /// Register budget in 32-bit units/thread.  Zero derives the tested
  /// launch-bound curve min(127, floor(1024 / activeWarps)).
  double registerBudgetPerThread = 0.0;
  double spillTransactionCoverage = 1.0;
  SimtSourceFit globalLoadSourceFit;
  SimtSourceFit globalStoreSourceFit;
  SimtSourceFit sharedLoadSourceFit;
  SimtSourceFit sharedStoreSourceFit;
  SimtSourceFit privateLoadSourceFit;
  SimtSourceFit privateStoreSourceFit;
  /// Conditional whole-VF floors.
  std::vector<DotComponentPrivateStackFloor> dotComponentPrivateStackFloors;
  /// Whole-kernel entry convergence and final-drain completion curves.
  std::vector<WholeSimdCompletionCurve> wholeSimdCompletionCurves;
  std::vector<WholeSimtCompletionCurve> wholeSimtCompletionCurves;
  std::map<int64_t, double> loadWarpInstructionsPerCycleByWarpCount;
  std::map<int64_t, double> storeWarpInstructionsPerCycleByWarpCount;
  std::map<std::string, StageMemoryLayoutCost> loadLayoutCosts;
  std::map<std::string, StageMemoryLayoutCost> storeLayoutCosts;
  /// FMA accumulator -> row-major store layout exchange. Keys describe the
  /// measured dtype/shape/base-warps/F domain, not an operator or Stage name.
  std::map<std::string, StageMemoryLayoutCost> dotStoreSharedLayoutCosts;
  /// Composite whole-SIMT FMA-dot paths, including operand GM/shared traffic.
  /// Key includes TTIR geometry, base warp count and SuperBlock factor.
  std::map<std::string, StageMemoryLayoutCost> dotOperandPathCosts;
  /// Optimistic full-tile service estimate, not first-round completion or
  /// barrier/stack cost. The unit is fitted address coverage, not an ISA claim.
  int64_t storeFootprintUnitBytes = 0;
  double storeCyclesPerFootprintUnit = 0.0;
  std::map<int64_t, double> storeIssueFloorByWarpCount;
  double predicateOperationsPerCycle = 0.0;
  /// Source-level row-reduction envelope.
  double reductionOperationSetupCycles = 0.0;
  double reductionLaneStepsPerCycle = 0.0;
  int64_t scalarizedMultiResultReductionMinimumExtent = 0;
  int64_t scalarizedMultiResultReductionReferenceResults = 0;
  double scalarizedMultiResultReductionSetupCycles = 0.0;
  double scalarizedMultiResultReductionElementsPerCycle = 0.0;
  std::string scalarizedMultiResultReductionEvidence;
  /// Completion service of ascend.sort/hivm.hir.vsort in its measured geometry
  /// domain.
  int64_t sortMinimumExtent = 0;
  int64_t sortMaximumExtent = 0;
  int64_t sortElementBits = 0;
  double sortSetupCycles = 0.0;
  double sortRowsPerCycle = 0.0;
  std::string sortEvidence;
  double maskedAccumulatorReductionCompletionSetupCycles = 0.0;
  int64_t rowwiseReductionGroupBaseWarpCount = 0;
  int64_t rowwiseReductionGroupReferenceOperations = 0;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      rowwiseReductionGroupCyclesByFactorRowsAndExtent;
  /// Additional completion surfaces keyed first by the exact number of TTIR
  /// reduction operations.
  std::map<int64_t,
           std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>>
      rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent;
  std::string rowwiseReductionGroupEvidence;
  /// Whole-SIMT completion body for one multi-result tt.reduce.
  int64_t multiResultReductionGroupBaseWarpCount = 0;
  int64_t multiResultReductionResultCount = 0;
  int64_t multiResultReductionElementBits = 0;
  int64_t multiResultReductionPaddedAxisExtent = 0;
  std::map<std::string, std::map<int64_t, double>>
      multiResultReductionGroupCyclesByTailClassAndRows;
  std::string multiResultReductionGroupEvidence;
  bool twoPassRowwiseOwnsSegmentedIndirectLoads = false;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      twoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot;
  std::string twoPassRowwiseCompletionEvidence;
  double shuffleLanesPerCycle = 0.0;
  std::map<int64_t, double> shuffleLanesPerCycleByWarpCount;
  /// Latency of one dependent scan step. For SIMD this is one dependent
  /// vector add; for SIMT it is one dependent shuffle/add/control step.
  double prefixScanStepLatencyCycles = 0.0;
  /// Completion-inclusive throughput of the backend external helper used by a
  /// straight-line SIMT scan.
  double straightLinePrefixScanElementsPerCycle = 0.0;
  std::string straightLinePrefixScanCompletionEvidence;
  /// SIMD helper completion curves keyed by structural combine class.
  std::map<std::string, StraightLinePrefixScanCost>
      straightLinePrefixScanCostsByCombine;
  /// Dynamic source-loop prefix-scan body/startup curves, indexed by semantic
  /// scan extent and kept separate by direction.
  std::map<int64_t, double> gridStrideForwardScanCyclesByExtent;
  std::map<int64_t, double> gridStrideForwardScanStartupCyclesByExtent;
  std::map<int64_t, double> gridStrideReverseScanCyclesByExtent;
  std::map<int64_t, double> gridStrideReverseScanStartupCyclesByExtent;
  /// Factor-specific source-loop scan service.
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideForwardScanCyclesByFactorAndExtent;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideForwardScanStartupCyclesByFactorAndExtent;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideReverseScanCyclesByFactorAndExtent;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideReverseScanStartupCyclesByFactorAndExtent;
  int64_t gridStrideResidentMemoryMaximumBytes = 0;
  double gridStrideResidentLoadWarpInstructionsPerCycle = 0.0;
  double gridStrideResidentStoreWarpInstructionsPerCycle = 0.0;
  std::string gridStrideResidentMemoryEvidence;
  double prefixScanDependencyFactor = 1.0;
  double dotSetupCycles = 0.0;
  double dotFlopsPerCycle = 0.0;
  /// Optional SIMD MMAD service rates keyed by TTIR operand dtype/precision.
  /// Missing keys retain the generic dotSetupCycles/dotFlopsPerCycle fallback.
  std::map<std::string, DotInputDomainRate> dotByInputDomain;
  /// Aggregate dependent-dot throughput selected by the physical active-warp
  /// count.
  std::map<int64_t, double> dotFlopsPerCycleByWarpCount;
  double dotPartialCompositeResidualRelativeScale = 1.0;
  double issueOperationsPerCycle = 0.0;
  /// Target UB capacity available to one physical SIMT group.  Zero means the
  /// profile does not provide a pre-lowering UB legality bound.
  int64_t usableUbBytes = 0;
  std::map<int64_t, double> indirectLoadSystemCyclesByWarpInstructions;
  /// SIMD scalarizes a loaded-index tensor into lane transactions.
  std::map<int64_t, double> indirectLoadSystemCyclesByElements;
  std::map<int64_t, double> indirectLoadStartupSystemCyclesByWarpInstructions;
  std::map<int64_t, double> indirectStoreSystemCyclesBySegments;
  /// Physical-group loaded-index curves measured independently for each
  /// SuperBlock factor.
  std::map<int64_t, std::map<int64_t, double>>
      indirectLoadGroupCyclesByFactorAndWarpInstructions;
  std::map<int64_t, std::map<int64_t, double>>
      indirectLoadGroupStartupCyclesByFactorAndWarpInstructions;
  /// Completion-exclusive segmented loaded-index service.
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      segmentedIndirectLoadGroupCyclesByFactorSegmentsAndContiguousElements;
  /// FP32 counterpart of the segmented loaded-index surface.
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      segmentedIndirectLoadFp32GroupCyclesByFactorSegmentsAndContiguousElements;
  /// Dynamic grid-stride loops serialize successive source-loop bodies on one
  /// resident program.
  std::map<int64_t, double> gridStrideIndirectLoadCyclesByWarpInstructions;
  std::map<int64_t, double>
      gridStrideIndirectLoadStartupCyclesByWarpInstructions;
  /// Factor-specific dynamic source-loop service.
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideIndirectLoadCyclesByFactorAndWarpInstructions;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideIndirectLoadStartupCyclesByFactorAndWarpInstructions;
  /// Completion-inclusive dot cost for one dynamic grid-stride body,
  /// indexed by semantic FLOPs of one dot, plus its route-shared startup.
  std::map<int64_t, double> gridStrideDotCyclesByFlops;
  std::map<int64_t, double> gridStrideDotStartupCyclesByFlops;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideDotCyclesByFactorAndFlops;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideDotStartupCyclesByFactorAndFlops;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      gridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      gridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry;
  std::string gridStrideIndirectDotCompositionEvidence;
  /// Measured cost of one physical SuperBlock group's recurrence iteration.
  int64_t recurrenceGroupBaseWarpCount = 0;
  std::map<int64_t, std::map<int64_t, double>>
      recurrenceReductionGroupCyclesByFactorAndLaneSteps;
  /// One-time startup of the same measured recurrence group.
  std::map<int64_t, std::map<int64_t, double>>
      recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps;
  bool singleResultRowReductionRecurrenceCompletionEnabled = false;
  int64_t singleResultRowReductionRecurrenceMinimumExtent = 0;
  int64_t singleResultRowReductionRecurrenceMaximumExtent = 0;
  int64_t singleResultRowReductionRecurrenceMinimumIterations = 0;
  int64_t singleResultRowReductionRecurrenceMaximumIterations = 0;
  int64_t singleResultRowReductionRecurrenceMinimumPrograms = 0;
  int64_t singleResultRowReductionRecurrenceMaximumPrograms = 0;
  double singleResultRowReductionRecurrenceIterationCycles = 0.0;
  double singleResultRowReductionRecurrenceStartupInterceptCycles = 0.0;
  double singleResultRowReductionRecurrenceStartupSqrtProgramCycles = 0.0;
  std::string singleResultRowReductionRecurrenceEvidence;
  /// Steady incremental cost of one ordinary FP32 load/store operation in a
  /// physical group.
  int64_t continuousMemoryGroupBaseWarpCount = 0;
  std::map<int64_t, std::map<int64_t, double>>
      continuousLoadGroupCyclesByFactorAndBytes;
  std::map<int64_t, std::map<int64_t, double>>
      continuousStoreGroupCyclesByFactorAndBytes;
  /// Optional one-time Stage-direction startup fitted independently from the
  /// steady operation slope.
  std::map<int64_t, std::map<int64_t, double>>
      continuousLoadGroupStartupCyclesByFactorAndBytes;
  std::map<int64_t, std::map<int64_t, double>>
      continuousStoreGroupStartupCyclesByFactorAndBytes;
  /// High-wave completion service for the SIMD parent of a Mixed route.  It
  /// scales with compiler-derived parent memory bytes and AIC loop count.
  bool mixedParentMemoryCompletionEnabled = false;
  int64_t mixedParentMemoryMinimumIterations = 0;
  int64_t mixedParentMemoryReferenceBytesPerProgram = 0;
  double mixedParentMemoryCyclesPerExcessIteration = 0.0;
  std::string mixedParentMemoryCompletionEvidence;
  double spillTransactionsPerCycle = 0.0;
  /// Loaded-index memory cannot use the continuous MTE/LSU throughput model.
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
  /// Base warp count of one logical SIMT program.
  int64_t baseSimtWarpCount = 1;
  /// CV1:2 work partition, applied before local SuperBlock grouping. This is
  /// not a divisor of Q or of the launch warp count.
  int64_t cvAivSubblocks = 1;
  /// Resource/grid bound for analytical estimates, independent of anchors and
  /// executable candidate enablement. Does not grant materialization legality.
  int64_t maximumSimtEstimationFactor = 4;
  /// Logical warp groups available to one SIMT program.
  int64_t logicalWarpGroupCount = 1;
  /// Long-lived recurrence state consumes finite register/stack bandwidth.
  int64_t superblockUsefulFactorLimit = 1;
  int64_t superblockPersistentStatePressureFreeFactor = 1;
  double superblockPersistentStateBytesPerCycle = 1.0;
  StageModeProfile simd;
  StageModeProfile simt;
  ScopeHandoffCost scopeHandoff;

  bool isValid() const;
};

class StageCostEvaluator {
public:
  llvm::Expected<StageCostTable>
  evaluate(const StagePartition &partition, const HardwareProfile &profile,
           bool collectCalibrationEstimates = true) const;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H
