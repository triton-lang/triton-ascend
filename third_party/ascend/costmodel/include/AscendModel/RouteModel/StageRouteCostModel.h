//===- StageRouteCostModel.h - Logical-stage route model -------*- C++ -*-===//
//
// A kernel is represented as serial algorithm stages.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::ascend {

enum class StageMode { SIMD, SIMT };
enum class StageKernelRouteKind { AllSIMD, AllSIMT, Mixed };
enum class StageDependencyKind { IntraIterationData, LoopCarriedData };
enum class StageScheduleKind {
  StraightLine,
  IndependentPipelined,
  LoopCarriedSerial,
  PartiallyDependent,
};

/// Completion-inclusive correction for a structurally matched whole-SIMT dot
/// schedule.
struct WholeSimtCompletionCurve {
  bool enabled = false;
  int64_t activeWarpCount = 0;
  int64_t superblockFactor = 0;
  int64_t minimumComponentNodes = 0;
  int64_t maximumComponentNodes = 0;
  int64_t m = 0, n = 0, k = 0;
  int64_t elementBits = 0;
  std::string dotUseSignature;
  int64_t referenceRowStrideBytes = 0;
  double referencePreRecurrenceScalarOperations = 0.0;

  /// Optional factor-specific corrections for the measured recurrence and
  /// post-recurrence dot body.
  double recurrenceScale = 1.0;
  double dotRegionScale = 1.0;

  double entryGroupMidpoint = 0.0;
  double entryGroupPower = 0.0;
  std::array<double, 8> entryAdditiveCoefficients = {};

  /// Completion-inclusive final-drain service using the same eight-term basis.
  bool exitRampExponential = true;
  double exitGroupTimeConstant = 0.0;
  double exitGroupMidpoint = 0.0;
  double exitGroupPower = 0.0;
  std::array<double, 8> exitAbsoluteCoefficients = {};
  double exitTailLog2ActiveFractionCoefficient = 0.0;

  /// Scalar prologue/epilogue work outside the four instrumented body phases.
  /// This contributes to the whole-kernel total and is reported separately.
  double kernelBoundarySystemCycles = 0.0;
  std::string evidence;

  bool isValid() const;
};

/// Completion curve for a structurally matched whole-SIMD dot schedule.
struct WholeSimdCompletionCurve {
  bool enabled = false;
  int64_t minimumComponentNodes = 0;
  int64_t maximumComponentNodes = 0;
  int64_t m = 0, n = 0, k = 0;
  int64_t elementBits = 0;
  std::string dotUseSignature;
  int64_t minimumRowStrideBytes = 0;
  int64_t maximumRowStrideBytes = 0;
  int64_t minimumGroupCount = 0;
  int64_t maximumGroupCount = 0;

  double entryBaseScale = 1.0;
  double entrySaturatedScale = 1.0;
  /// Large multi-wave schedules have different convergence for power-of-two and
  /// non-power-of-two row strides.
  double entryUnalignedGroupMidpoint = 0.0;
  double entryGroupMidpoint = 0.0;
  double entryGroupPower = 0.0;
  /// One-wave completion floor as a Hill curve of active scheduling slots.
  double singleWaveEntryBaseSystemCycles = 0.0;
  double singleWaveEntrySaturatedSystemCycles = 0.0;
  double singleWaveEntrySlotMidpoint = 0.0;
  double singleWaveEntrySlotPower = 0.0;
  /// Two-wave completion floor as a function of the last-wave active-slot
  /// fraction.
  double twoWaveEntryBaseSystemCycles = 0.0;
  double twoWaveEntryGaussianSystemCycles = 0.0;
  double twoWaveEntryTailCenter = 0.0;
  double twoWaveEntryTailWidth = 0.0;
  double twoWaveEntryHighTailPenaltySystemCycles = 0.0;
  double twoWaveEntryHighTailMidpoint = 0.0;
  double twoWaveEntryHighTailPower = 0.0;
  double twoWaveEntryFullTailSystemCycles = 0.0;
  double recurrenceSetupSystemCyclesPerGroup = 0.0;
  double dotRegionScale = 1.0;
  double singleWaveDotBaseSystemCycles = 0.0;
  double singleWaveDotSaturatedSystemCycles = 0.0;
  double singleWaveDotSlotMidpoint = 0.0;
  double singleWaveDotSlotPower = 0.0;
  double twoWaveDotTargetSystemCycles = 0.0;
  double exitBaseScale = 1.0;
  double exitSaturatedScale = 1.0;
  double exitUnalignedGroupMidpoint = 0.0;
  double exitGroupMidpoint = 0.0;
  double exitGroupPower = 0.0;
  double singleWaveExitBaseSystemCycles = 0.0;
  double singleWaveExitSaturatedSystemCycles = 0.0;
  double singleWaveExitSlotMidpoint = 0.0;
  double singleWaveExitSlotPower = 0.0;
  double twoWaveExitBaseSystemCycles = 0.0;
  double twoWaveExitGaussianSystemCycles = 0.0;
  double twoWaveExitTailCenter = 0.0;
  double twoWaveExitTailWidth = 0.0;
  double kernelBoundarySystemCycles = 0.0;
  std::string evidence;

  bool isValid() const;
};

llvm::StringRef stringifyStageMode(StageMode mode);

llvm::StringRef stringifyStageDependencyKind(StageDependencyKind kind);

/// One explicit same-iteration or loop-carried data dependency between
/// logical Stage instances. Stage IDs, rather than vector indices, keep
/// reports stable after StageKindClassifier assigns final semantic names.
struct StageDependency {
  std::string loopStage;
  std::string sourceStage;
  std::string targetStage;
  StageDependencyKind kind = StageDependencyKind::LoopCarriedData;
  int64_t iterationDistance = 1;
  /// Retained report keys for compatibility; same-iteration edges use the
  /// synthetic name `inter_stage_ssa` and index zero.
  int64_t carriedValueIndex = 0;
  std::string carriedValueName;
  int64_t carriedValueBytes = 0;
  std::string sourceLocation;

  bool isValid() const;
  llvm::json::Object toJSON() const;
};

struct StageImplementation {
  StageMode mode = StageMode::SIMD;
  /// SIMD always uses factor=1.
  int64_t superblockFactor = 1;
  bool localScope = false;
  /// AIVs sharing ONE logical program in a CV parent, not scheduling slots.
  /// Standalone SIMT uses one; local CV implementations may use two.
  int64_t aivSubblocks = 1;

  bool isValid() const;
  llvm::json::Object toJSON() const;
};

/// Structural facts owned by one logical Stage.
struct StageModelFeatures {
  bool hasLoop = false;
  bool hasLoopCarriedDataDependency = false;
  bool hasPointerInduction = false;
  bool hasContiguousMemory = false;
  bool hasIndirectMemory = false;
  bool hasAtomicMemory = false;
  bool hasReduction = false;
  bool hasPrefixScan = false;
  bool hasDot = false;
  bool hasConversionPack = false;
  /// The Stage is inside the AutoBlockify V1 scheduling loop, possibly below a
  /// nested structured region.
  bool insideAutoBlockifyV1Loop = false;
  /// A source-level grid-stride loop has lower bound tt.get_program_id and step
  /// tt.get_num_programs.
  bool isGridStrideLoopShell = false;
  bool insideGridStrideLoopBody = false;
  int64_t conditionalBranchCount = 0;
  int64_t divergentBranchCount = 0;
  int64_t loopBackedgeCount = 0;
  // Workload estimates, not legality gates. A bounded loop may execute fewer
  // iterations at runtime (including zero); an unknown loop uses a fallback.
  int64_t upperBoundLoopCount = 0;
  int64_t unknownTripCountLoopCount = 0;
  int64_t synchronizationCount = 0;
  /// Number of mutually independent loop-carried recurrence groups owned by
  /// this Stage.
  int64_t parallelRecurrenceGroupCount = 1;
  double activeLaneRatio = 1.0;

  bool isValid() const;
  bool permitsSimdRoofline() const;
  llvm::json::Object toJSON() const;
};

/// Logical GM access, before SIMD sub-block or SIMT lane mapping. Unknown
/// strides are not zero and a unit inner stride does not imply a packed tile.
struct StageMemoryAccess {
  /// Logical address geometry, not proof of physical instructions or lanes.
  enum class LayoutClass {
    Unknown,
    ContiguousShort,
    ContiguousWide,
    StridedShort,
    StridedWide,
  };
  std::vector<int64_t> shape;
  std::vector<std::optional<int64_t>> strides;
  int64_t elementBits = 0;
  double count = 1.0;
  bool store = false;
  bool masked = false;
  /// True when at least one address component is produced by a memory load.
  bool addressDependsOnLoadedIndex = false;
  bool addressHasNonAffineTensorComputation = false;
  /// True when a tensor address is selected by scalar control flow (currently
  /// an scf.if result).
  bool addressHasTensorControlFlowSelection = false;
  bool maskedStoreHasMatchingLoadDependency = false;
  /// Fraction of statically provable active elements in the memory mask.
  double activeElementRatio = 1.0;
  /// TTIR provenance only, not an unconditional layout/ISA claim. A measured
  /// lowering contract must additionally match shape, dtype and launch shape.
  bool fromDotAccumulator = false;
  /// Nonempty only when every use is a direct dot input in the same block.
  /// A composite dot profile may then own this load instead of the GM model.
  std::vector<std::string> dotConsumerPaths;

  int64_t effectiveRank() const;
  int64_t contiguousAxisBytes() const;
  int64_t rowBytes() const;
  int64_t rowStrideBytes() const;
  int64_t rowCount() const;
  int64_t provenContiguousElements() const;
  /// Conditional piece count before dimension folding; not a DMA selection.
  int64_t descriptorCount(unsigned maxRank) const;
  LayoutClass layoutClass() const;
  std::string layoutKey() const;
  bool hasStaticLayout() const;
  bool hasUnitInnerStride() const;
  bool hasNonUnitInnerStride() const;
  bool isFlatContiguous() const;
  bool isValid() const;
  llvm::json::Object toJSON() const;
};

llvm::StringRef
stringifyStageMemoryLayout(StageMemoryAccess::LayoutClass layout);

/// Route-independent semantic description of one dynamically executed TTIR
/// atomic operation. Physical transactions remain a mode/profile concern.
struct AtomicWorkload {
  std::string kind;
  std::string dataType;
  std::string memorySemantic;
  std::string memoryScope;
  double logicalElements = 0.0;
  double logicalOperationInstances = 0.0;
  double provenContiguousRunWidth = 0.0;
  bool resultUsed = false;
  bool addressDependsOnLoadedIndex = false;
  bool contentionUnknown = true;

  std::string profileKey() const;
  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// A group of tensor operations that share the same projected pointwise
/// lowering geometry.
struct TensorOperationWorkload {
  std::string operation;
  int64_t elementBitWidth = 0;
  double logicalElements = 0.0;
  double segmentCount = 0.0;
  int64_t contiguousElementsPerSegment = 0;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// Semantic TTIR dot key. A physical resource observation may be used only
/// after its lowering contract has been admitted for the current target.
using DotShapeKey = std::tuple<int64_t, int64_t, int64_t, int64_t, std::string,
                               std::string, std::string, std::string>;

/// Mode-independent work owned exactly once by one Stage.  Values are
/// logical elements/bytes, not mode-specific instructions or cycles.
struct StageWorkload {
  // Element count of one operation -> dynamic occurrences per Stage iteration.
  // Keep operation boundaries until mode-specific lane rounding is applied.
  llvm::StringMap<std::map<int64_t, double>> operationCounts;
  std::vector<TensorOperationWorkload> tensorOperationWorkloads;
  double scalarOperations = 0.0;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  /// Number of logical load/store operations. Operation boundaries are kept
  /// because bytes alone cannot model short-transfer and pipeline behavior.
  double loadOperations = 0.0;
  double storeOperations = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  /// Lowering-provided generated vector traffic.
  double sharedLocalLoadInstructions = 0.0;
  double sharedLocalStoreInstructions = 0.0;
  double privateStackLoadInstructions = 0.0;
  double privateStackStoreInstructions = 0.0;
  std::vector<StageMemoryAccess> memoryAccesses;
  std::vector<AtomicWorkload> atomicWorkloads;
  /// (bytes, logical elements) in one ordinary contiguous load/store -> dynamic
  /// occurrences.
  std::map<std::pair<int64_t, int64_t>, double> continuousLoadCounts;
  std::map<std::pair<int64_t, int64_t>, double> continuousStoreCounts;
  /// (bytes, logical elements) in one loaded-index load -> dynamic load
  /// occurrences.
  std::map<std::pair<int64_t, int64_t>, double> indirectLoadCounts;
  std::map<std::tuple<int64_t, int64_t, int64_t>, double>
      segmentedIndirectLoadCounts;
  std::map<std::pair<int64_t, int64_t>, double> indirectStoreCounts;
  std::map<std::tuple<int64_t, int64_t, int64_t>, double>
      segmentedIndirectStoreCounts;
  std::map<int64_t, double> predicateCounts;
  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, bool, std::string>,
           double>
      prefixScanCounts;
  /// Dynamic row-reduction operations per Stage iteration.
  double reductionOperations = 0.0;
  double shuffleLaneSteps = 0.0;
  /// (independent rows, reduction-axis extent) -> reduction occurrences.
  std::map<std::tuple<int64_t, int64_t, int64_t>, double>
      reductionGeometryCounts;
  /// Associative operation returned by a tt.reduce combine region.
  std::map<std::string, double> reductionCombineClassCounts;
  /// Multi-result reductions that the current SIMD lowering serializes in
  /// scalar UB loops.
  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, double>
      scalarizedMultiResultReductionCounts;
  /// Native ascend.sort invocations keyed by (independent rows, sort-axis
  /// extent, element bit width).
  std::map<std::tuple<int64_t, int64_t, int64_t>, double> sortCounts;
  std::map<std::pair<int64_t, int64_t>, double>
      oneTripMaskedAccumulatorReductionExtents;
  /// Dynamic dot matrix instances per Stage iteration (including batch).
  double dotOperations = 0.0;
  /// Per-matrix FLOP shape -> dynamic matrix instances.
  std::map<int64_t, double> dotFlopCounts;
  /// (M, N, K, batch, lhs type, rhs type, accumulator type, precision) ->
  /// source-level dot invocations per Stage iteration. Diagnostic only: the
  /// cost is still owned by dotFlopCounts/dotFlops exactly once.
  std::map<DotShapeKey, double> dotShapeCounts;
  /// TTIR operand provenance/layout key -> (FLOPs per dot, occurrences).
  /// Does not assert a physical shared-memory instruction count.
  std::map<std::string, std::pair<int64_t, double>> dotOperandPaths;
  double dotFlops = 0.0;
  double issueElements = 0.0;
  // Dynamic events normalized by iterationCount, like the other workload.
  double loopBackedges = 0.0;
  double conditionalBranches = 0.0;
  double divergentBranches = 0.0;
  double synchronizations = 0.0;
  /// Scalar GM operations owned by this Stage per iteration.
  double scalarLoadCount = 0.0;
  double scalarStoreCount = 0.0;
  bool paysKernelSetup = false;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

struct StageResourceCycles {
  double setup = 0.0;
  double scalar = 0.0;
  double load = 0.0;
  double store = 0.0;
  bool rowwiseAdditionalLoadSetupsOwnedOnce = false;
  /// Generated-stream attribution.  These cycles are already included in
  /// `load`/`store`; they are diagnostic decomposition, not extra terms.
  double sharedLocalLoad = 0.0;
  double sharedLocalStore = 0.0;
  double privateStackLoad = 0.0;
  double privateStackStore = 0.0;
  /// Residual needed to raise already attributed private-stack work to a
  /// measured combined load/store group floor.
  double privateStackCombinedFloor = 0.0;
  /// SIMD backend completion for a tensor load whose non-affine address DAG
  /// lowers to a scalar per-lane service loop. Already included in `load`.
  double computedAddressScalarizedLoad = 0.0;
  /// SIMD backend completion for a load whose tensor address is selected by
  /// scalar control flow. Already included in `load`.
  double controlFlowAddressScalarizedLoad = 0.0;
  double maskedLoadStoreReadModifyWrite = 0.0;
  bool dotComponentPrivateStackFloorApplied = false;
  /// Shared-memory layout exchange, including its internal barriers.
  double sharedLayout = 0.0;
  double atomic = 0.0;
  /// Alternative atomic service for the one-physical-program SIMD regime.
  double singlePhysicalProgramAtomic = 0.0;
  bool hasSinglePhysicalProgramAtomicRate = false;
  double compute = 0.0;
  /// Serialized backend completion paid once per semantic tensor-operation
  /// instance.
  double operationCompletion = 0.0;
  double predicate = 0.0;
  /// Row-reduction execution envelope.
  double reduction = 0.0;
  /// SIMD scalar-loop service selected for structurally matched multi-result
  /// reductions.  It replaces the optimistic vector-reduction envelope.
  double scalarizedMultiResultReduction = 0.0;
  /// Completion-exclusive service of the native backend row-sort primitive.
  double sort = 0.0;
  double maskedAccumulatorReductionCompletionSetup = 0.0;
  /// Completion-inclusive row-reduction body for one physical SIMT group.
  double rowwiseReductionGroupBody = 0.0;
  /// Completion-inclusive body of a structurally matched whole-SIMT
  /// multi-result reduction lowering.
  double multiResultReductionGroupBody = 0.0;
  double shuffle = 0.0;
  /// Portion of shuffle attributed to tt.scan dependency levels.
  double scanShuffle = 0.0;
  double dot = 0.0;
  /// True when `dot` includes the qualified SIMD physical resource formula.
  /// No completion or ready latency is implied.
  bool dotPhysicalResourceFormulaApplied = false;
  /// Diagnostic attribution only: already included in dot, not additive.
  double dotOperandGroup = 0.0;
  double dotInputLoadsCovered = 0.0;
  double loopControl = 0.0;
  double branchControl = 0.0;
  double divergence = 0.0;
  double synchronization = 0.0;
  double spill = 0.0;
  double issue = 0.0;
  /// SIMD-only completion bounds for rectangular tiles.
  double irregularMaskedTileFloor = 0.0;
  double irregularPhysicalMaterialization = 0.0;
  /// Full physical materialization before resident row-group ownership.
  double irregularFullPhysicalMaterialization = 0.0;
  double irregularTailMaterialization = 0.0;
  /// Prefix scan has two independent lower bounds: aggregate instruction
  /// issue across all sequences and the longest dependent scan chain.
  double prefixScanIssue = 0.0;
  double prefixScanCriticalPath = 0.0;
  /// Completion-inclusive body cost from an isolated dynamic grid-stride
  /// prefix-scan sweep. It replaces the decomposed scan/load/store estimate.
  double gridStridePrefixScanBody = 0.0;
  /// The selected dynamic source-loop resource curve was measured for this
  /// concrete SuperBlock factor.
  bool gridStrideFactorSpecificResourceBody = false;
  /// Candidate-specific measured lower bound for one physical SuperBlock
  /// group's recurrence body iteration.
  double recurrenceGroupCriticalPath = 0.0;
  /// One-time intercept of the matching physical-group recurrence sweep.
  double recurrenceGroupStartup = 0.0;
  /// Candidate-specific measured incremental memory cost for one physical
  /// group.
  double continuousLoadGroup = 0.0;
  double continuousStoreGroup = 0.0;
  /// Measured loaded-index envelope for one physical execution group.
  double indirectLoadGroup = 0.0;
  /// Kernel-wide intercept of the loaded-index group curve.
  double indirectLoadGroupStartup = 0.0;
  /// Startup selected from isolated dynamic grid-stride resource sweeps.
  /// Route composition pays the maximum matched resource envelope once.
  double gridStrideStartup = 0.0;
  /// Full-tile store service floor divided by F, like the analytical store
  /// resource. SuperBlock applies F once. Excludes first-round/sync costs.
  double storeLayoutService = 0.0;
  double criticalPath = 0.0;
  bool indirectMemoryAnalyticalFallback = false;

  std::array<std::pair<llvm::StringRef, double>, 51> keyedValues() const;
  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
  /// Preserve every non-zero resource while omitting default-valued
  /// diagnostic fields from the online compiler report.
  llvm::json::Object toCompactJSON() const;
};

struct StageImplementationCost {
  StageImplementation implementation;
  double totalCycles = 0.0;
  /// A profiled semantic operation in this implementation already includes
  /// the fixed local-SIMT scope enter/exit completion envelope.
  bool localScopeCompletionInclusive = false;
  /// Diagnostics for a physical CV1:N payload partition.
  double cvPartitionIdealTotalCycles = 0.0;
  double cvPartitionPayloadEfficiency = 1.0;
  double cvPartitionInefficiencyCorrectionCycles = 0.0;
  StageResourceCycles resources;
  /// Describes the strongest formula evidence used for this candidate.  It
  /// is report provenance, not a route multiplier.
  std::string formulaEvidence;
  /// Per-AIV work before SuperBlock replication. LogicalStageCost::workload
  /// remains the original, mode-independent logical-program workload.
  std::optional<StageWorkload> perAivWorkload;

  bool isValid() const;
  llvm::json::Object toJSON() const;
  llvm::json::Object toCompactJSON() const;
};

/// A dot's operands and the distinct values retained across that operation.
struct SimtDotPressureSite {
  int64_t m = 0, n = 0, k = 0;
  int64_t elementBits = 0;
  int64_t operandUnitsPerThread = 0;
  int64_t retainedTensorUnitsPerThread = 0;
  int64_t retainedScalarUnitsPerThread = 0;

  llvm::json::Object toJSON() const;
};

/// One source-ordered TTIR dot node.
struct SimtDotDependencySite {
  int64_t ordinal = 0;
  int64_t operationOrdinal = 0;
  std::vector<int64_t> predecessorOrdinals;
  int64_t depth = 0;
  int64_t successorCount = 0;
  int64_t nonDotPureConsumerCount = 0;
  int64_t sideEffectFrontierCount = 0;
  int64_t maxNonDotPureUseSpan = 0;
  int64_t maxSideEffectUseSpan = 0;

  llvm::json::Object toJSON() const;
};

/// Simultaneous TTIR values, in 32-bit units per thread of one logical program.
enum class SimtPrivateStackChargeRegion {
  None,
  WholeFunction,
  BeforeRecurrence,
  DotRegion,
  AfterDot,
};

struct SimtRegisterPressure {
  int64_t baseWarpCount = 0;
  int64_t peakLiveUnitsPerThread = 0;
  int64_t peakTensorUnitsPerThread = 0;
  int64_t peakScalarUnitsPerThread = 0;
  int64_t peakCrossStageUnitsPerThread = 0;
  int64_t peakLoopCarriedUnitsPerThread = 0;
  int64_t peakRematerializableUnitsPerThread = 0;
  /// Predicate structure remains separate from 32-bit GPR pressure.
  int64_t peakLivePredicateUnitsPerThread = 0;
  int64_t predicateDefinitionUnitsPerThread = 0;
  int64_t predicateUseUnitsPerThread = 0;
  int64_t maxPredicateFanout = 0;
  /// TTIR predicate topology, kept separate from machine P-register claims.
  int64_t ttirIntegerEqualityPredicateDefinitionUnitsPerThread = 0;
  int64_t ttirIntegerOrderedPredicateDefinitionUnitsPerThread = 0;
  int64_t ttirPredicateShapeDefinitionUnitsPerThread = 0;
  int64_t ttirOtherPredicateDefinitionUnitsPerThread = 0;
  int64_t ttirPredicateDagEdgeUnitsPerThread = 0;
  int64_t ttirMaxPredicateDagDepth = 0;
  int64_t ttirSelectPredicateUseUnitsPerThread = 0;
  int64_t ttirBooleanPredicateUseUnitsPerThread = 0;
  /// Predicate values defined outside an scf.for and consumed inside it.
  int64_t loopInvariantPredicateUnitsPerThread = 0;
  int64_t loopInvariantPredicateUseUnitsPerThread = 0;
  /// Separate address features for calibrating lowering expansion.
  int64_t peakPointerUnitsPerThread = 0;
  int64_t peakBlockPointerCount = 0;
  int64_t peakBlockPointerRankSum = 0;
  int64_t unknownValueCount = 0;
  /// Optional backend evidence.
  int64_t loweredPeakLiveUnitsPerThread = 0;
  int64_t backendUnrollFactor = 1;
  /// TTIR dot dependency topology.
  int64_t ttirDotDefinitionCount = 0;
  int64_t ttirDotDependencyEdgeCount = 0;
  int64_t ttirDotDependencySourceCount = 0;
  int64_t ttirDotDependencySinkCount = 0;
  int64_t ttirMaxDotDependencyDepth = 0;
  /// Function-wide weak connectivity of the dot DAG.
  int64_t ttirMaximumWeakDotComponentNodes = 0;
  bool ttirMaximumWeakDotComponentGeometryKnown = false;
  int64_t ttirMaximumWeakDotComponentM = 0;
  int64_t ttirMaximumWeakDotComponentN = 0;
  int64_t ttirMaximumWeakDotComponentK = 0;
  int64_t ttirMaximumWeakDotComponentElementBits = 0;
  /// Canonical function-wide dot topology plus observable use context.
  std::string ttirFunctionDotUseSignature;
  SimtPrivateStackChargeRegion dotComponentPrivateStackChargeRegion =
      SimtPrivateStackChargeRegion::None;
  std::vector<SimtDotDependencySite> dotDependencySites;
  /// Simultaneous context at each dot, not a sum of unrelated peak values.
  std::vector<SimtDotPressureSite> dotSites;

  llvm::json::Object toJSON() const;
};

struct LogicalStageCost {
  std::string id;
  std::string model;
  /// Ordered outer-to-inner split-loop shells containing this Stage.
  std::vector<std::string> enclosingLoopStages;
  StageScheduleKind schedule = StageScheduleKind::StraightLine;
  int64_t iterationCount = 1;
  StageModelFeatures features;
  StageWorkload workload;
  int64_t ownedOperationCount = 0;
  /// Unique source locations of the TTIR operations owned by this Stage.
  std::vector<std::string> sourceLocations;
  int64_t liveInCount = 0;
  int64_t liveOutCount = 0;
  /// Static tensor footprint crossing the Stage boundary.
  int64_t liveInBytes = 0;
  int64_t liveOutBytes = 0;
  std::optional<SimtRegisterPressure> wholeSimtRegisterPressure;
  int64_t simtUbWorkingSetBytesPerProgram = 0;
  int64_t simtUsableUbBytes = 0;
  std::vector<int64_t> simtFactorsRejectedByUb;
  /// Exact tensor footprint crossing the one compound local SIMT scope.
  int64_t scopeInputTensorBytes = 0;
  int64_t scopeOutputTensorBytes = 0;
  std::vector<unsigned> simtAnchorIndices;
  bool localSimtMaterializable = false;
  bool localSuperblockMaterializable = false;
  /// Factors legal for a whole-kernel pure-SIMT schedule.
  std::vector<int64_t> legalSimtFactors;
  /// Factors legal when this Stage alone is materialized as a local scope.
  std::vector<int64_t> localSimtFactors;
  /// Analytical local-SIMT body estimates, including Stages with no scope
  /// anchor.
  std::vector<StageImplementationCost> simtCalibrationEstimates;
  std::vector<StageImplementationCost> implementations;

  llvm::json::Object toJSON(bool includeDiagnosticImplementations = true) const;
};

/// Profile-driven completion terms which need whole-route structure.
struct StageRouteCompletionProfile {
  struct MultiResultPrefixScan {
    bool enabled = false;
    int64_t minimumPrograms = 0, maximumPrograms = 0;
    int64_t minimumPaddedAxis = 0, maximumPaddedAxis = 0;
    int64_t requiredResultCount = 0, requiredIndependentSequences = 0;
    double minimumActiveRatio = 0.0, maximumActiveRatio = 0.0;
    double elapsedInterceptCycles = 0.0;
    double elapsedSqrtProgramCycles = 0.0;
    double elapsedInactiveRatioCycles = 0.0;
    double elapsedPaddedAxisCycles = 0.0;
    double fixedBoundaryCycles = 0.0;
    std::string evidence;
    bool isValid() const;
  } multiResultPrefixScan;

  struct PairwiseSelectionNetwork {
    bool enabled = false;
    int64_t minimumInputElements = 0, maximumInputElements = 0;
    int64_t minimumOutputColumns = 0, maximumOutputColumns = 0;
    int64_t minimumReductionStages = 0;
    double interceptCycles = 0.0, inputElementCycles = 0.0;
    double log2OutputColumnCycles = 0.0;
    std::string evidence;
    bool isValid() const;
  } pairwiseSelectionNetwork;

  struct LargeSequenceMinmaxScan {
    bool enabled = false;
    int64_t minimumIndependentSequences = 0;
    int64_t maximumIndependentSequences = 0;
    int64_t minimumExtent = 0, maximumExtent = 0;
    double perIndependentSequenceSystemCycles = 0.0;
    std::string evidence;
    bool isValid() const;
  } largeSequenceMinmaxScan;

  struct MultiSequenceMinmaxScan {
    bool enabled = false;
    int64_t minimumIndependentSequences = 0;
    int64_t maximumIndependentSequences = 0;
    int64_t minimumExtent = 0, maximumExtent = 0;
    double baseReductionSystemCycles = 0.0;
    double perIndependentSequenceReductionSystemCycles = 0.0;
    double maximumReductionSystemCycles = 0.0;
    int64_t largeMinimumIndependentSequences = 0;
    int64_t largeMaximumIndependentSequences = 0;
    int64_t largeMinimumExtent = 0, largeMaximumExtent = 0;
    double largeBaseReductionSystemCycles = 0.0;
    double largePerIndependentSequenceReductionSystemCycles = 0.0;
    double largeMaximumReductionSystemCycles = 0.0;
    std::string evidence;
    bool isValid() const;
  } multiSequenceMinmaxScan;

  struct StraightLineAddReduction {
    bool enabled = false;
    int64_t minimumRows = 0, maximumRows = 0;
    int64_t minimumExtent = 0, maximumExtent = 0;
    double sharedCompletionSystemCycles = 0.0;
    std::string evidence;
    bool isValid() const;
  } straightLineAddReduction;

  struct TwoLoopSingleReduction {
    bool enabled = false;
    int64_t minimumSourcePrograms = 0, maximumSourcePrograms = 0;
    int64_t minimumRuntimeIterations = 0, maximumRuntimeIterations = 0;
    int64_t requiredReductionExtent = 0;
    double firstIterationSystemCycles = 0.0;
    double additionalIterationSystemCycles = 0.0;
    std::string evidence;
    bool isValid() const;
  } twoLoopSingleReduction;

  bool isValid() const;
};

struct StageCostTable {
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  int64_t logicalProgramCountHint = 0;
  /// Launch count before backend row coalescing.
  int64_t sourceLogicalProgramCountHint = 0;
  /// AIV/Vector-core count used by pure-AIV routes.
  int64_t physicalCoreCountHint = 0;
  /// AIC core count used by Cube-bearing routes.
  int64_t physicalAiCoreCountHint = 0;
  int64_t cvAivSubblocks = 1;
  int64_t baseSimtWarpCount = 1;
  /// Per-transfer MTE2 program establishment cost from the SIMD hardware
  /// profile.
  double simdLoadOperationStartupCycles = 0.0;
  /// One-time kernel establishment cost for a group of independent SIMD load
  /// streams.
  double simdIndependentLoadGroupKernelStartupCycles = 0.0;
  /// Completion-inclusive backend-helper rate for a straight-line SIMT scan.
  double simtStraightLinePrefixScanElementsPerCycle = 0.0;
  std::string simtStraightLinePrefixScanCompletionEvidence;
  std::vector<WholeSimdCompletionCurve> wholeSimdCompletionCurves;
  std::vector<WholeSimtCompletionCurve> wholeSimtCompletionCurves;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simdGridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simdGridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry;
  std::string simdGridStrideIndirectDotCompositionEvidence;
  int64_t simdIrregularTileLargeWorkingSetFullPhysicalMinimumBytes = 0;
  std::string simdIrregularTileCompletionEvidence;
  int64_t simtGridStrideResidentMemoryMaximumBytes = 0;
  double simtGridStrideResidentLoadWarpInstructionsPerCycle = 0.0;
  double simtGridStrideResidentStoreWarpInstructionsPerCycle = 0.0;
  std::string simtGridStrideResidentMemoryEvidence;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simtGridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simtGridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry;
  std::string simtGridStrideIndirectDotCompositionEvidence;
  bool simtTwoPassRowwiseOwnsSegmentedIndirectLoads = false;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simtTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot;
  std::string simtTwoPassRowwiseCompletionEvidence;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simdTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot;
  std::string simdTwoPassRowwiseCompletionEvidence;
  struct SingleResultReductionCompletion {
    bool enabled = false;
    int64_t minimumExtent = 0, maximumExtent = 0;
    int64_t minimumIterations = 0, maximumIterations = 0;
    int64_t minimumPrograms = 0, maximumPrograms = 0;
    double iterationCycles = 0.0;
    double startupInterceptCycles = 0.0;
    double startupSqrtProgramCycles = 0.0;
    std::string evidence;
  } simdSingleResultReduction, simtSingleResultReduction;
  StageRouteCompletionProfile routeCompletions;
  bool mixedParentMemoryCompletionEnabled = false;
  int64_t mixedParentMemoryMinimumIterations = 0;
  int64_t mixedParentMemoryReferenceBytesPerProgram = 0;
  double mixedParentMemoryCyclesPerExcessIteration = 0.0;
  std::string mixedParentMemoryCompletionEvidence;
  std::vector<LogicalStageCost> stages;
  std::vector<StageDependency> dependencies;
  /// Apply the generic predecessor/resource schedule to recurrence loops.
  bool stageDependencyScheduleEnabled = false;
};

struct ScopeHandoffCost {
  /// Completion-inclusive cost of entering and leaving one empty local SIMT
  /// scope.
  double fixedScopeCycles = 0.0;
  double cvCrossCoreSynchronizationCycles = 0.0;
  /// Fraction of the ideal CV payload speedup that is realized.
  double cvPartitionPayloadEfficiency = 1.0;
  /// Incremental aggregate boundary-copy rates measured with the exact scope
  /// lowering.
  double inputHandoffBytesPerCycle = 1.0;
  double outputHandoffBytesPerCycle = 1.0;

  bool isValid() const;
  double fixedCyclesForAivSubblocks(int64_t aivSubblocks) const;
  /// Data movement caused by the scope boundary.
  double estimateScopeBoundaryData(int64_t inputBytes, int64_t outputBytes,
                                   int64_t superblockFactor) const;
  /// Backward-compatible combined query: fixed scope envelope plus boundary
  /// tensor data.
  double estimateScopeHandoff(int64_t inputBytes, int64_t outputBytes,
                              int64_t superblockFactor) const;
  llvm::json::Object toJSON() const;
};

struct StageRouteDiagnostics {
  using Value = std::variant<bool, int64_t, double, std::string>;
  std::map<std::string, Value> values;

  template <typename T> T &value(llvm::StringRef key) {
    auto [it, inserted] = values.try_emplace(key.str(), T{});
    assert(inserted || std::holds_alternative<T>(it->second));
    return std::get<T>(it->second);
  }

  template <typename T> const T &value(llvm::StringRef key) const {
    static const T empty{};
    auto it = values.find(key.str());
    if (it == values.end())
      return empty;
    assert(std::holds_alternative<T>(it->second));
    return std::get<T>(it->second);
  }

  void appendTo(llvm::json::Object &result) const;
};

struct StageRuntimeSchedule {
  int64_t schedulingSlotCount = 0;
  int64_t logicalProgramsPerSchedulingSlot = 0;
  int64_t fullGroupCount = 0;
  int64_t tailProgramCount = 0;
  int64_t loopIterationCount = 0;
  std::string executionDomain;
  std::string tailMode;
};

struct StageRoutePlan {
  StageKernelRouteKind candidate = StageKernelRouteKind::AllSIMD;
  bool legal = false;
  bool diagnosticCostAvailable = false;
  /// Structural backend-lowerability reason when this candidate is rejected
  /// before cost comparison. Empty for ordinary missing-implementation cases.
  std::string ineligibilityReason;
  std::vector<StageImplementation> implementations;
  std::vector<double> scopeHandoffCycles;
  std::vector<double> logicalStageCycles;
  int64_t routeSuperblockFactor = 1;
  /// Number of launched physical blocks on the selected route.
  int64_t runtimePhysicalProgramCount = 0;
  /// Number of execution slots used to divide logical-program ownership.
  int64_t runtimeSchedulingSlotCount = 0;
  int64_t runtimeLogicalProgramsPerSchedulingSlot = 0;
  int64_t runtimeFullGroupCount = 0;
  int64_t runtimeTailProgramCount = 0;
  /// Per-Stage execution-domain schedule.
  std::vector<StageRuntimeSchedule> runtimeStages;
  int64_t runtimeLoopIterationCount = 1;
  StageRouteDiagnostics routeDiagnostics;
  double totalCycles = 0.0;

  template <typename T> T &diagnostic(llvm::StringRef key) {
    return routeDiagnostics.value<T>(key);
  }
  template <typename T> const T &diagnostic(llvm::StringRef key) const {
    return routeDiagnostics.value<T>(key);
  }

  llvm::json::Object toJSON() const;
};

struct StageCostModelSummary {
  bool applied = false;
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  std::vector<LogicalStageCost> stages;
  ScopeHandoffCost scopeHandoff;
  std::vector<StageDependency> dependencies;
  StageRoutePlan allSimd;
  StageRoutePlan allSimt;
  /// Auditable pure-SIMT plans for each legal whole-kernel factor.
  std::map<int64_t, StageRoutePlan> allSimtFactorPlans;
  StageRoutePlan mixed;
  /// Fixed parent context for comparing individual SIMT Stage bodies. This
  /// does not describe an all-SIMT executable (which may have a different Q).
  int64_t calibrationLogicalPrograms = 0;
  int64_t calibrationSchedulingSlots = 0;
  int64_t calibrationAivPartitions = 1;

  llvm::json::Object toJSON(bool includeDiagnosticImplementations = true) const;
};

llvm::Expected<StageCostModelSummary>
solveStageRoutes(const StageCostTable &costTable,
                 const ScopeHandoffCost &scopeHandoff);

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H
