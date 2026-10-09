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
  bool hasAtomicMemory = false;
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

/// Route-independent semantic description of one dynamically executed TTIR
/// atomic operation.  This deliberately preserves information that ordinary
/// store byte counts cannot represent.  Physical instruction/transaction
/// counts remain a route-specific profile concern.
struct AtomicWorkload {
  std::string kind;
  std::string dataType;
  std::string memorySemantic;
  std::string memoryScope;
  double logicalElements = 0.0;
  double logicalOperationInstances = 0.0;
  /// Zero means that TTIR did not prove a contiguous address run width.  Do
  /// not infer it from tensor shape alone: a shaped pointer may still contain
  /// arbitrary loaded indices.
  double provenContiguousRunWidth = 0.0;
  bool resultUsed = false;
  bool addressDependsOnLoadedIndex = false;
  /// Runtime pointer values can alias even when SSA has no data recurrence.
  /// Until an injectivity proof or launch hint exists, route scoring must use
  /// the profile's explicit unknown-contention policy.
  bool contentionUnknown = true;

  std::string profileKey() const;
  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// A compact group of tensor operations with the same route-independent
/// projected lowering signature. Contiguous pointwise axes are merged before
/// counting vector instructions; broadcast boundaries retain separate segments.
/// Equal signatures are aggregated, so
/// this is diagnostic/cost state rather than an op-level graph.
struct TensorOperationWorkload {
  std::string operation;
  int64_t elementBitWidth = 0;
  double logicalElements = 0.0;
  double segmentCount = 0.0;
  int64_t contiguousElementsPerSegment = 0;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// Mode-independent work owned exactly once by one Stage.  Values are
/// logical elements/bytes, not mode-specific instructions or cycles.
struct StageWorkload {
  llvm::StringMap<double> operationElements;
  std::vector<TensorOperationWorkload> tensorOperationWorkloads;
  double scalarOperations = 0.0;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  /// Subsets of the total load/store fields whose address depends on loaded
  /// data (or is an explicit gather).  Keeping totals and subsets preserves
  /// report compatibility while allowing direct and indirect work to be
  /// priced independently.  Atomic RMW work is never included in store totals.
  double indirectLoadBytes = 0.0;
  double indirectStoreBytes = 0.0;
  double indirectLoadTransactions = 0.0;
  double indirectStoreTransactions = 0.0;
  std::vector<AtomicWorkload> atomicWorkloads;
  double predicateElements = 0.0;
  double shuffleLaneSteps = 0.0;
  /// Portion of shuffleLaneSteps contributed by tt.scan (prefix-scan class).
  /// The remainder is contributed by tt.reduce.  Only the scan portion
  /// consumes the prefix-scan dependency factor inside recurrence stages.
  /// The 1D contribution is recorded in the regime's own cost unit: element
  /// count for extent <= 64 and Sklansky element-rounds above it (see the
  /// bucket fields below), so the refund terms below cancel it exactly.
  double scanShuffleLaneSteps = 0.0;
  /// Element count of the 1D tt.scan portion (S = N).  Consumed by the SIMT
  /// segment table, whose N key and unit costs are in element units, and by
  /// the element-unit fallback pricing; the SIMD 1D price is built from the
  /// bucket fields below.
  double scanShuffleLaneSteps1d = 0.0;
  /// Portion of scanShuffleLaneSteps1d contributed by 1D tt.scan whose scan
  /// extent is <= 64 elements.  The 1D cumsum library symbol
  /// (_mlir_ciface_cumsum_1d_<dtype>_dim0) is owned by CumsumSimtSklansky.cpp,
  /// whose first dispatch is on the element count: L <= 64 runs
  /// sklansky_regbuf_16, a pure scalar-register Sklansky network (16 elements
  /// in 16 scalar registers, 3 adds per element, no vector instruction at
  /// all).  That network is capped at 16 elements and the blocks are chained by
  /// a serial carry, so this regime's work is O(N) rather than the O(N log N)
  /// of a full-length Sklansky network.  Its unit cost is several times
  /// higher than the warp/block path above 64, so it consumes its own
  /// dependency factor.  Dtype independent (element count based).
  double scanShuffleLaneSteps1dSmall = 0.0;
  /// Sklansky *work* of the two rvec 1D regimes, in element-rounds:
  /// sum(N * log2(N)) over the scans of the bucket.  Above 64 elements the
  /// template runs a 32-lane block scan whose per-element round count is
  /// log2(N) (5 intra-warp rounds plus ceil(log2 k) cross-warp rounds,
  /// k = ceil(N/32)), so the cost driver is N*log2(N) and not N.  The two
  /// regimes share one intercept (the block-invocation setup) but
  /// have their own slope: 64 < extent <= 1024 runs a single block, while
  /// extent > 1024 tiles into 1024-element blocks and pays a serial
  /// cross-block carry.  Both are element-count based and dtype independent.
  double scanShuffleLaneSteps1dMidWork = 0.0;
  double scanShuffleLaneSteps1dTiledWork = 0.0;
  /// The 1D scan has no separate SIMD-view step field: none of the three 1D
  /// regimes packs elements into a vector lane, so their unit costs are dtype
  /// independent.  (The multi-dim column-parallel path still divides by BpE.)
  /// Total-element (N = extent * columns) count of the multi-dim tt.scan
  /// portion, consumed by the SIMT lowering.  SIMT lowers the multi-dim scan
  /// against the total element count (per-thread lane work), whereas the SIMD
  /// lowering is column-parallel with S = extent * ceil(columns / BpE).  The
  /// shared scanShuffleLaneSteps field keeps the SIMD column-parallel count so
  /// SIMD pricing is unchanged; the SIMT resource computation swaps the
  /// multi-dim portion for this total-element count.  Zero for 1D scans
  /// (S = N in both modes).
  double scanShuffleLaneStepsSimtMulti = 0.0;
  /// Sum of scan extents over the multi-dim tt.scan portion.  The warp-aware
  /// SIMT multi-dim pricing derives the AddPartialReduce UB chain length
  /// k = min(num_warps, ceil(extent/32)) from it, so it cannot be recovered
  /// from N alone.  A stage holding k scans prices them as one aggregate scan
  /// with E = sum(extents) (exact for one scan).
  double scanSimtMultiExtentSum = 0.0;
  /// Portion of the multi-dim column-parallel steps that sit above the
  /// SIMD hinge (S = extent * ceil(columns/BpE) = 256).  The 2D SIMD cost is
  /// piecewise linear in that S, so these steps are charged an extra factor;
  /// the hinge is on S, not on the extent.  Same unit as
  /// scanShuffleLaneSteps (column-parallel steps), so it is a subset of it.
  double scanShuffleLaneStepsMultiTail = 0.0;
  /// Bytes moved by the two transposes the SIMD cumsum template pays when the
  /// scan axis is not the leading axis (Cumsum.cpp: the template transposes the
  /// scan axis into position 0, scans, and transposes back, so both directions
  /// move the whole tensor).  Only multi-dim scans whose axis != 0 have it: a
  /// 1D scan is already leading-axis, and axis == 0 needs no transpose.  The
  /// value is 2 * elementCount * sizeof(elementType).  SIMT has no such term -
  /// its lowering does not go through the template - so the SIMD price is the
  /// only consumer.
  double scanTransposeBytes = 0.0;
  /// Subset of scanTransposeBytes contributed by rank-2 scans.  The template
  /// transposes rank >= 3 with transpose_dim_01 but rank 2 with
  /// transpose_ar2ra, a gather/scatter implementation, so the two need separate
  /// rates.
  double scanTransposeBytesRank2 = 0.0;
  double dotFlops = 0.0;
  double issueElements = 0.0;
  double estimatedSpillTransactions = 0.0;
  /// Scalar GM operations owned by this Stage per iteration.
  double scalarLoadCount = 0.0;
  double scalarStoreCount = 0.0;
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
  double atomic = 0.0;
  double compute = 0.0;
  double predicate = 0.0;
  double shuffle = 0.0;
  /// Cycles for the tt.scan-contributed portion of shuffle at the ideal rate.
  double scanShuffle = 0.0;
  /// Cycles for the 1D tt.scan-contributed portion at the ideal rate.
  double scanShuffle1d = 0.0;
  /// Cycles for the extent <= 64 scalar-register 1D tt.scan portion at the
  /// ideal rate (sklansky_regbuf_16), in element units.
  double scanShuffle1dSmall = 0.0;
  /// Sklansky-work view of the two rvec 1D regimes: the same portions at the
  /// ideal rate but in element-round units (sum(N * log2(N)) / lane rate), so
  /// the SIMD 1D refund cancels the work its base charge actually used.
  double scanShuffle1dMidWork = 0.0;
  double scanShuffle1dTiledWork = 0.0;
  /// Column-parallel steps above the multi-dim SIMD hinge, at the ideal rate.
  /// Charged an extra factor on top of the base dependency factor, because the
  /// 2D SIMD cost per step is higher once the aggregate scan steps pass 256.
  double scanShuffleMultiTail = 0.0;
  /// Cycles for the multi-dim SIMD scan transposes at their byte rates.
  /// Already in cycles (the rates are cyc/byte), so it is added to the scan
  /// critical path unscaled - a transpose is pure data movement with no
  /// dependency-factor amortization.
  double scanTranspose = 0.0;
  /// Fixed per-scan-execution cycles (SIMT thread barrier, fixed thread
  /// launch shape, UB round-trip) summed over the present scan segments.
  double scanStartup = 0.0;
  double dot = 0.0;
  double loopControl = 0.0;
  double branchControl = 0.0;
  double divergence = 0.0;
  double synchronization = 0.0;
  double spill = 0.0;
  double issue = 0.0;
  double criticalPath = 0.0;
  /// Per-iteration cycles attributed to each execution pipe, reported for
  /// diagnostics only.  load/store/atomic run on MTE, compute/predicate/
  /// shuffle/dot and the scan network on VEC, scalar/control/spill on SI; a
  /// loop-carried recurrence is a single dependency chain and reports its body
  /// as pipeSerial instead.  Route aggregation does NOT fold these in - see the
  /// comment on solveStageRoutes for why overlapping them across Stages is
  /// wrong.  They exist to locate which pipe a Stage's price is going
  /// to when a route prediction is off.
  double pipeVec = 0.0;
  double pipeMte = 0.0;
  double pipeSi = 0.0;
  double pipeSerial = 0.0;

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
  /// Fixed cost of one complete local SIMD -> SIMT -> SIMD scope pair.
  /// This is an operational pair measurement, not a directional latency.
  double fixedPairCycles = 0.0;
  /// Local scope values cross the SIMD/SIMT register-file boundary through
  /// UB.  SIMD rates are aggregate vector-pipeline rates; SIMT rates are
  /// explicitly per active thread and are aggregated over one logical warp.
  double simdUbLoadBytesPerCycle = 1.0;
  double simdUbStoreBytesPerCycle = 1.0;
  double simtUbLoadBytesPerThreadPerCycle = 1.0;
  double simtUbStoreBytesPerThreadPerCycle = 1.0;
  int64_t simtWarpSize = 1;

  bool isValid() const;
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
