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
#include <map>
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
  /// Physical SIMD instruction width.  Unlike vectorWidth, this preserves
  /// bytes/bits so FP16 and FP32 short-axis segments are priced correctly.
  int64_t vectorWidthBits = 1;
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
  /// Element-unit dependency factor for 1D tt.scan (S = N), used by the SIMT
  /// legacy pricing and by the SIMT legacy refund in the standalone PrefixScan
  /// model (the SIMT 1D price comes from the warp segment table).
  /// The SIMD 1D price is built from the two work factors below instead, so
  /// this value is only a placeholder for SIMD profiles.  Falls back to
  /// prefixScanDependencyFactor when the profile omits it.
  double prefixScanDependencyFactor1d = 1.0;
  /// Dependency factor for the extent <= 64 scalar-register 1D tt.scan
  /// segment (sklansky_regbuf_16: 16 elements in 16 scalar registers, pure
  /// SCALAR pipe).  Its work is O(N) - the network is capped at 16 elements and
  /// chained by a serial carry - and its unit cost is several times
  /// higher than the rvec path above 64, so it consumes its own factor.
  /// Falls back to prefixScanDependencyFactor1d when the profile omits it.
  double prefixScanDependencyFactor1dSmall = 1.0;
  /// Dependency factor of the 64 < extent <= 1024 1D tt.scan regime, in
  /// Sklansky element-rounds: one unit per element per network round, so the
  /// priced feature is sum(N * log2(N)).  Falls back to
  /// prefixScanDependencyFactor (the multi-dim factor) when the profile omits
  /// it, which conservatively keeps the rvec 1D pool priced as multi-dim scan
  /// work instead of under-pricing it.
  double prefixScanDependencyFactor1dMidWork = 1.0;
  /// Same, for the extent > 1024 tiled regime: 1024-element blocks plus a
  /// serial cross-block carry (a slightly steeper work slope).
  double prefixScanDependencyFactor1dTiledWork = 1.0;
  /// Extra dependency factor charged on the multi-dim column-parallel steps
  /// that sit above the SIMD hinge (S = 256): the 2D SIMD cost is piecewise
  /// linear in S, so the tail steps pay `prefixScanDependencyFactor` + this
  /// value.  Unit: factor per step, i.e. cyc/step = value / 64.  Default 0
  /// keeps a single-slope profile.  SIMD only: the SIMT multi-dim price comes
  /// from its own warp segment table.
  double prefixScanDependencyFactorMultiTail = 0.0;
  /// Transpose cost for the multi-dim SIMD scan, as a fixed part plus a byte
  /// rate: `cycles = fixed + bytes / bytesPerCycle`.  Cumsum.cpp transposes a
  /// non-leading scan axis into position 0 and back, moving the whole tensor
  /// twice.  The body is transpose_dim_01 for rank >= 3 and transpose_ar2ra
  /// (gather/scatter) for rank 2.  The rate is NOT constant in the tensor size:
  /// transpose_ar2ra never reaches its asymptotic bandwidth on small tiles (its
  /// gather mask only fills min(M, 64) of the 64 lanes and the per-iteration
  /// overhead dominates), so the fixed part absorbs the per-call overhead and a
  /// profile that calibrates only the rate keeps the legacy pure-rate
  /// behaviour.  Defaults of 0 charge no transpose at all.  SIMD only.
  double prefixScanTransposeDim01FixedCycles = 0.0;
  double prefixScanTransposeDim01BytesPerCycle = 0.0;
  double prefixScanTransposeAr2raFixedCycles = 0.0;
  double prefixScanTransposeAr2raBytesPerCycle = 0.0;
  /// Fixed per-scan-execution cost, charged once per present scan segment
  /// (multi-dim scan, 1D rvec segment, 1D scalar-register segment all have
  /// their own).  Both modes pay it: SIMT pays a thread barrier, a fixed
  /// launch shape and a UB round-trip, and the SIMD 2D library template pays
  /// a fixed issue/setup cost as well.
  double prefixScanStartupCycles = 0.0;
  /// Startup for the rvec 1D regimes (extent > 64): the block-invocation setup
  /// of cce::async_invoke, shared by the single-block and tiled buckets.
  double prefixScanStartupCycles1d = 0.0;
  /// Startup for the extent <= 64 scalar-register 1D regime (library call plus
  /// the pipe barrier around it).
  double prefixScanStartupCycles1dSmall = 0.0;
  /// v3.1 warp-shape-aware three-segment pricing for SIMT 1D scans.  The
  /// ScanOpToLLVM lowering switches structure at the warp boundary, so the
  /// per-scan cost is a function of the element count N and the kernel's
  /// num_warps (w) rather than a factor * laneSteps product:
  ///   N <= 32:                cLocal            (single-warp fast path)
  ///   32 < N <= min(32*w, 1024):
  ///                           cFixed + cRound * ceil(log2 k)
  ///                                             (cross-warp Sklansky merge;
  ///                                              k = ceil(N/32) is the number
  ///                                              of axis warps, so ceil(log2
  ///                                              k) is exactly the trip count
  ///                                              of the Sklansky `for (h=1;
  ///                                              h<k; h<<=1)` loop.  The 1024
  ///                                              cap is
  ///                                              canUseMultiWarpSklansky's
  ///                                              k <= warpSize gate: a
  ///                                              num_warps = 64 kernel leaves
  ///                                              the Sklansky path at N =
  ///                                              2048)
  ///   N > min(32*w, 1024):     cGen + serialRate * N
  ///                                             (general path, serial adds;
  ///                                              the per-element slope is
  ///                                              charged from N = 0, i.e. the
  ///                                              breakpoint is folded into
  ///                                              cGen)
  /// The structural boundaries are axisNumWarps > 1 (N = 32), k > warpSize
  /// (N = 1024) and getAxisNumElementsPerThread() == 1 (N = 32*w); the
  /// cross-warp merge is one code path, so it is priced by one segment.
  /// Key = warp bucket (2/4/8/16/32/64); empty table falls back to the legacy
  /// factor formula above.  The "2" bucket is absent on purpose: num_warps = 2
  /// reuses the "4" bucket through lower_bound.
  struct PrefixScan1dWarpSegment {
    double cLocal = 0.0;
    double cFixed = 0.0;
    double cRound = 0.0;
    double cGen = 0.0;
    double serialRate = 0.0;
  };
  std::map<unsigned, PrefixScan1dWarpSegment> prefixScan1dWarpSegments;
  /// v3.2 warp-shape-aware pricing for SIMT multi-dim (2D) scans.  The
  /// ScanOpToLLVM generic scan lowering (AddPartialReduce) distributes the
  /// scan across warps with a serial UB load chain whose cost is additive in
  /// the total element count, so the per-scan cost is
  ///   a + b * N + c * (N * k) / T + d * max(0, N / T - r0)
  /// where N = extent * columns (total elements), T = 32 * num_warps, and
  /// k = warpsPerCTA[axis] = min(num_warps, ceil(extent / 32)) is the
  /// AddPartialReduce UB chain length.  b * N prices the per-element serial
  /// add in the thread-local scan, c * (N * k) / T prices the dependent UB
  /// partial loads (k loads per element chunk, N / T elements per thread),
  /// and the hinge prices per-thread register pressure beyond r0
  /// elements/thread.  Key = warp bucket (4/8/16/32) selected by the kernel
  /// warp count; empty table falls back to the legacy factor formula.
  /// Extents below one warp (extent < 32) exercise a sub-warp scan whose cost
  /// is a separate linear-in-N regime: the axis warp holds fewer than 32 valid
  /// lanes, so the chain/hinge terms above do not apply and a dedicated
  /// a_subwarp + b_subwarp * N form is used (see the profile description).
  /// Every coefficient is constrained >= 0 so the cost is monotone in N and
  /// in k; a bucket may degenerate to fewer effective terms (exact zeros).
  struct PrefixScan2dWarpSegment {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    double r0 = 0.0;
    /// Sub-warp regime (extent < 32): cost = aSubwarp + bSubwarp * N.  Both 0
    /// (absent key) keeps the warp-aligned formula for every extent.
    double aSubwarp = 0.0;
    double bSubwarp = 0.0;
  };
  std::map<unsigned, PrefixScan2dWarpSegment> prefixScan2dWarpSegments;
  double dotSetupCycles = 0.0;
  double dotFlopsPerCycle = 0.0;
  double scalarOperationsPerCycle = 0.0;
  double issueOperationsPerCycle = 0.0;
  double spillTransactionsPerCycle = 0.0;
  /// Scalar white-box terms in SYS_CNT cycles (CAModel 1.8GHz + 988.9/1800).
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
