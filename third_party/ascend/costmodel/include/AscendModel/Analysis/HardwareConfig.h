//===- HardwareConfig.h - Hardware Configuration Interface ------*- C++ -*-===//
//
// This file defines the HardwareConfig class for loading and querying
// hardware parameters from JSON configuration files.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_HARDWARECONFIG_H
#define ASCENDMODEL_HARDWARECONFIG_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

//===----------------------------------------------------------------------===//
// Memory Space
//===----------------------------------------------------------------------===//

enum class MemoryType {
  OffChip,      // e.g., HBM
  OnChipShared, // e.g., L2
  OnChipLocal,  // e.g., L1, UB
  RegisterFile  // e.g., L0A, L0B, L0C
};

struct MemorySpace {
  std::string name;
  MemoryType type;
  size_t sizeBytes; // Total size in bytes
  double bandwidthBytesPerCycle;
  int latencyCycles;
  std::string description;

  // Convenience methods
  double sizeKB() const { return sizeBytes / 1024.0; }
  double sizeMB() const { return sizeBytes / (1024.0 * 1024.0); }
  double sizeGB() const { return sizeBytes / (1024.0 * 1024.0 * 1024.0); }
};

//===----------------------------------------------------------------------===//
// Compute Unit
//===----------------------------------------------------------------------===//

enum class ComputeUnitType {
  MatrixEngine, // e.g., Cube
  SIMDEngine,   // e.g., Vector
  ScalarEngine
};

/// Fractal (tile) dimensions for matrix operations
/// Ascend Cube Core computes C[m,n] += A[m,k] * B[k,n]
struct FractalSize {
  int m = 16; // Output rows
  int k = 16; // Reduction dimension
  int n = 16; // Output columns
};

struct ComputeUnit {
  std::string name;
  ComputeUnitType type;

  // Performance specs
  double tflopsFP16;
  double tflopsFP32;
  double tflopsINT8;

  // Matrix engine specific - default tile size (for backward compatibility)
  int tileM, tileN, tileK;

  // Fractal sizes per data type (key: "fp16", "bf16", "fp32", "int8")
  llvm::StringMap<FractalSize> fractalSizes;

  std::vector<std::string> inputSpaces;
  std::string outputSpace;

  // SIMD engine specific
  int widthElements; // Number of elements processed per cycle
  int widthBytes;
  std::string computeSpace;

  std::vector<std::string> supportedOps;
  std::vector<std::string> supportedDtypes;
};

//===----------------------------------------------------------------------===//
// Data Mover
//===----------------------------------------------------------------------===//

struct DataMover {
  std::string name;
  std::string description;
  std::string srcSpace;
  std::vector<std::string> dstSpaces;
  double bandwidthBytesPerCycle;
  int maxBurstBytes;
  int alignmentBytes;
  bool supportsAccumulate;
  bool supportsCast;
};

//===----------------------------------------------------------------------===//
// tilesim-style micro-architecture tables (migrated from tilesim 910B1)
//===----------------------------------------------------------------------===//

/// One row of the per-(intrinsic, dtype) vector instruction cycle table.
/// Mirrors tilesim VecIntrinsics: computing/head/interval cycles per repeat.
struct VecCycleEntry {
  int compute = 0;
  int head = 0;
  int interval = 0;
};

/// Small-packet bandwidth fitting coefficients (tilesim pkt_param).
/// Applied when the transfer packet size is below ``thresholdBytes``.
struct SmallPacketCoeffs {
  bool enabled = false;
  int thresholdBytes = 256;
  // Bucket selection mirrors tilesim: <64B -> "64B", <128B -> "128B",
  // <256B -> "256B". Each bucket has (a, b) fitting coefficients.
  double a64 = 0, b64 = 0;
  double a128 = 0, b128 = 0;
  double a256 = 0, b256 = 0;
};

/// A bandwidth entry keyed by "src:dst" memory spaces.
/// Either core-independent (singleGbps) or per-core (degrades as more cores
/// contend for the same off-chip port, e.g. GM<->UB on 910B).
struct BandwidthTable {
  bool hasPerCore = false;
  double singleGbps = 0.0;           // core-independent bandwidth (GB/s)
  std::map<int, double> perCoreGbps; // core_num -> GB/s (1..N)
  SmallPacketCoeffs smallPacket;
};

/// Cube (GEMM) micro-architecture parameters migrated from tilesim
/// (cube_throughput, cube_repeat_cycles, L0 tile limit for best_k0).
struct CubeModelConfig {
  // basic throughput per repeat: [basicM, basicK, basicN].
  // tilesim stores cube_throughput as [16, 32, 16]; the K value is a
  // numerator divided by the element byte size, i.e. basicK = 32/elemBytes
  // (fp16->16, fp32->8, int8->32), which equals the fractal K dimension.
  int basicM = 16;
  int basicKNumerator = 32; // basicK = basicKNumerator / (elemBytes)
  int basicN = 16;
  // repeat cycles per dtype key ("fp32","fp16","bf16","int8")
  llvm::StringMap<int> repeatCycles;
  // L0 tile footprint limit (bytes) controlling best_k0 selection.
  int l0TileLimitKb = 32;
};

//===----------------------------------------------------------------------===//
// Pipeline Stage
//===----------------------------------------------------------------------===//

struct PipelinePath {
  std::string name;
  std::vector<std::string> stages;
  std::string description;
};

//===----------------------------------------------------------------------===//
// HardwareConfig
//===----------------------------------------------------------------------===//

/// Result of a bandwidth lookup, mirroring tilesim ``lookup_bw``.
/// ``bwGBs`` is the per-(src,dst,corenum,pkt) bandwidth in GB/s using the
/// tilesim GiB convention (1 GB = 1024^3 B). ``isSmallPacket`` records
/// whether the small-packet fitting branch was taken.
struct BandwidthEntry {
  double bwGBs = 0.0;
  bool isSmallPacket = false;
};

class HardwareConfig {
public:
  HardwareConfig();
  ~HardwareConfig();

  // Factory methods
  static std::unique_ptr<HardwareConfig> loadFromFile(llvm::StringRef path);
  static std::unique_ptr<HardwareConfig>
  loadFromJSON(const llvm::json::Value &json);
  static std::unique_ptr<HardwareConfig> getDefault910B();

private:
  static std::unique_ptr<HardwareConfig> createHardcodedDefault910B();

public:
  // Basic info
  llvm::StringRef getName() const { return name; }
  llvm::StringRef getVendor() const { return vendor; }

  // Clock
  double getClockFrequencyGHz() const { return clockFreqGHz; }
  int getCyclesPerMicrosecond() const {
    return static_cast<int>(clockFreqGHz * 1000);
  }
  double cyclesToMicroseconds(int64_t cycles) const {
    return static_cast<double>(cycles) / (clockFreqGHz * 1000.0);
  }

  // Memory spaces
  const MemorySpace *getMemorySpace(llvm::StringRef name) const;
  double getMemoryBandwidthTBps(llvm::StringRef name) const;
  double getMemoryBandwidthBytesPerCycle(llvm::StringRef name) const;
  int getMemoryLatencyCycles(llvm::StringRef name) const;
  size_t getMemorySizeBytes(llvm::StringRef name) const;
  std::vector<std::string> getMemorySpaceNames() const;

  // Compute units
  const ComputeUnit *getComputeUnit(llvm::StringRef name) const;

  // Cube (matrix engine)
  double getCubeTFlopsFP16() const;
  double getCubeTFLOPS() const; // Alias for getCubeTFlopsFP16
  void getCubeTileSize(int &m, int &n, int &k) const;
  void getCubeFractalSize(int elementBits, int &m, int &n, int &k) const;
  llvm::StringRef getCubeOutputSpace() const;

  // Vector (SIMD engine)
  double getVectorTFlopsFP32() const;
  double getVectorTFLOPS() const; // Alias for getVectorTFlopsFP32
  int getVectorWidthElements() const;
  int getVectorWidthBytes() const;
  llvm::StringRef getVectorComputeSpace() const;
  int getVectorOpCyclesPerInstruction(llvm::StringRef opName) const;

  // HBM bandwidth (convenience)
  double getHBMBandwidthGBs() const;
  double getHBMBandwidthTBs() const;

  // Startup latencies (from hardware params, or reasonable defaults)
  int getMTE2StartupLatency() const;
  int getMTE3StartupLatency() const;
  int getFixPipeStartupLatency() const;
  int getCubeStartupLatency() const;
  int getVectorStartupLatency() const;

  // Calibrated overhead parameters (derived from profiling _attn_fwd,
  // BLOCK_M={16,32,48,64} on Ascend 910B with 20 AIC + 20 AIV cores).
  //
  // Scalar overhead: loop control, pointer arithmetic, and pipe_barrier
  // synchronisation account for 27-36% of AIV wall time and 42-48% of AIC
  // wall time.  The factor below converts pure vector/MAC cycles to the
  // estimated total path time: total_aiv ≈ vec_cycles * (1 + scalar_factor).
  // Calibrated to aiv_vec_ratio = 0.211 in steady state → factor = 3.74.
  double getAIVScalarOverheadFactor() const;

  // Number of AIC / AIV execution cores per block.
  // Profiling was run with Block Dim=20 (AIC) + 20 AIV = Mix Block Dim 40.
  int getNumAICCores() const;
  int getNumAIVCores() const;

  // Cycles spent in pipe_barrier per inner-loop iteration (AIC↔AIV sync).
  // Calibrated from the 39% idle fraction observed on AIV for BM=64, 1 wave:
  // idle_cycles ≈ 23 000 cycles / 3 iterations ≈ 7 500 cycles per iteration.
  int getPipeBarrierCyclesPerIter() const;

  // Data movers
  const DataMover *getDataMover(llvm::StringRef name) const;
  std::vector<std::string> getDataMoverNames() const;

  // Performance estimation
  int64_t estimateCubeCycles(int64_t M, int64_t N, int64_t K) const;
  int64_t estimateVectorCycles(int64_t numElements) const;

  /// Cycles for one vector instruction over ``numElements`` elements, using
  /// the migrated per-(intrinsic, dtype) cycle table:
  ///   repeats = ceil(numElements * elemBytes / 256B)
  ///   cycles  = compute(intrinsic, dtype) * repeats + vector startup
  /// ``intrinsic`` is a tilesim mnemonic such as "VADD" or "VEXP"; an unknown
  /// or empty mnemonic falls back to one cycle per repeat.
  ///
  /// Prefer this over ``estimateVectorCycles``, which charges every vector
  /// operation the same regardless of which instruction it lowers to.
  int64_t estimateVectorCyclesFromTable(int64_t numElements, int elementBits,
                                        llvm::StringRef intrinsic) const;
  int64_t estimateMemoryCycles(llvm::StringRef moverName, int64_t bytes) const;
  int64_t estimateMemoryCyclesWithLatency(llvm::StringRef space,
                                          int64_t bytes) const;

  //===--------------------------------------------------------------------===//
  // tilesim-migrated micro-architecture queries (root cause ①/②)
  //===--------------------------------------------------------------------===//

  /// Look up the per-(src,dst,corenum,pkt) bandwidth in GB/s, mirroring
  /// tilesim ``ArcConfig.lookup_bw``. ``coreNum<=0`` falls back to the
  /// hardware default active core count (see ``getActiveBandwidthCores``).
  /// ``pktBytes>0`` enables the small-packet fitting branch when configured.
  /// Unknown (src,dst) pairs fall back to the aggregate HBM bandwidth so the
  /// estimate degrades gracefully instead of failing.
  BandwidthEntry lookupBandwidth(llvm::StringRef src, llvm::StringRef dst,
                                 int coreNum, int64_t pktBytes) const;

  /// Look up the per-(intrinsic,dtype) vector cycle triplet
  /// {compute, head, interval}, mirroring tilesim ``lookup_vec_cycle``.
  /// ``elementBits`` selects the dtype bucket; unknown (intrinsic,dtype)
  /// falls back to fp32 then to a sensible per-intrinsic default.
  VecCycleEntry lookupVecCycle(llvm::StringRef intrinsic,
                               int elementBits) const;

  /// Estimate transfer cycles for moving ``bytes`` from ``src`` to ``dst``
  /// across ``coreNum`` concurrent cores, using the tilesim GiB bandwidth
  /// convention and clock frequency. Includes small-packet fitting.
  int64_t estimateTransferCycles(llvm::StringRef src, llvm::StringRef dst,
                                 int64_t bytes, int coreNum) const;

  /// Default number of concurrent cores used for bandwidth degradation of
  /// vector transfers (GM<->UB). Defaults to the 910B1 saturation point (48).
  int getActiveBandwidthCores() const { return activeBandwidthCores; }
  void setActiveBandwidthCores(int cores) { activeBandwidthCores = cores; }

  /// Whether small-packet bandwidth fitting is enabled (tilesim
  /// ``enable_small_package``). Defaults to false to match tilesim.
  bool getEnableSmallPacketBw() const { return enableSmallPacketBw; }
  void setEnableSmallPacketBw(bool v) { enableSmallPacketBw = v; }

  /// Whether two hardware units (by costmodel name, e.g. "vec_mte2"/"mte3")
  /// are mutex partners: they share a physical pipeline and cannot execute in
  /// parallel (tilesim ``MutexComponents`` / ``pipe_exclusive_config``). On
  /// 910B the AIV MTE2 (vector load) and MTE3 (vector store) share one
  /// pipeline.
  bool areMutexUnits(llvm::StringRef a, llvm::StringRef b) const;

  /// Fixed cost of crossing one synchronisation barrier, in cycles: the flag
  /// write, its propagation, and the wait that observes it. This is latency,
  /// not occupancy -- a barrier does not keep a pipe busy, it prevents the
  /// core from issuing -- so a consumer of this number belongs in a schedule
  /// and not in a roofline.
  ///
  /// Defaults to 0, i.e. off, because it has not been measured on this target
  /// and an unmeasured constant should not silently move everyone's numbers.
  /// It matters as soon as the number of barriers becomes something a search
  /// varies: with it at zero, splitting a compute block in two is free.
  int64_t getBarrierCycles() const { return barrierCycles; }
  void setBarrierCycles(int64_t cycles) { barrierCycles = cycles; }

  /// How many vector cores execute the vector work of one compute block.
  ///
  /// They split the data, not the instruction stream: on a profiled kernel both
  /// vector sub-cores of a block report the same instruction count -- 152 166
  /// MTE3 instructions each, against the 152 165 this model derives for one
  /// core -- while each moves half of every tile. A consumer therefore divides
  /// the repeat count of an instruction by this number and leaves the
  /// instruction's startup latency alone.
  ///
  /// Defaults to 1, the behaviour before the field existed, because it
  /// describes one part's core pairing and no profile should acquire it by
  /// inheritance.
  int getVectorCoresPerBlock() const { return vectorCoresPerBlock; }
  void setVectorCoresPerBlock(int cores) {
    vectorCoresPerBlock = cores >= 1 ? cores : 1;
  }

  /// Bytes one vector core consumes per cycle.
  ///
  /// vec_cycle_tables gives cycles per 256-byte repeat, so its numbers already
  /// contain a datapath width: compute=2 for a plain add is 128 B/cycle. What
  /// carries over from the part those tables were measured on is the *ratio*
  /// between intrinsics -- a divide costs four times an add there and should
  /// here too -- not the absolute count. So a consumer rescales the table by
  /// kVecTableBytesPerCycle / this, and the table itself is left as migrated.
  ///
  /// Defaults to the table's own width, which leaves every existing profile
  /// costed exactly as before.
  int getVectorBytesPerCycle() const { return vectorBytesPerCycle; }

  /// Width the migrated vec_cycle_tables were measured at: a 256-byte repeat
  /// at compute=2 is 128 bytes per cycle.
  static constexpr int kVecTableBytesPerCycle = 128;

  /// Cycles one scalar instruction occupies its core's scalar unit for.
  ///
  /// Address arithmetic, loop bookkeeping and predicate evaluation are not
  /// free: the profiler puts them at a fifth to a quarter of each core's
  /// active time, and they scale with the number of blocks rather than with
  /// the data, so they are exactly what makes a finer block partition cost
  /// something. They overlap the compute pipes rather than queueing behind
  /// them, so a consumer takes the maximum against those, not the sum.
  ///
  /// Defaults to 0, i.e. not charged, because the per-instruction figure has
  /// not been measured yet -- only the per-iteration total has. The report
  /// prints the instruction count next to that total so the division can be
  /// done from one profiled run rather than guessed at here.
  int getScalarCyclesPerInstruction() const {
    return scalarCyclesPerInstruction;
  }

  /// Fraction of a steady-state iteration that one prologue or epilogue
  /// iteration of a software-pipelined loop costs.
  ///
  /// AddControlFlowCondition turns a loop body into a chain of predicated
  /// stages and extends the bound by one iteration per stage, so the pipeline
  /// can fill and drain. Those iterations run on the hardware; they simply do
  /// less than a full one, because during the fill only the leading stages are
  /// enabled and during the drain only the trailing ones.
  ///
  /// The default is a derivation, not a fit. Filling a pipeline of S stages
  /// takes S-1 iterations in which 1, 2, ... S-1 stages are live, so the mean
  /// occupancy over the ramp is (S-1)/2S, which tends to one half; draining is
  /// the mirror image. Half is therefore what the geometry gives for any stage
  /// count, on any part, without reference to a kernel.
  ///
  /// Unlike the fields above this one does NOT default to the old behaviour.
  /// The old behaviour was to charge nothing, which makes a deeper pipeline
  /// free and lets a search buy overlap it never pays for: measured on flash
  /// attention, a variant that grew the ramp from 3 iterations to 138 scored
  /// 10% better here and ran 70% slower.
  double getPrologueWorkFraction() const { return prologueWorkFraction; }

  // Cube (GEMM) micro-architecture: migrated from tilesim cube_config.
  void getCubeModelThroughput(int elementBits, int &basicM, int &basicK,
                              int &basicN) const;
  int getCubeModelRepeatCycles(int elementBits) const;
  int getCubeModelL0TileLimitBytes() const;

  // Pipeline info
  const PipelinePath *getPipelinePath(llvm::StringRef name) const;
  bool canRunInParallel(llvm::StringRef path1, llvm::StringRef path2) const;

  // Validation
  bool validate(std::string &error) const;

  // Debug

private:
  bool parseJSON(const llvm::json::Value &json, std::string &error);

  /// Populate the tilesim-migrated tables with 910B1 measured values
  /// (hardcoded-fallback path; JSON is authoritative in production).
  void populateTilesimDefaults910B();

  std::string name;
  std::string vendor;
  std::string version;

  double clockFreqGHz;

  llvm::StringMap<MemorySpace> memorySpaces;
  llvm::StringMap<ComputeUnit> computeUnits;
  llvm::StringMap<DataMover> dataMovers;
  llvm::StringMap<PipelinePath> pipelinePaths;
  llvm::StringMap<int> vectorOpCyclesPerInstruction;

  // Parallelism info
  llvm::StringMap<bool> parallelismFlags;

  // tilesim-migrated micro-architecture tables.
  // Bandwidth keyed by "src:dst" (costmodel space names: hbm/l2/l1/l0a/l0b/
  // l0c/ub). Vector instruction cycles keyed by intrinsic -> dtype -> entry.
  llvm::StringMap<BandwidthTable> bandwidthTables;
  llvm::StringMap<llvm::StringMap<VecCycleEntry>> vecCycleTables;
  CubeModelConfig cubeModel;
  int activeBandwidthCores = 48; // 910B1 vec_core_num saturation point
  bool enableSmallPacketBw = false;
  // Global small-packet coefficients (tilesim pkt_param): a single (a,b) set
  // per size bucket applied to ANY mover lookup when enableSmallPacketBw is on
  // and the mover's own table has no per-table small_packet entry. tilesim
  // keeps one global pkt_param dict on ArcConfig (910B1.yaml); per-table
  // overrides it.
  SmallPacketCoeffs globalSmallPacket;
  bool hasGlobalSmallPacket = false;
  // Mutex unit groups (tilesim MutexComponents): each group is a clique of
  // unit-name strings that share a pipeline and cannot run in parallel.
  // Default 910B: {"vec_mte2", "mte3"} (AIV MTE2<->MTE3).
  std::vector<std::vector<std::string>> mutexGroups;
  // Cost of one synchronisation barrier in cycles. Zero means "not modelled",
  // which is what every profile says until someone measures it.
  int64_t barrierCycles = 0;
  // Vector cores sharing one compute block's vector work. One means "no
  // pairing described", which leaves every existing profile unchanged.
  int vectorCoresPerBlock = 1;
  // Vector datapath width. Defaulting to the table's own width makes the
  // rescaling a no-op for profiles that do not declare one.
  int vectorBytesPerCycle = kVecTableBytesPerCycle;
  // Cycles per scalar instruction. Zero means "not modelled", which is what
  // every profile says until the number is measured.
  int scalarCyclesPerInstruction = 0;
  // What one pipeline fill/drain iteration costs relative to a full one. Half
  // comes from the ramp's geometry rather than from a profile, so a config
  // that says nothing still charges the ramp.
  double prologueWorkFraction = 0.5;
};

//===----------------------------------------------------------------------===//
// Global Hardware Config Access
//===----------------------------------------------------------------------===//

/// Get the current hardware configuration.
/// Returns default 910B config if not set.
HardwareConfig &getHardwareConfig();

/// Set the global hardware configuration.
void setHardwareConfig(std::unique_ptr<HardwareConfig> config);

/// Load and set hardware config from file.
bool loadHardwareConfigFromFile(llvm::StringRef path, std::string &error);

/// Load an independent hardware configuration for one analysis invocation.
/// Returns the default 910B config when path is empty.
std::shared_ptr<const HardwareConfig>
loadHardwareConfigForAnalysis(llvm::StringRef path, std::string &error);

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_HARDWARECONFIG_H
