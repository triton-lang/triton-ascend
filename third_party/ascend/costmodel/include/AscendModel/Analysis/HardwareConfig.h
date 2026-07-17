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
  int64_t estimateMemoryCycles(llvm::StringRef moverName, int64_t bytes) const;
  int64_t estimateMemoryCyclesWithLatency(llvm::StringRef space,
                                          int64_t bytes) const;

  // Pipeline info
  const PipelinePath *getPipelinePath(llvm::StringRef name) const;
  bool canRunInParallel(llvm::StringRef path1, llvm::StringRef path2) const;

  // Validation
  bool validate(std::string &error) const;

  // Debug

private:
  bool parseJSON(const llvm::json::Value &json, std::string &error);

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
