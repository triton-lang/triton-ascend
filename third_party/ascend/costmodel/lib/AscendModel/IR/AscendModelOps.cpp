//===- AscendModelOps.cpp - AscendModel operations implementation ---------===//
//
// This file implements the operations in the AscendModel dialect.
// Each operation implements the EstimateCyclesOpInterface to provide
// hardware-aware cycle estimation.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/Analysis/PipelineAnalysis.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::ascend;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Get number of elements from a tensor type.
static int64_t getNumElementsFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    int64_t count = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 1024;  // Default estimate for dynamic shapes
      count *= dim;
    }
    return count;
  }
  return 1;
}

/// Get element bit width from a tensor type.
static int getElementBitsFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type elemType = tensorType.getElementType();
    if (elemType.isF16() || elemType.isBF16())
      return 16;
    else if (elemType.isF32())
      return 32;
    else if (elemType.isF64())
      return 64;
    else if (auto intType = dyn_cast<IntegerType>(elemType))
      return intType.getWidth();
  }
  return 32;  // Default to FP32
}

/// Estimate vector operation cycles.
/// Vector unit is 2048 bits wide = 128 FP16 = 64 FP32 elements per cycle.
static int64_t estimateVectorCycles(int64_t numElements, int cyclesPerVectorOp,
                                    int elementBits, int startupLatency) {
  // Vector width based on element type
  int64_t vectorWidth = 2048 / elementBits;
  
  // Number of vector operations
  int64_t numVectorOps = (numElements + vectorWidth - 1) / vectorWidth;
  
  // Total cycles
  return numVectorOps * cyclesPerVectorOp + startupLatency;
}

/// Estimate memory transfer cycles.
static int64_t estimateMemoryCycles(int64_t bytes, const HardwareConfig &config,
                                    int startupLatency) {
  double bandwidth_gbs = config.getHBMBandwidthGBs();
  if (bandwidth_gbs <= 0) bandwidth_gbs = 1600.0;
  
  double time_seconds = static_cast<double>(bytes) / (bandwidth_gbs * 1e9);
  double cycles = time_seconds * config.getClockFrequencyGHz() * 1e9;
  return static_cast<int64_t>(cycles) + startupLatency;
}

//===----------------------------------------------------------------------===//
// Include generated definitions
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "AscendModel/IR/AscendModelOps.cpp.inc"

//===----------------------------------------------------------------------===//
// EstimateCyclesOpInterface helpers (defined in interface)
//===----------------------------------------------------------------------===//

int EstimateCyclesOpInterface::getElementBits() {
  Operation *op = getOperation();
  if (op->getNumOperands() > 0)
    return getElementBitsFromType(op->getOperand(0).getType());
  if (op->getNumResults() > 0)
    return getElementBitsFromType(op->getResult(0).getType());
  return 32;
}

int64_t EstimateCyclesOpInterface::getNumElements() {
  Operation *op = getOperation();
  if (op->getNumOperands() > 0)
    return getNumElementsFromType(op->getOperand(0).getType());
  if (op->getNumResults() > 0)
    return getNumElementsFromType(op->getResult(0).getType());
  return 1;
}

//===----------------------------------------------------------------------===//
// MatmulOp Interface Implementation
//===----------------------------------------------------------------------===//

int64_t MatmulOp::estimateCycles(const HardwareConfig &config) {
  int64_t m = getM();
  int64_t n = getN();
  int64_t k = getK();
  
  // Get element type bit width
  int elemBits = getElementBitsFromType(getLhs().getType());
  
  // Get fractal dimensions from hardware config based on data type
  int fracM, fracN, fracK;
  config.getCubeFractalSize(elemBits, fracM, fracN, fracK);
  
  // Calculate number of fractals needed in each dimension
  // Each fractal computes fracM x fracN x fracK multiply-accumulate
  int64_t numFracM = (m + fracM - 1) / fracM;
  int64_t numFracN = (n + fracN - 1) / fracN;
  int64_t numFracK = (k + fracK - 1) / fracK;
  
  // Total number of fractal operations
  // For a full matmul, we need numFracM * numFracN output tiles,
  // each requiring numFracK accumulation steps
  int64_t totalFractals = numFracM * numFracN * numFracK;
  
  // Each fractal operation takes 1 cycle on the Cube Core
  // (this is the fundamental unit of computation)
  int64_t computeCycles = totalFractals;
  
  // Add startup latency
  computeCycles += config.getCubeStartupLatency();
  
  return computeCycles;
}

HWUnit MatmulOp::getHWUnit() { return HWUnit::Cube; }

int64_t MatmulOp::getFlops() {
  return 2 * getM() * getN() * getK();
}

//===----------------------------------------------------------------------===//
// Memory Operations Interface Implementation
//===----------------------------------------------------------------------===//

int64_t CubeLoadOp::estimateCycles(const HardwareConfig &config) {
  return estimateMemoryCycles(getBytes(), config, config.getMTE2StartupLatency());
}
HWUnit CubeLoadOp::getHWUnit() { return HWUnit::CubeMTE2; }
int64_t CubeLoadOp::getTransferBytes() { return getBytes(); }

int64_t CubeStoreOp::estimateCycles(const HardwareConfig &config) {
  return estimateMemoryCycles(getBytes(), config, config.getFixPipeStartupLatency());
}
HWUnit CubeStoreOp::getHWUnit() { return HWUnit::FixPipe; }
int64_t CubeStoreOp::getTransferBytes() { return getBytes(); }

int64_t VectorLoadOp::estimateCycles(const HardwareConfig &config) {
  return estimateMemoryCycles(getBytes(), config, config.getMTE2StartupLatency());
}
HWUnit VectorLoadOp::getHWUnit() { return HWUnit::VecMTE2; }
int64_t VectorLoadOp::getTransferBytes() { return getBytes(); }

int64_t VectorStoreOp::estimateCycles(const HardwareConfig &config) {
  return estimateMemoryCycles(getBytes(), config, config.getMTE3StartupLatency());
}
HWUnit VectorStoreOp::getHWUnit() { return HWUnit::MTE3; }
int64_t VectorStoreOp::getTransferBytes() { return getBytes(); }

//===----------------------------------------------------------------------===//
// Simple Vector Binary Operations (1 cycle per vector op)
//===----------------------------------------------------------------------===//

#define IMPL_SIMPLE_VECTOR_BINARY(OpClass)                                      \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {               \
    int64_t n = getNumElementsFromType(getLhs().getType());                     \
    int bits = getElementBitsFromType(getLhs().getType());                      \
    return estimateVectorCycles(n, 1, bits, config.getVectorStartupLatency());  \
  }                                                                             \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                        \
  int64_t OpClass::getFlops() { return getNumElementsFromType(getLhs().getType()); }

IMPL_SIMPLE_VECTOR_BINARY(AddOp)
IMPL_SIMPLE_VECTOR_BINARY(SubOp)
IMPL_SIMPLE_VECTOR_BINARY(MulOp)
IMPL_SIMPLE_VECTOR_BINARY(MaxOp)
IMPL_SIMPLE_VECTOR_BINARY(MinOp)

#undef IMPL_SIMPLE_VECTOR_BINARY

//===----------------------------------------------------------------------===//
// Complex Vector Binary Operations
//===----------------------------------------------------------------------===//

int64_t DivOp::estimateCycles(const HardwareConfig &config) {
  int64_t n = getNumElementsFromType(getLhs().getType());
  int bits = getElementBitsFromType(getLhs().getType());
  // Calibrated: 12 cycles (was 4). Division is latency-limited on Ascend 910B.
  return estimateVectorCycles(n, 12, bits, config.getVectorStartupLatency());
}
HWUnit DivOp::getHWUnit() { return HWUnit::Vector; }
int DivOp::getCyclesPerVectorOp() { return 12; }
int64_t DivOp::getFlops() { return getNumElementsFromType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// Vector Comparison Operations (1 cycle per vector op)
//===----------------------------------------------------------------------===//

#define IMPL_VECTOR_CMP(OpClass)                                                 \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {                \
    int64_t n = getNumElementsFromType(getLhs().getType());                      \
    int bits = getElementBitsFromType(getLhs().getType());                       \
    return estimateVectorCycles(n, 1, bits, config.getVectorStartupLatency());   \
  }                                                                              \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                         \
  int64_t OpClass::getFlops() { return getNumElementsFromType(getLhs().getType()); }

IMPL_VECTOR_CMP(CmpEqOp)
IMPL_VECTOR_CMP(CmpNeOp)
IMPL_VECTOR_CMP(CmpLtOp)
IMPL_VECTOR_CMP(CmpLeOp)
IMPL_VECTOR_CMP(CmpGtOp)
IMPL_VECTOR_CMP(CmpGeOp)

#undef IMPL_VECTOR_CMP

//===----------------------------------------------------------------------===//
// Simple Vector Unary Operations (1 cycle per vector op)
//===----------------------------------------------------------------------===//

#define IMPL_SIMPLE_VECTOR_UNARY(OpClass)                                        \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {                \
    int64_t n = getNumElementsFromType(getInput().getType());                    \
    int bits = getElementBitsFromType(getInput().getType());                     \
    return estimateVectorCycles(n, 1, bits, config.getVectorStartupLatency());   \
  }                                                                              \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                         \
  int64_t OpClass::getFlops() { return getNumElementsFromType(getInput().getType()); }

IMPL_SIMPLE_VECTOR_UNARY(NegOp)
IMPL_SIMPLE_VECTOR_UNARY(AbsOp)
IMPL_SIMPLE_VECTOR_UNARY(ReluOp)
IMPL_SIMPLE_VECTOR_UNARY(CastOp)

#undef IMPL_SIMPLE_VECTOR_UNARY

//===----------------------------------------------------------------------===//
// Complex Vector Unary Operations
//===----------------------------------------------------------------------===//

#define IMPL_COMPLEX_VECTOR_UNARY(OpClass, CyclesPerOp)                          \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {                \
    int64_t n = getNumElementsFromType(getInput().getType());                    \
    int bits = getElementBitsFromType(getInput().getType());                     \
    return estimateVectorCycles(n, CyclesPerOp, bits,                            \
                                config.getVectorStartupLatency());               \
  }                                                                              \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                         \
  int OpClass::getCyclesPerVectorOp() { return CyclesPerOp; }                    \
  int64_t OpClass::getFlops() { return getNumElementsFromType(getInput().getType()); }

IMPL_COMPLEX_VECTOR_UNARY(SqrtOp, 6)
IMPL_COMPLEX_VECTOR_UNARY(RsqrtOp, 6)
// Calibrated transcendental costs for Ascend 910B Vector unit.
// Profiling of _attn_fwd (flash attention) across BLOCK_M={16,32,48,64}
// showed the model underestimated vector time by ~3-4x vs. observed aiv_vec_time.
// Root cause: each vector instruction in a dependency chain stalls until the
// previous result is written back to UB (read-after-write penalty), and the
// Triton-Ascend compiler emits more instructions than the idealised model counts.
// Calibration factor: 3x applied uniformly to all transcendental ops.
// exp: 9 cycles  (was 3, hardware-accelerated but latency-limited in chains)
// log: 12 cycles (was 4)
// tanh: 18 cycles (was 6, internally uses multiple exp steps)
// sigmoid: 15 cycles (was 5, 1/(1+exp(-x)))
IMPL_COMPLEX_VECTOR_UNARY(ExpOp, 9)
IMPL_COMPLEX_VECTOR_UNARY(LogOp, 12)
IMPL_COMPLEX_VECTOR_UNARY(TanhOp, 18)
IMPL_COMPLEX_VECTOR_UNARY(SigmoidOp, 15)

#undef IMPL_COMPLEX_VECTOR_UNARY

//===----------------------------------------------------------------------===//
// Reduce Operations
//===----------------------------------------------------------------------===//

#define IMPL_REDUCE_OP(OpClass)                                                  \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {                \
    int64_t numElems = getNumElementsFromType(getInput().getType());             \
    int bits = getElementBitsFromType(getInput().getType());                     \
    int64_t vectorWidth = 2048 / bits;                                           \
    int64_t numVectors = (numElems + vectorWidth - 1) / vectorWidth;             \
    int vectorReduceCycles = 0;                                                  \
    int64_t w = vectorWidth;                                                     \
    while (w > 1) { w /= 2; vectorReduceCycles++; }                              \
    int crossVectorCycles = 0;                                                   \
    int64_t v = numVectors;                                                      \
    while (v > 1) { v /= 2; crossVectorCycles++; }                               \
    return numVectors + vectorReduceCycles + crossVectorCycles +                 \
           config.getVectorStartupLatency();                                     \
  }                                                                              \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                         \
  int64_t OpClass::getFlops() { return getNumElementsFromType(getInput().getType()); }

IMPL_REDUCE_OP(ReduceSumOp)
IMPL_REDUCE_OP(ReduceMaxOp)
IMPL_REDUCE_OP(ReduceMinOp)
IMPL_REDUCE_OP(ReduceProdOp)

#undef IMPL_REDUCE_OP

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

int64_t BroadcastOp::estimateCycles(const HardwareConfig &config) {
  // Broadcast is essentially free, just startup latency
  return 1 + config.getVectorStartupLatency();
}
HWUnit BroadcastOp::getHWUnit() { return HWUnit::Vector; }

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

int64_t SelectOp::estimateCycles(const HardwareConfig &config) {
  int64_t n = getNumElementsFromType(getResult().getType());
  int bits = getElementBitsFromType(getResult().getType());
  return estimateVectorCycles(n, 1, bits, config.getVectorStartupLatency());
}
HWUnit SelectOp::getHWUnit() { return HWUnit::Vector; }
int64_t SelectOp::getFlops() {
  return getNumElementsFromType(getResult().getType());
}
