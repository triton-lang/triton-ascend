/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//===----------------------------------------------------------------------===//
//
// Estimates the execution cost of the IR produced by the dynamic CV pipeline.
//
// Unlike the standalone costmodel pipeline (-ascend-perf-model), which runs on
// Triton IR and has to *guess* what the compiler will do, this pass runs at the
// end of the CV pipeline and reads the decisions the compiler actually made:
// the Cube/Vector split is taken from the ssbuffer.core_type attributes stamped
// by OpClassifierPass rather than re-derived. Only the hardware cost formulas
// (HardwareConfig) are reused from the costmodel; the schedule itself is built
// here, over blocks rather than over the costmodel's own operation list.
//
// The estimate is assembled from compute blocks, because the block is what the
// pipeline plans, schedules and synchronises -- and because a variant that
// merely regroups the same operations has to score differently, which a single
// roofline over all operations cannot do.
//
// Operations are costed per hardware pipe and scheduled per pipe: each pipe
// runs one thing at a time, but a block does not occupy its whole core, so
// block B+1's loads run underneath block B's compute the way the hardware
// actually issues them. What does order blocks is the IR's own barriers -- a
// block waiting on a synchronisation flag cannot start before the block that
// sets it has finished -- plus barriers found inside a block, which split it
// into serial segments. The module total is the larger of two bounds:
//
//   resource   -- the busiest pipe's total busy time;
//   recurrence -- the buffer that is held longest relative to how many copies
//                 of it the pipeline allocated.
//
// The recurrence term is what a plain "II * (N - 1) + latency" gets wrong.
// That formula assumes the pipeline reaches steady state, which needs enough
// buffering to hide the dependency; the inter-core depth defaults to one, so
// on a kernel that alternates Cube and Vector the next iteration cannot start
// writing a buffer until this one has finished reading it. Deep buffering
// leaves the resource bound standing, a single buffer exposes the alternation,
// and the maximum of the two crosses over on its own.
//
// The number is intended for *ranking* CV pipeline variants against each other,
// not for absolute latency prediction.
//
// Because a single number hides how much of it was actually modelled, every
// operation is classified (see CostConfidence) and a per-operation-kind
// breakdown can be printed. Operations charged generically are the ones worth
// teaching the model about next; operations of unknown size contribute nothing
// at all and would otherwise be invisible.
//
//   TRITON_ASCEND_CV_COST_VERBOSE=1   one-line summary
//   TRITON_ASCEND_CV_COST_VERBOSE=2   summary + per-operation breakdown
//
// A loop whose bound is not a compile-time constant has no trip count, and its
// body is usually where the kernel spends its time. Such a bound comes either
// from a kernel argument or from the launch grid -- TritonToLinalg appends
// program_id and num_programs to the entry function's arguments, so naming
// them binds them. Failing that, say how many iterations to assume:
//
//   TRITON_ASCEND_CV_COST_ARG_BINDINGS=pid_x=0,num_programs_x=28,arg3=98432
//   TRITON_ASCEND_CV_COST_DEFAULT_TRIP_COUNT=32
//
// The hardware profile defaults to a 910B. Selecting another one matters on
// targets whose data paths differ: the Cube accumulator can drain straight
// into UB, and UB and L1 exchange data on chip instead of through HBM. Which
// engine a transfer occupies is derived from the address spaces the compiler
// put on the memrefs, so those paths are costed as soon as the profile
// describes them:
//
//   TRITON_ASCEND_CV_COST_HARDWARE_CONFIG=/path/to/ascend_custom.json
//
// Operations inside one block issue back to back on one core, so intermediates
// can stay in registers rather than making a round trip through memory. Each
// operation is nonetheless charged as a separate full pass, which over-counts.
// These scale a block's cost to compensate, per core because the two fuse for
// different reasons. Both default to 1.0, i.e. off:
//
//   TRITON_ASCEND_CV_COST_FUSION_CUBE=0.8
//   TRITON_ASCEND_CV_COST_FUSION_VECTOR=0.8
//
//===----------------------------------------------------------------------===//

#include "ascend/include/DynamicCVPipeline/EstimateCVPipelineCost.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
#include "AscendModel/Analysis/HardwareConfig.h"
#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/Analysis/Utils.h"
#endif

using namespace mlir;
using namespace mlir::triton;

static constexpr const char *DEBUG_TYPE = "estimate-cv-pipeline-cost";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LOG_DEBUG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL

namespace {

/// Opt-in stderr summary. LLVM_DEBUG requires a debug build plus -debug-only,
/// which is impractical when driving compilation from Python; this env var
/// gives the same one-line summary from a release build.
constexpr const char *kVerboseEnvVar = "TRITON_ASCEND_CV_COST_VERBOSE";

/// Values for entry-function arguments, so that loops bounded by a kernel
/// argument can be resolved: "arg3=98432,arg5=128".
constexpr const char *kArgBindingsEnvVar = "TRITON_ASCEND_CV_COST_ARG_BINDINGS";

/// Iterations assumed for loops that no binding resolves. Defaults to 1, which
/// counts such a body exactly once.
constexpr const char *kDefaultTripCountEnvVar =
    "TRITON_ASCEND_CV_COST_DEFAULT_TRIP_COUNT";

/// Path to the hardware profile JSON. Without this there is no way to select
/// anything other than the built-in default, which is a 910B.
constexpr const char *kHardwareConfigEnvVar =
    "TRITON_ASCEND_CV_COST_HARDWARE_CONFIG";

/// Fraction of a block's cost that survives intra-block fusion, per core.
/// Operations inside one compute block are issued back to back on one core, so
/// intermediate results can stay in registers instead of making a round trip
/// through memory; the model charges each operation as a separate full pass and
/// therefore over-counts. These scale a block's cost to compensate.
///
/// Separate knobs per core because the two fuse for different reasons: the
/// Vector core chains elementwise instructions, the Cube core mostly does not.
/// 1.0 disables the correction, which is the default -- an unmeasured factor
/// should not silently change everyone's numbers.
///
/// These belong in the hardware profile eventually; they are read from the
/// environment for now so they can be swept without a rebuild.
constexpr const char *kFusionCubeEnvVar = "TRITON_ASCEND_CV_COST_FUSION_CUBE";
constexpr const char *kFusionVectorEnvVar =
    "TRITON_ASCEND_CV_COST_FUSION_VECTOR";

using mlir::ascend::HardwareConfig;
using mlir::ascend::HWUnit;
using mlir::ascend::MemorySpace;

//===----------------------------------------------------------------------===//
// Shape helpers
//===----------------------------------------------------------------------===//
// The CV pipeline operates on partially bufferized IR: the same logical buffer
// appears as a tensor before bufferization and as a memref after. Both are
// ShapedType, so cost extraction is written against that rather than against
// RankedTensorType (which is all the costmodel's own helpers handle).

/// Number of elements, or nullopt when the shape is not statically known.
std::optional<int64_t> getShapedElementCount(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape()) {
    return std::nullopt;
  }
  int64_t count = 1;
  for (int64_t dim : shaped.getShape()) {
    count *= dim;
  }
  return count;
}

int getShapedElementBits(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  Type elemType = shaped ? shaped.getElementType() : type;
  if (elemType.isIntOrFloat()) {
    return elemType.getIntOrFloatBitWidth();
  }
  return 32; // conservative default for opaque element types
}

/// Byte size, or nullopt when the shape is not statically known.
std::optional<int64_t> getShapedByteSize(Type type) {
  auto count = getShapedElementCount(type);
  if (!count) {
    return std::nullopt;
  }
  return *count * ((getShapedElementBits(type) + 7) / 8);
}

/// First operand or result carrying a shaped type, used to size elementwise
/// ops whose cost is driven by their working-set size.
Type getRepresentativeShapedType(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (isa<ShapedType>(operand.getType())) {
      return operand.getType();
    }
  }
  for (Value result : op->getResults()) {
    if (isa<ShapedType>(result.getType())) {
      return result.getType();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Core-type resolution
//===----------------------------------------------------------------------===//

/// Resolve whether an op runs on the Cube core. Primary source is the
/// ssbuffer.core_type attribute stamped by OpClassifierPass; ops that never got
/// stamped (or carry a multi-valued type) fall back to the tcore_type of the
/// enclosing scope::ScopeOp created by SeparateCVScopePass.
bool runsOnCubeCore(Operation *op) {
  switch (CVPipeline::getOpCoreType(op)) {
  case CVPipeline::CoreType::CUBE_ONLY:
    return true;
  case CVPipeline::CoreType::VECTOR_ONLY:
    return false;
  default:
    break;
  }

  if (auto scopeOp = op->getParentOfType<scope::ScopeOp>()) {
    bool isCube = false;
    bool isVector = false;
    if (succeeded(getScopeType(scopeOp, isCube, isVector)) && isCube &&
        !isVector) {
      return true;
    }
  }
  return false; // Vector is the default core, matching OpClassifierPass.
}

//===----------------------------------------------------------------------===//
// Operation classification
//===----------------------------------------------------------------------===//

/// Ops that carry no hardware cost: structure, metadata, addressing and scalar
/// bookkeeping. Address arithmetic is absorbed into DMA descriptors on Ascend,
/// and allocations are compile-time placement rather than runtime work.
bool isZeroCostOp(Operation *op) {
  if (isa<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp>(op)) {
    return true;
  }
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return true; // yields/returns transfer values, they do not compute
  }
  // A linalg body describes the elementwise kernel of its parent op; the parent
  // is costed as a whole, so charging its body again would double-count.
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<linalg::LinalgDialect>(parent->getDialect())) {
      return true;
    }
  }
  if (CVPipeline::isScfOp(op)) {
    return true; // loop/branch structure; the body is costed on its own
  }
  if (isa<scope::ScopeOp>(op)) {
    return true; // container introduced by SeparateCVScopePass
  }
  if (isa<annotation::MarkOp>(op)) {
    return true; // annotation only
  }
  if (isa<memref::AllocOp, memref::AllocaOp, memref::DeallocOp,
          tensor::EmptyOp, bufferization::AllocTensorOp,
          bufferization::ToTensorOp>(op)) {
    return true; // buffer placement / tensor-memref bridging
  }
  if (CVPipeline::isViewLike(op)) {
    return true; // subview/reshape: descriptor math, no data movement
  }
  // Shape metadata: reshape/expand/collapse/cast rewrite how a buffer is
  // indexed, they never move an element. Charging them as vector work is a
  // pure over-estimate. (The memref equivalents are ViewLikeOpInterface and
  // are already covered above.)
  if (isa<tensor::ReshapeOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
          tensor::CastOp>(op)) {
    return true;
  }
  if (isa<arith::ConstantOp>(op)) {
    return true;
  }
  // Scalar arith/math (loop bounds, offsets, predicates) is bookkeeping; only
  // shaped operations occupy a compute pipe.
  if (isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
    return !getRepresentativeShapedType(op);
  }
  return false;
}

/// Cross-core synchronisation. These occupy real time -- a core waiting on a
/// flag is idle -- but modelling that needs the flag graph the CV pipeline
/// builds, which this pass does not read yet. Recognised explicitly so they
/// are reported as un-modelled rather than silently mistaken for compute.
bool isSyncOp(Operation *op) {
  return isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp, hivm::SyncBlockOp,
             hivm::SyncBlockLockOp, hivm::SyncBlockUnlockOp,
             hivm::CreateSyncBlockLockOp>(op);
}

/// Address arithmetic and single-element memory access: the scalar half of a
/// kernel, which runs on its core's scalar unit rather than on a compute pipe.
///
/// Recognised by operation, not by dialect. It used to be by dialect -- these
/// were llvm.load / llvm.store / llvm.inttoptr and testing for LLVMDialect
/// caught all of them -- until the CV pipeline stopped emitting LLVM here and
/// started emitting the MLIR equivalents. The dialect test then matched
/// nothing, and because a memref.load names a memref, the generic path charged
/// it for every element of the buffer it indexes: 18 of them came to 94 million
/// cycles on flash attention, on the Cube pipe, where the profiler measures the
/// matmul term alone to five figures. A memref.load reads one element by
/// definition, so its cost cannot scale with the shape it is addressing.
///
/// The old spelling stays recognised: other paths into this pass still produce
/// it, and a dialect that is simply absent costs nothing to keep testing for.
bool isScalarMemoryOp(Operation *op) {
  if (isa<LLVM::LLVMDialect>(op->getDialect())) {
    return true;
  }
  // memref.load / memref.store address one element; the surrounding loop is
  // what makes them numerous, and the loop multiplier already counts that.
  // Bulk movement is memref.copy and the hivm transfers, handled elsewhere.
  if (isa<memref::LoadOp, memref::StoreOp>(op)) {
    return true;
  }
  // hivm.hir.pointer_cast is the replacement for llvm.inttoptr: it reinterprets
  // an address and moves nothing.
  if (isa<hivm::PointerCastOp>(op)) {
    return true;
  }
  return false;
}


/// Whether a value's backing buffer is a local (on-chip) allocation, as opposed
/// to a kernel argument living in global memory. Mirrors the dest-tracing rule
/// MarkGMLoadPass uses to recognise GM loads.
bool isBackedByLocalAlloc(Value value) {
  Value current = value;
  while (Operation *def = current.getDefiningOp()) {
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(def)) {
      current = viewLike.getViewSource();
      continue;
    }
    if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(def)) {
      current = extractSlice.getSource();
      continue;
    }
    break;
  }
  Operation *def = current.getDefiningOp();
  return def && isa<memref::AllocOp, memref::AllocaOp>(def);
}

/// Costmodel name of the memory a buffer lives in, read from the address space
/// the compiler attached to its memref. Empty when there is none to read -- a
/// tensor that has not been bufferized yet, typically.
///
/// Reading this rather than inferring it from the operation's role is what
/// makes the newer topology expressible at all: there, the same memref.copy
/// may target HBM or UB or L1, and only the type says which.
llvm::StringRef getMemorySpaceName(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType) {
    return {};
  }
  auto addressSpace =
      dyn_cast_or_null<hivm::AddressSpaceAttr>(memrefType.getMemorySpace());
  if (!addressSpace) {
    return {};
  }
  switch (addressSpace.getAddressSpace()) {
  case hivm::AddressSpace::GM:
    return "hbm"; // the bandwidth tables call global memory "hbm"
  case hivm::AddressSpace::L1:
    return "l1";
  case hivm::AddressSpace::L0A:
    return "l0a";
  case hivm::AddressSpace::L0B:
    return "l0b";
  case hivm::AddressSpace::L0C:
    return "l0c";
  case hivm::AddressSpace::UB:
    return "ub";
  default:
    return {};
  }
}

/// Which data mover executes a transfer between two memory spaces.
///
/// The engine follows from where the data comes from and goes to, not from
/// which core issued it. On the newer topology that distinction matters:
/// draining the accumulator to UB (FixPipeUB) and to HBM (FixPipe) are
/// different engines, and UB<->L1 traffic never leaves the chip at all.
HWUnit getTransferUnit(llvm::StringRef src, llvm::StringRef dst) {
  if (src == "l0c") {
    return dst == "ub" ? HWUnit::FixPipeUB : HWUnit::FixPipe;
  }
  if (dst == "l0a" || dst == "l0b") {
    // Into the Cube's operand buffers. From L1 this is its MTE1, an engine the
    // profiler reports apart from MTE2 and which nothing charged until now.
    if (src == "l1") {
      return HWUnit::CubeMTE1;
    }
    return src == "ub" ? HWUnit::MTE3ToL1 : HWUnit::CubeMTE2;
  }
  if (dst == "l1") {
    // Feeding the Cube: from HBM/L2 this is its MTE2; from UB it is the vector
    // store engine writing on chip instead of out to HBM, which this target
    // can do directly -- the bandwidth table has a ub:l1 entry, so no round
    // trip through L2 is charged.
    return src == "ub" ? HWUnit::MTE3ToL1 : HWUnit::CubeMTE2;
  }
  if (dst == "ub") {
    // From L1 this is the on-chip return path, again direct (l1:ub); from
    // HBM/L2 it is an ordinary vector load.
    return src == "l1" ? HWUnit::MTE1ToUB : HWUnit::VecMTE2;
  }
  if (src == "ub") {
    return HWUnit::MTE3;
  }
  return HWUnit::VecMTE2;
}

/// Memory spaces to assume when the IR does not carry an address space, e.g.
/// while values are still tensors. Reproduces the 910B-shaped guess the pass
/// used before address spaces were read.
std::pair<llvm::StringRef, llvm::StringRef> guessTransferSpaces(bool isCube,
                                                                bool isLoad) {
  using Spaces = std::pair<llvm::StringRef, llvm::StringRef>;
  if (isCube) {
    return isLoad ? Spaces{"hbm", "l1"} : Spaces{"l0c", "hbm"};
  }
  return isLoad ? Spaces{"hbm", "ub"} : Spaces{"ub", "hbm"};
}

/// Source and destination spaces of a transfer, preferring what the IR says
/// and falling back to the guess above.
std::pair<llvm::StringRef, llvm::StringRef>
resolveTransferSpaces(Type srcType, Type dstType, bool isCube, bool isLoad) {
  llvm::StringRef src = getMemorySpaceName(srcType);
  llvm::StringRef dst = getMemorySpaceName(dstType);
  if (src.empty() || dst.empty()) {
    auto guessed = guessTransferSpaces(isCube, isLoad);
    if (src.empty()) {
      src = guessed.first;
    }
    if (dst.empty()) {
      dst = guessed.second;
    }
  }
  return {src, dst};
}

/// Both ends of an hivm data-movement operation.
///
/// hivm.hir.fixpipe and hivm.hir.copy are both built as (source, destination)
/// -- see InterCoreTransferAndSync, which creates them and labels the operands.
/// The roles come from that convention rather than from scanning for whichever
/// operand happens to carry an address space: fixpipe's source is often still a
/// tensor, so such a scan finds only the destination and, taking it for the
/// source, turns a Cube drain into a vector store.
struct TransferEnds {
  Value source;
  Value dest;
};

std::optional<TransferEnds> getHivmTransferEnds(Operation *op) {
  if (op->getNumOperands() < 2) {
    return std::nullopt;
  }
  return TransferEnds{op->getOperand(0), op->getOperand(1)};
}

/// Spaces of an hivm transfer, per side, falling back independently so that an
/// unreadable end never shifts the other end's role.
std::pair<llvm::StringRef, llvm::StringRef>
resolveHivmTransferSpaces(Operation *op, llvm::StringRef fallbackSrc,
                          llvm::StringRef fallbackDst) {
  llvm::StringRef src;
  llvm::StringRef dst;
  if (auto ends = getHivmTransferEnds(op)) {
    src = getMemorySpaceName(ends->source.getType());
    dst = getMemorySpaceName(ends->dest.getType());
  }
  return {src.empty() ? fallbackSrc : src, dst.empty() ? fallbackDst : dst};
}

/// Size of a transfer: what lands in the destination, falling back to whatever
/// shaped value the operation exposes.
Type getTransferSizingType(Operation *op) {
  if (auto ends = getHivmTransferEnds(op)) {
    if (isa<ShapedType>(ends->dest.getType())) {
      return ends->dest.getType();
    }
  }
  return getRepresentativeShapedType(op);
}

int64_t getTransferStartupLatency(HWUnit unit, const HardwareConfig &config) {
  switch (unit) {
  case HWUnit::CubeMTE2:
  case HWUnit::VecMTE2:
  // Reading L1 into UB is still the load engine starting up.
  case HWUnit::MTE1ToUB:
  // So is staging L1 into the Cube's operand buffers.
  case HWUnit::CubeMTE1:
    return config.getMTE2StartupLatency();
  case HWUnit::FixPipe:
  case HWUnit::FixPipeUB:
    return config.getFixPipeStartupLatency();
  case HWUnit::MTE3:
  // Writing UB out to L1 is still the store engine starting up.
  case HWUnit::MTE3ToL1:
    return config.getMTE3StartupLatency();
  default:
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// Cost estimation
//===----------------------------------------------------------------------===//

/// How much the model actually knows about an operation, as opposed to how
/// confident the resulting number looks. Reported per operation kind so that
/// the list of things worth teaching the model is visible rather than guessed.
enum class CostConfidence {
  /// The operation kind has a dedicated cost model and every input that model
  /// needs (shape, dtype, sizes) is statically known.
  Modelled,
  /// No dedicated model for this operation kind: it was charged generically
  /// from its element count. The number is an order of magnitude, not a model.
  Generic,
  /// Recognised as something that does occupy hardware, but deliberately
  /// charged zero because no model exists yet. Cross-core synchronisation is
  /// the important case: it is real time that this pass cannot yet account
  /// for. Distinct from Generic (which does produce a number) and from the
  /// zero-cost ops (which really are free).
  NotModelled,
  /// A required input is not statically known -- a dynamic shape -- so no cost
  /// could be computed at all and the operation contributes nothing.
  UnknownSize,
};

llvm::StringRef stringifyConfidence(CostConfidence confidence) {
  switch (confidence) {
  case CostConfidence::Modelled:
    return "modelled";
  case CostConfidence::Generic:
    return "generic";
  case CostConfidence::NotModelled:
    return "not-modelled";
  case CostConfidence::UnknownSize:
    return "unknown-size";
  }
  return "?";
}

struct OpCost {
  HWUnit unit = HWUnit::Scalar;
  int64_t cycles = 0;
  int64_t bytes = 0;
  int64_t flops = 0;
  /// A second engine the same operation occupies, for the case where one IR
  /// operation drives two of them at once. A matmul is the reason this exists:
  /// it runs the MACs *and* stages its operands from L1 into L0A/L0B, and the
  /// second is a separate engine with its own busy time -- one the profiler
  /// reports as a third of the Cube's. Zero cycles means "only one engine",
  /// which is every other operation.
  HWUnit sideUnit = HWUnit::Scalar;
  int64_t sideCycles = 0;
  CostConfidence confidence = CostConfidence::Modelled;
  /// Address spaces a transfer moves between, empty for everything else.
  /// Kept so the report can show which path each transfer took: the same
  /// operation kind can land on several engines with very different
  /// bandwidths, and the by-kind table collapses that away. Safe as
  /// StringRefs -- getMemorySpaceName returns string literals.
  llvm::StringRef srcSpace;
  llvm::StringRef dstSpace;
};

/// Cost of a linalg.matmul on the Cube core.
std::optional<OpCost> estimateMatmul(linalg::MatmulOp matmulOp,
                                     const HardwareConfig &config) {
  OpCost cost;
  cost.unit = HWUnit::Cube;

  auto inputs = matmulOp.getInputs();
  if (inputs.size() < 2) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }
  auto lhsType = dyn_cast<ShapedType>(inputs[0].getType());
  auto rhsType = dyn_cast<ShapedType>(inputs[1].getType());
  if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape()) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }

  int64_t m = lhsType.getShape()[0];
  int64_t k = lhsType.getShape()[1];
  int64_t n = rhsType.getShape()[1];

  // No startup latency. The fractal count alone matches the hardware: msprof
  // reports 299648 cube instructions costing 153.4M cycles on a kernel whose
  // tiles are 128x128x128, which is 511.9 cycles each against ceil(128/16)^3 =
  // 512. Adding the 20-cycle pipeline fill on top made it 532 and was the whole
  // of the error -- back-to-back matmuls fill the cube pipe once, not once per
  // operation, and this kernel never issues an isolated one.
  //
  // TODO: switch to the tilesim-migrated L1-pipe model used by
  // ascend::MatmulOp::estimateCycles; that formula is currently tied to the
  // costmodel's own dialect op and needs extracting before it can be reused.
  cost.cycles = config.estimateCubeCycles(m, n, k);
  cost.flops = 2 * m * n * k;

  // Both operands have to be staged from L1 into L0A/L0B before the MACs can
  // run, on the Cube's MTE1. No operation in the IR stands for that move --
  // the matmul reads L1 and the transfer is implicit -- so it is charged from
  // the operand shapes here, or it is not charged at all.
  //
  // The count checks out against the hardware: msprof reports exactly four
  // MTE1 instructions per inner iteration of a kernel whose loop body holds
  // two matmuls, which is one per operand. The time did too -- 128 KB moved in
  // about 560 cycles is ~234 B/cycle, against the 355 GB/s the profiler
  // reports for that engine when active.
  if (auto lhsBytes = getShapedByteSize(lhsType)) {
    if (auto rhsBytes = getShapedByteSize(rhsType)) {
      const int cores = config.getActiveBandwidthCores();
      cost.sideUnit = HWUnit::CubeMTE1;
      cost.sideCycles =
          config.estimateTransferCycles("l1", "l0a", *lhsBytes, cores) +
          config.estimateTransferCycles("l1", "l0b", *rhsBytes, cores);
    }
  }
  return cost;
}

/// Cost of a bulk data transfer between two memory spaces.
OpCost estimateTransfer(Type shapedType, llvm::StringRef srcSpace,
                        llvm::StringRef dstSpace,
                        const HardwareConfig &config) {
  OpCost cost;
  cost.unit = getTransferUnit(srcSpace, dstSpace);
  cost.srcSpace = srcSpace;
  cost.dstSpace = dstSpace;

  auto bytes = getShapedByteSize(shapedType);
  if (!bytes) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }
  cost.bytes = *bytes;

  int64_t transferCycles = config.estimateTransferCycles(
      srcSpace, dstSpace, *bytes, config.getActiveBandwidthCores());
  cost.cycles = transferCycles + getTransferStartupLatency(cost.unit, config);
  return cost;
}

/// Map an operation to the vector instruction whose measured cost it is closest
/// to. The names are tilesim mnemonics; the migrated cycle table distinguishes
/// cheap ALU ops from transcendentals, so this mapping is what makes an exp or
/// a divide cost more than an add. Returning an empty name means "no mapping",
/// which the caller reports as a gap rather than hiding.
llvm::StringRef getVectorIntrinsic(Operation *op) {
  using Ret = llvm::StringRef;
  return llvm::TypeSwitch<Operation *, Ret>(op)
      // Arithmetic.
      .Case<arith::AddFOp, arith::AddIOp>(
          [](Operation *) { return Ret("VADD"); })
      .Case<arith::SubFOp, arith::SubIOp>(
          [](Operation *) { return Ret("VSUB"); })
      .Case<arith::MulFOp, arith::MulIOp>(
          [](Operation *) { return Ret("VMUL"); })
      .Case<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(
          [](Operation *) { return Ret("VDIV"); })
      .Case<arith::MaxNumFOp, arith::MaximumFOp, arith::MaxSIOp,
            arith::MaxUIOp>([](Operation *) { return Ret("VMAX"); })
      .Case<arith::MinNumFOp, arith::MinimumFOp, arith::MinSIOp,
            arith::MinUIOp>([](Operation *) { return Ret("VMIN"); })
      .Case<arith::NegFOp>([](Operation *) { return Ret("VSUB"); })
      .Case<arith::SelectOp>([](Operation *) { return Ret("VSEL"); })
      .Case<arith::CmpFOp, arith::CmpIOp>(
          [](Operation *) { return Ret("VCMPV_GE"); })
      // Conversions all cost about one pass over the data.
      .Case<arith::TruncFOp, arith::ExtFOp, arith::TruncIOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
            arith::FPToUIOp, arith::BitcastOp>(
          [](Operation *) { return Ret("VCOPY"); })
      // Transcendentals.
      .Case<math::ExpOp, math::Exp2Op>([](Operation *) { return Ret("VEXP"); })
      .Case<math::LogOp, math::Log2Op>([](Operation *) { return Ret("LOG"); })
      .Case<math::SqrtOp, math::RsqrtOp>(
          [](Operation *) { return Ret("VSQRT"); })
      .Case<math::AbsFOp, math::AbsIOp>([](Operation *) { return Ret("VABS"); })
      // Structured ops that are really one pass over the data.
      .Case<linalg::FillOp, linalg::TransposeOp, linalg::CopyOp>(
          [](Operation *) { return Ret("VCOPY"); })
      .Case<linalg::BroadcastOp>([](Operation *) { return Ret("VBRCB"); })
      .Default([](Operation *) { return Ret(); });
}

/// How much of a tensor one vector core actually sees.
///
/// A compute block runs on one Cube core paired with several vector cores, and
/// they split the data rather than the instruction stream: each core issues the
/// same instructions over its own share of the tile. So the repeat count of a
/// vector instruction shrinks by this factor while the instruction's own
/// startup latency does not -- which is why this divides the element count fed
/// to the cycle table instead of dividing the resulting cycles.
///
/// Returns the count unchanged when told of one core, which is both what an
/// unpaired profile says and what Cube-side work gets: a fill staged for the
/// Cube runs on the one Cube core and is priced from this same table, so it
/// must not be divided by a pairing that does not apply to it.
int64_t elementsPerVectorCore(int64_t elements, int vectorCores) {
  if (vectorCores <= 1) {
    return elements;
  }
  return (elements + vectorCores - 1) / vectorCores;
}

/// Cost of a reduction. A reduction is not one pass over the data: after the
/// elementwise pass it costs a logarithmic tree inside each vector register and
/// another one across registers. Mirrors the costmodel's own reduce model.
OpCost estimateReduce(Operation *op, Type shapedType, int64_t elements,
                      const HardwareConfig &config, int vectorCores) {
  OpCost cost;
  cost.unit = HWUnit::Vector;

  int elementBits = getShapedElementBits(shapedType);
  int64_t vectorWidth = elementBits > 0 ? 2048 / elementBits : 64;
  if (vectorWidth <= 0) {
    vectorWidth = 1;
  }
  // Each core reduces its own share of the rows, so no cross-core combining
  // step follows and the split is a plain division of the work.
  int64_t numVectors = (elementsPerVectorCore(elements, vectorCores) +
                        vectorWidth - 1) /
                       vectorWidth;

  auto log2Steps = [](int64_t value) {
    int steps = 0;
    while (value > 1) {
      value /= 2;
      ++steps;
    }
    return steps;
  };

  cost.cycles = numVectors + log2Steps(vectorWidth) + log2Steps(numVectors) +
                config.getVectorStartupLatency();
  cost.flops = elements;
  return cost;
}

/// Cost of an elementwise compute op on the Vector core.
OpCost estimateVectorCompute(Operation *op, const HardwareConfig &config,
                             int vectorCores) {
  OpCost cost;
  cost.unit = HWUnit::Vector;

  Type shapedType = getRepresentativeShapedType(op);
  if (!shapedType) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }
  auto elements = getShapedElementCount(shapedType);
  if (!elements) {
    cost.confidence = CostConfidence::UnknownSize;
    return cost;
  }

  if (isa<linalg::ReduceOp>(op)) {
    return estimateReduce(op, shapedType, *elements, config, vectorCores);
  }

  llvm::StringRef intrinsic = getVectorIntrinsic(op);
  cost.cycles = config.estimateVectorCyclesFromTable(
      elementsPerVectorCore(*elements, vectorCores),
      getShapedElementBits(shapedType), intrinsic);
  // Work done, not time spent: this is the whole block's arithmetic however
  // many cores it was spread over.
  cost.flops = *elements;
  // With a known instruction the cost comes from the measured table; without
  // one it falls back to a flat cycle per pass, which is a guess worth
  // surfacing rather than a model.
  cost.confidence = intrinsic.empty() ? CostConfidence::Generic
                                      : CostConfidence::Modelled;
  return cost;
}

/// Classify an operation and estimate what it costs. Returns nullopt for ops
/// that do not occupy any hardware pipe.
std::optional<OpCost> estimateOpCost(Operation *op,
                                     const HardwareConfig &config) {
  if (isZeroCostOp(op)) {
    return std::nullopt;
  }

  // Cross-core synchronisation: real time, no model yet.
  if (isSyncOp(op)) {
    OpCost cost;
    cost.unit = HWUnit::Scalar;
    cost.confidence = CostConfidence::NotModelled;
    return cost;
  }

  // Pointer arithmetic and single-element loads and stores. They run on their
  // core's scalar unit, which the profiler measures at a fifth to a quarter of
  // that core's active time -- so not free, and worth charging as soon as the
  // per-instruction figure is known. Until then the profile leaves it at zero
  // and the cost is recognised but not counted, which at least keeps it from
  // masquerading as compute of unknown size.
  if (isScalarMemoryOp(op)) {
    OpCost cost;
    cost.unit = HWUnit::Scalar;
    cost.cycles = config.getScalarCyclesPerInstruction();
    cost.confidence = cost.cycles > 0 ? CostConfidence::Generic
                                      : CostConfidence::NotModelled;
    return cost;
  }

  const bool isCube = runsOnCubeCore(op);

  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    return estimateMatmul(matmulOp, config);
  }

  // hivm.hir.fixpipe drains the Cube accumulator. It may land in UB
  // instead of HBM, which is a different engine entirely, so the destination
  // is read from the operands rather than assumed.
  if (isa<hivm::FixpipeOp>(op)) {
    // The source is the accumulator by definition; only the destination is in
    // question, and it decides whether this is a FixPipe to HBM or
    // the on-chip drain into UB.
    auto [srcSpace, dstSpace] =
        resolveHivmTransferSpaces(op, /*fallbackSrc=*/"l0c",
                                  /*fallbackDst=*/"hbm");
    Type shapedType = getTransferSizingType(op);
    if (!shapedType) {
      OpCost cost;
      cost.unit = getTransferUnit(srcSpace, dstSpace);
      cost.confidence = CostConfidence::UnknownSize;
      return cost;
    }
    return estimateTransfer(shapedType, srcSpace, dstSpace, config);
  }

  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    // A copy into a local allocation is a load; a copy out of one is a
    // write-back. Used only when the memrefs carry no address space.
    const bool targetIsLocal = isBackedByLocalAlloc(copyOp.getTarget());
    const bool sourceIsLocal = isBackedByLocalAlloc(copyOp.getSource());
    const bool isLoad = targetIsLocal || !sourceIsLocal;
    auto [srcSpace, dstSpace] =
        resolveTransferSpaces(copyOp.getSource().getType(),
                              copyOp.getTarget().getType(), isCube, isLoad);
    return estimateTransfer(copyOp.getTarget().getType(), srcSpace, dstSpace,
                            config);
  }

  // hivm.hir.copy moves a buffer, it does not compute over it: it belongs to a
  // data mover, not to the vector ALU.
  if (isa<hivm::CopyOp>(op)) {
    // The CV pipeline emits these to stage data between the cores; on the newer
    // topology the destination is often L1, keeping the traffic on chip.
    //
    // The destination carries an address space, the source usually does not --
    // it is still a tensor at this point -- so the source has to be inferred,
    // and inferring it from the operation's core gets the important case
    // backwards. A copy that lands in L1 is the Vector core handing its result
    // up to the Cube: it reads UB and writes L1, which is the on-chip
    // mte3_ub_l1 mover. Falling back to "hbm" instead makes it a Cube load from
    // off-chip memory -- wrong core, wrong engine, wrong bandwidth.
    //
    // Profiling caught this directly: the cycles showed up under the Cube's
    // MTE2 while the hardware was spending them on the Vector core's MTE3,
    // which the model had at almost zero.
    llvm::StringRef knownSrc;
    llvm::StringRef knownDst;
    if (auto ends = getHivmTransferEnds(op)) {
      knownSrc = getMemorySpaceName(ends->source.getType());
      knownDst = getMemorySpaceName(ends->dest.getType());
    }
    const bool destFeedsCube =
        knownDst == "l1" || knownDst == "l0a" || knownDst == "l0b";

    auto guessed = guessTransferSpaces(isCube, /*isLoad=*/true);
    llvm::StringRef fallbackSrc = destFeedsCube ? llvm::StringRef("ub")
                                                : guessed.first;

    // On the Vector core, with the source unreadable and the destination not
    // naming a hand-off to the Cube, nothing left in the IR says this copy
    // leaves the chip -- and assuming it does is not a harmless default. With
    // 64x64 tiles that assumption charged 1.36 billion cycles, 39% of all the
    // work in the estimate, to the vector load engine and made it the pipe the
    // whole roofline was built from; msprof measures that engine at 5.7 us over
    // the same run and its GM->UB traffic at 6 KB, against the tens of
    // gigabytes implied. The same copies resolve to ub:l1 at 128x128, so what
    // varies is what the IR still carries, not what the hardware does.
    //
    // These copies exist to stage data between the cores and between buffers,
    // which is on-chip work; a transfer that leaves the chip is the case that
    // has to be proven. Charge nothing and say so: as an unmodelled kind it
    // shows up in the report as a hole, which is recoverable, rather than as a
    // bottleneck, which is not.
    if (!isCube && knownSrc.empty() && !destFeedsCube) {
      OpCost cost;
      cost.unit = HWUnit::VecMTE2;
      cost.confidence = CostConfidence::NotModelled;
      // "?" rather than a word: the transfer table gives the path a narrow
      // column, and a long name runs into the operation name beside it.
      cost.srcSpace = "?";
      cost.dstSpace = knownDst.empty() ? llvm::StringRef("?") : knownDst;
      if (Type sizingType = getTransferSizingType(op)) {
        if (auto bytes = getShapedByteSize(sizingType)) {
          cost.bytes = *bytes;
        }
      }
      return cost;
    }

    auto [srcSpace, dstSpace] =
        resolveHivmTransferSpaces(op, fallbackSrc, guessed.second);
    Type shapedType = getTransferSizingType(op);
    if (!shapedType) {
      OpCost cost;
      cost.unit = getTransferUnit(srcSpace, dstSpace);
      cost.confidence = CostConfidence::UnknownSize;
      return cost;
    }
    return estimateTransfer(shapedType, srcSpace, dstSpace, config);
  }

  if (CVPipeline::isStoreLike(op)) {
    Type shapedType = getRepresentativeShapedType(op);
    auto [srcSpace, dstSpace] = guessTransferSpaces(isCube, /*isLoad=*/false);
    if (!shapedType) {
      OpCost cost;
      cost.unit = getTransferUnit(srcSpace, dstSpace);
      cost.confidence = CostConfidence::UnknownSize;
      return cost;
    }
    return estimateTransfer(shapedType, srcSpace, dstSpace, config);
  }

  // Anything left that touches no shaped value is scalar bookkeeping: index
  // queries, block ids, predicates. Checked after the specific kinds above so
  // it can never swallow one of them.
  if (!getRepresentativeShapedType(op)) {
    OpCost cost;
    cost.unit = HWUnit::Scalar;
    cost.cycles = config.getScalarCyclesPerInstruction();
    cost.confidence = cost.cycles > 0 ? CostConfidence::Generic
                                      : CostConfidence::NotModelled;
    return cost;
  }

  // Cube-side fills and transposes cost nothing of their own, and that is a
  // measurement rather than an omission.
  //
  // The profiler's aic_cube_time is the matmul term and nothing else. At
  // 128x128 it reads 92 981.68 us, which at 1.65 GHz is 153 420 272 cycles
  // against the 153 419 776 this model charges for linalg.matmul alone --
  // 496 cycles apart out of 153 million. At 64x64 the same holds. And there is
  // no other engine with room for them: MTE1, MTE2 and the two drain engines
  // are each measured and each already accounted for within a few percent,
  // while these two would add 59.8M cycles.
  //
  // What the hardware does with them follows from that. Zeroing the
  // accumulator is the matmul's own initialisation -- the first MAC along K
  // writes instead of accumulating -- and transposing an operand is folded
  // into the layout conversion that staging it into L0B already performs,
  // whose cost is inside the MTE1 figure this model now matches to 1.4%.
  //
  // Charged as modelled-at-zero rather than dropped, so the report still shows
  // they were seen: "no cycles" and "not noticed" have to stay distinguishable.
  if (isCube && isa<linalg::FillOp, linalg::TransposeOp>(op)) {
    OpCost cost;
    cost.unit = HWUnit::Cube;
    cost.confidence = CostConfidence::Modelled;
    return cost;
  }

  // Remaining shaped compute. Cube-side non-matmul work is charged to the Cube
  // unit; everything else is Vector work. Only the latter is spread over the
  // block's vector cores: the Cube side of a block is one core, so it sees the
  // whole tile.
  OpCost cost = estimateVectorCompute(
      op, config, isCube ? 1 : config.getVectorCoresPerBlock());
  if (isCube) {
    cost.unit = HWUnit::Cube;
  }
  return cost;
}

//===----------------------------------------------------------------------===//
// Loop weighting
//===----------------------------------------------------------------------===//

/// What the caller can supply that the IR itself does not say.
///
/// A loop bounded by a kernel argument -- a sequence length, say -- has no
/// trip count at compile time, yet its body is usually where the kernel spends
/// its time. Counting such a body once understates the estimate by however many
/// iterations run, so there are two ways to fill the gap: bind the arguments to
/// the sizes the kernel will actually be called with, or, failing that, assume
/// a fixed iteration count.
struct TripCountOptions {
  /// Values for entry-function arguments, by argument index.
  llvm::DenseMap<unsigned, int64_t> argBindings;
  /// Values for the launch grid, keyed "pid_x" / "num_programs_x" and so on.
  llvm::StringMap<int64_t> gridBindings;
  /// Iteration count assumed for loops that remain unresolved. One reproduces
  /// the behaviour of not assuming anything.
  int64_t defaultTripCount = 1;
  /// What one pipeline fill/drain iteration costs relative to a full one,
  /// copied from the hardware profile so the loop weighting does not have to
  /// carry the whole config around. See HardwareConfig for why it is a half.
  double prologueWorkFraction = 0.5;
};

/// TritonToLinalg does not lower tl.program_id to an operation: it appends the
/// launch grid to the entry function's arguments and replaces the op with one
/// of them. The last three are program_id x/y/z and the three before them are
/// num_programs x/y/z -- see FunctionConverter.cpp, which does the append.
///
/// That makes the grid bindable through the ordinary argument mechanism, but
/// only if the caller counts arguments by hand. Naming the slots instead is
/// what makes `pid_x=0` work.
constexpr unsigned kLaunchGridRank = 3;

/// Grid slot a function argument corresponds to, or empty if it is an ordinary
/// argument. `numArgs` is the entry function's argument count.
std::string getGridBindingName(unsigned argIndex, unsigned numArgs) {
  if (numArgs < 2 * kLaunchGridRank) {
    return {}; // no launch grid was appended to this function
  }
  static constexpr llvm::StringLiteral kDims[] = {"x", "y", "z"};

  unsigned programIdBase = numArgs - kLaunchGridRank;
  if (argIndex >= programIdBase) {
    return ("pid_" + kDims[argIndex - programIdBase]).str();
  }
  unsigned numProgramsBase = numArgs - 2 * kLaunchGridRank;
  if (argIndex >= numProgramsBase) {
    return ("num_programs_" + kDims[argIndex - numProgramsBase]).str();
  }
  return {};
}

/// Resolve an integer value given bindings for the entry function's arguments.
///
/// Only entry-block arguments of a func.func are bound. This matters: a loop
/// induction variable and a loop-carried value are BlockArguments too, and
/// binding those by index would silently yield a plausible but wrong number.
std::optional<int64_t> evaluateWithBindings(Value value,
                                            const TripCountOptions &options,
                                            int depth = 0) {
  // Guards against a pathological or cyclic expression.
  if (depth > 16) {
    return std::nullopt;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto funcOp =
        dyn_cast_or_null<func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!funcOp || blockArg.getOwner() != &funcOp.getBody().front()) {
      return std::nullopt; // induction variable / iter_arg, not a kernel arg
    }
    unsigned argIndex = blockArg.getArgNumber();
    auto byIndex = options.argBindings.find(argIndex);
    if (byIndex != options.argBindings.end()) {
      return byIndex->second;
    }
    // Only consult the grid slots when the caller named some, so a kernel
    // whose real arguments happen to sit in those positions is unaffected.
    if (!options.gridBindings.empty()) {
      std::string gridName =
          getGridBindingName(argIndex, funcOp.getNumArguments());
      auto byName = options.gridBindings.find(gridName);
      if (!gridName.empty() && byName != options.gridBindings.end()) {
        return byName->second;
      }
    }
    return std::nullopt;
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    return std::nullopt;
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(def)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
    return std::nullopt;
  }

  auto operand = [&](unsigned index) {
    return evaluateWithBindings(def->getOperand(index), options, depth + 1);
  };

  // Pass-through casts.
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::IndexCastOp>(def)) {
    return operand(0);
  }

  if (def->getNumOperands() < 2) {
    return std::nullopt;
  }
  auto lhs = operand(0);
  auto rhs = operand(1);
  if (!lhs || !rhs) {
    return std::nullopt;
  }

  if (isa<arith::AddIOp>(def)) {
    return *lhs + *rhs;
  }
  if (isa<arith::SubIOp>(def)) {
    return *lhs - *rhs;
  }
  if (isa<arith::MulIOp>(def)) {
    return *lhs * *rhs;
  }
  if (isa<arith::DivSIOp, arith::DivUIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    return *lhs / *rhs;
  }
  // Rounding divisions matter more than they look: AddControlFlowCondition
  // rewrites every pipelined loop's upper bound as
  //   lb + step * (ceildiv(ceildiv(ub - lb, step) * buffers, x) + ifCount)
  // so without these a loop with entirely constant bounds still looks dynamic.
  if (isa<arith::CeilDivUIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    auto unsignedLhs = static_cast<uint64_t>(*lhs);
    auto unsignedRhs = static_cast<uint64_t>(*rhs);
    return static_cast<int64_t>((unsignedLhs + unsignedRhs - 1) / unsignedRhs);
  }
  if (isa<arith::CeilDivSIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    int64_t quotient = *lhs / *rhs;
    int64_t remainder = *lhs % *rhs;
    // Truncation rounds towards zero; step back up when the exact result was
    // positive and inexact.
    if (remainder != 0 && ((remainder > 0) == (*rhs > 0))) {
      ++quotient;
    }
    return quotient;
  }
  if (isa<arith::FloorDivSIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    int64_t quotient = *lhs / *rhs;
    int64_t remainder = *lhs % *rhs;
    if (remainder != 0 && ((remainder < 0) != (*rhs < 0))) {
      --quotient;
    }
    return quotient;
  }
  if (isa<arith::RemSIOp, arith::RemUIOp>(def)) {
    if (*rhs == 0) {
      return std::nullopt;
    }
    return *lhs % *rhs;
  }
  if (isa<arith::MinSIOp, arith::MinUIOp>(def)) {
    return std::min(*lhs, *rhs);
  }
  if (isa<arith::MaxSIOp, arith::MaxUIOp>(def)) {
    return std::max(*lhs, *rhs);
  }
  return std::nullopt;
}

/// Trip count of a loop once the supplied bindings are taken into account.
std::optional<int64_t>
resolveTripCount(scf::ForOp forOp, const TripCountOptions &options) {
  // Runs even with no bindings: a bound can be entirely constant yet still
  // defeat the simpler static analysis, because the pipelining rewrite buries
  // it under rounding divisions.
  auto lower = evaluateWithBindings(forOp.getLowerBound(), options);
  auto upper = evaluateWithBindings(forOp.getUpperBound(), options);
  auto step = evaluateWithBindings(forOp.getStep(), options);
  if (!lower || !upper || !step || *step == 0) {
    return std::nullopt;
  }
  if (*upper <= *lower) {
    return 0;
  }
  return (*upper - *lower + *step - 1) / *step;
}

struct LoopWeight {
  int64_t multiplier = 1;
  /// Innermost enclosing loop whose trip count could not be determined, even
  /// with bindings. Null means `multiplier` is trustworthy. The whole body of
  /// such a loop is assumed together, so the loop -- not each operation in it
  /// -- is the thing worth reporting.
  Operation *dynamicLoop = nullptr;
  /// Iteration count assumed for `dynamicLoop`, so the report can say what
  /// the estimate is actually based on.
  int64_t assumedTripCount = 1;
  /// How many mutually exclusive branches the operation sits in, multiplied
  /// over every enclosing two-sided scf.if. Exactly one of those branches runs
  /// per iteration, so charging all of them would multiply the work by this
  /// factor. See computeBranchDivisors for why only two-sided ifs count.
  int64_t branchDivisor = 1;

  /// The operation's contribution over every iteration it runs in.
  int64_t weighted(int64_t cycles) const {
    return cycles * multiplier / branchDivisor;
  }
  /// The same for a single iteration.
  int64_t perIteration(int64_t cycles) const { return cycles / branchDivisor; }
};

/// Divisor to apply to every operation inside an scf.if, keyed by that if.
///
/// The CV pipeline builds multi-buffering out of branches: with two buffers,
/// AllocMultiCache emits the producing operations twice, once per buffer, and
/// AddControlFlowCondition wraps them so exactly one copy runs per iteration.
/// Summing both copies makes multi-buffering look like twice the work, which
/// is why raising inter_core_buf_count used to make a kernel score worse
/// instead of better.
///
/// Both spellings the pipeline emits are handled, and they are not the same
/// question:
///
///   * a two-sided if -- work in both arms -- is a genuine either/or. Over a
///     long loop the predicate alternates, so each arm runs about half the
///     iterations and the honest total is the average of the arms, i.e. every
///     operation in either arm charged at half. For arms of equal cost, which
///     is what buffer rotation produces, that is exactly one arm.
///   * a one-sided if -- a guard with no else, or with an else that does no
///     work -- is a predicated stage, and in steady state such a guard holds
///     on all but the prologue and epilogue iterations. Charging it in full is
///     right to within those few iterations, so it gets a divisor of one.
///
/// Deciding by "does this arm contain work" rather than by "does an else
/// region exist" is what keeps an empty else from halving real work.
///
/// Nested ifs multiply, so a three-deep rotation built out of two-sided ifs
/// comes out at 1/8 rather than 1/3. Buffer counts that high are already
/// warned against by BufferCountManager, so this is left as is.
llvm::DenseMap<Operation *, int64_t>
computeBranchDivisors(ModuleOp module, const HardwareConfig &config) {
  llvm::DenseMap<Operation *, int64_t> divisors;

  auto regionDoesWork = [&](Region &region) {
    bool found = false;
    region.walk([&](Operation *op) {
      auto cost = estimateOpCost(op, config);
      if (cost && cost->cycles > 0) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  };

  module.walk([&](scf::IfOp ifOp) {
    int64_t arms = 0;
    if (regionDoesWork(ifOp.getThenRegion())) {
      ++arms;
    }
    if (!ifOp.getElseRegion().empty() &&
        regionDoesWork(ifOp.getElseRegion())) {
      ++arms;
    }
    // One arm (or none) means there is nothing to choose between.
    divisors[ifOp.getOperation()] = arms > 1 ? arms : 1;
  });
  return divisors;
}

/// How many times an operation executes, given the enclosing loop nest and the
/// exclusive branches it sits in.
///
/// Each loop is resolved in three steps, most trustworthy first: constant
/// bounds in the IR, then the caller's argument bindings, then the assumed
/// default. Only a loop that reaches the third step is reported, since only
/// then is the number a guess.
/// Where a loop's trip count came from, worst case last.
enum class TripCountSource { Static, Bindings, Assumed };

llvm::StringRef stringifyTripCountSource(TripCountSource source) {
  switch (source) {
  case TripCountSource::Static:
    return "static";
  case TripCountSource::Bindings:
    return "bindings";
  case TripCountSource::Assumed:
    return "assumed";
  }
  return "?";
}

struct ResolvedTripCount {
  /// Iterations that carry the body's work. This is what the kernel's author
  /// wrote, and what the report calls "trip".
  int64_t count = 1;
  TripCountSource source = TripCountSource::Assumed;
  /// What the loop bound in the IR says, when that differs from `count`
  /// because the bound was extended for pipelining. Zero when it does not.
  int64_t rewrittenCount = 0;
  /// Software-pipeline stages, one predicated scf.if each, from
  /// ssbuffer.iter_extension[2]. Also the number of iterations spent filling
  /// and draining the pipeline, since a stage costs one iteration to reach.
  int64_t stages = 0;
  /// The buffer rescaling factors, ssbuffer.iter_extension[0] and [1]. Kept
  /// for the report: without them the gap between `count` and `rewrittenCount`
  /// cannot be split into the part that conserves work and the part that does
  /// not, and those two behave completely differently.
  int64_t requiredBuffers = 1;
  int64_t divisor = 1;
  /// Iterations the estimate actually multiplies the body by: `count` plus the
  /// fill and drain, discounted by the profile's prologue fraction. Equals
  /// `count` on a loop that was never pipelined.
  int64_t chargedCount = 1;
};

/// Split the loop extension AddControlFlowCondition applied for pipelining into
/// the part that conserves work and the part that does not.
///
/// It rewrites a pipelined loop's bound to
///     ceildiv(originalIterations * requiredBuffers, x) + ifCount
/// and the two terms mean different things.
///
/// `ceildiv(originalIterations * requiredBuffers, x)` is a rescaling: the same
/// work spread over a different number of iterations, each doing a fraction of
/// what it used to. Taking it at face value charges the whole body for
/// iterations where most of it is disabled. On a loop that is not unrolled the
/// inflation is a couple of percent and invisible; with the main loop unrolled
/// it was observed at 3x, which is enough to rank an unrolled variant far worse
/// than it is. So this term is divided back out, and that is measurable: two
/// configurations differing only in this factor -- ramps of 68 and 4 iterations
/// -- run within 2.4% of each other on the hardware.
///
/// `+ ifCount` is not a rescaling. It is one iteration per predicated stage,
/// spent filling the pipeline at the start and draining it at the end, and
/// those iterations run. Dividing them out too, which is what this used to do,
/// makes a deeper pipeline free of charge -- and a deeper pipeline is exactly
/// what buys the shorter dependency chains the recurrence bound rewards. A
/// search let loose on that combination goes looking for depth: on flash
/// attention it moved the ramp from 3 iterations to 138, scored the result 10%
/// better, and lost 70% on the hardware. So the fill and drain are charged,
/// at the fraction of a full iteration the profile gives.
///
/// The three factors are read from ssbuffer.iter_extension rather than
/// pattern-matched out of the bound expression: the pass that applied them
/// knows them exactly, and a formula change there would silently defeat any
/// matching done here.
///
/// The inversion is exact whenever the ceildiv divided evenly, and off by at
/// most one iteration otherwise -- against a trip count in the hundreds that
/// is noise, and it is always closer than not inverting at all.
struct LoopExtension {
  int64_t workIterations = 0;
  int64_t stages = 0;
  int64_t requiredBuffers = 1;
  int64_t divisor = 1;
};

std::optional<LoopExtension> splitLoopExtension(scf::ForOp forOp,
                                                int64_t rewrittenCount) {
  auto attr = forOp->getAttrOfType<ArrayAttr>(CVPipeline::kIterExtension);
  if (!attr || attr.size() != 3) {
    return std::nullopt;
  }
  auto requiredBuffers = dyn_cast<IntegerAttr>(attr[0]);
  auto divisor = dyn_cast<IntegerAttr>(attr[1]);
  auto ifCount = dyn_cast<IntegerAttr>(attr[2]);
  if (!requiredBuffers || !divisor || !ifCount ||
      requiredBuffers.getInt() <= 0 || divisor.getInt() <= 0 ||
      ifCount.getInt() < 0) {
    return std::nullopt;
  }

  const int64_t scaled = rewrittenCount - ifCount.getInt();
  if (scaled <= 0) {
    return std::nullopt; // nothing but prologue; leave the bound alone
  }
  const int64_t original = scaled * divisor.getInt() / requiredBuffers.getInt();
  if (original <= 0) {
    return std::nullopt;
  }
  LoopExtension split;
  split.workIterations = original;
  split.stages = ifCount.getInt();
  split.requiredBuffers = requiredBuffers.getInt();
  split.divisor = divisor.getInt();
  return split;
}

/// The three-step resolution, in one place so the number the estimate uses and
/// the number the report prints can never disagree.
ResolvedTripCount resolveLoopTripCount(scf::ForOp forOp,
                                       const TripCountOptions &options) {
  ResolvedTripCount resolved;
  auto tripCount = mlir::ascend::utils::analyzeScfForTripCount(forOp);
  if (tripCount.isStatic) {
    resolved.count = tripCount.staticTripCount;
    resolved.source = TripCountSource::Static;
  } else if (auto bound = resolveTripCount(forOp, options)) {
    resolved.count = *bound;
    resolved.source = TripCountSource::Bindings;
  } else {
    // Nothing resolved the bound, so there is no extension to undo either.
    resolved.count = options.defaultTripCount;
    resolved.source = TripCountSource::Assumed;
    resolved.chargedCount = resolved.count;
    return resolved;
  }

  // What was read is the rewritten bound. Part of the difference is a rescaling
  // that conserves work and comes back out; the rest is the pipeline's fill and
  // drain, which runs and is charged.
  if (auto split = splitLoopExtension(forOp, resolved.count)) {
    resolved.rewrittenCount = resolved.count;
    resolved.count = split->workIterations;
    resolved.stages = split->stages;
    resolved.requiredBuffers = split->requiredBuffers;
    resolved.divisor = split->divisor;
  }

  // Rounded up: a ramp that exists costs at least one iteration, and rounding
  // it away is how it became free in the first place.
  const int64_t ramp = static_cast<int64_t>(
      std::ceil(resolved.stages * options.prologueWorkFraction));
  resolved.chargedCount = resolved.count + ramp;
  return resolved;
}

LoopWeight
getLoopWeight(Operation *op, const TripCountOptions &options,
              const llvm::DenseMap<Operation *, int64_t> &branchDivisors) {
  LoopWeight weight;
  // Walking outwards, so the first unresolved loop met is the innermost one.
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      auto found = branchDivisors.find(ifOp.getOperation());
      if (found != branchDivisors.end()) {
        weight.branchDivisor *= found->second;
      }
      continue;
    }

    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp) {
      continue;
    }

    const ResolvedTripCount resolved = resolveLoopTripCount(forOp, options);
    // The charged count, not the work count: an operation inside a pipelined
    // loop is issued during the fill and drain too, just alongside fewer of its
    // neighbours, and that discount is already inside chargedCount.
    weight.multiplier *= resolved.chargedCount;
    if (resolved.source == TripCountSource::Assumed && !weight.dynamicLoop) {
      weight.dynamicLoop = forOp;
      weight.assumedTripCount = resolved.count;
    }
  }
  return weight;
}

/// One loop, as the estimate sees it.
struct LoopReport {
  Operation *loop = nullptr;
  int64_t depth = 0;
  int64_t tripCount = 1;
  TripCountSource source = TripCountSource::Static;
  /// The bound as written in the IR, when pipelining extended it past the
  /// number of iterations that actually do the body's work. Zero otherwise.
  int64_t rewrittenTripCount = 0;
  /// Software-pipeline stages, and the two buffer rescaling factors. Reported
  /// because the gap between `tripCount` and `rewrittenTripCount` has two
  /// causes that behave oppositely -- one conserves work, one adds it -- and
  /// no amount of staring at the two totals separates them.
  int64_t stages = 0;
  int64_t requiredBuffers = 1;
  int64_t divisor = 1;
  /// Iterations actually charged: `tripCount` plus the discounted ramp.
  int64_t chargedTripCount = 1;
  /// What an operation directly in this loop's body gets multiplied by: this
  /// loop's charged count times every enclosing loop's.
  int64_t bodyMultiplier = 1;
  /// Exclusive branches the loop itself sits in, which divide that again.
  int64_t branchDivisor = 1;
};

/// Every loop and what the estimate multiplies its body by.
///
/// Worth reporting even when nothing is wrong, because a loop multiplier is the
/// single largest lever on the result and it is invisible in the totals: an
/// unrolled loop whose bound resolves to the wrong number looks exactly like a
/// kernel that genuinely does more work. Without this the only way to notice is
/// to divide per-operation cycles between two runs by hand.
llvm::SmallVector<LoopReport>
collectLoopReports(ModuleOp module, const TripCountOptions &options,
                   const llvm::DenseMap<Operation *, int64_t> &branchDivisors) {
  llvm::SmallVector<LoopReport> reports;
  module.walk([&](scf::ForOp forOp) {
    LoopReport report;
    report.loop = forOp.getOperation();

    const ResolvedTripCount resolved = resolveLoopTripCount(forOp, options);
    report.tripCount = resolved.count;
    report.source = resolved.source;
    report.rewrittenTripCount = resolved.rewrittenCount;
    report.stages = resolved.stages;
    report.requiredBuffers = resolved.requiredBuffers;
    report.divisor = resolved.divisor;
    report.chargedTripCount = resolved.chargedCount;

    // The loop's own position: enclosing loops multiply, enclosing branches
    // divide. getLoopWeight looks at parents only, which is what is wanted
    // here -- this loop's own charged count is applied on top, and it has to be
    // the charged one so the printed multiplier is the one the estimate used.
    const LoopWeight enclosing =
        getLoopWeight(forOp.getOperation(), options, branchDivisors);
    report.bodyMultiplier = enclosing.multiplier * resolved.chargedCount;
    report.branchDivisor = enclosing.branchDivisor;

    for (Operation *parent = forOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isa<scf::ForOp>(parent)) {
        ++report.depth;
      }
    }
    reports.push_back(report);
  });
  return reports;
}

/// Canonical name for a launch-grid binding. Accepts the spellings the
/// standalone costmodel uses, so the same string works in both. Empty when the
/// name is not a grid slot and should be read as an argument index.
std::string canonicalGridName(llvm::StringRef name) {
  auto isDim = [](llvm::StringRef dim) {
    return dim == "x" || dim == "y" || dim == "z";
  };
  if (name.consume_front("pid_") || name.consume_front("program_id_")) {
    return isDim(name) ? ("pid_" + name).str() : std::string();
  }
  if (name.consume_front("num_programs_")) {
    return isDim(name) ? ("num_programs_" + name).str() : std::string();
  }
  return {};
}

/// Read the trip-count options from the environment.
///
/// Bindings are "arg3=98432,arg5=128" (the "arg" prefix is optional), naming
/// entry-function argument indices. The environment is process-wide, which is
/// fine for inspecting one kernel but is the reason these are not the right
/// channel for per-configuration autotuning; that will want the values passed
/// in explicitly.
TripCountOptions readTripCountOptions() {
  TripCountOptions options;

  if (const char *raw = std::getenv(kArgBindingsEnvVar)) {
    llvm::SmallVector<llvm::StringRef> entries;
    llvm::StringRef(raw).split(entries, ',', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/false);
    for (llvm::StringRef entry : entries) {
      auto [rawName, rawValue] = entry.split('=');
      llvm::StringRef name = rawName.trim();
      int64_t value = 0;
      if (rawValue.trim().getAsInteger(10, value)) {
        LOG_DEBUG("ignoring malformed binding: " << entry);
        continue;
      }
      if (std::string gridName = canonicalGridName(name); !gridName.empty()) {
        options.gridBindings[gridName] = value;
        continue;
      }
      llvm::StringRef indexText = name;
      indexText.consume_front("arg");
      unsigned index = 0;
      if (indexText.getAsInteger(10, index)) {
        LOG_DEBUG("ignoring malformed binding: " << entry);
        continue;
      }
      options.argBindings[index] = value;
    }
  }

  if (const char *raw = std::getenv(kDefaultTripCountEnvVar)) {
    int64_t value = 0;
    if (!llvm::StringRef(raw).trim().getAsInteger(10, value) && value > 0) {
      options.defaultTripCount = value;
    } else {
      LOG_DEBUG("ignoring invalid default trip count: " << raw);
    }
  }

  return options;
}

//===----------------------------------------------------------------------===//
// Intra-block fusion
//===----------------------------------------------------------------------===//

/// How much of a block's summed operation cost is actually paid, per core.
struct FusionFactors {
  double cube = 1.0;
  double vector = 1.0;
  /// Whether the caller asked for any correction at all. Tracked rather than
  /// compared against 1.0 so the report can say "off" without a float equality
  /// test, and so an explicit 1.0 still reads as a deliberate choice.
  bool enabled = false;
};

/// Read one factor, ignoring anything outside (0, 1]. Above 1 would mean a
/// block costs more than its parts, which is not what this models; at or below
/// zero it would make blocks free. Leaves `factor` alone and reports false when
/// the variable is unset or unusable.
bool readFusionFactor(const char *envVar, double &factor) {
  const char *raw = std::getenv(envVar);
  if (!raw) {
    return false;
  }
  double value = 0.0;
  if (llvm::StringRef(raw).trim().getAsDouble(value) || value <= 0.0 ||
      value > 1.0) {
    LOG_DEBUG("ignoring invalid fusion factor for " << envVar << ": " << raw);
    return false;
  }
  factor = value;
  return true;
}

FusionFactors readFusionFactors() {
  FusionFactors factors;
  const bool cubeSet = readFusionFactor(kFusionCubeEnvVar, factors.cube);
  const bool vectorSet = readFusionFactor(kFusionVectorEnvVar, factors.vector);
  factors.enabled = cubeSet || vectorSet;
  return factors;
}

//===----------------------------------------------------------------------===//
// Roofline combination
//===----------------------------------------------------------------------===//

/// Combine per-unit busy cycles into a module-level estimate.
///
/// Ascend overlaps independent pipes, so a path costs as much as its busiest
/// unit rather than the sum of its units. The exception is the AIV MTE2/MTE3
/// pair, which shares one physical pipeline and therefore serialises; the
/// hardware profile declares that via its mutex groups. Cube and Vector cores
/// run concurrently, so the module costs as much as the slower of the two.
int64_t cyclesOfUnit(const llvm::DenseMap<HWUnit, int64_t> &unitCycles,
                     HWUnit unit) {
  auto it = unitCycles.find(unit);
  return it == unitCycles.end() ? 0 : it->second;
}

/// Busy time of the Cube core's own pipes.
///
/// FixPipeUB is Cube-side work even though it lands in Vector memory: it is the
/// Cube's own drain engine. Leaving it out would silently drop its cycles.
int64_t cubePathRoofline(const llvm::DenseMap<HWUnit, int64_t> &unitCycles,
                         const HardwareConfig &config) {
  auto cyclesOf = [&](HWUnit unit) { return cyclesOfUnit(unitCycles, unit); };

  // Two engines drain the accumulator, to HBM and to UB. Modelled as
  // independent unless the profile declares them a mutex clique.
  const int64_t drainCycles =
      config.areMutexUnits("fixpipe", "fixpipe_ub")
          ? cyclesOf(HWUnit::FixPipe) + cyclesOf(HWUnit::FixPipeUB)
          : std::max(cyclesOf(HWUnit::FixPipe), cyclesOf(HWUnit::FixPipeUB));

  // MTE1 and the Cube's scalar issue are engines of their own and overlap the
  // rest: the profiler has aic_mte1_ratio and aic_scalar_ratio adding to well
  // over one alongside aic_cube_ratio, so they run at the same time as the
  // MACs rather than queueing behind them.
  return std::max({cyclesOf(HWUnit::Cube), cyclesOf(HWUnit::CubeMTE2),
                   cyclesOf(HWUnit::CubeMTE1), cyclesOf(HWUnit::ScalarCube),
                   drainCycles});
}

/// Busy time of the Vector core's own pipes.
int64_t vectorPathRoofline(const llvm::DenseMap<HWUnit, int64_t> &unitCycles,
                           const HardwareConfig &config) {
  auto cyclesOf = [&](HWUnit unit) { return cyclesOfUnit(unitCycles, unit); };

  // Off-chip traffic. The AIV load and store movers share one physical
  // pipeline on some parts, which is what the mutex clique in the profile
  // says; there they serialise instead of overlapping.
  const int64_t offChipTransferCycles =
      config.areMutexUnits("vec_mte2", "mte3")
          ? cyclesOf(HWUnit::VecMTE2) + cyclesOf(HWUnit::MTE3)
          : std::max(cyclesOf(HWUnit::VecMTE2), cyclesOf(HWUnit::MTE3));

  // On-chip UB<->L1 staging has movers of its own: UB can be draining into L1
  // at the same time as it streams out to L2/HBM, so this overlaps the traffic
  // above rather than queueing behind it. If a part turns out to share a pipe
  // after all, declaring the units a mutex clique in the hardware profile
  // serialises them with no code change.
  const int64_t onChipTransferCycles =
      config.areMutexUnits("mte1_l1_ub", "mte3_ub_l1")
          ? cyclesOf(HWUnit::MTE1ToUB) + cyclesOf(HWUnit::MTE3ToL1)
          : std::max(cyclesOf(HWUnit::MTE1ToUB), cyclesOf(HWUnit::MTE3ToL1));

  // Scalar keeps its meaning as the Vector core's own issue; the Cube's went
  // to ScalarCube. Same reasoning as on the Cube path: aiv_vec_ratio and
  // aiv_scalar_ratio sum past one, so they overlap.
  return std::max({cyclesOf(HWUnit::Vector), cyclesOf(HWUnit::Scalar),
                   offChipTransferCycles, onChipTransferCycles});
}

int64_t combineRoofline(const llvm::DenseMap<HWUnit, int64_t> &unitCycles,
                        const HardwareConfig &config) {
  return std::max(cubePathRoofline(unitCycles, config),
                  vectorPathRoofline(unitCycles, config));
}

/// Availability slot a unit occupies while scheduling.
///
/// Units that share one physical pipeline must share one slot, or the schedule
/// would let a single engine do two things at once. Which pairs share is a
/// property of the part, declared as a mutex clique in the hardware profile, so
/// this reads the profile rather than hard-coding a topology.
HWUnit getPipeResource(HWUnit unit, const HardwareConfig &config) {
  if (unit == HWUnit::MTE3 && config.areMutexUnits("vec_mte2", "mte3")) {
    return HWUnit::VecMTE2;
  }
  if (unit == HWUnit::MTE3ToL1 &&
      config.areMutexUnits("mte1_l1_ub", "mte3_ub_l1")) {
    return HWUnit::MTE1ToUB;
  }
  if (unit == HWUnit::FixPipeUB &&
      config.areMutexUnits("fixpipe", "fixpipe_ub")) {
    return HWUnit::FixPipe;
  }
  return unit;
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//
// The breakdown exists to answer one question: which operations does the model
// actually understand? Everything charged generically is a candidate for a
// dedicated cost model, and everything with an unknown size silently
// contributes nothing. Both are invisible in the single summary number.

struct OpKindStats {
  int64_t count = 0;
  int64_t weightedCycles = 0;
  HWUnit unit = HWUnit::Scalar;
  CostConfidence confidence = CostConfidence::Modelled;
};

/// Ordering used when several operations of one kind disagree: report the
/// least-known outcome, since that is the one worth acting on.
int confidenceRank(CostConfidence confidence) {
  switch (confidence) {
  case CostConfidence::Modelled:
    return 0;
  case CostConfidence::Generic:
    return 1;
  case CostConfidence::NotModelled:
    return 2;
  case CostConfidence::UnknownSize:
    return 3;
  }
  return 0;
}

/// A loop whose trip count is unknown, summarised by what it contains. One
/// such loop understates the estimate once, not once per operation inside it,
/// so this is the granularity the report uses.
struct DynamicLoopStats {
  int64_t costedOps = 0;
  int64_t bodyCycles = 0;
  int64_t assumedTripCount = 1;
};

/// A run of operations inside one block with no synchronisation between them.
///
/// Fusion is what a segment is for. Operations that issue back to back on one
/// core can keep intermediates out of memory, but a barrier ends that: the core
/// stops, and whatever follows starts a fresh chain. Most blocks turn out to be
/// a single segment -- InterCoreTransferAndSync places its set after the
/// producing block's last operation and its wait before the consuming block's
/// first one -- but the cube-to-vector direct-store path puts a wait right
/// before the store it guards, which can sit anywhere in the block. Splitting
/// here means that case is handled instead of assumed away.
struct BlockSegment {
  llvm::DenseMap<HWUnit, int64_t> unitCycles;        ///< loop-weighted
  llvm::DenseMap<HWUnit, int64_t> unitCyclesOneIter; ///< per iteration
  int64_t costedOps = 0;
};

/// One compute block, as planned by PlanComputeBlock and identified by
/// ssbuffer.block_id. The block is the unit the CV pipeline actually schedules
/// and synchronises, so it is the natural granularity to cost at: operations
/// inside one block run on one core, back to back, and ReorderOpsByBlockId has
/// already made them contiguous in the IR.
struct BlockStats {
  bool isCube = false;
  bool mixedCore = false; ///< set if the block's ops disagree, which is a bug
  int64_t costedOps = 0;
  int64_t workCycles = 0;     ///< sum over operations, loop-weighted
  int64_t loopMultiplier = 1; ///< iterations the block runs for
  llvm::DenseMap<HWUnit, int64_t> unitCycles;
  llvm::DenseMap<HWUnit, int64_t> unitCyclesOneIter;
  llvm::SmallVector<BlockSegment> segments;
  llvm::MapVector<llvm::StringRef, int64_t> opCounts;

  /// Segments that actually carry work. More than one means a barrier landed
  /// in the middle of the block, so it cannot be treated as one fused unit.
  int64_t countWorkingSegments() const {
    int64_t working = 0;
    for (const BlockSegment &segment : segments) {
      if (segment.costedOps > 0) {
        ++working;
      }
    }
    return working;
  }
  bool hasInteriorBarrier() const { return countWorkingSegments() > 1; }
};

/// Synchronisation edges between blocks, read from what the pipeline stamped
/// rather than re-derived.
///
/// InterCoreTransferAndSync and AllocMultiCache tag both ends of every transfer
/// with [groupId, role] -- role 1 produces, 0 consumes, matching
/// InitDependentMap::collectDepsByGroup. Those pairs are the flag waits the
/// hardware will actually perform. SSA cannot show them: after bufferization
/// the two ends are joined through a buffer, not through a value, so a
/// value-based walk sees nothing at all where the real barrier is.
void collectSyncDeps(ModuleOp module,
                     llvm::MapVector<int64_t, llvm::SetVector<int64_t>> &deps) {
  for (llvm::StringLiteral attrName :
       {CVPipeline::kCrossCoreDeps, CVPipeline::kIntraDeps}) {
    // group -> (producing blocks, consuming blocks)
    using BlockSet = llvm::SetVector<int64_t>;
    llvm::MapVector<int64_t, std::pair<BlockSet, BlockSet>> byGroup;

    module.walk([&](Operation *op) {
      auto attr = op->getAttrOfType<ArrayAttr>(attrName);
      if (!attr || attr.size() < 2) {
        return;
      }
      auto groupAttr = dyn_cast<IntegerAttr>(attr[0]);
      auto roleAttr = dyn_cast<IntegerAttr>(attr[1]);
      auto blockId = CVPipeline::getOpBlockId(op);
      if (!groupAttr || !roleAttr || !blockId) {
        return;
      }
      auto &ends = byGroup[groupAttr.getInt()];
      if (roleAttr.getInt() == CVPipeline::crossCoreProducerId) {
        ends.first.insert(*blockId);
      } else {
        ends.second.insert(*blockId);
      }
    });

    for (const auto &[group, ends] : byGroup) {
      for (int64_t consumer : ends.second) {
        for (int64_t producer : ends.first) {
          if (producer != consumer) {
            deps[consumer].insert(producer);
          }
        }
      }
    }
  }
}

/// Dataflow edges between blocks, read from what DataDependencyAnalysis
/// recorded rather than re-derived here.
///
/// That analysis already resolves the cases an SSA walk gets wrong: values
/// carried through scf.for iter_args, producer/consumer pairs joined only by a
/// memory effect, and transposed operands. It is pass-local and holds raw
/// Operation pointers that later passes invalidate by cloning, so it publishes
/// its block-level result to the module under kBlockDeps, keyed by block id --
/// ids survive cloning, pointers do not.
///
/// Two caveats, both conservative here. RefineArgsBlockId can move an operation
/// to a different block afterwards, so an edge may outlive the reason it was
/// added; and blocks created after the analysis carry no edges at all. Either
/// way the schedule only ever loses ordering it could have enforced, which
/// moves the estimate towards the roofline rather than away from it.
void collectDataDeps(ModuleOp module,
                     llvm::MapVector<int64_t, llvm::SetVector<int64_t>> &deps) {
  auto edges = module->getAttrOfType<ArrayAttr>(CVPipeline::kBlockDeps);
  if (!edges) {
    return;
  }
  for (Attribute entry : edges) {
    auto edge = dyn_cast<DictionaryAttr>(entry);
    if (!edge) {
      continue;
    }
    auto producer = edge.getAs<IntegerAttr>("producer");
    auto consumer = edge.getAs<IntegerAttr>("consumer");
    if (!producer || !consumer || producer.getInt() == consumer.getInt()) {
      continue;
    }
    deps[consumer.getInt()].insert(producer.getInt());
  }
}

/// Sentinel for operations the pipeline left without a block id.
constexpr int64_t kNoBlockId = -1;

/// consumer block -> blocks it depends on.
using BlockDepMap = llvm::MapVector<int64_t, llvm::SetVector<int64_t>>;

/// One shape of data transfer: same operation kind, same engine, same pair of
/// address spaces, same size, same loop weighting.
///
/// Grouped rather than listed per operation because the question this answers
/// is whether a kind that got more expensive did so by running more often or
/// by moving more bytes -- and the by-kind table cannot say, since it collapses
/// every engine a kind touched into whichever one was seen last.
struct TransferStats {
  llvm::StringRef kind;
  HWUnit unit = HWUnit::Scalar;
  llvm::StringRef srcSpace;
  llvm::StringRef dstSpace;
  int64_t bytes = 0;
  int64_t cyclesEach = 0;
  int64_t multiplier = 1;
  int64_t branchDivisor = 1;
  int64_t count = 0;
  int64_t weightedCycles = 0;
};

struct CostBreakdown {
  llvm::MapVector<llvm::StringRef, OpKindStats> byOpKind;
  /// Every bulk transfer, grouped. Small enough to search linearly: a kernel
  /// has a handful of distinct transfer shapes, not thousands.
  llvm::SmallVector<TransferStats> transfers;
  llvm::SmallVector<Operation *> unknownSizeOps;
  llvm::MapVector<Operation *, DynamicLoopStats> dynamicLoops;
  llvm::MapVector<int64_t, BlockStats> blocks;
  /// Block -> blocks whose values it reads, as found by
  /// DataDependencyAnalysis. Ordering that no transfer group covers.
  llvm::MapVector<int64_t, llvm::SetVector<int64_t>> dataDeps;
  /// Block -> blocks it waits on through a synchronisation flag. These are
  /// hard barriers: the consumer cannot start until the producer signals.
  llvm::MapVector<int64_t, llvm::SetVector<int64_t>> syncDeps;
  int64_t genericOps = 0;
  /// Operations sitting in a mutually exclusive branch, charged at a fraction
  /// of their cost because only one branch runs. Reported so a jump in the
  /// estimate can be traced to branch structure rather than looking arbitrary.
  int64_t branchedOps = 0;
  /// Scalar instructions issued over the whole run, loop weights applied and
  /// synchronisation excluded (its stall is the schedule's business, not the
  /// scalar unit's). Split by core, because that is how the profiler reports
  /// the time they cost: dividing aic_scalar_time by a count that also
  /// contains the Vector core's instructions gives a number that means
  /// nothing.
  int64_t weightedScalarOpsCube = 0;
  int64_t weightedScalarOpsVector = 0;
  int64_t totalWeightedCycles = 0;

  void recordBlock(Operation *op, const OpCost &cost, const LoopWeight &weight,
                   bool isCube) {
    auto blockId = CVPipeline::getOpBlockId(op);
    int64_t id = blockId ? static_cast<int64_t>(*blockId) : kNoBlockId;

    BlockStats &stats = blocks[id];
    if (stats.costedOps == 0) {
      stats.isCube = isCube;
      stats.loopMultiplier = weight.multiplier;
    } else if (stats.isCube != isCube) {
      stats.mixedCore = true;
    }
    stats.costedOps += 1;
    stats.workCycles += weight.weighted(cost.cycles);
    // Scalar work belongs to the core that issues it. Both cores have their
    // own scalar unit and the profiler reports them apart, so leaving them on
    // one unit would make Cube and Vector blocks queue behind each other on a
    // pipe that does not exist.
    const HWUnit unit = cost.unit == HWUnit::Scalar && isCube
                            ? HWUnit::ScalarCube
                            : cost.unit;
    stats.unitCycles[unit] += weight.weighted(cost.cycles);
    stats.unitCyclesOneIter[unit] += weight.perIteration(cost.cycles);
    if (cost.sideCycles > 0) {
      stats.workCycles += weight.weighted(cost.sideCycles);
      stats.unitCycles[cost.sideUnit] += weight.weighted(cost.sideCycles);
      stats.unitCyclesOneIter[cost.sideUnit] +=
          weight.perIteration(cost.sideCycles);
    }
    stats.opCounts[op->getName().getStringRef()] += 1;

    // emplace_back rather than push_back({}): DenseMap's default constructor
    // is explicit, so brace copy-initialising a struct that contains one does
    // not compile.
    if (stats.segments.empty()) {
      stats.segments.emplace_back();
    }
    if (isSyncOp(op)) {
      // The core stops here, so nothing after this point can fuse with what
      // came before it. A barrier at the very start or end of the block leaves
      // an empty segment, which countWorkingSegments() ignores.
      stats.segments.emplace_back();
    } else {
      BlockSegment &segment = stats.segments.back();
      segment.costedOps += 1;
      segment.unitCycles[unit] += weight.weighted(cost.cycles);
      segment.unitCyclesOneIter[unit] += weight.perIteration(cost.cycles);
      if (cost.sideCycles > 0) {
        segment.unitCycles[cost.sideUnit] += weight.weighted(cost.sideCycles);
        segment.unitCyclesOneIter[cost.sideUnit] +=
            weight.perIteration(cost.sideCycles);
      }
    }
  }

  /// Transfers only: an operation that never went through estimateTransfer
  /// leaves both address spaces empty and is skipped.
  void recordTransfer(Operation *op, const OpCost &cost,
                      const LoopWeight &weight) {
    if (cost.srcSpace.empty() && cost.dstSpace.empty()) {
      return;
    }
    const llvm::StringRef kind = op->getName().getStringRef();
    for (TransferStats &row : transfers) {
      if (row.kind == kind && row.unit == cost.unit &&
          row.srcSpace == cost.srcSpace && row.dstSpace == cost.dstSpace &&
          row.bytes == cost.bytes && row.cyclesEach == cost.cycles &&
          row.multiplier == weight.multiplier &&
          row.branchDivisor == weight.branchDivisor) {
        row.count += 1;
        row.weightedCycles += weight.weighted(cost.cycles);
        return;
      }
    }
    TransferStats row;
    row.kind = kind;
    row.unit = cost.unit;
    row.srcSpace = cost.srcSpace;
    row.dstSpace = cost.dstSpace;
    row.bytes = cost.bytes;
    row.cyclesEach = cost.cycles;
    row.multiplier = weight.multiplier;
    row.branchDivisor = weight.branchDivisor;
    row.count = 1;
    row.weightedCycles = weight.weighted(cost.cycles);
    transfers.push_back(row);
  }

  void record(Operation *op, const OpCost &cost, const LoopWeight &weight,
              bool isCube) {
    recordBlock(op, cost, weight, isCube);
    recordTransfer(op, cost, weight);

    OpKindStats &stats = byOpKind[op->getName().getStringRef()];
    stats.count += 1;
    stats.weightedCycles += weight.weighted(cost.cycles);
    stats.unit = cost.unit;
    if (confidenceRank(cost.confidence) > confidenceRank(stats.confidence)) {
      stats.confidence = cost.confidence;
    }

    totalWeightedCycles += weight.weighted(cost.cycles);
    if ((cost.unit == HWUnit::Scalar || cost.unit == HWUnit::ScalarCube) &&
        !isSyncOp(op)) {
      (isCube ? weightedScalarOpsCube : weightedScalarOpsVector) +=
          weight.weighted(1);
    }
    if (weight.branchDivisor > 1) {
      ++branchedOps;
    }
    if (cost.confidence == CostConfidence::Generic) {
      ++genericOps;
    }
    if (cost.confidence == CostConfidence::UnknownSize) {
      unknownSizeOps.push_back(op);
    }
    // Attribute the operation to its loop rather than listing it separately.
    // Zero-cycle operations are skipped: they cannot understate anything.
    if (weight.dynamicLoop && cost.cycles > 0) {
      DynamicLoopStats &loopStats = dynamicLoops[weight.dynamicLoop];
      loopStats.costedOps += 1;
      loopStats.bodyCycles += weight.weighted(cost.cycles);
      loopStats.assumedTripCount = weight.assumedTripCount;
    }
  }
};

//===----------------------------------------------------------------------===//
// Block scheduling
//===----------------------------------------------------------------------===//
// Three things constrain a block:
//   * pipe availability -- a pipe runs one thing at a time, and pipes that
//     share physical hardware share one slot (getPipeResource);
//   * its barriers -- a block that waits on a flag cannot start before the
//     block that sets it has finished;
//   * segments -- a barrier *inside* a block really does stop the core, so the
//     runs of work either side of it are serial with respect to each other.
// Everything else overlaps, including consecutive blocks on the same core:
// the issue queue moves on while a pipe drains, so block B+1's loads run under
// block B's compute whenever they need different pipes.
//
// An earlier version instead gave each core a single availability slot, so a
// block occupied its whole core for its whole duration. That charges ordinary
// pipe overlap as serial time, and the error is not a constant factor: it grows
// as buffering shrinks, because the pessimism lands hardest on the
// configuration with the least overlap to give away. A model whose error
// depends on the configuration cannot rank configurations.
//
// Note what this means for the two terms of the estimate. The resource bound is
// now partition-invariant: per-pipe busy time does not care how operations were
// grouped, which is physically right. Sensitivity to the partition lives in the
// schedule, and therefore in the recurrence bound -- which is also right, since
// grouping changes where the barriers fall, not how much work there is.

/// Intra-block fusion factor for a block, by the core it runs on. A block that
/// somehow mixes cores gets the more conservative of the two.
double getFusionFactor(const BlockStats &stats, const FusionFactors &fusion) {
  if (stats.mixedCore) {
    return std::max(fusion.cube, fusion.vector);
  }
  return stats.isCube ? fusion.cube : fusion.vector;
}

/// Cost of one block on its own, after intra-block fusion, summed over its
/// segments. Segments are serial with respect to each other (a barrier
/// separates them), so their rooflines add; within a segment the units overlap.
///
/// Reported rather than scheduled: the schedule works per unit, so a block's
/// span there also depends on what the units were already busy with. The
/// difference between this and (finish - start) is contention with neighbours.
int64_t computeBlockCycles(const BlockStats &stats,
                           const HardwareConfig &config,
                           const FusionFactors &fusion, bool perIteration) {
  const double factor = getFusionFactor(stats, fusion);

  int64_t total = 0;
  for (const BlockSegment &segment : stats.segments) {
    const llvm::DenseMap<HWUnit, int64_t> &units =
        perIteration ? segment.unitCyclesOneIter : segment.unitCycles;
    if (units.empty()) {
      continue;
    }
    total += static_cast<int64_t>(combineRoofline(units, config) * factor);
  }
  return total;
}

/// When a block ran, and what held it up.
struct BlockSchedule {
  int64_t start = 0;
  int64_t finish = 0;
  int64_t cycles = 0;
  /// Block whose completion decided this one's start, or kNoBlockId when
  /// nothing did. Reported so a long critical path can be traced by eye.
  int64_t criticalPred = kNoBlockId;
  /// Set when the block was held up by a pipe it needed still being busy,
  /// rather than by a dependency -- that is throughput pressure, not a
  /// serialisation problem, and it is answered by giving the pipe less work
  /// rather than by moving barriers.
  bool waitedOnCore = false;
};

/// Busy time per hardware unit. The roofline over these is the resource bound
/// no schedule can beat, and unlike a per-core total it does not depend on how
/// operations were grouped into blocks -- which is correct: grouping changes
/// when a pipe stalls, not how much work it has to do.
using UnitBusyCycles = llvm::DenseMap<HWUnit, int64_t>;

/// Order the blocks so that a producer is scheduled before its consumers, and
/// so that each core keeps its blocks in the order it will issue them.
///
/// Program order alone will not do. ReorderOpsByBlockId makes the blocks of one
/// core topological among themselves, but SeparateCVScope then groups the
/// module by core, so every Vector block can precede every Cube block in the IR
/// even when a Cube block produces what a Vector block consumes. Scheduling in
/// that order would treat each such edge as loop-carried and drop it --
/// silently discarding exactly the cross-core barriers this model exists to
/// honour.
///
/// Dependency order alone will not do either: it leaves an unconstrained block
/// free to float to the front and fill a gap the hardware does not have, which
/// understates the critical path. Hence the same-core chain below, which pins
/// each core's sequence to program order and lets only cross-core edges move.
///
/// Kahn's algorithm, with ties broken by program order so the result is
/// deterministic. A genuine cycle (a loop-carried dependency) cannot be
/// ordered: the earliest remaining block in program order is emitted anyway
/// and its unsatisfied incoming edges are the ones dropped, which is the
/// intended reading -- this schedules one pass through the graph.
llvm::SmallVector<int64_t>
topologicalBlockOrder(const CostBreakdown &breakdown) {
  llvm::SmallVector<int64_t> programOrder;
  for (const auto &entry : breakdown.blocks) {
    programOrder.push_back(entry.first);
  }

  llvm::MapVector<int64_t, llvm::SmallVector<int64_t>> successors;
  llvm::DenseMap<int64_t, int64_t> inDegree;
  for (int64_t id : programOrder) {
    inDegree[id] = 0;
  }

  llvm::DenseSet<std::pair<int64_t, int64_t>> seenEdge;
  // The two edge kinds overlap: a memory dependency shows up both as a
  // transfer group and as a dataflow edge. Counting it twice would leave a
  // permanent in-degree and force the cycle-breaking path.
  auto addEdge = [&](int64_t producer, int64_t consumer) {
    if (producer == consumer || !inDegree.count(producer) ||
        !inDegree.count(consumer)) {
      return;
    }
    if (!seenEdge.insert({producer, consumer}).second) {
      return;
    }
    successors[producer].push_back(consumer);
    inDegree[consumer] += 1;
  };

  auto addEdges = [&](const BlockDepMap &deps) {
    for (const auto &entry : deps) {
      for (int64_t producer : entry.second) {
        addEdge(producer, entry.first);
      }
    }
  };
  addEdges(breakdown.syncDeps);
  addEdges(breakdown.dataDeps);

  // A core issues its blocks in program order and never reorders them, so
  // consecutive blocks on one core are chained here. Without this the sort is
  // free to slide an unconstrained block into a gap that the hardware does not
  // actually have, which understates the critical path. A block whose
  // operations disagree about their core sits in both chains.
  int64_t lastCube = 0;
  int64_t lastVector = 0;
  bool haveCube = false;
  bool haveVector = false;
  for (int64_t id : programOrder) {
    auto entry = breakdown.blocks.find(id);
    if (entry == breakdown.blocks.end()) {
      continue;
    }
    const BlockStats &stats = entry->second;
    const bool onCube = stats.mixedCore || stats.isCube;
    const bool onVector = stats.mixedCore || !stats.isCube;
    if (onCube) {
      if (haveCube) {
        addEdge(lastCube, id);
      }
      lastCube = id;
      haveCube = true;
    }
    if (onVector) {
      if (haveVector) {
        addEdge(lastVector, id);
      }
      lastVector = id;
      haveVector = true;
    }
  }

  llvm::SmallVector<int64_t> order;
  llvm::DenseSet<int64_t> emitted;
  auto emit = [&](int64_t id) {
    order.push_back(id);
    emitted.insert(id);
    auto it = successors.find(id);
    if (it == successors.end()) {
      return;
    }
    for (int64_t consumer : it->second) {
      inDegree[consumer] -= 1;
    }
  };

  while (order.size() < programOrder.size()) {
    int64_t next = 0;
    bool found = false;
    for (int64_t id : programOrder) {
      if (!emitted.count(id) && inDegree[id] <= 0) {
        next = id;
        found = true;
        break;
      }
    }
    if (!found) {
      // Cyclic: fall back to program order for the rest of this round.
      for (int64_t id : programOrder) {
        if (!emitted.count(id)) {
          next = id;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      break; // cannot happen, but never loop forever on a malformed graph
    }
    emit(next);
  }
  return order;
}

/// ASAP schedule over the block graph, in dependency order, at the granularity
/// of a single hardware pipe.
///
/// A core issues its blocks in order, but it does not wait for one to retire
/// before starting the next: the issue queue moves on while a pipe drains, so
/// block B+1's loads run underneath block B's compute whenever they need
/// different pipes. Treating a block as occupying its whole core -- which is
/// what an earlier version did -- charges that overlap as serial time.
///
/// So each pipe carries its own availability. A block's segments are still
/// serial with respect to each other, because a barrier inside a block really
/// does stop the core; and pipes that share physical hardware share one
/// availability slot, per getPipeResource.
///
/// An edge from a block that has not been scheduled yet survived the
/// topological sort only because it closes a cycle, i.e. it is loop-carried;
/// it is skipped, since this schedules one pass through the graph and the
/// iteration count is applied separately.
/// How much of a schedule was barrier overhead rather than work, so the report
/// can say it instead of leaving the number to be inferred from a gap.
struct BarrierTally {
  int64_t count = 0;
  int64_t cycles = 0;
  void charge(int64_t barrierCycles) {
    ++count;
    cycles += barrierCycles;
  }
};

int64_t scheduleBlockGraph(const CostBreakdown &breakdown,
                           llvm::ArrayRef<int64_t> order,
                           const HardwareConfig &config,
                           const FusionFactors &fusion, bool perIteration,
                           llvm::MapVector<int64_t, BlockSchedule> &schedule,
                           UnitBusyCycles &busy, BarrierTally &barriers) {
  llvm::DenseMap<HWUnit, int64_t> pipeAvailable;
  int64_t makespan = 0;
  const int64_t barrierCycles = config.getBarrierCycles();

  for (int64_t blockId : order) {
    auto blockEntry = breakdown.blocks.find(blockId);
    if (blockEntry == breakdown.blocks.end()) {
      continue;
    }
    const int64_t id = blockId;
    const BlockStats &stats = blockEntry->second;

    int64_t earliest = 0;
    int64_t criticalPred = kNoBlockId;
    bool waitedOnFlag = false;
    auto considerDeps = [&](const BlockDepMap &deps, bool isSync) {
      auto it = deps.find(id);
      if (it == deps.end()) {
        return;
      }
      for (int64_t producer : it->second) {
        auto scheduled = schedule.find(producer);
        if (scheduled == schedule.end()) {
          continue; // back edge: the producer runs in a later pass
        }
        if (isSync) {
          waitedOnFlag = true;
        }
        if (scheduled->second.finish > earliest) {
          earliest = scheduled->second.finish;
          criticalPred = producer;
        }
      }
    };
    // Flags first: those are the barriers the hardware really enforces.
    considerDeps(breakdown.syncDeps, /*isSync=*/true);
    considerDeps(breakdown.dataDeps, /*isSync=*/false);

    // A flag wait is not free even once the producer has finished: the flag
    // has to be written, propagated and observed. Only synchronisation edges
    // are charged -- a dataflow edge inside one core is ordered by the issue
    // queue, with no flag involved.
    if (waitedOnFlag && barrierCycles > 0) {
      earliest += barrierCycles;
      barriers.charge(barrierCycles);
    }

    const double factor = getFusionFactor(stats, fusion);

    // `cursor` is when the current segment may begin: the dependency-ready
    // time for the first, and the previous segment's end after that.
    int64_t cursor = earliest;
    int64_t firstStart = earliest;
    bool sawWork = false;
    bool waitedOnPipe = false;

    bool sawWorkingSegment = false;
    for (const BlockSegment &segment : stats.segments) {
      const llvm::DenseMap<HWUnit, int64_t> &units =
          perIteration ? segment.unitCyclesOneIter : segment.unitCycles;
      // A barrier separates this segment from the previous one, so the core
      // pays for it again here. Charged only between segments that both carry
      // work: an empty segment is an artefact of two adjacent sync operations,
      // not a second stop.
      if (segment.costedOps > 0) {
        if (sawWorkingSegment && barrierCycles > 0) {
          cursor += barrierCycles;
          barriers.charge(barrierCycles);
        }
        sawWorkingSegment = true;
      }
      int64_t segmentFinish = cursor;
      for (const auto &entry : units) {
        const int64_t cycles = static_cast<int64_t>(entry.second * factor);
        if (cycles <= 0) {
          continue;
        }
        const HWUnit pipe = getPipeResource(entry.first, config);
        int64_t &available = pipeAvailable[pipe];
        if (available > cursor) {
          waitedOnPipe = true;
        }
        const int64_t begin = std::max(cursor, available);
        available = begin + cycles;
        segmentFinish = std::max(segmentFinish, available);
        busy[entry.first] += cycles;
        if (!sawWork || begin < firstStart) {
          firstStart = begin;
        }
        sawWork = true;
      }
      cursor = segmentFinish;
    }

    makespan = std::max(makespan, cursor);
    BlockSchedule &result = schedule[id];
    result.start = firstStart;
    result.finish = cursor;
    result.cycles = computeBlockCycles(stats, config, fusion, perIteration);
    result.criticalPred = criticalPred;
    result.waitedOnCore = waitedOnPipe;
  }
  return makespan;
}

/// How deeply the pipeline is buffered, i.e. how many iterations can be in
/// flight across a dependency before the producer has to wait for a buffer to
/// come free.
///
/// These are read from the module, not assumed. BufferCountManager stamps them
/// (and Python overrides them per configuration through set_buffer_count), so
/// the attribute is the value this compilation actually used. The fallbacks
/// below only apply when the attribute is missing altogether, and match
/// BufferCountManager's own defaults -- inter-core is *one*, meaning nothing
/// overlaps across a Cube/Vector barrier at all.
/// Most inter-core buffering the pipeline can actually produce.
///
/// AddMultiBufferOuterScope reduces the count to a yes/no --
/// `isDoubleBuf = (interCoreBufNum > 1)` -- and never looks at the number
/// again, so asking for three buffers yields exactly the IR that two do.
/// Dividing by a larger number here would credit the schedule with an overlap
/// the IR does not contain, and would make the model prefer a configuration
/// that compiles to something identical.
constexpr int64_t kMaxUsefulInterCoreDepth = 2;

struct BufferDepths {
  int64_t intra = 2;
  int64_t inter = 1;
  /// What the module asked for, before the clamp above. Kept so the report can
  /// say that a larger value was requested and had no effect, rather than
  /// leaving that to be discovered by experiment.
  int64_t requestedInter = 1;
  /// Read only to report it. Nothing here divides by it: GM-load prefetching
  /// is the business of DecoupleComputeAndMemory, which this pipeline does not
  /// run, so raising it changes neither the IR nor the estimate.
  int64_t requestedLoad = 1;
  /// The pipeline asked for inter-core double buffering and the flag budget
  /// refused it, so the IR is single-buffered whatever the count says.
  bool interDowngraded = false;

  /// Depth that gates an edge: crossing cores goes through an inter-core
  /// buffer, staying on one core through an intra-core one.
  int64_t forEdge(bool crossesCores) const {
    return crossesCores ? inter : intra;
  }
};

/// Buffer depths as this compilation actually used them.
///
/// The intra-core count is taken at face value: AddMultiBufferInnerScope
/// creates exactly that many UB allocations from it. The inter-core count is
/// clamped, because the pipeline only distinguishes one from more than one.
BufferDepths readBufferDepths(ModuleOp module) {
  BufferDepths depths;
  auto read = [&](llvm::StringLiteral name, int64_t &slot) {
    if (auto attr = module->getAttrOfType<IntegerAttr>(name)) {
      if (attr.getInt() >= 1) {
        slot = attr.getInt();
      }
    }
  };
  read(CVPipeline::kIntraBufCount, depths.intra);
  read(CVPipeline::kInterCoreBufCount, depths.requestedInter);
  read(CVPipeline::kLoadStoreBufCount, depths.requestedLoad);
  depths.inter = std::min(depths.requestedInter, kMaxUsefulInterCoreDepth);
  // AddMultiBufferOuterScope compiles single-buffered when the double-buffer
  // groups would not fit in the flag budget, and leaves the count attribute
  // alone while doing it. Dividing by two here would then credit an overlap
  // that is not in the IR -- and hide the cost of the very thing that caused
  // the downgrade, since one extra flag is all it takes.
  if (module->hasAttr(CVPipeline::kInterCoreBufDowngraded)) {
    depths.interDowngraded = true;
    depths.inter = 1;
  }
  return depths;
}

/// How many synchronisation flags the compiled module uses, and how many are
/// live at the same time.
///
/// The hardware has a small fixed set of set/wait flags and the pipeline
/// allocates out of it: FlagIdManager caps ids at 14, at 7 while inter-core
/// double buffering is on, and AnalyzeFlag abandons the whole CV pipeline for
/// any id outside that range. So the budget is a hard limit on how finely
/// blocks may be split -- and it appears in no other number this report
/// prints, which is why a variant can fail to compile, or quietly lose its
/// double buffering, for a reason the estimate never mentions.
///
/// Two counts, answering different questions. Distinct ids is what the module
/// spends as it stands. Peak live is what it would need if ids were reassigned
/// as well as possible: FlagIdReuseManager colours an interference graph, so
/// what it can get away with is that graph's chromatic number, and the peak
/// number of overlapping lifetimes is a lower bound on it.
///
/// Lifetimes are taken in program order here -- a flag is live from its first
/// set to its last wait. The allocator uses reachability, which can prove two
/// flags disjoint where program order cannot, so this peak is never lower than
/// what the allocator needs and may be higher. It is therefore reported and
/// nothing else: rejecting a variant on a number known to be pessimistic would
/// discard variants that would have compiled.
struct FlagCensus {
  int distinct = 0;
  int peakLive = 0;
  int highestId = -1;
};

FlagCensus takeFlagCensus(ModuleOp module) {
  FlagCensus census;

  // Flag id -> [index of its first set/wait, index of its last], counted over
  // sync operations only so the indices are dense in what matters.
  llvm::DenseMap<int, std::pair<int64_t, int64_t>> span;
  int64_t index = 0;
  module.walk([&](Operation *op) {
    if (!isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(op)) {
      return;
    }
    int id = -1;
    // The pipeline spells this three ways depending on which pass wrote it;
    // getFlagFromSyncOp in AddMultiBufferOuterScope reads the same three.
    for (llvm::StringRef name : {"flag_id", "static_flag_id", "flag"}) {
      if (auto attr = op->getAttrOfType<IntegerAttr>(name)) {
        id = static_cast<int>(attr.getInt());
        break;
      }
    }
    if (id < 0) {
      return;
    }
    ++index;
    auto it = span.find(id);
    if (it == span.end()) {
      span[id] = {index, index};
    } else {
      it->second.second = index;
    }
    census.highestId = std::max(census.highestId, id);
  });

  census.distinct = static_cast<int>(span.size());

  // Sweep the lifetimes. Every index belongs to exactly one operation and so
  // to one flag, which is why opens and closes of different flags never tie;
  // the +1 before -1 ordering only matters for a flag whose set and wait are
  // the same operation index.
  llvm::SmallVector<std::pair<int64_t, int>> events;
  events.reserve(span.size() * 2);
  for (const auto &entry : span) {
    events.push_back({entry.second.first, +1});
    events.push_back({entry.second.second, -1});
  }
  llvm::sort(events, [](const std::pair<int64_t, int> &a,
                        const std::pair<int64_t, int> &b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    return a.second > b.second;
  });
  int live = 0;
  for (const auto &event : events) {
    live += event.second;
    census.peakLive = std::max(census.peakLive, live);
  }
  return census;
}

/// How much Unified Buffer the module asks for.
///
/// This is the constraint that decides whether a block partition can exist at
/// all, and nothing in the estimate knew about it. Splitting a run of vector
/// work into more blocks materialises the values that cross the new boundary:
/// a score tile consumed in place inside one block has to become a real buffer
/// once a barrier lands in the middle of it. So a finer partition buys overlap
/// with UB, and a variant can be ranked best here and then be rejected outright
/// by the binary compiler -- which is exactly what happened to a candidate that
/// needed 345 600 bytes against the 253 952 the part has.
///
/// Two numbers, because the useful one needs the schedule and the other does
/// not. The plain sum counts buffers that never coexist and so is wrong by more
/// than a factor of two on this kernel -- 526 336 bytes summed against a part
/// that has 253 952 and compiles the module anyway. The peak is what the
/// hardware has to hold at once, and it is a computation rather than a guess:
/// every allocation's size is exact, and the interval over which it is live is
/// read off the same block schedule the recurrence bound uses.
///
/// Neither figure equals what the binary compiler will demand. That compiler
/// multi-buffers again on top of what it is handed, aligns to banks, and adds
/// temporaries after this pass has run. What survives all of that is the
/// *difference* between two variants of one kernel, which is why a consumer
/// compares a candidate against the untouched pipeline rather than against a
/// capacity.
struct UBAllocation {
  int64_t bytes = 0;
  /// Compute blocks that touch the buffer, in no particular order: only the
  /// first and last of them in schedule order are used.
  llvm::SmallVector<int64_t, 4> blocks;
};

struct UBFootprint {
  int64_t bytes = 0;
  int64_t peakBytes = 0;
  int64_t allocations = 0;
  llvm::SmallVector<UBAllocation, 8> perAllocation;
};

UBFootprint measureUBFootprint(ModuleOp module) {
  UBFootprint footprint;
  module.walk([&](memref::AllocOp alloc) {
    if (getMemorySpaceName(alloc.getType()) != "ub") {
      return;
    }
    auto bytes = getShapedByteSize(alloc.getType());
    if (!bytes) {
      return;
    }

    UBAllocation entry;
    entry.bytes = *bytes;

    // Everything that can reach the buffer, following the casts and views that
    // hand it on without copying it: a block holding nothing but a
    // reinterpret_cast still keeps the buffer alive for whoever reads the
    // result. Following them over-states the live range where a view outlives
    // its use, which widens intervals rather than narrowing them -- the safe
    // direction, since a peak that is too wide rejects a variant that would
    // have fitted only if the same widening did not also apply to the baseline
    // it is compared against, and it does.
    llvm::SmallVector<Value, 8> worklist{alloc.getResult()};
    llvm::SmallPtrSet<Operation *, 16> seen;
    llvm::SmallDenseSet<int64_t, 4> blocks;
    auto note = [&](Operation *op) {
      if (auto id = CVPipeline::getOpBlockId(op)) {
        blocks.insert(static_cast<int64_t>(*id));
      }
    };
    note(alloc);
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      for (Operation *user : value.getUsers()) {
        if (!seen.insert(user).second) {
          continue;
        }
        note(user);
        for (Value result : user->getResults()) {
          if (isa<ShapedType>(result.getType())) {
            worklist.push_back(result);
          }
        }
      }
    }

    entry.blocks.assign(blocks.begin(), blocks.end());
    footprint.bytes += entry.bytes;
    ++footprint.allocations;
    footprint.perAllocation.push_back(std::move(entry));
  });
  return footprint;
}

/// Most Unified Buffer live at any one point of the block schedule.
///
/// A buffer is taken to be live from the first block that touches it to the
/// last, which is the same span the recurrence bound measures for a transfer,
/// and for the same reason: nothing may reuse the storage in between. Swept
/// with a difference array over the schedule, so the cost is linear in blocks.
void computeUBPeak(UBFootprint &footprint, ArrayRef<int64_t> order) {
  if (order.empty() || footprint.perAllocation.empty()) {
    // No schedule to be live against; the sum is the only honest answer.
    footprint.peakBytes = footprint.bytes;
    return;
  }

  llvm::DenseMap<int64_t, size_t> position;
  for (size_t i = 0; i < order.size(); ++i) {
    position[order[i]] = i;
  }

  llvm::SmallVector<int64_t> delta(order.size() + 1, 0);
  for (const UBAllocation &entry : footprint.perAllocation) {
    size_t lo = order.size();
    size_t hi = 0;
    bool placed = false;
    for (int64_t id : entry.blocks) {
      auto it = position.find(id);
      if (it == position.end()) {
        continue;
      }
      lo = std::min(lo, it->second);
      hi = std::max(hi, it->second);
      placed = true;
    }
    if (!placed) {
      // Touched only by operations the pipeline left unassigned. Nothing says
      // when it dies, so it is charged for the whole schedule.
      lo = 0;
      hi = order.size() - 1;
    }
    delta[lo] += entry.bytes;
    delta[hi + 1] -= entry.bytes;
  }

  int64_t live = 0;
  for (size_t i = 0; i < order.size(); ++i) {
    live += delta[i];
    footprint.peakBytes = std::max(footprint.peakBytes, live);
  }
}

/// Whether an edge between two blocks crosses the Cube/Vector boundary, and so
/// goes through an inter-core buffer rather than an intra-core one. A block
/// whose operations disagree about their core touches both, so any edge to it
/// is treated as crossing -- the tighter of the two readings.
bool edgeCrossesCores(const BlockStats &producer, const BlockStats &consumer) {
  if (producer.mixedCore || consumer.mixedCore) {
    return true;
  }
  return producer.isCube != consumer.isCube;
}

/// The buffer whose reuse constrains the kernel most.
///
/// Every dependency between blocks is carried by a buffer, and that buffer is
/// occupied from the moment its producer starts writing until its consumer has
/// finished reading. The next iteration cannot reuse it until then, so with B
/// buffers allocated for that transfer the kernel cannot go faster than
/// occupancy/B.
///
/// The bound is taken per edge, not over a whole chain: each transfer has its
/// own buffers, so a producer waits only for *its* consumer to release *its*
/// buffer, not for the far end of the chain. Treating the chain as one buffer
/// would charge the alternation several times over.
///
/// Occupancy is read off the schedule of a **single iteration**, and the result
/// is therefore a lower bound on the initiation interval, not a whole-module
/// figure: the caller scales it.
///
/// Measuring it on the loop-weighted schedule instead, as an earlier version
/// did, is wrong in a way that is easy to miss. There each block occupies its
/// entire N-iteration duration before the next one starts, so the span from a
/// producer's start to its consumer's finish covers most of the kernel. That is
/// not a buffer's lifetime -- a buffer is held for one iteration's worth of
/// producer-to-consumer span and then reused. Measuring it that way makes the
/// bound grow with the trip count, so it eventually beats the resource bound on
/// any long enough loop and the model starts crediting extra buffers with gains
/// the hardware cannot deliver.
///
/// Every kind of dependency counts: Cube to Vector, Cube to Cube, Vector to
/// Vector. Only the depth differs, since crossing cores goes through an
/// inter-core buffer and staying on one core through an intra-core one.
struct BufferRecurrence {
  int64_t producer = kNoBlockId;
  int64_t consumer = kNoBlockId;
  int64_t occupancy = 0; ///< how long the buffer is held, for one iteration
  int64_t depth = 1;
  int64_t bound = 0; ///< occupancy / depth: the recurrence-constrained II
  bool crossesCores = false;
};

BufferRecurrence tightestBufferRecurrence(
    const CostBreakdown &breakdown,
    const llvm::MapVector<int64_t, BlockSchedule> &schedule,
    const BufferDepths &depths) {
  BufferRecurrence tightest;

  auto consider = [&](const BlockDepMap &deps) {
    for (const auto &entry : deps) {
      const int64_t consumerId = entry.first;
      auto consumerSlot = schedule.find(consumerId);
      auto consumerStats = breakdown.blocks.find(consumerId);
      if (consumerSlot == schedule.end() ||
          consumerStats == breakdown.blocks.end()) {
        continue;
      }
      for (int64_t producerId : entry.second) {
        auto producerSlot = schedule.find(producerId);
        auto producerStats = breakdown.blocks.find(producerId);
        if (producerSlot == schedule.end() ||
            producerStats == breakdown.blocks.end()) {
          continue;
        }
        const int64_t occupancy =
            consumerSlot->second.finish - producerSlot->second.start;
        if (occupancy <= 0) {
          continue; // the consumer was scheduled first: a loop-carried edge
        }
        const bool crosses =
            edgeCrossesCores(producerStats->second, consumerStats->second);
        const int64_t depth = depths.forEdge(crosses);
        const int64_t bound = occupancy / std::max<int64_t>(1, depth);
        if (bound > tightest.bound) {
          tightest.producer = producerId;
          tightest.consumer = consumerId;
          tightest.occupancy = occupancy;
          tightest.depth = depth;
          tightest.bound = bound;
          tightest.crossesCores = crosses;
        }
      }
    }
  };
  consider(breakdown.syncDeps);
  consider(breakdown.dataDeps);
  return tightest;
}

/// Everything the module estimate is assembled from, kept together so the
/// report can show how the final number was reached rather than just assert it.
struct ModuleEstimate {
  /// Roofline over the per-pipe busy times: the busiest pipe decides, and no
  /// schedule can beat it. Independent of how operations were grouped into
  /// blocks, which is correct -- grouping changes stalls, not workload.
  int64_t throughputBound = 0;
  /// The same bound for a single iteration: the initiation interval.
  int64_t initiationInterval = 0;
  /// One pass through the block graph with barriers and pipe contention
  /// respected. Unlike throughputBound this *does* depend on the grouping.
  int64_t iterationLatency = 0;
  /// The full schedule run on loop-weighted block costs: what the kernel costs
  /// if nothing whatsoever overlaps between iterations.
  int64_t serialisedBound = 0;
  /// Buffer depths as recorded on the module by BufferCountManager.
  BufferDepths buffers;
  /// Synchronisation flags spent, against a budget nothing else here shows.
  FlagCensus flags;
  /// Unified Buffer asked for, against the capacity that decides whether this
  /// partition compiles at all.
  UBFootprint ub;
  /// The buffer whose reuse constrains the kernel most, and its bound.
  BufferRecurrence recurrence;
  int64_t recurrenceBound = 0;
  int64_t total = 0;

  /// Blocks in the order they were scheduled: dependency order, not IR order.
  llvm::SmallVector<int64_t> order;
  llvm::MapVector<int64_t, BlockSchedule> iterationSchedule;
  /// The same schedule on loop-weighted costs, which is what buffer occupancy
  /// is measured against.
  llvm::MapVector<int64_t, BlockSchedule> weightedSchedule;
  UnitBusyCycles weightedBusy;
  UnitBusyCycles iterationBusy;
  /// Barriers charged in the one-iteration schedule, which is the schedule the
  /// recurrence bound is measured on. Zero whenever the profile leaves
  /// barrier_cycles at its default.
  BarrierTally iterationBarriers;
};

/// Combine the block schedule into one number, as the larger of two bounds.
///
///   resource bound   -- the busiest hardware pipe's total busy time. No
///                       schedule beats it, whatever the block partition.
///   recurrence bound -- the tightest dependency chain, costed for every
///                       iteration and divided by how many iterations its
///                       buffers let run at once.
///
/// The second is what a plain modulo-scheduling formula gets wrong here.
/// "total = II * (N - 1) + latency" assumes the pipeline reaches steady state,
/// i.e. that buffering is deep enough to hide the chain; the inter-core depth
/// defaults to one, so on a kernel that alternates Cube and Vector nothing is
/// hidden and the chain is paid every iteration. Taking the maximum reproduces
/// both extremes: deep buffering leaves the resource bound standing, a single
/// buffer gives the serialised cost, and the two cross over on their own.
///
/// The initiation interval and one-iteration latency are still computed, but
/// as diagnostics -- with blocks running different numbers of times they
/// cannot be composed into a single global "II * (N - 1)".
ModuleEstimate computeModuleEstimate(const CostBreakdown &breakdown,
                                     const HardwareConfig &config,
                                     const FusionFactors &fusion,
                                     const BufferDepths &buffers,
                                     const FlagCensus &flags,
                                     const UBFootprint &ub) {
  ModuleEstimate estimate;
  estimate.buffers = buffers;
  estimate.flags = flags;
  estimate.ub = ub;
  estimate.order = topologicalBlockOrder(breakdown);
  // Needs the order, so it cannot be done where the allocations were collected.
  computeUBPeak(estimate.ub, estimate.order);

  BarrierTally weightedBarriers;
  estimate.serialisedBound = scheduleBlockGraph(
      breakdown, estimate.order, config, fusion, /*perIteration=*/false,
      estimate.weightedSchedule, estimate.weightedBusy, weightedBarriers);
  estimate.throughputBound = combineRoofline(estimate.weightedBusy, config);

  estimate.iterationLatency = scheduleBlockGraph(
      breakdown, estimate.order, config, fusion, /*perIteration=*/true,
      estimate.iterationSchedule, estimate.iterationBusy,
      estimate.iterationBarriers);
  estimate.initiationInterval = combineRoofline(estimate.iterationBusy, config);

  // Measured on one iteration, so the result is a lower bound on the
  // initiation interval. Scaling it by throughputBound/II puts it in the same
  // units as the resource bound without needing a single global iteration
  // count -- that ratio *is* the iteration count, expressed through the two
  // numbers already computed.
  estimate.recurrence = tightestBufferRecurrence(
      breakdown, estimate.iterationSchedule, buffers);
  estimate.recurrenceBound =
      estimate.initiationInterval > 0
          ? estimate.throughputBound * estimate.recurrence.bound /
                estimate.initiationInterval
          : 0;
  estimate.total =
      std::max(estimate.throughputBound, estimate.recurrenceBound);
  return estimate;
}

/// Cap on how many individual operations are listed per category, so a large
/// kernel produces a readable report rather than a wall of text.
constexpr size_t kMaxListedOps = 20;

void printOpList(llvm::raw_ostream &os, llvm::StringRef title,
                 llvm::ArrayRef<Operation *> ops) {
  if (ops.empty()) {
    return;
  }
  os << "[" << DEBUG_TYPE << "] " << ops.size() << " " << title << ":\n";
  for (size_t i = 0; i < ops.size() && i < kMaxListedOps; ++i) {
    os << "[" << DEBUG_TYPE << "]     "
       << ops[i]->getName().getStringRef() << "  at " << ops[i]->getLoc()
       << "\n";
  }
  if (ops.size() > kMaxListedOps) {
    os << "[" << DEBUG_TYPE << "]     ... and " << (ops.size() - kMaxListedOps)
       << " more\n";
  }
}

/// Busiest unit in a block, which is what its own estimate is bounded by.
HWUnit getBlockBottleneck(const BlockStats &stats) {
  HWUnit worst = HWUnit::Scalar;
  int64_t worstCycles = -1;
  for (const auto &[unit, cycles] : stats.unitCycles) {
    if (cycles > worstCycles) {
      worstCycles = cycles;
      worst = unit;
    }
  }
  return worst;
}

/// Short, readable summary of what a block contains: the heaviest few kinds,
/// then a count of the rest, so a large block stays one line.
std::string describeBlockContents(const BlockStats &stats) {
  llvm::SmallVector<std::pair<llvm::StringRef, int64_t>> kinds(
      stats.opCounts.begin(), stats.opCounts.end());
  llvm::sort(kinds, [](const auto &lhs, const auto &rhs) {
    return lhs.second > rhs.second;
  });

  constexpr size_t kMaxShown = 3;
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (size_t i = 0; i < kinds.size() && i < kMaxShown; ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << kinds[i].first;
    if (kinds[i].second > 1) {
      stream << " x" << kinds[i].second;
    }
  }
  if (kinds.size() > kMaxShown) {
    stream << ", +" << (kinds.size() - kMaxShown) << " more kind(s)";
  }
  return stream.str();
}

std::string describeBlockId(int64_t id) {
  return id == kNoBlockId ? std::string("--") : std::to_string(id);
}

llvm::StringRef describeBlockCore(const BlockStats &stats) {
  return stats.mixedCore ? "MIXED" : stats.isCube ? "CUBE" : "VECTOR";
}

/// Per-block schedule. The block is what the pipeline schedules and
/// synchronises, so this is the view that maps onto what it decided: the
/// per-operation table says what things cost, this says when they run and what
/// they had to wait for.
///
/// Rows are ordered by start time rather than by block id or IR position, so
/// the table reads as a timeline and a critical path can be followed downwards
/// through the "waits" column.
void printBlocks(llvm::raw_ostream &os, const CostBreakdown &breakdown,
                 const ModuleEstimate &estimate) {
  if (breakdown.blocks.empty()) {
    return;
  }

  os << "[" << DEBUG_TYPE
     << "] block schedule for one iteration (ssbuffer.block_id, as planned by"
        " PlanComputeBlock). Each pipe runs one thing at a time, so blocks"
        " overlap wherever they need different pipes; only barriers order them."
        " 'cycles' is what the block costs alone after fusion, so finish-start"
        " above it is contention with neighbours. 'waits' names the block whose"
        " completion decided this one's start ('core' = a pipe it needed was"
        " still busy):\n";
  os << "[" << DEBUG_TYPE
     << "]    block  core     cycles      start     finish  seg  ops  waits"
        "   bottleneck   contents\n";

  // Sorted by start time so the table reads as a timeline and the critical
  // path can be followed downwards through the "waits" column.
  struct Row {
    int64_t id;
    const BlockStats *stats;
    const BlockSchedule *slot;
  };
  llvm::SmallVector<Row> rows;
  for (const auto &entry : breakdown.blocks) {
    auto scheduled = estimate.iterationSchedule.find(entry.first);
    if (scheduled != estimate.iterationSchedule.end()) {
      rows.push_back({entry.first, &entry.second, &scheduled->second});
    }
  }
  llvm::sort(rows, [](const Row &lhs, const Row &rhs) {
    if (lhs.slot->start != rhs.slot->start) {
      return lhs.slot->start < rhs.slot->start;
    }
    return lhs.slot->finish < rhs.slot->finish;
  });

  for (const Row &row : rows) {
    const int64_t id = row.id;
    const BlockStats &stats = *row.stats;
    const BlockSchedule &slot = *row.slot;

    std::string waits = slot.waitedOnCore
                            ? std::string("core")
                            : (slot.criticalPred == kNoBlockId
                                   ? std::string("-")
                                   : describeBlockId(slot.criticalPred));
    const int64_t segments = stats.countWorkingSegments();
    // Held in a local: right_justify keeps a StringRef, not a copy.
    const std::string blockName = describeBlockId(id);

    os << "[" << DEBUG_TYPE << "] " << llvm::right_justify(blockName, 8) << "  "
       << llvm::left_justify(describeBlockCore(stats), 7)
       << llvm::format("%9lld", static_cast<long long>(slot.cycles))
       << llvm::format("%11lld", static_cast<long long>(slot.start))
       << llvm::format("%11lld", static_cast<long long>(slot.finish))
       << llvm::format("%5lld", static_cast<long long>(segments))
       << llvm::format("%5lld", static_cast<long long>(stats.costedOps)) << "  "
       << llvm::left_justify(waits, 7)
       << llvm::left_justify(
              mlir::ascend::stringifyHWUnit(getBlockBottleneck(stats)), 13)
       << describeBlockContents(stats) << "\n";
  }

  // Blocks whose barrier is not at an edge. These are the ones that break the
  // "a block is one fused chain" assumption, so their count is worth knowing
  // even when it is zero -- especially then, since it says the simple model
  // holds for this kernel.
  llvm::SmallVector<std::string> interiorBarrierBlocks;
  for (const auto &entry : breakdown.blocks) {
    if (entry.second.hasInteriorBarrier()) {
      interiorBarrierBlocks.push_back(describeBlockId(entry.first));
    }
  }
  os << "[" << DEBUG_TYPE << "] " << interiorBarrierBlocks.size()
     << " block(s) contain a barrier between two runs of work, so they are"
        " fused per segment rather than as a whole";
  if (!interiorBarrierBlocks.empty()) {
    os << ": ";
    llvm::interleaveComma(interiorBarrierBlocks, os);
  }
  os << "\n";

  auto printDeps =
      [&](const llvm::MapVector<int64_t, llvm::SetVector<int64_t>> &deps,
          llvm::StringRef tag, llvm::StringRef what) {
        if (deps.empty()) {
          return;
        }
        os << "[" << DEBUG_TYPE << "] block " << tag << " (" << what << "):\n";
        for (const auto &[id, producers] : deps) {
          os << "[" << DEBUG_TYPE << "]     block " << id << " <- ";
          llvm::interleaveComma(producers, os);
          os << "\n";
        }
      };
  // Sync edges first: those are the ones that actually stall a core.
  printDeps(breakdown.syncDeps, "sync",
            "waits on a flag the other block sets");
  printDeps(breakdown.dataDeps, "dataflow",
            "reads a value the other block produced");
}

/// The per-pipe busy times again, grouped and named the way the hardware
/// profiler names them.
///
/// Comparing an estimate against msprof is otherwise a manual exercise every
/// time: the model counts cycles per HWUnit, the profiler reports microseconds
/// per column, and the two vocabularies do not line up. Printing the prediction
/// in the profiler's own terms makes calibration a subtraction -- which is how
/// the fixpipe bandwidth turned out to be three times too slow and how a
/// UB->L1 copy was found charged to the Cube's load engine.
///
/// Two columns are listed with nothing in them on purpose. A blank there is a
/// statement: the model has no term for that hardware at all, so whatever the
/// profiler reports in it is unaccounted for rather than mispredicted.
void printPipeComparison(llvm::raw_ostream &os,
                         const llvm::DenseMap<HWUnit, int64_t> &busy,
                         const HardwareConfig &config) {
  auto cyclesOf = [&](HWUnit unit) { return cyclesOfUnit(busy, unit); };

  struct Row {
    const char *column;
    int64_t cycles;
    const char *from;
  };
  const Row rows[] = {
      {"aic_mac_time", cyclesOf(HWUnit::Cube), "cube"},
      {"aic_mte2_time", cyclesOf(HWUnit::CubeMTE2), "cube_mte2"},
      {"aic_fixpipe_time",
       cyclesOf(HWUnit::FixPipe) + cyclesOf(HWUnit::FixPipeUB),
       "fixpipe + fixpipe_ub"},
      {"aic_mte1_time", cyclesOf(HWUnit::CubeMTE1), "cube_mte1"},
      {"aiv_vec_time", cyclesOf(HWUnit::Vector), "vector"},
      {"aiv_mte2_time", cyclesOf(HWUnit::VecMTE2) + cyclesOf(HWUnit::MTE1ToUB),
       "vec_mte2 + mte1_l1_ub"},
      {"aiv_mte3_time", cyclesOf(HWUnit::MTE3) + cyclesOf(HWUnit::MTE3ToL1),
       "mte3 + mte3_ub_l1"},
      {"aic_scalar_time", cyclesOf(HWUnit::ScalarCube), "scalar_cube"},
      {"aiv_scalar_time", cyclesOf(HWUnit::Scalar), "scalar"},
  };

  os << "[" << DEBUG_TYPE
     << "]     predicted busy time per pipe, named as msprof names it, so a"
        " profiler row can be subtracted from this one without translating:\n";
  for (const Row &row : rows) {
    os << "[" << DEBUG_TYPE << "]       "
       << llvm::left_justify(row.column, 21)
       << llvm::format("%12.1f us", config.cyclesToMicroseconds(row.cycles))
       << "   " << row.from << "\n";
  }
}

/// How the module number was assembled. Printed as a derivation rather than a
/// result so that a surprising total can be attributed to a term.
void printEstimate(llvm::raw_ostream &os, const ModuleEstimate &estimate,
                   const FusionFactors &fusion, const HardwareConfig &config,
                   int64_t branchedOps, int64_t scalarOpsCube,
                   int64_t scalarOpsVector, int64_t pipelineStages) {
  auto line = [&](llvm::StringRef what, int64_t cycles) {
    os << "[" << DEBUG_TYPE << "]     " << llvm::left_justify(what, 42)
       << llvm::format("%14lld", static_cast<long long>(cycles)) << " cycles ("
       << llvm::format("%.3f", config.cyclesToMicroseconds(cycles)) << " us)\n";
  };

  const bool resourceBound =
      estimate.throughputBound >= estimate.recurrenceBound;

  os << "[" << DEBUG_TYPE
     << "] how the estimate is built (the total is the larger of the two"
        " bounds):\n";
  line(resourceBound ? "> resource bound: busiest pipe"
                     : "  resource bound: busiest pipe",
       estimate.throughputBound);
  line("    Cube path", cubePathRoofline(estimate.weightedBusy, config));
  line("    Vector path", vectorPathRoofline(estimate.weightedBusy, config));

  // Which pipe actually decides, and by how much. A pipe that is only just
  // ahead means the kernel is balanced and the bound will move under any
  // change; one far ahead of the rest is the thing worth attacking.
  llvm::SmallVector<std::pair<HWUnit, int64_t>> pipes(
      estimate.weightedBusy.begin(), estimate.weightedBusy.end());
  llvm::sort(pipes, [](const auto &lhs, const auto &rhs) {
    return lhs.second > rhs.second;
  });
  os << "[" << DEBUG_TYPE << "]     busiest pipes:";
  for (size_t i = 0; i < pipes.size() && i < 4; ++i) {
    if (pipes[i].second <= 0) {
      break;
    }
    os << (i ? ", " : " ") << mlir::ascend::stringifyHWUnit(pipes[i].first)
       << " " << pipes[i].second;
  }
  os << "\n";
  printPipeComparison(os, estimate.weightedBusy, config);
  line(resourceBound ? "  recurrence bound: buffer reuse"
                     : "> recurrence bound: buffer reuse",
       estimate.recurrenceBound);
  const BufferRecurrence &tight = estimate.recurrence;
  if (tight.bound > 0) {
    os << "[" << DEBUG_TYPE << "]         tightest buffer: block "
       << describeBlockId(tight.producer) << " -> "
       << describeBlockId(tight.consumer) << ", held " << tight.occupancy
       << " cycles per iteration, "
       << (tight.crossesCores ? "inter-core" : "intra-core") << " depth "
       << tight.depth << " -> " << tight.bound << " cycles per iteration\n";
    // The comparison that decides everything: a buffer that turns over faster
    // than the busiest pipe is not what limits the kernel, however few copies
    // of it there are.
    os << "[" << DEBUG_TYPE << "]         against an initiation interval of "
       << estimate.initiationInterval << " cycles, so the buffer is "
       << (tight.bound > estimate.initiationInterval ? "the constraint"
                                                     : "not the constraint")
       << "\n";
  }
  line("  every dependency serialised, no overlap", estimate.serialisedBound);
  line("TOTAL", estimate.total);

  os << "[" << DEBUG_TYPE << "]     buffer depths read from the module: intra "
     << estimate.buffers.intra << ", inter-core " << estimate.buffers.inter
     << "\n";
  if (estimate.flags.distinct > 0) {
    // The budget is a hardware limit, not a cost: exceeding it does not make
    // the kernel slow, it makes it not compile (AnalyzeFlag) or silently
    // single-buffered (AddMultiBufferOuterScope). Printed next to the buffer
    // depths because the two interact -- double buffering halves the budget.
    const int budget = estimate.buffers.inter > 1 ? 8 : 15;
    os << "[" << DEBUG_TYPE << "]     synchronisation flags: "
       << estimate.flags.distinct << " distinct, highest id "
       << estimate.flags.highestId << ", at most " << estimate.flags.peakLive
       << " live at once, against a budget of " << budget
       << (estimate.buffers.inter > 1 ? " (halved by inter-core double"
                                        " buffering)"
                                      : "")
       << ". Live count is by program order, so it is an over-estimate of what"
          " the allocator's interference colouring would need\n";
  }
  if (estimate.ub.allocations > 0) {
    const MemorySpace *ub = config.getMemorySpace("ub");
    os << "[" << DEBUG_TYPE << "]     unified buffer: "
       << estimate.ub.allocations << " allocation(s), " << estimate.ub.peakBytes
       << " bytes live at the peak of the schedule, " << estimate.ub.bytes
       << " summed";
    if (ub && ub->sizeBytes > 0) {
      os << ", against a capacity of " << ub->sizeBytes;
    }
    os << ". The peak is the one to read: sizes are exact and live ranges come"
          " from the block schedule, so the difference between two variants of"
          " one kernel is the buffers a finer partition forced into existence."
          " The absolute value is not what the binary compiler will ask for --"
          " it multi-buffers again on top, aligns to banks, and adds"
          " temporaries after this pass -- which is why a search compares"
          " candidates against the untouched pipeline and not against the"
          " capacity\n";
  }
  if (scalarOpsCube + scalarOpsVector > 0) {
    const int perInstruction = config.getScalarCyclesPerInstruction();
    os << "[" << DEBUG_TYPE << "]     scalar issue: " << scalarOpsCube
       << " instruction(s) on the Cube core and " << scalarOpsVector
       << " on the Vector core over the run, at " << perInstruction
       << " cycle(s) each";
    if (perInstruction == 0) {
      // The counts are the useful half. One profiled run turns them into the
      // constant: aic_scalar_time belongs to the first count and
      // aiv_scalar_time to the second, so the measured microseconds times the
      // clock and divided by the matching count is the figure to set. Guessing
      // it here instead would put an unmeasured number into every estimate.
      os << ", so nothing is charged for them. aic_scalar_time divided by the"
            " first count and aiv_scalar_time by the second give the figure"
            " for scalar_issue.cycles_per_instruction -- if the two disagree,"
            " the cost is not per instruction and a per-block term is what is"
            " missing";
    }
    os << "\n";
  }
  if (estimate.buffers.interDowngraded) {
    os << "[" << DEBUG_TYPE
       << "]       note: inter-core double buffering was asked for and refused"
          " -- the synchronisation flags it needs do not fit the hardware"
          " budget, so AddMultiBufferOuterScope compiled single-buffered."
          " Costed at depth 1, which is what the IR contains\n";
  } else if (estimate.buffers.requestedInter > estimate.buffers.inter) {
    os << "[" << DEBUG_TYPE << "]       note: inter_core_buf_count="
       << estimate.buffers.requestedInter
       << " was requested, but the pipeline only distinguishes 1 from more"
          " than 1 -- it compiles to the same IR as "
       << estimate.buffers.inter << ", so that is what is costed\n";
  }
  if (estimate.buffers.requestedLoad > 1) {
    os << "[" << DEBUG_TYPE << "]       note: load_store_buf_count="
       << estimate.buffers.requestedLoad
       << " was requested, but GM-load prefetching is applied by"
          " DecoupleComputeAndMemory, which this pipeline does not run, so it"
          " changes neither the IR nor this estimate\n";
  }

  // Multi-buffering is built out of branches, so raising a buffer count adds
  // copies of the producing operations. Without this line the estimate would
  // move for reasons nothing else in the report explains.
  os << "[" << DEBUG_TYPE << "]     " << branchedOps
     << " operation(s) in mutually exclusive branches, charged at a fraction"
        " of their cost because only one branch runs per iteration\n";

  // What the software pipeline's ramp costs. Unlike a barrier this is
  // occupancy, not latency: the fill and drain are extra iterations of the
  // loop, so they scale every operation inside it and move both bounds. Worth
  // saying next to the barrier line, because these are the two things a finer
  // block partition buys its overlap with, and this is the larger of the two.
  if (pipelineStages > 0) {
    const double fraction = config.getPrologueWorkFraction();
    os << "[" << DEBUG_TYPE << "]     software pipeline: " << pipelineStages
       << " stage(s) in the deepest loop, so "
       << static_cast<long long>(std::ceil(pipelineStages * fraction))
       << " extra iteration(s) charged for filling and draining it"
       << llvm::format(" (at %.2f of a full iteration each, from"
                       " software_pipeline.prologue_work_fraction)",
                       fraction)
       << "\n";
  }

  // What each barrier costs, and what that came to. A barrier is latency, not
  // occupancy: it stops the core from issuing rather than keeping an engine
  // busy, so it moves the schedule and through it the recurrence bound, and
  // can never move the resource bound. Worth saying, because at zero a finer
  // block partition is free and any search over partitions degenerates.
  os << "[" << DEBUG_TYPE << "]     barrier cost: ";
  if (config.getBarrierCycles() <= 0) {
    os << "0 cycles (off; set synchronisation.barrier_cycles in the hardware"
          " profile -- until then splitting a block costs nothing)\n";
  } else {
    os << config.getBarrierCycles() << " cycles each, "
       << estimate.iterationBarriers.count << " crossed per iteration, "
       << estimate.iterationBarriers.cycles << " cycles";
    if (estimate.iterationLatency > 0) {
      os << llvm::format(" (%.1f%% of the one-iteration schedule)",
                         100.0 *
                             static_cast<double>(
                                 estimate.iterationBarriers.cycles) /
                             static_cast<double>(estimate.iterationLatency));
    }
    os << "\n";
  }

  os << "[" << DEBUG_TYPE << "]     per iteration, for reference: II "
     << estimate.initiationInterval << " cycles, barrier chain "
     << estimate.iterationLatency << " cycles";
  if (estimate.initiationInterval > 0) {
    os << llvm::format(" (%.2fx II)",
                       static_cast<double>(estimate.iterationLatency) /
                           static_cast<double>(estimate.initiationInterval));
  }
  os << "\n";

  // The gap to the roofline is what the barriers cost. Zero means either that
  // nothing orders Cube against Vector, or that buffering hides all of it.
  const int64_t penalty = estimate.total - estimate.throughputBound;
  os << "[" << DEBUG_TYPE << "]     charged by barriers over the roofline: "
     << penalty << " cycles";
  if (estimate.throughputBound > 0) {
    os << llvm::format(" (%.1f%%)", 100.0 * static_cast<double>(penalty) /
                                        static_cast<double>(
                                            estimate.throughputBound));
  }
  os << "\n";

  os << "[" << DEBUG_TYPE << "]     intra-block fusion factors: cube="
     << llvm::format("%.3f", fusion.cube) << ", vector="
     << llvm::format("%.3f", fusion.vector);
  if (!fusion.enabled) {
    os << " (off; set " << kFusionCubeEnvVar << " / " << kFusionVectorEnvVar
       << " to enable)";
  }
  os << "\n";
}

/// Every loop, its trip count, and where that number came from.
///
/// The multiplier column is the one to read when two runs of the same kernel
/// disagree by a suspiciously round factor: it is what every operation in that
/// loop's body is scaled by, so an error here moves the whole estimate without
/// showing up anywhere else in the report.
void printLoops(llvm::raw_ostream &os,
                llvm::ArrayRef<LoopReport> loops) {
  if (loops.empty()) {
    return;
  }

  os << "[" << DEBUG_TYPE
     << "] loops, and what each multiplies its body by. 'trip' is how many"
        " times this loop's body does its work and 'source' is how that was"
        " obtained -- static means the IR said so, bindings means an argument"
        " value supplied by the caller resolved it, assumed means nothing did"
        " and the default was used.\n";
  os << "[" << DEBUG_TYPE
     << "]   The remaining columns appear when AddControlFlowCondition"
        " pipelined the loop and rewrote its bound to"
        " ceildiv(trip * rb, x) + stages. 'buf' is that rb/x: a rescaling that"
        " spreads the same work over more iterations, so it is divided back out"
        " and costs nothing. 'stages' is the predicated scf.if count, and it is"
        " also the iterations spent filling and draining the pipeline -- those"
        " do run, so 'charged' is trip plus that ramp at the profile's prologue"
        " fraction, and 'charged' is what 'body x' and the whole estimate are"
        " built on. 'ir bound' is the bound as actually written in the IR.\n";
  os << "[" << DEBUG_TYPE
     << "]    depth        trip    buf  stages     charged    ir bound"
        "  source       body x  location\n";

  for (const LoopReport &loop : loops) {
    os << "[" << DEBUG_TYPE << "] "
       << llvm::format("%8lld", static_cast<long long>(loop.depth))
       << llvm::format("%12lld", static_cast<long long>(loop.tripCount));

    const bool pipelined = loop.rewrittenTripCount > 0;
    if (pipelined) {
      const std::string buf = std::to_string(loop.requiredBuffers) + "/" +
                              std::to_string(loop.divisor);
      os << llvm::right_justify(buf, 7)
         << llvm::format("%8lld", static_cast<long long>(loop.stages))
         << llvm::format("%12lld",
                         static_cast<long long>(loop.chargedTripCount))
         << llvm::format("%12lld",
                         static_cast<long long>(loop.rewrittenTripCount));
    } else {
      os << llvm::right_justify("-", 7) << llvm::right_justify("-", 8)
         << llvm::format("%12lld",
                         static_cast<long long>(loop.chargedTripCount))
         << llvm::right_justify("-", 12);
    }

    os << "  " << llvm::left_justify(stringifyTripCountSource(loop.source), 11)
       << llvm::format("%11lld", static_cast<long long>(loop.bodyMultiplier));
    if (loop.branchDivisor > 1) {
      os << " /" << loop.branchDivisor;
    }
    os << "  " << loop.loop->getLoc() << "\n";
  }
}

/// Every distinct transfer the kernel performs: which engine, which path,
/// how many bytes, and how often.
///
/// This is the view the by-kind table cannot give. There, one row per
/// operation kind hides both the engine (`record()` keeps only the last one
/// seen, so a kind split across engines shows one of them) and the size (a
/// kind whose total doubled may be moving twice the bytes or running twice as
/// often, and those call for opposite fixes). Bandwidths differ by more than
/// 4x between paths on this profile, so which path a transfer took decides
/// most of its cost.
void printTransfers(llvm::raw_ostream &os, const CostBreakdown &breakdown) {
  if (breakdown.transfers.empty()) {
    return;
  }

  llvm::SmallVector<TransferStats> rows(breakdown.transfers.begin(),
                                        breakdown.transfers.end());
  llvm::sort(rows, [](const TransferStats &lhs, const TransferStats &rhs) {
    return lhs.weightedCycles > rhs.weightedCycles;
  });

  os << "[" << DEBUG_TYPE
     << "] data transfers, grouped by engine, path and size. 'each' is one"
        " execution and 'x' how many times the loops run it, so 'cycles' is"
        " each x count x, divided by the branch divisor where one applies."
        " Two transfers of the same operation kind can sit on engines whose"
        " bandwidths differ several-fold, which the table above cannot show:\n";
  os << "[" << DEBUG_TYPE
     << "]     count       bytes       each          x        cycles  unit    "
        "     path       operation\n";

  for (const TransferStats &row : rows) {
    // Built into a local: left_justify keeps a StringRef, not a copy.
    std::string path = row.srcSpace.str();
    path += ":";
    path += row.dstSpace.str();

    os << "[" << DEBUG_TYPE << "] "
       << llvm::format("%9lld", static_cast<long long>(row.count))
       << llvm::format("%12lld", static_cast<long long>(row.bytes))
       << llvm::format("%11lld", static_cast<long long>(row.cyclesEach))
       << llvm::format("%11lld", static_cast<long long>(row.multiplier))
       << llvm::format("%14lld", static_cast<long long>(row.weightedCycles))
       << "  "
       << llvm::left_justify(mlir::ascend::stringifyHWUnit(row.unit), 11)
       << llvm::left_justify(path, 11) << row.kind;
    if (row.branchDivisor > 1) {
      os << "  (/" << row.branchDivisor << ", one branch of "
         << row.branchDivisor << " runs)";
    }
    os << "\n";
  }
}

void printBreakdown(llvm::raw_ostream &os, const CostBreakdown &breakdown) {
  // Heaviest first: that is the order in which teaching the model new
  // operations pays off.
  llvm::SmallVector<std::pair<llvm::StringRef, OpKindStats>> rows;
  for (const auto &entry : breakdown.byOpKind) {
    rows.push_back({entry.first, entry.second});
  }
  llvm::sort(rows, [](const auto &lhs, const auto &rhs) {
    return lhs.second.weightedCycles > rhs.second.weightedCycles;
  });

  const double total =
      breakdown.totalWeightedCycles > 0
          ? static_cast<double>(breakdown.totalWeightedCycles)
          : 1.0;

  os << "[" << DEBUG_TYPE << "] cost breakdown by operation kind"
     << " (share is of total work, not of the estimate):\n";
  os << "[" << DEBUG_TYPE << "]     count       cycles   share  confidence    "
     << "unit        operation\n";
  for (const auto &[name, stats] : rows) {
    os << "[" << DEBUG_TYPE << "] "
       << llvm::format("%9lld", static_cast<long long>(stats.count))
       << llvm::format("%13lld", static_cast<long long>(stats.weightedCycles))
       << llvm::format("%7.1f%%", 100.0 * stats.weightedCycles / total) << "  "
       // "not-modelled" is itself 12 wide, so 12 leaves no gap at all.
       << llvm::left_justify(stringifyConfidence(stats.confidence), 14)
       << llvm::left_justify(mlir::ascend::stringifyHWUnit(stats.unit), 12)
       << name << "\n";
  }

  // The actionable lists.
  auto listKinds = [&](CostConfidence confidence, llvm::StringRef what) {
    llvm::SmallVector<llvm::StringRef> kinds;
    for (const auto &[name, stats] : rows) {
      if (stats.confidence == confidence) {
        kinds.push_back(name);
      }
    }
    if (kinds.empty()) {
      return;
    }
    os << "[" << DEBUG_TYPE << "] " << kinds.size() << " operation kind(s) "
       << what << ":\n";
    for (llvm::StringRef name : kinds) {
      os << "[" << DEBUG_TYPE << "]     " << name << "\n";
    }
  };

  printTransfers(os, breakdown);

  listKinds(CostConfidence::Generic,
            "charged by element count only, i.e. with no dedicated cost model");
  listKinds(CostConfidence::NotModelled,
            "recognised but charged zero cycles of their own. For the "
            "synchronisation ops that is now right: the stall they cause is "
            "modelled by the block schedule instead, so charging the op too "
            "would count it twice. Scalar work really is unaccounted for");

  printOpList(os, "operation(s) with a non-static shape, contributing 0 cycles",
              breakdown.unknownSizeOps);

  // Loops, not their contents: one loop with an unknown trip count is one
  // problem, however many operations it happens to contain.
  if (!breakdown.dynamicLoops.empty()) {
    os << "[" << DEBUG_TYPE << "] " << breakdown.dynamicLoops.size()
       << " loop(s) whose trip count no binding resolved; the count below is"
          " assumed, so the estimate is only as good as that assumption. Set "
       << kArgBindingsEnvVar << " to bind the kernel arguments that bound them,"
          " or "
       << kDefaultTripCountEnvVar << " to change the assumption:\n";
    for (const auto &[loop, stats] : breakdown.dynamicLoops) {
      os << "[" << DEBUG_TYPE << "]     " << stats.costedOps
         << " costed op(s), " << stats.bodyCycles << " cycles over an assumed "
         << stats.assumedTripCount << " iteration(s)  at " << loop->getLoc()
         << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Verbosity requested through the environment: 0 = silent, 1 = one-line
/// summary, >=2 = summary plus the per-operation breakdown.
int getVerbosity() {
  const char *raw = std::getenv(kVerboseEnvVar);
  if (!raw) {
    return 0;
  }
  int level = std::atoi(raw);
  // Any non-numeric but present value means "on"; only an explicit 0 is off.
  if (level == 0 && raw[0] != '0') {
    return 1;
  }
  return level;
}

void estimateModuleCost(ModuleOp module, llvm::StringRef hardwareConfigPath) {
  // An explicitly constructed path wins; otherwise the environment selects the
  // profile, so a run on another target does not silently get 910B numbers.
  std::string configPath = hardwareConfigPath.str();
  if (configPath.empty()) {
    if (const char *fromEnv = std::getenv(kHardwareConfigEnvVar)) {
      configPath = fromEnv;
    }
  }

  std::string configError;
  auto config =
      mlir::ascend::loadHardwareConfigForAnalysis(configPath, configError);
  if (!config) {
    LOG_DEBUG("failed to load hardware config: " << configError);
    return;
  }

  TripCountOptions tripCountOptions = readTripCountOptions();
  // The one loop-weighting input that is a property of the part rather than of
  // the caller, so it comes from the profile and not from the environment.
  tripCountOptions.prologueWorkFraction = config->getPrologueWorkFraction();
  const FusionFactors fusion = readFusionFactors();
  // Worked out once up front: deciding whether an scf.if is a real either/or
  // means looking at both its arms, which cannot be done while walking a
  // single operation.
  const llvm::DenseMap<Operation *, int64_t> branchDivisors =
      computeBranchDivisors(module, *config);
  const llvm::SmallVector<LoopReport> loopReports =
      collectLoopReports(module, tripCountOptions, branchDivisors);

  llvm::DenseMap<HWUnit, int64_t> unitCycles;
  CostBreakdown breakdown;
  int64_t nextOpId = 0;
  int64_t unknownOps = 0;

  module.walk([&](Operation *op) {
    auto cost = estimateOpCost(op, *config);
    if (!cost) {
      return;
    }

    LoopWeight weight = getLoopWeight(op, tripCountOptions, branchDivisors);
    // Counts operations whose own cost could not be computed. An operation in
    // a loop of unknown trip count is not one of those: its per-iteration cost
    // is known, it is the iteration count that is not, which is accounted for
    // per loop rather than per operation.
    if (cost->confidence == CostConfidence::UnknownSize) {
      ++unknownOps;
    }
    breakdown.record(op, *cost, weight, runsOnCubeCore(op));
    ++nextOpId;

    unitCycles[cost->unit] += weight.weighted(cost->cycles);
  });

  // Both edge kinds are properties of the module, not of any one operation, so
  // they are read once rather than per operation. Synchronisation edges come
  // from the pipeline's transfer groups; dataflow edges from what
  // DataDependencyAnalysis published.
  collectSyncDeps(module, breakdown.syncDeps);
  collectDataDeps(module, breakdown.dataDeps);

  if (nextOpId == 0) {
    LOG_DEBUG("no costed operations found; skipping estimate");
    return;
  }

  // The estimate is assembled from the blocks, not from a single roofline over
  // every operation: the whole point is that regrouping the same operations
  // into different blocks has to change the number.
  const ModuleEstimate estimate =
      computeModuleEstimate(breakdown, *config, fusion,
                            readBufferDepths(module), takeFlagCensus(module),
                            measureUBFootprint(module));
  const int64_t totalCycles = estimate.total;

  // Kept for comparison: what the model reported before blocks constrained it,
  // i.e. Cube and Vector assumed to overlap unconditionally.
  const int64_t rooflineCycles = combineRoofline(unitCycles, *config);

  Builder builder(module.getContext());
  module->setAttr(kCVPipelineEstimatedCycles,
                  builder.getI64IntegerAttr(totalCycles));
  module->setAttr(kCVPipelineCostRoofline,
                  builder.getI64IntegerAttr(rooflineCycles));
  // Separately, so a consumer can see which of the two the total came from and
  // how much room the other one had.
  module->setAttr(kCVPipelineCostResource,
                  builder.getI64IntegerAttr(estimate.throughputBound));
  module->setAttr(kCVPipelineCostRecurrence,
                  builder.getI64IntegerAttr(estimate.recurrenceBound));
  module->setAttr(kCVPipelineCostUBBytes,
                  builder.getI64IntegerAttr(estimate.ub.bytes));
  module->setAttr(kCVPipelineCostUBPeak,
                  builder.getI64IntegerAttr(estimate.ub.peakBytes));
  // Deepest software-pipeline stretch, over every loop. Diagnostic: it is the
  // one number directly comparable with the bound written in the IR, but it
  // mixes the rescaling with the ramp, so the two below separate them.
  int64_t loopExtension = 0;
  int64_t pipelineStages = 0;
  for (const LoopReport &loop : loopReports) {
    if (loop.rewrittenTripCount > loop.tripCount) {
      loopExtension =
          std::max(loopExtension, loop.rewrittenTripCount - loop.tripCount);
    }
    pipelineStages = std::max(pipelineStages, loop.stages);
  }
  module->setAttr(kCVPipelineCostLoopExtension,
                  builder.getI64IntegerAttr(loopExtension));
  // Stage count: a cost, already inside the estimate, published so a consumer
  // that sees an expensive variant can tell at a glance whether depth is why.
  module->setAttr(kCVPipelineCostPipelineStages,
                  builder.getI64IntegerAttr(pipelineStages));

  // The buffer rescaling factor, which is a correctness limit rather than a
  // cost: the binary compiler refuses a module whose prologue outruns the
  // buffers behind it, and that shows up here as a step from 1 to 2. A search
  // guards on this one; it does not guard on the stage count above, which is
  // paid for in the score instead.
  int64_t pipelineDepth = 1;
  module.walk([&](scf::ForOp forOp) {
    auto attr = forOp->getAttrOfType<ArrayAttr>(CVPipeline::kIterExtension);
    if (!attr || attr.size() != 3) {
      return;
    }
    if (auto required = dyn_cast<IntegerAttr>(attr[0])) {
      pipelineDepth = std::max(pipelineDepth, required.getInt());
    }
  });
  module->setAttr(kCVPipelineCostPipelineDepth,
                  builder.getI64IntegerAttr(pipelineDepth));

  if (const MemorySpace *ubSpace = config->getMemorySpace("ub")) {
    module->setAttr(
        kCVPipelineCostUBCapacity,
        builder.getI64IntegerAttr(static_cast<int64_t>(ubSpace->sizeBytes)));
  }
  module->setAttr(kCVPipelineCostHardware,
                  builder.getStringAttr(config->getName()));
  module->setAttr(kCVPipelineCostUnknownOps,
                  builder.getI64IntegerAttr(unknownOps));
  module->setAttr(kCVPipelineCostGenericOps,
                  builder.getI64IntegerAttr(breakdown.genericOps));
  module->setAttr(
      kCVPipelineCostDynamicLoops,
      builder.getI64IntegerAttr(
          static_cast<int64_t>(breakdown.dynamicLoops.size())));

  // Per-block results, machine-readable. Kept on the module rather than on the
  // operations themselves: a block is a set of operations sharing an id, not an
  // IR entity, and stamping every operation would bury the IR in annotations.
  if (!breakdown.blocks.empty()) {
    llvm::SmallVector<Attribute> blockAttrs;
    for (const auto &[id, stats] : breakdown.blocks) {
      // A structured binding cannot be captured by a lambda before C++20, so
      // the id is copied into an ordinary local first.
      const int64_t blockId = id;
      auto collect = [&](const BlockDepMap &deps) {
        llvm::SmallVector<Attribute> result;
        auto it = deps.find(blockId);
        if (it != deps.end()) {
          for (int64_t producer : it->second) {
            result.push_back(builder.getI64IntegerAttr(producer));
          }
        }
        return result;
      };
      llvm::SmallVector<Attribute> dependsOn = collect(breakdown.dataDeps);
      llvm::SmallVector<Attribute> syncsWith = collect(breakdown.syncDeps);

      // Where the block landed in the one-iteration schedule. Absent only if
      // the block carried no work at all.
      int64_t start = 0;
      int64_t finish = 0;
      auto scheduled = estimate.iterationSchedule.find(blockId);
      if (scheduled != estimate.iterationSchedule.end()) {
        start = scheduled->second.start;
        finish = scheduled->second.finish;
      }

      blockAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("id", builder.getI64IntegerAttr(id)),
          builder.getNamedAttr("core",
                               builder.getStringAttr(describeBlockCore(stats))),
          builder.getNamedAttr(
              "cycles", builder.getI64IntegerAttr(computeBlockCycles(
                            stats, *config, fusion, /*perIteration=*/false))),
          builder.getNamedAttr(
              "iter_cycles", builder.getI64IntegerAttr(computeBlockCycles(
                                 stats, *config, fusion,
                                 /*perIteration=*/true))),
          builder.getNamedAttr("work_cycles",
                               builder.getI64IntegerAttr(stats.workCycles)),
          builder.getNamedAttr(
              "iterations", builder.getI64IntegerAttr(stats.loopMultiplier)),
          builder.getNamedAttr("iter_start", builder.getI64IntegerAttr(start)),
          builder.getNamedAttr("iter_finish",
                               builder.getI64IntegerAttr(finish)),
          builder.getNamedAttr(
              "segments",
              builder.getI64IntegerAttr(stats.countWorkingSegments())),
          builder.getNamedAttr("ops",
                               builder.getI64IntegerAttr(stats.costedOps)),
          builder.getNamedAttr(
              "bottleneck",
              builder.getStringAttr(
                  mlir::ascend::stringifyHWUnit(getBlockBottleneck(stats)))),
          builder.getNamedAttr("depends_on",
                               builder.getArrayAttr(dependsOn)),
          builder.getNamedAttr("sync_depends_on",
                               builder.getArrayAttr(syncsWith)),
      }));
    }
    module->setAttr(kCVPipelineBlocks, builder.getArrayAttr(blockAttrs));
  }

  // Which kernel this report is about, and which of its orderings. Without
  // them a log from a run over a suite of kernels is a wall of numbers with
  // nothing to attach them to, and a variant search makes that worse: the
  // report is for the ordering that won, which is not the one an unlabelled
  // reader would assume. The applied seed is read from the record
  // ReorderOpsByBlockId leaves behind, because the seed attribute itself has
  // been overwritten with -1 by the time this pass runs. Absent means no
  // variant was applied, which is the untouched pipeline, i.e. seed 0.
  const llvm::StringRef kernelName = CVPipeline::getKernelName(module);
  int64_t appliedSeed = 0;
  if (auto applied =
          module->getAttrOfType<IntegerAttr>(CVPipeline::kReorderSeedApplied)) {
    appliedSeed = applied.getInt();
  }

  auto reportTo = [&](llvm::raw_ostream &os) {
    os << "[" << DEBUG_TYPE << "] " << kernelName << " (seed " << appliedSeed
       << "): " << config->getName() << ": " << totalCycles
       << " cycles ("
       << llvm::format("%.3f", config->cyclesToMicroseconds(totalCycles))
       << " us) over " << breakdown.blocks.size() << " block(s), "
       << rooflineCycles << " cycles if fully overlapped, " << nextOpId
       << " ops costed, " << unknownOps << " of unknown cost, "
       << breakdown.genericOps << " without a dedicated model, "
       << breakdown.dynamicLoops.size() << " loop(s) of unknown trip count\n";
  };

  auto reportDetail = [&](llvm::raw_ostream &os) {
    printEstimate(os, estimate, fusion, *config, breakdown.branchedOps,
                  breakdown.weightedScalarOpsCube,
                  breakdown.weightedScalarOpsVector, pipelineStages);
    printLoops(os, loopReports);
    printBlocks(os, breakdown, estimate);
    printBreakdown(os, breakdown);
  };

  // During a variant search the estimate runs once per candidate, and the full
  // report for a few hundred of them is unreadable. The search silences every
  // attempt and prints one line itself when a candidate improves on the best so
  // far; the winner is then recompiled without the flag, so the report the user
  // ends up reading describes exactly the IR that was kept.
  if (module->hasAttr(kCVPipelineCostQuiet)) {
    return;
  }

  const int verbosity = getVerbosity();
  LLVM_DEBUG(reportTo(llvm::dbgs()); reportDetail(llvm::dbgs()));
  if (verbosity >= 1) {
    reportTo(llvm::errs());
  }
  if (verbosity >= 2) {
    reportDetail(llvm::errs());
  }
}

} // namespace

#endif // TRITON_ASCEND_HAS_INPROC_COSTMODEL

void EstimateCVPipelineCostPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The CV pipeline already gave up on this module; there is nothing whose
  // cost would be meaningful.
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
  estimateModuleCost(module, hardwareConfigPath);
#else
  LOG_DEBUG("costmodel not built into this binary; skipping cost estimation");
#endif
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createEstimateCVPipelineCostPass(std::string hardwareConfigPath) {
  return std::make_unique<EstimateCVPipelineCostPass>(
      std::move(hardwareConfigPath));
}

void registerEstimateCVPipelineCostPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createEstimateCVPipelineCostPass();
  });
}

} // namespace triton
} // namespace mlir
