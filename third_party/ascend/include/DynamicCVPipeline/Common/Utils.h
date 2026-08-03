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

#ifndef ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#define ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string_view>

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"

namespace mlir {
namespace CVPipeline {

inline constexpr llvm::StringLiteral kCoreType = "ssbuffer.core_type";
inline constexpr llvm::StringLiteral kBlockId = "ssbuffer.block_id";
inline constexpr llvm::StringLiteral kTransferId = "ssbuffer.transfer_id";
inline constexpr llvm::StringLiteral kCubeFirst = "ssbuffer.cube_first";
inline constexpr llvm::StringLiteral kVectorFirst = "ssbuffer.vector_first";
inline constexpr llvm::StringLiteral kAddFromMatmul =
    "ssbuffer.add_from_matmul";
inline constexpr llvm::StringLiteral kMainLoop = "ssbuffer.main_loop";
inline constexpr llvm::StringLiteral kTcoreType = "hivm.tcore_type";
inline constexpr llvm::StringLiteral kIf = "ssbuffer.if";
inline constexpr llvm::StringLiteral kIntraBuffer = "ssbuffer.intra_buffer";
inline constexpr llvm::StringLiteral kIntraBufCount =
    "ssbuffer.intra_buf_count";
inline constexpr llvm::StringLiteral kInterCoreBufCount =
    "ssbuffer.inter_core_buf_count";
inline constexpr llvm::StringLiteral kLoadStoreBufCount =
    "ssbuffer.load_store_buf_count";
inline constexpr llvm::StringLiteral kAnalyzeFlagId =
    "ssbuffer.analyze_flag_id";
inline constexpr llvm::StringLiteral kLoopCarriedL0C =
    "ssbuffer.loop_carried_l0c";
inline constexpr llvm::StringLiteral kCrossCoreDeps = "ssbuffer.crossCoreDeps";
inline constexpr llvm::StringLiteral kIntraDeps = "ssbuffer.intraDeps";
inline constexpr llvm::StringLiteral kMemCrossDeps = "ssbuffer.memCrossDeps";
inline constexpr llvm::StringLiteral kMayNotExec = "ssbuffer.may_not_exec";
inline constexpr llvm::StringLiteral kIterCounter = "ssbuffer.iterCounter";
inline constexpr llvm::StringLiteral kClone = "ssbuffer.clone";
inline constexpr llvm::StringLiteral kEnableUbRefineOpt =
    "ssbuffer.enable_ub_refine_opt";
inline constexpr llvm::StringLiteral kInsertionOptimization =
    "ssbuffer.insertionOptimization";
static constexpr llvm::StringLiteral kInlinableQuantScaleAttr =
    "enable_fast_tf32_mul";
inline constexpr llvm::StringLiteral kHIVMMatmulLimitedInCubeAttr =
    "hivm.matmul_limited_in_cube";
inline constexpr llvm::StringLiteral kTightlyCoupledBufferAttr =
    "hivm.tightly_coupled_buffer";
inline constexpr llvm::StringLiteral kCoreTypeCube = "CUBE";
inline constexpr llvm::StringLiteral kCoreTypeVector = "VECTOR";

inline constexpr const char *ERRCODE_ATTR =
    "triton_ascend.dynamic_cv_pipeline.rc";
static constexpr const int ERRCODE_FAILED = 1;
static constexpr const int ERRCODE_IGNORED = 2;
constexpr int64_t CACHE_TABLE_BUFFER_SIZE = 4096;
constexpr int64_t BYTE_SIZE = 8;
static constexpr int crossCoreProducerId = 1;
static constexpr int crossCoreConsumerId = 0;

enum CoreType {
  UNDETERMINED = 0,
  VECTOR_ONLY = 1 << 0,
  CUBE_ONLY = 1 << 1,
  CUBE_AND_VECTOR = VECTOR_ONLY | CUBE_ONLY,
};

inline constexpr CoreType fromStrCoreType(std::string_view s) {
  if (s == "VECTOR") {
    return CoreType::VECTOR_ONLY;
  }
  if (s == "CUBE") {
    return CoreType::CUBE_ONLY;
  }

  return CoreType::UNDETERMINED;
}

void setEnableCubeBlockMerge(bool enable);
bool isCubeBlockMergeEnabled();

void setEnableUBRefineOpt(bool enable);
bool isUBRefineOptEnabled();

// Functions for managing core types
CoreType getOpCoreType(Operation *op);
std::optional<int> getOpBlockId(Operation *op);
llvm::LogicalResult verifyOpBlockId(Operation *op);
int getAvailableBlockId(ModuleOp module);
void setFallbackAttr(ModuleOp module, int errorCode);
bool hasFallbackAttr(ModuleOp module);
bool isScfOp(Operation *op);
bool isOnlyDirectlyUse(Operation *preOp, Operation *nextOp,
                       const CVPipeline::MemoryDependenceGraph &memGraph);

// Wrapper around a "main loop" — either scf.for or scf.while carrying the
// ssbuffer.main_loop attribute. Lets downstream code treat both uniformly.
class MainLoop {
public:
  Operation *op = nullptr;
  Block *body = nullptr;
  Value iterCounter;

  Block *getBody() const;
  Operation *getOperation() const;
  MLIRContext *getContext() const;
  Location getLoc() const;
  Block *getBlock() const;
  Block::iterator getIterator() const;
  Operation *operator->() const;
  bool isWhile() const;

  // Iter args carried across loop iterations, as BlockArguments.
  // forOp:   getRegionIterArgs().
  // whileOp: after-body args.
  SmallVector<Value> getIterArgs() const;

  // Only meaningful for whileOp (before-body args); forOp returns empty.
  // Same count/types as getIterArgs() on whileOp, distinct Value identity.
  SmallVector<Value> getBeforeIterArgs() const;

  explicit MainLoop(Operation *loopOp);

  // Returns the scf.yield terminator of a forOp's body / whileOp's after
  // body. Returns {} if `loopOp` is neither.
  static scf::YieldOp getLoopYieldOp(Operation *loopOp);
};

// True when `op` is a main_loop loop op (forOp / whileOp carrying the tag).
inline bool isMainLoopOp(Operation *op) {
  return op && isa<scf::ForOp, scf::WhileOp>(op) && op->hasAttr(kMainLoop);
}

inline bool isCubeOp(Operation *op) {
  return !isScfOp(op) && CVPipeline::getOpCoreType(op) == CoreType::CUBE_ONLY;
}

// ============================================================================
// Unified Loop Helpers: abstract ForOp/WhileOp differences
// ============================================================================
// Get the body block of a loop (ForOp's body or WhileOp's after-body block)
inline Block *getLoopBodyBlock(Operation *loop) {
  if (auto forOp = dyn_cast<scf::ForOp>(loop))
    return forOp.getBody();
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop))
    return whileOp.getAfterBody()->getNextNode();
  return nullptr;
}

// Get the init values of a loop (ForOp's initArgs or WhileOp's inits)
inline ValueRange getLoopInitValues(Operation *loop) {
  if (auto forOp = dyn_cast<scf::ForOp>(loop))
    return forOp.getInitArgs();
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop))
    return whileOp.getInits();
  return {};
}

// Get the yield terminator of a loop's body
inline Operation *getLoopYieldOp(Operation *loop) {
  if (auto forOp = dyn_cast<scf::ForOp>(loop))
    return forOp.getBody()->getTerminator();
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop))
    return whileOp.getAfterBody()->getTerminator();
  return nullptr;
}

// Check if a block argument is an iter_arg of a loop (ForOp body or WhileOp
// after-body)
inline bool isLoopIterArg(BlockArgument blockArg) {
  Operation *parentOp = blockArg.getOwner()->getParentOp();
  if (isa<scf::ForOp>(parentOp))
    return true;
  if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
    return blockArg.getOwner() == whileOp.getAfterBody()->getNextNode();
  return false;
}

// Helper: Check if a value is a scalar (not a tensor type)
inline bool isScalarType(Value value) {
  return !isa<RankedTensorType>(value.getType());
}

// Helper: Check if a value is a scalar iter_arg from scf.for or scf.while
inline bool isScalarIterArgOp(Value iterArg) {
  auto blockArg = dyn_cast<BlockArgument>(iterArg);
  if (!blockArg)
    return false;
  if (!isLoopIterArg(blockArg))
    return false;
  return isScalarType(iterArg);
}

bool isVectorOnlyOp(Operation *op);

bool isScalarLike(Value value);
bool isStoreLike(Operation *op);
bool isViewLike(Operation *op);

// Returns true iff `v` is the result of a `linalg.fill` initialized with a
// 0 scalar constant. Used to detect the "add 0" operand of VECTOR pseudo-ops
// (`arith.addf` / `arith.addi` carrying `ssbuffer.add_from_matmul`).
bool isZeroFillValue(Value v);

// Read the `hivm.tightly_coupled_buffer<N>` id attached to a `memref.alloc`
// via its `annotation.mark` user. Returns nullopt when no annotation with
// a concrete id is present, or when `allocVal` is null.
std::optional<int> getTightlyCoupledBufferId(Value allocVal);

// Walk back through opaque memref casts (`memref.memory_space_cast`,
// `memref.cast`) to recover the underlying `memref.alloc` that backs a
// `bufferization.to_tensor`'s source. Returns the input unchanged when no
// such cast is found.
Value traceBackToMemrefAlloc(Value v);

} // namespace CVPipeline
} // namespace mlir

#endif
