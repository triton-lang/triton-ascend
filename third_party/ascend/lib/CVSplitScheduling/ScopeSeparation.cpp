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

#include "ascend/include/CVSplitScheduling/ScopeSeparation.h"
#include "ascend/include/CVSplitScheduling/HardwareConstants.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

static StringRef engineTypeToStr(EngineType e) {
  switch (e) {
  case EngineType::CUBE:
    return "CUBE";
  case EngineType::VECTOR:
    return "VECTOR";
  }
  return "INVALID";
}

//
// 1. Move all function body ops into a scope::ScopeOp (VECTOR)
// 2. Clone that scope to create a second one (CUBE)
// 3. Strip wrong-type ops from each scope using our classification
// 4. Neutralize yield slots for stripped ops with zero/alloc placeholders
//
// This gives bishengir two isolated scopes it can compile independently.
// ============================================================================

// FIXME: These placeholders currently mainly satisfy scf.for yield slots whose
// original producers were stripped from the CUBE clone. The corresponding
// loop results do not escape the CUBE scope and are semantically unused.
// Rebuild each per-core loop with only its required iter_args/results so these
// dummy yield values are unnecessary.
static Value buildNeutralPlaceholder(OpBuilder &builder, Type type,
                                     Location loc) {
  if (auto memrefTy = dyn_cast<MemRefType>(type)) {
    if (!memrefTy.hasStaticShape())
      return Value();
    return builder.create<memref::AllocOp>(loc, memrefTy).getResult();
  }

  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasStaticShape() && !isa<MemRefType>(type)) {
      if (Attribute elemZero = builder.getZeroAttr(shapedTy.getElementType())) {
        auto zeroDense = DenseElementsAttr::get(shapedTy, elemZero);
        return builder
            .create<arith::ConstantOp>(loc, type, cast<TypedAttr>(zeroDense))
            .getResult();
      }
    }
  }

  if (Attribute zeroAttr = builder.getZeroAttr(type)) {
    if (auto typedZero = dyn_cast<TypedAttr>(zeroAttr))
      return builder.create<arith::ConstantOp>(loc, type, typedZero)
          .getResult();
  }

  return Value();
}

static FailureOr<EngineType> getStampedEngineType(Operation *op) {
  auto coreType = op->getAttrOfType<StringAttr>("ssbuffer.core_type");
  if (!coreType) {
    op->emitError("missing core classification attribute");
    return failure();
  }

  if (coreType.getValue() == "CUBE")
    return EngineType::CUBE;
  if (coreType.getValue() == "VECTOR")
    return EngineType::VECTOR;

  op->emitError("expected a single-core classification {CUBE, VECTOR}, got ")
      << coreType.getValue();
  return failure();
}

static FailureOr<size_t> stripWrongTypeOpsFromBlock(Block &block,
                                                    EngineType keepType) {
  Operation *terminator = block.getTerminator();
  SmallVector<Operation *> toErase;

  for (Operation &op : block) {
    if (&op == terminator || isa<scf::ForOp>(&op))
      continue;

    if (isa<arith::ConstantOp>(&op)) {
      // Constants are deliberately replicated. Stamp each copy with its
      // containing scope so the intermediate classification stays truthful.
      setOpEngineTypeAttr(&op, keepType);
      continue;
    }

    // Keep scalar/index ops (address math and loop control) in both scopes.
    bool allScalar = op.getNumResults() > 0;
    for (Value result : op.getResults()) {
      if (!result.getType().isIntOrIndexOrFloat()) {
        allScalar = false;
        break;
      }
    }
    if (allScalar) {
      setOpEngineTypeAttr(&op, keepType);
      continue;
    }

    FailureOr<EngineType> opType = getStampedEngineType(&op);
    if (failed(opType))
      return failure();
    if (*opType != keepType)
      toErase.push_back(&op);
  }

  size_t erasedCount = toErase.size();
  for (Operation *op : llvm::reverse(toErase)) {
    OpBuilder builder(op);
    for (Value result : op->getResults()) {
      if (result.use_empty())
        continue;

      Value placeholder =
          buildNeutralPlaceholder(builder, result.getType(), op->getLoc());
      if (!placeholder) {
        op->emitError("cannot build a neutral placeholder for live result ")
            << result.getType();
        return failure();
      }
      result.replaceAllUsesWith(placeholder);
    }
    op->erase();
  }

  return erasedCount;
}

static LogicalResult stripWrongTypeOps(scope::ScopeOp scopeOp,
                                       EngineType keepType) {
  LLVM_DEBUG(llvm::dbgs() << "[cv-split] Stripping "
                          << (keepType == EngineType::CUBE ? "VECTOR" : "CUBE")
                          << " ops from " << engineTypeToStr(keepType)
                          << " scope\n");

  Block &scopeBlock = scopeOp.getBodyRegion().front();

  FailureOr<size_t> topLevelErased =
      stripWrongTypeOpsFromBlock(scopeBlock, keepType);
  if (failed(topLevelErased))
    return failure();

  size_t loopErased = 0;
  WalkResult walkResult = scopeOp.walk([&](scf::ForOp forOp) {
    FailureOr<size_t> erased =
        stripWrongTypeOpsFromBlock(*forOp.getBody(), keepType);
    if (failed(erased))
      return WalkResult::interrupt();
    loopErased += *erased;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Erased " << *topLevelErased
                          << " top-level + " << loopErased << " loop ops from "
                          << engineTypeToStr(keepType) << " scope\n");

  // Final cleanup: remove dead allocs whose only users are annotation.mark
  SmallVector<Operation *> deadMarks;
  SmallVector<memref::AllocOp> deadAllocs;
  scopeOp.walk([&](memref::AllocOp allocOp) {
    Value result = allocOp.getResult();
    bool allUsersAreAnnotations = true;
    SmallVector<Operation *> markUsers;
    for (Operation *user : result.getUsers()) {
      if (isa<annotation::MarkOp>(user)) {
        markUsers.push_back(user);
      } else {
        allUsersAreAnnotations = false;
        break;
      }
    }
    if (allUsersAreAnnotations) {
      deadMarks.append(markUsers);
      deadAllocs.push_back(allocOp);
    }
  });
  for (Operation *markOp : llvm::reverse(deadMarks))
    markOp->erase();
  for (memref::AllocOp allocOp : llvm::reverse(deadAllocs)) {
    assert(allocOp->use_empty() && "dead allocation still has users");
    allocOp.erase();
  }
  if (!deadMarks.empty() || !deadAllocs.empty())
    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Cleaned up "
                            << deadMarks.size() + deadAllocs.size()
                            << " dead alloc/mark ops\n");
  return success();
}

// Rebuild an scf.for without loop-carried values that are unused both inside
// the body and outside the loop. The init operand, region iter argument, loop
// result, and yield operand at each removed index are dropped together.
static scf::ForOp removeUnusedLoopCarriedValues(scf::ForOp loop) {
  SmallVector<unsigned> keptIndices;
  unsigned numIterArgs = loop.getNumRegionIterArgs();
  keptIndices.reserve(numIterArgs);
  for (unsigned i = 0; i < numIterArgs; ++i) {
    if (!loop.getRegionIterArgs()[i].use_empty() ||
        !loop.getResult(i).use_empty())
      keptIndices.push_back(i);
  }

  if (keptIndices.size() == numIterArgs)
    return loop;

  auto oldYield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<Value> newInitArgs;
  SmallVector<Value> newYieldValues;
  newInitArgs.reserve(keptIndices.size());
  newYieldValues.reserve(keptIndices.size());
  for (unsigned oldIndex : keptIndices) {
    newInitArgs.push_back(loop.getInitArgs()[oldIndex]);
    newYieldValues.push_back(oldYield.getOperand(oldIndex));
  }

  OpBuilder builder(loop);
  auto newLoop = builder.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                                            loop.getUpperBound(),
                                            loop.getStep(), newInitArgs);
  newLoop->setAttrs(loop->getAttrs());

  Block *oldBody = loop.getBody();
  Block *newBody = newLoop.getBody();
  loop.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());
  for (auto [newIndex, oldIndex] : llvm::enumerate(keptIndices))
    loop.getRegionIterArgs()[oldIndex].replaceAllUsesWith(
        newLoop.getRegionIterArgs()[newIndex]);

  // Depending on the builder overload, the fresh body is either empty or has
  // an empty scf.yield. Remove that placeholder and build the real yield after
  // moving the original body operations.
  if (!newBody->empty()) {
    assert(isa<scf::YieldOp>(newBody->back()) &&
           "fresh scf.for body must contain only its yield");
    newBody->back().erase();
  }
  while (&oldBody->front() != oldYield.getOperation())
    oldBody->front().moveBefore(newBody, newBody->end());
  OpBuilder yieldBuilder = OpBuilder::atBlockEnd(newBody);
  yieldBuilder.create<scf::YieldOp>(oldYield.getLoc(), newYieldValues);

  for (auto [newIndex, oldIndex] : llvm::enumerate(keptIndices))
    loop.getResult(oldIndex).replaceAllUsesWith(newLoop.getResult(newIndex));

  unsigned removedCount = numIterArgs - keptIndices.size();
  loop.erase();
  LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Removed " << removedCount
                          << " unused loop-carried value(s)\n");
  return newLoop;
}

static LogicalResult
retileVectorScopeForRowSplit(
    scope::ScopeOp vecScope, scf::ForOp cubeLoop, scf::ForOp vectorLoop,
    const CrossScopeTransferInfo &transferInfo,
    bool materializePostSplitSchedule,
    const PostCVSplitDetachedSchedule *detachedSchedule);
static LogicalResult sinkCubeLoadChainsToMatmul(
    Block *body, unsigned productOperandPrefetchDepth);

// Sink each cube matmul's operand load chain to immediately before the matmul.
// See the call site (createScopeSeparation step 6b) for the rationale.
//
// For each linalg.matmul in the cube loop body (program order), gather the
// backward "load chain" of ops feeding its tensor operands -- restricted to the
// same block and to load-chain op types (reinterpret_cast / alloc / memref.copy
// / to_tensor / transpose). An op is moved only if it belongs to exactly one
// matmul chain and every use of its results is internal to that chain or its
// matmul. This leaves shared operands (the loop-invariant Q cbuf, the P pack
// consumed by all PV matmuls, accumulator fill tensors) in place while moving
// private portions of each chain. Movable ops retain their existing relative
// order and are placed immediately before their matmul.
static LogicalResult
sinkCubeLoadChainsToMatmul(Block *body,
                           unsigned productOperandPrefetchDepth) {
  auto isChainType = [](Operation *o) {
    return isa<memref::ReinterpretCastOp, memref::AllocOp, memref::CopyOp,
               bufferization::ToTensorOp, linalg::TransposeOp>(o);
  };

  struct MatmulChain {
    explicit MatmulChain(Operation *matmul) : matmul(matmul) {}
    Operation *matmul;
    SetVector<Operation *> ops;
  };

  SmallVector<MatmulChain> chains;
  DenseMap<Operation *, unsigned> chainUseCount;

  for (Operation &op : *body)
    if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(op))
      chains.emplace_back(&op);

  for (MatmulChain &matmulChain : chains) {
    // 1. Build the candidate chain via a worklist over operands. memref.copy
    //    writes its dst alloc by side effect (no SSA result), so when we reach
    //    an alloc we also pull in the copy that writes it and that copy's
    //    source (reinterpret_cast).
    Operation *mm = matmulChain.matmul;
    SetVector<Operation *> &chain = matmulChain.ops;
    SmallVector<Value> worklist(mm->operand_begin(), mm->operand_end());

    auto enqueueOperands = [&](Operation *op) {
      for (Value v : op->getOperands())
        worklist.push_back(v);
    };

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != body || !isChainType(def))
        continue;
      if (!chain.insert(def))
        continue;
      ++chainUseCount[def];
      enqueueOperands(def);
      // For an alloc, find the memref.copy in this block that writes it.
      if (isa<memref::AllocOp>(def)) {
        for (Operation *user : def->getResult(0).getUsers()) {
          auto copy = dyn_cast<memref::CopyOp>(user);
          if (copy && copy->getBlock() == body &&
              copy.getTarget() == def->getResult(0)) {
            if (chain.insert(copy)) {
              ++chainUseCount[copy.getOperation()];
              enqueueOperands(copy);
            }
          }
        }
      }
    }
  }

  // 2. Select private operations independently within each chain. An operation
  //    shared by multiple matmul chains stays in place. For result-producing
  //    operations, all users must remain inside this chain or be its matmul.
  //    memref.copy has no result, so its destination allocation must also be
  //    private to this chain.
  auto safeToMove = [&](Operation *op, Operation *mm,
                        const SetVector<Operation *> &chain) {
    if (chainUseCount.lookup(op) != 1)
      return false;

    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      Operation *target = copy.getTarget().getDefiningOp();
      return target && chain.contains(target) &&
             chainUseCount.lookup(target) == 1;
    }

    for (Operation *user : op->getUsers())
      if (user != mm && !chain.contains(user))
        return false;
    return true;
  };

  for (MatmulChain &matmulChain : chains) {
    Operation *mm = matmulChain.matmul;
    SetVector<Operation *> &chain = matmulChain.ops;

    SmallVector<Operation *> movable;

    for (Operation *op : chain)
      if (safeToMove(op, mm, chain))
        movable.push_back(op);

    // 3. Move in existing program order so relative topological order holds.
    llvm::sort(movable, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    for (Operation *op : movable)
      op->moveBefore(mm);
  }

  if (productOperandPrefetchDepth == 0)
    return failure();
  if (productOperandPrefetchDepth == 1)
    return success();

  MLIRContext *context = body->getParentOp()->getContext();
  auto pipeMte3 = hivm::PipeAttr::get(context, hivm::PIPE::PIPE_MTE3);
  auto pipeMte1 = hivm::PipeAttr::get(context, hivm::PIPE::PIPE_MTE1);
  SmallVector<Operation *> probabilityWaits;
  SmallVector<Operation *> productMatmuls;
  for (Operation &operation : *body) {
    auto wait = dyn_cast<hivm::SyncBlockWaitOp>(&operation);
    if (!wait || wait.getTpipe() != pipeMte3 || wait.getPipe() != pipeMte1)
      continue;
    Operation *matmul = nullptr;
    for (Operation *cursor = operation.getNextNode(); cursor;
         cursor = cursor->getNextNode()) {
      if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(cursor)) {
        matmul = cursor;
        break;
      }
      if (auto laterWait = dyn_cast<hivm::SyncBlockWaitOp>(cursor);
          laterWait && laterWait.getTpipe() == pipeMte3 &&
          laterWait.getPipe() == pipeMte1)
        return failure();
      if (cursor->hasTrait<OpTrait::IsTerminator>())
        break;
    }
    if (!matmul)
      return failure();
    probabilityWaits.push_back(wait);
    productMatmuls.push_back(matmul);
  }
  if (productMatmuls.empty() ||
      productOperandPrefetchDepth > productMatmuls.size())
    return failure();

  Operation *lastScoreMatmul = nullptr;
  for (MatmulChain &chain : chains)
    if (chain.matmul->isBeforeInBlock(probabilityWaits.front()))
      lastScoreMatmul = chain.matmul;
  if (!lastScoreMatmul || !lastScoreMatmul->getNextNode())
    return failure();

  auto findMatmulChain = [&](Operation *matmul) -> MatmulChain * {
    for (MatmulChain &chain : chains)
      if (chain.matmul == matmul)
        return &chain;
    return nullptr;
  };
  auto collectOperandChain = [&](Value root) {
    SetVector<Operation *> chain;
    SmallVector<Value> worklist{root};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getBlock() != body ||
          !isChainType(definition) || !chain.insert(definition))
        continue;
      for (Value operand : definition->getOperands())
        worklist.push_back(operand);
      if (isa<memref::AllocOp>(definition))
        for (Operation *user : definition->getResult(0).getUsers())
          if (auto copy = dyn_cast<memref::CopyOp>(user);
              copy && copy->getBlock() == body &&
              copy.getTarget() == definition->getResult(0) &&
              chain.insert(copy))
            for (Value operand : copy->getOperands())
              worklist.push_back(operand);
    }
    return chain;
  };

  SmallVector<SmallVector<Operation *>> productOperandLoadChains;
  productOperandLoadChains.reserve(productMatmuls.size());
  for (Operation *matmul : productMatmuls) {
    MatmulChain *fullChain = findMatmulChain(matmul);
    if (!fullChain)
      return failure();
    SmallVector<Operation *> selected;
    unsigned copiedOperandChains = 0;
    for (Value operand : matmul->getOperands()) {
      SetVector<Operation *> operandChain = collectOperandChain(operand);
      unsigned copyCount = llvm::count_if(operandChain, [](Operation *op) {
        return isa<memref::CopyOp>(op);
      });
      if (copyCount == 0)
        continue;
      if (copyCount != 1 || copiedOperandChains++ != 0)
        return failure();
      for (Operation *operation : operandChain) {
        if (!safeToMove(operation, matmul, fullChain->ops))
          return failure();
        selected.push_back(operation);
      }
    }
    if (copiedOperandChains != 1 || selected.empty())
      return failure();
    llvm::sort(selected, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    productOperandLoadChains.push_back(std::move(selected));
  }

  const unsigned prefetchDistance = productOperandPrefetchDepth - 1;
  for (auto [lane, loadChain] :
       llvm::enumerate(productOperandLoadChains)) {
    Operation *anchor =
        lane < prefetchDistance
            ? lastScoreMatmul->getNextNode()
            : productMatmuls[lane - prefetchDistance];
    DenseSet<Operation *> moving;
    moving.insert(loadChain.begin(), loadChain.end());
    for (Operation *operation : loadChain)
      for (Value operand : operation->getOperands())
        if (Operation *definition = operand.getDefiningOp();
            definition && definition->getBlock() == body &&
            !moving.contains(definition) &&
            (definition == anchor || anchor->isBeforeInBlock(definition)))
          return failure();
    for (Operation *operation : loadChain)
      operation->moveBefore(anchor);
  }

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] product-operand-prefetch depth="
             << productOperandPrefetchDepth
             << " distance=" << prefetchDistance
             << " lanes=" << productOperandLoadChains.size() << "\n");
  return success();
}

// ============================================================================
// ROW_SPLIT vector re-tile.
// The fixpipe ROW_SPLIT delivers 16 rows (M/2) to each veccore's UB. The VECTOR
// scope was cloned at the original 32-row tile size, so we re-tile its whole
// softmax DAG to 16 rows per veccore:
//   - leading (M) dimension 32 -> 16 on all vector tensors / UB memrefs
//   - external vector-only init constants (fills/empties) halved
//   - V->C P pack rebuilt as 16x32 -> 2x1x16x16, copied into the veccore's
//     subview [0, sub_block_idx, 0, 0] of the shared 2x2x16x16 L1 buffer
//   - output store offset += sub_block_idx * 16 * leadingStride, sizes M=16
// Both veccores then do useful work (2x vector throughput), matching the
// target.
// ============================================================================
// Change the leading dimension of a [BLOCK_M, ...] tensor or memref to
// [BLOCK_M/2, ...], giving each vector core its ROW_SPLIT row band. Types whose
// leading dimension is not BLOCK_M pass through unchanged.
static Type retileRowHalve(Type t, int64_t Mfull) {
  int64_t half = Mfull / 2;
  if (auto rt = dyn_cast<RankedTensorType>(t)) {
    auto sh = rt.getShape();
    if (!sh.empty() && sh[0] == Mfull) {
      SmallVector<int64_t> ns(sh.begin(), sh.end());
      ns[0] = half;
      return RankedTensorType::get(ns, rt.getElementType(), rt.getEncoding());
    }
  } else if (auto mt = dyn_cast<MemRefType>(t)) {
    auto sh = mt.getShape();
    if (!sh.empty() && sh[0] == Mfull) {
      // Retiling preserves explicit strides. Reject arbitrary affine
      // layouts here: changing their leading extent can change address
      // semantics in a way this local transformation cannot prove.
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(mt.getStridesAndOffset(strides, offset)))
        return t;
      SmallVector<int64_t> ns(sh.begin(), sh.end());
      ns[0] = half;
      return MemRefType::get(ns, mt.getElementType(), mt.getLayout(),
                             mt.getMemorySpace());
    }
  }
  return t;
}

// V->C pack chain detached in step 2 and rebuilt per-veccore in step 6: the
// softmax P tensor (pSrc), the shared L1 destination (l1Alloc), and the op the
// rebuilt pack is inserted before (anchor).
struct VectorToCubePack {
  Value pSrc;
  Value l1Alloc;
  Operation *anchor;
  Operation *scoreReleaseAnchor = nullptr;
};

struct ReleaseLanePlan {
  unsigned lane = 0;
  unsigned scoreReleaseFlag = 0;
  unsigned productReleaseFlag = 0;
  PrincipalResource scoreSignalingResource =
      PrincipalResource::ScalarControl;
  PrincipalResource scoreWaitingResource = PrincipalResource::ScalarControl;
  PrincipalResource productSignalingResource =
      PrincipalResource::ScalarControl;
  PrincipalResource productWaitingResource =
      PrincipalResource::ScalarControl;
  Operation *scoreCubeDrain = nullptr;
  Operation *scoreVectorRelease = nullptr;
  Operation *productCubeDrain = nullptr;
  Operation *productVectorRelease = nullptr;
};

struct InitialReleaseSignal {
  PostCVSplitLineageRole role = PostCVSplitLineageRole::Score;
  unsigned slot = 0;
  unsigned flag = 0;
  PrincipalResource signalingResource = PrincipalResource::ScalarControl;
  PrincipalResource waitingResource = PrincipalResource::ScalarControl;
};

struct ReleaseProtocolPlan {
  SmallVector<ReleaseLanePlan> lanes;
  SmallVector<InitialReleaseSignal> initialSignals;
  Operation *legacyCubeWait = nullptr;
  Operation *legacyVectorSet = nullptr;
};

struct OwnershipLanePlan {
  PostCVSplitLineageRole role = PostCVSplitLineageRole::Score;
  unsigned lane = 0;
  unsigned slot = 0;
  Value oldBuffer;
  Operation *cubeDrain = nullptr;
  Operation *vectorCast = nullptr;
};

struct BufferOwnershipPlan {
  SmallVector<OwnershipLanePlan> lanes;
  SmallVector<Value> oldBuffers;
  MemRefType scoreBufferType;
  MemRefType productBufferType;
  unsigned scoreSlotCount = 0;
  unsigned productSlotCount = 0;
};

static std::optional<unsigned> getStaticEventFlag(Operation *operation) {
  std::optional<IntegerAttr> flag;
  if (auto set = dyn_cast<hivm::SyncBlockSetOp>(operation))
    flag = set.getStaticFlagId();
  else if (auto wait = dyn_cast<hivm::SyncBlockWaitOp>(operation))
    flag = wait.getStaticFlagId();
  if (!flag || (*flag).getInt() < 0)
    return std::nullopt;
  return static_cast<unsigned>((*flag).getInt());
}

static const PostCVSplitDetachedCommand *findDetachedCommand(
    ArrayRef<PostCVSplitDetachedCommand> commands,
    PostCVSplitDetachedCommandKind kind, unsigned lane) {
  auto command = llvm::find_if(commands, [&](const auto &candidate) {
    return candidate.kind == kind && candidate.lane == lane;
  });
  return command == commands.end() ? nullptr : &*command;
}

static const PostCVSplitDetachedCommand *findDetachedCommandByFlag(
    ArrayRef<PostCVSplitDetachedCommand> commands,
    PostCVSplitDetachedCommandKind kind, unsigned flag) {
  auto command = llvm::find_if(commands, [&](const auto &candidate) {
    return candidate.kind == kind && candidate.hasEvent &&
           candidate.logicalFlagId == flag;
  });
  return command == commands.end() ? nullptr : &*command;
}

static hivm::PipeAttr pipeForResource(MLIRContext *context,
                                             PrincipalResource resource) {
  switch (resource) {
  case PrincipalResource::Fixpipe:
    return hivm::PipeAttr::get(context, hivm::PIPE::PIPE_FIX);
  case PrincipalResource::Vector:
    return hivm::PipeAttr::get(context, hivm::PIPE::PIPE_V);
  case PrincipalResource::Mte3:
    return hivm::PipeAttr::get(context, hivm::PIPE::PIPE_MTE3);
  default:
    return nullptr;
  }
}

// Step 1: emit get_sub_block_idx at the top of the scope and return it as an
// index value (0 or 1 — which of the core's two veccores is executing).
static Value emitSubBlockIndex(scope::ScopeOp vecScope, Location loc) {
  Block &block = vecScope.getBodyRegion().front();
  OpBuilder topB(&block, block.begin());
  auto sbid = topB.create<hivm::GetSubBlockIdxOp>(loc, topB.getI64Type());
  return topB
      .create<arith::IndexCastOp>(loc, topB.getIndexType(), sbid.getResult())
      .getResult();
}

// Step 2: detach the whole-tile V->C pack chain
// (truncf -> reshape -> transpose -> reshape -> to_memref -> cast -> copy),
// recording each P source / L1 alloc / insertion anchor so step 6 can rebuild a
// per-veccore pack. The truncf (the actual P value) is kept; the rest is
// erased.
static SmallVector<VectorToCubePack> detachVectorToCubePacks(
    scope::ScopeOp vecScope,
    ArrayRef<VectorToCubeTransferChain> vectorToCubeChains) {
  SmallVector<VectorToCubePack> packs;
  SmallVector<Operation *> toErase;
  for (const VectorToCubeTransferChain &chain : vectorToCubeChains) {
    packs.push_back({chain.pSrc, chain.l1Alloc, chain.anchor});
    llvm::append_range(toErase, chain.operationsToErase);
  }
  // The handles were recorded in creation order; erase users before their
  // defining operations.
  for (Operation *o : llvm::reverse(toErase))
    o->erase();
  return packs;
}

// Step 3: clone external BLOCK_M-row init fills/empties with BLOCK_M/2 rows and
// rewrite only the VECTOR-scope uses. These inits may still be shared with the
// CUBE scope, so they must not be retiled in place. Returns the clone count.
static unsigned cloneExternalInitsAsHalfHeight(scope::ScopeOp vecScope,
                                               Location loc, int64_t Mfull) {
  // Clone each full-height initializer instead of changing the original:
  // CUBE retains [BLOCK_M, ...], while VECTOR uses [BLOCK_M/2, ...].
  DenseSet<Operation *> vecOps;
  vecScope.walk([&](Operation *o) { vecOps.insert(o); });
  OpBuilder cb(vecScope);
  DenseMap<Value, Value> cloneMap;
  DenseSet<Operation *> originalFills;
  DenseSet<Operation *> originalEmpties;
  unsigned clonedCount = 0;
  vecScope.walk([&](Operation *op) {
    for (OpOperand &opd : op->getOpOperands()) {
      Value v = opd.get();
      Operation *d = v.getDefiningOp();
      if (!d || vecOps.count(d))
        continue; // external defs only
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      if (!rt || rt.getRank() == 0 || rt.getShape()[0] != Mfull)
        continue;
      auto it = cloneMap.find(v);
      Value repl;
      if (it != cloneMap.end()) {
        repl = it->second;
      } else {
        Type retiledType = retileRowHalve(rt, Mfull);
        auto ntt = cast<RankedTensorType>(retiledType);
        SmallVector<int64_t> expectedShape(rt.getShape());
        expectedShape[0] = Mfull / 2;
        auto expectedType = RankedTensorType::get(
            expectedShape, rt.getElementType(), rt.getEncoding());
        assert(ntt == expectedType &&
               "retile must only halve the BLOCK_M dimension");
        if (auto fill = dyn_cast<linalg::FillOp>(d)) {
          originalFills.insert(d);
          Value init = fill.getDpsInitOperand(0)->get();
          if (auto empty = init.getDefiningOp<tensor::EmptyOp>())
            originalEmpties.insert(empty);
          Value ne = cb.create<tensor::EmptyOp>(
              loc, ntt.getShape(), ntt.getElementType(), ntt.getEncoding());
          repl =
              cb.create<linalg::FillOp>(loc, fill.getInputs(), ValueRange{ne})
                  .getResult(0);
        } else if (isa<tensor::EmptyOp>(d)) {
          originalEmpties.insert(d);
          repl = cb.create<tensor::EmptyOp>(
              loc, ntt.getShape(), ntt.getElementType(), ntt.getEncoding());
        } else {
          continue;
        }
        cloneMap[v] = repl;
        ++clonedCount;
      }
      opd.set(repl);
    }
  });

  // Erase fills before their backing empties so each empty can become dead
  // before it is checked.
  unsigned removedCount = 0;
  for (Operation *fill : originalFills)
    if (isOpTriviallyDead(fill)) {
      fill->erase();
      ++removedCount;
    }
  for (Operation *empty : originalEmpties)
    if (isOpTriviallyDead(empty)) {
      empty->erase();
      ++removedCount;
    }
  LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Removed " << removedCount
                          << " dead original initializer(s)\n");

  return clonedCount;
}

// Steps 4 & 4.5: retile every in-scope op result (and loop-carried arg) from
// BLOCK_M to BLOCK_M/2 rows, then repair any DPS tensor.empty init whose type
// drifted from its retiled result. (arith.constant splats get their value attr
// rebuilt; reinterpret_cast is handled in steps 5/6 and skipped here.)
static LogicalResult retileVectorScopeOps(scope::ScopeOp vecScope,
                                          int64_t Mfull) {
  // ---- 4. Generic re-tile of every vector op (skip reinterpret_cast) ----
  SmallVector<arith::ConstantOp> constants;
  vecScope.walk(
      [&](arith::ConstantOp constant) { constants.push_back(constant); });

  for (arith::ConstantOp c : constants) {
    Type nt = retileRowHalve(c.getType(), Mfull);
    if (nt == c.getType())
      continue;

    auto dense = dyn_cast<DenseElementsAttr>(c.getValue());
    if (!dense || !dense.isSplat()) {
      c.emitError("VECTOR retiling only supports shaped splat constants");
      return failure();
    }

    auto nst = cast<ShapedType>(nt);
    c.setValueAttr(
        DenseElementsAttr::get(nst, dense.getSplatValue<Attribute>()));
    c.getResult().setType(nt);
  }

  vecScope.walk([&](Operation *op) {
    // Its explicit offset/size/stride metadata must be rebuilt together with
    // its type. Output casts are handled by retileOutputStores(); V->C packing
    // views are reconstructed by rebuildVectorToCubePacks().
    // Constants are also skipped because they were re-tiled above together
    // with their value attributes.
    if (isa<memref::ReinterpretCastOp, arith::ConstantOp>(op))
      return;
    for (Value r : op->getResults())
      r.setType(retileRowHalve(r.getType(), Mfull));
    if (auto f = dyn_cast<scf::ForOp>(op)) {
      Block *b = f.getBody();
      for (unsigned i = 1; i < b->getNumArguments(); ++i)
        b->getArgument(i).setType(
            retileRowHalve(b->getArgument(i).getType(), Mfull));
    }
  });

  // Validate that the preceding retiling updated every tensor DPS init and its
  // tied result consistently. Do not silently repair a missed transformation.
  SmallVector<DestinationStyleOpInterface> dpsOps;
  vecScope.walk(
      [&](DestinationStyleOpInterface dps) { dpsOps.push_back(dps); });
  for (DestinationStyleOpInterface dps : dpsOps) {
    for (OpOperand &initOperand : dps.getDpsInitsMutable()) {
      Value init = initOperand.get();
      if (!isa<TensorType>(init.getType()))
        continue;
      OpResult result = dps.getTiedOpResult(&initOperand);
      if (init.getType() != result.getType()) {
        dps->emitError("VECTOR retiling produced mismatched DPS init/result "
                       "types");
        return failure();
      }
    }
  }

  // Generic type retiling must keep each scf.for result, init operand, and
  // region iter_arg identical. A missed external initializer is a candidate
  // rejection, never permission to leave temporarily invalid IR for a later
  // verifier to diagnose without context.
  WalkResult loopTypes = vecScope.walk([&](scf::ForOp loop) {
    for (auto [init, iterArg, result] : llvm::zip_equal(
             loop.getInitArgs(), loop.getRegionIterArgs(), loop.getResults())) {
      if (init.getType() == iterArg.getType() &&
          init.getType() == result.getType())
        continue;
      loop.emitError("VECTOR retiling produced mismatched scf.for "
                     "init/iter_arg/result types");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (loopTypes.wasInterrupted())
    return failure();
  return success();
}

// Step 5: shift each output store to this veccore's BLOCK_M/2-row band —
// offset += sub_block_idx * (BLOCK_M/2) * leadingStride, size M = BLOCK_M/2.
// Returns the store count.
static FailureOr<unsigned> retileOutputStores(scope::ScopeOp vecScope,
                                              Value sbidx, Location loc,
                                              int64_t Mfull) {
  int64_t half = Mfull / 2;
  // ---- 5. Output stores: offset += sbid*half*leadingStride, sizes M=half ----
  SmallVector<bufferization::MaterializeInDestinationOp> mats;
  vecScope.walk(
      [&](bufferization::MaterializeInDestinationOp m) { mats.push_back(m); });
  unsigned retiledCount = 0;
  for (auto m : mats) {
    auto ric = m.getDest().getDefiningOp<memref::ReinterpretCastOp>();
    if (!ric) {
      m.emitError("VECTOR output retiling requires a destination defined "
                  "by memref.reinterpret_cast");
      return failure();
    }
    // Build the per-veccore reinterpret_cast right before the materialize (it
    // is inside the scope, so sub_block_idx dominates it). The original ric may
    // be a loop-invariant op hoisted into the parent block, where sbid is not
    // in scope.
    OpBuilder b(m);
    auto offsets = ric.getMixedOffsets();
    auto sizes = ric.getMixedSizes();
    auto strides = ric.getMixedStrides();
    if (offsets.empty() || sizes.empty() || strides.empty()) {
      ric.emitError("expected offset, size, and stride metadata for dimension "
                    "0");
      return failure();
    }
    Value leadingStride = getValueOrCreateConstantIndexOp(b, loc, strides[0]);
    Value origOff = getValueOrCreateConstantIndexOp(b, loc, offsets[0]);
    Value halfValue = b.create<arith::ConstantIndexOp>(loc, half);
    Value step = b.create<arith::MulIOp>(loc, halfValue, leadingStride);
    Value add = b.create<arith::MulIOp>(loc, sbidx, step);
    Value newOff = b.create<arith::AddIOp>(loc, origOff, add);
    sizes[0] = b.getIndexAttr(half);
    MemRefType oldType = ric.getType();
    if (oldType.getRank() < 1 || oldType.getDimSize(0) != Mfull) {
      ric.emitError("expected output leading dimension to equal BLOCK_M");
      return failure();
    }
    auto newType = cast<MemRefType>(retileRowHalve(oldType, Mfull));
    if (newType == oldType || newType.getDimSize(0) != half) {
      ric.emitError("failed to retile output from BLOCK_M to BLOCK_M/2");
      return failure();
    }
    auto newRic = b.create<memref::ReinterpretCastOp>(
        loc, newType, ric.getSource(), getAsOpFoldResult(newOff), sizes,
        strides);
    m.getDestMutable().set(newRic.getResult());
    if (ric->use_empty())
      ric.erase();
    ++retiledCount;
  }
  return retiledCount;
}

// Step 6: rebuild each detached V->C pack as a per-veccore pack: 16x32 ->
// 2x1x16x16, copied into this veccore's subview [0, sub_block_idx, 0, 0] of the
// shared L1 buffer.
static LogicalResult
rebuildVectorToCubePacks(ArrayRef<VectorToCubePack> packs, Value sbidx,
                         hivm::AddressSpaceAttr ubAddrSpace, Location loc) {
  // ---- 6. Regenerate V->C packs: 16x32 -> 2x1x16x16 -> subview[0,sbid,0,0]
  // ----
  for (auto &p : packs) {
    if (!p.pSrc) {
      emitError(loc, "V->C pack must have a source tensor");
      return failure();
    }
    if (!p.anchor || !isa<hivm::SyncBlockSetOp>(p.anchor)) {
      emitError(loc, "V->C pack must have a sync_block_set anchor");
      return failure();
    }
    auto pType = dyn_cast<RankedTensorType>(p.pSrc.getType());
    if (!pType || !pType.hasStaticShape() ||
        (pType.getRank() != 2 && pType.getRank() != 3)) {
      emitError(loc, "V->C pack source must be a static rank-2 tensor or "
                     "direct [N/16,M,16] tensor");
      return failure();
    }
    bool directPacked = pType.getRank() == 3;
    int64_t M = directPacked ? pType.getShape()[1] : pType.getShape()[0];
    int64_t N16 =
        directPacked ? pType.getShape()[0] : pType.getShape()[1] / kNzTileSize;
    int64_t N = N16 * kNzTileSize;
    int64_t M16 = M / kNzTileSize;
    if (M <= 0 || N <= 0 || M % kNzTileSize != 0 ||
        (directPacked && pType.getShape()[2] != kNzTileSize)) {
      emitError(loc, "invalid direct NZ pack geometry");
      return failure();
    }
    Type elemType = pType.getElementType();
    Operation *pProducer = p.pSrc.getDefiningOp();
    if (!pProducer || pProducer->getBlock() != p.anchor->getBlock() ||
        !pProducer->isBeforeInBlock(p.anchor)) {
      emitError(loc, "V->C pack source must be defined before its sync anchor "
                     "in the vector scope");
      return failure();
    }

    // Begin the layout-only pack immediately after P is formed. This lets
    // the transpose overlap the independent denominator/alpha update in
    // the same VF, matching the hand-unrolled schedule. The final reshape,
    // UB view, copy, and signal remain at the phase-end anchor below.
    OpBuilder b(pProducer);
    b.setInsertionPointAfter(pProducer);
    auto l1AddrSpace =
        b.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
    auto expectedL1Type =
        MemRefType::get({N16, 2 * M16, kNzTileSize, kNzTileSize}, elemType,
                        nullptr, l1AddrSpace);
    if (!p.l1Alloc || p.l1Alloc.getType() != expectedL1Type) {
      emitError(loc) << "V->C pack L1 destination must have type "
                     << expectedL1Type;
      return failure();
    }
    auto i64Ty = b.getI64Type();
    Value packedTensor = p.pSrc;
    if (!directPacked) {
      // Generic fallback: reshape [M,N] and transpose to [N16,M,16].
      auto s3Type = RankedTensorType::get({3}, i64Ty);
      auto s3 = b.create<arith::ConstantOp>(
          loc, s3Type,
          DenseElementsAttr::get(
              s3Type, ArrayRef<int64_t>{M, N16, kNzTileSize}));
      auto resh1Type = RankedTensorType::get({M, N16, kNzTileSize}, elemType);
      auto resh1 =
          b.create<tensor::ReshapeOp>(loc, resh1Type, p.pSrc, s3.getResult());
      auto emptyT = b.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{N16, M, kNzTileSize}, elemType);
      auto transpose = b.create<linalg::TransposeOp>(
          loc, resh1.getResult(), emptyT.getResult(),
          ArrayRef<int64_t>{1, 0, 2});
      packedTensor = transpose->getResult(0);
    }

    b.setInsertionPoint(p.anchor);
    // reshape [N16,M,kNzTileSize] -> [N16,M16,kNzTileSize,kNzTileSize]
    auto s4Type = RankedTensorType::get({4}, i64Ty);
    auto s4 = b.create<arith::ConstantOp>(
        loc, s4Type,
        DenseElementsAttr::get(
            s4Type, ArrayRef<int64_t>{N16, M16, kNzTileSize, kNzTileSize}));
    auto nzType =
        RankedTensorType::get({N16, M16, kNzTileSize, kNzTileSize}, elemType);
    auto resh2 = b.create<tensor::ReshapeOp>(loc, nzType, packedTensor,
                                             s4.getResult());
    // to_memref + cast to UB
    auto memT = MemRefType::get({N16, M16, kNzTileSize, kNzTileSize}, elemType);
    auto toMem =
        b.create<bufferization::ToBufferOp>(loc, memT, resh2.getResult());
    auto ubMemT = MemRefType::get({N16, M16, kNzTileSize, kNzTileSize},
                                  elemType, nullptr, ubAddrSpace);
    auto cast =
        b.create<memref::MemorySpaceCastOp>(loc, ubMemT, toMem.getResult());
    // subview of L1 alloc [0, sbid*M16, 0, 0] [N16,M16,16,16]: each veccore
    // owns M16 = (rows/veccore)/16 fractal-row blocks, so veccore `sbid` writes
    // the band starting at block sbid*M16 (NOT bare sbid — that only coincided
    // for BLOCK_M=32 where M16==1, and overlapped/misaligned for BLOCK_M>=64).
    Value m16c = b.create<arith::ConstantIndexOp>(loc, M16);
    Value off1 = b.create<arith::MulIOp>(loc, sbidx, m16c);
    SmallVector<OpFoldResult, 4> offs{b.getIndexAttr(0), off1,
                                      b.getIndexAttr(0), b.getIndexAttr(0)};
    SmallVector<OpFoldResult, 4> szs{b.getIndexAttr(N16), b.getIndexAttr(M16),
                                     b.getIndexAttr(kNzTileSize),
                                     b.getIndexAttr(kNzTileSize)};
    SmallVector<OpFoldResult, 4> strs{b.getIndexAttr(1), b.getIndexAttr(1),
                                      b.getIndexAttr(1), b.getIndexAttr(1)};
    auto subview = b.create<memref::SubViewOp>(loc, p.l1Alloc, offs, szs, strs);
    b.create<hivm::CopyOp>(loc, mlir::TypeRange{}, cast.getResult(),
                           subview.getResult());
  }
  return success();
}

static FailureOr<Value>
materializeVectorLaneRegion(VectorToCubePack &pack, unsigned lane) {
  Operation *probabilityProducer = pack.pSrc.getDefiningOp();
  Operation *anchor = pack.anchor;
  if (!probabilityProducer || !anchor ||
      probabilityProducer->getBlock() != anchor->getBlock() ||
      !probabilityProducer->isBeforeInBlock(anchor))
    return failure();

  Operation *precedingWait = nullptr;
  for (Operation *cursor = probabilityProducer->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    if (isa<hivm::SyncBlockWaitOp>(cursor)) {
      precedingWait = cursor;
      break;
    }
  }
  if (!precedingWait || !precedingWait->getNextNode() ||
      precedingWait->getNextNode() == anchor)
    return failure();

  Operation *first = precedingWait->getNextNode();
  SmallVector<Operation *> operations;
  DenseSet<Operation *> operationSet;
  bool containsProbabilityProducer = false;
  for (Operation *cursor = first; cursor && cursor != anchor;
       cursor = cursor->getNextNode()) {
    if (cursor->hasTrait<OpTrait::IsTerminator>() ||
        isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(cursor))
      return failure();
    bool hasShapedResult = llvm::any_of(cursor->getResultTypes(), [](Type type) {
      return isa<ShapedType>(type);
    });
    if (!hasShapedResult)
      continue;
    operations.push_back(cursor);
    operationSet.insert(cursor);
    containsProbabilityProducer |= cursor == probabilityProducer;
  }
  if (!containsProbabilityProducer || operations.empty())
    return failure();

  SetVector<Value> outputs;
  outputs.insert(pack.pSrc);
  for (Operation *operation : operations) {
    for (Value result : operation->getResults()) {
      bool usedOutside = llvm::any_of(result.getUses(), [&](OpOperand &use) {
        return !operationSet.contains(use.getOwner());
      });
      if (usedOutside)
        outputs.insert(result);
    }
  }

  SmallVector<Type> resultTypes;
  for (Value output : outputs)
    resultTypes.push_back(output.getType());
  Location loc = probabilityProducer->getLoc();
  MLIRContext *context = probabilityProducer->getContext();
  OpBuilder builder(first);
  auto simdScope = builder.create<scope::ScopeOp>(loc, resultTypes);
  simdScope.getBodyRegion().emplaceBlock();
  simdScope->setAttr("noinline", UnitAttr::get(context));
  simdScope->setAttr("outline", BoolAttr::get(context, true));
  simdScope->setAttr("vector_mode", StringAttr::get(context, "simd"));
  setOpEngineTypeAttr(simdScope, EngineType::VECTOR);

  Block *scopeBlock = &simdScope.getBodyRegion().front();
  for (Operation *operation : operations) {
    operation->remove();
    scopeBlock->push_back(operation);
  }
  OpBuilder syncBuilder(probabilityProducer);
  auto syncToken = syncBuilder.create<arith::ConstantIntOp>(loc, 0, 64);
  auto syncMark =
      syncBuilder.create<annotation::MarkOp>(loc, syncToken.getResult());
  syncMark->setAttr("SYNC_IN_VF", StringAttr::get(context, "VST_VLD"));
  setOpEngineTypeAttr(syncToken, EngineType::VECTOR);
  setOpEngineTypeAttr(syncMark, EngineType::VECTOR);
  OpBuilder returnBuilder(scopeBlock, scopeBlock->end());
  auto returnOp =
      returnBuilder.create<scope::ReturnOp>(loc, outputs.getArrayRef());

  Value outlinedProbability;
  for (auto [index, output] : llvm::enumerate(outputs)) {
    Value replacement = simdScope->getResult(index);
    if (output == pack.pSrc)
      outlinedProbability = replacement;
    SmallVector<OpOperand *> outsideUses;
    for (OpOperand &use : output.getUses()) {
      Operation *owner = use.getOwner();
      if (owner != returnOp && !simdScope->isAncestor(owner))
        outsideUses.push_back(&use);
    }
    for (OpOperand *use : outsideUses)
      use->set(replacement);
  }
  if (!outlinedProbability)
    return failure();

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] materialized-vector-lane lane=" << lane
             << " operations=" << operations.size()
             << " results=" << outputs.size()
             << " vector-mode=simd publication=detached\n");
  return outlinedProbability;
}

static RankedTensorType getRowSliceType(RankedTensorType type, int64_t rows) {
  if (!type.hasStaticShape() || type.getRank() == 0 ||
      type.getDimSize(0) != rows)
    return type;
  SmallVector<int64_t> shape(type.getShape());
  shape[0] = 1;
  return RankedTensorType::get(shape, type.getElementType(),
                               type.getEncoding());
}

static FailureOr<Value>
materializeRowwiseVectorRegion(VectorToCubePack &pack, unsigned lane) {
  auto probabilityType = dyn_cast<RankedTensorType>(pack.pSrc.getType());
  Operation *probabilityProducer = pack.pSrc.getDefiningOp();
  Operation *anchor = pack.anchor;
  if (!probabilityType || !probabilityType.hasStaticShape() ||
      probabilityType.getRank() != 2 || probabilityType.getDimSize(0) <= 0 ||
      !probabilityProducer || !anchor ||
      probabilityProducer->getBlock() != anchor->getBlock() ||
      !probabilityProducer->isBeforeInBlock(anchor))
    return failure();
  int64_t rows = probabilityType.getDimSize(0);

  Operation *precedingWait = nullptr;
  for (Operation *cursor = probabilityProducer->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    if (isa<hivm::SyncBlockWaitOp>(cursor)) {
      precedingWait = cursor;
      break;
    }
  }
  if (!precedingWait || !precedingWait->getNextNode())
    return failure();

  SmallVector<Operation *> operations;
  DenseSet<Operation *> operationSet;
  bool containsProbabilityProducer = false;
  for (Operation *cursor = precedingWait->getNextNode(); cursor;
       cursor = cursor->getNextNode()) {
    if (cursor->hasTrait<OpTrait::IsTerminator>())
      break;
    if (isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(cursor))
      continue;
    bool hasRowTensorResult = llvm::any_of(
        cursor->getResultTypes(), [&](Type type) {
          auto tensor = dyn_cast<RankedTensorType>(type);
          return tensor && tensor.hasStaticShape() && tensor.getRank() > 0 &&
                 tensor.getDimSize(0) == rows;
        });
    if (!hasRowTensorResult || isa<bufferization::ToTensorOp>(cursor) ||
        isa<arith::ConstantOp>(cursor))
      continue;
    operations.push_back(cursor);
    operationSet.insert(cursor);
    containsProbabilityProducer |= cursor == probabilityProducer;
  }
  if (!containsProbabilityProducer || operations.empty())
    return failure();

  SetVector<Value> outputs;
  outputs.insert(pack.pSrc);
  for (Operation *operation : operations)
    for (Value result : operation->getResults()) {
      auto tensor = dyn_cast<RankedTensorType>(result.getType());
      if (!tensor || tensor.getRank() == 0 || tensor.getDimSize(0) != rows)
        continue;
      if (llvm::any_of(result.getUses(), [&](OpOperand &use) {
            return !operationSet.contains(use.getOwner());
          }))
        outputs.insert(result);
    }

  SmallVector<Type> resultTypes;
  SmallVector<Value> initValues;
  for (Value output : outputs)
    resultTypes.push_back(output.getType());
  Location loc = probabilityProducer->getLoc();
  MLIRContext *context = probabilityProducer->getContext();
  OpBuilder builder(operations.front());
  auto simdScope = builder.create<scope::ScopeOp>(loc, resultTypes);
  simdScope.getBodyRegion().emplaceBlock();
  simdScope->setAttr("noinline", UnitAttr::get(context));
  simdScope->setAttr("outline", BoolAttr::get(context, true));
  simdScope->setAttr("vector_mode", StringAttr::get(context, "simd"));
  setOpEngineTypeAttr(simdScope, EngineType::VECTOR);

  Block *scopeBlock = &simdScope.getBodyRegion().front();
  OpBuilder scopeBuilder = OpBuilder::atBlockEnd(scopeBlock);
  for (Value output : outputs) {
    auto type = cast<RankedTensorType>(output.getType());
    initValues.push_back(scopeBuilder.create<tensor::EmptyOp>(
        loc, type.getShape(), type.getElementType(), type.getEncoding()));
  }
  Value lower = scopeBuilder.create<arith::ConstantIndexOp>(loc, 0);
  Value upper = scopeBuilder.create<arith::ConstantIndexOp>(loc, rows);
  Value step = scopeBuilder.create<arith::ConstantIndexOp>(loc, 1);
  auto rowLoop = scopeBuilder.create<scf::ForOp>(loc, lower, upper, step,
                                                 initValues);
  Block *rowBody = rowLoop.getBody();
  if (!rowBody->empty())
    rowBody->back().erase();
  OpBuilder rowBuilder = OpBuilder::atBlockEnd(rowBody);
  Value row = rowLoop.getInductionVar();
  IRMapping mapping;

  auto mapExternal = [&](Value value) -> Value {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor || !tensor.hasStaticShape() || tensor.getRank() == 0 ||
        tensor.getDimSize(0) != rows)
      return value;
    RankedTensorType rowType = getRowSliceType(tensor, rows);
    SmallVector<OpFoldResult> offsets(tensor.getRank(),
                                      rowBuilder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(tensor.getRank(),
                                      rowBuilder.getIndexAttr(1));
    offsets[0] = row;
    for (int64_t size : rowType.getShape())
      sizes.push_back(rowBuilder.getIndexAttr(size));
    Value slice = rowBuilder
                      .create<tensor::ExtractSliceOp>(
                          loc, rowType, value, offsets, sizes, strides)
                      .getResult();
    mapping.map(value, slice);
    return slice;
  };

  for (Operation *operation : operations) {
    for (Value operand : operation->getOperands())
      if (!mapping.contains(operand) &&
          !operationSet.contains(operand.getDefiningOp()))
        (void)mapExternal(operand);
    if (operation == probabilityProducer) {
      auto token = rowBuilder.create<arith::ConstantIntOp>(loc, 0, 64);
      auto mark =
          rowBuilder.create<annotation::MarkOp>(loc, token.getResult());
      mark->setAttr("SYNC_IN_VF", StringAttr::get(context, "VST_VLD"));
    }
    if (isa<linalg::ReduceOp>(operation) &&
        operation->getNumOperands() == 2 &&
        operation->getNumResults() == 1) {
      Value input = mapping.lookupOrDefault(operation->getOperand(0));
      Value accumulator =
          mapping.lookupOrDefault(operation->getOperand(1));
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      constexpr int64_t chunkWidth = 4 * kNzTileSize;
      if (inputType && inputType.getRank() == 2 &&
          inputType.getDimSize(0) == 1 &&
          inputType.getDimSize(1) > chunkWidth &&
          inputType.getDimSize(1) % chunkWidth == 0) {
        for (int64_t chunk = 0; chunk < inputType.getDimSize(1);
             chunk += chunkWidth) {
          auto chunkType = RankedTensorType::get(
              {1, chunkWidth}, inputType.getElementType(),
              inputType.getEncoding());
          SmallVector<OpFoldResult> offsets{
              rowBuilder.getIndexAttr(0), rowBuilder.getIndexAttr(chunk)};
          SmallVector<OpFoldResult> sizes{
              rowBuilder.getIndexAttr(1),
              rowBuilder.getIndexAttr(chunkWidth)};
          SmallVector<OpFoldResult> strides{
              rowBuilder.getIndexAttr(1), rowBuilder.getIndexAttr(1)};
          Value slice = rowBuilder
                            .create<tensor::ExtractSliceOp>(
                                loc, chunkType, input, offsets, sizes, strides)
                            .getResult();
          IRMapping reductionMapping;
          reductionMapping.map(operation->getOperand(0), slice);
          reductionMapping.map(operation->getOperand(1), accumulator);
          Operation *clone = rowBuilder.clone(*operation, reductionMapping);
          Value reduced = clone->getResult(0);
          auto originalType =
              cast<RankedTensorType>(operation->getResult(0).getType());
          reduced.setType(getRowSliceType(originalType, rows));
          accumulator = reduced;
        }
        mapping.map(operation->getResult(0), accumulator);
        continue;
      }
    }
    Operation *clone = rowBuilder.clone(*operation, mapping);
    for (auto [original, cloned] :
         llvm::zip_equal(operation->getResults(), clone->getResults())) {
      auto tensor = dyn_cast<RankedTensorType>(original.getType());
      if (tensor)
        cloned.setType(getRowSliceType(tensor, rows));
    }
  }

  SmallVector<Value> yielded;
  for (auto [index, output] : llvm::enumerate(outputs)) {
    Value rowValue = mapping.lookupOrNull(output);
    if (!rowValue)
      return failure();
    auto rowType = cast<RankedTensorType>(rowValue.getType());
    SmallVector<OpFoldResult> offsets(rowType.getRank(),
                                      rowBuilder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(rowType.getRank(),
                                      rowBuilder.getIndexAttr(1));
    offsets[0] = row;
    for (int64_t size : rowType.getShape())
      sizes.push_back(rowBuilder.getIndexAttr(size));
    yielded.push_back(rowBuilder.create<tensor::InsertSliceOp>(
        loc, rowValue, rowLoop.getRegionIterArgs()[index], offsets, sizes,
        strides));
  }
  rowBuilder.create<scf::YieldOp>(loc, yielded);
  scopeBuilder.create<scope::ReturnOp>(loc, rowLoop.getResults());

  Value outlinedProbability;
  for (auto [index, output] : llvm::enumerate(outputs)) {
    Value replacement = simdScope->getResult(index);
    if (output == pack.pSrc)
      outlinedProbability = replacement;
    SmallVector<OpOperand *> outsideUses;
    for (OpOperand &use : output.getUses())
      if (!operationSet.contains(use.getOwner()))
        outsideUses.push_back(&use);
    for (OpOperand *use : outsideUses)
      use->set(replacement);
  }
  for (Operation *operation : llvm::reverse(operations))
    operation->erase();
  if (!outlinedProbability)
    return failure();

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] materialized-rowwise-lane lane=" << lane
             << " operations=" << operations.size() << " rows=" << rows
             << " results=" << outputs.size()
             << " vector-mode=simd publication=detached\n");
  return outlinedProbability;
}

struct OnlineSoftmaxLaneState {
  Value oldMaximum;
  Value oldDenominator;
  Value maximum;
  Value sum;
  Value packedProbability;
  arith::SubFOp alphaDifference;
  math::ExpOp alpha;
  arith::MulFOp scaledDenominator;
  arith::AddFOp newDenominator;
  Operation *scoreReleaseAnchor = nullptr;
};

static FailureOr<OnlineSoftmaxLaneState>
materializeOnlineSoftmaxLane(VectorToCubePack &pack, unsigned lane,
                                      bool deferLaneSum,
                                      Value &sharedScaleScalar,
                                      Value &sharedScaleRow) {
  Operation *probabilityProducer = pack.pSrc.getDefiningOp();
  Operation *anchor = pack.anchor;
  auto probabilityType = dyn_cast<RankedTensorType>(pack.pSrc.getType());
  if (!probabilityProducer || !anchor || !probabilityType ||
      !probabilityType.hasStaticShape() || probabilityType.getRank() != 2 ||
      probabilityProducer->getBlock() != anchor->getBlock())
    return failure();
  int64_t rows = probabilityType.getDimSize(0);
  int64_t width = probabilityType.getDimSize(1);
  constexpr int64_t chunkWidth = 4 * kNzTileSize;
  if (rows <= 0 || width <= 0 || width % chunkWidth != 0)
    return failure();

  Operation *precedingWait = nullptr;
  for (Operation *cursor = probabilityProducer->getPrevNode(); cursor;
       cursor = cursor->getPrevNode())
    if (isa<hivm::SyncBlockWaitOp>(cursor)) {
      precedingWait = cursor;
      break;
    }
  if (!precedingWait)
    return failure();

  SmallVector<Operation *> operations;
  DenseSet<Operation *> operationSet;
  arith::MulFOp scaledScore;
  linalg::ReduceOp maxReduce;
  arith::MaximumFOp newMaximum;
  math::ExpOp probabilityExp;
  linalg::ReduceOp sumReduce;
  math::ExpOp alpha;
  arith::AddFOp newDenominator;
  for (Operation *cursor = precedingWait->getNextNode(); cursor;
       cursor = cursor->getNextNode()) {
    if (cursor->hasTrait<OpTrait::IsTerminator>())
      break;
    if (isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(cursor))
      continue;
    bool hasRowTensorResult = llvm::any_of(
        cursor->getResultTypes(), [&](Type type) {
          auto tensor = dyn_cast<RankedTensorType>(type);
          return tensor && tensor.hasStaticShape() && tensor.getRank() > 0 &&
                 tensor.getDimSize(0) == rows;
        });
    if (!hasRowTensorResult || isa<bufferization::ToTensorOp>(cursor) ||
        isa<arith::ConstantOp>(cursor))
      continue;
    if (auto op = dyn_cast<arith::MulFOp>(cursor)) {
      auto result = dyn_cast<RankedTensorType>(op.getType());
      if (result && result.getRank() == 2 && !scaledScore)
        scaledScore = op;
    } else if (auto op = dyn_cast<linalg::ReduceOp>(cursor)) {
      if (!maxReduce)
        maxReduce = op;
      else if (!sumReduce)
        sumReduce = op;
    } else if (auto op = dyn_cast<arith::MaximumFOp>(cursor)) {
      auto result = dyn_cast<RankedTensorType>(op.getType());
      if (result && result.getRank() == 1)
        newMaximum = op;
    } else if (auto op = dyn_cast<math::ExpOp>(cursor)) {
      auto result = dyn_cast<RankedTensorType>(op.getType());
      if (result && result.getRank() == 2)
        probabilityExp = op;
      else if (result && result.getRank() == 1)
        alpha = op;
    } else if (auto op = dyn_cast<arith::AddFOp>(cursor)) {
      auto result = dyn_cast<RankedTensorType>(op.getType());
      if (result && result.getRank() == 1)
        newDenominator = op;
    }
    if (scaledScore && maxReduce && newMaximum && probabilityExp &&
        sumReduce && alpha && newDenominator)
      break;
  }
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-match lane=" << lane
             << " scaled=" << (scaledScore ? "yes" : "no")
             << " max-reduce=" << (maxReduce ? "yes" : "no")
             << " maximum=" << (newMaximum ? "yes" : "no")
             << " probability-exp=" << (probabilityExp ? "yes" : "no")
             << " sum-reduce=" << (sumReduce ? "yes" : "no")
             << " alpha=" << (alpha ? "yes" : "no")
             << " denominator=" << (newDenominator ? "yes" : "no")
             << "\n");
  if (!scaledScore || !maxReduce || !newMaximum || !probabilityExp ||
      !sumReduce || !alpha || !newDenominator ||
      probabilityProducer != pack.pSrc.getDefiningOp())
    return failure();

  Value score = scaledScore.getLhs();
  Value scale = scaledScore.getRhs();
  Value oldMaximum = newMaximum.getLhs() == maxReduce.getResult(0)
                         ? newMaximum.getRhs()
                         : newMaximum.getLhs();
  Value alphaInput = alpha.getOperand();
  auto alphaSub = alphaInput.getDefiningOp<arith::SubFOp>();
  if (!alphaSub)
    return failure();
  Value oldDenominator;
  arith::MulFOp scaledDenominator;
  bool addsLaneSum = false;
  for (Value operand : newDenominator->getOperands())
    if (auto mul = operand.getDefiningOp<arith::MulFOp>()) {
      if (mul.getLhs() == alpha.getResult()) {
        oldDenominator = mul.getRhs();
        scaledDenominator = mul;
      } else if (mul.getRhs() == alpha.getResult()) {
        oldDenominator = mul.getLhs();
        scaledDenominator = mul;
      }
    } else if (operand == sumReduce.getResult(0)) {
      addsLaneSum = true;
    }
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-denominator lane=" << lane
             << " old=" << (oldDenominator ? "yes" : "no") << "\n");
  if (!oldDenominator || !scaledDenominator || !addsLaneSum ||
      alphaSub.getLhs() != oldMaximum ||
      alphaSub.getRhs() != newMaximum.getResult())
    return failure();

  DenseSet<Value> externalValues{score, scale, oldMaximum, oldDenominator,
                                 maxReduce.getDpsInits()[0],
                                 sumReduce.getDpsInits()[0]};
  SmallVector<Operation *> worklist{
      scaledScore.getOperation(), maxReduce.getOperation(),
      newMaximum.getOperation(), probabilityExp.getOperation(),
      probabilityProducer, sumReduce.getOperation()};
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    if (!operationSet.insert(operation).second)
      continue;
    for (Value operand : operation->getOperands()) {
      if (externalValues.contains(operand))
        continue;
      Operation *definition = operand.getDefiningOp();
      if (!definition || definition->getBlock() != anchor->getBlock() ||
          isa<bufferization::ToTensorOp, arith::ConstantOp>(definition))
        continue;
      bool hasRowTensorResult = llvm::any_of(
          definition->getResultTypes(), [&](Type type) {
            auto tensor = dyn_cast<RankedTensorType>(type);
            return tensor && tensor.hasStaticShape() && tensor.getRank() > 0 &&
                   tensor.getDimSize(0) == rows;
          });
      if (hasRowTensorResult)
        worklist.push_back(definition);
    }
  }
  for (Operation &operation : *anchor->getBlock())
    if (operationSet.contains(&operation))
      operations.push_back(&operation);
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-closure lane=" << lane
             << " operations=" << operations.size() << "\n");
  if (operations.empty())
    return failure();

  Location loc = probabilityProducer->getLoc();
  MLIRContext *context = probabilityProducer->getContext();
  Type f32 = cast<RankedTensorType>(scaledScore.getType()).getElementType();
  Type pElement = probabilityType.getElementType();
  auto rowVectorType = RankedTensorType::get({1, chunkWidth}, f32);
  auto rowScalarType = RankedTensorType::get({1}, f32);
  auto maximumType = RankedTensorType::get({rows}, f32);
  auto scaledType = RankedTensorType::get({rows, width}, f32);
  int64_t n16 = width / kNzTileSize;
  auto packedType =
      RankedTensorType::get({n16, rows, kNzTileSize}, pElement);

  auto scaleFill = scale.getDefiningOp<linalg::FillOp>();
  if (!scaleFill || scaleFill.getInputs().size() != 1 ||
      scaleFill->getNumResults() != 1 || scaleFill.getResult(0) != scale)
    return failure();
  Value scaleScalar = scaleFill.getInputs()[0];
  if (sharedScaleScalar) {
    if (sharedScaleScalar != scaleScalar ||
        sharedScaleRow.getType() != rowVectorType)
      return failure();
  } else {
    OpBuilder scaleBuilder(scaleFill);
    Location scaleLoc =
        NameLoc::get(scaleBuilder.getStringAttr("cvsplit.softmax.shared-scale-row"),
                     scaleFill.getLoc());
    Value scaleRowInit = scaleBuilder.create<tensor::EmptyOp>(
        scaleLoc, rowVectorType.getShape(), rowVectorType.getElementType(),
        rowVectorType.getEncoding());
    sharedScaleRow = scaleBuilder
                         .create<linalg::FillOp>(
                             scaleLoc, ValueRange{scaleScalar},
                             ValueRange{scaleRowInit})
                         .getResult(0);
    sharedScaleScalar = scaleScalar;
  }

  Operation *insertionAnchor = operations.front();
  auto createLoopStorage = [&](OpBuilder &storageBuilder,
                               RankedTensorType tensorType,
                               StringRef role) -> Value {
    Location storageLoc =
        NameLoc::get(storageBuilder.getStringAttr(role), loc);
    auto ubAddressSpace = storageBuilder.getAttr<hivm::AddressSpaceAttr>(
        hivm::AddressSpace::UB);
    auto ubType = MemRefType::get(tensorType.getShape(),
                                  tensorType.getElementType(), nullptr,
                                  ubAddressSpace);
    auto allocation = storageBuilder.create<memref::AllocOp>(storageLoc,
                                                              ubType);
    auto mark = storageBuilder.create<annotation::MarkOp>(
        storageLoc, allocation.getResult());
    mark->setAttr("effects",
                  storageBuilder.getArrayAttr(
                      {storageBuilder.getStringAttr("write"),
                       storageBuilder.getStringAttr("read")}));
    auto plainType = MemRefType::get(tensorType.getShape(),
                                     tensorType.getElementType());
    auto cast = storageBuilder.create<memref::MemorySpaceCastOp>(
        storageLoc, plainType, allocation.getResult());
    return storageBuilder
        .create<bufferization::ToTensorOp>(
            storageLoc, tensorType, cast.getResult(),
            /*restrict=*/true, /*writable=*/true)
        .getResult();
  };
  auto createReductionInit =
      [&](OpBuilder &initBuilder, linalg::ReduceOp reduction,
          RankedTensorType initType) -> FailureOr<Value> {
    auto fill = reduction.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    if (!fill || fill.getInputs().size() != 1)
      return failure();
    Value fillInput = fill.getInputs()[0];
    if (Operation *definition = fillInput.getDefiningOp();
        definition && definition->getBlock() == insertionAnchor->getBlock() &&
        insertionAnchor->isBeforeInBlock(definition)) {
      if (!isa<arith::ConstantOp>(definition))
        return failure();
      fillInput = initBuilder.clone(*definition)->getResult(0);
    }
    Value empty = initBuilder
                      .create<tensor::EmptyOp>(
                          loc, initType.getShape(), initType.getElementType(),
                          initType.getEncoding())
                      .getResult();
    return initBuilder
        .create<linalg::FillOp>(loc, ValueRange{fillInput}, ValueRange{empty})
        .getResult(0);
  };

  OpBuilder builder(insertionAnchor);
  Value sumRowsInit =
      createLoopStorage(builder, maximumType, "cvsplit.softmax.sum-rows");
  Value scaledRowsInit =
      createLoopStorage(builder, scaledType, "cvsplit.softmax.scaled-rows");
  Value packedRowsInit =
      createLoopStorage(builder, packedType, "cvsplit.softmax.packed-rows");
  Value maxRowsInit =
      createLoopStorage(builder, maximumType, "cvsplit.softmax.max-rows");
  Value deferredAddInit;
  if (deferLaneSum) {
    Location deferredLoc =
        NameLoc::get(builder.getStringAttr("cvsplit.softmax.deferred-add-row"), loc);
    deferredAddInit = builder.create<tensor::EmptyOp>(
        deferredLoc, rowVectorType.getShape(), rowVectorType.getElementType(),
        rowVectorType.getEncoding());
  }
  SmallVector<Type> scopeResults;
  if (deferLaneSum)
    scopeResults.append({scaledType, maximumType, packedType});
  else
    scopeResults.append({maximumType, maximumType, packedType});
  auto simdScope = builder.create<scope::ScopeOp>(loc, scopeResults);
  simdScope.getBodyRegion().emplaceBlock();
  simdScope->setAttr("noinline", UnitAttr::get(context));
  simdScope->setAttr("outline", BoolAttr::get(context, true));
  simdScope->setAttr("vector_mode", StringAttr::get(context, "simd"));
  setOpEngineTypeAttr(simdScope, EngineType::VECTOR);
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=scope-created\n");
  Block *scopeBlock = &simdScope.getBodyRegion().front();
  OpBuilder b = OpBuilder::atBlockEnd(scopeBlock);
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=scope-builder-ready\n");

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=parent-loop-inits-ready\n");
  Value lower = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value upper = b.create<arith::ConstantIntOp>(loc, rows, 32);
  Value step = b.create<arith::ConstantIntOp>(loc, 1, 32);
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-loop-bounds-created\n");
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-loop-create-begin\n");
  auto maxLoop = b.create<scf::ForOp>(
      loc, lower, upper, step,
      ValueRange{maxRowsInit, scaledRowsInit});
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-loop-create-end\n");
  Block *maxBody = maxLoop.getBody();
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-loop-body-ready\n");
  if (!maxBody->empty())
    maxBody->back().erase();
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-loop-default-yield-erased\n");
  OpBuilder mb = OpBuilder::atBlockEnd(maxBody);
  Value row = mb.create<arith::IndexCastOp>(
      loc, mb.getIndexType(), maxLoop.getInductionVar());
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-loop-shell-created\n");

  auto extractRowChunk = [&](OpBuilder &rb, Value tensorValue, int64_t chunk,
                             Type elementType) {
    auto type = RankedTensorType::get({1, chunkWidth}, elementType);
    SmallVector<OpFoldResult> offsets{row, rb.getIndexAttr(chunk)};
    SmallVector<OpFoldResult> sizes{rb.getIndexAttr(1),
                                    rb.getIndexAttr(chunkWidth)};
    SmallVector<OpFoldResult> strides{rb.getIndexAttr(1), rb.getIndexAttr(1)};
    return rb.create<tensor::ExtractSliceOp>(loc, type, tensorValue, offsets,
                                              sizes, strides)
        .getResult();
  };
  auto insertRowChunk = [&](OpBuilder &rb, Value value, Value destination,
                            int64_t chunk) {
    SmallVector<OpFoldResult> offsets{row, rb.getIndexAttr(chunk)};
    SmallVector<OpFoldResult> sizes{rb.getIndexAttr(1),
                                    rb.getIndexAttr(chunkWidth)};
    SmallVector<OpFoldResult> strides{rb.getIndexAttr(1), rb.getIndexAttr(1)};
    return rb.create<tensor::InsertSliceOp>(loc, value, destination, offsets,
                                             sizes, strides)
        .getResult();
  };
  Value scaledRows = maxLoop.getRegionIterArgs()[1];
  Value combinedMaximum;
  for (int64_t chunk = 0; chunk < width; chunk += chunkWidth) {
    Value scoreChunk = extractRowChunk(mb, score, chunk, f32);
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split] online-softmax-build-progress lane=" << lane
               << " checkpoint=max-score-chunk-extracted chunk=" << chunk
               << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split] online-softmax-build-progress lane=" << lane
               << " checkpoint=max-shared-scale-row-ready chunk=" << chunk
               << "\n");
    Value scaled =
        mb.create<arith::MulFOp>(loc, scoreChunk, sharedScaleRow);
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split] online-softmax-build-progress lane=" << lane
               << " checkpoint=max-chunk-scaled chunk=" << chunk << "\n");
    scaledRows = insertRowChunk(mb, scaled, scaledRows, chunk);
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split] online-softmax-build-progress lane=" << lane
               << " checkpoint=max-scaled-chunk-inserted chunk=" << chunk
               << "\n");
    combinedMaximum = combinedMaximum
                          ? mb.create<arith::MaximumFOp>(loc, combinedMaximum,
                                                        scaled)
                                .getResult()
                          : scaled;
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-split] online-softmax-build-progress lane=" << lane
               << " checkpoint=max-chunks-combined chunk=" << chunk << "\n");
  }
  SmallVector<OpFoldResult> scalarOffset{row};
  SmallVector<OpFoldResult> scalarSize{mb.getIndexAttr(1)};
  SmallVector<OpFoldResult> scalarStride{mb.getIndexAttr(1)};
  FailureOr<Value> maxInit = createReductionInit(mb, maxReduce, rowScalarType);
  if (failed(maxInit))
    return failure();
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-init-materialized\n");
  IRMapping maxMapping;
  maxMapping.map(maxReduce.getDpsInputs()[0], combinedMaximum);
  maxMapping.map(maxReduce.getDpsInits()[0], *maxInit);
  Operation *maxClone = mb.clone(*maxReduce, maxMapping);
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-reduce-cloned\n");
  maxClone->getResult(0).setType(rowScalarType);
  Value maxRows = mb.create<tensor::InsertSliceOp>(
      loc, maxClone->getResult(0), maxLoop.getRegionIterArgs()[0],
      scalarOffset, scalarSize, scalarStride);
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-result-inserted\n");
  mb.create<scf::YieldOp>(loc, ValueRange{maxRows, scaledRows});
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=max-loop-created\n");

  Value maximum =
      b.create<arith::MaximumFOp>(loc, oldMaximum, maxLoop.getResult(0));
  auto syncToken = b.create<arith::ConstantIntOp>(loc, 0, 64);
  auto syncMark = b.create<annotation::MarkOp>(loc, syncToken.getResult());
  syncMark->setAttr("SYNC_IN_VF", StringAttr::get(context, "VST_VLD"));
  Value expRowsInit = deferLaneSum ? maxLoop.getResult(1) : sumRowsInit;
  auto expLoop = b.create<scf::ForOp>(
      loc, lower, upper, step, ValueRange{expRowsInit, packedRowsInit});
  Block *expBody = expLoop.getBody();
  if (!expBody->empty())
    expBody->back().erase();
  OpBuilder eb = OpBuilder::atBlockEnd(expBody);
  row = eb.create<arith::IndexCastOp>(loc, eb.getIndexType(),
                                      expLoop.getInductionVar());
  Value maximumRow = eb.create<tensor::ExtractSliceOp>(
      loc, rowScalarType, maximum, SmallVector<OpFoldResult>{row}, scalarSize,
      scalarStride);
  auto broadcastInit =
      eb.create<tensor::EmptyOp>(loc, rowVectorType.getShape(), f32);
  auto maximumBroadcastOp = eb.create<linalg::BroadcastOp>(
      loc, maximumRow, broadcastInit, ArrayRef<int64_t>{1});
  Value maximumBroadcast = maximumBroadcastOp->getResult(0);
  Value sumChunks;
  Value expRows = expLoop.getRegionIterArgs()[0];
  Value packedRows = expLoop.getRegionIterArgs()[1];
  for (int64_t chunk = 0; chunk < width; chunk += chunkWidth) {
    Value scaled = extractRowChunk(eb, maxLoop.getResult(1), chunk, f32);
    Value shifted = eb.create<arith::SubFOp>(loc, scaled, maximumBroadcast);
    Value exponential = eb.create<math::ExpOp>(loc, shifted);
    if (deferLaneSum)
      expRows = insertRowChunk(eb, exponential, expRows, chunk);
    else
      sumChunks = sumChunks
                      ? eb.create<arith::AddFOp>(loc, sumChunks, exponential)
                            .getResult()
                      : exponential;
    auto shapeType = RankedTensorType::get({3}, b.getI64Type());
    auto shape = eb.create<arith::ConstantOp>(
        loc, shapeType,
        DenseElementsAttr::get(
            shapeType,
            ArrayRef<int64_t>{chunkWidth / kNzTileSize, 1, kNzTileSize}));
    auto packedFloatChunkType = RankedTensorType::get(
        {chunkWidth / kNzTileSize, 1, kNzTileSize}, f32);
    Value packedFloatChunk = eb.create<tensor::ReshapeOp>(
        loc, packedFloatChunkType, exponential, shape);
    auto packedChunkType = RankedTensorType::get(
        {chunkWidth / kNzTileSize, 1, kNzTileSize}, pElement);
    Value packedChunk = eb.create<arith::TruncFOp>(
        loc, packedChunkType, packedFloatChunk);
    SmallVector<OpFoldResult> offsets{
        eb.getIndexAttr(chunk / kNzTileSize), row, eb.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{
        eb.getIndexAttr(chunkWidth / kNzTileSize), eb.getIndexAttr(1),
        eb.getIndexAttr(kNzTileSize)};
    SmallVector<OpFoldResult> strides(3, eb.getIndexAttr(1));
    packedRows = eb.create<tensor::InsertSliceOp>(
        loc, packedChunk, packedRows, offsets, sizes, strides);
  }
  if (!deferLaneSum) {
    FailureOr<Value> sumInit =
        createReductionInit(eb, sumReduce, rowScalarType);
    if (failed(sumInit))
      return failure();
    IRMapping sumMapping;
    sumMapping.map(sumReduce.getDpsInputs()[0], sumChunks);
    sumMapping.map(sumReduce.getDpsInits()[0], *sumInit);
    Operation *sumClone = eb.clone(*sumReduce, sumMapping);
    sumClone->getResult(0).setType(rowScalarType);
    expRows = eb.create<tensor::InsertSliceOp>(
        loc, sumClone->getResult(0), expRows,
        SmallVector<OpFoldResult>{row}, scalarSize, scalarStride);
  }
  eb.create<scf::YieldOp>(loc, ValueRange{expRows, packedRows});
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=exp-pack-loop-created\n");

  if (deferLaneSum)
    b.create<scope::ReturnOp>(
        loc, ValueRange{expLoop.getResult(0), maximum, expLoop.getResult(1)});
  else
    b.create<scope::ReturnOp>(
        loc, ValueRange{maximum, expLoop.getResult(0), expLoop.getResult(1)});
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] online-softmax-build-progress lane=" << lane
             << " checkpoint=scope-return-created\n");

  Value materializedMaximum =
      simdScope->getResult(deferLaneSum ? 1 : 0);
  Value materializedSum = simdScope->getResult(1);
  Operation *scoreReleaseAnchor = anchor;
  if (deferLaneSum) {
    OpBuilder deferredBuilder(anchor);
    deferredBuilder.setInsertionPointAfter(anchor);
    auto deferredScope =
        deferredBuilder.create<scope::ScopeOp>(loc, TypeRange{maximumType});
    deferredScope.getBodyRegion().emplaceBlock();
    deferredScope->setAttr("noinline", UnitAttr::get(context));
    deferredScope->setAttr("outline", BoolAttr::get(context, true));
    deferredScope->setAttr("vector_mode", StringAttr::get(context, "simd"));
    setOpEngineTypeAttr(deferredScope, EngineType::VECTOR);

    OpBuilder db =
        OpBuilder::atBlockEnd(&deferredScope.getBodyRegion().front());
    Value deferredLower = db.create<arith::ConstantIntOp>(loc, 0, 32);
    Value deferredUpper = db.create<arith::ConstantIntOp>(loc, rows, 32);
    Value deferredStep = db.create<arith::ConstantIntOp>(loc, 1, 32);
    auto deferredLoop = db.create<scf::ForOp>(
        loc, deferredLower, deferredUpper, deferredStep,
        ValueRange{sumRowsInit});
    Block *deferredBody = deferredLoop.getBody();
    if (!deferredBody->empty())
      deferredBody->back().erase();
    OpBuilder rb = OpBuilder::atBlockEnd(deferredBody);
    Value deferredRow = rb.create<arith::IndexCastOp>(
        loc, rb.getIndexType(), deferredLoop.getInductionVar());
    Value deferredChunks;
    for (int64_t chunk = 0; chunk < width; chunk += chunkWidth) {
      SmallVector<OpFoldResult> offsets{deferredRow,
                                        rb.getIndexAttr(chunk)};
      SmallVector<OpFoldResult> sizes{rb.getIndexAttr(1),
                                      rb.getIndexAttr(chunkWidth)};
      SmallVector<OpFoldResult> strides{rb.getIndexAttr(1),
                                        rb.getIndexAttr(1)};
      Value probabilityChunk = rb.create<tensor::ExtractSliceOp>(
          loc, rowVectorType, simdScope->getResult(0), offsets, sizes,
          strides);
      if (!deferredChunks) {
        deferredChunks = probabilityChunk;
      } else {
        auto add = rb.create<linalg::AddOp>(
            loc, ValueRange{deferredChunks, probabilityChunk},
            ValueRange{deferredAddInit});
        deferredChunks = add.getResult(0);
      }
    }
    FailureOr<Value> deferredInit =
        createReductionInit(rb, sumReduce, rowScalarType);
    if (failed(deferredInit))
      return failure();
    IRMapping deferredMapping;
    deferredMapping.map(sumReduce.getDpsInputs()[0], deferredChunks);
    deferredMapping.map(sumReduce.getDpsInits()[0], *deferredInit);
    Operation *deferredClone = rb.clone(*sumReduce, deferredMapping);
    deferredClone->getResult(0).setType(rowScalarType);
    Value deferredRows = rb.create<tensor::InsertSliceOp>(
        loc, deferredClone->getResult(0), deferredLoop.getRegionIterArgs()[0],
        SmallVector<OpFoldResult>{deferredRow}, scalarSize, scalarStride);
    rb.create<scf::YieldOp>(loc, ValueRange{deferredRows});
    db.create<scope::ReturnOp>(loc, deferredLoop.getResult(0));
    materializedSum = deferredScope->getResult(0);
    scoreReleaseAnchor = deferredScope;
  }

  SmallVector<std::pair<Value, Value>> replacements{
      {newMaximum.getResult(), materializedMaximum},
      {sumReduce.getResult(0), materializedSum}};
  for (auto [oldValue, replacement] : replacements) {
    SmallVector<OpOperand *> outsideUses;
    for (OpOperand &use : oldValue.getUses())
      if (!operationSet.contains(use.getOwner()))
        outsideUses.push_back(&use);
    for (OpOperand *use : outsideUses)
      use->set(replacement);
  }
  for (Operation *operation : llvm::reverse(operations))
    operation->erase();

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] materialized-online-softmax lane=" << lane
             << " rows=" << rows << " chunk-width=" << chunkWidth
             << " chunks=" << (width / chunkWidth)
             << " direct-nz=yes sum="
             << (deferLaneSum ? "deferred" : "inline")
             << " publication=detached\n");
  return OnlineSoftmaxLaneState{
      oldMaximum,
      oldDenominator,
      materializedMaximum,
      materializedSum,
      simdScope->getResult(2),
      alphaSub,
      alpha,
      scaledDenominator,
      newDenominator,
      scoreReleaseAnchor,
  };
}

static LogicalResult materializeGroupedSoftmaxRecurrence(
    ArrayRef<OnlineSoftmaxLaneState> lanes,
    Operation *&groupedAlphaScope) {
  if (lanes.empty())
    return failure();

  OnlineSoftmaxLaneState firstState = lanes.front();
  OnlineSoftmaxLaneState finalState = lanes.back();
  Block *block = firstState.alphaDifference->getBlock();
  auto rowType = dyn_cast<RankedTensorType>(firstState.maximum.getType());
  if (!block || !rowType || !rowType.hasStaticShape() ||
      rowType.getRank() != 1)
    return failure();

  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    OnlineSoftmaxLaneState state = lanes[lane];
    OnlineSoftmaxLaneState previousState =
        lane == 0 ? firstState : lanes[lane - 1];
    if (state.alphaDifference->getBlock() != block ||
        state.alpha->getBlock() != block ||
        state.scaledDenominator->getBlock() != block ||
        state.newDenominator->getBlock() != block ||
        state.maximum.getType() != rowType || state.sum.getType() != rowType ||
        state.oldDenominator.getType() != rowType ||
        state.alphaDifference.getLhs() != state.oldMaximum ||
        state.alphaDifference.getRhs() != state.maximum ||
        (lane != 0 && state.oldMaximum != previousState.maximum) ||
        (lane != 0 &&
         state.oldDenominator != previousState.newDenominator.getResult()))
      return failure();
  }

  DenseSet<Operation *> recurrenceOperations;
  for (OnlineSoftmaxLaneState state : lanes) {
    recurrenceOperations.insert(state.alphaDifference);
    recurrenceOperations.insert(state.alpha);
    recurrenceOperations.insert(state.scaledDenominator);
    recurrenceOperations.insert(state.newDenominator);
  }

  SmallVector<OpOperand *> finalDenominatorUses;
  for (OpOperand &use : finalState.newDenominator.getResult().getUses())
    if (!recurrenceOperations.contains(use.getOwner()))
      finalDenominatorUses.push_back(&use);
  if (finalDenominatorUses.empty())
    return failure();
  Operation *affineInsertion = finalDenominatorUses.front()->getOwner();
  if (affineInsertion->getBlock() != block ||
      llvm::any_of(finalDenominatorUses, [&](OpOperand *use) {
        return use->getOwner() != affineInsertion;
      }))
    return failure();

  Location loc = finalState.alphaDifference.getLoc();
  MLIRContext *context = block->getParentOp()->getContext();
  OpBuilder alphaBuilder(finalState.alphaDifference);
  SmallVector<Type> alphaResultTypes(lanes.size(), rowType);
  auto alphaScope =
      alphaBuilder.create<scope::ScopeOp>(loc, alphaResultTypes);
  alphaScope.getBodyRegion().emplaceBlock();
  alphaScope->setAttr("noinline", UnitAttr::get(context));
  alphaScope->setAttr("outline", BoolAttr::get(context, true));
  alphaScope->setAttr("vector_mode", StringAttr::get(context, "simd"));
  setOpEngineTypeAttr(alphaScope, EngineType::VECTOR);
  OpBuilder ab = OpBuilder::atBlockEnd(&alphaScope.getBodyRegion().front());
  SmallVector<Value> groupedAlphas;
  Value previousMaximum = firstState.oldMaximum;
  for (OnlineSoftmaxLaneState state : lanes) {
    Value difference =
        ab.create<arith::SubFOp>(loc, previousMaximum, state.maximum);
    groupedAlphas.push_back(ab.create<math::ExpOp>(loc, difference));
    previousMaximum = state.maximum;
  }
  ab.create<scope::ReturnOp>(loc, groupedAlphas);

  // Each lane is an affine transform l' = A*l + B. Compose adjacent
  // transforms as a balanced tree:
  //   (Ar, Br) o (Al, Bl) = (Ar*Al, Ar*Bl + Br).
  // This preserves lane order while making the reduction depth logarithmic
  // for any positive lane count, including non-power-of-two unroll factors.
  OpBuilder affineBuilder(affineInsertion);
  auto affineScope =
      affineBuilder.create<scope::ScopeOp>(loc, TypeRange{rowType});
  affineScope.getBodyRegion().emplaceBlock();
  affineScope->setAttr("noinline", UnitAttr::get(context));
  affineScope->setAttr("outline", BoolAttr::get(context, true));
  affineScope->setAttr("vector_mode", StringAttr::get(context, "simd"));
  setOpEngineTypeAttr(affineScope, EngineType::VECTOR);
  OpBuilder fb = OpBuilder::atBlockEnd(&affineScope.getBodyRegion().front());
  struct AffineSegment {
    Value scale;
    Value offset;
  };
  SmallVector<AffineSegment> segments;
  segments.reserve(lanes.size());
  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    OnlineSoftmaxLaneState state = lanes[lane];
    segments.push_back({alphaScope->getResult(lane), state.sum});
  }
  while (segments.size() > 1) {
    SmallVector<AffineSegment> next;
    next.reserve((segments.size() + 1) / 2);
    for (size_t index = 0; index < segments.size(); index += 2) {
      if (index + 1 == segments.size()) {
        next.push_back(segments[index]);
        continue;
      }
      const AffineSegment &left = segments[index];
      const AffineSegment &right = segments[index + 1];
      Value scale = fb.create<arith::MulFOp>(loc, left.scale, right.scale);
      Value scaledLeft =
          fb.create<arith::MulFOp>(loc, left.offset, right.scale);
      Value offset = fb.create<arith::AddFOp>(loc, scaledLeft, right.offset);
      next.push_back({scale, offset});
    }
    segments = std::move(next);
  }
  Value scaledOld = fb.create<arith::MulFOp>(
      loc, firstState.oldDenominator, segments.front().scale);
  Value finalDenominator =
      fb.create<arith::AddFOp>(loc, scaledOld, segments.front().offset);
  fb.create<scope::ReturnOp>(loc, finalDenominator);

  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    OnlineSoftmaxLaneState state = lanes[lane];
    SmallVector<OpOperand *> alphaUses;
    for (OpOperand &use : state.alpha.getResult().getUses())
      alphaUses.push_back(&use);
    for (OpOperand *use : alphaUses)
      use->set(alphaScope->getResult(lane));
  }
  for (OpOperand *use : finalDenominatorUses)
    use->set(affineScope->getResult(0));

  for (auto iterator = lanes.rbegin(); iterator != lanes.rend(); ++iterator) {
    OnlineSoftmaxLaneState state = *iterator;
    if (!state.newDenominator->use_empty())
      return failure();
    state.newDenominator.erase();
    if (!state.scaledDenominator->use_empty())
      return failure();
    state.scaledDenominator.erase();
    if (!state.alpha->use_empty())
      return failure();
    state.alpha.erase();
    if (!state.alphaDifference->use_empty())
      return failure();
    state.alphaDifference.erase();
  }

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] materialized-grouped-recurrence lanes="
             << lanes.size() << " alpha-scope=yes affine-tree=yes\n");
  groupedAlphaScope = alphaScope;
  return success();
}

static LogicalResult
materializeVectorLaneRegions(MutableArrayRef<VectorToCubePack> packs,
                            Operation *&groupedAlphaScope) {
  SmallVector<OnlineSoftmaxLaneState> lanes;
  lanes.reserve(packs.size());
  Value sharedScaleScalar;
  Value sharedScaleRow;
  for (auto [lane, pack] : llvm::enumerate(packs)) {
    FailureOr<OnlineSoftmaxLaneState> result =
        materializeOnlineSoftmaxLane(
            pack, lane, lane + 1 == packs.size(), sharedScaleScalar,
            sharedScaleRow);
    if (failed(result))
      return failure();
    pack.pSrc = result->packedProbability;
    pack.scoreReleaseAnchor = result->scoreReleaseAnchor;
    lanes.push_back(*result);
  }
  return materializeGroupedSoftmaxRecurrence(lanes, groupedAlphaScope);
}

static FailureOr<BufferOwnershipPlan> buildBufferOwnershipPlan(
    scf::ForOp cubeLoop, scf::ForOp vectorLoop,
    const CrossScopeTransferInfo &transferInfo,
    const PostCVSplitDetachedSchedule &schedule) {
  if (schedule.status != PostCVSplitDetachedScheduleStatus::Ready ||
      !schedule.verified || schedule.logicalLaneCount == 0 ||
      transferInfo.cubeToVectorChains.size() !=
          2 * schedule.logicalLaneCount)
    return failure();

  auto findCubeDrain = [&](unsigned forwardFlag) -> Operation * {
    Operation *drain = nullptr;
    for (Operation &operation : *cubeLoop.getBody()) {
      auto set = dyn_cast<hivm::SyncBlockSetOp>(&operation);
      if (!set || getStaticEventFlag(set) != forwardFlag)
        continue;
      if (drain || !isa_and_nonnull<hivm::FixpipeOp>(set->getPrevNode()))
        return nullptr;
      drain = set->getPrevNode();
    }
    return drain;
  };

  BufferOwnershipPlan plan;
  DenseSet<Value> oldBufferSet;
  DenseSet<unsigned> scoreLanes;
  DenseSet<unsigned> productLanes;
  for (const CubeToVectorTransferChain &chain :
       transferInfo.cubeToVectorChains) {
    if (!chain.wait || chain.wait->getBlock() != vectorLoop.getBody() ||
        chain.forwardFlagId < 0)
      return failure();
    unsigned forwardFlag = static_cast<unsigned>(chain.forwardFlagId);
    const PostCVSplitDetachedCommand *publish = findDetachedCommandByFlag(
        schedule.cubeCommands, PostCVSplitDetachedCommandKind::ScorePublish,
        forwardFlag);
    PostCVSplitLineageRole role = PostCVSplitLineageRole::Score;
    if (!publish) {
      publish = findDetachedCommandByFlag(
          schedule.cubeCommands,
          PostCVSplitDetachedCommandKind::ProductPublish, forwardFlag);
      role = PostCVSplitLineageRole::Product;
    }
    if (!publish || publish->lane >= schedule.logicalLaneCount)
      return failure();
    DenseSet<unsigned> &seen = role == PostCVSplitLineageRole::Score
                                   ? scoreLanes
                                   : productLanes;
    if (!seen.insert(publish->lane).second)
      return failure();

    Operation *toTensor = chain.transferredValue.getDefiningOp();
    if (!toTensor || !isa<bufferization::ToTensorOp>(toTensor) ||
        toTensor->getNumOperands() != 1)
      return failure();
    Operation *vectorCast = toTensor->getOperand(0).getDefiningOp();
    if (!vectorCast || !isa<memref::MemorySpaceCastOp>(vectorCast) ||
        vectorCast->getNumOperands() != 1)
      return failure();
    Value oldBuffer = vectorCast->getOperand(0);
    auto bufferType = dyn_cast<MemRefType>(oldBuffer.getType());
    Operation *cubeDrain = findCubeDrain(forwardFlag);
    if (!bufferType || !cubeDrain ||
        !llvm::is_contained(cubeDrain->getOperands(), oldBuffer))
      return failure();

    MemRefType &roleType = role == PostCVSplitLineageRole::Score
                               ? plan.scoreBufferType
                               : plan.productBufferType;
    unsigned &slotCount = role == PostCVSplitLineageRole::Score
                              ? plan.scoreSlotCount
                              : plan.productSlotCount;
    if (roleType && roleType != bufferType)
      return failure();
    roleType = bufferType;
    slotCount = std::max(slotCount, publish->slot + 1);
    plan.lanes.push_back({role, publish->lane, publish->slot, oldBuffer,
                          cubeDrain, vectorCast});
    if (oldBufferSet.insert(oldBuffer).second)
      plan.oldBuffers.push_back(oldBuffer);
  }
  if (scoreLanes.size() != schedule.logicalLaneCount ||
      productLanes.size() != schedule.logicalLaneCount ||
      !plan.scoreBufferType || !plan.productBufferType ||
      plan.scoreSlotCount == 0 || plan.productSlotCount == 0)
    return failure();

  auto cubeScope = cubeLoop->getParentOfType<scope::ScopeOp>();
  if (!cubeScope)
    return failure();
  for (Value oldBuffer : plan.oldBuffers) {
    auto allocation = oldBuffer.getDefiningOp<memref::AllocOp>();
    if (!allocation || allocation->getBlock() != cubeScope->getBlock())
      return failure();
  }
  return plan;
}

static LogicalResult materializeBufferOwnership(
    scf::ForOp cubeLoop, scf::ForOp vectorLoop,
    const CrossScopeTransferInfo &transferInfo,
    const PostCVSplitDetachedSchedule &schedule) {
  FailureOr<BufferOwnershipPlan> plan = buildBufferOwnershipPlan(
      cubeLoop, vectorLoop, transferInfo, schedule);
  if (failed(plan))
    return failure();

  auto cubeScope = cubeLoop->getParentOfType<scope::ScopeOp>();
  if (!cubeScope)
    return failure();
  MLIRContext *context = cubeLoop.getContext();
  Location loc = cubeLoop.getLoc();
  OpBuilder builder(cubeScope);
  auto createSlots = [&](MemRefType type, unsigned count, StringRef role) {
    SmallVector<Value> slots;
    slots.reserve(count);
    for (unsigned slot = 0; slot < count; ++slot) {
      Location slotLoc = NameLoc::get(
          builder.getStringAttr((role + "-slot").str()), loc);
      auto allocation = builder.create<memref::AllocOp>(slotLoc, type);
      auto mark =
          builder.create<annotation::MarkOp>(slotLoc, allocation.getResult());
      mark->setAttr("effects",
                    builder.getArrayAttr({builder.getStringAttr("write"),
                                          builder.getStringAttr("read")}));
      slots.push_back(allocation.getResult());
    }
    return slots;
  };
  SmallVector<Value> scoreSlots = createSlots(
      plan->scoreBufferType, plan->scoreSlotCount, "cvsplit.schedule.score");
  SmallVector<Value> productSlots = createSlots(
      plan->productBufferType, plan->productSlotCount, "cvsplit.schedule.product");

  for (const OwnershipLanePlan &lane : plan->lanes) {
    ArrayRef<Value> slots = lane.role == PostCVSplitLineageRole::Score
                                ? ArrayRef<Value>(scoreSlots)
                                : ArrayRef<Value>(productSlots);
    if (lane.slot >= slots.size())
      return failure();
    Value replacement = slots[lane.slot];
    lane.cubeDrain->replaceUsesOfWith(lane.oldBuffer, replacement);
    lane.vectorCast->setOperand(0, replacement);
  }

  for (Value oldBuffer : plan->oldBuffers) {
    SmallVector<Operation *> marks;
    for (Operation *user : oldBuffer.getUsers()) {
      if (!isa<annotation::MarkOp>(user))
        return failure();
      marks.push_back(user);
    }
    for (Operation *mark : marks)
      mark->erase();
    auto allocation = oldBuffer.getDefiningOp<memref::AllocOp>();
    if (!allocation || !allocation->use_empty())
      return failure();
    allocation.erase();
  }

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] materialized-buffer-ownership lanes="
             << schedule.logicalLaneCount
             << " score-slots=" << plan->scoreSlotCount
             << " product-slots=" << plan->productSlotCount
             << " retired-union-buffers=" << plan->oldBuffers.size()
             << "\n");
  return success();
}

static FailureOr<ReleaseProtocolPlan> buildReleaseProtocolPlan(
    scf::ForOp cubeLoop, scf::ForOp vectorLoop,
    ArrayRef<VectorToCubePack> packs,
    const CrossScopeTransferInfo &transferInfo,
    const PostCVSplitDetachedSchedule &schedule) {
  if (schedule.status != PostCVSplitDetachedScheduleStatus::Ready ||
      !schedule.verified || schedule.logicalLaneCount == 0 ||
      schedule.logicalLaneCount != packs.size())
    return failure();

  ReleaseProtocolPlan plan;
  plan.lanes.reserve(schedule.logicalLaneCount);
  auto findCubeDrain = [&](unsigned forwardFlag) -> Operation * {
    Operation *drain = nullptr;
    for (Operation &operation : *cubeLoop.getBody()) {
      auto set = dyn_cast<hivm::SyncBlockSetOp>(&operation);
      if (!set || getStaticEventFlag(set) != forwardFlag)
        continue;
      if (drain || !isa_and_nonnull<hivm::FixpipeOp>(set->getPrevNode()))
        return nullptr;
      drain = set->getPrevNode();
    }
    return drain;
  };
  auto findProductVectorRelease = [&](unsigned forwardFlag) -> Operation * {
    const CubeToVectorTransferChain *matchingChain = nullptr;
    for (const CubeToVectorTransferChain &chain :
         transferInfo.cubeToVectorChains) {
      if (chain.forwardFlagId != static_cast<int>(forwardFlag))
        continue;
      if (matchingChain)
        return nullptr;
      matchingChain = &chain;
    }
    if (!matchingChain || !matchingChain->wait ||
        matchingChain->wait->getBlock() != vectorLoop.getBody())
      return nullptr;
    Operation *lastConsumer = nullptr;
    for (Operation *consumer : matchingChain->consumers) {
      if (!consumer || consumer->getBlock() != vectorLoop.getBody())
        return nullptr;
      if (!lastConsumer || lastConsumer->isBeforeInBlock(consumer))
        lastConsumer = consumer;
    }
    return lastConsumer;
  };

  for (unsigned lane = 0; lane < schedule.logicalLaneCount; ++lane) {
    const PostCVSplitDetachedCommand *scorePublish = findDetachedCommand(
        schedule.cubeCommands, PostCVSplitDetachedCommandKind::ScorePublish,
        lane);
    const PostCVSplitDetachedCommand *scoreReleaseWait = findDetachedCommand(
        schedule.cubeCommands,
        PostCVSplitDetachedCommandKind::ScoreReleaseWait, lane);
    const PostCVSplitDetachedCommand *scoreRelease = findDetachedCommand(
        schedule.vectorCommands, PostCVSplitDetachedCommandKind::ScoreRelease,
        lane);
    const PostCVSplitDetachedCommand *probabilityPublish = findDetachedCommand(
        schedule.vectorCommands,
        PostCVSplitDetachedCommandKind::ProbabilityPublish, lane);
    const PostCVSplitDetachedCommand *productPublish = findDetachedCommand(
        schedule.cubeCommands, PostCVSplitDetachedCommandKind::ProductPublish,
        lane);
    const PostCVSplitDetachedCommand *productReleaseWait = findDetachedCommand(
        schedule.cubeCommands,
        PostCVSplitDetachedCommandKind::ProductReleaseWait, lane);
    const PostCVSplitDetachedCommand *productRelease = findDetachedCommand(
        schedule.vectorCommands,
        PostCVSplitDetachedCommandKind::ProductRelease, lane);
    if (!scorePublish || !scoreReleaseWait || !scoreRelease ||
        !probabilityPublish || !productPublish || !productReleaseWait ||
        !productRelease || !scorePublish->hasEvent ||
        !scoreReleaseWait->hasEvent || !scoreRelease->hasEvent ||
        !probabilityPublish->hasEvent || !productPublish->hasEvent ||
        !productReleaseWait->hasEvent || !productRelease->hasEvent ||
        scoreReleaseWait->logicalFlagId != scoreRelease->logicalFlagId ||
        productReleaseWait->logicalFlagId != productRelease->logicalFlagId ||
        scoreReleaseWait->slot != scoreRelease->slot ||
        productReleaseWait->slot != productRelease->slot)
      return failure();

    VectorToCubePack pack = packs[lane];
    if (!pack.anchor || !pack.scoreReleaseAnchor ||
        getStaticEventFlag(pack.anchor) !=
            probabilityPublish->logicalFlagId)
      return failure();

    ReleaseLanePlan lanePlan;
    lanePlan.lane = lane;
    lanePlan.scoreReleaseFlag = scoreRelease->logicalFlagId;
    lanePlan.productReleaseFlag = productRelease->logicalFlagId;
    lanePlan.scoreSignalingResource = scoreRelease->resource;
    lanePlan.scoreWaitingResource = scoreReleaseWait->resource;
    lanePlan.productSignalingResource = productRelease->resource;
    lanePlan.productWaitingResource = productReleaseWait->resource;
    lanePlan.scoreCubeDrain = findCubeDrain(scorePublish->logicalFlagId);
    lanePlan.scoreVectorRelease = pack.scoreReleaseAnchor;
    lanePlan.productCubeDrain = findCubeDrain(productPublish->logicalFlagId);
    lanePlan.productVectorRelease =
        findProductVectorRelease(productPublish->logicalFlagId);
    if (!lanePlan.scoreCubeDrain || !lanePlan.scoreVectorRelease ||
        !lanePlan.productCubeDrain || !lanePlan.productVectorRelease)
      return failure();
    plan.lanes.push_back(lanePlan);
  }

  auto appendInitial = [&](PostCVSplitLineageRole role, unsigned slot,
                           unsigned flag, PrincipalResource signaling,
                           PrincipalResource waiting) {
    for (const InitialReleaseSignal &existing : plan.initialSignals) {
      if (existing.flag != flag)
        continue;
      return existing.role == role && existing.slot == slot &&
             existing.signalingResource == signaling &&
             existing.waitingResource == waiting;
    }
    plan.initialSignals.push_back({role, slot, flag, signaling, waiting});
    return true;
  };
  for (PostCVSplitLineageRole role : {PostCVSplitLineageRole::Score,
                                      PostCVSplitLineageRole::Product}) {
    for (unsigned lane = 0; lane < schedule.logicalLaneCount; ++lane) {
      PostCVSplitDetachedCommandKind kind =
          role == PostCVSplitLineageRole::Score
              ? PostCVSplitDetachedCommandKind::ScoreRelease
              : PostCVSplitDetachedCommandKind::ProductRelease;
      const PostCVSplitDetachedCommand *release =
          findDetachedCommand(schedule.vectorCommands, kind, lane);
      const PostCVSplitDetachedCommand *wait = findDetachedCommand(
          schedule.cubeCommands,
          role == PostCVSplitLineageRole::Score
              ? PostCVSplitDetachedCommandKind::ScoreReleaseWait
              : PostCVSplitDetachedCommandKind::ProductReleaseWait,
          lane);
      if (!release || !wait ||
          !appendInitial(role, release->slot, release->logicalFlagId,
                         release->resource, wait->resource))
        return failure();
    }
  }

  DenseSet<unsigned> productReleaseFlags;
  for (const InitialReleaseSignal &initial : plan.initialSignals)
    if (initial.role == PostCVSplitLineageRole::Product)
      productReleaseFlags.insert(initial.flag);
  MLIRContext *context = cubeLoop.getContext();
  auto pipeV = hivm::PipeAttr::get(context, hivm::PIPE::PIPE_V);
  auto pipeFix = hivm::PipeAttr::get(context, hivm::PIPE::PIPE_FIX);
  for (Operation &operation : *cubeLoop.getBody()) {
    auto wait = dyn_cast<hivm::SyncBlockWaitOp>(&operation);
    if (!wait || wait.getTpipe() != pipeV || wait.getPipe() != pipeFix)
      continue;
    std::optional<unsigned> flag = getStaticEventFlag(wait);
    if (!flag || productReleaseFlags.contains(*flag))
      continue;
    if (plan.legacyCubeWait)
      return failure();
    plan.legacyCubeWait = wait;
  }
  for (Operation &operation : *vectorLoop.getBody()) {
    auto set = dyn_cast<hivm::SyncBlockSetOp>(&operation);
    if (!set || set.getTpipe() != pipeV || set.getPipe() != pipeFix)
      continue;
    std::optional<unsigned> flag = getStaticEventFlag(set);
    if (!flag || productReleaseFlags.contains(*flag))
      continue;
    if (plan.legacyVectorSet)
      return failure();
    plan.legacyVectorSet = set;
  }
  if (!plan.legacyCubeWait || !plan.legacyVectorSet ||
      getStaticEventFlag(plan.legacyCubeWait) !=
          getStaticEventFlag(plan.legacyVectorSet))
    return failure();
  return plan;
}

static LogicalResult materializeReleaseProtocol(
    scf::ForOp cubeLoop, scf::ForOp vectorLoop,
    ArrayRef<VectorToCubePack> packs,
    const CrossScopeTransferInfo &transferInfo,
    const PostCVSplitDetachedSchedule &schedule) {
  FailureOr<ReleaseProtocolPlan> plan =
      buildReleaseProtocolPlan(cubeLoop, vectorLoop, packs,
                                      transferInfo, schedule);
  if (failed(plan))
    return failure();

  plan->legacyVectorSet->erase();
  plan->legacyCubeWait->erase();

  MLIRContext *context = cubeLoop.getContext();
  Location loc = cubeLoop.getLoc();
  auto cubeCore =
      hivm::TCoreTypeAttr::get(context, hivm::TCoreType::CUBE);
  auto vectorCore =
      hivm::TCoreTypeAttr::get(context, hivm::TCoreType::VECTOR);
  scf::ForOp cubeOuterLoop = cubeLoop->getParentOfType<scf::ForOp>();
  scf::ForOp vectorOuterLoop = vectorLoop->getParentOfType<scf::ForOp>();
  if (!cubeOuterLoop || !vectorOuterLoop ||
      cubeOuterLoop.getOperation() != vectorOuterLoop.getOperation() ||
      vectorOuterLoop.getInductionVar().getType() !=
          vectorOuterLoop.getLowerBound().getType())
    return failure();

  OpBuilder initialBuilder(vectorLoop);
  auto firstOuterIteration = initialBuilder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, vectorOuterLoop.getInductionVar(),
      vectorOuterLoop.getLowerBound());
  auto seedIf = initialBuilder.create<scf::IfOp>(
      loc, firstOuterIteration, /*withElseRegion=*/false);
  OpBuilder seedBuilder = seedIf.getThenBodyBuilder();
  for (const InitialReleaseSignal &initial : plan->initialSignals) {
    hivm::PipeAttr signaling =
        pipeForResource(context, initial.signalingResource);
    hivm::PipeAttr waiting =
        pipeForResource(context, initial.waitingResource);
    if (!signaling || !waiting)
      return failure();
    auto set = seedBuilder.create<hivm::SyncBlockSetOp>(
        loc, vectorCore, signaling, waiting,
        OpFoldResult(seedBuilder.getI64IntegerAttr(initial.flag)));
    setOpEngineTypeAttr(set, EngineType::VECTOR);
  }

  for (const ReleaseLanePlan &lane : plan->lanes) {
    hivm::PipeAttr scoreSignaling =
        pipeForResource(context, lane.scoreSignalingResource);
    hivm::PipeAttr scoreWaiting =
        pipeForResource(context, lane.scoreWaitingResource);
    hivm::PipeAttr productSignaling =
        pipeForResource(context, lane.productSignalingResource);
    hivm::PipeAttr productWaiting =
        pipeForResource(context, lane.productWaitingResource);
    if (!scoreSignaling || !scoreWaiting || !productSignaling ||
        !productWaiting)
      return failure();

    OpBuilder scoreCubeBuilder(lane.scoreCubeDrain);
    auto scoreWait = scoreCubeBuilder.create<hivm::SyncBlockWaitOp>(
        loc, cubeCore, scoreSignaling, scoreWaiting,
        OpFoldResult(
            scoreCubeBuilder.getI64IntegerAttr(lane.scoreReleaseFlag)));
    setOpEngineTypeAttr(scoreWait, EngineType::CUBE);

    OpBuilder scoreVectorBuilder(lane.scoreVectorRelease);
    scoreVectorBuilder.setInsertionPointAfter(lane.scoreVectorRelease);
    auto scoreSet = scoreVectorBuilder.create<hivm::SyncBlockSetOp>(
        loc, vectorCore, scoreSignaling, scoreWaiting,
        OpFoldResult(
            scoreVectorBuilder.getI64IntegerAttr(lane.scoreReleaseFlag)));
    setOpEngineTypeAttr(scoreSet, EngineType::VECTOR);

    OpBuilder productCubeBuilder(lane.productCubeDrain);
    auto productWait = productCubeBuilder.create<hivm::SyncBlockWaitOp>(
        loc, cubeCore, productSignaling, productWaiting,
        OpFoldResult(
            productCubeBuilder.getI64IntegerAttr(lane.productReleaseFlag)));
    setOpEngineTypeAttr(productWait, EngineType::CUBE);

    OpBuilder productVectorBuilder(lane.productVectorRelease);
    productVectorBuilder.setInsertionPointAfter(lane.productVectorRelease);
    auto productSet = productVectorBuilder.create<hivm::SyncBlockSetOp>(
        loc, vectorCore, productSignaling, productWaiting,
        OpFoldResult(
            productVectorBuilder.getI64IntegerAttr(lane.productReleaseFlag)));
    setOpEngineTypeAttr(productSet, EngineType::VECTOR);
  }

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] materialized-release-protocol lanes="
             << plan->lanes.size()
             << " initial=" << plan->initialSignals.size()
             << " cube-waits=" << (2 * plan->lanes.size())
             << " vector-sets=" << (2 * plan->lanes.size())
             << " initial-once-per-outer-loop=yes"
             << " legacy-pairs-replaced=1\n");
  return success();
}

static LogicalResult orderGroupedAlphaBeforeProductWait(
    scf::ForOp vectorLoop, Operation *groupedAlphaScope,
    const PostCVSplitDetachedSchedule &schedule) {
  const PostCVSplitDetachedCommand *firstProductWait = nullptr;
  for (const PostCVSplitDetachedCommand &command : schedule.vectorCommands) {
    if (command.kind == PostCVSplitDetachedCommandKind::ProductWait) {
      firstProductWait = &command;
      break;
    }
  }
  if (!groupedAlphaScope || !firstProductWait ||
      !firstProductWait->hasEvent ||
      groupedAlphaScope->getBlock() != vectorLoop.getBody())
    return failure();

  Operation *waitOperation = nullptr;
  for (Operation &operation : *vectorLoop.getBody()) {
    auto wait = dyn_cast<hivm::SyncBlockWaitOp>(&operation);
    if (!wait ||
        getStaticEventFlag(wait) != firstProductWait->logicalFlagId)
      continue;
    if (waitOperation)
      return failure();
    waitOperation = wait;
  }
  if (!waitOperation)
    return failure();
  bool moved = waitOperation->isBeforeInBlock(groupedAlphaScope);
  if (moved)
    groupedAlphaScope->moveBefore(waitOperation);
  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] ordered-grouped-alpha-before-product-wait"
             << " moved=" << (moved ? "yes" : "no") << "\n");
  return success();
}

static LogicalResult orderCubeDrainsBySchedule(
    scf::ForOp cubeLoop, const PostCVSplitDetachedSchedule &schedule) {
  DenseMap<unsigned, unsigned> ordinalByForwardFlag;
  for (auto [ordinal, command] : llvm::enumerate(schedule.cubeCommands)) {
    if (!command.hasEvent ||
        (command.kind != PostCVSplitDetachedCommandKind::ScorePublish &&
         command.kind != PostCVSplitDetachedCommandKind::ProductPublish))
      continue;
    if (!ordinalByForwardFlag
             .try_emplace(command.logicalFlagId,
                          static_cast<unsigned>(ordinal))
             .second)
      return failure();
  }

  struct DrainGroup {
    Operation *releaseWait;
    Operation *drain;
    Operation *publish;
    unsigned ordinal;
  };
  SmallVector<DrainGroup> groups;
  for (Operation &operation : *cubeLoop.getBody()) {
    auto publish = dyn_cast<hivm::SyncBlockSetOp>(&operation);
    if (!publish)
      continue;
    std::optional<unsigned> flag = getStaticEventFlag(publish);
    if (!flag || !ordinalByForwardFlag.contains(*flag))
      continue;
    Operation *drain = publish->getPrevNode();
    Operation *releaseWait = drain ? drain->getPrevNode() : nullptr;
    if (!isa_and_nonnull<hivm::FixpipeOp>(drain) ||
        !isa_and_nonnull<hivm::SyncBlockWaitOp>(releaseWait))
      return failure();
    groups.push_back(
        {releaseWait, drain, publish, ordinalByForwardFlag.lookup(*flag)});
  }
  if (groups.size() != 2 * schedule.logicalLaneCount)
    return failure();

  unsigned reorderedClusters = 0;
  for (size_t begin = 0; begin < groups.size();) {
    size_t end = begin + 1;
    while (end < groups.size() &&
           groups[end - 1].publish->getNextNode() ==
               groups[end].releaseWait)
      ++end;
    if (end - begin > 1) {
      SmallVector<DrainGroup> ordered(groups.begin() + begin,
                                      groups.begin() + end);
      llvm::sort(ordered, [](const DrainGroup &left, const DrainGroup &right) {
        return left.ordinal < right.ordinal;
      });
      bool changed = false;
      for (size_t index = 0; index < ordered.size(); ++index)
        changed |= ordered[index].publish != groups[begin + index].publish;
      if (changed) {
        Operation *afterCluster = groups[end - 1].publish->getNextNode();
        if (!afterCluster)
          return failure();
        for (const DrainGroup &group : ordered) {
          group.releaseWait->moveBefore(afterCluster);
          group.drain->moveBefore(afterCluster);
          group.publish->moveBefore(afterCluster);
        }
        ++reorderedClusters;
      }
    }
    begin = end;
  }

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-split] ordered-cube-drain-clusters groups="
             << groups.size() << " reordered=" << reorderedClusters
             << "\n");
  return success();
}

// Re-tile the VECTOR scope for ROW_SPLIT so both veccores do useful work (2x
// vector throughput): M/2 rows per veccore, addressed by get_sub_block_idx,
// matching the target IR. Runs the six steps in order; see each helper.
static LogicalResult
retileVectorScopeForRowSplit(scope::ScopeOp vecScope, scf::ForOp cubeLoop,
                             scf::ForOp vectorLoop,
                             const CrossScopeTransferInfo &transferInfo,
                             bool materializePostSplitSchedule,
                             const PostCVSplitDetachedSchedule *
                                 detachedSchedule) {
  Location loc = vecScope.getLoc();
  MLIRContext *ctx = vecScope.getContext();
  auto ubAddrSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
  int64_t blockM = transferInfo.blockM;

  SmallVector<VectorToCubePack> packs =
      detachVectorToCubePacks(vecScope, transferInfo.vectorToCubeChains);
  Operation *groupedAlphaScope = nullptr;

  Value sbidx = emitSubBlockIndex(vecScope, loc);
  unsigned clonedCount = cloneExternalInitsAsHalfHeight(vecScope, loc, blockM);
  if (failed(retileVectorScopeOps(vecScope, blockM)))
    return failure();
  if (materializePostSplitSchedule &&
      failed(materializeVectorLaneRegions(packs, groupedAlphaScope)))
    return failure();
  FailureOr<unsigned> nStores =
      retileOutputStores(vecScope, sbidx, loc, blockM);
  if (failed(nStores))
    return failure();
  if (failed(rebuildVectorToCubePacks(packs, sbidx, ubAddrSpace, loc)))
    return failure();
  if (materializePostSplitSchedule &&
      (!detachedSchedule ||
       failed(materializeBufferOwnership(
           cubeLoop, vectorLoop, transferInfo, *detachedSchedule)) ||
       failed(materializeReleaseProtocol(
           cubeLoop, vectorLoop, packs, transferInfo,
           *detachedSchedule)) ||
       failed(orderGroupedAlphaBeforeProductWait(
           vectorLoop, groupedAlphaScope, *detachedSchedule)) ||
       failed(orderCubeDrainsBySchedule(cubeLoop,
                                            *detachedSchedule))))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "[cv-split]   ROW_SPLIT re-tile (BLOCK_M="
                          << blockM << " -> " << (blockM / 2)
                          << "/veccore): " << packs.size() << " V->C packs, "
                          << *nStores << " stores, " << clonedCount
                          << " ext consts cloned\n");
  return success();
}

} // namespace

LogicalResult
createScopeSeparation(func::FuncOp funcOp, scf::ForOp innerLoop,
                      const CrossScopeTransferInfo &transferInfo,
                      bool materializePostSplitSchedule,
                      const PostCVSplitDetachedSchedule *
                          detachedSchedule,
                      const CrossCoreScheduleCandidate *
                          scheduleCandidate) {

  MLIRContext *ctx = funcOp.getContext();
  Location loc = innerLoop.getLoc();

  // Strategy (matching reference FA kernel pattern):
  // - CUBE scope: clone of inner loop with only CUBE ops (before VECTOR scope)
  // - VECTOR scope: original inner loop + ALL subsequent ops in the parent
  // block
  //   (epilogue: normalize + store). Both go inside the VECTOR scope.
  // - Preamble (q load, ptr setup) stays in parent block before both scopes.
  // - Shared allocs stay at the function level (outside any scope).
  //
  // This ensures:
  //   parent_block {
  //     preamble...
  //     scope(CUBE) { inner_loop_clone { cube ops } }
  //     scope(VECTOR) { inner_loop { vector ops }; epilogue... }
  //   }

  Block *parentBlock = innerLoop->getBlock();
  OpBuilder builder(ctx);

  // Collect epilogue: all ops AFTER the inner loop in the parent block
  // (up to but not including the terminator)
  auto *terminator = parentBlock->getTerminator();
  SmallVector<Operation *> epilogueOps;
  bool afterLoop = false;
  for (Operation &op : *parentBlock) {
    if (&op == terminator)
      break;
    if (afterLoop)
      epilogueOps.push_back(&op);
    if (&op == innerLoop.getOperation())
      afterLoop = true;
  }

  // Step 0.5: Validate that the epilogue contains only VECTOR operations,
  // because it will be moved into the VECTOR scope.
  // FIXME: A general solution would return the inner loop results from the
  // VECTOR scope and leave the epilogue outside it.
  for (Operation *op : epilogueOps) {
    auto coreType = op->getAttrOfType<StringAttr>("ssbuffer.core_type");
    if (!coreType || coreType.getValue() != "VECTOR") {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cv-split] Non-VECTOR or unclassified epilogue op: "
                 << op->getName() << "\n");
      return failure();
    }
  }

  // Step 1: Create CUBE scope (placed BEFORE the inner loop)
  builder.setInsertionPoint(innerLoop);
  auto cubeScope = builder.create<scope::ScopeOp>(loc, ArrayRef<Type>{});
  cubeScope.getBodyRegion().emplaceBlock();
  cubeScope->setAttr("noinline", UnitAttr::get(ctx));

  // Clone the inner loop into CUBE scope
  Block *cubeBlock = &cubeScope.getBodyRegion().front();
  OpBuilder cubeBuilder(cubeBlock, cubeBlock->end());
  IRMapping cubeMapping;
  auto cubeLoop = cast<scf::ForOp>(
      cubeBuilder.clone(*innerLoop.getOperation(), cubeMapping));
  cubeBuilder.create<scope::ReturnOp>(loc);

  // Step 2: Create VECTOR scope (wraps the original inner loop + epilogue)
  builder.setInsertionPoint(innerLoop);
  auto vecScope = builder.create<scope::ScopeOp>(loc, ArrayRef<Type>{});
  vecScope.getBodyRegion().emplaceBlock();
  vecScope->setAttr("noinline", UnitAttr::get(ctx));
  Block *vecBlock = &vecScope.getBodyRegion().front();
  // Move original inner loop into VECTOR scope
  innerLoop->remove();
  vecBlock->push_back(innerLoop.getOperation());
  // Move epilogue ops into VECTOR scope
  for (Operation *op : epilogueOps) {
    op->remove();
    vecBlock->push_back(op);
  }
  OpBuilder vecBuilder(vecBlock, vecBlock->end());
  vecBuilder.create<scope::ReturnOp>(loc);

  // Step 3: Set core type attributes.
  cubeScope->setAttr(hivm::TCoreTypeAttr::name,
                     hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE));
  vecScope->setAttr(hivm::TCoreTypeAttr::name,
                    hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR));

  // Step 4: Strip wrong-type operations using their stamped core attributes.
  // Existing operations retain their engine classification; later
  // transformations stamp every operation they create, and cloning preserves
  // those attributes in both scopes.
  if (failed(stripWrongTypeOps(cubeScope, EngineType::CUBE)) ||
      failed(stripWrongTypeOps(vecScope, EngineType::VECTOR)))
    return failure();

  // Stripping can leave dead loop-carried state behind because the cloned
  // loops retain the original init/result/yield signature. Remove only entries
  // whose region argument and loop result are both unused.
  cubeLoop = removeUnusedLoopCarriedValues(cubeLoop);
  innerLoop = removeUnusedLoopCarriedValues(innerLoop);

  // Step 6: Hoist convert_layout ops out of the CUBE scope's loop.
  // These are view reshapes on L1 buffers (NZ→ND) that don't depend on loop
  // iteration state — they can be computed once before the loop starts.
  {
    Block *loopBody = cubeLoop.getBody();
    SmallVector<hivm::ConvertLayoutOp> toHoist;
    SmallVector<memref::MemorySpaceCastOp> castsToHoist;
    for (Operation &op : *loopBody) {
      if (auto cvtOp = dyn_cast<hivm::ConvertLayoutOp>(&op)) {
        // Only hoist if its input is defined OUTSIDE the loop (shared L1 alloc)
        Value input = cvtOp.getSource();
        bool conversionIsLoopInvariant =
            input.getDefiningOp() &&
            input.getDefiningOp()->getBlock() != loopBody;
        if (conversionIsLoopInvariant) {
          toHoist.push_back(cvtOp);

          // Hoist the pure memory-space view that directly consumes this
          // conversion. The later to_tensor remains after synchronization.
          for (Operation *user : cvtOp.getResult().getUsers()) {
            if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(user);
                castOp && castOp->getBlock() == loopBody)
              castsToHoist.push_back(castOp);
          }
        }
      }
    }
    // Move them before the loop (inside the scope block, before the scf.for)
    for (auto cvtOp : toHoist)
      cvtOp->moveBefore(cubeLoop);
    for (auto castOp : castsToHoist)
      castOp->moveBefore(cubeLoop);
  }

  // Step 6b: Sink each cube matmul's operand load chain to immediately before
  // the matmul. The BFS dependency scheduler groups every unrolled K/V
  // load at the top of the cube loop body, so all 8 cbuf staging buffers stay
  // simultaneously live. PlanMemory then assigns them high L1 offsets and the
  // matmul's FB operand load falls back to register offset (mode 2), which the
  // simulator's dmamov_decode_to_fb path rejects. Interleaving the loads (the
  // manual kernel allocates one K tile right before each matmul) keeps only the
  // in-flight operands live -> low static offsets -> immediate offset mode.
  unsigned productOperandPrefetchDepth = 1;
  if (materializePostSplitSchedule) {
    if (!detachedSchedule || !scheduleCandidate ||
        scheduleCandidate->logicalLaneCount !=
            detachedSchedule->logicalLaneCount ||
        scheduleCandidate->prefetchLimit == 0)
      return failure();
    productOperandPrefetchDepth = scheduleCandidate->prefetchLimit;
  }
  if (failed(sinkCubeLoadChainsToMatmul(
          cubeLoop.getBody(), productOperandPrefetchDepth)))
    return failure();

  // Step 7: ROW_SPLIT re-tile of the VECTOR scope (BLOCK_M/2 rows per veccore,
  // both veccores active). Replaces the single-veccore NO_DUAL guard.
  if (failed(retileVectorScopeForRowSplit(
          vecScope, cubeLoop, innerLoop, transferInfo,
          materializePostSplitSchedule, detachedSchedule)))
    return failure();

  LLVM_DEBUG(
      llvm::dbgs()
      << "[cv-split] Scope separation done: CUBE scope then VECTOR scope "
      << "(inside parent block, matching reference pattern)\n");
  return success();
}

} // namespace mlir::triton::cv_split
