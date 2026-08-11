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

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

namespace {

static StringRef engineTypeToStr(EngineType e)
{
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
static Value buildNeutralPlaceholder(OpBuilder &builder, Type type, Location loc)
{
    if (auto memrefTy = dyn_cast<MemRefType>(type)) {
        if (!memrefTy.hasStaticShape())
            return Value();
        return builder.create<memref::AllocOp>(loc, memrefTy).getResult();
    }

    if (auto shapedTy = dyn_cast<ShapedType>(type)) {
        if (shapedTy.hasStaticShape() && !isa<MemRefType>(type)) {
            if (Attribute elemZero = builder.getZeroAttr(shapedTy.getElementType())) {
                auto zeroDense = DenseElementsAttr::get(shapedTy, elemZero);
                return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(zeroDense)).getResult();
            }
        }
    }

    if (Attribute zeroAttr = builder.getZeroAttr(type)) {
        if (auto typedZero = dyn_cast<TypedAttr>(zeroAttr))
            return builder.create<arith::ConstantOp>(loc, type, typedZero).getResult();
    }

    return Value();
}

static FailureOr<EngineType> getStampedEngineType(Operation *op)
{
    auto coreType = op->getAttrOfType<StringAttr>("ssbuffer.core_type");
    if (!coreType) {
        op->emitError("missing core classification attribute");
        return failure();
    }

    if (coreType.getValue() == "CUBE")
        return EngineType::CUBE;
    if (coreType.getValue() == "VECTOR")
        return EngineType::VECTOR;

    op->emitError("expected a single-core classification {CUBE, VECTOR}, got ") << coreType.getValue();
    return failure();
}

static FailureOr<size_t> stripWrongTypeOpsFromBlock(Block &block, EngineType keepType)
{
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

            Value placeholder = buildNeutralPlaceholder(builder, result.getType(), op->getLoc());
            if (!placeholder) {
                op->emitError("cannot build a neutral placeholder for live result ") << result.getType();
                return failure();
            }
            result.replaceAllUsesWith(placeholder);
        }
        op->erase();
    }

    return erasedCount;
}

static LogicalResult stripWrongTypeOps(scope::ScopeOp scopeOp, EngineType keepType)
{
    LLVM_DEBUG(llvm::dbgs() << "[cv-split] Stripping " << (keepType == EngineType::CUBE ? "VECTOR" : "CUBE")
                            << " ops from " << engineTypeToStr(keepType) << " scope\n");

    Block &scopeBlock = scopeOp.getBodyRegion().front();

    FailureOr<size_t> topLevelErased = stripWrongTypeOpsFromBlock(scopeBlock, keepType);
    if (failed(topLevelErased))
        return failure();

    size_t loopErased = 0;
    WalkResult walkResult = scopeOp.walk([&](scf::ForOp forOp) {
        FailureOr<size_t> erased = stripWrongTypeOpsFromBlock(*forOp.getBody(), keepType);
        if (failed(erased))
            return WalkResult::interrupt();
        loopErased += *erased;
        return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
        return failure();

    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Erased " << *topLevelErased << " top-level + " << loopErased
                            << " loop ops from " << engineTypeToStr(keepType) << " scope\n");

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
        LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Cleaned up " << deadMarks.size() + deadAllocs.size()
                                << " dead alloc/mark ops\n");
    return success();
}

// Rebuild an scf.for without loop-carried values that are unused both inside
// the body and outside the loop. The init operand, region iter argument, loop
// result, and yield operand at each removed index are dropped together.
static scf::ForOp removeUnusedLoopCarriedValues(scf::ForOp loop)
{
    SmallVector<unsigned> keptIndices;
    unsigned numIterArgs = loop.getNumRegionIterArgs();
    keptIndices.reserve(numIterArgs);
    for (unsigned i = 0; i < numIterArgs; ++i) {
        if (!loop.getRegionIterArgs()[i].use_empty() || !loop.getResult(i).use_empty())
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
    auto newLoop = builder.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
                                              newInitArgs);
    newLoop->setAttrs(loop->getAttrs());

    Block *oldBody = loop.getBody();
    Block *newBody = newLoop.getBody();
    loop.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());
    for (auto [newIndex, oldIndex] : llvm::enumerate(keptIndices))
        loop.getRegionIterArgs()[oldIndex].replaceAllUsesWith(newLoop.getRegionIterArgs()[newIndex]);

    // Depending on the builder overload, the fresh body is either empty or has
    // an empty scf.yield. Remove that placeholder and build the real yield after
    // moving the original body operations.
    if (!newBody->empty()) {
        assert(isa<scf::YieldOp>(newBody->back()) && "fresh scf.for body must contain only its yield");
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
    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Removed " << removedCount << " unused loop-carried value(s)\n");
    return newLoop;
}

static LogicalResult retileVectorScopeForRowSplit(scope::ScopeOp vecScope, const CrossScopeTransferInfo &transferInfo);
static void sinkCubeLoadChainsToMatmul(Block *body);

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
static void sinkCubeLoadChainsToMatmul(Block *body)
{
    auto isChainType = [](Operation *o) {
        return isa<memref::ReinterpretCastOp, memref::AllocOp, memref::CopyOp, bufferization::ToTensorOp,
                   linalg::TransposeOp>(o);
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
                    if (copy && copy->getBlock() == body && copy.getTarget() == def->getResult(0)) {
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
    auto safeToMove = [&](Operation *op, Operation *mm, const SetVector<Operation *> &chain) {
        if (chainUseCount.lookup(op) != 1)
            return false;

        if (auto copy = dyn_cast<memref::CopyOp>(op)) {
            Operation *target = copy.getTarget().getDefiningOp();
            return target && chain.contains(target) && chainUseCount.lookup(target) == 1;
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
        llvm::sort(movable, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
        for (Operation *op : movable)
            op->moveBefore(mm);
    }
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
// Both veccores then do useful work (2x vector throughput), matching the target.
// ============================================================================
// Change the leading dimension of a [BLOCK_M, ...] tensor or memref to
// [BLOCK_M/2, ...], giving each vector core its ROW_SPLIT row band. Types whose
// leading dimension is not BLOCK_M pass through unchanged.
static Type retileRowHalve(Type t, int64_t Mfull)
{
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
            if (failed(getStridesAndOffset(mt, strides, offset)))
                return t;
            SmallVector<int64_t> ns(sh.begin(), sh.end());
            ns[0] = half;
            return MemRefType::get(ns, mt.getElementType(), mt.getLayout(), mt.getMemorySpace());
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
};

// Step 1: emit get_sub_block_idx at the top of the scope and return it as an
// index value (0 or 1 — which of the core's two veccores is executing).
static Value emitSubBlockIndex(scope::ScopeOp vecScope, Location loc)
{
    Block &block = vecScope.getBodyRegion().front();
    OpBuilder topB(&block, block.begin());
    auto sbid = topB.create<hivm::GetSubBlockIdxOp>(loc, topB.getI64Type());
    return topB.create<arith::IndexCastOp>(loc, topB.getIndexType(), sbid.getResult()).getResult();
}

// Step 2: detach the whole-tile V->C pack chain
// (truncf -> reshape -> transpose -> reshape -> to_memref -> cast -> copy),
// recording each P source / L1 alloc / insertion anchor so step 6 can rebuild a
// per-veccore pack. The truncf (the actual P value) is kept; the rest is erased.
static SmallVector<VectorToCubePack> detachVectorToCubePacks(scope::ScopeOp vecScope,
                                                             ArrayRef<VectorToCubeTransferChain> vectorToCubeChains)
{
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
static unsigned cloneExternalInitsAsHalfHeight(scope::ScopeOp vecScope, Location loc, int64_t Mfull)
{
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
                auto expectedType = RankedTensorType::get(expectedShape, rt.getElementType(), rt.getEncoding());
                assert(ntt == expectedType && "retile must only halve the BLOCK_M dimension");
                if (auto fill = dyn_cast<linalg::FillOp>(d)) {
                    originalFills.insert(d);
                    Value init = fill.getDpsInitOperand(0)->get();
                    if (auto empty = init.getDefiningOp<tensor::EmptyOp>())
                        originalEmpties.insert(empty);
                    Value ne = cb.create<tensor::EmptyOp>(loc, ntt.getShape(), ntt.getElementType(), ntt.getEncoding());
                    repl = cb.create<linalg::FillOp>(loc, fill.getInputs(), ValueRange {ne}).getResult(0);
                } else if (isa<tensor::EmptyOp>(d)) {
                    originalEmpties.insert(d);
                    repl = cb.create<tensor::EmptyOp>(loc, ntt.getShape(), ntt.getElementType(), ntt.getEncoding());
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
    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Removed " << removedCount << " dead original initializer(s)\n");

    return clonedCount;
}

// Steps 4 & 4.5: retile every in-scope op result (and loop-carried arg) from
// BLOCK_M to BLOCK_M/2 rows, then repair any DPS tensor.empty init whose type
// drifted from its retiled result. (arith.constant splats get their value attr
// rebuilt; reinterpret_cast is handled in steps 5/6 and skipped here.)
static LogicalResult retileVectorScopeOps(scope::ScopeOp vecScope, int64_t Mfull)
{
    // ---- 4. Generic re-tile of every vector op (skip reinterpret_cast) ----
    SmallVector<arith::ConstantOp> constants;
    vecScope.walk([&](arith::ConstantOp constant) { constants.push_back(constant); });

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
        c.setValueAttr(DenseElementsAttr::get(nst, dense.getSplatValue<Attribute>()));
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
                b->getArgument(i).setType(retileRowHalve(b->getArgument(i).getType(), Mfull));
        }
    });

    // Validate that the preceding retiling updated every tensor DPS init and its
    // tied result consistently. Do not silently repair a missed transformation.
    SmallVector<DestinationStyleOpInterface> dpsOps;
    vecScope.walk([&](DestinationStyleOpInterface dps) { dpsOps.push_back(dps); });
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
        for (auto [init, iterArg, result] :
             llvm::zip_equal(loop.getInitArgs(), loop.getRegionIterArgs(), loop.getResults())) {
            if (init.getType() == iterArg.getType() && init.getType() == result.getType())
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
static FailureOr<unsigned> retileOutputStores(scope::ScopeOp vecScope, Value sbidx, Location loc, int64_t Mfull)
{
    int64_t half = Mfull / 2;
    // ---- 5. Output stores: offset += sbid*half*leadingStride, sizes M=half ----
    SmallVector<bufferization::MaterializeInDestinationOp> mats;
    vecScope.walk([&](bufferization::MaterializeInDestinationOp m) { mats.push_back(m); });
    unsigned retiledCount = 0;
    for (auto m : mats) {
        auto ric = m.getDest().getDefiningOp<memref::ReinterpretCastOp>();
        if (!ric)
            continue;
        // Build the per-veccore reinterpret_cast right before the materialize (it is
        // inside the scope, so sub_block_idx dominates it). The original ric may be
        // a loop-invariant op hoisted into the parent block, where sbid is not in
        // scope.
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
        auto newRic = b.create<memref::ReinterpretCastOp>(loc, newType, ric.getSource(), getAsOpFoldResult(newOff),
                                                          sizes, strides);
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
static LogicalResult rebuildVectorToCubePacks(ArrayRef<VectorToCubePack> packs, Value sbidx,
                                              hivm::AddressSpaceAttr ubAddrSpace, Location loc)
{
    // ---- 6. Regenerate V->C packs: 16x32 -> 2x1x16x16 -> subview[0,sbid,0,0] ----
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
        if (!pType || pType.getRank() != 2 || !pType.hasStaticShape()) {
            emitError(loc, "V->C pack source must be a static rank-2 tensor");
            return failure();
        }
        int64_t M = pType.getShape()[0];
        int64_t N = pType.getShape()[1];
        int64_t N16 = N / kNzTileSize, M16 = M / kNzTileSize;
        Type elemType = pType.getElementType();
        Operation *pProducer = p.pSrc.getDefiningOp();
        if (!pProducer || pProducer->getBlock() != p.anchor->getBlock() || !pProducer->isBeforeInBlock(p.anchor)) {
            emitError(loc, "V->C pack source must be defined before its sync anchor in the vector scope");
            return failure();
        }

        // Begin the layout-only pack immediately after P is formed. This lets
        // the transpose overlap the independent denominator/alpha update in
        // the same VF, matching the hand-unrolled schedule. The final reshape,
        // UB view, copy, and signal remain at the phase-end anchor below.
        OpBuilder b(pProducer);
        b.setInsertionPointAfter(pProducer);
        auto l1AddrSpace = b.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
        auto expectedL1Type = MemRefType::get({N16, 2 * M16, kNzTileSize, kNzTileSize}, elemType, nullptr, l1AddrSpace);
        if (!p.l1Alloc || p.l1Alloc.getType() != expectedL1Type) {
            emitError(loc) << "V->C pack L1 destination must have type " << expectedL1Type;
            return failure();
        }
        auto i64Ty = b.getI64Type();
        // reshape [M,N] -> [M, N16, kNzTileSize]
        auto s3Type = RankedTensorType::get({3}, i64Ty);
        auto s3 = b.create<arith::ConstantOp>(loc, s3Type,
                                              DenseElementsAttr::get(s3Type, ArrayRef<int64_t> {M, N16, kNzTileSize}));
        auto resh1Type = RankedTensorType::get({M, N16, kNzTileSize}, elemType);
        auto resh1 = b.create<tensor::ReshapeOp>(loc, resh1Type, p.pSrc, s3.getResult());
        // transpose [M,N16,kNzTileSize] -> [N16,M,kNzTileSize]
        auto emptyT = b.create<tensor::EmptyOp>(loc, ArrayRef<int64_t> {N16, M, kNzTileSize}, elemType);
        auto transp =
            b.create<linalg::TransposeOp>(loc, resh1.getResult(), emptyT.getResult(), ArrayRef<int64_t> {1, 0, 2});

        b.setInsertionPoint(p.anchor);
        // reshape [N16,M,kNzTileSize] -> [N16,M16,kNzTileSize,kNzTileSize]
        auto s4Type = RankedTensorType::get({4}, i64Ty);
        auto s4 = b.create<arith::ConstantOp>(
            loc, s4Type, DenseElementsAttr::get(s4Type, ArrayRef<int64_t> {N16, M16, kNzTileSize, kNzTileSize}));
        auto nzType = RankedTensorType::get({N16, M16, kNzTileSize, kNzTileSize}, elemType);
        auto resh2 = b.create<tensor::ReshapeOp>(loc, nzType, transp->getResult(0), s4.getResult());
        // to_memref + cast to UB
        auto memT = MemRefType::get({N16, M16, kNzTileSize, kNzTileSize}, elemType);
        auto toMem = b.create<bufferization::ToMemrefOp>(loc, memT, resh2.getResult());
        auto ubMemT = MemRefType::get({N16, M16, kNzTileSize, kNzTileSize}, elemType, nullptr, ubAddrSpace);
        auto cast = b.create<memref::MemorySpaceCastOp>(loc, ubMemT, toMem.getResult());
        // subview of L1 alloc [0, sbid*M16, 0, 0] [N16,M16,16,16]: each veccore owns
        // M16 = (rows/veccore)/16 fractal-row blocks, so veccore `sbid` writes the
        // band starting at block sbid*M16 (NOT bare sbid — that only coincided for
        // BLOCK_M=32 where M16==1, and overlapped/misaligned for BLOCK_M>=64).
        Value m16c = b.create<arith::ConstantIndexOp>(loc, M16);
        Value off1 = b.create<arith::MulIOp>(loc, sbidx, m16c);
        SmallVector<OpFoldResult, 4> offs {b.getIndexAttr(0), off1, b.getIndexAttr(0), b.getIndexAttr(0)};
        SmallVector<OpFoldResult, 4> szs {b.getIndexAttr(N16), b.getIndexAttr(M16), b.getIndexAttr(kNzTileSize),
                                          b.getIndexAttr(kNzTileSize)};
        SmallVector<OpFoldResult, 4> strs {b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1)};
        auto subview = b.create<memref::SubViewOp>(loc, p.l1Alloc, offs, szs, strs);
        b.create<hivm::CopyOp>(loc, mlir::TypeRange {}, cast.getResult(), subview.getResult());
    }
    return success();
}

// Re-tile the VECTOR scope for ROW_SPLIT so both veccores do useful work (2x
// vector throughput): M/2 rows per veccore, addressed by get_sub_block_idx,
// matching the target IR. Runs the six steps in order; see each helper.
static LogicalResult retileVectorScopeForRowSplit(scope::ScopeOp vecScope, const CrossScopeTransferInfo &transferInfo)
{
    Location loc = vecScope.getLoc();
    MLIRContext *ctx = vecScope.getContext();
    auto ubAddrSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
    int64_t blockM = transferInfo.blockM;

    SmallVector<VectorToCubePack> packs = detachVectorToCubePacks(vecScope, transferInfo.vectorToCubeChains);

    Value sbidx = emitSubBlockIndex(vecScope, loc);
    unsigned clonedCount = cloneExternalInitsAsHalfHeight(vecScope, loc, blockM);
    if (failed(retileVectorScopeOps(vecScope, blockM)))
        return failure();
    FailureOr<unsigned> nStores = retileOutputStores(vecScope, sbidx, loc, blockM);
    if (failed(nStores))
        return failure();
    if (failed(rebuildVectorToCubePacks(packs, sbidx, ubAddrSpace, loc)))
        return failure();

    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   ROW_SPLIT re-tile (BLOCK_M=" << blockM << " -> " << (blockM / 2)
                            << "/veccore): " << packs.size() << " V->C packs, " << *nStores << " stores, "
                            << clonedCount << " ext consts cloned\n");
    return success();
}

} // namespace

LogicalResult createScopeSeparation(func::FuncOp funcOp, scf::ForOp innerLoop,
                                    const CrossScopeTransferInfo &transferInfo)
{

    MLIRContext *ctx = funcOp.getContext();
    Location loc = innerLoop.getLoc();

    // Strategy (matching reference FA kernel pattern):
    // - CUBE scope: clone of inner loop with only CUBE ops (before VECTOR scope)
    // - VECTOR scope: original inner loop + ALL subsequent ops in the parent block
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
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Non-VECTOR or unclassified epilogue op: " << op->getName() << "\n");
            return failure();
        }
    }

    // Step 1: Create CUBE scope (placed BEFORE the inner loop)
    builder.setInsertionPoint(innerLoop);
    auto cubeScope = builder.create<scope::ScopeOp>(loc, ArrayRef<Type> {});
    cubeScope.getBodyRegion().emplaceBlock();
    cubeScope->setAttr("noinline", UnitAttr::get(ctx));

    // Clone the inner loop into CUBE scope
    Block *cubeBlock = &cubeScope.getBodyRegion().front();
    OpBuilder cubeBuilder(cubeBlock, cubeBlock->end());
    IRMapping cubeMapping;
    auto cubeLoop = cast<scf::ForOp>(cubeBuilder.clone(*innerLoop.getOperation(), cubeMapping));
    cubeBuilder.create<scope::ReturnOp>(loc);

    // Step 2: Create VECTOR scope (wraps the original inner loop + epilogue)
    builder.setInsertionPoint(innerLoop);
    auto vecScope = builder.create<scope::ScopeOp>(loc, ArrayRef<Type> {});
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
    cubeScope->setAttr(hivm::TCoreTypeAttr::name, hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE));
    vecScope->setAttr(hivm::TCoreTypeAttr::name, hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR));

    // Step 4: Strip wrong-type operations using their stamped core attributes.
    // Existing operations retain their Stage-3 classification; later
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
                bool conversionIsLoopInvariant = input.getDefiningOp() && input.getDefiningOp()->getBlock() != loopBody;
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
    // the matmul. The BFS level scheduler (stage 7) groups every unrolled K/V
    // load at the top of the cube loop body, so all 8 cbuf staging buffers stay
    // simultaneously live. PlanMemory then assigns them high L1 offsets and the
    // matmul's FB operand load falls back to register offset (mode 2), which the
    // simulator's dmamov_decode_to_fb path rejects. Interleaving the loads (the
    // manual kernel allocates one K tile right before each matmul) keeps only the
    // in-flight operands live -> low static offsets -> immediate offset mode.
    sinkCubeLoadChainsToMatmul(cubeLoop.getBody());

    // Step 7: ROW_SPLIT re-tile of the VECTOR scope (BLOCK_M/2 rows per veccore,
    // both veccores active). Replaces the single-veccore NO_DUAL guard.
    if (failed(retileVectorScopeForRowSplit(vecScope, transferInfo)))
        return failure();

    LLVM_DEBUG(llvm::dbgs() << "[cv-split] Scope separation done: CUBE scope then VECTOR scope "
                            << "(inside parent block, matching reference pattern)\n");
    return success();
}

} // namespace mlir::triton::cv_split
