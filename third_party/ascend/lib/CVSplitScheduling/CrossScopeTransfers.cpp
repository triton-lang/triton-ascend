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

#include "ascend/include/CVSplitScheduling/CrossScopeTransfers.h"
#include "ascend/include/CVSplitScheduling/HardwareConstants.h"
#include "ascend/include/CVSplitScheduling/UnrollOrigin.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <map>

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"
namespace {

// ============================================================================
// Stage 8: Insert cross-scope transfers and synchronization
//
// For each SSA value that crosses from CUBE→VECTOR or VECTOR→CUBE:
//   C→V: alloc UB buffer, insert fixpipe + sync_block_set/wait, replace uses
//   V→C: alloc L1 buffer (NZ), insert copy + sync_block_set/wait, replace uses
//
// This runs BEFORE scope separation so both scopes see the shared buffers.
// ============================================================================

/// Flags and release duties for one transfer, resolved by the allocation in
/// `insertCrossScopeTransfers`.
struct TransferSyncPlan {
    /// Producer-to-consumer handoff: the consumer waits before reading.
    int forwardFlagId;
    /// Physical slot set this transfer draws from, and which slot in it. Two
    /// phases sharing a key share their buffers.
    int64_t slotGroupKey;
    unsigned slot;
    /// Storage type of the physical slot. Null means "whatever this transfer
    /// needs"; otherwise the slot is a union sized for the largest role in the
    /// group and this transfer takes a contiguous view of its front.
    MemRefType slotAllocType;
};

struct CrossScopeTransfer {
    Value value;
    Operation *producer;
    Operation *transferInsertionAnchor;
    Operation *waitInsertionAnchor;
    SmallVector<Operation *> consumers;
    enum Direction { CUBE_TO_VECTOR, VECTOR_TO_CUBE } direction;
    int64_t originId;
};

static FailureOr<int64_t> getUnrollOriginId(Operation *producer)
{
    auto originAttr = producer->getAttrOfType<IntegerAttr>(kUnrollOriginIdAttrName);
    if (!originAttr) {
        producer->emitError("cross-scope transfer producer is missing its "
                            "unroll-origin ID");
        return failure();
    }
    return originAttr.getInt();
}

static FailureOr<SmallVector<CrossScopeTransfer>>
findCrossScopeValues(Block *body, const DenseMap<Operation *, EngineType> &classification,
                     const DenseMap<Operation *, Operation *> &transferPhaseEnds)
{
    SmallVector<CrossScopeTransfer> transfers;

    for (Operation &op : *body) {
        if (isa<scf::YieldOp>(&op))
            continue;
        auto prodIt = classification.find(&op);
        if (prodIt == classification.end()) {
            op.emitError("body operation is missing a core classification");
            return failure();
        }
        EngineType prodType = prodIt->second;

        // C→V: Only transfer results of linalg.matmul (QK and PV dot products)
        // V→C: Only transfer values that DIRECTLY feed into linalg.matmul as operands,
        //      these are the P values after softmax+cast
        //
        // Reference pattern for K=4:
        //   4× QK matmul results (C→V, fixpipe, flags 0-3)
        //   4× P inputs to PV matmul (V→C, copy UB→L1, flags 4-7)
        //   4× PV matmul results (C→V, fixpipe, flags 8-11)

        if (prodType == EngineType::CUBE && isa<linalg::MatmulOp>(&op)) {
            // C→V: matmul result consumed by VECTOR ops
            for (Value result : op.getResults()) {
                if (!isa<RankedTensorType>(result.getType())) {
                    op.emitError("CUBE-to-VECTOR matmul result must be a ranked tensor");
                    return failure();
                }
                SmallVector<Operation *> crossUsers;
                for (Operation *user : result.getUsers()) {
                    if (user->getBlock() != body || isa<scf::YieldOp>(user))
                        continue;
                    auto consIt = classification.find(user);
                    if (consIt == classification.end()) {
                        user->emitError("body operation is missing a core classification");
                        return failure();
                    }
                    if (consIt->second == EngineType::VECTOR)
                        crossUsers.push_back(user);
                }
                if (!crossUsers.empty()) {
                    FailureOr<int64_t> originId = getUnrollOriginId(&op);
                    if (failed(originId))
                        return failure();
                    transfers.push_back(
                        {result, &op, &op, crossUsers.front(), crossUsers, CrossScopeTransfer::CUBE_TO_VECTOR,
                         *originId});
                }
            }
        } else if (prodType == EngineType::VECTOR) {
            // V→C: transfer VECTOR results that feed matmul input operands.
            // A VECTOR-produced DPS init should have been removed by unfusePVMatmuls.
            for (Value result : op.getResults()) {
                if (!isa<RankedTensorType>(result.getType()))
                    continue;
                SmallVector<Operation *> crossUsers;
                for (Operation *user : result.getUsers()) {
                    if (user->getBlock() != body || !isa<linalg::MatmulOp>(user))
                        continue;
                    bool feedsMatmulInput = user->getOperand(0) == result || user->getOperand(1) == result;
                    if (!feedsMatmulInput) {
                        user->emitError("VECTOR-produced matmul accumulator was not "
                                        "successfully unfused");
                        return failure();
                    }
                    crossUsers.push_back(user);
                }
                if (!crossUsers.empty()) {
                    FailureOr<int64_t> originId = getUnrollOriginId(&op);
                    if (failed(originId))
                        return failure();
                    Operation *transferAnchor = transferPhaseEnds.lookup(&op);
                    if (!transferAnchor) {
                        op.emitError("VECTOR-to-CUBE producer is missing its scheduled phase end");
                        return failure();
                    }
                    transfers.push_back(
                        {result, &op, transferAnchor, crossUsers.front(), crossUsers,
                         CrossScopeTransfer::VECTOR_TO_CUBE,
                         *originId});
                }
            }
        }
    }
    return transfers;
}

// Per-pass attribute bundle shared by the transfer emitters: the core-type and
// pipe attributes are constant for the whole pass, and `loop` is where the
// shared buffers are allocated (just before the inner loop).
struct TransferEmitContext {
    MLIRContext *ctx;
    Location loc;
    scf::ForOp loop;
    hivm::TCoreTypeAttr cubeCoreAttr;
    hivm::TCoreTypeAttr vecCoreAttr;
    hivm::PipeAttr pipeFixAttr;
    hivm::PipeAttr pipeVAttr;
    hivm::PipeAttr pipeMte3Attr;
    hivm::PipeAttr pipeMte1Attr;
};

// alloc + annotation.mark{effects=["write","read"]}. One shared buffer per
// transfer, like the reference FA kernel. (The target IR carries only
// `effects`; the hivm.tightly_coupled_buffer<N> attribute is intentionally
// omitted.)
static memref::AllocOp createAnnotatedAlloc(OpBuilder &builder, Location loc, MemRefType allocType)
{
    auto allocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markOp = builder.create<annotation::MarkOp>(loc, allocOp.getResult());
    auto writeAttr = builder.getStringAttr("write");
    auto readAttr = builder.getStringAttr("read");
    markOp->setAttr("effects", builder.getArrayAttr({writeAttr, readAttr}));
    return allocOp;
}

// Configured ping/pong buffer pool. Transfers that share an identical buffer type
// (e.g. all the unrolled qk_ub fixpipe targets, or all the P L1 packs) reuse a
// rotating set of `depth` physical allocations instead of one fresh buffer per
// unrolled stage. This mirrors the manual kernel's qk_ub_0/1, pv_ub_0/1,
// p_l1_0/1 double buffering and keeps peak UB/L1 bounded so unroll>=2 fits in
// the 248 KB UB. Reuse serializes stage i and stage i+depth on the same buffer
// (WAR), which BiShengIR's GraphSyncSolver covers via the existing
// sync_block_set/wait flags — exactly the depth-2 software pipeline the manual
// kernel uses.
struct BufferPool {
    // Physical slots keyed by *slot group*, not by origin. Two phases that share
    // a group share their buffers: lane i of each lands in slot i.
    DenseMap<int64_t, DenseMap<Type, SmallVector<memref::AllocOp, 4>>> slots;
    // Cached 2D ND view for each physical L1 buffer.
    llvm::DenseMap<Operation *, Value> ndView;

    // Allocation for `slot` of `groupKey`. Creates one only while the set is
    // still growing; slots must be requested in order. `builder`'s insertion
    // point must already be set (before the loop) for the create case.
    memref::AllocOp getOrCreate(OpBuilder &builder, Location loc, int64_t groupKey, unsigned slot,
                                MemRefType allocType)
    {
        auto &vec = slots[groupKey][allocType];
        if (slot < vec.size())
            return vec[slot];
        auto allocOp = createAnnotatedAlloc(builder, loc, allocType);
        vec.push_back(allocOp);
        return allocOp;
    }

    // Contiguous re-view of a union slot as one role's tile.
    //
    // A slot shared by roles of different sizes is allocated for the largest of
    // them; a smaller role occupies the front of it. The view must come out
    // *contiguous*, not strided: a vector function's parameters bufferize with
    // an identity layout map, so a genuinely strided window can never be cast
    // across that boundary, while an offset-0 identity-layout view casts freely.
    // Taking the whole front of the buffer rather than a sub-rectangle of it is
    // what keeps the view contiguous.
    Value getOrCreateSlotView(OpBuilder &builder, Location loc, memref::AllocOp slotAlloc, MemRefType viewType)
    {
        if (slotAlloc.getType() == viewType)
            return slotAlloc.getResult();

        Value &cached = slotViews[slotAlloc.getOperation()][viewType];
        if (cached)
            return cached;

        ArrayRef<int64_t> shape = viewType.getShape();
        SmallVector<OpFoldResult, 4> sizes, strides(shape.size());
        int64_t stride = 1;
        for (unsigned i = shape.size(); i-- > 0;) {
            strides[i] = builder.getIndexAttr(stride);
            stride *= shape[i];
        }
        for (int64_t dim : shape)
            sizes.push_back(builder.getIndexAttr(dim));

        auto viewOp = builder.create<memref::ReinterpretCastOp>(loc, viewType, slotAlloc.getResult(),
                                                                /*offset=*/builder.getIndexAttr(0), sizes, strides);
        cached = viewOp.getResult();
        return cached;
    }

    // Cached views per physical slot, keyed by the type the role needs.
    DenseMap<Operation *, DenseMap<Type, Value>> slotViews;
};

// The UB buffer a CUBE->VECTOR transfer lands in, without emitting anything.
// Mirrors `emitCubeToVectorTransfer` exactly so the slot policy can compare
// buffer types before the first allocation is created.
static MemRefType getCubeToVectorUbType(MLIRContext *ctx, RankedTensorType tensorType)
{
    OpBuilder builder(ctx);
    auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
    SmallVector<int64_t, 4> ubShape(tensorType.getShape().begin(), tensorType.getShape().end());
    ubShape[0] /= 2; // ROW_SPLIT writes an M/2-row shard to each vector core.
    return MemRefType::get(ubShape, tensorType.getElementType(), nullptr, ubAddrSpace);
}

// Physical buffer this transfer lands in, without emitting anything. Mirrors the
// allocation types the two emitters build, so the slot policy can be decided
// before the first buffer is created.
struct TransferFootprint {
    uint64_t bytes;
    bool isUB;
};

static TransferFootprint getTransferFootprint(const CrossScopeTransfer &xfer, RankedTensorType tensorType)
{
    ArrayRef<int64_t> shape = tensorType.getShape();
    uint64_t elements = 1;
    for (int64_t dim : shape)
        elements *= static_cast<uint64_t>(dim);

    // CUBE->VECTOR lands in UB, and ROW_SPLIT gives each vector core an
    // M/2-row shard. VECTOR->CUBE lands in L1; the NZ fractal reshapes the
    // tile but does not change its element count.
    const bool isUB = xfer.direction == CrossScopeTransfer::CUBE_TO_VECTOR;
    if (isUB)
        elements /= 2;

    const uint64_t bits = elements * tensorType.getElementTypeBitWidth();
    return TransferFootprint {/*bytes=*/(bits + 7) / 8, isUB};
}

// Return one 2D ND view per physical L1 buffer. The MTE UB->L1 copy performs
// the hardware-specific blocked traversal; convert_layout does not move data
// and only exposes the resulting L1 storage as an [M, N] memref.
//
// The view is cached because the same ping/pong buffer is reused by multiple
// unrolled stages. Emitting a convert_layout for every stage creates aliasing
// ND views that BiShengIR can mis-track into a misaligned / zero-burst L1->L0
// load. It is inserted at the start of the loop so it dominates every reuse.
// Keeping it inside the loop is also required by SplitMixKernel, which cannot
// obtain out-operands for a scope-level convert_layout.
static Value getOrCreateL1NdView(const TransferEmitContext &c, memref::AllocOp sharedL1AllocOp,
                                 RankedTensorType tensorType, BufferPool &bufferPool)
{
    Operation *l1Key = sharedL1AllocOp.getOperation();
    if (Value ndView = bufferPool.ndView.lookup(l1Key))
        return ndView;

    OpBuilder builder(c.ctx);
    scf::ForOp loop = c.loop;
    builder.setInsertionPointToStart(loop.getBody());

    ArrayRef<int64_t> shape = tensorType.getShape();
    Type elemType = tensorType.getElementType();
    auto l1AddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
    auto ndLayout = hivm::DataLayoutAttr::get(c.ctx, hivm::DataLayout::ND);
    auto ndL1Type = MemRefType::get(shape, elemType, nullptr, l1AddrSpace);
    auto convertOp =
        builder.create<hivm::ConvertLayoutOp>(c.loc, ndL1Type, sharedL1AllocOp.getResult(), ndLayout, ndLayout,
                                              DenseI64ArrayAttr::get(c.ctx, shape), ValueRange {});
    setOpEngineTypeAttr(convertOp, EngineType::CUBE);

    auto plainMemrefType = MemRefType::get(shape, elemType);
    auto castOp = builder.create<memref::MemorySpaceCastOp>(c.loc, plainMemrefType, convertOp.getResult());
    setOpEngineTypeAttr(castOp, EngineType::CUBE);
    Value ndView = castOp.getResult();
    bufferPool.ndView[l1Key] = ndView;
    return ndView;
}

// CUBE -> VECTOR: the matmul (L0C) result is fixpipe'd to a shared UB buffer,
// CUBE signals via sync_block_set, VECTOR waits and reads it back as a tensor.
// ROW_SPLIT: for an M-row result, the UB buffer has M/2 rows and fixpipe sends
// one half to each vector core's private UB. The VECTOR scope is re-tiled to
// M/2 rows per vector core in a later stage.

/// A VECTOR-side release whose placement is resolved after every transfer has
/// been emitted, because the synchronization it should sit next to may belong
/// to a later phase that does not exist yet while this one is being built.

static void emitCubeToVectorTransfer(const TransferEmitContext &c, CrossScopeTransfer &xfer,
                                     RankedTensorType tensorType, const TransferSyncPlan &plan,
                                     BufferPool &bufferPool)
{
    const int flagId = plan.forwardFlagId;
    Type elemType = tensorType.getElementType();
    ArrayRef<int64_t> shape = tensorType.getShape();

    OpBuilder builder(c.ctx);

    auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
    SmallVector<int64_t, 4> ubShape(shape.begin(), shape.end());
    ubShape[0] /= 2; // ROW_SPLIT writes M/2-row shard to each vector core's UB.
    auto allocType = MemRefType::get(ubShape, elemType, nullptr, ubAddrSpace);
    auto halfTensorType = RankedTensorType::get(ubShape, elemType);

    // Ping/pong shared alloc before the loop (depth-2 reuse across unrolled
    // clones of this original transfer operation).
    builder.setInsertionPoint(c.loop);
    MemRefType slotType = plan.slotAllocType ? plan.slotAllocType : allocType;
    auto sharedAllocOp = bufferPool.getOrCreate(builder, c.loc, plan.slotGroupKey, plan.slot, slotType);
    Value slotBuffer = bufferPool.getOrCreateSlotView(builder, c.loc, sharedAllocOp, allocType);

    // fixpipe after the producer (inside loop body) -> writes the shared buffer.
    // Keep all same-phase VECTOR state maintenance (for example softmax's
    // denominator reduction) in one fusible region.  Inserting the pack and
    // sync immediately after the P producer splits that region into two VFs
    // and keeps its full score tile live across the synchronization boundary.
    builder.setInsertionPointAfter(xfer.producer);


    auto dmaModeAttr = hivm::FixpipeDMAModeAttr::get(c.ctx, hivm::FixpipeDMAMode::NZ2ND);
    auto dualDstAttr = hivm::FixpipeDualDstModeAttr::get(c.ctx, hivm::FixpipeDualDstMode::ROW_SPLIT);
    auto fixpipeOp = builder.create<hivm::FixpipeOp>(c.loc, mlir::TypeRange {},
                                                     xfer.value,                // src (full M-row tile from matmul)
                                                     slotBuffer,                // dst (M/2-row shared UB slot)
                                                     mlir::ValueRange {}, dmaModeAttr, dualDstAttr, nullptr, nullptr,
                                                     nullptr, mlir::ArrayAttr {}, nullptr);
    setOpEngineTypeAttr(fixpipeOp, EngineType::CUBE);

    // CUBE signals VECTOR.
    auto syncSetOp = builder.create<hivm::SyncBlockSetOp>(c.loc, c.cubeCoreAttr, c.pipeFixAttr, c.pipeVAttr,
                                                          OpFoldResult(builder.getI64IntegerAttr(flagId)));
    setOpEngineTypeAttr(syncSetOp, EngineType::CUBE);

    // Consumer side: wait + read the shared buffer back as an M/2-row tensor.
    // The precomputed anchor includes the independent prefix of the consumer's
    // VECTOR dependency chain so that prefix and the transfer consumer remain
    // in one vector function.
    builder.setInsertionPoint(xfer.waitInsertionAnchor);

    auto syncWaitOp = builder.create<hivm::SyncBlockWaitOp>(c.loc, c.vecCoreAttr, c.pipeFixAttr, c.pipeVAttr,
                                                            OpFoldResult(builder.getI64IntegerAttr(flagId)));
    setOpEngineTypeAttr(syncWaitOp, EngineType::VECTOR);

    auto plainMemrefType = MemRefType::get(ubShape, elemType);
    auto castOp = builder.create<memref::MemorySpaceCastOp>(c.loc, plainMemrefType, slotBuffer);
    setOpEngineTypeAttr(castOp, EngineType::VECTOR);
    auto toTensorOp = builder.create<bufferization::ToTensorOp>(c.loc, halfTensorType, castOp.getResult(),
                                                                /*restrict=*/true, /*writable=*/true);
    setOpEngineTypeAttr(toTensorOp, EngineType::VECTOR);

    for (auto *consumer : xfer.consumers)
        consumer->replaceUsesOfWith(xfer.value, toTensorOp.getResult());


    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   C→V transfer #" << flagId << ": " << xfer.producer->getName() << " → " << ubShape[0] << "x" << ubShape[1]
                            << " UB buffer (ROW_SPLIT)\n");
}

// VECTOR -> CUBE: a softmax/cast result is NZ-packed and copied UB->L1 into a
// shared L1 buffer, VECTOR signals via sync_block_set, CUBE waits and reads it
// back through a convert_layout (NZ fractal -> ND view) for matmul consumption.
// NZ packing applies only when both dims are multiples of 16; otherwise the L1
// buffer keeps the flat [M, N] layout.
static VectorToCubeTransferChain emitVectorToCubeTransfer(const TransferEmitContext &c, CrossScopeTransfer &xfer,
                                                          RankedTensorType tensorType, const TransferSyncPlan &plan,
                                                          BufferPool &bufferPool)
{
    const int flagId = plan.forwardFlagId;
    Type elemType = tensorType.getElementType();
    ArrayRef<int64_t> shape = tensorType.getShape();

    OpBuilder builder(c.ctx);

    int64_t M = shape[0];
    int64_t N = shape[1];
    auto l1AddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);

    // NZ-fractal L1 layout: ND [M, N] is stored as [N/16, M/16, 16, 16] (B16
    // fractal) when both dims are multiples of 16; otherwise fall back to flat.
    bool useNZ = (M % kNzTileSize == 0) && (N % kNzTileSize == 0);
    int64_t N16 = N / kNzTileSize, M16 = M / kNzTileSize;
    SmallVector<int64_t, 4> l1Shape =
        useNZ ? SmallVector<int64_t, 4> {N16, M16, kNzTileSize, kNzTileSize} : SmallVector<int64_t, 4> {M, N};
    auto l1AllocType = MemRefType::get(l1Shape, elemType, nullptr, l1AddrSpace);

    // Ping/pong shared L1 alloc before the loop (depth-2 reuse across unrolled
    // clones of this original transfer operation).
    builder.setInsertionPoint(c.loop);
    auto sharedL1AllocOp = bufferPool.getOrCreate(builder, c.loc, plan.slotGroupKey, plan.slot, l1AllocType);

    // Start the layout-only part of the NZ pack as soon as P is available. The
    // hand-written schedule overlaps this work with the independent recurrent
    // softmax-state update that follows P in the same vector function.
    builder.setInsertionPointAfter(xfer.producer);
    auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
    Value packedTensor = xfer.value;
    SmallVector<Operation *> packingOps;
    SmallVector<int64_t, 4> srcShape =
        useNZ ? SmallVector<int64_t, 4> {N16, M16, kNzTileSize, kNzTileSize} : SmallVector<int64_t, 4> {M, N};

    if (useNZ) {
        // ND [M,N] -> NZ [N/16, M/16, 16, 16] via reshape -> transpose -> reshape.
        auto i64Ty = builder.getI64Type();
        auto s3Type = RankedTensorType::get({3}, i64Ty);
        auto s3Const = builder.create<arith::ConstantOp>(
            c.loc, s3Type, DenseElementsAttr::get(s3Type, ArrayRef<int64_t> {M, N16, kNzTileSize}));
        setOpEngineTypeAttr(s3Const, EngineType::VECTOR);
        auto resh1Type = RankedTensorType::get({M, N16, kNzTileSize}, elemType);
        auto resh1 = builder.create<tensor::ReshapeOp>(c.loc, resh1Type, xfer.value, s3Const.getResult());
        packingOps.push_back(resh1);
        setOpEngineTypeAttr(resh1, EngineType::VECTOR);
        auto emptyT = builder.create<tensor::EmptyOp>(c.loc, ArrayRef<int64_t> {N16, M, kNzTileSize}, elemType);
        setOpEngineTypeAttr(emptyT, EngineType::VECTOR);
        auto transp = builder.create<linalg::TransposeOp>(c.loc, resh1.getResult(), emptyT.getResult(),
                                                          ArrayRef<int64_t> {1, 0, 2});
        packingOps.push_back(transp);
        setOpEngineTypeAttr(transp, EngineType::VECTOR);
        packedTensor = transp->getResult(0);
    }

    // Commit the UB->L1 handoff only after every operation in this producer
    // phase has executed. This keeps P packing, denominator/alpha maintenance,
    // and the copy in one VF instead of splitting the state update behind a
    // synchronization boundary.
    Operation *lateAnchor = xfer.transferInsertionAnchor;
    if (lateAnchor == xfer.producer && !packingOps.empty())
        lateAnchor = packingOps.back();
    builder.setInsertionPointAfter(lateAnchor);

    if (useNZ) {
        auto i64Ty = builder.getI64Type();
        auto s4Type = RankedTensorType::get({4}, i64Ty);
        auto s4Const = builder.create<arith::ConstantOp>(
            c.loc, s4Type, DenseElementsAttr::get(s4Type, ArrayRef<int64_t> {N16, M16, kNzTileSize, kNzTileSize}));
        setOpEngineTypeAttr(s4Const, EngineType::VECTOR);
        auto nzTensorType = RankedTensorType::get({N16, M16, kNzTileSize, kNzTileSize}, elemType);
        auto resh2 = builder.create<tensor::ReshapeOp>(c.loc, nzTensorType, packedTensor, s4Const.getResult());
        packingOps.push_back(resh2);
        setOpEngineTypeAttr(resh2, EngineType::VECTOR);
        packedTensor = resh2.getResult();
    }

    auto srcMemrefType = MemRefType::get(srcShape, elemType);
    auto toMemrefOp = builder.create<bufferization::ToMemrefOp>(c.loc, srcMemrefType, packedTensor);
    packingOps.push_back(toMemrefOp);
    setOpEngineTypeAttr(toMemrefOp, EngineType::VECTOR);
    auto ubMemrefType = MemRefType::get(srcShape, elemType, nullptr, ubAddrSpace);
    auto ubCastOp = builder.create<memref::MemorySpaceCastOp>(c.loc, ubMemrefType, toMemrefOp.getResult());
    packingOps.push_back(ubCastOp);
    setOpEngineTypeAttr(ubCastOp, EngineType::VECTOR);

    // UB -> L1 copy (same NZ/flat shape on both sides).
    auto copyOp = builder.create<hivm::CopyOp>(c.loc, mlir::TypeRange {},
                                               ubCastOp.getResult(),         // src (UB memref)
                                               sharedL1AllocOp.getResult()); // dst (shared L1 memref)
    packingOps.push_back(copyOp);
    setOpEngineTypeAttr(copyOp, EngineType::VECTOR);

    // VECTOR signals CUBE.
    auto syncSetOp = builder.create<hivm::SyncBlockSetOp>(c.loc, c.vecCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr,
                                                          OpFoldResult(builder.getI64IntegerAttr(flagId)));
    setOpEngineTypeAttr(syncSetOp, EngineType::VECTOR);

    // Consumer (CUBE) side: wait, then expose the MTE-populated L1 buffer as a
    // 2D ND view for matmul.
    Operation *firstConsumer = xfer.consumers.front();
    for (auto *cons : xfer.consumers)
        if (cons->isBeforeInBlock(firstConsumer))
            firstConsumer = cons;
    builder.setInsertionPoint(firstConsumer);

    auto syncWaitOp = builder.create<hivm::SyncBlockWaitOp>(c.loc, c.cubeCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr,
                                                            OpFoldResult(builder.getI64IntegerAttr(flagId)));
    setOpEngineTypeAttr(syncWaitOp, EngineType::CUBE);

    Value ndViewVal = getOrCreateL1NdView(c, sharedL1AllocOp, tensorType, bufferPool);

    // Fresh to_tensor per consumer group (after the wait), like the manual.
    auto toTensorOp = builder.create<bufferization::ToTensorOp>(c.loc, tensorType, ndViewVal, true, true);
    setOpEngineTypeAttr(toTensorOp, EngineType::CUBE);

    for (auto *consumer : xfer.consumers)
        consumer->replaceUsesOfWith(xfer.value, toTensorOp.getResult());


    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   V→C transfer #" << flagId << ": " << xfer.producer->getName() << " → " << M << "x" << N << " L1 buffer\n");

    return VectorToCubeTransferChain {xfer.value, sharedL1AllocOp.getResult(), syncSetOp, std::move(packingOps)};
}

// Close the loop back-edge for a merged slot group.
//
// Inside one iteration a merged slot is safe without extra ordering: the score
// tile is read before the reverse-direction handoff that this lane's product
// matmul waits on, so CUBE cannot overwrite the slot early. Across the back edge
// that argument does not hold. The slot's last reader is the VECTOR consumer of
// the *product* tile, which runs after the final handoff CUBE waits for, so the
// next iteration's score fixpipe can overtake it.
//
// One flag closes it for the whole group. VECTOR is in-order, so its last read
// of any slot in the group precedes the set, and CUBE's first write of any slot
// in the next iteration follows the wait.
//
// The wait sits at the *end* of CUBE's body rather than before the first write
// it protects. A wait there would be the first operation of the CUBE scope, and
// that structure hangs on this hardware -- priming it with a set before the loop
// does not rescue it. At the end of the body iteration 0 reaches the wait only
// after issuing all its work, by which point VECTOR has long since set the flag.
// The cost is that CUBE starts the next iteration one consumer later than it
// strictly must; the alternative does not run.
static void emitMergedSlotRelease(const TransferEmitContext &c, ArrayRef<CrossScopeTransfer> transfers,
                                  const DenseMap<int64_t, int64_t> &slotGroupOfOrigin, int64_t groupKey,
                                  int flagId)
{
    Operation *lastConsumer = nullptr;
    for (const CrossScopeTransfer &xfer : transfers) {
        if (xfer.direction != CrossScopeTransfer::CUBE_TO_VECTOR)
            continue;
        auto it = slotGroupOfOrigin.find(xfer.originId);
        if (it == slotGroupOfOrigin.end() || it->second != groupKey)
            continue;
        for (Operation *consumer : xfer.consumers)
            if (!lastConsumer || lastConsumer->isBeforeInBlock(consumer))
                lastConsumer = consumer;
    }
    if (!lastConsumer)
        return;

    OpBuilder builder(c.ctx);

    builder.setInsertionPointAfter(lastConsumer);
    auto syncSetOp = builder.create<hivm::SyncBlockSetOp>(c.loc, c.vecCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr,
                                                          OpFoldResult(builder.getI64IntegerAttr(flagId)));
    setOpEngineTypeAttr(syncSetOp, EngineType::VECTOR);

    scf::ForOp loop = c.loop;
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    auto syncWaitOp = builder.create<hivm::SyncBlockWaitOp>(c.loc, c.cubeCoreAttr, c.pipeMte3Attr, c.pipeMte1Attr,
                                                            OpFoldResult(builder.getI64IntegerAttr(flagId)));
    setOpEngineTypeAttr(syncWaitOp, EngineType::CUBE);

    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   Back-edge release #" << flagId << " for slot group " << groupKey
                            << ": VECTOR sets after its last read, CUBE waits at the end of the body\n");
}

} // namespace

FailureOr<CrossScopeTransferInfo>
insertCrossScopeTransfers(scf::ForOp loop, const DenseMap<Operation *, EngineType> &classification,
                          const DenseMap<Operation *, Operation *> &transferPhaseEnds,
                          unsigned interCoreBufferDepth, uint64_t privateBufferUbBudgetBytes,
                          bool promotePrivateBufferPools, unsigned vectorToCubeSlotOverride)
{

    MLIRContext *ctx = loop.getContext();
    Location loc = loop.getLoc();
    Block *body = loop.getBody();

    FailureOr<SmallVector<CrossScopeTransfer>> transferResult =
        findCrossScopeValues(body, classification, transferPhaseEnds);
    if (failed(transferResult))
        return failure();
    SmallVector<CrossScopeTransfer> transfers = std::move(*transferResult);
    if (transfers.empty()) {
        loop.emitError() << "CVSplitScheduling requires at least one CUBE-to-VECTOR "
                            "transfer to determine BLOCK_M";
        return failure();
    }

    // Flag demand is checked once the transfers are grouped into phases, below.
    // It depends on the number of phases and the buffer depth, not on the
    // transfer count, because a phase's flags are reused across its lanes.

    // DependencyScheduler completes sibling VECTOR state before each V->C
    // boundary producer.  The producer is therefore both the actual insertion
    // point of the P copy/signal and the authoritative end of that phase.
    // Keeping this boundary exact is also required when placing the following
    // C->V wait; extending it through later VECTOR work would describe an order
    // different from the IR that emitVectorToCubeTransfer actually creates.

    // A CUBE->VECTOR result may join a VECTOR chain whose other input is
    // already computable.  Waiting immediately before the join lets that
    // independent prefix become a separate vector function.  For attention,
    // this turns `acc * alpha + PV` into one VF for `acc * alpha` and another
    // VF for the add, whereas the hand-written unroll emits one fused VF after
    // waiting for PV.
    //
    // Move the wait to the earliest VECTOR predecessor of the first consumer,
    // but never across the preceding VECTOR->CUBE handoff.  That handoff is
    // the phase boundary: moving a PV wait before it would serialize softmax
    // with the prior CUBE work.  The walk is purely SSA/dependency based and
    // therefore also covers other join patterns without recognizing attention.
    for (CrossScopeTransfer &xfer : transfers) {
        if (xfer.direction != CrossScopeTransfer::CUBE_TO_VECTOR)
            continue;

        Operation *firstConsumer = xfer.consumers.front();
        for (Operation *consumer : xfer.consumers)
            if (consumer->isBeforeInBlock(firstConsumer))
                firstConsumer = consumer;

        Operation *lowerBoundary = nullptr;
        for (const CrossScopeTransfer &candidate : transfers) {
            if (candidate.direction == CrossScopeTransfer::VECTOR_TO_CUBE) {
                if (!candidate.transferInsertionAnchor->isBeforeInBlock(firstConsumer))
                    continue;
                if (!lowerBoundary || lowerBoundary->isBeforeInBlock(candidate.transferInsertionAnchor))
                    lowerBoundary = candidate.transferInsertionAnchor;
                continue;
            }

            // A preceding C->V consumer completes the prior join phase.  Do
            // not walk through it when placing the next wait, otherwise all
            // waits in an accumulator chain collapse at the start of the
            // chain and serialize CUBE and VECTOR instead of alternating.
            for (Operation *candidateConsumer : candidate.consumers) {
                if (candidateConsumer == firstConsumer || !candidateConsumer->isBeforeInBlock(firstConsumer))
                    continue;
                if (!lowerBoundary || lowerBoundary->isBeforeInBlock(candidateConsumer))
                    lowerBoundary = candidateConsumer;
            }
        }

        // The scheduler has already formed phase-contiguous VECTOR regions.
        // Put the wait before the first VECTOR operation in this region.  The
        // lower boundary is either the preceding P handoff or the preceding
        // C->V join, so this retains QK/P/PV overlap while preventing a prefix
        // of the current join from being outlined as a standalone VF.
        Operation *waitAnchor = lowerBoundary ? lowerBoundary->getNextNode() : &body->front();
        if (!waitAnchor || firstConsumer->isBeforeInBlock(waitAnchor))
            waitAnchor = firstConsumer;
        xfer.waitInsertionAnchor = waitAnchor;
    }

    // Allocate shared scalar-buffer IDs in the same three phases as the
    // reference schedule: QK C->V, P V->C, then PV C->V.  At this point all QK
    // producers precede the final P producer, while PV producers follow it.
    Operation *lastVectorToCubeProducer = nullptr;
    for (const CrossScopeTransfer &xfer : transfers)
        if (xfer.direction == CrossScopeTransfer::VECTOR_TO_CUBE)
            lastVectorToCubeProducer = xfer.producer;
    llvm::stable_sort(transfers, [&](const CrossScopeTransfer &lhs, const CrossScopeTransfer &rhs) {
        auto phase = [&](const CrossScopeTransfer &xfer) {
            if (xfer.direction == CrossScopeTransfer::VECTOR_TO_CUBE)
                return 1;
            return lastVectorToCubeProducer && xfer.producer->isBeforeInBlock(lastVectorToCubeProducer) ? 0 : 2;
        };
        return phase(lhs) < phase(rhs);
    });

    // ROW_SPLIT is materialized by the C->V fixpipes below. Their source is the
    // full CUBE result [BLOCK_M, ...], while their destination UB allocation is
    // [BLOCK_M/2, ...]. Derive BLOCK_M from that semantic boundary and require
    // every C->V transfer to agree before mutating the IR.
    std::optional<int64_t> blockM;
    for (const CrossScopeTransfer &xfer : transfers) {
        if (xfer.direction != CrossScopeTransfer::CUBE_TO_VECTOR)
            continue;

        auto tensorType = dyn_cast<RankedTensorType>(xfer.value.getType());
        if (!tensorType || tensorType.getRank() != 2 || tensorType.isDynamicDim(0)) {
            loop.emitError() << "CVSplitScheduling requires each CUBE-to-VECTOR "
                                "transfer to have a static rank-2 tensor type";
            return failure();
        }

        int64_t candidateBlockM = tensorType.getDimSize(0);
        if (blockM && *blockM != candidateBlockM) {
            loop.emitError() << "CVSplitScheduling found inconsistent BLOCK_M "
                             << "values across CUBE-to-VECTOR transfers: " << *blockM << " and " << candidateBlockM;
            return failure();
        }
        blockM = candidateBlockM;
    }

    if (!blockM) {
        loop.emitError() << "CVSplitScheduling requires at least one CUBE-to-VECTOR "
                            "transfer to determine BLOCK_M";
        return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[cv-split] Found " << transfers.size() << " cross-scope value transfers\n");

    const TransferEmitContext ec {ctx,
                                  loc,
                                  loop,
                                  hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE),
                                  hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR),
                                  hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX),
                                  hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_V),
                                  hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE3),
                                  hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE1)};

    // Flag allocation.
    //
    // Every unrolled clone of one original transfer forms a phase, and the
    // clones of a phase rotate over `interCoreBufferDepth` physical buffers.
    // Two lanes that share a buffer are already forced apart by that buffer, so
    // they can share flags too: a phase needs one flag per buffer slot rather
    // than one per lane, which makes the allocation independent of the unroll
    // factor. `BufferPool` assigns slots per origin ID in the same order this
    // loop visits them, so the slot is the visit ordinal modulo the depth.
    //
    // Each slot takes two flags. The forward flag is the existing producer to
    // consumer handoff, ordering the consumer's read after the producer's write.
    // The release flag runs the other way, ordering the next lane's write to the
    // slot after the previous lane's read of it. Without it the producing core
    // may overwrite a buffer that the consuming core is still reading; the two
    // cores run concurrently and nothing else orders that pair.
    //
    // The first lane of a slot does not wait, and the last does not signal: the
    // release only has to order lanes within one iteration. Across iterations
    // the ordering already exists transitively, because the producing core must
    // wait for the reverse-direction handoff of the phase that follows, and the
    // consumer signals that only after reading this slot.
    //
    // A pool can also opt out of reuse entirely. Given one slot per lane there
    // is no write-after-read pair left to order inside an iteration, so the
    // release flag disappears and the producing core never stalls on the
    // consumer mid-iteration. That costs buffers -- `laneCount` slots instead of
    // `depth` -- which only some pools can afford, so the choice is made per
    // pool rather than for the whole schedule.
    DenseMap<int64_t, unsigned> laneCountByOrigin;
    DenseMap<int64_t, unsigned> phaseIndexByOrigin;
    DenseMap<int64_t, TransferFootprint> footprintByOrigin;
    for (const CrossScopeTransfer &xfer : transfers) {
        ++laneCountByOrigin[xfer.originId];
        phaseIndexByOrigin.try_emplace(xfer.originId, phaseIndexByOrigin.size());
        auto tensorType = dyn_cast<RankedTensorType>(xfer.value.getType());
        if (tensorType && tensorType.getRank() == 2)
            footprintByOrigin.try_emplace(xfer.originId, getTransferFootprint(xfer, tensorType));
    }

    const unsigned phaseCount = phaseIndexByOrigin.size();

    // Order the candidates by what promoting them costs, cheapest first, so a
    // budget that cannot cover every pool still buys the ones it can. L1 pools
    // cost nothing against the UB budget and therefore always sort first --
    // matching the hand-written kernel, where the P pool is in L1 and its extra
    // slots are free at any tile size.
    SmallVector<int64_t> candidates;
    for (const auto &entry : phaseIndexByOrigin)
        candidates.push_back(entry.first);
    llvm::sort(candidates, [&](int64_t a, int64_t b) {
        auto cost = [&](int64_t origin) -> uint64_t {
            auto it = footprintByOrigin.find(origin);
            if (it == footprintByOrigin.end() || !it->second.isUB)
                return 0;
            const unsigned lanes = laneCountByOrigin[origin];
            if (lanes <= interCoreBufferDepth)
                return 0;
            return static_cast<uint64_t>(lanes - interCoreBufferDepth) * it->second.bytes;
        };
        const uint64_t ca = cost(a), cb = cost(b);
        return ca != cb ? ca < cb : a < b;
    });

    DenseMap<int64_t, unsigned> slotCountByOrigin;
    for (int64_t origin : candidates)
        slotCountByOrigin[origin] = interCoreBufferDepth;

    // The schedule pipelines a VECTOR->CUBE boundary's consuming work against
    // the boundary that reuses its slot, so the reorder distance and this
    // pool's slot count are the same number seen from two stages. The caller
    // decides it once, before scheduling, and hands it down here; deciding it
    // twice would let the two disagree, and a distance larger than the rotation
    // period puts a slot's read after the write that reuses it.
    if (vectorToCubeSlotOverride > 0)
        for (int64_t origin : candidates) {
            auto footprint = footprintByOrigin.find(origin);
            if (footprint != footprintByOrigin.end() && !footprint->second.isUB)
                slotCountByOrigin[origin] = vectorToCubeSlotOverride;
        }

    // Role sharing: one slot per lane, shared between same-typed CUBE->VECTOR
    // phases.
    //
    // The QK score tile and the PV product both travel CUBE->VECTOR, and when
    // HEAD_DIM equals BLOCK_N they land in identically shaped UB buffers. A
    // lane's score tile is already dead when its PV tile is written: the
    // schedule places this phase's reverse-direction wait between the two, and
    // the consumer only signals that after reading the score tile. So one
    // buffer can serve both roles for lane i, ordered by a flag the schedule
    // already emits.
    //
    // Giving every lane its own slot is what makes that safe. The rotating set
    // is what forces lane i and lane i+depth onto one buffer, and it is exactly
    // that pair the schedule does *not* order.
    //
    // Cost is unchanged whenever `laneCount <= originsInGroup * depth`: at
    // unroll four with two same-typed phases at depth two, four shared slots
    // replace two pools of two. Wider unrolls fail that test and keep the
    // rotating set.
    // Roles of unequal size still share, because the slot is sized for the
    // largest of them and the smaller takes a contiguous view of its front.
    // That is what lets HEAD_DIM differ from BLOCK_N: the score tile is
    // [BLOCK_M/2, BLOCK_N] and the product tile [BLOCK_M/2, HEAD_DIM], so at
    // HEAD_DIM < BLOCK_N the product is a strict prefix of the score buffer.
    // Grouping is therefore keyed by element type, not by the whole type.
    uint64_t ubBudgetLeft = privateBufferUbBudgetBytes;
    // Flags the schedule owes as it stands: one forward flag per slot. Merging
    // and promotion both add to this, and both are capped by it.
    unsigned flagsUsed = 0;
    for (int64_t origin : candidates)
        flagsUsed += slotCountByOrigin[origin];

    DenseMap<int64_t, int64_t> slotGroupOfOrigin;
    for (int64_t origin : candidates)
        slotGroupOfOrigin[origin] = origin; // its own group by default
    DenseMap<int64_t, MemRefType> unionTypeOfOrigin;

    DenseMap<int64_t, MemRefType> ubTypeByOrigin;
    DenseMap<Type, SmallVector<int64_t>> cubeToVectorOriginsByElemType;
    for (const CrossScopeTransfer &xfer : transfers) {
        if (xfer.direction != CrossScopeTransfer::CUBE_TO_VECTOR)
            continue;
        auto tensorType = dyn_cast<RankedTensorType>(xfer.value.getType());
        if (!tensorType || tensorType.getRank() != 2)
            continue;
        ubTypeByOrigin.try_emplace(xfer.originId, getCubeToVectorUbType(ctx, tensorType));
        auto &vec = cubeToVectorOriginsByElemType[tensorType.getElementType()];
        if (!llvm::is_contained(vec, xfer.originId))
            vec.push_back(xfer.originId);
    }

    SmallVector<int64_t> mergedGroupKeys;
    for (auto &entry : cubeToVectorOriginsByElemType) {
        SmallVector<int64_t> &group = entry.second;
        if (group.size() < 2)
            continue;

        // One slot per lane only makes sense if every role in the group has the
        // same number of lanes; otherwise "lane i" does not name one buffer.
        const unsigned lanes = laneCountByOrigin[group.front()];
        if (lanes == 0 ||
            llvm::any_of(group, [&](int64_t origin) { return laneCountByOrigin[origin] != lanes; }))
            continue;

        // Cost of the rotating pools this would replace, against the cost of
        // one union slot per lane. Equal-sized roles break even exactly; a
        // smaller second role makes the union cost more than the pools it
        // replaces, and that difference has to come out of the budget.
        uint64_t pooledBytes = 0, maxBytes = 0;
        MemRefType unionType;
        bool usable = true;
        for (int64_t origin : group) {
            auto footprint = footprintByOrigin.find(origin);
            auto ubType = ubTypeByOrigin.find(origin);
            if (footprint == footprintByOrigin.end() || ubType == ubTypeByOrigin.end()) {
                usable = false;
                break;
            }
            pooledBytes += std::min(lanes, interCoreBufferDepth) * footprint->second.bytes;
            if (footprint->second.bytes > maxBytes) {
                maxBytes = footprint->second.bytes;
                unionType = ubType->second;
            }
        }
        if (!usable || !unionType)
            continue;

        const uint64_t unionBytes = static_cast<uint64_t>(lanes) * maxBytes;
        const uint64_t extraBytes = unionBytes > pooledBytes ? unionBytes - pooledBytes : 0;
        if (extraBytes > ubBudgetLeft) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Phases " << group.front() << ".." << group.back()
                                    << ": kept separate pools; one union slot per lane needs " << extraBytes
                                    << " more bytes of UB but " << ubBudgetLeft << " remain\n");
            continue;
        }

        // A slot per lane costs a forward flag per lane instead of one per
        // rotating slot, and the group's back-edge release costs one more.
        unsigned extraFlags = 1;
        for (int64_t origin : group)
            extraFlags += lanes - slotCountByOrigin[origin];
        if (flagsUsed + extraFlags > kMaxTransferFlags) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Phases " << group.front() << ".." << group.back()
                                    << ": kept separate pools; one union slot per lane needs " << extraFlags
                                    << " more flags but only " << (kMaxTransferFlags - flagsUsed) << " remain\n");
            continue;
        }

        ubBudgetLeft -= extraBytes;
        flagsUsed += extraFlags;

        const int64_t key = group.front();
        for (int64_t origin : group) {
            slotGroupOfOrigin[origin] = key;
            slotCountByOrigin[origin] = lanes;
            unionTypeOfOrigin[origin] = unionType;
        }
        mergedGroupKeys.push_back(key);
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Phases " << group.front() << ".." << group.back() << " share " << lanes
                                << " union slot(s) of " << maxBytes << " bytes, one per lane (" << pooledBytes
                                << " -> " << unionBytes << " bytes): a lane's buffer carries every role\n");
    }
    // Deterministic flag ids regardless of the DenseMap iteration order above.
    llvm::sort(mergedGroupKeys);

    // Promoting individual pools is opt-in, separately from the budget that
    // funds the merge above.
    //
    // The reuse it removes is a lane writing a slot an earlier lane still owns,
    // and in the schedules measured so far that pair is already ordered by a
    // flag the schedule needs for its own data: the consumer waits for the
    // previous lane's reverse-direction result before it reaches the write.
    // Promotion therefore buys no removed stall at those unroll factors, while
    // costing buffers and -- more scarce -- flags. What it does buy is that the
    // ordering stops being emergent, so it is offered rather than assumed.
    for (int64_t origin : promotePrivateBufferPools ? ArrayRef<int64_t>(candidates) : ArrayRef<int64_t>{}) {
        const unsigned lanes = laneCountByOrigin[origin];
        if (lanes <= interCoreBufferDepth)
            continue; // nothing is reused; already private
        if (slotCountByOrigin[origin] >= lanes)
            continue; // a union merge already gave this pool a slot per lane
        auto it = footprintByOrigin.find(origin);
        if (it == footprintByOrigin.end())
            continue; // shape the emitter will reject anyway

        const unsigned extraSlots = lanes - interCoreBufferDepth;
        const uint64_t extraBytes = it->second.isUB ? extraSlots * it->second.bytes : 0;
        const unsigned extraFlags = extraSlots; // one forward flag per slot

        // Say what was left on the table. A pool that silently stays on the
        // rotating set reads as "nothing to promote" in the output, which is
        // indistinguishable from a pool that could not be afforded.
        if (extraBytes > ubBudgetLeft) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Pool " << origin << ": kept the rotating set; one buffer per lane "
                                    << "needs " << extraBytes << " more bytes of UB but " << ubBudgetLeft
                                    << " remain\n");
            continue;
        }
        if (flagsUsed + extraFlags > kMaxTransferFlags) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Pool " << origin << ": kept the rotating set; one buffer per lane "
                                    << "needs " << extraFlags << " more flags but only "
                                    << (kMaxTransferFlags - flagsUsed) << " remain\n");
            continue;
        }

        ubBudgetLeft -= extraBytes;
        flagsUsed += extraFlags;
        slotCountByOrigin[origin] = lanes;
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Pool " << origin << ": one buffer per lane (" << lanes
                                << " slots, +" << extraBytes << " bytes "
                                << (it->second.isUB ? "UB" : "L1") << ", no reuse to order)\n");
    }

    // Phases no longer take the same number of flags, so bases are a prefix sum
    // over the phase order rather than a fixed stride.
    SmallVector<int64_t> phaseOrder(phaseCount);
    for (const auto &entry : phaseIndexByOrigin)
        phaseOrder[entry.second] = entry.first;
    DenseMap<int64_t, unsigned> phaseFlagOffsetByOrigin;
    unsigned runningOffset = 0;
    for (int64_t origin : phaseOrder) {
        phaseFlagOffsetByOrigin[origin] = runningOffset;
        runningOffset += slotCountByOrigin[origin];
    }
    // One release flag per merged group closes its loop back-edge; see
    // `emitMergedSlotRelease`.
    DenseMap<int64_t, unsigned> releaseFlagOffsetByGroup;
    for (int64_t key : mergedGroupKeys)
        releaseFlagOffsetByGroup[key] = runningOffset++;
    const unsigned requiredFlags = runningOffset;

    // The module may already synchronize on some IDs. Allocate this schedule
    // above them: the manager scans the module on construction, so its first
    // handed-out ID is the lowest one not already in use.
    auto module = loop->getParentOfType<ModuleOp>();
    FlagIdManager flagIdManager(module, /*firstAvailableId=*/0);
    const int flagBase = flagIdManager.acquireId(/*insertionPoint=*/nullptr);
    if (flagBase < 0 || static_cast<unsigned>(flagBase) + requiredFlags > kMaxTransferFlags) {
        loop.emitError() << "CVSplitScheduling requires " << requiredFlags << " synchronization flags for "
                         << phaseCount << " transfer phase(s) at buffer depth " << interCoreBufferDepth
                         << " starting at ID " << flagBase << ", but only IDs 0.." << kMaxTransferFlagId
                         << " are available";
        // Promotion is capped against kMaxTransferFlags above, so overflow here
        // means the un-promoted schedule was already too wide.
        return failure();
    }

    // The shared DCVP buffer-count policy controls the pool depth. Same-typed
    // buffers (all unrolled qk_ub, all pv_ub, all P L1) rotate over that many
    // physical allocations; absence of a frontend policy defaults to two.
    BufferPool bufferPool;
    SmallVector<VectorToCubeTransferChain> vectorToCubeChains;
    DenseMap<int64_t, unsigned> laneOrdinalByOrigin;

    for (auto &xfer : transfers) {
        auto tensorType = cast<RankedTensorType>(xfer.value.getType());
        if (tensorType.getRank() != 2) {
            loop.emitError("cross-scope transfers require rank-2 tensors");
            return failure();
        }

        const unsigned ordinal = laneOrdinalByOrigin[xfer.originId]++;
        const unsigned slot = ordinal % slotCountByOrigin[xfer.originId];
        const unsigned phaseBase = flagBase + phaseFlagOffsetByOrigin[xfer.originId];
        const TransferSyncPlan plan {
            /*forwardFlagId=*/static_cast<int>(phaseBase + slot),
            /*slotGroupKey=*/slotGroupOfOrigin[xfer.originId],
            /*slot=*/slot,
            /*slotAllocType=*/unionTypeOfOrigin.lookup(xfer.originId)};

        if (xfer.direction == CrossScopeTransfer::CUBE_TO_VECTOR)
            emitCubeToVectorTransfer(ec, xfer, tensorType, plan, bufferPool);
        else
            vectorToCubeChains.push_back(emitVectorToCubeTransfer(ec, xfer, tensorType, plan, bufferPool));
    }

    // Every consumer is in place now, so the last reader of each merged group is
    // known and its back-edge can be closed.
    for (int64_t key : mergedGroupKeys)
        emitMergedSlotRelease(ec, transfers, slotGroupOfOrigin, key,
                              flagBase + static_cast<int>(releaseFlagOffsetByGroup[key]));


    LLVM_DEBUG(llvm::dbgs() << "[cv-split] Inserted " << transfers.size() << " transfers across " << phaseCount
                            << " phase(s) using " << requiredFlags << " sync flags\n");
    return CrossScopeTransferInfo {*blockM, std::move(vectorToCubeChains)};
}

} // namespace mlir::triton::cv_split
