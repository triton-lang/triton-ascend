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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"
namespace {

static constexpr unsigned kMaxTransferFlagId = 14;
static constexpr unsigned kMaxTransferFlags = kMaxTransferFlagId + 1;

// ============================================================================
// Stage 8: Insert cross-scope transfers and synchronization
//
// For each SSA value that crosses from CUBE→VECTOR or VECTOR→CUBE:
//   C→V: alloc UB buffer, insert fixpipe + sync_block_set/wait, replace uses
//   V→C: alloc L1 buffer (NZ), insert copy + sync_block_set/wait, replace uses
//
// This runs BEFORE scope separation so both scopes see the shared buffers.
// ============================================================================

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
    DenseMap<int64_t, DenseMap<Type, SmallVector<memref::AllocOp, 2>>> slots;
    DenseMap<int64_t, DenseMap<Type, unsigned>> useCount;
    // Cached 2D ND view for each physical L1 buffer.
    llvm::DenseMap<Operation *, Value> ndView;
    explicit BufferPool(unsigned depth) : depth(depth) {}

    unsigned depth;

    // Allocation to use for the next transfer of `allocType`: a new physical
    // buffer only while the rotating set is smaller than `depth`, otherwise the
    // round-robin reuse. `builder`'s insertion point must already be set (before
    // the loop) for the create case.
    memref::AllocOp getOrCreate(OpBuilder &builder, Location loc, int64_t originId, MemRefType allocType)
    {
        auto &vec = slots[originId][allocType];
        unsigned slot = useCount[originId][allocType]++ % depth;
        if (slot < vec.size())
            return vec[slot];
        auto allocOp = createAnnotatedAlloc(builder, loc, allocType);
        vec.push_back(allocOp);
        return allocOp;
    }
};

struct LaneAliasedUbPlan {
    SmallVector<CrossScopeTransfer *, 4> qkTransfers;
    SmallVector<CrossScopeTransfer *, 4> pTransfers;
    SmallVector<CrossScopeTransfer *, 4> pvTransfers;
    DenseMap<Operation *, Value> bufferByProducer;
    Operation *releaseAnchor = nullptr;
};

static bool reachesOperation(Value root, Operation *target, Block *body)
{
    SmallVector<Operation *> worklist(root.getUsers().begin(), root.getUsers().end());
    llvm::SmallPtrSet<Operation *, 32> visited;
    while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val();
        if (op == target)
            return true;
        if (op->getBlock() != body || !visited.insert(op).second)
            continue;
        for (Value result : op->getResults())
            llvm::append_range(worklist, result.getUsers());
    }
    return false;
}

// Recognize only the exact U4 QK -> P -> PV transfer schedule. Ordered
// occurrence pairing is intentional: recurrent softmax state makes an
// unrestricted backward slice ambiguous (QK_i may reach later lanes too).
static std::optional<LaneAliasedUbPlan>
matchLaneAliasedUbPlan(SmallVectorImpl<CrossScopeTransfer> &transfers,
                       Operation *lastVectorToCubeProducer, Block *body)
{
    LaneAliasedUbPlan plan;
    for (CrossScopeTransfer &xfer : transfers) {
        if (xfer.direction == CrossScopeTransfer::VECTOR_TO_CUBE) {
            plan.pTransfers.push_back(&xfer);
        } else if (lastVectorToCubeProducer && xfer.producer->isBeforeInBlock(lastVectorToCubeProducer)) {
            plan.qkTransfers.push_back(&xfer);
        } else {
            plan.pvTransfers.push_back(&xfer);
        }
    }

    constexpr unsigned kSupportedLanes = 4;
    if (plan.qkTransfers.size() != kSupportedLanes || plan.pTransfers.size() != kSupportedLanes ||
        plan.pvTransfers.size() != kSupportedLanes) {
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Lane-alias reject: transfer counts qk="
                                << plan.qkTransfers.size() << " p=" << plan.pTransfers.size()
                                << " pv=" << plan.pvTransfers.size() << "\n");
        return std::nullopt;
    }

    auto sameOrigin = [](ArrayRef<CrossScopeTransfer *> group) {
        return llvm::all_of(group, [&](CrossScopeTransfer *xfer) {
            return xfer->originId == group.front()->originId;
        });
    };
    if (!sameOrigin(plan.qkTransfers) || !sameOrigin(plan.pTransfers) || !sameOrigin(plan.pvTransfers)) {
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Lane-alias reject: unroll origins are not uniform\n");
        return std::nullopt;
    }

    for (unsigned lane = 0; lane < kSupportedLanes; ++lane) {
        CrossScopeTransfer *qk = plan.qkTransfers[lane];
        CrossScopeTransfer *p = plan.pTransfers[lane];
        CrossScopeTransfer *pv = plan.pvTransfers[lane];
        if (p->consumers.size() != 1 || p->consumers.front() != pv->producer ||
            !reachesOperation(qk->value, p->producer, body)) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Lane-alias reject: lane " << lane
                                    << " does not form QK->P->PV\n");
            return std::nullopt;
        }

        auto qkType = dyn_cast<RankedTensorType>(qk->value.getType());
        auto pvType = dyn_cast<RankedTensorType>(pv->value.getType());
        if (!qkType || !pvType || qkType.getRank() != 2 || pvType.getRank() != 2 ||
            !qkType.hasStaticShape() || !pvType.hasStaticShape() ||
            !qkType.getElementType().isF32() || qkType.getElementType() != pvType.getElementType() ||
            qkType.getDimSize(0) != pvType.getDimSize(0) || qkType.getDimSize(0) % 2 != 0) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Lane-alias reject: lane " << lane
                                    << " has unsupported tensor types\n");
            return std::nullopt;
        }
    }

    plan.releaseAnchor = plan.pvTransfers.back()->consumers.front();
    for (Operation *consumer : plan.pvTransfers.back()->consumers)
        if (plan.releaseAnchor->isBeforeInBlock(consumer))
            plan.releaseAnchor = consumer;
    return plan;
}

static Value createContiguousUbView(OpBuilder &builder, Location loc, Value root, ArrayRef<int64_t> shape,
                                    Type elementType)
{
    auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
    auto viewType = MemRefType::get(shape, elementType, nullptr, ubAddrSpace);
    if (root.getType() == viewType)
        return root;

    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(shape.size());
    sizes.reserve(shape.size());
    int64_t stride = 1;
    for (int64_t dim : shape)
        sizes.push_back(builder.getIndexAttr(dim));
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = builder.getIndexAttr(stride);
        stride *= shape[i];
    }
    return builder
        .create<memref::ReinterpretCastOp>(loc, viewType, root, builder.getIndexAttr(0), sizes, strides)
        .getResult();
}

static void materializeLaneAliasedUbPlan(const TransferEmitContext &c, LaneAliasedUbPlan &plan)
{
    OpBuilder builder(c.ctx);
    builder.setInsertionPoint(c.loop);
    for (unsigned lane = 0; lane < plan.qkTransfers.size(); ++lane) {
        auto qkType = cast<RankedTensorType>(plan.qkTransfers[lane]->value.getType());
        auto pvType = cast<RankedTensorType>(plan.pvTransfers[lane]->value.getType());
        int64_t rows = qkType.getDimSize(0) / 2;
        int64_t qkCols = qkType.getDimSize(1);
        int64_t pvCols = pvType.getDimSize(1);
        int64_t rootCols = std::max(qkCols, pvCols);
        auto ubAddrSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
        auto rootType = MemRefType::get({rows, rootCols}, qkType.getElementType(), nullptr, ubAddrSpace);
        Value root = createAnnotatedAlloc(builder, c.loc, rootType).getResult();
        plan.bufferByProducer[plan.qkTransfers[lane]->producer] =
            createContiguousUbView(builder, c.loc, root, {rows, qkCols}, qkType.getElementType());
        plan.bufferByProducer[plan.pvTransfers[lane]->producer] =
            createContiguousUbView(builder, c.loc, root, {rows, pvCols}, pvType.getElementType());
    }
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
static void emitCubeToVectorTransfer(const TransferEmitContext &c, CrossScopeTransfer &xfer,
                                     RankedTensorType tensorType, int flagId, BufferPool &bufferPool,
                                     Value laneAliasedBuffer = {})
{
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
    Value sharedBuffer = laneAliasedBuffer;
    if (!sharedBuffer)
        sharedBuffer = bufferPool.getOrCreate(builder, c.loc, xfer.originId, allocType).getResult();

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
                                                     sharedBuffer, // dst (M/2-row shared UB alloc)
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
    auto castOp = builder.create<memref::MemorySpaceCastOp>(c.loc, plainMemrefType, sharedBuffer);
    setOpEngineTypeAttr(castOp, EngineType::VECTOR);
    auto toTensorOp = builder.create<bufferization::ToTensorOp>(c.loc, halfTensorType, castOp.getResult(),
                                                                /*restrict=*/true, /*writable=*/true);
    setOpEngineTypeAttr(toTensorOp, EngineType::VECTOR);

    for (auto *consumer : xfer.consumers)
        consumer->replaceUsesOfWith(xfer.value, toTensorOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   C→V transfer #" << flagId << ": " << xfer.producer->getName() << " → "
                            << ubShape[0] << "x" << ubShape[1] << " UB buffer (ROW_SPLIT)\n");
}

// VECTOR -> CUBE: a softmax/cast result is NZ-packed and copied UB->L1 into a
// shared L1 buffer, VECTOR signals via sync_block_set, CUBE waits and reads it
// back through a convert_layout (NZ fractal -> ND view) for matmul consumption.
// NZ packing applies only when both dims are multiples of 16; otherwise the L1
// buffer keeps the flat [M, N] layout.
static VectorToCubeTransferChain emitVectorToCubeTransfer(const TransferEmitContext &c, CrossScopeTransfer &xfer,
                                                          RankedTensorType tensorType, int flagId,
                                                          BufferPool &bufferPool)
{
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
    auto sharedL1AllocOp = bufferPool.getOrCreate(builder, c.loc, xfer.originId, l1AllocType);

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

    LLVM_DEBUG(llvm::dbgs() << "[cv-split]   V→C transfer #" << flagId << ": " << xfer.producer->getName() << " → " << M
                            << "x" << N << " L1 buffer\n");

    return VectorToCubeTransferChain {xfer.value, sharedL1AllocOp.getResult(), syncSetOp, std::move(packingOps)};
}

} // namespace

FailureOr<CrossScopeTransferInfo>
insertCrossScopeTransfers(scf::ForOp loop, const DenseMap<Operation *, EngineType> &classification,
                          const DenseMap<Operation *, Operation *> &transferPhaseEnds,
                          unsigned interCoreBufferDepth)
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

    if (transfers.size() > kMaxTransferFlags) {
        loop.emitError() << "CVSplitScheduling requires " << transfers.size() << " synchronization flags, but only "
                         << kMaxTransferFlags << " are available (IDs 0.." << kMaxTransferFlagId << ")";
        return failure();
    }

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

    std::optional<LaneAliasedUbPlan> laneAliasedPlan =
        matchLaneAliasedUbPlan(transfers, lastVectorToCubeProducer, body);

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

    // FIXME: replace the per-direction counters below with one shared flag
    // counter. The hardware exposes 16 shared scalar-buffer IDs; pipe direction
    // does not create a separate ID namespace. Match DCVP by allocating IDs
    // 0..14, reserving ID 15 for control-flow synchronization, and reject the
    // transformation before mutation when more than 15 transfers are required.
    // With unique IDs, the currently supported unroll factor 4 needs 12 IDs.
    //
    // FIXME: add counter reuse. Build dependency-ordered transfer phases and
    // reuse one ID as a counter within a phase only when the source-core set
    // order and destination-core wait order contain the same transfers in the
    // same order. Reject the transformation if the resulting allocation still
    // requires more than 15 IDs.
    auto module = loop->getParentOfType<ModuleOp>();
    FlagIdManager flagIdManager(module, /*firstAvailableId=*/0);

    // Reserve every ID before mutating the IR. Lane-local UB aliasing needs one
    // additional cross-iteration ownership token beyond the twelve U4 transfer
    // flags. Existing flags in the module may shift all assigned IDs.
    SmallVector<int> transferFlagIds;
    transferFlagIds.reserve(transfers.size());
    for (CrossScopeTransfer &xfer : transfers)
        transferFlagIds.push_back(flagIdManager.acquireId(xfer.producer));
    std::optional<int64_t> laneOwnershipFlagId;
    if (laneAliasedPlan)
        laneOwnershipFlagId = flagIdManager.acquireId(loop);
    int highestFlagId = laneOwnershipFlagId ? static_cast<int>(*laneOwnershipFlagId)
                                            : (transferFlagIds.empty() ? -1 : transferFlagIds.back());
    if (highestFlagId > static_cast<int>(kMaxTransferFlagId)) {
        loop.emitError() << "CVSplitScheduling exhausted synchronization flags; "
                         << "next ID is " << highestFlagId << " but IDs 0.." << kMaxTransferFlagId
                         << " are available";
        return failure();
    }

    // The shared DCVP buffer-count policy controls the pool depth. Same-typed
    // buffers (all unrolled qk_ub, all pv_ub, all P L1) rotate over that many
    // physical allocations; absence of a frontend policy defaults to two.
    BufferPool bufferPool(interCoreBufferDepth);
    SmallVector<VectorToCubeTransferChain> vectorToCubeChains;

    if (laneAliasedPlan) {
        materializeLaneAliasedUbPlan(ec, *laneAliasedPlan);

        OpBuilder ownershipBuilder(ctx);
        ownershipBuilder.setInsertionPointToStart(body);
        auto wait = ownershipBuilder.create<hivm::SyncBlockWaitOp>(
            loc, ec.cubeCoreAttr, ec.pipeVAttr, ec.pipeFixAttr,
            OpFoldResult(ownershipBuilder.getI64IntegerAttr(*laneOwnershipFlagId)));
        setOpEngineTypeAttr(wait, EngineType::CUBE);

        ownershipBuilder.setInsertionPointAfter(laneAliasedPlan->releaseAnchor);
        auto release = ownershipBuilder.create<hivm::SyncBlockSetOp>(
            loc, ec.vecCoreAttr, ec.pipeVAttr, ec.pipeFixAttr,
            OpFoldResult(ownershipBuilder.getI64IntegerAttr(*laneOwnershipFlagId)));
        setOpEngineTypeAttr(release, EngineType::VECTOR);
    }

    for (auto indexedTransfer : llvm::enumerate(transfers)) {
        CrossScopeTransfer &xfer = indexedTransfer.value();
        int syncFlagId = transferFlagIds[indexedTransfer.index()];
        auto tensorType = cast<RankedTensorType>(xfer.value.getType());
        if (tensorType.getRank() != 2) {
            loop.emitError("cross-scope transfers require rank-2 tensors");
            return failure();
        }

        if (xfer.direction == CrossScopeTransfer::CUBE_TO_VECTOR) {
            Value laneAliasedBuffer;
            if (laneAliasedPlan)
                laneAliasedBuffer = laneAliasedPlan->bufferByProducer.lookup(xfer.producer);
            emitCubeToVectorTransfer(ec, xfer, tensorType, syncFlagId, bufferPool, laneAliasedBuffer);
        }
        else
            vectorToCubeChains.push_back(emitVectorToCubeTransfer(ec, xfer, tensorType, syncFlagId, bufferPool));
    }

    LLVM_DEBUG(llvm::dbgs() << "[cv-split] Inserted " << transfers.size() << " transfers with " << transfers.size()
                            << " sync flags\n");
    return CrossScopeTransferInfo {*blockM, std::move(vectorToCubeChains), laneOwnershipFlagId};
}

} // namespace mlir::triton::cv_split
