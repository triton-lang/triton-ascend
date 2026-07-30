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

#include "ascend/include/CVSplitScheduling/QStaging.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::cv_split {

#define DEBUG_TYPE "cv-split-scheduling"

using QStagingCandidates = llvm::MapVector<Value, scope::ScopeOp>;

struct QSourceChain {
    Value source;
    memref::AllocOp allocation;
    memref::CopyOp copy;
    bufferization::ToTensorOp toTensor;
};

static FailureOr<QStagingCandidates> collectQStagingCandidates(func::FuncOp funcOp)
{
    QStagingCandidates candidates;
    funcOp.walk([&](Operation *op) {
        Value lhs;
        if (auto m = dyn_cast<linalg::MatmulTransposeBOp>(op))
            lhs = m.getInputs()[0];
        else if (auto m = dyn_cast<linalg::MatmulOp>(op))
            lhs = m.getInputs()[0];
        else
            return;

        auto tt = dyn_cast<RankedTensorType>(lhs.getType());
        if (!tt || tt.getRank() != 2 || !tt.getElementType().isF16())
            return;

        Operation *def = lhs.getDefiningOp();
        if (!def)
            return; // block argument: not a stage-able buffer

        auto enclosingFor = op->getParentOfType<scf::ForOp>();
        if (!enclosingFor)
            return;
        // Loop-variant LHS (e.g. P, produced inside the loop) is already
        // cbuf-backed.
        if (enclosingFor->isProperAncestor(def))
            return;

        auto cubeScope = op->getParentOfType<scope::ScopeOp>();
        if (!cubeScope)
            return; // only handle matmuls that ended up inside a scope
        // The LHS definition must dominate the scope.
        if (cubeScope->isProperAncestor(def))
            return;

        candidates.insert({lhs, cubeScope});
    });

    if (candidates.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] No Q staging candidate found\n");
        return failure();
    }
    return candidates;
}

static FailureOr<QSourceChain> findQSourceChain(Value q, func::FuncOp funcOp)
{
    auto toTensor = q.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensor) {
        funcOp.emitError("loop-invariant matmul LHS is not backed by to_tensor");
        return failure();
    }

    memref::CopyOp fillCopy;
    Value qMem = toTensor.getMemref();
    for (Operation *user : qMem.getUsers()) {
        auto copy = dyn_cast<memref::CopyOp>(user);
        if (!copy || copy.getTarget() != qMem)
            continue;
        if (fillCopy) {
            funcOp.emitError("multiple copies initialize a loop-invariant matmul LHS");
            return failure();
        }
        fillCopy = copy;
    }

    if (!fillCopy) {
        funcOp.emitError("failed to find the GM source for a loop-invariant matmul LHS");
        return failure();
    }

    auto allocation = qMem.getDefiningOp<memref::AllocOp>();
    if (!allocation) {
        funcOp.emitError("loop-invariant matmul LHS is not backed by memref.alloc");
        return failure();
    }

    return QSourceChain {fillCopy.getSource(), allocation, fillCopy, toTensor};
}

static memref::AllocOp createPersistentQBuffer(OpBuilder &builder, Location loc, RankedTensorType tt,
                                               scope::ScopeOp cubeScope)
{
    // The persistent cbuf buffer must live in the shared parent block, outside
    // both engine scopes. SplitMixKernel clones that block into the AIC and AIV
    // functions; keeping the allocation there gives both clones the same L1
    // layout and therefore identical addresses for cross-engine buffers.
    //
    // Q must also be the first cbuf buffer. PlanMemory lays out cbuf allocations
    // in order, and placing Q at offset zero keeps the L1-to-feature-buffer load
    // within the immediate-offset range supported by the simulator.
    Operation *firstCbufAlloc = nullptr;
    for (Operation &op : *cubeScope->getBlock()) {
        auto alloc = dyn_cast<memref::AllocOp>(&op);
        if (!alloc)
            continue;
        auto memrefType = dyn_cast<MemRefType>(alloc.getType());
        if (!memrefType)
            continue;
        auto addressSpace = dyn_cast_or_null<hivm::AddressSpaceAttr>(memrefType.getMemorySpace());
        if (addressSpace && addressSpace.getAddressSpace() == hivm::AddressSpace::L1) {
            firstCbufAlloc = &op;
            break;
        }
    }
    if (firstCbufAlloc)
        builder.setInsertionPoint(firstCbufAlloc);
    else
        builder.setInsertionPoint(cubeScope);

    auto cbufAddressSpace = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
    auto cbufType = MemRefType::get(tt.getShape(), tt.getElementType(), nullptr, cbufAddressSpace);
    auto cbufAlloc = builder.create<memref::AllocOp>(loc, cbufType);

    auto uniqueMark = builder.create<annotation::MarkOp>(loc, cbufAlloc.getResult());
    uniqueMark->setAttr("mem_unique", builder.getUnitAttr());

    auto effectsMark = builder.create<annotation::MarkOp>(loc, cbufAlloc.getResult());
    effectsMark->setAttr("effects",
                         builder.getArrayAttr({builder.getStringAttr("write"), builder.getStringAttr("read")}));
    return cbufAlloc;
}

static bufferization::ToTensorOp stageQIntoCbuf(OpBuilder &builder, Location loc, RankedTensorType tt, Block *scopeBody,
                                                Value srcMemref, memref::AllocOp cbufAlloc)
{
    // The load, binding, and read view are CUBE-only and must remain inside the
    // CUBE scope so SplitMixKernel drops them from the AIV clone.
    builder.setInsertionPointToStart(scopeBody);

    auto plainType = MemRefType::get(tt.getShape(), tt.getElementType());
    auto qAlloc = builder.create<memref::AllocOp>(loc, plainType);
    builder.create<memref::CopyOp>(loc, srcMemref, qAlloc.getResult());
    auto qBindTensor =
        builder.create<bufferization::ToTensorOp>(loc, tt, qAlloc.getResult(), /*restrict=*/true, /*writable=*/true);

    builder.create<annotation::MarkOp>(loc, qBindTensor.getResult(), ValueRange {cbufAlloc.getResult()},
                                       builder.getStrArrayAttr({"bind_buffer"}));

    // Read Q from the persistent cbuf buffer. Feeding the bound tensor directly
    // makes BiShengIR insert nd2nz and multi-buffering for Q, which can overflow
    // UB; this explicit view keeps Q single-buffered.
    auto qCastView = builder.create<memref::MemorySpaceCastOp>(loc, plainType, cbufAlloc.getResult());
    return builder.create<bufferization::ToTensorOp>(loc, tt, qCastView.getResult(), /*restrict=*/true,
                                                     /*writable=*/true);
}

static void replaceQMatmulLhsUses(Value q, bufferization::ToTensorOp qReadTensor, scope::ScopeOp cubeScope)
{
    q.replaceUsesWithIf(qReadTensor.getResult(), [&](OpOperand &use) {
        Operation *owner = use.getOwner();
        return cubeScope->isProperAncestor(owner) && isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(owner) &&
               use.getOperandNumber() == 0;
    });
}

static void eraseDeadOriginalQStaging(QSourceChain sourceChain)
{
    if (!sourceChain.toTensor->use_empty())
        return;

    sourceChain.toTensor.erase();
    sourceChain.copy.erase();
    if (sourceChain.allocation->use_empty())
        sourceChain.allocation.erase();
}

// Bind the loop-invariant matmul LHS (Q in flash-attention) into a dedicated L1
// (cbuf) buffer, matching the manual kernel.
//
// In the manual kernel Q is staged once into a persistent cbuf buffer
// (mem_unique) and the QK matmul reads it directly from L1:
//   %alloc_q = memref.alloc() : memref<MxKxf16, cbuf>
//   annotation.mark %alloc_q {mem_unique}
//   annotation.mark %alloc_q {effects = ["write","read"]}
//   annotation.mark %q keys = ["bind_buffer"] values = [%alloc_q : cbuf]
//
// Without this, our QK matmul reads Q from a loop-invariant plain memref and
// BiShengIR inserts an implicit stage whose descriptor is misaligned
// (runtime fixp_addr_misal / zero-burst MOV_SRC_TO_FB on the cube). Binding Q to
// a cbuf buffer gives the matmul an aligned, persistent L1 operand.
//
// We target only the loop-invariant LHS (defined outside the innermost loop):
// that is Q for QK. The PV matmul LHS (P) is loop-variant (produced by the
// vector scope) and already lives in cbuf via convert_layout, so it is skipped.
LogicalResult bindLoopInvariantMatmulLhsToCbuf(func::FuncOp funcOp)
{
    // Map each distinct loop-invariant LHS value to the CUBE scope that consumes
    // it. The persistent cbuf allocation lives in the scope's shared parent
    // block; the load, bind, read view, and operand rewrite live inside the CUBE
    // scope so they remain self-contained on the AIC side after the MIX split.
    FailureOr<QStagingCandidates> lhsToScope = collectQStagingCandidates(funcOp);
    if (failed(lhsToScope))
        return failure();

    for (auto &kv : *lhsToScope) {
        Value q = kv.first;
        scope::ScopeOp cubeScope = kv.second;
        Block *scopeBody = &cubeScope.getBodyRegion().front();

        // Trace Q back through its staging copy. The new Q load is placed inside
        // the CUBE scope and reads directly from the original GM view. Capturing
        // the old staged tensor would break dominance after the MIX split, while
        // copying from its cbuf-bound buffer would create an unsupported
        // cbuf-to-cbuf copy.
        FailureOr<QSourceChain> sourceChain = findQSourceChain(q, funcOp);
        if (failed(sourceChain))
            return failure();
        Value srcMemref = sourceChain->source;

        OpBuilder builder(cubeScope.getContext());
        Location loc = q.getLoc();
        auto tt = cast<RankedTensorType>(q.getType());
        auto cbufAlloc = createPersistentQBuffer(builder, loc, tt, cubeScope);
        auto qReadTensor = stageQIntoCbuf(builder, loc, tt, scopeBody, srcMemref, cbufAlloc);
        replaceQMatmulLhsUses(q, qReadTensor, cubeScope);

        // Drop the now-dead original load chain. The copy is side-effecting, so DCE
        // cannot remove it automatically.
        eraseDeadOriginalQStaging(*sourceChain);

        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Staged loop-invariant matmul LHS into cbuf " << tt
                                << " (inside CUBE scope, bind_buffer)\n");
    }

    return success();
}

} // namespace mlir::triton::cv_split
