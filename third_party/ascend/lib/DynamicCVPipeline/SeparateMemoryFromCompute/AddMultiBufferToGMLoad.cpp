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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadInternal.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadPass.h"

using namespace mlir;
using namespace triton;
using namespace gmload;

// Default multi-buffer depth for GM load operations.
static constexpr int kDefaultBufferDepth = 2;

// ============================================================================
// Step functions
// ============================================================================

int AddMultiBufferToGMLoadPass::collectAndGroupMarkedOps()
{
    auto module = getOperation();

    // Scan all IR for marked ops (generic, no loop dependency).
    markedOps_ = collectMarkedOps(module);
    if (markedOps_.empty()) {
        LDBG("No marked loads found, nothing to transform");
        return 0;
    }
    LDBG("Marked loads collected, start transformation");

    // Group by enclosing scf::ForOp.
    groupByEnclosingForOp(markedOps_, contexts_);

    // Apply depth policy: skip loops whose compile-time trip count is too small
    // to benefit, then record the slot count on each group.
    int depth = kDefaultBufferDepth;
    llvm::erase_if(contexts_, [depth](const ForBufferCtx &ctx) {
        if (auto tc = getConstantTripCount(ctx.forOp))
            return *tc <= depth;
        return false;
    });
    for (auto &ctx : contexts_)
        for (auto &g : ctx.groups)
            g.depth = depth;

    if (contexts_.empty())
        LDBG("No bufferable loops found");

    return 0;
}

int AddMultiBufferToGMLoadPass::sortContextsInnerFirst()
{
    if (contexts_.empty())
        return 0;

    // Sort contexts inner-first so that inner loops are transformed
    // before the outer loops that contain them.
    auto getNestingDepth = [](Operation *op) {
        unsigned d = 0;
        for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
            ++d;
        return d;
    };

    llvm::sort(contexts_, [&getNestingDepth](const ForBufferCtx &lhs, const ForBufferCtx &rhs) {
        Operation *lhsOp = const_cast<scf::ForOp &>(lhs.forOp).getOperation();
        Operation *rhsOp = const_cast<scf::ForOp &>(rhs.forOp).getOperation();
        unsigned lhsD = getNestingDepth(lhsOp);
        unsigned rhsD = getNestingDepth(rhsOp);
        if (lhsD != rhsD)
            return lhsD > rhsD;
        if (lhsOp->getBlock() == rhsOp->getBlock())
            return lhsOp->isBeforeInBlock(rhsOp);
        return false;
    });

    return 0;
}

int AddMultiBufferToGMLoadPass::transformAllLoops()
{
    if (contexts_.empty())
        return 0;

    allCtxForOps_.clear();
    for (auto &ctx : contexts_)
        allCtxForOps_.insert(ctx.forOp.getOperation());

    // Process inner loops before outer loops.
    for (auto &ctx : contexts_)
        transformFor(ctx, allCtxForOps_);

    return 0;
}

int AddMultiBufferToGMLoadPass::cleanupTransformedIR()
{
    if (contexts_.empty())
        return 0;

    auto module = getOperation();

    // Deduplicate untagged constants that inner-loop processing created
    // before outer-loop processing could provide the dominating equivalents.
    deduplicateConstants(module);

    // Erase replaced original for ops.
    llvm::DenseSet<Operation *> nestedForOps;
    for (auto &ctx : contexts_) {
        Operation *parent = ctx.forOp->getParentOp();
        while (parent) {
            if (allCtxForOps_.contains(parent)) {
                nestedForOps.insert(ctx.forOp.getOperation());
                break;
            }
            parent = parent->getParentOp();
        }
    }

    for (auto &ctx : llvm::reverse(contexts_)) {
        if (nestedForOps.contains(ctx.forOp.getOperation()))
            continue;
        ctx.forOp.erase();
    }

    return 0;
}

// ============================================================================
// Pass entry point
// ============================================================================

void AddMultiBufferToGMLoadPass::runOnOperation()
{
    auto module = getOperation();
    LDBG("Enter add-multi-buffer-to-gm-load pass");
    LLVM_DEBUG({ DBGS() << "Before add-multi-buffer-to-gm-load:\n" << module << "\n"; });

    // Step 1: Collect marked ops and group by enclosing forOp
    if (collectAndGroupMarkedOps() != 0) {
        LDBG("Step 1 collectAndGroupMarkedOps failed");
        signalPassFailure();
        return;
    }
    if (contexts_.empty())
        return;

    // Step 2: Sort contexts inner-first
    if (sortContextsInnerFirst() != 0) {
        LDBG("Step 2 sortContextsInnerFirst failed");
        signalPassFailure();
        return;
    }

    // Step 3: Transform each for loop with multi-buffer logic
    if (transformAllLoops() != 0) {
        LDBG("Step 3 transformAllLoops failed");
        signalPassFailure();
        return;
    }

    // Step 4: Cleanup transformed IR
    if (cleanupTransformedIR() != 0) {
        LDBG("Step 4 cleanupTransformedIR failed");
        signalPassFailure();
        return;
    }

    LLVM_DEBUG({ DBGS() << "After add-multi-buffer-to-gm-load:\n" << module << "\n"; });
    LDBG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferToGMLoadPass()
{
    return std::make_unique<AddMultiBufferToGMLoadPass>();
}

void registerAddMultiBufferToGMLoadPasses()
{
    registerPass(
        []() -> std::unique_ptr<mlir::Pass> { return createAddMultiBufferToGMLoadPass(); });
}

} // namespace triton
} // namespace mlir
