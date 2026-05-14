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

namespace gmload {

static constexpr int kExtraIterArgsPerGroup = 2;

// ============================================================================
// Dead iter_arg elimination
// ============================================================================

/// Return true if the iter_arg at `iterArgIdx` in `forOp` is dead
///   (a) its loop result has no uses outside the loop, AND
///   (b) its block argument is only used in a chain of side-effect-free ops
///       whose results ultimately feed only the loop yield.
bool isDeadIterArg(scf::ForOp forOp, unsigned iterArgIdx)
{
    if (!forOp.getResult(iterArgIdx).use_empty()) {
        return false;
    }

    Value blockArg = forOp.getBody()->getArgument(iterArgIdx + 1);
    Block *forBody = forOp.getBody();
    Operation *yieldOp = forBody->getTerminator();

    llvm::SmallSetVector<Value, kInitialRefCapacity> frontier;
    frontier.insert(blockArg);

    while (!frontier.empty()) {
        Value v = frontier.pop_back_val();
        for (OpOperand &use : v.getUses()) {
            Operation *user = use.getOwner();
            Operation *ancestor = getAncestorInBlock(user, forBody);
            if (!ancestor) {
                continue;
            }
            if (ancestor == yieldOp) {
                continue;
            }
            if (!mlir::isMemoryEffectFree(ancestor)) {
                return false;
            }
            for (Value result : ancestor->getResults()) {
                frontier.insert(result);
            }
        }
    }
    return true;
}

/// Rebuild `forOp` dropping all iter_args that satisfy `isDeadIterArg`.
/// Only the first `candidateCount` iter_args are considered for pruning.
/// Returns the pruned ForOp, or `forOp` itself when nothing was pruned.
scf::ForOp pruneDeadIterArgs(OpBuilder &builder, scf::ForOp forOp, unsigned candidateCount)
{
    unsigned numIter = forOp.getNumResults();
    assert(candidateCount <= numIter && "candidateCount must not exceed total iter_arg count");

    llvm::DenseSet<unsigned> deadSet;
    for (unsigned i = 0; i < candidateCount; ++i) {
        if (isDeadIterArg(forOp, i)) {
            deadSet.insert(i);
        }
    }

    if (deadSet.empty()) {
        return forOp;
    }

    SmallVector<Value> liveInits;
    liveInits.reserve(numIter - deadSet.size());
    for (unsigned i = 0; i < numIter; ++i) {
        if (!deadSet.count(i)) {
            liveInits.push_back(forOp.getInitArgs()[i]);
        }
    }

    builder.setInsertionPoint(forOp);
    auto newFor = builder.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
                                             forOp.getStep(), liveInits);
    newFor->setAttrs(forOp->getAttrs());

    Block *oldBody = forOp.getBody();
    Block *newBody = newFor.getBody();
    auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());

    IRMapping mapping;
    mapping.map(oldBody->getArgument(0), newFor.getInductionVar());
    unsigned liveJ = 0;
    for (unsigned i = 0; i < numIter; ++i) {
        Value oldArg = oldBody->getArgument(i + 1);
        if (deadSet.count(i)) {
            mapping.map(oldArg, forOp.getInitArgs()[i]);
        } else {
            mapping.map(oldArg, newBody->getArgument(liveJ++ + 1));
        }
    }

    builder.setInsertionPointToStart(newBody);
    for (auto &op : oldBody->without_terminator()) {
        builder.clone(op, mapping);
    }

    SmallVector<Value> liveYield;
    liveYield.reserve(numIter - deadSet.size());
    for (unsigned i = 0; i < numIter; ++i) {
        if (!deadSet.count(i)) {
            liveYield.push_back(mapping.lookupOrDefault(oldYield.getOperand(i)));
        }
    }
    builder.create<scf::YieldOp>(oldYield.getLoc(), liveYield);

    liveJ = 0;
    for (unsigned i = 0; i < numIter; ++i) {
        if (!deadSet.count(i)) {
            forOp.getResult(i).replaceAllUsesWith(newFor.getResult(liveJ++));
        }
    }

    forOp.erase();
    return newFor;
}

/// Erase side-effect-free ops in `body` whose results are entirely unused.
/// Propagates: erasing an op may expose its operand-defining ops as newly dead.
/// Only considers ops directly in `body` (not nested regions).
void eraseDeadBodyOps(Block *body)
{
    SmallVector<Operation *> worklist;
    llvm::SmallDenseSet<Operation *, kInitialRefCapacity> queued;
    auto enqueueIfDead = [&body, &queued, &worklist](Operation *op) {
        if (!op || op->getBlock() != body || op == body->getTerminator()) {
            return;
        }
        if (!queued.insert(op).second) {
            return;
        }
        worklist.push_back(op);
    };

    for (auto &op : *body) {
        if (&op == body->getTerminator()) {
            continue;
        }
        if (mlir::isMemoryEffectFree(&op) && llvm::all_of(op.getResults(), [](Value v) { return v.use_empty(); })) {
            enqueueIfDead(&op);
        }
    }

    while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val();
        queued.erase(op);
        if (!mlir::isMemoryEffectFree(op)) {
            continue;
        }
        if (!llvm::all_of(op->getResults(), [](Value v) { return v.use_empty(); })) {
            continue;
        }

        SmallVector<Operation *, kInitialRefCapacity> operandDefs;
        for (Value operand : op->getOperands()) {
            auto *defOp = operand.getDefiningOp();
            if (defOp && defOp->getBlock() == body) {
                operandDefs.push_back(defOp);
            }
        }
        op->erase();

        for (Operation *defOp : operandDefs) {
            if (mlir::isMemoryEffectFree(defOp) &&
                llvm::all_of(defOp->getResults(), [](Value v) { return v.use_empty(); })) {
                enqueueIfDead(defOp);
            }
        }
    }
}

// ============================================================================
// Producer address projection helpers
// ============================================================================

/// Return true if `v` is defined outside `forOp` (i.e. loop-invariant).
bool isLoopInvariant(Value v, scf::ForOp forOp)
{
    if (auto ba = dyn_cast<BlockArgument>(v)) {
        return ba.getOwner() != forOp.getBody();
    }
    Operation *defOp = v.getDefiningOp();
    Operation *parent = defOp->getParentOp();
    while (parent) {
        if (parent == forOp.getOperation()) {
            return false;
        }
        parent = parent->getParentOp();
    }
    return true;
}

/// If the yield value for iter-arg `iterArg` is of the form
/// arith.addi(iterArg, delta) where `delta` is loop-invariant,
/// return true and set `delta`.
bool getLinearIterArgDelta(Value iterArg, Value yieldVal, scf::ForOp forOp, Value &delta)
{
    auto addOp = yieldVal.getDefiningOp<arith::AddIOp>();
    if (!addOp) {
        return false;
    }
    Value candidate;
    if (addOp.getLhs() == iterArg) {
        candidate = addOp.getRhs();
    } else if (addOp.getRhs() == iterArg) {
        candidate = addOp.getLhs();
    } else {
        return false;
    }
    if (!isLoopInvariant(candidate, forOp)) {
        return false;
    }
    delta = candidate;
    return true;
}

Value castIndexTo(OpBuilder &builder, Location loc, Value val, Type targetType)
{
    if (val.getType() == targetType) {
        return val;
    }
    return builder.create<arith::IndexCastOp>(loc, targetType, val);
}

Value castToIndex(OpBuilder &builder, Location loc, Value val)
{
    if (val.getType().isIndex()) {
        return val;
    }
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

Value createIndexConstant(OpBuilder &builder, Location loc, int64_t val)
{
    return builder.create<arith::ConstantIndexOp>(loc, val);
}

Value createBoolConstant(OpBuilder &builder, Location loc, bool val)
{
    return builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(val));
}

// ============================================================================
// Block-id tagging helper
// ============================================================================

/// Set kBlockIdAttr on `v`'s defining op if both the op and the attribute exist.
void tagWithBlockId(Value v, IntegerAttr blockId)
{
    if (blockId) {
        if (auto *defOp = v.getDefiningOp()) {
            defOp->setAttr(kBlockIdAttr, blockId);
        }
    }
}

// ============================================================================
// Core transformation helpers
// ============================================================================

/// Allocate one memref slot per (depth × load) pair before the loop.
void allocateBufferSlots(OpBuilder &builder, Location loc, scf::ForOp forOp, LoadGroup &group)
{
    builder.setInsertionPoint(forOp);
    int depth = group.depth;
    int numLoads = static_cast<int>(group.loads.size());
    group.bufSlots.resize(depth);
    for (int s = 0; s < depth; ++s) {
        group.bufSlots[s].resize(numLoads);
        for (int l = 0; l < numLoads; ++l) {
            auto newAlloc = builder.create<memref::AllocOp>(loc, group.loads[l].allocOp.getType());
            newAlloc->setAttrs(group.loads[l].allocOp.getOperation()->getAttrs());
            group.bufSlots[s][l] = newAlloc;
        }
    }
}

/// Build a new scf.for whose iter_args cover all groups:
///   [original..., (flags_g[depth], prodCounter_g, consCounter_g) for each g].
/// Returns metadata including per-group iter_arg handles.
ExtendedForInfo buildExtendedFor(OpBuilder &builder, Location loc, scf::ForOp forOp, ArrayRef<LoadGroup> groups,
                                 ConstantCache &cache)
{
    builder.setInsertionPoint(forOp);
    int numOrig = static_cast<int>(forOp.getInitArgs().size());
    int depth = groups.empty() ? 0 : groups[0].depth;

    Value falseVal = cache.getFalse(builder, loc);
    Value c0 = cache.getIndex(builder, loc, 0);

    SmallVector<Value> inits;
    inits.reserve(numOrig + static_cast<int>(groups.size()) * (depth + kExtraIterArgsPerGroup));
    for (Value v : forOp.getInitArgs()) {
        inits.push_back(v);
    }
    for (auto &group : groups) {
        for (int s = 0; s < group.depth; ++s) {
            inits.push_back(falseVal);
        }
        inits.push_back(c0); // prodCounter
        inits.push_back(c0); // consCounter
    }

    auto newFor = builder.create<scf::ForOp>(loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), inits);
    newFor->setAttrs(forOp->getAttrs());
    Block *oldBody = forOp.getBody();
    Block *newBody = newFor.getBody();

    IRMapping mapping;
    mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
    for (int i = 0; i < numOrig; ++i) {
        mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));
    }

    if (!newBody->empty()) {
        newBody->getTerminator()->erase();
    }
    builder.setInsertionPointToEnd(newBody);

    // +1 accounts for the induction variable at argument index 0
    static constexpr unsigned kInductionVarOffset = 1;

    SmallVector<GroupIterArgs> groupArgs;
    int argOffset = numOrig;
    for (auto &group : groups) {
        GroupIterArgs ga;
        for (int s = 0; s < group.depth; ++s) {
            ga.flagArgs.push_back(newBody->getArgument(kInductionVarOffset + argOffset + s));
        }
        ga.prodCounter = newBody->getArgument(kInductionVarOffset + argOffset + group.depth);
        ga.consCounter = newBody->getArgument(kInductionVarOffset + argOffset + group.depth + 1);
        argOffset += group.depth + kExtraIterArgsPerGroup;
        groupArgs.push_back(std::move(ga));
    }

    return {newFor, oldBody, newBody, std::move(mapping), std::move(groupArgs), falseVal, numOrig, depth};
}

// ============================================================================
// Clone availability checks
// ============================================================================

/// Return true if `value` can be used while cloning into the new body with the
/// current mapping. Values defined outside the old body still dominate; values
/// defined in the old body must already have been cloned and mapped.
bool isAvailableForClone(Value value, Block *oldBody, const IRMapping &mapping)
{
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        return blockArg.getOwner() != oldBody || static_cast<bool>(mapping.lookupOrNull(value));
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != oldBody) {
        return true;
    }
    return static_cast<bool>(mapping.lookupOrNull(value));
}

/// Check all direct and region-captured operands before cloning an op out of the
/// old loop body.
bool areOperandsAvailableForClone(Operation *op, Block *oldBody, const IRMapping &mapping)
{
    for (Value operand : collectOperandsIncludingRegions(op)) {
        if (!isAvailableForClone(operand, oldBody, mapping)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Body-run utilities
// ============================================================================

std::optional<int32_t> getBlockId(Operation *op)
{
    if (auto bid = op->getAttrOfType<IntegerAttr>(kBlockIdAttr)) {
        return static_cast<int32_t>(bid.getInt());
    }
    return std::nullopt;
}

bool sameRunBlockId(const BodyRun &run, std::optional<int32_t> bid)
{
    if (run.hasBlockId != bid.has_value()) {
        return false;
    }
    return !run.hasBlockId || run.blockId == *bid;
}

SmallVector<BodyRun> collectBodyRuns(Block *body)
{
    SmallVector<BodyRun> runs;
    for (auto &op : body->without_terminator()) {
        std::optional<int32_t> bid = getBlockId(&op);
        if (runs.empty() || !sameRunBlockId(runs.back(), bid)) {
            BodyRun run;
            if (bid) {
                run.hasBlockId = true;
                run.blockId = *bid;
            }
            runs.push_back(std::move(run));
        }
        runs.back().ops.push_back(&op);
    }
    return runs;
}

int findRunContainingOp(ArrayRef<BodyRun> runs, Operation *target)
{
    for (size_t idx = 0; idx < runs.size(); ++idx) {
        for (Operation *op : runs[idx].ops) {
            if (op == target) {
                return static_cast<int>(idx);
            }
        }
    }
    return -1;
}

int findFirstRunWithBlockId(ArrayRef<BodyRun> runs, int32_t blockId)
{
    for (size_t idx = 0; idx < runs.size(); ++idx) {
        if (runs[idx].hasBlockId && runs[idx].blockId == blockId) {
            return static_cast<int>(idx);
        }
    }
    return -1;
}

void logRepeatedTopLevelBlockRuns(Block *body, llvm::StringRef stage)
{
    llvm::DenseSet<int32_t> closed;
    bool hasCurrent = false;
    int32_t current = 0;

    for (auto &op : body->without_terminator()) {
        std::optional<int32_t> bid = getBlockId(&op);
        if (!bid) {
            if (hasCurrent) {
                closed.insert(current);
            }
            hasCurrent = false;
            continue;
        }

        if (hasCurrent && current == *bid) {
            continue;
        }

        if (hasCurrent) {
            closed.insert(current);
        }

        if (closed.contains(*bid)) {
            LLVM_DEBUG(DBGS() << "non-consecutive top-level ssbuffer.block_id run after " << stage
                              << ": block_id=" << *bid << "\n");
        }
        current = *bid;
        hasCurrent = true;
    }
}

// ============================================================================
// Core transformation
// ============================================================================

/// Rewrite one scf.for with multi-buffer logic.
/// allCtxForOps carries the set of all forOps being transformed so that inner
/// forOps (already handled) are not cloned again into the consumer body.
void transformFor(ForBufferCtx &ctx, const llvm::DenseSet<Operation *> &allCtxForOps)
{
    scf::ForOp forOp = ctx.forOp;
    int numGroups = static_cast<int>(ctx.groups.size());
    Location loc = forOp.getLoc();
    OpBuilder builder(forOp);

    // 1. Allocate buffer slots for every group before the loop.
    for (auto &group : ctx.groups) {
        allocateBufferSlots(builder, loc, forOp, group);
    }

    // Scan parent block once so buildExtendedFor can reuse existing constants.
    ConstantCache cache;
    cache.scan(forOp->getBlock());

    // 2. Shared loop-level infrastructure — emitted before buildExtendedFor so
    //    these loop-invariant ops land outside the new loop body.
    builder.setInsertionPoint(forOp);
    IntegerAttr forBlockId = forOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);
    Value lb = forOp.getLowerBound();
    Value step = forOp.getStep();
    Value ub = forOp.getUpperBound();
    Value lbIdx = castToIndex(builder, loc, lb);
    if (lbIdx != lb) {
        tagWithBlockId(lbIdx, forBlockId);
    }
    Value stepIdx = castToIndex(builder, loc, step);
    if (stepIdx != step) {
        tagWithBlockId(stepIdx, forBlockId);
    }
    Value ubIdx = castToIndex(builder, loc, ub);
    if (ubIdx != ub) {
        tagWithBlockId(ubIdx, forBlockId);
    }
    Value range = builder.create<arith::SubIOp>(loc, ubIdx, lbIdx);
    tagWithBlockId(range, forBlockId);
    Value tripCount = builder.create<arith::CeilDivUIOp>(loc, range, stepIdx);
    tagWithBlockId(tripCount, forBlockId);

    // Per-group loop-invariant constants, all created before buildExtendedFor
    // so they land outside the new loop body.
    SmallVector<Value> groupTrueFlagVals(numGroups);
    SmallVector<Value> groupIndexOneVals(numGroups);
    SmallVector<SmallVector<Value>> allGroupSlotConsts(numGroups);
    SmallVector<Value> allGroupDepthConsts(numGroups);
    for (int g = 0; g < numGroups; ++g) {
        auto &group = ctx.groups[g];
        IntegerAttr blockId =
            group.loads.empty() ? IntegerAttr {} : group.loads[0].markedOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);
        int gdepth = group.depth;
        groupTrueFlagVals[g] = createBoolConstant(builder, loc, true);
        tagWithBlockId(groupTrueFlagVals[g], blockId);
        groupIndexOneVals[g] = createIndexConstant(builder, loc, 1); // constant 1 for counter increment
        tagWithBlockId(groupIndexOneVals[g], blockId);
        allGroupSlotConsts[g].resize(gdepth);
        for (int s = 0; s < gdepth; ++s) {
            allGroupSlotConsts[g][s] = createIndexConstant(builder, loc, s);
            tagWithBlockId(allGroupSlotConsts[g][s], blockId);
        }
        allGroupDepthConsts[g] = createIndexConstant(builder, loc, gdepth);
        tagWithBlockId(allGroupDepthConsts[g], blockId);
    }

    // 3. Build one extended scf.for carrying all groups' iter-args at once.
    //    After this call the builder is positioned inside the new loop body.
    auto info = buildExtendedFor(builder, loc, forOp, ctx.groups, cache);

    auto oldYieldOp = cast<scf::YieldOp>(info.oldBody->getTerminator());
    SmallVector<Value> iterArgDeltas(info.numOrig, Value {});
    for (int i = 0; i < info.numOrig; ++i) {
        Value delta;
        if (getLinearIterArgDelta(info.oldBody->getArgument(i + 1), oldYieldOp->getOperand(i), forOp, delta)) {
            iterArgDeltas[i] = delta;
        }
    }

    // 4. Per-group producer + consumer emission (merged so each group's ops are
    //    consecutive in the output, giving consecutive ssbuffer.block_id runs).
    emitConsumer(builder, loc, ctx.groups, info, groupIndexOneVals, allGroupSlotConsts, allGroupDepthConsts,
                 allCtxForOps, tripCount, lb, step, groupTrueFlagVals, iterArgDeltas, forOp);

    // 6. Redirect original forOp results to the new forOp.
    for (int i = 0; i < info.numOrig; ++i) {
        forOp.getResult(i).replaceAllUsesWith(info.newForOp.getResult(i));
    }

    // 7. Prune dead iter-args introduced by the transformation.
    scf::ForOp finalFor = pruneDeadIterArgs(builder, info.newForOp, static_cast<unsigned>(info.numOrig));

    // 8. Erase side-effect-free ops whose results became unused after pruning.
    eraseDeadBodyOps(finalFor.getBody());

    // Debug-only diagnostic: keep this as a log instead of asserting so the
    // pass can still dump the transformed IR for investigation.
    logRepeatedTopLevelBlockRuns(finalFor.getBody(), "gm-load multi-buffer");
}

// ============================================================================
// Cleanup utilities
// ============================================================================

bool isAncestorBlock(Block *ancestor, Block *descendant)
{
    for (Block *b = descendant; b;) {
        if (b == ancestor) {
            return true;
        }
        auto *parentOp = b->getParentOp();
        if (!parentOp) {
            break;
        }
        b = parentOp->getBlock();
    }
    return false;
}

void deduplicateConstants(ModuleOp module)
{
    llvm::DenseMap<mlir::Attribute, Value> canonical;
    SmallVector<Operation *> toErase;

    module->walk<mlir::WalkOrder::PreOrder>([&](arith::ConstantOp cst) {
        if (cst->hasAttr(kBlockIdAttr)) {
            return mlir::WalkResult::advance();
        }
        auto [it, inserted] = canonical.try_emplace(cst.getValue(), cst.getResult());
        if (!inserted && it->second != cst.getResult()) {
            Block *canonBlock = it->second.getParentBlock();
            Block *thisBlock = cst->getBlock();
            if (canonBlock == thisBlock || isAncestorBlock(canonBlock, thisBlock)) {
                cst.getResult().replaceAllUsesWith(it->second);
                toErase.push_back(cst);
            }
        }
        return mlir::WalkResult::advance();
    });

    for (Operation *op : llvm::reverse(toErase)) {
        op->erase();
    }
}

} // namespace gmload
