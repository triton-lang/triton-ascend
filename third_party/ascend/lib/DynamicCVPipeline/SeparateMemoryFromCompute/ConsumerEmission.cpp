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

/// Number of extra iter args per group: prodCounter + consCounter.
static constexpr int kExtraIterArgsPerGroup = 2;

namespace gmload {

// ============================================================================
// Producer emission helpers
// ============================================================================

static bool areAllResultsMapped(Operation *op, const IRMapping &mapping)
{
    if (op->getNumResults() == 0) {
        return false;
    }
    return llvm::all_of(op->getResults(),
                        [&](Value result) { return static_cast<bool>(mapping.lookupOrNull(result)); });
}

static bool materializeProducerOp(OpBuilder &builder, Operation *op, Block *oldBody,
                                  const llvm::DenseSet<Operation *> &producerSourceOps, IRMapping &mapping,
                                  llvm::DenseSet<Operation *> &materializedOps, llvm::DenseSet<Operation *> &activeOps);

static bool materializeProducerValue(OpBuilder &builder, Value value, Block *oldBody,
                                     const llvm::DenseSet<Operation *> &producerSourceOps, IRMapping &mapping,
                                     llvm::DenseSet<Operation *> &materializedOps,
                                     llvm::DenseSet<Operation *> &activeOps)
{
    if (mapping.lookupOrNull(value)) {
        return true;
    }

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (blockArg.getOwner() == oldBody) {
            LLVM_DEBUG(DBGS() << "producer value is an unmapped old-body block argument\n");
            return false;
        }
        return true;
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp) {
        return true;
    }

    if (producerSourceOps.contains(defOp) || defOp->getBlock() == oldBody)
        return materializeProducerOp(builder, defOp, oldBody, producerSourceOps, mapping, materializedOps, activeOps);

    return true;
}

static bool materializeProducerOp(OpBuilder &builder, Operation *op, Block *oldBody,
                                  const llvm::DenseSet<Operation *> &producerSourceOps, IRMapping &mapping,
                                  llvm::DenseSet<Operation *> &materializedOps, llvm::DenseSet<Operation *> &activeOps)
{
    if (materializedOps.contains(op) || areAllResultsMapped(op, mapping)) {
        return true;
    }

    if (activeOps.contains(op)) {
        LLVM_DEBUG(DBGS() << "cycle while materializing producer-private op: " << op->getName() << "\n");
        return false;
    }

    if (!producerSourceOps.contains(op) && op->getBlock() == oldBody) {
        if (!mlir::isMemoryEffectFree(op) || op->getNumRegions() != 0) {
            LLVM_DEBUG(DBGS() << "cannot clone old-body producer-private dependency: " << op->getName() << "\n");
            return false;
        }
    }

    activeOps.insert(op);
    for (Value operand : collectOperandsIncludingRegions(op)) {
        if (!materializeProducerValue(builder, operand, oldBody, producerSourceOps, mapping, materializedOps,
                                      activeOps)) {
            activeOps.erase(op);
            return false;
        }
    }
    activeOps.erase(op);

    builder.clone(*op, mapping);
    materializedOps.insert(op);
    return true;
}

/// Emit the producer scf.if for one buffer slot.  Returns {flagNext, prodCounterNext}.
std::pair<Value, Value> emitProducerSlot(OpBuilder &builder, Location loc, Block *oldBody, Block *newBody,
                                         const LoadGroup &group, int slotIdx, Value flagArg, Value prodCounterCur,
                                         Value tripCount, Value falseVal, Value loopLb, Value loopStep, Value trueFlag,
                                         Value indexOne, const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp)
{
    int numOrig = static_cast<int>(origForOp.getInitArgs().size());
    int numLoads = static_cast<int>(group.loads.size());
    IntegerAttr blockId =
        group.loads.empty() ? IntegerAttr {} : group.loads[0].markedOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);

    Value flagEmpty = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, flagArg, falseVal);
    tagWithBlockId(flagEmpty, blockId);
    Value prodLtN = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, prodCounterCur, tripCount);
    tagWithBlockId(prodLtN, blockId);
    Value cond = builder.create<arith::AndIOp>(loc, flagEmpty, prodLtN);
    tagWithBlockId(cond, blockId);

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange {builder.getI1Type(), builder.getIndexType()}, cond, true);
    if (blockId)
        ifOp->setAttr(kBlockIdAttr, blockId);
    ifOp->setAttr(kLoadStoreAttr, builder.getUnitAttr());

    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(ifOp.thenBlock());

        IRMapping prodMapping;
        Value oldIV = oldBody->getArgument(0);
        Type ivType = oldIV.getType();

        Value lb = castIndexTo(builder, loc, loopLb, ivType);
        if (lb != loopLb)
            tagWithBlockId(lb, blockId);
        Value step = castIndexTo(builder, loc, loopStep, ivType);
        if (step != loopStep)
            tagWithBlockId(step, blockId);
        Value k = castIndexTo(builder, loc, prodCounterCur, ivType);
        tagWithBlockId(k, blockId);
        Value ivMul = builder.create<arith::MulIOp>(loc, k, step);
        tagWithBlockId(ivMul, blockId);
        Value ivAdd = builder.create<arith::AddIOp>(loc, lb, ivMul);
        tagWithBlockId(ivAdd, blockId);
        prodMapping.map(oldIV, ivAdd);

        // k is already cast to ivType above; cache casts by type to avoid
        // redundant index_cast ops when multiple iter_args share the same type.
        llvm::DenseMap<Type, Value> kByType;
        kByType[ivType] = k;

        for (int i = 0; i < numOrig; ++i) {
            Value oldArg = oldBody->getArgument(i + 1);
            if (Value delta = iterArgDeltas[i]) {
                Type argType = oldArg.getType();
                Value initArg = origForOp.getInitArgs()[i];
                // If initArg == lb and delta == step this iter_arg tracks the IV
                // exactly — reuse ivAdd instead of recomputing lb + k*step.
                if (initArg == loopLb && delta == loopStep && argType == ivType) {
                    prodMapping.map(oldArg, ivAdd);
                    continue;
                }
                auto [it, inserted] = kByType.try_emplace(argType, Value {});
                if (inserted) {
                    it->second = castIndexTo(builder, loc, prodCounterCur, argType);
                    tagWithBlockId(it->second, blockId);
                }
                Value argMul = builder.create<arith::MulIOp>(loc, it->second, delta);
                tagWithBlockId(argMul, blockId);
                Value argAdd = builder.create<arith::AddIOp>(loc, initArg, argMul);
                tagWithBlockId(argAdd, blockId);
                prodMapping.map(oldArg, argAdd);
            } else {
                prodMapping.map(oldArg, newBody->getArgument(i + 1));
            }
        }

        for (int l = 0; l < numLoads; ++l)
            prodMapping.map(group.loads[l].allocOp->getResult(0), group.bufSlots[slotIdx][l]);

        llvm::DenseSet<Operation *> skipSet;
        for (int l = 0; l < numLoads; ++l) {
            skipSet.insert(group.loads[l].allocOp);
            skipSet.insert(group.loads[l].markedOp);
        }
        llvm::DenseSet<Operation *> producerSourceOps(group.mergedChain.begin(), group.mergedChain.end());
        llvm::DenseSet<Operation *> materializedOps;
        llvm::DenseSet<Operation *> activeOps;
        for (Operation *op : group.mergedChain) {
            if (skipSet.contains(op))
                continue;
            if (!materializeProducerOp(builder, op, oldBody, producerSourceOps, prodMapping, materializedOps,
                                       activeOps)) {
                LLVM_DEBUG(DBGS() << "failed to materialize producer op, skip clone: " << op->getName() << "\n");
            }
        }

        Value prodNext = builder.create<arith::AddIOp>(loc, prodCounterCur, indexOne);
        tagWithBlockId(prodNext, blockId);
        builder.create<scf::YieldOp>(loc, ValueRange {trueFlag, prodNext});
    }

    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(ifOp.elseBlock());
        builder.create<scf::YieldOp>(loc, ValueRange {flagArg, prodCounterCur});
    }

    return {ifOp.getResult(0), ifOp.getResult(1)};
}

// ============================================================================
// Consumer emission helpers
// ============================================================================

/// Emit slot-selection for one load: one ToTensorOp per slot, followed by a
/// (depth-1)-level comparison/select chain that picks the slot matching
/// `target`.  Sets mapping[markedOp->result] = selected value.
Value emitLoadSlotSelection(OpBuilder &builder, Location loc, const MarkedLoad &load, ArrayRef<Value> slotBufs,
                            Value target, int depth, IRMapping &mapping, ArrayRef<Value> slotConsts)
{
    IntegerAttr blockId = load.markedOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);
    Type tensorTy = load.markedOp->getResult(0).getType();

    SmallVector<Value> slotVals(depth);
    for (int s = 0; s < depth; ++s) {
        auto toTensor = builder.create<bufferization::ToTensorOp>(loc, tensorTy, slotBufs[s], true, true);
        if (blockId)
            toTensor->setAttr(kBlockIdAttr, blockId);
        slotVals[s] = toTensor;
    }

    Value result = slotVals[depth - 1];
    for (int s = depth - 2; s >= 0; --s) {
        Value eq = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, target, slotConsts[s]);
        tagWithBlockId(eq, blockId);
        result = builder.create<arith::SelectOp>(loc, eq, slotVals[s], result);
        tagWithBlockId(result, blockId);
    }

    mapping.map(load.markedOp->getResult(0), result);
    return result;
}

/// Emit the scf.if that clears slot `slotIdx`'s flag when it is the consumed
/// slot (`target == slotIdx`).  Returns the updated flag value.
Value emitSlotFlagClear(OpBuilder &builder, Location loc, Value target, Value slotConst, Value flagNext, Value falseVal,
                        IntegerAttr blockId)
{
    Value isConsumed = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, target, slotConst);
    tagWithBlockId(isConsumed, blockId);

    auto ifClear = builder.create<scf::IfOp>(loc, TypeRange {builder.getI1Type()}, isConsumed, true);
    if (blockId)
        ifClear->setAttr(kBlockIdAttr, blockId);
    ifClear->setAttr(kLoadStoreAttr, builder.getUnitAttr());
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(ifClear.thenBlock());
        builder.create<scf::YieldOp>(loc, ValueRange {falseVal});
    }
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(ifClear.elseBlock());
        builder.create<scf::YieldOp>(loc, ValueRange {flagNext});
    }
    return ifClear.getResult(0);
}

// ============================================================================

/// Emit the full producer + consumer body for all groups plus a combined scf.yield.
void emitConsumer(OpBuilder &builder, Location loc, SmallVectorImpl<LoadGroup> &groups, ExtendedForInfo &info,
                  const SmallVector<Value> &groupIndexOneVals,
                  const SmallVector<SmallVector<Value>> &allGroupSlotConsts,
                  const SmallVector<Value> &allGroupDepthConsts, const llvm::DenseSet<Operation *> &allCtxForOps,
                  Value tripCount, Value lb, Value step, const SmallVector<Value> &groupTrueFlagVals,
                  const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp)
{
    int numGroups = static_cast<int>(groups.size());
    Block *oldBody = info.oldBody;

    // A retained chain op is a chain op that must still be cloned into the
    // consumer body because non-chain body code uses it.  Retained ops are
    // emitted in their original top-level run, not in the load group that
    // discovered them.
    llvm::DenseSet<Operation *> allChainOps;
    llvm::DenseSet<Operation *> retainedChainOpSet;
    for (int g = 0; g < numGroups; ++g) {
        llvm::DenseSet<Operation *> skipSet = computeSkipInConsumer(groups[g].loads, oldBody);
        for (Operation *op : groups[g].mergedChain) {
            allChainOps.insert(op);
            if (!skipSet.contains(op))
                retainedChainOpSet.insert(op);
        }
    }

    SmallVector<SmallVector<Value>> allFlagFinals(numGroups);
    SmallVector<Value> allConsCounterNexts(numGroups);
    SmallVector<Value> allProdCounterFinals(numGroups);
    SmallVector<SmallVector<Value>> groupFlagNexts(numGroups);
    SmallVector<Value> groupTargets(numGroups);
    SmallVector<char> groupPrefixEmitted(numGroups, 0);
    SmallVector<char> groupFinalEmitted(numGroups, 0);

    auto isSkippedOldBodyOp = [&allCtxForOps, &allChainOps, &retainedChainOpSet](Operation *op) {
        if (allCtxForOps.contains(op)) {
            return true;
        }
        return allChainOps.contains(op) && !retainedChainOpSet.contains(op);
    };

    auto emitGroupPrefix = [&groups, &builder, &loc, &oldBody, &info, &tripCount, &lb, &step, &groupTrueFlagVals,
                            &groupIndexOneVals, &iterArgDeltas, &origForOp, &allGroupDepthConsts, &groupFlagNexts,
                            &allProdCounterFinals, &allGroupSlotConsts, &groupTargets, &groupPrefixEmitted](int g) {
        if (groupPrefixEmitted[g]) {
            return;
        }
        auto &group = groups[g];
        int gdepth = group.depth;
        IntegerAttr blockId =
            group.loads.empty() ? IntegerAttr {} : group.loads[0].markedOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);

        Value prodCounterCur = info.groupArgs[g].prodCounter;
        SmallVector<Value> flagNexts(gdepth);
        for (int s = 0; s < gdepth; ++s) {
            auto [fn, pc] = emitProducerSlot(
                builder, loc, oldBody, info.newBody, group, s, info.groupArgs[g].flagArgs[s], prodCounterCur, tripCount,
                info.falseVal, lb, step, groupTrueFlagVals[g], groupIndexOneVals[g], iterArgDeltas, origForOp);
            flagNexts[s] = fn;
            prodCounterCur = pc;
        }
        groupFlagNexts[g] = std::move(flagNexts);
        allProdCounterFinals[g] = prodCounterCur;

        Value target = builder.create<arith::RemUIOp>(loc, info.groupArgs[g].consCounter, allGroupDepthConsts[g]);
        tagWithBlockId(target, blockId);
        groupTargets[g] = target;

        for (int l = 0; l < static_cast<int>(group.loads.size()); ++l) {
            SmallVector<Value> slotBufs(gdepth);
            for (int s = 0; s < gdepth; ++s)
                slotBufs[s] = group.bufSlots[s][l];
            emitLoadSlotSelection(builder, loc, group.loads[l], slotBufs, target, gdepth, info.mapping,
                                  allGroupSlotConsts[g]);
        }
        groupPrefixEmitted[g] = 1;
    };

    auto emitGroupFinal = [&groupFinalEmitted, &groupPrefixEmitted, &emitGroupPrefix, &groups, &builder, &loc,
                           &groupTargets, &allGroupSlotConsts, &groupFlagNexts, &info, &allFlagFinals,
                           &allConsCounterNexts, &groupIndexOneVals](int g) {
        if (groupFinalEmitted[g]) {
            return;
        }
        if (!groupPrefixEmitted[g]) {
            emitGroupPrefix(g);
        }

        auto &group = groups[g];
        int gdepth = group.depth;
        IntegerAttr blockId =
            group.loads.empty() ? IntegerAttr {} : group.loads[0].markedOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);
        allFlagFinals[g].resize(gdepth);
        for (int s = 0; s < gdepth; ++s)
            allFlagFinals[g][s] = emitSlotFlagClear(builder, loc, groupTargets[g], allGroupSlotConsts[g][s],
                                                    groupFlagNexts[g][s], info.falseVal, blockId);

        allConsCounterNexts[g] =
            builder.create<arith::AddIOp>(loc, info.groupArgs[g].consCounter, groupIndexOneVals[g]);
        tagWithBlockId(allConsCounterNexts[g], blockId);
        groupFinalEmitted[g] = 1;
    };

    llvm::DenseMap<Operation *, SmallVector<int>> groupsByOwnerOp;
    for (int g = 0; g < numGroups; ++g) {
        if (groups[g].loads.empty()) {
            continue;
        }
        Operation *owner = getAncestorInBlock(groups[g].loads[0].markedOp, oldBody);
        if (!owner) {
            LLVM_DEBUG(DBGS() << "failed to find old-body owner for marked load group " << g << "\n");
            continue;
        }
        groupsByOwnerOp[owner].push_back(g);
    }

    for (auto &op : oldBody->without_terminator()) {
        auto ownerIt = groupsByOwnerOp.find(&op);
        if (ownerIt != groupsByOwnerOp.end()) {
            for (int g : ownerIt->second)
                emitGroupPrefix(g);
        }

        if (!isSkippedOldBodyOp(&op)) {
            if (areOperandsAvailableForClone(&op, oldBody, info.mapping)) {
                builder.clone(op, info.mapping);
            } else {
                LLVM_DEBUG(DBGS() << "skip old body op to preserve scheduling order; operands unavailable: "
                                  << op.getName() << "\n");
            }
        }

        if (ownerIt != groupsByOwnerOp.end()) {
            for (int g : ownerIt->second)
                emitGroupFinal(g);
        }
    }

    for (int g = 0; g < numGroups; ++g) {
        if (groupPrefixEmitted[g]) {
            continue;
        }
        LLVM_DEBUG(DBGS() << "emit marked load group without old-body owner at body end: " << g << "\n");
        emitGroupPrefix(g);
        emitGroupFinal(g);
    }

    // 8. Combined yield.
    auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
    SmallVector<Value> yieldVals;
    yieldVals.reserve(info.numOrig + numGroups * (info.depth + kExtraIterArgsPerGroup));
    for (Value v : oldYield.getOperands())
        yieldVals.push_back(info.mapping.lookupOrDefault(v));
    for (int g = 0; g < numGroups; ++g) {
        for (int s = 0; s < groups[g].depth; ++s)
            yieldVals.push_back(allFlagFinals[g][s]);
        yieldVals.push_back(allProdCounterFinals[g]);
        yieldVals.push_back(allConsCounterNexts[g]);
    }
    builder.create<scf::YieldOp>(loc, yieldVals);
}

} // namespace gmload
