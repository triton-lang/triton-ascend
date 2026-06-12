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

#include "ascend/include/DynamicCVPipeline/SplitIfByBlockIdPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

static constexpr const char *DEBUG_TYPE = "SplitIfByBlockId";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                             \
  LLVM_DEBUG({                                \
    DBGS();                                   \
    llvm::dbgs() << __VA_ARGS__ << "\n";      \
  })

using namespace mlir;
using namespace triton;

// ============================================================================
// Part1: Discovery & Grouping (merged Part1 + Part2)
// ============================================================================
//
// Walk the entire module.  For each scf::IfOp whose body contains ops from
// >= 2 different ssbuffer.block_id values (excluding block 16 and absent),
// group the contained ops by block_id so that later parts can materialize
// one split if per group.

namespace {

/// One group of ops sharing the same block_id inside an scf.if.
/// N=0: all storage is heap-allocated, keeping sizeof small.
struct BlockGroup {
    int64_t blockId;
    SmallVector<Operation *, 0> ops;       // ops in this group (original order)
    SmallVector<scf::IfOp, 0> nestedIfs;   // nested ifs that belong to this group
};

} // namespace

// ============================================================================
// Part2 data structures (CrossGroupValue, YieldAugmentation)
// ============================================================================

/// A value-level cross-group SSA dependency.
/// Tracks which group produces a value and which groups consume it.
struct CrossGroupValue {
    Value value;                              // the SSA value crossing groups
    unsigned fromGroupIdx;                    // producer group index
    SmallVector<unsigned, 0> toGroupIndices;  // consumer group indices
};

/// Part3 yield plan for a candidate.
/// Both Case A and Case B use unified yield augmentation:
///   Slots 0..N-1:  original yield slots (Case A: N=0; Case B: N=original yield count)
///   Slots N..N+M-1: augmented cross-group values not in original yield
struct YieldAugmentation {
    /// Cross-group values (deduplicated, in yield slot order for Case A).
    SmallVector<CrossGroupValue, 0> crossValues;

    /// For each group (indexed by group order, NOT raw group index):
    ///   per-slot action: -1 = passthrough, >=0 = splitIf idx (oi) that produces this slot
    SmallVector<SmallVector<int, 0>, 0> groupSlotActions;

    /// Number of original yield slots (0 for Case A, N for Case B).
    unsigned numOriginalSlots = 0;

    /// Augmented cross-group values NOT in original yield (Case B only).
    SmallVector<Value, 0> augmentedValues;

    bool empty() const { return crossValues.empty(); }
};

/// Computed slot metadata for Part3 materialization.
/// Bundles everything Step 3.1 derives from CandidateIf + YieldAugmentation.
struct SlotPlan {
    unsigned nSlots = 0;
    SmallVector<Type, 0> resultTypes;
    SmallVector<Value, 0> slotValues;       // the SSA value each slot represents
    llvm::SmallDenseMap<Value, unsigned> valueToSlot;  // cross-group val -> slot idx

    /// Case B only: original other-region yield values for first-if passthrough.
    SmallVector<Value, 4> origOtherYieldValues;

    unsigned nResultIfs = 0;
    bool lastIsVoid = false;
};

struct CandidateIf {
    scf::IfOp ifOp;
    int64_t selfBlockId;                       // block_id on the if op itself
    bool hasYield;                             // Case A (false) vs Case B (true)
    SmallVector<BlockGroup, 0> thenGroups;     // groups in then region
    SmallVector<BlockGroup, 0> elseGroups;     // groups in else region

    // Part2 fills this:
    YieldAugmentation yieldAug;                   // value-level cross-group + yield plan

    /// True when any single branch (then or else) contains >= 2 distinct
    /// block_ids — only then is there something to split within that branch.
    bool needsSplit() const {
        return thenGroups.size() >= 2 || elseGroups.size() >= 2;
    }
};

// ---------------------------------------------------------------------------
// Debug helpers
// ---------------------------------------------------------------------------

static bool hasNestedIfs(const CandidateIf &c)
{
    for (auto &g : c.thenGroups)
        if (!g.nestedIfs.empty()) return true;
    for (auto &g : c.elseGroups)
        if (!g.nestedIfs.empty()) return true;
    return false;
}

static void dumpCandidates(const SmallVector<CandidateIf> &candidates)
{
    for (unsigned i = 0; i < candidates.size(); ++i) {
        auto &c = candidates[i];
        LDBG("  [" << i << "] scf.if  selfBlockId=" << c.selfBlockId
             << "  groups(then=" << static_cast<unsigned>(c.thenGroups.size())
             << " else=" << static_cast<unsigned>(c.elseGroups.size())
             << ")  yield=" << c.hasYield
             << "  nested=" << hasNestedIfs(c));
        for (auto &g : c.thenGroups)
            LDBG("      then block_id=" << g.blockId
                 << "  ops=" << static_cast<unsigned>(g.ops.size())
                 << "  nestedIfs=" << static_cast<unsigned>(g.nestedIfs.size()));
        for (auto &g : c.elseGroups)
            LDBG("      else block_id=" << g.blockId
                 << "  ops=" << static_cast<unsigned>(g.ops.size())
                 << "  nestedIfs=" << static_cast<unsigned>(g.nestedIfs.size()));
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

/// Walk a single block (then or else body), grouping ops by block_id.
/// Nested ifs attach to the nearest preceding group.
/// Ops without a splittable id are kept in the current active group.
static SmallVector<BlockGroup> groupOpsInBlock(Block &block)
{
    SmallVector<BlockGroup> groups;
    llvm::SmallDenseMap<int64_t, unsigned> idToIdx; // block_id -> index in groups

    auto getOrCreateGroup = [&](int64_t bid) -> BlockGroup & {
        auto it = idToIdx.find(bid);
        if (it != idToIdx.end())
            return groups[it->second];
        idToIdx[bid] = groups.size();
        groups.push_back({bid, {}, {}});
        return groups.back();
    };

    int64_t currentId = -1; // which group nested / ambient ops attach to

    for (auto &op : block) {
        if (isa<scf::YieldOp>(op))
            continue;

        if (auto nestedIf = dyn_cast<scf::IfOp>(op)) {
            // Attach nested if to the current active group
            if (currentId != -1)
                getOrCreateGroup(currentId).nestedIfs.push_back(nestedIf);
            continue;
        }

        auto bid = CVPipeline::getOpBlockId(&op);
        if (bid.has_value() && *bid != -1) {
            getOrCreateGroup(*bid).ops.push_back(&op);
            currentId = *bid;
        } else if (currentId != -1) {
            // ambient op (id == -1 or 16): keep in the current group
            getOrCreateGroup(currentId).ops.push_back(&op);
        }
    }

    return groups;
}

static SmallVector<CandidateIf> discoverCandidates(ModuleOp module)
{
    SmallVector<CandidateIf> result;

    module.walk([&](scf::IfOp ifOp) {
        CandidateIf cand;
        cand.ifOp = ifOp;
        cand.hasYield = (ifOp->getNumResults() > 0);

        // self block_id
        auto selfBlockId = CVPipeline::getOpBlockId(ifOp);
        cand.selfBlockId = selfBlockId.value_or(-1);

        // Group then region
        cand.thenGroups = groupOpsInBlock(*ifOp.thenBlock());

        // Group else region
        Block *elseBlk = ifOp.elseBlock();
        if (elseBlk)
            cand.elseGroups = groupOpsInBlock(*elseBlk);

        if (cand.needsSplit())
            result.push_back(cand);
    });

    return result;
}

// ============================================================================
// Part2: Dependency Analysis & Yield Planning
// ============================================================================
//
// For each candidate, in a single pass over operands:
//   - Group-level: detect cross-group SSA dependencies (for topo sort).
//   - Value-level: track which specific SSA values cross group boundaries.
//
// Then plan yield augmentation:
//   - Case A: promote all cross-group values to new yield slots.
//   - Case B: augment yield with extra slots for cross-group values
//     not in original yield (→ unified yield chain, no memref bridges).
//   - Compute slot actions for each split if.
//   - Topo sort groups so producers execute before consumers.
//
// Part4 consumes the resulting YieldAugmentation plan.

static void dumpYieldAugmentation(const CandidateIf &c)
{
    auto &ya = c.yieldAug;
    bool splitThen = c.thenGroups.size() >= 2;
    auto &groups = splitThen ? c.thenGroups : c.elseGroups;
    const char *region = splitThen ? "then" : "else";

    if (ya.crossValues.empty()) {
        LDBG("    [Part2] no cross-group values (region=" << region << ")");
        return;
    }

    // --- value-level cross-group deps ---
    if (!ya.crossValues.empty()) {
        LDBG("    [Part2] value-level cross-group deps ("
             << region << ", " << ya.crossValues.size() << " values):");
        for (auto &cv : ya.crossValues) {
            std::string consumerStr;
            for (unsigned toG : cv.toGroupIndices) {
                if (!consumerStr.empty()) consumerStr += ", ";
                consumerStr += std::to_string(groups[toG].blockId);
            }
            LDBG("      val from block_id=" << groups[cv.fromGroupIdx].blockId
                 << " -> consumed by block_id(s): [" << consumerStr << "]"
                 << "  type=" << cv.value.getType());
        }
    }

    // --- yield augmentation plan ---
    if (!ya.crossValues.empty()) {
        if (!c.hasYield) {
            LDBG("    [Part2] Case A yield augmentation: "
                 << ya.crossValues.size() << " new yield slot(s)");
            for (unsigned si = 0; si < ya.crossValues.size(); ++si)
                LDBG("      slot[" << si << "] = " << ya.crossValues[si].value
                     << "  (producer=block_id "
                     << groups[ya.crossValues[si].fromGroupIdx].blockId << ")");
        } else if (!ya.augmentedValues.empty()) {
            LDBG("    [Part2] Case B yield augmentation: "
                 << ya.augmentedValues.size() << " additional yield slot(s) on top of "
                 << ya.numOriginalSlots << " original");
            for (unsigned si = 0; si < ya.augmentedValues.size(); ++si)
                LDBG("      slot[" << (ya.numOriginalSlots + si)
                     << "] = " << ya.augmentedValues[si]
                     << "  (type=" << ya.augmentedValues[si].getType() << ")");
        }
    }

    // --- per-group slot actions ---
    if (!ya.groupSlotActions.empty()) {
        unsigned nCols = ya.groupSlotActions[0].size();
        unsigned nSplitIfs = splitThen ? c.thenGroups.size() : c.elseGroups.size();
        LDBG("    [Part2] per-group slot actions (" << nSplitIfs
             << " split ifs x " << nCols << " slots):");
        for (unsigned gi = 0; gi < nSplitIfs; ++gi) {
            std::string slotStr;
            for (int action : ya.groupSlotActions[gi]) {
                if (!slotStr.empty()) slotStr += ", ";
                slotStr += std::to_string(action);
            }
            LDBG("      splitIf[" << gi << "] (block_id="
                 << groups[gi].blockId << "): slots = [" << slotStr << "]");
        }
    }

}

/// Single-pass scan: value-level crossValueMap only.
/// Group order is implicitly the natural discovery order (block_ids appear
/// in dependency order within a sequential basic block).
/// Used for both then and else regions.
static void scanRegion(SmallVector<BlockGroup, 0> &groups,
                       llvm::SmallDenseMap<Operation *, unsigned> &opToGroup,
                       llvm::SmallDenseMap<Value,
                           std::pair<unsigned, llvm::SmallPtrSet<Operation *, 4>>> &crossValueMap)
{
    unsigned n = groups.size();
    if (n < 2) return;

    for (unsigned gi = 0; gi < n; ++gi) {
        // Scan regular ops' operands
        for (auto *op : groups[gi].ops) {
            for (auto &operand : op->getOpOperands()) {
                Value val = operand.get();
                auto *defOp = val.getDefiningOp();
                if (!defOp)
                    continue;
                auto it = opToGroup.find(defOp);
                if (it != opToGroup.end() && it->second != gi) {
                    auto &entry = crossValueMap[val];
                    entry.first = it->second;
                    entry.second.insert(op);
                }
            }
            // Walk nested regions (e.g., scf.for body) to find cross-group
            // SSA uses hidden inside region-bearing ops.
            for (auto &region : op->getRegions()) {
                region.walk([&](Operation *nestedOp) {
                    for (auto &nestedOperand : nestedOp->getOpOperands()) {
                        Value val = nestedOperand.get();
                        auto *defOp = val.getDefiningOp();
                        if (!defOp)
                            continue;
                        auto it = opToGroup.find(defOp);
                        if (it != opToGroup.end() && it->second != gi) {
                            auto &entry = crossValueMap[val];
                            entry.first = it->second;
                            entry.second.insert(op);
                        }
                    }
                });
            }
        }
        // Scan nested ifs' own operands (e.g., their condition) which may be
        // defined in a different group.
        for (auto nestedIf : groups[gi].nestedIfs) {
            for (auto &operand : nestedIf->getOpOperands()) {
                Value val = operand.get();
                auto *defOp = val.getDefiningOp();
                if (!defOp)
                    continue;
                auto it = opToGroup.find(defOp);
                if (it != opToGroup.end() && it->second != gi) {
                    auto &entry = crossValueMap[val];
                    entry.first = it->second;
                    entry.second.insert(nestedIf.getOperation());
                }
            }
        }
    }
}

/// Compute slotOwner vector and groupSlotActions matrix for Case B.
/// N original yield slots + M augmented slots (crossValues not in yield).
static void planYieldCaseB(CandidateIf &c,
                           bool splitThen,
                           ArrayRef<BlockGroup> groups,
                           llvm::SmallDenseMap<Operation *, unsigned> &opToGroup)
{
    unsigned nGroups = groups.size();
    auto &ya = c.yieldAug;

    auto splitYieldOp = cast<scf::YieldOp>(
        splitThen ? c.ifOp.thenBlock()->getTerminator()
                  : c.ifOp.elseBlock()->getTerminator());
    SmallPtrSet<Value, 8> yieldValues;
    for (auto operand : splitYieldOp->getOperands())
        yieldValues.insert(operand);

    ya.numOriginalSlots = splitYieldOp->getNumOperands();
    ya.augmentedValues.clear();
    for (auto &cv : ya.crossValues) {
        if (!yieldValues.contains(cv.value))
            ya.augmentedValues.push_back(cv.value);
    }

    unsigned N = ya.numOriginalSlots;
    unsigned M = ya.augmentedValues.size();
    unsigned nSlots = N + M;

    SmallVector<int, 0> slotOwner(nSlots, -1);
    for (unsigned slot = 0; slot < N; ++slot) {
        auto *defOp = splitYieldOp->getOperand(slot).getDefiningOp();
        if (!defOp) continue;
        auto it = opToGroup.find(defOp);
        if (it != opToGroup.end())
            slotOwner[slot] = static_cast<int>(it->second);
    }

    llvm::SmallDenseMap<Value, unsigned> augValueToSlot;
    for (unsigned si = 0; si < M; ++si)
        augValueToSlot[ya.augmentedValues[si]] = N + si;
    for (unsigned gi = 0; gi < nGroups; ++gi) {
        for (auto *op : groups[gi].ops) {
            for (auto result : op->getResults()) {
                auto it = augValueToSlot.find(result);
                if (it != augValueToSlot.end())
                    slotOwner[it->second] = static_cast<int>(gi);
            }
        }
    }

    for (unsigned gi = 0; gi < nGroups; ++gi) {
        SmallVector<int> actions(nSlots, -1);
        for (unsigned slot = 0; slot < nSlots; ++slot) {
            if (slotOwner[slot] == static_cast<int>(gi))
                actions[slot] = static_cast<int>(gi);
        }
        ya.groupSlotActions.push_back(std::move(actions));
    }
}

/// Compute groupSlotActions matrix for Case A.
/// All crossValues are promoted to new yield slots.
static void planYieldCaseA(CandidateIf &c,
                           ArrayRef<BlockGroup> groups)
{
    unsigned nGroups = groups.size();
    auto &ya = c.yieldAug;

    llvm::SmallDenseMap<Value, unsigned> valueToSlot;
    for (unsigned si = 0; si < ya.crossValues.size(); ++si)
        valueToSlot[ya.crossValues[si].value] = si;

    for (unsigned gi = 0; gi < nGroups; ++gi) {
        SmallVector<int> actions(ya.crossValues.size(), -1);
        for (auto *op : groups[gi].ops) {
            for (auto result : op->getResults()) {
                auto it = valueToSlot.find(result);
                if (it != valueToSlot.end())
                    actions[it->second] = static_cast<int>(gi);
            }
        }
        ya.groupSlotActions.push_back(std::move(actions));
    }
}

/// Plan yield augmentation from the collected crossValueMap.
/// Groups are already in natural dependency order (block_ids appear in
/// execution order within a sequential basic block), so group index gi
/// is used directly as split-if position.
static void planYield(CandidateIf &c,
                      bool splitThen,
                      ArrayRef<BlockGroup> groups,
                      llvm::SmallDenseMap<Operation *, unsigned> &opToGroup,
                      llvm::SmallDenseMap<Value,
                          std::pair<unsigned, llvm::SmallPtrSet<Operation *, 4>>> &crossValueMap)
{
    unsigned nGroups = groups.size();
    auto &ya = c.yieldAug;
    ya.crossValues.clear();
    ya.augmentedValues.clear();
    ya.groupSlotActions.clear();

    // Step 2.3.1: convert crossValueMap to CrossGroupValue list
    for (auto &[val, info] : crossValueMap) {
        unsigned fromG = info.first;
        SmallVector<unsigned, 2> toGroups;
        SmallVector<bool, 8> groupHasConsumer(nGroups, false);
        for (auto *consumerOp : info.second) {
            auto git = opToGroup.find(consumerOp);
            if (git != opToGroup.end() && !groupHasConsumer[git->second]) {
                groupHasConsumer[git->second] = true;
                toGroups.push_back(git->second);
            }
        }
        ya.crossValues.push_back({val, fromG, std::move(toGroups)});
    }

    // Stable sort for deterministic slot order: by producer group, then value ptr.
    llvm::sort(ya.crossValues, [](const CrossGroupValue &a,
                                   const CrossGroupValue &b) {
        if (a.fromGroupIdx != b.fromGroupIdx)
            return a.fromGroupIdx < b.fromGroupIdx;
        return a.value.getAsOpaquePointer() < b.value.getAsOpaquePointer();
    });

    // Step 2.3.2: Case B groupSlotActions first (must have dimensions even when
    // crossValues is empty, for Part3 to iterate over).
    if (c.hasYield)
        planYieldCaseB(c, splitThen, groups, opToGroup);

    if (ya.crossValues.empty())
        return;

    if (!c.hasYield)
        planYieldCaseA(c, groups);
}

static void analyzeDependencies(SmallVector<CandidateIf> &candidates)
{
    for (auto &c : candidates) {
        // Step 2.1: Build op → group index maps (including nested ifs)
        llvm::SmallDenseMap<Operation *, unsigned> opToThenGroup;
        for (unsigned gi = 0; gi < c.thenGroups.size(); ++gi) {
            for (auto *op : c.thenGroups[gi].ops)
                opToThenGroup[op] = gi;
            for (auto nestedIf : c.thenGroups[gi].nestedIfs)
                opToThenGroup[nestedIf.getOperation()] = gi;
        }

        llvm::SmallDenseMap<Operation *, unsigned> opToElseGroup;
        for (unsigned gi = 0; gi < c.elseGroups.size(); ++gi) {
            for (auto *op : c.elseGroups[gi].ops)
                opToElseGroup[op] = gi;
            for (auto nestedIf : c.elseGroups[gi].nestedIfs)
                opToElseGroup[nestedIf.getOperation()] = gi;
        }

        // Step 2.2: Single-pass scan — value-level crossValueMap only
        llvm::SmallDenseMap<Value,
            std::pair<unsigned, llvm::SmallPtrSet<Operation *, 4>>> thenValueMap, elseValueMap;

        scanRegion(c.thenGroups, opToThenGroup, thenValueMap);
        scanRegion(c.elseGroups, opToElseGroup, elseValueMap);

        // Step 2.3: Plan yield augmentation for the active region
        // Groups are in natural discovery order (block_ids appear in
        // dependency order within a sequential basic block).
        bool splitThen = c.thenGroups.size() >= 2;
        if (splitThen) {
            planYield(c, /*splitThen=*/true,
                      ArrayRef(c.thenGroups), opToThenGroup, thenValueMap);
        } else if (c.elseGroups.size() >= 2) {
            planYield(c, /*splitThen=*/false,
                      ArrayRef(c.elseGroups), opToElseGroup, elseValueMap);
        }

        // Step 2.4: Debug yield augmentation
        dumpYieldAugmentation(c);
    }
}

// ============================================================================
// Part3: Materialization
// ============================================================================
//
// Unified yield-chain approach for both Case A and Case B.
// For each candidate, create a chain of split ifs.  Each if has the same
// result types; non-producing slots passthrough from the previous if.
// Case A: cross-group values become new yield slots; first if's else
//   uses zero placeholders (since original if has no else values).
// Case B: original yield slots preserved; first if's else passes through
//   original else values.

/// Create a zero/default SSA value of the given type.
/// Used for placeholder values in Case A else blocks when the original
/// scf.if has no yield results to passthrough.
static Value createPlaceholderValue(OpBuilder &builder, Location loc, Type type)
{
    Value result;
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
        result = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                  tensorType.getElementType());
    } else if (auto floatType = dyn_cast<FloatType>(type)) {
        result = builder.create<arith::ConstantOp>(loc,
            builder.getFloatAttr(floatType, 0.0));
    } else if (auto intType = dyn_cast<IntegerType>(type)) {
        result = builder.create<arith::ConstantOp>(loc,
            builder.getIntegerAttr(intType, 0));
    } else if (type.isIndex()) {
        result = builder.create<arith::ConstantOp>(loc,
            builder.getIndexAttr(0));
    } else if (auto memrefType = dyn_cast<MemRefType>(type)) {
        // Create a base memref placeholder.
        // For memrefs with address_space (cbuf/ub), use alloca since these
        // special memory spaces require a valid allocation.
        // For plain memrefs, use a non-alloc-like placeholder (tensor.empty +
        // to_memref) so that downstream traceback (e.g. bisheng's
        // tracebackMemRefToAlloc) doesn't stop here prematurely and pick the
        // wrong alloc when both branches yield alloc-like values.
        Value baseMemref;
        if (memrefType.getMemorySpace()) {
            SmallVector<Value> allocaSizes;
            for (int64_t i = 0; i < memrefType.getRank(); ++i)
                if (memrefType.isDynamicDim(i))
                    allocaSizes.push_back(builder.create<arith::ConstantOp>(loc,
                        builder.getIndexAttr(1)).getResult());
            auto simpleType = MemRefType::get(memrefType.getShape(),
                                              memrefType.getElementType());
            baseMemref = builder.create<memref::AllocaOp>(loc, simpleType,
                                                          allocaSizes).getResult();
        } else {
            // For plain memrefs without address_space, prefer tensor.empty +
            // to_memref so downstream traceback doesn't stop on the wrong alloc.
            // But if the memref has dynamic dims, tensor.empty can't represent
            // them — fall back to memref.alloca.
            bool hasDynamic = llvm::any_of(memrefType.getShape(),
                                           ShapedType::isDynamic);
            if (hasDynamic) {
                SmallVector<Value> allocaSizes;
                for (int64_t i = 0; i < memrefType.getRank(); ++i)
                    if (memrefType.isDynamicDim(i))
                        allocaSizes.push_back(builder.create<arith::ConstantOp>(loc,
                            builder.getIndexAttr(1)).getResult());
                auto simpleType = MemRefType::get(memrefType.getShape(),
                                                  memrefType.getElementType());
                baseMemref = builder.create<memref::AllocaOp>(loc, simpleType,
                                                              allocaSizes).getResult();
            } else {
                auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                                        memrefType.getElementType());
                auto empty = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                        tensorType.getElementType());
                empty->setAttr(CVPipeline::kBlockId,
                              builder.getI32IntegerAttr(-1));
                auto simpleType = MemRefType::get(memrefType.getShape(),
                                                  memrefType.getElementType());
                baseMemref = builder.create<bufferization::ToMemrefOp>(
                    loc, simpleType, empty).getResult();
            }
        }
        // Apply strided layout via reinterpret_cast if needed.
        auto layout = memrefType.getLayout();
        if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(layout)) {
            SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
            SmallVector<Value> dynOffsets, dynSizes, dynStrides;
            // Offset
            int64_t off = stridedLayout.getOffset();
            staticOffsets.push_back(off);
            if (ShapedType::isDynamic(off))
                dynOffsets.push_back(builder.create<arith::ConstantOp>(loc,
                    builder.getIndexAttr(0)).getResult());
            // Sizes
            for (int64_t sz : memrefType.getShape()) {
                staticSizes.push_back(sz);
                if (ShapedType::isDynamic(sz))
                    dynSizes.push_back(builder.create<arith::ConstantOp>(loc,
                        builder.getIndexAttr(1)).getResult());
            }
            // Strides
            for (int64_t stride : stridedLayout.getStrides()) {
                staticStrides.push_back(stride);
                if (ShapedType::isDynamic(stride))
                    dynStrides.push_back(builder.create<arith::ConstantOp>(loc,
                        builder.getIndexAttr(1)).getResult());
            }
            OperationState state(loc, memref::ReinterpretCastOp::getOperationName());
            state.addTypes(memrefType);
            state.addOperands(baseMemref);
            state.addOperands(dynOffsets);
            state.addOperands(dynSizes);
            state.addOperands(dynStrides);
            state.addAttribute("operandSegmentSizes",
                builder.getDenseI32ArrayAttr({
                    1,
                    static_cast<int32_t>(dynOffsets.size()),
                    static_cast<int32_t>(dynSizes.size()),
                    static_cast<int32_t>(dynStrides.size())
                }));
            state.addAttribute("static_offsets",
                builder.getDenseI64ArrayAttr(staticOffsets));
            state.addAttribute("static_sizes",
                builder.getDenseI64ArrayAttr(staticSizes));
            state.addAttribute("static_strides",
                builder.getDenseI64ArrayAttr(staticStrides));
            result = builder.create(state)->getResult(0);
        } else {
            result = baseMemref;
        }
    } else {
        llvm_unreachable("unsupported type for placeholder value in Case A else block");
    }
    // Seed placeholders are ambient (don't belong to any specific block_id group).
    // Tag them with block_id = -1 so downstream passes scanning for block_id
    // attributes (e.g. AddControlFlowCondition) won't complain.
    result.getDefiningOp()->setAttr(CVPipeline::kBlockId,
        builder.getI32IntegerAttr(-1));
    return result;
}

// ============================================================================
// Part3 helpers
// ============================================================================

/// Get or create a zero/default placeholder of the given type.
/// Placeholders are cached by type and inserted before `insertBefore` so they
/// dominate every split-if in the chain.
static Value getOrCreatePlaceholder(Type type,
                                    llvm::SmallDenseMap<Type, Value> &cache,
                                    Operation *insertBefore,
                                    Location loc,
                                    OpBuilder &builder)
{
    auto it = cache.find(type);
    if (it != cache.end())
        return it->second;
    auto savedPoint = builder.saveInsertionPoint();
    builder.setInsertionPoint(insertBefore);
    Value val = createPlaceholderValue(builder, loc, type);
    cache[type] = val;
    builder.restoreInsertionPoint(savedPoint);
    return val;
}

/// 3.1 Compute slot metadata from CandidateIf + YieldAugmentation.
static SlotPlan buildSlotPlan(CandidateIf &c, bool splitThen)
{
    auto originalIf = c.ifOp;
    auto &ya = c.yieldAug;
    SlotPlan plan;

    if (!c.hasYield) {
        // Case A: slots = cross-group values
        plan.nSlots = ya.crossValues.size();
        for (unsigned s = 0; s < plan.nSlots; ++s) {
            plan.resultTypes.push_back(ya.crossValues[s].value.getType());
            plan.slotValues.push_back(ya.crossValues[s].value);
            plan.valueToSlot[ya.crossValues[s].value] = s;
        }
    } else {
        // Case B: slots = original yield results + augmented cross-group values
        unsigned N = ya.numOriginalSlots;
        unsigned M = ya.augmentedValues.size();
        plan.nSlots = N + M;

        for (unsigned s = 0; s < N; ++s)
            plan.resultTypes.push_back(originalIf.getResult(s).getType());
        for (unsigned s = 0; s < M; ++s)
            plan.resultTypes.push_back(ya.augmentedValues[s].getType());

        auto thenYield = cast<scf::YieldOp>(
            splitThen ? originalIf.thenBlock()->getTerminator()
                      : originalIf.elseBlock()->getTerminator());
        for (unsigned s = 0; s < N; ++s)
            plan.slotValues.push_back(thenYield->getOperand(s));
        for (unsigned s = 0; s < M; ++s)
            plan.slotValues.push_back(ya.augmentedValues[s]);

        for (auto &cv : ya.crossValues) {
            for (unsigned i = 0; i < N; ++i) {
                if (plan.slotValues[i] == cv.value) {
                    plan.valueToSlot[cv.value] = i;
                    break;
                }
            }
            for (unsigned i = 0; i < M; ++i) {
                if (ya.augmentedValues[i] == cv.value) {
                    plan.valueToSlot[cv.value] = N + i;
                    break;
                }
            }
        }

        // Snapshot original other-region yield values for Case B first-if passthrough
        Block *otherBlk = splitThen ? originalIf.elseBlock() : originalIf.thenBlock();
        LDBG("    [Part3.1] snapshot otherBlock=" << (otherBlk ? "yes" : "null"));
        if (otherBlk) {
            auto otherYield = cast<scf::YieldOp>(otherBlk->getTerminator());
            LDBG("    [Part3.1] got otherYield with " << otherYield->getNumOperands() << " operands");
            for (unsigned s = 0; s < N; ++s)
                plan.origOtherYieldValues.push_back(otherYield->getOperand(s));
        }
    }

    unsigned nGroups = splitThen ? c.thenGroups.size() : c.elseGroups.size();
    plan.lastIsVoid = !c.hasYield;
    plan.nResultIfs = plan.lastIsVoid ? nGroups - 1 : nGroups;

    return plan;
}

/// Rewire cross-group SSA uses in a group's ops and nested ifs, then move
/// them into the target block.
static void rewireAndMoveOps(BlockGroup &group,
                             llvm::SmallDenseMap<Value, Value> &crossValueReplacement,
                             Block &targetBlock)
{
    for (auto *op : group.ops) {
        for (auto &operand : op->getOpOperands()) {
            auto it = crossValueReplacement.find(operand.get());
            if (it != crossValueReplacement.end())
                operand.set(it->second);
        }
        // Walk nested regions of non-if region-bearing ops (e.g. scf.for)
        // to rewire cross-group SSA uses inside their bodies.
        if (!isa<scf::IfOp>(op)) {
            op->walk([&](Operation *nestedOp) {
                for (auto &operand : nestedOp->getOpOperands()) {
                    auto it = crossValueReplacement.find(operand.get());
                    if (it != crossValueReplacement.end())
                        operand.set(it->second);
                }
            });
        }
    }
    for (auto nestedIf : group.nestedIfs) {
        for (auto &operand : nestedIf->getOpOperands()) {
            auto it = crossValueReplacement.find(operand.get());
            if (it != crossValueReplacement.end())
                operand.set(it->second);
        }
        nestedIf->walk([&](Operation *nestedOp) {
            for (auto &operand : nestedOp->getOpOperands()) {
                auto it = crossValueReplacement.find(operand.get());
                if (it != crossValueReplacement.end())
                    operand.set(it->second);
            }
        });
    }

    // Merge ops and nestedIfs, then sort by original position to preserve
    // interleaving order before moving into the split-if's then block.
    SmallVector<Operation *> allOps;
    allOps.reserve(group.ops.size() + group.nestedIfs.size());
    allOps.append(group.ops.begin(), group.ops.end());
    for (auto nestedIf : group.nestedIfs)
        allOps.push_back(nestedIf.getOperation());
    llvm::sort(allOps, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
    });
    for (auto *op : allOps)
        op->moveBefore(&targetBlock, targetBlock.end());
}

/// Fill one result-bearing split-if: rewire & move ops to then block,
/// build then+else yields, and update cross-group SSA replacements.
/// oi == split-if position in the chain (also == group index).
/// Step 3.5.2: Build then yield — decide whether to fill true value,
/// passthrough from previous if, or use placeholder for each slot.
static void buildThenYield(
    unsigned oi, bool firstIf,
    const SlotPlan &plan, const YieldAugmentation &ya,
    bool hasYield, bool otherSideHasOps,
    SmallVector<scf::IfOp, 4> &splitIfs,
    llvm::SmallDenseMap<Type, Value> &placeholderCache,
    Operation *placeholderInsertBefore,
    Block &thenBlock, Location loc, OpBuilder &builder)
{
    builder.setInsertionPointToEnd(&thenBlock);
    SmallVector<Value, 4> yieldVals;
    for (unsigned slot = 0; slot < plan.nSlots; ++slot) {
        int action = ya.groupSlotActions[oi][slot];
        if (action == static_cast<int>(oi)) {
            yieldVals.push_back(plan.slotValues[slot]);
        } else if (firstIf && !hasYield) {
            yieldVals.push_back(getOrCreatePlaceholder(
                plan.resultTypes[slot], placeholderCache,
                placeholderInsertBefore, loc, builder));
        } else if (firstIf && hasYield) {
            if (slot < ya.numOriginalSlots && !otherSideHasOps &&
                !plan.origOtherYieldValues.empty())
                yieldVals.push_back(plan.origOtherYieldValues[slot]);
            else
                yieldVals.push_back(getOrCreatePlaceholder(
                    plan.resultTypes[slot], placeholderCache,
                    placeholderInsertBefore, loc, builder));
        } else {
            yieldVals.push_back(splitIfs[oi - 1].getResult(slot));
        }
    }
    builder.create<scf::YieldOp>(loc, yieldVals);
}

/// Step 3.5.3: Build else yield.  The else block is always passthrough —
/// no ops are ever moved into it except for Scene 3/4 (last if absorbs
/// the other side's ops).
static void buildElseYield(
    unsigned oi, bool firstIf, bool isLastIf,
    const SlotPlan &plan, const YieldAugmentation &ya,
    bool hasYield, bool otherSideHasOps,
    ArrayRef<Operation *> otherSideOps,
    ArrayRef<Value> otherSideYieldValues,
    SmallVector<scf::IfOp, 4> &splitIfs,
    Block &elseBlock,
    llvm::SmallDenseMap<Type, Value> &placeholderCache,
    Operation *placeholderInsertBefore,
    Location loc, OpBuilder &builder)
{
    if (elseBlock.mightHaveTerminator())
        elseBlock.getTerminator()->erase();

    builder.setInsertionPointToEnd(&elseBlock);
    SmallVector<Value, 4> yieldVals;

    if (isLastIf && otherSideHasOps) {
        // Scene 3/4: last split-if's else absorbs other side ops.
        for (auto *op : otherSideOps)
            op->moveBefore(&elseBlock, elseBlock.end());
        for (unsigned slot = 0; slot < plan.nSlots; ++slot) {
            if (hasYield && slot < ya.numOriginalSlots &&
                slot < otherSideYieldValues.size())
                yieldVals.push_back(otherSideYieldValues[slot]);
            else
                yieldVals.push_back(getOrCreatePlaceholder(
                    plan.resultTypes[slot], placeholderCache,
                    placeholderInsertBefore, loc, builder));
        }
    } else if (firstIf && !hasYield) {
        for (unsigned slot = 0; slot < plan.nSlots; ++slot)
            yieldVals.push_back(getOrCreatePlaceholder(
                plan.resultTypes[slot], placeholderCache,
                placeholderInsertBefore, loc, builder));
    } else if (firstIf && hasYield) {
        if (!otherSideHasOps && !plan.origOtherYieldValues.empty()) {
            yieldVals = plan.origOtherYieldValues;
            for (unsigned slot = ya.numOriginalSlots; slot < plan.nSlots; ++slot)
                yieldVals.push_back(getOrCreatePlaceholder(
                    plan.resultTypes[slot], placeholderCache,
                    placeholderInsertBefore, loc, builder));
        } else {
            for (unsigned slot = 0; slot < plan.nSlots; ++slot)
                yieldVals.push_back(getOrCreatePlaceholder(
                    plan.resultTypes[slot], placeholderCache,
                    placeholderInsertBefore, loc, builder));
        }
    } else {
        for (unsigned slot = 0; slot < plan.nSlots; ++slot)
            yieldVals.push_back(splitIfs[oi - 1].getResult(slot));
    }
    builder.create<scf::YieldOp>(loc, yieldVals);
}

/// Step 3.5.4: Register the split-if's produced slots so downstream
/// groups can rewire their cross-group SSA uses to the new results.
static void updateCrossValueReplacement(
    unsigned oi,
    const SlotPlan &plan, const YieldAugmentation &ya,
    scf::IfOp splitIf, Block &thenBlock,
    llvm::SmallDenseMap<Value, Value> &crossValueReplacement)
{
    SmallPtrSet<Block *, 4> thenBlocks;
    thenBlock.walk([&](Block *b) { thenBlocks.insert(b); });

    for (unsigned slot = 0; slot < plan.nSlots; ++slot) {
        if (ya.groupSlotActions[oi][slot] == static_cast<int>(oi)) {
            Value oldVal = plan.slotValues[slot];
            if (plan.valueToSlot.count(oldVal)) {
                Value newVal = splitIf.getResult(slot);
                oldVal.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
                    return !thenBlocks.contains(
                        operand.getOwner()->getBlock());
                });
                crossValueReplacement[oldVal] = newVal;
            }
        }
    }
}

/// Fill a single split-if with ops and yield.
/// Orchestrates: rewireAndMoveOps → then yield → else yield → register results.
static void materializeResultIf(
    unsigned oi,
    BlockGroup &group,
    const SlotPlan &plan,
    const CandidateIf &c,
    bool otherSideHasOps,
    bool isLastIf,
    ArrayRef<Operation *> otherSideOps,
    ArrayRef<Value> otherSideYieldValues,
    SmallVector<scf::IfOp, 4> &splitIfs,
    llvm::SmallDenseMap<Value, Value> &crossValueReplacement,
    llvm::SmallDenseMap<Type, Value> &placeholderCache,
    Operation *placeholderInsertBefore,
    Location loc,
    OpBuilder &builder)
{
    auto &ya = c.yieldAug;
    auto splitIf = splitIfs[oi];
    bool firstIf = (oi == 0);

    // Step 3.5.1: Rewire cross-group SSA & move ops into then block
    Block &thenBlock = splitIf.getThenRegion().front();
    if (thenBlock.mightHaveTerminator())
        thenBlock.getTerminator()->erase();
    rewireAndMoveOps(group, crossValueReplacement, thenBlock);

    // Step 3.5.2: Build then yield
    buildThenYield(oi, firstIf, plan, ya, c.hasYield, otherSideHasOps,
                   splitIfs, placeholderCache, placeholderInsertBefore,
                   thenBlock, loc, builder);

    // Step 3.5.3: Build else yield
    if (plan.nSlots > 0) {
        Block &elseBlock = splitIf.getElseRegion().front();
        buildElseYield(oi, firstIf, isLastIf, plan, ya, c.hasYield,
                       otherSideHasOps, otherSideOps, otherSideYieldValues,
                       splitIfs, elseBlock,
                       placeholderCache, placeholderInsertBefore, loc, builder);
    }

    // Step 3.5.4: Update cross-group SSA replacements for downstream groups
    updateCrossValueReplacement(oi, plan, ya, splitIf, thenBlock,
                                crossValueReplacement);
}

/// Create a trailing void split-if for Case A's last group.
/// When hasElse is true (Scene 3/4), the else block absorbs the other side's ops.
static void materializeVoidIf(
    BlockGroup &group,
    scf::IfOp insertAfter,
    Value condition,
    bool hasElse,
    ArrayRef<Operation *> otherSideOps,
    ArrayRef<Value> otherSideYieldValues,
    llvm::SmallDenseMap<Value, Value> &crossValueReplacement,
    Location loc,
    OpBuilder &builder,
    CVPipeline::ComputeBlockIdManager &bm)
{
    LDBG("    [Part3.6] creating void split-if for last group hasElse=" << hasElse);

    builder.setInsertionPointAfter(insertAfter);
    auto voidIf = builder.create<scf::IfOp>(loc, condition, hasElse);
    voidIf->setAttr(CVPipeline::kBlockId,
                    builder.getI32IntegerAttr(bm.getNextId()));
    if (voidIf.getThenRegion().empty())
        voidIf.getThenRegion().emplaceBlock();
    Block &thenBlock = voidIf.getThenRegion().front();
    if (thenBlock.mightHaveTerminator())
        thenBlock.getTerminator()->erase();

    rewireAndMoveOps(group, crossValueReplacement, thenBlock);

    builder.setInsertionPointToEnd(&thenBlock);
    builder.create<scf::YieldOp>(loc);

    // Scene 3/4: else block absorbs other side's ops
    if (hasElse) {
        if (voidIf.getElseRegion().empty())
            voidIf.getElseRegion().emplaceBlock();
        Block &elseBlock = voidIf.getElseRegion().front();
        if (elseBlock.mightHaveTerminator())
            elseBlock.getTerminator()->erase();

        for (auto *op : otherSideOps)
            op->moveBefore(&elseBlock, elseBlock.end());

        builder.setInsertionPointToEnd(&elseBlock);
        builder.create<scf::YieldOp>(loc);
    }
}

// ============================================================================
// Part3 helpers: other-side collection & placeholders
// ============================================================================

struct OtherSideContext {
    bool hasOps;
    SmallVector<Operation *> ops;
    SmallVector<Value> yieldValues;
};

/// Collect ops (and yield values for Case B) from the side that is NOT being
/// split.  In Scene 3/4 the last split-if's else block absorbs them.
static OtherSideContext collectOtherSideInfo(const CandidateIf &c, bool splitThen)
{
    OtherSideContext ctx;
    auto &otherGroups = splitThen ? c.elseGroups : c.thenGroups;
    ctx.hasOps = !otherGroups.empty();
    if (!ctx.hasOps)
        return ctx;

    auto originalIf = c.ifOp;
    for (auto &g : otherGroups) {
        for (auto *op : g.ops)
            ctx.ops.push_back(op);
        for (auto nestedIf : g.nestedIfs)
            ctx.ops.push_back(nestedIf.getOperation());
    }
    llvm::sort(ctx.ops, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
    });

    if (c.hasYield) {
        Block *otherBlk = splitThen ? originalIf.elseBlock()
                                    : originalIf.thenBlock();
        auto otherYield = cast<scf::YieldOp>(otherBlk->getTerminator());
        for (auto v : otherYield->getOperands())
            ctx.yieldValues.push_back(v);
    }
    return ctx;
}

/// Pre-create placeholder values so they dominate every split-if.
/// Case A: all slots need placeholders (original if has no yield).
/// Case B: only augmented slots (indices >= numOriginalSlots) need placeholders;
///   original slots get their passthrough from else values.
/// When the other side has ops, non-last split ifs use placeholders for
///   original slots too (origOtherYieldValues may dangle after erase).
static void preCreatePlaceholders(
    const SlotPlan &plan, const CandidateIf &c,
    bool otherSideHasOps,
    llvm::SmallDenseMap<Type, Value> &cache,
    Operation *insertBefore, Location loc, OpBuilder &builder)
{
    auto &ya = c.yieldAug;
    if (c.hasYield) {
        for (unsigned slot = ya.numOriginalSlots; slot < plan.nSlots; ++slot)
            getOrCreatePlaceholder(plan.resultTypes[slot], cache,
                                   insertBefore, loc, builder);
        if (otherSideHasOps) {
            for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot)
                getOrCreatePlaceholder(plan.resultTypes[slot], cache,
                                       insertBefore, loc, builder);
        }
    } else {
        for (Type t : plan.resultTypes)
            getOrCreatePlaceholder(t, cache, insertBefore, loc, builder);
    }
}

// ============================================================================
// Part3: Materialization (orchestrator)
// ============================================================================

/// Unified materialization for Case A and Case B.
/// Creates a chain of split ifs.  Each if yields the slots it produces
/// and passthroughs the rest from the previous if.
/// Cross-group SSA uses are rewired to previous split if results.
static void materializeCandidate(CandidateIf &c, OpBuilder &builder,
                                  CVPipeline::ComputeBlockIdManager &bm)
{
    auto originalIf = c.ifOp;
    auto loc = originalIf.getLoc();
    Value condition = originalIf.getCondition();
    auto &ya = c.yieldAug;

    LDBG("    [Part3] enter materializeCandidate hasYield=" << c.hasYield);

    // Determine active region
    bool splitThen = c.thenGroups.size() >= 2;
    auto &groups = splitThen ? c.thenGroups : c.elseGroups;
    unsigned nGroups = groups.size();
    if (nGroups < 2) return;

    // Collect other-side ops (scene 3/4: absorbed into last split-if's else)
    auto otherCtx = collectOtherSideInfo(c, splitThen);

    LDBG("    [Part3.1] splitThen=" << splitThen << " nGroups=" << nGroups
         << " otherSideHasOps=" << otherCtx.hasOps);

    // Step 3.1: Compute slot metadata
    SlotPlan plan = buildSlotPlan(c, splitThen);

    LDBG("    [Part3.1] nSlots=" << plan.nSlots << " nGroups=" << nGroups
         << " nResultIfs=" << plan.nResultIfs << " lastIsVoid=" << plan.lastIsVoid);

    builder.setInsertionPoint(originalIf);

    // Step 3.2: Negate condition when splitting else — ops originally in
    // the else branch (condition=false) are placed in the split if's then
    // branch, so the condition must be inverted to preserve semantics.
    if (!splitThen) {
        auto trueVal = builder.create<arith::ConstantOp>(
            loc, builder.getIntegerAttr(builder.getI1Type(), 1));
        condition = builder.create<arith::XOrIOp>(loc, condition, trueVal).getResult();
    }

    // Step 3.3: Pre-create placeholders (dominate every split-if)
    llvm::SmallDenseMap<Type, Value> placeholderCache;
    preCreatePlaceholders(plan, c, otherCtx.hasOps, placeholderCache,
                          originalIf, loc, builder);

    // Step 3.4: Create empty split-if shells
    SmallVector<scf::IfOp, 4> splitIfs;
    for (unsigned gi = 0; gi < plan.nResultIfs; ++gi) {
        auto newIf = builder.create<scf::IfOp>(loc, plan.resultTypes, condition,
                                               /*hasElse=*/plan.nSlots > 0);
        newIf->setAttr(CVPipeline::kBlockId,
                        builder.getI32IntegerAttr(bm.getNextId()));
        if (newIf.getThenRegion().empty())
            newIf.getThenRegion().emplaceBlock();
        if (plan.nSlots > 0 && newIf.getElseRegion().empty())
            newIf.getElseRegion().emplaceBlock();
        splitIfs.push_back(newIf);
    }

    // Step 3.5: Fill each result-bearing split if
    llvm::SmallDenseMap<Value, Value> crossValueReplacement;
    for (unsigned gi = 0; gi < plan.nResultIfs; ++gi) {
        bool isLastResultIf = (gi == plan.nResultIfs - 1) && !plan.lastIsVoid;
        materializeResultIf(gi, groups[gi],
                            plan, c,
                            otherCtx.hasOps, isLastResultIf,
                            otherCtx.ops, otherCtx.yieldValues,
                            splitIfs,
                            crossValueReplacement, placeholderCache,
                            originalIf, loc, builder);
    }

    // Step 3.6: Last void split-if (Case A only)
    if (plan.lastIsVoid) {
        materializeVoidIf(groups.back(),
                          splitIfs.back(), condition,
                          otherCtx.hasOps,
                          otherCtx.ops, otherCtx.yieldValues,
                          crossValueReplacement, loc, builder, bm);
    }

    // Step 3.7: Replace original if uses & erase
    if (c.hasYield) {
        auto lastIf = splitIfs.back();
        for (unsigned ri = 0; ri < ya.numOriginalSlots; ++ri)
            originalIf.getResult(ri).replaceAllUsesWith(lastIf.getResult(ri));
    }

    originalIf->erase();
}

/// Count the number of ancestor scf::IfOps; used to sort outermost-first so
/// that nested-if groups still hold valid pointers when an outer candidate
/// is materialized.
static unsigned ifDepth(Operation *op) {
    unsigned d = 0;
    for (auto *parent = op->getParentOp(); parent;
         parent = parent->getParentOp())
        if (isa<scf::IfOp>(parent))
            ++d;
    return d;
}

static void materializeCandidates(SmallVector<CandidateIf> &candidates, bool &changed,
                                   CVPipeline::ComputeBlockIdManager &bm)
{
    if (candidates.empty())
        return;

    // Process outermost first: when the inner if is moved as a whole into an
    // outer split-if, its internal ops stay valid.  If we did inner first,
    // its originalIf would be erased, leaving dangling pointers in the outer
    // candidate's group.nestedIfs.
    llvm::stable_sort(candidates, [](const CandidateIf &a, const CandidateIf &b) {
        return ifDepth(a.ifOp) < ifDepth(b.ifOp);
    });

    OpBuilder builder(candidates[0].ifOp.getContext());

    for (auto &c : candidates) {
        LDBG("    [Part3] materializing candidate (hasYield=" << c.hasYield << ")");
        materializeCandidate(c, builder, bm);
    }

    changed = true;
}

// ============================================================================
// Pass entry point
// ============================================================================

void SplitIfByBlockIdPass::runOnOperation()
{
    ModuleOp module = getOperation();
    LDBG("SplitIfByBlockIdPass entered.");

    // Dump the pre-split IR in debug builds
    LDBG("//===--- Before SplitIfByBlockId ---\n" << module);
    LDBG("//===--- End Before SplitIfByBlockId ---");

    bool changed = true;
    for (unsigned iteration = 1; changed; ++iteration) {
        changed = false;

        // Part1: Discovery & Grouping
        auto candidates = discoverCandidates(module);
        LDBG("  iter=" << iteration
             << "  candidates=" << candidates.size());

        if (candidates.empty())
            break;

        dumpCandidates(candidates);

        // Part2: Dependency Analysis & Yield Planning
        analyzeDependencies(candidates);

        // Part3: Materialization
        CVPipeline::ComputeBlockIdManager bm(module);
        materializeCandidates(candidates, changed, bm);
    }

    LDBG("SplitIfByBlockIdPass completed.");
}

// ============================================================================
// Pass registration
// ============================================================================

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSplitIfByBlockIdPass()
{
    return std::make_unique<SplitIfByBlockIdPass>();
}

void registerSplitIfByBlockIdPasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createSplitIfByBlockIdPass();
    });
}

void SplitIfByBlockIdPass::getDependentDialects(DialectRegistry &registry) const
{
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
}

} // namespace triton
} // namespace mlir
