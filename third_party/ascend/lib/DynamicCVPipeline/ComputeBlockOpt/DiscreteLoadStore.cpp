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

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

static constexpr const char *DEBUG_TYPE = "DiscreteLoadStore";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace CVPipeline;

// ============================================================================
// Function Name: traceToDefVal
// ============================================================================
/**
 * @brief Iteratively traces an aliased value back to its original definition.
 *
 * **Purpose**:
 * To locate the ultimate root definition of a value by continuously unwrapping alias-introducing operations.
 *
 * **Inputs & Assumptions**:
 * - `value` (mlir::Value): The starting value.
 * - Assumptions: The SSA graph is acyclic, ensuring the trace loop will always terminate.
 *
 * **Outputs & Guarantees**:
 * - Returns the root `mlir::Value` that is not defined by any recognizable alias-introducing operation.
 * - Guarantees the returned value is non-null if the input is non-null.
 *
 * **Safety Boundaries & Constraints**:
 * - Gracefully returns the last resolved value if a step in the chain results in a null value or has no defining op.
 */
static Value traceToDefVal(Value value)
{
    while (auto source = getAliasSource(value)) {
        value = source;
    }

    return value;
}

// ============================================================================
// Function Name: isGMArg
// ============================================================================
/**
 * @brief Checks if a value is an argument to the entry block of a function (representing Global Memory).
 *
 * **Purpose**:
 * To distinguish global inputs/outputs (GM arguments) from intermediate buffers allocated locally.
 *
 * **Inputs & Assumptions**:
 * - `value` (mlir::Value): The value to check.
 *
 * **Outputs & Guarantees**:
 * - Returns `true` if the value is a block argument belonging to the entry block of a `func::FuncOp`.
 * - Returns `false` otherwise.
 *
 * **Safety Boundaries & Constraints**:
 * - Safe against values that are not block arguments, or block arguments residing in nested region blocks.
 */
static bool isGMArg(Value value)
{
    auto blockArg = llvm::dyn_cast_if_present<BlockArgument>(value);
    if (!blockArg) {
        return false;
    }
    auto *ownerBlock = blockArg.getOwner();
    auto funcOp = llvm::dyn_cast_or_null<func::FuncOp>(ownerBlock->getParentOp());
    return funcOp && &funcOp.getFunctionBody().front() == ownerBlock;
}

namespace {

enum class MemEffectType { LOAD_FROM_GM, STORE_TO_GM };

struct AnalysisResult {
    MemEffectType type;
    Operation *localOp;
};
}

static llvm::StringLiteral literalEffectType(std::optional<MemEffectType> effectType)
{
    if (effectType == MemEffectType::LOAD_FROM_GM) {
        return "LOAD";
    }
    if (effectType == MemEffectType::STORE_TO_GM) {
        return "STORE";
    }

    return "NULL";
}

// ============================================================================
// Function Name: analyseMemOp
// ============================================================================
/**
 * @brief Analyzes a memory-effect-bearing operation to classify it as a discrete load or store.
 *
 * **Purpose**:
 * To identify if an operation acts as a transfer bridge between Global Memory (GM) and local buffers.
 *
 * **Inputs & Assumptions**:
 * - `memOp` (MemoryEffectOpInterface): The memory operation to analyze.
 *
 * **Outputs & Guarantees**:
 * - Returns a populated `std::optional<AnalysisResult>` containing the transfer direction (LOAD/STORE)
 *   and the local buffer defining operation if a consistent GM-to-local (or local-to-GM) pattern is found.
 * - Returns `std::nullopt` if the memory effects are inconsistent, null, or do not represent a discrete transfer.
 *
 * **Safety Boundaries & Constraints**:
 * - Restricts memory effect collections using a local buffer of size `kExpectedMaxMemEffects`.
 * - Ignores memory effects that do not have a concrete target value.
 */
static std::optional<AnalysisResult> analyseMemOp(MemoryEffectOpInterface memOp)
{
    LOG_DEBUG("Analysing: " << *memOp.getOperation() << "\n");
    Operation *localOp = nullptr;
    std::optional<MemEffectType> localType;
    std::optional<MemEffectType> gmType;

    constexpr size_t kExpectedMaxMemEffects = 2;
    llvm::SmallVector<MemoryEffects::EffectInstance, kExpectedMaxMemEffects> memEffects;
    memOp.getEffects(memEffects);
    for (auto effectInstance : memEffects) {
        auto target = effectInstance.getValue();
        if (!target) {
            continue;
        }

        auto defVal = traceToDefVal(target);
        LOG_DEBUG("\n====== target ======\n" << target << "\n====== defVal ======\n" << defVal << "\n");
        if (isGMArg(defVal)) {
            if (gmType.has_value()) {
                continue;
            }
            if (isa<MemoryEffects::Write>(effectInstance.getEffect())) {
                gmType = MemEffectType::STORE_TO_GM;
            } else if (isa<MemoryEffects::Read>(effectInstance.getEffect())) {
                gmType = MemEffectType::LOAD_FROM_GM;
            }
            continue;
        }

        // source is some local operation
        if (localType.has_value()) {
            continue;
        }
        auto *defOp = defVal.getDefiningOp();
        if (!defOp) {
            continue;
        }
        localOp = defOp;
        if (isa<MemoryEffects::Write>(effectInstance.getEffect()) && isa<memref::AllocOp>(defOp)) {
            localType = MemEffectType::LOAD_FROM_GM;
        } else if (isa<MemoryEffects::Read>(effectInstance.getEffect())) {
            localType = MemEffectType::STORE_TO_GM;
        }
    }

    LOG_DEBUG("GM Type: " << literalEffectType(gmType) << "; LocalType: " << literalEffectType(localType) << "\n");
    if (!(gmType.has_value() && localType.has_value() && gmType == localType && localOp != nullptr)) {
        return std::nullopt;
    }

    return AnalysisResult {localType.value(), localOp};
}

static llvm::FailureOr<AnalysisResult> analyseExtractedLoadStore(scf::ForOp forOp)
{
    if (!forOp->hasAttr(hivm::ExtractLoadStoreAttr)) {
        return llvm::failure();
    }
    std::optional<AnalysisResult> resultOpt;
    auto walkResult = forOp->walk([&](MemoryEffectOpInterface memOp) {
        auto currResult = analyseMemOp(memOp);
        if (!currResult.has_value()) {
            return WalkResult::advance();
        }
        if (resultOpt.has_value()) {
            return WalkResult::interrupt();
        }
        resultOpt = currResult;
        return WalkResult::advance();
    });
    if (walkResult.wasInterrupted() || !resultOpt.has_value()) {
        LOG_DEBUG("expected exactly one single mem op from/to gm in " << forOp << "\n");
        return llvm::failure();
    }

    return resultOpt.value();
}

// ============================================================================
// Function Name: getCommonBlockId
// ============================================================================
/**
 * @brief Checks if a list of operations all share the exact same block ID.
 *
 * **Purpose**:
 * To determine if a set of operations is already clustered together into the same scheduling group.
 *
 * **Inputs & Assumptions**:
 * - `ops` (llvm::ArrayRef<Operation *>): The operations to examine.
 *
 * **Outputs & Guarantees**:
 * - Returns `std::optional<int64_t>` containing the block ID if all operations share the same ID.
 * - Returns `std::nullopt` if there is a block ID mismatch, or if any operation lacks an assigned block ID.
 *
 * **Safety Boundaries & Constraints**:
 * - Safe to execute with empty lists.
 */
static std::optional<int> getCommonBlockId(llvm::ArrayRef<Operation *> ops)
{
    std::optional<int> blockIdOpt;
    for (auto *op : ops) {
        auto currBlockId = getOpBlockId(op);
        if (!blockIdOpt.has_value()) {
            blockIdOpt = currBlockId;
        }
        if (currBlockId.has_value() && currBlockId != blockIdOpt) {
            return std::nullopt;
        }
    }
    return blockIdOpt;
}

// ============================================================================
// Function Name: applyFixes
// ============================================================================
/**
 * @brief Coordinates block ID updates for discrete load/store loops and their dependencies.
 *
 * **Purpose**:
 * To unify the block IDs of the loop structure, its local buffer, and optionally its upstream
 * condition-calculating operations to match the block ID of the external users.
 *
 * **Inputs & Assumptions**:
 * - `forOp` (scf::ForOp): The loop operation executing the load/store.
 * - `analysis` (AnalysisResult): Structural analysis of the memory transfer.
 * - `memGraph` (const MemoryDependenceGraph &): The memory dependence graph for safety analysis.
 * - `bm` (ComputeBlockIdManager &): The block ID manager.
 *
 * **Outputs & Guarantees**:
 * - Returns `llvm::success()` if block IDs were successfully unified (either with or without upstreams).
 * - Returns `llvm::failure()` if no target block ID could be determined.
 *
 * **Safety Boundaries & Constraints**:
 * - Attempts a progressive update: first tries to update loop operations together with upstream condition
 *   predecessors; if that fails due to cycles, falls back to updating only the core loop operations.
 */
static llvm::LogicalResult applyFixes(scf::ForOp forOp,
                                      AnalysisResult analysis,
                                      const MemoryDependenceGraph &memGraph,
                                      ComputeBlockIdManager &bm)
{
    auto *localOp = analysis.localOp;

    // store / no users - follow defOp
    auto targetBlockId = getOpBlockId(localOp);

    // load - follow users
    if (analysis.type == MemEffectType::LOAD_FROM_GM) {
        auto users = llvm::to_vector(
            llvm::make_filter_range(localOp->getUsers(), [&](Operation *user) { return !forOp->isAncestor(user); }));
        if (!users.empty()) {
            targetBlockId = getCommonBlockId(users);
            if (!targetBlockId.has_value()) {
                LOG_DEBUG("Users do not share the same block id: " << *localOp << "\n");
                return llvm::failure();
            }
        }
    }

    if (!targetBlockId.has_value()) {
        LOG_DEBUG("No users and alloc does not have block id: " << *localOp << "\n");
        return llvm::failure();
    }

    // We attempt a two-tier progressive update to maximize optimization while ensuring safety:
    //
    // 1. Extended Ops (Core + Upstreams): Unifying upstream condition-calculating operations
    //    or loop bounds with the core operations is highly preferred. Keeping them in the
    //    same block reduces inter-block dependencies and synchronization overhead, leading
    //    to faster execution and a smaller memory/register footprint.
    //
    // 2. Core Ops Only (Fallback): Extended operations (like shared loop bounds or conditional
    //    checks) are often shared among multiple independent structures. Forcing them into
    //    the same block can create dependency cycles. If that happens, we fall back to
    //    unifying only the core operations, which is the minimum requirement for correctness.
    llvm::SmallVector<Operation *> coreOps {localOp};
    forOp->walk([&](Operation *op) { coreOps.push_back(op); });

    SmallVector<Operation *> extendedOps = CVPipeline::collectBlockPredecessors(
        {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()}, forOp->getBlock());
    extendedOps.append(coreOps);

    if (llvm::succeeded(tryUpdate(extendedOps, memGraph, targetBlockId.value(), bm))) {
        LOG_DEBUG("Successfully applied fixes to users and for upstreams: " << *localOp << "\n");
        return llvm::success();
    }

    if (llvm::succeeded(tryUpdate(coreOps, memGraph, targetBlockId.value(), bm))) {
        LOG_DEBUG("Successfully applied fixes to users: " << *localOp << "\n");
        return llvm::success();
    }

    // The core operations represent the absolute minimum functional unit that must reside
    // within the same block for correctness (e.g., the local buffer allocation, its write/fill
    // operations, and its direct users).
    //
    // If even this minimal set cannot be unified without introducing dependency cycles,
    // it indicates a fundamental structural conflict or an unsupported IR pattern.
    // In this case, we must abort the unification process and trigger a fallback.
    LOG_DEBUG("Failed to apply fixes to required ops: " << *localOp << "\n");
    return llvm::failure();
}

namespace {

// ============================================================================
// Class: DiscreteLoadStore
// ============================================================================
/**
 * @class DiscreteLoadStore
 * @brief Optimization pass targeting discrete memory transfer loops.
 *
 * **Purpose**:
 * Analyzes `scf.for` loops marked for load/store extraction and unifies their block IDs with
 * local buffers and external users to prevent execution bottlenecks.
 *
 * **Inputs & Assumptions**:
 * - Operates on a `ModuleOp` containing operations from the SCF, MemRef, and Func dialects.
 * - Assumes the existence of `AliasAnalysis` to build accurate memory dependency graphs.
 *
 * **Outputs & Guarantees**:
 * - Guarantees that scheduling block IDs are updated in place for valid discrete transfer loops.
 * - Guarantees that the resulting block-level graph is cycle-free, preserving correctness.
 *
 * **Safety Boundaries & Constraints**:
 * - If analysis fails, reports a pass failure and emits compiler errors on the offending loop operation.
 */
class DiscreteLoadStore : public PassWrapper<DiscreteLoadStore, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DiscreteLoadStore)
    DiscreteLoadStore() = default;

    StringRef getArgument() const override { return "discrete-load-store"; }

    void runOnOperation() override
    {
        ModuleOp moduleOp = getOperation();
        LOG_DEBUG("Before: " << moduleOp << "\n");
        auto &aa = getAnalysis<AliasAnalysis>();
        MemoryDependenceGraph memGraph(moduleOp, aa);
        auto bm = ComputeBlockIdManager(moduleOp);

        auto result = moduleOp->walk([&](scf::ForOp forOp) {
            if (!forOp->hasAttr(hivm::ExtractLoadStoreAttr)) {
                return WalkResult::advance();
            }

            auto analysisResult = analyseExtractedLoadStore(forOp);
            if (llvm::failed(analysisResult)) {
                forOp.emitError() << "\n[" << DEBUG_TYPE << "] Analysis failed";
                return WalkResult::interrupt();
            }

            if (llvm::failed(applyFixes(forOp, analysisResult.value(), memGraph, bm))) {
                forOp.emitError() << "\n[" << DEBUG_TYPE << "] Block Id fixes failed (unrecoverable)";
                return WalkResult::interrupt();
            }

            return WalkResult::advance();
        });

        if (result.wasInterrupted()) {
            signalPassFailure();
            return;
        }

        LOG_DEBUG("After: " << moduleOp << "\n");
    }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createDiscreteLoadStorePass()
{
    return std::make_unique<DiscreteLoadStore>();
}

void registerDiscreteLoadStorePass()
{
    registerPass(createDiscreteLoadStorePass);
}

} // namespace triton
} // namespace mlir
