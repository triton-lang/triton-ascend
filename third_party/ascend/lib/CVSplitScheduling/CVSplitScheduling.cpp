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

#include "ascend/include/CVSplitScheduling/CVSplitScheduling.h"
#include "ascend/include/CVSplitScheduling/CrossScopeTransfers.h"
#include "ascend/include/CVSplitScheduling/DependencyScheduler.h"
#include "ascend/include/CVSplitScheduling/PreCheck.h"
#include "ascend/include/CVSplitScheduling/ScopeSeparation.h"
#include "ascend/include/CVSplitScheduling/UnfusePVMatmuls.h"
#include "ascend/include/CVSplitScheduling/UnrollOrigin.h"
#include "ascend/include/CVSplitScheduling/classifyAllOps.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <queue>

#define DEBUG_TYPE "cv-split-scheduling"

using namespace mlir;
using namespace mlir::triton;

// ============================================================================
// CV-Split Scheduling
// ----------------------------------------------------------------------------
// Splits the innermost loop of a fused kernel into two co-running engine scopes
// — a CUBE scope (matmul / fixpipe) and a VECTOR scope (elementwise / softmax) —
// with explicit cross-engine buffers and synchronization, so the Ascend cube
// and vector units overlap instead of running serially.
//
// Pipeline (driven by CVSplitSchedulingPass::processFunction):
//   1.  findInnermostLoop            locate the loop to split
//   1b. hasStoresInBody              bail if the body already stores (not fusable)
//   2.  loopUnrollByFactor           unroll by `unroll-factor` to expose ILP
//   3.  classifyAllOps               tag each op CUBE or VECTOR (matmul-seeded,
//                                    data-feeders pulled into CUBE, rest VECTOR)
//   4-7 DependencyScheduler          graph -> BFS levels -> reorder so
//                                    same-engine work is contiguous
//   7.5 unfusePVMatmuls              undo matmul(p,v,acc) fusion that entangles
//                                    the engines
//   8.  insertCrossScopeTransfers    materialize C->V (fixpipe->UB) and V->C
//                                    (NZ pack->L1) buffers + sync_block_set/wait
//   9.  createScopeSeparation        clone CUBE ops into a CUBE scope, wrap the
//                                    loop+epilogue in a VECTOR scope, strip the
//                                    wrong-engine ops from each, then ROW_SPLIT
//                                    re-tile the VECTOR scope across both veccores
//
// Generality: the pass is engine-pattern driven, not kernel-name driven — it
// keys off op semantics (matmul == CUBE, float elementwise == VECTOR) and bails
// out cleanly (leaving the IR untouched) whenever an assumption does not hold:
// no innermost loop, stores already present, unroll-factor <= 1, no CUBE ops, or
// CUBE/VECTOR work that the level scheduler finds entangled. Flash-Attention is
// the validated driver kernel; other fused cube+vector loops that satisfy the
// same structural contract are handled by the same path, and anything else
// falls through unmodified.
// ============================================================================

namespace {

using cv_split::kUnrollOriginIdAttrName;

static void tagUnrollOriginIds(scf::ForOp loop)
{
    Builder builder(loop.getContext());
    int64_t originId = 0;
    for (Operation &op : *loop.getBody()) {
        if (isa<scf::YieldOp>(op))
            continue;
        op.setAttr(kUnrollOriginIdAttrName, builder.getI64IntegerAttr(originId++));
    }
}

static void removeUnrollOriginIdAttrs(Operation *root)
{
    root->walk([](Operation *op) { op->removeAttr(kUnrollOriginIdAttrName); });
}

// loopUnrollByFactor clones the tensor.empty destination of a transpose once
// per unrolled lane.  These destinations carry no data, and every transpose
// overwrites the complete tensor before its immediately following matmul reads
// it.  Keeping the clones makes bufferization materialize one UB allocation per
// lane, while the hand-unrolled reference deliberately reuses one destination.
// Fold only clones of the same original empty (identified by the temporary
// unroll-origin ID), and only when every use is the destination of a transpose.
static void reuseUnrolledTransposeDestinations(scf::ForOp loop)
{
    DenseMap<int64_t, DenseMap<Type, tensor::EmptyOp>> canonicalByOrigin;
    SmallVector<tensor::EmptyOp> empties;
    for (Operation &op : *loop.getBody())
        if (auto empty = dyn_cast<tensor::EmptyOp>(op))
            empties.push_back(empty);

    for (tensor::EmptyOp empty : empties) {
        auto origin = empty->getAttrOfType<IntegerAttr>(kUnrollOriginIdAttrName);
        if (!origin)
            continue;

        bool isTransposeDestination = !empty.getResult().use_empty();
        for (OpOperand &use : empty.getResult().getUses()) {
            auto transpose = dyn_cast<linalg::TransposeOp>(use.getOwner());
            if (!transpose || transpose.getDpsInitOperand(0)->get() != empty.getResult()) {
                isTransposeDestination = false;
                break;
            }
        }
        if (!isTransposeDestination)
            continue;

        auto &byType = canonicalByOrigin[origin.getInt()];
        auto [it, inserted] = byType.try_emplace(empty.getType(), empty);
        if (inserted)
            continue;

        empty.getResult().replaceAllUsesWith(it->second.getResult());
        empty.erase();
    }
}

// Unrolling also clones tensor-semantics fills that initialize reductions.
// A zero-filled tensor is immutable, so clones of the same original fill can
// share one result exactly as the hand-unrolled kernel does.  Restrict this to
// fills whose results are used only as reduction init operands; other fills may
// represent lane-specific state and must remain independent.
static void reuseUnrolledReductionInitializers(scf::ForOp loop)
{
    DenseMap<int64_t, DenseMap<Type, linalg::FillOp>> canonicalByOrigin;
    SmallVector<linalg::FillOp> fills;
    for (Operation &op : *loop.getBody())
        if (auto fill = dyn_cast<linalg::FillOp>(op))
            fills.push_back(fill);

    for (linalg::FillOp fill : fills) {
        auto origin = fill->getAttrOfType<IntegerAttr>(kUnrollOriginIdAttrName);
        if (!origin || fill->getNumResults() != 1)
            continue;

        Value result = fill->getResult(0);
        bool isReductionInitializer = !result.use_empty();
        for (OpOperand &use : result.getUses()) {
            auto reduce = dyn_cast<linalg::ReduceOp>(use.getOwner());
            if (!reduce || reduce.getDpsInitOperand(0)->get() != result) {
                isReductionInitializer = false;
                break;
            }
        }
        if (!isReductionInitializer)
            continue;

        auto &byType = canonicalByOrigin[origin.getInt()];
        auto [it, inserted] = byType.try_emplace(result.getType(), fill);
        if (inserted)
            continue;

        linalg::FillOp canonical = it->second;
        Value fillValue = fill.getInputs().front();
        Value canonicalValue = canonical.getInputs().front();
        bool sameFillValue = fillValue == canonicalValue;
        if (!sameFillValue) {
            auto fillConstant = fillValue.getDefiningOp<arith::ConstantOp>();
            auto canonicalConstant = canonicalValue.getDefiningOp<arith::ConstantOp>();
            sameFillValue = fillConstant && canonicalConstant &&
                            fillConstant.getValue() == canonicalConstant.getValue();
        }
        if (!sameFillValue)
            continue;

        result.replaceAllUsesWith(canonical->getResult(0));
        fill.erase();
    }
}

// Keep immutable tensor templates outside the physical-program loop.  The
// hand-unrolled FA kernel exposes its accumulator initializers in that form:
// the template is filled once and each program iteration receives a cheap UB
// copy.  CV splitting otherwise leaves the same invariant fills in the parent
// loop body, so bufferization emits vector fill functions on every program.
//
// Restrict the motion to direct children of the immediately enclosing loop,
// pure tensor fills with static tensor.empty destinations, and scalar fill
// values defined outside that loop.  Reusing tensor.empty destinations of the
// same type is valid because tensor.empty carries no data and every linalg.fill
// returns a new tensor value.
static void hoistInvariantTensorFillTemplates(scf::ForOp outerLoop)
{
    if (!outerLoop)
        return;

    SmallVector<linalg::FillOp> fills;
    for (Operation &op : *outerLoop.getBody()) {
        auto fill = dyn_cast<linalg::FillOp>(op);
        if (!fill || fill->getNumResults() != 1 || !isa<RankedTensorType>(fill->getResult(0).getType()))
            continue;

        Value fillValue = fill.getInputs().front();
        Operation *fillValueDef = fillValue.getDefiningOp();
        if (!fillValueDef)
            continue;

        // The tensor template and its scalar splat are both loop invariant.
        // Constants commonly remain next to the fill until scope separation;
        // move only that trivially safe producer.  Any other producer inside
        // the physical-program loop rejects the transformation.
        bool moveFillValue = outerLoop->isAncestor(fillValueDef);
        if (moveFillValue &&
            (!isa<arith::ConstantOp>(fillValueDef) || fillValueDef->getBlock() != outerLoop.getBody()))
            continue;

        auto empty = fill.getOutputs().front().getDefiningOp<tensor::EmptyOp>();
        if (!empty || empty->getBlock() != outerLoop.getBody() || !empty.getDynamicSizes().empty())
            continue;
        fills.push_back(fill);
    }

    DenseMap<Type, tensor::EmptyOp> canonicalEmptyByType;
    for (linalg::FillOp fill : fills) {
        Operation *fillValueDef = fill.getInputs().front().getDefiningOp();
        if (outerLoop->isAncestor(fillValueDef))
            fillValueDef->moveBefore(outerLoop);

        auto empty = fill.getOutputs().front().getDefiningOp<tensor::EmptyOp>();
        Type type = empty.getType();
        auto [it, inserted] = canonicalEmptyByType.try_emplace(type, empty);
        if (inserted) {
            empty->moveBefore(outerLoop);
        } else {
            fill.getDpsInitOperand(0)->set(it->second.getResult());
            if (empty.getResult().use_empty())
                empty.erase();
        }
        fill->moveBefore(outerLoop);
    }
}

struct LinearUnrolledAddress {
    arith::AddIOp address;
    Value outerBase;
    Value loopCarriedBase;
    int64_t logicalOffset;
    int64_t elementStride;
};

static FailureOr<std::pair<Value, int64_t>> peelConstantAddChain(Value value)
{
    Value base = value;
    int64_t offset = 0;
    while (auto add = base.getDefiningOp<arith::AddIOp>()) {
        std::optional<int64_t> lhs = getConstantIntValue(add.getLhs());
        std::optional<int64_t> rhs = getConstantIntValue(add.getRhs());
        Value nextBase;
        int64_t increment;
        if (lhs) {
            nextBase = add.getRhs();
            increment = *lhs;
        } else if (rhs) {
            nextBase = add.getLhs();
            increment = *rhs;
        } else
            break;

        int64_t nextOffset;
        if (llvm::AddOverflow(offset, increment, nextOffset))
            return failure();
        offset = nextOffset;
        base = nextBase;
    }
    return std::make_pair(base, offset);
}

static bool isNonNegativeLoopCarriedValue(Value value, scf::ForOp loop)
{
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != loop.getBody() || argument.getArgNumber() == 0)
        return false;

    unsigned iterArgIndex = argument.getArgNumber() - 1;
    if (iterArgIndex >= loop.getInitArgs().size())
        return false;
    std::optional<int64_t> initial = getConstantIntValue(loop.getInitArgs()[iterArgIndex]);
    if (!initial || *initial < 0)
        return false;

    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    Value next = yield.getOperand(iterArgIndex);
    FailureOr<std::pair<Value, int64_t>> nextBaseAndOffset = peelConstantAddChain(next);
    if (failed(nextBaseAndOffset))
        return false;
    return nextBaseAndOffset->first == value && nextBaseAndOffset->second >= 0;
}

static FailureOr<std::pair<Value, int64_t>> matchLoopBaseAndOffset(Value value, scf::ForOp loop)
{
    if (auto maximum = value.getDefiningOp<arith::MaxSIOp>()) {
        std::optional<int64_t> lhs = getConstantIntValue(maximum.getLhs());
        std::optional<int64_t> rhs = getConstantIntValue(maximum.getRhs());
        if (lhs && *lhs == 0)
            value = maximum.getRhs();
        else if (rhs && *rhs == 0)
            value = maximum.getLhs();
        else
            return failure();
    }

    // loopUnrollByFactor builds a recurrence chain rather than independent
    // `base + laneOffset` expressions:
    //
    //   lane1 = base + step
    //   lane2 = lane1 + step
    //   lane3 = lane2 + step
    //
    // Peel the complete constant chain so every cloned address is related to
    // the same loop-carried base.  A single-add matcher recognizes lane 1 but
    // rejects all later lanes and therefore leaves the expensive cast/multiply
    // chain intact.
    FailureOr<std::pair<Value, int64_t>> baseAndOffset = peelConstantAddChain(value);
    if (failed(baseAndOffset))
        return failure();
    Value base = baseAndOffset->first;
    int64_t offset = baseAndOffset->second;

    if (offset < 0 || !isNonNegativeLoopCarriedValue(base, loop))
        return failure();
    return std::make_pair(base, offset);
}

static FailureOr<LinearUnrolledAddress> matchLinearUnrolledAddress(arith::AddIOp address, scf::ForOp loop)
{
    arith::MulIOp multiply = address.getLhs().getDefiningOp<arith::MulIOp>();
    Value outerBase = address.getRhs();
    if (!multiply) {
        multiply = address.getRhs().getDefiningOp<arith::MulIOp>();
        outerBase = address.getLhs();
    }
    if (!multiply)
        return failure();

    Value castValue;
    std::optional<int64_t> stride = getConstantIntValue(multiply.getLhs());
    if (stride)
        castValue = multiply.getRhs();
    else {
        stride = getConstantIntValue(multiply.getRhs());
        castValue = multiply.getLhs();
    }
    if (!stride || *stride <= 0)
        return failure();

    auto indexCast = castValue.getDefiningOp<arith::IndexCastOp>();
    if (!indexCast || !isa<IndexType>(indexCast.getType()))
        return failure();
    FailureOr<std::pair<Value, int64_t>> baseAndOffset = matchLoopBaseAndOffset(indexCast.getIn(), loop);
    if (failed(baseAndOffset))
        return failure();

    return LinearUnrolledAddress{address, outerBase, baseAndOffset->first, baseAndOffset->second, *stride};
}

// Triton block-pointer advances become scalar address arithmetic before this
// pass runs.  Generic loop unrolling consequently clones the complete
// `(state + lane) * stride + base` chain for every lane.  The hand-unrolled
// kernel instead computes the first address once and issues later DMA loads at
// constant offsets from it.  Recreate that form only for address expressions
// with the same outer base, loop-carried base, and element stride, and only
// after proving that the loop-carried state and all lane offsets are
// nonnegative (so stripping the max-with-zero is exact).
static FailureOr<scf::ForOp> strengthReduceUnrolledAddresses(scf::ForOp loop)
{
    SmallVector<LinearUnrolledAddress> canonicals;
    SmallVector<arith::AddIOp> addresses;
    for (Operation &op : *loop.getBody())
        if (auto add = dyn_cast<arith::AddIOp>(op))
            addresses.push_back(add);

    for (arith::AddIOp address : addresses) {
        FailureOr<LinearUnrolledAddress> match = matchLinearUnrolledAddress(address, loop);
        if (failed(match))
            continue;

        auto it = llvm::find_if(canonicals, [&](const LinearUnrolledAddress &candidate) {
            return candidate.outerBase == match->outerBase &&
                   candidate.loopCarriedBase == match->loopCarriedBase &&
                   candidate.elementStride == match->elementStride;
        });
        if (it == canonicals.end()) {
            canonicals.push_back(*match);
            continue;
        }
        LinearUnrolledAddress &canonical = *it;
        if (!canonical.address->isBeforeInBlock(match->address.getOperation()))
            continue;

        int64_t logicalDelta;
        int64_t elementDelta;
        if (llvm::SubOverflow(match->logicalOffset, canonical.logicalOffset, logicalDelta) ||
            llvm::MulOverflow(logicalDelta, match->elementStride, elementDelta))
            continue;

        OpBuilder builder(address);
        Value replacement = canonical.address.getResult();
        if (elementDelta != 0) {
            Value delta = builder.create<arith::ConstantIndexOp>(address.getLoc(), elementDelta);
            replacement = builder.create<arith::AddIOp>(address.getLoc(), replacement, delta);
        }
        address.getResult().replaceAllUsesWith(replacement);
        address.erase();
    }

    // Carry the first physical address of each proven linear group through the
    // loop. The generic unroller otherwise recomputes
    // `max(logical, 0) -> index_cast -> logical * stride + outerBase` for the
    // first K/V load of every unrolled iteration. The hand-unrolled kernel
    // carries the already-scaled index and advances it once per iteration.
    struct PhysicalRecurrence {
        LinearUnrolledAddress address;
        Value initialAddress;
        int64_t elementIncrement;
    };
    SmallVector<PhysicalRecurrence> recurrences;
    OpBuilder initBuilder(loop);
    for (const LinearUnrolledAddress &canonical : canonicals) {
        auto argument = cast<BlockArgument>(canonical.loopCarriedBase);
        unsigned iterArgIndex = argument.getArgNumber() - 1;
        Value initialLogical = loop.getInitArgs()[iterArgIndex];
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        FailureOr<std::pair<Value, int64_t>> nextBaseAndOffset =
            peelConstantAddChain(yield.getOperand(iterArgIndex));
        if (failed(nextBaseAndOffset) || nextBaseAndOffset->first != canonical.loopCarriedBase ||
            nextBaseAndOffset->second <= 0 || !isa<IndexType>(canonical.outerBase.getType()))
            continue;

        int64_t physicalIncrement;
        if (llvm::MulOverflow(nextBaseAndOffset->second, canonical.elementStride, physicalIncrement))
            continue;

        Value initialIndex = initialLogical;
        if (!isa<IndexType>(initialIndex.getType()))
            initialIndex = initBuilder.create<arith::IndexCastOp>(loop.getLoc(), initBuilder.getIndexType(),
                                                                  initialLogical);
        Value scaledInitial = initialIndex;
        if (canonical.elementStride != 1) {
            Value stride = initBuilder.create<arith::ConstantIndexOp>(loop.getLoc(), canonical.elementStride);
            scaledInitial = initBuilder.create<arith::MulIOp>(loop.getLoc(), initialIndex, stride);
        }
        Value initialAddress =
            initBuilder.create<arith::AddIOp>(loop.getLoc(), scaledInitial, canonical.outerBase);
        recurrences.push_back({canonical, initialAddress, physicalIncrement});
    }

    if (recurrences.empty())
        return loop;

    SmallVector<Value> newInitArgs(loop.getInitArgs());
    for (const PhysicalRecurrence &recurrence : recurrences)
        newInitArgs.push_back(recurrence.initialAddress);

    OpBuilder builder(loop);
    auto newLoop = builder.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
                                               loop.getStep(), newInitArgs);
    newLoop->setAttrs(loop->getAttrs());

    IRMapping mapping;
    mapping.map(loop.getInductionVar(), newLoop.getInductionVar());
    for (auto [oldArg, newArg] : llvm::zip(loop.getRegionIterArgs(), newLoop.getRegionIterArgs().take_front(
                                                                            loop.getRegionIterArgs().size())))
        mapping.map(oldArg, newArg);

    Block *newBody = newLoop.getBody();
    Operation *newYield;
    if (newBody->empty()) {
        builder.setInsertionPointToEnd(newBody);
        newYield = builder.create<scf::YieldOp>(loop.getLoc(), newLoop.getRegionIterArgs());
    } else {
        newYield = &newBody->back();
    }
    builder.setInsertionPoint(newYield);
    for (Operation &op : loop.getBody()->without_terminator())
        builder.clone(op, mapping);

    SmallVector<Value> newYields;
    auto oldYield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (Value value : oldYield.getOperands())
        newYields.push_back(mapping.lookupOrDefault(value));

    unsigned oldIterArgCount = loop.getRegionIterArgs().size();
    for (auto [index, recurrence] : llvm::enumerate(recurrences)) {
        BlockArgument physicalAddress = newLoop.getRegionIterArgs()[oldIterArgCount + index];
        Value increment = builder.create<arith::ConstantIndexOp>(loop.getLoc(), recurrence.elementIncrement);
        newYields.push_back(builder.create<arith::AddIOp>(loop.getLoc(), physicalAddress, increment));

        Value clonedAddress = mapping.lookup(recurrence.address.address.getResult());
        clonedAddress.replaceAllUsesWith(physicalAddress);
    }
    cast<scf::YieldOp>(newYield).getResultsMutable().assign(newYields);

    for (auto [oldResult, newResult] : llvm::zip(loop.getResults(), newLoop.getResults().take_front(loop.getNumResults())))
        oldResult.replaceAllUsesWith(newResult);
    loop.erase();
    return newLoop;
}

static void commitModuleClone(ModuleOp destination, ModuleOp source)
{
    Operation *destinationOp = destination.getOperation();
    Operation *sourceOp = source.getOperation();

    destinationOp->setLoc(sourceOp->getLoc());
    destinationOp->setAttrs(sourceOp->getAttrs());
    if (destinationOp->getPropertiesStorageSize() != 0)
        destinationOp->copyProperties(sourceOp->getPropertiesStorage());
    destination.getBodyRegion().takeBody(source.getBodyRegion());
}

static void commitFunctionClone(func::FuncOp destination, func::FuncOp source)
{
    Operation *destinationOp = destination.getOperation();
    Operation *sourceOp = source.getOperation();

    destinationOp->setLoc(sourceOp->getLoc());
    destinationOp->setAttrs(sourceOp->getAttrs());
    if (destinationOp->getPropertiesStorageSize() != 0)
        destinationOp->copyProperties(sourceOp->getPropertiesStorage());
    destination.getBody().takeBody(source.getBody());
}

struct FunctionBackup {
    explicit FunctionBackup(func::FuncOp function) : function(function), backup(function.clone()) {}

    func::FuncOp function;
    OwningOpRef<func::FuncOp> backup;
};

struct CandidateState {
    FunctionBackup *functionBackup;
    scf::ForOp loop;
};

static void restoreFunction(FunctionBackup &state)
{
    commitFunctionClone(state.function, *state.backup);
}

static void restoreAndRefreshFunctionBackup(FunctionBackup &state)
{
    restoreFunction(state);
    state.backup = OwningOpRef<func::FuncOp>(state.function.clone());
}
// ============================================================================
// Pass entry point
// ============================================================================
class CVSplitSchedulingPass : public ::impl::CVSplitSchedulingBase<CVSplitSchedulingPass> {
  public:
    explicit CVSplitSchedulingPass(const CVSplitSchedulingOptions &options)
    {
        this->compileOn91095 = options.compileOn91095;
        this->unrollFactor = options.unrollFactor;
    }

    void runOnOperation() override
    {
        if (!compileOn91095) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Not A5 target, skipping\n");
            return;
        }

        ModuleOp moduleOp = getOperation();
        LLVM_DEBUG(llvm::dbgs() << "\n[cv-split] ============================\n"
                                << "[cv-split]  CVSplitScheduling START\n"
                                << "[cv-split]  unrollFactor=" << unrollFactor << "\n"
                                << "[cv-split] ============================\n\n");

        // Run the transformation transactionally on one clone of the input module.
        // Each function has its own backup, so a failed candidate can be restored
        // without discarding successful candidates in other functions.
        OwningOpRef<ModuleOp> transformedModule = moduleOp.clone();
        SmallVector<FunctionBackup> functionBackups;
        for (func::FuncOp funcOp : transformedModule->getOps<func::FuncOp>())
            functionBackups.emplace_back(funcOp);

        SmallVector<CandidateState> candidates = prepareCandidates(functionBackups);
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Functions: " << functionBackups.size()
                                << ", prepared candidates: " << candidates.size() << "\n");
        if (candidates.empty()) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] No candidate found; keeping original IR\n");
            return;
        }

        // Stage 3: DCVP remains unchanged and classifies the whole working module
        // exactly once. Non-candidate functions are restored immediately afterward
        // so classifier-side rewrites cannot leak into them.
        if (failed(cv_split::runDCVPClassifier(*transformedModule))) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] DCVP classification failed; keeping original "
                                       "IR\n");
            return;
        }
        restoreNonCandidates(functionBackups, candidates);

        // Stages 4 onward: finish every prepared candidate independently. A failed
        // function is restored from its original backup and processing continues.
        if (!processCandidates(candidates)) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] No candidate transformed; keeping original "
                                       "IR\n");
            return;
        }

        // Safety cleanup for functions that returned before the normal Stage 8
        // cleanup point.
        removeUnrollOriginIdAttrs(*transformedModule);
        cv_split::removeDCVPClassificationAttrs(*transformedModule);

        if (failed(verify(*transformedModule))) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Transformed IR failed verification; keeping "
                                       "original IR\n");
            return;
        }

        commitModuleClone(moduleOp, *transformedModule);

        LLVM_DEBUG(llvm::dbgs() << "\n[cv-split] ============================\n"
                                << "[cv-split]  CVSplitScheduling END\n"
                                << "[cv-split] ============================\n\n");
    }

  private:
    SmallVector<CandidateState> prepareCandidates(MutableArrayRef<FunctionBackup> functionBackups)
    {
        SmallVector<CandidateState> candidates;
        for (FunctionBackup &state : functionBackups) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Function: " << state.function.getName() << "\n");
            FailureOr<scf::ForOp> preCheckResult = cv_split::preCheckCVSplitScheduling(state.function, unrollFactor);
            if (failed(preCheckResult)) {
                LLVM_DEBUG(llvm::dbgs() << "[cv-split] Pre-check rejected function, skip\n");
                continue;
            }

            scf::ForOp candidateLoop = *preCheckResult;
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Pre-check accepted candidate loop\n");
            if (failed(unrollCandidateLoop(candidateLoop))) {
                LLVM_DEBUG(llvm::dbgs() << "[cv-split] Candidate preparation failed; trying "
                                           "next function\n");
                restoreAndRefreshFunctionBackup(state);
                continue;
            }

            candidates.push_back({&state, candidateLoop});
        }
        return candidates;
    }

    static void restoreNonCandidates(MutableArrayRef<FunctionBackup> functionBackups,
                                     ArrayRef<CandidateState> candidates)
    {
        llvm::DenseSet<Operation *> candidateFunctions;
        for (const CandidateState &candidate : candidates)
            candidateFunctions.insert(candidate.functionBackup->function);

        for (FunctionBackup &state : functionBackups)
            if (!candidateFunctions.contains(state.function))
                restoreFunction(state);
    }

    bool processCandidates(MutableArrayRef<CandidateState> candidates)
    {
        bool transformedAnyCandidate = false;
        for (CandidateState &candidate : candidates) {
            FunctionBackup &state = *candidate.functionBackup;
            if (failed(processFunction(state.function, candidate.loop)) || failed(verify(state.function))) {
                LLVM_DEBUG(llvm::dbgs() << "[cv-split] Candidate failed; restoring function and "
                                           "trying next function\n");
                restoreFunction(state);
                continue;
            }
            transformedAnyCandidate = true;
        }
        return transformedAnyCandidate;
    }

    LogicalResult unrollCandidateLoop(scf::ForOp &loop)
    {
        tagUnrollOriginIds(loop);

        // Stage 2: Unroll the innermost loop
        LogicalResult unrollResult = loopUnrollByFactor(loop, unrollFactor);
        if (failed(unrollResult)) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Unroll failed, bail\n");
            return failure();
        }
        reuseUnrolledTransposeDestinations(loop);
        reuseUnrolledReductionInitializers(loop);
        FailureOr<scf::ForOp> reducedLoop = strengthReduceUnrolledAddresses(loop);
        if (failed(reducedLoop))
            return failure();
        loop = *reducedLoop;
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Unrolled by " << unrollFactor << "\n");
        return success();
    }

    LogicalResult processFunction(func::FuncOp funcOp, scf::ForOp loop)
    {
        scf::ForOp outerLoop = loop->getParentOfType<scf::ForOp>();
        Block *body = loop.getBody();

        // Stage 3: Import the classifications stamped by the single module-level
        // DCVP classifier invocation in runOnOperation().
        FailureOr<cv_split::Classification> classificationResult = cv_split::readDCVPClassification(body);
        if (failed(classificationResult)) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Failed to read DCVP classification, bail\n");
            return failure();
        }
        cv_split::Classification classification = std::move(*classificationResult);
        if (!cv_split::checkCoreClassifications(body, classification)) {
            LLVM_DEBUG(llvm::dbgs() << "[cv-split] Loop must contain both CUBE and VECTOR ops, "
                                       "skip\n");
            return failure();
        }

        // Stages 4-7: build the dependency graph, assign BFS levels, verify the
        // CUBE/VECTOR work is cleanly separable, and reorder the body by level.
        cv_split::DependencyScheduler scheduler;
        llvm::DenseMap<Operation *, Operation *> transferPhaseEnds;
        if (failed(scheduler.run(body, classification, transferPhaseEnds)))
            return failure();

        // Stage 7.5: Unfuse PV matmuls (split matmul(p,v,acc*alpha) into pv + addf)
        if (failed(cv_split::unfusePVMatmuls(body, classification)))
            return failure();

        // Stage 8: Insert cross-scope transfers (BEFORE scope separation)
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] === Stage 8: cross-scope transfers ===\n");
        FailureOr<cv_split::CrossScopeTransferInfo> transferInfo =
            cv_split::insertCrossScopeTransfers(loop, classification, transferPhaseEnds);
        if (failed(transferInfo)) {
            return failure();
        }
        // Origin IDs are temporary unroll-lineage metadata. Transfer grouping is
        // their final consumer, so do not expose them to scope/backend passes.
        removeUnrollOriginIdAttrs(funcOp);
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Stage 8 complete\n");

        // Stage 9: Scope separation (like DynamicCVPipeline/SeparateCVScope)
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] === Stage 9: scope separation ===\n");
        if (failed(cv_split::createScopeSeparation(funcOp, loop, *transferInfo))) {
            return failure();
        }
        hoistInvariantTensorFillTemplates(outerLoop);
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Stage 9 complete\n");

        // Stage 10: Ensure function has mix_mode attribute (it should already)
        // Note: do NOT add hivm.func_core_type=MIX — that triggers SplitMixKernel
        // which conflicts with our already-scoped IR. The scope::ScopeOp attrs +
        // mix_mode="mix" are sufficient for BiShengIR to handle the scopes.
        if (!funcOp->hasAttr("mix_mode"))
            funcOp->setAttr("mix_mode", StringAttr::get(funcOp.getContext(), "mix"));
        LLVM_DEBUG(llvm::dbgs() << "[cv-split] Function attributes set on " << funcOp.getName() << "\n");

        // The pass has already split the M dimension across both vector
        // sub-blocks and rebuilt every cross-core/output view with the
        // sub-block index.  The generic TileAndBindSubBlock fallback interprets
        // a disabled auto-tiler as a 1:1 kernel and wraps every UB->L1 copy and
        // GM store in `sub_block_id == 0`.  That drops the second ROW_SPLIT
        // half.  Skip only that later transformation; AutoBlockifyParallelLoop
        // is independent and must still form the physical-block loop.
        if (auto moduleOp = funcOp->getParentOfType<ModuleOp>())
            moduleOp->setAttr("hivm.disable_auto_tile_and_bind_subblock", UnitAttr::get(funcOp.getContext()));

        return success();
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCVSplitSchedulingPass(const CVSplitSchedulingOptions &options)
{
    return std::make_unique<CVSplitSchedulingPass>(options);
}
