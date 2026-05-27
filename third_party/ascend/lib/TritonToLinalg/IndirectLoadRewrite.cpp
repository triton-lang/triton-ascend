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

#include "TritonToLinalg/IndirectLoadRewrite.h"
#include "TritonToLinalg/ImplicitPermute.h"
#include "TritonToStructured/PtrAnalysis.h"
#include "Utils/Utils.h"

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#include "llvm/Support/Debug.h"

#include <cstdlib>

#define DEBUG_TYPE "triton-to-linalg-indirect-load-rewrite"

namespace IndirectLoadRewrite {

using namespace mlir;
using namespace triton;

namespace {

// V1 fast-path supports up to 5D tensors, mirroring
// UnstructureConversionPass::tryRewriteIndirectFastPath.
constexpr size_t kFastPathRankLimit = 5;

// Returns true iff `v` is a static integer constant with |v| > 1.
static bool isStaticConstAbsGtOne(Value v) {
    IntegerAttr scalarAttr;
    if (matchPattern(v, m_Constant(&scalarAttr)))
        return std::abs(scalarAttr.getValue().getSExtValue()) > 1;
    DenseElementsAttr denseAttr;
    if (matchPattern(v, m_Constant(&denseAttr)) && denseAttr.isSplat() &&
        denseAttr.getElementType().isInteger())
        return std::abs(denseAttr.getSplatValue<llvm::APInt>().getSExtValue()) > 1;
    return false;
}

// Lightweight pre-check: walks the offset's defining-op tree (bounded depth,
// staying within tensor-typed values) looking for any arith.muli whose result
// is a tensor and one operand is a static constant with |c| > 1. Returns
// false if no such per-element multiplication exists, in which case the
// per-element stride must be 1 and we should NOT invoke the heavier
// PtrAnalysis (which mutates IR via the rewriter; calling it before we
// commit to rewriting would violate MLIR's pattern contract -- the greedy
// driver would treat our return-failure() as a real change and loop until
// max iterations, failing the PassManager).
//
// Crucially, we do NOT recurse through scalar values: scalar arithmetic
// (e.g. `xoffset = pid * BLOCK_SIZE`) does not affect per-element stride;
// only tensor-level multiplications do. Without this restriction, kernels
// that compute a scalar block offset by multiplying by the block size would
// be incorrectly flagged as "possibly stride>1".
static bool offsetMayContainStrideGtOne(Value offset, int depthBudget = 16) {
    if (depthBudget <= 0) {
        // Give up cheaply and let PtrAnalysis decide downstream.
        return true;
    }
    // Scalar-typed values can't carry per-element stride information.
    if (!isa<RankedTensorType>(offset.getType())) {
        return false;
    }
    Operation *defOp = offset.getDefiningOp();
    if (!defOp) {
        return false;
    }
    if (auto mul = dyn_cast<arith::MulIOp>(defOp)) {
        if (isStaticConstAbsGtOne(mul.getLhs()) ||
            isStaticConstAbsGtOne(mul.getRhs())) {
            return true;
        }
        return offsetMayContainStrideGtOne(mul.getLhs(), depthBudget - 1) ||
               offsetMayContainStrideGtOne(mul.getRhs(), depthBudget - 1);
    }
    for (Value operand : defOp->getOperands()) {
        if (offsetMayContainStrideGtOne(operand, depthBudget - 1)) {
            return true;
        }
    }
    return false;
}

// Walk through tt.splat to find the underlying scalar !tt.ptr<T>.
// Returns null if the base is not a simple splat of a scalar pointer.
static Value getScalarBasePtr(Value tensorPtr) {
    if (auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>()) {
        Value src = splatOp.getSrc();
        if (isa<triton::PointerType>(src.getType())) {
            return src;
        }
    }
    return Value();
}

// Ensure the per-element offset tensor has i64 element type, matching the
// convention used elsewhere (UnstructureConversionPass::parseAddPtr).
static Value ensureI64OffsetTensor(Value offsetTensor, Location loc,
                                   PatternRewriter &rewriter) {
    auto tensorTy = dyn_cast<RankedTensorType>(offsetTensor.getType());
    if (!tensorTy) return Value();
    auto eltTy = dyn_cast<IntegerType>(tensorTy.getElementType());
    if (!eltTy) return Value();
    if (eltTy.getWidth() == 64) return offsetTensor;
    auto newTy = RankedTensorType::get(tensorTy.getShape(),
                                       rewriter.getIntegerType(64));
    return rewriter.create<arith::ExtSIOp>(loc, newTy, offsetTensor);
}

}  // namespace

LogicalResult LoadConverter::matchAndRewrite(triton::LoadOp op,
                                             PatternRewriter &rewriter) const {
    auto loc = op.getLoc();

    // ---- Re-entry / cross-step guards ----
    // Already inspected by this sub-step and chosen not to rewrite -- bail
    // immediately to avoid re-running PtrAnalysis (which mutates IR via the
    // rewriter and would otherwise drive the greedy pattern driver into a
    // re-application loop).
    if (op->hasAttr(InspectedByIndirectLoadRewriteTAG)) {
        return failure();
    }
    // Already produced by this sub-step.
    if (op->hasAttr(RewrittenByIndirectLoadRewriteTAG)) {
        return failure();
    }
    // Already handled by ImplicitPermute — strict zero-overlap contract.
    if (op->hasAttr(ImplicitPermute::ImplicitPermuteHandledTAG)) {
        return failure();
    }
    // Already marked as discrete by an earlier pass (e.g. UnstructureConversion
    // scalar-fallback). Don't double-handle.
    if (op->hasAttr(mlir::ConverterUtils::discreteAttrName)) {
        return failure();
    }

    // ---- Source op shape restrictions (V1) ----
    // V1 only handles the canonical "tt.load (tt.addptr %splat, %offsets)"
    // pattern. make_tensor_ptr / advance / chained-addptr / iter-arg-ptr
    // cases bail and continue through the legacy strided memref.copy lowering.
    auto addPtrOp = op.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!addPtrOp) {
        return failure();
    }
    // boundary_check belongs to make_tensor_ptr loads; bail for safety.
    if (!op.getBoundaryCheck().empty()) {
        return failure();
    }
    // The base must be a simple tt.splat of a scalar pointer.
    Value scalarBase = getScalarBasePtr(addPtrOp.getPtr());
    if (!scalarBase) {
        return failure();
    }

    // ---- Result shape / rank limit ----
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) {
        // Scalar load -- not our domain.
        return failure();
    }
    if (resultType.getShape().size() > kFastPathRankLimit) {
        return failure();
    }

    // ---- Cheap pre-filter ----
    // PtrAnalysis::visitOperand mutates IR through the rewriter; if we run it
    // and then return failure(), the greedy driver treats that as a successful
    // rewrite and re-walks the same op, triggering an infinite re-application
    // loop that terminates as PassManager::run failed. So before invoking
    // PtrAnalysis we cheaply prove "no static stride > 1 can possibly exist"
    // by inspecting the offset SSA chain for any arith.muli by a static
    // constant > 1. No such multiplication means stride == 1 and we exit
    // without touching IR.
    if (!offsetMayContainStrideGtOne(addPtrOp.getOffset())) {
        return failure();
    }

    // ---- Stride analysis: reuse PtrAnalysis (same machinery as ImplicitPermute) ----
    // From this point on, PtrAnalysis may insert helper IR via the rewriter.
    // Every early-out path below MUST stamp InspectedByIndirectLoadRewriteTAG
    // on the load and return success() so the greedy driver does not re-walk
    // the same op (which would re-run PtrAnalysis and accumulate dead IR
    // until maxIterations -> PassManager::run failed).
    TritonToStructured::PtrAnalysis ptrAnalysis;
    TritonToStructured::PtrState ptrState;
    auto markInspectedAndReturn = [&]() {
        op->setAttr(InspectedByIndirectLoadRewriteTAG,
                    UnitAttr::get(rewriter.getContext()));
        return success();
    };
    if (ptrAnalysis.visitOperand(op.getPtr(), ptrState, loc, rewriter).failed()) {
        return markInspectedAndReturn();
    }
    if (ptrState.stateInfo.empty()) {
        return markInspectedAndReturn();
    }
    // Defensive: corner case where ImplicitPermute's gate skipped a permuted
    // load (compileOn91095Flag && !existDotFlag && SIMT). Leave it to the
    // legacy strided memref.copy path.
    ptrState.analyzePermute();
    if (ptrState.isPermuted) {
        return markInspectedAndReturn();
    }

    // ---- Trigger condition: static |last_stride| > 1 ----
    auto lastStrideOpt = getConstantIntValue(ptrState.stateInfo.back().stride);
    if (!lastStrideOpt.has_value()) {
        // Dynamic last-axis stride: Lazy default, do not rewrite.
        return markInspectedAndReturn();
    }
    int64_t lastStride = std::abs(lastStrideOpt.value());
    if (lastStride <= 1) {
        return markInspectedAndReturn();
    }
    // Yield priority to DeinterleaveStatusOptimization for the stride==2
    // even-size case (vectorized strided copy + extract_slice is faster than
    // a SIMT gather).
    if (lastStride == 2) {
        auto lastShapeOpt = getConstantIntValue(ptrState.stateInfo.back().shape);
        if (lastShapeOpt.has_value() && lastShapeOpt.value() % 2 == 0) {
            return markInspectedAndReturn();
        }
    }

    // ---- Emit tt.indirect_load ----
    Value offsetTensor = ensureI64OffsetTensor(addPtrOp.getOffset(), loc, rewriter);
    if (!offsetTensor) {
        return failure();
    }
    Value mask = op.getMask();
    Value other = op.getOther();
    auto indirectLoad = rewriter.create<triton::ascend::IndirectLoadOp>(
        loc, resultType, scalarBase, offsetTensor, mask, other);
    indirectLoad->setAttr(RewrittenByIndirectLoadRewriteTAG,
                          UnitAttr::get(rewriter.getContext()));

    LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "----------------------------------------------\n";
        os << "IndirectLoadRewrite: tt.load -> tt.indirect_load\n";
        os << "  last_stride = " << lastStride << "\n";
        os << "  result type = " << resultType << "\n";
        os << indirectLoad << "\n";
        os << "----------------------------------------------\n";
    });

    rewriter.replaceOp(op, indirectLoad.getResult());
    return success();
}

}  // namespace IndirectLoadRewrite
