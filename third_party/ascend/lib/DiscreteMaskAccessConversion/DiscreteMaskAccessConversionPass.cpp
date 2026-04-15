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

#include "Utils/Utils.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"

#include "ascend/include/TritonToLinalg/MaskAnalysis.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DISCRETEMASKACCESSCONVERSION
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

LogicalResult isDiscreteMask(Operation *op, Value mask, PatternRewriter &rewriter)
{
    if (!mask)
        return failure();

    MaskState mstate;
    auto isContMask = mstate.parse(mask, op->getLoc(), rewriter);
    if (!isContMask.failed()) {
        mstate.eraseInsertedOps(op, rewriter);
        return failure();
    }
    return success();
}

struct DiscreteMaskStoreConversion : OpRewritePattern<triton::StoreOp> {
    using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::StoreOp op, PatternRewriter &rewriter) const final
    {
        auto mask = op.getMask();
        auto loc = op.getLoc();
        auto dst = op.getPtr();
        auto src = op.getValue();

        if (failed(isDiscreteMask(op, mask, rewriter)))
            return failure();

        auto loadFromDstOp = rewriter.create<triton::LoadOp>(loc, dst, op.getCache(), op.getEvict(), false);

        auto selOp = rewriter.create<arith::SelectOp>(loc, mask, src, loadFromDstOp.getResult());
        auto newStore = rewriter.create<triton::StoreOp>(loc, dst, selOp, op.getCache(), op.getEvict());
        newStore->setAttr(ConverterUtils::discreteMaskAttrName, UnitAttr::get(rewriter.getContext()));
        rewriter.replaceOp(op, newStore);
        return success();
    }
};

struct DiscreteMaskLoadConversion : OpRewritePattern<triton::LoadOp> {
    using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::LoadOp op, PatternRewriter &rewriter) const final
    {
        auto loc = op.getLoc();
        auto other = op.getOther();
        auto mask = op.getMask();
        auto ptr = op.getPtr();

        if (failed(isDiscreteMask(op, mask, rewriter)))
            return failure();
        if (compileOn91095Flag && forceSimtTemplateFlag)
            return failure();

        if (!other) {
            FailureOr<Value> constant =
                specializeTypelessValueToConstant(TypelessValue::Zero, ptr.getType(), loc, rewriter);
            if (failed(constant))
                llvm_unreachable("Unsupported type for constant creation");
            other = *constant;
        }

        auto newLoadOp = rewriter.create<triton::LoadOp>(loc, ptr, op.getCache(), op.getEvict(), op.getIsVolatile());
        auto discreteMaskOp = rewriter.create<arith::SelectOp>(loc, mask, newLoadOp, other);
        rewriter.replaceOp(op, discreteMaskOp);
        return success();
    }
};

struct DiscreteMaskAtomicConversion : OpRewritePattern<mlir::triton::AtomicRMWOp> {
    using OpRewritePattern<mlir::triton::AtomicRMWOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::triton::AtomicRMWOp op, PatternRewriter &rewriter) const final
    {
        auto loc = op.getLoc();
        auto ptr = op.getPtr();
        auto src = op.getVal();
        auto mask = op.getMask();
        RMWOp rmwOp = op.getAtomicRmwOp();

        if (failed(isDiscreteMask(op, mask, rewriter)))
            return failure();

        const std::map<RMWOp, TypelessValue> initMap = {
            {RMWOp::FADD, TypelessValue::Zero}, {RMWOp::ADD, TypelessValue::Zero},
            {RMWOp::UMAX, TypelessValue::Zero}, {RMWOp::OR, TypelessValue::Zero},
            {RMWOp::MIN, TypelessValue::Max},   {RMWOp::UMIN, TypelessValue::Max},
            {RMWOp::AND, TypelessValue::Max},   {RMWOp::MAX, TypelessValue::Min},
            {RMWOp::XOR, TypelessValue::Zero},  {RMWOp::XCHG, TypelessValue::Undefined},
        };
        assert(initMap.find(rmwOp) != initMap.end());
        auto typelessVal = initMap.at(rmwOp);
        if (typelessVal == TypelessValue::Undefined) {
            // Undefined default value atomic op will be decomposed in AscendNPU-IR
            op->setAttr(ConverterUtils::discreteMaskAttrName, UnitAttr::get(rewriter.getContext()));
            return failure();
        }

        FailureOr<mlir::Value> fill = specializeTypelessValueToConstant(typelessVal, src.getType(), loc, rewriter);
        if (failed(fill))
            op->emitError("Unsupported atomic operation.");

        auto maskedValue = rewriter.create<arith::SelectOp>(loc, mask, src, *fill);
        auto newAtomicOp = rewriter.create<mlir::triton::AtomicRMWOp>(loc, src.getType(), rmwOp, ptr, maskedValue,
                                                                      mlir::Value(), op.getSem(), op.getScope());
        rewriter.replaceOp(op, newAtomicOp);
        return success();
    }
};

DiscreteMaskAccessConversionPass::DiscreteMaskAccessConversionPass(const DiscreteMaskAccessConversionOptions &options)
    : DiscreteMaskAccessConversionBase(options)
{
}

void DiscreteMaskAccessConversionPass::runOnOperation()
{
    compileOn91095Flag = this->compileOn91095;
    forceSimtTemplateFlag = this->forceSimtTemplate;

    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<DiscreteMaskLoadConversion, DiscreteMaskStoreConversion, DiscreteMaskAtomicConversion>(
        patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
        moduleOp->emitError("failed to apply discrete mask access patterns");
        signalPassFailure();
    }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createDiscreteMaskAccessConversionPass(const DiscreteMaskAccessConversionOptions &options)
{
    return std::make_unique<DiscreteMaskAccessConversionPass>(options);
}
