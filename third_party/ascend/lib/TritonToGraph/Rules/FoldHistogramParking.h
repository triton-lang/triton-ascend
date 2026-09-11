// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_ASCEND_GRAPH_FOLD_HISTOGRAM_PARKING_H
#define TRITON_ASCEND_GRAPH_FOLD_HISTOGRAM_PARKING_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

// Ascend histogram templates ignore out-of-range indices, including -1.
// hist(select(mask, x, 0)) - select(bin_id == 0, sum(!mask), 0)
// therefore equals hist(select(mask, x, -1)). Keep the histogram unmasked so
// the existing SIMD implementation remains available; do not change its ABI.
struct FoldHistogramParking : OpRewritePattern<arith::SubIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SubIOp op,
                                PatternRewriter &rewriter) const override {
    auto hist = op.getLhs().getDefiningOp<HistogramOp>();
    auto correction = op.getRhs().getDefiningOp<arith::SelectOp>();
    // Do not duplicate an expensive histogram used elsewhere.
    if (!hist || !hist->hasOneUse() || hist->getNumOperands() != 1 ||
        !correction || !matchPattern(correction.getFalseValue(), m_Zero()))
      return failure();
    auto inputType = dyn_cast<RankedTensorType>(hist.getSrc().getType());
    auto binsType = dyn_cast<RankedTensorType>(hist.getType());
    if (!inputType || inputType.getRank() != 1 ||
        !inputType.getElementType().isInteger(32) || !binsType ||
        binsType.getRank() != 1 || !binsType.getElementType().isInteger(32) ||
        binsType.getDimSize(0) <= 0)
      return failure();
    auto input = hist.getSrc().getDefiningOp<arith::SelectOp>();
    if (!input || !matchPattern(input.getFalseValue(), m_Zero()))
      return failure();

    auto isZero = correction.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!isZero || isZero.getPredicate() != arith::CmpIPredicate::eq)
      return failure();
    Value ids = isZero.getLhs();
    if (matchPattern(ids, m_Zero()))
      ids = isZero.getRhs();
    else if (!matchPattern(isZero.getRhs(), m_Zero()))
      return failure();
    auto range = ids.getDefiningOp<MakeRangeOp>();
    if (!range || range.getStart() != 0 ||
        range.getEnd() != binsType.getDimSize(0))
      return failure();

    auto countSplat = correction.getTrueValue().getDefiningOp<SplatOp>();
    if (!countSplat)
      return failure();
    auto count = countSplat.getSrc().getDefiningOp<ReduceOp>();
    if (!count || count->getNumOperands() != 1 || count->getNumResults() != 1 ||
        count.getAxis() != 0)
      return failure();
    Block &body = count->getRegion(0).front();
    if (body.getNumArguments() != 2 ||
        !llvm::hasSingleElement(body.without_terminator()))
      return failure();
    auto add = dyn_cast<arith::AddIOp>(body.front());
    if (!add || body.getTerminator()->getNumOperands() != 1 ||
        body.getTerminator()->getOperand(0) != add.getResult() ||
        !((add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1)) ||
          (add.getLhs() == body.getArgument(1) &&
           add.getRhs() == body.getArgument(0))))
      return failure();

    Value counted = count->getOperand(0);
    if (auto reshape = counted.getDefiningOp<ReshapeOp>())
      counted = reshape.getSrc();
    auto extend = counted.getDefiningOp<arith::ExtUIOp>();
    if (!extend)
      return failure();
    auto invert = extend.getIn().getDefiningOp<arith::XOrIOp>();
    if (!invert || !((invert.getLhs() == input.getCondition() &&
                      matchPattern(invert.getRhs(), m_One())) ||
                     (invert.getRhs() == input.getCondition() &&
                      matchPattern(invert.getLhs(), m_One()))))
      return failure();

    Value invalid = rewriter.create<arith::ConstantOp>(
        op.getLoc(), inputType,
        DenseElementsAttr::get(inputType, rewriter.getI32IntegerAttr(-1)));
    Value indices = rewriter.create<arith::SelectOp>(
        op.getLoc(), input.getCondition(), input.getTrueValue(), invalid);
    // Preserve attributes (e.g. source provenance). Other uses of count remain
    // intact; the original histogram becomes dead after replacing subtraction.
    Operation *replacement = rewriter.clone(*hist.getOperation());
    replacement->setOperand(0, indices);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};
} // namespace mlir::triton
#endif
