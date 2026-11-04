/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "TritonToLinalg/ArgMinMaxConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <cstdint>
#include <limits>

using namespace mlir;

namespace {
// A unique matching lane turns a full-vector max into a guarded extraction.
// Keep max(payload[winner], fill): every other lane contributes the fill value.
struct FoldOneHotPayload : OpRewritePattern<triton::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAxis() != 0 || op.getNumOperands() != 1 ||
        op.getNumResults() != 1 || !op->getResult(0).getType().isInteger(32))
      return failure();
    auto select = op.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!select)
      return failure();
    auto type = dyn_cast<RankedTensorType>(select.getType());
    if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
        type.getDimSize(0) <= 1 ||
        type.getDimSize(0) > std::numeric_limits<int32_t>::max() ||
        type.getEncoding() || !type.getElementType().isInteger(32))
      return failure();
    Block *body = op.getBody();
    if (body->getNumArguments() != 2 || body->getOperations().size() != 2)
      return failure();
    auto max = dyn_cast<arith::MaxSIOp>(body->front());
    auto yield = dyn_cast<triton::ReduceReturnOp>(body->getTerminator());
    if (!max || !yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != max.getResult() ||
        !((max.getLhs() == body->getArgument(0) &&
           max.getRhs() == body->getArgument(1)) ||
          (max.getLhs() == body->getArgument(1) &&
           max.getRhs() == body->getArgument(0))))
      return failure();

    auto cmp = select.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
      return failure();
    auto range = cmp.getLhs().getDefiningOp<triton::MakeRangeOp>();
    auto splat = cmp.getRhs().getDefiningOp<triton::SplatOp>();
    if (!range || !splat) {
      range = cmp.getRhs().getDefiningOp<triton::MakeRangeOp>();
      splat = cmp.getLhs().getDefiningOp<triton::SplatOp>();
    }
    if (!range || !splat || range.getStart() != 0 ||
        range.getEnd() != type.getDimSize(0) ||
        !splat.getSrc().getType().isInteger(32))
      return failure();
    auto fillOp = select.getFalseValue().getDefiningOp<arith::ConstantOp>();
    if (!fillOp)
      return failure();
    auto fillAttr = dyn_cast<DenseIntElementsAttr>(fillOp.getValue());
    if (!fillAttr || !fillAttr.isSplat())
      return failure();

    Location loc = op.getLoc();
    Value winner = splat.getSrc();
    Value fill = rewriter.create<arith::ConstantIntOp>(
        loc, fillAttr.getSplatValue<APInt>().getSExtValue(), 32);
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value size =
        rewriter.create<arith::ConstantIntOp>(loc, type.getDimSize(0), 32);
    Value lower = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 winner, zero);
    Value upper = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 winner, size);
    Value inRange = rewriter.create<arith::AndIOp>(loc, lower, upper);
    auto result = rewriter.create<scf::IfOp>(
        loc, inRange,
        [&](OpBuilder &builder, Location nestedLoc) {
          Value index = builder.create<arith::IndexCastOp>(
              nestedLoc, builder.getIndexType(), winner);
          Value selected = builder.create<tensor::ExtractOp>(
              nestedLoc, select.getTrueValue(), ValueRange{index});
          // N > 1: at least one non-selected lane contributes the fill value.
          Value clamped =
              builder.create<arith::MaxSIOp>(nestedLoc, selected, fill);
          builder.create<scf::YieldOp>(nestedLoc, clamped);
        },
        [&](OpBuilder &builder, Location nestedLoc) {
          builder.create<scf::YieldOp>(nestedLoc, fill);
        });
    rewriter.replaceOp(op, result.getResults());
    return success();
  }
};

// Updating a single score must not materialize full-vector prefix copies.
// Keep tensor SSA semantics so bufferization can preserve other users of the
// original tensor instead of unconditionally mutating its storage.
struct FoldOneHotUpdate : OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
        type.getDimSize(0) < 1 ||
        type.getDimSize(0) > std::numeric_limits<int32_t>::max() ||
        type.getEncoding() || !type.getElementType().isF32())
      return failure();

    auto cmp = op.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
      return failure();
    auto range = cmp.getLhs().getDefiningOp<triton::MakeRangeOp>();
    auto position = cmp.getRhs().getDefiningOp<triton::SplatOp>();
    if (!range || !position) {
      range = cmp.getRhs().getDefiningOp<triton::MakeRangeOp>();
      position = cmp.getLhs().getDefiningOp<triton::SplatOp>();
    }
    if (!range || !position || range.getStart() != 0 ||
        range.getEnd() != type.getDimSize(0) ||
        !position.getSrc().getType().isInteger(32))
      return failure();

    Value replacement;
    FloatAttr replacementAttr;
    if (auto splat = op.getTrueValue().getDefiningOp<triton::SplatOp>()) {
      replacement = splat.getSrc();
    } else if (auto constant =
                   op.getTrueValue().getDefiningOp<arith::ConstantOp>()) {
      auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
      if (!dense || !dense.isSplat())
        return failure();
      replacementAttr =
          FloatAttr::get(type.getElementType(), dense.getSplatValue<APFloat>());
    } else {
      return failure();
    }

    Location loc = op.getLoc();
    if (!replacement)
      replacement = rewriter.create<arith::ConstantOp>(loc, replacementAttr);
    Value winner = position.getSrc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value size =
        rewriter.create<arith::ConstantIntOp>(loc, type.getDimSize(0), 32);
    Value lower = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 winner, zero);
    Value upper = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 winner, size);
    Value inRange = rewriter.create<arith::AndIOp>(loc, lower, upper);
    auto result = rewriter.create<scf::IfOp>(
        loc, inRange,
        [&](OpBuilder &builder, Location nestedLoc) {
          Value index = builder.create<arith::IndexCastOp>(
              nestedLoc, builder.getIndexType(), winner);
          Value updated = builder.create<tensor::InsertOp>(
              nestedLoc, replacement, op.getFalseValue(), ValueRange{index});
          builder.create<scf::YieldOp>(nestedLoc, updated);
        },
        [&](OpBuilder &builder, Location nestedLoc) {
          builder.create<scf::YieldOp>(nestedLoc, op.getFalseValue());
        });
    rewriter.replaceOp(op, result.getResults());
    return success();
  }
};

} // namespace

namespace TTOpConverters {
void populatePreConversionCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldOneHotPayload, FoldOneHotUpdate>(patterns.getContext());
}
} // namespace TTOpConverters
