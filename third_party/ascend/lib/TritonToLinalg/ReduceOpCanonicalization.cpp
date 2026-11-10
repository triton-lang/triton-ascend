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

#include "ascend/include/TritonToLinalg/TritonOpConverter.h"
#include "ascend/include/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace {
struct FoldOneHotGatherAfterReduceWithIndex
    : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const final {
    if (op->getAttrOfType<StringAttr>("reduce_mode"))
      return failure();
    if (op.getNumResults() != 1 || op.getInputs().size() != 1)
      return failure();
    if (!isAddFReduction(op))
      return failure();

    auto mulOp = op.getInputs()[0].getDefiningOp<arith::MulFOp>();
    if (!mulOp)
      return failure();

    Value logits;
    Value maskAsFloat;
    if (mulOp.getRhs().getDefiningOp<arith::UIToFPOp>()) {
      logits = mulOp.getLhs();
      maskAsFloat = mulOp.getRhs();
    } else if (mulOp.getLhs().getDefiningOp<arith::UIToFPOp>()) {
      logits = mulOp.getRhs();
      maskAsFloat = mulOp.getLhs();
    } else {
      return failure();
    }

    auto castOp = maskAsFloat.getDefiningOp<arith::UIToFPOp>();
    if (!castOp)
      return failure();

    auto cmpOp = castOp.getIn().getDefiningOp<arith::CmpIOp>();
    if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::eq)
      return failure();

    Value lhsBase = stripShapeAndBroadcast(cmpOp.getLhs());
    Value rhsBase = stripShapeAndBroadcast(cmpOp.getRhs());

    linalg::ReduceOp reduceWithIndex = getMaxWithIndexFromIndexValue(lhsBase);
    Value arangeBase = rhsBase;
    if (!reduceWithIndex) {
      reduceWithIndex = getMaxWithIndexFromIndexValue(rhsBase);
      arangeBase = lhsBase;
    }
    if (!reduceWithIndex)
      return failure();

    if (reduceWithIndex.getNumResults() != 2 ||
        reduceWithIndex.getInputs().size() < 2)
      return failure();

    Value reduceIndexBase =
        stripShapeAndBroadcast(reduceWithIndex.getInputs()[1]);
    if (arangeBase != reduceIndexBase)
      return failure();
    if (op.getDimensionsAttr() != reduceWithIndex.getDimensionsAttr())
      return failure();

    Value maxInput = reduceWithIndex.getInputs()[0];
    auto selectOp = maxInput.getDefiningOp<arith::SelectOp>();
    if (selectOp && selectOp.getTrueValue() != logits &&
        selectOp.getFalseValue() != logits)
      return failure();
    if (!selectOp && maxInput != logits)
      return failure();

    if (op.getResult(0).getType() != reduceWithIndex.getResult(0).getType())
      return failure();

    rewriter.replaceOp(op, reduceWithIndex.getResult(0));
    return success();
  }

private:
  static bool isAddFReduction(linalg::ReduceOp op) {
    Region &region = op.getRegion();
    if (!region.hasOneBlock())
      return false;
    Block &block = region.front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      return false;
    return yieldOp.getOperand(0).getDefiningOp<arith::AddFOp>() != nullptr;
  }

  static Value stripShapeAndBroadcast(Value value) {
    while (Operation *def = value.getDefiningOp()) {
      if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(def) ||
          def->getName().getStringRef() == "linalg.broadcast") {
        if (def->getNumOperands() == 0)
          break;
        value = def->getOperand(0);
        continue;
      }
      break;
    }
    return value;
  }

  static linalg::ReduceOp getMaxWithIndexFromIndexValue(Value value) {
    auto result = dyn_cast<OpResult>(value);
    if (!result || result.getResultNumber() != 1)
      return nullptr;

    auto reduceOp = dyn_cast<linalg::ReduceOp>(result.getOwner());
    if (!reduceOp)
      return nullptr;

    auto reduceMode = reduceOp->getAttrOfType<StringAttr>("reduce_mode");
    if (!reduceMode || reduceMode != "max_with_index")
      return nullptr;
    return reduceOp;
  }
};
} // namespace

void TTOpConverters::populatePostConversionCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldOneHotGatherAfterReduceWithIndex>(patterns.getContext());
}
