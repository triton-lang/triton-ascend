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

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/Passes.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

#include "DynamicCVPipeline/Common/Utils.h"

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "PatternMatchRewrites";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

static constexpr llvm::StringLiteral needSplitAllFuncNme[]{""};

namespace {

OpFoldResult getSingletonElement(Value tensor) {
  if (auto fill = tensor.getDefiningOp<linalg::FillOp>())
    return fill.getInputs().front();
  if (auto insert = tensor.getDefiningOp<tensor::InsertOp>();
      insert && llvm::all_of(insert.getIndices(), [](Value index) {
        return matchPattern(index, m_Zero());
      }))
    return insert.getScalar();
  if (auto elements = tensor.getDefiningOp<tensor::FromElementsOp>())
    return elements.getElements().front();
  if (auto constant = tensor.getDefiningOp<arith::ConstantOp>())
    if (auto splat = dyn_cast<SplatElementsAttr>(constant.getValue()))
      return splat.getSplatValue<TypedAttr>();
  return {};
}

Value scalarizeScale(Value scale, PatternRewriter &rewriter,
                     llvm::DenseMap<Value, Value> &scalars) {
  if (Value cached = scalars.lookup(scale))
    return cached;

  OpBuilder::InsertionGuard guard(rewriter);
  Value result = scale;
  Operation *def = scale.getDefiningOp();
  if (auto extract = dyn_cast_if_present<tensor::ExtractOp>(def)) {
    auto mul = extract.getTensor().getDefiningOp<arith::MulFOp>();
    auto type = cast<RankedTensorType>(extract.getTensor().getType());
    if (mul && type.hasStaticShape() && type.getNumElements() == 1 &&
        llvm::all_of(extract.getIndices(), [](Value index) {
          return matchPattern(index, m_Zero());
        })) {
      OpFoldResult lhs = getSingletonElement(mul.getLhs());
      OpFoldResult rhs = getSingletonElement(mul.getRhs());
      if (lhs && rhs) {
        auto materialize = [&](OpFoldResult element) -> Value {
          if (auto value = dyn_cast<Value>(element))
            return scalarizeScale(value, rewriter, scalars);
          return rewriter.create<arith::ConstantOp>(
              extract.getLoc(), cast<TypedAttr>(cast<Attribute>(element)));
        };
        rewriter.setInsertionPoint(extract);
        Value left = materialize(lhs), right = materialize(rhs);
        result = rewriter.create<arith::MulFOp>(mul.getLoc(), left, right,
                                                mul.getFastmath());
      }
    }
  } else if (def && def->hasTrait<OpTrait::Elementwise>() &&
             def->getNumResults() == 1 && isPure(def) &&
             llvm::all_of(def->getOperandTypes(), [](Type type) {
               return type.isIntOrIndexOrFloat();
             })) {
    IRMapping mapping;
    for (Value operand : def->getOperands())
      if (Value scalar = scalarizeScale(operand, rewriter, scalars);
          scalar != operand)
        mapping.map(operand, scalar);
    if (!mapping.getValueMap().empty()) {
      rewriter.setInsertionPoint(def);
      result = rewriter.clone(*def, mapping)->getResult(0);
    }
  }
  scalars.try_emplace(scale, result);
  return result;
}

struct ScalarizeFixpipeScale : OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp cast,
                                PatternRewriter &rewriter) const override {
    auto folded = matchFixpipeOutputCast(cast);
    if (!folded || !folded->quantScale)
      return failure();

    // Recover the scale before cube/vector planning sees its tensor producers.
    llvm::DenseMap<Value, Value> scalars;
    Value scale = scalarizeScale(folded->quantScale, rewriter, scalars);
    if (scale == folded->quantScale)
      return failure();

    auto fill = mlir::cast<linalg::FillOp>(folded->foldedOps.back());
    if (fill->hasOneUse()) {
      rewriter.modifyOpInPlace(fill, [&] { fill->setOperand(0, scale); });
    } else {
      Value original = fill->getResult(0);
      auto *user = *llvm::find_if(folded->foldedOps, [&](Operation *op) {
        return llvm::is_contained(op->getOperands(), original);
      });
      rewriter.setInsertionPoint(fill);
      IRMapping mapping;
      mapping.map(folded->quantScale, scale);
      Value updated = rewriter.clone(*fill, mapping)->getResult(0);
      rewriter.modifyOpInPlace(
          user, [&] { user->replaceUsesOfWith(original, updated); });
    }
    return success();
  }
};

class PatternMatchRewritePass
    : public PassWrapper<PatternMatchRewritePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PatternMatchRewritePass);

  PatternMatchRewritePass() = default;
  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect>();
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-standardize-op-pattern-match";
  }
};

} // namespace

void PatternMatchRewritePass::runOnOperation() {
  auto moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  LOG_DEBUG("Input mlir:\n" << moduleOp);

  bool needSplitAll = false;
  moduleOp.walk([&](func::FuncOp funcOp) -> WalkResult {
    if (!llvm::is_contained(needSplitAllFuncNme, funcOp.getSymName())) {
      return WalkResult::advance();
    }
    LOG_DEBUG("[INFO] Matmul should split anyway: " << funcOp.getSymName());
    needSplitAll = true;
    return WalkResult::interrupt();
  });

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<SplitMatmulPattern>(ctx, needSplitAll);
  patterns.add<FoldExpandExtCollapse, ScalarizeFixpipeScale>(ctx);

  // the way we handle matmul dependencies across for blocks
  // requres the patternmatching to go top-down
  GreedyRewriteConfig config = GreedyRewriteConfig().setUseTopDownTraversal();
  if (llvm::failed(
          applyPatternsGreedily(moduleOp, std::move(patterns), config))) {
    LOG_DEBUG("matchAndRewrite does not converge!");
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LOG_DEBUG("Output mlir:\n" << moduleOp);
}

std::unique_ptr<OperationPass<ModuleOp>>
CVPipeline::createPatternMatchRewritePass() {
  return std::make_unique<PatternMatchRewritePass>();
}
