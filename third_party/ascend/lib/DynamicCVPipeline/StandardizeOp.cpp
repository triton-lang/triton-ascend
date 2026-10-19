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

#include "bishengir/Dialect/HIVM/IR/CustomOp/CustomOpUtils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "StandardizeOp";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {
bool hasTensorType(Value value, ArrayRef<int64_t> shape, Type element) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == shape && type.getElementType() == element &&
         !type.getEncoding();
}

bool isPositiveZeroInit(Value value) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  if (!fill || !fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>())
    return false;
  return matchPattern(fill.getDpsInputs()[0], m_PosZeroFloat());
}

bool isCanonicalPV(linalg::MatmulOp op) {
  auto *ctx = op.getContext();
  auto fp8 = Float8E4M3FNType::get(ctx);
  auto f32 = Float32Type::get(ctx);
  if (op.getNumResults() != 1 || op.getDpsInputs().size() != 2 ||
      op.getDpsInits().size() != 1)
    return false;
  auto result = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  auto lhs = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  if (!result || !lhs || !result.hasStaticShape() || !lhs.hasStaticShape() ||
      result.getRank() != 2 || lhs.getRank() != 2 ||
      result.getDimSize(0) <= 0 || result.getDimSize(1) <= 0 ||
      lhs.getDimSize(1) <= 0 || result.getElementType() != f32 ||
      result.getEncoding() ||
      !hasTensorType(op.getDpsInputs()[0],
                     {result.getDimSize(0), lhs.getDimSize(1)}, fp8) ||
      !hasTensorType(op.getDpsInputs()[1],
                     {lhs.getDimSize(1), result.getDimSize(1)}, fp8))
    return false;
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  SmallVector<AffineMap> expected{AffineMap::get(3, 0, {m, k}, ctx),
                                  AffineMap::get(3, 0, {k, n}, ctx),
                                  AffineMap::get(3, 0, {m, n}, ctx)};
  return op.getIndexingMapsArray() == expected;
}

bool hasPairedF16SoftmaxTensorABI(hivm::CustomOp custom) {
  // Tensor ABI, optionally with per-row causal limits and a scalar mask flag.
  // This is not an ownership shortcut for a bufferized custom.
  if (!hivm::canPartitionCustomOpRows(custom))
    return false;
  for (Type type : custom->getOperandTypes())
    if (auto tensor = dyn_cast<RankedTensorType>(type);
        tensor && tensor.getEncoding())
      return false;
  return true;
}

linalg::MatmulOp matchOnlineSoftmaxFullPV(hivm::CustomOp custom) {
  auto *ctx = custom.getContext();
  auto fp8 = Float8E4M3FNType::get(ctx);
  if (!hasPairedF16SoftmaxTensorABI(custom))
    return nullptr;

  // Consume the complete logical P in one Cube matmul. The
  // Fractal-to-ND conversion is a layout view, not a split or copy.
  auto probability = cast<RankedTensorType>(custom.getResult(0).getType());
  if (!probability.hasStaticShape() || probability.getRank() != 4 ||
      !custom.getResult(0).hasOneUse())
    return nullptr;
  auto shape = probability.getShape();
  auto convert =
      dyn_cast<hivm::ConvertLayoutOp>(*custom.getResult(0).getUsers().begin());
  if (!convert || convert->getBlock() != custom->getBlock() ||
      !convert.getResult().hasOneUse() ||
      convert.getSrcLayout().getDataLayout() != hivm::DataLayout::Fractal ||
      convert.getDstLayout().getDataLayout() != hivm::DataLayout::ND ||
      convert.getSrcLayout().getTransposeValue().value_or(false) ||
      convert.getDstLayout().getTransposeValue().value_or(false) ||
      !convert.getSrcLayout().getFractalSizes() ||
      convert.getSrcLayout().getFractalSizes().asArrayRef() !=
          shape.take_back(2) ||
      !hasTensorType(convert.getResult(),
                     {shape[1] * shape[2], shape[0] * shape[3]}, fp8))
    return nullptr;
  auto matmul =
      dyn_cast<linalg::MatmulOp>(*convert.getResult().getUsers().begin());
  if (!matmul || matmul->getBlock() != custom->getBlock() ||
      !isCanonicalPV(matmul) ||
      matmul.getDpsInputs()[0] != convert.getResult() ||
      !isPositiveZeroInit(matmul.getDpsInits()[0]) ||
      !matmul.getResult(0).hasOneUse())
    return nullptr;
  return matmul;
}

struct OwnershipPlan {
  arith::ExtFOp oldMaxExt;
  hivm::CustomOp softmax;
};

static std::optional<unsigned> getStateIndex(Value value, scf::ForOp loop) {
  int index = CVPipeline::getLoopCarriedArgIndex(value, loop.getBody());
  return index < 0 ? std::nullopt : std::optional<unsigned>(index);
}

static Value otherOperand(arith::AddFOp op, Value value) {
  return op.getLhs() == value ? op.getRhs() : op.getLhs();
}

static bool broadcastsRows(linalg::BroadcastOp broadcast, Value source,
                           Block *block) {
  return broadcast && broadcast->getBlock() == block &&
         broadcast.getInput() == source &&
         broadcast.getDimensions() == ArrayRef<int64_t>({1}) &&
         broadcast.getNumResults() == 1 && broadcast->getResult(0).hasOneUse();
}

static FailureOr<OwnershipPlan> matchOwnership(hivm::CustomOp softmax) {
  auto loop = dyn_cast<scf::ForOp>(softmax->getParentOp());
  auto pv = matchOnlineSoftmaxFullPV(softmax);
  if (!loop || !pv)
    return failure();
  unsigned softmaxCount = 0;
  for (Operation &op : *loop.getBody())
    if (auto custom = dyn_cast<hivm::CustomOp>(op);
        custom && custom.getRowPartitionInfo())
      ++softmaxCount;
  if (softmaxCount != 1)
    return failure();

  Block *body = loop.getBody();
  Value oldMax = softmax.getInputs()[1];
  auto maxIndex = getStateIndex(oldMax, loop);
  auto yield = dyn_cast<scf::YieldOp>(body->getTerminator());
  if (!maxIndex || !yield ||
      yield.getOperand(*maxIndex) != softmax.getResult(1) ||
      !softmax.getResult(2).hasOneUse())
    return failure();

  auto lAdd = dyn_cast<arith::AddFOp>(*softmax.getResult(2).getUsers().begin());
  Value lProduct = lAdd ? otherOperand(lAdd, softmax.getResult(2)) : Value{};
  auto lMul = lProduct ? lProduct.getDefiningOp<arith::MulFOp>() : nullptr;
  if (!lAdd || !lMul || lAdd->getBlock() != body || lMul->getBlock() != body ||
      !lAdd.getResult().hasOneUse() || !lMul.getResult().hasOneUse())
    return failure();
  auto sumIndex = getStateIndex(lMul.getLhs(), loop);
  Value oldSum = lMul.getLhs();
  Value alphaValue = lMul.getRhs();
  if (!sumIndex) {
    sumIndex = getStateIndex(lMul.getRhs(), loop);
    oldSum = lMul.getRhs();
    alphaValue = lMul.getLhs();
  }
  auto alpha = alphaValue.getDefiningOp<math::ExpOp>();
  if (!sumIndex || *sumIndex == *maxIndex || !oldSum.hasOneUse() ||
      yield.getOperand(*sumIndex) != lAdd.getResult() || !alpha ||
      alpha->getBlock() != body)
    return failure();
  auto delta = alpha.getOperand().getDefiningOp<arith::SubFOp>();
  if (!delta || delta->getBlock() != body || !delta.getResult().hasOneUse())
    return failure();
  auto oldMaxExt = delta.getLhs().getDefiningOp<arith::ExtFOp>();
  auto nextMaxExt = delta.getRhs().getDefiningOp<arith::ExtFOp>();
  if (!oldMaxExt || !nextMaxExt || oldMaxExt->getBlock() != body ||
      nextMaxExt->getBlock() != body || oldMaxExt.getIn() != oldMax ||
      nextMaxExt.getIn() != softmax.getResult(1) ||
      !oldMaxExt.getResult().hasOneUse() || !nextMaxExt.getResult().hasOneUse())
    return failure();
  for (OpOperand &use : oldMax.getUses()) {
    if (!((use.getOwner() == softmax && use.getOperandNumber() == 1) ||
          (use.getOwner() == oldMaxExt && use.getOperandNumber() == 0)))
      return failure();
  }

  auto pvScale = dyn_cast<arith::MulFOp>(*pv.getResult(0).getUsers().begin());
  if (!pvScale || pvScale->getBlock() != body ||
      !pvScale.getResult().hasOneUse())
    return failure();
  auto accAdd =
      dyn_cast<arith::AddFOp>(*pvScale.getResult().getUsers().begin());
  Value accProduct =
      accAdd ? otherOperand(accAdd, pvScale.getResult()) : Value{};
  auto accMul =
      accProduct ? accProduct.getDefiningOp<arith::MulFOp>() : nullptr;
  if (!accAdd || !accMul || accAdd->getBlock() != body ||
      accMul->getBlock() != body || !accAdd.getResult().hasOneUse() ||
      !accMul.getResult().hasOneUse())
    return failure();
  auto accIndex = getStateIndex(accMul.getLhs(), loop);
  Value oldAcc = accMul.getLhs();
  Value broadcastValue = accMul.getRhs();
  if (!accIndex) {
    accIndex = getStateIndex(accMul.getRhs(), loop);
    oldAcc = accMul.getRhs();
    broadcastValue = accMul.getLhs();
  }
  auto alphaBroadcast = broadcastValue.getDefiningOp<linalg::BroadcastOp>();
  if (!accIndex || *accIndex == *maxIndex || *accIndex == *sumIndex ||
      !oldAcc.hasOneUse() ||
      yield.getOperand(*accIndex) != accAdd.getResult() ||
      !broadcastsRows(alphaBroadcast, alpha.getResult(), body))
    return failure();
  for (Operation *user : alpha.getResult().getUsers()) {
    if (user != lMul.getOperation() && user != alphaBroadcast.getOperation())
      return failure();
  }

  Value finalAcc = loop.getResult(*accIndex);
  if (!finalAcc.hasOneUse())
    return failure();
  auto normalize = dyn_cast<arith::DivFOp>(*finalAcc.getUsers().begin());
  if (!normalize || normalize->getBlock() != loop->getBlock() ||
      normalize.getLhs() != finalAcc || !normalize.getResult().hasOneUse() ||
      !isa<bufferization::MaterializeInDestinationOp>(
          *normalize.getResult().getUsers().begin()) ||
      !broadcastsRows(normalize.getRhs().getDefiningOp<linalg::BroadcastOp>(),
                      loop.getResult(*sumIndex), loop->getBlock()))
    return failure();

  return OwnershipPlan{oldMaxExt, softmax};
}

void prepareOnlineSoftmaxRecurrences(ModuleOp module) {
  SmallVector<OwnershipPlan> plans;
  module.walk([&](hivm::CustomOp custom) {
    if (!custom.getRowPartitionInfo())
      return;
    auto plan = matchOwnership(custom);
    if (succeeded(plan))
      plans.push_back(*plan);
  });
  Builder builder(module.getContext());
  for (const OwnershipPlan &plan : plans) {
    if (plan.softmax->isBeforeInBlock(plan.oldMaxExt))
      plan.oldMaxExt->moveBefore(plan.softmax);
    plan.oldMaxExt->setAttr(CVPipeline::kOnlineSoftmaxOldMaxRead,
                            builder.getUnitAttr());
  }
}
} // namespace

namespace mlir::triton {

void StandardizeOpPass::runOnOperation() {
  auto op = getOperation();

  if (CVPipeline::hasFallbackAttr(op)) {
    return;
  }

  prepareOnlineSoftmaxRecurrences(op);

  LOG_DEBUG("Input mlir:\n" << op);
  OpPassManager pm(op.getOperationName());
  pm.addPass(createPatternMatchRewritePass());

  if (llvm::failed(runPipeline(pm, op))) {
    LOG_DEBUG("Pipeline Failed!");
    CVPipeline::setFallbackAttr(op, CVPipeline::ERRCODE_FAILED);
    return;
  }

  bool findMayNotExec = false;
  op->walk([&](linalg::MatmulOp matmulOp) {
    if (matmulOp->hasAttr(CVPipeline::kMayNotExec)) {
      findMayNotExec = true;
    }
  });

  if (findMayNotExec) {
    LOG_DEBUG("Matmul may not execute!");
    CVPipeline::setFallbackAttr(op, CVPipeline::ERRCODE_IGNORED);
    return;
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createStandardizeOpPass() {
  return std::make_unique<StandardizeOpPass>();
};

void registerStandardizeOpPasses() {
  registerPass(createStandardizeOpPass);
  registerPass(createPatternMatchRewritePass);
}

} // namespace mlir::triton
