/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */

#include "ascend/include/DynamicCVPipeline/NormalizeOuterLoop.h"
#include "DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/Common.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/OuterEventLoop.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton {
#define GEN_PASS_DECL_NORMALIZEOUTERLOOPPASS
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"
#define GEN_PASS_DEF_NORMALIZEOUTERLOOPPASS
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::CVPipeline;

namespace {

// The user selects the outer loop; only a nested loop with both Cube and
// Vector computation needs the new scheduling path. Single-level, Cube-only
// and Vector-only loops retain the ordinary DynamicCV implementation.
static bool hasMixedCVInnerLoop(scf::ForOp outer) {
  return outer
      .walk([&](scf::ForOp inner) {
        if (inner == outer)
          return WalkResult::advance();
        bool cube = false, vector = false;
        inner.walk([&](Operation *op) {
          if (isa<linalg::MatmulOp>(op))
            cube = true;
          else
            vector |= isRealTensorComputeOp(op);
        });
        return cube && vector ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

static FailureOr<OuterEventLoopPlan> analyzeNest(scf::ForOp outer) {
  SmallVector<scf::ForOp, 2> innerLoops;
  auto result = outer.walk([&](Operation *op) {
    if (op == outer.getOperation() || !isa<LoopLikeOpInterface>(op))
      return WalkResult::advance();
    auto inner = dyn_cast<scf::ForOp>(op);
    if (!inner || inner->getParentOp() != outer || innerLoops.size() == 2) {
      outer.emitError(
          "outer mainloop supports at most two direct inner scf.for "
          "loops and no deeper loops");
      return WalkResult::interrupt();
    }
    if (inner->hasAttr(kOuterMainLoopHint)) {
      outer.emitError("nested outer-mainloop requests are not supported");
      return WalkResult::interrupt();
    }
    APInt lower, step;
    if (!matchPattern(inner.getLowerBound(), m_ConstantInt(&lower)) ||
        !lower.isZero() ||
        !matchPattern(inner.getStep(), m_ConstantInt(&step)) || !step.isOne()) {
      inner.emitError("outer mainloop requires inner lower bound 0 and step 1");
      return WalkResult::interrupt();
    }
    innerLoops.push_back(inner);
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return OuterEventLoopPlan::analyze(outer, innerLoops);
}

// OpClassifier may clone or move operations. Use its normal main-loop walk on
// a temporary module and carry only the selected loop identities back.
static void selectOuterLoops(ModuleOp module) {
  IRMapping mapping;
  OwningOpRef<ModuleOp> classified(cast<ModuleOp>(module->clone(mapping)));
  PassManager classify(module.getContext(), ModuleOp::getOperationName());
  classify.addPass(triton::createOpClassifierPass());
  if (failed(classify.run(*classified)) || hasFallbackAttr(*classified))
    return;

  llvm::DenseSet<Operation *> selected;
  (void)SplitIf::walkMainLoop(classified->getOperation(), [&](Operation *loop) {
    if (auto parent = dyn_cast_or_null<scf::ForOp>(loop->getParentOp()))
      selected.insert(parent);
    return success();
  });
  if (selected.empty())
    return;

  auto mark = UnitAttr::get(module.getContext());
  module.walk([&](scf::ForOp outer) {
    if (selected.contains(mapping.lookupOrNull(outer.getOperation())))
      outer->setAttr(kOuterMainLoopHint, mark);
  });
}

class NormalizeOuterLoopPass
    : public triton::impl::NormalizeOuterLoopPassBase<NormalizeOuterLoopPass> {
public:
  using Base = triton::impl::NormalizeOuterLoopPassBase<NormalizeOuterLoopPass>;
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (hasFallbackAttr(module))
      return;
    if (discoverMainLoop)
      selectOuterLoops(module);
    SmallVector<OuterEventLoopPlan, 1> plans;
    auto result = module.walk<WalkOrder::PreOrder>([&](scf::ForOp outer) {
      if (!outer->hasAttr(kOuterMainLoopHint))
        return WalkResult::advance();
      if (!hasMixedCVInnerLoop(outer)) {
        outer->removeAttr(kOuterMainLoopHint);
        return WalkResult::advance();
      }
      auto plan = analyzeNest(outer);
      if (failed(plan))
        return WalkResult::interrupt();
      plans.push_back(std::move(*plan));
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      setFallbackAttr(module, ERRCODE_IGNORED);
      return signalPassFailure();
    }
    module->removeAttr(kOuterMainLoopHint);
    // Buffer counts are module-wide. Apply the task-local schedule only
    // when it owns the module's single function and selected loop.
    bool allowTaskLocalAccumulation =
        plans.size() == 1 &&
        llvm::hasSingleElement(module.getOps<func::FuncOp>());
    for (OuterEventLoopPlan &plan : plans) {
      // A pair that cannot safely share its matmuls remains an ordinary nest.
      // Joining is transactional: failure leaves the original body untouched.
      if (plan.inner.size() == 2 &&
          failed(tryJoinSparseInnerLoops(plan, allowTaskLocalAccumulation))) {
        plan.outer->removeAttr(kOuterMainLoopHint);
        continue;
      }
      if (plan.serialFallback) {
        // The coalesced Cube stage keeps the outputs of independent
        // matmuls live together. Bound the cross-core slots for that stage.
        triton::BufferCountManager(module).setBufferCount(
            triton::BufferCountManager::DepType::InterCore, 1);
      }
      plan.rewrite();
      module->setAttr(kOuterMainLoopHint, UnitAttr::get(module.getContext()));
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createNormalizeOuterLoopPass(bool discoverMainLoop) {
  NormalizeOuterLoopPassOptions options;
  options.discoverMainLoop = discoverMainLoop;
  return std::make_unique<NormalizeOuterLoopPass>(options);
}
