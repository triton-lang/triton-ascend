// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_FINALIZER_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_FINALIZER_H
#include "TritonToGraph/BlockBatching/BatchRewriteResult.h"
#include "TritonToGraph/BlockBatching/BatchRewriteWorkspace.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace mlir::triton::cfg {
LogicalResult markBatchEffects(ArrayRef<Operation *> effects);
void copyBatchEffectIdentity(Operation *source, Operation *target);
class BatchRewriteFinalizer {
public:
  static BatchResult<BatchAppliedRewrite>
  programLanes(ModuleOp module, std::shared_ptr<const BatchPlan> plan,
               unsigned originalArgCount,
               SmallVector<BatchLayoutResult, 0> layouts);
  static BatchResult<BatchAppliedRewrite>
  sequentialIterations(ModuleOp module, std::shared_ptr<const BatchPlan> plan,
                       unsigned originalArgCount);
  static BatchResult<BatchAppliedRewrite>
  matrixLanes(ModuleOp module, std::shared_ptr<const BatchPlan> plan,
              unsigned originalArgCount, StringRef matrixGuard,
              SmallVector<BatchLayoutResult, 0> layouts = {});

private:
  static BatchResult<BatchAppliedRewrite>
  createChecked(ModuleOp module, std::shared_ptr<const BatchPlan> plan,
                unsigned originalArgCount,
                SmallVector<BatchLayoutResult, 0> layouts,
                BatchVerification verification, BatchSchedule schedule,
                StringRef matrixGuard);
};
} // namespace mlir::triton::cfg
#endif
