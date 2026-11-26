/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */
#ifndef TRITON_ASCEND_DYNAMICCVPIPELINE_OUTEREVENTLOOP_H
#define TRITON_ASCEND_DYNAMICCVPIPELINE_OUTEREVENTLOOP_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir::CVPipeline {

Value getOuterEventBufferRoot(Value value);
Operation *getOuterEventScratchAllocation(Value value);
bool hasOnlyLocalMemoryEffects(Operation *op, scf::ForOp loop, bool allowFree);
OpBuilder createOuterEventBodyBuilder(Block &block);
SmallVector<Value> mapOuterEventValues(ValueRange values,
                                       const IRMapping &mapping);
scf::IfOp createOuterEventIf(
    OpBuilder &builder, Location loc, Value condition, TypeRange types,
    function_ref<SmallVector<Value>(OpBuilder &, unsigned)> emitBranch);
scf::IfOp
createOuterEventGuard(OpBuilder &builder, Location loc, Value condition,
                      ValueRange previous,
                      function_ref<SmallVector<Value>(OpBuilder &)> emit);

// Scalar cursor updates may read immutable metadata, but never payload storage
// or volatile flags. The count prepass additionally allows structured loops.
bool isOuterEventScalarControl(Operation *root, bool &readsMemory,
                               bool allowLoops = false);

/// The first iter_args are control state. The remaining arguments
/// carry tensors between events. Scheduling is a separate transformation.
struct OuterEventLoopPlan {
  scf::ForOp outer;
  SmallVector<scf::ForOp, 2> inner;
  SmallVector<SmallVector<Operation *>, 3> segments;
  SmallVector<SmallVector<Operation *>, 2> boundSlices;
  SmallVector<Value> carried;
  // The original nest is retained when Cube accumulators must stay within
  // a single event per task.
  OwningOpRef<scf::ForOp> serialFallback;

  static FailureOr<OuterEventLoopPlan> analyze(scf::ForOp outer,
                                               ArrayRef<scf::ForOp> inner);
  void rewrite();
};

// Transactionally join a compatible partial/full pair, preserving arithmetic.
LogicalResult tryJoinSparseInnerLoops(OuterEventLoopPlan &plan,
                                      bool allowTaskLocalAccumulation);

} // namespace mlir::CVPipeline

#endif
