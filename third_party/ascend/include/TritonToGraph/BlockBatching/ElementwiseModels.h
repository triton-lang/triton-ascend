// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_ELEMENTWISE_MODELS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_ELEMENTWISE_MODELS_H
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "mlir/IR/PatternMatch.h"
namespace mlir::triton::cfg {
// Keep inactive-lane input sanitization elementwise through lowering so it
// can fuse with its producer and consumer rather than materialize a copy.
inline constexpr llvm::StringLiteral kBatchSafeInputAttr =
    "hacc.block_batching_safe_input";

// Unconstrained floating operations may produce IEEE NaN/Inf in a newly
// inactive lane. They do not have the immediate UB of integer div/shift or
// FP-to-integer conversion. No nnan/ninf promises are admitted by the registry.
bool needsBatchActiveMask(Operation *op, const BatchOpDescriptor &descriptor);

// Operands already have their final common shape; activeMask has the same
// shape with i1 elements. Only newly inactive lanes may have inputs replaced.
SmallVector<Value> prepareBatchElementwiseOperands(
    Operation *op, const BatchOpDescriptor &descriptor, ValueRange operands,
    Value activeMask, Location loc, IRRewriter &rewriter);
Operation *rewriteBatchElementwise(Operation *op,
                                   const BatchOpDescriptor &descriptor,
                                   ValueRange operands,
                                   RankedTensorType resultType, Location loc,
                                   IRRewriter &rewriter);
} // namespace mlir::triton::cfg
#endif
