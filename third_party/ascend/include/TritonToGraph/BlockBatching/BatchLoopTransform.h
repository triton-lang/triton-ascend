// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_TRANSFORM_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LOOP_TRANSFORM_H
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <array>

namespace mlir::triton::cfg {

// Sequential composition threads every yielded value into the next copy;
// neither original program ownership nor reduction order changes.
LogicalResult groupBatchLoopIterations(scf::ForOp loop, int64_t factor);

// Layer 3: transformation. Recompute only the program-axis loop's three
// bounds. Existing reconstruction in BatchRewrite.cpp handles the body,
// iter_args and zero-iteration path.
//
// The caller supplies the original program ID and the folded scalar group*F;
// this layer need not recognize triton::GetProgramIdOp. It depends only on
// arith and scf, without importing the Triton dialect.
//
// No factor argument is needed: cyclic bounds substitute the program ID and
// retain the original step. F is already included in foldedProgramId; the
// layout layer enlarges the chunk to axis.chunk * F.
//
// Map an original value in a bound to its counterpart in the new function
// body. Bound expressions reference the original IR, whose body is erased
// when rewriting finishes. Keeping a program-independent subexpression,
// such as R in pid*R, would leave a dangling reference. The caller owns the
// original-to-rewritten value map.
//
// A null result means no counterpart exists; reject the entire bound.
using BatchValueRemap = llvm::function_ref<Value(Value)>;

// Return a diagnostic on failure. The caller must abandon the candidate
// instead of continuing the rewrite.
BatchResult<std::array<Value, 3>>
retileBatchLoopBounds(OpBuilder &builder, scf::ForOp loop,
                      const BatchLogicalAxis &axis, Value programId,
                      Value foldedProgramId, BatchValueRemap remap);

} // namespace mlir::triton::cfg
#endif
