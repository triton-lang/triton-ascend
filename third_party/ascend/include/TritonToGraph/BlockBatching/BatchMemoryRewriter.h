// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_MEMORY_REWRITER_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_MEMORY_REWRITER_H
#include "TritonToGraph/BlockBatching/BatchLayoutPlan.h"
#include "mlir/IR/PatternMatch.h"
namespace mlir::triton::cfg {
class BatchMemoryRewriter {
public:
  // Read-only proof for automatic axis selection. Uses the same closed-region
  // check as the memory rewrite and requires unit-stride column accesses.
  static bool probeColumnRegion(triton::FuncOp function,
                                const BatchRewriteWorkspace &scope, int axis,
                                int64_t factor);

  static LogicalResult rewriteViews(ModuleOp module,
                                    const BatchRewriteWorkspace &scope,
                                    ArrayRef<BatchLayoutPlan> plans,
                                    SmallVectorImpl<BatchLayoutResult> &results,
                                    ArrayRef<Operation *> semanticAccesses,
                                    Value group, unsigned originalArgCount,
                                    int axis, int64_t factor);

private:
  explicit BatchMemoryRewriter(triton::FuncOp function,
                               unsigned originalArgs = 0)
      : r(function.getContext()), loc(function.getLoc()), function(function),
        originalArgs(originalArgs) {}
  IRRewriter r;
  Location loc;
  triton::FuncOp function;
  unsigned originalArgs;
  SmallVector<int64_t> targetShape, coordinateMap;
  llvm::DenseMap<int64_t, Value> inductionVars;
  Block *block = nullptr;
  SmallVector<int64_t> semanticShape, regionShape;
  llvm::DenseSet<Operation *> loads, stores, oldOperations;
  llvm::DenseSet<Value> checked;
  unsigned budget = 4096;
  // Bindings belong to exactly one revision of the private layout clone.
  BatchRewriteWorkspace bindings;
  LogicalResult rewriteMemoryView(Operation *op, const BatchFacts &facts,
                                  const BatchLayoutPlan &plan,
                                  BatchLayoutResult &result, Value mappedGroup,
                                  bool inColumnRegion, int axis,
                                  int64_t factor);
  LogicalResult finish(ModuleOp sandbox, bool columnRegion);
  Value constant(int64_t n);
  Value wide(Value value);
  Value upper(BatchExprRef e);
  Value binary(BatchExpr::Kind kind, Value a, Value b);
  std::optional<unsigned> mappedAxis(int64_t id) const;
  Value splat(Value v);
  Value expandAxis(Value value, unsigned axis);
  Value range(unsigned axis);
  Value axisMask(Value base, Value limit, unsigned axis);
  Value predicate(BatchExprRef expr);
  Value expression(BatchExprRef e);
  Value reshape(Value v, ArrayRef<int64_t> shape);
  bool tensorShape(Value value) const;
  Attribute uniformConstant(Value value) const;
  Value uniformScalar(Value value) const;
  bool supports(Value value);
  bool closedUsers(Value value, llvm::DenseSet<Value> &visited);
  bool proveClosedRegion();
  Value materializeRegion(Value value);
};
} // namespace mlir::triton::cfg
#endif
