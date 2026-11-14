// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_BATCHANALYZER_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_BATCHANALYZER_H
#include "TritonToGraph/BlockBatching/BatchResult.h"
#include "TritonToGraph/BlockBatching/BatchRewriteWorkspace.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
namespace mlir::triton::cfg {
unsigned getBatchIndexBitWidth(ModuleOp module);
class BatchAnalyzer {
  using E = BatchExprRef;
  using Axes = SmallVector<E>;
  struct Pointer {
    int root = -1;
    E offset;
    SmallVector<E> checks;
    explicit operator bool() const { return root >= 0; }
  };
  triton::FuncOp function;
  BatchRewriteWorkspace &workspace;
  unsigned indexBits, budget = 1024;
  bool scalarBounds = false, allowDeviceAddress = false,
       normalizedPointer = false, pointerRecurrences = false;
  bool spend(Value);
  void fail(Value, llvm::StringRef);
  std::optional<unsigned> sourceLoopID(Operation *loop);
  Operation *missingLoop = nullptr;
  unsigned bits(Type);
  unsigned width(Value, bool normalizedBool = false);
  std::optional<Axes> sourceAxes(Operation *, const Axes &);
  E integer(Value, const Axes &);
  std::optional<std::pair<Value, Value>> pointerRecurrence(BlockArgument);
  int rootOnly(Value, unsigned &);
  Pointer pointer(Value, const Axes &, bool normalizedBool = false);
  Pointer addressInteger(Value, const Axes &);
  E mask(Value, const Axes &);

public:
  std::string reason;
  Operation *operation = nullptr;
  BatchAnalyzer(triton::FuncOp function, unsigned indexBits,
                BatchRewriteWorkspace &workspace)
      : function(function), workspace(workspace), indexBits(indexBits) {}
  BatchResult<std::shared_ptr<const BatchFacts>>
  collect(bool pointerRecurrences, Operation *fallbackLocation);
  LogicalResult
  analyzeVariation(int axis,
                   const llvm::DenseSet<Operation *> &replicatedLoads = {});
  BatchResult<bool> analyzeAggregation();
};
} // namespace mlir::triton::cfg
#endif
