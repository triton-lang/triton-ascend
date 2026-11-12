// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_PLAN_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_LAYOUT_PLAN_H
#include "TritonToGraph/BlockBatching/BatchAccessAnalysis.h"
namespace mlir::triton::cfg {
inline constexpr unsigned kBatchLayoutPlannerVersion = 2;
struct BatchLayoutTerm {
  BatchExpr::Kind kind;
  int64_t id;
  BatchExprRef coefficient;
};
struct BatchLayoutPlan {
  enum Action { Keep, MemoryView, PointwiseRegion };
  Action action = Keep;
  unsigned accessID = 0;
  int gridAxis = -1;
  SmallVector<int64_t> localAxes, shape;
  int64_t tileExtent = 0, origin = 0;
  // Uniform integer expressions, independent of all program/local coordinates.
  // Owned DAGs survive the original IR, unlike an MLIR Value captured here.
  BatchExprRef elementOrigin, elementStride;
  bool symbolic = false;
  std::string runtimeGuardJSON;
  bool linearMemoryView = false;
  SmallVector<BatchLayoutTerm> restTerms;
  SmallVector<BatchExprRef> residualMasks;
  std::string reason = "DIRECTION_UNPROVEN", maskClass = "unknown";
  std::string location;
  BatchExprRef coordinate, upper;
  bool rewritten = false;
  bool controlFlowIndependent = false;
};
SmallVector<BatchLayoutPlan, 0>
planBatchLayouts(const BatchAccessReport &accesses, int axis, int64_t factor);
// Read-only proof for automatic axis selection. Uses the same closed-region
// check as the memory rewrite and requires unit-stride column accesses.
bool hasBatchColumnPointwiseRegion(triton::FuncOp function,
                                   const BatchAccessReport &accesses, int axis,
                                   int64_t factor);
std::string serializeBatchLayouts(ArrayRef<BatchLayoutPlan> plans);
LogicalResult rewriteBatchMemoryViews(ModuleOp module,
                                      const BatchAccessReport &accesses,
                                      SmallVectorImpl<BatchLayoutPlan> &plans,
                                      ArrayRef<Operation *> semanticAccesses,
                                      Value group, unsigned originalArgCount,
                                      int axis, int64_t factor);
} // namespace mlir::triton::cfg
#endif
