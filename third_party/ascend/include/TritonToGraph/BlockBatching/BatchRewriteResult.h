// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_RESULT_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_REWRITE_RESULT_H
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "TritonToGraph/BlockBatching/BatchResult.h"
#include "llvm/Support/JSON.h"
namespace mlir::triton::cfg {
struct BatchMappedEffect {
  unsigned accessID;
  llvm::SmallVector<unsigned, 0> generatedIDs;
};
class BatchRewriteFinalizer;
// Record exactly which validation was performed. Matrix lowering retains its
// existing structured proof; it does not claim per-effect provenance checking.
enum class BatchVerification { EffectCorrespondence, StructuredMatrix };
// Only the IR finalizer can create a completed result. No IR handle survives.
class BatchAppliedRewrite {
public:
  BatchAppliedRewrite(const BatchAppliedRewrite &) = delete;
  BatchAppliedRewrite &operator=(const BatchAppliedRewrite &) = delete;
  BatchAppliedRewrite(BatchAppliedRewrite &&) = default;
  const BatchPlan &plan() const { return *acceptedPlan; }
  BatchVerification verification() const { return validation; }
  llvm::ArrayRef<BatchMappedEffect> effects() const { return effectMappings; }
  llvm::ArrayRef<BatchLayoutResult> layouts() const { return layoutResults; }

  const llvm::json::Object &matrixExtensions() const { return extensions; }

private:
  friend class BatchRewriteFinalizer;
  BatchAppliedRewrite(std::shared_ptr<const BatchPlan> plan,
                      BatchVerification validation,
                      llvm::SmallVector<BatchMappedEffect, 0> effects,
                      llvm::SmallVector<BatchLayoutResult, 0> layouts,
                      llvm::json::Object matrixExtensions)
      : acceptedPlan(std::move(plan)), validation(validation),
        effectMappings(std::move(effects)), layoutResults(std::move(layouts)),
        extensions(std::move(matrixExtensions)) {}
  std::shared_ptr<const BatchPlan> acceptedPlan;
  BatchVerification validation;
  llvm::SmallVector<BatchMappedEffect, 0> effectMappings;
  llvm::SmallVector<BatchLayoutResult, 0> layoutResults;
  llvm::json::Object extensions;
};
BatchResult<std::string> encodeBatchGuard(const BatchAppliedRewrite &applied);
} // namespace mlir::triton::cfg
#endif
