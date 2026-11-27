// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_PLAN_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_PLAN_H
#include "TritonToGraph/BlockBatching/BatchFacts.h"
#include "TritonToGraph/BlockBatching/BatchLayoutFacts.h"
#include "TritonToGraph/BlockBatching/BatchLoopSchedule.h"
#include "TritonToGraph/BlockBatching/BatchResult.h"
#include <memory>
#include <optional>

namespace mlir::triton::cfg {
// Strategy describes lowering semantics, not a second class hierarchy.
enum class BatchStrategy {
  VectorLanes,
  SequentialLoop,
  MatrixStaticLanes,
  MatrixSpatial,
  MatrixBatch
};

enum class BatchEffectRule {
  ReplicateEffect,
  PreserveVolatile,
  ShareInvariantLoad,
  SupportedEffect
};
inline const char *batchEffectRuleName(BatchEffectRule rule) {
  switch (rule) {
  case BatchEffectRule::ReplicateEffect:
    return "ReplicateEffect";
  case BatchEffectRule::PreserveVolatile:
    return "PreserveVolatile";
  case BatchEffectRule::ShareInvariantLoad:
    return "ShareInvariantLoad";
  case BatchEffectRule::SupportedEffect:
    return "SupportedEffect";
  }
  return nullptr;
}
enum class BatchEffectProof { ByteDomain, LaneCorrespondence };
struct BatchEffectObligation {
  unsigned accessID;
  BatchEffectRule rule;
  BatchEffectProof proof;
};
// All consumers share exactly the decision validated by createChecked. Layout
// and effect buffers transfer only on success so refusals can still report
// the original obligations.
class BatchPlan {
public:
  BatchPlan(const BatchPlan &) = delete;
  BatchPlan &operator=(const BatchPlan &) = delete;
  static BatchResult<std::shared_ptr<const BatchPlan>>
  createChecked(std::shared_ptr<const BatchFacts> facts, int axis,
                int64_t factor, BatchStrategy strategy,
                std::optional<BatchLogicalAxis> logicalAxis,
                llvm::SmallVector<BatchLayoutPlan, 0> &&layouts,
                llvm::SmallVector<BatchEffectObligation, 0> &&effects,
                llvm::SmallVector<unsigned, 0> &&terminalReads) {
    using Result = BatchResult<std::shared_ptr<const BatchPlan>>;
    auto invalid = [](std::string reason,
                      std::optional<unsigned> sourceID = {}) {
      return Result::error({std::move(reason), "plan", {}, {}, {}, sourceID});
    };
    if (!facts || axis < 0 || axis >= 3 || factor <= 1 || factor > INT32_MAX)
      return invalid("INVALID_BATCH_PLAN_REQUEST");
    if (!validEffects(*facts, effects, terminalReads))
      return invalid("INVALID_BATCH_EFFECT_OBLIGATIONS");
    if (logicalAxis &&
        (logicalAxis->gridAxis != axis ||
         !facts->findLoop(logicalAxis->loopID) || logicalAxis->chunk < 1 ||
         (logicalAxis->distribution != BatchLogicalAxis::Blocked &&
          logicalAxis->distribution != BatchLogicalAxis::Cyclic)))
      return invalid("INVALID_BATCH_LOGICAL_AXIS");
    switch (strategy) {
    case BatchStrategy::VectorLanes:
    case BatchStrategy::MatrixBatch:
      if (layouts.size() != facts->accesses().size())
        return invalid("INCOMPLETE_BATCH_LAYOUT_MAPPING");
      for (unsigned id = 0; id < layouts.size(); ++id)
        if (layouts[id].accessID != id)
          return invalid("INVALID_BATCH_LAYOUT_MAPPING", id);
      break;
    case BatchStrategy::SequentialLoop:
      if (!logicalAxis)
        return invalid("MISSING_BATCH_LOGICAL_AXIS");
      [[fallthrough]];
    case BatchStrategy::MatrixSpatial:
    case BatchStrategy::MatrixStaticLanes:
      if (!layouts.empty())
        return invalid("UNEXPECTED_BATCH_LAYOUTS");
      break;
    default:
      return invalid("INVALID_BATCH_STRATEGY");
    }
    return Result::success(std::shared_ptr<const BatchPlan>(new BatchPlan(
        std::move(facts), axis, factor, strategy, logicalAxis,
        std::move(layouts), std::move(effects), std::move(terminalReads))));
  }

  static bool validEffects(const BatchFacts &facts,
                           llvm::ArrayRef<BatchEffectObligation> effects,
                           llvm::ArrayRef<unsigned> terminalReads) {
    if (effects.size() != facts.accesses().size())
      return false;
    for (unsigned id = 0; id < effects.size(); ++id)
      if (effects[id].accessID != id ||
          !batchEffectRuleName(effects[id].rule) ||
          (effects[id].proof != BatchEffectProof::ByteDomain &&
           effects[id].proof != BatchEffectProof::LaneCorrespondence))
        return false;
    for (unsigned id : terminalReads)
      if (id >= effects.size())
        return false;
    return true;
  }
  BatchStrategy strategy() const { return selectedStrategy; }
  bool isMatrix() const {
    return selectedStrategy == BatchStrategy::MatrixStaticLanes ||
           selectedStrategy == BatchStrategy::MatrixSpatial ||
           selectedStrategy == BatchStrategy::MatrixBatch;
  }
  const BatchFacts &facts() const { return *sourceFacts; }
  int axis() const { return selectedAxis; }
  int64_t factor() const { return laneWidth; }
  BatchSchedule schedule() const {
    return selectedStrategy == BatchStrategy::SequentialLoop
               ? BatchSchedule::SequentialIterations
               : BatchSchedule::ProgramLanes;
  }
  bool preservesGrid() const {
    return schedule() == BatchSchedule::SequentialIterations;
  }
  unsigned hiddenExtentCount() const { return preservesGrid() ? 0 : 3; }
  const std::optional<BatchLogicalAxis> &logicalAxis() const {
    return loopAxis;
  }
  const BatchLoopFact *groupedLoop() const {
    return preservesGrid() ? sourceFacts->findLoop(loopAxis->loopID) : nullptr;
  }
  llvm::ArrayRef<BatchLayoutPlan> layouts() const { return layoutPlans; }
  llvm::ArrayRef<BatchEffectObligation> effects() const { return effectRules; }
  llvm::ArrayRef<unsigned> terminalReads() const { return terminalReadIDs; }

private:
  BatchPlan(std::shared_ptr<const BatchFacts> facts, int axis, int64_t factor,
            BatchStrategy strategy, std::optional<BatchLogicalAxis> logicalAxis,
            llvm::SmallVector<BatchLayoutPlan, 0> &&layouts,
            llvm::SmallVector<BatchEffectObligation, 0> &&effects,
            llvm::SmallVector<unsigned, 0> &&terminalReads)
      : sourceFacts(std::move(facts)), selectedAxis(axis), laneWidth(factor),
        selectedStrategy(strategy), loopAxis(logicalAxis),
        layoutPlans(std::move(layouts)), effectRules(std::move(effects)),
        terminalReadIDs(std::move(terminalReads)) {}
  std::shared_ptr<const BatchFacts> sourceFacts;
  int selectedAxis;
  int64_t laneWidth;
  BatchStrategy selectedStrategy;
  std::optional<BatchLogicalAxis> loopAxis;
  llvm::SmallVector<BatchLayoutPlan, 0> layoutPlans;
  llvm::SmallVector<BatchEffectObligation, 0> effectRules;
  llvm::SmallVector<unsigned, 0> terminalReadIDs;
};
} // namespace mlir::triton::cfg
#endif
