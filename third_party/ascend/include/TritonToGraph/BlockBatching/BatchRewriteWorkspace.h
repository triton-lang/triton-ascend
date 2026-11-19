// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_BATCHREWRITEWORKSPACE_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_BATCHREWRITEWORKSPACE_H
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
namespace mlir::triton::cfg {
enum class BatchVariation { Unknown, Uniform, Varying };
class BatchRewriteWorkspace {
public:
  struct GeneratedBinding {
    Value value;
    bool varying = false;
  };

private:
  struct ValueState {
    BatchVariation source = BatchVariation::Unknown;
    GeneratedBinding generated;
    bool aggregationPresent = false, aggregationUnknown = false;
    llvm::SmallBitVector aggregationRows;
  };
  llvm::DenseMap<Value, ValueState> values;
  SmallVector<Operation *> sources;
  llvm::DenseMap<Operation *, unsigned> sourceIDs, loopIDs;

public:
  std::shared_ptr<const BatchFacts> facts;
  std::shared_ptr<const BatchPlan> acceptedPlan;
  llvm::SmallVector<BatchEffectObligation, 0> pendingEffects;
  llvm::SmallVector<unsigned, 0> pendingTerminalReads;
  std::optional<int> pendingGuardAxis;
  llvm::DenseSet<Operation *> replicatedLoads;
  SmallVector<Operation *> semanticAccesses;
  ArrayRef<Operation *> sourceAccesses() const { return sources; }
  const auto &sourceIDsByOp() const { return sourceIDs; }
  void recordAccess(Operation *op) {
    sourceIDs[op] = sources.size();
    sources.push_back(op);
  }
  void recordLoop(Operation *op, unsigned id) { loopIDs[op] = id; }
  std::optional<unsigned> loopID(Operation *op) const {
    auto i = loopIDs.find(op);
    return i == loopIDs.end() ? std::nullopt
                              : std::optional<unsigned>(i->second);
  }
  bool validSourceMapping(size_t count) const {
    if (sources.size() != count || sourceIDs.size() != count)
      return false;
    for (auto [id, source] : llvm::enumerate(sources)) {
      auto it = sourceIDs.find(source);
      if (!source || it == sourceIDs.end() || it->second != id)
        return false;
    }
    return true;
  }
  void beginVariationScope() {
    for (auto &item : values)
      item.second.source = BatchVariation::Unknown;
  }
  bool known(Value v) const {
    auto i = values.find(v);
    return i != values.end() && i->second.source != BatchVariation::Unknown;
  }
  bool isVarying(Value v) const {
    auto i = values.find(v);
    return i != values.end() && i->second.source == BatchVariation::Varying;
  }
  void recordVariation(Value v, bool varies) {
    if (varies || !known(v))
      values[v].source =
          varies ? BatchVariation::Varying : BatchVariation::Uniform;
  }
  bool mergeRows(Value v, const llvm::SmallBitVector &rows, bool unknown) {
    if (rows.none() && !unknown)
      return false;
    auto &state = values[v];
    state.aggregationPresent = true;
    auto old = state.aggregationRows;
    bool changed = unknown && !state.aggregationUnknown;
    state.aggregationUnknown |= unknown;
    // operator|= widens to the longer operand by itself. Resizing to
    // rows.size() first would let a narrower later merge truncate the
    // recorded domain, silently dropping rows and reporting that loss as a
    // change. Widening only keeps the merge monotone.
    state.aggregationRows |= rows;
    return changed || old != state.aggregationRows;
  }
  std::optional<std::pair<llvm::SmallBitVector, bool>>
  aggregation(Value v) const {
    auto i = values.find(v);
    if (i == values.end() || !i->second.aggregationPresent)
      return std::nullopt;
    return std::make_pair(i->second.aggregationRows,
                          i->second.aggregationUnknown);
  }
  void bindGenerated(Value v, Value generated, bool varies) {
    values[v].generated = {generated, varies};
  }
  GeneratedBinding generatedBinding(Value v) const {
    auto i = values.find(v);
    return i == values.end() ? GeneratedBinding{} : i->second.generated;
  }
  Value lookupGenerated(Value v) const { return generatedBinding(v).value; }
  // Every external operand must be rebound before clone() sees it. MLIR's
  // default for an omitted entry would retain the original source Value.
  std::optional<IRMapping> cloneOperands(Operation *op,
                                         ValueRange captures = {}) const {
    IRMapping mapped;
    auto bind = [&](ValueRange operands) {
      for (Value v : operands) {
        Value generated = lookupGenerated(v);
        if (!generated)
          return false;
        mapped.map(v, generated);
      }
      return true;
    };
    if (!bind(op->getOperands()) || !bind(captures))
      return std::nullopt;
    return mapped;
  }
  void invalidateSource() {
    sources.clear();
    sourceIDs.clear();
    loopIDs.clear();
    replicatedLoads.clear();
    values.clear();
  }
  void invalidateGenerated() {
    semanticAccesses.clear();
    for (auto &entry : values)
      entry.second.generated = {};
  }
};
} // namespace mlir::triton::cfg
#endif
