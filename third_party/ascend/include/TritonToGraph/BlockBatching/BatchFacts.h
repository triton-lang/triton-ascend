// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_BATCHFACTS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_BATCHFACTS_H
#include "TritonToGraph/BlockBatching/BatchLoopFacts.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
namespace mlir::triton::cfg {
// Owned original-domain facts. Effects normally borrow the corresponding
// access; only joint-store normalization records a replacement shape/offset or
// mask.
class BatchFacts {
public:
  struct Access {
    int root = -1;
    unsigned bytes = 0;
    bool write = false;
    llvm::SmallVector<int64_t> shape;
    BatchExprRef byteOffset, mask;
    llvm::SmallVector<BatchExprRef> pointerChecks;
    llvm::SmallVector<unsigned, 0> loopIDs;
    bool isVolatile = false, deviceAddress = false, normalizedPointer = false;
    int64_t cache = 0, evict = 0;
  };
  struct EffectSlot {
    BatchExprRef offset, mask;
    int64_t jointWidth = 0;
  };
  static std::optional<BatchFacts>
  createChecked(llvm::SmallVector<Access, 0> accesses,
                llvm::SmallVector<EffectSlot, 0> slots,
                llvm::SmallVector<BatchLoopFact, 0> loops) {
    if (accesses.size() != slots.size())
      return std::nullopt;
    BatchFacts facts(std::move(accesses), std::move(slots), std::move(loops));
    for (unsigned i = 1; i < facts.loopFacts.size(); ++i)
      if (facts.loopFacts[i - 1].id == facts.loopFacts[i].id)
        return std::nullopt;
    for (const auto &access : facts.accessFacts)
      for (unsigned id : access.loopIDs)
        if (!facts.findLoop(id))
          return std::nullopt;
    return facts;
  }
  llvm::ArrayRef<Access> accesses() const { return accessFacts; }
  const Access &access(unsigned id) const { return accessFacts[id]; }
  llvm::ArrayRef<EffectSlot> effects() const { return effectSlots; }
  llvm::ArrayRef<BatchLoopFact> loops() const { return loopFacts; }
  const BatchLoopFact *findLoop(unsigned id) const {
    auto it =
        llvm::lower_bound(loopFacts, id, [](const auto &loop, unsigned key) {
          return loop.id < key;
        });
    return it != loopFacts.end() && it->id == id ? &*it : nullptr;
  }

private:
  BatchFacts(llvm::SmallVector<Access, 0> accesses,
             llvm::SmallVector<EffectSlot, 0> slots,
             llvm::SmallVector<BatchLoopFact, 0> loops)
      : accessFacts(std::move(accesses)), effectSlots(std::move(slots)),
        loopFacts(std::move(loops)) {
    llvm::sort(loopFacts,
               [](const auto &a, const auto &b) { return a.id < b.id; });
  }
  llvm::SmallVector<Access, 0> accessFacts;
  llvm::SmallVector<EffectSlot, 0> effectSlots;
  llvm::SmallVector<BatchLoopFact, 0> loopFacts;
};
} // namespace mlir::triton::cfg
#endif
