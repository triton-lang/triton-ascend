//===- StageMemoryPatternAnalysis.h - Cached memory address facts --------===//

#ifndef TRITON_ASCEND_MODEL_STAGE_MEMORY_PATTERN_ANALYSIS_H
#define TRITON_ASCEND_MODEL_STAGE_MEMORY_PATTERN_ANALYSIS_H

#include "AscendModel/RouteModel/StageRouteCostModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace mlir::ascend {

struct PartialContinuousTile {
  double rows = 0.0;
  double elementsPerRow = 0.0;
};

// Run OffsetAnalysis on a detached clone without modifying the input module.
// Cache only original-operation facts, never cloned Values. Classification
// and calibrated pricing share this snapshot; rebuild it after IR mutation.
class StageMemoryPatternAnalysis {
public:
  explicit StageMemoryPatternAnalysis(ModuleOp module);

  std::optional<PartialContinuousTile> lookup(Operation *operation) const {
    auto found = tiles.find(operation);
    return found == tiles.end() ? std::nullopt : found->second;
  }

  const AddressPatternSummary *lookupSummary(Operation *operation) const {
    auto found = summaries.find(operation);
    return found == summaries.end() ? nullptr : &found->second;
  }

  std::optional<bool> hasUnstructuredAxes(Operation *operation) const {
    auto found = unstructuredMemory.find(operation);
    if (found == unstructuredMemory.end())
      return std::nullopt;
    return found->second;
  }

private:
  llvm::DenseMap<Operation *, std::optional<PartialContinuousTile>> tiles;
  llvm::DenseMap<Operation *, AddressPatternSummary> summaries;
  llvm::DenseMap<Operation *, bool> unstructuredMemory;
};

} // namespace mlir::ascend

#endif // TRITON_ASCEND_MODEL_STAGE_MEMORY_PATTERN_ANALYSIS_H
