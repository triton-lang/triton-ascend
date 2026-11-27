// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_MEMORY_MODELS_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_MEMORY_MODELS_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>
#include <utility>
namespace mlir::triton::cfg {
// Candidate-only normalization; discard the clone on rejection.
LogicalResult normalizeBatchMemory(ModuleOp module, Operation *&failedOp,
                                   std::string &reason);
bool isBatchBoolToByteCast(Operation *op);
bool isBatchNormalizedBoolLoad(Operation *op);
// Read-only observation of the lowered candidate. Does not alter its masks,
// memory layout or eligibility for batching; unowned modules return false.
bool hasBatchDynamicContiguousCopy(ModuleOp module);
struct BatchPointerPlan {
  Value base;
  llvm::SmallVector<std::pair<Value, bool>> byteOffsets; // value, subtract
};
std::optional<BatchPointerPlan> getBatchPointerPlan(Value address,
                                                    unsigned bytes);
} // namespace mlir::triton::cfg
#endif
