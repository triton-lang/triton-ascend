// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_OP_REGISTRY_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_OP_REGISTRY_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir::triton::cfg {
inline constexpr unsigned kBatchImplementationVersion = 46;
enum class BatchModel {
  Domain,
  Constant,
  Elementwise,
  Shape,
  Memory,
  Reduction,
  Scan,
  ControlFlow,
  IndexCast,
  Dot,
  Structural,
  Deferred,
  Rejected
};
enum class BatchAttributePolicy {
  None,
  Constant,
  Axis,
  Range,
  Memory,
  Rounding,
  Clamp,
  Compare,
  Overflow,
  FastMath,
  Reshape,
  Transpose,
  Scan,
  External,
  Dot
};
enum class BatchInactivePolicy {
  Preserve,
  IEEE,
  Zero,
  Positive,
  Divide,
  Shift,
  FloatToInt
};
enum class BatchRegionContext {
  BlockBody,
  ReduceCombiner,
  ScanCombiner,
  IfThen,
  IfElse,
  ForBody
};

struct BatchOpDescriptor {
  llvm::StringLiteral name;
  char specLevel;
  BatchModel model;
  BatchAttributePolicy attributes;
  BatchInactivePolicy inactiveInputs;
  bool implemented;
};

llvm::ArrayRef<BatchOpDescriptor> getBatchOpRegistry();
const BatchOpDescriptor *lookupBatchOp(llvm::StringRef name);
llvm::StringRef getBatchModelName(BatchModel model);
llvm::StringRef getBatchAttributePolicyName(BatchAttributePolicy policy);
llvm::StringRef getBatchInactivePolicyName(BatchInactivePolicy policy);
bool isBatchOpSupported(llvm::StringRef name);
LogicalResult verifyBatchAttributes(Operation *op,
                                    const BatchOpDescriptor &descriptor);
} // namespace mlir::triton::cfg
#endif
