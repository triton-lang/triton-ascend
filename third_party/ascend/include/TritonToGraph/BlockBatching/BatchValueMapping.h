// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_TO_GRAPH_BLOCK_BATCHING_VALUE_MAPPING_H
#define TRITON_TO_GRAPH_BLOCK_BATCHING_VALUE_MAPPING_H
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include <cassert>

namespace mlir::triton::cfg {
enum class BatchVariation { Unknown, Uniform, Varying };
struct BatchValueInfo {
  Type originalType;
  Value rewritten;
  BatchVariation variation = BatchVariation::Unknown;
};
// Every result has its own record. Missing facts are never interpreted as U.
class BatchValueMapping {
public:
  IRMapping mapping;
  void map(Value original, Value rewritten, bool varying) {
    mapping.map(original, rewritten);
    facts[original] = {original.getType(), rewritten,
                       varying ? BatchVariation::Varying
                               : BatchVariation::Uniform};
  }
  bool contains(Value value) const { return facts.count(value); }
  bool isVarying(Value value) const {
    auto it = facts.find(value);
    assert(it != facts.end() && "missing batch value fact");
    return it->second.variation == BatchVariation::Varying;
  }

private:
  llvm::DenseMap<Value, BatchValueInfo> facts;
};
} // namespace mlir::triton::cfg
#endif
