/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H
#define TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>

namespace mlir {
namespace triton {
namespace cfg {

enum class GraphOptimizationRuleId : uint8_t {
  LoadStoreTranspose = 1,
  TransposePointwiseReorder = 2,
  UBPreload = 4,
};

constexpr uint8_t getGraphOptimizationRuleMask(GraphOptimizationRuleId rule) {
  return static_cast<uint8_t>(rule);
}

constexpr uint8_t kAllGraphOptimizationRuleMask =
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::LoadStoreTranspose) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::TransposePointwiseReorder) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::UBPreload);

constexpr bool isValidGraphOptimizationRuleMask(uint8_t ruleMask) {
  return (ruleMask & static_cast<uint8_t>(~kAllGraphOptimizationRuleMask)) ==
         0;
}

struct GraphOptimizationOptions {
  // A zero mask intentionally disables every graph optimization rule.
  uint8_t enabledRuleMask = kAllGraphOptimizationRuleMask;
  unsigned maxRewritesPerFunction = 64;
  unsigned ubCapacityBytes = 0;
  bool emitRemarks = false;
};

std::unique_ptr<OperationPass<ModuleOp>>
createGraphOptimizePass(GraphOptimizationOptions options = {});

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H
