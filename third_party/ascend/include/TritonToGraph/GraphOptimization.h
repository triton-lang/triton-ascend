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
#include <string>

namespace mlir {
namespace triton {
namespace cfg {

// The identity of every rule is one bit of the pass rule mask, so the mask
// width bounds how many rules can ever be registered.  It is 16-bit because
// the 8-bit range is already fully assigned.
enum class GraphOptimizationRuleId : uint16_t {
  LoadStoreTranspose = 1,
  TransposePointwiseReorder = 2,
  StoreCoalescing = 4,
  // RowCoalescing is a pure-SIMT-only graph rule.  It is intentionally
  // scheduled once after the normal per-function phases because its launch
  // contract must not be suppressed by their rewrite budget.
  RowCoalescing = 8,
  // DiagonalMaskRemoval rewrites compute logic rather than memory access, so
  // it is architecture and template independent.
  DiagonalMaskRemoval = 128,
  // ConvertModuloToMask linearizes wrapped tile addresses.  It runs before the
  // memory-access rules so they see the linear form.
  ConvertModuloToMask = 256,
  // The following identities are owned by the layout/memory compatibility
  // passes.  They deliberately are not GraphOptimizationRule candidates and
  // are not added to GraphOptimizePass's per-function phase loop.
  StridedAxisCoalescing = 16,
  ChunkCoalescing = 32,
  StridedLoadStoreRewrite = 64,
};

constexpr const char *
getGraphOptimizationRuleName(GraphOptimizationRuleId rule) {
  switch (rule) {
  case GraphOptimizationRuleId::LoadStoreTranspose:
    return "LoadStoreTranspose";
  case GraphOptimizationRuleId::TransposePointwiseReorder:
    return "TransposePointwiseReorder";
  case GraphOptimizationRuleId::StoreCoalescing:
    return "StoreCoalescing";
  case GraphOptimizationRuleId::RowCoalescing:
    return "RowCoalescing";
  case GraphOptimizationRuleId::DiagonalMaskRemoval:
    return "DiagonalMaskRemoval";
  case GraphOptimizationRuleId::ConvertModuloToMask:
    return "ConvertModuloToMask";
  case GraphOptimizationRuleId::StridedAxisCoalescing:
    return "StridedAxisCoalescing";
  case GraphOptimizationRuleId::ChunkCoalescing:
    return "ChunkCoalescing";
  case GraphOptimizationRuleId::StridedLoadStoreRewrite:
    return "StridedLoadStoreRewrite";
  }
  return "Unknown";
}

constexpr uint16_t getGraphOptimizationRuleMask(GraphOptimizationRuleId rule) {
  return static_cast<uint16_t>(rule);
}

constexpr uint16_t kAllGraphOptimizationRuleMask =
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::LoadStoreTranspose) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::TransposePointwiseReorder) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::StoreCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::RowCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::DiagonalMaskRemoval) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::ConvertModuloToMask) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StridedAxisCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::ChunkCoalescing) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StridedLoadStoreRewrite);

constexpr bool isValidGraphOptimizationRuleMask(uint16_t ruleMask) {
  constexpr uint16_t unknownRuleBits =
      static_cast<uint16_t>(~kAllGraphOptimizationRuleMask);
  return (ruleMask & unknownRuleBits) == 0;
}

struct GraphOptimizationOptions {
  // A zero mask intentionally disables every native GraphOptimizationRule.
  // Layout/memory compatibility stages retain their original fixed scheduling
  // and do not use this option as a new opt-out.
  uint16_t enabledRuleMask = kAllGraphOptimizationRuleMask;
  unsigned maxRewritesPerFunction = 64;
  unsigned ubCapacityBytes = 0;
  // RowCoalescing changes the launch grid and is valid only for
  // compile_mode="simt_only".  Keep the source selector rather than a
  // second derived force flag so every consumer follows one mode contract.
  std::string compileMode = "simd_simt_template";
};

std::unique_ptr<OperationPass<ModuleOp>>
createGraphOptimizePass(GraphOptimizationOptions options = {});

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H
