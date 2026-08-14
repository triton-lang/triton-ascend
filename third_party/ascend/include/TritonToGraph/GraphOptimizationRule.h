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

#ifndef TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_RULE_H
#define TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_RULE_H

#include "TritonToGraph/GraphOptimization.h"
#include "TritonToGraph/GraphOptimizationContext.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace mlir {
namespace triton {
namespace cfg {

// A plan is created for one immutable GraphOptimizationContext epoch. Its
// apply() implementation must be transactional: on failure it leaves the IR
// unchanged.
class RewritePlan {
public:
  virtual ~RewritePlan() = default;

  virtual GraphOptimizationRuleId getRuleId() const = 0;
  virtual unsigned getBenefit() const = 0;
  virtual Operation *getAnchor() const = 0;
  virtual unsigned getCreationEpoch() const = 0;

  // Failure means this plan is no longer applicable to the current epoch.
  virtual LogicalResult revalidate(GraphOptimizationContext &context) const = 0;
  virtual LogicalResult apply(IRRewriter &rewriter) = 0;
};

class GraphOptimizationRule {
public:
  virtual ~GraphOptimizationRule() = default;

  virtual GraphOptimizationRuleId getId() const = 0;
  virtual AnalysisRequirement getAnalysisRequirements() const = 0;

  // Candidate discovery may inspect analyses but must not mutate IR.
  virtual LogicalResult
  findCandidates(GraphOptimizationContext &context,
                 SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) = 0;
};

// Internal extension point for the builtin rule set. Concrete rules receive a
// stable copy of the pass options when they are constructed.
void populateBuiltinGraphOptimizationRules(
    const GraphOptimizationOptions &options,
    SmallVectorImpl<std::unique_ptr<GraphOptimizationRule>> &rules);

std::unique_ptr<GraphOptimizationRule> createTransposePointwiseReorderRule();
std::unique_ptr<GraphOptimizationRule> createLoadStoreTransposeRule();
std::unique_ptr<GraphOptimizationRule>
createStoreCoalescingRule(unsigned ubCapacityBytes);
std::unique_ptr<GraphOptimizationRule> createRowCoalescingRule();
std::unique_ptr<GraphOptimizationRule> createDiagonalMaskRemovalRule();
std::unique_ptr<GraphOptimizationRule> createConvertModuloToMaskRule();

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_RULE_H
