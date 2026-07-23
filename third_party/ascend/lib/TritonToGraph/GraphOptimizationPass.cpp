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

#include "TritonToGraph/GraphOptimizationContext.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <utility>

namespace mlir {
namespace triton {
namespace cfg {
#define GEN_PASS_DEF_GRAPHOPTIMIZE
#include "ascend/include/TritonToGraph/Passes.h.inc"
} // namespace cfg
} // namespace triton
} // namespace mlir

namespace mlir {
namespace triton {
namespace cfg {
namespace {

constexpr std::array<GraphOptimizationRuleId, 3> kRulePhases = {
    GraphOptimizationRuleId::LoadStoreTranspose,
    GraphOptimizationRuleId::TransposePointwiseReorder,
    GraphOptimizationRuleId::UBPreload,
};

using ProgramOrderMap = llvm::DenseMap<Operation *, unsigned>;

ProgramOrderMap buildProgramOrderMap(triton::FuncOp function) {
  ProgramOrderMap programOrder;
  unsigned nextOrder = 0;
  function.walk([&](Operation *operation) {
    programOrder[operation] = nextOrder++;
  });
  return programOrder;
}

unsigned getProgramOrder(const RewritePlan &plan,
                         const ProgramOrderMap &programOrder) {
  Operation *anchor = plan.getAnchor();
  if (!anchor)
    return std::numeric_limits<unsigned>::max();

  auto it = programOrder.find(anchor);
  if (it == programOrder.end())
    return std::numeric_limits<unsigned>::max();
  return it->second;
}

bool isRuleEnabled(uint8_t ruleMask, GraphOptimizationRuleId ruleId) {
  return (ruleMask & getGraphOptimizationRuleMask(ruleId)) != 0;
}

class GraphOptimizePass final
    : public impl::GraphOptimizeBase<GraphOptimizePass> {
public:
  GraphOptimizePass() = default;

  explicit GraphOptimizePass(const GraphOptimizationOptions &options) {
    this->ruleMask = options.enabledRuleMask;
    this->maxRewritesPerFunction = options.maxRewritesPerFunction;
    this->ubCapacityBytes = options.ubCapacityBytes;
    this->emitRemarks = options.emitRemarks;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;

private:
  LogicalResult getStableOptions(GraphOptimizationOptions &options);
};

LogicalResult GraphOptimizePass::getStableOptions(
    GraphOptimizationOptions &options) {
  const uint64_t cliRuleMask = this->ruleMask;
  const uint64_t cliMaxRewrites = this->maxRewritesPerFunction;
  const uint64_t cliUBCapacityBytes = this->ubCapacityBytes;

  if (cliRuleMask > std::numeric_limits<uint8_t>::max() ||
      !isValidGraphOptimizationRuleMask(
          static_cast<uint8_t>(cliRuleMask))) {
    getOperation().emitError()
        << "graph-optimize rule-mask contains unknown or out-of-range bits: "
        << cliRuleMask;
    return failure();
  }

  if (cliMaxRewrites > std::numeric_limits<unsigned>::max()) {
    getOperation().emitError()
        << "graph-optimize max-rewrites-per-function is out of range: "
        << cliMaxRewrites;
    return failure();
  }

  if (cliUBCapacityBytes > std::numeric_limits<unsigned>::max()) {
    getOperation().emitError()
        << "graph-optimize ub-capacity-bytes is out of range: "
        << cliUBCapacityBytes;
    return failure();
  }

  options.enabledRuleMask = static_cast<uint8_t>(cliRuleMask);
  options.maxRewritesPerFunction = static_cast<unsigned>(cliMaxRewrites);
  options.ubCapacityBytes = static_cast<unsigned>(cliUBCapacityBytes);
  options.emitRemarks = this->emitRemarks;
  return success();
}

void GraphOptimizePass::runOnOperation() {
  GraphOptimizationOptions options;
  if (failed(getStableOptions(options))) {
    signalPassFailure();
    return;
  }

  SmallVector<std::unique_ptr<GraphOptimizationRule>> ownedRules;
  populateBuiltinGraphOptimizationRules(options, ownedRules);

  SmallVector<GraphOptimizationRule *> enabledRules;
  for (const std::unique_ptr<GraphOptimizationRule> &rule : ownedRules) {
    if (rule && isRuleEnabled(options.enabledRuleMask, rule->getId()))
      enabledRules.push_back(rule.get());
  }

  ModuleOp module = getOperation();
  for (triton::FuncOp function : module.getOps<triton::FuncOp>()) {
    GraphOptimizationContext context(function);
    unsigned rewriteCount = 0;

    for (GraphOptimizationRuleId phase : kRulePhases) {
      if (!isRuleEnabled(options.enabledRuleMask, phase))
        continue;

      while (rewriteCount < options.maxRewritesPerFunction) {
        ProgramOrderMap programOrder = buildProgramOrderMap(function);
        SmallVector<std::unique_ptr<RewritePlan>> plans;

        for (GraphOptimizationRule *rule : enabledRules) {
          if (rule->getId() != phase)
            continue;

          if (failed(context.ensure(rule->getAnalysisRequirements()))) {
            function.emitError() << "graph-optimize failed to build analyses";
            signalPassFailure();
            return;
          }

          if (failed(rule->findCandidates(context, plans))) {
            function.emitError()
                << "graph-optimize failed to discover rewrite candidates";
            signalPassFailure();
            return;
          }
        }

        plans.erase(
            std::remove_if(plans.begin(), plans.end(),
                           [phase](const std::unique_ptr<RewritePlan> &plan) {
                             return !plan || plan->getRuleId() != phase;
                           }),
            plans.end());
        if (plans.empty())
          break;

        std::stable_sort(
            plans.begin(), plans.end(),
            [&programOrder](const std::unique_ptr<RewritePlan> &lhs,
                            const std::unique_ptr<RewritePlan> &rhs) {
              if (lhs->getBenefit() != rhs->getBenefit())
                return lhs->getBenefit() > rhs->getBenefit();

              const unsigned lhsOrder = getProgramOrder(*lhs, programOrder);
              const unsigned rhsOrder = getProgramOrder(*rhs, programOrder);
              if (lhsOrder != rhsOrder)
                return lhsOrder < rhsOrder;

              return static_cast<unsigned>(lhs->getRuleId()) <
                     static_cast<unsigned>(rhs->getRuleId());
            });

        std::unique_ptr<RewritePlan> selectedPlan;
        for (std::unique_ptr<RewritePlan> &plan : plans) {
          if (plan->getCreationEpoch() != context.getEpoch())
            continue;
          if (failed(plan->revalidate(context)))
            continue;

          selectedPlan = std::move(plan);
          break;
        }

        if (!selectedPlan) {
          plans.clear();
          break;
        }

        const GraphOptimizationRuleId appliedRuleId =
            selectedPlan->getRuleId();
        IRRewriter rewriter(&getContext());
        if (failed(selectedPlan->apply(rewriter))) {
          selectedPlan.reset();
          plans.clear();
          function.emitError() << "graph-optimize failed to apply rewrite";
          signalPassFailure();
          return;
        }

        if (options.emitRemarks)
          function.emitRemark()
              << "applied graph optimization rule "
              << static_cast<unsigned>(appliedRuleId);

        // Plans can retain pointers into analysis results, so destroy all of
        // them before invalidating the context for the next IR epoch.
        selectedPlan.reset();
        plans.clear();
        context.invalidate();
        ++rewriteCount;
      }

      if (rewriteCount == options.maxRewritesPerFunction)
        break;
    }
  }
}

} // namespace

void populateBuiltinGraphOptimizationRules(
    const GraphOptimizationOptions &options,
    SmallVectorImpl<std::unique_ptr<GraphOptimizationRule>> &rules) {
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::LoadStoreTranspose)) {
    rules.push_back(createLoadStoreTransposeRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::TransposePointwiseReorder)) {
    rules.push_back(createTransposePointwiseReorderRule());
  }
  if (isRuleEnabled(options.enabledRuleMask, GraphOptimizationRuleId::UBPreload)) {
    rules.push_back(createUBPreloadRule(options.ubCapacityBytes));
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createGraphOptimizePass(GraphOptimizationOptions options) {
  return std::make_unique<GraphOptimizePass>(options);
}

} // namespace cfg
} // namespace triton
} // namespace mlir
