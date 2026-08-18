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

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

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

constexpr std::array<GraphOptimizationRuleId, 5> kRulePhases = {
    // DiagonalMaskRemoval runs first because it deletes a quadratic
    // intermediate tensor, so the later phases match and budget UB against the
    // already shrunken IR.
    GraphOptimizationRuleId::DiagonalMaskRemoval,
    // ConvertModuloToMask runs before the memory-access phases so that they see
    // linear tile addresses instead of wrapped ones.
    GraphOptimizationRuleId::ConvertModuloToMask,
    GraphOptimizationRuleId::LoadStoreTranspose,
    GraphOptimizationRuleId::TransposePointwiseReorder,
    GraphOptimizationRuleId::StoreCoalescing,
};

using ProgramOrderMap = llvm::DenseMap<Operation *, unsigned>;

ProgramOrderMap buildProgramOrderMap(triton::FuncOp function) {
  ProgramOrderMap programOrder;
  unsigned nextOrder = 0;
  function.walk(
      [&](Operation *operation) { programOrder[operation] = nextOrder++; });
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

bool isRuleEnabled(uint16_t ruleMask, GraphOptimizationRuleId ruleId) {
  return (ruleMask & getGraphOptimizationRuleMask(ruleId)) != 0;
}

bool isGraphDiagnosticsRule(GraphOptimizationRuleId ruleId) {
  switch (ruleId) {
  case GraphOptimizationRuleId::LoadStoreTranspose:
  case GraphOptimizationRuleId::TransposePointwiseReorder:
  case GraphOptimizationRuleId::StoreCoalescing:
  case GraphOptimizationRuleId::RowCoalescing:
    return true;
  default:
    return false;
  }
}

llvm::StringRef getGraphRuleName(GraphOptimizationRuleId ruleId) {
  switch (ruleId) {
  case GraphOptimizationRuleId::LoadStoreTranspose:
    return "LoadStoreTranspose";
  case GraphOptimizationRuleId::TransposePointwiseReorder:
    return "TransposePointwiseReorder";
  case GraphOptimizationRuleId::StoreCoalescing:
    return "StoreCoalescing";
  case GraphOptimizationRuleId::RowCoalescing:
    return "RowCoalescing";
  default:
    return "Unknown";
  }
}

void writeGraphLogPrefix() { llvm::errs() << "[GRAPH] "; }

void logCandidateSummary(GraphOptimizationRuleId ruleId, size_t count,
                         unsigned epoch) {
  writeGraphLogPrefix();
  llvm::errs() << "event=candidates rule=" << getGraphRuleName(ruleId)
               << " count=" << count << " epoch=" << epoch << '\n';
}

void logPlanEvent(llvm::StringRef event, const RewritePlan &plan,
                  unsigned ordinal, unsigned epoch,
                  llvm::StringRef suffix = "") {
  writeGraphLogPrefix();
  llvm::errs() << "event=" << event
               << " rule=" << getGraphRuleName(plan.getRuleId())
               << " benefit=" << plan.getBenefit() << " ordinal=" << ordinal
               << " epoch=" << epoch;
  plan.printDebug(llvm::errs());
  if (!suffix.empty())
    llvm::errs() << ' ' << suffix;
  llvm::errs() << '\n';
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
    this->forceSimtOnly = options.forceSimtOnly;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override;

private:
  LogicalResult getStableOptions(GraphOptimizationOptions &options);
};

LogicalResult
GraphOptimizePass::getStableOptions(GraphOptimizationOptions &options) {
  const uint64_t cliRuleMask = this->ruleMask;
  const uint64_t cliMaxRewrites = this->maxRewritesPerFunction;
  const uint64_t cliUBCapacityBytes = this->ubCapacityBytes;

  if (cliRuleMask > std::numeric_limits<uint16_t>::max() ||
      !isValidGraphOptimizationRuleMask(static_cast<uint16_t>(cliRuleMask))) {
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

  options.enabledRuleMask = static_cast<uint16_t>(cliRuleMask);
  options.maxRewritesPerFunction = static_cast<unsigned>(cliMaxRewrites);
  options.ubCapacityBytes = static_cast<unsigned>(cliUBCapacityBytes);
  options.emitRemarks = this->emitRemarks;
  options.forceSimtOnly = this->forceSimtOnly;
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
    unsigned rowRewriteCount = 0;
    unsigned staleSkipped = 0;
    unsigned revalidationSkipped = 0;
    bool budgetExhausted = false;
    std::array<unsigned, 4> successfulRewrites = {};

    auto recordSuccessfulRewrite = [&](GraphOptimizationRuleId ruleId) {
      switch (ruleId) {
      case GraphOptimizationRuleId::LoadStoreTranspose:
        ++successfulRewrites[0];
        break;
      case GraphOptimizationRuleId::TransposePointwiseReorder:
        ++successfulRewrites[1];
        break;
      case GraphOptimizationRuleId::StoreCoalescing:
        ++successfulRewrites[2];
        break;
      case GraphOptimizationRuleId::RowCoalescing:
        ++successfulRewrites[3];
        break;
      default:
        break;
      }
    };

    writeGraphLogPrefix();
    llvm::errs() << "event=pass-start function=" << function.getName()
                 << " rule-mask="
                 << static_cast<unsigned>(options.enabledRuleMask)
                 << " max-rewrites=" << options.maxRewritesPerFunction
                 << " ub-capacity-bytes=" << options.ubCapacityBytes
                 << " force-simt-only="
                 << (options.forceSimtOnly ? "true" : "false") << '\n';

    for (GraphOptimizationRuleId phase : kRulePhases) {
      if (!isRuleEnabled(options.enabledRuleMask, phase))
        continue;

      const bool logPhase = isGraphDiagnosticsRule(phase);
      bool loggedCandidateSummary = false;
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
        if (logPhase && !loggedCandidateSummary) {
          logCandidateSummary(phase, plans.size(), context.getEpoch());
          loggedCandidateSummary = true;
        }
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
        unsigned selectedOrdinal = 0;
        unsigned ordinal = 0;
        for (std::unique_ptr<RewritePlan> &plan : plans) {
          ++ordinal;
          if (plan->getCreationEpoch() != context.getEpoch()) {
            ++staleSkipped;
            continue;
          }
          if (failed(plan->revalidate(context))) {
            ++revalidationSkipped;
            continue;
          }

          selectedPlan = std::move(plan);
          selectedOrdinal = ordinal;
          break;
        }

        if (!selectedPlan) {
          plans.clear();
          break;
        }

        const GraphOptimizationRuleId appliedRuleId = selectedPlan->getRuleId();
        if (logPhase)
          logPlanEvent("plan-selected", *selectedPlan, selectedOrdinal,
                       context.getEpoch());
        IRRewriter rewriter(&getContext());
        if (failed(selectedPlan->apply(rewriter))) {
          selectedPlan.reset();
          plans.clear();
          function.emitError() << "graph-optimize failed to apply rewrite";
          signalPassFailure();
          return;
        }
        if (logPhase)
          logPlanEvent("apply-ok", *selectedPlan, selectedOrdinal,
                       context.getEpoch());

        if (options.emitRemarks)
          function.emitRemark() << "applied graph optimization rule "
                                << static_cast<unsigned>(appliedRuleId);

        // Plans can retain pointers into analysis results, so destroy all of
        // them before invalidating the context for the next IR epoch.
        selectedPlan.reset();
        plans.clear();
        context.invalidate();
        ++rewriteCount;
        if (logPhase)
          recordSuccessfulRewrite(appliedRuleId);
      }

      if (logPhase && !loggedCandidateSummary) {
        writeGraphLogPrefix();
        llvm::errs() << "event=candidates rule=" << getGraphRuleName(phase)
                     << " status=not-scanned reason=rewrite-budget-exhausted"
                     << " epoch=" << context.getEpoch() << '\n';
      }

      if (rewriteCount == options.maxRewritesPerFunction) {
        budgetExhausted = true;
        break;
      }
    }

    if (budgetExhausted) {
      writeGraphLogPrefix();
      llvm::errs() << "event=budget-exhausted function=" << function.getName()
                   << " rewrites=" << rewriteCount
                   << " max-rewrites=" << options.maxRewritesPerFunction
                   << '\n';
    }

    // RowCoalescing has the same function-local candidate/rewrite interface
    // as the native graph rules, but it has distinct launch semantics.  Its
    // historical pass ran once and was not subject to the generic rewrite
    // budget, so run it once after LoadStoreTranspose,
    // TransposePointwiseReorder, and StoreCoalescing even when that budget
    // has already been exhausted.
    if (isRuleEnabled(options.enabledRuleMask,
                      GraphOptimizationRuleId::RowCoalescing)) {
      if (!options.forceSimtOnly) {
        writeGraphLogPrefix();
        llvm::errs() << "event=rule-skip rule=RowCoalescing"
                     << " reason=force-simt-only-false\n";
      } else {
        GraphOptimizationRule *rowRule = nullptr;
        for (GraphOptimizationRule *rule : enabledRules) {
          if (rule->getId() == GraphOptimizationRuleId::RowCoalescing) {
            rowRule = rule;
            break;
          }
        }
        if (!rowRule) {
          writeGraphLogPrefix();
          llvm::errs() << "event=rule-skip rule=RowCoalescing"
                       << " reason=not-registered\n";
        } else {
          if (failed(context.ensure(rowRule->getAnalysisRequirements()))) {
            function.emitError()
                << "graph-optimize failed to build Row analyses";
            signalPassFailure();
            return;
          }

          SmallVector<std::unique_ptr<RewritePlan>> rowPlans;
          if (failed(rowRule->findCandidates(context, rowPlans))) {
            function.emitError()
                << "graph-optimize failed to discover Row candidate";
            signalPassFailure();
            return;
          }
          rowPlans.erase(
              std::remove_if(
                  rowPlans.begin(), rowPlans.end(),
                  [](const std::unique_ptr<RewritePlan> &plan) {
                    return !plan || plan->getRuleId() !=
                                        GraphOptimizationRuleId::RowCoalescing;
                  }),
              rowPlans.end());
          logCandidateSummary(GraphOptimizationRuleId::RowCoalescing,
                              rowPlans.size(), context.getEpoch());

          if (!rowPlans.empty()) {
            ProgramOrderMap programOrder = buildProgramOrderMap(function);
            std::stable_sort(
                rowPlans.begin(), rowPlans.end(),
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

            std::unique_ptr<RewritePlan> selectedRowPlan;
            unsigned selectedRowOrdinal = 0;
            unsigned ordinal = 0;
            for (std::unique_ptr<RewritePlan> &plan : rowPlans) {
              ++ordinal;
              if (plan->getCreationEpoch() != context.getEpoch()) {
                ++staleSkipped;
                continue;
              }
              if (failed(plan->revalidate(context))) {
                ++revalidationSkipped;
                continue;
              }
              selectedRowPlan = std::move(plan);
              selectedRowOrdinal = ordinal;
              break;
            }

            if (selectedRowPlan) {
              logPlanEvent("plan-selected", *selectedRowPlan,
                           selectedRowOrdinal, context.getEpoch(),
                           "sandbox=not-run");
              IRRewriter rewriter(&getContext());
              if (failed(selectedRowPlan->apply(rewriter))) {
                selectedRowPlan.reset();
                rowPlans.clear();
                function.emitError()
                    << "graph-optimize failed to apply Row rewrite";
                signalPassFailure();
                return;
              }
              logPlanEvent("apply-ok", *selectedRowPlan, selectedRowOrdinal,
                           context.getEpoch(), "sandbox=verified");

              if (options.emitRemarks)
                function.emitRemark()
                    << "applied graph optimization rule "
                    << static_cast<unsigned>(
                           GraphOptimizationRuleId::RowCoalescing);

              selectedRowPlan.reset();
              rowPlans.clear();
              context.invalidate();
              ++rowRewriteCount;
              recordSuccessfulRewrite(GraphOptimizationRuleId::RowCoalescing);
            }
          }
        }
      }
    }

    writeGraphLogPrefix();
    llvm::errs() << "event=pass-end function=" << function.getName()
                 << " rewrites-general=" << rewriteCount
                 << " rewrites-row=" << rowRewriteCount
                 << " rewrites-total=" << rewriteCount + rowRewriteCount
                 << " success-load-store=" << successfulRewrites[0]
                 << " success-transpose-pointwise=" << successfulRewrites[1]
                 << " success-store-coalescing=" << successfulRewrites[2]
                 << " success-row-coalescing=" << successfulRewrites[3]
                 << " stale-skipped=" << staleSkipped
                 << " revalidation-skipped=" << revalidationSkipped
                 << " budget-exhausted=" << (budgetExhausted ? "true" : "false")
                 << '\n';
  }
}

} // namespace

void populateBuiltinGraphOptimizationRules(
    const GraphOptimizationOptions &options,
    SmallVectorImpl<std::unique_ptr<GraphOptimizationRule>> &rules) {
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::DiagonalMaskRemoval)) {
    rules.push_back(createDiagonalMaskRemovalRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::ConvertModuloToMask)) {
    rules.push_back(createConvertModuloToMaskRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::LoadStoreTranspose)) {
    rules.push_back(createLoadStoreTransposeRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::TransposePointwiseReorder)) {
    rules.push_back(createTransposePointwiseReorderRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::StoreCoalescing)) {
    rules.push_back(createStoreCoalescingRule(options.ubCapacityBytes));
  }
  if (options.forceSimtOnly &&
      isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::RowCoalescing)) {
    rules.push_back(createRowCoalescingRule());
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createGraphOptimizePass(GraphOptimizationOptions options) {
  return std::make_unique<GraphOptimizePass>(options);
}

} // namespace cfg
} // namespace triton
} // namespace mlir
