//===- SelectSimdSimtCostModel.cpp - C++ SIMD/SIMT selection ------------===//
//
// This pass is the online owner of SIMD/SIMT candidate selection.  Python
// only schedules the pass and reacts to its machine-readable execution intent.
// Feature extraction, calibrated scoring, confidence/target/margin gates, and
// mixed-operation planning stay in C++.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/RouteModel/SimtAnchorAnalysis.h"
#include "AscendModel/RouteModel/SimtSelection.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <system_error>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_SELECTSIMDSIMTCOSTMODELPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using namespace simt_selection;

inline constexpr llvm::StringLiteral kRecommendedExecutionAttr =
    "ascend.simt_costmodel.recommended";
inline constexpr llvm::StringLiteral kSelectionSourceAttr =
    "ascend.simt_costmodel.selection_source";
inline constexpr llvm::StringLiteral kRankingConfidenceAttr =
    "ascend.simt_costmodel.ranking_confidence";
inline constexpr llvm::StringLiteral kAllSimdScoreAttr =
    "ascend.simt_costmodel.all_simd_score";
inline constexpr llvm::StringLiteral kAllSimtScoreAttr =
    "ascend.simt_costmodel.all_simt_score";
inline constexpr llvm::StringLiteral kMixedScoreAttr =
    "ascend.simt_costmodel.mixed_score";
inline constexpr llvm::StringLiteral kReportJSONAttr =
    "ascend.simt_costmodel.report_json";

static bool containsExplicitVectorScope(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "scope.scope") {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static void clearPreviousSelection(ModuleOp module) {
  module->removeAttr(kEffectiveExecutionAttr);
  module->removeAttr(kRecommendedExecutionAttr);
  module->removeAttr(kSelectionSourceAttr);
  module->removeAttr(kRankingConfidenceAttr);
  module->removeAttr(kAllSimdScoreAttr);
  module->removeAttr(kAllSimtScoreAttr);
  module->removeAttr(kMixedScoreAttr);
  module->removeAttr(kReportJSONAttr);
}

static LogicalResult appendJSONLine(llvm::StringRef path,
                                    llvm::StringRef json) {
  if (path.empty())
    return success();
  std::error_code error;
  llvm::raw_fd_ostream os(path, error, llvm::sys::fs::OF_Append);
  if (error)
    return failure();
  os << json << '\n';
  return success();
}

struct SelectSimdSimtCostModelPass
    : public impl::SelectSimdSimtCostModelPassBase<
          SelectSimdSimtCostModelPass> {
  using SelectSimdSimtCostModelPassBase::SelectSimdSimtCostModelPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    clearPreviousSelection(module);
    bool autoMode = mode.getValue() == "auto";

    SimdSimtCostModelOptions options;
    options.profilePath = profilePath.getValue();
    options.actualTarget = actualTarget.getValue();
    options.numWarps =
        static_cast<unsigned>(std::max<int64_t>(1, numWarps.getValue()));
    options.marginRatio = std::max(0.0, std::min(1.0, marginRatio.getValue()));
    options.includeFeaturesInJSON = true;
    options.scoreOutsideCalibrationCoverage = !autoMode;
    options.compileOn91095 = compileOn91095.getValue();

    SimtAnchorPlan anchorPlan =
        buildMixedSimtAnchorPlan(module, options.compileOn91095);
    auto reportOr = analyzeSimdSimtCandidates(module, anchorPlan, options);
    if (!reportOr) {
      module.emitError("C++ SIMD/SIMT cost model failed: ")
          << llvm::toString(reportOr.takeError());
      signalPassFailure();
      return;
    }
    SimdSimtCostReport report = std::move(*reportOr);

    std::string recommended =
        report.candidateCostsEvaluated
            ? stringifySimdSimtCandidate(report.decision).str()
            : kBackendDefault.str();
    std::string effective = kBackendDefault.str();
    std::string selectionSource = "backend_default";
    std::string applicationReason;
    SmallVector<Operation *> mixedAnchors;

    bool actionSupported = true;
    bool hasExplicitScope = containsExplicitVectorScope(module);
    if (recommended == kMixedSimdSimt) {
      if (hasExplicitScope) {
        actionSupported = false;
        applicationReason = "explicit_scope_present";
      } else {
        mixedAnchors = anchorPlan.materializableRoots();
        if (mixedAnchors.empty()) {
          actionSupported = false;
          applicationReason = "no_materializable_mixed_anchor";
        }
      }
    } else if (recommended == kAllSimtOnly && hasExplicitScope) {
      // Preserve explicit local SIMD/SIMT/cube scope semantics instead of
      // replacing the whole kernel with a pure-SIMT route.
      actionSupported = false;
      applicationReason = "explicit_scope_present";
    }

    const bool onlyInsufficientGain =
        report.gateReasons.size() == 1 &&
        report.gateReasons.front() ==
            "decision_advantage_not_above_required_gain";
    if (autoMode && report.gatePassed && actionSupported) {
      effective = recommended;
      selectionSource = "cpp_cost_model";
      applicationReason = "cpp_cost_model_admitted";
    } else if (autoMode && onlyInsufficientGain &&
               report.allSimdCandidateLegal) {
      // A failed switch-margin check means "do not leave the established
      // baseline", not "hand control back to legacy backend heuristics".
      // Keeping all-SIMD here avoids accidentally taking the historical
      // compile_mode=simd_simt force path after the model rejected a marginal
      // SIMT/mixed recommendation.
      effective = kAllSimd.str();
      selectionSource = "cpp_cost_model_safe_baseline";
      applicationReason = "decision_gain_below_margin_keep_all_simd";
    } else if (!autoMode) {
      applicationReason = "report_mode";
    } else if (!report.gatePassed) {
      applicationReason = report.gateReasons.empty()
                              ? "model_gate_rejected"
                              : report.gateReasons.front();
    }

    Builder builder(module.getContext());
    module->setAttr(kRecommendedExecutionAttr,
                    builder.getStringAttr(recommended));
    module->setAttr(kEffectiveExecutionAttr, builder.getStringAttr(effective));
    module->setAttr(kSelectionSourceAttr,
                    builder.getStringAttr(selectionSource));
    module->setAttr(kRankingConfidenceAttr,
                    builder.getStringAttr(report.rankingConfidence));
    if (report.candidateCostsEvaluated) {
      module->setAttr(kAllSimdScoreAttr,
                      builder.getF64FloatAttr(report.candidateCosts.allSimd));
      module->setAttr(
          kAllSimtScoreAttr,
          builder.getF64FloatAttr(report.candidateCosts.allSimtOnly));
      module->setAttr(
          kMixedScoreAttr,
          builder.getF64FloatAttr(report.candidateCosts.mixedSimdSimt));
    }

    // Selector and Materializer consume the same immutable anchor plan in one
    // pass invocation.  No per-operation marker is persisted in TTIR.
    if (effective == kMixedSimdSimt &&
        failed(materializeSimtAnchorPlan(module, anchorPlan))) {
      signalPassFailure();
      return;
    }

    llvm::json::Object reportJSON = report.toJSON();
    reportJSON["mode"] = mode.getValue();
    reportJSON["recommended_decision_kind"] = recommended;
    reportJSON["effective_decision_kind"] = effective;
    reportJSON["selection_source"] = selectionSource;
    reportJSON["application_reason"] = applicationReason;
    reportJSON["action_supported"] = actionSupported;
    reportJSON["materialized_simt_anchor_count"] =
        static_cast<int64_t>(mixedAnchors.size());
    std::string json =
        llvm::formatv("{0}", llvm::json::Value(std::move(reportJSON))).str();
    module->setAttr(kReportJSONAttr, builder.getStringAttr(json));

    if (failed(appendJSONLine(reportFile.getValue(), json)))
      module.emitWarning("failed to append C++ SIMD/SIMT report to ")
          << reportFile.getValue();
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
