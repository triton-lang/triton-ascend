//===- SelectSimdSimtCostModel.cpp - C++ SIMD/SIMT selection ------------===//
//
// This pass is the online owner of SIMD/SIMT candidate selection.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Transforms/SimtSelection.h"
#include "ascend/include/Utils/SuperBlockFactor.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

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
inline constexpr llvm::StringLiteral kAllSimdScoreAttr =
    "ascend.simt_costmodel.all_simd_score";
inline constexpr llvm::StringLiteral kAllSimtScoreAttr =
    "ascend.simt_costmodel.all_simt_score";
inline constexpr llvm::StringLiteral kMixedScoreAttr =
    "ascend.simt_costmodel.mixed_score";
inline constexpr llvm::StringLiteral kReportJSONAttr =
    "ascend.simt_costmodel.report_json";
inline constexpr llvm::StringLiteral kSuperblockFactorAttr =
    "ascend.simt_costmodel.superblock_factor";
inline constexpr llvm::StringLiteral kRuntimeScalarValueAttr =
    "ascend.costmodel.runtime_value";

static LogicalResult
attachRuntimeScalarValues(ModuleOp module,
                          const llvm::json::Object &capability) {
  const auto *values = capability.getArray("runtime_scalar_values");
  if (!values)
    return success();

  llvm::SmallVector<std::pair<unsigned, int64_t>> parsed;
  for (const llvm::json::Value &value : *values) {
    const auto *entry = value.getAsObject();
    if (!entry)
      return failure();
    auto argumentIndex = entry->getInteger("argument_index");
    auto runtimeValue = entry->getInteger("value");
    if (!argumentIndex || *argumentIndex < 0 || !runtimeValue)
      return failure();
    parsed.emplace_back(static_cast<unsigned>(*argumentIndex), *runtimeValue);
  }

  Builder builder(module.getContext());
  bool attached = parsed.empty();
  module.walk([&](FunctionOpInterface function) {
    auto visibility = function->getAttrOfType<StringAttr>("sym_visibility");
    if (visibility && visibility.getValue() != "public")
      return;
    if (llvm::any_of(parsed, [&](const auto &entry) {
          return entry.first >= function.getNumArguments();
        }))
      return;
    for (const auto &[argumentIndex, runtimeValue] : parsed)
      function.setArgAttr(argumentIndex, kRuntimeScalarValueAttr,
                          builder.getI64IntegerAttr(runtimeValue));
    attached = true;
  });
  return success(attached);
}

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
  module->removeAttr(kAllSimdScoreAttr);
  module->removeAttr(kAllSimtScoreAttr);
  module->removeAttr(kMixedScoreAttr);
  module->removeAttr(kReportJSONAttr);
  module->removeAttr(kSuperblockFactorAttr);
}

static SimtAnchorPlan
buildSelectedMixedAnchorPlan(const StageCostModelSummary &stageModel,
                             const SimtAnchorPlan &completePlan) {
  SimtAnchorPlan selected;
  selected.kernelLowerability = completePlan.kernelLowerability;
  if (!stageModel.mixed.legal ||
      stageModel.mixed.implementations.size() != stageModel.stages.size())
    return selected;

  llvm::DenseSet<unsigned> included;
  for (size_t stageIndex = 0; stageIndex < stageModel.stages.size();
       ++stageIndex) {
    const LogicalStageCost &stage = stageModel.stages[stageIndex];
    const StageImplementation &implementation =
        stageModel.mixed.implementations[stageIndex];
    if (implementation.mode != StageMode::SIMT)
      continue;

    llvm::SmallVector<unsigned> stageAnchorIndices;
    for (unsigned index : stage.simtAnchorIndices)
      if (index < completePlan.anchors.size() && included.insert(index).second)
        stageAnchorIndices.push_back(index);
    auto merged = mergeSimtStageAnchors(completePlan, stageAnchorIndices);
    if (!stageAnchorIndices.empty() && !merged)
      return SimtAnchorPlan{};
    if (merged)
      selected.anchors.push_back(std::move(*merged));
  }
  return selected;
}

static bool
anchorPlansHaveCompatibleIndices(const SimtAnchorPlan &analysis,
                                 const SimtAnchorPlan &materialization) {
  if (analysis.anchors.size() != materialization.anchors.size())
    return false;
  for (auto [analysisAnchor, materializationAnchor] :
       llvm::zip_equal(analysis.anchors, materialization.anchors))
    if (analysisAnchor.kind != materializationAnchor.kind ||
        analysisAnchor.materializable != materializationAnchor.materializable)
      return false;
  return true;
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
    const bool autoMode = mode.getValue() == "auto";

    OwningOpRef<ModuleOp> parsedAnalysisModule;
    ModuleOp analysisModule = module;
    if (!analysisModulePath.getValue().empty()) {
      parsedAnalysisModule = parseSourceFile<ModuleOp>(
          analysisModulePath.getValue(), module.getContext());
      if (!parsedAnalysisModule) {
        module.emitError("failed to parse SIMD/SIMT analysis module: ")
            << analysisModulePath.getValue();
        signalPassFailure();
        return;
      }
      analysisModule = *parsedAnalysisModule;
    }

    SimdSimtCostModelOptions options;
    options.profilePath = profilePath.getValue();
    options.actualTarget = actualTarget.getValue();
    // The disposable analysis module may contain AutoBlockify scheduling
    // loops. Cube completion matches the route-neutral executable workload.
    options.cubeCompositionModule = module;
    options.numWarps =
        static_cast<unsigned>(std::max<int64_t>(1, numWarps.getValue()));
    options.includeFeaturesInJSON = true;
    const char *detailedReport =
        std::getenv("TRITON_ASCEND_COSTMODEL_DETAILED_REPORT");
    const bool includeDiagnosticStageDetails =
        !autoMode || (detailedReport && *detailedReport &&
                      llvm::StringRef(detailedReport) != "0");
    options.includeDiagnosticStageDetailsInJSON = includeDiagnosticStageDetails;
    options.collectCalibrationEstimates = includeDiagnosticStageDetails;
    options.compileOn91095 = compileOn91095.getValue();
    options.wholeKernelSuperblockMaterializable =
        wholeKernelSuperblockMaterializable.getValue();
    options.scopeSuperblockMaterializable =
        scopeSuperblockMaterializable.getValue();
    options.maximumWholeKernelSuperblockFactor =
        std::max<int64_t>(1, maximumWholeKernelSuperblockFactor.getValue());
    options.maximumScopeSuperblockFactor =
        std::max<int64_t>(1, maximumScopeSuperblockFactor.getValue());
    options.logicalProgramCountHint =
        std::max<int64_t>(0, logicalProgramCountHint.getValue());
    if (auto capability =
            llvm::json::parse(routeTransformCapabilityJSON.getValue()))
      if (auto *object = capability->getAsObject()) {
        if (const auto *compile = object->getObject("compile_options")) {
          if (auto stages = compile->getInteger("num_stages"))
            if (*stages > 0 && *stages <= std::numeric_limits<unsigned>::max())
              options.numStages = static_cast<unsigned>(*stages);
          if (auto enabled = compile->getBoolean("enable_dynamic_cv_pipeline"))
            options.dynamicCvPipeline = *enabled;
          if (auto enabled = compile->getBoolean("debug"))
            options.debug = *enabled;
        }
        if (const auto *launch = object->getObject("launch_options"))
          if (auto disabled = launch->getBoolean("disable_ffts"))
            options.disableFfts = *disabled;
        if (failed(attachRuntimeScalarValues(analysisModule, *object))) {
          module.emitError("invalid runtime scalar values in route capability");
          signalPassFailure();
          return;
        }
        if (auto count =
                object->getInteger("source_logical_program_count_hint"))
          options.sourceLogicalProgramCountHint = std::max<int64_t>(0, *count);
        if (auto count = object->getInteger("physical_vector_core_count_hint"))
          options.physicalVectorCoreCountHint = std::max<int64_t>(0, *count);
        if (auto count = object->getInteger("physical_aicore_count_hint"))
          options.physicalAiCoreCountHint = std::max<int64_t>(0, *count);
        if (auto count = object->getInteger("cv_aiv_subblock_count")) {
          if (*count != 1 && *count != 2) {
            module.emitError("invalid CV AIV subblock count");
            signalPassFailure();
            return;
          }
          options.cvAivSubblocks = *count;
        }
        if (auto forced = object->getString("backend_forced_route_candidate")) {
          if (*forced == kAllSimd)
            options.backendForcedDecision = SimdSimtCandidateKind::AllSIMD;
          else if (*forced == kAllSimtOnly)
            options.backendForcedDecision = SimdSimtCandidateKind::AllSIMTOnly;
          else if (*forced == kMixedSimdSimt)
            options.backendForcedDecision =
                SimdSimtCandidateKind::MixedSIMDSIMT;
          else {
            module.emitError("invalid backend-forced SIMD/SIMT route");
            signalPassFailure();
            return;
          }
        }
        auto readFactors = [&](StringRef key, int64_t maximum,
                               std::vector<int64_t> &factors) {
          if (!object->get(key))
            return true;
          auto *values = object->getArray(key);
          if (!values)
            return false;
          factors.clear();
          for (const auto &value : *values) {
            auto factor = value.getAsInteger();
            if (!factor || *factor < 1 || *factor > maximum ||
                (*factor & (*factor - 1)) != 0)
              return false;
            factors.push_back(*factor);
          }
          return true;
        };
        if (!readFactors("whole_kernel_superblock_factors",
                         options.maximumWholeKernelSuperblockFactor,
                         options.wholeKernelSuperblockFactors) ||
            !readFactors("scope_superblock_factors",
                         options.maximumScopeSuperblockFactor,
                         options.scopeSuperblockFactors)) {
          module.emitError("invalid backend SuperBlock factor set");
          signalPassFailure();
          return;
        }
      }

    SimtAnchorPlan anchorPlan =
        buildMixedSimtAnchorPlan(module, options.compileOn91095);
    SimtAnchorPlan analysisAnchorPlan =
        buildMixedSimtAnchorPlan(analysisModule, options.compileOn91095);
    auto reportOr =
        analyzeSimdSimtCandidates(analysisModule, analysisAnchorPlan, options);
    if (!reportOr) {
      module.emitError("C++ SIMD/SIMT cost model failed: ")
          << llvm::toString(reportOr.takeError());
      signalPassFailure();
      return;
    }
    SimdSimtCostReport report = std::move(*reportOr);

    std::string recommended = stringifySimdSimtCandidate(report.decision).str();
    std::string effective = kBackendDefault.str();
    std::string selectionSource = "backend_default";
    std::string applicationReason;
    SmallVector<Operation *> mixedAnchors;
    SimtAnchorPlan selectedMixedAnchorPlan;
    int64_t selectedSuperblockFactor = 1;
    if (report.decision == SimdSimtCandidateKind::AllSIMD)
      selectedSuperblockFactor =
          report.stageModel.allSimd.routeSuperblockFactor;
    else if (report.decision == SimdSimtCandidateKind::AllSIMTOnly)
      selectedSuperblockFactor =
          report.stageModel.allSimt.routeSuperblockFactor;
    else
      selectedSuperblockFactor = report.stageModel.mixed.routeSuperblockFactor;

    bool actionSupported = true;
    bool hasExplicitScope = containsExplicitVectorScope(module);
    if (recommended == kMixedSimdSimt) {
      if (hasExplicitScope) {
        actionSupported = false;
        applicationReason = "explicit_scope_present";
      } else if (!anchorPlansHaveCompatibleIndices(analysisAnchorPlan,
                                                   anchorPlan)) {
        actionSupported = false;
        applicationReason = "analysis_materialization_anchor_mismatch";
      } else {
        selectedMixedAnchorPlan =
            buildSelectedMixedAnchorPlan(report.stageModel, anchorPlan);
        mixedAnchors = selectedMixedAnchorPlan.materializableRoots();
        if (mixedAnchors.empty()) {
          actionSupported = false;
          applicationReason = "no_materializable_mixed_anchor";
        }
      }
      if (selectedSuperblockFactor > 1 &&
          !options.scopeSuperblockMaterializable) {
        actionSupported = false;
        applicationReason = "scope_superblock_not_materializable";
      }
    } else if (recommended == kAllSimtOnly && hasExplicitScope) {
      // Preserve explicit local SIMD/SIMT/cube scope semantics instead of
      // replacing the whole kernel with a pure-SIMT route.
      actionSupported = false;
      applicationReason = "explicit_scope_present";
    }
    // Both whole-kernel and mixed-kernel V1 schedules launch numWarps * factor
    // logical warp groups.
    const int64_t selectedWarpCount =
        selectedSuperblockFactor * options.numWarps;
    if (selectedSuperblockFactor > 1 && selectedWarpCount > 64) {
      actionSupported = false;
      applicationReason = "superblock_warp_limit_exceeded";
    }
    if (recommended == kAllSimtOnly && selectedSuperblockFactor > 1 &&
        !report.features.autoBlockifyV1Applied &&
        !options.wholeKernelSuperblockMaterializable) {
      actionSupported = false;
      applicationReason = "superblock_requires_auto_blockify_v1";
    }

    if (autoMode && actionSupported) {
      effective = recommended;
      selectionSource = "cpp_cost_model";
      applicationReason = "minimum_cost_candidate";
    } else if (!autoMode) {
      applicationReason = "report_mode";
    } else if (applicationReason.empty()) {
      applicationReason = "candidate_not_materializable";
    }

    Builder builder(module.getContext());
    module->setAttr(kRecommendedExecutionAttr,
                    builder.getStringAttr(recommended));
    module->setAttr(kEffectiveExecutionAttr, builder.getStringAttr(effective));
    module->setAttr(kSelectionSourceAttr,
                    builder.getStringAttr(selectionSource));
    module->setAttr(kAllSimdScoreAttr,
                    builder.getF64FloatAttr(report.candidateCosts.allSimd));
    module->setAttr(kAllSimtScoreAttr,
                    builder.getF64FloatAttr(report.candidateCosts.allSimtOnly));
    module->setAttr(kMixedScoreAttr, builder.getF64FloatAttr(
                                         report.candidateCosts.mixedSimdSimt));
    module->setAttr(kSuperblockFactorAttr,
                    builder.getI64IntegerAttr(selectedSuperblockFactor));

    // Selector and Materializer consume the same immutable anchor plan in one
    // pass invocation.  No per-operation marker is persisted in TTIR.
    if (effective == kMixedSimdSimt &&
        failed(materializeSimtAnchorPlan(module, selectedMixedAnchorPlan,
                                         selectedSuperblockFactor))) {
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
    reportJSON["analysis_ir_source"] = analysisModule == module
                                           ? "route_neutral_ttir"
                                           : "post_auto_blockify_v1_ttir";
    if (auto capability =
            llvm::json::parse(routeTransformCapabilityJSON.getValue())) {
      reportJSON["route_transform_capability"] = std::move(*capability);
    } else {
      module.emitError("invalid route-transform-capability-json");
      signalPassFailure();
      return;
    }
    reportJSON["materialized_simt_anchor_count"] =
        static_cast<int64_t>(mixedAnchors.size());
    reportJSON["selected_superblock_factor"] = selectedSuperblockFactor;
    reportJSON["logical_program_count_hint"] = options.logicalProgramCountHint;
    reportJSON["source_logical_program_count_hint"] =
        options.sourceLogicalProgramCountHint;
    const StageRoutePlan *selectedRuntimePlan =
        recommended == kAllSimd       ? &report.stageModel.allSimd
        : recommended == kAllSimtOnly ? &report.stageModel.allSimt
                                      : &report.stageModel.mixed;
    reportJSON["runtime_physical_program_count"] =
        selectedRuntimePlan->runtimePhysicalProgramCount;
    reportJSON["runtime_scheduling_slot_count"] =
        selectedRuntimePlan->runtimeSchedulingSlotCount;
    reportJSON["logical_programs_per_scheduling_slot"] =
        selectedRuntimePlan->runtimeLogicalProgramsPerSchedulingSlot;
    reportJSON["full_group_count"] = selectedRuntimePlan->runtimeFullGroupCount;
    reportJSON["tail_count"] = selectedRuntimePlan->runtimeTailProgramCount;
    reportJSON["runtime_loop_iteration_count"] =
        selectedRuntimePlan->runtimeLoopIterationCount;
    const int64_t effectiveRuntimeFactor =
        recommended == kAllSimd ? 1
        : recommended == kMixedSimdSimt &&
                selectedRuntimePlan->runtimeFullGroupCount == 0
            ? 1
            : selectedSuperblockFactor;
    reportJSON["effective_runtime_factor"] = effectiveRuntimeFactor;
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
