//===- StageRouteCostModel.cpp - Logical-stage route solver ---------------===//

#include "AscendModel/RouteModel/StageRouteCostModel.h"
#include "AscendModel/Support/CostModelLogger.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static double mixedEquivalentStageCost(const LogicalStageCost &stage,
                                       const StageImplementationCost &selected,
                                       const StageTransitionCost &transition) {
  if (selected.implementation.mode != StageMode::SIMT ||
      !selected.implementation.localScope || !stage.localSimtMaterializable)
    return selected.totalCycles;
  // Materializer currently creates one scope per primitive anchor.  The
  // route DP otherwise observes only one Stage-mode change and would charge
  // one transition pair even when the generated TTIR contains several local
  // scopes.  Scope-local SuperBlock groups F independent logical programs in
  // one outlined SIMT VF, so those programs share each fixed SIMD/SIMT mode
  // switch.  The fixed transition cost is therefore amortized by F.  Tensor
  // handoff bytes are not divided: the materializer still transfers every
  // logical program's live-in/live-out values through UB.
  const int64_t scopeCount = std::max<int64_t>(1, stage.localSimtScopeCount);
  const double factor = static_cast<double>(
      std::max<int64_t>(1, selected.implementation.superblockFactor));
  const double fixedScopeTransitions =
      static_cast<double>(scopeCount) / factor *
      (transition.get(StageMode::SIMD, StageMode::SIMT) +
       transition.get(StageMode::SIMT, StageMode::SIMD));
  const double activeThreads =
      std::max(1.0, static_cast<double>(transition.simtWarpSize) *
                        std::clamp(stage.features.activeLaneRatio, 0.0, 1.0));
  const double simtLoadBytesPerCycle =
      transition.simtUbLoadBytesPerThreadPerCycle * activeThreads;
  const double simtStoreBytesPerCycle =
      transition.simtUbStoreBytesPerThreadPerCycle * activeThreads;
  const double inputBytes = static_cast<double>(stage.scopeInputTensorBytes);
  const double outputBytes = static_cast<double>(stage.scopeOutputTensorBytes);
  // SIMD producer register -> UB -> SIMT register.
  const double inputHandoffCycles =
      inputBytes / transition.simdUbStoreBytesPerCycle +
      inputBytes / simtLoadBytesPerCycle;
  // SIMT producer register -> UB -> SIMD register.
  const double outputHandoffCycles =
      outputBytes / simtStoreBytesPerCycle +
      outputBytes / transition.simdUbLoadBytesPerCycle;
  return selected.totalCycles + fixedScopeTransitions + inputHandoffCycles +
         outputHandoffCycles;
}

static double mixedExecutionMultiplicity(const LogicalStageCost &stage,
                                         const StageImplementationCost &cost,
                                         int64_t factor) {
  if (factor <= 1 || cost.implementation.mode != StageMode::SIMD ||
      !stage.features.replicatedByLocalSuperBlock)
    return 1.0;
  // MaterializeSIMTScopeSuperBlock groups F logical programs in one physical
  // V1-loop iteration.  The selected SIMT scope executes them as one F-way
  // VF, but every SIMD segment left in the logical body is cloned F times.
  return static_cast<double>(factor);
}

static double mixedBaseStageCost(const LogicalStageCost &stage,
                                 const StageImplementationCost &cost,
                                 int64_t factor) {
  return cost.totalCycles * mixedExecutionMultiplicity(stage, cost, factor);
}

/// AutoBlockify V1 is a route-conditional execution schedule.  The analysis
/// view contains its real dispatch/loop operations so pure-SIMT and Mixed can
/// pay them, but an all-SIMD executable restores the original logical grid.
/// Keep the Stage positions for report alignment and remove only their cost
/// from the all-SIMD candidate.
static void removeAutoBlockifyCostFromAllSIMD(StageRoutePlan &plan,
                                              const StageCostTable &costTable) {
  if (!plan.legal || plan.logicalStageCycles.size() != costTable.stages.size())
    return;
  for (size_t index = 0; index < costTable.stages.size(); ++index) {
    const llvm::StringRef model = costTable.stages[index].model;
    if (model != "auto_blockify_dispatch" && model != "auto_blockify_loop")
      continue;
    plan.totalCycles -= plan.logicalStageCycles[index];
    plan.logicalStageCycles[index] = 0.0;
    plan.entryTransitionCycles[index] = 0.0;
  }
  plan.totalCycles = std::max(0.0, plan.totalCycles);
}

/// One log block per candidate route: legality, factor, per-Stage mode and
/// cycles, and the summed total.
static void logRoutePlan(llvm::StringRef name, const StageRoutePlan &plan,
                         const StageCostTable &costTable) {
  costModelLog() << "route " << name << ": legal=" << plan.legal
                 << " factor=" << plan.routeSuperblockFactor
                 << " total=" << plan.totalCycles << " cycles";
  if (plan.runtimePhysicalProgramCount > 0)
    costModelLog() << " (physicalPrograms=" << plan.runtimePhysicalProgramCount
                   << " waves=" << plan.runtimeWaveCount << ")";
  costModelLog() << "\n";
  for (size_t index = 0; index < plan.implementations.size(); ++index) {
    const StageImplementation &implementation = plan.implementations[index];
    costModelLog() << "  stage '" << costTable.stages[index].id << "': "
                   << stringifyStageMode(implementation.mode)
                   << " factor=" << implementation.superblockFactor
                   << (implementation.localScope ? " local_scope" : "")
                   << " cycles=" << plan.logicalStageCycles[index]
                   << " transition=" << plan.entryTransitionCycles[index]
                   << "\n";
  }
}

} // namespace

llvm::StringRef mlir::ascend::stringifyStageMode(StageMode mode) {
  return mode == StageMode::SIMD ? "simd" : "simt";
}

static llvm::StringRef stringifyStageKernelRoute(StageKernelRouteKind kind) {
  switch (kind) {
  case StageKernelRouteKind::AllSIMD:
    return "all_simd";
  case StageKernelRouteKind::AllSIMT:
    return "all_simt_only";
  case StageKernelRouteKind::Mixed:
    return "mixed_simd_simt";
  }
  llvm_unreachable("unknown stage kernel route kind");
}

static llvm::StringRef stringifyStageSchedule(StageScheduleKind kind) {
  switch (kind) {
  case StageScheduleKind::StraightLine:
    return "straight_line";
  case StageScheduleKind::IndependentPipelined:
    return "independent_pipelined";
  case StageScheduleKind::LoopCarriedSerial:
    return "loop_carried_serial";
  case StageScheduleKind::PartiallyDependent:
    return "partially_dependent";
  }
  llvm_unreachable("unknown stage schedule kind");
}

bool StageImplementation::isValid() const {
  if (superblockFactor <= 0 || (superblockFactor & (superblockFactor - 1)) != 0)
    return false;
  if (mode == StageMode::SIMD)
    return superblockFactor == 1 && !localScope;
  return true;
}

llvm::json::Object StageImplementation::toJSON() const {
  return llvm::json::Object{
      {"mode", stringifyStageMode(mode)},
      {"superblock_factor", superblockFactor},
      {"materialization",
       localScope ? "local_simt_scope_with_kernel_v1" : "whole_kernel"}};
}

bool StageModelFeatures::isValid() const {
  return conditionalBranchCount >= 0 && divergentBranchCount >= 0 &&
         loopBackedgeCount >= 0 && synchronizationCount >= 0 &&
         parallelRecurrenceGroupCount > 0 && std::isfinite(activeLaneRatio) &&
         activeLaneRatio >= 0.0 && activeLaneRatio <= 1.0 &&
         (!hasLoopCarriedDataDependency || hasLoop);
}

bool StageModelFeatures::permitsSimdRoofline() const {
  return !hasLoopCarriedDataDependency;
}

bool StageWorkload::isFiniteAndNonNegative() const {
  const std::array<double, 10> values = {scalarOperations,
                                         loadBytes,
                                         storeBytes,
                                         loadWarpInstructions,
                                         storeWarpInstructions,
                                         predicateElements,
                                         shuffleLaneSteps,
                                         dotFlops,
                                         issueElements,
                                         estimatedSpillTransactions};
  if (!std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value) && value >= 0.0;
      }))
    return false;
  return llvm::all_of(operationElements, [](const auto &entry) {
    return std::isfinite(entry.second) && entry.second >= 0.0;
  });
}

llvm::json::Object StageWorkload::toJSON() const {
  llvm::json::Object result;
  llvm::json::Object operations;
  for (const auto &[name, elements] : operationElements)
    operations[name] = elements;
  result["operation_elements_per_iteration"] = std::move(operations);
  result["scalar_operations_per_iteration"] = scalarOperations;
  result["load_bytes_per_iteration"] = loadBytes;
  result["store_bytes_per_iteration"] = storeBytes;
  result["load_warp_instructions_per_iteration"] = loadWarpInstructions;
  result["store_warp_instructions_per_iteration"] = storeWarpInstructions;
  result["predicate_elements_per_iteration"] = predicateElements;
  result["shuffle_lane_steps_per_iteration"] = shuffleLaneSteps;
  result["dot_flops_per_iteration"] = dotFlops;
  result["issue_elements_per_iteration"] = issueElements;
  result["estimated_spill_transactions_per_iteration"] =
      estimatedSpillTransactions;
  result["pays_kernel_setup"] = paysKernelSetup;
  return result;
}

llvm::json::Object StageModelFeatures::toJSON() const {
  llvm::json::Object result;
  result["has_loop"] = hasLoop;
  result["has_loop_carried_data_dependency"] = hasLoopCarriedDataDependency;
  result["has_pointer_induction"] = hasPointerInduction;
  result["has_contiguous_memory"] = hasContiguousMemory;
  result["has_indirect_memory"] = hasIndirectMemory;
  result["has_reduction"] = hasReduction;
  result["has_prefix_scan"] = hasPrefixScan;
  result["has_dot"] = hasDot;
  result["has_conversion_pack"] = hasConversionPack;
  result["replicated_by_local_superblock"] = replicatedByLocalSuperBlock;
  result["conditional_branch_count"] = conditionalBranchCount;
  result["divergent_branch_count"] = divergentBranchCount;
  result["loop_backedge_count"] = loopBackedgeCount;
  result["synchronization_count"] = synchronizationCount;
  result["parallel_recurrence_group_count"] = parallelRecurrenceGroupCount;
  result["active_lane_ratio"] = activeLaneRatio;
  result["simd_roofline_permitted"] = permitsSimdRoofline();
  return result;
}

bool StageResourceCycles::isFiniteAndNonNegative() const {
  const std::array<double, 15> values = {
      setup,      scalar,          load,  store,       compute,
      predicate,  shuffle,         dot,   loopControl, branchControl,
      divergence, synchronization, spill, issue,       criticalPath};
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

llvm::json::Object StageResourceCycles::toJSON() const {
  llvm::json::Object result;
  result["setup"] = setup;
  result["scalar_per_iteration"] = scalar;
  result["load_per_iteration"] = load;
  result["store_per_iteration"] = store;
  result["compute_per_iteration"] = compute;
  result["predicate_per_iteration"] = predicate;
  result["shuffle_per_iteration"] = shuffle;
  result["dot_per_iteration"] = dot;
  result["loop_control_per_iteration"] = loopControl;
  result["branch_control_per_iteration"] = branchControl;
  result["divergence_per_iteration"] = divergence;
  result["synchronization_per_iteration"] = synchronization;
  result["spill_per_iteration"] = spill;
  result["issue_per_iteration"] = issue;
  result["critical_path_per_iteration"] = criticalPath;
  return result;
}

bool StageImplementationCost::isValid() const {
  return implementation.isValid() && std::isfinite(totalCycles) &&
         totalCycles >= 0.0 && resources.isFiniteAndNonNegative();
}

llvm::json::Object StageImplementationCost::toJSON() const {
  return llvm::json::Object{{"implementation", implementation.toJSON()},
                            {"total_system_cycles", totalCycles},
                            {"resource_system_cycles", resources.toJSON()}};
}

llvm::json::Object LogicalStageCost::toJSON() const {
  llvm::json::Object result;
  result["id"] = id;
  result["model"] = model;
  result["schedule_kind"] = stringifyStageSchedule(schedule);
  result["iteration_count"] = iterationCount;
  result["features"] = features.toJSON();
  result["workload"] = workload.toJSON();
  result["owned_operation_count"] = ownedOperationCount;
  llvm::json::Array locations;
  for (const std::string &location : sourceLocations)
    locations.push_back(location);
  result["source_locations"] = std::move(locations);
  result["live_in_count"] = liveInCount;
  result["live_out_count"] = liveOutCount;
  result["live_in_bytes"] = liveInBytes;
  result["live_out_bytes"] = liveOutBytes;
  result["local_simt_scope_count"] = localSimtScopeCount;
  result["scope_input_tensor_bytes"] = scopeInputTensorBytes;
  result["scope_output_tensor_bytes"] = scopeOutputTensorBytes;
  llvm::json::Array anchorIndices;
  for (unsigned index : simtAnchorIndices)
    anchorIndices.push_back(static_cast<int64_t>(index));
  result["simt_anchor_indices"] = std::move(anchorIndices);
  result["local_simt_materializable"] = localSimtMaterializable;
  result["local_superblock_materializable"] = localSuperblockMaterializable;
  llvm::json::Array legalFactors;
  for (int64_t factor : legalSimtFactors)
    legalFactors.push_back(factor);
  result["legal_simt_factors"] = std::move(legalFactors);
  llvm::json::Array localFactors;
  for (int64_t factor : localSimtFactors)
    localFactors.push_back(factor);
  result["local_simt_factors"] = std::move(localFactors);
  llvm::json::Array costs;
  for (const StageImplementationCost &implementation : implementations)
    costs.push_back(implementation.toJSON());
  result["implementations"] = std::move(costs);
  return result;
}

bool StageTransitionCost::isValid() const {
  return std::isfinite(simdToSimtCycles) && std::isfinite(simtToSimdCycles) &&
         simdToSimtCycles >= 0.0 && simtToSimdCycles >= 0.0 &&
         std::isfinite(simdUbLoadBytesPerCycle) &&
         simdUbLoadBytesPerCycle > 0.0 &&
         std::isfinite(simdUbStoreBytesPerCycle) &&
         simdUbStoreBytesPerCycle > 0.0 &&
         std::isfinite(simtUbLoadBytesPerThreadPerCycle) &&
         simtUbLoadBytesPerThreadPerCycle > 0.0 &&
         std::isfinite(simtUbStoreBytesPerThreadPerCycle) &&
         simtUbStoreBytesPerThreadPerCycle > 0.0 && simtWarpSize > 0;
}

double StageTransitionCost::get(StageMode from, StageMode to) const {
  if (from == to)
    return 0.0;
  return from == StageMode::SIMD ? simdToSimtCycles : simtToSimdCycles;
}

llvm::json::Object StageTransitionCost::toJSON() const {
  llvm::json::Object result;
  result["simd_to_simt_system_cycles"] = simdToSimtCycles;
  result["simt_to_simd_system_cycles"] = simtToSimdCycles;
  result["simd_ub_load_bytes_per_system_cycle"] = simdUbLoadBytesPerCycle;
  result["simd_ub_store_bytes_per_system_cycle"] = simdUbStoreBytesPerCycle;
  result["simt_ub_load_bytes_per_thread_per_system_cycle"] =
      simtUbLoadBytesPerThreadPerCycle;
  result["simt_ub_store_bytes_per_thread_per_system_cycle"] =
      simtUbStoreBytesPerThreadPerCycle;
  result["simt_warp_size"] = simtWarpSize;
  return result;
}

llvm::json::Object StageRoutePlan::toJSON() const {
  llvm::json::Object result;
  result["candidate"] = stringifyStageKernelRoute(candidate);
  result["legal"] = legal;
  result["total_system_cycles"] = totalCycles;
  result["route_superblock_factor"] = routeSuperblockFactor;
  result["runtime_physical_program_count"] = runtimePhysicalProgramCount;
  result["runtime_wave_count"] = runtimeWaveCount;
  llvm::json::Array stages;
  for (size_t i = 0; i < implementations.size(); ++i) {
    llvm::json::Object stage;
    stage["implementation"] = implementations[i].toJSON();
    stage["entry_transition_system_cycles"] = entryTransitionCycles[i];
    stage["logical_stage_system_cycles"] = logicalStageCycles[i];
    stages.push_back(std::move(stage));
  }
  result["stages"] = std::move(stages);
  return result;
}

llvm::json::Object StageCostModelSummary::toJSON() const {
  llvm::json::Object result;
  result["applied"] = applied;
  result["boundary_source"] = "operation_graph";
  result["operation_ownership_complete"] = operationOwnershipComplete;
  result["modeled_operation_count"] = modeledOperationCount;
  result["profile_version"] = profileVersion;
  llvm::json::Array stageArray;
  for (const LogicalStageCost &stage : stages)
    stageArray.push_back(stage.toJSON());
  result["logical_stages"] = std::move(stageArray);
  result["transition_cost"] = transition.toJSON();
  llvm::json::Object routes;
  routes["all_simd"] = allSimd.toJSON();
  routes["all_simt_only"] = allSimt.toJSON();
  routes["mixed_simd_simt"] = mixed.toJSON();
  result["routes"] = std::move(routes);
  return result;
}

llvm::Expected<StageCostModelSummary>
mlir::ascend::solveStageRoutes(const StageCostTable &costTable,
                               const StageTransitionCost &transition) {
  if (costTable.stages.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "stage route model requires at least one "
                                   "logical stage");
  if (!transition.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "stage transition costs must be finite and "
                                   "non-negative");

  auto findImplementation =
      [](const LogicalStageCost &stage, StageMode mode, int64_t factor,
         bool localScope) -> const StageImplementationCost * {
    for (const StageImplementationCost &cost : stage.implementations)
      if (cost.implementation.mode == mode &&
          cost.implementation.superblockFactor == factor &&
          cost.implementation.localScope == localScope)
        return &cost;
    return nullptr;
  };

  auto buildPlan = [&](StageKernelRouteKind kind,
                       int64_t factor) -> StageRoutePlan {
    StageRoutePlan plan;
    plan.candidate = kind;
    plan.routeSuperblockFactor = factor;
    struct MixedChoice {
      const StageImplementationCost *simd = nullptr;
      const StageImplementationCost *simt = nullptr;
      double simdCycles = std::numeric_limits<double>::infinity();
      double simtCycles = std::numeric_limits<double>::infinity();
    };
    std::vector<MixedChoice> mixedChoices;
    mixedChoices.reserve(costTable.stages.size());
    for (const LogicalStageCost &stage : costTable.stages) {
      for (const StageImplementationCost &cost : stage.implementations)
        if (!cost.isValid())
          return plan;

      const StageImplementationCost *selected = nullptr;
      double stageCycles = 0.0;
      if (kind == StageKernelRouteKind::AllSIMD) {
        selected = findImplementation(stage, StageMode::SIMD, 1, false);
      } else if (kind == StageKernelRouteKind::AllSIMT) {
        selected = findImplementation(stage, StageMode::SIMT, factor, false);
      } else {
        const StageImplementationCost *simd =
            findImplementation(stage, StageMode::SIMD, 1, false);
        const StageImplementationCost *simt =
            findImplementation(stage, StageMode::SIMT, factor, true);
        const double simdCycles = simd
                                      ? mixedBaseStageCost(stage, *simd, factor)
                                      : std::numeric_limits<double>::infinity();
        const double simtCycles =
            simt ? mixedEquivalentStageCost(stage, *simt, transition)
                 : std::numeric_limits<double>::infinity();
        selected = simtCycles < simdCycles ? simt : simd;
        stageCycles = std::min(simdCycles, simtCycles);
        mixedChoices.push_back({simd, simt, simdCycles, simtCycles});
      }
      if (!selected)
        return plan;
      if (kind != StageKernelRouteKind::Mixed)
        stageCycles = selected->totalCycles;

      const double transitionCycles =
          stageCycles - mixedBaseStageCost(stage, *selected, factor);
      plan.implementations.push_back(selected->implementation);
      plan.entryTransitionCycles.push_back(transitionCycles);
      plan.logicalStageCycles.push_back(stageCycles);
      plan.totalCycles += stageCycles;
    }
    if (kind == StageKernelRouteKind::Mixed) {
      if (factor > 1) {
        // The current local-SuperBlock ABI widens one AutoBlockify V1 loop
        // around exactly one selected local SIMT scope.  Selecting two
        // factor>1 scopes in the same loop would require preserving two
        // independent warp/task mappings across the intervening SIMD
        // segment, which the backend does not yet implement.  Enumerate the
        // one-SIMT-Stage routes instead of reporting an unmaterializable
        // independent per-Stage minimum.
        size_t bestSimtIndex = mixedChoices.size();
        double bestTotal = std::numeric_limits<double>::infinity();
        for (size_t simtIndex = 0; simtIndex < mixedChoices.size();
             ++simtIndex) {
          const MixedChoice &simtChoice = mixedChoices[simtIndex];
          if (!simtChoice.simt ||
              std::max<int64_t>(
                  1, costTable.stages[simtIndex].localSimtScopeCount) != 1)
            continue;
          double candidateTotal = simtChoice.simtCycles;
          bool candidateLegal = mixedChoices.size() > 1;
          for (size_t index = 0; index < mixedChoices.size(); ++index) {
            if (index == simtIndex)
              continue;
            if (!mixedChoices[index].simd) {
              candidateLegal = false;
              break;
            }
            candidateTotal += mixedChoices[index].simdCycles;
          }
          if (candidateLegal && candidateTotal < bestTotal) {
            bestTotal = candidateTotal;
            bestSimtIndex = simtIndex;
          }
        }
        if (bestSimtIndex == mixedChoices.size()) {
          StageRoutePlan invalid;
          invalid.candidate = kind;
          return invalid;
        }

        plan.implementations.clear();
        plan.entryTransitionCycles.clear();
        plan.logicalStageCycles.clear();
        plan.totalCycles = 0.0;
        for (size_t index = 0; index < mixedChoices.size(); ++index) {
          const MixedChoice &choice = mixedChoices[index];
          const StageImplementationCost *selected =
              index == bestSimtIndex ? choice.simt : choice.simd;
          const double selectedCycles =
              index == bestSimtIndex ? choice.simtCycles : choice.simdCycles;
          plan.implementations.push_back(selected->implementation);
          plan.entryTransitionCycles.push_back(
              selectedCycles -
              mixedBaseStageCost(costTable.stages[index], *selected, factor));
          plan.logicalStageCycles.push_back(selectedCycles);
          plan.totalCycles += selectedCycles;
        }
      }

      auto countMode = [&](StageMode mode) {
        return llvm::count_if(plan.implementations, [&](const auto &selected) {
          return selected.mode == mode;
        });
      };
      auto forceOneMode = [&](StageMode required) {
        size_t bestIndex = plan.implementations.size();
        double bestPenalty = std::numeric_limits<double>::infinity();
        for (size_t index = 0; index < mixedChoices.size(); ++index) {
          const MixedChoice &choice = mixedChoices[index];
          const StageImplementationCost *replacement =
              required == StageMode::SIMD ? choice.simd : choice.simt;
          if (!replacement)
            continue;
          const double replacementCycles = required == StageMode::SIMD
                                               ? choice.simdCycles
                                               : choice.simtCycles;
          const double penalty =
              replacementCycles - plan.logicalStageCycles[index];
          if (penalty < bestPenalty) {
            bestPenalty = penalty;
            bestIndex = index;
          }
        }
        if (bestIndex == plan.implementations.size())
          return false;
        const MixedChoice &choice = mixedChoices[bestIndex];
        const StageImplementationCost *replacement =
            required == StageMode::SIMD ? choice.simd : choice.simt;
        const double replacementCycles =
            required == StageMode::SIMD ? choice.simdCycles : choice.simtCycles;
        plan.totalCycles +=
            replacementCycles - plan.logicalStageCycles[bestIndex];
        plan.implementations[bestIndex] = replacement->implementation;
        plan.logicalStageCycles[bestIndex] = replacementCycles;
        plan.entryTransitionCycles[bestIndex] =
            replacementCycles - mixedBaseStageCost(costTable.stages[bestIndex],
                                                   *replacement, factor);
        return true;
      };

      // Mixed is a constrained candidate, not the unconstrained per-Stage
      // minimum.  If the latter collapses to all-SIMD or all-SIMT, switch the
      // Stage with the smallest incremental cost so the reported candidate
      // is the cheapest route that genuinely contains both modes.
      if ((countMode(StageMode::SIMT) == 0 && !forceOneMode(StageMode::SIMT)) ||
          (countMode(StageMode::SIMD) == 0 && !forceOneMode(StageMode::SIMD)) ||
          countMode(StageMode::SIMD) == 0 || countMode(StageMode::SIMT) == 0) {
        StageRoutePlan invalid;
        invalid.candidate = kind;
        return invalid;
      }
    }
    if (costTable.logicalProgramCountHint > 0) {
      plan.runtimePhysicalProgramCount =
          (costTable.logicalProgramCountHint + factor - 1) / factor;
      if (costTable.physicalCoreCountHint > 0)
        plan.runtimeWaveCount = (plan.runtimePhysicalProgramCount +
                                 costTable.physicalCoreCountHint - 1) /
                                costTable.physicalCoreCountHint;
      for (double &cycles : plan.logicalStageCycles)
        cycles *= static_cast<double>(plan.runtimeWaveCount);
      plan.totalCycles *= static_cast<double>(plan.runtimeWaveCount);
    }
    plan.legal = true;
    return plan;
  };

  auto bestFactoredPlan = [&](StageKernelRouteKind kind) {
    StageRoutePlan best;
    best.candidate = kind;
    for (int64_t factor : {1, 2, 4}) {
      StageRoutePlan candidate = buildPlan(kind, factor);
      if (candidate.legal &&
          (!best.legal || candidate.totalCycles < best.totalCycles))
        best = std::move(candidate);
    }
    return best;
  };

  StageCostModelSummary result;
  result.applied = true;
  result.operationOwnershipComplete = costTable.operationOwnershipComplete;
  result.modeledOperationCount = costTable.modeledOperationCount;
  result.profileVersion = costTable.profileVersion;
  result.stages = costTable.stages;
  result.transition = transition;
  result.allSimd = buildPlan(StageKernelRouteKind::AllSIMD, 1);
  result.allSimt = bestFactoredPlan(StageKernelRouteKind::AllSIMT);
  result.mixed = bestFactoredPlan(StageKernelRouteKind::Mixed);
  removeAutoBlockifyCostFromAllSIMD(result.allSimd, costTable);

  if (logEnabled()) {
    logRoutePlan("all_simd", result.allSimd, costTable);
    logRoutePlan("all_simt_only", result.allSimt, costTable);
    logRoutePlan("mixed_simd_simt", result.mixed, costTable);
  }

  return result;
}
