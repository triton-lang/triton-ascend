// Qualified normal-Store first cell. Source-callsite candidate, isolated only.
#ifndef ASCENDMODEL_QUALIFIED_NORMAL_SEGMENTS_STORE_CANDIDATE_H
#define ASCENDMODEL_QUALIFIED_NORMAL_SEGMENTS_STORE_CANDIDATE_H
#include "AscendModel/Analysis/StageMemoryPatternAnalysis.h"
#include "AscendModel/RouteModel/NormalSegmentsStoreModel.h"
#include "AscendModel/RouteModel/SegmentsStoreOwnershipAdapter.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>
namespace mlir::ascend::qualified_normal_segments {
inline bool tryApplyStoreCandidate(ModuleOp module,
                                   const StagePartition &partition,
                                   const HardwareProfile &hardware,
                                   const StageCostTable &originalContext,
                                   const StagePartitionerOptions &analysis,
                                   const SimdSimtCostModelOptions &options,
                                   double actualProfileSystemCounterMHz,
                                   unsigned actualWarps,
                                   StageCostModelSummary &routes) {
  // Clock is the private CandidateProfile's real profile reference unit.
  // No runtime CardID/address/clock query and no complete-profile hash gate.
  if (hardware.target != "Ascend950PR/dav-c310" ||
      options.actualTarget != "Ascend950PR_9579" ||
      !std::isfinite(actualProfileSystemCounterMHz) ||
      actualProfileSystemCounterMHz != 1000 || actualWarps != 4 ||
      options.logicalProgramCountHint != 1 ||
      analysis.logicalProgramCountHint != 1 ||
      options.sourceLogicalProgramCountHint > 1 || !routes.allSimd.legal ||
      routes.allSimd.routeSuperblockFactor != 1 ||
      partition.modeledOperationCount != 32 ||
      (hardware.cvAivSubblocks != 1 && hardware.cvAivSubblocks != 2))
    return false;
  // CV capability does not imply that this pure vector route shares a program.
  // Preserve the caller profile, but require the actual selected
  // implementation.
  for (const auto &impl : routes.allSimd.implementations)
    if (!impl.isValid() || impl.mode != StageMode::SIMD || impl.localScope ||
        impl.superblockFactor != 1 || impl.aivSubblocks != 1)
      return false;
  for (const auto &stage : partition.stages)
    if (stage.features.hasDot)
      return false;
  Operation *target = nullptr;
  int stores = 0;
  for (const auto &stage : partition.stages)
    for (Operation *op : stage.operations)
      if (op->getName().getStringRef() == "tt.store") {
        target = op;
        ++stores;
      }
  if (stores != 1)
    return false;
  StageMemoryPatternAnalysis patterns(module);
  auto bundle = specialized_segments_ownership::prepareKnownNormalStoreBundle(
      target, partition, routes.allSimd.implementations, analysis, &patterns);
  if (!bundle) {
    llvm::consumeError(bundle.takeError());
    return false;
  }
  if (bundle->requiresWholeSimtPressureReanalysis ||
      bundle->uniformImplementation.mode != StageMode::SIMD ||
      bundle->uniformImplementation.localScope ||
      bundle->uniformImplementation.aivSubblocks != 1)
    return false;
  normal_segments_store::Geometry g{bundle->rows,
                                    bundle->columns,
                                    bundle->knownRows,
                                    bundle->knownColumns,
                                    1,
                                    bundle->rowPitchElements};
  auto quote = normal_segments_store::evaluate(
      bundle->payloadIsI32    ? normal_segments_store::Payload::I32
      : bundle->payloadIsF16  ? normal_segments_store::Payload::F16
      : bundle->payloadIsBF16 ? normal_segments_store::Payload::BF16
      : bundle->payloadIsU8   ? normal_segments_store::Payload::U8
      : bundle->payloadIsI64  ? normal_segments_store::Payload::I64
                              : normal_segments_store::Payload::F32,
      g, actualWarps, routes.allSimd.routeSuperblockFactor,
      options.logicalProgramCountHint, actualProfileSystemCounterMHz);
  if (!quote || !std::isfinite(quote->profileSystemCycles) ||
      quote->profileSystemCycles < 0)
    return false;
  const char *modelSHA = quote->modelSHA256;
  const char *devSHA = quote->independentDEVSHA256;
  auto rawAux = StageCostEvaluator().evaluate(bundle->auxiliaryPartition,
                                              hardware, false);
  if (!rawAux) {
    llvm::consumeError(rawAux.takeError());
    return false;
  }
  StageCostTable context = originalContext;
  context.stages = rawAux->stages;
  context.dependencies = rawAux->dependencies;
  context.modeledOperationCount = rawAux->modeledOperationCount;
  context.operationOwnershipComplete = rawAux->operationOwnershipComplete;
  auto aux = solveStageRoutes(context, hardware.scopeHandoff);
  if (!aux) {
    llvm::consumeError(aux.takeError());
    return false;
  }
  if (!aux->allSimd.legal)
    return false;
  const size_t count = partition.stages.size();
  const auto &old = routes.allSimd;
  if (old.implementations.size() != count ||
      old.scopeHandoffCycles.size() != count ||
      old.logicalStageCycles.size() != count ||
      (!old.runtimeStages.empty() && old.runtimeStages.size() != count) ||
      aux->allSimd.logicalStageCycles.size() !=
          bundle->auxiliaryPartition.stages.size())
    return false;
  llvm::DenseMap<Operation *, size_t> owner, ordinal;
  size_t targetOwner = count, nextOrdinal = 0;
  for (size_t i = 0; i < count; ++i)
    for (Operation *op : partition.stages[i].operations) {
      owner[op] = i;
      ordinal[op] = nextOrdinal++;
      if (op == target)
        targetOwner = i;
    }
  if (targetOwner == count)
    return false;
  StageRoutePlan candidate = old;
  candidate.logicalStageCycles.assign(count, 0);
  candidate.scopeHandoffCycles.assign(count, 0);
  llvm::json::Array graph, mapping, inputEdges;
  int setupOwners = 0;
  for (size_t j = 0; j < bundle->auxiliaryPartition.stages.size(); ++j) {
    const auto &stage = bundle->auxiliaryPartition.stages[j];
    std::optional<size_t> parent;
    for (Operation *op : stage.operations) {
      auto found = owner.find(op);
      if (found == owner.end())
        return false;
      if (parent && *parent != found->second)
        return false;
      parent = found->second;
      mapping.push_back(llvm::json::Object{
          {"original_operation_ordinal", int64_t(ordinal.lookup(op))},
          {"operation", op->getName().getStringRef()},
          {"owner", "auxiliary"},
          {"original_stage", partition.stages[*parent].id},
          {"route_node", stage.id}});
    }
    if (!parent)
      return false;
    setupOwners += stage.workload.paysKernelSetup;
    candidate.logicalStageCycles[*parent] += aux->allSimd.logicalStageCycles[j];
    graph.push_back(llvm::json::Object{
        {"id", stage.id},
        {"owned_operation_count", int64_t(stage.operations.size())},
        {"system_cycles", aux->allSimd.logicalStageCycles[j]},
        {"pays_kernel_setup", stage.workload.paysKernelSetup},
        {"original_stage_slot", int64_t(*parent)},
        {"workload_load_bytes", stage.workload.loadBytes},
        {"workload_store_bytes", stage.workload.storeBytes},
        {"IR_live_in_tensor_bytes", stage.liveInBytes},
        {"IR_live_out_tensor_bytes", stage.liveOutBytes},
        {"IR_scope_input_tensor_bytes", stage.scopeInputTensorBytes},
        {"IR_scope_output_tensor_bytes", stage.scopeOutputTensorBytes}});
  }
  for (Operation *op : bundle->coveredOperations)
    mapping.push_back(llvm::json::Object{
        {"original_operation_ordinal", int64_t(ordinal.lookup(op))},
        {"operation", op->getName().getStringRef()},
        {"owner", "target_bundle"},
        {"original_stage", partition.stages[owner.lookup(op)].id},
        {"route_node", "qualified_normal_Store_terminal_bundle"}});
  for (const auto &id : bundle->bundleInputAuxiliaryStageIds)
    inputEdges.push_back(llvm::json::Object{
        {"source", id}, {"target", "qualified_normal_Store_terminal_bundle"}});
  if (setupOwners != 1 || bundle->coveredOperations.size() != 17 ||
      bundle->auxiliaryPartition.modeledOperationCount != 15 ||
      mapping.size() != 32)
    return false;
  candidate.logicalStageCycles[targetOwner] += quote->profileSystemCycles;
  candidate.totalCycles = aux->allSimd.totalCycles + quote->profileSystemCycles;
  // These old complete-fragment overrides must not replace the new target+aux.
  // Other candidates and all original global Stage data stay untouched.
  for (const char *k :
       {"specialized_memory_complete_applied",
        "specialized_memory_elapsed_system_cycles",
        "specialized_memory_model_sha256", "native_indirect_chain_applied",
        "native_indirect_chain_elapsed_system_cycles",
        "native_indirect_chain_warp_count"})
    candidate.routeDiagnostics.values.erase(k);
  graph.push_back(
      llvm::json::Object{{"id", "qualified_normal_Store_terminal_bundle"},
                         {"owned_operation_count", 17},
                         {"system_cycles", quote->profileSystemCycles},
                         {"pays_kernel_setup", false},
                         {"original_stage_slot", int64_t(targetOwner)}});
  llvm::json::Object report{
      {"schema", "qualified_normal_segments_Store_per_route_graph_v1"},
      {"model_SHA256", modelSHA},
      {"independent_DEV_SHA256", devSHA},
      {"profile_target", hardware.target},
      {"payload", bundle->payloadIsI32    ? "i32"
                  : bundle->payloadIsF16  ? "f16"
                  : bundle->payloadIsBF16 ? "bf16"
                  : bundle->payloadIsU8   ? "u8"
                  : bundle->payloadIsI64  ? "i64"
                                          : "f32"},
      {"actual_profile_system_counter_MHz", actualProfileSystemCounterMHz},
      {"logical_program_count", 1},
      {"full_original_stage_slots", int64_t(count)},
      {"target_owner_slot", int64_t(targetOwner)},
      {"auxiliary_solver_body_system_cycles", aux->allSimd.totalCycles},
      {"target_system_cycles", quote->profileSystemCycles},
      {"new_body_system_cycles", candidate.totalCycles},
      {"external_kernel_boundary_added_here", false},
      {"external_argument_setup_added_here", false},
      {"auxiliary_kernel_setup_owners", setupOwners},
      {"target_kernel_setup_owners", 0},
      {"graph", std::move(graph)},
      {"bundle_input_edges", std::move(inputEdges)},
      {"Op_pointer_derived_mapping", std::move(mapping)},
      {"note",
       "Body from actual auxiliary solver plus target once; logical slots are "
       "audit projections and need not sum to dependency-scheduled total."}};
  candidate.diagnostic<bool>("qualified_normal_segments_bundle_applied") = true;
  candidate.diagnostic<double>(
      "qualified_normal_segments_target_system_cycles") =
      quote->profileSystemCycles;
  candidate.diagnostic<double>(
      "qualified_normal_segments_auxiliary_system_cycles") =
      aux->allSimd.totalCycles;
  candidate.diagnostic<std::string>(
      "qualified_normal_segments_per_route_graph_json") =
      llvm::formatv("{0}", llvm::json::Value(std::move(report))).str();
  routes.allSimd = std::move(candidate);
  return true;
}
} // namespace mlir::ascend::qualified_normal_segments
#endif
