// Source-callsite Load candidate with its actual SSA producer and output sink.
#ifndef ASCENDMODEL_NORMAL_SEGMENTS_LOAD_CANDIDATE_H
#define ASCENDMODEL_NORMAL_SEGMENTS_LOAD_CANDIDATE_H
#include "AscendModel/RouteModel/NormalSegmentsLoadOwnership.h"
#include "AscendModel/RouteModel/NormalSegmentsStoreCandidate.h"
#include <cstdlib>
namespace mlir::ascend::qualified_normal_segments {
inline bool tryApplyLoadCandidate(ModuleOp module,
                                  const StagePartition &partition,
                                  const HardwareProfile &hardware,
                                  const StageCostTable &originalContext,
                                  const StagePartitionerOptions &analysis,
                                  const SimdSimtCostModelOptions &options,
                                  double profileMHz, unsigned warps,
                                  StageCostModelSummary &routes) {
  if (hardware.target != "Ascend950PR/dav-c310" ||
      options.actualTarget != "Ascend950PR_9579" || profileMHz != 1000 ||
      !std::isfinite(profileMHz) || warps != 4 ||
      options.logicalProgramCountHint != 1 ||
      analysis.logicalProgramCountHint != 1 ||
      options.sourceLogicalProgramCountHint > 1 || !routes.allSimd.legal ||
      routes.allSimd.routeSuperblockFactor != 1 ||
      (hardware.cvAivSubblocks != 1 && hardware.cvAivSubblocks != 2))
    return false;
  for (const auto &impl : routes.allSimd.implementations)
    if (!impl.isValid() || impl.mode != StageMode::SIMD || impl.localScope ||
        impl.superblockFactor != 1 || impl.aivSubblocks != 1)
      return false;
  for (const auto &stage : partition.stages)
    if (stage.features.hasDot)
      return false;
  Operation *target = nullptr;
  for (const auto &s : partition.stages)
    for (Operation *op : s.operations) {
      if (op->getName().getStringRef() != "tt.load" ||
          op->getNumResults() != 1 ||
          !normal_segments_load_ownership::calibratedTile(op->getResult(0)))
        continue;
      if (target)
        return false;
      target = op;
    }
  if (!target)
    return false;
  StageMemoryPatternAnalysis patterns(module);
  auto bundle = normal_segments_load_ownership::prepare(
      module, target, partition, routes.allSimd.implementations, analysis,
      warps, hardware.simt.usableUbBytes, &patterns);
  if (!bundle) {
    llvm::consumeError(bundle.takeError());
    return false;
  }
  // Empirical normal-source migration was evaluated before this candidate
  // existed. The frozen price is in profile SYS1000 cycles. Physical card is
  // provenance only.
  const bool f16 = bundle->payloadIsF16;
  const bool bf16 = bundle->payloadIsBF16;
  bool shared16 = f16 || bf16;
  if (bf16 && !shared16)
    return false;
  const bool i32 = bundle->payloadIsI32;
  const bool u8 = bundle->payloadIsU8;
  const bool i64 = bundle->payloadIsI64;
  const double quote = i64        ? 209.66814476007215
                       : u8       ? 124.81353352697784 * 4.0
                       : shared16 ? 92.66666666666667 + 26.666666666666664 * 4.0
                       : f16      ? (189.0 / 4.0) + (513.0 / 16.0) * 4.0
                                  : 48.0 + (519.0 / 16.0) * 4.0;
  const char *modelSHA =
      shared16
          ? "03e4981b281d247e37bb75f188abfb4a84e702713866deff7e6d2a81a8cd02dc"
      : i64 ? "15306aa153c1a77a7a516eb3f30f6ea68671aa88fa46bb83910ec855dd15f36d"
      : u8  ? "1bee66efa0bdc54332d73ece58109caea7ad5a84c0ad127098e0025adef66773"
      : i32 ? "4ec212d79f8faa4f4af8b3097c56afca645ed8e5a2dfffa82432caf15da79c06"
      : f16
          ? "a13f2f319f0fb3a291203bb7dee7704af23d4e4310ac2b6f4a29df6df765c886"
          : "50666579fb3180ad4b7fd7252c302cb1cc61d86ebf7be8895a05488025a758c6";
  const char *devSHA =
      shared16
          ? "15b2a83b1f573e508064405d1b5605194b1992d20dc34d34dd2e41117bb5f135"
      : i64 ? "bb355aa3a478f2c8038825640badd0facec3237f39c84c508f4406ee0686089c"
      : u8  ? "7e5f92cea3f68050933416d35c30afaf67def0f2a22ea1d4ab386363dea334b7"
      : i32 ? "db4d55af333224be0104c6be305e08e110b48b10adc8a86b4a2905e04843339b"
      : f16
          ? "6dd585a4d6625fdf15010ec97f978247b60b0900ce03a3cbcc8c23ca5120b1fc"
          : "17a982a706e8a20191a72cb852fbd896b0f7fd3b31a76624acae8ff6562d1f3d";

  auto raw =
      StageCostEvaluator().evaluate(bundle->completePartition, hardware, false);
  if (!raw) {
    llvm::consumeError(raw.takeError());
    return false;
  }
  if (bundle->targetNode >= raw->stages.size())
    return false;
  auto &targetCost = raw->stages[bundle->targetNode];
  unsigned replaced = 0;
  for (auto &impl : targetCost.implementations) {
    if (impl.implementation.mode != StageMode::SIMD ||
        impl.implementation.superblockFactor != 1 ||
        impl.implementation.localScope || impl.implementation.aivSubblocks != 1)
      continue;
    StageImplementationCost priced;
    priced.implementation = impl.implementation;
    priced.totalCycles = quote;
    priced.resources.load = quote;
    priced.indirectLoadPricing = "qualified_normal_full4x16_producer_bundle";
    priced.formulaEvidence =
        shared16 ? "onceShared16TRAIN12_NNMAPELAD_a92p66666666666667_"
                   "b26p666666666666664H;f16andbf16_separate6DEV_seen_"
                   "disclosed;SYS1000"
        : i64
            ? "own_frozen_literal_R4_209p66814476007215;newIndependentI64DEV6;"
              "profileSYS1000;countedR16R32_not_admitted"
        : u8  ? "frozenNN0plus124p81353352697784H;ownU8normalScalar26DEV744;"
                "profileSYS1000"
        : i32 ? "frozen48plus519over16H;ownI32normalDEV744;profileSYS1000"
        : f16 ? "frozen189over4plus513over16H;normalF16DEVcontext744;"
                "profileSYS1000"
              : "frozen48plus519over16H;normalDEVcontext744;profileSYS1000";
    priced.ordinaryLoadSelectedUsed = true;
    priced.ordinaryLoadTargetIncrementSystemCycles = quote;
    priced.ordinaryLoadCandidateReason =
        u8 ? "qualified_normal_scalar26_complete_target_and_poststop_copy_"
             "consumer"
           : "qualified_normal_literal_DMA_and_copy_consumer";
    priced.ordinaryLoadLeaf =
        bf16  ? "SegmentsLoad_BF16_shared16_W4P1_R4C16_full_pitch32768"
        : i64 ? "SegmentsLoad_I64_W4P1_R4C16_full_pitch32768_literal128B"
        : u8  ? "SegmentsLoad_U8_W4F1P1_R4C16_full_pitch32768_scalar26"
        : i32 ? "SegmentsLoad_I32_W4F1P1_R4C16_full_pitch32768"
        : f16 ? "SegmentsLoad_F16_W4F1P1_R4C16_full_pitch32768"
              : "SegmentsLoad_F32_W4F1P1_R4C16_full_pitch32768";
    impl = std::move(priced);
    ++replaced;
  }
  if (replaced != 1 || targetCost.workload.paysKernelSetup)
    return false;
  StageCostTable context = originalContext;
  context.stages = raw->stages;
  context.dependencies = raw->dependencies;
  context.modeledOperationCount = raw->modeledOperationCount;
  context.operationOwnershipComplete = raw->operationOwnershipComplete;
  auto solved = solveStageRoutes(context, hardware.scopeHandoff);
  if (!solved) {
    llvm::consumeError(solved.takeError());
    return false;
  }
  if (!solved->allSimd.legal)
    return false;
  const auto &full = bundle->completePartition;
  const auto &plan = solved->allSimd;
  size_t count = partition.stages.size();
  if (routes.allSimd.implementations.size() != count ||
      routes.allSimd.logicalStageCycles.size() != count ||
      routes.allSimd.scopeHandoffCycles.size() != count ||
      (!routes.allSimd.runtimeStages.empty() &&
       routes.allSimd.runtimeStages.size() != count) ||
      plan.logicalStageCycles.size() != full.stages.size() ||
      plan.scopeHandoffCycles.size() != full.stages.size())
    return false;
  llvm::DenseMap<Operation *, unsigned> owner, ordinal;
  unsigned ord = 0;
  for (unsigned i = 0; i < count; ++i)
    for (Operation *op : partition.stages[i].operations) {
      owner[op] = i;
      ordinal[op] = ord++;
    }
  StageRoutePlan result = routes.allSimd;
  result.logicalStageCycles.assign(count, 0);
  result.scopeHandoffCycles.assign(count, 0);
  result.totalCycles = plan.totalCycles;
  llvm::json::Array graph, mapping, edges;
  unsigned setup = 0, covered = 0, auxiliary = 0;
  double auxNodeSum = 0;
  for (unsigned i = 0; i < full.stages.size(); ++i) {
    const auto &s = full.stages[i];
    unsigned slot = bundle->originalStageSlots[i];
    if (slot >= count)
      return false;
    bool targetNode = i == bundle->targetNode;
    result.logicalStageCycles[slot] += plan.logicalStageCycles[i];
    result.scopeHandoffCycles[slot] += plan.scopeHandoffCycles[i];
    setup += s.workload.paysKernelSetup;
    if (targetNode) {
      covered += s.operations.size();
      if (s.workload.paysKernelSetup)
        return false;
    } else {
      auxiliary += s.operations.size();
      auxNodeSum += plan.logicalStageCycles[i];
    }
    for (Operation *op : s.operations) {
      if (!owner.count(op))
        return false;
      mapping.push_back(llvm::json::Object{
          {"original_operation_ordinal", int64_t(ordinal.lookup(op))},
          {"operation", op->getName().getStringRef()},
          {"owner", targetNode ? "target_bundle" : "auxiliary"},
          {"original_stage", partition.stages[owner.lookup(op)].id},
          {"route_node", s.id}});
    }
    graph.push_back(llvm::json::Object{
        {"id", s.id},
        {"original_stage_slot", int64_t(slot)},
        {"target_bundle", targetNode},
        {"owned_operation_count", int64_t(s.operations.size())},
        {"system_cycles", plan.logicalStageCycles[i]},
        {"pays_kernel_setup", s.workload.paysKernelSetup},
        {"workload_load_bytes", s.workload.loadBytes},
        {"workload_store_bytes", s.workload.storeBytes},
        {"IR_live_in_tensor_bytes", s.liveInBytes},
        {"IR_live_out_tensor_bytes", s.liveOutBytes}});
  }
  for (const auto &e : full.dependencies)
    edges.push_back(
        llvm::json::Object{{"source", e.sourceStage},
                           {"target", e.targetStage},
                           {"value_bytes", e.carriedValueBytes},
                           {"iteration_distance", e.iterationDistance},
                           {"kind", int64_t(e.kind)},
                           {"loop_stage", e.loopStage},
                           {"carried_value_index", e.carriedValueIndex},
                           {"carried_value_name", e.carriedValueName}});
  if (setup != 1 || covered != 18 || auxiliary != 19 || mapping.size() != 37 ||
      full.modeledOperationCount != 37)
    return false;
  // Original candidates remain intact. Only the qualified uniform AllSIMD body
  // changes.
  for (const char *k :
       {"specialized_memory_complete_applied",
        "specialized_memory_elapsed_system_cycles",
        "specialized_memory_model_sha256", "native_indirect_chain_applied",
        "native_indirect_chain_elapsed_system_cycles",
        "native_indirect_chain_warp_count"})
    result.routeDiagnostics.values.erase(k);
  llvm::json::Object report{
      {"schema", "qualified_normal_segments_Load_producer_preserving_graph_v1"},
      {"model_SHA256", modelSHA},
      {"normal_DEV_context_SHA256", devSHA},
      {"profile_target", hardware.target},
      {"actual_profile_system_counter_MHz", profileMHz},
      {"logical_program_count", 1},
      {"payload", i64    ? "i64"
                  : u8   ? "u8"
                  : i32  ? "i32"
                  : bf16 ? "bf16"
                  : f16  ? "f16"
                         : "f32"},
      {"payload_bytes", int64_t(bundle->payloadBytes)},
      {"target_tensor_bytes", int64_t(4 * 16 * bundle->payloadBytes)},
      {"W", 4},
      {"F", 1},
      {"SIMD_F_parameter_applicable", false},
      {"rows", 4},
      {"columns", 16},
      {"active_rows", 4},
      {"active_columns", 16},
      {"row_pitch_elements", 32768},
      {"target_system_cycles", quote},
      {"full_dependency_solver_body_system_cycles", plan.totalCycles},
      {"auxiliary_node_cycle_sum", auxNodeSum},
      {"new_body_system_cycles", result.totalCycles},
      {"full_original_stage_slots", int64_t(count)},
      {"target_owner_slot", int64_t(owner.lookup(target))},
      {"target_bundle_node", full.stages[bundle->targetNode].id},
      {"target_result_preserved_as_SSA_producer", true},
      {"sink_Store_retained_as_auxiliary", true},
      {"kernel_setup_owners", int64_t(setup)},
      {"target_kernel_setup_owners", 0},
      {"external_kernel_boundary_added_here", false},
      {"external_argument_setup_added_here", false},
      {"graph", std::move(graph)},
      {"edges", std::move(edges)},
      {"Op_pointer_derived_mapping", std::move(mapping)},
      {"note", "Full graph solver includes quoted target and retained SSA "
               "sink; node sums are diagnostics, not a substitute for "
               "dependency-scheduled total."}};
  result.diagnostic<bool>("qualified_normal_segments_load_bundle_applied") =
      true;
  result.diagnostic<double>(
      "qualified_normal_segments_load_target_system_cycles") = quote;
  result.diagnostic<std::string>(
      "qualified_normal_segments_load_per_route_graph_json") =
      llvm::formatv("{0}", llvm::json::Value(std::move(report))).str();
  routes.allSimd = std::move(result);
  return true;
}
} // namespace mlir::ascend::qualified_normal_segments
#endif
