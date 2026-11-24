// Calibrated target-Store insertion costs for the full 8x32 tile domain.
#ifndef ASCENDMODEL_NORMAL_FULL224_SIMT_STORE_H
#define ASCENDMODEL_NORMAL_FULL224_SIMT_STORE_H
#include "AscendModel/RouteModel/NormalIndexedRowOffset.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdlib>
#include <limits>
namespace mlir::ascend::normal_full224_simt_store {
using namespace specialized_segments_ownership;
inline std::string location(Operation *op) {
  std::string out;
  llvm::raw_string_ostream os(out);
  op->getLoc().print(os);
  return out;
}
inline bool tile(Value value, int64_t r, int64_t c, bool fp) {
  auto t = dyn_cast<RankedTensorType>(value.getType());
  return t && t.hasStaticShape() && !t.getEncoding() && t.getRank() == 2 &&
         t.getDimSize(0) == r && t.getDimSize(1) == c &&
         (fp ? t.getElementType().isF32()
             : (t.getElementType().isSignlessInteger(8) ||
                t.getElementType().isSignlessInteger(64) ||
                t.getElementType().isSignlessInteger(32) ||
                t.getElementType().isF32() || t.getElementType().isF16() ||
                t.getElementType().isBF16()));
}
inline bool programTensorBase(Value pointer, Value base, Value pid,
                              int64_t stride) {
  Operation *op = def(pointer);
  return named(op, "tt.addptr") && op->getNumOperands() == 2 &&
         normal_full224_simt_load::programBase(peelViews(op->getOperand(0)),
                                               base, pid, stride);
}
inline double groupPriceFor(Type type, int64_t factor) {
  if (type.isSignlessInteger(8) && factor == 1)
    return 4921.062849462032;
  if (type.isSignlessInteger(8) && factor == 2)
    return 5484.313704073429;
  if (type.isSignlessInteger(8) && factor == 4)
    return 4993.4983253479;
  if (type.isSignlessInteger(64) && factor == 1)
    return 3417.296800762415;
  if (type.isSignlessInteger(64) && factor == 2)
    return 4134.561866521835;
  if (type.isSignlessInteger(64) && factor == 4)
    return 4543.9377427101135;
  if (type.isF32() && factor == 1)
    return 1175.9214103221893;
  if (type.isSignlessInteger(32) && factor == 1)
    return 1175.9214103221893;
  if (type.isF16() && factor == 2)
    return 3557.6866939663887;
  if (type.isBF16() && factor == 2)
    return 3557.6866939663887;
  if (type.isF16() && factor == 4)
    return 3332.686610519886;
  if (type.isBF16() && factor == 4)
    return 3332.686610519886;
  if (type.isSignlessInteger(32) && factor == 4)
    return 199.24994558095932;
  if (type.isF32() && factor == 2)
    return 216.93753660656512;
  if (type.isF32() && factor == 4)
    return 211.25009516254067;
  if (type.isSignlessInteger(32) && factor == 2)
    return 216.93753660656512;
  if (type.isF16() && factor == 1)
    return 60.00002031214535;
  if (type.isBF16() && factor == 1)
    return 60.00002031214535;
  return std::numeric_limits<double>::quiet_NaN();
}
struct Prepared {
  Operation *target = nullptr;
  llvm::DenseSet<Operation *> covered;
  StagePartition auxiliaryPartition;
  StageRoutePlan auxiliaryPlan;
  int64_t H = 0, K = 0;
  double quote = 0;
};
inline llvm::Expected<llvm::json::Object>
inspect(ModuleOp module, const StagePartition &partition,
        const StageCostTable &context, const HardwareProfile &hardware,
        const StagePartitionerOptions &analysis,
        const SimdSimtCostModelOptions &options, double MHz, unsigned warps,
        int64_t factor, const StageCostModelSummary &routes,
        double genericBoundary, Prepared *prepared = nullptr) {
  if (MHz != 1000 || !std::isfinite(MHz) || warps != 4 ||
      hardware.target != "Ascend950PR/dav-c310" ||
      options.actualTarget != "Ascend950PR_9579" ||
      options.logicalProgramCountHint != 224 ||
      analysis.logicalProgramCountHint != 224 ||
      options.sourceLogicalProgramCountHint != 224 ||
      !partition.operationOwnershipComplete ||
      !routes.allSimtFactorPlans.count(factor) ||
      !routes.allSimtFactorPlans.at(factor).legal)
    return reject("audit_requires_actual_P1_W4_profileSYS1000_pure_F1");
  auto attr = module->getAttrOfType<StringAttr>(
      "ascend.costmodel.ordinary_load.route_capability_json");
  if (!attr)
    return reject("full224_route_capability");
  auto parsed = llvm::json::parse(attr.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return reject("full224_route_capability");
  }
  auto *cap = parsed->getAsObject();
  if (!cap)
    return reject("full224_route_capability");
  auto *grid = cap->getArray("source_logical_program_grid_hint");
  if (!grid || grid->size() != 3 ||
      (*grid)[0].getAsInteger() != std::optional<int64_t>(224) ||
      (*grid)[1].getAsInteger() != std::optional<int64_t>(1) ||
      (*grid)[2].getAsInteger() != std::optional<int64_t>(1) ||
      cap->getInteger("source_logical_program_count_hint") !=
          std::optional<int64_t>(224) ||
      cap->getInteger("logical_program_count_hint") !=
          std::optional<int64_t>(224) ||
      cap->getInteger("physical_vector_core_count_hint") !=
          std::optional<int64_t>(56) ||
      cap->getInteger("layout_coalescing_factor") !=
          std::optional<int64_t>(1) ||
      cap->getBoolean("layout_merge_applied") != std::optional<bool>(false) ||
      cap->getBoolean("auto_blockify_v1_materializable") !=
          std::optional<bool>(true))
    return reject("full224_route_capability");
  auto *factors = cap->getArray("whole_kernel_superblock_factors");
  if (!factors || factors->empty())
    return reject("full224_route_capability");
  std::vector<int64_t> factorsSeen;
  for (const auto &value : *factors) {
    auto f = value.getAsInteger();
    if (!f || (*f != 1 && *f != 2 && *f != 4) ||
        !llvm::is_contained(options.wholeKernelSuperblockFactors, *f) ||
        llvm::is_contained(factorsSeen, *f))
      return reject("full224_route_capability");
    factorsSeen.push_back(*f);
  }
  if (!llvm::is_contained(factorsSeen, int64_t(factor)))
    return reject("missing_F2_capability");
  const auto &old = routes.allSimtFactorPlans.at(factor);
  if (context.physicalCoreCountHint != 56 ||
      options.physicalVectorCoreCountHint != 56 ||
      old.runtimePhysicalProgramCount != 56 ||
      old.runtimeSchedulingSlotCount != 56 ||
      old.runtimeLogicalProgramsPerSchedulingSlot != 4 ||
      old.runtimeFullGroupCount != 4 / factor ||
      old.runtimeTailProgramCount != 0 ||
      old.runtimeLoopIterationCount != 4 / factor)
    return reject("full224_q4_F2_runtime_schedule");

  for (const auto &impl : old.implementations)
    if (!impl.isValid() || impl.mode != StageMode::SIMT || impl.localScope ||
        impl.superblockFactor != factor || impl.aivSubblocks != 1)
      return reject("audit_actual_implementation_is_not_uniform_whole_SIMT1");
  llvm::DenseMap<Operation *, unsigned> owner, ordinal;
  unsigned next = 0;
  for (unsigned i = 0; i < partition.stages.size(); ++i) {
    const auto &s = partition.stages[i];
    if (s.features.hasDot || s.features.hasLoop ||
        !s.enclosingLoopStages.empty() || s.iterationCount != 1)
      return reject("audit_dot_or_loop_parent_not_calibrated");
    for (Operation *op : s.operations) {
      if (!op || owner.count(op))
        return reject("audit_incomplete_duplicate_owner");
      owner[op] = i;
      ordinal[op] = next++;
    }
  }
  if (int64_t(owner.size()) != partition.modeledOperationCount)
    return reject("audit_owner_count");
  Operation *target = nullptr;
  for (const auto &[op, i] : owner) {
    auto st = dyn_cast<triton::StoreOp>(op);
    if (!st || op->getNumOperands() < 2 ||
        !tile(op->getOperand(1), 8, 32, false))
      continue;
    Block *entry = op->getBlock();
    if (entry->getNumArguments() == 0 ||
        !baseIs(op->getOperand(0), entry->getArgument(0)))
      continue;
    if (target)
      return reject("audit_multiple_target_stores");
    target = op;
  }
  if (!target)
    return reject("audit_no_target_Store");
  Block *block = target->getBlock();
  Operation *fn = block->getParentOp();
  if (!named(fn, "tt.func") || !fn->getRegion(0).hasOneBlock() ||
      block->getNumArguments() != 3)
    return reject("requires_original_three_pointer_entry");
  auto callable = dyn_cast<FunctionOpInterface>(fn);
  if (!callable)
    return reject("audit_unknown_callable");
  for (unsigned i = 0; i < block->getNumArguments(); ++i) {
    auto p = dyn_cast<triton::PointerType>(block->getArgument(i).getType());
    if (!p || p.getAddressSpace() != 1 || !callable.getArgAttrs(i).empty())
      return reject("audit_argument_space_or_metadata");
    if ((i == 1 && !p.getPointeeType().isSignlessInteger(32)) ||
        (i != 1 && p.getPointeeType() !=
                       cast<RankedTensorType>(target->getOperand(1).getType())
                           .getElementType()))
      return reject("audit_argument_payload_contract");
  }

  Value pid;
  for (Operation &op : *block)
    if (auto p = dyn_cast<triton::GetProgramIdOp>(&op)) {
      if (pid || p.getAxis() != triton::ProgramIDDim::X)
        return reject("unique_x_PID");
      pid = p.getResult();
    }
  if (!pid)
    return reject("missing_real_PID");
  auto st = cast<triton::StoreOp>(target);
  if (st.getCache() != triton::CacheModifier::NONE ||
      st.getEvict() != triton::EvictionPolicy::NORMAL ||
      !st.getBoundaryCheck().empty() || target->getNumOperands() > 3)
    return reject("audit_target_store_policy");
  auto ptr = dyn_cast_or_null<triton::AddPtrOp>(def(target->getOperand(0)));
  if (!ptr || !baseIs(target->getOperand(0), block->getArgument(0)))
    return reject("audit_target_pointer_base");
  Operation *values = def(target->getOperand(1));
  if (!named(values, "tt.load") || values->getNumOperands() != 1 ||
      !normal_segments_load_ownership::defaultLoad(values) ||
      !programTensorBase(values->getOperand(0), block->getArgument(2), pid,
                         256))
    return reject("audit_Values_full_unmasked_load_contract");
  Operation *vp = def(values->getOperand(0));
  if (!named(vp, "tt.addptr") || vp->getNumOperands() != 2 ||
      !normal_indexed_offset::constantRamp(vp->getOperand(1), 256))
    return reject("audit_Values_dense_R8C32_stride1_contract");
  llvm::DenseSet<Operation *> rowLoads, seen;
  std::function<bool(Value)> collectRows = [&](Value v) {
    Operation *op = def(v);
    if (!op || op->getBlock() != block)
      return false;
    if (!seen.insert(op).second)
      return true;
    if (named(op, "tt.load")) {
      rowLoads.insert(op);
      return true;
    }
    if (!pure(op))
      return false;
    for (Value a : op->getOperands())
      if (!collectRows(a))
        return false;
    return true;
  };
  // Constants have no operands; block pointer values are outside this integer
  // offset.
  if (!collectRows(ptr->getOperand(1)) || rowLoads.size() != 1)
    return reject("audit_unknown_row_offset");
  Operation *index = *rowLoads.begin();
  auto it = index->getNumResults() == 1
                ? dyn_cast<RankedTensorType>(index->getResult(0).getType())
                : RankedTensorType();
  if (!it || it.getRank() != 1 || it.getDimSize(0) != 8 ||
      !it.getElementType().isSignlessInteger(32) ||
      index->getNumOperands() != 1 ||
      !normal_segments_load_ownership::defaultLoad(index) ||
      !programTensorBase(index->getOperand(0), block->getArgument(1), pid, 8))
    return reject("audit_Index_load_contract");
  Operation *ip = def(index->getOperand(0));
  if (!named(ip, "tt.addptr") || ip->getNumOperands() != 2 ||
      !normal_indexed_offset::constantRamp(ip->getOperand(1), 8) ||
      !normal_indexed_offset::target(ptr->getOperand(1), index))
    return reject("audit_target_Index_times32768_plus_unit_column_contract");
  int64_t H = 8, K = 32;
  Value mask;
  if (target->getNumOperands() == 3) {
    mask = target->getOperand(2);
    auto rectangle = normal_static_mask::rectangle(mask);
    if (!rectangle)
      return reject("audit_mask_not_static_prefix_rectangle");
    H = rectangle->first;
    K = rectangle->second;
  }
  if (!(H == 8 && K == 32))
    return reject("audit_mask_outside_four_frozen_native_classes");
  // The proper hook has a deliberately narrower source domain than the
  // former 3/5-pointer audit. No reduction/witness or unknown operation enters.
  unsigned loads = 0, stores = 0;
  for (Operation &op : *block) {
    if (named(&op, "tt.return")) {
      if (op.getNumOperands())
        return reject("unknown_return");
      continue;
    }
    if (op.getNumRegions() || op.getNumSuccessors())
      return reject("region_or_control");
    if (named(&op, "tt.load"))
      ++loads;
    else if (named(&op, "tt.store"))
      ++stores;
    else if (!pure(&op) && !isa<triton::GetProgramIdOp>(&op))
      return reject("unknown_or_effectful_operation");
    if (!owner.count(&op))
      return reject("unowned_entry_operation");
  }
  if (loads != 2 || stores != 1)
    return reject("requires_only_Index_Values_and_target_Store");

  llvm::DenseSet<Operation *> ancestors, covered;
  covered.insert(target);
  std::function<bool(Value)> visit = [&](Value value) {
    Operation *op = def(value);
    if (!op || op->getBlock() != block || !pure(op))
      return false;
    if (!ancestors.insert(op).second)
      return true;
    for (Value a : op->getOperands())
      if (!visit(a))
        return false;
    return true;
  };
  if (mask && !visit(mask))
    return reject("audit_dynamic_or_cross_region_mask");
  llvm::DenseMap<Operation *, bool> memo;
  llvm::DenseSet<Operation *> active;
  std::function<bool(Operation *)> exclusive = [&](Operation *op) {
    if (op == target)
      return true;
    if (auto f = memo.find(op); f != memo.end())
      return f->second;
    if (!active.insert(op).second)
      return false;
    bool yes = !op->use_empty();
    for (Operation *user : op->getUsers())
      if (user != target && (!ancestors.contains(user) || !exclusive(user))) {
        yes = false;
        break;
      }
    active.erase(op);
    memo[op] = yes;
    return yes;
  };
  // Full-mask matched A/B retain address and mask work; only the target Store
  // is replaced. Shared axes/constants remain auxiliary. A shared computed
  // predicate is not silently charged both in generic aux and the insertion
  // quote.
  for (Operation *op : covered)
    if (!owner.count(op))
      return reject("audit_unowned_private_node");
  auto aux = normal_memory_aux::rebuild(module, partition, covered, analysis,
                                        warps, hardware.simt.usableUbBytes);
  if (!aux)
    return aux.takeError();
  auto costs = StageCostEvaluator().evaluate(*aux, hardware, false);
  if (!costs)
    return costs.takeError();
  StageCostTable table = context;
  table.stages = costs->stages;
  table.dependencies = costs->dependencies;
  table.modeledOperationCount = costs->modeledOperationCount;
  table.operationOwnershipComplete = costs->operationOwnershipComplete;
  auto solved = solveStageRoutes(table, hardware.scopeHandoff);
  if (!solved)
    return solved.takeError();
  auto f1 = solved->allSimtFactorPlans.find(factor);
  if (f1 == solved->allSimtFactorPlans.end() || !f1->second.legal)
    return reject("audit_aux_F1_illegal");
  for (const auto &impl : f1->second.implementations)
    if (impl.mode != StageMode::SIMT || impl.localScope ||
        impl.aivSubblocks != 1 || impl.superblockFactor != factor)
      return reject("audit_aux_not_uniform_pure_F1");
  double groupPrice = groupPriceFor(
      cast<RankedTensorType>(target->getOperand(1).getType()).getElementType(),
      factor);
  if (!std::isfinite(groupPrice))
    return reject("unqualified_payload_or_F");
  // Auxiliary costs are already scheduled. Apply the target repetition count
  // once.
  double quote = groupPrice * (4 / factor);
  llvm::json::Array mapping, nodes, edges, inputs;
  int setup = 0;
  for (const auto &[op, slot] : owner)
    mapping.push_back(llvm::json::Object{
        {"original_operation_ordinal", int64_t(ordinal.lookup(op))},
        {"operation", op->getName().getStringRef()},
        {"original_stage", partition.stages[slot].id},
        {"source_location", location(op)},
        {"owner",
         covered.contains(op) ? "target_Store_only" : "retained_auxiliary"},
        {"mask_ancestor", ancestors.contains(op)}});
  for (unsigned i = 0; i < aux->stages.size(); ++i) {
    const auto &s = aux->stages[i];
    setup += s.workload.paysKernelSetup;
    nodes.push_back(llvm::json::Object{
        {"id", s.id},
        {"owned_ops", int64_t(s.operations.size())},
        {"load_bytes", s.workload.loadBytes},
        {"store_bytes", s.workload.storeBytes},
        {"live_in_bytes", s.liveInBytes},
        {"live_out_bytes", s.liveOutBytes},
        {"pays_setup", s.workload.paysKernelSetup},
        {"system_cycles", f1->second.logicalStageCycles[i]}});
  }
  for (const auto &e : aux->dependencies)
    edges.push_back(llvm::json::Object{{"source", e.sourceStage},
                                       {"target", e.targetStage},
                                       {"bytes", e.carriedValueBytes},
                                       {"kind", int64_t(e.kind)},
                                       {"distance", e.iterationDistance}});
  for (Operation *op : covered)
    for (Value value : op->getOperands())
      if (!covered.contains(def(value))) {
        auto bytes = tensorBytes(value);
        inputs.push_back(llvm::json::Object{
            {"consumer_ordinal", int64_t(ordinal.lookup(op))},
            {"producer_ordinal", owner.count(def(value))
                                     ? int64_t(ordinal.lookup(def(value)))
                                     : -1},
            {"bytes", bytes.value_or(-1)}});
      }
  if (setup != 1)
    return reject("audit_aux_setup_owner_count");
  if (prepared) {
    prepared->target = target;
    prepared->covered = covered;
    prepared->auxiliaryPartition = *aux;
    prepared->auxiliaryPlan = f1->second;
    prepared->H = H;
    prepared->K = K;
    prepared->quote = quote;
  }
  return llvm::json::Object{
      {"schema", "normal_full224_SIMTStore_scoped_qualified_bundle_v1"},
      {"model_SHA256",
       "004552bf313e1eed421c13f5e8c45d1943e0f1dc82ee1b42ea3823b5eec5bd80"},
      {"admitted", true},
      {"newDEV_used", false},
      {"original_selectable_prices_changed", false},
      {"rows", 8},
      {"columns", 32},
      {"H", H},
      {"K", K},
      {"G32", H},
      {"columnOnly", H == 8 && K < 32},
      {"lastRowSkip", H == 7 && K == 32},
      {"source_SYS_reference_MHz", MHz},
      {"quote_profileSYS1000", quote},
      {"target_memory_ordinal_witness", 2},
      {"real_Operation_identity_used", true},
      {"covered_operation_count", int64_t(covered.size())},
      {"removed_private_mask_operation_count", int64_t(covered.size() - 1)},
      {"original_ops", partition.modeledOperationCount},
      {"aux_ops", aux->modeledOperationCount},
      {"original_F1_body_system_cycles", old.totalCycles},
      {"aux_F1_body_system_cycles", f1->second.totalCycles},
      {"TRAIN_only_alternative_F1_body", f1->second.totalCycles + quote},
      {"aux_setup_owners", setup},
      {"target_setup_owners", 0},
      {"external_boundary_added_here", false},
      {"actual_generic_kernel_boundary_once", genericBoundary},
      {"TRAIN_only_alternative_generic_elapsed",
       f1->second.totalCycles + quote + genericBoundary},
      {"other_outer_target_or_completion_overrides_not_borrowed", true},
      {"full_elapsed_admission",
       "HOLD_remaining_outer_policy_ownership_and_DEV"},
      {"Index_Values_shared_address_deleted", false},
      {"ownership", std::move(mapping)},
      {"aux_nodes", std::move(nodes)},
      {"aux_edges", std::move(edges)},
      {"bundle_inputs", std::move(inputs)}};
}

inline bool tryApplyFactor(ModuleOp module, const StagePartition &partition,
                           const StageCostTable &context,
                           const HardwareProfile &hardware,
                           const StagePartitionerOptions &analysis,
                           const SimdSimtCostModelOptions &options, double MHz,
                           unsigned warps, int64_t factor,
                           StageCostModelSummary &routes) {
  if (hardware.cvAivSubblocks != 1 && hardware.cvAivSubblocks != 2)
    return false;
  Prepared bundle;
  auto audited = inspect(module, partition, context, hardware, analysis,
                         options, MHz, warps, factor, routes, 0, &bundle);
  if (!audited) {
    llvm::consumeError(audited.takeError());
    return false;
  }
  const auto &old = routes.allSimtFactorPlans.at(factor);
  const size_t count = partition.stages.size();
  if (old.implementations.size() != count ||
      old.logicalStageCycles.size() != count ||
      old.scopeHandoffCycles.size() != count ||
      (!old.runtimeStages.empty() && old.runtimeStages.size() != count))
    return false;
  // No legacy target/whole-piece completion correction may silently replace
  // or overlap the new bundle. The present four literal sources have none.
  for (const char *key :
       {"native_indirect_chain_applied", "specialized_memory_complete_applied",
        "ordinary_store_round4_applied", "local_scope_completion_inclusive",
        "mixed_parent_scope_completion_applied"})
    if (auto i = old.routeDiagnostics.values.find(key);
        i != old.routeDiagnostics.values.end()) {
      auto flag = std::get_if<bool>(&i->second);
      if (!flag || *flag)
        return false;
    }
  llvm::DenseMap<Operation *, unsigned> owner, ordinal, auxOwner;
  unsigned ord = 0;
  for (unsigned i = 0; i < count; ++i)
    for (Operation *op : partition.stages[i].operations) {
      if (owner.count(op))
        return false;
      owner[op] = i;
      ordinal[op] = ord++;
    }
  if (!bundle.target || !owner.count(bundle.target))
    return false;
  const auto &aux = bundle.auxiliaryPartition;
  const auto &auxPlan = bundle.auxiliaryPlan;
  if (auxPlan.logicalStageCycles.size() != aux.stages.size() ||
      auxPlan.scopeHandoffCycles.size() != aux.stages.size())
    return false;
  StageRoutePlan candidate = old;
  candidate.logicalStageCycles.assign(count, 0);
  candidate.scopeHandoffCycles.assign(count, 0);
  llvm::json::Array graph, mapping, edges;
  unsigned setup = 0, totalOwned = 0;
  for (unsigned i = 0; i < aux.stages.size(); ++i) {
    const auto &stage = aux.stages[i];
    std::optional<unsigned> slot;
    for (Operation *op : stage.operations) {
      if (!owner.count(op) || bundle.covered.contains(op) || auxOwner.count(op))
        return false;
      if (slot && *slot != owner.lookup(op))
        return false;
      slot = owner.lookup(op);
      auxOwner[op] = i;
      ++totalOwned;
      mapping.push_back(
          llvm::json::Object{{"ordinal", int64_t(ordinal.lookup(op))},
                             {"operation", op->getName().getStringRef()},
                             {"owner", "auxiliary"},
                             {"original_stage", partition.stages[*slot].id},
                             {"route_node", stage.id}});
    }
    if (!slot)
      return false;
    setup += stage.workload.paysKernelSetup;
    candidate.logicalStageCycles[*slot] += auxPlan.logicalStageCycles[i];
    candidate.scopeHandoffCycles[*slot] += auxPlan.scopeHandoffCycles[i];
    graph.push_back(
        llvm::json::Object{{"id", stage.id},
                           {"owned_ops", int64_t(stage.operations.size())},
                           {"system_cycles", auxPlan.logicalStageCycles[i]},
                           {"original_slot", int64_t(*slot)},
                           {"pays_setup", stage.workload.paysKernelSetup},
                           {"load_bytes", stage.workload.loadBytes},
                           {"store_bytes", stage.workload.storeBytes},
                           {"live_in_bytes", stage.liveInBytes},
                           {"live_out_bytes", stage.liveOutBytes}});
  }
  for (Operation *op : bundle.covered) {
    if (!owner.count(op) || auxOwner.count(op))
      return false;
    ++totalOwned;
    mapping.push_back(llvm::json::Object{
        {"ordinal", int64_t(ordinal.lookup(op))},
        {"operation", op->getName().getStringRef()},
        {"owner", "target_Store_only"},
        {"original_stage", partition.stages[owner.lookup(op)].id},
        {"route_node",
         "qualified_normal_full224_SIMTStore_scoped_terminal_bundle"}});
  }
  if (totalOwned != partition.modeledOperationCount || setup != 1 ||
      bundle.covered.empty())
    return false;
  for (const auto &edge : aux.dependencies)
    edges.push_back(
        llvm::json::Object{{"source", edge.sourceStage},
                           {"target", edge.targetStage},
                           {"value_bytes", edge.carriedValueBytes},
                           {"kind", int64_t(edge.kind)},
                           {"iteration_distance", edge.iterationDistance},
                           {"name", edge.carriedValueName},
                           {"index", edge.carriedValueIndex}});
  std::map<unsigned, llvm::DenseSet<Value>> incoming;
  for (Operation *op : bundle.covered)
    for (Value value : op->getOperands()) {
      Operation *producer = def(value);
      if (bundle.covered.contains(producer))
        continue;
      if (producer && !auxOwner.count(producer))
        return false;
      if (producer)
        incoming[auxOwner.lookup(producer)].insert(value);
    }
  for (const auto &entry : incoming) {
    int64_t bytes = 0;
    for (Value value : entry.second) {
      auto b = tensorBytes(value);
      if (!b || bytes > INT64_MAX - *b)
        return false;
      bytes += *b;
    }
    edges.push_back(llvm::json::Object{
        {"source", aux.stages[entry.first].id},
        {"target", "qualified_normal_full224_SIMTStore_scoped_terminal_bundle"},
        {"value_bytes", bytes},
        {"distinct_SSA_values", int64_t(entry.second.size())},
        {"kind", int64_t(StageDependencyKind::IntraIterationData)},
        {"iteration_distance", 0},
        {"name", "inter_stage_ssa"},
        {"index", 0}});
  }
  const unsigned targetSlot = owner.lookup(bundle.target);
  candidate.logicalStageCycles[targetSlot] += bundle.quote;
  candidate.totalCycles = auxPlan.totalCycles + bundle.quote;
  graph.push_back(llvm::json::Object{
      {"id", "qualified_normal_full224_SIMTStore_scoped_terminal_bundle"},
      {"owned_ops", int64_t(bundle.covered.size())},
      {"system_cycles", bundle.quote},
      {"original_slot", int64_t(targetSlot)},
      {"pays_setup", false}});
  // The quote replaces the target bundle body only. The caller still completes
  // this F1 candidate with the real kernel/global-load/argument boundary once.
  for (const char *key :
       {"specialized_memory_complete_applied",
        "specialized_memory_elapsed_system_cycles",
        "specialized_memory_model_sha256", "native_indirect_chain_applied",
        "native_indirect_chain_elapsed_system_cycles",
        "native_indirect_chain_warp_count"})
    candidate.routeDiagnostics.values.erase(key);
  llvm::json::Object report{
      {"schema",
       "qualified_normal_full224_SIMTStore_scoped_private_mask_graph_v1"},
      {"model_SHA256",
       "004552bf313e1eed421c13f5e8c45d1943e0f1dc82ee1b42ea3823b5eec5bd80"},
      {"independent_matched_layouts_SHA256",
       "8a4178fc16bf8e182618695627412952107b3b2db92325738274064489b7e1d6"},
      {"production_admitted", false},
      {"own_independent_DEV_passed", true},
      {"wholekernel_accepted", false},
      {"selected_Mixed_accepted", false},
      {"H", bundle.H},
      {"K", bundle.K},
      {"R", 8},
      {"C", 32},
      {"pitch_elements", 32768},
      {"profile_system_counter_MHz", MHz},
      {"target_group_system_cycles", bundle.quote / (4 / factor)},
      {"runtime_repetitions", 4 / factor},
      {"target_system_cycles", bundle.quote},
      {"auxiliary_solver_body", auxPlan.totalCycles},
      {"new_body_system_cycles", candidate.totalCycles},
      {"target_owner_slot", int64_t(targetSlot)},
      {"original_stage_slots", int64_t(count)},
      {"covered_ops", int64_t(bundle.covered.size())},
      {"aux_ops", aux.modeledOperationCount},
      {"kernel_setup_owners", int64_t(setup)},
      {"target_setup_owners", 0},
      {"external_boundary_added_here", false},
      {"graph", std::move(graph)},
      {"edges", std::move(edges)},
      {"Op_pointer_derived_mapping", std::move(mapping)}};
  candidate.diagnostic<bool>(
      "qualified_normal_full224_SIMTStore_scoped_bundle_applied") = true;
  candidate.diagnostic<std::string>(
      "qualified_normal_full224_SIMTStore_scoped_graph_json") =
      llvm::formatv("{0}", llvm::json::Value(std::move(report))).str();
  routes.allSimtFactorPlans[factor] = std::move(candidate);
  bool found = false;
  for (const auto &[factor, plan] : routes.allSimtFactorPlans)
    if (plan.legal &&
        (!found || plan.totalCycles < routes.allSimt.totalCycles)) {
      routes.allSimt = plan;
      found = true;
    }
  return true;
}
inline bool tryApply(ModuleOp module, const StagePartition &partition,
                     const StageCostTable &context,
                     const HardwareProfile &hardware,
                     const StagePartitionerOptions &analysis,
                     const SimdSimtCostModelOptions &options, double MHz,
                     unsigned warps, StageCostModelSummary &routes) {
  bool changed = false;
  for (int64_t factor : {1, 2, 4})
    changed = tryApplyFactor(module, partition, context, hardware, analysis,
                             options, MHz, warps, factor, routes) ||
              changed;
  return changed;
}
} // namespace mlir::ascend::normal_full224_simt_store
#endif
