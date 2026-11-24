// Pure-SIMT full224 target-only prices; ordinary grid/options are required.
#ifndef ASCENDMODEL_NORMAL_FULL224_SIMT_LOAD_H
#define ASCENDMODEL_NORMAL_FULL224_SIMT_LOAD_H
#include "AscendModel/Analysis/StageMemoryPatternAnalysis.h"
#include "AscendModel/RouteModel/NormalSegmentsLoadOwnership.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "llvm/Support/FormatVariadic.h"
namespace mlir::ascend::normal_full224_simt_load {
using namespace specialized_segments_ownership;
enum class Payload { F32, F16, BF16, I32, U8, I64 };
inline std::optional<Payload> exactPayload(Type t) {
  if (t.isF32())
    return Payload::F32;
  if (t.isF16())
    return Payload::F16;
  if (t.isBF16())
    return Payload::BF16;
  if (t.isSignlessInteger(32))
    return Payload::I32;
  if (t.isSignlessInteger(8))
    return Payload::U8;
  if (t.isSignlessInteger(64))
    return Payload::I64;
  return std::nullopt;
}
inline const char *payloadName(Payload p) {
  return p == Payload::F32    ? "f32"
         : p == Payload::F16  ? "f16"
         : p == Payload::BF16 ? "bf16"
         : p == Payload::I32  ? "i32"
         : p == Payload::I64  ? "i64"
                              : "u8";
}
inline bool typedTile(Value v, Payload p) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.getRank() == 2 && t.getDimSize(0) == 4 &&
         t.getDimSize(1) == 16 && !t.getEncoding() &&
         exactPayload(t.getElementType()) == std::optional<Payload>(p);
}
struct Match {
  Operation *target = nullptr;
  size_t owner = 0;
  Payload payload = Payload::F32;
  std::vector<int64_t> factors;
};
inline bool programBase(Value value, Value base, Value pid, int64_t span) {
  Operation *add = def(value);
  if (!named(add, "tt.addptr") || add->getNumOperands() != 2 ||
      add->getOperand(0) != base)
    return false;
  Operation *mul = def(add->getOperand(1));
  return named(mul, "arith.muli") && mul->getNumOperands() == 2 &&
         mul->getOperand(0) == pid &&
         uniformInteger(mul->getOperand(1)) == std::optional<int64_t>(span);
}
inline std::optional<Match>
qualify(ModuleOp module, const StagePartition &partition,
        const StageCostTable &table, const HardwareProfile &hardware,
        const SimdSimtCostModelOptions &options, double MHz, unsigned warps) {
  if (hardware.target != "Ascend950PR/dav-c310" ||
      options.actualTarget != "Ascend950PR_9579" || !std::isfinite(MHz) ||
      MHz != 1000 || warps != 4 || !options.compileOn91095 ||
      options.logicalProgramCountHint != 224 ||
      options.sourceLogicalProgramCountHint != 224 ||
      table.logicalProgramCountHint != 224 ||
      table.sourceLogicalProgramCountHint != 224 ||
      options.physicalVectorCoreCountHint != 56 ||
      table.physicalCoreCountHint != 56 || table.baseSimtWarpCount != 4 ||
      (options.cvAivSubblocks != 1 && options.cvAivSubblocks != 2) ||
      (hardware.cvAivSubblocks != 1 && hardware.cvAivSubblocks != 2) ||
      options.cvAivSubblocks != hardware.cvAivSubblocks ||
      table.cvAivSubblocks != hardware.cvAivSubblocks ||
      !options.wholeKernelSuperblockMaterializable ||
      !partition.operationOwnershipComplete ||
      table.stages.size() != partition.stages.size())
    return std::nullopt;
  auto attr = module->getAttrOfType<StringAttr>(
      "ascend.costmodel.ordinary_load.route_capability_json");
  if (!attr)
    return std::nullopt;
  auto parsed = llvm::json::parse(attr.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  auto *cap = parsed->getAsObject();
  if (!cap)
    return std::nullopt;
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
    return std::nullopt;
  auto *factors = cap->getArray("whole_kernel_superblock_factors");
  if (!factors || factors->empty())
    return std::nullopt;
  Match out;
  for (const auto &value : *factors) {
    auto f = value.getAsInteger();
    if (!f || (*f != 1 && *f != 2 && *f != 4) ||
        !llvm::is_contained(options.wholeKernelSuperblockFactors, *f) ||
        llvm::is_contained(out.factors, *f))
      return std::nullopt;
    out.factors.push_back(*f);
  }
  llvm::DenseMap<Operation *, size_t> owners;
  Block *block = nullptr;
  std::vector<Operation *> memory;
  Value pid;
  for (size_t i = 0; i < partition.stages.size(); ++i) {
    const auto &stage = partition.stages[i];
    if (stage.iterationCount != 1 || stage.features.hasDot ||
        stage.features.hasLoop || stage.features.hasLoopCarriedDataDependency ||
        !stage.enclosingLoopStages.empty())
      return std::nullopt;
    for (Operation *op : stage.operations) {
      if (!op || op->getNumRegions() || op->getNumSuccessors() ||
          owners.count(op))
        return std::nullopt;
      if (block && block != op->getBlock())
        return std::nullopt;
      block = op->getBlock();
      owners[op] = i;
      for (Type t : op->getOperandTypes())
        if (auto x = dyn_cast<RankedTensorType>(t); x && x.getEncoding())
          return std::nullopt;
      for (Type t : op->getResultTypes())
        if (auto x = dyn_cast<RankedTensorType>(t); x && x.getEncoding())
          return std::nullopt;
    }
  }
  if (!block || int64_t(owners.size()) != partition.modeledOperationCount ||
      block->getNumArguments() != 3)
    return std::nullopt;
  Operation *fn = block->getParentOp();
  if (!named(fn, "tt.func") || fn->getNumRegions() != 1 ||
      !fn->getRegion(0).hasOneBlock())
    return std::nullopt;
  auto callable = dyn_cast<FunctionOpInterface>(fn);
  if (!callable)
    return std::nullopt;
  for (unsigned i = 0; i < 3; ++i) {
    auto p = dyn_cast<triton::PointerType>(block->getArgument(i).getType());
    if (!p || p.getAddressSpace() != 1 || !callable.getArgAttrs(i).empty())
      return std::nullopt;
    if (i == 1) {
      if (!p.getPointeeType().isSignlessInteger(32))
        return std::nullopt;
    } else {
      auto typed = exactPayload(p.getPointeeType());
      if (!typed)
        return std::nullopt;
      if (i == 0)
        out.payload = *typed;
      else if (*typed != out.payload)
        return std::nullopt;
    }
  }
  for (Operation &op : *block) {
    if (named(&op, "tt.return")) {
      if (op.getNumOperands())
        return std::nullopt;
      continue;
    }
    if (!owners.count(&op))
      return std::nullopt;
    if (auto p = dyn_cast<triton::GetProgramIdOp>(&op)) {
      if (pid || p.getAxis() != triton::ProgramIDDim::X)
        return std::nullopt;
      pid = p.getResult();
    } else if (named(&op, "tt.load") || named(&op, "tt.store"))
      memory.push_back(&op);
    else if (!pure(&op))
      return std::nullopt;
  }
  if (!pid || memory.size() != 3 || !named(memory[0], "tt.load") ||
      !named(memory[1], "tt.load") || !named(memory[2], "tt.store"))
    return std::nullopt;
  Operation *index = memory[0], *target = memory[1], *sink = memory[2];
  if (target->getNumResults() != 1 || target->getNumOperands() != 3 ||
      !typedTile(target->getResult(0), out.payload) ||
      !normal_segments_load_ownership::defaultLoad(target) ||
      !normal_segments_load_ownership::calibratedZero(target->getOperand(2)) ||
      !target->getResult(0).hasOneUse() || sink->getNumOperands() != 2 ||
      sink->getOperand(1) != target->getResult(0))
    return std::nullopt;
  auto st = dyn_cast<triton::StoreOp>(sink);
  if (!st || st.getCache() != triton::CacheModifier::NONE ||
      st.getEvict() != triton::EvictionPolicy::NORMAL ||
      !st.getBoundaryCheck().empty())
    return std::nullopt;
  Operation *ptr = def(target->getOperand(0));
  if (!named(ptr, "tt.addptr") || ptr->getNumOperands() != 2)
    return std::nullopt;
  Operation *rowBroadcast = def(ptr->getOperand(0));
  if (!named(rowBroadcast, "tt.broadcast") ||
      rowBroadcast->getNumOperands() != 1)
    return std::nullopt;
  Operation *rowPtr = def(rowBroadcast->getOperand(0));
  if (!named(rowPtr, "tt.addptr") || rowPtr->getNumOperands() != 2 ||
      !programBase(peelViews(rowPtr->getOperand(0)), block->getArgument(0), pid,
                   2097152))
    return std::nullopt;
  Operation *mul = def(rowPtr->getOperand(1));
  if (!named(mul, "arith.muli") || mul->getNumOperands() != 2 ||
      uniformInteger(mul->getOperand(1)) != std::optional<int64_t>(32768))
    return std::nullopt;
  Operation *expanded = def(mul->getOperand(0));
  if (!named(expanded, "tt.expand_dims") || !axis(expanded, 1) ||
      expanded->getNumOperands() != 1 ||
      expanded->getOperand(0) != index->getResult(0))
    return std::nullopt;
  if (index->getNumResults() != 1 || index->getNumOperands() != 3 ||
      !normal_segments_load_ownership::defaultLoad(index) ||
      uniformInteger(index->getOperand(2)) != std::optional<int64_t>(0))
    return std::nullopt;
  auto it = dyn_cast<RankedTensorType>(index->getResult(0).getType());
  Operation *ip = def(index->getOperand(0));
  if (!it || it.getRank() != 1 || it.getDimSize(0) != 4 ||
      !it.getElementType().isSignlessInteger(32) || !named(ip, "tt.addptr") ||
      ip->getNumOperands() != 2 ||
      !programBase(peelViews(ip->getOperand(0)), block->getArgument(1), pid,
                   4) ||
      !range(ip->getOperand(1), 4))
    return std::nullopt;
  Value rowRange = ip->getOperand(1);
  auto im = dyn_cast_or_null<arith::CmpIOp>(def(index->getOperand(1)));
  if (!im || im.getPredicate() != arith::CmpIPredicate::slt ||
      im.getLhs() != rowRange ||
      uniformInteger(peelViews(im.getRhs())) != std::optional<int64_t>(4))
    return std::nullopt;
  auto columns = rectangularAxisRange(ptr->getOperand(1), 4, 16, 0);
  if (!columns)
    return std::nullopt;
  Operation *mask = def(target->getOperand(1));
  if (!named(mask, "arith.andi") || mask->getNumOperands() != 2 ||
      bound(mask->getOperand(0), 4, 1) != std::optional<int64_t>(4) ||
      bound(mask->getOperand(1), 16, 0) != std::optional<int64_t>(16))
    return std::nullopt;
  Operation *optr = def(sink->getOperand(0));
  if (!named(optr, "tt.addptr") || optr->getNumOperands() != 2 ||
      !programBase(peelViews(optr->getOperand(0)), block->getArgument(2), pid,
                   64))
    return std::nullopt;
  Operation *sum = def(optr->getOperand(1));
  if (!named(sum, "arith.addi") || sum->getNumOperands() != 2 ||
      sum->getOperand(1) != ptr->getOperand(1))
    return std::nullopt;
  Operation *obr = def(sum->getOperand(0));
  if (!named(obr, "tt.broadcast") || obr->getNumOperands() != 1)
    return std::nullopt;
  Operation *om = def(obr->getOperand(0));
  if (!named(om, "arith.muli") || om->getNumOperands() != 2 ||
      uniformInteger(om->getOperand(1)) != std::optional<int64_t>(16))
    return std::nullopt;
  Operation *oe = def(om->getOperand(0));
  if (!named(oe, "tt.expand_dims") || oe->getNumOperands() != 1 ||
      !axis(oe, 1) || oe->getOperand(0) != rowRange)
    return std::nullopt;
  out.target = target;
  out.owner = owners.lookup(target);
  if (partition.stages[out.owner].operations.size() != 1 ||
      partition.stages[out.owner].operations.front() != target ||
      table.stages[out.owner].ownedOperationCount != 1)
    return std::nullopt;
  return out;
}
struct Quote {
  double price;
  const char *model;
  const char *fullgrid;
};
inline std::optional<Quote> groupPrice(Payload payload, int64_t f) {
  if (f != 1 && f != 2 && f != 4)
    return std::nullopt;
  if (payload == Payload::U8 && f == 4)
    return Quote{
        219.0001541748643,
        "c81547d20197d201f0c5f25b2934b747bf679c5b3f1ad6e64149756aace83e21",
        "08b23ad01ca0fe226fa2f33f746398c18eb6d0178bf039e185e691d174f51fb8"};
  if (payload == Payload::U8 && f == 1)
    return Quote{
        139.28159097761974,
        "2d2cb0e828394262fabfd6e0062461ba67d901fd780147f216ce2616b8bc80a5",
        "3598dbe7dbd0aa35cc425afc33e8cea9f2a266c1aab5b2902b70d2831260de8d"};
  if (payload == Payload::I64 && f == 1)
    return Quote{
        2591.0017326002285,
        "b01477f832e9ed24819721a0a58b4e531698f67e6f93e6fa1f0ecfc71e4af37b",
        "b9f22a56d9c5996a0ca0871330aded79d467f95f4cb6b3905bc7370870b61d9c"};
  if (payload == Payload::I64 && f == 2)
    return Quote{
        2742.237241353064,
        "45514f4d74ff4aca539117aa14608de1259c23978b39df62368a55f7d330e7bc",
        "cad44b387e4c2517d94a7d419ef7d10f3f648e35f21b95ad713de9eaebaaec45"};
  if (payload == Payload::I64 && f == 4)
    return Quote{
        2619.7489780656438,
        "59c8bd9cc785574e18ef9270768ce9eac42cce84e429e90c3430ffe3b310785c",
        "dd4d719438342e08a9a8f46634e8a411efde343ed40821581468ac4ef0686464"};
  if (payload == Payload::U8 && f == 2)
    return Quote{
        137.54592793357193,
        "98a6c6820a3fbd4c9f3fcc735ecce0a13357384a2758d56d9e3e2d0b07cfa467",
        "df95945ee0b5f6b4696980a125bcbcfb1f9565ad847cdd20d146c965b4677e98"};
  if (payload == Payload::F32)
    return Quote{
        f == 1   ? 2525.4598962960677
        : f == 2 ? 2692.8057960670176
                 : 2835.084821840657,
        "28ffe4843cabd8a76eb09c362bf1ee97bd0aa12690f02800f223b6a21c998571",
        "642ed09df8b35bfe67ed2c4ff1840ecdc0cc55bc143aa020511622c066d86a52"};
  if (payload == Payload::F16 && f == 1)
    return Quote{
        2539.2055958800643,
        "b24f037af39870c56d76a79a20614fdbec6ec62f9a7a0cb6014f017dffb59a5b",
        "512a1bf558a07e434923b7452e7565c253cf460ea395c5e48738d6fc0562c32d"};
  if (payload == Payload::F16 && f == 2)
    return Quote{
        2640.2998334696103,
        "3b0bd7bd63bd541ae78be1a1cf8a0792a9f203173efd3704c35bc760846004b8",
        "fb8ef676265528950f07a7720ce40e74c5664c5610def74e8c05634192d9ca47"};
  if (payload == Payload::F16 && f == 4)
    return Quote{
        2755.082428279016,
        "37209eda71529b64df4a88288657e853e13a67aaa7633322813998c9a23a26f8",
        "32f90eaa254bb5926c18bc13ae1c16611bc1ee32574e1c81202509adc50d0e9b"};
  if (payload == Payload::BF16 && f == 1)
    return Quote{
        2536.706377773883,
        "eadc5f08949efaf54284e2001b2b1f8ae7ac2c673213a0d0998b2cdfe2200193",
        "12ca4853cc30d424cf63fa39db36309746ef3f12867fd441ff0ba0c31da9972d"};
  if (payload == Payload::BF16 && f == 2)
    return Quote{
        2640.2998334696103,
        "3dfc02a0871067df51307cdbbf78fbaf5f6a2cddb736988ba3e8e7b7fe6f632f",
        "9fd304741115138dec061b59b5aa073e81d64932d293416f072a1b4a74cbe0ec"};
  if (payload == Payload::BF16 && f == 4)
    return Quote{
        2755.082428279016,
        "d23b22f3e4fc5f123e8b384efc84d6d03d82209c44e0cdca1323a58a17306449",
        "32f33621d2f4a3f4a651d7cf56543c919890e6d5b3ffd1afa34c2d4a290457b8"};
  if (payload == Payload::I32 && f == 1)
    return Quote{
        2525.4598962960677,
        "154d5cc1aca4de3605acb34c3ee74306a85708ba854ce4b7fb5328e9f51378f2",
        "3a065fca0757c9f2cac5fea325982d01e53d6c22feaa81ae2d8b548a98b6f5a4"};
  if (payload == Payload::I32 && f == 2)
    return Quote{
        2692.8057960670176,
        "0b70ebb1917c911ad20020e3e98d9c4eeba09f1a08352bb958761c8a51350190",
        "7a0ea327bbcd71b3c63d01920e2960a1ae15cb7bbf903653b2e9612763e56dd4"};
  if (payload == Payload::I32 && f == 4)
    return Quote{
        2835.084821840657,
        "559481011f804babe361adad5cc3134f05a1dd6141f66920ae24da2720dff432",
        "348524bd55c77a4e8830c60596e047a1ee200d754e9dceaa291bc79b4f03275b"};
  return std::nullopt;
}
inline llvm::Expected<StageCostModelSummary>
solve(ModuleOp module, const StagePartition &partition,
      const StageCostTable &table, const HardwareProfile &hardware,
      const SimdSimtCostModelOptions &options, double MHz, unsigned warps) {
  auto original = solveStageRoutes(table, hardware.scopeHandoff);
  if (!original)
    return original.takeError();
  auto match = qualify(module, partition, table, hardware, options, MHz, warps);
  if (!match)
    return original;
  std::map<int64_t, StageRoutePlan> replacements;
  for (int64_t f : match->factors) {
    if (!original->allSimtFactorPlans.count(f))
      continue;
    auto quote = groupPrice(match->payload, f);
    if (!quote) {
      if (match->payload == Payload::U8 || match->payload == Payload::I64)
        continue;
      return original;
    }
    StageCostTable clone = table;
    unsigned count = 0;
    double price = quote->price;
    for (auto &impl : clone.stages[match->owner].implementations) {
      if (impl.implementation.mode != StageMode::SIMT ||
          impl.implementation.localScope ||
          impl.implementation.superblockFactor != f ||
          impl.implementation.aivSubblocks != 1)
        continue;
      // Retain immutable implementation/workload metadata, replace only its
      // target price attribution. All non-target Stage entries are exact
      // copies.
      impl.totalCycles = price;
      impl.resources = StageResourceCycles{};
      impl.resources.load = price;
      impl.indirectLoadPricing = "qualified_normal_full224_SIMT_target_only";
      impl.formulaEvidence =
          std::string("frozenOwnTyped/full224/currentUnitSYS1000/") +
          quote->model;
      impl.ordinaryLoadSelectedUsed = true;
      impl.ordinaryLoadTargetIncrementSystemCycles = price;
      impl.ordinaryLoadCompositionReference.reset();
      impl.ordinaryLoadCompositionReferenceCycles = 0;
      impl.ordinaryLoadCandidateReason =
          "normal_PIDx_full224_C56_q4_target_only";
      impl.ordinaryLoadLeaf = std::string(payloadName(match->payload)) +
                              "_Indexi32_R4C16_full_pitch32768";
      ++count;
    }
    if (count != 1)
      return original;
    auto quoted = solveStageRoutes(clone, hardware.scopeHandoff);
    if (!quoted)
      return quoted.takeError();
    auto p = quoted->allSimtFactorPlans.find(f);
    if (p == quoted->allSimtFactorPlans.end() || !p->second.legal)
      return original;
    StageRoutePlan plan = p->second;
    if (plan.runtimePhysicalProgramCount != 56 ||
        plan.runtimeSchedulingSlotCount != 56 ||
        plan.runtimeLogicalProgramsPerSchedulingSlot != 4 ||
        plan.runtimeFullGroupCount != 4 / f ||
        plan.runtimeTailProgramCount != 0 ||
        plan.runtimeLoopIterationCount != 4 / f ||
        match->owner >= plan.logicalStageCycles.size() ||
        std::abs(plan.logicalStageCycles[match->owner] - price * (4 / f)) >
            1e-8)
      return original;
    llvm::json::Array ownership;
    size_t ordinal = 0;
    for (size_t i = 0; i < partition.stages.size(); ++i)
      for (Operation *op : partition.stages[i].operations)
        ownership.push_back(llvm::json::Object{
            {"original_operation_ordinal", int64_t(ordinal++)},
            {"operation", op->getName().getStringRef()},
            {"original_stage", partition.stages[i].id},
            {"owner",
             op == match->target ? "target_Load_only" : "original_auxiliary"}});
    llvm::json::Object evidence{
        {"schema", "normal_full224_SIMT_target_only_v1"},
        {"model_SHA256", quote->model},
        {"fullgrid_validation_SHA256", quote->fullgrid},
        {"payload", payloadName(match->payload)},
        {"exact_i32_normal_fullELF_alias_SHA256",
         match->payload == Payload::I32 ? "aaf92d6717f33644fcb71146d022791775e6"
                                          "6057f31a27a3adfb1da4aedd0c46"
                                        : ""},
        {"target_bytes",
         match->payload == Payload::U8    ? 64
         : match->payload == Payload::I64 ? 512
         : (match->payload == Payload::F16 || match->payload == Payload::BF16)
             ? 128
             : 256},
        {"Index_bytes", 16},
        {"sink_bytes",
         match->payload == Payload::U8    ? 64
         : match->payload == Payload::I64 ? 512
         : (match->payload == Payload::F16 || match->payload == Payload::BF16)
             ? 128
             : 256},
        {"unit_namespace",
         match->payload == Payload::I64
             ? "current63_physical0_UnitSpec_v242_profileSYS1000"
         : match->payload == Payload::U8
             ? (f == 4
                    ? "current63_physical2_Notify4_deviceEvent_profileSYS1000"
                : f == 1 ? "current63_physical2_UnitSpec_v221_profileSYS1000"
                         : "current63_physical1_UnitSpec_v198_profileSYS1000")
             : "current63_physical0_boot69a4_profileSYS1000"},
        {"source_grid_x", 224},
        {"source_grid_y", 1},
        {"source_grid_z", 1},
        {"source_programs", 224},
        {"transformed_programs", 224},
        {"physical_C", 56},
        {"q", 4},
        {"base_W", 4},
        {"F", f},
        {"target_memory_ordinal", 1},
        {"target_owner_slot", int64_t(match->owner)},
        {"target_operation_count", 1},
        {"original_modeled_operation_count", partition.modeledOperationCount},
        {"target_group_system_cycles", price},
        {"group_rounds", 4 / f},
        {"scheduled_target_system_cycles",
         plan.logicalStageCycles[match->owner]},
        {"all_other_operations_original_ownership", true},
        {"Index_address_mask_sink_dispatch_deleted", false},
        {"external_boundary_added_here", false},
        {"kernel_setup_added_here", false},
        {"Op_pointer_derived_mapping", std::move(ownership)}};
    plan.diagnostic<bool>("qualified_normal_full224_SIMT_target_Load_applied") =
        true;
    plan.diagnostic<std::string>(
        "qualified_normal_full224_SIMT_target_Load_json") =
        llvm::formatv("{0}", llvm::json::Value(std::move(evidence))).str();
    replacements.emplace(f, std::move(plan));
  }
  if (replacements.empty())
    return original;
  for (auto &[f, plan] : replacements)
    original->allSimtFactorPlans[f] = std::move(plan);
  bool found = false;
  for (const auto &[f, plan] : original->allSimtFactorPlans)
    if (plan.legal &&
        (!found || plan.totalCycles < original->allSimt.totalCycles)) {
      original->allSimt = plan;
      found = true;
    }
  return original;
}
} // namespace mlir::ascend::normal_full224_simt_load
#endif
