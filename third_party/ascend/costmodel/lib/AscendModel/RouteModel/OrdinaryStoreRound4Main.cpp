// Compiler-side Store target increment with an explicit native contract.
#include "AscendModel/RouteModel/OrdinaryStoreRound4Main.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/OrdinaryStoreRound4MainAnalysis.h"
#include "AscendModel/RouteModel/OrdinaryStoreRound4MainContract.h"
#include "AscendModel/RouteModel/OrdinaryStoreRound4VisibilityAnalysis.h"
#include "AscendModel/RouteModel/OrdinaryStoreSelected.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <array>

using namespace mlir;
using namespace mlir::ascend;
namespace {
using namespace indirect_store_round4;
using namespace mlir::ascend::ordinary_store_main;
constexpr llvm::StringLiteral prefix = "ascend.costmodel.ordinary_store.";

// Fingerprints include the complete actual post-transform IR, excluding only
// this transport's own attrs. Caller names/seeds do not select model features.
std::string fingerprint(Operation *operation) {
  Operation *copy = operation->clone();
  copy->walk([](Operation *op) {
    SmallVector<StringAttr> names;
    for (NamedAttribute a : op->getAttrs())
      if (a.getName().getValue().starts_with(prefix) ||
          a.getName().getValue().starts_with("ascend.costmodel.ordinary_load."))
        names.push_back(a.getName());
    for (StringAttr name : names)
      op->removeAttr(name);
  });
  std::string text;
  llvm::raw_string_ostream stream(text);
  copy->print(stream, OpPrintingFlags().useLocalScope());
  stream.flush();
  copy->destroy();
  auto bytes = llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(text.data()), text.size());
  auto digest = llvm::SHA256::hash(bytes);
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}
std::string text(const llvm::json::Object &o, StringRef key) {
  auto v = o.getString(key);
  return v ? v->str() : std::string{};
}
bool truth(const llvm::json::Object &o, StringRef key) {
  auto v = o.getBoolean(key);
  return v && *v;
}
bool singletonOne(const llvm::json::Object &o, StringRef key) {
  auto *v = o.getArray(key);
  return v && v->size() == 1 && (*v)[0].getAsInteger() == int64_t(1);
}
// Only pointer-shape operations and equal-width pointer bitcasts are
// transparent to logical payload provenance. Numeric convert operations never
// enter this proof; physical StoreOp type and logical public pointer differ.
} // namespace

OrdinaryStoreMainResult
mlir::ascend::ordinaryStoreMainCost(const LogicalStage &stage,
                                    const HardwareProfile &profile,
                                    const StageImplementation &implementation,
                                    bool operationOwnershipComplete) {
  OrdinaryStoreMainResult result;
  auto fail = [&](const char *reason) {
    result.reason = reason;
    return result;
  };
  // Main has no b93 partialStructuredMemoryModel field. Do not copy its
  // non-existent profile gate. Bind the actual main profile identities below.
  if (stage.operations.empty())
    return result;
  ModuleOp module = stage.operations.front()->getParentOfType<ModuleOp>();
  if (!module)
    return fail("no_actual_module");
  auto carrier = module->getAttrOfType<StringAttr>(
      "ascend.costmodel.ordinary_store.main_route_capability_json");
  if (!carrier)
    return result;
  if (carrier.getValue().size() > 131072)
    return fail("oversized_compile_caller_contract");
  auto parsed = llvm::json::parse(carrier.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return fail("malformed_compile_caller_contract");
  }
  auto *capability = parsed->getAsObject();
  auto *c = capability
                ? capability->getObject("ordinary_store_main_compile_contract")
                : nullptr;
  if (!c || !truth(*c, "enabled"))
    return result;
  // Additive candidate only: existing R4 carriers/leases cannot silently
  // switch formulas. Callers must explicitly bind BOTH selected hashes.
  if (text(*c, "selected_SIMD_model_sha256") !=
          indirect_store_selected::selectedSimdModelSha256 ||
      text(*c, "selected_SIMT_model_sha256") !=
          indirect_store_selected::selectedSimtModelSha256)
    return fail("selected_formula_pair_not_explicitly_bound");
  if (!operationOwnershipComplete)
    return fail("operation_partition_ownership_not_complete");
  if ((text(*c, "schema") != "ordinary_store_main_explicit_compile_v1" ||
       text(*c, "ir_contract") !=
           "mlir-local-no-debug-ordinary-attrs-stripped-v1" ||
       text(*c, "invocation_scope") != "explicit_native_cost_analysis_only"))
    return fail("unknown_explicit_native_compile_contract");
  // This is a native cost-analysis API. It does not declare a JIT launch safe,
  // enable a Python backend option, create an allocation or issue a permit.
  if (implementation.superblockFactor != 1 || implementation.localScope ||
      implementation.aivSubblocks != 1 || stage.iterationCount != 1 ||
      stage.features.hasLoop || stage.features.hasAtomicMemory ||
      stage.features.hasDot || stage.features.hasReduction ||
      stage.features.hasPrefixScan ||
      (stage.costModelKind != StageCostModelKind::IndirectGatherMemory &&
       stage.costModelKind != StageCostModelKind::ContinuousTileStore &&
       stage.costModelKind != StageCostModelKind::ContinuousTileMemory))
    return fail("unsupported_stage_or_F_scope");
  if (profile.target != "Ascend950PR/dav-c310")
    return fail("unsupported_profile_target");
  // Exact direct roots only. Re-analysis below removes exactly one root;
  // nested ownership requires a separate exclusion-aware analysis adapter.
  for (Operation *op : stage.operations)
    if (op->getNumRegions())
      return fail("nested_operation_ownership_not_supported");
  auto entryIndex = c->getInteger("entry_pointer_argument"),
       ordinal = c->getInteger("store_ordinal");
  triton::FuncOp entry;
  unsigned entries = 0;
  module.walk([&](triton::FuncOp f) {
    auto visibility = f->getAttrOfType<StringAttr>("sym_visibility");
    if (!visibility || visibility.getValue() == "public") {
      entry = f;
      ++entries;
    }
  });
  if (entries != 1 || !entry || entry.getBody().empty() || !entryIndex ||
      !ordinal || *entryIndex < 0 || *ordinal < 0 ||
      static_cast<uint64_t>(*entryIndex) >= entry.getNumArguments())
    return fail("unique_entry_and_pointer_argument_not_bound");
  Value argument = entry.getArgument(static_cast<unsigned>(*entryIndex));
  if (!isa<triton::PointerType>(argument.getType()))
    return fail("bound_argument_is_not_pointer");
  SmallVector<triton::StoreOp> stores;
  entry.walk([&](triton::StoreOp s) { stores.push_back(s); });
  if (static_cast<uint64_t>(*ordinal) >= stores.size())
    return fail("bound_store_ordinal_missing");
  triton::StoreOp store = stores[*ordinal];
  if (!llvm::is_contained(stage.operations, store.getOperation()))
    return fail("target_not_owned_by_this_stage");
  main_ta::CallerFacts caller;
  {
    caller.enabled = true;
    caller.sourceSha256 = text(*c, "source_sha256");
    caller.entryAndIRBound =
        pointerRoot(store.getPtr()) == argument &&
        fingerprint(module) == text(*c, "module_fingerprint_sha256") &&
        fingerprint(store) == text(*c, "store_fingerprint_sha256");
    caller.addressIntegerDomainDeclared =
        text(*c, "loaded_index_value_contract") ==
            "signed_nonnegative_axis_extent_v1" &&
        main_ta::sha256(text(*c, "loaded_index_bounds_proof_sha256")) &&
        text(*c, "integer_address_arithmetic") ==
            "no_wrap_in_actual_IR_integer_width" &&
        main_ta::sha256(text(*c, "integer_no_wrap_proof_sha256"));
    const auto *transform = c->getObject("transform");
    caller.transformBound =
        transform && transform->getBoolean("layout_merge_applied") &&
        transform->getBoolean("layout_merge_applied") ==
            capability->getBoolean("layout_merge_applied") &&
        transform->getInteger("layout_coalescing_factor") == int64_t(1) &&
        transform->getInteger("layout_coalescing_factor") ==
            capability->getInteger("layout_coalescing_factor") &&
        transform->getInteger("layout_coalescing_axis") == int64_t(-1) &&
        transform->getInteger("layout_coalescing_axis") ==
            capability->getInteger("layout_coalescing_axis") &&
        singletonOne(*transform, "whole_kernel_superblock_factors") &&
        singletonOne(*transform, "scope_superblock_factors") &&
        singletonOne(*capability, "whole_kernel_superblock_factors") &&
        singletonOne(*capability, "scope_superblock_factors");
    if (const auto *life = c->getObject("lifecycle")) {
      caller.firstStore = truth(*life, "first_store");
      caller.allocationFresh =
          text(*life, "allocation") == "fresh_for_this_launch";
      caller.initializationGuardsOnly =
          text(*life, "initialization") == "uninitialized_payload_guards_only";
      caller.exclusiveAlias =
          text(*life, "aliasing") == "exclusive_target_no_external_alias";
      caller.notReused = text(*life, "reuse") == "not_reused";
      caller.retainedUntilCompletion =
          text(*life, "retention") == "valid_until_kernel_completion";
      caller.launchId = text(*life, "launch_id");
      caller.allocationId = text(*life, "allocation_id");
      caller.allocationEpoch = life->getInteger("allocation_epoch").value_or(0);
      std::array<StringRef, 5> names = {
          "allocation_proof_sha256", "initialization_proof_sha256",
          "alias_proof_sha256", "reuse_proof_sha256", "retention_proof_sha256"};
      for (size_t i = 0; i < names.size(); ++i)
        caller.lifecycleProofSha256[i] = text(*life, names[i]);
    }
    caller.incrementContract = text(*c, "increment_contract");
    caller.accountingBoundary = text(*c, "accounting_boundary");
    caller.unitContract = text(*c, "unit_contract");
    caller.ticksPerSystemCycle =
        c->getNumber("syscnt_ticks_per_system_cycle").value_or(0.0);
    caller.profileSha256 = text(*c, "profile_sha256");
    caller.unitSourceSha256 = text(*c, "unit_source_sha256");
    result.reason = main_ta::validateCaller(
        caller, profile.ordinaryStoreActiveProfileSha256,
        profile.ordinaryStoreUnitSourceSha256);
    if (!result.reason.empty())
      return result;
  }
  const auto visibility = text(*c, "visibility");
  if (visibility != "known" && visibility != "host" && visibility != "device")
    return fail("unknown_actual_coefficient_visibility");
  auto tensor = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!tensor || !tensor.hasStaticShape() || tensor.getRank() < 2 ||
      tensor.getRank() > 5 || tensor.getNumElements() > 8192 ||
      tensor.getNumElements() < 4)
    return fail("unsupported_static_tensor_geometry");
  Type element = tensor.getElementType();
  const bool integer = element.isInteger(1) || element.isInteger(8) ||
                       element.isInteger(16) || element.isInteger(32) ||
                       element.isInteger(64);
  const bool floating =
      isa<Float16Type, BFloat16Type, Float32Type, Float8E4M3FNType,
          Float8E5M2Type>(element) ||
      (implementation.mode == StageMode::SIMT && element.isF64());
  if (!integer && !floating)
    return fail("unsupported_value_type");
  if (store.getCache() != triton::CacheModifier::NONE)
    return fail("nondefault_cache_policy");
  if (store.getEvict() != triton::EvictionPolicy::NORMAL)
    return fail("nondefault_eviction_policy");
  if (!store.getBoundaryCheck().empty())
    return fail("block_pointer_boundary_not_supported");
  if (auto a = store->getAttrOfType<BoolAttr>("isVolatile"))
    if (a.getValue())
      return fail("volatile_target_not_supported");
  if (!integer && element.getIntOrFloatBitWidth() == 64 &&
      implementation.mode == StageMode::SIMD)
    return fail("SIMD_float64_outside_frozen_coverage");
  indirect_store_selected::Metadata m;
  m.mode = implementation.mode == StageMode::SIMD ? Mode::SIMD : Mode::SIMT;
  // Main's options.numWarps selects SIMT launch shape. The SIMD profile is
  // resolved before the SIMT warp curves and its frozen Store formula uses W1.
  m.factor = 1;
  m.baseWarps =
      implementation.mode == StageMode::SIMD ? 1 : profile.baseSimtWarpCount;
  const auto bits = element.getIntOrFloatBitWidth();
  m.elementBytes = bits == 1 ? 1 : bits / 8;
  Value logicalRoot = pointerRoot(store.getPtr());
  auto logicalPointer = dyn_cast<triton::PointerType>(argument.getType());
  if (logicalRoot != argument || !logicalPointer)
    return fail("logical_payload_entry_pointer_not_proven");
  Type logicalPayload = logicalPointer.getPointeeType();
  auto logicalBits = scalarPayloadBits(logicalPayload);
  if (!logicalBits || *logicalBits != bits)
    return fail("logical_physical_payload_width_mismatch");
  const bool logicalInteger =
      logicalPayload.isInteger(1) || logicalPayload.isInteger(8) ||
      logicalPayload.isInteger(16) || logicalPayload.isInteger(32) ||
      logicalPayload.isInteger(64);
  const bool logicalFloating =
      isa<Float16Type, BFloat16Type, Float32Type, Float8E4M3FNType,
          Float8E5M2Type>(logicalPayload) ||
      (implementation.mode == StageMode::SIMT && logicalPayload.isF64());
  if (!logicalInteger && !logicalFloating)
    return fail("unsupported_logical_payload_type");
  m.payloadTypeVerified = true;
  m.payloadIsFloat8 = isa<Float8E4M3FNType, Float8E5M2Type>(logicalPayload);
  m.payloadIsInt1 = logicalPayload.isInteger(1);
  if (m.payloadIsFloat8 && element.isInteger(8) &&
      implementation.mode == StageMode::SIMD)
    return fail("SIMD_FP8_bit_transport_outside_original_native_coverage");
  if (m.payloadIsFloat8 && !element.isInteger(8) &&
      implementation.mode == StageMode::SIMT)
    return fail(
        "SIMT_FP8_native_store_outside_original_bit_transport_coverage");
  m.shape.assign(tensor.getShape().begin(), tensor.getShape().end());
  std::vector<int64_t> runtimeCoefficients;
  if (visibility == "known") {
    auto terms = axes(store.getPtr());
    if (terms.size() != m.shape.size())
      return fail("static_axis_proof_missing");
    for (size_t i = 0; i < terms.size(); ++i) {
      const auto &axis = terms[i];
      if (m.shape[i] <= 1 || !axis.valid || (axis.loaded && axis.stride) ||
          (axis.loaded ? axis.loadedScale <= 0 : axis.stride <= 0))
        return fail("axis_not_exact_positive_separable_static_S_or_loaded_D");
      m.dAxes.push_back(bool(axis.loaded));
      m.knownStrides.push_back(axis.loaded ? axis.loadedScale : axis.stride);
    }
    m.visibility = Visibility::Known;
  } else {
    const auto terms = visibilityAxes(store.getPtr(), entry);
    if (terms.size() != m.shape.size())
      return fail("dynamic_axis_proof_missing");
    bool sawHost = false, sawDevice = false;
    for (size_t i = 0; i < terms.size(); ++i) {
      const auto &axis = terms[i];
      if (m.shape[i] <= 1 || !axis.valid || !axis.coordinate ||
          axis.pitch.scale <= 0)
        return fail("axis_or_uniform_scalar_coefficient_not_proven");
      m.dAxes.push_back(bool(axis.loadedIndex));
      sawHost |= axis.pitch.kind == PitchKind::HostArgument;
      sawDevice |= axis.pitch.kind == PitchKind::DeviceScalarLoad;
      if (visibility == "host") {
        auto value = hostPitchValue(axis.pitch, entry, *capability);
        if (!value)
          return fail("public_integer_scalar_option_and_IR_fact_mismatch");
        runtimeCoefficients.push_back(*value);
      }
    }
    if ((visibility == "host" && (!sawHost || sawDevice)) ||
        (visibility == "device" && (!sawDevice || sawHost)))
      return fail("declared_visibility_does_not_match_actual_scalar_sources");
    if (visibility == "device") {
      // Device coefficients stay opaque. Positivity/no-wrap is an explicit
      // native caller assumption bound by the existing actual IR/source hash;
      // no allocator/business producer or device-load value is invented here.
      if ((text(*c, "device_stride_value_contract") !=
               "signed_positive_integer_no_wrap_v1" ||
           !main_ta::sha256(text(*c, "device_stride_bounds_proof_sha256"))))
        return fail("opaque_device_coefficient_domain_not_declared");
      m.visibility = Visibility::Device;
      // Deliberately leave both vectors empty; hidden features remain zero.
    } else
      m.visibility = Visibility::Host;
  }
  m.lower.assign(m.shape.size(), 0);
  m.upper = m.shape;
  const auto representation = text(*c, "mask_representation");
  m.hasMask = representation == "rectangular";
  if (!m.hasMask && representation != "unmasked")
    return fail("source_mask_semantics_missing");
  Value mask = store.getMask();
  if (bool(mask) != m.hasMask)
    return fail("source_mask_IR_presence_conflict_or_canonicalized_full_mask");
  if (auto reason = rectangleMask(mask, tensor, m,
                                  "mask_not_static_constant_range_expression",
                                  "nonrectangular_mask"))
    return fail(reason);
  m.stageOwnershipVerified = true; // Exact target membership checked above.
  m.firstStore = caller.firstStore;
  m.matchedIncrementContractVerified = true; // Exact contract checked above.
  auto estimate = indirect_store_selected::tryPredict(
      m, m.visibility == Visibility::Host ? &runtimeCoefficients : nullptr,
      true);
  result.reason = estimate.reason;
  result.leaf = estimate.leaf;
  if (!estimate.eligible || !estimate.ticks)
    return result;
  StagePartition auxiliary;
  auxiliary.operationOwnershipComplete = true;
  auxiliary.stages.push_back(stage);
  auto &rest = auxiliary.stages.front();
  rest.operations.clear();
  for (Operation *op : stage.operations)
    if (op != store.getOperation())
      rest.operations.push_back(op);
  // Keep address/index/value/mask helpers and all auxiliary memory. The fitted
  // A-minus-B target price covers exactly the target store, not these helpers.
  if (auto error = StageWorkloadAnalysis().analyze(auxiliary)) {
    llvm::consumeError(std::move(error));
    return fail("auxiliary_workload_reanalysis_failed");
  }
  if (rest.iterationCount != 1)
    return fail("auxiliary_reanalysis_changed_iteration_contract");
  result.auxiliaryStage = std::move(rest);
  result.targetSystemCycles = *estimate.ticks / caller.ticksPerSystemCycle;
  result.reason =
      "selected_formula_native_compile_contract_eligible_unaccepted";
  return result;
}
