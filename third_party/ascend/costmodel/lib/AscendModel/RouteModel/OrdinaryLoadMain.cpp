// Visible-geometry Load projection. Device index contents are never inspected.
#include "AscendModel/RouteModel/OrdinaryLoadMain.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/OrdinaryLoadSelected.h"
#include "AscendModel/RouteModel/OrdinaryStoreRound4VisibilityAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
using namespace mlir;
using namespace mlir::ascend;
namespace {
using namespace mlir::ascend::ordinary_store_main;
using namespace indirect_store_round4;
OrdinaryLoadMainResult targetCost(triton::LoadOp load,
                                  const LogicalStage &stage,
                                  const HardwareProfile &profile,
                                  const StageImplementation &implementation,
                                  const llvm::json::Object &capability) {
  OrdinaryLoadMainResult result;
  auto fail = [&](const char *why) {
    result.reason = why;
    return result;
  };
  auto tensor = dyn_cast<RankedTensorType>(load.getType());
  if (!tensor || !tensor.hasStaticShape() || tensor.getRank() < 2 ||
      tensor.getRank() > 5 || tensor.getNumElements() < 4 ||
      tensor.getNumElements() > 8192)
    return fail("Load_static_geometry_outside_domain");
  auto entry = load->getParentOfType<triton::FuncOp>();
  auto root = dyn_cast_or_null<BlockArgument>(pointerRoot(load.getPtr()));
  if (!entry || !root || root.getOwner() != &entry.getBody().front())
    return fail("Load_entry_pointer_provenance_not_verified");
  auto pointer = dyn_cast<triton::PointerType>(root.getType());
  const Type element = tensor.getElementType();
  // A byte transport cannot silently change logical FP8 into integer data.
  if (!pointer || !isa<IntegerType, FloatType>(pointer.getPointeeType()) ||
      !isa<IntegerType, FloatType>(element) ||
      pointer.getPointeeType().getIntOrFloatBitWidth() !=
          element.getIntOrFloatBitWidth())
    return fail("Load_logical_payload_type_not_verified");
  const bool integer = element.isInteger(1) || element.isInteger(8) ||
                       element.isInteger(16) || element.isInteger(32) ||
                       element.isInteger(64);
  const bool floating =
      isa<Float16Type, BFloat16Type, Float32Type, Float8E4M3FNType,
          Float8E5M2Type>(element) ||
      (implementation.mode == StageMode::SIMT && element.isF64());
  if (!integer && !floating)
    return fail("Load_payload_type_outside_domain");
  const Type logicalPayload = pointer.getPointeeType();
  const bool logicalFP8 = isa<Float8E4M3FNType, Float8E5M2Type>(logicalPayload);
  if (logicalPayload != element && !(logicalFP8 && element.isInteger(8)))
    return fail("Load_logical_payload_cast_outside_supported_transport");
  if (logicalFP8 &&
      ((implementation.mode == StageMode::SIMD && element.isInteger(8)) ||
       (implementation.mode == StageMode::SIMT && !element.isInteger(8))))
    return fail("Load_FP8_transport_does_not_match_native_mode");
  if (load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL ||
      !load.getBoundaryCheck().empty() || load.getIsVolatile())
    return fail("Load_nondefault_memory_semantics");
  indirect_load_selected::Metadata m;
  m.shape.assign(tensor.getShape().begin(), tensor.getShape().end());
  m.mode = implementation.mode == StageMode::SIMD ? Mode::SIMD : Mode::SIMT;
  m.factor = 1;
  m.baseWarps = m.mode == Mode::SIMD ? 1 : profile.baseSimtWarpCount;
  const unsigned bits = element.getIntOrFloatBitWidth();
  m.elementBytes = bits == 1 ? 1 : bits / 8;
  m.payloadTypeVerified = true;
  m.payloadIsInt1 = logicalPayload.isInteger(1);
  m.payloadIsFloat8 = logicalFP8;
  m.logical.verified = true;
  m.logical.integer = isa<IntegerType>(logicalPayload);
  m.logical.floating = isa<FloatType>(logicalPayload);
  m.logical.bfloat16 = isa<BFloat16Type>(logicalPayload);
  auto staticAxes = axes(load.getPtr());
  const bool known =
      staticAxes.size() == m.shape.size() &&
      llvm::all_of(staticAxes, [](const Axis &axis) {
        return axis.valid && !(axis.loaded && axis.stride) &&
               (axis.loaded ? axis.loadedScale > 0 : axis.stride > 0);
      });
  std::vector<int64_t> runtime;
  if (known) {
    m.visibility = Visibility::Known;
    for (const auto &axis : staticAxes) {
      m.dAxes.push_back(bool(axis.loaded));
      m.knownStrides.push_back(axis.loaded ? axis.loadedScale : axis.stride);
    }
  } else {
    auto dynamicAxes = visibilityAxes(load.getPtr(), entry);
    if (dynamicAxes.size() != m.shape.size())
      return fail("Load_separable_axes_not_proven");
    bool host = false, device = false;
    for (const auto &axis : dynamicAxes) {
      if (!axis.valid || !axis.coordinate || axis.pitch.scale <= 0)
        return fail("Load_separable_scalar_stride_not_proven");
      host |= axis.pitch.kind == PitchKind::HostArgument;
      device |= axis.pitch.kind == PitchKind::DeviceScalarLoad;
      m.dAxes.push_back(bool(axis.loadedIndex));
    }
    if (host && device)
      return fail("Load_mixed_Host_Device_strides_unsupported");
    if (device) {
      // Unknown pitch features stay opaque. The positive/no-wrap domain must
      // be an explicit caller contract; it cannot be inferred from a load.
      if (!capability.getBoolean("ordinary_load_device_stride_positive_no_wrap")
               .value_or(false))
        return fail("Load_Device_stride_domain_not_declared");
      m.visibility = Visibility::Device;
    } else if (host) {
      m.visibility = Visibility::Host;
      for (const auto &axis : dynamicAxes) {
        auto value = hostPitchValue(axis.pitch, entry, capability);
        if (!value)
          return fail("Load_Host_scalar_option_IR_binding_missing");
        runtime.push_back(*value);
      }
    } else
      return fail("Load_unknown_stride_source");
  }
  m.lower.assign(m.shape.size(), 0);
  m.upper = m.shape;
  Value mask = load.getMask();
  m.hasMask = bool(mask);
  if (auto reason = rectangleMask(
          mask, tensor, m,
          "Load_data_dependent_mask_requires_Host_contract_or_fallback",
          "Load_nonrectangular_mask"))
    return fail(reason);
  m.stageOwnershipVerified = true;
  m.matchedIncrementContractVerified = true;
  auto estimate = indirect_load_selected::tryPredict(
      m, m.visibility == Visibility::Host ? &runtime : nullptr);
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
    if (op != load.getOperation())
      rest.operations.push_back(op);
  if (auto error = StageWorkloadAnalysis().analyze(auxiliary)) {
    llvm::consumeError(std::move(error));
    return fail("Load_auxiliary_reanalysis_failed");
  }
  if (rest.iterationCount != 1)
    return fail("Load_auxiliary_iteration_contract_changed");
  result.auxiliaryStage = std::move(rest);
  result.targetSystemCycles =
      *estimate.ticks; // SYS_CNT / profile system cycle = 1.
  result.reason = "visible_Load_geometry_random_index_prior_target_only";
  return result;
}
} // namespace

OrdinaryLoadMainResult
mlir::ascend::ordinaryLoadMainCost(const LogicalStage &stage,
                                   const HardwareProfile &profile,
                                   const StageImplementation &implementation,
                                   bool operationOwnershipComplete) {
  OrdinaryLoadMainResult result;
  result.reason = "Load_no_eligible_single_target";
  if (!operationOwnershipComplete || stage.operations.empty() ||
      implementation.superblockFactor != 1 || implementation.localScope ||
      implementation.aivSubblocks != 1 || stage.iterationCount != 1 ||
      stage.features.hasLoop || stage.features.hasAtomicMemory ||
      stage.features.hasDot || stage.features.hasReduction ||
      stage.features.hasPrefixScan ||
      profile.target != "Ascend950PR/dav-c310" ||
      profile.ordinaryStoreActiveProfileSha256.size() != 64 ||
      profile.ordinaryStoreUnitSourceSha256.size() != 64)
    return result;
  auto module = stage.operations.front()->getParentOfType<ModuleOp>();
  if (!module)
    return result;
  llvm::json::Object capability;
  if (auto attr = module->getAttrOfType<StringAttr>(
          "ascend.costmodel.ordinary_load.route_capability_json")) {
    if (attr.getValue().size() > 131072)
      return result;
    auto parsed = llvm::json::parse(attr.getValue());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return result;
    }
    auto object = parsed->getAsObject();
    if (!object)
      return result;
    capability = std::move(*object);
  }
  if (!capability.getBoolean("ordinary_load_selected_enabled").value_or(true)) {
    result.reason = "Load_formula_explicitly_disabled";
    return result;
  }
  for (Operation *op : stage.operations) {
    if (op->getNumRegions())
      return OrdinaryLoadMainResult{};
    auto load = dyn_cast<triton::LoadOp>(op);
    if (!load)
      continue;
    auto candidate =
        targetCost(load, stage, profile, implementation, capability);
    if (!candidate.targetSystemCycles) {
      if (!result.targetSystemCycles)
        result.reason = candidate.reason;
      continue;
    }
    if (result.targetSystemCycles) {
      result = OrdinaryLoadMainResult{};
      result.reason = "Load_multiple_targets_require_separate_accounting";
      return result;
    }
    result = std::move(candidate);
  }
  return result;
}
