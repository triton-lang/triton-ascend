#include "AscendModel/RouteModel/Models/PartialStructuredMemoryCostModel.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/Models/PartialStructuredMemoryModel.h"
#include "AscendModel/RouteModel/Models/PartialStructuredSimtLoadCostModel.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::ascend;

StageMemoryCost
PartialStructuredMemoryCostModel::cost(const LogicalStage &stage,
                                       const HardwareProfile &profile,
                                       const StageImplementation &impl) const {
  if (profile.partialStructuredMemoryModel == "ttir_static_partial_20261020")
    return costStatic(stage, profile, impl);
  StageMemoryCost result;
  if (impl.mode == StageMode::SIMT && impl.superblockFactor > 1)
    return PartialStructuredSimtLoadCostModel().cost(stage, profile, impl);
  if ((profile.partialStructuredMemoryModel !=
           "ttir_q1_short_formula_20261014" &&
       profile.partialStructuredMemoryModel !=
           "ttir_static_simt_load_factor_20261017") ||
      profile.target != "Ascend950PR/dav-c310" || impl.superblockFactor != 1 ||
      impl.localScope || impl.aivSubblocks != 1 || stage.features.hasLoop ||
      stage.features.hasAtomicMemory || stage.features.activeLaneRatio != 1 ||
      stage.workload.privateStackLoadInstructions != 0 ||
      stage.workload.privateStackStoreInstructions != 0 ||
      stage.workload.addressPatterns.size() != 1 ||
      (stage.costModelKind != StageCostModelKind::PartialContinuousTileMemory &&
       stage.costModelKind != StageCostModelKind::IndirectGatherMemory))
    return result;
  const auto &p = stage.workload.addressPatterns.front();
  bool store = p.memoryOp == "tt.store";
  if ((!store && p.memoryOp != "tt.load") || p.masked || p.stageId != stage.id)
    return result;
  Operation *memory = nullptr;
  // Accept just one tensor memory root and its pure address producers.
  // Other arithmetic, value preparation, control and memory retain fallback.
  for (Operation *op : stage.operations) {
    auto name = op->getName().getStringRef();
    if (name == "tt.load" || name == "tt.store") {
      if (memory || name != p.memoryOp)
        return result;
      memory = op;
    } else if (!llvm::is_contained(
                   std::initializer_list<StringRef>{
                       StringRef("tt.splat"), StringRef("tt.broadcast"),
                       StringRef("tt.expand_dims"), StringRef("tt.make_range"),
                       StringRef("tt.addptr"), StringRef("tt.bitcast"),
                       StringRef("arith.constant"), StringRef("arith.addi"),
                       StringRef("arith.subi"), StringRef("arith.muli"),
                       StringRef("arith.extsi"), StringRef("arith.extui"),
                       StringRef("arith.index_cast")},
                   name))
      return result;
  }
  if (!memory || memory->getNumOperands() != (store ? 2u : 1u))
    return result;
  llvm::SmallPtrSet<Operation *, 16> addressOps;
  auto visit = [&](auto &&self, Value value) -> void {
    auto *op = value.getDefiningOp();
    if (!op || !addressOps.insert(op).second)
      return;
    for (Value operand : op->getOperands())
      self(self, operand);
  };
  visit(visit, memory->getOperand(0));
  // Default cache/eviction only; calibrated kernels did not exercise hints.
  if (auto value = memory->getAttrOfType<IntegerAttr>("cache"))
    if (value.getInt() != int64_t(triton::CacheModifier::NONE))
      return result;
  if (auto value = memory->getAttrOfType<IntegerAttr>("evict"))
    if (value.getInt() != int64_t(triton::EvictionPolicy::NORMAL))
      return result;
  if (auto value = memory->getAttrOfType<BoolAttr>("isVolatile"))
    if (value.getValue())
      return result;
  Type type =
      store ? memory->getOperand(1).getType() : memory->getResult(0).getType();
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() ||
      p.axes.size() != size_t(tensor.getRank()))
    return result;
  Type element = tensor.getElementType();
  const bool integer = element.isInteger(8) || element.isInteger(16) ||
                       element.isInteger(32) || element.isInteger(64);
  const bool floating =
      isa<Float16Type, BFloat16Type, Float32Type, Float8E4M3FNType,
          Float8E5M2Type>(element) ||
      (element.isF64() && !store && impl.mode == StageMode::SIMT);
  if (!integer && !floating)
    return result;
  partial_memory::Geometry g;
  g.bits = p.elementBitWidth;
  g.floatingPoint = p.floatingPoint;
  g.warps = profile.baseSimtWarpCount;
  for (const auto &a : p.axes) {
    bool d = a.hasUnknownComponent;
    if (d) {
      if (!a.opaqueMultiplier || *a.opaqueMultiplier <= 0 ||
          (a.knownStride && *a.knownStride != 0) ||
          a.regularity != "opaque_loaded")
        return result;
    } else if (!a.knownStride || *a.knownStride <= 0)
      return result;
    g.axes.push_back({a.extent, d ? *a.opaqueMultiplier : *a.knownStride, d});
  }
  const bool simd = impl.mode == StageMode::SIMD;
  auto kind = simd ? (store ? partial_memory::Model::SimdStore
                            : partial_memory::Model::SimdLoad)
                   : (store ? partial_memory::Model::SimtStore
                            : partial_memory::Model::SimtLoad);
  auto prediction = partial_memory::predict(g, kind);
  if (!prediction)
    return result;
  if (simd) {
    // Preserve preparation for a later output or a value bitcast owned by
    // this stage. Only the target and its own address producers are covered.
    StagePartition auxiliary;
    auxiliary.operationOwnershipComplete = true;
    auxiliary.stages.push_back(stage);
    auto &remaining = auxiliary.stages.front();
    remaining.operations.clear();
    for (Operation *op : stage.operations)
      if (op != memory && !addressOps.contains(op))
        remaining.operations.push_back(op);
    if (auto error = StageWorkloadAnalysis().analyze(auxiliary)) {
      llvm::consumeError(std::move(error));
      return result;
    }
    result.remainingWorkload = std::move(remaining.workload);
  }
  if (store)
    result.store = prediction;
  else
    result.load = prediction;
  result.replacesTensorMemory = true;
  result.includesAddressWork = simd;
  result.pricing = std::string(simd ? "partial_simd_" : "partial_simt_") +
                   (store ? "store_first_" : "load_") +
                   "short_formula_20261014";
  return result;
}
