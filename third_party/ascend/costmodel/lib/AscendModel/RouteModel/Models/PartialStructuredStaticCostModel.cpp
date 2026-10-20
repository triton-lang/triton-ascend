#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/Models/PartialStructuredMemoryCostModel.h"
#include "AscendModel/RouteModel/Models/PartialStructuredStaticModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

using namespace mlir;
using namespace mlir::ascend;
#include "PartialStructuredStaticAnalysis.h"

StageMemoryCost PartialStructuredMemoryCostModel::costStatic(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &impl) const {
  StageMemoryCost result;
  if (profile.target != "Ascend950PR/dav-c310" || impl.localScope ||
      impl.aivSubblocks != 1 || stage.features.hasLoop ||
      stage.features.hasAtomicMemory ||
      stage.workload.privateStackLoadInstructions ||
      stage.workload.privateStackStoreInstructions ||
      (stage.costModelKind != StageCostModelKind::PartialContinuousTileMemory &&
       stage.costModelKind != StageCostModelKind::IndirectGatherMemory))
    return result;
  const AddressPatternSummary *target = nullptr;
  for (const auto &p : stage.workload.addressPatterns) {
    bool d = false, s = false;
    for (const auto &a : p.axes) {
      d |= a.hasUnknownComponent;
      s |= !a.hasUnknownComponent;
    }
    if (d && s && (p.memoryOp == "tt.load" || p.memoryOp == "tt.store")) {
      if (target)
        return result;
      target = &p;
    }
  }
  if (!target || target->stageId != stage.id)
    return result;
  const auto &p = *target;
  bool store = p.memoryOp == "tt.store", simd = impl.mode == StageMode::SIMD;
  Operation *memory = nullptr;
  bool ambiguous = false;
  for (Operation *root : stage.operations)
    root->walk([&](Operation *op) {
      if (op->getName().getStringRef() != p.memoryOp)
        return;
      Type t = store ? op->getOperand(1).getType() : op->getResult(0).getType();
      auto tensor = dyn_cast<RankedTensorType>(t);
      if (!tensor || !tensor.hasStaticShape() ||
          size_t(tensor.getRank()) != p.axes.size() ||
          !std::equal(tensor.getShape().begin(), tensor.getShape().end(),
                      p.axes.begin(),
                      [](int64_t n, const AddressAxisSummary &a) {
                        return n == a.extent;
                      }))
        return;
      if (memory)
        ambiguous = true;
      memory = op;
    });
  if (ambiguous)
    return result;
  if (!memory || (simd && !llvm::is_contained(stage.operations, memory)))
    return result;
  auto tensor = cast<RankedTensorType>(store ? memory->getOperand(1).getType()
                                             : memory->getResult(0).getType());
  auto element = tensor.getElementType();
  const bool integer = element.isInteger(1) || element.isInteger(8) ||
                       element.isInteger(16) || element.isInteger(32) ||
                       element.isInteger(64);
  const bool floating = isa<Float16Type, BFloat16Type, Float32Type,
                            Float8E4M3FNType, Float8E5M2Type>(element) ||
                        (!simd && element.isF64());
  if (!integer && !floating)
    return result;
  if (auto a = memory->getAttrOfType<IntegerAttr>("cache"))
    if (a.getInt() != int64_t(triton::CacheModifier::NONE))
      return result;
  if (auto a = memory->getAttrOfType<IntegerAttr>("evict"))
    if (a.getInt() != int64_t(triton::EvictionPolicy::NORMAL))
      return result;
  if (auto a = memory->getAttrOfType<BoolAttr>("isVolatile"))
    if (a.getValue())
      return result;
  partial_memory::FactorLoadGeometry c;
  c.factor = impl.superblockFactor;
  c.geometry.bits = element.isInteger(1) ? 8 : p.elementBitWidth;
  c.geometry.floatingPoint = floating;
  c.geometry.warps = profile.baseSimtWarpCount;
  for (const auto &a : p.axes) {
    bool d = a.hasUnknownComponent;
    if (d ? (!a.opaqueMultiplier || *a.opaqueMultiplier <= 0 ||
             (a.knownStride && *a.knownStride != 0) ||
             a.regularity != "opaque_loaded")
          : (!a.knownStride || *a.knownStride <= 0))
      return result;
    c.geometry.axes.push_back(
        {a.extent, d ? *a.opaqueMultiplier : *a.knownStride, d});
    c.lower.push_back(0);
    c.upper.push_back(a.extent);
  }
  if (c.factor > 1) {
    auto term = programTerm(memory->getOperand(0));
    if (!term.valid || term.coefficient <= 0)
      return result;
    c.programStride = term.coefficient;
  }
  Value mask;
  if (auto l = dyn_cast<triton::LoadOp>(memory)) {
    mask = l.getMask();

  } else if (auto s = dyn_cast<triton::StoreOp>(memory))
    mask = s.getMask();
  if (mask) {
    StaticMask evaluator;
    auto v = evaluator.evaluate(mask);
    if (!v || v->shape != SmallVector<int64_t>(tensor.getShape()) ||
        v->values.size() != size_t(tensor.getNumElements()))
      return result;
    std::fill(c.lower.begin(), c.lower.end(),
              std::numeric_limits<int64_t>::max());
    std::fill(c.upper.begin(), c.upper.end(), 0);
    for (int64_t k = 0; k < tensor.getNumElements(); ++k)
      if (v->values[k]) {
        int64_t flat = k;
        for (size_t i = p.axes.size(); i-- > 0;) {
          int64_t j = flat % p.axes[i].extent;
          flat /= p.axes[i].extent;
          c.lower[i] = std::min(c.lower[i], j);
          c.upper[i] = std::max(c.upper[i], j + 1);
        }
      }
    for (int64_t k = 0; k < tensor.getNumElements(); ++k) {
      int64_t flat = k;
      bool active = true;
      for (size_t i = p.axes.size(); i-- > 0;) {
        int64_t j = flat % p.axes[i].extent;
        flat /= p.axes[i].extent;
        active &= j >= c.lower[i] && j < c.upper[i];
      }
      if (active != bool(v->values[k]))
        return result;
    }
  }
  bool masked = false;
  for (size_t i = 0; i < p.axes.size(); ++i)
    masked |= c.lower[i] != 0 || c.upper[i] != p.axes[i].extent;
  if (auto l = dyn_cast<triton::LoadOp>(memory)) {
    if (masked && l.getOther()) {
      StaticMask evaluator;
      auto v = evaluator.evaluate(l.getOther());
      auto *op = l.getOther().getDefiningOp();
      auto fp = op ? op->getAttrOfType<DenseFPElementsAttr>("value")
                   : DenseFPElementsAttr{};
      if ((!v || llvm::any_of(v->values, [](int64_t n) { return n != 0; })) &&
          (!fp || !llvm::all_of(fp.getValues<APFloat>(),
                                [](const APFloat &n) { return n.isZero(); })))
        return result;
    }
  }
  auto model = simd ? (store ? partial_memory::Model::SimdStore
                             : partial_memory::Model::SimdLoad)
                    : (store ? partial_memory::Model::SimtStore
                             : partial_memory::Model::SimtLoad);
  auto prediction = partial_memory::predictStatic(c, model);
  if (!prediction)
    return result;
  StagePartition auxiliary;
  auxiliary.operationOwnershipComplete = true;
  auxiliary.stages.push_back(stage);
  if (simd) {
    // A native fragment covers only pure target-address helpers owned by this
    // stage. External index/value loads and all other operations remain priced.
    llvm::SmallPtrSet<Operation *, 32> address;
    auto visit = [&](auto &&self, Value v) -> bool {
      auto *op = v.getDefiningOp();
      if (!op || !llvm::is_contained(stage.operations, op))
        return true;
      if (!address.insert(op).second)
        return true;
      auto name = op->getName().getStringRef();
      if (!llvm::is_contained(
              {StringRef("tt.splat"), StringRef("tt.broadcast"),
               StringRef("tt.expand_dims"), StringRef("tt.make_range"),
               StringRef("tt.addptr"), StringRef("tt.bitcast"),
               StringRef("arith.constant"), StringRef("arith.addi"),
               StringRef("arith.subi"), StringRef("arith.muli"),
               StringRef("arith.extsi"), StringRef("arith.extui"),
               StringRef("arith.index_cast")},
              name))
        return false;
      for (auto operand : op->getOperands())
        if (!self(self, operand))
          return false;
      return true;
    };
    if (!visit(visit, memory->getOperand(0)))
      return result;
    auto &rest = auxiliary.stages.front();
    rest.operations.clear();
    for (auto *op : stage.operations)
      if (op != memory && !address.contains(op))
        rest.operations.push_back(op);
    if (auto error = StageWorkloadAnalysis().analyze(auxiliary)) {
      llvm::consumeError(std::move(error));
      return result;
    }
  } else if (auto error = StageWorkloadAnalysis().analyzeExcludingMemory(
                 auxiliary, memory)) {
    llvm::consumeError(std::move(error));
    return result;
  }
  result.remainingWorkload = std::move(auxiliary.stages.front().workload);
  // Both SIMT prices cover all F logical programs. The resource aggregator
  // separates auxiliary work, applies its factor, and adds this group once.
  if (store)
    result.store = prediction;
  else
    result.load = prediction;
  result.replacesTensorMemory = true;
  result.includesAddressWork = simd;
  result.coversSuperblockGroup = !simd;
  result.pricing = std::string("partial_") + (simd ? "simd_" : "simt_") +
                   (store ? "store_first_" : "load_") +
                   (masked ? "mask_" : "unmasked_") + "static_20261020";
  return result;
}
