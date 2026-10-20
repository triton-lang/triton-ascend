#include "AscendModel/RouteModel/Models/PartialStructuredSimtLoadCostModel.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/Models/PartialStructuredSimtLoadModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
using namespace mlir::ascend;
#include "PartialStructuredStaticAnalysis.h"

StageMemoryCost PartialStructuredSimtLoadCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &impl) const {
  StageMemoryCost result;
  if (profile.partialStructuredMemoryModel !=
          "ttir_static_simt_load_factor_20261017" ||
      profile.target != "Ascend950PR/dav-c310" ||
      impl.mode != StageMode::SIMT ||
      (impl.superblockFactor != 2 && impl.superblockFactor != 4) ||
      impl.localScope || impl.aivSubblocks != 1 || stage.features.hasLoop ||
      stage.features.hasAtomicMemory ||
      stage.workload.privateStackLoadInstructions ||
      stage.workload.privateStackStoreInstructions ||
      (stage.costModelKind != StageCostModelKind::PartialContinuousTileMemory &&
       stage.costModelKind != StageCostModelKind::IndirectGatherMemory))
    return result;
  const AddressPatternSummary *target = nullptr;
  for (const auto &pattern : stage.workload.addressPatterns) {
    const bool discrete =
        llvm::any_of(pattern.axes, [](const AddressAxisSummary &a) {
          return a.hasUnknownComponent;
        });
    const bool regular =
        llvm::any_of(pattern.axes, [](const AddressAxisSummary &a) {
          return !a.hasUnknownComponent;
        });
    if (pattern.memoryOp == "tt.store" && discrete)
      return result;
    if (pattern.memoryOp == "tt.load" && discrete && regular) {
      if (target)
        return result;
      target = &pattern;
    }
  }
  if (!target || target->stageId != stage.id)
    return result;
  const auto &p = *target;
  Operation *memory = nullptr;
  bool unsupported = false;
  for (Operation *root : stage.operations)
    root->walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "tt.load" || name == "tt.store") {
        if (name == "tt.load") {
          auto t = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (t && t.getRank() == int64_t(p.axes.size()) &&
              std::equal(t.getShape().begin(), t.getShape().end(),
                         p.axes.begin(),
                         [](int64_t n, const AddressAxisSummary &a) {
                           return n == a.extent;
                         })) {
            if (memory)
              unsupported = true;
            memory = op;
          }
        }
      } else if (!llvm::is_contained({StringRef("tt.splat"),
                                      StringRef("tt.broadcast"),
                                      StringRef("tt.expand_dims"),
                                      StringRef("tt.reshape"),
                                      StringRef("tt.make_range"),
                                      StringRef("tt.addptr"),
                                      StringRef("tt.get_program_id"),
                                      StringRef("tt.bitcast"),
                                      StringRef("arith.constant"),
                                      StringRef("arith.addi"),
                                      StringRef("arith.subi"),
                                      StringRef("arith.muli"),
                                      StringRef("arith.extsi"),
                                      StringRef("arith.extui"),
                                      StringRef("arith.index_cast"),
                                      StringRef("arith.cmpi"),
                                      StringRef("arith.andi"),
                                      StringRef("arith.ori"),
                                      StringRef("arith.divsi"),
                                      StringRef("arith.remsi"),
                                      StringRef("arith.divui"),
                                      StringRef("arith.remui"),
                                      StringRef("scf.if"),
                                      StringRef("scf.yield")},
                                     name))
        unsupported = true;
    });
  if (!memory || unsupported)
    return result;
  auto load = dyn_cast<triton::LoadOp>(memory);
  auto tensor =
      load ? dyn_cast<RankedTensorType>(load.getType()) : RankedTensorType{};
  if (!tensor || !tensor.hasStaticShape() || tensor.getNumElements() > 2048 ||
      p.axes.size() != size_t(tensor.getRank()) ||
      load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL || load.getIsVolatile())
    return result;
  auto program = programTerm(load.getPtr());
  if (!program.valid || program.coefficient <= 0)
    return result;
  partial_memory::FactorLoadGeometry c;
  c.factor = impl.superblockFactor;
  c.programStride = program.coefficient;
  c.geometry.bits = p.elementBitWidth;
  c.geometry.floatingPoint = p.floatingPoint;
  c.geometry.warps = profile.baseSimtWarpCount;
  for (const auto &a : p.axes) {
    const bool d = a.hasUnknownComponent;
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
  if (load.getMask()) {
    StaticMask evaluator;
    auto mask = evaluator.evaluate(load.getMask());
    if (!mask || mask->shape != SmallVector<int64_t>(tensor.getShape()))
      return result;
    if (load.getOther()) {
      auto other = evaluator.evaluate(load.getOther());
      // Integer zero tensors are common; floating zero needs an explicit
      // constant check rather than interpreting any runtime value as zero.
      auto *def = load.getOther().getDefiningOp();
      auto floats = def ? def->getAttrOfType<DenseFPElementsAttr>("value")
                        : DenseFPElementsAttr{};
      if ((!other ||
           llvm::any_of(other->values, [](int64_t v) { return v != 0; })) &&
          (!floats ||
           !llvm::all_of(floats.getValues<APFloat>(),
                         [](const APFloat &v) { return v.isZero(); })))
        return result;
    }
    std::fill(c.lower.begin(), c.lower.end(),
              std::numeric_limits<int64_t>::max());
    std::fill(c.upper.begin(), c.upper.end(), 0);
    for (int64_t t = 0; t < tensor.getNumElements(); ++t) {
      if (!mask->values[t])
        continue;
      int64_t flat = t;
      for (size_t i = p.axes.size(); i-- > 0;) {
        const int64_t j = flat % p.axes[i].extent;
        flat /= p.axes[i].extent;
        c.lower[i] = std::min(c.lower[i], j);
        c.upper[i] = std::max(c.upper[i], j + 1);
      }
    }
    // Prove the entire mask is the inferred rectangle, not only its bounding
    // box.
    for (int64_t t = 0; t < tensor.getNumElements(); ++t) {
      int64_t flat = t;
      bool active = true;
      for (size_t i = p.axes.size(); i-- > 0;) {
        int64_t j = flat % p.axes[i].extent;
        flat /= p.axes[i].extent;
        active &= j >= c.lower[i] && j < c.upper[i];
      }
      if (active != bool(mask->values[t]))
        return result;
    }
  }
  auto prediction = partial_memory::predictFactorLoad(c);
  if (!prediction)
    return result;
  StagePartition auxiliary;
  auxiliary.operationOwnershipComplete = true;
  auxiliary.stages.push_back(stage);
  if (auto error =
          StageWorkloadAnalysis().analyzeExcludingMemory(auxiliary, memory)) {
    llvm::consumeError(std::move(error));
    return result;
  }
  result.remainingWorkload = std::move(auxiliary.stages.front().workload);
  result.load = prediction;
  result.replacesTensorMemory = true;
  result.coversSuperblockGroup = true;
  const bool masked =
      c.lower != std::vector<int64_t>(c.lower.size(), 0) ||
      !std::equal(
          c.upper.begin(), c.upper.end(), p.axes.begin(),
          [](int64_t n, const AddressAxisSummary &a) { return n == a.extent; });
  result.pricing = std::string("partial_simt_load_") +
                   (masked ? "mask_" : "unmasked_") + "f" +
                   std::to_string(c.factor) + "_group_formula_20261017";
  return result;
}
