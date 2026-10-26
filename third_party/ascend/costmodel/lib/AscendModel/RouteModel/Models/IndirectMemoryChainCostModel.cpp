// Native indirect memory-chain ownership and complete-cost accounting.
#include "AscendModel/Analysis/StageMemoryPatternAnalysis.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/Models/IndirectGatherMemoryCostModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::ascend;

namespace {
using Operations = llvm::SmallPtrSet<Operation *, 32>;

static bool defaultMemory(Operation *op) {
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return !load.getMask() && !load.getIsVolatile() &&
           load.getCache() == triton::CacheModifier::NONE &&
           load.getEvict() == triton::EvictionPolicy::NORMAL;
  if (auto store = dyn_cast<triton::StoreOp>(op))
    return !store.getMask() &&
           store.getCache() == triton::CacheModifier::NONE &&
           store.getEvict() == triton::EvictionPolicy::NORMAL;
  return false;
}

static void loadedLeaves(Value value, Operations &loads, Operations &visited) {
  Operation *op = value.getDefiningOp();
  if (!op || !visited.insert(op).second)
    return;
  if (isa<triton::LoadOp>(op)) {
    loads.insert(op);
    return;
  }
  for (Value operand : op->getOperands())
    loadedLeaves(operand, loads, visited);
}

static bool depends(Value value, Operation *source) {
  Operations loads, visited;
  loadedLeaves(value, loads, visited);
  return loads.contains(source);
}

static void coverReshapes(Value value, Operations &covered) {
  while (auto reshape = value.getDefiningOp<triton::ReshapeOp>()) {
    if (reshape.getAllowReorder())
      return;
    covered.insert(reshape);
    value = reshape.getSrc();
  }
}

// Only these helpers are part of the measured canonical address chain.
// Arithmetic consuming loaded values is deliberately left for normal pricing.
static bool coverAddress(Value value, Operations &covered) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return true;
  if (!isa<triton::SplatOp, triton::AddPtrOp, triton::MakeRangeOp,
           triton::ReshapeOp, triton::ExpandDimsOp, triton::BroadcastOp,
           arith::ConstantOp, arith::AddIOp, arith::MulIOp>(op))
    return false;
  if (auto reshape = dyn_cast<triton::ReshapeOp>(op))
    if (reshape.getAllowReorder())
      return false;
  if (!llvm::all_of(op->getOperands(), [&](Value operand) {
        return coverAddress(operand, covered);
      }))
    return false;
  covered.insert(op);
  return true;
}

static bool contiguous(Operation *op, const StageMemoryPatternAnalysis &facts,
                       int64_t elements) {
  const auto *pattern = facts.lookupSummary(op);
  if (!pattern || pattern->axes.empty())
    return false;
  int64_t stride = 1;
  for (const auto &axis : llvm::reverse(pattern->axes)) {
    if (axis.extent <= 0 || axis.offsetAxisInfo != "structured" ||
        axis.knownStride != stride)
      return false;
    stride *= axis.extent;
  }
  return stride == elements;
}
} // namespace

std::optional<double>
IndirectGatherMemoryCostModel::nativeChainCost(const StagePartition &partition,
                                               const HardwareProfile &profile,
                                               const StageRoutePlan &plan) {
  const int64_t w = profile.baseSimtWarpCount, f = plan.routeSuperblockFactor;
  if (!plan.legal || plan.runtimePhysicalProgramCount != 1 ||
      plan.runtimeLogicalProgramsPerSchedulingSlot != f ||
      profile.target != "Ascend950PR/dav-c310" || w <= 0 || f <= 0 ||
      w > 64 / f || !partition.operationOwnershipComplete ||
      llvm::any_of(plan.implementations, [](const StageImplementation &impl) {
        return impl.mode != StageMode::SIMT || impl.localScope ||
               impl.aivSubblocks != 1;
      }))
    return std::nullopt;
  Operations all, memory;
  Operation *target = nullptr;
  for (const LogicalStage &stage : partition.stages) {
    if (stage.iterationCount != 1 || stage.features.hasLoop ||
        stage.features.conditionalBranchCount ||
        stage.features.divergentBranchCount || stage.features.hasAtomicMemory ||
        stage.features.hasReduction || stage.features.hasPrefixScan ||
        stage.features.hasDot || stage.workload.synchronizations ||
        stage.workload.sharedLocalLoadInstructions ||
        stage.workload.sharedLocalStoreInstructions ||
        stage.workload.privateStackLoadInstructions ||
        stage.workload.privateStackStoreInstructions)
      return std::nullopt;
    for (Operation *op : stage.operations) {
      if (op->getNumRegions())
        return std::nullopt;
      all.insert(op);
      if (!isa<triton::LoadOp, triton::StoreOp>(op)) {
        if (!isMemoryEffectFree(op) && !isa<triton::ReturnOp>(op))
          return std::nullopt;
        continue;
      }
      if (!defaultMemory(op))
        return std::nullopt;
      memory.insert(op);
      if (stage.costModelKind == StageCostModelKind::IndirectGatherMemory) {
        if (target || stage.workload.addressPatterns.size() != 1 ||
            stage.workload.addressPatterns[0].axes.empty() ||
            !llvm::all_of(stage.workload.addressPatterns[0].axes,
                          [](const AddressAxisSummary &a) {
                            return a.offsetAxisInfo == "unstructured";
                          }))
          return std::nullopt;
        target = op;
      }
    }
  }
  if (!target)
    return std::nullopt;
  const bool store = isa<triton::StoreOp>(target);
  const auto &selector =
      store ? profile.simtIndirectStoreModel : profile.simtIndirectLoadModel;
  if (selector != "unmasked_complete_factor_20261020" &&
      selector != "prefix_mask_rank_factor_20261017" &&
      selector != "random_mask_rank_factor_20261017")
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(
      store ? target->getOperand(1).getType() : target->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      !type.getElementType().isIntOrFloat())
    return std::nullopt;
  const int bits = type.getElementType().getIntOrFloatBitWidth();
  Type element = type.getElementType();
  if (!isa<IntegerType, Float16Type, BFloat16Type, Float32Type, Float64Type,
           Float8E4M3FNType, Float8E5M2Type>(element))
    return std::nullopt;
  const int64_t e = type.getNumElements();
  if ((bits != 8 && bits != 16 && bits != 32 && bits != 64) || e <= 0 ||
      llvm::any_of(type.getShape(), [](int64_t n) { return n <= 0; }))
    return std::nullopt;
  auto address = target->getOperand(0).getDefiningOp<triton::AddPtrOp>();
  if (!address)
    return std::nullopt;
  StageMemoryPatternAnalysis facts(target->getParentOfType<ModuleOp>());
  Operations indices, visited;
  loadedLeaves(address.getOffset(), indices, visited);
  Operation *index = nullptr;
  for (Operation *op : all)
    if (indices.contains(op) && (!index || op->isBeforeInBlock(index)))
      index = op;
  if (!index || !contiguous(index, facts, e))
    return std::nullopt;
  auto indexType = dyn_cast<RankedTensorType>(index->getResult(0).getType());
  if (!indexType || (!indexType.getElementType().isInteger(32) &&
                     !indexType.getElementType().isInteger(64)))
    return std::nullopt;
  Operations covered, coveredMemory;
  coveredMemory.insert(target);
  coveredMemory.insert(index);
  Operation *companion = nullptr;
  if (store) {
    Operations values;
    visited.clear();
    loadedLeaves(target->getOperand(1), values, visited);
    if (values.size() != 1)
      return std::nullopt;
    companion = *values.begin();
  } else {
    for (Operation *op : memory)
      if (isa<triton::StoreOp>(op) && depends(op->getOperand(1), target)) {
        if (companion)
          return std::nullopt;
        companion = op;
      }
  }
  if (!companion || companion == index || !all.contains(companion) ||
      !contiguous(companion, facts, e))
    return std::nullopt;
  auto companionType =
      dyn_cast<RankedTensorType>(store ? companion->getResult(0).getType()
                                       : companion->getOperand(1).getType());
  if (!companionType || companionType.getNumElements() != e ||
      companionType.getElementType() != type.getElementType())
    return std::nullopt;
  coveredMemory.insert(companion);
  for (Operation *op : memory)
    if (!coveredMemory.contains(op) &&
        (isa<triton::StoreOp>(op) || !contiguous(op, facts, e)))
      return std::nullopt;
  covered = coveredMemory;
  for (Operation *op : coveredMemory)
    if (op != target && !coverAddress(op->getOperand(0), covered))
      return std::nullopt;
  covered.insert(address);
  if (!coverAddress(address.getPtr(), covered))
    return std::nullopt;
  coverReshapes(address.getOffset(), covered);
  coverReshapes(store ? target->getOperand(1) : companion->getOperand(1),
                covered);
  // A shared helper still belongs to the remaining workload.
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> remove;
    for (Operation *op : covered)
      if (!coveredMemory.contains(op) &&
          llvm::any_of(op->getUsers(), [&](Operation *user) {
            return !covered.contains(user);
          }))
        remove.push_back(op);
    for (Operation *op : remove) {
      covered.erase(op);
      changed = true;
    }
  } while (changed);
  StagePartition extra = partition;
  for (LogicalStage &stage : extra.stages) {
    llvm::erase_if(stage.operations,
                   [&](Operation *op) { return covered.contains(op); });
    stage.workload.paysKernelSetup = false;
  }
  llvm::erase_if(extra.stages, [](const LogicalStage &stage) {
    return stage.operations.empty();
  });
  double remainder = 0;
  if (!extra.stages.empty()) {
    if (auto error = StageWorkloadAnalysis().analyze(extra)) {
      llvm::consumeError(std::move(error));
      return std::nullopt;
    }
    auto costs = StageCostEvaluator().evaluate(extra, profile, false);
    if (!costs) {
      llvm::consumeError(costs.takeError());
      return std::nullopt;
    }
    costs->logicalProgramCountHint = costs->sourceLogicalProgramCountHint = f;
    costs->physicalCoreCountHint = 1;
    auto routes = solveStageRoutes(*costs, profile.scopeHandoff);
    if (!routes) {
      llvm::consumeError(routes.takeError());
      return std::nullopt;
    }
    auto found = routes->allSimtFactorPlans.find(f);
    if (found == routes->allSimtFactorPlans.end() || !found->second.legal)
      return std::nullopt;
    remainder = found->second.totalCycles;
  }
  if (!profile.indirectMemoryParameters)
    return std::nullopt;
  return nativeChainCycles(*profile.indirectMemoryParameters, store, type,
                           indexType.getElementTypeBitWidth(), w, f) +
         remainder;
}
