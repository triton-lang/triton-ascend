//===- SpecializedSegmentsOwnershipAdapter.h --------------------*- C++ -*-===//
// Match and partition the calibrated target without mutating the input IR.
// Replace a proven target bundle only after independently qualifying its price.
// The original IR is never mutated. Unsupported cases retain the full old
// route.
#ifndef ASCENDMODEL_SPECIALIZED_SEGMENTS_OWNERSHIP_ADAPTER_H
#define ASCENDMODEL_SPECIALIZED_SEGMENTS_OWNERSHIP_ADAPTER_H

#include "AscendModel/Analysis/StagePartitioner.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace mlir::ascend::specialized_segments_ownership {

struct TargetBundle {
  Operation *target = nullptr;
  // Ordered in original block order; identity is Operation*, never a location.
  std::vector<Operation *> coveredOperations;
  StagePartition auxiliaryPartition;
  std::vector<Value> bundleInputs;
  std::vector<std::string> bundleInputAuxiliaryStageIds;
  StageImplementation uniformImplementation;
  int64_t rows = 0, columns = 0, knownRows = 0, knownColumns = 0;
  int64_t rowPitchElements = 0;
  bool payloadIsI32 = false;
  bool payloadIsF16 = false;
  bool payloadIsBF16 = false;
  bool payloadIsU8 = false;
  bool payloadIsI64 = false;
  bool auxiliaryCostsRequireReevaluation = true;
  // A SIMT caller must additionally rebuild whole-route pressure/resource
  // legality; stale pressure copied from the original partition is not usable.
  bool requiresWholeSimtPressureReanalysis = false;
};

inline llvm::Error reject(llvm::StringRef reason) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
}
inline bool named(Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}
inline Operation *def(Value value) { return value.getDefiningOp(); }

inline std::optional<int64_t> uniformInteger(Value value) {
  Operation *op = def(value);
  if (!named(op, "arith.constant"))
    return std::nullopt;
  Attribute attr = op->getAttr("value");
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attr)) {
    if (integer.getType().getIntOrFloatBitWidth() > 64)
      return std::nullopt;
    return integer.getInt();
  }
  if (auto dense = dyn_cast_or_null<DenseIntElementsAttr>(attr))
    if (dense.isSplat() && dense.getElementType().getIntOrFloatBitWidth() <= 64)
      return dense.getSplatValue<llvm::APInt>().getSExtValue();
  return std::nullopt;
}
inline bool pure(Operation *op) {
  if (!op || op->getNumRegions() || op->getNumSuccessors() ||
      !isMemoryEffectFree(op))
    return false;
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Cases("arith.constant", "arith.muli", "arith.addi", true)
      .Cases("arith.cmpi", "arith.andi", "tt.make_range", true)
      .Cases("tt.splat", "tt.expand_dims", "tt.broadcast", true)
      .Case("tt.addptr", true)
      .Default(false);
}
inline Value peelViews(Value value) {
  while (Operation *op = def(value)) {
    if (!(named(op, "tt.splat") || named(op, "tt.expand_dims") ||
          named(op, "tt.broadcast")) ||
        op->getNumOperands() != 1)
      break;
    value = op->getOperand(0);
  }
  return value;
}
inline bool range(Value value, int64_t end) {
  Operation *op = def(value);
  if (!named(op, "tt.make_range"))
    return false;
  auto start = op->getAttrOfType<IntegerAttr>("start");
  auto bound = op->getAttrOfType<IntegerAttr>("end");
  return start && bound && start.getInt() == 0 && bound.getInt() == end;
}
inline bool axis(Operation *op, int64_t wanted) {
  auto attr = op ? op->getAttrOfType<IntegerAttr>("axis") : IntegerAttr{};
  return attr && attr.getInt() == wanted;
}
inline std::optional<int64_t> bound(Value value, int64_t extent,
                                    int64_t expansionAxis) {
  Operation *broadcast = def(value);
  if (!named(broadcast, "tt.broadcast") || broadcast->getNumOperands() != 1)
    return std::nullopt;
  auto cmp = dyn_cast_or_null<arith::CmpIOp>(def(broadcast->getOperand(0)));
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return std::nullopt;
  Operation *expanded = def(cmp.getLhs()), *splat = def(cmp.getRhs());
  if (!named(expanded, "tt.expand_dims") || expanded->getNumOperands() != 1 ||
      !axis(expanded, expansionAxis) ||
      !range(expanded->getOperand(0), extent) || !named(splat, "tt.splat") ||
      splat->getNumOperands() != 1)
    return std::nullopt;
  auto literal = uniformInteger(splat->getOperand(0));
  if (!literal || *literal <= 0 || *literal > extent)
    return std::nullopt;
  return literal;
}
inline bool baseIs(Value pointer, Value base) {
  Operation *addptr = def(pointer);
  return named(addptr, "tt.addptr") && addptr->getNumOperands() == 2 &&
         peelViews(addptr->getOperand(0)) == base;
}
// Prove the original row/column broadcast, not only tensor shape or op count.
inline std::optional<Value> rectangularAxisRange(Value plane, int64_t rows,
                                                 int64_t cols,
                                                 int64_t expansionAxis) {
  auto planeType = dyn_cast<RankedTensorType>(plane.getType());
  Operation *broadcast = def(plane);
  if (!planeType || planeType.getRank() != 2 ||
      planeType.getDimSize(0) != rows || planeType.getDimSize(1) != cols ||
      planeType.getEncoding() || !named(broadcast, "tt.broadcast") ||
      broadcast->getNumOperands() != 1)
    return std::nullopt;
  Operation *expanded = def(broadcast->getOperand(0));
  if (!named(expanded, "tt.expand_dims") || expanded->getNumOperands() != 1 ||
      !axis(expanded, expansionAxis) ||
      !range(expanded->getOperand(0), expansionAxis == 1 ? rows : cols))
    return std::nullopt;
  return expanded->getOperand(0);
}
inline bool admittedCalibratedPayloadTile(Value value, int64_t rows,
                                          int64_t cols) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 2 && type.getDimSize(0) == rows &&
         type.getDimSize(1) == cols &&
         (type.getElementType().isF32() ||
          type.getElementType().isSignlessInteger(32) ||
          type.getElementType().isF16() || type.getElementType().isBF16() ||
          type.getElementType().isSignlessInteger(8) ||
          type.getElementType().isSignlessInteger(64));
}
inline std::optional<int64_t> tensorBytes(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return int64_t(
        0); // pointers/scalars are tracked as values, not UB tensors.
  Type elem = type.getElementType();
  if (!type.hasStaticShape())
    return std::nullopt;
  if (!elem.isIntOrFloat())
    return int64_t(0);
  uint64_t n = type.getNumElements(), bits = elem.getIntOrFloatBitWidth();
  if (bits && n > (uint64_t(std::numeric_limits<int64_t>::max()) - 7) / bits)
    return std::nullopt;
  return int64_t((n * bits + 7) / 8);
}

/// Narrow first domain: a loop-free normal three-pointer indexed Store,
/// exact f32/i32/f16/bf16/signless-i8/signless-i64 R16C256 payload, i32 row
/// indices, natural row pitch 256 and literal full/tail mask. It deliberately
/// does not claim loaded addresses are bounded or unique: those independent
/// semantic/sampling qualifications belong to the price gate.
inline llvm::Expected<TargetBundle> prepareKnownNormalStoreBundle(
    Operation *target, const StagePartition &original,
    llvm::ArrayRef<StageImplementation> implementationByOriginalStage,
    const StagePartitionerOptions &originalAnalysisOptions,
    const StageMemoryPatternAnalysis *memoryPatterns = nullptr) {
  if (!named(target, "tt.store") || target->getNumOperands() != 3 ||
      target->getNumResults() || !original.operationOwnershipComplete ||
      original.stages.empty() ||
      implementationByOriginalStage.size() != original.stages.size())
    return reject("unsupported_target_or_incomplete_partition");
  Block *block = target->getBlock();
  Operation *function = block ? block->getParentOp() : nullptr;
  if (!named(function, "tt.func") || function->getNumRegions() != 1 ||
      !function->getRegion(0).hasOneBlock() || block->getNumArguments() != 3)
    return reject("requires_one_entry_block_and_three_data_arguments");
  auto callable = dyn_cast<FunctionOpInterface>(function);
  if (!callable)
    return reject("entry_function_argument_attributes_unavailable");
  for (unsigned i = 0; i < 3; ++i) {
    auto pointer =
        dyn_cast<triton::PointerType>(block->getArgument(i).getType());
    if (!pointer || pointer.getAddressSpace() != 1)
      return reject("all_three_original_data_arguments_must_be_global");
    // This calibrated normal lowering had no pointer divisibility/contiguity
    // argument hints. Unknown or different hints require separate
    // qualification.
    if (!callable.getArgAttrs(i).empty())
      return reject("unqualified_entry_argument_metadata");
  }
  auto typedStore = dyn_cast<triton::StoreOp>(target);
  if (!typedStore || typedStore.getCache() != triton::CacheModifier::NONE ||
      typedStore.getEvict() != triton::EvictionPolicy::NORMAL ||
      !typedStore.getBoundaryCheck().empty())
    return reject("unqualified_Store_cache_eviction_or_boundary_policy");
  Type leafType = typedStore.getPtr().getType();
  if (auto tensor = dyn_cast<RankedTensorType>(leafType))
    leafType = tensor.getElementType();
  auto leafPointer = dyn_cast<triton::PointerType>(leafType);
  auto basePointer =
      dyn_cast<triton::PointerType>(block->getArgument(0).getType());
  if (!leafPointer || leafPointer.getAddressSpace() != 1 || !basePointer ||
      basePointer.getAddressSpace() != 1)
    return reject("Store_target_must_be_global_address_space1");
  StageImplementation uniform = implementationByOriginalStage.front();
  for (const StageImplementation &impl : implementationByOriginalStage)
    if (!impl.isValid() || impl.localScope || impl.aivSubblocks != 1 ||
        impl.mode != uniform.mode ||
        impl.superblockFactor != uniform.superblockFactor)
      return reject("whole_original_route_not_uniform_mode_and_factor");

  llvm::DenseSet<Operation *> owned;
  for (const LogicalStage &stage : original.stages) {
    // Partial memory may be address-dependent without containing a loop.
    const bool targetPartialMemory =
        stage.scheduleKind == StageScheduleKind::PartiallyDependent &&
        stage.costModelKind ==
            StageCostModelKind::PartialContinuousTileMemory &&
        stage.operations.size() == 1 && stage.operations.front() == target &&
        !stage.features.hasLoop && !stage.features.hasLoopCarriedDataDependency;
    if ((stage.scheduleKind != StageScheduleKind::StraightLine &&
         !targetPartialMemory) ||
        stage.iterationCount != 1 || !stage.enclosingLoopStages.empty())
      return reject("region_or_recurrence_schedule_unsupported");
    for (Operation *op : stage.operations)
      if (!op || op->getBlock() != block || op->getNumRegions() ||
          op->getNumSuccessors() || !owned.insert(op).second)
        return reject("duplicate_or_cross_region_ownership");
  }
  for (Operation *op : owned) {
    for (Type type : op->getOperandTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        if (tensor.getEncoding())
          return reject("unqualified_ranked_tensor_layout_encoding");
    for (Type type : op->getResultTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        if (tensor.getEncoding())
          return reject("unqualified_ranked_tensor_layout_encoding");
  }
  if (!owned.contains(target) ||
      int64_t(owned.size()) != original.modeledOperationCount)
    return reject("modeled_count_or_target_owner_mismatch");
  SmallVector<Operation *> loads;
  for (Operation &operation : *block) {
    Operation *op = &operation;
    if (named(op, "tt.return")) {
      if (op->getNumOperands())
        return reject("return_values_unsupported");
      continue;
    }
    if (!owned.contains(op))
      return reject("entry_operation_without_unique_owner");
    if (op == target)
      continue;
    if (named(op, "tt.load")) {
      if (op->getNumResults() != 1 || op->getNumOperands() != 1)
        return reject("masked_or_nonstandard_preload_unsupported");
      auto typedLoad = dyn_cast<triton::LoadOp>(op);
      if (!typedLoad || typedLoad.getCache() != triton::CacheModifier::NONE ||
          typedLoad.getEvict() != triton::EvictionPolicy::NORMAL ||
          !typedLoad.getBoundaryCheck().empty())
        return reject("unqualified_preload_cache_eviction_or_boundary");
      if (auto attr = op->getAttrOfType<BoolAttr>("isVolatile"))
        if (attr.getValue())
          return reject("volatile_preload_unsupported");
      loads.push_back(op);
    } else if (!pure(op)) {
      return reject("unsupported_operation_or_side_effect");
    }
  }
  if (loads.size() != 2 ||
      !admittedCalibratedPayloadTile(target->getOperand(1), 16, 256))
    return reject(
        "requires_two_preloads_and_explicit_calibrated_payload_R16C256");
  auto *valuesLoad = def(target->getOperand(1));
  if (!named(valuesLoad, "tt.load") ||
      !baseIs(valuesLoad->getOperand(0), block->getArgument(2)))
    return reject("stored_values_not_direct_values_preload");
  Operation *targetPtr = def(target->getOperand(0));
  if (!baseIs(target->getOperand(0), block->getArgument(0)))
    return reject("target_pointer_base_not_entry_argument0");
  Operation *sum = def(targetPtr->getOperand(1));
  if (!named(sum, "arith.addi") || sum->getNumOperands() != 2)
    return reject("target_offset_not_row_plus_column");
  Operation *mul = def(sum->getOperand(0));
  if (!named(mul, "arith.muli") || mul->getNumOperands() != 2 ||
      uniformInteger(mul->getOperand(1)) != std::optional<int64_t>(256))
    return reject("target_row_pitch_must_be_256_not_Load_32768");
  Operation *indexLoad = def(peelViews(mul->getOperand(0)));
  auto indexType =
      indexLoad && indexLoad->getNumResults() == 1
          ? dyn_cast<RankedTensorType>(indexLoad->getResult(0).getType())
          : RankedTensorType{};
  if (!named(indexLoad, "tt.load") || indexLoad == valuesLoad || !indexType ||
      indexType.getRank() != 1 || indexType.getDimSize(0) != 16 ||
      !indexType.getElementType().isInteger(32) ||
      !baseIs(indexLoad->getOperand(0), block->getArgument(1)) ||
      !range(peelViews(sum->getOperand(1)), 256))
    return reject("loaded_i32_index_or_column_range_not_proven");
  // Independent preloads belong to aux pricing, but their normal-lowering
  // contract still controls the calibrated UB allocation/consumer context.
  Operation *valuesPointer = def(valuesLoad->getOperand(0));
  Operation *valuesOffset = def(valuesPointer->getOperand(1));
  if (!named(valuesOffset, "arith.addi") || valuesOffset->getNumOperands() != 2)
    return reject("Values_preload_requires_dense_row_plus_column_offset");
  Operation *valuesRow = def(valuesOffset->getOperand(0));
  if (!named(valuesRow, "arith.muli") || valuesRow->getNumOperands() != 2 ||
      uniformInteger(valuesRow->getOperand(1)) != std::optional<int64_t>(256))
    return reject("Values_preload_requires_original_dense_pitch256");
  auto rowRange = rectangularAxisRange(valuesRow->getOperand(0), 16, 256, 1);
  auto columnRange =
      rectangularAxisRange(valuesOffset->getOperand(1), 16, 256, 0);
  Operation *indexPointer = def(indexLoad->getOperand(0));
  if (!rowRange || !columnRange ||
      valuesOffset->getOperand(1) != sum->getOperand(1) ||
      indexPointer->getOperand(1) != *rowRange)
    return reject("Index_preload_must_be_same_contiguous_original_row_range");
  Operation *indexBroadcast = def(mul->getOperand(0));
  Operation *indexExpanded =
      indexBroadcast && indexBroadcast->getNumOperands() == 1
          ? def(indexBroadcast->getOperand(0))
          : nullptr;
  if (!named(indexBroadcast, "tt.broadcast") ||
      !named(indexExpanded, "tt.expand_dims") || !axis(indexExpanded, 1) ||
      indexExpanded->getNumOperands() != 1 ||
      indexExpanded->getOperand(0) != indexLoad->getResult(0))
    return reject(
        "Store_row_selector_must_use_original_row_axis_Index_broadcast");

  Operation *mask = def(target->getOperand(2));
  if (!named(mask, "arith.andi") || mask->getNumOperands() != 2)
    return reject("mask_not_literal_rectangle");
  auto h = bound(mask->getOperand(0), 16, 1);
  auto k = bound(mask->getOperand(1), 256, 0);
  if (!h || !k || !((*h == 16 && *k == 256) || (*h == 12 && *k == 192)))
    return reject("outside_first_two_literal_mask_corners");

  // Inverse closure stops at all memory operations and block arguments.
  llvm::DenseSet<Operation *> ancestors;
  std::function<llvm::Error(Value)> visit = [&](Value value) -> llvm::Error {
    Operation *op = def(value);
    if (!op)
      return value.getParentBlock() == block ? llvm::Error::success()
                                             : reject("cross_region_argument");
    if (op->getBlock() != block)
      return reject("cross_region_definition");
    if (named(op, "tt.load"))
      return llvm::Error::success();
    if (!pure(op))
      return reject("unsupported_inverse_closure_operation");
    if (!ancestors.insert(op).second)
      return llvm::Error::success();
    for (Value operand : op->getOperands())
      if (auto error = visit(operand))
        return error;
    return llvm::Error::success();
  };
  if (auto error = visit(target->getOperand(0)))
    return std::move(error);
  if (auto error = visit(target->getOperand(2)))
    return std::move(error);

  // A pure op is covered only when every transitive user terminates at target.
  // Shared make_range/view/constants remain auxiliary even if target uses them.
  llvm::DenseMap<Operation *, bool> exclusiveMemo;
  llvm::DenseSet<Operation *> active;
  bool unsupportedUser = false;
  std::function<bool(Operation *)> exclusive = [&](Operation *op) {
    if (op == target)
      return true;
    if (auto it = exclusiveMemo.find(op); it != exclusiveMemo.end())
      return it->second;
    if (!active.insert(op).second) {
      unsupportedUser = true;
      return false;
    }
    bool yes = !op->use_empty();
    for (Operation *user : op->getUsers()) {
      if (user->getBlock() != block || user->getNumRegions()) {
        unsupportedUser = true;
        yes = false;
      } else if (user != target && !pure(user)) {
        // Other preloads are legitimate shared consumers, not bundled work.
        if (!named(user, "tt.load"))
          unsupportedUser = true;
        yes = false;
      } else if (!exclusive(user)) {
        yes = false;
      }
    }
    active.erase(op);
    exclusiveMemo[op] = yes;
    return yes;
  };
  llvm::DenseSet<Operation *> covered;
  covered.insert(target);
  for (Operation *op : ancestors)
    if (exclusive(op))
      covered.insert(op);
  if (unsupportedUser)
    return reject("unsupported_or_cross_region_forward_consumer");

  // A measured bundle includes its dedicated row-address and literal-mask
  // work. If an extra consumer shares that work, retaining it as auxiliary
  // while adding the whole bundle would double charge it. Fall back instead.
  SmallVector<Operation *> requiredAddress{targetPtr,
                                           def(targetPtr->getOperand(0)), sum,
                                           mul, def(mul->getOperand(1))};
  Value indexProjection = mul->getOperand(0);
  while (def(indexProjection) != indexLoad) {
    Operation *view = def(indexProjection);
    if (!(named(view, "tt.broadcast") || named(view, "tt.expand_dims")) ||
        view->getNumOperands() != 1)
      return reject("unsupported_index_projection");
    requiredAddress.push_back(view);
    indexProjection = view->getOperand(0);
  }
  for (Operation *op : requiredAddress)
    if (!covered.contains(op))
      return reject("dedicated_target_address_has_shared_consumer");
  llvm::DenseSet<Operation *> dedicatedMask;
  std::function<void(Value)> collectMask = [&](Value value) {
    Operation *op = def(value);
    if (!op || named(op, "tt.make_range") ||
        (named(op, "tt.expand_dims") &&
         range(op->getOperand(0), axis(op, 1) ? 16 : 256)))
      return;
    if (!dedicatedMask.insert(op).second)
      return;
    for (Value input : op->getOperands())
      collectMask(input);
  };
  collectMask(target->getOperand(2));
  for (Operation *op : dedicatedMask)
    if (!covered.contains(op))
      return reject("dedicated_target_mask_has_shared_consumer");

  TargetBundle result;
  result.target = target;
  result.uniformImplementation = uniform;
  result.rows = 16;
  result.columns = 256;
  result.knownRows = *h;
  result.knownColumns = *k;
  result.rowPitchElements = 256;
  result.payloadIsI32 = cast<RankedTensorType>(target->getOperand(1).getType())
                            .getElementType()
                            .isSignlessInteger(32);
  result.payloadIsF16 = cast<RankedTensorType>(target->getOperand(1).getType())
                            .getElementType()
                            .isF16();
  result.payloadIsBF16 = cast<RankedTensorType>(target->getOperand(1).getType())
                             .getElementType()
                             .isBF16();
  result.payloadIsU8 = cast<RankedTensorType>(target->getOperand(1).getType())
                           .getElementType()
                           .isSignlessInteger(8);
  result.payloadIsI64 = cast<RankedTensorType>(target->getOperand(1).getType())
                            .getElementType()
                            .isSignlessInteger(64);
  result.requiresWholeSimtPressureReanalysis = uniform.mode == StageMode::SIMT;
  llvm::DenseSet<Value> inputs;
  for (Operation &op : *block) {
    if (!covered.contains(&op))
      continue;
    result.coveredOperations.push_back(&op);
    for (Value value : op.getOperands())
      if (!covered.contains(def(value)) && inputs.insert(value).second)
        result.bundleInputs.push_back(value);
  }

  // Keep the exact remaining operations; do not retain stale per-stage prices.
  StagePartition &aux = result.auxiliaryPartition;
  aux.operationOwnershipComplete = true;
  for (const LogicalStage &stage : original.stages) {
    LogicalStage remaining = stage;
    remaining.operations.clear();
    for (Operation *op : stage.operations)
      if (!covered.contains(op))
        remaining.operations.push_back(op);
    if (remaining.operations.empty()) {
      if (stage.workload.paysKernelSetup)
        return reject("cannot_drop_kernel_setup_owner");
      continue;
    }
    remaining.wholeSimtRegisterPressure.reset();
    remaining.simtUbWorkingSetBytesPerProgram = 0;
    remaining.simtFactorsRejectedByUb.clear();
    aux.modeledOperationCount += remaining.operations.size();
    aux.stages.push_back(std::move(remaining));
  }
  if (aux.modeledOperationCount + int64_t(covered.size()) !=
      original.modeledOperationCount)
    return reject("covered_auxiliary_union_not_exact");
  // A Store bundle has no SSA result. An auxiliary use of a covered value would
  // invalidate its claimed exclusivity and must not silently get re-costed.
  for (const LogicalStage &stage : aux.stages)
    for (Operation *op : stage.operations)
      for (Value value : op->getOperands())
        if (covered.contains(def(value)))
          return reject("covered_value_escapes_to_auxiliary");

  if (auto error = StageWorkloadAnalysis().analyze(
          aux, originalAnalysisOptions.logicalProgramCountHint, memoryPatterns))
    return std::move(error);
  if (auto error = StageFeatureAnalysis().analyze(aux, memoryPatterns))
    return std::move(error);
  if (auto error = StageKindClassifier().analyze(
          aux, originalAnalysisOptions.tinyDotFlopsMax))
    return std::move(error);

  // Filtered workload invalidates cached legality. Re-run structural legality
  // with the original caller's exact analysis options; do not use device
  // literals.
  if (auto error = StageModeLegalityAnalysis().analyze(
          aux, originalAnalysisOptions.maximumSuperblockFactor,
          originalAnalysisOptions.scopeSuperblockMaterializable,
          originalAnalysisOptions.maximumScopeSuperblockFactor))
    return std::move(error);
  int64_t originalUsableUb = 0;
  for (const LogicalStage &stage : original.stages) {
    if (stage.simtUsableUbBytes <= 0)
      continue;
    if (originalUsableUb && originalUsableUb != stage.simtUsableUbBytes)
      return reject("inconsistent_original_SIMT_UB_capacity");
    originalUsableUb = stage.simtUsableUbBytes;
  }
  if (originalUsableUb > 0) {
    if (auto error = StageResourceLegalityAnalysis().applySimtUbCapacity(
            aux, originalUsableUb))
      return std::move(error);
  } else {
    // Missing SIMT capacity is never converted into an unqualified SIMT route.
    for (LogicalStage &stage : aux.stages) {
      stage.simtLegal = false;
      stage.legalSimtFactors.clear();
      stage.localSimtFactors.clear();
      stage.localSimtMaterializable = false;
      stage.localSuperblockMaterializable = false;
    }
  }

  // Rebuild straight-line SSA boundaries after filtering and classification.
  // Original side-effect ordering into the terminal Store is represented by
  // bundleInputs, not by pretending the deleted Stage still owns the Store.
  llvm::DenseMap<Operation *, unsigned> owner;
  for (unsigned i = 0; i < aux.stages.size(); ++i)
    for (Operation *op : aux.stages[i].operations)
      owner[op] = i;
  for (LogicalStage &stage : aux.stages) {
    llvm::DenseSet<Operation *> local;
    llvm::DenseSet<Value> in, out;
    for (Operation *op : stage.operations)
      local.insert(op);
    stage.liveIns.clear();
    stage.liveOuts.clear();
    stage.liveInBytes = 0;
    stage.liveOutBytes = 0;
    for (Operation *op : stage.operations) {
      for (Value value : op->getOperands())
        if (!local.contains(def(value)) && in.insert(value).second)
          stage.liveIns.push_back(value);
      for (Value value : op->getResults())
        for (Operation *user : value.getUsers())
          if (!local.contains(user) && out.insert(value).second)
            stage.liveOuts.push_back(value);
    }
    for (Value value : stage.liveIns) {
      auto bytes = tensorBytes(value);
      if (!bytes || stage.liveInBytes > INT64_MAX - *bytes)
        return reject("unsupported_live_in_tensor_bytes");
      stage.liveInBytes += *bytes;
    }
    for (Value value : stage.liveOuts) {
      auto bytes = tensorBytes(value);
      if (!bytes || stage.liveOutBytes > INT64_MAX - *bytes)
        return reject("unsupported_live_out_tensor_bytes");
      stage.liveOutBytes += *bytes;
    }
  }
  std::map<std::pair<unsigned, unsigned>, llvm::DenseSet<Value>> pairValues;
  for (unsigned j = 0; j < aux.stages.size(); ++j)
    for (Operation *op : aux.stages[j].operations)
      for (Value value : op->getOperands()) {
        auto found = owner.find(def(value));
        if (found != owner.end() && found->second != j)
          pairValues[{found->second, j}].insert(value);
      }
  for (const auto &entry : pairValues) {
    StageDependency edge;
    edge.sourceStage = aux.stages[entry.first.first].id;
    edge.targetStage = aux.stages[entry.first.second].id;
    edge.kind = StageDependencyKind::IntraIterationData;
    edge.iterationDistance = 0;
    edge.carriedValueName = "inter_stage_ssa";
    for (Value value : entry.second) {
      auto bytes = tensorBytes(value);
      if (!bytes || edge.carriedValueBytes > INT64_MAX - *bytes)
        return reject("unsupported_dependency_tensor_bytes");
      edge.carriedValueBytes += *bytes;
    }
    aux.dependencies.push_back(std::move(edge));
  }
  std::set<unsigned> inputOwners;
  for (Value value : result.bundleInputs) {
    auto found = owner.find(def(value));
    if (found != owner.end())
      inputOwners.insert(found->second);
  }
  for (unsigned i : inputOwners)
    result.bundleInputAuxiliaryStageIds.push_back(aux.stages[i].id);
  return result;
}
} // namespace mlir::ascend::specialized_segments_ownership
#endif
