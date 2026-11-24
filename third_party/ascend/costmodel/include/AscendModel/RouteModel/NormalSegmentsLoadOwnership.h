// Normal full4x16 Segments Load: complete producer-preserving ownership graph.
#ifndef ASCENDMODEL_NORMAL_SEGMENTS_LOAD_OWNERSHIP_H
#define ASCENDMODEL_NORMAL_SEGMENTS_LOAD_OWNERSHIP_H
#include "AscendModel/Analysis/SimtRegisterPressureAnalysis.h"
#include "AscendModel/RouteModel/SegmentsStoreOwnershipAdapter.h"
namespace mlir::ascend::normal_segments_load_ownership {
using namespace specialized_segments_ownership;
struct LoadBundle {
  Operation *target = nullptr;
  Operation *sink = nullptr;
  StagePartition completePartition;
  std::vector<Operation *> coveredOperations;
  std::vector<unsigned> originalStageSlots;
  unsigned targetNode = 0;
  unsigned payloadBytes = 0;
  bool payloadIsF16 = false;
  bool payloadIsBF16 = false;
  bool payloadIsI32 = false;
  bool payloadIsU8 = false;
  bool payloadIsI64 = false;
};
inline bool f32Tile(Value v) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.getRank() == 2 && t.getDimSize(0) == 4 &&
         t.getDimSize(1) == 16 && t.getElementType().isF32() &&
         !t.getEncoding();
}
inline bool calibratedTile(Value v) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.getRank() == 2 && t.getDimSize(0) == 4 &&
         t.getDimSize(1) == 16 &&
         (t.getElementType().isF32() || t.getElementType().isF16() ||
          t.getElementType().isBF16() ||
          t.getElementType().isSignlessInteger(32) ||
          t.getElementType().isSignlessInteger(8) ||
          t.getElementType().isSignlessInteger(64)) &&
         !t.getEncoding();
}
inline bool floatZero(Value v) {
  auto op = dyn_cast_or_null<arith::ConstantOp>(def(v));
  if (!op)
    return false;
  auto a = dyn_cast<DenseFPElementsAttr>(op.getValue());
  return a && a.isSplat() && a.getSplatValue<llvm::APFloat>().isZero();
}
inline bool calibratedZero(Value v) {
  if (floatZero(v))
    return true;
  auto t = dyn_cast<RankedTensorType>(v.getType());
  auto op = dyn_cast_or_null<arith::ConstantOp>(def(v));
  if (!t ||
      (!t.getElementType().isSignlessInteger(32) &&
       !t.getElementType().isSignlessInteger(8) &&
       !t.getElementType().isSignlessInteger(64)) ||
      !op)
    return false;
  auto a = dyn_cast<DenseIntElementsAttr>(op.getValue());
  return a && a.isSplat() && a.getSplatValue<llvm::APInt>().isZero();
}
inline bool defaultLoad(Operation *op) {
  auto x = dyn_cast_or_null<triton::LoadOp>(op);
  return x && x.getCache() == triton::CacheModifier::NONE &&
         x.getEvict() == triton::EvictionPolicy::NORMAL &&
         x.getBoundaryCheck().empty() && !x.getIsVolatile();
}
inline llvm::Expected<LoadBundle>
prepare(ModuleOp module, Operation *target, const StagePartition &original,
        llvm::ArrayRef<StageImplementation> implementations,
        const StagePartitionerOptions &options, int64_t actualWarps,
        int64_t actualUsableSimtUB,
        const StageMemoryPatternAnalysis *patterns) {
  if (!named(target, "tt.load") || target->getNumOperands() != 3 ||
      target->getNumResults() != 1 || !calibratedTile(target->getResult(0)) ||
      !defaultLoad(target) || !calibratedZero(target->getOperand(2)) ||
      !original.operationOwnershipComplete || original.stages.empty() ||
      implementations.size() != original.stages.size())
    return reject("Load_target_or_partition_outside_calibrated_normal_full");
  Block *block = target->getBlock();
  Operation *fn = block ? block->getParentOp() : nullptr;
  if (!named(fn, "tt.func") || fn->getNumRegions() != 1 ||
      !fn->getRegion(0).hasOneBlock() || block->getNumArguments() != 3)
    return reject("Load_requires_single_three_pointer_entry");
  auto callable = dyn_cast<FunctionOpInterface>(fn);
  if (!callable)
    return reject("Load_entry_argument_metadata_missing");
  for (unsigned i = 0; i < 3; ++i) {
    auto p = dyn_cast<triton::PointerType>(block->getArgument(i).getType());
    if (!p || p.getAddressSpace() != 1 || !callable.getArgAttrs(i).empty())
      return reject("Load_unqualified_argument_space_or_metadata");
  }
  auto xType = dyn_cast<triton::PointerType>(block->getArgument(0).getType());
  auto iType = dyn_cast<triton::PointerType>(block->getArgument(1).getType());
  auto oType = dyn_cast<triton::PointerType>(block->getArgument(2).getType());
  Type payloadType =
      cast<RankedTensorType>(target->getResult(0).getType()).getElementType();
  if (xType.getPointeeType() != payloadType ||
      !iType.getPointeeType().isSignlessInteger(32) ||
      oType.getPointeeType() != payloadType)
    return reject("Load_exact_admitted_F32_F16_I32_or_U8_payload_IndexI32_same_"
                  "payload_entry_required");
  llvm::DenseMap<Operation *, unsigned> oldOwner;
  for (unsigned i = 0; i < original.stages.size(); ++i) {
    auto &s = original.stages[i];
    auto &impl = implementations[i];
    if (!impl.isValid() || impl.localScope || impl.mode != StageMode::SIMD ||
        impl.superblockFactor != 1 || impl.aivSubblocks != 1 ||
        s.iterationCount != 1 || !s.enclosingLoopStages.empty() ||
        s.features.hasLoop || s.features.hasLoopCarriedDataDependency)
      return reject("Load_requires_uniform_loopfree_SIMD_F1");
    bool partial =
        s.scheduleKind == StageScheduleKind::PartiallyDependent &&
        s.costModelKind == StageCostModelKind::PartialContinuousTileMemory &&
        s.operations.size() == 1 && s.operations.front() == target;
    if (s.scheduleKind != StageScheduleKind::StraightLine && !partial)
      return reject("Load_unqualified_schedule");
    for (Operation *op : s.operations) {
      if (!op || op->getBlock() != block || op->getNumRegions() ||
          op->getNumSuccessors() || oldOwner.count(op))
        return reject("Load_duplicate_or_nonlocal_owner");
      oldOwner[op] = i;
      for (Type ty : op->getOperandTypes())
        if (auto t = dyn_cast<RankedTensorType>(ty); t && t.getEncoding())
          return reject("Load_unknown_tensor_encoding");
      for (Type ty : op->getResultTypes())
        if (auto t = dyn_cast<RankedTensorType>(ty); t && t.getEncoding())
          return reject("Load_unknown_tensor_encoding");
    }
  }
  if (int64_t(oldOwner.size()) != original.modeledOperationCount ||
      !oldOwner.count(target))
    return reject("Load_incomplete_owner_map");
  Operation *sink = nullptr;
  SmallVector<Operation *> loads;
  for (Operation &op : *block) {
    if (named(&op, "tt.return")) {
      if (op.getNumOperands())
        return reject("Load_return_values_unsupported");
      continue;
    }
    if (!oldOwner.count(&op))
      return reject("Load_unowned_entry_operation");
    if (named(&op, "tt.load")) {
      if (!defaultLoad(&op))
        return reject("Load_nondefault_load_policy");
      loads.push_back(&op);
    } else if (named(&op, "tt.store")) {
      if (sink)
        return reject("Load_multiple_output_sinks");
      sink = &op;
    } else if (!pure(&op))
      return reject("Load_unknown_effect_or_operation");
  }
  if (loads.size() != 2 || !sink || sink->getNumOperands() != 2 ||
      sink->getOperand(1) != target->getResult(0) ||
      !target->getResult(0).hasOneUse())
    return reject("Load_requires_unique_direct_unmasked_copy_consumer");
  auto ts = dyn_cast<triton::StoreOp>(sink);
  if (!ts || ts.getCache() != triton::CacheModifier::NONE ||
      ts.getEvict() != triton::EvictionPolicy::NORMAL ||
      !ts.getBoundaryCheck().empty())
    return reject("Load_sink_policy_differs_from_matched_consumer");
  // Match the complete target address DAG:
  // broadcast(X+expand(Index)*32768)+column.
  Operation *targetPtr = def(target->getOperand(0));
  if (!named(targetPtr, "tt.addptr") || targetPtr->getNumOperands() != 2)
    return reject("Load_target_address_not_addptr");
  Operation *rowBroadcast = def(targetPtr->getOperand(0));
  if (!named(rowBroadcast, "tt.broadcast") ||
      rowBroadcast->getNumOperands() != 1)
    return reject("Load_target_row_pointer_broadcast_missing");
  Operation *rowPtr = def(rowBroadcast->getOperand(0));
  if (!named(rowPtr, "tt.addptr") || rowPtr->getNumOperands() != 2 ||
      peelViews(rowPtr->getOperand(0)) != block->getArgument(0))
    return reject("Load_target_source_base_not_original_X");
  Operation *mul = def(rowPtr->getOperand(1));
  if (!named(mul, "arith.muli") || mul->getNumOperands() != 2 ||
      uniformInteger(mul->getOperand(1)) != std::optional<int64_t>(32768))
    return reject("Load_rowpitch_must_be_32768");
  Operation *indexExpanded = def(mul->getOperand(0));
  if (!named(indexExpanded, "tt.expand_dims") || !axis(indexExpanded, 1) ||
      indexExpanded->getNumOperands() != 1)
    return reject("Load_Index_axis_projection_not_original");
  Operation *indexLoad = def(indexExpanded->getOperand(0));
  if (!named(indexLoad, "tt.load") || indexLoad == target ||
      indexLoad->getNumOperands() != 3 || indexLoad->getNumResults() != 1 ||
      !defaultLoad(indexLoad) ||
      !baseIs(indexLoad->getOperand(0), block->getArgument(1)) ||
      uniformInteger(indexLoad->getOperand(2)) != std::optional<int64_t>(0))
    return reject("Load_original_masked_Index_preload_missing");
  auto idxTy = dyn_cast<RankedTensorType>(indexLoad->getResult(0).getType());
  Operation *indexPtr = def(indexLoad->getOperand(0));
  Value rowRange = indexPtr->getOperand(1);
  auto indexMask =
      dyn_cast_or_null<arith::CmpIOp>(def(indexLoad->getOperand(1)));
  if (!idxTy || idxTy.getRank() != 1 || idxTy.getDimSize(0) != 4 ||
      !idxTy.getElementType().isSignlessInteger(32) || !range(rowRange, 4) ||
      !indexMask || indexMask.getPredicate() != arith::CmpIPredicate::slt ||
      indexMask.getLhs() != rowRange ||
      uniformInteger(peelViews(indexMask.getRhs())) !=
          std::optional<int64_t>(4))
    return reject("Load_Index_preload_contiguity_or_fullrow_mask_unknown");
  auto colRange = rectangularAxisRange(targetPtr->getOperand(1), 4, 16, 0);
  if (!colRange)
    return reject("Load_column_stride_not_original_unit_ramp");
  Operation *mask = def(target->getOperand(1));
  if (!named(mask, "arith.andi") || mask->getNumOperands() != 2 ||
      bound(mask->getOperand(0), 4, 1) != std::optional<int64_t>(4) ||
      bound(mask->getOperand(1), 16, 0) != std::optional<int64_t>(16))
    return reject("Load_target_requires_full_literal_rectangle");
  // Dense output with the same row/column axes is part of the matched consumer.
  if (!baseIs(sink->getOperand(0), block->getArgument(2)))
    return reject("Load_output_base_not_original_Output");
  Operation *outPtr = def(sink->getOperand(0));
  Operation *outSum = def(outPtr->getOperand(1));
  if (!named(outSum, "arith.addi") || outSum->getNumOperands() != 2 ||
      outSum->getOperand(1) != targetPtr->getOperand(1))
    return reject("Load_output_column_axis_differs");
  Operation *outBroadcast = def(outSum->getOperand(0));
  if (!named(outBroadcast, "tt.broadcast") ||
      outBroadcast->getNumOperands() != 1)
    return reject("Load_output_row_broadcast_missing");
  Operation *outMul = def(outBroadcast->getOperand(0));
  if (!named(outMul, "arith.muli") || outMul->getNumOperands() != 2 ||
      uniformInteger(outMul->getOperand(1)) != std::optional<int64_t>(16))
    return reject("Load_output_must_be_dense_pitch16");
  Operation *outExpanded = def(outMul->getOperand(0));
  if (!named(outExpanded, "tt.expand_dims") || !axis(outExpanded, 1) ||
      outExpanded->getNumOperands() != 1 ||
      outExpanded->getOperand(0) != rowRange)
    return reject("Load_output_must_share_original_row_range");
  llvm::DenseSet<Operation *> ancestors, covered;
  std::function<llvm::Error(Value)> visit = [&](Value value) -> llvm::Error {
    Operation *op = def(value);
    if (!op)
      return value.getParentBlock() == block
                 ? llvm::Error::success()
                 : reject("Load_cross_region_argument");
    if (op->getBlock() != block)
      return reject("Load_cross_region_definition");
    if (named(op, "tt.load"))
      return op == indexLoad ? llvm::Error::success()
                             : reject("Load_unexpected_memory_ancestor");
    if (!pure(op))
      return reject("Load_unknown_inverse_operation");
    if (!ancestors.insert(op).second)
      return llvm::Error::success();
    for (Value x : op->getOperands())
      if (auto e = visit(x))
        return e;
    return llvm::Error::success();
  };
  for (Value v : target->getOperands())
    if (auto e = visit(v))
      return std::move(e);
  llvm::DenseMap<Operation *, bool> memo;
  llvm::DenseSet<Operation *> active;
  bool badUser = false;
  std::function<bool(Operation *)> exclusive = [&](Operation *op) {
    if (op == target)
      return true;
    if (auto p = memo.find(op); p != memo.end())
      return p->second;
    if (!active.insert(op).second) {
      badUser = true;
      return false;
    }
    bool yes = !op->use_empty();
    for (Operation *u : op->getUsers()) {
      if (u->getBlock() != block || u->getNumRegions()) {
        badUser = true;
        yes = false;
      } else if (u == indexLoad || u == sink)
        yes = false;
      else if (u != target && !pure(u)) {
        badUser = true;
        yes = false;
      } else if (!exclusive(u))
        yes = false;
    }
    active.erase(op);
    memo[op] = yes;
    return yes;
  };
  covered.insert(target);
  for (Operation *op : ancestors)
    if (exclusive(op))
      covered.insert(op);
  if (badUser)
    return reject("Load_unsupported_shared_forward_user");
  for (Operation *op :
       {targetPtr, rowBroadcast, rowPtr, def(rowPtr->getOperand(0)), mul,
        def(mul->getOperand(1)), indexExpanded, def(target->getOperand(2))})
    if (!covered.contains(op))
      return reject("Load_dedicated_address_or_zero_shared");
  std::function<bool(Value)> dedicatedMask = [&](Value v) {
    Operation *op = def(v);
    if (!op || named(op, "tt.make_range") ||
        (named(op, "tt.expand_dims") && op->getNumOperands() == 1 &&
         (range(op->getOperand(0), 4) || range(op->getOperand(0), 16))))
      return true;
    if (!covered.contains(op))
      return false;
    for (Value a : op->getOperands())
      if (!dedicatedMask(a))
        return false;
    return true;
  };
  if (!dedicatedMask(target->getOperand(1)))
    return reject("Load_dedicated_mask_shared");
  LoadBundle result;
  result.target = target;
  result.sink = sink;
  result.payloadIsF16 = payloadType.isF16();
  result.payloadIsBF16 = payloadType.isBF16();
  result.payloadIsI32 = payloadType.isSignlessInteger(32);
  result.payloadIsU8 = payloadType.isSignlessInteger(8);
  result.payloadIsI64 = payloadType.isSignlessInteger(64);
  result.payloadBytes = result.payloadIsI64                             ? 8
                        : result.payloadIsU8                            ? 1
                        : (result.payloadIsF16 || result.payloadIsBF16) ? 2
                                                                        : 4;
  const int64_t targetTensorBytes = 4 * 16 * result.payloadBytes;
  unsigned targetOwner = oldOwner.lookup(target);
  LogicalStage bundled = original.stages[targetOwner];
  bundled.id = "qualified_normal_Load_producer_bundle";
  bundled.operations.clear();
  bundled.wholeSimtRegisterPressure.reset();
  for (Operation &op : *block)
    if (covered.contains(&op)) {
      result.coveredOperations.push_back(&op);
      bundled.operations.push_back(&op);
    }
  for (unsigned i = 0; i < original.stages.size(); ++i) {
    LogicalStage s = original.stages[i];
    s.operations.clear();
    for (Operation *op : original.stages[i].operations)
      if (!covered.contains(op))
        s.operations.push_back(op);
    if (!s.operations.empty()) {
      s.wholeSimtRegisterPressure.reset();
      s.simtUbWorkingSetBytesPerProgram = 0;
      s.simtFactorsRejectedByUb.clear();
      result.completePartition.stages.push_back(std::move(s));
      result.originalStageSlots.push_back(i);
    }
    if (i == targetOwner) {
      result.targetNode = result.completePartition.stages.size();
      result.completePartition.stages.push_back(std::move(bundled));
      result.originalStageSlots.push_back(targetOwner);
    }
  }
  StagePartition &p = result.completePartition;
  p.operationOwnershipComplete = true;
  p.modeledOperationCount = 0;
  llvm::DenseMap<Operation *, unsigned> owner;
  for (unsigned i = 0; i < p.stages.size(); ++i)
    for (Operation *op : p.stages[i].operations) {
      if (owner.count(op))
        return reject("Load_compressed_duplicate_owner");
      owner[op] = i;
      ++p.modeledOperationCount;
    }
  if (p.modeledOperationCount != original.modeledOperationCount ||
      owner.size() != oldOwner.size())
    return reject("Load_compressed_union_not_exact");
  for (LogicalStage &s : p.stages) {
    llvm::DenseSet<Operation *> local;
    llvm::DenseSet<Value> in, out;
    for (Operation *op : s.operations)
      local.insert(op);
    s.liveIns.clear();
    s.liveOuts.clear();
    s.liveInBytes = s.liveOutBytes = 0;
    for (Operation *op : s.operations) {
      for (Value v : op->getOperands())
        if (!local.contains(def(v)) && in.insert(v).second)
          s.liveIns.push_back(v);
      for (Value v : op->getResults())
        for (Operation *u : v.getUsers())
          if (!local.contains(u) && out.insert(v).second)
            s.liveOuts.push_back(v);
    }
    for (Value v : s.liveIns) {
      auto b = tensorBytes(v);
      if (!b || s.liveInBytes > INT64_MAX - *b)
        return reject("Load_unknown_livein");
      s.liveInBytes += *b;
    }
    for (Value v : s.liveOuts) {
      auto b = tensorBytes(v);
      if (!b || s.liveOutBytes > INT64_MAX - *b)
        return reject("Load_unknown_liveout");
      s.liveOutBytes += *b;
    }
  }
  std::map<std::pair<unsigned, unsigned>, llvm::DenseSet<Value>> edges;
  for (unsigned j = 0; j < p.stages.size(); ++j)
    for (Operation *op : p.stages[j].operations)
      for (Value v : op->getOperands())
        if (auto f = owner.find(def(v)); f != owner.end() && f->second != j)
          edges[{f->second, j}].insert(v);
  if (auto e = StageWorkloadAnalysis().analyze(
          p, options.logicalProgramCountHint, patterns))
    return std::move(e);
  if (auto e = StageFeatureAnalysis().analyze(p, patterns))
    return std::move(e);
  if (auto e = StageKindClassifier().analyze(p, options.tinyDotFlopsMax))
    return std::move(e);
  for (auto &stage : p.stages)
    for (auto &pattern : stage.workload.addressPatterns)
      pattern.stageId = stage.id;
  if (auto e = StageModeLegalityAnalysis().analyze(
          p, options.maximumSuperblockFactor,
          options.scopeSuperblockMaterializable,
          options.maximumScopeSuperblockFactor))
    return std::move(e);
  if (auto e = SimtRegisterPressureAnalysis().analyze(module, p, actualWarps))
    return std::move(e);
  if (actualUsableSimtUB <= 0)
    return reject("Load_actual_profile_UB_missing");
  if (auto e = StageResourceLegalityAnalysis().applySimtUbCapacity(
          p, actualUsableSimtUB))
    return std::move(e);
  // Generate edge names only after Classifier has assigned final stage IDs.
  for (const auto &e : edges) {
    StageDependency d;
    d.sourceStage = p.stages[e.first.first].id;
    d.targetStage = p.stages[e.first.second].id;
    d.kind = StageDependencyKind::IntraIterationData;
    d.iterationDistance = 0;
    d.carriedValueIndex = 0;
    d.carriedValueName = "inter_stage_ssa";
    for (Value v : e.second) {
      auto b = tensorBytes(v);
      if (!b || d.carriedValueBytes > INT64_MAX - *b)
        return reject("Load_unknown_edge_bytes");
      d.carriedValueBytes += *b;
    }
    p.dependencies.push_back(std::move(d));
  }
  if (auto e = StagePartitionVerifier().verify(p))
    return std::move(e);
  if (p.stages[result.targetNode].workload.paysKernelSetup)
    return reject("Load_target_cannot_own_kernel_setup");
  // The actual loaded SSA tensor must be produced here and consumed by the
  // sink.
  auto sinkOwner = owner.lookup(sink);
  if (owner.lookup(target) != result.targetNode ||
      sinkOwner == result.targetNode ||
      p.stages[result.targetNode].liveOutBytes < targetTensorBytes)
    return reject("Load_producer_output_boundary_missing");
  bool resultEdge = false;
  for (const auto &e : p.dependencies)
    if (e.sourceStage == p.stages[result.targetNode].id &&
        e.targetStage == p.stages[sinkOwner].id &&
        e.carriedValueBytes >= targetTensorBytes)
      resultEdge = true;
  if (!resultEdge)
    return reject("Load_target_to_output_sink_dependency_missing");
  return result;
}
} // namespace mlir::ascend::normal_segments_load_ownership
#endif
