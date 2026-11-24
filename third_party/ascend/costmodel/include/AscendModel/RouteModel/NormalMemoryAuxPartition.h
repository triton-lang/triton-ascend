// Owned research audit: rebuild a filtered auxiliary partition without pricing
// subtraction.
#ifndef ASCENDMODEL_NORMAL_MEMORY_AUX_PARTITION_H
#define ASCENDMODEL_NORMAL_MEMORY_AUX_PARTITION_H
#include "AscendModel/RouteModel/NormalSegmentsLoadOwnership.h"
namespace mlir::ascend::normal_memory_aux {
using namespace specialized_segments_ownership;
inline llvm::Expected<StagePartition>
rebuild(ModuleOp module, const StagePartition &original,
        const llvm::DenseSet<Operation *> &covered,
        const StagePartitionerOptions &options, int64_t warps,
        int64_t usableUB) {
  StagePartition out;
  out.modeledOperationCount = 0;
  llvm::DenseMap<Operation *, unsigned> owner;
  for (const LogicalStage &source : original.stages) {
    LogicalStage s = source;
    s.operations.clear();
    s.wholeSimtRegisterPressure.reset();
    s.simtUbWorkingSetBytesPerProgram = 0;
    s.simtFactorsRejectedByUb.clear();
    for (Operation *op : source.operations)
      if (!covered.contains(op))
        s.operations.push_back(op);
    if (s.operations.empty())
      continue;
    unsigned slot = out.stages.size();
    for (Operation *op : s.operations) {
      if (owner.count(op))
        return reject("aux_duplicate_original_owner");
      owner[op] = slot;
    }
    out.modeledOperationCount += s.operations.size();
    out.stages.push_back(std::move(s));
  }
  if (out.stages.empty())
    return reject("aux_empty_no_setup_owner");
  out.operationOwnershipComplete = true;
  if (out.modeledOperationCount + int64_t(covered.size()) !=
      original.modeledOperationCount)
    return reject("aux_covered_union_not_exact");
  for (LogicalStage &s : out.stages) {
    llvm::DenseSet<Operation *> local;
    llvm::DenseSet<Value> in, live;
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
        for (Operation *user : v.getUsers())
          if (!local.contains(user) && live.insert(v).second)
            s.liveOuts.push_back(v);
    }
    for (Value v : s.liveIns) {
      auto b = tensorBytes(v);
      if (!b || s.liveInBytes > INT64_MAX - *b)
        return reject("aux_unknown_livein");
      s.liveInBytes += *b;
    }
    for (Value v : s.liveOuts) {
      auto b = tensorBytes(v);
      if (!b || s.liveOutBytes > INT64_MAX - *b)
        return reject("aux_unknown_liveout");
      s.liveOutBytes += *b;
    }
  }
  std::map<std::pair<unsigned, unsigned>, llvm::DenseSet<Value>> edges;
  for (unsigned j = 0; j < out.stages.size(); ++j)
    for (Operation *op : out.stages[j].operations)
      for (Value v : op->getOperands())
        if (auto f = owner.find(def(v)); f != owner.end() && f->second != j)
          edges[{f->second, j}].insert(v);
  StageMemoryPatternAnalysis patterns(module);
  if (auto e = StageWorkloadAnalysis().analyze(
          out, options.logicalProgramCountHint, &patterns))
    return std::move(e);
  if (auto e = StageFeatureAnalysis().analyze(out, &patterns))
    return std::move(e);
  if (auto e = StageKindClassifier().analyze(out, options.tinyDotFlopsMax))
    return std::move(e);
  for (auto &s : out.stages)
    for (auto &p : s.workload.addressPatterns)
      p.stageId = s.id;
  if (auto e = StageModeLegalityAnalysis().analyze(
          out, options.maximumSuperblockFactor,
          options.scopeSuperblockMaterializable,
          options.maximumScopeSuperblockFactor))
    return std::move(e);
  if (auto e = SimtRegisterPressureAnalysis().analyze(module, out, warps))
    return std::move(e);
  if (usableUB <= 0)
    return reject("aux_missing_actual_UB");
  if (auto e =
          StageResourceLegalityAnalysis().applySimtUbCapacity(out, usableUB))
    return std::move(e);
  for (const auto &entry : edges) {
    StageDependency e;
    e.sourceStage = out.stages[entry.first.first].id;
    e.targetStage = out.stages[entry.first.second].id;
    e.kind = StageDependencyKind::IntraIterationData;
    e.iterationDistance = 0;
    e.carriedValueName = "inter_stage_ssa";
    e.carriedValueIndex = 0;
    for (Value v : entry.second) {
      auto b = tensorBytes(v);
      if (!b || e.carriedValueBytes > INT64_MAX - *b)
        return reject("aux_unknown_edge");
      e.carriedValueBytes += *b;
    }
    out.dependencies.push_back(std::move(e));
  }
  if (auto e = StagePartitionVerifier().verify(out))
    return std::move(e);
  return out;
}
} // namespace mlir::ascend::normal_memory_aux
#endif
