// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchRewriteFinalizer.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
namespace mlir::triton::cfg {
namespace {
constexpr llvm::StringLiteral effectID = "__vv_effect_id";
bool matchesSourcePolicy(Operation *op, const BatchFacts::Access &source) {
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return !source.write && load.getIsVolatile() == source.isVolatile &&
           static_cast<int64_t>(load.getCache()) == source.cache &&
           static_cast<int64_t>(load.getEvict()) == source.evict;
  auto store = cast<triton::StoreOp>(op);
  return source.write && !source.isVolatile &&
         static_cast<int64_t>(store.getCache()) == source.cache &&
         static_cast<int64_t>(store.getEvict()) == source.evict;
}
} // namespace
LogicalResult markBatchEffects(ArrayRef<Operation *> effects) {
  for (auto [id, op] : llvm::enumerate(effects)) {
    if (!op)
      return failure();
    op->setAttr(effectID, Builder(op->getContext()).getI64IntegerAttr(id));
  }
  return success();
}
void copyBatchEffectIdentity(Operation *source, Operation *target) {
  if (auto identity = source->getAttr(effectID))
    target->setAttr(effectID, identity);
}

BatchResult<BatchAppliedRewrite> BatchRewriteFinalizer::programLanes(
    ModuleOp module, std::shared_ptr<const BatchPlan> plan,
    unsigned originalArgCount, SmallVector<BatchLayoutResult, 0> layouts) {
  return createChecked(
      module, std::move(plan), originalArgCount, std::move(layouts),
      BatchVerification::EffectCorrespondence, BatchSchedule::ProgramLanes, {});
}
BatchResult<BatchAppliedRewrite> BatchRewriteFinalizer::sequentialIterations(
    ModuleOp module, std::shared_ptr<const BatchPlan> plan,
    unsigned originalArgCount) {
  return createChecked(module, std::move(plan), originalArgCount, {},
                       BatchVerification::EffectCorrespondence,
                       BatchSchedule::SequentialIterations, {});
}
BatchResult<BatchAppliedRewrite> BatchRewriteFinalizer::matrixLanes(
    ModuleOp module, std::shared_ptr<const BatchPlan> plan,
    unsigned originalArgCount, StringRef matrixGuard,
    SmallVector<BatchLayoutResult, 0> layouts) {
  return createChecked(module, std::move(plan), originalArgCount,
                       std::move(layouts), BatchVerification::StructuredMatrix,
                       BatchSchedule::ProgramLanes, matrixGuard);
}

BatchResult<BatchAppliedRewrite> BatchRewriteFinalizer::createChecked(
    ModuleOp module, std::shared_ptr<const BatchPlan> plan,
    unsigned originalArgCount, SmallVector<BatchLayoutResult, 0> layouts,
    BatchVerification verification, BatchSchedule schedule,
    StringRef matrixGuard) {
  using Result = BatchResult<BatchAppliedRewrite>;
  auto fail = [](StringRef reason) {
    return Result::error({reason.str(), "finalize", {}, {}, {}, {}});
  };
  bool matrix = verification == BatchVerification::StructuredMatrix;
  if (!plan || plan->schedule() != schedule || plan->isMatrix() != matrix)
    return fail("invalid finalized batch strategy");
  if (layouts.size() != plan->layouts().size())
    return fail("incomplete finalized batch layouts");
  for (auto [id, layout] : llvm::enumerate(layouts)) {
    auto planned = plan->layouts()[id].action;
    if (layout.accessID != id ||
        (layout.action != planned && layout.action != BatchLayoutPlan::Keep) ||
        (layout.rewritten && layout.action == BatchLayoutPlan::Keep))
      return fail("invalid finalized batch layout outcome");
  }
  triton::FuncOp function;
  for (auto candidate : module.getOps<triton::FuncOp>())
    if (candidate.isPublic()) {
      if (function)
        return fail("multiple finalized batch entry points");
      function = candidate;
    }
  if (!function || function.getNumArguments() !=
                       originalArgCount + plan->hiddenExtentCount())
    return fail("invalid finalized batch entry ABI");
  auto grid = module->getAttr(kProgramGridTransformsAttr);
  if (plan->preservesGrid()) {
    if (grid || !plan->groupedLoop())
      return fail("invalid finalized sequential grid");
  } else {
    if (!hasProgramGridHiddenExtentArguments(function, 3) || !grid)
      return fail("missing finalized batch grid ABI");
    auto contract = parseProgramGridTransformContract(grid);
    if (failed(contract) || contract->version != 3 ||
        contract->transforms.size() != 1)
      return fail("invalid finalized batch grid contract");
    const auto &transform = contract->transforms.front();
    if (transform.axis != plan->axis() || transform.factor != plan->factor() ||
        transform.order != 0 || transform.persistentCoverage ||
        transform.gridStrideAbiVerified)
      return fail("invalid finalized batch grid transform");
  }
  llvm::SmallVector<BatchMappedEffect, 0> mappings;
  llvm::json::Object extensions;
  if (matrix) {
    auto parsed = llvm::json::parse(matrixGuard);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return fail("invalid matrix guard extensions");
    }
    auto *object = parsed->getAsObject();
    if (!object)
      return fail("invalid matrix guard extensions");
    for (auto &entry : *object)
      if (entry.first.str().compare(0, 3, "cv_") == 0)
        extensions[entry.first.str()] = std::move(entry.second);
  } else {
    // Source memory policy is authoritative even after the original IR is gone.
    const auto &facts = plan->facts();
    for (unsigned id = 0; id < facts.accesses().size(); ++id)
      mappings.push_back({id, {}});
    unsigned targetID = 0;
    auto matched = function.walk([&](Operation *op) -> WalkResult {
      if (!isa<triton::LoadOp, triton::StoreOp>(op))
        return WalkResult::advance();
      auto id = op->getAttrOfType<IntegerAttr>(effectID);
      if (!id || id.getInt() < 0 ||
          static_cast<size_t>(id.getInt()) >= mappings.size() ||
          !matchesSourcePolicy(op, facts.access(id.getInt())))
        return WalkResult::interrupt();
      op->removeAttr(effectID);
      mappings[id.getInt()].generatedIDs.push_back(targetID++);
      return WalkResult::advance();
    });
    if (matched.wasInterrupted() || plan->effects().size() != mappings.size())
      return fail("invalid finalized VV effect correspondence");
    for (const auto &effect : mappings) {
      size_t count = effect.generatedIDs.size();
      bool grouped = plan->preservesGrid() &&
                     llvm::is_contained(facts.access(effect.accessID).loopIDs,
                                        plan->groupedLoop()->id);
      if (grouped ? (count != static_cast<size_t>(plan->factor()) &&
                     count != static_cast<size_t>(plan->factor() + 1))
                  : count != 1)
        return fail("invalid finalized VV effect multiplicity");
    }
  }
  if (failed(verify(module)))
    return fail("invalid finalized batch IR");
  return Result::success(
      BatchAppliedRewrite(std::move(plan), verification, std::move(mappings),
                          std::move(layouts), std::move(extensions)));
}
} // namespace mlir::triton::cfg
