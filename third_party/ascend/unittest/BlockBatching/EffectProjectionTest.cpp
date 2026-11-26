// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Compile and link only the owned facts and production codec, without MLIR.
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "TritonToGraph/BlockBatching/BatchRewriteResult.h"
#include <cstdlib>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>

using namespace mlir::triton::cfg;

template <typename T, typename = void> struct HasOperation : std::false_type {};
template <typename T>
struct HasOperation<T, std::void_t<decltype(std::declval<T>().operation)>>
    : std::true_type {};
static_assert(!HasOperation<BatchFacts::Access>::value,
              "owned access facts must not retain an IR operation");

static void check(bool ok, const char *message) {
  if (!ok) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

static_assert(!std::is_default_constructible_v<BatchAppliedRewrite>);
static_assert(!std::is_copy_constructible_v<BatchAppliedRewrite>);
static_assert(std::is_move_constructible_v<BatchAppliedRewrite>);
static_assert(!std::is_copy_constructible_v<BatchPlan>);

int main() {
  std::optional<BatchFacts> vv;
  {
    BatchFacts::Access source;
    source.root = 2;
    source.bytes = 4;
    source.write = true;
    source.shape = {8};
    source.byteOffset = BatchExpr::leaf(BatchExpr::Argument, 3, 64);
    source.pointerChecks = {source.byteOffset};
    // The wire order is innermost first, not numeric LoopId order.
    BatchLoopFact inner, outer;
    inner.id = 7;
    outer.id = 2;
    inner.bits = outer.bits = 32;
    inner.lower = outer.lower = BatchExpr::leaf(BatchExpr::Constant, 0, 32);
    inner.upper = outer.upper = BatchExpr::leaf(BatchExpr::Argument, 4, 32);
    inner.step = outer.step = BatchExpr::leaf(BatchExpr::Constant, 1, 32);
    source.loopIDs = {inner.id, outer.id};
    llvm::SmallVector<BatchFacts::Access, 0> accesses{source, source};
    llvm::SmallVector<BatchFacts::EffectSlot, 0> overrides(2);
    overrides[1].mask = BatchExpr::leaf(BatchExpr::False);
    vv = BatchFacts::createChecked(
        std::move(accesses), std::move(overrides),
        llvm::SmallVector<BatchLoopFact, 0>{inner, outer});
  }
  check(vv.has_value(), "sparse valid loop IDs must freeze");
  auto wire = batchEffectsJSON(*vv);
  auto *object = wire.getAsObject();
  auto *accesses = object->getArray("accesses");
  check(object->getInteger("version") == 2 && accesses->size() == 2,
        "legacy access protocol and member slots must remain intact");
  auto *covered = (*accesses)[1].getAsObject();
  check(covered->getInteger("root") == 2 &&
            covered->getBoolean("write") == true,
        "covered member must preserve its effect identity");
  check(covered->getArray("mask")->front().getAsString() == "false",
        "covered member suppresses only its redundant alias hull");
  check(covered->getArray("pointer_checks")->size() == 1,
        "covered member must still retain original pointer arithmetic");
  auto *loops = covered->getArray("loops");
  check((*loops)[0].getAsObject()->getInteger("id") == 7 &&
            (*loops)[1].getAsObject()->getInteger("id") == 2,
        "inner-to-outer loop order must not change");
  check(!(*loops)[0].getAsObject()->get("carriesValues"),
        "compiler-only facts must not leak into the runtime ABI");
  check(vv->loops().size() == 2 && vv->accesses().size() == 2,
        "one loop table must serve all original accesses and effects");
  check(vv->accesses()[1].loopIDs[0] == 7 &&
            vv->accesses()[1].loopIDs[1] == 2 &&
            vv->findLoop(7)->upper->value == 4,
        "stable sparse IDs must survive source destruction");
  check(!vv->accesses()[1].mask &&
            vv->effects()[1].mask->kind == BatchExpr::False,
        "guard normalization must not modify original facts");
  check(vv->loops()[0].id == 2 && vv->loops()[1].id == 7,
        "loop table order is independent of the access nesting order");
  BatchFacts::Access dangling;
  dangling.loopIDs.push_back(9);
  check(!BatchFacts::createChecked({dangling}, {}, {}),
        "missing effect slot must fail");
  check(!BatchFacts::createChecked({dangling}, {{}}, {}),
        "unknown loop ID must fail");
  auto facts = std::make_shared<const BatchFacts>(*vv);
  llvm::SmallVector<BatchLayoutPlan, 0> layouts(2);
  layouts[0].accessID = 0;
  layouts[1].accessID = 1;
  using Strategy = BatchStrategy;
  const llvm::SmallVector<BatchEffectObligation, 0> effects = {
      {0, BatchEffectRule::SupportedEffect, BatchEffectProof::ByteDomain},
      {1, BatchEffectRule::SupportedEffect, BatchEffectProof::ByteDomain}};
  const BatchLogicalAxis loopAxis{BatchLogicalAxis::Blocked, 7, 1, 1};
  auto planFor = [&](Strategy strategy, int axis, int64_t factor,
                     std::optional<BatchLogicalAxis> loop,
                     llvm::SmallVector<BatchLayoutPlan, 0> layout,
                     llvm::SmallVector<BatchEffectObligation, 0> obligations,
                     llvm::SmallVector<unsigned, 0> terminalReads = {}) {
    auto result = BatchPlan::createChecked(
        facts, axis, factor, strategy, loop, std::move(layout),
        std::move(obligations), std::move(terminalReads));
    if (!result.hasValue()) {
      check(result.status() == decltype(result)::Status::InternalError &&
                result.diagnostic().stage == "plan" &&
                !result.diagnostic().reason.empty(),
            "contradictory planner evidence is an internal error");
      return std::shared_ptr<const BatchPlan>{};
    }
    return std::move(result).takeValue();
  };
  auto program =
      planFor(Strategy::VectorLanes, 1, 4, loopAxis, layouts, effects);
  auto sequential =
      planFor(Strategy::SequentialLoop, 1, 4, loopAxis, {}, effects);
  check(program && sequential && !program->groupedLoop() &&
            program->logicalAxis()->loopID == 7 && program->factor() == 4 &&
            !program->preservesGrid() && program->hiddenExtentCount() == 3 &&
            sequential->groupedLoop()->id == 7 && sequential->preservesGrid() &&
            sequential->hiddenExtentCount() == 0,
        "lowering and guard must use the accepted coordinates and derived ABI");
  check(!planFor(Strategy::VectorLanes, 3, 4, {}, layouts, effects) &&
            !planFor(Strategy::VectorLanes, 0, 1, {}, layouts, effects) &&
            !planFor(Strategy::VectorLanes, 0, 4, {}, {}, effects) &&
            !planFor(Strategy::SequentialLoop, 0, 4, {}, {}, effects) &&
            !planFor(Strategy::SequentialLoop, 0, 4, loopAxis, {}, effects),
        "invalid axes, factors, missing layouts or loop coordinates must fail");
  auto badLoop = loopAxis;
  badLoop.loopID = 99;
  check(!planFor(Strategy::SequentialLoop, 1, 4, badLoop, {}, effects),
        "an unknown grouped loop must fail");
  auto invalidEffects = effects;
  invalidEffects[1].accessID = 0;
  check(
      !planFor(Strategy::VectorLanes, 0, 4, {}, layouts, invalidEffects) &&
          !planFor(Strategy::SequentialLoop, 1, 4, loopAxis, {}, effects, {2}),
      "inconsistent effect identity and terminal read IDs must fail");
  auto matrix = planFor(Strategy::MatrixBatch, 0, 4, {}, layouts, effects);
  auto spatial = planFor(Strategy::MatrixSpatial, 0, 4, {}, {}, effects);
  auto cloned = planFor(Strategy::MatrixStaticLanes, 0, 4, {}, {}, effects);
  check(matrix && spatial && cloned && matrix->isMatrix() &&
            !program->isMatrix() && !spatial->preservesGrid() &&
            cloned->hiddenExtentCount() == 3,
        "matrix and vector plans must share the grid and hidden ABI model");
  check(!planFor(Strategy::MatrixBatch, 0, 4, {}, {}, effects) &&
            !planFor(Strategy::MatrixSpatial, 0, 4, {}, layouts, effects) &&
            !planFor(Strategy::MatrixStaticLanes, 0, 4, {}, {}, invalidEffects),
        "invalid layouts or effect evidence must not form matrix plans");
  auto ownedEffects = effects;
  llvm::SmallVector<unsigned, 0> ownedReads{1};
  const auto *storage = ownedEffects.data();
  auto readStorage = ownedReads.data();
  layouts[1].accessID = 0;
  auto refused = BatchPlan::createChecked(
      facts, 1, 4, Strategy::VectorLanes, {}, std::move(layouts),
      std::move(ownedEffects), std::move(ownedReads));
  check(!refused.hasValue() &&
            refused.status() == decltype(refused)::Status::InternalError &&
            refused.diagnostic().sourceID == 1 &&
            ownedEffects.data() == storage &&
            ownedReads.data() == readStorage && ownedEffects.size() == 2 &&
            ownedReads.front() == 1,
        "a refused plan must leave the source proof available for diagnostics");
  layouts[1].accessID = 1;
  auto frozen = BatchPlan::createChecked(
      facts, 1, 4, Strategy::VectorLanes, {}, std::move(layouts),
      std::move(ownedEffects), std::move(ownedReads));
  check(frozen.hasValue(), "valid proof must freeze");
  auto accepted = std::move(frozen).takeValue();
  check(accepted && accepted->effects().data() == storage &&
            accepted->terminalReads().data() == readStorage,
        "freezing transfers evidence storage into the shared accepted plan");
  vv.reset();
  facts.reset();
  check(program->facts().accesses().size() == 2 &&
            sequential->groupedLoop()->id == 7,
        "accepted plans must own facts independently of the analysis producer");
  std::cout << "owned effect projection regressions passed\n";
}
