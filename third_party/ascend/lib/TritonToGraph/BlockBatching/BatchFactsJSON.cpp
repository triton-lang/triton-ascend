// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchRewriteResult.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::triton::cfg {
namespace {
llvm::json::Value exprJSON(const BatchExpr &self);
} // namespace

llvm::json::Value batchEffectsJSON(const BatchFacts &facts) {
  llvm::json::Array accesses;
  using A = llvm::json::Array;
  for (auto [id, slot] : llvm::enumerate(facts.effects())) {
    const auto &fact = facts.access(id);
    A checks, shape, loops;
    for (const auto &step : fact.pointerChecks)
      checks.push_back(batchExprJSON(step));
    if (slot.jointWidth)
      shape.push_back(slot.jointWidth);
    else
      for (int64_t extent : fact.shape)
        shape.push_back(extent);
    for (unsigned loop : fact.loopIDs)
      loops.push_back(batchLoopJSON(*facts.findLoop(loop)));
    accesses.push_back(llvm::json::Object{
        {"root", fact.root},
        {"offset", batchExprJSON(slot.offset ? slot.offset : fact.byteOffset)},
        {"shape", std::move(shape)},
        {"mask", batchExprJSON(slot.mask ? slot.mask : fact.mask)},
        {"write", fact.write},
        {"bytes", fact.bytes},
        {"pointer_checks", std::move(checks)},
        {"loops", std::move(loops)},
        {"volatile", fact.isVolatile},
        {"device_address", fact.deviceAddress}});
  }
  return llvm::json::Object{{"version", 2}, {"accesses", std::move(accesses)}};
}

namespace {
llvm::json::Value batchProofJSON(const BatchFacts &facts,
                                 llvm::ArrayRef<BatchEffectObligation> effects,
                                 llvm::ArrayRef<unsigned> terminalIDs, int axis,
                                 int64_t factor) {
  auto wire = batchEffectsJSON(facts);
  auto &object = *wire.getAsObject();
  object["version"] = 3;
  object["plan_kind"] = "unified";
  object["semantic_contract_version"] = 4;
  object["memory_policy"] = "disjoint_or_terminal_store";
  object["factor"] = factor;
  object["axis"] = axis;
  llvm::json::Array obligations, terminalReads;
  for (const auto &obligation : effects)
    obligations.push_back(llvm::json::Object{
        {"access_id", static_cast<int64_t>(obligation.accessID)},
        {"rule", batchEffectRuleName(obligation.rule)},
        {"proof", obligation.proof == BatchEffectProof::ByteDomain
                      ? "byte_domain"
                      : "lane_correspondence"}});
  for (unsigned id : terminalIDs)
    terminalReads.push_back(id);
  object["terminal_read_ids"] = std::move(terminalReads);
  object["obligations"] = std::move(obligations);
  return wire;
}
} // namespace

// Refusal diagnostics are encoded on demand by the outer transaction. They
// are never accepted as an execution contract or stored in a planning object.
std::optional<std::string> encodeBatchPendingGuard(
    const BatchFacts &facts, llvm::ArrayRef<BatchEffectObligation> effects,
    llvm::ArrayRef<unsigned> terminalIDs, int axis, int64_t factor) {
  if (!BatchPlan::validEffects(facts, effects, terminalIDs))
    return std::nullopt;
  return llvm::formatv(
             "{0}", batchProofJSON(facts, effects, terminalIDs, axis, factor))
      .str();
}

BatchResult<std::string> encodeBatchGuard(const BatchAppliedRewrite &applied) {
  using Result = BatchResult<std::string>;
  const auto &plan = applied.plan();
  auto wire = batchProofJSON(plan.facts(), plan.effects(), plan.terminalReads(),
                             plan.axis(), plan.factor());
  auto &object = *wire.getAsObject();
  auto &obligations = *object.getArray("obligations");
  llvm::json::Array layoutChecks;
  for (const auto &layout : applied.layouts()) {
    const auto &planned = plan.layouts()[layout.accessID];
    if (layout.rewritten)
      (*obligations[layout.accessID].getAsObject())["proof"] = "byte_domain";
    if (layout.runtimeGuardJSON.empty() &&
        !(planned.symbolic && layout.rewritten))
      continue;
    auto invalid = [&] {
      return Result::error({"invalid finalized symbolic layout guard",
                            "encode",
                            {},
                            {},
                            {},
                            layout.accessID});
    };
    if (!planned.symbolic || !layout.rewritten)
      return invalid();
    auto parsed = llvm::json::parse(layout.runtimeGuardJSON);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return invalid();
    }
    const auto *guard = parsed->getAsObject();
    const auto &access = plan.facts().access(layout.accessID);
    if (!guard || guard->getInteger("version") != 1 ||
        guard->getInteger("axis") != plan.axis() ||
        guard->getInteger("factor") != plan.factor() ||
        guard->getInteger("access_id") != layout.accessID ||
        guard->getInteger("root") != access.root ||
        guard->getInteger("bytes") != access.bytes ||
        !guard->getArray("loops") ||
        guard->getArray("loops")->size() != access.loopIDs.size())
      return invalid();
    for (llvm::StringRef key :
         {"shape", "checks", "positive", "pointer_checks"})
      if (!guard->getArray(key) || guard->getArray(key)->empty())
        return invalid();
    layoutChecks.push_back(std::move(*parsed));
  }
  if (plan.preservesGrid())
    object["loop_schedule"] =
        llvm::json::Object{{"version", 1},
                           {"kind", "sequential_iterations_v1"},
                           {"grid_policy", "preserve"},
                           {"factor", plan.factor()},
                           {"loop", batchLoopJSON(*plan.groupedLoop())}};
  if (!layoutChecks.empty())
    object["layout_checks"] = std::move(layoutChecks);
  for (const auto &entry : applied.matrixExtensions())
    object[entry.first.str()] = entry.second;
  return Result::success(llvm::formatv("{0}", wire).str());
}

llvm::json::Value batchExprJSON(const BatchExprRef &expr) {
  return expr ? exprJSON(*expr) : llvm::json::Value(nullptr);
}
llvm::json::Value batchLoopJSON(const BatchLoopFact &loop) {
  return llvm::json::Object{{"id", loop.id},
                            {"bits", loop.bits},
                            {"lower", batchExprJSON(loop.lower)},
                            {"upper", batchExprJSON(loop.upper)},
                            {"step", batchExprJSON(loop.step)}};
}

namespace {
llvm::json::Value exprJSON(const BatchExpr &self) {
  using A = llvm::json::Array;
  switch (self.kind) {
  case BatchExpr::Constant:
    return A{"const", self.value, self.bits};
  case BatchExpr::Argument:
    return A{"arg", self.value, self.bits};
  case BatchExpr::Program:
    return A{"pid", self.value, self.bits};
  case BatchExpr::ProgramQuotient:
  case BatchExpr::ProgramRemainder:
    return A{self.kind == BatchExpr::ProgramQuotient ? "pid_div" : "pid_rem",
             self.value, self.extent, self.bits};
  case BatchExpr::Grid:
    return A{"grid", self.value, self.bits};
  case BatchExpr::Coordinate:
    return A{"coord", self.value, self.extent, self.bits};
  case BatchExpr::Induction:
    return A{"iv", self.value, self.bits};
  case BatchExpr::True:
    return A{"true"};
  case BatchExpr::False:
    return A{"false"};
  case BatchExpr::Unknown:
    return A{"unknown"};
  case BatchExpr::And:
    return A{"and", batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  case BatchExpr::SLT:
  case BatchExpr::ULT:
  case BatchExpr::EQ:
    return A{self.kind == BatchExpr::SLT   ? "slt"
             : self.kind == BatchExpr::ULT ? "ult"
                                           : "eq",
             batchExprJSON(self.lhs), batchExprJSON(self.rhs), self.bits};
  case BatchExpr::ExtSI:
  case BatchExpr::ExtUI:
  case BatchExpr::Trunc:
    return A{self.kind == BatchExpr::ExtSI   ? "extsi"
             : self.kind == BatchExpr::ExtUI ? "extui"
                                             : "trunci",
             self.bits, batchExprJSON(self.lhs), self.extent};
  case BatchExpr::MinSI:
  case BatchExpr::MaxSI:
    return A{self.kind == BatchExpr::MinSI ? "minsi" : "maxsi", self.bits,
             batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  case BatchExpr::Add:
  case BatchExpr::Sub:
  case BatchExpr::Mul:
    return A{self.kind == BatchExpr::Add   ? "addi"
             : self.kind == BatchExpr::Sub ? "subi"
                                           : "muli",
             self.bits, batchExprJSON(self.lhs), batchExprJSON(self.rhs)};
  }
  llvm_unreachable("unhandled batch coordinate expression");
}
} // namespace

} // namespace mlir::triton::cfg
