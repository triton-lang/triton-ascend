// Frozen observed-data Load selection. No file, device or label access.
#ifndef ORDINARY_LOAD_SELECTED_H
#define ORDINARY_LOAD_SELECTED_H
#include "OrdinaryLoadVisibleFeatures.h"
namespace indirect_load_selected {
using Estimate = indirect_store_selected::Estimate;
using Mode = indirect_store_selected::Mode;
using Term = indirect_store_selected::Term;
using Features = indirect_store_selected::Features;
struct Metadata : indirect_store_selected::Metadata {
  indirect_load_visible::LogicalKind logical;
};
inline constexpr bool productionEnabled = true;
inline constexpr const char *simdSourceSha256 =
    "529a0219c8c4241e1f9565888132ef2cc0a82de806c2580a7d7f301612e4fc65";
inline constexpr const char *simtSourceSha256 =
    "0e66084dd52a8f272c167fdbd6a613646e30206d425739a513064a414b6c096c";
enum class Instruction { Linear, Exp, LE, EQ, Scale, Blend };
struct ProgramNode {
  Instruction kind;
  int a, b;
  const char *feature;
  double value;
  const Term *terms;
  unsigned count;
};
#include "OrdinaryLoadProgram.inc"
struct Piece {
  const Term *terms = nullptr;
  unsigned count = 0;
  double scale = 1., shift = 0., floor = 1.;
  bool exponential = false;
};
inline std::optional<double> value(const Piece &p, const Features &x) {
  if (!p.terms || !p.count || p.count > 8)
    return std::nullopt;
  double result = 0.;
  for (unsigned i = 0; i < p.count; ++i) {
    auto f = x.find(p.terms[i].name);
    if (f == x.end() || !std::isfinite(f->second))
      return std::nullopt;
    result += f->second * p.terms[i].coefficient;
  }
  if (p.exponential)
    result = std::exp(std::max(-20., std::min(30., result)));
  result = std::max(p.floor, p.scale * result + p.shift);
  return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
}
inline std::optional<Piece> expression(int index, const Features &x,
                                       unsigned depth = 0) {
  if (depth > 128)
    return std::nullopt;
  unsigned steps = 0;
  for (;;) {
    if (index < 0 || size_t(index) >= std::size(programNodes) ||
        ++steps > std::size(programNodes))
      return std::nullopt;
    const auto &n = programNodes[index];
    if (n.kind == Instruction::LE || n.kind == Instruction::EQ) {
      auto f = x.find(n.feature);
      if (f == x.end() || !std::isfinite(f->second))
        return std::nullopt;
      bool condition = n.kind == Instruction::LE
                           ? f->second <= n.value
                           : int64_t(f->second) == n.value;
      index = condition ? n.a : n.b;
      continue;
    }
    if (n.kind == Instruction::Linear || n.kind == Instruction::Exp)
      return Piece{n.terms, n.count, 1., 0., 1., n.kind == Instruction::Exp};
    auto first = expression(n.a, x, depth + 1);
    if (!first || first->exponential)
      return std::nullopt;
    if (n.kind == Instruction::Scale) {
      if (!(n.value > 0.) || !std::isfinite(n.value))
        return std::nullopt;
      first->scale *= n.value;
      first->shift *= n.value;
      first->floor = std::max(1., first->floor * n.value);
      return first;
    }
    if (n.kind != Instruction::Blend || n.value < 0. || n.value > 1.)
      return std::nullopt;
    auto second = expression(n.b, x, depth + 1);
    if (!second || second->exponential || second->count != 1 ||
        std::string(second->terms[0].name) != "one")
      return std::nullopt;
    bool hasConstant = false;
    for (unsigned i = 0; i < first->count; ++i)
      hasConstant |= std::string(first->terms[i].name) == "one";
    if (!hasConstant)
      return std::nullopt;
    auto constant = value(*second, x);
    if (!constant)
      return std::nullopt;
    first->scale *= n.value;
    first->shift = n.value * first->shift + (1. - n.value) * *constant;
    first->floor = n.value * first->floor + (1. - n.value) * *constant;
    // The shift merges into the existing constant term: still <=8 terms.
    return first;
  }
}
inline Estimate
tryPredict(const Metadata &m,
           const std::vector<int64_t> *runtimeCoefficients = nullptr,
           bool enabled = productionEnabled) {
  auto fail = [](const char *reason) {
    Estimate e;
    e.reason = reason;
    return e;
  };
  if (!enabled)
    return fail("Load_formula_explicitly_disabled");
  if (!m.stageOwnershipVerified || !m.matchedIncrementContractVerified)
    return fail("Load_target_partition_or_increment_contract_missing");
  if (!m.logical.verified)
    return fail("Load_logical_type_not_verified");
  std::string reason;
  auto x = indirect_load_visible::features(m, runtimeCoefficients, m.logical,
                                           reason);
  if (!x) {
    Estimate e;
    e.reason = reason;
    return e;
  }
  Estimate e;
  if (x->count("semantic_zero")) {
    e.eligible = true;
    e.ticks = 0.;
    e.reason = "semantic_zero";
    return e;
  }
  auto p = expression(m.mode == Mode::SIMD ? simdRoot : simtRoot, *x);
  if (!p)
    return fail("Load_physical_program_contract_invalid");
  auto ticks = value(*p, *x);
  if (!ticks)
    return fail("Load_prediction_nonfinite_or_missing_feature");
  e.eligible = true;
  e.ticks = *ticks;
  e.leaf = "physical_short8";
  e.reason = "observed_resplit_Load_physical_short8";
  return e;
}
} // namespace indirect_load_selected
#endif
