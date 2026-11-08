// Frozen observed-data Load selection. No file, device or label access.
#ifndef ORDINARY_LOAD_SELECTED_H
#define ORDINARY_LOAD_SELECTED_H
#include "OrdinaryLoadVisibleFeatures.h"
#include <array>
namespace indirect_load_selected {
using Estimate = indirect_store_selected::Estimate;
using Mode = indirect_store_selected::Mode;
using Term = indirect_store_selected::Term;
using Features = indirect_store_selected::Features;
struct Metadata : indirect_store_selected::Metadata {
  indirect_load_visible::LogicalKind logical;
};
inline constexpr bool productionEnabled = true;
struct Piece {
  const Term *terms;
  unsigned count;
};
struct RouteColumn {
  const char *feature;
  bool thresholded;
  double threshold;
};
struct PhysicalRoute {
  std::array<int64_t, 18> key;
  unsigned piece;
};
struct BaseRoute {
  int64_t width;
  bool mask, flag;
  unsigned piece;
};
#include "OrdinaryLoadProgram.inc"
inline std::optional<Piece> physicalRoot(Mode mode, const Features &x) {
  std::array<int64_t, 18> key;
  for (size_t i = 0; i < key.size(); ++i) {
    const auto &c = routeColumns[i];
    auto f = x.find(c.feature);
    if (f == x.end() || !std::isfinite(f->second) ||
        (!c.thresholded && (f->second < -0x1p63 || f->second >= 0x1p63)))
      return std::nullopt;
    key[i] = c.thresholded ? f->second > c.threshold : int64_t(f->second);
  }
  const auto *rs = mode == Mode::SIMD ? simdRoutes : simtRoutes;
  const auto n =
      mode == Mode::SIMD ? std::size(simdRoutes) : std::size(simtRoutes);
  for (size_t i = 0; i < n; ++i)
    if (rs[i].key == key)
      return pieces[rs[i].piece];
  const bool prefix = mode == Mode::SIMD ? simdPrefix : simtPrefix;
  const bool flag = prefix ? key[7] != 0 : key[4] == 1;
  const auto *bs = mode == Mode::SIMD ? simdBase : simtBase;
  const auto bn =
      mode == Mode::SIMD ? std::size(simdBase) : std::size(simtBase);
  for (size_t i = 0; i < bn; ++i)
    if (bs[i].width == key[0] && bs[i].mask == (key[2] != 0) &&
        bs[i].flag == flag)
      return pieces[bs[i].piece];
  return pieces[mode == Mode::SIMD ? simdFallback : simtFallback];
}
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
  result = std::max(1., result);
  return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
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
  auto root = physicalRoot(m.mode, *x);
  if (!root)
    return fail("Load_physical_route_missing_feature");
  auto ticks = value(*root, *x);
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
