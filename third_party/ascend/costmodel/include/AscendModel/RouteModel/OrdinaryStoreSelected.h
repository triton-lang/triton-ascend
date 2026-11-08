// Additive selected-formula feature bank. No LLVM, file, device or label
// access. The existing R4 guard/accounting boundary remains the source of
// eligibility.
#ifndef INDIRECT_STORE_SELECTED_FEATURES_H
#define INDIRECT_STORE_SELECTED_FEATURES_H
#include "OrdinaryStoreRound4.h"
namespace indirect_store_selected {
using Mode = indirect_store_round4::Mode;
using Visibility = indirect_store_round4::Visibility;
using Estimate = indirect_store_round4::Estimate;
using Features = indirect_store_round4::Features;
using Term = indirect_store_round4::Term;
using Formula = indirect_store_round4::Formula;
struct Metadata : indirect_store_round4::Metadata {
  // Supplied by the actual target StoreOp element type, never storage width.
  bool payloadTypeVerified = false;
  bool payloadIsFloat8 = false;
  bool payloadIsInt1 = false;
};
#include "OrdinaryStoreSelectedFormulas.inc"

inline std::optional<Features>
features(const Metadata &m, const std::vector<int64_t> *runtimeCoefficients,
         std::string &reason) {
  auto x = indirect_store_round4::features(m, runtimeCoefficients, reason);
  if (!x || x->count("semantic_zero"))
    return x;
  if (!m.payloadTypeVerified || (m.payloadIsFloat8 && m.payloadIsInt1) ||
      ((m.payloadIsFloat8 || m.payloadIsInt1) && m.elementBytes != 1)) {
    reason = "actual_payload_type_facts_missing_or_inconsistent";
    return std::nullopt;
  }
  const auto E = int64_t(x->at("E")), K = int64_t(x->at("K"));
  const auto W = m.baseWarps;
  int64_t S = 1, D = 1, lastS = -1, smallD = 0, maskedAxes = 0;
  bool prefixMasked = false;
  for (size_t i = 0; i < m.shape.size(); ++i) {
    if (m.dAxes[i]) {
      D *= m.shape[i];
      smallD += m.shape[i] <= 4;
    } else {
      S *= m.shape[i];
      lastS = int64_t(i);
    }
    const bool masked = m.upper[i] - m.lower[i] != m.shape[i];
    maskedAxes += masked;
    prefixMasked |= masked && i + 1 < m.shape.size();
  }
  // lastS is a logical axis index, not the total S extent or physical stride.
  int64_t suffixD = 1, fullSuffix = 1;
  for (size_t i = size_t(lastS + 1); i < m.shape.size(); ++i) {
    suffixD *= m.upper[i] - m.lower[i];
    fullSuffix *= m.shape[i];
  }
  const double lw = std::log2(double(W)), ld = x->at("logDstride");
  const double activeS = x->at("activeS"), activeD = x->at("activeD");
  const int64_t work = (E + 31) / 32;
  const Features additions{
      {"D", double(D)},
      {"S", double(S)},
      {"KlogTail", K * x->at("logTail")},
      {"suffixD", double(suffixD)},
      {"fullSuffix", double(fullSuffix)},
      {"KlogDstride2", K * ld * ld},
      {"KmaxSlog", K * x->at("maxSlog")},
      {"KoverTail", double(K) / (m.upper.back() - m.lower.back())},
      {"Kfp8", m.payloadIsFloat8 ? double(K) : 0.0},
      {"Kbool", m.payloadIsInt1 ? double(K) : 0.0},
      {"KmaskedAxes", double(K * maskedAxes)},
      {"prefixMasked", prefixMasked ? 1.0 : 0.0},
      {"tailMasked",
       m.upper.back() - m.lower.back() != m.shape.back() ? 1.0 : 0.0},
      {"KnD", K * x->at("nD")},
      {"EminusK", double(E - K)},
      {"KlogRuns", K * std::log2(std::max(x->at("activeRuns"), 1.0))},
      {"logW", lw},
      {"KlogW2", K * lw * lw},
      {"sqrtW", std::sqrt(double(W))},
      {"baseWarps", double(W)},
      {"activeWarpProxy", double(std::min(W, work))},
      {"idleWarpProxy", double(std::max<int64_t>(W - work, 0))},
      {"highWarp32", double(std::max<int64_t>(W - 32, 0))},
      {"logK", std::log2(double(K))},
      {"sqrtD", std::sqrt(activeD)},
      {"activeSlog", std::log2(activeS)},
      {"logW2", lw * lw},
      {"logWAbove2", std::max(lw - 2, 0.0)},
      {"logWAbove4", std::max(lw - 4, 0.0)},
      {"minW4", double(std::min<int64_t>(W, 4))},
      {"minW8", double(std::min<int64_t>(W, 8))},
      {"oneWarp", W == 1 ? 1.0 : 0.0},
      {"smallE32", E <= 32 ? 1.0 : 0.0},
      {"KlogWAbove2", K * std::max(lw - 2, 0.0)},
      {"KsuffixLog", K * std::log2(double(std::max<int64_t>(suffixD, 1)))},
      {"KsmallD", double(K * smallD)}};
  x->insert(additions.begin(), additions.end());
  // Construction has a fixed 88-key schema; validate each constructed value.
  if (x->size() != 88 ||
      std::any_of(x->begin(), x->end(), [](const auto &item) {
        return !std::isfinite(item.second);
      })) {
    reason = "complete_88_feature_bank_missing_or_nonfinite";
    return std::nullopt;
  }
  return x;
}

inline Estimate
tryPredict(const Metadata &m,
           const std::vector<int64_t> *runtimeCoefficients = nullptr,
           bool enabled = selectedDefaultEnabled) {
  auto fail = [](const char *why) {
    Estimate e;
    e.reason = why;
    return e;
  };
  if (!enabled)
    return fail("selected_formula_candidate_disabled");
  if (!m.stageOwnershipVerified)
    return fail("stage_ownership_not_proven");
  if (!m.firstStore)
    return fail("first_store_not_proven");
  if (!m.matchedIncrementContractVerified)
    return fail("matched_increment_contract_not_proven");
  std::string reason;
  auto x = indirect_store_selected::features(m, runtimeCoefficients, reason);
  if (!x) {
    Estimate e;
    e.reason = reason;
    return e;
  }
  if (x->count("semantic_zero")) {
    Estimate e;
    e.eligible = true;
    e.ticks = 0.;
    e.reason = "semantic_zero";
    return e;
  }
  const bool simd = m.mode == Mode::SIMD;
  std::string strategy = simd ? selectedSimdStrategy : selectedSimtStrategy;
  std::string key = std::to_string(m.elementBytes);
  if (strategy == "width_mask_size")
    key += std::string("_") + (m.hasMask ? "True" : "False") + "_" +
           (x->at("E") <= 128 ? "True" : "False");
  else if (strategy != "width")
    return fail("unsupported_model_strategy");
  const auto &bank = simd ? selectedSimdFormulas : selectedSimtFormulas;
  const Formula *formula = &bank.back();
  for (const auto &candidate : bank)
    if (key == candidate.key) {
      formula = &candidate;
      break;
    }
  double ticks = 0.;
  for (size_t i = 0; i < formula->size; ++i) {
    const auto it = x->find(formula->terms[i].name);
    if (it == x->end() || !std::isfinite(it->second))
      return fail("required_feature_missing_or_nonfinite");
    ticks += it->second * formula->terms[i].coefficient;
  }
  if (!std::isfinite(ticks))
    return fail("prediction_nonfinite");
  Estimate e;
  e.eligible = true;
  e.ticks = std::max(1., ticks);
  e.leaf = formula->key;
  e.reason = "eligible_selected_formula_production_unaccepted";
  // Independent metrics and native parity are separate evidence; never a
  // permission to enable a production launcher or mint a fresh-output lease.
  return e;
}
} // namespace indirect_store_selected
#endif
