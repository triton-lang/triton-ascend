// Visible physical facts only; no file, device or label access.
#ifndef ORDINARY_LOAD_VISIBLE_FEATURES_H
#define ORDINARY_LOAD_VISIBLE_FEATURES_H
#include "OrdinaryStoreSelected.h"
namespace indirect_load_visible {
struct LogicalKind {
  bool verified = false, integer = false, floating = false, bfloat16 = false;
};
inline std::optional<indirect_store_selected::Features>
features(const indirect_store_selected::Metadata &m,
         const std::vector<int64_t> *runtimeCoefficients, LogicalKind kind,
         std::string &reason) {
  auto result =
      indirect_store_selected::features(m, runtimeCoefficients, reason);
  if (!result || result->count("semantic_zero"))
    return result;
  if (!kind.verified || kind.integer == kind.floating ||
      (kind.bfloat16 && (!kind.floating || m.elementBytes != 2))) {
    reason = "actual_Load_logical_payload_kind_missing_or_inconsistent";
    return std::nullopt;
  }
  using Visibility = indirect_store_selected::Visibility;
  const std::vector<int64_t> *coefficients = nullptr;
  if (m.visibility == Visibility::Known)
    coefficients = &m.knownStrides;
  else if (m.visibility == Visibility::Host)
    coefficients = runtimeCoefficients;
  // Device deliberately leaves coefficients null, without even reading size().
  std::vector<int64_t> active;
  for (size_t i = 0; i < m.shape.size(); ++i)
    active.push_back(m.upper[i] - m.lower[i]);
  for (size_t axis = 0; axis < 5; ++axis) {
    const bool exists = axis < active.size();
    const bool d = exists && m.dAxes[axis], s = exists && !m.dAxes[axis];
    const std::string suffix = "_axis" + std::to_string(axis);
    (*result)["D" + suffix + "_extent"] = d ? double(active[axis]) : 0.;
    (*result)["S" + suffix + "_extent"] = s ? double(active[axis]) : 0.;
    const double pitch =
        exists && coefficients
            ? std::log2(double((*coefficients)[axis]) * double(m.elementBytes))
            : 0.;
    (*result)["D" + suffix + "_stride_log2"] = d ? pitch : 0.;
    (*result)["S" + suffix + "_stride_log2"] = s ? pitch : 0.;
    double innerBytes = double(m.elementBytes);
    if (d)
      for (size_t j = axis + 1; j < active.size(); ++j)
        innerBytes *= double(active[j]);
    (*result)["D" + suffix + "_inner_payload_bytes"] = d ? innerBytes : 0.;
  }
  double dRuns = 0., sToD = 0., dToS = 0.;
  for (size_t i = 0; i < m.dAxes.size(); ++i) {
    dRuns += m.dAxes[i] && (i == 0 || !m.dAxes[i - 1]);
    if (i > 0) {
      sToD += !m.dAxes[i - 1] && m.dAxes[i];
      dToS += m.dAxes[i - 1] && !m.dAxes[i];
    }
  }
  (*result)["D_runs"] = dRuns;
  (*result)["S_to_D_transitions"] = sToD;
  (*result)["D_to_S_transitions"] = dToS;
  (*result)["payload_integer"] = kind.integer;
  (*result)["payload_floating"] = kind.floating;
  (*result)["payload_bfloat16"] = kind.bfloat16;
  const double K = result->at("K");
  for (unsigned i = 0; i < 5; ++i) {
    const std::string axis = "_axis" + std::to_string(i);
    for (const char *ds : {"D", "S"}) {
      const std::string key = std::string(ds) + axis;
      (*result)["K_" + key + "_stride_log2"] =
          K * result->at(key + "_stride_log2");
      (*result)["K_" + key + "_extent_log2"] =
          K * std::log2(std::max(1., result->at(key + "_extent")));
    }
    (*result)["K_D" + axis + "_inner_log2"] =
        K * std::log2(
                std::max(1., result->at("D" + axis + "_inner_payload_bytes")));
    const double d = result->at("D" + axis + "_extent");
    (*result)["K_over_D" + axis + "_extent"] = d ? K / d : 0.;
  }
  for (const char *name :
       {"D_runs", "S_to_D_transitions", "D_to_S_transitions",
        "payload_bfloat16", "payload_floating", "payload_integer", "hidden",
        "prefixMasked", "tailMasked"})
    (*result)[std::string("K_") + name] = K * result->at(name);
  (*result)["activeRows_logW"] = result->at("activeRows") * result->at("logW");
  (*result)["width"] = m.elementBytes;
  (*result)["mask"] = m.hasMask;
  (*result)["rank"] = m.shape.size();
  (*result)["warp_band"] = m.baseWarps <= 4 ? 0. : m.baseWarps <= 16 ? 1. : 2.;
  (*result)["K_band"] = K <= 16 ? 0. : K <= 128 ? 1. : 2.;
  (*result)["leading_band"] = result->at("leading") <= 1   ? 0.
                              : result->at("leading") <= 8 ? 1.
                                                           : 2.;
  for (const auto &item : *result)
    if (!std::isfinite(item.second)) {
      reason = "Load_axis_feature_nonfinite";
      return std::nullopt;
    }
  return result;
}
} // namespace indirect_load_visible
#endif
