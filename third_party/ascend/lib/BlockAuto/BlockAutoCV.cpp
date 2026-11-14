// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "Selection.h"
#include <tuple>

namespace mlir::triton::ascend::blockauto {
std::string mergeKind(const O &metadata, int64_t factor, bool frontend) {
  if (factor == 1)
    return "unmerged";
  const O &report = frontend ? metadata : optional(metadata, "block_batching");
  check(!frontend || str(report, "compile_status") == "REWRITTEN",
        "frontend did not admit the CV rewrite");
  if (str(report, "cv_mode") == "independent_batch_v1")
    return frontend ? "batch" : "independent_batch";
  std::string lowering = str(report, "cv_lowering");
  if (frontend && StringRef(lowering).ends_with("_uniform_control"))
    lowering.resize(lowering.size() - 16);
  if (str(report, "cv_mode") == "spatial_concat_v1") {
    if (lowering == "spatial_m_v1")
      return "M";
    if (lowering == "spatial_n_v1")
      return "N";
  }
  throw std::runtime_error("compiler CV axis role is not covered");
}
O planCV(const O &request) {
  const A &grid = array(sub(request, "contract"), "grid");
  int64_t selectedAxis = integer(request, "axis", -1);
  check(grid.size() == 3 && selectedAxis >= -1 && selectedAxis < 3,
        "invalid CV launch grid or axis");
  product(grid);
  A probes;
  auto result = [](StringRef reason, A candidates = {}) {
    return O{{"reason", reason}, {"candidates", std::move(candidates)}};
  };
  if (!request.get("reports")) {
    for (int64_t axis = 0; axis < 3; ++axis)
      for (int64_t factor : {2, 4})
        if ((selectedAxis == -1 || selectedAxis == axis) &&
            integer(grid[axis]) >= factor && integer(grid[axis]) % factor == 0)
          probes.push_back(O{{"axis", axis}, {"factor", factor}});
    return O{{"probes", std::move(probes)}};
  }
  std::map<std::string, A> admitted;
  std::map<int64_t, std::string> roles;
  for (const V &value : array(request, "reports")) {
    try {
      O item(object(value));
      int64_t axis = integer(item, "axis"), factor = integer(item, "factor");
      check(axis >= 0 && axis < 3 && (factor == 2 || factor == 4) &&
                (selectedAxis == -1 || axis == selectedAxis) &&
                integer(grid[axis]) >= factor &&
                integer(grid[axis]) % factor == 0,
            "invalid probe");
      const O &report = sub(item, "report");
      check(integer(report, "selected_axis", -1) == axis &&
                integer(report, "requested_factor") == factor,
            "frontend selected a different axis or factor");
      std::string kind = mergeKind(report, factor, true);
      auto [role, inserted] = roles.emplace(axis, kind);
      if (!inserted && role->second != kind)
        return result("CV_AXIS_ROLE_DEPENDS_ON_FACTOR");
      admitted[kind].push_back(
          O{{"axis", axis},
            {"factor", factor},
            {"key", "a" + std::to_string(axis) + "_f" + std::to_string(factor)},
            {"merge_kind", kind}});
    } catch (const std::exception &) {
    }
  }
  auto order = [](const V &v) {
    const O &o = object(v);
    return std::make_pair(integer(o, "axis"), integer(o, "factor"));
  };
  for (auto &[kind, candidates] : admitted)
    std::sort(candidates.begin(), candidates.end(),
              [&](const V &a, const V &b) { return order(a) < order(b); });
  A spatial;
  const O *m = nullptr, *n = nullptr;
  for (const std::string kind : {"M", "N"})
    for (const V &v : admitted[kind]) {
      const O &o = object(v);
      spatial.push_back(v);
      if (integer(o, "factor") == 4) {
        auto &slot = kind == "M" ? m : n;
        if (!slot)
          slot = &o;
      }
    }
  if (m && n)
    return result("CV_M4_N4", A{O(*m), O(*n)});
  if (m || n) {
    A candidates;
    for (const V &v : spatial)
      if (integer(object(v), "axis") == integer(m ? *m : *n, "axis"))
        candidates.push_back(v);
    return result("CV_ONE_SPATIAL_AXIS_F2_F4", std::move(candidates));
  }
  if (!spatial.empty()) {
    while (spatial.size() > 2)
      spatial.pop_back();
    return result("CV_SPATIAL_F2_ONLY", std::move(spatial));
  }
  A batch;
  for (const V &v : admitted["batch"])
    if (batch.empty() ||
        integer(object(v), "axis") == integer(object(batch.front()), "axis"))
      batch.push_back(v);
  StringRef reason = batch.empty() ? "CV_NO_ADMITTED_MERGE" : "CV_BATCH_F2_F4";
  return result(reason, std::move(batch));
}
O rankCV(const O &records, const O &contract) {
  A incomplete;
  for (const auto &[key, value] : records) {
    const O &row = object(value);
    if (!flag(row, "resource_checked") &&
        (integer(row, "factor") == 1 || !excluded(str(row, "status"), false)))
      incomplete.push_back(key.str());
  }
  if (!incomplete.empty())
    return O{{"reason", "CV_MODEL_INCOMPLETE"},
             {"incomplete_candidates", std::move(incomplete)}};
  const O &baseline = sub(records, "unmerged");
  const A &original = array(contract, "grid");
  std::vector<std::string> eligible;
  for (const auto &[key, value] : records) {
    const O &row = object(value);
    int64_t factor = integer(row, "factor"), axis = integer(row, "axis");
    if (factor == 1 || !flag(row, "resource_checked") ||
        integer(original[axis]) % factor)
      continue;
    if (get(row, "mode") == get(baseline, "mode") &&
        get(row, "cube_core_count") == get(baseline, "cube_core_count") &&
        get(row, "useful_flops") == get(baseline, "useful_flops"))
      eligible.push_back(key.str());
  }
  auto order = [&](const std::string &key) {
    const O &r = sub(records, key);
    return std::make_tuple(number(r, "estimated_cycles"), integer(r, "factor"),
                           integer(r, "axis"));
  };
  std::string selected = "unmerged";
  for (const auto &key : eligible)
    if (order(key) < order(selected))
      selected = key;
  std::string reason = eligible.empty()         ? "MERGED_RESOURCE_UNAVAILABLE"
                       : selected == "unmerged" ? "ORIGINAL_LOWEST_LATENCY"
                                                : "LOWER_ESTIMATED_LATENCY";
  bool spatial = std::any_of(records.begin(), records.end(), [](const auto &v) {
    auto kind = str(object(v.second), "merge_kind");
    return kind == "M" || kind == "N";
  });
  if (contract.getBoolean("cv_empirical_short_batch").value_or(true) &&
      product(original) <= integer(baseline, "cube_core_count") && !spatial) {
    std::vector<std::pair<int64_t, std::string>> preferred;
    for (const auto &key : eligible) {
      const O &row = sub(records, key);
      int64_t axis = integer(row, "axis");
      bool single = true;
      for (int64_t i = 0; i < 3; ++i)
        if (i != axis && integer(original[i]) != 1)
          single = false;
      if (str(row, "merge_kind") == "independent_batch" &&
          integer(row, "factor") == 2 && single)
        preferred.emplace_back(axis, key);
    }
    if (!preferred.empty()) {
      selected = std::min_element(preferred.begin(), preferred.end())->second;
      reason = "EMPIRICAL_SINGLE_TILE_SHORT_BATCH_X2";
    }
  }
  return O{
      {"selected_key", selected}, {"reason", reason}, {"model_complete", true}};
}
} // namespace mlir::triton::ascend::blockauto
