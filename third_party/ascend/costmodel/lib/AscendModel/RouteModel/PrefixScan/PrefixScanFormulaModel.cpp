#include "AscendModel/RouteModel/PrefixScan/PrefixScanFormulaModel.h"
#include "AscendModel/RouteModel/StageCostModels.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

using namespace mlir::ascend;

namespace {
const std::set<std::string> featureNames = {"one",
                                            "E",
                                            "C",
                                            "logE",
                                            "logW",
                                            "invW",
                                            "f16",
                                            "B",
                                            "chain",
                                            "vecwork",
                                            "payload",
                                            "tileR",
                                            "paddedwork",
                                            "doubling",
                                            "treework",
                                            "q",
                                            "H",
                                            "axisK",
                                            "local",
                                            "warp",
                                            "merge",
                                            "combined",
                                            "shared",
                                            "updates",
                                            "N",
                                            "Ba",
                                            "P",
                                            "blocks",
                                            "rounds",
                                            "live_words",
                                            "partial_values",
                                            "pressure24",
                                            "pressure48",
                                            "pressure127",
                                            "pressure2_8",
                                            "pressure2_16",
                                            "pressure2_24",
                                            "pressure2_48",
                                            "pressure2_64",
                                            "pressure2_96",
                                            "partial_pressure8",
                                            "partial_pressure16",
                                            "partial_pressure2_cta",
                                            "scan_spill_warp_work",
                                            "scan_spill_subwarp_work",
                                            "direct_group_work",
                                            "direct_narrow_group_work",
                                            "direct_half_row_groups"};
const std::set<std::string> branches = {"short",
                                        "long",
                                        "tree",
                                        "chain",
                                        "direct",
                                        "direct_full_groups",
                                        "transpose",
                                        "product_multi",
                                        "scalar_or_generic",
                                        "warp_local",
                                        "cross_warp",
                                        "multi_element",
                                        "multi",
                                        "layout_any",
                                        "layout_warp_local",
                                        "layout_cross_warp",
                                        "layout_cross_warp_full_lane",
                                        "layout_cross_warp_subwarp",
                                        "layout_multi_block"};
double ceilDiv(double a, double b) { return std::ceil(a / b); }
} // namespace

llvm::Expected<std::vector<PrefixScanFormula>>
mlir::ascend::parsePrefixScanFormulas(const llvm::json::Array &array) {
  std::vector<PrefixScanFormula> result;
  std::set<std::string> ids;
  auto invalid = [] {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "invalid prefixscan formula asset");
  };
  for (const auto &entry : array) {
    auto *object = entry.getAsObject();
    if (!object)
      return invalid();
    auto id = object->getString("id");
    auto *terms = object->getObject("terms");
    auto *domains = object->getArray("domains");
    if (!id || id->empty() || !terms || terms->empty() || !domains ||
        domains->empty() || !ids.insert(id->str()).second)
      return invalid();
    PrefixScanFormula formula;
    formula.id = id->str();
    if (auto projection = object->getString("projection")) {
      if (*projection != "legacy" && *projection != "row_major")
        return invalid();
      formula.projection = projection->str();
    } else if (object->get("projection"))
      return invalid();
    for (const auto &term : *terms) {
      auto value = term.second.getAsNumber();
      if (!value || !std::isfinite(*value) ||
          !featureNames.count(term.first.str()))
        return invalid();
      formula.terms[term.first.str()] = *value;
    }
    // Expand compact alternatives once at load time; matching stays scalar.
    llvm::json::Array expanded;
    for (const auto &value : *domains) {
      const auto *domain = value.getAsObject();
      if (!domain)
        return invalid();
      llvm::json::Array alternatives;
      alternatives.push_back(llvm::json::Object(*domain));
      for (const char *field : {"operation", "data_type", "warps"}) {
        const auto *values = domain->getArray(field);
        if (!values)
          continue;
        if (values->empty())
          return invalid();
        llvm::json::Array next;
        for (const auto &alternative : alternatives)
          for (const auto &item : *values) {
            llvm::json::Object copy(*alternative.getAsObject());
            copy[field] = item;
            next.push_back(std::move(copy));
          }
        alternatives = std::move(next);
      }
      for (auto &alternative : alternatives)
        expanded.push_back(std::move(alternative));
    }
    for (const auto &value : expanded) {
      auto *domain = value.getAsObject();
      if (!domain)
        return invalid();
      auto op = domain->getString("operation");
      auto type = domain->getString("data_type");
      auto branch = domain->getString("branch");
      auto rank = domain->getInteger("rank"), axis = domain->getInteger("axis");
      auto warps = domain->getInteger("warps");
      if (!op || !type || !branch || !rank || !axis || !warps || *rank < 1 ||
          *axis < 0 || *axis >= *rank || *warps < 0 || *warps > 64 ||
          !branches.count(branch->str()) ||
          (branch->starts_with("layout_") &&
           formula.projection != "row_major") ||
          (*op != "cumprod" && *op != "cummax" && *op != "cummin") ||
          (*type != "f16" && *type != "f32" && *type != "i32" &&
           *type != "u32" && *type != "i64" && *type != "u64" &&
           *type != "integer32" && *type != "integer64"))
        return invalid();
      formula.domains.push_back(
          {op->str(), type->str(), branch->str(), *rank, *axis, *warps});
    }
    result.push_back(std::move(formula));
  }
  return result;
}

std::optional<PrefixScanEstimate> mlir::ascend::estimatePrefixScanFormula(
    const StageWorkload::PrefixScanKey &scan, unsigned warps, bool simd,
    const std::vector<PrefixScanFormula> &formulas) {
  if (formulas.empty() || (!simd && (warps == 0 || warps > 64)) ||
      std::get<3>(scan) != 1 || !std::get<12>(scan))
    return std::nullopt;
  auto shape = std::get<10>(scan);
  int64_t axis = std::get<8>(scan);
  const int64_t bits = std::get<9>(scan);
  if (shape.empty() || axis < 0 || axis >= int64_t(shape.size()) || bits <= 0 ||
      bits > 128 ||
      std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d <= 0; }))
    return std::nullopt;
  const double E = std::get<0>(scan), C = std::get<1>(scan);
  if (E <= 1 || C < 1 || double(shape[axis]) != E)
    return std::nullopt;
  // Use the same aggregate per-AIV geometry as cumsum, keeping post fixed.
  const double post = std::get<2>(scan);
  const double elements = std::accumulate(shape.begin(), shape.end(), 1.0,
                                          std::multiplies<double>());
  if (simd) {
    if (elements / E != C)
      return std::nullopt;
  } else if (shape.size() == 1) {
    if (C != 1)
      return std::nullopt;
  } else if (elements / E != C) {
    // Preserve the existing aggregate per-AIV geometry. Resize one non-axis
    // dimension while retaining the remaining shape factors.
    const size_t adjusted = axis == 0 ? 1 : 0;
    double fixed = 1;
    for (size_t i = 0; i < shape.size(); ++i)
      if (i != adjusted && i != size_t(axis))
        fixed *= shape[i];
    const double extent = C / fixed;
    if (!std::isfinite(extent) || extent < 1 || std::floor(extent) != extent ||
        extent >= double(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    shape[adjusted] = int64_t(extent);
  }
  // Higher ranks use the agreed canonical scan geometry. Keep measured rank
  // two/three shapes intact so their within-axis layout factors are preserved.
  const bool foldedHigherRank = shape.size() > 3;
  if (foldedHigherRank) {
    double pre = 1, after = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i < size_t(axis))
        pre *= shape[i];
      if (i > size_t(axis))
        after *= shape[i];
    }
    if (!std::isfinite(pre) || !std::isfinite(after) ||
        pre >= double(std::numeric_limits<int64_t>::max()) ||
        after >= double(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    if (axis == 0) {
      shape = {int64_t(E), int64_t(after)};
      axis = 0;
    } else if (axis == int64_t(shape.size()) - 1) {
      shape = {int64_t(pre), int64_t(E)};
      axis = 1;
    } else {
      shape = {int64_t(pre), int64_t(E), int64_t(after)};
      axis = 1;
    }
  }
  const std::string &combine = std::get<11>(scan);
  std::string operation, type;
  const bool floating = combine == "arith.mulf" || combine == "arith.maxnumf" ||
                        combine == "arith.minnumf";
  // Calibration used tl.maximum/minimum with PropagateNan.NONE.
  // Propagating-NaN maximumf/minimumf are distinct and must fall back.
  if (combine == "arith.mulf" || combine == "arith.muli")
    operation = "cumprod";
  else if (combine == "arith.maxnumf" || combine == "arith.maxsi" ||
           combine == "arith.maxui")
    operation = "cummax";
  else if (combine == "arith.minnumf" || combine == "arith.minsi" ||
           combine == "arith.minui")
    operation = "cummin";
  else
    return std::nullopt;
  if (floating) {
    // bf16/f16 have equal storage width but are not the same measured dtype.
    type = std::get<6>(scan);
    if (type != "f16" && type != "f32")
      return std::nullopt;
  } else if (combine == "arith.muli")
    type = "integer" + std::to_string(bits);
  else
    type = std::string(combine == "arith.maxui" || combine == "arith.minui"
                           ? "u"
                           : "i") +
           std::to_string(bits);
  const double W = warps, N = E * C;
  double pre = 1;
  for (int64_t i = 0; i < axis; ++i)
    pre *= shape[i];
  std::map<std::string, double> f = {{"one", 1},
                                     {"E", E},
                                     {"C", C},
                                     {"logE", std::log2(E)},
                                     {"f16", type == "f16"}};
  f["N"] = N;
  std::map<std::string, double> dense;
  std::string layoutBranch;
  if (simd) {
    f["B"] = ceilDiv(E, 64);
    f["chain"] = C * (E - 1);
    f["vecwork"] = (E - 1) * ceilDiv(C, 64);
    f["payload"] = 4 * N;
    f["tileR"] = ceilDiv(pre, 4) * ceilDiv(post, 64) * (E - 1);
    const double groups = std::floor(pre / 4);
    f["direct_group_work"] = groups * ceilDiv(post, 64) * (E - 1);
    // Quarter-register rows in a retained outer group loop require additional
    // vector-store ordering in the calibrated lowering. Wider rows do not
    // share this cost, even when their element count is larger.
    f["direct_narrow_group_work"] =
        groups > 1 && post < 32 ? f["direct_group_work"] : 0;
    f["direct_half_row_groups"] =
        groups > 1 && post >= 32 && post < 64 ? groups : 0;
    f["paddedwork"] = (E - 1) * ceilDiv(8 * ceilDiv(C, 8), 64);
    f["doubling"] = 0;
    f["treework"] = 0;
    for (double step = 1; step < E; step *= 2) {
      f["doubling"] += std::max(0.0, std::floor(E / 64) - ceilDiv(step, 64));
      f["treework"] +=
          ceilDiv(C, 64) * (std::floor(E / (2 * step)) * step +
                            std::max(0.0, std::fmod(E, 2 * step) - step));
    }
  } else {
    auto projectLayout = [&](bool rowMajor,
                             std::map<std::string, double> &values) {
      double A, L, K, Ba, P = 1, Bp = 1;
      if (shape.size() == 1) {
        A = std::min(std::floor(128.0 / bits),
                     std::max(std::floor(E / (32 * W)), 1.0));
        L = std::min(32.0, std::max(std::floor(E / A), 1.0));
        K = std::min(W, std::max(std::floor(E / (A * L)), 1.0));
      } else {
        std::vector<double> s(shape.size(), 1), t(shape.size()),
            v(shape.size());
        double vector = std::min(std::floor(128.0 / bits),
                                 std::max(std::floor(N / (32 * W)), 1.0));
        if (rowMajor) {
          for (int i = int(shape.size()) - 1; i >= 0; --i) {
            s[i] = std::min(vector, double(shape[i]));
            vector = std::floor(vector / s[i]);
          }
        } else
          s.back() = std::min(vector, double(shape.back()));
        double remaining = 32;
        for (int i = int(shape.size()) - 1; i >= 0; --i) {
          t[i] = i == 0 ? remaining
                        : std::min(remaining,
                                   std::max(std::floor(shape[i] / s[i]), 1.0));
          remaining = std::floor(remaining / t[i]);
        }
        remaining = W;
        for (int i = int(shape.size()) - 1; i >= 0; --i) {
          v[i] = i == 0
                     ? remaining
                     : std::min(
                           remaining,
                           std::max(std::floor(shape[i] / (s[i] * t[i])), 1.0));
          remaining = std::floor(remaining / v[i]);
        }
        A = s[axis];
        L = std::min(t[axis], std::max(std::floor(E / A), 1.0));
        K = std::min(v[axis], std::max(std::floor(E / (A * L)), 1.0));
        for (size_t i = 0; i < shape.size(); ++i)
          if (i != size_t(axis)) {
            P *= s[i];
            Bp *= ceilDiv(shape[i], s[i] * t[i] * v[i]);
          }
      }
      Ba = ceilDiv(E, A * L * K);
      const double H = Ba * Bp * P;
      values["logW"] = std::log2(W);
      values["invW"] = 1 / W;
      values["q"] = E / (32 * W);
      values["H"] = H;
      values["axisK"] = K;
      values["axis_lanes"] = L;
      values["local"] = H * (A - 1);
      values["warp"] = H * std::log2(L);
      values["merge"] = P * Bp * (K * Ba - 1);
      values["shared"] = K > 1 ? H * K : 0;
      values["updates"] = H * A;
      values["combined"] = values["local"] + values["warp"] + values["merge"];
      values["Ba"] = Ba;
      values["P"] = P;
      values["blocks"] = Ba * Bp;
      values["rounds"] = std::ceil(std::log2(K));
      const double words = std::ceil(double(bits) / 32);
      values["live_words"] = (H * A + P * Bp) * words;
      values["partial_values"] = H * K * words;
      // Soft pressure proxy beyond one CTA-sized set of scan partials.
      const double partialExcess =
          std::max(0.0, values["partial_values"] - 32.0 * W);
      values["partial_pressure2_cta"] = partialExcess * partialExcess;
      values["pressure127"] = std::max(0.0, values["live_words"] - 127.0);
      // Approximate live input/output words against the per-thread budget.
      const double registerWords = 32 * W <= 256    ? 127
                                   : 32 * W <= 512  ? 64
                                   : 32 * W <= 1024 ? 32
                                                    : 16;
      const double valueExcess =
          std::max(0.0, 2.0 * H * A * words - registerWords);
      values["scan_spill_warp_work"] = valueExcess * std::log2(L);
      values["scan_spill_subwarp_work"] =
          valueExcess * std::max(0.0, 32.0 / L - 4.0);
      for (int threshold : {8, 16, 24, 32, 48, 64, 96}) {
        double excess = std::max(0.0, values["live_words"] - threshold);
        values["pressure" + std::to_string(threshold)] = excess;
        values["pressure2_" + std::to_string(threshold)] = excess * excess;
      }
      for (int reserve : {8, 16})
        values["partial_pressure" + std::to_string(reserve)] = std::max(
            0.0, values["partial_values"] -
                     std::max(1.0, std::min(128.0, 1024 / W) - reserve));
      if (rowMajor)
        layoutBranch = K > 1    ? "layout_cross_warp"
                       : Ba > 1 ? "layout_multi_block"
                                : "layout_warp_local";
      // Full-lane pricing is enabled only by a matching calibrated domain.
      if (rowMajor && K > 1 && L == 32) {
        const bool fullLaneAsset =
            llvm::any_of(formulas, [&](const auto &formula) {
              return llvm::any_of(formula.domains, [&](const auto &d) {
                return d.branch == "layout_cross_warp_full_lane" &&
                       d.operation == operation && d.dataType == type &&
                       d.rank == int64_t(shape.size()) && d.axis == axis &&
                       (d.warps == 0 || d.warps == warps);
              });
            });
        if (fullLaneAsset)
          layoutBranch = "layout_cross_warp_full_lane";
      }
    };
    dense = f;
    projectLayout(false, f);
    projectLayout(true, dense);
  }
  std::string branch;
  if (simd) {
    if (type == "i64" || type == "u64" || type == "integer64" ||
        (type == "u32" && operation != "cumprod"))
      branch = "scalar_or_generic";
    else if (C == 1)
      branch = E <= 128 ? "short" : "long";
    else if (operation != "cumprod" &&
             (axis == 0 || axis == int64_t(shape.size()) - 1))
      branch = E >= 8 && ceilDiv(C, 64) <= 2 ? "tree" : "chain";
    else if (operation != "cumprod")
      branch = post >= 16 ? "direct" : "transpose";
    else
      branch = "product_multi";
  } else if (shape.size() == 1) {
    branch = E <= 32                ? "warp_local"
             : W > 1 && E <= 32 * W ? "cross_warp"
                                    : "multi_element";
  } else
    branch = "multi";
  // A lane-specific calibrated domain takes precedence over a coarse domain.
  if (!simd && branch == "multi" && dense["axisK"] > 1 &&
      dense["axis_lanes"] < 32) {
    const bool subwarpAsset = llvm::any_of(formulas, [&](const auto &formula) {
      return llvm::any_of(formula.domains, [&](const auto &d) {
        return d.branch == "layout_cross_warp_subwarp" &&
               d.operation == operation && d.dataType == type &&
               d.rank == int64_t(shape.size()) && d.axis == axis &&
               (d.warps == 0 || d.warps == warps);
      });
    });
    if (subwarpAsset)
      branch = layoutBranch = "layout_cross_warp_subwarp";
  }
  // A qualified asset may refine the direct path with complete groups of
  // four rows. Preserve older assets and remainder-row
  // paths; the refined domain takes explicit priority, never JSON order.
  if (simd && branch == "direct" && pre >= 4 && std::fmod(pre, 4) == 0) {
    const bool hasFullGroups = llvm::any_of(formulas, [&](const auto &formula) {
      return llvm::any_of(formula.domains, [&](const auto &d) {
        return d.operation == operation && d.dataType == type &&
               d.rank == int64_t(shape.size()) && d.axis == axis &&
               d.branch == "direct_full_groups" &&
               (d.warps == 0 || d.warps == warps);
      });
    });
    if (hasFullGroups)
      branch = "direct_full_groups";
  }
  auto matchesFeatures = [&](const PrefixScanFormulaDomain &d) {
    return d.operation == operation && d.dataType == type &&
           (d.branch == branch || (!simd && (d.branch == layoutBranch ||
                                             d.branch == "layout_any"))) &&
           (d.warps == 0 || d.warps == warps);
  };
  const bool directDomain =
      std::any_of(formulas.begin(), formulas.end(), [&](const auto &formula) {
        return std::any_of(
            formula.domains.begin(), formula.domains.end(), [&](const auto &d) {
              return matchesFeatures(d) && d.rank == int64_t(shape.size()) &&
                     d.axis == axis;
            });
      });
  std::optional<PrefixScanEstimate> result;
  for (const auto &formula : formulas) {
    bool matches = false;
    for (const auto &d : formula.domains) {
      const bool canonical = d.rank == int64_t(shape.size()) && d.axis == axis;
      // Some head/tail layout regimes were measured only with rank three.
      // A folded high-rank edge uses that family only if no rank-two domain
      // exists, so selection is independent of asset ordering.
      const bool edgeFamily = foldedHigherRank && !directDomain &&
                              shape.size() == 2 && d.rank == 3 &&
                              d.axis == (axis == 0 ? 0 : 2);
      matches |= matchesFeatures(d) && (canonical || edgeFamily);
    }
    if (!matches)
      continue;
    const auto &values = !simd && formula.projection == "row_major" ? dense : f;
    // Mode-incompatible assets are rejected without evaluating absent features.
    if (std::any_of(
            formula.terms.begin(), formula.terms.end(),
            [&](const auto &term) { return !values.count(term.first); }) ||
        (simd && std::any_of(formula.domains.begin(), formula.domains.end(),
                             [](const auto &d) { return d.warps != 0; })))
      return std::nullopt;
    // Conflicting IDs cannot be selected by JSON order or measurement labels.
    if (result)
      return std::nullopt;
    double cycles = 0;
    for (const auto &[name, coefficient] : formula.terms)
      cycles += coefficient * values.at(name);
    if (!std::isfinite(cycles) || cycles <= 0)
      return std::nullopt;
    result = PrefixScanEstimate{formula.id, cycles};
  }
  return result;
}

static PrefixScanEstimate
simtPrefixScan1dSegmentCost(const CalibratedPrefixScanProfile &simt,
                            int64_t numWarps, double n) {
  const auto &segments = simt.prefixScan1dWarpSegments;
  auto it = segments.lower_bound(
      static_cast<unsigned>(std::max<int64_t>(1, numWarps)));
  if (it == segments.end())
    it = std::prev(segments.end());
  const auto &segment = it->second;
  const std::string id = "cumsum_simt_1d_w" + std::to_string(it->first);
  const double warps = static_cast<double>(it->first);
  const double threads = 32.0 * warps;
  if (n <= 32.0)
    return {id + "_local", segment.cLocal};
  // warpsPerCTA[axis] = ceil(N/32), capped by the kernel warp count.
  const double chainWarps = std::min(warps, std::ceil(n / 32.0));
  // Cross-warp Sklansky merge.  Both gates mirror canUseMultiWarpSklansky:
  // one element per thread (N <= 32*w) and at most warpSize axis warps
  // (k <= 32, i.e. N <= 1024).  A kernel with more warps than that
  // (num_warps = 64) drops to the general path already at N = 2048 although
  // every thread still owns one element, so pricing it as Sklansky would
  // under-predict it.  ceil(log2 k) is the trip count of the Sklansky
  // `for (h = 1; h < k; h <<= 1)` loop, so the pricing tracks the real round
  // count instead of a linear proxy.
  if (n <= threads && chainWarps <= 32.0) {
    const double rounds = std::ceil(std::log2(std::max(1.0, chainWarps)));
    return {id + "_cross_warp", segment.cFixed + segment.cRound * rounds};
  }
  // General path.  The per-element serial slope is charged from N = 0; the
  // 32*w anchor of the (N - 32*w) form is folded into cGen, so cGen is not
  // directly comparable with the cross-warp segment constant.
  return {id + "_general", segment.cGen + segment.serialRate * n};
}

// SIMT cumsum segmented fit for multidimensional scans. The generic scan
// lowering (AddPartialReduce) distributes the scan across warps with a serial
// UB load chain whose cost is additive in the total element count N, plus a
// per-thread-element hinge that prices register pressure beyond r0.  E is the
// aggregate scan extent (exact for a single multi-dim scan per Stage) and
// determines k = warpsPerCTA[axis], the UB chain length.  Extents below one
// warp (E < 32) take the separate linear regime described inline below.
static PrefixScanEstimate
simtPrefixScan2dSegmentCost(const CalibratedPrefixScanProfile &simt,
                            int64_t numWarps, double extentSum, double n) {
  const auto &segments = simt.prefixScan2dWarpSegments;
  auto it = segments.lower_bound(
      static_cast<unsigned>(std::max<int64_t>(1, numWarps)));
  if (it == segments.end())
    it = std::prev(segments.end());
  const auto &segment = it->second;
  const std::string id = "cumsum_simt_multi_w" + std::to_string(it->first);
  const double extent = std::max(1.0, extentSum);
  // Sub-warp extent: the warp owning the axis holds fewer than 32 valid lanes,
  // so the axis warp never needs a cross-warp chain (k = 1) and the cost
  // collapses onto a single linear-in-N regime.  Layout-coalesced small scans
  // (e.g. a 16-element cumsum promoted to an (8,16,16) scan) land here.
  if (extent < 32.0 && (segment.aSubwarp > 0.0 || segment.bSubwarp > 0.0))
    return {id + "_subwarp", segment.aSubwarp + segment.bSubwarp * n};
  const double threads = 32.0 * static_cast<double>(it->first);
  // warpsPerCTA[axis]: warps holding unique data along the scan axis.
  const double chainWarps =
      std::min(static_cast<double>(it->first), std::ceil(extent / 32.0));
  const double chainLoads = n * std::max(1.0, chainWarps) / threads;
  const double hinge = std::max(0.0, n / threads - segment.r0);
  return {id + "_general", segment.a + segment.b * n + segment.c * chainLoads +
                               segment.d * hinge};
}

// Return the complete cost of one scan. A miss leaves all original resource
// and special-stage accounting to the caller.
std::optional<PrefixScanEstimate>
mlir::ascend::estimateFittedScan(const StageWorkload::PrefixScanKey &shape,
                                 const StageModeProfile &profile, bool simd,
                                 bool allowTableFormulas) {
  if (std::get<0>(shape) <= 1 || std::get<1>(shape) <= 0 ||
      !std::get<12>(shape))
    return std::nullopt;
  const auto &combine = std::get<11>(shape);
  if (combine != "arith.addf" && combine != "arith.addi") {
    if (allowTableFormulas)
      return estimatePrefixScanFormula(
          shape,
          profile.prefixScanWarpCount > 0 ? profile.prefixScanWarpCount
                                          : profile.activeWarpCount,
          simd, profile.calibratedPrefixScan.formulas);
    return std::nullopt;
  }
  const auto &calibration = profile.calibratedPrefixScan;
  if (!calibration.enabled || std::get<7>(shape) <= 0)
    return std::nullopt;
  const int64_t extent = std::get<0>(shape);
  const int64_t independentSequences = std::get<1>(shape);
  const double n = static_cast<double>(extent) * independentSequences;
  double completion = 0.0;
  std::string formulaId;
  if (simd && independentSequences == 1) {
    formulaId = extent <= 64     ? "cumsum_simd_1d_small"
                : extent <= 1024 ? "cumsum_simd_1d_mid"
                                 : "cumsum_simd_1d_long";
    if (extent <= 64)
      completion =
          calibration.prefixScanStartupCycles1dSmall +
          extent * calibration.prefixScanDependencyFactor1dSmall / 64.0;
    else
      completion =
          calibration.prefixScanStartupCycles1d +
          extent * std::log2(static_cast<double>(extent)) *
              (extent <= 1024
                   ? calibration.prefixScanDependencyFactor1dMidWork
                   : calibration.prefixScanDependencyFactor1dTiledWork) /
              64.0;
  } else if (simd) {
    const int64_t elementBits = std::get<9>(shape);
    const double elementBytes = elementBits >= 8 ? elementBits / 8 : 0;
    const double vectorElements =
        elementBytes > 0.0 ? 256.0 / elementBytes : 64.0;
    const double steps =
        extent * std::ceil(independentSequences / vectorElements);
    formulaId = steps <= 256 ? "cumsum_simd_multi" : "cumsum_simd_multi_tail";
    completion = calibration.prefixScanStartupCycles +
                 (steps * calibration.prefixScanDependencyFactor +
                  std::max(0.0, steps - 256.0) *
                      calibration.prefixScanDependencyFactorMultiTail) /
                     64.0;
    if (std::get<8>(shape) != 0 && elementBytes > 0.0) {
      const bool rank2 = std::get<7>(shape) == 2;
      const double rate =
          rank2 ? calibration.prefixScanTransposeAr2raBytesPerCycle
                : calibration.prefixScanTransposeDim01BytesPerCycle;
      if (rate > 0.0) {
        formulaId += rank2 ? "_transpose_ar2ra" : "_transpose_dim01";
        completion +=
            (rank2 ? calibration.prefixScanTransposeAr2raFixedCycles
                   : calibration.prefixScanTransposeDim01FixedCycles) +
            2.0 * n * elementBytes / rate;
      }
    }
  } else if (independentSequences == 1 &&
             !calibration.prefixScan1dWarpSegments.empty()) {
    auto result =
        simtPrefixScan1dSegmentCost(calibration, profile.activeWarpCount, n);
    formulaId = std::move(result.formulaId);
    completion = result.cycles;
  } else if (independentSequences > 1 &&
             !calibration.prefixScan2dWarpSegments.empty()) {
    auto result = simtPrefixScan2dSegmentCost(
        calibration, profile.activeWarpCount, extent, n);
    formulaId = std::move(result.formulaId);
    completion = result.cycles;
  }
  if (!std::isfinite(completion) || completion <= 0.0)
    return std::nullopt;
  return PrefixScanEstimate{std::move(formulaId), completion};
}

bool CalibratedPrefixScanProfile::isValid() const {
  auto valid = [](std::initializer_list<double> values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
      return std::isfinite(value) && value >= 0.0;
    });
  };
  if (!valid({prefixScanDependencyFactor, prefixScanDependencyFactor1d,
              prefixScanDependencyFactor1dSmall,
              prefixScanDependencyFactor1dMidWork,
              prefixScanDependencyFactor1dTiledWork,
              prefixScanDependencyFactorMultiTail,
              prefixScanTransposeDim01FixedCycles,
              prefixScanTransposeDim01BytesPerCycle,
              prefixScanTransposeAr2raFixedCycles,
              prefixScanTransposeAr2raBytesPerCycle, prefixScanStartupCycles,
              prefixScanStartupCycles1d, prefixScanStartupCycles1dSmall}))
    return false;
  for (const auto &[warps, s] : prefixScan1dWarpSegments)
    if (warps == 0 ||
        !valid({s.cLocal, s.cFixed, s.cRound, s.cGen, s.serialRate}))
      return false;
  for (const auto &[warps, s] : prefixScan2dWarpSegments)
    if (warps == 0 ||
        !valid({s.a, s.b, s.c, s.d, s.r0, s.aSubwarp, s.bSubwarp}))
      return false;
  return true;
}
