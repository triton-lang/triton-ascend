#include "AscendModel/RouteModel/Models/PartialStructuredStaticModel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <set>
#include <string_view>

using namespace mlir::ascend::partial_memory;
namespace {
double lg(double x) { return std::log2(std::max(x, 1.0)); }
int64_t reverse(int64_t j, int64_t n) {
  int64_t value = 0;
  for (; n > 1; n >>= 1, j >>= 1)
    value = (value << 1) | (j & 1);
  return value;
}
bool simd(Model m) { return m == Model::SimdLoad || m == Model::SimdStore; }
bool store(Model m) { return m == Model::SimdStore || m == Model::SimtStore; }
struct StaticDecision {
  const char *feature;
  double threshold;
  int left, right;
};
struct StaticFormula {
  std::array<const char *, 10> terms;
  std::array<double, 10> coefficients;
  size_t count;
};
struct StaticCalibration {
  const char *key;
  double alpha, beta;
};
struct StaticKey {
  const char *key;
  int group;
};
struct StaticModel {
  const char *name;
  bool power, tree;
  const StaticDecision *decisions;
  const StaticFormula *formulas;
  size_t count;
  const StaticCalibration *calibration;
  size_t calibrationCount;
  const StaticKey *keys;
  size_t keyCount;
  std::array<const char *, 4> keyFeatures;
  size_t keyFeatureCount;
  int fallback;
  std::array<const char *, 64> referencePatterns;
  size_t referenceCount;
};
#include "PartialStructuredStaticModels.inc"
bool supported(const FactorLoadGeometry &c, Model m) {
  const auto &g = c.geometry;
  if (static_cast<unsigned>(m) > 3 || g.axes.empty() || g.axes.size() > 5 ||
      c.lower.size() != g.axes.size() || c.upper.size() != g.axes.size() ||
      (c.factor != 1 && c.factor != 2 && c.factor != 4) ||
      (simd(m) && c.factor != 1) || g.warps < 1 || (g.warps & (g.warps - 1)) ||
      g.warps > 64 / c.factor ||
      (g.bits != 8 && g.bits != 16 && g.bits != 32 && g.bits != 64))
    return false;
  int64_t E = 1;
  unsigned d = 0;
  for (size_t i = 0; i < g.axes.size(); ++i) {
    const auto &a = g.axes[i];
    if (a.extent < 2 || (a.extent & (a.extent - 1)) || E > 2048 / a.extent ||
        a.pitch <= 0 || a.pitch > (int64_t(1) << 31) || c.lower[i] < 0 ||
        c.upper[i] > a.extent || c.lower[i] >= c.upper[i])
      return false;
    E *= a.extent;
    d += a.discrete;
  }
  if (E < 32 || !d || (g.axes.size() > 1 && d == g.axes.size()))
    return false;
  if (c.factor > 1) {
    if (g.axes.size() < 2)
      return false;
    // The complete-group panel uses disjoint, aligned program workspaces.
    const int64_t b = g.bits / 8;
    if (c.programStride <= 0 || c.programStride % (256 / b) ||
        c.programStride > (16 * 1024 * 1024) / (c.factor * b))
      return false;
    int64_t span = 1;
    for (const auto &a : g.axes) {
      const int64_t maximum = (a.discrete ? 4 * a.extent : a.extent) - 1;
      if (span > c.programStride ||
          a.pitch > (c.programStride - span) / maximum)
        return false;
      span += maximum * a.pitch;
    }
  }
  // F1 panels cover quarter, half and three-quarter prefix rectangles.
  // Nonzero lower bounds are restricted to the measured tail half-window.
  if (c.factor == 1) {
    bool prefix = true;
    for (size_t i = 0; i < g.axes.size(); ++i) {
      const int64_t n = g.axes[i].extent, hi = c.upper[i];
      prefix &= c.lower[i] == 0 &&
                (hi == n || 4 * hi == n || 2 * hi == n || 4 * hi == 3 * n);
    }
    if (prefix)
      return true;
  }
  // Do not extrapolate arbitrary mask density or window offsets.
  for (int kind = 0; kind < 5; ++kind) {
    bool match = true;
    for (size_t i = 0; i < g.axes.size(); ++i) {
      int64_t lo = 0, hi = g.axes[i].extent;
      if ((kind == 1 || kind == 3) && i == 0)
        hi /= 2;
      if ((kind == 2 || kind == 3) && i + 1 == g.axes.size())
        hi /= 2;
      if (kind == 4 && i + 1 == g.axes.size()) {
        lo = hi / 4;
        hi = 3 * hi / 4;
      }
      match &= c.lower[i] == lo && c.upper[i] == hi;
    }
    if (match)
      return true;
  }
  return false;
}
// All enumerated coordinates are logical positions, NOT loaded index values.
struct Position {
  int64_t off = 0, soff = 0, dkey = 0;
  bool active = true;
};
std::vector<Position> positions(const FactorLoadGeometry &c,
                                const std::vector<size_t> &order,
                                int permutation = 0) {
  int64_t E = 1;
  for (const auto &a : c.geometry.axes)
    E *= a.extent;
  std::vector<Position> out(E);
  for (int64_t k = 0; k < E; ++k) {
    int64_t rest = k;
    std::vector<int64_t> j(order.size());
    for (size_t t = order.size(); t-- > 0;) {
      size_t i = order[t];
      j[i] = rest % c.geometry.axes[i].extent;
      rest /= c.geometry.axes[i].extent;
    }
    for (size_t i = 0; i < j.size(); ++i) {
      const auto &a = c.geometry.axes[i];
      int64_t v = j[i];
      out[k].active &= v >= c.lower[i] && v < c.upper[i];
      if (a.discrete) {
        out[k].dkey = out[k].dkey * a.extent + v;
        if (permutation == 1)
          v = reverse(v, a.extent);
        if (permutation == 2) {
          int64_t p = std::min<int64_t>(4, a.extent);
          v = reverse(v / p, a.extent / p) * p + v % p;
        }
      }
      int64_t term = v * a.pitch * (c.geometry.bits / 8);
      out[k].off += term;
      if (!a.discrete)
        out[k].soff += term;
    }
  }
  return out;
}
struct WaveWork {
  double groups = 0, partial = 0, D = 0, H = 0, G = 0;
};
WaveWork waves(const std::vector<Position> &v, int block = 128) {
  WaveWork w;
  for (size_t k = 0; k < v.size(); k += 32) {
    std::set<int64_t> d, g;
    std::set<std::pair<int64_t, int64_t>> h;
    int active = 0;
    for (size_t j = k; j < std::min(k + 32, v.size()); ++j)
      if (v[j].active) {
        ++active;
        d.insert(v[j].dkey);
        g.insert(v[j].off / block);
        h.insert({v[j].dkey, v[j].soff / block});
      }
    w.groups += active > 0;
    w.partial += active > 0 && active < 32;
    w.D += d.size();
    w.H += h.size();
    w.G += g.size();
  }
  return w;
}
} // namespace

Features
mlir::ascend::partial_memory::extractStaticFeatures(const FactorLoadGeometry &c,
                                                    Model m) {
  if (!supported(c, m))
    return {};
  const auto &g = c.geometry;
  Features out = extractFeatures(g);
  const double E = out.at("E"), W = g.warps, b = g.bits / 8, F = c.factor;
  const size_t rank = g.axes.size();
  int64_t K = 1, ad = 1, as = 1;
  double sspan = b;
  Geometry active = g;
  for (size_t i = 0; i < rank; ++i) {
    auto size = c.upper[i] - c.lower[i];
    K *= size;
    active.axes[i].extent = size;
    if (g.axes[i].discrete)
      ad *= size;
    else {
      as *= size;
      sspan += (g.axes[i].extent - 1) * g.axes[i].pitch * b;
    }
  }
  out["W"] = W;
  out["K"] = K;
  out["F"] = F;
  out["density"] = K / E;
  out["active_D_product"] = ad;
  out["active_S_product"] = as;
  out["mask_axes"] = 0;
  out["lower_bound_axes"] = 0;
  for (size_t i = 0; i < rank; ++i) {
    out["mask_axes"] += active.axes[i].extent != g.axes[i].extent;
    out["lower_bound_axes"] += c.lower[i] > 0;
  }
  out["tail_masked"] = active.axes.back().extent != g.axes.back().extent;
  if (c.factor == 1 && K == E)
    return out;
  std::vector<size_t> order(rank);
  std::iota(order.begin(), order.end(), 0);
  auto logical = positions(c, order);
  auto w = waves(logical);
  double H = 0;
  std::set<std::pair<int64_t, int64_t>> h;
  std::set<int64_t> soff, off, off32, rows;
  int runs = 0;
  bool previous = false;
  for (size_t k = 0; k < logical.size(); ++k) {
    const auto &v = logical[k];
    if (v.active) {
      h.insert({v.dkey, v.soff / 128});
      soff.insert(v.soff / 128);
      off.insert(v.off / 128);
      off32.insert(v.off / 32);
      rows.insert(k / g.axes.back().extent);
    }
    runs += v.active && !previous;
    previous = v.active;
  }
  H = h.size();
  const double I = std::ceil(E / 32), participating = std::min(I, W);
  double issueGroups = 0, issueH = 0, issueG = 0;
  for (int k = 0; k < std::max(I, W); ++k) {
    size_t start = (k % int(I)) * 32;
    std::vector<Position> wave(logical.begin() + start,
                               logical.begin() +
                                   std::min(start + 32, logical.size()));
    auto t = waves(wave);
    issueGroups += t.groups;
    issueH += t.H;
    issueG += t.G;
  }
  int64_t contiguous = 1;
  for (size_t i = rank; i-- > 0;) {
    if (g.axes[i].discrete || g.axes[i].pitch != contiguous)
      break;
    contiguous *= active.axes[i].extent;
  }
  if (c.factor > 1) {
    out = extractFactorReferenceFeatures(c);
    out["one"] = 1;
    out["E"] = E;
    out["W"] = W;
    out["F"] = F;
    out["K"] = K;
    out["bits"] = g.bits;
    out["logW"] = lg(W);
    out["S_span_log"] = lg(sspan);
    out["S_work"] = F * soff.size() * ad;
    out["J"] = F * w.groups;
    out["sqrtG"] = std::sqrt(F * off.size());
    out["sqrtG32"] = std::sqrt(F * off32.size());
    out["issue_J"] = F * issueGroups;
    out["sqrt_issue_H"] = std::sqrt(F * issueG);
    out["small_run32_work"] = g.bits < 32 && contiguous * b < 32 ? F * K : 0;
    return out;
  }
  if (K == E)
    return out;
  auto af = extractFeatures(active);
  double minD = 2048;
  for (const auto &a : g.axes)
    if (a.discrete)
      minD = std::min(minD, double(a.extent));
  out["log_min_D_n"] = lg(minD);
  Features base = out;
  for (const auto &kv : base)
    out["F_" + kv.first] = kv.second;
  for (const auto &kv : af)
    out["A_" + kv.first] = kv.second;
  for (const char *key :
       {"regular_rank", "tile_large", "tail_large", "all_regular_rank"})
    out[key] = af.at(key);
  out["I"] = I;
  out["H"] = H;
  out["G"] = w.G;
  out["sqrt_H"] = std::sqrt(H);
  out["sqrt_G"] = std::sqrt(w.G);
  out["active_rows"] = rows.size();
  out["row_work"] = rows.size() * g.axes.back().extent;
  out["row_bytes"] = g.axes.back().extent * b;
  out["active_row_bytes"] = active.axes.back().extent * b;
  out["log_K"] = lg(K);
  out["log_row_work"] = lg(out["row_work"]);
  out["mask_runs"] = runs;
  out["S_stride_bytes"] = g.axes.back().discrete ? 0 : g.axes.back().pitch * b;
  out["active_row_aligned32"] = int64_t(out["active_row_bytes"]) % 32 == 0;
  out["active_issue_groups"] = issueGroups;
  out["issued_H_wave_work"] = issueH;
  out["sqrt_I_H_wave"] = std::sqrt(I * w.H);
  out["participating_warps"] = participating;
  out["log_participating_warps"] = lg(participating);
  out["excess_warps"] = W - participating;
  out["remaining_issue_groups"] = I - participating;
  out["contiguous_large"] = contiguous * b >= 32;
  out["contiguous_work"] = contiguous * b >= 32 ? K / double(contiguous) : K;
  for (bool sfast : {true, false}) {
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) {
      return (g.axes[i].discrete == sfast) > (g.axes[j].discrete == sfast);
    });
    auto t = waves(positions(c, order));
    std::string p = sfast ? "S_fast_" : "D_fast_";
    out[p + "groups"] = t.groups;
    out[p + "G_work"] = t.G;
    out[p + "H_work"] = t.H;
    out[p + "D_work"] = t.D;
    out[p + "G_per_instruction"] = t.G / I;
    out[p + "H_per_instruction"] = t.H / I;
    out[p + "partial_fraction"] = t.partial / I;
  }
  std::iota(order.begin(), order.end(), 0);
  for (int mode : {1, 2}) {
    auto v = positions(c, order, mode);
    auto t = waves(v);
    std::string p = mode == 1 ? "reverse" : "page";
    out[p + "_waves128"] = t.G;
    out[p + "_sqrt_I_waves128"] = std::sqrt(I * t.G);
    for (int block : {32, 128}) {
      double issued = 0;
      for (int k = 0; k < std::max(I, W); ++k) {
        size_t start = (k % int(I)) * 32;
        std::vector<Position> wave(v.begin() + start,
                                   v.begin() + std::min(start + 32, v.size()));
        issued += waves(wave, block).G;
      }
      out[p + "_sqrt_issued_waves" + std::to_string(block)] = std::sqrt(issued);
    }
  }
  int64_t inner = 1;
  for (size_t i = rank; i-- > 0;) {
    int64_t lanes =
        std::min(g.axes[i].extent, std::max<int64_t>(1, 32 / inner));
    inner *= g.axes[i].extent;
    out["L_log_lanes" + std::to_string(i)] = lg(lanes);
    out["L_log_span" + std::to_string(i)] =
        lg((lanes - 1) * g.axes[i].pitch * b + b);
    out["P_gcd128_" + std::to_string(i)] =
        std::gcd(int64_t(g.axes[i].pitch * b), int64_t(128));
    out["P_mod128_" + std::to_string(i)] = int64_t(g.axes[i].pitch * b) % 128;
  }
  for (size_t i = rank; i < 5; ++i) {
    out["L_log_lanes" + std::to_string(i)] = 0;
    out["L_log_span" + std::to_string(i)] = 0;
  }
  return out;
}

std::optional<double>
mlir::ascend::partial_memory::predictStatic(const FactorLoadGeometry &c,
                                            Model kind) {
  auto f = extractStaticFeatures(c, kind);
  if (f.empty())
    return std::nullopt;
  double E = f.at("E"), K = f.at("K");
  bool masked = E != K;
  std::string pattern;
  for (auto a : c.geometry.axes)
    pattern += a.discrete ? 'D' : 'S';
  std::string name = std::string(simd(kind) ? "SIMD_" : "SIMT_") +
                     (store(kind) ? "store_" : "load_") +
                     (masked ? "masked_" : "unmasked_") +
                     (c.factor == 1 ? "F1" : "F24");
  const StaticModel *m = nullptr;
  for (const auto &item : staticModels)
    if (name == item.name)
      m = &item;
  if (!m)
    return std::nullopt;
  bool missing = false;
  auto feature = [&](const char *n) {
    auto i = f.find(n);
    if (i == f.end() || !std::isfinite(i->second)) {
      missing = true;
      return 0.;
    }
    return i->second;
  };
  int group = m->fallback;
  bool fallback = false;
  if (m->tree) {
    int node = 0;
    while (m->decisions[node].feature) {
      auto &d = m->decisions[node];
      node = feature(d.feature) <= d.threshold ? d.left : d.right;
    }
    group = -m->decisions[node].left;
  } else {
    std::string key;
    for (size_t i = 0; i < m->keyFeatureCount; ++i) {
      if (i)
        key += '_';
      key += std::to_string(int(feature(m->keyFeatures[i])));
    }
    fallback = true;
    for (size_t i = 0; i < m->keyCount; ++i)
      if (key == m->keys[i].key) {
        group = m->keys[i].group;
        fallback = false;
        break;
      }
  }
  if (group < 1 || size_t(group) > m->count)
    return std::nullopt;
  const auto &formula = m->formulas[group - 1];
  double value = 0;
  for (size_t i = 0; i < formula.count; ++i)
    value += feature(formula.terms[i]) * formula.coefficients[i];
  if (m->power)
    value = std::exp(std::clamp(value, -50., 50.));
  value = std::max(1., value);
  const std::string key = std::to_string(c.factor) + "|" + pattern,
                    star = std::to_string(c.factor) + "|*";
  const StaticCalibration *exact = nullptr, *general = nullptr;
  for (size_t i = 0; i < m->calibrationCount; ++i) {
    const auto &a = m->calibration[i];
    if (key == a.key)
      exact = &a;
    if (star == a.key)
      general = &a;
  }
  bool reference = false;
  for (size_t i = 0; i < m->referenceCount; ++i)
    reference |= pattern == m->referencePatterns[i];
  auto a = exact ? exact : (reference ? nullptr : general);
  double scale = a ? a->alpha + a->beta * (lg(E) - 8) : 1.;
  if (fallback)
    scale = 1;
  if (!std::isfinite(scale) || scale <= 0 || missing)
    return std::nullopt;
  value = std::max(1., value * scale);
  const int bits = c.geometry.bits, bi = bits == 8    ? 0
                                         : bits == 16 ? 1
                                         : bits == 32 ? 2
                                                      : 3;
  if (simd(kind) && masked && c.geometry.axes.size() == 2 && K / E == .5 &&
      f.at("mask_axes") == 1 && f.at("tail_masked") == 1 &&
      f.at("lower_bound_axes") == 1 &&
      (E == 32 || E == 64 || E == 128 || E == 512 || E == 2048) &&
      ((pattern == "DS" && std::abs(f.at("F_pitch1") - std::log2(3.)) < 1e-9) ||
       (pattern == "SD" &&
        std::abs(f.at("F_pitch0") - f.at("F_axis1") - f.at("F_pitch1") -
                 std::log2(12.)) < 1e-9))) {
    if (!store(kind)) {
      static const double p[4][4] = {{95.5250000000002, 6.519761061483759,
                                      12.073046874999998, 2.916911508129931},
                                     {-43.83333333333544, 3.1002604166666563,
                                      15.791666666666686, 53.27607654502429},
                                     {179.29166666666518, 1.9344075520833235,
                                      21.24772135416668, 26.470468696762556},
                                     {264.51190476190885, 1.3165922619047714,
                                      28.228422619047578, 7.686629518701458}};
      value =
          p[bi][0] + (pattern == "DS" ? p[bi][1] * E + p[bi][3] * std::sqrt(E)
                                      : p[bi][2] * E);
    } else if (pattern == "DS" && E <= 64)
      value = 284.3749999999998 + 43.13281250000001 * E;
    else {
      static const double p[4][4] = {
          {32.633630610924406, 52.86796613396849, 50.339094471463035, 128},
          {121.62499999999973, 61.899414062499986, 52.01513671875, 64},
          {951.5833333333345, 52.5390625, 54.40224202473958, 32},
          {475.9572784810127, 59.84379944620253, 62.342367051522935, 16}};
      double D = f.at("active_D_product"), ld = p[bi][3] * D;
      double sat = bi == 0 ? ld * (-std::expm1(-E / ld)) : E / (1 + E / ld);
      value = p[bi][0] + (pattern == "DS" ? p[bi][1] * sat : p[bi][2] * E);
    }
  }
  if (!simd(kind) && store(kind) && c.factor > 1 && E >= 1024 &&
      c.geometry.warps * c.factor == 64 && f.at("S_span_log") > 16) {
    static const double scale[4] = {0.5838640529604238, 0.15499014792707266,
                                    0.5142273527611588, 0.5719614175909105};
    if (pattern == "SD")
      value = std::max(1., value * scale[bi]);
    if (pattern == "SSD" && masked && c.factor == 2)
      value = std::max(1., value * 0.6217429478713083);
  }
  if (!simd(kind) && !store(kind) && c.factor > 1 && pattern == "DS" &&
      E == 256 && c.geometry.warps == 8)
    value = std::max(
        1., value * (c.factor == 2 ? 0.7661406868670503 : 0.593825125484509));
  return std::isfinite(value) ? std::optional<double>(std::max(1., value))
                              : std::nullopt;
}
