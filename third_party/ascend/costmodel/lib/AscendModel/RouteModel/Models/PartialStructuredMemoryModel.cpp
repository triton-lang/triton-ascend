// Deterministic TTIR geometry used by the active partial-memory formulas.
#include "AscendModel/RouteModel/Models/PartialStructuredMemoryModel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <set>

using namespace mlir::ascend::partial_memory;
namespace {
using Segment = std::pair<double, double>;
using Segments = std::vector<Segment>;
double lg(double x) { return std::log2(std::max(x, 1.0)); }
void appendSegment(Segments &s, const Axis &a) {
  if (!s.empty() && a.pitch == s.front().first * s.front().second)
    s.front().first *= a.extent;
  else
    s.insert(s.begin(), {double(a.extent), double(a.pitch)});
}
double tileElements(const Segments &s) {
  double r = 1;
  for (size_t i = s.size() > 3 ? s.size() - 3 : 0; i < s.size(); ++i)
    r *= s[i].first;
  return r;
}
} // namespace

Features mlir::ascend::partial_memory::extractFeatures(const Geometry &g) {
  const auto &a = g.axes;
  const size_t rank = a.size();
  if (!rank || rank > 5 || g.bits < 8 || g.warps <= 0)
    return {};
  const double b = g.bits / 8.0;
  double E = 1, dp = 1, sp = 1, de = 0, ss = 1, maxDP = 0, minDP = 1e300;
  double maxSS = 1, logicalSpan = b;
  int first = -1, last = -1, dc = 0, mask = 0;
  int64_t alignment = 0;
  for (size_t i = 0; i < rank; ++i) {
    if (a[i].extent <= 0 || a[i].pitch <= 0 || E > 2048 / a[i].extent)
      return {};
    E *= a[i].extent;
    logicalSpan += (a[i].extent - 1) * double(a[i].pitch) * b;
    alignment = std::gcd(alignment, a[i].pitch * (g.bits / 8));
    if (a[i].discrete) {
      ++dc;
      dp *= a[i].extent;
      de += a[i].extent;
      minDP = std::min(minDP, double(a[i].pitch));
      maxDP = std::max(maxDP, double(a[i].pitch));
      if (first < 0)
        first = i;
      last = i;
      mask |= 1 << i;
    } else {
      sp *= a[i].extent;
      ss += (a[i].extent - 1) * double(a[i].pitch);
      maxSS = std::max(maxSS, double(a[i].pitch) * b);
    }
  }
  if (first < 0)
    return {};
  const auto &tail = a.back();
  double suffix = 1, suffixSpan = b, C = 1;
  Segments regular, all;
  for (size_t i = rank; i-- > 0;) {
    if (a[i].discrete)
      break;
    suffix *= a[i].extent;
    suffixSpan += (a[i].extent - 1) * double(a[i].pitch) * b;
    appendSegment(regular, a[i]);
  }
  for (size_t i = rank; i-- > 0;) {
    if (a[i].discrete || a[i].pitch != C)
      break;
    C *= a[i].extent;
  }
  for (size_t i = rank; i-- > 0;)
    if (!a[i].discrete)
      appendSegment(all, a[i]);
  const double tile = tileElements(regular), allTile = tileElements(all);
  Segments t(regular.end() - std::min<size_t>(regular.size(), 3),
             regular.end());
  double minDim = 1;
  int minAxis = -1;
  if (!t.empty()) {
    auto it = std::min_element(t.begin(), t.end(), [](Segment x, Segment y) {
      return x.first < y.first;
    });
    minDim = it->first;
    minAxis = it - t.begin();
  }
  double inv = !tail.discrete && tail.pitch > 1 ? 1 / (tail.pitch * b) : 0;
  double lane128 = tail.discrete
                       ? E
                       : E / std::min({double(tail.extent), 32.0,
                                       std::max(1.0, 128 / (tail.pitch * b))});
  double lane32 = tail.discrete
                      ? E
                      : E / std::min({double(tail.extent), 32.0,
                                      std::max(1.0, 32 / (tail.pitch * b))});
  double dw = 1, inner = 1;
  for (size_t i = rank; i-- > 0;) {
    double lanes =
        std::min(double(a[i].extent), std::max(1.0, std::floor(32 / inner)));
    if (a[i].discrete)
      dw *= a[i].extent / lanes;
    inner *= a[i].extent;
  }
  Features f = {
      {"one", 1},
      {"E", E},
      {"sqrt_E", std::sqrt(E)},
      {"log_E", lg(E)},
      {"log_W", lg(g.warps)},
      {"inverse_W", 1.0 / g.warps},
      {"bits", double(g.bits)},
      {"rank", double(rank)},
      {"rows", E / tail.extent},
      {"D_count", double(dc)},
      {"D_mask", double(mask)},
      {"D_product", dp},
      {"D_elements", de},
      {"first_D", double(first)},
      {"last_D", double(last)},
      {"format_float", double(g.floatingPoint)},
      {"log_S_span", lg(ss)},
      {"log_D_pitch", lg(maxDP)},
      {"log_min_D_pitch", lg(minDP)},
      {"log_max_step", lg(maxSS)},
      {"C", C},
      {"E_over_C", E / C},
      {"suffix_rows", E / suffix},
      {"log_suffix", lg(suffix)},
      {"log_tail_n", lg(tail.extent)},
      {"tail_kind", tail.discrete     ? 0.0
                    : tail.pitch == 1 ? 1.0
                                      : 2.0},
      {"tail_large", double(tail.extent * b >= 32)},
      {"suffix_large", double(suffix * b >= 32)},
      {"log_step", tail.discrete ? 0 : lg(tail.pitch * b)},
      {"log_alignment", lg(std::gcd(alignment, int64_t(128)))},
      {"log_tail_span", lg((tail.extent - 1) * double(tail.pitch) * b + b)},
      {"log_suffix_span", lg(suffixSpan)},
      {"log_suffix_density", lg(suffixSpan / (suffix * b))},
      {"tail_mod32", std::fmod(tail.pitch * b, 32)},
      {"tail_mod128", std::fmod(tail.pitch * b, 128)},
      {"regular_rank", double(regular.size())},
      {"all_regular_rank", double(all.size())},
      {"min_tile_dim", minDim},
      {"tile_min_axis", double(minAxis)},
      {"tile_large", double(tile * b >= 32)},
      {"static_work", tile * b >= 32 ? E / tile : E},
      {"all_S_work", sp * b >= 32 ? E / sp : E},
      {"all_tile_work", allTile * b >= 32 ? E / allTile : E},
      {"tile_scalar_rows", t.empty() ? E : E / t.back().first},
      {"tile_outer_work", t.empty() ? E : E / tile * minDim},
      {"stride_work", E * inv},
      {"clipped_stride_work", E * std::max(inv - 1.0 / 128, 0.0)},
      {"E_after256", std::max(E - 256, 0.0)},
      {"E_after512", std::max(E - 512, 0.0)},
      {"E_log_span", E * lg(ss)},
      {"sqrt_E_log_span", std::sqrt(E) * lg(ss)},
      {"sqrt_E_log_pitch", std::sqrt(E) * lg(maxDP)},
      {"sqrt_E_log_density", std::sqrt(E) * lg(suffixSpan / (suffix * b))},
      {"lane_work128", lane128},
      {"lane_work32", lane32},
      {"sqrt_lane_work128", std::sqrt(lane128)},
      {"log_lane_work128", lg(lane128)},
      {"D_warp_product", dw},
      {"log_logical_span", lg(logicalSpan)},
      {"logical_density", E * b / logicalSpan}};
  for (size_t i = 0; i < 5; ++i) {
    int64_t pitch = i < rank ? a[i].pitch : 1;
    auto k = std::to_string(i);
    f["axis" + k] = i < rank ? lg(a[i].extent) : 0;
    f["pitch" + k] = lg(pitch);
    f["D" + k] = i < rank && a[i].discrete;
    f["v2pitch" + k] = lg(pitch & -pitch);
  }
  t.insert(t.begin(), 3 - t.size(), {1, 1});
  for (auto layout : {"dense", "row32"}) {
    std::string p = layout;
    std::array<double, 3> u = {
        0, p == "dense" ? t[2].first : std::ceil(t[2].first * b / 32) * 32 / b,
        1};
    u[0] = t[1].first * u[1];
    bool a0 = std::fmod(u[0] * b, 32) == 0, a1 = std::fmod(u[1] * b, 32) == 0,
         a2 = std::fmod(u[2] * b, 32) == 0;
    bool m0 =
        u[0] == u[1] * t[1].first && t[0].second == t[1].second * t[1].first;
    bool m1 =
        u[1] == u[2] * t[2].first && t[1].second == t[2].second * t[2].first;
    bool legal = t[2].second == 1
                     ? (t[1].second == t[2].first && u[1] == t[2].first ? a0
                        : m0                                            ? a1
                             : a0 && a1)
                     : (m0 && a1) || (m1 && a0 && a2);
    f[p + "_legal"] = legal;
    f[p + "_work"] = legal ? E / tile : E;
  }
  // Counterfactual basis coordinates are logical positions 0..extent-1,
  // including D axes. These describe TTIR coefficients, not actual indices
  // or expected random-address occupancy. At most 2048 positions are visited.
  std::set<int64_t> full, structured, wave;
  size_t waves = 0;
  for (int64_t k = 0; k < int64_t(E); ++k) {
    int64_t logical = k, offset = 0, soffset = 0;
    for (size_t i = rank; i-- > 0;) {
      int64_t term = logical % a[i].extent * a[i].pitch * (g.bits / 8);
      logical /= a[i].extent;
      offset += term;
      if (!a[i].discrete)
        soffset += term;
    }
    full.insert(offset / 128);
    structured.insert(soffset / 128);
    wave.insert(offset / 128);
    if (k % 32 == 31) {
      waves += wave.size();
      wave.clear();
    }
  }
  f["basis_bins128"] = full.size();
  f["S_basis_bins128"] = structured.size();
  f["S_basis_work128"] = dp * structured.size();
  f["basis_waves128"] = waves;
  return f;
}
