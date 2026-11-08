// Frozen ordinary-Store candidate; standard C++17, no LLVM/runtime dependency.
// Values are matched target increments in SYS_CNT ticks, never whole-kernel
// cost.
#ifndef INDIRECT_STORE_ORDINARY_STORE_ROUND4_H
#define INDIRECT_STORE_ORDINARY_STORE_ROUND4_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace indirect_store_round4 {
enum class Mode { Unknown, SIMD, SIMT };
enum class Visibility { Unknown, Known, Host, Device };
struct Metadata {
  std::vector<int64_t> shape, lower, upper;
  std::vector<bool> dAxes;
  std::vector<int64_t> knownStrides;
  Mode mode = Mode::Unknown;
  Visibility visibility = Visibility::Unknown;
  int64_t elementBytes = 0, baseWarps = 0, factor = 0;
  bool hasMask = false;
  bool rectangularMaskVerified = false;
  bool stageOwnershipVerified = false;
  bool firstStore = false;
  bool matchedIncrementContractVerified = false;
};
struct Estimate {
  bool eligible = false;
  std::optional<double> ticks;
  std::string reason;
  std::string leaf;
  const char *scope = "matched_target_increment_ticks";
};
using Features = std::map<std::string, double>;
struct Term {
  const char *name;
  double coefficient;
};
struct Formula {
  const char *key;
  const Term *terms;
  size_t size;
};

inline std::string validate(const Metadata &m) {
  if (m.mode != Mode::SIMD && m.mode != Mode::SIMT)
    return "unsupported_mode";
  if (m.factor != 1)
    return "F1_only";
  if (m.shape.size() < 2 || m.shape.size() > 5 ||
      m.dAxes.size() != m.shape.size() || m.lower.size() != m.shape.size() ||
      m.upper.size() != m.shape.size())
    return "invalid_mixed_axis_rank_or_mask";
  if (!m.dAxes.back() ||
      std::all_of(m.dAxes.begin(), m.dAxes.end(), [](bool d) { return d; }))
    return "mixed_S_D_with_logical_tail_D_required";
  if (m.elementBytes != 1 && m.elementBytes != 2 && m.elementBytes != 4 &&
      m.elementBytes != 8)
    return "unsupported_storage_width";
  if (m.baseWarps < 1 || m.baseWarps > 64 || (m.baseWarps & (m.baseWarps - 1)))
    return "base_warps_outside_frozen_domain";
  if (m.visibility != Visibility::Known && m.visibility != Visibility::Host &&
      m.visibility != Visibility::Device)
    return "missing_visibility";
  int64_t elements = 1;
  for (size_t i = 0; i < m.shape.size(); ++i) {
    const auto n = m.shape[i];
    if (n < 1 || n > 4096 || elements > 8192 / n)
      return "tile_exceeds_bounded_frozen_domain";
    elements *= n;
    if (m.lower[i] < 0 || m.lower[i] > m.upper[i] || m.upper[i] > n)
      return "invalid_rectangular_bounds";
    if (!m.hasMask && (m.lower[i] != 0 || m.upper[i] != n))
      return "unmasked_bounds_mismatch";
  }
  if (elements < 4)
    return "tile_below_frozen_domain";
  if (!m.rectangularMaskVerified)
    return "rectangular_mask_not_proven";
  return {};
}

inline int64_t product(const std::vector<int64_t> &values) {
  int64_t result = 1;
  for (auto v : values)
    result *= v;
  return result;
}

// Only logical coordinates are enumerated, bounded by E<=8192. No loaded index,
// device tensor, observed address, seed, input ID, kernel name, U/V, q, or E/W.
inline std::optional<Features>
features(const Metadata &m, const std::vector<int64_t> *runtimeCoefficients,
         std::string &reason) {
  reason = validate(m);
  if (!reason.empty())
    return std::nullopt;
  const int64_t E = product(m.shape), W = m.baseWarps, b = m.elementBytes;
  std::vector<int64_t> active;
  for (size_t i = 0; i < m.shape.size(); ++i)
    active.push_back(m.upper[i] - m.lower[i]);
  const int64_t K = product(active);
  if (!K)
    return Features{{"semantic_zero", 1.0}};
  const std::vector<int64_t> *co = nullptr;
  // Do not access runtimeCoefficients or knownStrides for Device visibility.
  if (m.visibility == Visibility::Known)
    co = &m.knownStrides;
  else if (m.visibility == Visibility::Host)
    co = runtimeCoefficients;
  if (m.visibility != Visibility::Device) {
    if (!co || co->size() < m.shape.size()) {
      reason = "explicit_visible_coefficients_required";
      return std::nullopt;
    }
    for (size_t i = 0; i < m.shape.size(); ++i)
      if ((*co)[i] <= 0 || (*co)[i] > (int64_t(1) << 31)) {
        reason = "visible_stride_outside_safe_positive_domain";
        return std::nullopt;
      }
  }
  int64_t S = 1, D = 1, activeS = 1, activeD = 1, nD = 0, lastS = -1;
  double invD = 0, invActiveD = 0, smallActiveD = 0;
  std::vector<int64_t> ss, ds;
  int64_t unitS = 0;
  for (size_t i = 0; i < m.shape.size(); ++i) {
    if (m.dAxes[i]) {
      ++nD;
      D *= m.shape[i];
      activeD *= active[i];
      invD += 1.0 / m.shape[i];
      invActiveD += 1.0 / std::max<int64_t>(active[i], 1);
      smallActiveD += active[i] <= 4;
      if (co)
        ds.push_back((*co)[i] * b);
    } else {
      lastS = static_cast<int64_t>(i);
      S *= m.shape[i];
      activeS *= active[i];
      if (co) {
        ss.push_back((*co)[i] * b);
        if ((*co)[i] == 1)
          unitS += m.shape[i];
      }
    }
  }
  const double logStride =
      ss.empty() ? 0
                 : std::log2(std::max<int64_t>(
                       *std::min_element(ss.begin(), ss.end()), 1));
  const double logDstride =
      ds.empty() ? 0
                 : std::log2(std::max<int64_t>(
                       *std::min_element(ds.begin(), ds.end()), 1));
  const double maxSlog =
      ss.empty() ? 0 : std::log2(*std::max_element(ss.begin(), ss.end()));
  int64_t leading = 1;
  for (int64_t i = 0; i < lastS; ++i)
    leading *= active[i];
  std::vector<bool> logical(E);
  int64_t runs = 0, partial32 = 0, activeLogical32 = 0;
  for (int64_t index = 0; index < E; ++index) {
    int64_t suffix = E;
    bool alive = true;
    for (size_t i = 0; i < m.shape.size(); ++i) {
      suffix /= m.shape[i];
      const auto coordinate = (index / suffix) % m.shape[i];
      alive = alive && m.lower[i] <= coordinate && coordinate < m.upper[i];
    }
    logical[index] = alive;
    runs += alive && (!index || !logical[index - 1]);
  }
  for (int64_t start = 0; start < E; start += 32) {
    bool any = false, all = true;
    for (int64_t i = start; i < std::min<int64_t>(start + 32, E); ++i) {
      any |= logical[i];
      all &= logical[i];
    }
    partial32 += any && !all;
    activeLogical32 += any;
  }
  double ratio = 0, under128 = 0;
  if (co) {
    std::set<int64_t> offsets{0};
    for (size_t i = 0; i < m.shape.size(); ++i) {
      if (m.dAxes[i]) {
        under128 += (*co)[i] * b < 128;
        continue;
      }
      std::set<int64_t> next;
      for (auto off : offsets)
        for (int64_t coordinate = 0; coordinate < active[i]; ++coordinate)
          next.insert(off + coordinate * (*co)[i] * b);
      offsets = std::move(next);
    }
    ratio = double(activeD) * (*offsets.rbegin() - *offsets.begin() + b) /
            (double(offsets.size()) * b);
  }
  Features x{
      {"one", 1},
      {"E", double(E)},
      {"K", double(K)},
      {"groups32", std::ceil(double(K) / 32)},
      {"sqrtK", std::sqrt(double(K))},
      {"KlogW", K * std::log2(double(W))},
      {"KlogStride", K * logStride},
      {"hidden", co ? 0.0 : 1.0},
      {"nD", double(nD)},
      {"nS", double(m.shape.size() - nD)},
      {"KlogDstride", K * logDstride},
      {"logDstride", logDstride},
      {"logTail", std::log2(double(m.shape.back()))},
      {"minSbytes",
       ss.empty() ? 0.0 : double(*std::min_element(ss.begin(), ss.end()))},
      {"maxSlog", maxSlog},
      {"KunitS", double(K * unitS)},
      {"KsumInvD", K * invD},
      {"activeS", double(activeS)},
      {"activeD", double(activeD)},
      {"activeDlog", std::log2(double(std::max<int64_t>(activeD, 1)))},
      {"KactiveDlog", K * std::log2(double(std::max<int64_t>(activeD, 1)))},
      {"KactiveSlog", K * std::log2(double(std::max<int64_t>(activeS, 1)))},
      {"activeRows", double(K) / std::max<int64_t>(active.back(), 1)},
      {"leading", double(leading)},
      {"lastSSize", double(active[lastS])},
      {"KoverSqrtActiveD",
       K / std::sqrt(double(std::max<int64_t>(activeD, 1)))},
      {"KsumInvActiveD", K * invActiveD},
      {"KsmallActiveD", K * smallActiveD},
      {"KoverSqrtS", K / std::sqrt(double(activeS))},
      {"activeRuns", double(runs)},
      {"partialLogical32", double(partial32)},
      {"activeLogical32", double(activeLogical32)},
      {"logE", std::log2(double(E))},
      {"sqrtS", std::sqrt(double(activeS))},
      {"sqrtE", std::sqrt(double(E))},
      {"smallE128", E <= 128 ? 1.0 : 0.0},
      {"KoneWarp", W == 1 ? double(K) : 0.0},
      {"Krank", double(K * m.shape.size())},
      {"D_SspanRatio", ratio},
      {"Dunder128", under128}};
  for (int64_t boundary : {8, 16, 32, 64, 128, 256}) {
    x["KAbove" + std::to_string(boundary)] = std::max<int64_t>(K - boundary, 0);
    x["EAbove" + std::to_string(boundary)] = std::max<int64_t>(E - boundary, 0);
  }
  return x;
}

} // namespace indirect_store_round4
#endif
