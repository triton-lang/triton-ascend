#include "AscendModel/RouteModel/Models/PartialStructuredSimtLoadModel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <set>

using namespace mlir::ascend::partial_memory;
namespace {
bool powerOfTwo(int64_t n) { return n > 0 && !(n & (n - 1)); }
int64_t reverseBits(int64_t value, int64_t extent) {
  int64_t result = 0;
  for (; extent > 1; extent >>= 1, value >>= 1)
    result = (result << 1) | (value & 1);
  return result;
}
double log2safe(double value) { return std::log2(std::max(value, 1.0)); }
struct FactorNode {
  const char *feature;
  double threshold;
  int left, right; // A null feature denotes a zero-based coefficient row.
};
#include "PartialStructuredSimtLoadModels.inc"

bool supported(const FactorLoadGeometry &c) {
  const auto &g = c.geometry;
  const auto rank = g.axes.size();
  if ((c.factor != 2 && c.factor != 4) || rank < 2 || rank > 5 ||
      !powerOfTwo(g.warps) || g.warps > 64 / c.factor ||
      (g.bits != 8 && g.bits != 16 && g.bits != 32 && g.bits != 64) ||
      g.floatingPoint != (g.bits == 32) || c.lower.size() != rank ||
      c.upper.size() != rank || c.programStride <= 0 ||
      c.programStride > (16 * 1024 * 1024) / (c.factor * (g.bits / 8)) ||
      c.programStride % (256 / (g.bits / 8)))
    return false;
  int64_t elements = 1;
  unsigned discrete = 0;
  // Prevent overflow and remain inside the actual finite workspace coverage.
  int64_t span = 1;
  for (size_t i = 0; i < rank; ++i) {
    const auto &a = g.axes[i];
    if (!powerOfTwo(a.extent) || a.extent < 2 || a.pitch <= 0 ||
        elements > 2048 / a.extent || c.lower[i] < 0 ||
        c.upper[i] <= c.lower[i] || c.upper[i] > a.extent)
      return false;
    elements *= a.extent;
    discrete += a.discrete;
    const int64_t maximum = (a.discrete ? 4 * a.extent : a.extent) - 1;
    if (a.pitch > (c.programStride - span) / maximum)
      return false;
    span += maximum * a.pitch;
  }
  if (elements < 32 || !discrete || discrete == rank)
    return false;
  // Only the four measured constant rectangular mask families, or no mask.
  for (int kind = 0; kind < 5; ++kind) {
    bool same = true;
    for (size_t i = 0; i < rank; ++i) {
      int64_t lo = 0, hi = g.axes[i].extent;
      if ((kind == 1 || kind == 3) && i == 0)
        hi /= 2;
      if ((kind == 2 || kind == 3) && i == rank - 1)
        hi /= 2;
      if (kind == 4 && i == rank - 1) {
        lo = hi / 4;
        hi = lo + hi / 2;
      }
      same &= c.lower[i] == lo && c.upper[i] == hi;
    }
    if (same)
      return true;
  }
  return false;
}
} // namespace

Features mlir::ascend::partial_memory::extractFactorReferenceFeatures(
    const FactorLoadGeometry &c) {
  const auto &g = c.geometry;
  const size_t rank = g.axes.size();
  int64_t E = 1, K = 1, activeD = 1;
  for (size_t i = 0; i < rank; ++i) {
    E *= g.axes[i].extent;
    K *= c.upper[i] - c.lower[i];
    if (g.axes[i].discrete)
      activeD *= c.upper[i] - c.lower[i];
  }
  std::vector<size_t> order(rank);
  std::iota(order.begin(), order.end(), 0);
  if (E < 32 * g.warps)
    std::rotate(order.begin(), order.end() - 1, order.end());
  else
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      auto contiguity = [&](size_t i) {
        return !g.axes[i].discrete && g.axes[i].pitch == 1 ? g.axes[i].extent
                                                           : int64_t(1);
      };
      return contiguity(a) > contiguity(b);
    });
  const int64_t issued = std::max(E, 32 * g.warps);
  std::vector<std::set<int64_t>> bins(issued / 32);
  std::vector<bool> warpActive(g.warps, false);
  std::set<int64_t> global, structured, group;
  const int64_t bytes = g.bits / 8;
  for (int64_t t = 0; t < issued; ++t) {
    int64_t div = 1, offset = 0, soff = 0;
    bool active = true;
    for (size_t i : order) {
      const auto &a = g.axes[i];
      const int64_t j = t / div % a.extent;
      div *= a.extent;
      active &= j >= c.lower[i] && j < c.upper[i];
      const int64_t page = std::min<int64_t>(4, a.extent);
      const int64_t coordinate =
          a.discrete
              ? 4 * reverseBits(j / page, a.extent / page) * page + j % page
              : j;
      offset += coordinate * a.pitch * bytes;
      if (!a.discrete)
        soff += j * a.pitch * bytes;
    }
    if (!active)
      continue;
    bins[t / 32].insert(offset / 128);
    warpActive[t / 32 % g.warps] = true;
    global.insert(offset / 128);
    structured.insert(soff / 128);
    for (int64_t f = 0; f < c.factor; ++f)
      group.insert((offset + f * c.programStride * bytes) / 128);
  }
  double groups = 0, H = 0;
  for (const auto &b : bins) {
    groups += !b.empty();
    H += b.size();
  }
  const double W = std::count(warpActive.begin(), warpActive.end(), true);
  std::array<int64_t, 32> aliases{};
  for (int64_t bin : group)
    ++aliases[bin % 32];
  const int64_t used = std::count_if(aliases.begin(), aliases.end(),
                                     [](int64_t n) { return n != 0; });
  int64_t contiguous = 1;
  for (size_t i = rank; i-- > 0;) {
    if (g.axes[i].discrete || g.axes[i].pitch != contiguous)
      break;
    contiguous *= c.upper[i] - c.lower[i];
  }
  const double j = log2safe(c.factor * groups);
  const double h = log2safe(c.factor * H);
  Features out{{"one", 1},
               {"K", double(K)},
               {"bits", double(g.bits)},
               {"log_native_J", j},
               {"log_native_active_warps", log2safe(W)},
               {"logF", log2safe(c.factor)},
               {"native_order0", double(order.front())},
               {"native_page_H128", c.factor * H},
               {"log_native_page_H128", h},
               {"log_spread_page_G128", log2safe(c.factor * global.size())},
               {"S_work", double(c.factor * activeD * structured.size())},
               {"tail_stride", double(g.axes.back().pitch * bytes)},
               {"active_contiguous_bytes", double(contiguous * bytes)},
               {"ref_log_max_alias32",
                log2safe(*std::max_element(aliases.begin(), aliases.end()))},
               {"ref_log_mean_alias32", log2safe(double(group.size()) / used)},
               {"log_native_J_x_log_native_J", j * j},
               {"log_native_page_H128_x_log_native_page_H128", h * h},
               {"logW_x_log_native_page_H128", log2safe(g.warps) * h},
               {"logW_x_log_bits", log2safe(g.warps) * log2safe(g.bits)}};
  for (int k : {1, 3})
    out["native_later" + std::to_string(k)] =
        c.factor * std::max(groups - k * W, 0.0);
  return out;
}

Features mlir::ascend::partial_memory::extractFactorLoadFeatures(
    const FactorLoadGeometry &c) {
  return supported(c) ? extractFactorReferenceFeatures(c) : Features{};
}

std::optional<double>
mlir::ascend::partial_memory::predictFactorLoad(const FactorLoadGeometry &c) {
  if (!supported(c))
    return std::nullopt;
  const auto f = extractFactorLoadFeatures(c);
  if (f.empty())
    return std::nullopt;
  int64_t E = 1;
  for (auto a : c.geometry.axes)
    E *= a.extent;
  const bool masked = f.at("K") != E;
  const auto *nodes = masked ? maskNodes : unmaskedNodes;
  const auto *coeff = masked ? maskCoefficients : unmaskedCoefficients;
  const auto *terms = masked ? maskTerms : unmaskedTerms;
  const int count = masked ? 8 : 10;
  int node = 0;
  while (nodes[node].feature) {
    const auto &n = nodes[node];
    node = f.at(n.feature) <= n.threshold ? n.left : n.right;
  }
  double value = 0;
  for (int k = 0; k < count; ++k)
    value += coeff[nodes[node].left][k] * f.at(terms[k]);
  return std::max(1.0, std::exp(std::clamp(value, -20.0, 25.0)));
}
