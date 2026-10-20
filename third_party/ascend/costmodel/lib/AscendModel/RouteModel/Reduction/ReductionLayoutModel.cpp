#include "AscendModel/RouteModel/Reduction/ReductionLayoutModel.h"
#include <algorithm>
#include <cmath>
#include <limits>
using namespace mlir::ascend;
namespace {
int64_t ceilDiv(int64_t n, int64_t d) { return n / d + (n % d != 0); }
} // namespace
std::optional<MiddleAxisGeometry>
mlir::ascend::normalizeMiddleAxis(const ReductionWorkload &w) {
  if (!w.isFiniteAndNonNegative() || w.axis <= 0 ||
      w.axis + 1 >= static_cast<int64_t>(w.shape.size()))
    return std::nullopt;
  MiddleAxisGeometry g{1, w.shape[w.axis], 1};
  int64_t elements = 1;
  for (int64_t extent : w.shape) {
    if (elements > std::numeric_limits<int64_t>::max() / extent)
      return std::nullopt;
    elements *= extent;
  }
  // The complete product was checked before forming geometry subsets.
  for (int64_t i = 0; i < w.axis; ++i)
    g.outer *= w.shape[i];
  for (size_t i = w.axis + 1; i < w.shape.size(); ++i)
    g.inner *= w.shape[i];
  return g;
}
std::optional<ReductionLayoutProjection>
mlir::ascend::projectReductionLayout(const MiddleAxisGeometry &g, int64_t bits,
                                     int64_t W) {
  if (g.outer <= 0 || g.reduction <= 0 || g.inner <= 0 || W < 1 || W > 64 ||
      (W & (W - 1)) || (bits != 8 && bits != 16 && bits != 32 && bits != 64))
    return std::nullopt;
  if (g.outer > std::numeric_limits<int64_t>::max() / g.reduction ||
      g.outer * g.reduction > std::numeric_limits<int64_t>::max() / g.inner)
    return std::nullopt;
  int64_t n = g.outer * g.reduction * g.inner;
  ReductionLayoutProjection b{
      {1, 1,
       std::min({128 / bits, 16 / std::max(bits / 8, int64_t(1)), g.inner,
                 std::max(n / (32 * W), int64_t(1))})},
      {1, 1, 1},
      {1, 1, 1},
      0};
  std::array<int64_t, 3> shape{g.outer, g.reduction, g.inner};
  int64_t lanes = 32, threads = 32 * W, warps = W, pl = 1, pw = 1;
  for (int i : {2, 0}) {
    int64_t count =
        std::min(std::max(threads, int64_t(1)),
                 std::max(shape[i] / b.sizePerThread[i], int64_t(1)));
    b.threadsPerWarp[i] = std::min(count, lanes);
    b.warpsPerCTA[i] =
        std::min(std::max(count / b.threadsPerWarp[i], int64_t(1)), warps);
    lanes /= b.threadsPerWarp[i];
    warps /= b.warpsPerCTA[i];
    threads /= count;
    pl *= b.threadsPerWarp[i];
    pw *= b.warpsPerCTA[i];
  }
  b.threadsPerWarp[1] = 32 / pl;
  b.warpsPerCTA[1] = W / pw;
  for (int i = 0; i < 3; ++i) {
    if ((b.sizePerThread[i] & (b.sizePerThread[i] - 1)) ||
        (b.threadsPerWarp[i] & (b.threadsPerWarp[i] - 1)) ||
        (b.warpsPerCTA[i] & (b.warpsPerCTA[i] - 1)) ||
        shape[i] < b.sizePerThread[i] * b.threadsPerWarp[i] * b.warpsPerCTA[i])
      return std::nullopt;
  }
  b.route = b.warpsPerCTA[1] > 1 ? 2 : b.threadsPerWarp[1] > 1 ? 1 : 0;
  return b;
}
llvm::StringMap<double>
mlir::ascend::middleAxisFeatures(const MiddleAxisGeometry &g,
                                 const ReductionLayoutProjection &b,
                                 int64_t W) {
  auto s = b.sizePerThread, t = b.threadsPerWarp, w = b.warpsPerCTA;
  double R = ceilDiv(g.reduction, s[1] * t[1] * w[1]) * s[1];
  double A = static_cast<double>(ceilDiv(g.outer, s[0] * t[0] * w[0])) * s[0] *
             ceilDiv(g.inner, s[2] * t[2] * w[2]) * s[2];
  double G = static_cast<double>(g.outer) * g.inner,
         u = std::min(w[1], ceilDiv(g.reduction, s[1] * t[1]));
  double local = A * (R - 1), S = G * u, N = G * g.reduction;
  llvm::StringMap<double> f;
  f["O"] = g.outer;
  f["K"] = g.reduction;
  f["H"] = g.inner;
  f["W"] = W;
  f["G"] = G;
  f["N"] = N;
  f["route_id"] = b.route;
  f["R_t"] = R;
  f["A_t"] = A;
  f["N_local"] = local;
  f["N_warp"] = A * std::log2(std::min(t[1], g.reduction));
  f["N_cross"] = A * std::log2(w[1]);
  f["uK"] = u;
  f["wK"] = w[1];
  f["tK"] = t[1];
  f["tO"] = t[0];
  f["tH"] = t[2];
  f["sH"] = s[2];
  f["wO"] = w[0];
  f["wH"] = w[2];
  f["S"] = S;
  f["S2"] = S * u;
  f["Sr"] = S * R;
  f["Lr"] = local * R;
  f["Lm"] = local * A * u;
  f["Sc"] = S / t[1];
  f["gLane"] = G / t[1];
  f["mem"] = A * w[1];
  f["Npack"] = N / s[2];
  f["localTo"] = local / t[0];
  f["sqrtN"] = std::sqrt(N);
  f["one"] = 1;
  f["bar"] = std::log2(w[1]);
  f["Glogu"] = G * std::log2(u);
  f["hingeS"] = std::max(S - 256, 0.);
  f["hingeS2"] = std::max(S * u - 2048, 0.);
  f["packedTraffic"] = S * std::min(t[1], int64_t(2));
  f["r16To"] = std::max(R - 16, 0.) / t[0];
  f["R3"] = R * R * R;
  return f;
}
