#include "AscendModel/RouteModel/Reduction/ReductionCostModel.h"
#include "AscendModel/RouteModel/Reduction/ReductionLayoutModel.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
using namespace mlir::ascend;

static const std::vector<double> *
getReductionParameters(const ReductionCostProfile &profile, llvm::StringRef key,
                       size_t expectedSize) {
  auto found = profile.parameters.find(key);
  if (found == profile.parameters.end() || found->second.size() != expectedSize)
    return nullptr;
  return &found->second;
}

static std::string reductionRoute(const ReductionWorkload &work) {
  return work.kind + "_" + work.dataType;
}

static bool isSupportedReductionRoute(const ReductionWorkload &work) {
  const llvm::StringRef op(work.kind);
  const llvm::StringRef dtype(work.dataType);
  if (op == "sum" || op == "max" || op == "min")
    return dtype == "f16" || dtype == "bf16" || dtype == "f32" ||
           dtype == "i32";
  if (op == "prod")
    return dtype == "f16" || dtype == "bf16" || dtype == "f32" ||
           dtype == "i8" || dtype == "i16" || dtype == "i32" || dtype == "i64";
  if (op == "xor" || op == "or" || op == "and")
    return dtype == "i8" || dtype == "i16" || dtype == "i32" || dtype == "i64";
  return false;
}

static std::string canonicalStandardRoute(const ReductionWorkload &work) {
  const std::string route = reductionRoute(work);
  if (work.kind == "min")
    return "max_" +
           (work.dataType == "bf16" ? std::string("f16") : work.dataType);
  if (work.dataType == "bf16" && (work.kind == "sum" || work.kind == "max"))
    return work.kind + "_f16";
  return route;
}

static std::optional<int64_t> checkedProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

static std::optional<int64_t> nextPowerOfTwo(int64_t value) {
  int64_t result = 1;
  while (result < value) {
    if (result > std::numeric_limits<int64_t>::max() / 2)
      return std::nullopt;
    result *= 2;
  }
  return result;
}

static std::optional<double>
estimateSimdRank1(const ReductionWorkload &work,
                  const ReductionCostProfile &profile, int64_t k) {
  const std::string route = canonicalStandardRoute(work);
  if (const auto *p =
          getReductionParameters(profile, "r1_standard_" + route, 3)) {
    const double q = (*p)[0];
    return (*p)[1] + (*p)[2] * std::max(0.0, std::ceil(k / q) - 1.0);
  }
  const std::string raw = reductionRoute(work);
  if (const auto *p =
          getReductionParameters(profile, "r1_piecewise_" + raw, 6)) {
    if (k < 32) {
      if (k != 2 && k != 4 && k != 8 && k != 16)
        return std::nullopt;
      unsigned index = static_cast<unsigned>(std::log2(k)) - 1;
      return (*p)[index];
    }
    return (*p)[4] + (*p)[5] * static_cast<double>(k);
  }
  return std::nullopt;
}

static std::optional<double>
estimateSimdRankN(const ReductionWorkload &work,
                  const ReductionCostProfile &profile, int64_t g, int64_t k) {
  const std::string route = canonicalStandardRoute(work);
  if (route == "xor_i64" && k == 2) {
    if (const auto *p = getReductionParameters(profile, "rn_xor_i64_k2", 2))
      return (*p)[0] + (*p)[1] * static_cast<double>(g);
    return std::nullopt;
  }
  if (const auto *p =
          getReductionParameters(profile, "rn_standard_" + route, 5)) {
    const double h = std::max(0.0, std::ceil(k / (*p)[0]) - 1.0);
    const double shortK = k >= 2 && k <= 8 ? 1.0 : 0.0;
    return (*p)[1] + (*p)[2] * g + (*p)[3] * g * h + (*p)[4] * g * shortK;
  }
  const std::string raw = reductionRoute(work);
  if (const auto *p = getReductionParameters(profile, "rn_linear_" + raw, 2)) {
    const double combines = static_cast<double>(g) * (k - 1);
    return std::max((*p)[0], (*p)[1] * combines);
  }
  return std::nullopt;
}

static std::string simtRank1Class(const ReductionWorkload &work) {
  if (work.kind == "prod" &&
      (work.dataType == "f16" || work.dataType == "bf16"))
    return "prod16";
  if (work.dataType == "i64" && work.kind == "prod")
    return "prod_i64";
  if (work.dataType == "i64" &&
      (work.kind == "xor" || work.kind == "or" || work.kind == "and"))
    return "i64";
  return "standard";
}

static std::optional<double>
rank1WarpFactor(const ReductionCostProfile &profile, llvm::StringRef routeClass,
                int64_t warps, int64_t b) {
  const auto *p = getReductionParameters(
      profile, ("r1_w_" + routeClass + "_" + llvm::Twine(warps)).str(), 12);
  if (!p || b < 2 || (b & (b - 1)) != 0)
    return std::nullopt;
  // The warp-factor table is calibrated for B = 2..4096 (12 entries).
  // For larger reductions, reuse the largest calibrated factor so the tree
  // formula extrapolates without an out-of-range fallback.
  const int64_t clampedB = std::min(b, int64_t(4096));
  return (*p)[static_cast<size_t>(std::log2(clampedB)) - 1];
}

static std::optional<double>
estimateSimtRank1(const ReductionWorkload &work,
                  const ReductionCostProfile &profile, int64_t warps,
                  int64_t k) {
  const auto padded = nextPowerOfTwo(k);
  if (!padded)
    return std::nullopt;
  const int64_t b = *padded;
  if (b < 2)
    return std::nullopt;
  const std::string routeClass = simtRank1Class(work);
  const auto factor = rank1WarpFactor(profile, routeClass, warps, b);
  if (!factor)
    return std::nullopt;
  const double l = std::log2(static_cast<double>(b));
  double base = 0.0;
  if (routeClass == "prod16") {
    const auto *p = getReductionParameters(profile, "r1_base_prod16", 9);
    if (!p)
      return std::nullopt;
    base = b <= 128 ? (*p)[static_cast<size_t>(l) - 1]
                    : (*p)[7] + (*p)[8] * (l - 7.0);
  } else {
    const std::string baseClass =
        routeClass == "prod_i64" ? std::string("i64") : routeClass;
    const auto *p = getReductionParameters(profile, "r1_base_" + baseClass, 4);
    if (!p)
      return std::nullopt;
    base = (*p)[0] + (*p)[1] * l + (*p)[2] * (b >= 64 ? 1.0 : 0.0) +
           (*p)[3] * std::max(l - 6.0, 0.0);
  }
  return base * *factor;
}

static std::string simtRankNBaseRoute(const ReductionWorkload &work) {
  if (work.kind == "min")
    return "max_" +
           (work.dataType == "bf16" ? std::string("f16") : work.dataType);
  if (work.kind == "or" || work.kind == "and")
    return "xor_" + work.dataType;
  if (work.dataType == "bf16" &&
      (work.kind == "sum" || work.kind == "max" || work.kind == "prod"))
    return work.kind + "_f16";
  if (work.kind == "prod" && work.dataType == "i32")
    return "prod_f32";
  return reductionRoute(work);
}

static std::optional<double>
estimateSimtRankN(const ReductionWorkload &work,
                  const ReductionCostProfile &profile, int64_t warps, int64_t g,
                  int64_t k) {
  const std::string baseRoute = simtRankNBaseRoute(work);
  const std::string candidateKey = "rn_candidate_" + baseRoute;
  if (profile.parameters.count(candidateKey)) {
    // Q, N0, then coefficients for: 1, G, G*H, P, G*H^2, P^2/64,
    // G*log2(K), G*sqrt(K), G*K/64, I(K<=128), G*I(K<=128),
    // G*H/(1+G/16), H, P*I(G<=4), G*K/64*I(G<=4), G^2.
    const auto *p = getReductionParameters(profile, candidateKey, 18);
    const auto *scale =
        getReductionParameters(profile, "rn_alias_" + reductionRoute(work), 1);
    const auto *warp = getReductionParameters(
        profile, "rn_w_" + baseRoute + "_" + std::to_string(warps), 1);
    if (!p || !scale || !warp || (*p)[0] <= 0.0 || (*p)[1] < 0.0 ||
        (*scale)[0] <= 0.0 || (*warp)[0] <= 0.0)
      return std::nullopt;
    const double groups = static_cast<double>(g);
    const double extent = static_cast<double>(k);
    const double h = std::max(0.0, std::ceil(extent / (*p)[0]) - 1.0);
    const double n = groups * extent / 64.0;
    const double excess = std::max(groups * extent - (*p)[1], 0.0) / 64.0;
    const double smallK = k <= 128 ? 1.0 : 0.0;
    const double smallG = g <= 4 ? 1.0 : 0.0;
    const std::array<double, 16> terms = {1.0,
                                          groups,
                                          groups * h,
                                          excess,
                                          groups * h * h,
                                          excess * excess / 64.0,
                                          groups * std::log2(extent),
                                          groups * std::sqrt(extent),
                                          n,
                                          smallK,
                                          groups * smallK,
                                          groups * h / (1.0 + groups / 16.0),
                                          h,
                                          excess * smallG,
                                          n * smallG,
                                          groups * groups};
    double base = 0.0;
    for (size_t i = 0; i < terms.size(); ++i)
      base += (*p)[i + 2] * terms[i];
    // A nonpositive extrapolation is not a free reduction. Use the complete
    // legacy shuffle fallback, rather than clipping it to zero or combining
    // new multipliers with the previous calibrated polynomial.
    if (!std::isfinite(base) || base <= 0.0)
      return std::nullopt;
    return (*scale)[0] * (*warp)[0] * base;
  }
  // Compatibility for profiles that still use the original eight-term model.
  const auto *p = getReductionParameters(profile, "rn_base_" + baseRoute, 8);
  const auto *scale =
      getReductionParameters(profile, "rn_alias_" + reductionRoute(work), 1);
  const auto *warp = getReductionParameters(
      profile, "rn_w_" + baseRoute + "_" + std::to_string(warps), 1);
  if (!p || !scale || !warp)
    return std::nullopt;
  const double h = std::max(0.0, std::ceil(k / (*p)[0]) - 1.0);
  const double excess =
      std::max(static_cast<double>(g) * k - (*p)[1], 0.0) / 64.0;
  const double base = (*p)[2] + (*p)[3] * g + (*p)[4] * g * h +
                      (*p)[5] * excess + (*p)[6] * g * h * h +
                      (*p)[7] * excess * excess / 64.0;
  return (*scale)[0] * (*warp)[0] * base;
}

std::optional<double> mlir::ascend::estimateTailAxisReductionCycles(
    const ReductionWorkload &work, const ReductionCostProfile &profile,
    ReductionMode mode, int64_t numWarps) {
  if (!work.isFiniteAndNonNegative() || work.axis + 1 != work.shape.size() ||
      !isSupportedReductionRoute(work))
    return std::nullopt;
  const int64_t k = work.shape.back();
  if (k <= 1)
    return 0.0;
  std::optional<double> cycles;
  if (work.shape.size() == 1) {
    cycles = mode == ReductionMode::SIMD
                 ? estimateSimdRank1(work, profile, k)
                 : estimateSimtRank1(work, profile, numWarps, k);
  } else {
    auto g = checkedProduct(llvm::ArrayRef<int64_t>(work.shape).drop_back());
    if (!g)
      return std::nullopt;
    cycles = mode == ReductionMode::SIMD
                 ? estimateSimdRankN(work, profile, *g, k)
                 : estimateSimtRankN(work, profile, numWarps, *g, k);
  }
  if (!cycles || !std::isfinite(*cycles) || *cycles < 0.0)
    return std::nullopt;
  return cycles;
}

namespace {
std::optional<double> evaluate(const ReductionExpression &e,
                               const llvm::StringMap<double> &features) {
  if (e.kind == ReductionExpression::Kind::Constant)
    return e.constant;
  if (e.kind == ReductionExpression::Kind::Feature) {
    auto i = features.find(e.name);
    return i == features.end() ? std::nullopt
                               : std::optional<double>(i->second);
  }
  std::vector<double> args;
  for (const auto &a : e.arguments) {
    auto v = evaluate(a, features);
    if (!v)
      return std::nullopt;
    args.push_back(*v);
  }
  double x = args[0];
  llvm::StringRef op = e.name;
  if (op == "neg")
    x = -x;
  else if (op == "not")
    x = !x;
  else if (op == "indicator")
    x = x != 0;
  else if (op == "ceil")
    x = std::ceil(x);
  else if (op == "sqrt") {
    if (x < 0)
      return std::nullopt;
    x = std::sqrt(x);
  } else if (op == "log2") {
    if (x <= 0)
      return std::nullopt;
    x = std::log2(x);
  } else if (op == "sub")
    x -= args[1];
  else if (op == "div") {
    if (args[1] <= 0)
      return std::nullopt;
    x /= args[1];
  } else if (op == "pow") {
    if (x < 0)
      return std::nullopt;
    x = std::pow(x, args[1]);
  } else if (op == "lt")
    x = x < args[1];
  else if (op == "le")
    x = x <= args[1];
  else if (op == "gt")
    x = x > args[1];
  else if (op == "ge")
    x = x >= args[1];
  else if (op == "eq")
    x = x == args[1];
  else if (op == "ne")
    x = x != args[1];
  else
    for (size_t i = 1; i < args.size(); ++i) {
      if (op == "add")
        x += args[i];
      else if (op == "mul")
        x *= args[i];
      else if (op == "min")
        x = std::min(x, args[i]);
      else if (op == "max")
        x = std::max(x, args[i]);
      else if (op == "and")
        x = x && args[i];
      else if (op == "or")
        x = x || args[i];
      else
        return std::nullopt;
    }
  return std::isfinite(x) ? std::optional<double>(x) : std::nullopt;
}
} // namespace
llvm::Expected<std::optional<double>>
mlir::ascend::estimateFirstAxisReductionCycles(const ReductionWorkload &work,
                                               const ReductionProfile &profile,
                                               ReductionMode mode,
                                               int64_t warps) {
  if (!work.isFiniteAndNonNegative() || work.axis != 0 || work.shape.size() < 2)
    return std::nullopt;
  auto found = profile.findFirstAxis(mode, work.kind, work.dataType);
  if (!found)
    return found.takeError();
  const auto *function = *found;
  if (!function || std::find(function->supportedWarps.begin(),
                             function->supportedWarps.end(),
                             warps) == function->supportedWarps.end())
    return std::nullopt;
  if (work.shape[0] == 1)
    return 0.0;
  int64_t groups = 1;
  for (size_t i = 1; i < work.shape.size(); ++i) {
    if (groups > INT64_MAX / work.shape[i])
      return std::nullopt;
    groups *= work.shape[i];
  }
  const double r = work.shape[0], g = groups;
  llvm::StringMap<double> features;
  features["R"] = r;
  features["G"] = g;
  features["W"] = warps;
  features["rank"] = work.shape.size();
  features["d1"] = work.shape[1];
  features["d2"] = work.shape.size() > 2 ? work.shape[2] : 1;
  const ReductionSegment *matched = nullptr;
  for (const auto &segment : function->segments) {
    features["L"] = segment.laneWidth;
    features["Q"] = std::ceil(g / segment.laneWidth);
    auto condition = evaluate(segment.condition, features);
    if (!condition)
      return std::nullopt;
    if (*condition == 0)
      continue;
    if (matched)
      return std::nullopt;
    matched = &segment;
  }
  if (!matched)
    return std::nullopt;
  features["L"] = matched->laneWidth;
  features["Q"] = std::ceil(g / matched->laneWidth);
  auto cost = evaluate(matched->expression, features);
  if (!cost || *cost <= 0)
    return std::nullopt;
  return cost;
}

llvm::Expected<std::optional<double>>
mlir::ascend::estimateMiddleAxisReductionCycles(const ReductionWorkload &work,
                                                const ReductionProfile &profile,
                                                ReductionMode mode,
                                                int64_t warps) {
  auto g = normalizeMiddleAxis(work);
  if (!g)
    return std::nullopt;
  auto found = profile.findMiddleAxis(mode, work.kind, work.dataType);
  if (!found)
    return found.takeError();
  const auto *fn = *found;
  if (!fn || fn->layoutPolicy != "row_major_128bit_v1" ||
      std::find(fn->supportedWarps.begin(), fn->supportedWarps.end(), warps) ==
          fn->supportedWarps.end())
    return std::nullopt;
  if (g->reduction == 1)
    return 0.0;
  int64_t bits = work.dataType == "f16" || work.dataType == "bf16" ? 16
                 : work.dataType == "f32"                          ? 32
                                                                   : 0;
  if (!bits && work.dataType.size() > 1 &&
      (work.dataType[0] == 'i' || work.dataType[0] == 'u')) {
    auto str = llvm::StringRef(work.dataType).drop_front();
    if (str.getAsInteger(10, bits))
      return std::nullopt;
  }
  auto layout = projectReductionLayout(*g, bits, warps);
  if (mode == ReductionMode::SIMT && !layout)
    return std::nullopt;
  llvm::StringMap<double> features;
  if (layout)
    features = middleAxisFeatures(*g, *layout, warps);
  features["O"] = g->outer;
  features["K"] = g->reduction;
  features["H"] = g->inner;
  features["G"] = static_cast<double>(g->outer) * g->inner;
  features["N"] = features["G"] * g->reduction;
  features["W"] = warps;
  const ReductionSegment *matched = nullptr;
  for (const auto &segment : fn->segments) {
    features["L"] = segment.laneWidth;
    auto cond = evaluate(segment.condition, features);
    if (!cond)
      return std::nullopt;
    if (!*cond)
      continue;
    if (matched)
      return std::nullopt;
    matched = &segment;
  }
  if (!matched)
    return std::nullopt;
  features["L"] = matched->laneWidth;
  auto c = evaluate(matched->expression, features);
  return c && *c > 0 ? c : std::nullopt;
}
llvm::Expected<std::optional<FittedReductionEstimate>>
mlir::ascend::estimateReductionCycles(const ReductionWorkload &work,
                                      const ReductionProfile &profile,
                                      ReductionMode mode, int64_t warps) {
  if (!work.isFiniteAndNonNegative())
    return std::nullopt;
  ReductionAxisKind axis;
  std::optional<double> c;
  if (work.axis + 1 == work.shape.size()) {
    axis = ReductionAxisKind::Tail;
    c = estimateTailAxisReductionCycles(work, profile.getModeProfile(mode),
                                        mode, warps);
  } else if (work.axis == 0) {
    axis = ReductionAxisKind::First;
    auto result = estimateFirstAxisReductionCycles(work, profile, mode, warps);
    if (!result)
      return result.takeError();
    c = *result;
  } else {
    axis = ReductionAxisKind::Middle;
    auto result = estimateMiddleAxisReductionCycles(work, profile, mode, warps);
    if (!result)
      return result.takeError();
    c = *result;
  }
  if (!c || !std::isfinite(*c) || *c < 0)
    return std::nullopt;
  return FittedReductionEstimate{*c, axis};
}
