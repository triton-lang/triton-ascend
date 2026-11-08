//===- SpecializedMemoryCostModel.cpp - Frozen structured GM model
//----------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "AscendModel/RouteModel/SpecializedMemoryCostModel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <string_view>

namespace mlir::ascend {
namespace {
using Family = SpecializedMemoryFamily;
using Mode = SpecializedMemoryMode;
using Operation = SpecializedMemoryOperation;
using Mask = SpecializedMemoryMask;
using Visibility = SpecializedMemoryVisibility;
using Geometry = SpecializedMemoryGeometry;

struct PublicFeatures {
  const Geometry &g;
  explicit PublicFeatures(const Geometry &geometry) : g(geometry) {}
  std::vector<double> base, affine, page13, rich, simtMasked;
  std::optional<std::array<double, 7>> run;
  double discretePitchWork = 0.0;
  int discreteAxes = 0, regularAxes = 0;
  bool unitStep = false;
};

double product(const std::vector<int64_t> &shape, size_t first, size_t last) {
  double value = 1.0;
  for (size_t i = first; i < last; ++i)
    value *= static_cast<double>(shape[i]);
  return value;
}

bool buildFeatures(const Geometry &g, PublicFeatures &f, const char *&reason) {
  auto fail = [&](const char *message) {
    reason = message;
    return false;
  };
  if (!g.completeFragmentBoundary)
    return fail("complete_fragment_boundary_required");
  if (g.outerFactor != 1)
    return fail("F1_only_outer_factor_not_validated");
  if (g.family != Family::Segments && g.family != Family::Paged)
    return fail("unknown_specialized_family");
  if (g.mode != Mode::SIMD && g.mode != Mode::SIMT)
    return fail("unknown_mode");
  if (g.operation != Operation::Load && g.operation != Operation::Store)
    return fail("unknown_operation");
  if (g.visibility != Visibility::Constexpr &&
      g.visibility != Visibility::Host &&
      g.visibility != Visibility::DeviceHidden)
    return fail("unknown_visibility");
  if (g.mask != Mask::None && g.mask != Mask::Prefix &&
      g.mask != Mask::Rectangular && g.mask != Mask::WholeRows &&
      g.mask != Mask::WholePages)
    return fail("unknown_structured_mask");
  if (g.shape.size() < 2 || g.shape.size() > 5 ||
      g.pattern.size() != g.shape.size() || g.pattern.back() != 'S' ||
      g.pattern.find('D') == std::string::npos ||
      g.pattern.find_first_not_of("DS") != std::string::npos ||
      std::any_of(g.shape.begin(), g.shape.end(),
                  [](int64_t n) { return n <= 0; }))
    return fail("invalid_tail_S_specialized_geometry");
  if (g.warps < 1 || g.warps > 64 || (g.warps & (g.warps - 1)) != 0 ||
      (g.mode == Mode::SIMD && g.warps != 1))
    return fail("invalid_warp_count");
  const std::array<std::pair<std::string_view, int64_t>, 14> types{
      {{"int8", 1},
       {"uint8", 1},
       {"int16", 2},
       {"uint16", 2},
       {"int32", 4},
       {"uint32", 4},
       {"int64", 8},
       {"uint64", 8},
       {"float16", 2},
       {"bfloat16", 2},
       {"float32", 4},
       {"float64", 8},
       {"float8e4nv", 1},
       {"float8e5", 1}}};
  auto type = std::find_if(types.begin(), types.end(), [&](const auto &entry) {
    return entry.first == g.payloadType;
  });
  if (type == types.end() || type->second != g.storageBytes)
    return fail("unsupported_or_inconsistent_payload_storage");
  if (g.mode == Mode::SIMD && g.payloadType == "float64")
    return fail("SIMD_float64_native_toolchain_unsupported");
  if ((g.regularStep && *g.regularStep <= 0) ||
      (g.visibility == Visibility::DeviceHidden && g.regularStep) ||
      (g.visibility != Visibility::DeviceHidden && !g.regularStep))
    return fail("regular_step_visibility_or_proof_invalid");
  for (const auto *coefficients :
       {&g.regularAxisCoefficients, &g.discreteAxisCoefficients}) {
    if (!coefficients->empty() && coefficients->size() != g.shape.size())
      return fail("invalid_axis_coefficient_rank");
    for (const auto &v : *coefficients)
      if (v && *v <= 0)
        return fail("nonpositive_proved_axis_coefficient");
  }
  if (g.family == Family::Paged &&
      (g.pattern != "DS" || g.shape.size() != 2 ||
       (g.mask != Mask::None && g.mask != Mask::WholePages)))
    return fail("unsupported_paged_structure");
  if (g.family == Family::Segments && g.mask == Mask::WholePages)
    return fail("page_mask_on_segments");
  for (const auto &value :
       {g.regularStep, g.maskBound, g.pageSize, g.logicalOrigin})
    if (value && (*value > std::numeric_limits<int32_t>::max() ||
                  *value < std::numeric_limits<int32_t>::min()))
      return fail("public_scalar_outside_i32_launch_ABI");
  if (g.mask == Mask::None && g.maskBound)
    return fail("mask_bound_without_mask");
  if (g.logicalOriginIsLiteral &&
      (g.visibility != Visibility::Constexpr || !g.logicalOrigin))
    return fail("literal_origin_proof_invalid");
  if (bool(g.pageSize) != bool(g.logicalOrigin) ||
      (g.pageSize && (*g.pageSize <= 0 || *g.logicalOrigin < 0 ||
                      *g.logicalOrigin >= *g.pageSize)))
    return fail("invalid_logical_page_geometry");
  if (g.family == Family::Paged && g.maskBound && !g.pageSize)
    return fail("paged_mask_bound_requires_proved_origin_and_page_size");

  const double E = product(g.shape, 0, g.shape.size());
  const double B = static_cast<double>(g.storageBytes);
  // Protect exact integer-shaped geometry before converting it to doubles.
  if (!std::isfinite(E) || E * B > 9007199254740991.0)
    return fail("public_geometry_not_exactly_representable");
  const size_t lastD = g.pattern.rfind('D');
  const double R = product(g.shape, 0, lastD + 1);
  const double segmentBytes = product(g.shape, lastD + 1, g.shape.size()) * B;
  const double logW = std::log2(static_cast<double>(g.warps));
  f.discreteAxes = std::count(g.pattern.begin(), g.pattern.end(), 'D');
  f.regularAxes = std::count(g.pattern.begin(), g.pattern.end(), 'S');
  f.unitStep = g.regularStep && *g.regularStep == 1;
  f.base = {1.0,
            E,
            E * B,
            R,
            segmentBytes,
            g.regularStep ? std::log2(1.0 + *g.regularStep) * E * B : 0.0,
            logW,
            double(g.mask != Mask::None)};
  double pageEntries = 0.0;
  if (g.family == Family::Paged && g.pageSize)
    pageEntries = static_cast<double>(
        (*g.logicalOrigin + g.shape[0] - 1) / *g.pageSize + 1);
  const bool known = g.mask == Mask::None || bool(g.maskBound);
  double active = g.mask == Mask::None ? E : 0.0;
  if (g.mask != Mask::None && g.maskBound) {
    const double bound = static_cast<double>(*g.maskBound);
    if (g.mask == Mask::Prefix)
      active = std::clamp(bound, 0.0, E);
    else if (g.mask == Mask::WholeRows)
      active = std::clamp(bound, 0.0, double(g.shape[0])) *
               product(g.shape, 1, g.shape.size());
    else if (g.mask == Mask::Rectangular)
      active = std::clamp(bound, 0.0, double(g.shape.back())) *
               product(g.shape, 0, g.shape.size() - 1);
    else
      active = std::clamp(bound * *g.pageSize - *g.logicalOrigin, 0.0,
                          double(g.shape[0])) *
               double(g.shape[1]);
  }
  const std::vector<double> extra = {active * B, pageEntries, double(!known)};
  std::vector<double> affine(8, 0.0);
  bool regularKnown = g.family == Family::Segments;
  for (size_t i = 0; i < g.shape.size(); ++i)
    if (g.pattern[i] == 'S' && g.shape[i] > 1)
      regularKnown &= g.regularAxisCoefficients.size() == g.shape.size() &&
                      bool(g.regularAxisCoefficients[i]);
  if (g.family == Family::Segments && !regularKnown)
    affine[7] = 1.0;
  if (regularKnown) {
    double axisCount = 0, nonunit = 0, span = 1, logSum = 0;
    double logMax = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < g.shape.size(); ++i) {
      if (g.pattern[i] != 'S' || g.shape[i] == 1)
        continue;
      const double slope = static_cast<double>(*g.regularAxisCoefficients[i]);
      const double trailing = product(g.shape, i + 1, g.shape.size());
      const double log = std::log2(slope / trailing);
      ++axisCount;
      nonunit += slope != trailing;
      span += (g.shape[i] - 1) * slope;
      logSum += log;
      logMax = std::max(logMax, log);
    }
    double leading = 1.0;
    for (size_t i = 0; i < g.shape.size() && g.pattern[i] == 'S'; ++i)
      leading *= static_cast<double>(g.shape[i]);
    affine = {axisCount, nonunit,
              span * B,  std::log2(span * B),
              logSum,    std::isfinite(logMax) ? logMax : 0.0,
              leading,   0.0};
    double contiguous = 1.0;
    for (size_t i = g.shape.size(); i-- > 0;) {
      if (g.pattern[i] != 'S')
        break;
      if (g.shape[i] == 1)
        continue;
      if (!g.regularAxisCoefficients[i] ||
          double(*g.regularAxisCoefficients[i]) != contiguous)
        break;
      contiguous *= double(g.shape[i]);
    }
    const size_t last = g.shape.size() - 1;
    const auto slope = g.regularAxisCoefficients.size() == g.shape.size()
                           ? g.regularAxisCoefficients[last]
                           : std::nullopt;
    const bool unit = g.pattern[last] == 'S' &&
                      (g.shape[last] == 1 || (slope && *slope == 1));
    const double runs = unit ? E / contiguous : 0.0;
    {
      const bool wide = contiguous * B >= 32;
      f.run = std::array<double, 7>{
          1.0,
          double(f.discreteAxes),
          unit && !wide ? runs : 0.0,
          unit && wide ? runs : 0.0,
          unit ? 0.0 : nonunit,
          slope && *slope > 1 ? std::log2(1.0 + *slope) : 0.0,
          unit && wide ? E : 0.0};
    }
  }
  if (g.family == Family::Segments && g.mask == Mask::None &&
      f.discreteAxes >= 2 &&
      g.discreteAxisCoefficients.size() == g.shape.size()) {
    double ratio = 0.0;
    bool allKnown = true;
    for (size_t i = 0; i < g.shape.size(); ++i) {
      if (g.pattern[i] != 'D')
        continue;
      if (!g.discreteAxisCoefficients[i]) {
        allKnown = false;
        break;
      }
      ratio = std::max(ratio, double(*g.discreteAxisCoefficients[i]) /
                                  product(g.shape, i + 1, g.shape.size()));
    }
    if (allKnown) {
      if (ratio < 1.0 || !std::isfinite(ratio))
        return fail("invalid_static_discrete_pitch_proof");
      f.discretePitchWork = R * (ratio - 1.0);
    }
  }
  auto append = [](std::vector<double> &a, const std::vector<double> &b) {
    a.insert(a.end(), b.begin(), b.end());
  };
  f.affine = f.base;
  append(f.affine, extra);
  append(f.affine, affine);
  const bool pageMasked = g.family == Family::Paged && g.mask != Mask::None;
  const bool pageOriginKnown = !pageMasked || g.logicalOriginIsLiteral;
  const std::vector<double> page = {
      pageMasked && pageOriginKnown && *g.logicalOrigin == 0 ? active * B : 0.0,
      double(pageMasked && !pageOriginKnown)};
  f.page13 = f.base;
  append(f.page13, extra);
  append(f.page13, page);
  const double host = g.visibility == Visibility::Host;
  const double hidden = g.visibility == Visibility::DeviceHidden;
  const double fraction = known ? active / E : 0.0;
  const double zero = g.mask != Mask::None && known && active == 0;
  f.rich = f.affine;
  append(f.rich,
         {host, hidden, double(B == 1), double(B == 2), zero,
          double(f.discreteAxes), double(g.mask == Mask::Prefix),
          double(g.mask == Mask::Rectangular),
          double(g.mask == Mask::WholeRows), E * host, E * hidden, fraction});
  f.simtMasked = {1.,
                  E,
                  active * B,
                  R,
                  f.base[5],
                  logW,
                  double(!known),
                  host,
                  hidden,
                  double(g.mask == Mask::Prefix),
                  double(g.mask == Mask::Rectangular),
                  double(g.mask == Mask::WholeRows),
                  fraction,
                  zero};
  return true;
}

std::string smallRoute(const PublicFeatures &f) {
  const auto &g = f.g;
  const std::string family =
      g.family == Family::Segments ? "segments" : "paged";
  if (g.mask != Mask::None && f.affine[10] == 0 && f.affine[8] == 0)
    return family + "_zero";
  if (g.family == Family::Paged)
    return g.mask == Mask::None ? "paged_unmasked" : "paged_masked";
  if (g.mask != Mask::None)
    return "segments_masked";
  if (g.visibility == Visibility::Host)
    return "segments_host";
  if (g.visibility == Visibility::DeviceHidden)
    return "segments_unknown";
  const auto &run = f.run;
  if (run && (*run)[3] > 0)
    return f.discreteAxes <= 1 ? "segments_unit_few" : "segments_unit_many";
  return "segments_other";
}

std::vector<double> smallVector(const PublicFeatures &f, std::string_view name,
                                double rowThreshold, bool maskedRows) {
  const auto &g = f.g;
  const auto &b = f.base;
  const auto &a = f.affine;
  const double byte1 = g.storageBytes == 1;
  const double fraction = a[10] == 0 ? a[8] / b[2] : 0.0;
  if (name == "segments_zero" || name == "paged_zero")
    return {1.};
  if (name == "paged_unmasked") {
    {
      const double slow = !f.unitStep || b[1] / b[3] < rowThreshold;
      return {1., b[1] * slow, b[2], a[9], b[3] * slow};
    }
  }
  if (name == "paged_masked")
    return {1., maskedRows ? b[3] : b[1], a[8], a[9], a[10], byte1, fraction};
  if (name == "segments_host")
    return {1., b[1], b[2], b[3], b[5], byte1};
  if (name == "segments_unknown")
    return {1., b[1], b[2], b[3], byte1};
  if (name == "segments_masked")
    return {1.,
            maskedRows ? b[3] : b[1],
            b[2],
            a[8],
            b[3],
            fraction,
            byte1,
            double(g.mask == Mask::Prefix),
            double(g.mask == Mask::Rectangular),
            double(g.visibility == Visibility::Host),
            double(g.visibility == Visibility::DeviceHidden),
            a[8] * double(g.visibility == Visibility::Host),
            a[8] * double(g.visibility == Visibility::DeviceHidden)};
  if (name == "segments_unit_few" || name == "segments_unit_many")
    return {1., b[1], b[2], b[3], b[4], byte1};
  return {1., b[1], b[2], b[3], b[5], double(f.discreteAxes), byte1};
}

std::string wideRoute(const PublicFeatures &f) {
  const auto &run = f.run;
  if (f.g.family != Family::Segments || f.g.mask != Mask::None || !run ||
      (*run)[3] <= 0)
    return {};
  return f.discreteAxes <= 1 ? "few_discrete_axes" : "several_discrete_axes";
}

double extraValue(const PublicFeatures &f, std::string_view name) {
  const auto &g = f.g;
  const auto &a = f.affine;
  const auto &b = f.base;
  const double E = b[1], W = b[6];
  const bool segments = g.family == Family::Segments,
             unmasked = g.mask == Mask::None;
  if (name == "E")
    return E;
  if (name == "E_log2W")
    return E * W;
  if (name == "E_log2W_hinge2")
    return E * std::max(0., W - 2.);
  if (name == "E_log2W_hinge4")
    return E * std::max(0., W - 4.);
  if (name == "regular_segments")
    return b[3];
  if (name == "segments/public_index_entries") {
    if (!segments || !unmasked)
      return 0.;
    double result = 0.;
    for (size_t i = 0; i < g.shape.size(); ++i)
      if (g.pattern[i] == 'D')
        result += double(g.shape[i]);
    return result;
  }
  if (name == "segments/address_element_discrete_adds")
    return segments && unmasked ? E * std::max(0, f.discreteAxes - 1) : 0.;
  if (name == "segments/address_element_regular_adds")
    return segments && unmasked ? E * std::max(0, f.regularAxes - 1) : 0.;
  if (name == "segments/proved_discrete_pitch_work")
    return f.discretePitchWork;
  if (name == "segments/proved_regular_span")
    return segments && unmasked ? a[13] : 0.;
  if (name == "segments/proved_narrow_unit_runs")
    return segments && unmasked && f.run ? (*f.run)[2] : 0.;
  if (name == "segments/proved_nonunit_extra_regular_axis_work")
    return segments && unmasked && f.run && (*f.run)[2] == 0 && (*f.run)[3] == 0
               ? E * std::max(0, f.regularAxes - 1)
               : 0.;
  if (name == "segments/known_mask_payload_bytes")
    return segments && !unmasked && a[10] == 0 ? a[8] : 0.;
  if (name == "segments/known_mask_fraction")
    return segments && !unmasked && a[10] == 0 ? a[8] / b[2] : 0.;
  if (name == "segments/mask_prefix_indicator")
    return segments && g.mask == Mask::Prefix;
  if (name == "segments/mask_rectangular_indicator")
    return segments && g.mask == Mask::Rectangular;
  if (name == "segments/proved_whole_row_active_segments") {
    if (!segments || g.mask != Mask::WholeRows || g.pattern != "DS" ||
        a[10] != 0)
      return 0.;
    const double rows = a[8] / b[4];
    return std::abs(rows - std::round(rows)) <= 1e-9 && rows >= 0 &&
                   rows <= b[3]
               ? rows
               : 0.;
  }
  if (name == "masked/logical_output_rows")
    return unmasked ? 0. : b[3];
  if (name == "masked/host_logical_output_rows")
    return !unmasked && g.visibility == Visibility::Host ? b[3] : 0.;
  if (name == "masked/hidden_logical_output_rows")
    return !unmasked && g.visibility == Visibility::DeviceHidden ? b[3] : 0.;
  return std::numeric_limits<double>::quiet_NaN();
}

bool matchesHead(std::string_view requested, std::string_view head) {
  return requested == "*" || requested == head ||
         (requested == "segments/*" && head.substr(0, 9) == "segments_");
}

double linear(const std::vector<double> &v,
              std::initializer_list<std::pair<unsigned, double>> terms) {
  double p = 0.;
  for (auto [column, coefficient] : terms)
    p += coefficient * v[column];
  return p;
}
struct Adjustment {
  const char *head, *feature;
  double coefficient;
};
double adjustments(const PublicFeatures &f, std::string_view head,
                   std::initializer_list<Adjustment> terms) {
  double p = 0.;
  for (auto t : terms)
    if (matchesHead(t.head, head))
      p += t.coefficient * extraValue(f, t.feature);
  return p;
}

// Frozen 950PR coefficients; no runtime model-file access.
#include "SpecializedMemoryParameters.inc"
} // namespace

SpecializedMemoryEstimate estimateSpecializedMemory(const Geometry &g) {
  SpecializedMemoryEstimate result;
  result.modelSHA256 = getSpecializedMemoryModelSHA256();
  PublicFeatures features{g};
  if (!buildFeatures(g, features, result.reason))
    return result;
  const unsigned group = (g.mode == Mode::SIMT ? 2u : 0u) +
                         (g.operation == Operation::Store ? 1u : 0u);
  if (g.directThreePointerEnvelope &&
      (g.family != Family::Segments || g.pattern != "DS" ||
       g.mask != Mask::None || g.payloadType != "float32" ||
       g.regularStep != std::optional<int64_t>(1) || g.warps != 1 ||
       g.visibility != Visibility::Constexpr)) {
    result.reason = "direct_fragment_envelope_not_validated";
    return result;
  }
  double value = predictFrozenGroup(features, group);
  if (g.directThreePointerEnvelope && group == 0)
    value += DirectFragmentStartup;
  if (!std::isfinite(value) || value <= 0.) {
    result.reason = "nonpositive_or_nonfinite_formula";
    return result;
  }
  result.supported = true;
  result.microseconds = value;
  result.reason = "supported";
  return result;
}

const char *getSpecializedMemoryModelSHA256() { return FrozenModelSHA256; }
unsigned getSpecializedMemoryParameterCount(Mode mode, Operation operation) {
  if ((mode != Mode::SIMD && mode != Mode::SIMT) ||
      (operation != Operation::Load && operation != Operation::Store))
    return 0;
  return FrozenParameterCounts[(mode == Mode::SIMT ? 2u : 0u) +
                               (operation == Operation::Store ? 1u : 0u)];
}
} // namespace mlir::ascend
