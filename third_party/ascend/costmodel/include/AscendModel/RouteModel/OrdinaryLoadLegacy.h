// Existing TRAIN-fitted Load coefficients. Export only; no refit.
#ifndef ORDINARY_LOAD_LEGACY_H
#define ORDINARY_LOAD_LEGACY_H
#include "OrdinaryStoreSelected.h"
namespace indirect_load_legacy {
using Metadata = indirect_store_selected::Metadata;
using Estimate = indirect_store_selected::Estimate;
using Mode = indirect_store_selected::Mode;
using Term = indirect_store_selected::Term;
using Formula = indirect_store_selected::Formula;
inline constexpr bool productionEnabled = true;
inline constexpr const char *simdSourceSha256 =
    "1c3bce7f57ef10c37ac7e5102b0ab391dac7d847e1688432384feca7538bd710";
inline constexpr Term simdTerms0[] = {
    {"one", -1218.1474234456643},    {"KlogStride", 1.6604159467629864},
    {"EAbove64", 5.804748967039562}, {"logDstride", -33.27708112683776},
    {"logK", 223.73309879229276},    {"tailMasked", 750.1125982444346},
    {"hidden", 575.2267783432326},   {"KlogDstride2", -0.30377688241806816},
};
inline constexpr Term simdTerms1[] = {
    {"one", 115.35334162735573},      {"KlogStride", 0.6683464665017559},
    {"hidden", 464.56006880663347},   {"K", 6.531401428005312},
    {"sqrtD", 77.89397668940744},     {"logDstride", -99.44794731447354},
    {"KlogTail", 0.6058686862966182}, {"activeRows", 5.192608311895434},
};
inline constexpr Term simdTerms2[] = {
    {"one", 335.4209269099979},
    {"K", 5.402953728840071},
    {"partialLogical32", -46.427734328381824},
    {"KsumInvActiveD", 13.202835348365257},
    {"leading", -13.863715556016663},
    {"KlogTail", 1.9661633842665984},
    {"logDstride", -331.92687921087656},
    {"maxSlog", 22.55457402720241},
};
inline constexpr Term simdTerms3[] = {
    {"one", -28.91550447938594},
    {"KAbove32", 8.410437138128525},
    {"nD", 372.1073030571015},
    {"KlogTail", 0.6569222305871572},
    {"KlogDstride", -3.9887406474688305},
    {"KlogStride", 0.24527696249468053},
};
inline constexpr Term simdTerms4[] = {
    {"one", 739.7875270730963},
    {"EAbove32", 21.679226906748568},
    {"KlogTail", 1.7763040925564038},
    {"hidden", 581.8827121721006},
};
inline constexpr Term simdTerms5[] = {
    {"one", 1011.7306376173069},
    {"KAbove32", 19.96067844860511},
    {"activeRows", -44.07311426911842},
    {"KAbove16", 19.34796702476496},
};
inline constexpr Term simdTerms6[] = {
    {"one", 132.10990897548822},         {"KnD", 2.3345439663185514},
    {"hidden", 540.0611120269323},       {"KmaxSlog", 0.8291233166401781},
    {"activeDlog", -23.869251182629295},
};
inline constexpr Term simdTerms7[] = {
    {"one", 506.5589068719385},       {"KnD", 1.9047812659229872},
    {"hidden", 291.26974587304045},   {"Kbool", 0.46413886063371723},
    {"KlogTail", 0.5728496445832683}, {"logTail", -104.80191247522849},
    {"leading", -14.952955530391678},
};
inline constexpr Term simdTerms8[] = {
    {"one", -202.61673175100182},
    {"KmaskedAxes", 13.085044068168388},
    {"groups32", 253.16518646751427},
};
inline constexpr Term simdTerms9[] = {
    {"one", -216.68939756648578},       {"KlogDstride", 8.423489273707496},
    {"KmaxSlog", 0.6545428527160453},   {"KlogDstride2", -1.318517324624459},
    {"KmaskedAxes", 4.183832554594925}, {"hidden", 688.9059014243014},
    {"nD", 269.56937282359524},
};
inline constexpr Formula simdFormulas[] = {
    {"4_True_False", simdTerms0, 8}, {"4_False_False", simdTerms1, 8},
    {"2_True_False", simdTerms2, 8}, {"2_False_False", simdTerms3, 6},
    {"8_True_False", simdTerms4, 4}, {"8_False_False", simdTerms5, 4},
    {"1_True_False", simdTerms6, 5}, {"1_False_False", simdTerms7, 7},
    {"4_True_True", simdTerms8, 3},  {"fallback", simdTerms9, 7},
};
inline constexpr const char *simtSourceSha256 =
    "286824a6e6c9334e2aa3402209fc1d251bb091a1f8c16ec332ed1a2a465ae8a2";
inline constexpr Term simtTerms0[] = {
    {"one", 127.61542792792788},
    {"KactiveSlog", 0.16439681869369366},
    {"logW", -6.823761261261227},
};
inline constexpr Term simtTerms1[] = {
    {"one", 114.80783800890134},
    {"KactiveSlog", 0.16042114461097892},
    {"activeWarpProxy", -29.980183358162492},
    {"logW", 28.0686467045722},
};
inline constexpr Term simtTerms2[] = {
    {"one", 119.96333333333335},
    {"KactiveSlog", -0.2076692708333333},
    {"leading", -14.104166666666666},
    {"Krank", 0.6496419270833332},
};
inline constexpr Term simtTerms3[] = {
    {"one", 90.63277174204367},
    {"Krank", 0.2763278036517837},
    {"leading", -5.5840502237022065},
    {"KlogStride", 0.010615974121984072},
};
inline constexpr Term simtTerms4[] = {
    {"one", 173.91666666666669},
    {"leading", -2.8541666666666665},
};
inline constexpr Term simtTerms5[] = {
    {"one", 49.75167013442798},
    {"suffixD", 20.927253092547236},
};
inline constexpr Term simtTerms6[] = {
    {"one", 135.6400804507522},
    {"KlogStride", 0.33135335078317396},
    {"KmaxSlog", -0.11691122844875931},
};
inline constexpr Term simtTerms7[] = {
    {"one", 167.08288051610364},
    {"KlogTail", 0.2467031683229735},
    {"KoverSqrtActiveD", -0.45953545434444315},
    {"suffixD", 5.177600629248739},
};
inline constexpr Term simtTerms8[] = {
    {"one", 135.4937150024541},
    {"Krank", 0.026952471637546573},
    {"activeLogical32", 6.5463263386283685},
    {"KlogDstride2", -0.05829904659547497},
    {"KlogDstride", 0.21686373539816697},
};
inline constexpr Term simtTerms9[] = {
    {"one", 165.0956203105217},          {"hidden", -121.59562031052171},
    {"KlogDstride", 0.4697244264755573}, {"maxSlog", -3.256673787249455},
    {"KlogW2", -0.03515625000000004},
};
inline constexpr Term simtTerms10[] = {
    {"one", 112.28760480208716},
    {"Krank", 0.2526258669667858},
    {"sqrtK", -4.287634400528372},
};
inline constexpr Term simtTerms11[] = {
    {"one", 127.52280254064283},        {"Krank", 0.18732647412355646},
    {"KlogTail", 0.02890526903877666},  {"KlogW", -0.04521556668452322},
    {"logDstride", -6.455781447272833},
};
inline constexpr Term simtTerms12[] = {
    {"one", 65.31095707680231},        {"KlogDstride", 0.5113625794505182},
    {"leading", 35.46921174136866},    {"KoverSqrtActiveD", 1.4914910915045192},
    {"KlogTail", 0.17237137184175189}, {"KlogStride", -0.06555658439961404},
};
inline constexpr Term simtTerms13[] = {
    {"one", 58.795213602956835},
    {"Krank", 0.6746914326690584},
    {"hidden", 5.406619160122722},
    {"KlogStride", -0.03218614127618382},
    {"KlogDstride", 0.2780655958162735},
    {"KoverSqrtActiveD", -0.9098773288809474},
};
inline constexpr Term simtTerms14[] = {
    {"one", 124.2516375432288},
    {"activeLogical32", 4.9104632847748935},
    {"logDstride", 8.592680724008813},
    {"Krank", 0.2299188183618695},
    {"tailMasked", 30.979482460659835},
    {"nS", -23.89436696998389},
    {"KsmallActiveD", -0.1482974288414639},
    {"activeWarpProxy", -8.36513480472331},
};
inline constexpr Formula simtFormulas[] = {
    {"1_True_True", simtTerms0, 3},    {"1_False_True", simtTerms1, 4},
    {"2_True_True", simtTerms2, 4},    {"2_False_True", simtTerms3, 4},
    {"1_True_False", simtTerms4, 2},   {"1_False_False", simtTerms5, 2},
    {"2_True_False", simtTerms6, 3},   {"2_False_False", simtTerms7, 4},
    {"4_True_True", simtTerms8, 5},    {"4_True_False", simtTerms9, 5},
    {"4_False_False", simtTerms10, 3}, {"4_False_True", simtTerms11, 5},
    {"8_True_True", simtTerms12, 6},   {"8_False_True", simtTerms13, 6},
    {"fallback", simtTerms14, 8},
};
inline Estimate
tryPredict(const Metadata &m,
           const std::vector<int64_t> *runtimeCoefficients = nullptr,
           bool enabled = productionEnabled) {
  auto fail = [](const char *why) {
    Estimate e;
    e.reason = why;
    return e;
  };
  if (!enabled)
    return fail("Load_formula_explicitly_disabled");
  if (!m.stageOwnershipVerified || !m.matchedIncrementContractVerified)
    return fail("Load_target_partition_or_increment_contract_missing");
  std::string reason;
  auto features =
      indirect_store_selected::features(m, runtimeCoefficients, reason);
  if (!features) {
    Estimate e;
    e.reason = reason;
    return e;
  }
  if (features->count("semantic_zero")) {
    Estimate e;
    e.eligible = true;
    e.ticks = 0.;
    e.reason = "semantic_zero";
    return e;
  }
  const bool simd = m.mode == Mode::SIMD;
  const bool prefix = [&] {
    for (size_t i = 0; i + 1 < m.shape.size(); ++i)
      if (m.upper[i] - m.lower[i] != m.shape[i])
        return true;
    return false;
  }();
  std::string key =
      std::to_string(m.elementBytes) + "_" + (m.hasMask ? "True" : "False") +
      "_" + ((simd ? prefix : features->at("nS") == 1) ? "True" : "False");
  const Formula *bank = simd ? simdFormulas : simtFormulas;
  const size_t count = simd ? std::size(simdFormulas) : std::size(simtFormulas);
  const Formula *formula = &bank[count - 1];
  for (size_t i = 0; i < count; ++i)
    if (key == bank[i].key) {
      formula = &bank[i];
      break;
    }
  double value = 0.;
  for (size_t i = 0; i < formula->size; ++i) {
    const auto &term = formula->terms[i];
    auto it = features->find(term.name);
    if (it == features->end() || !std::isfinite(it->second))
      return fail("Load_feature_missing_or_nonfinite");
    value += term.coefficient * it->second;
  }
  if (!std::isfinite(value))
    return fail("Load_prediction_nonfinite");
  Estimate e;
  e.eligible = true;
  e.ticks = std::max(1., value);
  e.leaf = formula->key;
  e.reason = "existing_Load_selected_formula";
  return e;
}
} // namespace indirect_load_legacy
#endif
