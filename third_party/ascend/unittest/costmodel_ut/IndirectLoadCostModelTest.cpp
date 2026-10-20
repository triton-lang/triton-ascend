// Formula, resource ownership and domain tests; IR construction is shared.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "IndirectMemoryTestUtils.h"
#include <functional>

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(IndirectLoadCostModelTest,
     FactorCandidateUsesGroupUnitsAndRequiresIndexEvidence) {
  IRTestContext c(false, true);
  MemoryCase t(c, Float32Type::get(&c), {128});
  auto p = calibrationProfile();
  p.baseSimtWarpCount = 4;
  p.simtIndirectLoadModel = "unmasked_factor_candidate_20261017";
  const IndirectGatherMemoryCostModel model;
  EXPECT_FALSE(model.cost(t.stage, p, {StageMode::SIMT, 4}).load);
  t.addIndex();
  t.stage.iterationCount = 1;
  t.stage.legalSimtFactors = {1, 2, 4};
  t.stage.workload.operationCounts.clear();
  t.stage.workload.issueElements = 0;
  for (const auto &cost : indirectCosts(t.stage, p)) {
    auto impl = cost.implementation;
    if (impl.mode != StageMode::SIMT)
      continue;
    auto fit = model.cost(t.stage, p, impl);
    ASSERT_TRUE(fit.load);
    double expected = 120.07090406650597 + 1.5500745996624559 * 128 -
                      .43870878244183104 * 96 + 1.640631714734442 * 2 +
                      .15541731034924547 * (impl.superblockFactor - 1);
    EXPECT_NEAR(*fit.load * impl.superblockFactor, expected, 1e-10);
    EXPECT_NEAR(cost.totalCycles, cost.resources.setup + expected, 1e-8);
    EXPECT_STREQ(fit.loadPricing, "unmasked_factor_candidate_20261017");
  }
  for (StageImplementation impl :
       {StageImplementation{StageMode::SIMT, 4, true},
        {StageMode::SIMT, 4, false, 2}})
    EXPECT_FALSE(model.cost(t.stage, p, impl).load);
  t.stage.workload.addressPatterns[0].axes[0].offsetAxisInfo = "structured";
  EXPECT_FALSE(model.cost(t.stage, p, {StageMode::SIMT, 4}).load);
}

TEST(IndirectLoadCostModelTest, FitsReplaceOnlyTheTargetResource) {
  IRTestContext c(false, true);
  struct Case {
    const char *model;
    bool simd;
    int e;
    double expected;
  };
  for (auto row :
       {Case{"random_f32_matched_ab_20261007", true, 32,
             83.56748010753823 * 32},
        {"random_dtype_matched_ab_20261008", true, 32, 81.94869549595556 * 32},
        {"random_i32_six_term_20261007", false, 256,
         62.930686069640686 + 1.5502588407166296 * 256 +
             .19338028618547126 * 256}}) {
    MemoryCase t(c,
                 row.simd ? Type(Float32Type::get(&c))
                          : Type(IntegerType::get(&c, 32)),
                 {row.e});
    auto &w = t.stage.workload;
    w.scalarOperations = 7;
    w.storeBytes = 16;
    w.storeWarpInstructions = 2;
    auto p = calibrationProfile();
    p.baseSimtWarpCount = 4;
    auto before = indirectCosts(t.stage, p);
    (row.simd ? p.simdIndirectLoadModel : p.simtIndirectLoadModel) = row.model;
    auto after = indirectCosts(t.stage, p);
    unsigned mode = row.simd ? 0 : 1;
    EXPECT_NEAR(after[mode].resources.load, row.expected, 1e-9);
    expectOnlyMemoryChanged(before[mode], after[mode]);
    EXPECT_DOUBLE_EQ(before[1 - mode].totalCycles, after[1 - mode].totalCycles);
    (row.simd ? p.simd : p.simt).indirectDependencyLatencyCycles = 10000;
    EXPECT_DOUBLE_EQ(indirectCosts(t.stage, p)[mode].totalCycles,
                     after[mode].totalCycles);
  }
}

TEST(IndirectLoadCostModelTest, FitReplacesUpstreamCurveAndRejectsCvPartition) {
  IRTestContext c(false, true);
  MemoryCase t(c, Float32Type::get(&c), {128});
  auto &w = t.stage.workload;
  w.loadOperations = 1;
  w.loadWarpInstructions = w.indirectLoadTransactions = 4;
  w.indirectLoadCounts[{512, 128}] = 1;
  w.scalarLoadCount = 1;
  auto p = calibrationProfile();
  p.baseSimtWarpCount = 4;
  p.simt.simtUniformLoadPrepCycles = 5;
  p.simtIndirectLoadModel = "random_dtype_six_term_20261008";
  p.simt.indirectLoadSystemCyclesByElements[128] = 100000;
  const IndirectGatherMemoryCostModel model;
  auto fit = model.cost(t.stage, p, {StageMode::SIMT, 1});
  ASSERT_TRUE(fit.load);
  auto withHelper = indirectCosts(t.stage, p)[1];
  EXPECT_DOUBLE_EQ(withHelper.resources.indirectLoadGroup, 0);
  w.scalarLoadCount = 0;
  EXPECT_DOUBLE_EQ(indirectCosts(t.stage, p)[1].resources.load, *fit.load);
  EXPECT_GT(withHelper.resources.load, *fit.load);
  EXPECT_FALSE(model.cost(t.stage, p, {StageMode::SIMT, 1, true, 2}).load);
  for (StringRef axis : {"structured", "unknown", ""}) {
    w.addressPatterns[0].axes[0].offsetAxisInfo = axis.str();
    EXPECT_FALSE(model.cost(t.stage, p, {StageMode::SIMT, 1}).load);
  }
}

TEST(IndirectLoadCostModelTest, SimdDtypeRankAndLegacyAxisDomains) {
  IRTestContext c(false, true);
  const IndirectGatherMemoryCostModel model;
  auto p = calibrationProfile();
  p.simdIndirectLoadModel = "random_dtype_matched_ab_20261008";
  for (Type element : payloadTypes(c))
    for (const SmallVector<int64_t> &shape :
         SmallVector<SmallVector<int64_t>>{{4},
                                           {8},
                                           {16},
                                           {32},
                                           {2048},
                                           {8, 16},
                                           {8, 64},
                                           {2, 4, 16},
                                           {2, 2, 4, 8},
                                           {2, 2, 2, 4, 8}}) {
      MemoryCase t(c, element, shape);
      int64_t e =
          cast<RankedTensorType>(t.op->getResult(0).getType()).getNumElements();
      bool small =
          element.getIntOrFloatBitWidth() == 32 && shape.size() == 1 && e <= 16;
      for (int64_t w : {1, 2, 4, 8, 16, 32, 64}) {
        p.baseSimtWarpCount = w;
        auto fit = model.cost(t.stage, p, {StageMode::SIMD, 1});
        ASSERT_TRUE(fit.load);
        EXPECT_DOUBLE_EQ(*fit.load,
                         e * (small ? 49.57621548794853 : 81.94869549595556));
      }
    }
  MemoryCase legacy(c, Float32Type::get(&c), {4, 8});
  p.simdIndirectLoadModel = "random_f32_matched_ab_20261007";
  for (StringRef axis :
       {"fixed_stride", "opaque_loaded", "computed_nonaffine"}) {
    auto &a = legacy.stage.workload.addressPatterns[0].axes[0];
    a.regularity = axis.str();
    a.knownStride = 4096;
    a.offsetAxisInfo = axis == "fixed_stride" ? "structured" : "unstructured";
    auto fit = model.cost(legacy.stage, p, {StageMode::SIMD, 1});
    EXPECT_EQ(bool(fit.load), axis == "opaque_loaded");
    if (fit.load)
      EXPECT_DOUBLE_EQ(*fit.load, 83.56748010753823 * 32);
  }
}

TEST(IndirectLoadCostModelTest, SimtDtypeRankFitUsesStorageWidth) {
  IRTestContext c(false, true);
  const double b[][6] = {
      {7.617911783542814, 1.5780971099377332, 92.18952965944916,
       .19387737354356369, 0, .049471659398567715},
      {12.456944150614481, 1.6732132663368477, 95.13859585355112,
       .11214618741568584, 5.634318857353958, .026588037490546692},
      {50.71936539506111, 1.693396365504775, 60.38681262899275,
       .09265123974451413, 82.91193957489877, .026814982781878355},
      {96.09572807125711, 1.937402636239715, 24.962585867214496,
       .20099293498091206, 41.8669299352394, .03855485734655757}};
  const IndirectGatherMemoryCostModel model;
  auto p = calibrationProfile();
  p.simtIndirectLoadModel = "random_dtype_six_term_20261008";
  for (Type element : payloadTypes(c, true))
    for (int64_t w : {1, 2, 4, 8, 16, 32, 64})
      for (int64_t q : {1, 2, 4})
        for (int64_t rank = 1; rank <= 5; ++rank) {
          int64_t e = 32 * w * q, bytes = element.getIntOrFloatBitWidth() / 8;
          SmallVector<int64_t> shape(rank, 2);
          shape.back() =
              rank == 1 ? e : std::min<int64_t>(32, e / (1LL << (rank - 1)));
          if (rank > 1)
            shape[rank - 2] = e / (shape.back() * (1LL << (rank - 2)));
          MemoryCase t(c, element, shape);
          t.addHelper(rank >= 3);
          p.baseSimtWarpCount = w;
          const auto &a = b[bytes == 1   ? 0
                            : bytes == 2 ? 1
                            : bytes == 4 ? 2
                                         : 3];
          double cols = rank == 1 ? 32 : shape.back();
          double expected = a[0] + a[1] * e + a[2] * (e <= 128) +
                            a[3] * e * (rank == 1) +
                            a[4] * std::max(4.0 * q / cols - 2, 0.0) +
                            a[5] * e * std::max(std::log2(32.0 / cols), 0.0);
          auto fit = model.cost(t.stage, p, {StageMode::SIMT, 1});
          ASSERT_TRUE(fit.load);
          EXPECT_NEAR(*fit.load, expected, 1e-9);
        }
}

TEST(IndirectLoadCostModelTest, RejectsUnsupportedGeometryAndWorkloads) {
  IRTestContext c(false, true);
  const IndirectGatherMemoryCostModel model;
  for (bool simd : {false, true}) {
    auto p = calibrationProfile();
    (simd ? p.simdIndirectLoadModel : p.simtIndirectLoadModel) =
        simd ? "random_dtype_matched_ab_20261008"
             : "random_dtype_six_term_20261008";
    StageImplementation impl{simd ? StageMode::SIMD : StageMode::SIMT, 1};
    for (const SmallVector<int64_t> &shape :
         SmallVector<SmallVector<int64_t>>{{2},
                                           {16},
                                           {4096},
                                           {4, 2},
                                           {2, 128, 2},
                                           {2, 2, 2, 2, 2, 2},
                                           {1, 32},
                                           {2, 256}}) {
      MemoryCase t(c, IntegerType::get(&c, 32), shape);
      p.baseSimtWarpCount = (shape == SmallVector<int64_t>{2, 128, 2} ||
                             shape == SmallVector<int64_t>{2, 256})
                                ? 4
                                : 1;
      // Rank-three narrow tails are covered by SIMD, unlike rank-two C>128.
      bool accepted = simd && (shape == SmallVector<int64_t>{16} ||
                               shape == SmallVector<int64_t>{2, 128, 2});
      EXPECT_EQ(bool(model.cost(t.stage, p, impl).load), accepted);
    }
    p.baseSimtWarpCount = 1;
    MemoryCase t(c, IntegerType::get(&c, 32), {32});
    const std::function<void(LogicalStage &)> guards[] = {
        [](auto &s) { s.workload.predicateCounts[1] = 1; },
        [](auto &s) { s.workload.privateStackLoadInstructions = 1; },
        [](auto &s) { s.features.activeLaneRatio = .5; },
        [](auto &s) { s.workload.indirectStoreBytes = 4; },
        [](auto &s) {
          s.workload.addressPatterns[0].axes[0].regularity = "fixed_stride";
        },
        [](auto &s) {
          s.workload.addressPatterns[0].dependsOnLoadedValue = false;
        },
        [](auto &s) { s.features.hasAtomicMemory = true; }};
    for (auto &guard : guards) {
      auto invalid = t.stage;
      guard(invalid);
      // Atomic exclusion belongs to the legacy SIMT fit.
      if (simd && invalid.features.hasAtomicMemory)
        continue;
      EXPECT_FALSE(model.cost(invalid, p, impl).load);
    }
    auto *second = t.op->clone();
    auto multiple = t.stage;
    multiple.operations.push_back(second);
    EXPECT_FALSE(model.cost(multiple, p, impl).load);
    second->destroy();
    for (Type unsupported :
         {Type(IntegerType::get(&c, 1)), Type(Float8E4M3FNUZType::get(&c)),
          Type(Float64Type::get(&c))}) {
      MemoryCase bad(c, unsupported, {32});
      EXPECT_EQ(bool(model.cost(bad.stage, p, impl).load),
                !simd && unsupported.isF64());
    }
    (simd ? p.simdIndirectLoadModel : p.simtIndirectLoadModel) =
        "unknown_load_fit";
    EXPECT_FALSE(p.isValid());
  }
}

TEST(IndirectLoadCostModelTest, DtypeFitsAcceptPartitionedLoad) {
  IRTestContext c(true);
  auto module = memoryModule(c, {32}, "i16", false);
  ASSERT_TRUE(module);
  auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition));
  auto p = calibrationProfile();
  p.simdIndirectLoadModel = "random_dtype_matched_ab_20261008";
  p.simtIndirectLoadModel = "random_dtype_six_term_20261008";
  unsigned hits = 0;
  for (const auto &stage : partition->stages) {
    if (stage.costModelKind != StageCostModelKind::IndirectGatherMemory)
      continue;
    auto prices = indirectCosts(stage, p);
    EXPECT_DOUBLE_EQ(prices[0].resources.load, 81.94869549595556 * 32);
    EXPECT_NEAR(prices[1].resources.load,
                12.456944150614481 + 1.6732132663368477 * 32 +
                    95.13859585355112 + .11214618741568584 * 32,
                1e-10);
    ++hits;
  }
  EXPECT_EQ(hits, 1u);
}
