// Tests for IndirectStoreCostModel responsibilities.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "IndirectMemoryTestUtils.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(IndirectStoreCostModelTest, RandomIndirectStoreStateAndResourceBoundary) {
  IRTestContext context(false, true);
  MemoryCase memory(context, Float32Type::get(&context), {32}, true);
  auto &stage = memory.stage;
  stage.iterationCount = 1;
  auto &work = stage.workload;
  work.storeBytes = work.indirectStoreBytes = 128;
  work.storeWarpInstructions = work.indirectStoreTransactions = 1;
  work.scalarOperations = 7;
  work.loadBytes = 16;
  work.addressPatterns = {loadedAddressPattern(stage, "tt.store", {32})};
  auto profile = calibrationProfile();
  auto old = indirectCosts(stage, profile);
  profile.simdIndirectStoreModel = "random_f32_store_no_fill_first_20261007";
  profile.simtIndirectStoreModel = "random_f32_store_fill_ab_20261007";
  auto fit = indirectCosts(stage, profile);
  const auto &a = old;
  const auto &b = fit;
  const IndirectGatherMemoryCostModel memoryModel;
  const StageCostModel &model = memoryModel;
  const auto increment =
      model.cost(stage, profile, {StageMode::SIMD, 1, false});
  ASSERT_TRUE(increment.store);
  EXPECT_DOUBLE_EQ(*increment.store, 150.13769870695648 * 32);
  EXPECT_FALSE(increment.load);
  EXPECT_DOUBLE_EQ(b[0].resources.store, 150.13769870695648 * 32);
  EXPECT_NEAR(b[1].resources.store, 145.04451206999323, 1e-9);
  for (unsigned i = 0; i < 2; ++i) {
    expectOnlyMemoryChanged(a[i], b[i], true, 1);
    EXPECT_EQ(b[i].indirectLoadPricing, "legacy_transactions");
    EXPECT_NE(b[i].indirectStorePricing, "legacy_transactions");
  }
  profile.simd.indirectDependencyLatencyCycles = 10000;
  auto changed = indirectCosts(stage, profile);
  EXPECT_DOUBLE_EQ(changed[0].totalCycles, b[0].totalCycles);
  profile.simdIndirectStoreModel = "random_f32_store_no_fill_reuse_20261007";
  changed = indirectCosts(stage, profile);
  EXPECT_EQ(changed[0].indirectStorePricing, "legacy_transactions");
  work.hasProvenIndirectStoreReuse = true;
  changed = indirectCosts(stage, profile);
  EXPECT_DOUBLE_EQ(changed[0].resources.store, 65.26772542192847 * 32);
  work.predicateElements = 1;
  changed = indirectCosts(stage, profile);
  for (const auto &cost : changed)
    EXPECT_EQ(cost.indirectStorePricing, "legacy_transactions");
  profile.simdIndirectStoreModel = "unknown";
  EXPECT_FALSE(profile.isValid());
  profile.simdIndirectStoreModel.clear();
  profile.target = "other-target";
  EXPECT_FALSE(profile.isValid());
}

TEST(IndirectStoreCostModelTest, RandomStoreFitAcceptsPartitionedTritonStore) {
  IRTestContext context(true, false);
  auto module = context.parse(R"mlir(
    module {
      tt.func public @scatter(%x: !tt.ptr<f32>, %idx: !tt.ptr<i32>,
                              %values: !tt.ptr<f32>) {
        %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
        %ip = tt.splat %idx : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
        %ip2 = tt.addptr %ip, %r : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
        %index = tt.load %ip2 : tensor<128x!tt.ptr<i32>>
        %vp = tt.splat %values : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %vp2 = tt.addptr %vp, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %value = tt.load %vp2 : tensor<128x!tt.ptr<f32>>
        %xp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>> loc("scatter.py":15:12)
        %xp2 = tt.addptr %xp, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32> loc("scatter.py":15:12)
        tt.store %xp2, %value : tensor<128x!tt.ptr<f32>> loc("scatter.py":15:15)
        tt.return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto partition = StagePartitioner().partition(
      *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition));
  const LogicalStage *payload = nullptr;
  for (const auto &stage : partition->stages)
    if (stage.workload.indirectStoreBytes > 0)
      payload = &stage;
  ASSERT_NE(payload, nullptr);
  ASSERT_EQ(payload->operations.size(), 3u);
  auto profile = calibrationProfile();
  profile.logicalWarpGroupCount = 1;
  auto old = indirectCosts(*payload, profile);
  profile.simdIndirectStoreModel = profile.simtIndirectStoreModel =
      "random_store_no_fill_first_20261008";
  auto fit = indirectCosts(*payload, profile);
  const double expected[] = {125.41805018354371 + 162.87111975882544 * 128,
                             382.1012717872757 + 1.3935292896269613 * 128 -
                                 19.191779247532597 * 4};
  for (unsigned i = 0; i < 2; ++i) {
    const auto &cost = fit[i];
    const auto &before = old[i];
    EXPECT_EQ(cost.indirectStorePricing, profile.simdIndirectStoreModel);
    EXPECT_NEAR(cost.resources.store, expected[i], 1e-8);
    expectOnlyMemoryChanged(before, cost, true, 1);
  }
  auto multiple = *payload;
  multiple.operations.push_back(payload->operations.back());
  auto rejected = indirectCosts(multiple, profile);
  for (const auto &cost : rejected)
    EXPECT_EQ(cost.indirectStorePricing, "legacy_transactions");
}

TEST(IndirectStoreCostModelTest, RandomStoreDtypeRankStateAndDomain) {
  IRTestContext context(false, true);
  auto p = calibrationProfile();
  for (int bits : {1, 8, 16, 32, 64}) {
    for (int e : {8, 128, 256, 512, 2048}) {
      for (int w : {1, 2, 4, 8, 16, 32, 64}) {
        for (bool reuse : {false, true}) {
          // Rank 1, 3 and 8 exercise shape-independent pricing. High ranks
          // rely on per-axis loaded evidence, not the legacy rank-5 predicate.
          for (bool rank3 : {false, true}) {
            llvm::SmallVector<int64_t> shape =
                rank3 ? llvm::SmallVector<int64_t>{2, 2, e / 4}
                      : llvm::SmallVector<int64_t>{e};
            if (rank3 && e >= 256)
              shape = {2, 2, 2, 2, 2, 2, 2, e / 128};
            MemoryCase memory(context, IntegerType::get(&context, bits), shape,
                              true);
            auto &s = memory.stage;
            auto &a = s.workload;
            a.hasProvenIndirectStoreReuse = reuse;
            a.addressPatterns[0].dependsOnLoadedValue = shape.size() <= 5;
            p.logicalWarpGroupCount = w;
            p.simdIndirectStoreModel = p.simtIndirectStoreModel =
                reuse ? "random_store_no_fill_reuse_20261008"
                      : "random_store_no_fill_first_20261008";
            auto fit = indirectCosts(s, p);
            int g = bits <= 16 ? 0 : (bits == 32 ? 1 : 2);
            const double beta[2][3] = {
                {152.8758408085307, 162.87111975882544, 161.07674811788297},
                {68.58965840703893, 69.21236829277788, 64.45492494749357}};
            const double b[2][3] = {
                {4.664669494263474, 1.3935292896269613, 2.0019961511928295},
                {4.4638287665056, 1.1754145133075695, 2.076648173015399}};
            const double d[2][3] = {
                {-25.526112727120946, -19.191779247532597, -35.604138981973364},
                {-9.072748388478542, -20.88472709201174, -38.4225252759396}};
            bool supported[] = {w == 1,
                                e <= 512 * w && (w == 1 || e >= 32 * w)};
            double expected[] = {
                (reuse ? 0 : 125.41805018354371) + beta[reuse][g] * e,
                (reuse ? 70.0956884939018 : 382.1012717872757) +
                    b[reuse][g] * e +
                    (reuse ? .8774560851959086 : .6797627278581438) *
                        std::max(e - (reuse ? 128 : 256), 0) +
                    d[reuse][g] * e / (32.0 * w)};
            for (unsigned i = 0; i < 2; ++i) {
              if (supported[i]) {
                EXPECT_EQ(fit[i].indirectStorePricing,
                          p.simdIndirectStoreModel);
                EXPECT_NEAR(fit[i].resources.store, expected[i], 1e-7);
              } else
                EXPECT_EQ(fit[i].indirectStorePricing, "legacy_transactions");
            }
            if (reuse) {
              a.hasProvenIndirectStoreReuse = false;
              for (const auto &cost : indirectCosts(s, p))
                EXPECT_EQ(cost.indirectStorePricing, "legacy_transactions");
              a.hasProvenIndirectStoreReuse = true;
            }
            a.addressPatterns[0].axes[0].regularity = "fixed_stride";
            for (const auto &cost : indirectCosts(s, p))
              EXPECT_EQ(cost.indirectStorePricing, "legacy_transactions");
          }
        }
      }
    }
  }
}
