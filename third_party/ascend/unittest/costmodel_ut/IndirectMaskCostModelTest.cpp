#include "AscendModel/Analysis/StagePartitioner.h"
#include "IndirectMemoryTestUtils.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(IndirectMaskCostModelTest, RankFactorPrefixAndExplicitRandomPrior) {
  IRTestContext context(true);
  const IndirectGatherMemoryCostModel model;
  for (bool store : {false, true})
    for (bool random : {false, true})
      for (int64_t rank = 1; rank <= 5; ++rank)
        for (int64_t factor : {1, 2, 4}) {
          const int64_t fi = factor == 4 ? 2 : factor == 2 ? 1 : 0;
          const int64_t e = 32 * (1LL << ((rank + 2 + fi) % 3));
          SmallVector<int64_t> shape(rank, 2);
          shape.back() = e / (1LL << (rank - 1));
          auto module = memoryModule(context, shape, "f32", store,
                                     random ? "loaded" : "prefix", e / 2);
          ASSERT_TRUE(module);
          MemoryCase test(context, Float32Type::get(&context), shape, store);
          module->walk([&](Operation *op) {
            if (op->getName().getStringRef() ==
                (store ? "tt.store" : "tt.load"))
              test.stage.operations = {op};
          });
          auto profile = calibrationProfile();
          profile.simtIndirectLoadModel = profile.simtIndirectStoreModel =
              "prefix_mask_rank_factor_20261017";
          EXPECT_TRUE(profile.isValid());
          const auto prefix =
              model.cost(test.stage, profile, {StageMode::SIMT, factor});
          EXPECT_EQ(bool(store ? prefix.store : prefix.load), !random);
          profile.simtIndirectLoadModel = profile.simtIndirectStoreModel =
              "random_mask_rank_factor_20261017";
          const auto result =
              model.cost(test.stage, profile, {StageMode::SIMT, factor});
          ASSERT_TRUE(store ? result.store : result.load);
          EXPECT_GT(*(store ? result.store : result.load), 0);
          test.stage.iterationCount = 1;
          test.stage.legalSimtFactors = {factor};
          test.stage.workload.operationCounts.clear();
          test.stage.workload.issueElements = 0;
          for (const auto &cost : indirectCosts(test.stage, profile)) {
            if (cost.implementation.mode != StageMode::SIMT)
              continue;
            const auto price =
                model.cost(test.stage, profile, cost.implementation);
            ASSERT_TRUE(store ? price.store : price.load);
            const double memory = *(store ? price.store : price.load);
            EXPECT_NEAR(store ? cost.resources.store : cost.resources.load,
                        memory, 1e-10);
            EXPECT_NEAR(cost.totalCycles,
                        cost.resources.setup +
                            memory * cost.implementation.superblockFactor,
                        1e-8);
          }
          EXPECT_FALSE(
              model.cost(test.stage, profile, {StageMode::SIMT, 8}).load);
          EXPECT_FALSE(
              model.cost(test.stage, profile, {StageMode::SIMT, 8}).store);
          profile.baseSimtWarpCount = 16; // q<1 must return to #2528.
          const auto small =
              model.cost(test.stage, profile, {StageMode::SIMT, factor});
          EXPECT_FALSE(small.load || small.store);
        }
}

TEST(IndirectMaskCostModelTest, PrefixRejectsWrongCountAndReuse) {
  IRTestContext context(true);
  MemoryCase test(context, Float32Type::get(&context), {32}, true);
  auto profile = calibrationProfile();
  profile.simtIndirectStoreModel = "prefix_mask_rank_factor_20261017";
  const IndirectGatherMemoryCostModel model;
  for (int64_t k : {8, 16}) {
    auto module = memoryModule(context, {32}, "f32", true, "prefix", k);
    ASSERT_TRUE(module);
    module->walk([&](triton::StoreOp op) { test.stage.operations = {op}; });
    auto result = model.cost(test.stage, profile, {StageMode::SIMT, 1});
    EXPECT_EQ(bool(result.store), k == 16); // q=1 is included.
    if (result.store)
      EXPECT_NEAR(*result.store, 166.52232075147882, 1e-10);
    test.stage.workload.hasProvenIndirectStoreReuse = true;
    EXPECT_FALSE(model.cost(test.stage, profile, {StageMode::SIMT, 1}).store);
    test.stage.workload.hasProvenIndirectStoreReuse = false;
  }
}

TEST(IndirectMaskCostModelTest, PartitionedRankThreeUsesMaskFormula) {
  IRTestContext context(true);
  for (bool store : {false, true}) {
    auto module = memoryModule(context, {2, 2, 32}, "f32", store, "prefix", 64);
    ASSERT_TRUE(module);
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition));
    auto profile = calibrationProfile();
    profile.simtIndirectLoadModel = profile.simtIndirectStoreModel =
        "prefix_mask_rank_factor_20261017";
    unsigned hits = 0;
    for (const auto &stage : partition->stages) {
      auto result = IndirectGatherMemoryCostModel().cost(stage, profile,
                                                         {StageMode::SIMT, 1});
      hits += bool(store ? result.store : result.load);
    }
    EXPECT_EQ(hits, 1u);
  }
}
