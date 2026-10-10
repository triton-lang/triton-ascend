// Formula and guard tests for the independent indirect memory model.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "IndirectMemoryTestUtils.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(IndirectLoadCostModelTest,
     SimdMatchedIndirectFitReplacesOnlyLoadResource) {
  IRTestContext context(false, true);
  MemoryCase test(context, Float32Type::get(&context), {32});
  auto &stage = test.stage;
  stage.iterationCount = 1;
  auto &work = stage.workload;
  work.loadBytes = work.indirectLoadBytes = 128;
  work.loadWarpInstructions = work.indirectLoadTransactions = 32;
  work.scalarOperations = 7;
  work.storeBytes = 16;
  work.addressPatterns = {loadedAddressPattern(stage, "tt.load", {32})};
  auto profile = calibrationProfile();
  auto old = indirectCosts(stage, profile);
  profile.simdIndirectLoadModel = "random_f32_matched_ab_20261007";
  auto fit = indirectCosts(stage, profile);
  const auto &a = old;
  const auto &b = fit;
  const IndirectGatherMemoryCostModel memoryModel;
  const StageCostModel &model = memoryModel;
  const auto increment =
      model.cost(stage, profile, {StageMode::SIMD, 1, false});
  ASSERT_TRUE(increment.load);
  EXPECT_DOUBLE_EQ(*increment.load, 83.56748010753823 * 32);
  EXPECT_FALSE(increment.store);
  EXPECT_DOUBLE_EQ(b[0].resources.load, 83.56748010753823 * 32);
  expectOnlyMemoryChanged(a[0], b[0], false, 1);
  EXPECT_DOUBLE_EQ(b[1].totalCycles, a[1].totalCycles);
  profile.simd.indirectDependencyLatencyCycles = 10000;
  auto changed = indirectCosts(stage, profile);
  EXPECT_DOUBLE_EQ(changed[0].totalCycles, b[0].totalCycles);
}

TEST(IndirectLoadCostModelTest, SimdMatchedIndirectNarrowRankTwoDomain) {
  IRTestContext context(false, true);
  MemoryCase test(context, Float32Type::get(&context), {4, 8});
  auto &stage = test.stage;
  stage.iterationCount = 1;
  auto &work = stage.workload;
  work.loadBytes = work.indirectLoadBytes = 128;
  work.loadWarpInstructions = work.indirectLoadTransactions = 32;
  mlir::ascend::AddressPatternSummary pattern;
  pattern.memoryOp = "tt.load";
  pattern.stageId = stage.id;
  pattern.dependsOnLoadedValue = true;
  mlir::ascend::AddressAxisSummary outer, inner;
  outer.extent = 4;
  outer.regularity = "fixed_stride";
  outer.knownStride = 4096;
  inner.extent = 8;
  inner.regularity = "opaque_loaded";
  pattern.axes = {outer, inner};
  work.addressPatterns = {pattern};
  auto profile = calibrationProfile();
  profile.simdIndirectLoadModel = "random_f32_matched_ab_20261007";
  for (const char *category : {"fixed_stride", "opaque_loaded"}) {
    work.addressPatterns[0].axes[0].regularity = category;
    auto result = indirectCosts(stage, profile);
    EXPECT_DOUBLE_EQ(result[0].resources.load, 83.56748010753823 * 32);
  }
  work.addressPatterns[0].axes[0].regularity = "computed_nonaffine";
  auto fallback = indirectCosts(stage, profile);
  EXPECT_EQ(fallback[0].indirectLoadPricing, "legacy_transactions");
}

TEST(IndirectLoadCostModelTest, SimdDtypeRankFitPreservesIndependentResources) {
  IRTestContext context(false, true);
  const llvm::SmallVector<mlir::Type> types = {
      mlir::IntegerType::get(&context, 8),
      mlir::IntegerType::get(&context, 16),
      mlir::IntegerType::get(&context, 32),
      mlir::IntegerType::get(&context, 64),
      mlir::Float16Type::get(&context),
      mlir::BFloat16Type::get(&context),
      mlir::Float32Type::get(&context),
      mlir::Float8E4M3FNType::get(&context),
      mlir::Float8E5M2Type::get(&context)};
  const llvm::SmallVector<llvm::SmallVector<int64_t>> shapes = {
      {4},     {8},     {16},       {32},         {2048},
      {8, 16}, {8, 64}, {2, 4, 16}, {2, 2, 4, 8}, {2, 2, 2, 4, 8}};
  for (mlir::Type element : types) {
    for (const auto &shape : shapes) {
      MemoryCase test(context, element, shape);
      auto &stage = test.stage;
      const int64_t elements =
          cast<RankedTensorType>(test.op->getResult(0).getType())
              .getNumElements();
      stage.features.loopBackedgeCount = 1;
      auto &work = stage.workload;
      work.scalarOperations = 7;
      work.storeBytes = 16;
      work.storeWarpInstructions = 2;
      work.addressPatterns = {
          loadedAddressPattern(stage, "tt.load", shape, "opaque")};
      auto profile = calibrationProfile();
      auto legacy = indirectCosts(stage, profile);
      profile.simdIndirectLoadModel = "random_dtype_matched_ab_20261008";
      auto calibrated = indirectCosts(stage, profile);
      const auto &old = legacy;
      const auto &fit = calibrated;
      const bool small32 = element.getIntOrFloatBitWidth() == 32 &&
                           shape.size() == 1 && elements <= 16;
      EXPECT_EQ(fit[0].indirectLoadPricing, profile.simdIndirectLoadModel);
      EXPECT_DOUBLE_EQ(fit[0].resources.load,
                       elements *
                           (small32 ? 49.57621548794853 : 81.94869549595556));
      expectOnlyMemoryChanged(old[0], fit[0]);
      EXPECT_DOUBLE_EQ(fit[1].totalCycles, old[1].totalCycles);
      profile.simd.indirectDependencyLatencyCycles = 10000;
      for (int64_t warps : {1, 2, 4, 8, 16, 32, 64}) {
        profile.logicalWarpGroupCount = warps;
        auto result = indirectCosts(stage, profile);
        EXPECT_DOUBLE_EQ(result[0].totalCycles, fit[0].totalCycles);
      }
      work.predicateElements = 1;
      auto masked = indirectCosts(stage, profile);
      EXPECT_EQ(masked[0].indirectLoadPricing, "legacy_transactions");
      work.predicateElements = 0;
      work.estimatedSpillTransactions = 1;
      auto spilled = indirectCosts(stage, profile);
      EXPECT_EQ(spilled[0].indirectLoadPricing, "legacy_transactions");
    }
  }
}

TEST(IndirectLoadCostModelTest, SimdDtypeRankFitRejectsUnvalidatedDomains) {
  IRTestContext context(false, true);
  const llvm::SmallVector<std::pair<mlir::Type, llvm::SmallVector<int64_t>>>
      cases = {{mlir::Float64Type::get(&context), {32}},
               {mlir::IntegerType::get(&context, 1), {32}},
               {mlir::Float8E4M3FNUZType::get(&context), {32}},
               {mlir::IntegerType::get(&context, 32), {2}},
               {mlir::IntegerType::get(&context, 32), {4096}},
               {mlir::IntegerType::get(&context, 32), {1, 32}},
               {mlir::IntegerType::get(&context, 32), {4, 2}},
               {mlir::IntegerType::get(&context, 32), {2, 2, 2, 2, 2, 2}}};
  for (const auto &test : cases) {
    MemoryCase memory(context, test.first, test.second);
    auto &stage = memory.stage;
    auto profile = calibrationProfile();
    profile.simdIndirectLoadModel = "random_dtype_matched_ab_20261008";
    auto result = indirectCosts(stage, profile);
    EXPECT_EQ(result[0].indirectLoadPricing, "legacy_transactions");
  }
}

TEST(IndirectLoadCostModelTest, SimdDtypeRankFitAcceptsPartitionedTritonLoad) {
  IRTestContext context(true, false);
  auto module = context.parse(R"mlir(
    module {
      tt.func public @probe(%x: !tt.ptr<i16>, %idx: !tt.ptr<i32>, %out: !tt.ptr<i16>) {
        %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
        %ip = tt.splat %idx : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
        %ip2 = tt.addptr %ip, %r : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
        %index = tt.load %ip2 : tensor<32x!tt.ptr<i32>>
        %xp = tt.splat %x : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>>
        %xp2 = tt.addptr %xp, %index : tensor<32x!tt.ptr<i16>>, tensor<32xi32>
        %payload = tt.load %xp2 : tensor<32x!tt.ptr<i16>>
        %yp = tt.splat %out : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>>
        %yp2 = tt.addptr %yp, %r : tensor<32x!tt.ptr<i16>>, tensor<32xi32>
        tt.store %yp2, %payload : tensor<32x!tt.ptr<i16>>
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
    if (stage.costModelKind == StageCostModelKind::IndirectGatherMemory)
      payload = &stage;
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(llvm::count_if(payload->operations,
                           [](mlir::Operation *op) {
                             return op->getName().getStringRef() == "tt.load";
                           }),
            1);
  auto profile = calibrationProfile();
  profile.simdIndirectLoadModel = "random_dtype_matched_ab_20261008";
  auto result = indirectCosts(*payload, profile);
  const auto &fit = result[0];
  EXPECT_EQ(fit.indirectLoadPricing, profile.simdIndirectLoadModel);
  EXPECT_DOUBLE_EQ(fit.resources.load, 81.94869549595556 * 32);
  // The same partitioned load must also admit SIMT dtype pricing.
  profile.simtIndirectLoadModel = "random_dtype_six_term_20261008";
  auto simtResult = indirectCosts(*payload, profile);
  const auto &simt = simtResult[1];
  EXPECT_EQ(simt.indirectLoadPricing, profile.simtIndirectLoadModel);
  EXPECT_NEAR(simt.resources.load,
              12.456944150614481 + 1.6732132663368477 * 32 + 95.13859585355112 +
                  0.11214618741568584 * 32,
              1e-10);
}

TEST(IndirectLoadCostModelTest,
     SimtDtypeRankFitUsesStorageWidthAndPreservesResources) {
  IRTestContext context(false, true);
  const llvm::SmallVector<mlir::Type> types = {
      mlir::IntegerType::get(&context, 8),
      mlir::IntegerType::get(&context, 16),
      mlir::IntegerType::get(&context, 32),
      mlir::IntegerType::get(&context, 64),
      mlir::IntegerType::get(&context, 32, mlir::IntegerType::Unsigned),
      mlir::Float16Type::get(&context),
      mlir::BFloat16Type::get(&context),
      mlir::Float32Type::get(&context),
      mlir::Float64Type::get(&context),
      mlir::Float8E4M3FNType::get(&context),
      mlir::Float8E5M2Type::get(&context)};
  // Ordering: alpha, beta, sigma, phi, gamma, nu (same as the frozen model).
  const double coefficients[][6] = {
      {7.617911783542814, 1.5780971099377332, 92.18952965944916,
       0.19387737354356369, 0, 0.049471659398567715},
      {12.456944150614481, 1.6732132663368477, 95.13859585355112,
       0.11214618741568584, 5.634318857353958, 0.026588037490546692},
      {50.71936539506111, 1.693396365504775, 60.38681262899275,
       0.09265123974451413, 82.91193957489877, 0.026814982781878355},
      {96.09572807125711, 1.937402636239715, 24.962585867214496,
       0.20099293498091206, 41.8669299352394, 0.03855485734655757}};
  for (mlir::Type element : types) {
    const int64_t bytes = element.getIntOrFloatBitWidth() / 8;
    const auto &beta = coefficients[bytes == 1   ? 0
                                    : bytes == 2 ? 1
                                    : bytes == 4 ? 2
                                                 : 3];
    for (int64_t warps : {1, 2, 4, 8, 16, 32, 64}) {
      for (int64_t q : {1, 2, 4}) {
        const int64_t elements = 32 * warps * q;
        for (int64_t rank = 1; rank <= 5; ++rank) {
          llvm::SmallVector<int64_t> shape(rank, 2);
          if (rank == 1) {
            shape[0] = elements;
          } else {
            shape.back() =
                std::min<int64_t>(32, elements / (1LL << (rank - 1)));
            shape[rank - 2] = elements / (shape.back() * (1LL << (rank - 2)));
          }
          SCOPED_TRACE(element.getIntOrFloatBitWidth());
          SCOPED_TRACE(warps);
          SCOPED_TRACE(q);
          SCOPED_TRACE(rank);
          MemoryCase test(context, element, shape);
          test.addHelper(rank >= 3);
          auto &stage = test.stage;
          stage.features.loopBackedgeCount = 1;
          auto &work = stage.workload;
          work.loadWarpInstructions = work.indirectLoadTransactions =
              elements / 32;
          work.scalarOperations = 7;
          work.storeBytes = 16;
          work.storeWarpInstructions = 2;
          work.addressPatterns = {
              loadedAddressPattern(stage, "tt.load", shape)};
          auto profile = calibrationProfile();
          profile.logicalWarpGroupCount = warps;
          auto old = indirectCosts(stage, profile);
          profile.simtIndirectLoadModel = "random_dtype_six_term_20261008";
          auto result = indirectCosts(stage, profile);
          const auto &fit = result[1];
          const auto &legacy = old[1];
          const double columns = rank == 1 ? 32 : shape.back();
          const double expected =
              beta[0] + beta[1] * elements + beta[2] * (elements <= 128) +
              beta[3] * elements * (rank == 1) +
              beta[4] * std::max(4.0 * q / columns - 2, 0.0) +
              beta[5] * elements * std::max(std::log2(32.0 / columns), 0.0);
          EXPECT_EQ(fit.indirectLoadPricing, profile.simtIndirectLoadModel);
          EXPECT_NEAR(fit.resources.load, expected, 1e-9);
          expectOnlyMemoryChanged(legacy, fit);
          EXPECT_DOUBLE_EQ(result[0].totalCycles, old[0].totalCycles);
          profile.simt.indirectDependencyLatencyCycles = 10000;
          auto noDoubleCharge = indirectCosts(stage, profile);
          EXPECT_DOUBLE_EQ(noDoubleCharge[1].totalCycles, fit.totalCycles);
        }
      }
    }
  }
}

TEST(IndirectLoadCostModelTest, SimtDtypeRankFitRejectsUnvalidatedDomains) {
  IRTestContext context(false, true);
  const llvm::SmallVector<llvm::SmallVector<int64_t>> shapes = {
      {32}, {16}, {2, 128, 2}, {2, 2, 2, 2, 2, 2}, {1, 32}, {2, 256}};
  for (size_t index = 0; index < shapes.size(); ++index) {
    MemoryCase memory(context, IntegerType::get(&context, 32), shapes[index]);
    auto &stage = memory.stage;
    auto *load = memory.op;
    stage.workload.indirectLoadTransactions =
        stage.workload.loadWarpInstructions = 1;
    auto profile = calibrationProfile();
    profile.logicalWarpGroupCount = (index == 2 || index == 5) ? 4 : 1;
    profile.simtIndirectLoadModel = "random_dtype_six_term_20261008";
    auto result = indirectCosts(stage, profile);
    EXPECT_EQ(result[1].indirectLoadPricing, index == 0
                                                 ? profile.simtIndirectLoadModel
                                                 : "legacy_transactions");
    if (index == 0) {
      for (unsigned guard = 0; guard < 8; ++guard) {
        auto invalid = stage;
        mlir::Operation *secondLoad = nullptr;
        if (guard == 0)
          invalid.workload.predicateElements = 1;
        if (guard == 1)
          invalid.workload.estimatedSpillTransactions = 1;
        if (guard == 2)
          invalid.features.activeLaneRatio = 0.5;
        if (guard == 3) {
          invalid.workload.storeBytes = invalid.workload.indirectStoreBytes = 4;
        }
        if (guard == 4)
          invalid.workload.addressPatterns[0].axes[0].regularity =
              "fixed_stride";
        if (guard == 5)
          invalid.workload.addressPatterns[0].dependsOnLoadedValue = false;
        if (guard == 6) {
          secondLoad = load->clone();
          invalid.operations.push_back(secondLoad);
        }
        if (guard == 7)
          invalid.features.hasAtomicMemory = true;
        auto rejected = indirectCosts(invalid, profile);
        EXPECT_EQ(rejected[1].indirectLoadPricing, "legacy_transactions");
        if (secondLoad)
          secondLoad->destroy();
      }
      profile.simtIndirectLoadModel = "unknown_load_fit";
      EXPECT_FALSE(profile.isValid());
    }
  }
}

TEST(IndirectLoadCostModelTest, RandomIndirectFitReplacesOnlyLoadResource) {
  IRTestContext context(false, true);
  MemoryCase test(context, IntegerType::get(&context, 32), {256});
  auto &stage = test.stage;
  auto &work = stage.workload;
  work.loadBytes = work.indirectLoadBytes = 1024;
  work.loadWarpInstructions = work.indirectLoadTransactions = 8;
  work.scalarOperations = 7;
  work.storeWarpInstructions = 2;
  work.storeBytes = 16;
  work.addressPatterns = {loadedAddressPattern(stage, "tt.load", {256})};
  auto profile = calibrationProfile();
  profile.logicalWarpGroupCount = 4;
  auto legacy = indirectCosts(stage, profile);
  profile.simtIndirectLoadModel = "random_i32_six_term_20261007";
  auto calibrated = indirectCosts(stage, profile);
  const auto &oldCosts = legacy;
  const auto &newCosts = calibrated;
  EXPECT_DOUBLE_EQ(newCosts[0].totalCycles, oldCosts[0].totalCycles);
  const double fitted =
      62.930686069640686 + 1.5502588407166296 * 256 + 0.19338028618547126 * 256;
  EXPECT_DOUBLE_EQ(newCosts[1].resources.load, fitted);
  expectOnlyMemoryChanged(oldCosts[1], newCosts[1]);
  // Changing the old dependency latency cannot affect a fitted load.
  profile.simt.indirectDependencyLatencyCycles = 10000;
  auto changed = indirectCosts(stage, profile);
  EXPECT_DOUBLE_EQ(changed[1].totalCycles, newCosts[1].totalCycles);
  // Unknown/non-affine axes and spill remain outside the calibrated domain.
  profile.simt.indirectDependencyLatencyCycles = 20;
  stage.workload.addressPatterns.front().axes.front().regularity =
      "computed_nonaffine";
  auto fallback = indirectCosts(stage, profile);
  EXPECT_DOUBLE_EQ(fallback[1].totalCycles, oldCosts[1].totalCycles);
}
