#include "AscendModel/Analysis/StagePartitioner.h"
#include "IndirectMemoryTestUtils.h"
#include <gtest/gtest.h>

TEST(IndirectGatherMemoryCostModelTest, GeometryMaskAndGuards) {
  for (bool store : {false, true})
    for (int rank : {1, 2, 3, 4, 5})
      for (auto type : {"i8", "i16", "f32", "i64"})
        for (int mask : {0, 1, 2})
          for (int f : {1, 2, 4})
            for (int q : {1, 2, 4}) {
              int w = rank % 2 ? 1 : 4;
              SCOPED_TRACE(llvm::formatv("{0}/{1}/{2}/{3}/{4}/{5}", store, rank,
                                         type, mask, f, w)
                               .str());
              Memory t(store, rank, type, mask, 32 * w * q, w);
              ASSERT_TRUE(t.module);
              auto fit = t.cost(StageMode::SIMT, f);
              ASSERT_TRUE(t.value(fit));
              EXPECT_GT(*t.value(fit), 0);
              EXPECT_STREQ(t.pricing(fit),
                           mask == 0   ? "unmasked_complete_factor_20261020"
                           : mask == 1 ? "prefix_mask_rank_factor_20261017"
                                       : "random_mask_rank_factor_20261017");
              EXPECT_EQ(t.value(t.cost(StageMode::SIMT, f, true)),
                        t.value(fit));
              EXPECT_FALSE(t.value(t.cost(StageMode::SIMT, f, true, 2)));
              t.profile.simtIndirectLoadModel =
                  t.profile.simtIndirectStoreModel =
                      "random_mask_rank_factor_20261017";
              EXPECT_EQ(t.value(t.cost(StageMode::SIMT, f)), t.value(fit));
              t.stage.workload.privateStackLoadInstructions = 1;
              EXPECT_FALSE(t.value(t.cost(StageMode::SIMT, f)));
              t.stage.workload.privateStackLoadInstructions = 0;
              for (auto axis : {"structured", "unknown", ""}) {
                t.stage.workload.addressPatterns[0].axes[0].offsetAxisInfo =
                    axis;
                EXPECT_FALSE(t.value(t.cost(StageMode::SIMT, f)));
                EXPECT_FALSE(t.value(t.cost(StageMode::SIMD)));
              }
            }
}

TEST(IndirectGatherMemoryCostModelTest,
     SimdEnvelopeOwnershipAndWarpIndependence) {
  for (bool store : {false, true})
    for (int rank : {1, 3, 5})
      for (int mask : {0, 1, 2}) {
        Memory t(store, rank, "f32", mask);
        ASSERT_TRUE(t.module);
        auto fit = t.cost(StageMode::SIMD);
        ASSERT_TRUE(t.value(fit));
        EXPECT_EQ(fit.nativeSimdEnvelope, mask != 0);
        EXPECT_EQ(bool(fit.coveredPredicate),
                  mask && (store || (mask == 1 && rank == 1)));
        EXPECT_EQ(bool(fit.densityRange), mask == 2);
        for (int w : {2, 4, 8, 16, 32, 64}) {
          t.profile.baseSimtWarpCount = w;
          EXPECT_EQ(t.value(t.cost(StageMode::SIMD)), t.value(fit));
        }
        t.stage.operations.erase(t.stage.operations.begin());
        EXPECT_FALSE(t.cost(StageMode::SIMD).coveredPredicate);
        EXPECT_EQ(t.value(t.cost(StageMode::SIMD)), t.value(fit));
      }
}

TEST(IndirectGatherMemoryCostModelTest, NumericRegressionsAndExtrapolation) {
  const double cycles[2][3] = {
      {299.60660582547052, 424.44774782363083, 663.93891572228188},
      {378.26707268408427, 497.90360695645398, 842.34628477005697}};
  for (bool store : {false, true}) {
    Memory t(store, 1, "f32", 0, 128, 4);
    for (int f : {1, 2, 4}) {
      auto value = t.value(t.cost(StageMode::SIMT, f));
      ASSERT_TRUE(value);
      EXPECT_NEAR(*value * f, cycles[store][f == 4 ? 2 : f - 1], 1e-9);
    }
    Memory tail(store, 1, "i8", store ? 1 : 2, 512);
    auto value = tail.value(tail.cost(StageMode::SIMD));
    ASSERT_TRUE(value);
    EXPECT_NEAR(*value, store ? 43654.060736814834 : 46652.20020282864, 1e-6);
    Memory small(store, 1, "f32", 0, 8, 4, "%i", 64);
    ASSERT_TRUE(small.module);
    EXPECT_TRUE(small.value(small.cost(StageMode::SIMT)));
    auto type =
        cast<RankedTensorType>(store ? small.target->getOperand(1).getType()
                                     : small.target->getResult(0).getType());
    EXPECT_NEAR(
        IndirectGatherMemoryCostModel::nativeChainCycles(
            *small.profile.indirectMemoryParameters, store, type, 64, 4, 1),
        store ? 1162.0972208286435 : 1214.3855319589786, 1e-8);
    for (int rank : {1, 2, 6})
      for (int mask : {0, 1, 2})
        for (int w : {1, 2, 4, 8, 16, 32, 64})
          for (int f : {1, 2, 4}) {
            Memory edge(store, rank, "bf16", mask, 32, w, "%i", 64);
            ASSERT_TRUE(edge.module);
            EXPECT_EQ(bool(edge.value(edge.cost(StageMode::SIMT, f))),
                      w * f <= 64);
            EXPECT_TRUE(edge.value(edge.cost(StageMode::SIMD)));
          }
  }
}

TEST(IndirectGatherMemoryCostModelTest, PointerClassificationAndReshape) {
  for (bool store : {false, true})
    for (int rank : {1, 2})
      for (auto offset : {"%r", "%i", "%square"}) {
        if (rank == 2 && StringRef(offset) != "%i")
          continue;
        Memory t(store, rank, "i8", 0, 512, 1, offset);
        ASSERT_TRUE(t.module);
        auto partition = StagePartitioner().partition(*t.module, {}, {});
        ASSERT_TRUE(bool(partition));
        auto stage =
            llvm::find_if(partition->stages, [&](const LogicalStage &s) {
              return llvm::is_contained(s.operations, t.target);
            });
        ASSERT_NE(stage, partition->stages.end());
        EXPECT_EQ(stage->costModelKind ==
                      StageCostModelKind::IndirectGatherMemory,
                  StringRef(offset) != "%r");
        if (store && rank == 2 && StringRef(offset) == "%i") {
          t.stage = *stage;
          t.profile.baseSimtWarpCount = 8;
          auto value = t.value(t.cost(StageMode::SIMD));
          ASSERT_TRUE(value);
          EXPECT_NEAR(*value, 125.41805018354371 + 152.8758408085307 * 512,
                      1e-9);
          value = t.value(t.cost(StageMode::SIMT, 1, true));
          ASSERT_TRUE(value);
          EXPECT_NEAR(*value, 2520.614863361812, 1e-9);
        }
      }
}

TEST(IndirectGatherMemoryCostModelTest, NativeChainOwnsFixedBudgetOnce) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"(
    module { tt.func @probe(%x: !tt.ptr<f32>, %i: !tt.ptr<i64>, %o: !tt.ptr<f32>) {
      %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
      %ip = tt.splat %i : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
      %ia = tt.addptr %ip, %r : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
      %idx = tt.load %ia : tensor<8x!tt.ptr<i64>>
      %xp = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %xa = tt.addptr %xp, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
      %v = tt.load %xa : tensor<8x!tt.ptr<f32>>
      %op = tt.splat %o : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %oa = tt.addptr %op, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      tt.store %oa, %v : tensor<8x!tt.ptr<f32>>
      tt.return
    } })",
                                            &context);
  ASSERT_TRUE(module);
  SimdSimtCostModelOptions options;
  options.numWarps = 4;
  options.compileOn91095 = options.wholeKernelSuperblockMaterializable = true;
  options.physicalVectorCoreCountHint = 1;
  options.backendForcedDecision = SimdSimtCandidateKind::AllSIMTOnly;
  for (int f : {1, 2, 4}) {
    options.wholeKernelSuperblockFactors = {f};
    options.logicalProgramCountHint = options.sourceLogicalProgramCountHint = f;
    auto result = analyzeSimdSimtCandidates(*module, options);
    ASSERT_TRUE(bool(result));
    EXPECT_EQ(result->selectedRouteElapsed.calibrationName,
              "native_indirect_chain_20261021");
    EXPECT_DOUBLE_EQ(result->selectedRouteElapsed.fixedSystemCycles, 0);
    EXPECT_NEAR(result->candidateElapsedCosts.allSimtOnly,
                result->selectedRouteElapsed.predictedSystemCounterCycles,
                1e-9);
  }
  options.wholeKernelSuperblockFactors = {1};
  options.logicalProgramCountHint = options.sourceLogicalProgramCountHint = 4;
  options.physicalVectorCoreCountHint = 4;
  auto result = analyzeSimdSimtCandidates(*module, options);
  ASSERT_TRUE(bool(result));
  EXPECT_NE(result->selectedRouteElapsed.calibrationName,
            "native_indirect_chain_20261021");
}
