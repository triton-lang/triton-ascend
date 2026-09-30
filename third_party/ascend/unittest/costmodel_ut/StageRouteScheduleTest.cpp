#include "AscendModel/RouteModel/StageRouteCostModel.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/RouteModel/TargetSchedule.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <set>

using namespace mlir::ascend;

namespace {

StageImplementationCost implementation(StageMode mode, int64_t factor,
                                       double cycles, bool local = false) {
  StageImplementationCost cost;
  cost.implementation = {mode, factor, local};
  cost.totalCycles = cycles;
  return cost;
}

LogicalStageCost simdBody(llvm::StringRef id, double cycles) {
  LogicalStageCost stage;
  stage.id = id.str();
  stage.features.replicatedByLocalSuperBlock = true;
  stage.implementations = {implementation(StageMode::SIMD, 1, cycles)};
  return stage;
}

LogicalStageCost localScope(llvm::StringRef id, double simd, double f1,
                           double f4) {
  auto stage = simdBody(id, simd);
  stage.localSimtMaterializable = true;
  stage.localSuperblockMaterializable = true;
  stage.localSimtScopeCount = 1;
  stage.localSimtFactors = {1, 4};
  stage.implementations.push_back(implementation(StageMode::SIMT, 1, f1, true));
  stage.implementations.push_back(implementation(StageMode::SIMT, 4, f4, true));
  return stage;
}

StageCostTable mixedTable(int64_t programs, double simd = 10.0,
                         double f1 = 11.0, double f4 = 24.0) {
  StageCostTable table;
  table.logicalProgramCountHint = programs;
  table.physicalCoreCountHint = 56;
  table.mixedSchedule = {StageProgramScheduleKind::Strided, 32,
                         "test_cube_schedule", true};
  table.stages = {simdBody("load", simd),
                  localScope("recurrence", 1000.0, f1, f4)};
  return table;
}

struct EnumeratedSchedule {
  int64_t active = 0;
  int64_t totalFull = 0;
  int64_t totalTail = 0;
  int64_t totalMasked = 0;
  int64_t criticalTasks = 0;
  int64_t criticalFull = 0;
  int64_t criticalTail = 0;
  int64_t criticalMasked = 0;
  double cycles = -1.0;
  std::array<double, 3> stageCycles{};
};

// Independent, deliberately non-cohort oracle: enumerate actual logical PIDs
// assigned to every core, then execute each group/tail invocation in order.
// Only positive grids reach this helper; zero is an unknown launch hint.
EnumeratedSchedule enumerateSchedule(int64_t programs, int64_t cores,
                                     int64_t factor, bool mixed, double setup,
                                     double groupedScope) {
  EnumeratedSchedule result;
  const int64_t chunk = (programs + cores - 1) / cores;
  for (int64_t core = 0; core < cores; ++core) {
    int64_t tasks = 0;
    if (mixed) {
      for (int64_t pid = core; pid < programs; pid += cores)
        ++tasks;
    } else {
      for (int64_t pid = core * chunk;
           pid < programs && pid < (core + 1) * chunk; ++pid)
        ++tasks;
    }
    if (tasks == 0)
      continue;
    ++result.active;
    int64_t remaining = tasks, full = 0, tail = 0, masked = 0;
    std::array<double, 3> costs{setup, 0.0, 0.0};
    while (remaining >= factor) {
      ++full;
      remaining -= factor;
      costs[1] += 3.0 * factor;
      costs[2] += groupedScope;
    }
    if (mixed) {
      while (remaining > 0) {
        --remaining;
        ++tail;
        costs[1] += 3.0;
        costs[2] += 11.0;
      }
    } else if (remaining != 0) {
      ++masked;
      costs[1] += 3.0 * factor;
      costs[2] += groupedScope;
    }
    result.totalFull += full;
    result.totalTail += tail;
    result.totalMasked += masked;
    const double cycles = costs[0] + costs[1] + costs[2];
    if (cycles > result.cycles) {
      result.cycles = cycles;
      result.criticalTasks = tasks;
      result.criticalFull = full;
      result.criticalTail = tail;
      result.criticalMasked = masked;
      result.stageCycles = costs;
    }
  }
  return result;
}

void expectEnumeratedSchedule(const StageRoutePlan &plan,
                              const EnumeratedSchedule &expected,
                              int64_t cores, int64_t factor) {
  ASSERT_TRUE(plan.legal);
  ASSERT_TRUE(plan.runtimeScheduleKnown);
  ASSERT_TRUE(plan.runtimeCostsScaled);
  EXPECT_EQ(plan.routeSuperblockFactor, factor);
  EXPECT_EQ(plan.runtimeSchedule.physicalCoreCount, cores);
  EXPECT_EQ(plan.runtimeActiveCoreCount, expected.active);
  EXPECT_EQ(plan.runtimeTotalFullGroups, expected.totalFull);
  EXPECT_EQ(plan.runtimeTotalTailPrograms, expected.totalTail);
  EXPECT_EQ(plan.runtimeTotalMaskedGroups, expected.totalMasked);
  EXPECT_EQ(plan.runtimePhysicalProgramCount,
            expected.totalFull + expected.totalTail + expected.totalMasked);
  EXPECT_EQ(plan.runtimeCriticalCoreLogicalPrograms, expected.criticalTasks);
  EXPECT_EQ(plan.runtimeFullGroups, expected.criticalFull);
  EXPECT_EQ(plan.runtimeTailPrograms, expected.criticalTail);
  EXPECT_EQ(plan.runtimeMaskedGroups, expected.criticalMasked);
  EXPECT_EQ(plan.runtimeWaveCount, expected.criticalFull +
                                       expected.criticalTail +
                                       expected.criticalMasked);
  EXPECT_DOUBLE_EQ(plan.totalCycles, expected.cycles);
  ASSERT_EQ(plan.logicalStageCycles.size(), expected.stageCycles.size());
  ASSERT_EQ(plan.entryTransitionCycles.size(), expected.stageCycles.size());
  for (size_t i = 0; i < expected.stageCycles.size(); ++i) {
    EXPECT_DOUBLE_EQ(plan.logicalStageCycles[i], expected.stageCycles[i]);
    EXPECT_DOUBLE_EQ(plan.entryTransitionCycles[i], 0.0);
  }
}

} // namespace

TEST(StageRouteScheduleTest, Mixed9579GridSweepMatchesEnumeratedCores) {
  std::set<int64_t> grids{1, 27, 28, 29, 512};
  for (int64_t k = 0; k <= 4; ++k)
    for (int64_t r = 0; r < 4; ++r)
      for (int64_t delta : {-1, 0, 1, 27}) {
        const int64_t programs = 28 * (4 * k + r) + delta;
        if (programs > 0)
          grids.insert(programs);
      }
  for (int64_t factor : {1, 2, 4, 8}) {
    for (double setup : {0.0, 7.0}) {
      for (int64_t programs : grids) {
        SCOPED_TRACE(::testing::Message()
                     << "N=" << programs << " F=" << factor
                     << " setup=" << setup);
        StageCostTable table;
        table.logicalProgramCountHint = programs;
        table.allSimdSchedule = {StageProgramScheduleKind::Strided, 28,
                                 "9579_cube", true};
        table.mixedSchedule = table.allSimdSchedule;
        auto dispatch = simdBody("dispatch", setup);
        dispatch.features.replicatedByLocalSuperBlock = false;
        dispatch.perPhysicalProgramSetup = true;
        auto scope = simdBody("scope", 1000.0);
        scope.localSimtMaterializable = true;
        scope.localSuperblockMaterializable = true;
        scope.localSimtScopeCount = 1;
        scope.localSimtFactors = {1};
        scope.implementations.push_back(
            implementation(StageMode::SIMT, 1, 11.0, true));
        if (factor != 1) {
          scope.localSimtFactors.push_back(factor);
          scope.implementations.push_back(
              implementation(StageMode::SIMT, factor, factor + 5.0, true));
        }
        table.stages = {dispatch, simdBody("body", 3.0), scope};
        auto routes = solveStageRoutes(table, {});
        if (!routes)
          FAIL() << llvm::toString(routes.takeError());
        auto expected = enumerateSchedule(programs, 28, 1, true, setup, 11.0);
        int64_t expectedFactor = 1;
        if (factor != 1) {
          auto grouped = enumerateSchedule(programs, 28, factor, true, setup,
                                           factor + 5.0);
          // F1 is also a real candidate because it must exist for tails.
          // On exact ties the solver preserves the earlier, smaller factor.
          if (grouped.cycles < expected.cycles) {
            expected = grouped;
            expectedFactor = factor;
          }
        }
        expectEnumeratedSchedule(routes->mixed, expected, 28, expectedFactor);
      }
    }
  }
}

TEST(StageRouteScheduleTest, PureSimt9579ChunkSweepMatchesEnumeratedCores) {
  for (int64_t factor : {1, 2, 4, 8}) {
    std::set<int64_t> grids{1, 55, 56, 57, 512};
    for (int64_t k = 0; k <= 4; ++k)
      for (int64_t r : {-1, 0, 1})
        for (int64_t delta : {-1, 0, 1, 55}) {
          const int64_t programs = 56 * (k * factor + r) + delta;
          if (programs > 0)
            grids.insert(programs);
        }
    for (double setup : {0.0, 7.0}) {
      for (int64_t programs : grids) {
        SCOPED_TRACE(::testing::Message()
                     << "N=" << programs << " F=" << factor
                     << " setup=" << setup);
        StageCostTable table;
        table.logicalProgramCountHint = programs;
        table.allSimtSchedule = {StageProgramScheduleKind::ContiguousChunks,
                                 56, "9579_vector", true};
        const double scopeCycles = factor == 1 ? 11.0 : factor + 5.0;
        for (double cycles : {setup, 3.0 * factor, scopeCycles}) {
          LogicalStageCost stage;
          stage.id = "stage_" + std::to_string(table.stages.size());
          stage.implementations = {
              implementation(StageMode::SIMT, factor, cycles)};
          stage.perPhysicalProgramSetup = table.stages.empty();
          table.stages.push_back(stage);
        }
        auto routes = solveStageRoutes(table, {});
        if (!routes)
          FAIL() << llvm::toString(routes.takeError());
        const auto expected = enumerateSchedule(programs, 56, factor, false,
                                                 setup, scopeCycles);
        expectEnumeratedSchedule(routes->allSimt, expected, 56, factor);
      }
    }
  }
}

TEST(StageRouteScheduleTest, NPUIRGroupsBeforeStridedCoreAssignment) {
  struct Case {
    int64_t programs;
    int64_t activeCores;
    int64_t totalFull;
    int64_t totalMasked;
    int64_t criticalPrograms;
    int64_t criticalFull;
    int64_t criticalMasked;
    int64_t waves;
  };
  for (const Case &test : std::array<Case, 3>{
           Case{140, 35, 35, 0, 4, 1, 0, 1},
           Case{142, 36, 35, 1, 2, 0, 1, 1},
           Case{400, 64, 100, 0, 8, 2, 0, 2}}) {
    SCOPED_TRACE(::testing::Message() << "N=" << test.programs);
    StageCostTable table;
    table.logicalProgramCountHint = test.programs;
    table.allSimtSchedule = {StageProgramScheduleKind::GlobalGroupsStrided,
                             64, "npuir_v1", true};
    LogicalStageCost stage;
    stage.id = "payload";
    stage.implementations = {implementation(StageMode::SIMT, 4, 10.0)};
    table.stages = {stage};

    auto routes = solveStageRoutes(table, {});
    if (!routes)
      FAIL() << llvm::toString(routes.takeError());
    const StageRoutePlan &plan = routes->allSimt;
    ASSERT_TRUE(plan.legal);
    EXPECT_EQ(plan.routeSuperblockFactor, 4);
    EXPECT_EQ(plan.runtimeActiveCoreCount, test.activeCores);
    EXPECT_EQ(plan.runtimePhysicalProgramCount,
              test.totalFull + test.totalMasked);
    EXPECT_EQ(plan.runtimeTotalFullGroups, test.totalFull);
    EXPECT_EQ(plan.runtimeTotalMaskedGroups, test.totalMasked);
    EXPECT_EQ(plan.runtimeTotalTailPrograms, 0);
    EXPECT_EQ(plan.runtimeCriticalCoreLogicalPrograms,
              test.criticalPrograms);
    EXPECT_EQ(plan.runtimeFullGroups, test.criticalFull);
    EXPECT_EQ(plan.runtimeMaskedGroups, test.criticalMasked);
    EXPECT_EQ(plan.runtimeWaveCount, test.waves);
    EXPECT_DOUBLE_EQ(plan.totalCycles, 10.0 * test.waves);
  }
}

TEST(StageRouteScheduleTest, ZeroProgramHintRemainsUnknownNotEmptyExecution) {
  auto table = mixedTable(0);
  table.mixedSchedule = {StageProgramScheduleKind::Strided, 28,
                         "9579_cube", true};
  table.allSimtSchedule = {StageProgramScheduleKind::ContiguousChunks, 56,
                           "9579_vector", true};
  for (auto &stage : table.stages)
    stage.implementations.push_back(implementation(StageMode::SIMT, 4, 24.0));
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  for (const auto *plan :
       {&routes->allSimd, &routes->allSimt, &routes->mixed}) {
    ASSERT_TRUE(plan->legal);
    EXPECT_FALSE(plan->runtimeScheduleKnown);
    EXPECT_FALSE(plan->runtimeCostsScaled);
    EXPECT_EQ(plan->runtimeActiveCoreCount, 0);
    EXPECT_EQ(plan->runtimeCriticalCoreLogicalPrograms, 0);
    EXPECT_EQ(plan->runtimeTotalFullGroups, 0);
    EXPECT_EQ(plan->runtimeTotalTailPrograms, 0);
    EXPECT_EQ(plan->runtimeTotalMaskedGroups, 0);
    EXPECT_EQ(plan->runtimePhysicalProgramCount, 0);
    EXPECT_EQ(plan->runtimeWaveCount, 1);
    EXPECT_GT(plan->totalCycles, 0.0);
  }
}

TEST(StageRouteScheduleTest, MixedUsesPerCoreGroupsAndPaysDispatchOnce) {
  auto table = mixedTable(512);
  auto dispatch = simdBody("dispatch", 7.0);
  dispatch.model = "auto_blockify_dispatch";
  dispatch.features.replicatedByLocalSuperBlock = false;
  dispatch.perPhysicalProgramSetup = true;
  table.stages.insert(table.stages.begin(), dispatch);

  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &plan = routes->mixed;
  ASSERT_TRUE(plan.legal);
  ASSERT_TRUE(plan.runtimeScheduleKnown);
  EXPECT_EQ(plan.routeSuperblockFactor, 4);
  EXPECT_EQ(plan.runtimeSchedule.physicalCoreCount, 32);
  EXPECT_EQ(plan.runtimeActiveCoreCount, 32);
  EXPECT_EQ(plan.runtimeCriticalCoreLogicalPrograms, 16);
  EXPECT_EQ(plan.runtimeWaveCount, 4);
  EXPECT_EQ(plan.runtimeFullGroups, 4);
  EXPECT_EQ(plan.runtimeTailPrograms, 0);
  EXPECT_EQ(plan.runtimeTotalFullGroups, 128);
  EXPECT_EQ(plan.runtimeTotalTailPrograms, 0);
  // Dispatch is outside the loop: 7 + 4 * (4 * 10 + 24).
  EXPECT_DOUBLE_EQ(plan.totalCycles, 263.0);
  ASSERT_EQ(plan.logicalStageCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(plan.logicalStageCycles[0], 7.0);
  EXPECT_DOUBLE_EQ(plan.logicalStageCycles[1], 160.0);
  EXPECT_DOUBLE_EQ(plan.logicalStageCycles[2], 96.0);
}

TEST(StageRouteScheduleTest, MixedPricesF1TailOnEachStridedCore) {
  auto routes = solveStageRoutes(mixedTable(140), {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &plan = routes->mixed;
  ASSERT_TRUE(plan.legal);
  EXPECT_EQ(plan.routeSuperblockFactor, 4);
  EXPECT_EQ(plan.runtimeActiveCoreCount, 32);
  EXPECT_EQ(plan.runtimeCriticalCoreLogicalPrograms, 5);
  EXPECT_EQ(plan.runtimeFullGroups, 1);
  EXPECT_EQ(plan.runtimeTailPrograms, 1);
  EXPECT_EQ(plan.runtimeWaveCount, 2);
  EXPECT_EQ(plan.runtimeTotalFullGroups, 32);
  EXPECT_EQ(plan.runtimeTotalTailPrograms, 12);
  EXPECT_EQ(plan.runtimeTotalMaskedGroups, 0);
  // Twelve cores run one F4 group and one F1 tail; twenty run only F4.
  EXPECT_DOUBLE_EQ(plan.totalCycles, (4 * 10 + 24) + (10 + 11));
}

TEST(StageRouteScheduleTest, ExpensiveTailCanMakeShorterCoreCritical) {
  auto routes = solveStageRoutes(mixedTable(125, 0.0), {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &plan = routes->mixed;
  ASSERT_TRUE(plan.legal);
  EXPECT_EQ(plan.routeSuperblockFactor, 4);
  EXPECT_EQ(plan.runtimeActiveCoreCount, 32);
  EXPECT_EQ(plan.runtimeTotalFullGroups, 29);
  EXPECT_EQ(plan.runtimeTotalTailPrograms, 9);
  // The three cores with three F1 tails cost 33, versus 24 for four tasks.
  EXPECT_EQ(plan.runtimeCriticalCoreLogicalPrograms, 3);
  EXPECT_EQ(plan.runtimeFullGroups, 0);
  EXPECT_EQ(plan.runtimeTailPrograms, 3);
  EXPECT_EQ(plan.runtimeWaveCount, 3);
  EXPECT_DOUBLE_EQ(plan.totalCycles, 33.0);
}

TEST(StageRouteScheduleTest, PureSimtMasksEachContiguousChunkSeparately) {
  StageCostTable table;
  table.logicalProgramCountHint = 130;
  table.allSimtSchedule = {StageProgramScheduleKind::ContiguousChunks, 64,
                           "test_vector_schedule", true};
  LogicalStageCost stage;
  stage.id = "payload";
  stage.implementations = {implementation(StageMode::SIMT, 1, 11.0),
                           implementation(StageMode::SIMT, 4, 24.0)};
  table.stages = {stage};
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &plan = routes->allSimt;
  ASSERT_TRUE(plan.legal);
  EXPECT_EQ(plan.routeSuperblockFactor, 4);
  // Chunk size is ceil(130/64)=3: 43 chunks of three and one of one.
  EXPECT_EQ(plan.runtimeActiveCoreCount, 44);
  EXPECT_EQ(plan.runtimeTotalFullGroups, 0);
  EXPECT_EQ(plan.runtimeTotalMaskedGroups, 44);
  EXPECT_EQ(plan.runtimeTotalTailPrograms, 0);
  EXPECT_EQ(plan.runtimePhysicalProgramCount, 44);
  EXPECT_EQ(plan.runtimeMaskedGroups, 1);
  EXPECT_EQ(plan.runtimeTailPrograms, 0);
  EXPECT_EQ(plan.runtimeWaveCount, 1);
  EXPECT_DOUBLE_EQ(plan.totalCycles, 24.0);
}

TEST(StageRouteScheduleTest, MixedTailRequiresMatchingF1Implementation) {
  auto table = mixedTable(140);
  auto &scope = table.stages.back();
  scope.localSimtFactors = {4};
  scope.implementations.erase(scope.implementations.begin() + 1);
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_FALSE(routes->mixed.legal);

  // The same F4-only scope is valid when every core has complete groups.
  table.logicalProgramCountHint = 128;
  routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_EQ(routes->mixed.runtimeTotalTailPrograms, 0);
}

TEST(StageRouteScheduleTest, UnknownMixedScheduleDoesNotUseVectorHint) {
  auto table = mixedTable(512, 10.0, 100.0);
  table.mixedSchedule = {};
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_FALSE(routes->mixed.runtimeScheduleKnown);
  EXPECT_EQ(routes->mixed.runtimeSchedule.physicalCoreCount, 0);
  EXPECT_EQ(routes->mixed.runtimeActiveCoreCount, 0);
  EXPECT_EQ(routes->mixed.runtimeWaveCount, 1);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 64.0);
}

TEST(StageRouteScheduleTest, UnknownMixedCoreCountDoesNotUseVectorHint) {
  auto table = mixedTable(512, 10.0, 100.0);
  table.mixedSchedule.physicalCoreCount = 0;
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_FALSE(routes->mixed.runtimeScheduleKnown);
  EXPECT_EQ(routes->mixed.runtimeSchedule.physicalCoreCount, 0);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 64.0);
}

TEST(StageRouteScheduleTest, UnknownCandidateKeepsAllCostsOnTheSameBasis) {
  auto table = mixedTable(512, 10.0, 100.0);
  table.mixedSchedule = {};
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_TRUE(routes->allSimd.runtimeScheduleKnown);
  EXPECT_FALSE(routes->mixed.runtimeScheduleKnown);
  EXPECT_FALSE(routes->allSimd.runtimeCostsScaled);
  EXPECT_FALSE(routes->mixed.runtimeCostsScaled);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 1010.0);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 64.0);
}

TEST(StageRouteScheduleTest, CandidateModesUseTheirOwnSchedulingUnits) {
  auto table = mixedTable(512);
  table.allSimdSchedule = {StageProgramScheduleKind::Strided, 32, "cube", true};
  table.allSimtSchedule = {StageProgramScheduleKind::ContiguousChunks, 64,
                          "vector", true};
  for (auto &stage : table.stages) {
    stage.implementations.push_back(implementation(StageMode::SIMT, 1, 11.0));
    stage.implementations.push_back(implementation(StageMode::SIMT, 4, 24.0));
  }
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.runtimeCostsScaled);
  ASSERT_TRUE(routes->allSimt.runtimeCostsScaled);
  ASSERT_TRUE(routes->mixed.runtimeCostsScaled);
  EXPECT_EQ(routes->allSimd.runtimeWaveCount, 16);
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 4);
  EXPECT_EQ(routes->allSimt.runtimeWaveCount, 2);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_EQ(routes->mixed.runtimeWaveCount, 4);
}

TEST(StageRouteScheduleTest, ModelResolvesTargetAndTaSchedulingSeparately) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @dot_kernel(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) {
        %zero = arith.constant dense<0.0> : tensor<16x16xf32>
        %result = "tt.dot"(%a, %b, %zero) :
          (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
        return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);
  SimdSimtCostModelOptions options;
  options.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
  options.actualTarget = "Ascend950PR_9589";
  options.compileOn91095 = true;
  options.numWarps = 4;
  options.logicalProgramCountHint = 130;
  options.physicalVectorCoreCountHint = 56;
  options.wholeKernelSuperblockMaterializable = true;
  options.simdAutoBlockifyV1 = true;
  auto report = analyzeSimdSimtCandidates(*module, options);
  if (!report)
    FAIL() << llvm::toString(report.takeError());
  EXPECT_EQ(report->stageModel.allSimd.runtimeSchedule.physicalCoreCount, 32);
  EXPECT_EQ(report->stageModel.allSimt.runtimeSchedule.physicalCoreCount, 64);
  EXPECT_EQ(report->stageModel.allSimt.runtimeSchedule.kind,
            StageProgramScheduleKind::GlobalGroupsStrided);

  options.enableTaAutoBlockifyV1 = true;
  options.taPhysicalVectorCoreCount = 56;
  report = analyzeSimdSimtCandidates(*module, options);
  if (!report)
    FAIL() << llvm::toString(report.takeError());
  EXPECT_EQ(report->stageModel.allSimt.runtimeSchedule.physicalCoreCount, 56);
  EXPECT_EQ(report->stageModel.allSimt.runtimeSchedule.kind,
            StageProgramScheduleKind::ContiguousChunks);

  options.wholeKernelSuperblockMaterializable = false;
  options.simdAutoBlockifyV1 = false;
  report = analyzeSimdSimtCandidates(*module, options);
  if (!report)
    FAIL() << llvm::toString(report.takeError());
  EXPECT_EQ(report->stageModel.allSimt.runtimeSchedule.kind,
            StageProgramScheduleKind::FlatGrid);
  EXPECT_EQ(report->stageModel.allSimt.runtimeSchedule.physicalCoreCount, 64);
  EXPECT_EQ(report->stageModel.allSimt.runtimeActiveCoreCount, 64);
  EXPECT_EQ(report->stageModel.allSimt.runtimeWaveCount, 3);
}

TEST(StageRouteScheduleTest, ScopeSelectionIncludesTailCost) {
  auto table = mixedTable(140);
  table.stages = {localScope("A", 10.0, 100.0, 1.0),
                  localScope("B", 10.0, 1.0, 2.0)};
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &plan = routes->mixed;
  ASSERT_TRUE(plan.legal);
  EXPECT_EQ(plan.routeSuperblockFactor, 4);
  ASSERT_EQ(plan.implementations.size(), 2u);
  EXPECT_EQ(plan.implementations[0].mode, StageMode::SIMD);
  EXPECT_EQ(plan.implementations[1].mode, StageMode::SIMT);
  // A has a cheaper F4 group but its F1 tail makes it cost 151.
  // Selecting B costs (4*10+2)+(10+1)=53, beating F1's 5*(10+1)=55.
  EXPECT_DOUBLE_EQ(plan.totalCycles, 53.0);
}

TEST(StageRouteScheduleTest, MaximumProgramCountCeilDivisionDoesNotOverflow) {
  StageCostTable table;
  table.logicalProgramCountHint = std::numeric_limits<int64_t>::max();
  table.allSimtSchedule = {StageProgramScheduleKind::ContiguousChunks, 64,
                           "test_vector_schedule", true};
  LogicalStageCost stage;
  stage.id = "payload";
  stage.implementations = {implementation(StageMode::SIMT, 4, 1.0)};
  table.stages = {stage};
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &plan = routes->allSimt;
  ASSERT_TRUE(plan.legal);
  EXPECT_EQ(plan.runtimeActiveCoreCount, 64);
  EXPECT_EQ(plan.runtimeTotalFullGroups,
            std::numeric_limits<int64_t>::max() / 4);
  EXPECT_EQ(plan.runtimeTotalMaskedGroups, 1);
  EXPECT_EQ(plan.runtimePhysicalProgramCount,
            std::numeric_limits<int64_t>::max() / 4 + 1);
  EXPECT_EQ(plan.runtimeWaveCount, int64_t{1} << 55);
  EXPECT_DOUBLE_EQ(plan.totalCycles, static_cast<double>(int64_t{1} << 55));
}

TEST(StageRouteScheduleTest, MaximumProgramCountOnOneCoreDoesNotOverflow) {
  StageCostTable table;
  table.logicalProgramCountHint = std::numeric_limits<int64_t>::max();
  table.allSimdSchedule = {StageProgramScheduleKind::FlatGrid, 1,
                           "test_single_core", false};
  table.stages = {simdBody("payload", 1.0)};
  auto routes = solveStageRoutes(table, {});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &plan = routes->allSimd;
  ASSERT_TRUE(plan.legal);
  EXPECT_EQ(plan.runtimeActiveCoreCount, 1);
  EXPECT_EQ(plan.runtimeCriticalCoreLogicalPrograms,
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ(plan.runtimeWaveCount, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(plan.runtimeTotalFullGroups, std::numeric_limits<int64_t>::max());
  EXPECT_DOUBLE_EQ(plan.totalCycles,
                   static_cast<double>(std::numeric_limits<int64_t>::max()));
}

TEST(StageRouteScheduleTest, ResolvesSchedulingTargetCounts) {
  auto counts = resolveTargetCoreCounts("Ascend950PR_9589");
  if (!counts)
    FAIL() << llvm::toString(counts.takeError());
  EXPECT_EQ(counts->cube, 32);
  EXPECT_EQ(counts->vector, 64);
}

TEST(StageRouteScheduleTest, Resolves9579SchedulingTargetCounts) {
  auto counts = resolveTargetCoreCounts("Ascend950PR_9579");
  if (!counts)
    FAIL() << llvm::toString(counts.takeError());
  EXPECT_EQ(counts->cube, 28);
  EXPECT_EQ(counts->vector, 56);
}

TEST(StageRouteScheduleTest, CustomCountsOnlyReduceTargetCounts) {
  auto reduced = resolveTargetCoreCounts("Ascend950PR_9589", 16, 24);
  if (!reduced)
    FAIL() << llvm::toString(reduced.takeError());
  EXPECT_EQ(reduced->cube, 16);
  EXPECT_EQ(reduced->vector, 24);
  auto capped = resolveTargetCoreCounts("Ascend950PR_9589", 64, 32);
  if (!capped)
    FAIL() << llvm::toString(capped.takeError());
  EXPECT_EQ(capped->cube, 32);
  EXPECT_EQ(capped->vector, 32);
}

TEST(StageRouteScheduleTest, RejectsUnknownTargetAndIncompleteCustomCounts) {
  auto unknown = resolveTargetCoreCounts("unrecognized_target");
  ASSERT_FALSE(unknown);
  EXPECT_NE(llvm::toString(unknown.takeError()).find("unknown scheduling target"),
            std::string::npos);
  auto incomplete =
      resolveTargetCoreCounts("Ascend950PR_9589", 16, 0);
  ASSERT_FALSE(incomplete);
  EXPECT_NE(llvm::toString(incomplete.takeError()).find("positive int pair"),
            std::string::npos);
}
