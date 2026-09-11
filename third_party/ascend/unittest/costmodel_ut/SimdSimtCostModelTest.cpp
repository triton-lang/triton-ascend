#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/StageCostModels.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using mlir::ascend::HardwareProfile;
using mlir::ascend::LogicalStage;
using mlir::ascend::LogicalStageCost;
using mlir::ascend::SimdSimtFeatureSummary;
using mlir::ascend::solveStageRoutes;
using mlir::ascend::StageCostEvaluator;
using mlir::ascend::StageCostModelKind;
using mlir::ascend::StageCostTable;
using mlir::ascend::StageFeatureAnalysis;
using mlir::ascend::StageImplementationCost;
using mlir::ascend::StageMode;
using mlir::ascend::StageModeLegalityAnalysis;
using mlir::ascend::StagePartition;
using mlir::ascend::StagePartitioner;
using mlir::ascend::StagePartitionerOptions;
using mlir::ascend::StageScheduleKind;
using mlir::ascend::StageTransitionCost;
using mlir::ascend::StageWorkload;
using mlir::ascend::StageWorkloadAnalysis;
using mlir::ascend::TriangularSolveFacts;

namespace {

SimdSimtFeatureSummary triangularBt16StageFeatures() {
  SimdSimtFeatureSummary f;
  f.simtAnchors.count = 1;
  TriangularSolveFacts triangular;
  triangular.blockRows = 16;
  triangular.blockColumns = 16;
  triangular.accumulatorType = "f32";
  triangular.recurrenceStartRow = 2;
  triangular.recurrenceLoopCount = 14;
  f.simtAnchors.triangularSolves.push_back(triangular);
  return f;
}

} // namespace

namespace {

HardwareProfile hardwareProfile(StageTransitionCost transition = {}) {
  HardwareProfile profile;
  profile.profileVersion = "unit-test-profile-v1";
  profile.target = "Ascend950PR_9579";
  profile.superblockUsefulFactorLimit = 4;
  profile.superblockPersistentStatePressureFreeFactor = 2;
  profile.superblockPersistentStateBytesPerCycle = 8.0;
  auto fill = [](auto &mode) {
    mode.setupCycles = 10.0;
    mode.vectorWidth = 64;
    mode.issueWidth = 64;
    mode.operationRates["f32.add"] = {1.0, 1.0};
    mode.operationRates["f32.mul"] = {1.0, 1.0};
    mode.operationRates["f32.max"] = {1.0, 1.0};
    mode.operationRates["convert.cast"] = {1.0, 1.0};
    mode.loadBytesPerCycle = 32.0;
    mode.storeBytesPerCycle = 16.0;
    mode.loadWarpInstructionsPerCycle = 1.0;
    mode.storeWarpInstructionsPerCycle = 1.0;
    mode.predicateOperationsPerCycle = 1.0;
    mode.shuffleLanesPerCycle = 32.0;
    mode.dotSetupCycles = 8.0;
    mode.dotFlopsPerCycle = 64.0;
    mode.scalarOperationsPerCycle = 1.0;
    mode.issueOperationsPerCycle = 4.0;
    mode.spillTransactionsPerCycle = 1.0;
    mode.indirectLoadTransactionsPerCycle = 0.5;
    mode.indirectStoreTransactionsPerCycle = 0.5;
    mode.indirectDependencyLatencyCycles = 20.0;
    mode.atomicRates["default"] = {8.0, 0.0, 0.0, 1.0};
    mode.controlFlow = {2.0, 3.0, 10.0, 7.0};
  };
  fill(profile.simd);
  fill(profile.simt);
  profile.simt.vectorWidth = 1;
  profile.simt.issueWidth = 32;
  profile.transition = std::move(transition);
  return profile;
}

LogicalStage
logicalStage(llvm::StringRef id, StageCostModelKind kind,
             StageScheduleKind schedule = StageScheduleKind::StraightLine,
             int64_t iterations = 1) {
  LogicalStage stage;
  stage.id = id.str();
  stage.costModelKind = kind;
  stage.scheduleKind = schedule;
  stage.iterationCount = iterations;
  stage.simdLegal = true;
  stage.simtLegal = true;
  stage.legalSimtFactors = {1};
  stage.workload.paysKernelSetup = true;
  stage.workload.operationElements["f32.add"] = 64.0;
  stage.workload.issueElements = 4.0;
  return stage;
}

llvm::Expected<StageCostTable>
evaluateOneStage(LogicalStage stage,
                 HardwareProfile profile = hardwareProfile()) {
  StagePartition partition;
  partition.stages.push_back(std::move(stage));
  return StageCostEvaluator().evaluate(partition, profile);
}

} // namespace

TEST(SimdSimtCostModelTest, StageHasOnlySimdOrSimtImplementations) {
  LogicalStage stage = logicalStage("scalar", StageCostModelKind::ScalarIssue);
  auto table = evaluateOneStage(std::move(stage));
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  EXPECT_EQ(table->stages.front().implementations[0].implementation.mode,
            StageMode::SIMD);
  EXPECT_EQ(table->stages.front().implementations[1].implementation.mode,
            StageMode::SIMT);
}

TEST(SimdSimtCostModelTest,
     ScopeSuperBlockLegalityRequiresBackendAndResourceMaximum) {
  auto makePartition = [](int64_t independentGroups) {
    StagePartition partition;
    LogicalStage stage =
        logicalStage("payload", StageCostModelKind::LoopCarriedRecurrence,
                     StageScheduleKind::LoopCarriedSerial, /*iterations=*/16);
    stage.features.hasLoop = true;
    stage.features.hasLoopCarriedDataDependency = true;
    stage.features.parallelRecurrenceGroupCount = independentGroups;
    stage.localSimtMaterializable = true;
    stage.localSuperblockMaterializable = true;
    stage.localSimtFactors = {1};
    partition.stages.push_back(std::move(stage));
    return partition;
  };

  StagePartition f1Only = makePartition(/*independentGroups=*/4);
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(f1Only, 4, false))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(f1Only.stages[0].localSimtFactors, (std::vector<int64_t>{1}));

  StagePartition scopeSuperblock = makePartition(/*independentGroups=*/4);
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(scopeSuperblock, 4, true))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(scopeSuperblock.stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));

  // ABI-v2 creates an F1 V1 scheduling loop and refines only the selected
  // scope after bufferization, but local and whole-kernel factors still share
  // the same target/runtime warp-resource maximum.
  StagePartition mixedOnly = makePartition(/*independentGroups=*/4);
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(mixedOnly, 1, true))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(mixedOnly.stages[0].legalSimtFactors, (std::vector<int64_t>{1}));
  EXPECT_EQ(mixedOnly.stages[0].localSimtFactors, (std::vector<int64_t>{1}));
  auto mixedOnlyCosts = evaluateOneStage(mixedOnly.stages[0]);
  if (!mixedOnlyCosts)
    FAIL() << llvm::toString(mixedOnlyCosts.takeError());
  ASSERT_EQ(mixedOnlyCosts->stages[0].implementations.size(), 3u);
  EXPECT_EQ(mixedOnlyCosts->stages[0].legalSimtFactors,
            (std::vector<int64_t>{1}));
  EXPECT_EQ(mixedOnlyCosts->stages[0].localSimtFactors,
            (std::vector<int64_t>{1}));

  StagePartition oneWorkGroup = makePartition(/*independentGroups=*/1);
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(oneWorkGroup, 4, true))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(oneWorkGroup.stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));
}

TEST(SimdSimtCostModelTest, LocalScopeFactorsHonorKernelResourceMaximum) {
  StagePartition partition;
  LogicalStage stage;
  stage.id = "indirect_tile_gather";
  stage.costModelKind = StageCostModelKind::IndirectGatherMemory;
  stage.scheduleKind = StageScheduleKind::PartiallyDependent;
  stage.iterationCount = 1;
  stage.localSimtMaterializable = true;
  stage.localSuperblockMaterializable = true;
  partition.stages.push_back(std::move(stage));

  ASSERT_FALSE(StageModeLegalityAnalysis().analyze(
      partition, /*maximumSuperblockFactor=*/2,
      /*scopeSuperblockMaterializable=*/true));
  const LogicalStage &result = partition.stages.front();
  EXPECT_EQ(result.legalSimtFactors, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(result.localSimtFactors, (std::vector<int64_t>{1, 2}));
}

TEST(SimdSimtCostModelTest, KernelMixedRouteComesFromAdjacentStageModes) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto addStage = [&](llvm::StringRef id, double simd, double simt) {
    mlir::ascend::LogicalStageCost stage;
    stage.id = id.str();
    stage.localSimtMaterializable = true;
    stage.localSimtFactors = {1};
    auto cost = [&](StageMode mode, double cycles, bool localScope = false) {
      mlir::ascend::StageImplementationCost result;
      result.implementation = {mode, 1, localScope};
      result.totalCycles = cycles;
      return result;
    };
    stage.implementations = {cost(StageMode::SIMD, simd),
                             cost(StageMode::SIMT, simt),
                             cost(StageMode::SIMT, simt, true)};
    table.stages.push_back(stage);
  };
  addStage("head", 10.0, 20.0);
  addStage("payload", 100.0, 50.0);
  addStage("store", 30.0, 45.0);

  StageTransitionCost transition;
  transition.simdToSimtCycles = 5.0;
  transition.simtToSimdCycles = 7.0;
  auto result = solveStageRoutes(table, transition);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  EXPECT_DOUBLE_EQ(result->allSimd.totalCycles, 140.0);
  EXPECT_DOUBLE_EQ(result->allSimt.totalCycles, 115.0);
  EXPECT_DOUBLE_EQ(result->mixed.totalCycles, 102.0);
  ASSERT_EQ(result->mixed.implementations.size(), 3u);
  EXPECT_EQ(result->mixed.implementations[0].mode, StageMode::SIMD);
  EXPECT_EQ(result->mixed.implementations[1].mode, StageMode::SIMT);
  EXPECT_EQ(result->mixed.implementations[2].mode, StageMode::SIMD);
  ASSERT_EQ(result->mixed.entryTransitionCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(result->mixed.entryTransitionCycles[1], 12.0);
}

TEST(SimdSimtCostModelTest, MixedScopePaysExactBidirectionalUbHandoffCost) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.implementations = {makeCost(StageMode::SIMD, 10.0),
                          makeCost(StageMode::SIMT, 20.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "large_result_payload";
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1};
  payload.localSimtScopeCount = 2;
  payload.scopeInputTensorBytes = 4096;
  payload.scopeOutputTensorBytes = 16384;
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 10.0),
                             makeCost(StageMode::SIMT, 10.0, true)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, payload, tail};

  StageTransitionCost transition;
  transition.simdUbLoadBytesPerCycle = 512.0;
  transition.simdUbStoreBytesPerCycle = 256.0;
  transition.simtUbLoadBytesPerThreadPerCycle = 4.0;
  transition.simtUbStoreBytesPerThreadPerCycle = 4.0;
  transition.simtWarpSize = 32;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // Input: 4096/256 + 4096/(4*32) = 48 cycles.
  // Output: 16384/(4*32) + 16384/512 = 160 cycles.
  // Head/payload/tail: 10 + (10 + 208) + 10 = 238 cycles.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 238.0);
  ASSERT_EQ(routes->mixed.entryTransitionCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[1], 208.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[2], 0.0);
  EXPECT_GT(routes->mixed.totalCycles, routes->allSimd.totalCycles);
}

TEST(SimdSimtCostModelTest, MixedScopeSuperBlockAmortizesOnlyFixedTransitions) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, int64_t factor, double cycles,
                      bool localScope = false) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost head;
  head.id = "head";
  head.implementations = {makeCost(StageMode::SIMD, 1, 10.0),
                          makeCost(StageMode::SIMT, 1, 100.0)};

  LogicalStageCost payload;
  payload.id = "payload";
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1, 2, 4};
  payload.localSimtScopeCount = 1;
  payload.scopeInputTensorBytes = 4096;
  payload.scopeOutputTensorBytes = 4096;
  payload.implementations = {makeCost(StageMode::SIMD, 1, 1000.0),
                             makeCost(StageMode::SIMT, 1, 100.0),
                             makeCost(StageMode::SIMT, 1, 100.0, true),
                             makeCost(StageMode::SIMT, 2, 100.0, true),
                             makeCost(StageMode::SIMT, 4, 100.0, true)};

  LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, payload, tail};

  StageTransitionCost transition;
  transition.simdToSimtCycles = 40.0;
  transition.simtToSimdCycles = 40.0;
  transition.simdUbLoadBytesPerCycle = 512.0;
  transition.simdUbStoreBytesPerCycle = 256.0;
  transition.simtUbLoadBytesPerThreadPerCycle = 4.0;
  transition.simtUbStoreBytesPerThreadPerCycle = 4.0;
  transition.simtWarpSize = 32;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());

  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  // The 80-cycle fixed transition pair is amortized to 20 cycles.  The
  // 4096-byte input/output handoff remains 48 + 40 cycles per program.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles,
                   10.0 + 100.0 + 20.0 + 48.0 + 40.0 + 10.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[1], 108.0);
}

TEST(SimdSimtCostModelTest, PrefixScanUsesModeSpecificDependencyFactor) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.workload.operationElements.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 64.0;
  stage.workload.shuffleLaneSteps = 320.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.prefixScanDependencyFactor = 2.5;
  profile.simt.prefixScanDependencyFactor = 1.0;
  auto table = evaluateOneStage(std::move(stage), profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());

  const auto &implementations = table->stages.front().implementations;
  ASSERT_EQ(implementations.size(), 2u);
  EXPECT_EQ(implementations[0].implementation.mode, StageMode::SIMD);
  EXPECT_EQ(implementations[1].implementation.mode, StageMode::SIMT);
  EXPECT_GT(implementations[0].totalCycles, implementations[1].totalCycles);
}

TEST(SimdSimtCostModelTest, IndependentLoopUsesSimdRooflineAndSerialSimtCost) {
  LogicalStage stage =
      logicalStage("independent", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasLoop = true;
  stage.features.hasPointerInduction = true;

  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.loadWarpInstructions = 20.0;
  stage.workload.storeWarpInstructions = 10.0;
  stage.workload.dotFlops = 512.0;
  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_TRUE(stage.features.permitsSimdRoofline());
  EXPECT_LT(table->stages[0].implementations[0].totalCycles,
            table->stages[0].implementations[1].totalCycles);
}

TEST(SimdSimtCostModelTest, SimdCubeStageUsesCvPipelineCriticalPath) {
  LogicalStage stage =
      logicalStage("cube", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasDot = true;
  stage.features.hasContiguousMemory = true;
  stage.workload.operationElements.clear();
  stage.workload.scalarOperations = 16.0;
  stage.workload.loadBytes = 2048.0;
  stage.workload.storeBytes = 512.0;
  stage.workload.dotFlops = 8192.0;
  stage.workload.issueElements = 128.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());

  const StageImplementationCost &simd = table->stages[0].implementations[0];
  ASSERT_EQ(simd.implementation.mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(simd.resources.setup, 18.0);
  EXPECT_DOUBLE_EQ(simd.resources.load, 64.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 32.0);
  EXPECT_DOUBLE_EQ(simd.resources.dot, 128.0);
  // SIMD/Cube CV resources overlap inside one Stage.  The model charges the
  // critical resource (MMAD here), rather than serializing load+MMAD+store.
  EXPECT_DOUBLE_EQ(simd.totalCycles, 18.0 + 4.0 * (16.0 + 128.0));
}

TEST(SimdSimtCostModelTest, TrueLoopCarriedDependencyDisablesSimdRoofline) {
  LogicalStage stage =
      logicalStage("dependent", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;

  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.dotFlops = 512.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_FALSE(stage.features.permitsSimdRoofline());
  EXPECT_GT(table->stages[0].implementations[0].totalCycles, 0.0);
}

TEST(SimdSimtCostModelTest, ControlFlowUsesCountsRatesAndLaneActivity) {
  LogicalStage stage =
      logicalStage("control", StageCostModelKind::ScalarControl,
                   StageScheduleKind::StraightLine, 2);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.features.conditionalBranchCount = 3;
  stage.features.divergentBranchCount = 2;
  stage.features.loopBackedgeCount = 1;
  stage.features.synchronizationCount = 1;
  stage.features.activeLaneRatio = 0.5;

  stage.workload.scalarOperations = 1.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages[0].implementations[0];
  EXPECT_DOUBLE_EQ(cost.resources.loopControl, 2.0);
  EXPECT_DOUBLE_EQ(cost.resources.branchControl, 9.0);
  EXPECT_DOUBLE_EQ(cost.resources.divergence, 10.0);
  EXPECT_DOUBLE_EQ(cost.resources.synchronization, 7.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 196.0);
}

TEST(SimdSimtCostModelTest, RecurrenceAccumulatesCriticalPathAndTraffic) {
  LogicalStage stage =
      logicalStage("recurrence", StageCostModelKind::LoopCarriedRecurrence,
                   StageScheduleKind::LoopCarriedSerial, 4);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;

  stage.workload.loadWarpInstructions = 10.0;
  stage.workload.storeWarpInstructions = 5.0;
  stage.workload.estimatedSpillTransactions = 7.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_GT(table->stages[0].implementations[0].totalCycles, 100.0);
  EXPECT_GT(table->stages[0].implementations[0].resources.criticalPath, 0.0);
}

TEST(SimdSimtCostModelTest, SimdRecurrenceChargesPersistentLiveStateBytes) {
  LogicalStage baseline = logicalStage(
      "baseline_recurrence", StageCostModelKind::LoopCarriedRecurrence,
      StageScheduleKind::LoopCarriedSerial);
  baseline.features.hasLoop = true;
  baseline.features.hasLoopCarriedDataDependency = true;
  LogicalStage withState = baseline;
  withState.id = "stateful_recurrence";
  withState.liveOutBytes = 800;

  auto baselineTable = evaluateOneStage(baseline);
  auto stateTable = evaluateOneStage(withState);
  if (!baselineTable)
    FAIL() << llvm::toString(baselineTable.takeError());
  if (!stateTable)
    FAIL() << llvm::toString(stateTable.takeError());
  const double baselineSimd =
      baselineTable->stages[0].implementations[0].totalCycles;
  const double stateSimd = stateTable->stages[0].implementations[0].totalCycles;
  EXPECT_DOUBLE_EQ(stateSimd - baselineSimd, 800.0 / 8.0);
}

TEST(SimdSimtCostModelTest,
     SimtRecurrenceInterleavesIndependentGroupsButKeepsIssueFloor) {
  LogicalStage serial = logicalStage("serial_recurrence",
                                     StageCostModelKind::LoopCarriedRecurrence,
                                     StageScheduleKind::LoopCarriedSerial, 16);
  serial.simdLegal = false;
  serial.features.hasLoop = true;
  serial.features.hasLoopCarriedDataDependency = true;
  serial.workload.shuffleLaneSteps = 128.0;
  serial.workload.issueElements = 64.0;

  LogicalStage grouped = serial;
  grouped.id = "grouped_recurrence";
  grouped.features.parallelRecurrenceGroupCount = 4;
  HardwareProfile profile = hardwareProfile();
  profile.logicalWarpGroupCount = 4;

  auto serialTable = evaluateOneStage(std::move(serial), profile);
  auto groupedTable = evaluateOneStage(std::move(grouped), profile);
  if (!serialTable)
    FAIL() << llvm::toString(serialTable.takeError());
  if (!groupedTable)
    FAIL() << llvm::toString(groupedTable.takeError());
  const double serialCycles =
      serialTable->stages[0].implementations[0].totalCycles;
  const double groupedCycles =
      groupedTable->stages[0].implementations[0].totalCycles;
  EXPECT_LT(groupedCycles, serialCycles);
  const auto &resources = groupedTable->stages[0].implementations[0].resources;
  EXPECT_GE(groupedCycles, resources.setup + 16.0 * resources.issue);
}

TEST(SimdSimtCostModelTest,
     SuperBlockRecurrenceContentionStartsAbovePressureFreeFactor) {
  LogicalStage stage = logicalStage("stateful_recurrence",
                                    StageCostModelKind::LoopCarriedRecurrence,
                                    StageScheduleKind::LoopCarriedSerial, 16);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.features.parallelRecurrenceGroupCount = 4;
  stage.liveOutBytes = 4096;
  stage.workload.loadWarpInstructions = 16.0;
  stage.workload.shuffleLaneSteps = 128.0;
  stage.workload.issueElements = 64.0;

  HardwareProfile profile = hardwareProfile();
  profile.logicalWarpGroupCount = 4;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_LE(costs[1].totalCycles, costs[0].totalCycles);
  EXPECT_LE(costs[2].totalCycles, costs[1].totalCycles);
  EXPECT_GE(costs[2].totalCycles,
            costs[2].resources.setup + 16.0 * costs[2].resources.issue);

  stage.workload.estimatedSpillTransactions = 32.0;
  auto spillingTable = evaluateOneStage(stage, profile);
  if (!spillingTable)
    FAIL() << llvm::toString(spillingTable.takeError());
  const auto &spillingCosts = spillingTable->stages.front().implementations;
  ASSERT_EQ(spillingCosts.size(), 3u);
  EXPECT_GT(spillingCosts[2].totalCycles, spillingCosts[1].totalCycles);
}

TEST(SimdSimtCostModelTest,
     MixedLocalStageSuperBlockHidesRecurrenceLatencyButKeepsIssueFloor) {
  LogicalStage stage = logicalStage(
      "mixed_recurrence", StageCostModelKind::LoopCarriedRecurrence,
      StageScheduleKind::LoopCarriedSerial, /*iterations=*/16);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1};
  stage.localSimtMaterializable = true;
  stage.localSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.workload.loadWarpInstructions = 16.0;
  stage.workload.shuffleLaneSteps = 128.0;
  stage.workload.issueElements = 64.0;

  auto table = evaluateOneStage(stage, hardwareProfile());
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 4u);
  EXPECT_FALSE(costs[0].implementation.localScope);
  EXPECT_TRUE(costs[1].implementation.localScope);
  EXPECT_TRUE(costs[2].implementation.localScope);
  EXPECT_TRUE(costs[3].implementation.localScope);
  EXPECT_LE(costs[2].totalCycles, costs[1].totalCycles);
  EXPECT_LE(costs[3].totalCycles, costs[2].totalCycles);
  EXPECT_GE(costs[3].totalCycles,
            costs[3].resources.setup + 16.0 * costs[3].resources.issue);
}

TEST(SimdSimtCostModelTest, IndirectMemoryUsesDependencyProfile) {
  LogicalStage stage =
      logicalStage("indirect", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.indirectLoadBytes = 1024.0;
  stage.workload.indirectLoadTransactions = 8.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.indirectLoadTransactionsPerCycle = 0.25;
  profile.simd.indirectDependencyLatencyCycles = 80.0;
  profile.simt.indirectLoadTransactionsPerCycle = 1.0;
  profile.simt.indirectDependencyLatencyCycles = 20.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 112.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 28.0);
  EXPECT_LT(costs[1].totalCycles, costs[0].totalCycles);
}

TEST(SimdSimtCostModelTest,
     AtomicCostDoesNotRepriceOrdinaryLoadsAsIndirectMemory) {
  LogicalStage stage = logicalStage("atomic", StageCostModelKind::AtomicMemory,
                                    StageScheduleKind::PartiallyDependent);
  stage.features.hasAtomicMemory = true;
  stage.features.hasIndirectMemory = true;
  stage.features.hasContiguousMemory = true;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;
  mlir::ascend::AtomicWorkload atomic;
  atomic.kind = "fadd";
  atomic.dataType = "f32";
  atomic.memorySemantic = "acq_rel";
  atomic.memoryScope = "gpu";
  atomic.logicalElements = 64.0;
  atomic.logicalOperationInstances = 1.0;
  atomic.contentionUnknown = true;
  stage.workload.atomicWorkloads.push_back(atomic);

  HardwareProfile profile = hardwareProfile();
  profile.simd.atomicRates["fadd.f32"] = {8.0, 2.0, 5.0, 3.0};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &simd = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(simd.resources.load, 32.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 0.0);
  EXPECT_DOUBLE_EQ(simd.resources.atomic, 30.0);
}

TEST(SimdSimtCostModelTest,
     WorkloadPreservesAtomicSemanticsWithoutCountingAStore) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%load_pointer: tensor<64x!tt.ptr<f32>>,
                        %index_pointer: tensor<64x!tt.ptr<i64>>,
                        %base_pointer: tensor<64x!tt.ptr<f32>>,
                        %value: tensor<64xf32>) {
        %regular = "tt.load"(%load_pointer)
          : (tensor<64x!tt.ptr<f32>>) -> tensor<64xf32>
        %index = "tt.load"(%index_pointer)
          : (tensor<64x!tt.ptr<i64>>) -> tensor<64xi64>
        %address = "tt.addptr"(%base_pointer, %index)
          : (tensor<64x!tt.ptr<f32>>, tensor<64xi64>)
            -> tensor<64x!tt.ptr<f32>>
        %mask = arith.constant dense<true> : tensor<64xi1>
        %old = "tt.atomic_rmw"(%address, %value, %mask)
          {atomic_rmw_op = 5 : i32, sem = 4 : i32, scope = 1 : i32}
          : (tensor<64x!tt.ptr<f32>>, tensor<64xf32>, tensor<64xi1>)
            -> tensor<64xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("atomic_workload", StageCostModelKind::AtomicMemory);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  ASSERT_TRUE(function);
  mlir::Block &body = function.getBody().front();
  for (mlir::Operation &operation : body.without_terminator())
    stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageFeatureAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 8192))
    FAIL() << llvm::toString(std::move(error));

  const LogicalStage &analyzed = partition.stages.front();
  EXPECT_TRUE(analyzed.features.hasAtomicMemory);
  EXPECT_TRUE(analyzed.features.hasIndirectMemory);
  EXPECT_TRUE(analyzed.features.hasContiguousMemory);
  EXPECT_DOUBLE_EQ(analyzed.workload.storeBytes, 0.0);
  EXPECT_DOUBLE_EQ(analyzed.workload.storeWarpInstructions, 0.0);
  ASSERT_EQ(analyzed.workload.atomicWorkloads.size(), 1u);
  const auto &atomic = analyzed.workload.atomicWorkloads.front();
  EXPECT_EQ(atomic.kind, "fadd");
  EXPECT_EQ(atomic.dataType, "f32");
  EXPECT_EQ(atomic.memorySemantic, "acq_rel");
  EXPECT_EQ(atomic.memoryScope, "gpu");
  EXPECT_DOUBLE_EQ(atomic.logicalElements, 64.0);
  EXPECT_FALSE(atomic.resultUsed);
  EXPECT_TRUE(atomic.addressDependsOnLoadedIndex);
  EXPECT_TRUE(atomic.contentionUnknown);
  EXPECT_EQ(analyzed.costModelKind, StageCostModelKind::AtomicMemory);
}

TEST(SimdSimtCostModelTest, MixedRouteRejectsUnmaterializableSimtStage) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.localSimtMaterializable = false;
  head.implementations = {makeCost(StageMode::SIMD, 1.0),
                          makeCost(StageMode::SIMT, 100.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "unmaterializable_payload";
  payload.localSimtMaterializable = false;
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 1.0)};
  table.stages = {head, payload};
  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimt.legal);
  EXPECT_FALSE(routes->mixed.legal);
}

TEST(SimdSimtCostModelTest,
     MixedRouteReportsCheapestConstrainedRouteWhenLocalScopeLoses) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  mlir::ascend::LogicalStageCost gather;
  gather.id = "indirect_tile_gather";
  gather.localSimtMaterializable = true;
  gather.localSimtScopeCount = 1;
  gather.implementations = {makeCost(StageMode::SIMD, 100.0),
                            makeCost(StageMode::SIMT, 130.0),
                            makeCost(StageMode::SIMT, 130.0, true)};
  mlir::ascend::LogicalStageCost dot;
  dot.id = "tiny_cube_dot";
  dot.implementations = {makeCost(StageMode::SIMD, 40.0),
                         makeCost(StageMode::SIMT, 90.0)};
  table.stages = {gather, dot};

  StageTransitionCost transition;
  transition.simdToSimtCycles = 10.0;
  transition.simtToSimdCycles = 10.0;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  ASSERT_EQ(routes->mixed.implementations.size(), 2u);
  EXPECT_EQ(routes->mixed.implementations[0].mode, StageMode::SIMT);
  EXPECT_TRUE(routes->mixed.implementations[0].localScope);
  EXPECT_EQ(routes->mixed.implementations[1].mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 190.0);
}

TEST(SimdSimtCostModelTest, AllSimdDoesNotPayRouteConditionalAutoBlockify) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, false};
    cost.totalCycles = cycles;
    return cost;
  };

  mlir::ascend::LogicalStageCost dispatch;
  dispatch.id = "physical_program_dispatch";
  dispatch.model = "auto_blockify_dispatch";
  dispatch.implementations = {makeCost(StageMode::SIMD, 40.0),
                              makeCost(StageMode::SIMT, 30.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "payload";
  payload.model = "scalar_issue";
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 80.0)};
  table.stages = {dispatch, payload};

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  ASSERT_EQ(routes->allSimd.logicalStageCycles.size(), 2u);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[1], 100.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 100.0);
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 110.0);
}

TEST(SimdSimtCostModelTest, MixedRouteChargesEveryMaterializedScope) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.implementations = {makeCost(StageMode::SIMD, 1.0),
                          makeCost(StageMode::SIMT, 100.0)};
  mlir::ascend::LogicalStageCost gather;
  gather.id = "two_anchor_gather";
  gather.localSimtMaterializable = true;
  gather.localSimtFactors = {1};
  gather.localSimtScopeCount = 2;
  gather.implementations = {makeCost(StageMode::SIMD, 100.0),
                            makeCost(StageMode::SIMT, 1.0),
                            makeCost(StageMode::SIMT, 1.0, true)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, gather, tail};

  StageTransitionCost transition;
  transition.simdToSimtCycles = 10.0;
  transition.simtToSimdCycles = 10.0;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // 1 SIMD head + (10 enter + 1 payload + 20 extra scope pair) +
  // (10 leave + 1 SIMD tail).
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 43.0);
  ASSERT_EQ(routes->mixed.entryTransitionCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[1], 40.0);
}

TEST(SimdSimtCostModelTest, SuperBlockLatencyHidingStopsAtUsefulFactorLimit) {
  LogicalStage stage =
      logicalStage("simt_payload", StageCostModelKind::ScalarIssue);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.workload.loadWarpInstructions = 40.0;
  HardwareProfile cappedProfile = hardwareProfile();
  cappedProfile.superblockUsefulFactorLimit = 2;
  cappedProfile.superblockPersistentStatePressureFreeFactor = 2;
  auto table = evaluateOneStage(stage, cappedProfile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  table->logicalProgramCountHint = 64;
  table->physicalCoreCountHint = 32;
  auto routes = solveStageRoutes(*table, cappedProfile.transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimt.legal);
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 2);
  EXPECT_LT(routes->allSimt.totalCycles,
            2.0 * table->stages[0].implementations[0].totalCycles);
}

TEST(SimdSimtCostModelTest, PureSimtRouteUsesOneUniformSuperBlockFactor) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](int64_t factor, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost first;
  first.id = "first";
  first.implementations = {makeCost(1, 5.0), makeCost(2, 1.0),
                           makeCost(4, 3.0)};
  mlir::ascend::LogicalStageCost second;
  second.id = "second";
  second.implementations = {makeCost(1, 5.0), makeCost(2, 4.0),
                            makeCost(4, 1.0)};
  table.stages = {first, second};

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 4);
  ASSERT_EQ(routes->allSimt.implementations.size(), 2u);
  EXPECT_EQ(routes->allSimt.implementations[0].superblockFactor, 4);
  EXPECT_EQ(routes->allSimt.implementations[1].superblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 4.0);
}

TEST(SimdSimtCostModelTest, MixedScopeSuperBlockUsesSelectedFactorCost) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, int64_t factor, double cycles,
                      bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost prefix;
  prefix.id = "simd_prefix";
  prefix.implementations = {makeCost(StageMode::SIMD, 1, 5.0),
                            makeCost(StageMode::SIMT, 1, 50.0),
                            makeCost(StageMode::SIMT, 2, 25.0),
                            makeCost(StageMode::SIMT, 4, 12.5),
                            makeCost(StageMode::SIMT, 1, 50.0, true),
                            makeCost(StageMode::SIMT, 2, 25.0, true),
                            makeCost(StageMode::SIMT, 4, 12.5, true)};
  prefix.localSimtMaterializable = true;
  prefix.localSimtFactors = {1, 2, 4};

  mlir::ascend::LogicalStageCost payload;
  payload.id = "local_simt_payload";
  payload.implementations = {makeCost(StageMode::SIMD, 1, 100.0),
                             makeCost(StageMode::SIMT, 1, 10.0),
                             makeCost(StageMode::SIMT, 2, 1.0),
                             makeCost(StageMode::SIMT, 4, 0.5),
                             makeCost(StageMode::SIMT, 1, 10.0, true),
                             makeCost(StageMode::SIMT, 2, 1.0, true),
                             makeCost(StageMode::SIMT, 4, 0.5, true)};
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1, 2, 4};
  table.stages = {prefix, payload};

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 5.5);
}

TEST(SimdSimtCostModelTest,
     FactoredMixedRouteUsesOneBackendMaterializableLocalScope) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, int64_t factor, double cycles,
                      bool localScope = false) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost first;
  first.id = "first_local_candidate";
  first.features.replicatedByLocalSuperBlock = true;
  first.localSimtMaterializable = true;
  first.localSimtScopeCount = 1;
  first.implementations = {makeCost(StageMode::SIMD, 1, 100.0),
                           makeCost(StageMode::SIMT, 4, 1.0, true)};
  LogicalStageCost second = first;
  second.id = "second_local_candidate";
  LogicalStageCost tail;
  tail.id = "simd_tail";
  tail.implementations = {makeCost(StageMode::SIMD, 1, 1.0)};
  table.stages = {first, second, tail};

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_EQ(llvm::count_if(
                routes->mixed.implementations,
                [](const mlir::ascend::StageImplementation &implementation) {
                  return implementation.localScope;
                }),
            1);
  // One scope is SIMT (1 cycle); the other Stage remains SIMD and is cloned
  // once per grouped logical program (100 * F4); the outside tail is not.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 402.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[1], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[2], 0.0);
}

TEST(SimdSimtCostModelTest,
     OperationGraphBoundaryOwnsEveryRootAndDerivesLiveValues) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16x16xf32>
        %result = scf.for %i = %c2 to %c16 step %c1
            iter_args(%state = %loaded) -> tensor<16x16xf32> {
          %next = arith.addf %state, %loaded : tensor<16x16xf32>
          scf.yield %next : tensor<16x16xf32>
        }
        "tt.store"(%pointer, %result) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *recurrence = nullptr;
  module->walk([&](mlir::scf::ForOp loop) { recurrence = loop; });
  ASSERT_NE(recurrence, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = recurrence;
  anchor.scopeOperations.push_back(recurrence);
  anchor.scopeInsertionPoint = recurrence;
  anchor.kind = mlir::ascend::SimtAnchorKind::TriangularSolveLoop;
  anchor.triangularSolve =
      triangularBt16StageFeatures().simtAnchors.triangularSolves.front();
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  auto structure =
      mlir::ascend::ProgramStructureAnalysis().analyze(*module, anchorPlan);
  if (!structure)
    FAIL() << llvm::toString(structure.takeError());
  EXPECT_EQ(structure->rootOperations.size(), 6u);

  auto result = StagePartitioner().partition(*module, anchorPlan,
                                             StagePartitionerOptions{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  const StagePartition &partition = *result;
  EXPECT_TRUE(partition.operationOwnershipComplete);

  int64_t ownedRootCount = 0;
  const LogicalStage *recurrenceStage = nullptr;
  for (const LogicalStage &stage : partition.stages) {
    ownedRootCount += static_cast<int64_t>(stage.operations.size());
    if (llvm::is_contained(stage.operations, recurrence))
      recurrenceStage = &stage;
  }
  EXPECT_EQ(ownedRootCount, partition.modeledOperationCount);
  ASSERT_NE(recurrenceStage, nullptr);
  EXPECT_EQ(recurrenceStage->operations.size(), 1u);
  EXPECT_FALSE(recurrenceStage->liveIns.empty());
  EXPECT_EQ(recurrenceStage->liveOuts.size(), 1u);
  EXPECT_EQ(recurrenceStage->simtAnchorIndices, std::vector<unsigned>({0}));
  EXPECT_TRUE(recurrenceStage->localSimtMaterializable);
  // The arith.addf is the scf.for body and therefore represents 256 element
  // additions on every one of the 14 dynamic recurrence iterations.  The
  // per-iteration Stage workload must remain 256, not be divided by 14.
  auto add = recurrenceStage->workload.operationElements.find("f32.add");
  ASSERT_NE(add, recurrenceStage->workload.operationElements.end());
  EXPECT_DOUBLE_EQ(add->second, 256.0);
}

TEST(SimdSimtCostModelTest,
     SameStatementSupportOperationsJoinTheDominantResourceStage) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %index = arith.constant 0 : i64
        %mask = arith.cmpi eq, %index, %index : i64
        %value = "tt.load"(%pointer, %mask) : (i64, i1) -> f32
        %tail = arith.constant 1 : i64
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::Operation *> roots;
  module->walk([&](mlir::func::FuncOp function) {
    for (mlir::Operation &operation : function.getBody().front())
      if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
        roots.push_back(&operation);
  });
  ASSERT_EQ(roots.size(), 4u);
  auto statement = mlir::FileLineColLoc::get(&context, "kernel.py", 10, 1);
  for (mlir::Operation *operation : llvm::ArrayRef(roots).take_front(3))
    operation->setLoc(statement);
  roots.back()->setLoc(mlir::FileLineColLoc::get(&context, "kernel.py", 11, 1));

  mlir::ascend::ProgramStructure structure;
  structure.rootOperations.assign(roots.begin(), roots.end());
  auto result = mlir::ascend::StageBoundaryAnalysis().analyze(
      structure, mlir::ascend::SimtAnchorPlan{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->stages.size(), 2u);
  EXPECT_EQ(result->stages.front().operations.size(), 3u);
  EXPECT_EQ(result->stages.front().costModelKind,
            StageCostModelKind::ContinuousTileMemory);
  ASSERT_EQ(result->stages.back().operations.size(), 1u);
  EXPECT_EQ(result->stages.back().operations.front(), roots.back());
}

TEST(SimdSimtCostModelTest,
     CompoundScopeOrderIsNormalizedBeforeStagePartitioning) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %setup = arith.constant dense<0> : tensor<16xi32>
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16x16xf32>
        %result = scf.for %i = %c2 to %c16 step %c1
            iter_args(%state = %loaded) -> tensor<16x16xf32> {
          %next = arith.addf %state, %loaded : tensor<16x16xf32>
          scf.yield %next : tensor<16x16xf32>
        }
        "tt.store"(%pointer, %result) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *setup = nullptr;
  mlir::Operation *recurrence = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "arith.constant" &&
        operation->getNumResults() == 1 &&
        mlir::isa<mlir::RankedTensorType>(operation->getResult(0).getType()))
      setup = operation;
    if (mlir::isa<mlir::scf::ForOp>(operation))
      recurrence = operation;
  });
  ASSERT_NE(setup, nullptr);
  ASSERT_NE(recurrence, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = recurrence;
  anchor.scopeOperations = {setup, recurrence};
  anchor.scopeInsertionPoint = recurrence;
  anchor.kind = mlir::ascend::SimtAnchorKind::TriangularSolveLoop;
  anchor.triangularSolve =
      triangularBt16StageFeatures().simtAnchors.triangularSolves.front();
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  auto structure =
      mlir::ascend::ProgramStructureAnalysis().analyze(*module, anchorPlan);
  if (!structure)
    FAIL() << llvm::toString(structure.takeError());
  auto setupPosition = llvm::find(structure->rootOperations, setup);
  auto recurrencePosition = llvm::find(structure->rootOperations, recurrence);
  ASSERT_NE(setupPosition, structure->rootOperations.end());
  ASSERT_NE(recurrencePosition, structure->rootOperations.end());
  EXPECT_EQ(recurrencePosition - setupPosition, 1);

  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());
  const LogicalStage *loadStage = nullptr;
  const LogicalStage *recurrenceStage = nullptr;
  for (const LogicalStage &stage : partition->stages) {
    if (llvm::any_of(stage.operations, [&](mlir::Operation *operation) {
          return operation->getName().getStringRef() == "tt.load";
        }))
      loadStage = &stage;
    if (llvm::is_contained(stage.operations, recurrence))
      recurrenceStage = &stage;
  }
  ASSERT_NE(loadStage, nullptr);
  ASSERT_NE(recurrenceStage, nullptr);
  EXPECT_EQ(loadStage->operations.size(), 1u);
  EXPECT_EQ(recurrenceStage->operations.size(), 2u);
  EXPECT_EQ(recurrenceStage->simtAnchorIndices, std::vector<unsigned>({0}));
}

TEST(SimdSimtCostModelTest,
     NestedLocalScopeDoesNotAdvertiseUnsupportedSuperBlockFactors) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64, %condition: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          scf.if %condition {
            %value = "tt.load"(%pointer) : (i64) -> tensor<16xf32>
          }
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::scf::ForOp v1Loop;
  mlir::Operation *nestedLoad = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation))
      v1Loop = loop;
    if (operation->getName().getStringRef() == "tt.load")
      nestedLoad = operation;
  });
  ASSERT_TRUE(v1Loop);
  ASSERT_NE(nestedLoad, nullptr);
  v1Loop->setAttr("ta.auto_blockify_v1.loop", mlir::UnitAttr::get(&context));

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = nestedLoad;
  anchor.scopeOperations.push_back(nestedLoad);
  anchor.scopeInsertionPoint = nestedLoad;
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  StagePartitionerOptions options;
  options.maximumSuperblockFactor = 4;
  options.scopeSuperblockMaterializable = true;
  auto result = StagePartitioner().partition(*module, anchorPlan, options);
  if (!result)
    FAIL() << llvm::toString(result.takeError());

  const LogicalStage *nestedStage = nullptr;
  for (const LogicalStage &stage : result->stages)
    if (!stage.simtAnchorIndices.empty())
      nestedStage = &stage;
  ASSERT_NE(nestedStage, nullptr);
  EXPECT_TRUE(nestedStage->localSimtMaterializable);
  EXPECT_FALSE(nestedStage->localSuperblockMaterializable);
  EXPECT_FALSE(nestedStage->operations.empty());
  EXPECT_EQ(nestedStage->localSimtFactors, (std::vector<int64_t>{1}));

  auto costs = StageCostEvaluator().evaluate(*result, hardwareProfile());
  if (!costs)
    FAIL() << llvm::toString(costs.takeError());
  auto nestedCost = llvm::find_if(costs->stages, [&](const auto &stage) {
    return stage.id == nestedStage->id;
  });
  ASSERT_NE(nestedCost, costs->stages.end());
  EXPECT_FALSE(nestedCost->sourceLocations.empty());
}

TEST(SimdSimtCostModelTest, GenericSemanticStagesDoNotRequireAWorkloadDomain) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @unrelated_kernel(%pointer: i64) {
        %zero = arith.constant dense<0.0> : tensor<16xf32>
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16xf32>
        %mask = arith.cmpf ogt, %loaded, %zero : tensor<16xf32>
        "tt.store"(%pointer, %loaded) : (i64, tensor<16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::ascend::SimtAnchorPlan anchorPlan;
  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  ASSERT_TRUE(partition->operationOwnershipComplete);
  EXPECT_EQ(partition->modeledOperationCount, 4);
  ASSERT_EQ(partition->stages.size(), 4u);
  EXPECT_EQ(partition->stages[0].costModelKind,
            StageCostModelKind::ScalarIssue);
  EXPECT_EQ(partition->stages[1].costModelKind,
            StageCostModelKind::ContinuousTileMemory);
  EXPECT_EQ(partition->stages[2].costModelKind,
            StageCostModelKind::PredicateMask);
  EXPECT_EQ(partition->stages[3].costModelKind,
            StageCostModelKind::ContinuousTileStore);
}

TEST(SimdSimtCostModelTest, AdjacentStructuredLoopsRemainSerialStages) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @two_serial_loops(%pointer: i64) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %first = "tt.load"(%pointer) : (i64) -> f32
        }
        scf.for %i = %c0 to %c4 step %c1 {
          %second = "tt.load"(%pointer) : (i64) -> f32
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::ascend::SimtAnchorPlan anchorPlan;
  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  llvm::SmallVector<const LogicalStage *> loopStages;
  for (const LogicalStage &stage : partition->stages)
    if (stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop)
      loopStages.push_back(&stage);
  ASSERT_EQ(loopStages.size(), 2u);
  EXPECT_NE(loopStages[0]->operations.front(),
            loopStages[1]->operations.front());
}

TEST(SimdSimtCostModelTest,
     LocalScopeReturningPointerTensorIsRejectedBeforeScoring) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%base: !tt.ptr<f16>) {
        %indices = arith.constant dense<0> : tensor<16xi32>
        %ptrs = "tt.addptr"(%base, %indices)
            : (!tt.ptr<f16>, tensor<16xi32>) -> tensor<16x!tt.ptr<f16>>
        %values = "tt.load"(%ptrs)
            : (tensor<16x!tt.ptr<f16>>) -> tensor<16xf16>
        %reduced = "tt.reduce"(%values) : (tensor<16xf16>) -> f16
        "tt.store"(%base, %reduced) : (!tt.ptr<f16>, f16) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::Operation *> roots;
  module->walk([&](mlir::func::FuncOp function) {
    for (mlir::Operation &operation : function.getBody().front())
      if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
        roots.push_back(&operation);
  });
  ASSERT_EQ(roots.size(), 5u);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = roots[1];
  anchor.scopeOperations = {roots[1]};
  anchor.scopeInsertionPoint = roots[1];
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  mlir::ascend::ProgramStructure structure;
  structure.rootOperations.assign(roots.begin(), roots.end());
  auto result =
      mlir::ascend::StageBoundaryAnalysis().analyze(structure, anchorPlan);
  if (!result)
    FAIL() << llvm::toString(result.takeError());

  const LogicalStage *gather = nullptr;
  for (const LogicalStage &stage : result->stages)
    if (llvm::is_contained(stage.operations, roots[1]))
      gather = &stage;
  ASSERT_NE(gather, nullptr);
  EXPECT_FALSE(gather->localSimtMaterializable);
  EXPECT_TRUE(gather->localSimtFactors.empty());
  EXPECT_TRUE(gather->simtAnchorIndices.empty());
}

TEST(SimdSimtCostModelTest, PointerInductionLoopIsNotADataRecurrence) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%start: i64) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %step = arith.constant 16 : i64
        %address = scf.for %i = %c0 to %c8 step %c1
            iter_args(%current = %start) -> i64 {
          %value = "tt.load"(%current) : (i64) -> f32
          %next = arith.addi %current, %step : i64
          scf.yield %next : i64
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *loop = nullptr;
  module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
  ASSERT_NE(loop, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("pointer_loop", StageCostModelKind::ConversionPack,
                   StageScheduleKind::IndependentPipelined, 8);
  stage.operations.push_back(loop);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageFeatureAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 8192))
    FAIL() << llvm::toString(std::move(error));
  const LogicalStage &classified = partition.stages.front();
  EXPECT_TRUE(classified.features.hasLoop);
  EXPECT_TRUE(classified.features.hasPointerInduction);
  EXPECT_FALSE(classified.features.hasLoopCarriedDataDependency);
  EXPECT_EQ(classified.costModelKind,
            StageCostModelKind::IndependentPipelinedLoop);
}

TEST(SimdSimtCostModelTest, IncompatibleDominantStructuresRequireStageSplit) {
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("gather_dot", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::PartiallyDependent, 1);
  stage.features.hasDot = true;
  stage.features.hasIndirectMemory = true;
  partition.stages.push_back(std::move(stage));

  llvm::Error error =
      mlir::ascend::StageKindClassifier().analyze(partition, 16384);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("requires_split"),
            std::string::npos);
}
