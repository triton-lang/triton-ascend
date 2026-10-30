#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Analysis/LoopWorkloadAnalysis.h"
#include "AscendModel/Analysis/StageMemoryPatternAnalysis.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/StageCostModels.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <array>
#include <thread>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>

using mlir::ascend::analyzeLoopTripCountInfos;
using mlir::ascend::HardwareProfile;
using mlir::ascend::LogicalStage;
using mlir::ascend::LogicalStageCost;
using mlir::ascend::ScopeHandoffCost;
using mlir::ascend::SimdSimtFeatureSummary;
using mlir::ascend::solveStageRoutes;
using mlir::ascend::StageCostEvaluator;
using mlir::ascend::StageCostModelKind;
using mlir::ascend::StageCostTable;
using mlir::ascend::StageFeatureAnalysis;
using mlir::ascend::StageImplementationCost;
using mlir::ascend::StageMode;
using mlir::ascend::StageModeLegalityAnalysis;
using mlir::ascend::StageModeProfile;
using mlir::ascend::StagePartition;
using mlir::ascend::StagePartitioner;
using mlir::ascend::StagePartitionerOptions;
using mlir::ascend::StageScheduleKind;
using mlir::ascend::StageWorkload;
using mlir::ascend::StageWorkloadAnalysis;
using mlir::ascend::TensorOperationWorkload;
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

HardwareProfile hardwareProfile(ScopeHandoffCost scopeHandoff = {}) {
  HardwareProfile profile;
  profile.profileVersion = "unit-test-profile-v1";
  profile.target = "Ascend950PR_9579";
  profile.superblockUsefulFactorLimit = 4;
  profile.superblockPersistentStatePressureFreeFactor = 2;
  profile.superblockPersistentStateBytesPerCycle = 8.0;
  auto fill = [](auto &mode) {
    mode.setupCycles = 10.0;
    mode.vectorWidthBits = 2048;
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
    mode.reductionLaneStepsPerCycle = 1.0;
    mode.shuffleLanesPerCycle = 32.0;
    mode.reductionLaneStepsPerCycle = 1.0;
    mode.prefixScanStepLatencyCycles = 1.0;
    mode.dotSetupCycles = 8.0;
    mode.dotFlopsPerCycle = 64.0;
    mode.issueOperationsPerCycle = 4.0;
    mode.prefixScanStepLatencyCycles = 2.0;
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
  profile.simt.vectorWidthBits = 1;
  profile.simt.issueWidth = 32;
  profile.scopeHandoff = std::move(scopeHandoff);
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
  stage.workload.operationCounts["f32.add"][64] = 1.0;
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

TEST(SimdSimtCostModelTest, SimdPricesShortAxesPerSegmentAndElementWidth) {
  auto simdCost = [](int64_t elementBits, int64_t contiguousElements,
                     double segmentCount) {
    LogicalStage stage =
        logicalStage("short-axis", StageCostModelKind::ScalarMath);
    stage.workload.operationCounts.clear();
    stage.workload
        .operationCounts["f32.add"][contiguousElements * segmentCount] = 1.0;
    TensorOperationWorkload tensor;
    tensor.operation = "f32.add";
    tensor.elementBitWidth = elementBits;
    tensor.logicalElements =
        static_cast<double>(contiguousElements) * segmentCount;
    tensor.segmentCount = segmentCount;
    tensor.contiguousElementsPerSegment = contiguousElements;
    stage.workload.tensorOperationWorkloads.push_back(std::move(tensor));
    auto table = evaluateOneStage(std::move(stage));
    if (!table) {
      ADD_FAILURE() << llvm::toString(table.takeError());
      return mlir::ascend::StageResourceCycles{};
    }
    return table->stages.front().implementations.front().resources;
  };

  // These are already projected segments (e.g. broadcast-constrained rows),
  // not initial TTIR shapes. A short segment still uses a vector instruction.
  auto fourBySixteen = simdCost(32, 16, 4.0);
  auto eightByEight = simdCost(32, 8, 8.0);
  auto sixteenByFour = simdCost(32, 4, 16.0);
  EXPECT_DOUBLE_EQ(fourBySixteen.compute, 4.0);
  EXPECT_DOUBLE_EQ(eightByEight.compute, 8.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.compute, 16.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.scalar, 0.0);
  EXPECT_DOUBLE_EQ(fourBySixteen.issue, 1.0);
  EXPECT_DOUBLE_EQ(eightByEight.issue, 2.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.issue, 4.0);

  // FP16 uses twice as many elements per 256-byte instruction; width is not
  // silently inherited from the old FP32-only vectorWidth field.
  auto fp16Rows = simdCost(16, 128, 2.0);
  EXPECT_DOUBLE_EQ(fp16Rows.compute, 2.0);
}

TEST(SimdSimtCostModelTest,
     WorkloadDoesNotInferSegmentsOrScalarFallbackFromLogicalRows) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%a4x16: tensor<4x16xf32>,
                        %b4x16: tensor<4x16xf32>,
                        %a8x8: tensor<8x8xf32>,
                        %b8x8: tensor<8x8xf32>,
                        %a16x4: tensor<16x4xf32>,
                        %b16x4: tensor<16x4xf32>) {
        %four_by_sixteen = arith.addf %a4x16, %b4x16 : tensor<4x16xf32>
        %eight_by_eight = arith.addf %a8x8, %b8x8 : tensor<8x8xf32>
        %sixteen_by_four = arith.addf %a16x4, %b16x4 : tensor<16x4xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("short_axis_geometry", StageCostModelKind::ScalarMath);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  ASSERT_TRUE(function);
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator())
    stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  const auto &workloads =
      partition.stages.front().workload.tensorOperationWorkloads;
  ASSERT_EQ(workloads.size(), 1u);
  EXPECT_EQ(workloads.front().contiguousElementsPerSegment, 64);
  EXPECT_EQ(workloads.front().elementBitWidth, 32);
  EXPECT_DOUBLE_EQ(workloads.front().logicalElements, 192.0);
  EXPECT_DOUBLE_EQ(workloads.front().segmentCount, 3.0);
}

TEST(SimdSimtCostModelTest,
     WorkloadRetainsRowBroadcastAndComparisonInputWidth) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%a: tensor<8x8xf32>, %row: tensor<8x1xf32>,
                        %a3: tensor<2x4x8xf32>, %row3: tensor<2x4x1xf32>) {
        %b = "tt.broadcast"(%row) : (tensor<8x1xf32>) -> tensor<8x8xf32>
        %sum = arith.addf %a, %b : tensor<8x8xf32>
        %mask = arith.cmpf olt, %a, %sum : tensor<8x8xf32>
        %b3 = "tt.broadcast"(%row3) : (tensor<2x4x1xf32>) -> tensor<2x4x8xf32>
        %sum3 = arith.addf %a3, %b3 : tensor<2x4x8xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("broadcast", StageCostModelKind::ScalarMath);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator())
    if (mlir::isa<mlir::arith::AddFOp, mlir::arith::CmpFOp>(operation))
      stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  const auto &workloads =
      partition.stages.front().workload.tensorOperationWorkloads;
  ASSERT_EQ(workloads.size(), 2u);
  for (const auto &workload : workloads) {
    EXPECT_EQ(workload.elementBitWidth, 32);
    if (workload.operation == "f32.add") {
      EXPECT_EQ(workload.contiguousElementsPerSegment, 8);
      EXPECT_DOUBLE_EQ(workload.segmentCount, 16);
    } else {
      EXPECT_EQ(workload.operation, "predicate.cmp");
      EXPECT_EQ(workload.contiguousElementsPerSegment, 64);
      EXPECT_DOUBLE_EQ(workload.segmentCount, 1);
    }
  }
}

TEST(SimdSimtCostModelTest, WorkloadUsesPostFlattenBroadcastSuffix) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%x: tensor<2x4x8xf32>,
                        %leading: tensor<1x4x8xf32>,
                        %middle: tensor<2x1x8xf32>,
                        %x4: tensor<2x4x8x4xf32>,
                        %leading4: tensor<1x4x8x4xf32>,
                        %unit: tensor<2x4x8x1xf32>,
                        %row: tensor<2x4x1x1xf32>,
                        %ix: tensor<2x4x8xi32>, %iy: tensor<2x1x8xi32>) {
        %a = "tt.broadcast"(%leading) : (tensor<1x4x8xf32>) -> tensor<2x4x8xf32>
        %s0 = arith.addf %x, %a : tensor<2x4x8xf32>
        %b = "tt.broadcast"(%middle) : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
        %s1 = arith.addf %x, %b : tensor<2x4x8xf32>
        %c = "tt.broadcast"(%leading4) : (tensor<1x4x8x4xf32>) -> tensor<2x4x8x4xf32>
        %s2 = arith.addf %x4, %c : tensor<2x4x8x4xf32>
        %d = "tt.broadcast"(%row) : (tensor<2x4x1x1xf32>) -> tensor<2x4x8x1xf32>
        %s3 = arith.addf %unit, %d : tensor<2x4x8x1xf32>
        %e = "tt.broadcast"(%iy) : (tensor<2x1x8xi32>) -> tensor<2x4x8xi32>
        %s4 = arith.addi %ix, %e : tensor<2x4x8xi32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator()) {
    if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(operation))
      continue;
    LogicalStage stage =
        logicalStage("flatten", StageCostModelKind::ScalarMath);
    stage.operations.push_back(&operation);
    partition.stages.push_back(std::move(stage));
  }
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  ASSERT_EQ(partition.stages.size(), 5u);
  const int64_t widths[] = {32, 8, 128, 8, 8};
  const double segments[] = {2, 8, 2, 8, 8};
  for (size_t i = 0; i < partition.stages.size(); ++i) {
    const auto &groups = partition.stages[i].workload.tensorOperationWorkloads;
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].contiguousElementsPerSegment, widths[i]);
    EXPECT_DOUBLE_EQ(groups[0].segmentCount, segments[i]);
  }
}

TEST(SimdSimtCostModelTest,
     ScopeSuperBlockLegalityRequiresBackendAndResourceMaximum) {
  auto makePartition = []() {
    StagePartition partition;
    LogicalStage stage =
        logicalStage("payload", StageCostModelKind::LoopCarriedRecurrence,
                     StageScheduleKind::LoopCarriedSerial, /*iterations=*/16);
    stage.features.hasLoop = true;
    stage.features.hasLoopCarriedDataDependency = true;
    stage.localSimtMaterializable = true;
    stage.localSuperblockMaterializable = true;
    stage.localSimtFactors = {1};
    partition.stages.push_back(std::move(stage));
    return partition;
  };

  StagePartition f1Only = makePartition();
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(f1Only, 4, false))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(f1Only.stages[0].localSimtFactors, (std::vector<int64_t>{1}));

  StagePartition scopeSuperblock = makePartition();
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(scopeSuperblock, 4, true, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(scopeSuperblock.stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));

  // ABI-v2 creates an F1 V1 scheduling loop and refines only the selected
  // scope after bufferization, but local and whole-kernel factors still share
  // the same target/runtime warp-resource maximum.
  StagePartition mixedOnly = makePartition();
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(mixedOnly, 1, true, 1))
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
      /*scopeSuperblockMaterializable=*/true,
      /*maximumScopeSuperblockFactor=*/2));
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

  ScopeHandoffCost transition;
  transition.fixedScopeCycles = 12.0;
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
  ASSERT_EQ(result->mixed.scopeHandoffCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(result->mixed.scopeHandoffCycles[1], 12.0);
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
  payload.scopeInputTensorBytes = 4096;
  payload.scopeOutputTensorBytes = 16384;
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 10.0),
                             makeCost(StageMode::SIMT, 10.0, true)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, payload, tail};

  ScopeHandoffCost transition;
  transition.inputHandoffBytesPerCycle = 4096.0 / 48.0;
  transition.outputHandoffBytesPerCycle = 16384.0 / 160.0;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // Input: 4096/256 + 4096/(4*32) = 48 cycles.
  // Output: 16384/(4*32) + 16384/512 = 160 cycles.
  // Head/payload/tail: 10 + (10 + 208) + 10 = 238 cycles.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 238.0);
  ASSERT_EQ(routes->mixed.scopeHandoffCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[1], 208.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[2], 0.0);
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

  ScopeHandoffCost transition;
  transition.fixedScopeCycles = 80.0;
  transition.inputHandoffBytesPerCycle = 4096.0 / 48.0;
  transition.outputHandoffBytesPerCycle = 4096.0 / 40.0;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());

  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  // The 80-cycle fixed transition pair is amortized to 20 cycles.  The
  // 4096-byte input/output handoff remains 48 + 40 cycles per program.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles,
                   10.0 + 100.0 + 20.0 + 48.0 + 40.0 + 10.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[1], 108.0);
}

TEST(SimdSimtCostModelTest, PrefixScanUsesModeSpecificDependencyFactor) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.workload.operationCounts.clear();
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

TEST(SimdSimtCostModelTest, LoopCarriedRecurrenceAppliesScanDependencyFactor) {
  auto buildStage = [] {
    LogicalStage stage = logicalStage(
        "recurrence_scan", StageCostModelKind::LoopCarriedRecurrence,
        StageScheduleKind::LoopCarriedSerial, /*iterations=*/4);
    stage.features.hasLoop = true;
    stage.features.hasLoopCarriedDataDependency = true;
    stage.features.hasPrefixScan = true;
    stage.workload.operationCounts.clear();
    stage.workload.scalarOperations = 0.0;
    stage.workload.issueElements = 64.0;
    // Mixed shuffle pool: 320 scan-class steps (tt.scan) + 320 reduce-class
    // steps (tt.reduce).
    stage.workload.shuffleLaneSteps = 640.0;
    stage.workload.prefixScanCounts[{32, 10, 1, 1, false, "add", "", 0, 0, 0}] =
        1.0;
    return stage;
  };

  // Identity factors keep the legacy recurrence scoring.
  HardwareProfile baselineProfile = hardwareProfile();
  baselineProfile.simd.prefixScanDependencyFactor = 1.0;
  baselineProfile.simt.prefixScanDependencyFactor = 1.0;
  auto baseline = evaluateOneStage(buildStage(), baselineProfile);
  if (!baseline)
    FAIL() << llvm::toString(baseline.takeError());
  ASSERT_EQ(baseline->stages.front().implementations.size(), 2u);

  // Scan factors: SIMD 2.5 / SIMT 1.0 (production profile shape).
  HardwareProfile scanProfile = baselineProfile;
  scanProfile.simd.prefixScanDependencyFactor = 2.5;
  auto scaled = evaluateOneStage(buildStage(), scanProfile);
  if (!scaled)
    FAIL() << llvm::toString(scaled.takeError());
  ASSERT_EQ(scaled->stages.front().implementations.size(), 2u);

  // Only the scan-class half is scaled: SIMD scan-shuffle cycles grow from
  // 320/32 = 10 to 10 * 2.5 = 25, so the stage total grows by 4 iterations *
  // (25 - 10) = 60 cycles.  The reduce-class half keeps the ideal rate and
  // SIMT keeps the identity factor, so both must not move.
  EXPECT_DOUBLE_EQ(scaled->stages.front().implementations[0].totalCycles,
                   baseline->stages.front().implementations[0].totalCycles +
                       60.0);
  EXPECT_DOUBLE_EQ(scaled->stages.front().implementations[1].totalCycles,
                   baseline->stages.front().implementations[1].totalCycles);
  EXPECT_GT(scaled->stages.front().implementations[0].totalCycles,
            scaled->stages.front().implementations[1].totalCycles);

  // A recurrence whose shuffle pool is purely reduce-class must not consume
  // the scan factor even though the scan feature flag is set.
  LogicalStage reduceOnlyStage = buildStage();
  reduceOnlyStage.workload.prefixScanCounts.clear();
  auto reduceOnlyScaled =
      evaluateOneStage(std::move(reduceOnlyStage), scanProfile);
  if (!reduceOnlyScaled)
    FAIL() << llvm::toString(reduceOnlyScaled.takeError());
  EXPECT_DOUBLE_EQ(
      reduceOnlyScaled->stages.front().implementations[0].totalCycles,
      baseline->stages.front().implementations[0].totalCycles);

  // A recurrence without a prefix scan must not consume the scan factor.
  LogicalStage plainStage = buildStage();
  plainStage.features.hasPrefixScan = false;
  auto plainScaled = evaluateOneStage(std::move(plainStage), scanProfile);
  if (!plainScaled)
    FAIL() << llvm::toString(plainScaled.takeError());
  EXPECT_DOUBLE_EQ(plainScaled->stages.front().implementations[0].totalCycles,
                   baseline->stages.front().implementations[0].totalCycles);
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
  stage.workload.operationCounts.clear();
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

TEST(SimdSimtCostModelTest, PhysicalCubeFormulaPricesSupportedShapes) {
  HardwareProfile profile = hardwareProfile();
  profile.simd.dotPhysicalResourceTarget = "Ascend950PR_9579";
  profile.simd.dotPhysicalResourceLoweringContract = "test-mmadL1-tile16";
  StageModeProfile::DotPhysicalResourceFormula formula;
  formula.tinyMacFloorCycles = 55.0;
  formula.cubeMacCyclesPerSqrtFlop = 0.13;
  formula.cubeMinimumFlops = 262144.0;
  formula.mte1StartupCycles = 10.0;
  formula.mte1BytesExponent = 1.0;
  formula.lhsMte1CyclesPerBytePower = 0.001;
  formula.rhsMte1CyclesPerBytePower = 0.002;
  formula.tileSize = 16;
  formula.narrowNReferenceTiles = 2.0;
  formula.wideNReferenceTiles = 4.0;
  formula.narrowNOccupancyExponent = 1.0;
  formula.wideNOccupancyExponent = 0.75;
  formula.evidence = "independent MAC/MTE1 resource fits";
  profile.simd.dotPhysicalResourceFormula = formula;

  LogicalStage stage = logicalStage("cube", StageCostModelKind::CubeRoofline,
                                    StageScheduleKind::IndependentPipelined);
  stage.features.hasDot = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.dotOperations = 1.0;
  stage.workload.dotFlops = 262144.0;
  stage.workload.dotShapeCounts[std::make_tuple(128, 32, 32, 1, "f16", "f16",
                                                "f32", "ieee")] = 1.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &simd = table->stages[0].implementations[0];
  const auto &simt = table->stages[0].implementations[1];
  EXPECT_NEAR(simd.resources.dot, 66.56, 1e-8);
  EXPECT_TRUE(simd.resources.dotPhysicalResourceFormulaApplied);
  EXPECT_NE(simd.formulaEvidence.find("physical_tile_resource_formula"),
            std::string::npos);
  EXPECT_FALSE(simt.resources.dotPhysicalResourceFormulaApplied);

  stage.costModelKind = StageCostModelKind::TinyCubeRoofline;
  stage.workload.dotFlops = 8192.0;
  stage.workload.dotShapeCounts.clear();
  stage.workload.dotShapeCounts[std::make_tuple(16, 16, 16, 1, "f16", "f16",
                                                "f32", "ieee")] = 1.0;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages[0].implementations[0].resources.dot, 55.0);

  stage.costModelKind = StageCostModelKind::CubeRoofline;
  stage.workload.dotFlops = 32768.0;
  stage.workload.dotShapeCounts.clear();
  stage.workload.dotShapeCounts[std::make_tuple(16, 16, 64, 1, "f16", "f16",
                                                "f32", "ieee")] = 1.0;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_FALSE(table->stages[0]
                   .implementations[0]
                   .resources.dotPhysicalResourceFormulaApplied);
  EXPECT_DOUBLE_EQ(table->stages[0].implementations[0].resources.dot, 520.0);
}

TEST(SimdSimtCostModelTest, DependencyDistanceSetsLoopInitiationInterval) {
  const auto buildTable = [](bool loopCarried) {
    StageCostTable table;
    table.profileVersion = "unit-test-profile-v1";
    table.logicalProgramCountHint = 1;
    table.physicalCoreCountHint = 1;
    table.physicalAiCoreCountHint = 1;
    table.stageDependencyScheduleEnabled = true;

    const auto makeCost = [](StageMode mode, double cycles) {
      StageImplementationCost cost;
      cost.implementation = {mode, 1, false};
      cost.totalCycles = cycles;
      return cost;
    };
    const auto implementations = [&](double simdCycles) {
      return std::vector<StageImplementationCost>{
          makeCost(StageMode::SIMD, simdCycles),
          makeCost(StageMode::SIMT, simdCycles + 100.0)};
    };

    LogicalStageCost shell;
    shell.id = "loop";
    shell.model = "loop_carried_recurrence";
    shell.iterationCount = 4;
    shell.implementations = implementations(0.0);

    LogicalStageCost dot;
    dot.id = "dot";
    dot.model = "cube_roofline";
    dot.enclosingLoopStages = {shell.id};
    dot.iterationCount = 4;
    dot.features.hasDot = true;
    dot.implementations = implementations(40.0);

    LogicalStageCost update;
    update.id = "update";
    update.model = "vector_issue";
    update.enclosingLoopStages = {shell.id};
    update.iterationCount = 4;
    update.implementations = implementations(20.0);
    table.stages = {shell, dot, update};

    mlir::ascend::StageDependency sameIteration;
    sameIteration.loopStage = shell.id;
    sameIteration.sourceStage = dot.id;
    sameIteration.targetStage = update.id;
    sameIteration.kind = mlir::ascend::StageDependencyKind::IntraIterationData;
    sameIteration.iterationDistance = 0;
    sameIteration.carriedValueName = "inter_stage_ssa";
    table.dependencies.push_back(sameIteration);

    if (loopCarried) {
      mlir::ascend::StageDependency recurrence;
      recurrence.loopStage = shell.id;
      recurrence.sourceStage = update.id;
      recurrence.targetStage = dot.id;
      recurrence.kind = mlir::ascend::StageDependencyKind::LoopCarriedData;
      recurrence.iterationDistance = 1;
      recurrence.carriedValueName = "state";
      table.dependencies.push_back(recurrence);
    }
    return table;
  };

  auto independent = solveStageRoutes(buildTable(false), ScopeHandoffCost{});
  auto recurrent = solveStageRoutes(buildTable(true), ScopeHandoffCost{});
  if (!independent)
    FAIL() << llvm::toString(independent.takeError());
  if (!recurrent)
    FAIL() << llvm::toString(recurrent.takeError());

  const auto &independentRoute = independent->allSimd;
  const auto &recurrentRoute = recurrent->allSimd;
  EXPECT_TRUE(
      independentRoute.diagnostic<bool>("stage_dependency_schedule_applied"));
  EXPECT_TRUE(
      recurrentRoute.diagnostic<bool>("stage_dependency_schedule_applied"));
  EXPECT_EQ(independentRoute.diagnostic<std::string>(
                "stage_dependency_schedule_status"),
            "symbolic_affine");
  EXPECT_EQ(recurrentRoute.diagnostic<std::string>(
                "stage_dependency_schedule_solver"),
            "periodic_difference_constraints_v1");
  EXPECT_DOUBLE_EQ(independentRoute.diagnostic<double>(
                       "stage_dependency_schedule_initiation_interval_system_"
                       "cycles"),
                   10.0);
  EXPECT_DOUBLE_EQ(recurrentRoute.diagnostic<double>(
                       "stage_dependency_schedule_initiation_interval_system_"
                       "cycles"),
                   15.0);
  EXPECT_DOUBLE_EQ(independentRoute.totalCycles, 45.0);
  EXPECT_DOUBLE_EQ(recurrentRoute.totalCycles, 60.0);
}

TEST(SimdSimtCostModelTest,
     DependencyScheduleFallsBackForUnsupportedRecurrences) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 1;
  table.physicalCoreCountHint = 1;
  table.physicalAiCoreCountHint = 1;
  table.stageDependencyScheduleEnabled = true;

  const auto makeCost = [](StageMode mode, double cycles) {
    StageImplementationCost cost;
    cost.implementation = {mode, 1, false};
    cost.totalCycles = cycles;
    return cost;
  };
  const auto implementations = [&](double simdCycles) {
    return std::vector<StageImplementationCost>{
        makeCost(StageMode::SIMD, simdCycles),
        makeCost(StageMode::SIMT, simdCycles + 100.0)};
  };

  for (int64_t index = 0; index < 2; ++index) {
    LogicalStageCost shell;
    shell.id = "loop" + std::to_string(index);
    shell.model = "loop_carried_recurrence";
    shell.iterationCount = 4;
    shell.implementations = implementations(0.0);
    LogicalStageCost first;
    first.id = "first" + std::to_string(index);
    first.model = "vector_issue";
    first.enclosingLoopStages = {shell.id};
    first.iterationCount = 4;
    first.implementations = implementations(index == 0 ? 20.0 : 10.0);
    LogicalStageCost second = first;
    second.id = "second" + std::to_string(index);
    mlir::ascend::StageDependency recurrence;
    recurrence.loopStage = shell.id;
    recurrence.sourceStage = second.id;
    recurrence.targetStage = first.id;
    recurrence.kind = mlir::ascend::StageDependencyKind::LoopCarriedData;
    recurrence.iterationDistance = 1;
    recurrence.carriedValueName = "state" + std::to_string(index);
    mlir::ascend::StageDependency sameIteration = recurrence;
    sameIteration.sourceStage = first.id;
    sameIteration.targetStage = second.id;
    sameIteration.kind = mlir::ascend::StageDependencyKind::IntraIterationData;
    sameIteration.iterationDistance = 0;
    sameIteration.carriedValueName = "intra" + std::to_string(index);
    table.stages.push_back(std::move(shell));
    table.stages.push_back(std::move(first));
    table.stages.push_back(std::move(second));
    table.dependencies.push_back(std::move(sameIteration));
    table.dependencies.push_back(std::move(recurrence));
  }

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &route = routes->allSimd;
  EXPECT_FALSE(route.diagnostic<bool>("stage_dependency_schedule_applied"));
  EXPECT_EQ(route.diagnostic<std::string>("stage_dependency_schedule_status"),
            "multiple_or_nested_recurrences_unsupported");
  EXPECT_EQ(route.diagnostic<std::string>("stage_dependency_schedule_fallback"),
            "existing_route_cost");
  EXPECT_DOUBLE_EQ(route.totalCycles, 60.0);
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
  stage.workload.privateStackLoadInstructions = 7.0;

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
     SuperBlockRecurrenceContentionStartsAbovePressureFreeFactor) {
  LogicalStage stage = logicalStage("stateful_recurrence",
                                    StageCostModelKind::LoopCarriedRecurrence,
                                    StageScheduleKind::LoopCarriedSerial, 16);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.liveOutBytes = 4096;
  stage.workload.loadWarpInstructions = 16.0;
  stage.workload.shuffleLaneSteps = 128.0;
  stage.workload.issueElements = 64.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_LE(costs[1].totalCycles, costs[0].totalCycles);
  EXPECT_LE(costs[2].totalCycles, costs[1].totalCycles);
  EXPECT_GE(costs[2].totalCycles,
            costs[2].resources.setup + 16.0 * costs[2].resources.issue);

  stage.workload.privateStackLoadInstructions = 32.0;
  auto spillingTable = evaluateOneStage(stage);
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
  stage.workload.indirectLoadCounts[{1024, 256}] = 1.0;

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
  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
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
  gather.implementations = {makeCost(StageMode::SIMD, 100.0),
                            makeCost(StageMode::SIMT, 130.0),
                            makeCost(StageMode::SIMT, 130.0, true)};
  mlir::ascend::LogicalStageCost dot;
  dot.id = "tiny_cube_dot";
  dot.implementations = {makeCost(StageMode::SIMD, 40.0),
                         makeCost(StageMode::SIMT, 90.0)};
  table.stages = {gather, dot};

  ScopeHandoffCost transition;
  transition.fixedScopeCycles = 20.0;
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

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
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

TEST(SimdSimtCostModelTest, AllSimdPreservesSetupOnConditionalSchedule) {
  for (const auto *model : {"auto_blockify_dispatch", "auto_blockify_loop"}) {
    for (int64_t programs : {1, 5}) {
      StageCostTable table;
      table.profileVersion = "unit-test-profile-v1";
      table.logicalProgramCountHint = programs;
      table.physicalCoreCountHint = 2;
      auto makeCost = [](StageMode mode, double total, double setup) {
        mlir::ascend::StageImplementationCost cost;
        cost.implementation = {mode, 1, false};
        cost.totalCycles = total;
        cost.resources.setup = setup;
        return cost;
      };
      mlir::ascend::LogicalStageCost dispatch;
      dispatch.id = "dispatch";
      dispatch.model = model;
      dispatch.implementations = {makeCost(StageMode::SIMD, 40.0, 20.0),
                                  makeCost(StageMode::SIMT, 30.0, 25.0)};
      mlir::ascend::LogicalStageCost payload;
      payload.id = "payload";
      payload.model = "scalar_issue";
      payload.implementations = {makeCost(StageMode::SIMD, 100.0, 0.0),
                                 makeCost(StageMode::SIMT, 80.0, 0.0)};
      table.stages = {dispatch, payload};
      auto routes = solveStageRoutes(table, ScopeHandoffCost{});
      if (!routes)
        FAIL() << llvm::toString(routes.takeError());
      const double waves = (programs + 1) / 2;
      ASSERT_TRUE(routes->allSimd.legal);
      EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[0], 20.0 * waves);
      EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[1], 100.0 * waves);
      EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 120.0 * waves);
      ASSERT_TRUE(routes->allSimt.legal);
      EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 110.0 * waves);

      // Moving setup to the payload must not change the all-SIMD total.
      table.stages[0].implementations[0].resources.setup = 0.0;
      table.stages[0].implementations[0].totalCycles -= 20.0;
      table.stages[1].implementations[0].resources.setup = 20.0;
      table.stages[1].implementations[0].totalCycles += 20.0;
      auto moved = solveStageRoutes(table, ScopeHandoffCost{});
      if (!moved)
        FAIL() << llvm::toString(moved.takeError());
      EXPECT_DOUBLE_EQ(moved->allSimd.totalCycles, routes->allSimd.totalCycles);
    }
  }
}

TEST(SimdSimtCostModelTest,
     GridStrideBoundaryOwnsFirstIndirectServiceForEveryTripCount) {
  for (bool hasDot : {false, true}) {
    for (int64_t programs : {56, 112}) {
      StageCostTable table;
      table.profileVersion = "unit-test-profile-v1";
      table.logicalProgramCountHint = programs;
      table.physicalCoreCountHint = 56;

      auto makeCost = [](double cycles) {
        StageImplementationCost cost;
        cost.implementation = {StageMode::SIMT, 1, false};
        cost.totalCycles = cycles;
        return cost;
      };

      LogicalStageCost shell;
      shell.id = "grid_stride_shell";
      shell.model = "auto_blockify_loop";
      shell.features.isGridStrideLoopShell = true;
      shell.implementations = {makeCost(0.0)};

      LogicalStageCost gather;
      gather.id = "grid_stride_gather";
      gather.features.insideGridStrideLoopBody = true;
      gather.implementations = {makeCost(100.0)};
      gather.implementations.front().resources.indirectLoadGroup = 100.0;
      gather.implementations.front().resources.gridStrideStartup = 120.0;

      table.stages = {shell, gather};
      if (hasDot) {
        LogicalStageCost dot;
        dot.id = "grid_stride_dot";
        dot.features.insideGridStrideLoopBody = true;
        dot.features.hasDot = true;
        dot.implementations = {makeCost(10.0)};
        table.stages.push_back(std::move(dot));
      }

      auto routes = solveStageRoutes(table, ScopeHandoffCost{});
      if (!routes)
        FAIL() << llvm::toString(routes.takeError());
      ASSERT_TRUE(routes->allSimt.legal);
      const double trips = programs / 56;
      const double expected =
          120.0 + (trips - 1.0) * 100.0 + (hasDot ? trips * 10.0 : 0.0);
      EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, expected);
    }
  }
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
  gather.implementations = {makeCost(StageMode::SIMD, 100.0),
                            makeCost(StageMode::SIMT, 1.0),
                            makeCost(StageMode::SIMT, 1.0, true)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, gather, tail};

  ScopeHandoffCost transition;
  transition.fixedScopeCycles = 20.0;
  auto routes = solveStageRoutes(table, transition);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // 1 SIMD head + (10 enter + 1 payload + 20 extra scope pair) +
  // (10 leave + 1 SIMD tail).
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 43.0);
  ASSERT_EQ(routes->mixed.scopeHandoffCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[1], 40.0);
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
  auto routes = solveStageRoutes(*table, cappedProfile.scopeHandoff);
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

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
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

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
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
  first.features.insideAutoBlockifyV1Loop = true;
  first.localSimtMaterializable = true;
  first.implementations = {makeCost(StageMode::SIMD, 1, 100.0),
                           makeCost(StageMode::SIMT, 4, 1.0, true)};
  LogicalStageCost second = first;
  second.id = "second_local_candidate";
  LogicalStageCost tail;
  tail.id = "simd_tail";
  tail.implementations = {makeCost(StageMode::SIMD, 1, 1.0)};
  table.stages = {first, second, tail};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
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
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[1], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[2], 0.0);
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
  auto add = recurrenceStage->workload.operationCounts.find("f32.add");
  ASSERT_NE(add, recurrenceStage->workload.operationCounts.end());
  EXPECT_DOUBLE_EQ(add->second.at(256), 1.0);
}

TEST(SimdSimtCostModelTest,
     SourceLocationsDoNotChangeStructuralStagePartitioning) {
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
  auto withSourceLocations = mlir::ascend::StageBoundaryAnalysis().analyze(
      structure, mlir::ascend::SimtAnchorPlan{});
  if (!withSourceLocations)
    FAIL() << llvm::toString(withSourceLocations.takeError());

  for (mlir::Operation *operation : roots)
    operation->setLoc(mlir::UnknownLoc::get(&context));
  auto withoutSourceLocations = mlir::ascend::StageBoundaryAnalysis().analyze(
      structure, mlir::ascend::SimtAnchorPlan{});
  if (!withoutSourceLocations)
    FAIL() << llvm::toString(withoutSourceLocations.takeError());

  ASSERT_EQ(withSourceLocations->stages.size(), 2u);
  ASSERT_EQ(withoutSourceLocations->stages.size(),
            withSourceLocations->stages.size());
  for (size_t index = 0; index < withSourceLocations->stages.size(); ++index) {
    const LogicalStage &withSource = withSourceLocations->stages[index];
    const LogicalStage &withoutSource = withoutSourceLocations->stages[index];
    EXPECT_EQ(withoutSource.operations.size(), withSource.operations.size());
    EXPECT_EQ(withoutSource.costModelKind, withSource.costModelKind);
    EXPECT_EQ(withoutSource.scheduleKind, withSource.scheduleKind);
  }

  EXPECT_EQ(withSourceLocations->stages.front().operations.size(), 3u);
  EXPECT_EQ(withSourceLocations->stages.front().costModelKind,
            StageCostModelKind::ScalarLoad);
  ASSERT_EQ(withSourceLocations->stages.back().operations.size(), 1u);
  EXPECT_EQ(withSourceLocations->stages.back().operations.front(),
            roots.back());
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

TEST(SimdSimtCostModelTest, ClippedDynamicLoopUsesStaticTripCountCap) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  for (bool capped : {false, true}) {
    std::string source = R"mlir(
      module {
        func.func @kernel(%limit: i32, %initial: i32) -> i32 {
          %c1 = arith.constant 1 : i32
          %c2 = arith.constant 2 : i32
          %c16 = arith.constant 16 : i32
          %cap = arith.minsi %limit, %c16 : i32
          %result = scf.for %i = %c2 to UPPER step %c1
              iter_args(%state = %initial) -> (i32) : i32 {
            %next = arith.addi %state, %i : i32
            scf.yield %next : i32
          }
          %second = scf.for %j = %c2 to %cap step %c1
              iter_args(%state = %result) -> (i32) : i32 {
            %next = arith.addi %state, %j : i32
            scf.yield %next : i32
          }
          return %second : i32
        }
      }
    )mlir";
    source.replace(source.find("UPPER"), 5, capped ? "%cap" : "%limit");
    if (!capped)
      source.replace(source.find("to %cap"), 7, "to %limit");
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    auto tripCounts = analyzeLoopTripCountInfos(*module);
    ASSERT_EQ(tripCounts.size(), 2u);
    for (const auto &tripCount : tripCounts) {
      EXPECT_EQ(tripCount.upperBoundTripCount,
                capped ? std::optional<int64_t>(14) : std::nullopt);
      EXPECT_EQ(tripCount.workloadTripCount, capped ? 14 : 1);
      EXPECT_EQ(tripCount.workloadUsesUpperBound, capped);
      EXPECT_EQ(tripCount.workloadUsesFallback, !capped);
    }
    mlir::ascend::SimtAnchorPlan anchors;
    auto partition = StagePartitioner().partition(*module, anchors,
                                                  StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    if (llvm::Error error = StageFeatureAnalysis().analyze(*partition))
      FAIL() << llvm::toString(std::move(error));
    int64_t maximumIterations = 1;
    for (const auto &stage : partition->stages) {
      maximumIterations = std::max(maximumIterations, stage.iterationCount);
      EXPECT_EQ(stage.features.parallelRecurrenceGroupCount, 1);
    }
    EXPECT_EQ(maximumIterations, capped ? 14 : 1);
  }
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

  auto tripCounts = analyzeLoopTripCountInfos(*module);
  ASSERT_EQ(tripCounts.size(), 2u);
  for (const auto &tripCount : tripCounts) {
    ASSERT_TRUE(tripCount.constantTripCount);
    EXPECT_EQ(*tripCount.constantTripCount, 4);
    EXPECT_EQ(tripCount.workloadTripCount, 4);
    EXPECT_FALSE(tripCount.workloadUsesFallback);
  }

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

// Ordinary tail reduction calibration must not replace measured Stage bodies.
#include "AscendModel/RouteModel/Reduction/ReductionCostModel.h"

namespace {
template <typename T> T takeReductionResult(llvm::Expected<T> result) {
  if (!result) {
    ADD_FAILURE() << llvm::toString(result.takeError());
    return T{};
  }
  return std::move(*result);
}
HardwareProfile tailMigrationProfile() {
  auto profile = hardwareProfile();
  profile.simd.reductionLaneStepsPerCycle = 32.0;
  profile.simt.reductionLaneStepsPerCycle = 32.0;
  auto path = std::string(TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH);
  path = path.substr(0, path.find_last_of('/')) +
         "/../reduction/david_v100_reduction_v1.json";
  auto reduction = mlir::ascend::ReductionProfile::loadFromFile(path);
  if (!reduction) {
    ADD_FAILURE() << llvm::toString(reduction.takeError());
    return profile;
  }
  if (auto error = reduction->validateAllFunctions()) {
    ADD_FAILURE() << llvm::toString(std::move(error));
    return profile;
  }
  profile.reductionProfile = std::move(*reduction);
  return profile;
}
LogicalStage ordinaryTailStage() {
  auto stage =
      logicalStage("ordinary_tail", StageCostModelKind::RowwiseReduction);
  stage.features.hasReduction = true;
  stage.workload.reductionOperations = 1;
  stage.workload.shuffleLaneSteps = 64 * 6;
  stage.workload.reductionGeometryCounts[{1, 64, 32}] = 1;
  stage.workload.reductionWorkloads.push_back({"sum", "f32", {64}, 0, 1});
  mlir::ascend::StageMemoryAccess input;
  input.shape = {64};
  input.strides = {1};
  input.elementBits = 32;
  stage.workload.memoryAccesses.push_back(input);
  stage.reductionExternalWorkload = stage.workload;
  return stage;
}
} // namespace

TEST(SimdSimtCostModelTest, OrdinaryTailUsesFrozenFormulaAndRecomputesPath) {
  auto profile = tailMigrationProfile();
  auto stage = ordinaryTailStage();
  auto expected = mlir::ascend::estimateTailAxisReductionCycles(
      stage.workload.reductionWorkloads.front(),
      profile.reductionProfile->getModeProfile(
          mlir::ascend::ReductionMode::SIMD),
      mlir::ascend::ReductionMode::SIMD, 4);
  ASSERT_TRUE(expected);
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table))
      << (table ? "" : llvm::toString(table.takeError()));
  bool found = false;
  for (const auto &cost : table->stages.front().implementations) {
    if (cost.implementation.mode != StageMode::SIMD)
      continue;
    found = true;
    EXPECT_TRUE(cost.calibratedTailReduction);
    EXPECT_DOUBLE_EQ(cost.resources.reduction, *expected);
    EXPECT_DOUBLE_EQ(cost.resources.criticalPath,
                     cost.resources.compute + cost.resources.predicate +
                         cost.resources.reduction + cost.resources.shuffle);
  }
  EXPECT_TRUE(found);
}

TEST(SimdSimtCostModelTest, TailCalibrationPreservesCandidateExecutionDomain) {
  auto profile = tailMigrationProfile();
  auto stage = ordinaryTailStage();
  stage.legalSimtFactors = {1, 2, 4};
  stage.localSimtMaterializable = true;
  stage.localSuperblockMaterializable = true;
  stage.localSimtFactors = {1, 2, 4};
  auto expected = mlir::ascend::estimateTailAxisReductionCycles(
      stage.workload.reductionWorkloads.front(),
      profile.reductionProfile->getModeProfile(
          mlir::ascend::ReductionMode::SIMT),
      mlir::ascend::ReductionMode::SIMT, profile.baseSimtWarpCount);
  ASSERT_TRUE(expected);
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  unsigned whole = 0, local = 0;
  for (const auto &cost : table->stages.front().implementations) {
    if (cost.implementation.mode != StageMode::SIMT)
      continue;
    EXPECT_TRUE(cost.calibratedTailReduction);
    // The window is per logical program: F is composed later, not folded
    // into the formula's numWarps or multiplied into reduction twice.
    EXPECT_DOUBLE_EQ(cost.resources.reduction, *expected);
    EXPECT_GT(cost.totalCycles, 0.0);
    if (cost.implementation.localScope)
      ++local;
    else
      ++whole;
  }
  EXPECT_EQ(whole, 3u);
  EXPECT_EQ(local, 3u);
}

TEST(SimdSimtCostModelTest,
     TailUnsupportedFallsBackAndStridedInputStillCalibrates) {
  for (bool unsupported : {false, true}) {
    auto stage = ordinaryTailStage();
    if (unsupported)
      stage.workload.reductionWorkloads.front().kind = "custom";
    else
      stage.workload.memoryAccesses.front().strides = {2};
    auto profile = tailMigrationProfile();
    auto changed = evaluateOneStage(stage, profile);
    profile.reductionProfile.reset();
    auto baseline = evaluateOneStage(stage, profile);
    ASSERT_TRUE(static_cast<bool>(changed));
    ASSERT_TRUE(static_cast<bool>(baseline));
    ASSERT_EQ(changed->stages.front().implementations.size(),
              baseline->stages.front().implementations.size());
    for (size_t i = 0; i < changed->stages.front().implementations.size();
         ++i) {
      if (!unsupported) {
        EXPECT_TRUE(
            changed->stages.front().implementations[i].calibratedTailReduction);
        continue;
      }
      EXPECT_FALSE(
          changed->stages.front().implementations[i].calibratedTailReduction);
      EXPECT_DOUBLE_EQ(changed->stages.front().implementations[i].totalCycles,
                       baseline->stages.front().implementations[i].totalCycles);
    }
  }
}

TEST(SimdSimtCostModelTest,
     SyntheticStageWithoutContributionsPreservesAllOrNothingFallback) {
  auto stage = ordinaryTailStage();
  stage.workload.reductionWorkloads.push_back({"custom", "f32", {64}, 0, 1});
  stage.workload.reductionOperations = 2;
  auto table = evaluateOneStage(stage, tailMigrationProfile());
  ASSERT_TRUE(static_cast<bool>(table));
  for (const auto &cost : table->stages.front().implementations)
    EXPECT_FALSE(cost.calibratedTailReduction);
}

TEST(SimdSimtCostModelTest, PerReductionMixedAxesAndPartialFallback) {
  auto profile = tailMigrationProfile();
  profile.baseSimtWarpCount = 4;
  auto stage = ordinaryTailStage();
  stage.workload = {};
  stage.reductionExternalWorkload = StageWorkload{};
  std::vector<mlir::ascend::ReductionWorkload> reductions = {
      {"sum", "f32", {64}, 0, 1},
      {"sum", "f32", {8, 64}, 0, 2},
      {"max", "f32", {2, 32, 32}, 1, 1},
      {"unknown", "f32", {16}, 0, 3}};
  double fitted = 0;
  for (const auto &work : reductions) {
    StageWorkload contribution;
    contribution.reductionWorkloads.push_back(work);
    contribution.reductionOperations = work.instances;
    double elements = 1;
    for (auto d : work.shape)
      elements *= d;
    contribution.shuffleLaneSteps =
        work.instances * elements * std::ceil(std::log2(work.shape[work.axis]));
    contribution.reductionGeometryCounts[{
        static_cast<int64_t>(elements / work.shape[work.axis]),
        work.shape[work.axis], 32}] = work.instances;
    stage.reductionContributions.push_back(contribution);
    mlir::ascend::mergeReductionResidualWorkload(stage.workload, contribution);
    auto estimate = takeReductionResult(mlir::ascend::estimateReductionCycles(
        work, *profile.reductionProfile, mlir::ascend::ReductionMode::SIMD,
        profile.baseSimtWarpCount));
    if (estimate)
      fitted += work.instances * estimate->cycles;
  }
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  partition.stages.push_back(stage);
  auto table = StageCostEvaluator().evaluate(partition, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_TRUE(cost.calibratedFirstReduction);
  EXPECT_TRUE(cost.calibratedTailReduction);
  EXPECT_TRUE(cost.calibratedMiddleReduction);
  EXPECT_EQ(cost.reductionFallbackInstances, 3);
  EXPECT_GE(cost.resources.reduction, fitted);
  auto oldProfile = profile;
  oldProfile.reductionProfile.reset();
  auto allLegacy = StageCostEvaluator().evaluate(partition, oldProfile);
  ASSERT_TRUE(static_cast<bool>(allLegacy));
  EXPECT_FALSE(allLegacy->stages.front()
                   .implementations.front()
                   .calibratedTailReduction);
}

TEST(SimdSimtCostModelTest, PartialReductionPreservesOnlyUnmatchedCombine) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
 module {
  func.func @kernel(%a: tensor<64xf32>) {
   %x = "tt.reduce"(%a) ({
    ^bb0(%l:f32,%r:f32):
     %v=arith.addf %l,%r:f32
     "tt.reduce.return"(%v):(f32)->()
   }) {axis=0:i32} : (tensor<64xf32>)->f32
   %y = "tt.reduce"(%a) ({
    ^bb0(%l:f32,%r:f32):
     %v=arith.addf %l,%r:f32
     %w=arith.mulf %v,%r:f32
     "tt.reduce.return"(%w):(f32)->()
   }) {axis=0:i32} : (tensor<64xf32>)->f32
   return
  }
 })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto stage = logicalStage("partial", StageCostModelKind::RowwiseReduction);
  stage.features.hasReduction = true;
  auto fn = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  for (auto &op : fn.getBody().front().without_terminator())
    stage.operations.push_back(&op);
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  partition.stages.push_back(stage);
  if (auto err = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(err));
  auto &analyzed = partition.stages.front();
  ASSERT_EQ(analyzed.reductionContributions.size(), 2u);
  EXPECT_EQ(analyzed.reductionExternalWorkload->reductionOperations, 0);
  EXPECT_TRUE(analyzed.reductionExternalWorkload->operationCounts.empty());
  auto profile = tailMigrationProfile();
  auto got = StageCostEvaluator().evaluate(partition, profile);
  ASSERT_TRUE(static_cast<bool>(got));
  StagePartition residual = partition;
  residual.stages.front().workload = analyzed.reductionContributions[1];
  residual.stages.front().reductionContributions.clear();
  residual.stages.front().reductionExternalWorkload.reset();
  auto plainProfile = profile;
  plainProfile.reductionProfile.reset();
  auto baseline = StageCostEvaluator().evaluate(residual, plainProfile);
  ASSERT_TRUE(static_cast<bool>(baseline));
  auto &cost = got->stages.front().implementations.front();
  auto &old = baseline->stages.front().implementations.front();
  EXPECT_TRUE(cost.calibratedTailReduction);
  EXPECT_EQ(cost.reductionFallbackInstances, 1);
  EXPECT_DOUBLE_EQ(cost.resources.compute, old.resources.compute);
  EXPECT_DOUBLE_EQ(cost.resources.issue, old.resources.issue);
  auto estimate = takeReductionResult(mlir::ascend::estimateReductionCycles(
      analyzed.reductionContributions[0].reductionWorkloads.front(),
      *profile.reductionProfile, mlir::ascend::ReductionMode::SIMD,
      profile.baseSimtWarpCount));
  ASSERT_TRUE(estimate);
  EXPECT_NEAR(cost.resources.reduction,
              old.resources.reduction + estimate->cycles, 1e-9);
}

TEST(SimdSimtCostModelTest, ReductionProfileHierarchyAndLegacyAreEquivalent) {
  using namespace mlir::ascend;
  const char *first =
      R"({"supported_warps":[4],"segments":[{"id":"f","lane_width":1,"condition":"1","expression":"42"}]})";
  const char *middle =
      R"({"layout_policy_id":"row_major_128bit_v1","supported_warps":[4],"segments":[{"id":"m","lane_width":1,"condition":"1","expression":"43"}]})";
  auto makeRecord = [](const char *text, llvm::StringRef family) {
    auto parsed = llvm::json::parse(text);
    if (!parsed) {
      ADD_FAILURE() << llvm::toString(parsed.takeError());
      return llvm::json::Object{};
    }
    auto record = *parsed->getAsObject();
    record["family"] = family.str();
    record["mode"] = "simd";
    record["op"] = "sum";
    record["dtype"] = "f32";
    return record;
  };
  for (bool hierarchical : {false, true}) {
    llvm::json::Object families{
        {"tail_axis", llvm::json::Object{{"simd", llvm::json::Object{}},
                                         {"simt", llvm::json::Object{}}}}};
    for (const auto &entry : {std::make_pair("first_axis", first),
                              std::make_pair("middle_axis", middle)}) {
      if (hierarchical) {
        auto parsed = llvm::json::parse(entry.second);
        ASSERT_TRUE(static_cast<bool>(parsed))
            << llvm::toString(parsed.takeError());
        families[entry.first] = llvm::json::Object{
            {"simd",
             llvm::json::Object{
                 {"sum", llvm::json::Object{{"f32", std::move(*parsed)}}}}}};
      } else {
        families[entry.first] =
            llvm::json::Array{makeRecord(entry.second, entry.first)};
      }
    }
    llvm::json::Object root{{"schema_version", 1},
                            {"target", "test"},
                            {"profile_version", "test"},
                            {"unit", "syscnt_cycle"},
                            {"families", std::move(families)}};
    int fd;
    llvm::SmallString<128> path;
    ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("reduction-hierarchy",
                                                    "json", fd, path));
    {
      llvm::raw_fd_ostream stream(fd, true);
      stream << llvm::json::Value(std::move(root));
    }
    auto profile = ReductionProfile::loadFromFile(path);
    llvm::sys::fs::remove(path);
    ASSERT_TRUE(static_cast<bool>(profile))
        << llvm::toString(profile.takeError());
    ReductionWorkload work;
    work.kind = "sum";
    work.dataType = "f32";
    work.shape = {8, 16};
    work.axis = 0;
    auto cost = takeReductionResult(estimateFirstAxisReductionCycles(
        work, *profile, ReductionMode::SIMD, 4));
    ASSERT_TRUE(cost);
    EXPECT_DOUBLE_EQ(*cost, 42);
    work.shape = {2, 8, 16};
    work.axis = 1;
    cost = takeReductionResult(estimateMiddleAxisReductionCycles(
        work, *profile, ReductionMode::SIMD, 4));
    ASSERT_TRUE(cost);
    EXPECT_DOUBLE_EQ(*cost, 43);
  }
}

TEST(SimdSimtCostModelTest, StringReductionExpressions) {
  using namespace mlir::ascend;
  auto load = [](llvm::json::Value expression,
                 llvm::StringRef condition = "1") {
    llvm::json::Object segment{{"id", "test"},
                               {"lane_width", 1},
                               {"condition", condition.str()},
                               {"expression", std::move(expression)}};
    llvm::json::Object function{
        {"supported_warps", llvm::json::Array{4}},
        {"segments", llvm::json::Array{std::move(segment)}}};
    llvm::json::Object root{
        {"schema_version", 1},
        {"target", "test"},
        {"profile_version", "test"},
        {"unit", "syscnt_cycle"},
        {"families",
         llvm::json::Object{
             {"tail_axis", llvm::json::Object{{"simd", llvm::json::Object{}},
                                              {"simt", llvm::json::Object{}}}},
             {"first_axis",
              llvm::json::Object{
                  {"simd",
                   llvm::json::Object{
                       {"sum", llvm::json::Object{
                                   {"f32", std::move(function)}}}}}}}}}};
    int fd;
    llvm::SmallString<128> path;
    if (auto ec = llvm::sys::fs::createTemporaryFile("reduction-string", "json",
                                                     fd, path))
      return llvm::Expected<ReductionProfile>(llvm::errorCodeToError(ec));
    {
      llvm::raw_fd_ostream stream(fd, true);
      stream << llvm::json::Value(std::move(root));
    }
    auto result = ReductionProfile::loadFromFile(path);
    llvm::sys::fs::remove(path);
    if (result)
      if (auto error = result->validateAllFunctions())
        return llvm::Expected<ReductionProfile>(std::move(error));
    return result;
  };
  for (auto entry :
       {std::make_pair("2 + 3 * 4", 14.), std::make_pair("(2 + 3) * 4", 20.),
        std::make_pair("2 ** 3", 8.), std::make_pair("pow(2, -2) + 1e-3", .251),
        std::make_pair("max(ceil(1.2), sqrt(9), log2(16))", 4.),
        std::make_pair("neg(-2) + indicator(R >= 8)", 3.)}) {
    auto p = load(entry.first,
                  "all_of(R >= 8, any_of(G == 16, logical_not(W == 4)))");
    ASSERT_TRUE(static_cast<bool>(p)) << llvm::toString(p.takeError());
    ReductionWorkload work;
    work.kind = "sum";
    work.dataType = "f32";
    work.shape = {8, 16};
    work.axis = 0;
    auto value = takeReductionResult(
        estimateFirstAxisReductionCycles(work, *p, ReductionMode::SIMD, 4));
    ASSERT_TRUE(value);
    EXPECT_NEAR(*value, entry.second, 1e-12);
  }
  for (auto text :
       {"", "R +", "unknown", "sqrt()", "pow(R, G)", "pow(R, 5)", "1e999", "(R",
        "R)", "1;2", "max(1,)", "1..2", "sqrt(1,2)"}) {
    auto p = load(text);
    EXPECT_FALSE(static_cast<bool>(p)) << text;
    if (!p)
      llvm::consumeError(p.takeError());
  }
  for (auto text : {std::string(16385, '1'),
                    std::string(40, '(') + "1" + std::string(40, ')')}) {
    auto p = load(text);
    EXPECT_FALSE(static_cast<bool>(p));
    if (!p)
      llvm::consumeError(p.takeError());
  }
  auto old = load(llvm::json::Object{{"constant", 42}});
  EXPECT_FALSE(static_cast<bool>(old));
  if (!old)
    llvm::consumeError(old.takeError());
}

namespace {
llvm::Expected<mlir::ascend::ReductionProfile>
loadLazyTestProfile(llvm::StringRef usedExpression = "42",
                    llvm::StringRef unusedExpression = "43") {
  auto function = [](llvm::StringRef expression, bool middle) {
    llvm::json::Object result{
        {"supported_warps", llvm::json::Array{4}},
        {"segments", llvm::json::Array{llvm::json::Object{
                         {"id", "segment"},
                         {"lane_width", 1},
                         {"condition", "1"},
                         {"expression", expression.str()}}}}};
    if (middle)
      result["layout_policy_id"] = "row_major_128bit_v1";
    return result;
  };
  llvm::json::Object root{
      {"schema_version", 1},
      {"target", "test"},
      {"profile_version", "test"},
      {"unit", "syscnt_cycle"},
      {"families",
       llvm::json::Object{
           {"tail_axis", llvm::json::Object{{"simd", llvm::json::Object{}},
                                            {"simt", llvm::json::Object{}}}},
           {"first_axis",
            llvm::json::Object{
                {"simd",
                 llvm::json::Object{
                     {"sum", llvm::json::Object{{"f32", function(usedExpression,
                                                                 false)}}}}}}},
           {"middle_axis",
            llvm::json::Object{
                {"simt", llvm::json::Object{
                             {"sum", llvm::json::Object{
                                         {"f32", function(unusedExpression,
                                                          true)}}}}}}}}}};
  int fd;
  llvm::SmallString<128> path;
  if (auto ec = llvm::sys::fs::createTemporaryFile("lazy-reduction", "json", fd,
                                                   path))
    return llvm::errorCodeToError(ec);
  {
    llvm::raw_fd_ostream stream(fd, true);
    stream << llvm::json::Value(std::move(root));
  }
  auto result = mlir::ascend::ReductionProfile::loadFromFile(path);
  llvm::sys::fs::remove(path);
  return result;
}
} // namespace

TEST(SimdSimtCostModelTest, LazyReductionParsesOnlyRequestedGroups) {
  using namespace mlir::ascend;
  auto p = loadLazyTestProfile();
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  EXPECT_EQ(p->getParsedFunctionCount(), 0u);
  // A valid tail miss and a missing group must not initialize any AST.
  auto tail = estimateReductionCycles({"sum", "f32", {64}, 0, 1}, *p,
                                      ReductionMode::SIMD, 4);
  ASSERT_TRUE(bool(tail));
  EXPECT_FALSE(bool(*tail));
  auto missing = p->findFirstAxis(ReductionMode::SIMD, "prod", "i64");
  ASSERT_TRUE(bool(missing));
  EXPECT_EQ(*missing, nullptr);
  EXPECT_EQ(p->getParsedFunctionCount(), 0u);
  auto first = p->findFirstAxis(ReductionMode::SIMD, "sum", "f32");
  ASSERT_TRUE(bool(first));
  ASSERT_NE(*first, nullptr);
  auto copy = *p;
  auto moved = std::move(copy);
  auto again = moved.findFirstAxis(ReductionMode::SIMD, "sum", "f32");
  ASSERT_TRUE(bool(again));
  EXPECT_EQ(*first, *again);
  EXPECT_EQ(p->getParsedFunctionCount(), 1u);
  auto middle = moved.findMiddleAxis(ReductionMode::SIMT, "sum", "f32");
  ASSERT_TRUE(bool(middle));
  ASSERT_NE(*middle, nullptr);
  EXPECT_EQ(p->getParsedFunctionCount(), 2u);
  auto error = p->validateAllFunctions();
  EXPECT_FALSE(bool(error));
  if (error)
    llvm::consumeError(std::move(error));
}

TEST(SimdSimtCostModelTest, LazyReductionErrorsAreDeferredAndPropagated) {
  using namespace mlir::ascend;
  auto p = loadLazyTestProfile("42", "unknown + 1");
  ASSERT_TRUE(bool(p));
  auto valid = estimateFirstAxisReductionCycles({"sum", "f32", {8, 16}, 0, 1},
                                                *p, ReductionMode::SIMD, 4);
  ASSERT_TRUE(bool(valid));
  ASSERT_TRUE(bool(*valid));
  EXPECT_DOUBLE_EQ(**valid, 42);
  EXPECT_EQ(p->getParsedFunctionCount(), 1u);
  std::string message;
  for (int i = 0; i < 2; ++i) {
    auto result = estimateReductionCycles({"sum", "f32", {2, 8, 16}, 1, 1}, *p,
                                          ReductionMode::SIMT, 4);
    ASSERT_FALSE(bool(result));
    auto error = llvm::toString(result.takeError());
    EXPECT_NE(error.find("middle_axis/simt/sum/f32/segment"),
              std::string::npos);
    if (i == 0)
      message = error;
    else
      EXPECT_EQ(message, error);
  }
  EXPECT_EQ(p->getParsedFunctionCount(), 1u);
  auto error = p->validateAllFunctions();
  ASSERT_TRUE(bool(error));
  EXPECT_EQ(message, llvm::toString(std::move(error)));

  // Invalid calibration is an error at the Stage boundary, not a fitted miss.
  auto broken = loadLazyTestProfile("unknown");
  ASSERT_TRUE(bool(broken));
  auto hardware = hardwareProfile();
  hardware.reductionProfile = std::move(*broken);
  auto stage = ordinaryTailStage();
  stage.workload.reductionWorkloads = {{"sum", "f32", {8, 16}, 0, 1}};
  stage.workload.reductionGeometryCounts.clear();
  stage.workload.reductionGeometryCounts[{8, 16, 32}] = 1;
  stage.reductionExternalWorkload = StageWorkload{};
  StagePartition partition;
  partition.stages.push_back(std::move(stage));
  auto table = StageCostEvaluator().evaluate(partition, hardware);
  ASSERT_FALSE(bool(table));
  EXPECT_NE(llvm::toString(table.takeError()).find("first_axis/simd/sum/f32"),
            std::string::npos);
}

TEST(SimdSimtCostModelTest, LazyReductionConcurrentInitializationAndSnapshots) {
  using namespace mlir::ascend;
  auto p = loadLazyTestProfile();
  ASSERT_TRUE(bool(p));
  std::array<const ReductionFunction *, 16> pointers{};
  std::array<std::string, 16> errors{};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < pointers.size(); ++i)
    threads.emplace_back([&, i] {
      auto result = i % 2
                        ? p->findFirstAxis(ReductionMode::SIMD, "sum", "f32")
                        : p->findMiddleAxis(ReductionMode::SIMT, "sum", "f32");
      if (!result)
        errors[i] = llvm::toString(result.takeError());
      else
        pointers[i] = *result;
    });
  for (auto &thread : threads)
    thread.join();
  for (size_t i = 0; i < pointers.size(); ++i) {
    EXPECT_TRUE(errors[i].empty());
    ASSERT_NE(pointers[i], nullptr);
    EXPECT_EQ(pointers[i], pointers[i % 2]);
  }
  EXPECT_EQ(p->getParsedFunctionCount(), 2u);
  auto newer = loadLazyTestProfile("84");
  ASSERT_TRUE(bool(newer));
  auto n = newer->findFirstAxis(ReductionMode::SIMD, "sum", "f32");
  ASSERT_TRUE(bool(n));
  EXPECT_NE(*n, pointers[1]);
  EXPECT_DOUBLE_EQ((*n)->segments[0].expression.constant, 84);
  EXPECT_DOUBLE_EQ(pointers[1]->segments[0].expression.constant, 42);
}

TEST(SimdSimtCostModelTest, LazyReductionFullProductValidation) {
  using namespace mlir::ascend;
  auto path = std::string(TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH);
  path = path.substr(0, path.find_last_of('/')) +
         "/../reduction/david_v100_reduction_v1.json";
  auto p = ReductionProfile::loadFromFile(path);
  ASSERT_TRUE(bool(p)) << llvm::toString(p.takeError());
  EXPECT_EQ(p->getParsedFunctionCount(), 0u);
  auto error = p->validateAllFunctions();
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  EXPECT_EQ(p->getParsedFunctionCount(), 252u);
  // All cached trees remain valid after the original wrapper is destroyed.
  auto shared = *p;
  *p = ReductionProfile{};
  auto result = shared.findMiddleAxis(ReductionMode::SIMT, "prod", "i64");
  ASSERT_TRUE(bool(result));
  ASSERT_NE(*result, nullptr);
  EXPECT_EQ(shared.getParsedFunctionCount(), 252u);
}

TEST(SimdSimtCostModelTest, LazyReductionConcurrentErrorIsStable) {
  using namespace mlir::ascend;
  auto p = loadLazyTestProfile("sqrt()");
  ASSERT_TRUE(bool(p));
  std::array<std::string, 8> errors;
  std::vector<std::thread> threads;
  for (size_t i = 0; i < errors.size(); ++i)
    threads.emplace_back([&, i] {
      auto result = p->findFirstAxis(ReductionMode::SIMD, "sum", "f32");
      if (!result)
        errors[i] = llvm::toString(result.takeError());
    });
  for (auto &thread : threads)
    thread.join();
  ASSERT_FALSE(errors[0].empty());
  for (const auto &error : errors)
    EXPECT_EQ(error, errors[0]);
  EXPECT_EQ(p->getParsedFunctionCount(), 0u);
}

TEST(SimdSimtCostModelTest, ReductionExternalSnapshotPreservesMemoryPatterns) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"(
    module { tt.func @probe(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
      %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
      %p = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %a = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %v = tt.load %a : tensor<64x!tt.ptr<f32>>
      %s = "tt.reduce"(%v) ({
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          tt.reduce.return %sum : f32
      }) {axis = 0 : i32} : (tensor<64xf32>) -> f32
      tt.store %y, %s : !tt.ptr<f32>
      tt.return
    } })",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("reduction_with_memory",
                            StageCostModelKind::RowwiseReduction);
  module->walk([&](mlir::triton::FuncOp function) {
    for (mlir::Operation &operation : function.getBody().front())
      if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
        stage.operations.push_back(&operation);
  });
  partition.stages.push_back(std::move(stage));
  mlir::ascend::StageMemoryPatternAnalysis patterns(*module);
  if (auto error = StageWorkloadAnalysis().analyze(partition, 8, &patterns))
    FAIL() << llvm::toString(std::move(error));
  bool found = false;
  for (const auto &stage : partition.stages) {
    if (stage.workload.reductionWorkloads.empty())
      continue;
    found = true;
    ASSERT_TRUE(stage.reductionExternalWorkload);
    ASSERT_FALSE(stage.workload.addressPatterns.empty());
    EXPECT_EQ(stage.workload.addressPatterns.size(),
              stage.reductionExternalWorkload->addressPatterns.size());
    EXPECT_EQ(stage.workload.memoryAccesses.size(),
              stage.reductionExternalWorkload->memoryAccesses.size());
    EXPECT_EQ(stage.reductionContributions.size(), 1u);
    EXPECT_DOUBLE_EQ(stage.reductionExternalWorkload->reductionOperations, 0);
  }
  EXPECT_TRUE(found);
}

TEST(SimdSimtCostModelTest, ReductionRemappingPreservesStridedShortCompletion) {
  auto profile = tailMigrationProfile();
  auto stage = ordinaryTailStage();
  stage.workload.memoryAccesses.front().strides = {2};
  stage.workload.loadBytes = 256;
  stage.workload.continuousLoadCounts[{256, 32}] = 1;
  stage.reductionExternalWorkload = stage.workload;
  mlir::ascend::StridedShortCompletionFit fit;
  fit.intercept = 123;
  fit.evidence = "unit-test";
  profile.simd.stridedShortCompletionFits["simd_load_stride2_r1"] = fit;
  auto fitted = evaluateOneStage(stage, profile);
  profile.reductionProfile.reset();
  auto baseline = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(fitted));
  ASSERT_TRUE(static_cast<bool>(baseline));
  const auto &cost = fitted->stages.front().implementations.front();
  const auto &old = baseline->stages.front().implementations.front();
  ASSERT_EQ(cost.implementation.mode, StageMode::SIMD);
  EXPECT_TRUE(cost.calibratedTailReduction);
  EXPECT_GT(cost.resources.load, 0);
  EXPECT_DOUBLE_EQ(cost.resources.load, old.resources.load);
}
