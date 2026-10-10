// Test inputs for this PR's added indirect-memory tests only.
#ifndef ASCEND_COSTMODEL_UT_COSTMODELTESTUTILS_H
#define ASCEND_COSTMODEL_UT_COSTMODELTESTUTILS_H

#include "AscendModel/RouteModel/StageCostModels.h"
#include <gtest/gtest.h>
#include <utility>

namespace mlir::ascend::test {

inline HardwareProfile hardwareProfile(StageTransitionCost transition = {}) {
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
  profile.simt.vectorWidthBits = 1;
  profile.simt.issueWidth = 32;
  profile.transition = std::move(transition);
  return profile;
}

inline LogicalStage
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

inline llvm::Expected<StageCostTable>
evaluateOneStage(LogicalStage stage,
                 HardwareProfile profile = hardwareProfile()) {
  StagePartition partition;
  partition.stages.push_back(std::move(stage));
  return StageCostEvaluator().evaluate(partition, profile);
}

} // namespace mlir::ascend::test
#endif // ASCEND_COSTMODEL_UT_COSTMODELTESTUTILS_H
