#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>
#include <algorithm>

using namespace mlir;
using namespace mlir::ascend;

namespace {

class LoopWorkloadCalibrationTest : public ::testing::Test {
protected:
  LoopWorkloadCalibrationTest() {
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
  }

  void analyze(llvm::StringRef body, int64_t normalization = 1) {
    std::string source = R"mlir(
      module {
        func.func @work(%a: tensor<16xf32>, %n: index) {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c3 = arith.constant 3 : index
          %c4 = arith.constant 4 : index
          %c14 = arith.constant 14 : index
    )mlir";
    source += body.str();
    source += "return\n }\n }";
    module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    partition = StagePartition{};
    partition.operationOwnershipComplete = true;
    LogicalStage stage;
    stage.id = "work";
    stage.iterationCount = normalization;
    auto function = cast<func::FuncOp>(&module->getBody()->front());
    for (Operation &operation : function.getBody().front())
      stage.operations.push_back(&operation);
    partition.stages.push_back(std::move(stage));
    auto error = StageWorkloadAnalysis().analyze(partition);
    ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  }

  double total(llvm::StringRef operation) {
    const auto &stage = partition.stages.front();
    auto entry = stage.workload.operationElements.find(operation);
    return entry == stage.workload.operationElements.end()
               ? 0.0
               : entry->second * stage.iterationCount;
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  StagePartition partition;
};

TEST_F(LoopWorkloadCalibrationTest, SetupIsNotReplicatedWithLoopBody) {
  analyze(R"mlir(
    %setup = arith.mulf %a, %a : tensor<16xf32>
    scf.for %i = %c0 to %c14 step %c1 {
      %v = arith.addf %setup, %a : tensor<16xf32>
    }
  )mlir", 14);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_DOUBLE_EQ(total("f32.add"), 14.0 * 16);
  EXPECT_DOUBLE_EQ(total("f32.mul"), 16.0);
  EXPECT_TRUE(partition.stages.front().dynamicWorkloadKnown);
  EXPECT_EQ(partition.stages.front().unknownLoopTripCount, 0);
}

TEST_F(LoopWorkloadCalibrationTest, UnknownTripsRemainVisibleInCostReport) {
  analyze(R"mlir(
    scf.for %i = %c0 to %n step %c1 {
      %v = arith.addf %a, %a : tensor<16xf32>
    }
  )mlir");
  ASSERT_FALSE(HasFatalFailure());
  SimdSimtCostModelOptions options;
  options.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
  options.actualTarget = "Ascend950PR_9579";
  options.numWarps = 4;
  options.compileOn91095 = true;
  auto report = analyzeSimdSimtCandidates(*module, options);
  if (!report)
    FAIL() << llvm::toString(report.takeError());
  bool observed = false;
  for (const auto &stage : report->stageModel.stages) {
    if (stage.dynamicWorkloadKnown)
      continue;
    observed = true;
    auto json = stage.toJSON();
    EXPECT_EQ(json.getBoolean("dynamic_workload_known"), false);
    EXPECT_EQ(json.getInteger("unknown_loop_trip_count"), 1);
    EXPECT_EQ(json.getString("loop_workload_status"),
              "nominal_unknown_trip_count");
    EXPECT_NE(std::find(report->unsupported.begin(), report->unsupported.end(),
                        "unknown_loop_trip_count:" + stage.id),
              report->unsupported.end());
  }
  EXPECT_TRUE(observed);
}

TEST_F(LoopWorkloadCalibrationTest, ZeroTripDoesNotExecuteBody) {
  analyze(R"mlir(
    scf.for %i = %c0 to %c0 step %c1 {
      %v = arith.addf %a, %a : tensor<16xf32>
    }
  )mlir");
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_DOUBLE_EQ(total("f32.add"), 0.0);
  EXPECT_TRUE(partition.stages.front().dynamicWorkloadKnown);
}

TEST_F(LoopWorkloadCalibrationTest, ReversedBoundsDoNotExecuteBody) {
  analyze(R"mlir(
    scf.for %i = %c14 to %c3 step %c1 {
      %v = arith.addf %a, %a : tensor<16xf32>
    }
  )mlir");
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_DOUBLE_EQ(total("f32.add"), 0.0);
}

TEST_F(LoopWorkloadCalibrationTest, PositiveStepUsesCeilingDivision) {
  analyze(R"mlir(
    scf.for %i = %c1 to %c14 step %c4 {
      %v = arith.addf %a, %a : tensor<16xf32>
    }
  )mlir");
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_DOUBLE_EQ(total("f32.add"), 4.0 * 16);
}

TEST_F(LoopWorkloadCalibrationTest, NestedConstantLoopsMultiplyOnlyBodies) {
  analyze(R"mlir(
    scf.for %i = %c0 to %c3 step %c1 {
      %setup = arith.mulf %a, %a : tensor<16xf32>
      scf.for %j = %c0 to %c4 step %c1 {
        %v = arith.addf %setup, %a : tensor<16xf32>
      }
    }
  )mlir", 4);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_DOUBLE_EQ(total("f32.add"), 3.0 * 4 * 16);
  EXPECT_DOUBLE_EQ(total("f32.mul"), 3.0 * 16);
}

TEST_F(LoopWorkloadCalibrationTest, UnknownDoesNotInheritSiblingTripCount) {
  analyze(R"mlir(
    scf.for %i = %c0 to %c14 step %c1 {
      %known = arith.mulf %a, %a : tensor<16xf32>
    }
    scf.for %j = %c0 to %n step %c1 {
      %unknown = arith.addf %a, %a : tensor<16xf32>
    }
  )mlir", 14);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_DOUBLE_EQ(total("f32.mul"), 14.0 * 16);
  EXPECT_DOUBLE_EQ(total("f32.add"), 16.0); // Explicitly nominal, not exact.
  EXPECT_FALSE(partition.stages.front().dynamicWorkloadKnown);
  EXPECT_EQ(partition.stages.front().unknownLoopTripCount, 1);
}

TEST_F(LoopWorkloadCalibrationTest, ZeroTripPrunesUnreachableUnknownLoop) {
  analyze(R"mlir(
    scf.for %i = %c0 to %c0 step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %v = arith.addf %a, %a : tensor<16xf32>
      }
    }
  )mlir");
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_DOUBLE_EQ(total("f32.add"), 0.0);
  EXPECT_TRUE(partition.stages.front().dynamicWorkloadKnown);
  EXPECT_EQ(partition.stages.front().unknownLoopTripCount, 0);
}

TEST_F(LoopWorkloadCalibrationTest, WhileTripCountRemainsUnknown) {
  analyze(R"mlir(
    %result = scf.while (%i = %c0) : (index) -> index {
      %condition = arith.cmpi slt, %i, %n : index
      scf.condition(%condition) %i : index
    } do {
    ^bb0(%i: index):
      %v = arith.addf %a, %a : tensor<16xf32>
      %next = arith.addi %i, %c1 : index
      scf.yield %next : index
    }
  )mlir");
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_FALSE(partition.stages.front().dynamicWorkloadKnown);
  EXPECT_EQ(partition.stages.front().unknownLoopTripCount, 1);
}

TEST_F(LoopWorkloadCalibrationTest, UnrepresentableTripCountIsNotWrapped) {
  analyze(R"mlir(
    %low = arith.constant -9223372036854775808 : index
    %high = arith.constant 9223372036854775807 : index
    scf.for %i = %low to %high step %c1 {
      %v = arith.addf %a, %a : tensor<16xf32>
    }
  )mlir");
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_FALSE(partition.stages.front().dynamicWorkloadKnown);
  EXPECT_EQ(partition.stages.front().unknownLoopTripCount, 1);
  EXPECT_DOUBLE_EQ(total("f32.add"), 16.0); // Nominal, not a wrapped count.
}

TEST_F(LoopWorkloadCalibrationTest, UnsignedBoundsAreNotMisreadAsSignedZeroTrip) {
  analyze(R"mlir(
    %minus_one = arith.constant -1 : index
    scf.for %i = %c0 to %minus_one step %c1 {
      %v = arith.addf %a, %a : tensor<16xf32>
    }
  )mlir");
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_DOUBLE_EQ(total("f32.add"), 0.0);
  module->walk([&](scf::ForOp loop) {
    loop->setAttr("unsignedCmp", UnitAttr::get(&context));
  });
  auto error = StageWorkloadAnalysis().analyze(partition);
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  EXPECT_FALSE(partition.stages.front().dynamicWorkloadKnown);
  EXPECT_EQ(partition.stages.front().unknownLoopTripCount, 1);
  EXPECT_DOUBLE_EQ(total("f32.add"), 16.0); // Nominal; unsigned trip unsupported.
}

} // namespace
