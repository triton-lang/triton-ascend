// CPU integration contract only: these are model outputs, not device timings.
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>

namespace {
constexpr char program[] = R"mlir(
module {
  tt.func public @gather(%data: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %out: !tt.ptr<f32>, %flag: !tt.ptr<f32>) {
    %i = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %j = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %p = tt.splat %idx : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %q = tt.addptr %p, %j : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %k = tt.load %q : tensor<8x!tt.ptr<i32>>
    %row = tt.expand_dims %i {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %rows = tt.broadcast %row : tensor<4x1xi32> -> tensor<4x8xi32>
    %col = tt.expand_dims %k {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %cols = tt.broadcast %col : tensor<1x8xi32> -> tensor<4x8xi32>
    %lane = tt.expand_dims %j {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %lanes = tt.broadcast %lane : tensor<1x8xi32> -> tensor<4x8xi32>
    %pitch = arith.constant dense<8> : tensor<4x8xi32>
    %base = arith.muli %rows, %pitch : tensor<4x8xi32>
    %read = arith.addi %base, %cols : tensor<4x8xi32>
    %write = arith.addi %base, %lanes : tensor<4x8xi32>
    %src = tt.splat %data : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %dst = tt.splat %out : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %a = tt.addptr %src, %read : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %b = tt.addptr %dst, %write : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %v = tt.load %a : tensor<4x8x!tt.ptr<f32>> loc("target.load")
    tt.store %b, %v : tensor<4x8x!tt.ptr<f32>>
    %one = arith.constant 1.0 : f32
    tt.store %flag, %one : !tt.ptr<f32>
    tt.return
  }
}
)mlir";
struct Observation {
  std::string profile, stage, reason;
  bool selected = false;
  double increment = 0, total = 0;
};

void observe(bool disable, Observation &out) {
  using namespace mlir;
  namespace model = mlir::ascend;
  MLIRContext context;
  context.disableMultithreading();
  context.getOrLoadDialect<triton::TritonDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  auto module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  if (disable)
    (*module)->setAttr(
        "ascend.costmodel.ordinary_load.route_capability_json",
        StringAttr::get(&context,
                        R"({"ordinary_load_selected_enabled":false})"));
  ASSERT_TRUE(succeeded(verify(*module)));
  std::string target;
  unsigned marks = 0;
  module->walk([&](triton::LoadOp load) {
    auto loc = dyn_cast<NameLoc>(load.getLoc());
    if (loc && loc.getName().getValue() == "target.load") {
      ++marks;
      llvm::raw_string_ostream os(target);
      load.getLoc().print(os);
      os.flush();
    }
  });
  ASSERT_EQ(marks, 1u);
  model::SimdSimtCostModelOptions options;
  options.profilePath = TRITON_ASCEND_LOAD_ROUTE_TEST_PROFILE;
  options.compileOn91095 = true;
  options.collectCalibrationEstimates = false;
  auto result = model::analyzeSimdSimtCandidates(*module, options);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_TRUE(result->stageModel.applied);
  ASSERT_TRUE(result->stageModel.operationOwnershipComplete);
  unsigned owners = 0, rows = 0;
  for (const auto &stage : result->stageModel.stages) {
    if (!llvm::is_contained(stage.sourceLocations, target))
      continue;
    ++owners;
    for (const auto &cost : stage.implementations) {
      const auto &impl = cost.implementation;
      if (impl.mode != model::StageMode::SIMT || impl.localScope ||
          impl.superblockFactor != 1 || impl.aivSubblocks != 1)
        continue;
      ++rows;
      out = {result->profileContentSha256,
             stage.id,
             cost.ordinaryLoadCandidateReason,
             cost.ordinaryLoadSelectedUsed,
             cost.ordinaryLoadTargetIncrementSystemCycles,
             cost.totalCycles};
    }
  }
  ASSERT_EQ(owners, 1u) << target;
  ASSERT_EQ(rows, 1u);
}

TEST(OrdinaryLoadRoute, SimtDefaultAndExplicitOff) {
  Observation on, off;
  ASSERT_NO_FATAL_FAILURE(observe(false, on));
  ASSERT_NO_FATAL_FAILURE(observe(true, off));
  ASSERT_EQ(on.profile.size(), 64u);
  ASSERT_EQ(on.profile, off.profile);
  ASSERT_EQ(on.stage, off.stage);
  EXPECT_TRUE(on.selected) << on.reason;
  EXPECT_GT(on.increment, 0);
  EXPECT_FALSE(off.selected) << off.reason;
  EXPECT_EQ(off.reason, "Load_formula_explicitly_disabled");
  EXPECT_EQ(off.increment, 0);
  // Fallback pricing can change other charges: do not equate total delta to
  // Load.
  EXPECT_GT(on.total, 0);
  EXPECT_GT(off.total, 0);
  llvm::outs() << "synthetic SIMT F1 model cycles: Load default="
               << on.increment << " off=" << off.increment
               << "; stage default=" << on.total << " off=" << off.total
               << '\n';
  RecordProperty("default_stage_cycles", std::to_string(on.total));
  RecordProperty("disabled_stage_cycles", std::to_string(off.total));
  RecordProperty("default_load_cycles", std::to_string(on.increment));
}
} // namespace
