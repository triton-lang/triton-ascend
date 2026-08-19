#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/RouteModel/SimtAnchorAnalysis.h"
#include "AscendModel/RouteModel/SimtSelection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/Support/JSON.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningOpRef;
using mlir::Pass;
using mlir::PassManager;
using mlir::StringAttr;
using mlir::ascend::analyzeSimdSimtFeatures;
using mlir::ascend::buildMixedSimtAnchorPlan;
using mlir::ascend::createAssignOpIDsPass;
using mlir::ascend::createEstimateCyclesPass;
using mlir::ascend::createPerfReportPass;
using mlir::ascend::createPipelineAnalysisPass;
using mlir::ascend::createSelectSimdSimtCostModelPass;
using mlir::ascend::EstimateCyclesPassOptions;
using mlir::ascend::materializeSimtAnchorPlan;
using mlir::ascend::SelectSimdSimtCostModelPassOptions;

namespace {

constexpr const char *kVectorModule = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = ascend.vector_load %arg0 {bytes = 16 : i64} : tensor<4xf32> -> tensor<4xf32>
    %1 = ascend.add %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    ascend.vector_store %1 {bytes = 16 : i64} : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
)mlir";

constexpr const char *kOutOfSimdSimtCoverageModule = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";

void registerDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::ascend::AscendModelDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::scope::ScopeDialect>();
}

OwningOpRef<ModuleOp> parseModule(mlir::MLIRContext &context,
                                  llvm::StringRef source) {
  registerDialects(context);
  return mlir::parseSourceString<ModuleOp>(source, &context);
}

template <typename... PassTs>
bool runPasses(ModuleOp module, PassTs &&...passes) {
  PassManager pm(module.getContext());
  (pm.addPass(std::forward<PassTs>(passes)), ...);
  return mlir::succeeded(pm.run(module));
}

Operation *findFirstOp(ModuleOp module, llvm::StringRef name) {
  Operation *result = nullptr;
  module.walk([&](Operation *op) {
    if (!result && op->getName().getStringRef() == name)
      result = op;
  });
  return result;
}

int64_t getI64Attr(Operation *op, llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  return attr ? attr.getInt() : -1;
}

} // namespace

TEST(CostModelPassesTest, AssignOpIDsPassAnnotatesAscendOpsOnly) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%arg0: i32, %arg1: i32, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.addi %arg0, %arg1 : i32
    %0 = ascend.add %arg2, %arg2 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createAssignOpIDsPass()));

  auto totalOps = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.total_ops");
  ASSERT_TRUE(totalOps);
  EXPECT_EQ(totalOps.getInt(), 1);

  Operation *addOp = findFirstOp(*module, "ascend.add");
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(getI64Attr(addOp, "op_id"), 0);

  Operation *arithOp = findFirstOp(*module, "arith.addi");
  ASSERT_NE(arithOp, nullptr);
  EXPECT_FALSE(arithOp->hasAttr("op_id"));
}

TEST(CostModelPassesTest, EstimateCyclesAnnotatesComputeAndTransferOps) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createEstimateCyclesPass()));

  Operation *loadOp = findFirstOp(*module, "ascend.vector_load");
  Operation *addOp = findFirstOp(*module, "ascend.add");
  Operation *storeOp = findFirstOp(*module, "ascend.vector_store");
  ASSERT_NE(loadOp, nullptr);
  ASSERT_NE(addOp, nullptr);
  ASSERT_NE(storeOp, nullptr);

  EXPECT_GT(getI64Attr(loadOp, "estimated_cycles"), 0);
  EXPECT_GT(getI64Attr(addOp, "estimated_cycles"), 0);
  EXPECT_GT(getI64Attr(storeOp, "estimated_cycles"), 0);
  EXPECT_EQ(getI64Attr(loadOp, "bytes"), 16);
  EXPECT_EQ(getI64Attr(storeOp, "bytes"), 16);
  EXPECT_EQ(getI64Attr(addOp, "flops"), 4);
  EXPECT_TRUE(loadOp->getAttrOfType<StringAttr>("hw_unit"));
  EXPECT_TRUE(addOp->getAttrOfType<StringAttr>("hw_unit"));
  EXPECT_TRUE(storeOp->getAttrOfType<StringAttr>("hw_unit"));
}

TEST(CostModelPassesTest, DavidF32AddConsumesSharedThroughputInAbsoluteModel) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = ascend.add %arg0, %arg0
      : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  EstimateCyclesPassOptions options;
  options.hardwareConfigPath = TRITON_ASCEND_DAVID_TEST_CONFIG_PATH;
  ASSERT_TRUE(runPasses(*module, createEstimateCyclesPass(options)));

  Operation *addOp = findFirstOp(*module, "ascend.add");
  ASSERT_NE(addOp, nullptr);
  // The absolute model uses the TileSim VADD table: 1024 f32 values / 64
  // lanes = 16 repeats, plus 35 startup cycles.
  EXPECT_EQ(getI64Attr(addOp, "estimated_cycles"), 51);
}

TEST(CostModelPassesTest,
     SimtFeatureExtractionUsesSSAProvenanceAndUniqueMasks) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(
      %index_ptr: tensor<4x16xi64>,
      %data_ptr: tensor<4x16xi64>) -> tensor<4x16xf32> {
    %idx = "tt.load"(%index_ptr)
      : (tensor<4x16xi64>) -> tensor<4x16xi64>
    %addr = arith.addi %data_ptr, %idx : tensor<4x16xi64>
    %zero = arith.constant dense<0> : tensor<4x16xi64>
    %mask = arith.cmpi sge, %idx, %zero : tensor<4x16xi64>
    %data = "tt.load"(%addr, %mask)
      : (tensor<4x16xi64>, tensor<4x16xi1>) -> tensor<4x16xf32>
    %reduced = "tt.reduce"(%data) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      "tt.reduce.return"(%sum) : (f32) -> ()
    }) {axis = 1 : i32}
      : (tensor<4x16xf32>) -> tensor<4xf32>
    return %data : tensor<4x16xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  auto plan = buildMixedSimtAnchorPlan(*module, /*compileOn91095=*/true);
  ASSERT_EQ(plan.anchors.size(), 1u);
  EXPECT_EQ(plan.materializableCount(), 1);
  EXPECT_EQ(mlir::ascend::stringifySimtAnchorKind(plan.anchors[0].kind),
            "loaded_index_dependent_memory");

  auto features = analyzeSimdSimtFeatures(*module, plan);
  if (!features)
    FAIL() << llvm::toString(features.takeError());

  EXPECT_EQ(features->loadedIndexDependentMemoryOps, 1);
  EXPECT_EQ(features->simtAnchors.loadedIndexDependentMemoryOps, 1);
  EXPECT_EQ(features->simtAnchors.recognizedCount, 1);
  EXPECT_EQ(features->simtAnchors.count, 1);
  ASSERT_EQ(features->simtAnchors.mechanismKinds.size(), 1u);
  EXPECT_EQ(features->simtAnchors.mechanismKinds.front(),
            "loaded_index_dependent_memory");

  // The same predicate is produced once and consumed by tt.load.  The legacy
  // rank sum sees both uses, while the hardware-facing fields count one SSA
  // value and its 64 predicate elements exactly once.
  EXPECT_GT(features->maskRankSum, features->uniqueMaskRankSum);
  EXPECT_EQ(features->uniqueMaskValues, 1);
  EXPECT_EQ(features->uniqueMaskRankSum, 2);
  EXPECT_EQ(features->predicateElements, 64);
  EXPECT_EQ(features->simtAnchors.uniqueMaskValues, 1);
  EXPECT_EQ(features->simtAnchors.predicateElements, 64);

  EXPECT_EQ(features->rowLocalReduceOps, 1);
  EXPECT_EQ(features->maxReduceAxisExtent, 16);
  EXPECT_EQ(features->weightedReduceAxisElements, 16);
}

TEST(CostModelPassesTest,
     SimtAnchorAnalysisExtractsHistogramFactsAndLowerability) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%input: tensor<64xi32>) -> tensor<256xi32> {
    %histogram = "tt.histogram"(%input)
      : (tensor<64xi32>) -> tensor<256xi32>
    return %histogram : tensor<256xi32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  auto plan = buildMixedSimtAnchorPlan(*module, /*compileOn91095=*/true);
  ASSERT_EQ(plan.anchors.size(), 1u);
  const auto &anchor = plan.anchors.front();
  EXPECT_EQ(anchor.kind, mlir::ascend::SimtAnchorKind::Histogram);
  const auto *facts = std::get_if<mlir::ascend::HistogramFacts>(&anchor.facts);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->inputElements, 64);
  EXPECT_EQ(facts->numBins, 256);
  EXPECT_EQ(facts->inputType, "i32");
  EXPECT_EQ(facts->resultType, "i32");

  EXPECT_EQ(anchor.lowerability.allSimd,
            mlir::ascend::CandidateLoweringStatus::Unsupported);
  EXPECT_EQ(anchor.lowerability.allSimtOnly,
            mlir::ascend::CandidateLoweringStatus::Unsupported);
  EXPECT_EQ(anchor.lowerability.mixed,
            mlir::ascend::CandidateLoweringStatus::Native);
  EXPECT_TRUE(anchor.materializable);
  EXPECT_EQ(plan.kernelLowerability.allSimd,
            mlir::ascend::CandidateLoweringStatus::Unsupported);
  EXPECT_EQ(plan.kernelLowerability.allSimtOnly,
            mlir::ascend::CandidateLoweringStatus::Unsupported);
  EXPECT_EQ(plan.kernelLowerability.mixed,
            mlir::ascend::CandidateLoweringStatus::Native);
}

TEST(CostModelPassesTest,
     SimtAnchorAnalysisExtractsPlainCumsumFactsAndLowerability) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%input: tensor<1x128x1xf32>)
      -> tensor<1x128x1xf32> {
    %cumsum = "tt.scan"(%input) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      "tt.scan.return"(%sum) : (f32) -> ()
    }) {axis = 1 : i32, reverse = true}
      : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    return %cumsum : tensor<1x128x1xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  auto plan = buildMixedSimtAnchorPlan(*module, /*compileOn91095=*/true);
  ASSERT_EQ(plan.anchors.size(), 1u);
  const auto &anchor = plan.anchors.front();
  EXPECT_EQ(anchor.kind,
            mlir::ascend::SimtAnchorKind::PlainOneDimensionalCumsum);
  const auto *facts =
      std::get_if<mlir::ascend::PlainCumsumFacts>(&anchor.facts);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->axisExtent, 128);
  EXPECT_EQ(facts->elementType, "f32");
  EXPECT_TRUE(facts->reverse);

  EXPECT_EQ(anchor.lowerability.allSimd,
            mlir::ascend::CandidateLoweringStatus::AliasesMixed);
  EXPECT_EQ(anchor.lowerability.allSimtOnly,
            mlir::ascend::CandidateLoweringStatus::BackendConditional);
  EXPECT_EQ(anchor.lowerability.mixed,
            mlir::ascend::CandidateLoweringStatus::Native);
  EXPECT_TRUE(anchor.materializable);
  EXPECT_EQ(plan.kernelLowerability.allSimd,
            mlir::ascend::CandidateLoweringStatus::AliasesMixed);
  EXPECT_EQ(plan.kernelLowerability.allSimtOnly,
            mlir::ascend::CandidateLoweringStatus::BackendConditional);
  EXPECT_EQ(plan.kernelLowerability.mixed,
            mlir::ascend::CandidateLoweringStatus::Native);
}

TEST(CostModelPassesTest,
     SimtAnchorAnalysisExtractsTensorAtomicFactsAndLowerability) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(
      %index_pointer: tensor<64xi64>,
      %base_pointer: tensor<64xi64>,
      %value: tensor<64xf32>) -> tensor<64xf32> {
    %index = "tt.load"(%index_pointer)
      : (tensor<64xi64>) -> tensor<64xi64>
    %address = "tt.addptr"(%base_pointer, %index)
      : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi64>
    %mask = arith.constant dense<true> : tensor<64xi1>
    %old = "tt.atomic_rmw"(%address, %value, %mask)
      {atomic_rmw_op = 5 : i32}
      : (tensor<64xi64>, tensor<64xf32>, tensor<64xi1>)
        -> tensor<64xf32>
    return %old : tensor<64xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  auto plan = buildMixedSimtAnchorPlan(*module, /*compileOn91095=*/true);
  ASSERT_EQ(plan.anchors.size(), 1u);
  const auto &anchor = plan.anchors.front();
  EXPECT_EQ(anchor.kind, mlir::ascend::SimtAnchorKind::TensorAtomic);
  const auto *facts =
      std::get_if<mlir::ascend::TensorAtomicFacts>(&anchor.facts);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->updateElements, 64);
  EXPECT_EQ(facts->addressRank, 1);
  EXPECT_EQ(facts->valueType, "f32");
  EXPECT_EQ(facts->offsetType, "i64");
  EXPECT_EQ(facts->operation, "fadd");
  EXPECT_TRUE(facts->hasMask);
  ASSERT_TRUE(facts->staticMaskActiveFraction.has_value());
  EXPECT_DOUBLE_EQ(*facts->staticMaskActiveFraction, 1.0);
  EXPECT_TRUE(facts->resultUsed);
  EXPECT_TRUE(facts->addressIsLaneVarying);
  EXPECT_TRUE(facts->addressDependsOnLoadedIndex);

  EXPECT_EQ(anchor.lowerability.allSimd,
            mlir::ascend::CandidateLoweringStatus::Native);
  EXPECT_EQ(anchor.lowerability.allSimtOnly,
            mlir::ascend::CandidateLoweringStatus::BackendConditional);
  EXPECT_EQ(anchor.lowerability.mixed,
            mlir::ascend::CandidateLoweringStatus::Native);
  EXPECT_TRUE(anchor.materializable);
  EXPECT_EQ(plan.kernelLowerability.allSimd,
            mlir::ascend::CandidateLoweringStatus::Native);
  EXPECT_EQ(plan.kernelLowerability.allSimtOnly,
            mlir::ascend::CandidateLoweringStatus::BackendConditional);
  EXPECT_EQ(plan.kernelLowerability.mixed,
            mlir::ascend::CandidateLoweringStatus::Native);
}

TEST(CostModelPassesTest,
     SimtAnchorAnalysisRecognizesTriangularSolveLoopGroup) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%input: tensor<16xi32>, %state: tensor<16x16xf32>,
                  %limit: index)
      -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %range = "tt.make_range"() {end = 16 : i32, start = 0 : i32}
      : () -> tensor<16xi32>
    %row = "tt.expand_dims"(%range) {axis = 1 : i32}
      : (tensor<16xi32>) -> tensor<16x1xi32>
    %column = "tt.expand_dims"(%range) {axis = 0 : i32}
      : (tensor<16xi32>) -> tensor<1x16xi32>
    %rows = "tt.broadcast"(%row)
      : (tensor<16x1xi32>) -> tensor<16x16xi32>
    %columns = "tt.broadcast"(%column)
      : (tensor<1x16xi32>) -> tensor<16x16xi32>
    %mask = arith.cmpi sgt, %rows, %columns : tensor<16x16xi32>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32>
    %initial = "tt.load"(%state)
      : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %a = scf.for %i = %c0 to %limit step %c1 iter_args(%acc = %state)
        -> (tensor<16x16xf32>) {
      %load = "tt.load"(%input) : (tensor<16xi32>) -> tensor<16xf32>
      %red = "tt.reduce"(%acc) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "tt.reduce.return"(%sum) : (f32) -> ()
      }) {axis = 0 : i32} : (tensor<16x16xf32>) -> tensor<16xf32>
      %sel = arith.select %mask, %acc, %zero : tensor<16x16xi1>, tensor<16x16xf32>
      scf.yield %sel : tensor<16x16xf32>
    }
    %b = scf.for %i = %c0 to %limit step %c1 iter_args(%acc = %a)
        -> (tensor<16x16xf32>) {
      %load = "tt.load"(%input) : (tensor<16xi32>) -> tensor<16xf32>
      %red = "tt.reduce"(%acc) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "tt.reduce.return"(%sum) : (f32) -> ()
      }) {axis = 0 : i32} : (tensor<16x16xf32>) -> tensor<16xf32>
      %sel = arith.select %mask, %acc, %zero : tensor<16x16xi1>, tensor<16x16xf32>
      scf.yield %sel : tensor<16x16xf32>
    }
    return %b : tensor<16x16xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  auto plan = buildMixedSimtAnchorPlan(*module, /*compileOn91095=*/true);
  ASSERT_EQ(plan.anchors.size(), 1u);
  EXPECT_EQ(plan.materializableCount(), 1);
  for (const auto &anchor : plan.anchors) {
    EXPECT_EQ(anchor.kind, mlir::ascend::SimtAnchorKind::TriangularSolveLoop);
    EXPECT_EQ(anchor.lowerability.mixed,
              mlir::ascend::CandidateLoweringStatus::Native);
    EXPECT_TRUE(anchor.materializable);
    // Both recurrence loops are one physical SIMT scope and therefore one
    // scored/materialized anchor, not two independent route decisions.
    EXPECT_GE(anchor.scopeOperations.size(), 2u);
    EXPECT_EQ(anchor.scopeInsertionPoint, anchor.operation);
    EXPECT_EQ(anchor.scopeOperations.front()->getName().getStringRef(),
              "tt.make_range");
  }
  EXPECT_EQ(plan.kernelLowerability.mixed,
            mlir::ascend::CandidateLoweringStatus::Native);

  auto features = analyzeSimdSimtFeatures(*module, plan);
  if (!features)
    FAIL() << llvm::toString(features.takeError());
  EXPECT_EQ(features->simtAnchors.count, 1);
  EXPECT_EQ(features->simtAnchors.staticLoopCount, 2);
  EXPECT_EQ(features->simtAnchors.staticLoopTripCountSum, 28);
  EXPECT_EQ(features->simtAnchors.modeledDynamicLoopCount, 2);
  EXPECT_EQ(features->simtAnchors.modeledDynamicLoopTripCountSum, 28);
  EXPECT_EQ(features->simtAnchors.reduceOps, 2);
  EXPECT_EQ(features->simtAnchors.weightedOps.lookup("reduce"), 28);
  EXPECT_EQ(features->simtAnchors.shuffleLaneSteps, 28672);
  // 2 loop-weighted mask uses: 2 * 256 * 14, plus the 256-lane mask setup
  // that is moved into the same scope.
  EXPECT_EQ(features->simtAnchors.predicateLaneEvaluations, 7424);
  EXPECT_TRUE(features->hasUnknownTripCount);

  ASSERT_TRUE(mlir::succeeded(materializeSimtAnchorPlan(*module, plan)));
  Operation *scope = findFirstOp(*module, "scope.scope");
  ASSERT_NE(scope, nullptr);
  EXPECT_EQ(scope->getAttrOfType<mlir::StringAttr>("vector_mode").getValue(),
            "simt");
  Operation *initialLoad = findFirstOp(*module, "tt.load");
  ASSERT_NE(initialLoad, nullptr);
  EXPECT_NE(initialLoad->getParentOp(), scope);
  bool tensorMaskSetupMovedIntoScope = false;
  Operation *floatingZero = nullptr;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "arith.constant" ||
        op->getNumResults() != 1)
      return;
    auto type =
        llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (type && type.getElementType().isF32())
      floatingZero = op;
  });
  scope->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "arith.cmpi" ||
        op->getNumResults() != 1)
      return;
    auto type =
        llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (type && type.getElementType().isInteger(1))
      tensorMaskSetupMovedIntoScope = true;
  });
  EXPECT_TRUE(tensorMaskSetupMovedIntoScope);
  ASSERT_NE(floatingZero, nullptr);
  EXPECT_NE(floatingZero->getParentOp(), scope);
}

TEST(CostModelPassesTest, EstimateCyclesReportsInvalidArgBindings) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  EstimateCyclesPassOptions options;
  options.argBindingsStr = "arg0";

  EXPECT_FALSE(runPasses(*module, createEstimateCyclesPass(options)));
}

TEST(CostModelPassesTest, PipelineAnalysisSetsCycleSummaryAttrs) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createAssignOpIDsPass(),
                        createEstimateCyclesPass(),
                        createPipelineAnalysisPass()));

  auto scheduled = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.scheduled_cycles_one_iter");
  auto roofline = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.roofline_cycles");
  auto simple = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.simple_sum_cycles");
  ASSERT_TRUE(scheduled);
  ASSERT_TRUE(roofline);
  ASSERT_TRUE(simple);
  EXPECT_GT(scheduled.getInt(), 0);
  EXPECT_GT(roofline.getInt(), 0);
  EXPECT_GT(simple.getInt(), 0);
}

TEST(CostModelPassesTest, PerfReportPassAcceptsEstimatedPipeline) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  EXPECT_TRUE(runPasses(*module, createAssignOpIDsPass(),
                        createEstimateCyclesPass(),
                        createPipelineAnalysisPass(), createPerfReportPass()));
}

TEST(CostModelPassesTest, SimdSimtCoverageShortCircuitIsAutoOnly) {
  auto configureOptions = [](SelectSimdSimtCostModelPassOptions &options,
                             llvm::StringRef mode) {
    options.mode = mode.str();
    options.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
    options.actualTarget = "Ascend950PR_9579";
    options.numWarps = 4;
    options.marginRatio = 0.10;
    options.compileOn91095 = true;
  };

  mlir::MLIRContext autoContext;
  auto autoModule = parseModule(autoContext, kOutOfSimdSimtCoverageModule);
  ASSERT_TRUE(autoModule);
  SelectSimdSimtCostModelPassOptions autoOptions;
  configureOptions(autoOptions, "auto");
  ASSERT_TRUE(
      runPasses(*autoModule, createSelectSimdSimtCostModelPass(autoOptions)));

  auto autoEffective =
      (*autoModule)
          ->getAttrOfType<StringAttr>("ascend.simt_costmodel.effective");
  auto autoRecommended =
      (*autoModule)
          ->getAttrOfType<StringAttr>("ascend.simt_costmodel.recommended");
  auto autoReport =
      (*autoModule)
          ->getAttrOfType<StringAttr>("ascend.simt_costmodel.report_json");
  ASSERT_TRUE(autoEffective);
  ASSERT_TRUE(autoRecommended);
  ASSERT_TRUE(autoReport);
  EXPECT_EQ(autoEffective.getValue(), "backend_default");
  EXPECT_EQ(autoRecommended.getValue(), "backend_default");
  EXPECT_FALSE((*autoModule)->hasAttr("ascend.simt_costmodel.all_simd_score"));
  auto autoJSON = llvm::json::parse(autoReport.getValue());
  ASSERT_TRUE(static_cast<bool>(autoJSON));
  auto *autoObject = autoJSON->getAsObject();
  ASSERT_NE(autoObject, nullptr);
  auto autoEvaluated = autoObject->getBoolean("candidate_costs_evaluated");
  ASSERT_TRUE(autoEvaluated);
  EXPECT_FALSE(*autoEvaluated);
  auto *autoCandidateCosts = autoObject->get("candidate_costs");
  auto *autoDecision = autoObject->get("decision_kind");
  ASSERT_NE(autoCandidateCosts, nullptr);
  ASSERT_NE(autoDecision, nullptr);
  EXPECT_TRUE(autoCandidateCosts->getAsNull().has_value());
  EXPECT_TRUE(autoDecision->getAsNull().has_value());
  auto autoReason = autoObject->getString("application_reason");
  ASSERT_TRUE(autoReason);
  EXPECT_EQ(*autoReason, "selection_score_invalid");

  mlir::MLIRContext reportContext;
  auto reportModule = parseModule(reportContext, kOutOfSimdSimtCoverageModule);
  ASSERT_TRUE(reportModule);
  SelectSimdSimtCostModelPassOptions reportOptions;
  configureOptions(reportOptions, "report");
  ASSERT_TRUE(runPasses(*reportModule,
                        createSelectSimdSimtCostModelPass(reportOptions)));

  auto reportEffective =
      (*reportModule)
          ->getAttrOfType<StringAttr>("ascend.simt_costmodel.effective");
  auto reportJSONAttr =
      (*reportModule)
          ->getAttrOfType<StringAttr>("ascend.simt_costmodel.report_json");
  ASSERT_TRUE(reportEffective);
  ASSERT_TRUE(reportJSONAttr);
  EXPECT_EQ(reportEffective.getValue(), "backend_default");
  EXPECT_TRUE((*reportModule)->hasAttr("ascend.simt_costmodel.all_simd_score"));
  auto reportJSON = llvm::json::parse(reportJSONAttr.getValue());
  ASSERT_TRUE(static_cast<bool>(reportJSON));
  auto *reportObject = reportJSON->getAsObject();
  ASSERT_NE(reportObject, nullptr);
  auto reportEvaluated = reportObject->getBoolean("candidate_costs_evaluated");
  ASSERT_TRUE(reportEvaluated);
  EXPECT_TRUE(*reportEvaluated);
  auto reportReason = reportObject->getString("application_reason");
  ASSERT_TRUE(reportReason);
  EXPECT_EQ(*reportReason, "report_mode");
}

TEST(CostModelPassesTest, MaterializeSimtScopePreservesEscapingSSAResult) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module attributes {
  ascend.simt_costmodel.effective = "mixed_simd_simt"
} {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg1 : i32
    return %1 : i32
  }
}
)mlir");
  ASSERT_TRUE(module);

  Operation *addBeforeMaterialization = findFirstOp(*module, "arith.addi");
  ASSERT_NE(addBeforeMaterialization, nullptr);
  mlir::ascend::SimtAnchorPlan plan;
  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = addBeforeMaterialization;
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.materializable = true;
  plan.anchors.push_back(anchor);
  ASSERT_TRUE(mlir::succeeded(materializeSimtAnchorPlan(*module, plan)));

  Operation *scopeOp = findFirstOp(*module, "scope.scope");
  ASSERT_NE(scopeOp, nullptr);
  ASSERT_EQ(scopeOp->getNumRegions(), 1u);
  ASSERT_EQ(scopeOp->getNumResults(), 1u);
  ASSERT_TRUE(scopeOp->getAttrOfType<StringAttr>("vector_mode"));
  EXPECT_EQ(scopeOp->getAttrOfType<StringAttr>("vector_mode").getValue(),
            "simt");

  auto &scopeBody = scopeOp->getRegion(0).front();
  Operation *scopedAdd = nullptr;
  Operation *scopeReturn = nullptr;
  for (Operation &nested : scopeBody) {
    if (nested.getName().getStringRef() == "arith.addi")
      scopedAdd = &nested;
    if (nested.getName().getStringRef() == "scope.return")
      scopeReturn = &nested;
  }
  ASSERT_NE(scopedAdd, nullptr);
  ASSERT_NE(scopeReturn, nullptr);
  ASSERT_EQ(scopeReturn->getNumOperands(), 1u);
  EXPECT_EQ(scopeReturn->getOperand(0), scopedAdd->getResult(0));

  Operation *mulOp = findFirstOp(*module, "arith.muli");
  ASSERT_NE(mulOp, nullptr);
  ASSERT_EQ(mulOp->getNumOperands(), 2u);
  EXPECT_EQ(mulOp->getOperand(0), scopeOp->getResult(0));
  EXPECT_NE(mulOp->getParentOp(), scopeOp);

  EXPECT_FALSE(module->getOperation()->hasAttr(
      "ascend.simt_costmodel.scope_materialized"));
}

TEST(CostModelPassesTest, NativeWholeBodySimtScopeDetectionAndInlining) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func public @main(%arg0: i32) {
    %c1 = arith.constant 1 : i32
    "scope.scope"() ({
      "scope.scope"() ({
        %0 = arith.addi %arg0, %c1 : i32
        "scope.return"() : () -> ()
      }) {vector_mode = "simt"} : () -> ()
      "scope.return"() : () -> ()
    }) {vector_mode = "simt"} : () -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  Operation *scope = mlir::ascend::simt_selection::findWholeBodyVoidSimtScope(
      module->getOperation());
  ASSERT_NE(scope, nullptr);
  EXPECT_EQ(mlir::ascend::simt_selection::inlineVoidSimtScopesForPureSimt(
                module->getOperation()),
            2);
  EXPECT_EQ(findFirstOp(*module, "scope.scope"), nullptr);
  EXPECT_NE(findFirstOp(*module, "arith.addi"), nullptr);
  EXPECT_EQ(mlir::ascend::simt_selection::findWholeBodyVoidSimtScope(
                module->getOperation()),
            nullptr);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

  auto resultBearing = parseModule(context, R"mlir(
module {
  func.func public @main(%arg0: i32) -> i32 {
    %0 = "scope.scope"() ({
      "scope.return"(%arg0) : (i32) -> ()
    }) {vector_mode = "simt"} : () -> i32
    return %0 : i32
  }
}
)mlir");
  ASSERT_TRUE(resultBearing);
  EXPECT_EQ(mlir::ascend::simt_selection::findWholeBodyVoidSimtScope(
                resultBearing->getOperation()),
            nullptr);
  EXPECT_EQ(mlir::ascend::simt_selection::inlineVoidSimtScopesForPureSimt(
                resultBearing->getOperation()),
            0);
  EXPECT_NE(findFirstOp(*resultBearing, "scope.scope"), nullptr);
}

TEST(CostModelPassesTest, ModelControlledRoutingIgnoresLegacyGlobalForce) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module attributes {
  ascend.simt_costmodel.effective = "all_simd"
} {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
)mlir");
  ASSERT_TRUE(module);

  Operation *addOp = findFirstOp(*module, "arith.addi");
  ASSERT_NE(addOp, nullptr);
  EXPECT_TRUE(mlir::ascend::simt_selection::isModelControlled(addOp));
  EXPECT_FALSE(mlir::ascend::simt_selection::shouldUseSimtTemplate(
      addOp, /*legacyForceSimt=*/true));

  (*module)->setAttr(mlir::ascend::simt_selection::kEffectiveExecutionAttr,
                     mlir::StringAttr::get(&context, "mixed_simd_simt"));
  mlir::ascend::SimtAnchorPlan plan;
  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = addOp;
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.materializable = true;
  plan.anchors.push_back(anchor);
  ASSERT_TRUE(mlir::succeeded(materializeSimtAnchorPlan(*module, plan)));
  addOp = findFirstOp(*module, "arith.addi");
  ASSERT_NE(addOp, nullptr);
  EXPECT_TRUE(mlir::ascend::simt_selection::shouldUseSimtTemplate(
      addOp, /*legacyForceSimt=*/false));

  (*module)->setAttr(mlir::ascend::simt_selection::kEffectiveExecutionAttr,
                     mlir::StringAttr::get(&context, "backend_default"));
  EXPECT_FALSE(mlir::ascend::simt_selection::isModelControlled(addOp));
  EXPECT_TRUE(mlir::ascend::simt_selection::shouldUseSimtTemplate(
      addOp, /*legacyForceSimt=*/true));
}
