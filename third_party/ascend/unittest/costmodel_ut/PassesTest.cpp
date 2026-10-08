#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Transforms/SimtSelection.h"

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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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

struct PredicateWork {
  double elements = 0, selects = 0, generic = 0;
  double maskElements = 0, cmp16Elements = 0, cmp32Elements = 0;
  double describedElements = 0;
  double projected16Elements = 0, projected32Elements = 0;
};

bool collectPredicateWork(llvm::StringRef source, PredicateWork &out) {
  mlir::MLIRContext context;
  auto module = parseModule(context, source);
  if (!module)
    return false;
  SelectSimdSimtCostModelPassOptions options;
  options.mode = "report";
  options.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
  options.actualTarget = "Ascend950PR_9579";
  options.numWarps = 4;
  options.compileOn91095 = true;
  if (!runPasses(*module, createSelectSimdSimtCostModelPass(options)))
    return false;
  auto report =
      (*module)->getAttrOfType<StringAttr>("ascend.simt_costmodel.report_json");
  if (!report)
    return false;
  auto parsed = llvm::json::parse(report.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return false;
  }
  auto *root = parsed->getAsObject();
  auto *model = root ? root->getObject("stage_model") : nullptr;
  auto *stages = model ? model->getArray("logical_stages") : nullptr;
  if (!stages)
    return false;
  for (const auto &value : *stages) {
    const auto *stage = value.getAsObject();
    const auto *work = stage->getObject("workload");
    out.elements +=
        work->getNumber("predicate_elements_per_iteration").value_or(0);
    const auto *ops = work->getObject("operation_elements_per_iteration");
    out.selects += ops->getNumber("predicate.select").value_or(0);
    out.generic += ops->getNumber("generic.issue").value_or(0);
    for (const auto &tensor : *work->getArray("tensor_operation_workloads")) {
      const auto *group = tensor.getAsObject();
      const double projectedElements =
          group->getNumber("logical_elements_per_iteration").value_or(0);
      switch (group->getInteger("simd_predicate_bit_width").value_or(0)) {
      case 16:
        out.projected16Elements += projectedElements;
        break;
      case 32:
        out.projected32Elements += projectedElements;
        break;
      }
      if (group->getString("operation")
              .value_or("")
              .starts_with("predicate.") &&
          group->getString("operation") != "predicate.select") {
        const double elements =
            group->getNumber("logical_elements_per_iteration").value_or(0);
        out.describedElements += elements;
        switch (group->getInteger("element_bit_width").value_or(0)) {
        case 1:
          out.maskElements += elements;
          break;
        case 16:
          out.cmp16Elements += elements;
          break;
        case 32:
          out.cmp32Elements += elements;
          break;
        }
      }
    }
  }
  return true;
}

} // namespace

TEST(CostModelPassesTest, PredicateMasksProjectClosedCompareLogicSelectWidths) {
  struct Case {
    const char *first, *second, *selected, *logic;
    int width;
  };
  // Same/mixed widths, all logic opcodes, multi-hop propagation and unsupported
  // widths share one fixture. Instruction rounding is tested in the cost tests.
  for (const auto &c : {Case{"i16", "i16", "i16", "andi", 16},
                        Case{"i32", "i32", "i32", "ori", 32},
                        Case{"f16", "f16", "f16", "xori", 16},
                        Case{"f32", "f32", "f32", "andi", 32},
                        Case{"i16", "i32", "i16", "ori", 32},
                        Case{"i16", "i16", "i32", "xori", 32},
                        Case{"f16", "f32", "f16", "andi", 32},
                        Case{"i8", "i8", "i8", "andi", 0},
                        Case{"i64", "i64", "i64", "andi", 0}}) {
    SCOPED_TRACE(
        llvm::formatv("{0}/{1}/{2}/{3}", c.first, c.second, c.selected, c.logic)
            .str());
    const char *cmp = c.first[0] == 'f' ? "arith.cmpf olt" : "arith.cmpi slt";
    const auto source =
        llvm::formatv(R"mlir(
module {{ func.func @main(%a: tensor<128x{0}>, %b: tensor<128x{0}>,
    %c: tensor<128x{1}>, %d: tensor<128x{1}>,
    %e: tensor<128x{0}>, %f: tensor<128x{0}>,
    %x: tensor<128x{2}>, %y: tensor<128x{2}>) -> tensor<128x{2}> {{
  %p = {3}, %a, %b : tensor<128x{0}>
  %q = {3}, %c, %d : tensor<128x{1}>
  %r = {3}, %e, %f : tensor<128x{0}>
  %m = arith.{4} %p, %q : tensor<128xi1>
  %n = arith.xori %m, %r : tensor<128xi1>
  %s = arith.select %n, %x, %y : tensor<128xi1>, tensor<128x{2}>
  return %s : tensor<128x{2}>
} })mlir",
                      c.first, c.second, c.selected, cmp, c.logic)
            .str();
    PredicateWork work;
    ASSERT_TRUE(collectPredicateWork(source, work));
    EXPECT_DOUBLE_EQ(work.maskElements, 2 * 128);
    EXPECT_DOUBLE_EQ(work.cmp16Elements,
                     (2 * llvm::StringRef(c.first).ends_with("16") +
                      llvm::StringRef(c.second).ends_with("16")) *
                         128);
    EXPECT_DOUBLE_EQ(work.elements, 5 * 128);
    EXPECT_DOUBLE_EQ(work.selects, 128);
    EXPECT_DOUBLE_EQ(work.projected16Elements, c.width == 16 ? 6 * 128 : 0);
    EXPECT_DOUBLE_EQ(work.projected32Elements, c.width == 32 ? 6 * 128 : 0);
  }
}

TEST(CostModelPassesTest, PredicateMasksKeepUnsupportedGroupsUnprojected) {
  const std::string prefix = R"mlir(
module {
  func.func private @consume(tensor<128xi32>)
  func.func @main(%a: tensor<128xi16>, %b: tensor<128xi16>,
                  %x: tensor<128xi32>, %y: tensor<128xi32>,
                  %loaded: tensor<128xi1>) -> tensor<128xi16> {
    %p = arith.cmpi slt, %a, %b : tensor<128xi16>
    %q = arith.cmpi eq, %a, %b : tensor<128xi16>
    %mask = arith.andi %p, %q : tensor<128xi1>
)mlir";
  for (const char *tail :
       {// A shared mask can cross two backend groups, with transfer costs.
        R"mlir(
    %s = arith.select %mask, %a, %b : tensor<128xi1>, tensor<128xi16>
    %t = arith.select %mask, %x, %y : tensor<128xi1>, tensor<128xi32>
    func.call @consume(%t) : (tensor<128xi32>) -> ()
    return %s : tensor<128xi16>
)mlir",
        // Data arithmetic can import a larger fusion group.
        R"mlir(
    %sum = arith.addi %a, %b : tensor<128xi16>
    %s = arith.select %mask, %a, %sum : tensor<128xi1>, tensor<128xi16>
    return %s : tensor<128xi16>
)mlir",
        // Loaded/argument masks need not use predicate-register logic.
        R"mlir(
    %m = arith.ori %mask, %loaded : tensor<128xi1>
    %s = arith.select %m, %a, %b : tensor<128xi1>, tensor<128xi16>
    return %s : tensor<128xi16>
)mlir"}) {
    PredicateWork work;
    ASSERT_TRUE(collectPredicateWork(prefix + tail + " } }", work));
    EXPECT_GT(work.elements, 0);
    EXPECT_DOUBLE_EQ(work.projected16Elements, 0);
    EXPECT_DOUBLE_EQ(work.projected32Elements, 0);
  }
}

TEST(CostModelPassesTest, PredicateMasksKeepDifferentProjectedGroupsSeparate) {
  PredicateWork work;
  ASSERT_TRUE(collectPredicateWork(R"mlir(
module {
  func.func @main(%a: tensor<128xi16>, %b: tensor<128xi16>,
                  %x: tensor<128xi32>, %y: tensor<128xi32>)
      -> (tensor<128xi16>, tensor<128xi32>) {
    %p = arith.cmpi slt, %a, %b : tensor<128xi16>
    %q = arith.cmpi eq, %a, %b : tensor<128xi16>
    %m = arith.andi %p, %q : tensor<128xi1>
    %s = arith.select %m, %a, %b : tensor<128xi1>, tensor<128xi16>
    %p2 = arith.cmpi sgt, %a, %b : tensor<128xi16>
    %q2 = arith.cmpi ne, %a, %b : tensor<128xi16>
    %m2 = arith.andi %p2, %q2 : tensor<128xi1>
    %s2 = arith.select %m2, %x, %y : tensor<128xi1>, tensor<128xi32>
    return %s, %s2 : tensor<128xi16>, tensor<128xi32>
  }
})mlir",
                                   work));
  EXPECT_DOUBLE_EQ(work.maskElements, 256);
  EXPECT_DOUBLE_EQ(work.cmp16Elements, 512);
  EXPECT_DOUBLE_EQ(work.projected16Elements, 512);
  EXPECT_DOUBLE_EQ(work.projected32Elements, 512);
}

TEST(CostModelPassesTest, PredicateMasksCountI1LogicAndSelectSeparately) {
  PredicateWork work;
  ASSERT_TRUE(collectPredicateWork(R"mlir(
module {
  func.func @main(%a: tensor<64xi32>, %b: tensor<64xi32>)
      -> (tensor<64xi1>, tensor<64xi1>, tensor<64xi32>) {
    %p = arith.cmpi slt, %a, %b : tensor<64xi32>
    %q = arith.cmpi eq, %a, %b : tensor<64xi32>
    %and = arith.andi %p, %q : tensor<64xi1>
    %or = arith.ori %p, %q : tensor<64xi1>
    %xor = arith.xori %p, %q : tensor<64xi1>
    %sel = arith.select %xor, %a, %b : tensor<64xi1>, tensor<64xi32>
    return %and, %or, %sel : tensor<64xi1>, tensor<64xi1>, tensor<64xi32>
  }
})mlir",
                                   work));
  EXPECT_DOUBLE_EQ(work.elements, 5 * 64);
  EXPECT_DOUBLE_EQ(work.describedElements, work.elements);
  EXPECT_DOUBLE_EQ(work.selects, 64);
  EXPECT_DOUBLE_EQ(work.generic, 0);
}

TEST(CostModelPassesTest, PredicateMasksDoNotChargeIntegerBitwiseAsPredicates) {
  PredicateWork work;
  ASSERT_TRUE(collectPredicateWork(R"mlir(
module {
  func.func @main(%a: tensor<64xi32>, %b: tensor<64xi32>)
      -> (tensor<64xi32>, tensor<64xi32>, tensor<64xi32>) {
    %and = arith.andi %a, %b : tensor<64xi32>
    %or = arith.ori %a, %b : tensor<64xi32>
    %xor = arith.xori %a, %b : tensor<64xi32>
    return %and, %or, %xor : tensor<64xi32>, tensor<64xi32>, tensor<64xi32>
  }
})mlir",
                                   work));
  EXPECT_DOUBLE_EQ(work.elements, 0);
  EXPECT_DOUBLE_EQ(work.describedElements, 0);
}

TEST(CostModelPassesTest, PredicateMasksCountXorWithoutAssumingFusion) {
  for (bool floating : {false, true})
    for (bool invert : {false, true})
      for (bool commute : {false, true})
        for (int sharing : {0, 1, 2}) {
          SCOPED_TRACE(floating);
          SCOPED_TRACE(invert);
          SCOPED_TRACE(commute);
          SCOPED_TRACE(sharing); // single NOT, shared compare, shared NOT.
          const std::string data =
              floating ? "tensor<64xf32>" : "tensor<64xi32>";
          const std::string types = sharing ? data + ", " + data : data;
          const std::string extra =
              sharing ? " %s2 = arith.select " +
                            std::string(sharing == 1 ? "%p" : "%n") +
                            ", %b, %a : tensor<64xi1>, " + data + "\n"
                      : "";
          PredicateWork work;
          ASSERT_TRUE(collectPredicateWork(
              "module { func.func @main(%a: " + data + ", %b: " + data +
                  ") -> (" + types +
                  ") {\n"
                  " %c = arith.constant dense<" +
                  (invert ? "true" : "false") +
                  "> : tensor<64xi1>\n"
                  " %p = " +
                  (floating ? "arith.cmpf olt, " : "arith.cmpi slt, ") +
                  "%a, %b : " + data +
                  "\n"
                  " %n = arith.xori " +
                  (commute ? "%c, %p" : "%p, %c") +
                  " : tensor<64xi1>\n"
                  " %s = arith.select %n, %a, %b : tensor<64xi1>, " +
                  data + "\n" + extra + " return " +
                  (sharing ? "%s, %s2" : "%s") + " : " + types + "\n} }",
              work));
          // The IR contains a compare and an XOR regardless of constants,
          // sharing or whether a backend could absorb XOR into the select.
          EXPECT_DOUBLE_EQ(work.elements, 2 * 64);
          EXPECT_DOUBLE_EQ(work.describedElements, work.elements);
          EXPECT_DOUBLE_EQ(work.selects, (sharing ? 2 : 1) * 64);
          EXPECT_DOUBLE_EQ(work.projected16Elements, 0);
          EXPECT_DOUBLE_EQ(work.projected32Elements, 0);
        }
  // Returning the mask uses the same accounting as a select consumer.
  PredicateWork returnedMask;
  ASSERT_TRUE(collectPredicateWork(R"mlir(
module { func.func @main(%a: tensor<64xi32>, %b: tensor<64xi32>) -> tensor<64xi1> {
  %c = arith.constant dense<true> : tensor<64xi1>
  %p = arith.cmpi slt, %a, %b : tensor<64xi32>
  %n = arith.xori %p, %c : tensor<64xi1>
  return %n : tensor<64xi1>
} })mlir",
                                   returnedMask));
  EXPECT_DOUBLE_EQ(returnedMask.elements, 2 * 64);
  EXPECT_DOUBLE_EQ(returnedMask.describedElements, returnedMask.elements);
  EXPECT_DOUBLE_EQ(returnedMask.selects, 0);
}

TEST(CostModelPassesTest, PredicateMasksPreserveSourceWidthsInWorkload) {
  PredicateWork work;
  ASSERT_TRUE(collectPredicateWork(R"mlir(
module {
  func.func @main(%a: tensor<128xi32>, %b: tensor<128xi32>,
                  %h: tensor<128xi16>, %k: tensor<128xi16>) -> tensor<128xi32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%state = %a) -> tensor<128xi32> {
      %p = arith.cmpi slt, %state, %b : tensor<128xi32>
      %q = arith.cmpi slt, %h, %k : tensor<128xi16>
      %m = arith.andi %p, %q : tensor<128xi1>
      %s = arith.select %m, %state, %b : tensor<128xi1>, tensor<128xi32>
      scf.yield %s : tensor<128xi32>
    }
    return %result : tensor<128xi32>
  }
})mlir",
                                   work));
  EXPECT_DOUBLE_EQ(work.elements, 3 * 128);
  EXPECT_DOUBLE_EQ(work.describedElements, work.elements);
  // These are source IR widths, not backend fusion-group lane counts.
  // A fused i16 comparison can use the i32 consumer's narrower lane count.
  EXPECT_DOUBLE_EQ(work.cmp32Elements, 128);
  EXPECT_DOUBLE_EQ(work.cmp16Elements, 128);
  EXPECT_DOUBLE_EQ(work.maskElements, 128);
  EXPECT_DOUBLE_EQ(work.selects, 128);
  // Loop-carried masks and data cross the closed-island boundary.
  EXPECT_DOUBLE_EQ(work.projected16Elements, 0);
  EXPECT_DOUBLE_EQ(work.projected32Elements, 0);
}

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
  EXPECT_EQ(plan.materializableRoots().size(), 1u);
  EXPECT_EQ(mlir::ascend::stringifySimtAnchorKind(plan.anchors[0].kind),
            "loaded_index_dependent_memory");

  auto features = analyzeSimdSimtFeatures(*module, plan);
  if (!features)
    FAIL() << llvm::toString(features.takeError());

  EXPECT_EQ(features->simtAnchors.count, 1);
}

TEST(CostModelPassesTest,
     LoopDependencyAnalysisSeparatesDataStateFromPointerInduction) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%base: tensor<4xi64>, %initial: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %one = arith.constant 1 : i32
    %step = arith.constant dense<1> : tensor<4xi64>
    %ptr_result = scf.for %i = %c0 to %c4 step %c1
        iter_args(%ptr = %base) -> tensor<4xi64> {
      %loaded = "tt.load"(%ptr) : (tensor<4xi64>) -> tensor<4xf32>
      %next = "tt.addptr"(%ptr, %step)
        : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
      scf.yield %next : tensor<4xi64>
    }
    %sum = scf.for %i = %c0 to %c4 step %c1
        iter_args(%acc = %initial) -> i32 {
      %next = arith.addi %acc, %one : i32
      scf.yield %next : i32
    }
    return %sum : i32
  }
}
)mlir");
  ASSERT_TRUE(module);

  // Loop dependency classification is owned by StageFeatureAnalysis; the
  // kernel summary deliberately no longer duplicates it.
  auto plan = buildMixedSimtAnchorPlan(*module, /*compileOn91095=*/true);
  EXPECT_TRUE(plan.anchors.empty());
}

TEST(CostModelPassesTest, SimtAnchorAnalysisClassifiesHistogramLowerability) {
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
  EXPECT_FALSE(anchor.lowerability.allSimd);
  EXPECT_FALSE(anchor.lowerability.allSimtOnly);
  EXPECT_TRUE(anchor.lowerability.mixed);
  EXPECT_TRUE(anchor.materializable);
  EXPECT_FALSE(plan.kernelLowerability.allSimd);
  EXPECT_FALSE(plan.kernelLowerability.allSimtOnly);
  EXPECT_TRUE(plan.kernelLowerability.mixed);
}

TEST(CostModelPassesTest, SimtAnchorAnalysisClassifiesPlainCumsumLowerability) {
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
  EXPECT_TRUE(anchor.lowerability.allSimd);
  EXPECT_TRUE(anchor.lowerability.allSimtOnly);
  EXPECT_TRUE(anchor.lowerability.mixed);
  EXPECT_TRUE(anchor.materializable);
  EXPECT_TRUE(plan.kernelLowerability.allSimd);
  EXPECT_TRUE(plan.kernelLowerability.allSimtOnly);
  EXPECT_TRUE(plan.kernelLowerability.mixed);
}

TEST(CostModelPassesTest,
     SimtAnchorAnalysisClassifiesTensorAtomicLowerability) {
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
  EXPECT_TRUE(anchor.lowerability.allSimd);
  EXPECT_TRUE(anchor.lowerability.allSimtOnly);
  EXPECT_TRUE(anchor.lowerability.mixed);
  EXPECT_TRUE(anchor.materializable);
  EXPECT_TRUE(plan.kernelLowerability.allSimd);
  EXPECT_TRUE(plan.kernelLowerability.allSimtOnly);
  EXPECT_TRUE(plan.kernelLowerability.mixed);
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
    %dot = "tt.dot"(%b, %b)
      : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %dot : tensor<16x16xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  auto plan = buildMixedSimtAnchorPlan(*module, /*compileOn91095=*/true);
  ASSERT_EQ(plan.anchors.size(), 1u);
  EXPECT_EQ(plan.materializableRoots().size(), 1u);
  for (const auto &anchor : plan.anchors) {
    EXPECT_EQ(anchor.kind, mlir::ascend::SimtAnchorKind::TriangularSolveLoop);
    EXPECT_TRUE(anchor.lowerability.mixed);
    EXPECT_TRUE(anchor.materializable);
    // Both recurrence loops are one physical SIMT scope and therefore one
    // scored/materialized anchor, not two independent route decisions.
    EXPECT_GE(anchor.scopeOperations.size(), 2u);
    EXPECT_EQ(anchor.scopeInsertionPoint, anchor.operation);
    EXPECT_EQ(anchor.scopeOperations.front()->getName().getStringRef(),
              "tt.make_range");
    ASSERT_TRUE(anchor.triangularSolve);
    EXPECT_EQ(anchor.triangularSolve->blockRows, 16);
    EXPECT_EQ(anchor.triangularSolve->blockColumns, 16);
    EXPECT_EQ(anchor.triangularSolve->accumulatorType, "f32");
    EXPECT_EQ(anchor.triangularSolve->recurrenceStartRow, 2);
    // Two recurrence loops, each with 14 body iterations.
    EXPECT_EQ(anchor.triangularSolve->recurrenceLoopCount, 28);
    EXPECT_EQ(anchor.triangularSolve->denseDotTailOps, 1);
    EXPECT_TRUE(anchor.triangularSolve->requiresCubeTailPartition);
    EXPECT_TRUE(anchor.lowerability.allSimtOnly);
  }
  EXPECT_TRUE(plan.kernelLowerability.allSimtOnly);
  EXPECT_TRUE(plan.kernelLowerability.mixed);

  auto features = analyzeSimdSimtFeatures(*module, plan);
  if (!features)
    FAIL() << llvm::toString(features.takeError());
  EXPECT_EQ(features->simtAnchors.count, 1);
  ASSERT_TRUE(mlir::succeeded(materializeSimtAnchorPlan(*module, plan, 4)));
  Operation *scope = findFirstOp(*module, "scope.scope");
  ASSERT_NE(scope, nullptr);
  EXPECT_EQ(scope->getAttrOfType<mlir::StringAttr>("vector_mode").getValue(),
            "simt");
  EXPECT_EQ(
      scope->getAttrOfType<mlir::IntegerAttr>("ascend.scope_superblock.factor")
          .getInt(),
      4);
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

TEST(CostModelPassesTest, SimdSimtScoresGenericSemanticStages) {
  auto configureOptions = [](SelectSimdSimtCostModelPassOptions &options,
                             llvm::StringRef mode) {
    options.mode = mode.str();
    options.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
    options.actualTarget = "Ascend950PR_9579";
    options.numWarps = 4;
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
  EXPECT_NE(autoEffective.getValue(), "backend_default");
  EXPECT_EQ(autoEffective.getValue(), autoRecommended.getValue());
  EXPECT_TRUE((*autoModule)->hasAttr("ascend.simt_costmodel.all_simd_score"));
  auto autoJSON = llvm::json::parse(autoReport.getValue());
  ASSERT_TRUE(static_cast<bool>(autoJSON));
  auto *autoObject = autoJSON->getAsObject();
  ASSERT_NE(autoObject, nullptr);
  auto autoDecision = autoObject->getString("decision_kind");
  ASSERT_TRUE(autoDecision);
  EXPECT_NE(*autoDecision, "backend_default");
  auto autoReason = autoObject->getString("application_reason");
  ASSERT_TRUE(autoReason);
  EXPECT_EQ(*autoReason, "minimum_cost_candidate");

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
  auto reportDecision = reportObject->getString("decision_kind");
  ASSERT_TRUE(reportDecision);
  EXPECT_NE(*reportDecision, "backend_default");
  auto reportReason = reportObject->getString("application_reason");
  ASSERT_TRUE(reportReason);
  EXPECT_EQ(*reportReason, "report_mode");
}

TEST(CostModelPassesTest, SimdSimtSelectionUsesExternalAnalysisIR) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kOutOfSimdSimtCoverageModule);
  ASSERT_TRUE(module);

  llvm::SmallString<128> analysisPath;
  int analysisFd = -1;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "simd_simt_v1_analysis", "mlir", analysisFd, analysisPath));
  {
    llvm::raw_fd_ostream analysisFile(analysisFd, true);
    analysisFile << R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
      attributes {ta.auto_blockify_v1} {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";
  }

  SelectSimdSimtCostModelPassOptions options;
  options.mode = "report";
  options.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
  options.actualTarget = "Ascend950PR_9579";
  options.numWarps = 4;
  options.compileOn91095 = true;
  options.analysisModulePath = analysisPath.str().str();
  const bool succeeded =
      runPasses(*module, createSelectSimdSimtCostModelPass(options));
  llvm::sys::fs::remove(analysisPath);
  ASSERT_TRUE(succeeded);

  auto reportAttr =
      (*module)->getAttrOfType<StringAttr>("ascend.simt_costmodel.report_json");
  ASSERT_TRUE(reportAttr);
  auto report = llvm::json::parse(reportAttr.getValue());
  ASSERT_TRUE(static_cast<bool>(report));
  auto *object = report->getAsObject();
  ASSERT_NE(object, nullptr);
  auto analysisSource = object->getString("analysis_ir_source");
  ASSERT_TRUE(analysisSource);
  EXPECT_EQ(*analysisSource, "post_auto_blockify_v1_ttir");
  auto *features = object->getObject("features");
  ASSERT_NE(features, nullptr);
  auto *postTransform = features->getObject("post_transform");
  ASSERT_NE(postTransform, nullptr);
  auto v1Applied = postTransform->getBoolean("auto_blockify_v1_applied");
  ASSERT_TRUE(v1Applied);
  EXPECT_TRUE(*v1Applied);
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

TEST(CostModelPassesTest, SameStageAnchorsMaterializeAsOneCompoundScope) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg1 : i32
    %2 = arith.addi %1, %arg0 : i32
    %3 = arith.addi %0, %2 : i32
    return %3 : i32
  }
}
)mlir");
  ASSERT_TRUE(module);

  Operation *first = findFirstOp(*module, "arith.muli")->getPrevNode();
  Operation *middle = findFirstOp(*module, "arith.muli");
  Operation *second = middle->getNextNode();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(second, nullptr);

  mlir::ascend::SimtAnchorPlan plan;
  for (Operation *operation : {first, second}) {
    mlir::ascend::SimtAnchorDescriptor anchor;
    anchor.operation = operation;
    anchor.scopeOperations.push_back(operation);
    anchor.scopeInsertionPoint = operation;
    anchor.kind = mlir::ascend::SimtAnchorKind::LoadedIndexDependentMemory;
    anchor.materializable = true;
    plan.anchors.push_back(std::move(anchor));
  }

  auto merged = mlir::ascend::mergeSimtStageAnchors(plan, {0, 1});
  ASSERT_TRUE(merged);
  ASSERT_EQ(merged->scopeOperations.size(), 3u);
  EXPECT_EQ(merged->scopeOperations[0], first);
  EXPECT_EQ(merged->scopeOperations[1], middle);
  EXPECT_EQ(merged->scopeOperations[2], second);

  mlir::ascend::SimtAnchorPlan selected;
  selected.anchors.push_back(std::move(*merged));
  ASSERT_TRUE(mlir::succeeded(materializeSimtAnchorPlan(*module, selected, 2)));
  Operation *scope = findFirstOp(*module, "scope.scope");
  ASSERT_NE(scope, nullptr);
  EXPECT_EQ(
      scope->getAttrOfType<mlir::IntegerAttr>("ascend.scope_superblock.factor")
          .getInt(),
      2);
  EXPECT_EQ(middle->getParentOp(), scope);
  int64_t scopeCount = 0;
  module->walk([&](Operation *operation) {
    scopeCount += operation->getName().getStringRef() == "scope.scope";
  });
  EXPECT_EQ(scopeCount, 1);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
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
