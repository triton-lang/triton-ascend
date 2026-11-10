/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "PtrAnalysisTestUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include <tuple>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;

namespace {
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      tensor::TensorDialect, triton::TritonDialect>();
}

Operation *tagged(Operation *scope, StringRef name) {
  Operation *found = nullptr;
  scope->walk([&](Operation *op) {
    if (auto tag = op->getAttrOfType<StringAttr>("tag");
        tag && tag.getValue() == name)
      found = op;
  });
  return found;
}

void expectScalar(OpFoldResult scalar, int64_t expected,
                  const test::Environment &env) {
  auto value = test::evaluate(scalar, env);
  ASSERT_TRUE(value);
  ASSERT_EQ(value->size(), 1u);
  EXPECT_EQ(value->front().getSExtValue(), expected);
}

void expectKinds(PtrAnalysis &analysis, Value value,
                 ArrayRef<AxisKind> expected) {
  auto info = analysis.analyze(value);
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->offsets.valueType, value.getType());
  EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
  EXPECT_EQ(test::axisKinds(info->offsets), llvm::to_vector(expected));
}

void expectConstantStrides(PtrAnalysis &analysis, Value value,
                           ArrayRef<int64_t> expected) {
  auto info = analysis.analyze(value);
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
  ASSERT_EQ(info->offsets.axes.size(), expected.size());
  for (auto [axis, stride] : llvm::zip_equal(info->offsets.axes, expected)) {
    auto constant =
        mlir::triton::ptr::detail::getIntegerConstant(axis.stride, 64);
    ASSERT_TRUE(succeeded(constant));
    EXPECT_EQ(constant->getSExtValue(), stride);
  }
}

TEST(CommonPtrSCF, ForPreservesConstantStridesAcrossUniformUpdates) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %negative = arith.constant dense<-2> : tensor<4xi32>
      %reverse = arith.muli %range, %negative : tensor<4xi32>
      %step = arith.constant dense<8> : tensor<4xi32>
      %out:3 = scf.for %i = %c0 to %n step %c1
          iter_args(%x = %range, %y = %reverse, %u = %step)
          -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
        %next_x = arith.addi %x, %step : tensor<4xi32>
        %next_y = arith.subi %y, %step : tensor<4xi32>
        %next_u = arith.addi %u, %step : tensor<4xi32>
        scf.yield %next_x, %next_y, %next_u : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  const int64_t expectedStrides[] = {1, -2, 0};
  for (auto [i, stride] : llvm::enumerate(expectedStrides)) {
    expectConstantStrides(analysis, loop.getRegionIterArgs()[i], {stride});
    expectConstantStrides(analysis, loop.getResult(i), {stride});
  }
  // A proved stride needs no unit-coordinate sample. Origins still refer to
  // the current SSA, including the zero-trip loop result.
  module->walk([&](tensor::ExtractOp extract) {
    for (Value index : extract.getIndices()) {
      auto constant = mlir::triton::ptr::detail::getIntegerConstant(index, 64);
      ASSERT_TRUE(succeeded(constant));
      EXPECT_TRUE(constant->isZero());
    }
  });
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, ConstantStrideLossPropagatesBetweenCarriers) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %two = arith.constant dense<2> : tensor<4xi32>
      %out:2 = scf.for %i = %c0 to %n step %c1
          iter_args(%x = %range, %y = %range) -> (tensor<4xi32>, tensor<4xi32>) {
        %next_y = arith.muli %y, %two : tensor<4xi32>
        scf.yield %y, %next_y : tensor<4xi32>, tensor<4xi32>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  // x still sees stride 1 during the first round, while y loses its constant.
  // A second round is required although both axes remain Structured.
  for (Value value :
       {Value(loop.getRegionIterArgs()[0]), Value(loop.getRegionIterArgs()[1]),
        loop.getResult(0), loop.getResult(1)}) {
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->offsets.axes[0].kind, AxisKind::Structured);
    EXPECT_TRUE(failed(mlir::triton::ptr::detail::getIntegerConstant(
        info->offsets.axes[0].stride, 64)));
    test::Environment env;
    for (int64_t lane : {3, 7, 11, 15})
      env[value].emplace_back(32, lane);
    expectScalar(info->offsets.axes[0].stride, 4, env);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, WhilePreservesDistinctBeforeAndAfterConstantStrides) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%condition: i1) {
      %zero = arith.constant dense<0> : tensor<4xi32>
      %two = arith.constant dense<2> : tensor<4xi32>
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %out = scf.while (%x = %range, %u = %zero)
          : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32> {
        %scaled = arith.muli %x, %two : tensor<4xi32>
        scf.condition(%condition) %scaled : tensor<4xi32>
      } do {
      ^bb0(%y: tensor<4xi32>):
        scf.yield %range, %zero : tensor<4xi32>, tensor<4xi32>
      } attributes {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto loop = cast<scf::WhileOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  expectConstantStrides(analysis, loop.getBeforeArguments()[0], {1});
  expectConstantStrides(analysis, loop.getBeforeArguments()[1], {0});
  expectConstantStrides(analysis, loop.getAfterArguments()[0], {2});
  expectConstantStrides(analysis, loop.getResult(0), {2});
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, WhileConstantStrideLossOnEitherEdgePropagates) {
  for (bool changeCondition : {false, true}) {
    MLIRContext context;
    load(context);
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @loop(%condition: i1) {
        %one = arith.constant dense<1> : tensor<4xi32>
        %two = arith.constant dense<2> : tensor<4xi32>
        %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
        %out = scf.while (%x = %range) : (tensor<4xi32>) -> tensor<4xi32> {
          %condition_value = arith.muli %x, %one {tag="condition_scale"} : tensor<4xi32>
          scf.condition(%condition) %condition_value : tensor<4xi32>
        } do {
        ^bb0(%y: tensor<4xi32>):
          %next = arith.muli %y, %one {tag="backedge_scale"} : tensor<4xi32>
          scf.yield %next : tensor<4xi32>
        } attributes {tag="loop"}
        tt.return
      } }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto loop = cast<scf::WhileOp>(tagged(*module, "loop"));
    auto constants = loop->getParentOfType<triton::FuncOp>()
                         .getBody()
                         .front()
                         .getOps<arith::ConstantOp>();
    arith::ConstantOp two = *std::next(constants.begin());
    tagged(*module, changeCondition ? "condition_scale" : "backedge_scale")
        ->setOperand(1, two.getResult());
    ASSERT_TRUE(succeeded(verify(*module)));
    OpBuilder builder(loop);
    PtrAnalysis analysis(builder);
    for (Value value :
         {Value(loop.getBeforeArguments()[0]),
          Value(loop.getAfterArguments()[0]), loop.getResult(0)}) {
      auto info = analysis.analyze(value);
      ASSERT_TRUE(succeeded(info));
      EXPECT_EQ(info->offsets.axes[0].kind, AxisKind::Structured);
      EXPECT_TRUE(failed(mlir::triton::ptr::detail::getIntegerConstant(
          info->offsets.axes[0].stride, 64)));
    }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(CommonPtrSCF, NestedLoopsDiscardInvalidatedConstantStrideHypotheses) {
  for (bool innerFirst : {false, true}) {
    MLIRContext context;
    load(context);
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @loop(%n: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
        %one = arith.constant dense<1> : tensor<4xi32>
        %two = arith.constant dense<2> : tensor<4xi32>
        %outer = scf.for %i = %c0 to %n step %c1
            iter_args(%x = %range) -> tensor<4xi32> {
          %inner = scf.for %j = %c0 to %n step %c1
              iter_args(%y = %x) -> tensor<4xi32> {
            %next_y = arith.addi %y, %one : tensor<4xi32>
            scf.yield %next_y : tensor<4xi32>
          } {tag="inner"}
          %next_x = arith.muli %inner, %two : tensor<4xi32>
          scf.yield %next_x : tensor<4xi32>
        } {tag="outer"}
        tt.return
      } }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto outer = cast<scf::ForOp>(tagged(*module, "outer"));
    auto inner = cast<scf::ForOp>(tagged(*module, "inner"));
    OpBuilder builder(outer);
    PtrAnalysis analysis(builder);
    Value first = innerFirst ? inner.getResult(0) : outer.getResult(0);
    ASSERT_TRUE(succeeded(analysis.analyze(first)));
    for (Value value : {Value(outer.getRegionIterArgs()[0]),
                        Value(inner.getRegionIterArgs()[0]), inner.getResult(0),
                        outer.getResult(0)}) {
      auto info = analysis.analyze(value);
      ASSERT_TRUE(succeeded(info));
      EXPECT_EQ(info->offsets.axes[0].kind, AxisKind::Structured);
      EXPECT_TRUE(failed(mlir::triton::ptr::detail::getIntegerConstant(
          info->offsets.axes[0].stride, 64)));
    }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(CommonPtrSCF, ForUsesCurrentFieldsAndPreservesUniformCarrier) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%base: !tt.ptr<i32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant dense<0> : tensor<4xi32>
      %one = arith.constant dense<1> : tensor<4xi32>
      %two = arith.constant dense<2> : tensor<4xi32>
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %p = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %out:2 = scf.for %i = %c0 to %n step %c1 iter_args(%x = %range, %u = %zero) -> (tensor<4xi32>, tensor<4xi32>) {
        %ptr = tt.addptr %p, %x {tag="pointer"} : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        %next = arith.muli %x, %two {tag="next"} : tensor<4xi32>
        %uniform = arith.addi %u, %one : tensor<4xi32>
        scf.yield %next, %uniform : tensor<4xi32>, tensor<4xi32>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  Value x = loop.getRegionIterArgs()[0], u = loop.getRegionIterArgs()[1];
  OpBuilder builder(loop);
  auto ip = builder.saveInsertionPoint();
  PtrAnalysis analysis(builder);
  // Enter through an expression that itself is on the backedge. Its active
  // recursion state must not suppress re-analysis under the local hypotheses.
  auto next = analysis.analyze(tagged(*module, "next")->getResult(0));
  ASSERT_TRUE(succeeded(next));
  EXPECT_TRUE(next->offsets.hasStructuredForm());
  auto pointer = analysis.analyze(tagged(*module, "pointer")->getResult(0));
  ASSERT_TRUE(succeeded(pointer));
  EXPECT_TRUE(pointer->offsets.hasStructuredForm());
  test::Environment env;
  for (int64_t lane : {5, 8, 11, 14})
    env[x].emplace_back(32, lane);
  for (int64_t lane : {7, 13, 19, 25})
    env[loop.getResult(0)].emplace_back(32, lane);
  env[u].assign(4, llvm::APInt(32, 9));
  for (auto [value, origin, stride] :
       {std::tuple<Value, int64_t, int64_t>{x, 5, 3},
        {loop.getResult(0), 7, 6}}) {
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    expectScalar(info->offsets.scalarOffset, origin, env);
    expectScalar(info->offsets.axes[0].stride, stride, env);
  }
  auto uniform = analysis.analyze(u);
  ASSERT_TRUE(succeeded(uniform));
  EXPECT_TRUE(uniform->offsets.isUniform());
  expectScalar(uniform->offsets.scalarOffset, 9, env);
  expectKinds(analysis, loop.getResult(1), {AxisKind::Invariant});
  expectScalar(next->offsets.scalarOffset, 10, env);
  expectScalar(next->offsets.axes[0].stride, 6, env);
  EXPECT_EQ(builder.getInsertionPoint(), ip.getPoint());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, ForJoinsAllCarriersAndRejectsSecondIterationNonlinearity) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant dense<0> : tensor<4xi32>
      %one = arith.constant dense<1> : tensor<4xi32>
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %out:4 = scf.for %i = %c0 to %n step %c1 iter_args(%bad = %zero, %a = %zero, %b = %range, %c = %one) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
        %square = arith.muli %bad, %bad : tensor<4xi32>
        %badnext = arith.addi %square, %range : tensor<4xi32>
        %anext = arith.addi %a, %b : tensor<4xi32>
        %bnext = arith.addi %b, %c : tensor<4xi32>
        %cnext = arith.addi %c, %one : tensor<4xi32>
        scf.yield %badnext, %anext, %bnext, %cnext : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  SmallVector<AxisKind> expected{AxisKind::Unknown, AxisKind::Structured,
                                 AxisKind::Structured, AxisKind::Invariant};
  for (unsigned i = 0; i < expected.size(); ++i) {
    expectKinds(analysis, loop.getRegionIterArgs()[i], {expected[i]});
    expectKinds(analysis, loop.getResult(i), {expected[i]});
  }
  // x0 = [0,0,0,0], x1 = [0,1,2,3], x2 = [0,2,6,12].
  auto bad = analysis.analyze(loop.getRegionIterArgs()[0]);
  ASSERT_TRUE(succeeded(bad));
  EXPECT_FALSE(bad->offsets.scalarOffset);
  EXPECT_FALSE(bad->offsets.hasStructuredDecomposition());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, ForResultIncludesZeroTripInitialValue) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%unknown: tensor<4xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %out = scf.for %i = %c0 to %c0 step %c1 iter_args(%x = %unknown) -> tensor<4xi32> {
        scf.yield %range : tensor<4xi32>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  expectKinds(analysis, loop.getResult(0), {AxisKind::Unknown});
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, IfJoinsAxesAndMaterializesFieldsAfterTheBranches) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @branch(%condition: i1, %scalar: i32) {
      %out:3 = scf.if %condition -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
        %r = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
        %u = tt.splat %scalar : i32 -> tensor<4xi32>
        scf.yield %r, %u, %r : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
      } else {
        %r = tt.make_range {start=4:i32, end=8:i32} : tensor<4xi32>
        %u = arith.constant dense<13> : tensor<4xi32>
        %two = arith.constant dense<2> : tensor<4xi32>
        %scaled = arith.muli %r, %two : tensor<4xi32>
        scf.yield %r, %u, %scaled : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
      } {tag="branch"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto branch = cast<scf::IfOp>(tagged(*module, "branch"));
  OpBuilder builder(branch);
  PtrAnalysis analysis(builder);
  expectKinds(analysis, branch.getResult(0), {AxisKind::Structured});
  expectKinds(analysis, branch.getResult(1), {AxisKind::Invariant});
  test::Environment env;
  for (int64_t lane : {4, 5, 6, 7})
    env[branch.getResult(0)].emplace_back(32, lane);
  env[branch.getResult(1)].assign(4, llvm::APInt(32, 13));
  auto affine = analysis.analyze(branch.getResult(0));
  auto uniform = analysis.analyze(branch.getResult(1));
  ASSERT_TRUE(succeeded(affine) && succeeded(uniform));
  expectScalar(affine->offsets.scalarOffset, 4, env);
  expectScalar(affine->offsets.axes[0].stride, 1, env);
  expectScalar(uniform->offsets.scalarOffset, 13, env);
  expectConstantStrides(analysis, branch.getResult(0), {1});
  expectConstantStrides(analysis, branch.getResult(1), {0});
  auto varying = analysis.analyze(branch.getResult(2));
  ASSERT_TRUE(succeeded(varying));
  EXPECT_EQ(varying->offsets.axes[0].kind, AxisKind::Structured);
  EXPECT_TRUE(failed(mlir::triton::ptr::detail::getIntegerConstant(
      varying->offsets.axes[0].stride, 64)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, WhileKeepsBeforeAndAfterEdgeMappingsSeparate) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%condition: i1) {
      %zero = arith.constant dense<0> : tensor<4xi32>
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %out = scf.while (%a = %zero, %b = %range) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32> {
        scf.condition(%condition) %b : tensor<4xi32>
      } do {
      ^bb0(%c: tensor<4xi32>):
        %a1 = arith.addi %c, %range : tensor<4xi32>
        %b1 = arith.addi %c, %c : tensor<4xi32>
        scf.yield %a1, %b1 : tensor<4xi32>, tensor<4xi32>
      } attributes {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto loop = cast<scf::WhileOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  for (Value arg : loop.getBeforeArguments())
    expectKinds(analysis, arg, {AxisKind::Structured});
  expectKinds(analysis, loop.getAfterArguments()[0], {AxisKind::Structured});
  expectKinds(analysis, loop.getResult(0), {AxisKind::Structured});
  test::Environment env;
  for (int64_t lane : {10, 14, 18, 22})
    env[loop.getResult(0)].emplace_back(32, lane);
  auto result = analysis.analyze(loop.getResult(0));
  ASSERT_TRUE(succeeded(result));
  expectScalar(result->offsets.scalarOffset, 10, env);
  expectScalar(result->offsets.axes[0].stride, 4, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, WhileConditionAndBackedgeBothParticipateInClosure) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%condition: i1) {
      %zero = arith.constant dense<0> : tensor<4xi32>
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %out = scf.while (%a = %zero) : (tensor<4xi32>) -> tensor<4xi32> {
        %c = arith.addi %a, %range : tensor<4xi32>
        scf.condition(%condition) %c : tensor<4xi32>
      } do {
      ^bb0(%b: tensor<4xi32>):
        %square = arith.muli %b, %b : tensor<4xi32>
        scf.yield %square : tensor<4xi32>
      } attributes {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto loop = cast<scf::WhileOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  expectKinds(analysis, loop.getBeforeArguments()[0], {AxisKind::Unknown});
  expectKinds(analysis, loop.getAfterArguments()[0], {AxisKind::Unknown});
  expectKinds(analysis, loop.getResult(0), {AxisKind::Unknown});
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, NestedLoopsAndBranchesUseCurrentOuterHypotheses) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%n: index, %cond: i1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant dense<0> : tensor<4xi32>
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %out = scf.for %i = %c0 to %n step %c1 iter_args(%x = %zero) -> tensor<4xi32> {
        %inner = scf.for %j = %c0 to %n step %c1 iter_args(%y = %x) -> tensor<4xi32> {
          %next = scf.if %cond -> tensor<4xi32> {
            %add = arith.addi %y, %range : tensor<4xi32>
            scf.yield %add : tensor<4xi32>
          } else {
            scf.yield %x : tensor<4xi32>
          }
          scf.yield %next : tensor<4xi32>
        } {tag="inner"}
        scf.yield %inner : tensor<4xi32>
      } {tag="outer"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  for (StringRef tag : {"outer", "inner"}) {
    auto loop = cast<scf::ForOp>(tagged(*module, tag));
    expectKinds(analysis, loop.getRegionIterArgs()[0], {AxisKind::Structured});
    expectKinds(analysis, loop.getResult(0), {AxisKind::Structured});
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, UnitAxesNeverNeedAnOutOfBoundsStrideSample) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%n: index, %cond: i1, %seed: tensor<1x4xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %out = scf.for %i = %c0 to %n step %c1 iter_args(%x = %seed) -> tensor<1x4xi32> {
        %next = scf.if %cond -> tensor<1x4xi32> {
          scf.yield %x : tensor<1x4xi32>
        } else {
          scf.yield %seed : tensor<1x4xi32>
        }
        scf.yield %next : tensor<1x4xi32>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  for (bool structured : {false, true}) {
    PtrAnalysis analysis(builder);
    if (structured) {
      auto seed = buildOffsetInfo(f.getArgument(2), builder);
      ASSERT_TRUE(succeeded(seed));
      seed->scalarOffset = builder.getIndexAttr(0);
      // A unit dimension permits any formal stride, since its only index is 0.
      seed->axes[0] = AxisInfo(AxisKind::Structured, builder.getIndexAttr(99));
      seed->axes[1] = AxisInfo(AxisKind::Structured, builder.getIndexAttr(1));
      ASSERT_TRUE(
          succeeded(analysis.setInfo(f.getArgument(2), PtrInfo{{}, *seed})));
    }
    expectKinds(analysis, loop.getResult(0),
                {AxisKind::Invariant,
                 structured ? AxisKind::Structured : AxisKind::Unknown});
  }
  module->walk([&](tensor::ExtractOp op) {
    auto type = cast<RankedTensorType>(op.getTensor().getType());
    for (auto [i, index] : llvm::enumerate(op.getIndices())) {
      auto coordinate = test::evaluate(index);
      ASSERT_TRUE(coordinate);
      EXPECT_GE(coordinate->front().getSExtValue(), 0);
      EXPECT_LT(coordinate->front().getSExtValue(), type.getDimSize(i));
    }
  });
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(CommonPtrSCF, PartialAxesAndUnsupportedTensorShapesKeepExactValues) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%n: index, %partial: tensor<2x4xi32>, %empty: tensor<0xi32>, %dynamic: tensor<?xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %one = arith.constant dense<1> : tensor<2x4xi32>
      %out:3 = scf.for %i = %c0 to %n step %c1 iter_args(%x = %partial, %e = %empty, %d = %dynamic) -> (tensor<2x4xi32>, tensor<0xi32>, tensor<?xi32>) {
        %next = arith.addi %x, %one : tensor<2x4xi32>
        scf.yield %next, %e, %d : tensor<2x4xi32>, tensor<0xi32>, tensor<?xi32>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  auto partial = buildOffsetInfo(f.getArgument(1), builder);
  ASSERT_TRUE(succeeded(partial));
  partial->axes[1] = AxisInfo(AxisKind::Structured, builder.getIndexAttr(2));
  ASSERT_TRUE(
      succeeded(analysis.setInfo(f.getArgument(1), PtrInfo{{}, *partial})));
  expectKinds(analysis, loop.getRegionIterArgs()[0],
              {AxisKind::Unknown, AxisKind::Structured});
  expectKinds(analysis, loop.getResult(0),
              {AxisKind::Unknown, AxisKind::Structured});
  for (unsigned i : {1u, 2u}) {
    auto info = analysis.analyze(loop.getRegionIterArgs()[i]);
    ASSERT_TRUE(succeeded(info));
    EXPECT_FALSE(info->offsets.scalarOffset);
    expectKinds(analysis, loop.getResult(i), {AxisKind::Unknown});
  }
  test::Environment env;
  for (int64_t lane : {9, 11, 13, 15, 20, 22, 24, 26})
    env[loop.getRegionIterArgs()[0]].emplace_back(32, lane);
  auto info = analysis.analyze(loop.getRegionIterArgs()[0]);
  ASSERT_TRUE(succeeded(info));
  expectScalar(info->offsets.scalarOffset, 9, env);
  expectScalar(info->offsets.axes[1].stride, 2, env);
  auto knownStride = mlir::triton::ptr::detail::getIntegerConstant(
      info->offsets.axes[1].stride, 64);
  ASSERT_TRUE(succeeded(knownStride));
  EXPECT_EQ(knownStride->getSExtValue(), 2);
  // Unknown's placeholder zero must not become a constant-stride proof.
  EXPECT_EQ(info->offsets.axes[0].kind, AxisKind::Unknown);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(CommonPtrSCF, InnerFirstQueryCanReenterTheSameLoopUnderOuterHypotheses) {
  for (bool argumentFirst : {false, true}) {
    MLIRContext context;
    load(context);
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @reentry(%n: index, %a: i32, %b: i32) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %zero = arith.constant dense<0> : tensor<4xi32>
        %range = tt.make_range {start=0:i32,end=4:i32} : tensor<4xi32>
        %scale = arith.muli %a, %b {tag="scale"} : i32
        %outer = scf.for %i = %c0 to %n step %c1 iter_args(%x = %zero) -> (tensor<4xi32>) {
          %inner = scf.for %j = %c0 to %n step %c1 iter_args(%y = %zero) -> (tensor<4xi32>) {
            %factor = tt.splat %scale : i32 -> tensor<4xi32>
            %sum = arith.addi %y, %x : tensor<4xi32>
            %next = arith.muli %sum, %factor {tag="next"} : tensor<4xi32>
            scf.yield %next : tensor<4xi32>
          } {tag="inner"}
          %square = arith.muli %inner, %inner : tensor<4xi32>
          %next = arith.addi %square, %range : tensor<4xi32>
          scf.yield %next : tensor<4xi32>
        } {tag="outer"}
        tt.return
      } }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    auto inner = cast<scf::ForOp>(tagged(*module, "inner"));
    auto outer = cast<scf::ForOp>(tagged(*module, "outer"));
    Value next = tagged(*module, "next")->getResult(0);
    Value scale = tagged(*module, "scale")->getResult(0);
    OpBuilder builder(inner);
    PtrAnalysis analysis(builder);
    expectKinds(analysis,
                argumentFirst ? Value(inner.getRegionIterArgs()[0]) : next,
                {AxisKind::Unknown});
    // Inner results computed under an outer candidate stay in that round.
    EXPECT_FALSE(analysis.findCached(inner.getResult(0)));
    expectKinds(analysis, outer.getResult(0), {AxisKind::Unknown});
    expectKinds(analysis, inner.getResult(0), {AxisKind::Unknown});
    ASSERT_TRUE(analysis.findCached(scale));
    OpFoldResult origin = analysis.findCached(scale)->offsets.scalarOffset;
    auto reused = analysis.analyze(scale);
    ASSERT_TRUE(succeeded(reused));
    EXPECT_EQ(reused->offsets.scalarOffset, origin);
    unsigned casts = 0;
    module->walk([&](arith::IndexCastOp op) { casts += op.getIn() == scale; });
    EXPECT_EQ(casts, 1u);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(CommonPtrSCF, InnerWhileBootstrapCanReenterUnderOuterHypotheses) {
  for (bool afterFirst : {true, false}) {
    MLIRContext context;
    load(context);
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @reentry(%n: index, %cond: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %zero = arith.constant dense<0> : tensor<4xi32>
        %range = tt.make_range {start=0:i32,end=4:i32} : tensor<4xi32>
        %outer = scf.for %i = %c0 to %n step %c1 iter_args(%x = %zero) -> (tensor<4xi32>) {
          %inner = scf.while (%before = %zero) : (tensor<4xi32>) -> tensor<4xi32> {
            %next = arith.addi %before, %x : tensor<4xi32>
            scf.condition(%cond) %next : tensor<4xi32>
          } do {
          ^bb0(%after: tensor<4xi32>):
            scf.yield %after : tensor<4xi32>
          } attributes {tag="inner"}
          %square = arith.muli %inner, %inner : tensor<4xi32>
          %next = arith.addi %square, %range : tensor<4xi32>
          scf.yield %next : tensor<4xi32>
        } {tag="outer"}
        tt.return
      } }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    auto inner = cast<scf::WhileOp>(tagged(*module, "inner"));
    auto outer = cast<scf::ForOp>(tagged(*module, "outer"));
    OpBuilder builder(inner);
    PtrAnalysis analysis(builder);
    Value first = afterFirst ? inner.getAfterArguments()[0]
                             : inner.getBeforeArguments()[0];
    expectKinds(analysis, first, {AxisKind::Unknown});
    EXPECT_FALSE(analysis.findCached(inner.getResult(0)));
    expectKinds(analysis, inner.getResult(0), {AxisKind::Unknown});
    expectKinds(analysis, outer.getResult(0), {AxisKind::Unknown});
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(CommonPtrSCF, AProvidedBodyFactRemainsVisibleUnderLoopHypotheses) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @provided(%n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %range = tt.make_range {start=0:i32,end=4:i32} : tensor<4xi32>
      %zero = arith.constant dense<0> : tensor<4xi32>
      %one = arith.constant dense<1> : tensor<4xi32>
      %out = scf.for %i = %c0 to %n step %c1 iter_args(%x = %range) -> (tensor<4xi32>) {
        %opaque = arith.shli %x, %zero {tag="opaque"} : tensor<4xi32>
        %next = arith.addi %opaque, %one : tensor<4xi32>
        scf.yield %next : tensor<4xi32>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  Value opaque = tagged(*module, "opaque")->getResult(0);
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(&context);
  builder.setInsertionPointAfterValue(opaque);
  Value zero = builder.create<arith::ConstantIndexOp>(opaque.getLoc(), 0);
  Value origin = builder.create<tensor::ExtractOp>(opaque.getLoc(), opaque,
                                                   ValueRange{zero});
  origin = builder.create<arith::IndexCastOp>(opaque.getLoc(),
                                              builder.getIndexType(), origin);
  auto offset = buildOffsetInfo(opaque, builder);
  ASSERT_TRUE(succeeded(offset));
  offset->scalarOffset = origin;
  offset->axes[0] = AxisInfo(AxisKind::Structured, builder.getIndexAttr(1));
  PtrAnalysis analysis(builder);
  ASSERT_TRUE(succeeded(analysis.setInfo(opaque, PtrInfo{{}, *offset})));
  expectKinds(analysis, loop.getResult(0), {AxisKind::Structured});
  ASSERT_TRUE(analysis.findCached(opaque));
  EXPECT_EQ(analysis.findCached(opaque)->offsets.fullOffsets,
            OpFoldResult(opaque));
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
