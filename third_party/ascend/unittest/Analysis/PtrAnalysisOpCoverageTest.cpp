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
#include "TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;

namespace {
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, TritonDialect, tensor::TensorDialect,
                      scf::SCFDialect>();
}
Value tagged(Operation *scope, StringRef name, unsigned result = 0) {
  Value value;
  scope->walk([&](Operation *op) {
    if (auto tag = op->getAttrOfType<StringAttr>("tag"))
      if (tag.getValue() == name)
        value = op->getResult(result);
  });
  return value;
}
template <typename T>
void expectLanes(const T &value, ArrayRef<int64_t> expected,
                 const test::Environment &env = test::Environment()) {
  auto lanes = test::evaluate(value, env);
  ASSERT_TRUE(lanes);
  ASSERT_EQ(lanes->size(), expected.size());
  for (unsigned i = 0; i < expected.size(); ++i)
    EXPECT_EQ((*lanes)[i].getSExtValue(), expected[i]) << "lane " << i;
}
void expectNoWideTensors(Operation *scope) {
  scope->walk([&](Operation *op) {
    for (Type type : op->getResultTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        EXPECT_FALSE(tensor.getElementType().isIndex() ||
                     tensor.getElementType().isInteger(64))
            << op->getName().getStringRef().str();
  });
}

TEST(PtrAnalysisOpCoverage, SingletonLoadsAndAtomicsSupplyDynamicOrigins) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @loaded(%input:!tt.ptr<i32>, %base:!tt.ptr<f32>, %mask:tensor<1xi1>) {
      %p = tt.splat %input : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
      %other = arith.constant dense<0> : tensor<1xi32>
      %v = tt.load %p {tag="loaded"} : tensor<1x!tt.ptr<i32>>
      %filled = tt.load %p,%mask,%other {tag="filled"} : tensor<1x!tt.ptr<i32>>
      %poison = tt.load %p,%mask {tag="poison"} : tensor<1x!tt.ptr<i32>>
      %atomic = tt.atomic_rmw add, relaxed, gpu, %p, %other {tag="atomic"}
        : (tensor<1x!tt.ptr<i32>>,tensor<1xi32>) -> tensor<1xi32>
      %masked = tt.atomic_rmw add, relaxed, gpu, %p, %other, %mask {tag="masked_atomic"}
        : (tensor<1x!tt.ptr<i32>>,tensor<1xi32>,tensor<1xi1>) -> tensor<1xi32>
      %true = arith.constant dense<true> : tensor<1xi1>
      %active_load = tt.load %p,%true {tag="active_load"} : tensor<1x!tt.ptr<i32>>
      %active = tt.atomic_rmw add, relaxed, gpu, %p, %other, %true {tag="active_atomic"}
        : (tensor<1x!tt.ptr<i32>>,tensor<1xi32>,tensor<1xi1>) -> tensor<1xi32>
      %b = tt.broadcast %v : tensor<1xi32> -> tensor<8xi32>
      %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
      %off = arith.addi %b,%r : tensor<8xi32>
      %ps = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %ptr = tt.addptr %ps,%off {tag="ptr"} : tensor<8x!tt.ptr<f32>>,tensor<8xi32>
      %widep = tt.splat %input : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
      %wide = tt.load %widep {tag="wide"} : tensor<8x!tt.ptr<i32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  for (StringRef tag :
       {"loaded", "filled", "active_load", "atomic", "active_atomic"}) {
    Value value = tagged(f, tag);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->offsets.isUniform());
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    test::Environment env{{value, {llvm::APInt(32, -17, true)}}};
    expectLanes(info->offsets.scalarOffset, {-17}, env);
  }
  for (StringRef tag : {"poison", "masked_atomic", "wide"}) {
    auto info = analysis.analyze(tagged(f, tag));
    ASSERT_TRUE(succeeded(info));
    EXPECT_FALSE(info->offsets.hasStructuredForm());
    EXPECT_FALSE(info->offsets.scalarOffset);
  }
  auto ptr = analysis.analyze(tagged(f, "ptr"));
  ASSERT_TRUE(succeeded(ptr));
  ASSERT_TRUE(ptr->offsets.hasStructuredForm());
  expectLanes(ptr->offsets.axes[0].stride, {1});
  test::Environment env{{tagged(f, "loaded"), {llvm::APInt(32, 123)}}};
  expectLanes(ptr->offsets.scalarOffset, {123}, env);
  expectLanes(ptr->offsets, {123, 124, 125, 126, 127, 128, 129, 130}, env);
  EXPECT_EQ(std::distance(f.getBody().front().getOps<triton::LoadOp>().begin(),
                          f.getBody().front().getOps<triton::LoadOp>().end()),
            5);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, ScalarAndRankZeroMemoryOrigins) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @scalar(%p:!tt.ptr<i32>, %mask:i1) {
      %zero = arith.constant 0 : i32
      %v = tt.load %p,%mask {tag="scalar_load"} : !tt.ptr<i32>
      %a = tt.atomic_rmw add, relaxed, gpu, %p, %zero, %mask {tag="scalar_atomic"}
        : (!tt.ptr<i32>,i32,i1) -> i32
      tt.return
    }
    tt.func @rankzero(%p:tensor<!tt.ptr<i32>>, %mask:tensor<i1>) {
      %zero = arith.constant dense<0> : tensor<i32>
      %true = arith.constant dense<true> : tensor<i1>
      %false = arith.constant dense<false> : tensor<i1>
      %v = tt.load %p {tag="loaded"} : tensor<!tt.ptr<i32>>
      %filled = tt.load %p,%mask,%zero {tag="filled"} : tensor<!tt.ptr<i32>>
      %active = tt.load %p,%true {tag="active_load"} : tensor<!tt.ptr<i32>>
      %masked = tt.load %p,%mask {tag="masked_load"} : tensor<!tt.ptr<i32>>
      %inactive = tt.load %p,%false {tag="inactive_load"} : tensor<!tt.ptr<i32>>
      %a = tt.atomic_rmw add, relaxed, gpu, %p, %zero {tag="atomic"}
        : (tensor<!tt.ptr<i32>>,tensor<i32>) -> tensor<i32>
      %aa = tt.atomic_rmw add, relaxed, gpu, %p, %zero, %true {tag="active_atomic"}
        : (tensor<!tt.ptr<i32>>,tensor<i32>,tensor<i1>) -> tensor<i32>
      %ma = tt.atomic_rmw add, relaxed, gpu, %p, %zero, %mask {tag="masked_atomic"}
        : (tensor<!tt.ptr<i32>>,tensor<i32>,tensor<i1>) -> tensor<i32>
      %ia = tt.atomic_rmw add, relaxed, gpu, %p, %zero, %false {tag="inactive_atomic"}
        : (tensor<!tt.ptr<i32>>,tensor<i32>,tensor<i1>) -> tensor<i32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  for (StringRef tag :
       {"masked_load", "inactive_load", "masked_atomic", "inactive_atomic"}) {
    Value value = tagged(*module, tag);
    Block *block = value.getDefiningOp()->getBlock();
    auto operationCount = block->getOperations().size();
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    EXPECT_FALSE(info->offsets.scalarOffset);
    EXPECT_FALSE(info->offsets.hasStructuredForm());
    EXPECT_EQ(block->getOperations().size(), operationCount);
  }
  for (StringRef tag : {"scalar_load", "scalar_atomic", "loaded", "filled",
                        "active_load", "atomic", "active_atomic"}) {
    Value value = tagged(*module, tag);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->offsets.isUniform());
    test::Environment env{{value, {llvm::APInt(32, -17, true)}}};
    expectLanes(info->offsets.scalarOffset, {-17}, env);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, FloatToIntKeepsUniformityWithoutInventingStrides) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @casts(%scalar:f32, %rows:tensor<2xf32>) {
      %s = tt.splat %scalar : f32 -> tensor<4xf32>
      %c = arith.constant dense<2.5> : tensor<4xf32>
      %m = arith.mulf %s,%c : tensor<4xf32>
      %sum = arith.addf %m,%m : tensor<4xf32>
      %uniform = arith.fptosi %sum {tag="uniform"} : tensor<4xf32> to tensor<4xi32>
      %e = tt.expand_dims %rows {axis=1:i32} : tensor<2xf32> -> tensor<2x1xf32>
      %b = tt.broadcast %e : tensor<2x1xf32> -> tensor<2x4xf32>
      %partial = arith.fptosi %b {tag="partial"} : tensor<2x4xf32> to tensor<2x4xi32>
      %r = tt.make_range {start=0:i32,end=4:i32} : tensor<4xi32>
      %rf = arith.sitofp %r : tensor<4xi32> to tensor<4xf32>
      %varying = arith.fptosi %rf {tag="varying"} : tensor<4xf32> to tensor<4xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto uniform = analysis.analyze(tagged(f, "uniform"));
  ASSERT_TRUE(succeeded(uniform));
  EXPECT_TRUE(uniform->offsets.isUniform());
  test::Environment env{
      {tagged(f, "uniform"), test::Lanes(4, llvm::APInt(32, -5, true))}};
  expectLanes(uniform->offsets.scalarOffset, {-5}, env);
  auto partial = analysis.analyze(tagged(f, "partial"));
  ASSERT_TRUE(succeeded(partial));
  EXPECT_EQ(test::axisKinds(partial->offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Invariant}));
  EXPECT_FALSE(partial->offsets.scalarOffset);
  auto varying = analysis.analyze(tagged(f, "varying"));
  ASSERT_TRUE(succeeded(varying));
  EXPECT_FALSE(varying->offsets.hasStructuredForm());
  EXPECT_EQ(varying->offsets.fullOffsets, OpFoldResult(tagged(f, "varying")));
  expectNoWideTensors(f);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, ReduceRemovesTheReducedAxisAndKeepsExactResults) {
  auto check = [](auto rules) {
    MLIRContext context;
    load(context);
    auto module = parseSourceString<ModuleOp>(R"mlir(module {
      tt.func @reductions(%rows:tensor<2xi32>, %cols:tensor<4xi32>) {
        %e = tt.expand_dims %rows {axis=1:i32} : tensor<2xi32> -> tensor<2x1xi32>
        %b = tt.broadcast %e : tensor<2x1xi32> -> tensor<2x4xi32>
        %uniform = "tt.reduce"(%b) ({
        ^bb0(%x:i32, %y:i32):
          %s = arith.addi %x,%y : i32
          tt.reduce.return %s : i32
        }) {axis=0:i32,tag="uniform"} : (tensor<2x4xi32>) -> tensor<4xi32>
        %c = tt.expand_dims %cols {axis=0:i32} : tensor<4xi32> -> tensor<1x4xi32>
        %bc = tt.broadcast %c : tensor<1x4xi32> -> tensor<2x4xi32>
        %varying = "tt.reduce"(%bc) ({
        ^bb0(%x:i32, %y:i32):
          %s = arith.addi %x,%y : i32
          tt.reduce.return %s : i32
        }) {axis=0:i32,tag="varying"} : (tensor<2x4xi32>) -> tensor<4xi32>
        %e3 = tt.expand_dims %c {axis=2:i32} : tensor<1x4xi32> -> tensor<1x4x1xi32>
        %b3 = tt.broadcast %e3 : tensor<1x4x1xi32> -> tensor<2x4x8xi32>
        %middle = "tt.reduce"(%b3) ({
        ^bb0(%x:i32, %y:i32):
          %s = arith.maxsi %x,%y : i32
          tt.reduce.return %s : i32
        }) {axis=1:i32,tag="middle"} : (tensor<2x4x8xi32>) -> tensor<2x8xi32>
        %scalar = "tt.reduce"(%cols) ({
        ^bb0(%x:i32, %y:i32):
          %s = arith.addi %x,%y : i32
          tt.reduce.return %s : i32
        }) {axis=0:i32,tag="scalar"} : (tensor<4xi32>) -> i32
        %pair:2 = "tt.reduce"(%b,%bc) ({
        ^bb0(%x:i32, %u:i32, %y:i32, %v:i32):
          %s = arith.addi %x,%y : i32
          %t = arith.addi %u,%v : i32
          %mixed = arith.addi %s,%t : i32
          tt.reduce.return %mixed,%t : i32,i32
        }) {axis=0:i32,tag="pair"} : (tensor<2x4xi32>,tensor<2x4xi32>) -> (tensor<4xi32>,tensor<4xi32>)
        tt.return
      }
    })mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder builder(&context);
    PtrAnalysis<decltype(rules)> analysis(builder, rules);
    for (StringRef tag : {"uniform", "middle", "scalar"}) {
      Value value = tagged(f, tag);
      auto info = analysis.analyze(value);
      ASSERT_TRUE(succeeded(info));
      EXPECT_TRUE(info->offsets.isUniform()) << tag.str();
      EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    }
    auto varying = analysis.analyze(tagged(f, "varying"));
    ASSERT_TRUE(succeeded(varying));
    EXPECT_EQ(test::axisKinds(varying->offsets),
              (SmallVector<AxisKind>{AxisKind::Unknown}));
    for (unsigned i = 0; i < 2; ++i) {
      Value value = tagged(f, "pair", i);
      auto info = analysis.analyze(value);
      ASSERT_TRUE(succeeded(info));
      EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
      EXPECT_FALSE(info->offsets.hasStructuredForm());
    }
    expectNoWideTensors(f);
    EXPECT_TRUE(succeeded(verify(*module)));
  };
  check(CommonPtrRules{});
}

TEST(PtrAnalysisOpCoverage, InsertUpdatesSameBaseCompletePointerOffsets) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @insert(%base:!tt.ptr<f32>, %other:!tt.ptr<f32>, %offset:i32, %lane:index) {
      %p = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
      %dest = tt.addptr %p,%r : tensor<8x!tt.ptr<f32>>,tensor<8xi32>
      %src = tt.addptr %base,%offset : !tt.ptr<f32>,i32
      %same = tensor.insert %src into %dest[%lane] {tag="same"} : tensor<8x!tt.ptr<f32>>
      %different = tensor.insert %other into %dest[%lane] {tag="different"} : tensor<8x!tt.ptr<f32>>
      %zero = tensor.insert %base into %p[%lane] {tag="zero"} : tensor<8x!tt.ptr<f32>>
      %three = arith.constant 3 : i32
      %ps3 = tt.addptr %base,%three : !tt.ptr<f32>,i32
      %constant = tensor.insert %ps3 into %p[%lane] {tag="constant"} : tensor<8x!tt.ptr<f32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto same = analysis.analyze(tagged(f, "same"));
  ASSERT_TRUE(succeeded(same));
  ASSERT_FALSE(same->isEmpty());
  EXPECT_EQ(same->base, f.getArgument(0));
  EXPECT_TRUE(
      succeeded(analysis.getRules().validateBinding(tagged(f, "same"), *same)));
  test::Environment env{{f.getArgument(2), {llvm::APInt(32, -13, true)}},
                        {f.getArgument(3), {llvm::APInt(64, 2)}}};
  expectLanes(same->offsets, {0, 1, -13, 3, 4, 5, 6, 7}, env);
  auto other = analysis.analyze(tagged(f, "different"));
  ASSERT_TRUE(succeeded(other));
  EXPECT_TRUE(other->isEmpty());
  auto zero = analysis.analyze(tagged(f, "zero"));
  ASSERT_TRUE(succeeded(zero));
  EXPECT_TRUE(zero->offsets.isUniform());
  EXPECT_TRUE(
      succeeded(analysis.getRules().validateBinding(tagged(f, "zero"), *zero)));
  expectLanes(zero->offsets, {0, 0, 0, 0, 0, 0, 0, 0}, env);
  auto constant = analysis.analyze(tagged(f, "constant"));
  ASSERT_TRUE(succeeded(constant));
  expectLanes(constant->offsets, {0, 0, 3, 0, 0, 0, 0, 0}, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage,
     InsertSlicePreservesRankReductionAndWideAddressSums) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @slices(%base:!tt.ptr<f32>, %other:!tt.ptr<f32>, %column:index) {
      %d = tt.splat %base : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
      %dr = arith.constant dense<[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]> : tensor<2x8xi32>
      %dest = tt.addptr %d,%dr : tensor<2x8x!tt.ptr<f32>>,tensor<2x8xi32>
      %s = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %sr = tt.make_range {start=0:i32,end=4:i32} : tensor<4xi32>
      %s1 = tt.addptr %s,%sr : tensor<4x!tt.ptr<f32>>,tensor<4xi32>
      %max = arith.constant dense<2147483647> : tensor<4xi32>
      %source = tt.addptr %s1,%max : tensor<4x!tt.ptr<f32>>,tensor<4xi32>
      %same = tensor.insert_slice %source into %dest[1,%column] [1,4] [1,2] {tag="same"}
        : tensor<4x!tt.ptr<f32>> into tensor<2x8x!tt.ptr<f32>>
      %o = tt.splat %other : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %different = tensor.insert_slice %o into %dest[1,%column] [1,4] [1,2] {tag="different"}
        : tensor<4x!tt.ptr<f32>> into tensor<2x8x!tt.ptr<f32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto same = analysis.analyze(tagged(f, "same"));
  ASSERT_TRUE(succeeded(same));
  ASSERT_FALSE(same->isEmpty());
  EXPECT_TRUE(
      succeeded(analysis.getRules().validateBinding(tagged(f, "same"), *same)));
  test::Environment env{{f.getArgument(2), {llvm::APInt(64, 1)}}};
  expectLanes(same->offsets,
              {0, 1, 2, 3, 4, 5, 6, 7, 8, 2147483647, 10, 2147483648, 12,
               2147483649, 14, 2147483650},
              env);
  auto different = analysis.analyze(tagged(f, "different"));
  ASSERT_TRUE(succeeded(different));
  EXPECT_TRUE(different->isEmpty());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage,
     LoopResultsRemainRuntimeValuesWithoutInitSubstitution) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @loop(%n:index, %initial:i32, %offsets:tensor<4xi32>, %base:!tt.ptr<i32>) {
      %lb = arith.constant 0 : index
      %step = arith.constant 1 : index
      %one = arith.constant 1 : i32
      %ones = arith.constant dense<1> : tensor<4xi32>
      %out:3 = scf.for %i = %lb to %n step %step
          iter_args(%s=%initial, %t=%offsets, %p=%base) -> (i32,tensor<4xi32>,!tt.ptr<i32>) {
        %ns = arith.addi %s,%one : i32
        %nt = arith.addi %t,%ones : tensor<4xi32>
        %np = tt.addptr %p,%ns : !tt.ptr<i32>,i32
        scf.yield %ns,%nt,%np : i32,tensor<4xi32>,!tt.ptr<i32>
      } {tag="loop"}
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  for (unsigned i = 0; i < 2; ++i) {
    Value result = tagged(f, "loop", i);
    auto info = analysis.analyze(result);
    ASSERT_TRUE(succeeded(info));
    ASSERT_FALSE(info->isEmpty());
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(result));
    EXPECT_EQ(info->offsets.isUniform(), i == 0);
  }
  auto pointer = analysis.analyze(tagged(f, "loop", 2));
  ASSERT_TRUE(succeeded(pointer));
  EXPECT_EQ(pointer->base, tagged(f, "loop", 2));
  expectLanes(pointer->offsets, {0});
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisOpCoverage, ScalarLeavesKeepDynamicStridesWithoutRecursion) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @scalar(%x:i32) {
      %four = arith.constant 4 : i32
      %one = arith.constant 1 : i32
      %product = arith.muli %x,%four {tag="product"} : i32
      %scale = arith.addi %product,%one {tag="scale"} : i32
      %s = tt.splat %scale : i32 -> tensor<8xi32>
      %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
      %offset = arith.muli %r,%s {tag="offset"} : tensor<8xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  Value scale = tagged(f, "scale");
  auto scalar = analysis.analyze(scale);
  ASSERT_TRUE(succeeded(scalar));
  EXPECT_EQ(scalar->offsets.fullOffsets, OpFoldResult(scale));
  EXPECT_TRUE(scalar->offsets.isUniform());
  EXPECT_FALSE(analysis.findCached(tagged(f, "product")));
  EXPECT_FALSE(analysis.findCached(f.getArgument(0)));
  auto tensor = analysis.analyze(tagged(f, "offset"));
  ASSERT_TRUE(succeeded(tensor));
  ASSERT_TRUE(tensor->offsets.hasStructuredForm());
  test::Environment env{{f.getArgument(0), {llvm::APInt(32, 5)}}};
  expectLanes(tensor->offsets.scalarOffset, {0}, env);
  expectLanes(tensor->offsets.axes[0].stride, {21}, env);
  EXPECT_TRUE(succeeded(analysis.bind(scale, *scalar)));
  EXPECT_FALSE(analysis.findCached(tagged(f, "offset")));
  ASSERT_TRUE(succeeded(analysis.analyze(tagged(f, "offset"))));
  EXPECT_FALSE(analysis.findCached(tagged(f, "product")));
  expectNoWideTensors(f);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, ScalarConstantFoldingPrecedesSubAxisDiscovery) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @constants() {
      %four = arith.constant 4 : i32
      %sum = arith.addi %four,%four : i32
      %unfolded = tt.splat %sum : i32 -> tensor<32xi32>
      %folded = arith.constant dense<8> : tensor<32xi32>
      %r = tt.make_range {start=0:i32,end=32:i32} : tensor<32xi32>
      %a = arith.remui %r,%unfolded {tag="unfolded"} : tensor<32xi32>
      %b = arith.remui %r,%folded {tag="folded"} : tensor<32xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto unfolded = analysis.analyze(tagged(f, "unfolded"));
  auto folded = analysis.analyze(tagged(f, "folded"));
  ASSERT_TRUE(succeeded(unfolded));
  ASSERT_TRUE(succeeded(folded));
  EXPECT_FALSE(unfolded->offsets.hasStructuredDecomposition());
  EXPECT_TRUE(folded->offsets.hasStructuredDecomposition());
  ASSERT_FALSE(folded->offsets.axes[0].subAxes.empty());
  // Both exact values remain usable; only the proof depends on folding.
  EXPECT_EQ(test::evaluate(unfolded->offsets), test::evaluate(folded->offsets));
  expectNoWideTensors(f);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, SingletonBoundariesKeepTheirCurrentRuntimeValue) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @singleton(%arg:tensor<1x1xi32>, %n:index, %cond:i1) {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %increment = arith.constant dense<1> : tensor<1x1xi32>
      %loop = scf.for %i = %zero to %n step %one iter_args(%v=%arg) -> tensor<1x1xi32> {
        %next = arith.addi %v,%increment : tensor<1x1xi32>
        scf.yield %next : tensor<1x1xi32>
      } {tag="loop"}
      %selected = arith.select %cond,%arg,%loop {tag="selected"} : tensor<1x1xi32>
      %a = tt.broadcast %arg {tag="arg"} : tensor<1x1xi32> -> tensor<2x4xi32>
      %b = tt.broadcast %loop {tag="broadcast_loop"} : tensor<1x1xi32> -> tensor<2x4xi32>
      %c = tt.broadcast %selected {tag="broadcast_select"} : tensor<1x1xi32> -> tensor<2x4xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  test::Environment env{{f.getArgument(0), {llvm::APInt(32, 5)}},
                        {f.getArgument(2), {llvm::APInt(1, 0)}},
                        {tagged(f, "loop"), {llvm::APInt(32, 11)}}};
  for (StringRef tag :
       {"loop", "selected", "arg", "broadcast_loop", "broadcast_select"}) {
    Value value = tagged(f, tag);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->offsets.isUniform());
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    expectLanes(info->offsets.scalarOffset, {tag == "arg" ? 5 : 11}, env);
  }
  expectNoWideTensors(f);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, ClampPreservesCommonInvariantAxes) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @clamp(%x:f32, %rows:tensor<2xf32>) {
      %x8 = tt.splat %x : f32 -> tensor<8xf32>
      %lo = arith.constant dense<0.0> : tensor<8xf32>
      %hi = arith.constant dense<1.0> : tensor<8xf32>
      %clamp = tt.clampf %x8,%lo,%hi,propagateNan=none : tensor<8xf32>
      %uniform = arith.fptosi %clamp {tag="uniform"} : tensor<8xf32> to tensor<8xi32>
      %e = tt.expand_dims %rows {axis=1:i32} : tensor<2xf32> -> tensor<2x1xf32>
      %b = tt.broadcast %e : tensor<2x1xf32> -> tensor<2x4xf32>
      %min = arith.constant dense<0.0> : tensor<2x4xf32>
      %max = arith.constant dense<1.0> : tensor<2x4xf32>
      %c = tt.clampf %b,%min,%max,propagateNan=none : tensor<2x4xf32>
      %partial = arith.fptosi %c {tag="partial"} : tensor<2x4xf32> to tensor<2x4xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto uniform = analysis.analyze(tagged(f, "uniform"));
  ASSERT_TRUE(succeeded(uniform));
  ASSERT_TRUE(uniform->offsets.isUniform());
  EXPECT_EQ(uniform->offsets.fullOffsets, OpFoldResult(tagged(f, "uniform")));
  test::Environment env{
      {tagged(f, "uniform"), test::Lanes(8, llvm::APInt(32, 1))}};
  expectLanes(uniform->offsets.scalarOffset, {1}, env);
  auto partial = analysis.analyze(tagged(f, "partial"));
  ASSERT_TRUE(succeeded(partial));
  EXPECT_EQ(test::axisKinds(partial->offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Invariant}));
  EXPECT_FALSE(partial->offsets.scalarOffset);
  expectNoWideTensors(f);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, TensorSelectIntersectsConditionAndBranchAxes) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @select(%c:i1, %a:i32, %b:i32, %rows:tensor<2xi1>, %varying:tensor<2x4xi1>) {
      %c8 = tt.splat %c : i1 -> tensor<8xi1>
      %a8 = tt.splat %a : i32 -> tensor<8xi32>
      %b8 = tt.splat %b : i32 -> tensor<8xi32>
      %u = arith.select %c8,%a8,%b8 {tag="uniform"} : tensor<8xi1>,tensor<8xi32>
      %e = tt.expand_dims %rows {axis=1:i32} : tensor<2xi1> -> tensor<2x1xi1>
      %cb = tt.broadcast %e : tensor<2x1xi1> -> tensor<2x4xi1>
      %ab = tt.splat %a : i32 -> tensor<2x4xi32>
      %bb = tt.splat %b : i32 -> tensor<2x4xi32>
      %p = arith.select %cb,%ab,%bb {tag="partial"} : tensor<2x4xi1>,tensor<2x4xi32>
      %v = arith.select %varying,%ab,%bb {tag="varying"} : tensor<2x4xi1>,tensor<2x4xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto uniform = analysis.analyze(tagged(f, "uniform"));
  ASSERT_TRUE(succeeded(uniform));
  ASSERT_TRUE(uniform->offsets.isUniform());
  for (bool condition : {false, true}) {
    test::Environment env{{f.getArgument(0), {llvm::APInt(1, condition)}},
                          {f.getArgument(1), {llvm::APInt(32, -7, true)}},
                          {f.getArgument(2), {llvm::APInt(32, 13)}}};
    expectLanes(uniform->offsets.scalarOffset, {condition ? -7 : 13}, env);
  }
  auto partial = analysis.analyze(tagged(f, "partial"));
  ASSERT_TRUE(succeeded(partial));
  EXPECT_EQ(test::axisKinds(partial->offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Invariant}));
  auto varying = analysis.analyze(tagged(f, "varying"));
  ASSERT_TRUE(succeeded(varying));
  EXPECT_EQ(test::axisKinds(varying->offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Unknown}));
  expectNoWideTensors(f);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, LinearOffsetDoesNotDetermineBlockBounds) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @coordinates(%base:!tt.ptr<i32>) {
      %shape = arith.constant 4 : i64
      %stride = arith.constant 1 : i64
      %zero = arith.constant 0 : i32
      %four = arith.constant 4 : i32
      %one = arith.constant 1 : i32
      %minusfour = arith.constant -4 : i32
      %outside = tt.make_tensor_ptr %base,[%shape,%shape],[%shape,%stride],[%zero,%four]
        {order=array<i32:1,0>,tag="outside"}:!tt.ptr<tensor<1x1xi32>>
      %inside = tt.advance %outside,[%one,%minusfour] {tag="inside"}:!tt.ptr<tensor<1x1xi32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(f.getBody().front().getTerminator());
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto outside = analysis.analyze(tagged(f, "outside"));
  auto inside = analysis.analyze(tagged(f, "inside"));
  ASSERT_TRUE(succeeded(outside));
  ASSERT_TRUE(succeeded(inside));
  ASSERT_TRUE(outside->block);
  ASSERT_TRUE(inside->block);
  expectLanes(outside->offsets.scalarOffset, {4});
  expectLanes(inside->offsets.scalarOffset, {4});
  auto inBounds = [](const BlockPtrInfo &block) {
    for (auto [coordinate, shape] :
         llvm::zip(block.coordinateOffsets, block.parentShape)) {
      auto c = test::evaluate(coordinate)->front().getSExtValue();
      auto n = test::evaluate(shape)->front().getSExtValue();
      if (c < 0 || c >= n)
        return false;
    }
    return true;
  };
  EXPECT_FALSE(inBounds(*outside->block));
  EXPECT_TRUE(inBounds(*inside->block));
  expectLanes(outside->block->coordinateOffsets[0], {0});
  expectLanes(outside->block->coordinateOffsets[1], {4});
  expectLanes(inside->block->coordinateOffsets[0], {1});
  expectLanes(inside->block->coordinateOffsets[1], {0});
  EXPECT_TRUE(succeeded(verify(*module)));
}
} // namespace
