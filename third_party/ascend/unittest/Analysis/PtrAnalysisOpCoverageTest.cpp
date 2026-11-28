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
#include "TritonMemoryAccess/PtrAnalysis.h"
#include "TritonMemoryAccess/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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
void expectAffineLanes(const OffsetInfo &info, ArrayRef<int64_t> expected,
                       const test::Environment &env = test::Environment()) {
  ASSERT_TRUE(info.hasStructuredForm());
  ASSERT_EQ(info.axes.size(), 1u);
  ASSERT_EQ(info.getShape()[0], expected.size());
  auto origin = test::evaluate(info.scalarOffset, env);
  auto stride = test::evaluate(info.axes[0].stride, env);
  ASSERT_TRUE(origin && stride);
  ASSERT_EQ(origin->size(), 1u);
  ASSERT_EQ(stride->size(), 1u);
  for (unsigned i = 0; i < expected.size(); ++i)
    EXPECT_EQ(origin->front().getSExtValue() +
                  int64_t(i) * stride->front().getSExtValue(),
              expected[i])
        << "lane " << i;
}

TEST(PtrAnalysisOpCoverage, OptionalInvarianceFailureKeepsExactResults) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @optional(%mask: tensor<*xi1>, %input: tensor<*xi32>) {
      %condition = tensor.cast %mask {tag="condition"} : tensor<*xi1> to tensor<4xi1>
      %one = arith.constant dense<1> : tensor<4xi32>
      %two = arith.constant dense<2> : tensor<4xi32>
      %selected = arith.select %condition, %one, %two {tag="selected"} : tensor<4xi1>, tensor<4xi32>
      %source = tensor.cast %input {tag="source"} : tensor<*xi32> to tensor<2x4xi32>
      %float = arith.sitofp %source : tensor<2x4xi32> to tensor<2x4xf32>
      %integer = arith.fptosi %float {tag="integer"} : tensor<2x4xf32> to tensor<2x4xi32>
      %sum = "tt.reduce"(%source) ({
        ^bb0(%a: i32, %b: i32):
          %s = arith.addi %a, %b : i32
          tt.reduce.return %s : i32
      }) {axis=0:i32, tag="sum"} : (tensor<2x4xi32>) -> tensor<4xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(func.getBody().front().getTerminator());
  PtrAnalysis analysis(builder);

  // Unranked sources cannot supply per-axis facts through these casts.
  EXPECT_TRUE(failed(analysis.analyze(tagged(func, "condition"))));
  EXPECT_TRUE(failed(analysis.analyze(tagged(func, "source"))));
  for (StringRef tag : {"selected", "integer", "sum"}) {
    Value value = tagged(func, tag);
    auto result = analysis.analyze(value);
    ASSERT_TRUE(succeeded(result)) << tag.str();
    EXPECT_EQ(result->offsets.fullOffsets, OpFoldResult(value));
    EXPECT_EQ(result->offsets.valueType, value.getType());
    for (const auto &axis : result->offsets.axes)
      EXPECT_EQ(axis.kind, AxisKind::Unknown);
  }
  auto selected = analysis.analyze(tagged(func, "selected"));
  ASSERT_TRUE(succeeded(selected));
  test::Environment env{{tagged(func, "condition"),
                         {llvm::APInt(1, 1), llvm::APInt(1, 0),
                          llvm::APInt(1, 1), llvm::APInt(1, 0)}}};
  expectLanes(selected->offsets, {1, 2, 1, 2}, env);
  EXPECT_TRUE(succeeded(verify(*module)));
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
  PtrAnalysis analysis(builder);
  for (StringRef tag :
       {"loaded", "filled", "active_load", "atomic", "active_atomic", "wide"}) {
    Value value = tagged(f, tag);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->offsets.isUniform());
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    test::Environment env{
        {value,
         test::Lanes(cast<RankedTensorType>(value.getType()).getNumElements(),
                     llvm::APInt(32, -17, true))}};
    expectLanes(info->offsets.scalarOffset, {-17}, env);
  }
  for (StringRef tag : {"poison", "masked_atomic"}) {
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

TEST(PtrAnalysisOpCoverage, LoadInvarianceKeepsDependentAddressesStructured) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @loads(%index: !tt.ptr<i32>, %data: !tt.ptr<f32>, %condition: i1,
                   %mask: tensor<4xi1>, %other: tensor<4xi32>,
                   %opaque: tensor<4x!tt.ptr<i32>>) {
      %p = tt.splat %index : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %m = tt.splat %condition : i1 -> tensor<4xi1>
      %true = arith.constant dense<true> : tensor<4xi1>
      %zero = arith.constant dense<0> : tensor<4xi32>
      %plain = tt.load %p {tag="plain"} : tensor<4x!tt.ptr<i32>>
      %filled = tt.load %p, %m, %zero {tag="filled"} : tensor<4x!tt.ptr<i32>>
      %active = tt.load %p, %true, %other {tag="active"} : tensor<4x!tt.ptr<i32>>
      %missing = tt.load %p, %m {tag="missing"} : tensor<4x!tt.ptr<i32>>
      %mixed_mask = tt.load %p, %mask, %zero {tag="mixed_mask"} : tensor<4x!tt.ptr<i32>>
      %mixed_other = tt.load %p, %m, %other {tag="mixed_other"} : tensor<4x!tt.ptr<i32>>
      %volatile = tt.load %p {isVolatile=true, tag="volatile"} : tensor<4x!tt.ptr<i32>>
      %lanes = tt.load %opaque {tag="opaque"} : tensor<4x!tt.ptr<i32>>
      %f = tt.splat %data : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %fzero = arith.constant dense<0.0> : tensor<4xf32>
      %float = tt.load %f, %m, %fzero : tensor<4x!tt.ptr<f32>>
      %integer = arith.fptosi %float {tag="float"} : tensor<4xf32> to tensor<4xi32>
      %fmixed = tt.load %f, %mask, %fzero : tensor<4x!tt.ptr<f32>>
      %imixed = arith.fptosi %fmixed {tag="float_mixed"} : tensor<4xf32> to tensor<4xi32>
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %offset = arith.addi %plain, %range : tensor<4xi32>
      %pointer = tt.addptr %f, %offset {tag="pointer"} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %expanded = tt.expand_dims %range {axis=0:i32} : tensor<4xi32> -> tensor<1x4xi32>
      %rows = tt.broadcast %expanded : tensor<1x4xi32> -> tensor<2x4xi32>
      %base = tt.splat %index : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
      %row_ptrs = tt.addptr %base, %rows : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi32>
      %partial = tt.load %row_ptrs {tag="partial"} : tensor<2x4x!tt.ptr<i32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto func = *module->getOps<triton::FuncOp>().begin();
  unsigned loadsBefore = 0;
  func.walk([&](triton::LoadOp) { ++loadsBefore; });
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  for (StringRef tag : {"plain", "filled", "active", "float"}) {
    Value value = tagged(func, tag);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info)) << tag.str();
    EXPECT_TRUE(info->offsets.isUniform()) << tag.str();
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    test::Environment env{{value, test::Lanes(4, llvm::APInt(32, 5))}};
    expectLanes(info->offsets.scalarOffset, {5}, env);
  }
  for (StringRef tag : {"missing", "mixed_mask", "mixed_other", "volatile",
                        "opaque", "float_mixed"}) {
    Value value = tagged(func, tag);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info)) << tag.str();
    EXPECT_FALSE(info->offsets.isUniform()) << tag.str();
    EXPECT_FALSE(info->offsets.scalarOffset);
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
  }
  auto pointer = analysis.analyze(tagged(func, "pointer"));
  ASSERT_TRUE(succeeded(pointer));
  EXPECT_EQ(pointer->base, func.getArgument(1));
  ASSERT_TRUE(pointer->offsets.hasStructuredForm());
  test::Environment env{
      {tagged(func, "plain"), test::Lanes(4, llvm::APInt(32, 5))}};
  expectLanes(pointer->offsets.scalarOffset, {5}, env);
  expectLanes(pointer->offsets.axes[0].stride, {1});
  expectLanes(pointer->offsets, {5, 6, 7, 8}, env);
  auto partial = analysis.analyze(tagged(func, "partial"));
  ASSERT_TRUE(succeeded(partial));
  EXPECT_EQ(test::axisKinds(partial->offsets),
            (SmallVector<AxisKind>{AxisKind::Invariant, AxisKind::Unknown}));
  EXPECT_FALSE(partial->offsets.scalarOffset);
  unsigned loadsAfter = 0;
  func.walk([&](triton::LoadOp) { ++loadsAfter; });
  EXPECT_EQ(loadsBefore, loadsAfter);
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
  PtrAnalysis analysis(builder);
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
  PtrAnalysis analysis(builder);
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

TEST(PtrAnalysisOpCoverage, FloatIfJoinsAxesWithoutExportingBranchValues) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @branches(%condition: i1, %a: f32, %b: f32,
                     %varying: tensor<4xf32>, %rows: tensor<2xf32>) {
      %chosen:2 = scf.if %condition -> (tensor<4xf32>, tensor<4xf32>) {
        %x = tt.splat %a : f32 -> tensor<4xf32>
        %nested = scf.if %condition -> tensor<4xf32> {
          scf.yield %x : tensor<4xf32>
        } else {
          %y = tt.splat %b : f32 -> tensor<4xf32>
          scf.yield %y : tensor<4xf32>
        }
        scf.yield %nested, %x : tensor<4xf32>, tensor<4xf32>
      } else {
        %y = tt.splat %b : f32 -> tensor<4xf32>
        scf.yield %y, %varying : tensor<4xf32>, tensor<4xf32>
      }
      %uniform = arith.fptosi %chosen#0 {tag="uniform"} : tensor<4xf32> to tensor<4xi32>
      %unknown = arith.fptosi %chosen#1 {tag="unknown"} : tensor<4xf32> to tensor<4xi32>
      %partial = scf.if %condition -> tensor<2x4xf32> {
        %e = tt.expand_dims %rows {axis=1:i32} : tensor<2xf32> -> tensor<2x1xf32>
        %r = tt.broadcast %e : tensor<2x1xf32> -> tensor<2x4xf32>
        scf.yield %r : tensor<2x4xf32>
      } else {
        %r = tt.splat %a : f32 -> tensor<2x4xf32>
        scf.yield %r : tensor<2x4xf32>
      }
      %ints = arith.fptosi %partial {tag="partial"} : tensor<2x4xf32> to tensor<2x4xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  Value uniformValue = tagged(func, "uniform");
  auto uniform = analysis.analyze(uniformValue);
  ASSERT_TRUE(succeeded(uniform));
  ASSERT_TRUE(uniform->offsets.isUniform());
  EXPECT_EQ(uniform->offsets.fullOffsets, OpFoldResult(uniformValue));
  auto origin = cast<Value>(uniform->offsets.scalarOffset);
  auto castOp = origin.getDefiningOp<arith::IndexCastOp>();
  ASSERT_TRUE(castOp);
  auto extract = castOp.getIn().getDefiningOp<tensor::ExtractOp>();
  ASSERT_TRUE(extract);
  EXPECT_EQ(extract.getTensor(), uniformValue);
  test::Environment env{{uniformValue, test::Lanes(4, llvm::APInt(32, 9))}};
  expectLanes(uniform->offsets.scalarOffset, {9}, env);
  auto unknown = analysis.analyze(tagged(func, "unknown"));
  ASSERT_TRUE(succeeded(unknown));
  EXPECT_EQ(test::axisKinds(unknown->offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown}));
  EXPECT_FALSE(unknown->offsets.scalarOffset);
  auto partial = analysis.analyze(tagged(func, "partial"));
  ASSERT_TRUE(succeeded(partial));
  EXPECT_EQ(test::axisKinds(partial->offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Invariant}));
  EXPECT_FALSE(partial->offsets.scalarOffset);
  expectNoWideTensors(func);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, IntegerSumScalesOnlyTheRemainingAxes) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @sums() {
      %i = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %j = tt.make_range {start=0:i32, end=2:i32} : tensor<2xi32>
      %three = arith.constant dense<3> : tensor<4xi32>
      %two = arith.constant dense<2> : tensor<2xi32>
      %rows = arith.muli %i, %three : tensor<4xi32>
      %cols = arith.muli %j, %two : tensor<2xi32>
      %er = tt.expand_dims %rows {axis=1:i32} : tensor<4xi32> -> tensor<4x1xi32>
      %ec = tt.expand_dims %cols {axis=0:i32} : tensor<2xi32> -> tensor<1x2xi32>
      %br = tt.broadcast %er : tensor<4x1xi32> -> tensor<4x2xi32>
      %bc = tt.broadcast %ec : tensor<1x2xi32> -> tensor<4x2xi32>
      %ten = arith.constant dense<10> : tensor<4x2xi32>
      %zero = arith.constant dense<0> : tensor<4x2xi32>
      %terms = arith.addi %br, %bc : tensor<4x2xi32>
      %x = arith.addi %terms, %ten {tag="source"} : tensor<4x2xi32>
      %negative = arith.subi %zero, %x : tensor<4x2xi32>
      %along_cols = "tt.reduce"(%x) ({
      ^bb0(%a: i32, %b: i32):
        %sum = arith.addi %a, %b : i32
        tt.reduce.return %sum : i32
      }) {axis=1:i32, tag="cols"} : (tensor<4x2xi32>) -> tensor<4xi32>
      %along_rows = "tt.reduce"(%x) ({
      ^bb0(%a: i32, %b: i32):
        %sum = arith.addi %b, %a : i32
        tt.reduce.return %sum : i32
      }) {axis=0:i32, tag="rows"} : (tensor<4x2xi32>) -> tensor<2xi32>
      %neg = "tt.reduce"(%negative) ({
      ^bb0(%a: i32, %b: i32):
        %sum = arith.addi %a, %b : i32
        tt.reduce.return %sum : i32
      }) {axis=1:i32, tag="negative"} : (tensor<4x2xi32>) -> tensor<4xi32>
      %product = "tt.reduce"(%x) ({
      ^bb0(%a: i32, %b: i32):
        %mul = arith.muli %a, %b : i32
        tt.reduce.return %mul : i32
      }) {axis=1:i32, tag="product"} : (tensor<4x2xi32>) -> tensor<4xi32>
      %pair:2 = "tt.reduce"(%x, %negative) ({
      ^bb0(%a: i32, %b: i32, %c: i32, %d: i32):
        %ac = arith.addi %a, %c : i32
        %bd = arith.addi %b, %d : i32
        tt.reduce.return %ac, %bd : i32, i32
      }) {axis=1:i32, tag="pair"} : (tensor<4x2xi32>, tensor<4x2xi32>) -> (tensor<4xi32>, tensor<4xi32>)
      %sixtythree = arith.constant dense<63> : tensor<4xi32>
      %large = arith.addi %i, %sixtythree : tensor<4xi32>
      %narrow = arith.trunci %large : tensor<4xi32> to tensor<4xi8>
      %en = tt.expand_dims %narrow {axis=1:i32} : tensor<4xi8> -> tensor<4x1xi8>
      %bn = tt.broadcast %en {tag="narrow_source"} : tensor<4x1xi8> -> tensor<4x2xi8>
      %wrapped = "tt.reduce"(%bn) ({
      ^bb0(%a: i8, %b: i8):
        %sum = arith.addi %a, %b : i8
        tt.reduce.return %sum : i8
      }) {axis=1:i32, tag="wrapped"} : (tensor<4x2xi8>) -> tensor<4xi8>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  auto source = analysis.analyze(tagged(func, "source"));
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(source->offsets.hasStructuredForm());
  expectLanes(source->offsets, {10, 12, 13, 15, 16, 18, 19, 21});
  auto check = [&](StringRef tag, int64_t origin, int64_t stride,
                   ArrayRef<int64_t> expected) {
    Value value = tagged(func, tag);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->offsets.hasStructuredForm());
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    expectLanes(info->offsets.scalarOffset, {origin});
    expectLanes(info->offsets.axes[0].stride, {stride});
    // Compare the decomposition against hand-computed reductions rather than
    // the implementation's reduction formula.
    expectAffineLanes(info->offsets, expected);
  };
  check("cols", 22, 6, {22, 28, 34, 40});
  check("rows", 58, 8, {58, 66});
  check("negative", -22, -6, {-22, -28, -34, -40});
  auto narrow = analysis.analyze(tagged(func, "narrow_source"));
  ASSERT_TRUE(succeeded(narrow));
  ASSERT_TRUE(narrow->offsets.hasStructuredForm());
  expectLanes(narrow->offsets, {63, 63, 64, 64, 65, 65, 66, 66});
  for (Value value : {tagged(func, "product"), tagged(func, "pair", 0),
                      tagged(func, "pair", 1), tagged(func, "wrapped")}) {
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
    EXPECT_FALSE(info->offsets.hasStructuredForm());
  }
  auto wrapped = analysis.analyze(tagged(func, "wrapped"));
  ASSERT_TRUE(succeeded(wrapped));
  test::Environment env{
      {tagged(func, "wrapped"),
       {llvm::APInt(8, 126), llvm::APInt(8, -128, true),
        llvm::APInt(8, -126, true), llvm::APInt(8, -124, true)}}};
  expectLanes(wrapped->offsets, {126, -128, -126, -124}, env);
  expectNoWideTensors(func);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, I64SumKeepsItsDynamicOrigin) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @dynamic(%bias: i64) {
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %wide = arith.extsi %range : tensor<4xi32> to tensor<4xi64>
      %expanded = tt.expand_dims %wide {axis=1:i32} : tensor<4xi64> -> tensor<4x1xi64>
      %varying = tt.broadcast %expanded : tensor<4x1xi64> -> tensor<4x2xi64>
      %uniform = tt.splat %bias : i64 -> tensor<4x2xi64>
      %source = arith.addi %uniform, %varying : tensor<4x2xi64>
      %sum = "tt.reduce"(%source) ({
      ^bb0(%a: i64, %b: i64):
        %next = arith.addi %a, %b : i64
        tt.reduce.return %next : i64
      }) {axis=1:i32, tag="sum"} : (tensor<4x2xi64>) -> tensor<4xi64>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  Value value = tagged(func, "sum");
  auto info = analysis.analyze(value);
  ASSERT_TRUE(succeeded(info));
  ASSERT_TRUE(info->offsets.hasStructuredForm());
  EXPECT_EQ(info->offsets.fullOffsets, OpFoldResult(value));
  test::Environment env{{func.getArgument(0), {llvm::APInt(64, 7)}}};
  expectLanes(info->offsets.scalarOffset, {14}, env);
  expectLanes(info->offsets.axes[0].stride, {2});
  expectAffineLanes(info->offsets, {14, 16, 18, 20}, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, ReduceRemovesTheReducedAxisAndKeepsExactResults) {
  auto check = []() {
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
    PtrAnalysis analysis(builder);
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
  check();
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
  PtrAnalysis analysis(builder);
  auto same = analysis.analyze(tagged(f, "same"));
  ASSERT_TRUE(succeeded(same));
  ASSERT_FALSE(same->isEmpty());
  EXPECT_EQ(same->base, f.getArgument(0));
  EXPECT_TRUE(succeeded(
      mlir::triton::ptr::detail::validateInfo(tagged(f, "same"), *same)));
  test::Environment env{{f.getArgument(2), {llvm::APInt(32, -13, true)}},
                        {f.getArgument(3), {llvm::APInt(64, 2)}}};
  expectLanes(same->offsets, {0, 1, -13, 3, 4, 5, 6, 7}, env);
  auto other = analysis.analyze(tagged(f, "different"));
  ASSERT_TRUE(succeeded(other));
  EXPECT_EQ(other->base, tagged(f, "different"));
  EXPECT_FALSE(other->hasScalarBase());
  auto zero = analysis.analyze(tagged(f, "zero"));
  ASSERT_TRUE(succeeded(zero));
  EXPECT_TRUE(zero->offsets.isUniform());
  EXPECT_TRUE(succeeded(
      mlir::triton::ptr::detail::validateInfo(tagged(f, "zero"), *zero)));
  expectLanes(zero->offsets, {0, 0, 0, 0, 0, 0, 0, 0}, env);
  auto constant = analysis.analyze(tagged(f, "constant"));
  ASSERT_TRUE(succeeded(constant));
  expectLanes(constant->offsets, {0, 0, 3, 0, 0, 0, 0, 0}, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, InsertSupportsDynamicPointerTensors) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @insert(%base:!tt.ptr<f32>, %other:!tt.ptr<f32>, %offset:i32,
                    %size:index, %lane:index) {
      %p = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
      %ptr = tt.addptr %p,%r : tensor<8x!tt.ptr<f32>>,tensor<8xi32>
      %dest = tensor.extract_slice %ptr[1] [%size] [1] {tag="dest"}
        : tensor<8x!tt.ptr<f32>> to tensor<?x!tt.ptr<f32>>
      %src = tt.addptr %base,%offset {tag="src"} : !tt.ptr<f32>,i32
      %same = tensor.insert %src into %dest[%lane] {tag="same"} : tensor<?x!tt.ptr<f32>>
      %different = tensor.insert %other into %dest[%lane] {tag="different"} : tensor<?x!tt.ptr<f32>>
      %uniform = tensor.extract_slice %p[0] [%size] [1] {tag="uniform"}
        : tensor<8x!tt.ptr<f32>> to tensor<?x!tt.ptr<f32>>
      %zero = tensor.insert %base into %uniform[%lane] {tag="zero"} : tensor<?x!tt.ptr<f32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(f.getBody().front().getTerminator());
  auto insertion = builder.saveInsertionPoint();
  PtrAnalysis analysis(builder);
  // Both inputs already have valid facts before the insertion is analyzed.
  auto source = analysis.analyze(tagged(f, "src"));
  auto dest = analysis.analyze(tagged(f, "dest"));
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(dest));
  auto same = analysis.analyze(tagged(f, "same"));
  ASSERT_TRUE(succeeded(same));
  EXPECT_EQ(same->base, f.getArgument(0));
  EXPECT_EQ(
      same->offsets.valueType,
      RankedTensorType::get({ShapedType::kDynamic}, builder.getIndexType()));
  auto full =
      cast<Value>(same->offsets.fullOffsets).getDefiningOp<tensor::InsertOp>();
  ASSERT_TRUE(full);
  EXPECT_EQ(full.getDest(), cast<Value>(dest->offsets.fullOffsets));
  EXPECT_EQ(full.getIndices(), ValueRange{f.getArgument(4)});
  EXPECT_EQ(full.getType(), RankedTensorType::get({ShapedType::kDynamic},
                                                  builder.getI64Type()));
  EXPECT_EQ(same->offsets.axes[0].kind, AxisKind::Unknown);
  EXPECT_FALSE(same->offsets.scalarOffset);
  EXPECT_TRUE(succeeded(
      mlir::triton::ptr::detail::validateInfo(tagged(f, "same"), *same)));
  test::Environment env{{f.getArgument(2), {llvm::APInt(32, -13, true)}},
                        {f.getArgument(3), {llvm::APInt(64, 5)}},
                        {f.getArgument(4), {llvm::APInt(64, 2)}}};
  expectLanes(same->offsets, {1, 2, -13, 4, 5}, env);
  env[f.getArgument(3)] = {llvm::APInt(64, 3)};
  env[f.getArgument(4)] = {llvm::APInt(64, 0)};
  expectLanes(same->offsets, {-13, 2, 3}, env);

  // Uniform insertion retains its origin; unrelated pointer bases still fail.
  auto zero = analysis.analyze(tagged(f, "zero"));
  ASSERT_TRUE(succeeded(zero));
  EXPECT_TRUE(zero->offsets.isUniform());
  expectLanes(zero->offsets, {0, 0, 0}, env);
  ASSERT_TRUE(succeeded(analysis.analyze(f.getArgument(1))));
  auto opaque = analysis.analyze(tagged(f, "different"));
  ASSERT_TRUE(succeeded(opaque));
  EXPECT_EQ(opaque->base, tagged(f, "different"));

  // Reject a missing complete value at binding, before any later insertion.
  auto uniform = analysis.analyze(tagged(f, "uniform"));
  ASSERT_TRUE(succeeded(uniform));
  ASSERT_TRUE(uniform->offsets.isUniform());
  uniform->offsets.fullOffsets = {};
  EXPECT_TRUE(failed(analysis.setInfo(tagged(f, "uniform"), *uniform)));
  EXPECT_TRUE(succeeded(analysis.analyze(tagged(f, "zero"))));
  EXPECT_EQ(builder.getInsertionBlock(), insertion.getBlock());
  EXPECT_EQ(builder.getInsertionPoint(), insertion.getPoint());
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
  PtrAnalysis analysis(builder);
  auto same = analysis.analyze(tagged(f, "same"));
  ASSERT_TRUE(succeeded(same));
  ASSERT_FALSE(same->isEmpty());
  EXPECT_TRUE(succeeded(
      mlir::triton::ptr::detail::validateInfo(tagged(f, "same"), *same)));
  test::Environment env{{f.getArgument(2), {llvm::APInt(64, 1)}}};
  expectLanes(same->offsets,
              {0, 1, 2, 3, 4, 5, 6, 7, 8, 2147483647, 10, 2147483648, 12,
               2147483649, 14, 2147483650},
              env);
  auto different = analysis.analyze(tagged(f, "different"));
  ASSERT_TRUE(succeeded(different));
  EXPECT_FALSE(different->hasScalarBase());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisOpCoverage, InsertSliceSupportsDynamicPointerTensors) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @slices(%base:!tt.ptr<f32>, %other:!tt.ptr<f32>, %rows:index,
                    %count:index, %column:index, %step:index) {
      %d = tt.splat %base : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
      %dr = arith.constant dense<[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]> : tensor<2x8xi32>
      %dest = tt.addptr %d,%dr {tag="dest"} : tensor<2x8x!tt.ptr<f32>>,tensor<2x8xi32>
      %dynamic_dest = tensor.extract_slice %dest[0,0] [%rows,8] [1,1] {tag="dynamic_dest"}
        : tensor<2x8x!tt.ptr<f32>> to tensor<?x8x!tt.ptr<f32>>
      %s = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %sr = arith.constant dense<[100,101,102,103]> : tensor<4xi32>
      %source = tt.addptr %s,%sr {tag="source"} : tensor<4x!tt.ptr<f32>>,tensor<4xi32>
      %dynamic_source = tensor.extract_slice %source[0] [%count] [1] {tag="dynamic_source"}
        : tensor<4x!tt.ptr<f32>> to tensor<?x!tt.ptr<f32>>
      %source_only = tensor.insert_slice %dynamic_source into %dest[1,%column] [1,%count] [1,%step] {tag="source_only"}
        : tensor<?x!tt.ptr<f32>> into tensor<2x8x!tt.ptr<f32>>
      %dest_only = tensor.insert_slice %source into %dynamic_dest[1,%column] [1,4] [1,%step] {tag="dest_only"}
        : tensor<4x!tt.ptr<f32>> into tensor<?x8x!tt.ptr<f32>>
      %both = tensor.insert_slice %dynamic_source into %dynamic_dest[1,%column] [1,%count] [1,%step] {tag="both"}
        : tensor<?x!tt.ptr<f32>> into tensor<?x8x!tt.ptr<f32>>
      %o = tt.splat %other : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %different = tensor.insert_slice %o into %dynamic_dest[1,%column] [1,4] [1,%step] {tag="different"}
        : tensor<4x!tt.ptr<f32>> into tensor<?x8x!tt.ptr<f32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(f.getBody().front().getTerminator());
  auto insertion = builder.saveInsertionPoint();
  PtrAnalysis analysis(builder);
  test::Environment env{{f.getArgument(2), {llvm::APInt(64, 2)}},
                        {f.getArgument(3), {llvm::APInt(64, 3)}},
                        {f.getArgument(4), {llvm::APInt(64, 1)}},
                        {f.getArgument(5), {llvm::APInt(64, 2)}}};
  // Exercise each formerly rejected shape independently, including the
  // rank-reduced source and runtime slice coordinates.
  for (StringRef name : {"source_only", "dest_only", "both"}) {
    auto original = tagged(f, name).getDefiningOp<tensor::InsertSliceOp>();
    auto source = analysis.analyze(original.getSource());
    auto dest = analysis.analyze(original.getDest());
    ASSERT_TRUE(succeeded(source)) << name.str();
    ASSERT_TRUE(succeeded(dest)) << name.str();
    auto result = analysis.analyze(original.getResult());
    ASSERT_TRUE(succeeded(result)) << name.str();
    EXPECT_EQ(result->base, f.getArgument(0));
    EXPECT_EQ(result->offsets.valueType,
              original.getType().clone(builder.getIndexType()));
    auto full = cast<Value>(result->offsets.fullOffsets)
                    .getDefiningOp<tensor::InsertSliceOp>();
    ASSERT_TRUE(full);
    EXPECT_EQ(full.getSource(), cast<Value>(source->offsets.fullOffsets));
    EXPECT_EQ(full.getDest(), cast<Value>(dest->offsets.fullOffsets));
    EXPECT_EQ(full.getType(), original.getType().clone(builder.getI64Type()));
    EXPECT_EQ(full.getMixedOffsets(), original.getMixedOffsets());
    EXPECT_EQ(full.getMixedSizes(), original.getMixedSizes());
    EXPECT_EQ(full.getMixedStrides(), original.getMixedStrides());
    EXPECT_EQ(full.getDroppedDims(), original.getDroppedDims());
    EXPECT_EQ(test::axisKinds(result->offsets),
              (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Unknown}));
    EXPECT_FALSE(result->offsets.scalarOffset);
    EXPECT_TRUE(succeeded(mlir::triton::ptr::detail::validateInfo(
        original.getResult(), *result)));
    SmallVector<int64_t> expected{0, 1,   2,  3,   4,  5,   6,  7,
                                  8, 100, 10, 101, 12, 102, 14, 15};
    if (name == "dest_only")
      expected.back() = 103;
    expectLanes(result->offsets, expected, env);
  }
  auto different =
      tagged(f, "different").getDefiningOp<tensor::InsertSliceOp>();
  ASSERT_TRUE(succeeded(analysis.analyze(different.getSource())));
  auto opaque = analysis.analyze(different.getResult());
  ASSERT_TRUE(succeeded(opaque));
  EXPECT_EQ(opaque->base, different.getResult());
  EXPECT_EQ(builder.getInsertionBlock(), insertion.getBlock());
  EXPECT_EQ(builder.getInsertionPoint(), insertion.getPoint());
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
  PtrAnalysis analysis(builder);
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
  PtrAnalysis analysis(builder);
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
  EXPECT_TRUE(succeeded(analysis.setInfo(scale, *scalar)));
  EXPECT_TRUE(analysis.findCached(tagged(f, "offset")));
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
  PtrAnalysis analysis(builder);
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
  PtrAnalysis analysis(builder);
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
  PtrAnalysis analysis(builder);
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
  PtrAnalysis analysis(builder);
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
  PtrAnalysis analysis(builder);
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
