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
#include "TritonMemoryAccess/PtrAnalysisRules/StructuredPtrRules.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;
using test::buildAddressView;
using test::eraseDeadPtrArithmetic;
namespace {
using test::evaluate;
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, triton::TritonDialect,
                      tensor::TensorDialect>();
}
Value tagged(Operation *scope, StringRef name) {
  Value value;
  scope->walk([&](Operation *op) {
    if (auto tag = op->getAttrOfType<StringAttr>("tag"))
      if (tag.getValue() == name)
        value = op->getResult(0);
  });
  return value;
}
unsigned operations(Operation *scope) {
  unsigned count = 0;
  scope->walk([&](Operation *) { ++count; });
  return count;
}
template <typename T>
void expectLanes(const T &value, ArrayRef<int64_t> expected,
                 const test::Environment &env = test::Environment()) {
  auto lanes = evaluate(value, env);
  ASSERT_TRUE(lanes);
  ASSERT_EQ(lanes->size(), expected.size());
  for (unsigned i = 0; i < expected.size(); ++i)
    EXPECT_EQ((*lanes)[i].getSExtValue(), expected[i]) << "lane " << i;
}
// Expand the public coordinate description independently of its combiners.
void expectDecomposition(const OffsetInfo &info,
                         const test::Environment &env = test::Environment()) {
  ASSERT_TRUE(info.hasCompleteForm());
  auto actual = evaluate(info, env);
  auto origin = evaluate(info.scalarOffset, env);
  ASSERT_TRUE(actual && origin);
  for (unsigned lane = 0; lane < actual->size(); ++lane) {
    auto value = origin->front();
    int64_t rest = lane;
    for (int i = info.shape.size() - 1; i >= 0; --i) {
      int64_t coordinate = rest % info.shape[i];
      rest /= info.shape[i];
      const auto &axis = info.axes[i];
      if (axis.subAxes) {
        for (auto &sub : llvm::reverse(*axis.subAxes)) {
          auto shape = evaluate(sub.shape, env),
               stride = evaluate(sub.stride, env);
          ASSERT_TRUE(shape && stride);
          int64_t size = shape->front().getSExtValue();
          value += llvm::APInt(64, coordinate % size)
                       .zextOrTrunc(value.getBitWidth()) *
                   stride->front();
          coordinate /= size;
        }
      } else {
        auto stride = evaluate(axis.stride, env);
        ASSERT_TRUE(stride);
        value += llvm::APInt(64, coordinate).zextOrTrunc(value.getBitWidth()) *
                 stride->front();
      }
    }
    EXPECT_EQ(value, (*actual)[lane]) << "lane " << lane;
  }
}

TEST(PtrAnalysisShape, OrderedReshapePreservesAddressesAndRadixStructure) {
  MLIRContext context;
  load(context);
  for (unsigned width : {32u, 64u}) {
    for (auto strides :
         {SmallVector<int64_t>{8, 4, 1}, SmallVector<int64_t>{64, 16, 1},
          SmallVector<int64_t>{0, 0, 1}}) {
      std::string text =
          R"mlir(module { tt.func @shape(%base: !tt.ptr<i16, 3>, %off: tensor<2x2x4xiW>) {
        %p = tt.splat %base : !tt.ptr<i16, 3> -> tensor<2x2x4x!tt.ptr<i16, 3>>
        %a = tt.addptr %p, %off : tensor<2x2x4x!tt.ptr<i16, 3>>, tensor<2x2x4xiW>
        %r = "tt.reshape"(%a) {tag="reshape"} : (tensor<2x2x4x!tt.ptr<i16, 3>>) -> tensor<4x4x!tt.ptr<i16, 3>>
        tt.return
      } })mlir";
      for (size_t pos; (pos = text.find("iW")) != std::string::npos;)
        text.replace(pos, 2, "i" + std::to_string(width));
      auto module = parseSourceString<ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      auto f = *module->getOps<triton::FuncOp>().begin();
      OpBuilder b(f.getBody().front().getTerminator());
      auto ip = b.saveInsertionPoint();
      PtrAnalysis<StructuredPtrRules> analysis(b, StructuredPtrRules{});
      auto offsets = makeOpaqueOffset(f.getArgument(1), b);
      ASSERT_TRUE(succeeded(offsets));
      Type scalar = b.getIndexType();
      offsets->scalarOffset = b.getIntegerAttr(scalar, 0);
      for (unsigned i = 0; i < 3; ++i)
        offsets->axes[i] =
            AxisInfo(strides[i] ? AxisKind::Structured : AxisKind::Invariant,
                     b.getIntegerAttr(scalar, strides[i]));
      ASSERT_TRUE(succeeded(
          analysis.bind(f.getArgument(1), *makeIntegerInfo(*offsets))));
      Value reshape = tagged(f, "reshape");
      auto result = test::getPtrInfo(analysis.analyze(reshape));
      ASSERT_TRUE(succeeded(result));
      EXPECT_EQ(result->base, f.getArgument(0));
      EXPECT_EQ(result->getAddressSpace(), 3u);
      EXPECT_EQ(result->getElementType(), b.getI16Type());
      EXPECT_EQ(result->offsets.valueType,
                RankedTensorType::get({4, 4}, b.getIndexType()));
      test::Environment env;
      SmallVector<int64_t> expected;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          for (int k = 0; k < 4; ++k) {
            int64_t n = i * strides[0] + j * strides[1] + k;
            env[f.getArgument(1)].emplace_back(width, n);
            expected.push_back(n);
          }
      expectLanes(result->offsets, expected, env);
      expectDecomposition(result->offsets, env);
      if (strides[0] == 64) {
        EXPECT_EQ(result->offsets.axes[0].kind, AxisKind::Unknown);
        ASSERT_TRUE(result->offsets.axes[0].subAxes);
        EXPECT_EQ(result->offsets.axes[0].subAxes->size(), 2u);
      }
      unsigned count = operations(*module);
      ASSERT_TRUE(succeeded(analysis.analyze(reshape)));
      EXPECT_EQ(operations(*module), count);
      EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
      EXPECT_TRUE(succeeded(verify(*module)));
    }
  }
}

TEST(PtrAnalysisShape, SliceCombinesOriginStrideAndRankReduction) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @slice(%base: !tt.ptr<i32>, %off: tensor<2x1x8xi64>) {
    %p = tt.splat %base : !tt.ptr<i32> -> tensor<2x1x8x!tt.ptr<i32>>
    %a = tt.addptr %p, %off : tensor<2x1x8x!tt.ptr<i32>>, tensor<2x1x8xi64>
    %s = tensor.extract_slice %a[1, 0, 1] [1, 1, 4] [1, 1, 2] {tag="slice"} : tensor<2x1x8x!tt.ptr<i32>> to tensor<4x!tt.ptr<i32>>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto offsets = makeOpaqueOffset(f.getArgument(1), b);
  offsets->scalarOffset = b.getIndexAttr(5);
  offsets->axes = {{AxisKind::Structured, b.getIndexAttr(16)},
                   {AxisKind::Invariant, b.getIndexAttr(0)},
                   {AxisKind::Structured, b.getIndexAttr(1)}};
  ASSERT_TRUE(
      succeeded(analysis.bind(f.getArgument(1), *makeIntegerInfo(*offsets))));
  auto result = test::getPtrInfo(analysis.analyze(tagged(f, "slice")));
  ASSERT_TRUE(succeeded(result));
  expectLanes(*result->offsets.scalarOffset, {22});
  expectLanes(result->offsets.axes[0].stride, {2});
  test::Environment env;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 8; ++j)
      env[f.getArgument(1)].emplace_back(64, 5 + 16 * i + j);
  expectLanes(result->offsets, {22, 24, 26, 28}, env);
  expectDecomposition(result->offsets, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape,
     UnknownSliceOriginIsNotInventedAndIdentityCastKeepsType) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @partial(%off: tensor<2x4xi64>) {
    %s = tensor.extract_slice %off[1, 1] [1, 2] [1, 1] {tag="slice"} : tensor<2x4xi64> to tensor<2xi64>
    %c = "tensor.cast"(%s) {tag="cast"} : (tensor<2xi64>) -> tensor<2xi64>
    %d = "tensor.cast"(%s) {tag="dynamic"} : (tensor<2xi64>) -> tensor<?xi64>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto offsets = makeOpaqueOffset(f.getArgument(0), b);
  offsets->axes[1] = AxisInfo(AxisKind::Structured, b.getIndexAttr(1));
  ASSERT_TRUE(
      succeeded(analysis.bind(f.getArgument(0), *makeIntegerInfo(*offsets))));
  auto result = test::getOffsets(analysis.analyze(tagged(f, "cast")));
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result->scalarOffset);
  EXPECT_EQ(result->axes[0].kind, AxisKind::Structured);
  EXPECT_EQ(result->valueType, tagged(f, "cast").getType());
  test::Environment env;
  for (int64_t v : {30, 31, 32, 33, 90, 91, 92, 93})
    env[f.getArgument(0)].emplace_back(64, v);
  expectLanes(*result, {91, 92}, env);
  auto dynamic = analysis.analyze(tagged(f, "dynamic"));
  ASSERT_TRUE(succeeded(dynamic));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(*dynamic));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, ReorderingReshapeLeavesSourceAndUsersIntact) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @reorder(%base: !tt.ptr<i32>) -> tensor<2x2x!tt.ptr<i32>> {
    %p = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %r = "tt.reshape"(%p) {allow_reorder, tag="reshape"} : (tensor<4x!tt.ptr<i32>>) -> tensor<2x2x!tt.ptr<i32>>
    tt.return %r : tensor<2x2x!tt.ptr<i32>>
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  Value r = tagged(f, "reshape");
  auto attrs = r.getDefiningOp()->getAttrDictionary();
  auto count = operations(*module);
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto result = analysis.analyze(r);
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(*result));
  EXPECT_EQ(r.getDefiningOp()->getAttrDictionary(), attrs);
  EXPECT_EQ(f.getBody().front().getTerminator()->getOperand(0), r);
  EXPECT_EQ(operations(*module), count);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, ShapeHelpersArePureCollectedAndCleanableAfterSession) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @cleanup(%base:!tt.ptr<i32>,%x:i32) -> tensor<2x!tt.ptr<i32>> {
    %a=tt.addptr %base,%x:!tt.ptr<i32>,i32
    %p=tt.splat %a:!tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
    %r="tt.reshape"(%p) : (tensor<2x4x!tt.ptr<i32>>) -> tensor<8x!tt.ptr<i32>>
    %s=tensor.extract_slice %r[1] [2] [2] {tag="slice"}:tensor<8x!tt.ptr<i32>> to tensor<2x!tt.ptr<i32>>
    tt.return %s:tensor<2x!tt.ptr<i32>>
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto originalCount = operations(*module);
  SmallVector<std::pair<Operation *, DictionaryAttr>> original;
  f.walk([&](Operation *op) {
    original.emplace_back(op, op->getAttrDictionary());
  });
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto result = test::getPtrInfo(analysis.analyze(tagged(f, "slice")));
  ASSERT_TRUE(succeeded(result));
  EXPECT_GT(operations(*module), originalCount);
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  for (auto [op, attrs] : original)
    EXPECT_EQ(op->getAttrDictionary(), attrs);
  analysis.clear();
  IRRewriter rewriter(&context);
  eraseDeadPtrArithmetic(f, rewriter);
  EXPECT_EQ(operations(*module), originalCount);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisShape, UnitAxesAndEncodedIdentityKeepTheirContracts) {
  MLIRContext context;
  load(context);
  context.loadDialect<triton::gpu::TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #layout = #ttg.blocked<{sizePerThread=[1],threadsPerWarp=[32],warpsPerCTA=[4],order=[0]}>
    module { tt.func @units(%x:tensor<4xi32,#layout>) {
      %r=tt.make_range {start=0:i32,end=4:i32}:tensor<4xi32>
      %expand="tt.reshape"(%r) {tag="expand"}:(tensor<4xi32>) -> tensor<1x4x1xi32>
      %squeeze="tt.reshape"(%expand) {tag="squeeze"}:(tensor<1x4x1xi32>) -> tensor<4xi32>
      %cast="tensor.cast"(%x) {tag="encoded"}:(tensor<4xi32,#layout>) -> tensor<4xi32,#layout>
      tt.return
    } })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto expanded = test::getOffsets(analysis.analyze(tagged(f, "expand")));
  ASSERT_TRUE(succeeded(expanded));
  EXPECT_EQ(test::axisKinds(*expanded),
            (SmallVector<AxisKind>{AxisKind::Invariant, AxisKind::Structured,
                                   AxisKind::Invariant}));
  expectDecomposition(*expanded);
  auto squeezed = test::getOffsets(analysis.analyze(tagged(f, "squeeze")));
  ASSERT_TRUE(succeeded(squeezed));
  expectLanes(*squeezed, {0, 1, 2, 3});
  expectDecomposition(*squeezed);
  auto encoded = test::getOffsets(analysis.analyze(tagged(f, "encoded")));
  ASSERT_TRUE(succeeded(encoded));
  EXPECT_EQ(encoded->valueType, f.getArgument(0).getType());
  EXPECT_FALSE(encoded->hasCompleteForm());
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisShape, NonDivisibleContinuousReshapeRetainsStrides) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @continuous(%base:!tt.ptr<i32>) {
    %r=tt.make_range {start=0:i32,end=12:i32}:tensor<12xi32>
    %a="tt.reshape"(%r):(tensor<12xi32>) -> tensor<2x6xi32>
    %b="tt.reshape"(%a) {tag="reshape"}:(tensor<2x6xi32>) -> tensor<3x4xi32>
    %p=tt.splat %base:!tt.ptr<i32> -> tensor<3x4x!tt.ptr<i32>>
    %ptr=tt.addptr %p,%b {tag="ptr"}:tensor<3x4x!tt.ptr<i32>>,tensor<3x4xi32>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto offsets = test::getOffsets(analysis.analyze(tagged(f, "reshape")));
  ASSERT_TRUE(succeeded(offsets));
  expectLanes(*offsets, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  EXPECT_TRUE(offsets->hasStructuredForm());
  expectLanes(offsets->axes[0].stride, {4});
  expectLanes(offsets->axes[1].stride, {1});
  auto ptr = test::getPtrInfo(analysis.analyze(tagged(f, "ptr")));
  ASSERT_TRUE(succeeded(ptr));
  EXPECT_EQ(ptr->base, f.getArgument(0));
  expectDecomposition(ptr->offsets);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, NonDivisibleUniformReshapeRemainsInvariant) {
  MLIRContext context;
  load(context);
  auto module =
      parseSourceString<ModuleOp>(R"mlir(module { tt.func @uniform(%x:i64) {
    %s=tt.splat %x:i64 -> tensor<2x6xi64>
    %r="tt.reshape"(%s) {tag="reshape"}:(tensor<2x6xi64>) -> tensor<3x4xi64>
    tt.return
  } })mlir",
                                  &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto offsets = test::getOffsets(analysis.analyze(tagged(f, "reshape")));
  ASSERT_TRUE(succeeded(offsets));
  ASSERT_TRUE(offsets->scalarOffset);
  auto origin =
      cast<Value>(*offsets->scalarOffset).getDefiningOp<arith::IndexCastOp>();
  ASSERT_TRUE(origin);
  EXPECT_EQ(origin.getIn(), f.getArgument(0));
  EXPECT_TRUE(offsets->isUniform());
  EXPECT_EQ(test::axisKinds(*offsets),
            (SmallVector<AxisKind>{AxisKind::Invariant, AxisKind::Invariant}));
  test::Environment env{{f.getArgument(0), {llvm::APInt(64, -7, true)}}};
  expectLanes(*offsets, SmallVector<int64_t>(12, -7), env);
  expectDecomposition(*offsets, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, NonDivisibleOuterReshapePreservesInnerAxis) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @partial(%x:tensor<2x6x4xi64>) {
    %r="tt.reshape"(%x) {tag="reshape"}:(tensor<2x6x4xi64>) -> tensor<3x4x4xi64>
    %unit="tt.reshape"(%x) {tag="with_unit"}:(tensor<2x6x4xi64>) -> tensor<1x3x4x4xi64>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto input = makeOpaqueOffset(f.getArgument(0), b);
  ASSERT_TRUE(succeeded(input));
  input->axes[2] = AxisInfo(AxisKind::Structured, b.getIndexAttr(1));
  ASSERT_TRUE(
      succeeded(analysis.bind(f.getArgument(0), *makeIntegerInfo(*input))));
  auto offsets = test::getOffsets(analysis.analyze(tagged(f, "reshape")));
  ASSERT_TRUE(succeeded(offsets));
  EXPECT_FALSE(offsets->scalarOffset);
  EXPECT_EQ(test::axisKinds(*offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Unknown,
                                   AxisKind::Structured}));
  expectLanes(offsets->axes[2].stride, {1});
  test::Environment env;
  SmallVector<int64_t> expected;
  for (int group = 0; group < 12; ++group)
    for (int lane = 0; lane < 4; ++lane) {
      int64_t value = group * 10 + lane;
      env[f.getArgument(0)].emplace_back(64, value);
      expected.push_back(value);
    }
  expectLanes(*offsets, expected, env);
  auto withUnit = test::getOffsets(analysis.analyze(tagged(f, "with_unit")));
  ASSERT_TRUE(succeeded(withUnit));
  EXPECT_FALSE(withUnit->scalarOffset);
  EXPECT_EQ(test::axisKinds(*withUnit),
            (SmallVector<AxisKind>{AxisKind::Invariant, AxisKind::Unknown,
                                   AxisKind::Unknown, AxisKind::Structured}));
  expectLanes(withUnit->axes.back().stride, {1});
  expectLanes(*withUnit, expected, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisShape, EmptyReshapeIsUnavailableWithoutCoordinateArithmetic) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @empty(%x:tensor<2x0x4xi64>) {
    %r="tt.reshape"(%x) {tag="reshape"}:(tensor<2x0x4xi64>) -> tensor<0x8xi64>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  unsigned count = operations(*module);
  auto result = analysis.analyze(tagged(f, "reshape"));
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(*result));
  EXPECT_EQ(operations(*module), count);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisShape, BoundIndexTensorUsesOnlyCompatibleShapeOperations) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @bound(%base:!tt.ptr<i32>,%p:tensor<1x4x!tt.ptr<i32>>,%exact:tensor<1x4xindex>) {
      %r="tt.reshape"(%p) {tag="reshape"}:(tensor<1x4x!tt.ptr<i32>>) -> tensor<4x!tt.ptr<i32>>
      %b=tt.broadcast %p {tag="broadcast"}:tensor<1x4x!tt.ptr<i32>> -> tensor<2x4x!tt.ptr<i32>>
      %e=tt.expand_dims %p {axis=0:i32,tag="expand"}:tensor<1x4x!tt.ptr<i32>> -> tensor<1x1x4x!tt.ptr<i32>>
      %s=tensor.extract_slice %p[0,1] [1,2] [1,1] {tag="slice"}:tensor<1x4x!tt.ptr<i32>> to tensor<1x2x!tt.ptr<i32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto offsets = makeOpaqueOffset(f.getArgument(2), b);
  ASSERT_TRUE(succeeded(offsets));
  PtrInfo bound;
  bound.base = f.getArgument(0);
  bound.offsets = *offsets;
  ASSERT_TRUE(succeeded(analysis.bind(f.getArgument(1), bound)));
  for (StringRef name : {"reshape", "broadcast", "expand"}) {
    unsigned count = operations(*module);
    auto result = analysis.analyze(tagged(f, name));
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(std::holds_alternative<std::monostate>(*result));
    EXPECT_EQ(operations(*module), count);
  }
  // tensor dialect operations accept the existing index tensor without
  // widening.
  auto sliced = test::getPtrInfo(analysis.analyze(tagged(f, "slice")));
  ASSERT_TRUE(succeeded(sliced));
  Value zero = b.create<arith::ConstantIndexOp>(f.getLoc(), 0);
  Value one = b.create<arith::ConstantIndexOp>(f.getLoc(), 1);
  auto lane = materializeOffsetAt(sliced->offsets, {zero, one}, b, f.getLoc());
  ASSERT_TRUE(succeeded(lane));
  test::Environment env{{f.getArgument(2),
                         {llvm::APInt(64, 5), llvm::APInt(64, 6),
                          llvm::APInt(64, 7), llvm::APInt(64, 8)}}};
  expectLanes(*lane, {7}, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
