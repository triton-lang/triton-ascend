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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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
  ASSERT_TRUE(info.hasStructuredDecomposition());
  auto actual = evaluate(info, env);
  auto origin = evaluate(info.scalarOffset, env);
  ASSERT_TRUE(actual && origin);
  for (unsigned lane = 0; lane < actual->size(); ++lane) {
    auto value = origin->front();
    int64_t rest = lane;
    for (int i = info.getShape().size() - 1; i >= 0; --i) {
      int64_t coordinate = rest % info.getShape()[i];
      rest /= info.getShape()[i];
      const auto &axis = info.axes[i];
      if (!axis.subAxes.empty()) {
        for (auto &sub : llvm::reverse(axis.subAxes)) {
          auto stride = evaluate(sub.stride, env);
          ASSERT_TRUE(stride);
          int64_t size = sub.shape;
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

TEST(PtrAnalysisShape, OrderedReshapeKeepsAddressesWithoutRebuildingAxes) {
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
      PtrAnalysis analysis(b);
      auto offsets = buildOffsetInfo(f.getArgument(1), b);
      ASSERT_TRUE(succeeded(offsets));
      Type scalar = b.getIndexType();
      offsets->scalarOffset = b.getIntegerAttr(scalar, 0);
      for (unsigned i = 0; i < 3; ++i)
        offsets->axes[i] =
            AxisInfo(strides[i] ? AxisKind::Structured : AxisKind::Invariant,
                     b.getIntegerAttr(scalar, strides[i]));
      ASSERT_TRUE(
          succeeded(analysis.setInfo(f.getArgument(1), PtrInfo{{}, *offsets})));
      Value reshape = tagged(f, "reshape");
      auto result = test::getInfo(analysis.analyze(reshape));
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
      expectLanes(result->offsets.scalarOffset, {0});
      EXPECT_EQ(test::axisKinds(result->offsets),
                (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Unknown}));
      for (const auto &axis : result->offsets.axes)
        EXPECT_TRUE(axis.subAxes.empty());
      unsigned count = operations(*module);
      ASSERT_TRUE(succeeded(analysis.analyze(reshape)));
      EXPECT_EQ(operations(*module), count);
      EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
      EXPECT_TRUE(succeeded(verify(*module)));
    }
  }
}

TEST(PtrAnalysisShape, SliceRecoversUniformOriginAfterDroppingUnknownAxis) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @uniform_slice(%base: !tt.ptr<i32>, %x: tensor<4x4xi32>, %row: index) {
      %s = tensor.extract_slice %x[%row, 0] [1, 4] [1, 1] {tag="slice"} : tensor<4x4xi32> to tensor<4xi32>
      %r = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %m = arith.muli %s, %r {tag="multiply"} : tensor<4xi32>
      %p = tt.splat %base : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
      %a = tt.addptr %p, %x : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
      %ps = tensor.extract_slice %a[%row, 0] [1, 4] [1, 1] {tag="pointer"} : tensor<4x4x!tt.ptr<i32>> to tensor<4x!tt.ptr<i32>>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto input = buildOffsetInfo(f.getArgument(1), b);
  ASSERT_TRUE(succeeded(input));
  input->axes[1] = mlir::triton::ptr::detail::makeInvariantAxisInfo(b);
  ASSERT_TRUE(
      succeeded(analysis.setInfo(f.getArgument(1), PtrInfo{{}, *input})));
  test::Environment env;
  for (int64_t row : {7, 13, -5, 23})
    for (unsigned lane = 0; lane < 4; ++lane)
      env[f.getArgument(1)].emplace_back(32, row, true);
  env[f.getArgument(2)].emplace_back(64, 1);
  for (StringRef name : {"slice", "pointer"}) {
    auto info = analysis.analyze(tagged(f, name));
    ASSERT_TRUE(succeeded(info));
    EXPECT_TRUE(info->offsets.isUniform());
    expectLanes(info->offsets.scalarOffset, {13}, env);
    expectLanes(info->offsets, {13, 13, 13, 13}, env);
  }
  auto product = analysis.analyze(tagged(f, "multiply"));
  ASSERT_TRUE(succeeded(product));
  EXPECT_TRUE(product->offsets.hasStructuredForm());
  expectLanes(product->offsets.axes[0].stride, {13}, env);
  expectLanes(product->offsets, {0, 13, 26, 39}, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, CastsCompleteUniformFactsFromExternalBindings) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @casts(%dynamic: tensor<?xi32>, %fixed: tensor<4xi32>) {
      %shape = tensor.cast %dynamic {tag="shape"} : tensor<?xi32> to tensor<4xi32>
      %wide = arith.extsi %fixed {tag="wide"} : tensor<4xi32> to tensor<4xi64>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  test::Environment env;
  for (Value input : f.getArguments()) {
    auto offsets = buildOffsetInfo(input, b);
    ASSERT_TRUE(succeeded(offsets));
    offsets->axes[0] = mlir::triton::ptr::detail::makeInvariantAxisInfo(b);
    ASSERT_TRUE(succeeded(analysis.setInfo(input, PtrInfo{{}, *offsets})));
    env[input].assign(4, llvm::APInt(32, -7, true));
  }
  for (StringRef tag : {"shape", "wide"}) {
    auto info = analysis.analyze(tagged(f, tag));
    ASSERT_TRUE(succeeded(info));
    EXPECT_TRUE(info->offsets.isUniform());
    expectLanes(info->offsets.scalarOffset, {-7}, env);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
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
  PtrAnalysis analysis(b);
  auto offsets = buildOffsetInfo(f.getArgument(1), b);
  offsets->scalarOffset = b.getIndexAttr(5);
  offsets->axes = {{AxisKind::Structured, b.getIndexAttr(16)},
                   {AxisKind::Invariant, b.getIndexAttr(0)},
                   {AxisKind::Structured, b.getIndexAttr(1)}};
  ASSERT_TRUE(
      succeeded(analysis.setInfo(f.getArgument(1), PtrInfo{{}, *offsets})));
  auto result = test::getInfo(analysis.analyze(tagged(f, "slice")));
  ASSERT_TRUE(succeeded(result));
  expectLanes(result->offsets.scalarOffset, {22});
  expectLanes(result->offsets.axes[0].stride, {2});
  test::Environment env;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 8; ++j)
      env[f.getArgument(1)].emplace_back(64, 5 + 16 * i + j);
  expectLanes(result->offsets, {22, 24, 26, 28}, env);
  expectDecomposition(result->offsets, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, UnitSliceAxesKeepOriginWithoutStrideScaling) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @unit_slice(%offsets:tensor<4x8xi64>, %stride:index, %step:index) {
      %s = tensor.extract_slice %offsets[1, 0] [1, 4] [%step, 2] {tag="slice"}
        : tensor<4x8xi64> to tensor<1x4xi64>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  auto sourceInfo = buildOffsetInfo(f.getArgument(0), builder);
  ASSERT_TRUE(succeeded(sourceInfo));
  sourceInfo->scalarOffset = builder.getIndexAttr(5);
  sourceInfo->axes = {{AxisKind::Structured, f.getArgument(1)},
                      {AxisKind::Structured, builder.getIndexAttr(1)}};
  ASSERT_TRUE(succeeded(
      analysis.setInfo(f.getArgument(0), PtrInfo{{}, std::move(*sourceInfo)})));

  auto result = test::getOffsets(analysis.analyze(tagged(f, "slice")));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(test::axisKinds(*result),
            (SmallVector<AxisKind>{AxisKind::Invariant, AxisKind::Structured}));
  test::Environment env{{f.getArgument(1), {llvm::APInt(64, 16)}},
                        {f.getArgument(2), {llvm::APInt(64, 3)}}};
  for (int64_t i = 0; i < 4; ++i)
    for (int64_t j = 0; j < 8; ++j)
      env[f.getArgument(0)].emplace_back(64, 5 + 16 * i + j);
  // The unit axis still shifts the origin, but its slice step is irrelevant.
  expectLanes(result->scalarOffset, {21}, env);
  expectLanes(result->axes[0].stride, {0}, env);
  expectLanes(result->axes[1].stride, {2}, env);
  expectLanes(*result, {21, 23, 25, 27}, env);
  expectDecomposition(*result, env);
  unsigned strideProducts = 0;
  f.walk([&](arith::MulIOp) { ++strideProducts; });
  EXPECT_EQ(strideProducts, 0u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, SliceChecksUnknownStartsBeforeBuildingOrigin) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @partial_slice(%offsets:tensor<4x4xi64>, %stride:index) {
      %shifted = tensor.extract_slice %offsets[2, 1] [1, 2] [1, 1] {tag="shifted"}
        : tensor<4x4xi64> to tensor<2xi64>
      %unshifted = tensor.extract_slice %offsets[2, 0] [1, 2] [1, 1] {tag="unshifted"}
        : tensor<4x4xi64> to tensor<2xi64>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  auto sourceInfo = buildOffsetInfo(f.getArgument(0), builder);
  ASSERT_TRUE(succeeded(sourceInfo));
  sourceInfo->scalarOffset = builder.getIndexAttr(5);
  sourceInfo->axes[0] = AxisInfo(AxisKind::Structured, f.getArgument(1));
  ASSERT_TRUE(succeeded(
      analysis.setInfo(f.getArgument(0), PtrInfo{{}, std::move(*sourceInfo)})));
  test::Environment env{{f.getArgument(1), {llvm::APInt(64, 16)}}};
  for (int64_t row = 0; row < 4; ++row)
    for (int64_t offset : {0, 3, 1, 7})
      env[f.getArgument(0)].emplace_back(64, 5 + 16 * row + offset);

  // The later unknown start must not leave an unused 5 + 2 * stride behind.
  unsigned count = operations(*module);
  auto shifted = test::getOffsets(analysis.analyze(tagged(f, "shifted")));
  ASSERT_TRUE(succeeded(shifted));
  EXPECT_FALSE(shifted->scalarOffset);
  EXPECT_EQ(shifted->axes[0].kind, AxisKind::Unknown);
  expectLanes(*shifted, {40, 38}, env);
  EXPECT_EQ(operations(*module), count);

  // Starting at zero on the unknown axis still permits the known-axis shift.
  auto unshifted = test::getOffsets(analysis.analyze(tagged(f, "unshifted")));
  ASSERT_TRUE(succeeded(unshifted));
  expectLanes(unshifted->scalarOffset, {37}, env);
  EXPECT_EQ(unshifted->axes[0].kind, AxisKind::Unknown);
  expectLanes(*unshifted, {37, 40}, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape,
     TensorCastKeepsIntegerFactsAcrossStaticAndDynamicShapes) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @partial(%off: tensor<2x4xi64>) {
    %s = tensor.extract_slice %off[1, 1] [1, 2] [1, 1] {tag="slice"} : tensor<2x4xi64> to tensor<2xi64>
    %c = "tensor.cast"(%s) {tag="cast"} : (tensor<2xi64>) -> tensor<2xi64>
    %d = "tensor.cast"(%s) {tag="dynamic"} : (tensor<2xi64>) -> tensor<?xi64>
    %e = tensor.cast %d {tag="dynamic_identity"} : tensor<?xi64> to tensor<?xi64>
    %f = tensor.cast %e {tag="static"} : tensor<?xi64> to tensor<2xi64>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto offsets = buildOffsetInfo(f.getArgument(0), b);
  offsets->axes[1] = AxisInfo(AxisKind::Structured, b.getIndexAttr(1));
  ASSERT_TRUE(
      succeeded(analysis.setInfo(f.getArgument(0), PtrInfo{{}, *offsets})));
  auto result = test::getOffsets(analysis.analyze(tagged(f, "cast")));
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result->scalarOffset);
  EXPECT_EQ(result->axes[0].kind, AxisKind::Structured);
  EXPECT_EQ(result->valueType, tagged(f, "cast").getType());
  test::Environment env;
  for (int64_t v : {30, 31, 32, 33, 90, 91, 92, 93})
    env[f.getArgument(0)].emplace_back(64, v);
  expectLanes(*result, {91, 92}, env);
  auto originalCount = operations(*module);
  for (StringRef tag : {"dynamic", "dynamic_identity", "static"}) {
    Value value = tagged(f, tag);
    auto result = analysis.analyze(value);
    ASSERT_TRUE(succeeded(result)) << tag.str();
    EXPECT_EQ(result->offsets.valueType, value.getType());
    EXPECT_EQ(result->offsets.fullOffsets, OpFoldResult(value));
    EXPECT_FALSE(result->offsets.scalarOffset);
    EXPECT_EQ(result->offsets.axes[0].kind, AxisKind::Structured);
    EXPECT_EQ(result->offsets.axes[0].stride, OpFoldResult(b.getIndexAttr(1)));
    EXPECT_TRUE(
        triton::ptr::detail::validOffsetBinding(value, result->offsets));
    expectLanes(result->offsets, {91, 92}, env);
  }
  EXPECT_EQ(operations(*module), originalCount);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, TensorCastPreservesPointerFactsAndOffsetStorage) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @casts(%base: !tt.ptr<f32>, %offsets: tensor<4xi32>) {
      %p = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %source = tt.addptr %p, %offsets {tag="source"} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %identity = tensor.cast %source {tag="identity"} : tensor<4x!tt.ptr<f32>> to tensor<4x!tt.ptr<f32>>
      %dynamic = tensor.cast %source {tag="dynamic"} : tensor<4x!tt.ptr<f32>> to tensor<?x!tt.ptr<f32>>
      %same = tensor.cast %dynamic {tag="dynamic_identity"} : tensor<?x!tt.ptr<f32>> to tensor<?x!tt.ptr<f32>>
      %static = tensor.cast %same {tag="static"} : tensor<?x!tt.ptr<f32>> to tensor<4x!tt.ptr<f32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  auto insertionPoint = b.saveInsertionPoint();
  PtrAnalysis analysis(b);
  Value offsets = f.getArgument(1);
  auto offsetInfo = buildOffsetInfo(offsets, b);
  ASSERT_TRUE(succeeded(offsetInfo));
  offsetInfo->scalarOffset = b.getIndexAttr(7);
  offsetInfo->axes[0] = AxisInfo(AxisKind::Structured, b.getIndexAttr(2));
  ASSERT_TRUE(succeeded(analysis.setInfo(offsets, PtrInfo{{}, *offsetInfo})));
  Value source = tagged(f, "source");
  auto sourceInfo = analysis.analyze(source);
  ASSERT_TRUE(succeeded(sourceInfo));
  test::Environment env;
  for (int64_t value : {7, 9, 11, 13})
    env[offsets].emplace_back(32, value);

  // Exercise ordinary i64 offsets and bound i32 values/attributes.
  SmallVector<OpFoldResult> completeOffsets{
      sourceInfo->offsets.fullOffsets, offsets,
      DenseIntElementsAttr::get(cast<RankedTensorType>(offsets.getType()),
                                ArrayRef<int32_t>{7, 9, 11, 13})};
  for (OpFoldResult complete : completeOffsets) {
    PtrInfo bound = *sourceInfo;
    bound.offsets.fullOffsets = complete;
    ASSERT_TRUE(succeeded(analysis.reset({{source, bound}})));
    for (StringRef tag :
         {"identity", "dynamic", "dynamic_identity", "static"}) {
      Value value = tagged(f, tag);
      auto originalCount = operations(*module);
      auto result = analysis.analyze(value);
      ASSERT_TRUE(succeeded(result)) << tag.str();
      EXPECT_EQ(result->base, f.getArgument(0));
      EXPECT_EQ(result->offsets.valueType,
                triton::ptr::detail::addressValueType(value.getType(), b));
      EXPECT_EQ(result->offsets.scalarOffset, OpFoldResult(b.getIndexAttr(7)));
      EXPECT_EQ(result->offsets.axes[0].kind, AxisKind::Structured);
      EXPECT_EQ(result->offsets.axes[0].stride,
                OpFoldResult(b.getIndexAttr(2)));
      EXPECT_TRUE(triton::ptr::detail::validPtrBinding(value, *result));
      if (tag == "identity" || tag == "dynamic_identity")
        EXPECT_EQ(operations(*module), originalCount);
      expectLanes(result->offsets, {7, 9, 11, 13}, env);
      if (tag == "dynamic") {
        auto full = cast<Value>(result->offsets.fullOffsets);
        auto castOp = full.getDefiningOp<tensor::CastOp>();
        ASSERT_TRUE(castOp);
        auto elementType =
            cast<RankedTensorType>(full.getType()).getElementType();
        EXPECT_EQ(elementType, complete == sourceInfo->offsets.fullOffsets
                                   ? b.getI64Type()
                                   : b.getI32Type());
      }
    }
  }
  // Identity pointer casts can retain attribute-backed complete offsets.
  PtrInfo uniform = *sourceInfo;
  uniform.offsets.scalarOffset = {};
  uniform.offsets.axes[0] = mlir::triton::ptr::detail::makeInvariantAxisInfo(b);
  uniform.offsets.fullOffsets =
      DenseIntElementsAttr::get(cast<RankedTensorType>(offsets.getType()),
                                ArrayRef<int32_t>{-9, -9, -9, -9});
  ASSERT_TRUE(succeeded(analysis.reset({{source, uniform}})));
  unsigned count = operations(*module);
  auto identity = analysis.analyze(tagged(f, "identity"));
  ASSERT_TRUE(succeeded(identity));
  EXPECT_TRUE(identity->offsets.isUniform());
  expectLanes(identity->offsets.scalarOffset, {-9});
  EXPECT_EQ(operations(*module), count);
  EXPECT_EQ(b.getInsertionPoint(), insertionPoint.getPoint());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, TensorCastDropsOnlyDynamicAxisPartitions) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @partitions(%base: !tt.ptr<f32>, %p: tensor<2x4x!tt.ptr<f32>>, %offsets: tensor<2x4xi32>) {
      %outer = tensor.cast %p {tag="outer"} : tensor<2x4x!tt.ptr<f32>> to tensor<?x4x!tt.ptr<f32>>
      %inner = tensor.cast %p {tag="inner"} : tensor<2x4x!tt.ptr<f32>> to tensor<2x?x!tt.ptr<f32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  auto offsets = buildOffsetInfo(f.getArgument(2), b);
  ASSERT_TRUE(succeeded(offsets));
  offsets->valueType =
      triton::ptr::detail::addressValueType(f.getArgument(1).getType(), b);
  offsets->scalarOffset = b.getIndexAttr(7);
  offsets->axes[0] = AxisInfo(AxisKind::Structured, b.getIndexAttr(4));
  offsets->axes[0].subAxes = {{2, b.getIndexAttr(4)}};
  offsets->axes[1].subAxes = {{2, b.getIndexAttr(0)}, {2, b.getIndexAttr(1)}};
  PtrInfo sourceInfo{f.getArgument(0), *offsets};
  ASSERT_TRUE(succeeded(analysis.setInfo(f.getArgument(1), sourceInfo)));
  test::Environment env;
  for (int64_t value : {7, 8, 7, 8, 11, 12, 11, 12})
    env[f.getArgument(2)].emplace_back(32, value);
  for (StringRef tag : {"outer", "inner"}) {
    Value value = tagged(f, tag);
    auto result = analysis.analyze(value);
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(triton::ptr::detail::validPtrBinding(value, *result));
    EXPECT_EQ(result->offsets.scalarOffset, OpFoldResult(b.getIndexAttr(7)));
    EXPECT_EQ(test::axisKinds(result->offsets),
              (SmallVector<AxisKind>{AxisKind::Structured, AxisKind::Unknown}));
    EXPECT_EQ(result->offsets.axes[0].subAxes.size(), tag == "outer" ? 0u : 1u);
    EXPECT_EQ(result->offsets.axes[1].subAxes.size(), tag == "inner" ? 0u : 2u);
    expectLanes(result->offsets, {7, 8, 7, 8, 11, 12, 11, 12}, env);
  }

  // Ordinary pointers require complete offsets even with complete sub-axes.
  sourceInfo.offsets.fullOffsets = {};
  ASSERT_TRUE(sourceInfo.offsets.hasStructuredDecomposition());
  EXPECT_TRUE(failed(analysis.setInfo(f.getArgument(1), sourceInfo)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, DynamicSliceKeepsIntegerValuesAndLinearFacts) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @dynamic_slice(%unknown:tensor<16xi32>, %dynamic:tensor<?xi32>,
                                   %start:index, %size:index, %step:index) {
      %r=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
      %linear=tensor.extract_slice %r[%start] [%size] [%step] {tag="linear"}:tensor<16xi32> to tensor<?xi32>
      %opaque=tensor.extract_slice %unknown[%start] [%size] [%step] {tag="opaque"}:tensor<16xi32> to tensor<?xi32>
      %d=tensor.extract_slice %dynamic[%start] [%size] [%step] {tag="dynamic"}:tensor<?xi32> to tensor<?xi32>
      tt.return
    } })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  Value linear = tagged(f, "linear");
  ASSERT_TRUE(succeeded(analysis.analyze(
      linear.getDefiningOp<tensor::ExtractSliceOp>().getSource())));
  auto result = test::getOffsets(analysis.analyze(linear));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->valueType, linear.getType());
  EXPECT_EQ(cast<Value>(result->fullOffsets), linear);
  EXPECT_EQ(test::axisKinds(*result),
            (SmallVector<AxisKind>{AxisKind::Structured}));
  EXPECT_TRUE(succeeded(
      mlir::triton::ptr::detail::validateInfo(linear, PtrInfo{{}, *result})));

  // Unknown values still retain their exact SSA, including dynamic sources.
  auto opaque = test::getOffsets(analysis.analyze(tagged(f, "opaque")));
  ASSERT_TRUE(succeeded(opaque));
  for (StringRef name : {"opaque", "dynamic"}) {
    Value value = tagged(f, name);
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(cast<Value>(info->fullOffsets), value);
    EXPECT_FALSE(info->scalarOffset);
    EXPECT_EQ(test::axisKinds(*info),
              (SmallVector<AxisKind>{AxisKind::Unknown}));
    EXPECT_TRUE(info->axes[0].subAxes.empty());
    EXPECT_TRUE(succeeded(
        mlir::triton::ptr::detail::validateInfo(value, PtrInfo{{}, *info})));
  }
  test::Environment env{{f.getArgument(2), {llvm::APInt(64, 1)}},
                        {f.getArgument(4), {llvm::APInt(64, 2)}}};
  for (int64_t value : {4, -3, 50, 9, 8, -7, 20, 10, 3, 6, 11, 2, 0, 15, 1, 5})
    env[f.getArgument(0)].emplace_back(32, value, true);
  for (int64_t length : {0, 3}) {
    env[f.getArgument(3)] = {llvm::APInt(64, length)};
    expectLanes(result->scalarOffset, {1}, env);
    expectLanes(result->axes[0].stride, {2}, env);
    if (length == 0) {
      expectLanes(*result, {}, env);
      expectLanes(*opaque, {}, env);
    } else {
      expectLanes(*result, {1, 3, 5}, env);
      expectLanes(*opaque, {-3, 9, -7}, env);
    }
  }
  unsigned count = operations(*module);
  ASSERT_TRUE(succeeded(analysis.analyze(linear)));
  EXPECT_EQ(operations(*module), count);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, DynamicPointerSliceKeepsBaseOffsetsAndRankReduction) {
  MLIRContext context;
  load(context);
  for (bool structured : {false, true}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @dynamic_pointer_slice(%base:!tt.ptr<i32>, %off:tensor<2x1x8xi32>,
                                             %start:index, %size:index, %step:index) {
        %p=tt.splat %base:!tt.ptr<i32> -> tensor<2x1x8x!tt.ptr<i32>>
        %a=tt.addptr %p,%off {tag="source"}:tensor<2x1x8x!tt.ptr<i32>>,tensor<2x1x8xi32>
        %s=tensor.extract_slice %a[1,0,%start] [1,1,%size] [1,1,%step] {tag="slice"}:tensor<2x1x8x!tt.ptr<i32>> to tensor<?x!tt.ptr<i32>>
        tt.return
      } })mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder b(f.getBody().front().getTerminator());
    auto insertion = b.saveInsertionPoint();
    PtrAnalysis analysis(b);
    if (structured) {
      auto input = buildOffsetInfo(f.getArgument(1), b);
      ASSERT_TRUE(succeeded(input));
      input->scalarOffset = b.getIndexAttr(5);
      input->axes = {{AxisKind::Structured, b.getIndexAttr(16)},
                     {AxisKind::Invariant, b.getIndexAttr(0)},
                     {AxisKind::Structured, b.getIndexAttr(1)}};
      ASSERT_TRUE(succeeded(
          analysis.setInfo(f.getArgument(1), PtrInfo{{}, std::move(*input)})));
    }
    auto source = test::getInfo(analysis.analyze(tagged(f, "source")));
    ASSERT_TRUE(succeeded(source));
    Value value = tagged(f, "slice");
    auto result = test::getInfo(analysis.analyze(value));
    ASSERT_TRUE(succeeded(result));
    EXPECT_EQ(result->base, f.getArgument(0));
    EXPECT_EQ(result->offsets.valueType,
              RankedTensorType::get({ShapedType::kDynamic}, b.getIndexType()));
    Value full = cast<Value>(result->offsets.fullOffsets);
    EXPECT_EQ(full.getType(),
              RankedTensorType::get({ShapedType::kDynamic}, b.getI64Type()));
    auto slice = full.getDefiningOp<tensor::ExtractSliceOp>();
    ASSERT_TRUE(slice);
    auto original = value.getDefiningOp<tensor::ExtractSliceOp>();
    EXPECT_EQ(slice.getSource(), cast<Value>(source->offsets.fullOffsets));
    EXPECT_EQ(slice.getMixedOffsets(), original.getMixedOffsets());
    EXPECT_EQ(slice.getMixedSizes(), original.getMixedSizes());
    EXPECT_EQ(slice.getMixedStrides(), original.getMixedStrides());
    EXPECT_TRUE(
        succeeded(mlir::triton::ptr::detail::validateInfo(value, *result)));
    EXPECT_EQ(b.getInsertionBlock(), insertion.getBlock());
    EXPECT_EQ(b.getInsertionPoint(), insertion.getPoint());

    test::Environment env{{f.getArgument(2), {llvm::APInt(64, 1)}},
                          {f.getArgument(3), {llvm::APInt(64, 3)}},
                          {f.getArgument(4), {llvm::APInt(64, 2)}}};
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 8; ++j)
        env[f.getArgument(1)].emplace_back(32, 5 + 16 * i + j);
    expectLanes(result->offsets, {22, 24, 26}, env);
    if (structured) {
      EXPECT_EQ(result->offsets.axes[0].kind, AxisKind::Structured);
      expectLanes(result->offsets.scalarOffset, {22}, env);
      expectLanes(result->offsets.axes[0].stride, {2}, env);
    } else {
      EXPECT_FALSE(result->offsets.scalarOffset);
      EXPECT_EQ(result->offsets.axes[0].kind, AxisKind::Unknown);
    }
    unsigned count = operations(*module);
    ASSERT_TRUE(succeeded(analysis.analyze(value)));
    EXPECT_EQ(operations(*module), count);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysisShape, DynamicSliceDropsSubAxesButKeepsIndependentAxes) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @partitioned_slice(%off:tensor<4x8xi64>, %rows:index, %start:index) {
      %full=tensor.extract_slice %off[0,0] [4,8] [1,1] {tag="full"}:tensor<4x8xi64> to tensor<4x8xi64>
      %dynamic_start=tensor.extract_slice %off[%start,0] [4,8] [1,1] {tag="dynamic_start"}:tensor<4x8xi64> to tensor<4x8xi64>
      %zero_step=tensor.extract_slice %off[1,0] [4,8] [0,1] {tag="zero_step"}:tensor<4x8xi64> to tensor<4x8xi64>
      %slice=tensor.extract_slice %off[0,0] [%rows,8] [1,1] {tag="slice"}:tensor<4x8xi64> to tensor<?x8xi64>
      tt.return
    } })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto input = buildOffsetInfo(f.getArgument(0), b);
  ASSERT_TRUE(succeeded(input));
  input->scalarOffset = b.getIndexAttr(5);
  input->axes[0].subAxes = {{2, b.getIndexAttr(32)}, {2, b.getIndexAttr(8)}};
  input->axes[1] = AxisInfo(AxisKind::Structured, b.getIndexAttr(1));
  ASSERT_TRUE(succeeded(
      analysis.setInfo(f.getArgument(0), PtrInfo{{}, std::move(*input)})));
  auto full = test::getOffsets(analysis.analyze(tagged(f, "full")));
  ASSERT_TRUE(succeeded(full));
  ASSERT_EQ(full->axes[0].subAxes.size(), 2u);
  // A full-length unit-step slice can only execute in bounds with start zero.
  auto dynamicStart =
      test::getOffsets(analysis.analyze(tagged(f, "dynamic_start")));
  ASSERT_TRUE(succeeded(dynamicStart));
  EXPECT_EQ(dynamicStart->axes[0].subAxes.size(), 2u);
  // Equal lengths alone do not preserve the axis when the step is zero.
  auto zeroStep = test::getOffsets(analysis.analyze(tagged(f, "zero_step")));
  ASSERT_TRUE(succeeded(zeroStep));
  EXPECT_TRUE(zeroStep->axes[0].subAxes.empty());
  expectLanes(zeroStep->axes[1].stride, {1});
  auto slice = test::getOffsets(analysis.analyze(tagged(f, "slice")));
  ASSERT_TRUE(succeeded(slice));
  EXPECT_EQ(cast<Value>(slice->fullOffsets), tagged(f, "slice"));
  EXPECT_EQ(test::axisKinds(*slice),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Structured}));
  EXPECT_TRUE(slice->axes[0].subAxes.empty());
  expectLanes(slice->scalarOffset, {5});
  expectLanes(slice->axes[1].stride, {1});
  EXPECT_TRUE(succeeded(mlir::triton::ptr::detail::validateInfo(
      tagged(f, "slice"), PtrInfo{{}, *slice})));
  test::Environment env{{f.getArgument(1), {llvm::APInt(64, 3)}},
                        {f.getArgument(2), {llvm::APInt(64, 0)}}};
  for (int64_t row : {5, 13, 37, 45})
    for (int j = 0; j < 8; ++j)
      env[f.getArgument(0)].emplace_back(64, row + j);
  SmallVector<int64_t> expected;
  for (int64_t row : {5, 13, 37})
    for (int j = 0; j < 8; ++j)
      expected.push_back(row + j);
  expectDecomposition(*full, env);
  expectLanes(*slice, expected, env);
  expected.clear();
  for (int64_t row : {5, 13, 37, 45})
    for (int j = 0; j < 8; ++j)
      expected.push_back(row + j);
  expectLanes(*dynamicStart, expected, env);
  expected.clear();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      expected.push_back(13 + j);
  expectLanes(*zeroStep, expected, env);
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
  PtrAnalysis analysis(b);
  auto result = analysis.analyze(r);
  EXPECT_TRUE(failed(result));
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
  PtrAnalysis analysis(b);
  auto result = test::getInfo(analysis.analyze(tagged(f, "slice")));
  ASSERT_TRUE(succeeded(result));
  EXPECT_GT(operations(*module), originalCount);
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  for (auto [op, attrs] : original)
    EXPECT_EQ(op->getAttrDictionary(), attrs);
  ASSERT_TRUE(succeeded(analysis.reset()));
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
      %two=arith.constant dense<2>:tensor<4xi32>
      %q=arith.divsi %r,%two:tensor<4xi32>
      %q_expand="tt.reshape"(%q) {tag="q_expand"}:(tensor<4xi32>) -> tensor<1x4x1xi32>
      %q_squeeze="tt.reshape"(%q_expand) {tag="q_squeeze"}:(tensor<1x4x1xi32>) -> tensor<4xi32>
      %cast="tensor.cast"(%x) {tag="encoded"}:(tensor<4xi32,#layout>) -> tensor<4xi32,#layout>
      tt.return
    } })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
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
  // Unit dimensions preserve the existing div/rem decomposition unchanged.
  auto quotient = test::getOffsets(analysis.analyze(tagged(f, "q_expand")));
  ASSERT_TRUE(succeeded(quotient));
  EXPECT_EQ(test::axisKinds(*quotient),
            (SmallVector<AxisKind>{AxisKind::Invariant, AxisKind::Unknown,
                                   AxisKind::Invariant}));
  ASSERT_EQ(quotient->axes[1].subAxes.size(), 2u);
  expectDecomposition(*quotient);
  auto quotientSqueezed =
      test::getOffsets(analysis.analyze(tagged(f, "q_squeeze")));
  ASSERT_TRUE(succeeded(quotientSqueezed));
  ASSERT_EQ(quotientSqueezed->axes[0].subAxes.size(), 2u);
  expectLanes(*quotientSqueezed, {0, 0, 1, 1});
  expectDecomposition(*quotientSqueezed);
  auto encoded = test::getOffsets(analysis.analyze(tagged(f, "encoded")));
  ASSERT_TRUE(succeeded(encoded));
  EXPECT_EQ(encoded->valueType, f.getArgument(0).getType());
  EXPECT_FALSE(encoded->hasStructuredDecomposition());
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisShape, RegroupingReshapeKeepsExactOffsetsForAddPtr) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @continuous(%base:!tt.ptr<i32>) {
    %r=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
    %a="tt.reshape"(%r):(tensor<16xi32>) -> tensor<2x8xi32>
    %b="tt.reshape"(%a) {tag="reshape"}:(tensor<2x8xi32>) -> tensor<4x4xi32>
    %p=tt.splat %base:!tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
    %ptr=tt.addptr %p,%b {tag="ptr"}:tensor<4x4x!tt.ptr<i32>>,tensor<4x4xi32>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto offsets = test::getOffsets(analysis.analyze(tagged(f, "reshape")));
  ASSERT_TRUE(succeeded(offsets));
  expectLanes(*offsets, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  EXPECT_FALSE(offsets->hasStructuredDecomposition());
  EXPECT_EQ(test::axisKinds(*offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Unknown}));
  expectLanes(offsets->scalarOffset, {0});
  auto ptr = test::getInfo(analysis.analyze(tagged(f, "ptr")));
  ASSERT_TRUE(succeeded(ptr));
  EXPECT_EQ(ptr->base, f.getArgument(0));
  expectLanes(ptr->offsets,
              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, UniformReshapeRemainsInvariant) {
  MLIRContext context;
  load(context);
  auto module =
      parseSourceString<ModuleOp>(R"mlir(module { tt.func @uniform(%x:i64) {
    %s=tt.splat %x:i64 -> tensor<2x8xi64>
    %r="tt.reshape"(%s) {tag="reshape"}:(tensor<2x8xi64>) -> tensor<4x4xi64>
    tt.return
  } })mlir",
                                  &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto offsets = test::getOffsets(analysis.analyze(tagged(f, "reshape")));
  ASSERT_TRUE(succeeded(offsets));
  ASSERT_TRUE(offsets->scalarOffset);
  auto origin =
      cast<Value>(offsets->scalarOffset).getDefiningOp<arith::IndexCastOp>();
  ASSERT_TRUE(origin);
  EXPECT_EQ(origin.getIn(), f.getArgument(0));
  EXPECT_TRUE(offsets->isUniform());
  EXPECT_EQ(test::axisKinds(*offsets),
            (SmallVector<AxisKind>{AxisKind::Invariant, AxisKind::Invariant}));
  test::Environment env{{f.getArgument(0), {llvm::APInt(64, -7, true)}}};
  expectLanes(*offsets, SmallVector<int64_t>(16, -7), env);
  expectDecomposition(*offsets, env);
  EXPECT_TRUE(succeeded(verify(*module)));

  // Complete uniform facts even when an external binding omits its origin.
  Value source =
      tagged(f, "reshape").getDefiningOp<triton::ReshapeOp>().getSrc();
  auto bound = buildOffsetInfo(source, b);
  ASSERT_TRUE(succeeded(bound));
  for (auto &axis : bound->axes)
    axis.kind = AxisKind::Invariant;
  ASSERT_TRUE(
      succeeded(analysis.reset({{source, PtrInfo{{}, std::move(*bound)}}})));
  auto withoutOrigin = test::getOffsets(analysis.analyze(tagged(f, "reshape")));
  ASSERT_TRUE(succeeded(withoutOrigin));
  EXPECT_TRUE(withoutOrigin->isUniform());
  expectLanes(withoutOrigin->scalarOffset, {-7}, env);
  EXPECT_EQ(cast<Value>(withoutOrigin->fullOffsets), tagged(f, "reshape"));
  EXPECT_EQ(test::axisKinds(*withoutOrigin),
            (SmallVector<AxisKind>{AxisKind::Invariant, AxisKind::Invariant}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, RegroupingReshapeRetainsOnlyUnitAxes) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @partial(%x:tensor<2x8x4xi64>) {
    %r="tt.reshape"(%x) {tag="reshape"}:(tensor<2x8x4xi64>) -> tensor<4x4x4xi64>
    %unit="tt.reshape"(%x) {tag="with_unit"}:(tensor<2x8x4xi64>) -> tensor<1x4x4x4xi64>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto input = buildOffsetInfo(f.getArgument(0), b);
  ASSERT_TRUE(succeeded(input));
  input->axes[2] = AxisInfo(AxisKind::Structured, b.getIndexAttr(1));
  ASSERT_TRUE(
      succeeded(analysis.setInfo(f.getArgument(0), PtrInfo{{}, *input})));
  auto offsets = test::getOffsets(analysis.analyze(tagged(f, "reshape")));
  ASSERT_TRUE(succeeded(offsets));
  EXPECT_FALSE(offsets->scalarOffset);
  EXPECT_EQ(test::axisKinds(*offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Unknown,
                                   AxisKind::Unknown}));
  for (const auto &axis : offsets->axes)
    EXPECT_TRUE(axis.subAxes.empty());
  test::Environment env;
  SmallVector<int64_t> expected;
  for (int group = 0; group < 16; ++group)
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
                                   AxisKind::Unknown, AxisKind::Unknown}));
  expectLanes(*withUnit, expected, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisShape,
     EmptyReshapeKeepsItsExactValueWithoutCoordinateArithmetic) {
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
  PtrAnalysis analysis(b);
  unsigned count = operations(*module);
  auto result = analysis.analyze(tagged(f, "reshape"));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(cast<Value>(result->offsets.fullOffsets), tagged(f, "reshape"));
  EXPECT_FALSE(result->offsets.scalarOffset);
  EXPECT_EQ(test::axisKinds(result->offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Unknown}));
  EXPECT_EQ(operations(*module), count);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisShape, BoundIndexTensorConvertsForTritonShapeOperations) {
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
  PtrAnalysis analysis(b);
  auto offsets = buildOffsetInfo(f.getArgument(2), b);
  ASSERT_TRUE(succeeded(offsets));
  PtrInfo bound;
  bound.base = f.getArgument(0);
  bound.offsets = *offsets;
  ASSERT_TRUE(succeeded(analysis.setInfo(f.getArgument(1), bound)));
  for (StringRef name : {"reshape", "broadcast", "expand"}) {
    auto result = analysis.analyze(tagged(f, name));
    ASSERT_TRUE(succeeded(result));
    ASSERT_FALSE(result->isEmpty());
    EXPECT_TRUE(cast<RankedTensorType>(
                    cast<Value>(result->offsets.fullOffsets).getType())
                    .getElementType()
                    .isInteger(64));
  }
  auto sliced = test::getInfo(analysis.analyze(tagged(f, "slice")));
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

TEST(PtrAnalysisShape, LinearAxisExpansionKeepsOnlyUsedCoefficients) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @add_partition(%linear: tensor<8xi64>, %partitioned: tensor<8xi64>,
                           %stride: index) {
      %sum = arith.addi %linear, %partitioned : tensor<8xi64>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto &block = func.getBody().front();
  Value sum = *block.getOps<arith::AddIOp>().begin();
  OpBuilder builder(block.getTerminator());
  PtrAnalysis analysis(builder);
  auto linear = buildOffsetInfo(func.getArgument(0), builder);
  auto partitioned = buildOffsetInfo(func.getArgument(1), builder);
  ASSERT_TRUE(succeeded(linear) && succeeded(partitioned));
  linear->scalarOffset = builder.getIndexAttr(0);
  linear->axes[0] = AxisInfo(AxisKind::Structured, func.getArgument(2));
  partitioned->scalarOffset = builder.getIndexAttr(0);
  partitioned->axes[0].subAxes = {{2, builder.getIndexAttr(1)},
                                  {4, builder.getIndexAttr(0)}};
  ASSERT_TRUE(
      succeeded(analysis.setInfo(func.getArgument(0), PtrInfo{{}, *linear})));
  ASSERT_TRUE(succeeded(
      analysis.setInfo(func.getArgument(1), PtrInfo{{}, *partitioned})));
  auto info = test::getOffsets(analysis.analyze(sum));
  ASSERT_TRUE(succeeded(info));
  test::Environment env{{func.getArgument(2), {llvm::APInt(64, 3)}}};
  for (int64_t lane = 0; lane < 8; ++lane) {
    env[func.getArgument(0)].emplace_back(64, lane * 3);
    env[func.getArgument(1)].emplace_back(64, lane / 4);
  }
  expectDecomposition(*info, env);
  for (auto multiply : block.getOps<arith::MulIOp>())
    EXPECT_FALSE(multiply.getResult().use_empty());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisShape, SubAxisMaterializationKeepsOnlyUsedQuotients) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @lane(%index: index) { tt.return }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto &block = func.getBody().front();
  OpBuilder builder(block.getTerminator());
  OffsetInfo info;
  info.valueType = RankedTensorType::get({8}, builder.getIndexType());
  info.scalarOffset = builder.getIndexAttr(5);
  info.axes.emplace_back(AxisKind::Unknown, builder.getIndexAttr(0));
  info.axes[0].subAxes = {{2, builder.getIndexAttr(9)},
                          {4, builder.getIndexAttr(2)}};
  auto offset =
      materializeOffsetAt(info, {func.getArgument(0)}, builder, func.getLoc());
  ASSERT_TRUE(succeeded(offset));
  for (int64_t lane = 0; lane < 8; ++lane) {
    test::Environment env{{func.getArgument(0), {llvm::APInt(64, lane)}}};
    expectLanes(*offset, {5 + (lane / 4) * 9 + (lane % 4) * 2}, env);
  }
  for (auto divide : block.getOps<arith::DivUIOp>())
    EXPECT_FALSE(divide.getResult().use_empty());
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
