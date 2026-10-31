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

#include "TritonMemoryAccess/PtrAnalysis.h"
#include "PtrAnalysisTestUtils.h"
#include "TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;
using test::buildAddressView;
using test::eraseDeadPtrArithmetic;

namespace {

using test::evaluate;

std::optional<int64_t> scalar(OpFoldResult value) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    if (auto integer = dyn_cast<IntegerAttr>(attr))
      return integer.getValue().getSExtValue();
    return std::nullopt;
  }
  auto values = evaluate(cast<Value>(value));
  if (!values || values->size() != 1)
    return std::nullopt;
  return values->front().getSExtValue();
}

std::optional<int64_t> scalar(const std::optional<OpFoldResult> &value) {
  return value ? scalar(*value) : std::nullopt;
}

TEST(PtrAnalysis, SignedDeltaIsAppliedAtEveryAddPtr) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @pointer_deltas(%base: !tt.ptr<i32>) {
        %a = arith.constant 127 : i8
        %b = arith.constant 1 : i8
        %sum = arith.addi %a, %b : i8
        %once = tt.addptr %base, %sum : !tt.ptr<i32>, i8
        %first = tt.addptr %base, %a : !tt.ptr<i32>, i8
        %twice = tt.addptr %first, %b : !tt.ptr<i32>, i8
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto add = *func.getBody().front().getOps<arith::AddIOp>().begin();
  SmallVector<triton::AddPtrOp> pointers;
  func.walk([&](triton::AddPtrOp op) { pointers.push_back(op); });
  ASSERT_EQ(pointers.size(), 3u);

  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto source = test::getOffsets(analysis.analyze(add.getResult()));
  ASSERT_TRUE(succeeded(source));
  auto delta = test::getOffsets(analysis.analyze(add.getResult()));
  ASSERT_TRUE(succeeded(delta));
  EXPECT_EQ(evaluate(*delta)->front().getSExtValue(), -128);
  auto once = test::getPtrInfo(analysis.analyze(pointers[0]));
  auto twice = test::getPtrInfo(analysis.analyze(pointers[2]));
  ASSERT_TRUE(succeeded(once));
  ASSERT_TRUE(succeeded(twice));
  EXPECT_EQ(once->base, func.getArgument(0));
  EXPECT_EQ(twice->base, func.getArgument(0));
  EXPECT_EQ(scalar(once->offsets.fullOffsets), -128);
  EXPECT_EQ(scalar(twice->offsets.fullOffsets), 128);
  EXPECT_TRUE(test::getOffsets(analysis.findCached(add.getResult())));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, NarrowWrapLosesAddressStrideButKeepsExactTensor) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @wrap() {
        %range = tt.make_range {start = 126 : i32, end = 130 : i32} : tensor<4xi32>
        %narrow = arith.trunci %range : tensor<4xi32> to tensor<4xi8>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto narrow = *func.getBody().front().getOps<arith::TruncIOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto address = test::getOffsets(analysis.analyze(narrow));
  auto source = test::getOffsets(analysis.analyze(narrow));
  ASSERT_TRUE(succeeded(address));
  ASSERT_TRUE(succeeded(source));
  EXPECT_EQ(test::axisKinds(*source),
            SmallVector<AxisKind>({AxisKind::Unknown}));
  EXPECT_EQ(test::axisKinds(*address),
            SmallVector<AxisKind>({AxisKind::Unknown}));
  EXPECT_FALSE(address->scalarOffset);
  EXPECT_EQ(scalar(address->axes[0].stride), 0);
  auto values = evaluate(*address);
  ASSERT_TRUE(values);
  ASSERT_EQ(values->size(), 4u);
  const int64_t expected[] = {126, 127, -128, -127};
  for (unsigned i = 0; i < values->size(); ++i)
    EXPECT_EQ((*values)[i].getSExtValue(), expected[i]);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, AKnownAxisSurvivesAnUnrelatedOpaqueAxis) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @partial(%row: tensor<2xi32>) {
        %mask = arith.constant dense<255> : tensor<2xi32>
        %bounded = arith.andi %row, %mask : tensor<2xi32>
        %row_expanded = tt.expand_dims %bounded {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
        %rows = tt.broadcast %row_expanded : tensor<2x1xi32> -> tensor<2x4xi32>
        %column = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
        %column_expanded = tt.expand_dims %column {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
        %columns = tt.broadcast %column_expanded : tensor<1x4xi32> -> tensor<2x4xi32>
        %sum = arith.addi %rows, %columns : tensor<2x4xi32>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto sum = *func.getBody().front().getOps<arith::AddIOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto result = test::getOffsets(analysis.analyze(sum));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->shape, SmallVector<int64_t>({2, 4}));
  EXPECT_EQ(test::axisKinds(*result),
            SmallVector<AxisKind>({AxisKind::Unknown, AxisKind::Structured}));
  EXPECT_EQ(result->fullOffsets, OpFoldResult(sum.getResult()));
  EXPECT_FALSE(result->scalarOffset);
  EXPECT_EQ(scalar(result->axes[0].stride), 0);
  EXPECT_EQ(scalar(result->axes[1].stride), 1);
  EXPECT_FALSE(result->hasStructuredForm());
  EXPECT_FALSE(result->scalarOffset);
  EXPECT_TRUE(result->axes.size() <= 0 ||
              result->axes[0].kind == AxisKind::Unknown);
  ASSERT_NE(result->axes[1].kind, AxisKind::Unknown);
  EXPECT_EQ(scalar(result->axes[1].stride), 1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, DefaultIndexWidthAndWideSourcePreservation) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @widths(%index: index) {
        %wide = arith.constant 170141183460469231731687303715884105727 : i128
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto wide = *func.getBody().front().getOps<arith::ConstantOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> unspecified(
      builder, CommonPtrRules{AnalysisOptions{64, std::nullopt}});
  EXPECT_TRUE(
      failed(test::getOffsets(unspecified.analyze(func.getArgument(0)))));
  PtrAnalysis<CommonPtrRules> configured(
      builder, CommonPtrRules{AnalysisOptions{64, 64}});
  EXPECT_TRUE(
      succeeded(test::getOffsets(configured.analyze(func.getArgument(0)))));
  EXPECT_TRUE(
      succeeded(test::getOffsets(configured.analyze(func.getArgument(0)))));
  auto wideSource = test::getOffsets(configured.analyze(wide));
  ASSERT_TRUE(succeeded(wideSource));
  EXPECT_FALSE(wideSource->scalarOffset);
  EXPECT_EQ(test::evaluate(wideSource->fullOffsets)->front().getBitWidth(),
            128u);
  builder.setInsertionPoint(func.getBody().front().getTerminator());
  EXPECT_TRUE(
      failed(materializeOffsetAt(*wideSource, {}, builder, func.getLoc())));
  PtrAnalysis<CommonPtrRules> defaultWidth(builder, CommonPtrRules{});
  EXPECT_TRUE(
      succeeded(test::getOffsets(defaultWidth.analyze(func.getArgument(0)))));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, BroadcastDoesNotInventAUniformPtrOffset) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @pointer_broadcast(%base: !tt.ptr<i32>, %condition: tensor<1xi1>, %left: tensor<1xi32>, %right: tensor<1xi32>) {
        %bases = tt.splat %base : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
        %left_ptr = tt.addptr %bases, %left : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
        %right_ptr = tt.addptr %bases, %right : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
        %selected = arith.select %condition, %left_ptr, %right_ptr : tensor<1xi1>, tensor<1x!tt.ptr<i32>>
        %repeated = tt.broadcast %selected : tensor<1x!tt.ptr<i32>> -> tensor<4x!tt.ptr<i32>>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto repeated = *func.getBody().front().getOps<triton::BroadcastOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto ptrInfo = test::getPtrInfo(analysis.analyze(repeated));
  ASSERT_TRUE(succeeded(ptrInfo));
  EXPECT_EQ(ptrInfo->base, func.getArgument(0));
  EXPECT_EQ(test::axisKinds(ptrInfo->offsets),
            SmallVector<AxisKind>({AxisKind::Invariant}));
  EXPECT_FALSE(ptrInfo->offsets.scalarOffset);
  EXPECT_EQ(scalar(ptrInfo->offsets.axes[0].stride), 0);
  EXPECT_TRUE(succeeded(verify(*module)));
}

// C1: each request completion must already produce valid SSA.
TEST(PtrAnalysis, UniformAddressHelpersDominateUses) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (int order = 0; order != 3; ++order) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module {
        tt.func @uniform(%a: i32, %b: i32) {
          %a4 = tt.splat %a : i32 -> tensor<4xi32>
          %b4 = tt.splat %b : i32 -> tensor<4xi32>
          %sum = arith.addi %a4, %b4 : tensor<4xi32>
          tt.return
        }
      }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto func = *module->getOps<triton::FuncOp>().begin();
    Value sum = (*func.getBody().front().getOps<arith::AddIOp>().begin());
    OpBuilder builder(&context);
    PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
    if (order == 1) {
      ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(sum))));
      ASSERT_TRUE(succeeded(verify(*module)));
    }
    ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(sum))));
    ASSERT_TRUE(succeeded(verify(*module)));
    ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(sum))));
    ASSERT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysis, DenseSplatAddressConstantIsLegal) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @dense() {
      %x = arith.constant dense<-7> : tensor<4xi8>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  Value x = *func.getBody().front().getOps<arith::ConstantOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(x))));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, SameWidthUnsignedIndexCastRange) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @cast(%x: index) {
      %y = arith.index_castui %x : index to i64
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  Value y = *func.getBody().front().getOps<arith::IndexCastUIOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder,
                                       CommonPtrRules{AnalysisOptions{64, 64}});
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(y))));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, Index32ConstantUsesIndexAttributeStorage) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @constant() {
      %x = arith.constant -1 : index
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  Value x = *func.getBody().front().getOps<arith::ConstantOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder,
                                       CommonPtrRules{AnalysisOptions{64, 32}});
  auto result = test::getOffsets(analysis.analyze(x));
  ASSERT_TRUE(succeeded(result));
  auto attr = cast<IntegerAttr>(cast<Attribute>(*result->scalarOffset));
  EXPECT_EQ(attr.getValue().getBitWidth(), 64u);
  EXPECT_EQ(attr.getInt(), -1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, DenseSplatPtrLanes) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (unsigned width : {8u, 32u})
    for (int n : {-7, 7}) {
      OpBuilder b(&context);
      auto module = ModuleOp::create(b.getUnknownLoc());
      OwningOpRef<ModuleOp> owner(module);
      b.setInsertionPointToStart(module.getBody());
      auto ptr = triton::PointerType::get(b.getI32Type(), 1);
      auto f = b.create<triton::FuncOp>(b.getUnknownLoc(), "dense",
                                        b.getFunctionType({ptr}, {}));
      b.setInsertionPointToStart(f.addEntryBlock());
      auto type = RankedTensorType::get({4}, b.getIntegerType(width));
      Value x = b.create<arith::ConstantOp>(
          b.getUnknownLoc(),
          DenseIntElementsAttr::get(type, llvm::APInt(width, n, true)));
      Value p = b.create<triton::SplatOp>(
          b.getUnknownLoc(), RankedTensorType::get({4}, ptr), f.getArgument(0));
      auto add =
          b.create<triton::AddPtrOp>(b.getUnknownLoc(), p.getType(), p, x);
      b.create<triton::ReturnOp>(b.getUnknownLoc());
      PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
      auto delta = test::getOffsets(analysis.analyze(x));
      auto result = test::getPtrInfo(analysis.analyze(add));
      ASSERT_TRUE(succeeded(delta));
      ASSERT_TRUE(succeeded(result));
      for (const auto &ofr : {*delta, result->offsets}) {
        auto lanes = evaluate(ofr);
        ASSERT_TRUE(lanes);
        ASSERT_EQ(lanes->size(), 4u);
        for (auto lane : *lanes)
          EXPECT_EQ(lane, llvm::APInt(64, n, true));
      }
      EXPECT_TRUE(succeeded(verify(module)));
    }
}

TEST(PtrAnalysis, IndexCastWidthAndRangeMatrix) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (unsigned iw : {32u, 64u})
    for (unsigned w : {16u, 32u, 64u, 128u})
      for (bool toIndex : {false, true})
        for (bool ui : {false, true})
          for (int rank : {-1, 0, 1}) {
            OpBuilder b(&context);
            auto module = ModuleOp::create(b.getUnknownLoc());
            OwningOpRef<ModuleOp> owner(module);
            b.setInsertionPointToStart(module.getBody());
            Type in =
                toIndex ? Type(b.getIntegerType(w)) : Type(b.getIndexType());
            Type out =
                toIndex ? Type(b.getIndexType()) : Type(b.getIntegerType(w));
            if (rank >= 0) {
              SmallVector<int64_t> shape =
                  rank == 0 ? SmallVector<int64_t>{} : SmallVector<int64_t>{4};
              in = RankedTensorType::get(shape, in);
              out = RankedTensorType::get(shape, out);
            }
            auto f = b.create<triton::FuncOp>(b.getUnknownLoc(), "cast",
                                              b.getFunctionType({in}, {}));
            b.setInsertionPointToStart(f.addEntryBlock());
            Value x = f.getArgument(0);
            Value y = ui ? Value(b.create<arith::IndexCastUIOp>(
                               b.getUnknownLoc(), out, x))
                         : Value(b.create<arith::IndexCastOp>(b.getUnknownLoc(),
                                                              out, x));
            b.create<triton::ReturnOp>(b.getUnknownLoc());
            PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules({64, iw}));
            auto result = test::getIntegerInfo(analysis.analyze(y));
            ASSERT_TRUE(succeeded(result));
            for (int64_t n : {-1LL, 0LL, 1LL, 2147483647LL, 2147483648LL}) {
              test::Lanes lanes;
              for (unsigned lane = 0; lane < (rank == 1 ? 4u : 1u); ++lane)
                lanes.push_back(llvm::APInt(64, n + lane, true)
                                    .sextOrTrunc(toIndex ? w : iw));
              test::Environment env{{x, lanes}};
              auto actual = evaluate(OpFoldResult(y), env, iw);
              ASSERT_TRUE(actual);
              auto source = evaluate(result->offsets.fullOffsets, env, iw);
              ASSERT_TRUE(source);
              EXPECT_EQ(*source, *actual);
              for (const auto &lane : *actual)
                EXPECT_TRUE(result->range.contains(lane));
              // Descriptor arithmetic has the target index width; source
              // values and range facts retain the original result width.
              b.setInsertionPoint(f.getBody().front().getTerminator());
              SmallVector<Value> indices;
              if (rank == 1)
                indices.push_back(
                    b.create<arith::ConstantIndexOp>(f.getLoc(), 0));
              auto lane = materializeOffsetAt(result->offsets, indices, b,
                                              f.getLoc(), {64, iw});
              unsigned targetWidth = toIndex ? iw : w;
              if (targetWidth > iw) {
                EXPECT_TRUE(failed(lane));
              } else {
                ASSERT_TRUE(succeeded(lane));
                auto materialized = evaluate(*lane, env, iw);
                ASSERT_TRUE(materialized);
                EXPECT_EQ(materialized->front(),
                          actual->front().sextOrTrunc(iw));
              }
            }
            EXPECT_EQ(mlir::triton::ptr::getIntegerInfo(*analysis.findCached(y))
                          ->offsets.fullOffsets,
                      result->offsets.fullOffsets);
            auto input = test::getIntegerInfo(analysis.analyze(x));
            ASSERT_TRUE(succeeded(input));
            unsigned sourceWidth = toIndex ? w : iw;
            input->range = llvm::ConstantRange(llvm::APInt(sourceWidth, 0),
                                               llvm::APInt(sourceWidth, 4));
            ASSERT_TRUE(
                succeeded(analysis.bind(x, CommonPtrRules::Result(*input))));
            auto bounded = test::getIntegerInfo(analysis.analyze(y));
            ASSERT_TRUE(succeeded(bounded));
            unsigned targetWidth = toIndex ? iw : w;
            EXPECT_EQ(bounded->range,
                      llvm::ConstantRange(llvm::APInt(targetWidth, 0),
                                          llvm::APInt(targetWidth, 4)));
            EXPECT_TRUE(succeeded(verify(module)));
          }
}

TEST(PtrAnalysis, IndexConstantArithmeticMatrix) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (unsigned iw : {32u, 64u}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @index() {
        %a = arith.constant 2147483647 : index
        %b = arith.constant 1 : index
        %n = arith.constant -2147483648 : index
        %sum = arith.addi %a, %b : index
        %sub = arith.subi %n, %b : index
        %mul = arith.muli %a, %n : index
        %i = arith.constant -7 : i64
        %c = arith.index_cast %i : i64 to index
        %u = arith.index_castui %i : i64 to index
        tt.return
      } }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    SmallVector<Value> values;
    module->walk([&](Operation *op) {
      if (op->getNumResults() == 1 && op->getResult(0).getType().isIndex())
        values.push_back(op->getResult(0));
    });
    OpBuilder b(&context);
    PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{{64, iw}});
    for (Value v : values) {
      auto result = test::getOffsets(analysis.analyze(v));
      ASSERT_TRUE(succeeded(result));
      ASSERT_TRUE(result->scalarOffset);
      if (auto attr = dyn_cast<Attribute>(*result->scalarOffset)) {
        EXPECT_EQ(cast<IntegerAttr>(attr).getValue().getBitWidth(), 64u);
      } else {
        EXPECT_TRUE(cast<Value>(*result->scalarOffset).getType().isIndex());
      }
      EXPECT_EQ(evaluate(result->scalarOffset, test::Environment(), iw),
                evaluate(OpFoldResult(v), test::Environment(), iw));
      ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(v))));
      EXPECT_TRUE(succeeded(verify(*module)));
    }
  }
}

TEST(PtrAnalysis, UnsignedWideningPreservesProvenShapeAndNegativeStride) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @unsigned(%unknown: tensor<4xi32>) {
      %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
      %e = tt.expand_dims %r {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %b = tt.broadcast %e : tensor<1x4xi32> -> tensor<2x4xi32>
      %u = arith.extui %b : tensor<2x4xi32> to tensor<2x4xi64>
      %three = arith.constant dense<3> : tensor<4xi32>
      %reverse = arith.subi %three, %r : tensor<4xi32>
      %descending = arith.extui %reverse : tensor<4xi32> to tensor<4xi64>
      %opaque = arith.extui %unknown : tensor<4xi32> to tensor<4xi64>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  SmallVector<arith::ExtUIOp> casts;
  module->walk([&](arith::ExtUIOp op) { casts.push_back(op); });
  for (unsigned k = 0; k < casts.size(); ++k) {
    auto state = test::getOffsets(analysis.analyze(casts[k]));
    ASSERT_TRUE(succeeded(state));
    EXPECT_EQ(cast<Value>(state->fullOffsets), casts[k].getResult());
    if (k == 2) {
      EXPECT_EQ(state->axes[0].kind, AxisKind::Unknown);
      continue;
    }
    auto actual = evaluate(*state);
    ASSERT_TRUE(actual);
    auto c = evaluate(state->scalarOffset);
    ASSERT_TRUE(c);
    auto s = evaluate(state->axes.back().stride);
    ASSERT_TRUE(s);
    EXPECT_EQ(s->front(), llvm::APInt(64, k == 0 ? 1 : -1, true));
    for (unsigned lane = 0; lane < actual->size(); ++lane)
      EXPECT_EQ((*actual)[lane],
                c->front() + llvm::APInt(64, lane % 4) * s->front());
    auto address = test::getOffsets(analysis.analyze(casts[k]));
    ASSERT_TRUE(succeeded(address));
    EXPECT_EQ(test::axisKinds(*address), test::axisKinds(*state));
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, IntegerSelectUsesTheSameConditionForAffineFields) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @select(%cond: i1, %mask: tensor<4xi1>) {
      %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
      %three = arith.constant dense<3> : tensor<4xi32>
      %reverse = arith.subi %three, %r : tensor<4xi32>
      %s = arith.select %cond, %r, %reverse : tensor<4xi32>
      %same = arith.select %mask, %r, %r : tensor<4xi1>, tensor<4xi32>
      %mixed = arith.select %mask, %r, %reverse : tensor<4xi1>, tensor<4xi32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<arith::SelectOp> selects;
  f.walk([&](arith::SelectOp op) { selects.push_back(op); });
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  for (unsigned k = 0; k < selects.size(); ++k) {
    auto result = test::getIntegerInfo(analysis.analyze(selects[k]));
    ASSERT_TRUE(succeeded(result));
    auto &state = result->offsets;
    EXPECT_EQ(cast<Value>(state.fullOffsets), selects[k].getResult());
    EXPECT_EQ(state.axes[0].kind,
              k == 2 ? AxisKind::Unknown : AxisKind::Structured);
    for (bool condition : {false, true}) {
      test::Environment env{{f.getArgument(0), {llvm::APInt(1, condition)}},
                            {f.getArgument(1),
                             {llvm::APInt(1, 0), llvm::APInt(1, 1),
                              llvm::APInt(1, 0), llvm::APInt(1, 1)}}};
      auto actual = evaluate(state, env);
      ASSERT_TRUE(actual);
      for (unsigned lane = 0; lane < 4; ++lane) {
        EXPECT_TRUE(result->range.contains(
            (*actual)[lane].sextOrTrunc(result->range.getBitWidth())));
        if (k != 2) {
          auto c = evaluate(state.scalarOffset, env),
               s = evaluate(state.axes[0].stride, env);
          ASSERT_TRUE(c);
          ASSERT_TRUE(s);
          EXPECT_EQ((*actual)[lane],
                    c->front() + llvm::APInt(64, lane) * s->front());
        }
      }
    }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysis, OpaqueRowsKeepBroadcastColumnStrides) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @rows(%base: !tt.ptr<i32>, %rows: tensor<2x1xi64>) {
      %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
      %r64 = arith.extsi %r : tensor<4xi32> to tensor<4xi64>
      %col = tt.expand_dims %r64 {axis = 0 : i32} : tensor<4xi64> -> tensor<1x4xi64>
      %cols = tt.broadcast %col : tensor<1x4xi64> -> tensor<2x4xi64>
      %wide_rows = tt.broadcast %rows : tensor<2x1xi64> -> tensor<2x4xi64>
      %sum = arith.addi %wide_rows, %cols : tensor<2x4xi64>
      %bases = tt.splat %base : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
      %row_ptr = tt.addptr %bases, %rows : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi64>
      %wide_ptr = tt.broadcast %row_ptr : tensor<2x1x!tt.ptr<i32>> -> tensor<2x4x!tt.ptr<i32>>
      %ptr = tt.addptr %wide_ptr, %cols : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi64>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto sum = *f.getBody().front().getOps<arith::AddIOp>().begin();
  triton::AddPtrOp ptr;
  f.walk([&](triton::AddPtrOp op) { ptr = op; });
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto offset = test::getOffsets(analysis.analyze(sum));
  auto ptrInfo = test::getPtrInfo(analysis.analyze(ptr));
  ASSERT_TRUE(succeeded(offset));
  ASSERT_TRUE(succeeded(ptrInfo));
  test::Environment env{
      {f.getArgument(1), {llvm::APInt(64, -13, true), llvm::APInt(64, 19)}}};
  for (const auto &info : {*offset, ptrInfo->offsets}) {
    EXPECT_EQ(test::axisKinds(info),
              SmallVector<AxisKind>({AxisKind::Unknown, AxisKind::Structured}));
    EXPECT_FALSE(info.scalarOffset);
    EXPECT_TRUE(info.axes.size() <= 0 ||
                info.axes[0].kind == AxisKind::Unknown);
    ASSERT_NE(info.axes[1].kind, AxisKind::Unknown);
    EXPECT_EQ(scalar(info.axes[1].stride), 1);
    auto values = evaluate(info, env);
    ASSERT_TRUE(values);
    ASSERT_EQ(values->size(), 8u);
    for (unsigned i = 0; i < 8; ++i)
      EXPECT_EQ((*values)[i].getSExtValue(),
                (i / 4 == 0 ? -13 : 19) + int(i % 4));
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, MixedAxisPtrBroadcastAndUniformDeltaRemainExact) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @mixed(%base: !tt.ptr<i32>, %rows: tensor<2xi64>, %uniform: i32) {
      %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
      %base2 = tt.splat %base : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
      %p = tt.addptr %base2, %row : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi64>
      %wide = tt.broadcast %p : tensor<2x1x!tt.ptr<i32>> -> tensor<2x4x!tt.ptr<i32>>
      %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
      %col = tt.expand_dims %r {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %cols = tt.broadcast %col : tensor<1x4xi32> -> tensor<2x4xi32>
      %indexed = tt.addptr %wide, %cols : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi32>
      %u = tt.splat %uniform : i32 -> tensor<2x4xi32>
      %next = tt.addptr %indexed, %u : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<triton::AddPtrOp> pointers;
  f.walk([&](triton::AddPtrOp op) { pointers.push_back(op); });
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto result = test::getPtrInfo(analysis.analyze(pointers.back()));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(test::axisKinds(result->offsets),
            SmallVector<AxisKind>({AxisKind::Unknown, AxisKind::Structured}));
  EXPECT_FALSE(result->offsets.scalarOffset);
  EXPECT_EQ(scalar(result->offsets.axes[0].stride), 0);
  EXPECT_EQ(scalar(result->offsets.axes[1].stride), 1);
  test::Environment env{
      {f.getArgument(1), {llvm::APInt(64, -13, true), llvm::APInt(64, 19)}},
      {f.getArgument(2), {llvm::APInt(32, -7, true)}}};
  auto values = evaluate(result->offsets, env);
  ASSERT_TRUE(values);
  for (unsigned i = 0; i < 8; ++i)
    EXPECT_EQ((*values)[i],
              llvm::APInt(64, (i / 4 == 0 ? -13 : 19) + int(i % 4) - 7, true));
  Value offset = pointers.back().getOffset();
  auto before = test::getOffsets(analysis.analyze(offset));
  ASSERT_TRUE(succeeded(before));
  analysis.clear();
  b.setInsertionPoint(f.getBody().front().getTerminator());
  b.create<arith::AddIOp>(f.getLoc(), offset, offset);
  auto after = test::getOffsets(analysis.analyze(offset));
  ASSERT_TRUE(succeeded(after));
  EXPECT_EQ(test::axisKinds(*before), test::axisKinds(*after));
  EXPECT_EQ(before->fullOffsets, after->fullOffsets);
  EXPECT_EQ(evaluate(before->scalarOffset, env),
            evaluate(after->scalarOffset, env));
  EXPECT_EQ(test::axisStrides(*before), test::axisStrides(*after));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, BoundNonSplatConstantPreservesEveryLane) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @bound(%x: tensor<4xi8>) { tt.return } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  Value x = f.getArgument(0);
  auto binding = test::getOffsets(analysis.analyze(x));
  ASSERT_TRUE(succeeded(binding));
  SmallVector<llvm::APInt> lanes;
  for (int64_t n : {-128, -1, 0, 127})
    lanes.push_back(llvm::APInt(8, n, true));
  binding->fullOffsets =
      DenseIntElementsAttr::get(cast<RankedTensorType>(x.getType()), lanes);
  ASSERT_TRUE(succeeded(analysis.bind(
      x, *makeIntegerInfo(*binding, analysis.getRules().getOptions()))));
  auto address = test::getOffsets(analysis.analyze(x));
  ASSERT_TRUE(succeeded(address));
  auto actual = evaluate(*address);
  ASSERT_TRUE(actual);
  ASSERT_EQ(actual->size(), lanes.size());
  for (unsigned i = 0; i < lanes.size(); ++i)
    EXPECT_EQ((*actual)[i], lanes[i].sext(64));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, RankZeroSplatAddressAndPtrHaveScalarUniform) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (unsigned width : {8u, 32u}) {
    OpBuilder b(&context);
    auto module = ModuleOp::create(b.getUnknownLoc());
    OwningOpRef<ModuleOp> owner(module);
    b.setInsertionPointToStart(module.getBody());
    auto ptr = triton::PointerType::get(b.getI32Type(), 1);
    auto integer = b.getIntegerType(width);
    auto f = b.create<triton::FuncOp>(b.getUnknownLoc(), "rank_zero",
                                      b.getFunctionType({ptr, integer}, {}));
    b.setInsertionPointToStart(f.addEntryBlock());
    Value p = b.create<triton::SplatOp>(
        f.getLoc(), RankedTensorType::get({}, ptr), f.getArgument(0));
    Value d = b.create<triton::SplatOp>(
        f.getLoc(), RankedTensorType::get({}, integer), f.getArgument(1));
    Value next = b.create<triton::AddPtrOp>(f.getLoc(), p.getType(), p, d);
    b.create<triton::ReturnOp>(f.getLoc());
    PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
    auto source = test::getOffsets(analysis.analyze(d));
    ASSERT_TRUE(succeeded(source));
    auto origin =
        cast<Value>(*source->scalarOffset).getDefiningOp<arith::IndexCastOp>();
    ASSERT_TRUE(origin);
    EXPECT_EQ(origin.getIn(), f.getArgument(1));
    auto delta = test::getOffsets(analysis.analyze(d));
    ASSERT_TRUE(succeeded(delta));
    EXPECT_EQ(delta->fullOffsets, OpFoldResult(d));
    EXPECT_EQ(delta->valueType, d.getType());
    EXPECT_TRUE(delta->addressTerms.empty());
    ASSERT_TRUE(cast<Value>(*delta->scalarOffset).getType().isIndex());
    auto ptrInfo = test::getPtrInfo(analysis.analyze(next));
    ASSERT_TRUE(succeeded(ptrInfo));
    EXPECT_EQ(ptrInfo->base, f.getArgument(0));
    ASSERT_TRUE(succeeded(analysis.bind(next, *ptrInfo)));
    for (llvm::APInt n : {llvm::APInt::getSignedMinValue(width),
                          llvm::APInt::getSignedMaxValue(width),
                          llvm::APInt(width, -1, true)}) {
      test::Environment env{{f.getArgument(1), {n}}};
      for (auto actual :
           {evaluate(*delta, env), evaluate(delta->scalarOffset, env),
            evaluate(ptrInfo->offsets, env),
            evaluate(ptrInfo->offsets.scalarOffset, env)}) {
        ASSERT_TRUE(actual);
        ASSERT_EQ(actual->size(), 1u);
        EXPECT_EQ(actual->front(), n.sext(64));
      }
    }
    EXPECT_TRUE(succeeded(verify(module)));
  }
}

TEST(PtrAnalysis, RankZeroOpaqueLeafCanBeReboundAndConsumed) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @opaque(%x: tensor<i32>) {
      %sum = arith.addi %x, %x : tensor<i32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  Value x = f.getArgument(0);
  Value sum = *f.getBody().front().getOps<arith::AddIOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto state = test::getOffsets(analysis.analyze(x));
  ASSERT_TRUE(succeeded(state));
  ASSERT_TRUE(cast<Value>(*state->scalarOffset).getType().isIndex());
  ASSERT_TRUE(succeeded(analysis.bind(
      x, *makeIntegerInfo(*state, analysis.getRules().getOptions()))));
  auto derived = test::getOffsets(analysis.analyze(sum));
  ASSERT_TRUE(succeeded(derived));
  ASSERT_TRUE(cast<Value>(*derived->scalarOffset).getType().isIndex());
  for (int64_t n : {-2147483648LL, -7LL, 2147483647LL}) {
    test::Environment env{{x, {llvm::APInt(32, n, true)}}};
    auto expected = (llvm::APInt(32, n, true) * llvm::APInt(32, 2)).sext(64);
    auto complete = evaluate(*derived, env);
    auto uniform = evaluate(derived->scalarOffset, env);
    ASSERT_TRUE(complete);
    ASSERT_TRUE(uniform);
    EXPECT_EQ(complete->front(), expected);
    EXPECT_EQ(uniform->front(), expected);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, DynamicTensorIndexPreservesTargetWidth) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (unsigned iw : {32u, 64u}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @index(%x: tensor<4xindex>) { tt.return } }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder b(&context);
    PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{{64, iw}});
    auto address = test::getOffsets(analysis.analyze(f.getArgument(0)));
    ASSERT_TRUE(succeeded(address));
    test::Lanes lanes{llvm::APInt::getSignedMinValue(iw),
                      llvm::APInt(iw, -1, true), llvm::APInt(iw, 0),
                      llvm::APInt::getSignedMaxValue(iw)};
    test::Environment env{{f.getArgument(0), lanes}};
    auto actual = evaluate(*address, env, iw);
    ASSERT_TRUE(actual);
    ASSERT_EQ(actual->size(), lanes.size());
    for (unsigned i = 0; i < lanes.size(); ++i)
      EXPECT_EQ((*actual)[i], lanes[i]);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysis, TensorIndexConstantsAndBindingsUseTargetWidth) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (unsigned iw : {32u, 64u})
    for (bool splat : {false, true}) {
      OpBuilder b(&context);
      auto module = ModuleOp::create(b.getUnknownLoc());
      OwningOpRef<ModuleOp> owner(module);
      b.setInsertionPointToStart(module.getBody());
      auto type = RankedTensorType::get({4}, b.getIndexType());
      auto f = b.create<triton::FuncOp>(b.getUnknownLoc(), "constants",
                                        b.getFunctionType({type}, {}));
      b.setInsertionPointToStart(f.addEntryBlock());
      test::Lanes stored;
      for (int64_t n :
           {-2147483648LL, 2147483647LL, 2147483648LL, 4294967295LL})
        stored.push_back(llvm::APInt(64, splat ? 4294967295LL : n, true));
      auto attr = DenseIntElementsAttr::get(type, stored);
      Value constant = b.create<arith::ConstantOp>(f.getLoc(), attr);
      b.create<triton::ReturnOp>(f.getLoc());
      PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{{64, iw}});
      auto bound = test::getOffsets(analysis.analyze(f.getArgument(0)));
      ASSERT_TRUE(succeeded(bound));
      bound->fullOffsets = attr;
      ASSERT_TRUE(succeeded(analysis.bind(
          f.getArgument(0),
          *makeIntegerInfo(*bound, analysis.getRules().getOptions()))));
      for (Value value : {constant, Value(f.getArgument(0))}) {
        auto offsets = test::getOffsets(analysis.analyze(value));
        ASSERT_TRUE(succeeded(offsets));
        auto actual = evaluate(*offsets, test::Environment(), iw);
        ASSERT_TRUE(actual);
        auto source = evaluate(OpFoldResult(attr), test::Environment(), iw);
        ASSERT_TRUE(source);
        ASSERT_EQ(actual->size(), stored.size());
        for (unsigned lane = 0; lane < stored.size(); ++lane) {
          llvm::APInt expected = stored[lane].sextOrTrunc(iw);
          EXPECT_EQ((*source)[lane], expected);
          EXPECT_EQ((*actual)[lane], expected);
        }
        EXPECT_TRUE(succeeded(verify(module)));
      }
    }
}

TEST(PtrAnalysis, RankZeroIndexAndEncodedTensorKeepTheirTypes) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (unsigned iw : {32u, 64u})
    for (bool rankZero : {false, true}) {
      OpBuilder b(&context);
      auto module = ModuleOp::create(b.getUnknownLoc());
      OwningOpRef<ModuleOp> owner(module);
      b.setInsertionPointToStart(module.getBody());
      auto shape = rankZero ? SmallVector<int64_t>{} : SmallVector<int64_t>{4};
      auto encoding = b.getStringAttr("test_encoding");
      auto type = RankedTensorType::get(shape, b.getIndexType(), encoding);
      auto f = b.create<triton::FuncOp>(b.getUnknownLoc(), "encoded",
                                        b.getFunctionType({type}, {}));
      b.setInsertionPointToStart(f.addEntryBlock());
      b.create<triton::ReturnOp>(f.getLoc());
      PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{{64, iw}});
      auto source = test::getOffsets(analysis.analyze(f.getArgument(0)));
      ASSERT_TRUE(succeeded(source));
      ASSERT_TRUE(succeeded(analysis.bind(
          f.getArgument(0),
          *makeIntegerInfo(*source, analysis.getRules().getOptions()))));
      auto offsets = test::getOffsets(analysis.analyze(f.getArgument(0)));
      ASSERT_TRUE(succeeded(offsets));
      EXPECT_EQ(offsets->valueType,
                RankedTensorType::get(shape, b.getIndexType(), encoding));
      test::Lanes lanes(rankZero ? 1 : 4, llvm::APInt(iw, -1, true));
      test::Environment env{{f.getArgument(0), lanes}};
      auto actual = evaluate(*offsets, env, iw);
      ASSERT_TRUE(actual);
      EXPECT_EQ(*actual,
                test::Lanes(rankZero ? 1 : 4, llvm::APInt(iw, -1, true)));
      if (rankZero) {
        ASSERT_TRUE(cast<Value>(*source->scalarOffset).getType().isIndex());
        ASSERT_TRUE(cast<Value>(*offsets->scalarOffset).getType().isIndex());
        EXPECT_EQ(evaluate(offsets->scalarOffset, env, iw), actual);
      }
      EXPECT_TRUE(succeeded(verify(module)));
    }
}

TEST(PtrAnalysis, OpaqueFactoryPreservesSourceAndRankZeroContracts) {
  EXPECT_FALSE(OffsetInfo{}.hasStructuredForm());
  EXPECT_FALSE(OffsetInfo{}.isUniform());
  EXPECT_FALSE(OffsetInfo{}.scalarOffset);
  EXPECT_TRUE(OffsetInfo{}.axes.size() <= 0 ||
              OffsetInfo{}.axes[0].kind == AxisKind::Unknown);
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @opaque(%scalar: i32, %lanes: tensor<4xi64>,
                            %rankzero: tensor<i64>, %idx: index, %float: f32) {
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  builder.setInsertionPoint(f.getBody().front().getTerminator());
  auto point = builder.getInsertionPoint();
  auto scalar = makeOpaqueOffset(f.getArgument(0), builder);
  ASSERT_TRUE(succeeded(scalar));
  EXPECT_TRUE(scalar->hasStructuredForm());
  EXPECT_TRUE(scalar->isUniform());
  ASSERT_TRUE(scalar->scalarOffset);
  EXPECT_TRUE(cast<Value>(*scalar->scalarOffset).getType().isIndex());
  EXPECT_TRUE(scalar->axes.size() <= 0 ||
              scalar->axes[0].kind == AxisKind::Unknown);
  EXPECT_TRUE(failed(makeOpaqueOffset(f.getArgument(4), builder)));
  EXPECT_TRUE(failed(makeOpaqueOffset(Value(), builder)));
  EXPECT_TRUE(succeeded(makeOpaqueOffset(f.getArgument(3), builder)));
  auto lanes = makeOpaqueOffset(f.getArgument(1), builder);
  ASSERT_TRUE(succeeded(lanes));
  EXPECT_EQ(lanes->fullOffsets, OpFoldResult(f.getArgument(1)));
  EXPECT_FALSE(lanes->hasStructuredForm());
  EXPECT_FALSE(lanes->isUniform());
  EXPECT_FALSE(lanes->scalarOffset);
  EXPECT_TRUE(lanes->axes.size() <= 0 ||
              lanes->axes[0].kind == AxisKind::Unknown);
  EXPECT_TRUE(lanes->axes.size() <= 1 ||
              lanes->axes[1].kind == AxisKind::Unknown);
  auto rankzero = makeOpaqueOffset(f.getArgument(2), builder);
  ASSERT_TRUE(succeeded(rankzero));
  ASSERT_TRUE(rankzero->scalarOffset);
  Value origin = cast<Value>(*rankzero->scalarOffset);
  EXPECT_TRUE(origin.getType().isIndex());
  auto castOp = origin.getDefiningOp<arith::IndexCastOp>();
  ASSERT_TRUE(castOp);
  EXPECT_TRUE(castOp.getIn().getDefiningOp<tensor::ExtractOp>());
  EXPECT_EQ(builder.getInsertionPoint(), point);
  test::Environment inputs;
  inputs[f.getArgument(2)] = {llvm::APInt(64, -19)};
  auto evaluated = evaluate(origin, inputs);
  ASSERT_TRUE(evaluated);
  EXPECT_EQ(evaluated->front().getSExtValue(), -19);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, RangeProofSurvivesArithmeticCastsAndShapes) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @ranges(%cond: i1) {
      %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
      %one = arith.constant dense<1> : tensor<4xi32>
      %three = arith.constant dense<3> : tensor<4xi32>
      %mul = arith.muli %r, %three : tensor<4xi32>
      %positive = arith.addi %mul, %one : tensor<4xi32>
      %negative = arith.subi %r, %one : tensor<4xi32>
      %selected = arith.select %cond, %positive, %positive : tensor<4xi32>
      %row = tt.expand_dims %selected {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %tile = tt.broadcast %row : tensor<1x4xi32> -> tensor<4x4xi32>
      %narrow = arith.trunci %tile : tensor<4x4xi32> to tensor<4x4xi8>
      %good = arith.extui %narrow : tensor<4x4xi8> to tensor<4x4xi64>
      %negative8 = arith.trunci %negative : tensor<4xi32> to tensor<4xi8>
      %bad = arith.extui %negative8 : tensor<4xi8> to tensor<4xi64>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<arith::ExtUIOp> casts;
  f.walk([&](arith::ExtUIOp op) { casts.push_back(op); });
  ASSERT_EQ(casts.size(), 2u);
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto good = test::getOffsets(analysis.analyze(casts[0]));
  auto bad = test::getOffsets(analysis.analyze(casts[1]));
  ASSERT_TRUE(succeeded(good));
  ASSERT_TRUE(succeeded(bad));
  EXPECT_TRUE(good->hasStructuredForm());
  EXPECT_EQ(test::axisKinds(*good),
            SmallVector<AxisKind>({AxisKind::Invariant, AxisKind::Structured}));
  EXPECT_EQ(scalar(good->scalarOffset), 1);
  EXPECT_EQ(scalar(good->axes[0].stride), 0);
  EXPECT_EQ(scalar(good->axes[1].stride), 3);
  // Nonnegative range information propagates through all these operations.
  // Extending a range that includes -1 as unsigned cannot keep stride 1.
  EXPECT_FALSE(bad->hasStructuredForm());
  EXPECT_EQ(test::axisKinds(*bad), SmallVector<AxisKind>({AxisKind::Unknown}));
  test::Environment env{{f.getArgument(0), {llvm::APInt(1, 1)}}};
  auto values = evaluate(*good, env);
  auto unsignedValues = evaluate(*bad);
  ASSERT_TRUE(values);
  ASSERT_TRUE(unsignedValues);
  ASSERT_EQ(values->size(), 16u);
  for (unsigned lane = 0; lane < values->size(); ++lane)
    EXPECT_EQ((*values)[lane], llvm::APInt(64, 1 + 3 * (lane % 4)));
  EXPECT_EQ(*unsignedValues,
            (test::Lanes{llvm::APInt(64, 255), llvm::APInt(64, 0),
                         llvm::APInt(64, 1), llvm::APInt(64, 2)}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
