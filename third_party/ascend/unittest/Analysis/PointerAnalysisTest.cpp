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

#include "TritonMemoryAccess/PointerAnalysis.h"
#include "PointerAnalysisTestUtils.h"
#include "TritonMemoryAccess/PointerAnalysisTransfer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pointer;

namespace {
namespace pointer_detail = mlir::triton::pointer::detail;

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

TEST(PointerAnalysis, SignedDeltaIsAppliedAtEveryAddPtr) {
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
  PointerAnalysis analysis(builder);
  auto source = analysis.analyzeOffset(add.getResult());
  ASSERT_TRUE(succeeded(source));
  EXPECT_EQ(source->domain, ArithmeticDomain::SourceInteger);
  auto delta = analysis.analyzeAddressDelta(add.getResult());
  ASSERT_TRUE(succeeded(delta));
  ASSERT_EQ(delta->domain, ArithmeticDomain::ElementAddress);
  EXPECT_EQ(
      evaluate(cast<Value>(delta->completeOffset))->front().getSExtValue(),
      -128);
  auto once = analysis.analyzePointer(pointers[0]);
  auto twice = analysis.analyzePointer(pointers[2]);
  ASSERT_TRUE(succeeded(once));
  ASSERT_TRUE(succeeded(twice));
  EXPECT_EQ(once->base, func.getArgument(0));
  EXPECT_EQ(twice->base, func.getArgument(0));
  EXPECT_EQ(scalar(once->offsets.completeOffset), -128);
  EXPECT_EQ(scalar(twice->offsets.completeOffset), 128);
  EXPECT_TRUE(analysis.findCachedOffset(add.getResult()));
  EXPECT_TRUE(analysis.findCachedAddressDelta(add.getResult()));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, NarrowWrapLosesAddressStrideButKeepsExactTensor) {
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
  PointerAnalysis analysis(builder);
  auto address = analysis.analyzeAddressDelta(narrow);
  auto source = analysis.analyzeOffset(narrow);
  ASSERT_TRUE(succeeded(address));
  ASSERT_TRUE(succeeded(source));
  EXPECT_EQ(source->axes, SmallVector<AxisKind>({AxisKind::Structured}));
  EXPECT_EQ(address->axes, SmallVector<AxisKind>({AxisKind::Unknown}));
  EXPECT_EQ(scalar(address->uniformOffset), 0);
  EXPECT_EQ(scalar(address->strides[0]), 0);
  auto values = evaluate(cast<Value>(address->completeOffset));
  ASSERT_TRUE(values);
  ASSERT_EQ(values->size(), 4u);
  const int64_t expected[] = {126, 127, -128, -127};
  for (unsigned i = 0; i < values->size(); ++i)
    EXPECT_EQ((*values)[i].getSExtValue(), expected[i]);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, AKnownAxisSurvivesAnUnrelatedOpaqueAxis) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @partial(%row: tensor<2xi32>) {
        %row_expanded = tt.expand_dims %row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
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
  PointerAnalysis analysis(builder);
  auto result = analysis.analyzeOffset(sum);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->shape, SmallVector<int64_t>({2, 4}));
  EXPECT_EQ(result->axes,
            SmallVector<AxisKind>({AxisKind::Unknown, AxisKind::Structured}));
  EXPECT_EQ(result->completeOffset, OpFoldResult(sum.getResult()));
  EXPECT_EQ(scalar(result->uniformOffset), 0);
  EXPECT_EQ(scalar(result->strides[0]), 0);
  EXPECT_EQ(scalar(result->strides[1]), 1);
  EXPECT_FALSE(result->hasAffineForm());
  EXPECT_FALSE(result->getAffineOrigin());
  EXPECT_FALSE(result->getKnownStride(0));
  ASSERT_TRUE(result->getKnownStride(1));
  EXPECT_EQ(scalar(*result->getKnownStride(1)), 1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, IndexWidthMustBeExplicitAndWideSourceIsPreserved) {
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
  PointerAnalysis unspecified(builder);
  EXPECT_TRUE(failed(unspecified.analyzeOffset(func.getArgument(0))));
  PointerAnalysis configured(builder, AnalysisOptions{64, 64});
  EXPECT_TRUE(succeeded(configured.analyzeOffset(func.getArgument(0))));
  EXPECT_TRUE(succeeded(configured.analyzeAddressDelta(func.getArgument(0))));
  auto wideSource = configured.analyzeOffset(wide);
  ASSERT_TRUE(succeeded(wideSource));
  Attribute wideUniform = dyn_cast<Attribute>(wideSource->uniformOffset);
  ASSERT_TRUE(wideUniform);
  EXPECT_EQ(cast<IntegerAttr>(wideUniform).getValue().getBitWidth(), 128u);
  EXPECT_TRUE(failed(configured.analyzeAddressDelta(wide)));
  PointerAnalysis invalidAddress(builder, AnalysisOptions{32, 64});
  EXPECT_TRUE(failed(invalidAddress.analyzeAddressDelta(func.getArgument(0))));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, BroadcastDoesNotInventAUniformPointerOffset) {
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
  PointerAnalysis analysis(builder);
  auto pointer = analysis.analyzePointer(repeated);
  ASSERT_TRUE(succeeded(pointer));
  EXPECT_EQ(pointer->base, func.getArgument(0));
  EXPECT_EQ(pointer->offsets.axes, SmallVector<AxisKind>({AxisKind::Unknown}));
  EXPECT_EQ(scalar(pointer->offsets.uniformOffset), 0);
  EXPECT_EQ(scalar(pointer->offsets.strides[0]), 0);
  EXPECT_TRUE(succeeded(verify(*module)));
}

// C1: each request completion must already produce valid SSA.
TEST(PointerAnalysis, UniformAddressHelpersDominateUses) {
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
    PointerAnalysis analysis(builder);
    if (order == 1) {
      ASSERT_TRUE(succeeded(analysis.analyzeOffset(sum)));
      ASSERT_TRUE(succeeded(verify(*module)));
    }
    ASSERT_TRUE(succeeded(analysis.analyzeAddressDelta(sum)));
    ASSERT_TRUE(succeeded(verify(*module)));
    ASSERT_TRUE(succeeded(analysis.analyzeOffset(sum)));
    ASSERT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PointerAnalysis, DenseSplatAddressConstantIsLegal) {
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
  PointerAnalysis analysis(builder);
  ASSERT_TRUE(succeeded(analysis.analyzeAddressDelta(x)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, SameWidthUnsignedIndexCastRange) {
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
  PointerAnalysis analysis(builder, AnalysisOptions{64, 64});
  ASSERT_TRUE(succeeded(analysis.analyzeOffset(y)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, Index32ConstantUsesIndexAttributeStorage) {
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
  PointerAnalysis analysis(builder, AnalysisOptions{64, 32});
  auto result = analysis.analyzeOffset(x);
  ASSERT_TRUE(succeeded(result));
  auto attr = cast<IntegerAttr>(cast<Attribute>(result->uniformOffset));
  EXPECT_EQ(attr.getValue().getBitWidth(), 64u);
  EXPECT_EQ(attr.getInt(), -1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, DenseSplatPointerLanes) {
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
      PointerAnalysis analysis(b);
      auto delta = analysis.analyzeAddressDelta(x);
      auto result = analysis.analyzePointer(add);
      ASSERT_TRUE(succeeded(delta));
      ASSERT_TRUE(succeeded(result));
      for (auto ofr : {delta->completeOffset, result->offsets.completeOffset}) {
        auto lanes = evaluate(ofr);
        ASSERT_TRUE(lanes);
        ASSERT_EQ(lanes->size(), 4u);
        for (auto lane : *lanes)
          EXPECT_EQ(lane, llvm::APInt(64, n, true));
      }
      EXPECT_TRUE(succeeded(verify(module)));
    }
}

TEST(PointerAnalysis, IndexCastWidthAndRangeMatrix) {
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
            pointer_detail::AnalysisDriver<pointer_detail::PointerRules> driver(
                b, pointer_detail::PointerRules({64, iw}));
            auto result = driver.analyze({y, pointer_detail::SourceInteger});
            ASSERT_TRUE(succeeded(result));
            ASSERT_TRUE(result->integerRange);
            for (int64_t n : {-1LL, 0LL, 1LL, 2147483647LL, 2147483648LL}) {
              test::Lanes lanes;
              for (unsigned lane = 0; lane < (rank == 1 ? 4u : 1u); ++lane)
                lanes.push_back(llvm::APInt(64, n + lane, true)
                                    .sextOrTrunc(toIndex ? w : iw));
              test::Environment env{{x, lanes}};
              auto actual = evaluate(OpFoldResult(y), env, iw);
              ASSERT_TRUE(actual);
              for (auto lane : *actual)
                EXPECT_TRUE(result->integerRange->contains(lane));
              auto address = driver.analyze({y, pointer_detail::AddressDelta});
              if (!toIndex && w > 64) {
                EXPECT_TRUE(failed(address));
                continue;
              }
              ASSERT_TRUE(succeeded(address));
              auto projected =
                  evaluate(address->offset->completeOffset, env, iw);
              ASSERT_TRUE(projected);
              ASSERT_EQ(projected->size(), actual->size());
              for (unsigned lane = 0; lane < actual->size(); ++lane) {
                EXPECT_EQ((*projected)[lane], (*actual)[lane].sextOrTrunc(64));
                EXPECT_TRUE(
                    address->integerRange->contains((*projected)[lane]));
              }
            }
            auto input = driver.analyze({x, pointer_detail::SourceInteger});
            ASSERT_TRUE(succeeded(input));
            unsigned sourceWidth = toIndex ? w : iw;
            input->integerRange = llvm::ConstantRange(
                llvm::APInt(sourceWidth, 0), llvm::APInt(sourceWidth, 4));
            driver.bindBoundary({x, pointer_detail::SourceInteger}, *input);
            auto bounded = driver.analyze({y, pointer_detail::SourceInteger});
            ASSERT_TRUE(succeeded(bounded));
            unsigned targetWidth = toIndex ? iw : w;
            EXPECT_EQ(*bounded->integerRange,
                      llvm::ConstantRange(llvm::APInt(targetWidth, 0),
                                          llvm::APInt(targetWidth, 4)));
            EXPECT_TRUE(succeeded(verify(module)));
          }
}

TEST(PointerAnalysis, IndexConstantArithmeticMatrix) {
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
    PointerAnalysis analysis(b, {64, iw});
    for (Value v : values) {
      auto result = analysis.analyzeOffset(v);
      ASSERT_TRUE(succeeded(result));
      auto attr = dyn_cast<IntegerAttr>(cast<Attribute>(result->uniformOffset));
      ASSERT_TRUE(attr);
      EXPECT_EQ(attr.getValue().getBitWidth(), 64u);
      EXPECT_EQ(evaluate(result->uniformOffset, test::Environment(), iw),
                evaluate(OpFoldResult(v), test::Environment(), iw));
      ASSERT_TRUE(succeeded(analysis.analyzeAddressDelta(v)));
      EXPECT_TRUE(succeeded(verify(*module)));
    }
  }
}

TEST(PointerAnalysis, UnsignedWideningPreservesProvenShapeAndNegativeStride) {
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
  PointerAnalysis analysis(b);
  SmallVector<arith::ExtUIOp> casts;
  module->walk([&](arith::ExtUIOp op) { casts.push_back(op); });
  for (unsigned k = 0; k < casts.size(); ++k) {
    auto state = analysis.analyzeOffset(casts[k]);
    ASSERT_TRUE(succeeded(state));
    EXPECT_EQ(cast<Value>(state->completeOffset), casts[k].getResult());
    if (k == 2) {
      EXPECT_EQ(state->axes[0], AxisKind::Unknown);
      continue;
    }
    auto actual = evaluate(state->completeOffset);
    ASSERT_TRUE(actual);
    auto c = evaluate(state->uniformOffset);
    ASSERT_TRUE(c);
    auto s = evaluate(state->strides.back());
    ASSERT_TRUE(s);
    EXPECT_EQ(s->front(), llvm::APInt(64, k == 0 ? 1 : -1, true));
    for (unsigned lane = 0; lane < actual->size(); ++lane)
      EXPECT_EQ((*actual)[lane],
                c->front() + llvm::APInt(64, lane % 4) * s->front());
    auto address = analysis.analyzeAddressDelta(casts[k]);
    ASSERT_TRUE(succeeded(address));
    EXPECT_EQ(address->axes, state->axes);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, IntegerSelectUsesTheSameConditionForAffineFields) {
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
  pointer_detail::AnalysisDriver<pointer_detail::PointerRules> driver(
      b, pointer_detail::PointerRules({}));
  for (unsigned k = 0; k < selects.size(); ++k) {
    auto result = driver.analyze({selects[k], pointer_detail::SourceInteger});
    ASSERT_TRUE(succeeded(result));
    auto &state = *result->offset;
    EXPECT_EQ(cast<Value>(state.completeOffset), selects[k].getResult());
    EXPECT_EQ(state.axes[0], k == 2 ? AxisKind::Unknown : AxisKind::Structured);
    for (bool condition : {false, true}) {
      test::Environment env{{f.getArgument(0), {llvm::APInt(1, condition)}},
                            {f.getArgument(1),
                             {llvm::APInt(1, 0), llvm::APInt(1, 1),
                              llvm::APInt(1, 0), llvm::APInt(1, 1)}}};
      auto actual = evaluate(state.completeOffset, env);
      ASSERT_TRUE(actual);
      for (unsigned lane = 0; lane < 4; ++lane) {
        EXPECT_TRUE(result->integerRange->contains((*actual)[lane]));
        if (k != 2) {
          auto c = evaluate(state.uniformOffset, env),
               s = evaluate(state.strides[0], env);
          ASSERT_TRUE(c);
          ASSERT_TRUE(s);
          EXPECT_EQ((*actual)[lane],
                    c->front() + llvm::APInt(32, lane) * s->front());
        }
      }
    }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PointerAnalysis, MixedAxisPointerBroadcastAndUniformDeltaRemainExact) {
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
  PointerAnalysis analysis(b);
  auto result = analysis.analyzePointer(pointers.back());
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->offsets.axes,
            SmallVector<AxisKind>({AxisKind::Unknown, AxisKind::Structured}));
  EXPECT_EQ(scalar(result->offsets.uniformOffset), 0);
  EXPECT_EQ(scalar(result->offsets.strides[0]), 0);
  EXPECT_EQ(scalar(result->offsets.strides[1]), 1);
  test::Environment env{
      {f.getArgument(1), {llvm::APInt(64, -13, true), llvm::APInt(64, 19)}},
      {f.getArgument(2), {llvm::APInt(32, -7, true)}}};
  auto values = evaluate(result->offsets.completeOffset, env);
  ASSERT_TRUE(values);
  for (unsigned i = 0; i < 8; ++i)
    EXPECT_EQ((*values)[i],
              llvm::APInt(64, (i / 4 == 0 ? -13 : 19) + int(i % 4) - 7, true));
  Value offset = pointers.back().getOffset();
  auto before = analysis.analyzeOffset(offset);
  ASSERT_TRUE(succeeded(before));
  analysis.clear();
  b.setInsertionPoint(f.getBody().front().getTerminator());
  b.create<arith::AddIOp>(f.getLoc(), offset, offset);
  auto after = analysis.analyzeOffset(offset);
  ASSERT_TRUE(succeeded(after));
  EXPECT_EQ(before->axes, after->axes);
  EXPECT_EQ(before->completeOffset, after->completeOffset);
  EXPECT_EQ(before->uniformOffset, after->uniformOffset);
  EXPECT_EQ(before->strides, after->strides);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, BoundNonSplatConstantPreservesEveryLane) {
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
  PointerAnalysis analysis(b);
  Value x = f.getArgument(0);
  auto binding = analysis.analyzeOffset(x);
  ASSERT_TRUE(succeeded(binding));
  SmallVector<llvm::APInt> lanes;
  for (int64_t n : {-128, -1, 0, 127})
    lanes.push_back(llvm::APInt(8, n, true));
  binding->completeOffset =
      DenseIntElementsAttr::get(cast<RankedTensorType>(x.getType()), lanes);
  ASSERT_TRUE(succeeded(analysis.bindOffset(x, *binding)));
  auto address = analysis.analyzeAddressDelta(x);
  ASSERT_TRUE(succeeded(address));
  auto actual = evaluate(address->completeOffset);
  ASSERT_TRUE(actual);
  ASSERT_EQ(actual->size(), lanes.size());
  for (unsigned i = 0; i < lanes.size(); ++i)
    EXPECT_EQ((*actual)[i], lanes[i].sext(64));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, RankZeroSplatAddressAndPointerHaveScalarUniform) {
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
    PointerAnalysis analysis(b);
    auto source = analysis.analyzeOffset(d);
    ASSERT_TRUE(succeeded(source));
    EXPECT_EQ(cast<Value>(source->uniformOffset), f.getArgument(1));
    auto delta = analysis.analyzeAddressDelta(d);
    ASSERT_TRUE(succeeded(delta));
    EXPECT_EQ(cast<Value>(delta->completeOffset).getType(),
              RankedTensorType::get({}, b.getI64Type()));
    ASSERT_TRUE(cast<Value>(delta->uniformOffset).getType().isInteger(64));
    auto pointer = analysis.analyzePointer(next);
    ASSERT_TRUE(succeeded(pointer));
    EXPECT_EQ(pointer->base, f.getArgument(0));
    ASSERT_TRUE(succeeded(analysis.bindPointer(next, *pointer)));
    for (llvm::APInt n : {llvm::APInt::getSignedMinValue(width),
                          llvm::APInt::getSignedMaxValue(width),
                          llvm::APInt(width, -1, true)}) {
      test::Environment env{{f.getArgument(1), {n}}};
      for (auto field :
           {delta->completeOffset, delta->uniformOffset,
            pointer->offsets.completeOffset, pointer->offsets.uniformOffset}) {
        auto actual = evaluate(field, env);
        ASSERT_TRUE(actual);
        ASSERT_EQ(actual->size(), 1u);
        EXPECT_EQ(actual->front(), n.sext(64));
      }
    }
    EXPECT_TRUE(succeeded(verify(module)));
  }
}

TEST(PointerAnalysis, RankZeroOpaqueLeafCanBeReboundAndConsumed) {
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
  PointerAnalysis analysis(b);
  auto state = analysis.analyzeOffset(x);
  ASSERT_TRUE(succeeded(state));
  ASSERT_TRUE(cast<Value>(state->uniformOffset).getType().isInteger(32));
  ASSERT_TRUE(succeeded(analysis.bindOffset(x, *state)));
  auto derived = analysis.analyzeAddressDelta(sum);
  ASSERT_TRUE(succeeded(derived));
  ASSERT_TRUE(cast<Value>(derived->uniformOffset).getType().isInteger(64));
  for (int64_t n : {-2147483648LL, -7LL, 2147483647LL}) {
    test::Environment env{{x, {llvm::APInt(32, n, true)}}};
    auto expected = (llvm::APInt(32, n, true) * llvm::APInt(32, 2)).sext(64);
    auto complete = evaluate(derived->completeOffset, env);
    auto uniform = evaluate(derived->uniformOffset, env);
    ASSERT_TRUE(complete);
    ASSERT_TRUE(uniform);
    EXPECT_EQ(complete->front(), expected);
    EXPECT_EQ(uniform->front(), expected);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysis, DynamicTensorIndexAddressProjection) {
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
    PointerAnalysis analysis(b, {64, iw});
    auto address = analysis.analyzeAddressDelta(f.getArgument(0));
    ASSERT_TRUE(succeeded(address));
    test::Lanes lanes{llvm::APInt::getSignedMinValue(iw),
                      llvm::APInt(iw, -1, true), llvm::APInt(iw, 0),
                      llvm::APInt::getSignedMaxValue(iw)};
    test::Environment env{{f.getArgument(0), lanes}};
    auto actual = evaluate(address->completeOffset, env, iw);
    ASSERT_TRUE(actual);
    ASSERT_EQ(actual->size(), lanes.size());
    for (unsigned i = 0; i < lanes.size(); ++i)
      EXPECT_EQ((*actual)[i], lanes[i].sextOrTrunc(64));
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PointerAnalysis, TensorIndexConstantsAndBindingsUseTargetWidth) {
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
      PointerAnalysis analysis(b, {64, iw});
      auto bound = analysis.analyzeOffset(f.getArgument(0));
      ASSERT_TRUE(succeeded(bound));
      bound->completeOffset = attr;
      ASSERT_TRUE(succeeded(analysis.bindOffset(f.getArgument(0), *bound)));
      for (Value value : {constant, Value(f.getArgument(0))}) {
        auto projected = analysis.analyzeAddressDelta(value);
        ASSERT_TRUE(succeeded(projected));
        auto actual =
            evaluate(projected->completeOffset, test::Environment(), iw);
        ASSERT_TRUE(actual);
        auto source = evaluate(OpFoldResult(attr), test::Environment(), iw);
        ASSERT_TRUE(source);
        ASSERT_EQ(actual->size(), stored.size());
        for (unsigned lane = 0; lane < stored.size(); ++lane) {
          llvm::APInt expected = stored[lane].sextOrTrunc(iw);
          EXPECT_EQ((*source)[lane], expected);
          EXPECT_EQ((*actual)[lane], expected.sextOrTrunc(64));
        }
        EXPECT_TRUE(succeeded(verify(module)));
      }
    }
}

TEST(PointerAnalysis, RankZeroIndexAndEncodedTensorKeepTheirTypes) {
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
      PointerAnalysis analysis(b, {64, iw});
      auto source = analysis.analyzeOffset(f.getArgument(0));
      ASSERT_TRUE(succeeded(source));
      ASSERT_TRUE(succeeded(analysis.bindOffset(f.getArgument(0), *source)));
      auto projected = analysis.analyzeAddressDelta(f.getArgument(0));
      ASSERT_TRUE(succeeded(projected));
      EXPECT_EQ(projected->valueType,
                RankedTensorType::get(shape, b.getI64Type(), encoding));
      test::Lanes lanes(rankZero ? 1 : 4, llvm::APInt(iw, -1, true));
      test::Environment env{{f.getArgument(0), lanes}};
      auto actual = evaluate(projected->completeOffset, env, iw);
      ASSERT_TRUE(actual);
      EXPECT_EQ(*actual,
                test::Lanes(rankZero ? 1 : 4, llvm::APInt(64, -1, true)));
      if (rankZero) {
        ASSERT_TRUE(cast<Value>(source->uniformOffset).getType().isIndex());
        ASSERT_TRUE(
            cast<Value>(projected->uniformOffset).getType().isInteger(64));
        EXPECT_EQ(evaluate(projected->uniformOffset, env, iw), actual);
      }
      EXPECT_TRUE(succeeded(verify(module)));
    }
}

TEST(PointerAnalysis, OpaqueFactoryPreservesDomainAndRankZeroContracts) {
  EXPECT_FALSE(OffsetComponents{}.hasAffineForm());
  EXPECT_FALSE(OffsetComponents{}.isUniform());
  EXPECT_FALSE(OffsetComponents{}.getAffineOrigin());
  EXPECT_FALSE(OffsetComponents{}.getKnownStride(0));
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
  EXPECT_TRUE(scalar->hasAffineForm());
  EXPECT_TRUE(scalar->isUniform());
  ASSERT_TRUE(scalar->getAffineOrigin());
  EXPECT_EQ(*scalar->getAffineOrigin(), OpFoldResult(f.getArgument(0)));
  EXPECT_FALSE(scalar->getKnownStride(0));
  EXPECT_TRUE(failed(makeOpaqueOffset(f.getArgument(0), builder,
                                      ArithmeticDomain::ElementAddress)));
  EXPECT_TRUE(failed(makeOpaqueOffset(f.getArgument(4), builder)));
  EXPECT_TRUE(failed(makeOpaqueOffset(Value(), builder)));
  EXPECT_TRUE(succeeded(makeOpaqueOffset(f.getArgument(3), builder)));
  auto lanes = makeOpaqueOffset(f.getArgument(1), builder,
                                ArithmeticDomain::ElementAddress);
  ASSERT_TRUE(succeeded(lanes));
  EXPECT_EQ(lanes->completeOffset, OpFoldResult(f.getArgument(1)));
  EXPECT_EQ(lanes->domain, ArithmeticDomain::ElementAddress);
  EXPECT_FALSE(lanes->hasAffineForm());
  EXPECT_FALSE(lanes->isUniform());
  EXPECT_FALSE(lanes->getAffineOrigin());
  EXPECT_FALSE(lanes->getKnownStride(0));
  EXPECT_FALSE(lanes->getKnownStride(1));
  auto rankzero = makeOpaqueOffset(f.getArgument(2), builder);
  ASSERT_TRUE(succeeded(rankzero));
  ASSERT_TRUE(rankzero->getAffineOrigin());
  Value origin = cast<Value>(*rankzero->getAffineOrigin());
  EXPECT_TRUE(origin.getType().isInteger(64));
  EXPECT_TRUE(origin.getDefiningOp<tensor::ExtractOp>());
  EXPECT_EQ(builder.getInsertionPoint(), point);
  test::Environment inputs;
  inputs[f.getArgument(2)] = {llvm::APInt(64, -19)};
  auto evaluated = evaluate(origin, inputs);
  ASSERT_TRUE(evaluated);
  EXPECT_EQ(evaluated->front().getSExtValue(), -19);
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
