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
#include "mlir/IR/Diagnostics.h"
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
  if (!value)
    return std::nullopt;
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

TEST(PtrAnalysis, UnsupportedInputTypesEmitErrors) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @unsupported(%scalar: f32, %tensor: tensor<4xf32>,
                           %wide: i65, %wider: i128, %wide_tensor: tensor<4xi128>) {
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  SmallVector<std::string> diagnostics;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    EXPECT_EQ(diagnostic.getSeverity(), DiagnosticSeverity::Error);
    diagnostics.push_back(diagnostic.str());
    return success();
  });

  for (Value value : func.getArguments()) {
    EXPECT_TRUE(failed(analysis.analyze(value)));
    EXPECT_FALSE(analysis.findCached(value));
    EXPECT_TRUE(failed(buildOffsetInfo(value, builder)));
  }
  ASSERT_EQ(diagnostics.size(), 5u);
  EXPECT_EQ(diagnostics[0],
            "unsupported type passed to pointer analysis: 'f32'");
  EXPECT_EQ(diagnostics[1],
            "unsupported type passed to pointer analysis: 'tensor<4xf32>'");
  EXPECT_EQ(diagnostics[2],
            "unsupported type passed to pointer analysis: 'i65'");
  EXPECT_EQ(diagnostics[3],
            "unsupported type passed to pointer analysis: 'i128'");
  EXPECT_EQ(diagnostics[4],
            "unsupported type passed to pointer analysis: 'tensor<4xi128>'");

  PtrInfo wideBinding;
  wideBinding.offsets.valueType = func.getArgument(3).getType();
  wideBinding.offsets.fullOffsets = func.getArgument(3);
  EXPECT_TRUE(failed(analysis.bind(func.getArgument(3), wideBinding)));
}

TEST(PtrAnalysis, MakeRangeKeepsSignedOrigins) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @signed_ranges() {
      %negative = tt.make_range {start = -4 : i32, end = -2 : i32} : tensor<2xi32>
      %to_zero = tt.make_range {start = -4 : i32, end = 0 : i32} : tensor<4xi32>
      %cross_zero = tt.make_range {start = -4 : i32, end = 4 : i32} : tensor<8xi32>
      %positive = tt.make_range {start = 2 : i32, end = 6 : i32} : tensor<4xi32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  SmallVector<int64_t> starts{-4, -4, -4, 2};
  unsigned index = 0;
  for (auto range : func.getBody().front().getOps<triton::MakeRangeOp>()) {
    auto info = test::getOffsets(analysis.analyze(range));
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->hasStructuredForm());
    int64_t start = starts[index++];
    EXPECT_EQ(scalar(info->scalarOffset), start);
    EXPECT_EQ(scalar(info->axes[0].stride), 1);
    auto lanes = evaluate(info->fullOffsets);
    ASSERT_TRUE(lanes);
    ASSERT_EQ(lanes->size(), range.getType().getNumElements());
    for (unsigned lane = 0; lane < lanes->size(); ++lane)
      EXPECT_EQ((*lanes)[lane].getSExtValue(), start + lane);
  }
  EXPECT_EQ(index, starts.size());
  EXPECT_TRUE(succeeded(verify(*module)));
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
  auto once = test::getInfo(analysis.analyze(pointers[0]));
  auto twice = test::getInfo(analysis.analyze(pointers[2]));
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
  EXPECT_EQ(result->getShape(), ArrayRef<int64_t>({2, 4}));
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

TEST(PtrAnalysis, SupportedIntegerWidthsAndIndex) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @widths(%i1: i1, %i4: i4, %i8: i8, %i16: i16,
                     %i32: i32, %i64: i64, %index: index) {
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  for (Value value : func.getArguments()) {
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->valueType, value.getType());
    EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
    ASSERT_TRUE(info->scalarOffset);
    EXPECT_TRUE(cast<Value>(info->scalarOffset).getType().isIndex());
  }
  auto index = test::getOffsets(analysis.analyze(func.getArgument(6)));
  ASSERT_TRUE(succeeded(index));
  EXPECT_EQ(index->scalarOffset, OpFoldResult(func.getArgument(6)));
  builder.setInsertionPoint(func.getBody().front().getTerminator());
  auto materialized = materializeOffsetAt(*index, {}, builder, func.getLoc());
  ASSERT_TRUE(succeeded(materialized));
  EXPECT_EQ(*materialized, OpFoldResult(func.getArgument(6)));
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
  auto ptrInfo = test::getInfo(analysis.analyze(repeated));
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

TEST(PtrAnalysis, UniformConstantsHaveScalarOriginsWithoutHelpers) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @constants() {
      %scalar = arith.constant -7 : i8
      %rankzero = arith.constant dense<-7> : tensor<i8>
      %singleton = arith.constant dense<-7> : tensor<1x1xi8>
      %dense = arith.constant dense<-7> : tensor<4xi8>
      %splat_one = tt.splat %scalar : i8 -> tensor<1xi8>
      %splat_many = tt.splat %scalar : i8 -> tensor<4xi8>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto &block = func.getBody().front();
  SmallVector<Value> values;
  for (Operation &op : block.without_terminator())
    values.push_back(op.getResult(0));
  auto operationCount = block.getOperations().size();
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  for (Value value : values) {
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info));
    EXPECT_TRUE(info->isUniform());
    EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
    auto attr = dyn_cast<Attribute>(info->scalarOffset);
    ASSERT_TRUE(attr);
    auto origin = dyn_cast<IntegerAttr>(attr);
    ASSERT_TRUE(origin);
    EXPECT_TRUE(origin.getType().isIndex());
    EXPECT_EQ(origin.getInt(), -7);
  }
  EXPECT_EQ(block.getOperations().size(), operationCount);
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
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(y))));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, IndexConstantUses64BitAttributeStorage) {
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
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto result = test::getOffsets(analysis.analyze(x));
  ASSERT_TRUE(succeeded(result));
  auto attr = cast<IntegerAttr>(cast<Attribute>(result->scalarOffset));
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
      auto result = test::getInfo(analysis.analyze(add));
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

TEST(PtrAnalysis, IndexCastWidthAndExactValueMatrix) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  for (unsigned w : {16u, 32u, 64u})
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
          Value y = ui ? Value(b.create<arith::IndexCastUIOp>(b.getUnknownLoc(),
                                                              out, x))
                       : Value(b.create<arith::IndexCastOp>(b.getUnknownLoc(),
                                                            out, x));
          b.create<triton::ReturnOp>(b.getUnknownLoc());
          PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
          auto result = test::getInfo(analysis.analyze(y));
          ASSERT_TRUE(succeeded(result));
          for (int64_t n : {-1LL, 0LL, 1LL, 2147483647LL, 2147483648LL}) {
            test::Lanes lanes;
            for (unsigned lane = 0; lane < (rank == 1 ? 4u : 1u); ++lane)
              lanes.push_back(llvm::APInt(64, n + lane, true)
                                  .sextOrTrunc(toIndex ? w : 64));
            test::Environment env{{x, lanes}};
            auto actual = evaluate(OpFoldResult(y), env);
            ASSERT_TRUE(actual);
            auto source = evaluate(result->offsets.fullOffsets, env);
            ASSERT_TRUE(source);
            EXPECT_EQ(*source, *actual);
            // Descriptor arithmetic uses 64-bit index; exact
            // source values retain the original result width.
            b.setInsertionPoint(f.getBody().front().getTerminator());
            SmallVector<Value> indices;
            if (rank == 1)
              indices.push_back(
                  b.create<arith::ConstantIndexOp>(f.getLoc(), 0));
            auto lane =
                materializeOffsetAt(result->offsets, indices, b, f.getLoc());
            ASSERT_TRUE(succeeded(lane));
            auto materialized = evaluate(*lane, env);
            ASSERT_TRUE(materialized);
            EXPECT_EQ(materialized->front(), actual->front().sextOrTrunc(64));
          }
          EXPECT_EQ(analysis.findCached(y)->offsets.fullOffsets,
                    result->offsets.fullOffsets);
          EXPECT_TRUE(succeeded(verify(module)));
        }
}

TEST(PtrAnalysis, IndexConstantArithmeticMatrix) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
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
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  for (Value v : values) {
    auto result = test::getOffsets(analysis.analyze(v));
    ASSERT_TRUE(succeeded(result));
    ASSERT_TRUE(result->scalarOffset);
    if (auto attr = dyn_cast<Attribute>(result->scalarOffset)) {
      EXPECT_EQ(cast<IntegerAttr>(attr).getValue().getBitWidth(), 64u);
    } else {
      EXPECT_TRUE(cast<Value>(result->scalarOffset).getType().isIndex());
    }
    EXPECT_EQ(evaluate(result->scalarOffset), evaluate(OpFoldResult(v)));
    ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(v))));
    EXPECT_TRUE(succeeded(verify(*module)));
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
    auto result = test::getInfo(analysis.analyze(selects[k]));
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
  auto ptrInfo = test::getInfo(analysis.analyze(ptr));
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
  auto result = test::getInfo(analysis.analyze(pointers.back()));
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
  ASSERT_TRUE(succeeded(analysis.bind(x, PtrInfo{{}, *binding})));
  auto address = test::getOffsets(analysis.analyze(x));
  ASSERT_TRUE(succeeded(address));
  auto actual = evaluate(*address);
  ASSERT_TRUE(actual);
  ASSERT_EQ(actual->size(), lanes.size());
  for (unsigned i = 0; i < lanes.size(); ++i)
    EXPECT_EQ((*actual)[i], lanes[i].sext(64));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, SplatOriginsReuseScalarAndBindings) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @splats(%x: i8) {
      %rankzero = tt.splat %x : i8 -> tensor<i8>
      %one = tt.splat %x : i8 -> tensor<1xi8>
      %one2d = tt.splat %x : i8 -> tensor<1x1xi8>
      %many = tt.splat %x : i8 -> tensor<4xi8>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto &block = func.getBody().front();
  Value source = func.getArgument(0);
  SmallVector<Value> values;
  for (auto splat : block.getOps<triton::SplatOp>())
    values.push_back(splat.getResult());
  OpBuilder builder(&context);
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  auto bound = buildOffsetInfo(source, builder);
  ASSERT_TRUE(succeeded(bound));
  ASSERT_TRUE(succeeded(analysis.bind(source, PtrInfo{{}, *bound})));
  test::Environment env{{source, {llvm::APInt(8, -7, true)}}};
  for (Value value : values) {
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->isUniform());
    EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
    auto origin =
        cast<Value>(info->scalarOffset).getDefiningOp<arith::IndexCastOp>();
    ASSERT_TRUE(origin);
    EXPECT_EQ(origin.getIn(), source);
    auto actual = evaluate(*info, env);
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->size(),
              cast<RankedTensorType>(value.getType()).getNumElements());
    for (const auto &lane : *actual)
      EXPECT_EQ(lane.getSExtValue(), -7);
  }
  EXPECT_TRUE(block.getOps<tensor::ExtractOp>().empty());

  // A supplied origin is reused by parseSplat without another cast.
  bound->scalarOffset = builder.getIndexAttr(23);
  ASSERT_TRUE(succeeded(analysis.bind(source, PtrInfo{{}, *bound})));
  auto operationCount = block.getOperations().size();
  for (Value value : values) {
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info));
    EXPECT_TRUE(info->isUniform());
    EXPECT_EQ(info->scalarOffset, bound->scalarOffset);
  }
  EXPECT_EQ(block.getOperations().size(), operationCount);
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
        cast<Value>(source->scalarOffset).getDefiningOp<arith::IndexCastOp>();
    ASSERT_TRUE(origin);
    EXPECT_EQ(origin.getIn(), f.getArgument(1));
    auto delta = test::getOffsets(analysis.analyze(d));
    ASSERT_TRUE(succeeded(delta));
    EXPECT_EQ(delta->fullOffsets, OpFoldResult(d));
    EXPECT_EQ(delta->valueType, d.getType());
    ASSERT_TRUE(cast<Value>(delta->scalarOffset).getType().isIndex());
    auto ptrInfo = test::getInfo(analysis.analyze(next));
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
  ASSERT_TRUE(cast<Value>(state->scalarOffset).getType().isIndex());
  ASSERT_TRUE(succeeded(analysis.bind(x, PtrInfo{{}, *state})));
  auto derived = test::getOffsets(analysis.analyze(sum));
  ASSERT_TRUE(succeeded(derived));
  ASSERT_TRUE(cast<Value>(derived->scalarOffset).getType().isIndex());
  for (int64_t n :
       {-2147483648LL, -1073741824LL, -7LL, 0LL, 1073741823LL, 2147483647LL}) {
    test::Environment env{{x, {llvm::APInt(32, n, true)}}};
    auto expected = (llvm::APInt(32, n, true) * llvm::APInt(32, 2)).sext(64);
    auto complete = evaluate(*derived, env);
    ASSERT_TRUE(complete);
    EXPECT_EQ(complete->front(), expected);
    // The exact source retains wrapping; index descriptors assume that the
    // source addition does not overflow, including at its signed boundaries.
    if (n >= -1073741824LL && n <= 1073741823LL) {
      auto uniform = evaluate(derived->scalarOffset, env);
      ASSERT_TRUE(uniform);
      EXPECT_EQ(uniform->front(), expected);
    }
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, DynamicTensorIndexPreserves64BitValues) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @index(%x: tensor<4xindex>) { tt.return } }
    )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
  auto address = test::getOffsets(analysis.analyze(f.getArgument(0)));
  ASSERT_TRUE(succeeded(address));
  test::Lanes lanes{llvm::APInt::getSignedMinValue(64),
                    llvm::APInt(64, -1, true), llvm::APInt(64, 0),
                    llvm::APInt::getSignedMaxValue(64)};
  test::Environment env{{f.getArgument(0), lanes}};
  auto actual = evaluate(*address, env);
  ASSERT_TRUE(actual);
  ASSERT_EQ(actual->size(), lanes.size());
  for (unsigned i = 0; i < lanes.size(); ++i)
    EXPECT_EQ((*actual)[i], lanes[i]);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysis, TensorIndexConstantsAndBindingsPreserve64BitValues) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
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
    for (int64_t n : {-2147483648LL, 2147483647LL, 2147483648LL, 4294967295LL})
      stored.push_back(llvm::APInt(64, splat ? 4294967295LL : n, true));
    auto attr = DenseIntElementsAttr::get(type, stored);
    Value constant = b.create<arith::ConstantOp>(f.getLoc(), attr);
    b.create<triton::ReturnOp>(f.getLoc());
    PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
    auto bound = test::getOffsets(analysis.analyze(f.getArgument(0)));
    ASSERT_TRUE(succeeded(bound));
    bound->fullOffsets = attr;
    ASSERT_TRUE(
        succeeded(analysis.bind(f.getArgument(0), PtrInfo{{}, *bound})));
    for (Value value : {constant, Value(f.getArgument(0))}) {
      auto offsets = test::getOffsets(analysis.analyze(value));
      ASSERT_TRUE(succeeded(offsets));
      auto actual = evaluate(*offsets);
      ASSERT_TRUE(actual);
      auto source = evaluate(OpFoldResult(attr));
      ASSERT_TRUE(source);
      ASSERT_EQ(actual->size(), stored.size());
      for (unsigned lane = 0; lane < stored.size(); ++lane) {
        llvm::APInt expected = stored[lane].sextOrTrunc(64);
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
    PtrAnalysis<CommonPtrRules> analysis(b, CommonPtrRules{});
    auto source = test::getOffsets(analysis.analyze(f.getArgument(0)));
    ASSERT_TRUE(succeeded(source));
    ASSERT_TRUE(
        succeeded(analysis.bind(f.getArgument(0), PtrInfo{{}, *source})));
    auto offsets = test::getOffsets(analysis.analyze(f.getArgument(0)));
    ASSERT_TRUE(succeeded(offsets));
    EXPECT_EQ(offsets->valueType,
              RankedTensorType::get(shape, b.getIndexType(), encoding));
    test::Lanes lanes(rankZero ? 1 : 4, llvm::APInt(64, -1, true));
    test::Environment env{{f.getArgument(0), lanes}};
    auto actual = evaluate(*offsets, env);
    ASSERT_TRUE(actual);
    EXPECT_EQ(*actual,
              test::Lanes(rankZero ? 1 : 4, llvm::APInt(64, -1, true)));
    if (rankZero) {
      ASSERT_TRUE(cast<Value>(source->scalarOffset).getType().isIndex());
      ASSERT_TRUE(cast<Value>(offsets->scalarOffset).getType().isIndex());
      EXPECT_EQ(evaluate(offsets->scalarOffset, env), actual);
    }
    EXPECT_TRUE(succeeded(verify(module)));
  }
}

TEST(PtrAnalysis, OffsetInfoConstructionLeavesOriginsToAnalysis) {
  EXPECT_FALSE(OffsetInfo{}.hasStructuredForm());
  EXPECT_FALSE(OffsetInfo{}.isUniform());
  EXPECT_FALSE(OffsetInfo{}.scalarOffset);
  EXPECT_TRUE(OffsetInfo{}.axes.empty());
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @offset_info(%scalar: i32, %lanes: tensor<4xi64>,
                                 %rankzero: tensor<i64>, %idx: index,
                                 %float: f32, %singleton: tensor<1xi32>) {
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  builder.setInsertionPoint(f.getBody().front().getTerminator());
  auto point = builder.getInsertionPoint();
  auto operationCount = f.getBody().front().getOperations().size();
  for (unsigned i : {0u, 1u, 2u, 3u, 5u}) {
    Value value = f.getArgument(i);
    auto info = buildOffsetInfo(value, builder);
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->valueType, value.getType());
    EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
    EXPECT_FALSE(info->scalarOffset);
    EXPECT_FALSE(info->hasStructuredForm());
    for (const AxisInfo &axis : info->axes)
      EXPECT_EQ(axis.kind, AxisKind::Unknown);
  }
  EXPECT_TRUE(failed(buildOffsetInfo(f.getArgument(4), builder)));
  EXPECT_TRUE(failed(buildOffsetInfo(Value(), builder)));
  EXPECT_EQ(f.getBody().front().getOperations().size(), operationCount);
  EXPECT_EQ(builder.getInsertionPoint(), point);

  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  for (unsigned i : {0u, 2u, 3u, 5u}) {
    Value value = f.getArgument(i);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->offsets.isUniform());
    EXPECT_TRUE(cast<Value>(info->offsets.scalarOffset).getType().isIndex());
    unsigned width = (i == 0 || i == 5) ? 32 : 64;
    test::Environment inputs{{value, {llvm::APInt(width, -19, true)}}};
    auto evaluated = evaluate(info->offsets.scalarOffset, inputs);
    ASSERT_TRUE(evaluated);
    ASSERT_EQ(evaluated->size(), 1u);
    EXPECT_EQ(evaluated->front().getSExtValue(), -19);
  }
  Value origin =
      cast<Value>(analysis.findCached(f.getArgument(2))->offsets.scalarOffset);
  auto castOp = origin.getDefiningOp<arith::IndexCastOp>();
  ASSERT_TRUE(castOp);
  EXPECT_TRUE(castOp.getIn().getDefiningOp<tensor::ExtractOp>());
  EXPECT_EQ(builder.getInsertionPoint(), point);
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
