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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;
namespace {
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect, scf::SCFDialect>();
}
Value tagged(Operation *scope, StringRef tag) {
  Value result;
  scope->walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<StringAttr>("tag"))
      if (attr.getValue() == tag)
        result = op->getResult(0);
  });
  return result;
}
unsigned count(Operation *scope) {
  unsigned n = 0;
  scope->walk([&](Operation *) { ++n; });
  return n;
}
void expectLanes(const OffsetInfo &info, ArrayRef<int64_t> expected,
                 OpBuilder &b, Location loc, const test::Environment &env) {
  ASSERT_EQ(info.getShape().size(), 2u);
  auto point = b.saveInsertionPoint();
  for (unsigned i = 0; i < expected.size(); ++i) {
    SmallVector<Value> indices{
        b.create<arith::ConstantIndexOp>(loc, i / info.getShape()[1]),
        b.create<arith::ConstantIndexOp>(loc, i % info.getShape()[1])};
    auto scalar = materializeOffsetAt(info, indices, b, loc);
    ASSERT_TRUE(succeeded(scalar));
    auto actual = test::evaluate(*scalar, env);
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->front().getSExtValue(), expected[i]);
  }
  EXPECT_EQ(b.getInsertionPoint(), point.getPoint());
}
TEST(PtrAnalysisAddress, IndependentExtensionsSurviveShapeAndSelect) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @chain(%base:!tt.ptr<i32,3>, %x:tensor<4xi32>, %y:tensor<4xi32>, %c:tensor<2x2xi1>) {
      %p=tt.splat %base:!tt.ptr<i32,3> -> tensor<4x!tt.ptr<i32,3>>
      %first=tt.addptr %p,%x:tensor<4x!tt.ptr<i32,3>>,tensor<4xi32>
      %twice=tt.addptr %first,%y:tensor<4x!tt.ptr<i32,3>>,tensor<4xi32>
      %sum=arith.addi %x,%y:tensor<4xi32>
      %once=tt.addptr %p,%sum:tensor<4x!tt.ptr<i32,3>>,tensor<4xi32>
      %a="tt.reshape"(%twice) {tag="twice"}:(tensor<4x!tt.ptr<i32,3>>) -> tensor<2x2x!tt.ptr<i32,3>>
      %b="tt.reshape"(%once) {tag="once"}:(tensor<4x!tt.ptr<i32,3>>) -> tensor<2x2x!tt.ptr<i32,3>>
      %s=arith.select %c,%a,%b {tag="selected"}:tensor<2x2xi1>,tensor<2x2x!tt.ptr<i32,3>>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  test::Environment env{
      {f.getArgument(1),
       {llvm::APInt(32, INT32_MAX), llvm::APInt(32, -5, true),
        llvm::APInt(32, INT32_MAX), llvm::APInt(32, -5, true)}},
      {f.getArgument(2), test::Lanes(4, llvm::APInt(32, 1))},
      {f.getArgument(3),
       {llvm::APInt(1, 1), llvm::APInt(1, 1), llvm::APInt(1, 0),
        llvm::APInt(1, 0)}}};
  for (auto name : {"twice", "once", "selected"}) {
    auto info = test::getInfo(analysis.analyze(tagged(f, name)));
    ASSERT_TRUE(succeeded(info));
    EXPECT_TRUE(info->offsets.fullOffsets);
    EXPECT_FALSE(info->offsets.hasStructuredDecomposition());
    EXPECT_EQ(info->getAddressSpace(), 3u);
    unsigned before = count(*module);
    ASSERT_TRUE(succeeded(analysis.analyze(tagged(f, name))));
    EXPECT_EQ(count(*module), before);
    SmallVector<int64_t> expected =
        StringRef(name) == "twice"
            ? SmallVector<int64_t>{2147483648LL, -4, 2147483648LL, -4}
        : StringRef(name) == "once"
            ? SmallVector<int64_t>{-2147483648LL, -4, -2147483648LL, -4}
            : SmallVector<int64_t>{2147483648LL, -4, -2147483648LL, -4};
    expectLanes(info->offsets, expected, b, f.getLoc(), env);
  }
  module->walk([&](Operation *op) {
    // Triton tensor operations use integer storage, not index elements.
    for (Type type : op->getResultTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        EXPECT_FALSE(tensor.getElementType().isIndex());
  });
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisAddress, SelectNarrowNegativeAndFullOffsetBindings) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @select(%base:!tt.ptr<i32>,%x:i8,%y:tensor<4xi8>,%c:tensor<4xi1>,%wide:tensor<4xi64>) {
      %p=tt.splat %base:!tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %xx=tt.splat %x:i8 -> tensor<4xi8>
      %a=tt.addptr %p,%xx {tag="left"}:tensor<4x!tt.ptr<i32>>,tensor<4xi8>
      %b=tt.addptr %p,%y:tensor<4x!tt.ptr<i32>>,tensor<4xi8>
      %s=arith.select %c,%a,%b {tag="selected"}:tensor<4xi1>,tensor<4x!tt.ptr<i32>>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  test::Environment env{
      {f.getArgument(1), {llvm::APInt(8, -128, true)}},
      {f.getArgument(2), test::Lanes(4, llvm::APInt(8, -2, true))},
      {f.getArgument(3),
       {llvm::APInt(1, 1), llvm::APInt(1, 0), llvm::APInt(1, 1),
        llvm::APInt(1, 0)}},
      {f.getArgument(4), test::Lanes(4, llvm::APInt(64, 300))}};
  auto check = [&](int64_t left) {
    auto info = test::getInfo(analysis.analyze(tagged(f, "selected")));
    ASSERT_TRUE(succeeded(info));
    for (unsigned i = 0; i < 4; ++i) {
      Value index = b.create<arith::ConstantIndexOp>(f.getLoc(), i);
      auto lane = materializeOffsetAt(info->offsets, {index}, b, f.getLoc());
      ASSERT_TRUE(succeeded(lane));
      auto actual = test::evaluate(*lane, env);
      ASSERT_TRUE(actual);
      EXPECT_EQ(actual->front().getSExtValue(), i % 2 ? -2 : left);
    }
  };
  check(-128);
  auto left = test::getInfo(analysis.analyze(tagged(f, "left")));
  ASSERT_TRUE(succeeded(left));
  left->offsets.fullOffsets = f.getArgument(4);
  left->offsets.scalarOffset = {};
  left->offsets.axes[0] = AxisInfo(AxisKind::Unknown, b.getIndexAttr(0));
  ASSERT_TRUE(succeeded(analysis.reset({{tagged(f, "left"), *left}})));
  check(300);
  // Neither a linear nor a uniform description replaces complete offsets.
  left->offsets.fullOffsets = {};
  left->offsets.scalarOffset = b.getIndexAttr(17);
  left->offsets.axes[0] = AxisInfo(AxisKind::Structured, b.getIndexAttr(2));
  EXPECT_TRUE(failed(analysis.reset({{tagged(f, "left"), *left}})));
  left->offsets.axes[0] = AxisInfo(AxisKind::Invariant, b.getIndexAttr(0));
  EXPECT_TRUE(failed(analysis.reset({{tagged(f, "left"), *left}})));
  EXPECT_TRUE(failed(triton::ptr::detail::materializeFullOffset(
      left->offsets, f.getArgument(4).getType(), b, f.getLoc())));
  check(300);

  // A complete constant tensor remains a valid uniform binding.
  left->offsets.fullOffsets = DenseIntElementsAttr::get(
      cast<RankedTensorType>(f.getArgument(4).getType()), int64_t(17));
  ASSERT_TRUE(succeeded(analysis.reset({{tagged(f, "left"), *left}})));
  check(17);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisAddress, EmptyNestedInsertionChecksDefinitionsAndIndices) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @nested(%early:index,%tensor:tensor<4xi32>,%index:index) {
      %region=scf.execute_region -> index { scf.yield %early:index }
      %late=arith.addi %early,%early:index
      %lateindex=arith.addi %index,%index:index
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto region = *f.getBody().front().getOps<scf::ExecuteRegionOp>().begin();
  SmallVector<Value> late;
  f.walk([&](arith::AddIOp op) { late.push_back(op); });
  auto &inner = region.getRegion().front();
  inner.getTerminator()->erase();
  OpBuilder b(&context);
  b.setInsertionPointToEnd(&inner);
  auto ip = b.saveInsertionPoint();
  OffsetInfo scalar;
  scalar.valueType = b.getIndexType();
  for (Value invalid : {late[0], region.getResult(0)}) {
    scalar.fullOffsets = invalid;
    unsigned n = count(*module);
    EXPECT_TRUE(failed(materializeOffsetAt(scalar, {}, b, f.getLoc())));
    EXPECT_EQ(count(*module), n);
    EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  }
  OffsetInfo tensor;
  tensor.valueType = RankedTensorType::get({4}, b.getIndexType());
  tensor.fullOffsets = f.getArgument(1);
  EXPECT_TRUE(failed(materializeOffsetAt(tensor, {late[1]}, b, f.getLoc())));
  EXPECT_TRUE(failed(materializeOffsetAt(tensor, {}, b, f.getLoc())));
  EXPECT_TRUE(
      failed(materializeOffsetAt(tensor, {f.getArgument(1)}, b, f.getLoc())));
  scalar.fullOffsets = f.getArgument(0);
  scalar.scalarOffset = late[0]; // Not used when fullOffsets is available.
  tensor.scalarOffset = late[0];
  tensor.axes = {AxisInfo(AxisKind::Structured, late[0])};
  auto good = materializeOffsetAt(scalar, {}, b, f.getLoc());
  ASSERT_TRUE(succeeded(good));
  auto lane = materializeOffsetAt(tensor, {f.getArgument(2)}, b, f.getLoc());
  ASSERT_TRUE(succeeded(lane));
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  // A nonempty block's end has the same requirements, including origin fields.
  scalar.fullOffsets = {};
  scalar.scalarOffset = late[0];
  EXPECT_TRUE(failed(materializeOffsetAt(scalar, {}, b, f.getLoc())));
  scalar.scalarOffset = f.getArgument(0);
  EXPECT_TRUE(succeeded(materializeOffsetAt(scalar, {}, b, f.getLoc())));
  b.create<scf::YieldOp>(f.getLoc(), cast<Value>(*good));
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisAddress,
     BindingValidatesFullOffsetsBaseTypesAndBorrowedResults) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @bind(%base:!tt.ptr<i32,3>,%wrongspace:!tt.ptr<i32>,%wrongtype:!tt.ptr<i8,3>,%x:tensor<4xi32>,%other:i32) {
      %p=tt.splat %base:!tt.ptr<i32,3> -> tensor<4x!tt.ptr<i32,3>>
      %a=tt.addptr %p,%x:tensor<4x!tt.ptr<i32,3>>,tensor<4xi32>
      %n=arith.constant 16:i64
      %zero=arith.constant 0:i32
      %block=tt.make_tensor_ptr %base,[%n],[%n],[%zero] {order=array<i32:0>}:!tt.ptr<tensor<4xi32>,3>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  auto add = *f.getBody().front().getOps<triton::AddPtrOp>().begin();
  auto block = *f.getBody().front().getOps<triton::MakeTensorPtrOp>().begin();
  PtrAnalysis analysis(b);
  for (Value value : {Value(add.getResult()), Value(block.getResult())}) {
    auto original = test::getInfo(analysis.analyze(value));
    ASSERT_TRUE(succeeded(original));
    for (Value wrong : {f.getArgument(1), f.getArgument(2)}) {
      auto invalid = *original;
      invalid.base = wrong;
      EXPECT_TRUE(failed(analysis.setInfo(value, invalid)));
      ASSERT_TRUE(analysis.findCached(value));
      EXPECT_EQ(analysis.findCached(value)->base, f.getArgument(0));
    }
  }
  auto original = test::getInfo(analysis.analyze(add));
  ASSERT_TRUE(succeeded(original));
  auto invalid = *original;
  invalid.offsets.fullOffsets = f.getArgument(0);
  EXPECT_TRUE(failed(analysis.setInfo(add, invalid)));
  invalid.offsets.fullOffsets = {};
  EXPECT_TRUE(
      failed(analysis.setInfo(add, invalid))); // partial proof is not exact
  auto scalar = analysis.analyze(f.getArgument(0));
  ASSERT_TRUE(succeeded(scalar));
  ASSERT_TRUE(scalar->offsets.hasStructuredDecomposition());
  scalar->offsets.fullOffsets = {};
  EXPECT_TRUE(failed(analysis.setInfo(f.getArgument(0), *scalar)));
  ASSERT_TRUE(succeeded(analysis.setInfo(add, *analysis.findCached(add))));
  ASSERT_TRUE(analysis.findCached(add));
  PtrInfo other{{}, *buildOffsetInfo(f.getArgument(4), b)};
  ASSERT_TRUE(succeeded(analysis.setInfo(f.getArgument(4), other)));
  EXPECT_TRUE(
      analysis.findCached(add)); // another boundary, not a computed entry
  EXPECT_TRUE(analysis.findCached(block));
  ASSERT_TRUE(succeeded(analysis.reset()));
  EXPECT_FALSE(analysis.findCached(add));
  EXPECT_FALSE(analysis.findCached(f.getArgument(4)));

  // Block pointers still bind and reconstruct lanes from their decomposition.
  auto blockInfo = analysis.analyze(block);
  ASSERT_TRUE(succeeded(blockInfo));
  EXPECT_FALSE(blockInfo->offsets.fullOffsets);
  ASSERT_TRUE(blockInfo->offsets.hasStructuredDecomposition());
  ASSERT_TRUE(succeeded(analysis.setInfo(block, *blockInfo)));
  int64_t expectedOffsets[] = {0, 16, 32, 48};
  for (auto [i, expected] : llvm::enumerate(expectedOffsets)) {
    Value index = b.create<arith::ConstantIndexOp>(f.getLoc(), i);
    auto offset =
        materializeOffsetAt(blockInfo->offsets, {index}, b, f.getLoc());
    ASSERT_TRUE(succeeded(offset));
    auto actual = test::evaluate(*offset);
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->front().getSExtValue(), expected);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisAddress, ScalarAddPtrSharesOneSumAfterWideningEachIncrement) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @scalar(%base:!tt.ptr<i32>,%x:i8,%y:i8) {
      %sum=arith.addi %x,%y:i8
      %once=tt.addptr %base,%sum:!tt.ptr<i32>,i8
      %first=tt.addptr %base,%x:!tt.ptr<i32>,i8
      %twice=tt.addptr %first,%y:!tt.ptr<i32>,i8
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<triton::AddPtrOp> adds;
  f.walk([&](triton::AddPtrOp op) { adds.push_back(op); });
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  PtrAnalysis analysis(b);
  auto once = test::getInfo(analysis.analyze(adds[0]));
  auto twice = test::getInfo(analysis.analyze(adds[2]));
  ASSERT_TRUE(succeeded(once));
  ASSERT_TRUE(succeeded(twice));
  EXPECT_EQ(twice->offsets.fullOffsets, twice->offsets.scalarOffset);
  unsigned sums = 0;
  f.walk([&](arith::AddIOp op) { sums += op.getType().isIndex(); });
  EXPECT_EQ(sums, 1u);
  for (auto [x, y] :
       {std::pair<int64_t, int64_t>{127, 1}, {-128, -1}, {17, -5}}) {
    test::Environment env{{f.getArgument(1), {llvm::APInt(8, x, true)}},
                          {f.getArgument(2), {llvm::APInt(8, y, true)}}};
    auto narrow = test::evaluate(once->offsets, env);
    auto wide = test::evaluate(twice->offsets, env);
    ASSERT_TRUE(narrow);
    ASSERT_TRUE(wide);
    EXPECT_EQ(narrow->front(), llvm::APInt(64, x + y, true).trunc(8).sext(64));
    EXPECT_EQ(wide->front(), llvm::APInt(64, x + y, true));
  }
  EXPECT_EQ(b.getInsertionBlock(), ip.getBlock());
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisAddress, ScalarExactBindingsFeedAddPtrAndBlockWithoutProof) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @scalar(%base:!tt.ptr<i32>,%boundary:!tt.ptr<i32>,%dynamic:i64,%x:i32,%y:i32) {
      %one=arith.constant 1:i8
      %next=tt.addptr %boundary,%one:!tt.ptr<i32>,i8
      %shape=arith.constant 16:i64
      %stride=arith.constant 2:i64
      %coord=arith.constant 3:i32
      %block=tt.make_tensor_ptr %boundary,[%shape],[%stride],[%coord] {order=array<i32:0>}:!tt.ptr<tensor<4xi32>>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto add = *f.getBody().front().getOps<triton::AddPtrOp>().begin();
  auto block = *f.getBody().front().getOps<triton::MakeTensorPtrOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  test::Environment env{{f.getArgument(2), {llvm::APInt(64, 2147483648LL)}},
                        {f.getArgument(3), {llvm::APInt(32, INT32_MAX)}},
                        {f.getArgument(4), {llvm::APInt(32, 1)}}};
  {
    PtrAnalysis analysis(b);
    PtrInfo bound;
    bound.base = f.getArgument(0);
    bound.offsets.valueType = b.getIndexType();
    bound.offsets.fullOffsets = f.getArgument(2);
    ASSERT_FALSE(bound.offsets.scalarOffset);
    ASSERT_TRUE(succeeded(analysis.setInfo(f.getArgument(1), bound)));
    auto next = test::getInfo(analysis.analyze(add));
    ASSERT_TRUE(succeeded(next));
    auto value = test::evaluate(next->offsets, env);
    ASSERT_TRUE(value);
    EXPECT_EQ(value->front().getSExtValue(), 2147483649LL);
    auto made = test::getInfo(analysis.analyze(block));
    ASSERT_TRUE(succeeded(made));
    EXPECT_EQ(made->base, f.getArgument(0));
    Value index = b.create<arith::ConstantIndexOp>(f.getLoc(), 2);
    auto lane = materializeOffsetAt(made->offsets, {index}, b, f.getLoc());
    ASSERT_TRUE(succeeded(lane));
    auto actual = test::evaluate(*lane, env);
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->front().getSExtValue(), 2147483658LL);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisAddress, BlockAdvanceReplacesPreviouslyCompleteOffsets) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @block(%base:!tt.ptr<i32>) {
      %shape=arith.constant 16:i64
      %stride=arith.constant 1:i64
      %zero=arith.constant 0:i32
      %four=arith.constant 4:i32
      %ptr=tt.make_tensor_ptr %base,[%shape],[%stride],[%zero] {order=array<i32:0>}:!tt.ptr<tensor<4xi32>>
      %next=tt.advance %ptr,[%four]:!tt.ptr<tensor<4xi32>>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto block = *f.getBody().front().getOps<triton::MakeTensorPtrOp>().begin();
  auto advance = *f.getBody().front().getOps<triton::AdvanceOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  auto bound = test::getInfo(analysis.analyze(block));
  ASSERT_TRUE(succeeded(bound));
  bound->offsets.fullOffsets =
      DenseIntElementsAttr::get(RankedTensorType::get({4}, b.getI32Type()),
                                ArrayRef<int32_t>{0, 1, 2, 3});
  ASSERT_TRUE(succeeded(analysis.setInfo(block, *bound)));
  auto next = test::getInfo(analysis.analyze(advance));
  ASSERT_TRUE(succeeded(next));
  EXPECT_FALSE(next->offsets.fullOffsets);
  for (unsigned i = 0; i < 4; ++i) {
    Value index = b.create<arith::ConstantIndexOp>(f.getLoc(), i);
    auto lane = materializeOffsetAt(next->offsets, {index}, b, f.getLoc());
    ASSERT_TRUE(succeeded(lane));
    auto value = test::evaluate(*lane);
    ASSERT_TRUE(value);
    EXPECT_EQ(value->front().getSExtValue(), i + 4);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisAddress, OpaqueTensorBasePreservesLanesAndOffsetUnits) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @opaque(%raw:tensor<4x!tt.ptr<i32,3>>) {
      %base="tt.reshape"(%raw) {tag="base"}:(tensor<4x!tt.ptr<i32,3>>) -> tensor<2x2x!tt.ptr<i32,3>>
      %r=tt.make_range {start=0:i32,end=4:i32}:tensor<4xi32>
      %offset="tt.reshape"(%r):(tensor<4xi32>) -> tensor<2x2xi32>
      %shift=tt.addptr %base,%offset {tag="shift"}:tensor<2x2x!tt.ptr<i32,3>>,tensor<2x2xi32>
      %slice=tensor.extract_slice %shift[0,1][2,1][1,1]:tensor<2x2x!tt.ptr<i32,3>> to tensor<2x1x!tt.ptr<i32,3>>
      %cast=tt.bitcast %slice {tag="cast"}:tensor<2x1x!tt.ptr<i32,3>> -> tensor<2x1x!tt.ptr<i16,3>>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  auto base = analysis.analyze(tagged(f, "base"));
  ASSERT_TRUE(succeeded(base));
  EXPECT_EQ(base->base, tagged(f, "base"));
  EXPECT_FALSE(base->hasScalarBase());
  auto shift = analysis.analyze(tagged(f, "shift"));
  ASSERT_TRUE(succeeded(shift));
  EXPECT_EQ(shift->base, base->base);
  expectLanes(shift->offsets, {0, 1, 2, 3}, b, f.getLoc(), test::Environment());

  auto cast = analysis.analyze(tagged(f, "cast"));
  ASSERT_TRUE(succeeded(cast));
  EXPECT_FALSE(cast->hasScalarBase());
  EXPECT_EQ(cast->getElementType(), b.getI16Type());
  EXPECT_EQ(cast->getAddressSpace(), 3u);
  auto baseCast = cast->base.getDefiningOp<triton::BitcastOp>();
  ASSERT_TRUE(baseCast);
  auto baseSlice = baseCast.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
  ASSERT_TRUE(baseSlice);
  EXPECT_EQ(baseSlice.getSource(), base->base);
  // The second column has offsets 1 and 3 in i32 elements: 2 and 6 in i16.
  expectLanes(cast->offsets, {2, 6}, b, f.getLoc(), test::Environment());
  unsigned before = count(*module);
  ASSERT_TRUE(succeeded(analysis.analyze(tagged(f, "cast"))));
  EXPECT_EQ(count(*module), before);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisAddress, DynamicOpaqueTensorHasAnExactZeroDisplacement) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @dynamic(%raw:tensor<?x!tt.ptr<i32>>) { tt.return }
    })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  auto info = analysis.analyze(f.getArgument(0));
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->base, f.getArgument(0));
  EXPECT_FALSE(info->hasScalarBase());
  EXPECT_TRUE(info->offsets.isUniform());
  EXPECT_TRUE(succeeded(
      mlir::triton::ptr::detail::validateInfo(f.getArgument(0), *info)));
  Value full = dyn_cast<Value>(info->offsets.fullOffsets);
  ASSERT_TRUE(full);
  auto generated = full.getDefiningOp<tensor::GenerateOp>();
  ASSERT_TRUE(generated);
  EXPECT_EQ(generated.getType(),
            RankedTensorType::get({ShapedType::kDynamic}, b.getI64Type()));
  auto zero = test::evaluate(
      generated.getBody().front().getTerminator()->getOperand(0));
  ASSERT_TRUE(zero);
  EXPECT_TRUE(zero->front().isZero());
  unsigned before = count(*module);
  ASSERT_TRUE(succeeded(analysis.analyze(f.getArgument(0))));
  EXPECT_EQ(count(*module), before);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisAddress, DenseMapGrowthAndCSEAfterSession) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @cache(%base:!tt.ptr<i32>,%x:i32) {
      %a=tt.addptr %base,%x:!tt.ptr<i32>,i32
      %b=tt.addptr %base,%x:!tt.ptr<i32>,i32
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<Value> ptrs;
  f.walk([&](triton::AddPtrOp op) { ptrs.push_back(op); });
  OpBuilder b(f.getBody().front().getTerminator());
  SmallVector<OpFoldResult> kept;
  {
    PtrAnalysis analysis(b);
    auto first = test::getInfo(analysis.analyze(ptrs[0]));
    ASSERT_TRUE(succeeded(first));
    auto copy = *analysis.findCached(ptrs[0]);
    for (unsigned i = 0; i < 2048; ++i) {
      Value v = b.create<arith::ConstantIntOp>(f.getLoc(), i, 32);
      ASSERT_TRUE(succeeded(analysis.analyze(v)));
    }
    ASSERT_TRUE(analysis.findCached(ptrs[0]));
    EXPECT_EQ(copy.offsets.fullOffsets,
              analysis.findCached(ptrs[0])->offsets.fullOffsets);
    unsigned n = count(*module);
    EXPECT_TRUE(succeeded(analysis.analyze(ptrs[0])));
    EXPECT_EQ(count(*module), n);
    for (auto p : ptrs) {
      auto result = test::getInfo(analysis.analyze(p));
      ASSERT_TRUE(succeeded(result));
      kept.push_back(result->offsets.scalarOffset);
    }
  }
  Value sum = b.create<arith::AddIOp>(f.getLoc(), cast<Value>(kept[0]),
                                      cast<Value>(kept[1]));
  Value narrow = b.create<arith::IndexCastOp>(f.getLoc(), b.getI32Type(), sum);
  b.create<triton::StoreOp>(f.getLoc(), f.getArgument(0), narrow, Value(),
                            triton::CacheModifier::NONE,
                            triton::EvictionPolicy::NORMAL);
  unsigned before = 0;
  f.walk([&](arith::IndexCastOp op) { before += op.getType().isIndex(); });
  EXPECT_GE(before, 2u);
  PassManager manager(&context);
  manager.addPass(createCSEPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  unsigned after = 0;
  f.walk([&](arith::IndexCastOp op) { after += op.getType().isIndex(); });
  EXPECT_EQ(after, 1u);
  EXPECT_TRUE(succeeded(verify(*module)));
}
} // namespace
