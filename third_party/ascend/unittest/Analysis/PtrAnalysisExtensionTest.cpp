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
#include "TritonMemoryAccess/PtrAnalysisRules.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;
using test::buildAddressView;
using test::eraseDeadPtrArithmetic;
namespace {
namespace pd = mlir::triton::ptr::detail;
using test::evaluate;
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, triton::TritonDialect,
                      tensor::TensorDialect, scf::SCFDialect>();
}
unsigned operations(Operation *scope) {
  unsigned n = 0;
  scope->walk([&](Operation *) { ++n; });
  return n;
}

// Neither a basic view nor a pointer/integer variant. This rule establishes
// that Driver only requires its chosen Result and the lifecycle contract.
struct TokenInfo {
  Value source, helper;
};
struct TokenRules {
  using Result = TokenInfo;
  llvm::DenseMap<Value, unsigned> *counts = nullptr;
  template <typename Context>
  FailureOr<Result> parse(Value value, Context &context) const {
    if (counts)
      ++(*counts)[value];
    if (auto *op = value.getDefiningOp())
      if (op->hasAttr("cycle"))
        return context.analyze(value);
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
      auto lhsInfo = context.analyze(add.getLhs()),
           rhsInfo = context.analyze(add.getRhs());
      if (failed(lhsInfo) || failed(rhsInfo))
        return failure();
      context.getBuilder().setInsertionPointAfterValue(value);
      Value helper = context.getBuilder().template create<arith::AddIOp>(
          value.getLoc(), lhsInfo->helper, rhsInfo->helper);
      return Result{value, helper};
    }
    return Result{value, value};
  }
  LogicalResult validateBinding(Value value, const Result &info) const {
    return success(info.source && info.helper &&
                   info.source.getType() == value.getType() &&
                   info.helper.getType() == value.getType());
  }
  void clearCaches() const {}
};
TEST(PtrAnalysisExtension, DriverAcceptsAnIndependentResultAndSharesHelpers) {
  MLIRContext context;
  load(context);
  auto module =
      parseSourceString<ModuleOp>(R"mlir(module { tt.func @test(%x:i32,%y:i32) {
    %zero=arith.constant 0:i32
    %sum=arith.addi %x,%y:i32
    %twice=arith.addi %sum,%sum:i32
    %alias=arith.addi %sum,%zero:i32
    tt.return
  } })mlir",
                                  &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<arith::AddIOp> adds;
  f.walk([&](arith::AddIOp op) { adds.push_back(op); });
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  llvm::DenseMap<Value, unsigned> counts;
  PtrAnalysis<TokenRules> driver(b, TokenRules{&counts});
  auto info = driver.analyze(adds[1]);
  ASSERT_TRUE(succeeded(info));
  unsigned n = operations(*module);
  const auto *cached = driver.findCached(adds[1]);
  for (unsigned i = 0; i < 3; ++i)
    ASSERT_TRUE(succeeded(driver.analyze(adds[1])));
  for (const auto &entry : counts)
    EXPECT_EQ(entry.second, 1u);
  EXPECT_EQ(operations(*module), n);
  ASSERT_TRUE(succeeded(driver.bind(adds[2], *driver.findCached(adds[0]))));
  EXPECT_FALSE(driver.findCached(adds[1]));
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), TokenInfo{})));
  EXPECT_FALSE(driver.findCached(adds[1]));
  ASSERT_TRUE(succeeded(driver.bind(
      f.getArgument(0), TokenInfo{f.getArgument(1), f.getArgument(1)})));
  EXPECT_FALSE(driver.findCached(adds[1]));
  ASSERT_TRUE(succeeded(driver.analyze(adds[1])));
  driver.clear();
  EXPECT_FALSE(driver.findCached(f.getArgument(0)));
  EXPECT_EQ(b.getInsertionBlock(), ip.getBlock());
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisExtension, RecursiveCurrentNodeIsAnError) {
  MLIRContext context;
  load(context);
  auto module =
      parseSourceString<ModuleOp>(R"mlir(module {tt.func @test(%x:i32) {
    %sum=arith.addi %x,%x {cycle}:i32
    tt.return
  }})mlir",
                                  &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto op = *f.getBody().front().getOps<arith::AddIOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  PtrAnalysis<TokenRules> driver(b, TokenRules{});
  EXPECT_TRUE(failed(driver.analyze(op)));
  EXPECT_FALSE(driver.findCached(op));
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
}

TEST(PtrAnalysisExtension, SubAxesAndOriginPropagateAndRecombine) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module {tt.func @split(%base:!tt.ptr<i32>,%scale:i32) {
    %k=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
    %four=arith.constant dense<4>:tensor<16xi32>
    %five=arith.constant dense<5>:tensor<16xi32>
    %two=arith.constant dense<2>:tensor<16xi32>
    %r=arith.remui %k,%four:tensor<16xi32>
    %q=arith.divui %k,%four:tensor<16xi32>
    %shift=arith.addi %r,%five:tensor<16xi32>
    %scaled=arith.muli %shift,%two:tensor<16xi32>
    %row=tt.expand_dims %scaled {axis=0:i32}:tensor<16xi32> -> tensor<1x16xi32>
    %rows=tt.broadcast %row:tensor<1x16xi32> -> tensor<2x16xi32>
    %q4=arith.muli %q,%four:tensor<16xi32>
    %linear=arith.addi %q4,%r:tensor<16xi32>
    %limit=arith.constant 255:i32
    %bounded=arith.andi %scale,%limit:i32
    %factor=tt.splat %bounded:i32 -> tensor<16xi32>
    %dynamic=arith.muli %scaled,%factor:tensor<16xi32>
    %p=tt.splat %base:!tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %pointer=tt.addptr %p,%scaled:tensor<16x!tt.ptr<i32>>,tensor<16xi32>
    tt.return
  }})mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto rem = *f.getBody().front().getOps<arith::RemUIOp>().begin();
  auto rows = *f.getBody().front().getOps<triton::BroadcastOp>().begin();
  SmallVector<arith::AddIOp> adds;
  f.walk([&](arith::AddIOp op) { adds.push_back(op); });
  SmallVector<arith::MulIOp> muls;
  f.walk([&](arith::MulIOp op) { muls.push_back(op); });
  auto ptr = *f.getBody().front().getOps<triton::AddPtrOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  PtrAnalysis<CommonPtrRules> ordinary(b, CommonPtrRules{});
  auto common = test::getOffsets(ordinary.analyze(rem));
  ASSERT_TRUE(succeeded(common));
  // Common now analyzes div/rem inputs for invariance and range facts. Only
  // Structured strengthens this varying result into a mixed-radix form.
  EXPECT_TRUE(test::getOffsets(ordinary.findCached(rem.getLhs())));
  EXPECT_FALSE(common->hasCompleteForm());
  EXPECT_FALSE(common->axes[0].subAxes);
  PtrAnalysis<StructuredPtrRules> driver(b, StructuredPtrRules{});
  auto shifted = test::getIntegerInfo(driver.analyze(adds[0]));
  ASSERT_TRUE(succeeded(shifted));
  EXPECT_FALSE(shifted->offsets.hasStructuredForm());
  EXPECT_TRUE(shifted->offsets.hasCompleteForm());
  EXPECT_EQ(evaluate(shifted->offsets.scalarOffset)->front(),
            llvm::APInt(64, 5));
  ASSERT_TRUE(shifted->offsets.axes[0].subAxes);
  EXPECT_EQ(shifted->offsets.axes[0].kind, AxisKind::Unknown);
  EXPECT_TRUE(driver.findCached(rem.getLhs()));
  EXPECT_TRUE(driver.findCached(rem.getRhs()));
  auto tiled = test::getIntegerInfo(driver.analyze(rows));
  ASSERT_TRUE(succeeded(tiled));
  EXPECT_EQ(tiled->offsets.axes[0].kind, AxisKind::Invariant);
  ASSERT_TRUE(tiled->offsets.axes[1].subAxes);
  auto values = evaluate(tiled->offsets);
  ASSERT_TRUE(values);
  for (unsigned i = 0; i < values->size(); ++i)
    EXPECT_EQ((*values)[i], llvm::APInt(64, 10 + 2 * (i % 4)));
  auto linear = test::getIntegerInfo(driver.analyze(adds[1]));
  ASSERT_TRUE(succeeded(linear));
  EXPECT_TRUE(linear->offsets.hasStructuredForm());
  EXPECT_EQ(evaluate(linear->offsets.axes[0].stride)->front(),
            llvm::APInt(64, 1));
  auto ptrInfo = driver.analyze(ptr);
  ASSERT_TRUE(succeeded(ptrInfo));
  auto *p = getPtrInfo(*ptrInfo);
  ASSERT_TRUE(p);
  ASSERT_TRUE(p->offsets.axes[0].subAxes);
  EXPECT_TRUE(p->offsets.hasCompleteForm());
  EXPECT_EQ(evaluate(p->offsets.scalarOffset)->front(), llvm::APInt(64, 10));
  auto dynamic = test::getIntegerInfo(driver.analyze(muls.back()));
  ASSERT_TRUE(succeeded(dynamic));
  auto coefficient = dynamic->offsets.axes[0].subAxes->back().stride;
  EXPECT_TRUE(isa<Value>(coefficient));
  test::Environment env{{f.getArgument(1), {llvm::APInt(32, 3)}}};
  EXPECT_EQ(evaluate(coefficient, env)->front(), llvm::APInt(64, 6));
  ASSERT_TRUE(succeeded(test::getOffsets(driver.analyze(muls.back()))));
  unsigned count = operations(*module);
  ASSERT_TRUE(succeeded(driver.analyze(rows)));
  ASSERT_TRUE(succeeded(driver.analyze(ptr)));
  ASSERT_TRUE(succeeded(driver.analyze(muls.back())));
  EXPECT_EQ(operations(*module), count);
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisExtension,
     UnavailableSplitAndMaterializationPreserveExactResults) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module {tt.func @test(%d:tensor<16xi32>,%wide:i128) {
    %k=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
    %rem=arith.remui %k,%d:tensor<16xi32>
    tt.return
  }})mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto rem = *f.getBody().front().getOps<arith::RemUIOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis<StructuredPtrRules> driver(b, StructuredPtrRules{});
  auto info = test::getIntegerInfo(driver.analyze(rem));
  ASSERT_TRUE(succeeded(info));
  EXPECT_FALSE(info->offsets.axes[0].subAxes);
  EXPECT_EQ(cast<Value>(info->offsets.fullOffsets), rem.getResult());
  EXPECT_FALSE(info->offsets.scalarOffset);
  ASSERT_TRUE(succeeded(driver.analyze(f.getArgument(1))));
  auto *cached = driver.findCached(f.getArgument(1));
  auto count = operations(*module);
  auto wide = test::getOffsets(driver.analyze(f.getArgument(1)));
  ASSERT_TRUE(succeeded(wide));
  EXPECT_TRUE(failed(materializeOffsetAt(*wide, {}, b, f.getLoc())));
  EXPECT_EQ(driver.findCached(f.getArgument(1)), cached);
  EXPECT_EQ(operations(*module), count);
}

TEST(PtrAnalysisExtension, SubAxisSSAControlsPlacementAndBindings) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module {tt.func @test(%cond:i1,%x:i32,%y:i32,%boundary:tensor<16xi32>,%block:!tt.ptr<tensor<4xi32>>) {
    %root:2=scf.if %cond -> (i32,index) {
      %four=arith.constant 4:index
      scf.yield %x,%four:i32,index
    } else {
      %four=arith.constant 4:index
      scf.yield %y,%four:i32,index
    }
    %rows=tt.expand_dims %boundary {axis=0:i32}:tensor<16xi32> -> tensor<1x16xi32>
    %advance=tt.advance %block,[%root#0]:!tt.ptr<tensor<4xi32>>
    tt.return
  }})mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto root = *f.getBody().front().getOps<scf::IfOp>().begin();
  auto rows = *f.getBody().front().getOps<triton::ExpandDimsOp>().begin();
  auto advance = *f.getBody().front().getOps<triton::AdvanceOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  PtrAnalysis<StructuredPtrRules> driver(b, StructuredPtrRules{});
  auto info = makeOpaqueOffset(f.getArgument(3), b);
  ASSERT_TRUE(succeeded(info));
  info->scalarOffset = b.getIndexAttr(5);
  b.setInsertionPoint(rows);
  Value rootStride = b.create<arith::IndexCastOp>(f.getLoc(), b.getIndexType(),
                                                  root.getResult(0));
  b.restoreInsertionPoint(ip);
  info->axes[0].subAxes = SmallVector<SubAxisInfo>{
      {root.getResult(1), b.getIndexAttr(0)}, {b.getIndexAttr(4), rootStride}};
  IntegerInfo integer{*info, llvm::ConstantRange::getFull(32)};
  ASSERT_TRUE(succeeded(
      driver.bind(f.getArgument(3), CommonPtrRules::Result(integer))));
  ASSERT_TRUE(succeeded(driver.analyze(rows)));
  ASSERT_TRUE(succeeded(test::getOffsets(driver.analyze(f.getArgument(3)))));
  auto rowsInfo = test::getOffsets(driver.analyze(rows));
  ASSERT_TRUE(succeeded(rowsInfo));
  EXPECT_EQ(rowsInfo->axes[1].subAxes->front().shape,
            OpFoldResult(root.getResult(1)));
  EXPECT_EQ(rowsInfo->axes[1].subAxes->back().stride, OpFoldResult(rootStride));
  EXPECT_EQ(b.getInsertionBlock(), ip.getBlock());
  ASSERT_TRUE(succeeded(
      driver.bind(f.getArgument(3), *driver.findCached(f.getArgument(3)))));
  EXPECT_FALSE(driver.findCached(rows));
  driver.clear();
  EXPECT_FALSE(driver.findCached(f.getArgument(3)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisExtension, SubAxisCastsKeepExactValuesAndRevokeInvalidProofs) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {tt.func @test() {
    %k=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
    %four=arith.constant dense<4>:tensor<16xi32>
    %eight=arith.constant dense<8>:tensor<16xi32>
    %shift=arith.constant dense<126>:tensor<16xi32>
    %r4=arith.remui %k,%four:tensor<16xi32>
    %r8=arith.remui %k,%eight:tensor<16xi32>
    %mixed=arith.addi %r4,%r8:tensor<16xi32>
    %offset=arith.addi %r4,%shift:tensor<16xi32>
    %wrapped=arith.trunci %offset:tensor<16xi32> to tensor<16xi8>
    %small=arith.trunci %r4:tensor<16xi32> to tensor<16xi8>
    %signed=arith.extsi %wrapped:tensor<16xi8> to tensor<16xi32>
    %unsigned=arith.extui %wrapped:tensor<16xi8> to tensor<16xi32>
    tt.return
  }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  SmallVector<arith::TruncIOp> truncs;
  f.walk([&](arith::TruncIOp op) { truncs.push_back(op); });
  auto mixed = *f.getBody().front().getOps<arith::AddIOp>().begin();
  auto ext = *f.getBody().front().getOps<arith::ExtSIOp>().begin();
  auto extu = *f.getBody().front().getOps<arith::ExtUIOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis<StructuredPtrRules> driver(b, StructuredPtrRules{});
  auto incompatible = test::getIntegerInfo(driver.analyze(mixed));
  ASSERT_TRUE(succeeded(incompatible));
  EXPECT_FALSE(incompatible->offsets.axes[0].subAxes);
  EXPECT_FALSE(incompatible->offsets.hasCompleteForm());
  EXPECT_EQ(evaluate(incompatible->offsets.scalarOffset)->front(),
            llvm::APInt(64, 0));
  auto wrapped = test::getIntegerInfo(driver.analyze(truncs[0]));
  ASSERT_TRUE(succeeded(wrapped));
  EXPECT_FALSE(wrapped->offsets.axes[0].subAxes);
  auto address = test::getOffsets(driver.analyze(truncs[0]));
  auto small = test::getOffsets(driver.analyze(truncs[1]));
  ASSERT_TRUE(succeeded(address));
  ASSERT_TRUE(succeeded(small));
  EXPECT_FALSE(address->axes[0].subAxes);
  EXPECT_TRUE(small->axes[0].subAxes);
  EXPECT_EQ(address->axes[0].kind, AxisKind::Unknown);
  EXPECT_FALSE(address->scalarOffset);
  auto signedInfo = test::getIntegerInfo(driver.analyze(ext)),
       unsignedInfo = test::getIntegerInfo(driver.analyze(extu));
  ASSERT_TRUE(succeeded(signedInfo));
  ASSERT_TRUE(succeeded(unsignedInfo));
  EXPECT_FALSE(signedInfo->offsets.axes[0].subAxes);
  EXPECT_FALSE(unsignedInfo->offsets.axes[0].subAxes);
  auto values = evaluate(*address);
  auto unsignedValues = evaluate(unsignedInfo->offsets);
  ASSERT_TRUE(values);
  ASSERT_TRUE(unsignedValues);
  for (unsigned i = 0; i < 16; ++i) {
    EXPECT_EQ((*values)[i], llvm::APInt(8, 126 + i % 4).sext(64));
    EXPECT_EQ((*unsignedValues)[i], llvm::APInt(64, 126 + i % 4));
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisExtension, SubAxisBindingsValidateIndexFieldsAndLengths) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @test(%boundary:tensor<16xi32>,%stride:index) {
      %rows=tt.expand_dims %boundary {axis=0:i32}:tensor<16xi32> -> tensor<1x16xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  auto rows = *f.getBody().front().getOps<ExpandDimsOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis<StructuredPtrRules> driver(b, StructuredPtrRules{});
  auto offset = makeOpaqueOffset(f.getArgument(0), b);
  ASSERT_TRUE(succeeded(offset));
  offset->scalarOffset = b.getIndexAttr(5);
  offset->axes[0].subAxes =
      SmallVector<SubAxisInfo>{{b.getIndexAttr(4), b.getIndexAttr(0)},
                               {b.getIndexAttr(4), f.getArgument(1)}};
  auto bound = *makeIntegerInfo(*offset);
  ASSERT_TRUE(succeeded(driver.bind(f.getArgument(0), bound)));
  auto *original = driver.findCached(f.getArgument(0));
  auto bad = bound;
  bad.offsets.axes[0].subAxes->front().shape = b.getIndexAttr(0);
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bad)));
  bad = bound;
  bad.offsets.axes[0].subAxes->back().stride = b.getI32IntegerAttr(1);
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bad)));
  EXPECT_EQ(driver.findCached(f.getArgument(0)), original);
  auto derived = test::getOffsets(driver.analyze(rows));
  ASSERT_TRUE(succeeded(derived));
  ASSERT_TRUE(derived->axes[1].subAxes);
  EXPECT_EQ(derived->axes[1].subAxes->back().stride,
            OpFoldResult(f.getArgument(1)));
  EXPECT_TRUE(succeeded(verify(*module)));
}
} // namespace
