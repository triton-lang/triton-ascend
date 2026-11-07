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

// This rule establishes that the session only requires its chosen Result and
// the lifecycle contract; it does not depend on PtrInfo fields.
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
TEST(PtrAnalysisExtension, HypothesisQueriesIsolateBindingsAndComputedResults) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @test(%x:i32, %y:i32) {
      %sum = arith.addi %x, %y : i32
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto sum = *f.getBody().front().getOps<arith::AddIOp>().begin();
  Value x = f.getArgument(0), y = f.getArgument(1);
  OpBuilder builder(f.getBody().front().getTerminator());
  auto ip = builder.saveInsertionPoint();
  PtrAnalysis<TokenRules> analysis(builder, TokenRules{});
  auto parent = analysis.analyze(sum);
  ASSERT_TRUE(succeeded(parent));
  TokenInfo childResult, grandchildResult;
  ASSERT_TRUE(succeeded(analysis.withBindings({{x, {y, y}}}, [&](auto &child) {
    EXPECT_FALSE(child.findCached(sum));
    auto info = child.analyze(sum);
    if (failed(info))
      return failure();
    childResult = *info;
    EXPECT_TRUE(
        succeeded(child.withBindings({{y, {x, x}}}, [&](auto &grandchild) {
          EXPECT_FALSE(grandchild.findCached(sum));
          auto nested = grandchild.analyze(sum);
          if (failed(nested))
            return failure();
          grandchildResult = *nested;
          return success();
        })));
    EXPECT_EQ(child.findCached(sum)->helper, childResult.helper);
    return success();
  })));
  EXPECT_EQ(analysis.findCached(sum)->helper, parent->helper);
  EXPECT_FALSE(analysis.findCached(x)->helper == y);
  test::Environment env{{x, {llvm::APInt(32, 2)}}, {y, {llvm::APInt(32, 5)}}};
  auto before = test::evaluate(parent->helper, env);
  auto local = test::evaluate(childResult.helper, env);
  auto nested = test::evaluate(grandchildResult.helper, env);
  ASSERT_TRUE(before && local && nested);
  EXPECT_EQ(before->front().getSExtValue(), 7);
  EXPECT_EQ(local->front().getSExtValue(), 10);
  EXPECT_EQ(nested->front().getSExtValue(), 7);
  bool called = false;
  EXPECT_TRUE(failed(analysis.withBindings({{x, {}}}, [&](auto &) {
    called = true;
    return success();
  })));
  EXPECT_FALSE(called);
  EXPECT_EQ(analysis.findCached(sum)->helper, parent->helper);
  EXPECT_EQ(builder.getInsertionPoint(), ip.getPoint());
  EXPECT_TRUE(succeeded(verify(*module)));
}

// An extension can recognize an exact identity that Common does not model.
// Its configuration and recursive input queries must survive loop hypotheses.
struct IdentityShiftRules : CommonPtrRules {
  IdentityShiftRules(bool enabled, unsigned &queries)
      : enabled(enabled), queries(queries) {}
  bool enabled;
  unsigned &queries;
  template <typename Context>
  FailureOr<Result> parse(Value value, Context &context) const {
    if (auto shift = value.getDefiningOp<arith::ShLIOp>(); shift && enabled) {
      auto amount =
          mlir::triton::ptr::detail::getIntegerConstant(shift.getRhs(), 64);
      if (succeeded(amount) && amount->isZero()) {
        ++queries;
        auto input = context.analyze(shift.getLhs());
        if (failed(input))
          return failure();
        input->offsets.fullOffsets = value;
        return input;
      }
    }
    return CommonPtrRules::parse(value, context);
  }
};

TEST(PtrAnalysisExtension, LoopProofKeepsConfiguredRulesAndExternalBindings) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%seed: tensor<4xi32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant dense<0> : tensor<4xi32>
      %out = scf.for %i = %c0 to %n step %c1 iter_args(%x = %seed) -> tensor<4xi32> {
        %shift = arith.shli %x, %zero : tensor<4xi32>
        %next = arith.addi %shift, %seed : tensor<4xi32>
        scf.yield %next : tensor<4xi32>
      }
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto loop = *f.getBody().front().getOps<scf::ForOp>().begin();
  OpBuilder builder(loop);
  auto seed = buildOffsetInfo(f.getArgument(0), builder);
  ASSERT_TRUE(succeeded(seed));
  seed->scalarOffset = builder.getIndexAttr(7);
  seed->axes[0] = AxisInfo(AxisKind::Structured, builder.getIndexAttr(3));
  unsigned queries = 0;
  for (bool enabled : {false, true}) {
    PtrAnalysis<IdentityShiftRules> analysis(
        builder, IdentityShiftRules(enabled, queries));
    ASSERT_TRUE(succeeded(analysis.bind(f.getArgument(0), PtrInfo{{}, *seed})));
    auto result = analysis.analyze(loop.getResult(0));
    ASSERT_TRUE(succeeded(result));
    EXPECT_EQ(result->offsets.axes[0].kind,
              enabled ? AxisKind::Structured : AxisKind::Unknown);
    EXPECT_EQ(analysis.findCached(f.getArgument(0))->offsets.scalarOffset,
              seed->scalarOffset);
  }
  EXPECT_GT(queries, 0u);
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
  PtrAnalysis<CommonPtrRules> driver(b, CommonPtrRules{});
  auto common = test::getOffsets(driver.analyze(rem));
  ASSERT_TRUE(succeeded(common));
  EXPECT_TRUE(common->hasStructuredDecomposition());
  EXPECT_FALSE(common->axes[0].subAxes.empty());
  auto shifted = test::getInfo(driver.analyze(adds[0]));
  ASSERT_TRUE(succeeded(shifted));
  EXPECT_FALSE(shifted->offsets.hasStructuredForm());
  EXPECT_TRUE(shifted->offsets.hasStructuredDecomposition());
  EXPECT_EQ(evaluate(shifted->offsets.scalarOffset)->front(),
            llvm::APInt(64, 5));
  ASSERT_FALSE(shifted->offsets.axes[0].subAxes.empty());
  EXPECT_EQ(shifted->offsets.axes[0].kind, AxisKind::Unknown);
  EXPECT_TRUE(driver.findCached(rem.getLhs()));
  EXPECT_TRUE(driver.findCached(rem.getRhs()));
  auto tiled = test::getInfo(driver.analyze(rows));
  ASSERT_TRUE(succeeded(tiled));
  EXPECT_EQ(tiled->offsets.axes[0].kind, AxisKind::Invariant);
  ASSERT_FALSE(tiled->offsets.axes[1].subAxes.empty());
  auto values = evaluate(tiled->offsets);
  ASSERT_TRUE(values);
  for (unsigned i = 0; i < values->size(); ++i)
    EXPECT_EQ((*values)[i], llvm::APInt(64, 10 + 2 * (i % 4)));
  auto linear = test::getInfo(driver.analyze(adds[1]));
  ASSERT_TRUE(succeeded(linear));
  EXPECT_TRUE(linear->offsets.hasStructuredForm());
  EXPECT_EQ(evaluate(linear->offsets.axes[0].stride)->front(),
            llvm::APInt(64, 1));
  auto ptrInfo = driver.analyze(ptr);
  ASSERT_TRUE(succeeded(ptrInfo));
  ASSERT_FALSE(ptrInfo->isEmpty());
  ASSERT_TRUE(ptrInfo->base);
  auto *p = &*ptrInfo;
  ASSERT_FALSE(p->offsets.axes[0].subAxes.empty());
  EXPECT_TRUE(p->offsets.hasStructuredDecomposition());
  EXPECT_EQ(evaluate(p->offsets.scalarOffset)->front(), llvm::APInt(64, 10));
  auto dynamic = test::getInfo(driver.analyze(muls.back()));
  ASSERT_TRUE(succeeded(dynamic));
  auto coefficient = dynamic->offsets.axes[0].subAxes.back().stride;
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
      R"mlir(module {tt.func @test(%d:tensor<16xi32>) {
    %k=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
    %rem=arith.remui %k,%d:tensor<16xi32>
    tt.return
  }})mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto rem = *f.getBody().front().getOps<arith::RemUIOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis<CommonPtrRules> driver(b, CommonPtrRules{});
  auto info = test::getInfo(driver.analyze(rem));
  ASSERT_TRUE(succeeded(info));
  EXPECT_TRUE(info->offsets.axes[0].subAxes.empty());
  EXPECT_EQ(cast<Value>(info->offsets.fullOffsets), rem.getResult());
  EXPECT_FALSE(info->offsets.scalarOffset);
  auto *cached = driver.findCached(rem);
  auto count = operations(*module);
  // A rank-one offset needs one coordinate to select a lane.
  EXPECT_TRUE(failed(materializeOffsetAt(info->offsets, {}, b, f.getLoc())));
  EXPECT_EQ(driver.findCached(rem), cached);
  EXPECT_EQ(operations(*module), count);
}

TEST(PtrAnalysisExtension, SubAxisStrideSSAControlsPlacementAndBindings) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module {tt.func @test(%cond:i1,%x:i32,%y:i32,%boundary:tensor<16xi32>,%block:!tt.ptr<tensor<4xi32>>) {
    %root=scf.if %cond -> (i32) {
      scf.yield %x:i32
    } else {
      scf.yield %y:i32
    }
    %rows=tt.expand_dims %boundary {axis=0:i32}:tensor<16xi32> -> tensor<1x16xi32>
    %advance=tt.advance %block,[%root]:!tt.ptr<tensor<4xi32>>
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
  PtrAnalysis<CommonPtrRules> driver(b, CommonPtrRules{});
  auto info = buildOffsetInfo(f.getArgument(3), b);
  ASSERT_TRUE(succeeded(info));
  info->scalarOffset = b.getIndexAttr(5);
  b.setInsertionPoint(rows);
  Value rootStride = b.create<arith::IndexCastOp>(f.getLoc(), b.getIndexType(),
                                                  root.getResult(0));
  b.restoreInsertionPoint(ip);
  info->axes[0].subAxes =
      SmallVector<SubAxisInfo>{{4, b.getIndexAttr(0)}, {4, rootStride}};
  PtrInfo integer{{}, *info};
  ASSERT_TRUE(succeeded(
      driver.bind(f.getArgument(3), CommonPtrRules::Result(integer))));
  ASSERT_TRUE(succeeded(driver.analyze(rows)));
  ASSERT_TRUE(succeeded(test::getOffsets(driver.analyze(f.getArgument(3)))));
  auto rowsInfo = test::getOffsets(driver.analyze(rows));
  ASSERT_TRUE(succeeded(rowsInfo));
  EXPECT_EQ(rowsInfo->axes[1].subAxes.front().shape, 4);
  EXPECT_EQ(rowsInfo->axes[1].subAxes.back().stride, OpFoldResult(rootStride));
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
  PtrAnalysis<CommonPtrRules> driver(b, CommonPtrRules{});
  auto incompatible = test::getInfo(driver.analyze(mixed));
  ASSERT_TRUE(succeeded(incompatible));
  EXPECT_TRUE(incompatible->offsets.axes[0].subAxes.empty());
  EXPECT_FALSE(incompatible->offsets.hasStructuredDecomposition());
  EXPECT_EQ(evaluate(incompatible->offsets.scalarOffset)->front(),
            llvm::APInt(64, 0));
  auto wrapped = test::getInfo(driver.analyze(truncs[0]));
  ASSERT_TRUE(succeeded(wrapped));
  EXPECT_TRUE(wrapped->offsets.axes[0].subAxes.empty());
  auto address = test::getOffsets(driver.analyze(truncs[0]));
  auto small = test::getOffsets(driver.analyze(truncs[1]));
  ASSERT_TRUE(succeeded(address));
  ASSERT_TRUE(succeeded(small));
  EXPECT_TRUE(address->axes[0].subAxes.empty());
  EXPECT_FALSE(small->axes[0].subAxes.empty());
  EXPECT_EQ(address->axes[0].kind, AxisKind::Unknown);
  EXPECT_FALSE(address->scalarOffset);
  auto signedInfo = test::getInfo(driver.analyze(ext)),
       unsignedInfo = test::getInfo(driver.analyze(extu));
  ASSERT_TRUE(succeeded(signedInfo));
  ASSERT_TRUE(succeeded(unsignedInfo));
  EXPECT_TRUE(signedInfo->offsets.axes[0].subAxes.empty());
  EXPECT_TRUE(unsignedInfo->offsets.axes[0].subAxes.empty());
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
  PtrAnalysis<CommonPtrRules> driver(b, CommonPtrRules{});
  auto offset = buildOffsetInfo(f.getArgument(0), b);
  ASSERT_TRUE(succeeded(offset));
  offset->scalarOffset = b.getIndexAttr(5);
  offset->axes[0].subAxes =
      SmallVector<SubAxisInfo>{{4, b.getIndexAttr(0)}, {4, f.getArgument(1)}};
  auto bound = PtrInfo{{}, *offset};
  ASSERT_TRUE(succeeded(driver.bind(f.getArgument(0), bound)));
  auto *original = driver.findCached(f.getArgument(0));
  for (int64_t length : {0, -1, 3}) {
    auto bad = bound;
    bad.offsets.axes[0].subAxes.front().shape = length;
    EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bad)));
  }
  auto bad = bound;
  bad.offsets.axes[0].subAxes.back().stride = b.getI32IntegerAttr(1);
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bad)));
  EXPECT_EQ(driver.findCached(f.getArgument(0)), original);
  auto derived = test::getOffsets(driver.analyze(rows));
  ASSERT_TRUE(succeeded(derived));
  ASSERT_FALSE(derived->axes[1].subAxes.empty());
  EXPECT_EQ(derived->axes[1].subAxes.back().stride,
            OpFoldResult(f.getArgument(1)));
  // Reconstruct lanes from static lengths and a dynamic stride.
  offset->fullOffsets = {};
  test::Environment env{{f.getArgument(1), {llvm::APInt(64, 7)}}};
  for (int64_t lane : {0, 3, 4, 15}) {
    Value index = b.create<arith::ConstantIndexOp>(f.getLoc(), lane);
    auto value = materializeOffsetAt(*offset, ValueRange{index}, b, f.getLoc());
    ASSERT_TRUE(succeeded(value));
    auto actual = evaluate(*value, env);
    ASSERT_TRUE(actual);
    ASSERT_EQ(actual->size(), 1u);
    EXPECT_EQ(actual->front().getSExtValue(), 5 + (lane % 4) * 7);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisExtension, AxisBindingsRequireConsistentFacts) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @test(%boundary:tensor<16xi32>,%unit:tensor<1xi32>,%stride:index) {
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis<CommonPtrRules> driver(b, CommonPtrRules{});
  auto offset = buildOffsetInfo(f.getArgument(0), b);
  ASSERT_TRUE(succeeded(offset));
  PtrInfo bound{{}, *offset};
  auto &axis = bound.offsets.axes[0];
  unsigned count = operations(*module);

  // Unknown stride fields are placeholders, not facts to upgrade or rewrite.
  axis.stride = b.getIndexAttr(7);
  ASSERT_TRUE(succeeded(driver.bind(f.getArgument(0), bound)));
  EXPECT_EQ(driver.findCached(f.getArgument(0))->offsets.axes[0].stride,
            axis.stride);
  axis.kind = AxisKind::Invariant;
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bound)));
  axis = AxisInfo(AxisKind::Structured, b.getIndexAttr(0));
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bound)));

  // Nonlinear digits retain Unknown. Linear lanes 0..15 require stride one.
  axis = AxisInfo(AxisKind::Unknown, b.getIndexAttr(7));
  axis.subAxes = {{4, b.getIndexAttr(0)}, {4, b.getIndexAttr(1)}};
  ASSERT_TRUE(succeeded(driver.bind(f.getArgument(0), bound)));
  axis.subAxes = {{4, b.getIndexAttr(4)}, {4, b.getIndexAttr(1)}};
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bound)));
  axis.kind = AxisKind::Structured;
  axis.stride = b.getIndexAttr(1);
  ASSERT_TRUE(succeeded(driver.bind(f.getArgument(0), bound)));
  axis.stride = b.getIndexAttr(2);
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bound)));
  // The invalid length product would wrap back to 16 without overflow checks.
  axis.stride = b.getIndexAttr(1);
  axis.subAxes.front().shape = (int64_t{1} << 62) + 4;
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bound)));

  // Unit digits have no coordinate contribution; the sole varying digit
  // can carry a dynamic stride directly.
  axis.stride = f.getArgument(2);
  axis.subAxes = {
      {1, b.getIndexAttr(9)}, {16, f.getArgument(2)}, {1, b.getIndexAttr(3)}};
  ASSERT_TRUE(succeeded(driver.bind(f.getArgument(0), bound)));
  axis.kind = AxisKind::Unknown;
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bound)));
  axis.kind = AxisKind::Structured;
  axis.stride = b.getIndexAttr(1);
  EXPECT_TRUE(failed(driver.bind(f.getArgument(0), bound)));

  // A singleton partition is invariant even with arbitrary coefficients.
  auto unit = buildOffsetInfo(f.getArgument(1), b);
  ASSERT_TRUE(succeeded(unit));
  unit->axes[0] = AxisInfo(AxisKind::Invariant, b.getIndexAttr(0));
  unit->axes[0].subAxes = {{1, f.getArgument(2)}, {1, b.getIndexAttr(9)}};
  ASSERT_TRUE(succeeded(driver.bind(f.getArgument(1), PtrInfo{{}, *unit})));
  unit->axes[0].kind = AxisKind::Unknown;
  EXPECT_TRUE(failed(driver.bind(f.getArgument(1), PtrInfo{{}, *unit})));
  EXPECT_EQ(operations(*module), count);
  EXPECT_TRUE(succeeded(verify(*module)));
}
} // namespace
