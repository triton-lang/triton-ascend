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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;
using test::buildAddressView;
using test::eraseDeadPtrArithmetic;

namespace {

unsigned countOperations(Operation *scope) {
  unsigned count = 0;
  scope->walk([&](Operation *) { ++count; });
  return count;
}

TEST(PtrAnalysisLifecycle, FailedBlockSourcesCanBeProvidedForRetry) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @retry(%base: !tt.ptr<i32>, %unknown: !tt.ptr<tensor<4xi32>>) {
      %size = arith.constant 16 : i64
      %stride = arith.constant 1 : i64
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %known = tt.make_tensor_ptr %base, [%size], [%stride], [%zero]
          {order = array<i32: 0>} : !tt.ptr<tensor<4xi32>>
      %next = tt.advance %unknown, [%one] : !tt.ptr<tensor<4xi32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto known =
      *func.getBody().front().getOps<triton::MakeTensorPtrOp>().begin();
  auto next = *func.getBody().front().getOps<triton::AdvanceOp>().begin();
  OpBuilder builder(func.getBody().front().getTerminator());
  PtrAnalysis analysis(builder);
  unsigned before = countOperations(func);
  for (Value value :
       SmallVector<Value>{func.getArgument(1), next.getResult()}) {
    EXPECT_TRUE(failed(analysis.analyze(value)));
    EXPECT_EQ(analysis.findCached(value), nullptr);
  }
  EXPECT_EQ(countOperations(func), before);
  auto info = analysis.analyze(known.getResult());
  ASSERT_TRUE(succeeded(info));
  ASSERT_TRUE(succeeded(analysis.setInfo(func.getArgument(1), *info)));
  auto result = analysis.analyze(next);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->base, func.getArgument(0));
  auto coordinate = test::evaluate(result->block->coordinateOffsets[0]);
  ASSERT_TRUE(coordinate);
  EXPECT_EQ(coordinate->front().getSExtValue(), 1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, NumericHelpersCanBeCleanedAfterSession) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @cleanup(%delta: i32) {
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  ASSERT_TRUE(
      succeeded(test::getOffsets(analysis.analyze(func.getArgument(0)))));
  unsigned before = 0;
  func.walk([&](arith::IndexCastOp) { ++before; });
  EXPECT_EQ(before, 1u);
  ASSERT_TRUE(succeeded(analysis.reset()));
  IRRewriter rewriter(&context);
  eraseDeadPtrArithmetic(func, rewriter);
  unsigned after = 0;
  func.walk([&](arith::IndexCastOp) { ++after; });
  EXPECT_EQ(after, 0u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, ResetFactsRecomputesIntegerAndPointerConsumers) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @bindings(%base: !tt.ptr<i32>, %boundary: tensor<4x!tt.ptr<i32>>, %x: i32, %y: i32) {
      %opaque = arith.divsi %x, %y : i32
      %one = arith.constant 1 : i32
      %sum = arith.addi %opaque, %one : i32
      %delta = tt.splat %sum : i32 -> tensor<4xi32>
      %next = tt.addptr %boundary, %delta : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  Value opaque = *f.getBody().front().getOps<arith::DivSIOp>().begin();
  Value sum = *f.getBody().front().getOps<arith::AddIOp>().begin();
  auto next = *f.getBody().front().getOps<triton::AddPtrOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto opaquePointer = analysis.analyze(f.getArgument(1));
  ASSERT_TRUE(succeeded(opaquePointer));
  EXPECT_FALSE(opaquePointer->hasScalarBase());
  auto scalar = test::getOffsets(analysis.analyze(opaque));
  ASSERT_TRUE(succeeded(scalar));
  for (int64_t n : {5, -8}) {
    scalar->fullOffsets = b.getI32IntegerAttr(n);
    scalar->scalarOffset = b.getIndexAttr(n);
    PtrInfo boundary{{}, *scalar};
    PtrInfo p;
    p.base = f.getArgument(0);

    p.offsets.valueType = RankedTensorType::get({4}, b.getIndexType());
    p.offsets.fullOffsets = DenseIntElementsAttr::get(
        cast<RankedTensorType>(p.offsets.valueType), llvm::APInt(64, n, true));
    p.offsets.scalarOffset = b.getIndexAttr(n);
    p.offsets.axes = {AxisInfo(AxisKind::Invariant, b.getIndexAttr(0))};
    ASSERT_TRUE(
        succeeded(analysis.reset({{opaque, boundary}, {f.getArgument(1), p}})));
    auto derived = test::getInfo(analysis.analyze(next));
    ASSERT_TRUE(succeeded(derived));
    auto values =
        test::evaluate(derived->offsets,
                       test::Environment{{opaque, {llvm::APInt(32, n, true)}}});
    ASSERT_TRUE(values);
    for (auto value : *values)
      EXPECT_EQ(value, llvm::APInt(64, 2 * n + 1, true));
    EXPECT_FALSE(test::getOffsets(analysis.findCached(f.getArgument(2))));
    EXPECT_FALSE(test::getOffsets(analysis.findCached(f.getArgument(3))));
    auto bad = boundary;
    bad.offsets.scalarOffset = b.getI32IntegerAttr(n);
    EXPECT_TRUE(failed(analysis.setInfo(opaque, bad)));
    auto badPtr = p;
    badPtr.base = f.getArgument(2);
    EXPECT_TRUE(failed(analysis.setInfo(f.getArgument(1), badPtr)));
    auto retained = test::getOffsets(analysis.analyze(sum));
    ASSERT_TRUE(succeeded(retained));
    auto actual =
        test::evaluate(retained->scalarOffset,
                       test::Environment{{opaque, {llvm::APInt(32, n, true)}}});
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->front(), llvm::APInt(64, n + 1, true));
    EXPECT_TRUE(succeeded(verify(*module)));
  }
  ASSERT_TRUE(succeeded(analysis.reset()));
  EXPECT_FALSE(test::getOffsets(analysis.findCached(sum)));
  auto exact = analysis.analyze(next);
  ASSERT_TRUE(succeeded(exact));
  EXPECT_EQ(exact->base, f.getArgument(1));
  auto unbound = test::getOffsets(analysis.analyze(opaque));
  ASSERT_TRUE(succeeded(unbound));
  EXPECT_EQ(cast<Value>(unbound->fullOffsets), opaque);
}

TEST(PtrAnalysisLifecycle, SiblingConsumersReuseHelpersAndPreserveOriginalIR) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      tensor::TensorDialect, triton::TritonDialect>();
  for (bool sourceFirst : {false, true}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { tt.func @siblings(%base: !tt.ptr<i32>, %a: i32, %b: i32, %cond: i1) -> tensor<4xi32> {
        %aa = tt.splat %a : i32 -> tensor<4xi32>
        %bb = tt.splat %b : i32 -> tensor<4xi32>
        %sum = arith.addi %aa, %bb : tensor<4xi32>
        %twice = arith.addi %sum, %sum : tensor<4xi32>
        %ptr = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
        scf.if %cond {
          %p = tt.addptr %ptr, %twice : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
          %v = tt.load %p : tensor<4x!tt.ptr<i32>>
          tt.store %p, %v : tensor<4x!tt.ptr<i32>>
        } else {
          %p = tt.addptr %ptr, %twice : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
          %v = tt.load %p : tensor<4x!tt.ptr<i32>>
          tt.store %p, %v : tensor<4x!tt.ptr<i32>>
        }
        tt.return %twice : tensor<4xi32>
      } }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    SmallVector<std::pair<Operation *, SmallVector<Value>>> snapshot;
    SmallVector<triton::AddPtrOp> pointers;
    module->walk([&](Operation *op) {
      snapshot.emplace_back(op, llvm::to_vector(op->getOperands()));
    });
    f.walk([&](triton::AddPtrOp op) { pointers.push_back(op); });
    Value twice = f.getBody().front().getTerminator()->getOperand(0);
    OpBuilder b(&context);
    PtrAnalysis analysis(b);
    if (sourceFirst) {
      ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(twice))));
      ASSERT_TRUE(succeeded(verify(*module)));
    }
    for (auto p : pointers) {
      ASSERT_TRUE(succeeded(test::getInfo(analysis.analyze(p))));
      ASSERT_TRUE(succeeded(verify(*module)));
    }
    ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(twice))));
    ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(twice))));
    for (const auto &[op, operands] : snapshot) {
      EXPECT_EQ(llvm::to_vector(op->getOperands()), operands);
    }
    unsigned helpersBefore = countOperations(*module);
    for (auto p : pointers)
      ASSERT_TRUE(succeeded(test::getInfo(analysis.analyze(p))));
    ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(twice))));
    EXPECT_EQ(countOperations(*module), helpersBefore);
    module->walk([&](Operation *op) {
      if (llvm::any_of(snapshot,
                       [&](auto &entry) { return entry.first == op; }))
        return;
      EXPECT_TRUE(isMemoryEffectFree(op));
      EXPECT_TRUE((op->getName().getDialectNamespace() == "arith" ||
                   isa<tensor::ExtractOp, triton::SplatOp, triton::BroadcastOp,
                       triton::ExpandDimsOp, triton::MakeRangeOp>(op)));
    });
    auto address = test::getOffsets(analysis.analyze(twice));
    ASSERT_TRUE(succeeded(address));
    Value liveHelper = cast<Value>(address->scalarOffset);
    b.setInsertionPoint(f.getBody().front().getTerminator());
    Value narrowed =
        b.create<arith::IndexCastOp>(f.getLoc(), b.getI32Type(), liveHelper);
    auto store = b.create<triton::StoreOp>(
        f.getLoc(), f.getArgument(0), narrowed, Value(),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    ASSERT_TRUE(succeeded(analysis.reset()));
    IRRewriter rewriter(&context);
    eraseDeadPtrArithmetic(f, rewriter);
    EXPECT_EQ(store.getValue(), narrowed);
    EXPECT_EQ(narrowed.getDefiningOp()->getOperand(0), liveHelper);
    EXPECT_EQ(f.getBody().front().getTerminator()->getOperand(0), twice);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysisLifecycle, OpaqueFallbackLeavesOnlyCleanablePureHelpers) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @failure(%base: !tt.ptr<i32>, %x: i32, %other: !tt.ptr<i32>, %cond: i1) -> tensor<4x!tt.ptr<i32>> {
      %first = tt.addptr %base, %x : !tt.ptr<i32>, i32
      %s = tt.splat %first : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %t = tt.splat %other : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %bad = arith.select %cond, %s, %t : tensor<4x!tt.ptr<i32>>
      tt.return %bad : tensor<4x!tt.ptr<i32>>
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<triton::AddPtrOp> pointers;
  f.walk([&](triton::AddPtrOp op) { pointers.push_back(op); });
  SmallVector<std::pair<Operation *, SmallVector<Value>>> original;
  f.walk([&](Operation *op) {
    original.emplace_back(op, llvm::to_vector(op->getOperands()));
  });
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  Value selected = f.getBody().front().getTerminator()->getOperand(0);
  auto opaque = analysis.analyze(selected);
  ASSERT_TRUE(succeeded(opaque));
  EXPECT_EQ(opaque->base, selected);
  EXPECT_FALSE(opaque->hasScalarBase());
  unsigned helpers = 0;
  f.walk([&](Operation *op) {
    if (llvm::any_of(original, [&](auto &entry) { return entry.first == op; }))
      return;
    ++helpers;
    EXPECT_TRUE(isMemoryEffectFree(op));
  });
  EXPECT_GT(helpers, 0u);
  for (auto &[op, operands] : original)
    EXPECT_EQ(llvm::to_vector(op->getOperands()), operands);
  EXPECT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(analysis.reset()));
  IRRewriter rewriter(&context);
  eraseDeadPtrArithmetic(f, rewriter);
  for (auto &[op, operands] : original)
    EXPECT_EQ(llvm::to_vector(op->getOperands()), operands);
  unsigned extensions = 0;
  f.walk([&](arith::IndexCastOp) { ++extensions; });
  EXPECT_EQ(extensions, 0u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, ChangingIntegerIterArgKeepsCurrentValue) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      tensor::TensorDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%x: i32) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %r = scf.for %iv = %lb to %ub step %step iter_args(%current = %x) -> i32 {
        %next = arith.addi %current, %current : i32
        scf.yield %next : i32
      }
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto loop = *f.getBody().front().getOps<scf::ForOp>().begin();
  Value current = loop.getRegionIterArgs()[0];
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto state = test::getOffsets(analysis.analyze(current));
  ASSERT_TRUE(succeeded(state));
  EXPECT_EQ(cast<Value>(state->fullOffsets), current);
  auto origin =
      cast<Value>(state->scalarOffset).getDefiningOp<arith::IndexCastOp>();
  ASSERT_TRUE(origin);
  EXPECT_EQ(origin.getIn(), current);
  EXPECT_FALSE(test::getOffsets(analysis.findCached(f.getArgument(0))));
  auto address = test::getOffsets(analysis.analyze(current));
  ASSERT_TRUE(succeeded(address));
  test::Environment env{{current, {llvm::APInt(32, -9, true)}},
                        {f.getArgument(0), {llvm::APInt(32, 100)}}};
  auto actual = test::evaluate(*address, env);
  ASSERT_TRUE(actual);
  EXPECT_EQ(actual->front(), llvm::APInt(64, -9, true));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, RankZeroExtractionIsPureAndCleanable) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @rank_zero(%base: !tt.ptr<i64>, %dead: tensor<i32>, %live: tensor<i32>) {
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(f.getArgument(1)))));
  auto live = test::getOffsets(analysis.analyze(f.getArgument(2)));
  ASSERT_TRUE(succeeded(live));
  Value index = cast<Value>(live->scalarOffset);
  Value scalar =
      b.create<arith::IndexCastOp>(f.getLoc(), b.getI64Type(), index);
  f.walk([&](tensor::ExtractOp op) { EXPECT_TRUE(isMemoryEffectFree(op)); });
  auto store = b.create<triton::StoreOp>(f.getLoc(), f.getArgument(0), scalar,
                                         Value(), triton::CacheModifier::NONE,
                                         triton::EvictionPolicy::NORMAL);
  EXPECT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(analysis.reset()));
  IRRewriter rewriter(&context);
  eraseDeadPtrArithmetic(f, rewriter);
  EXPECT_TRUE(f.getArgument(1).use_empty());
  EXPECT_FALSE(f.getArgument(2).use_empty());
  EXPECT_EQ(store.getValue(), scalar);
  unsigned extracts = 0;
  f.walk([&](tensor::ExtractOp) { ++extracts; });
  EXPECT_EQ(extracts, 1u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, BindIntegerAndPointerResults) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @bindings(%base: !tt.ptr<i32>, %x: i32, %index: index) {
      %s = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %d = tt.splat %x : i32 -> tensor<4xi32>
      %p = tt.addptr %s, %d : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %shape = arith.constant 16 : i64
      %block = tt.make_tensor_ptr %base, [%shape], [%shape], [%x] {order = array<i32: 0>} : !tt.ptr<tensor<4xi32>>
      %next = tt.advance %block, [%x] : !tt.ptr<tensor<4xi32>>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto f = *module->getOps<triton::FuncOp>().begin();
  Value base = f.getArgument(0), x = f.getArgument(1);
  Value splat = *f.getBody().front().getOps<triton::SplatOp>().begin();
  Value add = *f.getBody().front().getOps<triton::AddPtrOp>().begin();
  auto make = *f.getBody().front().getOps<triton::MakeTensorPtrOp>().begin();
  auto advance = *f.getBody().front().getOps<triton::AdvanceOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis reference(builder);
  auto ptr = test::getInfo(reference.analyze(base));
  ASSERT_TRUE(succeeded(ptr));
  auto block = test::getInfo(reference.analyze(make));
  ASSERT_TRUE(succeeded(block));
  PtrAnalysis analysis(builder);
  auto integer = test::getOffsets(analysis.analyze(x));
  ASSERT_TRUE(succeeded(integer));
  ASSERT_TRUE(succeeded(analysis.setInfo(x, PtrInfo{{}, *integer})));
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(f.getArgument(2)))));
  for (Value p : {base, splat, add})
    EXPECT_TRUE(succeeded(test::getInfo(analysis.analyze(p))));
  EXPECT_TRUE(succeeded(analysis.setInfo(base, *ptr)));
  EXPECT_TRUE(succeeded(test::getInfo(analysis.analyze(base))));
  EXPECT_TRUE(succeeded(test::getOffsets(analysis.analyze(x))));
  EXPECT_TRUE(succeeded(test::getInfo(analysis.analyze(make))));
  EXPECT_TRUE(succeeded(analysis.setInfo(make, *block)));
  EXPECT_TRUE(succeeded(test::getInfo(analysis.analyze(make))));
  EXPECT_TRUE(succeeded(test::getInfo(analysis.analyze(advance))));
  EXPECT_NE(test::getInfo(analysis.findCached(make)), nullptr);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, SetInfoPreservesDerivedFactsAndResetRefreshesThem) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @facts(%x: tensor<4xi32>, %other: tensor<4xi32>) {
      %one = arith.constant dense<1> : tensor<4xi32>
      %sum = arith.addi %x, %one : tensor<4xi32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  Value x = f.getArgument(0);
  auto sum = *f.getBody().front().getOps<arith::AddIOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  auto derived = analysis.analyze(sum);
  ASSERT_TRUE(succeeded(derived));
  EXPECT_EQ(derived->offsets.axes[0].kind, AxisKind::Unknown);
  auto generic = analysis.analyze(x);
  ASSERT_TRUE(succeeded(generic));
  PtrInfo precise = *generic;
  precise.offsets.scalarOffset = builder.getIndexAttr(0);
  precise.offsets.axes[0] =
      mlir::triton::ptr::detail::makeInvariantAxisInfo(builder);
  ASSERT_TRUE(succeeded(analysis.setInfo(x, precise)));
  // Existing conservative results stay valid until the caller requests refresh.
  ASSERT_TRUE(analysis.findCached(sum));
  EXPECT_EQ(analysis.findCached(sum)->offsets.axes[0].kind, AxisKind::Unknown);
  ASSERT_TRUE(
      succeeded(analysis.setInfo(f.getArgument(1), *analysis.findCached(x))));
  ASSERT_TRUE(analysis.findCached(sum));
  // Invalid batches are atomic, including after one otherwise valid entry.
  PtrInfo invalid = precise;
  invalid.offsets.scalarOffset = builder.getI32IntegerAttr(0);
  EXPECT_TRUE(
      failed(analysis.reset({{x, precise}, {f.getArgument(1), invalid}})));
  ASSERT_TRUE(analysis.findCached(sum));
  EXPECT_TRUE(analysis.findCached(f.getArgument(1)));
  // Last duplicate wins; borrowed cached input is copied before replacement.
  ASSERT_TRUE(
      succeeded(analysis.reset({{x, *generic}, {x, *analysis.findCached(x)}})));
  EXPECT_FALSE(analysis.findCached(sum));
  EXPECT_FALSE(analysis.findCached(f.getArgument(1)));
  auto refined = analysis.analyze(sum);
  ASSERT_TRUE(succeeded(refined));
  EXPECT_TRUE(refined->offsets.isUniform());
  ASSERT_TRUE(succeeded(analysis.reset()));
  EXPECT_FALSE(analysis.findCached(x));
  EXPECT_FALSE(analysis.findCached(sum));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, FailedMaterializationPreservesSourceAndPeers) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @placement(%x: i32, %y: i32) {
      %sum = arith.addi %y, %y : i32
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto sum = *f.getBody().front().getOps<arith::AddIOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(f.getArgument(0)))));
  auto peer = *analysis.findCached(f.getArgument(0));
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(sum))));
  auto *source = test::getOffsets(analysis.findCached(sum));
  ASSERT_TRUE(source);
  EXPECT_EQ(source->valueType, b.getI32Type());
  EXPECT_EQ(cast<Value>(source->fullOffsets), sum.getResult());
  // A failed query before the source definition must leave its cache intact.
  b.setInsertionPoint(sum);
  auto point = b.getInsertionPoint();
  unsigned before = countOperations(*module);
  for (unsigned i = 0; i < 3; ++i) {
    EXPECT_TRUE(failed(materializeOffsetAt(*source, {}, b, f.getLoc())));
    EXPECT_TRUE(succeeded(test::getOffsets(analysis.analyze(sum))));
    EXPECT_EQ(test::getOffsets(analysis.findCached(sum)), source);
    ASSERT_TRUE(analysis.findCached(f.getArgument(0)));
    EXPECT_EQ(analysis.findCached(f.getArgument(0))->offsets.fullOffsets,
              peer.offsets.fullOffsets);
    EXPECT_EQ(b.getInsertionPoint(), point);
  }
  EXPECT_EQ(countOperations(*module), before);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, OpaqueLeavesDoNotAnalyzeUnneededOperands) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @opaque(%base: !tt.ptr<i32>, %other: !tt.ptr<i32>, %x: i32, %y: i32, %cond: i1) {
      %shift = arith.shli %x, %y : i32
      %loaded = tt.load %base : !tt.ptr<i32>
      %selected = arith.select %cond, %base, %other : !tt.ptr<i32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto shift = *f.getBody().front().getOps<arith::ShLIOp>().begin();
  auto load = *f.getBody().front().getOps<triton::LoadOp>().begin();
  auto select = *f.getBody().front().getOps<arith::SelectOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  unsigned before = countOperations(*module);
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(shift))));
  ASSERT_TRUE(succeeded(test::getOffsets(analysis.analyze(load))));
  auto ptr = test::getInfo(analysis.analyze(select));
  ASSERT_TRUE(succeeded(ptr));
  EXPECT_EQ(ptr->base, select.getResult());
  for (Value arg : f.getArguments()) {
    EXPECT_FALSE(test::getOffsets(analysis.findCached(arg)));
    EXPECT_FALSE(test::getInfo(analysis.findCached(arg)));
  }
  EXPECT_GT(countOperations(*module), before);
  unsigned after = countOperations(*module);
  ASSERT_TRUE(succeeded(analysis.analyze(shift)));
  ASSERT_TRUE(succeeded(analysis.analyze(load)));
  EXPECT_EQ(countOperations(*module), after);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, SiblingPointerConsumersUseOuterIndexHelpers) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      scf::SCFDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @placement(%base: !tt.ptr<i32>, %x: i32, %cond: i1) {
      %values = tt.splat %x : i32 -> tensor<4xi32>
      %three = arith.constant dense<3> : tensor<4xi32>
      %offset = arith.remsi %values, %three : tensor<4xi32>
      %bases = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      scf.if %cond {
        %left = tt.addptr %bases, %offset : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      } else {
        %right = tt.addptr %bases, %offset : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      }
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<triton::AddPtrOp> pointers;
  f.walk([&](triton::AddPtrOp op) { pointers.push_back(op); });
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  Value sharedOrigin;
  for (auto ptr : pointers) {
    b.setInsertionPoint(ptr);
    auto point = b.getInsertionPoint();
    auto info = test::getInfo(analysis.analyze(ptr));
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->offsets.isUniform());
    Value origin = cast<Value>(info->offsets.scalarOffset);
    EXPECT_EQ(origin.getParentBlock(), &f.getBody().front());
    EXPECT_TRUE(origin.getType().isIndex());
    if (sharedOrigin)
      EXPECT_EQ(origin, sharedOrigin);
    sharedOrigin = origin;
    EXPECT_EQ(b.getInsertionPoint(), point);
    // A conversion consumes the descriptor before replacing this pointer op.
    Value zero = b.create<arith::ConstantIndexOp>(ptr.getLoc(), 0);
    b.create<arith::AddIOp>(ptr.getLoc(), origin, zero);
    auto lane = materializeOffsetAt(info->offsets, {zero}, b, ptr.getLoc());
    ASSERT_TRUE(succeeded(lane));
    EXPECT_TRUE(cast<Value>(*lane).getType().isIndex());
    unsigned count = countOperations(*module);
    ASSERT_TRUE(succeeded(analysis.analyze(ptr)));
    EXPECT_EQ(countOperations(*module), count);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisLifecycle, BoundaryMaterializationChecksOnlyItsActualInputs) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @boundary(%x:index) {
      %source=arith.addi %x,%x:index
      %late=arith.addi %source,%x:index
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<Value> sums;
  f.walk([&](arith::AddIOp op) { sums.push_back(op); });
  OpBuilder b(sums[0].getDefiningOp());
  auto initial = b.saveInsertionPoint();
  OffsetInfo info;
  info.valueType = b.getIndexType();
  info.scalarOffset = sums[1];
  unsigned count = countOperations(*module);
  EXPECT_TRUE(failed(materializeOffsetAt(info, {}, b, f.getLoc())));
  EXPECT_EQ(b.getInsertionPoint(), initial.getPoint());
  EXPECT_EQ(countOperations(*module), count);
  // An unused descriptor field does not block the exact-value route.
  info.fullOffsets = f.getArgument(0);
  EXPECT_TRUE(succeeded(materializeOffsetAt(info, {}, b, f.getLoc())));
  b.setInsertionPointAfterValue(sums[1]);
  info.fullOffsets = {};
  EXPECT_TRUE(succeeded(materializeOffsetAt(info, {}, b, f.getLoc())));
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
