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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;

namespace {
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, triton::TritonDialect,
                      scf::SCFDialect, tensor::TensorDialect>();
}
Operation *tagged(Operation *scope, StringRef name) {
  Operation *result = nullptr;
  scope->walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<StringAttr>("tag");
        attr && attr.getValue() == name)
      result = op;
  });
  return result;
}
void expectPreserved(const ChangeInfo &info) {
  EXPECT_FALSE(info.base);
  EXPECT_FALSE(info.uniform);
  EXPECT_FALSE(info.residual);
  EXPECT_FALSE(info.order);
  for (ArrayRef<bool> fields :
       {ArrayRef<bool>(info.strides), ArrayRef<bool>(info.parentShape),
        ArrayRef<bool>(info.coordinates)})
    for (bool field : fields)
      EXPECT_FALSE(field);
}

TEST(PtrAnalysisChanges, TargetInitialAndOtherSlotsHaveDifferentMeanings) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @loop(%base: !tt.ptr<i32>, %n: index, %step: i32, %cond: i1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant dense<0> : tensor<4xi32>
      %delta = tt.splat %step : i32 -> tensor<4xi32>
      %init = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %out:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %init, %q = %init)
          -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
        %same = tt.addptr %p, %zero {tag="same"} : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        %next = tt.addptr %p, %delta {tag="next"} : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        %restore = scf.if %cond -> (tensor<4x!tt.ptr<i32>>) {
          scf.yield %p : tensor<4x!tt.ptr<i32>>
        } else {
          scf.yield %init : tensor<4x!tt.ptr<i32>>
        } {tag="restore"}
        scf.yield %restore, %next : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  Value p = loop.getRegionIterArgs()[0], q = loop.getRegionIterArgs()[1];
  Value init = loop.getInitArgs()[0];
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  for (Value value :
       SmallVector<Value>{p, init, tagged(*module, "same")->getResult(0),
                          tagged(*module, "restore")->getResult(0)}) {
    auto changes = analysis.analyzeChanges(value, p, init);
    ASSERT_TRUE(succeeded(changes));
    expectPreserved(*changes);
  }
  auto next =
      analysis.analyzeChanges(tagged(*module, "next")->getResult(0), p, init);
  ASSERT_TRUE(succeeded(next));
  EXPECT_TRUE(next->uniform);
  EXPECT_FALSE(next->base);
  EXPECT_FALSE(next->residual);
  EXPECT_EQ(next->strides, SmallVector<bool>({false}));
  auto other = analysis.analyzeChanges(q, p, init);
  ASSERT_TRUE(succeeded(other));
  EXPECT_TRUE(other->uniform);
  EXPECT_FALSE(other->base);
  EXPECT_FALSE(other->residual);
  EXPECT_EQ(other->strides, SmallVector<bool>({false}));
  // A new target starts a new memo, even when the two init values are equal.
  auto self = analysis.analyzeChanges(q, q, init);
  ASSERT_TRUE(succeeded(self));
  expectPreserved(*self);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisChanges, OtherSlotsRequireTheWholeLoopFixedPoint) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @slots(%base: !tt.ptr<i32>, %other: !tt.ptr<i32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %delta = arith.constant dense<16> : tensor<4xi32>
      %a = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %b = tt.splat %other : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %first = tt.addptr %a, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %second = tt.addptr %first, %delta : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %swapped:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %first, %q = %second)
          -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
        %inner = scf.for %j = %c0 to %n step %c1 iter_args(%r = %q)
            -> (tensor<4x!tt.ptr<i32>>) {
          scf.yield %r : tensor<4x!tt.ptr<i32>>
        }
        scf.yield %inner, %p : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } {tag="swap"}
      %delayed:3 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %a, %q = %a, %r = %b)
          -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
        scf.yield %q, %r, %r : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } {tag="delayed"}
      %moving:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %first, %q = %first)
          -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
        %next = tt.addptr %p, %delta : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        scf.yield %next, %q : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } {tag="moving"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  auto swap = cast<scf::ForOp>(tagged(*module, "swap"));
  for (unsigned target : {0u, 1u}) {
    auto changes = analysis.analyzeChanges(swap.getRegionIterArgs()[1 - target],
                                           swap.getRegionIterArgs()[target],
                                           swap.getInitArgs()[target]);
    ASSERT_TRUE(succeeded(changes));
    EXPECT_FALSE(changes->base);
    EXPECT_FALSE(changes->residual);
    EXPECT_EQ(changes->strides, SmallVector<bool>({false}));
    EXPECT_TRUE(changes->uniform);
  }
  auto delayed = cast<scf::ForOp>(tagged(*module, "delayed"));
  auto propagated = analysis.analyzeChanges(delayed.getRegionIterArgs()[1],
                                            delayed.getRegionIterArgs()[0],
                                            delayed.getInitArgs()[0]);
  ASSERT_TRUE(succeeded(propagated));
  // Equal initial bases do not survive the value coming from the third slot.
  EXPECT_TRUE(propagated->base);
  EXPECT_EQ(propagated->strides, SmallVector<bool>({false}));
  auto moving = cast<scf::ForOp>(tagged(*module, "moving"));
  auto fixed = analysis.analyzeChanges(moving.getRegionIterArgs()[1],
                                       moving.getRegionIterArgs()[0],
                                       moving.getInitArgs()[0]);
  ASSERT_TRUE(succeeded(fixed));
  // The second slot stays at its initial offset, but the target has moved.
  EXPECT_TRUE(fixed->uniform);
  EXPECT_FALSE(fixed->base);
  EXPECT_FALSE(fixed->residual);
  EXPECT_EQ(fixed->strides, SmallVector<bool>({false}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisChanges, SwappedBlockPointersKeepOnlyInvariantFields) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @blocks(%base: !tt.ptr<i32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %shape = arith.constant 32 : i64
      %stride = arith.constant 1 : i64
      %zero = arith.constant 0 : i32
      %four = arith.constant 4 : i32
      %a = tt.make_tensor_ptr %base, [%shape], [%stride], [%zero]
          {order=array<i32: 0>} : !tt.ptr<tensor<4xi32>>
      %b = tt.make_tensor_ptr %base, [%shape], [%stride], [%four]
          {order=array<i32: 0>} : !tt.ptr<tensor<4xi32>>
      %out:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %a, %q = %b)
          -> (!tt.ptr<tensor<4xi32>>, !tt.ptr<tensor<4xi32>>) {
        scf.yield %q, %p : !tt.ptr<tensor<4xi32>>, !tt.ptr<tensor<4xi32>>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  auto changes = analysis.analyzeChanges(loop.getRegionIterArgs()[1],
                                         loop.getRegionIterArgs()[0],
                                         loop.getInitArgs()[0]);
  ASSERT_TRUE(succeeded(changes));
  EXPECT_FALSE(changes->base);
  EXPECT_FALSE(changes->order);
  EXPECT_EQ(changes->parentShape, SmallVector<bool>({false}));
  EXPECT_EQ(changes->strides, SmallVector<bool>({false}));
  EXPECT_EQ(changes->coordinates, SmallVector<bool>({true}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisChanges, UnknownInitialAxesKeepZeroIncrementProofs) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @unknown(%base: !tt.ptr<i32>, %offsets: tensor<4xi32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant dense<0> : tensor<4xi32>
      %base_tensor = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %init = tt.addptr %base_tensor, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %same = tt.addptr %init, %zero : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %out:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %init, %q = %same)
          -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
        scf.yield %q, %p : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } {tag="loop"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto loop = cast<scf::ForOp>(tagged(*module, "loop"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  auto info = analysis.analyze(loop.getInitArgs()[0]);
  ASSERT_TRUE(succeeded(info));
  ASSERT_EQ(info->offsets.axes[0].kind, AxisKind::Unknown);
  auto changes = analysis.analyzeChanges(loop.getRegionIterArgs()[1],
                                         loop.getRegionIterArgs()[0],
                                         loop.getInitArgs()[0]);
  ASSERT_TRUE(succeeded(changes));
  expectPreserved(*changes);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisChanges, WhileArgumentsUseConditionAndBackedges) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @edges(%base: !tt.ptr<i32>, %other: !tt.ptr<i32>, %cond: i1) {
      %delta = arith.constant dense<4> : tensor<4xi32>
      %a = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %b = tt.splat %other : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %out:3 = scf.while (%p = %a, %q = %a)
          : (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>)
          -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
        %next = tt.addptr %q, %delta : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        scf.condition(%cond) %q, %p, %next : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } do {
      ^bb0(%left: tensor<4x!tt.ptr<i32>>, %right: tensor<4x!tt.ptr<i32>>, %shifted: tensor<4x!tt.ptr<i32>>):
        scf.yield %right, %shifted : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } attributes {tag="while"}
      %bad:2 = scf.while (%p = %a) : (tensor<4x!tt.ptr<i32>>)
          -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
        scf.condition(%cond) %p, %b : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } do {
      ^bb0(%left: tensor<4x!tt.ptr<i32>>, %right: tensor<4x!tt.ptr<i32>>):
        scf.yield %right : tensor<4x!tt.ptr<i32>>
      } attributes {tag="bad"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  auto loop = cast<scf::WhileOp>(tagged(*module, "while"));
  for (auto args : {loop.getBeforeArguments(), loop.getAfterArguments()}) {
    auto changes =
        analysis.analyzeChanges(args[0], args[1], loop.getInits()[0]);
    ASSERT_TRUE(succeeded(changes));
    EXPECT_FALSE(changes->base);
    EXPECT_FALSE(changes->residual);
    EXPECT_EQ(changes->strides, SmallVector<bool>({false}));
    EXPECT_TRUE(changes->uniform);
  }
  auto bad = cast<scf::WhileOp>(tagged(*module, "bad"));
  auto changes =
      analysis.analyzeChanges(bad.getAfterArguments()[1],
                              bad.getAfterArguments()[0], bad.getInits()[0]);
  ASSERT_TRUE(succeeded(changes));
  EXPECT_TRUE(changes->base);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisChanges, OpaqueCurrentBasesDoNotEraseCarriedStrideChanges) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @opaque(%init: tensor<4x!tt.ptr<i32>>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %range = tt.make_range {start=0:i32, end=4:i32} : tensor<4xi32>
      %out:3 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %init, %q = %init, %r = %init)
          -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
        %inner = scf.for %j = %c0 to %n step %c1 iter_args(%s = %q)
            -> (tensor<4x!tt.ptr<i32>>) {
          scf.yield %s : tensor<4x!tt.ptr<i32>>
        }
        %next = tt.addptr %q, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        scf.yield %inner, %next, %r : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
      } {tag="outer"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto loop = cast<scf::ForOp>(tagged(*module, "outer"));
  OpBuilder builder(loop);
  PtrAnalysis analysis(builder);
  auto initial = analysis.analyze(loop.getInitArgs()[0]);
  ASSERT_TRUE(succeeded(initial));
  ASSERT_FALSE(initial->hasScalarBase());
  ASSERT_TRUE(initial->offsets.isUniform());
  auto changes = analysis.analyzeChanges(loop.getRegionIterArgs()[0],
                                         loop.getRegionIterArgs()[2],
                                         loop.getInitArgs()[2]);
  ASSERT_TRUE(succeeded(changes));
  EXPECT_FALSE(changes->base);
  EXPECT_FALSE(changes->uniform);
  EXPECT_FALSE(changes->residual);
  EXPECT_EQ(changes->strides, SmallVector<bool>({true}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisChanges, LaneSelectionCanTurnUniformOffsetsIntoAResidual) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @select(%base: !tt.ptr<i32>, %mask: tensor<4xi1>, %cond: i1) {
      %one = arith.constant dense<1> : tensor<4xi32>
      %two = arith.constant dense<2> : tensor<4xi32>
      %init = tt.splat %base {tag="init"} : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %left = tt.addptr %init, %one : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %right = tt.addptr %init, %two : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %lanes = arith.select %mask, %left, %right {tag="lanes"} : tensor<4xi1>, tensor<4x!tt.ptr<i32>>
      %whole = arith.select %cond, %left, %right {tag="whole"} : tensor<4x!tt.ptr<i32>>
      %splat = tt.splat %cond : i1 -> tensor<4xi1>
      %uniform = arith.select %splat, %left, %right {tag="uniform"} : tensor<4xi1>, tensor<4x!tt.ptr<i32>>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  Value initial = tagged(*module, "init")->getResult(0);
  OpBuilder builder(tagged(*module, "init"));
  PtrAnalysis analysis(builder);
  auto lanes =
      analysis.analyzeChanges(tagged(*module, "lanes")->getResult(0), initial);
  ASSERT_TRUE(succeeded(lanes));
  EXPECT_FALSE(lanes->base);
  EXPECT_TRUE(lanes->uniform);
  EXPECT_TRUE(lanes->residual);
  for (auto name : {"whole", "uniform"}) {
    auto selected =
        analysis.analyzeChanges(tagged(*module, name)->getResult(0), initial);
    ASSERT_TRUE(succeeded(selected));
    EXPECT_FALSE(selected->base);
    EXPECT_TRUE(selected->uniform);
    EXPECT_FALSE(selected->residual);
    EXPECT_EQ(selected->strides, SmallVector<bool>({false}));
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisChanges, BlockFieldsCompareOperandsAndPreserveDescriptorBase) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @block(%base: !tt.ptr<i32>, %coord: i32, %step: i32) {
      %s0 = arith.constant 16 : i64
      %s1 = arith.constant 32 : i64
      %stride = arith.constant 1 : i64
      %zero = arith.constant 0 : i32
      %shift = arith.constant 3 : i32
      %shifted = tt.addptr %base, %shift : !tt.ptr<i32>, i32
      %init = tt.make_tensor_ptr %shifted, [%s0], [%stride], [%coord]
          {order=array<i32: 0>, tag="init"} : !tt.ptr<tensor<4xi32>>
      %shape = tt.make_tensor_ptr %shifted, [%s1], [%stride], [%coord]
          {order=array<i32: 0>, tag="shape"} : !tt.ptr<tensor<4xi32>>
      %same = tt.advance %init, [%zero] {tag="same"} : !tt.ptr<tensor<4xi32>>
      %next = tt.advance %init, [%step] {tag="next"} : !tt.ptr<tensor<4xi32>>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto init = cast<triton::MakeTensorPtrOp>(tagged(*module, "init"));
  OpBuilder builder(init);
  PtrAnalysis analysis(builder);
  auto info = analysis.analyze(init.getResult());
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->block->descriptorBase, init.getBase());
  EXPECT_NE(info->base, info->block->descriptorBase);
  auto shape =
      analysis.analyzeChanges(tagged(*module, "shape")->getResult(0), init);
  ASSERT_TRUE(succeeded(shape));
  EXPECT_EQ(shape->parentShape, SmallVector<bool>({true}));
  EXPECT_FALSE(shape->base);
  EXPECT_EQ(shape->strides, SmallVector<bool>({false}));
  EXPECT_EQ(shape->coordinates, SmallVector<bool>({false}));
  auto same =
      analysis.analyzeChanges(tagged(*module, "same")->getResult(0), init);
  ASSERT_TRUE(succeeded(same));
  expectPreserved(*same);
  auto next =
      analysis.analyzeChanges(tagged(*module, "next")->getResult(0), init);
  ASSERT_TRUE(succeeded(next));
  EXPECT_EQ(next->coordinates, SmallVector<bool>({true}));
  EXPECT_EQ(next->parentShape, SmallVector<bool>({false}));
  EXPECT_EQ(next->strides, SmallVector<bool>({false}));
  EXPECT_FALSE(next->base);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisChanges, NestedInitialAndWhileConditionEdgesParticipate) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @nested(%base: !tt.ptr<i32>, %n: index, %cond: i1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %one = arith.constant dense<1> : tensor<4xi32>
      %init = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %outer = scf.for %i = %c0 to %n step %c1 iter_args(%p = %init) -> (tensor<4x!tt.ptr<i32>>) {
        %shifted = tt.addptr %init, %one : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        %inner = scf.for %j = %c0 to %n step %c1 iter_args(%q = %shifted) -> (tensor<4x!tt.ptr<i32>>) {
          scf.yield %q : tensor<4x!tt.ptr<i32>>
        } {tag="inner"}
        %w:2 = scf.while (%before = %p) : (tensor<4x!tt.ptr<i32>>) -> (tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>) {
          %next = tt.addptr %before, %one : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
          scf.condition(%cond) %before, %next : tensor<4x!tt.ptr<i32>>, tensor<4x!tt.ptr<i32>>
        } do {
        ^bb0(%left: tensor<4x!tt.ptr<i32>>, %right: tensor<4x!tt.ptr<i32>>):
          scf.yield %left : tensor<4x!tt.ptr<i32>>
        } attributes {tag="while"}
        scf.yield %w#1 : tensor<4x!tt.ptr<i32>>
      } {tag="outer"}
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto outer = cast<scf::ForOp>(tagged(*module, "outer"));
  Value reference = outer.getRegionIterArgs()[0],
        initial = outer.getInitArgs()[0];
  OpBuilder builder(outer);
  PtrAnalysis analysis(builder);
  auto inner = analysis.analyzeChanges(tagged(*module, "inner")->getResult(0),
                                       reference, initial);
  ASSERT_TRUE(succeeded(inner));
  EXPECT_TRUE(
      inner->uniform); // its unchanged body does not erase its shifted init
  EXPECT_FALSE(inner->residual);
  auto loop = cast<scf::WhileOp>(tagged(*module, "while"));
  auto left = analysis.analyzeChanges(loop.getResult(0), reference, initial);
  auto right = analysis.analyzeChanges(loop.getResult(1), reference, initial);
  ASSERT_TRUE(succeeded(left));
  ASSERT_TRUE(succeeded(right));
  expectPreserved(*left);
  EXPECT_TRUE(right->uniform);
  EXPECT_FALSE(right->base);
  EXPECT_FALSE(right->residual);
  EXPECT_TRUE(succeeded(verify(*module)));
}
} // namespace
