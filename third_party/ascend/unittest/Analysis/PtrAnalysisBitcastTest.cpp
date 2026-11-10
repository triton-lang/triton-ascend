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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;
namespace {
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
}
Value tagged(Operation *scope, StringRef name) {
  Value result;
  scope->walk([&](Operation *op) {
    if (auto tag = op->getAttrOfType<StringAttr>("tag"))
      if (tag.getValue() == name)
        result = op->getResult(0);
  });
  return result;
}
unsigned count(Operation *scope) {
  unsigned n = 0;
  scope->walk([&](Operation *) { ++n; });
  return n;
}
unsigned elementBytes(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return cast<triton::PointerType>(type)
             .getPointeeType()
             .getIntOrFloatBitWidth() /
         8;
}
// Interpret original pointer operations in bytes, independently of PtrInfo.
// Each addptr uses its own input pointee unit; bitcast keeps the address.
std::optional<test::Lanes> byteAddresses(Value value,
                                         const test::Environment &env) {
  if (auto it = env.find(value); it != env.end())
    return it->second;
  if (auto op = value.getDefiningOp<triton::BitcastOp>())
    return byteAddresses(op.getSrc(), env);
  if (auto op = value.getDefiningOp<triton::ReshapeOp>())
    return byteAddresses(op.getSrc(), env);
  if (auto op = value.getDefiningOp<triton::SplatOp>()) {
    auto scalar = byteAddresses(op.getSrc(), env);
    if (!scalar || scalar->size() != 1)
      return std::nullopt;
    return test::Lanes(cast<RankedTensorType>(op.getType()).getNumElements(),
                       scalar->front());
  }
  if (auto op = value.getDefiningOp<triton::AddPtrOp>()) {
    auto base = byteAddresses(op.getPtr(), env);
    auto offsets = test::evaluate(op.getOffset(), env);
    if (!base || !offsets || base->size() != offsets->size())
      return std::nullopt;
    for (unsigned i = 0; i < base->size(); ++i)
      (*base)[i] +=
          (*offsets)[i].sextOrTrunc(64) * elementBytes(op.getPtr().getType());
    return base;
  }
  return std::nullopt;
}
void expectAddresses(Value source, const PtrInfo &info, OpBuilder &builder,
                     const test::Environment &env) {
  auto expected = byteAddresses(source, env);
  auto base = byteAddresses(info.base, env);
  ASSERT_TRUE(expected);
  ASSERT_TRUE(base);
  ASSERT_EQ(base->size(), 1u);
  auto point = builder.saveInsertionPoint();
  for (unsigned lane = 0; lane < expected->size(); ++lane) {
    SmallVector<Value> indices(info.offsets.getShape().size());
    unsigned remaining = lane;
    for (int axis = static_cast<int>(info.offsets.getShape().size()) - 1;
         axis >= 0; --axis) {
      indices[axis] = builder.create<arith::ConstantIndexOp>(
          source.getLoc(), remaining % info.offsets.getShape()[axis]);
      remaining /= info.offsets.getShape()[axis];
    }
    auto offset =
        materializeOffsetAt(info.offsets, indices, builder, source.getLoc());
    ASSERT_TRUE(succeeded(offset));
    auto evaluated = test::evaluate(*offset, env);
    ASSERT_TRUE(evaluated);
    EXPECT_EQ(base->front() +
                  evaluated->front() * elementBytes(source.getType()),
              (*expected)[lane]);
    EXPECT_EQ(builder.getInsertionPoint(), point.getPoint());
  }
}

TEST(PtrAnalysisBitcast,
     SameWidthChainPreservesByteAddressesAndHelperOwnership) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @chain(%base:!tt.ptr<f32,3>,%x:tensor<4xi32>,%y:tensor<4xi32>) {
      %p=tt.splat %base:!tt.ptr<f32,3> -> tensor<4x!tt.ptr<f32,3>>
      %first=tt.addptr %p,%x:tensor<4x!tt.ptr<f32,3>>,tensor<4xi32>
      %cast=tt.bitcast %first {tag="cast"}:tensor<4x!tt.ptr<f32,3>> -> tensor<4x!tt.ptr<i32,3>>
      %next=tt.addptr %cast,%y {tag="next"}:tensor<4x!tt.ptr<i32,3>>,tensor<4xi32>
      %back=tt.bitcast %next {tag="back"}:tensor<4x!tt.ptr<i32,3>> -> tensor<4x!tt.ptr<f32,3>>
      %shape="tt.reshape"(%back) {tag="shape"}:(tensor<4x!tt.ptr<f32,3>>) -> tensor<2x2x!tt.ptr<f32,3>>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<Operation *> originalCasts;
  f.walk([&](triton::BitcastOp op) { originalCasts.push_back(op); });
  test::Environment env{{f.getArgument(0), {llvm::APInt(64, 0x10000000000ULL)}},
                        {f.getArgument(1),
                         {llvm::APInt(32, INT32_MAX), llvm::APInt(32, -7, true),
                          llvm::APInt(32, -128, true), llvm::APInt(32, 123)}},
                        {f.getArgument(2),
                         {llvm::APInt(32, 1), llvm::APInt(32, -2, true),
                          llvm::APInt(32, 4), llvm::APInt(32, 5)}}};
  OpBuilder builder(f.getBody().front().getTerminator());
  auto check = [&](auto &analysis) {
    auto caller = builder.saveInsertionPoint();
    for (StringRef name : {"cast", "next", "back", "shape"}) {
      Value source = tagged(f, name);
      auto result = test::getInfo(analysis.analyze(source));
      ASSERT_TRUE(succeeded(result));
      EXPECT_EQ(builder.getInsertionPoint(), caller.getPoint());
      EXPECT_EQ(result->getAddressSpace(), 3u);
      Type element =
          cast<triton::PointerType>(
              cast<RankedTensorType>(source.getType()).getElementType())
              .getPointeeType();
      EXPECT_EQ(result->getElementType(), element);
      EXPECT_EQ(result->base.getType(), triton::PointerType::get(element, 3));
      EXPECT_TRUE(result->offsets.fullOffsets);
      unsigned n = count(*module);
      ASSERT_TRUE(succeeded(analysis.analyze(source)));
      EXPECT_EQ(count(*module), n);
      expectAddresses(source, *result, builder, env);
    }
    ASSERT_TRUE(succeeded(analysis.reset()));
  };
  PtrAnalysis analysis(builder);
  check(analysis);
  SmallVector<Operation *> helpers;
  f.walk([&](triton::BitcastOp op) {
    if (!llvm::is_contained(originalCasts, op.getOperation())) {
      EXPECT_TRUE(isa<triton::PointerType>(op.getType()));
      helpers.push_back(op);
    }
  });
  ASSERT_EQ(helpers.size(), 2u);
  f.walk([&](Operation *op) {
    for (Type type : op->getResultTypes())
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        EXPECT_FALSE(tensor.getElementType().isIndex());
  });
  EXPECT_TRUE(succeeded(verify(*module)));
  IRRewriter rewriter(&context);
  test::eraseDeadPtrArithmetic(f, rewriter, helpers);
  SmallVector<Operation *> remaining;
  f.walk([&](triton::BitcastOp op) { remaining.push_back(op); });
  EXPECT_EQ(remaining, originalCasts);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisBitcast, EncodingChangesCannotReuseSourceOffsets) {
  MLIRContext context;
  load(context);
  context.loadDialect<triton::gpu::TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #layout = #ttg.blocked<{sizePerThread=[1],threadsPerWarp=[32],warpsPerCTA=[4],order=[0]}>
    module attributes {"ttg.num-warps"=4:i32, "ttg.threads-per-warp"=32:i32, "ttg.num-ctas"=1:i32} {
      tt.func @remove_encoding(%base:!tt.ptr<i32>) {
        %p=tt.splat %base:!tt.ptr<i32> -> tensor<4x!tt.ptr<i32>,#layout>
        %c=tt.bitcast %p:tensor<4x!tt.ptr<i32>,#layout> -> tensor<4x!tt.ptr<f32>>
        tt.return
      }
      tt.func @add_encoding(%base:!tt.ptr<i32>) {
        %p=tt.splat %base:!tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
        %c=tt.bitcast %p:tensor<4x!tt.ptr<i32>> -> tensor<4x!tt.ptr<f32>,#layout>
        tt.return
      }
    })mlir",
                                            &context);
  ASSERT_TRUE(module);
  // The verifier permits one missing encoding; offset reuse needs equality.
  ASSERT_TRUE(succeeded(verify(*module)));
  for (auto f : module->getOps<triton::FuncOp>()) {
    auto cast = *f.getBody().front().getOps<triton::BitcastOp>().begin();
    OpBuilder builder(f.getBody().front().getTerminator());
    PtrAnalysis analysis(builder);
    ASSERT_TRUE(succeeded(analysis.analyze(cast.getSrc())));
    unsigned n = count(f);
    EXPECT_TRUE(failed(analysis.analyze(cast.getResult())));
    EXPECT_EQ(count(f), n);
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisBitcast, DynamicShapeKeepsSameWidthButRejectsRescaling) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @dynamic(%base8:!tt.ptr<i8>, %base32:!tt.ptr<i32>) {
      %r=tt.make_range {start=0:i32,end=4:i32}:tensor<4xi32>
      %four=arith.constant dense<4>:tensor<4xi32>
      %offsets=arith.muli %r,%four:tensor<4xi32>
      %p8=tt.splat %base8:!tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
      %a8=tt.addptr %p8,%offsets:tensor<4x!tt.ptr<i8>>,tensor<4xi32>
      %d8=tensor.cast %a8:tensor<4x!tt.ptr<i8>> to tensor<?x!tt.ptr<i8>>
      %widen=tt.bitcast %d8 {tag="widen"}:tensor<?x!tt.ptr<i8>> -> tensor<?x!tt.ptr<i32>>
      %p32=tt.splat %base32:!tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %a32=tt.addptr %p32,%offsets:tensor<4x!tt.ptr<i32>>,tensor<4xi32>
      %d32=tensor.cast %a32:tensor<4x!tt.ptr<i32>> to tensor<?x!tt.ptr<i32>>
      %narrow=tt.bitcast %d32 {tag="narrow"}:tensor<?x!tt.ptr<i32>> -> tensor<?x!tt.ptr<i8>>
      %same=tt.bitcast %d32 {tag="same"}:tensor<?x!tt.ptr<i32>> -> tensor<?x!tt.ptr<f32>>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(f.getBody().front().getTerminator());
  PtrAnalysis analysis(builder);

  // Both sources are analyzable, with aligned offsets [0, 4, 8, 12].
  for (StringRef name : {"widen", "narrow"}) {
    auto op = tagged(f, name).getDefiningOp<triton::BitcastOp>();
    auto source = analysis.analyze(op.getSrc());
    ASSERT_TRUE(succeeded(source));
    ASSERT_TRUE(
        mlir::triton::ptr::detail::canDivideDecomposition(source->offsets, 4));
    unsigned n = count(f);
    EXPECT_TRUE(failed(analysis.analyze(op.getResult())));
    EXPECT_EQ(count(f), n);
  }

  // Equal-width casts can retain the dynamic offsets without a factor tensor.
  auto op = tagged(f, "same").getDefiningOp<triton::BitcastOp>();
  auto source = analysis.analyze(op.getSrc());
  auto result = analysis.analyze(op.getResult());
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->offsets.fullOffsets, source->offsets.fullOffsets);
  EXPECT_TRUE(result->getElementType().isF32());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisBitcast, StructuredHalfWidthCastPreservesSubAxes) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @partition(%base:!tt.ptr<f16,3>) {
      %r=tt.make_range {start=0:i32,end=8:i32}:tensor<8xi32>
      %four=arith.constant dense<4>:tensor<8xi32>
      %rem=arith.remsi %r,%four:tensor<8xi32>
      %p=tt.splat %base:!tt.ptr<f16,3> -> tensor<8x!tt.ptr<f16,3>>
      %before=tt.addptr %p,%rem {tag="before"}:tensor<8x!tt.ptr<f16,3>>,tensor<8xi32>
      %cast=tt.bitcast %before {tag="cast"}:tensor<8x!tt.ptr<f16,3>> -> tensor<8x!tt.ptr<bf16,3>>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(f.getBody().front().getTerminator());
  PtrAnalysis analysis(builder);
  auto before = test::getInfo(analysis.analyze(tagged(f, "before")));
  auto after = test::getInfo(analysis.analyze(tagged(f, "cast")));
  ASSERT_TRUE(succeeded(before));
  ASSERT_TRUE(succeeded(after));
  EXPECT_EQ(before->offsets.fullOffsets, after->offsets.fullOffsets);
  ASSERT_FALSE(before->offsets.axes[0].subAxes.empty());
  ASSERT_FALSE(after->offsets.axes[0].subAxes.empty());
  ASSERT_EQ(before->offsets.axes[0].subAxes.size(),
            after->offsets.axes[0].subAxes.size());
  for (auto [left, right] : llvm::zip(before->offsets.axes[0].subAxes,
                                      after->offsets.axes[0].subAxes)) {
    EXPECT_EQ(left.shape, right.shape);
    EXPECT_EQ(left.stride, right.stride);
  }
  EXPECT_TRUE(after->getElementType().isBF16());
  EXPECT_EQ(after->getAddressSpace(), 3u);
  expectAddresses(tagged(f, "cast"), *after, builder,
                  {{f.getArgument(0), {llvm::APInt(64, 4096)}}});
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisBitcast, DifferentWidthScalarStartsANewBase) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @scalar(%base:!tt.ptr<f32,3>,%shift:i32,%x:tensor<4xi32>) {
      %before=tt.addptr %base,%shift:!tt.ptr<f32,3>,i32
      %root=tt.bitcast %before {tag="root"}:!tt.ptr<f32,3> -> !tt.ptr<i8,3>
      %p=tt.splat %root:!tt.ptr<i8,3> -> tensor<4x!tt.ptr<i8,3>>
      %after=tt.addptr %p,%x {tag="after"}:tensor<4x!tt.ptr<i8,3>>,tensor<4xi32>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(f.getBody().front().getTerminator());
  PtrAnalysis analysis(builder);
  unsigned n = count(*module);
  auto root = test::getInfo(analysis.analyze(tagged(f, "root")));
  ASSERT_TRUE(succeeded(root));
  EXPECT_EQ(count(*module), n);
  EXPECT_EQ(root->base, tagged(f, "root"));
  EXPECT_EQ(test::evaluate(root->offsets.fullOffsets)->front(),
            llvm::APInt(64, 0));
  auto after = test::getInfo(analysis.analyze(tagged(f, "after")));
  ASSERT_TRUE(succeeded(after));
  EXPECT_EQ(after->base, tagged(f, "root"));
  test::Environment env{{f.getArgument(0), {llvm::APInt(64, 4096)}},
                        {f.getArgument(1), {llvm::APInt(32, -3, true)}},
                        {f.getArgument(2),
                         {llvm::APInt(32, -5, true), llvm::APInt(32, 0),
                          llvm::APInt(32, 1), llvm::APInt(32, 7)}}};
  expectAddresses(tagged(f, "after"), *after, builder, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisBitcast, DifferentWidthsPreserveCombinedByteAddresses) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @units(%base:!tt.ptr<i8>, %pid:i32) {
      %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
      %four = arith.constant dense<4> : tensor<8xi32>
      %scaled = arith.muli %r,%four : tensor<8xi32>
      %one = arith.constant dense<1> : tensor<8xi32>
      %three = arith.constant dense<3> : tensor<8xi32>
      %rest = arith.addi %scaled,%three : tensor<8xi32>
      %p = tt.splat %base : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
      %p1 = tt.addptr %p,%one : tensor<8x!tt.ptr<i8>>,tensor<8xi32>
      %p4 = tt.addptr %p1,%rest : tensor<8x!tt.ptr<i8>>,tensor<8xi32>
      %wide = tt.bitcast %p4 {tag="wide"} : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<f32>>
      %advanced = tt.addptr %wide,%r {tag="advanced"} : tensor<8x!tt.ptr<f32>>,tensor<8xi32>
      %half = tt.bitcast %advanced {tag="half"} : tensor<8x!tt.ptr<f32>> -> tensor<8x!tt.ptr<i16>>
      %byte = tt.bitcast %half {tag="byte"} : tensor<8x!tt.ptr<i16>> -> tensor<8x!tt.ptr<i8>>
      %sixteen = arith.constant 16 : i32
      %start = arith.muli %pid,%sixteen : i32
      %s = tt.splat %start : i32 -> tensor<8xi32>
      %d = arith.addi %s,%scaled : tensor<8xi32>
      %dp = tt.addptr %p,%d : tensor<8x!tt.ptr<i8>>,tensor<8xi32>
      %dynamic = tt.bitcast %dp {tag="dynamic"} : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<f32>>
      %rem = arith.remsi %r,%four : tensor<8xi32>
      %rs = arith.muli %rem,%four : tensor<8xi32>
      %rp = tt.addptr %p,%rs : tensor<8x!tt.ptr<i8>>,tensor<8xi32>
      %subaxis = tt.bitcast %rp {tag="subaxis"} : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<f32>>
      %max = arith.constant dense<127> : tensor<8xi8>
      %inc = arith.constant dense<1> : tensor<8xi8>
      %a = tt.addptr %p,%max : tensor<8x!tt.ptr<i8>>,tensor<8xi8>
      %sum = tt.addptr %a,%inc : tensor<8x!tt.ptr<i8>>,tensor<8xi8>
      %separate = tt.bitcast %sum {tag="separate"} : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<f32>>
      %wrapped = arith.addi %max,%inc : tensor<8xi8>
      %wp = tt.addptr %p,%wrapped : tensor<8x!tt.ptr<i8>>,tensor<8xi8>
      %negative = tt.bitcast %wp {tag="wrapped"} : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<f32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(f.getBody().front().getTerminator());
  PtrAnalysis analysis(builder);
  for (StringRef tag : {"wide", "advanced", "half", "byte", "dynamic",
                        "subaxis", "separate", "wrapped"}) {
    Value value = tagged(f, tag);
    auto info = test::getInfo(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info)) << tag.str();
    EXPECT_TRUE(
        succeeded(mlir::triton::ptr::detail::validateInfo(value, *info)));
    EXPECT_EQ(info->offsets.getShape(), ArrayRef<int64_t>({8}));
    for (int64_t pid : {-2, 0, 7})
      expectAddresses(value, *info, builder,
                      {{f.getArgument(0), {llvm::APInt(64, 4096)}},
                       {f.getArgument(1), {llvm::APInt(32, pid, true)}}});
    if (tag == "subaxis") {
      ASSERT_FALSE(info->offsets.axes[0].subAxes.empty());
      EXPECT_TRUE(info->offsets.hasStructuredDecomposition());
    } else if (tag != "wrapped") {
      EXPECT_TRUE(info->offsets.hasStructuredForm());
    }
    if (tag == "separate" || tag == "wrapped") {
      auto offsets = test::evaluate(info->offsets);
      ASSERT_TRUE(offsets);
      EXPECT_EQ(offsets->front().getSExtValue(), tag == "separate" ? 32 : -32);
    }
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisBitcast, WideningKeepsProvenColumnsAndRejectsUnknownUnits) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @rows(%base:!tt.ptr<i8>, %rows:tensor<2xi32>) {
      %four2 = arith.constant dense<4> : tensor<2xi32>
      %bytes = arith.muli %rows,%four2 : tensor<2xi32>
      %e = tt.expand_dims %bytes {axis=1:i32} : tensor<2xi32> -> tensor<2x1xi32>
      %b = tt.broadcast %e : tensor<2x1xi32> -> tensor<2x4xi32>
      %r = tt.make_range {start=0:i32,end=4:i32} : tensor<4xi32>
      %four4 = arith.constant dense<4> : tensor<4xi32>
      %col = arith.muli %r,%four4 : tensor<4xi32>
      %ce = tt.expand_dims %col {axis=0:i32} : tensor<4xi32> -> tensor<1x4xi32>
      %cb = tt.broadcast %ce : tensor<1x4xi32> -> tensor<2x4xi32>
      %offset = arith.addi %b,%cb : tensor<2x4xi32>
      %p = tt.splat %base : !tt.ptr<i8> -> tensor<2x4x!tt.ptr<i8>>
      %src = tt.addptr %p,%offset : tensor<2x4x!tt.ptr<i8>>,tensor<2x4xi32>
      %wide = tt.bitcast %src {tag="wide"} : tensor<2x4x!tt.ptr<i8>> -> tensor<2x4x!tt.ptr<f32>>
      %back = tt.bitcast %wide {tag="back"} : tensor<2x4x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<i8>>
      %ue = tt.expand_dims %rows {axis=1:i32} : tensor<2xi32> -> tensor<2x1xi32>
      %ub = tt.broadcast %ue : tensor<2x1xi32> -> tensor<2x4xi32>
      %uo = arith.addi %ub,%cb : tensor<2x4xi32>
      %unknown = tt.addptr %p,%uo {tag="source"} : tensor<2x4x!tt.ptr<i8>>,tensor<2x4xi32>
      %rejected = tt.bitcast %unknown {tag="rejected"} : tensor<2x4x!tt.ptr<i8>> -> tensor<2x4x!tt.ptr<f32>>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(f.getBody().front().getTerminator());
  PtrAnalysis analysis(builder);
  test::Environment env{
      {f.getArgument(0), {llvm::APInt(64, 4096)}},
      {f.getArgument(1), {llvm::APInt(32, -7, true), llvm::APInt(32, 19)}}};
  for (StringRef tag : {"wide", "back"}) {
    auto info = test::getInfo(analysis.analyze(tagged(f, tag)));
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(test::axisKinds(info->offsets),
              (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Structured}));
    EXPECT_EQ(
        test::evaluate(info->offsets.axes[1].stride)->front().getSExtValue(),
        tag == "wide" ? 1 : 4);
    expectAddresses(tagged(f, tag), *info, builder, env);
  }
  ASSERT_TRUE(succeeded(test::getInfo(analysis.analyze(tagged(f, "source")))));
  unsigned before = count(f);
  auto rejected = analysis.analyze(tagged(f, "rejected"));
  EXPECT_TRUE(failed(rejected));
  EXPECT_EQ(count(f), before);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisBitcast,
     UnsupportedTensorUnitsAndNumericProofsStayUnavailable) {
  MLIRContext context;
  load(context);
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto ptr = [&](Type pointee, unsigned space = 3) {
    return triton::PointerType::get(pointee, space);
  };
  SmallVector<std::pair<Type, Type>> types{
      {ptr(builder.getI8Type()), ptr(builder.getF32Type())},
      {ptr(builder.getI1Type()), ptr(builder.getI1Type())},
      {ptr(builder.getIntegerType(24)), ptr(builder.getIntegerType(24))},
      {ptr(ptr(builder.getI8Type())), ptr(ptr(builder.getI8Type()))}};
  for (auto [from, to] : types) {
    auto source = RankedTensorType::get({4}, from);
    auto target = RankedTensorType::get({4}, to);
    auto fn = builder.create<triton::FuncOp>(
        loc, "unsupported", builder.getFunctionType({from}, {}));
    Block *block = fn.addEntryBlock();
    OpBuilder body(block, block->begin());
    Value splat =
        body.create<triton::SplatOp>(loc, source, block->getArgument(0));
    auto offsetType = RankedTensorType::get({4}, body.getI32Type());
    Value offset = body.create<arith::ConstantOp>(
        loc, offsetType, DenseIntElementsAttr::get(offsetType, int32_t(3)));
    Value pointers = body.create<triton::AddPtrOp>(loc, source, splat, offset);
    auto cast = body.create<triton::BitcastOp>(loc, target, pointers);
    body.create<triton::ReturnOp>(loc);
    PtrAnalysis analysis(body);
    // Reject the cast's units, not an opaque input that analysis cannot visit.
    ASSERT_TRUE(succeeded(test::getInfo(analysis.analyze(pointers))));
    unsigned n = count(fn);
    auto result = analysis.analyze(cast);
    EXPECT_TRUE(failed(result));
    EXPECT_EQ(count(fn), n);
    EXPECT_TRUE(succeeded(verify(fn)));
    fn.erase();
  }
  auto parsed = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @numeric(%x:tensor<4xf32>) {
      %y=tt.bitcast %x {tag="numeric"}:tensor<4xf32> -> tensor<4xi32>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(parsed);
  auto fn = *parsed->getOps<triton::FuncOp>().begin();
  OpBuilder body(fn.getBody().front().getTerminator());
  PtrAnalysis analysis(body);
  auto value = tagged(fn, "numeric");
  auto result = analysis.analyze(value);
  ASSERT_TRUE(succeeded(result));
  ASSERT_FALSE(result->isEmpty());
  ASSERT_FALSE(result->base);
  auto *integer = &*result;
  EXPECT_EQ(integer->offsets.fullOffsets, OpFoldResult(value));
  EXPECT_FALSE(integer->offsets.hasStructuredDecomposition());
  EXPECT_EQ(integer->offsets.axes[0].kind, AxisKind::Unknown);
  EXPECT_TRUE(succeeded(verify(*parsed)));
  module.erase();
}
} // namespace
