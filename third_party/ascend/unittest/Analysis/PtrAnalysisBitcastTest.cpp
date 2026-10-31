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
#include "TritonMemoryAccess/PtrAnalysisRules/StructuredPtrRules.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
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
    SmallVector<Value> indices(info.offsets.shape.size());
    unsigned remaining = lane;
    for (int axis = static_cast<int>(info.offsets.shape.size()) - 1; axis >= 0;
         --axis) {
      indices[axis] = builder.create<arith::ConstantIndexOp>(
          source.getLoc(), remaining % info.offsets.shape[axis]);
      remaining /= info.offsets.shape[axis];
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
  for (bool structured : {false, true}) {
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
    test::Environment env{
        {f.getArgument(0), {llvm::APInt(64, 0x10000000000ULL)}},
        {f.getArgument(1),
         {llvm::APInt(32, INT32_MAX), llvm::APInt(32, -7, true),
          llvm::APInt(32, -128, true), llvm::APInt(32, 123)}},
        {f.getArgument(2),
         {llvm::APInt(32, 1), llvm::APInt(32, -2, true), llvm::APInt(32, 4),
          llvm::APInt(32, 5)}}};
    OpBuilder builder(f.getBody().front().getTerminator());
    auto check = [&](auto &analysis) {
      auto caller = builder.saveInsertionPoint();
      for (StringRef name : {"cast", "next", "back", "shape"}) {
        Value source = tagged(f, name);
        auto result = test::getPtrInfo(analysis.analyze(source));
        ASSERT_TRUE(succeeded(result));
        EXPECT_EQ(builder.getInsertionPoint(), caller.getPoint());
        EXPECT_EQ(result->getAddressSpace(), 3u);
        Type element =
            cast<triton::PointerType>(
                cast<RankedTensorType>(source.getType()).getElementType())
                .getPointeeType();
        EXPECT_EQ(result->getElementType(), element);
        EXPECT_EQ(result->base.getType(), triton::PointerType::get(element, 3));
        EXPECT_FALSE(result->offsets.fullOffsets);
        unsigned n = count(*module);
        ASSERT_TRUE(succeeded(analysis.analyze(source)));
        EXPECT_EQ(count(*module), n);
        expectAddresses(source, *result, builder, env);
      }
      analysis.clear();
    };
    if (structured) {
      PtrAnalysis<StructuredPtrRules> analysis(builder, StructuredPtrRules{});
      check(analysis);
    } else {
      PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
      check(analysis);
    }
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
          EXPECT_FALSE(tensor.getElementType().isInteger(64));
    });
    EXPECT_TRUE(succeeded(verify(*module)));
    IRRewriter rewriter(&context);
    test::eraseDeadPtrArithmetic(f, rewriter, helpers);
    SmallVector<Operation *> remaining;
    f.walk([&](triton::BitcastOp op) { remaining.push_back(op); });
    EXPECT_EQ(remaining, originalCasts);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
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
  PtrAnalysis<StructuredPtrRules> analysis(builder, StructuredPtrRules{});
  auto before = test::getPtrInfo(analysis.analyze(tagged(f, "before")));
  auto after = test::getPtrInfo(analysis.analyze(tagged(f, "cast")));
  ASSERT_TRUE(succeeded(before));
  ASSERT_TRUE(succeeded(after));
  ASSERT_TRUE(before->offsets.axes[0].subAxes);
  ASSERT_TRUE(after->offsets.axes[0].subAxes);
  ASSERT_EQ(before->offsets.axes[0].subAxes->size(),
            after->offsets.axes[0].subAxes->size());
  for (auto [left, right] : llvm::zip(*before->offsets.axes[0].subAxes,
                                      *after->offsets.axes[0].subAxes)) {
    EXPECT_EQ(left.shape, right.shape);
    EXPECT_EQ(left.stride, right.stride);
  }
  EXPECT_EQ(before->offsets.addressTerms, after->offsets.addressTerms);
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
  PtrAnalysis<CommonPtrRules> analysis(builder, CommonPtrRules{});
  unsigned n = count(*module);
  auto root = test::getPtrInfo(analysis.analyze(tagged(f, "root")));
  ASSERT_TRUE(succeeded(root));
  EXPECT_EQ(count(*module), n);
  EXPECT_EQ(root->base, tagged(f, "root"));
  EXPECT_EQ(test::evaluate(root->offsets.fullOffsets)->front(),
            llvm::APInt(64, 0));
  auto after = test::getPtrInfo(analysis.analyze(tagged(f, "after")));
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
      {ptr(builder.getF32Type()), ptr(builder.getI16Type())},
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
    PtrAnalysis<CommonPtrRules> analysis(body, CommonPtrRules{});
    // Reject the cast's units, not an opaque input that analysis cannot visit.
    ASSERT_TRUE(succeeded(test::getPtrInfo(analysis.analyze(pointers))));
    unsigned n = count(fn);
    auto result = analysis.analyze(cast);
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(std::holds_alternative<std::monostate>(*result));
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
  PtrAnalysis<CommonPtrRules> analysis(body, CommonPtrRules{});
  auto value = tagged(fn, "numeric");
  auto result = analysis.analyze(value);
  ASSERT_TRUE(succeeded(result));
  auto *integer = getIntegerInfo(*result);
  ASSERT_TRUE(integer);
  EXPECT_EQ(integer->offsets.fullOffsets, OpFoldResult(value));
  EXPECT_FALSE(integer->offsets.hasCompleteForm());
  EXPECT_EQ(integer->offsets.axes[0].kind, AxisKind::Unknown);
  EXPECT_TRUE(succeeded(verify(*parsed)));
  module.erase();
}
} // namespace
