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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
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
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, triton::TritonDialect,
                      tensor::TensorDialect>();
}
Value tagged(Operation *scope, StringRef name) {
  Value value;
  scope->walk([&](Operation *op) {
    if (auto tag = op->getAttrOfType<StringAttr>("tag"))
      if (tag.getValue() == name)
        value = op->getResult(0);
  });
  return value;
}
unsigned operations(Operation *scope) {
  unsigned count = 0;
  scope->walk([&](Operation *) { ++count; });
  return count;
}
template <typename T>
void expectLanes(const T &value, ArrayRef<int64_t> expected,
                 const test::Environment &env = test::Environment()) {
  auto lanes = evaluate(value, env);
  ASSERT_TRUE(lanes);
  ASSERT_EQ(lanes->size(), expected.size());
  for (unsigned i = 0; i < expected.size(); ++i)
    EXPECT_EQ((*lanes)[i].getSExtValue(), expected[i]) << "lane " << i;
}
// Expand the public coordinate description independently of its combiners.
void expectDecomposition(const OffsetInfo &info,
                         const test::Environment &env = test::Environment()) {
  ASSERT_TRUE(info.hasStructuredDecomposition());
  auto actual = evaluate(info, env);
  auto origin = evaluate(info.scalarOffset, env);
  ASSERT_TRUE(actual && origin);
  for (unsigned lane = 0; lane < actual->size(); ++lane) {
    auto value = origin->front();
    int64_t rest = lane;
    for (int i = info.getShape().size() - 1; i >= 0; --i) {
      int64_t coordinate = rest % info.getShape()[i];
      rest /= info.getShape()[i];
      const auto &axis = info.axes[i];
      if (!axis.subAxes.empty()) {
        for (auto &sub : llvm::reverse(axis.subAxes)) {
          auto stride = evaluate(sub.stride, env);
          ASSERT_TRUE(stride);
          int64_t size = sub.shape;
          value += llvm::APInt(64, coordinate % size)
                       .zextOrTrunc(value.getBitWidth()) *
                   stride->front();
          coordinate /= size;
        }
      } else {
        auto stride = evaluate(axis.stride, env);
        ASSERT_TRUE(stride);
        value += llvm::APInt(64, coordinate).zextOrTrunc(value.getBitWidth()) *
                 stride->front();
      }
    }
    EXPECT_EQ(value, (*actual)[lane]) << "lane " << lane;
  }
}

TEST(PtrAnalysisInteger, PureIntegerOperationsKeepUniformAndPartialAxes) {
  MLIRContext context;
  load(context);
  for (StringRef opname :
       {"arith.divsi", "arith.divui", "arith.remsi", "arith.remui",
        "arith.minsi", "arith.minui", "arith.maxsi", "arith.maxui",
        "arith.andi", "arith.ori", "arith.xori", "arith.cmpi"}) {
    bool cmp = opname == "arith.cmpi";
    std::string text =
        R"mlir(module { tt.func @uniform(%x:i8, %base:!tt.ptr<i32>) {
      %s = tt.splat %x : i8 -> tensor<4xi8>
      %d = arith.constant dense<3> : tensor<4xi8>
      %v = "OP"(%s,%d) {ATTR tag="op"} : (tensor<4xi8>,tensor<4xi8>) -> tensor<4xRESULT>
      %p = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
      %a = tt.addptr %p,%v {tag="ptr"} : tensor<4x!tt.ptr<i32>>,tensor<4xRESULT>
      tt.return
    } })mlir";
    auto replace = [&](StringRef key, StringRef value) {
      for (size_t pos; (pos = text.find(key.str())) != std::string::npos;)
        text.replace(pos, key.size(), value.str());
    };
    replace("OP", opname);
    replace("ATTR", cmp ? "predicate = 2 : i64," : "");
    replace("RESULT", cmp ? "i1" : "i8");
    auto module = parseSourceString<ModuleOp>(text, &context);
    ASSERT_TRUE(module) << opname.str();
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder b(f.getBody().front().getTerminator());
    auto ip = b.saveInsertionPoint();
    PtrAnalysis analysis(b);
    Value v = tagged(f, "op");
    auto integer = test::getInfo(analysis.analyze(v));
    ASSERT_TRUE(succeeded(integer));
    EXPECT_TRUE(integer->offsets.isUniform());
    auto ptr = test::getInfo(analysis.analyze(tagged(f, "ptr")));
    ASSERT_TRUE(succeeded(ptr));
    EXPECT_TRUE(ptr->offsets.isUniform());
    for (int x : {-128, -7, 0, 7, 127}) {
      test::Environment env{{f.getArgument(0), {llvm::APInt(8, x, true)}}};
      expectDecomposition(integer->offsets, env);
      expectDecomposition(ptr->offsets, env);
      auto source = evaluate(v, env), projected = evaluate(ptr->offsets, env);
      ASSERT_TRUE(source && projected);
      EXPECT_EQ(projected->front(), source->front().sext(64));
    }
    auto count = operations(*module);
    ASSERT_TRUE(succeeded(analysis.analyze(tagged(f, "ptr"))));
    EXPECT_EQ(operations(*module), count);
    EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysisInteger, ElementwiseOnlyInheritsCommonInvariantAxes) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @partial(%x:tensor<2xi32>) {
    %e=tt.expand_dims %x {axis=1:i32}:tensor<2xi32> -> tensor<2x1xi32>
    %b=tt.broadcast %e:tensor<2x1xi32> -> tensor<2x4xi32>
    %mask=arith.constant dense<255>:tensor<2x4xi32>
    %a=arith.andi %b,%mask {tag="and"}:tensor<2x4xi32>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  auto info = test::getInfo(analysis.analyze(tagged(f, "and")));
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(test::axisKinds(info->offsets),
            (SmallVector<AxisKind>{AxisKind::Unknown, AxisKind::Invariant}));
  EXPECT_FALSE(info->offsets.scalarOffset);
  test::Environment env{
      {f.getArgument(0), {llvm::APInt(32, -1, true), llvm::APInt(32, 7)}}};
  expectLanes(info->offsets, {255, 255, 255, 255, 7, 7, 7, 7}, env);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisInteger, ZeroAxisContributionsBecomeInvariant) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @test() {
      %k=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
      %four=arith.constant dense<4>:tensor<16xi32>
      %zero=arith.constant dense<0>:tensor<16xi32>
      %r=arith.remui %k,%four:tensor<16xi32>
      %cancel=arith.subi %r,%r {tag="cancel"}:tensor<16xi32>
      %linear=arith.muli %k,%zero {tag="linear"}:tensor<16xi32>
      %split=arith.muli %r,%zero {tag="split"}:tensor<16xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  for (StringRef name : {"cancel", "linear", "split"}) {
    Value value = tagged(f, name);
    auto info = analysis.analyze(value);
    ASSERT_TRUE(succeeded(info)) << name.str();
    EXPECT_TRUE(info->offsets.isUniform()) << name.str();
    EXPECT_EQ(info->offsets.axes[0].kind, AxisKind::Invariant);
    EXPECT_EQ(info->offsets.axes[0].subAxes.empty(), name == "linear");
    expectLanes(info->offsets.axes[0].stride, {0});
    expectLanes(info->offsets, SmallVector<int64_t>(16, 0));
    expectDecomposition(info->offsets);
    EXPECT_TRUE(
        succeeded(mlir::triton::ptr::detail::validateInfo(value, *info)));
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisInteger, LoadedOriginAndDynamicStrideKeepStructuredAddresses) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @dynamic(%input:!tt.ptr<i32>, %base:!tt.ptr<f32>, %stride:i32) {
      %origin = tt.load %input : !tt.ptr<i32>
      %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
      %s = tt.splat %origin : i32 -> tensor<8xi32>
      %factor = tt.splat %stride : i32 -> tensor<8xi32>
      %scaled = arith.muli %r,%factor : tensor<8xi32>
      %offset = arith.addi %s,%scaled {tag="offset"} : tensor<8xi32>
      %p = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %ptr = tt.addptr %p,%offset {tag="ptr"} : tensor<8x!tt.ptr<f32>>,tensor<8xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  Value origin = *f.getBody().front().getOps<triton::LoadOp>().begin();
  Value offset = tagged(f, "offset");
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  auto integer = analysis.analyze(offset);
  auto pointer = analysis.analyze(tagged(f, "ptr"));
  ASSERT_TRUE(succeeded(integer) && succeeded(pointer));
  ASSERT_FALSE(integer->isEmpty());
  ASSERT_FALSE(pointer->isEmpty());
  EXPECT_FALSE(integer->base);
  EXPECT_EQ(pointer->base, f.getArgument(1));
  EXPECT_EQ(integer->offsets.fullOffsets, OpFoldResult(offset));
  ASSERT_TRUE(integer->offsets.hasStructuredForm());
  ASSERT_TRUE(pointer->offsets.hasStructuredForm());
  for (int start : {-123, 0, 127})
    for (int stride : {-3, 1, 16}) {
      test::Environment env{
          {origin, {llvm::APInt(32, start, true)}},
          {f.getArgument(2), {llvm::APInt(32, stride, true)}}};
      expectDecomposition(integer->offsets, env);
      expectDecomposition(pointer->offsets, env);
    }

  // Unified results still reject empty or mismatched boundary bindings.
  EXPECT_TRUE(failed(analysis.setInfo(offset, PtrInfo{})));
  PtrInfo wrong = *integer;
  wrong.base = f.getArgument(1);
  EXPECT_TRUE(failed(analysis.setInfo(offset, wrong)));
  wrong = *integer;
  wrong.block = BlockPtrInfo{};
  EXPECT_TRUE(failed(analysis.setInfo(offset, wrong)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisInteger, DynamicOriginsKeepLegacyStructureAndExactSources) {
  MLIRContext context;
  load(context);
  auto module =
      parseSourceString<ModuleOp>(R"mlir(module { tt.func @bounds(%x:i32) {
    %mask=arith.constant 255:i32
    %a=arith.andi %x,%mask:i32
    %s=tt.splat %a:i32 -> tensor<4xi32>
    %r=tt.make_range {start=0:i32,end=4:i32}:tensor<4xi32>
    %sum=arith.addi %s,%r {tag="sum"}:tensor<4xi32>
    %stride=arith.muli %s,%r {tag="stride"}:tensor<4xi32>
    %u=tt.splat %x:i32 -> tensor<4xi32>
    %unknown=arith.addi %u,%r {tag="unknown"}:tensor<4xi32>
    %c=arith.constant dense<2147483646>:tensor<4xi32>
    %wrap=arith.addi %c,%r {tag="wrap"}:tensor<4xi32>
    %narrow=arith.trunci %sum {tag="narrow"}:tensor<4xi32> to tensor<4xi8>
    %unsigned=arith.extui %narrow {tag="unsigned"}:tensor<4xi8> to tensor<4xi64>
    %signed=arith.extsi %sum {tag="signed"}:tensor<4xi32> to tensor<4xi64>
    tt.return
  } })mlir",
                                  &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  for (StringRef tag :
       {"sum", "stride", "unknown", "wrap", "narrow", "unsigned", "signed"}) {
    Value value = tagged(f, tag);
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info));
    bool decomposed =
        tag == "sum" || tag == "stride" || tag == "unknown" || tag == "signed";
    // Wrapped source values stay exact. Their index descriptor is outside
    // the legacy no-overflow contract and is not used for reconstruction.
    if (tag != "wrap")
      EXPECT_EQ(info->hasStructuredDecomposition(), decomposed) << tag.str();
    for (int64_t x : {0, 1, 127, 255, -1}) {
      test::Environment env{{f.getArgument(0), {llvm::APInt(32, x, true)}}};
      auto source = evaluate(value, env), address = evaluate(*info, env);
      ASSERT_TRUE(source && address);
      for (unsigned i = 0; i < source->size(); ++i)
        EXPECT_EQ((*address)[i], (*source)[i].sextOrTrunc(64));
      if (decomposed)
        expectDecomposition(*info, env);
    }
    if (tag == "wrap")
      expectLanes(*info, {2147483646, 2147483647, -2147483648, -2147483647});
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisInteger, ExactSourceWidthsAndIndexCoefficients) {
  MLIRContext context;
  load(context);
  for (auto [origin, step, valid] : {std::tuple<int, int, bool>{125, 1, true},
                                     {126, 1, false},
                                     {-127, -1, true},
                                     {-128, -1, false}}) {
    std::string text = R"mlir(module { tt.func @edge(%x:i8) {
      %mask=arith.constant 1:i8
      %masked=arith.andi %x,%mask:i8
      %origin=arith.constant ORIGIN:i8
      %a=arith.addi %masked,%origin:i8
      %s=tt.splat %a:i8 -> tensor<2xi8>
      %r=tt.make_range {start=0:i32,end=2:i32}:tensor<2xi32>
      %n=arith.trunci %r:tensor<2xi32> to tensor<2xi8>
      %step=arith.constant dense<STEP>:tensor<2xi8>
      %delta=arith.muli %n,%step:tensor<2xi8>
      %sum=arith.addi %s,%delta {tag="sum"}:tensor<2xi8>
      tt.return
    } })mlir";
    text.replace(text.find("ORIGIN"), 6, std::to_string(origin));
    text.replace(text.find("STEP"), 4, std::to_string(step));
    auto module = parseSourceString<ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder b(&context);
    PtrAnalysis analysis(b);
    Value v = tagged(f, "sum");
    auto info = test::getOffsets(analysis.analyze(v));
    ASSERT_TRUE(succeeded(info));
    EXPECT_TRUE(info->hasStructuredDecomposition());
    for (int x : {0, 1}) {
      test::Environment env{{f.getArgument(0), {llvm::APInt(8, x, true)}}};
      expectLanes(*info, {int8_t(origin + x), int8_t(origin + x + step)}, env);
      if (valid)
        expectDecomposition(*info, env);
    }
  }
  // The tensor sum fits [-100,100], but its i8 coefficient 100+100
  // wraps to -56. A final-value range alone cannot justify the decomposition.
  auto module =
      parseSourceString<ModuleOp>(R"mlir(module { tt.func @coefficient() {
    %r=tt.make_range {start=0:i32,end=2:i32}:tensor<2xi32>
    %n=arith.trunci %r:tensor<2xi32> to tensor<2xi8>
    %c=arith.constant dense<100>:tensor<2xi8>
    %m=arith.muli %n,%c:tensor<2xi8>
    %o=arith.constant dense<-50>:tensor<2xi8>
    %a=arith.addi %m,%o:tensor<2xi8>
    %sum=arith.addi %a,%a {tag="sum"}:tensor<2xi8>
    tt.return
  } })mlir",
                                  &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  Value v = tagged(f, "sum");
  auto info = test::getOffsets(analysis.analyze(v));
  ASSERT_TRUE(succeeded(info));
  ASSERT_TRUE(info->hasStructuredDecomposition());
  EXPECT_EQ(test::evaluate(info->scalarOffset)->front().getSExtValue(), -100);
  EXPECT_EQ(test::evaluate(info->axes[0].stride)->front().getSExtValue(), 200);
  expectLanes(*info, {-100, 100});
}

TEST(PtrAnalysisInteger, StructuredNonzeroOriginStrideAndNestedDigits) {
  MLIRContext context;
  load(context);
  for (StringRef op :
       {"arith.divui", "arith.remui", "arith.divsi", "arith.remsi"}) {
    for (int step : {2, 8}) {
      std::string text = R"mlir(module { tt.func @split(%base:!tt.ptr<i32>) {
        %r=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
        %s=arith.constant dense<STEP>:tensor<16xi32>
        %m=arith.muli %r,%s:tensor<16xi32>
        %o=arith.constant dense<8>:tensor<16xi32>
        %x=arith.addi %m,%o:tensor<16xi32>
        %d=arith.constant dense<4>:tensor<16xi32>
        %v=OP %x,%d {tag="split"}:tensor<16xi32>
        %e=tt.expand_dims %v {axis=0:i32}:tensor<16xi32> -> tensor<1x16xi32>
        %b=tt.broadcast %e:tensor<1x16xi32> -> tensor<2x16xi32>
        %eight=arith.constant dense<8>:tensor<2x16xi32>
        %scaled=arith.muli %b,%eight:tensor<2x16xi32>
        %four=arith.constant dense<4>:tensor<2x16xi32>
        %again=arith.divui %scaled,%four {tag="again"}:tensor<2x16xi32>
        %p=tt.splat %base:!tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
        %ptr=tt.addptr %p,%again {tag="ptr"}:tensor<2x16x!tt.ptr<i32>>,tensor<2x16xi32>
        tt.return
      } })mlir";
      text.replace(text.find("STEP"), 4, std::to_string(step));
      text.replace(text.find("OP"), 2, op.str());
      auto module = parseSourceString<ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      auto f = *module->getOps<triton::FuncOp>().begin();
      OpBuilder b(&context);
      PtrAnalysis analysis(b);
      for (StringRef tag : {"split", "again"}) {
        auto info = test::getOffsets(analysis.analyze(tagged(f, tag)));
        ASSERT_TRUE(succeeded(info));
        expectDecomposition(*info);
      }
      auto split = test::getOffsets(analysis.analyze(tagged(f, "split")));
      if (step == 2) {
        ASSERT_FALSE(split->axes[0].subAxes.empty());
        EXPECT_EQ(split->axes[0].subAxes.size(), 2u);
        EXPECT_EQ(split->axes[0].subAxes.front().shape, 8);
        EXPECT_EQ(split->axes[0].subAxes.back().shape, 2);
      } else
        EXPECT_TRUE(split->hasStructuredForm());
      auto ptr = test::getInfo(analysis.analyze(tagged(f, "ptr")));
      ASSERT_TRUE(succeeded(ptr));
      expectDecomposition(ptr->offsets);
      EXPECT_TRUE(succeeded(verify(*module)));
    }
  }
}

TEST(PtrAnalysisInteger, DynamicAlignedOriginsSplitDivRemAxes) {
  MLIRContext context;
  load(context);
  for (StringRef op :
       {"arith.divsi", "arith.remsi", "arith.divui", "arith.remui"}) {
    // The Structured consumer uses this program-id/range expression for tiles.
    std::string text = R"mlir(module {
      tt.func @tile(%base:!tt.ptr<i32>) {
        %pid = tt.get_program_id x {tag="pid"} : i32
        %size = arith.constant 1024 : i32
        %start = arith.muli %pid,%size : i32
        %origin = tt.splat %start : i32 -> tensor<1024xi32>
        %r = tt.make_range {start=0:i32,end=1024:i32} : tensor<1024xi32>
        %x = arith.addi %origin,%r {tag="input"} : tensor<1024xi32>
        %d = arith.constant dense<512> : tensor<1024xi32>
        %v = OP %x,%d {tag="split"} : tensor<1024xi32>
        %p = tt.splat %base : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %ptr = tt.addptr %p,%v {tag="ptr"} : tensor<1024x!tt.ptr<i32>>,tensor<1024xi32>
        tt.return
      }
    })mlir";
    text.replace(text.find("OP"), 2, op.str());
    auto module = parseSourceString<ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder builder(&context);
    PtrAnalysis analysis(builder);
    auto input = test::getOffsets(analysis.analyze(tagged(f, "input")));
    ASSERT_TRUE(succeeded(input));
    ASSERT_TRUE(input->hasStructuredForm());
    Value value = tagged(f, "split");
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
    ASSERT_TRUE(info->hasStructuredDecomposition());
    ASSERT_EQ(info->axes[0].subAxes.size(), 2u);
    EXPECT_EQ(info->axes[0].subAxes[0].shape, 2);
    EXPECT_EQ(info->axes[0].subAxes[1].shape, 512);
    bool divide = op.contains("div");
    expectLanes(info->axes[0].subAxes[0].stride, {divide ? 1 : 0});
    expectLanes(info->axes[0].subAxes[1].stride, {divide ? 0 : 1});
    auto ptr = test::getInfo(analysis.analyze(tagged(f, "ptr")));
    ASSERT_TRUE(succeeded(ptr));
    // Include the last program id whose entire tile fits in signed i32.
    for (int64_t pid : {0, 7, 2097151}) {
      test::Environment env{{tagged(f, "pid"), {llvm::APInt(32, pid)}}};
      expectLanes(info->scalarOffset, {divide ? pid * 2 : 0}, env);
      expectDecomposition(*info, env);
      expectDecomposition(ptr->offsets, env);
    }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysisInteger, NonnegativeScalarOriginsKeepDivRemSplits) {
  MLIRContext context;
  load(context);
  for (StringRef op :
       {"arith.divsi", "arith.remsi", "arith.divui", "arith.remui"}) {
    std::string text = R"mlir(module {
      tt.func @split(%x:i32, %small:i8) {
        %zero = arith.constant 0 : i32
        %positive = arith.maxsi %x,%zero : i32
        %unsigned = arith.extui %small : i8 to i32
        %sum = arith.addi %positive,%unsigned : i32
        %size = arith.constant 16 : i32
        %start = arith.muli %sum,%size : i32
        %origin = tt.splat %start : i32 -> tensor<16xi32>
        %r = tt.make_range {start=0:i32,end=16:i32} : tensor<16xi32>
        %xv = arith.addi %origin,%r : tensor<16xi32>
        %d = arith.constant dense<4> : tensor<16xi32>
        %v = OP %xv,%d {tag="split"} : tensor<16xi32>
        tt.return
      }
    })mlir";
    text.replace(text.find("OP"), 2, op.str());
    auto module = parseSourceString<ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder builder(&context);
    PtrAnalysis analysis(builder);
    auto info = test::getOffsets(analysis.analyze(tagged(f, "split")));
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->hasStructuredDecomposition());
    for (int64_t x : {-8, 0, 7}) {
      test::Environment env{{f.getArgument(0), {llvm::APInt(32, x, true)}},
                            {f.getArgument(1), {llvm::APInt(8, 255)}}};
      expectLanes(
          info->scalarOffset,
          {op.contains("div") ? (std::max<int64_t>(x, 0) + 255) * 4 : 0}, env);
      expectDecomposition(*info, env);
    }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysisInteger, DynamicSplitNeedsAlignedNonnegativeOrigins) {
  MLIRContext context;
  load(context);
  // Multiplicity alone does not prove the sign; narrowing also loses it.
  for (StringRef origin : {"%start = arith.muli %arg,%four : i32",
                           "%start = arith.addi %scaled,%one : i32",
                           "%small = arith.trunci %pid : i32 to i8\n"
                           "%extended = arith.extsi %small : i8 to i32\n"
                           "%start = arith.muli %extended,%four : i32",
                           "%index = arith.index_cast %pid : i32 to index\n"
                           "%small = arith.index_cast %index : index to i8\n"
                           "%extended = arith.extsi %small : i8 to i32\n"
                           "%start = arith.muli %extended,%four : i32"}) {
    for (StringRef op :
         {"arith.divsi", "arith.remsi", "arith.divui", "arith.remui"}) {
      std::string text = R"mlir(module {
        tt.func @unproved(%arg:i32) {
          %pid = tt.get_program_id x {tag="pid"} : i32
          %four = arith.constant 4 : i32
          %one = arith.constant 1 : i32
          %scaled = arith.muli %pid,%four : i32
          ORIGIN
          %origin = tt.splat %start : i32 -> tensor<8xi32>
          %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
          %x = arith.addi %origin,%r {tag="input"} : tensor<8xi32>
          %d = arith.constant dense<4> : tensor<8xi32>
          %v = OP %x,%d {tag="split"} : tensor<8xi32>
          tt.return
        }
      })mlir";
      text.replace(text.find("ORIGIN"), 6, origin.str());
      text.replace(text.find("OP"), 2, op.str());
      auto module = parseSourceString<ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      auto f = *module->getOps<triton::FuncOp>().begin();
      OpBuilder builder(&context);
      PtrAnalysis analysis(builder);
      auto input = test::getOffsets(analysis.analyze(tagged(f, "input")));
      ASSERT_TRUE(succeeded(input));
      ASSERT_TRUE(input->hasStructuredForm());
      Value value = tagged(f, "split");
      unsigned before = operations(f);
      auto info = test::getOffsets(analysis.analyze(value));
      ASSERT_TRUE(succeeded(info));
      EXPECT_FALSE(info->hasStructuredDecomposition());
      EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
      EXPECT_EQ(operations(f), before);
      test::Environment env{{f.getArgument(0), {llvm::APInt(32, -4, true)}},
                            {tagged(f, "pid"), {llvm::APInt(32, 252)}}};
      auto source = evaluate(value, env);
      ASSERT_TRUE(source);
      SmallVector<int64_t> expected;
      for (auto lane : *source)
        expected.push_back(lane.getSExtValue());
      expectLanes(*info, expected, env);
      EXPECT_TRUE(succeeded(verify(*module)));
    }
  }
}

TEST(PtrAnalysisInteger, DynamicSplitRejectsNegativeStridesAndResidualCarries) {
  MLIRContext context;
  load(context);
  for (StringRef op :
       {"arith.divsi", "arith.remsi", "arith.divui", "arith.remui"}) {
    std::string text = R"mlir(module {
      tt.func @axes() {
        %pid = tt.get_program_id x {tag="pid"} : i32
        %size = arith.constant 16 : i32
        %start = arith.muli %pid,%size : i32
        %origin = tt.splat %start : i32 -> tensor<8x8xi32>
        %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
        %row = tt.expand_dims %r {axis=1:i32} : tensor<8xi32> -> tensor<8x1xi32>
        %col = tt.expand_dims %r {axis=0:i32} : tensor<8xi32> -> tensor<1x8xi32>
        %rows = tt.broadcast %row : tensor<8x1xi32> -> tensor<8x8xi32>
        %cols = tt.broadcast %col : tensor<1x8xi32> -> tensor<8x8xi32>
        %base = arith.addi %origin,%rows : tensor<8x8xi32>
        %negative = arith.subi %base,%cols {tag="negative_input"} : tensor<8x8xi32>
        %carry = arith.addi %base,%cols {tag="carry_input"} : tensor<8x8xi32>
        %d = arith.constant dense<4> : tensor<8x8xi32>
        %neg = OP %negative,%d {tag="negative"} : tensor<8x8xi32>
        %sum = OP %carry,%d {tag="carry"} : tensor<8x8xi32>
        tt.return
      }
    })mlir";
    for (unsigned i = 0; i < 2; ++i)
      text.replace(text.find("OP"), 2, op.str());
    auto module = parseSourceString<ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder builder(&context);
    PtrAnalysis analysis(builder);
    for (StringRef tag : {"negative", "carry"}) {
      auto input =
          test::getOffsets(analysis.analyze(tagged(f, (tag + "_input").str())));
      ASSERT_TRUE(succeeded(input));
      ASSERT_TRUE(input->hasStructuredForm());
      unsigned before = operations(f);
      Value value = tagged(f, tag);
      auto info = test::getOffsets(analysis.analyze(value));
      ASSERT_TRUE(succeeded(info));
      EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
      EXPECT_FALSE(info->hasStructuredDecomposition());
      EXPECT_EQ(operations(f), before);
    }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysisInteger, DynamicSplitRequiresCompleteAxisFacts) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @partial() {
      %pid = tt.get_program_id x : i32
      %size = arith.constant 16 : i32
      %start = arith.muli %pid,%size : i32
      %origin = tt.splat %start : i32 -> tensor<8xi32>
      %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
      %x = arith.addi %origin,%r {tag="input"} : tensor<8xi32>
      %d = arith.constant dense<4> : tensor<8xi32>
      %v = arith.divsi %x,%d {tag="split"} : tensor<8xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  Value input = tagged(f, "input");
  auto bound = analysis.analyze(input);
  ASSERT_TRUE(succeeded(bound));
  ASSERT_TRUE(bound->offsets.hasStructuredForm());
  // A known origin with an Unknown axis is a valid partial binding. Its zero
  // stride is only a placeholder and must not become an invariance proof.
  bound->offsets.axes[0] = AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0));
  ASSERT_TRUE(succeeded(analysis.setInfo(input, *bound)));
  Value value = tagged(f, "split");
  auto info = test::getOffsets(analysis.analyze(value));
  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
  EXPECT_FALSE(info->hasStructuredDecomposition());
  EXPECT_EQ(info->axes[0].kind, AxisKind::Unknown);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisInteger,
     StructuredRejectsUnprovedDivisionWithoutLosingExactValues) {
  MLIRContext context;
  load(context);
  for (auto [origin, step, divisor] : {std::tuple<int, int, int>{-8, 2, 4},
                                       {1, 2, 4},
                                       {120, 2, 4},
                                       {0, 1, 3},
                                       {0, 1, 0},
                                       {-128, 0, -1}}) {
    std::string text = R"mlir(module { tt.func @fallback() {
      %r=tt.make_range {start=0:i32,end=8:i32}:tensor<8xi32>
      %n=arith.trunci %r:tensor<8xi32> to tensor<8xi8>
      %s=arith.constant dense<STEP>:tensor<8xi8>
      %m=arith.muli %n,%s:tensor<8xi8>
      %o=arith.constant dense<ORIGIN>:tensor<8xi8>
      %a=arith.addi %m,%o:tensor<8xi8>
      %d=arith.constant dense<DIV>:tensor<8xi8>
      %v=arith.divsi %a,%d {tag="div"}:tensor<8xi8>
      tt.return
    } })mlir";
    for (auto [key, value] : {std::pair<StringRef, int>{"STEP", step},
                              {"ORIGIN", origin},
                              {"DIV", divisor}})
      text.replace(text.find(key.str()), key.size(), std::to_string(value));
    auto module = parseSourceString<ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<triton::FuncOp>().begin();
    OpBuilder b(&context);
    PtrAnalysis analysis(b);
    Value v = tagged(f, "div");
    auto info = test::getOffsets(analysis.analyze(v));
    ASSERT_TRUE(succeeded(info));
    EXPECT_EQ(info->fullOffsets, OpFoldResult(v));
    if (step)
      EXPECT_FALSE(info->hasStructuredDecomposition());
    else
      EXPECT_TRUE(info->isUniform());
    EXPECT_TRUE(info->axes[0].subAxes.empty());
    if (divisor && divisor != -1) {
      SmallVector<int64_t> expected;
      for (int i = 0; i < 8; ++i)
        expected.push_back(int8_t(origin + i * step) / divisor);
      expectLanes(*info, expected);
    } else
      EXPECT_FALSE(evaluate(*info));
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PtrAnalysisInteger, ExactDivisionKeepsNegativeAndDynamicOrigins) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @exact(%pid:i32, %x:i32) {
      %r = tt.make_range {start=0:i32,end=8:i32} : tensor<8xi32>
      %four = arith.constant dense<4> : tensor<8xi32>
      %scaled = arith.muli %r,%four : tensor<8xi32>
      %negative = arith.constant dense<-16> : tensor<8xi32>
      %n = arith.addi %scaled,%negative : tensor<8xi32>
      %negativeDiv = arith.divsi %n,%four {tag="negative"} : tensor<8xi32>
      %negativeRem = arith.remsi %n,%four {tag="remainder"} : tensor<8xi32>
      %unsignedDiv = arith.divui %n,%four {tag="unsigned"} : tensor<8xi32>
      %sixteen = arith.constant 16 : i32
      %start = arith.muli %pid,%sixteen : i32
      %s = tt.splat %start : i32 -> tensor<8xi32>
      %d = arith.addi %s,%scaled : tensor<8xi32>
      %dynamicDiv = arith.divsi %d,%four {tag="dynamic"} : tensor<8xi32>
      %dynamicRem = arith.remsi %d,%four {tag="dynamic_rem"} : tensor<8xi32>
      %zero = arith.constant 0 : i32
      %hint = arith.addi %x,%zero {tt.divisibility=dense<4>:tensor<1xi32>} : i32
      %h = tt.splat %hint : i32 -> tensor<8xi32>
      %a = arith.addi %h,%scaled : tensor<8xi32>
      %annotatedDiv = arith.divsi %a,%four {tag="annotated"} : tensor<8xi32>
      %unknown = tt.splat %x : i32 -> tensor<8xi32>
      %u = arith.addi %unknown,%scaled : tensor<8xi32>
      %unproved = arith.divsi %u,%four {tag="unproved"} : tensor<8xi32>
      tt.return
    }
  })mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  PtrAnalysis analysis(builder);
  for (StringRef tag :
       {"negative", "remainder", "dynamic", "dynamic_rem", "annotated"}) {
    Value value = tagged(f, tag);
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info)) << tag.str();
    ASSERT_TRUE(info->hasStructuredForm()) << tag.str();
    EXPECT_EQ(info->fullOffsets, OpFoldResult(value));
    for (int64_t pid : {-3, 0, 5}) {
      test::Environment env{{f.getArgument(0), {llvm::APInt(32, pid, true)}},
                            {f.getArgument(1), {llvm::APInt(32, -12, true)}}};
      expectDecomposition(*info, env);
    }
    expectLanes(info->axes[0].stride,
                {tag.contains("rem") ? int64_t(0) : int64_t(1)});
  }
  for (StringRef tag : {"unsigned", "unproved"}) {
    auto info = test::getOffsets(analysis.analyze(tagged(f, tag)));
    ASSERT_TRUE(succeeded(info));
    EXPECT_FALSE(info->hasStructuredDecomposition());
    EXPECT_EQ(info->fullOffsets, OpFoldResult(tagged(f, tag)));
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PtrAnalysisInteger, DynamicOriginsSurviveResetAndReanalysis) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module { tt.func @lifecycle(%x:i32,%unseen:i32) {
    %m=arith.constant 255:i32
    %a=arith.andi %x,%m {tag="origin"}:i32
    %s=tt.splat %a:i32 -> tensor<4xi32>
    %r=tt.make_range {start=0:i32,end=4:i32}:tensor<4xi32>
    %sum=arith.addi %s,%r {tag="sum"}:tensor<4xi32>
    tt.return
  } })mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PtrAnalysis analysis(b);
  Value sum = tagged(f, "sum");
  auto address = test::getOffsets(analysis.analyze(sum));
  ASSERT_TRUE(succeeded(address));
  ASSERT_TRUE(address->hasStructuredForm());
  auto unseen = buildOffsetInfo(f.getArgument(1), b);
  ASSERT_TRUE(
      succeeded(analysis.reset({{f.getArgument(1), PtrInfo{{}, *unseen}}})));
  EXPECT_FALSE(analysis.findCached(sum));
  auto reproved = test::getOffsets(analysis.analyze(sum));
  ASSERT_TRUE(succeeded(reproved));
  EXPECT_TRUE(reproved->hasStructuredForm());
  Value newSum = sum, newOrigin = tagged(f, "origin");
  Value extended = b.create<arith::ExtSIOp>(
      f.getLoc(), RankedTensorType::get({4}, b.getI64Type()), newSum);
  auto info = test::getOffsets(analysis.analyze(extended));
  ASSERT_TRUE(succeeded(info));
  EXPECT_TRUE(info->hasStructuredForm());
  // Preserve the scalar's actual origin while refreshing dependent results.
  auto bound = analysis.analyze(newOrigin);
  ASSERT_TRUE(succeeded(bound));
  ASSERT_TRUE(succeeded(analysis.reset({{newOrigin, *bound}})));
  EXPECT_FALSE(analysis.findCached(extended));
  info = test::getOffsets(analysis.analyze(extended));
  ASSERT_TRUE(succeeded(info));
  EXPECT_TRUE(info->hasStructuredDecomposition());
  ASSERT_TRUE(succeeded(analysis.reset()));
  info = test::getOffsets(analysis.analyze(extended));
  ASSERT_TRUE(succeeded(info));
  EXPECT_TRUE(info->hasStructuredDecomposition());
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(PtrAnalysisInteger, DynamicTermsPropagateThroughSubAxisAlignment) {
  MLIRContext context;
  load(context);
  auto module =
      parseSourceString<ModuleOp>(R"mlir(module { tt.func @subaxes(%x:i32) {
    %mask=arith.constant 255:i32
    %a=arith.andi %x,%mask:i32
    %s=tt.splat %a:i32 -> tensor<16xi32>
    %r=tt.make_range {start=0:i32,end=16:i32}:tensor<16xi32>
    %d=arith.constant dense<4>:tensor<16xi32>
    %rem=arith.remui %r,%d:tensor<16xi32>
    %sum=arith.addi %rem,%s {tag="sum"}:tensor<16xi32>
    %scaled=arith.muli %sum,%s {tag="scaled"}:tensor<16xi32>
    tt.return
  } })mlir",
                                  &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  OpBuilder b(&context);
  PtrAnalysis analysis(b);
  for (StringRef name : {"sum", "scaled"}) {
    Value value = tagged(f, name);
    auto info = test::getOffsets(analysis.analyze(value));
    ASSERT_TRUE(succeeded(info));
    ASSERT_TRUE(info->hasStructuredDecomposition());
    ASSERT_FALSE(info->axes[0].subAxes.empty());
    for (int x : {0, 1, 255, -1}) {
      test::Environment env{{f.getArgument(0), {llvm::APInt(32, x, true)}}};
      expectDecomposition(*info, env);
    }
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}
} // namespace
