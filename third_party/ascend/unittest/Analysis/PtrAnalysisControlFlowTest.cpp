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
#include "TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.h"
#include "TritonMemoryAccess/PtrAnalysisRules/ControlFlowPtrRules.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ptr;
using test::buildAddressView;
using test::eraseDeadPtrArithmetic;
namespace {
using Driver = PtrAnalysis<ControlFlowPtrRules>;
unsigned operations(Operation *op) {
  unsigned n = 0;
  op->walk([&](Operation *) { ++n; });
  return n;
}
void load(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, TritonDialect, scf::SCFDialect>();
}
FailureOr<SymbolicPtrInfo> ptr(Driver &driver, Value v) {
  auto info = driver.analyze(v);
  if (failed(info))
    return failure();
  auto *p = std::get_if<SymbolicPtrInfo>(&*info);
  if (!p)
    return failure();
  return *p;
}
std::optional<int64_t> scalar(ComponentMaterializer &materializer,
                              const ComponentInfo &info) {
  auto value = materializer.materialize(info);
  if (failed(value))
    return std::nullopt;
  auto lanes = test::evaluate(*value);
  if (!lanes || lanes->size() != 1)
    return std::nullopt;
  return lanes->front().getSExtValue();
}
TEST(ControlFlowPtrRules, SymbolicBoundaryRetainsIrregularAndStrideIdentities) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @test(%current:tensor<2x4x!tt.ptr<i32>>, %step:i64, %newarg:i64) {
      %delta=tt.splat %step:i64 -> tensor<2x4xi64>
      %zero=arith.constant dense<0>:tensor<2x4xi64>
      %next=tt.addptr %current,%delta:tensor<2x4x!tt.ptr<i32>>,tensor<2x4xi64>
      %same=tt.addptr %next,%zero:tensor<2x4x!tt.ptr<i32>>,tensor<2x4xi64>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  SmallVector<AddPtrOp> adds;
  f.walk([&](AddPtrOp op) { adds.push_back(op); });
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  Driver driver(b, ControlFlowPtrRules{});
  auto boundary = ptr(driver, f.getArgument(0));
  ASSERT_TRUE(succeeded(boundary));
  auto source = f.getArgument(0);
  // A consumer binds current region identities, not concrete initial values.
  boundary->base =
      ComponentInfo::boundary(source, 10, PointerType::get(b.getI32Type(), 1));
  boundary->offsets.terms.origin =
      ComponentInfo::boundary(source, 11, b.getI64Type());
  boundary->offsets.terms.irregular =
      ComponentInfo::boundary(source, 12, boundary->offsets.valueType);
  boundary->offsets.fullOffsets =
      ComponentInfo::boundary(source, 13, boundary->offsets.valueType);
  boundary->offsets.terms.coefficients[1] =
      ComponentInfo{b.getI64Type(), {source, 14}, b.getI64IntegerAttr(4), {}};
  ASSERT_TRUE(succeeded(driver.bind(source, *boundary)));
  auto *bound = driver.findCached(source);
  auto malformed = *boundary;
  malformed.offsets.terms.origin->concrete = b.getI32IntegerAttr(7);
  EXPECT_TRUE(failed(driver.bind(source, malformed)));
  EXPECT_EQ(driver.findCached(source), bound);
  malformed = *boundary;
  malformed.offsets.valueType = b.getI32Type();
  EXPECT_TRUE(failed(driver.bind(source, malformed)));
  EXPECT_EQ(driver.findCached(source), bound);

  PtrAnalysis<CommonPtrRules> concrete(b, CommonPtrRules{});
  EXPECT_TRUE(failed(test::getPtrInfo(concrete.analyze(source))));
  unsigned count = operations(*module);
  auto next = ptr(driver, adds[0]), same = ptr(driver, adds[1]);
  ASSERT_TRUE(succeeded(next));
  ASSERT_TRUE(succeeded(same));
  EXPECT_TRUE(next->base.sameIdentity(boundary->base));
  EXPECT_TRUE(next->offsets.terms.irregular->sameIdentity(
      *boundary->offsets.terms.irregular));
  EXPECT_TRUE(next->offsets.terms.coefficients[1]->sameIdentity(
      *boundary->offsets.terms.coefficients[1]));
  EXPECT_FALSE(next->offsets.terms.origin->sameIdentity(
      *boundary->offsets.terms.origin));
  EXPECT_FALSE(
      next->offsets.fullOffsets.sameIdentity(boundary->offsets.fullOffsets));
  EXPECT_TRUE(
      same->offsets.terms.origin->sameIdentity(*next->offsets.terms.origin));
  EXPECT_TRUE(succeeded(driver.analyze(adds[0])));
  EXPECT_EQ(operations(*module), count); // Planning is read-only.
  ComponentMaterializer materializer(
      b, adds[0],
      [&](const ComponentIdentity &id, Type) -> std::optional<OpFoldResult> {
        if (id == boundary->offsets.terms.origin->identity)
          return f.getArgument(2);
        return std::nullopt;
      });
  auto offset = materializer.materialize(*next->offsets.terms.origin);
  ASSERT_TRUE(succeeded(offset));
  test::Environment env{{f.getArgument(1), {llvm::APInt(64, 3)}},
                        {f.getArgument(2), {llvm::APInt(64, 5)}}};
  auto values = test::evaluate(*offset, env);
  ASSERT_TRUE(values);
  EXPECT_EQ(values->front(), llvm::APInt(64, 8));
  count = operations(*module);
  EXPECT_TRUE(succeeded(materializer.materialize(*next->offsets.terms.origin)));
  EXPECT_EQ(operations(*module), count);
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(ControlFlowPtrRules, ExpressionsShareComponentEquationsAndShapeRules) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @test(%base:!tt.ptr<i32>,%rows:tensor<2xi64>,%loop:i64) {
      %row=tt.expand_dims %rows {axis=1:i32}:tensor<2xi64> -> tensor<2x1xi64>
      %rowgrid=tt.broadcast %row:tensor<2x1xi64> -> tensor<2x4xi64>
      %k=tt.make_range {start=0:i32,end=4:i32}:tensor<4xi32>
      %wide=arith.extsi %k:tensor<4xi32> to tensor<4xi64>
      %four=arith.constant dense<4>:tensor<4xi64>
      %col=arith.muli %wide,%four:tensor<4xi64>
      %column=tt.expand_dims %col {axis=0:i32}:tensor<4xi64> -> tensor<1x4xi64>
      %colgrid=tt.broadcast %column:tensor<1x4xi64> -> tensor<2x4xi64>
      %offset=arith.addi %rowgrid,%colgrid:tensor<2x4xi64>
      %basegrid=tt.splat %base:!tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
      %current=tt.addptr %basegrid,%offset:tensor<2x4x!tt.ptr<i32>>,tensor<2x4xi64>
      %delta=tt.splat %loop:i64 -> tensor<2x4xi64>
      %next=tt.addptr %current,%delta:tensor<2x4x!tt.ptr<i32>>,tensor<2x4xi64>
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  SmallVector<AddPtrOp> adds;
  f.walk([&](AddPtrOp op) { adds.push_back(op); });
  OpBuilder b(f.getBody().front().getTerminator());
  Driver driver(b, ControlFlowPtrRules{});
  unsigned count = operations(*module);
  auto before = ptr(driver, adds[0]), after = ptr(driver, adds[1]);
  ASSERT_TRUE(succeeded(before));
  ASSERT_TRUE(succeeded(after));
  EXPECT_EQ(operations(*module), count);
  EXPECT_TRUE(after->offsets.terms.irregular->sameIdentity(
      *before->offsets.terms.irregular));
  EXPECT_TRUE(after->offsets.terms.coefficients[1]->sameIdentity(
      *before->offsets.terms.coefficients[1]));
  ComponentMaterializer materializer(b, adds[1]);
  EXPECT_EQ(scalar(materializer, *after->offsets.terms.coefficients[1]), 4);
  auto exact = materializer.materialize(after->offsets.fullOffsets);
  auto residual = materializer.materialize(*after->offsets.terms.irregular);
  ASSERT_TRUE(succeeded(exact));
  ASSERT_TRUE(succeeded(residual));
  test::Environment env{
      {f.getArgument(1), {llvm::APInt(64, 7), llvm::APInt(64, 19)}},
      {f.getArgument(2), {llvm::APInt(64, 3)}}};
  auto fullValues = test::evaluate(*exact, env),
       residualValues = test::evaluate(*residual, env);
  ASSERT_TRUE(fullValues);
  ASSERT_TRUE(residualValues);
  for (unsigned i = 0; i < 8; ++i) {
    EXPECT_EQ((*fullValues)[i],
              llvm::APInt(64, (i < 4 ? 7 : 19) + 4 * (i % 4) + 3));
    EXPECT_EQ((*residualValues)[i], llvm::APInt(64, i < 4 ? 7 : 19));
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(ControlFlowPtrRules, SourceWrapAndEachAddressIncrementRemainSeparate) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module {tt.func @test(%base:!tt.ptr<i32>) {
    %max=arith.constant 127:i8
    %one=arith.constant 1:i8
    %wrapped=arith.addi %max,%one:i8
    %single=tt.addptr %base,%wrapped:!tt.ptr<i32>,i8
    %first=tt.addptr %base,%max:!tt.ptr<i32>,i8
    %second=tt.addptr %first,%one:!tt.ptr<i32>,i8
    tt.return
  }})mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  SmallVector<AddPtrOp> adds;
  f.walk([&](AddPtrOp op) { adds.push_back(op); });
  OpBuilder b(f.getBody().front().getTerminator());
  Driver driver(b, ControlFlowPtrRules{});
  auto single = ptr(driver, adds[0]), second = ptr(driver, adds[2]);
  ASSERT_TRUE(succeeded(single));
  ASSERT_TRUE(succeeded(second));
  ComponentMaterializer materializer(b, adds[2]);
  EXPECT_EQ(scalar(materializer, *single->offsets.terms.origin), -128);
  EXPECT_EQ(scalar(materializer, *second->offsets.terms.origin), 128);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(ControlFlowPtrRules, BlockAdvanceChangesOnlyUpdatedCoordinates) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module {tt.func @test(%base:!tt.ptr<i32>) {
    %shape=arith.constant 16:i64
    %stride0=arith.constant 4:i64
    %stride1=arith.constant 1:i64
    %baseoff=arith.constant 5:i32
    %base2=tt.addptr %base,%baseoff:!tt.ptr<i32>,i32
    %max=arith.constant 2147483647:i32
    %zero=arith.constant 0:i32
    %one=arith.constant 1:i32
    %p=tt.make_tensor_ptr %base2,[%shape,%shape],[%stride0,%stride1],[%max,%zero] {order=array<i32:1,0>}:!tt.ptr<tensor<2x4xi32>>
    %same=tt.advance %p,[%zero,%zero]:!tt.ptr<tensor<2x4xi32>>
    %next=tt.advance %same,[%one,%zero]:!tt.ptr<tensor<2x4xi32>>
    tt.return
  }})mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  auto make = *f.getBody().front().getOps<MakeTensorPtrOp>().begin();
  SmallVector<AdvanceOp> advances;
  f.walk([&](AdvanceOp op) { advances.push_back(op); });
  OpBuilder b(f.getBody().front().getTerminator());
  Driver driver(b, ControlFlowPtrRules{});
  auto initial = ptr(driver, make), same = ptr(driver, advances[0]),
       next = ptr(driver, advances[1]);
  ASSERT_TRUE(succeeded(initial));
  ASSERT_TRUE(succeeded(same));
  ASSERT_TRUE(succeeded(next));
  EXPECT_TRUE(
      same->block->coordinates[0].sameIdentity(initial->block->coordinates[0]));
  EXPECT_TRUE(
      next->block->coordinates[1].sameIdentity(initial->block->coordinates[1]));
  EXPECT_FALSE(
      next->block->coordinates[0].sameIdentity(initial->block->coordinates[0]));
  EXPECT_TRUE(next->base.sameIdentity(initial->base));
  EXPECT_TRUE(next->block->strides[0].sameIdentity(initial->block->strides[0]));
  ComponentMaterializer materializer(b, advances[1]);
  EXPECT_EQ(scalar(materializer, next->block->coordinates[0]), 2147483648LL);
  EXPECT_EQ(scalar(materializer, *next->offsets.terms.origin),
            5 + 4 * 2147483648LL);
  EXPECT_FALSE(next->offsets.fullOffsets.type);
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(ControlFlowPtrRules, SymbolicRecipesRespectPlacementAndBindings) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module {tt.func @test(%cond:i1,%base:!tt.ptr<i32>,%x:i64,%y:i64,%boundary:i64) {
    %root=scf.if %cond -> (i64) {scf.yield %x:i64} else {scf.yield %y:i64}
    %one=arith.constant 1:i64
    %derived=arith.addi %boundary,%one:i64
    %next=tt.addptr %base,%derived:!tt.ptr<i32>,i64
    %late=arith.addi %x,%y:i64
    tt.return
  }})mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  auto root = *f.getBody().front().getOps<scf::IfOp>().begin();
  auto next = *f.getBody().front().getOps<AddPtrOp>().begin();
  arith::AddIOp late;
  f.walk([&](arith::AddIOp op) { late = op; });
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  Driver driver(b, ControlFlowPtrRules{});
  auto source = driver.analyze(root.getResult(0));
  ASSERT_TRUE(succeeded(source));
  auto &integer = std::get<SymbolicOffsetInfo>(*source);
  integer.fullOffsets =
      ComponentInfo::boundary(root.getResult(0), 7, b.getI64Type());
  integer.terms.origin = integer.fullOffsets;
  ASSERT_TRUE(succeeded(driver.bind(f.getArgument(4), *source)));
  auto result = ptr(driver, next);
  ASSERT_TRUE(succeeded(result));
  {
    ComponentMaterializer bad(
        b, next,
        [&](const ComponentIdentity &, Type) -> std::optional<OpFoldResult> {
          return late.getResult();
        });
    EXPECT_TRUE(failed(bad.materialize(*result->offsets.terms.origin)));
    EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  }
  ComponentMaterializer materializer(
      b, next,
      [&](const ComponentIdentity &id, Type) -> std::optional<OpFoldResult> {
        if (id.source == root.getResult(0))
          return f.getArgument(2);
        return std::nullopt;
      });
  EXPECT_TRUE(
      succeeded(materializer.materialize(*result->offsets.terms.origin)));
  ASSERT_TRUE(succeeded(
      driver.bind(f.getArgument(4), *driver.findCached(f.getArgument(4)))));
  EXPECT_FALSE(driver.findCached(next));
  driver.clear();
  EXPECT_FALSE(driver.findCached(f.getArgument(4)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(ControlFlowPtrRules, TargetIndexWidthAndRankZeroTensorRemainTyped) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(
      R"mlir(module {tt.func @test(%base:!tt.ptr<i32>,%ix:index,%rank0:tensor<i32>) {
    %one=arith.constant 1:index
    %sum=arith.addi %ix,%one:index
    %unsigned=arith.index_castui %sum:index to i64
    %signed=arith.index_cast %sum:index to i64
    %signedptr=tt.addptr %base,%signed:!tt.ptr<i32>,i64
    %unsignedptr=tt.addptr %base,%unsigned:!tt.ptr<i32>,i64
    tt.return
  }})mlir",
      &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  SmallVector<AddPtrOp> adds;
  f.walk([&](AddPtrOp op) { adds.push_back(op); });
  OpBuilder b(f.getBody().front().getTerminator());
  Driver driver(b, ControlFlowPtrRules({64, 32}));
  auto rank0 = driver.analyze(f.getArgument(2));
  ASSERT_TRUE(succeeded(rank0));
  auto &info = std::get<SymbolicOffsetInfo>(*rank0);
  EXPECT_EQ(info.terms.origin->type, b.getI32Type());
  EXPECT_EQ(info.terms.irregular->type, f.getArgument(2).getType());
  EXPECT_TRUE(succeeded(driver.bind(f.getArgument(2), *rank0)));
  auto signedPtr = ptr(driver, adds[0]), unsignedPtr = ptr(driver, adds[1]);
  ASSERT_TRUE(succeeded(signedPtr));
  ASSERT_TRUE(succeeded(unsignedPtr));
  ComponentMaterializer materializer(b, adds[1], {}, {64, 32});
  auto signedValue = materializer.materialize(*signedPtr->offsets.terms.origin);
  auto unsignedValue =
      materializer.materialize(*unsignedPtr->offsets.terms.origin);
  ASSERT_TRUE(succeeded(signedValue));
  ASSERT_TRUE(succeeded(unsignedValue));
  test::Environment env{{f.getArgument(1), {llvm::APInt(32, 2147483647)}}};
  auto signedLanes = test::evaluate(*signedValue, env, 32),
       unsignedLanes = test::evaluate(*unsignedValue, env, 32);
  ASSERT_TRUE(signedLanes);
  ASSERT_TRUE(unsignedLanes);
  EXPECT_EQ(signedLanes->front().getSExtValue(), -2147483648LL);
  EXPECT_EQ(unsignedLanes->front().getSExtValue(), 2147483648LL);
  Driver unsupportedAddress(b, ControlFlowPtrRules({32, 32}));
  auto source = unsupportedAddress.analyze(f.getArgument(1));
  ASSERT_TRUE(succeeded(source));
  EXPECT_TRUE(std::holds_alternative<SymbolicOffsetInfo>(*source));
  EXPECT_TRUE(succeeded(verify(*module)));
}
TEST(ControlFlowPtrRules, ValidationAndMaterializationPreserveRecipeSharing) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @test(%x:i64, %y:i64) {
      %leaf=arith.divui %x,%y:i64
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  auto leaf = *f.getBody().front().getOps<arith::DivUIOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  constexpr unsigned depth = 12;
  Value value = leaf;
  for (unsigned i = 0; i < depth; ++i)
    value = arith::AddIOp::create(b, f.getLoc(), value, value);
  Driver driver(b, ControlFlowPtrRules{});
  unsigned count = operations(*module);
  auto result = driver.analyze(value);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(operations(*module), count);
  auto original = std::get<SymbolicOffsetInfo>(*result).fullOffsets.expression;
  ASSERT_TRUE(original);
  // Both operands at every level refer to the same immutable recipe.
  auto checkChain = [&](std::shared_ptr<const ComponentExpression> node) {
    unsigned unique = 0;
    while (node) {
      EXPECT_EQ(node->inputs.size(), 2u);
      EXPECT_EQ(node->inputs[0].expression, node->inputs[1].expression);
      node = node->inputs[0].expression;
      ++unique;
    }
    EXPECT_EQ(unique, depth);
  };
  checkChain(original);
  ASSERT_TRUE(succeeded(driver.bind(value, *result)));
  const auto &info = std::get<SymbolicOffsetInfo>(*driver.findCached(value));
  EXPECT_EQ(info.fullOffsets.expression, original);
  checkChain(info.fullOffsets.expression);
  ComponentMaterializer materializer(b, value);
  auto full = materializer.materialize(info.fullOffsets);
  ASSERT_TRUE(succeeded(full));
  test::Environment env{{f.getArgument(0), {llvm::APInt(64, 6)}},
                        {f.getArgument(1), {llvm::APInt(64, 3)}}};
  auto lanes = test::evaluate(*full, env);
  ASSERT_TRUE(lanes);
  EXPECT_EQ(lanes->front(), llvm::APInt(64, 2 << depth));
  count = operations(*module);
  EXPECT_TRUE(succeeded(materializer.materialize(info.fullOffsets)));
  EXPECT_EQ(operations(*module), count);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(ControlFlowPtrRules, IndexCastsRequireSupportedWidthBeforePlanning) {
  for (std::optional<unsigned> width :
       {std::optional<unsigned>{}, std::optional<unsigned>{16},
        std::optional<unsigned>{32}, std::optional<unsigned>{64}}) {
    SCOPED_TRACE(width.value_or(0));
    MLIRContext context;
    load(context);
    auto module = parseSourceString<ModuleOp>(R"mlir(module {
      tt.func @test() {
        %a=arith.constant 127:i8
        %b=arith.constant 1:i8
        %wrap=arith.addi %a,%b:i8
        %ts=tt.splat %wrap:i8 -> tensor<2xi8>
        %si=arith.index_cast %wrap:i8 to index
        %ui=arith.index_castui %wrap:i8 to index
        %tsi=arith.index_cast %ts:tensor<2xi8> to tensor<2xindex>
        %tui=arith.index_castui %ts:tensor<2xi8> to tensor<2xindex>
        %large=arith.constant 4294967295:i64
        %ti=arith.index_cast %large:i64 to index
        %large_tensor=arith.constant dense<4294967295>:tensor<2xi64>
        %tt=arith.index_cast %large_tensor:tensor<2xi64> to tensor<2xindex>
        %signed=arith.index_cast %ti:index to i64
        %unsigned=arith.index_castui %ti:index to i64
        %tsigned=arith.index_cast %tt:tensor<2xindex> to tensor<2xi64>
        %tunsigned=arith.index_castui %tt:tensor<2xindex> to tensor<2xi64>
        tt.return
      }})mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto f = *module->getOps<FuncOp>().begin();
    SmallVector<Value> casts;
    f.walk([&](Operation *op) {
      if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
        casts.push_back(op->getResult(0));
    });
    OpBuilder b(f.getBody().front().getTerminator());
    auto ip = b.saveInsertionPoint();
    Driver driver(b, ControlFlowPtrRules({64, width}));
    unsigned count = operations(*module);
    SmallVector<ControlFlowPtrRules::Result, 0> results;
    for (Value v : casts) {
      auto result = driver.analyze(v);
      ASSERT_TRUE(succeeded(result));
      EXPECT_EQ(std::holds_alternative<SymbolicOffsetInfo>(*result),
                width == 32 || width == 64);
      results.push_back(*result);
    }
    EXPECT_EQ(operations(*module), count);
    EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
    if (width == 32 || width == 64) {
      for (auto [v, result] : llvm::zip(casts, results)) {
        ComponentMaterializer materializer(b, v, {}, {64, width});
        auto full = materializer.materialize(
            std::get<SymbolicOffsetInfo>(result).fullOffsets);
        ASSERT_TRUE(succeeded(full));
        auto expected = test::evaluate(v, test::Environment{}, *width);
        auto actual = test::evaluate(*full, test::Environment{}, *width);
        ASSERT_TRUE(expected);
        ASSERT_TRUE(actual);
        EXPECT_EQ(*actual, *expected);
      }
      const auto &si = std::get<SymbolicOffsetInfo>(results[0]);
      const auto &ui = std::get<SymbolicOffsetInfo>(results[1]);
      ComponentMaterializer materializer(b, casts.back(), {}, {64, width});
      EXPECT_EQ(scalar(materializer, si.fullOffsets), -128);
      EXPECT_EQ(scalar(materializer, ui.fullOffsets), 128);
    }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}
void checkIndexZeroIdentities(unsigned width) {
  MLIRContext context;
  load(context);
  auto module = parseSourceString<ModuleOp>(R"mlir(module {
    tt.func @test(%base:!tt.ptr<i32>, %i:index, %step:i64) {
      %zero=arith.constant 0:index
      %right_add=arith.addi %i,%zero:index
      %left_add=arith.addi %zero,%i:index
      %sub=arith.subi %i,%zero:index
      %right_mul=arith.muli %i,%zero:index
      %left_mul=arith.muli %zero,%i:index
      %signed=arith.index_cast %right_mul:index to i64
      %unsigned=arith.index_castui %left_mul:index to i64
      %parent=tt.addptr %base,%step:!tt.ptr<i32>,i64
      %next=tt.addptr %parent,%signed:!tt.ptr<i32>,i64
      %other=tt.addptr %parent,%unsigned:!tt.ptr<i32>,i64
      tt.return
    }})mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<FuncOp>().begin();
  Value zero = *f.getBody().front().getOps<arith::ConstantOp>().begin();
  SmallVector<Value> unchanged, zeros;
  SmallVector<AddPtrOp> ptrs;
  f.walk([&](Operation *op) {
    if (isa<arith::AddIOp, arith::SubIOp>(op))
      unchanged.push_back(op->getResult(0));
    if (isa<arith::MulIOp, arith::IndexCastOp, arith::IndexCastUIOp>(op))
      zeros.push_back(op->getResult(0));
    if (auto add = dyn_cast<AddPtrOp>(op))
      ptrs.push_back(add);
  });
  ASSERT_EQ(unchanged.size(), 3u);
  ASSERT_EQ(zeros.size(), 4u);
  ASSERT_EQ(ptrs.size(), 3u);
  OpBuilder b(f.getBody().front().getTerminator());
  auto ip = b.saveInsertionPoint();
  unsigned count = operations(*module);
  Driver driver(b, ControlFlowPtrRules({64, width}));
  auto input = driver.analyze(f.getArgument(1));
  ASSERT_TRUE(succeeded(input));
  auto *x = std::get_if<SymbolicOffsetInfo>(&*input);
  ASSERT_TRUE(x);
  auto constant = driver.analyze(zero);
  ASSERT_TRUE(succeeded(constant));
  auto *z = std::get_if<SymbolicOffsetInfo>(&*constant);
  ASSERT_TRUE(z);
  EXPECT_TRUE(z->fullOffsets.isZero());
  EXPECT_TRUE(z->terms.origin->isZero());
  SmallVector<std::pair<Value, SymbolicOffsetInfo>, 0> planned;
  planned.emplace_back(zero, *z);
  for (Value value : unchanged) {
    auto result = driver.analyze(value);
    ASSERT_TRUE(succeeded(result));
    auto *info = std::get_if<SymbolicOffsetInfo>(&*result);
    ASSERT_TRUE(info);
    EXPECT_TRUE(info->fullOffsets.sameIdentity(x->fullOffsets));
    EXPECT_TRUE(info->terms.origin->sameIdentity(*x->terms.origin));
    planned.emplace_back(value, *info);
  }
  for (Value value : zeros) {
    auto result = driver.analyze(value);
    ASSERT_TRUE(succeeded(result));
    auto *info = std::get_if<SymbolicOffsetInfo>(&*result);
    ASSERT_TRUE(info);
    EXPECT_TRUE(info->fullOffsets.isZero());
    EXPECT_TRUE(info->terms.origin->isZero());
    EXPECT_FALSE(info->fullOffsets.expression);
    planned.emplace_back(value, *info);
  }
  auto parent = ptr(driver, ptrs[0]);
  ASSERT_TRUE(succeeded(parent));
  for (auto op : ArrayRef<AddPtrOp>(ptrs).drop_front()) {
    auto next = ptr(driver, op);
    ASSERT_TRUE(succeeded(next));
    EXPECT_TRUE(next->base.sameIdentity(parent->base));
    EXPECT_TRUE(
        next->offsets.fullOffsets.sameIdentity(parent->offsets.fullOffsets));
    EXPECT_TRUE(next->offsets.terms.origin->sameIdentity(
        *parent->offsets.terms.origin));
    EXPECT_TRUE(next->offsets.terms.irregular->sameIdentity(
        *parent->offsets.terms.irregular));
  }
  EXPECT_EQ(operations(*module), count);
  EXPECT_EQ(b.getInsertionBlock(), ip.getBlock());
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());

  ComponentMaterializer materializer(b, ptrs.back(), {}, {64, width});
  test::Environment env{{f.getArgument(1), {llvm::APInt(width, -7, true)}},
                        {f.getArgument(2), {llvm::APInt(64, 11)}}};
  for (const auto &[value, info] : planned) {
    auto full = materializer.materialize(info.fullOffsets);
    ASSERT_TRUE(succeeded(full));
    auto actual = test::evaluate(*full, env, width);
    auto expected = test::evaluate(value, env, width);
    ASSERT_TRUE(actual);
    ASSERT_TRUE(expected);
    EXPECT_EQ(*actual, *expected);
  }
  auto next = ptr(driver, ptrs.back());
  ASSERT_TRUE(succeeded(next));
  auto offset = materializer.materialize(next->offsets.fullOffsets);
  ASSERT_TRUE(succeeded(offset));
  auto actual = test::evaluate(*offset, env, width);
  ASSERT_TRUE(actual);
  EXPECT_EQ(actual->front(), llvm::APInt(64, 11));
  EXPECT_EQ(b.getInsertionBlock(), ip.getBlock());
  EXPECT_EQ(b.getInsertionPoint(), ip.getPoint());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(ControlFlowPtrRules, Index32ZeroPreservesComponentIdentities) {
  checkIndexZeroIdentities(32);
}
TEST(ControlFlowPtrRules, Index64ZeroPreservesComponentIdentities) {
  checkIndexZeroIdentities(64);
}
} // namespace
