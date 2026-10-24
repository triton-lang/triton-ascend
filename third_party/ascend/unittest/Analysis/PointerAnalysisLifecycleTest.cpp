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

#include "PointerAnalysisTestUtils.h"
#include "TritonMemoryAccess/PointerAnalysis.h"
#include "TritonMemoryAccess/PointerAnalysisDriver.h"
#include "TritonMemoryAccess/PointerAnalysisTransfer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pointer;

namespace {

namespace pointer_detail = mlir::triton::pointer::detail;

struct ExtraResult {
  pointer_detail::AnalysisResult basic;
  Value extra;
};

struct ExtraRules {
  using Result = ExtraResult;
  pointer_detail::PointerRules basic{AnalysisOptions{}};

  FailureOr<SmallVector<pointer_detail::AnalysisRequest>>
  collectInputs(pointer_detail::AnalysisRequest request) const {
    if (request.kind == pointer_detail::FirstExtensionRequest) {
      if (auto div = request.value.getDefiningOp<arith::DivSIOp>())
        return SmallVector<pointer_detail::AnalysisRequest>{
            {div.getLhs(), pointer_detail::SourceInteger},
            {div.getRhs(), pointer_detail::SourceInteger}};
      if (!request.value.getDefiningOp<arith::AddIOp>())
        return failure();
      return SmallVector<pointer_detail::AnalysisRequest>{
          {request.value, pointer_detail::SourceInteger}};
    }
    return basic.collectInputs(request);
  }

  FailureOr<Result> transfer(pointer_detail::AnalysisRequest request,
                             ArrayRef<Result> inputs,
                             OpBuilder &builder) const {
    if (request.kind == pointer_detail::FirstExtensionRequest) {
      if (request.value.getDefiningOp<arith::DivSIOp>()) {
        if (inputs.size() != 2 || !inputs[0].basic.offset ||
            !isa<Value>(inputs[0].basic.offset->uniformOffset))
          return failure();
        // Deliberately independent: this request has no basic pointer view.
        return Result{{}, cast<Value>(inputs[0].basic.offset->uniformOffset)};
      }
      if (inputs.size() != 1 || !inputs[0].basic.offset ||
          !isa<Value>(inputs[0].basic.offset->completeOffset))
        return failure();
      return Result{inputs[0].basic,
                    cast<Value>(inputs[0].basic.offset->completeOffset)};
    }
    SmallVector<pointer_detail::AnalysisResult, 1> operands;
    for (const Result &input : inputs)
      operands.push_back(input.basic);
    auto result = basic.transfer(request, operands, builder);
    if (failed(result))
      return failure();
    Result combined;
    combined.basic = *result;
    return combined;
  }

  LogicalResult
  mapResultValues(Result &result,
                  llvm::function_ref<FailureOr<Value>(Value)> mapper) const {
    if (failed(basic.mapResultValues(result.basic, mapper)))
      return failure();
    if (result.extra) {
      auto replacement = mapper(result.extra);
      if (failed(replacement))
        return failure();
      result.extra = *replacement;
    }
    return success();
  }
};

TEST(PointerAnalysisLifecycle, ExtensionSharesTheSourceNode) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @extension(%x: i32, %y: i32) {
        %sum = arith.addi %x, %y : i32
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto sum = *func.getBody().front().getOps<arith::AddIOp>().begin();
  OpBuilder builder(&context);
  pointer_detail::AnalysisDriver<ExtraRules> driver(builder, ExtraRules{});
  auto extended = driver.analyze({sum, pointer_detail::FirstExtensionRequest});
  ASSERT_TRUE(succeeded(extended));
  EXPECT_EQ(extended->extra, sum.getResult());
  EXPECT_EQ(driver.getVisitCount({sum, pointer_detail::SourceInteger}), 1u);
  EXPECT_EQ(driver.getVisitCount({sum, pointer_detail::FirstExtensionRequest}),
            1u);
  ASSERT_TRUE(succeeded(driver.analyze({sum, pointer_detail::SourceInteger})));
  ASSERT_TRUE(
      succeeded(driver.analyze({sum, pointer_detail::FirstExtensionRequest})));
  EXPECT_EQ(driver.getVisitCount({sum, pointer_detail::SourceInteger}), 1u);
  EXPECT_EQ(driver.getVisitCount({sum, pointer_detail::FirstExtensionRequest}),
            1u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle, ShapeRulesRequireSourceRange) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @shape_inputs(%x: i32, %lane: tensor<1xi32>) {
      %splat = tt.splat %x : i32 -> tensor<4xi32>
      %broadcast = tt.broadcast %lane : tensor<1xi32> -> tensor<4xi32>
      %expand = tt.expand_dims %lane {axis = 0 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  OpBuilder builder(&context);
  pointer_detail::PointerRules rules({});
  for (Operation &op : func.getBody().front()) {
    if (!isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp>(&op))
      continue;
    builder.setInsertionPointAfter(&op);
    auto offset = makeOpaqueOffset(op.getOperand(0), builder);
    ASSERT_TRUE(succeeded(offset));
    // An extension can supply an offset without the source range needed by
    // basic rules. Report an unsupported input instead of dereferencing it.
    SmallVector<pointer_detail::AnalysisResult, 1> inputs(1);
    inputs.front().offset = *offset;
    EXPECT_TRUE(failed(rules.transfer(
        {op.getResult(0), pointer_detail::SourceInteger}, inputs, builder)));
  }
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle, InvalidBindingAndMappingLeaveCacheIntact) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @boundaries(%x: i32, %equivalent: i32, %wrong: i64, %condition: i1, %replacement: i32) {
        scf.if %condition {
          %inner = arith.addi %x, %equivalent : i32
        }
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto conditional = *func.getBody().front().getOps<scf::IfOp>().begin();
  auto inner =
      *conditional.getThenRegion().front().getOps<arith::AddIOp>().begin();
  OpBuilder builder(&context);
  PointerAnalysis analysis(builder);
  Value x = func.getArgument(0);
  Value equivalent = func.getArgument(1);
  Value wrong = func.getArgument(2);
  Value replacement = func.getArgument(4);
  auto external = analysis.analyzeOffset(x);
  ASSERT_TRUE(succeeded(external));
  ASSERT_TRUE(succeeded(analysis.analyzeOffset(inner)));
  EXPECT_TRUE(analysis.findCachedOffset(x));
  EXPECT_TRUE(analysis.findCachedOffset(inner));

  OffsetComponents invalid = *external;
  invalid.domain = ArithmeticDomain::ElementAddress;
  EXPECT_TRUE(failed(analysis.bindOffset(x, invalid)));
  EXPECT_TRUE(analysis.findCachedOffset(x));
  IRMapping invalidMapping;
  invalidMapping.map(x, wrong);
  EXPECT_TRUE(
      failed(analysis.remapAndForgetScope(conditional, invalidMapping)));
  EXPECT_TRUE(analysis.findCachedOffset(x));
  EXPECT_TRUE(analysis.findCachedOffset(inner));

  IRMapping mapping;
  mapping.map(x, replacement);
  ASSERT_TRUE(succeeded(analysis.remapAndForgetScope(conditional, mapping)));
  EXPECT_FALSE(analysis.findCachedOffset(inner));
  const OffsetComponents *mapped = analysis.findCachedOffset(replacement);
  ASSERT_TRUE(mapped);
  EXPECT_EQ(cast<Value>(mapped->completeOffset), replacement);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle, NumericHelpersCanBeCleanedAfterSession) {
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
  PointerAnalysis analysis(builder);
  ASSERT_TRUE(succeeded(analysis.analyzeAddressDelta(func.getArgument(0))));
  unsigned before = 0;
  func.walk([&](arith::ExtSIOp) { ++before; });
  EXPECT_EQ(before, 1u);
  analysis.clear();
  IRRewriter rewriter(&context);
  eraseDeadPointerArithmetic(func, rewriter);
  unsigned after = 0;
  func.walk([&](arith::ExtSIOp) { ++after; });
  EXPECT_EQ(after, 0u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle, OffsetAndPointerRebindingRecomputesConsumers) {
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
  PointerAnalysis analysis(b);
  EXPECT_TRUE(failed(analysis.analyzePointer(f.getArgument(1))));
  auto scalar = analysis.analyzeOffset(opaque);
  ASSERT_TRUE(succeeded(scalar));
  for (int64_t n : {5, -8}) {
    scalar->completeOffset = scalar->uniformOffset = b.getI32IntegerAttr(n);
    ASSERT_TRUE(succeeded(analysis.bindOffset(opaque, *scalar)));
    PointerComponents p;
    p.base = f.getArgument(0);
    p.elementType = b.getI32Type();
    p.addressSpace = 1;
    p.offsets.valueType = RankedTensorType::get({4}, b.getI64Type());
    p.offsets.domain = ArithmeticDomain::ElementAddress;
    p.offsets.shape = {4};
    p.offsets.completeOffset = DenseIntElementsAttr::get(
        cast<RankedTensorType>(p.offsets.valueType), llvm::APInt(64, n, true));
    p.offsets.uniformOffset = b.getI64IntegerAttr(n);
    p.offsets.axes = {AxisKind::Invariant};
    p.offsets.strides = {b.getI64IntegerAttr(0)};
    ASSERT_TRUE(succeeded(analysis.bindPointer(f.getArgument(1), p)));
    auto derived = analysis.analyzePointer(next);
    ASSERT_TRUE(succeeded(derived));
    auto values =
        test::evaluate(derived->offsets.completeOffset,
                       test::Environment{{opaque, {llvm::APInt(32, n, true)}}});
    ASSERT_TRUE(values);
    for (auto value : *values)
      EXPECT_EQ(value, llvm::APInt(64, 2 * n + 1, true));
    EXPECT_FALSE(analysis.findCachedOffset(f.getArgument(2)));
    EXPECT_FALSE(analysis.findCachedOffset(f.getArgument(3)));
    auto bad = *scalar;
    bad.domain = ArithmeticDomain::ElementAddress;
    EXPECT_TRUE(failed(analysis.bindOffset(opaque, bad)));
    auto badPointer = p;
    badPointer.base = f.getArgument(2);
    EXPECT_TRUE(failed(analysis.bindPointer(f.getArgument(1), badPointer)));
    auto retained = analysis.analyzeOffset(sum);
    ASSERT_TRUE(succeeded(retained));
    auto actual = test::evaluate(retained->uniformOffset);
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->front(), llvm::APInt(32, n + 1, true));
    EXPECT_TRUE(succeeded(verify(*module)));
  }
  analysis.clear();
  EXPECT_FALSE(analysis.findCachedOffset(sum));
  EXPECT_TRUE(failed(analysis.analyzePointer(next)));
  auto unbound = analysis.analyzeOffset(opaque);
  ASSERT_TRUE(succeeded(unbound));
  EXPECT_EQ(cast<Value>(unbound->completeOffset), opaque);
}

TEST(PointerAnalysisLifecycle, CloneEraseRemapsAllPublicFieldsAtomically) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @clone(%cond: i1, %base: !tt.ptr<i32>, %x: i64, %o: tensor<4xi64>, %offset: tensor<4xi64>, %pointer: tensor<4x!tt.ptr<i32>>, %block: !tt.ptr<tensor<4xi32>>) {
      %r:3 = scf.if %cond -> (tensor<4xi64>, i64, !tt.ptr<i32>) {
        %inner = arith.addi %x, %x : i64
        scf.yield %o, %inner, %base : tensor<4xi64>, i64, !tt.ptr<i32>
      } else {
        scf.yield %o, %x, %base : tensor<4xi64>, i64, !tt.ptr<i32>
      }
      %d = arith.trunci %r#1 : i64 to i32
      %next = tt.advance %block, [%d] : !tt.ptr<tensor<4xi32>>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto root = *f.getBody().front().getOps<scf::IfOp>().begin();
  Value inner = *root.getThenRegion().front().getOps<arith::AddIOp>().begin();
  auto next = *f.getBody().front().getOps<triton::AdvanceOp>().begin();
  OpBuilder b(f.getBody().front().getTerminator());
  PointerAnalysis analysis(b);
  OffsetComponents offset;
  offset.valueType = f.getArgument(4).getType();
  offset.shape = {4};
  offset.completeOffset = root.getResult(0);
  offset.uniformOffset = root.getResult(1);
  offset.strides = {root.getResult(1)};
  offset.axes = {AxisKind::Structured};
  ASSERT_TRUE(succeeded(analysis.bindOffset(f.getArgument(4), offset)));
  PointerComponents pointer;
  pointer.base = root.getResult(2);
  pointer.elementType = b.getI32Type();
  pointer.addressSpace = 1;
  pointer.offsets = offset;
  pointer.offsets.domain = ArithmeticDomain::ElementAddress;
  ASSERT_TRUE(succeeded(analysis.bindPointer(f.getArgument(5), pointer)));
  PointerComponents block;
  block.block.emplace();
  block.base = root.getResult(2);
  block.offsets.valueType = RankedTensorType::get({4}, b.getI64Type());
  block.offsets.domain = ArithmeticDomain::ElementAddress;
  block.offsets.shape = {4};
  block.offsets.uniformOffset = root.getResult(1);
  block.offsets.axes = {AxisKind::Structured};
  block.block->order = {0};
  block.elementType = b.getI32Type();
  block.addressSpace = 1;
  block.block->parentShape = {root.getResult(1)};
  block.block->coordinateOffsets = {root.getResult(1)};
  block.offsets.strides = {root.getResult(1)};
  block.block->coordinateRanges = {llvm::ConstantRange::getFull(64)};
  ASSERT_TRUE(succeeded(analysis.bindPointer(f.getArgument(6), block)));
  ASSERT_TRUE(succeeded(analysis.analyzeOffset(inner)));
  ASSERT_TRUE(succeeded(analysis.analyzeOffset(f.getArgument(2))));
  ASSERT_TRUE(succeeded(analysis.analyzeOffset(root.getResult(0))));
  ASSERT_TRUE(succeeded(analysis.analyzeBlockAdvanceDelta(next)));
  pointer_detail::AnalysisDriver<pointer_detail::PointerRules> deltaDriver(
      b, pointer_detail::PointerRules({}));
  pointer_detail::AnalysisResult deltaBinding;
  deltaBinding.advanceDelta = BlockAdvanceDelta{
      {root.getResult(1)}, {llvm::ConstantRange::getFull(64)}};
  deltaDriver.bindBoundary({next, pointer_detail::AdvanceDeltaRequest},
                           deltaBinding);
  IRMapping bad;
  bad.map(root.getResult(1), f.getArgument(0));
  EXPECT_TRUE(failed(analysis.remapAndForgetScope(root, bad)));
  EXPECT_TRUE(analysis.findCachedOffset(inner));
  EXPECT_EQ(
      cast<Value>(analysis.findCachedOffset(f.getArgument(4))->uniformOffset),
      root.getResult(1));
  b.setInsertionPoint(root);
  IRMapping mapping;
  Operation *clone = b.clone(*root, mapping);
  for (auto pair : llvm::zip(root->getResults(), clone->getResults()))
    mapping.map(std::get<0>(pair), std::get<1>(pair));
  ASSERT_TRUE(succeeded(analysis.remapAndForgetScope(root, mapping)));
  ASSERT_TRUE(succeeded(deltaDriver.remapAndForgetScope(root, mapping)));
  EXPECT_FALSE(analysis.findCachedOffset(inner));
  EXPECT_FALSE(analysis.findCachedOffset(root.getResult(0)));
  root->replaceAllUsesWith(clone->getResults());
  root.erase();
  auto mappedOffset = analysis.analyzeOffset(f.getArgument(4));
  ASSERT_TRUE(succeeded(mappedOffset));
  EXPECT_EQ(cast<Value>(mappedOffset->completeOffset), clone->getResult(0));
  EXPECT_EQ(cast<Value>(mappedOffset->uniformOffset), clone->getResult(1));
  EXPECT_EQ(cast<Value>(mappedOffset->strides[0]), clone->getResult(1));
  auto mappedPointer = analysis.analyzePointer(f.getArgument(5));
  ASSERT_TRUE(succeeded(mappedPointer));
  EXPECT_EQ(mappedPointer->base, clone->getResult(2));
  EXPECT_EQ(cast<Value>(mappedPointer->offsets.completeOffset),
            clone->getResult(0));
  auto mappedBlock = analysis.analyzePointer(f.getArgument(6));
  ASSERT_TRUE(succeeded(mappedBlock));
  EXPECT_EQ(mappedBlock->base, clone->getResult(2));
  for (auto field :
       {mappedBlock->block->parentShape[0],
        mappedBlock->block->coordinateOffsets[0],
        mappedBlock->offsets.strides[0], mappedBlock->offsets.uniformOffset})
    EXPECT_EQ(cast<Value>(field), clone->getResult(1));
  EXPECT_TRUE(analysis.findCachedOffset(f.getArgument(2)));
  ASSERT_TRUE(analysis.findCachedBlockAdvanceDelta(next));
  auto mappedDelta =
      deltaDriver.analyze({next, pointer_detail::AdvanceDeltaRequest});
  ASSERT_TRUE(succeeded(mappedDelta));
  EXPECT_EQ(cast<Value>(mappedDelta->advanceDelta->coordinateDeltas[0]),
            clone->getResult(1));
  b.setInsertionPoint(f.getBody().front().getTerminator());
  ASSERT_TRUE(succeeded(buildAddressView(*mappedBlock, b, clone->getLoc())));
  ASSERT_TRUE(succeeded(analysis.analyzePointer(next)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle,
     SiblingConsumersReuseHelpersAndPreserveOriginalIR) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      triton::TritonDialect>();
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
    pointer_detail::AnalysisDriver<pointer_detail::PointerRules> driver(
        b, pointer_detail::PointerRules({}));
    if (sourceFirst) {
      ASSERT_TRUE(
          succeeded(driver.analyze({twice, pointer_detail::SourceInteger})));
      ASSERT_TRUE(succeeded(verify(*module)));
    }
    for (auto p : pointers) {
      ASSERT_TRUE(succeeded(driver.analyze({p, pointer_detail::PointerValue})));
      ASSERT_TRUE(succeeded(verify(*module)));
    }
    ASSERT_TRUE(
        succeeded(driver.analyze({twice, pointer_detail::SourceInteger})));
    ASSERT_TRUE(
        succeeded(driver.analyze({twice, pointer_detail::AddressDelta})));
    for (const auto &[op, operands] : snapshot) {
      EXPECT_EQ(llvm::to_vector(op->getOperands()), operands);
      for (Value result : op->getResults())
        EXPECT_LE(driver.getVisitCount({result, pointer_detail::SourceInteger}),
                  1u);
    }
    EXPECT_EQ(driver.getVisitCount({twice, pointer_detail::SourceInteger}), 1u);
    module->walk([&](Operation *op) {
      if (llvm::any_of(snapshot,
                       [&](auto &entry) { return entry.first == op; }))
        return;
      EXPECT_TRUE(isMemoryEffectFree(op));
      EXPECT_TRUE((op->getName().getDialectNamespace() == "arith" ||
                   isa<triton::SplatOp, triton::BroadcastOp,
                       triton::ExpandDimsOp, triton::MakeRangeOp>(op)));
    });
    auto address = driver.analyze({twice, pointer_detail::AddressDelta});
    ASSERT_TRUE(succeeded(address));
    Value liveHelper = cast<Value>(address->offset->uniformOffset);
    b.setInsertionPoint(f.getBody().front().getTerminator());
    Value narrowed =
        b.create<arith::TruncIOp>(f.getLoc(), b.getI32Type(), liveHelper);
    auto store = b.create<triton::StoreOp>(
        f.getLoc(), f.getArgument(0), narrowed, Value(),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    driver.clear();
    IRRewriter rewriter(&context);
    eraseDeadPointerArithmetic(f, rewriter);
    EXPECT_EQ(store.getValue(), narrowed);
    EXPECT_EQ(narrowed.getDefiningOp()->getOperand(0), liveHelper);
    EXPECT_EQ(f.getBody().front().getTerminator()->getOperand(0), twice);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(PointerAnalysisLifecycle,
     ExtensionAddsOperandsAndRemapsIndependentResult) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @extension(%x: i32, %y: i32, %cond: i1, %unknown: tensor<4x!tt.ptr<i32>>) {
      %r = scf.if %cond -> i32 {
        scf.yield %x : i32
      } else {
        scf.yield %y : i32
      }
      %s = tt.splat %r : i32 -> tensor<4xi32>
      %a = arith.divsi %s, %s : tensor<4xi32>
      %b = arith.divsi %s, %s : tensor<4xi32>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto root = *f.getBody().front().getOps<scf::IfOp>().begin();
  SmallVector<arith::DivSIOp> divs;
  f.walk([&](arith::DivSIOp op) { divs.push_back(op); });
  OpBuilder b(&context);
  pointer_detail::AnalysisDriver<ExtraRules> driver(b, ExtraRules{});
  ASSERT_TRUE(
      succeeded(driver.analyze({divs[0], pointer_detail::SourceInteger})));
  EXPECT_EQ(
      driver.getVisitCount({root.getResult(0), pointer_detail::SourceInteger}),
      0u);
  for (auto div : divs) {
    auto extra = driver.analyze({div, pointer_detail::FirstExtensionRequest});
    ASSERT_TRUE(succeeded(extra));
    EXPECT_FALSE(extra->basic.offset);
    EXPECT_EQ(extra->extra, root.getResult(0));
    ASSERT_TRUE(
        succeeded(driver.analyze({div, pointer_detail::SourceInteger})));
  }
  EXPECT_EQ(
      driver.getVisitCount({root.getResult(0), pointer_detail::SourceInteger}),
      1u);
  EXPECT_TRUE(failed(driver.analyze(
      {f.getArgument(0), pointer_detail::FirstExtensionRequest})));
  EXPECT_TRUE(succeeded(
      driver.analyze({f.getArgument(0), pointer_detail::SourceInteger})));
  EXPECT_TRUE(
      failed(driver.analyze({f.getArgument(3), pointer_detail::PointerValue})));
  b.setInsertionPoint(root);
  IRMapping mapping;
  Operation *clone = b.clone(*root, mapping);
  mapping.map(root.getResult(0), clone->getResult(0));
  ASSERT_TRUE(succeeded(driver.remapAndForgetScope(root, mapping)));
  root->replaceAllUsesWith(clone->getResults());
  root.erase();
  auto extra = driver.analyze({divs[0], pointer_detail::FirstExtensionRequest});
  ASSERT_TRUE(succeeded(extra));
  EXPECT_EQ(extra->extra, clone->getResult(0));
  b.setInsertionPoint(f.getBody().front().getTerminator());
  b.create<arith::AddIOp>(f.getLoc(), extra->extra, extra->extra);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle, FailedAnalysisLeavesOnlyCleanablePureHelpers) {
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
  PointerAnalysis analysis(b);
  EXPECT_TRUE(failed(analysis.analyzePointer(
      f.getBody().front().getTerminator()->getOperand(0))));
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
  analysis.clear();
  IRRewriter rewriter(&context);
  eraseDeadPointerArithmetic(f, rewriter);
  for (auto &[op, operands] : original)
    EXPECT_EQ(llvm::to_vector(op->getOperands()), operands);
  unsigned extensions = 0;
  f.walk([&](arith::ExtSIOp) { ++extensions; });
  EXPECT_EQ(extensions, 0u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle, ChangingIntegerIterArgKeepsCurrentValue) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      triton::TritonDialect>();
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
  PointerAnalysis analysis(b);
  auto state = analysis.analyzeOffset(current);
  ASSERT_TRUE(succeeded(state));
  EXPECT_EQ(cast<Value>(state->completeOffset), current);
  EXPECT_EQ(cast<Value>(state->uniformOffset), current);
  EXPECT_FALSE(analysis.findCachedOffset(f.getArgument(0)));
  auto address = analysis.analyzeAddressDelta(current);
  ASSERT_TRUE(succeeded(address));
  test::Environment env{{current, {llvm::APInt(32, -9, true)}},
                        {f.getArgument(0), {llvm::APInt(32, 100)}}};
  auto actual = test::evaluate(address->completeOffset, env);
  ASSERT_TRUE(actual);
  EXPECT_EQ(actual->front(), llvm::APInt(64, -9, true));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle, RankZeroExtractionIsPureAndCleanable) {
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
  PointerAnalysis analysis(b);
  ASSERT_TRUE(succeeded(analysis.analyzeAddressDelta(f.getArgument(1))));
  auto live = analysis.analyzeAddressDelta(f.getArgument(2));
  ASSERT_TRUE(succeeded(live));
  Value scalar = cast<Value>(live->uniformOffset);
  f.walk([&](tensor::ExtractOp op) { EXPECT_TRUE(isMemoryEffectFree(op)); });
  auto store = b.create<triton::StoreOp>(f.getLoc(), f.getArgument(0), scalar,
                                         Value(), triton::CacheModifier::NONE,
                                         triton::EvictionPolicy::NORMAL);
  EXPECT_TRUE(succeeded(verify(*module)));
  analysis.clear();
  IRRewriter rewriter(&context);
  eraseDeadPointerArithmetic(f, rewriter);
  EXPECT_TRUE(f.getArgument(1).use_empty());
  EXPECT_FALSE(f.getArgument(2).use_empty());
  EXPECT_EQ(store.getValue(), scalar);
  unsigned extracts = 0;
  f.walk([&](tensor::ExtractOp) { ++extracts; });
  EXPECT_EQ(extracts, 1u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(PointerAnalysisLifecycle, SessionRejectsUnsupportedAddressWidths) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @options(%base: !tt.ptr<i32>, %x: i32, %index: index) {
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
  PointerAnalysis reference(builder);
  auto pointer = reference.analyzePointer(base);
  ASSERT_TRUE(succeeded(pointer));
  auto block = reference.analyzePointer(make);
  ASSERT_TRUE(succeeded(block));
  for (unsigned addressWidth : {0u, 32u, 64u, 128u})
    for (unsigned indexWidth : {32u, 64u}) {
      PointerAnalysis analysis(builder, {addressWidth, indexWidth});
      const bool supported = addressWidth == 64;
      auto integer = analysis.analyzeOffset(x);
      ASSERT_TRUE(succeeded(integer));
      ASSERT_TRUE(succeeded(analysis.bindOffset(x, *integer)));
      ASSERT_TRUE(succeeded(analysis.analyzeOffset(f.getArgument(2))));
      auto *saved = analysis.findCachedOffset(x);
      ASSERT_TRUE(saved);
      auto *savedIndex = analysis.findCachedOffset(f.getArgument(2));
      ASSERT_TRUE(savedIndex);
      for (Value p : {base, splat, add})
        EXPECT_EQ(succeeded(analysis.analyzePointer(p)), supported);
      EXPECT_EQ(succeeded(analysis.bindPointer(base, *pointer)), supported);
      // Rejected address bindings must not invalidate the SourceInteger cache.
      if (!supported) {
        EXPECT_EQ(analysis.findCachedOffset(x), saved);
        EXPECT_EQ(analysis.findCachedOffset(f.getArgument(2)), savedIndex);
      }
      EXPECT_EQ(succeeded(analysis.analyzePointer(base)), supported);
      EXPECT_EQ(succeeded(analysis.analyzeAddressDelta(x)), supported);
      EXPECT_EQ(analysis.findCachedAddressDelta(x) != nullptr, supported);
      EXPECT_EQ(succeeded(analysis.analyzePointer(make)), supported);
      EXPECT_EQ(succeeded(analysis.analyzeBlockAdvanceDelta(advance)),
                supported);
      EXPECT_EQ(succeeded(analysis.bindPointer(make, *block)), supported);
      if (!supported) {
        EXPECT_EQ(analysis.findCachedOffset(x), saved);
        EXPECT_EQ(analysis.findCachedOffset(f.getArgument(2)), savedIndex);
      }
      EXPECT_EQ(succeeded(analysis.analyzePointer(make)), supported);
      EXPECT_EQ(succeeded(analysis.analyzePointer(advance)), supported);
      EXPECT_EQ(analysis.findCachedPointer(make) != nullptr, supported);
      // A successful binding invalidates derived cache; query again first.
      EXPECT_EQ(succeeded(analysis.analyzeBlockAdvanceDelta(advance)),
                supported);
      EXPECT_EQ(analysis.findCachedBlockAdvanceDelta(advance) != nullptr,
                supported);
      EXPECT_TRUE(succeeded(verify(*module)));
    }
}

TEST(PointerAnalysisLifecycle, UnseenBindingsPreserveCompletedRequests) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @bindings(%x: i64, %unseen: i64) {
      %sum = arith.addi %x, %x : i64
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto sum = *f.getBody().front().getOps<arith::AddIOp>().begin();
  OpBuilder builder(&context);
  PointerAnalysis analysis(builder);
  ASSERT_TRUE(succeeded(analysis.analyzeOffset(sum)));
  const auto *cached = analysis.findCachedOffset(sum);
  ASSERT_NE(cached, nullptr);
  auto unseen = makeOpaqueOffset(f.getArgument(1), builder);
  ASSERT_TRUE(succeeded(unseen));
  ASSERT_TRUE(succeeded(analysis.bindOffset(f.getArgument(1), *unseen)));
  EXPECT_EQ(analysis.findCachedOffset(sum), cached);
  // An observed leaf can already contribute to derived results even without an
  // explicit old binding. Installing its first binding must invalidate them.
  auto observed = makeOpaqueOffset(f.getArgument(0), builder);
  ASSERT_TRUE(succeeded(observed));
  ASSERT_TRUE(succeeded(analysis.bindOffset(f.getArgument(0), *observed)));
  EXPECT_EQ(analysis.findCachedOffset(sum), nullptr);
  ASSERT_TRUE(succeeded(analysis.analyzeOffset(sum)));
  // Replacing an existing binding retains the conservative invalidation rule.
  ASSERT_TRUE(succeeded(analysis.bindOffset(f.getArgument(1), *unseen)));
  EXPECT_EQ(analysis.findCachedOffset(sum), nullptr);
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
