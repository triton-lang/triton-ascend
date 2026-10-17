// Pointer-analysis coverage for the indirect and partially structured stages.
#include "AscendModel/Analysis/StageMemoryPatternAnalysis.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "CostModelTestUtils.h"
#include "StageIRTestUtils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::test;

namespace {
enum class IndexKind { Range, Loaded, Scalar, Computed, PointerArgument };

// Share address construction, not expected classifications or measured costs.
static OwningOpRef<ModuleOp> addressModule(IRTestContext &context,
                                           ArrayRef<int64_t> shape,
                                           ArrayRef<IndexKind> kinds,
                                           int64_t columnStep = 1,
                                           bool store = false) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto i32 = builder.getI32Type();
  auto f32 = builder.getF32Type();
  auto indexPtr = mlir::triton::PointerType::get(i32, 1);
  auto dataPtr = mlir::triton::PointerType::get(f32, 1);
  auto fullType = RankedTensorType::get(shape, i32);
  Type indicesType =
      kinds.front() == IndexKind::PointerArgument
          ? Type(RankedTensorType::get({shape.front()}, indexPtr))
          : Type(indexPtr);
  OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
  builder.setInsertionPointToStart(module->getBody());
  auto function = builder.create<func::FuncOp>(
      loc, "address",
      builder.getFunctionType(
          {indicesType, dataPtr, RankedTensorType::get(shape, f32)}, {}));
  auto &body = *function.addEntryBlock();
  builder.setInsertionPointToStart(&body);
  auto op = [&](StringRef name, Type result, ValueRange operands,
                ArrayRef<NamedAttribute> attrs = {}) {
    OperationState state(loc, name);
    state.addTypes(result);
    state.addOperands(operands);
    state.addAttributes(attrs);
    return builder.create(state)->getResult(0);
  };
  Value offset;
  for (unsigned axis = 0; axis < shape.size(); ++axis) {
    auto type = RankedTensorType::get({shape[axis]}, i32);
    Value index = op(
        "tt.make_range", type, {},
        {builder.getNamedAttr("start", builder.getI32IntegerAttr(0)),
         builder.getNamedAttr("end", builder.getI32IntegerAttr(shape[axis]))});
    if (kinds[axis] == IndexKind::Scalar)
      index = op("tt.splat", type, op("tt.load", i32, body.getArgument(0)));
    else if (kinds[axis] == IndexKind::Computed)
      index = builder.create<arith::MulIOp>(loc, index, index);
    else if (kinds[axis] != IndexKind::Range) {
      Value ptr = body.getArgument(0);
      if (kinds[axis] == IndexKind::Loaded) {
        auto ptrType = RankedTensorType::get({shape[axis]}, indexPtr);
        ptr = op("tt.addptr", ptrType, {op("tt.splat", ptrType, ptr), index});
      }
      index = op("tt.load", type, ptr);
    }
    for (unsigned dim = 0; dim < shape.size(); ++dim) {
      if (dim == axis)
        continue;
      SmallVector<int64_t> expanded(
          cast<RankedTensorType>(index.getType()).getShape());
      expanded.insert(expanded.begin() + dim, 1);
      index = op("tt.expand_dims", RankedTensorType::get(expanded, i32), index,
                 builder.getNamedAttr("axis", builder.getI32IntegerAttr(dim)));
    }
    if (index.getType() != fullType)
      index = op("tt.broadcast", fullType, index);
    int64_t stride = columnStep;
    for (unsigned dim = axis + 1; dim < shape.size(); ++dim)
      stride *= shape[dim];
    auto scale = builder.create<arith::ConstantOp>(
        loc,
        DenseElementsAttr::get(fullType, builder.getI32IntegerAttr(stride)));
    index = builder.create<arith::MulIOp>(loc, index, scale);
    if (offset)
      offset = builder.create<arith::AddIOp>(loc, offset, index);
    else
      offset = index;
  }
  auto ptrType = RankedTensorType::get(shape, dataPtr);
  auto ptr = op("tt.addptr", ptrType,
                {op("tt.splat", ptrType, body.getArgument(1)), offset});
  if (store) {
    OperationState state(loc, "tt.store");
    state.addOperands({ptr, body.getArgument(2)});
    builder.create(state);
  } else
    op("tt.load", RankedTensorType::get(shape, f32), ptr);
  builder.create<func::ReturnOp>(loc);
  return module;
}

static const LogicalStage *payloadStage(const StagePartition &partition,
                                        ArrayRef<int64_t> shape,
                                        bool store = false) {
  for (const auto &stage : partition.stages)
    for (auto *op : stage.operations) {
      if (op->getName().getStringRef() != (store ? "tt.store" : "tt.load"))
        continue;
      auto type = dyn_cast<RankedTensorType>(
          store ? op->getOperand(1).getType() : op->getResult(0).getType());
      if (type && type.getElementType().isF32() && type.getShape() == shape)
        return &stage;
    }
  return nullptr;
}
} // namespace

TEST(StageAddressPatternTest, PartialContinuousStageUsesUpstreamAggregation) {
  LogicalStage stage =
      logicalStage("partial", StageCostModelKind::PartialContinuousTileMemory,
                   StageScheduleKind::IndependentPipelined);
  stage.features.hasPartialContinuousMemory = true;
  stage.features.hasIndirectMemory = true;
  stage.workload.operationCounts.clear();
  stage.workload.loadBytes = 320.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.loadWarpInstructions = 10.0;
  stage.workload.storeWarpInstructions = 10.0;
  stage.workload.partialContinuousLoadRows = 10.0;
  stage.workload.partialContinuousStoreRows = 10.0;
  stage.workload.partialContinuousLoadBytes = 320.0;
  stage.workload.partialContinuousStoreBytes = 160.0;
  stage.workload.partialContinuousLoadWarpInstructions = 10.0;
  stage.workload.partialContinuousStoreWarpInstructions = 10.0;
  auto table = evaluateOneStage(stage);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  for (const StageImplementationCost &implementation :
       table->stages.front().implementations) {
    const auto &resource = implementation.resources;
    EXPECT_DOUBLE_EQ(implementation.totalCycles,
                     resource.setup +
                         (implementation.implementation.mode == StageMode::SIMD
                              ? std::max(resource.load, resource.store)
                              : resource.load + resource.store));
  }
}

TEST(StageAddressPatternTest, PartialContinuousRowsUseDirectMemoryPerRow) {
  for (int64_t columns : {32, 8, 1}) {
    IRTestContext context(true);
    auto module = addressModule(context, {8, columns},
                                {IndexKind::Loaded, IndexKind::Range});
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
    auto *payload = payloadStage(*partition, {8, columns});
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->costModelKind,
              StageCostModelKind::PartialContinuousTileMemory);
    const auto &work = payload->workload;
    ASSERT_EQ(work.addressPatterns.size(), 1u);
    const auto &address = work.addressPatterns.front();
    EXPECT_EQ(address.stageId, payload->id);
    EXPECT_TRUE(address.dependsOnLoadedValue);
    ASSERT_EQ(address.axes.size(), 2u);
    EXPECT_EQ(address.axes[0].extent, 8);
    EXPECT_EQ(address.axes[0].regularity, "opaque_loaded");
    if (columns > 1) {
      EXPECT_EQ(address.axes[1].regularity, "fixed_stride");
      EXPECT_EQ(address.axes[1].knownStride, 1);
    }
    EXPECT_TRUE(payload->features.hasPartialContinuousMemory);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadRows, 8);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadBytes, 8 * columns * 4);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadWarpInstructions,
                     std::ceil(8 * columns / 32.0));
    EXPECT_DOUBLE_EQ(work.indirectLoadTransactions, 0);
    auto table = evaluateOneStage(*payload);
    ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
    ASSERT_GE(table->stages.front().implementations.size(), 2u);
    EXPECT_DOUBLE_EQ(table->stages.front().implementations[0].resources.load,
                     8 * columns * 4 / 32.0);
  }
}

TEST(StageAddressPatternTest, PartialStructuredStoreAndNonunitStride) {
  for (bool store : {false, true}) {
    IRTestContext context(true);
    auto module = addressModule(
        context, {8, 8}, {IndexKind::Loaded, IndexKind::Range}, 2, store);
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
    auto *payload = payloadStage(*partition, {8, 8}, store);
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->costModelKind,
              StageCostModelKind::PartialContinuousTileMemory);
    EXPECT_DOUBLE_EQ(store ? payload->workload.partialContinuousStoreRows
                           : payload->workload.partialContinuousLoadRows,
                     8);
    if (store)
      EXPECT_DOUBLE_EQ(payload->workload.partialContinuousStoreWarpInstructions,
                       2);
    EXPECT_DOUBLE_EQ(store ? payload->workload.indirectStoreTransactions
                           : payload->workload.indirectLoadTransactions,
                     0);
  }
  IRTestContext context(true);
  auto module = addressModule(context, {8, 8},
                              {IndexKind::PointerArgument, IndexKind::Range});
  auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
  for (const auto &stage : partition->stages)
    EXPECT_NE(stage.costModelKind,
              StageCostModelKind::PartialContinuousTileMemory);
}

TEST(StageAddressPatternTest,
     PartialUnstructuredPrefixDoesNotRequireLoadedIndex) {
  for (bool store : {false, true}) {
    IRTestContext context(true);
    auto module = addressModule(
        context, {8, 8}, {IndexKind::Computed, IndexKind::Range}, 2, store);
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
    auto *payload = payloadStage(*partition, {8, 8}, store);
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->costModelKind,
              StageCostModelKind::PartialContinuousTileMemory);
    EXPECT_DOUBLE_EQ(store ? payload->workload.partialContinuousStoreRows
                           : payload->workload.partialContinuousLoadRows,
                     8);
  }
}

TEST(StageAddressPatternTest,
     PartialStructuredTailRequiresAllPrefixAxesUnstructured) {
  IRTestContext context(true);
  for (auto axes : {std::pair{true, false}, std::pair{false, true},
                    std::pair{true, true}}) {
    auto module = addressModule(
        context, {2, 4, 8},
        {axes.first ? IndexKind::Loaded : IndexKind::Range,
         axes.second ? IndexKind::Loaded : IndexKind::Range, IndexKind::Range});
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
    auto *payload = payloadStage(*partition, {2, 4, 8});
    ASSERT_NE(payload, nullptr);
    const bool partial = axes.first && axes.second;
    const auto &work = payload->workload;
    EXPECT_EQ(payload->costModelKind,
              StageCostModelKind::PartialContinuousTileMemory);
    EXPECT_TRUE(payload->features.hasPartialContinuousMemory);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadRows, partial ? 8 : 0);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadBytes, partial ? 256 : 0);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadWarpInstructions,
                     partial ? 2 : 0);
    EXPECT_DOUBLE_EQ(work.indirectLoadBytes, partial ? 0 : 256);
    if (partial)
      EXPECT_DOUBLE_EQ(work.indirectLoadTransactions, 0);
  }
}

TEST(StageAddressPatternTest, LoopPreservesIndirectAndPartialMemoryWork) {
  for (bool partial : {false, true}) {
    for (bool store : {false, true}) {
      IRTestContext context(true);
      const SmallVector<int64_t> shape =
          partial ? SmallVector<int64_t>{8, 8} : SmallVector<int64_t>{64};
      const SmallVector<IndexKind> kinds =
          partial ? SmallVector<IndexKind>{IndexKind::Loaded, IndexKind::Range}
                  : SmallVector<IndexKind>{IndexKind::Loaded};
      auto module = addressModule(context, shape, kinds, 1, store);
      auto function = *module->getOps<func::FuncOp>().begin();
      auto &body = function.getBody().front();
      SmallVector<Operation *> operations;
      for (auto &op : body.without_terminator())
        operations.push_back(&op);
      OpBuilder builder(&body, body.begin());
      auto loc = builder.getUnknownLoc();
      auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto three = builder.create<arith::ConstantIndexOp>(loc, 3);
      auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = builder.create<scf::ForOp>(loc, zero, three, one);
      for (auto *op : operations)
        op->moveBefore(loop.getBody()->getTerminator());
      auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                    StagePartitionerOptions{});
      ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
      auto total = [&](double StageWorkload::*field) {
        double value = 0;
        for (const auto &stage : partition->stages)
          value += stage.iterationCount * (stage.workload.*field);
        return value;
      };
      EXPECT_DOUBLE_EQ(total(store ? &StageWorkload::indirectStoreBytes
                                   : &StageWorkload::indirectLoadBytes),
                       partial ? 0 : 3 * 64 * 4);
      // Full-indirect fallback counts ceil(64 / 32) warp transactions per trip.
      EXPECT_DOUBLE_EQ(total(store ? &StageWorkload::indirectStoreTransactions
                                   : &StageWorkload::indirectLoadTransactions),
                       partial ? 0 : 3 * 2);
      EXPECT_DOUBLE_EQ(total(store ? &StageWorkload::partialContinuousStoreRows
                                   : &StageWorkload::partialContinuousLoadRows),
                       partial ? 3 * 8 : 0);
      EXPECT_DOUBLE_EQ(total(store
                                 ? &StageWorkload::partialContinuousStoreBytes
                                 : &StageWorkload::partialContinuousLoadBytes),
                       partial ? 3 * 64 * 4 : 0);
      EXPECT_DOUBLE_EQ(
          total(store ? &StageWorkload::partialContinuousStoreWarpInstructions
                      : &StageWorkload::partialContinuousLoadWarpInstructions),
          partial ? 3 * 2 : 0);
    }
  }
}

TEST(StageAddressPatternTest, LoadedScalarIndexRemainsUniform) {
  for (bool store : {false, true}) {
    IRTestContext context(true);
    auto module = addressModule(
        context, {8, 8}, {IndexKind::Scalar, IndexKind::Range}, 1, store);
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
    auto *payload = payloadStage(*partition, {8, 8}, store);
    ASSERT_NE(payload, nullptr);
    bool sawPayload = false;
    for (auto *op : payload->operations)
      if (op->getName().getStringRef() == (store ? "tt.store" : "tt.load")) {
        sawPayload = true;
        // #2528 treats a uniform scalar offset as base-address selection,
        // not a per-lane gather; the tensor's fixed-stride run is preserved.
        EXPECT_FALSE(isLoadedIndexDependentMemoryOp(op));
      }
    EXPECT_TRUE(sawPayload);
    EXPECT_NE(payload->costModelKind, StageCostModelKind::IndirectGatherMemory);
    EXPECT_FALSE(payload->features.hasIndirectMemory);
    EXPECT_FALSE(payload->features.hasPartialContinuousMemory);
    EXPECT_DOUBLE_EQ(store ? payload->workload.indirectStoreBytes
                           : payload->workload.indirectLoadBytes,
                     0);
    EXPECT_DOUBLE_EQ(store ? payload->workload.partialContinuousStoreRows
                           : payload->workload.partialContinuousLoadRows,
                     0);
  }
}

TEST(StageAddressPatternTest, CrossAxisProductStaysIndirect) {
  IRTestContext context(true);
  auto module = context.parse(R"mlir(
    module {
      tt.func @bilinear(%base: !tt.ptr<f32>) {
        %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
        %a = tt.expand_dims %r {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
        %b = tt.expand_dims %r {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
        %i = tt.broadcast %a : tensor<4x1xi32> -> tensor<4x4xi32>
        %j = tt.broadcast %b : tensor<1x4xi32> -> tensor<4x4xi32>
        %offset = arith.muli %i, %j : tensor<4x4xi32>
        %bp = tt.splat %base : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
        %ptr = tt.addptr %bp, %offset : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
        %value = tt.load %ptr : tensor<4x4x!tt.ptr<f32>>
        tt.return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
  auto *payload = payloadStage(*partition, {4, 4});
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->costModelKind, StageCostModelKind::IndirectGatherMemory);
  auto plan = buildMixedSimtAnchorPlan(*module, true);
  ASSERT_EQ(plan.anchors.size(), 1u);
  EXPECT_EQ(plan.anchors.front().kind, SimtAnchorKind::UnstructuredMemory);
}

TEST(StageAddressPatternTest, UnstructuredLoadStoreAnchorsMaterializeScopes) {
  for (bool store : {false, true})
    for (bool multidimensional : {false, true})
      for (IndexKind kind : {IndexKind::Range, IndexKind::Scalar,
                             IndexKind::Loaded, IndexKind::Computed}) {
        IRTestContext context(true, true);
        context.getOrLoadDialect<scope::ScopeDialect>();
        SmallVector<int64_t> shape = multidimensional
                                         ? SmallVector<int64_t>{4, 8}
                                         : SmallVector<int64_t>{32};
        SmallVector<IndexKind> kinds =
            multidimensional ? SmallVector<IndexKind>{kind, IndexKind::Range}
                             : SmallVector<IndexKind>{kind};
        auto module = addressModule(context, shape, kinds, 1, store);
        ASSERT_TRUE(module);
        // The shared builder uses OperationState; supply registered Triton's
        // optional-operand segments before verifying the materialized module.
        module->walk([&](Operation *op) {
          if (isa<triton::LoadOp>(op))
            op->setAttr("operandSegmentSizes",
                        DenseI32ArrayAttr::get(&context, {1, 0, 0}));
          if (isa<triton::StoreOp>(op))
            op->setAttr("operandSegmentSizes",
                        DenseI32ArrayAttr::get(&context, {1, 1, 0}));
        });
        ASSERT_TRUE(succeeded(verify(*module)));
        std::string original;
        llvm::raw_string_ostream stream(original);
        module->print(stream);
        auto plan = buildMixedSimtAnchorPlan(*module, true);
        std::string analyzed;
        llvm::raw_string_ostream analyzedStream(analyzed);
        module->print(analyzedStream);
        EXPECT_EQ(original, analyzed);
        bool indirect =
            kind == IndexKind::Loaded || kind == IndexKind::Computed;
        EXPECT_EQ(hasSimtAnchorInBody(*module), indirect);
        ASSERT_EQ(plan.anchors.size(), indirect ? 1u : 0u);
        if (!indirect)
          continue;
        auto &anchor = plan.anchors.front();
        EXPECT_EQ(anchor.kind, kind == IndexKind::Loaded
                                   ? SimtAnchorKind::LoadedIndexDependentMemory
                                   : SimtAnchorKind::UnstructuredMemory);
        EXPECT_EQ(anchor.operation->getName().getStringRef(),
                  store ? "tt.store" : "tt.load");
        EXPECT_TRUE(anchor.materializable);
        auto partition = StagePartitioner().partition(
            *module, plan, StagePartitionerOptions{});
        ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
        auto *payload = payloadStage(*partition, shape, store);
        ASSERT_NE(payload, nullptr);
        EXPECT_TRUE(payload->localSimtMaterializable);
        EXPECT_EQ(payload->simtAnchorIndices.size(), 1u);
        auto unsupported = buildMixedSimtAnchorPlan(*module, false);
        ASSERT_EQ(unsupported.anchors.size(), 1u);
        EXPECT_FALSE(unsupported.anchors.front().materializable);
        ASSERT_TRUE(succeeded(materializeSimtAnchorPlan(*module, plan)));
        EXPECT_EQ(anchor.operation->getParentOp()->getName().getStringRef(),
                  "scope.scope");
        EXPECT_TRUE(succeeded(verify(*module)));
      }
}

TEST(StageAddressPatternTest, AllDContractExcludesEveryMixedAxisPosition) {
  for (bool store : {false, true})
    for (auto left : {IndexKind::Range, IndexKind::Loaded, IndexKind::Computed})
      for (auto right :
           {IndexKind::Range, IndexKind::Loaded, IndexKind::Computed}) {
        IRTestContext context(true);
        auto module = addressModule(context, {4, 8}, {left, right}, 1, store);
        auto partition = StagePartitioner().partition(
            *module, SimtAnchorPlan{}, StagePartitionerOptions{});
        ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
        auto *payload = payloadStage(*partition, {4, 8}, store);
        ASSERT_NE(payload, nullptr);
        const bool allD = left != IndexKind::Range && right != IndexKind::Range;
        const bool mixed =
            (left == IndexKind::Range) != (right == IndexKind::Range);
        EXPECT_EQ(payload->costModelKind ==
                      StageCostModelKind::IndirectGatherMemory,
                  allD);
        EXPECT_EQ(payload->features.hasIndirectMemory, allD || mixed);
        EXPECT_EQ(payload->costModelKind ==
                      StageCostModelKind::PartialContinuousTileMemory,
                  mixed);
      }
}

TEST(StageAddressPatternTest, MixedNonRowFallbackKeepsIrregularPricing) {
  for (bool store : {false, true}) {
    IRTestContext context(true);
    auto module = addressModule(
        context, {4, 8}, {IndexKind::Range, IndexKind::Loaded}, 1, store);
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
    auto *payload = payloadStage(*partition, {4, 8}, store);
    ASSERT_NE(payload, nullptr);
    ASSERT_EQ(payload->costModelKind,
              StageCostModelKind::PartialContinuousTileMemory);
    EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadRows, 0);
    EXPECT_DOUBLE_EQ(payload->workload.partialContinuousStoreRows, 0);
    auto profile = hardwareProfile();
    profile.simd.loadOperationSetupCycles = 10.0;
    profile.simd.storeOperationSetupCycles = 10.0;
    auto costs = evaluateOneStage(*payload, profile);
    ASSERT_TRUE(bool(costs)) << llvm::toString(costs.takeError());
    const auto &resources =
        costs->stages.front().implementations.front().resources;
    // A byte-only fallback would cost 4 load / 8 store cycles here.
    EXPECT_TRUE(resources.indirectMemoryAnalyticalFallback);
    EXPECT_GE(store ? resources.store : resources.load, 14.0);
  }
}

TEST(StageAddressPatternTest, MixedAxesUseUpstreamCurvesInBothModes) {
  for (bool store : {false, true})
    for (bool tailStructured : {false, true}) {
      IRTestContext context(true);
      auto module = addressModule(
          context, {4, 8},
          tailStructured
              ? ArrayRef<IndexKind>({IndexKind::Loaded, IndexKind::Range})
              : ArrayRef<IndexKind>({IndexKind::Range, IndexKind::Loaded}),
          1, store);
      auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                    StagePartitionerOptions{});
      ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
      auto *payload = payloadStage(*partition, {4, 8}, store);
      ASSERT_NE(payload, nullptr);
      ASSERT_EQ(payload->costModelKind,
                StageCostModelKind::PartialContinuousTileMemory);
      auto profile = hardwareProfile();
      for (auto *mode : {&profile.simd, &profile.simt}) {
        mode->indirectLoadSystemCyclesByElements = {{32, 777.0}};
        mode->indirectLoadStartupSystemCyclesByWarpInstructions = {{1, 1.0}};
        mode->segmentedIndirectLoadFp32GroupCyclesByFactorSegmentsAndContiguousElements
            [1][4][8] = 777.0;
        mode->indirectStoreSystemCyclesBySegments = {{4, 777.0}, {32, 777.0}};
      }
      auto costs = evaluateOneStage(*payload, profile);
      ASSERT_TRUE(bool(costs)) << llvm::toString(costs.takeError());
      for (const auto &cost : costs->stages.front().implementations) {
        EXPECT_DOUBLE_EQ(store ? cost.resources.store
                               : cost.resources.indirectLoadGroup,
                         777.0);
        EXPECT_FALSE(cost.resources.indirectMemoryAnalyticalFallback);
      }
    }
}

TEST(StageAddressPatternTest, ScheduledUniformBranchRemainsStructured) {
  IRTestContext context(true);
  auto module = context.parse(R"mlir(
    module {
      tt.func @scheduled(%base: !tt.ptr<f32>) {
        %pid = tt.get_program_id x : i32
        %n = tt.get_num_programs x : i32
        %zero = arith.constant 0 : i32
        %one = arith.constant 1 : i32
        %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
        %two = arith.constant dense<2> : tensor<32xi32>
        %bp = tt.splat %base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
        scf.for %iv = %pid to %n step %one : i32 {
          %logical = arith.remui %iv, %n : i32
          %condition = arith.cmpi eq, %logical, %zero : i32
          %selected = scf.if %condition -> tensor<32xi32> {
            scf.yield %range : tensor<32xi32>
          } else {
            %scaled = arith.muli %range, %two : tensor<32xi32>
            scf.yield %scaled : tensor<32xi32>
          }
          %ptr = tt.addptr %bp, %selected : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
          %value = tt.load %ptr : tensor<32x!tt.ptr<f32>>
        } {ta.auto_blockify_v1.loop}
        tt.return
      }
    })mlir");
  ASSERT_TRUE(module);
  EXPECT_TRUE(collectUnstructuredMemoryOps(*module).empty());
  auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
  auto *payload = payloadStage(*partition, {32});
  ASSERT_NE(payload, nullptr);
  EXPECT_FALSE(payload->features.hasIndirectMemory);
  EXPECT_EQ(payload->costModelKind, StageCostModelKind::ContinuousTileMemory);
}

TEST(StageAddressPatternTest,
     ClampedStrideProofRejectsPartialClampingAndMisalignment) {
  for (int high : {62, 63, 65})
    for (int tile : {16, 32})
      for (int outer : {32, 64}) {
        IRTestContext context(true);
        context.getOrLoadDialect<math::MathDialect>();
        auto module = context.parse(llvm::formatv(R"mlir(
          module {{
            tt.func @clamped(%base: !tt.ptr<f32>, %block: i32, %row: i32) {{
              %tile = arith.constant {0} : i32
              %align = arith.constant {1} : i32
              %start = arith.muli %block, %tile : i32
              %rowbase = arith.muli %row, %align : i32
              %s = tt.splat %start : i32 -> tensor<32xi32>
              %r = tt.make_range {{start = 0 : i32, end = 32 : i32} : tensor<32xi32>
              %x = arith.addi %s, %r : tensor<32xi32>
              %f = arith.sitofp %x : tensor<32xi32> to tensor<32xf32>
              %two = arith.constant dense<2.0> : tensor<32xf32>
              %scaled = arith.mulf %f, %two : tensor<32xf32>
              %floor = math.floor %scaled : tensor<32xf32>
              %low = arith.constant dense<0.0> : tensor<32xf32>
              %high = arith.constant dense<{2}.0> : tensor<32xf32>
              %clamped = tt.clampf %floor, %low, %high, propagateNan = none : tensor<32xf32>
              %i = arith.fptosi %clamped : tensor<32xf32> to tensor<32xi32>
              %o = tt.splat %rowbase : i32 -> tensor<32xi32>
              %offset = arith.addi %o, %i : tensor<32xi32>
              %bp = tt.splat %base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
              %p = tt.addptr %bp, %offset : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
              %v = tt.load %p : tensor<32x!tt.ptr<f32>>
              tt.return
            }
          })mlir",
                                                  tile, outer, high)
                                        .str());
        ASSERT_TRUE(module);
        auto partition = StagePartitioner().partition(
            *module, SimtAnchorPlan{}, StagePartitionerOptions{});
        ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
        auto *payload = payloadStage(*partition, {32});
        ASSERT_NE(payload, nullptr);
        const bool proven = high < 64 && tile == 32 && outer == 64;
        EXPECT_EQ(payload->features.hasIndirectMemory, !proven);
        EXPECT_EQ(collectUnstructuredMemoryOps(*module).empty(), proven);
      }
}

TEST(StageAddressPatternTest, MemoryFactsDoNotMutateInputIR) {
  IRTestContext context(true);
  auto module =
      addressModule(context, {4, 8}, {IndexKind::Loaded, IndexKind::Range});
  auto print = [&] {
    std::string text;
    llvm::raw_string_ostream stream(text);
    module->print(stream);
    return text;
  };
  const auto before = print();
  StageMemoryPatternAnalysis facts(*module);
  EXPECT_EQ(print(), before);
  module->walk([&](triton::LoadOp load) {
    auto type = cast<RankedTensorType>(load.getPtr().getType());
    auto irregular = facts.hasUnstructuredAxes(load);
    ASSERT_TRUE(irregular.has_value());
    EXPECT_EQ(*irregular, type.getRank() == 2);
    const auto *summary = facts.lookupSummary(load);
    ASSERT_NE(summary, nullptr);
    EXPECT_EQ(summary->axes.size(), type.getRank());
    auto tile = facts.lookup(load);
    EXPECT_EQ(tile.has_value(), type.getRank() == 2);
    if (tile) {
      EXPECT_DOUBLE_EQ(tile->rows, 4);
      EXPECT_DOUBLE_EQ(tile->elementsPerRow, 8);
    }
  });
  EXPECT_EQ(print(), before);
}

TEST(StageAddressPatternTest, MissingMemoryFactsRemainUnknown) {
  IRTestContext context(true);
  auto module = addressModule(context, {8}, {IndexKind::PointerArgument});
  StageMemoryPatternAnalysis facts(*module);
  module->walk([&](triton::LoadOp load) {
    EXPECT_FALSE(facts.hasUnstructuredAxes(load).has_value());
    EXPECT_FALSE(facts.lookup(load).has_value());
    const auto *summary = facts.lookupSummary(load);
    ASSERT_NE(summary, nullptr);
    ASSERT_EQ(summary->axes.size(), 1u);
    EXPECT_EQ(summary->axes[0].offsetAxisInfo, "unknown");
  });
  EXPECT_EQ(facts.lookupSummary(*module), nullptr);
  EXPECT_FALSE(facts.hasUnstructuredAxes(*module).has_value());
}

TEST(StageAddressPatternTest, UniformBranchStridesDoNotBecomeIndirect) {
  // A scalar branch selects a whole affine tensor. A non-affine/loaded branch
  // or per-lane select must not receive that proof, for either load or store.
  for (bool store : {false, true})
    for (unsigned variant : {0u, 1u, 2u})
      for (StringRef expression :
           {"arith.muli %range, %two", "arith.addi %range, %two",
            "arith.subi %two, %range", "arith.muli %range, %large",
            "arith.muli %range, %range", "arith.remsi %range, %two",
            "tt.load %ip", "arith.select %mask, %range, %two"}) {
        SCOPED_TRACE(expression.str() + (store ? " store " : " load ") +
                     std::to_string(variant));
        IRTestContext context(true);
        auto module = context.parse(
            llvm::formatv(
                R"mlir(
        module {{
          tt.func @conditional(%base: !tt.ptr<f32>, %indices: !tt.ptr<i32>,
                               %condition: i1) {{
            %range = tt.make_range {{start = 0 : i32, end = 32 : i32} : tensor<32xi32>
            %two = arith.constant dense<2> : tensor<32xi32>
            %large = arith.constant dense<1073741824> : tensor<32xi32>
            %mask = arith.cmpi slt, %range, %two : tensor<32xi32>
            %ib = tt.splat %indices : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
            %ip = tt.addptr %ib, %range : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
            %loaded_condition = tt.load %indices : !tt.ptr<i32>
            %zero = arith.constant 0 : i32
            %unknown_condition = arith.cmpi eq, %loaded_condition, %zero : i32
            %selected = scf.if {4} -> tensor<32xi32> {{
              scf.yield %range : tensor<32xi32>
            } else {{
              %changed = {0} : {2}
              scf.yield %changed : tensor<32xi32>
            }
            {3}
            %bp = tt.splat %base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
            %ptr = tt.addptr %bp, {5} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
            %value = arith.constant dense<0.0> : tensor<32xf32>
            {1}
            tt.return
          }
        }
      )mlir",
                expression,
                store ? "tt.store %ptr, %value : tensor<32x!tt.ptr<f32>>"
                      : "%loaded = tt.load %ptr, %mask, %value : "
                        "tensor<32x!tt.ptr<f32>>",
                expression.starts_with("tt.load") ? "tensor<32x!tt.ptr<i32>>"
                : expression.starts_with("arith.select")
                    ? "tensor<32xi1>, tensor<32xi32>"
                    : "tensor<32xi32>",
                variant == 1
                    ? "%post = arith.muli %selected, %range : tensor<32xi32>"
                    : "",
                variant == 2 ? "%unknown_condition" : "%condition",
                variant == 1 ? "%post" : "%selected")
                .str());
        ASSERT_TRUE(module);
        auto anchors = collectUnstructuredMemoryOps(*module);
        const bool proven =
            variant == 0 && (expression == "arith.muli %range, %two" ||
                             expression == "arith.addi %range, %two" ||
                             expression == "arith.subi %two, %range");
        EXPECT_EQ(anchors.size(), proven ? 0u : 1u);
        auto partition = StagePartitioner().partition(
            *module, SimtAnchorPlan{}, StagePartitionerOptions{});
        ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
        auto *payload = payloadStage(*partition, {32}, store);
        ASSERT_NE(payload, nullptr);
        const bool affine =
            variant == 0 && (expression == "arith.muli %range, %two" ||
                             expression == "arith.addi %range, %two" ||
                             expression == "arith.subi %two, %range");
        EXPECT_EQ(payload->costModelKind,
                  affine ? (store ? StageCostModelKind::ContinuousTileStore
                                  : StageCostModelKind::ContinuousTileMemory)
                         : StageCostModelKind::IndirectGatherMemory);
        EXPECT_DOUBLE_EQ(store ? payload->workload.indirectStoreBytes
                               : payload->workload.indirectLoadBytes,
                         affine ? 0 : 128);
        if (affine && !store) {
          auto profile = hardwareProfile();
          profile.simd
              .controlFlowAddressScalarizedLoadResidualCyclesByElements = {
              {32, 100.0}};
          profile.simd.controlFlowAddressScalarizedLoadEvidence =
              "unit-test rate";
          auto costs = evaluateOneStage(*payload, profile);
          ASSERT_TRUE(bool(costs)) << llvm::toString(costs.takeError());
          EXPECT_DOUBLE_EQ(costs->stages.front()
                               .implementations.front()
                               .resources.controlFlowAddressScalarizedLoad,
                           100.0);
        }
      }
}
