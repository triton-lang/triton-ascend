// Pointer-analysis coverage for the indirect and partially structured stages.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/Models/PartialStructuredMemoryModel.h"
#include "CostModelTestUtils.h"
#include "StageIRTestUtils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
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

TEST(StageAddressPatternTest,
     PartialFitOwnsOneMemoryPriceAndNativeAddressWork) {
  for (bool store : {false, true}) {
    IRTestContext context(true);
    auto module = addressModule(
        context, {4, 8}, {IndexKind::Loaded, IndexKind::Range}, 1, store);
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());
    auto *payload = payloadStage(*partition, {4, 8}, store);
    ASSERT_NE(payload, nullptr);
    ASSERT_EQ(payload->workload.addressPatterns.size(), 1u);
    const auto &p = payload->workload.addressPatterns.front();
    ASSERT_TRUE(p.axes[0].opaqueMultiplier);
    EXPECT_EQ(*p.axes[0].opaqueMultiplier, 8);
    EXPECT_EQ(p.elementBitWidth, 32);
    auto profile = hardwareProfile();
    profile.target = "Ascend950PR/dav-c310";
    profile.partialStructuredMemoryModel =
        "ttir_q1_short_formula_20261014";
    auto table = evaluateOneStage(*payload, profile);
    ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
    partial_memory::Geometry g{{{4, 8, true}, {8, 1, false}}, 32, 1, true};
    for (const auto &cost : table->stages.front().implementations) {
      bool simd = cost.implementation.mode == StageMode::SIMD;
      auto m = simd ? (store ? partial_memory::Model::SimdStore
                             : partial_memory::Model::SimdLoad)
                    : (store ? partial_memory::Model::SimtStore
                             : partial_memory::Model::SimtLoad);
      auto expected = partial_memory::predict(g, m);
      ASSERT_TRUE(expected);
      EXPECT_NEAR(store ? cost.resources.store : cost.resources.load, *expected,
                  1e-8);
      EXPECT_TRUE((store ? cost.indirectStorePricing : cost.indirectLoadPricing)
                      .find("partial_") == 0);
      if (simd) {
        EXPECT_DOUBLE_EQ(cost.resources.scalar, 0);
        EXPECT_DOUBLE_EQ(cost.resources.compute, 0);
        EXPECT_DOUBLE_EQ(cost.resources.issue, 0);
      }
    }
    profile.partialStructuredMemoryModel.clear();
    auto legacy = evaluateOneStage(*payload, profile);
    ASSERT_TRUE(bool(legacy));
    EXPECT_EQ(
        legacy->stages.front().implementations.front().indirectLoadPricing,
        "generic_resource_model");
  }
}

TEST(StageAddressPatternTest, PartialFitUsesBaseWarpsAcrossCalibratedRange) {
  for (int64_t warps : {1, 2, 4, 8, 16, 32, 64}) {
    IRTestContext context(true);
    auto module = addressModule(
        context, {4, 8 * warps}, {IndexKind::Loaded, IndexKind::Range});
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                 StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition));
    auto *payload = payloadStage(*partition, {4, 8 * warps});
    ASSERT_NE(payload, nullptr);
    auto profile = hardwareProfile();
    profile.target = "Ascend950PR/dav-c310";
    profile.baseSimtWarpCount = warps;
    profile.logicalWarpGroupCount = 1;
    profile.partialStructuredMemoryModel = "ttir_q1_short_formula_20261014";
    auto table = evaluateOneStage(*payload, profile);
    ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
    partial_memory::Geometry g{
        {{4, 8 * warps, true}, {8 * warps, 1, false}}, 32, warps, true};
    for (const auto &cost : table->stages.front().implementations) {
      auto kind = cost.implementation.mode == StageMode::SIMD
                      ? partial_memory::Model::SimdLoad
                      : partial_memory::Model::SimtLoad;
      auto expected = partial_memory::predict(g, kind);
      ASSERT_TRUE(expected);
      EXPECT_NEAR(cost.resources.load, *expected, 1e-8);
    }
  }
}

TEST(StageAddressPatternTest,
     NativeFitRetainsUnrelatedPreparationAndCacheFallback) {
  IRTestContext context(true);
  auto module =
      addressModule(context, {4, 8}, {IndexKind::Loaded, IndexKind::Range});
  auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition));
  auto *payload = payloadStage(*partition, {4, 8});
  ASSERT_NE(payload, nullptr);
  LogicalStage stage = *payload;
  auto summary = stage.workload.addressPatterns;
  Operation *memory = nullptr;
  for (auto *op : stage.operations)
    if (op->getName().getStringRef() == "tt.load")
      memory = op;
  ASSERT_NE(memory, nullptr);
  OpBuilder builder(memory);
  OperationState state(builder.getUnknownLoc(), "tt.make_range");
  state.addTypes(RankedTensorType::get({8}, builder.getI32Type()));
  state.addAttribute("start", builder.getI32IntegerAttr(0));
  state.addAttribute("end", builder.getI32IntegerAttr(8));
  auto *preparation = builder.create(state);
  stage.operations.push_back(preparation);
  StagePartition single;
  single.operationOwnershipComplete = true;
  single.stages.push_back(stage);
  ASSERT_FALSE(bool(StageWorkloadAnalysis().analyze(single)));
  stage = single.stages.front();
  stage.workload.addressPatterns = summary;
  auto profile = hardwareProfile();
  profile.target = "Ascend950PR/dav-c310";
  profile.partialStructuredMemoryModel = "ttir_q1_short_formula_20261014";
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(table));
  auto &native = table->stages.front().implementations.front();
  EXPECT_TRUE(native.indirectLoadPricing.find("partial_simd_") == 0);
  EXPECT_GT(native.resources.issue, 0);
  EXPECT_EQ(native.memoryPricingBoundary,
            "simd_native_fragment_with_completion_barrier");
  memory->setAttr(
      "cache", builder.getI32IntegerAttr(int64_t(triton::CacheModifier::CA)));
  auto fallback = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(fallback));
  EXPECT_EQ(
      fallback->stages.front().implementations.front().indirectLoadPricing,
      "generic_resource_model");
}

TEST(StageAddressPatternTest, PartialContinuousStageAlwaysUsesSerialCost) {
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
                     resource.setup + resource.load + resource.store);
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
                     8 * std::ceil(columns / 32.0));
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
                       8);
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

TEST(StageAddressPatternTest, PartialStructuredStageCoversMixedPrefixAxes) {
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
                     partial ? 8 : 0);
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
          partial ? 3 * 8 : 0);
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
