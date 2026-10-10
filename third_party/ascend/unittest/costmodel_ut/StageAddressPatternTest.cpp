// Pointer-analysis coverage for the indirect and partially structured stages.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "CostModelTestUtils.h"
#include "StageIRTestUtils.h"
#include "mlir/IR/Builders.h"

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
      if (type && type.getShape() == shape)
        return &stage;
    }
  return nullptr;
}
} // namespace

TEST(StageAddressPatternTest, PartialContinuousStageAlwaysUsesSerialCost) {
  LogicalStage stage =
      logicalStage("partial", StageCostModelKind::PartialContinuousTileMemory,
                   StageScheduleKind::IndependentPipelined);
  stage.features.hasPartialContinuousMemory = true;
  stage.features.hasIndirectMemory = true;
  stage.workload.operationElements.clear();
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
              partial ? StageCostModelKind::PartialContinuousTileMemory
                      : StageCostModelKind::IndirectGatherMemory);
    EXPECT_EQ(payload->features.hasPartialContinuousMemory, partial);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadRows, partial ? 8 : 0);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadBytes, partial ? 256 : 0);
    EXPECT_DOUBLE_EQ(work.partialContinuousLoadWarpInstructions,
                     partial ? 8 : 0);
    EXPECT_DOUBLE_EQ(work.indirectLoadBytes, partial ? 0 : 256);
    if (partial)
      EXPECT_DOUBLE_EQ(work.indirectLoadTransactions, 0);
  }
}

TEST(StageAddressPatternTest, LoadedScalarIndexRemainsIndirect) {
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
        EXPECT_TRUE(isLoadedIndexDependentMemoryOp(op));
      }
    EXPECT_TRUE(sawPayload);
    EXPECT_EQ(payload->costModelKind, StageCostModelKind::IndirectGatherMemory);
    EXPECT_TRUE(payload->features.hasIndirectMemory);
    EXPECT_FALSE(payload->features.hasPartialContinuousMemory);
    EXPECT_DOUBLE_EQ(store ? payload->workload.indirectStoreBytes
                           : payload->workload.indirectLoadBytes,
                     256);
    EXPECT_DOUBLE_EQ(store ? payload->workload.partialContinuousStoreRows
                           : payload->workload.partialContinuousLoadRows,
                     0);
  }
}
