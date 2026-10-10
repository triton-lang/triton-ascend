#ifndef ASCEND_COSTMODEL_UT_INDIRECTMEMORYTESTUTILS_H
#define ASCEND_COSTMODEL_UT_INDIRECTMEMORYTESTUTILS_H

#include "AscendModel/RouteModel/Models/IndirectGatherMemoryCostModel.h"
#include "CostModelTestUtils.h"
#include "StageIRTestUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::ascend::test {

inline HardwareProfile calibrationProfile() {
  auto profile = hardwareProfile();
  profile.target = "Ascend950PR/dav-c310";
  return profile;
}

inline std::vector<StageImplementationCost>
indirectCosts(const LogicalStage &stage, const HardwareProfile &profile) {
  auto result = evaluateOneStage(stage, profile);
  if (!result)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(result.takeError())));
  return result->stages.front().implementations;
}

/// Construct the same per-axis evidence for load/store domain checks.
/// Keep special axes, masks, reuse proofs and expected prices in each test.
inline AddressPatternSummary
loadedAddressPattern(const LogicalStage &stage, llvm::StringRef memoryOp,
                     llvm::ArrayRef<int64_t> shape,
                     llvm::StringRef regularity = "opaque_loaded",
                     bool dependsOnLoadedValue = true) {
  AddressPatternSummary pattern;
  pattern.stageId = stage.id;
  pattern.memoryOp = memoryOp.str();
  pattern.dependsOnLoadedValue = dependsOnLoadedValue;
  for (int64_t extent : shape) {
    AddressAxisSummary axis;
    axis.extent = extent;
    axis.regularity = regularity.str();
    pattern.axes.push_back(axis);
  }
  return pattern;
}

// A Block owns the synthetic op and any optional address/scope helpers.
// Synthetic operands isolate formula guards; partition tests use real TTIR.
struct MemoryCase {
  Block block;
  Operation *op;
  LogicalStage stage =
      logicalStage("memory", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::StraightLine, 3);
  MemoryCase(MLIRContext &context, Type element, ArrayRef<int64_t> shape,
             bool store = false) {
    auto loc = UnknownLoc::get(&context);
    auto type = RankedTensorType::get(shape, element);
    auto ptr = block.addArgument(IndexType::get(&context), loc);
    OperationState state(loc, store ? "tt.store" : "tt.load");
    state.addOperands(ptr);
    if (store)
      state.addOperands(block.addArgument(type, loc));
    else
      state.addTypes(type);
    op = Operation::create(state);
    block.push_back(op);
    stage.operations = {op};
    auto &work = stage.workload;
    const int64_t elements = type.getNumElements();
    const double bytes =
        store ? std::max(1u, element.getIntOrFloatBitWidth() / 8) * elements
              : (element.getIntOrFloatBitWidth() * elements + 7) / 8;
    (store ? work.storeBytes : work.loadBytes) = bytes;
    (store ? work.indirectStoreBytes : work.indirectLoadBytes) = bytes;
    (store ? work.storeWarpInstructions : work.loadWarpInstructions) =
        (store ? work.indirectStoreTransactions
               : work.indirectLoadTransactions) = store ? 1 : elements;
    work.addressPatterns = {
        loadedAddressPattern(stage, store ? "tt.store" : "tt.load", shape)};
  }
  void addHelper(bool nested = false) {
    OperationState state(op->getLoc(), "test.address");
    auto *helper = Operation::create(state);
    block.push_back(helper);
    stage.operations = {helper, op};
    if (nested) {
      OperationState ownerState(op->getLoc(), "test.region");
      ownerState.addRegion();
      auto *owner = Operation::create(ownerState);
      block.push_back(owner);
      auto &body = owner->getRegion(0).emplaceBlock();
      helper->remove();
      op->remove();
      body.push_back(helper);
      body.push_back(op);
      stage.operations = {owner};
    }
  }
};

inline void expectOnlyMemoryChanged(const StageImplementationCost &before,
                                    const StageImplementationCost &after,
                                    bool store = false,
                                    int64_t iterations = 3) {
  auto adjusted = before;
  auto &oldMemory = store ? adjusted.resources.store : adjusted.resources.load;
  const double newMemory = store ? after.resources.store : after.resources.load;
  EXPECT_NEAR(after.totalCycles - before.totalCycles,
              iterations * (newMemory - oldMemory), 1e-7);
  oldMemory = newMemory;
  (store ? adjusted.indirectStorePricing : adjusted.indirectLoadPricing) =
      store ? after.indirectStorePricing : after.indirectLoadPricing;
  adjusted.totalCycles = after.totalCycles;
  EXPECT_EQ(llvm::formatv("{0}", llvm::json::Value(adjusted.toJSON())).str(),
            llvm::formatv("{0}", llvm::json::Value(after.toJSON())).str());
}

} // namespace mlir::ascend::test

#endif // ASCEND_COSTMODEL_UT_INDIRECTMEMORYTESTUTILS_H
