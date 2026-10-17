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

inline SmallVector<Type> payloadTypes(MLIRContext &c, bool simt = false) {
  SmallVector<Type> types = {
      IntegerType::get(&c, 8),  IntegerType::get(&c, 16),
      IntegerType::get(&c, 32), IntegerType::get(&c, 64),
      Float16Type::get(&c),     BFloat16Type::get(&c),
      Float32Type::get(&c),     Float8E4M3FNType::get(&c),
      Float8E5M2Type::get(&c)};
  if (simt)
    types.append({IntegerType::get(&c, 32, IntegerType::Unsigned),
                  Float64Type::get(&c)});
  return types;
}

// Shared real IR fixture: scalar bases, contiguous index/value loads, one
// indirect target; optional flat prefix or loaded-byte mask preserves rank.
inline OwningOpRef<ModuleOp>
memoryModule(IRTestContext &c, ArrayRef<int64_t> shape, StringRef type,
             bool store, StringRef mask = "none", int64_t k = 0) {
  int64_t e = 1;
  std::string dims;
  for (int64_t d : shape) {
    e *= d;
    dims += std::to_string(d) + "x";
  }
  auto fmt = [&](StringRef text) {
    return llvm::formatv(false, text.data(), e, dims, type, k).str();
  };
  std::string ir = fmt(R"(
    module {{ tt.func @probe(%x: !tt.ptr<{2}>, %idx: !tt.ptr<i32>,
                             %values: !tt.ptr<{2}>, %masks: !tt.ptr<i8>) {{
      %r = tt.make_range {{start = 0 : i32, end = {0} : i32} : tensor<{0}xi32>
      %ip = tt.splat %idx : !tt.ptr<i32> -> tensor<{0}x!tt.ptr<i32>>
      %ip2 = tt.addptr %ip, %r : tensor<{0}x!tt.ptr<i32>>, tensor<{0}xi32>
      %i = tt.load %ip2 : tensor<{0}x!tt.ptr<i32>>
  )");
  if (shape.size() > 1)
    ir += fmt("%index = tt.reshape %i : tensor<{0}xi32> -> tensor<{1}i32>\n");
  ir += fmt("%xp = tt.splat %x : !tt.ptr<{2}> -> tensor<{1}!tt.ptr<{2}>> "
            "loc(\"probe.py\":1:1)\n%ptr = tt.addptr %xp, ");
  ir += shape.size() > 1 ? "%index" : "%i";
  ir +=
      fmt(" : tensor<{1}!tt.ptr<{2}>>, tensor<{1}i32> loc(\"probe.py\":1:1)\n");
  if (mask == "prefix")
    ir += fmt("%k = arith.constant dense<{3}> : tensor<{0}xi32>\n%flat = "
              "arith.cmpi slt, %r, %k : tensor<{0}xi32>\n");
  if (mask == "loaded")
    ir += fmt(R"(
      %mp = tt.splat %masks : !tt.ptr<i8> -> tensor<{0}x!tt.ptr<i8>>
      %mp2 = tt.addptr %mp, %r : tensor<{0}x!tt.ptr<i8>>, tensor<{0}xi32>
      %m = tt.load %mp2 : tensor<{0}x!tt.ptr<i8>>
      %mz = arith.constant dense<0> : tensor<{0}xi8>
      %flat = arith.cmpi ne, %m, %mz : tensor<{0}xi8>
    )");
  if (mask != "none")
    ir += fmt("%mask = tt.reshape %flat : tensor<{0}xi1> -> tensor<{1}i1>\n");
  if (store) {
    ir += fmt(R"(
      %vp = tt.splat %values : !tt.ptr<{2}> -> tensor<{0}x!tt.ptr<{2}>>
      %vp2 = tt.addptr %vp, %r : tensor<{0}x!tt.ptr<{2}>>, tensor<{0}xi32>
      %v = tt.load %vp2 : tensor<{0}x!tt.ptr<{2}>>
    )");
    if (shape.size() > 1)
      ir += fmt("%value = tt.reshape %v : tensor<{0}x{2}> -> tensor<{1}{2}>\n");
    ir += shape.size() > 1 ? "tt.store %ptr, %value" : "tt.store %ptr, %v";
  } else
    ir += fmt(type.starts_with("f") ? "%zero = arith.constant dense<0.0> : "
                                      "tensor<{1}{2}>\n%value = tt.load %ptr"
                                    : "%zero = arith.constant dense<0> : "
                                      "tensor<{1}{2}>\n%value = tt.load %ptr");
  if (mask != "none")
    ir += store ? ", %mask" : ", %mask, %zero";
  ir += fmt(" : tensor<{1}!tt.ptr<{2}>> loc(\"probe.py\":1:2)\ntt.return } }");
  return c.parse(ir);
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
    axis.offsetAxisInfo = "unstructured";
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
  void addIndex() {
    OperationState state(op->getLoc(), "tt.load");
    state.addOperands(op->getOperand(0));
    state.addTypes(RankedTensorType::get(
        {stage.workload.addressPatterns[0].axes[0].extent},
        IntegerType::get(op->getContext(), 32)));
    auto *index = Operation::create(state);
    block.push_front(index);
    op->setOperand(0, index->getResult(0));
  }
};

inline void expectOnlyMemoryChanged(const StageImplementationCost &before,
                                    const StageImplementationCost &after,
                                    bool store = false,
                                    int64_t iterations = 3) {
  auto adjusted = before;
  auto &oldMemory = store ? adjusted.resources.store : adjusted.resources.load;
  const double newMemory = store ? after.resources.store : after.resources.load;
  oldMemory = newMemory;
  (store ? adjusted.indirectStorePricing : adjusted.indirectLoadPricing) =
      store ? after.indirectStorePricing : after.indirectLoadPricing;
  adjusted.totalCycles = after.totalCycles;
  adjusted.formulaEvidence = after.formulaEvidence;
  adjusted.resources.indirectMemoryAnalyticalFallback =
      after.resources.indirectMemoryAnalyticalFallback;
  EXPECT_EQ(llvm::formatv("{0}", llvm::json::Value(adjusted.toJSON())).str(),
            llvm::formatv("{0}", llvm::json::Value(after.toJSON())).str());
}

} // namespace mlir::ascend::test

#endif // ASCEND_COSTMODEL_UT_INDIRECTMEMORYTESTUTILS_H
