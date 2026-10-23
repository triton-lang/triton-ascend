#ifndef INDIRECT_MEMORY_TEST_UTILS_H
#define INDIRECT_MEMORY_TEST_UTILS_H
#include "AscendModel/RouteModel/Models/IndirectGatherMemoryCostModel.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
using namespace mlir;
using namespace mlir::ascend;
namespace {
// Shared real TTIR, with expected classifications and costs kept in each test.
struct Memory {
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  LogicalStage stage;
  HardwareProfile profile;
  Operation *target = nullptr;
  bool isStore;
  Memory(bool store, int rank = 1, StringRef type = "f32", int mask = 0,
         int e = 128, int w = 1, StringRef offset = "%i", int indexBits = 32)
      : isStore(store) {
    static auto parameters = [] {
      llvm::SmallString<256> path(getDefaultSimdSimtProfilePath());
      llvm::sys::path::remove_filename(path);
      llvm::sys::path::append(path, "../indirect_memory/ascend_950pr_v1.json");
      return llvm::cantFail(IndirectMemoryParameters::load(path));
    }();
    profile.indirectMemoryParameters = parameters;
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<triton::TritonDialect>();
    SmallVector<int64_t> shape(rank, 2);
    shape[w == 1 ? rank - 1 : 0] = e / (1LL << (rank - 1));
    std::string dims;
    for (int64_t d : shape)
      dims += std::to_string(d) + "x";
    auto fmt = [&](StringRef text) {
      return llvm::formatv(false, text.data(), e, dims, type,
                           w == 1 ? e / 2 : 3 * e / 4,
                           type.starts_with("i") ? "0" : "0.0", offset,
                           rank == 1 ? offset : "%index", indexBits,
                           offset == "%i" ? indexBits : 32)
          .str();
    };
    std::string ir = fmt(R"(
      module {{ tt.func @probe(%x: !tt.ptr<{2}>, %idx: !tt.ptr<i{7}>, %m: !tt.ptr<i8>) {{
        %r = tt.make_range {{start = 0 : i32, end = {0} : i32} : tensor<{0}xi32>
        %ip = tt.splat %idx : !tt.ptr<i{7}> -> tensor<{0}x!tt.ptr<i{7}>>
        %ips = tt.addptr %ip, %r : tensor<{0}x!tt.ptr<i{7}>>, tensor<{0}xi32>
        %i = tt.load %ips : tensor<{0}x!tt.ptr<i{7}>>
        %square = arith.muli %r, %r : tensor<{0}xi32>
    )");
    if (rank > 1)
      ir += fmt(
          "%index = tt.reshape {5} : tensor<{0}xi{8}> -> tensor<{1}i{8}>\n");
    ir += fmt(R"(
        %xp = tt.splat %x : !tt.ptr<{2}> -> tensor<{1}!tt.ptr<{2}>>
        %ptr = tt.addptr %xp, {6} : tensor<{1}!tt.ptr<{2}>>, tensor<{1}i{8}>
        %zero = arith.constant dense<{4}> : tensor<{1}{2}>
    )");
    if (mask == 2)
      ir += fmt(R"(
        %mp = tt.splat %m : !tt.ptr<i8> -> tensor<{0}x!tt.ptr<i8>>
        %mps = tt.addptr %mp, %r : tensor<{0}x!tt.ptr<i8>>, tensor<{0}xi32>
        %bytes = tt.load %mps : tensor<{0}x!tt.ptr<i8>>
        %z = arith.constant dense<0> : tensor<{0}xi8>
        %flat = arith.cmpi ne, %bytes, %z : tensor<{0}xi8>
      )");
    else
      ir += fmt("%k = arith.constant dense<{3}> : tensor<{0}xi32>\n"
                "%flat = arith.cmpi slt, %r, %k : tensor<{0}xi32>\n");
    ir += fmt("%mask = tt.reshape %flat : tensor<{0}xi1> -> tensor<{1}i1>\n");
    ir += store ? "tt.store %ptr, %zero" : "%value = tt.load %ptr";
    if (mask)
      ir += store ? ", %mask" : ", %mask, %zero";
    ir += fmt(" : tensor<{1}!tt.ptr<{2}>>\ntt.return } }");
    module = parseSourceString<ModuleOp>(ir, &context);
    if (!module)
      return;
    Operation *compare = nullptr;
    module->walk([&](arith::CmpIOp op) { compare = op; });
    module->walk([&](Operation *op) {
      if (isa<triton::LoadOp, triton::StoreOp>(op))
        target = op;
    });
    stage.id = "memory";
    stage.costModelKind = StageCostModelKind::IndirectGatherMemory;
    stage.operations = {compare, target};
    auto tensor =
        cast<RankedTensorType>(store ? target->getOperand(1).getType()
                                     : target->getResult(0).getType());
    auto &work = stage.workload;
    double bytes = e * tensor.getElementTypeBitWidth() / 8;
    (store ? work.storeBytes : work.loadBytes) = bytes;
    (store ? work.indirectStoreBytes : work.indirectLoadBytes) = bytes;
    AddressPatternSummary pattern;
    pattern.stageId = stage.id;
    pattern.memoryOp = store ? "tt.store" : "tt.load";
    pattern.dependsOnLoadedValue = true;
    for (int64_t d : shape) {
      AddressAxisSummary axis;
      axis.extent = d;
      axis.regularity = "opaque_loaded";
      axis.offsetAxisInfo = "unstructured";
      pattern.axes.push_back(axis);
    }
    work.addressPatterns = {pattern};
    if (mask)
      work.predicateCounts[e] = 1;
    profile.target = "Ascend950PR/dav-c310";
    profile.baseSimtWarpCount = w;
    profile.simtIndirectLoadModel = profile.simtIndirectStoreModel =
        "prefix_mask_rank_factor_20261017";
    profile.simdIndirectLoadModel = profile.simdIndirectStoreModel =
        "simd_mask_envelope_20261019";
    (store ? work.indirectStoreTransactions : work.indirectLoadTransactions) =
        e;
  }
  StageMemoryCost cost(StageMode mode, int factor = 1, bool local = false,
                       int subblocks = 1) {
    return IndirectGatherMemoryCostModel().cost(
        stage, profile, {mode, factor, local, subblocks});
  }
  std::optional<double> value(const StageMemoryCost &cost) {
    return isStore ? cost.store : cost.load;
  }
  const char *pricing(const StageMemoryCost &cost) {
    return isStore ? cost.storePricing : cost.loadPricing;
  }
};
} // namespace

#endif
