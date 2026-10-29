//===----------------------------------------------------------------------===//
// Lower generic Proton records to the Ascend HIVM profiling ABI.
//===----------------------------------------------------------------------===//

#include "ascend/include/ProtonToHIVM/Passes.h"

#include "Analysis/ScopeIdAllocation.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

constexpr int32_t kHeaderBytes = 40;
constexpr int32_t kCountVectorBytes = 4;
constexpr int32_t kCacheLineBytes = 64;
constexpr int32_t kDefaultDataSegmentBytes = 4096;
constexpr int32_t kDefaultSubBlocks = 1;
constexpr int32_t kProfileUnits = 1;
constexpr int32_t kRecordWords = 2;
constexpr int32_t kMaxScopeId = 0xFF;
constexpr uint32_t kPreamble = 0xDEADBEEF;

constexpr int32_t kOffPreamble = 0;
constexpr int32_t kOffBlockId = 1;
constexpr int32_t kOffProcId = 2;
constexpr int32_t kOffBufferSize = 3;
constexpr int32_t kOffInitTime = 4;
constexpr int32_t kOffPreFinalTime = 6;
constexpr int32_t kOffPostFinalTime = 8;
constexpr int32_t kOffCountVector = 10;

int32_t alignToCacheLine(int32_t bytes) {
  return (bytes + kCacheLineBytes - 1) & ~(kCacheLineBytes - 1);
}

struct AssignAscendProtonScopeIdsPass
    : PassWrapper<AssignAscendProtonScopeIdsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignAscendProtonScopeIdsPass)

  StringRef getArgument() const final {
    return "assign-ascend-proton-scope-ids";
  }
  StringRef getDescription() const final {
    return "Assign stable Proton scope IDs before Ascend TTIR lowering";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasRecords = false;
    module.walk([&](proton::RecordOp) { hasRecords = true; });
    if (!hasRecords)
      return;

    proton::ModuleScopeIdAllocation scopeInfo(module);
    Builder builder(module.getContext());
    bool invalid = false;
    module.walk([&](proton::RecordOp record) {
      size_t scopeId = scopeInfo.getOpScopeId(record);
      if (scopeId > kMaxScopeId) {
        record.emitOpError() << "Ascend instrumentation supports at most "
                             << kMaxScopeId + 1 << " scopes";
        invalid = true;
        return;
      }
      record->setAttr("proton.scope_id",
                      builder.getI32IntegerAttr(static_cast<int32_t>(scopeId)));
    });
    if (invalid)
      signalPassFailure();
  }
};

class ProtonFunctionLowering {
public:
  ProtonFunctionLowering(func::FuncOp func, int32_t dataSegmentBytes)
      : func(func), dataSegmentBytes(dataSegmentBytes),
        sectionBytes(alignToCacheLine(kHeaderBytes + kCountVectorBytes +
                                      dataSegmentBytes)) {}

  LogicalResult run() {
    SmallVector<proton::RecordOp> records;
    func.walk([&](proton::RecordOp record) { records.push_back(record); });
    if (records.empty())
      return success();
    if (!func->hasAttr("global_kernel")) {
      func.emitError("Ascend Proton records must be in the global kernel; "
                     "inline instrumented helper functions first");
      return failure();
    }
    constexpr unsigned kGridArgumentCount = 6;
    if (func.getNumArguments() < kGridArgumentCount) {
      func.emitError("Ascend Proton expected six trailing grid arguments");
      return failure();
    }

    if (failed(emitHeader()))
      return failure();
    if (failed(emitFooters()))
      return failure();

    for (proton::RecordOp record : records)
      if (failed(lowerRecord(record)))
        return failure();

    attachMetadata();
    return success();
  }

private:
  func::FuncOp func;
  int32_t dataSegmentBytes;
  int32_t sectionBytes;
  int32_t numSubBlocks = kDefaultSubBlocks;
  Value buffer;
  Value sectionOffset;

  Value readCounter(OpBuilder &builder, Location loc) {
    auto pipeAll =
        hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_ALL);
    builder.create<hivm::PipeBarrierOp>(loc, pipeAll);
    auto custom = builder.create<hivm::CustomOp>(
        loc, TypeRange{builder.getI64Type()},
        hivm::CustomOp::kBuiltinProtonGetSysCntName, ValueRange{}, ValueRange{},
        ValueRange{});
    custom.setCoreType(hivm::TCoreType::VECTOR);
    custom.setPipe(hivm::PIPE::PIPE_S);
    custom.setVFMode(hivm::VFMode::SIMD);
    custom.setInlineMode(hivm::InlineMode::AlwaysInline);
    custom.setGMAddrArgsIndices({});
    return custom.getResult(0);
  }

  void storeI32(OpBuilder &builder, Location loc, Value value,
                int32_t wordOffset) {
    auto offset = builder.create<arith::ConstantIndexOp>(loc, wordOffset);
    auto index = builder.create<arith::AddIOp>(loc, sectionOffset, offset);
    builder.create<memref::StoreOp>(loc, value, buffer, ValueRange{index});
  }

  void storeI64(OpBuilder &builder, Location loc, Value value,
                int32_t wordOffset) {
    auto i32Type = builder.getI32Type();
    auto mask = builder.create<arith::ConstantIntOp>(loc, 0xFFFFFFFFLL, 64);
    auto shift = builder.create<arith::ConstantIntOp>(loc, 32, 64);
    auto low = builder.create<arith::AndIOp>(loc, value, mask);
    auto highShifted = builder.create<arith::ShRUIOp>(loc, value, shift);
    auto high = builder.create<arith::AndIOp>(loc, highShifted, mask);
    storeI32(builder, loc, builder.create<arith::TruncIOp>(loc, i32Type, low),
             wordOffset);
    storeI32(builder, loc, builder.create<arith::TruncIOp>(loc, i32Type, high),
             wordOffset + 1);
  }

  LogicalResult emitHeader() {
    if (func.getBody().empty())
      return success();

    MLIRContext *ctx = func.getContext();
    auto i32Type = IntegerType::get(ctx, 32);
    auto i64Type = IntegerType::get(ctx, 64);
    auto gmSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::GM);
    auto bufferType = MemRefType::get({ShapedType::kDynamic}, i32Type,
                                      MemRefLayoutAttrInterface{}, gmSpace);

    constexpr unsigned kGridArgumentCount = 6;
    unsigned argIndex = func.getNumArguments() - kGridArgumentCount;
    if (failed(func.insertArgument(argIndex, bufferType,
                                   DictionaryAttr::get(ctx), func.getLoc()))) {
      func.emitError("failed to insert the Proton profiling buffer argument");
      return failure();
    }
    Block &entry = func.getBody().front();
    buffer = entry.getArgument(argIndex);

    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&entry);
    Location loc = func.getLoc();
    auto zeroI64 = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    auto zeroI32 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto blockId = builder.create<hivm::GetBlockIdxOp>(loc, i64Type);
    auto subBlockId = builder.create<hivm::GetSubBlockIdxOp>(loc, i64Type);
    auto subBlockCount =
        builder.create<arith::ConstantIntOp>(loc, numSubBlocks, 64);
    auto blockOffset =
        builder.create<arith::MulIOp>(loc, blockId, subBlockCount);
    auto originalSectionId =
        builder.create<arith::AddIOp>(loc, blockOffset, subBlockId);

    Value sampledSectionId = originalSectionId;
    auto sectionWords =
        builder.create<arith::ConstantIntOp>(loc, sectionBytes / 4, 64);
    auto sectionOffsetI64 =
        builder.create<arith::MulIOp>(loc, sampledSectionId, sectionWords);
    sectionOffset = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), sectionOffsetI64);

    auto emitStores = [&]() {
      auto preamble = builder.create<arith::ConstantIntOp>(
          loc, static_cast<int64_t>(kPreamble), 32);
      storeI32(builder, loc, preamble, kOffPreamble);
      auto blockIdI32 = builder.create<arith::TruncIOp>(loc, i32Type, blockId);
      storeI32(builder, loc, blockIdI32, kOffBlockId);
      auto procIdI32 =
          builder.create<arith::TruncIOp>(loc, i32Type, subBlockId);
      storeI32(builder, loc, procIdI32, kOffProcId);
      auto bufferSize =
          builder.create<arith::ConstantIntOp>(loc, dataSegmentBytes, 32);
      storeI32(builder, loc, bufferSize, kOffBufferSize);
      storeI64(builder, loc, readCounter(builder, loc), kOffInitTime);
      storeI64(builder, loc, zeroI64, kOffPreFinalTime);
      storeI64(builder, loc, zeroI64, kOffPostFinalTime);
      storeI32(builder, loc, zeroI32, kOffCountVector);
    };

    emitStores();
    return success();
  }

  LogicalResult emitFooters() {
    SmallVector<func::ReturnOp> returns;
    func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (func::ReturnOp ret : returns) {
      OpBuilder builder(ret);
      Location loc = ret.getLoc();
      auto emitStores = [&]() {
        storeI64(builder, loc, readCounter(builder, loc), kOffPreFinalTime);
        storeI64(builder, loc, readCounter(builder, loc), kOffPostFinalTime);
        builder.create<hivm::DCCIOp>(loc, hivm::DCCIMode::ALL_CACHE_LINES,
                                     hivm::DataCacheKind::ALL, Value{});
      };
      emitStores();
    }
    return success();
  }

  LogicalResult lowerRecord(proton::RecordOp record) {
    auto scopeAttr = record->getAttrOfType<IntegerAttr>("proton.scope_id");
    if (!scopeAttr) {
      record.emitOpError(
          "missing proton.scope_id; assign scope IDs before Ascend lowering");
      return failure();
    }

    int64_t scopeId = scopeAttr.getInt();
    if (scopeId < 0 || scopeId > kMaxScopeId) {
      record.emitOpError("scope ID is outside the Ascend ABI range");
      return failure();
    }

    OpBuilder builder(record);
    Location loc = record.getLoc();
    auto emitStore = [&]() {
      auto scopeIdValue =
          builder.create<arith::ConstantIntOp>(loc, scopeId, 32);
      auto isStartValue = builder.create<arith::ConstantIntOp>(
          loc, record.getIsStart() ? 1 : 0, 32);
      auto dataWordsValue =
          builder.create<arith::ConstantIntOp>(loc, dataSegmentBytes / 4, 32);
      SmallVector<Value> inputs{
          buffer,       sectionOffset, readCounter(builder, loc),
          scopeIdValue, isStartValue,  dataWordsValue};
      auto custom = builder.create<hivm::CustomOp>(
          loc, TypeRange{}, hivm::CustomOp::kBuiltinProtonCircularStoreName,
          inputs, ValueRange{}, ValueRange{});
      custom.setCoreType(hivm::TCoreType::VECTOR);
      custom.setPipe(hivm::PIPE::PIPE_S);
      custom.setVFMode(hivm::VFMode::SIMD);
      // Circular records carry a GM memref and are emitted inside Proton
      // scope/control-flow regions.  Keep this call out of the region
      // inliner: the LLVM lowering of an inlined library call would capture
      // the outer memref descriptor and violate LLVM region isolation.
      custom.setInlineMode(hivm::InlineMode::NoInline);
      custom.setGMAddrArgsIndices({0});
    };

    emitStore();
    record.erase();
    return success();
  }

  void attachMetadata() {
    Builder builder(func.getContext());
    func->setAttr("profile_scratch_size",
                  builder.getI32IntegerAttr(sectionBytes));
    func->setAttr("profile_scratch_align",
                  builder.getI32IntegerAttr(kCacheLineBytes));
    func->setAttr("profile_num_units",
                  builder.getI32IntegerAttr(kProfileUnits));
    func->setAttr("profile_sections_per_block",
                  builder.getI32IntegerAttr(numSubBlocks));
    func->setAttr("profile_sample_every_n", builder.getI32IntegerAttr(1));
  }
};

struct LowerAscendProtonToHIVMPass
    : PassWrapper<LowerAscendProtonToHIVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAscendProtonToHIVMPass)

  LowerAscendProtonToHIVMPass() = default;
  LowerAscendProtonToHIVMPass(const LowerAscendProtonToHIVMPass &other)
      : PassWrapper(other) {}

  Option<int32_t> dataSegmentBytes{
      *this, "data-segment-bytes",
      llvm::cl::desc("Per-section Proton circular data size in bytes"),
      llvm::cl::init(kDefaultDataSegmentBytes)};
  StringRef getArgument() const final { return "lower-ascend-proton-to-hivm"; }
  StringRef getDescription() const final {
    return "Lower generic Proton records to Ascend HIVM profiling operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, hivm::HIVMDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    constexpr int32_t kRecordBytes = kRecordWords * sizeof(int32_t);
    if (dataSegmentBytes < kRecordBytes ||
        dataSegmentBytes % kRecordBytes != 0) {
      getOperation().emitError(
          "Ascend Proton data-segment-bytes must be >= 8 and divisible by 8");
      return signalPassFailure();
    }
    bool failed = false;
    SmallVector<func::FuncOp> functions;
    getOperation().walk([&](func::FuncOp func) { functions.push_back(func); });
    for (func::FuncOp func : functions) {
      ProtonFunctionLowering lowering(func, dataSegmentBytes);
      if (mlir::failed(lowering.run())) {
        failed = true;
        break;
      }
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::triton::proton::createAssignAscendProtonScopeIdsPass() {
  return std::make_unique<AssignAscendProtonScopeIdsPass>();
}

std::unique_ptr<mlir::Pass>
mlir::triton::proton::createLowerAscendProtonToHIVMPass(
    int32_t dataSegmentBytes) {
  auto pass = std::make_unique<LowerAscendProtonToHIVMPass>();
  pass->dataSegmentBytes = dataSegmentBytes;
  return pass;
}

void mlir::triton::proton::registerAscendProtonToHIVMPasses() {
  PassRegistration<AssignAscendProtonScopeIdsPass>();
  PassRegistration<LowerAscendProtonToHIVMPass>();
}

namespace {
[[maybe_unused]] const bool ascendProtonPassesRegistered = [] {
  mlir::triton::proton::registerAscendProtonToHIVMPasses();
  return true;
}();
} // namespace
