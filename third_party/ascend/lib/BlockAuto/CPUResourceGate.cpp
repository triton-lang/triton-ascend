// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "IRAnalysis.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"

namespace mlir::triton::ascend::blockauto {
void registerCandidateDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, cf::ControlFlowDialect, DLTIDialect,
                  func::FuncDialect, LLVM::LLVMDialect, memref::MemRefDialect,
                  scf::SCFDialect, hacc::HACCDialect, hivm::HIVMDialect>();
}
OwningOpRef<ModuleOp> parseCandidateIR(StringRef input, MLIRContext &context) {
  std::string source = input.str();
  // BiSheng LLVM 20 emits ':' here; the host LLVM 22 DLTI syntax uses '='.
  StringRef old = "#dlti.target_system_spec<\"NPU\" :";
  if (auto pos = source.find(old.str()); pos != std::string::npos)
    source[pos + old.size() - 1] = '=';
  auto module =
      parseSourceString<ModuleOp>(source, ParserConfig(&context, false));
  check(bool(module), "cannot parse candidate final MLIR");
  return module;
}
CandidateIRAnalysis::CandidateIRAnalysis(Operation *op,
                                         AnalysisManager &manager)
    : module(cast<ModuleOp>(op)),
      layout(manager.getAnalysis<DataLayoutAnalysis>().getAtOrAbove(op)) {}
int64_t CandidateIRAnalysis::spec(StringRef key) const {
  auto *ctx = module->getContext();
  auto value = layout.getDevicePropertyValue(StringAttr::get(ctx, "NPU"),
                                             StringAttr::get(ctx, key));
  auto number = value ? dyn_cast<IntegerAttr>(*value) : IntegerAttr{};
  check(number && number.getInt() > 0,
        "missing target specification: " + key.str());
  return number.getInt();
}
std::string CandidateIRAnalysis::metaLibrary() const {
  auto bitcode =
      module->getAttrOfType<hivm::AIV_BITCODEAttr>("hivm.aiv_bitcode");
  return bitcode ? bitcode.getPath().getValue().str() : "";
}
void CandidateIRAnalysis::checkResources(const O &item) const {
  const O &metadata = sub(item, "metadata");
  std::string mode = str(metadata, "parallel_mode");
  check(str(sub(metadata, "target"), "arch") == "Ascend950PR_9579",
        "cost model target is outside the validated A5 profile");
  bool vv = str(metadata, "mix_mode") == "aiv",
       mixed = vv && mode == "mix_simd_simt", bound = false, hasUB = false;
  int64_t capacity = spec("UB_SIZE"),
          alignment = vv ? spec("UB_ALIGN_SIZE") : 8, end = -1;
  auto isUB = [](Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    auto space =
        memref
            ? dyn_cast_or_null<hivm::AddressSpaceAttr>(memref.getMemorySpace())
            : hivm::AddressSpaceAttr{};
    return space && space.getAddressSpace() == hivm::AddressSpace::UB;
  };
  module->walk([&](Operation *op) {
    hasUB |= llvm::any_of(op->getResultTypes(), isUB) ||
             llvm::any_of(op->getOperandTypes(), isUB);
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      check(!isUB(op->getResult(0).getType()),
            "unresolved UB allocation remains in final IR");
    auto cast = dyn_cast<hivm::PointerCastOp>(op);
    if (!cast || !isUB(cast.getType()))
      return;
    auto type = mlir::cast<MemRefType>(cast.getType());
    check(type.hasStaticShape() && type.getLayout().isIdentity(),
          "UB pointer has an unproved shape or layout");
    auto width = layout.getTypeSizeInBits(type.getElementType());
    check(!width.isScalable(), "scalable UB element size");
    int64_t bits = width.getFixedValue();
    bool conservative = vv && bits == 1;
    check(conservative || (bits >= 8 && bits % 8 == 0),
          "unsupported UB element width");
    A factors{conservative ? 8 : bits};
    for (int64_t dim : type.getShape())
      factors.push_back(dim);
    bits = product(factors);
    for (auto addr : cast.getAddrs()) {
      APInt offset;
      check(matchPattern(addr, m_ConstantInt(&offset)) &&
                offset.isNonNegative() && offset.isSignedIntN(64),
            "UB pointer has an unproved offset");
      int64_t start = offset.getSExtValue();
      check(bits <= INT64_MAX - alignment + 1 &&
                start <= (INT64_MAX - bits - alignment + 1) / 8,
            "UB extent overflow");
      end = std::max(end, start * 8 + int64_t(llvm::alignTo(bits, alignment)));
    }
    bound |= conservative;
  });
  if (end < 0) {
    check(!module->getRegion(0).getOps<func::FuncOp>().empty() && !vv &&
              !item.get("llvm_aiv") && !hasUB,
          "no verifiable static UB allocation in final IR");
    end = 0;
  }
  if (vv) {
    check(mode == "simd" || mixed,
          "only AIV SIMD and MIX resource proofs are supported");
    if (mixed) {
      int64_t dynamic = integer(metadata, "shared_mem_dynamic_size"),
              full = capacity;
      check(dynamic > 0 && dynamic <= INT64_MAX / 8,
            "missing runtime dynamic UB size");
      capacity = dynamic * 8;
      check(full - spec("MAXIMUM_D_CACHE_SIZE") <= capacity &&
                capacity <= full - spec("MINIMAL_D_CACHE_SIZE"),
            "MIX dynamic UB conflicts with D-cache reservation");
    }
  }
  check(end <= capacity, bound ? "i1 conservative UB upper bound exceeds "
                                 "capacity; actual overflow unconfirmed"
                               : "final UB address exceeds capacity");
}
} // namespace mlir::triton::ascend::blockauto
