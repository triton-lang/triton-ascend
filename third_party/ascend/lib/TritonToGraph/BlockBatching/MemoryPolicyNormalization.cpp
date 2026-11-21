// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "TritonToGraph/BlockBatching/MemoryModels.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::cfg {
LogicalResult normalizeBatchMemory(ModuleOp module, Operation *&failedOp,
                                   std::string &reason) {
  auto reject = [&](Operation *op, llvm::StringRef why) {
    failedOp = op;
    reason = why.str();
    return WalkResult::interrupt();
  };
  auto tensorPointer = [](Type type) {
    auto ptr = dyn_cast<triton::PointerType>(type);
    return ptr && isa<RankedTensorType>(ptr.getPointeeType());
  };
  bool hasBlockPointers = false;
  auto checked = module.walk([&](Operation *op) -> WalkResult {
    // These hints describe the old shape. Recompute rather than applying a
    // rank-one promise to the new batch dimension. Unknown attrs stay checked.
    for (llvm::StringRef name :
         {"tt.contiguity", "tt.divisibility", "tt.constancy"}) {
      Attribute hint = op->getAttr(name);
      if (!hint)
        continue;
      if (op->getNumResults() != 1)
        return reject(op, "INVALID_AXIS_HINT");
      auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      unsigned rank = type ? type.getRank() : 1;
      auto dense = dyn_cast<DenseIntElementsAttr>(hint);
      if (!dense || dense.getType().getRank() != 1 ||
          !dense.getElementType().isInteger(32) ||
          dense.getNumElements() != rank)
        return reject(op, "INVALID_AXIS_HINT");
      for (APInt value : dense.getValues<APInt>())
        if (value.isNegative() || value.isZero())
          return reject(op, "INVALID_AXIS_HINT");
      op->removeAttr(name);
    }
    bool usesBlockPointer =
        llvm::any_of(op->getOperandTypes(), tensorPointer) ||
        llvm::any_of(op->getResultTypes(), tensorPointer);
    if (!usesBlockPointer)
      return WalkResult::advance();
    hasBlockPointers = true;
    // The stock pass assumes make/advance provenance. Exclude block arguments,
    // selects, calls and loop-carried pointers before entering that pass.
    if (!isa<triton::MakeTensorPtrOp, triton::AdvanceOp, triton::LoadOp,
             triton::StoreOp>(op))
      return reject(op, "UNSUPPORTED_BLOCK_POINTER_FLOW");
    for (Value operand : op->getOperands())
      if (tensorPointer(operand.getType()) &&
          !isa_and_nonnull<triton::MakeTensorPtrOp, triton::AdvanceOp>(
              operand.getDefiningOp()))
        return reject(op, "UNSUPPORTED_BLOCK_POINTER_FLOW");
    if (isa<triton::MakeTensorPtrOp, triton::AdvanceOp>(op)) {
      for (NamedAttribute attr : op->getAttrs())
        if (!isa<triton::MakeTensorPtrOp>(op) || attr.getName() != "order")
          return reject(op, "UNSUPPORTED_OPERATION_ATTRIBUTE");
      auto ptr = cast<triton::PointerType>(op->getResult(0).getType());
      auto tile = cast<RankedTensorType>(ptr.getPointeeType());
      if (ptr.getAddressSpace() != 1 || tile.getEncoding() ||
          !tile.hasStaticShape() || tile.getRank() < 1 || tile.getRank() > 8)
        return reject(op, "UNSUPPORTED_BLOCK_POINTER_TYPE");
      int64_t elements = 1;
      for (int64_t extent : tile.getShape()) {
        if (extent <= 0 || extent > (1 << 20) / elements)
          return reject(op, "UNSUPPORTED_BLOCK_POINTER_TYPE");
        elements *= extent;
      }
    } else {
      auto *descriptor = lookupBatchOp(op->getName().getStringRef());
      if (!descriptor || failed(verifyBatchAttributes(op, *descriptor)))
        return reject(op, "UNSUPPORTED_OPERATION_ATTRIBUTE");
      if (auto load = dyn_cast<triton::LoadOp>(op)) {
        // Pick zero for unspecified masked-out values; explicit zero/NaN
        // padding and all memory policy attrs survive stock normalization.
        if (load.getBoundaryCheck().empty())
          load->removeAttr("padding");
        else if (!load.getPadding())
          load->setAttr("padding", triton::PaddingOptionAttr::get(
                                       module.getContext(),
                                       triton::PaddingOption::PAD_ZERO));
      }
    }
    return WalkResult::advance();
  });
  if (checked.wasInterrupted())
    return failure();
  if (!hasBlockPointers)
    return success();
  PassManager pm(module.getContext());
  pm.addPass(triton::createTritonRewriteTensorPointer());
  if (failed(pm.run(module))) {
    failedOp = module;
    reason = "BLOCK_POINTER_NORMALIZATION_FAILED";
    return failure();
  }
  return success();
}
} // namespace mlir::triton::cfg
