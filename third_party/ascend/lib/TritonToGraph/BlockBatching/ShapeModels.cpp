// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/ShapeModels.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::cfg {
Operation *rewriteBatchShape(Operation *op, TypeRange results,
                             IRMapping &operands, IRRewriter &rewriter) {
  if (isa<triton::UnsplatOp>(op))
    return rewriter
        .create<triton::ReshapeOp>(op->getLoc(), results.front(),
                                   operands.lookup(op->getOperand(0)))
        .getOperation();
  Operation *result = rewriter.clone(*op, operands);
  for (auto [value, type] : llvm::zip(result->getResults(), results))
    value.setType(type);
  if (auto axis = op->getAttrOfType<IntegerAttr>("axis"))
    result->setAttr("axis", rewriter.getI32IntegerAttr(axis.getInt() + 1));
  if (auto order = op->getAttrOfType<DenseI32ArrayAttr>("order")) {
    SmallVector<int32_t> mapped{0};
    for (int32_t axis : order.asArrayRef())
      mapped.push_back(axis + 1);
    result->setAttr("order", rewriter.getDenseI32ArrayAttr(mapped));
  }
  return result;
}
} // namespace mlir::triton::cfg
