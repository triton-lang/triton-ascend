/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ASCEND_UTILS_PTR_ANALYSIS_UTILS_H
#define TRITON_ASCEND_UTILS_PTR_ANALYSIS_UTILS_H

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "TritonMemoryAccess/PtrAnalysis.h"
#include "TritonMemoryAccess/Utils.h"
#include "Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
// CFO owns these qualitative proofs. Import them at the consumer boundary;
// obtain numerical origins/strides from the current complete offset SSA.
// The public analysis remains independent of conversion-pass attributes.
inline LogicalResult importPointerDescriptorFacts(Operation *root,
                                                  ptr::PtrAnalysis &analysis) {
  SmallVector<triton::AddPtrOp> rebuilds;
  root->walk([&](triton::AddPtrOp op) {
    if (op->hasAttr(controlflow::kPointerDescriptorStructuredAxesAttr))
      rebuilds.push_back(op);
  });
  SmallVector<std::pair<Value, ptr::PtrInfo>, 0> known;
  OpBuilder &builder = analysis.getBuilder();
  OpBuilder::InsertionGuard guard(builder);
  for (auto rebuild : rebuilds) {
    auto axes = rebuild->getAttrOfType<DenseI32ArrayAttr>(
        controlflow::kPointerDescriptorStructuredAxesAttr);
    auto type = dyn_cast<RankedTensorType>(rebuild.getOffset().getType());
    // These proofs describe the offset independently of the pointer base.
    // CFO also rebuilds pointers with a different base in each lane.
    if (!rebuild->hasAttr(controlflow::kPointerDescriptorRebuildAttr) ||
        !axes || !type || !type.hasStaticShape() ||
        axes.size() != type.getRank() ||
        llvm::any_of(axes.asArrayRef(),
                     [](int32_t axis) { return axis != 0 && axis != 1; }))
      return rebuild.emitError("invalid pointer descriptor axis handoff");
    if (auto form =
            rebuild->getAttr(controlflow::kPointerDescriptorOffsetFormAttr)) {
      auto name = dyn_cast<StringAttr>(form);
      if (!name || name.getValue() != controlflow::kStrided1DOffsetForm ||
          type.getRank() != 1 || axes[0] != 1)
        return rebuild.emitError("invalid pointer descriptor offset form");
    }
    auto info = analysis.analyze(rebuild.getOffset());
    if (failed(info))
      return failure();
    bool needsProof =
        llvm::any_of(llvm::enumerate(axes.asArrayRef()), [&](auto axis) {
          return axis.value() == 1 && info->offsets.axes[axis.index()].kind ==
                                          ptr::AxisKind::Unknown;
        });
    if (!needsProof)
      continue;
    // The imported key can have users before this rebuild or in sibling
    // regions. Its sampled fields must be available to all of those users.
    builder.setInsertionPointAfterValue(rebuild.getOffset());
    SmallVector<Value> indices(
        type.getRank(),
        builder.create<arith::ConstantIndexOp>(rebuild.getLoc(), 0));
    auto sample = [&]() -> Value {
      Value value = builder.create<tensor::ExtractOp>(
          rebuild.getLoc(), rebuild.getOffset(), indices);
      return *castIntegerLike(builder, rebuild.getLoc(), value,
                              builder.getIndexType());
    };
    Value origin = sample();
    info->offsets.scalarOffset = origin;
    for (auto [dim, proved] : llvm::enumerate(axes.asArrayRef())) {
      if (!proved || info->offsets.axes[dim].kind != ptr::AxisKind::Unknown)
        continue;
      if (type.getDimSize(dim) == 1) {
        info->offsets.axes[dim] = {ptr::AxisKind::Invariant,
                                   builder.getIndexAttr(0)};
        continue;
      }
      Value zero = indices[dim];
      indices[dim] =
          builder.create<arith::ConstantIndexOp>(rebuild.getLoc(), 1);
      Value stride =
          builder.create<arith::SubIOp>(rebuild.getLoc(), sample(), origin);
      indices[dim] = zero;
      info->offsets.axes[dim] = {ptr::AxisKind::Structured, stride};
    }
    known.emplace_back(rebuild.getOffset(), *info);
  }
  if (failed(analysis.reset(known)))
    return failure();
  SmallVector<Operation *> customOps;
  root->walk([&](Operation *op) {
    if (isDistributedTypeCustomOp(op))
      customOps.push_back(op);
  });
  for (Operation *op : customOps) {
    auto sources = op->getAttrOfType<DenseI32ArrayAttr>(
        ConverterUtils::customSrcPtrIndexAttrName);
    if (!sources)
      continue;
    for (auto result : op->getResults()) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !isa<triton::PointerType>(type.getElementType()))
        continue;
      unsigned index = result.getResultNumber();
      if (index >= sources.size() || sources[index] < 0 ||
          sources[index] >= op->getNumOperands())
        return op->emitError("invalid custom pointer layout source");
      auto layout = analysis.analyze(op->getOperand(sources[index]));
      if (failed(layout) || !layout->hasScalarBase() || layout->block ||
          layout->offsets.getShape() != type.getShape())
        return op->emitError(
            "custom pointer result requires a compatible scalar-base layout");
      // A distributed custom result is converted to one memref allocation.
      // This explicit type bridge names that allocation; SrcPtrIndex
      // contributes its layout, never the input allocation's address.
      Value base;
      for (Operation *user : result.getUsers()) {
        if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
            cast && cast.getInputs().size() == 1 &&
            cast.getOutputs().size() == 1 &&
            cast.getOutputs()[0].getType() == type.getElementType()) {
          base = cast.getOutputs()[0];
          break;
        }
      }
      if (!base) {
        builder.setInsertionPointAfter(op);
        base = builder
                   .create<UnrealizedConversionCastOp>(
                       op->getLoc(), type.getElementType(), result)
                   .getResult(0);
      }
      layout->base = base;
      if (failed(analysis.setInfo(result, *layout)))
        return failure();
    }
  }
  return success();
}
} // namespace mlir::triton
#endif
