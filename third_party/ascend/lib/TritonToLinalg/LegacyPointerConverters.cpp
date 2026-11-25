/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
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

// Preserve descriptor-based entry points for consumers such as TLAToLinalg.
// TritonToLinalg selects public analysis by passing MemoryPointerInfos and does
// not register these pointer-producer patterns.
#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "TritonToLinalg/LoadStoreConverter.h"
#include "TritonToLinalg/TritonOpConverter.h"
#include "Utils/DebugUtils.h"
#include "Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace LoadStoreConverter {
static bool shouldMaterializeCustomPointer(Value ptr) {
  auto result = dyn_cast<OpResult>(ptr);
  if (!result)
    return false;

  Operation *defOp = result.getOwner();
  if (!defOp || !isDistributedTypeCustomOp(defOp))
    return false;

  auto srcIndices = defOp->getAttrOfType<DenseI32ArrayAttr>(
      ConverterUtils::customSrcPtrIndexAttrName);
  if (!srcIndices)
    return false;

  auto values = srcIndices.asArrayRef();
  unsigned resultIdx = result.getResultNumber();
  return resultIdx < values.size() && values[resultIdx] >= 0;
}

static FailureOr<Value>
resolveMemoryPointer(Value originalPtr, Value convertedPtr,
                     ConversionPatternRewriter &rewriter) {
  llvm::SmallDenseMap<Value, BlockData> known;

  if (shouldMaterializeCustomPointer(originalPtr))
    return BlockDataParser::materializePointer(originalPtr, rewriter, known);

  if (isa<MemRefType>(convertedPtr.getType()))
    return convertedPtr;

  return BlockDataParser::materializePointer(originalPtr, rewriter, known);
}

LogicalResult
AddPtrConverter::matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  insertDebugNop(loc, rewriter);
  llvm::SmallDenseMap<Value, BlockData> known;
  return BlockDataParser::rewriteAddPtr(op, adaptor, rewriter, known);
}

FailureOr<Value> LoadConverter::materializeLegacyPointer(
    triton::LoadOp op, Value convertedPtr,
    ConversionPatternRewriter &rewriter) const {
  auto pointer = resolveMemoryPointer(op.getPtr(), convertedPtr, rewriter);
  if (failed(pointer))
    return failure();
  if (!isa<ShapedType>(op.getType()))
    return BlockDataParser::getScalarMemRef(op.getPtr(), *pointer, op.getLoc(),
                                            rewriter);
  return *pointer;
}

LogicalResult LoadConverter::rewriteLegacyIndirectLoad(
    triton::LoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!isa<scf::ForOp>(op->getParentOp()) || !op->hasAttr("IndirectLoad"))
    return failure();
  Operation *ptrOp = op.getPtr().getDefiningOp();
  if (!ptrOp || ptrOp->getBlock() == op->getBlock())
    return failure();

  auto loc = op.getLoc();
  auto forOp = op->getParentOfType<scf::ForOp>();
  Operation *firstOp = &forOp.getBody()->front();
  auto extractOp = cast<tensor::ExtractOp>(firstOp);
  auto ivs = extractOp.getIndices();
  // Single iterArg which is inserted by AddPtrConverter.
  auto iterArg = forOp.getRegionIterArg(0);
  auto ptr = adaptor.getPtr();

  rewriter.setInsertionPointAfter(op);
  Value castVal = ptr.getDefiningOp<memref::ReinterpretCastOp>();
  Value idxZero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value loadVal =
      rewriter.create<memref::LoadOp>(loc, castVal, ValueRange{idxZero});
  propagateWasBoolToInt8Attr(op.getOperation(), loadVal.getDefiningOp(),
                             rewriter);
  Value insertedVal =
      rewriter.create<tensor::InsertOp>(loc, loadVal, iterArg, ValueRange{ivs});
  // a yield op is already created by AddPtrConverter.
  // so we need to replace it with a new yield op.
  Operation *terminator = forOp.getBody()->getTerminator();
  scf::YieldOp oldYieldOp = cast<scf::YieldOp>(terminator);
  auto yieldOp = rewriter.create<scf::YieldOp>(loc, ValueRange{insertedVal});
  rewriter.replaceOp(oldYieldOp, yieldOp);
  // Now the scf.for is complete, we can replace tt.load with it.
  auto rank = cast<ShapedType>(op.getResult().getType()).getShape().size();
  Operation *rootForOp = op;
  while (rank != 0) {
    rank--;
    rootForOp = rootForOp->getParentOfType<scf::ForOp>();
  }
  rewriter.replaceOp(op, rootForOp);
  return success();
}

void LoadConverter::getLegacyBoundary(
    triton::LoadOp op, Value ptr, ConversionPatternRewriter &rewriter,
    SmallVectorImpl<OpFoldResult> &srcOffsets,
    SmallVectorImpl<OpFoldResult> &dstOffsets,
    SmallVectorImpl<OpFoldResult> &sizes) const {
  auto loc = op.getLoc();
  auto boundaryCheck = op.getBoundaryCheck();
  auto makeTensorPtrOp = op.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
  sizes = mlir::ConverterUtils::getBoundarySizes(
      boundaryCheck, /*remapped*/ ptr, loc, rewriter);
  srcOffsets.assign(sizes.size(), rewriter.getIndexAttr(0));
  if (makeTensorPtrOp) {
    auto zeroVal = rewriter.createOrFold<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(0));
    for (auto [idx, offVal] : llvm::enumerate(makeTensorPtrOp.getOffsets())) {
      if (llvm::find(boundaryCheck, idx) == boundaryCheck.end()) {
        dstOffsets.push_back(srcOffsets[idx]);
        continue;
      }
      Value offset = rewriter.createOrFold<arith::SubIOp>(loc, zeroVal, offVal);
      Value size = getValueOrCreateConstantIndexOp(rewriter, loc, sizes[idx]);
      offset = rewriter.createOrFold<arith::MaxSIOp>(loc, offset, zeroVal);
      offset = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), offset);
      OpFoldResult ofr;
      if (auto constOp = offset.getDefiningOp<arith::ConstantOp>()) {
        ofr = constOp.getValue();
      } else {
        ofr = offset;
      }
      ofr = minOpFoldResult(ofr, size, loc, rewriter);
      sizes[idx] = subOpFoldResult(size, ofr, loc, rewriter);
      dstOffsets.push_back(ofr);
    }
  } else {
    dstOffsets = srcOffsets;
  }
}
} // namespace LoadStoreConverter

namespace TTOpConverters {
LogicalResult MakeTensorPtrConverter::matchAndRewrite(
    triton::MakeTensorPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, BlockData> known;
  return BlockDataParser::rewriteMakeTensorPtrOp(op, adaptor.getBase(),
                                                 rewriter, known);
}

} // namespace TTOpConverters
