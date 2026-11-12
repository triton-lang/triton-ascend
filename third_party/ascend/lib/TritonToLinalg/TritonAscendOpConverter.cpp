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

#include "TritonToLinalg/TritonAscendOpConverter.h"
#include "Utils/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallString.h"

#include <algorithm>
#include <iterator>
#include <string>

namespace TritonAscendOpConverters {
using namespace mlir;
using namespace triton;

// Fractal (zN) block geometry for ascend.dot lowering comes from Utils.h,
// shared with DotOp::inferReturnTypes.

// zN block column count b = 32 bytes / element bytes.
static int64_t nzBlockCol(Type elemTy) {
  return triton::ascend::kBytesPerFractalCol /
         (elemTy.getIntOrFloatBitWidth() / triton::ascend::kBitsPerByte);
}

static hivm::DataLayoutAttr dotNdLayout(MLIRContext *ctx) {
  return hivm::DataLayoutAttr::get(ctx, hivm::DataLayout::ND);
}

static hivm::DataLayoutAttr dotFractalLayout(MLIRContext *ctx, int64_t blockRow,
                                             int64_t blockCol) {
  return hivm::DataLayoutAttr::get(
      ctx, hivm::DataLayout::Fractal, /*transpose=*/nullptr,
      DenseI64ArrayAttr::get(ctx, {blockRow, blockCol}));
}

// A 4D fractal dot operand -> 2D ND (no-op if already ND). Both operands are zN
// (non-transpose): lhs [K1,M1,16,b] -> [M,K] and rhs [N1,K1,16,b] -> [K,N],
// each = [d1*d2, d0*d3].
static Value dotFractalToND(ConversionPatternRewriter &rewriter, Location loc,
                            Value operand, bool isLhs) {
  auto operandTy = cast<RankedTensorType>(operand.getType());
  if (operandTy.getRank() != 4)
    return operand;
  auto blockedShape = operandTy.getShape();
  SmallVector<int64_t> ndShape{blockedShape[1] * blockedShape[2],
                               blockedShape[0] * blockedShape[3]};
  MLIRContext *ctx = rewriter.getContext();
  return rewriter.create<hivm::ConvertLayoutOp>(
      loc, RankedTensorType::get(ndShape, operandTy.getElementType()), operand,
      /*srcLayout=*/dotFractalLayout(ctx, blockedShape[2], blockedShape[3]),
      /*dstLayout=*/dotNdLayout(ctx), DenseI64ArrayAttr::get(ctx, ndShape),
      /*output_shape=*/ValueRange{});
}

// If `v` is a freshly-loaded buffer used only here -- to_tensor of an alloc
// with a single writer copy (the load) -- return that load copy so its DMA can
// be retargeted into the padded buffer (one copy, not load-then-copy); else
// null.
static memref::CopyOp findRedirectableLoad(Value v) {
  auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor || v.hasNUsesOrMore(2))
    return nullptr;
  Value srcBuf = toTensor.getBuffer();
  if (!srcBuf.getDefiningOp<memref::AllocOp>())
    return nullptr;
  memref::CopyOp loadCopy = nullptr;
  for (Operation *user : srcBuf.getUsers()) {
    if (user == toTensor.getOperation())
      continue;
    // Bail on a non-copy user, a source-use of srcBuf, or a second writer.
    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getTarget() != srcBuf || loadCopy)
      return nullptr;
    loadCopy = copyOp;
  }
  return loadCopy;
}

// Zero-pad `v`'s dimension `dim` up to `target` (no-op if already that size).
// Uses memref.alloc + fill(0) + subview + copy, NOT tensor.insert_slice: the
// latter lowers (in AscendNPU-IR AutoVectorizeV2) to a scalar copy loop that
// blows up cbuf.
static Value zeroPadDimTo(Value v, int64_t dim, int64_t target,
                          ConversionPatternRewriter &rewriter, Location loc) {
  auto t = cast<RankedTensorType>(v.getType());
  if (t.getShape()[dim] == target)
    return v;
  Type elemTy = t.getElementType();

  // Push the pad through a transpose (pad the mapped input dim, re-transpose).
  if (auto trans = v.getDefiningOp<linalg::TransposeOp>()) {
    ArrayRef<int64_t> perm = trans.getPermutation();
    Value paddedInput =
        zeroPadDimTo(trans.getInput(), perm[dim], target, rewriter, loc);
    SmallVector<int64_t> outShape(t.getShape().begin(), t.getShape().end());
    outShape[dim] = target;
    Value init = rewriter.create<tensor::EmptyOp>(loc, outShape, elemTy);
    return rewriter.create<linalg::TransposeOp>(loc, paddedInput, init, perm)
        .getResults()[0];
  }

  SmallVector<int64_t> paddedShape(t.getShape().begin(), t.getShape().end());
  paddedShape[dim] = target;

  // Allocate the padded buffer and zero-fill it.
  auto paddedMemTy = MemRefType::get(paddedShape, elemTy);
  Value padded = rewriter.create<memref::AllocOp>(loc, paddedMemTy);
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemTy));
  rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{padded});

  // Subview covering the real (unpadded) region, then fill it with one copy.
  int64_t rank = t.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes;
  for (int64_t d : t.getShape())
    sizes.push_back(rewriter.getIndexAttr(d));
  auto subviewTy =
      memref::SubViewOp::inferResultType(paddedMemTy, offsets, sizes, strides);
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, cast<MemRefType>(subviewTy), padded, offsets, sizes, strides);

  if (memref::CopyOp loadCopy = findRedirectableLoad(v)) {
    rewriter.create<memref::CopyOp>(loc, loadCopy.getSource(), subview);
    rewriter.eraseOp(loadCopy);
  } else {
    auto srcBufferTy = MemRefType::get(t.getShape(), elemTy);
    Value srcBuffer =
        rewriter.create<bufferization::ToMemrefOp>(loc, srcBufferTy, v);
    rewriter.create<memref::CopyOp>(loc, srcBuffer, subview);
  }

  return rewriter.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get(paddedShape, elemTy), padded,
      /*restrict=*/true, /*writable=*/true);
}

// Reconcile the ND operands' contraction (K) dim (a/b mutated in place). When
// exactly one operand is fractal, its K is block-rounded while the ND side may
// be shorter; zero-padding the ND side to the block-aligned K is safe (the
// extra K products are * 0). Only an in-one-block gap is padding; else error.
static LogicalResult
reconcileDotContractionK(triton::ascend::DotOp op, Value &a, Value &b,
                         Type elemTy, ConversionPatternRewriter &rewriter,
                         Location loc) {
  if (op.getFractalA() == op.getFractalB())
    return success();
  int64_t kA = cast<RankedTensorType>(a.getType()).getShape()[1]; // A_nd [M,kA]
  int64_t kB = cast<RankedTensorType>(b.getType()).getShape()[0]; // B_nd [kB,N]
  if (kA == kB)
    return success();
  int64_t paddedK = std::max(kA, kB), shortK = std::min(kA, kB);
  int64_t blockK = nzBlockCol(elemTy);
  if (paddedK % blockK != 0 || paddedK - shortK >= blockK)
    return op.emitError()
           << "dot: contraction dim mismatch (A K=" << kA << ", B K=" << kB
           << ") is larger than fractal padding; pad the operand "
              "explicitly";
  op.emitWarning() << "dot: auto zero-padding the non-fractal operand's K from "
                   << shortK << " to " << paddedK
                   << " to match the fractal operand's block-aligned K";
  a = zeroPadDimTo(a, /*dim=*/1, paddedK, rewriter, loc); // A_nd [M,K]
  b = zeroPadDimTo(b, /*dim=*/0, paddedK, rewriter, loc); // B_nd [K,N]
  return success();
}

// Decompose `ascend.dot` (D = A * B): fractal inputs -> ND, a plain
// linalg.matmul, then (if fractal_c) the ND result back to fractal.
LogicalResult
DotConverter::matchAndRewrite(triton::ascend::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  auto resTy = cast<RankedTensorType>(op.getType());
  // The result carries the cube accumulator dtype (f32/i32); the matmul inputs
  // and K-padding use the operands' own (input) dtype.
  Type accTy = resTy.getElementType();

  Value a = op.getFractalA()
                ? dotFractalToND(rewriter, loc, adaptor.getA(), /*isLhs=*/true)
                : adaptor.getA();
  Value b = op.getFractalB()
                ? dotFractalToND(rewriter, loc, adaptor.getB(), /*isLhs=*/false)
                : adaptor.getB();
  Type inElemTy = cast<RankedTensorType>(a.getType()).getElementType();
  if (failed(reconcileDotContractionK(op, a, b, inElemTy, rewriter, loc)))
    return failure();

  // Uninitialized accumulator (accTy): mmadL1 (accumulate=false) overwrites it.
  int64_t numRows = cast<RankedTensorType>(a.getType()).getShape()[0];
  int64_t numCols = cast<RankedTensorType>(b.getType()).getShape()[1];
  auto ndResultTy = RankedTensorType::get({numRows, numCols}, accTy);
  Value acc = rewriter.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{numRows, numCols}, accTy);
  auto matmul = rewriter.create<linalg::MatmulOp>(
      loc, TypeRange{ndResultTy}, ValueRange{a, b}, ValueRange{acc});
  matmul->setAttr("input_precision", rewriter.getStringAttr("ieee"));
  Value ndResult = matmul->getResult(0);

  if (!op.getFractalC()) {
    rewriter.replaceOp(op, ndResult);
    return success();
  }
  // fractal_c: ND result -> the L0C accumulator fractal (block [16, 16]).
  Value fractalResult = rewriter.create<hivm::ConvertLayoutOp>(
      loc, resTy, ndResult, /*srcLayout=*/dotNdLayout(ctx),
      /*dstLayout=*/
      dotFractalLayout(ctx, triton::ascend::kFractalBlock,
                       triton::ascend::kFractalBlock),
      DenseI64ArrayAttr::get(ctx, resTy.getShape()), /*output_shape=*/
      ValueRange{});
  rewriter.replaceOp(op, fractalResult);
  return success();
}

LogicalResult
FlipOpConverter::matchAndRewrite(triton::ascend::FlipOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value src = adaptor.getSrc();
  auto rankedSrcTy = cast<RankedTensorType>(src.getType());

  MLIRContext *ctx = rewriter.getContext();

  Type valuesTy = src.getType();
  Location loc = op.getLoc();

  auto dimAttr = op->getAttrOfType<IntegerAttr>("dim");
  if (!dimAttr) {
    op->emitError("missing 'dim' attribute");
    return failure();
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    op->emitError("must be inside a module");
    return failure();
  }

  // Unique callee name: triton_flip, triton_flip_1, …
  std::string funcName = baseFuncName.str();
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName))
    funcName = (baseFuncName + Twine("_") + Twine(uniqueId++)).str();

  auto i64Ty = IntegerType::get(ctx, 64);
  auto libFnType =
      rewriter.getFunctionType({rankedSrcTy, i64Ty}, {rankedSrcTy});

  // Declare the callee
  auto moduleIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName, libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  rewriter.restoreInsertionPoint(moduleIP);

  // dim constant
  Value dimVal =
      rewriter.create<arith::ConstantIntOp>(loc, dimAttr.getInt(), 64);

  // Call the backend function
  auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
  auto callOp = rewriter.create<func::CallOp>(
      loc, TypeRange({rankedSrcTy}), callee, ValueRange({src, dimVal}));

  Value finalValues = callOp.getResult(0);

  rewriter.replaceOp(op, {finalValues});
  return success();
}

LogicalResult
SortOpConverter::matchAndRewrite(triton::ascend::SortOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value src = adaptor.getSrc();
  auto rankedSrcTy = cast<RankedTensorType>(src.getType());
  auto srcElemTy = rankedSrcTy.getElementType();
  auto srcShape = rankedSrcTy.getShape();
  auto srcEnc = rankedSrcTy.getEncoding();

  MLIRContext *ctx = rewriter.getContext();

  Type backendElemTy = srcElemTy;
  if (srcElemTy.isInteger(8)) {
    backendElemTy = Float16Type::get(ctx); // i8 -> f16
  } else if (srcElemTy.isInteger(16)) {
    backendElemTy = Float32Type::get(ctx); // i16 -> f32
  }
  Type backendTensorTy = RankedTensorType::get(srcShape, backendElemTy, srcEnc);

  Type valuesTy = src.getType();

  Location loc = op.getLoc();
  auto dimAttr = op->getAttrOfType<IntegerAttr>("dim");
  auto descAttr = op->getAttrOfType<BoolAttr>("descending");
  if (!dimAttr || !descAttr) {
    op->emitError("missing 'dim' or 'descending' attribute");
    return failure();
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    op->emitError("must be inside a module");
    return failure();
  }

  llvm::SmallString<64> baseName("triton_sort");
  llvm::SmallString<64> funcName = baseName;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName = baseName;
    funcName += ("_" + std::to_string(uniqueId++));
  }

  auto i64Ty = IntegerType::get(ctx, 64);
  auto i1Ty = IntegerType::get(ctx, 1);
  auto libFnType = rewriter.getFunctionType({backendTensorTy, i64Ty, i1Ty},
                                            {backendTensorTy});

  auto moduleIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  rewriter.restoreInsertionPoint(moduleIP);

  Value srcForCall = src;
  if (backendElemTy != srcElemTy) {
    srcForCall = rewriter.create<arith::SIToFPOp>(loc, backendTensorTy, src);
  }

  Value dimVal =
      rewriter.create<arith::ConstantIntOp>(loc, dimAttr.getInt(), 64);
  Value descVal = rewriter.create<arith::ConstantIntOp>(
      loc, descAttr.getValue() ? 1 : 0, 1);

  auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
  auto callOp =
      rewriter.create<func::CallOp>(loc, TypeRange({backendTensorTy}), callee,
                                    ValueRange({srcForCall, dimVal, descVal}));

  Value valuesFloat = callOp.getResult(0); // tensor<f16/f32>

  Value finalValues = valuesFloat;
  if (backendElemTy != srcElemTy) {
    finalValues = rewriter.create<arith::FPToSIOp>(loc, valuesTy, valuesFloat);
  }

  rewriter.replaceOp(op, {finalValues});

  return success();
}

LogicalResult
IndexPutConverter::matchAndRewrite(triton::ascend::IndexPutOp op,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName =
      ConverterUtils::generateUniqueFuncName(moduleOp, funcNameBase);

  auto ptr = adaptor.getPtr();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto pointer =
        materializeMemoryPointer(op.getPtr(), ptr, pointerInfos, rewriter);
    if (failed(pointer))
      return failure();
    ptr = *pointer;
  }
  auto index = op.getIndex();
  auto value = op.getValue();
  auto dim = op.getDim();
  auto indexBoundary = op.getIndexBoundary();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();
  auto dstStride = adaptor.getDstStride();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto ptrTy = dyn_cast<MemRefType>(ptr.getType());
  if (!ptrTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for ptr");
  }
  SmallVector<Type> inputTypes({ptrTy, index.getType(), value.getType(),
                                dim.getType(), indexBoundary.getType()});
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());
  inputTypes.append(dstStride.getTypes().begin(), dstStride.getTypes().end());
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({ptr, index, value, dim, indexBoundary});
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  inputVals.append(dstStride.begin(), dstStride.end());
  rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}),
                                inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult GatherOutToUbConverter::matchAndRewrite(
    triton::ascend::GatherOutToUbOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName =
      ConverterUtils::generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto pointer =
        materializeMemoryPointer(op.getSrc(), src, pointerInfos, rewriter);
    if (failed(pointer))
      return rewriter.notifyMatchFailure(op,
                                         "cannot materialize memory address");
    src = *pointer;
  }
  auto index = op.getIndex();
  auto indexBoundary = op.getIndexBoundary();
  auto dim = op.getDim();
  auto srcStride = op.getSrcStride();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();
  auto other = op.getOther();

  auto res = op.getResult();
  auto resTy = res.getType();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }

  SmallVector<Type> inputTypes(
      {srcTy, index.getType(), indexBoundary.getType(), dim.getType()});
  inputTypes.append(srcStride.getTypes().begin(), srcStride.getTypes().end());
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());
  if (other)
    inputTypes.push_back(other.getType());

  auto libFnType = rewriter.getFunctionType(inputTypes, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, index, indexBoundary, dim});
  inputVals.append(srcStride.begin(), srcStride.end());
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  if (other)
    inputVals.push_back(other);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}), inputVals);
  rewriter.replaceOp(op, callOp);
  return success();
}

LogicalResult ScatterUbToOutConverter::matchAndRewrite(
    triton::ascend::ScatterUbToOutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName =
      ConverterUtils::generateUniqueFuncName(moduleOp, funcNameBase);

  auto ptr = adaptor.getPtr();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto pointer =
        materializeMemoryPointer(op.getPtr(), ptr, pointerInfos, rewriter);
    if (failed(pointer))
      return failure();
    ptr = *pointer;
  }
  auto value = op.getValue();
  auto index = op.getIndex();
  auto indexBoundary = op.getIndexBoundary();
  auto dim = op.getDim();
  auto dstStride = op.getDstStride();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto ptrTy = dyn_cast<MemRefType>(ptr.getType());
  if (!ptrTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for ptr");
  }

  SmallVector<Type> inputTypes({ptrTy, value.getType(), index.getType(),
                                indexBoundary.getType(), dim.getType()});
  inputTypes.append(dstStride.getTypes().begin(), dstStride.getTypes().end());
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());

  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({ptr, value, index, indexBoundary, dim});
  inputVals.append(dstStride.begin(), dstStride.end());
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}),
                                inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult IndirectLoadConverter::matchAndRewrite(
    triton::ascend::IndirectLoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName =
      ConverterUtils::generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto pointer =
        materializeMemoryPointer(op.getSrc(), src, pointerInfos, rewriter);
    if (failed(pointer))
      return rewriter.notifyMatchFailure(op,
                                         "cannot materialize memory address");
    src = *pointer;
  }
  auto offsets = op.getOffsets();
  auto mask = op.getMask();
  auto other = op.getOther();
  auto res = op.getResult();
  auto resTy = res.getType();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }
  SmallVector<Type> inputTypes({srcTy, offsets.getType()});
  if (mask)
    inputTypes.push_back(mask.getType());
  if (other)
    inputTypes.push_back(other.getType());
  auto libFnType = rewriter.getFunctionType(inputTypes, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  auto isVolatileAttr = rewriter.getBoolAttr(op.getIsVolatile());
  funcOp->setAttr("isVolatile", isVolatileAttr);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, offsets});
  if (mask)
    inputVals.push_back(mask);
  if (other)
    inputVals.push_back(other);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}), inputVals);
  callOp->setAttr("isVolatile", isVolatileAttr);
  rewriter.replaceOp(op, callOp);
  return success();
}

LogicalResult StrideLoadConverter::matchAndRewrite(
    triton::ascend::StrideLoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName =
      ConverterUtils::generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto pointer =
        materializeMemoryPointer(op.getSrc(), src, pointerInfos, rewriter);
    if (failed(pointer))
      return rewriter.notifyMatchFailure(op,
                                         "cannot materialize memory address");
    src = *pointer;
  }
  auto offset = adaptor.getOffset();
  auto other = adaptor.getOther();
  auto strides = adaptor.getStride();
  auto numels = adaptor.getNumel();
  auto resTy = op.getResult().getType();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }

  SmallVector<Type> inputTypes({srcTy});
  inputTypes.push_back(offset.getType());
  inputTypes.push_back(other.getType());
  for (Value stride : strides)
    inputTypes.push_back(stride.getType());
  for (Value numel : numels)
    inputTypes.push_back(numel.getType());
  auto libFnType = rewriter.getFunctionType(inputTypes, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  SmallVector<Value> inputVals({src});
  inputVals.push_back(offset);
  inputVals.push_back(other);
  inputVals.append(strides.begin(), strides.end());
  inputVals.append(numels.begin(), numels.end());

  rewriter.setInsertionPoint(op);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}), inputVals);
  rewriter.replaceOp(op, callOp);
  return success();
}

LogicalResult StrideStoreConverter::matchAndRewrite(
    triton::ascend::StrideStoreOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName =
      ConverterUtils::generateUniqueFuncName(moduleOp, funcNameBase);

  auto dst = adaptor.getDst();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto pointer =
        materializeMemoryPointer(op.getDst(), dst, pointerInfos, rewriter);
    if (failed(pointer))
      return rewriter.notifyMatchFailure(op,
                                         "cannot materialize memory address");
    dst = *pointer;
  }
  auto src = adaptor.getSrc();
  auto offset = adaptor.getOffset();
  auto strides = adaptor.getStride();
  auto numels = adaptor.getNumel();

  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!dstTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for dst");
  }

  SmallVector<Type> inputTypes({dstTy, src.getType(), offset.getType()});
  for (Value stride : strides)
    inputTypes.push_back(stride.getType());
  for (Value numel : numels)
    inputTypes.push_back(numel.getType());
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  SmallVector<Value> inputVals({dst, src, offset});
  inputVals.append(strides.begin(), strides.end());
  inputVals.append(numels.begin(), numels.end());

  rewriter.setInsertionPoint(op);
  rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}),
                                inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult IndirectStoreConverter::matchAndRewrite(
    triton::ascend::IndirectStoreOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName =
      ConverterUtils::generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto pointer =
        materializeMemoryPointer(op.getSrc(), src, pointerInfos, rewriter);
    if (failed(pointer))
      return rewriter.notifyMatchFailure(op,
                                         "cannot materialize memory address");
    src = *pointer;
  }
  auto offsets = op.getOffsets();
  auto value = op.getValue();
  auto mask = op.getMask();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }
  SmallVector<Type> inputTypes({srcTy, offsets.getType(), value.getType()});
  if (mask)
    inputTypes.push_back(mask.getType());

  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, offsets, value});
  if (mask)
    inputVals.push_back(mask);
  rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}),
                                inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult IndexSelectSimdConverter::matchAndRewrite(
    triton::ascend::IndexSelectSimdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // Get converted operands
  Value src = adaptor.getSrc();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto pointer =
        materializeMemoryPointer(op.getSrc(), src, pointerInfos, rewriter);
    if (failed(pointer))
      return rewriter.notifyMatchFailure(op,
                                         "cannot materialize memory address");
    src = *pointer;
  }
  Value indexTensor = adaptor.getIndex();
  auto srcShapeVals = adaptor.getSrcShape();
  auto srcOffsetVals = adaptor.getSrcOffset();
  auto readShapeAttr = op.getReadShape();
  int32_t dim = op.getDim();

  // Get result type
  auto resultTensorType = cast<RankedTensorType>(op.getResult().getType());
  static constexpr llvm::StringLiteral wasBoolToInt8AttrName =
      "was_bool_to_int8";
  bool wasBoolToInt8 = op->hasAttr(wasBoolToInt8AttrName);

  auto elemType = resultTensorType.getElementType();
  auto resultShape = resultTensorType.getShape();

  // Convert src (tt.ptr -> memref) to the correct memref shape
  // src is now memref<?xT> after type conversion, need to reinterpret to full
  // shape
  auto srcMemRefType = cast<MemRefType>(src.getType());

  // DenseI32ArrayAttr can be implicitly converted to ArrayRef<int32_t>
  ArrayRef<int32_t> readShape = readShapeAttr;

  // Helper lambda to convert Value to Index type if needed
  auto toIndexValue = [&](Value val) -> Value {
    if (!val.getType().isIndex()) {
      return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                 val);
    }
    return val;
  };

  // Build multi-dimensional memref type and determine static sizes
  // Merge two passes: build fullSrcShape and staticSizes in one loop
  SmallVector<int64_t> fullSrcShape;
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> sizes;

  for (size_t i = 0; i < srcShapeVals.size(); ++i) {
    bool isDynamicDim = (i == static_cast<size_t>(dim) && readShape[i] == -1);
    int64_t staticSize;

    if (isDynamicDim) {
      // Dynamic dimension: readShape[i] == -1 indicates dynamic
      staticSize = ShapedType::kDynamic;
      fullSrcShape.push_back(ShapedType::kDynamic);
      sizes.push_back(toIndexValue(srcShapeVals[i]));
    } else if (auto constOp =
                   srcShapeVals[i].getDefiningOp<arith::ConstantIndexOp>()) {
      // Static dimension: use constant value
      staticSize = constOp.value();
      fullSrcShape.push_back(staticSize);
    } else {
      // Runtime value: must use dynamic
      staticSize = ShapedType::kDynamic;
      fullSrcShape.push_back(ShapedType::kDynamic);
      sizes.push_back(toIndexValue(srcShapeVals[i]));
    }
    staticSizes.push_back(staticSize);
  }
  auto fullSrcMemRefType = MemRefType::get(fullSrcShape, elemType);

  // Build static offsets, sizes, and strides for ReinterpretCastOp
  SmallVector<Value> offsets, strides;
  SmallVector<int64_t> staticOffsets, staticStrides;
  staticOffsets.push_back(0); // offsets are 0

  // Calculate static strides: stride[i] = product of all dimensions after i
  for (size_t i = 0; i < srcShapeVals.size(); ++i) {
    int64_t staticStride = 1;
    bool isDynamic = false;

    // Check if stride needs to be dynamic (any dimension after i is dynamic)
    for (size_t j = i + 1; j < srcShapeVals.size(); ++j) {
      if (staticSizes[j] == ShapedType::kDynamic) {
        isDynamic = true;
        break;
      }
      staticStride *= staticSizes[j];
    }

    if (isDynamic) {
      staticStride = ShapedType::kDynamic;
      // Compute stride dynamically: stride[i] = product of sizes after i
      Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      for (size_t j = i + 1; j < srcShapeVals.size(); ++j) {
        if (staticSizes[j] != ShapedType::kDynamic) {
          strideVal = rewriter.create<arith::MulIOp>(
              loc, strideVal,
              rewriter.create<arith::ConstantIndexOp>(loc, staticSizes[j]));
        } else {
          // Dynamic dimension: use runtime value
          strideVal = rewriter.create<arith::MulIOp>(
              loc, strideVal, toIndexValue(srcShapeVals[j]));
        }
      }
      strides.push_back(strideVal);
    }
    staticStrides.push_back(staticStride);
  }

  auto srcMemRef = rewriter.create<memref::ReinterpretCastOp>(
      loc, fullSrcMemRefType, src, offsets, sizes, strides, staticOffsets,
      staticSizes, staticStrides);

  // Allocate output buffer
  auto resultMemRefType = MemRefType::get(resultShape, elemType);
  auto outputBuffer = rewriter.create<memref::AllocOp>(loc, resultMemRefType);

  // Get indices tensor type for extracting
  auto indicesTensorType = cast<RankedTensorType>(indexTensor.getType());
  int64_t numIndices = indicesTensorType.getShape()[0];

  // Create for loop
  auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto numIndicesVal = rewriter.create<arith::ConstantIndexOp>(loc, numIndices);
  auto stepOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto forOp =
      rewriter.create<scf::ForOp>(loc, zeroIdx, numIndicesVal, stepOne);

  // Mark as parallel loop
  forOp->setAttr("hivm.parallel_loop", rewriter.getUnitAttr());

  // Build loop body
  Block *loopBody = forOp.getBody();
  auto savedInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(loopBody);

  // Remove the terminator temporarily
  Operation *terminator = &loopBody->back();
  rewriter.setInsertionPoint(terminator);

  Value iv = forOp.getInductionVar();

  // Extract index from indices tensor
  Value selectedIdx =
      rewriter.create<tensor::ExtractOp>(loc, indexTensor, ValueRange{iv});
  Value selectedIdxAsIndex = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), selectedIdx);

  // Build source subview offsets/sizes/strides
  SmallVector<OpFoldResult> srcSubviewOffsets, srcSubviewSizes,
      srcSubviewStrides;

  for (size_t i = 0; i < srcOffsetVals.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      // Use the selected index for this dimension
      srcSubviewOffsets.push_back(selectedIdxAsIndex);
      srcSubviewSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      // Use provided offset and read size for other dimensions
      Value offsetVal = srcOffsetVals[i];
      if (!offsetVal.getType().isIndex()) {
        offsetVal = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), offsetVal);
      }
      srcSubviewOffsets.push_back(offsetVal);
      srcSubviewSizes.push_back(rewriter.getIndexAttr(readShape[i]));
    }
    srcSubviewStrides.push_back(rewriter.getIndexAttr(1));
  }

  auto srcSubview = rewriter.create<memref::SubViewOp>(
      loc, srcMemRef, srcSubviewOffsets, srcSubviewSizes, srcSubviewStrides);

  // Build destination subview
  SmallVector<OpFoldResult> dstSubviewOffsets, dstSubviewSizes,
      dstSubviewStrides;
  for (size_t i = 0; i < resultShape.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      dstSubviewOffsets.push_back(iv);
      dstSubviewSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      dstSubviewOffsets.push_back(rewriter.getIndexAttr(0));
      dstSubviewSizes.push_back(rewriter.getIndexAttr(readShape[i]));
    }
    dstSubviewStrides.push_back(rewriter.getIndexAttr(1));
  }

  auto dstSubview = rewriter.create<memref::SubViewOp>(
      loc, outputBuffer, dstSubviewOffsets, dstSubviewSizes, dstSubviewStrides);

  // Check if index_select is on the trailing axis (last dimension)
  if (static_cast<size_t>(dim) == fullSrcShape.size() - 1) {
    // For index_select on the trailing axis, mark as discrete memory access
    // This degrades to scalar read/write handling to avoid alignment issues
    auto copyOp = rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    if (wasBoolToInt8)
      copyOp->setAttr(wasBoolToInt8AttrName, rewriter.getBoolAttr(true));
    copyOp->setAttr(ConverterUtils::discreteAttrName, rewriter.getUnitAttr());
  } else {
    // For index_select on non-trailing axes, add stride alignment annotation
    // This tells the backend to handle address alignment for DMA operations
    auto dstMarkOp = rewriter.create<annotation::MarkOp>(loc, dstSubview);
    dstMarkOp->setAttr(
        "hfusion.stride_align_dims",
        rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(dim)}));
    dstMarkOp->setAttr("hfusion.stride_align_value_in_byte",
                       rewriter.getDenseI32ArrayAttr({32}));

    // Copy from source to destination
    auto copyOp = rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    if (wasBoolToInt8)
      copyOp->setAttr(wasBoolToInt8AttrName, rewriter.getBoolAttr(true));
  }

  // Restore insertion point
  rewriter.restoreInsertionPoint(savedInsertionPoint);

  // Convert memref to tensor
  auto resultTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, resultTensorType, outputBuffer, true, true);
  if (wasBoolToInt8)
    resultTensor->setAttr(wasBoolToInt8AttrName, rewriter.getBoolAttr(true));

  // Mark as index_select_simd
  resultTensor->setAttr("index_select_simd", rewriter.getUnitAttr());

  // Replace the original op
  rewriter.replaceOp(op, resultTensor);

  return success();
}

} // namespace TritonAscendOpConverters
