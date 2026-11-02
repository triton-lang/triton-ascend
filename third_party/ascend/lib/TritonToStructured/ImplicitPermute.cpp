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

#include "TritonToStructured/ImplicitPermute.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-structured-implicit-permute"

namespace ImplicitPermute {
using namespace mlir;
using namespace triton;
using MemOpTransformer = MemOpConverter::MemOpTransformer;

LogicalResult LoadConverter::matchAndRewrite(triton::LoadOp op,
                                             PatternRewriter &rewriter) const {
  // no need to analyze and rewrite
  if (compileOn91095Flag && !existDotFlag) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------------------------------------\n";
      llvm::dbgs() << "compileOn91095Flag :" << compileOn91095Flag << "\n";
      llvm::dbgs() << "!existDotFlag :" << !existDotFlag << "\n";
      llvm::dbgs() << "no need to analyze and rewrite Load" << "\n";
      llvm::dbgs() << "----------------------------------------------\n";
    });
    return success();
  }

  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldOther = op.getOther();

  MemOpTransformer tf(MemOpTransformer::MemType::load);

  Value newPtr = nullptr;
  if (oldPtr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    newPtr = tf.createNewTensorPtr(oldPtr, loc, rewriter);
  } else if (oldPtr.getDefiningOp<triton::AdvanceOp>()) {
    newPtr = tf.createNewAdvancePtr(oldPtr, loc, rewriter);
  } else if (oldPtr.getDefiningOp<triton::AddPtrOp>()) {
    newPtr = tf.createNewAddPtr(oldPtr, loc, rewriter);
  } else {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "PtrAnalysis: only MakeTensorPtrOp, "
                                 "AdvanceOp, and AddPtrOp are supported.";
    return success();
  }
  if (!tf.ptrState.isPermuted) {
    // no need to rewrite
    return success();
  }

  auto newMask = tf.createNewMask(oldMask, loc, rewriter);
  auto newOther = tf.createNewOther(oldOther, loc, rewriter);

  if (!newPtr) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "PtrAnalysis: failed to analyze load pointer.";
    return failure();
  }

  if (oldMask && !newMask) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "MaskAnalysis: failed to analyze load mask.";
    return failure();
  }

  auto newBoundaryCheck = tf.getBoundaryCheck(op.getBoundaryCheck());

  auto loadOp = rewriter.create<triton::LoadOp>(
      loc, newPtr, newMask, newOther, newBoundaryCheck, op.getPadding(),
      op.getCache(), op.getEvict(), op.getIsVolatile());
  loadOp->setAttr(ImplicitPermuteHandledTAG,
                  UnitAttr::get(rewriter.getContext()));

  auto permuteResult =
      tf.materializeImplicitPermute(loadOp.getResult(), loc, rewriter);

  rewriter.replaceOp(op, permuteResult);
  return success();
}

LogicalResult StoreConverter::matchAndRewrite(triton::StoreOp op,
                                              PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldValue = op.getValue();

  MemOpTransformer tf(MemOpTransformer::MemType::store);
  Value newPtr = nullptr;
  if (oldPtr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    newPtr = tf.createNewTensorPtr(oldPtr, loc, rewriter);
  } else if (oldPtr.getDefiningOp<triton::AdvanceOp>()) {
    newPtr = tf.createNewAdvancePtr(oldPtr, loc, rewriter);
  } else if (oldPtr.getDefiningOp<triton::AddPtrOp>()) {
    newPtr = tf.createNewAddPtr(oldPtr, loc, rewriter);
  } else {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "PtrAnalysis: only MakeTensorPtrOp, "
                                 "AdvanceOp, and AddPtrOp are supported.";
    return success();
  }
  if (!tf.ptrState.isPermuted) {
    // no need to rewrite
    return success();
  }
  auto newMask = tf.createNewMask(oldMask, loc, rewriter);

  if (!newPtr) {
    InFlightDiagnostic diag =
        emitWarning(loc) << "PtrAnalysis: failed to analyze store pointer.";
    return failure();
  }

  if (oldMask && !newMask) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "MaskAnalysis: failed to analyze store mask.";
    return failure();
  }

  auto permuteResult = tf.materializeImplicitPermute(oldValue, loc, rewriter);

  auto newBoundaryCheck = tf.getBoundaryCheck(op.getBoundaryCheck());

  auto storeOp = rewriter.create<triton::StoreOp>(loc, newPtr, permuteResult,
                                                  newMask, newBoundaryCheck,
                                                  op.getCache(), op.getEvict());
  storeOp->setAttr(ImplicitPermuteHandledTAG,
                   UnitAttr::get(rewriter.getContext()));

  rewriter.eraseOp(op);
  return success();
}

LogicalResult
AtomicRMWConverter::matchAndRewrite(triton::AtomicRMWOp op,
                                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldVal = op.getVal();

  MemOpTransformer tf(MemOpTransformer::MemType::store);

  Value newPtr = nullptr;
  if (oldPtr.getDefiningOp<triton::AddPtrOp>()) {
    newPtr = tf.createNewAddPtr(oldPtr, loc, rewriter);
  } else {
    emitWarning(loc) << "PtrAnalysis: AtomicRMW only support AddPtrOp.";
    return success();
  }

  Value newMask = tf.createNewMask(oldMask, loc, rewriter);

  if (!tf.ptrState.isPermuted) {
    return success();
  }
  if (!newPtr) {
    emitWarning(loc) << "PtrAnalysis: failed to analyze atomic_rmw pointer.";
    return failure();
  }
  if (oldMask && !newMask) {
    emitWarning(loc) << "MaskAnalysis: failed to analyze atomic_rmw mask.";
    return failure();
  }

  Value newVal = tf.materializeImplicitPermute(oldVal, loc, rewriter);

  Type newAtomicResTy = newVal.getType();

  auto newAtomic = rewriter.create<triton::AtomicRMWOp>(
      loc, newAtomicResTy, op.getAtomicRmwOp(), newPtr, newVal, newMask,
      op.getSem(), op.getScope());
  newAtomic->setAttr(ImplicitPermuteHandledTAG,
                     UnitAttr::get(rewriter.getContext()));

  // The returned old value should be in OLD layout for users => permute back
  // (load-side).
  MemOpTransformer tfLoad(MemOpTransformer::MemType::load);
  tfLoad.ptrState = tf.ptrState;
  Value permutedRes =
      tfLoad.materializeImplicitPermute(newAtomic.getResult(), loc, rewriter);

  rewriter.replaceOp(op, permutedRes);
  return success();
}

LogicalResult
AtomicCASConverter::matchAndRewrite(triton::AtomicCASOp op,
                                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldCmp = op.getCmp();
  auto oldVal = op.getVal();

  MemOpTransformer tf(MemOpTransformer::MemType::store);

  Value newPtr = nullptr;
  if (oldPtr.getDefiningOp<triton::AddPtrOp>()) {
    newPtr = tf.createNewAddPtr(oldPtr, loc, rewriter);
  } else {
    emitWarning(loc) << "PtrAnalysis: AtomicCAS only support AddPtrOp.";
    return success();
  }

  if (!tf.ptrState.isPermuted) {
    return success();
  }
  if (!newPtr) {
    emitWarning(loc) << "PtrAnalysis: failed to analyze atomic_cas pointer.";
    return failure();
  }

  Value newCmp = tf.materializeImplicitPermute(oldCmp, loc, rewriter);
  Value newVal = tf.materializeImplicitPermute(oldVal, loc, rewriter);

  // CAS result (old value) must have same shape as cmp/val operands.
  Type newAtomicResTy = newVal.getType();

  auto newAtomic = rewriter.create<triton::AtomicCASOp>(
      loc, newAtomicResTy, newPtr, newCmp, newVal, op.getSem(), op.getScope());
  newAtomic->setAttr(ImplicitPermuteHandledTAG,
                     UnitAttr::get(rewriter.getContext()));

  MemOpTransformer tfLoad(MemOpTransformer::MemType::load);
  tfLoad.ptrState = tf.ptrState;
  Value permutedRes =
      tfLoad.materializeImplicitPermute(newAtomic.getResult(), loc, rewriter);

  rewriter.replaceOp(op, permutedRes);
  return success();
}

} // namespace ImplicitPermute
