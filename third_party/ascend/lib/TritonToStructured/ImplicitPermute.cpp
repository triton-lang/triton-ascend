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
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-structured-implicit-permute"

namespace ImplicitPermute {
using namespace mlir;
using namespace triton;
using MemOpTransformer = MemOpConverter::MemOpTransformer;

// Only remap other when pointer rebuilding is a pure permutation:
// split or removed axes still need the existing mask analysis.
static bool canPermuteLoadOperands(const TritonToStructured::PtrState &state,
                                   Value oldPtr, Value newPtr) {
  auto oldType = dyn_cast<RankedTensorType>(oldPtr.getType());
  auto newType = dyn_cast<RankedTensorType>(newPtr.getType());
  if (state.dynamicOffsets.empty() || state.isBlockPtr() || !oldType ||
      !newType || oldType.getRank() != newType.getRank() ||
      state.sizes.size() != oldType.getRank() ||
      state.stateInfo.size() != oldType.getRank() ||
      state.permuteIds.size() != oldType.getRank())
    return false;

  for (auto [i, id] : llvm::enumerate(state.permuteIds)) {
    if (state.stateInfo[i].dimIndex != i ||
        newType.getDimSize(i) != oldType.getDimSize(id))
      return false;
  }
  return true;
}

static Value remapSplatOther(Value other, RankedTensorType targetType,
                             Location loc, PatternRewriter &rewriter) {
  if (!other)
    return nullptr;
  if (auto splat = other.getDefiningOp<triton::SplatOp>())
    return rewriter.create<triton::SplatOp>(loc, targetType, splat.getSrc());
  if (auto constant = other.getDefiningOp<arith::ConstantOp>()) {
    auto value = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (value && value.isSplat())
      return rewriter.create<arith::ConstantOp>(loc, value.reshape(targetType));
  }
  // FIXME: Support non-splat other without adding a data transpose. Reshaping
  // an arbitrary tensor would associate its values with the wrong mask lanes.
  return nullptr;
}

LogicalResult LoadConverter::matchAndRewrite(triton::LoadOp op,
                                             PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldOther = op.getOther();

  // enableDataTensorOffset: only the ImplicitPermute path accepts a tt.load
  // result (integer tensor) as an opaque data-dependent offset.
  MemOpTransformer tf(MemOpTransformer::MemType::load, false, true);

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
  if (!newPtr)
    return rewriter.notifyMatchFailure(op, "failed to analyze load pointer");

  if (!tf.ptrState.isPermuted) {
    // no need to rewrite
    return success();
  }

  // skip only loads without data-dependent offsets.
  if (compileOn91095Flag && !existDotFlag &&
      tf.ptrState.dynamicOffsets.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------------------------------------\n";
      llvm::dbgs() << "compileOn91095Flag :" << compileOn91095Flag << "\n";
      llvm::dbgs() << "!existDotFlag :" << !existDotFlag << "\n";
      llvm::dbgs() << "no need to analyze and rewrite Load" << "\n";
      llvm::dbgs() << "----------------------------------------------\n";
    });
    return success();
  }

  auto resultType = triton::getPointeeType(newPtr.getType());
  auto newMask = tf.createNewMask(oldMask, loc, rewriter);
  if (oldMask && !newMask)
    return rewriter.notifyMatchFailure(
        op, "cannot remap load mask without transpose");

  // Match the load's operand types, including shape, before creating it.
  if (newMask && newMask.getType() != triton::getI1SameShape(resultType))
    return rewriter.notifyMatchFailure(op, "mask type does not match pointer");

  bool permuteOperands = canPermuteLoadOperands(tf.ptrState, oldPtr, newPtr);
  auto newOther =
      permuteOperands
          ? remapSplatOther(oldOther, cast<RankedTensorType>(resultType), loc,
                            rewriter)
          : tf.createNewOther(oldOther, loc, rewriter);
  if (oldOther && !newOther)
    return rewriter.notifyMatchFailure(op, "failed to convert load other");
  if (newOther && newOther.getType() != resultType)
    return rewriter.notifyMatchFailure(op, "other type does not match pointer");

  auto newBoundaryCheck = tf.getBoundaryCheck(op.getBoundaryCheck());

  auto loadOp = rewriter.create<triton::LoadOp>(
      loc, newPtr, newMask, newOther, newBoundaryCheck, op.getPadding(),
      op.getCache(), op.getEvict(), op.getIsVolatile());
  loadOp->setAttr(ImplicitPermuteHandledTAG,
                  UnitAttr::get(rewriter.getContext()));

  auto broadCastResult =
      tf.materializeImplicitBroadcast(loadOp.getResult(), loc, rewriter);

  auto permuteResult =
      tf.materializeImplicitPermute(broadCastResult, loc, rewriter);

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
