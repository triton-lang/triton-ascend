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

#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "TritonMemoryAccess/PtrAnalysis.h"
#include "TritonMemoryAccess/Utils.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToStructured/CannonicalizerConverter.h"
#include "TritonToUnstructure/IndirectAtomicUtils.h"
#include "Utils/PtrAnalysisUtils.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

#define DEBUG_TYPE "triton-unstructure-converter"

using namespace mlir;
using namespace triton;

#include "llvm/Support/Debug.h"

static triton::ascend::CompileMode unstructureCompileMode =
    triton::ascend::CompileMode::Simd;

namespace {

static bool isTensorOfPointers(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

constexpr int64_t kBitsPerByte = 8;

static triton::PointerType getScalarPointerType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    type = tensorType.getElementType();
  return dyn_cast<triton::PointerType>(type);
}

// Keep this deliberately narrow. These are the byte-buffer views used by the
// paged-attention and KV-cache kernels covered by this canonicalization.
static std::optional<int64_t>
getSupportedPointerBitcastScale(triton::BitcastOp op) {
  auto srcType = getScalarPointerType(op.getSrc().getType());
  auto dstType = getScalarPointerType(op.getType());
  if (!srcType || !dstType ||
      srcType.getAddressSpace() != dstType.getAddressSpace() ||
      !srcType.getPointeeType().isInteger(8))
    return std::nullopt;

  Type dstPointeeType = dstType.getPointeeType();
  if (dstPointeeType.isF16() || dstPointeeType.isBF16())
    return 2;
  if (dstPointeeType.isF32() || dstPointeeType.isInteger(32))
    return 4;
  return std::nullopt;
}

static FailureOr<Value> scaleTensorPointerOffset(Value offset, int64_t divisor,
                                                 IRRewriter &rewriter) {
  auto offsetType = dyn_cast<RankedTensorType>(offset.getType());
  if (!offsetType || !offsetType.hasStaticShape())
    return failure();
  auto elementType = dyn_cast<IntegerType>(offsetType.getElementType());
  if (!elementType ||
      (elementType.getWidth() != 32 && elementType.getWidth() != 64))
    return failure();
  auto divisorAttr = rewriter.getIntegerAttr(elementType, divisor);
  Value divisorValue = rewriter.create<arith::ConstantOp>(
      offset.getLoc(), DenseElementsAttr::get(offsetType, divisorAttr));
  return rewriter.create<arith::DivSIOp>(offset.getLoc(), offset, divisorValue)
      .getResult();
}

static bool isPublicKernelArgument(Value value) {
  auto blockArgument = dyn_cast<BlockArgument>(value);
  if (!blockArgument)
    return false;
  auto funcOp =
      dyn_cast<triton::FuncOp>(blockArgument.getOwner()->getParentOp());
  return funcOp && funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

static FailureOr<Value> materializeScalarByteAddress(Value pointer,
                                                     Location loc,
                                                     IRRewriter &rewriter) {
  SmallVector<Value> byteOffsets;
  Value root = pointer;
  while (auto addPtrOp = root.getDefiningOp<triton::AddPtrOp>()) {
    auto offsetType = dyn_cast<IntegerType>(addPtrOp.getOffset().getType());
    if (isa<RankedTensorType>(addPtrOp.getType()) || !offsetType ||
        (offsetType.getWidth() != 32 && offsetType.getWidth() != 64))
      return failure();
    byteOffsets.push_back(addPtrOp.getOffset());
    root = addPtrOp.getPtr();
  }
  if (!isPublicKernelArgument(root))
    return failure();

  Value address =
      rewriter.create<triton::PtrToIntOp>(loc, rewriter.getI64Type(), root);
  for (Value offset : llvm::reverse(byteOffsets)) {
    auto offsetType = cast<IntegerType>(offset.getType());
    if (offsetType.getWidth() < 64)
      offset =
          rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offset);
    address = rewriter.create<arith::AddIOp>(loc, address, offset);
  }
  return address;
}

static bool hasOnlySupportedPointerUses(Value pointer) {
  if (pointer.use_empty())
    return false;
  for (Operation *user : pointer.getUsers()) {
    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(user)) {
      if (addPtrOp.getPtr() != pointer ||
          !hasOnlySupportedPointerUses(addPtrOp.getResult()))
        return false;
      continue;
    }
    if (auto splatOp = dyn_cast<triton::SplatOp>(user)) {
      if (splatOp.getSrc() != pointer ||
          !hasOnlySupportedPointerUses(splatOp.getResult()))
        return false;
      continue;
    }
    if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(user)) {
      if (expandDimsOp.getSrc() != pointer ||
          !hasOnlySupportedPointerUses(expandDimsOp.getResult()))
        return false;
      continue;
    }
    if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(user)) {
      if (broadcastOp.getSrc() != pointer ||
          !hasOnlySupportedPointerUses(broadcastOp.getResult()))
        return false;
      continue;
    }
    if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
      if (loadOp.getPtr() != pointer)
        return false;
      continue;
    }
    if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
      if (storeOp.getPtr() != pointer)
        return false;
      continue;
    }
    return false;
  }
  return true;
}

static bool canRewriteSupportedPointerBitcast(triton::BitcastOp op) {
  if (!getSupportedPointerBitcastScale(op) ||
      !hasOnlySupportedPointerUses(op.getResult()))
    return false;

  Value current = op.getSrc();
  while (Operation *producer = current.getDefiningOp()) {
    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(producer)) {
      Type offsetType = addPtrOp.getOffset().getType();
      if (isa<RankedTensorType>(current.getType())) {
        auto pointerType = cast<RankedTensorType>(current.getType());
        auto tensorType = dyn_cast<RankedTensorType>(offsetType);
        if (!pointerType.hasStaticShape() || !tensorType ||
            !tensorType.hasStaticShape() ||
            pointerType.getShape() != tensorType.getShape())
          return false;
        auto elementType = dyn_cast<IntegerType>(tensorType.getElementType());
        if (!elementType ||
            (elementType.getWidth() != 32 && elementType.getWidth() != 64))
          return false;
      } else if (auto intType = dyn_cast<IntegerType>(offsetType);
                 !intType ||
                 (intType.getWidth() != 32 && intType.getWidth() != 64)) {
        return false;
      }
      current = addPtrOp.getPtr();
      continue;
    }
    if (auto splatOp = dyn_cast<triton::SplatOp>(producer)) {
      if (isa<RankedTensorType>(splatOp.getSrc().getType()))
        return false;
      current = splatOp.getSrc();
      continue;
    }
    return false;
  }
  return !isa<RankedTensorType>(current.getType()) &&
         isPublicKernelArgument(current);
}

// Rewrite only the supported byte-buffer views into ordinary target-element
// addptr operations. All later pointer and control-flow handling remains on the
// existing main-dev paths.
static LogicalResult
rewriteSupportedPointerBitcast(triton::BitcastOp op, IRRewriter &rewriter,
                               triton::BitcastOp &nextBitcast) {
  auto divisor = getSupportedPointerBitcastScale(op);
  if (!divisor)
    return failure();

  Value src = op.getSrc();
  rewriter.setInsertionPoint(op);
  if (!isa<RankedTensorType>(src.getType())) {
    auto address = materializeScalarByteAddress(src, op.getLoc(), rewriter);
    if (failed(address))
      return failure();
    auto targetPtr = rewriter.create<triton::IntToPtrOp>(
        op.getLoc(), op.getType(), *address);
    rewriter.replaceOp(op, targetPtr.getResult());
    nextBitcast = nullptr;
    return success();
  }

  if (src.getDefiningOp<triton::AddPtrOp>()) {
    Value base = src;
    Value byteOffset;
    while (auto currentAddPtr = base.getDefiningOp<triton::AddPtrOp>()) {
      Value currentOffset = currentAddPtr.getOffset();
      auto offsetType = cast<RankedTensorType>(currentOffset.getType());
      auto elementType = cast<IntegerType>(offsetType.getElementType());
      if (elementType.getWidth() < 64) {
        auto i64OffsetType =
            RankedTensorType::get(offsetType.getShape(), rewriter.getI64Type());
        currentOffset = rewriter.create<arith::ExtSIOp>(
            op.getLoc(), i64OffsetType, currentOffset);
      }
      if (byteOffset)
        byteOffset = rewriter.create<arith::AddIOp>(op.getLoc(), byteOffset,
                                                    currentOffset);
      else
        byteOffset = currentOffset;
      base = currentAddPtr.getPtr();
    }
    auto scaledOffset =
        scaleTensorPointerOffset(byteOffset, *divisor, rewriter);
    if (failed(scaledOffset))
      return failure();
    nextBitcast =
        rewriter.create<triton::BitcastOp>(op.getLoc(), op.getType(), base);
    auto newAddPtr = rewriter.create<triton::AddPtrOp>(
        op.getLoc(), op.getType(), nextBitcast, *scaledOffset);
    rewriter.replaceOp(op, newAddPtr.getResult());
    return success();
  }

  if (auto splatOp = src.getDefiningOp<triton::SplatOp>()) {
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    nextBitcast = rewriter.create<triton::BitcastOp>(
        op.getLoc(), resultType.getElementType(), splatOp.getSrc());
    auto newSplat = rewriter.create<triton::SplatOp>(op.getLoc(), resultType,
                                                     nextBitcast.getResult());
    rewriter.replaceOp(op, newSplat.getResult());
    return success();
  }

  return failure();
}

static void canonicalizeSupportedPointerBitcasts(ModuleOp moduleOp) {
  SmallVector<triton::BitcastOp> bitcasts;
  moduleOp.walk([&](triton::BitcastOp op) {
    if (canRewriteSupportedPointerBitcast(op))
      bitcasts.push_back(op);
  });

  IRRewriter rewriter(moduleOp.getContext());
  for (triton::BitcastOp bitcast : bitcasts) {
    triton::BitcastOp current = bitcast;
    while (current) {
      triton::BitcastOp next;
      if (failed(rewriteSupportedPointerBitcast(current, rewriter, next)))
        break;
      current = next;
    }
  }
}

static RankedTensorType resolvePtrTensorType(Value ptr) {
  auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
  if (auto ptrPtrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    if (auto ptrTensorType =
            dyn_cast_or_null<RankedTensorType>(ptrPtrType.getPointeeType())) {
      ptrType = ptrTensorType;
    }
  }
  return ptrType;
}

static Type getResultElementType(RankedTensorType ptrType) {
  auto resultElementType = ptrType.getElementType();
  if (auto pointerType =
          dyn_cast<triton::PointerType>(ptrType.getElementType())) {
    resultElementType = pointerType.getPointeeType();
  }
  return resultElementType;
}

static int64_t getTypeSizeInByte(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() / kBitsPerByte;
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return floatType.getWidth() / kBitsPerByte;
  }
  llvm_unreachable("Unhandled element type of tensor");
}

template <typename MemAccOpTy>
void normalizeDiscreteMaskAccessForFallback(MemAccOpTy &op,
                                            SmallVector<bool> &structuredAxes,
                                            PatternRewriter &rewriter) {
  if (!op->hasAttr(ConverterUtils::discreteMaskAttrName)) {
    return;
  }

  if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
    auto selectOp = op.getValue().template getDefiningOp<arith::SelectOp>();
    op = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, op.getPtr(), selectOp.getTrueValue(), selectOp.getCondition(),
        op.getCache(), op.getEvict());
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
    if (auto selectOp = op.getVal().template getDefiningOp<arith::SelectOp>()) {
      op = rewriter.replaceOpWithNewOp<triton::AtomicRMWOp>(
          op, op.getType(), op.getAtomicRmwOp(), op.getPtr(),
          selectOp.getTrueValue(), selectOp.getCondition(), op.getSem(),
          op.getScope());
    }
  }

  rewriter.setInsertionPoint(op);
  llvm::fill(structuredAxes, false);
}

// ======================== 950 template-SIMT indirect fast-path lowering
// ========================
// Load/store uses the established indirect-template ABI when its rank and
// pointer requirements are met. Atomic operations use the same gate with their
// indirect custom-op lowering. Other cases fall back to scalar-loop lowering.
// ======================================================================================
static bool canUseIndirectFastPath(Value srcPtr, Value ptrOffset) {
  if (!srcPtr || !ptrOffset)
    return false;
  auto ptrTy = dyn_cast<triton::PointerType>(srcPtr.getType());
  if (!ptrTy || isa<ShapedType>(ptrTy.getPointeeType()))
    return false;
  return isa<RankedTensorType>(ptrOffset.getType());
}

template <typename MemAccOpTy>
LogicalResult tryRewriteIndirectFastPath(MemAccOpTy op, Location loc,
                                         Value srcPtr, Value ptrOffset,
                                         ArrayRef<int64_t> resultShape,
                                         PatternRewriter &rewriter) {
  // Indirect backend operations accept one scalar base plus lane offsets. An
  // opaque tensor base can contain a different pointer in every lane, so it
  // must be handled by the scalar-loop fallback below.
  if (!isa<triton::PointerType>(srcPtr.getType()))
    return failure();

  bool rankWithinIndirectLoadStoreFastPathLimit = resultShape.size() <= 5;

  if (!canUseIndirectFastPath(srcPtr, ptrOffset)) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "Skip SIMT indirect fast path: src must be scalar elem ptr and "
             "offset must be an int tensor (reject block_ptr)\n";
    });
    return failure();
  }

  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    if (!rankWithinIndirectLoadStoreFastPathLimit) {
      return failure();
    }

    assert(isa<triton::PointerType>(srcPtr.getType()) &&
           "src must be ptr type");
    Value mask = op.getMask();
    Value other = op.getOther();
    auto resultType = op.getType();
    auto indirect = rewriter.create<triton::ascend::IndirectLoadOp>(
        loc, resultType, srcPtr, ptrOffset, mask, other,
        op.getIsVolatile() ||
            ConverterUtils::requiresVolatileIndirectLoad(op.getPtr(), op));
    rewriter.replaceOp(op, indirect.getResult());
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Rewriting tt.load to tt.indirect_load\n";
      os << indirect << "\n";
    });
    return success();
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
    if (!rankWithinIndirectLoadStoreFastPathLimit) {
      return failure();
    }

    assert(isa<triton::PointerType>(srcPtr.getType()) &&
           "src must be ptr type");
    Value value = op.getValue();
    Value mask = op.getMask();

    // For bool store, unwrap ptr<i1> -> ptr<i8> bitcast before creating
    // indirect_store. Keep ptr<i1> so TypeConverter can map it to memref<?xi8>.
    if (auto bitcastOp = srcPtr.getDefiningOp<triton::BitcastOp>()) {
      auto srcPtrTy =
          dyn_cast<triton::PointerType>(bitcastOp.getSrc().getType());
      auto dstPtrTy = dyn_cast<triton::PointerType>(bitcastOp.getType());

      if (srcPtrTy && dstPtrTy && srcPtrTy.getPointeeType().isInteger(1) &&
          dstPtrTy.getPointeeType().isInteger(8)) {
        srcPtr = bitcastOp.getSrc();
      }
    }
    auto indirect = rewriter.create<triton::ascend::IndirectStoreOp>(
        loc, srcPtr, ptrOffset, value, mask);
    rewriter.eraseOp(op);
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Rewriting tt.store to tt.indirect_store\n";
      os << indirect << "\n";
    });
    return success();
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
    assert(isa<triton::PointerType>(srcPtr.getType()) &&
           "src must be ptr type");
    if (!IndirectAtomicUtils::canUseIndirectAtomicFastPath(op, ptrOffset)) {
      return failure();
    }

    auto customResult =
        IndirectAtomicUtils::tryConvertAtomicRmwToIndirectCustom(
            op, srcPtr, ptrOffset, rewriter);
    if (failed(customResult)) {
      return failure();
    }

    rewriter.replaceOp(op, *customResult);
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Rewriting tt.atomic_rmw to hivm.hir.custom indirect atomic\n";
    });
    return success();
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicCASOp>) {
    assert(isa<triton::PointerType>(srcPtr.getType()) &&
           "src must be ptr type");
    if (!IndirectAtomicUtils::canUseIndirectAtomicFastPath(op, ptrOffset)) {
      return failure();
    }

    auto customResult =
        IndirectAtomicUtils::tryConvertAtomicCasToIndirectCustom(
            op, srcPtr, ptrOffset, rewriter);
    if (failed(customResult)) {
      return failure();
    }

    rewriter.replaceOp(op, *customResult);
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Rewriting tt.atomic_cas to hivm.hir.custom indirect atomic\n";
    });
    return success();
  }
}

} // namespace

template <typename MemAccOpTy>
bool UnstructuredMemAccessConverter<MemAccOpTy>::checkUnstructureAnnotated(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  return llvm::any_of(op->getUsers(), [&rewriter](Operation *user) {
    auto annotationOp = dyn_cast<annotation::MarkOp>(user);
    if (annotationOp && annotationOp->hasAttr("mayDiscretememaccess")) {
      rewriter.eraseOp(annotationOp);
      return true;
    }
    return false;
  });
}

template <>
bool UnstructuredMemAccessConverter<triton::StoreOp>::checkUnstructureAnnotated(
    triton::StoreOp op, PatternRewriter &rewriter) const {
  return llvm::any_of(op.getValue().getUsers(), [&rewriter](Operation *user) {
    auto annotationOp = dyn_cast<annotation::MarkOp>(user);
    if (annotationOp && annotationOp->hasAttr("mayDiscretememaccess")) {
      rewriter.eraseOp(annotationOp);
      return true;
    }
    return false;
  });
}

// A null result denotes a predicate already enforced by the generated loops.
// Keep this limited to conjunctions and broadcast upper bounds. In particular,
// do not remove scalar gates or predicates with nonzero/unknown loop starts.
static Value getResidualLoadMask(Value mask, ArrayRef<bool> boundedLoopAxes,
                                 PatternRewriter &rewriter) {
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    Value lhs = getResidualLoadMask(andOp.getLhs(), boundedLoopAxes, rewriter);
    Value rhs = getResidualLoadMask(andOp.getRhs(), boundedLoopAxes, rewriter);
    if (!lhs)
      return rhs;
    if (!rhs)
      return lhs;
    if (lhs == andOp.getLhs() && rhs == andOp.getRhs())
      return mask;
    return rewriter.create<arith::AndIOp>(andOp.getLoc(), lhs, rhs);
  }

  auto broadcast = mask.getDefiningOp<triton::BroadcastOp>();
  if (!broadcast)
    return mask;
  auto sourceType = cast<RankedTensorType>(broadcast.getSrc().getType());
  if (sourceType.getShape().size() != boundedLoopAxes.size())
    return mask;
  bool hasLoopAxis = false;
  for (auto [axis, size] : llvm::enumerate(sourceType.getShape())) {
    if (size == 1)
      continue;
    if (!boundedLoopAxes[axis])
      return mask;
    hasLoopAxis = true;
  }
  if (!hasLoopAxis)
    return mask;

  Value predicate = broadcast.getSrc();
  while (auto expand = predicate.getDefiningOp<triton::ExpandDimsOp>())
    predicate = expand.getSrc();
  auto cmp = predicate.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return mask;
  return {};
}

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> iterIdx) const {
  if (!value || !isa<RankedTensorType>(value.getType()))
    return value;
  SmallVector<Value> indices;
  for (auto idxOfr : iterIdx) {
    auto idx = getValueOrCreateConstantIndexOp(rewriter, loc, idxOfr);
    indices.push_back(idx);
  }
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, value, indices);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides) const {
  if (!value)
    return value;
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracting\n";
    os << value << "\n";
  });
  auto extractedOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, value, offsets, sizes, strides);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <>
template <typename... Args>
triton::LoadOp UnstructuredMemAccessConverter<triton::LoadOp>::createMemAccOp(
    triton::LoadOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  Value mask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  Value other = createExtractOp(loc, op.getOther(), rewriter,
                                std::forward<Args>(args)...);
  return rewriter.create<triton::LoadOp>(loc, ptrToAccess, mask, other,
                                         op.getCache(), op.getEvict(),
                                         op.getIsVolatile());
}

template <>
template <typename... Args>
triton::AtomicRMWOp
UnstructuredMemAccessConverter<triton::AtomicRMWOp>::createMemAccOp(
    triton::AtomicRMWOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedValue =
      createExtractOp(loc, op.getVal(), rewriter, std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  Type targetType = ptrToAccess.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    targetType =
        RankedTensorType::get(tensorType.getShape(), ptrType.getPointeeType());
  } else {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
    targetType =
        RankedTensorType::get(scalarLikeShape, resultType.getElementType());
    ptrToAccess = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
        ptrToAccess);
    extractedValue = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
        extractedValue);
    if (extractedMask) {
      extractedMask = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, extractedMask.getType()),
          extractedMask);
    }
  }
  return rewriter.create<triton::AtomicRMWOp>(
      loc, targetType, op.getAtomicRmwOpAttr(), ptrToAccess, extractedValue,
      extractedMask, op.getSemAttr(), op.getScopeAttr());
}

template <>
template <typename... Args>
triton::AtomicCASOp
UnstructuredMemAccessConverter<triton::AtomicCASOp>::createMemAccOp(
    triton::AtomicCASOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedCmp =
      createExtractOp(loc, op.getCmp(), rewriter, std::forward<Args>(args)...);
  auto extractedValue =
      createExtractOp(loc, op.getVal(), rewriter, std::forward<Args>(args)...);
  Type targetType = ptrToAccess.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    targetType =
        RankedTensorType::get(tensorType.getShape(), ptrType.getPointeeType());
  } else {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
    targetType =
        RankedTensorType::get(scalarLikeShape, resultType.getElementType());
    ptrToAccess = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
        ptrToAccess);
    extractedCmp = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedCmp.getType()),
        extractedCmp);
    extractedValue = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
        extractedValue);
  }
  return rewriter.create<triton::AtomicCASOp>(
      loc, targetType, ptrToAccess, extractedCmp, extractedValue,
      op.getSemAttr(), op.getScopeAttr());
}

template <>
template <typename... Args>
triton::StoreOp UnstructuredMemAccessConverter<triton::StoreOp>::createMemAccOp(
    triton::StoreOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedValue = createExtractOp(loc, op.getValue(), rewriter,
                                        std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  return rewriter.create<triton::StoreOp>(loc, ptrToAccess, extractedValue,
                                          extractedMask, op.getCache(),
                                          op.getEvict());
}

template <>
template <>
LogicalResult
UnstructuredMemAccessConverter<triton::LoadOp>::splatAndLoadScenario<
    triton::LoadOp>(triton::LoadOp op, const mlir::triton::ptr::PtrInfo &info,
                    PatternRewriter &rewriter) const {
  // A masked or volatile load must retain the individual access semantics.
  if (op.getMask() || op.getIsVolatile() || !info.hasScalarBase() || info.block)
    return failure();
  Value origin = getValueOrCreateConstantIndexOp(rewriter, op.getLoc(),
                                                 info.offsets.scalarOffset);
  auto offset =
      castIntegerLike(rewriter, op.getLoc(), origin, rewriter.getI64Type());
  if (failed(offset))
    return failure();
  Value pointer = rewriter.create<triton::AddPtrOp>(
      op.getLoc(), info.base.getType(), info.base, *offset);
  Value loaded = rewriter.create<triton::LoadOp>(
      op.getLoc(), pointer, op.getCache(), op.getEvict(), op.getIsVolatile());
  rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getType(), loaded);
  return success();
}

static triton::LoadOp rewriteSimdDiscreteMaskLoad(triton::LoadOp op,
                                                  PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto ptr = op.getPtr();
  auto mask = op.getMask();
  auto other = op.getOther();
  auto [contMask, discMask] = decomposeAndMask(op, mask, loc, rewriter);
  if (!other) {
    FailureOr<Value> constant = specializeTypelessValueToConstant(
        TypelessValue::Zero, ptr.getType(), loc, rewriter);
    if (failed(constant))
      llvm_unreachable("Unsupported type for constant creation");
    other = *constant;
  }

  triton::LoadOp load;
  if (contMask && discMask) {
    // Bound memory accesses and initialize inactive lanes before selection.
    load = rewriter.create<triton::LoadOp>(loc, ptr, contMask, other,
                                           op.getCache(), op.getEvict(),
                                           op.getIsVolatile());
    mask = discMask;
  } else {
    load = rewriter.create<triton::LoadOp>(loc, ptr, op.getCache(),
                                           op.getEvict(), op.getIsVolatile());
  }
  rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, load.getResult(),
                                               other);
  return load;
}

template <typename MemAccOpTy>
UnstructuredMemAccessConverter<MemAccOpTy>::UnstructuredMemAccessConverter(
    MLIRContext *context, bool forceScalarizeMode,
    const mlir::triton::ptr::PtrInfo &info, bool fromTensorArgument)
    : OpRewritePattern<MemAccOpTy>(context), info(info),
      fromTensorArgument(fromTensorArgument),
      forceScalarizeMode(forceScalarizeMode) {}

template <typename MemAccOpTy>
LogicalResult UnstructuredMemAccessConverter<MemAccOpTy>::matchAndRewrite(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto ptr = op.getPtr();
  auto ptrType = resolvePtrTensorType(ptr);
  auto mixCompileDiscreteMask =
      op->hasAttr(ConverterUtils::mixCompileDiscreteMaskAttrName);
  auto simdDiscreteMask = op->hasAttr(ConverterUtils::simdDiscreteMaskAttrName);
  auto axisDiscreteMask = op->hasAttr(ConverterUtils::axisDiscreteMaskAttrName);
  bool runtimeLoopMask = op->hasAttr(ConverterUtils::runtimeLoopMaskAttrName);

  if (!ptrType || op->hasAttr(ConverterUtils::discreteAttrName))
    return failure();
  unsigned rank = info.offsets.axes.size();
  bool scalarLike =
      !info.block && info.hasScalarBase() && info.offsets.isUniform();
  SmallVector<bool> structuredAxes;
  for (auto [size, axis] :
       llvm::zip_equal(ptrType.getShape(), info.offsets.axes))
    structuredAxes.push_back(
        info.hasScalarBase() &&
        (axis.kind == mlir::triton::ptr::AxisKind::Structured || size == 1 ||
         (info.block && axis.kind == mlir::triton::ptr::AxisKind::Invariant)));
  auto isStructured = [&] {
    return llvm::all_of(structuredAxes, [](bool v) { return v; });
  };
  Value srcPtr = info.base;
  Value ptrOffset;
  if (!info.block) {
    auto full = mlir::triton::ptr::detail::materializeFullOffset(
        info.offsets,
        RankedTensorType::get(ptrType.getShape(), rewriter.getI64Type()),
        rewriter, loc);
    if (failed(full))
      return op.emitError("cannot materialize the complete pointer offset");
    ptrOffset = *full;
  }

  // Handle these accesses before any speculative fallback IR is created.
  // A failed match must leave the original operation intact for the final
  // unsupported-mask diagnostic, without repeatedly inserting dead helpers.
  if (runtimeLoopMask) {
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                  std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      // A one-element tensor needs only a scalar predicate, including in SIMD
      // mode. Use explicit control flow: the legacy scalar masked-load
      // converter otherwise loads unconditionally and selects afterwards.
      if (ptrType.getNumElements() == 1 &&
          getResultElementType(ptrType).isIntOrFloat() &&
          canUseIndirectFastPath(srcPtr, ptrOffset)) {
        SmallVector<OpFoldResult> indices(ptrType.getRank(),
                                          rewriter.getIndexAttr(0));
        Value offset = createExtractOp(loc, ptrOffset, rewriter, indices);
        Value base = srcPtr;
        Value scalarPtr = rewriter.create<triton::AddPtrOp>(loc, base.getType(),
                                                            base, offset);
        Value mask = createExtractOp(loc, op.getMask(), rewriter, indices);
        if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
          Value other = createExtractOp(loc, op.getOther(), rewriter, indices);
          if (!other)
            other = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getZeroAttr(getResultElementType(ptrType)));
          auto guardedLoad = rewriter.create<scf::IfOp>(
              loc, mask,
              [&](OpBuilder &b, Location loc) {
                auto load =
                    b.create<triton::LoadOp>(loc, scalarPtr, op.getCache(),
                                             op.getEvict(), op.getIsVolatile());
                b.create<scf::YieldOp>(loc, load.getResult());
              },
              [&](OpBuilder &b, Location loc) {
                b.create<scf::YieldOp>(loc, other);
              });
          rewriter.replaceOpWithNewOp<triton::SplatOp>(
              op, op.getType(), guardedLoad.getResult(0));
        } else {
          Value value = createExtractOp(loc, op.getValue(), rewriter, indices);
          rewriter.create<scf::IfOp>(
              loc, mask, [&](OpBuilder &b, Location loc) {
                b.create<triton::StoreOp>(loc, scalarPtr, value, op.getCache(),
                                          op.getEvict());
                b.create<scf::YieldOp>(loc);
              });
          rewriter.eraseOp(op);
        }
        return success();
      }
    }
    if (compileOn91095Flag &&
        triton::ascend::isSimtTemplateMode(unstructureCompileMode) &&
        succeeded(tryRewriteIndirectFastPath(op, loc, srcPtr, ptrOffset,
                                             ptrType.getShape(), rewriter)))
      return success();
    return op.emitError(
        "cannot lower this loop-carried mask with a masked "
        "indirect access for the selected target and compile mode");
  }

  if (checkUnstructureAnnotated(op, rewriter))
    llvm::fill(structuredAxes, false);

  bool maskRewritten = false;
  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    if (simdDiscreteMask || (axisDiscreteMask && isStructured())) {
      op = rewriteSimdDiscreteMaskLoad(op, rewriter);
      rewriter.setInsertionPoint(op);
      maskRewritten = true;
    }
  }

  if (isStructured() && !mixCompileDiscreteMask &&
      (!scalarLike ||
       llvm::all_of(ptrType.getShape(), [](int64_t dim) { return dim == 1; })))
    return success();

  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    // A select after an unconditional scalar load cannot protect an invalid
    // pointer when every mask lane is false. Keep masked loads on the guarded
    // scalar-loop path below.
    if (!op.getMask() && scalarLike &&
        succeeded(splatAndLoadScenario(op, info, rewriter))) {
      return success();
    }
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Converting " << op->getName() << "\n";
    os << op << "\n";
    for (auto structured : structuredAxes)
      os << static_cast<int>(structured);
    os << "\n";
    os << scalarLike << "\n";
  });

  std::optional<MaskState> mstate = runMaskAnalysis(op, rewriter);

  normalizeDiscreteMaskAccessForFallback(op, structuredAxes, rewriter);

  if (forceScalarizeMode || scalarLike || fromTensorArgument) {
    llvm::fill(structuredAxes, false);
  }

  if (!isa<triton::PointerType>(srcPtr.getType()) &&
      !isTensorOfPointers(srcPtr.getType()))
    return rewriter.notifyMatchFailure(
        op, "expected a scalar pointer or a tensor of scalar pointers");

  // LoadLike is operation with result
  bool isLoadLike = !op->use_empty();

  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto resultShape = ptrType.getShape();
  auto resultElementType = getResultElementType(ptrType);
  int64_t sizeInByte = getTypeSizeInByte(resultElementType);

  int64_t continuousSize = sizeInByte;
  for (int i = rank - 1; i >= 0; i--) {
    if (!structuredAxes[i])
      break;
    continuousSize *= resultShape[i];
  }

  int64_t discreteSize = sizeInByte;
  for (unsigned i = 0; i < rank; ++i) {
    if (!structuredAxes[i])
      discreteSize *= resultShape[i];
  }

  // Force scalarize if memory is not aligned
  if (continuousSize % 32 != 0) {
    llvm::fill(structuredAxes, false);
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "UnStructured Flag check:\n";
    os << "discrete axis size before scalarization: " << discreteSize << "\n";
    os << "structured access: " << isStructured() << "\n";
    os << "compileOn91095Flag: " << compileOn91095Flag << "\n";
    os << "compileMode: " << static_cast<int>(unstructureCompileMode) << "\n";
  });

  bool templateIndirectFastPathEnabled =
      compileOn91095Flag &&
      triton::ascend::isSimtTemplateMode(unstructureCompileMode) &&
      ((!axisDiscreteMask && !isStructured() &&
        continuousSize < 64) || // cont mask
       (axisDiscreteMask && discreteSize > continuousSize) ||
       mixCompileDiscreteMask);
  bool rankWithinIndirectLoadStoreFastPathLimit = resultShape.size() <= 5;
  if (templateIndirectFastPathEnabled &&
      succeeded(tryRewriteIndirectFastPath(op, loc, srcPtr, ptrOffset,
                                           resultShape, rewriter))) {
    return success();
  }

  // A rejected SIMT load still needs its continuous mask to bound fallback
  // accesses. Continue lowering the replacement load in this rewrite.
  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    if (axisDiscreteMask && !maskRewritten) {
      op = rewriteSimdDiscreteMaskLoad(op, rewriter);
      rewriter.setInsertionPoint(op);
      mstate = runMaskAnalysis(op, rewriter);
      isLoadLike = !op->use_empty();
    }
  }

  LLVM_DEBUG({
    if (triton::ascend::isSimtTemplateMode(unstructureCompileMode) &&
        continuousSize >= 64) {
      auto &os = llvm::dbgs();
      os << "Skip template-SIMT indirect fast path because continuous shape "
            "product is "
         << continuousSize << " (>=64)\n";
    }
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                  std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      if (templateIndirectFastPathEnabled &&
          !rankWithinIndirectLoadStoreFastPathLimit) {
        auto &os = llvm::dbgs();
        os << "Skip tt.indirect_load/store fast path because rank is "
           << resultShape.size() << " (>5), falling back to scalar loop path\n";
      }
    }
  });

  Value iterArg = nullptr;
  Value destinationPadding;
  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    // Prefix masks on partially structured loads can initialize the final
    // destination directly. Loop bounds protect indirect axes; the sliced
    // load below protects retained axes and supplies their inactive values.
    // A fresh destination is essential: reusing the constant `other` tensor
    // can introduce a copy when bufferization sees multiple padded dot inputs.
    // LoadConverter carries this fill onto the actual allocation used by DMA.
    // Keep the existing select path for unknown/non-prefix masks, arbitrary
    // tensor other values, scalar fallback and nonzero padding.
    if (isLoadLike && op.getMask() && op.getOther() && mstate &&
        mstate->isMask() && mstate->dims.size() == resultShape.size() &&
        mstate->offsets.size() == resultShape.size() &&
        llvm::any_of(llvm::zip_equal(structuredAxes, resultShape),
                     [](auto axis) {
                       auto [structured, size] = axis;
                       return structured && size > 1;
                     }) &&
        llvm::all_of(mstate->offsets, [](OpFoldResult offset) {
          return isConstantIntValue(offset, 0);
        })) {
      Value padding = op.getOther();
      while (auto *def = padding.getDefiningOp()) {
        if (!isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp>(
                def))
          break;
        padding = def->getOperand(0);
      }
      Attribute attr;
      if (matchPattern(padding, m_Constant(&attr)) &&
          (matchPattern(padding, m_Zero()) ||
           matchPattern(padding, m_AnyZeroFloat()))) {
        if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
          if (dense.isSplat())
            attr = dense.getSplatValue<Attribute>();
        }
        if (isa<FloatAttr, IntegerAttr>(attr))
          destinationPadding =
              rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(attr));
      }
    }
  }

  // Only load case
  if (isLoadLike) {
    if (destinationPadding) {
      Value empty =
          rewriter.create<tensor::EmptyOp>(loc, resultShape, resultElementType);
      iterArg = rewriter
                    .create<linalg::FillOp>(loc, ValueRange{destinationPadding},
                                            ValueRange{empty})
                    .getResult(0);
    } else {
      iterArg =
          rewriter.create<tensor::EmptyOp>(loc, resultShape, resultElementType);
    }
  }
  Value newOpResult = nullptr;

  auto insertPoint = rewriter.saveInsertionPoint();

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<int64_t> extractedShape;
  SmallVector<bool> boundedLoopAxes(resultShape.size(), false);

  for (size_t i = 0; i < resultShape.size(); i++) {
    auto size = resultShape[i];
    bool structured = structuredAxes[i];
    // handle indirect dimension
    strides.push_back(rewriter.getIndexAttr(1));
    Value sizeVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(size));
    if (structured) {
      offsets.push_back(rewriter.getIndexAttr(0));
      sizes.push_back(rewriter.getIndexAttr(size));
      extractedShape.push_back(size);
    } else {
      scf::ForOp forOp;

      Value loopLower = zeroIdx;
      Value loopUpper = sizeVal;
      if (mstate && i < mstate->dims.size() && i < mstate->offsets.size()) {
        Value maskOffset =
            getValueOrCreateConstantIndexOp(rewriter, loc, mstate->offsets[i]);
        maskOffset = rewriter.create<arith::MaxSIOp>(loc, maskOffset, zeroIdx);
        maskOffset = rewriter.create<arith::MinSIOp>(loc, maskOffset, sizeVal);
        loopLower = maskOffset;

        Value maskDim =
            getValueOrCreateConstantIndexOp(rewriter, loc, mstate->dims[i]);
        maskDim = rewriter.create<arith::AddIOp>(loc, maskOffset, maskDim);
        maskDim = rewriter.create<arith::MinSIOp>(loc, maskDim, sizeVal);
        loopUpper = maskDim;
        // With a zero mask offset, [0, loopUpper) is contained in the
        // analyzed interval. Do not infer this for a clipped negative start:
        // clamping the start can otherwise shift a nonempty interval.
        boundedLoopAxes[i] = mstate->isMask() &&
                             mstate->dims.size() == resultShape.size() &&
                             isConstantIntValue(mstate->offsets[i], 0);
      }

      if (isLoadLike) {
        forOp = rewriter.create<scf::ForOp>(loc, loopLower, loopUpper, oneIdx,
                                            ValueRange({iterArg}));
        if (!newOpResult) {
          newOpResult = forOp->getResult(0);
        } else {
          rewriter.create<scf::YieldOp>(loc, forOp->getResult(0));
        }
        iterArg = forOp.getRegionIterArg(0);
      } else {
        forOp = rewriter.create<scf::ForOp>(loc, loopLower, loopUpper, oneIdx);
      }
      sizes.push_back(rewriter.getIndexAttr(1));
      offsets.push_back(forOp.getInductionVar());
      extractedShape.push_back(1);
      forOp->setAttr("ExtractedLoadOrStore",
                     UnitAttr::get(rewriter.getContext()));
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
  }

  bool fullyUnstructured =
      llvm::none_of(structuredAxes, [](bool v) { return v; });
  auto extractedType = RankedTensorType::get(extractedShape, resultElementType);

  Value extractedOffset;
  if (info.block) {
    if (!fullyUnstructured)
      return op.emitError("partial block-pointer scalarization is unsupported");
    SmallVector<Value> indices;
    for (OpFoldResult offset : offsets)
      indices.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, offset));
    auto scalar = mlir::triton::ptr::materializeOffsetAt(info.offsets, indices,
                                                         rewriter, loc);
    if (failed(scalar))
      return op.emitError("cannot materialize the block pointer offset");
    Value indexOffset = getValueOrCreateConstantIndexOp(rewriter, loc, *scalar);
    extractedOffset =
        *castIntegerLike(rewriter, loc, indexOffset, rewriter.getI64Type());
  } else if (fullyUnstructured) {
    extractedOffset = createExtractOp(loc, ptrOffset, rewriter, offsets);
  } else {
    extractedOffset =
        createExtractOp(loc, ptrOffset, rewriter, offsets, sizes, strides);
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracted offset\n";
    os << extractedOffset << "\n";
  });

  // A tensor-of-pointers base is opaque: each lane may select a different
  // allocation. Extract the base with the same scalar or slice coordinates as
  // the offset, then form the access pointer lane by lane.
  if (isTensorOfPointers(srcPtr.getType())) {
    if (fullyUnstructured)
      srcPtr = createExtractOp(loc, srcPtr, rewriter, offsets);
    else
      srcPtr = createExtractOp(loc, srcPtr, rewriter, offsets, sizes, strides);
  }

  assert((isa<triton::PointerType>(srcPtr.getType()) ||
          isTensorOfPointers(srcPtr.getType())) &&
         "src must be a scalar pointer or tensor of pointers");
  if (!fullyUnstructured && isa<triton::PointerType>(srcPtr.getType())) {
    srcPtr = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(extractedShape, srcPtr.getType()), srcPtr);
  }
  Value ptrToAccess = rewriter.create<triton::AddPtrOp>(
      loc, srcPtr.getType(), srcPtr, extractedOffset);

  MemAccOpTy accessedOp;
  if (fullyUnstructured) {
    accessedOp = createMemAccOp(op, ptrToAccess, loc, rewriter, offsets);
  } else {
    accessedOp =
        createMemAccOp(op, ptrToAccess, loc, rewriter, offsets, sizes, strides);
  }
  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                std::is_same_v<MemAccOpTy, triton::StoreOp>) {
    if (info.block && fullyUnstructured && !op.getBoundaryCheck().empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(accessedOp);
      Value predicate = accessedOp.getMask();
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      for (int32_t dim : op.getBoundaryCheck()) {
        Value coordinate = getValueOrCreateConstantIndexOp(
            rewriter, loc,
            addOpFoldResult(info.block->coordinateOffsets[dim], offsets[dim],
                            loc, rewriter));
        Value extent = getValueOrCreateConstantIndexOp(
            rewriter, loc, info.block->parentShape[dim]);
        Value lower = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, coordinate, zero);
        Value upper = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, coordinate, extent);
        Value valid = rewriter.create<arith::AndIOp>(loc, lower, upper);
        predicate = predicate
                        ? rewriter.create<arith::AndIOp>(loc, predicate, valid)
                        : valid;
      }
      accessedOp.getMaskMutable().assign(predicate);
      if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
        if (auto padding = op.getPadding()) {
          TypedAttr value = rewriter.getZeroAttr(resultElementType);
          if (*padding == triton::PaddingOption::PAD_NAN)
            value = rewriter.getFloatAttr(
                resultElementType,
                llvm::APFloat::getNaN(
                    cast<FloatAttr>(value).getValue().getSemantics()));
          accessedOp.getOtherMutable().assign(
              rewriter.create<arith::ConstantOp>(loc, value).getResult());
        }
      }
    }
  }
  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    // Preserve the mask on retained dimensions, just as stores and atomics do.
    // BubbleUpOperation exposes the sliced mask for structured lowering.
    if (mstate && !fullyUnstructured) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(accessedOp);
      Value residualMask =
          getResidualLoadMask(op.getMask(), boundedLoopAxes, rewriter);
      if (residualMask) {
        accessedOp.getMaskMutable().assign(createExtractOp(
            loc, residualMask, rewriter, offsets, sizes, strides));
        if (destinationPadding) {
          Value other = rewriter.create<triton::SplatOp>(loc, extractedType,
                                                         destinationPadding);
          accessedOp.getOtherMutable().assign(other);
        }
      } else {
        // The loop bounds enforce the complete mask. Padding for unvisited
        // elements remains in the destination fill or outer select below.
        accessedOp.getMaskMutable().clear();
        accessedOp.getOtherMutable().clear();
      }
    }
  }

  accessedOp->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));

  if (isLoadLike) {
    assert(iterArg && "Load case must have iterArg in for loop");

    Value value = accessedOp->getResult(0);
    Value result;
    if (!isa<RankedTensorType>(value.getType()) &&
        (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp> ||
         std::is_same_v<MemAccOpTy, triton::AtomicCASOp>)) {
      value = rewriter.create<triton::SplatOp>(loc, extractedType, value);
    }
    if (!isa<RankedTensorType>(value.getType())) {
      SmallVector<Value> indices;
      for (auto idxOfr : offsets) {
        auto idx = getValueOrCreateConstantIndexOp(rewriter, loc, idxOfr);
        indices.push_back(idx);
      }
      result = rewriter.create<tensor::InsertOp>(loc, value, iterArg, indices);
    } else {
      result = rewriter.create<tensor::InsertSliceOp>(loc, value, iterArg,
                                                      offsets, sizes, strides);
    }
    rewriter.create<scf::YieldOp>(loc, result)
        ->setAttr(ConverterUtils::discreteAttrName,
                  UnitAttr::get(rewriter.getContext()));
    rewriter.restoreInsertionPoint(insertPoint);
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
      if (op.getMask() && op.getOther() && !destinationPadding) {
        rewriter
            .replaceOpWithNewOp<arith::SelectOp>(op, op.getMask(), newOpResult,
                                                 op.getOther())
            ->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));
      } else {
        rewriter.replaceOp(op, newOpResult);
      }
    } else {
      rewriter.replaceOp(op, newOpResult);
    }
  } else {
    if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
      if (fullyUnstructured && accessedOp.getMask()) {
        auto mask = createExtractOp(
            loc, accessedOp.getMask(), rewriter,
            SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(0)));
        rewriter.create<scf::IfOp>(loc, mask, [&](OpBuilder &b, Location loc) {
          b.create<triton::AtomicRMWOp>(
               loc, accessedOp.getType(), accessedOp.getAtomicRmwOp(),
               accessedOp.getPtr(), accessedOp.getVal(), nullptr,
               accessedOp.getSem(), accessedOp.getScope())
              ->setAttr(ConverterUtils::discreteAttrName,
                        UnitAttr::get(rewriter.getContext()));
          b.create<scf::YieldOp>(loc);
        });
        rewriter.eraseOp(accessedOp);
      }
    }
    rewriter.eraseOp(op);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After conversion\n"
       << ptrToAccess.getDefiningOp()
              ->template getParentOfType<triton::FuncOp>()
       << "\n";
  });
  return success();
}

static bool isFromTensorArg(Value v,
                            llvm::SmallDenseMap<Value, bool> &fromTensorArg) {
  if (fromTensorArg.contains(v))
    return fromTensorArg.at(v);
  auto *defOp = v.getDefiningOp();
  if (!defOp) {
    fromTensorArg[v] = isa<RankedTensorType>(v.getType());
    return isa<RankedTensorType>(v.getType());
  }
  for (auto opr : defOp->getOperands()) {
    if (isFromTensorArg(opr, fromTensorArg)) {
      fromTensorArg[v] = true;
      return true;
    }
  }
  fromTensorArg[v] = false;
  return false;
}

// A query owns one stable source-IR session. Consume its facts immediately;
// the next memory operation is analyzed after all earlier RAUW has completed.
template <typename OpTy>
static LogicalResult lowerMemoryAccess(OpTy op, bool forceScalarize,
                                       PatternRewriter &rewriter) {
  if (!resolvePtrTensorType(op.getPtr()) ||
      op->hasAttr(ConverterUtils::discreteAttrName))
    return success();
  rewriter.setInsertionPoint(op);
  FailureOr<mlir::triton::ptr::PtrInfo> info = failure();
  {
    mlir::triton::ptr::PtrAnalysis analysis(rewriter);
    if (failed(importPointerDescriptorFacts(
            op->template getParentOfType<triton::FuncOp>(), analysis)))
      return failure();
    info = analysis.analyze(op.getPtr());
  }
  if (failed(info))
    return op.emitError("public pointer analysis failed");
  llvm::SmallDenseMap<Value, bool> fromTensorArg;
  bool fromArgument = isFromTensorArg(op.getPtr(), fromTensorArg);
  UnstructuredMemAccessConverter<OpTy> converter(
      op.getContext(), forceScalarize, *info, fromArgument);
  return converter.matchAndRewrite(op, rewriter);
}

LogicalResult
TritonToUnstructurePass::processIfYieldAddHoistOperations(ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CannonicalizerConverter::IfYieldAddHoistConverter>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp.emitWarning("IfYieldAddHoist processing failed");
    return failure();
  }
  return success();
}

TritonToUnstructurePass::TritonToUnstructurePass(
    const TritonToUnstructureOptions &options)
    : TritonToUnstructureBase(options) {}

void TritonToUnstructurePass::runOnOperation() {
  compileOn91095Flag = this->compileOn91095;
  auto compileMode = triton::ascend::parseCompileMode(this->compileMode);
  if (!compileMode) {
    getOperation().emitError()
        << "triton-to-unstructure compile-mode is invalid: "
        << this->compileMode;
    signalPassFailure();
    return;
  }
  unstructureCompileMode = *compileMode;

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "TritonToUnstructurePass started with options:\n";
    os << "  compileOn91095: " << compileOn91095Flag << "\n";
    os << "  compileMode: " << this->compileMode << "\n";
  });

  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  canonicalizeSupportedPointerBitcasts(moduleOp);
  // Fix one permitted lane order before analyzing a pointer reshape. The
  // original pointers and the derived integer offsets must share that order.
  moduleOp.walk([](triton::ReshapeOp op) {
    if (isTensorOfPointers(op.getType()) && !op.getType().getEncoding())
      op->removeAttr("allow_reorder");
  });
  SmallVector<Operation *> customOps;
  moduleOp.walk([&](Operation *op) {
    if (isDistributedTypeCustomOp(op))
      customOps.push_back(op);
  });
  IRRewriter annotationRewriter(ctx);
  for (Operation *op : customOps)
    (void)checkStructureAnnotated(op, annotationRewriter);

  for (auto func : moduleOp.getOps<triton::FuncOp>()) {
    if (failed(replacePtrArguments(func))) {
      signalPassFailure();
      return;
    }
  }
  if (failed(processIfYieldAddHoistOperations(moduleOp))) {
    signalPassFailure();
    return;
  }
  SmallVector<Operation *> accesses;
  moduleOp.walk([&](Operation *op) {
    if (isa<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp,
            triton::AtomicCASOp>(op))
      accesses.push_back(op);
  });
  PatternRewriter rewriter(ctx);
  for (Operation *access : accesses) {
    LogicalResult result =
        llvm::TypeSwitch<Operation *, LogicalResult>(access)
            .Case<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp,
                  triton::AtomicCASOp>([&](auto op) {
              return lowerMemoryAccess(op, forceScalarizeMode, rewriter);
            });
    if (failed(result)) {
      signalPassFailure();
      return;
    }
  }

  bool unsupportedLoopMask = false;
  moduleOp.walk([&](Operation *op) {
    if (op->hasAttr(ConverterUtils::runtimeLoopMaskAttrName)) {
      op->emitError("cannot lower this loop-carried mask with a masked "
                    "indirect access for the selected target and compile mode");
      unsupportedLoopMask = true;
    }
  });
  if (unsupportedLoopMask) {
    signalPassFailure();
    return;
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }
}

void TritonToUnstructurePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect,
                  affine::AffineDialect, scf::SCFDialect, tensor::TensorDialect,
                  bufferization::BufferizationDialect, memref::MemRefDialect,
                  triton::TritonDialect, triton::ascend::TritonAscendDialect,
                  hivm::HIVMDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToUnstructurePass(
    const TritonToUnstructureOptions &options) {
  return std::make_unique<TritonToUnstructurePass>(options);
}
