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
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToStructured/CannonicalizerConverter.h"
#include "TritonToUnstructure/IndirectAtomicUtils.h"
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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "triton-unstructure-converter"

using namespace mlir;
using namespace triton;

#include "llvm/Support/Debug.h"

bool forceSimtTemplateFlag = false;

namespace {

constexpr int64_t kBitsPerByte = 8;

static constexpr const char *kRouteDiscreteMaskToSimtAttrName =
    "route_discrete_mask_to_simt";

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

static unsigned getPointeeByteWidth(triton::PointerType pointerType) {
  if (!pointerType)
    return 0;
  Type pointeeType = pointerType.getPointeeType();
  if (auto shapedType = dyn_cast<ShapedType>(pointeeType))
    pointeeType = shapedType.getElementType();
  if (!pointeeType.isIntOrFloat())
    return 0;
  unsigned bitWidth = pointeeType.getIntOrFloatBitWidth();
  if (bitWidth == 1)
    bitWidth = kBitsPerByte;
  if (bitWidth < kBitsPerByte || bitWidth % kBitsPerByte != 0)
    return 0;
  return bitWidth / kBitsPerByte;
}

static triton::PointerType getScalarPointerType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    type = shapedType.getElementType();
  return dyn_cast<triton::PointerType>(type);
}

static bool isDifferentWidthPointerBitcast(triton::BitcastOp op) {
  auto sourceType = getScalarPointerType(op.getSrc().getType());
  auto targetType = getScalarPointerType(op.getType());
  unsigned sourceWidth = getPointeeByteWidth(sourceType);
  unsigned targetWidth = getPointeeByteWidth(targetType);
  return sourceWidth && targetWidth && sourceWidth != targetWidth;
}

static bool isPointerValue(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  return isa<triton::PointerType>(shapedType ? shapedType.getElementType()
                                             : type);
}

// Byte-addressed offsets cannot be reconstructed by ReplaceArguments, whose
// tt.addptr rebuilding interprets every offset in the root pointee's element
// unit. Track only SSA values derived from a different-width pointer bitcast
// and reject an actual structured-control-flow transfer. Capturing such a value
// inside a region remains valid because no pointer state is reconstructed.
static LogicalResult rejectByteAddressedControlFlow(ModuleOp moduleOp) {
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  moduleOp.walk([&](triton::BitcastOp op) {
    if (isDifferentWidthPointerBitcast(op))
      worklist.push_back(op.getResult());
  });

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<scf::YieldOp, scf::ConditionOp, triton::ReturnOp, triton::CallOp>(
              user) ||
          isa<LoopLikeOpInterface>(user)) {
        user->emitError(
            "byte-addressed pointer cannot cross structured control flow or "
            "a function boundary");
        return failure();
      }

      // Memory operations consume the address but do not forward it. For all
      // other operations, conservatively propagate provenance through every
      // pointer result so an unfamiliar pointer-shaping op cannot hide a later
      // control-flow transfer.
      if (isa<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp,
              triton::AtomicCASOp, triton::PtrToIntOp>(user))
        continue;
      for (Value result : user->getResults())
        if (isPointerValue(result.getType()))
          worklist.push_back(result);
    }
  }
  return success();
}

static LogicalResult
validateByteAddressedPointerTypes(triton::PointerType sourceType,
                                  triton::PointerType targetType,
                                  Operation *diagnosticOp) {
  if (!sourceType || !targetType) {
    diagnosticOp->emitError(
        "byte-addressed pointer access requires a scalar root pointer");
    return failure();
  }
  if (!getPointeeByteWidth(sourceType) || !getPointeeByteWidth(targetType)) {
    diagnosticOp->emitError(
        "different-width pointer bitcast requires byte-addressable scalar "
        "integer or floating-point pointee types");
    return failure();
  }
  if (sourceType.getAddressSpace() != targetType.getAddressSpace()) {
    diagnosticOp->emitError(
        "cannot bitcast pointers between different address spaces");
    return failure();
  }
  return success();
}

template <typename MemAccOpTy>
void normalizeDiscreteMaskAccessForFallback(MemAccOpTy &op,
                                            PtrOffsetInfo &ptrOffsetInfo,
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
  ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
}

// ======================== 950 SIMT Indirect Fast-Path Lowering
// ========================
// 1. SIMT Fast-Path Gate
//    The SIMT indirect lowering path is enabled only when:
//      - compileOn91095Flag && forceSimtTemplateFlag
//      - and the access is either:
//          * unstructured, or has tag with 'route_discrete_mask_to_simt'
//
// 2. Op-Specific Lowering
//    (1) tt.load / tt.store
//        Entry requirements:
//          - SIMT fast-path gate enabled
//          - tensor rank <= 5, simt template only supports up to 5D tensors for
//          now
//        Lowering:
//          - tt.load / tt.store  -> tt.indirect_load / tt.indirect_store
//    (2) tt.atomic_rmw / tt.atomic_cas
//        Entry requirements:
//          - SIMT fast-path gate enabled
//          - offset/value/mask tensors have static shape (required for
//          flatten-to-1D lowering)
//        Lowering:
//          tt.atomic_rmw fadd, acq_rel, gpu, %src, %value, %mask
//          -> flatten offsets/data/mask to 1D
//          -> create a custom op:
//              hivm.hir.custom {
//                extra_attr = "operate=<atomic_op>"
//              } "__builtin_indirect_atomic" ins(%ptr, %offset, %value, %mask)
//              outs(%out)
//          -> reshape the returned 1D result back to the original tensor shape
//
// 3. Fallback Behavior
//    If SIMT indirect lowering cannot be formed for any operation,
//    conversion gracefully falls back to the legacy scalar-loop lowering path
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
    auto newPtr = srcPtr;
    if (auto *defOp = srcPtr.getDefiningOp()) {
      if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(defOp)) {
        auto zeroOffset = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(intToPtrOp.getSrc().getType()));
        newPtr = rewriter.create<triton::AddPtrOp>(loc, srcPtr.getType(),
                                                   srcPtr, zeroOffset);
      }
    }
    auto indirect = rewriter.create<triton::ascend::IndirectLoadOp>(
        loc, resultType, newPtr, ptrOffset, mask, other,
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

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> iterIdx) const {
  if (!value)
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
    PatternRewriter &rewriter, bool preserveLoadMask, Args &&...args) const {
  if (preserveLoadMask) {
    Value mask = createExtractOp(loc, op.getMask(), rewriter,
                                 std::forward<Args>(args)...);
    Value other = createExtractOp(loc, op.getOther(), rewriter,
                                  std::forward<Args>(args)...);
    return rewriter.create<triton::LoadOp>(loc, ptrToAccess, mask, other,
                                           op.getCache(), op.getEvict(),
                                           op.getIsVolatile());
  }
  return rewriter.create<triton::LoadOp>(loc, ptrToAccess, op.getCache(),
                                         op.getEvict(), op.getIsVolatile());
}

template <>
template <typename... Args>
triton::AtomicRMWOp
UnstructuredMemAccessConverter<triton::AtomicRMWOp>::createMemAccOp(
    triton::AtomicRMWOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, bool /*preserveLoadMask*/,
    Args &&...args) const {
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
    PatternRewriter &rewriter, bool /*preserveLoadMask*/,
    Args &&...args) const {
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
    PatternRewriter &rewriter, bool /*preserveLoadMask*/,
    Args &&...args) const {
  auto extractedValue = createExtractOp(loc, op.getValue(), rewriter,
                                        std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  return rewriter.create<triton::StoreOp>(loc, ptrToAccess, extractedValue,
                                          extractedMask);
}

template <>
template <>
void UnstructuredMemAccessConverter<triton::LoadOp>::splatAndLoadScenario<
    triton::LoadOp>(triton::LoadOp op, int rank,
                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  SmallVector<OpFoldResult> idx(rank, rewriter.getIndexAttr(0));
  auto extractedPtr = createExtractOp(loc, op.getPtr(), rewriter, idx);
  Value mask = op.getMask();
  Value other = op.getOther();
  Value loadedValue = rewriter.create<triton::LoadOp>(
      loc, extractedPtr, /*mask=*/nullptr, /*other=*/nullptr,
      /*boundaryCheck=*/ArrayRef<int32_t>(),
      /*PaddingOptionAttr=*/nullptr);
  loadedValue = rewriter.create<triton::SplatOp>(loc, op.getResult().getType(),
                                                 loadedValue);
  if (mask)
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, loadedValue, other);
  else
    rewriter.replaceOp(op, loadedValue);
}

template <typename MemAccOpTy>
UnstructuredMemAccessConverter<MemAccOpTy>::UnstructuredMemAccessConverter(
    MLIRContext *context, bool forceScalarizeMode,
    const llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
    const llvm::SmallDenseMap<Value, bool> &fromTensorArg)
    : OpRewritePattern<MemAccOpTy>(context),
      forceScalarizeMode(forceScalarizeMode), offsetMap(offsetMap),
      fromTensorArg(fromTensorArg) {}

template <typename MemAccOpTy>
LogicalResult UnstructuredMemAccessConverter<MemAccOpTy>::matchAndRewrite(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto ptr = op.getPtr();
  auto ptrType = resolvePtrTensorType(ptr);
  auto routeDiscreteMaskToSimt = op->hasAttr(kRouteDiscreteMaskToSimtAttrName);

  if (!ptrType || op->hasAttr(ConverterUtils::discreteAttrName))
    return failure();
  // Structural analysis failures are diagnosed once by the pass before greedy
  // rewriting. Returning failure here remains a plain pattern non-match.
  if (!offsetMap.contains(ptr))
    return failure();

  auto ptrOffsetInfo = offsetMap.at(ptr);

  auto srcPtr = ptrOffsetInfo.getPtr();
  auto sourcePointerType =
      srcPtr ? dyn_cast<triton::PointerType>(srcPtr.getType()) : nullptr;
  auto targetPointerType =
      dyn_cast<triton::PointerType>(ptrType.getElementType());
  if (!srcPtr)
    return failure();
  bool byteAddressed = ptrOffsetInfo.isByteAddressed();
  if (byteAddressed &&
      failed(validateByteAddressedPointerTypes(
          sourcePointerType, targetPointerType, op.getOperation())))
    return failure();

  if (checkUnstructureAnnotated(op, rewriter) || byteAddressed)
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  if (byteAddressed) {
    // The byte offset is exact but no longer affine in the target element
    // unit. Keep every lane opaque until it is materialized as an integer
    // address below. In particular, do not route atomics through BlockData.
    ptrOffsetInfo.setScalarLike(false);
  }

  if (ptrOffsetInfo.isStructured() && !routeDiscreteMaskToSimt &&
      (!ptrOffsetInfo.isScalarLike() ||
       llvm::all_of(ptrType.getShape(), [](int64_t dim) { return dim == 1; })))
    return failure();

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Converting " << op->getName() << "\n";
    os << op << "\n";
    for (auto structured : ptrOffsetInfo.getStructuredRef())
      os << static_cast<int>(structured);
    os << "\n";
    os << ptrOffsetInfo.isScalarLike() << "\n";
  });

  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    if (ptrOffsetInfo.isScalarLike()) {
      splatAndLoadScenario(op, ptrOffsetInfo.getRank(), rewriter);
      return success();
    }
  }

  // Same-width pointer bitcasts do not change the offset unit, so analysis
  // deliberately keeps the original scalar root. Create the typed view here,
  // where the cast immediately has a real memory-rewrite use and cannot become
  // a dangling analysis-only SSA value. targetPointerType is present only for
  // tensor-of-pointers; block pointers have a data tensor pointee and must stay
  // on the existing make_tensor_ptr path. Different-width boundaries use the
  // byte-addressed path below and must not be handled here.
  if (!byteAddressed && targetPointerType &&
      sourcePointerType != targetPointerType) {
    if (!sourcePointerType || !targetPointerType ||
        sourcePointerType.getAddressSpace() !=
            targetPointerType.getAddressSpace() ||
        getPointeeByteWidth(sourcePointerType) !=
            getPointeeByteWidth(targetPointerType)) {
      return op.emitError(
          "element-addressed pointer access requires matching pointer widths "
          "and address spaces");
    }
    srcPtr = rewriter.create<triton::BitcastOp>(loc, targetPointerType, srcPtr);
  }

  std::optional<MaskState> mstate = runMaskAnalysis(op, rewriter);

  normalizeDiscreteMaskAccessForFallback(op, ptrOffsetInfo, rewriter);

  if (forceScalarizeMode || ptrOffsetInfo.isScalarLike() ||
      fromTensorArg.at(ptr)) {
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  }

  auto ptrOffset = ptrOffsetInfo.getOffset();

  // LoadLike is operation with result
  bool isLoadLike = !op->use_empty();

  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto resultShape = ptrType.getShape();
  auto resultElementType = getResultElementType(ptrType);
  int64_t sizeInByte = getTypeSizeInByte(resultElementType);

  for (int i = ptrOffsetInfo.getRank() - 1; i >= 0; i--) {
    if (!ptrOffsetInfo.isStructured(i))
      break;
    sizeInByte *= resultShape[i];
  }

  // Force scalarize if memory is not aligned
  if (sizeInByte % 32 != 0) {
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "UnStructured Flag check:\n";
    os << "ptrOffsetInfo.isStructured: " << ptrOffsetInfo.isStructured()
       << "\n";
    os << "compileOn91095Flag: " << compileOn91095Flag << "\n";
    os << "forceSimtTemplateFlag: " << forceSimtTemplateFlag << "\n";
  });

  // SIMT Indirect Fast-Path Lowering in 950 seiries
  bool indirectFastPathEnabled =
      compileOn91095Flag && forceSimtTemplateFlag &&
      ((!ptrOffsetInfo.isStructured() && sizeInByte < 64) ||
       routeDiscreteMaskToSimt);
  bool rankWithinIndirectLoadStoreFastPathLimit = resultShape.size() <= 5;
  // TODO: Preserve the indirect SIMT fast path by materializing a vectorized
  // typed view from the exact byte addresses. Until that representation is
  // supported, use the existing scalar-loop lowering.
  if (indirectFastPathEnabled && !byteAddressed &&
      succeeded(tryRewriteIndirectFastPath(op, loc, srcPtr, ptrOffset,
                                           resultShape, rewriter))) {
    return success();
  }

  LLVM_DEBUG({
    if (sizeInByte >= 64) {
      auto &os = llvm::dbgs();
      os << "Skip SIMT indirect fast path because continuous shape product is "
         << sizeInByte << " (>=64)\n";
    }
    if (indirectFastPathEnabled && byteAddressed) {
      llvm::dbgs() << "Skip SIMT indirect fast path for a different-width "
                      "pointer bitcast\n";
    }
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                  std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      if (indirectFastPathEnabled &&
          !rankWithinIndirectLoadStoreFastPathLimit) {
        auto &os = llvm::dbgs();
        os << "Skip tt.indirect_load/store fast path because rank is "
           << resultShape.size() << " (>5), falling back to scalar loop path\n";
      }
    }
  });

  Value iterArg = nullptr;

  // Only load case
  if (isLoadLike) {
    iterArg =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, resultElementType);
  }
  Value newOpResult = nullptr;

  auto insertPoint = rewriter.saveInsertionPoint();

  Value byteBaseAddress;
  if (byteAddressed) {
    // PtrOffsetInfo::offset is a signed byte displacement in this mode. Hoist
    // conversion of the root pointer out of the generated scalar loops, then
    // add only the per-lane byte displacement inside the loop body.
    byteBaseAddress =
        rewriter.create<triton::PtrToIntOp>(loc, rewriter.getI64Type(), srcPtr);
  }

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<int64_t> extractedShape;

  for (size_t i = 0; i < resultShape.size(); i++) {
    auto size = resultShape[i];
    auto structured = ptrOffsetInfo.getStructuredRef()[i] ==
                      PtrOffsetInfo::AxisInfo::structured;
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
      if (auto mtptOp =
              srcPtr.template getDefiningOp<triton::MakeTensorPtrOp>()) {
        auto tptShape = mtptOp.getShape()[i];
        if (tptShape.getType() != rewriter.getIndexType()) {
          tptShape = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), tptShape);
        }
        sizeVal = rewriter.create<arith::MinSIOp>(loc, sizeVal, tptShape);
      }

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

  bool fullyUnstructured = ptrOffsetInfo.isUnstructuredOrScalarlike();
  auto extractedType = RankedTensorType::get(extractedShape, resultElementType);

  Value extractedOffset;
  if (fullyUnstructured) {
    if (auto mtptOp =
            srcPtr.template getDefiningOp<triton::MakeTensorPtrOp>()) {
      auto I64Type = rewriter.getIntegerType(64);
      srcPtr = mtptOp.getBase();
      extractedOffset = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      for (auto [indVar, offset, stride] : llvm::zip_equal(
               offsets, ptrOffsetInfo.getOffsets(), mtptOp.getStrides())) {
        Value inductionVar = rewriter.create<arith::IndexCastOp>(
            loc, I64Type, cast<Value>(indVar));
        Value tptOffset = rewriter.create<arith::ExtSIOp>(loc, I64Type, offset);
        Value tptStride = rewriter.create<arith::ExtSIOp>(loc, I64Type, stride);
        tptOffset = rewriter.create<arith::MulIOp>(loc, tptStride, tptOffset);
        tptStride =
            rewriter.create<arith::MulIOp>(loc, tptStride, inductionVar);
        extractedOffset =
            rewriter.create<arith::AddIOp>(loc, extractedOffset, tptOffset);
        extractedOffset =
            rewriter.create<arith::AddIOp>(loc, extractedOffset, tptStride);
      }
    } else {
      extractedOffset = createExtractOp(loc, ptrOffset, rewriter, offsets);
    }
  } else {
    extractedOffset =
        createExtractOp(loc, ptrOffset, rewriter, offsets, sizes, strides);
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracted offset\n";
    os << extractedOffset << "\n";
  });

  assert(isa<triton::PointerType>(srcPtr.getType()) && "src must be ptr type");
  if (!fullyUnstructured && !byteAddressed) {
    srcPtr = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(extractedShape, srcPtr.getType()), srcPtr);
  }
  Value ptrToAccess;
  if (byteAddressed) {
    assert(fullyUnstructured &&
           "byte-addressed pointer access must be scalarized");
    Value exactAddress =
        rewriter.create<arith::AddIOp>(loc, byteBaseAddress, extractedOffset);
    auto intToPtr = rewriter.create<triton::IntToPtrOp>(loc, targetPointerType,
                                                        exactAddress);
    intToPtr->setAttr(ConverterUtils::pointerBitcastPointerCastAttrName,
                      rewriter.getUnitAttr());
    ptrToAccess = intToPtr;
  } else {
    ptrToAccess = rewriter.create<triton::AddPtrOp>(loc, srcPtr.getType(),
                                                    srcPtr, extractedOffset);
  }

  MemAccOpTy accessedOp;
  if (fullyUnstructured) {
    accessedOp =
        createMemAccOp(op, ptrToAccess, loc, rewriter, byteAddressed, offsets);
  } else {
    accessedOp = createMemAccOp(op, ptrToAccess, loc, rewriter, byteAddressed,
                                offsets, sizes, strides);
  }

  accessedOp->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));
  if (byteAddressed)
    accessedOp->setAttr(ConverterUtils::pointerBitcastPointerCastAttrName,
                        rewriter.getUnitAttr());

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
      if (op.getMask() && op.getOther()) {
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
            SmallVector<OpFoldResult>(ptrOffsetInfo.getRank(),
                                      rewriter.getIndexAttr(0)));
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

static LogicalResult
runPreparseChecked(LoopLikeOpInterface op, MLIRContext *context,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMapForLoopArgs) {
  IRRewriter rewriter(context);
  Location loc = op.getLoc();

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Pre-parsing " << op->getName() << "\n" << op << "\n";
  });

  Block::BlockArgListType args;
  ValueRange yields;
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    args = whileOp.getBeforeArguments();
    yields = whileOp.getYieldOp().getOperands();
  } else {
    args = op.getRegionIterArgs();
    yields = op.getYieldedValues();
  }

  for (auto [arg, yield] : llvm::zip_equal(args, yields)) {
    if (!isa<RankedTensorType>(yield.getType()))
      continue;
    if (failed(parseChecked(yield, loc, rewriter, offsetMapForLoopArgs)))
      return failure();
    offsetMap[arg] = offsetMapForLoopArgs.at(yield);
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Pre-parsing result of\n" << arg << "\nis ";
      for (auto structured : offsetMap[arg].getStructuredRef())
        os << static_cast<int>(structured);
      os << '\n';
    });
  }
  return success();
}

void TritonToUnstructurePass::runPreparse(LoopLikeOpInterface op) {
  if (failed(runPreparseChecked(op, &getContext(), offsetMap,
                                offsetMapForLoopArgs)))
    signalPassFailure();
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

template <typename MemAccOpTy>
static LogicalResult
runParseChecked(MemAccOpTy op, MLIRContext *context,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                llvm::SmallDenseMap<Value, bool> &fromTensorArg) {
  IRRewriter rewriter(context);
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing " << op->getName() << "\n" << op << "\n";
  });
  if (failed(parseChecked(op.getPtr(), op.getLoc(), rewriter, offsetMap)))
    return failure();
  isFromTensorArg(op.getPtr(), fromTensorArg);
  return success();
}

template <typename MemAccOpTy, typename>
void TritonToUnstructurePass::runParse(MemAccOpTy op) {
  if (failed(runParseChecked(op, &getContext(), offsetMap, fromTensorArg)))
    signalPassFailure();
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
  forceSimtTemplateFlag = this->forceSimtTemplate;

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "TritonToUnstructurePass started with options:\n";
    os << "  compileOn91095: " << compileOn91095Flag << "\n";
    os << "  forceSimtTemplate: " << forceSimtTemplateFlag << "\n";
  });

  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();
  if (failed(rejectByteAddressedControlFlow(moduleOp))) {
    signalPassFailure();
    return;
  }

  moduleOp->walk([this](triton::FuncOp funcOp) {
    replacePtrArguments(funcOp, offsetMapForLoopArgs);
  });
  offsetMapForLoopArgs.clear();

  if (failed(processIfYieldAddHoistOperations(moduleOp))) {
    moduleOp.emitWarning("Failed to process IfYieldAddHoist operations");
  }

  WalkResult preparseResult =
      moduleOp->walk([&](LoopLikeOpInterface op) -> WalkResult {
        if (failed(
                runPreparseChecked(op, ctx, offsetMap, offsetMapForLoopArgs)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (preparseResult.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  WalkResult parseResult = moduleOp->walk([&](Operation *op) -> WalkResult {
    LogicalResult result = success();
    if (auto loadOp = dyn_cast<triton::LoadOp>(op))
      result = runParseChecked(loadOp, ctx, offsetMap, fromTensorArg);
    else if (auto storeOp = dyn_cast<triton::StoreOp>(op))
      result = runParseChecked(storeOp, ctx, offsetMap, fromTensorArg);
    else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op))
      result = runParseChecked(atomicRMWOp, ctx, offsetMap, fromTensorArg);
    else if (auto atomicCASOp = dyn_cast<triton::AtomicCASOp>(op))
      result = runParseChecked(atomicCASOp, ctx, offsetMap, fromTensorArg);
    return failed(result) ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (parseResult.wasInterrupted()) {
    signalPassFailure();
    return;
  }
  // A rewrite-pattern failure only means "not matched" to the greedy driver;
  // diagnostics emitted from matchAndRewrite do not make the pass fail. Check
  // this structural requirement after parsing so an unsupported tensor-root
  // access cannot survive the pass while triton-opt still exits successfully.
  WalkResult memoryAccessValidation =
      moduleOp->walk([&](Operation *op) -> WalkResult {
        Value ptr;
        if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
          ptr = loadOp.getPtr();
        } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
          ptr = storeOp.getPtr();
        } else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)) {
          ptr = atomicRMWOp.getPtr();
        } else if (auto atomicCASOp = dyn_cast<triton::AtomicCASOp>(op)) {
          ptr = atomicCASOp.getPtr();
        } else {
          return WalkResult::advance();
        }

        if (!resolvePtrTensorType(ptr) ||
            op->hasAttr(ConverterUtils::discreteAttrName))
          return WalkResult::advance();
        auto ptrInfo = offsetMap.find(ptr);
        // This validation was added for the different-width bitcast path. Do
        // not turn missing information in a legacy pointer analysis into a new
        // non-bitcast compilation failure.
        if (ptrInfo == offsetMap.end() || !ptrInfo->second.isByteAddressed() ||
            ptrInfo->second.getPtr())
          return WalkResult::advance();

        op->emitError(
            "unstructured pointer access requires a scalar root pointer");
        return WalkResult::interrupt();
      });
  if (memoryAccessValidation.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  RewritePatternSet patterns(ctx);

  patterns.add<UnstructuredMemAccessConverter<triton::LoadOp>,
               UnstructuredMemAccessConverter<triton::StoreOp>,
               UnstructuredMemAccessConverter<triton::AtomicRMWOp>,
               UnstructuredMemAccessConverter<triton::AtomicCASOp>>(
      ctx, forceScalarizeMode, offsetMap, fromTensorArg);

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing done\n";
  });

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Patterns");
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
