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

#include "ascend/include/TritonToLinalg/TritonOpConverter.h"
#include "ascend/include/TritonToLinalg/BlockPtrAnalysis.h"
#include "ascend/include/TritonToLinalg/MaskAnalysis.h"
#include "ascend/include/TritonToLinalg/TritonToLinalgPass.h"
#include "ascend/include/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

static const llvm::SmallVector<llvm::StringRef> libdeviceOps = {
    // Basic operations
    "__hmf_div_rz_fp32",
    "__hmf_fmod_fp32",
    "__hmf_float_as_int_fp32",
    "__hmf_trunc_fp32",
    "__hmf_trunc_fp16",
    "__hmf_nearbyint_fp32",
    "__hmf_signbit_fp32",
    "__hmf_signbit_fp16",
    "__hmf_copysign_fp32",
    "__hmf_log10_fp32",
    // Trigonometric operations
    "__hmf_asin_fp32",
    "__hmf_asin_fp16",
    "__hmf_acos_fp32",
    "__hmf_acos_fp16",
    "__hmf_atan2_fp32",
    "__hmf_atan2_fp16",
    "__hmf_sinh_fp32",
    "__hmf_sinh_fp16",
    "__hmf_cosh_fp32",
    "__hmf_cosh_fp16",
    "__hmf_asinh_fp32",
    "__hmf_asinh_fp16",
    "__hmf_acosh_fp32",
    "__hmf_acosh_fp16",
    "__hmf_atanh_fp32",
    "__hmf_atanh_fp16",
    // Other operations
    "__hmf_expm1_fp32",
    "__hmf_expm1_fp16",
    "__hmf_nextafter_fp32",
    "__hmf_nextafter_fp16",
    "__hmf_hypot_fp32",
    "__hmf_hypot_fp16",
    "__hmf_cyl_bessel_i0_fp32",
    "__hmf_cyl_bessel_i0_fp16",
    "__hmf_erfinv_fp32",
    "__hmf_lgamma_fp32",
};

/**
 * Retrieves a boolean environment variable.
 * @param envVar The name of the environment variable.
 * @param defaultValue The default value to return if the variable is not set or cannot be parsed.
 * @return true if the environment variable exists and its value is parsed as "true", otherwise returns defaultValue.
 * Parsing rules (case-insensitive): "true" values: any non-empty string not equal to "0", "false", "no", "off" is
 * considered true. "false" values: an empty string or a string equal to any of the false literals is considered false.
 */
bool getEnvBool(const char *envVar, bool defaultValue)
{
    const char *val = std::getenv(envVar);
    if (val == nullptr) {
        return defaultValue; // variable not set
    }

    std::string s(val);
    // Convert to lowercase for easier comparison
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });

    // Common false literals
    if (s.empty() || s == "0" || s == "false" || s == "no" || s == "off") {
        return false;
    }
    // All other cases (including "1", "true", "yes", "on", etc.) are considered true
    return true;
}

static llvm::SmallString<kFuncNameCap> generateUniqueFuncName(ModuleOp moduleOp, llvm::StringRef funcNameBase)
{
    llvm::SmallString<kFuncNameCap> funcName = funcNameBase;
    int uniqueId = 0;
    while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
        funcName = funcNameBase;
        funcName += ("_" + std::to_string(uniqueId++));
    }
    return funcName;
}

LogicalResult BitcastConverter::matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                                                ConversionPatternRewriter &rewriter) const
{
    Value result;
    auto loc = op.getLoc();

    if (auto dstPtrTy = dyn_cast<triton::PointerType>(op.getType())) {
        auto srcPtrTy = cast<triton::PointerType>(op.getSrc().getType());
        auto resType = MemRefType::get({ShapedType::kDynamic}, dstPtrTy.getPointeeType());

        auto i1Ty = rewriter.getIntegerType(1);
        auto i8Ty = rewriter.getIntegerType(8);
        bool isI1toI8 = (srcPtrTy.getPointeeType() == i1Ty) && (dstPtrTy.getPointeeType() == i8Ty);
        // handling special case: ptr<i1> -> ptr<i8>, directly forward without arith.bitcast
        if (isI1toI8) {
            // TypeConverter has already converted i1 to i8 memref,
            LLVM_DEBUG({
                llvm::dbgs() << "[BitcastConverter] Special i1->i8 pointer bitcast. Forward "
                                "without arith.bitcast. srcConvertedTy="
                             << adaptor.getSrc().getType() << "\n";
            });
            rewriter.replaceOp(op, adaptor.getSrc());
            return success();
        }
        result = rewriter.create<arith::BitcastOp>(loc, resType, adaptor.getSrc());
    } else {
        // handling normal case: bitcast between tensors/memrefs
        result = rewriter.create<arith::BitcastOp>(loc, op.getType(), adaptor.getSrc());
    }
    rewriter.replaceOp(op, result);
    return success();
}

LogicalResult TransposeConverter::matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                                                  ConversionPatternRewriter &rewriter) const
{
    auto src = adaptor.getSrc();
    auto res = ConverterUtils::getTransposedValue(src, op.getLoc(), rewriter, op.getOrder());
    rewriter.replaceOp(op, res);
    return success();
}

LogicalResult YieldConverter::matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                              ConversionPatternRewriter &rewriter) const
{
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
}

LogicalResult AdvanceConverter::matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                                                ConversionPatternRewriter &rewriter) const
{
    llvm::SmallDenseMap<Value, BlockData> known;
    BlockDataParser::rewriteAdvanceOp(op, rewriter, known);
    return success();
}

// ToDo:
// 1. Refactor MakeTensorPtrConverter and AdvanceConverter with
// memref::ReinterpretCastOp and memref::SubViewOp.
// Use recast to describe full shape of tensor, and use subview to represent
// current block tensor.
LogicalResult MakeTensorPtrConverter::matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                                                      ConversionPatternRewriter &rewriter) const
{
    llvm::SmallDenseMap<Value, BlockData> known;
    BlockDataParser::rewriteMakeTensorPtrOp(op, adaptor.getBase(), rewriter, known);
    return success();
}

LogicalResult PreciseDivConverter::matchAndRewrite(triton::PreciseDivFOp op, OpAdaptor adaptor,
                                                   ConversionPatternRewriter &rewriter) const
{
    Value opa = op.getX();
    Value opb = op.getY();
    auto loc = op.getLoc();

    auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto divOp = rewriter.create<arith::DivFOp>(loc, resType, opa, opb);

    rewriter.replaceOp(op, divOp);
    return success();
}

LogicalResult SelectCanonicalizer::matchAndRewrite(arith::SelectOp op, PatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    // 0. Shortcut for scalars
    auto type = dyn_cast<TensorType>(op.getResult().getType());
    if (!type) {
        // do nothing non-tensor select
        return failure();
    }
    auto tensorShape = type.getShape();
    auto mask = op.getCondition();
    if (!isa<ShapedType>(mask.getType())) {
        // do nothing for scalar mask
        return failure();
    }

    // 1. Check for continuous masked loads.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed()) {
        mstate.eraseInsertedOps(op, rewriter);
        return rewriter.notifyMatchFailure(op, "Cannot lower continuous masked selects");
    }

    // 2. Get mask position
    MaskPosition maskPos = mstate.getMaskPosition(tensorShape);
    LLVM_DEBUG({
        llvm::dbgs() << "[SelectAnalysis] MaskPosition detected: "
                     << (maskPos == MaskPosition::Head     ? "Head"
                         : maskPos == MaskPosition::Tail   ? "Tail"
                         : maskPos == MaskPosition::Middle ? "Middle"
                                                           : "Unknown")
                     << "\n";
    });

    if (maskPos == MaskPosition::Unknown) {
        return failure();
    }
    auto trueTensor = op.getTrueValue();
    auto falseTensor = op.getFalseValue();
    tensor::ExtractSliceOp extractSliceOp;
    tensor::InsertSliceOp insertSliceOp;

    // 3. Slice and insert out the masked part
    if (maskPos == MaskPosition::Head) {
        // Slice out the masked part of true tensor
        extractSliceOp = mstate.getExtractSlice(trueTensor, loc, rewriter);

        // Insert out the sliced true tensor into false tensor
        insertSliceOp = mstate.getInsertSlice(extractSliceOp, falseTensor, loc, rewriter);
    } else { // maskPos == MaskPosition::Tail or maskPos == MaskPosition::Middle
        SmallVector<OpFoldResult> invertOffsets;
        SmallVector<OpFoldResult> invertFalseDims;
        SmallVector<OpFoldResult> invertTrueDims;
        Value falseDimVal;
        OpFoldResult trueDimOp;
        size_t valDim = -1;
        for (size_t i = 0; i < mstate.getRank(); ++i) {
            const auto &offVal = mstate.offsets[i];
            const auto &dimVal = mstate.dims[i];
            auto constOffVal = getConstantIntValue(offVal);
            invertOffsets.push_back(rewriter.getIndexAttr(0));
            if (constOffVal.has_value() && constOffVal.value() == 0) {
                invertFalseDims.push_back(dimVal);
                invertTrueDims.push_back(dimVal);
            } else {
                valDim = i;
                falseDimVal = offVal.get<Value>();
                invertFalseDims.push_back(offVal);
                trueDimOp = addOpFoldResult(offVal, dimVal, loc, rewriter);
                invertTrueDims.push_back(trueDimOp);
            }
        }

        // Slice out the invert first masked part of false tensor
        auto falseExtractSliceOp = mstate.getExtractSlice(falseTensor, loc, rewriter, invertOffsets, invertFalseDims);
        // Insert out the sliced false tensor into true tensor
        auto trueInsertSliceOp =
            mstate.getInsertSlice(falseExtractSliceOp, trueTensor, loc, rewriter, invertOffsets, invertFalseDims);

        if (valDim != -1) {
            rewriter.setInsertionPointAfter(trueInsertSliceOp);
            Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            Value isNegative = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, falseDimVal, zeroIndex);

            Value sizeIndex = rewriter.create<arith::ConstantIndexOp>(loc, tensorShape[valDim]);
            Value isOutOfRange = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, falseDimVal, sizeIndex);
            auto orOp = rewriter.create<arith::OrIOp>(loc, isNegative, isOutOfRange);
            auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange {type}, orOp.getResult(), true, true);

            Block *thenBlock = &ifOp.getThenRegion().front();
            rewriter.setInsertionPointToStart(thenBlock);
            rewriter.create<scf::YieldOp>(loc, ValueRange {trueTensor});

            Block *elseBlock = &ifOp.getElseRegion().front();
            falseExtractSliceOp->moveBefore(elseBlock, elseBlock->begin());
            trueInsertSliceOp->moveBefore(elseBlock, elseBlock->end());
            rewriter.setInsertionPointToStart(elseBlock);
            {
                rewriter.setInsertionPointAfter(trueInsertSliceOp);
                rewriter.create<scf::YieldOp>(loc, ValueRange {trueInsertSliceOp.getResult()});
            }

            // Slice out the invert first masked and masked part of ifOp result
            rewriter.setInsertionPointAfter(ifOp);
            extractSliceOp = mstate.getExtractSlice(ifOp.getResult(0), loc, rewriter, invertOffsets, invertTrueDims);
        } else {
            // Slice out the invert first masked and masked part of inserted true tensor
            extractSliceOp = mstate.getExtractSlice(trueInsertSliceOp, loc, rewriter, invertOffsets, invertTrueDims);
        }
        // Insert out the sliced true tensor into false tensor
        insertSliceOp =
            mstate.getInsertSlice(extractSliceOp, falseTensor, loc, rewriter, invertOffsets, invertTrueDims);
        LLVM_DEBUG({
            llvm::dbgs() << "  -> [invert] Created false tensor extractSlice: " << *falseExtractSliceOp.getOperation()
                         << "\n"
                         << "  -> [invert] Created true tensor insertSlice: " << *trueInsertSliceOp.getOperation()
                         << "\n";
        });
    }
    LLVM_DEBUG({
        llvm::dbgs() << "  -> Created ExtractSlice: " << *extractSliceOp.getOperation() << "\n"
                     << "  -> Created InsertSlice: " << *insertSliceOp.getOperation() << "\n";
    });
    rewriter.replaceOp(op, insertSliceOp);
    return success();

    // 4. Fix if the offset is negative at runtime
    rewriter.setInsertionPointAfter(insertSliceOp);
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto offsets = mstate.offsets;
    SmallVector<Value> isInvalidVals;
    for (size_t i = 0; i < offsets.size(); i++) {
        auto &o = offsets[i];
        if (o.is<Value>()) {
            auto oVal = o.get<Value>();
            int64_t dimSize = tensorShape[i];
            Value sizeIndex = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);
            Value isNegative = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, oVal, zeroIndex);
            Value isOutOfRange = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, oVal, sizeIndex);
            isInvalidVals.push_back(isNegative);
            isInvalidVals.push_back(isOutOfRange);
        }
    }

    if (isInvalidVals.empty()) {
        rewriter.replaceOp(op, insertSliceOp);
        return success();
    }
    // At least one value
    Value invalidVal = isInvalidVals[0];
    if (isInvalidVals.size() > 1) {
        for (int i = 1; i < isInvalidVals.size(); ++i) {
            auto tmpOrOp = rewriter.create<arith::OrIOp>(loc, isInvalidVals[i], invalidVal);
            invalidVal = tmpOrOp.getResult();
        }
    }
    // else: what if the number of negative value checks is > 1?
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange {falseTensor.getType()}, invalidVal, true /* addThenBlock */,
                                           true /* addElseBlock */);
    // thenBuilder
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange {falseTensor});
    // elseBuilder
    Block *elseBlock = &ifOp.getElseRegion().front();
    extractSliceOp->moveBefore(elseBlock, elseBlock->begin());
    insertSliceOp->moveBefore(elseBlock, elseBlock->end());
    rewriter.setInsertionPointToStart(elseBlock);
    {
        rewriter.setInsertionPointAfter(insertSliceOp);
        rewriter.create<scf::YieldOp>(loc, ValueRange {insertSliceOp.getResult()});
    }

    rewriter.replaceOp(op, ifOp);

    return success();
}

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly applied
 * on function arguments
 */
LogicalResult BitcastCanonicalizer::matchAndRewrite(triton::BitcastOp bitcastOp, PatternRewriter &rewriter) const
{
    Value castSrc = bitcastOp.getSrc();
    Value castRes = bitcastOp.getResult();
    Type castSrcTy = castSrc.getType();
    Type castSrcPtrTy = isa<ShapedType>(castSrcTy) ? cast<ShapedType>(castSrcTy).getElementType() : castSrcTy;
    if (!isa<triton::PointerType>(castSrcPtrTy))
        return failure();

    auto origBitwidth = getPointeeBitWidth(castSrc.getType());
    auto castBitwidth = getPointeeBitWidth(castRes.getType());

    if (origBitwidth == 1)
        origBitwidth = 8;
    if (castBitwidth == 1)
        castBitwidth = 8;
    if (origBitwidth != castBitwidth) {
        bitcastOp.emitError() << "Casting pointers with unmatched bitwidth!\n";
        return failure();
    }

    Operation *beforeCastOp = castSrc.getDefiningOp();
    if (beforeCastOp == nullptr) {
        return failure();
    }

    auto newRes =
        TypeSwitch<Operation *, FailureOr<Operation *>>(beforeCastOp)
            // before: addptr - bitcast - load/store
            // after: bitcast - addptr - load/store
            .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptrOp) {
                auto newCastOp =
                    rewriter.create<triton::BitcastOp>(bitcastOp.getLoc(), castRes.getType(), addptrOp.getPtr());
                return rewriter.create<triton::AddPtrOp>(bitcastOp.getLoc(), castRes.getType(), newCastOp.getResult(),
                                                         addptrOp.getOffset());
            })
            .Case<triton::SplatOp>([&](triton::SplatOp splatOp) {
                Type newCastSrcTy = cast<RankedTensorType>(castRes.getType()).getElementType();

                Value splatSrc = splatOp.getSrc();
                Type splatSrcTy = splatSrc.getType();
                if (auto splatSrcTensorTy = dyn_cast<RankedTensorType>(splatSrcTy))
                    newCastSrcTy = splatSrcTensorTy.cloneWith(std::nullopt, newCastSrcTy);
                auto newCastOp = rewriter.create<triton::BitcastOp>(bitcastOp.getLoc(), newCastSrcTy, splatSrc);
                return rewriter.create<triton::SplatOp>(bitcastOp.getLoc(), castRes.getType(), newCastOp);
            })
            // before: bitcast - bitcast
            // after(fusion optimization): bitcast
            .Case<triton::BitcastOp>([&](triton::BitcastOp prevCastOp) {
                return rewriter.create<triton::BitcastOp>(bitcastOp.getLoc(), castRes.getType(), prevCastOp.getSrc());
            })
            .Default([&](Operation *op) { return rewriter.notifyMatchFailure(bitcastOp, "Unknown bitcast pattern"); });
    if (succeeded(newRes)) {
        rewriter.replaceOp(bitcastOp, newRes.value());
        if (beforeCastOp->use_empty()) {
            rewriter.eraseOp(beforeCastOp);
        }
        return success();
    }
    return failure();
}

LogicalResult FpToFpCanonicalizer::matchAndRewrite(triton::FpToFpOp op, PatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value input = op.getSrc();
    auto resultType = op.getResult().getType();

    // Check if rounding mode is specified
    auto roundingMode = op.getRounding();
    if (roundingMode.has_value() && roundingMode.value() != triton::RoundingMode::RTNE) {
        // Non-RTNE rounding modes (e.g., RTZ) should be handled by TritonToHFusion pass
        // Return failure here so this pattern doesn't match
        return failure();
    }

    // Handle RTNE (default) rounding mode with arith.truncf/extf
    auto srcType = cast<RankedTensorType>(input.getType());
    auto dstType = cast<RankedTensorType>(resultType);
    auto srcElemType = srcType.getElementType();
    auto dstElemType = dstType.getElementType();
    if (!isa<FloatType>(srcElemType) || !isa<FloatType>(dstElemType)) {
        return op.emitError("FpToFp expects floating point types");
    }

    unsigned srcBitwidth = srcElemType.getIntOrFloatBitWidth();
    unsigned dstBitwidth = dstElemType.getIntOrFloatBitWidth();

    // Create round_mode attribute (RINT for RTNE)
    auto roundModeAttr = hfusion::RoundModeAttr::get(rewriter.getContext(), hfusion::RoundMode::RINT);

    if (srcBitwidth > dstBitwidth) {
        // Downcast: use arith.truncf with round_mode=rint
        auto truncOp = rewriter.create<arith::TruncFOp>(loc, dstType, input);
        truncOp->setAttr("round_mode", roundModeAttr);
        rewriter.replaceOp(op, truncOp.getResult());
    } else if (srcBitwidth < dstBitwidth) {
        // Upcast: use arith.extf with round_mode=rint
        auto extOp = rewriter.create<arith::ExtFOp>(loc, dstType, input);
        extOp->setAttr("round_mode", roundModeAttr);
        rewriter.replaceOp(op, extOp.getResult());
    } else {
        // Same bitwidth, should not happen but handle gracefully
        rewriter.replaceOp(op, input);
    }

    return success();
}

void rewriteUserWithNewOrder(mlir::OpOperand *use, PatternRewriter &rewriter,
                             llvm::SmallVector<int64_t, 8> &blkShapeI64, // 8: container size
                             mlir::Location &loc, llvm::ArrayRef<int32_t> &order, size_t &orderSize)
{
    Operation *user = use->getOwner();
    rewriter.setInsertionPointAfter(user);
    if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
        auto loadResTy = loadOp.getResult().getType();
        auto loadResShapedTy = cast<ShapedType>(loadResTy);
        auto newLoadTy = loadResShapedTy.cloneWith(blkShapeI64, loadResShapedTy.getElementType());
        auto newLoadOp = rewriter.create<triton::LoadOp>(loc, newLoadTy, loadOp->getOperands(), loadOp->getAttrs());
        newLoadOp->setAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG, UnitAttr::get(rewriter.getContext()));
        rewriter.replaceOp(loadOp, newLoadOp);
        // load contiguous data then permute. thus the permute order is as
        // follows.
        SmallVector<int32_t, 8> permuteOrder; // 8: container size
        for (auto [i, v] : llvm::enumerate(order)) {
            permuteOrder.push_back(orderSize - 1 - order[i]);
        }
        auto permuteOp = rewriter.create<triton::TransOp>(loc, newLoadOp.getResult(),
                                                          DenseI32ArrayAttr::get(loadOp.getContext(), permuteOrder));
        newLoadOp.getResult().replaceAllUsesExcept(permuteOp.getResult(), permuteOp);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
        // permute to contiguous then store. thus the permute order is as follows.
        SmallVector<int32_t, 8> permuteOrder; // 8: container size
        for (auto [i, v] : llvm::enumerate(order)) {
            permuteOrder.push_back(order[orderSize - 1 - i]);
        }
        auto permuteOp = rewriter.create<triton::TransOp>(loc, storeOp.getValue(),
                                                          DenseI32ArrayAttr::get(storeOp.getContext(), permuteOrder));
        storeOp.getValue().replaceAllUsesExcept(permuteOp.getResult(), permuteOp);
        auto newStoreOp =
            rewriter.create<triton::StoreOp>(loc, storeOp.getPtr(), storeOp.getValue(), storeOp.getMask(),
                                             storeOp.getBoundaryCheck(), storeOp.getCache(), storeOp.getEvict());
        rewriter.replaceOp(storeOp, newStoreOp);
    } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
        auto advanceResPtrTy = cast<triton::PointerType>(advanceOp.getResult().getType());
        auto advanceResShapedTy = cast<ShapedType>(advanceResPtrTy.getPointeeType());
        auto newAdvanceResShapedTy = advanceResShapedTy.cloneWith(blkShapeI64, advanceResShapedTy.getElementType());
        auto newAdvanceResPtrTy = triton::PointerType::get(newAdvanceResShapedTy, advanceResPtrTy.getAddressSpace());
        auto advanceOffsets = advanceOp.getOffsets();
        llvm::SmallVector<Value, 8> newAdvanceOffsets; // 8: container size
        for (int i = orderSize - 1; i >= 0; i--) {
            newAdvanceOffsets.push_back(advanceOffsets[order[i]]);
        }
        SmallVector<OpOperand *> resUses;
        for (auto &use : advanceOp->getUses())
            resUses.push_back(&use);
        auto newAdvanceOp =
            rewriter.create<triton::AdvanceOp>(loc, newAdvanceResPtrTy, advanceOp.getPtr(), newAdvanceOffsets);
        rewriter.replaceOp(advanceOp, newAdvanceOp);
        for (auto resUse : resUses)
            rewriteUserWithNewOrder(resUse, rewriter, blkShapeI64, loc, order, orderSize);
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
        auto initArg = use->get();
        auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
        auto resultValue = loopOp.getTiedLoopResult(use);
        iterArg.setType(initArg.getType());
        resultValue.setType(initArg.getType());
        for (auto &argUse : iterArg.getUses())
            rewriteUserWithNewOrder(&argUse, rewriter, blkShapeI64, loc, order, orderSize);
        for (auto &resUse : resultValue.getUses())
            rewriteUserWithNewOrder(&resUse, rewriter, blkShapeI64, loc, order, orderSize);
    } else if (isa<scf::YieldOp>(user)) {
        return;
    } else {
        llvm_unreachable("[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
                         "not used by load/store/advance op");
    }
}

void markLoadUsers(mlir::OpOperand *use, PatternRewriter &rewriter)
{
    Operation *user = use->getOwner();
    if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
        loadOp->setAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG, UnitAttr::get(rewriter.getContext()));
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
        return;
    } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
        SmallVector<OpOperand *> resUses;
        for (auto &use : advanceOp->getUses())
            resUses.push_back(&use);
        for (auto resUse : resUses)
            markLoadUsers(resUse, rewriter);
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
        auto initArg = use->get();
        auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
        auto resultValue = loopOp.getTiedLoopResult(use);
        iterArg.setType(initArg.getType());
        resultValue.setType(initArg.getType());
        for (auto &argUse : iterArg.getUses())
            markLoadUsers(&argUse, rewriter);
        for (auto &resUse : resultValue.getUses())
            markLoadUsers(&resUse, rewriter);
    } else if (isa<scf::YieldOp>(user)) {
        return;
    } else {
        llvm_unreachable("[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
                         "not used by load/store/advance op");
    }
}

LogicalResult MakeTensorPtrCanonicalizer::matchAndRewrite(triton::MakeTensorPtrOp op, PatternRewriter &rewriter) const
{
    auto order = op.getOrder();
    auto orderSize = order.size();
    if (orderSize == 1) {
        return rewriter.notifyMatchFailure(op, "make_tensor_ptr's order has single value.");
    }

    bool isPermuted = false;
    for (auto [first, second] : llvm::zip(order.slice(0, orderSize - 1), order.slice(1, orderSize - 1))) {
        if (first != second + 1) {
            isPermuted = true;
            break;
        }
    }

    auto loc = op.getLoc();
    auto base = op.getBase();
    auto shape = op.getShape();
    auto strides = op.getStrides();
    auto offsets = op.getOffsets();
    auto result = op.getResult();
    SmallVector<OpOperand *> opUses;

    for (auto &use : result.getUses())
        opUses.push_back(&use);
    for (auto use : opUses)
        markLoadUsers(use, rewriter);

    if (!isPermuted) {
        return rewriter.notifyMatchFailure(op, "make_tensor_ptr's order is contiguous.");
    }

    llvm::SmallVector<int32_t, 8> blkShapeI32;
    llvm::SmallVector<int64_t, 8> blkShapeI64;
    auto resPtrType = cast<triton::PointerType>(result.getType());
    if (auto resShapedTy = dyn_cast<ShapedType>(resPtrType.getPointeeType())) {
        auto resBlkShape = resShapedTy.getShape();
        for (auto [i, v] : llvm::enumerate(resBlkShape)) {
            auto reverseI = orderSize - 1 - i;
            blkShapeI32.push_back(resBlkShape[order[reverseI]]);
            blkShapeI64.push_back(resBlkShape[order[reverseI]]);
        }
    }

    llvm::SmallVector<Value, 8> newShape;
    llvm::SmallVector<Value, 8> newStrides;
    llvm::SmallVector<Value, 8> newOffsets;
    for (int i = orderSize - 1; i >= 0; i--) {
        newShape.push_back(shape[order[i]]);
        newStrides.push_back(strides[order[i]]);
        newOffsets.push_back(offsets[order[i]]);
    }

    llvm::SmallVector<int, 8> contiguousOrder;
    for (int i = orderSize - 1; i >= 0; i--)
        contiguousOrder.push_back(i);

    rewriter.setInsertionPoint(op);
    auto newMakeTensorPtrOp = rewriter.create<triton::MakeTensorPtrOp>(
        loc, base, ValueRange(newShape), ValueRange(newStrides), ValueRange(newOffsets), blkShapeI32, contiguousOrder);
    rewriter.replaceOp(op, newMakeTensorPtrOp);
    for (auto use : opUses)
        rewriteUserWithNewOrder(use, rewriter, blkShapeI64, loc, order, orderSize);
    return success();
}

LogicalResult ReduceSingleCanonicalizer::matchAndRewrite(triton::ReduceOp reduceOp, PatternRewriter &rewriter) const
{
    assert(reduceOp.getSrcs().size() <= 2 && "Only reduce or reduce with index are supported");
    auto src = reduceOp.getSrcs()[0];
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcShape = srcType.getShape();
    if (llvm::any_of(srcShape, [](auto s) { return s != 1; }))
        return rewriter.notifyMatchFailure(reduceOp, "reduce's srcs are not all with single element");
    auto loc = reduceOp->getLoc();

    // Handle Reduce Value
    auto res = reduceOp.getResult()[0];
    Value extracted;
    if (srcType.getRank() == 1) {
        auto zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
        extracted = rewriter.create<tensor::ExtractOp>(loc, src, zero.getResult()).getResult();
    } else {
        auto resShape = cast<RankedTensorType>(res.getType()).getShape();
        auto collapseReassociationIndicesOptional = getReassociationIndicesForCollapse(srcShape, resShape);
        if (!collapseReassociationIndicesOptional.has_value()) {
            return rewriter.notifyMatchFailure(reduceOp, "Failure with getReassociationIndicesForCollapse call");
        }
        auto collapseReassociationIndices = collapseReassociationIndicesOptional.value();
        extracted = rewriter.create<tensor::CollapseShapeOp>(loc, src, collapseReassociationIndices).getResult();
    }
    res.replaceAllUsesWith(extracted);

    // Handle Reduce Index
    if (reduceOp.getSrcs().size() == 1)
        return success();

    auto resIdx = reduceOp.getResult()[1];
    auto zeroI32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    if (srcType.getRank() == 1) {
        resIdx.replaceAllUsesWith(zeroI32);
    } else {
        auto resIdxShape = cast<RankedTensorType>(resIdx.getType()).getShape();
        auto initTensor = rewriter.create<tensor::EmptyOp>(loc, resIdxShape, rewriter.getI32Type());
        auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange {zeroI32}, ValueRange {initTensor});
        resIdx.replaceAllUsesWith(fillOp.getResult(0));
    }

    return success();
}

LogicalResult DenseConstantConverter::matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                                      ConversionPatternRewriter &rewriter) const
{
    auto denseAttr = cast<DenseElementsAttr>(op.getValue());
    auto loc = op.getLoc();
    auto constSplatOp =
        arith::ConstantOp::materialize(rewriter, denseAttr.getSplatValue<Attribute>(), denseAttr.getElementType(), loc);
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, cast<RankedTensorType>(op.getResult().getType()).getShape(),
                                                    denseAttr.getElementType());

    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange {constSplatOp}, ValueRange {emptyOp});

    return success();
}

LogicalResult MakeRangeConverter::matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                                                  ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto type = cast<TensorType>(op.getResult().getType());
    auto shape = type.getShape();
    auto elementType = type.getElementType();
    auto context = op.getContext();

    assert(type.getShape().size() == 1 && isa<IntegerType>(type.getElementType()) &&
           type.getElementType().getIntOrFloatBitWidth() == 32 && "make range can only return 1D int32 tensor");

    SmallVector<AffineMap> indexingMaps {AffineMap::get(
        /* dimCount */ 1, /* symbolCount */ 0, {mlir::getAffineDimExpr(0, context)}, context)};

    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);

    auto nestedBody = [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value index = nestedBuilder.create<linalg::IndexOp>(loc, 0);
        Value res = nestedBuilder.create<arith::IndexCastOp>(loc, elementType, index);
        nestedBuilder.create<linalg::YieldOp>(loc, res);
    };

    auto linalgOp =
        rewriter.create<linalg::GenericOp>(loc, op->getResultTypes(), /* operands */ ValueRange {}, ValueRange {init},
                                           indexingMaps, ConverterUtils::getNParallelLoopsAttrs(1), nestedBody);

    linalgOp->setAttr("tt.from_make_range", mlir::UnitAttr::get(context));
    linalgOp->setAttr("tt.make_range_offset", mlir::IntegerAttr::get(mlir::IndexType::get(context), 0));
    linalgOp->setAttr("tt.make_range_size", mlir::IntegerAttr::get(mlir::IndexType::get(context), shape[0]));

    int32_t startVal = op.getStartAttr().getInt();
    if (startVal == 0) {
        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
    }

    // Apply start offset
    Value startScaler =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(startVal)));
    auto startInit = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    Value startTensor =
        rewriter.create<linalg::FillOp>(loc, ValueRange {startScaler}, ValueRange {startInit}).getResult(0);
    auto addOp = rewriter.create<arith::AddIOp>(loc, linalgOp->getResult(0), startTensor);
    rewriter.replaceOp(op, addOp);
    return success();
}

LogicalResult SplatConverter::matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                              ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto shape = op.getType().getShape();
    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, op.getType().getElementType());
    if (llvm::all_of(shape, [](int64_t dim) { return dim == 1; })) {
        SmallVector<Value> idx(shape.size(), rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
        rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getSrc(), init, idx);
    } else {
        rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange {adaptor.getSrc()}, ValueRange {init});
    }
    return success();
}

LogicalResult ReshapeConverter::matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                                                ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto dst = op.getResult();
    Value shape =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64TensorAttr(cast<ShapedType>(dst.getType()).getShape()));
    auto reshapeOp = rewriter.create<tensor::ReshapeOp>(loc, dst.getType(), src, shape);
    rewriter.replaceOp(op, reshapeOp.getResult());
    return success();
}

LogicalResult ExpandDimsConverter::matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                                                   ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto resShape = cast<ShapedType>(op.getResult().getType()).getShape();
    auto axis = op.getAxis();

    SmallVector<ReassociationIndices> reassociation;

    auto src_last_dim = resShape.size() - 2;
    auto map_func = [&](unsigned i) -> ReassociationIndices {
        if (i < axis) {
            return i == src_last_dim ? ReassociationIndices {i, i + 1} : ReassociationIndices {i};
        }
        return i == axis ? ReassociationIndices {i, i + 1} : ReassociationIndices {i + 1};
    };

    reassociation = llvm::to_vector(llvm::map_range(llvm::seq<unsigned>(0, src_last_dim + 1), map_func));

    auto expandShapeOp =
        rewriter.create<tensor::ExpandShapeOp>(op.getLoc(), op.getResult().getType(), src, reassociation);
    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
}

LogicalResult ClampFConverter::matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                                               ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto input = adaptor.getX();
    auto min_para = adaptor.getMin();
    auto max_para = adaptor.getMax();
    auto propagateNan_para = adaptor.getPropagateNan();

    if (auto input_type = dyn_cast<RankedTensorType>(input.getType())) {
        if (isa<FloatType>(min_para.getType())) {
            auto minEmptyTensor =
                rewriter.create<tensor::EmptyOp>(loc, input_type.getShape(), input_type.getElementType());
            min_para =
                rewriter.create<linalg::FillOp>(loc, ValueRange {min_para}, ValueRange {minEmptyTensor}).result();
        }
        if (isa<FloatType>(max_para.getType())) {
            auto maxEmptyTensor =
                rewriter.create<tensor::EmptyOp>(loc, input_type.getShape(), input_type.getElementType());
            max_para =
                rewriter.create<linalg::FillOp>(loc, ValueRange {max_para}, ValueRange {maxEmptyTensor}).result();
        }
    }

    if (propagateNan_para == PropagateNan::NONE) {
        auto minOp = rewriter.create<arith::MinNumFOp>(loc, input, max_para);
        auto maxOp = rewriter.create<arith::MaxNumFOp>(loc, min_para, minOp);
        rewriter.replaceOp(op, ValueRange {maxOp});
    } else if (propagateNan_para == PropagateNan::ALL) {
        auto minOp = rewriter.create<arith::MinimumFOp>(loc, input, max_para);
        auto maxOp = rewriter.create<arith::MaximumFOp>(loc, min_para, minOp);
        rewriter.replaceOp(op, ValueRange {maxOp});
    } else {
        return failure();
    }

    return success();
}

// Here convert tt.broadcast to linalg.broadcast
//
// before
// %out = tt.broadcast %in : tensor<1x4x8xf32> -> tensor<128x4x8xf32>
//
// after
// %collpased = tensor.collapse_shape %in [[0, 1], [2]] :
//                                    tensor<1x4x8xf32> into tensor<4x8xf32>
// %out = linalg.broadcast ins(%collpased : tensor<4x8xf32>)
//                         outs(%empty : tensor<128x4x8xf32>) dimensions = [0]
LogicalResult BroadcastConverter::matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                                                  ConversionPatternRewriter &rewriter) const
{
    assert(op->getNumResults() == 1 && "BroadcastOp assumes single result");

    RankedTensorType sourceType = cast<RankedTensorType>(adaptor.getSrc().getType());
    RankedTensorType resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();
    auto loc = op.getLoc();

    auto initEmpty = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elementType);

    SmallVector<int64_t> broadcastDims = ConverterUtils::getBroadcastDims(sourceType, resultType);
    SmallVector<int64_t> unbroadcastDims = ConverterUtils::getUnbroadcastDims(sourceType, resultType);

    SmallVector<ReassociationIndices> collapseReassociationIndices;
    auto collapseReassociationIndicesOptional =
        getReassociationIndicesForCollapse(sourceType.getShape(), unbroadcastDims);
    if (!collapseReassociationIndicesOptional.has_value()) {
        return rewriter.notifyMatchFailure(op, "Failure with getReassociationIndicesForCollapse call");
    }
    collapseReassociationIndices = collapseReassociationIndicesOptional.value();

    RankedTensorType collapseResultType = RankedTensorType::get(unbroadcastDims, sourceType.getElementType());

    auto collpasedOp = rewriter.create<tensor::CollapseShapeOp>(loc, collapseResultType, adaptor.getSrc(),
                                                                collapseReassociationIndices);

    auto broadcastOp =
        rewriter.create<linalg::BroadcastOp>(loc, collpasedOp, initEmpty, rewriter.getDenseI64ArrayAttr(broadcastDims));

    rewriter.replaceOp(op, broadcastOp.getResults());
    return success();
}

// Reduce Converter
bool ReduceConverter::isReductionOpSupported(Operation *redOp) const
{
    return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp, arith::MaximumFOp, arith::MaxNumFOp,
               arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
               arith::AndIOp, arith::OrIOp, arith::XOrIOp>(redOp);
}

bool ReduceConverter::isMultiReductionOpSupported(Operation *redOp)
{
    return isa<arith::SubFOp, arith::SubIOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp, arith::RemFOp,
               arith::RemSIOp, arith::RemUIOp>(redOp);
}

Value ReduceConverter::cloneReduceOps(OpBuilder &builder, Value in, Value out, Value opIns, Value opOuts,
                                      triton::ReduceOp op) const
{
    auto &reg = op->getRegion(0);
    assert(reg.getBlocks().size() == 1);
    auto &body = reg.getBlocks().front();
    auto numArguments = 2;
    assert(body.getNumArguments() == numArguments);

    Value ttIn = body.getArgument(0);
    Value ttOut = body.getArgument(1);

    IRMapping mapping;
    mapping.map(ttIn, in);
    mapping.map(ttOut, out);

    for (auto &op : body.without_terminator()) {
        builder.clone(op, mapping);
    }
    auto yield = cast<triton::ReduceReturnOp>(body.getTerminator());
    return mapping.lookup(yield->getOperand(0));
}

void ReduceConverter::checkIsNotCallOp(const llvm::SmallVector<Operation *> &reductionOps) const
{
    llvm::for_each(reductionOps, [](Operation *op) {
        assert(!isa<triton::CallOp>(op) &&
               "tt.call ops expected to be inlined in tt.reduce body in ttir building stage");
    });
}

bool ReduceConverter::isSCFOpReduce(const llvm::SmallVector<Operation *> &reductionOps) const
{
    return (reductionOps.size() == 1 &&
            reductionOps.front()->getDialect()->getNamespace() == scf::SCFDialect::getDialectNamespace());
}

bool ReduceConverter::isMultiOpReduce(const llvm::SmallVector<Operation *> &reductionOps) const
{
    this->checkIsNotCallOp(reductionOps);

    return (reductionOps.size() > 1) ||
           (reductionOps.size() == 1 && this->isMultiReductionOpSupported(reductionOps.front())) ||
           this->isSCFOpReduce(reductionOps);
}

Value ReduceConverter::computeReduceResultWithCompileFlag(OpBuilder &opBuilder, Location loc, Value lhs, Value rhs,
                                                          Value source, Value initTensor, triton::ReduceOp op,
                                                          bool compileOn91095Flag) const
{
    auto reductionOps = this->getReductionOps(op);
    assert(reductionOps.size() == 1);
    if (compileOn91095Flag) {
        return this->cloneReduceOps(opBuilder, lhs, rhs, source, initTensor, op);
    } else {
        auto rop = reductionOps.front();
        return this->getReductionElement(lhs, rhs, loc, rop, opBuilder, false);
    }
}

LogicalResult ReduceConverter::convertToTargetOp(triton::ReduceOp op, typename triton::ReduceOp::Adaptor adaptor,
                                                 ConversionPatternRewriter &rewriter) const
{
    auto source = adaptor.getOperands().front();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto elemType = sourceType.getElementType();
    auto resType = op.getResult().front().getType();
    auto loc = op.getLoc();
    auto reductionOps = this->getReductionOps(op);
    auto multiOpReduce = this->isMultiOpReduce(reductionOps);
    // Reduction of arbitrary operations isn't supported because using the first
    // element across the reduction dimension requires us to iterate over a
    // subview that skips over each first element.
    if (!multiOpReduce && !this->isReductionOpSupported(reductionOps.front())) {
        if (compileOn91095Flag) {
            llvm_unreachable("All reduction cases expected to be covered");
        }
        return rewriter.notifyMatchFailure(
            op, "Only support lowering reduction with single op and limited types of reduction");
    }

    auto rop = reductionOps.front();
    auto ropLoc = rop->getLoc();
    auto axis = op.getAxis();
    auto isVectorReduce = sourceType.getRank() == 1;

    auto constantType = elemType;

    auto accBaseConstOp = multiOpReduce ? this->getMultiOpReductionBaseConstOp(rewriter, op, ropLoc, constantType)
                                        : this->getReductionBaseConstOp(rewriter, rop, constantType);

    Value initTensor;

    if (isVectorReduce) {
        auto holder =
            rewriter.create<bufferization::AllocTensorOp>(loc, RankedTensorType::get({}, constantType), ValueRange {});
        initTensor = rewriter.create<linalg::FillOp>(loc, accBaseConstOp.getResult(), holder.getResult()).getResult(0);
    } else {
        Value init = rewriter.create<tensor::EmptyOp>(loc, cast<RankedTensorType>(resType).getShape(), constantType);
        initTensor = rewriter.create<linalg::FillOp>(loc, accBaseConstOp.getResult(), init).getResult(0);
    }

    Value finalResult =
        rewriter
            .create<linalg::ReduceOp>(loc, ValueRange {source}, ValueRange {initTensor}, SmallVector<int64_t> {axis},
                                      [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                                          assert(inputs.size() == 2);
                                          Value result = this->computeReduceResultWithCompileFlag(
                                              opBuilder, loc, inputs[0], inputs[1], source, initTensor, op,
                                              compileOn91095Flag);
                                          opBuilder.create<linalg::YieldOp>(loc, result);
                                      })
            .getResult(0);

    if (sourceType.getRank() == 1) {
        finalResult = rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
}

LogicalResult ReduceConverter::convertToTargetOpExtended(triton::ReduceOp op,
                                                         typename triton::ReduceOp::Adaptor adaptor,
                                                         ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto elemTypes = op.getElementTypes();

    auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
    const auto isScalarReduce = valueResultType == nullptr;

    SmallVector<Value> outputs;
    for (auto i = 0; i < op.getResult().size() && i < elemTypes.size(); i++) {
        auto result = dyn_cast<RankedTensorType>(op.getType(i));
        SmallVector<int64_t> resultShape {isScalarReduce ? SmallVector<int64_t> {}
                                                         : SmallVector<int64_t>(result.getShape())};
        outputs.push_back(rewriter.create<tensor::EmptyOp>(loc, resultShape, elemTypes[i]));
    }

    auto linalgOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getOperands(), outputs, SmallVector<int64_t> {adaptor.getAxis()},
        [&](OpBuilder &b, Location loc, ValueRange inputs) {
            auto tritonReduceBlock = op.getBody();
            IRMapping mapping;
            mapping.map(tritonReduceBlock->getArguments(), inputs);

            for (auto &op : tritonReduceBlock->without_terminator()) {
                b.clone(op, mapping);
            }

            auto tritonYield = tritonReduceBlock->getTerminator();
            auto results =
                llvm::map_to_vector(tritonYield->getOperands(), [&](Value val) { return mapping.lookup(val); });
            b.create<linalg::YieldOp>(loc, results);
        });

    auto params = getReduceWithIndexParams(op);
    if (failed(params)) {
        return rewriter.notifyMatchFailure(op, "meaningless reduce operation");
    } else if (params->withIndexType != ReduceWithIndexType::None) {
        addReduceWithIndexAttr(*params, rewriter, linalgOp);
    }

    if (isScalarReduce) {
        SmallVector<Value> reduceResults;
        for (auto i = 0; i < linalgOp.getResults().size() && i < elemTypes.size(); i++) {
            reduceResults.push_back(
                rewriter.create<tensor::ExtractOp>(loc, elemTypes[i], linalgOp.getResults()[i], ValueRange {}));
        }
        rewriter.replaceOp(op, reduceResults);
    } else {
        rewriter.replaceOp(op, linalgOp);
    }
    return success();
}

bool ScanConverter::isReductionOpSupported(Operation *reductionOp) const
{
    return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp>(reductionOp);
}

LogicalResult ScanConverter::convertToTargetOp(triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
                                               ConversionPatternRewriter &rewriter) const
{
    auto reductionOps = this->getReductionOps(op);
    if (reductionOps.empty()) {
        return rewriter.notifyMatchFailure(op, "No reduction op found in scan body");
    }

    llvm::SmallString<64> funcName;
    auto rop = reductionOps.front();
    if (this->isReductionOpSupported(reductionOps.front())) {
        if (isa<arith::AddFOp, arith::AddIOp>(rop)) {
            funcName = "triton_cumsum";
        } else if (isa<arith::MulFOp, arith::MulIOp>(rop)) {
            funcName = "triton_cumprod";
        }

        auto moduleOp = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));

        auto loc = op.getLoc();
        auto src = adaptor.getOperands().front();
        auto resTy = op.getResult().front().getType();
        auto libFnType =
            rewriter.getFunctionType({src.getType(), rewriter.getI32Type(), rewriter.getI1Type()}, {resTy});
        auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);

        SymbolTable symTab(moduleOp);
        auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
        if (failed(maybePrintFuncNameAttr)) {
            return op->emitError("failed to create a unique func name for device_print");
        }
        SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

        rewriter.setInsertionPoint(op);
        auto scanAxis = op.getAxis();
        auto scanReverse = op.getReverse();
        Value axis = rewriter.create<arith::ConstantIntOp>(loc, scanAxis, 32);
        Value reverseVal = rewriter.create<arith::ConstantIntOp>(loc, scanReverse, 1);
        auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({resTy}),
                                                    ValueRange({src, axis, reverseVal}));

        rewriter.replaceOp(op, callOp);

        return success();
    } else {
        // This branch is the associative_scan op.
        bool reverse = op.getReverse();

        auto loc = op.getLoc();

        Value scanInput = op.getOperand(0);

        auto srcType = mlir::dyn_cast<RankedTensorType>(scanInput.getType());
        if (!srcType) {
            return rewriter.notifyMatchFailure(op, "Expected RankedTensorType input for associative_scan");
        }

        auto elementType = srcType.getElementType();
        auto shape = srcType.getShape();
        int rank = shape.size();
        int axis = op.getAxis();

        if (axis < 0 || axis >= rank) {
            return rewriter.notifyMatchFailure(op, "Invalid scan axis: " + std::to_string(axis));
        }

        if (op->getNumRegions() < 1 || op->getRegion(0).empty()) {
            return rewriter.notifyMatchFailure(op, "Missing combine region");
        }

        OpBuilder::InsertionGuard guard(rewriter);

        auto memrefType = MemRefType::get(shape, elementType);
        Value inputMemRef = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, scanInput);
        Value outputMemRef = rewriter.create<memref::AllocOp>(loc, memrefType);

        auto processDimension = [&](ArrayRef<Value> baseIdxsArray) {
            auto startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
            if (reverse) {
                startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), shape[axis] - 1);
            }
            llvm::SmallVector<Value> baseIdxs(baseIdxsArray.begin(), baseIdxsArray.end());
            llvm::SmallVector<Value> firstIdx = baseIdxs;
            if (axis <= firstIdx.size()) {
                firstIdx.insert(firstIdx.begin() + axis, startInd);
            } else {
                firstIdx.push_back(startInd);
            }

            Value firstVal = rewriter.create<memref::LoadOp>(loc, inputMemRef, firstIdx);
            rewriter.create<memref::StoreOp>(loc, firstVal, outputMemRef, firstIdx);

            Value axisSize = rewriter.create<memref::DimOp>(loc, inputMemRef, axis).getResult();
            Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

            Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, axisSize, one);
            auto ifOp = rewriter.create<scf::IfOp>(loc, cmp, false);

            // Create a loop only when the axis size is greater than 1.
            rewriter.setInsertionPointToStart(ifOp.thenBlock());

            auto forOp = rewriter.create<scf::ForOp>(loc, one, axisSize, one);
            rewriter.setInsertionPointToStart(forOp.getBody());

            Value k = forOp.getInductionVar();
            if (reverse) {
                llvm::SmallVector<Value> fixInd;
                fixInd.push_back(rewriter.create<arith::ConstantIndexOp>(op.getLoc(), shape[axis] - 1).getResult());
                fixInd.push_back(k);
                auto fixIndVal = rewriter.create<arith::SubIOp>(op.getLoc(), fixInd);
                k = fixIndVal.getResult();
            }
            llvm::SmallVector<Value> currIdx = baseIdxs;
            if (axis <= currIdx.size()) {
                currIdx.insert(currIdx.begin() + axis, k);
            } else {
                currIdx.push_back(k);
            }

            Value km1 = rewriter.create<arith::SubIOp>(loc, k, one);
            if (reverse) {
                km1 = rewriter.create<arith::AddIOp>(loc, k, one);
            }
            llvm::SmallVector<Value> prevIdx = baseIdxs;
            if (axis <= prevIdx.size()) {
                prevIdx.insert(prevIdx.begin() + axis, km1);
            } else {
                prevIdx.push_back(km1);
            }

            Value currentVal = rewriter.create<memref::LoadOp>(loc, inputMemRef, currIdx);
            Value prevResult = rewriter.create<memref::LoadOp>(loc, outputMemRef, prevIdx);

            Region &combineRegion = op->getRegion(0);
            Block &combineBlock = combineRegion.front();
            IRMapping mapping;
            mapping.map(combineBlock.getArgument(0), prevResult);
            mapping.map(combineBlock.getArgument(1), currentVal);

            for (Operation &innerOp : combineBlock.without_terminator()) {
                rewriter.clone(innerOp, mapping);
            }

            Operation *yieldOp = combineBlock.getTerminator();
            Value resultVal = mapping.lookup(yieldOp->getOperand(0));

            rewriter.create<memref::StoreOp>(loc, resultVal, outputMemRef, currIdx);

            rewriter.setInsertionPointAfter(ifOp);
        };

        // Constructing loops for non-scanning dimensions
        llvm::SmallVector<int> nonScanDims;
        for (int i = 0; i < rank; ++i) {
            if (i != axis)
                nonScanDims.push_back(i);
        }

        createSimpleNestedLoops(rewriter, loc, outputMemRef, nonScanDims, processDimension);

        rewriter.setInsertionPointAfter(op);

        Value outputTensor = rewriter.create<bufferization::ToTensorOp>(loc, outputMemRef, true);
        rewriter.replaceOp(op, outputTensor);
        return success();
    }
}

LogicalResult ScanConverter::convertToTargetOpExtended(triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
                                                       ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    bool reverse = op.getReverse();

    // 1. Extract all input tensors (supports multiple inputs)
    auto operands = op->getOperands();
    if (operands.empty()) {
        return rewriter.notifyMatchFailure(op, "No input operands for extended scan");
    }

    // 2. Validate all inputs are of RankedTensorType
    llvm::SmallVector<RankedTensorType> inputTensTypes;
    for (auto operand : operands) {
        auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorTy) {
            return rewriter.notifyMatchFailure(op, "All inputs must be RankedTensorType");
        }
        inputTensTypes.push_back(tensorTy);
    }

    // 3. Validate all input tensors have the same shape (scan operation requires matching input dimensions)
    auto baseShape = inputTensTypes[0].getShape();
    int rank = baseShape.size();
    int axis = op.getAxis();
    if (axis < 0 || axis >= rank) {
        return rewriter.notifyMatchFailure(op, "Invalid scan axis: " + std::to_string(axis));
    }
    for (size_t i = 1; i < inputTensTypes.size(); ++i) {
        if (inputTensTypes[i].getShape() != baseShape) {
            return rewriter.notifyMatchFailure(op, "All inputs must have the same shape");
        }
    }

    // 4. Prepare MemRefs for multiple inputs/outputs
    llvm::SmallVector<Value> inputMemRefs;
    llvm::SmallVector<Value> outputMemRefs;
    llvm::SmallVector<MemRefType> memRefTypes;
    for (size_t i = 0; i < inputTensTypes.size(); ++i) {
        auto &tensorTy = inputTensTypes[i];
        auto memRefTy = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
        memRefTypes.push_back(memRefTy);
        // Convert input tensors to MemRefs
        inputMemRefs.push_back(rewriter.create<bufferization::ToMemrefOp>(loc, memRefTy, operands[i]));
        // Allocate MemRefs for outputs
        outputMemRefs.push_back(rewriter.create<memref::AllocOp>(loc, memRefTy));
    }

    // 5. Define scanning logic for multiple inputs/outputs
    LogicalResult loopResult = success();
    auto processDimension = [&](ArrayRef<Value> baseIdxsArray) {
        llvm::SmallVector<Value> baseIdxs(baseIdxsArray.begin(), baseIdxsArray.end());

        auto startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        if (reverse) {
            startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), baseShape[axis] - 1);
        }

        llvm::SmallVector<Value> firstIdx = baseIdxs;
        if (axis <= firstIdx.size()) {
            firstIdx.insert(firstIdx.begin() + axis, startInd);
        } else {
            firstIdx.push_back(startInd);
        }

        // 5.1 Process the first element: directly copy multiple inputs to multiple outputs (initialize cumulative
        // results)
        for (size_t i = 0; i < inputMemRefs.size(); ++i) {
            Value firstVal = rewriter.create<memref::LoadOp>(loc, inputMemRefs[i], firstIdx);
            rewriter.create<memref::StoreOp>(loc, firstVal, outputMemRefs[i], firstIdx);
        }

        Value axisSize = rewriter.create<arith::ConstantIndexOp>(loc, baseShape[axis]);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, axisSize, one);
        auto ifOp = rewriter.create<scf::IfOp>(loc, cmp, false);

        // Create a loop only when the axis size is greater than 1.
        rewriter.setInsertionPointToStart(ifOp.thenBlock());

        // Use a forward loop, but handle reverse indexing inside the loop.
        auto forOp = rewriter.create<scf::ForOp>(loc, one, axisSize, one);
        rewriter.setInsertionPointToStart(forOp.getBody());

        Value k = forOp.getInductionVar();

        if (reverse) {
            // Reverse scanning: Convert the forward loop index to the actual reverse index. (axis_size - 1) - k
            Value axisSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, baseShape[axis]);
            Value axisSizeMinusOne = rewriter.create<arith::SubIOp>(loc, axisSizeVal, one);
            k = rewriter.create<arith::SubIOp>(loc, axisSizeMinusOne, k);
        }

        llvm::SmallVector<Value> currIdx = baseIdxs;
        if (axis <= currIdx.size()) {
            currIdx.insert(currIdx.begin() + axis, k);
        } else {
            currIdx.push_back(k);
        }

        Value prevIndex;
        if (reverse) {
            prevIndex = rewriter.create<arith::AddIOp>(loc, k, one);
        } else {
            prevIndex = rewriter.create<arith::SubIOp>(loc, k, one);
        }

        llvm::SmallVector<Value> prevIdx = baseIdxs;
        if (axis <= prevIdx.size()) {
            prevIdx.insert(prevIdx.begin() + axis, prevIndex);
        } else {
            prevIdx.push_back(prevIndex);
        }

        // 5.4 Load current elements and previous cumulative results
        llvm::SmallVector<Value> currentVals;
        llvm::SmallVector<Value> prevResults;
        for (size_t i = 0; i < inputMemRefs.size(); ++i) {
            currentVals.push_back(rewriter.create<memref::LoadOp>(loc, inputMemRefs[i], currIdx));
            prevResults.push_back(rewriter.create<memref::LoadOp>(loc, outputMemRefs[i], prevIdx));
        }

        // 5.5 Bind parameters for custom reduction logic
        Region &combineRegion = op->getRegion(0);
        if (combineRegion.empty()) {
            op->emitError("Missing combine region in extended scan");
            loopResult = failure();
            return;
        }
        Block &combineBlock = combineRegion.front();
        // Validate that the number of reduction region arguments matches (number of previous results + number of
        // current elements)
        if (combineBlock.getNumArguments() != 2 * inputMemRefs.size()) {
            op->emitError("Combine region arguments mismatch with input count");
            loopResult = failure();
            return;
        }
        IRMapping mapping;
        for (size_t i = 0; i < inputMemRefs.size(); ++i) {
            // Bind previous results (previous value of the i-th output) to the i-th argument of the reduction region
            mapping.map(combineBlock.getArgument(i), prevResults[i]);
            // Bind current elements (current value of the i-th input) to the i+N-th argument of the reduction region (N
            // is the number of inputs)
            mapping.map(combineBlock.getArgument(i + inputMemRefs.size()), currentVals[i]);
        }

        // 5.6 Clone all operations within the reduction region
        for (Operation &innerOp : combineBlock.without_terminator()) {
            rewriter.clone(innerOp, mapping);
        }

        // 5.7 Extract reduction results and store them in outputMemRef
        Operation *yieldOp = combineBlock.getTerminator();
        if (yieldOp->getNumOperands() != outputMemRefs.size()) {
            op->emitError("Combine region returns mismatch with output count");
            loopResult = failure();
            return;
        }
        for (size_t i = 0; i < outputMemRefs.size(); ++i) {
            Value resultVal = mapping.lookup(yieldOp->getOperand(i));
            rewriter.create<memref::StoreOp>(loc, resultVal, outputMemRefs[i], currIdx);
        }

        rewriter.setInsertionPointAfter(ifOp);
    };

    // 6. Generate nested loops for non-scan dimensions
    llvm::SmallVector<int> nonScanDims;
    for (int i = 0; i < rank; ++i) {
        if (i != axis)
            nonScanDims.push_back(i);
    }
    createSimpleNestedLoops(rewriter, loc, outputMemRefs[0], nonScanDims, processDimension);

    if (failed(loopResult)) {
        return failure();
    }

    // 7. Convert multiple output MemRefs back to tensors and replace the original tt.scan operation
    llvm::SmallVector<Value> outputTensors;
    for (auto outputMemRef : outputMemRefs) {
        outputTensors.push_back(rewriter.create<bufferization::ToTensorOp>(loc, outputMemRef, true));
    }
    rewriter.replaceOp(op, outputTensors);

    return success();
}

LogicalResult ExternElementwiseClOpConverter::matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                                                              ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    if (!op.getPure()) {
        op->emitWarning() << "impure elementwise op!";
        return failure();
    }
    if (op.getSymbol().contains("__hmf_")) {
        // 1. get or create the declaration of external elementwise function
        Type dstTy = op.getResult().getType();
        bool isDstScalar = !isa<RankedTensorType>(dstTy);
        Type dstElemTy = isDstScalar ? dstTy : cast<RankedTensorType>(dstTy).getElementType();
        SmallVector<Type, 4> srcElemTys;
        SmallVector<Value, 4> srcs;
        for (auto src : op.getSrcs()) {
            if (!isa<RankedTensorType>(src.getType())) {
                src = rewriter.create<tensor::FromElementsOp>(op.getLoc(),
                                                              RankedTensorType::get({(int64_t)1}, src.getType()), src);
            }
            srcs.push_back(src);
            srcElemTys.push_back(cast<RankedTensorType>(src.getType()).getElementType());
        }
        FunctionType elemFuncType = FunctionType::get(rewriter.getContext(), srcElemTys, {dstElemTy});
        auto mod = SymbolTable::getNearestSymbolTable(op);
        auto extFunc = dyn_cast_or_null<SymbolOpInterface>(SymbolTable::lookupSymbolIn(mod, op.getSymbol()));
        // std::string symbol = op.getSymbol().str();
        bool is_libdevice =
            llvm::is_contained(libdeviceOps, op.getSymbol()) && getEnvBool("TRITON_ENABLE_LIBDEVICE", false);
        if (!extFunc) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(&mod->getRegion(0).front());
            extFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), op.getSymbol(), elemFuncType);
            extFunc.setPrivate();
            extFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(), UnitAttr::get(rewriter.getContext()));
            // set coreType for external func, otherwise InferFuncCoreTypePass will fail
            if (is_libdevice) {
                hivm::TFuncCoreType e = hivm::TFuncCoreType::AIV;
                extFunc->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(extFunc->getContext(), e));
            }
        }
        assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(mod, op.getSymbol())));
        // 2. prepare the output tensor
        Value output;
        if (isDstScalar) {
            dstTy = RankedTensorType::get({(int64_t)1}, dstElemTy);
        }
        bool found = false;
        for (Value v : srcs) {
            if (v.getType() == dstTy) {
                found = true;
                output = v;
                break;
            }
        }
        if (!found) {
            output = rewriter.create<tensor::EmptyOp>(op.getLoc(), cast<RankedTensorType>(dstTy).getShape(), dstElemTy);
        }

        if (is_libdevice) {
            auto srcType = cast<RankedTensorType>(srcs[0].getType());
            SmallVector<Value> dimSizes;
            int64_t rank = srcType.getRank();
            for (int i = 0; i < rank; ++i) {
                if (srcType.isDynamicDim(i)) {
                    auto dimOp = rewriter.create<tensor::DimOp>(loc, srcs[0], i);
                    dimSizes.push_back(dimOp);
                } else {
                    auto constOp = rewriter.create<arith::ConstantIndexOp>(loc, srcType.getDimSize(i));
                    dimSizes.push_back(constOp);
                }
            }
            // building nested loops by recursion
            std::function<Value(OpBuilder &, Location, SmallVector<Value>, Value)> buildLoops =
                [&](OpBuilder &b, Location loc, SmallVector<Value> indices, Value acc) -> Value {
                int64_t dim = indices.size();
                if (dim == rank) {
                    // innermost loop
                    SmallVector<Value> elemVals;
                    for (auto src : srcs) {
                        auto extract = b.create<tensor::ExtractOp>(loc, src, indices);
                        elemVals.push_back(extract);
                    }
                    auto call = b.create<func::CallOp>(loc, op.getSymbol(), dstElemTy, elemVals);
                    auto insert = b.create<tensor::InsertOp>(loc, call.getResult(0), acc, indices);
                    return insert;
                } else {
                    Value lower = b.create<arith::ConstantIndexOp>(loc, 0);
                    Value upper = dimSizes[dim];
                    Value step = b.create<arith::ConstantIndexOp>(loc, 1);
                    auto loop = b.create<scf::ForOp>(loc, lower, upper, step, ValueRange {acc});
                    Block *body = loop.getBody();
                    OpBuilder innerBuilder = OpBuilder::atBlockBegin(body);
                    SmallVector<Value> newIndices = indices;
                    newIndices.push_back(loop.getInductionVar());
                    Value innerAcc = loop.getRegionIterArgs()[0];
                    Value updatedAcc = buildLoops(innerBuilder, loc, newIndices, innerAcc);
                    innerBuilder.create<scf::YieldOp>(loc, updatedAcc);
                    return loop.getResult(0);
                }
            };

            Value result = buildLoops(rewriter, loc, {}, output);
            if (isDstScalar) {
                SmallVector<Value> zeroIndices(rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));
                auto extract = rewriter.create<tensor::ExtractOp>(loc, result, zeroIndices);
                rewriter.replaceOp(op, extract);
            } else {
                rewriter.replaceOp(op, result);
            }
            return success();
        }
        // 3. create the linalg.map op
        auto mapOp = rewriter.create<linalg::MapOp>(loc,
                                                    /*inputs=*/srcs,
                                                    /*init=*/output,
                                                    /*bodyBuilder=*/
                                                    [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
                                                        auto elemOp =
                                                            builder.create<func::CallOp>(loc,
                                                                                         /*name=*/op.getSymbol(),
                                                                                         /*resultType=*/dstElemTy,
                                                                                         /*operands=*/regionArgs);
                                                        builder.create<linalg::YieldOp>(loc, elemOp->getResults());
                                                    });
        if (isDstScalar) {
            // need to convert tensor back to scalar
            auto indexType = rewriter.getIndexType();
            Value zeroConstant =
                rewriter.create<arith::ConstantOp>(loc, indexType, rewriter.getIntegerAttr(indexType, 0));
            auto extractOp = rewriter.create<tensor::ExtractOp>(loc, mapOp.getResults()[0], zeroConstant);
            rewriter.replaceOp(op, extractOp);
        } else {
            rewriter.replaceOp(op, mapOp);
        }
        return success();
    }
    return failure();
}

LogicalResult UnrealizedCastConverter::matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                                       ConversionPatternRewriter &rewriter) const
{
    rewriter.eraseOp(op);
    return success();
}

LogicalResult JoinConverter::matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                                             ConversionPatternRewriter &rewriter) const
{
    Value opa = op.getLhs();
    Value opb = op.getRhs();
    auto loc = op.getLoc();

    auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
    Value emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(), resType.getElementType());

    auto shape = dyn_cast<RankedTensorType>(opa.getType()).getShape();
    auto sizes = llvm::map_to_vector(shape, [&](int64_t t) { return OpFoldResult(rewriter.getI64IntegerAttr(t)); });
    sizes.push_back(rewriter.getI64IntegerAttr(1));

    int64_t rank = resType.getRank();

    // Set last dimension stride to 2 in layout
    // As last dimension size is always 1, last dimension stride here could be
    // either 1 or 2, while stride `2` could carry interleave trait and it's
    // convenient for next lower.
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    strides.back() = rewriter.getIndexAttr(2);

    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));

    auto insert0 = rewriter.create<tensor::InsertSliceOp>(loc, opa, emptyOp, offsets, sizes, strides);

    offsets.back() = rewriter.getIndexAttr(1);
    auto insert1 = rewriter.create<tensor::InsertSliceOp>(loc, opb, insert0, offsets, sizes, strides);
    rewriter.replaceOp(op, insert1);
    return success();
}

LogicalResult CatConverter::matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                                            ConversionPatternRewriter &rewriter) const
{
    Value opa = op.getLhs();
    Value opb = op.getRhs();
    auto loc = op.getLoc();

    auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resType || resType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "only support 1D cat");
    }

    auto inputTypeA = dyn_cast<RankedTensorType>(opa.getType());
    auto inputTypeB = dyn_cast<RankedTensorType>(opb.getType());
    if (!inputTypeA || !inputTypeB || inputTypeA.getRank() != 1 || inputTypeB.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "inputs must be 1D tensors");
    }

    int64_t sizeA = inputTypeA.getShape()[0];
    int64_t sizeB = inputTypeB.getShape()[0];

    // Only handle the case where both inputs have size 1 (i.e., scalar-like)
    if (sizeA == 1 && sizeB == 1) {
        // Use scalar extract + insert
        auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(), resType.getElementType());

        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        Value scalarA = rewriter.create<tensor::ExtractOp>(loc, opa, zero);
        Value scalarB = rewriter.create<tensor::ExtractOp>(loc, opb, zero);

        Value inserted0 = rewriter.create<tensor::InsertOp>(loc, scalarA, emptyOp, zero);
        Value inserted1 = rewriter.create<tensor::InsertOp>(loc, scalarB, inserted0, one);

        rewriter.replaceOp(op, inserted1);
        return success();
    }

    // General case: use tensor.insert_slice
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(), resType.getElementType());

    auto rank = resType.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto inputType = dyn_cast<RankedTensorType>(opa.getType());

    SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
        inputType.getShape(), [&](int64_t t) { return OpFoldResult(rewriter.getI64IntegerAttr(t)); });

    auto insert0 = rewriter.create<tensor::InsertSliceOp>(loc, opa, emptyOp, offsets, sizes, strides);

    offsets[0] = rewriter.getIndexAttr(inputType.getRank() ? inputType.getShape()[0] : 1);
    auto insert1 = rewriter.create<tensor::InsertSliceOp>(loc, opb, insert0, offsets, sizes, strides);

    rewriter.replaceOp(op, insert1);
    return success();
}

/// @brief Convert tt.gather to func.call. BiShengIR captures the func
///        with assumed semantics.
/// @param op The `triton::GatherOp` operation to be rewritten.
/// @param adaptor An adaptor for the operation's operands.
/// @param rewriter A pattern rewriter used to modify the IR.
/// @return A `LogicalResult` indicating whether the rewrite was successful.
LogicalResult GatherConverter::matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                                               ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value idx = adaptor.getIndices();
    Value res = op.getResult();
    auto gatherAxis = op.getAxis();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));

    llvm::SmallString<kFuncNameCap> funcName = gatherFuncNameBase;
    int uniqueId = 0;
    while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
        funcName = gatherFuncNameBase;
        funcName += ("_" + std::to_string(uniqueId++));
    }

    auto resTy = res.getType();
    auto libFnType = rewriter.getFunctionType({src.getType(), idx.getType(), rewriter.getI32Type()}, {resTy});
    auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

    rewriter.setInsertionPoint(op);
    Value axis = rewriter.create<arith::ConstantIntOp>(loc, gatherAxis, 32);
    auto callOp =
        rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({resTy}), ValueRange({src, idx, axis}));

    rewriter.replaceOp(op, callOp);

    return success();
}

LogicalResult SplitConverter::matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                                              ConversionPatternRewriter &rewriter) const
{
    Value input = op.getSrc();
    auto loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(input.getType());

    int64_t rank = inputType.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    // Similar to JoinConverter, here adjust last dimension stride
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    strides.back() = rewriter.getIndexAttr(2);

    auto outType = dyn_cast<RankedTensorType>(op.getOutLHS().getType());
    auto sizes =
        llvm::map_to_vector(outType.getShape(), [&](int64_t t) { return OpFoldResult(rewriter.getIndexAttr(t)); });
    sizes.push_back(rewriter.getIndexAttr(1));

    auto slice0 = rewriter.create<tensor::ExtractSliceOp>(loc, outType, input, offsets, sizes, strides);

    offsets.back() = rewriter.getIndexAttr(1);
    auto slice1 = rewriter.create<tensor::ExtractSliceOp>(loc, outType, input, offsets, sizes, strides);

    SmallVector<Value, 2> slices = {slice0.getResult(), slice1.getResult()};
    rewriter.replaceOp(op, ValueRange(slices));
    return success();
}

/*
the element-wise most significant N bits of the 2N-bit product of x and y
%x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi32>
*/
LogicalResult TritonMulhiuiConverter::matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                                                      ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value opl = op.getX();
    Value opr = op.getY();
    Value res = op.getResult();
    auto newMulOp = rewriter.create<arith::MulSIExtendedOp>(loc, res.getType(), res.getType(), opl, opr);
    // triton only need the high value
    rewriter.replaceOp(op, ValueRange {newMulOp.getHigh()});
    return success();
}

LogicalResult TritonPreciseSqrtConverter::matchAndRewrite(triton::PreciseSqrtOp op, OpAdaptor adaptor,
                                                          ConversionPatternRewriter &rewriter) const
{
    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, adaptor.getOperands());
    return success();
}

LogicalResult DevicePrintConverter::matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                                                    ConversionPatternRewriter &rewriter) const
{
    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));
    SmallVector<Type, 4> inputTypes;
    for (auto arg : op.getArgs()) {
        inputTypes.push_back(arg.getType());
    }
    auto libFnType = rewriter.getFunctionType(inputTypes, {});
    auto funcOp = rewriter.create<func::FuncOp>(op.getLoc(), printFuncNameBase, libFnType);
    SymbolTable symTab(moduleOp);
    auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
    if (failed(maybePrintFuncNameAttr)) {
        return op->emitError("failed to create a unique func name for device_print");
    }
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
    auto prefixAttr = op.getPrefixAttr();
    funcOp->setAttr(prefixAttrName, prefixAttr);
    auto hexAttr = op.getHexAttr();
    funcOp->setAttr(hexAttrName, hexAttr);

    rewriter.setInsertionPoint(op);
    rewriter.create<func::CallOp>(op.getLoc(), funcOp, op.getArgs());

    rewriter.eraseOp(op);
    return success();
}

LogicalResult DeviceAssertConverter::matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                                                     mlir::ConversionPatternRewriter &rewriter) const
{
    auto msgAttr = op.getMessageAttr();
    // Filter out automatically inserted assert ops
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(msgAttr)) {
        llvm::StringRef msg = strAttr.getValue();
        if (msg.contains("overflow detected for operation")) {
            rewriter.eraseOp(op);
            return success();
        }
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));
    auto conditionType = op.getCondition().getType();

    auto libFnType = rewriter.getFunctionType({conditionType}, {});
    auto funcOp = rewriter.create<func::FuncOp>(op.getLoc(), printFuncNameBase, libFnType);
    mlir::SymbolTable symTab(moduleOp);
    auto maybePrintFuncNameAttr = symTab.renameToUnique(funcOp, {&symTab});
    if (failed(maybePrintFuncNameAttr)) {
        return op->emitError("failed to create a unique func name for device_assert");
    }
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
    funcOp->setAttr(msgAttrName, msgAttr);

    rewriter.setInsertionPoint(op);
    rewriter.create<func::CallOp>(op.getLoc(), funcOp, ValueRange {op.getCondition()});

    rewriter.eraseOp(op);
    return success();
}

LogicalResult MatmulConverter::matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                                               ConversionPatternRewriter &rewriter) const
{
    auto opa = adaptor.getA();
    auto opb = adaptor.getB();
    auto opc = adaptor.getC();
    auto dstType = cast<RankedTensorType>(op.getType());
    auto inputPrec = op.getInputPrecision();

    if (dstType.getRank() == 2) {
        auto matmulOp = rewriter.replaceOpWithNewOp<linalg::MatmulOp>(op, ValueRange {opa, opb}, ValueRange {opc});
        matmulOp->setAttr("input_precision", rewriter.getStringAttr(stringifyInputPrecision(inputPrec)));
    } else if (dstType.getRank() == 3) {
        auto matmulOp = rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(op, ValueRange {opa, opb}, ValueRange {opc});
        matmulOp->setAttr("input_precision", rewriter.getStringAttr(stringifyInputPrecision(inputPrec)));
    } else {
        llvm_unreachable("Datatype of DotOp operands could only be 2D or 3D");
    }
    return success();
}

LogicalResult FlipOpConverter::matchAndRewrite(triton::ascend::FlipOp op, OpAdaptor adaptor,
                                               ConversionPatternRewriter &rewriter) const
{
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
    auto libFnType = rewriter.getFunctionType({rankedSrcTy, i64Ty}, {rankedSrcTy});

    // Declare the callee
    auto moduleIP = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());
    auto funcOp = rewriter.create<func::FuncOp>(loc, funcName, libFnType);
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
    rewriter.restoreInsertionPoint(moduleIP);

    // dim constant
    Value dimVal = rewriter.create<arith::ConstantIntOp>(loc, dimAttr.getInt(), 64);

    // Call the backend function
    auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
    auto callOp = rewriter.create<func::CallOp>(loc, TypeRange({rankedSrcTy}), callee, ValueRange({src, dimVal}));

    Value finalValues = callOp.getResult(0);

    rewriter.replaceOp(op, {finalValues});
    return success();
}

LogicalResult SortOpConverter::matchAndRewrite(triton::ascend::SortOp op, OpAdaptor adaptor,
                                               ConversionPatternRewriter &rewriter) const
{
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
    auto libFnType = rewriter.getFunctionType({backendTensorTy, i64Ty, i1Ty}, {backendTensorTy});

    auto moduleIP = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());
    auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
    rewriter.restoreInsertionPoint(moduleIP);

    Value srcForCall = src;
    if (backendElemTy != srcElemTy) {
        srcForCall = rewriter.create<arith::SIToFPOp>(loc, backendTensorTy, src);
    }

    Value dimVal = rewriter.create<arith::ConstantIntOp>(loc, dimAttr.getInt(), 64);
    Value descVal = rewriter.create<arith::ConstantIntOp>(loc, descAttr.getValue() ? 1 : 0, 1);

    auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
    auto callOp = rewriter.create<func::CallOp>(loc, TypeRange({backendTensorTy}), callee,
                                                ValueRange({srcForCall, dimVal, descVal}));

    Value valuesFloat = callOp.getResult(0); // tensor<f16/f32>

    Value finalValues = valuesFloat;
    if (backendElemTy != srcElemTy) {
        finalValues = rewriter.create<arith::FPToSIOp>(loc, valuesTy, valuesFloat);
    }

    rewriter.replaceOp(op, {finalValues});

    return success();
}

LogicalResult DotScaledConverter::matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                                                  ConversionPatternRewriter &rewriter) const
{
    Location loc = op.getLoc();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value c = adaptor.getC();
    Value lhsScale = adaptor.getLhsScale();
    Value rhsScale = adaptor.getRhsScale();
    RankedTensorType dstType = cast<RankedTensorType>(op.getType());

    auto lhsTypeAttr = op->getAttrOfType<triton::F8F6F4TypeAttr>("lhs_type");
    auto rhsTypeAttr = op->getAttrOfType<triton::F8F6F4TypeAttr>("rhs_type");
    if (!lhsTypeAttr) {
        lhsTypeAttr = triton::F8F6F4TypeAttr::get(rewriter.getContext(), triton::F8F6F4Type::E4M3);
    }
    if (!rhsTypeAttr) {
        rhsTypeAttr = triton::F8F6F4TypeAttr::get(rewriter.getContext(), triton::F8F6F4Type::E4M3);
    }

    bool isFP8Input =
        (lhsTypeAttr.getValue() == triton::F8F6F4Type::E4M3 || lhsTypeAttr.getValue() == triton::F8F6F4Type::E5M2) &&
        (rhsTypeAttr.getValue() == triton::F8F6F4Type::E4M3 || rhsTypeAttr.getValue() == triton::F8F6F4Type::E5M2);
    if (isFP8Input) {
        if (!rhsScale) {
            RankedTensorType defaultScaleTy = RankedTensorType::get({1}, rewriter.getI8Type());
            Value defaultScaleVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(1));
            Value defaultScaleEmpty =
                rewriter.create<tensor::EmptyOp>(loc, defaultScaleTy.getShape(), defaultScaleTy.getElementType());
            rhsScale =
                rewriter.create<linalg::FillOp>(loc, ValueRange {defaultScaleVal}, ValueRange {defaultScaleEmpty})
                    .getResult(0);
        }
        Value acc = c ? c : rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), dstType.getElementType());

        Value matmulMxResult = rewriter.create<hfusion::MatMulMxOp>(loc, dstType, lhs, rhs, lhsScale, rhsScale, acc);

        Value finalResult = matmulMxResult;
        if (dstType.getElementType().isBF16()) {
            finalResult = rewriter.create<arith::TruncFOp>(loc, dstType, matmulMxResult);
        }
        rewriter.replaceOp(op, finalResult);
        return success();
    }

    if (!lhsScale) {
        return op.emitError("lhsScale is required for non-FP8 input");
    }

    RankedTensorType lhsTy = cast<RankedTensorType>(lhs.getType());
    RankedTensorType lhsScaleTy = cast<RankedTensorType>(lhsScale.getType());
    RankedTensorType rhsScaleTy = rhsScale ? cast<RankedTensorType>(rhsScale.getType()) : nullptr;
    RankedTensorType rhsTy = cast<RankedTensorType>(rhs.getType());

    Value lhsScaleOut;
    Value rhsScaleOut;
    Value c127 =
        rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI16Type(), rewriter.getI16IntegerAttr(127));
    Value c7 = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI16Type(), rewriter.getI16IntegerAttr(7));
    Type i16Ty = rewriter.getI16Type();
    Type bf16Ty = rewriter.getBF16Type();
    Type fp16Ty = rewriter.getF16Type();
    Type fp32Ty = rewriter.getF32Type();

    if (lhsScaleTy.getElementType().isIntOrIndex()) {
        RankedTensorType lhsScaleI16Ty = RankedTensorType::get(lhsScaleTy.getShape(), i16Ty);
        Value lhsScaleI16 = rewriter.create<arith::ExtSIOp>(op.getLoc(), lhsScaleI16Ty, lhsScale);

        Value lhsShift127Empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), lhsScaleI16Ty.getShape(), i16Ty);
        Value lhsShift127 =
            rewriter.create<linalg::FillOp>(op.getLoc(), ValueRange {c127}, ValueRange {lhsShift127Empty}).getResult(0);

        Value lhsScaleI16Add127 = rewriter.create<arith::AddIOp>(op.getLoc(), lhsScaleI16, lhsShift127);

        Value lhsShift7Empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), lhsScaleI16Ty.getShape(), i16Ty);
        Value lhsShift7 =
            rewriter.create<linalg::FillOp>(op.getLoc(), ValueRange {c7}, ValueRange {lhsShift7Empty}).getResult(0);
        Value lhsScaleI16Shifted = rewriter.create<arith::ShLIOp>(op.getLoc(), lhsScaleI16Add127, lhsShift7);

        RankedTensorType lhsScaleBF16Ty = RankedTensorType::get(lhsScaleTy.getShape(), bf16Ty);
        Value lhsScaleBF16 = rewriter.create<arith::BitcastOp>(op.getLoc(), lhsScaleBF16Ty, lhsScaleI16Shifted);
        if (lhsTy.getElementType() == fp16Ty) {
            RankedTensorType lhsScaleFp32Ty = RankedTensorType::get(lhsScaleTy.getShape(), fp32Ty);
            Value lhsScaleFp32 = rewriter.create<arith::ExtFOp>(op.getLoc(), lhsScaleFp32Ty, lhsScaleBF16);
            RankedTensorType lhsScaleFp16Ty = RankedTensorType::get(lhsScaleTy.getShape(), fp16Ty);
            lhsScaleOut = rewriter.create<arith::TruncFOp>(op.getLoc(), lhsScaleFp16Ty, lhsScaleFp32);
        } else {
            lhsScaleOut = lhsScaleBF16;
        }
    } else {
        lhsScaleOut =
            rewriter.create<arith::ExtFOp>(op.getLoc(), RankedTensorType::get(lhsScaleTy.getShape(), fp32Ty), lhsScale)
                .getResult();
    }

    if (rhsScale && rhsScaleTy.getElementType().isIntOrIndex()) {
        if (rhsScaleTy.getRank() != 2) {
            return op.emitError("rhsScale must be 2D for transpose");
        }

        SmallVector<int64_t> transposedShape = {rhsScaleTy.getShape()[1], rhsScaleTy.getShape()[0]};
        RankedTensorType transposedRhsScaleTy = RankedTensorType::get(transposedShape, rhsScaleTy.getElementType());

        Value transposedRhsScale =
            rewriter.create<triton::TransOp>(op.getLoc(), transposedRhsScaleTy, rhsScale,
                                             DenseI32ArrayAttr::get(rewriter.getContext(), ArrayRef<int32_t> {1, 0}));
        RankedTensorType rhsScaleI16Ty = RankedTensorType::get(transposedShape, i16Ty);
        Value rhsScaleI16 = rewriter.create<arith::ExtSIOp>(op.getLoc(), rhsScaleI16Ty, transposedRhsScale);
        Value rhsShift127Empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), rhsScaleI16Ty.getShape(), i16Ty);
        Value rhsShift127 =
            rewriter.create<linalg::FillOp>(op.getLoc(), ValueRange {c127}, ValueRange {rhsShift127Empty}).getResult(0);

        Value rhsScaleI16Add127 = rewriter.create<arith::AddIOp>(op.getLoc(), rhsScaleI16, rhsShift127);
        Value rhsShift7Empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), rhsScaleI16Ty.getShape(), i16Ty);
        Value rhsShift7 =
            rewriter.create<linalg::FillOp>(op.getLoc(), ValueRange {c7}, ValueRange {rhsShift7Empty}).getResult(0);
        Value rhsScaleI16Shifted = rewriter.create<arith::ShLIOp>(op.getLoc(), rhsScaleI16Add127, rhsShift7);

        RankedTensorType rhsScaleBF16Ty = RankedTensorType::get(transposedShape, bf16Ty);
        Value rhsScaleBF16 = rewriter.create<arith::BitcastOp>(op.getLoc(), rhsScaleBF16Ty, rhsScaleI16Shifted);

        if (rhsTy.getElementType() == fp16Ty) {
            RankedTensorType rhsScaleFp32Ty = RankedTensorType::get(transposedShape, fp32Ty);
            Value rhsScaleFp32 = rewriter.create<arith::ExtFOp>(op.getLoc(), rhsScaleFp32Ty, rhsScaleBF16);
            RankedTensorType rhsScaleFp16Ty = RankedTensorType::get(transposedShape, fp16Ty);
            rhsScaleOut = rewriter.create<arith::TruncFOp>(op.getLoc(), rhsScaleFp16Ty, rhsScaleFp32);
        } else {
            rhsScaleOut = rhsScaleBF16;
        }
        int64_t rhsD0 = rhsScaleTy.getShape()[1];
        int64_t rhsD1 = rhsScaleTy.getShape()[0];
        SmallVector<int64_t> rhsExpandedShape1 = {rhsD0, rhsD1, 1};
        RankedTensorType rhsExpandedTy1 = RankedTensorType::get(rhsExpandedShape1, rhsTy.getElementType());
        Value rhsExpanded1 =
            rewriter
                .create<triton::ExpandDimsOp>(op.getLoc(), rhsExpandedTy1, rhsScaleOut, rewriter.getI32IntegerAttr(2))
                .getResult();

        int64_t rhsDim1 = rhsTy.getShape()[0];
        if (rhsDim1 % rhsD0 != 0) {
            return op.emitError("rhs dim0 must be an integer multiple of rhsScale dim0");
        }
        int64_t rhsD2 = rhsDim1 / rhsD0;
        SmallVector<int64_t> rhsBroadcastShape = {rhsD0, rhsD1, rhsD2};
        RankedTensorType rhsBroadcastTy = RankedTensorType::get(rhsBroadcastShape, rhsTy.getElementType());
        Value rhsBroadcasted =
            rewriter.create<triton::BroadcastOp>(op.getLoc(), rhsBroadcastTy, rhsExpanded1).getResult();

        SmallVector<int32_t> transposeOrder = {0, 2, 1};
        Value transposedBroadcasted = rewriter.create<triton::TransOp>(
            op.getLoc(), RankedTensorType::get({rhsD0, rhsD2, rhsD1}, rhsTy.getElementType()), rhsBroadcasted,
            DenseI32ArrayAttr::get(rewriter.getContext(), transposeOrder));
        SmallVector<ReassociationIndices> rhsReassociation;
        rhsReassociation.push_back({0, 1});
        rhsReassociation.push_back({2});

        Value scaledRhs = rewriter
                              .create<tensor::CollapseShapeOp>(
                                  op.getLoc(), RankedTensorType::get({rhsD0 * rhsD2, rhsD1}, rhsTy.getElementType()),
                                  transposedBroadcasted, rhsReassociation)
                              .getResult();

        rhs = rewriter.create<arith::MulFOp>(op.getLoc(), rhs, scaledRhs).getResult();
    }

    int64_t D0 = lhsScaleTy.getShape()[0];
    int64_t D1 = lhsScaleTy.getShape()[1];
    SmallVector<int64_t> expandedShape1 = {D0, D1, 1};
    RankedTensorType expandedTy1 = RankedTensorType::get(expandedShape1, lhsTy.getElementType());
    Value expanded1 =
        rewriter.create<triton::ExpandDimsOp>(op.getLoc(), expandedTy1, lhsScaleOut, rewriter.getI32IntegerAttr(2))
            .getResult();

    int64_t lhsDim1 = lhsTy.getShape()[1];
    if (lhsDim1 % D1 != 0) {
        return op.emitError("lhs dim1 must be an integer multiple of lhsScale dim1");
    }
    int64_t D2 = lhsDim1 / D1;
    SmallVector<int64_t> broadcastShape = {D0, D1, D2};
    RankedTensorType broadcastTy = RankedTensorType::get(broadcastShape, lhsTy.getElementType());
    Value broadcasted = rewriter.create<triton::BroadcastOp>(op.getLoc(), broadcastTy, expanded1).getResult();

    SmallVector<ReassociationIndices> reassociation;
    reassociation.push_back({0});
    reassociation.push_back({1, 2});

    Value scaledLhs =
        rewriter
            .create<tensor::CollapseShapeOp>(op.getLoc(), RankedTensorType::get({D0, D1 * D2}, lhsTy.getElementType()),
                                             broadcasted, reassociation)
            .getResult();

    Value scaledLhsFinal = rewriter.create<arith::MulFOp>(op.getLoc(), lhs, scaledLhs).getResult();

    Operation *matmulOp;
    if (dstType.getRank() == 2) {
        matmulOp = rewriter.create<linalg::MatmulOp>(op.getLoc(), ValueRange {scaledLhsFinal, rhs}, ValueRange {c});
    } else if (dstType.getRank() == 3) {
        matmulOp =
            rewriter.create<linalg::BatchMatmulOp>(op.getLoc(), ValueRange {scaledLhsFinal, rhs}, ValueRange {c});
    } else {
        return op.emitError("DotScaledOp only support 2D or 3D tensor");
    }

    rewriter.replaceOp(op, matmulOp->getResults());
    return success();
}

LogicalResult PtrToIntConverter::matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                                                 ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value ptr = adaptor.getSrc();

    if (!mlir::isa<MemRefType>(ptr.getType())) {
        return rewriter.notifyMatchFailure(op, "input is not a memref type");
    }

    auto resultType = op.getType();

    // memref.extract_aligned_pointer_as_index is used to obtain the integer representation of the base address.
    auto ptrToIndexOp = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, ptr);

    Value intResult = rewriter.create<arith::IndexCastOp>(loc, resultType, ptrToIndexOp);

    rewriter.replaceOp(op, intResult);
    return success();
}

LogicalResult IndexPutConverter::matchAndRewrite(triton::ascend::IndexPutOp op, OpAdaptor adaptor,
                                                 ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));

    auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

    auto ptr = adaptor.getPtr();
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
    SmallVector<Type> inputTypes({ptrTy, index.getType(), value.getType(), dim.getType(), indexBoundary.getType()});
    inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
    inputTypes.append(startOffset.getTypes().begin(), startOffset.getTypes().end());
    inputTypes.append(dstStride.getTypes().begin(), dstStride.getTypes().end());
    auto libFnType = rewriter.getFunctionType(inputTypes, {});
    auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

    rewriter.setInsertionPoint(op);
    SmallVector<Value> inputVals({ptr, index, value, dim, indexBoundary});
    inputVals.append(endOffset.begin(), endOffset.end());
    inputVals.append(startOffset.begin(), startOffset.end());
    inputVals.append(dstStride.begin(), dstStride.end());
    rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}), inputVals);
    rewriter.eraseOp(op);
    return success();
}

LogicalResult GatherOutToUbConverter::matchAndRewrite(triton::ascend::GatherOutToUbOp op, OpAdaptor adaptor,
                                                      ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));

    auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

    auto src = adaptor.getSrc();
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

    SmallVector<Type> inputTypes({srcTy, index.getType(), indexBoundary.getType(), dim.getType()});
    inputTypes.append(srcStride.getTypes().begin(), srcStride.getTypes().end());
    inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
    inputTypes.append(startOffset.getTypes().begin(), startOffset.getTypes().end());
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
    auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({resTy}), inputVals);
    rewriter.replaceOp(op, callOp);
    return success();
}

LogicalResult ScatterUbToOutConverter::matchAndRewrite(triton::ascend::ScatterUbToOutOp op, OpAdaptor adaptor,
                                                       ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));

    auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

    auto ptr = adaptor.getPtr();
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

    SmallVector<Type> inputTypes({ptrTy, value.getType(), index.getType(), indexBoundary.getType(), dim.getType()});
    inputTypes.append(dstStride.getTypes().begin(), dstStride.getTypes().end());
    inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
    inputTypes.append(startOffset.getTypes().begin(), startOffset.getTypes().end());

    auto libFnType = rewriter.getFunctionType(inputTypes, {});
    auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

    rewriter.setInsertionPoint(op);
    SmallVector<Value> inputVals({ptr, value, index, indexBoundary, dim});
    inputVals.append(dstStride.begin(), dstStride.end());
    inputVals.append(endOffset.begin(), endOffset.end());
    inputVals.append(startOffset.begin(), startOffset.end());
    rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}), inputVals);
    rewriter.eraseOp(op);
    return success();
}

LogicalResult IndirectLoadConverter::matchAndRewrite(triton::ascend::IndirectLoadOp op, OpAdaptor adaptor,
                                                     ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));

    auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

    auto src = adaptor.getSrc();
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

    rewriter.setInsertionPoint(op);
    SmallVector<Value> inputVals({src, offsets});
    if (mask)
        inputVals.push_back(mask);
    if (other)
        inputVals.push_back(other);
    auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({resTy}), inputVals);
    rewriter.replaceOp(op, callOp);
    return success();
}

LogicalResult IndirectStoreConverter::matchAndRewrite(triton::ascend::IndirectStoreOp op, OpAdaptor adaptor,
                                                      ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(), std::prev(moduleOp.getBody()->end()));

    auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

    auto src = adaptor.getSrc();
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
    rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}), inputVals);
    rewriter.eraseOp(op);
    return success();
}

IndexSelectSimdConverter::IndexSelectSimdConverter(MLIRContext *context)
    : OpConversionPattern<triton::ascend::IndexSelectSimdOp>(context)
{
}

LogicalResult IndexSelectSimdConverter::matchAndRewrite(triton::ascend::IndexSelectSimdOp op, OpAdaptor adaptor,
                                                        ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    // Get converted operands
    Value src = adaptor.getSrc();
    Value indexTensor = adaptor.getIndex();
    auto srcShapeVals = adaptor.getSrcShape();
    auto srcOffsetVals = adaptor.getSrcOffset();
    auto readShapeAttr = op.getReadShape();
    int32_t dim = op.getDim();

    // Get result type
    auto resultTensorType = cast<RankedTensorType>(op.getResult().getType());
    auto elemType = resultTensorType.getElementType();
    auto resultShape = resultTensorType.getShape();

    // Convert src (tt.ptr -> memref) to the correct memref shape
    // src is now memref<?xT> after type conversion, need to reinterpret to full shape
    auto srcMemRefType = cast<MemRefType>(src.getType());

    // Build multi-dimensional memref type
    SmallVector<int64_t> fullSrcShape;
    for (auto shapeVal : srcShapeVals) {
        if (auto constOp = shapeVal.getDefiningOp<arith::ConstantIndexOp>()) {
            fullSrcShape.push_back(constOp.value());
        } else {
            fullSrcShape.push_back(ShapedType::kDynamic);
        }
    }
    auto fullSrcMemRefType = MemRefType::get(fullSrcShape, elemType);

    // Reinterpret cast from 1D to multi-dimensional
    // Build strides: stride[i] = product of all dimensions after i
    SmallVector<OpFoldResult> sizes, strides; // offsets are 0

    // Calculate strides from right to left (row-major layout)
    Value currentStride = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    strides.insert(strides.begin(), rewriter.getIndexAttr(1));

    for (int i = fullSrcShape.size() - 1; i > 0; --i) {
        // Update stride for next dimension: stride *= size[i]
        if (fullSrcShape[i] != ShapedType::kDynamic) {
            // Static dimension: record specific integer values
            currentStride = rewriter.createOrFold<arith::MulIOp>(
                loc, currentStride, rewriter.create<arith::ConstantIndexOp>(loc, fullSrcShape[i]));
            if (auto constOp = currentStride.getDefiningOp<arith::ConstantIndexOp>()) {
                strides.insert(strides.begin(), rewriter.getIndexAttr(constOp.value()));
                continue;
            }

            strides.insert(strides.begin(), currentStride);
            continue;
        }
        // Dynamic dimension: multiply by runtime value
        Value dimSize = srcShapeVals[i];
        if (!dimSize.getType().isIndex()) {
            dimSize = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), dimSize);
        }
        currentStride = rewriter.create<arith::MulIOp>(loc, currentStride, dimSize);

        strides.insert(strides.begin(), currentStride);
    }

    for (auto shapeVal : srcShapeVals) {
        OpFoldResult shapeSize;
        if (auto constOp = shapeVal.getDefiningOp<arith::ConstantIndexOp>()) {
            // Static dimension:
            // Use attributes to enable static_sizes to record specific integer values
            shapeSize = rewriter.getIndexAttr(constOp.value());
        } else {
            // Dynamic dimension: converted to the index type and used as a value
            Value idxVal = shapeVal;
            if (!idxVal.getType().isIndex())
                idxVal = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), idxVal);
            shapeSize = idxVal;
        }
        sizes.push_back(shapeSize);
    }

    auto peelAddPtrChain = [&]() -> Value {
        Value ptr = op.getSrc();
        Value offIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

        while (auto add = ptr.getDefiningOp<triton::AddPtrOp>()) {
            Value delta = add.getOffset();

            if (!delta.getType().isIndex()) {
                delta = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), delta);
            }

            offIdx = rewriter.create<arith::AddIOp>(loc, offIdx, delta);
            ptr = add.getPtr();
        }

        return offIdx;
    };

    OpFoldResult offset = peelAddPtrChain();
    auto srcMemRef = rewriter.create<memref::ReinterpretCastOp>(loc, fullSrcMemRefType, src, offset, sizes, strides);

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
    auto forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, numIndicesVal, stepOne);

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
    Value selectedIdx = rewriter.create<tensor::ExtractOp>(loc, indexTensor, ValueRange {iv});
    Value selectedIdxAsIndex = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), selectedIdx);

    // Build source subview offsets/sizes/strides
    SmallVector<OpFoldResult> srcSubviewOffsets, srcSubviewSizes, srcSubviewStrides;
    // DenseI32ArrayAttr can be implicitly converted to ArrayRef<int32_t>
    ArrayRef<int32_t> readShape = readShapeAttr;

    for (size_t i = 0; i < srcOffsetVals.size(); ++i) {
        if (i == static_cast<size_t>(dim)) {
            // Use the selected index for this dimension
            srcSubviewOffsets.push_back(selectedIdxAsIndex);
            srcSubviewSizes.push_back(rewriter.getIndexAttr(1));
        } else {
            // Use provided offset and read size for other dimensions
            Value offsetVal = srcOffsetVals[i];
            if (!offsetVal.getType().isIndex()) {
                offsetVal = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), offsetVal);
            }
            srcSubviewOffsets.push_back(offsetVal);
            srcSubviewSizes.push_back(rewriter.getIndexAttr(readShape[i]));
        }
        srcSubviewStrides.push_back(rewriter.getIndexAttr(1));
    }

    auto srcSubview =
        rewriter.create<memref::SubViewOp>(loc, srcMemRef, srcSubviewOffsets, srcSubviewSizes, srcSubviewStrides);

    // Build destination subview
    SmallVector<OpFoldResult> dstSubviewOffsets, dstSubviewSizes, dstSubviewStrides;
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

    auto dstSubview =
        rewriter.create<memref::SubViewOp>(loc, outputBuffer, dstSubviewOffsets, dstSubviewSizes, dstSubviewStrides);

    // Check if index_select is on the trailing axis (last dimension)
    if (static_cast<size_t>(dim) == fullSrcShape.size() - 1) {
        // For index_select on the trailing axis, mark as discrete memory access
        // This degrades to scalar read/write handling to avoid alignment issues
        auto copyOp = rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
        copyOp->setAttr(ConverterUtils::discreteAttrName, rewriter.getUnitAttr());
    } else {
        // For index_select on non-trailing axes, add stride alignment annotation
        // This tells the backend to handle address alignment for DMA operations
        auto dstMarkOp = rewriter.create<annotation::MarkOp>(loc, dstSubview);
        dstMarkOp->setAttr("hfusion.stride_align_dims", rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(dim)}));
        dstMarkOp->setAttr("hfusion.stride_align_value_in_byte", rewriter.getDenseI32ArrayAttr({32}));

        // Copy from source to destination
        rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    }

    // Restore insertion point
    rewriter.restoreInsertionPoint(savedInsertionPoint);

    // Convert memref to tensor
    auto resultTensor = rewriter.create<bufferization::ToTensorOp>(loc, resultTensorType, outputBuffer, true, true);

    // Mark as index_select_simd
    resultTensor->setAttr("index_select_simd", rewriter.getUnitAttr());

    // Replace the original op
    rewriter.replaceOp(op, resultTensor);

    return success();
}

} // namespace TTOpConverters
