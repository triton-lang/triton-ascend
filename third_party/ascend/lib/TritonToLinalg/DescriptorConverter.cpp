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

#include "TritonToLinalg/DescriptorConverter.h"
#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToLinalg/TritonOpConverter.h"
#include "TritonToLinalg/TritonToLinalgPass.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace DescriptorConverter {
using namespace mlir;
using namespace triton;

bool hasATensorDescriptorType(mlir::TypeRange types)
{
    return llvm::any_of(types, [](mlir::Type t) { return llvm::isa<mlir::triton::TensorDescType>(t); });
}

Descriptor unpackDescriptor(TensorDescType type, Value desc, ConversionPatternRewriter &rewriter)
{
    auto makeDescOp = desc.getDefiningOp<triton::MakeTensorDescOp>();
    assert(makeDescOp && "Descriptor must be defined by MakeTensorDescOp");

    Descriptor res;

    res.base = makeDescOp.getBase();
    for (auto s : makeDescOp.getShape()) {
        res.shape.push_back(rewriter.createOrFold<arith::ExtSIOp>(makeDescOp.getLoc(), rewriter.getI64Type(), s));
    }
    for (auto st : makeDescOp.getStrides()) {
        res.strides.push_back(rewriter.createOrFold<arith::ExtSIOp>(makeDescOp.getLoc(), rewriter.getI64Type(), st));
    }

    return res;
}

SmallVector<int32_t> computeOrder(ArrayRef<int64_t> shape)
{
    SmallVector<int32_t> order;
    int rank = shape.size();
    order.reserve(rank);
    // default by [dims - 1, ..., 0]
    for (int i = rank - 1; i >= 0; --i) {
        order.push_back(i);
    }
    return order;
}

LogicalResult DescriptorLoadConverter::matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                                                       ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto indices = op.getIndices();

    // 1. unpack descriptor
    auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

    // 2. create make_tensor_ptr
    SmallVector<int32_t> tensorShapeValues;
    for (auto dim : blockShape) {
        tensorShapeValues.push_back(static_cast<int32_t>(dim));
    }
    Value tensorPtr = rewriter.create<triton::MakeTensorPtrOp>(loc,
                                                               desc.base,               // base
                                                               desc.shape,              // shape
                                                               desc.strides,            // strides
                                                               indices,                 // offset
                                                               tensorShapeValues,       // tensorShape
                                                               computeOrder(blockShape) // order
    );
    // 3. replace tt.load
    auto boundaryCheck = rewriter.getDenseI32ArrayAttr({});
    triton::PaddingOptionAttr padding = nullptr;
    auto cache = triton::CacheModifierAttr::get(rewriter.getContext(), triton::CacheModifier::NONE);
    auto evict = triton::EvictionPolicyAttr::get(rewriter.getContext(), triton::EvictionPolicy::NORMAL);
    auto isVolatile = rewriter.getBoolAttr(false);

    if (auto a = op->getAttrOfType<triton::CacheModifierAttr>("cache"))
        cache = a;
    if (auto a = op->getAttrOfType<triton::EvictionPolicyAttr>("evict"))
        evict = a;
    if (auto a = op->getAttrOfType<BoolAttr>("isVolatile"))
        isVolatile = a;

    auto newLoad = rewriter.create<triton::LoadOp>(loc, descTy.getSignlessBlockType(), tensorPtr,
                                                   Value(), // mask
                                                   Value(), // other
                                                   boundaryCheck, padding, cache, evict, isVolatile);

    rewriter.replaceOp(op, newLoad.getResult());

    return success();
}

LogicalResult DescriptorStoreConverter::matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                                                        ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto indices = op.getIndices();

    // 1. unpack descriptor
    auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

    // 2. create make_tensor_ptr
    SmallVector<int32_t> tensorShapeValues;
    for (auto dim : blockShape) {
        tensorShapeValues.push_back(static_cast<int32_t>(dim));
    }
    Value tensorPtr = rewriter.create<triton::MakeTensorPtrOp>(loc,
                                                               desc.base,               // base
                                                               desc.shape,              // shape
                                                               desc.strides,            // strides
                                                               indices,                 // offset
                                                               tensorShapeValues,       // tensorShape
                                                               computeOrder(blockShape) // order
    );

    // 3. replace tt.store
    Value valueToStore = adaptor.getSrc();

    auto maskType = RankedTensorType::get(blockShape, rewriter.getI1Type());
    rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(maskType, true));
    auto boundaryCheck = rewriter.getDenseI32ArrayAttr({});
    auto cacheModifier = triton::CacheModifierAttr::get(rewriter.getContext(), triton::CacheModifier::NONE);
    auto evictionPolicy = triton::EvictionPolicyAttr::get(rewriter.getContext(), triton::EvictionPolicy::NORMAL);

    auto newStore = rewriter.create<triton::StoreOp>(loc, tensorPtr, valueToStore,
                                                     Value(), // mask
                                                     boundaryCheck, cacheModifier, evictionPolicy);

    rewriter.eraseOp(op);
    return success();
}

} // namespace DescriptorConverter
