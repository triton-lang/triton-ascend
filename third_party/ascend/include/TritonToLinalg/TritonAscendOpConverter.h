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

#ifndef TRITON_TO_LINALG_TRITONASCENDOPCONVERTER_H
#define TRITON_TO_LINALG_TRITONASCENDOPCONVERTER_H

#include "TritonToLinalg/MemoryPointer.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

namespace TritonAscendOpConverters {
using namespace mlir;
using namespace triton;

struct DotConverter : public OpConversionPattern<triton::ascend::DotOp> {
  using OpConversionPattern<triton::ascend::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ascend::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FlipOpConverter : public OpConversionPattern<triton::ascend::FlipOp> {
  using OpConversionPattern<triton::ascend::FlipOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ascend::FlipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  static constexpr StringRef baseFuncName = "triton_flip";
};

struct SortOpConverter : public OpConversionPattern<triton::ascend::SortOp> {
  using OpConversionPattern<triton::ascend::SortOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ascend::SortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class IndexPutConverter
    : public OpConversionPattern<triton::ascend::IndexPutOp> {
public:
  IndexPutConverter(MLIRContext *context, const MemoryPointerInfos &infos,
                    PatternBenefit benefit = PatternBenefit(1))
      : OpConversionPattern<triton::ascend::IndexPutOp>(context, benefit),
        pointerInfos(infos) {}

  LogicalResult
  matchAndRewrite(triton::ascend::IndexPutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_index_put";
  const MemoryPointerInfos &pointerInfos;
};

class GatherOutToUbConverter
    : public OpConversionPattern<triton::ascend::GatherOutToUbOp> {
public:
  GatherOutToUbConverter(MLIRContext *context, const MemoryPointerInfos &infos,
                         PatternBenefit benefit = PatternBenefit(1))
      : OpConversionPattern<triton::ascend::GatherOutToUbOp>(context, benefit),
        pointerInfos(infos) {}

  LogicalResult
  matchAndRewrite(triton::ascend::GatherOutToUbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton__gather_out_to_ub";
  const MemoryPointerInfos &pointerInfos;
};

class ScatterUbToOutConverter
    : public OpConversionPattern<triton::ascend::ScatterUbToOutOp> {
public:
  ScatterUbToOutConverter(MLIRContext *context, const MemoryPointerInfos &infos,
                          PatternBenefit benefit = PatternBenefit(1))
      : OpConversionPattern<triton::ascend::ScatterUbToOutOp>(context, benefit),
        pointerInfos(infos) {}

  LogicalResult
  matchAndRewrite(triton::ascend::ScatterUbToOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_scatter_ub_to_out";
  const MemoryPointerInfos &pointerInfos;
};

class IndirectLoadConverter
    : public OpConversionPattern<triton::ascend::IndirectLoadOp> {
public:
  IndirectLoadConverter(MLIRContext *context, const MemoryPointerInfos &infos,
                        PatternBenefit benefit = PatternBenefit(1))
      : OpConversionPattern<triton::ascend::IndirectLoadOp>(context, benefit),
        pointerInfos(infos) {}

  LogicalResult
  matchAndRewrite(triton::ascend::IndirectLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_indirect_load";
  const MemoryPointerInfos &pointerInfos;
};

class StrideLoadConverter
    : public OpConversionPattern<triton::ascend::StrideLoadOp> {
public:
  StrideLoadConverter(MLIRContext *context, const MemoryPointerInfos &infos,
                      PatternBenefit benefit = PatternBenefit(1))
      : OpConversionPattern<triton::ascend::StrideLoadOp>(context, benefit),
        pointerInfos(infos) {}

  LogicalResult
  matchAndRewrite(triton::ascend::StrideLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_stride_load";
  const MemoryPointerInfos &pointerInfos;
};

class StrideStoreConverter
    : public OpConversionPattern<triton::ascend::StrideStoreOp> {
public:
  StrideStoreConverter(MLIRContext *context, const MemoryPointerInfos &infos,
                       PatternBenefit benefit = PatternBenefit(1))
      : OpConversionPattern<triton::ascend::StrideStoreOp>(context, benefit),
        pointerInfos(infos) {}

  LogicalResult
  matchAndRewrite(triton::ascend::StrideStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_stride_store";
  const MemoryPointerInfos &pointerInfos;
};

class IndirectStoreConverter
    : public OpConversionPattern<triton::ascend::IndirectStoreOp> {
public:
  IndirectStoreConverter(MLIRContext *context, const MemoryPointerInfos &infos,
                         PatternBenefit benefit = PatternBenefit(1))
      : OpConversionPattern<triton::ascend::IndirectStoreOp>(context, benefit),
        pointerInfos(infos) {}

  LogicalResult
  matchAndRewrite(triton::ascend::IndirectStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_indirect_store";
  const MemoryPointerInfos &pointerInfos;
};

class IndexSelectSimdConverter
    : public OpConversionPattern<triton::ascend::IndexSelectSimdOp> {
public:
  IndexSelectSimdConverter(MLIRContext *context,
                           const MemoryPointerInfos &infos,
                           PatternBenefit benefit = PatternBenefit(1))
      : OpConversionPattern<triton::ascend::IndexSelectSimdOp>(context,
                                                               benefit),
        pointerInfos(infos) {}

  LogicalResult
  matchAndRewrite(triton::ascend::IndexSelectSimdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  const MemoryPointerInfos &pointerInfos;
};

} // namespace TritonAscendOpConverters

#endif
