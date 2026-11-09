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
#ifndef TRITON_ADAPTER_IMPLICIT_PERMUTE_H
#define TRITON_ADAPTER_IMPLICIT_PERMUTE_H

#include "TritonMemoryAccess/MemoryAccessTags.h"
#include "TritonToStructured/MemOpConverter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace ImplicitPermute {

using namespace mlir;
using namespace triton;

// Tag attached to load/store/atomic ops that this pass rewrote, so downstream
// sub-steps (e.g. StridedLoadStoreRewrite) can detect "already handled by
// ImplicitPermute" and avoid double-processing.
inline constexpr const char *ImplicitPermuteHandledTAG =
    mlir::triton::memory_access::ImplicitPermuteHandledTAG;

class LoadConverter : public OpRewritePattern<triton::LoadOp> {
public:
  explicit LoadConverter(MLIRContext *context, bool compileOn91095Flag = false,
                         bool existDotFlag = false)
      : OpRewritePattern<triton::LoadOp>(context),
        compileOn91095Flag(compileOn91095Flag), existDotFlag(existDotFlag) {}

  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override;

private:
  bool compileOn91095Flag = false;
  bool existDotFlag = false;
};

class StoreConverter : public OpRewritePattern<triton::StoreOp> {
public:
  explicit StoreConverter(MLIRContext *context)
      : OpRewritePattern<triton::StoreOp>(context) {}

  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

class AtomicRMWConverter : public OpRewritePattern<triton::AtomicRMWOp> {
public:
  explicit AtomicRMWConverter(MLIRContext *context)
      : OpRewritePattern<triton::AtomicRMWOp>(context) {}

  using OpRewritePattern<triton::AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override;
};

class AtomicCASConverter : public OpRewritePattern<triton::AtomicCASOp> {
public:
  explicit AtomicCASConverter(MLIRContext *context)
      : OpRewritePattern<triton::AtomicCASOp>(context) {}

  using OpRewritePattern<triton::AtomicCASOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AtomicCASOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace ImplicitPermute

#endif
