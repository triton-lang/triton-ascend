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

#ifndef TRITON_ASCEND_INDIRECT_LOAD_REWRITE_H
#define TRITON_ASCEND_INDIRECT_LOAD_REWRITE_H

#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace IndirectLoadRewrite {

using namespace mlir;
using namespace triton;

// Tag stamped on tt.indirect_load ops that this sub-step emits so the pattern
// driver does not re-enter on its own output.
inline constexpr const char *RewrittenByIndirectLoadRewriteTAG =
    "RewrittenByIndirectLoadRewrite";

// V1 SIMT IndirectLoad fast-path rewrite:
//   Convert tt.load to tt.indirect_load when the load's effective per-axis
//   strides have a statically-known last-axis stride > 1 with a non-permuted
//   layout (i.e. ImplicitPermute would not / did not touch it, and it isn't
//   the stride==2 even-size case handled by DeinterleaveStatusOptimization).
//
//   Runs as a sub-step of TritonToLinalgPass, after processImplicitPermute,
//   and is gated on `compileOn91095Flag && forceSimtTemplateFlag`.
class LoadConverter : public OpRewritePattern<triton::LoadOp> {
public:
    explicit LoadConverter(MLIRContext *context)
        : OpRewritePattern<triton::LoadOp>(context) {}

    using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::LoadOp op,
                                  PatternRewriter &rewriter) const override;
};

}  // namespace IndirectLoadRewrite

#endif  // TRITON_ASCEND_INDIRECT_LOAD_REWRITE_H
