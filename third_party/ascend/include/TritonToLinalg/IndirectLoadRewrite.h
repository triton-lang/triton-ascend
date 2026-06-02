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

// Tag stamped on tt.load ops that this sub-step has inspected but chose not
// to rewrite (e.g. last stride == 1, permuted, deinterleave path). Prevents
// the greedy pattern driver from re-invoking the pattern on the same op,
// which would re-run PtrAnalysis and accumulate dead helper IR until the
// driver gives up with PassManager::run failed.
inline constexpr const char *InspectedByIndirectLoadRewriteTAG =
    "InspectedByIndirectLoadRewrite";

// V1 SIMT IndirectLoad fast-path rewrite:
//   Convert tt.load to tt.indirect_load when the load's effective per-axis
//   strides have a dynamic or static non-power-of-two last-axis stride with a
//   non-permuted layout. Static power-of-two strides stay on the legacy
//   reinterpret_cast/memref.load/copy path.
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

// V2: mirror of LoadConverter for tt.store -> tt.indirect_store. Same stride
// dispatch policy, same source-op restrictions (AddPtr / make_tensor_ptr /
// one-level advance), same MLIR-pattern-contract handling via the
// Inspected/Rewritten tags.
class StoreConverter : public OpRewritePattern<triton::StoreOp> {
public:
    explicit StoreConverter(MLIRContext *context)
        : OpRewritePattern<triton::StoreOp>(context) {}

    using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::StoreOp op,
                                  PatternRewriter &rewriter) const override;
};

// Detects the FLA per-head strided base `base + (pid % S)` produced by splitting
// the H axis. Shared between the IndirectLoad kkt strided-DMA guard and the
// StridedAxisCoalescing rewrite. Returns the matching AddPtrOp, or a null
// AddPtrOp if the base is not an ih-split per-head pointer.
triton::AddPtrOp findIhAddPtr(mlir::Value base, int64_t S);

}  // namespace IndirectLoadRewrite

#endif  // TRITON_ASCEND_INDIRECT_LOAD_REWRITE_H
