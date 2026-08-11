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

#ifndef TRITON_ASCEND_STRIDED_LOAD_STORE_REWRITE_H
#define TRITON_ASCEND_STRIDED_LOAD_STORE_REWRITE_H

#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace StridedLoadStoreRewrite {

using namespace mlir;
using namespace triton;

// Tag stamped on Ascend ops that this sub-step emits so the pattern driver does
// not re-enter on its own output.
inline constexpr const char *RewrittenByStridedLoadStoreRewriteTAG =
    "RewrittenByStridedLoadStoreRewrite";

// Tag stamped on tt.load ops that this sub-step has inspected but chose not
// to rewrite (e.g. last stride == 1, permuted, deinterleave path). Prevents
// the greedy pattern driver from re-invoking the pattern on the same op,
// which would re-run PtrAnalysis and accumulate dead helper IR until the
// driver gives up with PassManager::run failed.
inline constexpr const char *InspectedByStridedLoadStoreRewriteTAG =
    "InspectedByStridedLoadStoreRewrite";

// V1 SIMT strided-access fast-path rewrite:
//   Convert tt.load to ascend.stride_load when its mask and rank fit the
//   strided template ABI, otherwise use ascend.unstructured_load. The access
//   must have a statically-known last-axis stride > 1 with a non-permuted
//   layout (i.e. ImplicitPermute would not / did not touch it, and it isn't
//   the stride==2 even-size case handled by DeinterleaveStatusOptimization).
//
//   Runs as a sub-step of TritonToLinalgPass, after processImplicitPermute,
//   and is gated on 910_95 with compile-mode=simd_simt_template. Cost-model
//   controlled routing can also enable it for selected operations in local
//   SIMT scopes while leaving unselected operations on the SIMD path.
//   model-controlled routing additionally filters every rewrite to operations
//   enclosed by the local SIMT route scope.
class LoadConverter : public OpRewritePattern<triton::LoadOp> {
public:
  explicit LoadConverter(MLIRContext *context)
      : OpRewritePattern<triton::LoadOp>(context) {}

  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override;
};

// V2: mirror of LoadConverter for stores. It emits ascend.stride_store or
// ascend.unstructured_store under the same trigger and source-op restrictions
// (AddPtr / make_tensor_ptr / one-level advance), with the same pattern-driver
// handling via the Inspected/Rewritten tags.
class StoreConverter : public OpRewritePattern<triton::StoreOp> {
public:
  explicit StoreConverter(MLIRContext *context)
      : OpRewritePattern<triton::StoreOp>(context) {}

  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace StridedLoadStoreRewrite

#endif // TRITON_ASCEND_STRIDED_LOAD_STORE_REWRITE_H
