
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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SCOPE_OP_UTILS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SCOPE_OP_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"

namespace mlir::CVPipeline {

// Pack ops into ScopeOp at the last op in ops
// Redirects the output. Returns nullptr if failed - the caller is responsible
// for restoring if needed Caller's responsibility to ensure:
// 1. Packed scopeop have correct dominance relationship
// 2. Ops themselves follows dominance order
scope::ScopeOp packScopeOp(llvm::ArrayRef<Operation *> ops);

// Move ops inside scopeop to the parent block
// Does not remove the scopeOp on failure
// Not suitable for monad-like results since it already modified the op
llvm::LogicalResult
unpackScopeOp(scope::ScopeOp scopeOp,
              llvm::SmallVectorImpl<Operation *> *movedOps = nullptr);

} // namespace mlir::CVPipeline

#endif
