
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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"

namespace mlir::CVPipeline {

// Marks a `scope::ScopeOp` that wraps a load unit: a unit moving data from GM
// into on-chip memory. The canonical shape is a local buffer
// (`memref.alloc -> <ops operating on it> -> bufferization.to_tensor`, possibly
// through a subview / reinterpret_cast); a load-type `hivm.hir.custom` wrapped
// by PackMemref carries the same tag.
constexpr llvm::StringLiteral kScopeLoadAttr = "ssbuffer.load";

/// Packs `ops` into a fresh `scope::ScopeOp` created at the position of the
/// last op. Values produced inside `ops` but consumed outside are routed
/// through `scope::ReturnOp`, and their external users are redirected.
///
/// PRECONDITIONS: `ops` must already follow dominance order, and the packed
/// scope must dominate its remaining users.
/// Returns nullptr on failure; the caller is responsible for any restoration.
scope::ScopeOp packScopeOp(llvm::ArrayRef<Operation *> ops);

/// Moves the ops inside `scopeOp` into the parent block and erases `scopeOp`.
///
/// On failure the `scopeOp` is left in place (already partially modified).
/// Not suitable for monad-like results, since the op is mutated in place.
llvm::LogicalResult
unpackScopeOp(scope::ScopeOp scopeOp,
              llvm::SmallVectorImpl<Operation *> *movedOps = nullptr);

} // namespace mlir::CVPipeline

#endif
