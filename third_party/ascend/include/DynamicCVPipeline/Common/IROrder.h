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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_IR_ORDER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_IR_ORDER_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir::CVPipeline {

/// Lowest common ancestor block of `a` and `b`.
/// Pre: either argument may be null.
/// Post: null when either is null or the two chains never meet.
Block *lowestCommonAncestorBlock(Block *a, Block *b);

/// True when `a` strictly precedes `b` in SSA order.
/// `a` or `b` null => false.
/// `a` is `b` => false.
/// Sibling regions => false, meaning unordered, not "`b` first".
///
/// Block cases:
///   same block -> SSA order, via the ancestor comparison.
///   nested block -> one is ordered by its ancestor in the common block.
///   sibling regions -> unordered: they hold no dominance relation at all.
/// `isBeforeInBlock` is invalid across blocks; this is the replacement.
/// Pre: callers exclude the sibling-region case, no runtime check is done.
bool isBeforeInIr(Operation *a, Operation *b);

} // namespace mlir::CVPipeline

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_IR_ORDER_H
