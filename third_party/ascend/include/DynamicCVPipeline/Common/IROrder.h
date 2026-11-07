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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir::CVPipeline {

/// Returns the lowest common ancestor block of `a` and `b`, or null when either
/// is null. Two-pointer walk: no allocation, O(depth).
Block *lowestCommonAncestorBlock(Block *a, Block *b);

/// True when `a` strictly precedes `b` in SSA order, i.e. when `a` dominates
/// `b`. Never call `isBeforeInBlock` on ops of different blocks; this is the
/// cross-block replacement.
/// Precondition: `a` and `b` must not live in sibling regions of one
/// multi-region op (`scf.if` then/else, `scf.for` body/...): such a pair has no
/// dominance relation at all, and `false` is then overloaded to mean both
/// "`b` precedes `a`" and "unordered". Callers feeding arbitrary op sets are
/// responsible for excluding that case; no runtime check is performed.
bool isBeforeInIr(Operation *a, Operation *b);

/// Sorts `ops` in place into the order given by `isBeforeInIr`.
void sortByDomOrder(MutableArrayRef<Operation *> ops);

/// Convenience overload for the containers used across the CV pipeline.
void sortByDomOrder(SmallVectorImpl<Operation *> &ops);

} // namespace mlir::CVPipeline

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_IR_ORDER_H
