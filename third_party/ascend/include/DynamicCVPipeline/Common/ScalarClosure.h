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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SCALAR_CLOSURE_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SCALAR_CLOSURE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir::CVPipeline {

/// Scalar and view ops that a set of ops depends on.
/// The walk stops at the first tensor dependency.
///
/// Used to clone index arithmetic into a split block.
/// Also decides which view ops travel with a packed scope.
///
/// `includeParent` -> also collect from the parent block of `block`.
/// `collectImpure` -> also collect producers with a memory effect.
/// Off by default: a closure op may be cloned, and a clone repeats its effect.
class ScalarClosure {
public:
  ScalarClosure(Block *block, ArrayRef<Operation *> ops, bool includeParent,
                bool collectImpure = false);

  /// Collects the closure of `ops` into `scalarOps`.
  void collect();

  /// Collected ops in insertion order, which is not dominance order.
  /// Callers needing order sort a copy with `llvm::sort(ops, isBeforeInIr)`.
  llvm::ArrayRef<Operation *> getScalars() const;

  /// Clones every collected op that is not in `ops` into `block`.
  /// Each clone is tagged `blockId`; originals in `ops` stay in place.
  /// Post: a clone reads earlier clones, never originals after it.
  /// Returns the clones and the original -> clone mapping.
  std::pair<SmallVector<Operation *>, IRMapping> capture(int blockId);

private:
  void collectScalarClosure(Value val);
  void collectOuterDependency(Operation *op);

  mlir::Block *block = nullptr;
  mlir::Block *parentBlock = nullptr;
  bool includeParent = true;
  bool collectImpure = false;
  llvm::ArrayRef<Operation *> ops;
  llvm::SetVector<Operation *> scalarOps;
};

} // namespace mlir::CVPipeline

#endif
