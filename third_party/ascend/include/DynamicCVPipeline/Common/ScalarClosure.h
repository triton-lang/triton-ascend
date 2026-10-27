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

/// Collects the scalar / view ops that a set of ops depends on, stopping at the
/// first tensor dependency. Used to clone minimal index computation into a
/// split block, and to decide which view ops travel with a packed scope.
class ScalarClosure {
public:
  ScalarClosure(Block *block, ArrayRef<Operation *> ops, bool includeParent);

  /// Collects the closure of `ops` into `scalarOps`.
  void collect();

  /// Returns the collected scalar ops in dominance order. Ops may live in
  /// either the current block or its parent block; they are ordered by their
  /// ancestor in the lowest common ancestor block first, then by their position
  /// in their own block. The internal SetVector is rewritten so that repeated
  /// calls keep a stable, sorted backing store.
  ///
  /// PRECONDITION: ops must not be spread over different regions of the same
  /// multi-region op (e.g. the then/else regions of one `scf.if`).
  /// `ScalarClosure` guarantees this by construction (it collects only from
  /// `block`/ `parentBlock`); external callers feeding arbitrary op sets are
  /// responsible for upholding it. No runtime check is performed.
  llvm::ArrayRef<Operation *> getScalarOps();

  /// Comparator for dominance order: true when `a` strictly precedes `b`. Used
  /// as the `llvm::sort` predicate in `getScalarOps`; both operations must be
  /// in `scalarOps`.
  ///
  /// PRECONDITION: same as `getScalarOps`. If a caller violates it, a
  /// sibling-region pair is reported as `false` in both orders, i.e. `false` is
  /// overloaded and means either "`b` precedes `a`" or "unordered".
  bool isBefore(Operation *a, Operation *b);

  /// Clones every collected op that is not already inside `ops` into the block
  /// of `ops`, tags each clone with `blockId`, and returns the clones plus the
  /// original -> clone mapping. Ops already in `ops` are left in place.
  std::pair<SmallVector<Operation *>, IRMapping> capture(int blockId);

private:
  void collectScalarClosure(Value val);
  void collectOuterDependency(Operation *op);

  mlir::Block *block = nullptr;
  mlir::Block *parentBlock = nullptr;
  bool includeParent = true;
  llvm::ArrayRef<Operation *> ops;
  llvm::SetVector<Operation *> scalarOps;
};

} // namespace mlir::CVPipeline

#endif
