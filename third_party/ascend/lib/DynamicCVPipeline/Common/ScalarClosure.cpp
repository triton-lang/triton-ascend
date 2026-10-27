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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"

#include "DynamicCVPipeline/Common/ScalarClosure.h"
#include "DynamicCVPipeline/Common/Utils.h"

static constexpr const char *DEBUG_TYPE = "ScalarClosure";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

using namespace mlir;
using namespace CVPipeline;

void ScalarClosure::collectOuterDependency(Operation *op) {
  op->walk([this](Operation *nestedOp) {
    for (auto operand : nestedOp->getOperands()) {
      collectScalarClosure(operand);
    }
  });
};

void ScalarClosure::collectScalarClosure(Value val) {
  if (!val || isa<TensorType>(val.getType())) {
    return;
  }
  Operation *defOp = val.getDefiningOp();
  if (!defOp || llvm::isa<memref::AllocOp, memref::AllocaOp>(defOp)) {
    return;
  }

  auto *defBlock = defOp->getBlock();
  if (!defBlock || (defBlock != block &&
                    // to prevent memref dependencies in main loop
                    (!includeParent || defBlock != parentBlock))) {
    return;
  }

  // hard to determine - if scf ops have scalar/memref result, should we clone
  // the op as well? Perhaps walk into the subregions and clone the
  // corresponding ops would be better. But for now let's just clone them
  collectOuterDependency(defOp);
  scalarOps.insert(defOp);
}

void ScalarClosure::collect() {
  if (block == nullptr) {
    return;
  }
  for (Operation *op : ops) {
    collectOuterDependency(op);
  }
}

ScalarClosure::ScalarClosure(Block *block, ArrayRef<Operation *> ops,
                             bool includeParent)
    : block(block), ops(ops), includeParent(includeParent) {
  if (!block) {
    return;
  }
  Operation *parentOp = block->getParentOp();
  if (!parentOp) {
    return;
  }
  parentBlock = parentOp->getBlock();
}

namespace {
// The block that directly contains the parent op of `block` (its conceptual
// parent in the nesting tree). Returns nullptr at the top level (module block).
Block *parentBlockOf(Block *block) {
  if (!block) {
    return nullptr;
  }
  Operation *parentOp = block->getParentOp();
  return parentOp ? parentOp->getBlock() : nullptr;
}

// Lowest common ancestor block of `a` and `b`, using the two-pointer technique
// (O(1) extra space). Both pointers advance one level per step; when one runs
// past the root it restarts at the other's start, so each walks
// depth(a) + depth(b) levels and they meet at the LCA. Two disjoint trees both
// end at nullptr and meet there.
//
// Do NOT implement this by walking `a` upward into a set while advancing `b`
// one step at a time: when the two nodes have different depths the shallower
// pointer reaches the LCA before the deeper one records it, so the LCA is
// missed (or a higher ancestor is wrongly returned).
Block *lowestCommonAncestorBlock(Block *a, Block *b) {
  if (!a || !b) {
    return nullptr;
  }
  Block *p = a;
  Block *q = b;
  while (p != q) {
    p = p ? parentBlockOf(p) : b;
    q = q ? parentBlockOf(q) : a;
  }
  return p;
}
} // namespace

bool ScalarClosure::isBefore(Operation *a, Operation *b) {
  if (!a || !b || a == b) {
    return false;
  }

  // Scalar ops may live in different blocks. Comparing them directly via
  // `isBeforeInBlock` is invalid, so first compare their ancestors in the
  // lowest common ancestor block, then fall back to the own in-block order.
  Block *commonBlock = lowestCommonAncestorBlock(a->getBlock(), b->getBlock());
  if (!commonBlock) {
    return false;
  }
  Operation *aTop = commonBlock->findAncestorOpInBlock(*a);
  Operation *bTop = commonBlock->findAncestorOpInBlock(*b);
  if (aTop != bTop) {
    return aTop->isBeforeInBlock(bTop);
  }

  // Same ancestor (both nested under the same op): order within the own block.
  if (a->getBlock() == b->getBlock()) {
    return a->isBeforeInBlock(b);
  }

  // Sibling regions of one multi-region op (e.g. `scf.if` then/else) have no
  // SSA dominance relation: report unordered and let the caller avoid it.
  return false;
}

llvm::ArrayRef<Operation *> ScalarClosure::getScalarOps() {
  SmallVector<Operation *> sorted(scalarOps.begin(), scalarOps.end());

  // Order by dominance. Ops may live in `block` or `parentBlock`, so
  // cross-block pairs are resolved by `isBefore` through their lowest common
  // ancestor block.
  llvm::sort(sorted,
             [this](Operation *a, Operation *b) { return isBefore(a, b); });

  scalarOps.clear();
  for (Operation *op : sorted) {
    scalarOps.insert(op);
  }
  return scalarOps.getArrayRef();
}

std::pair<SmallVector<Operation *>, IRMapping>
ScalarClosure::capture(int blockId) {
  if (scalarOps.empty()) {
    return {};
  }

  IRMapping mapper;
  llvm::ArrayRef<Operation *> opsToClone = getScalarOps();

  Operation *firstOp = nullptr;
  for (auto *op : ops) {
    if (op->getBlock() != block) {
      continue;
    }
    if (firstOp == nullptr || op->isBeforeInBlock(firstOp)) {
      firstOp = op;
    }
  }
  OpBuilder builder(firstOp);
  builder.setInsertionPoint(firstOp);

  SmallVector<Operation *> newOps;

  LDBG("Collecting scalar closures for blockId: " << blockId);
  for (Operation *scalarOp : opsToClone) {
    if (llvm::is_contained(ops, scalarOp)) {
      // the current operation have the same blockId, but will not be cloned
      // hence we need to move insertion point
      builder.setInsertionPointAfter(scalarOp);
      continue;
    }

    LDBG("Clone: " << *scalarOp);

    Operation *cloned = builder.clone(*scalarOp, mapper);
    cloned->setAttr(CVPipeline::kBlockId,
                    Builder(scalarOp->getContext()).getI32IntegerAttr(blockId));
    newOps.push_back(cloned);
  }

  return {newOps, mapper};
}
