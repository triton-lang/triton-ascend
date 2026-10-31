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

#include "llvm/ADT/STLExtras.h"

#include "ascend/include/DynamicCVPipeline/Common/IROrder.h"

using namespace mlir;
using namespace CVPipeline;

static Block *parentBlockOf(Block *block) {
  Operation *parentOp = block->getParentOp();
  return parentOp ? parentOp->getBlock() : nullptr;
}

// Two-pointer walk: both pointers advance one level per step, and a pointer
// that runs past the root restarts at the other's start, so each walks depth(a)
// + depth(b) levels and they meet at the LCA (two disjoint trees both end at
// nullptr and meet there).
//
// Do NOT implement this by walking `a` upward into a set while advancing `b`
// one step at a time: when the two nodes have different depths the shallower
// pointer reaches the LCA before the deeper one records it, so the LCA is
// missed (or a higher ancestor is wrongly returned).
Block *CVPipeline::lowestCommonAncestorBlock(Block *a, Block *b) {
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

bool CVPipeline::isBeforeInIr(Operation *a, Operation *b) {
  if (!a || !b || a == b) {
    return false;
  }

  // Ops may live in different blocks, where a direct `isBeforeInBlock` is
  // invalid: compare their ancestors in the lowest common ancestor block first,
  // then fall back to the order inside their own block.
  Block *commonBlock = lowestCommonAncestorBlock(a->getBlock(), b->getBlock());
  if (!commonBlock) {
    return false;
  }
  Operation *aTop = commonBlock->findAncestorOpInBlock(*a);
  Operation *bTop = commonBlock->findAncestorOpInBlock(*b);
  if (aTop != bTop) {
    return aTop->isBeforeInBlock(bTop);
  }

  if (a->getBlock() == b->getBlock()) {
    return a->isBeforeInBlock(b);
  }

  // Sibling regions of one multi-region op have no SSA order: report unordered.
  return false;
}

void CVPipeline::sortByDomOrder(MutableArrayRef<Operation *> ops) {
  llvm::sort(ops,
             [](Operation *a, Operation *b) { return isBeforeInIr(a, b); });
}

void CVPipeline::sortByDomOrder(SmallVectorImpl<Operation *> &ops) {
  sortByDomOrder(MutableArrayRef<Operation *>(ops));
}
