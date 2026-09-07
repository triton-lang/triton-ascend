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
 * all copies or substantial portions of the Software![](substantial portions of the Software).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT![](BUT NOT) LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN![](OTHER DEALINGS IN)
 * THE SOFTWARE.
 */
#include <algorithm>

#include "DynamicCVPipeline/Common/SyncWall.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::CVPipeline;

namespace {

// prefixCount[i] = #syncPoints at positions < i, for i in [0, idx]. O(idx).
llvm::SmallVector<unsigned, 4>
SyncWall::buildPrefixCount(const llvm::SmallVector<Operation *, 4> &syncs, unsigned idx) {
  llvm::SmallVector<unsigned, 4> prefix(idx + 1);
  unsigned acc = 0;
  for (unsigned i = 0, si = 0; i < idx; ++i) {
    prefix[i] = acc;
    if (si < syncs.size() && positionOf(syncs[si]) == i) {
      ++acc;
      ++si;
    }
  }
  prefix[idx] = acc;
  return prefix;
}

} // namespace

SyncWall::SyncWall(Block *block) {
  // Phase 1: assign source-order ordinals to block-level ops (PreOrder).
  unsigned idx = 0;
  block->walk<WalkOrder::PreOrder>([&](Operation *op) {
    Operation *owner = getAncestorInBlock(op, block);
    if (owner == nullptr) {
      return;
    }
    if (!ordinal.contains(owner)) {
      ordinal[owner] = idx++;
    }
  });

  llvm::DenseSet<Operation *> cubeSyncs;
  llvm::DenseSet<Operation *> vectorSyncs;

  auto addSyncPoint = [&](Operation *op, llvm::DenseSet<Operation *> &syncs) {
    if (syncs.contains(op)) {
      return true; // already added
    }
    if (isExternalSyncOp(op)) {
      syncs.insert(op);
      // mark parent as sync point, so that isSyncPoint() can detect 
      // it even if the parents
      auto *parent = op->getParentOp();
      if (parent != nullptr && getAncestorInBlock(parent, block)) {
        // ensure parentOp is also in this block
        syncs.insert(parent);
      }
      return true;
    }

    return false;
  };

  // Phase 2: identify all-level sync points via PostOrder walk. 
  block->walk<WalkOrder::PostOrder>([&](Operation *op) {

    auto core = CVPipeline::getOpCoreType(op);
    if (core == CoreType::CUBE_ONLY) {
      addSyncPoint(op, cubeSyncs); 
    }
    else if (core == CoreType::VECTOR_ONLY) {
      addSyncPoint(op, vectorSyncs); 
    }
    else {
      return; 
    }

  });

  auto byPos = [this](Operation *a, Operation *b) {
    return positionOf(a) < positionOf(b);
  };
  llvm::sort(cube.syncPoints, byPos);
  llvm::sort(vector.syncPoints, byPos);
  cube.prefixCount = buildPrefixCount(cube.syncPoints, idx);
  vector.prefixCount = buildPrefixCount(vector.syncPoints, idx);
}

unsigned SyncWall::positionOf(Operation *op) const {
  auto it = ordinal.find(op);
  if (it != ordinal.end()) {
    return it->second;
  }
  return 0;
}

bool SyncWall::hasSyncBetween(Operation *a, Operation *b) const {
  auto aCore = CVPipeline::getOpCoreType(a);
  if (aCore != CVPipeline::getOpCoreType(b)) {
    return false;
  }
  auto syncs = syncPointsOf(aCore);
  unsigned lo = positionOf(a);
  unsigned hi = positionOf(b);
  if (lo > hi) {
    std::swap(lo, hi);
  }
  auto it = std::upper_bound(syncs.begin(), syncs.end(), lo,
                             [this](Operation *o, unsigned v) {
                               return positionOf(o) < v;
                             });
  return it != syncs.end() && positionOf(*it) < hi;
}

unsigned SyncWall::segmentOf(Operation *op) const {
  unsigned pos = positionOf(op);
  auto core = CVPipeline::getOpCoreType(op);
  const llvm::SmallVector<unsigned, 4> *prefix = nullptr;
  if (core == CoreType::CUBE_ONLY) {
    prefix = &cube.prefixCount;
  } else if (core == CoreType::VECTOR_ONLY) {
    prefix = &vector.prefixCount;
  }
  if (prefix != nullptr && pos < prefix->size()) {
    return (*prefix)[pos];
  }
  return 0;
}

bool SyncWall::sameSegment(Operation *a, Operation *b) const {
  return CVPipeline::getOpCoreType(a) == CVPipeline::getOpCoreType(b) &&
         segmentOf(a) == segmentOf(b);
}

ArrayRef<Operation*> SyncWall::syncPointsOf(CoreType core) const {
  return core == CoreType::VECTOR_ONLY ? vector.syncPoints : cube.syncPoints;
}

bool SyncWall::isSyncPoint(Operation *op, CoreType core) const {
  auto contains = [&](const llvm::SmallVector<Operation*, 4> &syncs,
                     Operation *op) {
    auto it = std::lower_bound(syncs.begin(), syncs.end(), op,
                                   [&](Operation *a, Operation *b) {
                                     return positionOf(a) < positionOf(b);
                                   });
    return it != syncs.end() && *it == op;
  };
  if (core == CoreType::CUBE_ONLY) {
    return contains(cube.syncPoints, op);
  }
  if (core == CoreType::VECTOR_ONLY) {
    return contains(vector.syncPoints, op);
  }
  return contains(cube.syncPoints, op) || contains(vector.syncPoints, op);
}
