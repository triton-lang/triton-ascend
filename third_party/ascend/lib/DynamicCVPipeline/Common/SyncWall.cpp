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
#include <algorithm>

#include "ascend/include/DynamicCVPipeline/Common/SyncWall.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::CVPipeline;

// prefixCount[i] = #syncPoints at positions < i, for i in [0, idx]. O(idx).
llvm::SmallVector<unsigned>
SyncWall::buildPrefixCount(ArrayRef<Operation *> syncs, unsigned idx) {
  llvm::SmallVector<bool> hasSync(idx, false);
  for (Operation *op : syncs) {
    unsigned pos = positionOf(op);
    if (pos < idx) {
      hasSync[pos] = true;
    }
  }

  llvm::SmallVector<unsigned> prefix(idx + 1);
  unsigned acc = 0;
  for (unsigned i = 0; i < idx; ++i) {
    prefix[i] = acc;
    if (hasSync[i]) {
      ++acc;
    }
  }
  prefix[idx] = acc;
  return prefix;
}

SyncWall::SyncWall(Block *block) {
  unsigned idx = 0;
  block->walk<WalkOrder::PreOrder>([&](Operation *op) { ordinal[op] = idx++; });

  llvm::DenseSet<Operation *> cubeSyncs;
  llvm::DenseSet<Operation *> vectorSyncs;

  auto recordIfSyncPoint = [&](Operation *op,
                               llvm::DenseSet<Operation *> &syncs) {
    if (syncs.contains(op) || isExternalSyncOp(op)) {
      syncs.insert(op);
      // mark parent as sync point
      auto *parent = op->getParentOp();
      if (parent != nullptr && getAncestorInBlock(parent, block)) {
        syncs.insert(parent);
      }
    }
    return;
  };

  block->walk([&](Operation *op) {
    auto core = CVPipeline::getOpCoreType(op);
    if (core == CoreType::CUBE_ONLY) {
      recordIfSyncPoint(op, cubeSyncs);
    } else if (core == CoreType::VECTOR_ONLY) {
      recordIfSyncPoint(op, vectorSyncs);
    } else {
      return;
    }
  });

  llvm::append_range(cube.syncPoints, cubeSyncs);
  llvm::append_range(vector.syncPoints, vectorSyncs);

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

Operation *SyncWall::getPredSyncOpInSameBlock(Operation *op) const {
  auto syncs = syncPointsOf(CVPipeline::getOpCoreType(op));
  if (syncs.empty()) {
    return nullptr;
  }
  auto it = std::upper_bound(
      syncs.begin(), syncs.end(), positionOf(op),
      [&, this](unsigned v, Operation *o) { return v <= positionOf(o); });
  auto block = op->getBlock();
  if (it != syncs.begin()) {
    it--;
    return (*it)->getBlock() == op->getBlock() ? *it : nullptr;
  }
  return nullptr;
}

Operation *SyncWall::getNextSyncOpInSameBlock(Operation *op) const {
  auto syncs = syncPointsOf(CVPipeline::getOpCoreType(op));
  if (syncs.empty()) {
    return nullptr;
  }
  auto it = std::upper_bound(
      syncs.begin(), syncs.end(), positionOf(op),
      [this](unsigned v, Operation *o) { return v < positionOf(o); });

  for (; it != syncs.end(); it++) {
    if ((*it)->getBlock() == op->getBlock()) {
      return *it;
    }
  }
  return nullptr;
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
  auto it = std::upper_bound(
      syncs.begin(), syncs.end(), lo,
      [this](unsigned v, Operation *o) { return v < positionOf(o); });
  return it != syncs.end() && positionOf(*it) < hi;
}

unsigned SyncWall::segmentOf(Operation *op) const {
  unsigned pos = positionOf(op);
  auto core = CVPipeline::getOpCoreType(op);
  const llvm::SmallVector<unsigned> *prefix = nullptr;
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

ArrayRef<Operation *> SyncWall::syncPointsOf(CoreType core) const {
  return core == CoreType::VECTOR_ONLY ? vector.syncPoints : cube.syncPoints;
}

bool SyncWall::isSyncPoint(Operation *op, CoreType core) const {
  auto contains = [&](const llvm::SmallVector<Operation *> &syncs,
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
