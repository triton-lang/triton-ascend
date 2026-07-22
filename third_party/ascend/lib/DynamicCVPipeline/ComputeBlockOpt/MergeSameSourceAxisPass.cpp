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

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-same-source-axis";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;

namespace mlir {
namespace triton {

namespace {

// Walk SSA def-use forward from start, collecting VECTOR_ONLY ops. Cross-block
// by design — convergence is allowed to span physical blocks.
static void collectForwardReach(Operation *start,
                                llvm::DenseSet<Operation *> &reach) {
  if (!start) {
    return;
  }
  SmallVector<Operation *> workList;
  workList.push_back(start);
  reach.insert(start);
  for (size_t idx = 0; idx < workList.size(); ++idx) {
    for (Operation *user : workList[idx]->getUsers()) {
      if (CVPipeline::getOpCoreType(user) !=
          CVPipeline::CoreType::VECTOR_ONLY) {
        continue;
      }
      if (reach.insert(user).second) {
        workList.push_back(user);
      }
    }
  }
}

// DFS along VECTOR_ONLY ops from start to target. On success, path ends with
// target. visited/path are caller-owned and reused across calls.
static bool dfsForwardPath(Operation *start, Operation *target,
                           llvm::DenseSet<Operation *> &visited,
                           SmallVectorImpl<Operation *> &path) {
  if (!visited.insert(start).second) {
    return false;
  }
  path.push_back(start);
  if (start == target) {
    return true;
  }
  for (Operation *user : start->getUsers()) {
    if (CVPipeline::getOpCoreType(user) != CVPipeline::CoreType::VECTOR_ONLY) {
      continue;
    }
    if (dfsForwardPath(user, target, visited, path)) {
      return true;
    }
  }
  path.pop_back();
  return false;
}

// Find the closest downstream op reachable from both cons1 and cons2 (BFS
// from cons2 picks the convergent op nearest to cons2, minimizing chain
// length), then dedup-merge the cons1→K and cons2→K paths with K at the tail.
static bool tryBuildConvergentChain(Operation *cons1, Operation *cons2,
                                    SmallVectorImpl<Operation *> &chainOps) {
  if (!cons1 || !cons2 || cons1 == cons2) {
    return false;
  }

  llvm::DenseSet<Operation *> reachFromCons1;
  collectForwardReach(cons1, reachFromCons1);
  llvm::DenseSet<Operation *> reachFromCons2;
  collectForwardReach(cons2, reachFromCons2);

  Operation *convergentOp = nullptr;
  SmallVector<Operation *> bfsQueue;
  llvm::DenseSet<Operation *> visited;
  bfsQueue.push_back(cons2);
  visited.insert(cons2);
  for (size_t i = 0; i < bfsQueue.size(); ++i) {
    Operation *cur = bfsQueue[i];
    if (cur != cons1 && cur != cons2 && reachFromCons1.count(cur)) {
      convergentOp = cur;
      break;
    }
    for (Operation *user : cur->getUsers()) {
      if (CVPipeline::getOpCoreType(user) !=
          CVPipeline::CoreType::VECTOR_ONLY) {
        continue;
      }
      if (visited.insert(user).second) {
        bfsQueue.push_back(user);
      }
    }
  }
  if (!convergentOp || CVPipeline::getOpCoreType(convergentOp) !=
                           CVPipeline::CoreType::VECTOR_ONLY) {
    return false;
  }

  llvm::DenseSet<Operation *> visitedForCons1;
  SmallVector<Operation *> pathFromCons1;
  if (!dfsForwardPath(cons1, convergentOp, visitedForCons1, pathFromCons1)) {
    return false;
  }
  llvm::DenseSet<Operation *> visitedForCons2;
  SmallVector<Operation *> pathFromCons2;
  if (!dfsForwardPath(cons2, convergentOp, visitedForCons2, pathFromCons2)) {
    return false;
  }

  chainOps.clear();
  llvm::DenseSet<Operation *> inChain;
  for (Operation *op : pathFromCons1) {
    if (op != convergentOp && inChain.insert(op).second) {
      chainOps.push_back(op);
    }
  }
  for (Operation *op : pathFromCons2) {
    if (op != convergentOp && inChain.insert(op).second) {
      chainOps.push_back(op);
    }
  }
  chainOps.push_back(convergentOp);
  return true;
}

// For each pair of source's direct consumers (cons1, cons2), build a convergent
// chain and rewrite its block_id to source's if safe (no cycle). At most one
// merge per source — multi-pair rewrites can churn `bm` state unpredictably.
static void tryMergeSource(Operation *source,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           CVPipeline::ComputeBlockIdManager &bm) {
  auto srcBlockIdOpt = CVPipeline::getOpBlockId(source);
  if (!srcBlockIdOpt || *srcBlockIdOpt < 0) {
    return;
  }
  int srcBlockId = *srcBlockIdOpt;

  // Don't pre-filter by consumer block: the convergent op may sit in a
  // different block from cons1/cons2.
  SmallVector<Operation *> consumers;
  for (Operation *user : source->getUsers()) {
    if (CVPipeline::getOpCoreType(user) != CVPipeline::CoreType::VECTOR_ONLY) {
      continue;
    }
    auto bidOpt = CVPipeline::getOpBlockId(user);
    if (!bidOpt || *bidOpt == srcBlockId) {
      continue;
    }
    consumers.push_back(user);
  }
  if (consumers.size() < 2) {
    return;
  }

  bool merged = false;
  for (size_t i = 0; i < consumers.size() && !merged; ++i) {
    for (size_t j = i + 1; j < consumers.size() && !merged; ++j) {
      Operation *cons1 = consumers[i];
      Operation *cons2 = consumers[j];
      SmallVector<Operation *> chainOps;
      if (!tryBuildConvergentChain(cons1, cons2, chainOps)) {
        continue;
      }
      LOG_DEBUG("[tryMergeSource] candidate source="
                << *source << " cons1=" << *cons1 << " cons2=" << *cons2
                << " chainSize=" << chainOps.size());

      if (CVPipeline::willCreateCycle(chainOps, memGraph, srcBlockId, bm)) {
        LOG_DEBUG("[tryMergeSource] willCreateCycle, skip");
        continue;
      }

      for (Operation *op : chainOps) {
        bm.updateBlockId(op, srcBlockId);
      }
      LOG_DEBUG("[tryMergeSource] merged "
                << chainOps.size() << " ops into block_id=" << srcBlockId);
      merged = true;
    }
  }
}

} // anonymous namespace

class MergeSameSourceAxisPass
    : public PassWrapper<MergeSameSourceAxisPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeSameSourceAxisPass)

  MergeSameSourceAxisPass() = default;

  StringRef getArgument() const override { return "merge-same-source-axis"; }

  StringRef getDescription() const override {
    return "Rewrite ssbuffer.block_id so that >=2 vector consumers of a "
           "source op whose downstream converges at a common op end up in "
           "the same block as the source. Only VECTOR core ops are considered.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    LOG_DEBUG("Before: " << *module);
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    auto bm = CVPipeline::ComputeBlockIdManager(module);

    // Walk in program order so the same convergence isn't re-detected from
    // every direction.
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (CVPipeline::getOpCoreType(op) != CVPipeline::CoreType::VECTOR_ONLY) {
        return;
      }
      if (!CVPipeline::getOpBlockId(op)) {
        return;
      }
      if (op->getUsers().empty()) {
        return;
      }
      candidates.push_back(op);
    });

    for (Operation *source : candidates) {
      tryMergeSource(source, memGraph, bm);
    }

    LOG_DEBUG("After: " << *module);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMergeSameSourceAxisPass() {
  return std::make_unique<MergeSameSourceAxisPass>();
}

void registerMergeSameSourceAxisPass() {
  PassRegistration<MergeSameSourceAxisPass> reg;
}

} // namespace triton
} // namespace mlir
