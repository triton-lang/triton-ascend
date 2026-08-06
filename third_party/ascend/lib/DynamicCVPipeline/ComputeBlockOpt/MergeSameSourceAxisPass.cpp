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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-same-source-axis";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;

namespace mlir {
namespace triton {

namespace {

static bool findNearestConvergence(ArrayRef<Operation *> consumers,
                                   SmallVectorImpl<Operation *> &chainOps) {
  if (consumers.size() < 2) {
    return false;
  }

  llvm::DenseMap<Operation *, int> firstIdx;
  llvm::DenseMap<Operation *, Operation *> parent;
  SmallVector<std::pair<Operation *, int>> bfsQueue;

  for (size_t consumerIndex = 0; consumerIndex < consumers.size();
       ++consumerIndex) {
    Operation *consumer = consumers[consumerIndex];
    auto ins = firstIdx.insert({consumer, (int)consumerIndex});
    if (ins.second) {
      parent[consumer] = nullptr;
      bfsQueue.push_back({consumer, (int)consumerIndex});
    }
  }

  for (size_t queueIndex = 0; queueIndex < bfsQueue.size(); ++queueIndex) {
    Operation *cur = bfsQueue[queueIndex].first;
    int myIdx = bfsQueue[queueIndex].second;
    for (Operation *user : cur->getUsers()) {
      if (CVPipeline::getOpCoreType(user) !=
          CVPipeline::CoreType::VECTOR_ONLY) {
        continue;
      }
      auto it = firstIdx.find(user);
      if (it == firstIdx.end()) {
        firstIdx[user] = myIdx;
        parent[user] = cur;
        bfsQueue.push_back({user, myIdx});
      } else if (it->second != myIdx) {
        // Reject 1-step false convergence when `user` is itself a starting
        // consumer — adjacent siblings are not a real two-branch merge.
        if (llvm::is_contained(consumers, user)) {
          continue;
        }
        Operation *cons1 = consumers[it->second];
        Operation *cons2 = consumers[myIdx];
        Operation *convergenceOp = user;

        SmallVector<Operation *> p1;
        for (Operation *pathOp = convergenceOp; pathOp;
             pathOp = parent[pathOp]) {
          p1.push_back(pathOp);
          if (pathOp == cons1) {
            break;
          }
        }
        std::reverse(p1.begin(), p1.end());

        SmallVector<Operation *> p2;
        for (Operation *pathOp = cur; pathOp; pathOp = parent[pathOp]) {
          p2.push_back(pathOp);
          if (pathOp == cons2) {
            break;
          }
        }
        std::reverse(p2.begin(), p2.end());
        p2.push_back(convergenceOp);

        chainOps.clear();
        llvm::DenseSet<Operation *> inChain;
        for (Operation *op : p1) {
          if (op != convergenceOp && inChain.insert(op).second) {
            chainOps.push_back(op);
          }
        }
        for (Operation *op : p2) {
          if (op != convergenceOp && inChain.insert(op).second) {
            chainOps.push_back(op);
          }
        }
        chainOps.push_back(convergenceOp);

        // Pull the convergence op's same-block VECTOR_ONLY direct users
        // along so the moved convergence op doesn't leave its tail behind.
        auto convBlockIdOpt = CVPipeline::getOpBlockId(convergenceOp);
        if (convBlockIdOpt && *convBlockIdOpt >= 0) {
          int convBlockId = *convBlockIdOpt;
          for (Operation *user : convergenceOp->getUsers()) {
            if (CVPipeline::getOpCoreType(user) !=
                CVPipeline::CoreType::VECTOR_ONLY) {
              continue;
            }
            auto userBidOpt = CVPipeline::getOpBlockId(user);
            if (!userBidOpt || *userBidOpt != convBlockId) {
              continue;
            }
            if (inChain.insert(user).second) {
              chainOps.push_back(user);
            }
          }
        }
        return true;
      }
    }
  }
  return false;
}

static void tryMergeSource(Operation *source,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           CVPipeline::ComputeBlockIdManager &bm) {
  auto srcBlockIdOpt = CVPipeline::getOpBlockId(source);
  if (!srcBlockIdOpt || *srcBlockIdOpt < 0) {
    return;
  }
  int srcBlockId = *srcBlockIdOpt;

  // Skip constants as merge sources: they have no axis semantics and their
  // many users routinely trigger trivial 1-step BFS convergence.
  if (mlir::isa<mlir::arith::ConstantOp>(source)) {
    return;
  }

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

  SmallVector<Operation *> chainOps;
  if (!findNearestConvergence(consumers, chainOps)) {
    return;
  }
  LOG_DEBUG("[tryMergeSource] candidate source=" << *source << " chainSize="
                                                 << chainOps.size());

  if (CVPipeline::willCreateCycle(chainOps, memGraph, srcBlockId, bm)) {
    LOG_DEBUG("[tryMergeSource] willCreateCycle, skip");
    return;
  }

  for (Operation *op : chainOps) {
    bm.updateBlockId(op, srcBlockId);
  }
  LOG_DEBUG("[tryMergeSource] merged " << chainOps.size()
                                       << " ops into block_id=" << srcBlockId);
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
