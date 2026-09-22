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
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

static constexpr const char *DEBUG_TYPE = "merge-high-fanout-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

static constexpr int kMinCrossBlockEdgeCount = 16;

using namespace mlir;

namespace mlir {
namespace triton {

namespace {

struct ComputeBlock {
  int id;
  CVPipeline::CoreType coreType;
  SmallVector<Operation *> ops;
};

/// Group ops by block_id and build a cross-block source dependency map.
///
/// computeBlocks:        block_id → ComputeBlock
/// crossBlockSources:    srcBlockId → dstBlockId → set of distinct source ops
///                       in srcBlockId that dstBlockId's ops depend on via SSA
///                       use-def chains.
static void
buildBlockDependencyGraph(Block *block,
                          DenseMap<int, ComputeBlock> &computeBlocks,
                          DenseMap<int, DenseMap<int, DenseSet<Operation *>>>
                              &crossBlockSources) {
  computeBlocks.clear();
  crossBlockSources.clear();

  block->walk([&](Operation *op) {
    if(op->getBlock()!=block) {
      return;
    }
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return;
    }
    auto optId = CVPipeline::getOpBlockId(op);
    if (!optId.has_value()) {
      return;
    }
    int bid = *optId;
    if (!computeBlocks.contains(bid)) {
      computeBlocks[bid] = {bid, CVPipeline::getOpCoreType(op), {}};
    }
    computeBlocks[bid].ops.push_back(op);
  });

  if (computeBlocks.empty()) {
    return;
  }

  for (auto &kv : computeBlocks) {
    int curId = kv.first;
    for (Operation *op : kv.second.ops) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp) {
          continue;
        }
        if(defOp->getBlock() != block){
          continue;
        }
        // Operation *ancestor = CVPipeline::getAncestorInBlock(defOp, block);
        // if (!ancestor) {
        //   continue;
        // }
        auto ancIdOpt = CVPipeline::getOpBlockId(defOp);
        if (!ancIdOpt.has_value()) {
          continue;
        }
        int ancId = ancIdOpt.value();
        if (ancId == curId) {
          continue;
        }
        if ((kv.second.coreType != CVPipeline::CoreType::VECTOR_ONLY)) {
          continue;
        }
        crossBlockSources[ancId][curId].insert(defOp);
      }
    }
  }
}

/// Try to find and apply one merge within the given block.
///
/// Looks for a pair (producer=B, consumer=A) where more than
/// kMinCrossBlockEdgeCount distinct ops in B are depended upon by A's ops.
/// Merges A (consumer) into B (producer) without creating a cycle.
///
/// Returns true if a merge was applied, false otherwise.
static bool
tryMergeHighFanoutBlock(Block *block,
                        const CVPipeline::MemoryDependenceGraph &memGraph,
                        CVPipeline::ComputeBlockIdManager &bm) {
  DenseMap<int, ComputeBlock> computeBlocks;
  DenseMap<int, DenseMap<int, DenseSet<Operation *>>> crossBlockSources;
  buildBlockDependencyGraph(block, computeBlocks, crossBlockSources);

  // if (computeBlocks.size() < 2) {
  //   return false;
  // }

  struct MergeCandidate {
    int producerId;
    int consumerId;
    size_t edgeCount;
  };
  SmallVector<MergeCandidate> candidates;
  for (auto &[producerId, consumers] : crossBlockSources) {
    for (auto &[consumerId, sources] : consumers) {
      if (sources.size() > static_cast<size_t>(kMinCrossBlockEdgeCount)) {
        candidates.push_back({producerId, consumerId, sources.size()});
      }
    }
  }

  if (candidates.empty()) {
    return false;
  }

  // std::sort(candidates.begin(), candidates.end(),
  //           [](const MergeCandidate &a, const MergeCandidate &b) {
  //             return a.edgeCount > b.edgeCount;
  //           });

  for (const auto &cand : candidates) {
    auto it = computeBlocks.find(cand.consumerId);
    if (it == computeBlocks.end()) {
      continue;
    }
    SmallVector<Operation *> opsToMerge = it->second.ops;

    LOG_DEBUG("Trying merge consumer block "
              << cand.consumerId << " into producer block "
              << cand.producerId
              << " (edge count: " << cand.edgeCount << ")");

    if (CVPipeline::willCreateCycle(opsToMerge, memGraph, cand.producerId,
                                    bm)) {
      LOG_DEBUG("Merge would create cycle, skipping");
      continue;
    }

    for (Operation *op : opsToMerge) {
      bm.updateBlockId(op, cand.producerId);
    }
    LOG_DEBUG("Merged block " << cand.consumerId << " into block "
                              << cand.producerId);
    return true;
  }

  return false;
}

class MergeHighFanoutBlockPass
    : public PassWrapper<MergeHighFanoutBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeHighFanoutBlockPass)

  MergeHighFanoutBlockPass() = default;

  StringRef getArgument() const override { return "merge-high-fanout-block"; }

  StringRef getDescription() const override {
    return "Merge compute block A into block B when more than 16 of B's "
           "ops are depended upon by A's ops, without creating cycles.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    LOG_DEBUG("Before: " << *module);
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);

    module.walk([&](Block *block) {
      bool hasBlockIds = false;
      for (Operation &op : *block) {
        if (CVPipeline::getOpBlockId(&op).has_value()) {
          hasBlockIds = true;
          break;
        }
      }
      if (!hasBlockIds) {
        return;
      }

      while (tryMergeHighFanoutBlock(block, memGraph, bm)) {
      }
    });

    LOG_DEBUG("After: " << *module);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMergeHighFanoutBlockPass() {
  return std::make_unique<MergeHighFanoutBlockPass>();
}

void registerMergeHighFanoutBlockPass() {
  PassRegistration<MergeHighFanoutBlockPass> reg;
}

} // namespace triton
} // namespace mlir
