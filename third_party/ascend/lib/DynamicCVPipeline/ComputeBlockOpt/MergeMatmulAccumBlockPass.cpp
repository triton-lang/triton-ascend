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

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-matmul-accum-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

/**
 * @brief Collect the producer matmuls of an accumulation chain
 *
 * A matmul is in "accumulation state" when its bias (the `outs` accumulator)
 * is the result of another linalg.matmul, i.e.
 *     c1 = a0*b0 + c0,  c2 = a1*b1 + c1, ...
 * This function walks the bias chain backwards starting from @p matmulOp and
 * returns every producer matmul whose result is used directly as the bias of
 * the next matmul in the chain.
 *
 * Only the direct def-use pattern (a matmul result used as the bias of another
 * matmul in the same block) is considered; the walk stops as soon as the bias
 * is not produced by a matmul.
 *
 * @param matmulOp The consumer matmul to start from
 * @return SmallVector<Operation*> Producer matmuls, ordered from the nearest
 *         producer to the head of the chain
 */
static SmallVector<Operation *>
collectAccumProducers(linalg::MatmulOp matmulOp) {
  SmallVector<Operation *> producers;
  Value bias = matmulOp.getDpsInits()[0];
  while (auto *def = bias.getDefiningOp()) {
    auto producer = dyn_cast<linalg::MatmulOp>(def);
    if (!producer) {
      break;
    }
    if (producer->getBlock() != matmulOp->getBlock()) {
      break;
    }
    producers.push_back(producer);
    bias = producer.getDpsInits()[0];
  }
  return producers;
}

/**
 * @brief Try to merge an accumulation chain of matmuls into the consumer's
 * block
 *
 * The consumer matmul (the one whose bias is produced by another matmul) is
 * the "later" block: the whole block of every producer matmul is squashed into
 * the consumer's block_id. Only ops that live in the same MLIR block as the
 * consumer matmul are taken from the producer blocks. If the merge would create
 * a cycle in the block-level dependency graph, failure is returned so the
 * caller can report an error.
 *
 * @param matmulOp The consumer matmul
 * @param memGraph Memory dependence graph for cycle detection
 * @param bm Block-id manager used to query/update block ids
 * @return LogicalResult Returns failure if the merge would create a cycle,
 *         success otherwise
 */
static LogicalResult
tryMergeAccumChain(linalg::MatmulOp matmulOp,
                   const CVPipeline::MemoryDependenceGraph &memGraph,
                   CVPipeline::ComputeBlockIdManager &bm) {
  SmallVector<Operation *> producers = collectAccumProducers(matmulOp);
  if (producers.empty()) {
    return success();
  }

  int targetBlockId = bm.getBlockIdByOp(matmulOp.getOperation());
  if (targetBlockId == -1) {
    LOG_DEBUG("consumer matmul has no block_id, skip: " << *matmulOp);
    return success();
  }

  // Collect the whole block of every producer matmul (restricted to ops in the
  // same MLIR block as the consumer matmul).
  SmallVector<Operation *> opsToUnify;
  llvm::SmallDenseSet<int, CVPipeline::INIT_SIZE> seenProducerBlockIds;
  for (Operation *producer : producers) {
    int producerBlockId = bm.getBlockIdByOp(producer);
    if (producerBlockId == -1 || producerBlockId == targetBlockId ||
        !seenProducerBlockIds.insert(producerBlockId).second) {
      continue;
    }
    for (Operation *op : bm.getOpsByBlockId(producerBlockId)) {
      // getOpsByBlockId may return nested ops (a parent op and ops inside its
      // regions can share the same block_id). Only ops that live directly in
      // the same MLIR block as the consumer matmul belong to the block group
      // being merged; nested ops are skipped because they move together with
      // their containing op.
      if (op->getBlock() != matmulOp->getBlock()) {
        continue;
      }
      // Explicitly skip ops nested inside another op of the same block group
      // (parent and child sharing the same block_id).
      if (llvm::any_of(opsToUnify, [&](Operation *collected) {
            return collected->isAncestor(op) || op->isAncestor(collected);
          })) {
        continue;
      }
      opsToUnify.push_back(op);
    }
  }
  if (opsToUnify.empty()) {
    return success();
  }

  LOG_DEBUG("merge " << opsToUnify.size() << " op(s) of producer block(s) of "
                     << *matmulOp << " into block_id " << targetBlockId);
  for (Operation *op : opsToUnify) {
    LOG_DEBUG("op block_id=" << bm.getBlockIdByOp(op) << ": " << *op);
  }

  if (CVPipeline::willCreateCycle(opsToUnify, memGraph, targetBlockId, bm)) {
    LOG_DEBUG("merging would create a cycle, report error!");
    return failure();
  }

  for (Operation *op : opsToUnify) {
    bm.updateBlockId(op, targetBlockId);
  }
  LOG_DEBUG("successfully merged " << opsToUnify.size()
                                   << " op(s) into block_id " << targetBlockId);
  return success();
}

} // anonymous namespace

class MergeMatmulAccumBlockPass
    : public PassWrapper<MergeMatmulAccumBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeMatmulAccumBlockPass)

  MergeMatmulAccumBlockPass() = default;

  StringRef getArgument() const override { return "merge-matmul-accum-block"; }

  StringRef getDescription() const override {
    return "Merge matmuls in an accumulation chain (one matmul's output is "
           "another matmul's bias) into the consumer matmul's block";
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

    llvm::SmallVector<linalg::MatmulOp> matmulOps;
    module.walk(
        [&](linalg::MatmulOp matmulOp) { matmulOps.push_back(matmulOp); });

    for (linalg::MatmulOp matmulOp : matmulOps) {
      if (failed(tryMergeAccumChain(matmulOp, memGraph, bm))) {
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
        return;
      }
    }

    LOG_DEBUG("After: " << *module);
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeMatmulAccumBlockPass() {
  return std::make_unique<MergeMatmulAccumBlockPass>();
}

void registerMergeMatmulAccumBlockPass() {
  PassRegistration<MergeMatmulAccumBlockPass> reg;
}

} // namespace triton
} // namespace mlir
