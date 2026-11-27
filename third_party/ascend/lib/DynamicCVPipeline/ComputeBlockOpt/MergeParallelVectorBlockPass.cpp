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

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

static constexpr const char *DEBUG_TYPE = "merge-parallel-vector-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

using BlockIdToOps = llvm::DenseMap<int, llvm::SmallVector<Operation *>>;

// Group the direct top-level ops of `body` by their compute block id. Nested
// region ops are attributed to their enclosing top-level op; crossing values
// are handled in getSsaBlockDeps via getAncestorInBlock.
BlockIdToOps collectBodyBlocks(Block *body) {
  BlockIdToOps blocks;
  for (Operation &op : *body) {
    auto blockId = CVPipeline::getOpBlockId(&op);
    if (!blockId || *blockId <= 0)
      continue;
    blocks[*blockId].push_back(&op);
  }
  return blocks;
}

// The set of other compute blocks that `blockId` depends on through SSA.
llvm::DenseSet<int> getSsaBlockDeps(Block *body, const BlockIdToOps &blocks,
                                    int blockId) {
  llvm::DenseSet<int> deps;
  auto it = blocks.find(blockId);
  if (it == blocks.end())
    return deps;
  for (Operation *op : it->second) {
    op->walk([&](Operation *inner) {
      for (Value operand : inner->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        Operation *defInBody = CVPipeline::getAncestorInBlock(def, body);
        if (!defInBody)
          continue;
        auto defBlockId = CVPipeline::getOpBlockId(defInBody);
        if (defBlockId && *defBlockId > 0 && *defBlockId != blockId)
          deps.insert(*defBlockId);
      }
    });
  }
  return deps;
}

CVPipeline::CoreType getBlockCoreType(const BlockIdToOps &blocks, int blockId) {
  auto it = blocks.find(blockId);
  if (it == blocks.end() || it->second.empty())
    return CVPipeline::CoreType::UNDETERMINED;
  return CVPipeline::getOpCoreType(it->second.front());
}

// Every op of the block must be an elementwise tensor compute op.
bool isElementwiseBlock(const BlockIdToOps &blocks, int blockId) {
  auto it = blocks.find(blockId);
  if (it == blocks.end())
    return false;
  return llvm::all_of(it->second, [](Operation *op) {
    return CVPipeline::isTensorComputeOp(op) &&
           op->hasTrait<OpTrait::Elementwise>();
  });
}

// All compute-block consumers of `producerId` must be `consumerId`. Uses inside
// the producer itself and the loop terminator (yield) are ignored.
bool isOnlyUsedBy(Block *body, const BlockIdToOps &blocks, int producerId,
                  int consumerId) {
  auto it = blocks.find(producerId);
  if (it == blocks.end())
    return true;
  for (Operation *op : it->second) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        Operation *userInBody = CVPipeline::getAncestorInBlock(user, body);
        if (!userInBody)
          return false;
        if (userInBody == body->getTerminator())
          continue;
        auto userBlockId = CVPipeline::getOpBlockId(userInBody);
        if (userBlockId && *userBlockId > 0 && *userBlockId != producerId &&
            *userBlockId != consumerId)
          return false;
      }
    }
  }
  return true;
}

struct StagePair {
  int producerId;
  int elementwiseId;
};

// Match, within one `scf.for` body:
//   A: independent VECTOR block whose only compute-block user is B.
//   C: independent CUBE block whose only compute-block user is B.
//   B: all-elementwise VECTOR block depending on exactly A and C.
// Returns every distinct (A, B) pair found in the loop.
llvm::SmallVector<StagePair> findStagePairs(Block *body) {
  llvm::SmallVector<StagePair> pairs;
  BlockIdToOps blocks = collectBodyBlocks(body);
  for (auto &entry : blocks) {
    int bId = entry.first;
    if (getBlockCoreType(blocks, bId) != CVPipeline::CoreType::VECTOR_ONLY)
      continue;
    if (!isElementwiseBlock(blocks, bId))
      continue;

    llvm::DenseSet<int> deps = getSsaBlockDeps(body, blocks, bId);
    if (deps.size() != 2)
      continue;

    int vectorProducer = -1;
    int cubeProducer = -1;
    for (int dep : deps) {
      CVPipeline::CoreType depType = getBlockCoreType(blocks, dep);
      if (depType == CVPipeline::CoreType::VECTOR_ONLY && vectorProducer == -1)
        vectorProducer = dep;
      else if (depType == CVPipeline::CoreType::CUBE_ONLY && cubeProducer == -1)
        cubeProducer = dep;
      else
        vectorProducer = -1; // duplicate or unexpected core type: reject B.
    }
    if (vectorProducer == -1 || cubeProducer == -1)
      continue;

    if (!getSsaBlockDeps(body, blocks, vectorProducer).empty() ||
        !getSsaBlockDeps(body, blocks, cubeProducer).empty())
      continue;
    if (!isOnlyUsedBy(body, blocks, vectorProducer, bId) ||
        !isOnlyUsedBy(body, blocks, cubeProducer, bId))
      continue;

    pairs.push_back({vectorProducer, bId});
  }
  return pairs;
}

void processFor(scf::ForOp forOp,
                const CVPipeline::MemoryDependenceGraph &memGraph,
                CVPipeline::ComputeBlockIdManager &bm) {
  Block *body = forOp.getBody();
  llvm::SmallVector<StagePair> pairs = findStagePairs(body);

  if (pairs.size() != 1) {
    LOG_DEBUG("for has " << pairs.size() << " stage pair(s), skip");
    return;
  }

  int vectorProducer = pairs.front().producerId;
  int bId = pairs.front().elementwiseId;
  llvm::SmallVector<Operation *> opsToUnify = bm.getOpsByBlockId(bId);

  if (CVPipeline::willCreateCycle(opsToUnify, memGraph, vectorProducer, bm)) {
    LOG_DEBUG("merging block " << bId << " into " << vectorProducer
                               << " would create a cycle, skip");
    return;
  }

  for (Operation *op : opsToUnify)
    bm.updateBlockId(op, vectorProducer);
  LOG_DEBUG("merged elementwise block " << bId << " into " << vectorProducer);
}

} // namespace

class MergeParallelVectorBlockPass
    : public PassWrapper<MergeParallelVectorBlockPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeParallelVectorBlockPass)

  MergeParallelVectorBlockPass() = default;

  StringRef getArgument() const override {
    return "merge-parallel-vector-block";
  }

  StringRef getDescription() const override {
    return "Merge an independent elementwise VECTOR block into the VECTOR "
           "block it feeds so both share the same pipeline stage";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (CVPipeline::hasFallbackAttr(module))
      return;

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);

    // This pass only rewrites block_id attributes, so mutating during the walk
    // does not invalidate it.
    module.walk([&](scf::ForOp forOp) { processFor(forOp, memGraph, bm); });
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeParallelVectorBlockPass() {
  return std::make_unique<MergeParallelVectorBlockPass>();
}

} // namespace triton
} // namespace mlir
