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
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <utility>

static constexpr const char *DEBUG_TYPE = "merge-small-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

static constexpr int MIN_VF_SIZE = 3;

static int cntCalcuateOps(llvm::SmallVector<Operation *> ops) {
  int count = 0;
  for (Operation *op : ops) {
    if(isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::EmptyOp>(op)) {
      continue;
    }
    bool allOperandsTensor = llvm::all_of(op->getOperands(), [](Value operand) {
      return isa<RankedTensorType>(operand.getType());
    });
    bool allResultsTensor = llvm::all_of(op->getResults(), [](Value result) {
      return isa<RankedTensorType>(result.getType());
    });
    if (!allOperandsTensor || !allResultsTensor) {
      continue;
    }
    count++;
  }
  return count;
}

static bool isShapeChangeOp(Operation *op) {
  return isa<linalg::BroadcastOp, linalg::ReduceOp>(op);
}

static bool isConnectionFrom(Operation *op, int blockId, Block *block,
                             CVPipeline::ComputeBlockIdManager &bm) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      continue;
    }
    Operation *defInBlock = CVPipeline::getAncestorInBlock(defOp, block);
    if (!defInBlock) {
      continue;
    }
    if (bm.getBlockIdByOp(defInBlock) == blockId) {
      return true;
    }
  }
  return false;
}

static bool isConnectionTo(Operation *op, int blockId, Block *block,
                           CVPipeline::ComputeBlockIdManager &bm) {
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      Operation *userInBlock = CVPipeline::getAncestorInBlock(user, block);
      if (!userInBlock) {
        continue;
      }
      if (bm.getBlockIdByOp(userInBlock) == blockId) {
        return true;
      }
    }
  }
  return false;
}

static std::optional<int>
getUpBlock(int nowBlockId, Block *block,
           CVPipeline::ComputeBlockIdManager &bm) {
  auto ops = bm.getOpsByBlockId(nowBlockId);
  llvm::SmallDenseSet<int> upstreamIds;
  llvm::SmallVector<std::pair<Operation*, Operation *>> boundaryPairs;
  bool hasCubeDef = false;

  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        continue;
      }
      Operation *defInBlock = CVPipeline::getAncestorInBlock(defOp, block);
      if (!defInBlock) {
        continue;
      }
      if (CVPipeline::getOpCoreType(defInBlock) != CVPipeline::CoreType::VECTOR_ONLY) {
        hasCubeDef = true;
        break;
      }
      int bid = bm.getBlockIdByOp(defInBlock);
      if (bid != -1 && bid != nowBlockId) {
        upstreamIds.insert(bid);
        boundaryPairs.push_back({defInBlock, op});
      }
    }
  }

  if (hasCubeDef) {
    LOG_DEBUG("[getUpBlock] upstream has cube, skip");
    return std::nullopt;
  }

  if (upstreamIds.size() != 1) {
    LOG_DEBUG("[getUpBlock] upstreamIds is not only one ("<<upstreamIds.size()<< "), skip");
    return std::nullopt;
  }

  if (llvm::all_of(boundaryPairs, [&](std::pair<Operation*, Operation *> pr) {
        auto defOp = pr.first;
        auto userOp = pr.second;
        return isShapeChangeOp(defOp) || isShapeChangeOp(userOp);
      })) {
    LOG_DEBUG("[getUpBlock] all boundary ops are shape-change, skip");
    return std::nullopt;  
  }

  return *upstreamIds.begin();
}

static std::optional<int>
getDownBlock(int nowBlockId, Block *block, DenseMap<int, int> id2order,
             CVPipeline::ComputeBlockIdManager &bm) {
  auto ops = bm.getOpsByBlockId(nowBlockId);
  llvm::SmallDenseSet<int> downstreamIds;
  llvm::SmallVector<std::pair<Operation*, Operation *>> boundaryPairs;
  bool hasCubeUser = false;

  for (Operation *op : ops) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        Operation *userInBlock = CVPipeline::getAncestorInBlock(user, block);
        if (!userInBlock || (block->mightHaveTerminator() && userInBlock == block->getTerminator())) {
          continue;
        }
        if (CVPipeline::getOpCoreType(userInBlock) != CVPipeline::CoreType::VECTOR_ONLY) {
          hasCubeUser = true;
          break;
        }
        int bid = bm.getBlockIdByOp(userInBlock);
        if (bid != -1 && bid != nowBlockId) {
          downstreamIds.insert(bid);
          boundaryPairs.push_back({op, userInBlock});
        }
      }
    }
  }

  if (hasCubeUser) {
    LOG_DEBUG("[getDownBlock] downstream has cube, skip");
    return std::nullopt;
  }

  if (downstreamIds.size() < 1) {
    LOG_DEBUG("[getDownBlock] no downstream, skip");
    return std::nullopt;
  }

  int retBlockId = -1;
  for (auto id: downstreamIds) {
    if (retBlockId == -1 || (id2order.contains(id) && id2order[id] < id2order[retBlockId])) {
      retBlockId = id;
    }
  }

  if (llvm::all_of(boundaryPairs, [&](std::pair<Operation*, Operation *> pr) {
        auto defOp = pr.first;
        auto userOp = pr.second;
        if(bm.getBlockIdByOp(userOp) != retBlockId) {
          return false;
        }
        return isShapeChangeOp(defOp) || isShapeChangeOp(userOp);
      })) {
    LOG_DEBUG("[getDownBlock] all boundary ops are shape-change, skip");
    return std::nullopt;  
  }

  return retBlockId;
}

static SmallVector<int>
getBlockIdsInProgramOrder(Block *block,
                          CVPipeline::ComputeBlockIdManager &bm, SmallVector<int> &ordered, DenseMap<int, int> &id2order) {
  ordered.clear();
  id2order.clear();
  llvm::SmallDenseSet<int, 4> seen;
  for (Operation &op : *block) {
    int bid = bm.getBlockIdByOp(&op);
    if (bid != -1 && seen.insert(bid).second) {
      id2order[bid] = ordered.size();
      ordered.push_back(bid);
    }
  }
  return ordered;
}

} // namespace

namespace mlir {
namespace triton {

class MergeSmallBlockPass
    : public PassWrapper<MergeSmallBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeSmallBlockPass)

  MergeSmallBlockPass() = default;

  StringRef getArgument() const override { return "merge-small-block"; }

  StringRef getDescription() const override {
    return "Merge small compute blocks (<= 3 ops) into their operand or user "
           "block";
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

    llvm::SmallVector<Block *> blocks;
    module.walk([&](Block *block) { blocks.push_back(block); });

    for (Block *block : blocks) {
      SmallVector<int> orderedBlockIds;
      DenseMap<int, int> id2order;
      getBlockIdsInProgramOrder(block, bm, orderedBlockIds, id2order);

      for (int nowBlockId : orderedBlockIds) {
        LOG_DEBUG("Processing block " << nowBlockId );
        auto ops = bm.getOpsByBlockId(nowBlockId);
        if (ops.empty() || cntCalcuateOps(ops) > MIN_VF_SIZE) {
          continue;
        }
        if (CVPipeline::getOpCoreType(*ops.begin()) != CVPipeline::CoreType::VECTOR_ONLY) {
          continue;
        }

        LOG_DEBUG("Processing small block " << nowBlockId );
        for (auto op: ops) {
          LOG_DEBUG("op:"<< *op );
        }

        auto upBlock = getUpBlock(nowBlockId, block, bm);
        if (upBlock.has_value()) {
          if (CVPipeline::willCreateCycle(ops, memGraph, upBlock.value(), bm)) {
            LOG_DEBUG("would create cycle, skip\n");
          } else {
            LOG_DEBUG("Merging block " << nowBlockId << " into upstream block "
                                       << upBlock.value());
            for (Operation *op : ops) {
              bm.updateBlockId(op, upBlock.value());
            }
            continue;
          }
        }

        auto downBlock = getDownBlock(nowBlockId, block, id2order, bm);
        if (downBlock.has_value()) {
          if (CVPipeline::willCreateCycle(ops, memGraph, downBlock.value(), bm)) {
            LOG_DEBUG("would create cycle, skip\n");
            continue;
          }
          LOG_DEBUG("Merging downstream block " << nowBlockId
                                                << " into block "
                                                << downBlock.value());
          for (Operation *op : ops) {
            bm.updateBlockId(op, downBlock.value());
          }
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMergeSmallBlockPass() {
  return std::make_unique<MergeSmallBlockPass>();
}

void registerMergeSmallBlockPass() {
  PassRegistration<MergeSmallBlockPass> reg;
}

} // namespace triton
} // namespace mlir
