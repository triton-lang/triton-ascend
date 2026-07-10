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
#include "DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-cube-for-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

static bool isCubeForOp(scf::ForOp forOp) {
  if (CVPipeline::getOpBlockId(forOp.getOperation()).has_value()) {
    return false;
  }
  auto attr = forOp->getAttrOfType<StringAttr>(CVPipeline::kCoreType);
  if (!attr) {
    return false;
  }
  llvm::SmallVector<llvm::StringRef, 4> parts;
  attr.getValue().split(parts, ',');
  if (parts.empty()) {
    return false;
  }
  for (llvm::StringRef part : parts) {
    if (part.trim() != "CUBE") {
      return false;
    }
  }
  return true;
}

/**
 * @brief Collect the block_ids of the cube ops that consume the loop results,
 *        ordered by their first appearance in the parent block.
 */
static SmallVector<int>
collectDownstreamCubeBlockIds(scf::ForOp forOp,
                              CVPipeline::ComputeBlockIdManager &bm) {
  Block *parent = forOp->getBlock();
  llvm::DenseSet<Operation *> userAnchors;
  for (Value res : forOp->getResults()) {
    for (Operation *user : res.getUsers()) {
      if (auto *anc = CVPipeline::getAncestorInBlock(user, parent)) {
        if (CVPipeline::getOpCoreType(anc) != CVPipeline::CoreType::CUBE_ONLY) {
          continue;
        }
        userAnchors.insert(anc);
      }
    }
  }

  SmallVector<int> ordered;
  llvm::SmallDenseSet<int, CVPipeline::INIT_SIZE> seen;
  for (Operation &op : *parent) {
    if (!userAnchors.contains(&op)) {
      continue;
    }
    int bid = bm.getBlockIdByOp(&op);
    if (bid == -1) {
      continue;
    }
    if (seen.insert(bid).second) {
      ordered.push_back(bid);
    }
  }
  return ordered;
}

static void applyMerge(scf::ForOp forOp, int target,
                       CVPipeline::ComputeBlockIdManager &bm) {
  bm.updateBlockId(forOp.getOperation(), target);
  forOp.getBody()->walk([&](Operation *op) {
    if (op == forOp.getOperation()) {
      return;
    }
    if (CVPipeline::getOpBlockId(op).has_value()) {
      bm.updateBlockId(op, target);
    }
  });
}

/**
 * @brief Try to merge one cube-loader scf.for with a downstream consumer block
 */
static void tryMergeCubeFor(scf::ForOp forOp,
                            const CVPipeline::MemoryDependenceGraph &memGraph,
                            CVPipeline::ComputeBlockIdManager &bm) {
  SmallVector<int> downstream = collectDownstreamCubeBlockIds(forOp, bm);
  if (downstream.empty()) {
    LOG_DEBUG("[tryMergeCubeFor] no downstream cube consumer block for "
              << *forOp);
    return;
  }

  SmallVector<Operation *> forOnly = {forOp.getOperation()};
  for (int bid : downstream) {
    if (!CVPipeline::willCreateCycle(forOnly, memGraph, bid, bm)) {
      LOG_DEBUG("[tryMergeCubeFor] merging loader for into downstream block_id "
                << bid);
      applyMerge(forOp, bid,
                 bm); // Only merge with the first consumer cube block.
      return;
    }
    LOG_DEBUG("[tryMergeCubeFor] downstream block_id "
              << bid << " would create a cycle, skip");
  }
}

} // anonymous namespace

class MergeCubeForBlockPass
    : public PassWrapper<MergeCubeForBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeCubeForBlockPass)

  MergeCubeForBlockPass() = default;

  StringRef getArgument() const override { return "merge-cube-for-block"; }

  StringRef getDescription() const override {
    return "Merge cube loader scf.for loops with their downstream consuming "
           "matmul compute block";
  }

  void runOnOperation() override {
    // Gated by the enable_cube_block_merge compile option so other scenarios
    // are unaffected unless the feature is explicitly turned on.
    if (!CVPipeline::isCubeBlockMergeEnabled()) {
      return;
    }

    ModuleOp module = getOperation();
    LOG_DEBUG("Before: " << *module);
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    auto bm = CVPipeline::ComputeBlockIdManager(module);

    llvm::SmallVector<scf::ForOp> cubeForOps;
    module.walk([&](scf::ForOp forOp) {
      if (isCubeForOp(forOp)) {
        cubeForOps.push_back(forOp);
      }
    });

    for (scf::ForOp forOp : cubeForOps) {
      tryMergeCubeFor(forOp, memGraph, bm);
    }

    LOG_DEBUG("After: " << *module);
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeCubeForBlockPass() {
  return std::make_unique<MergeCubeForBlockPass>();
}

void registerMergeCubeForBlockPass() {
  PassRegistration<MergeCubeForBlockPass> reg;
}

} // namespace triton
} // namespace mlir
