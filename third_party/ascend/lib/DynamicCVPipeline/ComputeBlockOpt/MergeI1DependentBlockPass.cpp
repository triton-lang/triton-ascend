/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "merge-i1-dependent-block"

using namespace mlir;
using namespace triton;

namespace {

class MergeI1DependentBlockPass
    : public PassWrapper<MergeI1DependentBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeI1DependentBlockPass)

  StringRef getArgument() const override { return "merge-i1-dependent-block"; }

  StringRef getDescription() const override {
    return "Merge compute blocks connected by i1 tensor dependencies";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    CVPipeline::ComputeBlockIdManager bm(module);
    llvm::SetVector<std::pair<Operation *, Operation *>> dependencies;
    module.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        auto tensorType = dyn_cast<TensorType>(operand.getType());
        if (!tensorType || !tensorType.getElementType().isInteger(1)) {
          continue;
        }
        Operation *producer = operand.getDefiningOp();
        if (!producer || bm.getBlockIdByOp(producer) == -1) {
          continue;
        }
        // A captured tensor belongs to the enclosing compute block at the
        // producer's level. Do not merge across unassigned control flow.
        Operation *consumer =
            CVPipeline::getAncestorInBlock(op, producer->getBlock());
        if (!consumer || consumer->hasTrait<OpTrait::IsTerminator>() ||
            bm.getBlockIdByOp(consumer) == -1 ||
            bm.isSameBlock(producer, consumer)) {
          continue;
        }
        auto coreType = CVPipeline::getCoreTypeOfSimpleOpOrCf(producer);
        if ((coreType != CVPipeline::CoreType::VECTOR_ONLY &&
             coreType != CVPipeline::CoreType::CUBE_ONLY) ||
            coreType != CVPipeline::getCoreTypeOfSimpleOpOrCf(consumer)) {
          continue;
        }
        dependencies.insert({producer, consumer});
      }
    });

    if (dependencies.empty()) {
      return;
    }
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);

    // A later merge may remove an intervening block that prevented an earlier
    // merge. Revisit the edges until no more whole blocks can be fused.
    bool changed;
    do {
      changed = false;
      for (auto [producer, consumer] : dependencies) {
        int producerId = bm.getBlockIdByOp(producer);
        int consumerId = bm.getBlockIdByOp(consumer);
        if (producerId == consumerId) {
          continue;
        }
        auto ops = bm.getOpsByBlockId(consumerId);
        if (CVPipeline::willCreateCycle(ops, memGraph, producerId, bm)) {
          continue;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "[" << DEBUG_TYPE << "] Merging block " << consumerId
                   << " into " << producerId << "\n");
        for (Operation *member : ops) {
          bm.updateBlockId(member, producerId);
        }
        changed = true;
      }
    } while (changed);
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeI1DependentBlockPass() {
  return std::make_unique<MergeI1DependentBlockPass>();
}

} // namespace triton
} // namespace mlir
