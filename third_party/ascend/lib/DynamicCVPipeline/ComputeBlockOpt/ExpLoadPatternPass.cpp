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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"
#include "DynamicCVPipeline/Common/ScalarClosure.h"

static constexpr const char *DEBUG_TYPE = "extract-exp-load-pattern";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

struct ExpLoadPattern {
  linalg::BroadcastOp broadcastOp;
  Operation *expOp = nullptr;
  /// Ops of the load unit feeding `exp`: the atomic `ssbuffer.load` scope when
  /// PackMemref packed it, otherwise the raw `alloc -> copy -> to_tensor`
  /// chain.
  SmallVector<Operation *> loadOps;
  arith::MulFOp mulfOp;
  int blockId = -1;
  int newBlockId = -1;
};

static bool isExpLike(Operation *op) {
  return isa_and_present<math::ExpOp, math::Exp2Op>(op);
}

/// Collects the ops of a load unit that PackMemref left unpacked, i.e. the raw
/// `alloc -> memref.copy -> to_tensor` chain behind `loadValue`. Fails unless
/// every op of the chain already sits in \p blockId, which is what lets the
/// whole unit travel to another block.
static bool collectRawLoadUnit(Value loadValue,
                               CVPipeline::ComputeBlockIdManager &bm,
                               int blockId,
                               SmallVectorImpl<Operation *> &loadOps) {
  auto toTensorOp = loadValue.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensorOp || bm.getBlockIdByOp(toTensorOp) != blockId) {
    return false;
  }

  auto allocOp = toTensorOp.getBuffer().getDefiningOp<memref::AllocOp>();
  if (!allocOp || bm.getBlockIdByOp(allocOp) != blockId) {
    return false;
  }

  for (Operation *user : allocOp->getUsers()) {
    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getTarget() != allocOp.getMemref()) {
      continue;
    }
    if (bm.getBlockIdByOp(copyOp) != blockId ||
        copyOp->getBlock() != toTensorOp->getBlock()) {
      return false;
    }
    loadOps.assign({toTensorOp.getOperation(), copyOp.getOperation(),
                    allocOp.getOperation()});
    return true;
  }
  return false;
}

// Match pattern:
//   mulf <- broadcast <- exp/exp2 <- load
//        ↖
//           %arg
// `load` is an `ssbuffer.load` scope after PackMemref, or the raw alloc/copy/
// to_tensor chain when the unit was left unpacked.
static bool matchExpLoadPattern(linalg::BroadcastOp broadcastOp,
                                CVPipeline::ComputeBlockIdManager &bm,
                                ExpLoadPattern &info) {
  if (CVPipeline::getOpCoreType(broadcastOp) !=
      CVPipeline::CoreType::VECTOR_ONLY) {
    return false;
  }
  int blockId = bm.getBlockIdByOp(broadcastOp);
  if (blockId == -1) {
    return false;
  }

  if (broadcastOp.getDpsInputs().empty()) {
    return false;
  }
  Operation *expOp = broadcastOp.getDpsInputs()[0].getDefiningOp();
  if (!isExpLike(expOp) || bm.getBlockIdByOp(expOp) != blockId ||
      !expOp->hasOneUse()) {
    return false;
  }

  // The load unit must sit in the pattern's block: the scalar closure below is
  // collected with `includeParent = false`, so every op of the pattern has to
  // be a direct child of it.
  Value loadValue = expOp->getOperand(0);
  Operation *loadCarrier = loadValue.getDefiningOp();
  if (!loadCarrier || loadCarrier->getBlock() != broadcastOp->getBlock() ||
      bm.getBlockIdByOp(loadCarrier) != blockId || !loadCarrier->hasOneUse()) {
    return false;
  }
  SmallVector<Operation *> loadOps;
  if (CVPipeline::isLoadScope(loadCarrier)) {
    loadOps = {loadCarrier};
  } else if (!collectRawLoadUnit(loadValue, bm, blockId, loadOps)) {
    return false;
  }

  if (!broadcastOp->hasOneUse()) {
    return false;
  }
  auto mulfOp = dyn_cast<arith::MulFOp>(*broadcastOp->getUsers().begin());
  if (!mulfOp) {
    return false;
  }

  int newBlockId = bm.getBlockIdByOp(mulfOp);
  if (newBlockId == blockId) {
    return false;
  }
  if (CVPipeline::getOpCoreType(mulfOp) != CVPipeline::CoreType::VECTOR_ONLY) {
    return false;
  }

  Value broadcastResult = broadcastOp->getResult(0);
  Value otherOperand =
      (mulfOp.getLhs() == broadcastResult) ? mulfOp.getRhs() : mulfOp.getLhs();
  if (!isa<BlockArgument>(otherOperand)) {
    return false;
  }

  info.broadcastOp = broadcastOp;
  info.expOp = expOp;
  info.loadOps = std::move(loadOps);
  info.mulfOp = mulfOp;
  info.blockId = blockId;
  info.newBlockId = newBlockId;
  return true;
}

} // namespace

class ExpLoadPatternPass
    : public PassWrapper<ExpLoadPatternPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpLoadPatternPass)

  ExpLoadPatternPass() = default;

  StringRef getArgument() const override { return "exp-load-pattern"; }

  StringRef getDescription() const override {
    return "Match load->exp->broadcast from a VECTOR block into a fresh "
           "VECTOR block";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);

    SmallVector<ExpLoadPattern> patterns;
    module.walk([&](linalg::BroadcastOp broadcastOp) {
      ExpLoadPattern info;
      if (matchExpLoadPattern(broadcastOp, bm, info)) {
        LOG_DEBUG("Matched exp-load pattern in block " << info.blockId);
        patterns.push_back(info);
      }
    });

    for (auto &info : patterns) {
      if (bm.getBlockIdByOp(info.broadcastOp) != info.blockId) {
        LOG_DEBUG("Stale pattern (block id changed), skip");
        continue;
      }

      SmallVector<Operation *> coreOps{info.broadcastOp, info.expOp};
      coreOps.append(info.loadOps);

      CVPipeline::ScalarClosure closure{info.broadcastOp->getBlock(), coreOps,
                                        false};
      closure.collect();

      llvm::SetVector<Operation *> matchedOps;
      for (auto *op : coreOps) {
        matchedOps.insert(op);
      }
      for (Operation *op : closure.getScalarOps()) {
        matchedOps.insert(op);
      }
      CVPipeline::cloneScalarOpsForCrossBlockUses(bm, matchedOps,
                                                  info.newBlockId);

      if (!CVPipeline::willCreateCycle(matchedOps.getArrayRef(), memGraph,
                                       info.newBlockId, bm)) {
        for (Operation *op : matchedOps) {
          // A packed unit carries its block id on the scope only; stamping the
          // body would break the scope atomicity the earlier passes rely on.
          if (CVPipeline::isLoadScope(op)) {
            bm.updateBlockId(op, info.newBlockId);
            continue;
          }
          bm.updateBlockIdWithInner(op, info.newBlockId);
        }

        LOG_DEBUG("Process exp-load pattern: block " << info.blockId << " -> "
                                                     << info.newBlockId);
      }
    }
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createExpLoadPatternPass() {
  return std::make_unique<ExpLoadPatternPass>();
}

} // namespace triton
} // namespace mlir
