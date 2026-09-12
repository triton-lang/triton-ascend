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

// Extract a load -> to_tensor -> exp/exp2 -> broadcast sequence that was
// over-fused into a single VECTOR compute block by PlanVectorBlock, and move it
// into a fresh VECTOR block.  The broadcast's user (mulf) must already live in
// a *different* block, with its other operand being a BlockArgument (loop
// iter-arg); only then is the split performed.  mulf itself is NOT moved.
//
// Scalar index ops (and memref view ops) in the load chain that are shared with
// ops remaining in the original block are identified via ScalarClosure and
// cloned by cloneScalarOpsForCrossBlockUses: the clone keeps the original
// block_id, the original follows the extracted ops into the new block.
//
// This removes the "straddle" where one VECTOR block feeds both a CUBE matmul
// fixpipe chain (via extf) and a VECTOR softmax chain (via exp2), which would
// otherwise make MergeSmallBlock reject merging the softmax block due to a
// cycle through the CUBE matmul.

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "extract-exp-load-pattern";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

struct ExpLoadPattern {
  linalg::BroadcastOp broadcastOp;
  Operation *expOp = nullptr;
  bufferization::ToTensorOp toTensorOp;
  memref::CopyOp copyOp;
  memref::AllocOp allocOp;
  arith::MulFOp mulfOp;
  int blockId = -1;
  int newBlockId = -1;
};

static bool isExpLike(Operation *op) {
  return op && isa<math::ExpOp, math::Exp2Op>(op);
}

// Match (all in one VECTOR block B):
//   broadcast <- exp/exp2 <- to_tensor <- alloc <- memref.copy
// with broadcast's sole user being a mulf that lives OUTSIDE block B and whose
// other operand is a BlockArgument.
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

  // broadcast's input is exp/exp2, in block B.
  if (broadcastOp.getDpsInputs().empty()) {
    return false;
  }
  Operation *expOp = broadcastOp.getDpsInputs()[0].getDefiningOp();
  if (!isExpLike(expOp) || bm.getBlockIdByOp(expOp) != blockId) {
    return false;
  }

  // exp's input is bufferization.to_tensor, in block B.
  auto toTensorOp =
      expOp->getOperand(0).getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensorOp || bm.getBlockIdByOp(toTensorOp) != blockId) {
    return false;
  }

  // to_tensor's buffer is a memref.alloc, in block B.
  auto allocOp = toTensorOp.getBuffer().getDefiningOp<memref::AllocOp>();
  if (!allocOp || bm.getBlockIdByOp(allocOp) != blockId) {
    return false;
  }

  // alloc is filled by a memref.copy (copy target == alloc), in block B.
  memref::CopyOp copyOp;
  for (Operation *user : allocOp->getUsers()) {
    auto c = dyn_cast<memref::CopyOp>(user);
    if (c && c.getTarget() == allocOp.getMemref()) {
      copyOp = c;
      break;
    }
  }
  if (!copyOp || bm.getBlockIdByOp(copyOp) != blockId) {
    return false;
  }

  // broadcast's sole user is a mulf that lives OUTSIDE block B.
  if (!broadcastOp->hasOneUse()) {
    return false;
  }
  auto mulfOp = dyn_cast<arith::MulFOp>(*broadcastOp->getUsers().begin());
  if (!mulfOp) {
    return false;
  }

  int newBlockId = bm.getBlockIdByOp(mulfOp);
  if (newBlockId == blockId) {
    return false; // mulf must NOT be in the block being split
  }
  if (CVPipeline::getOpCoreType(mulfOp) != CVPipeline::CoreType::VECTOR_ONLY) {
    return false;
  }

  // mulf's other operand (the one that is not broadcast) must be a
  // BlockArgument.
  Value broadcastResult = broadcastOp->getResult(0);
  Value otherOperand =
      (mulfOp.getLhs() == broadcastResult) ? mulfOp.getRhs() : mulfOp.getLhs();
  if (!isa<BlockArgument>(otherOperand)) {
    return false;
  }

  info.broadcastOp = broadcastOp;
  info.expOp = expOp;
  info.toTensorOp = toTensorOp;
  info.copyOp = copyOp;
  info.allocOp = allocOp;
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
    return "Extract load->exp->broadcast from a VECTOR block into a fresh "
           "VECTOR "
           "block when its mulf user lives elsewhere, cloning shared scalar "
           "index ops via ScalarClosure.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    CVPipeline::ComputeBlockIdManager bm(module);

    // Phase 1: collect all candidate patterns (anchor = broadcast in the
    // block).
    SmallVector<ExpLoadPattern> patterns;
    module.walk([&](linalg::BroadcastOp broadcastOp) {
      ExpLoadPattern info;
      if (matchExpLoadPattern(broadcastOp, bm, info)) {
        LOG_DEBUG("Matched exp-load pattern in block " << info.blockId);
        patterns.push_back(info);
      }
    });

    // Phase 2: extract each pattern into a fresh VECTOR block.
    for (auto &info : patterns) {
      // A prior extraction may have moved this broadcast; skip if so.
      if (bm.getBlockIdByOp(info.broadcastOp) != info.blockId) {
        LOG_DEBUG("Stale pattern (block id changed), skip");
        continue;
      }

      // Core ops that move into the new block.  mulf is deliberately excluded:
      // it already lives in a different block.
      SmallVector<Operation *> coreOps = {info.broadcastOp, info.expOp,
                                          info.toTensorOp, info.copyOp,
                                          info.allocOp};

      // Identify scalar / memref-view dependencies of the load chain that live
      // in the same block via ScalarClosure (same utility used by FixpipeOpt).
      CVPipeline::SplitIf::ScalarClosure closure{
          info.broadcastOp->getBlock(), coreOps, /*includeParent=*/false};
      closure.collect();

      llvm::SetVector<Operation *> extracted;
      for (Operation *op : coreOps) {
        extracted.insert(op);
      }
      for (Operation *op : closure.scalarOps) {
        extracted.insert(op);
      }

      // Clone scalar ops whose users would be stranded in the original block:
      // the clone keeps the original block_id, external uses are redirected to
      // it, and the original op follows the extracted set into newBlockId.
      CVPipeline::cloneScalarOpsForCrossBlockUses(bm, extracted,
                                                  info.newBlockId);

      for (Operation *op : extracted) {
        bm.updateBlockIdWithInner(op, info.newBlockId);
      }

      LOG_DEBUG("Extracted exp-load pattern: block " << info.blockId << " -> "
                                                     << info.newBlockId);
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
