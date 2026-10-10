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

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ascend/include/DynamicCVPipeline/SplitDataflow/PreserveControlAttrsCanonicalize.h"

#include "DynamicCVPipeline/Common/Utils.h"

using namespace mlir;

#define DEBUG_TYPE "preserve-control-attrs-canonicalize"
#define LOG_DEBUG(msg)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg)

namespace {

static void debugDumpIr(StringRef stage, Operation *op) {
  LOG_DEBUG(stage << "\n"; op->print(llvm::dbgs()); llvm::dbgs() << "\n");
}

static bool isTrackedControlFlowOp(Operation *op) {
  return isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ParallelOp>(op);
}

static bool canTransferAttrs(Operation *from, Operation *to) {
  return from && to && from != to && isTrackedControlFlowOp(from) &&
         isTrackedControlFlowOp(to) && from->getName() == to->getName();
}

static bool haveCompatibleBlockIds(Operation *lhs, Operation *rhs) {
  return lhs->getAttr(CVPipeline::kBlockId) ==
         rhs->getAttr(CVPipeline::kBlockId);
}

/// Add a block_id check before SCF's generic if merge patterns run. The SCF
/// patterns are registered in an anonymous namespace, so identify the two
/// patterns by their debug names and delegate all merge logic to them.
class BlockIdAwareIfMergePattern : public OpRewritePattern<scf::IfOp> {
public:
  enum class Kind { AdjacentMerge, NestedMerge, ConditionPropagation };

  BlockIdAwareIfMergePattern(MLIRContext *ctx,
                             std::unique_ptr<RewritePattern> wrappedPattern,
                             Kind kind)
      : OpRewritePattern<scf::IfOp>(ctx, wrappedPattern->getBenefit()),
        wrappedPattern(std::move(wrappedPattern)),
        kind(kind) {
    setDebugName(this->wrappedPattern->getDebugName());
    addDebugLabels(this->wrappedPattern->getDebugLabels());
    setHasBoundedRewriteRecursion(
        this->wrappedPattern->hasBoundedRewriteRecursion());
  }

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (kind == Kind::ConditionPropagation) {
      for (OpOperand &use : ifOp.getCondition().getUses()) {
        auto nestedIf = dyn_cast<scf::IfOp>(use.getOwner());
        if (!nestedIf || nestedIf.getCondition() != use.get())
          continue;
        Region *useRegion = use.getOwner()->getParentRegion();
        if ((ifOp.getThenRegion().isAncestor(useRegion) ||
             ifOp.getElseRegion().isAncestor(useRegion)) &&
            !haveCompatibleBlockIds(ifOp, nestedIf)) {
          return rewriter.notifyMatchFailure(
              ifOp, "cannot propagate conditions across different block IDs");
        }
      }
      return wrappedPattern->matchAndRewrite(ifOp.getOperation(), rewriter);
    }

    Operation *otherIf = nullptr;
    if (kind == Kind::AdjacentMerge) {
      otherIf = ifOp->getPrevNode();
    } else {
      auto nestedOps = ifOp.thenBlock()->without_terminator();
      if (llvm::hasSingleElement(nestedOps)) {
        if (auto nestedIf = dyn_cast<scf::IfOp>(*nestedOps.begin()))
          otherIf = nestedIf;
      }
    }

    if (otherIf && isa<scf::IfOp>(otherIf) &&
        !haveCompatibleBlockIds(ifOp, otherIf)) {
      return rewriter.notifyMatchFailure(
          ifOp, "cannot merge scf.if operations with different block IDs");
    }

    return wrappedPattern->matchAndRewrite(ifOp.getOperation(), rewriter);
  }

private:
  std::unique_ptr<RewritePattern> wrappedPattern;
  Kind kind;
};

static void guardIfMergePatterns(MLIRContext *ctx,
                                 RewritePatternSet &patterns) {
  auto &nativePatterns = patterns.getNativePatterns();
  for (std::unique_ptr<RewritePattern> &pattern : nativePatterns) {
    StringRef debugName = pattern->getDebugName();
    std::optional<BlockIdAwareIfMergePattern::Kind> kind;
    if (debugName.ends_with("CombineIfs"))
      kind = BlockIdAwareIfMergePattern::Kind::AdjacentMerge;
    else if (debugName.ends_with("CombineNestedIfs"))
      kind = BlockIdAwareIfMergePattern::Kind::NestedMerge;
    else if (debugName.ends_with("ConditionPropagation"))
      kind = BlockIdAwareIfMergePattern::Kind::ConditionPropagation;
    if (!kind)
      continue;

    pattern = std::make_unique<BlockIdAwareIfMergePattern>(
        ctx, std::move(pattern), *kind);
  }
}

class PreserveControlAttrsListener : public RewriterBase::Listener {
public:
  void notifyOperationInserted(Operation *op, OpBuilder::InsertPoint) override {
    recentInserts.insert(op);
  }

  void notifyOperationErased(Operation *op) override {
    recentInserts.remove(op);
  }

  void notifyOperationReplaced(Operation *op, Operation *newOp) override {
    transferAttrs(op, newOp);
    transferBlockIdToInsertedReplacement(op, newOp);
  }

  void notifyOperationReplaced(Operation *op, ValueRange values) override {
    if (Operation *newOp = findReplacementOp(op, values)) {
      transferAttrs(op, newOp);
    }

    for (Value value : values) {
      if (!value)
        continue;
      if (Operation *defOp = value.getDefiningOp()) {
        transferBlockIdToInsertedReplacement(op, defOp);
      }
    }
  }

private:
  Operation *findReplacementOp(Operation *oldOp,
                               ValueRange replacements) const {
    if (!isTrackedControlFlowOp(oldOp))
      return nullptr;

    // Prefer direct replacement from definingOps
    for (Value value : replacements) {
      if (!value)
        continue;
      Operation *defOp = value.getDefiningOp();
      if (defOp && recentInserts.contains(defOp) &&
          canTransferAttrs(oldOp, defOp))
        return defOp;
    }

    Block *oldBlock = oldOp->getBlock();
    if (!oldBlock)
      return nullptr;

    // Fallback: search recentInserts in reverse, but only accept same-block
    // candidates.
    for (Operation *candidate : llvm::reverse(recentInserts.getArrayRef())) {
      if (!canTransferAttrs(oldOp, candidate))
        continue;

      if (candidate->getBlock() != oldBlock)
        continue;
      return candidate;
    }
    return nullptr;
  }

  static void transferAttrs(Operation *from, Operation *to) {
    if (!canTransferAttrs(from, to))
      return;

    for (NamedAttribute attr : from->getAttrs()) {
      if (to->hasAttr(attr.getName()))
        continue;
      to->setAttr(attr.getName(), attr.getValue());
    }
  }

  void transferBlockIdToInsertedReplacement(Operation *from,
                                            Operation *to) const {
    if (!from || !to || from == to || !isTrackedControlFlowOp(from)) {
      return;
    }

    Attribute blockId = from->getAttr(CVPipeline::kBlockId);
    if (!blockId || to->hasAttr(CVPipeline::kBlockId)) {
      return;
    }

    if (!recentInserts.contains(to) || from->getBlock() != to->getBlock()) {
      return;
    }

    to->setAttr(CVPipeline::kBlockId, blockId);
  }

  llvm::SetVector<Operation *> recentInserts;
};

static void populateCanonicalizationPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns) {
  for (Dialect *dialect : ctx->getLoadedDialects()) {
    dialect->getCanonicalizationPatterns(patterns);
  }

  for (RegisteredOperationName opName : ctx->getRegisteredOperations()) {
    opName.getCanonicalizationPatterns(patterns, ctx);
  }
}

} // namespace

void mlir::triton::PreserveControlAttrsCanonicalizePass::runOnOperation() {
  if (CVPipeline::hasFallbackAttr(getOperation())) {
    return;
  }

  debugDumpIr("before PreserveControlAttrsCanonicalizePass", getOperation());

  RewritePatternSet patterns(&getContext());
  populateCanonicalizationPatterns(&getContext(), patterns);
  guardIfMergePatterns(&getContext(), patterns);

  PreserveControlAttrsListener listener;
  GreedyRewriteConfig config;
  config.setListener(&listener);

  if (failed(applyPatternsGreedily(getOperation(),
                                   FrozenRewritePatternSet(std::move(patterns)),
                                   config))) {
    getOperation()->emitError("PreserveControlAttrsCanonicalizePass failed");
    CVPipeline::setFallbackAttr(getOperation(), CVPipeline::ERRCODE_FAILED);
    return;
  }

  debugDumpIr("after PreserveControlAttrsCanonicalizePass", getOperation());
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createPreserveControlAttrsCanonicalizePass() {
  return std::make_unique<PreserveControlAttrsCanonicalizePass>();
}

void registerPreserveControlAttrsCanonicalizePasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createPreserveControlAttrsCanonicalizePass();
  });
}

} // namespace triton
} // namespace mlir
