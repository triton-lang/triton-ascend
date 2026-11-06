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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <optional>

static constexpr const char *DEBUG_TYPE = "pos-mask-pattern-opt";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;

namespace mlir {
namespace triton {

class PosMaskPatternPass
    : public PassWrapper<PosMaskPatternPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PosMaskPatternPass)

  PosMaskPatternPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "pos-mask-pattern-opt"; }
  llvm::StringRef getDescription() const final {
    return "Sink position patterns and attention mask computations into "
           "their user blocks when safe";
  }
};

} // namespace triton
} // namespace mlir

namespace {

struct PosPattern {
  linalg::BroadcastOp broadcastOp;
  arith::CmpIOp eqcmpOp;
  arith::CmpIOp slecmpOp;
  arith::ExtUIOp eqextOp;
  arith::ExtUIOp sleextOp;

  llvm::SmallVector<Operation *> getAllOps() const {
    return {broadcastOp, eqcmpOp, slecmpOp, eqextOp, sleextOp};
  }
};

static bool isPosBroadcast(linalg::BroadcastOp broadcastOp) {
  if (CVPipeline::getOpCoreType(broadcastOp.getOperation()) !=
      CVPipeline::CoreType::VECTOR_ONLY) {
    return false;
  }
  return true;
}

static std::optional<PosPattern>
collectPosPattern(linalg::BroadcastOp broadcastOp,
                  CVPipeline::ComputeBlockIdManager &bm) {
  PosPattern pattern;
  if (llvm::range_size(broadcastOp.getResult().getUsers()) != 2) {
    return std::nullopt;
  }
  if (any_of(broadcastOp->getUsers(),
             [&](Operation *user) { return !isa<arith::CmpIOp>(user); })) {
    return std::nullopt;
  }

  arith::CmpIOp eqcmpOp = nullptr;
  arith::CmpIOp slecmpOp = nullptr;

  for (Operation *user : broadcastOp->getUsers()) {
    auto cmpOp = dyn_cast<arith::CmpIOp>(user);
    auto predicate = cmpOp.getPredicate();
    if (predicate == arith::CmpIPredicate::eq) {
      if (eqcmpOp) {
        return std::nullopt;
      }
      eqcmpOp = cmpOp;
    } else if (predicate == arith::CmpIPredicate::sle) {
      if (slecmpOp) {
        return std::nullopt;
      }
      slecmpOp = cmpOp;
    } else {
      return std::nullopt;
    }
  }

  if (!eqcmpOp || !slecmpOp) {
    return std::nullopt;
  }
  if (!eqcmpOp.getResult().hasOneUse() || !slecmpOp.getResult().hasOneUse()) {
    return std::nullopt;
  }

  arith::ExtUIOp eqextOp =
      dyn_cast<arith::ExtUIOp>(*eqcmpOp->getUsers().begin());
  arith::ExtUIOp sleextOp =
      dyn_cast<arith::ExtUIOp>(*slecmpOp->getUsers().begin());
  if (!eqextOp || !sleextOp) {
    return std::nullopt;
  }

  pattern.eqextOp = eqextOp;
  pattern.sleextOp = sleextOp;
  pattern.eqcmpOp = eqcmpOp;
  pattern.slecmpOp = slecmpOp;
  pattern.broadcastOp = broadcastOp;

  auto broadcastBlockId = bm.getBlockIdByOp(broadcastOp);
  auto broadcastBlock = broadcastOp->getBlock();
  if (!all_of(pattern.getAllOps(), [&](Operation *op) {
        return bm.getBlockIdByOp(op) == broadcastBlockId;
      })) {
    return std::nullopt;
  }
  if (!all_of(pattern.getAllOps(), [&](Operation *op) {
        return op->getBlock() == broadcastBlock;
      })) {
    return std::nullopt;
  }

  return pattern;
}

static std::optional<int>
findTargetBlock(const PosPattern &pattern,
                CVPipeline::ComputeBlockIdManager &bm) {
  llvm::SmallVector<Operation *> users;

  for (Operation *user : pattern.eqextOp->getUsers()) {
    if (user->getBlock() != pattern.broadcastOp->getBlock()) {
      continue;
    }
    users.push_back(user);
  }

  for (Operation *user : pattern.sleextOp->getUsers()) {
    if (user->getBlock() != pattern.broadcastOp->getBlock()) {
      continue;
    }
    users.push_back(user);
  }
  if (users.size() == 0) {
    return std::nullopt;
  }

  int targetBlockId = bm.getBlockIdByOp(users[0]);
  if (targetBlockId == -1) {
    return std::nullopt;
  }
  for (Operation *user : users) {
    if (bm.getBlockIdByOp(user) != targetBlockId) {
      return std::nullopt;
    }
  }

  return targetBlockId;
}

// Match the consumer, rather than treating every comparison as an attention
// mask. Constants may have been folded from linalg.fill to dense splats.
static std::optional<APFloat> getFloatSplat(Value value) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    return getFloatSplat(fill.getInputs()[0]);
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  if (auto scalar = dyn_cast<FloatAttr>(attr))
    return scalar.getValue();
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr);
      dense && dense.isSplat())
    return dense.getSplatValue<APFloat>();
  return std::nullopt;
}

static bool isMaskBias(arith::SelectOp select) {
  auto a = getFloatSplat(select.getTrueValue());
  auto b = getFloatSplat(select.getFalseValue());
  if (!a || !b)
    return false;
  auto isNegative = [](const APFloat &v) {
    return v.isNegative() && !v.isZero() && !v.isNaN();
  };
  return (a->isZero() && isNegative(*b)) || (b->isZero() && isNegative(*a));
}

static bool isScaledMatmul(Value value) {
  auto mul = value.getDefiningOp<arith::MulFOp>();
  if (!mul)
    return false;
  return mul.getLhs().getDefiningOp<linalg::MatmulOp>() ||
         mul.getRhs().getDefiningOp<linalg::MatmulOp>();
}

static bool isOneSplat(Value value) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    return isOneSplat(fill.getInputs()[0]);
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return false;
  if (auto scalar = dyn_cast<IntegerAttr>(attr))
    return scalar.getValue().isOne();
  auto dense = dyn_cast<DenseIntElementsAttr>(attr);
  return dense && dense.isSplat() && dense.getSplatValue<APInt>().isOne();
}

// Loads, to_tensor, control flow, and arbitrary regions remain inputs of
// the moved, side-effect-free tensor computation.
// Loads, to_tensor, control flow, and arbitrary regions remain inputs of
// the moved, side-effect-free tensor computation.
static bool isMaskComputation(Operation *op) {
  return isa<arith::CmpIOp, arith::ExtUIOp, arith::TruncIOp, arith::AndIOp,
             arith::OrIOp, arith::ShRUIOp, linalg::BroadcastOp,
             tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::ReshapeOp,
             tensor::ExtractSliceOp, tensor::InsertSliceOp>(op);
}

static void sinkAttentionMasks(ModuleOp module,
                               CVPipeline::MemoryDependenceGraph &memGraph,
                               CVPipeline::ComputeBlockIdManager &bm) {
  module.walk([&](arith::AddFOp add) {
    if (CVPipeline::getOpCoreType(add) != CVPipeline::CoreType::VECTOR_ONLY)
      return;
    arith::SelectOp bias;
    if (isScaledMatmul(add.getLhs()))
      bias = add.getRhs().getDefiningOp<arith::SelectOp>();
    else if (isScaledMatmul(add.getRhs()))
      bias = add.getLhs().getDefiningOp<arith::SelectOp>();
    if (!bias || !isMaskBias(bias) || bias->getBlock() != add->getBlock())
      return;
    int source = bm.getBlockIdByOp(bias);
    int target = bm.getBlockIdByOp(add);
    if (source < 0 || target < 0 || source == target)
      return;

    llvm::SetVector<Operation *> candidates;
    SmallVector<Operation *> pending{bias.getOperation()};
    while (!pending.empty()) {
      Operation *op = pending.pop_back_val();
      if (candidates.contains(op) || op->getBlock() != add->getBlock() ||
          bm.getBlockIdByOp(op) != source ||
          CVPipeline::getOpCoreType(op) != CVPipeline::CoreType::VECTOR_ONLY ||
          (op != bias.getOperation() && !isMaskComputation(op)))
        continue;
      candidates.insert(op);
      for (Value operand : op->getOperands())
        if (Operation *def = operand.getDefiningOp())
          pending.push_back(def);
    }

    // Shared producers stay in their original block; prune to a fixed point,
    // since retaining one producer can also require retaining its inputs.
    bool changed;
    do {
      changed = false;
      SmallVector<Operation *> retained;
      for (Operation *op : candidates)
        for (Operation *user : op->getUsers())
          if (!candidates.contains(user) &&
              (user->getBlock() != add->getBlock() ||
               bm.getBlockIdByOp(user) != target)) {
            retained.push_back(op);
            break;
          }
      for (Operation *op : retained) {
        candidates.remove(op);
        changed = true;
      }
    } while (changed);

    if (!candidates.contains(bias))
      return;
    // Do not turn this into a general-purpose select sinking rule. Require
    // either a comparison/boolean-combination chain or bit extraction.
    bool hasCompare = false, hasLogic = false, hasShift = false;
    for (Operation *op : candidates) {
      if (auto cmp = dyn_cast<arith::CmpIOp>(op))
        hasCompare |= cmp.getPredicate() != arith::CmpIPredicate::ne;
      hasLogic |= isa<arith::AndIOp, arith::OrIOp>(op);
      if (auto bitAnd = dyn_cast<arith::AndIOp>(op)) {
        auto isBitExtraction = [](Value bits, Value one) {
          return bits.getDefiningOp<arith::ShRUIOp>() && isOneSplat(one);
        };
        hasShift |= isBitExtraction(bitAnd.getLhs(), bitAnd.getRhs()) ||
                    isBitExtraction(bitAnd.getRhs(), bitAnd.getLhs());
      }
    }
    if (!hasLogic || (!hasCompare && !hasShift))
      return;
    SmallVector<Operation *> ops(candidates.begin(), candidates.end());
    if (CVPipeline::willCreateCycle(ops, memGraph, target, bm))
      return;
    LOG_DEBUG("Moving attention mask computation to block " << target << "\n");
    for (Operation *op : ops)
      bm.updateBlockId(op, target);
  });
}

} // namespace

namespace mlir {
namespace triton {

void PosMaskPatternPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  LOG_DEBUG("Input mlir:" << moduleOp);
  auto &aa = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph(moduleOp, aa);
  CVPipeline::ComputeBlockIdManager bm(moduleOp);

  llvm::SmallVector<linalg::BroadcastOp> broadcasts;
  moduleOp.walk([&](linalg::BroadcastOp op) {
    if (isPosBroadcast(op)) {
      broadcasts.push_back(op);
    }
  });

  for (linalg::BroadcastOp broadcastOp : broadcasts) {
    auto patternOpt = collectPosPattern(broadcastOp, bm);
    if (!patternOpt.has_value()) {
      continue;
    }
    auto pattern = patternOpt.value();
    std::optional<int> targetBlockId = findTargetBlock(pattern, bm);
    if (!targetBlockId) {
      LOG_DEBUG("No common target block_id, skip\n");
      continue;
    }

    int broadcastBlockId = bm.getBlockIdByOp(pattern.broadcastOp);
    if (broadcastBlockId == targetBlockId) {
      LOG_DEBUG("Already in same block, skip\n");
      continue;
    }

    llvm::SmallVector<Operation *> allOps = pattern.getAllOps();
    if (CVPipeline::willCreateCycle(allOps, memGraph, *targetBlockId, bm)) {
      LOG_DEBUG("Would create cycle, skip\n");
      continue;
    }

    LOG_DEBUG("Moving pos pattern ops to block " << *targetBlockId << "\n");
    for (Operation *op : allOps) {
      bm.updateBlockId(op, *targetBlockId);
    }
  }
  sinkAttentionMasks(moduleOp, memGraph, bm);
}

std::unique_ptr<OperationPass<ModuleOp>> createPosMaskPatternPass() {
  return std::make_unique<PosMaskPatternPass>();
}

} // namespace triton
} // namespace mlir
