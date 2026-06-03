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
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "merge-vector-if-block";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace {

//===----------------------------------------------------------------------===//
// Cycle detection (mirrors UnifyAllocBlockPass)
//
// Merging operations into a single block_id is only valid if the resulting
// block-level dependency graph stays acyclic. The helpers below temporarily
// assign the candidate ops to the target block_id, walk the SSA + memory
// dependency edges, and report whether a cycle would be introduced.
//===----------------------------------------------------------------------===//

struct CycleDfs {
  llvm::DenseSet<mlir::Operation *> &okSet;
  llvm::DenseSet<mlir::Operation *> visited;
  const CVPipeline::MemoryDependenceGraph &memGraph;
  CVPipeline::ComputeBlockIdManager &bm;
  Block *block;
  void clear() { visited.clear(); }
  bool operator()(Operation *cur);
  bool dfs(Operation *cur) { return (*this)(cur); };
  CycleDfs(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
           llvm::DenseSet<mlir::Operation *> &okSet, CVPipeline::ComputeBlockIdManager &bm)
      : okSet(okSet), memGraph(memGraph), bm(bm), block(block) {}
};

bool CycleDfs::operator()(Operation *cur) {
  if (okSet.contains(cur)) {
    return true;
  }
  if (!visited.insert(cur).second) {
    return false;
  }

  SmallVector<Operation *> allusers;
  allusers.append(cur->getUsers().begin(), cur->getUsers().end());
  for (auto *memUser : memGraph.getExecAfter(cur)) {
    allusers.push_back(memUser);
  }
  for (auto *user : allusers) {
    auto *userInBlock = CVPipeline::getAncestorInBlock(user, block);
    if (!userInBlock) continue;
    if (okSet.contains(userInBlock)) {
      LOG_DEBUG("[CycleDfs] Cycle found, userInBlock in okSet: " << *userInBlock);
      return true;
    }
    int userBlockId = bm.getBlockIdByOp(userInBlock);
    if (userBlockId == -1) {
      if (dfs(userInBlock)) {
        return true;
      }
    } else {
      for (auto *nx : bm.getOpsByBlockId(userBlockId)) {
        if (dfs(nx)) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * @brief Detect if unifying a list of operations to target block_id would create a cycle
 *
 * @param opsToUnify Block-level operations to add to the safe set (okSet)
 * @param memGraph Memory dependence graph for RAW/WAW/WAR dependency analysis
 * @param targetBlockId Target block_id after unification
 * @return bool Returns true if unification would create a cycle, false otherwise
 */
static bool willCreateCycle(ArrayRef<Operation *> opsToUnify,
                            const CVPipeline::MemoryDependenceGraph &memGraph,
                            int targetBlockId, CVPipeline::ComputeBlockIdManager &bm) {
  if (opsToUnify.empty()) {
    return false;
  }

  auto *block = opsToUnify.front()->getBlock();

  llvm::DenseSet<Operation *> okSet;
  for (auto *op : bm.getOpsByBlockId(targetBlockId)) {
    okSet.insert(op);
  }
  for (auto *op : opsToUnify) {
    okSet.insert(op);
  }

  DenseMap<Operation *, int> origBlockIdMap;
  for (auto *op : opsToUnify) {
    auto optBlockId = CVPipeline::getOpBlockId(op);
    origBlockIdMap[op] = optBlockId ? static_cast<int>(*optBlockId) : -1;
    bm.updateBlockId(op, targetBlockId);
  }

  CycleDfs dfs(block, memGraph, okSet, bm);
  bool hasCycle = false;

  for (mlir::Operation *okOp : okSet) {
    SmallVector<Operation *> allusers;
    allusers.append(okOp->getUsers().begin(), okOp->getUsers().end());
    for (auto *memUser : memGraph.getExecAfter(okOp)) {
      allusers.push_back(memUser);
    }
    for (auto *user : allusers) {
      auto *userInBlock = CVPipeline::getAncestorInBlock(user, block);
      if (!userInBlock) continue;
      if (okSet.contains(userInBlock)) {
        continue;
      }
      int userBlockId = bm.getBlockIdByOp(userInBlock);
      if (userBlockId == -1) {
        dfs.clear();
        if (dfs(userInBlock)) {
          hasCycle = true;
          break;
        }
      } else {
        for (auto *userOp : bm.getOpsByBlockId(userBlockId)) {
          dfs.clear();
          if (dfs(userOp)) {
            hasCycle = true;
            break;
          }
        }
      }
    }
    if (hasCycle) {
      break;
    }
  }

  for (auto &[op, origBlockId] : origBlockIdMap) {
    if (origBlockId == -1) {
      op->removeAttr(CVPipeline::kBlockId);
    } else {
      bm.updateBlockId(op, origBlockId);
    }
  }

  return hasCycle;
}

//===----------------------------------------------------------------------===//
// Pure-vector scf.if recognition
//===----------------------------------------------------------------------===//

/**
 * @brief Check whether an scf.if only contains Vector-tagged operations
 *
 * A pure-vector if has every (non-terminator) operation inside its regions
 * tagged as a VECTOR core op. Any operation that holds nested regions
 * (i.e. nested control flow such as scf.if / scf.for) is rejected because
 * such control ops carry no block tag and therefore cannot be merged.
 */
static bool isPureVectorIf(scf::IfOp ifOp) {
  for (Region &region : ifOp->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (op.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }
        // Nested control flow has no Vector tag -> illegal to merge.
        if (op.getNumRegions() > 0) {
          return false;
        }
        if (CVPipeline::getOpCoreType(&op) != CVPipeline::CoreType::VECTOR_ONLY) {
          return false;
        }
      }
    }
  }
  return true;
}

/**
 * @brief Determine the upstream block_id to merge the scf.if into
 *
 * The upstream is the data source of the if: its condition plus every value
 * consumed inside the regions that is produced outside the if. Constant-like
 * producers and values produced inside the if are ignored. When all remaining
 * data sources share a single block_id, that id is the merge target.
 */
static LogicalResult getUpstreamBlockId(scf::IfOp ifOp,
                                        CVPipeline::ComputeBlockIdManager &bm,
                                        int &target) {
  Block *parent = ifOp->getBlock();
  llvm::SmallDenseSet<int, 4> ids;

  auto addSource = [&](Value v) {
    Operation *def = v.getDefiningOp();
    if (!def) {
      return; // block argument: not a tagged data source
    }
    if (def->hasTrait<OpTrait::ConstantLike>()) {
      return; // constants are loop-invariant, ignore them
    }
    if (ifOp->isAncestor(def)) {
      return; // produced inside the if itself
    }
    Operation *anc = CVPipeline::getAncestorInBlock(def, parent);
    if (!anc) {
      return; // produced in an outer scope, not a sibling block op
    }
    int bid = bm.getBlockIdByOp(anc);
    if (bid != -1) {
      ids.insert(bid);
    }
  };

  addSource(ifOp.getCondition());
  ifOp->walk([&](Operation *op) {
    if (op == ifOp.getOperation()) {
      return;
    }
    for (Value v : op->getOperands()) {
      addSource(v);
    }
  });

  if (ids.size() != 1) {
    LOG_DEBUG("[getUpstreamBlockId] data sources are not consistent, count=" << ids.size());
    return failure();
  }
  target = *ids.begin();
  return success();
}

//===----------------------------------------------------------------------===//
// Merge application
//===----------------------------------------------------------------------===//

static void applyMerge(scf::IfOp ifOp, ArrayRef<Operation *> downstreamOps,
                       int target, CVPipeline::ComputeBlockIdManager &bm) {
  bm.updateBlockId(ifOp.getOperation(), target);
  // Rewrite the inner ops so the whole if body shares the upstream block_id.
  for (Region &region : ifOp->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (op.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }
        if (CVPipeline::getOpBlockId(&op)) {
          bm.updateBlockId(&op, target);
        }
      }
    }
  }
  for (Operation *op : downstreamOps) {
    bm.updateBlockId(op, target);
  }
}

/**
 * @brief Collect the downstream block_ids that consume the if results,
 *        ordered by their first appearance in the parent block.
 */
static SmallVector<int> collectDownstreamBlockIds(scf::IfOp ifOp, int target,
                                                  CVPipeline::ComputeBlockIdManager &bm) {
  Block *parent = ifOp->getBlock();
  llvm::DenseSet<Operation *> userAnchors;
  for (Value res : ifOp->getResults()) {
    for (Operation *user : res.getUsers()) {
      if (auto *anc = CVPipeline::getAncestorInBlock(user, parent)) {
        userAnchors.insert(anc);
      }
    }
  }

  SmallVector<int> ordered;
  llvm::SmallDenseSet<int, 4> seen;
  for (Operation &op : *parent) {
    if (!userAnchors.contains(&op)) {
      continue;
    }
    int bid = bm.getBlockIdByOp(&op);
    if (bid == -1 || bid == target) {
      continue;
    }
    if (seen.insert(bid).second) {
      ordered.push_back(bid);
    }
  }
  return ordered;
}

/**
 * @brief Try to merge one pure-vector scf.if with its upstream and a downstream block
 */
static void tryMergeIf(scf::IfOp ifOp,
                       const CVPipeline::MemoryDependenceGraph &memGraph,
                       CVPipeline::ComputeBlockIdManager &bm) {
  if (!isPureVectorIf(ifOp)) {
    return;
  }

  int target;
  if (failed(getUpstreamBlockId(ifOp, bm, target))) {
    return;
  }
  LOG_DEBUG("[tryMergeIf] upstream target block_id = " << target << " for " << *ifOp);

  SmallVector<int> downstream = collectDownstreamBlockIds(ifOp, target, bm);

  // Prefer merging the if together with the first downstream block that keeps
  // the dependency graph acyclic, forming one large block.
  for (int bid : downstream) {
    SmallVector<Operation *> downstreamOps = bm.getOpsByBlockId(bid);
    SmallVector<Operation *> opsToUnify;
    opsToUnify.push_back(ifOp.getOperation());
    opsToUnify.append(downstreamOps.begin(), downstreamOps.end());
    if (!willCreateCycle(opsToUnify, memGraph, target, bm)) {
      LOG_DEBUG("[tryMergeIf] merging if with downstream block_id " << bid << " into " << target);
      applyMerge(ifOp, downstreamOps, target, bm);
      return;
    }
    LOG_DEBUG("[tryMergeIf] downstream block_id " << bid << " would create a cycle, skip");
  }

  // Fallback: at least fold the if into the upstream block if that is safe.
  SmallVector<Operation *> ifOnly = {ifOp.getOperation()};
  if (!willCreateCycle(ifOnly, memGraph, target, bm)) {
    LOG_DEBUG("[tryMergeIf] merging if into upstream " << target << " without downstream");
    applyMerge(ifOp, {}, target, bm);
  }
}

} // anonymous namespace

class MergeVectorIfBlockPass
    : public PassWrapper<MergeVectorIfBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeVectorIfBlockPass)

  MergeVectorIfBlockPass() = default;

  StringRef getArgument() const override { return "merge-vector-if-block"; }

  StringRef getDescription() const override {
    return "Merge pure-Vector scf.if blocks with their upstream data-source "
           "block and a downstream consumer block";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LOG_DEBUG("Before: " << *module);
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    auto bm = CVPipeline::ComputeBlockIdManager(module);

    llvm::SmallVector<scf::IfOp> ifOps;
    module.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

    for (scf::IfOp ifOp : ifOps) {
      tryMergeIf(ifOp, memGraph, bm);
    }

    LOG_DEBUG("After: " << *module);
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeVectorIfBlockPass() {
  return std::make_unique<MergeVectorIfBlockPass>();
}

void registerMergeVectorIfBlockPass() {
  PassRegistration<MergeVectorIfBlockPass> reg;
}

} // namespace triton
} // namespace mlir
