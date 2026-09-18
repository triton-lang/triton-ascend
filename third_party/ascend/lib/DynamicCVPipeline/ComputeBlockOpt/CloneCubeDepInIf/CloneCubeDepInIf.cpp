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
#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"

static constexpr const char *DEBUG_TYPE = "clone-cube-dep-in-if";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;
using namespace SplitIf;

namespace {

/// A maximal run of consecutive ops in MLIR order inside a single if-region
/// that share the same block_id AND whose core_type is CUBE.
struct CubeBlock {
  int blockId;
  SmallVector<Operation *> ops;
};

/// One candidate if that SplitIfByBlockId would later split.
struct CandidateIf {
  scf::IfOp ifOp;
  SmallVector<CubeBlock> thenCubes;
  SmallVector<CubeBlock> elseCubes;
  bool thenNeedsSplit = false;
  bool elseNeedsSplit = false;
};

/// Pretty-print a CubeBlock for debug.
static std::string formatCubeBlock(const CubeBlock &c) {
  std::string s;
  s += "{bid=" + std::to_string(c.blockId) + " ops=[";
  bool first = true;
  for (Operation *op : c.ops) {
    if (!first) {
      s += ", ";
    }
    first = false;
    if (op->getName().getStringRef().size() > 32) {
      s += op->getName().getStringRef().substr(0, 32).str();
    } else {
      s += op->getName().getStringRef().str();
    }
  }
  s += "]}";
  return s;
}

} // namespace

/// Scan a region (then-block or else-block) and produce maximal CUBE runs.
static SmallVector<CubeBlock> collectCubeBlocksInRegion(Block &block) {
  SmallVector<CubeBlock> cubes;
  auto *parentOp = block.getParentOp();
  if (!parentOp) {
    return cubes;
  }

  for (auto &op : block) {
    if (isa<scf::YieldOp>(op)) {
      continue;
    }

    auto bid = CVPipeline::getOpBlockId(&op);
    if (!bid.has_value() || *bid == -1) {
      continue;
    }

    CoreType ct = CVPipeline::getOpCoreType(&op);
    if (ct != CoreType::CUBE_ONLY) {
      continue;
    }

    if (!cubes.empty() && cubes.back().blockId == *bid &&
        !cubes.back().ops.empty()) {
      cubes.back().ops.push_back(&op);
    } else {
      cubes.push_back({*bid, {&op}});
    }
  }

  return cubes;
}

/// Build the candidate if we plan to process.
static CandidateIf getCandidate(scf::IfOp ifOp) {
  CandidateIf cand;
  cand.ifOp = ifOp;

  cand.thenCubes = collectCubeBlocksInRegion(*ifOp.thenBlock());
  Block *elseBlk = ifOp.elseBlock();
  if (elseBlk) {
    cand.elseCubes = collectCubeBlocksInRegion(*elseBlk);
  }

  auto distinctCount = [](const SmallVector<CubeBlock> &cubes) {
    llvm::SmallDenseSet<int> ids;
    for (auto &c : cubes) {
      ids.insert(c.blockId);
    }
    return static_cast<unsigned>(ids.size());
  };
  cand.thenNeedsSplit = distinctCount(cand.thenCubes) >= 2;
  cand.elseNeedsSplit = distinctCount(cand.elseCubes) >= 2;

  return cand;
}

/// Check whether any op in `laterOps` references (via SSA use-def) any op in
/// `earlierOps` (transitively through nested regions).
static bool
laterCubeDependsOnEarlier(const llvm::SmallDenseSet<Operation *> &earlierOps,
                          ArrayRef<Operation *> laterOps) {
  if (earlierOps.empty()) {
    return false;
  }

  auto operandRefsEarlier = [&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      Operation *defOp = operand.get().getDefiningOp();
      if (defOp && earlierOps.contains(defOp)) {
        return true;
      }
    }
    for (auto &region : op->getRegions()) {
      bool found = false;
      region.walk([&](Operation *nestedOp) {
        if (found) {
          return;
        }
        for (auto &operand : nestedOp->getOpOperands()) {
          Operation *defOp = operand.get().getDefiningOp();
          if (defOp && earlierOps.contains(defOp)) {
            found = true;
            return;
          }
        }
      });
      if (found) {
        return true;
      }
    }
    return false;
  };

  for (auto *op : laterOps) {
    if (operandRefsEarlier(op)) {
      return true;
    }
  }
  return false;
}

/// Clone a single op using an IRMapping seeded with all previously cloned
/// values. Mirrors CloneOps::cloneOpWithMapping.
static Operation *
cloneOpWithMapping(Operation *op, OpBuilder &builder,
                   llvm::DenseMap<Value, Value> &valueMap) {
  IRMapping mapper;
  for (const auto &entry : valueMap) {
    mapper.map(entry.first, entry.second);
  }
  Operation *cloned = builder.clone(*op, mapper);
  for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
    valueMap[std::get<0>(it)] = std::get<1>(it);
  }
  return cloned;
}

/// Update op operands through valueMap, recursing through nested regions.
/// Mirrors CloneOps::updateCloneMapping.
static LogicalResult
updateCloneMapping(Operation *op, llvm::DenseMap<Value, Value> &valueMap,
                   const llvm::DenseSet<Value> &yieldValues) {
  if (!op) {
    return failure();
  }
  for (OpOperand &operand : op->getOpOperands()) {
    Value v = operand.get();
    if (yieldValues.contains(v)) {
      continue;
    }
    auto it = valueMap.find(v);
    if (it != valueMap.end()) {
      if (it->second.getType() != v.getType()) {
        LDBG("[Error]: type mismatch in value mapping: " << v.getType() << " vs "
                                                        << it->second.getType());
        return failure();
      }
      operand.set(it->second);
    }
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(updateCloneMapping(&nestedOp, valueMap, yieldValues))) {
          return failure();
        }
      }
    }
  }
  return success();
}

/// Like CloneOps::cloneOpsForBlock: clone every op from `earlierCubes` (in
/// MLIR order) into the position right before `insertBefore` and rewrite
/// `laterOps`'s operands to point at the cloned values. Inserted cloned ops
/// are tagged with kBlockId = `laterBlockId` and kClone = original block_id.
static LogicalResult
cloneEarlierCubesInto(ArrayRef<CubeBlock> earlierCubes, int laterBlockId,
                      ArrayRef<Operation *> laterOps, Operation *insertBefore,
                      const llvm::DenseSet<Value> &yieldValues,
                      SmallVectorImpl<Operation *> &newlyInserted) {
  if (earlierCubes.empty() || laterOps.empty() || insertBefore == nullptr) {
    return success();
  }

  // Collect all earlier ops in MLIR order across cubes.
  SmallVector<Operation *> toClone;
  for (auto &c : earlierCubes) {
    for (auto *op : c.ops) {
      toClone.push_back(op);
    }
  }
  if (toClone.empty()) {
    return success();
  }
  llvm::sort(toClone, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });

  LDBG("cloneEarlierCubesInto: laterBlockId=" << laterBlockId
                                              << " earlierCubes="
                                              << earlierCubes.size()
                                              << " toClone=" << toClone.size());

  OpBuilder builder(insertBefore);
  builder.setInsertionPoint(insertBefore);

  llvm::DenseMap<Value, Value> valueMap;
  for (Operation *op : toClone) {
    Operation *cloned = cloneOpWithMapping(op, builder, valueMap);
    cloned->setAttr(CVPipeline::kBlockId,
                    builder.getI32IntegerAttr(laterBlockId));
    if (auto origBlockIdOpt = CVPipeline::getOpBlockId(op)) {
      cloned->setAttr(
          CVPipeline::kClone,
          builder.getI32IntegerAttr(static_cast<int32_t>(*origBlockIdOpt)));
    }
    LDBG("  cloned " << op->getName() << " (orig block_id="
                     << CVPipeline::getOpBlockId(op).value_or(-1)
                     << ") -> new block_id=" << laterBlockId);
    newlyInserted.push_back(cloned);
  }

  // Rewrite laterOps' operands to point at cloned values.
  for (Operation *op : laterOps) {
    if (failed(updateCloneMapping(op, valueMap, yieldValues))) {
      return failure();
    }
  }
  LDBG("cloneEarlierCubesInto: done, valueMap size=" << valueMap.size());

  return success();
}

/// Process one side (then or else) of a candidate if. Mirrors
/// CloneOps::cloneOpsInMainLoop's reverse-order strategy: walk CUBE blocks
/// from last to first; for each CUBE block whose ops depend on any earlier
/// CUBE block, clone every op from earlier CUBE blocks in front of this
/// block and rewire uses.
static LogicalResult processSide(Block *block,
                                 MutableArrayRef<CubeBlock> cubes) {
  if (cubes.size() < 2 || block == nullptr) {
    return success();
  }

  llvm::DenseSet<Value> yieldValues;
  if (auto yieldOp = dyn_cast<scf::YieldOp>(block->getTerminator())) {
    for (Value operand : yieldOp.getOperands()) {
      yieldValues.insert(operand);
    }
  }

  LDBG("processSide: " << cubes.size() << " cubes");
  for (size_t k = 0; k < cubes.size(); ++k) {
    LDBG("  cube[" << k << "] " << formatCubeBlock(cubes[k]));
  }

  // Process CUBE blocks in reverse order. For each later cube, if it
  // depends on any earlier cube, clone every op in earlier cubes into
  // its front.
  for (size_t i = cubes.size(); i-- > 0;) {
    if (i == 0) {
      continue;
    }
    ArrayRef<CubeBlock> earlierCubes = ArrayRef<CubeBlock>(cubes).take_front(i);

    llvm::SmallDenseSet<Operation *> earlierOps;
    for (auto &c : earlierCubes) {
      for (auto *op : c.ops) {
        earlierOps.insert(op);
      }
    }

    bool depends = laterCubeDependsOnEarlier(earlierOps, cubes[i].ops);
    LDBG("processSide: i=" << i << " laterBid=" << cubes[i].blockId
                           << " depends=" << depends);
    if (!depends) {
      continue;
    }

    Operation *insertBefore = cubes[i].ops.front();
    SmallVector<Operation *> newlyInserted;
    if (failed(cloneEarlierCubesInto(earlierCubes, cubes[i].blockId,
                                     cubes[i].ops, insertBefore, yieldValues,
                                     newlyInserted))) {
      return failure();
    }
  }

  return success();
}

/// Cleanup: drop cloned ops whose entire SSA chain (results and memory
/// effects) is dead w.r.t. the rest of the program. Mirrors
/// CloneOps::shouldEraseOpForCube's Rule 2/3:
///
/// * Rule 2 — op has SSA results: erase only when no same-block_id user
///   of any result remains live. (User is "live" iff it is non-cloned
///   OR it has not been picked for erasure by the cascading cleanup.)
///
/// * Rule 3 — op has no results: erase only when no same-block_id
///   exec-after consumer remains live, where exec-after is determined
///   by MemoryDependenceGraph (real memory dependencies, not just MLIR
///   order).
///
/// `erasedOps` shields already-erased consumers so the cascade can
/// converge bottom-up.
static void cleanupSide(Block *block,
                       const MemoryDependenceGraph &memGraph) {
  if (block == nullptr) {
    return;
  }

  // Walk the block from back to front; identify each contiguous cloned
  // suffix and try to erase it. Mirrors CloneOps::cleanupClonedOps'
  // "find last cloned index, then contiguous cloned suffix above it"
  // strategy.
  SmallVector<Operation *> opsInBlock;
  opsInBlock.reserve(block->getOperations().size());
  for (Operation &op : *block) {
    opsInBlock.push_back(&op);
  }

  llvm::DenseSet<Operation *> erasedOps;

  for (int idx = static_cast<int>(opsInBlock.size()) - 1; idx >= 0; --idx) {
    Operation *op = opsInBlock[idx];
    if (isa<scf::YieldOp>(*op)) {
      continue;
    }
    if (!op->hasAttr(CVPipeline::kClone)) {
      continue;
    }
    if (erasedOps.contains(op) || !op->getBlock()) {
      continue;
    }

    // Decide whether to erase this op.
    auto hasLiveSameBlockIdUser = [&](Operation *op) {
      auto opBlockId = CVPipeline::getOpBlockId(op);
      if (!opBlockId) {
        return true;
      }
      for (auto result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (erasedOps.contains(user)) {
            continue;
          }
          auto userBlockId = CVPipeline::getOpBlockId(user);
          if (userBlockId && *userBlockId == *opBlockId) {
            return true;
          }
        }
      }
      return false;
    };

    auto hasLiveSameBlockIdExecAfter = [&](Operation *op) {
      auto opBlockId = CVPipeline::getOpBlockId(op);
      if (!opBlockId) {
        return true;
      }
      for (Operation *execOp : memGraph.getExecAfter(op)) {
        if (erasedOps.contains(execOp)) {
          continue;
        }
        auto execBlockId = CVPipeline::getOpBlockId(execOp);
        if (execBlockId && *execBlockId == *opBlockId) {
          return true;
        }
      }
      return false;
    };

    bool shouldErase = false;
    if (op->getNumResults() > 0) {
      // Rule 2
      shouldErase = !hasLiveSameBlockIdUser(op);
    } else {
      // Rule 3
      shouldErase = !hasLiveSameBlockIdExecAfter(op);
    }

    if (shouldErase) {
      LDBG("cleanupSide: erasing " << op->getName());
      op->erase();
      erasedOps.insert(op);
    } else {
      LDBG("cleanupSide: KEEP " << op->getName()
                                << " (live "
                                << (op->getNumResults() > 0 ? "user" : "exec-after")
                                << " in same block_id)");
    }
  }
}

static LogicalResult processCandidate(CandidateIf &cand,
                                      AliasAnalysis &aa) {
  if (cand.thenNeedsSplit) {
    if (failed(processSide(cand.ifOp.thenBlock(), cand.thenCubes))) {
      return failure();
    }
  }
  if (cand.elseNeedsSplit) {
    if (failed(processSide(cand.ifOp.elseBlock(), cand.elseCubes))) {
      return failure();
    }
  }

  // Rebuild the mem graph AFTER cloning so the cloned ops participate
  // in the exec-after edges. CloneOps takes the same approach (rebuild
  // the graph on demand for the loop op after each cloning round).
  MemoryDependenceGraph memGraph(cand.ifOp, aa);

  if (cand.thenNeedsSplit) {
    cleanupSide(cand.ifOp.thenBlock(), memGraph);
  }
  if (cand.elseNeedsSplit) {
    cleanupSide(cand.ifOp.elseBlock(), memGraph);
  }
  return success();
}

namespace {

class CloneCubeDepInIfPass
    : public PassWrapper<CloneCubeDepInIfPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CloneCubeDepInIfPass)

  CloneCubeDepInIfPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const final {
    return "clone-cube-dep-in-if";
  }

  llvm::StringRef getDescription() const final {
    return "Clone CUBE-block dependency chains inside scf.if ops so that each "
           "CUBE block owns its computations and is independent across the "
           "if split performed by SplitIfByBlockId.";
  }
};

} // namespace

void CloneCubeDepInIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (hasFallbackAttr(module)) {
    return;
  }

  LDBG("Before:\n" << module << "\n----------");

  // The memory-dependence graph is rebuilt PER CANDIDATE inside
  // processCandidate so that freshly cloned ops participate in the
  // exec-after edges (mirrors CloneOps' per-loop rebuild).
  auto &aa = getAnalysis<AliasAnalysis>();

  WalkResult walkRes = module->walk([&](scf::IfOp ifOp) -> WalkResult {
    CandidateIf cand = getCandidate(ifOp);
    if (!cand.thenNeedsSplit && !cand.elseNeedsSplit) {
      return WalkResult::advance();
    }
    LDBG("Processing if: " << ifOp);
    LDBG("  thenNeedsSplit=" << cand.thenNeedsSplit
                              << " elseNeedsSplit=" << cand.elseNeedsSplit
                              << " thenCubes=" << cand.thenCubes.size()
                              << " elseCubes=" << cand.elseCubes.size());
    if (failed(processCandidate(cand, aa))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkRes.wasInterrupted()) {
    LDBG("Clone cube deps in if failed, fallback to original");
    setFallbackAttr(module, ERRCODE_FAILED);
    return;
  }

  LDBG("After: \n" << module << "\n----------");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCloneCubeDepInIfPass() {
  return std::make_unique<CloneCubeDepInIfPass>();
}

} // namespace triton
} // namespace mlir