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

#include <algorithm>
#include <string_view>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "ReorderOpsByBlockIdPass";

#define DBGS(...) LLVM_DEBUG(llvm::dbgs() << __VA_ARGS__)
#define LOG_DEBUG(...) DBGS("\n[" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace triton;
using namespace CVPipeline;

namespace {

// A dependency DAG of both SSA and memory of the ops
struct BlockOpGraph {
  Block *block;
  ArrayRef<Operation *> ops;
  DenseMap<Operation *, unsigned> opIndex;               // op → position in ops
  DenseMap<Operation *, SmallVector<Operation *>> preds; // op → its defs
  DenseMap<Operation *, SmallVector<Operation *>> succs; // op → its uses
  BlockOpGraph(ArrayRef<Operation *> allOps, Block *block,
               const MemoryDependenceGraph &memGraph);
};

// Helper class to manage edges in OpGraph, mainly to reduce congitive
// complexity of the build function
struct EdgeHelper {
  BlockOpGraph &graph;
  DenseSet<std::pair<Operation *, Operation *>> seen;
  Block *block;

  // find the ancestor directly in the block, and in opIndex; return nullptr if
  // either fails
  Operation *resolveToBlockOp(Operation *op);

  void addEdge(Operation *pred, Operation *succ);

  void addEdgeToUser(Operation *op, Operation *user) {
    if (graph.opIndex.contains(user)) {
      return; // same-level use, already covered by the def-side loop
    }
    Operation *ancestor = resolveToBlockOp(user);
    addEdge(op, ancestor);
  };

  EdgeHelper(BlockOpGraph &g, Block *block) : graph(g), block(block) {};
};

} // namespace

Operation *EdgeHelper::resolveToBlockOp(Operation *op) {
  if (graph.opIndex.contains(op)) {
    return op;
  }
  Operation *ancestor = getAncestorInBlock(op, block);
  if (!ancestor || !graph.opIndex.contains(ancestor)) {
    return nullptr;
  }
  return ancestor;
}

void EdgeHelper::addEdge(Operation *pred, Operation *succ) {
  if (!pred || !succ || pred == succ) {
    return;
  }
  if (seen.insert({pred, succ}).second) {
    LOG_DEBUG("Adding edge from " << *pred << " to " << *succ);
    graph.succs[pred].push_back(succ);
    graph.preds[succ].push_back(pred);
  }
};

BlockOpGraph::BlockOpGraph(ArrayRef<Operation *> allOps, Block *block,
                           const MemoryDependenceGraph &memGraph)
    : block(block), ops(allOps) {
  for (unsigned i = 0; i < allOps.size(); ++i) {
    opIndex[allOps[i]] = i;
    preds[allOps[i]]; // ensure every node has an entry
    succs[allOps[i]];
  }

  EdgeHelper edges(*this, block);
  DependencyHelper depHelper{memGraph};

  for (Operation *op : allOps) {
    LOG_DEBUG("Processing op: " << *op);
    depHelper.forEachSource(op, [&](Operation *source) {
      Operation *def = edges.resolveToBlockOp(source);
      edges.addEdge(def, op);
    });
    depHelper.forEachUser(
        op, [&](Operation *user) { edges.addEdgeToUser(op, user); });
  }
}

static llvm::FailureOr<DenseMap<Operation *, int>>
collectBlockIds(ArrayRef<Operation *> allOps, ComputeBlockIdManager &bm) {
  DenseMap<Operation *, int> opBlockId;
  for (Operation *op : allOps) {
    if (llvm::failed(verifyOpBlockId(op))) {
      return llvm::failure();
    }
    auto blockIdOpt = getOpBlockId(op);
    if (blockIdOpt.has_value()) {
      opBlockId[op] = blockIdOpt.value();
      continue;
    }

    auto result = op->walk([&](Operation *nestedOp) {
      if (nestedOp != op &&
          !llvm::isa<scf::YieldOp, linalg::FillOp>(nestedOp)) {
        return WalkResult::interrupt();
      }
      auto currBlockIdOpt = getOpBlockId(nestedOp);
      if (!blockIdOpt.has_value()) {
        blockIdOpt = getOpBlockId(nestedOp);
      }
      if (currBlockIdOpt.has_value() && currBlockIdOpt != blockIdOpt) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted() || !blockIdOpt.has_value()) {
      blockIdOpt = bm.getNextId();
    } else {
      bm.updateBlockId(op, blockIdOpt.value());
    }
    opBlockId[op] = blockIdOpt.value();
  }
  return opBlockId;
}

namespace {

// Helper structure to hold the group-level graph data.
struct GroupAdjacencyGraph {
  Block *block;
  SmallVector<int> groupIds;
  SmallVector<SmallVector<unsigned>> succs;
  SmallVector<unsigned> inDeg;
  ComputeBlockIdManager &bm;
  GroupAdjacencyGraph(const BlockOpGraph &g,
                      const DenseMap<Operation *, int> &opBlockId,
                      ComputeBlockIdManager &bm);
  llvm::FailureOr<SmallVector<int>> computeTopologicalOrder();
};

} // namespace

/**
 * Step 1: Build the group-level dependency graph from operator-level edges.
 * Maps individual operations to their respective groups and identifies
 * dependencies between those groups.
 */
GroupAdjacencyGraph::GroupAdjacencyGraph(
    const BlockOpGraph &g, const DenseMap<Operation *, int> &opBlockId,
    ComputeBlockIdManager &bm)
    : block(g.block), bm(bm) {
  // 1. Collect distinct group IDs while preserving the first-appearance order.
  DenseSet<int> seenIds;
  for (Operation *op : g.ops) {
    int id = opBlockId.at(op);
    if (seenIds.insert(id).second) {
      groupIds.push_back(id);
    }
  }

  unsigned n = groupIds.size();
  succs.resize(n);
  inDeg.assign(n, 0);

  // Map group ID to its index in the groupIds vector for fast lookup.
  DenseMap<int, unsigned> groupPos;
  for (unsigned i = 0; i < n; ++i) {
    groupPos[groupIds[i]] = i;
  }

  // 2. Build group-level edges. Use a set to avoid duplicate edges between
  // groups.
  DenseSet<std::pair<unsigned, unsigned>> addedEdges;
  for (Operation *op : g.ops) {
    unsigned fromIdx = groupPos[opBlockId.at(op)];

    for (Operation *succ : g.succs.at(op)) {
      unsigned toIdx = groupPos[opBlockId.at(succ)];
      // Ignore intra-group dependencies and duplicate inter-group edges.
      if (fromIdx != toIdx && addedEdges.insert({fromIdx, toIdx}).second) {
        succs[fromIdx].push_back(toIdx);
        inDeg[toIdx]++;
      }
    }
  }

  // Logging the constructed group graph.
  LOG_DEBUG("Group-level edges:\n");
  for (unsigned i = 0; i < n; ++i) {
    DBGS("  Group " << groupIds[i] << " -> ");
    for (unsigned succIdx : succs[i]) {
      DBGS(groupIds[succIdx] << " ");
    }
    DBGS("\n");
  }
}

/**
 * Step 2: Perform a topological sort (Kahn's Algorithm) on the group graph.
 * Returns the group IDs in an order that satisfies all dependencies.
 */
llvm::FailureOr<SmallVector<int>>
GroupAdjacencyGraph::computeTopologicalOrder() {
  SmallVector<int> result;
  SmallVector<unsigned> ready; // Nodes with in-degree 0.
  unsigned n = groupIds.size();

  SmallVector<unsigned> startingVectorBlocks;
  for (auto [i, groupId] : llvm::enumerate(groupIds)) {
    if (inDeg[i] != 0) {
      continue;
    }
    auto ops = bm.getOpsRefByBlockId(groupId);
    if (ops.empty() ||
        getCoreTypeOfSimpleOpOrCf(ops.front()) == mlir::CVPipeline::CUBE_ONLY) {
      ready.push_back(i);
    } else {
      startingVectorBlocks.push_back(i);
    }
  }
  constexpr size_t kPriviledgedMaxComputeOpCnt = 1;
  std::stable_partition(startingVectorBlocks.begin(),
                        startingVectorBlocks.end(), [this](unsigned idx) {
                          const auto blockId = groupIds[idx];
                          const auto ops = bm.getOpsRefByBlockId(blockId);
                          auto computeOpCnt = 0;
                          for (auto op : ops) {
                            if (isTensorComputeOp(op)) {
                              computeOpCnt++;
                              LOG_DEBUG("Tensor compute op: " << *op);
                            } else {
                              LOG_DEBUG("Not tensor compute op: " << *op);
                            }
                          }
                          LOG_DEBUG("Summary: group id "
                                    << blockId
                                    << " compute ops: " << computeOpCnt);
                          return computeOpCnt > kPriviledgedMaxComputeOpCnt;
                        });
  ready.append(startingVectorBlocks);

  while (!ready.empty()) {
    auto cur = ready.pop_back_val();

    result.push_back(groupIds[cur]);

    for (unsigned succIdx : succs[cur]) {
      if (--inDeg[succIdx] == 0) {
        ready.push_back(succIdx);
      }
    }
  }

  LLVM_DEBUG({
    LOG_DEBUG("Group order: ");
    for (int id : result) {
      LOG_DEBUG(id << " ");
    }
    LOG_DEBUG("\n");
  });

  if (result.size() == n) {
    return result;
  }
  Operation *op = block->getParentOp();
  constexpr std::string_view kErrorPrefix =
      "Failed to compute topological order for ";
  if (!op) {
    llvm::errs() << kErrorPrefix
                 << "an unknown block that is not contained in an op";
    return llvm::failure();
  }
  size_t regionIdx = 0;
  bool found = false;
  for (auto [i, region] : llvm::enumerate(op->getRegions())) {
    for (auto &possibleBlock : region.getBlocks()) {
      if (&possibleBlock == block) {
        regionIdx = i;
      }
    }
  }
  op->emitError(kErrorPrefix) << "block in region " << regionIdx;
  return llvm::failure();
}

// Stable sort ops based on their group orders
static llvm::FailureOr<SmallVector<Operation *>>
buildReorderedOps(const BlockOpGraph &graph,
                  const DenseMap<Operation *, int> &opBlockId,
                  ComputeBlockIdManager &bm) {
  SmallVector<Operation *> reordered;
  GroupAdjacencyGraph adjacencyGraph{graph, opBlockId, bm};
  auto groupOrderResult = adjacencyGraph.computeTopologicalOrder();
  if (llvm::failed(groupOrderResult)) {
    return llvm::failure();
  }

  for (int const blockId : groupOrderResult.value()) {
    SmallVector<Operation *> storeOps;
    for (Operation *op : graph.ops) {
      if (opBlockId.at(op) == blockId) {
        if (isa<hivm::StoreOp, bufferization::MaterializeInDestinationOp>(op)) {
          storeOps.push_back(op);
          continue;
        }
        reordered.push_back(op);
      }
    }
    for (auto op : storeOps) {
      reordered.push_back(op);
    }
  }
  return reordered;
}

// Reorder the ops in the mlir representation
static void applyReorder(Block &block, ArrayRef<Operation *> reordered) {
  Operation *terminator =
      block.mightHaveTerminator() ? block.getTerminator() : nullptr;
  for (Operation *op : reordered) {
    op->moveBefore(&block, block.end());
  }

  if (terminator) {
    terminator->moveBefore(&block, block.end());
  }
}

static llvm::LogicalResult
reorderOpsInBlock(Block &block, const MemoryDependenceGraph &memGraph,
                  ComputeBlockIdManager &bm) {
  const auto allOps =
      llvm::to_vector(llvm::make_pointer_range(block.without_terminator()));

  const BlockOpGraph graph{allOps, &block, memGraph};
  llvm::FailureOr<DenseMap<Operation *, int>> opBlockIdOpt =
      collectBlockIds(allOps, bm);
  if (failed(opBlockIdOpt)) {
    return failure();
  }

  auto &opBlockId = *opBlockIdOpt;
  LOG_DEBUG("Initial opBlockIds:\n");
  for (Operation *op : allOps) {
    LOG_DEBUG("  Op: " << *op << ", opBlockId = " << opBlockId[op] << "\n");
  }

  const auto reorderedRes = buildReorderedOps(graph, opBlockId, bm);
  if (failed(reorderedRes)) {
    return failure();
  }

  applyReorder(block, reorderedRes.value());

  // Verify the sync fence invariant: every op that preceded (followed) a
  // gpu.barrier / hivm.sync_block_all in the original source order must still
  // precede (follow) it.
  LLVM_DEBUG({
    DenseMap<Operation *, unsigned> sourceIdx;
    for (unsigned i = 0; i < allOps.size(); ++i) {
      sourceIdx[allOps[i]] = i;
    }
    for (Operation &op : block) {
      if (!CVPipeline::isSyncOp(&op)) {
        continue;
      }
      bool seenBarrier = false;
      unsigned barrierIdx = sourceIdx[&op];
      for (Operation &it : block) {
        if (&it == &op) {
          seenBarrier = true;
          continue;
        }
        if (!sourceIdx.contains(&it)) {
          continue;
        }
        unsigned idx = sourceIdx.at(&it);
        if (seenBarrier && idx < barrierIdx) {
          LOG_DEBUG("Barrier fence violated: op after barrier in source moved "
                    << "before it: " << it << "\n");
        }
        if (!seenBarrier && idx > barrierIdx) {
          LOG_DEBUG("Barrier fence violated: op before barrier in source moved "
                    << "after it: " << it << "\n");
        }
      }
    }
  });

  return llvm::success();
}

void ReorderOpsByBlockIdPass::runOnOperation() {
  OpBuilder const builder(&getContext());

  auto moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  LOG_DEBUG("Input mlir:\n" << moduleOp << "\n");
  llvm::dbgs().flush();

  auto &aa = getAnalysis<AliasAnalysis>();
  auto memGraph = MemoryDependenceGraph(moduleOp, aa);
  auto bm = ComputeBlockIdManager(moduleOp);
  auto result = moduleOp.walk([&](Block *block) {
    auto *parentOp = block->getParentOp();
    if (!parentOp ||
        // whitelist ops to reorder
        !(isa<func::FuncOp>(parentOp) ||
          isa<scf::SCFDialect>(parentOp->getDialect()))) {
      return WalkResult::skip();
    }
    if (llvm::failed(reorderOpsInBlock(*block, memGraph, bm))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LOG_DEBUG("Output mlir:\n" << moduleOp << "\n");
  LOG_DEBUG("=== Pass TuningOpSeq complete ===\n");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createReorderOpsByBlockIdPass() {
  return std::make_unique<ReorderOpsByBlockIdPass>();
}
