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
#include <functional>
#include <string_view>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
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
#define LOG_DEBUG(...) DBGS("[" << DEBUG_TYPE << "] " << __VA_ARGS__)

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

static bool isStoreLikeWithRegion(Operation *op) {
  auto ret = op->walk([&](Operation *subOp) {
    if (isa<hivm::StoreOp, bufferization::MaterializeInDestinationOp>(subOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return ret.wasInterrupted();
}

static SmallVector<Operation *>
orderInOneCBlock(ArrayRef<Operation *> opsInSameBlock,
                 const MemoryDependenceGraph &memGraph) {
  // reorder in one compute block following rules:
  // 1. vecoter block should sink storeLike Op. (only Vector)
  // 2. Other
  SmallVector<Operation *> originOrder(opsInSameBlock.begin(),
                                       opsInSameBlock.end());
  if (llvm::any_of(opsInSameBlock, [&](Operation *op) {
        return CVPipeline::getCoreTypeOfSimpleOpOrCf(op) !=
               CVPipeline::VECTOR_ONLY;
      })) {
    // If this is one CUBE block, storeLike Ops no need to sink down.
    // Considering
    //  1. CUBE block's store is always use FIXPIPE
    //  2. C->V always just next to matmul.
    // So there are no conflict between store and  inter transfer
    return originOrder;
  }

  if (llvm::all_of(opsInSameBlock,
                   [&](Operation *op) { return !isStoreLikeWithRegion(op); })) {
    // If there are no store-like op, early return.
    return originOrder;
  }
  Block *block = opsInSameBlock.front()->getBlock();
  BlockOpGraph graph{opsInSameBlock, block, memGraph};

  // Kahn's topological sort. Track in-degree per op and seed the ready set
  // with all ops that have no predecessors.
  DenseMap<Operation *, unsigned> inDeg;
  SmallVector<Operation *> ready;
  for (Operation *op : opsInSameBlock) {
    inDeg[op] = graph.preds.at(op).size();
    if (inDeg[op] == 0) {
      ready.push_back(op);
    }
  }

  // Tie-breaker among ready (in-degree 0) ops:
  // 1. Non-store-like ops come first (sink store-like ops to the end).
  // 2. The op with a smaller opIndex wins (preserve original program order).
  auto comesBefore = [&](Operation *a, Operation *b) {
    bool aStore = isStoreLikeWithRegion(a);
    bool bStore = isStoreLikeWithRegion(b);
    if (aStore != bStore) {
      return !aStore;
    }
    return graph.opIndex.at(a) < graph.opIndex.at(b);
  };

  SmallVector<Operation *> ordered;
  ordered.reserve(opsInSameBlock.size());
  while (!ready.empty()) {
    // Pick the best candidate under the tie-breaking rules.
    auto bestIt = std::min_element(ready.begin(), ready.end(), comesBefore);
    Operation *cur = *bestIt;
    ready.erase(bestIt);

    ordered.push_back(cur);

    // Release successors; any that drop to in-degree 0 become ready.
    for (Operation *succ : graph.succs.at(cur)) {
      if (--inDeg[succ] == 0) {
        ready.push_back(succ);
      }
    }
  }

  return ordered;
}

// Ops that materialize kernel inputs (GM loads, views, local scratch buffers
// and scalar index arithmetic). A chain made only of these ops is a pure
// function of its operands and can be safely cloned per consumer block.
static bool isCloneableInputChainOp(Operation *op) {
  if (isa<
          arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
          arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
          arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::ExtSIOp,
          arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp, arith::MaxSIOp,
          arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp, arith::CmpIOp,
          arith::SelectOp, memref::ReinterpretCastOp, memref::SubViewOp,
          memref::AllocOp, memref::AllocaOp, memref::CastOp, memref::CopyOp,
          memref::LoadOp, bufferization::ToTensorOp, tensor::EmptyOp,
          tensor::ExtractSliceOp, tensor::CastOp, tensor::CollapseShapeOp,
          tensor::ExpandShapeOp, linalg::FillOp, linalg::CopyOp>(op)) {
    return true;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // Conditionally filled scratch buffers are cloneable as long as every op
    // inside the regions is itself a cloneable input-chain op.
    auto isScalarCond = !isa<ShapedType>(ifOp.getCondition().getType());
    if (!isScalarCond) {
      return false;
    }
    return !ifOp->walk<WalkOrder::PreOrder>([&](Operation *nested) {
             if (nested == ifOp.getOperation() ||
                 nested->hasTrait<OpTrait::IsTerminator>()) {
               return WalkResult::advance();
             }
             return isCloneableInputChainOp(nested) ? WalkResult::advance()
                                                    : WalkResult::interrupt();
           }).wasInterrupted();
  }
  return false;
}

// Walk up from `defOp` and collect the cloneable chain of ops that materialize
// the value. Ops in `targetGroup` (already where we clone to) and in
// non-cyclic groups are referenced, not cloned. The chain must not touch any
// other cyclic group, otherwise cloning could just move the cycle around.
// `chainTopo` is filled upstream-first so cloning in order keeps defs before
// uses.
// Ops that (indirectly) write into a memref value through view chains, e.g.
// the GM->UB copy and the conditional zero-fill of a load scratch buffer.
static Value getMemrefWriteDest(Operation *op) {
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    return copy.getTarget();
  }
  if (auto fill = dyn_cast<linalg::FillOp>(op)) {
    if (!fill.getOutputs().empty()) {
      return fill.getOutputs()[0];
    }
    return nullptr;
  }
  if (auto mat = dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
    return mat.getDest();
  }
  return nullptr;
}

// Collect top-level ops in `mlirBlock` whose write destination reaches
// `alloc` through memref view chains (subview / reinterpret_cast / cast).
static SmallVector<Operation *> findWritersOf(Operation *alloc,
                                              Block *mlirBlock) {
  SmallVector<Operation *> writers;
  DenseSet<Operation *> seen;
  mlirBlock->walk([&](Operation *w) {
    Value dest = getMemrefWriteDest(w);
    if (!dest) {
      return WalkResult::advance();
    }
    Value v = dest;
    while (Operation *d = v.getDefiningOp()) {
      if (d == alloc) {
        Operation *topLevel = mlirBlock->findAncestorOpInBlock(*w);
        if (topLevel && seen.insert(topLevel).second) {
          writers.push_back(topLevel);
        }
        return WalkResult::advance();
      }
      if (isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp>(d)) {
        v = d->getOperand(0);
        continue;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  return writers;
}

static bool collectPureInputChain(
    Operation *defOp, int sourceGroup, int targetGroup, Block *mlirBlock,
    const DenseMap<Operation *, int> &opBlockId,
    const DenseSet<int> &cyclicGroups, SmallPtrSetImpl<Operation *> &visited,
    SmallVectorImpl<Operation *> &chainTopo) {
  std::function<bool(Operation *)> visit = [&](Operation *op) -> bool {
    if (visited.contains(op)) {
      return true;
    }
    if (!isCloneableInputChainOp(op)) {
      LOG_DEBUG("chain bail: non-whitelisted op: " << *op << "\n");
      return false;
    }
    visited.insert(op);
    // Writers of a local scratch buffer produce the buffer's contents, so
    // they must move together with it (SSA operands alone would miss them).
    if (isa<memref::AllocOp>(op)) {
      for (Operation *writer : findWritersOf(op, mlirBlock)) {
        auto it = opBlockId.find(writer);
        if (it == opBlockId.end()) {
          return false; // unknown group: be conservative
        }
        int writerGroup = it->second;
        if (writerGroup == sourceGroup) {
          if (!visit(writer)) {
            return false;
          }
          continue;
        }
        if (cyclicGroups.contains(writerGroup)) {
          LOG_DEBUG("chain bail: writer in cyclic group " << writerGroup
                                                          << ": " << *writer << "\n");
          return false; // writer inside the cycle but outside source group
        }
        // Non-cyclic writer group: stays where it is; the moved buffer keeps
        // depending on it through the regular dependency machinery.
      }
    }
    for (OpOperand &operand : op->getOpOperands()) {
      Operation *def = operand.get().getDefiningOp();
      if (!def) {
        continue; // block arg / iter arg / func arg: reference as-is
      }
      Operation *ancestor = mlirBlock->findAncestorOpInBlock(*def);
      if (!ancestor || ancestor == op) {
        continue; // defined outside this MLIR block: reference as-is
      }
      auto it = opBlockId.find(ancestor);
      if (it == opBlockId.end()) {
        continue; // unknown group: reference as-is
      }
      int defGroup = it->second;
      if (defGroup == sourceGroup) {
        if (ancestor != defOp && !visit(ancestor)) {
          return false; // extend the chain within the source group
        }
        continue;
      }
      if (defGroup == targetGroup) {
        continue; // already materialized where we clone to: reference
      }
      if (cyclicGroups.contains(defGroup)) {
        LOG_DEBUG("chain bail: ancestor in cyclic group " << defGroup << ": "
                                                          << *ancestor << "\n");
        return false; // would drag another cyclic group into the chain
      }
      continue; // non-cyclic group: reference as-is
    }
    chainTopo.push_back(op);
    return true;
  };
  return visit(defOp);
}

// Groups that remain after trimming in-degree-zero nodes are stuck in (or
// downstream of) a cycle in the group-level dependency graph.
static DenseSet<int> computeCyclicGroups(
    const BlockOpGraph &graph, const DenseMap<Operation *, int> &opBlockId) {
  DenseSet<int> groups;
  for (Operation *op : graph.ops) {
    auto it = opBlockId.find(op);
    if (it != opBlockId.end()) {
      groups.insert(it->second);
    }
  }
  DenseMap<int, SmallVector<int>> groupSuccs;
  DenseMap<int, unsigned> inDeg;
  DenseSet<std::pair<int, int>> seen;
  for (Operation *pred : graph.ops) {
    int a = opBlockId.at(pred);
    for (Operation *succ : graph.succs.at(pred)) {
      int b = opBlockId.at(succ);
      if (a == b || !seen.insert({a, b}).second) {
        continue;
      }
      groupSuccs[a].push_back(b);
      inDeg[b]++;
    }
  }
  SmallVector<int> ready;
  for (int g : groups) {
    if (inDeg[g] == 0) {
      ready.push_back(g);
    }
  }
  DenseSet<int> removed;
  while (!ready.empty()) {
    int g = ready.pop_back_val();
    removed.insert(g);
    for (int s : groupSuccs[g]) {
      if (--inDeg[s] == 0) {
        ready.push_back(s);
      }
    }
  }
  DenseSet<int> cyclic;
  for (int g : groups) {
    if (!removed.contains(g)) {
      cyclic.insert(g);
    }
  }
  return cyclic;
}

static std::optional<StringRef> getGroupCoreType(Operation *opInGroup) {
  if (auto attr = opInGroup->getAttrOfType<StringAttr>(
          static_cast<StringRef>(CVPipeline::kCoreType))) {
    return attr.getValue();
  }
  return std::nullopt;
}

// Try to break cross-block cycles by splitting pure input chains (GM loads,
// views, scratch fills) out of their producing block into a dedicated new
// block, so the producing block no longer appears as a dependency of the
// consuming block. The chain stays a single shared load; consumers receive it
// through the regular inter-block dependency machinery. Only same-core group
// pairs are handled; anything else is left to the caller's fallback.
static bool breakCyclesBySplittingInputChains(
    const BlockOpGraph &graph, const DenseMap<Operation *, int> &opBlockId,
    Block *mlirBlock, ComputeBlockIdManager &bm) {
  DenseSet<int> cyclicGroups = computeCyclicGroups(graph, opBlockId);
  if (cyclicGroups.empty()) {
    return false;
  }

  for (Operation *succ : graph.ops) {
    auto succIt = opBlockId.find(succ);
    if (succIt == opBlockId.end() || !cyclicGroups.contains(succIt->second)) {
      continue;
    }
    int targetGroup = succIt->second;
    auto targetCore = getGroupCoreType(succ);
    if (!targetCore) {
      continue;
    }
    for (OpOperand &operand : succ->getOpOperands()) {
      Value value = operand.get();
      Operation *def = value.getDefiningOp();
      if (!def) {
        continue;
      }
      Operation *ancestor = mlirBlock->findAncestorOpInBlock(*def);
      if (!ancestor || ancestor == succ) {
        continue;
      }
      auto defIt = opBlockId.find(ancestor);
      if (defIt == opBlockId.end()) {
        continue;
      }
      int sourceGroup = defIt->second;
      if (sourceGroup == targetGroup || !cyclicGroups.contains(sourceGroup)) {
        continue;
      }
      auto sourceCore = getGroupCoreType(ancestor);
      if (!sourceCore || *sourceCore != *targetCore) {
        continue; // cross-core cycles are not handled here
      }

      SmallPtrSet<Operation *, 16> visited;
      SmallVector<Operation *> chainTopo;
      if (!collectPureInputChain(ancestor, sourceGroup, targetGroup, mlirBlock,
                                 opBlockId, cyclicGroups, visited, chainTopo)) {
        continue;
      }
      if (chainTopo.empty()) {
        continue;
      }

      OpBuilder builder(succ);
      // Move the chain ops into the consumer group (same core). The edge
      // producer->consumer is thereby reversed: the consumer now hosts the
      // shared input chain and the producing block only depends on it, so
      // the group graph becomes acyclic. The load stays single; cross-block
      // value delivery is handled by the regular inter-block dependency
      // machinery.
      int loadGroupId = targetGroup;
      for (Operation *chainOp : chainTopo) {
        // WithInner so ops nested in moved regions (e.g. the conditional fill
        // inside a scf.if) follow the new block id as well.
        bm.updateBlockIdWithInner(chainOp, loadGroupId);
        LOG_DEBUG("Cycle break: moved pure input chain op into new group "
                  << loadGroupId << ": " << *chainOp << "\n");
      }
      LOG_DEBUG("Cycle break: group " << sourceGroup << " -> " << targetGroup
                                      << " detached by splitting input chain "
                                         "into group "
                                      << loadGroupId << "\n");
      return true;
    }
  }
  return false;
}

// Stable sort ops based on their group orders
static llvm::FailureOr<SmallVector<Operation *>> buildReorderedOps(
    const BlockOpGraph &graph, const DenseMap<Operation *, int> &opBlockId,
    ComputeBlockIdManager &bm, const MemoryDependenceGraph &memGraph) {
  SmallVector<Operation *> reordered;
  GroupAdjacencyGraph adjacencyGraph{graph, opBlockId, bm};
  auto groupOrderResult = adjacencyGraph.computeTopologicalOrder();
  if (llvm::failed(groupOrderResult)) {
    return llvm::failure();
  }

  for (int const blockId : groupOrderResult.value()) {
    SmallVector<Operation *>
        originOrderOp; // collect ops following program order.
    for (Operation *op : graph.ops) {
      if (opBlockId.at(op) == blockId) {
        originOrderOp.push_back(op);
      }
    }
    SmallVector<Operation *> orderedInOneCBlock =
        orderInOneCBlock(originOrderOp, memGraph);
    reordered.append(orderedInOneCBlock);
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
  // If the group-level dependency graph is cyclic (e.g. a GM input load was
  // planned into one CUBE block while another block consuming it also feeds
  // back into the first), try to detach the offending input chains by cloning
  // them into a dedicated block before giving up.
  constexpr unsigned kMaxCycleBreakAttempts = 4;
  for (unsigned attempt = 0;; ++attempt) {
    const auto allOps =
        llvm::to_vector(llvm::make_pointer_range(block.without_terminator()));

    const BlockOpGraph graph{allOps, &block, memGraph};
    llvm::FailureOr<DenseMap<Operation *, int>> opBlockIdOpt =
        collectBlockIds(allOps, bm);
    if (failed(opBlockIdOpt)) {
      return failure();
    }

    auto &opBlockId = *opBlockIdOpt;
    if (attempt == 0) {
      LOG_DEBUG("Initial opBlockIds:\n");
      for (Operation *op : allOps) {
        LOG_DEBUG("  Op: " << *op << ", opBlockId = " << opBlockId[op] << "\n");
      }
    }

    const auto reorderedRes = buildReorderedOps(graph, opBlockId, bm, memGraph);
    if (succeeded(reorderedRes)) {
      applyReorder(block, reorderedRes.value());

      // Verify the sync fence invariant: every op that preceded (followed) a
      // gpu.barrier / hivm.sync_block_all in the original source order must
      // still precede (follow) it.
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
              LOG_DEBUG("Barrier fence violated: op after barrier in source "
                        "moved "
                        << "before it: " << it << "\n");
            }
            if (!seenBarrier && idx > barrierIdx) {
              LOG_DEBUG("Barrier fence violated: op before barrier in source "
                        "moved "
                        << "after it: " << it << "\n");
            }
          }
        }
      });

      return llvm::success();
    }

    if (attempt >= kMaxCycleBreakAttempts) {
      return failure();
    }
    LOG_DEBUG("Topological order failed (attempt " << attempt
                                                   << "); trying to break the "
                                                      "cycle by splitting pure "
                                                      "input chains\n");
    if (!breakCyclesBySplittingInputChains(graph, opBlockId, &block, bm)) {
      LOG_DEBUG("No splittable pure input chain found to break the cycle; "
                "falling back as before\n");
      return failure();
    }
  }
}

void ReorderOpsByBlockIdPass::runOnOperation() {
  OpBuilder const builder(&getContext());

  auto moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  // MergeComputeBlockPass sets kMergeComputeBlockApplied to record whether it
  // actually merged blocks. Skip reorder only when it ran but merged nothing;
  // consume the marker either way so it does not leak into the output IR.
  if (auto applied = moduleOp->getAttrOfType<BoolAttr>(
          CVPipeline::kMergeComputeBlockApplied)) {
    moduleOp->removeAttr(CVPipeline::kMergeComputeBlockApplied);
    if (!applied.getValue()) {
      LOG_DEBUG("Skip reorder: MergeComputeBlock ran but merged nothing");
      return;
    }
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
