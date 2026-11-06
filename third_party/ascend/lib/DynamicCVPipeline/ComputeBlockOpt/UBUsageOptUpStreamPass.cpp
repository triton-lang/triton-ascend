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
#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <limits>

#define DEBUG_TYPE "ub-usage-opt-up-stream"
#define LOG_DEBUG(msg)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg)

using namespace mlir;
using namespace mlir::triton;

namespace {
// Dynamic tensors are unsupported transfers, rather than a byte estimate.
// Compare their count first, then the statically known transfer size.
struct UBSize {
  int64_t dynamic = 0;
  int64_t bytes = 0;
  bool operator<(const UBSize &other) const {
    return std::make_pair(dynamic, bytes) <
           std::make_pair(other.dynamic, other.bytes);
  }
};

struct UBUsageGraph {
  DenseMap<Operation *, int> op2nodeId;
  SmallVector<Operation *> nodeId2op;
  SmallVector<SmallVector<int>> linkIn, linkOut;
  SmallVector<int> nodeBlockId;
};

class UBUsageOptUpStreamPass
    : public PassWrapper<UBUsageOptUpStreamPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UBUsageOptUpStreamPass)
  StringRef getArgument() const final { return "ub-usage-opt-up-stream"; }
  void runOnOperation() override;

private:
  void buildUBUsageGraph(Block *block, UBUsageGraph &graph,
                         const CVPipeline::MemoryDependenceGraph &memGraph,
                         CVPipeline::ComputeBlockIdManager &bm);
  SmallVector<int> collectNeedUbOpts(const UBUsageGraph &graph);
  bool isActiveEndNode(int optNode, int activateNode,
                       const UBUsageGraph &graph);
  UBSize sumIncomingLinkSize(Block *block, const UBUsageGraph &graph,
                             ArrayRef<int> chain, int optBlockId,
                             CVPipeline::ComputeBlockIdManager &bm);
  DenseMap<int, int>
  collectRecordChange(Block *block, ArrayRef<int> needUbOpts,
                      const UBUsageGraph &graph,
                      const CVPipeline::MemoryDependenceGraph &memGraph,
                      CVPipeline::ComputeBlockIdManager &bm);
  bool applyRecordChange(const DenseMap<int, int> &recordChange,
                         const UBUsageGraph &graph,
                         const CVPipeline::MemoryDependenceGraph &memGraph,
                         CVPipeline::ComputeBlockIdManager &bm);
  LogicalResult
  UBUsageOptimization(Block *block,
                      const CVPipeline::MemoryDependenceGraph &memGraph,
                      CVPipeline::ComputeBlockIdManager &bm);
};

void UBUsageOptUpStreamPass::buildUBUsageGraph(
    Block *block, UBUsageGraph &graph,
    const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  for (Operation &op : *block) {
    graph.op2nodeId[&op] = graph.nodeId2op.size();
    graph.nodeId2op.push_back(&op);
    graph.nodeBlockId.push_back(bm.getBlockIdByOp(&op));
    graph.linkIn.emplace_back();
    graph.linkOut.emplace_back();
  }
  CVPipeline::DependencyHelper depHelper(memGraph);
  for (Operation &op : *block) {
    int dst = graph.op2nodeId.lookup(&op);
    depHelper.forEachSourceInSameBlock<
        CVPipeline::DependencyHelper::SourceMode::AcrossIterArg>(
        &op, [&](Operation *source) {
          auto it = graph.op2nodeId.find(source);
          if (it == graph.op2nodeId.end())
            return;
          int src = it->second;
          if (llvm::is_contained(graph.linkIn[dst], src))
            return;
          graph.linkIn[dst].push_back(src);
          graph.linkOut[src].push_back(dst);
        });
  }
}

bool UBUsageOptUpStreamPass::isActiveEndNode(int optNode, int activateNode,
                                             const UBUsageGraph &graph) {
  Operation *op = graph.nodeId2op[activateNode];
  int optBlockId = graph.nodeBlockId[optNode];
  if (graph.nodeBlockId[activateNode] < 0 ||
      graph.nodeBlockId[activateNode] == optBlockId ||
      CVPipeline::getOpCoreType(op) != CVPipeline::VECTOR_ONLY ||
      op->hasTrait<OpTrait::IsTerminator>() || !isMemoryEffectFree(op) ||
      (op->getNumRegions() != 0 && !isa<linalg::LinalgOp>(op)) ||
      op->getNumResults() == 0)
    return false;
  // Memory-bearing operations and control flow are handled by other passes.
  if (llvm::any_of(op->getOperandTypes(),
                   [](Type t) { return isa<BaseMemRefType>(t); }) ||
      llvm::any_of(op->getResultTypes(),
                   [](Type t) { return isa<BaseMemRefType>(t); }))
    return false;
  // Do not move a producer away from another consumer block (including yield).
  return !graph.linkOut[activateNode].empty() &&
         llvm::all_of(graph.linkOut[activateNode], [&](int user) {
           return graph.nodeBlockId[user] == optBlockId;
         });
}

SmallVector<int>
UBUsageOptUpStreamPass::collectNeedUbOpts(const UBUsageGraph &graph) {
  SmallVector<int> needUbOpts;
  for (int optNode = 0; optNode < (int)graph.nodeId2op.size(); ++optNode) {
    if (graph.nodeBlockId[optNode] < 0 ||
        CVPipeline::getOpCoreType(graph.nodeId2op[optNode]) !=
            CVPipeline::VECTOR_ONLY)
      continue;
    if (llvm::any_of(graph.linkIn[optNode], [&](int activateNode) {
          return isActiveEndNode(optNode, activateNode, graph);
        }))
      needUbOpts.push_back(optNode);
  }
  return needUbOpts;
}

UBSize UBUsageOptUpStreamPass::sumIncomingLinkSize(
    Block *block, const UBUsageGraph &graph, ArrayRef<int> chain,
    int optBlockId, CVPipeline::ComputeBlockIdManager &bm) {
  DenseSet<Operation *> moved;
  for (int node : chain)
    moved.insert(graph.nodeId2op[node]);
  auto getBlockId = [&](Operation *op) {
    return moved.contains(op) ? optBlockId : bm.getBlockIdByOp(op);
  };
  DenseSet<std::pair<Value, int>> visited;
  UBSize totalSize;
  for (Operation &blockOp : *block) {
    int dstBlockId = getBlockId(&blockOp);
    blockOp.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        auto shaped = dyn_cast<ShapedType>(operand.getType());
        if (!shaped || isa<BaseMemRefType>(operand.getType()))
          continue;
        Operation *def = operand.getDefiningOp();
        // Resolve loop-carried tensors to their yielded defining operation.
        if (!def)
          def = CVPipeline::getLoopCarriedDefOp(operand, block);
        auto *src = def ? CVPipeline::getAncestorInBlock(def, block) : nullptr;
        if (src == &blockOp || (src && getBlockId(src) == dstBlockId))
          continue;
        // Values defined outside this block are boundary inputs as well.
        if (!src && def && bm.getBlockIdByOp(def) == dstBlockId)
          continue;
        if (!visited.insert({operand, dstBlockId}).second)
          continue;
        if (!shaped.hasStaticShape()) {
          ++totalSize.dynamic;
          continue;
        }
        auto elem = shaped.getElementType();
        int64_t elemBytes =
            elem.isIntOrFloat()
                ? std::max<int64_t>(1, (elem.getIntOrFloatBitWidth() + 7) / 8)
                : 1;
        // Saturate instead of overflowing transfer estimates.
        int64_t size = elemBytes;
        for (int64_t dim : shaped.getShape()) {
          if (dim && size > std::numeric_limits<int64_t>::max() / dim) {
            size = std::numeric_limits<int64_t>::max();
            break;
          }
          size *= dim;
        }
        totalSize.bytes += std::min(size, std::numeric_limits<int64_t>::max() -
                                              totalSize.bytes);
      }
    });
  }
  return totalSize;
}

DenseMap<int, int> UBUsageOptUpStreamPass::collectRecordChange(
    Block *block, ArrayRef<int> needUbOpts, const UBUsageGraph &graph,
    const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  // Compare the complete boundary: all operands/results and all users matter.
  UBSize originUBSize = sumIncomingLinkSize(block, graph, {}, -1, bm);
  for (int optNode : needUbOpts) {
    int optBlockId = graph.nodeBlockId[optNode];
    SmallVector<int> activateSet;
    for (int activateNode : graph.linkIn[optNode])
      if (isActiveEndNode(optNode, activateNode, graph))
        activateSet.push_back(activateNode);
    for (int activateNode : activateSet) {
      SmallVector<int> chain;
      int curNode = activateNode;
      UBSize minUBSize = originUBSize;
      int bestCutPointIdx = -1;
      while (true) {
        chain.push_back(curNode);
        auto nowUBSize =
            sumIncomingLinkSize(block, graph, chain, optBlockId, bm);
        SmallVector<Operation *> willaddOps;
        for (int node : chain)
          willaddOps.push_back(graph.nodeId2op[node]);
        if (nowUBSize < minUBSize &&
            !CVPipeline::willCreateCycle(willaddOps, memGraph, optBlockId,
                                         bm)) {
          minUBSize = nowUBSize;
          bestCutPointIdx = chain.size();
        }
        // Reverse single-chain search. Shared scalar dependencies stay put.
        SmallVector<int> dependNodes;
        for (int source : graph.linkIn[curNode]) {
          Operation *op = graph.nodeId2op[source];
          if (graph.nodeBlockId[source] != graph.nodeBlockId[activateNode] ||
              llvm::is_contained(chain, source) ||
              (op->getNumRegions() != 0 && !isa<linalg::LinalgOp>(op)) ||
              op->hasTrait<OpTrait::IsTerminator>() ||
              !isMemoryEffectFree(op) ||
              CVPipeline::getOpCoreType(op) != CVPipeline::VECTOR_ONLY ||
              llvm::any_of(op->getOperandTypes(),
                           [](Type t) { return isa<BaseMemRefType>(t); }) ||
              llvm::any_of(op->getResultTypes(),
                           [](Type t) { return isa<BaseMemRefType>(t); }))
            continue;
          if (llvm::all_of(graph.linkOut[source], [&](int user) {
                return llvm::is_contained(chain, user) ||
                       graph.nodeBlockId[user] == optBlockId;
              }))
            dependNodes.push_back(source);
        }
        if (dependNodes.size() != 1)
          break;
        curNode = dependNodes.front();
      }
      if (bestCutPointIdx > 0) {
        DenseMap<int, int> recordChange;
        for (int i = 0; i < bestCutPointIdx; ++i)
          recordChange[chain[i]] = optBlockId;
        // Apply one group, then rebuild candidates using the updated IDs.
        return recordChange;
      }
    }
  }
  return DenseMap<int, int>();
}

bool UBUsageOptUpStreamPass::applyRecordChange(
    const DenseMap<int, int> &recordChange, const UBUsageGraph &graph,
    const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  if (recordChange.empty())
    return false;
  int targetBlockId = recordChange.begin()->second;
  SmallVector<Operation *> willaddOps;
  // Preserve IR order rather than DenseMap iteration order.
  for (int node = 0; node < (int)graph.nodeId2op.size(); ++node)
    if (recordChange.contains(node))
      willaddOps.push_back(graph.nodeId2op[node]);
  if (CVPipeline::willCreateCycle(willaddOps, memGraph, targetBlockId, bm))
    return false;
  for (Operation *op : willaddOps) {
    LOG_DEBUG("Change " << *op << " TO " << targetBlockId << "\n");
    bm.updateBlockIdWithInner(op, targetBlockId);
  }
  return true;
}

LogicalResult UBUsageOptUpStreamPass::UBUsageOptimization(
    Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  if (!isa<scf::ForOp, scf::WhileOp>(block->getParentOp()))
    return success();
  // Each accepted update strictly decreases the boundary cost.
  while (true) {
    UBUsageGraph graph;
    buildUBUsageGraph(block, graph, memGraph, bm);
    auto needUbOpts = collectNeedUbOpts(graph);
    auto recordChange =
        collectRecordChange(block, needUbOpts, graph, memGraph, bm);
    if (!applyRecordChange(recordChange, graph, memGraph, bm))
      break;
  }
  return success();
}

void UBUsageOptUpStreamPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module))
    return;
  auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph(module, aliasAnalysis);
  CVPipeline::ComputeBlockIdManager bm(module);
  SmallVector<Block *> blocks;
  module.walk([&](Block *block) { blocks.push_back(block); });
  for (Block *block : blocks)
    if (failed(UBUsageOptimization(block, memGraph, bm)))
      LOG_DEBUG("UB usage upstream optimization failed in block.\n");
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createUBUsageOptUpStreamPass() {
  return std::make_unique<UBUsageOptUpStreamPass>();
}
