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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CrossCorePattern.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <utility>

static constexpr const char *DEBUG_TYPE = "CrossCorePattern";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

namespace {

// The core a computation block runs on.
enum class CoreKind { Unknown, Cube, Vector };

// A computation block: the scope that holds it and its block id. The block id
// is only unique inside a scope, so the scope is part of the identity.
using BlockNode = std::pair<Operation *, int>;

// The members of a cross-core dependency group: the ops that produce the
// group and the ops that consume it.
struct GroupMembers {
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
};

// A cross-core edge: the block that consumes the direction and the tightly
// coupled buffers the whole direction carries.
struct Edge {
  unsigned consumer;
  SmallVector<int> buffers;
};

// Every cross-core transfer of the validated pipeline is double buffered: the
// producer writes two tightly coupled buffers and the consumer reads them in
// turn. A chain of `chainLength` blocks has `chainLength - 1` transfers, so it
// carries 2 * (chainLength - 1) tightly coupled buffers.
constexpr unsigned kBuffersPerTransfer = 2;

// The core of the block that owns `op`: the core of the scope holding it, or
// the core the op declares itself when it does not live in a scope.
CoreKind getBlockCore(Operation *op) {
  if (auto scopeOp = op->getParentOfType<scope::ScopeOp>()) {
    if (auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
            CVPipeline::kTcoreType)) {
      if (attr.getTcoretype() == hivm::TCoreType::CUBE)
        return CoreKind::Cube;
      if (attr.getTcoretype() == hivm::TCoreType::VECTOR)
        return CoreKind::Vector;
    }
  }
  // No scope, or a scope that does not pin a single core: fall back to the
  // core the op declares itself.
  switch (CVPipeline::getOpCoreType(op)) {
  case CVPipeline::CUBE_ONLY:
    return CoreKind::Cube;
  case CVPipeline::VECTOR_ONLY:
    return CoreKind::Vector;
  default:
    return CoreKind::Unknown;
  }
}

// The block the op belongs to, or nullopt when the op has no block id.
std::optional<BlockNode> getBlockNode(Operation *op) {
  auto blockId = CVPipeline::getOpBlockId(op);
  if (!blockId)
    return std::nullopt;
  Operation *scopeOp = nullptr;
  if (auto scope = op->getParentOfType<scope::ScopeOp>())
    scopeOp = scope.getOperation();
  return BlockNode{scopeOp, *blockId};
}

// Collect the cross-core dependency groups of the module. The attribute is a
// flat list of pairs: [group, role, group, role, ...], role 1=producer,
// 0=consumer. One op may take part in several groups.
void collectCrossCoreGroups(ModuleOp module,
                            llvm::DenseMap<int, GroupMembers> &groups) {
  module.walk([&](Operation *op) {
    auto deps = op->getAttrOfType<ArrayAttr>(CVPipeline::kCrossCoreDeps);
    if (!deps || deps.size() % 2 != 0)
      return;
    for (unsigned i = 0; i < deps.size(); i += 2) {
      auto groupAttr = dyn_cast<IntegerAttr>(deps[i]);
      auto roleAttr = dyn_cast<IntegerAttr>(deps[i + 1]);
      if (!groupAttr || !roleAttr)
        continue;
      GroupMembers &members = groups[groupAttr.getInt()];
      if (roleAttr.getInt() == CVPipeline::crossCoreProducerId) {
        if (!llvm::is_contained(members.producers, op))
          members.producers.push_back(op);
      } else if (roleAttr.getInt() == CVPipeline::crossCoreConsumerId) {
        if (!llvm::is_contained(members.consumers, op))
          members.consumers.push_back(op);
      }
    }
  });
}

// The computation blocks the control flow conditions are computed for: the
// blocks of the body of the main loop. The conditions are created for them
// when the if ops are created, one condition per block of the body. The
// block is the pair of the scope that holds the main loop and the id of the
// block.
void collectMainLoopBlocks(ModuleOp module, llvm::DenseSet<BlockNode> &blocks) {
  module.walk([&](Operation *op) {
    if (!CVPipeline::isMainLoopOp(op))
      return;
    Block *body = CVPipeline::MainLoop(op).getBody();
    if (!body)
      return;
    Operation *scopeOp = nullptr;
    if (auto scope = op->getParentOfType<scope::ScopeOp>())
      scopeOp = scope.getOperation();
    for (Operation &child : body->without_terminator()) {
      auto blockId = child.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
      if (blockId)
        blocks.insert(BlockNode{scopeOp, blockId.getInt()});
    }
  });
}

// The tightly coupled buffers a transfer carries. The producer of a group
// writes the buffers the consumer reads, so the buffers of the producer
// describe the transfer.
void collectTransferBuffers(Operation *producer,
                            llvm::SmallDenseSet<int> &buffers) {
  for (Value operand : producer->getOperands()) {
    Value alloc = CVPipeline::traceBackToMemrefAlloc(operand);
    if (!alloc)
      continue;
    if (auto id = CVPipeline::getTightlyCoupledBufferId(alloc))
      buffers.insert(*id);
  }
}

// Depth-first search for the paths of `chainLength` blocks starting at
// `node`. Every edge of the graph joins two different cores, so a path that
// starts on the cube core alternates CUBE and VECTOR.
//
// The path also has to carry the tightly coupled buffers of the validated
// pipeline: every transfer of the path is double buffered and no buffer is
// shared by two transfers of the path. `buffers` counts how many times each
// buffer is used by the path so far.
void collectChains(const llvm::SmallVector<llvm::SmallVector<Edge>> &succs,
                   unsigned node, unsigned depth, unsigned chainLength,
                   llvm::SmallVector<unsigned> &path,
                   llvm::SmallDenseMap<int, unsigned> &buffers,
                   llvm::SmallVector<llvm::SmallVector<unsigned>> &chains) {
  if (depth == chainLength) {
    if (buffers.size() == kBuffersPerTransfer * (chainLength - 1))
      chains.push_back(path);
    return;
  }
  for (const Edge &edge : succs[node]) {
    if (edge.buffers.size() != kBuffersPerTransfer)
      continue;
    if (llvm::is_contained(path, edge.consumer))
      continue;
    path.push_back(edge.consumer);
    for (int buffer : edge.buffers)
      ++buffers[buffer];
    collectChains(succs, edge.consumer, depth + 1, chainLength, path, buffers,
                  chains);
    // Drop the buffers the path does not carry any more, so that the size of
    // the map stays the number of distinct buffers of the path.
    for (int buffer : edge.buffers) {
      --buffers[buffer];
      if (buffers[buffer] == 0)
        buffers.erase(buffer);
    }
    path.pop_back();
  }
}

// The blocks of the chains that belong to one pipeline. Two chains that share
// a block are two views of the same pipeline, so they count once.
class ChainGroups {
  llvm::DenseMap<unsigned, unsigned> parent;

public:
  unsigned find(unsigned block) {
    auto it = parent.find(block);
    if (it == parent.end()) {
      parent[block] = block;
      return block;
    }
    if (it->second == block)
      return block;
    return it->second = find(it->second);
  }

  void unite(unsigned lhs, unsigned rhs) {
    unsigned lhsRoot = find(lhs);
    unsigned rhsRoot = find(rhs);
    if (lhsRoot != rhsRoot)
      parent[lhsRoot] = rhsRoot;
  }

  size_t size() {
    llvm::SmallDenseSet<unsigned> roots;
    for (auto &entry : parent)
      roots.insert(find(entry.first));
    return roots.size();
  }
};

} // namespace

triton::CrossCorePattern triton::analyzeCrossCorePattern(ModuleOp module,
                                                         unsigned chainLength) {
  CrossCorePattern pattern;
  // A chain needs at least one producer and one consumer.
  if (chainLength < 2)
    return pattern;

  // Number the blocks of the module, the nodes of the graph.
  llvm::DenseMap<BlockNode, unsigned> nodeIds;
  SmallVector<CoreKind> cores;
  SmallVector<BlockNode> blockNodes;
  llvm::DenseMap<Operation *, unsigned> opNodeIds;
  module.walk([&](Operation *op) {
    auto node = getBlockNode(op);
    if (!node)
      return;
    auto it = nodeIds.find(*node);
    if (it == nodeIds.end()) {
      it =
          nodeIds.try_emplace(*node, static_cast<unsigned>(cores.size())).first;
      cores.push_back(getBlockCore(op));
      blockNodes.push_back(*node);
    }
    opNodeIds[op] = it->second;
  });

  // The cross-core edges: the producer block of a group feeds the consumer
  // block of the same group. The edges that stay on one core are dropped,
  // they are not cross-core dependencies.
  llvm::DenseMap<int, GroupMembers> groups;
  collectCrossCoreGroups(module, groups);
  llvm::SmallVector<llvm::SmallVector<Edge>> succs(cores.size());
  // The computation blocks that take part in a cross core transfer: they are
  // the blocks the conditions of the scheme are computed for.
  llvm::SmallDenseSet<unsigned> crossCoreBlocks;
  for (auto &entry : groups) {
    GroupMembers &members = entry.second;
    // The tightly coupled buffers the transfer of this group carries.
    llvm::SmallDenseSet<int> transferBuffers;
    for (Operation *producer : members.producers)
      collectTransferBuffers(producer, transferBuffers);
    SmallVector<int> buffers(transferBuffers.begin(), transferBuffers.end());
    llvm::sort(buffers);
    for (Operation *producer : members.producers) {
      auto producerIt = opNodeIds.find(producer);
      if (producerIt == opNodeIds.end())
        continue;
      for (Operation *consumer : members.consumers) {
        auto consumerIt = opNodeIds.find(consumer);
        if (consumerIt == opNodeIds.end())
          continue;
        unsigned producerId = producerIt->second;
        unsigned consumerId = consumerIt->second;
        if (producerId == consumerId)
          continue;
        if (cores[producerId] == CoreKind::Unknown ||
            cores[consumerId] == CoreKind::Unknown)
          continue;
        if (cores[producerId] == cores[consumerId])
          continue;
        crossCoreBlocks.insert(producerId);
        crossCoreBlocks.insert(consumerId);
        // One direction of the graph may be carried by several cross-core
        // groups. The tightly coupled buffers of the direction are the total
        // of the groups that carry it, not the buffers of a single group:
        // every transfer of the validated pipeline carries the same number
        // of tightly coupled buffers, so the groups of one direction have to
        // add up to that number.
        Edge *edge = nullptr;
        for (Edge &candidate : succs[producerId]) {
          if (candidate.consumer == consumerId) {
            edge = &candidate;
            break;
          }
        }
        if (!edge) {
          succs[producerId].push_back(Edge{consumerId, buffers});
          continue;
        }
        for (int buffer : buffers) {
          if (!llvm::is_contained(edge->buffers, buffer))
            edge->buffers.push_back(buffer);
        }
        llvm::sort(edge->buffers);
      }
    }
  }

  // The conditions are computed for the blocks that carry a condition, so
  // the cross core blocks of the kernel that carry a condition are the ones
  // the schemes of the validated pipelines are built for: four of them for
  // the mla pipeline and five of them for the paged decode pipeline. The
  // number is reported for the diagnostics, it does not gate the scheme.
  llvm::DenseSet<BlockNode> conditionedBlocks;
  collectMainLoopBlocks(module, conditionedBlocks);
  [[maybe_unused]] unsigned conditionedCrossCoreBlocks = 0;
  for (unsigned id : crossCoreBlocks)
    if (conditionedBlocks.contains(blockNodes[id]))
      ++conditionedCrossCoreBlocks;
  LLVM_DEBUG({
    DBGS() << "The kernel holds " << conditionedCrossCoreBlocks
           << " cross core blocks that carry a condition\n";
  });

  // The chain starts on any block of a known core. Every edge of the graph
  // joins two different cores, so the chain alternates the cores: it is
  // CUBE -> VECTOR -> CUBE -> VECTOR when it starts on the cube core and
  // VECTOR -> CUBE -> VECTOR -> CUBE when it starts on the vector core.
  llvm::SmallVector<llvm::SmallVector<unsigned>> chains;
  for (unsigned id = 0; id < cores.size(); ++id) {
    if (cores[id] == CoreKind::Unknown)
      continue;
    llvm::SmallVector<unsigned> path{id};
    llvm::SmallDenseMap<int, unsigned> buffers;
    collectChains(succs, id, 1, chainLength, path, buffers, chains);
  }

  // The chains that share a block are two views of the same pipeline: they
  // count once. The chains that do not share a block are independent
  // pipelines, one of them each.
  ChainGroups pipelines;
  for (const auto &chain : chains) {
    for (unsigned i = 1; i < chain.size(); ++i)
      pipelines.unite(chain[i - 1], chain[i]);
  }
  pattern.pipelines = pipelines.size();

  // The entry blocks of the chains: the blocks of the chains that no other
  // block of the chains feeds. A pipeline that is a single linear chain has
  // one entry. Two blocks feeding the same block is a merge, and the merge
  // has two entries: the pipeline is not the linear chain the scheme is
  // validated on.
  llvm::SmallDenseSet<unsigned> blocks;
  for (const auto &chain : chains)
    blocks.insert(chain.begin(), chain.end());
  llvm::SmallDenseSet<unsigned> fed;
  for (unsigned node = 0; node < succs.size(); ++node) {
    if (!blocks.contains(node))
      continue;
    for (const Edge &edge : succs[node])
      if (blocks.contains(edge.consumer))
        fed.insert(edge.consumer);
  }
  pattern.entries = blocks.size() - fed.size();

  LLVM_DEBUG({
    DBGS() << "Found " << chains.size() << " chains of " << chainLength
           << " blocks in " << pattern.pipelines << " pipelines with "
           << pattern.entries << " entries\n";
  });
  return pattern;
}
