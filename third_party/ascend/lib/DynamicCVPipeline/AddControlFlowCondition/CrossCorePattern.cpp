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

// The ops that produce and that consume a cross-core dependency group.
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

// A chain of computation blocks and the tightly coupled buffers it carries.
struct Chain {
  SmallVector<unsigned> nodes;
  SmallVector<int> buffers;
};

// Every cross-core transfer of the validated pipeline is double buffered: the
// producer writes two tightly coupled buffers and the consumer reads them in
// turn. A chain of `chainLength` blocks carries 2 * (chainLength - 1) of them.
constexpr unsigned kBuffersPerTransfer = 2;

// The chain of the hstu attention pipeline holds three blocks and two
// transfers.
constexpr unsigned kShortChainLength = 3;

// The tree of the nsa compression backward pipeline holds six double buffered
// transfers, so it carries twelve tightly coupled buffers.
constexpr unsigned kTreeTransfers = 6;

// A tree of n blocks holds n - 1 directions and every direction holds at least
// one transfer, so a tree of six transfers holds at most seven blocks.
constexpr unsigned kTreeMinBlocks = 6;
constexpr unsigned kTreeMaxBlocks = 7;

// The forest of the flex attention backward pipeline holds four chains of
// three blocks and single buffered transfers, so it holds eight directions
// and eight tightly coupled buffers.
constexpr unsigned kForestBuffersPerTransfer = 1;
constexpr unsigned kForestChains = 4;
constexpr unsigned kForestBlocks = kForestChains * kShortChainLength;

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
// blocks of the body of the main loop, one condition per block of the body.
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

// The tightly coupled buffers a transfer carries: the producer of a group
// writes the buffers the consumer reads.
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
// pipeline: every transfer of the path carries `buffersPerTransfer` buffers
// and no buffer is shared by two transfers of the path. `buffers` counts how
// many times each buffer is used by the path so far.
void collectChains(const llvm::SmallVector<llvm::SmallVector<Edge>> &succs,
                   unsigned node, unsigned depth, unsigned chainLength,
                   unsigned buffersPerTransfer,
                   llvm::SmallVector<unsigned> &path,
                   llvm::SmallDenseMap<int, unsigned> &buffers,
                   llvm::SmallVector<Chain> &chains,
                   const llvm::SmallDenseSet<unsigned> *allowed = nullptr) {
  if (depth == chainLength) {
    if (buffers.size() == buffersPerTransfer * (chainLength - 1)) {
      Chain chain;
      chain.nodes = path;
      for (const auto &entry : buffers)
        chain.buffers.push_back(entry.first);
      llvm::sort(chain.buffers);
      chains.push_back(std::move(chain));
    }
    return;
  }
  for (const Edge &edge : succs[node]) {
    if (edge.buffers.size() != buffersPerTransfer)
      continue;
    if (allowed && !allowed->contains(edge.consumer))
      continue;
    if (llvm::is_contained(path, edge.consumer))
      continue;
    path.push_back(edge.consumer);
    for (int buffer : edge.buffers)
      ++buffers[buffer];
    collectChains(succs, edge.consumer, depth + 1, chainLength,
                  buffersPerTransfer, path, buffers, chains, allowed);
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

// The chains of the graph that stay inside `blocks`.
llvm::SmallVector<Chain>
collectChainsOf(const llvm::SmallVector<llvm::SmallVector<Edge>> &succs,
                const llvm::SmallDenseSet<unsigned> &blocks,
                unsigned chainLength,
                unsigned buffersPerTransfer = kBuffersPerTransfer) {
  llvm::SmallVector<Chain> chains;
  for (unsigned id = 0; id < succs.size(); ++id) {
    if (!blocks.contains(id))
      continue;
    llvm::SmallVector<unsigned> path{id};
    llvm::SmallDenseMap<int, unsigned> buffers;
    collectChains(succs, id, 1, chainLength, buffersPerTransfer, path, buffers,
                  chains, &blocks);
  }
  return chains;
}

// Whether the chains are the two independent pipelines of the blocks: they
// hold every block of `blocks` once, they share no buffer and no block of a
// chain is fed by a block of another chain.
bool areIndependentChains(
    const llvm::SmallVector<Chain> &chains,
    const llvm::SmallDenseSet<unsigned> &blocks,
    const llvm::SmallVector<llvm::SmallVector<Edge>> &succs,
    unsigned chainLength) {
  if (chains.size() != 2)
    return false;
  llvm::SmallDenseSet<unsigned> covered;
  llvm::SmallDenseSet<int> buffers;
  for (const Chain &chain : chains) {
    for (unsigned node : chain.nodes) {
      // A block that is held by two chains belongs to a single pipeline.
      if (!covered.insert(node).second)
        return false;
    }
    for (int buffer : chain.buffers) {
      // Two pipelines that share a buffer are not independent.
      if (!buffers.insert(buffer).second)
        return false;
    }
  }
  // Every block that carries a condition has to belong to a chain.
  if (covered.size() != blocks.size())
    return false;
  // An edge that joins two chains makes them a single pipeline with a merge.
  unsigned edges = 0;
  for (unsigned node : blocks) {
    for (const Edge &edge : succs[node]) {
      if (!blocks.contains(edge.consumer))
        continue;
      ++edges;
    }
  }
  return edges == chains.size() * (chainLength - 1);
}

// Whether the chains are the chain of three computation blocks of the hstu
// pipeline: a single linear chain that holds every block of `blocks`.
bool isShortChain(const llvm::SmallVector<Chain> &chains,
                  const llvm::SmallDenseSet<unsigned> &blocks,
                  const llvm::SmallVector<llvm::SmallVector<Edge>> &succs,
                  unsigned chainLength) {
  // One chain: a second chain of the blocks is a second pipeline.
  if (chains.size() != 1)
    return false;
  // Every block that carries a condition belongs to the chain.
  if (chains[0].nodes.size() != blocks.size())
    return false;
  // The chain carries the buffers of two double buffered transfers.
  if (chains[0].buffers.size() != kBuffersPerTransfer * (chainLength - 1))
    return false;
  // The chain is a linear chain: the head is fed by no other block and every
  // other block is fed by the block before it and by no other one.
  for (unsigned i = 0; i < chains[0].nodes.size(); ++i) {
    unsigned predecessors = 0;
    for (const auto &edges : succs)
      for (const Edge &edge : edges)
        if (edge.consumer == chains[0].nodes[i])
          ++predecessors;
    if (predecessors != (i == 0 ? 0 : 1))
      return false;
  }
  return true;
}

// Whether the blocks that carry a condition form the forest of the flex
// attention backward pipeline: four chains of three blocks, single buffered.
bool isShortChainForest(
    const llvm::SmallVector<Chain> &chains,
    const llvm::SmallDenseSet<unsigned> &blocks,
    const llvm::SmallVector<llvm::SmallVector<Edge>> &succs) {
  // The forest holds kForestChains chains and no more.
  if (chains.size() != kForestChains) {
    LLVM_DEBUG(DBGS() << "The forest holds " << chains.size()
                      << " chains and not " << kForestChains << "\n");
    return false;
  }
  llvm::SmallDenseSet<unsigned> covered;
  llvm::SmallDenseSet<int> buffers;
  for (const Chain &chain : chains) {
    // Every chain holds three computation blocks.
    if (chain.nodes.size() != kShortChainLength) {
      LLVM_DEBUG(DBGS() << "A chain of the forest holds " << chain.nodes.size()
                        << " blocks and not " << kShortChainLength << "\n");
      return false;
    }
    // A block that is held by two chains belongs to a single pipeline.
    for (unsigned node : chain.nodes)
      if (!covered.insert(node).second) {
        LLVM_DEBUG(DBGS() << "A block of the forest belongs to two chains\n");
        return false;
      }
    // Two chains that share a buffer are not independent.
    for (int buffer : chain.buffers)
      if (!buffers.insert(buffer).second) {
        LLVM_DEBUG(DBGS() << "The buffer " << buffer
                          << " is shared by two chains of the forest\n");
        return false;
      }
  }
  // Every block that carries a condition has to belong to a chain.
  if (covered.size() != blocks.size()) {
    LLVM_DEBUG(DBGS() << "The forest holds " << covered.size()
                      << " blocks and not " << blocks.size() << "\n");
    return false;
  }
  // An edge that joins two chains makes them a single pipeline.
  unsigned edges = 0;
  for (unsigned node : blocks) {
    for (const Edge &edge : succs[node]) {
      if (!blocks.contains(edge.consumer))
        continue;
      ++edges;
      // Every direction carries exactly one buffer.
      if (edge.buffers.size() != kForestBuffersPerTransfer) {
        LLVM_DEBUG(DBGS() << "A direction of the forest carries "
                          << edge.buffers.size() << " buffers and not "
                          << kForestBuffersPerTransfer << "\n");
        return false;
      }
    }
  }
  if (edges != kForestChains * (kShortChainLength - 1)) {
    LLVM_DEBUG(DBGS() << "The forest holds " << edges << " directions and not "
                      << kForestChains * (kShortChainLength - 1) << "\n");
    return false;
  }
  // Every chain is a linear chain: the head is fed by no other block and
  // every other block is fed by the block before it and by no other one.
  for (const Chain &chain : chains) {
    for (unsigned i = 0; i < chain.nodes.size(); ++i) {
      unsigned predecessors = 0;
      for (const auto &edgesOfNode : succs)
        for (const Edge &edge : edgesOfNode)
          if (edge.consumer == chain.nodes[i])
            ++predecessors;
      if (predecessors != (i == 0 ? 0 : 1)) {
        LLVM_DEBUG(DBGS() << "The block " << chain.nodes[i]
                          << " of the forest is fed by " << predecessors
                          << " blocks\n");
        return false;
      }
    }
  }
  return true;
}

// Whether the blocks that carry a condition form the tree of the nsa
// compression backward pipeline: a connected graph without a cycle that may
// fork and merge, whose directions carry whole double buffered transfers and
// share no buffer.
bool isTreePipeline(const llvm::SmallVector<llvm::SmallVector<Edge>> &succs,
                    const llvm::SmallDenseSet<unsigned> &blocks) {
  // No direction of the tree may leave it: it would compare the counters of
  // a block that carries no condition.
  for (unsigned node = 0; node < succs.size(); ++node) {
    bool inside = blocks.contains(node);
    for (const Edge &edge : succs[node]) {
      if (blocks.contains(edge.consumer) != inside) {
        LLVM_DEBUG(DBGS() << "The tree is not closed: the direction of the "
                             "block "
                          << node << " leaves it\n");
        return false;
      }
    }
  }

  unsigned directions = 0;
  unsigned transfers = 0;
  llvm::SmallDenseSet<int> buffers;
  for (unsigned node : blocks) {
    for (const Edge &edge : succs[node]) {
      // Every direction carries a whole number of double buffered transfers.
      if (edge.buffers.empty() ||
          edge.buffers.size() % kBuffersPerTransfer != 0) {
        LLVM_DEBUG(DBGS() << "The direction of the block " << node
                          << " carries " << edge.buffers.size()
                          << " buffers\n");
        return false;
      }
      transfers += edge.buffers.size() / kBuffersPerTransfer;
      for (int buffer : edge.buffers)
        if (!buffers.insert(buffer).second) {
          LLVM_DEBUG(DBGS() << "The buffer " << buffer
                            << " is shared by two directions of the tree\n");
          return false;
        }
      ++directions;
    }
  }

  // The tree carries the buffers of the transfers of the pipeline.
  if (transfers != kTreeTransfers) {
    LLVM_DEBUG(DBGS() << "The tree carries " << transfers
                      << " transfers and not " << kTreeTransfers << "\n");
    return false;
  }

  // The blocks form a tree: one direction less than they have blocks.
  if (directions + 1 != blocks.size()) {
    LLVM_DEBUG(DBGS() << "The tree holds " << directions << " directions and "
                      << blocks.size() << " blocks\n");
    return false;
  }

  // The blocks are connected: the tree is an undirected graph.
  llvm::SmallVector<llvm::SmallVector<unsigned>> neighbours(succs.size());
  for (unsigned node : blocks) {
    for (const Edge &edge : succs[node]) {
      neighbours[node].push_back(edge.consumer);
      neighbours[edge.consumer].push_back(node);
    }
  }
  llvm::SmallVector<unsigned> stack;
  llvm::SmallDenseSet<unsigned> visited;
  unsigned components = 0;
  for (unsigned node : blocks) {
    if (visited.contains(node))
      continue;
    ++components;
    visited.insert(node);
    stack.push_back(node);
    while (!stack.empty()) {
      unsigned current = stack.pop_back_val();
      for (unsigned neighbour : neighbours[current])
        if (visited.insert(neighbour).second)
          stack.push_back(neighbour);
    }
  }
  if (components != 1) {
    LLVM_DEBUG(DBGS() << "The tree holds " << components
                      << " components and not one\n");
    return false;
  }
  return true;
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
  // block of the same group. The edges that stay on one core are dropped.
  llvm::DenseMap<int, GroupMembers> groups;
  collectCrossCoreGroups(module, groups);
  llvm::SmallVector<llvm::SmallVector<Edge>> succs(cores.size());
  // The computation blocks that take part in a cross core transfer.
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
        // One direction may be carried by several groups: the buffers of the
        // direction are the total of the groups that carry it.
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

  // The conditions are computed for the blocks that carry a condition.
  llvm::DenseSet<BlockNode> conditionedBlocks;
  collectMainLoopBlocks(module, conditionedBlocks);
  llvm::SmallDenseSet<unsigned> conditionedCrossCore;
  for (unsigned id : crossCoreBlocks)
    if (conditionedBlocks.contains(blockNodes[id]))
      conditionedCrossCore.insert(id);
  unsigned conditionedCrossCoreBlocks = conditionedCrossCore.size();
  LLVM_DEBUG({
    DBGS() << "The kernel holds " << conditionedCrossCoreBlocks
           << " cross core blocks that carry a condition\n";
  });

  // The chains start on any block of a known core; every edge joins two
  // different cores, so a chain alternates the cores.
  llvm::SmallVector<Chain> chains;
  for (unsigned id = 0; id < cores.size(); ++id) {
    if (cores[id] == CoreKind::Unknown)
      continue;
    llvm::SmallVector<unsigned> path{id};
    llvm::SmallDenseMap<int, unsigned> buffers;
    collectChains(succs, id, 1, chainLength, kBuffersPerTransfer, path, buffers,
                  chains);
  }

  // The chains that share a block are two views of the same pipeline.
  ChainGroups pipelines;
  for (const auto &chain : chains) {
    for (unsigned i = 1; i < chain.nodes.size(); ++i)
      pipelines.unite(chain.nodes[i - 1], chain.nodes[i]);
  }
  pattern.pipelines = pipelines.size();

  // The entry blocks of the chains: the blocks that no other block of the
  // chains feeds. A merge, two blocks feeding the same block, has two
  // entries.
  llvm::SmallDenseSet<unsigned> blocks;
  for (const auto &chain : chains)
    blocks.insert(chain.nodes.begin(), chain.nodes.end());
  llvm::SmallDenseSet<unsigned> fed;
  for (unsigned node = 0; node < succs.size(); ++node) {
    if (!blocks.contains(node))
      continue;
    for (const Edge &edge : succs[node])
      if (blocks.contains(edge.consumer))
        fed.insert(edge.consumer);
  }
  pattern.entries = blocks.size() - fed.size();

  // The chains of the forest: the transfers of that pipeline are single
  // buffered, so the chains of four blocks above are not its chains.
  llvm::SmallVector<Chain> forestChains;
  if (conditionedCrossCoreBlocks == kForestBlocks)
    forestChains =
        collectChainsOf(succs, conditionedCrossCore, kShortChainLength,
                        kForestBuffersPerTransfer);

  LLVM_DEBUG({
    if (!forestChains.empty())
      DBGS() << "Found " << forestChains.size() << " chains of "
             << kShortChainLength << " blocks in " << forestChains.size()
             << " pipelines\n";
    else
      DBGS() << "Found " << chains.size() << " chains of " << chainLength
             << " blocks in " << pattern.pipelines << " pipelines with "
             << pattern.entries << " entries\n";
  });

  // The shape of the graph.
  if (conditionedCrossCoreBlocks == chainLength ||
      conditionedCrossCoreBlocks == chainLength + 1) {
    // A single pipeline: the mla pipeline holds four computation blocks and
    // the paged decode pipeline five.
    if (pattern.pipelines == 1 && pattern.entries == 1)
      pattern.shape = CrossCoreShape::LinearChain;
  } else if (conditionedCrossCoreBlocks == 2 * chainLength) {
    // Two pipelines that run at the same time: the chains have to be
    // independent.
    llvm::SmallVector<Chain> conditionedChains =
        collectChainsOf(succs, conditionedCrossCore, chainLength);
    if (areIndependentChains(conditionedChains, conditionedCrossCore, succs,
                             chainLength))
      pattern.shape = CrossCoreShape::TwoChains;
  } else if (conditionedCrossCoreBlocks == kShortChainLength) {
    // The hstu attention pipeline: a chain of three blocks, and one more
    // block that carries a condition and takes no cross-core transfer.
    if (conditionedBlocks.size() == kShortChainLength + 1) {
      llvm::SmallVector<Chain> conditionedChains =
          collectChainsOf(succs, conditionedCrossCore, kShortChainLength);
      if (isShortChain(conditionedChains, conditionedCrossCore, succs,
                       kShortChainLength))
        pattern.shape = CrossCoreShape::ShortChain;
    }
  } else if (conditionedCrossCoreBlocks >= kTreeMinBlocks &&
             conditionedCrossCoreBlocks <= kTreeMaxBlocks) {
    // The nsa compression backward pipeline: a tree of computation blocks.
    if (conditionedBlocks.size() == conditionedCrossCoreBlocks &&
        isTreePipeline(succs, conditionedCrossCore))
      pattern.shape = CrossCoreShape::Tree;
    else
      LLVM_DEBUG(DBGS() << "The module holds " << conditionedBlocks.size()
                        << " blocks that carry a condition and "
                        << conditionedCrossCoreBlocks
                        << " of them take part in a cross core transfer\n");
  } else if (conditionedCrossCoreBlocks == kForestBlocks) {
    // The flex attention backward pipeline: a forest of four chains of three
    // blocks.
    if (conditionedBlocks.size() == conditionedCrossCoreBlocks) {
      if (isShortChainForest(forestChains, conditionedCrossCore, succs)) {
        pattern.shape = CrossCoreShape::ShortChainForest;
        // Every chain of the forest is a pipeline of its own with a single
        // entry: the head of the chain.
        pattern.pipelines = kForestChains;
        pattern.entries = kForestChains;
      } else {
        LLVM_DEBUG(DBGS() << "The module holds " << conditionedBlocks.size()
                          << " blocks that carry a condition and "
                          << conditionedCrossCoreBlocks
                          << " of them take part in a cross core "
                             "transfer\n");
      }
    } else {
      LLVM_DEBUG(DBGS() << "The module holds " << conditionedBlocks.size()
                        << " blocks that carry a condition and "
                        << conditionedCrossCoreBlocks
                        << " of them take part in a cross core transfer\n");
    }
  }
  LLVM_DEBUG({
    DBGS() << "The shape of the graph is "
           << (pattern.shape == CrossCoreShape::LinearChain  ? "linear chain"
               : pattern.shape == CrossCoreShape::TwoChains  ? "two chains"
               : pattern.shape == CrossCoreShape::ShortChain ? "short chain"
               : pattern.shape == CrossCoreShape::Tree       ? "tree"
               : pattern.shape == CrossCoreShape::ShortChainForest
                   ? "short chain forest"
                   : "unknown")
           << "\n";
  });
  return pattern;
}
