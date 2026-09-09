/* Copyright (c) Huawei Technologies Co., Ltd. 2026. */
#include "ascend/include/CVSplitScheduling/CrossCoreOwnershipProof.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <cstdint>

namespace mlir::triton::cv_split {
namespace {
struct GraphEdge { unsigned target; bool handoff; };
struct Reachability { bool reachable = false; bool handoff = false; };
struct ProofGraph {
  llvm::SmallVector<Operation *> operations;
  llvm::DenseMap<Operation *, unsigned> order;
  llvm::SmallVector<llvm::SmallVector<GraphEdge>> adjacency;
  llvm::SmallVector<int64_t> wrap;
};

static void addEdge(ProofGraph &g, unsigned from, unsigned to, bool handoff) {
  g.adjacency[from].push_back({to, handoff});
}

static FailureOr<ProofGraph> buildGraph(const CrossCorePipelinePlan &plan) {
  constexpr unsigned resources =
      static_cast<unsigned>(PrincipalResource::ScalarControl) + 1;
  if (plan.resourceUses.empty())
    return failure();
  ProofGraph g;
  const unsigned count = plan.resourceUses.size();
  g.operations.resize(count, nullptr);
  g.adjacency.resize(count);
  g.wrap.resize(count, -1);
  std::array<llvm::SmallVector<const PipelineResourceUse *>, resources> byRes;
  llvm::DenseMap<Operation *, const PipelineResourceUse *> useByOp;
  for (const PipelineResourceUse &use : plan.resourceUses) {
    if (!use.operation || use.order >= count || g.operations[use.order] ||
        !g.order.try_emplace(use.operation, use.order).second ||
        !useByOp.try_emplace(use.operation, &use).second)
      return failure();
    g.operations[use.order] = use.operation;
    byRes[static_cast<unsigned>(use.resource)].push_back(&use);
  }
  if (llvm::is_contained(g.operations, nullptr))
    return failure();
  for (auto &uses : byRes) {
    llvm::sort(uses, [](auto *a, auto *b) { return a->order < b->order; });
    for (unsigned i = 1; i < uses.size(); ++i)
      addEdge(g, uses[i - 1]->order, uses[i]->order, false);
    if (!uses.empty())
      g.wrap[uses.back()->order] = uses.front()->order;
  }
  for (const PipelineResourceUse &use : plan.resourceUses)
    for (Operation *user : use.operation->getUsers()) {
      auto it = useByOp.find(user);
      if (it != useByOp.end() && it->second->engine == use.engine)
        addEdge(g, use.order, it->second->order, false);
    }
  for (const CrossCoreBoundary &boundary : plan.boundaries) {
    auto producer = useByOp.find(boundary.producer);
    if (producer == useByOp.end() || boundary.consumers.empty())
      return failure();
    for (Operation *consumerOp : boundary.consumers) {
      auto consumer = useByOp.find(consumerOp);
      if (consumer == useByOp.end() ||
          consumer->second->engine == producer->second->engine)
        return failure();
      addEdge(g, producer->second->order, consumer->second->order, true);
    }
  }
  for (auto &edges : g.adjacency)
    llvm::sort(edges, [](const GraphEdge &a, const GraphEdge &b) {
      return a.target != b.target ? a.target < b.target : a.handoff < b.handoff;
    });
  return g;
}

static Reachability reachable(const ProofGraph &g, unsigned from, unsigned to,
                              unsigned targetIteration) {
  const unsigned n = g.operations.size();
  llvm::SmallVector<uint8_t> seen(n * 4, 0);
  llvm::SmallVector<unsigned> queue;
  auto id = [&](unsigned node, unsigned iter, bool handoff) {
    return (iter * 2 + static_cast<unsigned>(handoff)) * n + node;
  };
  auto push = [&](unsigned node, unsigned iter, bool handoff) {
    unsigned state = id(node, iter, handoff);
    if (!seen[state]) {
      seen[state] = 1;
      queue.push_back(state);
    }
  };
  push(from, 0, false);
  for (unsigned cursor = 0; cursor < queue.size(); ++cursor) {
    unsigned state = queue[cursor], node = state % n, tag = state / n;
    bool handoff = (tag % 2) != 0;
    unsigned iter = tag / 2;
    for (const GraphEdge &edge : g.adjacency[node])
      push(edge.target, iter, handoff || edge.handoff);
    if (iter == 0 && g.wrap[node] >= 0)
      push(static_cast<unsigned>(g.wrap[node]), 1, handoff);
  }
  if (seen[id(to, targetIteration, false)])
    return {true, false};
  if (seen[id(to, targetIteration, true)])
    return {true, true};
  return {};
}

static void count(CrossCoreResourcePlan &plan,
                  ResourceOwnershipOrdering ordering) {
  if (ordering == ResourceOwnershipOrdering::SameEngineOrder)
    ++plan.sameEngineOwnershipEdges;
  else if (ordering == ResourceOwnershipOrdering::ExistingCrossCorePath)
    ++plan.crossCoreOwnershipEdges;
  else if (ordering == ResourceOwnershipOrdering::ExplicitReleaseRequired)
    ++plan.explicitReleaseOwnershipEdges;
  else
    ++plan.unresolvedOwnershipEdges;
}
} // namespace

LogicalResult proveCrossCoreResourceOwnership(
    const CrossCorePipelinePlan &materializedPlan,
    llvm::ArrayRef<int64_t> delayedReleaseGroups,
    CrossCoreResourcePlan &resourcePlan) {
  if (!resourcePlan.anchorsComplete)
    return failure();
  FailureOr<ProofGraph> graphResult = buildGraph(materializedPlan);
  if (failed(graphResult))
    return failure();
  const ProofGraph &g = *graphResult;
  llvm::DenseSet<int64_t> delayed;
  for (int64_t group : delayedReleaseGroups)
    delayed.insert(group);
  resourcePlan.sameEngineOwnershipEdges = 0;
  resourcePlan.crossCoreOwnershipEdges = 0;
  resourcePlan.explicitReleaseOwnershipEdges = 0;
  resourcePlan.unresolvedOwnershipEdges = 0;
  resourcePlan.loopCarriedOwnershipEdges = 0;
  resourcePlan.seedRequirements = 0;
  for (ResourceOwnershipEdge &edge : resourcePlan.ownershipEdges) {
    auto reader = g.order.find(edge.lastReader);
    auto writer = g.order.find(edge.nextWriter);
    if (reader == g.order.end() || writer == g.order.end())
      return failure();
    Reachability path = reachable(g, reader->second, writer->second,
                                  edge.loopCarried ? 1u : 0u);
    edge.needsSeed = false;
    edge.usesCrossCorePath = path.handoff;
    if (path.reachable)
      edge.ordering = path.handoff
                          ? ResourceOwnershipOrdering::ExistingCrossCorePath
                          : ResourceOwnershipOrdering::SameEngineOrder;
    else if (edge.loopCarried && delayed.contains(edge.physicalGroup))
      edge.ordering = ResourceOwnershipOrdering::ExplicitReleaseRequired;
    else
      edge.ordering = ResourceOwnershipOrdering::Unresolved;
    if (edge.loopCarried)
      ++resourcePlan.loopCarriedOwnershipEdges;
    if (edge.needsSeed)
      ++resourcePlan.seedRequirements;
    count(resourcePlan, edge.ordering);
  }
  resourcePlan.ownershipResolved = resourcePlan.unresolvedOwnershipEdges == 0;
  return success();
}
} // namespace mlir::triton::cv_split
