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

#include "ascend/include/DynamicCVPipeline/Common/ControlFlowGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace mlir::CVPipeline;

namespace {

using NodeId = ControlFlowGraph::NodeId;

SmallVector<SmallVector<NodeId>>
computeSCCs(ArrayRef<SmallVector<NodeId>> successors, ArrayRef<bool> eligible) {
  SmallVector<int> index(successors.size(), -1), low(successors.size());
  SmallVector<bool> active(successors.size(), false);
  SmallVector<NodeId> stack;
  SmallVector<SmallVector<NodeId>> components;
  int next = 0;
  std::function<void(NodeId)> visit = [&](NodeId v) {
    index[v] = low[v] = next++;
    stack.push_back(v);
    active[v] = true;
    for (NodeId w : successors[v]) {
      if (index[w] == -1) {
        visit(w);
        low[v] = std::min(low[v], low[w]);
      } else if (active[w]) {
        low[v] = std::min(low[v], index[w]);
      }
    }
    if (low[v] != index[v]) {
      return;
    }
    SmallVector<NodeId> component;
    NodeId w;
    do {
      w = stack.pop_back_val();
      active[w] = false;
      component.push_back(w);
    } while (w != v);
    // Canonicalize SCC members by NodeId to stabilize downstream iteration and
    // make component.front() the minimum NodeId for the outer component sort.
    llvm::sort(component);
    components.push_back(std::move(component));
  };
  for (NodeId v = 0; v < successors.size(); ++v) {
    if (eligible[v] && index[v] == -1) {
      visit(v);
    }
  }
  return components;
}

} // namespace

void ControlFlowGraph::createNode(BlockArgument value, Kind kind,
                                  OpResult exitResult) {
  if (lookup(value)) {
    return;
  }
  Operation *op = value.getOwner()->getParentOp();
  NodeId id = nodes.size();
  nodes.push_back({kind, op, value, exitResult});
  valueToNodeId[value] = id;
  if (exitResult) {
    valueToNodeId[exitResult] = id;
  }
}

void ControlFlowGraph::createNode(OpResult value, Kind kind) {
  if (lookup(value)) {
    return;
  }
  Operation *op = value.getOwner();
  NodeId id = nodes.size();
  nodes.push_back({kind, op, value, value});
  valueToNodeId[value] = id;
  if (kind == Kind::Operation && value.getResultNumber() == 0) {
    OperationToNodeId[op] = id;
  }
}

void ControlFlowGraph::addEdge(NodeId from, NodeId to, EdgeKind kind,
                               OpOperand *use) {
  Edge edge{from, to, kind, use};
  linkOut[from].push_back(edge);
  linkIn[to].push_back(edge);
}

void ControlFlowGraph::createFor(scf::ForOp forOp) {
  // The induction variable is a loop-local phi with no exit result.
  // Each iter-arg is a phi that exits as the matching ForOp result; alias the
  // result to the iter-arg node so external consumers resolve here.
  for (BlockArgument arg : forOp.getBody()->getArguments()) {
    createNode(arg, Kind::Phi, forOp.getTiedLoopResult(arg));
  }
}

void ControlFlowGraph::createIf(scf::IfOp ifOp) {
  // IfOp results are branch-selected phis; the result is itself the exit
  // value, so no separate alias is needed (exitResult == value).
  for (OpResult result : ifOp->getOpResults()) {
    createNode(result, Kind::Phi);
  }
}

void ControlFlowGraph::createWhile(scf::WhileOp whileOp) {
  // Before-args: loop-carried phis, updated by the after-region's yield. They
  // do not exit the loop (while results come from the condition's forwarded
  // values, not the before-args), so no exit result.
  for (BlockArgument arg : whileOp.getBeforeArguments()) {
    createNode(arg, Kind::Phi);
  }
  // After-args: the values forwarded by scf.condition (operand i+1). Each
  // after-arg equals the while's result i, so it owns that exit result.
  for (auto [i, arg] : llvm::enumerate(whileOp.getAfterArguments())) {
    createNode(arg, Kind::Phi, whileOp->getResult(i));
  }
}

// Create the graph nodes for `op` (no edges) before connectEdges. FuncOp args
// and SCF region args become Phi nodes; ordinary ops get one node per result
// (result 0 doubles as the op representative), or a single valueless node.
void ControlFlowGraph::createNodes(Operation *op) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    for (BlockArgument arg : funcOp.getArguments()) {
      createNode(arg, Kind::Phi);
    }
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    createFor(forOp);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    createIf(ifOp);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    createWhile(whileOp);
  } else if (!isa<scf::YieldOp, scf::ConditionOp>(op)) {
    if (op->getNumResults() == 0) {
      NodeId id = nodes.size();
      nodes.push_back({Kind::Operation, op, {}, {}});
      OperationToNodeId[op] = id;
    } else {
      for (OpResult result : op->getOpResults()) {
        createNode(result, Kind::Operation);
      }
    }
  }
}

// scf.yield forwards its operands to the parent construct's phi: a ForOp
// iter-arg, an IfOp result, or a WhileOp before-arg. Add a Yield edge from
// each yielded value to its target phi node.
void ControlFlowGraph::connectYieldOp(scf::YieldOp yieldOp) {
  Operation *parent = yieldOp->getParentOp();
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    unsigned i = operand.getOperandNumber();
    Value target;
    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      target = loop.getRegionIterArg(i);
    } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
      target = branch.getResult(i);
    } else if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
      target = loop.getBeforeArguments()[i];
    } else {
      continue;
    }
    auto to = lookup(target);
    auto from = lookup(operand.get());
    if (from && to) {
      addEdge(*from, *to, EdgeKind::Yield, &operand);
    }
  }
}

void ControlFlowGraph::connectConditionOp(scf::ConditionOp condOp) {
  scf::WhileOp whileOp = condOp->getParentOfType<scf::WhileOp>();
  // Operand 0 is the loop predicate; operands 1..n forward to the after-args
  // (and to the while results on exit).
  for (unsigned i = 1; i < condOp->getNumOperands(); ++i) {
    OpOperand &operand = condOp->getOpOperand(i);
    auto from = lookup(operand.get());
    auto to = lookup(whileOp.getAfterArguments()[i - 1]);
    if (from && to) {
      addEdge(*from, *to, EdgeKind::Yield, &operand);
    }
  }
}

void ControlFlowGraph::connectEdges(
    Operation *op, llvm::function_ref<bool(OpOperand &)> includeDataOperand) {
  if (isa<scf::IfOp, scf::WhileOp, scf::ForOp>(op)) {
    // Seed the loop phis with their init operands. Per-iteration updates
    // arrive later via scf.yield / scf.condition (handled below).
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      for (BlockArgument arg : loop.getRegionIterArgs()) {
        OpOperand *init = loop.getTiedLoopInit(arg);
        if (auto from = lookup(init->get())) {
          addEdge(*from, *lookup(arg), EdgeKind::Data, init);
        }
      }
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      for (auto [i, arg] : llvm::enumerate(loop.getBeforeArguments())) {
        if (auto from = lookup(loop.getInits()[i])) {
          addEdge(*from, *lookup(arg), EdgeKind::Data, &op->getOpOperand(i));
        }
      }
    }
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    connectYieldOp(yieldOp);
  } else if (auto condOp = dyn_cast<scf::ConditionOp>(op)) {
    connectConditionOp(condOp);
  } else if (auto to = lookup(op)) {
    // Ordinary SSA operands feed result 0 (or the valueless operation node).
    for (OpOperand &operand : op->getOpOperands()) {
      if (!includeDataOperand(operand)) {
        continue;
      }
      if (auto from = lookup(operand.get())) {
        addEdge(*from, *to, EdgeKind::Data, &operand);
      }
    }
  }
}

ControlFlowGraph::ControlFlowGraph(Operation *root)
    : ControlFlowGraph(root, [](OpOperand &) { return true; }) {}

ControlFlowGraph::ControlFlowGraph(
    Operation *root, llvm::function_ref<bool(OpOperand &)> includeDataOperand) {
  // Create all nodes before connecting edges, so Phi targets and forwarded
  // values are available regardless of their order in the walk.
  root->walk([&](Operation *op) { createNodes(op); });
  // A synthetic sink preserves boundaries even when a result has no users.
  NodeId exit = nodes.size();
  nodes.push_back({Kind::Exit, nullptr, {}, {}});
  root->walk([&](Operation *op) { connectEdges(op, includeDataOperand); });
  // Preserve unused SSA results as component outputs, including loop results
  // whose nodes also represent iter-args. Check the actual result uses rather
  // than graph out-edges: an aliased phi may still have internal loop edges.
  // These synthetic edges have no OpOperand and do not affect SCC membership.
  for (NodeId id = 0; id < exit; ++id) {
    OpResult result = nodes[id].result;
    if (result && result.use_empty()) {
      addEdge(id, exit, EdgeKind::DeadResultExit);
    }
  }
}

// Tarjan's SCC over the eligible subgraph: Operation nodes for which `isLegal`
// holds, plus every Phi node. Edges between eligible nodes become successor
// arcs. Components with no Operation node are dropped (pure-phi cycles carry
// no computation to extract).
SmallVector<SmallVector<ControlFlowGraph::NodeId>>
ControlFlowGraph::getNodeSCCs(
    llvm::function_ref<bool(Operation *)> isLegal) const {
  // Phi nodes relay accumulator values; only legal computations may join them.
  // Excluded computations must not connect otherwise separate components.
  SmallVector<bool> eligible(nodes.size(), false);
  for (NodeId id = 0; id < nodes.size(); ++id) {
    const Node &node = nodes[id];
    switch (node.kind) {
    case Kind::Operation:
      eligible[id] = isLegal(node.op);
      break;
    case Kind::Phi:
      eligible[id] = true;
      break;
    case Kind::Exit:
      break;
    }
  }
  // Keep only edges whose endpoints both belong to the eligible subgraph.
  SmallVector<SmallVector<NodeId>> successors(nodes.size());

  for (auto id = 0; id < nodes.size(); id++) {
    for (auto e : linkOut.lookup(id)) {
      if (eligible[id] && eligible[e.to]) {
        successors[e.from].push_back(e.to);
      }
    }
  }
  auto components = computeSCCs(successors, eligible);
  // Tarjan emits SCCs in DFS completion order; sort by each component's
  // smallest NodeId to keep downstream component and unit ordering
  // deterministic.
  llvm::sort(components, [](const auto &lhs, const auto &rhs) {
    return lhs.front() < rhs.front();
  });
  llvm::erase_if(components, [&](const auto &component) {
    return llvm::none_of(component, [&](NodeId id) {
      return nodes[id].kind == Kind::Operation;
    });
  });
  return components;
}

SmallVector<ControlFlowGraph::ConnectedComponent, 0>
ControlFlowGraph::toComponents(ArrayRef<SmallVector<NodeId>> components) const {
  SmallVector<ConnectedComponent, 0> result;
  for (const auto &ids : components) {
    ConnectedComponent component(this, ids);
    if (!component.ops.empty()) {
      result.push_back(std::move(component));
    }
  }
  return result;
}

SmallVector<ControlFlowGraph::ConnectedComponent, 0>
ControlFlowGraph::getStronglyConnectedComponents(
    llvm::function_ref<bool(Operation *)> isLegal) const {
  return toComponents(getNodeSCCs(isLegal));
}

void ControlFlowGraph::ConnectedComponent::collectOutputResults() {
  llvm::DenseSet<Value> seen;
  for (NodeId id : nodeIds) {
    auto edges = g->linkOut.find(id);
    if (edges != g->linkOut.end()) {
      for (const Edge &edge : edges->second) {
        OpResult result = g->nodes[id].result;
        if (result && (edge.use || edge.kind == EdgeKind::DeadResultExit) &&
            !nodeIds.contains(edge.to) && seen.insert(result).second) {
          // Use the SSA value at the boundary, not the node's representative.
          // Since the node representative arg/result at the sametime.
          outputResults.push_back(result);
        }
      }
    }
  }
}

SmallVector<OpOperand *>
ControlFlowGraph::ConnectedComponent::getExternalUses() const {
  llvm::SmallPtrSet<OpOperand *, 16> internalUses;
  for (NodeId id : nodeIds) {
    auto edges = g->linkOut.find(id);
    if (edges == g->linkOut.end()) {
      continue;
    }
    for (const Edge &edge : edges->second) {
      if (edge.use && nodeIds.contains(edge.to)) {
        internalUses.insert(edge.use);
      }
    }
  }

  SmallVector<OpOperand *> externalUses;
  llvm::SmallPtrSet<OpOperand *, 16> seen;
  for (NodeId id : nodeIds) {
    const auto &node = g->nodes[id];
    for (Value value : {node.value, Value(node.result)}) {
      if (!value) {
        continue;
      }
      for (OpOperand &use : value.getUses()) {
        if (!internalUses.contains(&use) && seen.insert(&use).second) {
          externalUses.push_back(&use);
        }
      }
    }
  }
  return externalUses;
}

void ControlFlowGraph::ConnectedComponent::collectEntryOps() {
  DenseSet<Operation *> seen;
  for (NodeId nodeid : entries) {
    Operation *op = g->nodes[nodeid].op;
    if (op && seen.insert(op).second) {
      entryOps.push_back(op);
    }
  }
}

// Collect the distinct operand sites at which a value from outside the
// component flows into one of its entry nodes.
void ControlFlowGraph::ConnectedComponent::collectInputOperands() {
  llvm::SmallPtrSet<OpOperand *, 16> seen;
  for (NodeId nodeid : entries) {
    auto it = g->linkIn.find(nodeid);
    if (it == g->linkIn.end()) {
      continue;
    }
    for (const Edge &edge : it->second) {
      if (!edge.use || nodeIds.contains(edge.from)) {
        continue;
      }
      if (seen.insert(edge.use).second) {
        inputOperands.push_back(edge.use);
      }
    }
  }
}

// Build a component from an SCC's node ids: derive the distinct owner ops,
// the entry nodes (those with an incoming edge from outside the SCC), and the
// boundary input operands / output results / entry ops.
ControlFlowGraph::ConnectedComponent::ConnectedComponent(
    const ControlFlowGraph *g, ArrayRef<NodeId> nodeIds)
    : nodeIds(nodeIds.begin(), nodeIds.end()), g(g) {
  llvm::DenseSet<Operation *> seen;
  for (NodeId id : this->nodeIds) {
    Operation *op = g->nodes[id].op;
    if (op && seen.insert(op).second) {
      ops.push_back(op);
    }
  }
  llvm::SmallDenseSet<NodeId> entriesSet;
  for (NodeId nodeid : this->nodeIds) {
    auto it = g->linkIn.find(nodeid);
    if (it == g->linkIn.end()) {
      continue;
    }
    if (llvm::any_of(it->second, [&](const Edge &edge) {
          return !this->nodeIds.contains(edge.from);
        })) {
      entriesSet.insert(nodeid);
    }
  }
  entries.assign(entriesSet.begin(), entriesSet.end());
  collectOutputResults();
  collectInputOperands();
  collectEntryOps();
}
