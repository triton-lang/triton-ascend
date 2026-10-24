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

#include "DynamicCVPipeline/Common/ControlFlowGraph.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace mlir::CVPipeline;

ControlFlowGraph::NodeId ControlFlowGraph::addNode(Kind kind, Operation *op,
                                                   Value value,
                                                   unsigned index) {
  NodeId id = nodes.size();
  nodes.push_back({kind, op, value, index});
  return id;
}

ControlFlowGraph::ControlFlowGraph(Operation *root)
    : ControlFlowGraph(root, [](OpOperand &) { return true; }) {}

ControlFlowGraph::ControlFlowGraph(
    Operation *root, llvm::function_ref<bool(OpOperand &)> includeDataOperand)
    : root(root) {
  declare(root);
  connect(root, includeDataOperand);
}

void ControlFlowGraph::declare(Operation *op) {
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    auto cond = conditions[op] = addNode(Kind::Condition, op);
    Loop desc{op,
              cond,
              {},
              {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}};
    auto iv = loop.getInductionVar();
    values[iv] = addNode(Kind::Phi, op, iv, 0);
    desc.headers.push_back(values[iv]);
    for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
      values[arg] = addNode(Kind::Phi, op, arg, i + 1);
      values[loop.getResult(i)] = values[arg];
      desc.headers.push_back(values[arg]);
    }
    loops.push_back(std::move(desc));
  } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
    auto cond = conditions[op] = addNode(Kind::Condition, op);
    auto term =
        cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
    Loop desc{op, cond, {}, {term.getCondition()}};
    for (auto [i, arg] : llvm::enumerate(loop.getBeforeArguments())) {
      values[arg] = addNode(Kind::Phi, op, arg, i);
      desc.headers.push_back(values[arg]);
    }
    // before and after signatures can differ in both arity and type.
    for (auto [arg, forwarded] :
         llvm::zip(loop.getAfterArguments(), term.getArgs())) {
      aliases[arg] = forwarded;
    }
    for (auto [result, forwarded] :
         llvm::zip(loop.getResults(), term.getArgs())) {
      aliases[result] = forwarded;
    }
    loops.push_back(std::move(desc));
  } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
    conditions[op] = addNode(Kind::Condition, op);
    for (auto [i, result] : llvm::enumerate(branch.getResults())) {
      values[result] = addNode(Kind::Phi, op, result, i);
    }
  } else if (!isa<scf::YieldOp, scf::ConditionOp>(op)) {
    auto id = operations[op] = addNode(Kind::Operation, op);
    for (Value result : op->getResults()) {
      values[result] = id;
    }
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block) {
        declare(&nested);
      }
    }
  }
}

ControlFlowGraph::NodeId ControlFlowGraph::resolve(Value value) {
  auto found = values.find(value);
  if (found != values.end()) {
    return found->second;
  }
  auto alias = aliases.find(value);
  if (alias != aliases.end()) {
    auto id = resolve(alias->second);
    values[value] = id;
    return id;
  }
  auto id = addNode(Kind::External, nullptr, value);
  values[value] = id;
  return id;
}

std::optional<ControlFlowGraph::NodeId>
ControlFlowGraph::lookup(Value value) const {
  auto it = values.find(value);
  return it == values.end() ? std::nullopt : std::optional<NodeId>(it->second);
}
std::optional<ControlFlowGraph::NodeId>
ControlFlowGraph::lookup(Operation *op) const {
  auto it = operations.find(op);
  return it == operations.end() ? std::nullopt
                                : std::optional<NodeId>(it->second);
}

void ControlFlowGraph::edge(Value value, NodeId to, EdgeKind kind, Role role,
                            OpOperand *use, Region *transition) {
  edges.push_back({resolve(value), to, kind, role, use, transition});
}

void ControlFlowGraph::controlRegion(Operation *owner, Region &region,
                                     NodeId condition) {
  region.walk([&](Operation *op) {
    if (isa<scf::YieldOp, scf::ConditionOp>(op)) {
      return;
    }
    for (Value operand : op->getOperands()) {
      // The source after resolving a while after-argument is outside after.
      auto source = nodes[resolve(operand)];
      Region *defined =
          source.value ? source.value.getParentRegion()
                       : (source.op ? source.op->getParentRegion() : nullptr);
      if (defined && region.isAncestor(defined)) {
        continue;
      }
      if (auto id = lookup(op)) {
        edges.push_back({condition, *id, EdgeKind::Control, Role::Branch,
                         nullptr, &region});
      } else if (auto it = conditions.find(op); it != conditions.end()) {
        edges.push_back({condition, it->second, EdgeKind::Control, Role::Branch,
                         nullptr, &region});
      }
      break;
    }
  });
}

void ControlFlowGraph::connect(
    Operation *op, llvm::function_ref<bool(OpOperand &)> includeDataOperand) {
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    auto cond = conditions[op];
    edge(loop.getLowerBound(), resolve(loop.getInductionVar()), EdgeKind::Data,
         Role::Init, &op->getOpOperand(0));
    edge(loop.getLowerBound(), cond, EdgeKind::Control, Role::LowerBound,
         &op->getOpOperand(0));
    edge(loop.getUpperBound(), cond, EdgeKind::Control, Role::UpperBound,
         &op->getOpOperand(1));
    edge(loop.getStep(), cond, EdgeKind::Control, Role::Step,
         &op->getOpOperand(2));
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
      edge(loop.getInitArgs()[i], resolve(arg), EdgeKind::Data, Role::Init,
           &op->getOpOperand(i + 3));
      edge(yield.getOperand(i), resolve(arg), EdgeKind::Yield, Role::Backedge,
           &yield->getOpOperand(i), &loop.getRegion());
    }
    controlRegion(op, loop.getRegion(), cond);
  } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
    auto term =
        cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
    edge(term.getCondition(), conditions[op], EdgeKind::Control,
         Role::Predicate, &term->getOpOperand(0));
    auto yield = cast<scf::YieldOp>(loop.getAfter().front().getTerminator());
    for (auto [i, arg] : llvm::enumerate(loop.getBeforeArguments())) {
      edge(loop.getInits()[i], resolve(arg), EdgeKind::Data, Role::Init,
           &op->getOpOperand(i));
      edge(yield.getOperand(i), resolve(arg), EdgeKind::Yield, Role::Backedge,
           &yield->getOpOperand(i), &loop.getAfter());
    }
    for (Value v : loop.getAfterArguments()) {
      resolve(v);
    }
    for (Value v : loop.getResults()) {
      resolve(v);
    }
    controlRegion(op, loop.getAfter(), conditions[op]);
  } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
    auto cond = conditions[op];
    edge(branch.getCondition(), cond, EdgeKind::Control, Role::Predicate,
         &op->getOpOperand(0));
    for (Region &region : op->getRegions()) {
      if (region.empty()) {
        continue;
      }
      auto yield = cast<scf::YieldOp>(region.front().getTerminator());
      for (auto [i, operand] : llvm::enumerate(yield.getOperands())) {
        auto phi = resolve(branch.getResult(i));
        edge(operand, phi, EdgeKind::Yield, Role::Operand,
             &yield->getOpOperand(i), &region);
        if (!region.isAncestor(operand.getParentRegion())) {
          edges.push_back({cond, phi, EdgeKind::Control, Role::Selection,
                           nullptr, &region});
        }
      }
      controlRegion(op, region, cond);
    }
  } else if (auto id = lookup(op)) {
    for (OpOperand &operand : op->getOpOperands()) {
      if (includeDataOperand(operand)) {
        edge(operand.get(), *id, EdgeKind::Data, Role::Operand, &operand);
      }
    }
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block) {
        connect(&nested, includeDataOperand);
      }
    }
  }
}

SmallVector<SmallVector<ControlFlowGraph::NodeId>>
ControlFlowGraph::getStronglyConnectedNodeComponents(
    llvm::function_ref<bool(NodeId)> isEligible) const {
  SmallVector<SmallVector<NodeId>> successors(nodes.size());
  for (const Edge &e : edges) {
    if (isEligible(e.from) && isEligible(e.to)) {
      successors[e.from].push_back(e.to);
    }
  }
  SmallVector<int> index(nodes.size(), -1), low(nodes.size());
  SmallVector<bool> active(nodes.size(), false);
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
    llvm::sort(component);
    components.push_back(std::move(component));
  };
  for (NodeId v = 0; v < nodes.size(); ++v) {
    if (isEligible(v) && index[v] == -1) {
      visit(v);
    }
  }
  llvm::sort(components, [](const auto &lhs, const auto &rhs) {
    return lhs.front() < rhs.front();
  });
  return components;
}

SmallVector<SmallVector<ControlFlowGraph::NodeId>>
ControlFlowGraph::getNodeSCCs(
    llvm::function_ref<bool(Operation *)> isLegal) const {
  SmallVector<bool> eligible(nodes.size(), false);
  for (NodeId id = 0; id < nodes.size(); ++id) {
    const Node &node = nodes[id];
    switch (node.kind) {
    case Kind::Operation:
      eligible[id] = isLegal(node.op);
      break;
    case Kind::Phi:
    case Kind::Condition:
      eligible[id] = true;
      break;
    case Kind::External:
      break;
    }
  }
  auto components = getStronglyConnectedNodeComponents(
      [&](NodeId id) { return eligible[id]; });
  llvm::erase_if(components, [&](const auto &component) {
    return llvm::none_of(component, [&](NodeId id) {
      return nodes[id].kind == Kind::Operation;
    });
  });
  return components;
}

SmallVector<ControlFlowGraph::Components>
ControlFlowGraph::toComponents(ArrayRef<SmallVector<NodeId>> components) const {
  SmallVector<Components> result;
  for (const auto &ids : components) {
    Components component;
    llvm::SmallPtrSet<Operation *, 16> seen;
    for (NodeId id : ids) {
      component.nodeId.push_back(static_cast<int>(id));
      Operation *op = nodes[id].op;
      if (op && seen.insert(op).second) {
        component.Ops.push_back(op);
      }
    }
    if (!component.Ops.empty()) {
      result.push_back(std::move(component));
    }
  }
  return result;
}

SmallVector<ControlFlowGraph::Components>
ControlFlowGraph::getSCCs(llvm::function_ref<bool(Operation *)> isLegal) const {
  return toComponents(getNodeSCCs(isLegal));
}

SmallVector<ControlFlowGraph::Components>
ControlFlowGraph::getStronglyConnectedComponents() const {
  return toComponents(
      getStronglyConnectedNodeComponents([](NodeId) { return true; }));
}

SmallVector<int>
ControlFlowGraph::getLoopEntry(const Components &component) const {
  llvm::SmallDenseSet<int> members(component.nodeId.begin(),
                                   component.nodeId.end());
  llvm::SmallDenseSet<int> entrys;
  for (const Edge &edge : edges) {
    if (edge.kind != EdgeKind::Control && members.contains(edge.to) &&
        !members.contains(edge.from)) {
      entrys.insert(edge.to);
    }
  }
  SmallVector<int> result(entrys.begin(), entrys.end());
  llvm::sort(result);
  return result;
}

SmallVector<std::pair<ControlFlowGraph::NodeId, OpOperand *>>
ControlFlowGraph::getOutgoingUses(const Components &component) const {
  llvm::SmallDenseSet<int> members(component.nodeId.begin(),
                                   component.nodeId.end());
  llvm::SmallPtrSet<OpOperand *, 16> internalUses;
  llvm::SmallPtrSet<OpOperand *, 16> controlUses;
  for (const Edge &edge : edges) {
    if (!edge.use) {
      continue;
    }
    if (edge.kind == EdgeKind::Control) {
      controlUses.insert(edge.use);
    } else if (members.contains(edge.from) && members.contains(edge.to)) {
      internalUses.insert(edge.use);
    }
  }
  SmallVector<std::pair<NodeId, OpOperand *>> result;
  // One graph node can represent several results and SCF aliases. Inspect
  // every mapped value so that omitted A/B uses remain visible as exits.
  for (const auto &[value, id] : values) {
    if (!members.contains(id)) {
      continue;
    }
    for (OpOperand &use : value.getUses()) {
      if (internalUses.contains(&use) || controlUses.contains(&use)) {
        continue;
      }
      if (isa<scf::ConditionOp>(use.getOwner())) {
        // Non-predicate operands forward to after arguments and while results.
        // The predicate is a Control dependency. Neither is an outgoing read.
        continue;
      }
      result.emplace_back(id, &use);
    }
  }
  return result;
}

SmallVector<int>
ControlFlowGraph::getLoopExit(const Components &component) const {
  llvm::SmallDenseSet<int> exits;
  for (auto [id, use] : getOutgoingUses(component)) {
    exits.insert(id);
  }
  SmallVector<int> result(exits.begin(), exits.end());
  llvm::sort(result);
  return result;
}

SmallVector<Value> ControlFlowGraph::getSCCInputValues(
    const Components &component, ArrayRef<int> entrys,
    SmallVectorImpl<OpOperand *> *inputUses) const {
  if (inputUses) {
    inputUses->clear();
  }
  llvm::SmallDenseSet<int> members(component.nodeId.begin(),
                                   component.nodeId.end());
  llvm::SmallDenseSet<int> selected(entrys.begin(), entrys.end());
  llvm::SmallDenseSet<Value> seenValues;
  llvm::SmallPtrSet<OpOperand *, 16> seenUses;
  SmallVector<Value> result;
  for (const Edge &edge : edges) {
    if (edge.kind == EdgeKind::Control || !selected.contains(edge.to) ||
        !members.contains(edge.to) || members.contains(edge.from) ||
        !edge.use) {
      continue;
    }
    Value input = edge.use->get();
    if (seenValues.insert(input).second) {
      result.push_back(input);
    }
    if (inputUses && seenUses.insert(edge.use).second) {
      inputUses->push_back(edge.use);
    }
  }
  return result;
}

SmallVector<Value>
ControlFlowGraph::getSCCResultValues(const Components &component,
                                     ArrayRef<int> exits) const {
  llvm::SmallDenseSet<int> selected(exits.begin(), exits.end());
  llvm::SmallDenseSet<Value> seen;
  SmallVector<Value> result;
  for (auto [id, use] : getOutgoingUses(component)) {
    Value output = use->get();
    if (selected.contains(id) && seen.insert(output).second) {
      result.push_back(output);
    }
  }
  return result;
}
