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

#ifndef TRITON_ASCEND_DYNAMIC_CV_CONTROL_FLOW_GRAPH_H
#define TRITON_ASCEND_DYNAMIC_CV_CONTROL_FLOW_GRAPH_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <utility>

namespace mlir::CVPipeline {

// Local graph: definitions outside root become external value nodes. No IR is
// changed. Edges point from producer to consumer; Phi selection is NOT an AND
// dependency. `transition` records the actual region supplying an incoming
// value.
class ControlFlowGraph {
public:
  struct Components {
    // SCF operations owning Phi/Condition nodes are included once, alongside
    // the legal computation operations. nodeId retains every graph node.
    SmallVector<Operation *> Ops;
    SmallVector<int> nodeId;
  };

  explicit ControlFlowGraph(Operation *root);
  // Filter ordinary computation Data edges while building the graph. SCF
  // forwarding and control dependencies are preserved. The callback is used
  // synchronously during construction and is not retained by the graph.
  ControlFlowGraph(Operation *root,
                   llvm::function_ref<bool(OpOperand &)> includeDataOperand);

  // Strongly connected components of the induced legal-node directed graph.
  // Edges retain their original direction, including Control dependencies.
  // The predicate is called only for computation operations. Phi/Condition
  // remain eligible; their SCF owners are included in Ops. Components with
  // no computation operations are omitted. No boundary/residency constraints
  // are imposed here; a computation without a cycle forms a singleton SCC.
  SmallVector<Components>
  getSCCs(llvm::function_ref<bool(Operation *)> isLegal) const;

  // Conventional directed SCCs with the same operation/node representation.
  // Components containing only external nodes are omitted.
  SmallVector<Components> getStronglyConnectedComponents() const;

  // Return component-side boundary nodes, ignoring only Control edges.
  // These queries enumerate boundaries and never require them to be unique.
  SmallVector<int> getLoopEntry(const Components &component) const;
  SmallVector<int> getLoopExit(const Components &component) const;

  // Enumerate distinct SSA values crossing the selected boundary nodes.
  // Operand sites are optional and permit precise rewrites of every input.
  SmallVector<Value>
  getSCCInputValues(const Components &component, ArrayRef<int> entrys,
                    SmallVectorImpl<OpOperand *> *inputUses = nullptr) const;
  SmallVector<Value> getSCCResultValues(const Components &component,
                                        ArrayRef<int> exits) const;

private:
  using NodeId = unsigned;
  enum class Kind { Operation, Phi, Condition, External };
  enum class EdgeKind { Data, Yield, Control };
  enum class Role {
    Operand,
    Init,
    Backedge,
    LowerBound,
    UpperBound,
    Step,
    Predicate,
    Branch,
    Selection
  };
  struct Node {
    Kind kind;
    // For virtual nodes this is the SCF owner, not a corresponding real op.
    Operation *op;
    Value value;
    unsigned index = 0;
  };
  struct Edge {
    NodeId from, to;
    EdgeKind kind;
    Role role;
    OpOperand *use = nullptr;
    Region *transition = nullptr;
  };
  struct Loop {
    Operation *op;
    NodeId condition;
    SmallVector<NodeId> headers;
    // for: lb/ub/step; while: predicate. IV update is implicit, never
    // represented by a fabricated arith.addi operation.
    SmallVector<Value> domain;
  };

  std::optional<NodeId> lookup(Value value) const;
  std::optional<NodeId> lookup(Operation *op) const;
  SmallVector<SmallVector<NodeId>>
  getNodeSCCs(llvm::function_ref<bool(Operation *)> isLegal) const;
  SmallVector<SmallVector<NodeId>> getStronglyConnectedNodeComponents(
      llvm::function_ref<bool(NodeId)> isEligible) const;
  // Include actual SSA observations omitted by a computation-input filter;
  // they are outgoing uses of the selected value flow, not internal edges.
  SmallVector<std::pair<NodeId, OpOperand *>>
  getOutgoingUses(const Components &component) const;
  SmallVector<Components>
  toComponents(ArrayRef<SmallVector<NodeId>> components) const;
  Operation *root;
  SmallVector<Node> nodes;
  SmallVector<Edge> edges;
  SmallVector<Loop> loops;
  DenseMap<Value, NodeId> values;
  DenseMap<Operation *, NodeId> operations;
  DenseMap<Value, Value> aliases;
  DenseMap<Operation *, NodeId> conditions;
  NodeId addNode(Kind kind, Operation *op, Value value = {},
                 unsigned index = 0);
  NodeId resolve(Value value);
  void declare(Operation *op);
  void connect(Operation *op,
               llvm::function_ref<bool(OpOperand &)> includeDataOperand);
  void edge(Value value, NodeId to, EdgeKind kind, Role role,
            OpOperand *use = nullptr, Region *transition = nullptr);
  void controlRegion(Operation *owner, Region &region, NodeId condition);
};

} // namespace mlir::CVPipeline
#endif
