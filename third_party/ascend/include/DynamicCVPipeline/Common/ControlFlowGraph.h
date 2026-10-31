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
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <utility>

namespace mlir::scf {
class ConditionOp;
class ForOp;
class IfOp;
class WhileOp;
class YieldOp;
} // namespace mlir::scf

namespace mlir::CVPipeline {

// Local graph: definitions outside root are boundary inputs. No IR is
// changed. Edges point from producer to consumer; Phi selection is NOT an AND
// dependency. Values outside root do not create graph nodes.

class ControlFlowGraph {
public:
  using NodeId = unsigned;

  // A connected component is one SCC of the legal-node graph, packaged with
  // its boundary: the computation ops it spans, the entry nodes (nodes with
  // an incoming edge from outside), and the SSA operand/value sites crossing
  // that boundary. `nodeIds` retains every graph node in the SCC.
  class ConnectedComponent {
  public:
    explicit ConnectedComponent(const ControlFlowGraph *g) : g(g) {}
    ConnectedComponent(const ControlFlowGraph *g, ArrayRef<NodeId> nodeIds);
    SmallVector<Operation *> ops;
    llvm::SetVector<NodeId> nodeIds;
    const ControlFlowGraph *g;
    SmallVector<OpResult> outputResults;
    SmallVector<Operation *> entryOps;
    SmallVector<OpOperand *> inputOperands;
    SmallVector<NodeId> entries;
    // Include filtered operands and uses of both Phi arguments and results.
    SmallVector<OpOperand *> getExternalUses() const;

  private:
    void collectOutputResults();
    void collectEntryOps();
    void collectInputOperands();
  };

  explicit ControlFlowGraph(Operation *root);
  // Filter ordinary operation Data edges by operand. SCF control and
  // loop-carried/yield edges retain their structural meaning.
  ControlFlowGraph(Operation *root,
                   llvm::function_ref<bool(OpOperand &)> includeDataOperand);

  // Strongly connected components of the induced legal-node directed graph.
  // Edges retain their original direction. The predicate is called only for
  // computation ops; Phi nodes are always eligible, and their SCF/func owners
  // are folded into each component's `ops`. Components with no computation op
  // are dropped. A computation without a cycle forms a singleton SCC.
  SmallVector<ConnectedComponent, 0> getStronglyConnectedComponents(
      llvm::function_ref<bool(Operation *)> isLegal) const;

private:
  // Operation: a computation; one node per result, with result 0 doubling as
  // the op's representative (registered in OperationToNodeId).
  // Phi: an SSA value selected or updated by an SCF construct (for iter-arg,
  // while before/after-arg, if result) or a function argument.
  enum class Kind { Operation, Phi, Exit };
  // Data: ordinary SSA use (producer -> consumer).
  // Yield: a value forwarded by scf.yield or scf.condition into a phi/result.
  // DeadResultExit preserves unused result boundaries without an SSA use.
  enum class EdgeKind { Data, Yield, DeadResultExit };

  struct Node {
    Kind kind;
    // Defining op for Operation nodes; SCF/func owner for Phi nodes.
    Operation *op;
    Value value;     // the represented SSA value
    OpResult result; // exported OpResult; null for IV and function args
  };
  struct Edge {
    NodeId from, to;
    EdgeKind kind;
    OpOperand *use = nullptr;
  };
  std::optional<NodeId> lookup(Value value) const {
    auto it = valueToNodeId.find(value);
    return it == valueToNodeId.end() ? std::nullopt
                                     : std::optional<NodeId>(it->second);
  }
  std::optional<NodeId> lookup(Operation *op) const {
    auto it = OperationToNodeId.find(op);
    return it == OperationToNodeId.end() ? std::nullopt
                                         : std::optional<NodeId>(it->second);
  }

  SmallVector<ConnectedComponent, 0>
  toComponents(ArrayRef<SmallVector<NodeId>> components) const;

  SmallVector<SmallVector<NodeId>>
  getNodeSCCs(llvm::function_ref<bool(Operation *)> isLegal) const;

  // DS of the graph
  SmallVector<Node> nodes;
  DenseMap<Value, NodeId> valueToNodeId;
  DenseMap<Operation *, NodeId> OperationToNodeId;
  DenseMap<NodeId, SmallVector<Edge>> linkIn;
  DenseMap<NodeId, SmallVector<Edge>> linkOut;

  // Preconditions: values belong to the root walk. A non-null exitResult
  // aliases the block argument's node; result 0 represents a computation op.
  void createNode(BlockArgument value, Kind kind, OpResult exitResult = {});
  void createNode(OpResult value, Kind kind);
  void addEdge(NodeId from, NodeId to, EdgeKind kind, OpOperand *use = nullptr);
  void createFor(scf::ForOp forOp);
  void createIf(scf::IfOp ifOp);
  void createWhile(scf::WhileOp whileOp);
  void createNodes(Operation *op);
  void connectEdges(Operation *op,
                    llvm::function_ref<bool(OpOperand &)> includeDataOperand);
  void connectYieldOp(scf::YieldOp yieldOp);
  void connectConditionOp(scf::ConditionOp condOp);
};

} // namespace mlir::CVPipeline
#endif
