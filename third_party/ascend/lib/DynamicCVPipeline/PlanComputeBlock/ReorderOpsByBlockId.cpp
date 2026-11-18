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
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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

#include <fstream>
#include <string>
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"

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

void dumpBlockOpGraphToDot(const BlockOpGraph &graph, const std::string &filename) {
    std::ofstream os(filename);
    if (!os.is_open()) {
        llvm::errs() << "Failed to open file for DOT dump: " << filename << "\n";
        return;
    }

    os << "digraph BlockOpGraph {\n";
    os << "  rankdir=TB;\n"; // Top-to-bottom layout
    os << "  node [shape=box, fontname=\"Courier\", style=\"filled\", fillcolor=\"#f0f4f8\"];\n";
    os << "  edge [color=\"#333333\", arrowhead=\"normal\"];\n\n";

    // 1. Emit all nodes with formatted MLIR op representations
    for (size_t i = 0; i < graph.ops.size(); ++i) {
        mlir::Operation *op = graph.ops[i];

        // Print operation to a string
        std::string opStr;
        llvm::raw_string_ostream ss(opStr);
        op->print(ss, mlir::OpPrintingFlags().skipRegions().printGenericOpForm());

        // Escape special characters for DOT syntax
        std::string safeLabel;
        for (char c : ss.str()) {
            if (c == '"') safeLabel += "\\\"";
            else if (c == '\\') safeLabel += "\\\\";
            else if (c == '\n') safeLabel += "\\l"; // Left-align line breaks in Graphviz
            else safeLabel += c;
        }

        os << "  Node_" << i << " [label=\"[" << i << "] " 
           << op->getName().getStringRef().str() << "\\l" 
           << safeLabel << "\"];\n";
    }

    os << "\n  // Edges (Successors)\n";

    // 2. Emit directed edges from 'succs' map
    for (const auto &[op, successors] : graph.succs) {
        auto srcIt = graph.opIndex.find(op);
        if (srcIt == graph.opIndex.end()) continue;
        unsigned srcIdx = srcIt->second;

        for (mlir::Operation *succOp : successors) {
            auto dstIt = graph.opIndex.find(succOp);
            if (dstIt != graph.opIndex.end()) {
                unsigned dstIdx = dstIt->second;
                os << "  Node_" << srcIdx << " -> Node_" << dstIdx << ";\n";
            }
        }
    }

    os << "}\n";
    os.close();
    llvm::errs() << "Successfully dumped BlockOpGraph to " << filename << "\n";
}

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

//===----------------------------------------------------------------------===//
// Which ready block to emit next
//===----------------------------------------------------------------------===//
// Every point at which more than one block is ready is a point where a
// different -- and equally legal -- linearisation of the same dependency graph
// could be produced. The rule below is the whole of the pass's freedom, so it
// is made switchable: two variants of the IR differing only in it can be
// compiled and compared without touching anything else.
//
// Lifo is what the pass has always done (`pop_back_val()` on a vector is the
// cheapest way to take an element, which is most likely why it was chosen
// rather than for a reason). It follows a producer into its consumer, so a
// chain stays together, but among blocks that became ready at the same moment
// it takes the one that appeared *last* in the IR.
//
// Fifo takes the one that became ready first, which for blocks that start out
// ready means the earliest in the IR. On a graph where a compute block and a
// staging block become ready together, the two rules put them on opposite
// sides of everything that follows.
//
// Driven by an environment variable rather than a pass option: this pass is
// created inside PlanComputeBlock's own nested pipeline, so an option would
// have to be threaded through two pass managers, and the point is to be able
// to flip it between runs without rebuilding.

constexpr llvm::StringLiteral kReorderPolicyEnvVar =
    "TRITON_ASCEND_REORDER_POLICY";

enum class ReadyPolicy { Lifo, Fifo };

llvm::StringRef describeReadyPolicy(ReadyPolicy policy) {
  return policy == ReadyPolicy::Fifo ? "fifo" : "lifo";
}

/// Read once. A non-default choice announces itself unconditionally: a run
/// that silently ignored the variable would be indistinguishable from one that
/// honoured it, and the two are supposed to produce different IR.
ReadyPolicy getReadyPolicy() {
  static const ReadyPolicy policy = [] {
    const char *env = std::getenv(kReorderPolicyEnvVar.data());
    if (!env) {
      return ReadyPolicy::Lifo;
    }
    const std::string value = llvm::StringRef(env).lower();
    if (value == "fifo") {
      llvm::errs() << "[reorder-blocks] " << kReorderPolicyEnvVar
                   << "=fifo: emitting the block that became ready first"
                      " instead of last\n";
      return ReadyPolicy::Fifo;
    }
    if (value != "lifo") {
      llvm::errs() << "[reorder-blocks] unknown " << kReorderPolicyEnvVar
                   << "='" << value << "', expected lifo or fifo;"
                      " using lifo\n";
    }
    return ReadyPolicy::Lifo;
  }();
  return policy;
}

/// Seed selecting one operation-level variant, or nothing for the ordinary
/// block-level path.
///
/// Read from the module rather than the environment because the variant search
/// sets a different seed on every attempt within one process, which a
/// process-wide variable cannot express. The environment is still honoured so
/// a single variant can be reproduced by hand.
std::optional<uint64_t> getVariantSeed(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(CVPipeline::kReorderSeed)) {
    // A negative value marks a seed that has already been applied. This pass
    // runs twice -- at the end of PlanComputeBlock and again at the end of
    // SplitDataflow -- and only the first run may regroup: by the second one
    // the blocks carry transfer groups and synchronisation flags built around
    // their ids, so renumbering them there would leave those attributes
    // pointing at blocks that no longer exist.
    // Zero is the baseline: the pipeline's own grouping, untouched. The search
    // spends its first attempt on it deliberately, so "no variant beat the
    // best" and "no variant beat my own heuristic" cannot be confused -- the
    // number every other seed is compared against is the one the compiler
    // produces without any of this.
    if (attr.getInt() <= 0) {
      return std::nullopt;
    }
    return static_cast<uint64_t>(attr.getInt());
  }
  if (const char *env = std::getenv("TRITON_ASCEND_REORDER_SEED")) {
    uint64_t value = 0;
    if (!llvm::StringRef(env).getAsInteger(10, value)) {
      // Zero means the same here as it does in the attribute: the untouched
      // pipeline. Reproducing "the baseline the search compared against" by
      // hand has to name the same thing the search named.
      if (value == 0) {
        return std::nullopt;
      }
      return value;
    }
    llvm::errs() << "[reorder-blocks] TRITON_ASCEND_REORDER_SEED='" << env
                 << "' is not a number; ignored\n";
  }
  return std::nullopt;
}

/// Take one block out of the ready set, per the policy. Both rules yield a
/// legal topological order; only the order differs.
unsigned takeReady(SmallVector<unsigned> &ready, ReadyPolicy policy) {
  if (policy == ReadyPolicy::Fifo) {
    const unsigned first = ready.front();
    ready.erase(ready.begin());
    return first;
  }
  return ready.pop_back_val();
}

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

void dumpGroupAdjacencyGraphToDot(const GroupAdjacencyGraph &graph, const std::string &filename) {
    std::ofstream os(filename);
    if (!os.is_open()) {
        llvm::errs() << "Failed to open file for DOT dump: " << filename << "\n";
        return;
    }

    os << "digraph GroupAdjacencyGraph {\n";
    os << "  rankdir=TB;\n"; // Top-to-bottom layout
    
    // Aesthetic styling for group nodes
    os << "  node [shape=Mrecord, fontname=\"Helvetica\", style=\"filled\", fillcolor=\"#e2f0d9\", color=\"#548235\"];\n";
    os << "  edge [color=\"#385723\", penwidth=1.5];\n\n";

    // 1. Emit nodes (Group IDs)
    // The graph structure implies indices 0...groupIds.size()-1 map to the actual group IDs
    for (size_t i = 0; i < graph.groupIds.size(); ++i) {
        int actualGroupId = graph.groupIds[i];
        unsigned inDegree = (i < graph.inDeg.size()) ? graph.inDeg[i] : 0;
        
        // Node format: Index | Group ID | In-Degree
        os << "  Node_" << i << " [label=\"{Idx: " << i 
           << " | Group ID: " << actualGroupId 
           << " | In-Deg: " << inDegree << "}\"];\n";
    }

    os << "\n  // Edges (Group Successors)\n";

    // 2. Emit edges
    for (size_t i = 0; i < graph.succs.size(); ++i) {
        for (unsigned targetIdx : graph.succs[i]) {
            os << "  Node_" << i << " -> Node_" << targetIdx << ";\n";
        }
    }

    os << "}\n";
    os.close();
    llvm::errs() << "Successfully dumped GroupAdjacencyGraph to " << filename << "\n";
}

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

  const ReadyPolicy policy = getReadyPolicy();
  while (!ready.empty()) {
    auto cur = takeReady(ready, policy);

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

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//
// How much freedom the topological sort actually has is invisible once the IR
// has been linearised: a graph with exactly one legal order and a graph with
// thousands produce output that looks the same. This reports it, because
// whether generating IR variants along this axis is worth anything depends
// entirely on that number -- and so does the question of whether the freedom
// was lost when operations were grouped into blocks rather than when the blocks
// were ordered.
//
// Driven by an environment variable rather than LLVM_DEBUG because a debug
// build plus -debug-only is impractical when compilation is driven from Python.

namespace {

constexpr llvm::StringLiteral kReorderVerboseEnvVar =
    "TRITON_ASCEND_REORDER_VERBOSE";

int getReorderVerbosity() {
  static const int verbosity = [] {
    const char *env = std::getenv(kReorderVerboseEnvVar.data());
    if (!env) {
      return 0;
    }
    int value = 0;
    return llvm::StringRef(env).getAsInteger(10, value) ? 0 : value;
  }();
  return verbosity;
}

llvm::StringRef describeGroupCore(unsigned core) {
  switch (core) {
  case CoreType::CUBE_ONLY:
    return "CUBE";
  case CoreType::VECTOR_ONLY:
    return "VECTOR";
  case CoreType::CUBE_AND_VECTOR:
    return "MIXED";
  default:
    return "-";
  }
}

/// Above this many groups the exact count below is not attempted: the DP is
/// exponential in the node count, and 2^20 states is already 8 MB.
constexpr unsigned kMaxGroupsForExactCount = 20;

/// Number of distinct legal orders of the group graph, i.e. its linear
/// extensions. Counted by dynamic programming over the set of groups already
/// emitted: from a given set, any group whose predecessors are all in that set
/// may come next. That is exactly the choice the topological sort makes, so the
/// result is the size of the variant space this pass could explore.
uint64_t countLinearExtensions(const SmallVector<SmallVector<unsigned>> &succs,
                               unsigned n) {
  SmallVector<uint32_t> predMask(n, 0);
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned succ : succs[i]) {
      predMask[succ] |= (1u << i);
    }
  }

  const uint32_t full = (1u << n) - 1u;
  std::vector<uint64_t> ways(static_cast<size_t>(full) + 1, 0);
  ways[0] = 1;
  for (uint32_t done = 0; done < full; ++done) {
    if (ways[done] == 0) {
      continue;
    }
    for (unsigned i = 0; i < n; ++i) {
      if (done & (1u << i)) {
        continue; // already emitted
      }
      if ((predMask[i] & done) != predMask[i]) {
        continue; // a predecessor has not run yet
      }
      ways[done | (1u << i)] += ways[done];
    }
  }
  return ways[full];
}

/// Name of the nearest enclosing symbol, so the two mainloop bodies that
/// SeparateCVScope leaves behind can be told apart in the log.
std::string describeScope(Block *block) {
  for (Operation *op = block->getParentOp(); op; op = op->getParentOp()) {
    if (auto sym = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
      return sym.getValue().str();
    }
  }
  return "<anonymous>";
}

/// How wide the *operation* graph is, before grouping collapsed it. Reported
/// alongside the group graph to answer the question the group graph alone
/// cannot: if the operations were free to move and the groups are not, then it
/// is the block partition that removed the freedom, not the dependencies.
void reportOpGraphWidth(llvm::raw_ostream &os, const BlockOpGraph &graph) {
  DenseMap<Operation *, unsigned> inDeg;
  for (Operation *op : graph.ops) {
    inDeg[op] = graph.preds.at(op).size();
  }
  SmallVector<Operation *> ready;
  for (Operation *op : graph.ops) {
    if (inDeg[op] == 0) {
      ready.push_back(op);
    }
  }

  unsigned widest = 0;
  unsigned steps = 0;
  unsigned forced = 0;
  uint64_t total = 0;
  while (!ready.empty()) {
    widest = std::max<unsigned>(widest, ready.size());
    total += ready.size();
    if (ready.size() == 1) {
      ++forced;
    }
    ++steps;
    // Deliberately not the configurable policy: the operation graph is the
    // same whichever way blocks are ordered, so keeping this traversal fixed
    // makes the width comparable between runs that used different policies.
    Operation *cur = ready.pop_back_val();
    for (Operation *succ : graph.succs.at(cur)) {
      if (--inDeg[succ] == 0) {
        ready.push_back(succ);
      }
    }
  }

  os << "  op-level graph: " << graph.ops.size() << " op(s), widest "
     << widest << " ready at once, mean "
     << llvm::format("%.2f", steps ? static_cast<double>(total) /
                                         static_cast<double>(steps)
                                   : 0.0)
     << ", " << forced << " of " << steps << " step(s) forced\n";
}

/// Everything about one block's group graph: who depends on whom, how many
/// legal orders that leaves, and which one was taken.
void reportGroupGraph(const GroupAdjacencyGraph &adjacency,
                      const BlockOpGraph &graph,
                      const DenseMap<Operation *, int> &opBlockId,
                      ArrayRef<unsigned> inDegSnapshot,
                      ArrayRef<int> chosenOrder) {
  const unsigned n = adjacency.groupIds.size();
  llvm::raw_ostream &os = llvm::errs();

  // Per group: how many operations, which core(s), and the heaviest op kinds.
  // CoreType is a bit mask by construction, so a group whose operations
  // disagree ORs to CUBE_AND_VECTOR on its own.
  SmallVector<unsigned> opCount(n, 0);
  SmallVector<unsigned> core(n, CoreType::UNDETERMINED);
  SmallVector<SmallVector<std::pair<llvm::StringRef, unsigned>>> kinds(n);
  DenseMap<int, unsigned> groupPos;
  for (unsigned i = 0; i < n; ++i) {
    groupPos[adjacency.groupIds[i]] = i;
  }
  for (Operation *op : graph.ops) {
    const unsigned idx = groupPos[opBlockId.at(op)];
    ++opCount[idx];
    core[idx] |= static_cast<unsigned>(CVPipeline::getOpCoreType(op));
    const llvm::StringRef name = op->getName().getStringRef();
    bool found = false;
    for (auto &entry : kinds[idx]) {
      if (entry.first == name) {
        ++entry.second;
        found = true;
        break;
      }
    }
    if (!found) {
      kinds[idx].push_back({name, 1});
    }
  }

  os << "[reorder-blocks] === " << describeScope(adjacency.block) << " / "
     << adjacency.block->getParentOp()->getName().getStringRef() << ": "
     << graph.ops.size() << " op(s) in " << n << " group(s)\n";
  os << "[reorder-blocks]   group  core     ops  indeg  succs -> | contents\n";

  for (unsigned i = 0; i < n; ++i) {
    os << "[reorder-blocks] "
       << llvm::format("%7d", adjacency.groupIds[i]) << "  "
       << llvm::left_justify(describeGroupCore(core[i]), 7)
       << llvm::format("%4u", opCount[i])
       << llvm::format("%7u", inDegSnapshot[i]) << "  ";
    if (adjacency.succs[i].empty()) {
      os << "-";
    } else {
      for (size_t s = 0; s < adjacency.succs[i].size(); ++s) {
        if (s != 0) {
          os << ", ";
        }
        os << adjacency.groupIds[adjacency.succs[i][s]];
      }
    }
    os << " | ";
    llvm::sort(kinds[i], [](const auto &lhs, const auto &rhs) {
      return lhs.second > rhs.second;
    });
    for (size_t k = 0; k < kinds[i].size() && k < 3; ++k) {
      if (k != 0) {
        os << ", ";
      }
      os << kinds[i][k].first;
      if (kinds[i][k].second > 1) {
        os << " x" << kinds[i][k].second;
      }
    }
    if (kinds[i].size() > 3) {
      os << ", +" << (kinds[i].size() - 3) << " more";
    }
    os << "\n";
  }

  // Replay the sort with the rule the pass actually uses, recording only how
  // many groups were ready at each step. A step with one ready group is forced;
  // a step with several is a point where a different variant could be produced.
  SmallVector<unsigned> inDeg(inDegSnapshot.begin(), inDegSnapshot.end());
  SmallVector<unsigned> ready;
  for (unsigned i = 0; i < n; ++i) {
    if (inDeg[i] == 0) {
      ready.push_back(i);
    }
  }
  SmallVector<unsigned> choices;
  const ReadyPolicy policy = getReadyPolicy();
  while (!ready.empty()) {
    choices.push_back(ready.size());
    const unsigned cur = takeReady(ready, policy);
    for (unsigned succ : adjacency.succs[cur]) {
      if (--inDeg[succ] == 0) {
        ready.push_back(succ);
      }
    }
  }

  os << "[reorder-blocks]   ready groups per step:";
  for (unsigned c : choices) {
    os << " " << c;
  }
  os << "\n";

  os << "[reorder-blocks]   distinct legal orders: ";
  if (n == 0) {
    os << "0\n";
  } else if (n > kMaxGroupsForExactCount) {
    os << "not counted (" << n << " groups, cap is " << kMaxGroupsForExactCount
       << ")\n";
  } else {
    os << static_cast<unsigned long long>(
              countLinearExtensions(adjacency.succs, n))
       << "\n";
  }

  os << "[reorder-blocks] ";
  reportOpGraphWidth(os, graph);

  os << "[reorder-blocks]   chosen order ("
     << describeReadyPolicy(getReadyPolicy()) << "):";
  for (int id : chosenOrder) {
    os << " " << id;
  }
  os << "\n";
}

} // namespace

//===----------------------------------------------------------------------===//
// Operation-level variant generation
//===----------------------------------------------------------------------===//
// Ordering whole blocks can only ever produce as many variants as the block
// graph has linear extensions, and on a kernel whose blocks form a chain that
// is exactly one. Ordering the *operations* instead ignores the block grouping
// entirely and draws from a space that is larger by many orders of magnitude:
// the operation graph of one loop body was measured with 22 operations ready at
// once, and any of those 22! orders extends to a legal one.
//
// That also means exhaustive enumeration is out of the question, so orders are
// sampled instead. Priority is the operation's height -- the longest chain of
// operations that still has to run after it -- which is the classic list
// scheduling rule: start the longest remaining chain first and let everything
// else fill in around it. Seed 0 takes that rule exactly, giving a
// deterministic baseline; every other seed perturbs it, so the sample stays
// clustered around sensible schedules rather than wandering over random ones.

namespace {

/// Deterministic PRNG. Hand-rolled rather than <random> so that a seed means
/// the same order on every machine and standard library.
struct SplitMix64 {
  uint64_t state;
  explicit SplitMix64(uint64_t seed) : state(seed) {}
  uint64_t next() {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
};

/// Which of the ready operations to take, given they are sorted best-first.
///
/// Index 0 is the list scheduling choice. A seeded run keeps taking it about
/// half the time and slips to the next-best otherwise, so a variant differs
/// from the baseline in a few decisions rather than in all of them -- the point
/// is to explore around a good schedule, not to shuffle.
size_t pickReadyIndex(SplitMix64 &rng, size_t count) {
  size_t index = 0;
  while (index + 1 < count && (rng.next() & 1U) == 0) {
    ++index;
  }
  return index;
}

/// Longest chain of operations that must still run after each operation.
/// Computed over the reverse graph, so it needs no separate traversal order.
DenseMap<Operation *, unsigned> computeOpHeights(const BlockOpGraph &graph) {
  DenseMap<Operation *, unsigned> height;
  DenseMap<Operation *, unsigned> outDeg;
  SmallVector<Operation *> queue;
  for (Operation *op : graph.ops) {
    const unsigned degree = graph.succs.at(op).size();
    outDeg[op] = degree;
    height[op] = 0;
    if (degree == 0) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (Operation *pred : graph.preds.at(op)) {
      unsigned &predHeight = height[pred];
      predHeight = std::max(predHeight, height[op] + 1);
      if (--outDeg[pred] == 0) {
        queue.push_back(pred);
      }
    }
  }
  return height;
}

/// One legal order of the operations, sampled by `seed`.
SmallVector<Operation *> sampleOpOrder(const BlockOpGraph &graph,
                                       uint64_t seed) {
  const DenseMap<Operation *, unsigned> height = computeOpHeights(graph);

  // Looked up once: the comparator below runs O(n log n) times per step, and
  // reading an attribute that often is both slow and pointless.
  DenseMap<Operation *, CoreType> core;
  DenseMap<Operation *, unsigned> inDeg;
  SmallVector<Operation *> ready;
  for (Operation *op : graph.ops) {
    core[op] = CVPipeline::getOpCoreType(op);
    const unsigned degree = graph.preds.at(op).size();
    inDeg[op] = degree;
    if (degree == 0) {
      ready.push_back(op);
    }
  }

  // Whether emitting this operation would keep the current run going. An
  // operation that names no core -- a constant, an scf container -- never ends
  // a run, and at the very start nothing has been chosen yet so everything
  // qualifies.
  auto continuesRun = [&](Operation *op, CoreType current) {
    const CoreType opCore = core.lookup(op);
    return current == CoreType::UNDETERMINED ||
           opCore == CoreType::UNDETERMINED || opCore == current;
  };

  // The seed draws a whole recipe, not a single parameter. One hand-picked
  // rule only ever explores the shapes its author already had in mind; a
  // family sampled at random covers combinations nobody would think to write
  // down, and the estimate is there to throw the bad ones away.
  //
  // What every ingredient has in common is that it moves the *block
  // partition*. Reordering operations inside a block leaves the estimate
  // identical -- a block's cost is a roofline over a set of per-unit totals,
  // and a set has no order -- so a knob that only shuffles within blocks
  // produces a thousand candidates with one score.
  SplitMix64 recipe(seed * 0x9e3779b97f4a7c15ULL + 0x243f6a8885a308d3ULL);

  // What counts as urgent. Longest chain first is the textbook rule; shortest
  // first clears small work early and delays the critical path, which is
  // usually worse and occasionally exactly right; program order stays near the
  // input; random ignores structure altogether.
  const unsigned priorityMode = static_cast<unsigned>(recipe.next() % 4);
  // How long one core may run before the other gets a turn. Short runs
  // interleave the cores finely -- more barriers, more overlap; long runs do
  // the opposite, and at the extreme serialise the kernel.
  const size_t targetRun = 1 + static_cast<size_t>(recipe.next() % 32);
  // Whether to respect runs at all. A quarter of the recipes ignore the core
  // and let the two interleave freely; most of those end up rejected by the
  // block-count guard, and the ones that survive are shapes the run-based
  // rules cannot reach.
  const bool stickyCore = (recipe.next() % 4) != 0;
  // How readily a step passes over its best candidate.
  const unsigned noise = static_cast<unsigned>(recipe.next() % 3);

  // Fixed per operation so that a random priority is still a consistent
  // ordering rather than a coin flip at every comparison.
  DenseMap<Operation *, uint64_t> shuffleKey;
  if (priorityMode == 3) {
    for (Operation *op : graph.ops) {
      SplitMix64 keyRng(seed ^ (graph.opIndex.lookup(op) * 0x100000001b3ULL));
      shuffleKey[op] = keyRng.next();
    }
  }

  auto priorityOf = [&](Operation *op) -> uint64_t {
    switch (priorityMode) {
    case 0:
      return height.lookup(op);
    case 1:
      return ~static_cast<uint64_t>(height.lookup(op));
    case 2:
      return ~static_cast<uint64_t>(graph.opIndex.lookup(op));
    default:
      return shuffleKey.lookup(op);
    }
  };

  SplitMix64 rng(seed * 0x2545f4914f6cdd1dULL + 1);
  SmallVector<Operation *> order;
  order.reserve(graph.ops.size());
  CoreType currentCore = CoreType::UNDETERMINED;
  size_t opsInRun = 0;
  while (!ready.empty()) {
    // Staying on the core that just ran comes before everything else. Without
    // it a height-ordered schedule interleaves Cube and Vector work operation
    // by operation -- they are independent, so nothing stops it -- and since a
    // block cannot span two cores, the grouping rebuilt afterwards ends up
    // with a couple of operations per block. That is not a variant of the
    // input, it is a different and much worse shape, and the pipeline rejects
    // it. Runs of one core keep the result recognisable: the blocks stay about
    // as large as the ones that arrived, and what varies is which operations
    // land in each run and in what order.
    //
    // Below that: longest remaining chain first, then program order, so seed 0
    // is reproducible and starts from what the input already said.
    // Keep the run going until it has reached the length this seed asks for,
    // then hand over to the other core if anything there is ready. When only
    // one core has work the preference costs nothing: everything compares
    // equal on it and the ordering falls through to the rules below.
    // +1 keep this core going, -1 hand over to the other, 0 do not care.
    const int corePref = !stickyCore ? 0 : (opsInRun >= targetRun ? -1 : 1);
    llvm::sort(ready, [&](Operation *lhs, Operation *rhs) {
      if (corePref != 0) {
        const bool lhsRuns = continuesRun(lhs, currentCore);
        const bool rhsRuns = continuesRun(rhs, currentCore);
        if (lhsRuns != rhsRuns) {
          return corePref > 0 ? lhsRuns : rhsRuns;
        }
      }
      const uint64_t lhsPriority = priorityOf(lhs);
      const uint64_t rhsPriority = priorityOf(rhs);
      if (lhsPriority != rhsPriority) {
        return lhsPriority > rhsPriority;
      }
      return graph.opIndex.lookup(lhs) < graph.opIndex.lookup(rhs);
    });

    size_t pick = pickReadyIndex(rng, ready.size());
    for (unsigned extra = 0; extra < noise; ++extra) {
      pick = std::max(pick, pickReadyIndex(rng, ready.size()));
    }
    Operation *chosen = ready[pick];
    ready.erase(ready.begin() + pick);
    order.push_back(chosen);

    const CoreType chosenCore = core.lookup(chosen);
    if (chosenCore != CoreType::UNDETERMINED && chosenCore != currentCore) {
      currentCore = chosenCore;
      opsInRun = 1;
    } else {
      ++opsInRun;
    }

    for (Operation *succ : graph.succs.at(chosen)) {
      if (--inDeg[succ] == 0) {
        ready.push_back(succ);
      }
    }
  }
  return order;
}

/// Rebuild the block grouping to match a freshly permuted operation order.
///
/// This is not optional. InterCoreTransferAndSync locates a block by walking
/// linearly from its first operation and stopping at the first foreign id
/// (getBlockStartEnd, used at eight call sites there), so a block whose
/// operations are no longer adjacent would have its synchronisation flags
/// placed around the wrong range -- which on hardware is a kernel that waits
/// on a flag nobody sets. After an operation-level permutation the original
/// grouping no longer holds, so a new one is derived from the result: a block
/// is a maximal run of operations belonging to the same core.
///
/// The core assignment itself is untouched. That is the decision OpClassifier
/// made with alias analysis and it is not ours to revisit; only the boundaries
/// between blocks move.
void rederiveBlockIds(ArrayRef<Operation *> order, ComputeBlockIdManager &bm,
                      DenseMap<Operation *, int> &opBlockId) {
  int currentId = -1;
  CoreType currentCore = CoreType::UNDETERMINED;
  for (Operation *op : order) {
    const CoreType core = CVPipeline::getOpCoreType(op);
    // An operation that names no core -- a constant, an scf container -- joins
    // whatever run it landed in rather than cutting it in two.
    const bool startsNewBlock =
        currentId < 0 ||
        (core != CoreType::UNDETERMINED && core != currentCore);
    if (startsNewBlock) {
      currentId = bm.getNextId();
      currentCore = core;
    }
    bm.updateBlockId(op, currentId);
    opBlockId[op] = currentId;

    // Operations nested inside this one carry ids of their own, stamped by
    // PlanCubeBlock and PlanVectorBlock. Leaving those on the old id would
    // make the module disagree with itself: a later pass that walks every
    // operation rather than just the block-level ones would see two groupings
    // at once and take whichever it reached first.
    //
    // But this must not reach into an scf region. Those regions hold blocks of
    // their own, which this pass reorders and regroups separately, and the
    // walk over them runs innermost first -- so descending here would flatten
    // a loop body that was already grouped into a single id and take its core
    // assignment with it. Only regions nobody else regroups, such as a linalg
    // body, follow their parent.
    if (!CVPipeline::isScfOp(op)) {
      op->walk([&](Operation *nested) {
        if (nested != op && CVPipeline::getOpBlockId(nested).has_value()) {
          bm.updateBlockId(nested, currentId);
        }
      });
    }
  }
}

/// How many blocks the run-based regrouping above will produce, worked out
/// without touching the IR. Reported next to the count that arrived, so a run
/// shows how far each variant moved the partition -- the quantity the three
/// resource guards in the search are all spending.
///
/// Must mirror rederiveBlockIds exactly; the two are read together.
size_t countRunsByCore(ArrayRef<Operation *> order) {
  size_t runs = 0;
  bool started = false;
  CoreType currentCore = CoreType::UNDETERMINED;
  for (Operation *op : order) {
    const CoreType core = CVPipeline::getOpCoreType(op);
    if (!started || (core != CoreType::UNDETERMINED && core != currentCore)) {
      ++runs;
      started = true;
      currentCore = core;
    }
  }
  return runs;
}

/// How many blocks a rederived order ended up with, for the one-line report.
size_t countDistinctBlockIds(ArrayRef<Operation *> order,
                             const DenseMap<Operation *, int> &opBlockId) {
  DenseSet<int> seen;
  for (Operation *op : order) {
    auto it = opBlockId.find(op);
    if (it != opBlockId.end()) {
      seen.insert(it->second);
    }
  }
  return seen.size();
}

} // namespace

// Stable sort ops based on their group orders
static llvm::FailureOr<SmallVector<Operation *>> buildReorderedOps(
    const BlockOpGraph &graph, const DenseMap<Operation *, int> &opBlockId,
    ComputeBlockIdManager &bm, const MemoryDependenceGraph &memGraph) {
  SmallVector<Operation *> reordered;
  GroupAdjacencyGraph adjacencyGraph{graph, opBlockId, bm};

  // computeTopologicalOrder() consumes inDeg, so keep a copy for the report.
  const SmallVector<unsigned> inDegSnapshot = adjacencyGraph.inDeg;

  auto groupOrderResult = adjacencyGraph.computeTopologicalOrder();
  if (llvm::failed(groupOrderResult)) {
    return llvm::failure();
  }

  if (getReorderVerbosity() >= 1) {
    reportGroupGraph(adjacencyGraph, graph, opBlockId, inDegSnapshot,
                     groupOrderResult.value());
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
                  ComputeBlockIdManager &bm,
                  std::optional<uint64_t> variantSeed) {
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

  // Variant mode: order the operations themselves and rebuild the blocks
  // around the result. The block grouping that arrived here is deliberately
  // not consulted -- it is the thing being varied.
  if (variantSeed) {
    SmallVector<Operation *> order = sampleOpOrder(graph, *variantSeed);
    // How far the regrouping moved, for the report. There used to be a limit on
    // it, and it is gone: what it stood for -- synchronisation flags, Unified
    // Buffer, software-pipeline depth -- the search now measures directly and
    // per resource, against the untouched pipeline. A cap on the block count
    // was a proxy for all three and a poor one: it removed most of the search
    // space to guard against costs that are now counted, and being decided per
    // scope it could reject one loop nest while accepting another, leaving a
    // half-applied variant that nobody designed.
    //
    // What still stands guard is rederiveBlockIds below, which is about
    // correctness rather than aggressiveness: InterCoreTransferAndSync locates
    // a block by walking from its first operation to the first foreign id, so
    // a block whose operations are not adjacent puts its flags around the wrong
    // range and hangs the kernel.
    const size_t before = countDistinctBlockIds(allOps, opBlockId);
    const size_t after = countRunsByCore(order);
    if (order.size() != allOps.size()) {
      // A cycle in the dependency graph; the block-level path reports this
      // properly, so fall through to it rather than emitting a partial order.
      LOG_DEBUG("op-level order incomplete, falling back to block order\n");
    } else {
      rederiveBlockIds(order, bm, opBlockId);
      applyReorder(block, order);
      if (getReorderVerbosity() >= 1) {
        llvm::errs() << "[reorder-blocks] variant seed " << *variantSeed
                     << ": " << order.size() << " op(s) reordered, " << before
                     << " block(s) became " << after << " in "
                     << describeScope(&block) << "\n";
      }
      return llvm::success();
    }
  }

  const auto reorderedRes = buildReorderedOps(graph, opBlockId, bm, memGraph);
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
  LOG_DEBUG("\n=== Pass: TuningOpSeq ===\n");
  // Not const: the attribute builders below are non-const members.
  OpBuilder builder(&getContext());

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

  const std::optional<uint64_t> variantSeed = getVariantSeed(moduleOp);

  auto bm = ComputeBlockIdManager(moduleOp);
  auto result = moduleOp.walk([&](Block *block) {
    auto *parentOp = block->getParentOp();
    if (!parentOp ||
        // whitelist ops to reorder
        !(isa<func::FuncOp>(parentOp) ||
          isa<scf::SCFDialect>(parentOp->getDialect()))) {
      return WalkResult::skip();
    }
    if (llvm::failed(reorderOpsInBlock(*block, memGraph, bm, variantSeed))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Mark the seed as spent, so the second run of this pass -- at the end of
  // SplitDataflow, once transfers and sync flags refer to block ids -- takes
  // the ordinary block-level path instead of regrouping underneath them.
  //
  // Marking it destroys the value, so keep a copy first. Everything that wants
  // to know afterwards which ordering it is looking at -- the cost report, and
  // through it the seed table -- would otherwise have to re-derive the seed
  // from an attribute that now reads -1 for every variant alike. This runs
  // only on the first of the two passes, since by the second variantSeed is
  // already empty, so the record is written exactly once.
  if (variantSeed) {
    moduleOp->setAttr(CVPipeline::kReorderSeedApplied,
                      builder.getI64IntegerAttr(
                          static_cast<int64_t>(*variantSeed)));
    moduleOp->setAttr(CVPipeline::kReorderSeed, builder.getI64IntegerAttr(-1));
  }

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
