/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */

#include "ascend/include/DynamicCVPipeline/Common/DependencyHelper.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "merge-outer-vector-block"

using namespace mlir;
using namespace mlir::CVPipeline;

namespace {

// Keep unknown effects, including unknown effects nested in a region, as
// source-order fences. Known memory effects are ordered by the alias-aware
// MemoryDependenceGraph, while explicit synchronizations always remain fences.
static bool containsFence(Operation *root) {
  auto effects = getEffectsRecursively(root);
  if (!effects || llvm::any_of(*effects, [](const auto &effect) {
        return !effect.getValue();
      }))
    return true;
  return root
      ->walk([&](Operation *op) {
        if (isSyncOp(op))
          return WalkResult::interrupt();
        if (isa<MemoryEffectOpInterface>(op) || isMemoryEffectFree(op) ||
            op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
          return WalkResult::advance();
        return WalkResult::interrupt();
      })
      .wasInterrupted();
}

static bool isVectorBlockRoot(Operation *root) {
  if (getCoreTypeOfSimpleOpOrCf(root) == CoreType::VECTOR_ONLY)
    return true;

  // SplitIfByBlockId creates bypass placeholders without core_type.
  // A store guard has only nested labels, while a seed select has only
  // result labels. Check both the results and the region when present.
  auto guard = dyn_cast<scf::IfOp>(root);
  if (!guard || !llvm::all_of(guard.getResults(), [](Value result) {
        return getValueCoreType(result) == CoreType::VECTOR_ONLY;
      }))
    return false;

  return !guard
              ->walk<WalkOrder::PreOrder>([&](Operation *op) {
                if (op->hasTrait<OpTrait::IsTerminator>())
                  return WalkResult::skip();
                if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                  if (getOpCoreType(ifOp) == CoreType::CUBE_ONLY)
                    return WalkResult::interrupt();
                  if (!llvm::all_of(ifOp.getResults(), [](Value result) {
                        return getValueCoreType(result) ==
                                   CoreType::VECTOR_ONLY ||
                               result.getType().isIntOrIndex();
                      }))
                    return WalkResult::interrupt();
                  return WalkResult::advance();
                }
                auto scalarOrAddress = [](Type type) {
                  return type.isIntOrIndex() || isa<MemRefType>(type);
                };
                // Task-local metadata selection can be nested in a VECTOR
                // payload. Its scalar reads retain the alias graph's ordering
                // constraints.
                if (getOpCoreType(op) != CoreType::CUBE_ONLY &&
                    llvm::all_of(op->getOperandTypes(), scalarOrAddress) &&
                    llvm::all_of(op->getResultTypes(), scalarOrAddress) &&
                    (isa<memref::LoadOp>(op) || isMemoryEffectFree(op)))
                  return WalkResult::skip();
                if (op->hasAttr(kCoreType))
                  return getOpCoreType(op) == CoreType::VECTOR_ONLY
                             ? WalkResult::skip()
                             : WalkResult::interrupt();
                bool placeholder = isa<arith::ConstantOp, tensor::EmptyOp>(op);
                if (auto fill = dyn_cast<linalg::FillOp>(op))
                  placeholder = fill.hasPureTensorSemantics();
                return placeholder && isMemoryEffectFree(op)
                           ? WalkResult::skip()
                           : WalkResult::interrupt();
              })
              .wasInterrupted();
}

// Keep a task's recurrence reset at the stage that reads and updates it.
// Forwarding the previous value from the entry stage consumes a ring slot
// and ties the next task's preparation to the previous task's final stage.
static void sinkEventRecurrenceSeeds(scf::ForOp loop) {
  Block *body = loop.getBody();
  for (auto entry : llvm::make_early_inc_range(body->getOps<scf::IfOp>())) {
    Operation *condition = entry.getCondition().getDefiningOp();
    auto entryId = getOpBlockId(entry);
    if (!condition || !condition->hasAttr(kOuterEventControl) || !entryId ||
        entry.getElseRegion().empty())
      continue;
    for (auto [index, result] : llvm::enumerate(entry.getResults())) {
      if (result.use_empty() ||
          getValueCoreType(result) != CoreType::VECTOR_ONLY ||
          !isa<RankedTensorType>(result.getType()))
        continue;
      Value seed = entry.thenYield().getOperand(index);
      auto old = dyn_cast<BlockArgument>(entry.elseYield().getOperand(index));
      if (!old || old.getOwner() != body || old.getArgNumber() == 0 ||
          loop.getRegion().isAncestor(seed.getParentRegion()))
        continue;
      Operation *writer = getLoopCarriedDefOp(old, body);
      auto writerId = writer ? getOpBlockId(writer) : std::nullopt;
      if (!writerId || writerId == entryId || !isVectorBlockRoot(writer))
        continue;
      Operation *first = nullptr;
      for (Operation *user : result.getUsers()) {
        Operation *root = body->findAncestorOpInBlock(*user);
        if (!root || root->hasTrait<OpTrait::IsTerminator>() ||
            getOpBlockId(root) != writerId) {
          first = nullptr;
          break;
        }
        if (!first || root->isBeforeInBlock(first))
          first = root;
      }
      if (!first)
        continue;

      OpBuilder b(first);
      auto select = b.create<scf::IfOp>(
          entry.getLoc(), TypeRange{seed.getType()}, entry.getCondition(),
          /*withElseRegion=*/true);
      auto setStage = [&](Operation *op) {
        op->setAttr(kBlockId, b.getI32IntegerAttr(*writerId));
        op->setAttr(kCoreType, b.getStringAttr(kCoreTypeVector));
      };
      setStage(select);
      OpBuilder thenBuilder = select.getThenBodyBuilder();
      Value taskSeed = seed;
      if (auto fill = seed.getDefiningOp<linalg::FillOp>()) {
        auto empty = fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
        if (fill.hasPureTensorSemantics() && empty &&
            empty.getDynamicSizes().empty() &&
            fill.getDpsInputs()[0].getDefiningOp<arith::ConstantOp>()) {
          // A private fill in the first-event branch need not occupy UB for
          // the entire outer schedule. The original seed stays available to
          // the unscheduled fallback and to any other core that uses it.
          IRMapping mapping;
          setStage(thenBuilder.clone(*empty, mapping));
          Operation *localFill = thenBuilder.clone(*fill, mapping);
          setStage(localFill);
          taskSeed = localFill->getResult(0);
        }
      }
      setStage(thenBuilder.create<scf::YieldOp>(entry.getLoc(), taskSeed));
      setStage(select.getElseBodyBuilder().create<scf::YieldOp>(entry.getLoc(),
                                                                old));
      result.replaceAllUsesWith(select.getResult(0));
      // The dead entry result must no longer read the previous recurrence.
      entry.elseYield()->setOperand(index, seed);
    }
  }
}

// Moving an arbitrary private computation to Cube is not valid. Recognize
// only the GM load form emitted here: optional zero padding, one copy into
// a private allocation, and its tensor view. Memory spaces have not yet been
// assigned, so only unannotated function arguments establish a GM source.
static bool isTaskLocalGMLoad(scf::IfOp entry,
                              bufferization::ToTensorOp source) {
  Block *thenBlock = &entry.getThenRegion().front();
  auto alloc = source.getBuffer().getDefiningOp<memref::AllocOp>();
  if (!alloc || alloc->getBlock() != thenBlock ||
      source->getBlock() != thenBlock || alloc.getType().getMemorySpace())
    return false;

  auto base = [](Value value) {
    while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
      value = view.getViewSource();
    return value;
  };
  auto fromGM = [&](Value value) {
    auto type = dyn_cast<MemRefType>(value.getType());
    auto arg = dyn_cast<BlockArgument>(base(value));
    auto argType = arg ? dyn_cast<MemRefType>(arg.getType()) : MemRefType{};
    return type && !type.getMemorySpace() && argType &&
           arg.getOwner()->isEntryBlock() &&
           isa<func::FuncOp>(arg.getOwner()->getParentOp()) &&
           !argType.getMemorySpace();
  };
  memref::CopyOp copy;
  linalg::FillOp padding;
  auto result =
      entry.getThenRegion().walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op == alloc.getOperation() || op == source.getOperation() ||
            isa<scf::YieldOp>(op))
          return WalkResult::skip();
        if (auto load = dyn_cast<memref::CopyOp>(op)) {
          if (copy || load->getBlock() != thenBlock ||
              base(load.getTarget()) != alloc.getResult() ||
              !fromGM(load.getSource()))
            return WalkResult::interrupt();
          copy = load;
          return WalkResult::skip();
        }
        // Task-tensor slicing can leave integer metadata reads in the entry
        // guard. They may move with the load; local and volatile reads may not.
        if (auto load = dyn_cast<memref::LoadOp>(op))
          return load.getType().isIntOrIndex() && fromGM(load.getMemRef()) &&
                         !load->hasAttr(triton::kMemrefExtVolatile) &&
                         llvm::none_of(load.getResult().getUsers(),
                                       [](Operation *user) {
                                         return user->hasAttr(
                                             triton::kMemrefExtVolatile);
                                       })
                     ? WalkResult::skip()
                     : WalkResult::interrupt();
        if (auto fill = dyn_cast<linalg::FillOp>(op)) {
          Value value = fill.getDpsInputs()[0];
          if (padding || fill->getNumResults() != 0 ||
              fill.getDpsInits()[0] != alloc.getResult() ||
              (!matchPattern(value, m_Zero()) &&
               !matchPattern(value, m_AnyZeroFloat())))
            return WalkResult::interrupt();
          padding = fill;
          return WalkResult::skip();
        }
        if (auto guard = dyn_cast<scf::IfOp>(op))
          return guard.getNumResults() == 0 && guard.getElseRegion().empty()
                     ? WalkResult::advance()
                     : WalkResult::interrupt();
        if (isa<memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp>(
                op)) {
          Value value = op->getResult(0);
          return base(value) == alloc.getResult() || fromGM(value)
                     ? WalkResult::skip()
                     : WalkResult::interrupt();
        }
        auto scalar = [](Type type) { return type.isIntOrIndex(); };
        if (auto constant = dyn_cast<arith::ConstantOp>(op))
          return constant.getType().isIntOrIndexOrFloat()
                     ? WalkResult::skip()
                     : WalkResult::interrupt();
        return isa<arith::ArithDialect>(op->getDialect()) &&
                       llvm::all_of(op->getOperandTypes(), scalar) &&
                       llvm::all_of(op->getResultTypes(), scalar)
                   ? WalkResult::skip()
                   : WalkResult::interrupt();
      });
  if (result.wasInterrupted() || !copy || !copy->isBeforeInBlock(source))
    return false;
  return !padding ||
         thenBlock->findAncestorOpInBlock(*padding)->isBeforeInBlock(copy);
}

// A task-local load with only one Cube matmul user can keep its carry on
// Cube. This removes a Vector-to-Cube transfer on every event while retaining
// the original first-event load and reuse for subsequent events.
static void keepTaskLoadWithMatmul(scf::ForOp loop) {
  auto types = loop->getAttrOfType<StringAttr>(kCoreType);
  if (!types)
    return;
  SmallVector<StringRef> coreTypes;
  types.getValue().split(coreTypes, ", ");
  if (coreTypes.size() != loop.getNumResults())
    return;
  Block *body = loop.getBody();
  auto loopYield = cast<scf::YieldOp>(body->getTerminator());
  for (auto entry : llvm::make_early_inc_range(body->getOps<scf::IfOp>())) {
    Operation *cond = entry.getCondition().getDefiningOp();
    if (!cond || !cond->hasAttr(kOuterEventControl) ||
        entry.getElseRegion().empty())
      continue;
    Value live;
    for (Value result : entry.getResults()) {
      if (result.use_empty())
        continue;
      if (live) {
        live = nullptr;
        break;
      }
      live = result;
    }
    if (!live || getValueCoreType(live) != CoreType::VECTOR_ONLY)
      continue;
    unsigned resultIndex = cast<OpResult>(live).getResultNumber();
    auto old =
        dyn_cast<BlockArgument>(entry.elseYield().getOperand(resultIndex));
    auto source = entry.thenYield()
                      .getOperand(resultIndex)
                      .getDefiningOp<bufferization::ToTensorOp>();
    if (!old || old.getOwner() != body || old.getArgNumber() == 0 ||
        !old.hasOneUse() || !source ||
        !entry->isProperAncestor(source.getOperation()) ||
        !isTaskLocalGMLoad(entry, source))
      continue;
    unsigned carryIndex = old.getArgNumber() - 1;
    if (loopYield.getOperand(carryIndex) != live)
      continue;
    Operation *consumer = nullptr;
    for (OpOperand &use : live.getUses()) {
      if (use.getOwner() == loopYield.getOperation())
        continue;
      auto matmul = dyn_cast<linalg::MatmulOp>(use.getOwner());
      if (consumer || !matmul || use.getOperandNumber() >= 2 ||
          getOpCoreType(matmul) != CoreType::CUBE_ONLY) {
        consumer = nullptr;
        break;
      }
      consumer = body->findAncestorOpInBlock(*matmul);
    }
    auto target = consumer ? getOpBlockId(consumer) : std::nullopt;
    if (!target || *target <= 0 || !entry->isBeforeInBlock(consumer) ||
        containsFence(entry))
      continue;
    auto effects = getEffectsRecursively(entry);
    if (!effects || !llvm::all_of(*effects, [&](const auto &effect) {
          if (isa<MemoryEffects::Read>(effect.getEffect()))
            return true;
          Value storage = effect.getValue();
          while (auto view = storage.getDefiningOp<ViewLikeOpInterface>())
            storage = view.getViewSource();
          auto alloc = storage.getDefiningOp<memref::AllocOp>();
          return alloc && entry->isProperAncestor(alloc) &&
                 isa<MemoryEffects::Allocate, MemoryEffects::Write>(
                     effect.getEffect());
        }))
      continue;
    // Nothing may capture storage written by this prologue from outside
    // the original guard. The SSA result carries the tensor value.
    if (!llvm::all_of(entry.getElseRegion().front().without_terminator(),
                      [](Operation &op) { return isMemoryEffectFree(&op); }))
      continue;

    OpBuilder b(entry);
    auto local = b.create<scf::IfOp>(entry.getLoc(), TypeRange{live.getType()},
                                     entry.getCondition(),
                                     /*withElseRegion=*/true);
    auto tag = [&](Operation *op) {
      op->setAttr(kBlockId, b.getI32IntegerAttr(*target));
      op->setAttr(kCoreType, b.getStringAttr(kCoreTypeCube));
    };
    tag(local);
    OpBuilder thenBuilder = local.getThenBodyBuilder();
    IRMapping map;
    for (Operation &op : entry.getThenRegion().front().without_terminator()) {
      Operation *cloned = thenBuilder.clone(op, map);
      cloned->walk([&](Operation *nested) {
        if (nested == cloned || nested->hasAttr(kBlockId))
          tag(nested);
      });
    }
    tag(thenBuilder.create<scf::YieldOp>(entry.getLoc(),
                                         map.lookup(source.getResult())));
    tag(local.getElseBodyBuilder().create<scf::YieldOp>(entry.getLoc(), old));
    live.replaceAllUsesWith(local.getResult(0));
    coreTypes[carryIndex] = kCoreTypeCube;
    auto updated = b.getStringAttr(llvm::join(coreTypes, ", "));
    loop->setAttr(kCoreType, updated);
    loopYield->setAttr(kCoreType, updated);
    entry.erase();
  }
}

struct Group {
  int id;
  SmallVector<Operation *> roots;
  bool vector = true;
};

struct Edge {
  unsigned source;
  unsigned target;
  uint64_t elements;
};

struct Graph {
  SmallVector<Group> groups;
  DenseMap<Operation *, unsigned> node;
  SmallVector<SmallVector<unsigned>> successors;

  void addEdge(unsigned source, unsigned target) {
    if (source != target && !llvm::is_contained(successors[source], target))
      successors[source].push_back(target);
  }

  // Validate the entire contracted graph, including untagged scalar and
  // memory operations. A path through another group must not be collapsed
  // into a cycle (e.g. VECTOR -> CUBE -> VECTOR).
  SmallVector<unsigned> topologicalOrder(unsigned source,
                                         unsigned target) const {
    auto representative = [&](unsigned n) { return n == source ? target : n; };
    SmallVector<SmallVector<unsigned>> edges(groups.size());
    SmallVector<unsigned> indegree(groups.size(), 0);
    for (unsigned i = 0; i < groups.size(); ++i) {
      for (unsigned next : successors[i]) {
        unsigned from = representative(i), to = representative(next);
        if (from != to && !llvm::is_contained(edges[from], to)) {
          edges[from].push_back(to);
          ++indegree[to];
        }
      }
    }
    SmallVector<unsigned> ready;
    for (unsigned i = 0; i < groups.size(); ++i)
      if (indegree[i] == 0)
        ready.push_back(i);
    SmallVector<unsigned> order;
    while (!ready.empty()) {
      // Prefer the earliest original group when multiple groups are ready.
      auto first = llvm::min_element(ready);
      unsigned current = *first;
      ready.erase(first);
      order.push_back(current);
      for (unsigned next : edges[current])
        if (--indegree[next] == 0)
          ready.push_back(next);
    }
    return order;
  }

  bool acyclic(unsigned source, unsigned target) const {
    return topologicalOrder(source, target).size() == groups.size();
  }
};

static Graph buildGraph(Operation *loop, const MemoryDependenceGraph &memory) {
  Graph graph;
  Block *body = MainLoop(loop).getBody();
  DenseMap<int, unsigned> taggedNodes;
  SmallVector<Operation *> roots;
  for (Operation &op : body->without_terminator()) {
    auto id = getOpBlockId(&op);
    unsigned n = graph.groups.size();
    if (id && *id > 0 && taggedNodes.contains(*id)) {
      n = taggedNodes.lookup(*id);
    } else {
      graph.groups.push_back({id.value_or(-1), {}, true});
      if (id && *id > 0)
        taggedNodes[*id] = n;
    }
    graph.node[&op] = n;
    graph.groups[n].roots.push_back(&op);
    bool fence = containsFence(&op);
    bool vectorRoot = isVectorBlockRoot(&op);
    graph.groups[n].vector &= id && *id > 0 && !fence && vectorRoot;
    // SplitIfByBlockId should have made every region homogeneous. Do not
    // absorb another stage if that prerequisite is not satisfied.
    op.walk([&](Operation *nested) {
      if (auto nestedId = getOpBlockId(nested); nestedId != id && nestedId) {
        graph.groups[n].vector = false;
        LLVM_DEBUG(llvm::dbgs()
                   << "Mixed IDs " << id.value_or(-1) << " and " << *nestedId
                   << " at " << nested->getName() << "\n");
      }
    });
    LLVM_DEBUG(if (id && *id > 0 && (fence || !vectorRoot)) {
      llvm::dbgs() << "Nonmergeable root " << *id << " " << op.getName()
                   << " fence=" << fence << " vector=" << vectorRoot << "\n";
    });
    roots.push_back(&op);
  }
  graph.successors.resize(graph.groups.size());
  DependencyHelper helper(memory);
  for (Operation *root : roots) {
    helper.forEachSource(root, [&](Operation *source) {
      Operation *ancestor = body->findAncestorOpInBlock(*source);
      if (ancestor && graph.node.contains(ancestor))
        graph.addEdge(graph.node.lookup(ancestor), graph.node.lookup(root));
    });
  }
  for (unsigned i = 0; i < roots.size(); ++i) {
    if (!containsFence(roots[i]))
      continue;
    unsigned fence = graph.node.lookup(roots[i]);
    for (unsigned j = 0; j < i; ++j)
      graph.addEdge(graph.node.lookup(roots[j]), fence);
    for (unsigned j = i + 1; j < roots.size(); ++j)
      graph.addEdge(fence, graph.node.lookup(roots[j]));
  }
  if (loop->hasAttr(kOuterEventControlCount)) {
    // A normalized event carry still holds the previous event's tensor.
    // Keep every old-value reader before its update, both when validating a
    // contraction and when ordering the resulting groups. Otherwise a path
    // through a non-VECTOR reader can be hidden by an apparently direct
    // VECTOR producer/consumer edge.
    for (Value arg : MainLoop(loop).getIterArgs()) {
      if (!isa<RankedTensorType>(arg.getType()))
        continue;
      Operation *writer = getLoopCarriedDefOp(arg, body);
      if (!writer || !graph.node.contains(writer))
        continue;
      for (Operation *user : arg.getUsers()) {
        Operation *reader = body->findAncestorOpInBlock(*user);
        if (reader && graph.node.contains(reader))
          graph.addEdge(graph.node.lookup(reader), graph.node.lookup(writer));
      }
    }
  }
  return graph;
}

static SmallVector<Edge> collectCandidates(scf::ForOp loop,
                                           const Graph &graph) {
  SmallVector<Edge> candidates;
  for (unsigned source = 0; source < graph.groups.size(); ++source) {
    if (!graph.groups[source].vector)
      continue;
    for (Operation *root : graph.groups[source].roots) {
      for (Value result : root->getResults()) {
        auto type = dyn_cast<RankedTensorType>(result.getType());
        if (!type || !type.hasStaticShape() || type.getNumElements() <= 0)
          continue;
        // Prefer removing the largest transferred tensor. No kernel shape
        // or arithmetic pattern is required by this policy.
        uint64_t elements = type.getNumElements();
        llvm::SmallSet<unsigned, 4> seen;
        for (Operation *user : result.getUsers()) {
          Operation *ancestor = loop.getBody()->findAncestorOpInBlock(*user);
          if (!ancestor || !graph.node.contains(ancestor))
            continue;
          unsigned target = graph.node.lookup(ancestor);
          if (source == target || !graph.groups[target].vector ||
              !seen.insert(target).second)
            continue;
          candidates.push_back({source, target, elements});
        }
      }
    }
  }
  llvm::stable_sort(candidates, [](const Edge &a, const Edge &b) {
    return a.elements > b.elements;
  });
  return candidates;
}

static void mergeLoop(scf::ForOp loop, const MemoryDependenceGraph &memory,
                      ComputeBlockIdManager &manager) {
  // Ordinary loop-carried domains remain outside this experiment. Explicit
  // event domains add their old-carry anti-dependencies to the complete DAG.
  if (loop.getNumRegionIterArgs() != 0 &&
      !loop->hasAttr(kOuterEventControlCount))
    return;
  if (loop->walk([&](Operation *op) {
            return op != loop.getOperation() && isa<LoopLikeOpInterface>(op)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted())
    return;

  bool changed = false;
  while (true) {
    Graph graph = buildGraph(loop, memory);
    LLVM_DEBUG(for (unsigned i = 0; i < graph.groups.size(); ++i) {
      llvm::dbgs() << "Group " << graph.groups[i].id
                   << " vector=" << graph.groups[i].vector << " ->";
      for (unsigned next : graph.successors[i])
        llvm::dbgs() << " " << graph.groups[next].id;
      llvm::dbgs() << "\n";
    });
    if (graph.groups.empty() || !graph.acyclic(0, 0))
      return;
    bool merged = false;
    for (const Edge &edge : collectCandidates(loop, graph)) {
      if (!graph.acyclic(edge.source, edge.target))
        continue;
      int source = graph.groups[edge.source].id;
      int target = graph.groups[edge.target].id;
      auto sourceOps = manager.getOpsByBlockId(source);
      auto targetOps = manager.getOpsByBlockId(target);
      // Block ids are normally globally unique; do not rely on that for
      // malformed or hand-written IR with an id reused outside this loop.
      auto outsideLoop = [&](Operation *op) { return !loop->isAncestor(op); };
      if (llvm::any_of(sourceOps, outsideLoop) ||
          llvm::any_of(targetOps, outsideLoop))
        continue;
      for (Operation *op : sourceOps)
        manager.updateBlockId(op, target);
      merged = true;
      changed = true;
      break;
    }
    if (!merged) {
      // Only reorder the selected body, and only after an actual merge.
      // Keep each group's original operation order and every region intact.
      // The existing general reorder pass visits unrelated loops as well.
      if (changed)
        for (unsigned node : graph.topologicalOrder(0, 0))
          for (Operation *op : graph.groups[node].roots)
            op->moveBefore(loop.getBody()->getTerminator());
      return;
    }
  }
}

class MergeOuterVectorBlockPass
    : public PassWrapper<MergeOuterVectorBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeOuterVectorBlockPass)
  StringRef getArgument() const override { return "merge-outer-vector-block"; }
  StringRef getDescription() const override {
    return "Contract acyclic VECTOR tensor dependencies in explicit outer "
           "loops";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (hasFallbackAttr(module))
      return;
    module.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kExplicitMainLoop) &&
          loop->hasAttr(kOuterEventControlCount)) {
        sinkEventRecurrenceSeeds(loop);
        keepTaskLoadWithMatmul(loop);
      }
    });
    auto &aliases = getAnalysis<AliasAnalysis>();
    MemoryDependenceGraph memory(module, aliases);
    ComputeBlockIdManager manager(module);
    module.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kExplicitMainLoop))
        mergeLoop(loop, memory, manager);
    });
  }
};

} // namespace

// Use the same dependency graph for contraction and final carry ordering.
// Unknown effects and syncs are fences; every old-carry reader precedes its
// writer. Untagged roots remain independent groups with their original order.
void mlir::triton::ReorderOuterEventCarryPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (hasFallbackAttr(module))
    return;
  auto &aliases = getAnalysis<AliasAnalysis>();
  MemoryDependenceGraph memory(module, aliases);
  auto result = module.walk([&](Operation *loop) {
    if (!loop->hasAttr(kOuterEventControlCount))
      return WalkResult::advance();
    if (!isa<scf::ForOp, scf::WhileOp>(loop))
      return WalkResult::interrupt();
    for (Operation &op : MainLoop(loop).getBody()->without_terminator())
      if (failed(verifyOpBlockId(&op)))
        return WalkResult::interrupt();
    Graph graph = buildGraph(loop, memory);
    auto order = graph.topologicalOrder(0, 0);
    if (order.size() != graph.groups.size()) {
      loop->emitRemark(
          "outer event carry ordering has a dependency cycle; falling back");
      return WalkResult::interrupt();
    }
    for (unsigned node : order)
      for (Operation *op : graph.groups[node].roots)
        op->moveBefore(MainLoop(loop).getBody()->getTerminator());
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    setFallbackAttr(module, ERRCODE_IGNORED);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createReorderOuterEventCarryPass() {
  return std::make_unique<ReorderOuterEventCarryPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createMergeOuterVectorBlockPass() {
  return std::make_unique<MergeOuterVectorBlockPass>();
}
