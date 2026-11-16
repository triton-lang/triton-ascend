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
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "ub-usage-opt-up-stream"

using namespace mlir;

namespace {

// This models boundary transfers, not peak UB allocation. Dynamic shapes and
// unsupported element layouts are counted separately from known byte sizes.
struct BoundaryCost {
  uint64_t unknownTransfers = 0;
  uint64_t staticBytes = 0;

  bool operator<(const BoundaryCost &other) const {
    return std::tie(unknownTransfers, staticBytes) <
           std::tie(other.unknownTransfers, other.staticBytes);
  }

  bool operator==(const BoundaryCost &other) const {
    return std::tie(unknownTransfers, staticBytes) ==
           std::tie(other.unknownTransfers, other.staticBytes);
  }

  void add(const BoundaryCost &other) {
    auto saturatingAdd = [](uint64_t lhs, uint64_t rhs) {
      return lhs + std::min(rhs, std::numeric_limits<uint64_t>::max() - lhs);
    };
    unknownTransfers = saturatingAdd(unknownTransfers, other.unknownTransfers);
    staticBytes = saturatingAdd(staticBytes, other.staticBytes);
  }
};

// Classification is per use: an empty destination differs from a tensor whose
// initialized contents are read by the operation (e.g. a reduction init).
enum class OperandRole { TensorData, NonTensor, EmptyDestination };

static OperandRole classifyOperand(OpOperand &operand) {
  if (!isa<TensorType>(operand.get().getType()))
    return OperandRole::NonTensor;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(operand.getOwner());
  if (linalgOp && linalgOp.isDpsInit(&operand) &&
      !linalgOp.payloadUsesValueFromOperand(&operand) &&
      operand.get().getDefiningOp<tensor::EmptyOp>())
    return OperandRole::EmptyDestination;
  return OperandRole::TensorData;
}

// Repeated uses of one tensor are one dependency. Scalar parameters and
// write-only empty destinations stay in their original groups.
static std::optional<Value> getSingleDataInput(Operation *op) {
  Value input;
  for (OpOperand &operand : op->getOpOperands()) {
    if (classifyOperand(operand) != OperandRole::TensorData)
      continue;
    if (input && input != operand.get())
      return std::nullopt;
    input = operand.get();
  }
  if (!input)
    return std::nullopt;
  return input;
}

static BoundaryCost estimateTensorSize(TensorType type) {
  if (!type.hasStaticShape())
    return {1, 0};
  // An empty tensor stays empty even if an earlier dimension would overflow.
  if (llvm::is_contained(type.getShape(), 0))
    return {};
  Type elementType = type.getElementType();
  if (!elementType.isIntOrFloat())
    return {1, 0};
  uint64_t bytes = std::max<uint64_t>(
      1, (static_cast<uint64_t>(elementType.getIntOrFloatBitWidth()) + 7) / 8);
  for (int64_t dimension : type.getShape()) {
    uint64_t extent = static_cast<uint64_t>(dimension);
    if (bytes > std::numeric_limits<uint64_t>::max() / extent)
      return {0, std::numeric_limits<uint64_t>::max()};
    bytes *= extent;
  }
  return {0, bytes};
}

struct MovePlan {
  int sourceBlockId;
  int targetBlockId;
  SmallVector<Operation *> operations;
  BoundaryCost before;
  BoundaryCost after;
};

struct TensorUse {
  Value value;
  Operation *producerGroupOwner;
  Operation *consumerGroupOwner;
  int64_t unassignedConsumerId;
  BoundaryCost cost;
};

class UpstreamChainOptimizer {
public:
  UpstreamChainOptimizer(Block &body,
                         const CVPipeline::MemoryDependenceGraph &memoryGraph,
                         CVPipeline::ComputeBlockIdManager &blockIds)
      : body(body), memoryGraph(memoryGraph), blockIds(blockIds),
        dependencies(memoryGraph) {
    for (Operation &op : body)
      operations.push_back(&op);
    collectTensorUses();
  }

  void optimize();

private:
  Operation *findGroupOwner(Operation *op) const;
  bool hasSupportedRegion(Operation *op) const;
  bool isVectorConsumer(Operation *op) const;
  bool isMovableProducer(Operation *op) const;
  bool hasPrivateUsers(Operation *op, int targetBlockId,
                       const llvm::SmallPtrSetImpl<Operation *> &chain) const;
  void collectTensorUses();
  BoundaryCost evaluateCost(const llvm::SmallPtrSetImpl<Operation *> &moved,
                            int targetBlockId) const;
  std::optional<MovePlan> findBestPrefix(Operation *producer, int targetBlockId,
                                         BoundaryCost originalCost) const;
  bool applyPlan(const MovePlan &plan);
  bool optimizeOneChain();

  Block &body;
  const CVPipeline::MemoryDependenceGraph &memoryGraph;
  CVPipeline::ComputeBlockIdManager &blockIds;
  CVPipeline::DependencyHelper dependencies;
  SmallVector<Operation *> operations;
  SmallVector<TensorUse> tensorUses;
};

Operation *UpstreamChainOptimizer::findGroupOwner(Operation *op) const {
  for (Operation *current = op; current; current = current->getParentOp())
    if (blockIds.getBlockIdByOp(current) >= 0)
      return current;
  return nullptr;
}

bool UpstreamChainOptimizer::hasSupportedRegion(Operation *op) const {
  if (op->getNumRegions() == 0)
    return true;
  if (!isa<linalg::LinalgOp>(op))
    return false;
  // A Linalg payload is atomic here. Independently grouped descendants or
  // captured shaped values require a different region-aware optimization.
  WalkResult result = op->walk([&](Operation *nested) {
    if (nested == op)
      return WalkResult::advance();
    if (nested->hasAttr(CVPipeline::kBlockId) ||
        llvm::any_of(nested->getOperandTypes(),
                     [](Type type) { return isa<ShapedType>(type); }) ||
        llvm::any_of(nested->getResultTypes(),
                     [](Type type) { return isa<ShapedType>(type); }))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

bool UpstreamChainOptimizer::isVectorConsumer(Operation *op) const {
  return blockIds.getBlockIdByOp(op) >= 0 &&
         CVPipeline::getOpCoreType(op) == CVPipeline::VECTOR_ONLY &&
         !op->hasTrait<OpTrait::IsTerminator>() && hasSupportedRegion(op);
}

bool UpstreamChainOptimizer::isMovableProducer(Operation *op) const {
  if (!isVectorConsumer(op) || !isMemoryEffectFree(op) ||
      !llvm::any_of(op->getResultTypes(),
                    [](Type type) { return isa<TensorType>(type); }))
    return false;
  auto isSupportedType = [](Type type) {
    return isa<TensorType>(type) ||
           isa<IntegerType, IndexType, FloatType, ComplexType>(type);
  };
  if (!llvm::all_of(op->getOperandTypes(), isSupportedType) ||
      !llvm::all_of(op->getResultTypes(), isSupportedType))
    return false;
  // Initialization roots are not tensor data transformations and must not
  // become independent candidates. Their results still count when read.
  return getSingleDataInput(op).has_value();
}

bool UpstreamChainOptimizer::hasPrivateUsers(
    Operation *op, int targetBlockId,
    const llvm::SmallPtrSetImpl<Operation *> &chain) const {
  if (op->use_empty())
    return false;
  // Check actual users. Nested consumers cannot inherit an ancestor's id;
  // terminators pin loop state regardless of their group annotations.
  WalkResult result = dependencies.forEachUser(op, [&](Operation *user) {
    if (user->getBlock() != &body || user->hasTrait<OpTrait::IsTerminator>() ||
        (!chain.contains(user) &&
         (!isVectorConsumer(user) ||
          blockIds.getBlockIdByOp(user) != targetBlockId)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

void UpstreamChainOptimizer::collectTensorUses() {
  DenseMap<Operation *, int64_t> unassignedTargets;
  int64_t nextUnassignedTarget = -1;
  for (Operation *root : operations) {
    root->walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (classifyOperand(operand) != OperandRole::TensorData)
          continue;
        Value value = operand.get();
        Operation *definition = value.getDefiningOp();
        if (definition && root->isAncestor(definition))
          continue;
        if (auto argument = dyn_cast<BlockArgument>(value)) {
          // Nested arguments are local. Loop/function arguments are fixed
          // boundary inputs: never guess while mappings from yield positions.
          if (root->isAncestor(argument.getOwner()->getParentOp()))
            continue;
        }
        Operation *targetOwner = findGroupOwner(op);
        int64_t unassignedId = 0;
        if (!targetOwner) {
          auto inserted =
              unassignedTargets.try_emplace(op, nextUnassignedTarget);
          if (inserted.second)
            --nextUnassignedTarget;
          unassignedId = inserted.first->second;
        }
        tensorUses.push_back(
            {value, findGroupOwner(definition), targetOwner, unassignedId,
             estimateTensorSize(cast<TensorType>(value.getType()))});
      }
    });
  }
}

BoundaryCost UpstreamChainOptimizer::evaluateCost(
    const llvm::SmallPtrSetImpl<Operation *> &moved, int targetBlockId) const {
  auto assignedId = [&](Operation *owner) {
    return moved.contains(owner) ? targetBlockId
                                 : blockIds.getBlockIdByOp(owner);
  };
  DenseSet<std::pair<Value, int64_t>> visited;
  BoundaryCost total;
  for (const TensorUse &use : tensorUses) {
    int64_t destination = use.consumerGroupOwner
                              ? assignedId(use.consumerGroupOwner)
                              : use.unassignedConsumerId;
    if (use.producerGroupOwner &&
        assignedId(use.producerGroupOwner) == destination)
      continue;
    if (visited.insert({use.value, destination}).second)
      total.add(use.cost);
  }
  return total;
}

std::optional<MovePlan>
UpstreamChainOptimizer::findBestPrefix(Operation *producer, int targetBlockId,
                                       BoundaryCost originalCost) const {
  int sourceBlockId = blockIds.getBlockIdByOp(producer);
  size_t sourceGroupSize =
      llvm::count_if(blockIds.getOpsRefByBlockId(sourceBlockId),
                     [&](Operation *op) { return op->getBlock() == &body; });
  SmallVector<Operation *> chain;
  llvm::SmallPtrSet<Operation *, 16> chainMembers;
  BoundaryCost bestCost = originalCost;
  size_t bestPrefixLength = 0;
  for (Operation *current = producer; current;) {
    if (current->getBlock() != &body ||
        blockIds.getBlockIdByOp(current) != sourceBlockId ||
        chainMembers.contains(current) || !isMovableProducer(current) ||
        !hasPrivateUsers(current, targetBlockId, chainMembers))
      break;
    chain.push_back(current);
    chainMembers.insert(current);
    BoundaryCost cost = evaluateCost(chainMembers, targetBlockId);
    // Preserve the source group. Strict comparison keeps the first minimum
    // from the consumer; equal costs never pull additional operations over.
    if (chain.size() < sourceGroupSize && cost < bestCost &&
        !CVPipeline::willCreateCycle(chain, memoryGraph, targetBlockId,
                                     blockIds)) {
      bestCost = cost;
      bestPrefixLength = chain.size();
    }
    // Only the data input extends the chain. Scalar/empty producers retain
    // their ids, and another MLIR block or compute group stops the search.
    current = getSingleDataInput(current)->getDefiningOp();
  }
  if (bestPrefixLength == 0)
    return std::nullopt;
  chain.resize(bestPrefixLength);
  return MovePlan{sourceBlockId, targetBlockId, std::move(chain), originalCost,
                  bestCost};
}

bool UpstreamChainOptimizer::applyPlan(const MovePlan &plan) {
  llvm::SmallPtrSet<Operation *, 16> members;
  members.insert(plan.operations.begin(), plan.operations.end());
  // Reject stale or incomplete plans before changing any attributes.
  size_t sourceGroupSize =
      llvm::count_if(blockIds.getOpsRefByBlockId(plan.sourceBlockId),
                     [&](Operation *op) { return op->getBlock() == &body; });
  if (members.empty() || members.size() != plan.operations.size() ||
      plan.sourceBlockId == plan.targetBlockId ||
      members.size() >= sourceGroupSize ||
      !llvm::all_of(plan.operations, [&](Operation *op) {
        return op->getBlock() == &body &&
               blockIds.getBlockIdByOp(op) == plan.sourceBlockId &&
               isMovableProducer(op) &&
               hasPrivateUsers(op, plan.targetBlockId, members);
      }))
    return false;
  llvm::SmallPtrSet<Operation *, 1> noMoves;
  BoundaryCost before = evaluateCost(noMoves, -1);
  BoundaryCost after = evaluateCost(members, plan.targetBlockId);
  if (!(before == plan.before) || !(after == plan.after) || !(after < before))
    return false;
  if (CVPipeline::willCreateCycle(plan.operations, memoryGraph,
                                  plan.targetBlockId, blockIds))
    return false;
  LLVM_DEBUG(llvm::dbgs() << '[' << DEBUG_TYPE << "] move "
                          << plan.operations.size() << " ops from "
                          << plan.sourceBlockId << " to " << plan.targetBlockId
                          << "; boundary cost (" << plan.before.unknownTransfers
                          << ", " << plan.before.staticBytes << ") -> ("
                          << plan.after.unknownTransfers << ", "
                          << plan.after.staticBytes << ")\n");
  // Change exactly the selected top-level block_id attributes. Payload ops,
  // scalar parameters and empty templates must keep their original ids.
  for (Operation *op : operations)
    if (members.contains(op))
      blockIds.updateBlockId(op, plan.targetBlockId);
  return true;
}

bool UpstreamChainOptimizer::optimizeOneChain() {
  llvm::SmallPtrSet<Operation *, 1> noMoves;
  BoundaryCost originalCost = evaluateCost(noMoves, -1);
  DenseSet<std::pair<Operation *, int>> attempted;
  for (Operation *consumer : operations) {
    if (!isVectorConsumer(consumer))
      continue;
    int targetBlockId = blockIds.getBlockIdByOp(consumer);
    for (OpOperand &operand : consumer->getOpOperands()) {
      if (classifyOperand(operand) != OperandRole::TensorData)
        continue;
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || producer->getBlock() != &body ||
          blockIds.getBlockIdByOp(producer) == targetBlockId ||
          !attempted.insert({producer, targetBlockId}).second ||
          !isMovableProducer(producer))
        continue;
      if (auto plan = findBestPrefix(producer, targetBlockId, originalCost)) {
        if (applyPlan(*plan))
          return true;
      }
    }
  }
  return false;
}

void UpstreamChainOptimizer::optimize() {
  // Topology and use records stay unchanged; only group ids change. Every
  // accepted plan strictly decreases the same complete boundary cost.
  while (optimizeOneChain()) {
  }
}

class UBUsageOptUpStreamPass
    : public PassWrapper<UBUsageOptUpStreamPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UBUsageOptUpStreamPass)

  StringRef getArgument() const final { return "ub-usage-opt-up-stream"; }
  StringRef getDescription() const final {
    return "Move vector tensor chains into consumer groups at the first "
           "minimum boundary-cost cut";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (CVPipeline::hasFallbackAttr(module))
      return;
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memoryGraph(module, aliasAnalysis);
    CVPipeline::ComputeBlockIdManager blockIds(module);
    module.walk([&](Block *block) {
      if (isa<scf::ForOp, scf::WhileOp>(block->getParentOp()))
        UpstreamChainOptimizer(*block, memoryGraph, blockIds).optimize();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createUBUsageOptUpStreamPass() {
  return std::make_unique<UBUsageOptUpStreamPass>();
}
