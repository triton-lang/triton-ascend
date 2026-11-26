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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/ProcessLoopInvariantRead.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "process-loop-invariant-read"
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {

static scf::WhileOp getMainLoop(scope::ScopeOp scopeOp) {
  if (scopeOp.getBodyRegion().empty())
    return nullptr;
  Operation *mainLoop = nullptr;
  for (Operation &op : scopeOp.getBodyRegion().front().getOperations()) {
    if (!CVPipeline::isMainLoopOp(&op))
      continue;
    if (mainLoop)
      return nullptr;
    mainLoop = &op;
  }
  return dyn_cast_or_null<scf::WhileOp>(mainLoop);
}

static void tagCubeSide(Operation *op, Attribute blockId) {
  op->setAttr(CVPipeline::kCoreType,
              StringAttr::get(op->getContext(), CVPipeline::kCoreTypeCube));
  if (blockId)
    op->setAttr(CVPipeline::kBlockId, blockId);
}

static bool isTransferViewOp(Operation *op) {
  return isa<hivm::ConvertLayoutOp, memref::MemorySpaceCastOp,
             memref::SubViewOp, tensor::CastOp, tensor::ExpandShapeOp,
             tensor::CollapseShapeOp>(op);
}

/// The wait is emitted right in front of the data it protects:
/// wait -> convert_layout -> memory_space_cast -> to_tensor.
static Value findReceiveTensor(Operation *anchor) {
  for (Operation *next = anchor->getNextNode(); next;
       next = next->getNextNode()) {
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(next))
      return toTensor.getResult();
    if (!isTransferViewOp(next))
      return Value();
  }
  return Value();
}

/// View-like ops that carry the receive value without reading it.
static bool isConsumerViewOp(Operation *op) {
  return isa<hivm::ConvertLayoutOp, memref::MemorySpaceCastOp,
             memref::SubViewOp, bufferization::ToTensorOp,
             tensor::ExtractSliceOp, tensor::CastOp, tensor::ExpandShapeOp,
             tensor::CollapseShapeOp>(op);
}

/// Control flow, terminators and shape queries are not data consumers.
static bool isNonConsumerOp(Operation *op) {
  return op->hasTrait<OpTrait::IsTerminator>() ||
         isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::IndexSwitchOp,
             tensor::DimOp, memref::DimOp>(op);
}

/// DFS along the receive chain. View-like ops are crossed
static Operation *findConsumerInLoop(Value receive, scf::WhileOp loop) {
  if (!receive)
    return nullptr;
  llvm::DenseSet<Value> visited;
  Value value = receive;
  unsigned budget = 64;
  while (value && budget-- > 0) {
    if (!visited.insert(value).second)
      return nullptr;
    // A branch before the consumer makes the insertion point ambiguous.
    if (!value.hasOneUse())
      return nullptr;
    Operation *user = *value.user_begin();
    if (isConsumerViewOp(user)) {
      if (user->getNumResults() != 1)
        return nullptr;
      value = user->getResult(0);
      continue;
    }
    if (isNonConsumerOp(user))
      return nullptr;
    return loop->isAncestor(user) ? user : nullptr;
  }
  return nullptr;
}

/// The loop-carried counter of the ssbuffer block that contains \p ifOp.
/// entry: counter value before the block executes (dominates \p ifOp)
/// init: initial value of that loop-carried counter
struct BlockCounter {
  Value entry;
  Value init;
};

static std::optional<BlockCounter> getBlockCounter(scf::IfOp ifOp,
                                                   scf::WhileOp loop) {
  auto yield = ifOp.thenYield();
  if (!yield)
    return std::nullopt;
  for (Value yielded : yield.getOperands()) {
    auto add = yielded.getDefiningOp<arith::AddIOp>();
    if (!add)
      continue;
    auto step = add.getRhs().getDefiningOp<arith::ConstantIntOp>();
    if (!step || step.value() <= 0)
      continue;
    auto arg = dyn_cast<BlockArgument>(add.getLhs());
    if (!arg || arg.getOwner()->getParentOp() != loop.getOperation())
      continue;
    if (arg.getArgNumber() >= loop.getInits().size())
      continue;
    return BlockCounter{add.getLhs(), loop.getInits()[arg.getArgNumber()]};
  }
  return std::nullopt;
}

/// Clone the side-effect-free condition computation of \p loop's before-region
/// into the parent block and return its negation ("the loop body never runs").
/// Returns a null Value when the condition cannot be cloned safely.
static Value buildLoopIsEmptyCondition(OpBuilder &builder, scf::WhileOp loop,
                                       Attribute blockId) {
  if (!loop.getBefore().hasOneBlock())
    return Value();
  Block &before = loop.getBefore().front();
  auto conditionOp = dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!conditionOp)
    return Value();

  IRMapping mapping;
  for (auto [arg, init] : llvm::zip(before.getArguments(), loop.getInits()))
    mapping.map(arg, init);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loop);
  for (Operation &op : before.without_terminator()) {
    if (op.getNumRegions() != 0 || !isMemoryEffectFree(&op))
      return Value();
    Operation *cloned = builder.clone(op, mapping);
    tagCubeSide(cloned, blockId);
  }

  Value cond = mapping.lookupOrNull(conditionOp.getCondition());
  if (!cond)
    return Value();
  auto one = builder.create<arith::ConstantIntOp>(
      loop.getLoc(), 1, cast<IntegerType>(cond.getType()).getWidth());
  tagCubeSide(one.getOperation(), blockId);
  auto notRuns = builder.create<arith::XOrIOp>(loop.getLoc(), cond, one);
  tagCubeSide(notRuns.getOperation(), blockId);
  return notRuns;
}

static hivm::SyncBlockSetOp findMatchingSet(ArrayRef<hivm::SyncBlockSetOp> sets,
                                            hivm::SyncBlockWaitOp wait) {
  auto waitFlag = wait.getStaticFlagId();
  if (!waitFlag)
    return nullptr;
  for (hivm::SyncBlockSetOp set : sets) {
    auto setFlag = set.getStaticFlagId();
    if (!setFlag || setFlag->getInt() != waitFlag->getInt())
      continue;
    if (set.getTpipeAttr().getPipe() != wait.getTpipeAttr().getPipe() ||
        set.getPipeAttr().getPipe() != wait.getPipeAttr().getPipe())
      continue;
    return set;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class ProcessLoopInvariantReadPass
    : public PassWrapper<ProcessLoopInvariantReadPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProcessLoopInvariantReadPass)

  StringRef getArgument() const override {
    return "process-loop-invariant-read";
  }
  StringRef getDescription() const override {
    return "Sink loop-invariant Vector->Cube read waits into the consumer loop";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, scope::ScopeDialect, arith::ArithDialect,
                    scf::SCFDialect, bufferization::BufferizationDialect,
                    memref::MemRefDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;

private:
  bool processCubeScope(scope::ScopeOp cubeScope, scf::WhileOp loop,
                        ArrayRef<hivm::SyncBlockSetOp> vectorSets);
};

bool ProcessLoopInvariantReadPass::processCubeScope(
    scope::ScopeOp cubeScope, scf::WhileOp loop,
    ArrayRef<hivm::SyncBlockSetOp> vectorSets) {
  IRRewriter rewriter(&getContext());
  Block &body = cubeScope.getBodyRegion().front();

  // Collect all block-head waits first: they are the loop-invariant read
  // waits emitted by InterCoreTransferAndSync. Waits inside the loop are
  // already per-iteration and are left untouched.
  SmallVector<hivm::SyncBlockWaitOp> candidates;
  for (Operation &op : body.getOperations()) {
    auto wait = dyn_cast<hivm::SyncBlockWaitOp>(op);
    if (wait && wait->isBeforeInBlock(loop))
      candidates.push_back(wait);
  }

  bool changed = false;
  for (hivm::SyncBlockWaitOp wait : candidates) {
    if (!findMatchingSet(vectorSets, wait)) {
      LOG_DEBUG("no loop-invariant producer set for " << wait);
      continue;
    }
    Value receive = findReceiveTensor(wait);
    Operation *consumer = findConsumerInLoop(receive, loop);
    if (!consumer) {
      LOG_DEBUG("no in-loop consumer for " << wait);
      continue;
    }
    scf::IfOp blockIf;
    for (Operation *parent = consumer->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto ifOp = dyn_cast<scf::IfOp>(parent);
      if (ifOp && ifOp->hasAttr(CVPipeline::kIf)) {
        blockIf = ifOp;
        break;
      }
    }
    if (!blockIf) {
      LOG_DEBUG("consumer is not inside an ssbuffer block");
      continue;
    }
    std::optional<BlockCounter> counter = getBlockCounter(blockIf, loop);
    if (!counter) {
      LOG_DEBUG("cannot identify the ssbuffer block counter");
      continue;
    }

    Location loc = wait.getLoc();
    Attribute blockId = wait->getAttr(CVPipeline::kBlockId);
    Value loopIsEmpty = buildLoopIsEmptyCondition(rewriter, loop, blockId);
    if (!loopIsEmpty) {
      LOG_DEBUG("cannot rebuild the loop entry condition");
      continue;
    }

    auto cloneWait = [&](OpBuilder &builder) {
      auto cloned = builder.create<hivm::SyncBlockWaitOp>(
          loc, wait.getTcoreTypeAttr(), wait.getTpipeAttr(), wait.getPipeAttr(),
          wait.getFlagId());
      cloned->setAttrs(wait->getAttrs());
      return cloned;
    };

    // 1) Zero-trip fallback: the producer published its token but the consumer
    //    loop never runs, so consume it here. Exactly one of the two waits
    //    executes.
    rewriter.setInsertionPoint(loop);
    auto emptyIf = rewriter.create<scf::IfOp>(loc, loopIsEmpty,
                                              /*withElseRegion=*/false);
    tagCubeSide(emptyIf, blockId);
    rewriter.setInsertionPointToStart(&emptyIf.getThenRegion().front());
    cloneWait(rewriter);

    // 2) First-execution guard in front of the real consumer.
    rewriter.setInsertionPoint(consumer);
    Value isFirst = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, counter->entry, counter->init);
    tagCubeSide(isFirst.getDefiningOp(), blockId);
    auto firstIf = rewriter.create<scf::IfOp>(loc, isFirst,
                                              /*withElseRegion=*/false);
    tagCubeSide(firstIf, blockId);
    rewriter.setInsertionPointToStart(&firstIf.getThenRegion().front());
    cloneWait(rewriter);

    rewriter.eraseOp(wait);
    changed = true;
    LOG_DEBUG("sank read wait before " << *consumer);
  }
  return changed;
}

void ProcessLoopInvariantReadPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module))
    return;

  module.walk([&](func::FuncOp funcOp) {
    // The pipeline has one V/C scope per core; take the first of each.
    scope::ScopeOp vectorScope;
    scope::ScopeOp cubeScope;
    funcOp.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
      bool isCube = false;
      bool isVector = false;
      if (failed(getScopeType(scopeOp, isCube, isVector)))
        return WalkResult::advance();
      if (isVector && !vectorScope)
        vectorScope = scopeOp;
      else if (isCube && !cubeScope)
        cubeScope = scopeOp;
      return vectorScope && cubeScope ? WalkResult::interrupt()
                                      : WalkResult::advance();
    });
    if (!vectorScope || !cubeScope)
      return;

    scf::WhileOp vectorLoop = getMainLoop(vectorScope);
    scf::WhileOp cubeLoop = getMainLoop(cubeScope);
    if (!vectorLoop || !cubeLoop) {
      LOG_DEBUG("expected exactly one scf.while main loop per core");
      return;
    }

    // Producer sets that are loop invariant: emitted in the VECTOR scope body,
    // i.e. before the vector main loop.
    SmallVector<hivm::SyncBlockSetOp> vectorSets;
    for (Operation &op : vectorScope.getBodyRegion().front().getOperations()) {
      auto set = dyn_cast<hivm::SyncBlockSetOp>(op);
      if (set && set->isBeforeInBlock(vectorLoop))
        vectorSets.push_back(set);
    }
    if (vectorSets.empty())
      return;

    (void)processCubeScope(cubeScope, cubeLoop, vectorSets);
  });
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createProcessLoopInvariantReadPass() {
  return std::make_unique<ProcessLoopInvariantReadPass>();
}
