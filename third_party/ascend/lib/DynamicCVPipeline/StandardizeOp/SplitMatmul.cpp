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

#include "DynamicCVPipeline/StandardizeOp/SplitMatmul.h"
#include "DynamicCVPipeline/Common/ControlFlowGraph.h"
#include "DynamicCVPipeline/Common/SSBufferManager.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::CVPipeline;

static constexpr const char *DEBUG_TYPE = "SplitMatmul";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {
class SplitMatmulPass
    : public PassWrapper<SplitMatmulPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitMatmulPass)
  SplitMatmulPass();
  SplitMatmulPass(const SplitMatmulPass &other);
  StringRef getArgument() const final;
  StringRef getDescription() const final;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;

private:
  struct SelfReusedL0C {
    SmallVector<linalg::MatmulOp> matmuls;
    Value inputValue;
    Value outputValue;
    int resultIndex;
  };

  struct LinkReuseL0COps {
    SmallVector<unsigned> minis;
    Value input;
    Value output;
    int resultIndex;
  };

  static bool regionMayNotExecute(Operation *parent, Operation *child);
  static Operation *getSelfReusedL0CEntryOp(const SelfReusedL0C &self);
  static bool mayNotExecute(const SelfReusedL0C &self);
  static bool shouldSplitMatmul(Value input);
  static Value createZero(OpBuilder &builder, Location loc, Value shape);

  static SmallVector<OpOperand *>
  getSelfReusedL0CInputUses(const SelfReusedL0C &self,
                            const ControlFlowGraph &graph);
  void collectSelfReusedL0Cs(const ControlFlowGraph &graph);
  LogicalResult linkReuseL0COps();
  void analyzeSplitAndMNE(DenseMap<const SelfReusedL0C *, bool> &isMNE,
                          DenseMap<const LinkReuseL0COps *, bool> &shouldSplit,
                          DenseMap<const LinkReuseL0COps *, bool> &isMNELink);
  void
  printAnalysis(const DenseMap<const SelfReusedL0C *, bool> &isMNE,
                const DenseMap<const LinkReuseL0COps *, bool> &shouldSplit,
                const DenseMap<const LinkReuseL0COps *, bool> &isMNELink) const;
  static void splitMatmulChain(OpBuilder &builder, LinkReuseL0COps &link,
                               ArrayRef<OpOperand *> inputUses,
                               Operation *entry);
  void
  setAttrForMatmulChain(OpBuilder &builder, const LinkReuseL0COps &link,
                        const DenseMap<const SelfReusedL0C *, bool> &isMNE);
  LogicalResult
  splitMatmulChains(OpBuilder &builder, const ControlFlowGraph &graph,
                    const DenseMap<const LinkReuseL0COps *, bool> &shouldSplit);
  LogicalResult insertMNEGuardUB(OpBuilder &builder,
                                 const LinkReuseL0COps &link,
                                 SSBufferManager &manager);

  SmallVector<SelfReusedL0C> selfReused;
  SmallVector<LinkReuseL0COps> links;
  int coupledId = 0;
};

SplitMatmulPass::SplitMatmulPass() = default;

SplitMatmulPass::SplitMatmulPass(const SplitMatmulPass &other)
    : PassWrapper(other) {}

StringRef SplitMatmulPass::getArgument() const { return "ssbuf-split-matmul"; }

StringRef SplitMatmulPass::getDescription() const {
  return "Split accumulator reuse chains using a control-flow-aware graph";
}

void SplitMatmulPass::getDependentDialects(DialectRegistry &registry) const {
  registry
      .insert<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect,
              scf::SCFDialect, memref::MemRefDialect, hivm::HIVMDialect>();
}

bool SplitMatmulPass::regionMayNotExecute(Operation *parent, Operation *child) {
  if (auto loop = dyn_cast<scf::ForOp>(parent)) {
    IntegerAttr lb, ub, step;
    return !(matchPattern(loop.getLowerBound(), m_Constant(&lb)) &&
             matchPattern(loop.getUpperBound(), m_Constant(&ub)) && lb.getInt() < ub.getInt());
  }
  if (auto branch = dyn_cast<scf::IfOp>(parent)) {
    IntegerAttr cond;
    if (!matchPattern(branch.getCondition(), m_One())) {
      return true;
    }
  }
  if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
    if (loop.getBefore().isAncestor(child->getParentRegion())) {
      return false;
    }
    auto term =
        cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
    IntegerAttr cond;
    return !matchPattern(term.getCondition(), m_Constant(&cond)) ||
           cond.getValue().isZero();
  }
  // Unknown region semantics cannot prove execution.
  return false;
}

SmallVector<OpOperand *>
SplitMatmulPass::getSelfReusedL0CInputUses(const SelfReusedL0C &self,
                                           const ControlFlowGraph &graph) {
  // A local singleton is the fallback when the full component cannot safely
  // hoist C. Preserve that local scope instead of re-expanding through Phis.
  auto firstMatmul = self.matmuls.front();
  if (self.matmuls.size() == 1 && self.outputValue &&
      self.outputValue.getDefiningOp() == firstMatmul.getOperation()) {
    auto mm = self.matmuls.front();
    return {&mm.getDpsInitsMutable()[0]};
  }
  for (const auto &component :
       graph.getSCCs([](Operation *op) { return isa<linalg::MatmulOp>(op); })) {
    if (!llvm::is_contained(component.Ops, firstMatmul.getOperation())) {
      continue;
    }
    SmallVector<OpOperand *> uses;
    auto entrys = graph.getLoopEntry(component);
    graph.getSCCInputValues(component, entrys, &uses);
    return uses;
  }
  return {};
}

Operation *SplitMatmulPass::getSelfReusedL0CEntryOp(const SelfReusedL0C &self) {
  if (!self.outputValue || self.matmuls.empty()) {
    return nullptr;
  }
  // The first matmul's ancestor in the output scope is the chain entry.
  // This depends only on nesting and remains valid after replacing C.
  auto firstMatmul = self.matmuls.front();
  return self.outputValue.getDefiningOp()->getBlock()->findAncestorOpInBlock(
      *firstMatmul.getOperation());
}

bool SplitMatmulPass::mayNotExecute(const SelfReusedL0C &self) {
  if (utils::getAnnotateOpWithAttr(self.outputValue, kMatmulAtLeastOnceHint)) {
    return false;
  }
  if  (self.matmuls.size() == 1 && self.matmuls.front()->getResult(0) == self.outputValue) {
    // if mm's reslult is outputValue, then it just one matmul in the selfResusedL0C.
    return false;
  };
  Block *scope = self.outputValue.getDefiningOp()->getBlock();
  bool allMatmulisMNE = true;
  for (auto mm : self.matmuls) {
    Operation *child = mm;
    bool mne = false;
    for (Operation *parent = child->getParentOp(); parent;
         child = parent, parent = parent->getParentOp()) {
      if (regionMayNotExecute(parent, child)) {
        mne = true;
        break;
      }
      if (parent->getBlock() == scope) {
        break;
      }
    }
    if (!mne) {
      allMatmulisMNE = false;
      break;
    }
  }
  return allMatmulisMNE;
}

void SplitMatmulPass::collectSelfReusedL0Cs(const ControlFlowGraph &graph) {
  llvm::SmallPtrSet<Operation *, 16> assigned;
  for (const auto &component :
       graph.getSCCs([](Operation *op) { return isa<linalg::MatmulOp>(op); })) {
    auto entrys = graph.getLoopEntry(component);
    auto exits = graph.getLoopExit(component);
    if (entrys.size() != 1 || exits.size() != 1) {
      LOG_DEBUG("Skip residency component: entries="
                << entrys.size() << " exits=" << exits.size());
      continue;
    }
    auto inputs = graph.getSCCInputValues(component, entrys);
    auto outputs = graph.getSCCResultValues(component, exits);
    // A node may represent several results/SCF aliases. Check SSA uniqueness
    // explicitly instead of assuming that one node implies one Value.
    if (inputs.size() != 1 || outputs.size() != 1) {
      LOG_DEBUG("Skip residency component: inputs="
                << inputs.size() << " outputs=" << outputs.size());
      continue;
    }
    auto result = dyn_cast<OpResult>(outputs.front());
    if (!result) {
      continue;
    }
    SelfReusedL0C self;
    for (Operation *op : component.Ops) {
      if (auto mm = dyn_cast<linalg::MatmulOp>(op)) {
        self.matmuls.push_back(mm);
      }
    }
    if (self.matmuls.empty()) {
      continue;
    }
    self.inputValue = inputs.front();
    self.outputValue = outputs.front();
    self.resultIndex = result.getResultNumber();
    for (auto mm : self.matmuls) {
      assigned.insert(mm);
    }
    selfReused.push_back(std::move(self));
  }
  // Every matmul has exactly one owner, even if a component must split locally.
  getOperation()->walk([&](linalg::MatmulOp mm) {
    if (assigned.contains(mm)) {
      return;
    }
    SelfReusedL0C self;
    self.matmuls.push_back(mm);
    self.resultIndex = -1;
    if (mm.getNumResults() == 1) { // Avoid memref matmul
      self.inputValue = mm.getDpsInits()[0];
      self.outputValue = mm.getResult(0);
      self.resultIndex = 0;
    }
    selfReused.push_back(std::move(self));
  });
}

LogicalResult SplitMatmulPass::linkReuseL0COps() {
  SmallVector<bool> seen(selfReused.size(), false);
  DenseMap<Value, unsigned> inputToIndex;
  DenseMap<Value, unsigned> outputToIndex;
  for (unsigned i = 0; i < selfReused.size(); ++i) {
    if (selfReused[i].inputValue) {
      auto [it, inserted] =
          inputToIndex.try_emplace(selfReused[i].inputValue, i);
      if (!inserted) {
        // Shared inputs do not define a unique one-to-one successor.
        it->second = selfReused.size();
      }
    }
    Value output = selfReused[i].outputValue;
    if (output && output.hasOneUse()) {
      auto [it, inserted] = outputToIndex.try_emplace(output, i);
      if (!inserted) {
        it->second = selfReused.size();
      }
    }
  }
  auto nextIndex = [&](unsigned now) -> int {
    Value output = selfReused[now].outputValue;
    auto producer = outputToIndex.find(output);
    if (producer == outputToIndex.end() || producer->second != now) {
      return -1;
    }
    auto next = inputToIndex.find(output);
    if (next == inputToIndex.end() || next->second >= selfReused.size()) {
      return -1;
    }
    return next->second;
  };
  auto producerIndex = [&](unsigned now) -> int {
    auto producer = outputToIndex.find(selfReused[now].inputValue);
    if (producer == outputToIndex.end() ||
        producer->second >= selfReused.size() ||
        nextIndex(producer->second) != static_cast<int>(now)) {
      return -1;
    }
    return producer->second;
  };
  auto dfsProducer = [&](auto &&visit, unsigned now,
                         SmallVector<unsigned> &chain) -> LogicalResult {
    if (seen[now]) {
      return failure();
    }
    seen[now] = true;
    int producer = producerIndex(now);
    if (producer != -1 && failed(visit(visit, producer, chain))) {
      return failure();
    }
    // Post-order appending keeps the producer side in data-flow order.
    chain.push_back(now);
    return success();
  };
  auto dfsConsumer = [&](auto &&visit, unsigned now,
                         SmallVector<unsigned> &chain) -> LogicalResult {
    if (seen[now]) {
      return failure();
    }
    seen[now] = true;
    chain.push_back(now);
    int next = nextIndex(now);
    if (next != -1) {
      return visit(visit, next, chain);
    }
    return success();
  };
  for (unsigned i = 0; i < selfReused.size(); ++i) {
    if (seen[i]) {
      continue;
    }
    SmallVector<unsigned> chain;
    if (failed(dfsProducer(dfsProducer, i, chain))) {
      LOG_DEBUG("Producer DFS failed starting from SelfReusedL0C index " << i);
      return failure();
    }
    int next = nextIndex(i); // Producer traversal already included i; start the
                             // other side at its next.
    if (next != -1 && failed(dfsConsumer(dfsConsumer, next, chain))) {
      LOG_DEBUG("Consumer DFS failed starting from SelfReusedL0C index "
                << next << " (chain seed " << i << ")");
      return failure();
    }
    LinkReuseL0COps link;
    link.input = selfReused[chain.front()].inputValue;
    link.output = selfReused[chain.back()].outputValue;
    link.resultIndex = selfReused[chain.back()].resultIndex;
    link.minis = std::move(chain);
    links.push_back(std::move(link));
  }
  return success();
}

bool SplitMatmulPass::shouldSplitMatmul(Value input) {
  if (!input) {
    return false;
  }
  Operation *def = input.getDefiningOp();
  if (def && isa<tensor::EmptyOp>(def)) {
    return false;
  }
  if (auto broadcast = dyn_cast_or_null<linalg::BroadcastOp>(def)) {
    int64_t btUsage = CVPipeline::getBTSizeFromValidBroadcastOp(broadcast);
    if (btUsage > 0 && btUsage <= CVPipeline::CACHE_TABLE_BUFFER_SIZE) {
      return false;
    }
  }
  if (auto fill = dyn_cast_or_null<linalg::FillOp>(def)) {
    Value scalar = fill.getInputs()[0];
    if (matchPattern(scalar, m_Zero()) ||
        matchPattern(scalar, m_AnyZeroFloat())) {
      return false;
    }
  }
  return true;
}

void SplitMatmulPass::analyzeSplitAndMNE(
    DenseMap<const SelfReusedL0C *, bool> &isMNE,
    DenseMap<const LinkReuseL0COps *, bool> &shouldSplit,
    DenseMap<const LinkReuseL0COps *, bool> &isMNELink) {
  for (const auto &self : selfReused) {
    isMNE[&self] = mayNotExecute(self);
  }
  for (const auto &link : links) {
    shouldSplit[&link] = shouldSplitMatmul(link.input);
    isMNELink[&link] = llvm::all_of(link.minis, [&](unsigned index) {
      return isMNE.lookup(&selfReused[index]);
    });
  }
}

void SplitMatmulPass::printAnalysis(
    const DenseMap<const SelfReusedL0C *, bool> &isMNE,
    const DenseMap<const LinkReuseL0COps *, bool> &shouldSplit,
    const DenseMap<const LinkReuseL0COps *, bool> &isMNELink) const {
  LOG_DEBUG("split-matmul: self-reused=" << selfReused.size()
                                         << " links=" << links.size());
  for (const auto &[i, link] : llvm::enumerate(links)) {
    LOG_DEBUG("link " << i << ": minis=" << link.minis.size()
                      << " split=" << shouldSplit.lookup(&link)
                      << " may-not-execute=" << isMNELink.lookup(&link)
                      << "\n  inputValue: " << link.input
                      << "\n  outputValue: " << link.output);
    LOG_DEBUG("printing selfL0C...");
    for (unsigned id : link.minis) {
      LOG_DEBUG(" sub-self-reused " << id);
    }
  }
  LOG_DEBUG("------------------------------------------------------------");
  for (const auto &[i, self] : llvm::enumerate(selfReused)) {
    LOG_DEBUG("self-reused " << i << ": matmuls=" << self.matmuls.size()
                             << " may-not-execute=" << isMNE.lookup(&self)
                             << " result-index=" << self.resultIndex
                             << "\n  inputValue: " << self.inputValue
                             << "\n  outputValue: " << self.outputValue);
    for (auto mm : self.matmuls) {
      LOG_DEBUG("   sub-matmul: " << mm);
    }
  }
}

Value SplitMatmulPass::createZero(OpBuilder &builder, Location loc,
                                  Value shape) {
  auto type = cast<RankedTensorType>(shape.getType());
  SmallVector<Value> sizes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.isDynamicDim(i)) {
      sizes.push_back(builder.create<tensor::DimOp>(loc, shape, i));
    }
  }
  auto empty = builder.create<tensor::EmptyOp>(loc, type, sizes);
  auto scalar = builder.create<arith::ConstantOp>(
      loc, type.getElementType(), builder.getZeroAttr(type.getElementType()));
  return builder
      .create<linalg::FillOp>(loc, scalar.getResult(), empty.getResult())
      .getResult(0);
}

// Zero the chain input and add the original C to its final output.
// Update link.input to the new entry value used by the later MNE guard.
void SplitMatmulPass::splitMatmulChain(OpBuilder &builder,
                                       LinkReuseL0COps &link,
                                       ArrayRef<OpOperand *> inputUses,
                                       Operation *entry) {
  Value input = inputUses.front()->get();
  builder.setInsertionPoint(entry);
  Value zero = createZero(builder, entry->getLoc(), input);
  for (OpOperand *use : inputUses) {
    use->set(zero);
  }

  // Save the original consumers before the new add uses the chain output.
  SmallVector<OpOperand *> users;
  for (OpOperand &use : link.output.getUses()) {
    users.push_back(&use);
  }
  Operation *exit = link.output.getDefiningOp();
  builder.setInsertionPointAfter(exit);
  auto type = cast<RankedTensorType>(input.getType());
  Operation *add;
  if (isa<FloatType>(type.getElementType())) {
    add = builder.create<arith::AddFOp>(exit->getLoc(), link.output, input);
  } else {
    add = builder.create<arith::AddIOp>(exit->getLoc(), link.output, input);
  }
  add->setAttr(CVPipeline::kAddFromMatmul, builder.getUnitAttr());
  for (OpOperand *use : users) {
    use->set(add->getResult(0));
  }
  link.input = zero;
}

void SplitMatmulPass::setAttrForMatmulChain(
    OpBuilder &builder, const LinkReuseL0COps &link,
    const DenseMap<const SelfReusedL0C *, bool> &isMNE) {
  for (unsigned id : link.minis) {
    const auto &self = selfReused[id];
    for (auto mm : self.matmuls) {
      Operation *inner = mm->getParentOp();
      if (inner &&
          !inner->hasAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr)) {
        inner->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr,
                       builder.getUnitAttr());
      }
    }
    Operation *outer = self.outputValue.getDefiningOp();
    if (isMNE.lookup(&self) &&
        !outer->hasAttr(CVPipeline::kMayNotExecNPU)) {
      outer->setAttr(CVPipeline::kMayNotExecNPU, builder.getBoolAttr(true));
    }
  }
}

LogicalResult SplitMatmulPass::splitMatmulChains(
    OpBuilder &builder, const ControlFlowGraph &graph,
    const DenseMap<const LinkReuseL0COps *, bool> &shouldSplit) {
  for (auto &link : links) {
    const auto &first = selfReused[link.minis.front()];
    auto inputUses = getSelfReusedL0CInputUses(first, graph);
    Operation *entry = getSelfReusedL0CEntryOp(first);
    if (inputUses.empty() || !entry) {
      LOG_DEBUG("Failed to find split entry for " << link.output);
      return failure();
    }
    // Preceding chains may already have replaced the input producer.
    link.input = inputUses.front()->get();
    if (shouldSplit.lookup(&link)) {
      splitMatmulChain(builder, link, inputUses, entry);
    }
  }
  return success();
}

// Track each matmul independently; guard only when all were skipped.
LogicalResult SplitMatmulPass::insertMNEGuardUB(OpBuilder &builder,
                                                const LinkReuseL0COps &link,
                                                SSBufferManager &manager) {
  Operation *entry = getSelfReusedL0CEntryOp(selfReused[link.minis.front()]);
  if (!entry) {
    LOG_DEBUG("Failed to find MNE guard entry for " << link.output);
    return failure();
  }
  Location loc = entry->getLoc();
  SmallVector<Value> counters;
  for (unsigned id : link.minis) {
    for (auto mm : selfReused[id].matmuls) {
      auto address = manager.allocateMNECounterAddr();
      if (failed(address)) {
        LOG_DEBUG("Failed to allocate MNE counter address for " << mm);
        return failure();
      }
      builder.setInsertionPoint(entry);
      auto [addr, cast] = getSsbufConstAndPointerCast(builder, loc, *address,
                                                      builder.getI32Type());
      Value counter = cast.getResult();
      counters.push_back(counter);
      auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      auto init =
          builder.create<memref::StoreOp>(loc, zero, counter, ValueRange{});
      init->setAttr(CVPipeline::kMNEStore0, builder.getUnitAttr());
      builder.setInsertionPointAfter(mm);
      auto one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
      auto store =
          builder.create<memref::StoreOp>(loc, one, counter, ValueRange{});
      auto tag = builder.getI32IntegerAttr(coupledId++);
      mm->setAttr(CVPipeline::kCoupledMatmulAndStore, tag);
      store->setAttr(CVPipeline::kCoupledMatmulAndStore, tag);
    }
  }
  // Include the add introduced by the split stage among the uses to guard.
  SmallVector<OpOperand *> users;
  for (OpOperand &use : link.output.getUses()) {
    users.push_back(&use);
  }
  builder.setInsertionPointAfter(link.output.getDefiningOp());
  auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value allSkipped;
  for (Value counter : counters) {
    auto load = builder.create<memref::LoadOp>(loc, counter, ValueRange{});
    Value skipped = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  load, zero);
    if (allSkipped) {
      allSkipped = builder.create<arith::AndIOp>(loc, allSkipped, skipped);
    } else {
      allSkipped = skipped;
    }
  }
  auto guard =
      builder.create<scf::IfOp>(loc, link.output.getType(), allSkipped, true);
  builder.setInsertionPointToStart(&guard.getThenRegion().front());
  builder.create<scf::YieldOp>(loc, link.input);
  builder.setInsertionPointToStart(&guard.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, link.output);
  for (OpOperand *use : users) {
    use->set(guard.getResult(0));
  }
  builder.setInsertionPointAfter(guard);
  return success();
}

void SplitMatmulPass::runOnOperation() {
  auto module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
  // Reset invocation state before collecting operations and caching decisions.
  selfReused.clear();
  links.clear();
  coupledId = 0;

  // Stage 1: build the C value-flow graph and collect residency blocks.
  ControlFlowGraph graph(module, [](OpOperand &operand) {
    if (auto mm = dyn_cast<linalg::MatmulOp>(operand.getOwner())) {
      return &operand == &mm.getDpsInitsMutable()[0];
    }
    return true;
  });
  collectSelfReusedL0Cs(graph);
  // Stage 2: link one-to-one chains, including isolated residency blocks.
  if (failed(linkReuseL0COps())) {
    module.emitError("split-matmul: reuse chain must be acyclic");
    signalPassFailure();
    return;
  }
  // Stage 3: analyze decisions before logging or changing IR.
  DenseMap<const SelfReusedL0C *, bool> isMNE;
  DenseMap<const LinkReuseL0COps *, bool> shouldSplit;
  DenseMap<const LinkReuseL0COps *, bool> isMNELink;
  analyzeSplitAndMNE(isMNE, shouldSplit, isMNELink);
  LLVM_DEBUG(printAnalysis(isMNE, shouldSplit, isMNELink));
  // Stage 4: split selected chains.
  OpBuilder builder(module.getContext());
  if (failed(splitMatmulChains(builder, graph, shouldSplit))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }
  // Stage 5: insert a counter per matmul and guard the chain contribution.
  SSBufferManager manager;
  for (const auto &link : links) {
    if (isMNELink.lookup(&link)) {
      if (failed(insertMNEGuardUB(builder, link, manager))) {
        signalPassFailure();
        return;
      }
    }
  }

  // Stage 6: annotate each original residency boundary after all rewrites.
  for (const auto &link : links) {
    setAttrForMatmulChain(builder, link, isMNE);
  }
  for (auto &self : selfReused) {
  if (auto markOpOpt = utils::getAnnotateOpWithAttr(
          self.outputValue, kMatmulAtLeastOnceHint);
      markOpOpt.has_value()) {
    auto markOp = markOpOpt.value();
    markOp->removeAttr(kMatmulAtLeastOnceHint);

    if (markOp->getAttrs().empty() ||
        (markOp->getAttrs().size() == 1 &&
         markOp->getAttrs().front().getName() == "effects")) {
      markOp->erase();
    }
  }
}
  if (failed(verify(module))) {
    signalPassFailure();
  }
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::CVPipeline::createSplitMatmulPass() {
  return std::make_unique<SplitMatmulPass>();
}
