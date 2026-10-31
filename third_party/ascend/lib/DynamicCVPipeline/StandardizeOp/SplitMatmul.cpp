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

#include "ascend/include/DynamicCVPipeline/StandardizeOp/SplitMatmul.h"
#include "ascend/include/DynamicCVPipeline/Common/ControlFlowGraph.h"
#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::CVPipeline;

static constexpr const char *DEBUG_TYPE = "ssbuffer-split-matmul";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

using UnitId = unsigned;
struct L0CReuseUnit {
  SmallVector<linalg::MatmulOp> matmuls;
  Value inputValue;
  OpResult outputResult;
  Operation *entryOp = nullptr;
  OpOperand *inputOpOperand = nullptr;
  bool isMNE = false;
};

struct L0CReuseAdjacency {
  SmallVector<int> next;
  SmallVector<int> prev;
};

struct L0CReuseLink {
  SmallVector<UnitId> units;
  SmallVector<linalg::MatmulOp> matmuls;
  Value inputValue;
  OpResult outputResult;
  Operation *entryOp = nullptr;
  OpOperand *inputOpOperand = nullptr;
  bool shouldSplit = true;
  bool isMNE = false;

  // Preconditions: seen[seed] == false; the adjacency describes disjoint
  // linear chains (each unit has at most one producer and one consumer via
  // single-use links), so DFS from a fresh seed can never encounter an
  // already-seen node.
  static L0CReuseLink build(const SmallVector<L0CReuseUnit> &l0cReusUnits,
                            const L0CReuseAdjacency &adjacency,
                            SmallVector<bool> &seen, unsigned seed) {
    SmallVector<unsigned> chain;
    dfsProducer(adjacency, seen, seed, chain);
    // Producer traversal already included the seed; start the other side at
    // its successor.
    int next = adjacency.next[seed];
    if (next != -1) {
      dfsConsumer(adjacency, seen, next, chain);
    }

    L0CReuseLink link;
    link.inputValue = l0cReusUnits[chain.front()].inputValue;
    link.outputResult = l0cReusUnits[chain.back()].outputResult;
    link.entryOp = l0cReusUnits[chain.front()].entryOp;
    link.inputOpOperand = l0cReusUnits[chain.front()].inputOpOperand;
    link.units = std::move(chain);
    for (unsigned id : link.units) {
      link.matmuls.append(l0cReusUnits[id].matmuls);
    }
    return link;
  }

  static void dfsProducer(const L0CReuseAdjacency &adjacency,
                          SmallVector<bool> &seen, unsigned now,
                          SmallVector<unsigned> &chain) {
    if (seen[now]) {
      return;
    }
    seen[now] = true;
    int producer = adjacency.prev[now];
    if (producer != -1) {
      dfsProducer(adjacency, seen, producer, chain);
    }
    // Post-order appending keeps the producer side in data-flow order.
    chain.push_back(now);
  }

  static void dfsConsumer(const L0CReuseAdjacency &adjacency,
                          SmallVector<bool> &seen, unsigned now,
                          SmallVector<unsigned> &chain) {
    if (seen[now]) {
      return;
    }
    seen[now] = true;
    chain.push_back(now);
    int next = adjacency.next[now];
    if (next != -1) {
      dfsConsumer(adjacency, seen, next, chain);
    }
  }
};

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
  void collectL0CReuseUnits(SmallVector<L0CReuseUnit> &l0cReusUnits,
                            const ControlFlowGraph &graph);
  LogicalResult linkReuseL0COps(const SmallVector<L0CReuseUnit> &l0cReusUnits,
                                SmallVector<L0CReuseLink> &l0cReuselinks);
  void analyzeSplitAndMNE(SmallVector<L0CReuseUnit> &l0cReusUnits,
                          SmallVector<L0CReuseLink> &l0cReuselinks);
  void printAnalysis(const SmallVector<L0CReuseUnit> &l0cReusUnits,
                     const SmallVector<L0CReuseLink> &l0cReuselinks) const;
  static void splitMatmulChain(OpBuilder &builder, L0CReuseLink &link,
                               OpOperand *inputUse, Operation *entry);
  void setAttrForMatmulChain(OpBuilder &builder,
                             const SmallVector<L0CReuseUnit> &l0cReusUnits,
                             const L0CReuseLink &link);
  LogicalResult splitMatmulChains(OpBuilder &builder,
                                  SmallVector<L0CReuseLink> &l0cReuselinks);
  LogicalResult insertMNEGuardUB(OpBuilder &builder, const L0CReuseLink &link,
                                 int &coupledId, SSBufferManager &manager);
};

SplitMatmulPass::SplitMatmulPass() = default;

SplitMatmulPass::SplitMatmulPass(const SplitMatmulPass &other)
    : PassWrapper(other) {}

StringRef SplitMatmulPass::getArgument() const {
  return "ssbuffer-split-matmul";
}

StringRef SplitMatmulPass::getDescription() const {
  return "Split accumulator reuse chains using a control-flow-aware graph";
}

void SplitMatmulPass::getDependentDialects(DialectRegistry &registry) const {
  registry
      .insert<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect,
              scf::SCFDialect, memref::MemRefDialect, hivm::HIVMDialect>();
}

static bool regionMayNotExecute(Operation *parent, Operation *child) {
  if (auto loop = dyn_cast<scf::ForOp>(parent)) {
    IntegerAttr lb, ub, step;
    return !(matchPattern(loop.getLowerBound(), m_Constant(&lb)) &&
             matchPattern(loop.getUpperBound(), m_Constant(&ub)) &&
             lb.getInt() < ub.getInt());
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

static bool mayNotExecute(const L0CReuseUnit &self) {
  if (utils::getAnnotateOpWithAttr(self.outputResult, kMatmulAtLeastOnceHint)) {
    return false;
  }
  if (self.matmuls.size() == 1 &&
      self.matmuls.front()->getResult(0) == self.outputResult) {
    // if mm's reslult is outputValue, then it just one matmul in the
    // selfResusedL0C.
    return false;
  };
  // Terminus of the parent-walk below: the block owning the chain's terminal
  // output op. Traversal stops once an ancestor reaches this block, bounding
  // the may-not-execute check to the reuse scope.
  Block *scope = self.outputResult.getDefiningOp()->getBlock();
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
      return false;
    }
  }
  return true;
}

static bool isValidL0CReusedComponent(
    const ControlFlowGraph::ConnectedComponent &component) {
  if (component.inputOperands.size() != 1 ||
      component.outputResults.size() != 1 ||
      component.inputOperands.front()->getOwner() !=
          component.outputResults.front().getOwner()) {
    LOG_DEBUG("Skip residency component: inputs="
              << component.inputOperands.size()
              << " outputs=" << component.outputResults.size());
    if (component.inputOperands.size() == 1 &&
        component.outputResults.size() == 1) {
      LOG_DEBUG("component.inputOperands.front()->getOwner() = "
                << *component.inputOperands.front()->getOwner() << "\n");
      LOG_DEBUG("component.outputResults.front().getOwner() = "
                << *component.outputResults.front().getOwner() << "\n");
    }
    return false;
  }

  if (!llvm::any_of(component.ops,
                    [](Operation *op) { return isa<linalg::MatmulOp>(op); })) {
    return false;
  }

  {
    // Current backend limitation: internal accumulator values may only
    // follow
    // the component's C/SCF value-flow edges. Final output consumers are
    // allowed. Remove this block when other internal consumers, including
    // matmul A/B operands, are supported by the backend.
    Value output = component.outputResults.front();
    if (llvm::any_of(component.getExternalUses(), [output](OpOperand *use) {
          return use->get() != output;
        })) {
      LOG_DEBUG("skip residency component: backend does not support internal "
                "accumulator observers");
      return false;
    }
  }
  return true;
}

void SplitMatmulPass::collectL0CReuseUnits(
    SmallVector<L0CReuseUnit> &l0cReusUnits, const ControlFlowGraph &graph) {
  llvm::DenseSet<Operation *> assigned;
  auto components = graph.getStronglyConnectedComponents(
      [](Operation *op) { return isa<linalg::MatmulOp>(op); });
  for (const auto &component : components) {

    if (!isValidL0CReusedComponent(component)) {
      continue;
    }
    L0CReuseUnit self;
    for (Operation *op : component.ops) {
      if (auto mm = dyn_cast<linalg::MatmulOp>(op)) {
        self.matmuls.push_back(mm);
        assigned.insert(mm);
      }
    }
    self.inputValue = component.inputOperands.front()->get();
    self.outputResult = component.outputResults.front();
    self.entryOp = component.entryOps.front();
    self.inputOpOperand = component.inputOperands.front();
    l0cReusUnits.push_back(std::move(self));
  }
  // Every matmul has exactly one owner, even if a component must split locally.
  getOperation()->walk([&](linalg::MatmulOp mm) {
    if (assigned.contains(mm)) {
      return;
    }
    L0CReuseUnit self;
    self.matmuls.push_back(mm);
    self.entryOp = mm.getOperation();
    if (mm.getNumResults() == 1) { // Avoid memref matmul
      self.inputValue = mm.getDpsInits()[0];
      self.outputResult = mm->getOpResults().front();
      self.inputOpOperand = &mm.getDpsInitsMutable()[0];
    }
    l0cReusUnits.push_back(std::move(self));
  });
}

LogicalResult
SplitMatmulPass::linkReuseL0COps(const SmallVector<L0CReuseUnit> &l0cReusUnits,
                                 SmallVector<L0CReuseLink> &l0cReuselinks) {
  DenseMap<Value, UnitId> inputToIndex;
  DenseMap<Value, UnitId> outputToIndex;
  for (unsigned unitId = 0; unitId < l0cReusUnits.size(); ++unitId) {
    if (l0cReusUnits[unitId].inputValue.hasOneUse()) {
      // only consider single link
      inputToIndex.try_emplace(l0cReusUnits[unitId].inputValue, unitId);
    }
    if (l0cReusUnits[unitId].outputResult.hasOneUse()) {
      // only consider single link
      outputToIndex.try_emplace(l0cReusUnits[unitId].outputResult, unitId);
    }
  }
  L0CReuseAdjacency adjacency;
  adjacency.next.assign(l0cReusUnits.size(), -1);
  adjacency.prev.assign(l0cReusUnits.size(), -1);
  // Pass 1: next — the unit whose input equals my output.
  for (unsigned now = 0; now < l0cReusUnits.size(); ++now) {
    auto outputOwner = outputToIndex.find(l0cReusUnits[now].outputResult);
    if (outputOwner == outputToIndex.end() || outputOwner->second != now) {
      continue;
    }
    auto nextConsumer = inputToIndex.find(l0cReusUnits[now].outputResult);
    if (nextConsumer != inputToIndex.end()) {
      adjacency.next[now] = static_cast<int>(nextConsumer->second);
    }
    // Pass 2: prev — the unit whose output equals my input, verified
    // bidirectionally via next.
    auto prevProducer = outputToIndex.find(l0cReusUnits[now].inputValue);
    if (prevProducer == outputToIndex.end()) {
      continue;
    }
    if (adjacency.next[prevProducer->second] == static_cast<int>(now)) {
      adjacency.prev[now] = static_cast<int>(prevProducer->second);
    }
  }

  SmallVector<bool> seen(l0cReusUnits.size(), false);
  for (unsigned seed = 0; seed < l0cReusUnits.size(); ++seed) {
    if (seen[seed]) {
      continue;
    }
    l0cReuselinks.push_back(
        L0CReuseLink::build(l0cReusUnits, adjacency, seen, seed));
  }
  return success();
}

static bool shouldSplitMatmul(Value input) {
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
    SmallVector<L0CReuseUnit> &l0cReusUnits,
    SmallVector<L0CReuseLink> &l0cReuselinks) {
  for (auto &unit : l0cReusUnits) {
    unit.isMNE = mayNotExecute(unit);
  }
  for (auto &link : l0cReuselinks) {
    link.shouldSplit = shouldSplitMatmul(link.inputValue);
    link.isMNE = llvm::all_of(
        link.units, [&](unsigned index) { return l0cReusUnits[index].isMNE; });
  }
}

void SplitMatmulPass::printAnalysis(
    const SmallVector<L0CReuseUnit> &l0cReusUnits,
    const SmallVector<L0CReuseLink> &l0cReuselinks) const {
  LOG_DEBUG("split-matmul: self-reused=" << l0cReusUnits.size()
                                         << " links=" << l0cReuselinks.size());
  for (const auto &[i, link] : llvm::enumerate(l0cReuselinks)) {
    LOG_DEBUG("link " << i << ": units=" << link.units.size() << " split="
                      << link.shouldSplit << " may-not-execute=" << link.isMNE
                      << "\n  inputValue: " << link.inputValue
                      << "\n  outputValue: " << link.outputResult);
  }
  LOG_DEBUG("------------------------------------------------------------");
  for (const auto &[i, unit] : llvm::enumerate(l0cReusUnits)) {
    LOG_DEBUG("L0C reuse unit "
              << i << ": matmuls=" << unit.matmuls.size()
              << " may-not-execute=" << unit.isMNE << " result-index="
              << (unit.outputResult ? unit.outputResult.getResultNumber() : -1)
              << "\n  inputValue: " << unit.inputValue
              << "\n  outputValue: " << unit.outputResult);
    for (auto mm : unit.matmuls) {
      LOG_DEBUG("   sub-matmul: " << mm);
    }
  }
}

static Value createZero(OpBuilder &builder, Location loc, Value shape) {
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
// Update link.inputValue to the new entry value used by the later MNE guard.
void SplitMatmulPass::splitMatmulChain(OpBuilder &builder, L0CReuseLink &link,
                                       OpOperand *inputUse, Operation *entry) {
  Value input = inputUse->get();
  builder.setInsertionPoint(entry);
  Value zero = createZero(builder, entry->getLoc(), input);
  inputUse->set(zero);

  // Save the original consumers before the new add uses the chain output.
  SmallVector<OpOperand *> users;
  for (OpOperand &use : link.outputResult.getUses()) {
    users.push_back(&use);
  }
  Operation *exit = link.outputResult.getDefiningOp();
  builder.setInsertionPointAfter(exit);
  auto type = cast<RankedTensorType>(input.getType());
  Operation *add;
  if (isa<FloatType>(type.getElementType())) {
    add =
        builder.create<arith::AddFOp>(exit->getLoc(), link.outputResult, input);
  } else {
    add =
        builder.create<arith::AddIOp>(exit->getLoc(), link.outputResult, input);
  }
  add->setAttr(CVPipeline::kAddFromMatmul, builder.getUnitAttr());
  for (OpOperand *use : users) {
    use->set(add->getResult(0));
  }
  link.inputValue = zero;
}

void SplitMatmulPass::setAttrForMatmulChain(
    OpBuilder &builder, const SmallVector<L0CReuseUnit> &l0cReusUnits,
    const L0CReuseLink &link) {
  for (unsigned id : link.units) {
    const auto &self = l0cReusUnits[id];
    for (auto mm : self.matmuls) {
      Operation *inner = mm->getParentOp();
      if (inner && !inner->hasAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr)) {
        inner->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr,
                       builder.getUnitAttr());
      }
    }
    Operation *outer = self.outputResult.getDefiningOp();
    if (self.isMNE && !outer->hasAttr(CVPipeline::kMayNotExecNPU)) {
      outer->setAttr(CVPipeline::kMayNotExecNPU, builder.getBoolAttr(true));
    }
  }
}

LogicalResult
SplitMatmulPass::splitMatmulChains(OpBuilder &builder,
                                   SmallVector<L0CReuseLink> &l0cReuselinks) {
  for (auto &link : l0cReuselinks) {
    if (link.shouldSplit && link.inputOpOperand && link.entryOp) {
      splitMatmulChain(builder, link, link.inputOpOperand, link.entryOp);
    }
  }
  return success();
}

// Track each matmul independently; guard only when all were skipped.
LogicalResult SplitMatmulPass::insertMNEGuardUB(OpBuilder &builder,
                                                const L0CReuseLink &link,
                                                int &coupledId,
                                                SSBufferManager &manager) {
  Operation *entry = link.entryOp;
  Location loc = entry->getLoc();
  SmallVector<Value> counters;
  for (auto mm : link.matmuls) {
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
  // Include the add introduced by the split stage among the uses to guard.
  SmallVector<OpOperand *> users;
  for (OpOperand &use : link.outputResult.getUses()) {
    users.push_back(&use);
  }
  builder.setInsertionPointAfter(link.outputResult.getDefiningOp());
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
  auto guard = builder.create<scf::IfOp>(loc, link.outputResult.getType(),
                                         allSkipped, true);
  builder.setInsertionPointToStart(&guard.getThenRegion().front());

  // Create zero fill value for else branch (same shape/type as output)
  auto resultType = dyn_cast<RankedTensorType>(link.outputResult.getType());
  auto elmType = resultType.getElementType();

  auto emptyOp = builder.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 resultType.getElementType());
  Value zeroFill;
  if (auto floatType = dyn_cast<FloatType>(elmType)) {
    APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
    auto zeroVal =
        builder.create<arith::ConstantFloatOp>(loc, floatType, zeroAPFloat)
            .getResult();
    auto fillOp =
        builder.create<linalg::FillOp>(loc, zeroVal, emptyOp.getResult());
    zeroFill = fillOp.getResult(0);
  } else if (auto intType = dyn_cast<IntegerType>(elmType)) {
    auto zeroVal =
        builder.create<arith::ConstantIntOp>(loc, intType, 0).getResult();
    auto fillOp =
        builder.create<linalg::FillOp>(loc, zeroVal, emptyOp.getResult());
    zeroFill = fillOp.getResult(0);
  }
  builder.create<scf::YieldOp>(loc, zeroFill);
  builder.setInsertionPointToStart(&guard.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, link.outputResult);
  for (OpOperand *use : users) {
    use->set(guard.getResult(0));
  }
  builder.setInsertionPointAfter(guard);
  return success();
}

void SplitMatmulPass::runOnOperation() {
  SmallVector<L0CReuseUnit> l0cReusUnits;
  SmallVector<L0CReuseLink> l0cReuselinks;
  int coupledId = 0;
  auto module = getOperation();
  bool findMayNotExec = false;
  module->walk([&](linalg::MatmulOp matmulOp) {
    if (matmulOp->hasAttr(CVPipeline::kMayNotExec)) {
      findMayNotExec = true;
    }
  });
  if (!findMayNotExec) {
    LOG_DEBUG("Matmul may not execute!");
    return;
  }

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
  LOG_DEBUG("Input mlir:" << module);

  // Stage 1: build the C value-flow graph and collect residency blocks.
  ControlFlowGraph graph(module, [](OpOperand &operand) {
    if (auto mm = dyn_cast<linalg::MatmulOp>(operand.getOwner())) {
      return &operand == &mm.getDpsInitsMutable()[0];
    }
    return true;
  });
  collectL0CReuseUnits(l0cReusUnits, graph);
  // Stage 2: link one-to-one chains, including isolated residency blocks.
  if (failed(linkReuseL0COps(l0cReusUnits, l0cReuselinks))) {
    LOG_DEBUG(
        "reuse chain must be acyclic, we need split each one in the chain.");
  }
  // Stage 3: analyze decisions before logging or changing IR.
  analyzeSplitAndMNE(l0cReusUnits, l0cReuselinks);
  LLVM_DEBUG(printAnalysis(l0cReusUnits, l0cReuselinks));
  // Stage 4: split selected chains.
  OpBuilder builder(module.getContext());
  if (failed(splitMatmulChains(builder, l0cReuselinks))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }
  // Stage 5: insert a counter per matmul and guard the chain contribution.
  SSBufferManager manager;
  for (const auto &link : l0cReuselinks) {
    if (link.isMNE) {
      if (failed(insertMNEGuardUB(builder, link, coupledId, manager))) {
        signalPassFailure();
        return;
      }
    }
  }

  // Stage 6: annotate each original residency boundary after all rewrites.
  for (const auto &link : l0cReuselinks) {
    setAttrForMatmulChain(builder, l0cReusUnits, link);
  }
  for (auto &self : l0cReusUnits) {
    if (auto markOpOpt = utils::getAnnotateOpWithAttr(self.outputResult,
                                                      kMatmulAtLeastOnceHint);
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
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::CVPipeline::createSplitMatmulPass() {
  return std::make_unique<SplitMatmulPass>();
}
