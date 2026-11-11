// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT

#include "ascend/include/TritonToLinalg/ExecutionPolicy.h"
#include "ascend/include/TritonToLinalg/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-address-lookahead"

using namespace mlir;
using namespace mlir::triton;

namespace {
constexpr StringLiteral kLookahead = "tt.vector_address_lookahead";

// The pointer sets grow dynamically beyond their inline capacities.
constexpr unsigned kInlineAddressOpCapacity = 32;
constexpr unsigned kInlineCopySetupCapacity = 8;

// Bound the number of extra SSA values carried by the rewritten loop
// when adding to iter_arg.
constexpr unsigned kMaxLoopStateValues = 16;

static bool isSmallState(Type type) {
  if (type.isIntOrIndexOrFloat())
    return true;
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape() && tensor.getNumElements() == 1 &&
         tensor.getElementType().isIntOrIndexOrFloat();
}

static llvm::SetVector<Value> getDependencies(Operation *op) {
  llvm::SetVector<Value> inputs;
  inputs.insert(op->operand_begin(), op->operand_end());
  getUsedValuesDefinedAbove(op->getRegions(), inputs);
  return inputs;
}

// A tensor produced from a mutable memref may change across the insertion
// point. Only reuse external tensors with an SSA-only provenance.
static bool isStableTensor(Value value, llvm::DenseSet<Value> &visited) {
  if (isa<BaseMemRefType>(value.getType()))
    return false;
  if (!isa<TensorType>(value.getType()) || !visited.insert(value).second)
    return true;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isa<func::FuncOp>(arg.getOwner()->getParentOp());
  Operation *def = value.getDefiningOp();
  return canLookAhead(def) && llvm::all_of(getDependencies(def), [&](Value v) {
           return isStableTensor(v, visited);
         });
}

static bool isStableTensor(Value value) {
  llvm::DenseSet<Value> visited;
  return isStableTensor(value, visited);
}

struct LookaheadPlan {
  scf::ForOp loop;
  llvm::SmallPtrSet<Operation *, kInlineAddressOpCapacity> geometry;
  Operation *anchor = nullptr;
  SmallVector<Value> state;
};

static LogicalResult reject(scf::ForOp loop, StringRef reason) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": " << reason << " at "
                          << loop.getLoc() << '\n');
  return failure();
}

// A loop-carried argument is available in the prologue and in the next
// iteration when the loop passes it through unchanged.
static Value getInvariantInit(scf::ForOp loop, BlockArgument arg) {
  if (arg.getOwner() != loop.getBody() || arg == loop.getInductionVar())
    return {};
  unsigned idx = arg.getArgNumber() - 1;
  if (loop.getBody()->getTerminator()->getOperand(idx) != arg)
    return {};
  return loop.getInitArgs()[idx];
}

struct AddressSlice {
  // The map also caches whether an operation depends on this loop's IV.
  llvm::DenseMap<Operation *, bool> ops;
  bool hasVaryingVectorMath = false;
};

static FailureOr<bool> collectGeometry(Value value, scf::ForOp loop,
                                       AddressSlice &slice,
                                       ExecutionMode mode) {
  if (value == loop.getInductionVar())
    return true;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != loop.getBody()) {
      if (isStableTensor(arg))
        return false;
    } else if (Value init = getInvariantInit(loop, arg)) {
      if (isStableTensor(init))
        return false;
    }
    return failure();
  }
  Operation *def = value.getDefiningOp();
  if (!loop->isProperAncestor(def)) {
    if (isStableTensor(value))
      return false;
    return failure();
  }
  if (def->getBlock() != loop.getBody() || !canLookAhead(def))
    return failure();
  if (auto it = slice.ops.find(def); it != slice.ops.end())
    return it->second;

  bool varies = false;
  for (Value operand : getDependencies(def)) {
    FailureOr<bool> operandVaries = collectGeometry(operand, loop, slice, mode);
    if (failed(operandVaries))
      return failure();
    varies |= *operandVaries;
  }
  slice.ops.insert({def, varies});
  slice.hasVaryingVectorMath |=
      varies && classifyExecution(def, mode) == ExecutionKind::Vector;
  return varies;
}

static bool isAddressView(Operation *op) {
  return isa_and_nonnull<memref::ReinterpretCastOp, memref::SubViewOp,
                         memref::CastOp>(op);
}

struct CopyCandidate {
  AddressSlice address;
  llvm::SmallPtrSet<Operation *, kInlineCopySetupCapacity> setup;
};

static FailureOr<CopyCandidate>
analyzeCopy(memref::CopyOp copy, scf::ForOp loop, ExecutionMode mode) {
  CopyCandidate candidate;

  // Walk back through source views and collect the computations of their
  // dynamic offsets, sizes, and strides. Record the views as copy setup.
  Value source = copy.getSource();
  while (isAddressView(source.getDefiningOp())) {
    Operation *view = source.getDefiningOp();
    candidate.setup.insert(view);
    for (Value operand : view->getOperands().drop_front())
      if (failed(collectGeometry(operand, loop, candidate.address, mode)))
        return failure();
    source = view->getOperand(0);
  }
  // Require a function input and Vector math that varies with the loop IV.
  auto root = dyn_cast<BlockArgument>(source);
  if (!root || !isa<func::FuncOp>(root.getOwner()->getParentOp()) ||
      !candidate.address.hasVaryingVectorMath)
    return failure();

  // Follow the destination views back to an allocation in this loop.
  // The allocation, views, and copy will keep their relative order.
  Operation *userOnPath = copy;
  Value destination = copy.getTarget();
  while (true) {
    Operation *def = destination.getDefiningOp();
    if (!def || def->getBlock() != loop.getBody())
      return failure();
    // At each value, allow the next operation on the copy path and
    // same-block restricted tensor conversions. A fill or another writer
    // would introduce another user and make the copy ineligible.
    for (Operation *user : destination.getUsers()) {
      if (user == userOnPath)
        continue;
      auto toTensor = dyn_cast<bufferization::ToTensorOp>(user);
      if (!toTensor || toTensor->getBlock() != loop.getBody() ||
          !toTensor.getRestrict())
        return failure();
      candidate.setup.insert(user);
    }
    // The allocation completes the candidate. Otherwise continue through
    // a destination view that preserves the underlying buffer.
    candidate.setup.insert(def);
    if (isa<memref::AllocOp>(def))
      return candidate;
    if (!isa<memref::SubViewOp, memref::CastOp>(def))
      return failure();
    userOnPath = def;
    destination = def->getOperand(0);
  }
}

static FailureOr<LookaheadPlan> analyze(scf::ForOp loop, ExecutionMode mode) {
  LookaheadPlan plan;
  plan.loop = loop;

  // Check the loop form used by the rewrite to compute the next IV. With a
  // signed unit step, iv < ub <= signed_max ensures iv + 1 cannot overflow.
  auto step = loop.getStep().getDefiningOp<arith::ConstantIntOp>();
  if (!isa<IntegerType>(loop.getInductionVar().getType()) ||
      loop.getUnsignedCmp() || !step || step.value() != 1)
    return reject(loop, "requires a signed integer loop with unit step");

  // Analyze each copy and collect the setup and address operations that
  // may appear between copies in the same group.
  llvm::DenseMap<Operation *, AddressSlice> candidates;
  llvm::SmallPtrSet<Operation *, kInlineAddressOpCapacity> setup;
  for (auto copy : loop.getBody()->getOps<memref::CopyOp>()) {
    auto candidate = analyzeCopy(copy, loop, mode);
    if (failed(candidate))
      continue;
    setup.insert(candidate->setup.begin(), candidate->setup.end());
    for (auto &entry : candidate->address.ops)
      setup.insert(entry.first);
    candidates.try_emplace(copy, std::move(candidate->address));
  }

  // Select the first eligible copy group and merge its address slices. The
  // last copy becomes the insertion point for the next iteration's address
  // calculation. A consumer or other unrelated operation ends the group.
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto it = candidates.find(&op);
    if (it != candidates.end()) {
      plan.anchor = &op;
      for (auto &entry : it->second.ops)
        plan.geometry.insert(entry.first);
    } else if (plan.anchor && !setup.contains(&op) &&
               !isa<arith::ConstantOp>(&op)) {
      break;
    }
  }
  if (!plan.anchor)
    return reject(loop, "no loop-varying Vector address producer");

  // Include small, safe computations that consume the selected geometry
  // (e.g. hy/ly). Add their dependencies only if the entire slice is safe
  // to compute ahead.
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (plan.geometry.contains(&op) || !canLookAhead(&op) ||
        !llvm::all_of(op.getResultTypes(), isSmallState))
      continue;
    auto inputs = getDependencies(&op);
    bool touchesGeometry = llvm::any_of(inputs, [&](Value v) {
      return v.getDefiningOp() && plan.geometry.contains(v.getDefiningOp());
    });
    if (!touchesGeometry)
      continue;
    AddressSlice trial;
    if (llvm::all_of(inputs, [&](Value value) {
          return succeeded(collectGeometry(value, loop, trial, mode));
        })) {
      for (auto &entry : trial.ops)
        plan.geometry.insert(entry.first);
      plan.geometry.insert(&op);
    }
  }

  // Carry geometry results that the rest of the loop still consumes.
  // Visit producers in block order and results in result order to keep
  // the added loop arguments deterministic.
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (!plan.geometry.contains(&op))
      continue;
    for (Value result : op.getResults())
      if (llvm::any_of(result.getUses(), [&](OpOperand &use) {
            // Uses captured by another cloned op's region are internal.
            for (Operation *owner = use.getOwner(); owner;
                 owner = owner->getParentOp())
              if (plan.geometry.contains(owner))
                return false;
            return true;
          }))
        plan.state.push_back(result);
  }

  // Require a nonempty, bounded set of scalar or single-element tensor values
  // before adding them as loop arguments.
  if (plan.state.empty() || plan.state.size() > kMaxLoopStateValues ||
      !llvm::all_of(plan.state,
                    [](Value v) { return isSmallState(v.getType()); }))
    return reject(loop, "unsupported geometry state size");
  return plan;
}

static SmallVector<Value> cloneGeometry(OpBuilder &builder,
                                        const LookaheadPlan &plan, Value iv) {
  IRMapping mapping;
  auto loop = plan.loop;
  mapping.map(loop.getInductionVar(), iv);
  for (BlockArgument arg : loop.getRegionIterArgs())
    if (Value init = getInvariantInit(loop, arg))
      mapping.map(arg, init);
  for (Operation &op : loop.getBody()->without_terminator())
    if (plan.geometry.contains(&op))
      builder.clone(op, mapping);
  SmallVector<Value> state;
  for (Value value : plan.state)
    state.push_back(mapping.lookup(value));
  return state;
}

static void rewrite(const LookaheadPlan &plan) {
  auto oldLoop = plan.loop;
  OpBuilder builder(oldLoop);
  Location loc = oldLoop.getLoc();
  SmallVector<Type> stateTypes;
  for (Value value : plan.state)
    stateTypes.push_back(value.getType());

  Value nonempty = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 oldLoop.getLowerBound(),
                                                 oldLoop.getUpperBound());
  // Guard only the prologue. Keep the loop at its original nesting level
  // for downstream CV scheduling and synchronization.
  auto prologue = builder.create<scf::IfOp>(loc, stateTypes, nonempty, true);
  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(prologue.thenBlock());
    builder.create<scf::YieldOp>(
        loc, cloneGeometry(builder, plan, oldLoop.getLowerBound()));

    builder.setInsertionPointToStart(prologue.elseBlock());
    SmallVector<Value> unused;
    for (Type type : stateTypes) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        unused.push_back(builder.create<tensor::EmptyOp>(
            loc, tensor.getShape(), tensor.getElementType()));
      else
        unused.push_back(builder.create<arith::ConstantOp>(
            loc, type, builder.getZeroAttr(type)));
    }
    builder.create<scf::YieldOp>(loc, unused);
  }
  SmallVector<Value> init(oldLoop.getInitArgs());
  llvm::append_range(init, prologue.getResults());
  auto newLoop = builder.create<scf::ForOp>(
      loc, oldLoop.getLowerBound(), oldLoop.getUpperBound(), oldLoop.getStep(),
      init, [&](OpBuilder &body, Location bodyLoc, Value iv, ValueRange args) {
        IRMapping current;
        unsigned originalArgs = oldLoop.getNumRegionIterArgs();
        current.map(oldLoop.getInductionVar(), iv);
        current.map(oldLoop.getRegionIterArgs(), args.take_front(originalArgs));
        current.map(plan.state, args.drop_front(originalArgs));
        scf::IfOp next;
        for (Operation &op : oldLoop.getBody()->without_terminator()) {
          if (plan.geometry.contains(&op))
            continue;
          body.clone(op, current);
          if (&op != plan.anchor)
            continue;

          Value nextIV =
              body.create<arith::AddIOp>(bodyLoc, iv, oldLoop.getStep());
          Value hasNext =
              body.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::slt,
                                         nextIV, oldLoop.getUpperBound());
          next = body.create<scf::IfOp>(bodyLoc, stateTypes, hasNext, true);
          OpBuilder::InsertionGuard insertionGuard(body);
          body.setInsertionPointToStart(&next.getThenRegion().front());
          body.create<scf::YieldOp>(bodyLoc, cloneGeometry(body, plan, nextIV));
          body.setInsertionPointToStart(&next.getElseRegion().front());
          body.create<scf::YieldOp>(bodyLoc, args.drop_front(originalArgs));
        }
        SmallVector<Value> yielded;
        for (Value value : oldLoop.getBody()->getTerminator()->getOperands())
          yielded.push_back(current.lookupOrDefault(value));
        llvm::append_range(yielded, next.getResults());
        body.create<scf::YieldOp>(bodyLoc, yielded);
      });
  newLoop->setAttrs(oldLoop->getAttrs());
  newLoop->setAttr(kLookahead, builder.getUnitAttr());
  oldLoop.replaceAllUsesWith(
      newLoop.getResults().take_front(oldLoop.getNumResults()));
  oldLoop.erase();
}

struct VectorAddressLookaheadPass
    : public PassWrapper<VectorAddressLookaheadPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorAddressLookaheadPass)
  StringRef getArgument() const final { return "vector-address-lookahead"; }
  StringRef getDescription() const final {
    return "Look ahead pure SIMD Vector address computation after current GM "
           "loads";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect,
                    memref::MemRefDialect, bufferization::BufferizationDialect,
                    math::MathDialect>();
  }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops) {
      if (loop->hasAttr(kLookahead))
        continue;
      auto func = loop->getParentOfType<func::FuncOp>();
      auto mode = func ? func->getAttrOfType<StringAttr>("parallel_mode")
                       : StringAttr();
      // The execution policy applies to functions explicitly marked SIMD.
      if (!mode || mode.getValue() != "simd")
        continue;
      if (auto plan = analyze(loop, ExecutionMode::SIMD); succeeded(plan))
        rewrite(*plan);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createVectorAddressLookaheadPass() {
  return std::make_unique<VectorAddressLookaheadPass>();
}
