// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT

#include "ascend/include/TritonToLinalg/Passes.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-address-lookahead"

using namespace mlir;

namespace {
// This deliberately describes a small SIMD fp32 subset, not "tensor == V".
// The adapter tensorizes these scalar operations and the SIMD backend maps
// their singleton-tensor forms to elementwise Vector operations.
static bool isVectorAddressMath(Operation *op) {
  if (!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::MaxNumFOp,
           arith::MaximumFOp, arith::NegFOp, arith::MinNumFOp,
           arith::MinimumFOp, math::AbsFOp>(op))
    return false;
  auto ty = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return ty && ty.getShape() == ArrayRef<int64_t>{1} &&
         ty.getElementType().isF32();
}

static bool isSmallValue(Type ty) {
  if (ty.isIntOrIndexOrFloat())
    return true;
  auto tensor = dyn_cast<RankedTensorType>(ty);
  return tensor && tensor.hasStaticShape() && tensor.getNumElements() == 1;
}

// Clone only an explicit arithmetic/representation subset. In particular,
// never clone a load, a call, a region, or a potentially trapping int division.
static bool isGeometryOp(Operation *op) {
  if (!llvm::all_of(op->getResultTypes(), isSmallValue) ||
      op->getNumRegions() || !isMemoryEffectFree(op) || !isSpeculatable(op))
    return false;
  return isa<arith::ConstantOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
             arith::DivFOp, arith::NegFOp, arith::MaxNumFOp, arith::MaximumFOp,
             arith::MinNumFOp, arith::MinimumFOp, arith::AddIOp, arith::SubIOp,
             arith::MulIOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
             arith::IndexCastOp, arith::IndexCastUIOp, arith::SIToFPOp,
             arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp, arith::ExtFOp,
             arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
             tensor::EmptyOp, tensor::InsertOp, tensor::ExtractOp,
             tensor::FromElementsOp, math::AbsFOp>(op);
}

struct LookaheadPlan {
  scf::ForOp loop;
  llvm::SmallPtrSet<Operation *, 32> geometry;
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
                                       AddressSlice &slice) {
  if (value == loop.getInductionVar())
    return true;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != loop.getBody() || getInvariantInit(loop, arg))
      return false;
    return failure();
  }
  Operation *def = value.getDefiningOp();
  if (!loop->isProperAncestor(def))
    return false;
  if (def->getBlock() != loop.getBody() || !isGeometryOp(def))
    return failure();
  if (auto it = slice.ops.find(def); it != slice.ops.end())
    return it->second;

  bool varies = false;
  for (Value operand : def->getOperands()) {
    FailureOr<bool> operandVaries = collectGeometry(operand, loop, slice);
    if (failed(operandVaries))
      return failure();
    varies |= *operandVaries;
  }
  slice.ops.insert({def, varies});
  slice.hasVaryingVectorMath |= varies && isVectorAddressMath(def);
  return varies;
}

static bool isAddressView(Operation *op) {
  return isa_and_nonnull<memref::ReinterpretCastOp, memref::SubViewOp,
                         memref::CastOp>(op);
}

struct CopyCandidate {
  AddressSlice address;
  llvm::SmallPtrSet<Operation *, 8> setup;
};

static FailureOr<CopyCandidate> analyzeCopy(memref::CopyOp copy,
                                            scf::ForOp loop) {
  CopyCandidate candidate;
  Value source = copy.getSource();
  while (isAddressView(source.getDefiningOp())) {
    Operation *view = source.getDefiningOp();
    candidate.setup.insert(view);
    for (Value operand : view->getOperands().drop_front())
      if (failed(collectGeometry(operand, loop, candidate.address)))
        return failure();
    source = view->getOperand(0);
  }
  auto root = dyn_cast<BlockArgument>(source);
  if (!root || !isa<func::FuncOp>(root.getOwner()->getParentOp()) ||
      !candidate.address.hasVaryingVectorMath)
    return failure();

  // Recognize a regular input tile and its private destination. Keep the
  // destination setup, its views and all memory operations in their old order.
  // A tile initialized by a fill or shared by multiple writers is out of scope.
  Operation *userOnPath = copy;
  Value target = copy.getTarget();
  while (true) {
    Operation *def = target.getDefiningOp();
    if (!def || def->getBlock() != loop.getBody())
      return failure();
    for (Operation *user : target.getUsers()) {
      if (user == userOnPath)
        continue;
      auto toTensor = dyn_cast<bufferization::ToTensorOp>(user);
      if (!toTensor || toTensor->getBlock() != loop.getBody() ||
          !toTensor.getRestrict())
        return failure();
      candidate.setup.insert(user);
    }
    candidate.setup.insert(def);
    if (isa<memref::AllocOp>(def))
      return candidate;
    if (!isa<memref::SubViewOp, memref::CastOp>(def))
      return failure();
    userOnPath = def;
    target = def->getOperand(0);
  }
}

static FailureOr<LookaheadPlan> analyze(scf::ForOp loop) {
  LookaheadPlan plan;
  plan.loop = loop;
  // With signed unit-step semantics, iv < ub <= signed_max implies that
  // iv + 1 cannot overflow, including for the last iteration.
  auto step = loop.getStep().getDefiningOp<arith::ConstantIntOp>();
  if (!isa<IntegerType>(loop.getInductionVar().getType()) ||
      loop.getUnsignedCmp() || !step || step.value() != 1)
    return reject(loop, "requires a signed integer loop with unit step");

  llvm::DenseMap<Operation *, AddressSlice> candidates;
  llvm::SmallPtrSet<Operation *, 32> setup;
  for (auto copy : loop.getBody()->getOps<memref::CopyOp>()) {
    auto candidate = analyzeCopy(copy, loop);
    if (failed(candidate))
      continue;
    setup.insert(candidate->setup.begin(), candidate->setup.end());
    for (auto &entry : candidate->address.ops)
      setup.insert(entry.first);
    candidates.try_emplace(copy, std::move(candidate->address));
  }
  // Use the first eligible copy or adjacent bundle. Stop at a consumer or
  // unrelated operation so the lookahead can run before the first tile use.
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

  // Include small pure descendants sharing the geometry (e.g. hy/ly).
  // Stop at broadcasts, loaded data, recurrences, and large-tensor math.
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (plan.geometry.contains(&op) || !isGeometryOp(&op))
      continue;
    bool touchesGeometry = llvm::any_of(op.getOperands(), [&](Value v) {
      return v.getDefiningOp() && plan.geometry.contains(v.getDefiningOp());
    });
    if (!touchesGeometry)
      continue;
    AddressSlice trial;
    if (llvm::all_of(op.getOperands(), [&](Value value) {
          return succeeded(collectGeometry(value, loop, trial));
        })) {
      for (auto &entry : trial.ops)
        plan.geometry.insert(entry.first);
      plan.geometry.insert(&op);
    }
  }

  // The state frontier is every geometry result consumed outside geometry.
  // Follow original block order for deterministic signatures.
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (!plan.geometry.contains(&op))
      continue;
    for (Value result : op.getResults())
      if (llvm::any_of(result.getUses(), [&](OpOperand &use) {
            return !plan.geometry.contains(use.getOwner());
          }))
        plan.state.push_back(result);
  }
  if (plan.state.empty() || plan.state.size() > 16)
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
  SmallVector<Type> resultTypes(oldLoop.getResultTypes());
  Value nonempty = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 oldLoop.getLowerBound(),
                                                 oldLoop.getUpperBound());
  auto guard = builder.create<scf::IfOp>(loc, resultTypes, nonempty, true);
  for (Region *region : {&guard.getThenRegion(), &guard.getElseRegion()})
    if (!region->front().empty())
      region->front().back().erase();
  builder.setInsertionPointToStart(&guard.getThenRegion().front());
  SmallVector<Value> init(oldLoop.getInitArgs());
  llvm::append_range(init,
                     cloneGeometry(builder, plan, oldLoop.getLowerBound()));
  auto newLoop = builder.create<scf::ForOp>(
      loc, oldLoop.getLowerBound(), oldLoop.getUpperBound(), oldLoop.getStep(),
      init, [&](OpBuilder &body, Location bodyLoc, Value iv, ValueRange args) {
        IRMapping current;
        unsigned originalArgs = oldLoop.getNumRegionIterArgs();
        current.map(oldLoop.getInductionVar(), iv);
        current.map(oldLoop.getRegionIterArgs(), args.take_front(originalArgs));
        current.map(plan.state, args.drop_front(originalArgs));
        SmallVector<Type> stateTypes;
        for (Value value : plan.state)
          stateTypes.push_back(value.getType());

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
  builder.create<scf::YieldOp>(
      loc, newLoop.getResults().take_front(oldLoop.getNumResults()));
  builder.setInsertionPointToStart(&guard.getElseRegion().front());
  builder.create<scf::YieldOp>(loc, oldLoop.getInitArgs());
  oldLoop.replaceAllUsesWith(guard.getResults());
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
    // Narrow target scope: do not infer hardware mapping from tensor types
    // for SIMT or other architectures. The PoC is Ascend950 SIMD only.
    auto target =
        getOperation()->getAttrOfType<hacc::TargetAttr>("hacc.target");
    if (!target || !target.getTarget().getValue().starts_with("Ascend950"))
      return;
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops) {
      auto func = loop->getParentOfType<func::FuncOp>();
      auto mode = func ? func->getAttrOfType<StringAttr>("parallel_mode")
                       : StringAttr();
      if (!mode || mode.getValue() != "simd")
        continue;
      if (auto plan = analyze(loop); succeeded(plan))
        rewrite(*plan);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createVectorAddressLookaheadPass() {
  return std::make_unique<VectorAddressLookaheadPass>();
}
