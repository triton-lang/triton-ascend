/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */

#include "ascend/include/DynamicCVPipeline/OuterEventLoop.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"
#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::CVPipeline;

namespace {

constexpr unsigned kTask = 0, kInnerIV = 1, kPhase = 2, kBounds = 3;
constexpr int64_t kEventLimit = int64_t{1} << 30;

bool canRematerialize(Operation *op) {
  bool readsMemory;
  return op->getNumRegions() == 0 && isOuterEventScalarControl(op, readsMemory);
}

// A bound is a straight-line scalar slice of the task ID and immutable
// inputs. Clone the whole slice when initializing that phase, independently
// of the tensor payload.
LogicalResult collectBoundSlice(Value value, scf::ForOp outer,
                                llvm::SetVector<Operation *> &ops) {
  if (!value.getType().isIntOrIndexOrFloat() &&
      !isa<MemRefType>(value.getType()))
    return failure();
  Operation *def = value.getDefiningOp();
  if (!def)
    return success(value == outer.getInductionVar() ||
                   !outer.getRegion().isAncestor(value.getParentRegion()));
  if (!outer->isProperAncestor(def))
    return success();
  if (def->getBlock() != outer.getBody() || !canRematerialize(def))
    return failure();
  if (ops.contains(def))
    return success();
  for (Value operand : def->getOperands())
    if (failed(collectBoundSlice(operand, outer, ops)))
      return failure();
  ops.insert(def);
  return success();
}

LogicalResult addCarried(Value value, scf::ForOp outer,
                         llvm::SetVector<Value> &carried) {
  Operation *def = value.getDefiningOp();
  if (!def || !outer->isProperAncestor(def))
    return success();
  if (auto tensor = dyn_cast<RankedTensorType>(value.getType())) {
    if (!tensor.hasStaticShape())
      return failure();
    carried.insert(value);
    return success();
  }
  if (!canRematerialize(def))
    return failure();
  for (Value operand : def->getOperands())
    if (failed(addCarried(operand, outer, carried)))
      return failure();
  return success();
}

Value materialize(Value value, scf::ForOp outer, OpBuilder &builder,
                  IRMapping &mapping) {
  if (mapping.contains(value))
    return mapping.lookup(value);
  Operation *def = value.getDefiningOp();
  if (!def || !outer->isProperAncestor(def))
    return value;
  assert(canRematerialize(def) && "unplanned value crosses an event");
  for (Value operand : def->getOperands())
    materialize(operand, outer, builder, mapping);
  builder.clone(*def, mapping);
  return mapping.lookup(value);
}

void clonePayloadOp(Operation *op, scf::ForOp outer, OpBuilder &builder,
                    IRMapping &mapping) {
  op->walk([&](Operation *nested) {
    for (Value value : nested->getOperands()) {
      Operation *def = value.getDefiningOp();
      if (def && !op->isAncestor(def))
        materialize(value, outer, builder, mapping);
    }
  });
  builder.clone(*op, mapping);
}

Value constant(OpBuilder &b, Location loc, Type type, int64_t value) {
  return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(type, value));
}

arith::CmpIPredicate lessThan(scf::ForOp loop) {
  return loop->hasAttr("unsignedCmp") ? arith::CmpIPredicate::ult
                                      : arith::CmpIPredicate::slt;
}

IRMapping cloneBoundSlice(const OuterEventLoopPlan &plan, OpBuilder &builder,
                          Value task) {
  IRMapping mapping;
  scf::ForOp outer = plan.outer;
  mapping.map(outer.getInductionVar(), task);
  for (Operation *op : plan.boundSlices.front())
    builder.clone(*op, mapping);
  return mapping;
}

// The count loop reads the same metadata as the event loop. Saturation
// selects the original serial nest; it never truncates executed events.
Value countEvents(OuterEventLoopPlan &plan, OpBuilder &b) {
  auto outer = plan.outer;
  Location loc = outer.getLoc();
  Type wide = b.getI64Type();
  Value zero = constant(b, loc, wide, 0);
  Value one = constant(b, loc, wide, 1);
  Value overflow = constant(b, loc, wide, kEventLimit + 1);
  auto count =
      b.create<scf::ForOp>(loc, outer.getLowerBound(), outer.getUpperBound(),
                           outer.getStep(), ValueRange{zero});
  count->setAttr(kOuterEventControl, b.getUnitAttr());
  if (Attribute attr = outer->getAttr("unsignedCmp"))
    count->setAttr("unsignedCmp", attr);
  OpBuilder cb = createOuterEventBodyBuilder(*count.getBody());
  IRMapping mapping = cloneBoundSlice(plan, cb, count.getInductionVar());
  auto inner = plan.inner.front();
  Value bound = mapping.lookupOrDefault(inner.getUpperBound());
  Type type = bound.getType();
  if (!inner->hasAttr("unsignedCmp"))
    bound = cb.create<arith::MaxSIOp>(loc, bound, constant(cb, loc, type, 0));
  if (type.isIndex())
    bound = cb.create<arith::IndexCastUIOp>(loc, wide, bound);
  else if (type != wide)
    bound = cb.create<arith::ExtUIOp>(loc, wide, bound);
  bound = cb.create<arith::MinUIOp>(loc, bound, overflow);
  Value perTask;
  if (plan.serialFallback) {
    Value single =
        cb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, bound, one);
    perTask = cb.create<arith::SelectOp>(loc, single, one, overflow);
  } else {
    perTask = cb.create<arith::MaxUIOp>(loc, bound, one);
  }
  Value updated =
      cb.create<arith::AddIOp>(loc, count.getRegionIterArgs()[0], perTask);
  updated = cb.create<arith::MinUIOp>(loc, updated, overflow);
  cb.create<scf::YieldOp>(loc, updated);
  return count.getResult(0);
}

} // namespace

FailureOr<OuterEventLoopPlan>
OuterEventLoopPlan::analyze(scf::ForOp outer, ArrayRef<scf::ForOp> inner) {
  if (outer.getNumRegionIterArgs() || inner.empty() || inner.size() > 2) {
    outer.emitError("dynamic outer scheduling requires one or two inner loops "
                    "and no outer iter_args");
    return failure();
  }
  OuterEventLoopPlan plan;
  plan.outer = outer;
  plan.inner.append(inner.begin(), inner.end());
  plan.segments.resize(inner.size() + 1);
  plan.boundSlices.resize(inner.size());

  DenseMap<Operation *, unsigned> phaseOf;
  unsigned segment = 0;
  for (Operation &op : outer.getBody()->without_terminator()) {
    if (segment < inner.size() && &op == plan.inner[segment].getOperation()) {
      phaseOf[&op] = 2 * segment + 1;
      ++segment;
    } else {
      phaseOf[&op] = 2 * segment;
      plan.segments[segment].push_back(&op);
    }
  }

  Type ivType = plan.inner.front().getInductionVar().getType();
  if (auto integer = dyn_cast<IntegerType>(ivType);
      integer && integer.getWidth() > 64) {
    outer.emitError("unsupported event induction width");
    return failure();
  }
  for (auto [index, loop] : llvm::enumerate(plan.inner)) {
    if (loop.getInductionVar().getType() != ivType) {
      loop.emitError("inner loops must use the same induction type");
      return failure();
    }
    auto safe = [&](Operation *op) {
      return hasOnlyLocalMemoryEffects(op, outer, /*allowFree=*/true);
    };
    if (!llvm::all_of(plan.segments[index], safe) ||
        !llvm::all_of(loop.getBody()->without_terminator(),
                      [&](Operation &op) { return safe(&op); })) {
      loop.emitError("dynamic inner events and their prologue require "
                     "no external writes");
      return failure();
    }
    llvm::SetVector<Operation *> slice;
    if (failed(collectBoundSlice(loop.getUpperBound(), outer, slice))) {
      loop.emitError("dynamic bound requires a scalar slice of the outer IV "
                     "and loop-invariant inputs");
      return failure();
    }
    plan.boundSlices[index].append(slice.begin(), slice.end());
  }

  llvm::SmallDenseSet<Value, 4> metadataRoots;
  for (const auto &slice : plan.boundSlices)
    for (Operation *op : slice)
      if (auto load = dyn_cast<memref::LoadOp>(op))
        metadataRoots.insert(getOuterEventBufferRoot(load.getMemRef()));
  if (!metadataRoots.empty()) {
    for (Operation *op : plan.segments.back()) {
      auto effects = getEffectsRecursively(op);
      if (!effects || llvm::any_of(*effects, [&](const auto &effect) {
            return isa<MemoryEffects::Write>(effect.getEffect()) &&
                   (!effect.getValue() ||
                    metadataRoots.contains(
                        getOuterEventBufferRoot(effect.getValue())));
          })) {
        op->emitError("outer event metadata must stay invariant across tasks");
        return failure();
      }
    }
  }

  llvm::SetVector<Value> carried;
  for (Operation &op : outer.getBody()->without_terminator()) {
    bool isInner = llvm::is_contained(plan.inner, dyn_cast<scf::ForOp>(&op));
    for (Value value : op.getResults()) {
      if (plan.inner.size() == 1 && !isInner &&
          llvm::is_contained(plan.inner.front().getInitArgs(), value) &&
          llvm::all_of(value.getUsers(), [&](Operation *user) {
            return user == plan.inner.front().getOperation();
          }))
        continue;
      bool crosses = llvm::any_of(value.getUses(), [&](OpOperand &use) {
        Operation *root =
            outer.getBody()->findAncestorOpInBlock(*use.getOwner());
        return root && phaseOf.lookup(root) != phaseOf.lookup(&op);
      });
      if ((isInner || crosses) && failed(addCarried(value, outer, carried))) {
        op.emitError("unsupported value carried between outer events");
        return failure();
      }
    }
  }
  plan.carried.append(carried.begin(), carried.end());
  return plan;
}

void OuterEventLoopPlan::rewrite() {
  assert(inner.size() == 1 && "join inner loops before event normalization");
  Location loc = outer.getLoc();
  OpBuilder b(outer);
  Value total = countEvents(*this, b);
  Value limit = constant(b, loc, b.getI64Type(), kEventLimit);
  Value supported =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, total, limit);
  supported.getDefiningOp()->setAttr(kOuterEventControl, b.getUnitAttr());
  auto choice = b.create<scf::IfOp>(loc, supported, /*withElseRegion=*/true);
  OpBuilder fallbackBuilder(choice.elseYield());
  bool singleTile = static_cast<bool>(serialFallback);
  Operation *fallback;
  if (singleTile) {
    fallback = serialFallback.release();
    fallbackBuilder.insert(fallback);
    // Keep an empty tile's fill and selection on the Vector path.

    fallback->walk([](arith::SelectOp select) {
      auto isFilledFrom = [](Value value, Value init) {
        auto fill = value.getDefiningOp<linalg::FillOp>();
        return fill && fill.getOutputs().front() == init;
      };
      auto type = dyn_cast<RankedTensorType>(select.getType());
      if (!select.getCondition().getType().isInteger(1) || !type ||
          !type.hasStaticShape() ||
          (!isFilledFrom(select.getFalseValue(), select.getTrueValue()) &&
           !isFilledFrom(select.getTrueValue(), select.getFalseValue())))
        return;
      OpBuilder builder(select);
      auto maskType =
          RankedTensorType::get(type.getShape(), builder.getI1Type());
      Value mask = builder.create<tensor::SplatOp>(select.getLoc(), maskType,
                                                   select.getCondition());
      select.getConditionMutable().assign(mask);
    });
  } else {
    fallback = fallbackBuilder.clone(*outer.getOperation());
  }
  fallback->removeAttr(kOuterMainLoopHint);
  fallback->setAttr(kOuterEventFallback, b.getUnitAttr());
  b.setInsertionPoint(choice.thenYield());

  auto createEventLoop = [&](ValueRange initial, unsigned controlCount) {
    Value first = constant(b, loc, b.getI64Type(), 0);
    Value step = constant(b, loc, b.getI64Type(), 1);
    auto loop = b.create<scf::ForOp>(loc, first, total, step, initial);
    loop->setAttr(kExplicitMainLoop, b.getUnitAttr());
    loop->setAttr(kOuterEventControlCount, b.getI32IntegerAttr(controlCount));
    return loop;
  };

  Type ivType = inner.front().getInductionVar().getType();
  Value zero = constant(b, loc, ivType, 0);
  if (singleTile) {
    auto loop = createEventLoop(ValueRange{outer.getLowerBound()}, 1);
    OpBuilder body = createOuterEventBodyBuilder(*loop.getBody());
    auto original = inner.front();
    Value task = loop.getRegionIterArgs()[0];
    IRMapping mapping = cloneBoundSlice(*this, body, task);
    Value next = body.create<arith::AddIOp>(loc, task, outer.getStep());
    for (Operation &op : loop.getBody()->without_terminator())
      op.setAttr(kOuterEventControl, b.getUnitAttr());

    for (Operation *op : segments.front())
      if (!llvm::is_contained(boundSlices.front(), op))
        clonePayloadOp(op, outer, body, mapping);
    SmallVector<Value> seeds;
    for (Value init : original.getInitArgs())
      seeds.push_back(materialize(init, outer, body, mapping));
    mapping.map(original.getInductionVar(), zero);
    mapping.map(original.getRegionIterArgs(), seeds);
    for (Operation &op : original.getBody()->without_terminator())
      clonePayloadOp(&op, outer, body, mapping);
    mapping.map(original.getResults(),
                mapOuterEventValues(original.getYieldedValues(), mapping));
    for (Operation *op : segments.back())
      clonePayloadOp(op, outer, body, mapping);
    body.create<scf::YieldOp>(loc, next);
    outer.erase();
    return;
  }

  Value one = constant(b, loc, ivType, 1);
  Value phaseZero = constant(b, loc, b.getI32Type(), 0);
  SmallVector<Value> initial{outer.getLowerBound(), zero, phaseZero, zero};
  // A bound subexpression used by the payload is invariant within this task.
  // Keep it with the cursor instead of reloading the same metadata per KV tile.
  SmallVector<Value> cachedBounds;
  for (Operation *op : boundSlices.front()) {
    for (Value value : op->getResults()) {
      if (value != inner.front().getUpperBound() &&
          value.getType().isIntOrIndex() &&
          llvm::any_of(value.getUsers(), [&](Operation *user) {
            return inner.front()->isProperAncestor(user);
          }))
        cachedBounds.push_back(value);
    }
  }
  unsigned cachedBegin = initial.size();
  for (Value value : cachedBounds)
    initial.push_back(constant(b, loc, value.getType(), 0));
  unsigned controlCount = initial.size();

  for (Value value : carried) {
    auto type = cast<RankedTensorType>(value.getType());
    initial.push_back(
        b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(),
                                  ValueRange{}, type.getEncoding()));
  }

  auto loop = createEventLoop(initial, controlCount);

  Block *body = loop.getBody();
  if (!body->empty() && body->back().hasTrait<OpTrait::IsTerminator>())
    body->back().erase();
  b.setInsertionPointToEnd(body);
  auto args = loop.getRegionIterArgs();
  Value task = args[kTask], j = args[kInnerIV], phase = args[kPhase];
  SmallVector<Value> next(args);
  IRMapping live;
  live.map(outer.getInductionVar(), task);
  DenseMap<Value, unsigned> slots;
  for (auto [index, value] : llvm::enumerate(carried)) {
    unsigned slot = controlCount + index;
    slots[value] = slot;
    live.map(value, args[slot]);
  }

  auto loadBound = [&](Value initialize) -> Value {
    SmallVector<Value> previous{args[kBounds]};
    for (unsigned slot = 0; slot < cachedBounds.size(); ++slot)
      previous.push_back(args[cachedBegin + slot]);
    auto guard = createOuterEventGuard(
        b, loc, initialize, previous, [&](OpBuilder &entry) {
          IRMapping mapping = cloneBoundSlice(*this, entry, task);
          SmallVector<Value> loaded{
              mapping.lookupOrDefault(inner.front().getUpperBound())};
          llvm::append_range(loaded,
                             mapOuterEventValues(cachedBounds, mapping));
          return loaded;
        });
    for (unsigned slot = 0; slot < cachedBounds.size(); ++slot)
      next[cachedBegin + slot] = guard.getResult(slot + 1);
    return guard.getResult(0);
  };

  auto markControl = [&]() {
    for (Operation &op : *body)
      op.setAttr(kOuterEventControl, b.getUnitAttr());
  };

  // One logical event per KV tile. Initialize Q and the online reduction
  // on the first event, and store the result on the last. A zero-trip
  // task still has one event to preserve its original output.
  auto original = inner.front();
  Value first =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, phase, phaseZero);
  Value bound = loadBound(first);
  Value work = b.create<arith::CmpIOp>(loc, lessThan(original), zero, bound);
  Value advancedJ = b.create<arith::AddIOp>(loc, j, one);
  Value hasNext =
      b.create<arith::CmpIOp>(loc, lessThan(original), advancedJ, bound);
  Value last = b.create<arith::CmpIOp>(loc,
                                       original->hasAttr("unsignedCmp")
                                           ? arith::CmpIPredicate::uge
                                           : arith::CmpIPredicate::sge,
                                       advancedJ, bound);
  Value advancedTask = b.create<arith::AddIOp>(loc, task, outer.getStep());
  Value inTask = constant(b, loc, b.getI32Type(), 1);
  next[kTask] = b.create<arith::SelectOp>(loc, hasNext, task, advancedTask);
  next[kInnerIV] = b.create<arith::SelectOp>(loc, hasNext, advancedJ, zero);
  next[kPhase] = b.create<arith::SelectOp>(loc, hasNext, inTask, phaseZero);
  next[kBounds] = bound;
  markControl();

  SmallVector<Value> entryKeys;
  for (Value value : carried)
    if (value.getDefiningOp() != original.getOperation())
      entryKeys.push_back(value);
  unsigned taskTensorCount = entryKeys.size();

  llvm::append_range(entryKeys, original.getResults());
  auto entry = createOuterEventGuard(
      b, loc, first, mapOuterEventValues(entryKeys, live),
      [&](OpBuilder &builder) {
        IRMapping mapping(live);
        for (Operation *op : segments.front())
          clonePayloadOp(op, outer, builder, mapping);
        auto initialized = mapOuterEventValues(
            ArrayRef<Value>(entryKeys).take_front(taskTensorCount), mapping);
        for (Value value : original.getInitArgs())
          initialized.push_back(materialize(value, outer, builder, mapping));
        return initialized;
      });

  IRMapping current(live);
  current.map(entryKeys, entry.getResults());
  for (Value key : ArrayRef<Value>(entryKeys).take_front(taskTensorCount))
    next[slots.lookup(key)] = current.lookup(key);
  auto tile = createOuterEventGuard(
      b, loc, work, entry.getResults().drop_front(taskTensorCount),
      [&](OpBuilder &builder) {
        IRMapping mapping(current);
        mapping.map(original.getInductionVar(), j);
        mapping.map(original.getRegionIterArgs(),
                    entry.getResults().drop_front(taskTensorCount));
        if (!cachedBounds.empty()) {
          auto metadata = createOuterEventGuard(
              builder, loc, first, args.slice(cachedBegin, cachedBounds.size()),
              [&](OpBuilder &fresh) {
                return mapOuterEventValues(cachedBounds,
                                           cloneBoundSlice(*this, fresh, task));
              });
          mapping.map(cachedBounds, metadata.getResults());
        }
        for (Operation &op : original.getBody()->without_terminator())
          clonePayloadOp(&op, outer, builder, mapping);
        return mapOuterEventValues(original.getYieldedValues(), mapping);
      });

  current.map(original.getResults(), tile.getResults());
  for (Value key : original.getResults())
    next[slots.lookup(key)] = current.lookup(key);

  auto finish = b.create<scf::IfOp>(loc, last, /*withElseRegion=*/false);
  OpBuilder finishBuilder =
      createOuterEventBodyBuilder(finish.getThenRegion().front());
  for (Operation *op : segments.back())
    clonePayloadOp(op, outer, finishBuilder, current);
  finishBuilder.create<scf::YieldOp>(loc);
  b.create<scf::YieldOp>(loc, next);
  outer.erase();
}

Value mlir::CVPipeline::getOuterEventBufferRoot(Value value) {
  if (value)
    while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
      value = view.getViewSource();
  return value;
}

Operation *mlir::CVPipeline::getOuterEventScratchAllocation(Value value) {
  Value root = getOuterEventBufferRoot(value);
  Operation *def = root ? root.getDefiningOp() : nullptr;
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(def) ? def
                                                                 : nullptr;
}

bool mlir::CVPipeline::hasOnlyLocalMemoryEffects(Operation *op, scf::ForOp loop,
                                                 bool allowFree) {
  auto effects = getEffectsRecursively(op);
  return effects && llvm::all_of(*effects, [&](const auto &effect) {
           if (isa<MemoryEffects::Read>(effect.getEffect()))
             return true;
           if (!allowFree && isa<MemoryEffects::Free>(effect.getEffect()))
             return false;
           auto *alloc = getOuterEventScratchAllocation(effect.getValue());
           return alloc && loop->isProperAncestor(alloc);
         });
}

OpBuilder mlir::CVPipeline::createOuterEventBodyBuilder(Block &block) {
  if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>())
    block.back().erase();
  return OpBuilder::atBlockEnd(&block);
}

SmallVector<Value>
mlir::CVPipeline::mapOuterEventValues(ValueRange values,
                                      const IRMapping &mapping) {
  return llvm::map_to_vector(
      values, [&](Value value) { return mapping.lookupOrDefault(value); });
}

scf::IfOp mlir::CVPipeline::createOuterEventIf(
    OpBuilder &builder, Location loc, Value condition, TypeRange types,
    function_ref<SmallVector<Value>(OpBuilder &, unsigned)> emitBranch) {
  auto branch = builder.create<scf::IfOp>(loc, types, condition, true);
  for (unsigned side : {0u, 1u}) {
    OpBuilder body =
        createOuterEventBodyBuilder(side == 0 ? branch.getThenRegion().front()
                                              : branch.getElseRegion().front());
    body.create<scf::YieldOp>(loc, emitBranch(body, side));
  }
  return branch;
}

scf::IfOp mlir::CVPipeline::createOuterEventGuard(
    OpBuilder &builder, Location loc, Value condition, ValueRange previous,
    function_ref<SmallVector<Value>(OpBuilder &)> emit) {
  return createOuterEventIf(builder, loc, condition, TypeRange(previous),
                            [&](OpBuilder &body, unsigned side) {
                              return side == 0 ? emit(body)
                                               : SmallVector<Value>(previous);
                            });
}

bool mlir::CVPipeline::isOuterEventScalarControl(Operation *root,
                                                 bool &readsMemory,
                                                 bool allowLoops) {
  readsMemory = false;
  auto scalarOrAddress = [](Type type) {
    return type.isIntOrIndexOrFloat() || isa<MemRefType>(type);
  };
  return !root->walk([&](Operation *op) {
                if (isSyncOp(op) || op->hasAttr(triton::kMemrefExtVolatile) ||
                    !llvm::all_of(op->getOperandTypes(), scalarOrAddress) ||
                    !llvm::all_of(op->getResultTypes(), scalarOrAddress))
                  return WalkResult::interrupt();
                if (auto load = dyn_cast<memref::LoadOp>(op)) {
                  if (llvm::any_of(
                          load.getResult().getUsers(), [](Operation *user) {
                            return user->hasAttr(triton::kMemrefExtVolatile);
                          }))
                    return WalkResult::interrupt();
                  readsMemory = true;
                } else if (!isa<scf::IfOp, scf::YieldOp>(op) &&
                           !(allowLoops &&
                             isa<scf::ForOp, scf::WhileOp, scf::ConditionOp>(
                                 op)) &&
                           (op->getNumRegions() || !isMemoryEffectFree(op))) {
                  return WalkResult::interrupt();
                }
                return WalkResult::advance();
              })
              .wasInterrupted();
}
