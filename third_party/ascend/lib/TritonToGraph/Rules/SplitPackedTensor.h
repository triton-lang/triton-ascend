// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_ASCEND_GRAPH_SPLIT_PACKED_TENSOR_H
#define TRITON_ASCEND_GRAPH_SPLIT_PACKED_TENSOR_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace mlir::triton::split_packed_tensor {

inline RankedTensorType integerTensor(Value value, unsigned width = 0) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getEncoding())
    return {};
  auto element = dyn_cast<IntegerType>(type.getElementType());
  if (!element || element.getWidth() > 64 ||
      (width && element.getWidth() != width))
    return {};
  return type;
}

inline bool constant(Value value, uint64_t expected) {
  if (!integerTensor(value))
    return false;
  APInt bits;
  return matchPattern(value, m_ConstantInt(&bits)) &&
         bits.getZExtValue() == expected;
}

// This proves element values only. It deliberately does not prove one-hotness.
inline bool isBooleanMask(Value value, unsigned depth = 0) {
  auto type = integerTensor(value);
  if (!type || depth > 12)
    return false;
  if (type.getElementType().isInteger(1) || constant(value, 0) ||
      constant(value, 1))
    return true;
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    return range.getStart() == 0 && range.getEnd() == 2;
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return isBooleanMask(ext.getIn(), depth + 1);
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
    auto sourceType = integerTensor(ext.getIn());
    return sourceType && !sourceType.getElementType().isInteger(1) &&
           isBooleanMask(ext.getIn(), depth + 1);
  }
  if (auto expand = value.getDefiningOp<ExpandDimsOp>())
    return isBooleanMask(expand.getSrc(), depth + 1);
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return isBooleanMask(broadcast.getSrc(), depth + 1);
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return sub.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
           constant(sub.getLhs(), 1) && isBooleanMask(sub.getRhs(), depth + 1);
  return false;
}

// The range's non-singleton axis must remain the reduction axis. Broadcasting
// some other axis is harmless; broadcasting a singleton to the reduction axis
// would duplicate a one and invalidate the no-carry argument.
inline bool isOneHotAlongAxis(Value value, int axis, unsigned depth = 0) {
  auto type = integerTensor(value);
  if (!type || depth > 12 || axis < 0 || axis >= type.getRank() ||
      type.getDimSize(axis) != 2)
    return false;
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    return type.getRank() == 1 && axis == 0 && range.getStart() == 0 &&
           range.getEnd() == 2;
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return isOneHotAlongAxis(ext.getIn(), axis, depth + 1);
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
    auto sourceType = integerTensor(ext.getIn());
    return sourceType && !sourceType.getElementType().isInteger(1) &&
           isOneHotAlongAxis(ext.getIn(), axis, depth + 1);
  }
  if (auto expand = value.getDefiningOp<ExpandDimsOp>()) {
    int inserted = expand.getAxis();
    return axis != inserted &&
           isOneHotAlongAxis(expand.getSrc(), axis - (axis > inserted),
                             depth + 1);
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return isOneHotAlongAxis(broadcast.getSrc(), axis, depth + 1);
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return sub.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
           constant(sub.getLhs(), 1) &&
           isOneHotAlongAxis(sub.getRhs(), axis, depth + 1);
  return false;
}

inline Value booleanMultiplier(arith::MulIOp mul) {
  if (!integerTensor(mul.getResult(), 64) ||
      mul.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return {};
  if (isBooleanMask(mul.getRhs()))
    return mul.getRhs();
  if (isBooleanMask(mul.getLhs()))
    return mul.getLhs();
  return {};
}

inline bool isOneHotSum(ReduceOp reduce) {
  if (reduce->getNumOperands() != 1 || reduce->getNumResults() != 1 ||
      !integerTensor(reduce->getResult(0), 64))
    return false;
  Value input = reduce->getOperand(0);
  auto inputType = integerTensor(input, 64);
  if (!inputType || inputType.getRank() < 2 || reduce.getAxis() < 0 ||
      reduce.getAxis() >= inputType.getRank() ||
      inputType.getDimSize(reduce.getAxis()) != 2)
    return false;
  auto mul = input.getDefiningOp<arith::MulIOp>();
  if (!mul || mul.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
      !(isOneHotAlongAxis(mul.getLhs(), reduce.getAxis()) ||
        isOneHotAlongAxis(mul.getRhs(), reduce.getAxis())))
    return false;
  Block &body = reduce.getCombineOp().front();
  if (body.getNumArguments() != 2 ||
      !llvm::hasSingleElement(body.without_terminator()))
    return false;
  auto add = dyn_cast<arith::AddIOp>(body.front());
  auto yield = dyn_cast<ReduceReturnOp>(body.getTerminator());
  return add && yield &&
         add.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         yield->getNumOperands() == 1 &&
         yield->getOperand(0) == add.getResult() &&
         ((add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1)) ||
          (add.getLhs() == body.getArgument(1) &&
           add.getRhs() == body.getArgument(0)));
}

struct Halves {
  Value low;
  Value high;
};

class Splitter {
public:
  explicit Splitter(MLIRContext *context) : builder(context) {}

  void run(Operation *root) {
    // Snapshot the input. Newly generated operations are never split again.
    SmallVector<Operation *> original;
    root->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { original.push_back(op); });
    selectConnectedGraph(original);
    if (selected.empty())
      return;
    for (Operation *op : original)
      visit(op);

    // Keep the old graph until all proofs have inspected it. Only users outside
    // that graph require an i64 reconstruction, shared at the value definition.
    for (Operation *op : original) {
      if (!converted.contains(op))
        continue;
      for (Value result : op->getResults()) {
        if (!integerTensor(result, 64))
          continue;
        SmallVector<OpOperand *> externalUses;
        for (OpOperand &use : result.getUses())
          if (!converted.contains(use.getOwner()))
            externalUses.push_back(&use);
        if (externalUses.empty())
          continue;
        builder.setInsertionPoint(op);
        Value wide = pack(result, halves.lookup(result));
        for (OpOperand *use : externalUses)
          use->set(wide);
      }
    }
    // Erase only converted, now-dead originals. Never clone or delete an
    // unsupported producer such as a load or a loop carrying packed data.
    for (Operation *op : llvm::reverse(original))
      if (converted.contains(op) && isOpTriviallyDead(op))
        op->erase();
    for (Operation *op : llvm::reverse(created))
      if (isOpTriviallyDead(op))
        op->erase();
  }

private:
  template <typename Op, typename... Args> Op create(Args &&...args) {
    Op op = builder.create<Op>(std::forward<Args>(args)...);
    created.push_back(op.getOperation());
    return op;
  }

  Value integerConstant(Location loc, RankedTensorType type, int64_t value) {
    return create<arith::ConstantOp>(
        loc, type,
        DenseElementsAttr::get(
            type, builder.getIntegerAttr(type.getElementType(), value)));
  }

  Halves get(Value value) {
    auto found = halves.find(value);
    if (found != halves.end())
      return found->second;
    auto type = integerTensor(value, 64);
    assert(type && "only static unencoded i64 tensors may be split");
    auto narrowType = type.clone(builder.getI32Type());
    OpBuilder::InsertionGuard guard(builder);
    if (Operation *producer = value.getDefiningOp())
      builder.setInsertionPointAfter(producer);
    else
      builder.setInsertionPointToStart(value.getParentBlock());
    Location loc = value.getLoc();
    Value low = create<arith::TruncIOp>(loc, narrowType, value);
    Value high;
    if (isBooleanMask(value)) {
      high = integerConstant(loc, narrowType, 0);
    } else {
      Value shift = integerConstant(loc, type, 32);
      Value upper = create<arith::ShRUIOp>(loc, value, shift);
      high = create<arith::TruncIOp>(loc, narrowType, upper);
    }
    Halves pair{low, high};
    halves.try_emplace(value, pair);
    return pair;
  }

  Value pack(Value original, Halves pair) {
    auto type = cast<RankedTensorType>(original.getType());
    Location loc = original.getLoc();
    Value low = create<arith::ExtUIOp>(loc, type, pair.low);
    Value high = create<arith::ExtUIOp>(loc, type, pair.high);
    Value shift = integerConstant(loc, type, 32);
    Value upper = create<arith::ShLIOp>(loc, high, shift);
    return create<arith::OrIOp>(loc, low, upper);
  }

  Value cloneShape(Operation *op, Value source) {
    Operation *copy = builder.clone(*op);
    created.push_back(copy);
    copy->setOperand(0, source);
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    copy->getResult(0).setType(type.clone(builder.getI32Type()));
    return copy->getResult(0);
  }

  Value reduceHalf(ReduceOp original, Value input) {
    Location loc = original.getLoc();
    auto reduce = create<ReduceOp>(loc, ValueRange{input}, original.getAxis());
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *body = builder.createBlock(&reduce.getCombineOp());
      body->addArguments({builder.getI32Type(), builder.getI32Type()},
                         {loc, loc});
      Value sum = create<arith::AddIOp>(loc, body->getArgument(0),
                                        body->getArgument(1));
      create<ReduceReturnOp>(loc, ValueRange{sum});
    }
    return reduce->getResult(0);
  }

  // Preserve the initial key packing as one boundary rather than splitting its
  // two unfinished inputs independently. Its shift and tie arithmetic remain
  // at their original width and are evaluated exactly once.
  bool isPackingOr(arith::OrIOp op) {
    for (Value operand : op->getOperands())
      if (auto shift = operand.getDefiningOp<arith::ShLIOp>())
        if (constant(shift.getRhs(), 32))
          return true;
    return false;
  }

  // Return only edges carrying packed data. A shared mask, constant, block
  // argument, or unsupported producer is a boundary, not a bridge connecting
  // otherwise unrelated computations in the same function.
  SmallVector<Value, 2> dataOperands(Operation *op) {
    if (op->getNumResults() != 1 || !integerTensor(op->getResult(0), 64) ||
        isBooleanMask(op->getResult(0)))
      return {};
    if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      if (auto bitOr = dyn_cast<arith::OrIOp>(op))
        if (isPackingOr(bitOr))
          return {};
      return {op->getOperand(0), op->getOperand(1)};
    }
    if (auto select = dyn_cast<arith::SelectOp>(op))
      return {select.getTrueValue(), select.getFalseValue()};
    if (auto mul = dyn_cast<arith::MulIOp>(op)) {
      Value mask = booleanMultiplier(mul);
      if (mask)
        return {mask == mul.getRhs() ? mul.getLhs() : mul.getRhs()};
      return {};
    }
    if (auto reduce = dyn_cast<ReduceOp>(op))
      return isOneHotSum(reduce) ? SmallVector<Value, 2>{reduce->getOperand(0)}
                                 : SmallVector<Value, 2>{};
    if (isa<ExpandDimsOp, BroadcastOp, ReshapeOp, TransOp>(op)) {
      if (!integerTensor(op->getOperand(0), 64))
        return {};
      if (auto reshape = dyn_cast<ReshapeOp>(op))
        if (reshape.getAllowReorder())
          return {};
      return {op->getOperand(0)};
    }
    if (auto shift = dyn_cast<arith::ShRUIOp>(op))
      if (constant(shift.getRhs(), 32))
        return {shift.getLhs()};
    return {};
  }

  void selectConnectedGraph(ArrayRef<Operation *> original) {
    SmallVector<Operation *> worklist;
    auto enqueue = [&](Operation *op) {
      if (op && !dataOperands(op).empty() && selected.insert(op).second)
        worklist.push_back(op);
    };
    for (Operation *op : original)
      if (auto reduce = dyn_cast<ReduceOp>(op))
        if (isOneHotSum(reduce))
          enqueue(op);
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      for (Value input : dataOperands(op))
        enqueue(input.getDefiningOp());
      Value result = op->getResult(0);
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        // Merely being a mask or condition user does not connect an operation.
        if (llvm::is_contained(dataOperands(user), result))
          enqueue(user);
      }
    }
  }

  bool isSelectedValue(Value value) const {
    Operation *producer = value.getDefiningOp();
    return producer && selected.contains(producer);
  }

  void visit(Operation *op) {
    if (!op->getBlock())
      return;
    builder.setInsertionPoint(op);
    Location loc = op->getLoc();
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      if (cmp.getPredicate() != arith::CmpIPredicate::sgt ||
          !integerTensor(cmp.getLhs(), 64) ||
          !integerTensor(cmp.getRhs(), 64) ||
          !(isSelectedValue(cmp.getLhs()) || isSelectedValue(cmp.getRhs())))
        return;
      Halves lhs = get(cmp.getLhs()), rhs = get(cmp.getRhs());
      Value highGreater = create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                lhs.high, rhs.high);
      Value highEqual = create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              lhs.high, rhs.high);
      Value lowGreater = create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                               lhs.low, rhs.low);
      Value tied = create<arith::AndIOp>(loc, highEqual, lowGreater);
      Value result = create<arith::OrIOp>(loc, highGreater, tied);
      cmp.getResult().replaceAllUsesWith(result);
      converted.insert(op);
      return;
    }
    if (auto trunc = dyn_cast<arith::TruncIOp>(op)) {
      auto sourceType = integerTensor(trunc.getIn(), 64);
      if (!sourceType ||
          trunc.getType() != sourceType.clone(builder.getI32Type()) ||
          trunc.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
          !isSelectedValue(trunc.getIn()))
        return;
      trunc.getResult().replaceAllUsesWith(get(trunc.getIn()).low);
      converted.insert(op);
      return;
    }
    if (!selected.contains(op) || op->getNumResults() != 1 ||
        !integerTensor(op->getResult(0), 64) ||
        halves.contains(op->getResult(0)))
      return;
    Value result = op->getResult(0);
    auto narrowType = integerTensor(result, 64).clone(builder.getI32Type());
    std::optional<Halves> pair;
    if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      if (auto bitOr = dyn_cast<arith::OrIOp>(op))
        if (isPackingOr(bitOr))
          return;
      Halves lhs = get(op->getOperand(0)), rhs = get(op->getOperand(1));
      auto bitwise = [&](Value a, Value b) -> Value {
        if (isa<arith::AndIOp>(op))
          return create<arith::AndIOp>(loc, a, b);
        if (isa<arith::OrIOp>(op))
          return create<arith::OrIOp>(loc, a, b);
        return create<arith::XOrIOp>(loc, a, b);
      };
      pair = Halves{bitwise(lhs.low, rhs.low), bitwise(lhs.high, rhs.high)};
    } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
      Halves yes = get(select.getTrueValue()), no = get(select.getFalseValue());
      pair = Halves{
          create<arith::SelectOp>(loc, select.getCondition(), yes.low, no.low),
          create<arith::SelectOp>(loc, select.getCondition(), yes.high,
                                  no.high)};
    } else if (auto mul = dyn_cast<arith::MulIOp>(op)) {
      Value mask = booleanMultiplier(mul);
      if (!mask)
        return;
      Halves data = get(mask == mul.getRhs() ? mul.getLhs() : mul.getRhs());
      Value multiplier = get(mask).low;
      pair = Halves{create<arith::MulIOp>(loc, data.low, multiplier),
                    create<arith::MulIOp>(loc, data.high, multiplier)};
    } else if (auto reduce = dyn_cast<ReduceOp>(op)) {
      if (!isOneHotSum(reduce))
        return;
      Halves input = get(reduce->getOperand(0));
      // Exactly one active lane means neither low nor high sums can carry.
      pair =
          Halves{reduceHalf(reduce, input.low), reduceHalf(reduce, input.high)};
    } else if (isa<ExpandDimsOp, BroadcastOp, ReshapeOp, TransOp>(op)) {
      if (!integerTensor(op->getOperand(0), 64))
        return;
      if (auto reshape = dyn_cast<ReshapeOp>(op))
        if (reshape.getAllowReorder())
          return;
      Halves input = get(op->getOperand(0));
      pair = Halves{cloneShape(op, input.low), cloneShape(op, input.high)};
    } else if (auto shift = dyn_cast<arith::ShRUIOp>(op)) {
      if (!constant(shift.getRhs(), 32))
        return;
      pair =
          Halves{get(shift.getLhs()).high, integerConstant(loc, narrowType, 0)};
    }
    if (pair) {
      halves.try_emplace(result, *pair);
      converted.insert(op);
    }
  }

  OpBuilder builder;
  llvm::DenseMap<Value, Halves> halves;
  llvm::SmallPtrSet<Operation *, 32> selected;
  llvm::SmallPtrSet<Operation *, 32> converted;
  SmallVector<Operation *> created;
};

inline void run(ModuleOp module) {
  // Each function is limited further to supported dataflow components seeded
  // by a proven one-hot sum. Function arguments and loop state stay as borders.
  module.walk([&](triton::FuncOp function) {
    Splitter(module.getContext()).run(function.getOperation());
  });
}

} // namespace mlir::triton::split_packed_tensor

#endif // TRITON_ASCEND_GRAPH_SPLIT_PACKED_TENSOR_H
