/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "TritonToGraph/RowLifting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
namespace cfg {

bool dependsOnValue(Value v, Value pid) {
  if (v == pid)
    return true;

  Operation *op = v.getDefiningOp();
  return op && llvm::any_of(op->getOperands(),
                            [&](Value a) { return dependsOnValue(a, pid); });
}

static bool hasPidDependency(Value value, Value pid,
                             llvm::DenseSet<Value> &visited) {
  if (value == pid)
    return true;
  if (!visited.insert(value).second)
    return false;

  Operation *op = value.getDefiningOp();
  if (!op)
    return false;

  for (Value operand : op->getOperands())
    if (hasPidDependency(operand, pid, visited))
      return true;

  bool nested = false;
  op->walk([&](Operation *inner) {
    if (nested)
      return;
    for (Value operand : inner->getOperands())
      if (hasPidDependency(operand, pid, visited)) {
        nested = true;
        return;
      }
  });
  return nested;
}

static bool dependsOnValueThroughRegions(Value root, Value target,
                                         llvm::DenseSet<Value> &visited) {
  if (root == target)
    return true;
  if (!visited.insert(root).second)
    return false;

  Operation *op = root.getDefiningOp();
  if (!op)
    return false;

  bool found = false;
  op->walk([&](Operation *nested) {
    if (found)
      return;
    for (Value operand : nested->getOperands())
      if (dependsOnValueThroughRegions(operand, target, visited)) {
        found = true;
        return;
      }
  });
  return found;
}

static bool dependsOnValueThroughRegions(Value root, Value target) {
  llvm::DenseSet<Value> visited;
  return dependsOnValueThroughRegions(root, target, visited);
}

bool isPidInvariant(Value value, Value pid) {
  llvm::DenseSet<Value> visited;
  return !hasPidDependency(value, pid, visited);
}

std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value)))
    return value.getSExtValue();

  return std::nullopt;
}

std::optional<IntegerBounds> inferMonotonicBounds(Value value, Value pid,
                                                  int64_t grid) {
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (!integer || integer.getWidth() > 64)
    return std::nullopt;
  unsigned width = integer.getWidth();
  IntegerBounds domain{APInt::getSignedMinValue(width).getSExtValue(),
                       APInt::getSignedMaxValue(width).getSExtValue()};
  auto finish = [&](__int128 lo, __int128 hi) -> IntegerBounds {
    return {static_cast<int64_t>(std::clamp<__int128>(lo, domain.lo, domain.hi)),
            static_cast<int64_t>(std::clamp<__int128>(hi, domain.lo, domain.hi))};
  };
  if (value == pid)
    return finish(0, (grid > 0 ? grid : INT32_MAX) - 1);
  if (auto constant = getConstantIntValue(value))
    return IntegerBounds{*constant, *constant};
  if (isPidInvariant(value, pid))
    return domain;
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return inferMonotonicBounds(ext.getIn(), pid, grid);

  Operation *op = value.getDefiningOp();
  if (!op || op->getNumOperands() != 2)
    return std::nullopt;
  auto lhs = inferMonotonicBounds(op->getOperand(0), pid, grid);
  auto rhs = inferMonotonicBounds(op->getOperand(1), pid, grid);
  if (!lhs || !rhs)
    return std::nullopt;
  if (isa<arith::AddIOp>(op))
    return finish(static_cast<__int128>(lhs->lo) + rhs->lo,
                  static_cast<__int128>(lhs->hi) + rhs->hi);
  if (isa<arith::SubIOp>(op) && isPidInvariant(op->getOperand(1), pid))
    return finish(static_cast<__int128>(lhs->lo) - rhs->hi,
                  static_cast<__int128>(lhs->hi) - rhs->lo);
  if (isa<arith::MulIOp>(op)) {
    if (rhs->lo == rhs->hi && rhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->hi) * rhs->lo);
    if (lhs->lo == lhs->hi && lhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->lo) * rhs->hi);
    if (lhs->lo >= 0 && rhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->hi) * rhs->hi);
  }
  if (isa<arith::DivSIOp>(op) && rhs->lo == rhs->hi && rhs->lo > 0)
    return finish(lhs->lo / rhs->lo, lhs->hi / rhs->lo);
  return std::nullopt;
}

bool canUseLastRowLoopBound(scf::ForOp loop, Value pid, int64_t grid) {
  Value bound = loop.getUpperBound();
  auto unsafeArithmetic = [](Operation *op) {
    StringRef ns = op->getName().getDialectNamespace();
    return (ns == "arith" || ns == "math") && !isSpeculatable(op);
  };
  if (grid <= 0) {
    auto function = loop->getParentOfType<triton::FuncOp>();
    if (!function)
      return false;
    auto result = function.walk([&](Operation *op) {
      bool unsafe = unsafeArithmetic(op) &&
                    llvm::any_of(op->getOperands(), [&](Value operand) {
                      return !isPidInvariant(operand, pid);
                    });
      return unsafe ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return false;
  }
  if (!isPidInvariant(bound, pid)) {
    auto result = loop.getBody()->walk([&](Operation *op) {
      return unsafeArithmetic(op) ? WalkResult::interrupt()
                                  : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return false;
  }
  return inferMonotonicBounds(bound, pid, grid).has_value();
}

static bool isScalarBoundSlice(Operation *op) {
  StringRef ns = op->getName().getDialectNamespace();
  if (ns == "arith")
    return true;
  if (!isa<scf::IfOp>(op))
    return false;

  bool pure = true;
  op->walk([&](Operation *nested) {
    if (nested == op || !pure)
      return;
    if (isa<scf::YieldOp>(nested))
      return;
    StringRef inner = nested->getName().getDialectNamespace();
    if (inner != "arith" && inner != "math")
      pure = false;
  });
  return pure;
}

static bool isTensorPointer(Type type) {
  auto pointer = dyn_cast<triton::PointerType>(type);
  return pointer && isa<RankedTensorType>(pointer.getPointeeType());
}

int64_t selectRowsPerProgram(int64_t grid, int64_t maxDotRows,
                             int64_t maxTensorBytes, int64_t peakLiveBytes,
                             const RowCountPolicy &policy) {
  const int64_t budget = policy.maxLiftedValueBytes;

  std::optional<int64_t> limit;
  auto tighten = [&](int64_t bound) {
    limit = limit ? std::min(*limit, bound) : bound;
  };

  if (policy.maxL0ABytes > 0 && policy.l0ABudgetBytes > 0)
    tighten(static_cast<int64_t>(policy.l0ABudgetBytes /
                                 static_cast<uint64_t>(policy.maxL0ABytes)));
  if (policy.maxL0CBytes > 0 && policy.l0CBudgetBytes > 0)
    tighten(static_cast<int64_t>(policy.l0CBudgetBytes /
                                 static_cast<uint64_t>(policy.maxL0CBytes)));
  if (maxTensorBytes > 0)
    tighten(budget / maxTensorBytes);
  if (policy.ubBudgetBytes > 0 && peakLiveBytes > 0)
    tighten(static_cast<int64_t>(
        policy.ubBudgetBytes / static_cast<uint64_t>(peakLiveBytes)));
  if (policy.targetDotRows > 0 && maxDotRows > 0)
    tighten(policy.targetDotRows / maxDotRows);
  if (grid > 0 && policy.coreNum > 0)
    tighten(grid / static_cast<int64_t>(policy.coreNum));

  if (!limit)
    return 1;

  int64_t factor = 1;
  while (factor <= *limit / 2)
    factor *= 2;

  while (factor >= 2 && grid && grid % factor != 0)
    factor /= 2;

  return factor >= 2 ? factor : 1;
}

int64_t getElementStorageBytes(Type element) {
  if (auto integer = dyn_cast<IntegerType>(element))
    return (integer.getWidth() + 7) / 8;
  if (auto floatType = dyn_cast<FloatType>(element))
    return (floatType.getWidth() + 7) / 8;
  if (isa<triton::PointerType>(element))
    return 8;
  return 0;
}

static int64_t getTensorBytes(RankedTensorType type) {
  int64_t bytes = getElementStorageBytes(type.getElementType());
  if (bytes == 0)
    return 0;
  for (int64_t extent : type.getShape())
    bytes *= extent;
  return bytes;
}

static Value getRuntimeGridExtent(Block *body) {
  return body->getArgument(body->getNumArguments() - 2);
}

RowLiftBuilder::RowLiftBuilder(IRRewriter &rewriter, RowLiftCandidate candidate)
    : rewriter(rewriter), candidate(candidate) {}

RankedTensorType RowLiftBuilder::getLiftedType(Type old) {
  SmallVector<int64_t> shape{candidate.rowsPerProgram};
  Type elem = old;
  if (auto tensor = dyn_cast<RankedTensorType>(old)) {
    shape.append(tensor.getShape().begin(), tensor.getShape().end());
    elem = tensor.getElementType();
  }
  return RankedTensorType::get(shape, elem);
}

Value RowLiftBuilder::getMappedValue(Value old) {
  return valueMapping.lookup(old);
}

bool RowLiftBuilder::hasRowDimension(Value old) {
  if (!valueMapping.contains(old))
    return false;
  return valueMapping.lookup(old).getType() != old.getType();
}

Value RowLiftBuilder::ensureRowDimension(Value old) {
  Value value = valueMapping.contains(old) ? valueMapping.lookup(old) : Value();
  if (isTensorPointer(old.getType()))
    return value;
  if (hasRowDimension(old))
    return value;

  if (!isa<RankedTensorType>(old.getType()))
    return rewriter.create<triton::SplatOp>(
        old.getLoc(), getLiftedType(old.getType()), value);

  Value expanded =
      rewriter.create<triton::ExpandDimsOp>(old.getLoc(), value, 0);
  return rewriter.create<triton::BroadcastOp>(
      old.getLoc(), getLiftedType(old.getType()), expanded);
}

Value RowLiftBuilder::reshape(Value v, ArrayRef<int64_t> shape) {
  auto t = cast<RankedTensorType>(v.getType());
  return rewriter.create<triton::ReshapeOp>(
      v.getLoc(), RankedTensorType::get(shape, t.getElementType()), v, false);
}

Value RowLiftBuilder::flattenDotRows(Value v) {
  auto t = cast<RankedTensorType>(v.getType());
  return reshape(v, {t.getShape()[0] * t.getShape()[1], t.getShape()[2]});
}

Value RowLiftBuilder::broadcastRowValue(Value v, RankedTensorType target) {
  while (cast<RankedTensorType>(v.getType()).getRank() < target.getRank())
    v = rewriter.create<triton::ExpandDimsOp>(
        v.getLoc(), v, cast<RankedTensorType>(v.getType()).getRank());
  return rewriter.create<triton::BroadcastOp>(v.getLoc(), target, v);
}

Value RowLiftBuilder::combineMemoryMask(Value pointer, Value originalMask,
                                        Value extraMask, bool &ok) {
  Value mask = mergeMasks(
      pointer, Value(),
      {extraMask, candidate.runtimeGrid ? validRows : Value()}, ok);
  if (!ok)
    return Value();
  return mergeMasks(
      pointer, mask, {originalMask ? ensureRowDimension(originalMask) : Value()},
      ok);
}

Value RowLiftBuilder::mergeMasks(Value pointer, Value base, ArrayRef<Value> parts,
                                 bool &ok) {
  ok = true;
  auto pointerType = cast<RankedTensorType>(pointer.getType());
  auto maskType =
      RankedTensorType::get(pointerType.getShape(), rewriter.getI1Type());
  Value mask = base;
  auto merge = [&](Value part) {
    if (!part || !ok)
      return;
    if (part.getType() != maskType) {
      auto partType = dyn_cast<RankedTensorType>(part.getType());
      if (!partType || partType.getElementType() != rewriter.getI1Type() ||
          partType.getRank() >= maskType.getRank()) {
        ok = false;
        return;
      }
      part = broadcastRowValue(part, maskType);
    }
    mask = mask
               ? Value(rewriter.create<arith::AndIOp>(part.getLoc(), mask, part))
               : part;
  };
  for (Value part : parts)
    merge(part);
  return mask;
}

static Value buildMaskPadding(IRRewriter &rewriter, Location loc, Type type,
                              std::optional<triton::PaddingOption> padding) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  Type element = tensor ? tensor.getElementType() : type;
  if (padding != triton::PaddingOption::PAD_NAN)
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(type));
  auto floatType = dyn_cast<FloatType>(element);
  if (!floatType)
    return Value();
  Value nan = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(
               floatType, APFloat::getNaN(floatType.getFloatSemantics())));
  return tensor ? Value(rewriter.create<triton::SplatOp>(loc, type, nan)) : nan;
}

bool RowLiftBuilder::rewriteLoop(scf::ForOp old) {
  rowActiveMask = Value();

  SmallVector<Value> loopInit;
  for (Value arg : old.getInitArgs())
    loopInit.push_back(ensureRowDimension(arg));

  if (!lastRowMapping.contains(old.getUpperBound())) {
    LLVM_DEBUG(llvm::dbgs() << "row-lift: last-row bound missing\n");
    return false;
  }
  auto loop = rewriter.create<scf::ForOp>(
      old.getLoc(), getMappedValue(old.getLowerBound()),
      lastRowMapping.lookup(old.getUpperBound()), getMappedValue(old.getStep()),
      loopInit);
  loop->setAttrs(old->getAttrs());

  valueMapping.map(old.getInductionVar(), loop.getInductionVar());
  for (auto [index, arg] : llvm::enumerate(old.getRegionIterArgs()))
    valueMapping.map(arg, loop.getRegionIterArgs()[index]);

  rewriter.setInsertionPointToStart(loop.getBody());
  for (Operation &op : old.getBody()->without_terminator())
    if (!rewrite(&op))
      return false;

  auto yield = cast<scf::YieldOp>(old.getBody()->getTerminator());
  if (!rowActiveMask)
    rowActiveMask = createLoopActiveMask(old, loop);
  Value active = rowActiveMask;

  auto keep = [&](Value incoming, Value previous, RankedTensorType type) {
    Value mask = broadcastRowValue(
        active, RankedTensorType::get(type.getShape(), rewriter.getI1Type()));
    return Value(rewriter.create<arith::SelectOp>(old.getLoc(), mask, incoming,
                                                  previous));
  };

  SmallVector<Value> output;
  for (auto [index, value] : llvm::enumerate(yield.getOperands())) {
    auto previous =
        cast<RankedTensorType>(loop.getRegionIterArgs()[index].getType());
    output.push_back(keep(ensureRowDimension(value),
                          loop.getRegionIterArgs()[index], previous));
  }

  if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back()))
    rewriter.eraseOp(loop.getBody()->getTerminator());
  rewriter.setInsertionPointToEnd(loop.getBody());
  rewriter.create<scf::YieldOp>(old.getLoc(), output);

  for (auto [index, result] : llvm::enumerate(old.getResults()))
    valueMapping.map(result, loop.getResults()[index]);

  rewriter.setInsertionPointAfter(loop);
  return true;
}

Value RowLiftBuilder::createLoopActiveMask(scf::ForOp old, scf::ForOp loop) {
  Value ivSplat = rewriter.create<triton::SplatOp>(
      old.getLoc(), getLiftedType(old.getInductionVar().getType()),
      loop.getInductionVar());
  return rewriter.create<arith::CmpIOp>(
      old.getLoc(), arith::CmpIPredicate::slt, ivSplat,
      ensureRowDimension(old.getUpperBound()));
}

bool RowLiftBuilder::run(triton::FuncOp fn) {
  Block *old = &fn.getBody().front();
  auto loops = old->getOps<scf::ForOp>();
  if (loops.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "row-lift: no loop to lift\n");
    return false;
  }
  scf::ForOp loop = *loops.begin();
  Value upperBound = loop.getUpperBound();
  auto *fresh = new Block();
  fn.getBody().push_back(fresh);

  for (BlockArgument arg : old->getArguments()) {
    Value newArg = fresh->addArgument(arg.getType(), arg.getLoc());
    valueMapping.map(arg, newArg);
    lastRowMapping.map(arg, newArg);
  }

  rewriter.setInsertionPointToStart(fresh);
  for (Operation &op : old->without_terminator()) {
    if (&op == candidate.anchor) {
      auto pid = cast<triton::GetProgramIdOp>(rewriter.clone(op));
      Value base = rewriter.create<arith::MulIOp>(
          op.getLoc(), pid.getResult(),
          rewriter.create<arith::ConstantIntOp>(
              op.getLoc(), candidate.rowsPerProgram, 32));
      Value last = rewriter.create<arith::AddIOp>(
          op.getLoc(), base,
          rewriter.create<arith::ConstantIntOp>(
              op.getLoc(), candidate.rowsPerProgram - 1, 32));
      if (candidate.runtimeGrid) {
        Value extent = getRuntimeGridExtent(fresh);
        Value lastValid = rewriter.create<arith::SubIOp>(
            op.getLoc(), extent,
            rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 32));
        last = rewriter.create<arith::MinSIOp>(op.getLoc(), last, lastValid);
      }
      lastRowMapping.map(op.getResult(0), last);

      auto t = getLiftedType(op.getResult(0).getType());
      Value lanes = rewriter.create<triton::MakeRangeOp>(
          op.getLoc(), t, 0, candidate.rowsPerProgram);
      Value start = rewriter.create<triton::SplatOp>(op.getLoc(), t, base);
      valueMapping.map(
          op.getResult(0),
          rewriter.create<arith::AddIOp>(op.getLoc(), start, lanes)
              .getResult());
      if (candidate.runtimeGrid) {
        Value extent = getRuntimeGridExtent(fresh);
        Value rowsLeft = rewriter.create<arith::SubIOp>(op.getLoc(), extent,
                                                        base);
        Value limit = rewriter.create<triton::SplatOp>(op.getLoc(), t, rowsLeft);
        validRows = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::slt, lanes, limit);
      }
      continue;
    }

    if (op.getNumResults() == 1 &&
        !isa<RankedTensorType>(op.getResult(0).getType()) &&
        isScalarBoundSlice(&op) &&
        dependsOnValueThroughRegions(upperBound, op.getResult(0)))
      rewriter.clone(op, lastRowMapping);

    if (!rewrite(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "row-lift: rewrite failed at " << op << "\n");
      return false;
    }

    for (Value v : op.getResults()) {
      if (!valueMapping.contains(v))
        continue;
      Value mapped = valueMapping.lookup(v);
      if (!hasRowDimension(v) && !lastRowMapping.contains(v))
        lastRowMapping.map(v, mapped);
    }
  }

  rewriter.create<triton::ReturnOp>(fn.getLoc());
  old->dropAllDefinedValueUses();
  old->erase();
  return true;
}

bool RowLiftBuilder::rewrite(Operation *op) {
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return rewriteLoop(loop);

  bool dependent = llvm::any_of(op->getOperands(), [&](Value v) {
    return hasRowDimension(v);
  });
  if (!dependent)
    op->walk([&](Operation *nested) {
      for (Value v : nested->getOperands())
        if (!dependent && valueMapping.contains(v) && hasRowDimension(v))
          dependent = true;
    });
  if (!dependent) {
    rewriter.clone(*op, valueMapping);
    return true;
  }

  if (op->getNumRegions() != 0)
    return false;

  if (auto dot = dyn_cast<triton::DotOp>(op)) {
    if (!hasRowDimension(dot.getA()) || hasRowDimension(dot.getB()))
      return false;

    auto result = cast<RankedTensorType>(dot.getType());
    SmallVector<Value> operands{
        flattenDotRows(ensureRowDimension(dot.getA())),
        getMappedValue(dot.getB()),
        flattenDotRows(ensureRowDimension(dot.getC()))};
    Type flatType = RankedTensorType::get(
        {candidate.rowsPerProgram * result.getShape()[0],
         result.getShape()[1]},
        result.getElementType());
    Operation *d =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                        TypeRange{flatType}, op->getAttrs());
    valueMapping.map(
        dot.getResult(),
        reshape(d->getResult(0), getLiftedType(result).getShape()));
    return true;
  }

  if (auto splat = dyn_cast<triton::SplatOp>(op)) {
    valueMapping.map(splat.getResult(),
                     broadcastRowValue(ensureRowDimension(splat.getSrc()),
                                       getLiftedType(splat.getType())));
    return true;
  }

  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
    valueMapping.map(
        expand.getResult(),
        rewriter
            .create<triton::ExpandDimsOp>(op->getLoc(),
                                          ensureRowDimension(expand.getSrc()),
                                          expand.getAxis() + 1)
            .getResult());
    return true;
  }

  if (auto view = dyn_cast<triton::ReshapeOp>(op)) {
    valueMapping.map(view.getResult(),
                     reshape(ensureRowDimension(view.getSrc()),
                             getLiftedType(view.getType()).getShape()));
    return true;
  }

  if (auto trans = dyn_cast<triton::TransOp>(op)) {
    SmallVector<int32_t> order{0};
    for (int32_t axis : trans.getOrder())
      order.push_back(axis + 1);
    valueMapping.map(
        trans.getResult(),
        rewriter
            .create<triton::TransOp>(
                op->getLoc(), ensureRowDimension(trans.getSrc()), order)
            .getResult());
    return true;
  }

  if (candidate.runtimeGrid) {
    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      Value pointer = ensureRowDimension(load.getPtr());
      if (!isa<RankedTensorType>(pointer.getType()))
        return false;
      bool ok = true;
      Value mask = combineMemoryMask(pointer, load.getMask(), Value(), ok);
      if (!ok)
        return false;
      auto type = getLiftedType(load.getType());
      Value other = load.getOther()
                        ? ensureRowDimension(load.getOther())
                        : buildMaskPadding(rewriter, op->getLoc(), type,
                                           load.getPadding());
      if (!other)
        return false;
      auto replacement = rewriter.create<triton::LoadOp>(
          op->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      valueMapping.map(load.getResult(), replacement.getResult());
      return true;
    }
  }

  if (auto store = dyn_cast<triton::StoreOp>(op)) {
    if (candidate.runtimeGrid) {
      Value pointer = ensureRowDimension(store.getPtr());
      if (!isa<RankedTensorType>(pointer.getType()))
        return false;
      bool ok = true;
      Value mask = combineMemoryMask(pointer, store.getMask(), Value(), ok);
      if (!ok)
        return false;
      rewriter.create<triton::StoreOp>(
          op->getLoc(), pointer, ensureRowDimension(store.getValue()), mask,
          store.getBoundaryCheck(), store.getCache(), store.getEvict());
      return true;
    }
    if (op->getParentOfType<scf::ForOp>() && !store.getMask())
      return false;
  }

  SmallVector<Value> operands;
  for (Value v : op->getOperands())
    operands.push_back(ensureRowDimension(v));

  SmallVector<Type> types;
  for (Type t : op->getResultTypes())
    types.push_back(getLiftedType(t));

  Operation *cloned =
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                      types, op->getAttrs());
  valueMapping.map(op->getResults(), cloned->getResults());
  return true;
}

}
}
}