/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the MIT license.
 */

#include "StoreOwnershipAnalysis.h"
#include "ascend/include/TritonToStructured/PtrAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using TritonToStructured::PtrState;

namespace {

// Recognize a dense [batch, time, head, column] layout with square diagonal
// tiles. Other layouts, indirect indices and loop-carried pointers keep locks.
class DiagonalTile {
public:
  DiagonalTile(StoreOp store, PatternRewriter &rewriter)
      : store(store), rewriter(rewriter), context(store.getContext()) {}
  LogicalResult analyze();
  Value guard();

private:
  AffineExpr expression(Value value);
  FailureOr<PtrState> pointer(Value value);
  AffineExpr base(int64_t width);
  bool fits(const PtrState &state, bool output);
  StoreOp store;
  PatternRewriter &rewriter;
  MLIRContext *context;
  TritonToStructured::PtrAnalysis pointers;
  DenseMap<Value, AffineExpr> expressions;
  std::array<std::optional<Value>, 3> offsets;
  Value length, output;
  SmallVector<Value> inputs;
  int64_t heads = 1, chunk = 0, tile = 0, stride = 0, width = 0;
};

Value stripCast(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<arith::ExtSIOp, arith::IndexCastOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

AffineExpr DiagonalTile::expression(Value value) {
  value = stripCast(value);
  if (auto it = expressions.find(value); it != expressions.end())
    return it->second;
  if (auto integer = dyn_cast<IntegerType>(value.getType());
      integer && integer.getWidth() != 32 && integer.getWidth() != 64)
    return {};
  AffineExpr result;
  Value index = value, offset;
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    index = stripCast(add.getLhs());
    offset = stripCast(add.getRhs());
    if (!index.getDefiningOp<GetProgramIdOp>())
      std::swap(index, offset);
  }
  if (auto pid = index.getDefiningOp<GetProgramIdOp>()) {
    if (offset) {
      auto arg = dyn_cast<BlockArgument>(offset);
      if (!arg || !isa<FuncOp>(arg.getOwner()->getParentOp()))
        return {};
    }
    unsigned axis = pid.getAxisAsInt();
    if (offsets[axis] && *offsets[axis] != offset)
      return {};
    offsets[axis] = offset;
    result = getAffineDimExpr(axis, context);
  } else if (auto constant = getConstantIntValue(value)) {
    if (*constant < 0)
      return {};
    result = getAffineConstantExpr(*constant, context);
  } else if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (!isa<FuncOp>(arg.getOwner()->getParentOp()) ||
        !isa<IntegerType>(arg.getType()) || (length && length != arg))
      return {};
    length = arg;
    result = getAffineSymbolExpr(0, context);
  } else if (Operation *op = value.getDefiningOp();
             op &&
             isa<arith::AddIOp, arith::MulIOp, arith::DivSIOp, arith::RemSIOp>(
                 op)) {
    AffineExpr lhs = expression(op->getOperand(0));
    AffineExpr rhs = expression(op->getOperand(1));
    if (!lhs || !rhs)
      return {};
    if (isa<arith::AddIOp>(op))
      result = lhs + rhs;
    else if (isa<arith::MulIOp>(op)) {
      if (!rhs.isSymbolicOrConstant())
        std::swap(lhs, rhs);
      if (!rhs.isSymbolicOrConstant() || rhs == 0)
        return {};
      result = lhs * rhs;
    } else {
      auto divisor = dyn_cast<AffineConstantExpr>(rhs);
      if (lhs != getAffineDimExpr(2, context) || !divisor ||
          divisor.getValue() <= 0 ||
          ((chunk || heads != 1) && heads != divisor.getValue()))
        return {};
      heads = divisor.getValue();
      result = isa<arith::DivSIOp>(op) ? lhs.floorDiv(heads) : lhs % heads;
    }
  }
  if (result)
    expressions[value] = result;
  return result;
}

FailureOr<PtrState> DiagonalTile::pointer(Value value) {
  // PtrAnalysis also supports wrapping tensor arithmetic. This proof needs
  // nonnegative add/multiply and zero-based ranges to bound intermediates.
  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.omitUsesFromAbove = false;
  options.inclusive = true;
  if (failed(getBackwardSlice(value, &slice, options)))
    return failure();
  // Bound static coefficients before PtrAnalysis folds them. Replacing each
  // variable/range by one is conservative for nonnegative add/multiply; it
  // prevents wrapped constants from masquerading as small tile strides.
  DenseMap<Value, int64_t> weights;
  for (Operation *op : slice) {
    // Analysis-created offsets must dominate the target store.
    if (op->getBlock() == store->getBlock() && !op->isBeforeInBlock(store))
      return failure();
    if (op->getNumResults() != 1)
      return failure();
    int64_t weight = 1;
    for (Value operand : op->getOperands()) {
      if (auto arg = dyn_cast<BlockArgument>(operand);
          arg && isa<FuncOp>(arg.getOwner()->getParentOp()))
        weights.try_emplace(operand, isa<PointerType>(arg.getType()) ? 0 : 1);
      if (!weights.contains(operand))
        return failure();
    }
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      APInt integer;
      if (auto scalar = dyn_cast<IntegerAttr>(c.getValue()))
        integer = scalar.getValue();
      else if (auto tensor = dyn_cast<DenseIntElementsAttr>(c.getValue());
               tensor && tensor.isSplat())
        integer = tensor.getSplatValue<APInt>();
      else
        return failure();
      if (integer.isNegative() || integer.getActiveBits() > 31)
        return failure();
      weight = integer.getZExtValue();
    } else if (isa<arith::AddIOp, arith::MulIOp, AddPtrOp>(op)) {
      int64_t lhs = weights[op->getOperand(0)];
      int64_t rhs = weights[op->getOperand(1)];
      weight = isa<arith::MulIOp>(op) ? lhs * rhs : lhs + rhs;
    } else if (isa<arith::ExtSIOp, arith::IndexCastOp, SplatOp, ExpandDimsOp,
                   BroadcastOp>(op)) {
      weight = weights[op->getOperand(0)];
    } else if (!isa<GetProgramIdOp, MakeRangeOp, arith::DivSIOp,
                    arith::RemSIOp>(op)) {
      return failure();
    }
    if (weight > INT32_MAX)
      return failure();
    weights[op->getResult(0)] = weight;
    if (auto range = dyn_cast<MakeRangeOp>(op); range && range.getStart() != 0)
      return failure();
    if (op->getNumResults() == 1)
      if (auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
        if (auto integer = dyn_cast<IntegerType>(type.getElementType());
            integer && integer.getWidth() != 32 && integer.getWidth() != 64)
          return failure();
        if (!isa<arith::ConstantOp, arith::AddIOp, arith::MulIOp,
                 arith::ExtSIOp, MakeRangeOp, SplatOp, ExpandDimsOp,
                 BroadcastOp, AddPtrOp>(op))
          return failure();
      }
  }
  PtrState state;
  if (failed(pointers.visitOperand(value, state, store.getLoc(), rewriter)) ||
      !isa_and_nonnull<BlockArgument>(state.source) ||
      !isa<FuncOp>(
          cast<BlockArgument>(state.source).getOwner()->getParentOp()) ||
      state.sizes.size() != state.stateInfo.size() || state.isBlockPtr())
    return failure();
  for (auto [axis, info] : llvm::enumerate(state.stateInfo)) {
    auto size = getConstantIntValue(state.sizes[axis]);
    auto shape = getConstantIntValue(info.shape);
    if (!size || !shape || *size != *shape || info.dimIndex != axis)
      return failure();
  }
  return state;
}

AffineExpr DiagonalTile::base(int64_t columns) {
  auto x = getAffineDimExpr(0, context), y = getAffineDimExpr(1, context);
  auto z = getAffineDimExpr(2, context), t = getAffineSymbolExpr(0, context);
  return (z.floorDiv(heads) * t * heads + z % heads) * columns +
         (x * chunk + y * tile) * (heads * columns);
}

bool DiagonalTile::fits(const PtrState &state, bool isOutput) {
  int64_t columns = isOutput ? chunk : width;
  auto actual =
      getValueOrCreateConstantIndexOp(rewriter, store.getLoc(), state.offset);
  AffineExpr expr = expression(actual);
  if (!expr)
    return false;
  auto expected = base(columns);
  if (isOutput)
    expected = expected + getAffineDimExpr(1, context) * tile;
  auto delta =
      dyn_cast<AffineConstantExpr>(simplifyAffineExpr(expr - expected, 3, 1));
  if (!delta || delta.getValue() < 0)
    return false;
  SmallVector<int64_t> sizes, strides;
  for (auto size : state.sizes) {
    auto c = getConstantIntValue(size);
    if (!c)
      return false;
    sizes.push_back(*c);
  }
  for (auto info : state.stateInfo) {
    auto c = getConstantIntValue(info.stride);
    if (!c)
      return false;
    strides.push_back(*c);
  }
  if (sizes.size() == 2)
    return sizes[0] == tile && sizes[1] == (isOutput ? tile : width) &&
           strides[0] == heads * columns && strides[1] == 1 && delta == 0;
  int64_t step = isOutput ? 1 : heads * width;
  return sizes.size() == 1 && sizes[0] == (isOutput ? tile : width) &&
         strides[0] == (isOutput ? stride : 1) &&
         delta.getValue() % step == 0 && delta.getValue() / step < tile;
}

LogicalResult DiagonalTile::analyze() {
  auto target = pointer(store.getPtr());
  if (failed(target) || target->sizes.size() != 2 ||
      !cast<PointerType>(target->source.getType()).getPointeeType().isF32())
    return failure();
  auto rows = getConstantIntValue(target->sizes[0]);
  auto cols = getConstantIntValue(target->sizes[1]);
  auto rowStride = getConstantIntValue(target->stateInfo[0].stride);
  if (!rows || !cols || !rowStride || *rows != *cols || *rows <= 0 ||
      *rows % 8 || *rowStride <= 0 || *rowStride > INT32_MAX)
    return failure();
  tile = *rows;
  stride = *rowStride;
  auto offset =
      getValueOrCreateConstantIndexOp(rewriter, store.getLoc(), target->offset);
  if (!expression(offset) || !length ||
      llvm::any_of(offsets, [](auto offset) { return !offset.has_value(); }) ||
      stride % heads)
    return failure();
  chunk = stride / heads;
  if (!chunk || chunk % tile)
    return failure();
  output = target->source;
  auto function = store->getParentOfType<FuncOp>();
  WalkResult result = function.walk([&](Operation *op) -> WalkResult {
    auto load = dyn_cast<LoadOp>(op);
    auto write = dyn_cast<StoreOp>(op);
    if (!load && !write)
      return isa<FuncOp, gpu::BarrierOp>(op) || isMemoryEffectFree(op) ||
                     op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    if (op->getBlock() != store->getBlock())
      return WalkResult::interrupt();
    auto state = pointer(load ? load.getPtr() : write.getPtr());
    if (failed(state) || (write && state->source != output))
      return WalkResult::interrupt();
    Type element = cast<PointerType>(state->source.getType()).getPointeeType();
    if (!element.isF16() && !element.isBF16() && !element.isF32())
      return WalkResult::interrupt();
    bool isOutput = state->source == output;
    if (!isOutput && !width && state->sizes.size() == 2) {
      auto size = getConstantIntValue(state->sizes[1]);
      if (size && *size > 0 && *size <= INT32_MAX)
        width = *size;
    }
    if ((!isOutput && !width) || !fits(*state, isOutput))
      return WalkResult::interrupt();
    if (!isOutput && !llvm::is_contained(inputs, state->source))
      inputs.push_back(state->source);
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

Value DiagonalTile::guard() {
  auto loc = store.getLoc();
  auto c = [&](int64_t x) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, x, 64);
  };
  auto add = [&](Value a, Value b) -> Value {
    return rewriter.createOrFold<arith::AddIOp>(loc, a, b);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return rewriter.createOrFold<arith::MulIOp>(loc, a, b);
  };
  auto cmp = [&](arith::CmpIPredicate p, Value a, Value b) -> Value {
    return rewriter.createOrFold<arith::CmpIOp>(loc, p, a, b);
  };
  auto extend = [&](Value v) -> Value {
    return v.getType().isInteger(64) ? v
                                     : rewriter.createOrFold<arith::ExtSIOp>(
                                           loc, rewriter.getI64Type(), v);
  };
  Value valid = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  auto require = [&](Value condition) {
    valid = rewriter.createOrFold<arith::AndIOp>(loc, valid, condition);
  };
  using P = arith::CmpIPredicate;
  SmallVector<Value> end;
  for (unsigned axis = 0; axis < 3; ++axis) {
    Value n = extend(rewriter.create<GetNumProgramsOp>(
        loc, static_cast<ProgramIDDim>(axis)));
    Value off = *offsets[axis] ? extend(*offsets[axis]) : c(0);
    require(cmp(P::sgt, n, c(0)));
    require(cmp(P::sge, off, c(0)));
    require(cmp(P::ule, off, c(INT32_MAX)));
    end.push_back(add(n, off));
    require(cmp(P::ule, end.back(), c(INT32_MAX)));
  }
  Value t = extend(length);
  require(cmp(P::sgt, t, c(0)));
  require(cmp(P::ule, t, c(INT32_MAX)));
  require(cmp(P::ule, mul(end[0], c(chunk)), t));
  require(cmp(P::ule, end[1], c(chunk / tile)));
  require(cmp(P::eq,
              rewriter.createOrFold<arith::RemUIOp>(loc, end[2], c(heads)),
              c(0)));
  // A 32-bit total extent bounds every nonnegative intermediate in the
  // matched address expressions, including original i32 tile arithmetic.
  Value divisor = rewriter.createOrFold<arith::MaxUIOp>(loc, t, c(1));
  Value limit = rewriter.createOrFold<arith::DivUIOp>(
      loc, c(INT32_MAX / std::max(chunk, width)), divisor);
  require(cmp(P::ule, end[2], limit));
  Value elements = mul(end[2], t);
  auto range = [&](Value pointer, int64_t columns) {
    Value start =
        rewriter.create<PtrToIntOp>(loc, rewriter.getI64Type(), pointer);
    unsigned bytes = cast<PointerType>(pointer.getType())
                         .getPointeeType()
                         .getIntOrFloatBitWidth() /
                     8;
    Value stop = add(start, mul(elements, c(columns * bytes)));
    require(cmp(P::uge, stop, start));
    return std::make_pair(start, stop);
  };
  auto dst = range(output, chunk);
  require(cmp(P::eq,
              rewriter.createOrFold<arith::AndIOp>(loc, dst.first, c(31)),
              c(0)));
  for (Value input : inputs) {
    auto src = range(input, width);
    require(rewriter.createOrFold<arith::OrIOp>(
        loc, cmp(P::ule, dst.second, src.first),
        cmp(P::ule, src.second, dst.first)));
  }
  return valid;
}
} // namespace

FailureOr<Value>
mlir::triton::buildStoreOwnershipGuard(StoreOp store,
                                       PatternRewriter &rewriter) {
  DiagonalTile tile(store, rewriter);
  if (failed(tile.analyze()))
    return failure();
  return tile.guard();
}
