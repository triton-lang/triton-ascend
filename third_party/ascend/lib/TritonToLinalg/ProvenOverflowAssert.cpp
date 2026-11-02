// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT

#include "ascend/include/TritonToLinalg/ProvenOverflowAssert.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <iterator>
#include <optional>

namespace mlir::triton {
namespace {

// Bounds cover every tensor element, in the signed interpretation of its actual
// bit width. In particular, i1 true is the bit pattern 1 (signed value -1).
struct Interval {
  llvm::APInt min;
  llvm::APInt max;

  bool isSingleton() const { return min == max; }
  bool isTrue() const {
    return min.getBitWidth() == 1 && isSingleton() && min.isOne();
  }
};

using Range = std::optional<Interval>;

static unsigned getIntegerWidth(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  auto integer = dyn_cast<IntegerType>(type);
  return integer ? integer.getWidth() : 0;
}

static Interval singleton(const llvm::APInt &value) { return {value, value}; }

static Range getConstantRange(arith::ConstantOp op) {
  if (auto scalar = dyn_cast<IntegerAttr>(op.getValue()))
    return singleton(scalar.getValue());
  auto dense = dyn_cast<DenseIntElementsAttr>(op.getValue());
  if (!dense || dense.getNumElements() == 0)
    return std::nullopt;
  if (dense.isSplat())
    return singleton(dense.getSplatValue<llvm::APInt>());
  // Avoid scanning large literal tables just to optimize diagnostic code.
  if (dense.getNumElements() > 4096)
    return std::nullopt;
  Range result;
  for (const auto &value : dense.getValues<llvm::APInt>()) {
    if (!result) {
      result = singleton(value);
      continue;
    }
    if (value.slt(result->min))
      result->min = value;
    if (value.sgt(result->max))
      result->max = value;
  }
  return result;
}

// A signed interval crossing zero wraps around in unsigned order. Its unsigned
// hull must then include the entire bit-width range, not just its endpoints.
static Interval unsignedHull(const Interval &range) {
  if (range.min.isNegative() && !range.max.isNegative()) {
    unsigned width = range.min.getBitWidth();
    return {llvm::APInt::getZero(width), llvm::APInt::getMaxValue(width)};
  }
  return range;
}

static bool comparisonIsTrue(arith::CmpIPredicate predicate,
                             const Interval &lhs, const Interval &rhs) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return lhs.isSingleton() && rhs.isSingleton() && lhs.min == rhs.min;
  case arith::CmpIPredicate::ne:
    return lhs.max.slt(rhs.min) || rhs.max.slt(lhs.min);
  case arith::CmpIPredicate::slt:
    return lhs.max.slt(rhs.min);
  case arith::CmpIPredicate::sle:
    return lhs.max.sle(rhs.min);
  case arith::CmpIPredicate::sgt:
    return lhs.min.sgt(rhs.max);
  case arith::CmpIPredicate::sge:
    return lhs.min.sge(rhs.max);
  default:
    break;
  }
  auto left = unsignedHull(lhs);
  auto right = unsignedHull(rhs);
  switch (predicate) {
  case arith::CmpIPredicate::ult:
    return left.max.ult(right.min);
  case arith::CmpIPredicate::ule:
    return left.max.ule(right.min);
  case arith::CmpIPredicate::ugt:
    return left.min.ugt(right.max);
  case arith::CmpIPredicate::uge:
    return left.min.uge(right.max);
  default:
    return false;
  }
}

static Range arithmeticRange(Operation *op, const Interval &lhs,
                             const Interval &rhs) {
  unsigned width = lhs.min.getBitWidth();
  // Twice the original width holds every signed product (and every sum or
  // difference). Never use nsw/nuw to assume away the overflow being checked.
  auto lmin = lhs.min.sext(width * 2);
  auto lmax = lhs.max.sext(width * 2);
  auto rmin = rhs.min.sext(width * 2);
  auto rmax = rhs.max.sext(width * 2);
  llvm::APInt min(width * 2, 0), max(width * 2, 0);
  if (isa<arith::AddIOp>(op)) {
    min = lmin + rmin;
    max = lmax + rmax;
  } else if (isa<arith::SubIOp>(op)) {
    min = lmin - rmax;
    max = lmax - rmin;
  } else {
    llvm::APInt products[] = {lmin * rmin, lmin * rmax, lmax * rmin,
                              lmax * rmax};
    auto bounds = std::minmax_element(
        std::begin(products), std::end(products),
        [](const llvm::APInt &a, const llvm::APInt &b) { return a.slt(b); });
    min = *bounds.first;
    max = *bounds.second;
  }
  // A potentially wrapping intermediate is unknown: its mathematical bounds
  // cannot be propagated as bounds of the actual fixed-width result.
  if (!min.isSignedIntN(width) || !max.isSignedIntN(width))
    return std::nullopt;
  return Interval{min.trunc(width), max.trunc(width)};
}

class ProvenRanges {
public:
  Range get(Value value, unsigned depth = 0) {
    auto found = cache.find(value);
    if (found != cache.end())
      return found->second;
    // This is deliberately a bounded, local proof, not loop or argument range
    // inference. Dynamic values and unsupported producers stay unknown.
    if (depth >= 128)
      return std::nullopt;
    Range result = infer(value, depth);
    cache.try_emplace(value, result);
    return result;
  }

private:
  Range infer(Value value, unsigned depth) {
    unsigned width = getIntegerWidth(value.getType());
    if (width == 0 || width > 64)
      return std::nullopt;
    Operation *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    // Signed bounds alone do not establish nuw's independent poison condition:
    // e.g. -1 + 1 fits signed arithmetic but wraps in unsigned arithmetic. Keep
    // all nuw producers unknown, including trunci, rather than infer from
    // poison.
    if (auto flags = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(op);
        flags && flags.hasNoUnsignedWrap())
      return std::nullopt;
    if (auto constant = dyn_cast<arith::ConstantOp>(op))
      return getConstantRange(constant);
    if (auto range = dyn_cast<MakeRangeOp>(op)) {
      // Generated getStart()/getEnd() accessors are unsigned despite TTIR's
      // signed i32 range contract; read the signed attributes instead.
      int64_t start = range.getStartAttr().getInt();
      int64_t end = range.getEndAttr().getInt();
      if (width != 32 || start >= end)
        return std::nullopt;
      return Interval{llvm::APInt(width, start, true),
                      llvm::APInt(width, end - 1, true)};
    }
    if (isa<SplatOp, BroadcastOp, ExpandDimsOp, ReshapeOp, TransOp>(op))
      return get(op->getOperand(0), depth + 1);

    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
      Range input = get(op->getOperand(0), depth + 1);
      if (!input)
        return std::nullopt;
      if (isa<arith::ExtSIOp>(op))
        return Interval{input->min.sext(width), input->max.sext(width)};
      if (isa<arith::ExtUIOp>(op)) {
        auto bounds = unsignedHull(*input);
        return Interval{bounds.min.zext(width), bounds.max.zext(width)};
      }
      // Constant truncation is evaluated as a bit operation. For non-constant
      // ranges only keep the simple case where all signed values are preserved.
      bool preservesSignedValues =
          input->min.isSignedIntN(width) && input->max.isSignedIntN(width);
      // A flagged truncation can produce poison even for a singleton constant.
      // Prove nsw independently before evaluating the discarded bits.
      if (!preservesSignedValues &&
          cast<arith::ArithIntegerOverflowFlagsInterface>(op).hasNoSignedWrap())
        return std::nullopt;
      if (input->isSingleton())
        return singleton(input->min.trunc(width));
      if (preservesSignedValues)
        return Interval{input->min.trunc(width), input->max.trunc(width)};
      return std::nullopt;
    }

    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::CmpIOp,
             arith::AndIOp>(op))
      return std::nullopt;
    Range lhs = get(op->getOperand(0), depth + 1);
    Range rhs = get(op->getOperand(1), depth + 1);
    if (!lhs || !rhs)
      return std::nullopt;
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      if (comparisonIsTrue(cmp.getPredicate(), *lhs, *rhs))
        return singleton(llvm::APInt(1, 1));
      return std::nullopt;
    }
    if (isa<arith::AndIOp>(op)) {
      if (width == 1 && lhs->isTrue() && rhs->isTrue())
        return singleton(llvm::APInt(1, 1));
      return std::nullopt;
    }
    return arithmeticRange(op, *lhs, *rhs);
  }

  llvm::DenseMap<Value, Range> cache;
};

} // namespace

void eliminateProvenOverflowAssertions(ModuleOp module) {
  ProvenRanges ranges;
  SmallVector<AssertOp> proven;
  module.walk([&](AssertOp op) {
    if (!op->hasAttr("tt.auto_overflow_assert"))
      return;
    Range condition = ranges.get(op.getCondition());
    if (condition && condition->isTrue())
      proven.push_back(op);
  });
  for (AssertOp op : proven)
    op.erase();
}

} // namespace mlir::triton
