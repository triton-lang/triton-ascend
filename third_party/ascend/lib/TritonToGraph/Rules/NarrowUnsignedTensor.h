// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_ASCEND_GRAPH_NARROW_UNSIGNED_TENSOR_H
#define TRITON_ASCEND_GRAPH_NARROW_UNSIGNED_TENSOR_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
namespace narrow_unsigned_tensor {

inline bool constant(Value value, uint64_t &bits) {
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return false;
  bits = integer.getZExtValue();
  return true;
}

// Only values proven to have zero upper bits may change signed right shift
// into unsigned right shift. In particular, an arbitrary i64 is NOT a u32.
inline bool isU32(Value value) {
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return getElementTypeOrSelf(ext.getIn().getType()).isInteger(32);
  uint64_t bits;
  return constant(value, bits) && bits <= UINT32_MAX;
}

inline Type i32Type(Type type, PatternRewriter &rewriter) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.clone(rewriter.getI32Type());
  return rewriter.getI32Type();
}

// Compute the low 32 bits without materializing wide bitwise temporaries.
// Truncation commutes with these operations, but not with right shift.
inline Value lowBits(Value value, PatternRewriter &rewriter,
                     unsigned depth = 0) {
  Location loc = value.getLoc();
  Type type = i32Type(value.getType(), rewriter);
  if (value.getType() == type)
    return value;
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  uint64_t bits;
  if (constant(value, bits)) {
    auto attr =
        rewriter.getIntegerAttr(rewriter.getI32Type(), bits & UINT32_MAX);
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return rewriter.create<arith::ConstantOp>(
          loc, type, DenseElementsAttr::get(tensor, attr));
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }
  if (depth < 8) {
    if (auto op = value.getDefiningOp<SplatOp>())
      return rewriter.create<SplatOp>(
          loc, type, lowBits(op.getSrc(), rewriter, depth + 1));
    Operation *op = value.getDefiningOp();
    if (op && isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      Value lhs = lowBits(op->getOperand(0), rewriter, depth + 1);
      Value rhs = lowBits(op->getOperand(1), rewriter, depth + 1);
      if (isa<arith::AndIOp>(op))
        return rewriter.create<arith::AndIOp>(loc, lhs, rhs);
      if (isa<arith::OrIOp>(op))
        return rewriter.create<arith::OrIOp>(loc, lhs, rhs);
      return rewriter.create<arith::XOrIOp>(loc, lhs, rhs);
    }
  }
  return rewriter.create<arith::TruncIOp>(loc, type, value);
}

struct Narrow : RewritePattern {
  explicit Narrow(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2)
      return failure();
    auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    // Select has a predicate operand first.
    if (isa<arith::SelectOp>(op))
      type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!type || !type.getElementType().isInteger(64))
      return failure();
    Location loc = op->getLoc();
    Value result;
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      if (!isU32(select.getTrueValue()) || !isU32(select.getFalseValue()))
        return failure();
      result = rewriter.create<arith::SelectOp>(
          loc, select.getCondition(), lowBits(select.getTrueValue(), rewriter),
          lowBits(select.getFalseValue(), rewriter));
    } else if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      Value lhs = op->getOperand(0), rhs = op->getOperand(1);
      uint64_t mask;
      bool masked = isa<arith::AndIOp>(op) &&
                    ((constant(lhs, mask) && mask <= UINT32_MAX) ||
                     (constant(rhs, mask) && mask <= UINT32_MAX));
      if (!masked && !(isU32(lhs) && isU32(rhs)))
        return failure();
      result = lowBits(op->getResult(0), rewriter);
    } else if (isa<arith::ShRSIOp, arith::ShRUIOp>(op)) {
      uint64_t shift;
      if (!isU32(op->getOperand(0)) || !constant(op->getOperand(1), shift) ||
          shift >= 32)
        return failure();
      result = rewriter.create<arith::ShRUIOp>(
          loc, lowBits(op->getOperand(0), rewriter),
          lowBits(op->getOperand(1), rewriter));
    } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      if (cmp.getPredicate() != arith::CmpIPredicate::eq ||
          !isU32(cmp.getLhs()))
        return failure();
      if (isU32(cmp.getRhs())) {
        rewriter.replaceOpWithNewOp<arith::CmpIOp>(
            op, cmp.getPredicate(), lowBits(cmp.getLhs(), rewriter),
            lowBits(cmp.getRhs(), rewriter));
        return success();
      }
      // A scalar threshold may still be i64: keep its upper-bit check scalar,
      // instead of silently truncating or creating a tensor<i64> comparison.
      auto splat = cmp.getRhs().getDefiningOp<SplatOp>();
      if (!splat)
        return failure();
      Value limit = rewriter.create<arith::ConstantIntOp>(loc, UINT32_MAX, 64);
      Value fits = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ule, splat.getSrc(), limit);
      Value equal = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, lowBits(cmp.getLhs(), rewriter),
          lowBits(cmp.getRhs(), rewriter));
      Value fitsTensor = rewriter.create<SplatOp>(loc, equal.getType(), fits);
      rewriter.replaceOpWithNewOp<arith::AndIOp>(op, equal, fitsTensor);
      return success();
    } else {
      return failure();
    }
    rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, type, result);
    return success();
  }
};
} // namespace narrow_unsigned_tensor
} // namespace mlir::triton
#endif
