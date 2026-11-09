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

#ifndef TRITON_PTR_ANALYSIS_TEST_UTILS_H
#define TRITON_PTR_ANALYSIS_TEST_UTILS_H

#include "TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace mlir::triton::ptr::test {
// Verification-only tensor oracle and dead-helper cleanup. Production consumers
// use materializeOffsetAt and their own lowering-specific helper ownership.
FailureOr<OffsetInfo> buildAddressView(const PtrInfo &ptr, OpBuilder &builder,
                                       Location loc);
// Pointer casts are eligible only when explicitly recorded as new helpers;
// a dead original source cast must not be mistaken for analysis-owned IR.
void eraseDeadPtrArithmetic(Operation *scope, RewriterBase &rewriter,
                            ArrayRef<Operation *> pointerHelpers = {});
using Lanes = SmallVector<llvm::APInt>;
using Environment = llvm::DenseMap<Value, Lanes>;

// An interpreter of SSA integer operations, independent of analysis rules and
// affine descriptors. Index arithmetic is evaluated at 64 bits.
inline std::optional<Lanes> evaluate(OpFoldResult ofr,
                                     const Environment &env = Environment()) {
  if (!ofr)
    return std::nullopt;
  auto attribute = [&](Attribute attr) -> std::optional<Lanes> {
    if (auto integer = dyn_cast<IntegerAttr>(attr))
      return Lanes{integer.getValue().sextOrTrunc(
          integer.getType().isIndex() ? 64 : integer.getValue().getBitWidth())};
    if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
      Lanes lanes;
      for (auto integer : dense.getValues<llvm::APInt>())
        lanes.push_back(dense.getElementType().isIndex()
                            ? integer.sextOrTrunc(64)
                            : integer);
      return lanes;
    }
    return std::nullopt;
  };
  if (auto attr = dyn_cast<Attribute>(ofr))
    return attribute(attr);
  Value value = cast<Value>(ofr);
  if (auto it = env.find(value); it != env.end())
    return it->second;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto constant = dyn_cast<arith::ConstantOp>(op))
    return attribute(constant.getValue());
  if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
    Lanes lanes;
    int64_t start = range.getStartAttr().getInt();
    int64_t end = range.getEndAttr().getInt();
    for (int64_t i = start; i < end; ++i)
      lanes.emplace_back(32, i, /*isSigned=*/true);
    return lanes;
  }
  SmallVector<Lanes> operands;
  for (Value operand : op->getOperands()) {
    auto lanes = evaluate(OpFoldResult(operand), env);
    if (!lanes)
      return std::nullopt;
    operands.push_back(*lanes);
  }
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
    if (operands[0].size() != operands[1].size())
      return std::nullopt;
    for (unsigned i = 0; i < operands[0].size(); ++i) {
      auto &x = operands[0][i];
      auto y = operands[1][i];
      x = isa<arith::AddIOp>(op)   ? x + y
          : isa<arith::SubIOp>(op) ? x - y
                                   : x * y;
    }
    return operands[0];
  }
  if (isa<arith::DivUIOp, arith::RemUIOp, arith::DivSIOp, arith::RemSIOp>(op)) {
    if (operands[0].size() != operands[1].size())
      return std::nullopt;
    for (unsigned i = 0; i < operands[0].size(); ++i) {
      if (operands[1][i].isZero())
        return std::nullopt;
      auto x = operands[0][i], y = operands[1][i];
      if (isa<arith::DivSIOp, arith::RemSIOp>(op) && x.isMinSignedValue() &&
          y.isAllOnes())
        return std::nullopt;
      operands[0][i] = isa<arith::DivUIOp>(op)   ? x.udiv(y)
                       : isa<arith::RemUIOp>(op) ? x.urem(y)
                       : isa<arith::DivSIOp>(op) ? x.sdiv(y)
                                                 : x.srem(y);
    }
    return operands[0];
  }
  if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::MinSIOp,
          arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::CmpIOp>(op)) {
    Lanes lanes;
    for (unsigned i = 0; i < operands[0].size(); ++i) {
      auto x = operands[0][i], y = operands[1][i];
      if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        bool result;
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          result = x == y;
          break;
        case arith::CmpIPredicate::ne:
          result = x != y;
          break;
        case arith::CmpIPredicate::slt:
          result = x.slt(y);
          break;
        case arith::CmpIPredicate::sle:
          result = x.sle(y);
          break;
        case arith::CmpIPredicate::sgt:
          result = x.sgt(y);
          break;
        case arith::CmpIPredicate::sge:
          result = x.sge(y);
          break;
        case arith::CmpIPredicate::ult:
          result = x.ult(y);
          break;
        case arith::CmpIPredicate::ule:
          result = x.ule(y);
          break;
        case arith::CmpIPredicate::ugt:
          result = x.ugt(y);
          break;
        case arith::CmpIPredicate::uge:
          result = x.uge(y);
          break;
        }
        lanes.emplace_back(1, result);
      } else {
        lanes.push_back(isa<arith::AndIOp>(op)    ? x & y
                        : isa<arith::OrIOp>(op)   ? x | y
                        : isa<arith::XOrIOp>(op)  ? x ^ y
                        : isa<arith::MinSIOp>(op) ? (x.slt(y) ? x : y)
                        : isa<arith::MaxSIOp>(op) ? (x.sgt(y) ? x : y)
                        : isa<arith::MinUIOp>(op) ? (x.ult(y) ? x : y)
                                                  : (x.ugt(y) ? x : y));
      }
    }
    return lanes;
  }
  if (isa<arith::SelectOp>(op)) {
    Lanes lanes;
    for (unsigned i = 0; i < operands[1].size(); ++i)
      lanes.push_back(operands[0][operands[0].size() == 1 ? 0 : i].isZero()
                          ? operands[2][i]
                          : operands[1][i]);
    return lanes;
  }
  if (isa<triton::SplatOp>(op))
    return Lanes(cast<RankedTensorType>(value.getType()).getNumElements(),
                 operands[0][0]);
  if (isa<tensor::ExtractOp>(op)) {
    auto type = cast<RankedTensorType>(op->getOperand(0).getType());
    int64_t lane = 0;
    for (unsigned i = 1; i < operands.size(); ++i)
      lane = lane * type.getDimSize(i - 1) + operands[i][0].getSExtValue();
    if (lane < 0 || lane >= int64_t(operands[0].size()))
      return std::nullopt;
    return Lanes{operands[0][lane]};
  }
  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op)) {
    auto from = slice.getSourceType(), to = slice.getType();
    auto dropped = slice.getDroppedDims();
    auto starts = slice.getMixedOffsets(), steps = slice.getMixedStrides();
    Lanes lanes;
    for (int64_t i = 0; i < to.getNumElements(); ++i) {
      int64_t rest = i, offset = 0, stride = 1;
      int out = to.getRank() - 1;
      for (int d = from.getRank() - 1; d >= 0; --d) {
        auto start = evaluate(starts[d], env), step = evaluate(steps[d], env);
        if (!start || !step)
          return std::nullopt;
        int64_t coord = 0;
        if (!dropped.test(d)) {
          coord = rest % to.getDimSize(out);
          rest /= to.getDimSize(out--);
        }
        coord = (*start)[0].getSExtValue() + coord * (*step)[0].getSExtValue();
        offset += coord * stride;
        stride *= from.getDimSize(d);
      }
      if (offset < 0 || offset >= int64_t(operands[0].size()))
        return std::nullopt;
      lanes.push_back(operands[0][offset]);
    }
    return lanes;
  }
  if (auto insert = dyn_cast<tensor::InsertOp>(op)) {
    auto type = cast<RankedTensorType>(insert.getDest().getType());
    int64_t lane = 0;
    for (unsigned i = 2; i < operands.size(); ++i)
      lane = lane * type.getDimSize(i - 2) + operands[i][0].getSExtValue();
    if (lane < 0 || lane >= int64_t(operands[1].size()))
      return std::nullopt;
    operands[1][lane] = operands[0][0];
    return operands[1];
  }
  if (auto insert = dyn_cast<tensor::InsertSliceOp>(op)) {
    auto from = insert.getSourceType(), to = insert.getType();
    auto dropped = insert.getDroppedDims();
    auto starts = insert.getMixedOffsets(), steps = insert.getMixedStrides();
    Lanes lanes = operands[1];
    for (int64_t i = 0; i < from.getNumElements(); ++i) {
      int64_t rest = i, offset = 0, stride = 1;
      int in = from.getRank() - 1;
      for (int d = to.getRank() - 1; d >= 0; --d) {
        auto start = evaluate(starts[d], env);
        auto step = evaluate(steps[d], env);
        if (!start || !step)
          return std::nullopt;
        int64_t coord = 0;
        if (!dropped.test(d)) {
          coord = rest % from.getDimSize(in);
          rest /= from.getDimSize(in--);
        }
        offset +=
            ((*start)[0].getSExtValue() + coord * (*step)[0].getSExtValue()) *
            stride;
        stride *= to.getDimSize(d);
      }
      if (offset < 0 || offset >= int64_t(lanes.size()))
        return std::nullopt;
      lanes[offset] = operands[0][i];
    }
    return lanes;
  }
  if (isa<triton::ExpandDimsOp, triton::ReshapeOp, tensor::CastOp>(op))
    return operands[0];
  if (isa<triton::BroadcastOp>(op)) {
    auto from = cast<RankedTensorType>(op->getOperand(0).getType());
    auto to = cast<RankedTensorType>(value.getType());
    Lanes lanes;
    for (int64_t i = 0; i < to.getNumElements(); ++i) {
      int64_t rest = i, offset = 0, stride = 1;
      for (int d = to.getRank() - 1; d >= 0; --d) {
        int64_t coord = rest % to.getDimSize(d);
        rest /= to.getDimSize(d);
        offset += (from.getDimSize(d) == 1 ? 0 : coord) * stride;
        stride *= from.getDimSize(d);
      }
      lanes.push_back(operands[0][offset]);
    }
    return lanes;
  }
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
          arith::IndexCastUIOp>(op)) {
    Type type = value.getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    unsigned width = type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
    bool unsignedCast = isa<arith::ExtUIOp, arith::IndexCastUIOp>(op);
    for (auto &integer : operands[0])
      integer = unsignedCast ? integer.zextOrTrunc(width)
                             : integer.sextOrTrunc(width);
    return operands[0];
  }
  return std::nullopt;
}

inline std::optional<Lanes> evaluate(const OffsetInfo &info,
                                     const Environment &env = Environment()) {
  // Evaluate the exact source first, then interpret its signed value as an
  // index offset. The SSA interpreter above still evaluates at source width.
  if (info.fullOffsets) {
    auto values = evaluate(info.fullOffsets, env);
    if (values)
      for (auto &value : *values)
        value = value.sextOrTrunc(64);
    return values;
  }
  return std::nullopt;
}

inline SmallVector<AxisKind> axisKinds(const OffsetInfo &info) {
  SmallVector<AxisKind> kinds;
  for (const auto &axis : info.axes)
    kinds.push_back(axis.kind);
  return kinds;
}
inline SmallVector<OpFoldResult> axisStrides(const OffsetInfo &info) {
  SmallVector<OpFoldResult> strides;
  for (const auto &axis : info.axes)
    strides.push_back(axis.stride);
  return strides;
}

inline const PtrInfo *getInfo(const PtrInfo *result) {
  return result && !result->isEmpty() ? result : nullptr;
}
inline FailureOr<PtrInfo> getInfo(FailureOr<PtrInfo> result) {
  if (failed(result) || result->isEmpty())
    return failure();
  return *result;
}
inline const OffsetInfo *getOffsets(const PtrInfo *result) {
  return getInfo(result) ? &result->offsets : nullptr;
}
inline FailureOr<OffsetInfo> getOffsets(FailureOr<PtrInfo> result) {
  const auto *offsets = succeeded(result) ? getOffsets(&*result) : nullptr;
  if (!offsets)
    return failure();
  return *offsets;
}
template <typename Context>
FailureOr<SmallVector<PtrInfo, 0>> analyzeInputs(Context &context,
                                                 ValueRange values) {
  SmallVector<PtrInfo, 0> results;
  for (Value value : values) {
    auto result = context.analyze(value);
    if (failed(result))
      return failure();
    results.push_back(*result);
  }
  return results;
}
} // namespace mlir::triton::ptr::test
#endif
