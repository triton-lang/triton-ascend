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

#ifndef TRITON_POINTER_ANALYSIS_TEST_UTILS_H
#define TRITON_POINTER_ANALYSIS_TEST_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace mlir::triton::pointer::test {
using Lanes = SmallVector<llvm::APInt>;
using Environment = llvm::DenseMap<Value, Lanes>;

// An interpreter of SSA integer operations, independent of analysis rules and
// affine descriptors. Index arithmetic is evaluated at the selected width.
inline std::optional<Lanes> evaluate(OpFoldResult ofr,
                                     const Environment &env = Environment(),
                                     unsigned indexWidth = 64) {
  auto attribute = [&](Attribute attr) -> std::optional<Lanes> {
    if (auto integer = dyn_cast<IntegerAttr>(attr))
      return Lanes{integer.getValue().sextOrTrunc(
          integer.getType().isIndex() ? indexWidth
                                      : integer.getValue().getBitWidth())};
    if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
      Lanes lanes;
      for (auto integer : dense.getValues<llvm::APInt>())
        lanes.push_back(dense.getElementType().isIndex()
                            ? integer.sextOrTrunc(indexWidth)
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
    for (int32_t i = range.getStart(); i < range.getEnd(); ++i)
      lanes.emplace_back(32, i);
    return lanes;
  }
  SmallVector<Lanes> operands;
  for (Value operand : op->getOperands()) {
    auto lanes = evaluate(OpFoldResult(operand), env, indexWidth);
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
    if (operands[0].size() != 1 || op->getNumOperands() != 1)
      return std::nullopt;
    return operands[0];
  }
  if (isa<triton::ExpandDimsOp>(op))
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
    unsigned width =
        type.isIndex() ? indexWidth : cast<IntegerType>(type).getWidth();
    bool unsignedCast = isa<arith::ExtUIOp, arith::IndexCastUIOp>(op);
    for (auto &integer : operands[0])
      integer = unsignedCast ? integer.zextOrTrunc(width)
                             : integer.sextOrTrunc(width);
    return operands[0];
  }
  return std::nullopt;
}
} // namespace mlir::triton::pointer::test
#endif
