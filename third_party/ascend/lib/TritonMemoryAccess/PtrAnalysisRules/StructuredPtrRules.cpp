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

#include "TritonMemoryAccess/PtrAnalysisRules/StructuredPtrRules.h"

namespace mlir::triton::ptr {
FailureOr<StructuredPtrRules::Result>
StructuredPtrRules::parseOperation(Value value, Analyze analyze,
                                   OpBuilder &builder) const {
  if (auto op = value.getDefiningOp<arith::DivUIOp>())
    return parseDivRem(value, op.getLhs(), op.getRhs(), true, analyze, builder);
  if (auto op = value.getDefiningOp<arith::RemUIOp>())
    return parseDivRem(value, op.getLhs(), op.getRhs(), false, analyze,
                       builder);
  if (auto op = value.getDefiningOp<arith::DivSIOp>())
    return parseDivRem(value, op.getLhs(), op.getRhs(), true, analyze, builder);
  if (auto op = value.getDefiningOp<arith::RemSIOp>())
    return parseDivRem(value, op.getLhs(), op.getRhs(), false, analyze,
                       builder);
  return common.parseOperation(value, analyze, builder);
}

FailureOr<StructuredPtrRules::Result>
StructuredPtrRules::parseDivRem(Value value, Value lhs, Value rhs, bool divide,
                                Analyze analyze, OpBuilder &builder) const {
  auto lhsInfo = analyze(lhs);
  auto rhsInfo = analyze(rhs);
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  builder.setInsertionPointAfterValue(value);
  auto *left = getIntegerInfo(*lhsInfo), *right = getIntegerInfo(*rhsInfo);
  if (!left || !right)
    return Result{};
  auto fallback = common.combineElementwise(value, *left, *right, builder);
  if (failed(fallback))
    return failure();
  return refineDivRem(value, *lhsInfo, *rhsInfo, *fallback, divide, builder);
}

FailureOr<StructuredPtrRules::Result>
StructuredPtrRules::refineDivRem(Value value, const Result &lhsInfo,
                                 const Result &rhsInfo, const Result &fallback,
                                 bool divide, OpBuilder &builder) const {
  auto *left = getIntegerInfo(lhsInfo), *right = getIntegerInfo(rhsInfo);
  if (!left || !right || !right->range.isSingleElement())
    return fallback;
  unsigned width = left->range.getBitWidth();
  // Prove the mathematical source decomposition, not just its modular range.
  auto range = detail::constantDecompositionRange(left->offsets, width);
  if (!range || !range->isAllNonNegative())
    return fallback;
  auto origin = detail::getIntegerConstant(*left->offsets.scalarOffset, 64);
  auto divisor = right->range.getSingleElement()->sextOrTrunc(64);
  if (failed(origin) || divisor.isZero() || divisor.isNegative() ||
      !origin->urem(divisor).isZero())
    return fallback;
  uint64_t d = divisor.getZExtValue();
  OffsetInfo info = left->offsets;
  info.fullOffsets = value;
  Type scalar = builder.getIndexType();
  info.scalarOffset = detail::scalarIntegerAttr(
      scalar, divide ? origin->udiv(divisor) : llvm::APInt(64, 0), builder);
  __int128 residualMaximum = 0;
  for (unsigned i = 0; i < info.axes.size(); ++i) {
    auto &axis = info.axes[i];
    SmallVector<SubAxisInfo> input =
        axis.subAxes.value_or(SmallVector<SubAxisInfo>{
            {builder.getIndexAttr(info.shape[i]), axis.stride}});
    SmallVector<SubAxisInfo> output;
    for (const auto &sub : input) {
      auto n = detail::getIntegerConstant(sub.shape, 64);
      auto stride = detail::getIntegerConstant(sub.stride, 64);
      if (failed(n) || n->isNegative() || n->isZero() || failed(stride) ||
          stride->isNegative())
        return fallback;
      uint64_t size = n->getZExtValue(), c = stride->getZExtValue();
      auto push = [&](uint64_t length, uint64_t coefficient) {
        output.push_back({builder.getIndexAttr(length),
                          detail::scalarIntegerAttr(
                              scalar, llvm::APInt(64, coefficient), builder)});
      };
      if (c % d == 0) {
        push(size, divide ? c / d : 0);
      } else if (d % c == 0 && size % (d / c) == 0) {
        uint64_t inner = d / c;
        push(size / inner, divide ? 1 : 0);
        push(inner, divide ? 0 : c);
        residualMaximum += d - c;
      } else if (__int128(size - 1) * c < d) {
        push(size, divide ? 0 : c);
        residualMaximum += __int128(size - 1) * c;
      } else {
        return fallback;
      }
    }
    axis.subAxes = std::move(output);
  }
  // Multiple residual digits may carry into the quotient. Only accept their
  // independent decomposition when their total stays below the divisor.
  if (residualMaximum >= d ||
      failed(detail::normalizeAxes(info, getOptions(), builder)))
    return fallback;
  const auto &l = left->range, &r = right->range;
  return Result(IntegerInfo{std::move(info), divide ? l.udiv(r) : l.urem(r)});
}
} // namespace mlir::triton::ptr
