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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_PTR_INFO_UTILS_H
#define TRITON_ASCEND_MEMORY_ACCESS_PTR_INFO_UTILS_H
#include "TritonMemoryAccess/IntegerUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
namespace mlir::triton::ptr {

/// A proven decomposition: origin + coordinate terms + irregular contribution.
/// Missing fields mean no proof for that part; canonical zero is an explicit
/// Component supplied by the arithmetic adapter. This distinction preserves
/// an irregular component's identity during a uniform-only update.
template <typename Component> struct OffsetTerms {
  std::optional<Component> origin;
  SmallVector<std::optional<Component>> coefficients;
  std::optional<Component> irregular;
};

/// Shape facts use the same original-dimension mapping for concrete axes and
/// symbolic coefficients. The caller constructs a typed invariant fact.
template <typename Axis, typename MakeInvariant>
FailureOr<SmallVector<Axis>>
broadcastAxisFacts(ArrayRef<Axis> axes, ArrayRef<int64_t> sourceShape,
                   ArrayRef<int64_t> targetShape, MakeInvariant invariant) {
  if (axes.size() != sourceShape.size() || axes.size() != targetShape.size())
    return failure();
  SmallVector<Axis> result(axes);
  for (unsigned i = 0; i < axes.size(); ++i) {
    if (sourceShape[i] == targetShape[i])
      continue;
    if (sourceShape[i] != 1)
      return failure();
    result[i] = invariant();
  }
  return result;
}
template <typename Axis, typename MakeInvariant>
FailureOr<SmallVector<Axis>> expandAxisFacts(ArrayRef<Axis> axes, unsigned axis,
                                             MakeInvariant invariant) {
  if (axis > axes.size())
    return failure();
  SmallVector<Axis> result(axes);
  result.insert(result.begin() + axis, invariant());
  return result;
}

template <typename Component, typename Arithmetic>
FailureOr<OffsetTerms<Component>>
addOffsetTerms(const OffsetTerms<Component> &lhs,
               const OffsetTerms<Component> &rhs, bool subtract,
               Arithmetic &arithmetic) {
  if (lhs.coefficients.size() != rhs.coefficients.size())
    return failure();
  OffsetTerms<Component> info;
  auto combine = [&](const auto &left, const auto &right, auto &out) {
    if (!left || !right)
      return success();
    auto value = arithmetic.add(*left, *right, subtract);
    if (failed(value))
      return failure();
    out = std::move(*value);
    return success();
  };
  if (failed(combine(lhs.origin, rhs.origin, info.origin)) ||
      failed(combine(lhs.irregular, rhs.irregular, info.irregular)))
    return failure();
  info.coefficients.resize(lhs.coefficients.size());
  for (unsigned i = 0; i < lhs.coefficients.size(); ++i)
    if (failed(combine(lhs.coefficients[i], rhs.coefficients[i],
                       info.coefficients[i])))
      return failure();
  return info;
}

template <typename Component, typename Arithmetic>
FailureOr<OffsetTerms<Component>>
scaleOffsetTerms(const OffsetTerms<Component> &source, const Component &scale,
                 Arithmetic &arithmetic) {
  auto info = source;
  auto apply = [&](auto &field) {
    if (!field)
      return success();
    auto value = arithmetic.multiply(*field, scale);
    if (failed(value))
      return failure();
    field = std::move(*value);
    return success();
  };
  if (failed(apply(info.origin)) || failed(apply(info.irregular)))
    return failure();
  for (auto &coefficient : info.coefficients)
    if (failed(apply(coefficient)))
      return failure();
  return info;
}

template <typename Component, typename Arithmetic>
FailureOr<SmallVector<Component>>
advanceCoordinateTerms(ArrayRef<Component> coordinates,
                       ArrayRef<Component> deltas, Arithmetic &arithmetic) {
  if (coordinates.size() != deltas.size())
    return failure();
  SmallVector<Component> info;
  for (unsigned i = 0; i < coordinates.size(); ++i) {
    auto value = arithmetic.add(coordinates[i], deltas[i], false);
    if (failed(value))
      return failure();
    info.push_back(std::move(*value));
  }
  return info;
}

template <typename Component, typename Arithmetic>
FailureOr<Component>
addScaledCoordinateTerms(Component origin, ArrayRef<Component> coordinates,
                         ArrayRef<Component> strides, Arithmetic &arithmetic) {
  if (coordinates.size() != strides.size())
    return failure();
  for (unsigned i = 0; i < coordinates.size(); ++i) {
    auto term = arithmetic.multiply(coordinates[i], strides[i]);
    if (failed(term))
      return failure();
    auto sum = arithmetic.add(origin, *term, false);
    if (failed(sum))
      return failure();
    origin = std::move(*sum);
  }
  return origin;
}
/// Produce one exact index element offset at the caller's insertion point.
/// Narrow terms are extracted before extension. indices must be available index
/// values, one per logical axis. Full values/terms take precedence over proofs;
/// absent exact data requires a complete decomposition. No tensor is widened.
/// This consumes current SSA facts and preserves the builder insertion point.
FailureOr<OpFoldResult> materializeOffsetAt(const OffsetInfo &offsets,
                                            ValueRange indices,
                                            OpBuilder &builder, Location loc,
                                            AnalysisOptions options = {});

} // namespace mlir::triton::ptr
namespace mlir::triton::ptr::detail {
struct ConcreteArithmetic {
  Type type;
  AnalysisOptions options;
  OpBuilder &builder;
  Location loc;
  FailureOr<OpFoldResult> add(OpFoldResult lhs, OpFoldResult rhs,
                              bool subtract) {
    return foldOrCreateBinary(subtract ? BinaryKind::Subtract : BinaryKind::Add,
                              lhs, rhs, type, options, builder, loc);
  }
  FailureOr<OpFoldResult> multiply(OpFoldResult lhs, OpFoldResult rhs) {
    return foldOrCreateBinary(BinaryKind::Multiply, lhs, rhs, type, options,
                              builder, loc);
  }
};
LogicalResult normalizeAxes(OffsetInfo &info, AnalysisOptions options,
                            OpBuilder &builder);
FailureOr<OffsetInfo> addOffsets(OpFoldResult complete, const OffsetInfo &lhs,
                                 const OffsetInfo &rhs, bool subtract,
                                 AnalysisOptions options, OpBuilder &builder,
                                 Location loc);
FailureOr<OffsetInfo> scaleOffsets(OpFoldResult complete,
                                   const OffsetInfo &source, OpFoldResult scale,
                                   AnalysisOptions options, OpBuilder &builder,
                                   Location loc);
FailureOr<OffsetInfo> broadcastOffsets(OpFoldResult complete, Type target,
                                       const OffsetInfo &source,
                                       OpBuilder &builder);
FailureOr<OffsetInfo> expandDimsOffsets(OpFoldResult complete, Type target,
                                        const OffsetInfo &source, unsigned axis,
                                        OpBuilder &builder);
bool decompositionFitsSigned(const OffsetInfo &info, unsigned width);
/// Bounds of the mathematical decomposition, not the wrapped fullOffsets.
std::optional<llvm::ConstantRange>
constantDecompositionRange(const OffsetInfo &info, unsigned width);
FailureOr<OffsetInfo> reshapeOffsets(OpFoldResult complete,
                                     RankedTensorType target,
                                     const OffsetInfo &source,
                                     AnalysisOptions options,
                                     OpBuilder &builder, Location loc);
FailureOr<OffsetInfo>
sliceOffsets(OpFoldResult complete, RankedTensorType target,
             const OffsetInfo &source, ArrayRef<OpFoldResult> starts,
             ArrayRef<OpFoldResult> steps, ArrayRef<bool> dropped,
             AnalysisOptions options, OpBuilder &builder, Location loc);
SmallVector<OpFoldResult> axisStrides(const OffsetInfo &info);
} // namespace mlir::triton::ptr::detail

#endif
