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
#include "llvm/IR/ConstantRange.h"
#include <optional>
namespace mlir::triton::ptr {

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
/// Extract from fullOffsets and convert the scalar to index; if fullOffsets is
/// absent, reconstruct it from a complete decomposition (e.g. a block pointer).
/// indices must be available index values, one per logical axis.
/// This consumes current SSA facts and preserves the builder insertion point.
FailureOr<OpFoldResult> materializeOffsetAt(const OffsetInfo &offsets,
                                            ValueRange indices,
                                            OpBuilder &builder, Location loc);

} // namespace mlir::triton::ptr
namespace mlir::triton::ptr::detail {
struct ConcreteArithmetic {
  Type type;
  OpBuilder &builder;
  Location loc;
  FailureOr<OpFoldResult> add(OpFoldResult lhs, OpFoldResult rhs,
                              bool subtract) {
    return foldOrCreateBinary(subtract ? BinaryKind::Subtract : BinaryKind::Add,
                              lhs, rhs, type, builder, loc);
  }
  FailureOr<OpFoldResult> multiply(OpFoldResult lhs, OpFoldResult rhs) {
    return foldOrCreateBinary(BinaryKind::Multiply, lhs, rhs, type, builder,
                              loc);
  }
};

// Compare identity or integer constants in the 64-bit descriptor domain.
bool equalConstant(OpFoldResult lhs, OpFoldResult rhs);
bool zero(OpFoldResult value);

// Uniform origin materialization.
/// Populate a missing origin once all axes are invariant. May extract and cast
/// the value, placing helper IR after its actual scalar source.
void setUniformOrigin(OffsetInfo &info, Value value, OpBuilder &builder,
                      OpFoldResult scalar = {});

// Address types and complete offset values.
// Logical descriptors use index; complete tensor offsets use i64 storage.
Type addressValueType(Type source, OpBuilder &builder);
Type addressStorageType(Type source, OpBuilder &builder);
/// Use fullOffsets or a uniform origin; sign-extend without narrowing and splat
/// scalars to tensor targets at the caller's insertion point when necessary.
FailureOr<Value> materializeFullOffset(const OffsetInfo &info, Type target,
                                       OpBuilder &builder, Location loc);

// Axis alignment, queries and normalization.
/// Express compatible axes in the same sub-axis partition, if provable.
LogicalResult alignAxes(AxisInfo &lhs, AxisInfo &rhs,
                        ConcreteArithmetic &arithmetic);
SmallVector<OpFoldResult> axisStrides(const OffsetInfo &info);
LogicalResult normalizeAxes(OffsetInfo &info, OpBuilder &builder);

// Divisibility proofs and exact unit conversion.
bool canDivideDecomposition(const OffsetInfo &info, uint64_t divisor);
/// After the full value is proved divisible, convert descriptor fields whose
/// exact division can also be proved. Leave fullOffsets to the caller.
void divideDecomposition(OffsetInfo &info, uint64_t divisor, OpBuilder &builder,
                         Location loc);

// Range proofs for an existing decomposition.
bool decompositionFitsSigned(const OffsetInfo &info, unsigned width);
/// Bounds of the mathematical decomposition, not the wrapped fullOffsets.
std::optional<llvm::ConstantRange>
constantDecompositionRange(const OffsetInfo &info, unsigned width);
} // namespace mlir::triton::ptr::detail

#endif
