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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_INFO_H
#define TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_INFO_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::triton::ptr {

enum class AxisKind { Invariant, Structured, Unknown };

/// A mixed-radix digit within one original dimension, ordered outer to inner.
/// Positive static lengths multiply to that dimension's extent. Bound SSA
/// strides must be available wherever the analyzed value is used.
struct SubAxisInfo {
  int64_t shape = 0;
  OpFoldResult stride;
};
struct AxisInfo {
  AxisKind kind = AxisKind::Unknown;
  /// Known axes require an index-typed stride; Unknown uses a zero placeholder.
  OpFoldResult stride;
  /// Empty when this axis has no mixed-radix decomposition.
  SmallVector<SubAxisInfo> subAxes;
  AxisInfo() = default;
  AxisInfo(AxisKind kind, OpFoldResult stride) : kind(kind), stride(stride) {}
};

/// Exact values and an optional index-typed decomposition. Integer results keep
/// fullOffsets and valueType at the source width (at most 64 bits).
/// Pointer results use a logical
/// index valueType; ordinary pointer tensors store one complete i64 offset
/// tensor, while scalar descriptors use index. Each addptr increment is widened
/// before accumulation. This analysis stage uses 64-bit index arithmetic and
/// 64-bit address offsets. As in the legacy analysis, decompositions assume
/// source integer address arithmetic does not overflow; fullOffsets retains the
/// exact source SSA. Unknown axes never promise a zero stride or a zero origin.
/// Completed results have a valid valueType and one axis per shape dimension.
struct OffsetInfo {
  Type valueType;
  OpFoldResult fullOffsets;
  /// Null when no uniform origin is known; an index zero is a known origin.
  OpFoldResult scalarOffset;
  SmallVector<AxisInfo> axes;

  ArrayRef<int64_t> getShape() const {
    if (auto tensor = dyn_cast_or_null<RankedTensorType>(valueType))
      return tensor.getShape();
    return {};
  }

  /// Whether an origin and fixed strides for all original dimensions are known.
  bool hasStructuredForm() const;
  /// Whether an origin and a complete decomposition of every axis are known.
  /// Each axis can use a fixed stride or a full mixed-radix sub-axis form.
  bool hasStructuredDecomposition() const;
  bool isUniform() const;
};

/// Build OffsetInfo from a supported integer/index SSA value as fullOffsets.
/// Axes start as Unknown and scalarOffset is empty. This only initializes
/// metadata; analysis rules supply origins and axis facts. No IR is created.
FailureOr<OffsetInfo> buildOffsetInfo(Value value, OpBuilder &builder);

/// Additional source-descriptor facts needed by block boundary checks and SCF
/// reconstruction. Coordinates are effective index values; make_tensor_ptr and
/// advance source operands retain their original i32 arithmetic.
struct BlockPtrInfo {
  /// Full parent tensor shape for boundary checks; offsets.getShape() is the
  /// block.
  SmallVector<OpFoldResult> parentShape;
  /// Current starting coordinate on each parent axis, updated by advance.
  /// Linear offsets lose this information: with strides [4, 1], [0, 4] and
  /// [1, 0] have the same linear offset but different per-axis bounds.
  SmallVector<OpFoldResult> coordinateOffsets;
  SmallVector<int32_t> order;
};

/// One result for integer offsets and scalar, tensor-of-pointer or
/// block-pointer addresses. Integer results have no base or block metadata. A
/// default result is unavailable; known offsets with Unknown axes are still
/// available results.
struct PtrInfo {
  Value base;
  OffsetInfo offsets;
  bool isEmpty() const { return !offsets.valueType; }
  /// Pointer-only queries: base must be present.
  Type getElementType() const;
  unsigned getAddressSpace() const;
  std::optional<BlockPtrInfo> block;
};

} // namespace mlir::triton::ptr
namespace mlir::triton::ptr::detail {
bool isOrdinaryPtrType(Type type);
bool isBlockPtrType(Type type);

bool validOffsetBinding(Value boundary, const OffsetInfo &components);
bool validBlockPtrBinding(Value value, const PtrInfo &components);
bool validPtrBinding(Value boundary, const PtrInfo &components);

/// Actual operands selected by boundary materialization helpers.
void collectValues(ArrayRef<OpFoldResult> fields,
                   SmallVectorImpl<Value> &values);
} // namespace mlir::triton::ptr::detail

#endif // TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_INFO_H
