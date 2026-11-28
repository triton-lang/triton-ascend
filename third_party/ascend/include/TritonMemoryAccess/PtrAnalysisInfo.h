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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::arith {
class SelectOp;
}
namespace mlir::triton {
class MakeTensorPtrOp;
class AdvanceOp;
} // namespace mlir::triton

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
  /// Index-typed stride for known axes. For Unknown, this field is an unused
  /// placeholder, normally zero.
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
  /// Required for integer and ordinary pointer results, including bindings.
  /// Block pointers may omit it when their decomposition is complete.
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
  /// Base passed to make_tensor_ptr, including any earlier scalar addptr.
  /// PtrInfo::base may instead be an earlier root whose offset is in the
  /// address origin. Use this base with coordinates when rebuilding a block.
  Value descriptorBase;
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
/// block-pointer addresses. Integer results have no base or block metadata.
/// Successful analysis returns an initialized result; inability to form one is
/// failure. Known offsets with Unknown axes are still valid results.
struct PtrInfo {
  /// Scalar pointer, or an exact tensor of per-lane bases with the same shape
  /// as the result. Offset facts describe displacements relative to this base;
  /// uniform offsets do not imply uniform addresses when the base is a tensor.
  Value base;
  OffsetInfo offsets;
  bool isEmpty() const { return !offsets.valueType; }
  bool hasScalarBase() const {
    return base && !isa<RankedTensorType>(base.getType());
  }
  /// Pointer-only queries: base must be present.
  Type getElementType() const;
  unsigned getAddressSpace() const;
  std::optional<BlockPtrInfo> block;
};

/// Preservation of descriptor components relative to one fixed reference.
/// False proves that the component can reuse the reference/initial field;
/// true means it may change. These are not numeric values or a proof that
/// different descriptor layouts/types can be joined.
struct ChangeInfo {
  bool base = true;
  bool uniform = true;
  bool residual = true;
  SmallVector<bool> strides;
  // Block-only fields. Ordinary pointers have empty coordinate/shape vectors.
  SmallVector<bool> parentShape;
  SmallVector<bool> coordinates;
  bool order = true;
};

} // namespace mlir::triton::ptr
namespace mlir::triton::ptr::detail {
bool isOrdinaryPtrType(Type type);
bool isBlockPtrType(Type type);

// Initial result construction.
/// Build a known axis, classifying a zero stride as invariant.
AxisInfo buildAxisInfo(OpFoldResult stride);
/// Build an axis from a complete, valid sub-axis partition. Retain the
/// partition even when its coefficients prove a single axis stride.
AxisInfo buildAxisInfo(SmallVector<SubAxisInfo> subAxes, OpBuilder &builder);
/// Build an invariant axis with an index-typed zero stride.
AxisInfo makeInvariantAxisInfo(OpBuilder &builder);
/// Build invariant axes for a known-uniform integer value and obtain an index
/// origin from scalar/value. May create extraction and cast helpers.
OffsetInfo buildUniformOffsetInfo(Value value, OpBuilder &builder,
                                  OpFoldResult scalar = {});
/// Build integer facts from the value itself, without analyzing its operands.
/// May extract and cast an available uniform origin.
FailureOr<PtrInfo> buildIntegerInfo(Value value, OpBuilder &builder);
/// Use an ordinary scalar or opaque pointer tensor as the exact base, with
/// zero displacements. Dynamic tensors may need pure shape/zero-fill helpers.
FailureOr<PtrInfo> buildPointerInfo(Value value, OpBuilder &builder);

// Integer offset combinations.
/// These operations consume analyzed inputs at the caller-selected insertion
/// point. They do not recursively query operands or change session state.
FailureOr<PtrInfo> combineElementwise(Value value, const PtrInfo &lhs,
                                      const PtrInfo &rhs, OpBuilder &builder);
FailureOr<OffsetInfo> addOffsetInfo(OpFoldResult complete,
                                    const OffsetInfo &lhs,
                                    const OffsetInfo &rhs, bool subtract,
                                    OpBuilder &builder, Location loc);
FailureOr<OffsetInfo> scaleOffsetInfo(OpFoldResult complete,
                                      const OffsetInfo &source,
                                      OpFoldResult scale, OpBuilder &builder,
                                      Location loc);
PtrInfo refineDivRem(Value value, const PtrInfo &lhsInfo,
                     const PtrInfo &rhsInfo, const PtrInfo &fallback,
                     bool divide, OpBuilder &builder);

// Offset shape transformations.
/// Source facts must match the source value, and shape parameters must follow
/// the verified operation. Reshape preserves uniformity and axes changed only
/// by inserting or removing unit dimensions; other axis facts become unknown.
OffsetInfo reshapeOffsetInfo(OpFoldResult complete, RankedTensorType targetType,
                             const OffsetInfo &sourceInfo, OpBuilder &builder);
/// Slice keeps complete offsets and known origin/stride facts for dynamic
/// extents; sub-axis partitions require a proven unchanged extent.
FailureOr<OffsetInfo>
sliceOffsetInfo(OpFoldResult complete, RankedTensorType targetType,
                const OffsetInfo &sourceInfo, ArrayRef<OpFoldResult> starts,
                ArrayRef<OpFoldResult> steps, ArrayRef<bool> dropped,
                OpBuilder &builder, Location loc);

// Ordinary pointer offset accumulation.
/// Widen the integer increment before adding it to the parent's address offset.
FailureOr<PtrInfo> addPtrOffsets(Value value, const PtrInfo &parent,
                                 const OffsetInfo &delta, OpBuilder &builder);

// Selection and insertion of analyzed results.
FailureOr<PtrInfo> combineSelectInfo(arith::SelectOp op, const PtrInfo &lhsInfo,
                                     const PtrInfo &rhsInfo,
                                     OpBuilder &builder);
FailureOr<PtrInfo> insertPtrOffsets(
    Value value, const PtrInfo &source, const PtrInfo &dest, Type sourceType,
    llvm::function_ref<Value(Value, Value)> insert, OpBuilder &builder);

// Block descriptor construction and coordinate updates.
FailureOr<PtrInfo> makeBlockInfo(triton::MakeTensorPtrOp op,
                                 const PtrInfo &base,
                                 ArrayRef<OpFoldResult> coordinates,
                                 OpBuilder &builder);
FailureOr<PtrInfo> advanceBlockInfo(triton::AdvanceOp op, const PtrInfo &parent,
                                    ArrayRef<OpFoldResult> delta,
                                    OpBuilder &builder);

// Binding validation and dependency collection.
LogicalResult validateInfo(Value value, const PtrInfo &info);
bool validOffsetBinding(Value boundary, const OffsetInfo &components);
bool validBlockPtrBinding(Value value, const PtrInfo &components);
bool validPtrBinding(Value boundary, const PtrInfo &components);

/// Actual operands selected by boundary materialization helpers.
void collectValues(ArrayRef<OpFoldResult> fields,
                   SmallVectorImpl<Value> &values);
} // namespace mlir::triton::ptr::detail

#endif // TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_INFO_H
