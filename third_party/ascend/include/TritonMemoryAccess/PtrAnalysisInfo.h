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
#include "llvm/IR/ConstantRange.h"
#include <optional>

namespace mlir::triton::ptr {

enum class AxisKind { Invariant, Structured, Unknown };

/// A mixed-radix digit within one original dimension, ordered outer to inner.
/// The product of shapes equals that dimension's extent. Dynamic shapes in a
/// binding are caller-proven equalities. Bound SSA fields must be available
/// wherever the analyzed value is used.
struct SubAxisInfo {
  OpFoldResult shape;
  OpFoldResult stride;
};
struct AxisInfo {
  AxisKind kind = AxisKind::Unknown;
  OpFoldResult stride;
  std::optional<SmallVector<SubAxisInfo>> subAxes;
  AxisInfo() = default;
  AxisInfo(AxisKind kind, OpFoldResult stride = {})
      : kind(kind), stride(stride) {}
};

struct AnalysisOptions {
  // Common pointer queries require matching 64-bit address/index arithmetic.
  unsigned addressBitWidth = 64;
  std::optional<unsigned> indexBitWidth = 64;
};

/// Exact values and an optional index-typed decomposition. Integer results keep
/// fullOffsets and valueType at the source width. Pointer results use a logical
/// index valueType and may retain narrow addressTerms: each term is converted
/// to index separately before summation. No wide tensor is implied by that
/// type. scalarOffset and every stride describe the signed source values in
/// index arithmetic; Unknown axes never promise a zero stride or a zero origin.
struct OffsetInfo {
  Type valueType;
  SmallVector<int64_t> shape;
  OpFoldResult fullOffsets;
  /// Used for accumulated pointer offsets when fullOffsets is absent. Terms
  /// must be scalar or have valueType's shape/encoding, each with its own
  /// integer width.
  SmallVector<OpFoldResult> addressTerms;
  std::optional<OpFoldResult> scalarOffset;
  SmallVector<AxisInfo> axes;

  /// Valid origin and one fixed stride per original dimension.
  bool hasStructuredForm() const;
  /// Valid origin and a complete original-axis or mixed-radix form per axis.
  bool hasCompleteForm() const;
  bool isUniform() const;
};

/// Preserve an integer/index SSA value exactly without claiming tensor axes.
/// Scalar origins are converted to index after their source definition. A
/// rank-zero tensor also needs tensor.extract. Exact values retain their width;
/// the builder's insertion point is preserved.
FailureOr<OffsetInfo> makeOpaqueOffset(Value value, OpBuilder &builder,
                                       AnalysisOptions options = {});

/// Additional source-descriptor facts needed by block boundary checks and SCF
/// reconstruction. Coordinates are effective index values; make_tensor_ptr and
/// advance source operands retain their original i32 arithmetic.
struct BlockPtrInfo {
  /// Full parent tensor shape for boundary checks; offsets.shape is the block.
  SmallVector<OpFoldResult> parentShape;
  SmallVector<OpFoldResult> coordinateOffsets;
  SmallVector<int32_t> order;
};

/// One address representation for scalar, tensor-of-pointer and block-pointer
/// values. Shape, strides and origin live in offsets for all three forms.
/// Block metadata does not duplicate the base, layout or element type.
struct PtrInfo {
  Value base;
  OffsetInfo offsets;
  Type getElementType() const;
  unsigned getAddressSpace() const;
  std::optional<BlockPtrInfo> block;
};
/// Exact source values with a range in the original integer width.
struct IntegerInfo {
  OffsetInfo offsets;
  llvm::ConstantRange range;
};

/// Construct a boundary result without inventing a range proof. bind() checks
/// the supplied fields against the boundary type; the range is the full set.
FailureOr<IntegerInfo> makeIntegerInfo(OffsetInfo offsets,
                                       AnalysisOptions options = {});

} // namespace mlir::triton::ptr
namespace mlir::triton::ptr::detail {
bool isBlockPtrType(Type type);
bool validBlockPtrBinding(Value value, const PtrInfo &components);
bool isOrdinaryPtrType(Type type);
bool validOffsetBinding(Value boundary, const OffsetInfo &components,
                        AnalysisOptions options);
bool validPtrBinding(Value boundary, const PtrInfo &components,
                     AnalysisOptions options);

/// Actual operands selected by boundary materialization helpers.
void collectValues(ArrayRef<OpFoldResult> fields,
                   SmallVectorImpl<Value> &values);
} // namespace mlir::triton::ptr::detail

#endif // TRITON_ASCEND_MEMORY_ACCESS_PTR_ANALYSIS_INFO_H
