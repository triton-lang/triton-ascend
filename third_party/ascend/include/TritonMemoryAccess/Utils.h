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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_UTILS_H
#define TRITON_ASCEND_MEMORY_ACCESS_UTILS_H

#include "TritonMemoryAccess/OpFoldResultUtils.h"
#include "TritonMemoryAccess/PtrAnalysisInfo.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ConstantRange.h"
#include <cassert>
#include <optional>

namespace mlir {
class RewriterBase;
}

namespace mlir::triton::ptr {

/// Broadcast known axes across verified singleton dimensions.
void broadcastAxes(SmallVectorImpl<AxisInfo> &axes,
                   ArrayRef<int64_t> sourceShape, ArrayRef<int64_t> targetShape,
                   const AxisInfo &invariant);

/// Produce one exact index element offset at the caller's insertion point.
/// Extract from fullOffsets and convert the scalar to index; if fullOffsets is
/// absent, reconstruct it from a complete decomposition (e.g. a block pointer).
/// indices must be available index values, one per logical axis.
/// This consumes current SSA facts and preserves the builder insertion point.
FailureOr<OpFoldResult> materializeOffsetAt(const OffsetInfo &offsets,
                                            ValueRange indices,
                                            OpBuilder &builder, Location loc);

namespace detail {

// Integer types, constants, and source-width arithmetic.
enum class BinaryKind { Add, Subtract, Multiply };

/// Accept signless integers up to 64 bits, or index (64 bits at this stage).
/// Ranked tensors use their element type.
FailureOr<unsigned> getIntegerWidth(Type type);
/// Unwrap ranked tensors; leave all other types unchanged.
Type getElementTypeOrSelf(Type type);
FailureOr<OpFoldResult> typedZero(Type scalarType, OpBuilder &builder);
FailureOr<llvm::APInt> getIntegerConstant(OpFoldResult value, unsigned width);
/// Bounded divisibility proof from local integer expressions and hints.
bool isKnownMultiple(OpFoldResult operand, uint64_t divisor,
                     unsigned depth = 8);
/// Local sign proof for address decompositions. Add/mul follow the analysis's
/// no-overflow assumption; narrowing casts do not preserve the proof.
bool isKnownNonNegative(OpFoldResult operand, unsigned depth = 8);
FailureOr<OpFoldResult> foldOrCreateBinary(BinaryKind kind, OpFoldResult lhs,
                                           OpFoldResult rhs, Type resultType,
                                           OpBuilder &builder, Location loc);
FailureOr<OpFoldResult> foldOrCreateCast(OpFoldResult source, Type targetType,
                                         IntegerExtensionKind kind,
                                         OpBuilder &builder, Location loc);
FailureOr<Value> materialize(OpFoldResult value, Type expectedType,
                             OpBuilder &builder, Location loc);
FailureOr<Value> splatTo(Value scalar, RankedTensorType target,
                         OpBuilder &builder, Location loc);

// Address decompositions and axis proofs.
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
/// Materialize fullOffsets; sign-extend without narrowing and splat scalars to
/// tensor targets at the caller's insertion point when necessary.
FailureOr<Value> materializeFullOffset(const OffsetInfo &info, Type target,
                                       OpBuilder &builder, Location loc);

// Axis alignment and stride queries.
/// Remove unit digits and coalesce adjacent zero-stride digits within one
/// original dimension. Keep a unit digit when the entire dimension is unit.
void normalizeSubAxes(SmallVectorImpl<SubAxisInfo> &subAxes,
                      OpBuilder &builder);
/// Express compatible axes in the same sub-axis partition, if provable.
LogicalResult alignAxes(AxisInfo &lhs, AxisInfo &rhs,
                        ConcreteArithmetic &arithmetic);
SmallVector<OpFoldResult> axisStrides(const OffsetInfo &info);
/// Infer a single stride from a valid sub-axis partition without emitting IR.
/// Return null when the coefficients do not prove a linear axis.
OpFoldResult inferAxisStride(ArrayRef<SubAxisInfo> subAxes, OpBuilder &builder);

// Divisibility proofs and exact unit conversion.
bool canDivideDecomposition(const OffsetInfo &info, uint64_t divisor);
/// Fold or materialize exact division of an index descriptor field.
/// The caller has proved divisibility by the positive divisor.
OpFoldResult divideIndex(OpFoldResult value, uint64_t divisor,
                         OpBuilder &builder, Location loc);
/// After the full value is proved divisible, convert descriptor fields whose
/// exact division can also be proved. Leave fullOffsets to the caller.
void divideDecomposition(OffsetInfo &info, uint64_t divisor, OpBuilder &builder,
                         Location loc);

// Range proofs for an existing decomposition.
bool decompositionFitsSigned(const OffsetInfo &info, unsigned width);
/// Bounds of the mathematical decomposition, not the wrapped fullOffsets.
std::optional<llvm::ConstantRange>
constantDecompositionRange(const OffsetInfo &info, unsigned width);

} // namespace detail
} // namespace mlir::triton::ptr

#endif // TRITON_ASCEND_MEMORY_ACCESS_UTILS_H
