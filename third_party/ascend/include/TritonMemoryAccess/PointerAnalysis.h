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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_H
#define TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ConstantRange.h"

#include <memory>
#include <optional>

namespace mlir {
class RewriterBase;
}

namespace mlir::triton {
class AdvanceOp;
}

namespace mlir::triton::pointer {

enum class AxisKind { Invariant, Structured, Unknown };
enum class ArithmeticDomain { SourceInteger, ElementAddress };

/// An exact offset is represented by completeOffset or a proven affine form
/// c + sum(lane[d] * strides[d]). Integer and ordinary-pointer queries always
/// supply completeOffset. Block pointers initially supply only the affine form;
/// buildAddressView materializes their complete tensor on demand. If any axis
/// is unknown, completeOffset is required and c/unknown strides are typed
/// zeros.
struct OffsetComponents {
  Type valueType;
  ArithmeticDomain domain = ArithmeticDomain::SourceInteger;
  SmallVector<int64_t> shape;
  OpFoldResult completeOffset;
  OpFoldResult uniformOffset;
  SmallVector<OpFoldResult> strides;
  SmallVector<AxisKind> axes;

  /// These queries describe facts in `domain`, not downstream legality.
  bool hasAffineForm() const;
  bool isUniform() const;
  std::optional<OpFoldResult> getKnownStride(unsigned axis) const;
  std::optional<OpFoldResult> getAffineOrigin() const;
};

/// Preserve an integer/index SSA value exactly without claiming tensor axes.
/// ElementAddress requires an already converted i64 value; this does not cast.
/// A rank-zero tensor's scalar origin may require a pure tensor.extract, placed
/// after the value. The builder's insertion point is preserved.
FailureOr<OffsetComponents>
makeOpaqueOffset(Value value, OpBuilder &builder,
                 ArithmeticDomain domain = ArithmeticDomain::SourceInteger);

/// Additional source-descriptor facts needed by block boundary checks and SCF
/// reconstruction. Coordinates are effective i64 values; make_tensor_ptr and
/// advance source operands retain their original i32 arithmetic.
struct BlockPointerInfo {
  SmallVector<OpFoldResult> parentShape;
  SmallVector<OpFoldResult> coordinateOffsets;
  SmallVector<llvm::ConstantRange> coordinateRanges;
  SmallVector<int32_t> order;
};

/// One address representation for scalar, tensor-of-pointer and block-pointer
/// values. Shape, strides and origin live in offsets for all three forms.
/// Block metadata does not duplicate the base, layout or element type.
struct PointerComponents {
  Value base;
  OffsetComponents offsets;
  Type elementType;
  unsigned addressSpace = 0;
  std::optional<BlockPointerInfo> block;
};

struct BlockAdvanceDelta {
  SmallVector<OpFoldResult> coordinateDeltas;
  SmallVector<llvm::ConstantRange> ranges;
};

struct BlockCoordinates {
  SmallVector<OpFoldResult> values;
  SmallVector<llvm::ConstantRange> ranges;
};

/// Combine supplied fields without revisiting pointer producers. SCF adapters
/// can use newly built coordinates while retaining their own rewrite policy.
FailureOr<BlockCoordinates> advanceBlockOffsets(
    ArrayRef<OpFoldResult> current, ArrayRef<llvm::ConstantRange> currentRanges,
    const BlockAdvanceDelta &delta, OpBuilder &builder, Location loc);

/// Return the complete element offset, materializing the common affine form if
/// needed. Tensor materialization requires a static, positive, unencoded shape
/// with a power-of-two element count within Triton's limit. Analysis itself
/// also accepts block descriptors with shapes outside those restrictions.
FailureOr<OffsetComponents> buildAddressView(const PointerComponents &pointer,
                                             OpBuilder &builder, Location loc);

struct AnalysisOptions {
  // PR1 address queries and bindings require 64; SourceInteger is independent.
  unsigned addressBitWidth = 64;
  std::optional<unsigned> indexBitWidth;
};

/// A constructive analysis session for one stable source IR and boundary set.
/// It may insert only pure numeric/shape operations. Callers must load Arith,
/// Triton, and Tensor (rank-zero extraction) dialects. Call clear() before an
/// unrelated rewrite or CSE, or remapAndForgetScope() at an equivalent clone.
class PointerAnalysis {
public:
  PointerAnalysis(OpBuilder &builder, AnalysisOptions options = {});
  ~PointerAnalysis();
  PointerAnalysis(PointerAnalysis &&) noexcept;
  PointerAnalysis &operator=(PointerAnalysis &&) noexcept;

  FailureOr<OffsetComponents> analyzeOffset(Value value);
  FailureOr<OffsetComponents> analyzeAddressDelta(Value value);
  FailureOr<PointerComponents> analyzePointer(Value pointer);
  /// Query advance operands even when its input pointer is an unbound boundary.
  FailureOr<BlockAdvanceDelta> analyzeBlockAdvanceDelta(triton::AdvanceOp op);
  /// Borrowed cache entries: do not retain these pointers across bind*, clear,
  /// successful remap, move assignment or destruction. Copies also contain SSA
  /// references whose validity depends on the caller keeping the IR stable.
  const OffsetComponents *findCachedOffset(Value value) const;
  const OffsetComponents *findCachedAddressDelta(Value value) const;
  const PointerComponents *findCachedPointer(Value value) const;
  const BlockAdvanceDelta *
  findCachedBlockAdvanceDelta(triton::AdvanceOp op) const;
  /// Bindings are caller-proven equalities. Type/shape/domain checks do not
  /// prove numerical equivalence, including consistency between block
  /// coordinates and the affine origin. Replacing a binding or an analyzed
  /// boundary invalidates derived entries; an unseen boundary preserves them.
  LogicalResult bindOffset(Value boundary, const OffsetComponents &components);
  LogicalResult bindPointer(Value boundary,
                            const PointerComponents &components);
  LogicalResult remapAndForgetScope(Operation *oldScope,
                                    const IRMapping &resultMapping);
  void clear();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// Erase only trivially-dead numeric/shape producers, never source memory or
/// control-flow operations. Run after the session has stopped using its cache.
void eraseDeadPointerArithmetic(Operation *scope, RewriterBase &rewriter);

} // namespace mlir::triton::pointer

#endif // TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_H
