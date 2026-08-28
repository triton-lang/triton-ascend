/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "TritonControlFlowOpt/TensorPtrDecompose.h"

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::controlflow;

namespace {

static constexpr unsigned kOffsetsComponent = 0;
static constexpr unsigned kBaseInvariant = 0;
static constexpr unsigned kBaseIsScalarAttribute = 0;

/// Identifies tensor-of-pointers handled by this stage:
/// `tensor<...x!tt.ptr<T>>`. A block pointer has the different scalar type
/// `!tt.ptr<tensor<...xT>>` and is handled by BlockPtrDecompose.
static bool isTensorPointerType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

/// Tensor-pointer state used only by this policy:
///
///   control-flow components = [complete_offsets]
///   rewrite-only invariants = [common_base]
///   attributes = [base_is_scalar]
///
/// Only `components` expand an SCF signature. The common base is deliberately
/// kept out of iter-args and results; it must be identical at every incoming
/// edge and is used to rebuild the tensor-of-pointers inside and after the
/// rewritten control-flow operation.
static RankedTensorType getDefaultOffsetsType(Type pointerType) {
  auto pointerTensor = cast<RankedTensorType>(pointerType);
  return RankedTensorType::get(pointerTensor.getShape(),
                               IntegerType::get(pointerType.getContext(), 32),
                               pointerTensor.getEncoding());
}

/// Validates the policy-owned component layout, including the offsets shape
/// and the representation selected for the invariant base.
static bool hasValidSchema(Type originalType, Type offsetsType,
                           ArrayRef<Value> invariants,
                           ArrayRef<Attribute> attributes) {
  auto pointerTensor = dyn_cast<RankedTensorType>(originalType);
  auto offsetsTensor = dyn_cast<RankedTensorType>(offsetsType);
  if (!pointerTensor || !offsetsTensor || invariants.size() != 1 ||
      attributes.size() != 1 ||
      !isa<triton::PointerType>(pointerTensor.getElementType()) ||
      !isa<IntegerType>(offsetsTensor.getElementType()) ||
      pointerTensor.getShape() != offsetsTensor.getShape() ||
      pointerTensor.getEncoding() != offsetsTensor.getEncoding())
    return false;

  auto baseIsScalar = dyn_cast<BoolAttr>(attributes[kBaseIsScalarAttribute]);
  if (!baseIsScalar)
    return false;
  Type expectedBaseType =
      baseIsScalar.getValue() ? pointerTensor.getElementType() : originalType;
  return invariants[kBaseInvariant].getType() == expectedBaseType;
}

/// Validates the concrete Value-based state created while rewriting IR.
static bool hasValidLayout(const DecomposedValue &value) {
  return value.components.size() == 1 &&
         hasValidSchema(value.originalType,
                        value.components[kOffsetsComponent].getType(),
                        value.invariants, value.attributes);
}

/// Validates the read-only counterpart without materializing component Values.
static bool hasValidLayout(const AnalyzedValue &value) {
  return value.components.size() == 1 &&
         hasValidSchema(value.originalType,
                        value.components[kOffsetsComponent].type,
                        value.invariants, value.attributes);
}

template <typename StateT>
static bool haveSameTensorPtrBaseSchema(const StateT &lhs, const StateT &rhs) {
  return hasValidLayout(lhs) && hasValidLayout(rhs) &&
         lhs.originalType == rhs.originalType &&
         lhs.invariants == rhs.invariants && lhs.attributes == rhs.attributes;
}

/// Whether the invariant base must be broadcast before rebuilding addptr.
static bool hasScalarBase(const DecomposedValue &value) {
  return cast<BoolAttr>(value.attributes[kBaseIsScalarAttribute]).getValue();
}

/// Creates the initial complete offsets for an opaque tensor-of-pointers.
/// i32 is the current default. A later addptr or control-flow merge may promote
/// it to i64 through getWiderOffsetsType.
static Value createZeroOffsets(OpBuilder &builder, Location loc,
                               Type pointerType) {
  RankedTensorType offsetsType = getDefaultOffsetsType(pointerType);
  auto elementType = cast<IntegerType>(offsetsType.getElementType());
  auto zero = DenseElementsAttr::get(offsetsType,
                                     builder.getIntegerAttr(elementType, 0));
  return builder.create<arith::ConstantOp>(loc, zero);
}

/// Computes the common type used by an offset addition or SCF merge.
///
/// Complete offsets are always integer tensors with the pointer tensor's shape
/// and encoding. When element widths differ, use the wider signed width.
static FailureOr<Type> getWiderOffsetsType(Type lhs, Type rhs) {
  auto lhsTensor = dyn_cast<RankedTensorType>(lhs);
  auto rhsTensor = dyn_cast<RankedTensorType>(rhs);
  if (!lhsTensor || !rhsTensor ||
      lhsTensor.getShape() != rhsTensor.getShape() ||
      lhsTensor.getEncoding() != rhsTensor.getEncoding())
    return failure();
  auto lhsElementInt = dyn_cast<IntegerType>(lhsTensor.getElementType());
  auto rhsElementInt = dyn_cast<IntegerType>(rhsTensor.getElementType());
  if (!lhsElementInt || !rhsElementInt)
    return failure();
  if (lhsElementInt.getWidth() == rhsElementInt.getWidth()) {
    if (lhsElementInt != rhsElementInt)
      return failure();
    return lhs;
  }
  return lhsElementInt.getWidth() >= rhsElementInt.getWidth() ? lhs : rhs;
}

/// Adds two offset values after promoting both operands to their common type.
/// This is the concrete IR counterpart of joinComponentTypes used in the
/// read-only analysis.
static Value createOffsetsAdd(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs) {
  if (!lhs || !rhs)
    return nullptr;
  FailureOr<Type> type = getWiderOffsetsType(lhs.getType(), rhs.getType());
  if (failed(type))
    return nullptr;
  FailureOr<Value> convertedLhs = castIntegerLike(builder, loc, lhs, *type);
  FailureOr<Value> convertedRhs = castIntegerLike(builder, loc, rhs, *type);
  if (failed(convertedLhs) || failed(convertedRhs))
    return nullptr;
  return builder.create<arith::AddIOp>(loc, *convertedLhs, *convertedRhs);
}

/// Tensor-pointer semantics plugged into the shared control-flow machinery.
///
/// The analysis methods compute a symbolic schema without changing IR. The
/// rewrite methods later materialize the exact Values for that already-chosen
/// schema. Keeping both implementations here makes their layouts directly
/// comparable and avoids hiding pointer semantics in ControlFlowRewrite.
class TensorPtrDecomposePolicy final : public ControlFlowRewritePolicy {
public:
  /// Selects only tensor-of-pointers owned by this decomposition stage.
  bool matches(Type type) const override {
    // A tensor pointer here means tensor<...x!tt.ptr<...>>. Scalar block
    // pointers have already been handled by BlockPtrDecompose.
    return isTensorPointerType(type);
  }

  //===--------------------------------------------------------------------===//
  // Read-only schema analysis
  //===--------------------------------------------------------------------===//

  /// Recovers `{common_base, complete_offsets, base_is_scalar}` without
  /// creating constants, additions, or pointer operations.
  FailureOr<AnalyzedValue>
  analyzeValue(Value value,
               ControlFlowAnalysisContext &context) const override {
    // Region arguments and previously merged control-flow results are already
    // bound in the stage-scoped cache. Reusing them is what lets analysis cross
    // sibling and nested SCF without inspecting rewritten IR.
    if (const AnalyzedValue *known = context.lookupValue(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    // addptr preserves the upstream common base and adds another contribution
    // to complete_offsets. The result Value is used only as a symbolic identity
    // showing that this component changed.
    if (auto addPtr = value.getDefiningOp<triton::AddPtrOp>()) {
      FailureOr<AnalyzedValue> result = context.analyzeValue(addPtr.getPtr());
      if (failed(result) || !hasValidLayout(*result))
        return failure();
      FailureOr<Type> offsetsType =
          getWiderOffsetsType(result->components[kOffsetsComponent].type,
                              addPtr.getOffset().getType());
      if (failed(offsetsType))
        return failure();
      result->originalType = value.getType();
      result->components[kOffsetsComponent] = {
          *offsetsType, ComponentIdentity::fromValue(value, kOffsetsComponent)};
      return *result;
    }

    // Splatting one scalar pointer establishes the canonical common-base form:
    // every lane starts at the scalar base and therefore has zero offset.
    if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
      if (!isa<triton::PointerType>(splat.getSrc().getType()))
        return failure();
      Type offsetsType = getDefaultOffsetsType(value.getType());
      return AnalyzedValue{value.getType(),
                           {{offsetsType, ComponentIdentity::zero()}},
                           {splat.getSrc()},
                           {BoolAttr::get(value.getContext(), true)}};
    }

    // An otherwise opaque tensor-of-pointers is treated as an already-vector
    // base with zero additional offsets. This preserves current behavior until
    // common-base analysis is shared with TritonToUnstructure.
    if (!matches(value.getType()))
      return failure();
    Type offsetsType = getDefaultOffsetsType(value.getType());
    return AnalyzedValue{value.getType(),
                         {{offsetsType, ComponentIdentity::zero()}},
                         {value},
                         {BoolAttr::get(value.getContext(), false)}};
  }

  /// Tensor pointers have exactly one loop-transfer candidate: the complete
  /// per-lane offsets tensor at component index 0.
  FailureOr<SmallVector<unsigned>>
  getLoopCandidateComponents(const AnalyzedValue &value) const override {
    if (!hasValidLayout(value))
      return failure();
    return SmallVector<unsigned>{kOffsetsComponent};
  }

  /// Classifies the loop offsets as transferred only when the backedge changes
  /// their symbolic identity. The original pointer type, common base, and base
  /// representation must remain invariant for reconstruction to be valid.
  FailureOr<SmallVector<unsigned>>
  getLoopTransferredComponents(const AnalyzedValue &initial,
                               const AnalyzedValue &regionArgument,
                               const AnalyzedValue &next) const override {
    if (!haveSameTensorPtrBaseSchema(initial, regionArgument) ||
        !haveSameTensorPtrBaseSchema(initial, next) ||
        failed(joinComponentTypes(initial.components[kOffsetsComponent].type,
                                  next.components[kOffsetsComponent].type)))
      return failure();
    // Yielding the region argument unchanged requires no new SCF iter-arg.
    if (regionArgument.components[kOffsetsComponent].identity ==
        next.components[kOffsetsComponent].identity)
      return SmallVector<unsigned>{};
    return SmallVector<unsigned>{kOffsetsComponent};
  }

  /// Merges the two `scf.if` pointer states. Different complete-offset
  /// identities become an if result; the base and its scalar/tensor form must
  /// agree so one pointer can be rebuilt after the if.
  FailureOr<SmallVector<unsigned>>
  getIfTransferredComponents(const AnalyzedValue &thenValue,
                             const AnalyzedValue &elseValue) const override {
    if (!haveSameTensorPtrBaseSchema(thenValue, elseValue) ||
        failed(
            joinComponentTypes(thenValue.components[kOffsetsComponent].type,
                               elseValue.components[kOffsetsComponent].type)))
      return failure();
    // Identical symbolic offsets are available outside the if as an invariant.
    if (thenValue.components[kOffsetsComponent].identity ==
        elseValue.components[kOffsetsComponent].identity)
      return SmallVector<unsigned>{};
    return SmallVector<unsigned>{kOffsetsComponent};
  }

  /// Chooses the offsets type carried by the replacement control-flow op.
  FailureOr<Type> joinComponentTypes(Type lhs, Type rhs) const override {
    return getWiderOffsetsType(lhs, rhs);
  }

  //===--------------------------------------------------------------------===//
  // Concrete rewrite materialization
  //===--------------------------------------------------------------------===//

  /// addptr results must be decomposed immediately after cloning so later users
  /// can obtain their accumulated offsets from the rewrite context.
  bool shouldDecomposeOperation(Operation *op) const override {
    return isa<triton::AddPtrOp>(op);
  }

  /// Materializes the same decomposition described by analyzeValue. This may
  /// create zero constants and integer additions, so it is called only after
  /// the complete control-flow subtree has passed read-only analysis.
  FailureOr<DecomposedValue> decompose(Value value,
                                       const ControlFlowRewriteContext &context,
                                       OpBuilder &builder,
                                       Location loc) const override {
    // Rewritten region arguments and nested results have exact component Values
    // recorded by ControlFlowRewrite; never reconstruct them from old IR.
    if (const DecomposedValue *known = context.lookup(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    // Ordinary SSA inputs may already have been cloned into the replacement
    // region, so pointer producer matching must use the remapped Value.
    value = context.remap(value);
    // Recursively flatten an addptr chain into one complete offsets tensor.
    if (auto addPtr = value.getDefiningOp<triton::AddPtrOp>()) {
      FailureOr<DecomposedValue> result =
          decompose(addPtr.getPtr(), context, builder, loc);
      if (failed(result) || !hasValidLayout(*result))
        return failure();

      // Insert the accumulated addition beside the addptr being decomposed;
      // ControlFlowRewrite will use this Value in the expanded SCF signature.
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(addPtr);
      Value offsets = createOffsetsAdd(builder, addPtr.getLoc(),
                                       result->components[kOffsetsComponent],
                                       context.remap(addPtr.getOffset()));
      if (!offsets)
        return failure();
      result->originalType = value.getType();
      result->components[kOffsetsComponent] = offsets;
      return *result;
    }

    // A scalar pointer splat materializes zero offsets while retaining the
    // scalar source as the common-base invariant.
    if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
      if (!isa<triton::PointerType>(splat.getSrc().getType()))
        return failure();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(splat);
      Value offsets =
          createZeroOffsets(builder, splat.getLoc(), value.getType());
      if (!offsets)
        return failure();
      return DecomposedValue{value.getType(),
                             {offsets},
                             {splat.getSrc()},
                             {builder.getBoolAttr(true)}};
    }

    // Fallback for an opaque tensor base: use the entire tensor-of-pointers as
    // the invariant base and represent only subsequent displacement in offsets.
    if (!matches(value.getType()))
      return failure();

    OpBuilder::InsertionGuard guard(builder);
    if (Operation *definingOp = value.getDefiningOp())
      builder.setInsertionPointAfter(definingOp);
    else if (auto blockArg = dyn_cast<BlockArgument>(value))
      builder.setInsertionPointToStart(blockArg.getOwner());
    // Place the zero where it dominates every later reconstruction using it.
    Value offsets = createZeroOffsets(builder, loc, value.getType());
    if (!offsets)
      return failure();
    return DecomposedValue{
        value.getType(), {offsets}, {value}, {builder.getBoolAttr(false)}};
  }

  /// Rebuilds the original tensor-of-pointers from the invariant base and the
  /// complete offsets selected/carried by the rewritten control flow.
  Value recompose(const DecomposedValue &value, OpBuilder &builder,
                  Location loc) const override {
    if (!hasValidLayout(value))
      return nullptr;
    Value base = value.invariants[kBaseInvariant];
    // addptr requires matching tensor lanes; broadcast a scalar common base
    // only when the decomposition recorded `base_is_scalar = true`.
    if (hasScalarBase(value))
      base = builder.create<triton::SplatOp>(loc, value.originalType, base);
    return builder.create<triton::AddPtrOp>(
        loc, value.originalType, base, value.components[kOffsetsComponent]);
  }
};

} // namespace

namespace mlir::triton::controlflow {

/// Runs tensor-pointer decomposition after CFG structuring/block-pointer
/// handling. The shared driver analyzes each outermost SCF root, then rewrites
/// only the complete-offset components selected by this policy.
LogicalResult runTensorPtrDecompose(ModuleOp module) {
  // Carry only complete per-lane offsets through SCF. The common scalar base
  // remains a rewrite invariant and is used to rebuild tensor-of-pointers at
  // each region boundary. This decomposition is independent of
  // BlockPtrDecompose.
  // TODO: Replace this local extraction with TritonToUnstructure's common-base
  // analysis. Different or lane-wise bases must become explicit diagnostics
  // instead of pattern misses.
  TensorPtrDecomposePolicy policy;
  return rewriteControlFlow(module, policy);
}

} // namespace mlir::triton::controlflow
