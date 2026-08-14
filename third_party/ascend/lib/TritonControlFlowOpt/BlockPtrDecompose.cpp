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

#include "TritonControlFlowOpt/BlockPtrDecompose.h"

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::controlflow;

namespace {

static constexpr unsigned kBaseComponent = 0;
static constexpr unsigned getShapeStart() { return kBaseComponent + 1; }
static constexpr unsigned getStrideStart(unsigned rank) {
  return getShapeStart() + rank;
}
static constexpr unsigned getOffsetStart(unsigned rank) {
  return getStrideStart(rank) + rank;
}

/// Block-pointer component layout used only by this policy:
///
///   components = [base_address, shape..., strides..., offsets...]
///   attributes = [order]
///
/// Every supported SCF boundary carries all components in this exact order.
/// The base is represented as an i64 address rather than a pointer/memref so a
/// control-flow merge remains an SSA value selection and cannot be lowered as
/// a memory-object copy by the backend.
/// `order` remains policy-owned static metadata and must agree on every
/// incoming path.
///
/// Keep rank/layout checks local to this file: the generic control-flow
/// machinery intentionally does not know the descriptor format of a policy.
// Extracts the block-pointer rank from the original type shared by analysis
// and rewrite states. Invalid pointer and pointee types fail consistently.
static FailureOr<unsigned> getRank(Type originalType) {
  auto pointerType = dyn_cast<triton::PointerType>(originalType);
  if (!pointerType)
    return failure();
  auto tensorType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!tensorType)
    return failure();
  return tensorType.getRank();
}

// Recovers the scalar pointer type accepted by tt.make_tensor_ptr from the
// BlockPtr result type. The address space is preserved across the temporary
// integer carrier.
static FailureOr<triton::PointerType> getBasePointerType(Type originalType) {
  auto pointerType = dyn_cast<triton::PointerType>(originalType);
  if (!pointerType)
    return failure();
  auto tensorType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!tensorType)
    return failure();
  return triton::PointerType::get(tensorType.getElementType(),
                                  pointerType.getAddressSpace());
}

// Validates the common block-pointer schema for either state representation.
// Their component element types differ, but this check only needs field sizes.
template <typename StateT> static bool hasValidLayout(const StateT &state) {
  FailureOr<unsigned> rank = getRank(state.originalType);
  return succeeded(rank) && state.components.size() == 1 + 3 * *rank &&
         state.attributes.size() == 1 &&
         isa<DenseI32ArrayAttr>(state.attributes.front());
}

/// Returns the complete, ordered descriptor range used to expand one
/// block-pointer control-flow slot. Keeping this policy-local prevents the
/// generic SCF rewrite from depending on the BlockPtr field layout.
static SmallVector<unsigned>
getAllComponentIndices(const AnalyzedValue &value) {
  SmallVector<unsigned> indices;
  indices.reserve(value.components.size());
  for (unsigned index = 0; index < value.components.size(); ++index)
    indices.push_back(index);
  return indices;
}

class BlockPtrPolicy final : public ControlFlowRewritePolicy {
public:
  bool matches(Type type) const override {
    // A Triton block pointer is a scalar !tt.ptr whose pointee is a ranked
    // tensor. Tensor-of-pointers are handled by their own decomposition.
    auto pointerType = dyn_cast<triton::PointerType>(type);
    return pointerType && isa<RankedTensorType>(pointerType.getPointeeType());
  }

  FailureOr<AnalyzedValue>
  analyzeValue(Value value,
               ControlFlowAnalysisContext &context) const override {
    // Region arguments and control-flow results are installed by the generic
    // analysis after merging their incoming abstract component states.
    if (const AnalyzedValue *known = context.lookupValue(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    if (auto makePtr = value.getDefiningOp<triton::MakeTensorPtrOp>()) {
      // make_tensor_ptr exposes the complete descriptor directly. Record only
      // types and symbolic identities here; this phase must not create IR.
      AnalyzedValue result;
      result.originalType = value.getType();
      Value base = makePtr.getBase();
      result.components.push_back(
          {IntegerType::get(value.getContext(), 64),
           ComponentIdentity::fromValue(base, kBaseComponent)});
      unsigned componentIndex = getShapeStart();
      auto appendComponents = [&](ValueRange values) {
        for (Value component : values) {
          result.components.push_back(
              {component.getType(),
               ComponentIdentity::fromValue(component, componentIndex++)});
        }
      };
      appendComponents(makePtr.getShape());
      appendComponents(makePtr.getStrides());
      appendComponents(makePtr.getOffsets());
      result.attributes.push_back(makePtr.getOrderAttr());
      if (!hasValidLayout(result))
        return failure();
      return result;
    }

    auto advance = value.getDefiningOp<triton::AdvanceOp>();
    if (!advance)
      return failure();
    // tt.advance preserves base/shape/strides/order and produces new offsets.
    // Give those offsets identities tied to the result so a loop/if merge can
    // detect that they differ without materializing arith.addi operations.
    FailureOr<AnalyzedValue> result = context.analyzeValue(advance.getPtr());
    if (failed(result) || !hasValidLayout(*result))
      return failure();
    unsigned rank = *getRank(result->originalType);
    if (advance.getOffsets().size() != rank)
      return failure();
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      unsigned componentIndex = getOffsetStart(rank) + dimension;
      result->components[componentIndex].identity =
          ComponentIdentity::fromValue(value, componentIndex);
    }
    result->originalType = value.getType();
    return *result;
  }

  FailureOr<SmallVector<unsigned>>
  getLoopCandidateComponents(const AnalyzedValue &value) const override {
    if (!hasValidLayout(value))
      return failure();
    // A BlockPtr never crosses a supported loop boundary directly. Its entire
    // descriptor becomes ordinary loop-carried SSA state.
    return getAllComponentIndices(value);
  }

  FailureOr<SmallVector<unsigned>>
  getLoopTransferredComponents(const AnalyzedValue &initial,
                               const AnalyzedValue &regionArgument,
                               const AnalyzedValue &next) const override {
    if (!hasValidLayout(initial) || !hasValidLayout(regionArgument) ||
        !hasValidLayout(next) ||
        initial.originalType != regionArgument.originalType ||
        initial.originalType != next.originalType ||
        initial.attributes != regionArgument.attributes ||
        initial.attributes != next.attributes)
      return failure();

    // Full descriptor transfer permits every field to change, but each
    // position must retain the strict type required by tt.make_tensor_ptr.
    // In particular, tt.advance preserving its base does not make the base a
    // loop invariant: a nested if may rebuild the descriptor from a different
    // root base, and that if result becomes the next backedge value. A generic
    // SCF canonicalization may remove exactly forwarded components from an
    // advance-only loop later; this policy must retain changing-base semantics.
    for (unsigned index = 0; index < initial.components.size(); ++index) {
      if (failed(joinComponentTypes(initial.components[index].type,
                                    regionArgument.components[index].type)) ||
          failed(joinComponentTypes(initial.components[index].type,
                                    next.components[index].type)))
        return failure();
    }
    return getAllComponentIndices(initial);
  }

  FailureOr<SmallVector<unsigned>>
  getIfTransferredComponents(const AnalyzedValue &thenValue,
                             const AnalyzedValue &elseValue) const override {
    if (!hasValidLayout(thenValue) || !hasValidLayout(elseValue) ||
        thenValue.originalType != elseValue.originalType ||
        thenValue.attributes != elseValue.attributes)
      return failure();

    // Both branches yield a complete descriptor even when a field has the same
    // symbolic identity. This gives every rewritten BlockPtr boundary one
    // stable positional schema.
    for (unsigned index = 0; index < thenValue.components.size(); ++index) {
      if (failed(joinComponentTypes(thenValue.components[index].type,
                                    elseValue.components[index].type)))
        return failure();
    }
    return getAllComponentIndices(thenValue);
  }

  FailureOr<Type> joinComponentTypes(Type lhs, Type rhs) const override {
    // Block-pointer descriptor operands must have identical types on all
    // incoming paths. Tensor-pointer offsets use a more permissive integer
    // width join in their own policy.
    if (lhs != rhs)
      return failure();
    return lhs;
  }

  bool shouldDecomposeOperation(Operation *op) const override {
    // Recording each cloned advance lets downstream advances reuse its
    // flattened descriptor rather than walking through the rebuilt pointer.
    return isa<triton::AdvanceOp>(op);
  }

  bool requiresPointerDescriptorBoundaryMarker() const override { return true; }

  FailureOr<DecomposedValue> decompose(Value value,
                                       const ControlFlowRewriteContext &context,
                                       OpBuilder &builder,
                                       Location loc) const override {
    // Prefer decompositions recorded while rebuilding the enclosing region;
    // this is how analysis results cross nested SCF boundaries at rewrite time.
    if (const DecomposedValue *known = context.lookup(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    value = context.remap(value);
    if (auto makePtr = value.getDefiningOp<triton::MakeTensorPtrOp>()) {
      // Materialize the concrete counterpart of analyzeValue's descriptor.
      DecomposedValue result;
      result.originalType = value.getType();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(makePtr);
      Value baseAddress = builder.create<triton::PtrToIntOp>(
          makePtr.getLoc(), builder.getI64Type(), makePtr.getBase());
      result.components.push_back(baseAddress);
      result.components.append(makePtr.getShape().begin(),
                               makePtr.getShape().end());
      result.components.append(makePtr.getStrides().begin(),
                               makePtr.getStrides().end());
      result.components.append(makePtr.getOffsets().begin(),
                               makePtr.getOffsets().end());
      result.attributes.push_back(makePtr.getOrderAttr());
      if (!hasValidLayout(result))
        return failure();
      return result;
    }

    auto advance = value.getDefiningOp<triton::AdvanceOp>();
    if (!advance)
      return failure();

    FailureOr<DecomposedValue> result =
        decompose(advance.getPtr(), context, builder, loc);
    if (failed(result) || !hasValidLayout(*result))
      return failure();
    FailureOr<unsigned> rank = getRank(result->originalType);
    if (advance.getOffsets().size() != *rank)
      return failure();

    // Flatten an advance into offset arithmetic at the original operation's
    // position. Base, shape, strides, and order are inherited unchanged.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(advance);
    for (auto [dim, delta] : llvm::enumerate(advance.getOffsets())) {
      unsigned component = getOffsetStart(*rank) + dim;
      Value currentOffset = result->components[component];
      Value remappedDelta = context.remap(delta);
      if (!remappedDelta)
        return failure();
      Value offset = dyn_cast<Value>(
          addOpFoldResult(currentOffset, remappedDelta, advance.getLoc(),
                          builder, currentOffset.getType()));
      if (!offset)
        return failure();
      result->components[component] = offset;
    }
    result->originalType = value.getType();
    return *result;
  }

  Value recompose(const DecomposedValue &value, OpBuilder &builder,
                  Location loc) const override {
    // Rebuild the original pointer type immediately inside/after the rewritten
    // control-flow boundary so ordinary users remain untouched.
    if (!hasValidLayout(value))
      return nullptr;
    unsigned rank = *getRank(value.originalType);
    auto order = cast<DenseI32ArrayAttr>(value.attributes.front());
    FailureOr<triton::PointerType> basePointerType =
        getBasePointerType(value.originalType);
    if (failed(basePointerType) ||
        value.components[kBaseComponent].getType() != builder.getI64Type())
      return nullptr;
    Value base = builder.create<triton::IntToPtrOp>(
        loc, *basePointerType, value.components[kBaseComponent]);
    return builder.create<triton::MakeTensorPtrOp>(
        loc, value.originalType, base,
        ValueRange(value.components).slice(getShapeStart(), rank),
        ValueRange(value.components).slice(getStrideStart(rank), rank),
        ValueRange(value.components).slice(getOffsetStart(rank), rank), order);
  }
};

} // namespace

namespace mlir::triton::controlflow {

LogicalResult runBlockPtrDecompose(ModuleOp module) {
  // Expand every BlockPtr that crosses a supported SCF boundary into its full
  // ordered descriptor, then rebuild the pointer at each use site.
  BlockPtrPolicy policy;
  return rewriteControlFlow(module, policy);
}

} // namespace mlir::triton::controlflow
