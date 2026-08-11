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

/// Block-pointer component layout used only by this policy:
///
///   components = [shape..., strides..., offsets...]
///   invariants = [base]
///   attributes = [order]
///
/// Loops currently carry only `offsets`; shape and strides must remain
/// invariant across a backedge. An scf.if may select any component whose SSA
/// value differs between its branches.
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

// Validates the common block-pointer schema for either state representation.
// Their component element types differ, but this check only needs field sizes.
template <typename StateT> static bool hasValidLayout(const StateT &state) {
  FailureOr<unsigned> rank = getRank(state.originalType);
  return succeeded(rank) && state.components.size() == 3 * *rank &&
         state.invariants.size() == 1 && state.attributes.size() == 1 &&
         isa<DenseI32ArrayAttr>(state.attributes.front());
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
      unsigned componentIndex = 0;
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
      result.invariants.push_back(makePtr.getBase());
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
      unsigned componentIndex = 2 * rank + dimension;
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
    unsigned rank = *getRank(value.originalType);
    SmallVector<unsigned> indices;
    // Only the final rank entries (offsets) are legal loop-carried state in the
    // current block-pointer model.
    for (unsigned dimension = 0; dimension < rank; ++dimension)
      indices.push_back(2 * rank + dimension);
    return indices;
  }

  FailureOr<SmallVector<unsigned>>
  getLoopTransferredComponents(const AnalyzedValue &initial,
                               const AnalyzedValue &regionArgument,
                               const AnalyzedValue &next) const override {
    if (!hasValidLayout(initial) || !hasValidLayout(regionArgument) ||
        !hasValidLayout(next) ||
        initial.originalType != regionArgument.originalType ||
        initial.originalType != next.originalType ||
        initial.invariants != regionArgument.invariants ||
        initial.invariants != next.invariants ||
        initial.attributes != regionArgument.attributes ||
        initial.attributes != next.attributes)
      return failure();

    unsigned rank = *getRank(initial.originalType);
    // The current implementation does not expand shape or stride iter_args.
    // Reject a loop that changes either instead of silently reconstructing a
    // descriptor with stale values.
    for (unsigned index = 0; index < 2 * rank; ++index) {
      if (initial.components[index].type != next.components[index].type ||
          initial.components[index].identity != next.components[index].identity)
        return failure();
    }

    SmallVector<unsigned> transferred;
    // An offset is carried only if the backedge state depends on the region
    // argument. Constant/invariant offsets remain outside the loop signature.
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      unsigned index = 2 * rank + dimension;
      if (failed(joinComponentTypes(initial.components[index].type,
                                    next.components[index].type)))
        return failure();
      if (regionArgument.components[index].identity !=
          next.components[index].identity)
        transferred.push_back(index);
    }
    return transferred;
  }

  FailureOr<SmallVector<unsigned>>
  getIfTransferredComponents(const AnalyzedValue &thenValue,
                             const AnalyzedValue &elseValue) const override {
    if (!hasValidLayout(thenValue) || !hasValidLayout(elseValue) ||
        thenValue.originalType != elseValue.originalType ||
        thenValue.invariants != elseValue.invariants ||
        thenValue.attributes != elseValue.attributes ||
        thenValue.components.size() != elseValue.components.size())
      return failure();

    // Unlike loops, an if can select shape, stride, or offset components. Base
    // and order remain invariants because AdapterIR cannot represent a runtime
    // selection between heterogeneous pointer descriptors.
    SmallVector<unsigned> transferred;
    for (unsigned index = 0; index < thenValue.components.size(); ++index) {
      if (failed(joinComponentTypes(thenValue.components[index].type,
                                    elseValue.components[index].type)))
        return failure();
      if (thenValue.components[index].identity !=
          elseValue.components[index].identity)
        transferred.push_back(index);
    }
    return transferred;
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
      result.components.append(makePtr.getShape().begin(),
                               makePtr.getShape().end());
      result.components.append(makePtr.getStrides().begin(),
                               makePtr.getStrides().end());
      result.components.append(makePtr.getOffsets().begin(),
                               makePtr.getOffsets().end());
      result.invariants.push_back(makePtr.getBase());
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
      unsigned component = 2 * *rank + dim;
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
    return builder.create<triton::MakeTensorPtrOp>(
        loc, value.originalType, value.invariants.front(),
        ValueRange(value.components).take_front(rank),
        ValueRange(value.components).slice(rank, rank),
        ValueRange(value.components).take_back(rank), order);
  }
};

} // namespace

namespace mlir::triton::controlflow {

LogicalResult runBlockPtrDecompose(ModuleOp module) {
  // Make the explicit descriptor carried by a block pointer cross each
  // supported SCF boundary as ordinary SSA components.
  BlockPtrPolicy policy;
  return rewriteControlFlow(module, policy);
}

} // namespace mlir::triton::controlflow
