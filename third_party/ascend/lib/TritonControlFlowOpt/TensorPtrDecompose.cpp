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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::controlflow;

namespace {

static constexpr unsigned kBaseComponent = 0;
static constexpr unsigned kStrideComponent = 1;
static constexpr unsigned kOffsetComponent = 2;
static constexpr unsigned kResidualComponent = 3;
static constexpr unsigned kBaseIsScalarAttribute = 0;
static constexpr unsigned kAxisKindsAttribute = 1;

enum class AxisKind : int32_t { Opaque = 0, Structured = 1 };

static constexpr unsigned getStrideComponent(unsigned axis) {
  return kBaseComponent + 1 + axis;
}

static constexpr unsigned getUniformOffsetComponent(unsigned rank) {
  return kBaseComponent + 1 + rank;
}

static constexpr unsigned getOpaqueContributionComponent(unsigned rank) {
  return getUniformOffsetComponent(rank) + 1;
}

static SmallVector<AxisKind> getAxisKinds(unsigned rank, AxisKind kind) {
  return SmallVector<AxisKind>(rank, kind);
}

static DenseI32ArrayAttr getAxisKindsAttr(MLIRContext *context,
                                          ArrayRef<AxisKind> kinds) {
  SmallVector<int32_t> values;
  values.reserve(kinds.size());
  for (AxisKind kind : kinds)
    values.push_back(static_cast<int32_t>(kind));
  return DenseI32ArrayAttr::get(context, values);
}

template <typename StateT>
static FailureOr<SmallVector<AxisKind>> getAxisKinds(const StateT &value) {
  auto pointerTensor = dyn_cast<RankedTensorType>(value.originalType);
  if (!pointerTensor || value.attributes.size() != 2)
    return failure();
  auto attribute =
      dyn_cast<DenseI32ArrayAttr>(value.attributes[kAxisKindsAttribute]);
  if (!attribute || static_cast<int64_t>(attribute.asArrayRef().size()) !=
                        pointerTensor.getRank())
    return failure();
  SmallVector<AxisKind> kinds;
  kinds.reserve(attribute.asArrayRef().size());
  for (int32_t rawKind : attribute.asArrayRef()) {
    if (rawKind != static_cast<int32_t>(AxisKind::Opaque) &&
        rawKind != static_cast<int32_t>(AxisKind::Structured))
      return failure();
    kinds.push_back(static_cast<AxisKind>(rawKind));
  }
  return kinds;
}

static bool isTensorPointerType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

static bool isCompatiblePointerBroadcast(Type sourceType, Type resultType) {
  auto sourceTensor = dyn_cast<RankedTensorType>(sourceType);
  auto resultTensor = dyn_cast<RankedTensorType>(resultType);
  if (!sourceTensor || !resultTensor || !isTensorPointerType(sourceType) ||
      !isTensorPointerType(resultType) ||
      sourceTensor.getRank() != resultTensor.getRank() ||
      sourceTensor.getElementType() != resultTensor.getElementType() ||
      sourceTensor.getEncoding() != resultTensor.getEncoding())
    return false;

  for (auto [sourceExtent, resultExtent] :
       llvm::zip(sourceTensor.getShape(), resultTensor.getShape())) {
    if (sourceExtent != resultExtent && sourceExtent != 1)
      return false;
  }
  return true;
}

static bool isRankOneTensorPointer(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         isa<triton::PointerType>(tensorType.getElementType());
}

static RankedTensorType getDefaultOffsetsType(Type pointerType) {
  auto pointerTensor = cast<RankedTensorType>(pointerType);
  return RankedTensorType::get(pointerTensor.getShape(),
                               IntegerType::get(pointerType.getContext(), 32),
                               pointerTensor.getEncoding());
}

static bool isCompatibleOffsetTensor(Type pointerType, Type offsetType) {
  auto pointerTensor = dyn_cast<RankedTensorType>(pointerType);
  auto offsetTensor = dyn_cast<RankedTensorType>(offsetType);
  return pointerTensor && offsetTensor &&
         pointerTensor.getShape() == offsetTensor.getShape() &&
         pointerTensor.getEncoding() == offsetTensor.getEncoding() &&
         isa<IntegerType>(offsetTensor.getElementType());
}

static bool hasValidSchema(Type originalType, TypeRange componentTypes,
                           ArrayRef<Attribute> attributes) {
  auto pointerTensor = dyn_cast<RankedTensorType>(originalType);
  if (!pointerTensor || !isTensorPointerType(originalType) ||
      attributes.size() != 2)
    return false;

  auto baseIsScalar = dyn_cast<BoolAttr>(attributes[kBaseIsScalarAttribute]);
  if (!baseIsScalar)
    return false;
  Type expectedBaseType = baseIsScalar.getValue()
                              ? IntegerType::get(originalType.getContext(), 64)
                              : originalType;
  unsigned rank = pointerTensor.getRank();
  if (rank == 0 || componentTypes.size() != rank + 3 ||
      componentTypes[kBaseComponent] != expectedBaseType)
    return false;

  auto axisKinds = dyn_cast<DenseI32ArrayAttr>(attributes[kAxisKindsAttribute]);
  if (!axisKinds || axisKinds.asArrayRef().size() != rank)
    return false;
  for (int32_t kind : axisKinds.asArrayRef()) {
    if (kind != static_cast<int32_t>(AxisKind::Opaque) &&
        kind != static_cast<int32_t>(AxisKind::Structured))
      return false;
  }

  auto uniformType =
      dyn_cast<IntegerType>(componentTypes[getUniformOffsetComponent(rank)]);
  auto opaqueType = dyn_cast<RankedTensorType>(
      componentTypes[getOpaqueContributionComponent(rank)]);
  if (!uniformType || !opaqueType ||
      opaqueType.getElementType() != uniformType ||
      !isCompatibleOffsetTensor(originalType, opaqueType))
    return false;
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (componentTypes[getStrideComponent(axis)] != uniformType)
      return false;
  }
  return true;
}

static bool hasValidLayout(const DecomposedValue &value) {
  return hasValidSchema(value.originalType,
                        TypeRange{ValueRange{value.components}},
                        value.attributes);
}

static bool hasValidLayout(const AnalyzedValue &value) {
  SmallVector<Type> componentTypes;
  componentTypes.reserve(value.components.size());
  for (const AnalyzedComponent &component : value.components)
    componentTypes.push_back(component.type);
  return hasValidSchema(value.originalType, componentTypes, value.attributes);
}

template <typename StateT> static bool hasScalarBase(const StateT &value) {
  return cast<BoolAttr>(value.attributes[kBaseIsScalarAttribute]).getValue();
}

static bool hasSameRepresentation(const AnalyzedValue &lhs,
                                  const AnalyzedValue &rhs) {
  if (!hasValidLayout(lhs) || !hasValidLayout(rhs) ||
      lhs.originalType != rhs.originalType ||
      hasScalarBase(lhs) != hasScalarBase(rhs))
    return false;
  if (hasScalarBase(lhs))
    return true;
  return lhs.components[kBaseComponent].type ==
             rhs.components[kBaseComponent].type &&
         lhs.components[kBaseComponent].identity ==
             rhs.components[kBaseComponent].identity;
}

static FailureOr<Type> getWiderIntegerLikeType(Type lhs, Type rhs) {
  if (lhs == rhs)
    return lhs;

  if (auto lhsInt = dyn_cast<IntegerType>(lhs)) {
    auto rhsInt = dyn_cast<IntegerType>(rhs);
    if (!rhsInt)
      return failure();
    return lhsInt.getWidth() >= rhsInt.getWidth() ? lhs : rhs;
  }

  auto lhsTensor = dyn_cast<RankedTensorType>(lhs);
  auto rhsTensor = dyn_cast<RankedTensorType>(rhs);
  if (!lhsTensor || !rhsTensor ||
      lhsTensor.getShape() != rhsTensor.getShape() ||
      lhsTensor.getEncoding() != rhsTensor.getEncoding())
    return failure();
  auto lhsInt = dyn_cast<IntegerType>(lhsTensor.getElementType());
  auto rhsInt = dyn_cast<IntegerType>(rhsTensor.getElementType());
  if (!lhsInt || !rhsInt)
    return failure();
  return lhsInt.getWidth() >= rhsInt.getWidth() ? lhs : rhs;
}

static bool isZeroIdentity(const ComponentIdentity &identity) {
  return identity.kind == ComponentIdentity::Kind::Zero;
}

static ComponentIdentity getAddIdentity(const ComponentIdentity &lhs,
                                        const ComponentIdentity &rhs,
                                        Value result, unsigned componentIndex) {
  if (isZeroIdentity(lhs))
    return rhs;
  if (isZeroIdentity(rhs))
    return lhs;
  return ComponentIdentity::fromValue(result, componentIndex);
}

static ComponentIdentity getSubIdentity(const ComponentIdentity &lhs,
                                        const ComponentIdentity &rhs,
                                        Value result, unsigned componentIndex) {
  if (isZeroIdentity(rhs))
    return lhs;
  return ComponentIdentity::fromValue(result, componentIndex);
}

static std::optional<int64_t> getScalarConstant(Value value) {
  return getConstantIntValue(value);
}

struct Rank1AnalyzedOffset {
  AnalyzedComponent stride;
  AnalyzedComponent offset;
  AnalyzedComponent residual;
};

struct Rank1OffsetValues {
  Value stride;
  Value offset;
  Value residual;
};

static FailureOr<RankedTensorType> getRank1IntegerTensorType(Value value,
                                                             Type pointerType) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || tensorType.getRank() != 1 ||
      !isCompatibleOffsetTensor(pointerType, tensorType))
    return failure();
  return tensorType;
}

struct DenseIntegerSplat {
  APInt value;
};

static FailureOr<Value> getSplatScalar(Value value) {
  auto splat = value.getDefiningOp<triton::SplatOp>();
  if (!splat || !isa<IntegerType>(splat.getSrc().getType()))
    return failure();
  return splat.getSrc();
}

/// Returns the exact scalar bit pattern represented by an integer dense splat.
/// Lane-varying dense constants deliberately fail this check and remain opaque.
static FailureOr<DenseIntegerSplat> getDenseIntegerSplat(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!tensorType || tensorType.getRank() == 0 ||
      !tensorType.hasStaticShape() ||
      !isa<IntegerType>(tensorType.getElementType()) || !constant)
    return failure();
  auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat() || dense.getType() != tensorType)
    return failure();
  APInt splatValue = dense.getSplatValue<APInt>();
  if (splatValue.getBitWidth() !=
      cast<IntegerType>(tensorType.getElementType()).getWidth())
    return failure();
  return DenseIntegerSplat{std::move(splatValue)};
}

/// Returns true when every tensor lane is produced by the same integer scalar.
/// Both an explicit tt.splat and an integer dense splat carry this guarantee.
/// Lane-varying tensor constants are intentionally excluded.
static bool isUniformIntegerScale(Value value) {
  return succeeded(getSplatScalar(value)) ||
         succeeded(getDenseIntegerSplat(value));
}

static bool isUniformIntegerScaleConstant(Value scale, int64_t expected) {
  if (FailureOr<Value> scalar = getSplatScalar(scale); succeeded(scalar))
    return getScalarConstant(*scalar) == std::optional<int64_t>(expected);
  FailureOr<DenseIntegerSplat> dense = getDenseIntegerSplat(scale);
  if (failed(dense))
    return false;
  return expected == 0 ? dense->value.isZero()
                       : expected == 1 && dense->value.isOne();
}

static ComponentIdentity getScaledIdentity(const ComponentIdentity &input,
                                           Value scale, Value result,
                                           unsigned componentIndex) {
  if (isZeroIdentity(input) || isUniformIntegerScaleConstant(scale, 0))
    return ComponentIdentity::zero(componentIndex);
  if (isUniformIntegerScaleConstant(scale, 1))
    return input;
  return ComponentIdentity::fromValue(result, componentIndex);
}

static Rank1AnalyzedOffset getFallbackAnalyzedOffset(Value value, Type) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  Type scalarType = tensorType.getElementType();
  return {
      {scalarType, ComponentIdentity::zero(kStrideComponent)},
      {scalarType, ComponentIdentity::zero(kOffsetComponent)},
      {tensorType, ComponentIdentity::fromValue(value, kResidualComponent)}};
}

static bool isSafeIntegerExtensionLeaf(Value value) {
  if (value.getDefiningOp<triton::MakeRangeOp>())
    return true;
  return succeeded(getSplatScalar(value)) ||
         succeeded(getDenseIntegerSplat(value));
}

static FailureOr<Rank1AnalyzedOffset> analyzeRank1Offset(Value value,
                                                         Type pointerType) {
  FailureOr<RankedTensorType> tensorType =
      getRank1IntegerTensorType(value, pointerType);
  if (failed(tensorType))
    return failure();
  Type scalarType = (*tensorType).getElementType();

  if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
    int64_t extent = (*tensorType).getShape()[0];
    if (range.getEnd() - range.getStart() != extent)
      return getFallbackAnalyzedOffset(value, pointerType);
    ComponentIdentity offsetIdentity =
        range.getStart() == 0
            ? ComponentIdentity::zero(kOffsetComponent)
            : ComponentIdentity::fromValue(value, kOffsetComponent);
    return Rank1AnalyzedOffset{
        {scalarType, ComponentIdentity::fromValue(value, kStrideComponent)},
        {scalarType, offsetIdentity},
        {*tensorType, ComponentIdentity::zero(kResidualComponent)}};
  }

  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    if (!isa<IntegerType>(splat.getSrc().getType()))
      return getFallbackAnalyzedOffset(value, pointerType);
    ComponentIdentity offsetIdentity =
        getScalarConstant(splat.getSrc()) == std::optional<int64_t>(0)
            ? ComponentIdentity::zero(kOffsetComponent)
            : ComponentIdentity::fromValue(splat.getSrc(), kOffsetComponent);
    return Rank1AnalyzedOffset{
        {scalarType, ComponentIdentity::zero(kStrideComponent)},
        {scalarType, offsetIdentity},
        {*tensorType, ComponentIdentity::zero(kResidualComponent)}};
  }

  if (FailureOr<DenseIntegerSplat> dense = getDenseIntegerSplat(value);
      succeeded(dense)) {
    ComponentIdentity offsetIdentity =
        dense->value.isZero()
            ? ComponentIdentity::zero(kOffsetComponent)
            : ComponentIdentity::fromValue(value, kOffsetComponent);
    return Rank1AnalyzedOffset{
        {scalarType, ComponentIdentity::zero(kStrideComponent)},
        {scalarType, offsetIdentity},
        {*tensorType, ComponentIdentity::zero(kResidualComponent)}};
  }

  if (auto extension = value.getDefiningOp<arith::ExtSIOp>()) {
    Value source = extension.getIn();
    if (!isSafeIntegerExtensionLeaf(source))
      return getFallbackAnalyzedOffset(value, pointerType);
    FailureOr<Rank1AnalyzedOffset> sourceInfo =
        analyzeRank1Offset(source, pointerType);
    if (failed(sourceInfo))
      return failure();
    auto extendedIdentity = [&](const AnalyzedComponent &component,
                                unsigned componentIndex) {
      return isZeroIdentity(component.identity)
                 ? ComponentIdentity::zero(componentIndex)
                 : ComponentIdentity::fromValue(value, componentIndex);
    };
    return Rank1AnalyzedOffset{
        {scalarType, extendedIdentity(sourceInfo->stride, kStrideComponent)},
        {scalarType, extendedIdentity(sourceInfo->offset, kOffsetComponent)},
        {*tensorType,
         extendedIdentity(sourceInfo->residual, kResidualComponent)}};
  }

  auto analyzeBinary = [&](Value lhs, Value rhs,
                           bool subtract) -> FailureOr<Rank1AnalyzedOffset> {
    FailureOr<Rank1AnalyzedOffset> lhsInfo =
        analyzeRank1Offset(lhs, pointerType);
    FailureOr<Rank1AnalyzedOffset> rhsInfo =
        analyzeRank1Offset(rhs, pointerType);
    if (failed(lhsInfo) || failed(rhsInfo) ||
        lhsInfo->stride.type != rhsInfo->stride.type ||
        lhsInfo->offset.type != rhsInfo->offset.type ||
        lhsInfo->residual.type != rhsInfo->residual.type)
      return getFallbackAnalyzedOffset(value, pointerType);

    // One rank-1 axis is represented either entirely by scalar affine fields
    // or entirely by the opaque carrier. Do not retain a partial affine term
    // beside an existing residual.
    if (!isZeroIdentity(lhsInfo->residual.identity) ||
        !isZeroIdentity(rhsInfo->residual.identity))
      return getFallbackAnalyzedOffset(value, pointerType);

    auto combine = [&](const ComponentIdentity &lhsIdentity,
                       const ComponentIdentity &rhsIdentity,
                       unsigned componentIndex) {
      return subtract ? getSubIdentity(lhsIdentity, rhsIdentity, value,
                                       componentIndex)
                      : getAddIdentity(lhsIdentity, rhsIdentity, value,
                                       componentIndex);
    };
    return Rank1AnalyzedOffset{
        {lhsInfo->stride.type,
         combine(lhsInfo->stride.identity, rhsInfo->stride.identity,
                 kStrideComponent)},
        {lhsInfo->offset.type,
         combine(lhsInfo->offset.identity, rhsInfo->offset.identity,
                 kOffsetComponent)},
        {lhsInfo->residual.type,
         combine(lhsInfo->residual.identity, rhsInfo->residual.identity,
                 kResidualComponent)}};
  };

  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return analyzeBinary(add.getLhs(), add.getRhs(), false);
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return analyzeBinary(sub.getLhs(), sub.getRhs(), true);

  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    Value affineOperand = mul.getLhs();
    Value scale = mul.getRhs();
    if (!isUniformIntegerScale(scale)) {
      affineOperand = mul.getRhs();
      scale = mul.getLhs();
    }
    if (!isUniformIntegerScale(scale))
      return getFallbackAnalyzedOffset(value, pointerType);
    FailureOr<Rank1AnalyzedOffset> input =
        analyzeRank1Offset(affineOperand, pointerType);
    if (failed(input) || !isZeroIdentity(input->residual.identity))
      return getFallbackAnalyzedOffset(value, pointerType);
    return Rank1AnalyzedOffset{
        {input->stride.type, getScaledIdentity(input->stride.identity, scale,
                                               value, kStrideComponent)},
        {input->offset.type, getScaledIdentity(input->offset.identity, scale,
                                               value, kOffsetComponent)},
        {input->residual.type,
         getScaledIdentity(input->residual.identity, scale, value,
                           kResidualComponent)}};
  }

  // Pushing an extension through a previously computed tensor expression can
  // change the source-width overflow semantics. Keep the already-evaluated
  // tensor as residual until the schema records arithmetic width explicitly.
  return getFallbackAnalyzedOffset(value, pointerType);
}

static bool isConstantZero(Value value) {
  if (!value)
    return false;
  if (std::optional<int64_t> scalar = getConstantIntValue(value))
    return *scalar == 0;
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
  return dense && dense.isSplat() && dense.getSplatValue<APInt>().isZero();
}

static Value createScalarConstant(OpBuilder &builder, Location loc, Type type,
                                  int64_t value) {
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType)
    return nullptr;
  return builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(intType, value));
}

static Value createScalarConstant(OpBuilder &builder, Location loc, Type type,
                                  const APInt &value) {
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || intType.getWidth() != value.getBitWidth())
    return nullptr;
  return builder.create<arith::ConstantOp>(loc,
                                           IntegerAttr::get(intType, value));
}

/// Materializes the scalar represented by an integer-uniform tensor. Dense
/// splats are recreated with their exact APInt width so affine stride analysis
/// and concrete descriptor construction observe identical arithmetic.
static FailureOr<Value>
materializeUniformIntegerScale(Value value, OpBuilder &builder, Location loc) {
  if (FailureOr<Value> scalar = getSplatScalar(value); succeeded(scalar))
    return *scalar;
  FailureOr<DenseIntegerSplat> dense = getDenseIntegerSplat(value);
  if (failed(dense))
    return failure();
  auto tensorType = cast<RankedTensorType>(value.getType());
  Value scalar = createScalarConstant(builder, loc, tensorType.getElementType(),
                                      dense->value);
  if (!scalar)
    return failure();
  return scalar;
}

static Value createZeroOffsets(OpBuilder &builder, Location loc,
                               RankedTensorType offsetsType) {
  auto elementType = cast<IntegerType>(offsetsType.getElementType());
  return builder.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(offsetsType,
                                  builder.getIntegerAttr(elementType, 0)));
}

static Value createZeroOffsets(OpBuilder &builder, Location loc,
                               Type pointerType) {
  return createZeroOffsets(builder, loc, getDefaultOffsetsType(pointerType));
}

static Value createIntegerAdd(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs) {
  if (isConstantZero(lhs))
    return rhs;
  if (isConstantZero(rhs))
    return lhs;
  FailureOr<Type> type = getWiderIntegerLikeType(lhs.getType(), rhs.getType());
  if (failed(type))
    return nullptr;
  FailureOr<Value> convertedLhs = castIntegerLike(builder, loc, lhs, *type);
  FailureOr<Value> convertedRhs = castIntegerLike(builder, loc, rhs, *type);
  if (failed(convertedLhs) || failed(convertedRhs))
    return nullptr;
  return builder.create<arith::AddIOp>(loc, *convertedLhs, *convertedRhs);
}

static Value createIntegerSub(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs) {
  if (isConstantZero(rhs))
    return lhs;
  FailureOr<Type> type = getWiderIntegerLikeType(lhs.getType(), rhs.getType());
  if (failed(type))
    return nullptr;
  FailureOr<Value> convertedLhs = castIntegerLike(builder, loc, lhs, *type);
  FailureOr<Value> convertedRhs = castIntegerLike(builder, loc, rhs, *type);
  if (failed(convertedLhs) || failed(convertedRhs))
    return nullptr;
  return builder.create<arith::SubIOp>(loc, *convertedLhs, *convertedRhs);
}

static Value scaleIntegerValue(OpBuilder &builder, Location loc, Value value,
                               Value scale) {
  if (isConstantZero(value))
    return value;
  std::optional<int64_t> constant = getScalarConstant(scale);
  if (constant && *constant == 0) {
    if (auto tensorType = dyn_cast<RankedTensorType>(value.getType()))
      return createZeroOffsets(builder, loc, tensorType);
    return createScalarConstant(builder, loc, value.getType(), 0);
  }
  if (constant && *constant == 1)
    return value;

  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    FailureOr<Value> convertedScale =
        castIntegerLike(builder, loc, scale, tensorType.getElementType());
    if (failed(convertedScale))
      return nullptr;
    Value scaleTensor =
        builder.create<triton::SplatOp>(loc, tensorType, *convertedScale);
    return builder.create<arith::MulIOp>(loc, value, scaleTensor);
  }

  FailureOr<Type> type =
      getWiderIntegerLikeType(value.getType(), scale.getType());
  if (failed(type))
    return nullptr;
  FailureOr<Value> convertedValue = castIntegerLike(builder, loc, value, *type);
  FailureOr<Value> convertedScale = castIntegerLike(builder, loc, scale, *type);
  if (failed(convertedValue) || failed(convertedScale))
    return nullptr;
  return builder.create<arith::MulIOp>(loc, *convertedValue, *convertedScale);
}

static FailureOr<Rank1OffsetValues> materializeRank1Offset(Value value,
                                                           Type pointerType,
                                                           OpBuilder &builder,
                                                           Location loc) {
  FailureOr<RankedTensorType> tensorType =
      getRank1IntegerTensorType(value, pointerType);
  if (failed(tensorType))
    return failure();

  // Component identities are keyed by the analyzed tensor expression. Create
  // its concrete scalar fields next to that expression, where they dominate
  // every reuse (including sibling scf.if arms). Materializing at each addptr
  // user would make equal abstract identities refer to branch-local Values.
  OpBuilder::InsertionGuard insertionGuard(builder);
  if (Operation *definingOp = value.getDefiningOp())
    builder.setInsertionPointAfter(definingOp);
  else if (auto blockArgument = dyn_cast<BlockArgument>(value))
    builder.setInsertionPointToStart(blockArgument.getOwner());

  Type scalarType = (*tensorType).getElementType();

  auto fallback = [&]() -> FailureOr<Rank1OffsetValues> {
    Value stride = createScalarConstant(builder, loc, scalarType, 0);
    Value offset = createScalarConstant(builder, loc, scalarType, 0);
    if (!stride || !offset)
      return failure();
    return Rank1OffsetValues{stride, offset, value};
  };

  if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
    if (range.getEnd() - range.getStart() != (*tensorType).getShape()[0])
      return fallback();
    Value stride = createScalarConstant(builder, loc, scalarType, 1);
    Value offset =
        createScalarConstant(builder, loc, scalarType, range.getStart());
    Value residual = createZeroOffsets(builder, loc, *tensorType);
    if (!stride || !offset || !residual)
      return failure();
    return Rank1OffsetValues{stride, offset, residual};
  }

  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    if (!isa<IntegerType>(splat.getSrc().getType()))
      return fallback();
    FailureOr<Value> offset =
        castIntegerLike(builder, loc, splat.getSrc(), scalarType);
    Value stride = createScalarConstant(builder, loc, scalarType, 0);
    Value residual = createZeroOffsets(builder, loc, *tensorType);
    if (failed(offset) || !stride || !residual)
      return failure();
    return Rank1OffsetValues{stride, *offset, residual};
  }

  if (FailureOr<DenseIntegerSplat> dense = getDenseIntegerSplat(value);
      succeeded(dense)) {
    Value stride = createScalarConstant(builder, loc, scalarType, 0);
    Value offset = createScalarConstant(builder, loc, scalarType, dense->value);
    Value residual = createZeroOffsets(builder, loc, *tensorType);
    if (!stride || !offset || !residual)
      return failure();
    return Rank1OffsetValues{stride, offset, residual};
  }

  if (auto extension = value.getDefiningOp<arith::ExtSIOp>()) {
    Value source = extension.getIn();
    if (!isSafeIntegerExtensionLeaf(source))
      return fallback();
    FailureOr<Rank1OffsetValues> sourceInfo =
        materializeRank1Offset(source, pointerType, builder, loc);
    if (failed(sourceInfo))
      return failure();
    auto extendComponent = [&](Value component,
                               Type targetType) -> FailureOr<Value> {
      if (!isConstantZero(component))
        return castIntegerLike(builder, loc, component, targetType);
      if (auto targetTensor = dyn_cast<RankedTensorType>(targetType))
        return createZeroOffsets(builder, loc, targetTensor);
      Value zero = createScalarConstant(builder, loc, targetType, 0);
      if (!zero)
        return failure();
      return zero;
    };
    FailureOr<Value> stride = extendComponent(sourceInfo->stride, scalarType);
    FailureOr<Value> offset = extendComponent(sourceInfo->offset, scalarType);
    FailureOr<Value> residual =
        extendComponent(sourceInfo->residual, *tensorType);
    if (failed(stride) || failed(offset) || failed(residual))
      return failure();
    return Rank1OffsetValues{*stride, *offset, *residual};
  }

  auto materializeBinary = [&](Value lhs, Value rhs,
                               bool subtract) -> FailureOr<Rank1OffsetValues> {
    FailureOr<Rank1OffsetValues> lhsInfo =
        materializeRank1Offset(lhs, pointerType, builder, loc);
    FailureOr<Rank1OffsetValues> rhsInfo =
        materializeRank1Offset(rhs, pointerType, builder, loc);
    if (failed(lhsInfo) || failed(rhsInfo))
      return failure();
    if (!isConstantZero(lhsInfo->residual) ||
        !isConstantZero(rhsInfo->residual))
      return fallback();
    auto combine = [&](Value lhsValue, Value rhsValue) {
      return subtract ? createIntegerSub(builder, loc, lhsValue, rhsValue)
                      : createIntegerAdd(builder, loc, lhsValue, rhsValue);
    };
    Value stride = combine(lhsInfo->stride, rhsInfo->stride);
    Value offset = combine(lhsInfo->offset, rhsInfo->offset);
    Value residual = combine(lhsInfo->residual, rhsInfo->residual);
    if (!stride || !offset || !residual)
      return failure();
    return Rank1OffsetValues{stride, offset, residual};
  };

  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return materializeBinary(add.getLhs(), add.getRhs(), false);
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return materializeBinary(sub.getLhs(), sub.getRhs(), true);

  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    Value affineOperand = mul.getLhs();
    FailureOr<Value> scale =
        materializeUniformIntegerScale(mul.getRhs(), builder, loc);
    if (failed(scale)) {
      affineOperand = mul.getRhs();
      scale = materializeUniformIntegerScale(mul.getLhs(), builder, loc);
    }
    if (failed(scale))
      return fallback();
    FailureOr<Rank1OffsetValues> input =
        materializeRank1Offset(affineOperand, pointerType, builder, loc);
    if (failed(input))
      return failure();
    if (!isConstantZero(input->residual))
      return fallback();
    Value stride = scaleIntegerValue(builder, loc, input->stride, *scale);
    Value offset = scaleIntegerValue(builder, loc, input->offset, *scale);
    Value residual = scaleIntegerValue(builder, loc, input->residual, *scale);
    if (!stride || !offset || !residual)
      return failure();
    return Rank1OffsetValues{stride, offset, residual};
  }

  return fallback();
}

static Value materializeRank1CompleteOffsets(const DecomposedValue &value,
                                             OpBuilder &builder, Location loc) {
  if (!hasValidLayout(value) || !isRankOneTensorPointer(value.originalType))
    return nullptr;
  Value stride = value.components[kStrideComponent];
  Value offset = value.components[kOffsetComponent];
  Value residual = value.components[kResidualComponent];
  auto residualType = cast<RankedTensorType>(residual.getType());
  int64_t extent = residualType.getShape()[0];

  // Zero is a policy-level identity, not an SSA owner. A concrete zero may
  // have been created inside one arm of an scf.if and therefore cannot be used
  // by the rebuild after the if. Rematerialize every proven zero at the current
  // insertion point so empty/invariant slots remain dominance-correct.
  if (isConstantZero(stride))
    stride = createScalarConstant(builder, loc, stride.getType(), 0);
  if (isConstantZero(offset))
    offset = createScalarConstant(builder, loc, offset.getType(), 0);
  if (isConstantZero(residual))
    residual = createZeroOffsets(builder, loc, residualType);
  if (!stride || !offset || !residual)
    return nullptr;

  if (isConstantZero(stride) && isConstantZero(offset))
    return residual;
  if (extent < 0 || extent > std::numeric_limits<int32_t>::max())
    return nullptr;

  auto rangeType =
      RankedTensorType::get(residualType.getShape(), builder.getI32Type(),
                            residualType.getEncoding());
  Value range = builder.create<triton::MakeRangeOp>(
      loc, rangeType, 0, static_cast<int32_t>(extent));
  FailureOr<Value> convertedRange =
      castIntegerLike(builder, loc, range, residualType);
  FailureOr<Value> convertedStride =
      castIntegerLike(builder, loc, stride, residualType.getElementType());
  FailureOr<Value> convertedOffset =
      castIntegerLike(builder, loc, offset, residualType.getElementType());
  if (failed(convertedRange) || failed(convertedStride) ||
      failed(convertedOffset))
    return nullptr;

  Value strideTensor =
      builder.create<triton::SplatOp>(loc, residualType, *convertedStride);
  Value offsetTensor =
      builder.create<triton::SplatOp>(loc, residualType, *convertedOffset);
  Value affine =
      builder.create<arith::MulIOp>(loc, *convertedRange, strideTensor);
  affine = createIntegerAdd(builder, loc, affine, offsetTensor);
  if (!affine)
    return nullptr;
  if (isConstantZero(residual))
    return affine;
  return createIntegerAdd(builder, loc, affine, residual);
}

struct AnalyzedTensorOffset {
  SmallVector<AnalyzedComponent> strides;
  AnalyzedComponent uniformOffset;
  AnalyzedComponent opaqueContribution;
  SmallVector<AxisKind> axisKinds;
};

struct TensorOffsetValues {
  SmallVector<Value> strides;
  Value uniformOffset;
  Value opaqueContribution;
  SmallVector<AxisKind> axisKinds;
};

static FailureOr<RankedTensorType> getIntegerTensorType(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() == 0 || !isa<IntegerType>(type.getElementType()))
    return failure();
  return type;
}

static bool hasAnyOpaqueAxis(ArrayRef<AxisKind> kinds) {
  return llvm::is_contained(kinds, AxisKind::Opaque);
}

static AnalyzedTensorOffset getOpaqueAnalyzedOffset(Value value) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  Type scalarType = tensorType.getElementType();
  unsigned rank = tensorType.getRank();
  AnalyzedTensorOffset result;
  result.axisKinds = getAxisKinds(rank, AxisKind::Opaque);
  for (unsigned axis = 0; axis < rank; ++axis) {
    result.strides.push_back(
        {scalarType, ComponentIdentity::zero(getStrideComponent(axis))});
  }
  result.uniformOffset = {
      scalarType, ComponentIdentity::zero(getUniformOffsetComponent(rank))};
  result.opaqueContribution = {
      tensorType, ComponentIdentity::fromValue(
                      value, getOpaqueContributionComponent(rank))};
  return result;
}

static FailureOr<AnalyzedTensorOffset> analyzeTensorOffset(Value value) {
  FailureOr<RankedTensorType> tensorType = getIntegerTensorType(value);
  if (failed(tensorType))
    return failure();
  unsigned rank = (*tensorType).getRank();
  Type scalarType = (*tensorType).getElementType();

  if (rank == 1) {
    FailureOr<Rank1AnalyzedOffset> rankOne =
        analyzeRank1Offset(value, value.getType());
    if (failed(rankOne))
      return failure();
    bool opaque = !isZeroIdentity(rankOne->residual.identity);
    if (opaque)
      return getOpaqueAnalyzedOffset(value);
    return AnalyzedTensorOffset{{rankOne->stride},
                                rankOne->offset,
                                rankOne->residual,
                                {AxisKind::Structured}};
  }

  auto makeStructuredZero = [&]() {
    AnalyzedTensorOffset result;
    result.axisKinds = getAxisKinds(rank, AxisKind::Structured);
    for (unsigned axis = 0; axis < rank; ++axis) {
      result.strides.push_back(
          {scalarType, ComponentIdentity::zero(getStrideComponent(axis))});
    }
    result.uniformOffset = {
        scalarType, ComponentIdentity::zero(getUniformOffsetComponent(rank))};
    result.opaqueContribution = {
        *tensorType,
        ComponentIdentity::zero(getOpaqueContributionComponent(rank))};
    return result;
  };

  if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
    if (rank != 1 ||
        range.getEnd() - range.getStart() != (*tensorType).getShape()[0])
      return getOpaqueAnalyzedOffset(value);
    AnalyzedTensorOffset result = makeStructuredZero();
    result.strides[0].identity =
        ComponentIdentity::fromValue(value, getStrideComponent(0));
    if (range.getStart() != 0)
      result.uniformOffset.identity =
          ComponentIdentity::fromValue(value, getUniformOffsetComponent(rank));
    return result;
  }

  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    if (!isa<IntegerType>(splat.getSrc().getType()))
      return getOpaqueAnalyzedOffset(value);
    AnalyzedTensorOffset result = makeStructuredZero();
    if (getScalarConstant(splat.getSrc()) != std::optional<int64_t>(0))
      result.uniformOffset.identity = ComponentIdentity::fromValue(
          splat.getSrc(), getUniformOffsetComponent(rank));
    return result;
  }

  if (FailureOr<DenseIntegerSplat> dense = getDenseIntegerSplat(value);
      succeeded(dense)) {
    AnalyzedTensorOffset result = makeStructuredZero();
    if (!dense->value.isZero())
      result.uniformOffset.identity =
          ComponentIdentity::fromValue(value, getUniformOffsetComponent(rank));
    return result;
  }

  if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
    FailureOr<AnalyzedTensorOffset> source =
        analyzeTensorOffset(expand.getSrc());
    if (failed(source) || source->strides.size() + 1 != rank ||
        expand.getAxis() >= rank)
      return getOpaqueAnalyzedOffset(value);
    AnalyzedTensorOffset result = makeStructuredZero();
    unsigned sourceAxis = 0;
    for (unsigned axis = 0; axis < rank; ++axis) {
      if (axis == expand.getAxis())
        continue;
      result.axisKinds[axis] = source->axisKinds[sourceAxis];
      if (!isZeroIdentity(source->strides[sourceAxis].identity))
        result.strides[axis].identity =
            ComponentIdentity::fromValue(value, getStrideComponent(axis));
      ++sourceAxis;
    }
    if (!isZeroIdentity(source->uniformOffset.identity))
      result.uniformOffset.identity =
          ComponentIdentity::fromValue(value, getUniformOffsetComponent(rank));
    if (!isZeroIdentity(source->opaqueContribution.identity))
      result.opaqueContribution.identity = ComponentIdentity::fromValue(
          value, getOpaqueContributionComponent(rank));
    return result;
  }

  if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
    FailureOr<AnalyzedTensorOffset> source =
        analyzeTensorOffset(broadcast.getSrc());
    auto sourceType = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
    if (failed(source) || !sourceType || sourceType.getRank() != rank)
      return getOpaqueAnalyzedOffset(value);
    AnalyzedTensorOffset result = makeStructuredZero();
    for (unsigned axis = 0; axis < rank; ++axis) {
      bool expandsScalarAxis = sourceType.getShape()[axis] == 1 &&
                               (*tensorType).getShape()[axis] != 1;
      if (expandsScalarAxis)
        continue;
      result.axisKinds[axis] = source->axisKinds[axis];
      if (!isZeroIdentity(source->strides[axis].identity))
        result.strides[axis].identity =
            ComponentIdentity::fromValue(value, getStrideComponent(axis));
    }
    if (!isZeroIdentity(source->uniformOffset.identity))
      result.uniformOffset.identity =
          ComponentIdentity::fromValue(value, getUniformOffsetComponent(rank));
    if (!isZeroIdentity(source->opaqueContribution.identity))
      result.opaqueContribution.identity = ComponentIdentity::fromValue(
          value, getOpaqueContributionComponent(rank));
    return result;
  }

  auto analyzeBinary = [&](Value lhs, Value rhs,
                           bool subtract) -> FailureOr<AnalyzedTensorOffset> {
    FailureOr<AnalyzedTensorOffset> lhsInfo = analyzeTensorOffset(lhs);
    FailureOr<AnalyzedTensorOffset> rhsInfo = analyzeTensorOffset(rhs);
    if (failed(lhsInfo) || failed(rhsInfo) || lhsInfo->strides.size() != rank ||
        rhsInfo->strides.size() != rank)
      return getOpaqueAnalyzedOffset(value);
    AnalyzedTensorOffset result = makeStructuredZero();
    for (unsigned axis = 0; axis < rank; ++axis) {
      result.axisKinds[axis] =
          lhsInfo->axisKinds[axis] == AxisKind::Structured &&
                  rhsInfo->axisKinds[axis] == AxisKind::Structured
              ? AxisKind::Structured
              : AxisKind::Opaque;
      if (result.axisKinds[axis] == AxisKind::Structured) {
        result.strides[axis].identity =
            subtract ? getSubIdentity(lhsInfo->strides[axis].identity,
                                      rhsInfo->strides[axis].identity, value,
                                      getStrideComponent(axis))
                     : getAddIdentity(lhsInfo->strides[axis].identity,
                                      rhsInfo->strides[axis].identity, value,
                                      getStrideComponent(axis));
      }
    }
    if (hasAnyOpaqueAxis(result.axisKinds)) {
      result.opaqueContribution.identity = ComponentIdentity::fromValue(
          value, getOpaqueContributionComponent(rank));
      return result;
    }
    result.uniformOffset.identity =
        subtract ? getSubIdentity(lhsInfo->uniformOffset.identity,
                                  rhsInfo->uniformOffset.identity, value,
                                  getUniformOffsetComponent(rank))
                 : getAddIdentity(lhsInfo->uniformOffset.identity,
                                  rhsInfo->uniformOffset.identity, value,
                                  getUniformOffsetComponent(rank));
    return result;
  };

  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return analyzeBinary(add.getLhs(), add.getRhs(), false);
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return analyzeBinary(sub.getLhs(), sub.getRhs(), true);

  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    Value affineOperand = mul.getLhs();
    Value scale = mul.getRhs();
    if (!isUniformIntegerScale(scale)) {
      affineOperand = mul.getRhs();
      scale = mul.getLhs();
    }
    if (!isUniformIntegerScale(scale))
      return getOpaqueAnalyzedOffset(value);
    FailureOr<AnalyzedTensorOffset> input = analyzeTensorOffset(affineOperand);
    if (failed(input) || input->strides.size() != rank)
      return getOpaqueAnalyzedOffset(value);
    AnalyzedTensorOffset result = makeStructuredZero();
    result.axisKinds = input->axisKinds;
    for (unsigned axis = 0; axis < rank; ++axis) {
      if (result.axisKinds[axis] == AxisKind::Structured)
        result.strides[axis].identity =
            getScaledIdentity(input->strides[axis].identity, scale, value,
                              getStrideComponent(axis));
    }
    if (hasAnyOpaqueAxis(result.axisKinds)) {
      result.opaqueContribution.identity = ComponentIdentity::fromValue(
          value, getOpaqueContributionComponent(rank));
    } else {
      result.uniformOffset.identity =
          getScaledIdentity(input->uniformOffset.identity, scale, value,
                            getUniformOffsetComponent(rank));
    }
    return result;
  }

  return getOpaqueAnalyzedOffset(value);
}

static FailureOr<Value> materializeAxisRange(OpBuilder &builder, Location loc,
                                             RankedTensorType tensorType,
                                             unsigned axis) {
  if (axis >= static_cast<unsigned>(tensorType.getRank()))
    return failure();
  int64_t extent = tensorType.getShape()[axis];
  if (extent < 0 || extent > std::numeric_limits<int32_t>::max())
    return failure();
  auto rangeType = RankedTensorType::get({extent}, builder.getI32Type());
  Value range = builder.create<triton::MakeRangeOp>(
      loc, rangeType, 0, static_cast<int32_t>(extent));
  auto elementRangeType =
      RankedTensorType::get({extent}, tensorType.getElementType());
  FailureOr<Value> converted =
      castIntegerLike(builder, loc, range, elementRangeType);
  if (failed(converted))
    return failure();
  Value expanded = *converted;
  for (unsigned dimension = 0;
       dimension < static_cast<unsigned>(tensorType.getRank()); ++dimension) {
    if (dimension == axis)
      continue;
    expanded = builder.create<triton::ExpandDimsOp>(loc, expanded, dimension);
  }
  if (expanded.getType() != tensorType)
    expanded = builder.create<triton::BroadcastOp>(loc, tensorType, expanded);
  return expanded;
}

static FailureOr<Value> materializeStructuredContribution(
    ArrayRef<Value> strides, ArrayRef<AxisKind> axisKinds,
    RankedTensorType tensorType, OpBuilder &builder, Location loc) {
  if (strides.size() != axisKinds.size() ||
      strides.size() != static_cast<size_t>(tensorType.getRank()))
    return failure();
  Value contribution = createZeroOffsets(builder, loc, tensorType);
  for (unsigned axis = 0; axis < strides.size(); ++axis) {
    if (axisKinds[axis] != AxisKind::Structured ||
        isConstantZero(strides[axis]))
      continue;
    FailureOr<Value> range =
        materializeAxisRange(builder, loc, tensorType, axis);
    FailureOr<Value> stride = castIntegerLike(builder, loc, strides[axis],
                                              tensorType.getElementType());
    if (failed(range) || failed(stride))
      return failure();
    Value strideTensor =
        builder.create<triton::SplatOp>(loc, tensorType, *stride);
    Value axisContribution =
        builder.create<arith::MulIOp>(loc, *range, strideTensor);
    contribution =
        createIntegerAdd(builder, loc, contribution, axisContribution);
    if (!contribution)
      return failure();
  }
  return contribution;
}

static FailureOr<TensorOffsetValues>
materializeTensorOffsetFields(Value value, OpBuilder &builder, Location loc) {
  FailureOr<RankedTensorType> tensorType = getIntegerTensorType(value);
  if (failed(tensorType))
    return failure();
  unsigned rank = (*tensorType).getRank();
  Type scalarType = (*tensorType).getElementType();

  auto makeZero = [&](AxisKind kind) -> FailureOr<TensorOffsetValues> {
    TensorOffsetValues result;
    result.axisKinds = getAxisKinds(rank, kind);
    for (unsigned axis = 0; axis < rank; ++axis) {
      Value stride = createScalarConstant(builder, loc, scalarType, 0);
      if (!stride)
        return failure();
      result.strides.push_back(stride);
    }
    result.uniformOffset = createScalarConstant(builder, loc, scalarType, 0);
    result.opaqueContribution = createZeroOffsets(builder, loc, *tensorType);
    if (!result.uniformOffset || !result.opaqueContribution)
      return failure();
    return result;
  };

  auto fallback = [&]() -> FailureOr<TensorOffsetValues> {
    FailureOr<TensorOffsetValues> result = makeZero(AxisKind::Opaque);
    if (failed(result))
      return failure();
    result->opaqueContribution = value;
    return result;
  };

  if (rank == 1) {
    FailureOr<Rank1OffsetValues> rankOne =
        materializeRank1Offset(value, value.getType(), builder, loc);
    if (failed(rankOne))
      return failure();
    if (!isConstantZero(rankOne->residual))
      return fallback();
    return TensorOffsetValues{{rankOne->stride},
                              rankOne->offset,
                              rankOne->residual,
                              {AxisKind::Structured}};
  }

  if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
    if (rank != 1 ||
        range.getEnd() - range.getStart() != (*tensorType).getShape()[0])
      return fallback();
    FailureOr<TensorOffsetValues> result = makeZero(AxisKind::Structured);
    if (failed(result))
      return failure();
    result->strides[0] = createScalarConstant(builder, loc, scalarType, 1);
    result->uniformOffset =
        createScalarConstant(builder, loc, scalarType, range.getStart());
    if (!result->strides[0] || !result->uniformOffset)
      return failure();
    return result;
  }

  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    if (!isa<IntegerType>(splat.getSrc().getType()))
      return fallback();
    FailureOr<TensorOffsetValues> result = makeZero(AxisKind::Structured);
    if (failed(result))
      return failure();
    FailureOr<Value> uniform =
        castIntegerLike(builder, loc, splat.getSrc(), scalarType);
    if (failed(uniform))
      return failure();
    result->uniformOffset = *uniform;
    return result;
  }

  if (FailureOr<DenseIntegerSplat> dense = getDenseIntegerSplat(value);
      succeeded(dense)) {
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointAfter(value.getDefiningOp());
    FailureOr<TensorOffsetValues> result = makeZero(AxisKind::Structured);
    if (failed(result))
      return failure();
    Value uniform =
        createScalarConstant(builder, loc, scalarType, dense->value);
    if (!uniform)
      return failure();
    result->uniformOffset = uniform;
    return result;
  }

  if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
    FailureOr<TensorOffsetValues> source =
        materializeTensorOffsetFields(expand.getSrc(), builder, loc);
    if (failed(source) || source->strides.size() + 1 != rank ||
        expand.getAxis() >= rank)
      return fallback();
    FailureOr<TensorOffsetValues> result = makeZero(AxisKind::Structured);
    if (failed(result))
      return failure();
    unsigned sourceAxis = 0;
    for (unsigned axis = 0; axis < rank; ++axis) {
      if (axis == expand.getAxis())
        continue;
      result->strides[axis] = source->strides[sourceAxis];
      result->axisKinds[axis] = source->axisKinds[sourceAxis];
      ++sourceAxis;
    }
    result->uniformOffset = source->uniformOffset;
    return result;
  }

  if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
    FailureOr<TensorOffsetValues> source =
        materializeTensorOffsetFields(broadcast.getSrc(), builder, loc);
    auto sourceType = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
    if (failed(source) || !sourceType || sourceType.getRank() != rank)
      return fallback();
    FailureOr<TensorOffsetValues> result = makeZero(AxisKind::Structured);
    if (failed(result))
      return failure();
    for (unsigned axis = 0; axis < rank; ++axis) {
      bool expandsScalarAxis = sourceType.getShape()[axis] == 1 &&
                               (*tensorType).getShape()[axis] != 1;
      if (expandsScalarAxis)
        continue;
      result->strides[axis] = source->strides[axis];
      result->axisKinds[axis] = source->axisKinds[axis];
    }
    result->uniformOffset = source->uniformOffset;
    return result;
  }

  auto materializeBinary = [&](Value lhs, Value rhs,
                               bool subtract) -> FailureOr<TensorOffsetValues> {
    FailureOr<TensorOffsetValues> lhsInfo =
        materializeTensorOffsetFields(lhs, builder, loc);
    FailureOr<TensorOffsetValues> rhsInfo =
        materializeTensorOffsetFields(rhs, builder, loc);
    if (failed(lhsInfo) || failed(rhsInfo) || lhsInfo->strides.size() != rank ||
        rhsInfo->strides.size() != rank)
      return fallback();
    FailureOr<TensorOffsetValues> result = makeZero(AxisKind::Structured);
    if (failed(result))
      return failure();
    for (unsigned axis = 0; axis < rank; ++axis) {
      result->axisKinds[axis] =
          lhsInfo->axisKinds[axis] == AxisKind::Structured &&
                  rhsInfo->axisKinds[axis] == AxisKind::Structured
              ? AxisKind::Structured
              : AxisKind::Opaque;
      if (result->axisKinds[axis] == AxisKind::Structured) {
        result->strides[axis] =
            subtract ? createIntegerSub(builder, loc, lhsInfo->strides[axis],
                                        rhsInfo->strides[axis])
                     : createIntegerAdd(builder, loc, lhsInfo->strides[axis],
                                        rhsInfo->strides[axis]);
        if (!result->strides[axis])
          return failure();
      }
    }
    if (!hasAnyOpaqueAxis(result->axisKinds)) {
      result->uniformOffset =
          subtract ? createIntegerSub(builder, loc, lhsInfo->uniformOffset,
                                      rhsInfo->uniformOffset)
                   : createIntegerAdd(builder, loc, lhsInfo->uniformOffset,
                                      rhsInfo->uniformOffset);
      if (!result->uniformOffset)
        return failure();
    }
    return result;
  };

  FailureOr<TensorOffsetValues> result = failure();
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    result = materializeBinary(add.getLhs(), add.getRhs(), false);
  else if (auto sub = value.getDefiningOp<arith::SubIOp>())
    result = materializeBinary(sub.getLhs(), sub.getRhs(), true);
  else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    Value affineOperand = mul.getLhs();
    FailureOr<Value> scale =
        materializeUniformIntegerScale(mul.getRhs(), builder, loc);
    if (failed(scale)) {
      affineOperand = mul.getRhs();
      scale = materializeUniformIntegerScale(mul.getLhs(), builder, loc);
    }
    if (failed(scale))
      return fallback();
    result = materializeTensorOffsetFields(affineOperand, builder, loc);
    if (succeeded(result)) {
      for (unsigned axis = 0; axis < rank; ++axis) {
        if (result->axisKinds[axis] == AxisKind::Structured)
          result->strides[axis] =
              scaleIntegerValue(builder, loc, result->strides[axis], *scale);
        if (!result->strides[axis])
          return failure();
      }
      if (!hasAnyOpaqueAxis(result->axisKinds)) {
        result->uniformOffset =
            scaleIntegerValue(builder, loc, result->uniformOffset, *scale);
        if (!result->uniformOffset)
          return failure();
      }
    }
  } else {
    return fallback();
  }

  if (failed(result))
    return failure();
  if (!hasAnyOpaqueAxis(result->axisKinds))
    return result;

  FailureOr<Value> structured = materializeStructuredContribution(
      result->strides, result->axisKinds, *tensorType, builder, loc);
  if (failed(structured))
    return failure();
  FailureOr<Value> complete =
      castIntegerLike(builder, loc, value, (*tensorType));
  if (failed(complete))
    return failure();
  Value opaque = createIntegerSub(builder, loc, *complete, *structured);
  if (!opaque)
    return failure();
  result->uniformOffset = createScalarConstant(builder, loc, scalarType, 0);
  result->opaqueContribution = opaque;
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (result->axisKinds[axis] == AxisKind::Opaque)
      result->strides[axis] = createScalarConstant(builder, loc, scalarType, 0);
    if (!result->strides[axis])
      return failure();
  }
  if (!result->uniformOffset)
    return failure();
  return result;
}

static Value materializeTensorCompleteOffsets(const DecomposedValue &value,
                                              OpBuilder &builder,
                                              Location loc) {
  if (!hasValidLayout(value))
    return nullptr;
  auto pointerTensor = cast<RankedTensorType>(value.originalType);
  unsigned rank = pointerTensor.getRank();
  if (rank == 1)
    return materializeRank1CompleteOffsets(value, builder, loc);

  auto offsetsType = cast<RankedTensorType>(
      value.components[getOpaqueContributionComponent(rank)].getType());
  SmallVector<Value> strides;
  strides.reserve(rank);
  for (unsigned axis = 0; axis < rank; ++axis)
    strides.push_back(value.components[getStrideComponent(axis)]);
  SmallVector<AxisKind> allStructured =
      getAxisKinds(rank, AxisKind::Structured);
  FailureOr<Value> structured = materializeStructuredContribution(
      strides, allStructured, offsetsType, builder, loc);
  if (failed(structured))
    return nullptr;

  Value uniform = value.components[getUniformOffsetComponent(rank)];
  if (!isConstantZero(uniform)) {
    FailureOr<Value> converted =
        castIntegerLike(builder, loc, uniform, offsetsType.getElementType());
    if (failed(converted))
      return nullptr;
    Value uniformTensor =
        builder.create<triton::SplatOp>(loc, offsetsType, *converted);
    *structured = createIntegerAdd(builder, loc, *structured, uniformTensor);
    if (!*structured)
      return nullptr;
  }

  Value opaque = value.components[getOpaqueContributionComponent(rank)];
  if (isConstantZero(opaque))
    return *structured;
  return createIntegerAdd(builder, loc, *structured, opaque);
}

static FailureOr<Value> materializeScalarBaseAddress(OpBuilder &builder,
                                                     Location loc, Value base) {
  if (!isa<triton::PointerType>(base.getType()))
    return failure();

  if (auto intToPtr = base.getDefiningOp<triton::IntToPtrOp>();
      intToPtr && intToPtr.getSrc().getType().isInteger(64))
    return intToPtr.getSrc();

  // The symbolic base identity is the scalar pointer Value, so its concrete
  // i64 address must dominate every splat of that Value. In particular, two
  // sibling if arms may each build a tensor pointer from one function argument
  // while the rebuilt result lives after the if.
  OpBuilder::InsertionGuard insertionGuard(builder);
  if (auto blockArgument = dyn_cast<BlockArgument>(base)) {
    Block *owner = blockArgument.getOwner();
    Operation *lastLeadingAddress = nullptr;
    for (Operation &operation : *owner) {
      auto existing = dyn_cast<triton::PtrToIntOp>(&operation);
      if (!existing)
        break;
      if (existing.getSrc() == base &&
          existing.getResult().getType().isInteger(64))
        return existing.getResult();
      lastLeadingAddress = &operation;
    }
    if (lastLeadingAddress)
      builder.setInsertionPointAfter(lastLeadingAddress);
    else
      builder.setInsertionPointToStart(owner);
  } else if (Operation *definingOp = base.getDefiningOp()) {
    if (Operation *next = definingOp->getNextNode()) {
      if (auto existing = dyn_cast<triton::PtrToIntOp>(next);
          existing && existing.getSrc() == base &&
          existing.getResult().getType().isInteger(64))
        return existing.getResult();
    }
    builder.setInsertionPointAfter(definingOp);
  } else {
    return failure();
  }

  return builder.create<triton::PtrToIntOp>(loc, builder.getI64Type(), base)
      .getResult();
}

// Recover the original scalar pointer only when the descriptor component is
// the direct address produced from that pointer.  This preserves native
// memref provenance for loop-invariant tensor-of-pointer bases while keeping
// the conservative int-to-ptr fallback for a base that was actually changed
// by an SCF boundary.
static Value recoverNativeScalarBase(Value baseAddress,
                                     triton::PointerType expectedType) {
  auto ptrToInt = baseAddress.getDefiningOp<triton::PtrToIntOp>();
  if (!ptrToInt || ptrToInt.getSrc().getType() != expectedType)
    return nullptr;
  return ptrToInt.getSrc();
}

static SmallVector<unsigned>
getCandidateComponents(const AnalyzedValue &value) {
  SmallVector<unsigned> indices;
  unsigned first = hasScalarBase(value) ? kBaseComponent : kBaseComponent + 1;
  for (unsigned index = first; index < value.components.size(); ++index)
    indices.push_back(index);
  return indices;
}

static SmallVector<unsigned> getOffsetComponentIndices(Type pointerType) {
  unsigned rank = cast<RankedTensorType>(pointerType).getRank();
  SmallVector<unsigned> indices;
  for (unsigned axis = 0; axis < rank; ++axis)
    indices.push_back(getStrideComponent(axis));
  indices.push_back(getUniformOffsetComponent(rank));
  indices.push_back(getOpaqueContributionComponent(rank));
  return indices;
}

static bool offsetComponentTypesDiffer(const AnalyzedValue &lhs,
                                       const AnalyzedValue &rhs) {
  for (unsigned component : getOffsetComponentIndices(lhs.originalType)) {
    if (lhs.components[component].type != rhs.components[component].type)
      return true;
  }
  return false;
}

static bool canSafelyJoinOffsetTypes(const AnalyzedValue &lhs,
                                     const AnalyzedValue &rhs) {
  if (!offsetComponentTypesDiffer(lhs, rhs))
    return true;
  unsigned rank = cast<RankedTensorType>(lhs.originalType).getRank();
  auto lhsType = dyn_cast<IntegerType>(
      lhs.components[getUniformOffsetComponent(rank)].type);
  auto rhsType = dyn_cast<IntegerType>(
      rhs.components[getUniformOffsetComponent(rank)].type);
  if (!lhsType || !rhsType || lhsType.getWidth() == rhsType.getWidth())
    return false;
  const AnalyzedValue &narrower =
      lhsType.getWidth() < rhsType.getWidth() ? lhs : rhs;
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (!isZeroIdentity(narrower.components[getStrideComponent(axis)].identity))
      return false;
  }
  return isZeroIdentity(
      narrower.components[getUniformOffsetComponent(rank)].identity);
}

static FailureOr<SmallVector<AxisKind>>
joinAxisKinds(const AnalyzedValue &lhs, const AnalyzedValue &rhs) {
  FailureOr<SmallVector<AxisKind>> lhsKinds = getAxisKinds(lhs);
  FailureOr<SmallVector<AxisKind>> rhsKinds = getAxisKinds(rhs);
  if (failed(lhsKinds) || failed(rhsKinds) ||
      lhsKinds->size() != rhsKinds->size())
    return failure();
  SmallVector<AxisKind> result;
  for (auto [lhsKind, rhsKind] : llvm::zip(*lhsKinds, *rhsKinds)) {
    result.push_back(lhsKind == AxisKind::Structured &&
                             rhsKind == AxisKind::Structured
                         ? AxisKind::Structured
                         : AxisKind::Opaque);
  }
  return result;
}

// Keep only identity-changing fields that survive normalization to the joined
// axis schema. For example, joining a Structured axis with an Opaque axis
// makes that axis stride and the shared uniform offset canonical zero values;
// carrying either value through SCF would create dead positional slots. Any
// change represented by those fields is transferred through the complete
// opaque contribution instead.
static FailureOr<SmallVector<unsigned>>
filterTransferredComponentsForJoinedAxes(
    const AnalyzedValue &value, ArrayRef<AxisKind> joinedKinds,
    ArrayRef<unsigned> identityChangingComponents) {
  if (!hasValidLayout(value))
    return failure();
  unsigned rank = cast<RankedTensorType>(value.originalType).getRank();
  if (joinedKinds.size() != rank)
    return failure();

  bool hasOpaqueAxis = hasAnyOpaqueAxis(joinedKinds);
  bool opaqueContributionChanges = false;
  SmallVector<unsigned> transferred;
  for (unsigned component : identityChangingComponents) {
    if (component >= value.components.size())
      return failure();
    if (component == kBaseComponent) {
      transferred.push_back(component);
      continue;
    }
    if (component >= getStrideComponent(0) &&
        component < getStrideComponent(rank)) {
      unsigned axis = component - getStrideComponent(0);
      if (joinedKinds[axis] == AxisKind::Structured)
        transferred.push_back(component);
      else
        opaqueContributionChanges = true;
      continue;
    }
    if (component == getUniformOffsetComponent(rank)) {
      if (!hasOpaqueAxis)
        transferred.push_back(component);
      else
        opaqueContributionChanges = true;
      continue;
    }
    if (component == getOpaqueContributionComponent(rank)) {
      if (hasOpaqueAxis)
        opaqueContributionChanges = true;
      continue;
    }
    return failure();
  }
  if (opaqueContributionChanges)
    transferred.push_back(getOpaqueContributionComponent(rank));
  llvm::sort(transferred);
  return transferred;
}

static SmallVector<Attribute>
getTensorPointerAttributes(MLIRContext *context, bool baseIsScalar,
                           ArrayRef<AxisKind> axisKinds) {
  return {BoolAttr::get(context, baseIsScalar),
          getAxisKindsAttr(context, axisKinds)};
}

static LogicalResult normalizeOffsetComponents(DecomposedValue &value,
                                               ArrayRef<AxisKind> targetKinds,
                                               OpBuilder &builder,
                                               Location loc) {
  if (!hasValidLayout(value))
    return failure();
  unsigned rank = cast<RankedTensorType>(value.originalType).getRank();
  FailureOr<SmallVector<AxisKind>> currentKinds = getAxisKinds(value);
  if (failed(currentKinds) || targetKinds.size() != rank)
    return failure();
  for (unsigned axis = 0; axis < rank; ++axis) {
    if ((*currentKinds)[axis] == AxisKind::Opaque &&
        targetKinds[axis] == AxisKind::Structured)
      return failure();
  }

  Value complete = materializeTensorCompleteOffsets(value, builder, loc);
  if (!complete)
    return failure();
  auto completeType = cast<RankedTensorType>(complete.getType());
  Type scalarType = completeType.getElementType();
  SmallVector<Value> retainedStrides;
  for (unsigned axis = 0; axis < rank; ++axis) {
    FailureOr<Value> stride = castIntegerLike(
        builder, loc, value.components[getStrideComponent(axis)], scalarType);
    if (failed(stride))
      return failure();
    if (targetKinds[axis] == AxisKind::Opaque)
      *stride = createScalarConstant(builder, loc, scalarType, 0);
    if (!*stride)
      return failure();
    retainedStrides.push_back(*stride);
    value.components[getStrideComponent(axis)] = *stride;
  }

  if (hasAnyOpaqueAxis(targetKinds)) {
    FailureOr<Value> structured = materializeStructuredContribution(
        retainedStrides, targetKinds, completeType, builder, loc);
    if (failed(structured))
      return failure();
    Value opaque = createIntegerSub(builder, loc, complete, *structured);
    Value zero = createScalarConstant(builder, loc, scalarType, 0);
    if (!opaque || !zero)
      return failure();
    value.components[getUniformOffsetComponent(rank)] = zero;
    value.components[getOpaqueContributionComponent(rank)] = opaque;
  } else {
    FailureOr<Value> uniform = castIntegerLike(
        builder, loc, value.components[getUniformOffsetComponent(rank)],
        scalarType);
    if (failed(uniform))
      return failure();
    value.components[getUniformOffsetComponent(rank)] = *uniform;
    value.components[getOpaqueContributionComponent(rank)] =
        createZeroOffsets(builder, loc, completeType);
  }
  return success();
}

class TensorPtrDecomposePolicy final : public ControlFlowRewritePolicy {
public:
  bool matches(Type type) const override { return isTensorPointerType(type); }

  FailureOr<AnalyzedValue>
  analyzeValue(Value value,
               ControlFlowAnalysisContext &context) const override {
    if (const AnalyzedValue *known = context.lookupValue(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    // Broadcasting an already-structured pointer tensor only changes the
    // descriptor shape. An expanded unit dimension repeats the same pointer,
    // so that dimension has stride zero in the result descriptor. Restrict the
    // propagation to scalar-base descriptors with no opaque contribution;
    // unknown or partially opaque pointer tensors retain the established
    // complete-pointer fallback below.
    if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
      FailureOr<AnalyzedValue> result =
          context.analyzeValue(broadcast.getSrc());
      if (succeeded(result) && hasValidLayout(*result) &&
          hasScalarBase(*result) &&
          isCompatiblePointerBroadcast(broadcast.getSrc().getType(),
                                       value.getType())) {
        FailureOr<SmallVector<AxisKind>> kinds = getAxisKinds(*result);
        auto sourceType = cast<RankedTensorType>(broadcast.getSrc().getType());
        auto resultType = cast<RankedTensorType>(value.getType());
        unsigned rank = resultType.getRank();
        if (succeeded(kinds) && !llvm::is_contained(*kinds, AxisKind::Opaque) &&
            isZeroIdentity(
                result->components[getOpaqueContributionComponent(rank)]
                    .identity)) {
          for (unsigned axis = 0; axis < rank; ++axis) {
            if (sourceType.getShape()[axis] == 1 &&
                resultType.getShape()[axis] != 1) {
              result->components[getStrideComponent(axis)].identity =
                  ComponentIdentity::zero(getStrideComponent(axis));
            }
          }
          Type scalarType =
              result->components[getUniformOffsetComponent(rank)].type;
          auto resultOffsetsType = RankedTensorType::get(
              resultType.getShape(), scalarType, resultType.getEncoding());
          result->components[getOpaqueContributionComponent(rank)] = {
              resultOffsetsType,
              ComponentIdentity::zero(getOpaqueContributionComponent(rank))};
          result->originalType = value.getType();
          result->attributes[kAxisKindsAttribute] =
              getAxisKindsAttr(value.getContext(), *kinds);
          return *result;
        }
      }
    }

    if (auto addPtr = value.getDefiningOp<triton::AddPtrOp>()) {
      FailureOr<AnalyzedValue> result = context.analyzeValue(addPtr.getPtr());
      FailureOr<AnalyzedTensorOffset> delta =
          analyzeTensorOffset(addPtr.getOffset());
      if (failed(result) || !hasValidLayout(*result) || failed(delta))
        return failure();
      FailureOr<SmallVector<AxisKind>> currentKinds = getAxisKinds(*result);
      if (failed(currentKinds) || currentKinds->size() != delta->strides.size())
        return failure();
      unsigned rank = currentKinds->size();
      FailureOr<Type> joinedResidualType = getWiderIntegerLikeType(
          result->components[getOpaqueContributionComponent(rank)].type,
          delta->opaqueContribution.type);
      if (failed(joinedResidualType))
        return failure();
      auto joinedTensorType = cast<RankedTensorType>(*joinedResidualType);
      Type scalarType = joinedTensorType.getElementType();
      SmallVector<AxisKind> resultKinds;
      for (unsigned axis = 0; axis < rank; ++axis) {
        AxisKind kind = hasScalarBase(*result) &&
                                (*currentKinds)[axis] == AxisKind::Structured &&
                                delta->axisKinds[axis] == AxisKind::Structured
                            ? AxisKind::Structured
                            : AxisKind::Opaque;
        resultKinds.push_back(kind);
        result->components[getStrideComponent(axis)] = {
            scalarType,
            kind == AxisKind::Structured
                ? getAddIdentity(
                      result->components[getStrideComponent(axis)].identity,
                      delta->strides[axis].identity, value,
                      getStrideComponent(axis))
                : ComponentIdentity::zero(getStrideComponent(axis))};
      }
      if (hasAnyOpaqueAxis(resultKinds)) {
        result->components[getUniformOffsetComponent(rank)] = {
            scalarType,
            ComponentIdentity::zero(getUniformOffsetComponent(rank))};
        result->components[getOpaqueContributionComponent(rank)] = {
            joinedTensorType, ComponentIdentity::fromValue(
                                  value, getOpaqueContributionComponent(rank))};
      } else {
        result->components[getUniformOffsetComponent(rank)] = {
            scalarType,
            getAddIdentity(
                result->components[getUniformOffsetComponent(rank)].identity,
                delta->uniformOffset.identity, value,
                getUniformOffsetComponent(rank))};
        result->components[getOpaqueContributionComponent(rank)] = {
            joinedTensorType,
            ComponentIdentity::zero(getOpaqueContributionComponent(rank))};
      }
      result->originalType = value.getType();
      result->attributes[kAxisKindsAttribute] =
          getAxisKindsAttr(value.getContext(), resultKinds);
      return *result;
    }

    if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
      if (!isa<triton::PointerType>(splat.getSrc().getType()))
        return failure();
      auto offsetsType = getDefaultOffsetsType(value.getType());
      unsigned rank = cast<RankedTensorType>(value.getType()).getRank();
      Type scalarType = offsetsType.getElementType();
      SmallVector<AnalyzedComponent> components = {
          {IntegerType::get(value.getContext(), 64),
           ComponentIdentity::fromValue(splat.getSrc(), kBaseComponent)}};
      for (unsigned axis = 0; axis < rank; ++axis)
        components.push_back(
            {scalarType, ComponentIdentity::zero(getStrideComponent(axis))});
      components.push_back({scalarType, ComponentIdentity::zero(
                                            getUniformOffsetComponent(rank))});
      components.push_back(
          {offsetsType,
           ComponentIdentity::zero(getOpaqueContributionComponent(rank))});
      return AnalyzedValue{
          value.getType(), std::move(components),
          getTensorPointerAttributes(value.getContext(), true,
                                     getAxisKinds(rank, AxisKind::Structured))};
    }

    if (!matches(value.getType()))
      return failure();
    auto offsetsType = getDefaultOffsetsType(value.getType());
    unsigned rank = cast<RankedTensorType>(value.getType()).getRank();
    Type scalarType = offsetsType.getElementType();
    SmallVector<AnalyzedComponent> components = {
        {value.getType(), ComponentIdentity::fromValue(value, kBaseComponent)}};
    for (unsigned axis = 0; axis < rank; ++axis)
      components.push_back(
          {scalarType, ComponentIdentity::zero(getStrideComponent(axis))});
    components.push_back(
        {scalarType, ComponentIdentity::zero(getUniformOffsetComponent(rank))});
    components.push_back(
        {offsetsType,
         ComponentIdentity::zero(getOpaqueContributionComponent(rank))});
    return AnalyzedValue{
        value.getType(), std::move(components),
        getTensorPointerAttributes(value.getContext(), false,
                                   getAxisKinds(rank, AxisKind::Opaque))};
  }

  FailureOr<SmallVector<unsigned>>
  getLoopCandidateComponents(const AnalyzedValue &value) const override {
    if (!hasValidLayout(value))
      return failure();
    return getCandidateComponents(value);
  }

  FailureOr<SmallVector<unsigned>>
  getLoopTransferredComponents(const AnalyzedValue &initial,
                               const AnalyzedValue &regionArgument,
                               const AnalyzedValue &next) const override {
    if (!hasSameRepresentation(initial, regionArgument) ||
        !hasSameRepresentation(initial, next) ||
        !canSafelyJoinOffsetTypes(initial, next))
      return failure();
    FailureOr<SmallVector<AxisKind>> joinedKinds = joinAxisKinds(initial, next);
    if (failed(joinedKinds))
      return failure();

    SmallVector<unsigned> identityChangingComponents;
    for (unsigned component : getCandidateComponents(initial)) {
      const ComponentIdentity &nextIdentity =
          next.components[component].identity;
      bool forwardsCurrent =
          nextIdentity == regionArgument.components[component].identity;
      bool restoresInitial =
          nextIdentity == initial.components[component].identity;
      if (!forwardsCurrent && !restoresInitial)
        identityChangingComponents.push_back(component);
    }
    FailureOr<SmallVector<unsigned>> transferred =
        filterTransferredComponentsForJoinedAxes(initial, *joinedKinds,
                                                 identityChangingComponents);
    if (failed(transferred))
      return failure();
    for (unsigned component : *transferred) {
      if (failed(joinComponentTypes(initial.components[component].type,
                                    next.components[component].type)))
        return failure();
    }
    return *transferred;
  }

  FailureOr<SmallVector<unsigned>>
  getIfTransferredComponents(const AnalyzedValue &thenValue,
                             const AnalyzedValue &elseValue) const override {
    if (!hasSameRepresentation(thenValue, elseValue) ||
        !canSafelyJoinOffsetTypes(thenValue, elseValue))
      return failure();
    FailureOr<SmallVector<AxisKind>> joinedKinds =
        joinAxisKinds(thenValue, elseValue);
    if (failed(joinedKinds))
      return failure();
    SmallVector<unsigned> identityChangingComponents;
    for (unsigned component : getCandidateComponents(thenValue)) {
      if (thenValue.components[component].identity !=
          elseValue.components[component].identity)
        identityChangingComponents.push_back(component);
    }
    FailureOr<SmallVector<unsigned>> transferred =
        filterTransferredComponentsForJoinedAxes(thenValue, *joinedKinds,
                                                 identityChangingComponents);
    if (failed(transferred))
      return failure();
    for (unsigned component : *transferred) {
      if (failed(joinComponentTypes(thenValue.components[component].type,
                                    elseValue.components[component].type)))
        return failure();
    }
    return *transferred;
  }

  FailureOr<SmallVector<Attribute>>
  mergeControlFlowAttributes(const AnalyzedValue &lhs,
                             const AnalyzedValue &rhs) const override {
    if (!hasValidLayout(lhs) || !hasValidLayout(rhs) ||
        lhs.originalType != rhs.originalType ||
        hasScalarBase(lhs) != hasScalarBase(rhs))
      return failure();
    FailureOr<SmallVector<AxisKind>> kinds = joinAxisKinds(lhs, rhs);
    if (failed(kinds))
      return failure();
    return getTensorPointerAttributes(lhs.originalType.getContext(),
                                      hasScalarBase(lhs), *kinds);
  }

  FailureOr<Type> joinComponentTypes(Type lhs, Type rhs) const override {
    return getWiderIntegerLikeType(lhs, rhs);
  }

  bool shouldDecomposeOperation(Operation *op) const override {
    return isa<triton::AddPtrOp>(op);
  }

  bool requiresPointerDescriptorBoundaryMarker() const override { return true; }

  bool shouldMarkOperationRecomposition(Operation *op) const override {
    return isa<triton::AddPtrOp>(op);
  }

  LogicalResult normalizeControlFlowValue(DecomposedValue &value,
                                          ArrayRef<Attribute> targetAttributes,
                                          OpBuilder &builder,
                                          Location loc) const override {
    if (!hasValidLayout(value) || targetAttributes.size() != 2)
      return failure();
    auto targetBase =
        dyn_cast<BoolAttr>(targetAttributes[kBaseIsScalarAttribute]);
    auto targetKinds =
        dyn_cast<DenseI32ArrayAttr>(targetAttributes[kAxisKindsAttribute]);
    if (!targetBase || !targetKinds ||
        targetBase.getValue() != hasScalarBase(value))
      return failure();
    SmallVector<AxisKind> kinds;
    for (int32_t rawKind : targetKinds.asArrayRef()) {
      if (rawKind != static_cast<int32_t>(AxisKind::Opaque) &&
          rawKind != static_cast<int32_t>(AxisKind::Structured))
        return failure();
      kinds.push_back(static_cast<AxisKind>(rawKind));
    }
    if (failed(normalizeOffsetComponents(value, kinds, builder, loc)))
      return failure();
    value.attributes.assign(targetAttributes.begin(), targetAttributes.end());
    return success();
  }

  LogicalResult annotatePointerDescriptorRebuild(
      Operation *rebuildOp, const DecomposedValue &value) const override {
    if (!hasValidLayout(value) || !isa<triton::AddPtrOp>(rebuildOp))
      return failure();
    FailureOr<SmallVector<AxisKind>> kinds = getAxisKinds(value);
    if (failed(kinds))
      return failure();
    rebuildOp->setAttr(kPointerDescriptorStructuredAxesAttr,
                       getAxisKindsAttr(rebuildOp->getContext(), *kinds));
    if (kinds->size() == 1 && kinds->front() == AxisKind::Structured &&
        hasScalarBase(value) &&
        isConstantZero(value.components[kResidualComponent])) {
      rebuildOp->setAttr(
          kPointerDescriptorOffsetFormAttr,
          StringAttr::get(rebuildOp->getContext(), kStrided1DOffsetForm));
    } else {
      rebuildOp->removeAttr(kPointerDescriptorOffsetFormAttr);
    }
    return success();
  }

  FailureOr<DecomposedValue> decompose(Value value,
                                       const ControlFlowRewriteContext &context,
                                       OpBuilder &builder,
                                       Location loc) const override {
    if (const DecomposedValue *known = context.lookup(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    value = context.remap(value);
    if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
      FailureOr<DecomposedValue> result =
          decompose(broadcast.getSrc(), context, builder, loc);
      if (succeeded(result) && hasValidLayout(*result) &&
          hasScalarBase(*result) &&
          isCompatiblePointerBroadcast(broadcast.getSrc().getType(),
                                       value.getType())) {
        FailureOr<SmallVector<AxisKind>> kinds = getAxisKinds(*result);
        auto sourceType = cast<RankedTensorType>(broadcast.getSrc().getType());
        auto resultType = cast<RankedTensorType>(value.getType());
        unsigned rank = resultType.getRank();
        if (succeeded(kinds) && !llvm::is_contained(*kinds, AxisKind::Opaque) &&
            isConstantZero(
                result->components[getOpaqueContributionComponent(rank)])) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(broadcast);
          for (unsigned axis = 0; axis < rank; ++axis) {
            if (sourceType.getShape()[axis] == 1 &&
                resultType.getShape()[axis] != 1) {
              Type strideType =
                  result->components[getStrideComponent(axis)].getType();
              result->components[getStrideComponent(axis)] =
                  createScalarConstant(builder, broadcast.getLoc(), strideType,
                                       0);
              if (!result->components[getStrideComponent(axis)])
                return failure();
            }
          }
          Type scalarType =
              result->components[getUniformOffsetComponent(rank)].getType();
          auto resultOffsetsType = RankedTensorType::get(
              resultType.getShape(), scalarType, resultType.getEncoding());
          Value zeroOffsets =
              createZeroOffsets(builder, broadcast.getLoc(), resultOffsetsType);
          if (!zeroOffsets)
            return failure();
          result->components[getOpaqueContributionComponent(rank)] =
              zeroOffsets;
          result->originalType = value.getType();
          result->attributes[kAxisKindsAttribute] =
              getAxisKindsAttr(value.getContext(), *kinds);
          return *result;
        }
      }
    }

    if (auto addPtr = value.getDefiningOp<triton::AddPtrOp>()) {
      FailureOr<DecomposedValue> result =
          decompose(addPtr.getPtr(), context, builder, loc);
      if (failed(result) || !hasValidLayout(*result))
        return failure();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(addPtr);
      Value deltaValue = context.remap(addPtr.getOffset());
      FailureOr<TensorOffsetValues> delta =
          materializeTensorOffsetFields(deltaValue, builder, addPtr.getLoc());
      FailureOr<SmallVector<AxisKind>> currentKinds = getAxisKinds(*result);
      if (failed(delta) || failed(currentKinds) ||
          delta->strides.size() != currentKinds->size())
        return failure();
      unsigned rank = currentKinds->size();
      SmallVector<AxisKind> resultKinds;
      SmallVector<Value> resultStrides;
      for (unsigned axis = 0; axis < rank; ++axis) {
        AxisKind kind = hasScalarBase(*result) &&
                                (*currentKinds)[axis] == AxisKind::Structured &&
                                delta->axisKinds[axis] == AxisKind::Structured
                            ? AxisKind::Structured
                            : AxisKind::Opaque;
        resultKinds.push_back(kind);
        Value stride =
            kind == AxisKind::Structured
                ? createIntegerAdd(builder, addPtr.getLoc(),
                                   result->components[getStrideComponent(axis)],
                                   delta->strides[axis])
                : createScalarConstant(builder, addPtr.getLoc(),
                                       delta->strides[axis].getType(), 0);
        if (!stride)
          return failure();
        resultStrides.push_back(stride);
      }

      Value currentComplete =
          materializeTensorCompleteOffsets(*result, builder, addPtr.getLoc());
      Value nextComplete = currentComplete
                               ? createIntegerAdd(builder, addPtr.getLoc(),
                                                  currentComplete, deltaValue)
                               : nullptr;
      if (!nextComplete)
        return failure();
      auto nextType = cast<RankedTensorType>(nextComplete.getType());
      Type scalarType = nextType.getElementType();
      for (unsigned axis = 0; axis < rank; ++axis) {
        FailureOr<Value> converted = castIntegerLike(
            builder, addPtr.getLoc(), resultStrides[axis], scalarType);
        if (failed(converted))
          return failure();
        result->components[getStrideComponent(axis)] = *converted;
      }
      if (hasAnyOpaqueAxis(resultKinds)) {
        SmallVector<Value> strides;
        for (unsigned axis = 0; axis < rank; ++axis)
          strides.push_back(result->components[getStrideComponent(axis)]);
        FailureOr<Value> structured = materializeStructuredContribution(
            strides, resultKinds, nextType, builder, addPtr.getLoc());
        if (failed(structured))
          return failure();
        Value opaque = createIntegerSub(builder, addPtr.getLoc(), nextComplete,
                                        *structured);
        Value zero =
            createScalarConstant(builder, addPtr.getLoc(), scalarType, 0);
        if (!opaque || !zero)
          return failure();
        result->components[getUniformOffsetComponent(rank)] = zero;
        result->components[getOpaqueContributionComponent(rank)] = opaque;
      } else {
        Value uniform = createIntegerAdd(
            builder, addPtr.getLoc(),
            result->components[getUniformOffsetComponent(rank)],
            delta->uniformOffset);
        if (!uniform)
          return failure();
        FailureOr<Value> converted =
            castIntegerLike(builder, addPtr.getLoc(), uniform, scalarType);
        if (failed(converted))
          return failure();
        result->components[getUniformOffsetComponent(rank)] = *converted;
        result->components[getOpaqueContributionComponent(rank)] =
            createZeroOffsets(builder, addPtr.getLoc(), nextType);
      }
      result->originalType = value.getType();
      result->attributes[kAxisKindsAttribute] =
          getAxisKindsAttr(value.getContext(), resultKinds);
      return *result;
    }

    if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
      if (!isa<triton::PointerType>(splat.getSrc().getType()))
        return failure();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(splat);
      FailureOr<Value> baseAddress =
          materializeScalarBaseAddress(builder, splat.getLoc(), splat.getSrc());
      if (failed(baseAddress))
        return failure();
      RankedTensorType offsetsType = getDefaultOffsetsType(value.getType());
      unsigned rank = cast<RankedTensorType>(value.getType()).getRank();
      Type scalarType = offsetsType.getElementType();
      SmallVector<Value> components{*baseAddress};
      for (unsigned axis = 0; axis < rank; ++axis)
        components.push_back(
            createScalarConstant(builder, splat.getLoc(), scalarType, 0));
      components.push_back(
          createScalarConstant(builder, splat.getLoc(), scalarType, 0));
      components.push_back(
          createZeroOffsets(builder, splat.getLoc(), offsetsType));
      if (llvm::any_of(components, [](Value component) { return !component; }))
        return failure();
      return DecomposedValue{
          value.getType(), std::move(components),
          getTensorPointerAttributes(builder.getContext(), true,
                                     getAxisKinds(rank, AxisKind::Structured))};
    }

    if (!matches(value.getType()))
      return failure();
    OpBuilder::InsertionGuard guard(builder);
    if (Operation *definingOp = value.getDefiningOp())
      builder.setInsertionPointAfter(definingOp);
    else if (auto blockArg = dyn_cast<BlockArgument>(value))
      builder.setInsertionPointToStart(blockArg.getOwner());
    RankedTensorType offsetsType = getDefaultOffsetsType(value.getType());
    unsigned rank = cast<RankedTensorType>(value.getType()).getRank();
    Type scalarType = offsetsType.getElementType();
    SmallVector<Value> components{value};
    for (unsigned axis = 0; axis < rank; ++axis)
      components.push_back(createScalarConstant(builder, loc, scalarType, 0));
    components.push_back(createScalarConstant(builder, loc, scalarType, 0));
    components.push_back(createZeroOffsets(builder, loc, offsetsType));
    if (llvm::any_of(components, [](Value component) { return !component; }))
      return failure();
    return DecomposedValue{
        value.getType(), std::move(components),
        getTensorPointerAttributes(builder.getContext(), false,
                                   getAxisKinds(rank, AxisKind::Opaque))};
  }

  Value recompose(const DecomposedValue &value, OpBuilder &builder,
                  Location loc) const override {
    if (!hasValidLayout(value))
      return nullptr;
    auto pointerTensor = cast<RankedTensorType>(value.originalType);
    Value base = value.components[kBaseComponent];
    if (hasScalarBase(value)) {
      auto scalarPointerType =
          cast<triton::PointerType>(pointerTensor.getElementType());
      Value nativeBase = recoverNativeScalarBase(base, scalarPointerType);
      if (!nativeBase)
        nativeBase =
            builder.create<triton::IntToPtrOp>(loc, scalarPointerType, base);
      base = nativeBase;
      base = builder.create<triton::SplatOp>(loc, value.originalType, base);
    }

    Value completeOffsets =
        materializeTensorCompleteOffsets(value, builder, loc);
    if (!completeOffsets)
      return nullptr;
    return builder.create<triton::AddPtrOp>(loc, value.originalType, base,
                                            completeOffsets);
  }
};

} // namespace

namespace mlir::triton::controlflow {

LogicalResult runTensorPtrDecompose(ModuleOp module) {
  TensorPtrDecomposePolicy policy;
  return rewriteControlFlow(module, policy, /*allowUnsupportedFallback=*/true);
}

} // namespace mlir::triton::controlflow
