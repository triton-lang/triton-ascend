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

#include "TritonMemoryAccess/PtrAnalysisRules/CommonPtrRules.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::ptr {
using namespace detail;
using Result = CommonPtrRules::Result;

// Local helpers build and combine facts; operation parsers follow below.
namespace {
// Uniform integer facts.
void setUniformOrigin(OffsetInfo &info, Value value, OpBuilder &builder,
                      OpFoldResult scalar = {}) {
  if (info.scalarOffset || !llvm::all_of(info.axes, [](const AxisInfo &axis) {
        return axis.kind == AxisKind::Invariant;
      }))
    return;
  // Reuse the supplied scalar; otherwise obtain a representative value.
  if (!scalar) {
    if (auto constant = getIntegerConstant(value, 64); succeeded(constant)) {
      info.scalarOffset = builder.getIndexAttr(constant->getSExtValue());
      return;
    }
    // Read the result itself without rebuilding its arithmetic.
    scalar = value;
    if (isa<RankedTensorType>(value.getType())) {
      if (llvm::any_of(info.getShape(), [](int64_t size) { return size <= 0; }))
        return;
      builder.setInsertionPointAfterValue(value);
      SmallVector<Value> indices;
      for (unsigned i = 0; i < info.getShape().size(); ++i)
        indices.push_back(
            builder.create<arith::ConstantIndexOp>(value.getLoc(), 0));
      scalar = builder.create<tensor::ExtractOp>(value.getLoc(), value, indices)
                   .getResult();
    }
  }
  if (auto source = dyn_cast<Value>(scalar))
    builder.setInsertionPointAfterValue(source);
  auto origin =
      foldOrCreateCast(scalar, builder.getIndexType(),
                       IntegerExtensionKind::Signed, builder, value.getLoc());
  if (succeeded(origin))
    info.scalarOffset = *origin;
}

OffsetInfo buildUniformOffsetInfo(Value value, OpBuilder &builder,
                                  OpFoldResult scalar = {}) {
  OffsetInfo info;
  info.valueType = value.getType();
  info.fullOffsets = value;
  info.axes.assign(info.getShape().size(),
                   AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)));
  setUniformOrigin(info, value, builder, scalar);
  return info;
}

// Build integer facts from the value itself, without analyzing its operands.
FailureOr<Result> buildIntegerInfo(Value value, OpBuilder &builder) {
  Type type = value.getType();
  // Require a supported integer/index type, including tensor element types.
  if (failed(getIntegerWidth(type)))
    return failure();

  // Scalars, singleton tensors and splat constants are uniform.
  OpFoldResult scalar;
  if (auto constant = getIntegerConstant(value, 64); succeeded(constant))
    scalar = builder.getIndexAttr(constant->getSExtValue());
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (scalar || !tensor ||
      (tensor.hasStaticShape() && tensor.getNumElements() == 1)) {
    // Scalars already supply an origin. Tensor extraction requires an active
    // memory lane or an `other` value for masked-out load results.
    Value mask;
    if (tensor) {
      if (auto load = value.getDefiningOp<triton::LoadOp>();
          load && !load.getOther())
        mask = load.getMask();
      else if (auto atomic = value.getDefiningOp<triton::AtomicRMWOp>())
        mask = atomic.getMask();
    }
    bool extractSingleton = true;
    if (mask) {
      auto constantMask = getIntegerConstant(mask, 1);
      extractSingleton = succeeded(constantMask) && constantMask->isAllOnes();
    }
    if (extractSingleton)
      return Result{{}, buildUniformOffsetInfo(value, builder, scalar)};
  }

  // Otherwise retain the full value without assuming an origin or axis strides.
  auto info = buildOffsetInfo(value, builder);
  if (failed(info))
    return failure();
  return Result{{}, std::move(*info)};
}

// Address types and complete offset values.
Type addressValueType(Type source, OpBuilder &builder) {
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    return RankedTensorType::get(tensor.getShape(), builder.getIndexType(),
                                 tensor.getEncoding());
  return builder.getIndexType();
}

Type addressStorageType(Type source, OpBuilder &builder) {
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    return tensor.clone(builder.getI64Type());
  return builder.getIndexType();
}

FailureOr<Value> materializeFullOffset(const OffsetInfo &info, Type target,
                                       OpBuilder &builder, Location loc) {
  OpFoldResult offset = info.fullOffsets;
  if (!offset && info.isUniform())
    offset = info.scalarOffset;
  if (!offset)
    return failure();
  Type source = isa<Value>(offset)
                    ? cast<Value>(offset).getType()
                    : cast<TypedAttr>(cast<Attribute>(offset)).getType();
  auto width = getIntegerWidth(source);
  auto targetWidth = getIntegerWidth(target);
  if (failed(width) || failed(targetWidth) || *width > *targetWidth)
    return failure();
  auto tensor = dyn_cast<RankedTensorType>(target);
  bool broadcast = tensor && !isa<RankedTensorType>(source);
  Type castType = broadcast ? tensor.getElementType() : target;
  auto converted = foldOrCreateCast(offset, castType,
                                    IntegerExtensionKind::Signed, builder, loc);
  if (failed(converted))
    return failure();
  auto value = materialize(*converted, castType, builder, loc);
  if (failed(value))
    return failure();
  return broadcast ? splatTo(*value, tensor, builder, loc) : value;
}

Result buildScalarPtrInfo(Value value, OpBuilder &builder) {
  if (!isa<triton::PointerType>(value.getType()) ||
      !isOrdinaryPtrType(value.getType()))
    return Result{};
  PtrInfo info;
  info.base = value;
  info.offsets.valueType = builder.getIndexType();
  info.offsets.fullOffsets = builder.getIndexAttr(0);
  info.offsets.scalarOffset = builder.getIndexAttr(0);
  return Result(std::move(info));
}

// Divisibility proofs and exact unit conversion.
// Inspect only local divisibility facts, without rebuilding scalar arithmetic
// or adding a second analysis cache. Bound the local walk; expressions beyond
// it simply need an explicit fact. Power-of-two factors survive integer wrap.
bool isKnownMultiple(OpFoldResult operand, uint64_t divisor,
                     unsigned depth = 8) {
  if (divisor == 1)
    return true;
  auto constant = getIntegerConstant(operand, 64);
  if (succeeded(constant))
    return constant->srem(llvm::APInt(64, divisor)).isZero();
  auto value = dyn_cast<Value>(operand);
  auto *op = value ? value.getDefiningOp() : nullptr;
  if (!op || !depth)
    return false;
  if (!isa<ShapedType>(value.getType()))
    if (auto hint = op->getAttrOfType<DenseIntElementsAttr>("tt.divisibility"))
      if (hint.getNumElements() == 1) {
        auto multiple = hint.getValues<llvm::APInt>()[0].getSExtValue();
        if (multiple > 0 && uint64_t(multiple) % divisor == 0)
          return true;
      }
  auto multiple = [&](Value input, uint64_t d) {
    return isKnownMultiple(input, d, depth - 1);
  };
  if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
          arith::ExtSIOp>(op))
    return multiple(op->getOperand(0), divisor);
  if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
    auto from = getIntegerWidth(cast.getIn().getType());
    auto to = getIntegerWidth(value.getType());
    if (succeeded(from) && succeeded(to) && *from <= *to)
      return multiple(cast.getIn(), divisor);
  }
  // For other integer arithmetic, a power-of-two modulus also divides the
  // source width's wrap modulus. General factors need an explicit fact above.
  auto width = getIntegerWidth(value.getType());
  if (!llvm::isPowerOf2_64(divisor) || failed(width) ||
      llvm::Log2_64(divisor) >= *width)
    return false;
  if (isa<arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
          arith::IndexCastUIOp>(op)) {
    auto from = getIntegerWidth(op->getOperand(0).getType());
    return succeeded(from) && llvm::Log2_64(divisor) < *from &&
           multiple(op->getOperand(0), divisor);
  }
  if (isa<arith::AddIOp, arith::SubIOp>(op))
    return multiple(op->getOperand(0), divisor) &&
           multiple(op->getOperand(1), divisor);
  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    for (unsigned i = 0; i < 2; ++i) {
      auto factor = getIntegerConstant(mul->getOperand(i), 64);
      uint64_t remaining = divisor;
      if (succeeded(factor))
        remaining >>=
            std::min<unsigned>(factor->countr_zero(), llvm::Log2_64(divisor));
      if (multiple(mul->getOperand(1 - i), remaining))
        return true;
    }
  }
  return false;
}

bool canDivideAxis(const AxisInfo &axis, uint64_t divisor) {
  if (!axis.subAxes.empty())
    return llvm::all_of(axis.subAxes, [&](const SubAxisInfo &sub) {
      return isKnownMultiple(sub.stride, divisor);
    });
  return axis.kind != AxisKind::Unknown &&
         isKnownMultiple(axis.stride, divisor);
}

bool canDivideDecomposition(const OffsetInfo &info, uint64_t divisor) {
  return info.hasStructuredDecomposition() &&
         isKnownMultiple(info.scalarOffset, divisor) &&
         llvm::all_of(info.axes, [&](const AxisInfo &axis) {
           return canDivideAxis(axis, divisor);
         });
}

OpFoldResult divideIndex(OpFoldResult value, uint64_t divisor,
                         OpBuilder &builder, Location loc) {
  if (divisor == 1)
    return value;
  auto constant = getIntegerConstant(value, 64);
  if (succeeded(constant))
    return builder.getIndexAttr(
        constant->sdiv(llvm::APInt(64, divisor)).getSExtValue());
  Value d = builder.create<arith::ConstantIndexOp>(loc, divisor);
  return builder.create<arith::DivSIOp>(loc, cast<Value>(value), d).getResult();
}

// The caller has proved that the complete value can be divided exactly.
// Retain only descriptor fields whose units can also be converted exactly.
void divideDecomposition(OffsetInfo &info, uint64_t divisor, OpBuilder &builder,
                         Location loc) {
  if (info.scalarOffset) {
    if (isKnownMultiple(info.scalarOffset, divisor))
      info.scalarOffset = divideIndex(info.scalarOffset, divisor, builder, loc);
    else
      info.scalarOffset = {};
  }
  for (auto &axis : info.axes) {
    if (!canDivideAxis(axis, divisor)) {
      axis = AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0));
    } else if (!axis.subAxes.empty()) {
      for (auto &sub : axis.subAxes)
        sub.stride = divideIndex(sub.stride, divisor, builder, loc);
    } else {
      axis.stride = divideIndex(axis.stride, divisor, builder, loc);
    }
  }
}

// Ordinary pointer and selection combinations.
Result combineSelectInfo(arith::SelectOp op, const Result &lhsInfo,
                         const Result &rhsInfo, OpBuilder &builder) {
  Value value = op.getResult();
  if (!lhsInfo.base) {
    const auto &lhs = lhsInfo, &rhs = rhsInfo;
    // Identical branches preserve all known offset facts.
    if (op.getTrueValue() == op.getFalseValue()) {
      auto info = lhs.offsets;
      info.fullOffsets = value;
      return Result{{}, std::move(info)};
    }
    auto info = buildOffsetInfo(value, builder);
    if (failed(info))
      return Result{};
    if (op.getCondition().getType().isInteger(1) &&
        lhs.offsets.hasStructuredForm() && rhs.offsets.hasStructuredForm()) {
      Type scalar = builder.getIndexType();
      auto choose = [&](OpFoldResult l,
                        OpFoldResult r) -> FailureOr<OpFoldResult> {
        if (l == r)
          return l;
        auto left = materialize(l, scalar, builder, value.getLoc()),
             right = materialize(r, scalar, builder, value.getLoc());
        if (failed(left) || failed(right))
          return failure();
        return OpFoldResult(builder
                                .create<arith::SelectOp>(value.getLoc(),
                                                         op.getCondition(),
                                                         *left, *right)
                                .getResult());
      };
      auto origin = choose(lhs.offsets.scalarOffset, rhs.offsets.scalarOffset);
      if (failed(origin))
        return Result{};
      info->scalarOffset = *origin;
      for (unsigned i = 0; i < info->axes.size(); ++i) {
        auto stride =
            choose(lhs.offsets.axes[i].stride, rhs.offsets.axes[i].stride);
        if (failed(stride))
          return Result{};
        info->axes[i] =
            AxisInfo(lhs.offsets.axes[i].kind == AxisKind::Invariant &&
                             rhs.offsets.axes[i].kind == AxisKind::Invariant
                         ? AxisKind::Invariant
                         : AxisKind::Structured,
                     *stride);
      }
    }
    return Result{{}, std::move(*info)};
  }
  const auto &lhs = lhsInfo, &rhs = rhsInfo;
  if (lhs.base != rhs.base)
    return Result{};
  auto target =
      cast<RankedTensorType>(addressValueType(value.getType(), builder));
  Type storage = addressStorageType(target, builder);
  auto left = materializeFullOffset(lhs.offsets, storage, builder, op.getLoc());
  auto right =
      materializeFullOffset(rhs.offsets, storage, builder, op.getLoc());
  if (failed(left) || failed(right))
    return Result{};
  PtrInfo info = lhs;
  info.offsets.fullOffsets =
      builder
          .create<arith::SelectOp>(op.getLoc(), op.getCondition(), *left,
                                   *right)
          .getResult();
  info.offsets.scalarOffset = {};
  info.offsets.axes.assign(
      target.getRank(), AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  if (lhs.offsets.isUniform() && rhs.offsets.isUniform() &&
      lhs.offsets.scalarOffset == rhs.offsets.scalarOffset) {
    info.offsets.scalarOffset = lhs.offsets.scalarOffset;
    for (auto &axis : info.offsets.axes)
      axis.kind = AxisKind::Invariant;
  }
  return Result(std::move(info));
}

FailureOr<PtrInfo> addPtrOffsets(Value value, const PtrInfo &parent,
                                 const OffsetInfo &delta, OpBuilder &builder) {
  if (parent.getElementType() !=
      cast<triton::PointerType>(detail::getElementTypeOrSelf(value.getType()))
          .getPointeeType())
    return failure();
  Type type = addressValueType(value.getType(), builder);
  OffsetInfo lhs = parent.offsets, rhs = delta;
  OpFoldResult complete;
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    Type storage = addressStorageType(tensor, builder);
    // Widen each addptr increment before summing: two address increments do
    // not have the wrapping semantics of a single narrow integer addition.
    auto left = materializeFullOffset(lhs, storage, builder, value.getLoc());
    auto right = materializeFullOffset(rhs, storage, builder, value.getLoc());
    if (failed(left) || failed(right))
      return failure();
    complete = builder.create<arith::AddIOp>(value.getLoc(), *left, *right)
                   .getResult();
    for (auto *info : {&lhs, &rhs}) {
      if (!isa<RankedTensorType>(info->valueType))
        info->axes.assign(tensor.getRank(), AxisInfo(AxisKind::Invariant,
                                                     builder.getIndexAttr(0)));
      info->valueType = type;
    }
  } else {
    auto left = materializeOffsetAt(lhs, {}, builder, value.getLoc());
    auto right = materializeOffsetAt(rhs, {}, builder, value.getLoc());
    if (failed(left) || failed(right))
      return failure();
    auto sum = foldOrCreateBinary(BinaryKind::Add, *left, *right, type, builder,
                                  value.getLoc());
    if (failed(sum))
      return failure();
    PtrInfo info = parent;
    info.offsets.valueType = type;
    info.offsets.fullOffsets = *sum;
    info.offsets.scalarOffset = *sum;
    return info;
  }
  auto offsets = addOffsets(complete, lhs, rhs, false, builder, value.getLoc());
  if (failed(offsets))
    return failure();
  PtrInfo info = parent;
  info.offsets = std::move(*offsets);
  return info;
}

// Pointer insertions replace the same positions in the complete offset value.
FailureOr<Result> insertPtrOffsets(
    Value value, const PtrInfo &source, const PtrInfo &dest, Type sourceType,
    llvm::function_ref<Value(Value, Value)> insert, OpBuilder &builder) {
  if (!source.base || source.base != dest.base || source.block || dest.block)
    return Result{};
  Type insertedType = builder.getI64Type();
  if (auto tensor = dyn_cast<RankedTensorType>(sourceType))
    insertedType = tensor.clone(builder.getI64Type());
  auto src = materializeFullOffset(source.offsets, insertedType, builder,
                                   value.getLoc());
  auto dst = materializeFullOffset(dest.offsets,
                                   addressStorageType(value.getType(), builder),
                                   builder, value.getLoc());
  if (failed(src) || failed(dst))
    return Result{};
  PtrInfo info = dest;
  info.offsets.fullOffsets = insert(*src, *dst);
  info.offsets.scalarOffset = {};
  info.offsets.axes.assign(
      info.offsets.getShape().size(),
      AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  if (source.offsets.isUniform() && dest.offsets.isUniform() &&
      source.offsets.scalarOffset == dest.offsets.scalarOffset) {
    info.offsets.scalarOffset = dest.offsets.scalarOffset;
    for (auto &axis : info.offsets.axes)
      axis.kind = AxisKind::Invariant;
  }
  return info;
}

// Block descriptor construction and coordinate updates.
FailureOr<OpFoldResult> addScaledCoordinates(OpFoldResult origin,
                                             ArrayRef<OpFoldResult> coordinates,
                                             ArrayRef<OpFoldResult> strides,
                                             OpBuilder &builder, Location loc) {
  detail::ConcreteArithmetic arithmetic{builder.getIndexType(), builder, loc};
  return addScaledCoordinateTerms(origin, coordinates, strides, arithmetic);
}

FailureOr<Result> makeBlockInfo(triton::MakeTensorPtrOp op, const PtrInfo &base,
                                ArrayRef<OpFoldResult> coordinates,
                                OpBuilder &builder) {
  PtrInfo ptr = base;
  auto baseOffset = materializeOffsetAt(base.offsets, {}, builder, op.getLoc());
  if (failed(baseOffset))
    return failure();
  auto tensorType = cast<RankedTensorType>(
      cast<triton::PointerType>(op.getType()).getPointeeType());
  auto &offsets = ptr.offsets;
  offsets.valueType = RankedTensorType::get(
      tensorType.getShape(), builder.getIndexType(), tensorType.getEncoding());
  offsets.fullOffsets = {};
  offsets.axes.clear();
  for (Value stride : op.getStrides()) {
    auto constant = detail::getIntegerConstant(stride, 64);
    auto indexStride =
        foldOrCreateCast(stride, builder.getIndexType(),
                         IntegerExtensionKind::Signed, builder, op.getLoc());
    if (failed(indexStride))
      return Result{};
    offsets.axes.emplace_back(succeeded(constant) && constant->isZero()
                                  ? AxisKind::Invariant
                                  : AxisKind::Structured,
                              *indexStride);
  }
  auto origin =
      addScaledCoordinates(*baseOffset, coordinates,
                           detail::axisStrides(offsets), builder, op.getLoc());
  if (failed(origin))
    return Result{};
  offsets.scalarOffset = *origin;
  ptr.block.emplace();
  for (Value shape : op.getShape()) {
    auto indexShape =
        foldOrCreateCast(shape, builder.getIndexType(),
                         IntegerExtensionKind::Signed, builder, op.getLoc());
    if (failed(indexShape))
      return Result{};
    ptr.block->parentShape.push_back(*indexShape);
  }
  ptr.block->coordinateOffsets.assign(coordinates.begin(), coordinates.end());
  ptr.block->order = llvm::to_vector(op.getOrderAttr().asArrayRef());
  if (!detail::validPtrBinding(op.getResult(), ptr))
    return failure();
  return Result(std::move(ptr));
}

FailureOr<Result> advanceBlockInfo(triton::AdvanceOp op, const PtrInfo &parent,
                                   ArrayRef<OpFoldResult> delta,
                                   OpBuilder &builder) {
  PtrInfo ptr = parent;
  detail::ConcreteArithmetic arithmetic{builder.getIndexType(), builder,
                                        op.getLoc()};
  auto updated = advanceCoordinateTerms<OpFoldResult>(
      ptr.block->coordinateOffsets, delta, arithmetic);
  if (failed(updated))
    return Result{};
  // Add only this advance, retaining the existing scalar base contribution.
  auto origin = addScaledCoordinates(ptr.offsets.scalarOffset, delta,
                                     detail::axisStrides(ptr.offsets), builder,
                                     op.getLoc());
  if (failed(origin))
    return Result{};
  ptr.offsets.scalarOffset = *origin;
  ptr.offsets.fullOffsets = {};
  ptr.block->coordinateOffsets = std::move(*updated);
  if (!detail::validPtrBinding(op.getResult(), ptr))
    return failure();
  return Result(std::move(ptr));
}

// Invariant-axis facts through arithmetic and reductions.
// Uniformity survives deterministic scalar arithmetic, including a reduction
// combiner. It does not imply that floating arithmetic has an integer stride.
bool isScalarArithmetic(Operation &op) {
  return isa<arith::ConstantOp>(op) ||
         (op.getName().getDialectNamespace() == "arith" &&
          op.hasTrait<OpTrait::Elementwise>() && op.getNumRegions() == 0);
}

bool hasArithmeticCombiner(triton::ReduceOp op) {
  return llvm::all_of(op.getCombineOp().front().without_terminator(),
                      isScalarArithmetic);
}

// Float intermediates need only per-axis equality facts. Integer inputs still
// go through the caller's Rules. This local memo avoids rewalking shared float
// expressions; it owns no SSA helpers and has no session lifecycle.
FailureOr<SmallVector<bool>>
inferInvariantAxes(Value value, CommonPtrRules::Analyze analyze,
                   llvm::DenseMap<Value, SmallVector<bool>> &known) {
  if (auto it = known.find(value); it != known.end())
    return it->second;
  auto type = dyn_cast<RankedTensorType>(value.getType());
  SmallVector<bool> axes(type ? type.getRank() : 0, false);
  if (!type)
    return axes;
  auto remember = [&]() -> FailureOr<SmallVector<bool>> {
    known.try_emplace(value, axes);
    return axes;
  };
  if (!isa<FloatType>(type.getElementType())) {
    auto info = analyze(value);
    if (failed(info))
      return failure();
    if (!info->isEmpty())
      for (unsigned i = 0; i < axes.size(); ++i)
        axes[i] = info->offsets.axes[i].kind == AxisKind::Invariant;
    return remember();
  }
  if (value.getDefiningOp<triton::SplatOp>()) {
    std::fill(axes.begin(), axes.end(), true);
  } else if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue()))
      if (dense.isSplat())
        std::fill(axes.begin(), axes.end(), true);
  } else if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>()) {
    auto source = inferInvariantAxes(broadcast.getSrc(), analyze, known);
    if (failed(source))
      return failure();
    auto shape =
        cast<RankedTensorType>(broadcast.getSrc().getType()).getShape();
    for (unsigned i = 0; i < axes.size(); ++i)
      axes[i] = shape[i] == 1 || (*source)[i];
  } else if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>()) {
    auto source = inferInvariantAxes(expand.getSrc(), analyze, known);
    if (failed(source))
      return failure();
    axes = std::move(*source);
    axes.insert(axes.begin() + expand.getAxis(), true);
  } else if (Operation *op = value.getDefiningOp();
             op && (isScalarArithmetic(*op) || isa<triton::ClampFOp>(op))) {
    std::fill(axes.begin(), axes.end(), true);
    for (Value operand : op->getOperands()) {
      auto source = inferInvariantAxes(operand, analyze, known);
      if (failed(source))
        return failure();
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      if (source->size() != axes.size())
        return SmallVector<bool>(axes.size(), false);
      for (unsigned i = 0; i < axes.size(); ++i)
        axes[i] = axes[i] && (*source)[i];
    }
  }
  return remember();
}

} // namespace

// Elementwise combinations.
CommonPtrRules::Result
CommonPtrRules::combineElementwise(Value value, const PtrInfo &lhs,
                                   const PtrInfo &rhs,
                                   OpBuilder &builder) const {
  const auto *left = &lhs, *right = &rhs;
  auto offsets = buildOffsetInfo(value, builder);
  if (failed(offsets) || left->offsets.getShape() != offsets->getShape() ||
      right->offsets.getShape() != offsets->getShape())
    return Result{};
  Type scalar = builder.getIndexType();
  for (unsigned i = 0; i < offsets->axes.size(); ++i) {
    bool invariant = left->offsets.axes[i].kind == AxisKind::Invariant &&
                     right->offsets.axes[i].kind == AxisKind::Invariant;
    offsets->axes[i] =
        AxisInfo(invariant ? AxisKind::Invariant : AxisKind::Unknown,
                 builder.getIntegerAttr(scalar, 0));
  }
  setUniformOrigin(*offsets, value, builder);
  return Result{{}, std::move(*offsets)};
}

CommonPtrRules::Result
CommonPtrRules::refineDivRem(Value value, const Result &lhsInfo,
                             const Result &rhsInfo, const Result &fallback,
                             bool divide, OpBuilder &builder) const {
  if (!rhsInfo.offsets.isUniform())
    return fallback;
  auto width = detail::getIntegerWidth(lhsInfo.offsets.valueType);
  if (failed(width))
    return fallback;
  auto constant = detail::getIntegerConstant(rhsInfo.offsets.scalarOffset, 64);
  if (failed(constant) || constant->isZero() || constant->isNegative())
    return fallback;
  auto divisor = *constant;
  uint64_t d = divisor.getZExtValue();
  auto bounds = detail::constantDecompositionRange(lhsInfo.offsets, *width);
  bool nonnegative = bounds && bounds->isAllNonNegative();
  bool signedOp = isa<arith::DivSIOp, arith::RemSIOp>(value.getDefiningOp());
  // Exact division needs no residual splitting: negative or dynamic origins
  // work too. Unsigned division can reuse signed descriptors only above zero.
  if ((signedOp || nonnegative) && canDivideDecomposition(lhsInfo.offsets, d)) {
    OffsetInfo info = lhsInfo.offsets;
    info.fullOffsets = value;
    if (divide) {
      divideDecomposition(info, d, builder, value.getLoc());
    } else {
      info.scalarOffset = builder.getIndexAttr(0);
      info.axes.assign(info.getShape().size(),
                       AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)));
    }
    if (succeeded(detail::normalizeAxes(info, builder)))
      return Result{{}, std::move(info)};
    return fallback;
  }
  // Splitting residual digits still needs a nonnegative constant
  // decomposition; otherwise signed division may change across zero.
  if (!nonnegative)
    return fallback;
  auto origin = detail::getIntegerConstant(lhsInfo.offsets.scalarOffset, 64);
  if (failed(origin) || !origin->urem(divisor).isZero())
    return fallback;
  OffsetInfo info = lhsInfo.offsets;
  info.fullOffsets = value;
  Type scalar = builder.getIndexType();
  info.scalarOffset = builder.getIntegerAttr(
      scalar, divide ? origin->udiv(divisor) : llvm::APInt(64, 0));
  __int128 residualMaximum = 0;
  for (unsigned i = 0; i < info.axes.size(); ++i) {
    auto &axis = info.axes[i];
    SmallVector<SubAxisInfo> input = axis.subAxes;
    if (input.empty())
      input.push_back({info.getShape()[i], axis.stride});
    SmallVector<SubAxisInfo> output;
    for (const auto &sub : input) {
      auto stride = detail::getIntegerConstant(sub.stride, 64);
      if (sub.shape <= 0 || failed(stride) || stride->isNegative())
        return fallback;
      uint64_t size = sub.shape, c = stride->getZExtValue();
      auto push = [&](uint64_t length, uint64_t coefficient) {
        output.push_back(
            {static_cast<int64_t>(length),
             builder.getIntegerAttr(scalar, llvm::APInt(64, coefficient))});
      };
      if (c % d == 0) {
        push(size, divide ? c / d : 0);
      } else if (d % c == 0 && size % (d / c) == 0) {
        uint64_t inner = d / c;
        push(size / inner, divide ? 1 : 0);
        push(inner, divide ? 0 : c);
        residualMaximum += d - c;
      } else if (__int128(size - 1) * c < d) {
        push(size, divide ? 0 : c);
        residualMaximum += __int128(size - 1) * c;
      } else {
        return fallback;
      }
    }
    axis.subAxes = std::move(output);
  }
  // Multiple residual digits may carry into the quotient. Only accept their
  // independent decomposition when their total stays below the divisor.
  if (residualMaximum >= d || failed(detail::normalizeAxes(info, builder)))
    return fallback;
  return Result{{}, std::move(info)};
}

// Rule entry points.
LogicalResult CommonPtrRules::validateBinding(Value value,
                                              const Result &info) const {
  if (!value || info.isEmpty())
    return failure();
  if (info.base)
    return success(validPtrBinding(value, info));
  return success(!info.block && validOffsetBinding(value, info.offsets));
}

FailureOr<Result> CommonPtrRules::parseOperation(Value value, Analyze analyze,
                                                 OpBuilder &builder) const {
  // Validate the supported integer and pointer types before dispatching.
  if (!value)
    return failure();
  bool integer = succeeded(detail::getIntegerWidth(value.getType()));
  bool ordinary = detail::isOrdinaryPtrType(value.getType());
  bool block = detail::isBlockPtrType(value.getType());
  if (!integer && !ordinary && !block) {
    emitError(value.getLoc())
        << "unsupported type passed to pointer analysis: " << value.getType();
    return failure();
  }
  // Analyze splat sources even for singleton results, preserving bound origins.
  if (auto op = value.getDefiningOp<triton::SplatOp>())
    return parseSplat(op, analyze, builder);
  // A scalar or singleton tensor is already an exact dynamic origin/stride.
  // Reuse its final SSA instead of rebuilding its arithmetic in index;
  // larger tensors need axis analysis below.
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (integer &&
      (isa<IntegerType, IndexType>(value.getType()) ||
       (tensor && tensor.hasStaticShape() && tensor.getNumElements() == 1))) {
    builder.setInsertionPointAfterValue(value);
    return buildIntegerInfo(value, builder);
  }
  // Block pointers derive coordinates and strides from their descriptors.
  if (block) {
    if (auto op = value.getDefiningOp<triton::MakeTensorPtrOp>())
      return parseMakeTensorPtr(op, analyze, builder);
    if (auto op = value.getDefiningOp<triton::AdvanceOp>())
      return parseAdvance(op, analyze, builder);
    return Result{};
  }
  // Shape and selection rules apply to both offsets and ordinary pointers.
  if (auto op = value.getDefiningOp<triton::BroadcastOp>())
    return parseBroadcast(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::ExpandDimsOp>())
    return parseExpandDims(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::ReshapeOp>())
    return parseReshape(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::ExtractSliceOp>())
    return parseExtractSlice(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::InsertOp>())
    return parseInsert(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::InsertSliceOp>())
    return parseInsertSlice(op, analyze, builder);
  if (auto op = value.getDefiningOp<tensor::CastOp>())
    return parseTensorCast(op, analyze, builder);
  if (auto op = value.getDefiningOp<arith::SelectOp>())
    return parseSelect(op, analyze, builder);
  // Ordinary pointers combine a scalar base with element offsets.
  if (ordinary) {
    if (auto op = value.getDefiningOp<triton::AddPtrOp>())
      return parseAddPtr(op, analyze, builder);
    if (auto op = value.getDefiningOp<triton::BitcastOp>())
      return parseBitcast(op, analyze, builder);
    return buildScalarPtrInfo(value, builder);
  }
  // Integer tensor rules infer origins and relationships between lanes.
  if (auto op = value.getDefiningOp<triton::MakeRangeOp>()) {
    return parseMakeRange(op, builder);
  }
  if (auto op = value.getDefiningOp<arith::AddIOp>())
    return parseAdd(op, analyze, builder);
  if (auto op = value.getDefiningOp<arith::SubIOp>())
    return parseSub(op, analyze, builder);
  if (auto op = value.getDefiningOp<arith::MulIOp>())
    return parseMul(op, analyze, builder);
  if (auto *op = value.getDefiningOp())
    if (isa<arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
            arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
            arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::CmpIOp>(op))
      return parseElementwise(value, analyze, builder);
  if (auto op = value.getDefiningOp<arith::ExtSIOp>())
    return parseCast(value, op.getIn(), false, analyze, builder);
  if (auto op = value.getDefiningOp<arith::ExtUIOp>())
    return parseCast(value, op.getIn(), true, analyze, builder);
  if (auto op = value.getDefiningOp<arith::TruncIOp>())
    return parseCast(value, op.getIn(), false, analyze, builder);
  if (auto op = value.getDefiningOp<arith::IndexCastOp>())
    return parseCast(value, op.getIn(), false, analyze, builder);
  if (auto op = value.getDefiningOp<arith::IndexCastUIOp>())
    return parseCast(value, op.getIn(), true, analyze, builder);
  if (auto op = value.getDefiningOp<arith::FPToSIOp>())
    return parseFloatToInt(op, analyze, builder);
  if (auto op = value.getDefiningOp<triton::ReduceOp>())
    return parseReduce(value, op, analyze, builder);
  // Other integer producers may still provide facts through the value itself.
  builder.setInsertionPointAfterValue(value);
  return buildIntegerInfo(value, builder);
}

// Shape and selection rules.
FailureOr<Result> CommonPtrRules::parseSplat(triton::SplatOp op,
                                             Analyze analyze,
                                             OpBuilder &builder) const {
  // Analyze the scalar source so its bindings and origin are retained.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  if (sourceInfo->isEmpty())
    return Result{};
  builder.setInsertionPointAfter(op);

  // Integer splats use the source origin for every result lane.
  if (!sourceInfo->base) {
    if (!sourceInfo->offsets.getShape().empty())
      return Result{};
    OpFoldResult scalar = sourceInfo->offsets.scalarOffset;
    if (!scalar)
      scalar = op.getSrc();
    return Result{{}, buildUniformOffsetInfo(op.getResult(), builder, scalar)};
  }

  // Pointer splats broadcast the element offset and retain the scalar base.
  builder.setInsertionPoint(op);
  PtrInfo info = *sourceInfo;
  if (isa<RankedTensorType>(info.offsets.valueType))
    return Result{};
  auto target = cast<RankedTensorType>(addressValueType(op.getType(), builder));
  auto full = materializeFullOffset(
      info.offsets, addressStorageType(target, builder), builder, op.getLoc());
  if (failed(full))
    return Result{};

  // Every new tensor axis is invariant because all pointers came from one SSA.
  info.offsets.valueType = target;
  info.offsets.fullOffsets = *full;
  info.offsets.axes.assign(
      target.getRank(), AxisInfo(AxisKind::Invariant, builder.getIndexAttr(0)));
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseBroadcast(triton::BroadcastOp op,
                                                 Analyze analyze,
                                                 OpBuilder &builder) const {
  // Obtain the source facts before expanding its singleton dimensions.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  if (sourceInfo->isEmpty())
    return Result{};
  builder.setInsertionPointAfter(op);

  // Integer results already provide the complete value; broadcast their axes.
  if (!sourceInfo->base) {
    auto info = broadcastOffsets(op.getResult(), op.getType(),
                                 sourceInfo->offsets, builder);
    if (failed(info))
      return Result{};
    return Result{{}, std::move(*info)};
  }

  // Pointer results need the same broadcast applied to their complete offsets.
  PtrInfo info = *sourceInfo;
  builder.setInsertionPoint(op);
  auto target = cast<RankedTensorType>(addressValueType(op.getType(), builder));
  auto inputOffset = materializeFullOffset(
      info.offsets, addressStorageType(info.offsets.valueType, builder),
      builder, op.getLoc());
  if (failed(inputOffset))
    return Result{};
  Value full = builder.create<triton::BroadcastOp>(
      op.getLoc(), addressStorageType(target, builder), *inputOffset);

  // Keep the descriptor's axis facts consistent with the broadcast value.
  auto offsets = broadcastOffsets(full, target, info.offsets, builder);
  if (failed(offsets))
    return Result{};
  info.offsets = std::move(*offsets);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseExpandDims(triton::ExpandDimsOp op,
                                                  Analyze analyze,
                                                  OpBuilder &builder) const {
  // Obtain the source facts before inserting the singleton dimension.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  if (sourceInfo->isEmpty())
    return Result{};
  builder.setInsertionPointAfter(op);

  // Integer offsets gain an invariant axis at the requested position.
  if (!sourceInfo->base) {
    auto info = expandDimsOffsets(op.getResult(), op.getType(),
                                  sourceInfo->offsets, op.getAxis(), builder);
    if (failed(info))
      return Result{};
    return Result{{}, std::move(*info)};
  }

  // Insert the same dimension into a pointer tensor's complete offsets.
  PtrInfo info = *sourceInfo;
  builder.setInsertionPoint(op);
  auto target = cast<RankedTensorType>(addressValueType(op.getType(), builder));
  auto inputOffset = materializeFullOffset(
      info.offsets, addressStorageType(info.offsets.valueType, builder),
      builder, op.getLoc());
  if (failed(inputOffset))
    return Result{};
  Value full = builder.create<triton::ExpandDimsOp>(
      op.getLoc(), addressStorageType(target, builder), *inputOffset,
      op.getAxis());

  // Insert the corresponding invariant axis into the pointer descriptor.
  auto offsets =
      expandDimsOffsets(full, target, info.offsets, op.getAxis(), builder);
  if (failed(offsets))
    return Result{};
  info.offsets = std::move(*offsets);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseReshape(triton::ReshapeOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  // allow_reorder has no fixed lane correspondence. Normalization is a caller
  // decision; analysis must not change the source op or choose two
  // permutations.
  if (op.getAllowReorder())
    return Result{};

  // Obtain the source facts under the fixed lane ordering.
  auto input = analyze(op.getSrc());
  if (failed(input))
    return failure();
  if (input->isEmpty())
    return Result{};
  builder.setInsertionPointAfter(op);

  // Integer results reuse their SSA and repartition the known axis facts.
  if (!input->base) {
    auto offsets = reshapeOffsets(op.getResult(), op.getType(), input->offsets,
                                  builder, op.getLoc());
    if (failed(offsets))
      return Result{};
    return Result{{}, std::move(*offsets)};
  }

  // Pointer tensors reshape the complete offset value while keeping the base.
  PtrInfo info = *input;
  builder.setInsertionPoint(op);
  auto type = cast<RankedTensorType>(addressValueType(op.getType(), builder));
  auto inputOffset = materializeFullOffset(
      info.offsets, addressStorageType(info.offsets.valueType, builder),
      builder, op.getLoc());
  if (failed(inputOffset))
    return Result{};
  Value full = builder.create<triton::ReshapeOp>(
      op.getLoc(), addressStorageType(type, builder), *inputOffset, false,
      op.getEfficientLayout());

  // Repartition the descriptor to match the reshaped offset tensor.
  auto offsets = reshapeOffsets(full, type, info.offsets, builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  info.offsets = std::move(*offsets);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseExtractSlice(tensor::ExtractSliceOp op,
                                                    Analyze analyze,
                                                    OpBuilder &builder) const {
  // Require a static result shape and available source facts.
  auto target = op.getType();
  if (!target.hasStaticShape())
    return Result{};
  auto input = analyze(op.getSource());
  if (failed(input))
    return failure();
  if (input->isEmpty())
    return Result{};
  builder.setInsertionPointAfter(op);

  // Record which source dimensions disappear in a rank-reducing slice.
  auto mask = op.getDroppedDims();
  SmallVector<bool> dropped;
  for (unsigned i = 0; i < op.getSourceType().getRank(); ++i)
    dropped.push_back(mask.test(i));

  // Integer slices update the origin and axes using starts, steps and drops.
  if (!input->base) {
    auto offsets = sliceOffsets(op.getResult(), target, input->offsets,
                                op.getMixedOffsets(), op.getMixedStrides(),
                                dropped, builder, op.getLoc());
    if (failed(offsets))
      return Result{};
    return Result{{}, std::move(*offsets)};
  }

  // Pointer slices apply the identical slice to their complete offset tensor.
  builder.setInsertionPoint(op);
  PtrInfo info = *input;
  auto type = cast<RankedTensorType>(addressValueType(target, builder));
  auto inputOffset = materializeFullOffset(
      info.offsets, addressStorageType(info.offsets.valueType, builder),
      builder, op.getLoc());
  if (failed(inputOffset))
    return Result{};
  Value full = builder.create<tensor::ExtractSliceOp>(
      op.getLoc(), cast<RankedTensorType>(addressStorageType(type, builder)),
      *inputOffset, op.getMixedOffsets(), op.getMixedSizes(),
      op.getMixedStrides());

  // Apply that same coordinate transformation to the descriptor.
  auto offsets =
      sliceOffsets(full, type, info.offsets, op.getMixedOffsets(),
                   op.getMixedStrides(), dropped, builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  info.offsets = std::move(*offsets);
  return Result(std::move(info));
}

FailureOr<Result> CommonPtrRules::parseInsert(tensor::InsertOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  // Integer insertion keeps facts obtainable from the result value itself.
  if (!isOrdinaryPtrType(op.getType()))
    return buildIntegerInfo(op.getResult(), builder);

  // Analyze both pointer inputs for a statically shaped destination.
  if (!op.getType().hasStaticShape())
    return Result{};
  auto source = analyze(op.getScalar()), dest = analyze(op.getDest());
  if (failed(source) || failed(dest))
    return failure();

  // Insert the source offset into the destination offsets under a common base.
  builder.setInsertionPoint(op);
  return insertPtrOffsets(
      op.getResult(), *source, *dest, op.getScalar().getType(),
      [&](Value src, Value dst) -> Value {
        return builder.create<tensor::InsertOp>(op.getLoc(), src, dst,
                                                op.getIndices());
      },
      builder);
}

FailureOr<Result> CommonPtrRules::parseInsertSlice(tensor::InsertSliceOp op,
                                                   Analyze analyze,
                                                   OpBuilder &builder) const {
  // Integer insertion keeps facts obtainable from the result value itself.
  if (!isOrdinaryPtrType(op.getType()))
    return buildIntegerInfo(op.getResult(), builder);

  // Analyze both pointer tensors after checking their static shapes.
  if (!op.getType().hasStaticShape() || !op.getSourceType().hasStaticShape())
    return Result{};
  auto source = analyze(op.getSource()), dest = analyze(op.getDest());
  if (failed(source) || failed(dest))
    return failure();

  // Replace the same slice in the complete offsets under a common base.
  builder.setInsertionPoint(op);
  return insertPtrOffsets(
      op.getResult(), *source, *dest, op.getSourceType(),
      [&](Value src, Value dst) -> Value {
        return builder.create<tensor::InsertSliceOp>(
            op.getLoc(), src, dst, op.getMixedOffsets(), op.getMixedSizes(),
            op.getMixedStrides());
      },
      builder);
}

FailureOr<Result> CommonPtrRules::parseTensorCast(tensor::CastOp op,
                                                  Analyze analyze,
                                                  OpBuilder &builder) const {
  // Only identical static tensor types preserve the current shape facts.
  auto from = dyn_cast<RankedTensorType>(op.getSource().getType());
  auto to = dyn_cast<RankedTensorType>(op.getType());
  if (!from || !to || !from.hasStaticShape() || from != to)
    return Result{};

  // Reuse the source facts and update the complete integer value.
  auto input = analyze(op.getSource());
  if (failed(input))
    return failure();
  builder.setInsertionPointAfter(op);
  if (!input->isEmpty() && !input->base)
    input->offsets.fullOffsets = op.getResult();
  // Identical pointer tensor types need no extra helper: offsets already have
  // the same shape/encoding, and the cast preserves every lane.
  return *input;
}

FailureOr<Result> CommonPtrRules::parseSelect(arith::SelectOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  // A scalar pointer selection can serve directly as the resulting base.
  if (isa<triton::PointerType>(op.getType()))
    return buildScalarPtrInfo(op.getResult(), builder);

  // Obtain both branch results before combining their offset facts.
  auto lhsInfo = analyze(op.getTrueValue());
  auto rhsInfo = analyze(op.getFalseValue());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  if (lhsInfo->isEmpty() || rhsInfo->isEmpty())
    return Result{};

  // Combine branch facts; return pointer or fully structured results directly.
  if (!lhsInfo->base)
    builder.setInsertionPointAfter(op);
  else
    builder.setInsertionPoint(op);
  auto result = combineSelectInfo(op, *lhsInfo, *rhsInfo, builder);
  if (result.isEmpty() || result.base || result.offsets.hasStructuredForm())
    return result;

  // Otherwise use condition invariance to refine the integer result.
  auto condition = analyze(op.getCondition());
  if (failed(condition))
    return failure();
  if (condition->isEmpty())
    return result;
  // A tensor condition can be uniform on some axes too. Keep only equality
  // facts shared by all three inputs, then extract the original result if
  // every axis is invariant.
  for (unsigned i = 0; i < result.offsets.axes.size(); ++i) {
    bool uniformCondition =
        condition->offsets.getShape().empty() ||
        condition->offsets.axes[i].kind == AxisKind::Invariant;
    if (uniformCondition &&
        lhsInfo->offsets.axes[i].kind == AxisKind::Invariant &&
        rhsInfo->offsets.axes[i].kind == AxisKind::Invariant)
      result.offsets.axes[i].kind = AxisKind::Invariant;
  }

  // Recover a scalar origin when every result axis is invariant.
  setUniformOrigin(result.offsets, op.getResult(), builder);
  return result;
}

// Integer rules.
Result CommonPtrRules::parseMakeRange(triton::MakeRangeOp op,
                                      OpBuilder &builder) const {
  // Require a nonempty one-dimensional i32 range.
  auto width = getIntegerWidth(op.getType());
  if (failed(width) || *width != 32)
    return Result{};
  auto info = buildOffsetInfo(op.getResult(), builder);
  if (failed(info) || info->getShape().size() != 1 || info->getShape()[0] <= 0)
    return Result{};

  // Check that the signed bounds match the result extent.
  int64_t start = op.getStartAttr().getInt();
  int64_t end = op.getEndAttr().getInt();
  if (end <= start || end - start != info->getShape()[0])
    return Result{};

  // Represent the range as its starting offset plus a unit-stride axis.
  info->axes = {AxisInfo(AxisKind::Structured, builder.getIndexAttr(1))};
  info->scalarOffset = builder.getIndexAttr(start);
  return Result{{}, std::move(*info)};
}

FailureOr<Result> CommonPtrRules::parseAdd(arith::AddIOp op, Analyze analyze,
                                           OpBuilder &builder) const {
  // Obtain both operands' origins and axis facts through the current session.
  auto lhsInfo = analyze(op.getLhs());
  auto rhsInfo = analyze(op.getRhs());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  if (lhsInfo->isEmpty() || rhsInfo->isEmpty())
    return Result{};
  const auto &lhs = *lhsInfo, &rhs = *rhsInfo;
  builder.setInsertionPointAfter(op);

  // Add the origins and corresponding axis contributions.
  // Preserve the legacy no-overflow assumption for index descriptors. Dynamic
  // origins and strides do not need a separate propagated range proof.
  auto offsets = addOffsets(op.getResult(), lhs.offsets, rhs.offsets, false,
                            builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  return Result{{}, std::move(*offsets)};
}

FailureOr<Result> CommonPtrRules::parseSub(arith::SubIOp op, Analyze analyze,
                                           OpBuilder &builder) const {
  // Obtain both operands' origins and axis facts through the current session.
  auto lhsInfo = analyze(op.getLhs());
  auto rhsInfo = analyze(op.getRhs());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  if (lhsInfo->isEmpty() || rhsInfo->isEmpty())
    return Result{};
  const auto &lhs = *lhsInfo, &rhs = *rhsInfo;
  builder.setInsertionPointAfter(op);

  // Subtract the origins and corresponding axis contributions.
  auto offsets = addOffsets(op.getResult(), lhs.offsets, rhs.offsets, true,
                            builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  return Result{{}, std::move(*offsets)};
}

FailureOr<Result> CommonPtrRules::parseMul(arith::MulIOp op, Analyze analyze,
                                           OpBuilder &builder) const {
  // Obtain both operands' origins and axis facts through the current session.
  auto lhsInfo = analyze(op.getLhs());
  auto rhsInfo = analyze(op.getRhs());
  if (failed(lhsInfo) || failed(rhsInfo))
    return failure();
  if (lhsInfo->isEmpty() || rhsInfo->isEmpty())
    return Result{};
  const auto &lhs = *lhsInfo, &rhs = *rhsInfo;
  builder.setInsertionPointAfter(op);

  // A uniform operand can scale a decomposition; otherwise keep invariance.
  bool uniform = lhs.offsets.isUniform() || rhs.offsets.isUniform();
  if (!uniform)
    return combineElementwise(op, lhs, rhs, builder);

  // Use the uniform origin to scale the other operand's descriptor.
  const auto &varying = lhs.offsets.isUniform() ? rhs.offsets : lhs.offsets;
  const auto &scale = lhs.offsets.isUniform() ? lhs.offsets : rhs.offsets;
  auto offsets = scaleOffsets(op.getResult(), varying, scale.scalarOffset,
                              builder, op.getLoc());
  if (failed(offsets))
    return Result{};
  return Result{{}, std::move(*offsets)};
}

FailureOr<Result> CommonPtrRules::parseElementwise(Value value, Analyze analyze,
                                                   OpBuilder &builder) const {
  // Analyze both operands before propagating their shared axis facts.
  Operation *op = value.getDefiningOp();
  auto lhs = analyze(op->getOperand(0));
  auto rhs = analyze(op->getOperand(1));
  if (failed(lhs) || failed(rhs))
    return failure();
  if (lhs->isEmpty() || rhs->isEmpty())
    return Result{};

  // Start with axis invariance shared by both operands.
  builder.setInsertionPointAfterValue(value);
  auto fallback = combineElementwise(value, *lhs, *rhs, builder);
  if (fallback.isEmpty())
    return fallback;

  // Division and remainder may recover a stronger decomposition.
  if (isa<arith::DivSIOp, arith::DivUIOp>(op))
    return refineDivRem(value, *lhs, *rhs, fallback, true, builder);
  if (isa<arith::RemSIOp, arith::RemUIOp>(op))
    return refineDivRem(value, *lhs, *rhs, fallback, false, builder);
  return fallback;
}

FailureOr<Result> CommonPtrRules::parseCast(Value value, Value input,
                                            bool unsignedCast, Analyze analyze,
                                            OpBuilder &builder) const {
  // Obtain the source facts and compare the integer widths of the cast.
  auto sourceInfo = analyze(input);
  if (failed(sourceInfo))
    return failure();
  if (sourceInfo->isEmpty())
    return Result{};
  const auto &source = sourceInfo->offsets;
  builder.setInsertionPointAfterValue(value);
  auto width = getIntegerWidth(value.getType());
  auto oldWidth = getIntegerWidth(source.valueType);
  if (failed(width) || failed(oldWidth))
    return Result{};

  // Check whether every represented lane keeps the same mathematical value.
  bool keep = *width == *oldWidth || (*width > *oldWidth && !unsignedCast);
  if (!keep) {
    auto bounds = constantDecompositionRange(source, *oldWidth);
    keep = bounds && (*width > *oldWidth
                          ? bounds->isAllNonNegative()
                          : bounds->getSignedMin().isSignedIntN(*width) &&
                                bounds->getSignedMax().isSignedIntN(*width));
  }
  if (keep) {
    // Signed widening keeps index descriptors. Other casts need the local
    // check above because they can change the value represented by a lane.
    auto offsets = source;
    offsets.valueType = value.getType();
    offsets.fullOffsets = value;
    return Result{{}, std::move(offsets)};
  }

  // Otherwise retain the exact cast result and only the invariant axis facts.
  auto offsets = buildOffsetInfo(value, builder);
  if (failed(offsets))
    return Result{};
  for (unsigned i = 0; i < source.axes.size(); ++i)
    if (source.axes[i].kind == AxisKind::Invariant)
      offsets->axes[i].kind = AxisKind::Invariant;
  setUniformOrigin(*offsets, value, builder);
  return Result{{}, std::move(*offsets)};
}

FailureOr<Result> CommonPtrRules::parseFloatToInt(arith::FPToSIOp op,
                                                  Analyze analyze,
                                                  OpBuilder &builder) const {
  // First use any constant or uniform facts available from the integer result.
  auto info = buildIntegerInfo(op.getResult(), builder);
  if (failed(info) || info->offsets.getShape().empty())
    return info;

  // Floating arithmetic contributes axis equality, without integer strides.
  llvm::DenseMap<Value, SmallVector<bool>> known;
  auto axes = inferInvariantAxes(op.getIn(), analyze, known);
  if (failed(axes))
    return failure();
  for (unsigned i = 0; i < axes->size(); ++i)
    if ((*axes)[i])
      info->offsets.axes[i].kind = AxisKind::Invariant;
  // Extract the integer result, preserving the actual conversion semantics.
  setUniformOrigin(info->offsets, op.getResult(), builder);
  return info;
}

FailureOr<Result> CommonPtrRules::parseReduce(Value value, triton::ReduceOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  // Seed result facts; tensor axes require a supported arithmetic combiner.
  auto info = buildIntegerInfo(value, builder);
  if (failed(info) || info->offsets.getShape().empty() ||
      !hasArithmeticCombiner(op))
    return info;

  llvm::DenseMap<Value, SmallVector<bool>> known;
  SmallVector<bool> invariant(info->offsets.axes.size(), true);
  // Results can depend on every input (e.g. argmin). Intersect input facts,
  // then remove the reduced dimension; never copy a pre-reduction stride.
  for (Value operand : op.getSrcs()) {
    auto source = inferInvariantAxes(operand, analyze, known);
    if (failed(source))
      return failure();
    source->erase(source->begin() + op.getAxis());
    for (unsigned i = 0; i < invariant.size(); ++i)
      invariant[i] = invariant[i] && (*source)[i];
  }

  // Keep common invariant axes and singleton dimensions, then recover an
  // origin.
  for (unsigned i = 0; i < invariant.size(); ++i)
    if (invariant[i] || info->offsets.getShape()[i] == 1)
      info->offsets.axes[i].kind = AxisKind::Invariant;
  setUniformOrigin(info->offsets, value, builder);
  return info;
}

// Ordinary pointer rules.
FailureOr<Result> CommonPtrRules::parseAddPtr(triton::AddPtrOp op,
                                              Analyze analyze,
                                              OpBuilder &builder) const {
  // Analyze the pointer and its integer increment independently.
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  if (parentInfo->isEmpty())
    return Result{};
  auto sourceInfo = analyze(op.getOffset());
  if (failed(sourceInfo))
    return failure();
  if (sourceInfo->isEmpty())
    return Result{};

  // Descriptor fields are already index. The complete offset is widened
  // separately before accumulation, preserving each addptr increment.
  builder.setInsertionPoint(op);
  auto info =
      addPtrOffsets(op.getResult(), *parentInfo, sourceInfo->offsets, builder);
  if (failed(info))
    return Result{};
  return Result(std::move(*info));
}

FailureOr<Result> CommonPtrRules::parseBitcast(triton::BitcastOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  // A scalar cast is already an exact base, including across element widths.
  // Do not reinterpret offsets accumulated before that scalar boundary.
  if (isa<triton::PointerType>(op.getType()))
    return buildScalarPtrInfo(op.getResult(), builder);

  // Require matching tensor shapes, encodings and address spaces.
  auto target = dyn_cast<RankedTensorType>(op.getType());
  auto source = dyn_cast<RankedTensorType>(op.getSrc().getType());
  if (!source || !target || source.getShape() != target.getShape() ||
      source.getEncoding() != target.getEncoding())
    return Result{};
  auto from = dyn_cast<triton::PointerType>(source.getElementType());
  auto to = dyn_cast<triton::PointerType>(target.getElementType());
  if (!from || !to || from.getAddressSpace() != to.getAddressSpace())
    return Result{};

  // Restrict element-unit conversion to the supported storage widths.
  auto storageWidth = [](Type type) -> unsigned {
    if (!isa<IntegerType, FloatType>(type))
      return 0;
    unsigned width = type.getIntOrFloatBitWidth();
    return width == 8 || width == 16 || width == 32 || width == 64 ? width : 0;
  };
  unsigned width = storageWidth(from.getPointeeType());
  unsigned targetWidth = storageWidth(to.getPointeeType());
  if (!width || !targetWidth)
    return Result{};

  // Recover the scalar base and element offsets of the source pointer tensor.
  auto sourceInfo = analyze(op.getSrc());
  if (failed(sourceInfo))
    return failure();
  if (sourceInfo->isEmpty() || sourceInfo->block)
    return Result{};
  PtrInfo info = *sourceInfo;
  builder.setInsertionPoint(op);

  // Equal-width casts reuse offsets; other casts must change their units.
  if (width != targetWidth) {
    uint64_t ratio =
        std::max(width, targetWidth) / std::min(width, targetWidth);
    bool widen = width < targetWidth;
    // Widening needs exact division: offset_new * sizeof(new) must equal
    // offset_old * sizeof(old). Test the complete offset, never round terms.
    if (widen && !canDivideDecomposition(info.offsets, ratio) &&
        !isKnownMultiple(info.offsets.fullOffsets, ratio))
      return Result{};

    // Materialize the complete offset and the element-size conversion factor.
    auto full = materializeFullOffset(
        info.offsets, addressStorageType(info.offsets.valueType, builder),
        builder, op.getLoc());
    if (failed(full))
      return Result{};
    auto fullType = cast<RankedTensorType>((*full).getType());
    Value factor = builder.create<arith::ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get(fullType, int64_t(ratio)));
    if (widen) {
      // Divide the full offset and retain exactly divisible descriptor fields.
      info.offsets.fullOffsets =
          builder.create<arith::DivSIOp>(op.getLoc(), *full, factor)
              .getResult();
      divideDecomposition(info.offsets, ratio, builder, op.getLoc());
      if (failed(normalizeAxes(info.offsets, builder)))
        return Result{};
    } else {
      // Smaller elements scale up both the complete offset and its descriptor.
      Value scaled = builder.create<arith::MulIOp>(op.getLoc(), *full, factor);
      auto offsets =
          scaleOffsets(scaled, info.offsets, builder.getIndexAttr(ratio),
                       builder, op.getLoc());
      if (failed(offsets))
        return Result{};
      info.offsets = std::move(*offsets);
    }
  }

  // Express the scalar base in the destination pointer type.
  if (info.base.getType() != to)
    info.base = builder.create<triton::BitcastOp>(op.getLoc(), to, info.base);
  return Result(std::move(info));
}

// Block pointer rules.
FailureOr<Result> CommonPtrRules::parseMakeTensorPtr(triton::MakeTensorPtrOp op,
                                                     Analyze analyze,
                                                     OpBuilder &builder) const {
  // Start from an ordinary scalar pointer with an index-typed base offset.
  auto baseInfo = analyze(op.getBase());
  if (failed(baseInfo))
    return failure();
  if (baseInfo->isEmpty() || baseInfo->block ||
      !baseInfo->offsets.valueType.isIndex())
    return Result{};

  // Resolve each starting coordinate to its uniform index origin.
  SmallVector<OpFoldResult> coordinates;
  for (Value coordinate : op.getOffsets()) {
    auto sourceInfo = analyze(coordinate);
    if (failed(sourceInfo))
      return failure();
    if (sourceInfo->isEmpty() || !sourceInfo->offsets.isUniform())
      return Result{};
    coordinates.push_back(sourceInfo->offsets.scalarOffset);
  }

  // Combine the base, coordinates and strides into the block descriptor.
  builder.setInsertionPoint(op);
  return makeBlockInfo(op, *baseInfo, coordinates, builder);
}

FailureOr<Result> CommonPtrRules::parseAdvance(triton::AdvanceOp op,
                                               Analyze analyze,
                                               OpBuilder &builder) const {
  // An advance needs the existing block coordinates and strides.
  auto parentInfo = analyze(op.getPtr());
  if (failed(parentInfo))
    return failure();
  if (parentInfo->isEmpty() || !parentInfo->block)
    return Result{};

  // Resolve each coordinate increment to its uniform index origin.
  SmallVector<OpFoldResult> coordinates;
  for (Value coordinate : op.getOffsets()) {
    auto sourceInfo = analyze(coordinate);
    if (failed(sourceInfo))
      return failure();
    if (sourceInfo->isEmpty() || !sourceInfo->offsets.isUniform())
      return Result{};
    coordinates.push_back(sourceInfo->offsets.scalarOffset);
  }

  // Update the coordinates and add their stride-weighted offset contribution.
  builder.setInsertionPoint(op);
  return advanceBlockInfo(op, *parentInfo, coordinates, builder);
}

} // namespace mlir::triton::ptr
