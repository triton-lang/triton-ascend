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

#include "TritonMemoryAccess/PtrAnalysisInfo.h"
#include "TritonMemoryAccess/PtrInfoUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::triton::ptr {

bool OffsetInfo::hasStructuredForm() const {
  return valueType && scalarOffset && axes.size() == shape.size() &&
         llvm::all_of(axes, [](const AxisInfo &axis) {
           return axis.kind != AxisKind::Unknown && bool(axis.stride);
         });
}
bool OffsetInfo::hasCompleteForm() const {
  return valueType && scalarOffset && axes.size() == shape.size() &&
         llvm::all_of(axes, [](const AxisInfo &axis) {
           return (axis.kind != AxisKind::Unknown && bool(axis.stride)) ||
                  (axis.subAxes && !axis.subAxes->empty());
         });
}
bool OffsetInfo::isUniform() const {
  return hasStructuredForm() && llvm::all_of(axes, [](const AxisInfo &axis) {
           return axis.kind == AxisKind::Invariant;
         });
}
Type PtrInfo::getElementType() const {
  return cast<triton::PointerType>(base.getType()).getPointeeType();
}
unsigned PtrInfo::getAddressSpace() const {
  return cast<triton::PointerType>(base.getType()).getAddressSpace();
}

FailureOr<OffsetInfo> makeOpaqueOffset(Value value, OpBuilder &builder,
                                       AnalysisOptions options) {
  if (!value || (!isa<IntegerType, IndexType>(value.getType()) &&
                 !isa<RankedTensorType>(value.getType())))
    return failure();
  auto width = detail::getIntegerWidth(value.getType(), options);
  if (failed(width))
    return failure();
  OffsetInfo result;
  result.valueType = value.getType();
  result.shape = detail::getLogicalShape(value.getType());
  result.fullOffsets = value;
  result.axes.assign(result.shape.size(),
                     AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  if (!options.indexBitWidth || *width > *options.indexBitWidth)
    return result;
  Value scalar;
  OpBuilder::InsertionGuard guard(builder);
  if (!isa<RankedTensorType>(value.getType())) {
    scalar = value;
  } else if (result.shape.empty()) {
    if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
      scalar = splat.getSrc();
    } else {
      builder.setInsertionPointAfterValue(value);
      scalar = builder.create<tensor::ExtractOp>(value.getLoc(), value,
                                                 ValueRange{});
    }
  }
  if (scalar) {
    // The cast follows its actual input, including an extraction just created.
    builder.setInsertionPointAfterValue(scalar);
    auto origin = detail::foldOrCreateCast(scalar, builder.getIndexType(),
                                           detail::CastKind::Signed, options,
                                           builder, value.getLoc());
    if (failed(origin))
      return failure();
    result.scalarOffset = *origin;
  }
  return result;
}

namespace {

bool validOfrType(OpFoldResult value, Type expected) {
  if (!value)
    return false;
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType() == expected;
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) && cast<TypedAttr>(attr).getType() == expected;
  return false;
}

bool validOffset(Type resultType, const OffsetInfo &components,
                 AnalysisOptions options, bool allowDeferred = false) {
  Type scalarType = IndexType::get(resultType.getContext());
  if (components.valueType != resultType ||
      components.shape != detail::getLogicalShape(resultType) ||
      components.axes.size() != components.shape.size() ||
      failed(detail::getIntegerWidth(resultType, options)) ||
      (components.fullOffsets
           ? !validOfrType(components.fullOffsets, resultType) ||
                 !components.addressTerms.empty()
           : !allowDeferred || (components.addressTerms.empty() &&
                                !components.hasCompleteForm())) ||
      (!allowDeferred && !components.addressTerms.empty()) ||
      (components.scalarOffset &&
       !validOfrType(*components.scalarOffset, scalarType)))
    return false;
  for (OpFoldResult term : components.addressTerms) {
    if (!term)
      return false;
    Type type;
    if (auto value = dyn_cast<Value>(term))
      type = value.getType();
    else if (auto attr = dyn_cast<TypedAttr>(dyn_cast<Attribute>(term)))
      type = attr.getType();
    if (!type)
      return false;
    auto width = detail::getIntegerWidth(type, options);
    if (failed(width) || *width > 64)
      return false;
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      auto target = dyn_cast<RankedTensorType>(resultType);
      if (!target || tensor.getShape() != target.getShape() ||
          tensor.getEncoding() != target.getEncoding())
        return false;
    }
  }
  for (const auto &axis : components.axes) {
    if (!validOfrType(axis.stride, scalarType))
      return false;
    if (axis.subAxes)
      for (const auto &sub : *axis.subAxes) {
        if (!validOfrType(sub.stride, scalarType) || !sub.shape)
          return false;
        Type shapeType;
        if (auto value = dyn_cast<Value>(sub.shape))
          shapeType = value.getType();
        else {
          auto attr = dyn_cast<TypedAttr>(cast<Attribute>(sub.shape));
          if (!attr)
            return false;
          shapeType = attr.getType();
        }
        if (!shapeType.isIndex() && !shapeType.isSignlessInteger())
          return false;
      }
  }
  OpBuilder builder(resultType.getContext());
  OffsetInfo normalized = components;
  if (failed(detail::normalizeAxes(normalized, options, builder)))
    return false;
  for (unsigned i = 0; i < components.axes.size(); ++i) {
    if (normalized.axes[i].kind != components.axes[i].kind)
      return false;
    if (components.axes[i].kind != AxisKind::Unknown &&
        normalized.axes[i].stride != components.axes[i].stride) {
      unsigned width = *options.indexBitWidth;
      auto a = detail::getIntegerConstant(normalized.axes[i].stride, width);
      auto b = detail::getIntegerConstant(components.axes[i].stride, width);
      if (failed(a) || failed(b) || *a != *b)
        return false;
    }
  }
  return true;
}

bool validPtr(Value boundary, const PtrInfo &components,
              AnalysisOptions options) {
  if (!boundary || !components.base ||
      !isa<triton::PointerType>(components.base.getType()))
    return false;
  Type ptrElement = boundary.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(ptrElement))
    ptrElement = tensor.getElementType();
  auto ptrType = dyn_cast<triton::PointerType>(ptrElement);
  auto baseType = dyn_cast<triton::PointerType>(components.base.getType());
  if (!ptrType || !baseType)
    return false;
  auto blockType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
  if (!blockType && isa<ShapedType>(ptrType.getPointeeType()))
    return false;
  Type elementType =
      blockType ? blockType.getElementType() : ptrType.getPointeeType();
  if (baseType.getPointeeType() != elementType ||
      baseType.getAddressSpace() != ptrType.getAddressSpace() ||
      (blockType ? !detail::validBlockPtrBinding(boundary, components)
                 : components.block.has_value()))
    return false;
  Type source = blockType ? Type(blockType) : boundary.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    source = RankedTensorType::get(tensor.getShape(),
                                   IndexType::get(source.getContext()),
                                   tensor.getEncoding());
  else
    source = IndexType::get(source.getContext());
  return validOffset(source, components.offsets, options,
                     /*allowDeferred=*/true);
}

} // namespace

namespace detail {
bool validOffsetBinding(Value boundary, const OffsetInfo &components,
                        AnalysisOptions options) {
  return boundary && validOffset(boundary.getType(), components, options);
}

bool validPtrBinding(Value boundary, const PtrInfo &components,
                     AnalysisOptions options) {
  return supportsAddressAnalysis(options) &&
         options.indexBitWidth == options.addressBitWidth &&
         validPtr(boundary, components, options);
}
} // namespace detail

FailureOr<IntegerInfo> makeIntegerInfo(OffsetInfo offsets,
                                       AnalysisOptions options) {
  auto width = detail::getIntegerWidth(offsets.valueType, options);
  if (failed(width))
    return failure();
  return IntegerInfo{std::move(offsets), llvm::ConstantRange::getFull(*width)};
}

namespace detail {
namespace {
void collectOfr(OpFoldResult field, SmallVectorImpl<Value> &values) {
  if (!field)
    return;
  if (auto value = dyn_cast<Value>(field))
    values.push_back(value);
}
} // namespace
void collectValues(ArrayRef<OpFoldResult> fields,
                   SmallVectorImpl<Value> &values) {
  for (OpFoldResult field : fields)
    collectOfr(field, values);
}

bool isOrdinaryPtrType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  auto ptr = dyn_cast<triton::PointerType>(type);
  return ptr && !isa<ShapedType>(ptr.getPointeeType());
}

namespace {
bool isIndex(OpFoldResult value) {
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType().isIndex();
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) && cast<TypedAttr>(attr).getType().isIndex();
  return false;
}

} // namespace

bool isBlockPtrType(Type type) {
  auto ptr = dyn_cast<triton::PointerType>(type);
  return ptr && isa<RankedTensorType>(ptr.getPointeeType());
}

bool validBlockPtrBinding(Value value, const PtrInfo &components) {
  if (!value || !isBlockPtrType(value.getType()) || !components.block ||
      !components.offsets.hasStructuredForm())
    return false;
  const auto &block = *components.block;
  unsigned rank = components.offsets.shape.size();
  if (block.parentShape.size() != rank ||
      block.coordinateOffsets.size() != rank || block.order.size() != rank)
    return false;
  SmallVector<bool> seen(rank, false);
  for (int32_t dim : block.order) {
    if (dim < 0 || static_cast<unsigned>(dim) >= rank || seen[dim])
      return false;
    seen[dim] = true;
  }
  for (unsigned i = 0; i < rank; ++i)
    if (!isIndex(block.parentShape[i]) || !isIndex(block.coordinateOffsets[i]))
      return false;
  return true;
}

} // namespace detail
} // namespace mlir::triton::ptr
