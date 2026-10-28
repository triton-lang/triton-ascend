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

namespace mlir::triton::ptr {

// Result queries and construction.
bool OffsetInfo::hasStructuredForm() const {
  return scalarOffset && llvm::all_of(axes, [](const AxisInfo &axis) {
           return axis.kind != AxisKind::Unknown;
         });
}

bool OffsetInfo::hasStructuredDecomposition() const {
  return scalarOffset && llvm::all_of(axes, [](const AxisInfo &axis) {
           return axis.kind != AxisKind::Unknown || !axis.subAxes.empty();
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

FailureOr<OffsetInfo> buildOffsetInfo(Value value, OpBuilder &builder) {
  if (!value)
    return failure();
  Type type = value.getType();
  // Getting the width also checks for a supported integer/index element type.
  if (failed(detail::getIntegerWidth(type)))
    return failure();
  OffsetInfo result;
  result.valueType = type;
  result.fullOffsets = value;
  result.axes.assign(result.getShape().size(),
                     AxisInfo(AxisKind::Unknown, builder.getIndexAttr(0)));
  return result;
}

namespace {

// Local checks shared by binding validation.
bool validOfrType(OpFoldResult value, Type expected) {
  if (!value)
    return false;
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType() == expected;
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) && cast<TypedAttr>(attr).getType() == expected;
  return false;
}

bool isIndex(OpFoldResult value) {
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType().isIndex();
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) && cast<TypedAttr>(attr).getType().isIndex();
  return false;
}

bool validOffset(Type resultType, const OffsetInfo &components,
                 bool pointer = false) {
  Type scalarType = IndexType::get(resultType.getContext());
  if (components.valueType != resultType ||
      components.axes.size() != components.getShape().size() ||
      failed(detail::getIntegerWidth(resultType)) ||
      (!components.fullOffsets &&
       (!pointer || !components.hasStructuredDecomposition())) ||
      (components.scalarOffset &&
       !validOfrType(components.scalarOffset, scalarType)))
    return false;
  if (components.fullOffsets) {
    // Integer results retain their source type. Pointer descriptors use index,
    // but their complete offset tensor can use an integer storage type.
    Type type;
    if (auto value = dyn_cast<Value>(components.fullOffsets))
      type = value.getType();
    else if (auto attr = dyn_cast<TypedAttr>(
                 dyn_cast<Attribute>(components.fullOffsets)))
      type = attr.getType();
    if (!type)
      return false;
    if (failed(detail::getIntegerWidth(type)) ||
        (!pointer && type != resultType))
      return false;
    auto tensor = dyn_cast<RankedTensorType>(type);
    auto target = dyn_cast<RankedTensorType>(resultType);
    if (bool(tensor) != bool(target) ||
        (tensor && (tensor.getShape() != target.getShape() ||
                    tensor.getEncoding() != target.getEncoding())))
      return false;
  }
  for (const auto &axis : components.axes) {
    if (!validOfrType(axis.stride, scalarType))
      return false;
    for (const auto &sub : axis.subAxes)
      if (!validOfrType(sub.stride, scalarType))
        return false;
  }
  OpBuilder builder(resultType.getContext());
  OffsetInfo normalized = components;
  if (failed(detail::normalizeAxes(normalized, builder)))
    return false;
  for (unsigned i = 0; i < components.axes.size(); ++i) {
    if (normalized.axes[i].kind != components.axes[i].kind)
      return false;
    if (components.axes[i].kind != AxisKind::Unknown &&
        normalized.axes[i].stride != components.axes[i].stride) {
      auto a = detail::getIntegerConstant(normalized.axes[i].stride, 64);
      auto b = detail::getIntegerConstant(components.axes[i].stride, 64);
      if (failed(a) || failed(b) || *a != *b)
        return false;
    }
  }
  return true;
}

} // namespace

namespace detail {

// Pointer type queries.
bool isOrdinaryPtrType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  auto ptr = dyn_cast<triton::PointerType>(type);
  return ptr && !isa<ShapedType>(ptr.getPointeeType());
}

bool isBlockPtrType(Type type) {
  auto ptr = dyn_cast<triton::PointerType>(type);
  return ptr && isa<RankedTensorType>(ptr.getPointeeType());
}

// Binding validation.
bool validOffsetBinding(Value boundary, const OffsetInfo &components) {
  return boundary && validOffset(boundary.getType(), components);
}

bool validBlockPtrBinding(Value value, const PtrInfo &components) {
  if (!value || !isBlockPtrType(value.getType()) || !components.block ||
      !components.offsets.hasStructuredForm())
    return false;
  const auto &block = *components.block;
  unsigned rank = components.offsets.getShape().size();
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

bool validPtrBinding(Value boundary, const PtrInfo &components) {
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
  return validOffset(source, components.offsets,
                     /*pointer=*/true);
}

// SSA dependencies used by materialization.
void collectValues(ArrayRef<OpFoldResult> fields,
                   SmallVectorImpl<Value> &values) {
  for (OpFoldResult field : fields) {
    if (!field)
      continue;
    if (auto value = dyn_cast<Value>(field))
      values.push_back(value);
  }
}

} // namespace detail
} // namespace mlir::triton::ptr
