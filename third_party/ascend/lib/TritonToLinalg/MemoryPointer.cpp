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

#include "TritonToLinalg/MemoryPointer.h"
#include "TritonMemoryAccess/PtrAnalysis.h"
#include "TritonMemoryAccess/Utils.h"
#include "Utils/PtrAnalysisUtils.h"
#include "Utils/Utils.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
namespace mlir::triton {
hivm::PointerCastOp createScalarPointerCast(OpBuilder &builder, Location loc,
                                            MemRefType resultType,
                                            Value address) {
  SmallVector<Value> dynamicSizes;
  if (resultType.getNumDynamicDims() != 0) {
    Value defaultSize = builder.create<arith::ConstantIndexOp>(loc, 1);
    dynamicSizes.assign(resultType.getNumDynamicDims(), defaultSize);
  }
  auto pointerCast = builder.create<hivm::PointerCastOp>(
      loc, resultType, ValueRange{address}, ValueRange{dynamicSizes});
  pointerCast->setAttr(kScalarPointerCarrierAttr,
                       UnitAttr::get(builder.getContext()));
  return pointerCast;
}

FailureOr<Value>
materializePointerAddress(Value value, Location loc,
                          ConversionPatternRewriter &rewriter) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1 ||
        !isa<MemRefType>(cast.getInputs()[0].getType()))
      break;
    value = cast.getInputs()[0];
  }
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return failure();
  Type elementType = memrefType.getElementType();
  if (!elementType.isIntOrFloat())
    return failure();

  // A canonical scalar PointerCast already stores the complete byte address
  // that IntToPtr received. Extracting its aligned pointer immediately and
  // casting it back to i64 is an identity round trip. In current failing
  // kernels this chain also survives into InjectSync, so folding it keeps a
  // redundant region-local form out of the backend. Only fold the canonical
  // one-dimensional carrier; descriptors with a non-zero offset or a non-unit
  // stride still require the general address materialization below.
  if (auto pointerCast = value.getDefiningOp<hivm::PointerCastOp>();
      pointerCast && pointerCast->hasAttr(kScalarPointerCarrierAttr)) {
    auto [strides, offset] = memrefType.getStridesAndOffset();
    if (pointerCast.getAddrs().size() == 1 && memrefType.getRank() == 1 &&
        offset == 0 && strides.size() == 1 && strides.front() == 1) {
      Value address = pointerCast.getAddrs().front();
      if (address.getType().isInteger(64))
        return address;
      if (address.getType().isIndex())
        return rewriter
            .create<arith::IndexCastOp>(loc, rewriter.getI64Type(), address)
            .getResult();
    }
  }

  Value address =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, value);
  int64_t staticOffset = memrefType.getStridesAndOffset().second;
  if (staticOffset != 0) {
    Value elementOffset;
    if (ShapedType::isDynamic(staticOffset)) {
      elementOffset =
          rewriter.create<memref::ExtractStridedMetadataOp>(loc, value)
              .getOffset();
    } else {
      elementOffset =
          rewriter.create<arith::ConstantIndexOp>(loc, staticOffset);
    }

    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;
    if (elementBytes != 1) {
      Value scale = rewriter.create<arith::ConstantIndexOp>(loc, elementBytes);
      elementOffset = rewriter.create<arith::MulIOp>(loc, elementOffset, scale);
    }
    address = rewriter.create<arith::AddIOp>(loc, address, elementOffset);
  }
  return rewriter
      .create<arith::IndexCastOp>(loc, rewriter.getI64Type(), address)
      .getResult();
}

static bool isPointer(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return isa<triton::PointerType>(type);
}

LogicalResult collectMemoryPointerInfos(ModuleOp module,
                                        MemoryPointerInfos &infos) {
  infos.clear();
  llvm::SetVector<Value> values;
  module.walk([&](Operation *op) {
    for (Value value : op->getOperands())
      if (isPointer(value.getType()))
        values.insert(value);
  });
  OpBuilder builder(module.getContext());
  ptr::PtrAnalysis analysis(builder);
  if (failed(importPointerDescriptorFacts(module, analysis)))
    return failure();
  for (unsigned i = 0; i < values.size(); ++i) {
    Value value = values[i];
    auto info = analysis.analyze(value);
    if (failed(info))
      continue;
    infos.try_emplace(value, *info);
    // A tensor bitcast can introduce a scalar pointer base. Its own address
    // conversion needs the same source facts as pre-existing pointer users.
    if (info->base) {
      values.insert(info->base);
      if (auto *producer = info->base.getDefiningOp())
        for (Value operand : producer->getOperands())
          if (isPointer(operand.getType()))
            values.insert(operand);
    }
  }
  return success();
}

static OpFoldResult remap(OpFoldResult field,
                          ConversionPatternRewriter &rewriter) {
  if (auto value = dyn_cast<Value>(field)) {
    Value mapped = rewriter.getRemappedValue(value);
    return mapped ? OpFoldResult(mapped) : field;
  }
  return field;
}

static FailureOr<Value>
materializeMemoryPointer(const ptr::PtrInfo &info, Location loc,
                         ConversionPatternRewriter &rewriter,
                         OpFoldResult remappedOrigin = {}) {
  Value sourceBase = info.base;
  if (auto bridge = sourceBase.getDefiningOp<UnrealizedConversionCastOp>();
      bridge && bridge.getInputs().size() == 1 &&
      bridge.getInputs()[0].getDefiningOp() &&
      isDistributedTypeCustomOp(bridge.getInputs()[0].getDefiningOp()))
    sourceBase = bridge.getInputs()[0];
  Value base = rewriter.getRemappedValue(sourceBase);
  if (!base)
    return failure();
  while (auto cast = base.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1 ||
        !isa<MemRefType>(cast.getInputs()[0].getType()))
      break;
    base = cast.getInputs()[0];
  }
  auto baseType = dyn_cast<MemRefType>(base.getType());
  if (!baseType)
    return failure();
  // A scalar base with no displacement already denotes the requested address.
  // Reusing it also preserves the complete-address fold for int_to_ptr.
  if (info.offsets.getShape().empty() && baseType.getRank() == 1 &&
      isConstantIntValue(
          remappedOrigin ? remappedOrigin : info.offsets.scalarOffset, 0))
    return base;
  OpFoldResult baseOffset;
  if (auto cast = base.getDefiningOp<memref::ReinterpretCastOp>())
    baseOffset = cast.getConstifiedMixedOffset();
  else if (!ShapedType::isDynamic(baseType.getStridesAndOffset().second))
    baseOffset = rewriter.getIndexAttr(baseType.getStridesAndOffset().second);
  else
    baseOffset = rewriter.create<memref::ExtractStridedMetadataOp>(loc, base)
                     .getOffset();

  OpFoldResult offset = addOpFoldResult(
      baseOffset,
      remappedOrigin ? remappedOrigin
                     : remap(info.offsets.scalarOffset, rewriter),
      loc, rewriter, rewriter.getIndexType());
  SmallVector<OpFoldResult> sizes, strides;
  for (auto [size, axis] :
       llvm::zip_equal(info.offsets.getShape(), info.offsets.axes)) {
    if (ShapedType::isDynamic(size))
      return failure();
    sizes.push_back(rewriter.getIndexAttr(size));
    strides.push_back(size == 1 ? rewriter.getIndexAttr(1)
                                : remap(axis.stride, rewriter));
  }
  if (sizes.empty()) {
    sizes.push_back(rewriter.getIndexAttr(1));
    strides.push_back(rewriter.getIndexAttr(1));
  }
  // reinterpret_cast offsets are absolute from the underlying allocation;
  // composing with a converted base requires adding its offset exactly once.
  SmallVector<int64_t> shape, staticStrides;
  for (auto size : sizes)
    shape.push_back(*getConstantIntValue(size));
  for (auto stride : strides)
    staticStrides.push_back(
        getConstantIntValue(stride).value_or(ShapedType::kDynamic));
  int64_t staticOffset =
      getConstantIntValue(offset).value_or(ShapedType::kDynamic);
  if (staticOffset < 0 && !ShapedType::isDynamic(staticOffset)) {
    offset =
        rewriter.create<arith::ConstantIndexOp>(loc, staticOffset).getResult();
    staticOffset = ShapedType::kDynamic;
  }
  auto layout = StridedLayoutAttr::get(rewriter.getContext(), staticOffset,
                                       staticStrides);
  auto type = MemRefType::get(shape, baseType.getElementType(), layout,
                              baseType.getMemorySpace());
  return rewriter
      .create<memref::ReinterpretCastOp>(loc, type, base, offset, sizes,
                                         strides)
      .getResult();
}

FailureOr<Value> materializeMemoryPointer(Value pointer, Value converted,
                                          const MemoryPointerInfos &infos,
                                          ConversionPatternRewriter &rewriter) {
  if (!isPointer(pointer.getType())) {
    if (isa<MemRefType>(converted.getType()))
      return converted;
    return failure();
  }
  auto found = infos.find(pointer);
  if (found == infos.end())
    return failure();
  const ptr::PtrInfo &info = found->second;
  if (!info.hasScalarBase() || !info.offsets.hasStructuredForm())
    return failure();
  Location loc = pointer.getLoc();
  return materializeMemoryPointer(info, loc, rewriter);
}

FailureOr<BoundaryCheckedPointer> materializeBoundaryCheckedMemoryPointer(
    Value pointer, Value converted, ArrayRef<int32_t> checks,
    const MemoryPointerInfos &infos, ConversionPatternRewriter &rewriter) {
  auto unchanged = [&]() -> FailureOr<BoundaryCheckedPointer> {
    auto view = materializeMemoryPointer(pointer, converted, infos, rewriter);
    if (failed(view))
      return failure();
    return BoundaryCheckedPointer{*view, {}};
  };
  // Ascend950 memory planning is sensitive to equivalent descriptor
  // expressions. Retain the legacy nonnegative coordinates for checked axes
  // while getMemoryBoundary keeps the original tile padding and lengths.
  auto make = pointer.getDefiningOp<triton::MakeTensorPtrOp>();
  if (checks.empty() || !make ||
      !hacc::utils::isAscend910_95(make->getParentOfType<ModuleOp>()) ||
      make->hasAttr(controlflow::kPointerDescriptorRebuildAttr))
    return unchanged();

  auto found = infos.find(pointer);
  if (found == infos.end())
    return unchanged();
  const ptr::PtrInfo &info = found->second;
  if (!info.hasScalarBase() || !info.block ||
      !info.offsets.hasStructuredForm() ||
      info.block->descriptorBase != make.getBase())
    return unchanged();
  auto shape = info.offsets.getShape();
  if (llvm::any_of(shape, ShapedType::isDynamic) ||
      llvm::any_of(checks, [&](int32_t axis) {
        return axis < 0 || static_cast<size_t>(axis) >= shape.size();
      }))
    return unchanged();

  // The descriptor base may already include scalar addptr displacements.
  // Rebuild from that origin, not from the complete old block offset.
  auto base = infos.find(info.block->descriptorBase);
  if (base == infos.end())
    return unchanged();
  const ptr::PtrInfo &baseInfo = base->second;
  if (baseInfo.base != info.base || !baseInfo.hasScalarBase() ||
      baseInfo.block || !baseInfo.offsets.valueType.isIndex() ||
      !baseInfo.offsets.scalarOffset ||
      baseInfo.getElementType() != info.getElementType())
    return unchanged();

  SmallVector<Value> checkedCoordinates;
  for (int32_t axis : checks) {
    Value coordinate = rewriter.getRemappedValue(make.getOffsets()[axis]);
    if (!coordinate || !coordinate.getType().isInteger(32))
      return unchanged();
    checkedCoordinates.push_back(coordinate);
  }

  Location loc = pointer.getLoc();
  SmallVector<OpFoldResult> coordinates;
  for (OpFoldResult coordinate : info.block->coordinateOffsets)
    coordinates.push_back(remap(coordinate, rewriter));
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  for (auto [axis, coordinate] : llvm::zip_equal(checks, checkedCoordinates)) {
    Value clamped = rewriter.create<arith::MaxSIOp>(loc, coordinate, zero);
    coordinates[axis] =
        rewriter
            .create<arith::IndexCastOp>(loc, rewriter.getIndexType(), clamped)
            .getResult();
  }

  OpFoldResult origin = remap(baseInfo.offsets.scalarOffset, rewriter);
  for (auto [coordinate, axis] :
       llvm::zip_equal(coordinates, info.offsets.axes)) {
    // A unit tile axis may have memref stride 1; its address contribution
    // must still use the original descriptor stride.
    auto contribution =
        mulOpFoldResult(coordinate, remap(axis.stride, rewriter), loc, rewriter,
                        rewriter.getIndexType());
    origin = addOpFoldResult(origin, contribution, loc, rewriter,
                             rewriter.getIndexType());
  }
  auto view = materializeMemoryPointer(info, loc, rewriter, origin);
  if (failed(view))
    return failure();
  return BoundaryCheckedPointer{*view, llvm::to_vector(checks)};
}

FailureOr<MemoryBoundary> getMemoryBoundary(
    Value pointer, ArrayRef<int32_t> checks, ArrayRef<int32_t> zeroBasedAxes,
    const MemoryPointerInfos &infos, ConversionPatternRewriter &rewriter) {
  auto found = infos.find(pointer);
  if (found == infos.end() || !found->second.block)
    return failure();
  const auto &info = found->second;
  Location loc = pointer.getLoc();
  OpFoldResult zero = rewriter.getIndexAttr(0);
  MemoryBoundary boundary;
  for (auto [dim, extent] : llvm::enumerate(info.offsets.getShape())) {
    OpFoldResult size = rewriter.getIndexAttr(extent);
    OpFoldResult start = zero;
    if (llvm::is_contained(checks, static_cast<int32_t>(dim))) {
      OpFoldResult coordinate =
          remap(info.block->coordinateOffsets[dim], rewriter);
      OpFoldResult parent = remap(info.block->parentShape[dim], rewriter);
      start = minOpFoldResult(
          size,
          maxOpFoldResult(zero,
                          subOpFoldResult(zero, coordinate, loc, rewriter), loc,
                          rewriter),
          loc, rewriter);
      auto end = minOpFoldResult(
          size,
          maxOpFoldResult(zero,
                          subOpFoldResult(parent, coordinate, loc, rewriter),
                          loc, rewriter),
          loc, rewriter);
      size = maxOpFoldResult(zero, subOpFoldResult(end, start, loc, rewriter),
                             loc, rewriter);
    }
    boundary.tileStarts.push_back(start);
    boundary.gmStarts.push_back(
        llvm::is_contained(zeroBasedAxes, static_cast<int32_t>(dim)) ? zero
                                                                     : start);
    boundary.sizes.push_back(size);
  }
  return boundary;
}
} // namespace mlir::triton
