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

#include "PointerAnalysisInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::pointer {
namespace {

bool isI64(OpFoldResult value) {
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType().isSignlessInteger(64);
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) &&
           cast<TypedAttr>(attr).getType().isSignlessInteger(64);
  return false;
}

FailureOr<OpFoldResult> addScaledCoordinates(OpFoldResult origin,
                                             ArrayRef<OpFoldResult> coordinates,
                                             ArrayRef<OpFoldResult> strides,
                                             OpBuilder &builder, Location loc) {
  if (coordinates.size() != strides.size())
    return failure();
  for (auto [coordinate, stride] : llvm::zip(coordinates, strides)) {
    auto term = detail::foldOrCreateBinary(
        detail::BinaryKind::Multiply, coordinate, stride, builder.getI64Type(),
        AnalysisOptions{}, builder, loc);
    if (failed(term))
      return failure();
    auto sum = detail::foldOrCreateBinary(detail::BinaryKind::Add, origin,
                                          *term, builder.getI64Type(),
                                          AnalysisOptions{}, builder, loc);
    if (failed(sum))
      return failure();
    origin = *sum;
  }
  return origin;
}

FailureOr<BlockAdvanceDelta>
getCoordinates(ArrayRef<detail::AnalysisResult> inputs) {
  BlockAdvanceDelta delta;
  for (const auto &input : inputs) {
    if (!input.offset || !input.integerRange ||
        input.offset->domain != ArithmeticDomain::ElementAddress ||
        !input.offset->valueType.isSignlessInteger(64) ||
        !isI64(input.offset->completeOffset) ||
        input.integerRange->getBitWidth() != 64)
      return failure();
    delta.coordinateDeltas.push_back(input.offset->completeOffset);
    delta.ranges.push_back(*input.integerRange);
  }
  return delta;
}

} // namespace

FailureOr<BlockCoordinates> advanceBlockOffsets(
    ArrayRef<OpFoldResult> current, ArrayRef<llvm::ConstantRange> currentRanges,
    const BlockAdvanceDelta &delta, OpBuilder &builder, Location loc) {
  if (current.size() != currentRanges.size() ||
      current.size() != delta.coordinateDeltas.size() ||
      current.size() != delta.ranges.size())
    return failure();
  BlockCoordinates result;
  result.values.reserve(current.size());
  result.ranges.reserve(current.size());
  for (unsigned dim = 0; dim < current.size(); ++dim) {
    if (!isI64(current[dim]) || !isI64(delta.coordinateDeltas[dim]) ||
        currentRanges[dim].getBitWidth() != 64 ||
        delta.ranges[dim].getBitWidth() != 64)
      return failure();
    FailureOr<OpFoldResult> sum = detail::foldOrCreateBinary(
        detail::BinaryKind::Add, current[dim], delta.coordinateDeltas[dim],
        builder.getI64Type(), AnalysisOptions{}, builder, loc);
    if (failed(sum))
      return failure();
    result.values.push_back(*sum);
    result.ranges.push_back(currentRanges[dim].add(delta.ranges[dim]));
  }
  return result;
}

FailureOr<OffsetComponents> buildAddressView(const PointerComponents &pointer,
                                             OpBuilder &builder, Location loc) {
  const auto &offsets = pointer.offsets;
  if (offsets.domain != ArithmeticDomain::ElementAddress)
    return failure();
  if (offsets.completeOffset)
    return offsets;
  if (!offsets.hasAffineForm() || !isI64(offsets.uniformOffset) ||
      llvm::any_of(offsets.strides,
                   [](OpFoldResult stride) { return !isI64(stride); }))
    return failure();
  auto tensorType = dyn_cast<RankedTensorType>(offsets.valueType);
  if (!tensorType || !tensorType.getElementType().isSignlessInteger(64) ||
      offsets.shape != llvm::to_vector(tensorType.getShape()) ||
      !detail::canMaterializeAddressTensor(tensorType))
    return failure();

  auto uniform = detail::materialize(offsets.uniformOffset,
                                     builder.getI64Type(), builder, loc);
  if (failed(uniform))
    return failure();
  auto splat = detail::splatTo(*uniform, tensorType, builder, loc);
  if (failed(splat))
    return failure();
  Value complete = *splat;
  for (unsigned dim = 0; dim < offsets.shape.size(); ++dim) {
    if (offsets.axes[dim] == AxisKind::Invariant)
      continue;
    auto lane = detail::materializeAxisRange(tensorType, dim, builder, loc);
    auto stride = detail::materialize(offsets.strides[dim],
                                      builder.getI64Type(), builder, loc);
    if (failed(lane) || failed(stride))
      return failure();
    auto strideTensor = detail::splatTo(*stride, tensorType, builder, loc);
    if (failed(strideTensor))
      return failure();
    Value term = builder.create<arith::MulIOp>(loc, *lane, *strideTensor);
    complete = builder.create<arith::AddIOp>(loc, complete, term);
  }
  OffsetComponents result = offsets;
  result.completeOffset = complete;
  return result;
}

namespace detail {

bool isBlockPointerType(Type type) {
  auto pointer = dyn_cast<triton::PointerType>(type);
  return pointer && isa<RankedTensorType>(pointer.getPointeeType());
}

bool validBlockPointerBinding(Value value,
                              const PointerComponents &components) {
  if (!value || !isBlockPointerType(value.getType()) || !components.block ||
      !components.offsets.hasAffineForm())
    return false;
  const auto &block = *components.block;
  unsigned rank = components.offsets.shape.size();
  if (block.parentShape.size() != rank ||
      block.coordinateOffsets.size() != rank ||
      block.coordinateRanges.size() != rank || block.order.size() != rank)
    return false;
  SmallVector<bool> seen(rank, false);
  for (int32_t dim : block.order) {
    if (dim < 0 || static_cast<unsigned>(dim) >= rank || seen[dim])
      return false;
    seen[dim] = true;
  }
  for (unsigned i = 0; i < rank; ++i)
    if (!isI64(block.parentShape[i]) || !isI64(block.coordinateOffsets[i]) ||
        block.coordinateRanges[i].getBitWidth() != 64)
      return false;
  return true;
}

FailureOr<SmallVector<AnalysisRequest>>
collectBlockPointerInputs(AnalysisRequest request) {
  SmallVector<AnalysisRequest> inputs;
  if (request.kind == AdvanceDeltaRequest) {
    auto advance = request.value.getDefiningOp<triton::AdvanceOp>();
    if (!advance)
      return failure();
    for (Value delta : advance.getOffsets())
      inputs.push_back({delta, AddressDelta});
    return inputs;
  }
  if (request.kind != PointerValue)
    return failure();
  if (auto make = request.value.getDefiningOp<triton::MakeTensorPtrOp>()) {
    inputs.push_back({make.getBase(), PointerValue});
    for (Value coordinate : make.getOffsets())
      inputs.push_back({coordinate, AddressDelta});
    return inputs;
  }
  if (auto advance = request.value.getDefiningOp<triton::AdvanceOp>()) {
    inputs.push_back({advance.getPtr(), PointerValue});
    inputs.push_back({request.value, AdvanceDeltaRequest});
    return inputs;
  }
  return failure();
}

FailureOr<AnalysisResult> transferBlockPointer(AnalysisRequest request,
                                               ArrayRef<AnalysisResult> inputs,
                                               OpBuilder &builder) {
  if (request.kind == AdvanceDeltaRequest) {
    auto delta = getCoordinates(inputs);
    if (failed(delta))
      return failure();
    AnalysisResult result;
    result.advanceDelta = std::move(*delta);
    return result;
  }
  if (request.kind != PointerValue)
    return failure();
  PointerComponents pointer;
  if (auto make = request.value.getDefiningOp<triton::MakeTensorPtrOp>()) {
    if (inputs.size() != make.getOffsets().size() + 1 ||
        !inputs.front().pointer || inputs.front().pointer->block ||
        !inputs.front().pointer->offsets.valueType.isSignlessInteger(64))
      return failure();
    auto coordinates = getCoordinates(inputs.drop_front());
    if (failed(coordinates))
      return failure();
    pointer = *inputs.front().pointer;
    auto pointerType = cast<triton::PointerType>(make.getType());
    auto tensorType = cast<RankedTensorType>(pointerType.getPointeeType());
    auto &offsets = pointer.offsets;
    offsets.valueType = RankedTensorType::get(
        tensorType.getShape(), builder.getI64Type(), tensorType.getEncoding());
    offsets.shape = llvm::to_vector(tensorType.getShape());
    offsets.completeOffset = {};
    offsets.strides.clear();
    offsets.axes.clear();
    for (Value stride : make.getStrides()) {
      offsets.strides.push_back(stride);
      auto constant = getIntegerConstant(stride, 64);
      offsets.axes.push_back(succeeded(constant) && constant->isZero()
                                 ? AxisKind::Invariant
                                 : AxisKind::Structured);
    }
    auto origin = addScaledCoordinates(offsets.uniformOffset,
                                       coordinates->coordinateDeltas,
                                       offsets.strides, builder, make.getLoc());
    if (failed(origin))
      return failure();
    offsets.uniformOffset = *origin;
    pointer.block.emplace();
    auto &block = *pointer.block;
    for (Value shape : make.getShape())
      block.parentShape.push_back(shape);
    block.coordinateOffsets = std::move(coordinates->coordinateDeltas);
    block.coordinateRanges = std::move(coordinates->ranges);
    block.order = llvm::to_vector(make.getOrderAttr().asArrayRef());
  } else if (auto advance = request.value.getDefiningOp<triton::AdvanceOp>()) {
    if (inputs.size() != 2 || !inputs[0].pointer || !inputs[0].pointer->block ||
        !inputs[1].advanceDelta)
      return failure();
    pointer = *inputs[0].pointer;
    auto &block = *pointer.block;
    auto updated =
        advanceBlockOffsets(block.coordinateOffsets, block.coordinateRanges,
                            *inputs[1].advanceDelta, builder, advance.getLoc());
    if (failed(updated))
      return failure();
    // Add only the advance contribution, retaining any scalar base offset.
    auto origin = addScaledCoordinates(
        pointer.offsets.uniformOffset, inputs[1].advanceDelta->coordinateDeltas,
        pointer.offsets.strides, builder, advance.getLoc());
    if (failed(origin))
      return failure();
    pointer.offsets.uniformOffset = *origin;
    pointer.offsets.completeOffset = {};
    block.coordinateOffsets = std::move(updated->values);
    block.coordinateRanges = std::move(updated->ranges);
  } else {
    return failure();
  }
  if (!validPointerBinding(request.value, pointer, AnalysisOptions{}))
    return failure();
  AnalysisResult result;
  result.pointer = std::move(pointer);
  return result;
}

} // namespace detail
} // namespace mlir::triton::pointer
