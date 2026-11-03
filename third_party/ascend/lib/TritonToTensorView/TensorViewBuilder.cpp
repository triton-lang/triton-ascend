//===- TensorViewBuilder.cpp - Direct TensorView construction ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/TritonToTensorView/TensorViewBuilder.h"

#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>

using namespace mlir;
namespace tv = mlir::tv;

namespace {

static Value createBaseView(OpBuilder &b, Location loc, Value basePtr,
                            Type elementType, ArrayRef<int64_t> strideStatic,
                            ArrayRef<Value> strideDyn, ArrayRef<Value> extent) {
  basePtr = b.create<triton::PtrToIntOp>(loc, b.getI64Type(), basePtr);
  unsigned rank = strideStatic.size();

  SmallVector<Value> strideOperands;
  SmallVector<int64_t> strideTy;
  for (unsigned d = 0; d < rank; ++d) {
    if (strideStatic[d] == ShapedType::kDynamic) {
      strideOperands.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), strideDyn[d]));
      strideTy.push_back(ShapedType::kDynamic);
    } else {
      strideOperands.push_back(
          b.create<arith::ConstantIndexOp>(loc, strideStatic[d]));
      strideTy.push_back(strideStatic[d]);
    }
  }

  SmallVector<Value> sizeOperands;
  for (unsigned d = 0; d < rank; ++d) {
    if (d < extent.size() && extent[d])
      sizeOperands.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), extent[d]));
    else
      sizeOperands.push_back(b.create<arith::ConstantIndexOp>(
          loc, std::numeric_limits<int64_t>::max()));
  }

  SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
  auto baseTy = tv::TensorViewType::get(dynShape, elementType, strideTy,
                                        /*encoding=*/Attribute());
  return b
      .create<tv::MakeTensorViewOp>(loc, baseTy, basePtr, sizeOperands,
                                    strideOperands)
      .getResult();
}

static tv::TensorViewType getEncodedViewType(Value baseView,
                                             Attribute encoding) {
  auto baseTy = cast<tv::TensorViewType>(baseView.getType());
  return tv::TensorViewType::get(baseTy.getShape(), baseTy.getElementType(),
                                 baseTy.getStrides(), encoding);
}

static SmallVector<Value> castIndices(OpBuilder &b, Location loc,
                                      ValueRange rawIndices,
                                      ArrayRef<int64_t> sparseDims) {
  SmallVector<Value> indices;
  for (auto [d, idx] : llvm::enumerate(rawIndices)) {
    if (llvm::is_contained(sparseDims, d)) {
      indices.push_back(idx);
      continue;
    }
    indices.push_back(
        idx.getType().isIndex()
            ? idx
            : b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx));
  }
  return indices;
}

static SmallVector<Value>
createCoordinateTensors(OpBuilder &b, Location loc,
                        ArrayRef<ArrayRef<int64_t>> coordinates) {
  SmallVector<Value> indices;
  for (ArrayRef<int64_t> coordinate : coordinates) {
    auto type = RankedTensorType::get({static_cast<int64_t>(coordinate.size())},
                                      b.getI64Type());
    auto value = DenseIntElementsAttr::get(type, coordinate);
    indices.push_back(b.create<arith::ConstantOp>(loc, type, value));
  }
  return indices;
}

} // namespace

namespace mlir {
namespace tv {

Type getPtrPointeeType(Type type) {
  if (auto ptr = dyn_cast<tv::PtrType>(type))
    return ptr.getPointeeType();
  if (auto ptr = dyn_cast<triton::PointerType>(type))
    return ptr.getPointeeType();
  return Type();
}

FailureOr<Value> createTensorViewBase(OpBuilder &b, Location loc, Value basePtr,
                                      ValueRange shape, ValueRange strides) {
  if (basePtr.getDefiningOp<triton::AddPtrOp>()) {
    emitError(loc)
        << "TensorView base pointer must not be computed with pointer "
           "addition; represent the complete logical tensor in "
           "`shape`/`strides` and select positions through the encoded "
           "view's `index=` argument";
    return failure();
  }

  unsigned rank = shape.size();
  Type elementType = getPtrPointeeType(basePtr.getType());
  SmallVector<int64_t> strideStatic;
  strideStatic.reserve(rank);
  for (Value stride : strides)
    strideStatic.push_back(
        getConstantIntValue(stride).value_or(ShapedType::kDynamic));
  SmallVector<Value> strideDyn(strides.begin(), strides.end());
  SmallVector<Value> extent(shape.begin(), shape.end());
  return createBaseView(b, loc, basePtr, elementType, strideStatic, strideDyn,
                        extent);
}

Value createPartitionView(OpBuilder &b, Location loc, Value baseView,
                          ArrayRef<int64_t> tile, PaddingValue paddingValue) {
  auto encoding =
      tv::PartitionViewAttr::get(b.getContext(), tile, paddingValue);
  auto resultType = getEncodedViewType(baseView, encoding);
  return b.create<tv::MakePartitionViewOp>(loc, resultType, baseView)
      .getResult();
}

Value createStridedView(OpBuilder &b, Location loc, Value baseView,
                        ArrayRef<int64_t> tile,
                        ArrayRef<int64_t> traversalStrides,
                        PaddingValue paddingValue) {
  auto encoding = tv::StridedViewAttr::get(b.getContext(), tile,
                                           traversalStrides, paddingValue);
  auto resultType = getEncodedViewType(baseView, encoding);
  return b.create<tv::MakeStridedViewOp>(loc, resultType, baseView).getResult();
}

Value createGatherScatterView(OpBuilder &b, Location loc, Value baseView,
                              ArrayRef<int64_t> tile,
                              ArrayRef<int64_t> sparseDims,
                              PaddingValue paddingValue) {
  auto encoding = tv::GatherScatterViewAttr::get(b.getContext(), tile,
                                                 sparseDims, paddingValue);
  auto resultType = getEncodedViewType(baseView, encoding);
  return b.create<tv::MakeGatherScatterViewOp>(loc, resultType, baseView)
      .getResult();
}

Value tensorViewLoad(OpBuilder &b, Location loc, Value view, ValueRange index,
                     Type resultTy) {
  auto viewType = cast<tv::TensorViewType>(view.getType());
  SmallVector<int64_t> sparseDims =
      tv::getEncodingSparseDims(viewType.getEncoding());
  return b
      .create<tv::ViewLoadOp>(loc, resultTy, view,
                              castIndices(b, loc, index, sparseDims))
      .getResult();
}

void tensorViewStore(OpBuilder &b, Location loc, Value view, ValueRange index,
                     Value value) {
  auto viewType = cast<tv::TensorViewType>(view.getType());
  SmallVector<int64_t> sparseDims =
      tv::getEncodingSparseDims(viewType.getEncoding());
  b.create<tv::ViewStoreOp>(loc, view, value,
                            castIndices(b, loc, index, sparseDims));
}

Value tensorViewPtrLoad(OpBuilder &b, Location loc, Value pointers,
                        ArrayRef<ArrayRef<int64_t>> coordinates,
                        Type resultTy) {
  auto pointerType = cast<RankedTensorType>(pointers.getType());
  auto addressType =
      RankedTensorType::get(pointerType.getShape(), b.getI64Type());
  pointers = b.create<triton::PtrToIntOp>(loc, addressType, pointers);
  return b
      .create<tv::PtrLoadOp>(loc, resultTy, pointers,
                             createCoordinateTensors(b, loc, coordinates))
      .getResult();
}

void tensorViewPtrStore(OpBuilder &b, Location loc, Value pointers, Value value,
                        ArrayRef<ArrayRef<int64_t>> coordinates) {
  auto pointerType = cast<RankedTensorType>(pointers.getType());
  auto addressType =
      RankedTensorType::get(pointerType.getShape(), b.getI64Type());
  pointers = b.create<triton::PtrToIntOp>(loc, addressType, pointers);
  b.create<tv::PtrStoreOp>(loc, pointers, value,
                           createCoordinateTensors(b, loc, coordinates));
}

} // namespace tv
} // namespace mlir
