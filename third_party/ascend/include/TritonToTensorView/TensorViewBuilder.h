//===- TensorViewBuilder.h - Direct TensorView construction -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_TENSORVIEW_BUILDER_H
#define TRITON_ASCEND_TENSORVIEW_BUILDER_H

#include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace tv {

// Pointee type of a `!tv.ptr<T>` or `!tt.ptr<T>`, or null otherwise.
Type getPtrPointeeType(Type type);

// Build an unencoded view from a scalar pointer, shape, and element strides.
FailureOr<Value> createTensorViewBase(OpBuilder &b, Location loc, Value basePtr,
                                      ValueRange shape, ValueRange strides);

// Attach an explicit access encoding to a base view.
Value createPartitionView(OpBuilder &b, Location loc, Value baseView,
                          ArrayRef<int64_t> tile, PaddingValue paddingValue);
Value createStridedView(OpBuilder &b, Location loc, Value baseView,
                        ArrayRef<int64_t> tile,
                        ArrayRef<int64_t> traversalStrides,
                        PaddingValue paddingValue);
Value createGatherScatterView(OpBuilder &b, Location loc, Value baseView,
                              ArrayRef<int64_t> tile,
                              ArrayRef<int64_t> sparseDims,
                              PaddingValue paddingValue);

// Access a tile through an already encoded view.
Value tensorViewLoad(OpBuilder &b, Location loc, Value view, ValueRange index,
                     Type resultTy);
void tensorViewStore(OpBuilder &b, Location loc, Value view, ValueRange index,
                     Value value);

// Access an N-D tensor of pointers at zipped compile-time coordinates.
Value tensorViewPtrLoad(OpBuilder &b, Location loc, Value pointers,
                        ArrayRef<ArrayRef<int64_t>> coordinates, Type resultTy);
void tensorViewPtrStore(OpBuilder &b, Location loc, Value pointers, Value value,
                        ArrayRef<ArrayRef<int64_t>> coordinates);

} // namespace tv
} // namespace mlir

#endif // TRITON_ASCEND_TENSORVIEW_BUILDER_H
