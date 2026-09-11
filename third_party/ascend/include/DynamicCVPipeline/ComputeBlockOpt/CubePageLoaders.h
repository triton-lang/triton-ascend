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

#ifndef TRITON_ASCEND_DYNAMIC_CV_PIPELINE_CUBE_PAGE_LOADERS_H
#define TRITON_ASCEND_DYNAMIC_CV_PIPELINE_CUBE_PAGE_LOADERS_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::CVPipeline {

struct CubePageLoader {
  tensor::InsertSliceOp insert;
  bufferization::ToTensorOp tensor;
  memref::CopyOp copy;
  // The private page buffer, its views, and its zero initialization.
  SmallVector<Operation *> bufferOps;
};

// Recognize a complete, ordered row concatenation of private GM page loads.
// Shared by classification and materialization: a chain must be lowerable to
// direct GM-to-L1 DMA before it can be assigned to CUBE.
std::optional<SmallVector<CubePageLoader>>
getCubePageLoaders(tensor::InsertSliceOp root);

} // namespace mlir::CVPipeline

#endif
