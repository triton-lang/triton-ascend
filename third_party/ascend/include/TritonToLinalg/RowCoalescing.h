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

#ifndef TRITON_ASCEND_ROW_COALESCING_H
#define TRITON_ASCEND_ROW_COALESCING_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <memory>

// RowCoalescing: fold multiple independent logical rows onto one program.
//
// Pattern (phenomenon, not a specific kernel name): program_id(axis) represents
// a row id, and the row id feeds row-local scalar/gather inputs plus row-local
// vector load/reduce/store work. Rows are independent, so H adjacent row ids can
// be evaluated as a lane vector:
//
//     row  = program_id(axis)
//   ->
//     rows = program_id(axis) * H + arange(0, H)
//
// Scalar row values become tensor<H>, row-local data tensors become
// tensor<H x ...>, and intra-row reductions shift their axis by one. The pass is
// intentionally conservative: it bails if the row-id slice has side effects,
// cross-row communication, unsupported control flow, or an unsafe use of
// num_programs(axis).
namespace RowCoalescing {

void rewriteRowCoalesce(mlir::ModuleOp moduleOp);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRowCoalescingPass();

}  // namespace RowCoalescing

#endif  // TRITON_ASCEND_ROW_COALESCING_H
