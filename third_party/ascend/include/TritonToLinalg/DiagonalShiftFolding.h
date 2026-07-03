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

#ifndef TRITON_ASCEND_DIAGONAL_SHIFT_FOLDING_H
#define TRITON_ASCEND_DIAGONAL_SHIFT_FOLDING_H

#include "triton/Dialect/Triton/IR/Dialect.h"

// DiagonalShiftFolding: replace O(N^2) diagonal-select-reduce patterns with
// O(N) subtraction using the cumulative sum identity.
//
// Pattern: a cumsum result is broadcast to an NxN matrix, a diagonal mask
// selects exactly one element per row, and a row-reduce sums it -- effectively
// shifting the vector by one position. This is common in segmented-sum /
// cumsum kernels that compute inter-chunk corrections.
//
//   scan_result = tt.scan(scan_input, axis=0, addf)
//   exp         = tt.expand_dims(scan_result, axis=0)
//   bcast       = tt.broadcast(exp)                    // NxN, rows identical
//   mask        = cmpi eq(row_idx, col_idx +/- 1)      // unit diagonal
//   sel         = arith.select(mask, bcast, 0.0)
//   result      = tt.reduce(sel, axis=1, addf)
//
// By the cumulative sum identity:
//   fwd_cumsum(x)[i] - x[i] = fwd_cumsum(x)[i-1]   (left shift, fwd scan)
//   rev_cumsum(x)[i] - x[i] = rev_cumsum(x)[i+1]   (right shift, rev scan)
//
// Replacement:
//   result = arith.subf(scan_result, scan_input)
//
// The pass bails (leaves IR untouched) whenever ANY condition is not met:
// wrong reduce axis/combine, non-diagonal mask, shift != 1, scan direction
// mismatch, non-square broadcast, non-zero false value, multi-result scan, etc.
namespace DiagonalShiftFolding {

void rewriteDiagonalShiftFold(mlir::ModuleOp moduleOp);

}  // namespace DiagonalShiftFolding

#endif  // TRITON_ASCEND_DIAGONAL_SHIFT_FOLDING_H
