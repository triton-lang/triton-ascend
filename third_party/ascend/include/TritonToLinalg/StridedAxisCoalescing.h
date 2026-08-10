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

#ifndef TRITON_ASCEND_STRIDED_AXIS_COALESCING_H
#define TRITON_ASCEND_STRIDED_AXIS_COALESCING_H

#include "triton/Dialect/Triton/IR/Dialect.h"

// StridedAxisCoalescing: a distinct optimisation path from
// StridedLoadStoreRewrite.
//
// For the FLA "H-axis split" kernels (chunk_local_cumsum et al.) each program
// handles one (b,h) and reads its column along T with a stride-H (>1) access.
// StridedLoadStoreRewrite lowers such loads to a SIMT indirect gather; this
// pass instead folds the H axis into the inner lane: it rewrites the
//   strided ih-base load(s) -> lane-safe compute subgraph -> strided ih-base
//   store(s)
// chain into a 2D contiguous [BT,H] tile (H as a parallel inner lane), turning
// the per-element strided DMA into a contiguous one. The bishengir
// AutoBlockifyParallelLoop pass then divides the persistent-loop trip count by
// the recorded `hacc.coalesce_factor`.
namespace StridedAxisCoalescing {

// Default-on rewrite (generalised liftTo2D): fold the strided ih-base load ->
// lane-safe-compute subgraph -> strided ih-base store chain into a 2D
// contiguous [BT,H] tile (H as a parallel inner lane). Covers forward/reverse
// cumsum, scale, dtype casts and any elementwise body; bails (leaving the
// original path) on anything unsafe (e.g. a tt.dot in the subgraph).
void rewriteStridedAxisCoalesce(mlir::ModuleOp moduleOp);

} // namespace StridedAxisCoalescing

#endif // TRITON_ASCEND_STRIDED_AXIS_COALESCING_H
