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

#ifndef TRITON_ASCEND_TILE_CHUNK_COALESCING_H
#define TRITON_ASCEND_TILE_CHUNK_COALESCING_H

#include "triton/Dialect/Triton/IR/Dialect.h"

// TileChunkCoalescing: fix the "tiny per-tile DMA" pathology of kernels that
// over-decompose a contiguous problem axis onto the (outermost) launch grid.
//
// Pattern (the "phenomenon", not a specific kernel): the *last* program-id axis
// `a` is used purely as an independent tile index --
//     blk      = program_id(a) * T          (T = constexpr tile length)
//     offs     = blk + arange(0, T)
//     mask     = offs < BOUND                (BOUND = constexpr problem length)
//     ... load(offs) -> elementwise / intra-tile scan|reduce -> store(offs) ...
// with (1) `program_id(a)` flowing only into address / mask integer arithmetic
// (never into tensor *data*), (2) no cross-tile dependency (every scan/reduce
// runs along an intra-tile axis), and (3) tiles laid out contiguously so that H
// adjacent tiles form one contiguous block.
//
// When T is small the per-tile load/store are tiny DMAs (e.g. 16xf32 = 64B),
// wasting HBM bandwidth on transfer overhead. This pass folds H adjacent tiles
// into a single program by prepending an H lane to the whole load->store
// subgraph: the T-element loads/stores become one contiguous H*T block, scans
// and reduces keep their (now +1) intra-tile axis, and `program_id(a)` is
// replaced by the lane vector `program_id(a)*H + arange(0,H)`. H is chosen so
// the merged block is >= kMinContigBytes and the lifted footprint stays within
// the UB budget, and so that H divides the tile count.
//
// The pass records `hacc.coalesce_factor = H` and `hacc.coalesce_axis = a` on
// the module; the TA compiler exports them to launch metadata, and the host
// launcher divides grid[a] by H. The pass is a no-op (bails) whenever the
// pattern or the safety conditions above do not hold, including unmasked
// kernels whose runtime tile count cannot be proven from IR.
namespace TileChunkCoalescing {

void rewriteTileChunkCoalesce(mlir::ModuleOp moduleOp);

}  // namespace TileChunkCoalescing

#endif  // TRITON_ASCEND_TILE_CHUNK_COALESCING_H
