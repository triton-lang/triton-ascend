# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Coverage for the mask-implied address clamp fold of TritonToUnstructure.

``tl.maximum(off, 0)`` keeps a segmented gather inside the source allocation,
but it also hides the affine structure of the row offset, so the offset
analysis classifies the access as unstructured and lowers it to one row per
iteration. When the load mask already implies ``off >= 0``, the clamp is
irrelevant for every masked-in lane and is folded away, which restores the
structured (single tile) lowering.

The second kernel covers the safety side: when the mask says nothing about the
sign of ``off``, the clamp changes both the addresses and the loaded values and
therefore has to survive.
"""

import re

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _clamp_implied_kernel(out_ptr, src_ptr, start_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.arange(0, ROWS)
    start = tl.load(start_ptr)
    off = rows - start
    in_seg = off >= 0
    value = tl.load(src_ptr + tl.maximum(off, 0)[:, None] * COLS + tl.arange(0, COLS)[None, :], mask=in_seg[:, None],
                    other=0.0)
    tl.store(out_ptr + rows[:, None] * COLS + tl.arange(0, COLS)[None, :], value)


@triton.jit
def _clamp_unrelated_mask_kernel(out_ptr, src_ptr, start_ptr, n_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.arange(0, ROWS)
    start = tl.load(start_ptr)
    n = tl.load(n_ptr)
    off = rows - start
    value = tl.load(src_ptr + tl.maximum(off, 0)[:, None] * COLS + tl.arange(0, COLS)[None, :],
                    mask=(rows < n)[:, None], other=0.0)
    tl.store(out_ptr + rows[:, None] * COLS + tl.arange(0, COLS)[None, :], value)


def _scalar(value):
    return torch.tensor(value, dtype=torch.int32, device="npu")


def _i32_clamp_count(asm):
    return len(re.findall(r"arith\.maxsi %\S+, %c0_i32 : i32", asm))


@pytest.mark.parametrize("rows, cols, start", [
    (8, 64, 5),
    (8, 64, 1),
    (8, 64, 0),
    (8, 64, 8),
    (8, 64, 9),
    (4, 32, 2),
    (16, 64, 10),
])
def test_mask_implied_clamp_folds_to_structured_access(rows, cols, start):
    src = torch.arange(rows * cols, dtype=torch.float32, device="npu").reshape(rows, cols)
    out = torch.full_like(src, -7)

    kernel = _clamp_implied_kernel[(1, )](out, src, _scalar(start), ROWS=rows, COLS=cols)

    expected = torch.zeros_like(src)
    if start < rows:
        expected[start:] = src[:rows - start]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    asm = kernel.asm["ttadapter"]
    # The clamp is gone, the row offset stays affine, and the gather keeps its
    # tile shape instead of being split into a loop of single-row accesses.
    assert _i32_clamp_count(asm) == 0
    assert f"sizes: [{rows}, {cols}]" in asm
    assert f"sizes: [1, {cols}]" not in asm
    assert "scf.for" not in asm


@pytest.mark.parametrize("rows, cols, start, n", [
    (8, 64, 5, 8),
    (8, 64, 5, 3),
    (4, 32, 2, 4),
])
def test_clamp_is_kept_when_mask_does_not_imply_it(rows, cols, start, n):
    src = torch.arange(rows * cols, dtype=torch.float32, device="npu").reshape(rows, cols)
    out = torch.full_like(src, -7)

    kernel = _clamp_unrelated_mask_kernel[(1, )](out, src, _scalar(start), _scalar(n), ROWS=rows, COLS=cols)

    expected = torch.zeros_like(src)
    for row in range(min(n, rows)):
        expected[row] = src[max(row - start, 0)]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    asm = kernel.asm["ttadapter"]
    # `rows < n` does not imply `off >= 0`: here the clamp is what keeps the
    # masked-in lanes inside the allocation, so folding it would be wrong.
    assert _i32_clamp_count(asm) >= 1
