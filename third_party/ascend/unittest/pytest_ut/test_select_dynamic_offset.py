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

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _select_dynamic_offset_kernel(true_ptr, false_ptr, bounds_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
                                  STEPS: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, COLS)
    offsets = tl.arange(0, ROWS)[:, None] * COLS + cols[None, :]
    selected = tl.load(false_ptr + offsets)
    for step in range(STEPS):
        lo = tl.load(bounds_ptr + (pid * STEPS + step) * 2)
        hi = tl.load(bounds_ptr + (pid * STEPS + step) * 2 + 1)
        true_value = tl.load(true_ptr + step * ROWS * COLS + offsets)
        if MODE == "head":
            mask = cols < hi
        elif MODE == "tail":
            mask = cols >= lo
        elif MODE == "point":
            mask = cols == lo
        else:
            mask = (cols >= lo) & (cols < hi)
        selected = tl.where(mask[None, :], true_value, selected)
    tl.store(out_ptr + pid * ROWS * COLS + offsets, selected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("mode", ["head", "tail", "middle", "point"])
def test_select_dynamic_offset(dtype, rows, mode):
    # Bounds are loaded from memory without alignment hints. Include starts
    # around 32B boundaries, empty/disjoint intervals, and out-of-range points.
    cols, steps = 64, 3
    cases = torch.tensor([
        [-3, 5],
        [-1, 7],
        [0, 0],
        [1, 9],
        [7, 15],
        [8, 16],
        [9, 17],
        [63, 71],
        [64, 72],
        [67, 75],
        [7, 7],
        [9, 3],
    ], dtype=torch.int32)
    bounds = torch.stack([cases.roll(step, 0) for step in range(steps)], dim=1)
    generator = torch.Generator().manual_seed(13)
    true_value = torch.randint(-100, 100, (steps, rows, cols), generator=generator).to(dtype)
    false_value = torch.randint(-100, 100, (rows, cols), generator=generator).to(dtype)
    expected = false_value.expand(len(cases), rows, cols).clone()
    offsets = torch.arange(cols)
    for case in range(len(cases)):
        for step in range(steps):
            lo, hi = bounds[case, step]
            if mode == "head":
                mask = offsets < hi
            elif mode == "tail":
                mask = offsets >= lo
            elif mode == "point":
                mask = offsets == lo
            else:
                mask = (offsets >= lo) & (offsets < hi)
            expected[case] = torch.where(mask[None, :], true_value[step], expected[case])

    output = torch.empty(expected.shape, dtype=dtype, device="npu")
    _select_dynamic_offset_kernel[(len(cases), )](
        true_value.npu(),
        false_value.npu(),
        bounds.npu(),
        output,
        ROWS=rows,
        COLS=cols,
        STEPS=steps,
        MODE=mode,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
