# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#
import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _reverse_add(X, Y, Z, ROWS: tl.constexpr, COLS: tl.constexpr, STEP: tl.constexpr, NEGATIVE: tl.constexpr):
    i = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
    mask = (i >= 0) & (i < ROWS * COLS)
    base = (ROWS * COLS - 1) * STEP if NEGATIVE else 0
    stride = -STEP if NEGATIVE else STEP
    x = tl.load(X + base + i * stride, mask=mask, other=0.0)
    y = tl.load(Y + i, mask=mask, other=0.0)
    tl.store(Z + i, x + y)


@pytest.mark.parametrize("rows,cols", [(64, 64), (8, 16), (1, 64), (64, 1)])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("negative", [False, True])
def test_negative_stride_simd(rows, cols, step, negative):
    # #3231: reversing a two-dimensional load must retain the innermost
    # stride when the backend collapses the memory transfer to one dimension.
    count = rows * cols
    source = torch.arange(count * step, dtype=torch.float32, device="npu")
    addend = torch.ones(count, dtype=torch.float32, device="npu")
    output = torch.empty_like(addend)
    _reverse_add[(1, )](source, addend, output, rows, cols, step, negative, compile_mode="simd")
    expected = source.cpu()[::step]
    if negative:
        expected = expected.flip(0)
    torch.testing.assert_close(output.cpu(), expected + 1, rtol=0, atol=0)


@triton.jit
def _reverse_with_offset(X, Z, base, VALID_ROWS: tl.constexpr):
    i = tl.arange(0, 8)[:, None] * 16 + tl.arange(0, 16)[None, :]
    block = tl.program_id(0) * 128
    value = tl.load(X + base + block + 127 - i, mask=i < VALID_ROWS * 16, other=0.0)
    tl.store(Z + block + i, value)


@pytest.mark.parametrize("valid_rows", [7, 8])
def test_negative_stride_simd_offset_and_mask(valid_rows):
    # The view offset includes both a runtime argument and the program ID.
    source = torch.arange(17 + 2 * 128, dtype=torch.float32, device="npu")
    output = torch.empty(256, dtype=torch.float32, device="npu")
    _reverse_with_offset[(2, )](source, output, 17, valid_rows, compile_mode="simd")
    expected = source.cpu()[17:].reshape(2, 128).flip(1)
    expected[:, valid_rows * 16:] = 0
    torch.testing.assert_close(output.cpu().reshape(2, 128), expected, rtol=0, atol=0)
