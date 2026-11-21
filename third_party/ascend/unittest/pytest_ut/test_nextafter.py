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

import pytest

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import test_common

import torch
import torch_npu
import numpy as np


def torch_nextafter(x0, x1):
    return torch.nextafter(x0, x1)


@triton.jit
def triton_nextafter(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = libdevice.nextafter(tmp0, tmp1)
        tl.store(out_ptr0 + (x0), tmp2, None)


@pytest.mark.parametrize('param_list', [
    ['float32', (2, 4096, 8), 2, 32768, 1024],
    ['float16', (2, 4096, 8), 2, 32768, 1024],
])
def test_nextafter(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0_ref = test_common.generate_tensor(shape, dtype)
    x1_ref = test_common.generate_tensor(shape, dtype)
    x0 = x0_ref.npu()
    x1 = x1_ref.npu()
    y_ref = torch_nextafter(x0_ref, x1_ref)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_nextafter[ncore, 1, 1](x0, x1, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@triton.jit
def triton_nextafter_zero(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK
    idx = offset + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + idx)
    y = tl.load(y_ptr + idx)
    tl.store(out_ptr + idx, libdevice.nextafter(x, y))


def _nextafter_zero_cases():
    """Every (sign of x, direction of y) combination around zero.

    ``x_gt_0`` is False for both ``+0.0`` and ``-0.0``, so a sign taken from
    ``x`` is wrong for all of them; the expected value follows ``y``.
    """
    tiny = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    inputs = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0]
    targets = [1.0, 1.0, -1.0, -1.0, tiny, tiny, -tiny, -tiny]
    return inputs, targets


@pytest.mark.parametrize('dtype', ['float32'])
def test_nextafter_zero_sign(dtype):
    inputs, targets = _nextafter_zero_cases()
    x_ref = torch.tensor(inputs, dtype=eval('torch.' + dtype))
    y_ref = torch.tensor(targets, dtype=eval('torch.' + dtype))
    expect = torch_nextafter(x_ref, y_ref)

    n = len(inputs)
    x = x_ref.npu()
    y = y_ref.npu()
    out = torch.zeros(n, dtype=eval('torch.' + dtype)).npu()
    triton_nextafter_zero[(1, )](x, y, out, BLOCK=n)

    # Compare the raw bit patterns so the sign of a zero result is checked too.
    assert torch.equal(out.cpu().view(torch.int32), expect.view(torch.int32)), (
        f"nextafter zero sign mismatch: got {out.cpu()}, expect {expect}")


@pytest.mark.parametrize('dtype', ['float32'])
def test_nextafter_equal_zero_and_nonzero(dtype):
    """``x == y`` must return ``y`` unchanged, including ``-0.0`` vs ``+0.0``."""
    inputs = [0.0, -0.0, 1.0, -1.0, 2.5, 2.5]
    targets = [-0.0, 0.0, 1.0, -1.0, 2.5, 2.5]
    x_ref = torch.tensor(inputs, dtype=eval('torch.' + dtype))
    y_ref = torch.tensor(targets, dtype=eval('torch.' + dtype))
    expect = torch_nextafter(x_ref, y_ref)

    n = len(inputs)
    x = x_ref.npu()
    y = y_ref.npu()
    out = torch.zeros(n, dtype=eval('torch.' + dtype)).npu()
    triton_nextafter_zero[(1, )](x, y, out, BLOCK=n)

    assert torch.equal(out.cpu().view(torch.int32), expect.view(torch.int32)), (
        f"nextafter equal-input mismatch: got {out.cpu()}, expect {expect}")
