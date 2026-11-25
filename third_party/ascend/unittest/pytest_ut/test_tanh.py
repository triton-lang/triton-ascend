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
from triton.backends.ascend.utils import is_compile_on_910_95
import test_common

import torch
import torch_npu


def torch_tanh(x0):
    return torch.tanh(x0)


@triton.jit
def triton_tanh(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = libdevice.tanh(tmp0)
        tl.store(out_ptr0 + (x0), tmp1, None)


@pytest.mark.parametrize('param_list', [
    ['float16', (2, 4096, 8), 2, 32768, 1024],
    ['bfloat16', (2, 4096, 8), 2, 32768, 1024],
    ['float32', (2, 4096, 8), 2, 32768, 1024],
])
def test_tanh(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch_tanh(x0)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_tanh[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


target = triton.runtime.driver.active.get_current_target()


@triton.jit
def triton_tanh_negative_tail(in_ptr0, out_ptr0, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x0 = tl.load(in_ptr0 + offsets)
    tmp = libdevice.tanh(x0)
    tl.store(out_ptr0 + offsets, tmp)


@pytest.mark.skipif(not is_compile_on_910_95(target.arch), reason="Currently unsupported on A2A3")
def test_tanh_negative_tail():
    # Near negative saturation, 1 + tanh(x) approaches zero; cancellation makes
    # a one-ULP tanh error visible. Compare raw FP32 bits instead of a tolerance.
    x0 = torch.linspace(-8.0, -9.0, 4096, dtype=torch.float32).npu()
    y_ref = torch_tanh(x0).cpu()

    y_cal = torch.empty_like(x0)
    triton_tanh_negative_tail[(1, )](x0, y_cal, N=x0.numel())

    y_cal_bits = y_cal.cpu().view(torch.int32)
    y_ref_bits = y_ref.view(torch.int32)
    assert torch.equal(y_cal_bits, y_ref_bits), \
        "tanh(x) is not bitwise identical to torch_npu for negative tail input"
