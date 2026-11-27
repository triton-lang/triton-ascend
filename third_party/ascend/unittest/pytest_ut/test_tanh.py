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
def triton_tanh_high_precision(in_ptr0, out_ptr0, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x0 = tl.load(in_ptr0 + offsets)
    tmp = libdevice.tanh(x0, high_precision=True)
    tl.store(out_ptr0 + offsets, tmp)


@pytest.mark.skipif(not is_compile_on_910_95(target.arch), reason="Currently unsupported on A2A3")
def test_tanh_high_precision():
    x0 = test_common.generate_tensor((1024, ), 'float32')
    # Sample near zero, across the central range, and densely near both saturation tails.
    x0[:256].mul_(0.01)
    x0[256:512].mul_(4.0)
    x0[512:768].mul_(0.25).sub_(8.5)
    x0[768:].mul_(0.25).add_(8.5)
    max_fp32 = torch.finfo(x0.dtype).max
    # Issue #3356: tanh rounds to the input for the smallest FP32 subnormals of either sign.
    subnormal_bits = torch.tensor([0x00000001, -0x7fffffff], dtype=torch.int32)
    x0[:2].view(torch.int32).copy_(subnormal_bits)
    x0[2:11] = x0.new_tensor([
        0.0,
        -0.6,
        0.6,
        -9.010913848876953,
        9.010913848876953,
        -max_fp32,
        max_fp32,
        -float('inf'),
        float('inf'),
    ])
    x0 = x0.npu()
    y_ref = torch_tanh(x0)

    y_cal = torch.empty_like(x0)
    triton_tanh_high_precision[(1, )](x0, y_cal, N=x0.numel())
    torch.testing.assert_close(y_cal.cpu(), y_ref.cpu(), rtol=0, atol=0)
