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


def torch_atanh(x0):
    return torch.atanh(x0)


@triton.jit
def triton_atanh(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = libdevice.atanh(tmp0)
        tl.store(out_ptr0 + (x0), tmp1, None)


@pytest.mark.parametrize('param_list', [
    ['float32', (2, 4096, 8), 2, 32768, 1024],
    ['float16', (2, 4096, 8), 2, 32768, 1024],
    ['bfloat16', (2, 4096, 8), 2, 32768, 1024],
])
def test_atanh_common(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu().clamp(-0.99, 0.99)
    y_ref = torch_atanh(x0)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_atanh[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


input_vals = [-1, 1]


@pytest.mark.parametrize('param_list', [
    ['float32', (2, 4096, 8), 2, 32768, 1024],
    ['float16', (2, 4096, 8), 2, 32768, 1024],
    ['bfloat16', (2, 4096, 8), 2, 32768, 1024],
])
@pytest.mark.parametrize('input_val', input_vals)
def test_atanh_special(param_list, input_val):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = torch.full(shape, input_val, dtype=eval('torch.' + dtype)).npu()
    y_ref = torch_atanh(x0)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_atanh[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


# Tiny inputs: 0.5*(log(1+x) - log(1-x)) cancels and used to return 0.
@pytest.mark.parametrize('dtype', ['float32'])
def test_atanh_tiny_inputs(dtype):
    values = [1e-8, -1e-8, 1e-6, -1e-6, 1e-4, -1e-4, 0.0, -0.0]
    x0 = torch.tensor(values, dtype=eval("torch." + dtype)).npu()
    y_ref = torch.atanh(x0)
    y_cal = torch.zeros_like(y_ref)
    triton_atanh[1, 1, 1](x0, y_cal, x0.shape[0], x0.shape[0])
    torch.testing.assert_close(y_cal.cpu(), y_ref.cpu(), rtol=1e-6, atol=0.0, equal_nan=True)
    assert torch.equal(torch.signbit(y_cal.cpu()), torch.signbit(y_ref.cpu()))


# Near +/-1 the two logarithms straddle 0; the sign-symmetric form keeps both
# the positive and the negative branch accurate.
@pytest.mark.parametrize('dtype', ['float32'])
def test_atanh_near_one(dtype):
    values = [0.99, -0.99, 0.999999, -0.999999]
    x0 = torch.tensor(values, dtype=eval("torch." + dtype)).npu()
    y_ref = torch.atanh(x0)
    y_cal = torch.zeros_like(y_ref)
    triton_atanh[1, 1, 1](x0, y_cal, x0.shape[0], x0.shape[0])
    torch.testing.assert_close(y_cal.cpu(), y_ref.cpu(), rtol=1e-5, atol=0.0)
