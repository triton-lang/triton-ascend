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


def torch_expm1(x0):
    return torch.expm1(x0)


@triton.jit
def triton_expm1(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = libdevice.expm1(tmp0)
        tl.store(out_ptr0 + (x0), tmp1, None)


@pytest.mark.skip(reason="expm1 failed sometimes, wait for fix")
@pytest.mark.parametrize('param_list', [
    ['float32', (2, 4096, 8), 2, 32768, 1024],
])
def test_expm1(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0_ref = test_common.generate_tensor(shape, dtype)
    # error gains when input is large; expm1 used to handle small input when exp(x) close to 1
    x0_ref[x0_ref > 1] = 1
    y_ref = torch_expm1(x0_ref)

    x0 = x0_ref.npu()
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_expm1[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


# exp(x) - 1 rounds to exactly 0 for |x| below half an ulp of 1; the small
# branch evaluates a Taylor series instead.
@pytest.mark.parametrize('dtype', ['float32'])
def test_expm1_tiny_inputs(dtype):
    values = [1e-8, -1e-8, 1e-6, -1e-6, 1e-3, -1e-3, 0.009, -0.009, 0.0, -0.0]
    x0 = torch.tensor(values, dtype=eval("torch." + dtype)).npu()
    y_ref = torch.expm1(x0)
    y_cal = torch.zeros_like(y_ref)
    triton_expm1[1, 1, 1](x0, y_cal, x0.shape[0], x0.shape[0])
    torch.testing.assert_close(y_cal.cpu(), y_ref.cpu(), rtol=1e-5, atol=0.0, equal_nan=True)
    assert not torch.equal(y_cal.cpu()[0:6], torch.zeros(6)), f"expm1 flushed {y_cal.cpu()}"


# The series has to blend into exp(x) - 1 without a discontinuity at the cut.
@pytest.mark.parametrize('dtype', ['float32'])
def test_expm1_cut_point(dtype):
    values = [0.00999, 0.010001, -0.00999, -0.010001]
    x0 = torch.tensor(values, dtype=eval("torch." + dtype)).npu()
    y_ref = torch.expm1(x0)
    y_cal = torch.zeros_like(y_ref)
    triton_expm1[1, 1, 1](x0, y_cal, x0.shape[0], x0.shape[0])
    torch.testing.assert_close(y_cal.cpu(), y_ref.cpu(), rtol=1e-5, atol=0.0)
