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

import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common
import triton.language.extra.cann.libdevice as libdevice


def torch_pointwise(x0):
    res = torch.exp(x0)
    return res


@triton.jit
def triton_fast_expf(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp2 = libdevice.fast_expf(tmp0)
        tl.store(out_ptr0 + (x0), tmp2, None)


@pytest.mark.parametrize('param_list', [
    ['float32', (2, 4096, 8), 2, 32768, 1024],
])
def test_case(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch_pointwise(x0)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_fast_expf[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


# Subnormal results are flushed to zero on the vector unit; the half-and-square
# form keeps them representable.
@pytest.mark.parametrize('dtype', ['float32'])
def test_fast_expf_subnormal_result(dtype):
    values = [-88.0, -89.0, -87.5, -87.34]
    x0 = torch.tensor(values, dtype=eval("torch." + dtype)).npu()
    y_ref = torch.exp(x0)
    y_cal = torch.zeros_like(y_ref)
    triton_fast_expf[1, 1, 1](x0, y_cal, x0.shape[0], x0.shape[0])
    assert (y_cal.cpu() > 0).all(), f"fast_expf flushed a subnormal result to zero: {y_cal.cpu()}"
    torch.testing.assert_close(y_cal.cpu(), y_ref.cpu(), rtol=1e-3, atol=0.0)
