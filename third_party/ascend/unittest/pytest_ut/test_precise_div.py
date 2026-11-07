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
from triton.backends.ascend.utils import is_register_based_arch

import torch
import torch_npu
import test_common


def torch_divRn(x0, x1):
    return x0 / x1


@triton.jit
def triton_divRn(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tl.div_rn(tmp0, tmp1)
        tl.store(out_ptr0 + (x0), tmp2, None)


@pytest.mark.parametrize('param_list', [
    ['float32', (2, 4096, 8), 32, 2048, 64],
])
def test_divRn(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype)
    x2 = x1.masked_fill(x1 == 0, 1)
    x2 = x2.npu()
    y_ref = torch_divRn(x0, x2)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_divRn[ncore, 1, 1](x0, x2, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


def test_div_rn_near_integer_boundary():
    arch = triton.runtime.driver.active.get_current_target().arch
    if not is_register_based_arch(arch):
        pytest.skip("Requires a register-based Ascend target")

    block_size = 64
    numerator = torch.full((block_size, ), 0x3FF2B021, dtype=torch.int32).view(torch.float32).npu()
    denominator = torch.full((block_size, ), 0.001, dtype=torch.float32, device="npu")
    result = torch.empty_like(numerator)
    triton_divRn[(1, )](numerator, denominator, result, block_size, block_size)

    # The correctly rounded quotient is one f32 ULP below 1896.0.
    expected = torch.full((block_size, ), 1895.9998779296875, dtype=torch.float32)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
