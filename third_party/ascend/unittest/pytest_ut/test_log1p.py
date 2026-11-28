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
import torch
import test_common
import triton.language.extra.cann.libdevice as libdevice


def torch_log1p(x0, x1):
    res = x0 + torch.log1p(x1)
    return res


@triton.jit
def triton_log1p(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = tmp0 + libdevice.log1p(tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)


@pytest.mark.parametrize('param_list', [
    ['float32', (2, 4096, 8), 2, 32768, 1024],
])
def test_log1p(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    torch_res = torch_log1p(x0, x1)
    # triton结果
    triton_res = test_common.generate_tensor(shape, dtype).npu()
    triton_log1p[ncore, 1, 1](x0, x1, triton_res, x0.numel(), xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)


def test_log1p_boundary():
    vals = torch.tensor([1e-8, -1e-8, 1e-6, -1e-6, 1e-3, -1e-3, 0.0, -0.0], dtype=torch.float32)
    y_ref = torch.log1p(vals)
    n = vals.numel()
    zero = torch.zeros(n, dtype=torch.float32).npu()
    x0 = vals.npu()
    y_cal = torch.zeros_like(x0).npu()
    triton_log1p[1, 1, 1](zero, x0, y_cal, n, n, n)
    y_cal = y_cal.cpu()
    # Tight relative tolerance (atol=0): a zero result for a tiny input such as
    # log1p(1e-8) == 0 must not be accepted as matching the ~1e-8 reference.
    torch.testing.assert_close(y_cal, y_ref, rtol=1e-3, atol=0.0)
    # log1p(-0.0) must preserve the negative sign of zero.
    assert torch.signbit(y_cal[-1]), "log1p(-0.0) must be -0.0"
