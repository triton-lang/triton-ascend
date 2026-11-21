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
import test_common

import torch
import torch_npu


def standard_unary(x0, dtype):
    res = torch.log(x0)
    return res


@triton.jit
def triton_elementwise_unary(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    ret = tl.math.log(x)
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)


shapes = [
    (3, 32),
    (-32, 32),
    (37, 64),
    (-256, 256),
    (781, 1024),
]


@pytest.mark.parametrize('dtype,sigtype', [
    (torch.float32, 'float32'),
])
@pytest.mark.parametrize('N,NUMEL', shapes)
def test_elementwsie_common(dtype, sigtype, N, NUMEL):
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    x0 = test_common.generate_tensor(shape=(N, ), dtype=sigtype)

    ans = standard_unary(x0, dtype)
    x0 = x0.npu()

    out = torch.zeros((N, ), dtype=dtype).npu()
    triton_elementwise_unary[1, 1, 1](x0, out, N=N, NUMEL=NUMEL, debug=True)

    test_common.validate_cmp(sigtype, out, ans)


# Subnormal arguments are flushed to zero on the vector unit, so log returned
# -inf; the rescaled path keeps the finite result.
@pytest.mark.parametrize('sigtype', ['float32'])
def test_log_subnormal_inputs(sigtype):
    values = [1e-38, 1e-40, 1.4e-45, 1e-36]
    x0 = torch.tensor(values, dtype=torch.float32)
    ans = torch.log(x0)
    x0 = x0.npu()
    out = torch.zeros((len(values), ), dtype=torch.float32).npu()
    triton_elementwise_unary[1, 1, 1](x0, out, N=len(values), NUMEL=len(values))
    assert torch.isfinite(out.cpu()).all(), f"log of subnormal returned non-finite: {out.cpu()}"
    torch.testing.assert_close(out.cpu(), ans, rtol=1e-6, atol=0.0)
