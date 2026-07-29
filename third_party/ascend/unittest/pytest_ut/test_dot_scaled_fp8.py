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
import torch
import triton
import triton.language as tl
import numpy as np
import test_common
from triton.backends.ascend.utils import is_compile_on_910_95

# Chained dot_scaled: first matmul with zero init, second accumulates via acc=.
# Mirrors normalize-matmul.mlir @test_mmadmx_chain_no_elemwise_decompose:
#   mmadmxL1(init=true,  outs=empty) -> mmadmxL1(init=false, outs=first)
# No transpose. Result should equal 2 * dot_scaled(lhs, rhs, scales).


def ub_overflow_check(M, N, K):
    bytes_of_dtype = 4
    M32 = ((M + 31) // 32) * 32
    N32 = ((N + 31) // 32) * 32
    cond1 = M32 * K * bytes_of_dtype <= 60 * 1024
    cond2 = K * N32 * bytes_of_dtype <= 60 * 1024
    cond3 = M32 * N32 * bytes_of_dtype <= 256 * 1024
    return cond1 and cond2 and cond3


def torch_dot_scaled(a, b, sa, sb):
    sa = torch.exp2(sa - 127).repeat_interleave(32, dim=1)
    sb = torch.exp2(sb - 127).repeat_interleave(32, dim=1).T
    return torch.matmul(a * sa, b * sb)


@triton.jit
def dot_scale_chain_kernel(
    lhs_ptr,
    lhs_scale_ptr,
    rhs_ptr,
    rhs_scale_ptr,
    out_ptr,
    lhs_format: tl.constexpr,
    rhs_format: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    Midx = tl.arange(0, M)
    Nidx = tl.arange(0, N)
    Kidx = tl.arange(0, K)
    K1idx = tl.arange(0, K // 32)

    lhs = tl.load(lhs_ptr + Midx[:, None] * K + Kidx[None, :])
    lhs_scale = tl.load(lhs_scale_ptr + Midx[:, None] * (K // 32) + K1idx[None, :])
    rhs = tl.load(rhs_ptr + Kidx[:, None] * N + Nidx[None, :])
    rhs_scale = tl.load(rhs_scale_ptr + Nidx[:, None] * (K // 32) + K1idx[None, :])

    # First mmad: init=true equivalent (fresh accumulation).
    first = tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format)
    # Second mmad: init=false equivalent (accumulate into L0C).
    result = tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=first)
    tl.store(out_ptr + Midx[:, None] * N + Nidx[None, :], result)


testlist = [
    (1, 7, 64),
    (15, 128, 64),
    (128, 64, 64),
    (189, 175, 64),
    (1, 1, 128),
    (35, 37, 128),
    (62, 64, 192),
    (31, 27, 256),
]


@pytest.mark.skipif(not is_compile_on_910_95(), reason="only support in A5")
@pytest.mark.parametrize("dtype", ["fp8e4m3", "fp8e5m2"])
@pytest.mark.parametrize("M, N, K", testlist)
def test_2D_scaled_dot_chain(M, N, K, dtype):
    if not ub_overflow_check(M, N, K):
        pytest.skip("UB overflow")

    lhs_format = dtype[3:]
    rhs_format = lhs_format

    lhs = test_common.generate_tensor_fp4_fp8(shape=(M, K), dtype=dtype).npu()
    rhs = test_common.generate_tensor_fp4_fp8(shape=(K, N), dtype=dtype).npu()
    lhs_scale = torch.randint(low=110, high=128, size=(M, K // 32), dtype=torch.int8).npu()
    rhs_scale = torch.randint(low=110, high=128, size=(N, K // 32), dtype=torch.int8).npu()

    lhs_f32 = lhs.cpu().to(torch.float32)
    rhs_f32 = rhs.cpu().to(torch.float32)
    lhs_scale_f32 = lhs_scale.cpu().to(torch.float32)
    rhs_scale_f32 = rhs_scale.cpu().to(torch.float32)

    triton_res = torch.zeros((M, N), dtype=torch.float32).npu()
    dot_scale_chain_kernel[(1, )](lhs, lhs_scale, rhs, rhs_scale, triton_res, lhs_format, rhs_format, M, N, K)

    single = torch_dot_scaled(lhs_f32, rhs_f32, lhs_scale_f32, rhs_scale_f32)
    torch_ref = (single + single).to(torch.float32)

    if dtype == "fp8e4m3":
        torch.testing.assert_close(triton_res.cpu(), torch_ref.cpu(), rtol=125e-03, atol=125e-03, equal_nan=True)
    elif dtype == "fp8e5m2":
        torch.testing.assert_close(triton_res.cpu(), torch_ref.cpu(), rtol=75e-02, atol=75e-02, equal_nan=True)
