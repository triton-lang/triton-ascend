# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import test_common
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.backends.ascend.utils import is_register_based_arch


def _pack_int4(values):
    # Pack pairs along the last axis: first value in the low nibble, second in the high nibble.
    if values.shape[-1] % 2:
        values = torch.nn.functional.pad(values, (0, 1), value=0)
    low = values[:, 0::2] & 0x0F
    high = values[:, 1::2] & 0x0F
    return (low | (high << 4)).to(torch.int8)


@triton.jit
def _dot_s4_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # K and BLOCK_K count logical int4 values and must both be even.
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_packed = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, 2 * BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        # A packs along K: advance by k_start/2 bytes, keeping the full row stride K/2.
        a_k = k_start // 2 + offs_k_packed
        a_ptrs = a_ptr + offs_m[:, None] * (K // 2) + a_k[None, :]
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (a_k[None, :] < K // 2), other=0)

        # B packs along N: advance by k_start rows, each containing ceil(N/2) bytes.
        b_k = k_start + offs_k
        b_ptrs = b_ptr + b_k[:, None] * tl.cdiv(N, 2) + offs_n_packed[None, :]
        b = tl.load(b_ptrs, mask=(b_k[:, None] < K) & (offs_n_packed[None, :] < tl.cdiv(N, 2)), other=0)

        # [BLOCK_M, BLOCK_K/2] @ [BLOCK_K, BLOCK_N] adds a [BLOCK_M, 2*BLOCK_N] product to the accumulator.
        accumulator = al.dot_s4(a, b, accumulator)

    offs_n = tl.program_id(1) * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def test_dot_s4():
    if is_register_based_arch(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("dot_s4 requires a memory-based Ascend architecture")

    m, n, k = 33, 32, 160
    block_m = 16
    block_n = 8
    block_k = 64
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        a = test_common.generate_tensor((m, k), "int8") % 16 - 8
        b = test_common.generate_tensor((k, n), "int8") % 16 - 8
    expected = a.to(torch.int32) @ b.to(torch.int32)

    a_packed = _pack_int4(a).npu()
    b_packed = _pack_int4(b).npu()
    actual = torch.empty((m, n), dtype=torch.int32, device="npu")
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, 2 * block_n))
    _dot_s4_kernel[grid](a_packed, b_packed, actual, m, n, k, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
