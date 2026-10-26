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

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95


@triton.jit
def _fp8_indirect_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                                stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                BLOCK_SIZE_K: tl.constexpr, low_precision_acc: tl.constexpr,
                                num_stages: tl.constexpr = 3):
    # Preserve the community kernel's modulo addressing and loop-carried
    # pointers: the large B load must not materialize a full i64 address tile.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=num_stages):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator, max_num_imprecise_acc=low_precision_acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)


def _decode_fp8_reference(bits, dtype):
    """Evaluate sign, exponent and significand independently of Triton."""
    codes = np.asarray(bits).view(np.uint8)
    fraction_bits, bias = {
        "float8e4b15": (3, 15),
        "float8e4nv": (3, 7),
        "float8e5": (2, 15),
    }[dtype]
    fraction = (codes % (1 << fraction_bits)).astype(np.float64) / (1 << fraction_bits)
    exponent = ((codes & 0x7f) // (1 << fraction_bits)).astype(np.int32)
    magnitude = np.ldexp(np.where(exponent == 0, fraction, 1 + fraction),
                         np.where(exponent == 0, 1 - bias, exponent - bias))
    if dtype == "float8e4nv":
        magnitude = np.where((codes & 0x7f) == 0x7f, np.nan, magnitude)
    elif dtype == "float8e5":
        magnitude = np.where(exponent == 31, np.where(fraction == 0, np.inf, np.nan), magnitude)
    # E4M3B15 uses every encoding for a finite value, including 0x7f/0xff.
    return np.copysign(magnitude, np.where(codes >= 128, -1.0, 1.0)).astype(np.float32)


@pytest.mark.parametrize("dtype", ["float8e4b15", "float8e4nv", "float8e5"])
@pytest.mark.parametrize("block_m,block_n,block_k", [(128, 256, 128), (64, 64, 64)], ids=["large-tile", "small-tile"])
@pytest.mark.parametrize("low_precision_acc", [0, 32, 64, 128])
def test_fp8_indirect_matmul(dtype, block_m, block_n, block_k, low_precision_acc):
    """A5 regression for bounded indirect loads feeding the FP8 matmul path."""
    if not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("Bounded indirect-load lowering is specific to A5")
    m, n, k = 128, 256, 256
    # Match the community input generator, which resets seed 17 for each call.
    a_bits = np.random.RandomState(17).randint(20, 40, (m, k), dtype=np.int8)
    b_bits = np.random.RandomState(17).randint(20, 40, (k, n), dtype=np.int8)
    a_storage = torch.from_numpy(a_bits).npu()
    b_storage = torch.from_numpy(b_bits).npu()
    a = triton.reinterpret(a_storage, getattr(tl, dtype))
    b = triton.reinterpret(b_storage, getattr(tl, dtype))
    output = torch.full((m, n), float("nan"), dtype=torch.float32, device="npu")
    max_num_imprecise_acc = low_precision_acc if low_precision_acc <= block_k else None
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), 1)
    _fp8_indirect_matmul_kernel[grid](a, b, output, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                      output.stride(0), output.stride(1), block_m, block_n, block_k,
                                      max_num_imprecise_acc, num_warps=8, num_stages=3)
    torch.npu.synchronize()

    # Use an independent CPU oracle rather than a second Triton conversion
    # kernel, so an identical address/FP8 bug cannot affect both operands.
    a_reference = torch.from_numpy(_decode_fp8_reference(a_bits, dtype))
    b_reference = torch.from_numpy(_decode_fp8_reference(b_bits, dtype))
    reference = a_reference @ b_reference
    rtol = 0.01 if dtype == "float8e4nv" else 0.001
    # E4M3B15 outputs are smaller than the community test's 1e-3 absolute
    # tolerance. Tighten this regression so zeros or a permuted K dimension
    # cannot pass. These FP8 inputs and their FP32 sums are exactly representable.
    atol = 1e-10 if dtype == "float8e4b15" else rtol
    torch.testing.assert_close(reference, output.cpu(), rtol=rtol, atol=atol)
