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

import torch
import torch_npu  # noqa: F401  # Registers the "npu" device with PyTorch.

import triton
import triton.language as tl


@triton.jit
def zero_stride_store_broadcast_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)
    offsets = offs_m * STRIDE_M + offs_n * STRIDE_N

    # Every logical row aliases the same backing row, so all writers store the
    # same value. This makes the end-to-end result deterministic while keeping
    # the full-rank zero-stride pointer broadcast that StoreConverter must defer.
    # Match the triggering FlagGems path: 0.0 is a DSL scalar, represented as
    # a uniform full-rank TTIR value, so every aliased row writes the same value.
    tl.store(ptr + offsets, 0.0, mask=mask)


def test_zero_stride_pointer_broadcast_store():
    base = torch.full((1, 4), 7.0, dtype=torch.float32, device="npu")
    expanded = base.expand(3, 4)
    original_stride = expanded.stride()
    original_data_ptr = expanded.data_ptr()

    zero_stride_store_broadcast_kernel[(1, 1)](
        expanded,
        M=expanded.size(0),
        N=expanded.size(1),
        STRIDE_M=expanded.stride(0),
        STRIDE_N=expanded.stride(1),
        BLOCK_M=16,
        BLOCK_N=64,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(base, torch.zeros_like(base))
    torch.testing.assert_close(expanded, torch.zeros_like(expanded))
    assert expanded.stride() == original_stride == (0, 1)
    assert expanded.data_ptr() == original_data_ptr
