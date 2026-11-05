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
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu

params = [
    (16, 16),
    (16, 32),
    (16, 64),
    (16, 128),
    (16, 256),
    (16, 512),
    (16, 1024),
    (32, 16),
    (64, 16),
    (128, 16),
    (256, 16),
    (512, 16),
    (1024, 16),
]


@triton.jit
def _axis_discrete_mask_load_kernel(
    kv_ptr,
    idx_ptr,
    out_ptr,
    topk,  # runtime scalar row bound
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    g = tl.arange(0, G)
    k_offs = tl.arange(0, BLOCK_K)

    valid = k_offs < topk
    slot = tl.load(idx_ptr + k_offs, mask=valid, other=0)
    valid = valid & (slot >= 0)

    # Axis-discrete 2D load: ptr structured, mask constant along g.
    kv0 = tl.load(
        kv_ptr + k_offs[:, None] * G + g[None, :],
        mask=valid[:, None],
        other=0.0,
    )

    tl.store(out_ptr + k_offs[:, None] * G + g[None, :], kv0)


@triton.jit
def _axis_discrete_mask_load_kernel_ptr_broadcast(
    kv_ptr,
    idx_ptr,
    out_ptr,
    topk,  # runtime scalar row bound
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    g = tl.arange(0, G)
    k_offs = tl.arange(0, BLOCK_K)

    valid = k_offs < topk
    slot = tl.load(idx_ptr + k_offs, mask=valid, other=0)
    valid = valid & (slot >= 0)

    # Axis-discrete 2D load: ptr structured along g, mask constant along g.
    kv0 = tl.load(
        kv_ptr + g[None, :],
        mask=valid[:, None],
        other=0.0,
    )

    tl.store(out_ptr + k_offs[:, None] * G + g[None, :], kv0)


@pytest.mark.parametrize("G,BLOCK_K", [
    (128, 64),
    (64, 128),
    (256, 64),
    (32, 128),
])
def test_axis_discrete_mask_load(G, BLOCK_K):
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float16, device="npu")
    idx = torch.randint(0, BLOCK_K, (BLOCK_K, ), dtype=torch.int32, device="npu")
    idx[idx % 2 == 1] = -1  # force rows with odd idx values to be masked out
    out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")

    def _reference():
        # Rows with idx >= 0 are kept as-is; other rows are zero-filled.
        mask = (idx >= 0).to(kv.dtype).unsqueeze(1)  # (BLOCK_K, 1)
        return kv * mask

    _axis_discrete_mask_load_kernel[(1, )](
        kv,
        idx,
        out,
        BLOCK_K,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out, _reference(), rtol=0, atol=0)


@triton.jit
def _axis_continuous_mask_load_kernel(
    kv_ptr,
    idx_ptr,
    out_ptr,
    topk,  # runtime scalar row bound
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    g = tl.arange(0, G)
    k_offs = tl.arange(0, BLOCK_K)

    valid = g < topk
    slot = tl.load(idx_ptr + g, mask=valid, other=0)
    valid = valid & (slot >= 0)

    # Axis-continuous 2D load: ptr structured along g, mask constant along g.
    kv0 = tl.load(
        kv_ptr + k_offs[:, None] * G + g[None, :],
        mask=valid[None, :],
        other=0.0,
    )
    tl.store(out_ptr + k_offs[:, None] * G + g[None, :], kv0)


@pytest.mark.parametrize("G,BLOCK_K", [
    (128, 64),
    (64, 128),
    (256, 64),
    (32, 128),
])
def test_axis_continuous_mask_load(G, BLOCK_K):
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float16, device="npu")
    idx = torch.randint(0, G, (G, ), dtype=torch.int32, device="npu")
    idx[idx % 2 == 1] = -1  # force rows with odd idx values to be masked out
    out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")

    def _reference():
        # Columns with idx >= 0 are kept as-is; other columns are zero-filled.
        mask = (idx >= 0).to(kv.dtype).unsqueeze(0)  # (1, G)
        return kv * mask

    _axis_continuous_mask_load_kernel[(1, )](
        kv,
        idx,
        out,
        G,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out, _reference(), rtol=0, atol=0)
