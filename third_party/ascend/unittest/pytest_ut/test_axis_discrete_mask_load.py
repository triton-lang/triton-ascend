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
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu


@triton.jit
def _axis_discrete_mask_load_kernel(
    kv_ptr, idx_ptr, out_ptr,
    topk,                        # runtime scalar row bound
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    g = tl.arange(0, G)               # [G]
    k_offs = tl.arange(0, BLOCK_K)    # [K]

    valid = k_offs < topk
    slot = tl.load(idx_ptr + k_offs, mask=valid, other=0)
    valid = valid & (slot >= 0)

    # Axis-discrete 2D load: ptr structured along g, mask constant along g.
    kv0 = tl.load(
        kv_ptr + g[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    tl.store(out_ptr + k_offs[:, None] * G + g[None, :], kv0)


def _reference(kv, idx, topk, block_k):
    # Reference (on NPU): load row 0 of kv, masked by (k < topk) & (idx[k] >= 0).
    valid_ref = (torch.arange(block_k, device="npu") < topk) & (idx >= 0)
    out_ref = torch.zeros((block_k, kv.shape[1]), dtype=torch.float32, device="npu")
    out_ref[valid_ref, :] = kv[0]
    return out_ref


@pytest.mark.parametrize("G,BLOCK_K,topk", [
    (128, 64, 32),
    (64, 128, 64),
    (256, 64, 16),
    (32, 128, 128),
])
def test_axis_discrete_mask_load(G, BLOCK_K, topk):
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float32, device="npu")
    idx = torch.randint(0, topk, (BLOCK_K,), dtype=torch.int32, device="npu")
    idx[topk // 2] = -1  # force some rows to be masked out
    out = torch.zeros((BLOCK_K, G), dtype=torch.float32, device="npu")

    _axis_discrete_mask_load_kernel[(1, )](
        kv, idx, out,
        topk,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out, _reference(kv, idx, topk, BLOCK_K), rtol=0, atol=0)


@pytest.mark.parametrize("G,BLOCK_K,topk", [
    (128, 64, 32),
    (256, 64, 32),
    (512, 128, 64),
    (1024, 128, 64),
])
def test_axis_discrete_mask_load_perf(G, BLOCK_K, topk):
    """Benchmark the pure device-side latency of the axis-discrete mask load kernel."""
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float32, device="npu")
    idx = torch.randint(0, topk, (BLOCK_K,), dtype=torch.int32, device="npu")
    idx[topk // 2] = -1  # force some rows to be masked out
    out = torch.zeros((BLOCK_K, G), dtype=torch.float32, device="npu")

    def fn():
        _axis_discrete_mask_load_kernel[(1, )](
            kv, idx, out,
            topk,
            BLOCK_K=BLOCK_K,
            G=G,
        )

    ms = do_bench_npu(fn)
    print(f"axis-discrete-mask-load G={G} BLOCK_K={BLOCK_K} topk={topk}: {ms:.3f} ms")
