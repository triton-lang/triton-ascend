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
"""Axis-discrete mask load on the A5/A6 register pipeline.

Each test checks the numeric result and, on A5/A6, the routing of the same
compiled kernel (SIMD vs the SIMT indirect fallback).
"""

import re

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_register_based_arch


def _register_based():
    arch = triton.runtime.driver.active.get_current_target().arch
    return is_register_based_arch(arch)


def _skip_unless_register_based():
    if not _register_based():
        pytest.skip("axis-discrete SIMD routing requires A5/A6")


def _adapter(compiled, kernel_name):
    assert "ttadapter" in compiled.asm, f"{kernel_name}: no ttadapter stage was produced"
    return compiled.asm["ttadapter"]


def _assert_simd(compiled, kernel_name):
    """No SIMT fallback; the discrete part is folded into arith.select."""
    adapter = _adapter(compiled, kernel_name)
    assert "indirect_load" not in adapter, f"{kernel_name}: fell back to indirect_load"
    assert "indirect_store" not in adapter, f"{kernel_name}: fell back to indirect_store"
    assert "arith.select" in adapter, f"{kernel_name}: discrete part was not folded into select"
    return adapter


def _assert_simt(compiled, kernel_name):
    adapter = _adapter(compiled, kernel_name)
    assert "indirect_load" in adapter, f"{kernel_name}: expected the SIMT indirect fallback"
    return adapter


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


@pytest.mark.parametrize("G,BLOCK_K", [
    (128, 64),
    (64, 128),
])
def test_axis_discrete_mask_load(G, BLOCK_K):
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float16, device="npu")
    idx = torch.randint(0, BLOCK_K, (BLOCK_K, ), dtype=torch.int32, device="npu")
    idx[idx % 2 == 1] = -1  # force rows with odd idx values to be masked out
    out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")

    compiled = _axis_discrete_mask_load_kernel[(1, )](
        kv,
        idx,
        out,
        BLOCK_K,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    keep = (idx >= 0).to(kv.dtype).unsqueeze(1)
    torch.testing.assert_close(out, kv * keep, rtol=0, atol=0)
    if _register_based():
        _assert_simd(compiled, "axis_discrete_mask_load")


@triton.jit
def _axis_discrete_mask_load_kernel_2d_and(
    kv_ptr,
    out_ptr,
    limit,  # runtime scalar column bound
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    k_offs = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)

    # Top-level 2D AND: discrete rows & continuous columns.
    row_keep = (k_offs % 2) == 0
    col_keep = g < limit
    mask = row_keep[:, None] & col_keep[None, :]

    kv0 = tl.load(
        kv_ptr + k_offs[:, None] * G + g[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + k_offs[:, None] * G + g[None, :], kv0)


@pytest.mark.parametrize("G,BLOCK_K,limit", [
    (128, 64, 96),
    (64, 128, 32),
])
def test_axis_discrete_mask_load_2d_and(G, BLOCK_K, limit):
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float16, device="npu")
    out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")

    compiled = _axis_discrete_mask_load_kernel_2d_and[(1, )](
        kv,
        out,
        limit,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    ref = kv.clone()
    ref[1::2, :] = 0  # discrete rows
    ref[:, limit:] = 0  # continuous columns
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    if _register_based():
        adapter = _assert_simd(compiled, "axis_discrete_mask_load_2d_and")
        # contMask must narrow the DMA range at runtime.
        assert re.search(r"memref\.subview[^\n]*\[\d+, %\w+\]", adapter), \
            "contMask did not narrow the DMA range"


@triton.jit
def _axis_discrete_mask_load_kernel_rank3(
    kv_ptr,
    idx_ptr,
    out_ptr,
    topk,  # runtime scalar bound along the discrete axis
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    r = tl.arange(0, BLOCK_R)
    k_offs = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)

    # Axis-discrete rank-3 load: ptr structured, mask constant along both
    # the leading dim (BLOCK_R) and the contiguous dim (G).
    in_range = k_offs < topk
    slot = tl.load(idx_ptr + k_offs, mask=in_range, other=0)
    valid = in_range & (slot >= 0)

    offs = (r[:, None, None] * (BLOCK_K * G) + k_offs[None, :, None] * G + g[None, None, :])
    kv0 = tl.load(
        kv_ptr + offs,
        mask=valid[None, :, None],
        other=0.0,
    )
    tl.store(out_ptr + offs, kv0)


@pytest.mark.parametrize("BLOCK_R,BLOCK_K,G,topk", [
    (8, 64, 128, 48),
    (4, 128, 64, 96),
])
def test_axis_discrete_mask_load_rank3(BLOCK_R, BLOCK_K, G, topk):
    # The rank-3 tile does not fit A3 UB; the carve-out is A5/A6-only.
    _skip_unless_register_based()
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_R, BLOCK_K, G, dtype=torch.float16, device="npu")
    idx = torch.randint(0, BLOCK_K, (BLOCK_K, ), dtype=torch.int32, device="npu")
    idx[idx % 2 == 1] = -1  # force odd rows to be masked out
    out = torch.zeros((BLOCK_R, BLOCK_K, G), dtype=torch.float16, device="npu")

    compiled = _axis_discrete_mask_load_kernel_rank3[(1, )](
        kv,
        idx,
        out,
        topk,
        BLOCK_R=BLOCK_R,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    rows = (torch.arange(BLOCK_K) < topk) & (idx.cpu() >= 0)
    keep = rows.to(kv.dtype).view(1, BLOCK_K, 1).to(kv.device)
    torch.testing.assert_close(out, kv * keep, rtol=0, atol=0)
    if _register_based():
        _assert_simd(compiled, "axis_discrete_mask_load_rank3")


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
])
def test_axis_continuous_mask_load(G, BLOCK_K):
    """Control: a continuous mask must keep using the structured path."""
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float16, device="npu")
    idx = torch.randint(0, G, (G, ), dtype=torch.int32, device="npu")
    idx[idx % 2 == 1] = -1
    out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")

    compiled = _axis_continuous_mask_load_kernel[(1, )](
        kv,
        idx,
        out,
        G,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    keep = (idx >= 0).to(kv.dtype).unsqueeze(0)
    torch.testing.assert_close(out, kv * keep, rtol=0, atol=0)
    if _register_based():
        adapter = _adapter(compiled, "axis_continuous_mask_load")
        assert "indirect_load" not in adapter


@triton.jit
def _axis_ambiguous_mask_load_kernel(
    kv_ptr,
    out_ptr,
    limit,  # runtime bound
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    k_offs = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)

    # Negative control: the predicate varies along *both* axes.
    mask = (k_offs[:, None] + g[None, :]) < limit

    kv0 = tl.load(
        kv_ptr + k_offs[:, None] * G + g[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + k_offs[:, None] * G + g[None, :], kv0)


@pytest.mark.parametrize("G,BLOCK_K,limit", [
    (128, 64, 96),
])
def test_axis_ambiguous_mask_load_falls_back_to_simt(G, BLOCK_K, limit):
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float16, device="npu")
    out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")

    compiled = _axis_ambiguous_mask_load_kernel[(1, )](
        kv,
        out,
        limit,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    k = torch.arange(BLOCK_K, device="npu")[:, None]
    g = torch.arange(G, device="npu")[None, :]
    expected = torch.where((k + g) < limit, kv, torch.zeros_like(kv))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if _register_based():
        _assert_simt(compiled, "axis_ambiguous_mask_load")


@triton.jit
def _axis_c1_mask_load_kernel(
    kv_ptr,
    bounds_ptr,
    out_ptr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    k_offs = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)

    # C1: every row has its own runtime length; no constant axis.
    bound = tl.load(bounds_ptr + k_offs)
    mask = g[None, :] < bound[:, None]

    kv0 = tl.load(
        kv_ptr + k_offs[:, None] * G + g[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + k_offs[:, None] * G + g[None, :], kv0)


@pytest.mark.parametrize("G,BLOCK_K", [
    (64, 32),
])
def test_axis_c1_mask_load_falls_back_to_simt(G, BLOCK_K):
    torch.manual_seed(0)
    kv = torch.randn(BLOCK_K, G, dtype=torch.float16, device="npu")
    bounds = torch.randint(0, G + 1, (BLOCK_K, ), dtype=torch.int32, device="npu")
    out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")

    compiled = _axis_c1_mask_load_kernel[(1, )](
        kv,
        bounds,
        out,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    g = torch.arange(G, device="npu")[None, :]
    expected = torch.where(g < bounds[:, None], kv, torch.zeros_like(kv))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if _register_based():
        _assert_simt(compiled, "axis_c1_mask_load")


@triton.jit
def _axis_select_mask_load_kernel(
    kv_ptr,
    idx_ptr,
    out_ptr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    k_offs = tl.arange(0, BLOCK_K)
    g = tl.arange(0, G)
    half = BLOCK_K // 2

    # select form: all three operands are row-wise, so the result is constant
    # along g.
    slot = tl.load(idx_ptr + k_offs)
    cond = slot >= 0
    first = k_offs < half
    second = slot < half
    mask = tl.where(cond[:, None], first[:, None], second[:, None])

    kv0 = tl.load(
        kv_ptr + k_offs[:, None] * G + g[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + k_offs[:, None] * G + g[None, :], kv0)


@pytest.mark.parametrize("G,BLOCK_K", [
    (128, 64),
    (64, 128),
])
def test_axis_select_mask_load(G, BLOCK_K):
    torch.manual_seed(0)
    half = BLOCK_K // 2
    kv = torch.randn(BLOCK_K, G, dtype=torch.float16, device="npu")
    idx = torch.randint(0, BLOCK_K, (BLOCK_K, ), dtype=torch.int32, device="npu")
    idx[idx % 2 == 1] = -1
    out = torch.zeros((BLOCK_K, G), dtype=torch.float16, device="npu")

    compiled = _axis_select_mask_load_kernel[(1, )](
        kv,
        idx,
        out,
        BLOCK_K=BLOCK_K,
        G=G,
    )
    torch.npu.synchronize()

    k = torch.arange(BLOCK_K, device="npu")[:, None]
    keep = torch.where((idx >= 0)[:, None], k < half, (idx < half)[:, None])
    expected = torch.where(keep, kv, torch.zeros_like(kv))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if _register_based():
        _assert_simd(compiled, "axis_select_mask_load")
