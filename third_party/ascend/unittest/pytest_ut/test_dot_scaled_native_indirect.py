# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from test_dot_scaled_native_tiles_layout import _assert_layout_sensitive, _operand

from dot_scaled_test_utils import require_dot_scaled_formats


@triton.jit
def _native_indirect_tile_dot(a_ptr, b_ptr, sa_ptr, sb_ptr, row_indices, col_indices, out_ptr, stride_bk, stride_bn,
                              M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, A_FORMAT: tl.constexpr,
                              B_FORMAT: tl.constexpr, EXPLICIT_ACC: tl.constexpr, INDIRECT_A: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tl.static_assert(M % BLOCK_M == 0 and N % BLOCK_N == 0)
    AK: tl.constexpr = K // 2 if A_FORMAT == "e2m1" else K
    BK: tl.constexpr = K // 2 if B_FORMAT == "e2m1" else K
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    a_rows = tl.load(row_indices + rows) if INDIRECT_A else rows
    b_cols = cols if INDIRECT_A else tl.load(col_indices + cols)
    acc = .125 + rows[:, None].to(tl.float32) / 128 + cols[None, :].to(tl.float32) / 64
    for block in range(2):
        a = tl.load(a_ptr + a_rows[:, None] * (2 * AK) + block * AK + tl.arange(0, AK)[None, :])
        b = tl.load(b_ptr + (block * BK + tl.arange(0, BK)[:, None]) * stride_bk + b_cols[None, :] * stride_bn)
        if sa_ptr is not None:
            sa = tl.load(sa_ptr + a_rows[:, None] * (2 * K // 32) + block * (K // 32) + tl.arange(0, K // 32)[None, :])
        else:
            sa = None
        if sb_ptr is not None:
            sb = tl.load(sb_ptr + b_cols[:, None] * (2 * K // 32) + block * (K // 32) + tl.arange(0, K // 32)[None, :])
        else:
            sb = None
        if EXPLICIT_ACC:
            acc = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc)
        else:
            acc += tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc, cache_modifier=".wt")


@pytest.mark.parametrize("shape", [(128, 32, 128), (32, 128, 128), (128, 128, 128), (128, 128, 256)])
@pytest.mark.parametrize("a_format,b_format", [("e2m1", "e2m1"), ("e4m3", "e4m3"), ("e5m2", "e4m3")])
@pytest.mark.parametrize("with_lhs,with_rhs", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("explicit_acc", [False, True])
@pytest.mark.parametrize("indirect_a", [False, True])
def test_dot_scaled_native_indirect_nonperiodic(shape, a_format, b_format, with_lhs, with_rhs, explicit_acc,
                                                indirect_a):
    require_dot_scaled_formats(a_format, b_format)
    m, n, k = shape
    a, ar = _operand(m, 2 * k, a_format, 1, 0)
    b, br = _operand(2 * k, n, b_format, 0, 3)
    b = b.T.contiguous().T
    rows = (torch.arange(m) * 5 + 3) % m
    cols = (torch.arange(n) * 5 + 7) % n
    assert torch.unique(rows).numel() == m and torch.unique(cols).numel() == n
    ai = (torch.arange(m)[:, None] + torch.arange(2 * k // 32)[None, :]) % 4
    bi = (torch.arange(n)[:, None] * 3 + torch.arange(2 * k // 32)[None, :]) % 4
    sa = torch.tensor([125, 127, 128, 129], dtype=torch.uint8)[ai]
    sb = torch.tensor([126, 125, 128, 127], dtype=torch.uint8)[bi]
    if with_lhs:
        ar = ar * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
    if with_rhs:
        br = br * torch.exp2(sb.float() - 127).repeat_interleave(32, 1).T
    if indirect_a:
        ar = ar[rows, :]
    else:
        br = br[:, cols]
    expected = .125 + torch.arange(m)[:, None].float() / 128 + torch.arange(n)[None, :].float() / 64
    product = torch.zeros_like(expected)
    for block in range(2):
        partial = ar[:, block * k:(block + 1) * k] @ br[block * k:(block + 1) * k]
        expected += partial
        product += partial
    _assert_layout_sensitive(product)
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    # Bound per-program storage without reducing the matrix or splitting either K block.
    block_m, block_n = 32, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _native_indirect_tile_dot[grid](a.to("npu"), b.to("npu"), sa.to("npu") if with_lhs else None,
                                    sb.to("npu") if with_rhs else None, rows.to(torch.int32).to("npu"),
                                    cols.to(torch.int32).to("npu"), actual, *b.stride(), m, n, k, a_format, b_format,
                                    explicit_acc, indirect_a, block_m, block_n)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
