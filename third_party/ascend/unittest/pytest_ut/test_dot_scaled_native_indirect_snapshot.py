# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from dot_scaled_test_utils import require_dot_scaled_formats


@triton.jit
def _native_indirect_snapshot(A, B, SA, SB, Indices, Snapshot, Output, stride_bk, stride_bn, MODE: tl.constexpr,
                              EXPLICIT_ACC: tl.constexpr):
    m = tl.arange(0, 32)
    n = tl.arange(0, 64)
    k = tl.arange(0, 64)
    columns = tl.load(Indices + n)
    a = tl.load(A + m[:, None] * 64 + k[None, :])
    bp = B + k[:, None] * stride_bk + columns[None, :] * stride_bn
    if MODE == "masked":
        # Every address is allocated. A nonzero other distinguishes masking
        # from both an unconditional read and zero filling.
        mask = (k[:, None] * 3 + columns[None, :] * 5) % 7 != 0
        b = tl.load(bp, mask=mask, other=1.5)
    else:
        b = tl.load(bp, volatile=MODE == "volatile")
    if MODE == "alias_store":
        tl.store(bp, tl.full((64, 64), 0, tl.float32))
    if MODE == "shared":
        tl.store(Snapshot + k[:, None] * 64 + n[None, :], b)
    if SA is not None:
        sa = tl.load(SA + m[:, None] * 2 + tl.arange(0, 2)[None, :])
    else:
        sa = None
    if SB is not None:
        sb = tl.load(SB + columns[:, None] * 2 + tl.arange(0, 2)[None, :])
    else:
        sb = None
    initial = .125 + m[:, None].to(tl.float32) / 128 + n[None, :].to(tl.float32) / 64
    if EXPLICIT_ACC:
        result = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", initial)
    else:
        result = initial + tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3")
    tl.store(Output + m[:, None] * 64 + n[None, :], result, cache_modifier=".wt")


def _snapshot_inputs(mode, with_lhs, with_rhs):
    m = torch.arange(32)[:, None]
    n = torch.arange(64)[None, :]
    k = torch.arange(64)
    a = (((m * 7 + k[None, :] * 11 + m * k[None, :] * 3) % 17 - 8).float() / 8).to(torch.float8_e4m3fn)
    b = (((k[:, None] * 5 + n * 13 + (k[:, None] * n) % 19) % 23 - 11).float() / 8).to(torch.float8_e4m3fn)
    columns = (torch.arange(64) * 5 + 7) % 64
    assert torch.unique(columns).numel() == 64
    sa = (126 + (m + torch.arange(2)[None, :]) % 3).to(torch.uint8)
    sb = (126 + (torch.arange(64)[:, None] * 2 + torch.arange(2)[None, :]) % 3).to(torch.uint8)
    selected = b.float()[:, columns]
    if mode == "masked":
        mask = (k[:, None] * 3 + columns[None, :] * 5) % 7 != 0
        assert mask.any() and (~mask).any()
        selected = torch.where(mask, selected, 1.5)
    ar = a.float()
    br = selected
    if with_lhs:
        ar = ar * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
    if with_rhs:
        br = br * torch.exp2(sb[columns].float() - 127).repeat_interleave(32, 1).T
    # Every scaled input is an integer multiple of 1/16. Compute the dot in
    # int64, then divide once: this reference has no floating reduction error.
    ai, bi = (ar * 16).to(torch.int64), (br * 16).to(torch.int64)
    assert torch.equal(ai.float(), ar * 16) and torch.equal(bi.float(), br * 16)
    exact_product = ai @ bi
    # C is also on the 1/256 grid. The bound makes every reference and every
    # partial FP32 sum exactly representable, including the masked sentinel.
    initial = .125 + m.float() / 128 + n.float() / 64
    exact = exact_product + (initial * 256).to(torch.int64)
    assert (ai.abs() @ bi.abs() + (initial * 256).to(torch.int64)).max() < 2**24
    expected = exact.float() / 256
    product = exact_product.float() / 256
    assert torch.equal(ar @ br + initial, expected)
    # The observed regressions exchanged 16-wide row/column subblocks.
    assert not torch.equal(product[:16], product[16:])
    assert not torch.equal(product[:, 16:32], product[:, 32:48])
    assert not torch.equal(expected, initial)
    return a, b, columns, sa, sb, selected, expected


@pytest.mark.parametrize("mode", ["plain", "masked", "alias_store", "shared", "volatile"])
@pytest.mark.parametrize("with_lhs,with_rhs,explicit_acc", [(False, False, False), (True, False, True),
                                                            (False, True, False), (True, True, True)])
def test_dot_scaled_native_indirect_snapshot(mode, with_lhs, with_rhs, explicit_acc):
    require_dot_scaled_formats("e4m3", "e4m3")
    a, b, columns, sa, sb, selected, expected = _snapshot_inputs(mode, with_lhs, with_rhs)
    # Transfer exact FP8 storage bytes. Column-major B plus runtime indices
    # exercises the indirect-load path rather than a contiguous copy.
    an = a.view(torch.uint8).to("npu")
    bn = torch.empty_strided((64, 64), (1, 64), dtype=torch.uint8, device="npu")
    bn.copy_(b.view(torch.uint8))
    snapshot = torch.empty((64, 64), dtype=torch.uint8, device="npu")
    actual = torch.empty((32, 64), dtype=torch.float32, device="npu")
    _native_indirect_snapshot[(1, )](triton.reinterpret(an, tl.float8e4nv), triton.reinterpret(bn, tl.float8e4nv),
                                     sa.to("npu") if with_lhs else None, sb.to("npu") if with_rhs else None,
                                     columns.to(torch.int32).to("npu"), triton.reinterpret(snapshot, tl.float8e4nv),
                                     actual, *bn.stride(), mode, explicit_acc)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    if mode == "alias_store":
        torch.testing.assert_close(bn.cpu(), torch.zeros_like(b.view(torch.uint8)), atol=0, rtol=0)
    if mode == "shared":
        torch.testing.assert_close(snapshot.cpu(), selected.to(torch.float8_e4m3fn).view(torch.uint8), atol=0, rtol=0)
