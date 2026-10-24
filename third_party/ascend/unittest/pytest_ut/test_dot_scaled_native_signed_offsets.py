# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from dot_scaled_test_utils import require_dot_scaled_formats


@triton.jit
def _native_signed_offsets(A, CenterB, SA, SB, Indices, Snapshot, Output, Bias, Correction, FORMAT: tl.constexpr,
                           MODE: tl.constexpr, EXPLICIT_ACC: tl.constexpr):
    K: tl.constexpr = 128
    PK: tl.constexpr = K // 2 if FORMAT == "e2m1" else K
    CENTER: tl.constexpr = PK * 64 // 2
    m = tl.arange(0, 32)
    n = tl.arange(0, 64)
    p = tl.arange(0, PK)
    columns = tl.load(Indices + n)
    a = tl.load(A + m[:, None] * PK + p[None, :])
    offsets = p[:, None] + columns[None, :] * PK
    if MODE == "wrapped":
        # These are separate runtime arguments. The first addition overflows
        # signed i32; the subtraction returns to the original legal index.
        offsets = (offsets + Bias.to(tl.int32)) - Correction.to(tl.int32)
    offsets = offsets - CENTER
    # Preserve a direct signed extension at the helper boundary. CenterB is
    # already an interior pointer into the same allocated backing storage.
    bp = CenterB + offsets.to(tl.int64)
    if MODE == "masked":
        mask = (p[:, None] * 3 + columns[None, :] * 5) % 7 != 0
        OTHER: tl.constexpr = 0x67 if FORMAT == "e2m1" else 1.5
        b = tl.load(bp, mask=mask, other=OTHER)
    else:
        b = tl.load(bp, volatile=MODE == "volatile")
    if MODE == "shared":
        tl.store(Snapshot + p[:, None] * 64 + n[None, :], b)
    if SA is not None:
        sa = tl.load(SA + m[:, None] * 4 + tl.arange(0, 4)[None, :])
        sb = tl.load(SB + columns[:, None] * 4 + tl.arange(0, 4)[None, :])
    else:
        sa = None
        sb = None
    initial = .125 + m[:, None].to(tl.float32) / 128 + n[None, :].to(tl.float32) / 64
    if EXPLICIT_ACC:
        result = tl.dot_scaled(a, sa, FORMAT, b, sb, FORMAT, initial)
    else:
        result = initial + tl.dot_scaled(a, sa, FORMAT, b, sb, FORMAT)
    tl.store(Output + m[:, None] * 64 + n[None, :], result, cache_modifier=".wt")


def _decode_fp4_bytes(storage, axis):
    # Independent E2M1 decoding of the actual packed bytes: low nibble first.
    twice = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.int64)
    raw = storage.to(torch.int64)
    low, high = twice[raw & 15], twice[raw >> 4]
    if axis == 1:
        return torch.stack((low, high), dim=2).reshape(storage.shape[0], storage.shape[1] * 2).float() / 2
    return torch.stack((low, high), dim=1).reshape(storage.shape[0] * 2, storage.shape[1]).float() / 2


def _signed_offset_inputs(format, mode, with_scales):
    m = torch.arange(32)[:, None]
    n = torch.arange(64)[None, :]
    k = torch.arange(128)
    if format == "e2m1":
        palette = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11], dtype=torch.uint8)
        ac = palette[(m * 7 + k[None, :] * 5 + (m * k[None, :]) % 13) % 8]
        bc = palette[(k[:, None] * 3 + n * 7 + (k[:, None] * n) % 11) % 8]
        a = ac[:, 0::2] | (ac[:, 1::2] << 4)
        b = bc[0::2] | (bc[1::2] << 4)
        ar = _decode_fp4_bytes(a, 1)
    else:
        af = (((m * 7 + k[None, :] * 11 + m * k[None, :] * 3) % 17 - 8).float() / 8).to(torch.float8_e4m3fn)
        bf = (((k[:, None] * 5 + n * 13 + (k[:, None] * n) % 19) % 23 - 11).float() / 8).to(torch.float8_e4m3fn)
        a, b = af.view(torch.uint8), bf.view(torch.uint8)
        ar = a.view(torch.float8_e4m3fn).float()
    pk = b.shape[0]
    columns = (torch.arange(64) * 5 + 7) % 64
    assert torch.unique(columns).numel() == 64
    selected = b[:, columns].clone()
    physical_rows = torch.arange(pk)[:, None]
    flat_indices = physical_rows + columns[None, :] * pk
    center = pk * 64 // 2
    offsets = flat_indices - center
    assert offsets.min() < 0 < offsets.max()
    assert (offsets + center).min() == 0 and (offsets + center).max() == b.numel() - 1
    if mode == "wrapped":
        bias = 2**31 - 32
        mathematical_sum = flat_indices + bias
        assert (mathematical_sum > 2**31 - 1).any()
        # Model both i32 operations modulo 2^32, then interpret as signed.
        wrapped = ((mathematical_sum + 2**31) % 2**32) - 2**31
        restored = ((wrapped - bias + 2**31) % 2**32) - 2**31
        assert torch.equal(restored, flat_indices)
    if mode == "masked":
        mask = (physical_rows * 3 + columns[None, :] * 5) % 7 != 0
        sentinel = 0x67 if format == "e2m1" else torch.tensor(1.5).to(torch.float8_e4m3fn).view(torch.uint8).item()
        assert mask.any() and (~mask).any() and not (selected == sentinel).any()
        selected = torch.where(mask, selected, sentinel).to(torch.uint8)
    br = _decode_fp4_bytes(selected, 0) if format == "e2m1" else selected.contiguous().view(torch.float8_e4m3fn).float()
    sa = (126 + (m + torch.arange(4)[None, :]) % 3).to(torch.uint8)
    sb = (126 + (torch.arange(64)[:, None] * 2 + torch.arange(4)[None, :]) % 3).to(torch.uint8)
    if with_scales:
        ar = ar * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
        br = br * torch.exp2(sb[columns].float() - 127).repeat_interleave(32, 1).T
    # The scaled, quantized inputs lie on a 1/16 grid. The int64 dot is an
    # independent exact reference; no reduction tolerance hides address errors.
    ai, bi = (ar * 16).to(torch.int64), (br * 16).to(torch.int64)
    assert torch.equal(ai.float(), ar * 16) and torch.equal(bi.float(), br * 16)
    initial = .125 + m.float() / 128 + n.float() / 64
    ci = (initial * 256).to(torch.int64)
    exact_product = ai @ bi
    assert (ai.abs() @ bi.abs() + ci).max() < 2**24
    expected = (exact_product + ci).float() / 256
    assert torch.equal(ar @ br + initial, expected)
    product = exact_product.float() / 256
    assert not torch.equal(product[:16], product[16:])
    assert not torch.equal(product[:, 16:32], product[:, 32:48])
    assert not torch.equal(expected, initial)
    return a, b, columns, sa, sb, selected, expected


@pytest.mark.parametrize("format", ["e2m1", "e4m3"])
@pytest.mark.parametrize("mode", ["negative", "wrapped", "masked", "shared", "volatile"])
@pytest.mark.parametrize("with_scales", [False, True])
def test_dot_scaled_native_signed_offsets(format, mode, with_scales):
    require_dot_scaled_formats(format, format)
    a, b, columns, sa, sb, selected, expected = _signed_offset_inputs(format, mode, with_scales)
    # Keep the entire owner alive. The view's data pointer is in the middle of
    # this allocation, so every negative offset still addresses valid storage.
    an = a.to("npu")
    owner = b.T.contiguous().reshape(-1).to("npu")
    centered = owner[owner.numel() // 2:]
    snapshot = torch.empty_like(b, device="npu")
    actual = torch.empty((32, 64), dtype=torch.float32, device="npu")
    dtype = tl.uint8 if format == "e2m1" else tl.float8e4nv
    _native_signed_offsets[(1, )](triton.reinterpret(an, dtype),
                                  triton.reinterpret(centered, dtype), sa.to("npu") if with_scales else None,
                                  sb.to("npu") if with_scales else None, columns.to(torch.int32).to("npu"),
                                  triton.reinterpret(snapshot,
                                                     dtype), actual, 2**31 - 32, 2**31 - 32, format, mode, with_scales)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    if mode == "shared":
        torch.testing.assert_close(snapshot.cpu(), selected, atol=0, rtol=0)
