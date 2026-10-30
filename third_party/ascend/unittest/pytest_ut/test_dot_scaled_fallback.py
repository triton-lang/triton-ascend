# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from dot_scaled_test_utils import require_dot_scaled_formats


@triton.jit
def _scaled_fallback(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr, A_K_PACK: tl.constexpr, B_K_PACK: tl.constexpr,
                     GROUP: tl.constexpr):
    AM: tl.constexpr = M // 2 if A_FORMAT == "e2m1" and not A_K_PACK else M
    AK: tl.constexpr = K // 2 if A_FORMAT == "e2m1" and A_K_PACK else K
    BK: tl.constexpr = K // 2 if B_FORMAT == "e2m1" and B_K_PACK else K
    BN: tl.constexpr = N // 2 if B_FORMAT == "e2m1" and not B_K_PACK else N
    a = tl.load(a_ptr + tl.arange(0, AM)[:, None] * AK + tl.arange(0, AK)[None, :])
    b = tl.load(b_ptr + tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :])
    if sa_ptr is not None:
        sa = tl.load(sa_ptr + tl.arange(0, M)[:, None] * (K // GROUP) + tl.arange(0, K // GROUP)[None, :])
    else:
        sa = None
    if sb_ptr is not None:
        sb = tl.load(sb_ptr + tl.arange(0, N)[:, None] * (K // GROUP) + tl.arange(0, K // GROUP)[None, :])
    else:
        sb = None
    acc = tl.full((M, N), 0.125, tl.float32)
    out = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc, lhs_k_pack=A_K_PACK, rhs_k_pack=B_K_PACK)
    tl.store(out_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], out)


def _make_operand(shape, format, packed_axis):
    # Cover all 16 E2M1 bit patterns, including both signed zeros, in each axis.
    indices = (torch.arange(shape[0])[:, None] * 7 + torch.arange(shape[1])[None, :] * 5) % 16
    if format == "e2m1":
        lookup = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        logical = lookup[indices]
        if packed_axis == 0:
            packed = indices[0::2, :] | (indices[1::2, :] << 4)
        else:
            packed = indices[:, 0::2] | (indices[:, 1::2] << 4)
        return packed.to(torch.uint8).contiguous().to("npu"), logical
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2":
             torch.float8_e5m2}[format]
    value = ((indices.float() - 7) / 8).to(dtype)
    return value.contiguous().to("npu"), value.float()


def _make_scales(rows, k, scale_format):
    group = 32 if scale_format == "e8m0" else 16
    indices = (torch.arange(rows)[:, None] + torch.arange(k // group)[None, :]) % 4
    if scale_format == "e8m0":
        encoded = torch.tensor([124, 126, 127, 129], dtype=torch.uint8)[indices]
        decoded = torch.exp2(encoded.float() - 127)
    else:
        encoded = torch.tensor([.125, .375, 1.5, 3.])[indices].to(torch.float8_e4m3fn)
        decoded = encoded.float()
    return encoded.contiguous().to("npu"), decoded.repeat_interleave(group, dim=1)


@pytest.mark.parametrize("a_format,b_format,a_k_pack,b_k_pack", [
    ("e2m1", "fp16", True, True),
    ("e2m1", "fp16", False, True),
    ("fp16", "e2m1", True, True),
    ("fp16", "e2m1", True, False),
    ("e2m1", "e5m2", True, True),
    ("e4m3", "e2m1", True, False),
    ("e2m1", "e2m1", False, False),
    ("e2m1", "e2m1", True, True),
    ("fp16", "bf16", True, True),
])
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
@pytest.mark.parametrize("with_a_scale,with_b_scale", [(False, False), (False, True), (True, False), (True, True)])
def test_dot_scaled_fallback_formats(a_format, b_format, a_k_pack, b_k_pack, scale_format, with_a_scale, with_b_scale):
    require_dot_scaled_formats(a_format, b_format, scale_format if with_a_scale or with_b_scale else None)
    m, n, k = 16, 32, 64
    a, a_ref = _make_operand((m, k), a_format, 1 if a_k_pack else 0)
    b, b_ref = _make_operand((k, n), b_format, 0 if b_k_pack else 1)
    sa, sa_ref = _make_scales(m, k, scale_format)
    sb, sb_ref = _make_scales(n, k, scale_format)
    if not with_a_scale:
        sa, sa_ref = None, 1.
    if not with_b_scale:
        sb, sb_ref = None, 1.
    else:
        sb_ref = sb_ref.T
    expected = (a_ref * sa_ref) @ (b_ref * sb_ref) + .125
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_fallback[(1, )](a, b, sa, sb, actual, m, n, k, a_format, b_format, a_k_pack, b_k_pack,
                            32 if scale_format == "e8m0" else 16)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("nan_side", ["lhs", "rhs"])
def test_dot_scaled_fallback_nan(nan_side):
    require_dot_scaled_formats("e2m1", "fp16", "e8m0")
    m, n, k = 16, 32, 64
    a, a_ref = _make_operand((m, k), "e2m1", 0)
    b, b_ref = _make_operand((k, n), "fp16", 0)
    sa = torch.full((m, k // 32), 127, dtype=torch.uint8)
    sb = torch.full((n, k // 32), 127, dtype=torch.uint8)
    (sa if nan_side == "lhs" else sb)[3, 1] = 255
    expected = a_ref @ b_ref + .125
    if nan_side == "lhs":
        expected[3, :] = float("nan")
    else:
        expected[:, 3] = float("nan")
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_fallback[(1, )](a, b, sa.to("npu"), sb.to("npu"), actual, m, n, k, "e2m1", "fp16", False, True, 32)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5, equal_nan=True)
