# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from dot_scaled_test_utils import require_dot_scaled_formats


@triton.jit
def _batch_scaled(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                  K: tl.constexpr, A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr, A_K_PACK: tl.constexpr,
                  B_K_PACK: tl.constexpr, GROUP: tl.constexpr):
    AM: tl.constexpr = M // 2 if A_FORMAT == "e2m1" and not A_K_PACK else M
    AK: tl.constexpr = K // 2 if A_FORMAT == "e2m1" and A_K_PACK else K
    BK: tl.constexpr = K // 2 if B_FORMAT == "e2m1" and B_K_PACK else K
    BN: tl.constexpr = N // 2 if B_FORMAT == "e2m1" and not B_K_PACK else N
    batch = tl.arange(0, B)[:, None, None]
    a = tl.load(a_ptr + batch * AM * AK + tl.arange(0, AM)[None, :, None] * AK + tl.arange(0, AK)[None, None, :])
    b = tl.load(b_ptr + batch * BK * BN + tl.arange(0, BK)[None, :, None] * BN + tl.arange(0, BN)[None, None, :])
    if sa_ptr is not None:
        sa = tl.load(sa_ptr + batch * M * (K // GROUP) + tl.arange(0, M)[None, :, None] * (K // GROUP) +
                     tl.arange(0, K // GROUP)[None, None, :])
    else:
        sa = None
    if sb_ptr is not None:
        sb = tl.load(sb_ptr + batch * N * (K // GROUP) + tl.arange(0, N)[None, :, None] * (K // GROUP) +
                     tl.arange(0, K // GROUP)[None, None, :])
    else:
        sb = None
    acc = tl.full((B, M, N), .125, tl.float32) + batch.to(tl.float32) * .125
    out = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc, lhs_k_pack=A_K_PACK, rhs_k_pack=B_K_PACK)
    tl.store(out_ptr + batch * M * N + tl.arange(0, M)[None, :, None] * N + tl.arange(0, N)[None, None, :], out)


def _operand(rows, cols, format, pack_axis):
    codes = (torch.arange(2)[:, None, None] * 3 + torch.arange(rows)[None, :, None] * 7 +
             torch.arange(cols)[None, None, :] * 5) % 16
    if format == "e2m1":
        lookup = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        logical = lookup[codes]
        if pack_axis == 0:
            packed = codes[:, 0::2, :] | (codes[:, 1::2, :] << 4)
        else:
            packed = codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)
        return packed.to(torch.uint8).contiguous(), logical
    value = ((codes.float() - 7) / 8).to(torch.float16 if format == "fp16" else torch.bfloat16)
    return value, value.float()


def _scales(rows, k, format):
    group = 32 if format == "e8m0" else 16
    indices = (torch.arange(2)[:, None, None] * 2 + torch.arange(rows)[None, :, None] +
               torch.arange(k // group)[None, None, :]) % 4
    if format == "e8m0":
        encoded = torch.tensor([125, 126, 127, 128], dtype=torch.uint8)[indices]
    else:
        encoded = torch.tensor([.25, .5, 1.5, 2.])[indices].to(torch.float8_e4m3fn)
    return encoded.contiguous()


def _expanded_scale(scale, group):
    if scale is None:
        return 1.
    if scale.dtype == torch.uint8:
        decoded = torch.exp2(scale.float() - 127)
        decoded = torch.where(scale == 255, float("nan"), decoded)
    else:
        decoded = scale.float()
    return decoded.repeat_interleave(group, dim=2)


def _run_case(shape, a_format, b_format, a_k_pack, b_k_pack, scale_format, with_lhs, with_rhs, nan_side=None):
    m, n, k = shape
    group = 32 if scale_format == "e8m0" else 16
    a, ar = _operand(m, k, a_format, 1 if a_k_pack else 0)
    b, br = _operand(k, n, b_format, 0 if b_k_pack else 1)
    sa = _scales(m, k, scale_format) if with_lhs else None
    sb = _scales(n, k, scale_format) if with_rhs else None
    if nan_side in ("lhs", "both"):
        sa[0, 3, 1] = 255
    if nan_side in ("rhs", "both"):
        sb[1, 5, 0] = 255
    lhs_scale = _expanded_scale(sa, group)
    rhs_scale = _expanded_scale(sb, group)
    if sb is not None:
        rhs_scale = rhs_scale.transpose(1, 2)
    expected = (ar * lhs_scale) @ (br * rhs_scale) + torch.tensor([.125, .25])[:, None, None]
    actual = torch.empty((2, m, n), dtype=torch.float32, device="npu")
    _batch_scaled[(1, )](a.to("npu"), b.to("npu"), sa.to("npu") if sa is not None else None,
                         sb.to("npu") if sb is not None else None, actual, 2, m, n, k, a_format, b_format, a_k_pack,
                         b_k_pack, group)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("shape,a_format,b_format,a_k_pack,b_k_pack", [
    ((16, 32, 32), "fp16", "bf16", True, True),
    ((32, 64, 64), "e2m1", "fp16", True, True),
    ((64, 32, 64), "e2m1", "e2m1", False, False),
    ((32, 32, 128), "e2m1", "e2m1", True, True),
])
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
@pytest.mark.parametrize("with_lhs,with_rhs", [(False, False), (False, True), (True, False), (True, True)])
def test_dot_scaled_batch_scale_presence(shape, a_format, b_format, a_k_pack, b_k_pack, scale_format, with_lhs,
                                         with_rhs):
    require_dot_scaled_formats(a_format, b_format, scale_format if with_lhs or with_rhs else None)
    _run_case(shape, a_format, b_format, a_k_pack, b_k_pack, scale_format, with_lhs, with_rhs)


@pytest.mark.parametrize("nan_side", ["lhs", "rhs", "both"])
def test_dot_scaled_batch_nan(nan_side):
    require_dot_scaled_formats("e2m1", "fp16", "e8m0")
    _run_case((32, 64, 64), "e2m1", "fp16", False, True, "e8m0", True, True, nan_side)


def test_dot_scaled_batch_extreme_scales():
    # The two batches reverse 0/254 scales and operand exponents. Scaled
    # operands and results remain normal, finite, and exactly representable.
    # This detects either scale leakage between batches or flushing code 0.
    m = n = 32
    k = 64
    codes = torch.tensor([[0, 254], [254, 0]], dtype=torch.uint8)
    sa = codes[:, None, :].expand(2, m, 2).contiguous()
    sb = (254 - codes)[:, None, :].expand(2, n, 2).contiguous()
    exponent = torch.where(codes == 0, 64., -64.).repeat_interleave(32, dim=1)
    a = torch.exp2(exponent)[:, None, :].expand(2, m, k).to(torch.bfloat16).contiguous()
    b = torch.exp2(-exponent)[:, :, None].expand(2, k, n).to(torch.bfloat16).contiguous()
    expected = torch.full((2, m, n), float(k)) + torch.tensor([.125, .25])[:, None, None]
    actual = torch.empty((2, m, n), dtype=torch.float32, device="npu")
    _batch_scaled[(1, )](a.to("npu"), b.to("npu"), sa.to("npu"), sb.to("npu"), actual, 2, m, n, k, "bf16", "bf16", True,
                         True, 32)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
