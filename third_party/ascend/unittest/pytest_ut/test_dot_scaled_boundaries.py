# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from dot_scaled_test_utils import require_dot_scaled_formats


@triton.jit
def _dot_boundary(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr, A_K_PACK: tl.constexpr, B_K_PACK: tl.constexpr,
                  GROUP: tl.constexpr, INITIAL: tl.constexpr):
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
    acc = tl.full((M, N), INITIAL, tl.float32)
    out = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc, lhs_k_pack=A_K_PACK, rhs_k_pack=B_K_PACK)
    tl.store(out_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], out)


def _fp4_host(rows, cols, axis, phase=0):
    codes = (torch.arange(rows)[:, None] * 7 + torch.arange(cols)[None, :] * 5 + phase) % 16
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])[codes]
    if axis == 0:
        packed = codes[0::2, :] | (codes[1::2, :] << 4)
    else:
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.to(torch.uint8).contiguous(), values


@pytest.mark.parametrize("code,other_exponent,expected", [(0, 120, .5), (254, -120, 8192.)])
@pytest.mark.parametrize("scaled_side", ["lhs", "rhs"])
def test_dot_scaled_e8m0_extreme(code, other_exponent, expected, scaled_side):
    # Final values are normal and finite. In particular, scale zero is not
    # tested against a near-zero result that an absolute tolerance could hide.
    require_dot_scaled_formats("e2m1" if scaled_side == "lhs" else "bf16", "bf16" if scaled_side == "lhs" else "e2m1",
                               "e8m0")
    m, n, k = 16, 32, 64
    if scaled_side == "lhs":
        a = torch.full((m, k // 2), 0x22, dtype=torch.uint8)
        b = torch.full((k, n), 2.**other_exponent, dtype=torch.bfloat16)
        sa = torch.full((m, k // 32), code, dtype=torch.uint8).to("npu")
        sb = None
        a_format, b_format = "e2m1", "bf16"
    else:
        a = torch.full((m, k), 2.**other_exponent, dtype=torch.bfloat16)
        b = torch.full((k // 2, n), 0x22, dtype=torch.uint8)
        sa = None
        sb = torch.full((n, k // 32), code, dtype=torch.uint8).to("npu")
        a_format, b_format = "bf16", "e2m1"
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _dot_boundary[(1, )](a.to("npu"), b.to("npu"), sa, sb, actual, m, n, k, a_format, b_format, True, True, 32, 0.)
    torch.testing.assert_close(actual.cpu(), torch.full((m, n), expected), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 256, 256)])
@pytest.mark.parametrize("a_k_pack,b_k_pack", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_dot_scaled_large_packed(shape, a_k_pack, b_k_pack, scale_format):
    require_dot_scaled_formats("e2m1", "e2m1", scale_format)
    m, n, k = shape
    a, ar = _fp4_host(m, k, 1 if a_k_pack else 0)
    b, br = _fp4_host(k, n, 0 if b_k_pack else 1, phase=3)
    group = 32 if scale_format == "e8m0" else 16
    ai = (torch.arange(m)[:, None] + torch.arange(k // group)[None, :]) % 4
    bi = (torch.arange(n)[:, None] * 3 + torch.arange(k // group)[None, :]) % 4
    if scale_format == "e8m0":
        sa = torch.tensor([125, 127, 128, 129], dtype=torch.uint8)[ai]
        sb = torch.tensor([126, 125, 128, 127], dtype=torch.uint8)[bi]
        sar, sbr = torch.exp2(sa.float() - 127), torch.exp2(sb.float() - 127)
    else:
        sa = torch.tensor([.125, .375, 1.5, 3.])[ai].to(torch.float8_e4m3fn)
        sb = torch.tensor([.5, 1.5, .375, 2.])[bi].to(torch.float8_e4m3fn)
        sar, sbr = sa.float(), sb.float()
    expected = (ar * sar.repeat_interleave(group, 1)) @ (br * sbr.repeat_interleave(group, 1).T) + .125
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _dot_boundary[(1, )](a.to("npu"), b.to("npu"), sa.to("npu"), sb.to("npu"), actual, m, n, k, "e2m1", "e2m1",
                         a_k_pack, b_k_pack, group, .125)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 32, 32), (64, 64, 64)])
def test_dot_scaled_half_output(output_dtype, shape):
    require_dot_scaled_formats("e2m1", "bf16", "e8m0")
    m, n, k = shape
    a, ar = _fp4_host(m, k, 0)
    b = ((torch.arange(k)[:, None] + 3 * torch.arange(n)[None, :]) % 8 - 3).to(torch.bfloat16) / 8
    sa = torch.full((m, k // 32), 127, dtype=torch.uint8, device="npu")
    expected = (ar @ b.float() + .125).to(output_dtype)
    actual = torch.empty((m, n), dtype=output_dtype, device="npu")
    _dot_boundary[(1, )](a.to("npu"), b.to("npu"), sa, None, actual, m, n, k, "e2m1", "bf16", False, True, 32, .125)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _batch_dot_boundary(a_ptr, b_ptr, out_ptr, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        PACK_K: tl.constexpr):
    AM: tl.constexpr = M if PACK_K else M // 2
    AK: tl.constexpr = K // 2 if PACK_K else K
    ab = tl.arange(0, B)[:, None, None] * (AM * AK)
    a = tl.load(a_ptr + ab + tl.arange(0, AM)[None, :, None] * AK + tl.arange(0, AK)[None, None, :])
    bb = tl.arange(0, B)[:, None, None] * (K * N)
    b = tl.load(b_ptr + bb + tl.arange(0, K)[None, :, None] * N + tl.arange(0, N)[None, None, :])
    acc = tl.full((B, M, N), .125, tl.float32)
    out = tl.dot_scaled(a, None, "e2m1", b, None, "fp16", acc, lhs_k_pack=PACK_K)
    ob = tl.arange(0, B)[:, None, None] * (M * N)
    tl.store(out_ptr + ob + tl.arange(0, M)[None, :, None] * N + tl.arange(0, N)[None, None, :], out)


@pytest.mark.parametrize("pack_k", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dot_scaled_batch(pack_k, output_dtype):
    require_dot_scaled_formats("e2m1", "fp16")
    batch, m, n, k = 2, 16, 32, 32
    packed, reference = zip(*[_fp4_host(m, k, 1 if pack_k else 0, phase=i * 3) for i in range(batch)])
    a = torch.stack(packed)
    ar = torch.stack(reference)
    b = ((torch.arange(batch * k * n).reshape(batch, k, n) * 3 % 11 - 5).float() / 8).half()
    b[1] *= 2
    expected = (ar @ b.float() + .125).to(output_dtype)
    actual = torch.empty((batch, m, n), dtype=output_dtype, device="npu")
    _batch_dot_boundary[(1, )](a.to("npu"), b.to("npu"), actual, batch, m, n, k, pack_k)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
