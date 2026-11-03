# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95
from triton.compiler.errors import CompilationError


@triton.jit
def _scaled_dot(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr, A_K_PACK: tl.constexpr, B_K_PACK: tl.constexpr,
                GROUP: tl.constexpr, INITIAL: tl.constexpr = .125, OUT_DTYPE: tl.constexpr = tl.float32):
    AM: tl.constexpr = M // 2 if A_FORMAT == "e2m1" and not A_K_PACK else M
    AK: tl.constexpr = K // 2 if A_FORMAT == "e2m1" and A_K_PACK else K
    BK: tl.constexpr = K // 2 if B_FORMAT == "e2m1" and B_K_PACK else K
    BN: tl.constexpr = N // 2 if B_FORMAT == "e2m1" and not B_K_PACK else N
    a = tl.load(a_ptr + tl.arange(0, AM)[:, None] * AK + tl.arange(0, AK)[None, :])
    b = tl.load(b_ptr + tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :])
    sa = None
    sb = None
    if sa_ptr is not None:
        sa = tl.load(sa_ptr + tl.arange(0, M)[:, None] * (K // GROUP) + tl.arange(0, K // GROUP)[None, :])
    if sb_ptr is not None:
        sb = tl.load(sb_ptr + tl.arange(0, N)[:, None] * (K // GROUP) + tl.arange(0, K // GROUP)[None, :])
    acc = tl.full((M, N), INITIAL, OUT_DTYPE)
    out = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc, lhs_k_pack=A_K_PACK, rhs_k_pack=B_K_PACK,
                        out_dtype=OUT_DTYPE)
    tl.store(out_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], out)


def _operand(rows, cols, format, axis, phase=0):
    codes = (torch.arange(rows)[:, None] * 7 + torch.arange(cols)[None, :] * 5 + phase) % 16
    if format == "e2m1":
        values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])[codes]
        packed = (codes[0::2, :] | (codes[1::2, :] << 4)) if axis == 0 else (codes[:, 0::2] | (codes[:, 1::2] << 4))
        return packed.to(torch.uint8).contiguous().npu(), values
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2":
             torch.float8_e5m2}[format]
    values = ((codes.float() - 7) / 8).to(dtype)
    return values.contiguous().npu(), values.float()


@pytest.mark.parametrize("a_format,b_format,a_k_pack,b_k_pack,scale_format,with_lhs,with_rhs", [
    ("e2m1", "fp16", True, True, "e8m0", True, True),
    ("e2m1", "e5m2", False, True, "e8m0", True, False),
    ("e4m3", "e2m1", True, False, "e8m0", False, True),
    ("e2m1", "e2m1", False, False, "e8m0", False, False),
    ("e2m1", "e2m1", True, True, "e8m0", True, True),
    ("e2m1", "e2m1", True, True, "e4m3", True, True),
    ("fp16", "bf16", True, True, "e8m0", False, True),
    ("bf16", "fp16", True, True, "e8m0", True, False),
    ("fp16", "bf16", True, True, "e8m0", False, False),
])
def test_dot_scaled_fallback_formats(a_format, b_format, a_k_pack, b_k_pack, scale_format, with_lhs, with_rhs):
    if (a_format not in ("fp16", "bf16") or b_format not in ("fp16", "bf16") or scale_format == "e4m3"):
        if not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
            pytest.skip("FP4/FP8 operands and FP8 scales require Ascend 950")
    m, n, k = 16, 32, 32
    a, ar = _operand(m, k, a_format, 1 if a_k_pack else 0)
    b, br = _operand(k, n, b_format, 0 if b_k_pack else 1, phase=3)
    group = 32 if scale_format == "e8m0" else 16

    def scales(rows, present):
        if not present:
            return None, 1.
        indices = (torch.arange(rows)[:, None] + torch.arange(k // group)[None, :]) % 4
        if scale_format == "e8m0":
            encoded = torch.tensor([125, 127, 128, 129], dtype=torch.uint8)[indices]
            decoded = torch.exp2(encoded.float() - 127)
        else:
            encoded = torch.tensor([.125, .375, 1.5, 3.])[indices].to(torch.float8_e4m3fn)
            decoded = encoded.float()
        return encoded.contiguous().npu(), decoded.repeat_interleave(group, 1)

    sa, sar = scales(m, with_lhs)
    sb, sbr = scales(n, with_rhs)
    expected = (ar * sar) @ (br * (sbr.T if with_rhs else sbr)) + .125
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, sa, sb, actual, m, n, k, a_format, b_format, a_k_pack, b_k_pack, group)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("code,exponent,expected", [(0, 120, .25), (254, -120, 4096.)])
@pytest.mark.parametrize("side", ["lhs", "rhs"])
def test_dot_scaled_e8m0_extreme(code, exponent, expected, side):
    # Compensating operand exponents make the final result normal and finite.
    # A near-zero expected result would hide an incorrect zero-scale decode.
    m, n, k = 16, 32, 32
    a = torch.full((m, k), 2.**exponent if side == "lhs" else 1., dtype=torch.bfloat16, device="npu")
    b = torch.full((k, n), 2.**exponent if side == "rhs" else 1., dtype=torch.bfloat16, device="npu")
    scale = torch.full((m if side == "lhs" else n, 1), code, dtype=torch.uint8, device="npu")
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, scale if side == "lhs" else None, scale if side == "rhs" else None, actual, m, n, k,
                       "bf16", "bf16", True, True, 32, INITIAL=0.)
    torch.testing.assert_close(actual.cpu(), torch.full((m, n), expected), atol=0, rtol=0)


@pytest.mark.parametrize("side", ["lhs", "rhs"])
def test_dot_scaled_e8m0_nan(side):
    m, n, k = 16, 32, 32
    a = torch.ones((m, k), dtype=torch.float16, device="npu")
    b = torch.ones((k, n), dtype=torch.float16, device="npu")
    scale = torch.full((m if side == "lhs" else n, 1), 127, dtype=torch.uint8, device="npu")
    scale[3, 0] = 255
    expected = torch.full((m, n), k + .125)
    if side == "lhs":
        expected[3, :] = float("nan")
    else:
        expected[:, 3] = float("nan")
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, scale if side == "lhs" else None, scale if side == "rhs" else None, actual, m, n, k,
                       "fp16", "fp16", True, True, 32)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0, equal_nan=True)


def test_dot_scaled_f32_intermediates():
    # A*2 exceeds FP16, while (A*2)@(B/2)+acc remains finite in FP32.
    m, n, k = 16, 32, 32
    a = torch.full((m, k), 65504., dtype=torch.float16, device="npu")
    b = torch.full((k, n), 2.**-10, dtype=torch.float16, device="npu")
    sa = torch.full((m, 1), 128, dtype=torch.uint8, device="npu")
    sb = torch.full((n, 1), 126, dtype=torch.uint8, device="npu")
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, sa, sb, actual, m, n, k, "fp16", "fp16", True, True, 32, INITIAL=.5, OUT_DTYPE=tl.float32)
    expected = torch.full((m, n), 65504. * 2.**-10 * k + .5)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("out_dtype", [tl.float16, tl.bfloat16])
def test_dot_scaled_rejects_non_f32_output(out_dtype):
    # Keep the shared frontend's existing output-type contract unchanged.
    a = torch.ones((16, 32), dtype=torch.float16, device="npu")
    b = torch.ones((32, 32), dtype=torch.float16, device="npu")
    out = torch.empty((16, 32), dtype=torch.float32, device="npu")
    with pytest.raises(CompilationError, match="Only float32 is supported for out_dtype"):
        _scaled_dot[(1, )](a, b, None, None, out, 16, 32, 32, "fp16", "fp16", True, True, 32, OUT_DTYPE=out_dtype)
