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
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_dot_scaled_scale_nan(side, scale_format):
    if scale_format == "e4m3" and not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("FP8 scales require Ascend 950")
    m, n, k = 16, 32, 32
    a = torch.ones((m, k), dtype=torch.bfloat16, device="npu")
    b = torch.ones((k, n), dtype=torch.bfloat16, device="npu")
    group = 32 if scale_format == "e8m0" else 16
    dtype = torch.uint8 if scale_format == "e8m0" else torch.float8_e4m3fn
    scale = torch.full((m if side == "lhs" else n, k // group), 127 if scale_format == "e8m0" else 1., dtype=dtype)
    scale[3, 0] = 255 if scale_format == "e8m0" else float("nan")
    scale = scale.npu()
    expected = torch.full((m, n), k + .125)
    if side == "lhs":
        expected[3, :] = float("nan")
    else:
        expected[:, 3] = float("nan")
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, scale if side == "lhs" else None, scale if side == "rhs" else None, actual, m, n, k,
                       "bf16", "bf16", True, True, group)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("with_lhs,with_rhs", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("scale_format", ["e8m0", "e4m3"])
def test_dot_scaled_optional_and_floating_scales(with_lhs, with_rhs, scale_format):
    if scale_format == "e4m3" and not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("FP8 scales require Ascend 950")
    m, n, k = 16, 32, 32
    a = torch.ones((m, k), dtype=torch.bfloat16, device="npu")
    b = torch.ones((k, n), dtype=torch.bfloat16, device="npu")
    group = 32 if scale_format == "e8m0" else 16
    dtype = torch.uint8 if scale_format == "e8m0" else torch.float8_e4m3fn
    sa = torch.full((m, k // group), 128 if scale_format == "e8m0" else 2., dtype=dtype).npu() if with_lhs else None
    sb = torch.full((n, k // group), 126 if scale_format == "e8m0" else .5, dtype=dtype).npu() if with_rhs else None
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, sa, sb, actual, m, n, k, "bf16", "bf16", True, True, group)
    expected = torch.full((m, n), k * (2. if with_lhs else 1.) * (.5 if with_rhs else 1.) + .125)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
