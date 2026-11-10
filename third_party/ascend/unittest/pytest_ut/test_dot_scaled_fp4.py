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


@pytest.mark.parametrize("a_k_pack,b_k_pack", [(True, True), (True, False), (False, True), (False, False)])
def test_dot_scaled_fp4_packing(a_k_pack, b_k_pack):
    if not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("FP4 operands require Ascend 950")
    m, n, k = 16, 32, 32
    a, ar = _operand(m, k, "e2m1", 1 if a_k_pack else 0)
    b, br = _operand(k, n, "e2m1", 0 if b_k_pack else 1, phase=3)
    sa = torch.full((m, 1), 128, dtype=torch.uint8, device="npu")  # Scale 2.
    sb = torch.full((n, 1), 126, dtype=torch.uint8, device="npu")  # Scale 0.5.
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, sa, sb, actual, m, n, k, "e2m1", "e2m1", a_k_pack, b_k_pack, 32)
    torch.testing.assert_close(actual.cpu(), (ar * 2.) @ (br * .5) + .125, atol=0, rtol=0)
