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


def test_dot_scaled_f32_intermediates():
    # A*2 exceeds FP16, while (A*2)@(B/2)+acc remains finite in FP32.
    m, n, k = 16, 32, 32
    a = torch.full((m, k), torch.finfo(torch.float16).max, dtype=torch.float16, device="npu")
    b = torch.full((k, n), 2.**-10, dtype=torch.float16, device="npu")
    sa = torch.full((m, 1), 128, dtype=torch.uint8, device="npu")
    sb = torch.full((n, 1), 126, dtype=torch.uint8, device="npu")
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, sa, sb, actual, m, n, k, "fp16", "fp16", True, True, 32, INITIAL=.5, OUT_DTYPE=tl.float32)
    expected = torch.full((m, n), torch.finfo(torch.float16).max * 2.**-10 * k + .5)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("out_dtype", [tl.float16, tl.bfloat16])
def test_dot_scaled_rejects_non_f32_output(out_dtype):
    # Keep the shared frontend's existing output-type contract unchanged.
    a = torch.ones((16, 32), dtype=torch.float16, device="npu")
    b = torch.ones((32, 32), dtype=torch.float16, device="npu")
    out = torch.empty((16, 32), dtype=torch.float32, device="npu")
    with pytest.raises(CompilationError, match="Only float32 is supported for out_dtype"):
        _scaled_dot[(1, )](a, b, None, None, out, 16, 32, 32, "fp16", "fp16", True, True, 32, OUT_DTYPE=out_dtype)


@pytest.mark.parametrize("a_dtype,b_dtype", [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)])
def test_dot_scaled_mixed_half_precision(a_dtype, b_dtype):
    m, n, k = 16, 32, 32
    a = (torch.arange(m * k).reshape(m, k) % 17 - 8).to(a_dtype) / 8
    b = (torch.arange(k * n).reshape(k, n) % 13 - 6).to(b_dtype) / 8
    sa = torch.full((m, 1), 128, dtype=torch.uint8, device="npu")  # E8M0 scale 2.
    sb = torch.full((n, 1), 126, dtype=torch.uint8, device="npu")  # E8M0 scale 0.5.
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    formats = {torch.float16: "fp16", torch.bfloat16: "bf16"}
    _scaled_dot[(1, )](a.npu(), b.npu(), sa, sb, actual, m, n, k, formats[a_dtype], formats[b_dtype], True, True, 32)
    torch.testing.assert_close(actual.cpu(), (a.float() * 2) @ (b.float() * .5) + .125, atol=0, rtol=0)
