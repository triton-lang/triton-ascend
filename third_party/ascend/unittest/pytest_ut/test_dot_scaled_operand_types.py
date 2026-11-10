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
                A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr, OUT_DTYPE: tl.constexpr = tl.float32):
    a = tl.load(a_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    sa = tl.load(sa_ptr + tl.arange(0, M)[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    sb = None
    if sb_ptr is not None:
        sb = tl.load(sb_ptr + tl.arange(0, N)[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    acc = tl.full((M, N), .125, OUT_DTYPE)
    out = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc, out_dtype=OUT_DTYPE)
    tl.store(out_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], out)


@pytest.mark.parametrize("a_format,b_format", [("fp16", "bf16"), ("bf16", "fp16"), ("bf16", "bf16"), ("e4m3", "bf16"),
                                               ("bf16", "e5m2"), ("fp16", "e5m2")])
@pytest.mark.parametrize("rhs_scale", [False, True])
def test_dot_scaled_operand_types(a_format, b_format, rhs_scale):
    if any(fmt in ("e4m3", "e5m2") for fmt in (a_format, b_format)) and not is_compile_on_910_95():
        pytest.skip("FP8 operand conversion requires A5")
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    m, n, k = 16, 32, 64
    a = ((torch.arange(m * k).reshape(m, k) % 17 - 8).float() / 8).to(dtypes[a_format])
    b = ((torch.arange(k * n).reshape(k, n) % 13 - 6).float() / 8).to(dtypes[b_format])
    sa = torch.tensor([128, 126], dtype=torch.uint8).repeat(m, 1).npu()
    sb = torch.tensor([126, 128], dtype=torch.uint8).repeat(n, 1).npu() if rhs_scale else None
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a.npu(), b.npu(), sa, sb, actual, m, n, k, a_format, b_format)

    # Upstream converts both operands before applying scales in FP16/BF16.
    compute_dtype = torch.float16 if "fp16" in (a_format, b_format) else torch.bfloat16
    a_scaled = a.to(compute_dtype) * torch.tensor([2., .5], dtype=compute_dtype).repeat_interleave(32)
    b_scaled = b.to(compute_dtype)
    if rhs_scale:
        b_scaled = b_scaled * torch.tensor([.5, 2.], dtype=compute_dtype).repeat_interleave(32)[:, None]
    expected = a_scaled.float() @ b_scaled.float() + .125
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


def test_dot_scaled_converts_before_scaling():
    # The BF16 value is halfway to the smallest FP16 subnormal: RTNE gives 0.
    # Scaling first in FP32 would instead preserve a nonzero contribution.
    m, n, k = 16, 32, 32
    a = torch.full((m, k), 2.**-25, dtype=torch.bfloat16, device="npu")
    b = torch.ones((k, n), dtype=torch.float16, device="npu")
    sa = torch.full((m, 1), 128, dtype=torch.uint8, device="npu")
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _scaled_dot[(1, )](a, b, sa, None, actual, m, n, k, "bf16", "fp16")
    torch.testing.assert_close(actual.cpu(), torch.full((m, n), .125), atol=0, rtol=0)


@pytest.mark.parametrize("out_dtype", [tl.float16, tl.bfloat16])
def test_dot_scaled_rejects_non_f32_output(out_dtype):
    a = torch.ones((16, 32), dtype=torch.float16, device="npu")
    b = torch.ones((32, 32), dtype=torch.float16, device="npu")
    sa = torch.full((16, 1), 127, dtype=torch.uint8, device="npu")
    out = torch.empty((16, 32), dtype=torch.float32, device="npu")
    with pytest.raises(CompilationError, match="Only float32 is supported for out_dtype"):
        _scaled_dot[(1, )](a, b, sa, None, out, 16, 32, 32, "fp16", "fp16", OUT_DTYPE=out_dtype)
