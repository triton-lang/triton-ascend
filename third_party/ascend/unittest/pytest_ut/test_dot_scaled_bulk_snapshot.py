# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from dot_scaled_test_utils import require_dot_scaled_formats


@triton.jit
def _bulk_snapshot(A, B, Snapshot, C, stride_bk, stride_bn, MODE: tl.constexpr):
    m = tl.arange(0, 16)
    n = tl.arange(0, 64)
    k = tl.arange(0, 64)
    bp = B + k[:, None] * stride_bk + n[None, :] * stride_bn
    b = tl.load(bp, volatile=MODE == "volatile")
    a = tl.load(A + m[:, None] * 64 + k[None, :])
    if MODE == "alias_store":
        tl.store(bp, tl.full((64, 64), 0, tl.float16))
    if MODE == "shared":
        tl.store(Snapshot + k[:, None] * 64 + n[None, :], b)
    c = tl.dot_scaled(a, None, "fp16", b, None, "fp16")
    # Keep memory lowering in the unblocked path used by native MX kernels.
    tl.store(C + m[:, None] * 64 + n[None, :], c, cache_modifier=".wt")


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
@pytest.mark.parametrize("mode", ["plain", "alias_store", "shared", "volatile"])
def test_dot_scaled_bulk_snapshot(layout, mode):
    a = ((torch.arange(16 * 64).reshape(16, 64) % 11 - 5) / 8).half()
    b = ((torch.arange(64 * 64).reshape(64, 64) % 13 - 6) / 4).half()
    expected = a.float() @ b.float()
    strides = {"contiguous": (64, 1), "transpose": (1, 64), "strided": (160, 2)}[layout]
    bn = torch.empty_strided((64, 64), strides, dtype=torch.float16, device="npu")
    bn.copy_(b)
    snapshot = torch.empty((64, 64), dtype=torch.float16, device="npu")
    actual = torch.empty((16, 64), dtype=torch.float32, device="npu")
    _bulk_snapshot[(1, )](a.to("npu"), bn, snapshot, actual, *strides, mode)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    if mode == "alias_store":
        torch.testing.assert_close(bn.cpu(), torch.zeros_like(b), atol=0, rtol=0)
    if mode == "shared":
        torch.testing.assert_close(snapshot.cpu(), b, atol=0, rtol=0)


@triton.jit
def _native_bulk_snapshot(A, B, Snapshot, C, stride_bk, stride_bn, MODE: tl.constexpr):
    m = tl.arange(0, 16)
    n = tl.arange(0, 64)
    k = tl.arange(0, 64)
    bp = B + k[:, None] * stride_bk + n[None, :] * stride_bn
    b = tl.load(bp, volatile=MODE == "volatile")
    a = tl.load(A + m[:, None] * 64 + k[None, :])
    if MODE == "alias_store":
        tl.store(bp, tl.full((64, 64), 0, tl.float32))
    if MODE == "shared":
        tl.store(Snapshot + k[:, None] * 64 + n[None, :], b)
    c = tl.dot_scaled(a, None, "e4m3", b, None, "e4m3")
    tl.store(C + m[:, None] * 64 + n[None, :], c, cache_modifier=".wt")


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
@pytest.mark.parametrize("mode", ["plain", "alias_store", "shared", "volatile"])
def test_dot_scaled_native_bulk_snapshot(layout, mode):
    require_dot_scaled_formats("e4m3", "e4m3")
    a = ((torch.arange(16 * 64).reshape(16, 64) % 11 - 5) / 8).to(torch.float8_e4m3fn)
    b = ((torch.arange(64 * 64).reshape(64, 64) % 13 - 6) / 4).to(torch.float8_e4m3fn)
    expected = a.float() @ b.float()
    strides = {"contiguous": (64, 1), "transpose": (1, 64), "strided": (160, 2)}[layout]
    # Transfer the CPU-encoded FP8 bytes without relying on torch NPU FP8
    # allocation/copy support. Reinterpretation preserves the kernel dtype.
    an = a.view(torch.uint8).to("npu")
    bn = torch.empty_strided((64, 64), strides, dtype=torch.uint8, device="npu")
    bn.copy_(b.view(torch.uint8))
    snapshot = torch.empty((64, 64), dtype=torch.uint8, device="npu")
    actual = torch.empty((16, 64), dtype=torch.float32, device="npu")
    _native_bulk_snapshot[(1, )](triton.reinterpret(an, tl.float8e4nv), triton.reinterpret(bn, tl.float8e4nv),
                                 triton.reinterpret(snapshot, tl.float8e4nv), actual, *strides, mode)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    if mode == "alias_store":
        torch.testing.assert_close(bn.cpu(), torch.zeros_like(b.view(torch.uint8)), atol=0, rtol=0)
    if mode == "shared":
        torch.testing.assert_close(snapshot.cpu(), b.view(torch.uint8), atol=0, rtol=0)
