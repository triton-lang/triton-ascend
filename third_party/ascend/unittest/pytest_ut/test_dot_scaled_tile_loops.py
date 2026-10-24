# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_dot(a_ptr, b_ptr, c_ptr, out_ptr, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    if B == 0:
        a = tl.load(a_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
        b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
        out_offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    else:
        ab = tl.arange(0, B)[:, None, None] * M * K
        bb = tl.arange(0, B)[:, None, None] * K * N
        a = tl.load(a_ptr + ab + tl.arange(0, M)[None, :, None] * K + tl.arange(0, K)[None, None, :])
        b = tl.load(b_ptr + bb + tl.arange(0, K)[None, :, None] * N + tl.arange(0, N)[None, None, :])
        ob = tl.arange(0, B)[:, None, None] * M * N
        out_offsets = ob + tl.arange(0, M)[None, :, None] * N + tl.arange(0, N)[None, None, :]
    c = tl.load(c_ptr + out_offsets)
    out = tl.dot_scaled(a, None, "fp16", b, None, "bf16", c)
    tl.store(out_ptr + out_offsets, out)


@pytest.mark.parametrize("shape", [(64, 32, 32), (32, 64, 32), (64, 64, 32), (32, 32, 64), (64, 64, 64)])
@pytest.mark.parametrize("batch", [0, 2])
def test_dot_scaled_tiling_loops(shape, batch):
    m, n, k = shape
    batches = max(batch, 1)
    a = ((torch.arange(batches * m * k).reshape(batches, m, k) * 3 % 17 - 8).float() / 4).half()
    b = ((torch.arange(batches * k * n).reshape(batches, k, n) * 5 % 13 - 6).float() / 8).bfloat16()
    c = ((torch.arange(batches * m * n).reshape(batches, m, n) % 7).float() / 8)
    if batch == 0:
        a, b, c = a[0], b[0], c[0]
    expected = a.float() @ b.float() + c
    actual = torch.empty_like(c, device="npu")
    _tiled_dot[(1, )](a.to("npu"), b.to("npu"), c.to("npu"), actual, batch, m, n, k)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
