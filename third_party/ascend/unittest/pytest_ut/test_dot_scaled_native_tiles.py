# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from test_dot_scaled_boundaries import _fp4_host

from dot_scaled_test_utils import require_dot_scaled_formats


@triton.jit
def _native_tile_dot(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr, stride_bk, stride_bn, M: tl.constexpr, N: tl.constexpr,
                     K: tl.constexpr, A_FORMAT: tl.constexpr, B_FORMAT: tl.constexpr, EXPLICIT_ACC: tl.constexpr):
    AK: tl.constexpr = K // 2 if A_FORMAT == "e2m1" else K
    BK: tl.constexpr = K // 2 if B_FORMAT == "e2m1" else K
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    acc = .125 + rows[:, None].to(tl.float32) / 128 + cols[None, :].to(tl.float32) / 64
    for block in range(2):
        a = tl.load(a_ptr + rows[:, None] * (2 * AK) + block * AK + tl.arange(0, AK)[None, :])
        b = tl.load(b_ptr + (block * BK + tl.arange(0, BK)[:, None]) * stride_bk + cols[None, :] * stride_bn)
        if sa_ptr is not None:
            sa = tl.load(sa_ptr + rows[:, None] * (2 * K // 32) + block * (K // 32) + tl.arange(0, K // 32)[None, :])
        else:
            sa = None
        if sb_ptr is not None:
            sb = tl.load(sb_ptr + cols[:, None] * (2 * K // 32) + block * (K // 32) + tl.arange(0, K // 32)[None, :])
        else:
            sb = None
        if EXPLICIT_ACC:
            acc = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc)
        else:
            acc += tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT)
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], acc, cache_modifier=".wt")


def _operand(rows, cols, format, pack_axis, phase):
    if format == "e2m1":
        return _fp4_host(rows, cols, pack_axis, phase)
    dtype = torch.float8_e4m3fn if format == "e4m3" else torch.float8_e5m2
    values = (((torch.arange(rows)[:, None] * 3 + torch.arange(cols)[None, :] * 5 + phase) % 16 - 8).float() /
              8).to(dtype)
    return values, values.float()


@pytest.mark.parametrize("shape", [(128, 32, 128), (32, 128, 128), (128, 128, 128), (128, 128, 256)])
@pytest.mark.parametrize("a_format,b_format", [("e2m1", "e2m1"), ("e4m3", "e4m3"), ("e5m2", "e4m3")])
@pytest.mark.parametrize("with_lhs,with_rhs", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("explicit_acc", [False, True])
def test_dot_scaled_native_large_mn(shape, a_format, b_format, with_lhs, with_rhs, explicit_acc):
    require_dot_scaled_formats(a_format, b_format)
    m, n, k = shape
    a, ar = _operand(m, 2 * k, a_format, 1, 0)
    b, br = _operand(2 * k, n, b_format, 0, 3)
    # Column-major B exercises indirect addresses or a full load followed by
    # a transpose. Both must fit UB with the two original K blocks intact.
    b = b.T.contiguous().T
    ai = (torch.arange(m)[:, None] + torch.arange(2 * k // 32)[None, :]) % 4
    bi = (torch.arange(n)[:, None] * 3 + torch.arange(2 * k // 32)[None, :]) % 4
    sa = torch.tensor([125, 127, 128, 129], dtype=torch.uint8)[ai]
    sb = torch.tensor([126, 125, 128, 127], dtype=torch.uint8)[bi]
    if with_lhs:
        ar = ar * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
    if with_rhs:
        br = br * torch.exp2(sb.float() - 127).repeat_interleave(32, 1).T
    expected = .125 + torch.arange(m)[:, None].float() / 128 + torch.arange(n)[None, :].float() / 64
    for block in range(2):
        expected += ar[:, block * k:(block + 1) * k] @ br[block * k:(block + 1) * k]
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _native_tile_dot[(1, )](a.to("npu"), b.to("npu"), sa.to("npu") if with_lhs else None,
                            sb.to("npu") if with_rhs else None, actual, *b.stride(), m, n, k, a_format, b_format,
                            explicit_acc)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _batch_native_tile_dot(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           FORMAT: tl.constexpr, NONZERO_ACC: tl.constexpr):
    PK: tl.constexpr = K // 2 if FORMAT == "e2m1" else K
    batch = tl.arange(0, 2)[:, None, None]
    rows = tl.arange(0, M)[None, :, None]
    cols = tl.arange(0, N)[None, None, :]
    a = tl.load(a_ptr + batch * M * PK + rows * PK + tl.arange(0, PK)[None, None, :])
    b = tl.load(b_ptr + batch * PK * N + tl.arange(0, PK)[None, :, None] * N + cols)
    if sa_ptr is not None:
        sa = tl.load(sa_ptr + batch * M * (K // 32) + rows * (K // 32) + tl.arange(0, K // 32)[None, None, :])
    else:
        sa = None
    if sb_ptr is not None:
        sb = tl.load(sb_ptr + batch * N * (K // 32) + tl.arange(0, N)[None, :, None] * (K // 32) +
                     tl.arange(0, K // 32)[None, None, :])
    else:
        sb = None
    if NONZERO_ACC:
        acc = .125 + batch.to(tl.float32) / 4 + rows.to(tl.float32) / 128 + cols.to(tl.float32) / 64
    else:
        acc = tl.full((2, M, N), 0, tl.float32)
    out = tl.dot_scaled(a, sa, FORMAT, b, sb, FORMAT, acc)
    tl.store(out_ptr + batch * M * N + rows * N + cols, out, cache_modifier=".wt")


@pytest.mark.parametrize("shape", [(64, 16, 128), (16, 64, 128)])
@pytest.mark.parametrize("format", ["e2m1", "e4m3"])
@pytest.mark.parametrize("with_lhs,with_rhs", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("nonzero_acc", [False, True])
def test_dot_scaled_native_batched_mn(shape, format, with_lhs, with_rhs, nonzero_acc):
    require_dot_scaled_formats(format, format)
    m, n, k = shape
    a, ar = zip(*[_operand(m, k, format, 1, batch * 3) for batch in range(2)])
    b, br = zip(*[_operand(k, n, format, 0, batch * 5 + 3) for batch in range(2)])
    a, ar, b, br = (torch.stack(values) for values in (a, ar, b, br))
    ai = (torch.arange(2)[:, None, None] * 2 + torch.arange(m)[None, :, None] +
          torch.arange(k // 32)[None, None, :]) % 4
    bi = (torch.arange(2)[:, None, None] * 3 + torch.arange(n)[None, :, None] * 3 +
          torch.arange(k // 32)[None, None, :]) % 4
    sa = torch.tensor([125, 127, 128, 129], dtype=torch.uint8)[ai]
    sb = torch.tensor([126, 125, 128, 127], dtype=torch.uint8)[bi]
    if with_lhs:
        ar = ar * torch.exp2(sa.float() - 127).repeat_interleave(32, 2)
    if with_rhs:
        br = br * torch.exp2(sb.float() - 127).repeat_interleave(32, 2).transpose(1, 2)
    expected = ar @ br
    if nonzero_acc:
        expected += (.125 + torch.arange(2)[:, None, None].float() / 4 + torch.arange(m)[None, :, None].float() / 128 +
                     torch.arange(n)[None, None, :].float() / 64)
    actual = torch.empty((2, m, n), dtype=torch.float32, device="npu")
    _batch_native_tile_dot[(1, )](a.to("npu"), b.to("npu"), sa.to("npu") if with_lhs else None,
                                  sb.to("npu") if with_rhs else None, actual, m, n, k, format, nonzero_acc)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
