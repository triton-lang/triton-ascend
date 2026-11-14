# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared M/N and independent matrix grouping with structured tails."""
import pytest
import torch
import triton
import triton.language as tl

from block_batching_test_utils import assert_applied, batching_target  # noqa: F401


@triton.jit
def grouped_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr, LOOP: tl.constexpr,
                INTEGER: tl.constexpr, T: tl.constexpr = 32):
    p = tl.program_id(0)
    r, c, k = tl.arange(0, T), tl.arange(0, T), tl.arange(0, T)
    if MODE == 0:
        r = p * T + r
    elif MODE == 1:
        c = p * T + c
    else:
        A += p * M * K
        B += p * K * N
        C += p * M * N
    acc = tl.full((T, T), 0, tl.int32 if INTEGER else tl.float32)
    if LOOP:
        for step in range(tl.cdiv(K, T)):
            kk = step * T + k
            a = tl.load(A + r[:, None] * K + kk[None, :], (r[:, None] < M) & (kk[None, :] < K), 0)
            b = tl.load(B + kk[:, None] * N + c[None, :], (kk[:, None] < K) & (c[None, :] < N), 0)
            acc = tl.dot(a, b, acc, input_precision='ieee', out_dtype=acc.dtype)
    else:
        a = tl.load(A + r[:, None] * K + k[None, :], (r[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + k[:, None] * N + c[None, :], (k[:, None] < K) & (c[None, :] < N), 0)
        acc = tl.dot(a, b, acc, input_precision='ieee', out_dtype=acc.dtype)
    tl.store(C + r[:, None] * N + c[None, :], acc, (r[:, None] < M) & (c[None, :] < N))


@pytest.mark.parametrize('mode,loop,integer,k', [(0, False, False, 32), (1, False, False, 32), (2, False, False, 32),
                                                 (0, True, False, 63), (1, True, False, 63), (2, True, False, 64),
                                                 (2, True, False, 63), (0, False, True, 32), (2, False, True, 32)])
def test_grouped_matrix_results(mode, loop, integer, k):
    m, n = (127, 32) if mode == 0 else ((32, 127) if mode == 1 else (31, 29))
    batch = (4, ) if mode == 2 else ()
    torch.manual_seed(881)
    dtype = torch.int8 if integer else torch.float32
    acpu = torch.randint(-4, 5, (*batch, m, k)).to(dtype)
    bcpu = torch.randint(-4, 5, (*batch, k, n)).to(dtype)
    expected = (acpu.double() @ bcpu.double()).to(torch.int32 if integer else torch.float32)
    a, b = acpu.npu(), bcpu.npu()
    storage = torch.full((expected.numel() + 128, ), -12345, device='npu', dtype=expected.dtype)
    c = storage[64:-64].view(expected.shape)
    kernel = grouped_dot[(4, )](a, b, c, m, n, k, mode, loop, integer, block_batching_factor=2, block_batching_axis=0,
                                enable_hivm_batch_matmul=True)
    torch.npu.synchronize()
    torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)
    assert (storage[:64] == -12345).all() and (storage[-64:] == -12345).all()
    assert torch.equal(a.cpu(), acpu) and torch.equal(b.cpu(), bcpu)
    if mode == 2 and loop and k % 32:
        # The batch consumer requires a full reduction tile; preserve safe fallback.
        assert kernel.last_launch['runtime_status'] == 'BASELINE_SELECTED'
        assert kernel.last_launch['applied_factor'] == 1
        assert kernel.last_launch['reason'] == 'CV_BATCH_REDUCTION_MASK_NOT_PROVEN'
        return
    assert_applied(kernel, 2)
    expected_lowering = ('spatial_m' if mode == 0 else 'spatial_n') + ('_uniform_control_v1' if loop else '_v1')
    if mode == 2:
        expected_lowering = 'static_integer_lanes_v1' if integer else (
            'rank3_uniform_control_batch_v1' if loop else 'rank3_masked_batch_v1')
    assert kernel.block_batching['cv_lowering'] == expected_lowering
