# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Exact INT8 CV execution, output tails, canaries and immutable inputs."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def integer_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr, BM: tl.constexpr = 32,
                BN: tl.constexpr = 32):
    p = tl.program_id(0)
    if MODE == 0:
        rr = p * BM + tl.arange(0, BM)
        cc = tl.arange(0, BN)
    elif MODE == 1:
        rr = tl.arange(0, BM)
        cc = p * BN + tl.arange(0, BN)
    else:
        rr = tl.arange(0, BM)
        cc = tl.arange(0, BN)
        A += p * M * K
        B += p * K * N
        C += p * M * N
    kk = tl.arange(0, K)
    a = tl.load(A + rr[:, None] * K + kk[None, :], rr[:, None] < M, 0)
    b = tl.load(B + kk[:, None] * N + cc[None, :], cc[None, :] < N, 0)
    c = tl.dot(a, b, out_dtype=tl.int32)
    tl.store(C + rr[:, None] * N + cc[None, :], c, (rr[:, None] < M) & (cc[None, :] < N))


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('tail', [False, True])
@pytest.mark.parametrize('factor', [2, 4])
def test_integer_cv_device(mode, tail, factor):
    m, n, k = (128, 32, 64) if mode == 0 else ((32, 128, 64) if mode == 1 else (32, 32, 64))
    if tail:
        if mode == 0:
            m -= 1
        else:
            n -= 1
    ashape = (4, m, k) if mode == 2 else (m, k)
    bshape = (4, k, n) if mode == 2 else (k, n)
    cshape = (4, m, n) if mode == 2 else (m, n)
    torch.manual_seed(20261105 + mode)
    acpu = torch.randint(-128, 128, ashape, dtype=torch.int8)
    bcpu = torch.randint(-128, 128, bshape, dtype=torch.int8)
    expected = (acpu.double() @ bcpu.double()).to(torch.int32)
    a, b = acpu.npu(), bcpu.npu()
    count = expected.numel()
    storage = torch.full((count + 128, ), 0x13572468, device='npu', dtype=torch.int32)
    c = storage[64:-64].view(cshape)
    compiled = integer_dot[(4, )](a, b, c, m, n, k, mode, block_batching_cv=True, block_batching_factor=factor,
                                  block_batching_axis=0, enable_hivm_batch_matmul=True)
    torch.npu.synchronize()
    assert compiled.last_launch['runtime_status'] == 'APPLIED', compiled.last_launch
    assert compiled.block_batching['cv_lowering'] == ['spatial_m_v1', 'spatial_n_v1', 'static_integer_lanes_v1'][mode]
    torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)
    assert bool((storage[:64] == 0x13572468).all()) and bool((storage[-64:] == 0x13572468).all())
    assert torch.equal(a.cpu(), acpu) and torch.equal(b.cpu(), bcpu)


@triton.jit
def packed_integer_loop(A, B, C, M: tl.constexpr = 128, N: tl.constexpr = 32, K: tl.constexpr = 256,
                        BM: tl.constexpr = 32, BK: tl.constexpr = 32):
    """Liger-style packed int2 decode, with directly provable MN coordinates."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, N)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, N), 0, tl.int32)
    for plane in range(4):
        for chunk in range(K // 4 // BK):
            a = tl.load(A + rows[:, None] * K + (plane * (K // 4) + chunk * BK + kk[None, :]))
            packed = tl.load(B + (chunk * BK + kk[:, None]) * N + cols[None, :])
            bits = ((packed.to(tl.int32) >> (2 * plane)) & 3).to(tl.int8) - 1
            acc = tl.dot(a, bits, acc, out_dtype=tl.int32)
    tl.store(C + rows[:, None] * N + cols[None, :], acc)


@pytest.mark.parametrize('factor', [2, 4])
def test_packed_integer_nested_loop_device(factor):
    # This isolates unpacking and integer accumulation, not the original Liger
    # grouped PID and nested pointer recurrences (which remain unsupported).
    torch.manual_seed(20261106)
    acpu = torch.randint(-128, 128, (128, 256), dtype=torch.int8)
    packed = torch.randint(0, 256, (64, 32), dtype=torch.uint8)
    bcpu = torch.cat([((packed.int() >> (2 * plane)) & 3) - 1 for plane in range(4)])
    expected = (acpu.double() @ bcpu.double()).to(torch.int32)
    a, b = acpu.npu(), packed.npu()
    storage = torch.full((128 * 32 + 128, ), 0x13572468, device='npu', dtype=torch.int32)
    c = storage[64:-64].view(128, 32)
    compiled = packed_integer_loop[(4, )](a, b, c, block_batching_cv=True, block_batching_factor=factor,
                                          block_batching_axis=0, enable_hivm_batch_matmul=True)
    torch.npu.synchronize()
    assert compiled.last_launch['runtime_status'] == 'APPLIED', compiled.last_launch
    assert compiled.block_batching['cv_lowering'] == 'spatial_m_uniform_control_v1'
    torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)
    assert bool((storage[:64] == 0x13572468).all()) and bool((storage[-64:] == 0x13572468).all())
    assert torch.equal(a.cpu(), acpu) and torch.equal(b.cpu(), packed)
