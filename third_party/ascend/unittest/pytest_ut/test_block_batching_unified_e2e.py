# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Conservative candidate selection checked against independent CPU data."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _indirect_rows(W, IDS, Y, N: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    p = tl.program_id(0)
    c = tl.arange(0, 8)
    index = tl.load(IDS + 2 * p, mask=(p < N) & (p % 2 != 0), other=2)
    value = tl.load(W + index * S + c, mask=c < D, other=-7)
    tl.store(Y + p * D + c, value, mask=c < D)


@triton.jit
def _local_gather(X, I, Y):
    p = tl.program_id(0)
    c = tl.arange(0, 4)
    x = tl.load(X + p * 4 + c)
    index = tl.load(I + p * 4 + c)
    value = tl.gather(x, index, axis=0)
    tl.store(Y + p * 4 + c, value)


@triton.jit
def _add_one(X, Y):
    p = tl.program_id(0)
    value = tl.load(X + p)
    tl.store(Y + p, value + 1)


def rejected_candidate(kernel, reason):
    torch.npu.synchronize()
    assert kernel.candidate is None, kernel.block_batching
    assert kernel.last_launch['runtime_status'] == 'BASELINE_SELECTED', kernel.last_launch
    assert kernel.last_launch['reason'] == reason, kernel.last_launch


def applied(kernel):
    torch.npu.synchronize()
    assert kernel.candidate is not None, kernel.block_batching
    assert kernel.last_launch['runtime_status'] == 'APPLIED', kernel.last_launch


@pytest.mark.parametrize('factor', [3, 5])
@pytest.mark.parametrize('rows', [1, 7])
def test_dependent_indirect_reads_are_applied_with_same_pointer_content_changes(factor, rows):
    generator = torch.Generator().manual_seed(56)
    weights = torch.randn(6, 11, generator=generator)
    ids = torch.randint(0, 6, (rows * 2, ), generator=generator, dtype=torch.int32)
    w, index = weights.to('npu'), ids.to('npu')
    output = torch.full((rows * 7 + 16, ), 12345., device='npu')
    for change in (False, True):
        if change:
            ids.fill_(1)
            index.copy_(ids)
        kernel = _indirect_rows[(rows, )](w, index, output, rows, 7, 11, block_batching_factor=factor,
                                          block_batching_axis=0, enable_graph_optimize=False)
        applied(kernel)
        selected = ids[::2].clone()
        selected[::2] = 2
        torch.testing.assert_close(output[:rows * 7].cpu(), weights[selected.long(), :7].flatten(), rtol=0, atol=0)
        assert bool((output[rows * 7:].cpu() == 12345).all())


@triton.jit
def _selective_log_softmax(LOGITS, IDS, OUT, IDS_B, IDS_L, L: tl.constexpr, N: tl.constexpr, TEMPERATURE: tl.constexpr,
                           BLOCK_N: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    l = tl.program_id(1).to(tl.int64)
    logits_ptr = LOGITS + b * (L + 1) * N + l * N
    m = float('-inf')
    total = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        values = tl.load(logits_ptr + cols, cols < N, other=float('-inf')).to(tl.float32) / TEMPERATURE
        next_m = tl.maximum(m, tl.max(values))
        total = total * tl.exp(m - next_m) + tl.sum(tl.exp(values - next_m))
        m = next_m
    lse = m + tl.log(total)
    index = tl.load(IDS + b * IDS_B + l * IDS_L)
    selected = tl.load(logits_ptr + index).to(tl.float32) / TEMPERATURE
    tl.store(OUT + b * L + l, selected - lse)


@pytest.mark.parametrize('factor', [2, 3, 5])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('length', [1, 7])
def test_selective_log_softmax_applies_without_explicit_synchronization(factor, axis, length):
    # LK31's MASK=None / LSE=None path, including tiled statistics, row-local
    # device indices, non-power-of-two factors, both axes and a partial tile.
    batch, vocab = (5 if axis == 0 else 1), 67
    generator = torch.Generator().manual_seed(31)
    logits = torch.randn(batch, length + 1, vocab, generator=generator).bfloat16()
    ids = torch.randint(0, vocab, (batch, length * 2), generator=generator, dtype=torch.int64)
    device_logits, device_ids = logits.to('npu'), ids.to('npu')[:, ::2]
    storage = torch.full((batch * length + 16, ), -123., device='npu')
    kernel = _selective_log_softmax[(batch, length)](device_logits, device_ids, storage, *device_ids.stride(), length,
                                                     vocab, 0.9, 16, block_batching_factor=factor,
                                                     block_batching_axis=axis, enable_graph_optimize=False)
    applied(kernel)
    assert kernel.block_batching['guard']['terminal_read_ids']
    expected = torch.log_softmax(logits[:, :-1].float() / 0.9, dim=-1)
    expected = expected.gather(-1, ids[:, ::2].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(storage[:batch * length].cpu().view(batch, length), expected, rtol=2e-4, atol=2e-5)
    assert bool((storage[batch * length:].cpu() == -123).all())


@pytest.mark.parametrize('factor', [2, 3, 5])
def test_gather_uses_unmodified_baseline(factor):
    generator = torch.Generator().manual_seed(56)
    x = torch.randn(7, 4, generator=generator)
    indices = torch.randint(0, 4, (7, 4), generator=generator, dtype=torch.int32)
    device_x, device_indices = x.to('npu'), indices.to('npu')
    actual = torch.empty_like(device_x)
    kernel = _local_gather[(7, )](device_x, device_indices, actual, block_batching_factor=factor, block_batching_axis=0,
                                  enable_graph_optimize=False)
    rejected_candidate(kernel, 'UNSUPPORTED_BATCH_OP')
    torch.testing.assert_close(actual.cpu(), torch.gather(x, 1, indices.long()), rtol=0, atol=0)


def test_known_address_alias_selects_complete_baseline():
    x = torch.arange(7, dtype=torch.float32, device='npu')
    y = torch.empty_like(x)
    options = dict(block_batching_factor=3, block_batching_axis=0, enable_graph_optimize=False)
    kernel = _add_one[(7, )](x, y, **options)
    torch.npu.synchronize()
    assert kernel.last_launch['runtime_status'] == 'APPLIED', kernel.last_launch
    torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
    kernel = _add_one[(7, )](x, x, **options)
    torch.npu.synchronize()
    assert kernel.last_launch['runtime_status'] == 'BASELINE_SELECTED', kernel.last_launch
    assert kernel.last_launch['reason'] == 'WRITE_ALIAS'
    torch.testing.assert_close(x.cpu(), torch.arange(7, dtype=torch.float32) + 1, rtol=0, atol=0)


@triton.jit
def _index_chain(I, Y):
    p = tl.program_id(0)
    index = p
    for _ in range(3):
        index = tl.load(I + index)
    tl.store(Y + p, index)


@pytest.mark.parametrize('factor', [2, 3, 5])
def test_loop_carried_indices_with_output_use_baseline(factor):
    indices = torch.randperm(7, generator=torch.Generator().manual_seed(137)).int()
    expected = torch.arange(7, dtype=torch.int32)
    for _ in range(3):
        expected = indices[expected.long()]
    device_indices = indices.to('npu')
    actual = torch.empty(7, dtype=torch.int32, device='npu')
    kernel = _index_chain[(7, )](device_indices, actual, block_batching_factor=factor, block_batching_axis=0,
                                 enable_graph_optimize=False)
    rejected_candidate(kernel, 'UNPROVEN_MEMORY_NONINTERFERENCE')
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _mixed_pointwise(X, Y, M, N, IDS, Z):
    m = tl.program_id(0) * 4 + tl.arange(0, 4)
    n = tl.program_id(1) * 8 + tl.arange(0, 8)
    p = tl.program_id(1)
    index = tl.load(IDS + p)
    extra = tl.load(X + index)
    offset = m[:, None] * 32 + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    value = tl.load(X + offset, mask, other=7.0)
    tl.store(Y + offset, value + 1.0, mask)
    tl.store(Z + p, extra)


def test_mixed_indirect_reads_and_outputs_use_baseline():
    x_cpu = torch.arange(128, dtype=torch.float32) / 16
    ids_cpu = torch.tensor([0, 17, 52, 91, 127], dtype=torch.int32)
    x, ids = x_cpu.to('npu'), ids_cpu.to('npu')
    y = torch.full((144, ), -123., device='npu')
    z = torch.full((13, ), -123., device='npu')
    kernel = _mixed_pointwise[(1, 5)](x, y, 3, 19, ids, z, block_batching_factor=3, block_batching_axis=1,
                                      enable_graph_optimize=False)
    rejected_candidate(kernel, 'UNPROVEN_MEMORY_NONINTERFERENCE')
    expected = torch.full((144, ), -123.)
    expected[:128].view(4, 32)[:3, :19] = x_cpu.view(4, 32)[:3, :19] + 1
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(z[:5].cpu(), x_cpu[ids_cpu.long()], rtol=0, atol=0)
    assert bool((z[5:].cpu() == -123).all())
