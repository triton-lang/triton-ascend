# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""User-visible VV semantics, original-grid launches and complete fallback."""
import pytest
import torch
import triton
import triton.language as tl

from block_batching_test_utils import assert_applied, batching_target  # noqa: F401


@triton.jit
def pointwise(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    tl.store(Y + i, x * 2 + 1, i < N)


@pytest.mark.parametrize('factor', [1, 3])
def test_factor_tail_and_default(factor):
    cpu = torch.arange(1003, dtype=torch.float32)
    x = cpu.npu()
    storage = torch.full((1035, ), -123., device='npu')
    y = storage[16:-16]
    kernel = pointwise[(8, )](x, y, 1003, 128, block_batching_factor=factor)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), cpu * 2 + 1, rtol=0, atol=0)
    assert (storage[:16] == -123).all() and (storage[-16:] == -123).all()
    if factor > 1:
        assert_applied(kernel, factor)
    else:
        default = pointwise[(8, )](x, y, 1003, 128)
        assert default is kernel  # No implicit CPU selector or candidate compilation.
        assert not hasattr(default, 'candidate')


def test_alias_fallback_and_cache_recovery():
    cpu = torch.arange(1024, dtype=torch.float32)
    x, y = cpu.npu(), torch.empty(1024, device='npu')

    def launch(out):
        return pointwise[(8, )](x, out, 1024, 128, block_batching_factor=2, block_batching_axis=0)

    kernel = launch(y)
    assert_applied(kernel, 2)
    assert launch(x) is kernel
    torch.npu.synchronize()
    assert kernel.last_launch['runtime_status'] == 'BASELINE_SELECTED'
    assert kernel.last_launch['reason'] == 'WRITE_ALIAS'
    torch.testing.assert_close(x.cpu(), cpu * 2 + 1, rtol=0, atol=0)
    assert launch(y) is kernel
    torch.npu.synchronize()
    assert_applied(kernel, 2)
    torch.testing.assert_close(y.cpu(), (cpu * 2 + 1) * 2 + 1, rtol=0, atol=0)


def test_graph_capture_uses_baseline():
    x = torch.arange(1024, device='npu', dtype=torch.float32)
    y = torch.empty_like(x)

    def launch():
        return pointwise[(8, )](x, y, 1024, 128, block_batching_factor=2, block_batching_axis=0)

    kernel = launch()
    torch.npu.synchronize()
    assert_applied(kernel, 2)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    assert kernel.last_launch['reason'] == 'CAPTURE_BASELINE'
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(y, x * 2 + 1, rtol=0, atol=0)


@triton.jit
def grid_values(Y):
    x, y, z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    gx, gy, gz = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    tl.store(Y + x + gx * (y + gy * z), gx + 100 * gy + 10000 * gz)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_original_callable_grid_and_empty_launch(axis):
    calls = 0
    grid = [2, 2, 2]
    grid[axis] = 7
    out = torch.full((28, ), -1, device='npu', dtype=torch.int32)

    def get_grid(meta):
        nonlocal calls
        calls += 1
        return tuple(grid)

    kernel = grid_values[get_grid](out, block_batching_factor=3, block_batching_axis=axis)
    torch.npu.synchronize()
    assert calls == 1
    assert_applied(kernel, 3)
    torch.testing.assert_close(out, torch.full_like(out, grid[0] + 100 * grid[1] + 10000 * grid[2]))
    grid[axis] = 0
    out.fill_(-1)
    grid_values[get_grid](out, block_batching_factor=3, block_batching_axis=axis)
    torch.npu.synchronize()
    assert calls == 2 and kernel.last_launch['runtime_status'] == 'EMPTY_GRID'
    assert (out == -1).all()


@triton.jit
def row_reduction(X, Y, N: tl.constexpr, B: tl.constexpr, SCAN: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * N + col, col < N, 0)
    if SCAN:
        y = tl.cumsum(x, 0)
    else:
        inv = tl.rsqrt(tl.sum(x * x, 0) / N + 1.e-6)
        y = x * inv
    tl.store(Y + row * N + col, y, col < N)


@pytest.mark.parametrize('scan', [False, True])
def test_reduction_and_scan_preserve_independent_rows(scan):
    torch.manual_seed(881)
    cpu = torch.randint(-4, 5, (129, 255)).float()
    x, y = cpu.npu(), torch.empty((130, 255), device='npu').fill_(-12345)
    kernel = row_reduction[(129, )](x, y, 255, 256, scan, block_batching_factor=2)
    torch.npu.synchronize()
    expected = cpu.cumsum(1) if scan else cpu * torch.rsqrt((cpu * cpu).mean(1, keepdim=True) + 1.e-6)
    torch.testing.assert_close(y[:129].cpu(), expected, rtol=2.e-5, atol=2.e-5)
    assert (y[129] == -12345).all()
    assert_applied(kernel, 2)


@triton.jit(do_not_specialize=['N'])
def iteration_groups(X, Y, A, N, B: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    grid = tl.num_programs(0)
    offsets = tl.arange(0, B)
    acc = tl.full((B, ), 0., tl.float32)
    for start in range(pid * B, N, grid * B):
        idx = start + offsets
        x = tl.load(X + idx, idx < N, 0)
        acc = acc + x
        tl.store(Y + idx, acc, idx < N)
    tl.store(A + pid * B + offsets, acc)


def test_iteration_recurrence_and_short_loop_fallback():
    for n in (4099, 513):
        x = torch.ones(n, device='npu')
        y, a = torch.empty_like(x), torch.empty(1024, device='npu')
        kernel = iteration_groups[(64, )](x, y, a, n, 16, block_batching_factor=2, block_batching_axis=0)
        torch.npu.synchronize()
        idx = torch.arange(n)
        torch.testing.assert_close(y.cpu(), (idx // 1024 + 1).float(), rtol=0, atol=0)
        expected = ((n + 1023 - torch.arange(1024)) // 1024).clamp(min=0).float()
        torch.testing.assert_close(a.cpu(), expected, rtol=0, atol=0)
        if n > 1024:
            assert_applied(kernel, 2)
            assert kernel.last_launch['grouping'] == 'loop_iterations'
            assert kernel.last_launch['original_grid'] == kernel.last_launch['batch_grid']
        else:
            assert kernel.last_launch['runtime_status'] == 'BASELINE_SELECTED'
            assert kernel.last_launch['reason'] == 'LOOP_GROUP_NOT_FORMED'
