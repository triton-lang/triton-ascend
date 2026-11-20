# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Numerical VV/CV, memory-view and fallback checks on real NPU launches."""

import pytest
import torch
import triton
import triton.language as tl
import torch_npu

from block_batching_test_utils import assert_applied, require_batching_target
from triton.backends.ascend.block_batching import uses_grouping_policy


@triton.jit
def pointwise(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    tl.store(Y + i, x * 2 + 1, i < N)


@triton.jit
def grid_values(Y):
    x, y, z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    gx, gy, gz = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    tl.store(Y + x + gx * (y + gy * z), gx + 100 * gy + 10000 * gz)


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


@triton.jit(do_not_specialize=['N', 'SX', 'SO'])
def _independent_rows(X, Y, O, N, SX, SO, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
    y = tl.load(Y + row * SX + col, col < N, other=0).to(tl.float32)
    tl.store(O + row * SO + col, x * tl.sigmoid(x) + y, col < N)


@triton.jit(do_not_specialize=['T'])
def tiled_reduce(X, Y, T, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    src = tl.make_block_ptr(X + head * D, (T, D), (H * D, 1), (block * BT, 0), (BT, D), (1, 0))
    x = tl.load(src, boundary_check=(0, 1), padding_option='zero')
    total = tl.sum(tl.sum(x * x, 1), 0)
    tl.store(Y + block * H + head, total)


@triton.jit(do_not_specialize=['T'])
def tiled_scan(X, Y, T, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
               SCAN_AXIS: tl.constexpr = 0):
    column, block, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    src = tl.make_block_ptr(X + head * D, (T, D), (H * D, 1), (block * BT, column * BS), (BT, BS), (1, 0))
    dst = tl.make_block_ptr(Y + head * D, (T, D), (H * D, 1), (block * BT, column * BS), (BT, BS), (1, 0))
    x = tl.load(src, boundary_check=(0, 1), padding_option='zero')
    value = tl.cumsum(x, 0, reverse=True)
    tl.store(dst, value, boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def batched_tiled_scan(X, Y, T, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr):
    column, block, batch_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    base = (batch_head // H) * T * H * D + (batch_head % H) * D
    src = tl.make_block_ptr(X + base, (T, D), (H * D, 1), (block * BT, column * BS), (BT, BS), (1, 0))
    dst = tl.make_block_ptr(Y + base, (T, D), (H * D, 1), (block * BT, column * BS), (BT, BS), (1, 0))
    x = tl.load(src, boundary_check=(0, 1), padding_option='zero')
    tl.store(dst, tl.cumsum(x, 0, reverse=True), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def batched_tiled_map(X, Y, T, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr):
    column, block, batch_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    base = (batch_head // H) * T * H * D + (batch_head % H) * D
    src = tl.make_block_ptr(X + base, (T, D), (H * D, 1), (block * BT, column * BS), (BT, BS), (1, 0))
    dst = tl.make_block_ptr(Y + base, (T, D), (H * D, 1), (block * BT, column * BS), (BT, BS), (1, 0))
    x = tl.load(src, boundary_check=(0, 1), padding_option='zero')
    tl.store(dst, x + x, boundary_check=(0, 1))


@triton.jit
def row_softmax_with_weight(X, W, Y, N: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    value = tl.load(X + row * STRIDE + col, col < N, other=-float('inf'))
    weight = tl.load(W + col, col < N, other=0)
    exponent = tl.exp(value - tl.max(value, 0))
    normalized = exponent / tl.sum(exponent, 0)
    tl.store(Y + row * STRIDE + col, normalized * weight, col < N)


@triton.jit(do_not_specialize=['T', 'ITERS'])
def row_reduction_recurrence(X, Y, T, ITERS, D: tl.constexpr, BT: tl.constexpr, COLLAPSE: tl.constexpr):
    block = tl.program_id(0)
    rows = block * BT + tl.arange(0, BT)
    columns = tl.arange(0, D)
    x = tl.load(X + rows[:, None] * D + columns[None, :], rows[:, None] < T, other=0)
    value = tl.sum(x, 1)
    for _ in range(ITERS):
        value = tl.sum(x + value[:, None], 1)
    if COLLAPSE:
        total = tl.sum(value, 0)
        tl.store(Y + block, total)
    else:
        tl.store(Y + rows, value, rows < T)


@triton.jit(do_not_specialize=['N', 'SX', 'SY', 'SO'])
def _row_relative_vector(X, Y, O, P, N, SX, SY, SO, SCALE, SINGLETON: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    if SINGLETON:
        col = col[None, :]
    mask = col < N
    x = tl.load(X + row * SX + col, mask, other=0).to(tl.float32)
    y = tl.load(Y + row * SY + col, mask, other=0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    tl.store(O + row * SO + col, x * sig * y * SCALE, mask)
    tl.store(P + row * SO + col, sig * (1.0 + x * (1.0 - sig)) * y * SCALE, mask)


class TestVV:

    @pytest.fixture(autouse=True)
    def batching_target(self, monkeypatch):
        require_batching_target(monkeypatch, include_legacy=True)

    @pytest.mark.parametrize('factor', [1, 3])
    def test_factor_tail_and_default(self, factor):
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
            y.fill_(-123)
            default = pointwise[(8, )](x, y, 1003, 128)
            torch.npu.synchronize()
            explicit_auto = pointwise[(8, )](x, y, 1003, 128, block_batching_factor="auto")
            torch.npu.synchronize()
            assert default is explicit_auto
            assert default.block_auto_selection['selected_factor'] in (1, 2, 4)
            assert not hasattr(kernel, 'block_auto_selection')
            assert not hasattr(kernel, 'candidate')  # Explicit F1 disables selection.
            torch.testing.assert_close(y.cpu(), cpu * 2 + 1, rtol=0, atol=0)
            assert (storage[:16] == -123).all() and (storage[-16:] == -123).all()

    def test_alias_fallback_and_cache_recovery(self):
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

    def test_default_auto_large_pointwise(self):
        x = torch.arange(131072, device='npu', dtype=torch.float32)
        out = torch.full_like(x, -123.)
        warmed = pointwise.warmup(x, out, x.numel(), 512, grid=(256, ))
        torch.npu.synchronize()
        assert (out == -123).all()  # warmup may compile candidates, never execute them.
        assert warmed.block_auto_selection['candidate_plan']['candidates']
        kernel = pointwise[(256, )](x, out, x.numel(), 512)
        assert kernel is warmed
        torch.npu.synchronize()
        torch.testing.assert_close(out, x * 2 + 1, rtol=0, atol=0)
        if uses_grouping_policy(str(kernel.metadata.target.arch)):
            assert kernel.block_auto_selection['model'] == 'ttir-legacy-grouping-v1'
            assert_applied(kernel, 2)
            assert kernel.block_auto_selection['reason'] == 'LEGACY_TTIR_GROUPING'

    @pytest.mark.parametrize('explicit', [None, 'auto', 2])
    @pytest.mark.parametrize('generic', [False, True])
    def test_autotune_batching_default_and_override(self, explicit, generic):
        from triton.runtime.autotuner import autotune as generic_autotune

        seen = []

        def bench(fn, quantiles):
            fn()
            return 1.0  # Functional test; no device performance assertion.

        config_options = dict(block_batching_factor=2) if explicit == 2 else {}
        decorate = generic_autotune if generic else triton.autotune
        tuned = decorate(configs=[triton.Config(dict(B=b, **config_options)) for b in (256, 512)], key=['N'],
                         do_bench=bench)(pointwise)
        x = torch.arange(131072, device='npu', dtype=torch.float32)
        out = torch.full_like(x, -123.)
        options = dict(block_batching_factor='auto') if explicit == 'auto' else {}
        grid = lambda meta: (triton.cdiv(meta['N'], meta['B']), )
        from unittest.mock import patch
        from triton.backends.ascend import block_batching
        original = block_batching._prepare_launch

        def observe(*args, **kwargs):
            prepared = original(*args, **kwargs)
            # An omitted factor is the ordinary compilation, which the option
            # default already spells as 1, so the key stays out of the options.
            seen.append(prepared[0].get('block_batching_factor', 1))
            return prepared

        with patch.object(block_batching, '_prepare_launch', observe):
            warmed = tuned.warmup(x, out, x.numel(), grid=grid, **options)
            for item in warmed:
                if hasattr(item, 'result'):
                    item.result()
            torch.npu.synchronize()
            assert (out == -123).all()
            kernel = tuned[grid](x, out, x.numel(), **options)
            tuned[grid](x, out, x.numel(), **options)  # Best-config cache hit.
            tuned.compile_parallel = False
            tuned.cache.clear()
            tuned[grid](x, out, x.numel(), **options)  # Sequential candidate path.
        torch.npu.synchronize()
        assert seen and set(seen) == {1 if explicit is None else explicit}
        torch.testing.assert_close(out, x * 2 + 1, rtol=0, atol=0)
        if explicit is None:
            assert not hasattr(kernel, 'block_auto_selection') and not hasattr(kernel, 'candidate')
        elif explicit == 'auto':
            assert hasattr(kernel, 'block_auto_selection')
        else:
            assert_applied(kernel, explicit)

    @pytest.mark.parametrize('block_pointer', [False, True])
    @pytest.mark.parametrize('factor', [2, 4, 'auto'])
    def test_dynamic_permute_tail_falls_back(self, block_pointer, factor):
        from test_implicit_permute import (addptr_implicit_perm_load_store_3d,
                                           make_tensor_ptr_implicit_perm_load_store_3d)
        fn = make_tensor_ptr_implicit_perm_load_store_3d if block_pointer else addptr_implicit_perm_load_store_3d
        # The original block-pointer load is unmasked; pad its final physical tile.
        size = 100 * 40 * 32
        x = torch.arange(size + 32, dtype=torch.float32).npu()[:size].reshape(100, 40, 32)
        storage = torch.full((size + 32, ), -123., device='npu')
        out = storage[:size].reshape_as(x)
        kernel = fn[(4, 10, 25)](x, out, 100, 40, 32, 1280, 32, 1, 4, 4, 10, block_batching_factor=factor)
        torch.npu.synchronize()
        torch.testing.assert_close(out, x, rtol=0, atol=0)
        assert (storage[size:] == -123).all()
        if factor == 'auto':
            assert kernel.block_auto_selection['selected_factor'] == 1
        else:
            assert kernel.candidate is None
            assert kernel.block_batching['reason'] == 'VV_SYMBOLIC_LAYOUT_NOT_PROVEN'

    @pytest.mark.parametrize('mode', [0, 1])
    @pytest.mark.parametrize('blocks', [4, 128])
    def test_portable_spatial_merge(self, mode, blocks):
        m, n, k = (blocks * 32, 32, 32) if mode == 0 else (32, blocks * 32, 32)
        torch.manual_seed(881)
        acpu, bcpu = torch.randint(-4, 5, (m, k)).float(), torch.randint(-4, 5, (k, n)).float()
        a, b = acpu.npu(), bcpu.npu()
        out = torch.empty((m, n), device='npu')
        expected = acpu @ bcpu
        merged = grouped_dot[(blocks, )](a, b, out, m, n, k, mode, False, False, block_batching_factor=2,
                                         block_batching_axis=0)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        assert_applied(merged, 2)
        automatic = grouped_dot[(blocks, )](a, b, out, m, n, k, mode, False, False)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        selection = automatic.block_auto_selection
        assert selection['selected_factor'] in (1, 2, 4)
        assert selection['candidate_plan']['candidates']
        if blocks == 128 and str(automatic.metadata.target.arch).startswith('Ascend910_93'):
            assert_applied(automatic, 2)

    def test_graph_capture_uses_baseline(self):
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

    @pytest.mark.parametrize('axis', [0, 1, 2])
    def test_original_callable_grid_and_empty_launch(self, axis):
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

    @pytest.mark.parametrize('scan', [False, True])
    def test_reduction_and_scan_preserve_independent_rows(self, scan):
        torch.manual_seed(881)
        cpu = torch.randint(-4, 5, (129, 255)).float()
        x, y = cpu.npu(), torch.empty((130, 255), device='npu').fill_(-12345)
        kernel = row_reduction[(129, )](x, y, 255, 256, scan, block_batching_factor=2)
        torch.npu.synchronize()
        expected = cpu.cumsum(1) if scan else cpu * torch.rsqrt((cpu * cpu).mean(1, keepdim=True) + 1.e-6)
        torch.testing.assert_close(y[:129].cpu(), expected, rtol=2.e-5, atol=2.e-5)
        assert (y[129] == -12345).all()
        assert_applied(kernel, 2)

    def test_iteration_recurrence_and_short_loop_fallback(self):
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


class TestCV:

    @pytest.fixture(autouse=True)
    def batching_target(self, monkeypatch):
        require_batching_target(monkeypatch)

    def test_auto_batch_handle_keeps_capture_and_alias_fallbacks(self):
        cpu_a = torch.arange(4 * 32 * 32).reshape(4, 32, 32).remainder(7).float()
        cpu_b = cpu_a.flip([2]).contiguous()
        expected = cpu_a @ cpu_b
        a, b = cpu_a.npu(), cpu_b.npu()
        out = torch.empty_like(a)
        constants = (32, 32, 32, 2, False, False, 32)
        bundle = grouped_dot[(4, )](a, b, out, *constants, enable_hivm_batch_matmul=True)
        torch.npu.synchronize()
        assert bundle.block_auto_selection['selected_factor'] == 2
        assert_applied(bundle, 2)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

        # Reuse the same warmed handle: capture cannot compile or load F1.
        launch = bundle[(4, 1, 1)]
        out.zero_()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch(a, b, out, *constants)
        assert bundle.last_launch['reason'] == 'CAPTURE_BASELINE'
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

        launch(a, b, a, *constants)
        torch.npu.synchronize()
        assert bundle.last_launch['reason'] == 'WRITE_ALIAS'
        torch.testing.assert_close(a.cpu(), expected, rtol=0, atol=0)

    @pytest.mark.parametrize('mode,loop,integer,k', [(0, False, False, 32), (1, False, False, 32),
                                                     (2, False, False, 32), (0, True, False, 63), (1, True, False, 63),
                                                     (2, True, False, 64), (2, True, False, 63), (2, False, False, 31),
                                                     (0, False, True, 32), (2, False, True, 32)])
    def test_grouped_matrix_results(self, mode, loop, integer, k):
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
        if mode == 2 and loop:
            grouped_dot[(4, )](a, b, c, m, n, k, mode, loop, integer, block_batching_factor=1)
            torch.npu.synchronize()
            torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)
            c.fill_(-12345)
        kernel = grouped_dot[(4, )](a, b, c, m, n, k, mode, loop, integer, block_batching_factor=2,
                                    block_batching_axis=0, enable_hivm_batch_matmul=True)
        torch.npu.synchronize()
        torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)
        assert (storage[:64] == -12345).all() and (storage[-64:] == -12345).all()
        assert torch.equal(a.cpu(), acpu) and torch.equal(b.cpu(), bcpu)
        if mode == 2 and not loop and k % 32:
            # The batch consumer requires a full reduction tile; preserve safe fallback.
            assert kernel.last_launch['runtime_status'] == 'BASELINE_SELECTED'
            assert kernel.last_launch['applied_factor'] == 1
            assert kernel.last_launch['reason'] == 'CV_BATCH_REDUCTION_MASK_NOT_PROVEN'
            return
        assert_applied(kernel, 2)
        expected_lowering = ('spatial_m' if mode == 0 else 'spatial_n') + ('_uniform_control_v1' if loop else '_v1')
        if mode == 2:
            expected_lowering = 'static_integer_lanes_v1' if integer else (
                'static_uniform_control_lanes_v1' if loop else 'rank3_masked_batch_v1')
            if loop:
                assert kernel.block_batching['guard'][
                    'cv_loop_fusion_rejection'] == 'CV_BATCH_CONTROL_ACCUMULATOR_NOT_VALIDATED'
        assert kernel.block_batching['cv_lowering'] == expected_lowering
        if mode == 0 and not loop and not integer:
            c.fill_(-12345)
            default = grouped_dot[(4, )](a, b, c, m, n, k, mode, loop, integer)
            torch.npu.synchronize()
            torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)
            assert (storage[:64] == -12345).all() and (storage[-64:] == -12345).all()
            selection = default.block_auto_selection
            assert selection['selected_factor'] in (1, 2, 4)
            if selection['selected_factor'] > 1:
                assert_applied(default, selection['selected_factor'])


class TestMemoryViews:

    @pytest.fixture(autouse=True)
    def batching_target(self, monkeypatch):
        require_batching_target(monkeypatch)

    @pytest.mark.parametrize('rows,columns,padding', [(128, 512, 0), (129, 511, 7)])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    def test_independent_rows_preserve_packed_views_and_both_tails(self, rows, columns, padding, dtype):
        generator = torch.Generator().manual_seed(881)
        cpu = torch.randn((rows, 2 * columns + padding), generator=generator).to(dtype)
        packed = cpu.npu()
        x, y = packed[:, :columns], packed[:, columns:2 * columns]
        stride = columns + 11
        storage = [torch.full((rows * stride + 32, ), -123., dtype=dtype, device='npu') for _ in range(2)]
        outputs = [allocation[16:-16].view(rows, stride) for allocation in storage]
        grid = (rows, triton.cdiv(columns, 128))
        kernels = [
            _independent_rows[grid](x, y, out, columns, x.stride(0), stride, 128, block_batching_factor=factor,
                                    num_warps=4) for out, factor in zip(outputs, (1, 8))
        ]
        torch.npu.synchronize()
        expected = (cpu[:, :columns].float() * torch.sigmoid(cpu[:, :columns].float()) +
                    cpu[:, columns:2 * columns].float()).to(dtype)
        tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-4
        for out, allocation in zip(outputs, storage):
            torch.testing.assert_close(out[:, :columns].cpu(), expected, rtol=tolerance, atol=tolerance)
            assert (out[:, columns:].cpu() == -123.).all()
            assert (allocation[:16].cpu() == -123.).all()
            assert (allocation[-16:].cpu() == -123.).all()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(packed.cpu(), cpu, rtol=0, atol=0)
        bundle = kernels[1]
        assert bundle.last_launch['applied_factor'] == 8, bundle.last_launch
        assert bundle.last_launch['selected_axis'] == 0
        assert kernels[0].kernel == bundle.baseline.kernel
        assert all(access['reason'] == 'ALREADY_STRUCTURED' and access['action'] == 'Keep' and not access['rewritten']
                   for access in bundle.block_batching['layout']['accesses'])
        if triton.runtime.driver.active.get_current_target().arch.startswith('Ascend950'):
            assert bundle.block_batching['memory_lowering']['structured_row_copy'] is True
        assert 'triton_indirect' not in bundle.candidate.asm['ttadapter']

    @pytest.mark.parametrize('axis,scans', [(-1, True), (0, True), (1, True), (1, False)])
    def test_shared_divided_origin_with_multiple_batches_and_dynamic_pitch(self, axis, scans):
        kernel = batched_tiled_scan if scans else batched_tiled_map
        batch, heads, columns, block = 2, 4, 15, 4
        source = torch.arange(batch * 260 * heads * columns + 32, dtype=torch.float32) % 7 - 3
        x = source.npu()
        outputs = [torch.full_like(x, -12345) for _ in range(2)]
        previous = None
        for length in (259, 260, 259):
            for output in outputs:
                output.fill_(-12345)
            grid = (2, triton.cdiv(length, block), batch * heads)
            args = (length, heads, columns, block, 8)
            baseline = kernel[grid](x, outputs[0], *args, block_batching_factor=1, block_batching_axis=axis)
            batched = kernel[grid](x, outputs[1], *args, block_batching_factor=2, block_batching_axis=axis)
            torch.npu.synchronize()
            if scans:
                assert batched.last_launch['applied_factor'] == 1, batched.block_batching
                assert batched.candidate is None
                assert batched.block_batching['reason'] == 'VV_SCAN_DOMAIN_NOT_SUPPORTED'
            else:
                assert batched.last_launch['applied_factor'] == 2, batched.block_batching
                assert batched.last_launch['selected_axis'] == 1
                observed = batched.block_batching['layout']['lowering_observation']
                assert observed['indirect_load_ops'] == observed['indirect_store_ops'] == 0
            assert baseline.kernel == batched.baseline.kernel
            if previous is not None:
                assert batched is previous
            previous = batched
            count = batch * length * heads * columns
            expected = torch.full_like(source, -12345)
            reference = source[:count].reshape(batch, length, heads, columns)
            destination = expected[:count].reshape(batch, length, heads, columns)
            for start in range(0, length, block):
                value = reference[:, start:start + block]
                destination[:, start:start + block] = value.flip(1).cumsum(1).flip(1) if scans else value + value
            for output in outputs:
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    @pytest.mark.parametrize('kind,axis,applies', [('reduce', 0, False), ('scan', 1, False), ('scan', 0, False)])
    @pytest.mark.parametrize('factor', [2, 3])
    def test_tiled_aggregation_uses_proven_memory_views_or_reuses_fallback(self, kind, axis, applies, factor):
        # Keep the semantic regression small; real KDA tile sizes are benchmarked
        # separately. Both cases still exercise a two-dimensional block pointer.
        h, d, bt = (4, 64, 16) if kind == 'reduce' else (4, 16, 4)
        maximum = 1025
        # Exact small integers keep the reference independent of FP reassociation.
        source = (torch.arange((maximum + bt) * h * d, dtype=torch.float32) % 7 - 3).reshape(-1, h, d)
        x = source.npu()
        outputs = [torch.full_like(x, -12345) for _ in range(2)]
        previous = None
        for t in (1024, 1025, 1024):
            nt = triton.cdiv(t, bt)
            for output in outputs:
                output.fill_(-12345)
            kwargs = dict(num_warps=4, num_stages=1, block_batching_axis=axis)
            if kind == 'reduce':
                grid = (nt, h)
                args = (t, h, d, bt)
                kernel = tiled_reduce
            else:
                grid = (2, nt, h)
                args = (t, h, d, bt, 8)
                kernel = tiled_scan
            baseline = kernel[grid](x, outputs[0], *args, block_batching_factor=1, **kwargs)
            batched = kernel[grid](x, outputs[1], *args, block_batching_factor=factor, **kwargs)
            torch.npu.synchronize()
            assert batched.baseline.kernel == baseline.kernel
            if applies:
                assert batched.candidate is not None, batched.block_batching
                assert batched.last_launch['applied_factor'] == factor, batched.block_batching
                analysis = batched.block_batching
                assert all(access['rewritten'] for access in analysis['layout']['accesses'] if access['original_shape'])
            else:
                assert batched.candidate is None, batched.block_batching
                expected_reason = 'VV_MULTI_AXIS_REDUCTION_NOT_SUPPORTED' if kind == 'reduce' else 'VV_SCAN_DOMAIN_NOT_SUPPORTED'
                assert batched.block_batching['reason'] == expected_reason, batched.block_batching
                assert batched.last_launch['applied_factor'] == 1
                analysis = batched.block_batching
            if applies:
                observation = analysis['layout']['lowering_observation']
                assert observation['indirect_load_ops'] == observation['indirect_store_ops'] == 0
            if applies:
                assert analysis['operations']['tt.reduce'] > 0
            else:
                assert batched.block_batching['compile_status'] == 'REJECTED'
            if previous is not None:
                assert batched is previous
            previous = batched
            expected = torch.full_like(source, -12345)
            for block in range(nt):
                begin, end = block * bt, min(t, (block + 1) * bt)
                value = source[begin:end]
                if kind == 'reduce':
                    expected.view(-1)[block * h:(block + 1) * h] = (value * value).sum((0, 2))
                else:
                    expected[begin:end] = value.flip(0).cumsum(0).flip(0)
            for output in outputs:
                # Includes the tail and all unwritten storage, not only valid values.
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    def test_dependent_row_reductions_and_shared_weight_still_apply(self):
        from triton.runtime import driver
        rows = 2 * driver.active.utils.get_aivector_core_num() + 1
        columns, stride = 63, 71
        source = torch.randn((rows + 1, stride), generator=torch.Generator().manual_seed(117)) * .1
        weight = torch.linspace(.5, 1.5, columns)
        x, w = source.npu(), weight.npu()
        outputs = [torch.full_like(x, -12345) for _ in range(2)]
        baseline = row_softmax_with_weight[(rows, )](x, w, outputs[0], columns, stride, 64, block_batching_factor=1)
        batched = row_softmax_with_weight[(rows, )](x, w, outputs[1], columns, stride, 64, block_batching_factor=2)
        torch.npu.synchronize()
        assert batched.last_launch['applied_factor'] == 2, batched.block_batching
        assert batched.baseline.kernel == baseline.kernel
        assert batched.block_batching['operations']['tt.reduce'] == 2
        observed = batched.block_batching['layout']['lowering_observation']
        assert observed['indirect_load_ops'] == observed['indirect_store_ops'] == 0
        expected = torch.full_like(source, -12345)
        expected[:rows, :columns] = source[:rows, :columns].softmax(-1) * weight
        for output in outputs:
            torch.testing.assert_close(output.cpu(), expected, rtol=3e-4, atol=3e-5)

    @pytest.mark.parametrize('collapse', [False, True])
    @pytest.mark.parametrize('factor', [2, 3])
    def test_row_reduction_recurrence_retains_or_rejects_its_domain(self, collapse, factor):
        from triton.runtime import driver
        width, tile = 32, 8
        rows = (3 * driver.active.utils.get_aivector_core_num() + 2) * tile + 1
        source = (torch.arange(rows * width, dtype=torch.float32) % 7 - 3).reshape(rows, width)
        x = source.npu()
        outputs = [torch.full((rows + 32, ), -12345., device=x.device) for _ in range(2)]
        previous = None
        for iterations in (0, 1, 2, 0):
            for output in outputs:
                output.fill_(-12345)
            grid = (triton.cdiv(rows, tile), )
            args = (rows, iterations, width, tile, collapse)
            baseline = row_reduction_recurrence[grid](x, outputs[0], *args, block_batching_factor=1,
                                                      block_batching_axis=0)
            batched = row_reduction_recurrence[grid](x, outputs[1], *args, block_batching_factor=factor,
                                                     block_batching_axis=0)
            torch.npu.synchronize()
            if collapse:
                assert batched.candidate is None, batched.block_batching
                assert batched.block_batching['reason'] == 'VV_MULTI_AXIS_REDUCTION_NOT_SUPPORTED'
                assert batched.last_launch['applied_factor'] == 1
                assert batched.baseline.kernel == baseline.kernel
            else:
                assert batched.last_launch['applied_factor'] == factor, batched.block_batching
                assert batched.block_batching['admission']['source'] == 'proven'
                assert batched.block_batching['admission']['memory_lowering'] == 'proven'
            if previous is not None:
                assert batched is previous
            previous = batched
            reference = source.sum(1)
            for _ in range(iterations):
                reference = (source + reference[:, None]).sum(1)
            expected = torch.full((rows + 32, ), -12345.)
            if collapse:
                padded = torch.nn.functional.pad(reference, (0, triton.cdiv(rows, tile) * tile - rows))
                expected[:grid[0]] = padded.reshape(-1, tile).sum(1)
            else:
                expected[:rows] = reference
            for output in outputs:
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    @pytest.mark.parametrize('factor', [2, 3, 8])
    @pytest.mark.parametrize('axis', [-1, 1])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    def test_row_vector_and_singleton_expression_have_same_decision(self, factor, axis, dtype, monkeypatch):
        monkeypatch.setenv('TA_BLOCK_BATCHING_CORE_FLOOR', '0')
        hashes = {False: set(), True: set()}
        for n, padding in [(131, 13), (128, 21), (131, 19)]:
            rows, sx, sy, so = 5, n + padding, n + padding + 7, n + padding + 11
            gen = torch.Generator().manual_seed(810)
            xc = torch.randn((rows, sx), generator=gen).to(dtype)
            yc = torch.randn((rows, sy), generator=gen).to(dtype)
            x, y = xc.npu(), yc.npu()
            expected_x, expected_y = xc[:, :n].float(), yc[:, :n].float()
            sig = torch.sigmoid(expected_x)
            expected = [expected_x * sig * expected_y * .5, sig * (1 + expected_x * (1 - sig)) * expected_y * .5]
            pair_outputs = []
            for singleton in (False, True):
                outputs = []
                for f in (1, factor):
                    allocations = [torch.full((rows * so + 32, ), -123., device='npu', dtype=dtype) for _ in range(2)]
                    views = [v[16:-16].reshape(rows, so) for v in allocations]
                    kernel = _row_relative_vector[(rows, triton.cdiv(n, 32))](x, y, *views, n, sx, sy, so, .5,
                                                                              singleton, 32, block_batching_factor=f,
                                                                              block_batching_axis=axis, num_warps=4)
                    torch.npu.synchronize()
                    for out, backing, ref in zip(views, allocations, expected):
                        torch.testing.assert_close(out[:, :n].float().cpu(), ref,
                                                   rtol=.02 if dtype == torch.bfloat16 else 2e-4,
                                                   atol=.02 if dtype == torch.bfloat16 else 2e-5)
                        assert (out[:, n:].cpu() == -123).all()
                        assert (backing[:16].cpu() == -123).all() and (backing[-16:].cpu() == -123).all()
                    if f != 1:
                        assert kernel.last_launch['applied_factor'] == factor, kernel.block_batching
                        assert kernel.last_launch['selected_axis'] == 1
                        assert all(p['rewritten'] and p['action'] == 'PointwiseRegion'
                                   for p in kernel.block_batching['layout']['accesses'])
                        assert 'triton_indirect' not in kernel.candidate.asm['ttadapter']
                        assert 'linalg.transpose' not in kernel.candidate.asm['ttadapter']
                        hashes[singleton].add(kernel.candidate.hash)
                    outputs.append([v.cpu() for v in views])
                for baseline, candidate in zip(*outputs):
                    torch.testing.assert_close(baseline, candidate, rtol=0, atol=0)
                pair_outputs.append(outputs[1])
            for vector, singleton in zip(*pair_outputs):
                torch.testing.assert_close(vector, singleton, rtol=0, atol=0)
            torch.testing.assert_close(x.cpu(), xc, rtol=0, atol=0)
            torch.testing.assert_close(y.cpu(), yc, rtol=0, atol=0)
        assert all(len(v) == (2 if axis == -1 else 1) for v in hashes.values())
