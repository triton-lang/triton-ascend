"""Real compiled variants: explicit grids reuse, auto axes remain distinct."""
import json
import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def cached_row_sum(X, Y, B: tl.constexpr):
    row = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    col = tl.arange(0, B)
    value = tl.load(X + row * B + col)
    tl.store(Y + row, tl.sum(value, 0))


def record(tmp_path, name, value):
    root = os.getenv('TA_BLOCK_BATCHING_TEST_ARTIFACTS')
    folder = Path(root) if root else tmp_path
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name + '.json')).write_text(json.dumps(value, default=str, indent=2))


def test_explicit_axis_static_and_callable_grid_reuse(monkeypatch, tmp_path):
    torch.manual_seed(881)
    source = torch.randn(256, 256)
    x = source.npu()
    y = torch.full((257, ), -12345., device='npu')
    compilations, callable_evaluations, kernels = [], [], []
    original = cached_row_sum._do_compile

    def counted(*args, **kwargs):
        compilations.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cached_row_sum, '_do_compile', counted)

    def dynamic(meta):
        callable_evaluations.append(1)
        return (128, 1)

    for rows, grid in [(128, (128, 1)), (256, (256, 1)), (128, dynamic)]:
        y.fill_(-12345.)
        kernel = cached_row_sum[grid](x, y, 256, block_batching_factor=2, block_batching_axis=0, num_warps=4,
                                      num_stages=1)
        torch.npu.synchronize()
        torch.testing.assert_close(y[:rows].cpu(), source[:rows].sum(1), atol=1e-4, rtol=1e-4)
        assert torch.all(y[rows:].cpu() == -12345.)
        assert kernel.last_launch['applied_factor'] == 2
        assert kernel.baseline.metadata.block_batching_grid is None
        assert kernel.candidate.metadata.block_batching_grid is None
        kernels.append(kernel)
    assert kernels[0] is kernels[1] is kernels[2]
    assert len(compilations) == 1 and len(callable_evaluations) == 1
    record(
        tmp_path, 'explicit_axis_cache',
        dict(jit_compilations=len(compilations), same_bundle=True, callable_evaluations=len(callable_evaluations),
             launch=kernels[-1].last_launch))


def test_automatic_axis_candidates_share_baseline_without_grid_hint(tmp_path):
    torch.manual_seed(881)
    source = torch.randn(128, 256)
    x = source.npu()
    y = torch.full((129, ), -12345., device='npu')
    kernels, launches = [], []
    for grid, axis in [((128, 1), 0), ((1, 128), 1)]:
        y.fill_(-12345.)
        kernel = cached_row_sum[grid](x, y, 256, block_batching_factor=2, block_batching_axis=-1, num_warps=4,
                                      num_stages=1)
        torch.npu.synchronize()
        torch.testing.assert_close(y[:128].cpu(), source.sum(1), atol=1e-4, rtol=1e-4)
        assert y[128].item() == -12345.
        assert kernel.last_launch['applied_factor'] == 2
        assert kernel.last_launch['selected_axis'] == axis
        assert kernel.baseline.metadata.block_batching_grid is None
        kernels.append(kernel)
        launches.append(dict(kernel.last_launch))
    assert kernels[0].candidate.hash != kernels[1].candidate.hash
    assert kernels[0].baseline.hash == kernels[1].baseline.hash
    assert kernels[0].baseline.kernel == kernels[1].baseline.kernel
    record(tmp_path, 'auto_axis_cache', dict(distinct_candidates=True, same_baseline=True, launches=launches))
