# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['M', 'N'])
def _layout_tile(X, Y, M, N, STRIDE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    index = m[:, None] * STRIDE + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    x = tl.load(X + index, mask, other=7.0)
    tl.store(Y + index, x * 2.0 + 1.0, mask)


@triton.jit(do_not_specialize=['N'])
def _layout_decomposed(X, Y, N, D0: tl.constexpr, D1: tl.constexpr):
    local = tl.arange(0, D0 * D1).reshape((D0, D1))
    index = tl.program_id(0) * D0 * D1 + local
    x = tl.load(X + index, index < N, other=7.0)
    tl.store(Y + index, x * 2 + 1, index < N)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('factor', [2, 3, 6])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_row_and_column_tails_preserve_padding(factor, dtype, axis):
    stride, bm, bn = 32, 4, 8
    for m, n in ((0, 0), (1, 1), (3, 7), (4, 8), (5, 9), (13, 25)):
        standard = max(1, triton.cdiv(m, bm) if axis == 0 else triton.cdiv(n, bn))
        for selected in sorted({max(1, standard - 1), standard, standard + 2}):
            gm = selected if axis == 0 else max(1, triton.cdiv(m, bm))
            gn = selected if axis == 1 else max(1, triton.cdiv(n, bn))
            rows = gm * bm + 4
            x = torch.arange(rows * stride, device='npu', dtype=dtype)
            y = torch.full_like(x, -123)
            kernel = _layout_tile[(gm, gn)](x, y, m, n, stride, bm, bn, block_batching_factor=factor,
                                            block_batching_axis=axis)
            torch.npu.synchronize()
            expected = torch.full((rows, stride), -123, dtype=dtype)
            covered = min(m, gm * bm)
            columns = min(n, gn * bn)
            expected[:covered, :columns] = (
                torch.arange(rows * stride, dtype=dtype).reshape(rows, stride)[:covered, :columns] * 2 + 1)
            torch.testing.assert_close(y.cpu().reshape(rows, stride), expected, rtol=0, atol=0)
            launch = kernel.last_launch
            if selected == 1:
                assert launch['reason'] == 'GRID_NOT_REDUCED', launch
                assert launch['applied_factor'] == 1
            else:
                assert launch['runtime_status'] == 'APPLIED', launch
                assert launch['applied_factor'] == factor
            assert all(r['rewritten'] for r in kernel.block_batching['layout']['accesses'])
            observed = kernel.block_batching['layout']['lowering_observation']
            assert observed['indirect_load_ops'] == observed['indirect_store_ops'] == 0, observed
            assert observed['linalg_transpose_ops'] == 0, observed


@pytest.mark.parametrize('factor', [2, 3, 6])
def test_ordered_decomposition_and_two_tails(factor):
    for n in (0, 1, 15, 16, 17, 47, 48, 49):
        for grid in (1, 3, 5):
            size = max(n, grid * 16) + 16
            x = torch.arange(size, device='npu', dtype=torch.float32)
            y = torch.full_like(x, -123)
            kernel = _layout_decomposed[(grid, )](x, y, n, 2, 8, block_batching_factor=factor, block_batching_axis=0)
            torch.npu.synchronize()
            expected = torch.full((size, ), -123.0)
            count = min(n, grid * 16)
            expected[:count] = torch.arange(count) * 2 + 1
            torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
            launch = kernel.last_launch
            if grid == 1:
                assert launch['reason'] == 'GRID_NOT_REDUCED', launch
                assert launch['applied_factor'] == 1
            else:
                assert launch['runtime_status'] == 'APPLIED', launch
                assert launch['applied_factor'] == factor
            assert all(r['rewritten'] for r in kernel.block_batching['layout']['accesses'])
            observed = kernel.block_batching['layout']['lowering_observation']
            assert observed['indirect_load_ops'] == observed['indirect_store_ops'] == 0, observed
            assert observed['linalg_transpose_ops'] == 0, observed
