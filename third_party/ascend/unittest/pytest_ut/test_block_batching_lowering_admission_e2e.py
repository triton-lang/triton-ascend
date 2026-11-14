# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Real tiled reduce/scan kernels must not launch residual indirect GM helpers."""
import pytest
import torch
import triton
import triton.language as tl
from block_batching_test_utils import batching_target  # noqa: F401


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


@pytest.mark.parametrize('axis,scans', [(-1, True), (0, True), (1, True), (1, False)])
def test_shared_divided_origin_with_multiple_batches_and_dynamic_pitch(axis, scans):
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
def test_tiled_aggregation_uses_proven_memory_views_or_reuses_fallback(kind, axis, applies, factor):
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


@triton.jit
def row_softmax_with_weight(X, W, Y, N: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    value = tl.load(X + row * STRIDE + col, col < N, other=-float('inf'))
    weight = tl.load(W + col, col < N, other=0)
    exponent = tl.exp(value - tl.max(value, 0))
    normalized = exponent / tl.sum(exponent, 0)
    tl.store(Y + row * STRIDE + col, normalized * weight, col < N)


def test_dependent_row_reductions_and_shared_weight_still_apply():
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


@pytest.mark.parametrize('collapse', [False, True])
@pytest.mark.parametrize('factor', [2, 3])
def test_row_reduction_recurrence_retains_or_rejects_its_domain(collapse, factor):
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
        baseline = row_reduction_recurrence[grid](x, outputs[0], *args, block_batching_factor=1, block_batching_axis=0)
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
