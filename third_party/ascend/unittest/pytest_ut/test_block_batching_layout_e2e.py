# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import pytest
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def _layout_probe(X, Y, N, ORIGIN: tl.constexpr, B: tl.constexpr, OBSERVE_OTHER: tl.constexpr,
                  CLAMP: tl.constexpr = False):
    if CLAMP:
        N = tl.minimum(tl.maximum(N, 0), 19)
    p = tl.program_id(0)
    local = tl.arange(0, B)
    index = ORIGIN + p * B + local
    value = tl.load(X + index, index < N, other=7.0)
    result = value + p + tl.num_programs(0)
    if OBSERVE_OTHER:
        tl.store(Y + p * B + local, result)
    else:
        tl.store(Y + index, result, index < N)


@pytest.mark.parametrize('factor', [2, 3, 6])
@pytest.mark.parametrize('origin', [0, 4])
@pytest.mark.parametrize('observe_other', [False, True])
def test_arbitrary_element_and_group_tails(factor, origin, observe_other):
    block = 8
    seen = {}
    for n in (0, 1, 7, 8, 9, 23, 24, 25):
        standard = triton.cdiv(max(0, n - origin), block)
        for grid in sorted({0, max(0, standard - 1), standard, standard + 2}):
            size = max(n, origin + grid * block) + 16
            x = torch.arange(size, device='npu', dtype=torch.float32)
            y = torch.full((size, ), -123.0, device='npu')
            compiled = _layout_probe[(grid, )](x, y, n, origin, block, observe_other, block_batching_factor=factor,
                                               block_batching_axis=0)
            torch.npu.synchronize()
            expected = torch.full((size, ), -123.0)
            for p in range(grid):
                for c in range(block):
                    idx = origin + p * block + c
                    if observe_other or idx < n:
                        output = p * block + c if observe_other else idx
                        expected[output] = (idx if idx < n else 7.0) + p + grid
            torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
            if grid:
                assert compiled.last_launch['runtime_status'] == 'APPLIED', compiled.block_batching
                assert any(r['rewritten'] for r in compiled.block_batching['layout']['accesses'])
                assert compiled.block_batching['layout']['selected_policy'] == 'auto'
                key = (compiled.baseline.hash, compiled.candidate.hash)
                seen[key] = seen.get(key, 0) + 1
            else:
                assert compiled.last_launch['runtime_status'] == 'EMPTY_GRID'
    assert any(count > 1 for count in seen.values())


@pytest.mark.parametrize('graph', [False, True])
def test_auto_and_semantic_only_same_baseline(graph):
    x = torch.arange(48, device='npu', dtype=torch.float32)
    y = torch.full_like(x, -123)
    compiled = []
    outputs = []
    for policy in ('auto', 'semantic_only', 'auto'):
        y.fill_(-123)
        kernel = _layout_probe[(3, )](x, y, 19, 0, 8, False, block_batching_factor=3, block_batching_axis=0,
                                      block_batching_layout=policy, enable_graph_optimize=graph)
        torch.npu.synchronize()
        assert kernel.last_launch['runtime_status'] == 'APPLIED'
        compiled.append(kernel)
        outputs.append(y.clone())
    assert compiled[0] is compiled[2]
    assert compiled[0].candidate.hash != compiled[1].candidate.hash
    assert compiled[0].baseline.asm['ttir'] == compiled[1].baseline.asm['ttir']
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)


def test_clamped_scalar_bound_uses_same_jit_entry():
    x = torch.arange(64, device='npu', dtype=torch.float32)
    y = torch.full_like(x, -123)
    for n in (-1, 0, 7, 19, 27, 7):
        y.fill_(-123)
        kernel = _layout_probe[(5, )](x, y, n, 0, 8, False, True, block_batching_factor=3, block_batching_axis=0)
        torch.npu.synchronize()
        expected = torch.full((64, ), -123.0)
        limit = min(max(n, 0), 19)
        for i in range(limit):
            expected[i] = i + i // 8 + 5
        torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
        assert kernel.last_launch['runtime_status'] == 'APPLIED', kernel.block_batching
        assert all(r['rewritten'] for r in kernel.block_batching['layout']['accesses'])


def test_eight_byte_data_memory_view():
    x = torch.arange(64, device='npu', dtype=torch.int64)
    y = torch.full_like(x, -123)
    kernel = _layout_probe[(3, )](x, y, 19, 0, 8, False, block_batching_factor=3, block_batching_axis=0)
    torch.npu.synchronize()
    expected = torch.full((64, ), -123, dtype=torch.int64)
    for i in range(19):
        expected[i] = i + i // 8 + 3
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    assert kernel.last_launch['runtime_status'] == 'APPLIED', kernel.block_batching
    assert all(r['rewritten'] for r in kernel.block_batching['layout']['accesses'])
