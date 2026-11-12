"""Keep program ownership and recurrence order while grouping loop iterations."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def cyclic_recurrences(X, Y, A, C, N, B: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    grid = tl.num_programs(0)
    offsets = tl.arange(0, B)
    a = tl.full((B, ), 0., tl.float32)
    c = tl.full((B, ), 1., tl.float32)
    for start in range(pid * B, N, grid * B):
        row = start + offsets
        x = tl.load(X + row, row < N, other=0)
        a = a + x
        c = c * .5 + a
        tl.store(Y + row, c, row < N)
    tl.store(A + pid * B + offsets, a)
    tl.store(C + pid * B + offsets, c)


@pytest.mark.parametrize('factor', [2, 3, 6])
@pytest.mark.parametrize('tail', [0, 3, -7])
def test_iteration_groups_keep_all_program_results_bitwise(factor, tail):
    grid, b = 64, 16
    n = grid * b * factor + tail
    gen = torch.Generator().manual_seed(881)
    cpu = torch.randn(n, generator=gen)
    cpu[::11] *= 1e4
    x = cpu.npu()
    variants, snapshots = [], []
    for f in (1, factor):
        storage = [torch.full((size + 32, ), -123., device='npu') for size in (n, grid * b, grid * b)]
        y, a, c = [t[16:-16] for t in storage]
        kernel = cyclic_recurrences[(grid, )](x, y, a, c, n, b, block_batching_factor=f, block_batching_axis=0)
        torch.npu.synchronize()
        for t in storage:
            assert (t[:16] == -123.).all() and (t[-16:] == -123.).all()
        snapshots.append([t.cpu() for t in (y, a, c)])
        variants.append(kernel)
    for left, right in zip(*snapshots):
        assert torch.equal(left.view(torch.int32), right.view(torch.int32))
    expected_y = torch.empty(n)
    expected_a, expected_c = torch.empty(grid * b), torch.empty(grid * b)
    for p in range(grid):
        acc, carry = torch.zeros(b), torch.ones(b)
        for start in range(p * b, n, grid * b):
            take = min(b, n - start)
            value = torch.zeros(b)
            value[:take] = cpu[start:start + take]
            acc = acc + value
            carry = carry * .5 + acc
            expected_y[start:start + take] = carry[:take]
        expected_a[p * b:(p + 1) * b] = acc
        expected_c[p * b:(p + 1) * b] = carry
    for out, expected in zip(snapshots[1], (expected_y, expected_a, expected_c)):
        torch.testing.assert_close(out, expected, rtol=2e-6, atol=2e-5)
    assert torch.equal(x.cpu(), cpu)
    bundle = variants[1]
    assert bundle.last_launch['applied_factor'] == factor, bundle.block_batching
    assert bundle.last_launch['grouping'] == 'loop_iterations'
    assert bundle.last_launch['original_grid'] == bundle.last_launch['batch_grid']
    assert bundle.candidate.metadata.program_grid_transforms is None
    assert variants[0].kernel == bundle.baseline.kernel


def test_iteration_guard_falls_back_for_short_loops_and_recovers():
    grid, b = 64, 16
    seen = set()
    for n in (4099, 0, 513, 4099):
        x = torch.ones(max(n, 1), device='npu')
        outputs = [
            torch.full((max(n, 1), ), -123., device='npu'),
            torch.empty(grid * b, device='npu'),
            torch.empty(grid * b, device='npu')
        ]
        baseline = [torch.empty_like(t) for t in outputs]
        baseline[0].fill_(-123.)
        cyclic_recurrences[(grid, )](x, *baseline, n, b, block_batching_factor=1)
        bundle = cyclic_recurrences[(grid, )](x, *outputs, n, b, block_batching_factor=2, block_batching_axis=0)
        torch.npu.synchronize()
        for actual, expected in zip(outputs, baseline):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert bundle.last_launch['applied_factor'] == (2 if n == 4099 else 1)
        if n != 4099:
            assert bundle.last_launch['reason'] == 'LOOP_GROUP_NOT_FORMED'
        seen.add(bundle.candidate.hash)
    assert len(seen) == 1
