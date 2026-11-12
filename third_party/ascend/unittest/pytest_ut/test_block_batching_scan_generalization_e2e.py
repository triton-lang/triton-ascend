"""Layered device scan coverage; each result and every padding region is checked."""
import json
import math
import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def scan_combinations(X, A, C, N: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr, REVERSE: tl.constexpr,
                      COMBINATION: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * STRIDE + col, col < N, other=0)
    first = tl.cumsum(x, 0, reverse=REVERSE)
    if COMBINATION == 0:
        second = first
    elif COMBINATION == 1:
        other = tl.where(col < N, x * 2 + 1, 0)
        second = tl.cumsum(other, 0, reverse=REVERSE)
    else:
        other = tl.where(col < N, first, 0)
        second = tl.cumsum(other, 0, reverse=REVERSE)
    tl.store(A + row * STRIDE + col, first, col < N)
    tl.store(C + row * STRIDE + col, second, col < N)


# (factor, columns, grouping, dtype, reverse, combination, values, resource_case)
CASES = [(f, 31 if f >= 8 else 127, 'full' if i % 2 == 0 else 'tail', 'i32' if i % 2 else 'f32', bool(i % 2), 0,
          'integers', False) for i, f in enumerate((2, 3, 6, 8, 16, 32, 64))]
CASES += [(2, n, 'tail', 'f32', bool(i % 2), i % 3, 'random', n == 4097)
          for i, n in enumerate((15, 16, 17, 255, 256, 257, 4095, 4096, 4097))]
CASES += [(3, 257, 'tail', 'f32', bool(i % 2), 0, kind, False)
          for i, kind in enumerate(('mixed', 'cancellation', 'zeros', 'nan', 'positive_inf', 'negative_inf',
                                    'both_inf'))]
CASES += [(f, 63, grouping, 'i32', reverse, combination, 'integers', False)
          for f, grouping, reverse, combination in ((3, 'full', False, 1), (6, 'tail', True, 2), (8, 'tail', False, 1),
                                                    (16, 'full', True, 2))]
CASES += [(2, 127, grouping, 'f32', False, 0, 'integers', False)
          for grouping in ('below_floor', 'at_floor_tail', 'at_floor')]
CASES += [(f, 8193, 'tail', 'f32', False, 2, 'integers', True) for f in (32, 64)]


def case_id(case):
    return '-'.join(map(str, case))


def source_values(rows, width, dtype, kind):
    generator = torch.Generator().manual_seed(881)
    if kind == 'integers' or dtype == torch.int32:
        return torch.randint(-4, 5, (rows, width), generator=generator, dtype=torch.int32).to(dtype)
    value = torch.randn((rows, width), generator=generator, dtype=dtype) * .1
    if kind == 'mixed':
        value[:, ::3] *= 1e5
        value[:, 1::3] *= 1e-5
    elif kind == 'cancellation':
        pattern = torch.tensor([1e5, 1., -1e5, -1.], dtype=dtype)
        value = pattern.repeat(rows, triton.cdiv(width, 4))[:, :width].clone()
    elif kind == 'zeros':
        value.zero_()
        value[:, ::2] = -0.
    elif kind in ('nan', 'positive_inf', 'negative_inf', 'both_inf'):
        value[:, width // 3] = dict(nan=float('nan'), positive_inf=float('inf'), negative_inf=-float('inf'),
                                    both_inf=float('inf'))[kind]
        if kind == 'both_inf':
            value[:, 2 * width // 3] = -float('inf')
    return value


def reference_scan(value, reverse, dtype):
    ordered = value.flip(1) if reverse else value
    result = ordered.cumsum(1, dtype=torch.float64 if dtype == torch.float32 else torch.int64)
    return result.flip(1) if reverse else result


def check_values(actual, expected, inputs, reverse, dtype, combination=0):
    if dtype == torch.int32:
        torch.testing.assert_close(actual.to(torch.int64), expected.to(torch.int64), rtol=0, atol=0)
        return
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    # A prefix-tree addition error scales with depth and prefix absolute sum;
    # a fixed relative tolerance alone is unsuitable for strong cancellation.
    depth = math.ceil(math.log2(max(2, inputs.shape[1]))) + 1
    magnitude = reference_scan(inputs.abs().double(), reverse, torch.float32)
    if combination == 2:
        magnitude = reference_scan(magnitude, reverse, torch.float32)
    budget = 3e-5 + 3e-4 * expected.abs() + 8 * torch.finfo(torch.float32).eps * depth * magnitude
    assert torch.all((actual.double() - expected).abs()[finite] <= budget[finite])


@pytest.mark.parametrize('case', CASES, ids=case_id)
def test_scan_generalization(case, tmp_path):
    factor, columns, grouping, dtype_name, reverse, combination, values, resource_case = case
    cores = driver.active.utils.get_aivector_core_num()
    rows = factor * max(cores + 8, 64) + (1 if grouping == 'tail' else 0)
    if grouping == 'below_floor':
        rows = factor * (cores - 1)
    elif grouping == 'at_floor_tail':
        rows = factor * cores - 1
    elif grouping == 'at_floor':
        rows = factor * cores
    dtype = torch.float32 if dtype_name == 'f32' else torch.int32
    source = source_values(rows, columns, dtype, values)
    stride = columns + 7
    padded = torch.full((rows + 2, stride), -12345, dtype=dtype)
    padded[:rows, :columns] = source
    x = padded.npu()
    outputs = [[torch.full_like(x, -12345) for _ in range(2)] for _ in range(2)]
    output_root = os.getenv('TA_BLOCK_BATCHING_TEST_ARTIFACTS')
    folder = Path(output_root) / ('scan-' + case_id(case)) if output_root else tmp_path
    folder.mkdir(parents=True, exist_ok=True)
    kernels = []
    for f, out in zip((1, factor), outputs):
        args = (x, *out, columns, stride, triton.next_power_of_2(columns), reverse, combination)
        options = dict(block_batching_factor=f, num_warps=4, num_stages=1)
        compiled = scan_combinations.warmup(*args, grid=(rows, ), **options)
        kernel = compiled if f == 1 else compiled.candidate
        label = 'baseline' if f == 1 else 'candidate'
        if kernel is not None:
            for suffix in ('ttir', 'ttadapter'):
                (folder / (label + '.' + suffix)).write_text(kernel.asm.get(suffix, ''))
            (folder / (label + '.json')).write_text(json.dumps(kernel.metadata._asdict(), default=str, indent=2))
        if f != 1:
            (folder / 'compile_report.json').write_text(json.dumps(compiled.block_batching, default=str, indent=2))
        kernels.append(scan_combinations[(rows, )](*args, **options))
        torch.npu.synchronize()
    wrapper = kernels[1]
    record = dict(case=case, rows=rows, available_aiv=cores, launch=dict(wrapper.last_launch),
                  report=wrapper.block_batching, baseline_binary_identical=kernels[0].kernel == wrapper.baseline.kernel)
    (folder / 'result.json').write_text(json.dumps(record, default=str, indent=2))
    first = reference_scan(source, reverse, dtype)
    if combination == 0:
        second = first
    elif combination == 1:
        second = reference_scan(source * 2 + 1, reverse, dtype)
    else:
        second = reference_scan(first, reverse, dtype)
    host = [[out.cpu() for out in pair] for pair in outputs]
    for pair in host:
        for index, (out, expected) in enumerate(zip(pair, (first, second))):
            inputs = source * 2 + 1 if index == 1 and combination == 1 else source
            check_values(out[:rows, :columns], expected, inputs, reverse, dtype, combination if index else 0)
            assert torch.all(out[:rows, columns:] == -12345)
            assert torch.all(out[rows:] == -12345)
    for index, (base, batched) in enumerate(zip(host[0], host[1])):
        active_base, active_batch = base[:rows, :columns], batched[:rows, :columns]
        assert torch.equal(torch.isnan(active_base), torch.isnan(active_batch))
        if values == 'zeros':
            assert torch.equal(torch.signbit(active_base), torch.signbit(active_batch))
        inputs = source * 2 + 1 if index == 1 and combination == 1 else source
        check_values(active_batch, active_base.double(), inputs, reverse, dtype, combination if index else 0)
    assert record['baseline_binary_identical']
    if grouping == 'below_floor':
        assert wrapper.last_launch['reason'] == 'GRID_BELOW_CORE_FLOOR'
    elif resource_case:
        assert wrapper.last_launch['applied_factor'] == 1, record
        assert wrapper.block_batching['compile_status'] == 'REJECTED'
        diagnostic = json.dumps(wrapper.block_batching).lower()
        assert any(word in diagnostic for word in ('overflow', 'resource', 'capacity', 'out_of_resources'))
    else:
        assert wrapper.last_launch['applied_factor'] == factor, record
        assert wrapper.metadata.block_batching['lane_scans_lowered'] == (1 if combination == 0 else 2)
