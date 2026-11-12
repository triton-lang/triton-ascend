"""Real kernels with equivalent singleton/broadcast/control-flow expressions."""
import json
import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def equivalent_rows(X, Y, Z, Alpha, Gamma, N: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr,
                    SINGLETON: tl.constexpr, BROADCAST: tl.constexpr, CONTROL: tl.constexpr, SCANS: tl.constexpr, flag,
                    repeats):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    if SINGLETON:
        col = col[None, :]
    x = tl.load(X + row * STRIDE + col, col < N, other=0)
    gamma = tl.load(Gamma + col, col < N, other=0)
    if BROADCAST:
        alpha = tl.load(Alpha + tl.full(col.shape, 0, tl.int32))
    else:
        alpha = tl.load(Alpha)
    value = x * alpha + gamma
    if CONTROL == 1:
        if flag > 0:
            value = x * alpha + gamma
        else:
            value = x * alpha + gamma
    elif CONTROL == 2:
        for _ in range(repeats):
            value = value * 1.
    value = tl.where(col < N, value, 0.)
    if SCANS > 0:
        value = tl.cumsum(value, 1 if SINGLETON else 0)
    second = value
    if SCANS == 2:
        second = tl.cumsum(tl.where(col < N, value, 0.), 1 if SINGLETON else 0)
    tl.store(Y + row * STRIDE + col, value, col < N)
    tl.store(Z + row * STRIDE + col, second, col < N)


# Baseline expression is rank-one/scalar/plain. The compared side changes one
# expression, with a few crosses against tails and multiple live scans.
PAIRS = [
    ('singleton', True, False, 0, 0, False),
    ('singleton_tail', True, False, 0, 0, True),
    ('scalar_broadcast', False, True, 0, 0, True),
    ('scalar_vector_singleton', True, True, 0, 0, True),
    ('equivalent_if', False, False, 1, 0, True),
    ('equivalent_loop', False, False, 2, 0, True),
    ('scan_singleton', True, False, 0, 1, True),
    ('multi_scan_broadcast', False, True, 0, 2, True),
    ('multi_scan_if', False, False, 1, 2, True),
    ('multi_scan_loop', False, False, 2, 2, True),
]


def save_variant(folder, label, kernel):
    if kernel is None:
        return
    for ext in ('ttir', 'ttadapter'):
        (folder / (label + '.' + ext)).write_text(kernel.asm.get(ext, ''))
    (folder / (label + '.json')).write_text(json.dumps(kernel.metadata._asdict(), default=str, indent=2))


@pytest.fixture(scope='module')
def compiled_rewrites():
    return {}


@pytest.mark.parametrize('pair', PAIRS, ids=lambda p: p[0])
def test_equivalent_kernel_decisions(pair, tmp_path, monkeypatch, compiled_rewrites):
    from triton._C.libtriton import ascend
    rewrite = ascend.passes.ttir.rewrite_block_batching
    captures = []

    def observe(module, *args):
        normalized = str(module)
        report = rewrite(module, *args)
        captures.append((normalized, str(module)))
        return report

    monkeypatch.setattr(ascend.passes.ttir, 'rewrite_block_batching', observe)
    name, singleton, broadcast, control, scans, tail = pair
    rows = 2 * max(64, driver.active.utils.get_aivector_core_num() + 8) + int(tail)
    columns, stride = 129, 136
    torch.manual_seed(881)
    # Dyadic inputs make both scans and all equivalent arithmetic exact here.
    source = torch.randint(-4, 5, (rows, columns)).float() * .125
    gamma_host = torch.randint(-2, 3, (columns, )).float() * .125
    x = torch.full((rows + 2, stride), -12345., device='npu')
    x[:rows, :columns] = source.npu()
    alpha = torch.tensor([.5], device='npu')
    gamma = gamma_host.npu()
    results, decisions = [], []
    root = os.getenv('TA_BLOCK_BATCHING_TEST_ARTIFACTS')
    folder = Path(root) / ('equivalence-' + name) if root else tmp_path
    folder.mkdir(parents=True, exist_ok=True)
    expected = source * .5 + gamma_host
    if scans:
        expected = expected.cumsum(1)
    expected_second = expected.cumsum(1) if scans == 2 else expected
    for label, flags in [('plain', (False, False, 0)), ('transformed', (singleton, broadcast, control))]:
        outputs = [[torch.full_like(x, -12345.) for _ in range(2)] for _ in range(2)]
        kernels = [
            equivalent_rows[(rows, )](x, *out, alpha, gamma, columns, stride, 256, *flags, scans, 2, 2,
                                      block_batching_factor=factor, num_warps=4, num_stages=1)
            for out, factor in zip(outputs, (1, 2))
        ]
        torch.npu.synchronize()
        wrapper = kernels[1]
        if captures:
            compiled_rewrites[wrapper.hash] = captures[-1]
            captures.clear()
        captured = compiled_rewrites.get(wrapper.hash)
        if captured is not None:
            for stage, text in zip(('normalized', 'rewritten'), captured):
                (folder / (label + '_' + stage + '.ttir')).write_text(text)
        save_variant(folder, label + '_baseline', kernels[0])
        save_variant(folder, label + '_candidate', wrapper.candidate)
        decision = dict(expression=label, launch=dict(wrapper.last_launch), report=wrapper.block_batching,
                        baseline_binary_identical=kernels[0].kernel == wrapper.baseline.kernel,
                        frontend_ir='saved' if captured is not None else 'reused_from_disk_cache')
        decisions.append(decision)
        (folder / 'result.json').write_text(json.dumps(dict(pair=pair, decisions=decisions), default=str, indent=2))
        host = [[out.cpu() for out in group] for group in outputs]
        for group in host:
            for out, reference in zip(group, (expected, expected_second)):
                torch.testing.assert_close(out[:rows, :columns], reference, rtol=0, atol=0)
                assert torch.all(out[:rows, columns:] == -12345.)
                assert torch.all(out[rows:] == -12345.)
        assert decision['baseline_binary_identical']
        if scans:
            assert wrapper.last_launch['applied_factor'] == 2, decision
            assert wrapper.block_batching['lane_scans_lowered'] == scans
        else:
            assert wrapper.last_launch['reason'] == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED', decision
            analysis = wrapper.block_batching['candidate_analysis']
            assert analysis['selected_axis'] == 0
            assert all(access['control_flow_independent'] for access in analysis['layout']['accesses'])
        results.append(host[1])
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    same = decisions[0]['launch']['applied_factor'] == decisions[1]['launch']['applied_factor']
    record = dict(pair=pair, correct=True, same_decision=same, decisions=decisions)
    (folder / 'result.json').write_text(json.dumps(record, default=str, indent=2))
    assert same, record


@triton.jit
def conditional_access(X, Y, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr, flag):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    if flag > 0:
        value = tl.load(X + row * S + col, col < N, other=0.)
    else:
        value = tl.load(X + row * S + col + 1, col + 1 < N, other=0.)
    tl.store(Y + row * S + col, value, col < N)


@pytest.mark.parametrize('flag', [2, -2])
def test_control_flow_affecting_memory_keeps_existing_path(flag, tmp_path):
    rows, columns, stride = 129, 129, 136
    source = torch.arange(rows * columns).reshape(rows, columns).float()
    x = torch.full((rows + 2, stride), -12345., device='npu')
    x[:rows, :columns] = source.npu()
    outputs = [torch.full_like(x, -12345.) for _ in range(2)]
    kernels = []
    for factor, out in zip((1, 2), outputs):
        kernels.append(conditional_access[(rows, )](x, out, columns, stride, 256, flag, block_batching_factor=factor,
                                                    num_warps=4, num_stages=1))
        torch.npu.synchronize()
    expected = source if flag > 0 else torch.cat((source[:, 1:], torch.zeros(rows, 1)), 1)
    for out in outputs:
        host = out.cpu()
        torch.testing.assert_close(host[:rows, :columns], expected, rtol=0, atol=0)
        assert torch.all(host[rows:] == -12345.) and torch.all(host[:rows, columns:] == -12345.)
    wrapper = kernels[1]
    assert wrapper.last_launch['applied_factor'] == 2, wrapper.block_batching
    assert any(not access['control_flow_independent'] for access in wrapper.block_batching['layout']['accesses'])
    root = os.getenv('TA_BLOCK_BATCHING_TEST_ARTIFACTS')
    folder = Path(root) / ('control_memory_' + str(flag)) if root else tmp_path
    folder.mkdir(parents=True, exist_ok=True)
    save_variant(folder, 'baseline', kernels[0])
    save_variant(folder, 'candidate', wrapper.candidate)
    (folder / 'result.json').write_text(
        json.dumps(dict(launch=wrapper.last_launch, report=wrapper.block_batching), default=str, indent=2))
