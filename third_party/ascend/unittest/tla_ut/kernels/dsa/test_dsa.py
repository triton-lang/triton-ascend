"""Pytest accuracy and performance tests for sparse MLA prefill.

All tests use B1, SQ=KV=8192, TopK=2048, QK=576/V=512, BF16.
Profiling includes every device task in one invocation, including preprocessing.
No reference output, selected-value cache or input-specific dispatch is used.
"""

from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import tempfile

import pytest

from kernels.utils import device_snapshot, load_kernel, passed, pipe_counters, profile, save_environment, save_json

# Edit these settings to select tests; no DSA-specific environment variables.
HEADS = (128, 64, 32, 16)
SEED = 1103
ACCURACY_FAMILIES = ('normal', 'uniform', 'peaked', 'duplicates', 'invalid')
ACCURACY_OPERATOR = 'tla'  # tla, ascendc, both
CORE_LIMIT = None
RUN_PERF = False
PERF_OPERATOR = 'both'
PERF_ROUNDS = 3
PERF_SAMPLES = 40
PERF_COUNTERS = False  # Also export Level2 PipeUtilization using msprof.
PERF_OUTPUT = Path('dsa-results')


def make_inputs(heads, seed, family='normal'):
    shapes = [(1, 8192, heads, 512), (1, 8192, 1, 512), (1, 8192, heads, 64), (1, 8192, 1, 64)]
    generator = torch.Generator().manual_seed(seed)
    q, kv, qr, kr = [torch.randn(s, generator=generator).bfloat16().npu() for s in shapes]
    indices = (torch.stack([torch.randperm(8192, generator=generator)[:2048]
                            for _ in range(8192)]).reshape(1, 8192, 1, 2048).int().npu())
    if family == 'uniform':
        q.zero_()
        qr.zero_()
    elif family == 'peaked':
        q.mul_(4)
        qr.mul_(4)
    elif family == 'duplicates':
        indices[..., 1::2] = indices[..., 0::2]
    elif family == 'invalid':
        indices[..., 1024:] = -1
    return q, kv, qr, kr, indices


def ascendc(data):
    q, kv, qr, kr, indices = data
    return torch_npu.npu_sparse_flash_attention(
        q,
        kv,
        kv,
        indices,
        576**-0.5,
        query_rope=qr,
        key_rope=kr,
        attention_mode=2,
        sparse_mode=0,
        return_softmax_lse=True,
    )


def check_outputs(actuals, data, chunk=32):
    """Compare every element to independent FP32 gathered dense attention."""
    q, kv, qr, kr, ids = data
    names, thresholds = ('out', 'max', 'sum'), (0.005, 0.0001, 0.0001)
    result = {
        model: {
            name:
            dict(
                finite=True,
                max_error=0.0,
                max_reference=0.0,
                error_squared=0.0,
                reference_squared=0.0,
                elements=0,
            )
            for name in names
        }
        for model in actuals
    }
    for start in range(0, 8192, chunk):
        end = min(start + chunk, 8192)
        selected_ids = ids[0, start:end, 0].long()
        valid = (selected_ids >= 0) & (selected_ids < 8192)
        selected_ids = selected_ids.clamp(0, 8191)
        selected = kv[0, selected_ids, 0].float()
        rope = kr[0, selected_ids, 0].float()
        scores = (torch.bmm(q[0, start:end].float(), selected.transpose(1, 2)) +
                  torch.bmm(qr[0, start:end].float(), rope.transpose(1, 2))) / 24.0
        scores.masked_fill_(~valid[:, None, :], -float('inf'))
        maximum = scores.amax(-1)
        probability = (scores - maximum[:, :, None]).exp()
        denominator = probability.sum(-1)
        reference = torch.bmm(probability / denominator[:, :, None], selected)
        for model, values in actuals.items():
            pieces = values[0][0, start:end], values[1][0, 0, start:end], values[2][0, 0, start:end]
            for name, value, target in zip(names, pieces, (reference, maximum, denominator)):
                assert value.shape == target.shape
                delta = value.float() - target
                row = result[model][name]
                row['finite'] &= bool(value.isfinite().all())
                row['max_error'] = max(row['max_error'], float(delta.abs().max()))
                row['max_reference'] = max(row['max_reference'], float(target.abs().max()))
                row['error_squared'] += float(delta.square().sum())
                row['reference_squared'] += float(target.square().sum())
                row['elements'] += target.numel()
    for outputs in result.values():
        for name, threshold in zip(names, thresholds):
            row = outputs[name]
            row['max_normalized'] = row['max_error'] / max(row['max_reference'], 1e-30)
            row['relative_rms'] = (row['error_squared'] / max(row['reference_squared'], 1e-30))**0.5
            row['threshold'] = threshold
            row['passed'] = row['finite'] and row['max_normalized'] <= threshold
    return result


@contextmanager
def accuracy_case(kernel, heads, seed, family, operator):
    """Validate once and keep the same inputs alive for optional measurements."""
    previous_hf32 = torch.npu.matmul.allow_hf32
    data = outputs = None
    calls = {}
    try:
        torch.npu.matmul.allow_hf32 = False
        data = make_inputs(heads, seed, family)
        if operator != 'tla':
            calls['ascendc'] = lambda: ascendc(data)
        if operator != 'ascendc':
            calls['tla'] = lambda: kernel.forward(*data, core_limit=CORE_LIMIT)
        outputs = {}
        for name, call in calls.items():
            outputs[name] = call()
            torch.npu.synchronize()
        checks = check_outputs(outputs, data)
        outputs = None
        yield calls, checks
    finally:
        torch.npu.matmul.allow_hf32 = previous_hf32
        calls.clear()
        data = outputs = None
        gc.collect()
        torch.npu.empty_cache()


@pytest.fixture
def runtime(request):
    from utils.tla_test_utils import get_stages

    if not get_stages()[2]:
        pytest.skip('DSA only supports the run stage')
    if request.node.originalname == 'test_perf':
        if not RUN_PERF:
            pytest.skip('Set RUN_PERF = True to benchmark')
        if hasattr(request.config, 'workerinput'):
            pytest.skip('Run performance tests without pytest-xdist')
    assert HEADS and all(h in (128, 64, 32, 16) for h in HEADS)
    assert ACCURACY_FAMILIES and all(f in ('normal', 'uniform', 'peaked', 'duplicates', 'invalid')
                                     for f in ACCURACY_FAMILIES)
    assert ACCURACY_OPERATOR in ('tla', 'ascendc', 'both')
    assert PERF_OPERATOR in ('tla', 'ascendc', 'both')
    assert PERF_ROUNDS > 0 and PERF_SAMPLES > 0
    assert CORE_LIMIT is None or CORE_LIMIT > 0
    global torch, torch_npu
    torch = pytest.importorskip('torch')
    torch_npu = pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('DSA requires an NPU')
    return load_kernel(Path(__file__).with_name('dsa_kernel.py'))


@pytest.mark.parametrize('heads', HEADS)
def test_run(runtime, heads):
    for family in ACCURACY_FAMILIES:
        with accuracy_case(runtime, heads, SEED, family, ACCURACY_OPERATOR) as (_, checks):
            assert passed(checks), f'heads={heads}, family={family}, seed={SEED}: {json.dumps(checks, indent=2)}'


@pytest.mark.parametrize('heads', HEADS)
def test_perf(runtime, heads):
    """Precision gate, alternating-order timing, and optional pipe counters."""
    seed, family = SEED, 'normal'
    snapshot = device_snapshot()
    assert snapshot['returncode'] == 0, snapshot['text']
    assert not (set(snapshot['pids']) - {str(os.getpid())}), 'Device has other processes'
    PERF_OUTPUT.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'h{heads}-{seed}-{family}-', dir=PERF_OUTPUT))
    save_environment(
        folder, runtime,
        dict(heads=heads, seed=seed, family=family, operator=PERF_OPERATOR, rounds=PERF_ROUNDS, samples=PERF_SAMPLES,
             counters=PERF_COUNTERS, core_limit=CORE_LIMIT))
    row = dict(heads=heads, seed=seed, family=family, shape=[1, 8192, heads, 8192, 2048],
               config=asdict(runtime.default_config(heads)), profiles={}, passed=False)
    try:
        with accuracy_case(runtime, heads, seed, family, PERF_OPERATOR) as (calls, checks):
            row['checks'] = checks
            assert passed(checks), json.dumps(checks, indent=2)
            order = list(calls)
            for round_id in range(PERF_ROUNDS):
                offset = round_id % len(order)
                for position, name in enumerate(order[offset:] + order[:offset]):
                    measured = profile(calls[name], folder / f'{name}-{round_id}.json', PERF_SAMPLES)
                    measured.update(round=round_id, order_position=position)
                    row['profiles'].setdefault(name, []).append(measured)
                    save_json(folder / 'results.json', [row])
            if PERF_COUNTERS:
                row['pipe_counters'] = {name: pipe_counters(call, folder / name) for name, call in calls.items()}
            row['passed'] = True
    except Exception as exc:
        row['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', [row])
        print(f'DSA results: {folder}')
