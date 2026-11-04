"""Paged MLA decode: IR, compilation, full precision checks and profiling.

BF16, B1, SQ4, KV131072, QK576/V512, H96/48/24/12, 128-token pages.
The reference and AscendC use scale 0.04 and bottom-right causal attention.
Select HEADS/operators below, or run this file with --help. TLA_TEST_STAGE
uses the existing ir/compile/run/all convention; profiling is opt-in.
"""
import argparse
from contextlib import contextmanager
import gc
import importlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

import pytest

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernels.utils import device_snapshot, load_kernel, passed, pipe_counters, profile, save_environment, save_json

HEADS = (96, 48, 24, 12)
SEEDS = (1104, 2027)
FAMILIES = ('normal', 'uniform', 'peaked', 'repeated_pages', 'causal_boundary')
SCALE = 0.04
ACCURACY_OPERATOR = 'tla'  # tla, ascendc, both
ASCENDC_PACKAGE = 'cann_ops_transformer'
CORE_LIMIT = None  # Diagnostic cap; normal launches query runtime core counts.
RUN_PERF = False
PERF_OPERATOR = 'both'
PERF_ROUNDS = 3
PERF_SAMPLES = 40
PERF_COUNTERS = False
PERF_OUTPUT = Path('mla-results')


def make_inputs(heads, seed, family):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((1, 4, heads, 576), generator=generator).bfloat16()
    # Extra physical pages and a permutation catch contiguous-cache shortcuts.
    kv = torch.randn((1040, 128, 1, 576), generator=generator).bfloat16()
    table = torch.randperm(1040, generator=generator)[:1024].reshape(1, 1024).int()
    if family == 'uniform':
        q.zero_()
    elif family == 'peaked':
        q.mul_(4)
    elif family == 'repeated_pages':
        table[:, 1::2] = table[:, 0::2]
    elif family == 'causal_boundary':
        # Give each of the last four positions a distinct, highly attractive
        # key and value. Missing the last-query causal bounds then fails visibly.
        q.zero_()
        q[..., 512] = 16
        kv[..., 512] = 0
        last_page = int(table[0, -1])
        for tail in range(4):
            kv[last_page, 124 + tail, 0, 512] = 16 + tail
            kv[last_page, 124 + tail, 0, :512] = (tail + 1) * 0.25
    elif family != 'normal':
        raise ValueError(f'Unknown input family: {family}')
    return tuple(t.npu() for t in (q, kv, table))


def reference(data, scale=SCALE, causal=True):
    """Independent dense FP32 reference, including paged addressing and RoPE."""
    q, kv, table = data
    h = q.shape[2]
    keys = kv[table[0].long(), :, 0, :].reshape(131072, 576).float()
    qflat = q.reshape(-1, 576).float()
    output = torch.empty((4 * h, 512), device=q.device, dtype=torch.float32)
    lse = torch.empty((4 * h, ), device=q.device, dtype=torch.float32)
    positions = torch.arange(131072, device=q.device)
    previous_hf32 = torch.npu.matmul.allow_hf32
    try:
        torch.npu.matmul.allow_hf32 = False
        for start in range(0, 4 * h, 16):
            stop = min(start + 16, 4 * h)
            scores = torch.mm(qflat[start:stop], keys.T) * scale
            if causal:
                query = torch.arange(start, stop, device=q.device) // h
                bound = 131072 - 4 + query + 1
                scores.masked_fill_(positions[None, :] >= bound[:, None], -float('inf'))
            maximum = scores.amax(-1, keepdim=True)
            exponent = (scores - maximum).exp()
            denominator = exponent.sum(-1, keepdim=True)
            output[start:stop] = torch.mm(exponent / denominator, keys[:, :512])
            lse[start:stop] = (maximum + denominator.log()).flatten()
    finally:
        torch.npu.matmul.allow_hf32 = previous_hf32
    return output.reshape(1, 4, h, 512), lse.reshape(1, 4, h)


def check(actual, expected):
    metrics = {}
    for name, value, target, limit in zip(('out', 'lse'), actual, expected, (0.005, 0.0001)):
        assert tuple(value.shape) == tuple(target.shape), (name, value.shape, target.shape)
        error = value.float() - target
        peak = float(target.abs().max())
        norm = float(error.abs().max()) / max(peak, 1e-30)
        rms = math.sqrt(float(error.square().sum()) / max(float(target.square().sum()), 1e-30))
        finite = bool(value.isfinite().all())
        metrics[name] = dict(elements=value.numel(), finite=finite, max_error=float(error.abs().max()),
                             max_reference=peak, max_normalized=norm, relative_rms=rms, threshold=limit, passed=finite
                             and norm <= limit and rms <= limit)
    return metrics


def ascendc_api():
    """Resolve the actual Flash MLA API; never substitute a different attention op."""
    candidates = [
        (ASCENDC_PACKAGE + '.ops.attention.flash_mla', 'flash_mla', 'flash_mla_metadata'),
        (ASCENDC_PACKAGE + '.ops.attention.flash_mla_with_kvcache', 'flash_mla_with_kvcache',
         'flash_mla_with_kvcache_metadata'),
    ]
    errors = []
    for module, op, metadata in candidates:
        try:
            loaded = importlib.import_module(module)
            return getattr(loaded, op), getattr(loaded, metadata)
        except (ImportError, AttributeError) as exc:
            errors.append(f'{module}: {exc}')
    raise RuntimeError('AscendC Flash MLA baseline unavailable; ' + '; '.join(errors))


def native_inputs(data, causal=True):
    """Prepare input lengths and the immutable mask outside timing for AscendC."""
    q = data[0]
    lengths = torch.full((1, ), 131072, dtype=torch.int32, device=q.device)
    used_q = torch.full((1, ), 4, dtype=torch.int32, device=q.device)
    mask = torch.ones((2048, 2048), dtype=torch.int8, device=q.device).triu_(1) if causal else None
    return lengths, used_q, mask


def ascendc_metadata(data, api, inputs, causal=True):
    return api[1](cache_seqlens=inputs[0], num_heads_q=data[0].shape[2], num_heads_kv=1, seqused_q=inputs[1],
                  head_dim_qk=576, head_dim_v=512, mask_mode=3 if causal else 0, layout_q='BSND')


def ascendc(data, api, inputs, scale=SCALE, causal=True, metadata=None):
    """Run official attention, optionally reusing length/shape metadata."""
    op = api[0]
    q, kv, table = data
    lengths, used_q, mask = inputs
    mode = 3 if causal else 0
    if metadata is None:
        metadata = ascendc_metadata(data, api, inputs, causal)
    out, lse = op(q, kv, block_table=table, cache_seqlens=lengths, seqused_q=used_q, attn_mask=mask, metadata=metadata,
                  head_dim_v=512, softmax_scale=scale, mask_mode=mode, layout_q='BSND', layout_kv='PA_BBND',
                  layout_out='BSND', return_softmax_lse=True)
    # The official BSND interface returns LSE in BNS order.
    return out, lse.transpose(1, 2)


def offline_ir(kernel, heads, folder, cores=29, causal=True):
    """Frontend/HIVM check with a diagnostic core count; never initializes NPU."""
    from triton._C.libtriton import ir, buffer_ir, passes
    from triton._C.libtriton.ascend import ir as ascend_ir
    from triton.backends.ascend.compiler import NPUOptions
    from triton.language.extra.tla.compiler import ast_to_ttir, lower_ttir

    cfg = kernel.launch_config(heads, cores, 2 * cores)
    split_entry = kernel.split_kernel(cfg['block_m'])
    entries = [
        (split_entry, dict(Q='*bf16', KV='*bf16', TABLE='*i32', PART='*fp32', MAXIMUM='*fp32', DENOMINATOR='*fp32'),
         dict(H=heads, BM=cfg['block_m'], SPLITS=cfg['splits'], CORES=cfg['cube_grid'], SCALE=SCALE, CAUSAL=causal)),
        (kernel._mla_merge, dict(PART='*fp32', MAXIMUM='*fp32', DENOMINATOR='*fp32', OUT='*bf16',
                                 LSE='*fp32'), dict(ROWS=4 * heads, SPLITS=cfg['splits'], CORES=cfg['vector_grid'])),
    ]
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for function, signature, constants in entries:
        ctx = ir.context()
        ir.load_dialects(ctx)
        buffer_ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)
        options = NPUOptions(arch='Ascend950PR_9579', sanitize_overflow=False, enable_auto_bind_sub_block=False,
                             enable_preload=False)
        module = ast_to_ttir(function, signature, constants, ctx, options)
        (folder / f'{function.__name__}.ttir').write_text(str(module))
        pm = ir.pass_manager(ctx)
        passes.common.add_symbol_dce(pm)
        pm.run(module, 'module')
        module = lower_ttir(module, ctx)
        text = str(module)
        assert module.verify()
        assert '!tt.ptr' not in text and 'tensor<' not in text
        assert 'hacc.mode = "tla"' in text
        required = ('hivm.hir.nd2nz', 'hivm.hir.mmadL1', 'hivm.hir.fixpipe') if function == split_entry else ()
        for op in required:
            assert op in text, op
        if function == split_entry:
            assert 'hivm.func_core_type<MIX>' in text and 'dual_dst_mode = <ROW_SPLIT>' in text
            copies = [line for line in text.splitlines() if 'hivm.hir.copy ins(' in line]
            assert copies and all('address_space<ub>' in line and 'address_space<cbuf>' in line for line in copies)
            constants_ir = {
                name: int(value)
                for name, value in re.findall(r'(%[\w]+) = arith.constant (\d+) : index', text)
            }
            pv = [line for line in text.splitlines() if 'hivm.hir.mmadL1 ins(' in line]
            assert pv
            for line in pv:
                operands = re.search(r'ins\((.*?) :', line).group(1).split(', ')
                assert tuple(constants_ir[x] for x in operands[-3:]) == (cfg['block_m'], 128, 512)
        else:
            assert 'hivm.func_core_type<AIV>' in text
        (folder / f'{function.__name__}.ttadapter').write_text(text)
        records.append(dict(kernel=function.__name__, constants=constants, ir_verified=True))
    return dict(heads=heads, diagnostic_core_count=cores, config=cfg, kernels=records)


@contextmanager
def accuracy_case(kernel, heads, seed, family, operator, include_prepared=False):
    data = expected = None
    calls = {}
    try:
        data = make_inputs(heads, seed, family)
        expected = reference(data)
        if operator != 'tla':
            api, inputs = ascendc_api(), native_inputs(data)
            calls['ascendc'] = lambda: ascendc(data, api, inputs)
            if include_prepared:
                # This metadata depends on lengths/shapes only. Report the
                # official reuse path separately from regenerating it per call.
                prepared = ascendc_metadata(data, api, inputs)
                calls['ascendc_prepared'] = lambda: ascendc(data, api, inputs, metadata=prepared)
        if operator != 'ascendc':
            calls['tla'] = lambda: kernel.forward(*data, core_limit=CORE_LIMIT)
        checks = {name: check(call(), expected) for name, call in calls.items()}
        torch.npu.synchronize()
        expected = None
        yield calls, checks
    finally:
        calls.clear()
        data = expected = None
        gc.collect()
        torch.npu.empty_cache()


def load_runtime():
    global torch, torch_npu
    torch = pytest.importorskip('torch')
    torch_npu = pytest.importorskip('torch_npu')
    return load_kernel(Path(__file__).with_name('mla_kernel.py'))


@pytest.fixture
def kernel():
    return load_runtime()


@pytest.fixture
def runtime(request, kernel):
    from utils.tla_test_utils import get_stages

    do_ir, do_compile, do_run = get_stages()
    name = request.node.originalname
    if not do_compile or (name != 'test_compile' and not do_run):
        pytest.skip('Requested TLA stage does not execute this test')
    if not torch.npu.is_available():
        pytest.skip('MLA compile/run requires an Ascend NPU')
    if name == 'test_perf':
        if not RUN_PERF:
            pytest.skip('Set RUN_PERF = True to benchmark')
        if hasattr(request.config, 'workerinput'):
            pytest.skip('Run performance tests without pytest-xdist')
    assert ACCURACY_OPERATOR in ('tla', 'ascendc', 'both')
    assert PERF_OPERATOR in ('tla', 'ascendc', 'both')
    assert PERF_ROUNDS > 0 and PERF_SAMPLES > 0
    return kernel


@pytest.mark.parametrize('heads', HEADS)
def test_ir(kernel, heads, tmp_path):
    offline_ir(kernel, heads, tmp_path / f'h{heads}')


@pytest.mark.parametrize('heads', HEADS)
def test_compile(runtime, heads):
    from utils.tla_test_utils import check_kernel_binary

    device = torch.device('npu', torch.npu.current_device())
    q = torch.empty((1, 4, heads, 576), dtype=torch.bfloat16, device=device)
    kv = torch.empty((1024, 128, 1, 576), dtype=q.dtype, device=device)
    table = torch.empty((1, 1024), dtype=torch.int32, device=device)
    compiled = runtime.forward(q, kv, table, core_limit=CORE_LIMIT, compile_only=True)
    assert len(compiled) == 2
    for stage in compiled:
        check_kernel_binary(stage)


@pytest.mark.parametrize('heads', HEADS)
def test_run(runtime, heads):
    for seed in SEEDS:
        for family in FAMILIES:
            with accuracy_case(runtime, heads, seed, family, ACCURACY_OPERATOR) as (_, checks):
                assert passed(checks), f'H={heads}, seed={seed}, {family}: {json.dumps(checks, indent=2)}'


@pytest.mark.parametrize('heads', HEADS)
def test_perf(runtime, heads):
    """Gate timings on precision and occupancy; retain every round and device task."""
    # Force this process to register a live allocation. A single observed PID
    # must then be ours, even when npu-smi reports host rather than container PIDs.
    allocation = torch.empty((1, ), device='npu')
    torch.npu.synchronize()
    initial = device_snapshot()
    assert initial['returncode'] == 0 and len(set(initial['pids'])) == 1, initial['text']
    PERF_OUTPUT.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f'h{heads}-', dir=PERF_OUTPUT))
    settings = dict(heads=heads, seeds=SEEDS, scale=SCALE, operator=PERF_OPERATOR, rounds=PERF_ROUNDS,
                    samples=PERF_SAMPLES, core_limit=CORE_LIMIT)
    save_environment(folder, runtime, settings)
    row = dict(settings=settings, profiles={}, passed=False)
    try:
        with accuracy_case(runtime, heads, SEEDS[0], 'normal', PERF_OPERATOR, include_prepared=True) as (calls, checks):
            row['checks'] = checks
            assert passed(checks), json.dumps(checks, indent=2)
            order = list(calls)
            for round_id in range(PERF_ROUNDS):
                offset = round_id % len(order)
                for name in order[offset:] + order[:offset]:
                    measured = profile(calls[name], folder / f'{name}-{round_id}.json', PERF_SAMPLES)
                    measured['method'] = 'sum_of_AI_Core_Vector_tasks; complete_call_wall_time_separate'
                    row['profiles'].setdefault(name, []).append(measured)
                    save_json(folder / 'results.json', row)
                    assert measured['occupancy_consistent'], 'Contended timing rejected'
                    assert set(measured['device_snapshots'][0]['pids']) == set(initial['pids'])
            if PERF_COUNTERS:
                row['pipe_counters'] = {name: pipe_counters(call, folder / name) for name, call in calls.items()}
            row['passed'] = True
    except Exception as exc:
        row['error'] = repr(exc)
        raise
    finally:
        save_json(folder / 'results.json', row)
        print(f'MLA results: {folder}')
        del allocation


def main():
    """Standalone entry; stage selection is shared with the other TLA tests."""
    global HEADS, ACCURACY_OPERATOR, PERF_OPERATOR, PERF_SAMPLES, PERF_ROUNDS, PERF_OUTPUT, CORE_LIMIT
    from utils.tla_test_utils import get_stages

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=int, nargs='+', choices=HEADS, default=HEADS)
    parser.add_argument('--operator', choices=('tla', 'ascendc', 'both'), default=ACCURACY_OPERATOR)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--samples', type=int, default=PERF_SAMPLES)
    parser.add_argument('--rounds', type=int, default=PERF_ROUNDS)
    parser.add_argument('--out', type=Path, default=PERF_OUTPUT)
    parser.add_argument('--core-limit', type=int)
    args = parser.parse_args()
    HEADS, ACCURACY_OPERATOR, PERF_OPERATOR = args.heads, args.operator, args.operator
    PERF_SAMPLES, PERF_ROUNDS, PERF_OUTPUT, CORE_LIMIT = args.samples, args.rounds, args.out, args.core_limit
    assert PERF_SAMPLES > 0 and PERF_ROUNDS > 0
    kernel = load_runtime()
    do_ir, do_compile, do_run = get_stages()
    assert not args.profile or do_run, 'Profiling requires the run stage'
    with tempfile.TemporaryDirectory(prefix='tla-mla-ir-') as tmp:
        for heads in HEADS:
            if do_ir:
                test_ir(kernel, heads, Path(tmp))
            if do_compile:
                test_compile(kernel, heads)
            if do_run:
                test_run(kernel, heads)
                if args.profile:
                    test_perf(kernel, heads)


if __name__ == '__main__':
    main()
