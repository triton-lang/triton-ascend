"""Shape-matched Paged MLA validation and paired device/wall benchmarks.

Every implementation receives the same tensors and page table. Metadata is
prepared once for each implementation. Device timings sum all launched kernels.
The complete Python call, including allocations, is timed separately.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import torch
import torch_npu

HERE = Path(__file__).resolve().parent
FAMILIES = (('normal', 1103), ('normal', 1104), ('uniform', 1103), ('peaked', 1103))
CASES = {'full': (143360, ) * 8, 'ragged': (4, 129, 260, 1031)}
NATIVE_COMMIT = '897efb917ea657c759c4f7ce79a162e4729ebdd4'
DSL_COMMIT = '6e5bef20c29aad8bc74c66ae995aeeb01a4789f4'


def load(path):
    path = Path(path).resolve()
    name = '_mla_' + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(lengths, family, seed):
    """Use random physical pages, including unused slots in shorter requests."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    page_counts = [(length + 127) // 128 for length in lengths]
    ids = torch.randperm(sum(page_counts), generator=generator, dtype=torch.int32)
    table = torch.full((len(lengths), max(page_counts)), -1, dtype=torch.int32)
    for row, pages in enumerate(ids.split(page_counts)):
        table[row, :pages.numel()] = pages
    device = torch.device('npu', torch.npu.current_device())

    def random(shape, query=False):
        if query and family == 'uniform':
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        data = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
        if query and family == 'peaked':
            data.mul_(8)
        return data.to(device)

    # Retain the input generation order of the pre-cleanup benchmark.
    q = random((len(lengths), 4, 24, 512), query=True)
    qr = random((len(lengths), 4, 24, 64), query=True)
    kv = random((ids.numel(), 128, 512))
    kr = random((ids.numel(), 128, 64))
    packed_q = torch.cat((q, qr), dim=-1)
    packed_kv = torch.cat((kv, kr), dim=-1)
    return packed_q, packed_kv, table.to(device), torch.tensor(lengths, dtype=torch.int32, device=device)


def reference(inputs):
    """Independent FP32 attention with one full softmax per query row."""
    q, kv, table, lengths = inputs
    batch, sq, heads, _ = q.shape
    table_cpu, lengths_cpu = table.cpu(), lengths.cpu().tolist()
    outputs, lses = [], []
    for request, length in enumerate(lengths_cpu):
        pages = table_cpu[request, :(length + 127) // 128].to(q.device, torch.int64)
        keys = kv.index_select(0, pages).reshape(-1, 576)[:length].float()
        query = q[request].reshape(-1, 576).float()
        positions = torch.arange(length, device=q.device)
        visible_end = length - sq + torch.arange(sq, device=q.device).repeat_interleave(heads)
        values, normalizers = [], []
        for start in range(0, sq * heads, 16):
            rows = slice(start, start + 16)
            logits = (query[rows, :512] @ keys[:, :512].T + query[rows, 512:] @ keys[:, 512:].T) * 0.04
            logits.masked_fill_(positions[None, :] > visible_end[rows, None], -float('inf'))
            values.append(logits.softmax(-1) @ keys[:, :512])
            normalizers.append(logits.logsumexp(-1))
        outputs.append(torch.cat(values).reshape(sq, heads, 512))
        lses.append(torch.cat(normalizers).reshape(sq, heads))
    return torch.stack(outputs), torch.stack(lses)


def precision(call, expected):
    actual = call()
    torch.npu.synchronize()
    assert isinstance(actual, tuple) and len(actual) == 2, 'Expected both output and LSE'
    result = {}
    for name, output, target, dtype, tolerance in zip(('output', 'lse'), actual, expected,
                                                      (torch.bfloat16, torch.float32), (0.005, 0.0001)):
        assert output.shape == target.shape and output.dtype == dtype, name
        assert torch.isfinite(output).all(), name
        error = output.float() - target
        maximum = (error.abs().amax() / target.abs().amax().clamp_min(1e-30)).item()
        rms = (error.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1e-30)).item()
        assert maximum <= tolerance and rms <= tolerance, (name, maximum, rms)
        result[name] = dict(normalized_max_error=maximum, relative_rms=rms)
    return result


def prepare_ascendc(inputs):
    import cann_ops_transformer_custom as native
    from cann_ops_transformer_custom import ops

    q, kv, table, lengths = inputs
    used_q = torch.full_like(lengths, q.shape[1])
    mask = torch.ones((2048, 2048), dtype=torch.int8, device=q.device).triu_(1)
    metadata = ops.flash_mla_metadata(
        lengths,
        q.shape[2],
        1,
        seqused_q=used_q,
        head_dim_qk=576,
        head_dim_v=512,
        mask_mode=3,
        layout_q='BSND',
    )
    native_kv = kv.unsqueeze(2)

    def call():
        output, lse = ops.flash_mla(
            q,
            native_kv,
            block_table=table,
            cache_seqlens=lengths,
            seqused_q=used_q,
            attn_mask=mask,
            metadata=metadata,
            head_dim_v=512,
            softmax_scale=0.04,
            mask_mode=3,
            layout_q='BSND',
            layout_kv='PA_BBND',
            layout_out='BSND',
            return_softmax_lse=True,
        )
        return output, lse.transpose(1, 2)

    call.metadata = dict(
        module=native.__file__,
        expected_source_commit=NATIVE_COMMIT,
        vendor=os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
        prepared_metadata=True,
    )
    return call


def install_dsl_sample(source, destination):
    """Extend the sample's H24 allowlist in a recorded, isolated copy."""
    source = Path(source).resolve()
    destination = Path(destination)
    shutil.copytree(source, destination)
    checker = destination / 'checker.py'
    original = checker.read_text()
    before = 'SUPPORTED_HEAD_COUNTS = (64, 96)'
    after = 'SUPPORTED_HEAD_COUNTS = (24, 64, 96)'
    if original.count(before) != 1:
        raise ValueError('Expected the pinned upstream DSL checker with the H64/H96 allowlist')
    checker.write_text(original.replace(before, after))
    sys.path.insert(0, str(destination))
    revision = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return dict(
        source=str(source),
        expected_source_commit=DSL_COMMIT,
        actual_source_commit=revision.stdout.strip() if revision.returncode == 0 else None,
        h24_allowlist_extended=True,
        checker_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        checker_adapted_sha256=hashlib.sha256(checker.read_bytes()).hexdigest(),
        file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source.glob('*.py')},
    )


def prepare_cannbot(inputs):
    import cannbotdsl
    import flash_mla_with_kvcache as sample
    from flash_mla_with_kvcache_metadata import flash_mla_with_kvcache_metadata

    q, kv, table, lengths = inputs
    batch, sq, heads, width = q.shape
    query = q.view(batch * sq, heads, width)
    native_kv = kv.unsqueeze(2)
    offsets = torch.arange(0, (batch + 1) * sq, sq, dtype=torch.int32, device=q.device)
    used_q = torch.full_like(lengths, sq)
    mask = torch.ones((2048, 2048), dtype=torch.int8, device=q.device).triu_(1)
    metadata = flash_mla_with_kvcache_metadata(
        lengths,
        heads,
        1,
        cu_seqlens_q=offsets,
        seqused_q=used_q,
        head_dim_qk=576,
        head_dim_v=512,
        mask_mode=3,
        layout_q='TND',
    )

    def call():
        output, lse = sample.flash_mla_with_kvcache(
            query,
            native_kv,
            block_table=table,
            cache_seqlens=lengths,
            cu_seqlens_q=offsets,
            seqused_q=used_q,
            attn_mask=mask,
            metadata=metadata,
            head_dim_v=512,
            softmax_scale=0.04,
            mask_mode=3,
            layout_q='TND',
            layout_kv='PA_BBND',
            layout_out='NTD',
            return_softmax_lse=True,
        )
        return output.transpose(0, 1).view(batch, sq, heads, 512), lse.transpose(0, 1).view(batch, sq, heads)

    call.metadata = dict(
        module=sample.__file__,
        dsl_version=getattr(cannbotdsl, '__version__', None),
        expected_source_commit=DSL_COMMIT,
        h24_allowlist_extended=True,
        prepared_metadata=True,
    )
    return call


def snapshot():
    logical = torch.npu.current_device()
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    physical = visible.split(',')[logical].strip() if visible else str(logical)
    proc = subprocess.run(['npu-smi', 'info', '-t', 'proc-mem', '-i', physical], capture_output=True, text=True,
                          timeout=20)
    return dict(
        returncode=proc.returncode,
        pids=re.findall(r'Process id:(\d+)', proc.stdout),
        physical_device=physical,
        text=proc.stdout + proc.stderr,
    )


def wait_exclusive():
    for _ in range(90):
        state = snapshot()
        if state['returncode'] == 0 and set(state['pids']) == {str(os.getpid())}:
            return state
        time.sleep(2)
    raise RuntimeError('No exclusive device window within 180 seconds')


def profile(call, path, samples):
    for _ in range(5):
        call()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )
    snapshots = [snapshot()]
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=config,
    ) as profiler:
        for _ in range(samples):
            call()
        torch.npu.synchronize()
        snapshots.append(snapshot())
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())
    if isinstance(events, dict):
        events = events['traceEvents']
    tasks = sorted(
        (e for e in events
         if e.get('ph') == 'X' and str(e.get('args', {}).get('Task Type', '')).startswith(('AI_CORE', 'AI_VECTOR',
                                                                                           'MIX_', 'KERNEL'))),
        key=lambda e: e['ts'],
    )
    assert tasks and len(tasks) % samples == 0, 'Missing or incomplete device events'
    tasks_per_call = len(tasks) // samples
    names = [e['name'] for e in tasks[:tasks_per_call]]
    durations = []
    for start in range(0, len(tasks), tasks_per_call):
        group = tasks[start:start + tasks_per_call]
        assert [e['name'] for e in group] == names
        durations.append(sum(float(e['dur']) for e in group))
    wall = []
    for _ in range(min(samples, 10)):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        call()
        torch.npu.synchronize()
        wall.append((time.perf_counter_ns() - start) / 1000)
    snapshots.append(snapshot())
    consistent = all(s['returncode'] == 0 and set(s['pids']) == {str(os.getpid())} for s in snapshots)
    return dict(
        samples_us=durations,
        wall_samples_us=wall,
        median_us=statistics.median(durations),
        wall_median_us=statistics.median(wall),
        occupancy_consistent=consistent,
        snapshots=snapshots,
        tasks_per_call=tasks_per_call,
        task_names=names,
    )


def checked_profile(call, path, samples):
    for attempt in range(8):
        wait_exclusive()
        result = profile(call, path, samples)
        if result['occupancy_consistent']:
            print('ACCEPTED', path.stem, result['median_us'], flush=True)
            return result
        rejected = path.with_name(f'{path.stem}.rejected-{attempt}.json')
        path.replace(rejected)
        rejected.with_suffix('.stats.json').write_text(json.dumps(result, indent=2))
        print('REJECTED_OCCUPANCY', path.stem, flush=True)
    raise RuntimeError('Device remained shared during repeated measurements')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines', default='ascendc,cannbot')
    parser.add_argument('--dsl-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    baselines = [s for s in args.baselines.split(',') if s]
    if set(baselines) - {'ascendc', 'cannbot'} or args.rounds < 1 or args.samples < 1:
        parser.error('Use ascendc,cannbot baselines and positive rounds/samples')
    if 'cannbot' in baselines and args.dsl_source is None:
        parser.error('--dsl-source must name the pinned upstream sample directory')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    sentinel = torch.empty(1, device='npu')
    torch.npu.synchronize()
    module = load(HERE / 'mla_kernel.py')
    from triton.backends.ascend.utils import _get_npucompiler_path

    compiler = Path(_get_npucompiler_path()[0])
    import triton

    report = dict(
        phase='precision',
        specification=dict(
            bs=8,
            num_head=24,
            kv_head=1,
            head_dim_qk=576,
            head_dim_v=512,
            q_len=4,
            kv_len=143360,
            page=128,
            scale=0.04,
            dtype='bf16',
            output_dtype='bf16',
            lse_dtype='fp32',
            causal='bottom-right',
        ),
        entry_snapshot=wait_exclusive(),
        source_sha256=hashlib.sha256((HERE / 'mla_kernel.py').read_bytes()).hexdigest(),
        compiler=str(compiler),
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
        runtime=dict(
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            triton_module=triton.__file__,
            cann=os.environ.get('ASCEND_HOME_PATH'),
            dsl_bisheng=os.environ.get('CANNBOTDSL_BISHENG_PATH'),
        ),
        metaop_o2=True,
        precision={},
        formal={},
    )
    if 'cannbot' in baselines:
        report['dsl_sample'] = install_dsl_sample(args.dsl_source, args.output / 'dsl-sample')
    result_path = args.output / 'results.json'

    def save():
        result_path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    for case, lengths in CASES.items():
        for family, seed in FAMILIES:
            wait_exclusive()
            inputs = make_inputs(lengths, family, seed)
            expected = reference(inputs)
            calls = {'tla': module.prepare_call(*inputs, split_pages=2 if case == 'ragged' else None)}
            if case == 'full':
                if 'ascendc' in baselines:
                    calls['ascendc'] = prepare_ascendc(inputs)
                if 'cannbot' in baselines:
                    calls['cannbot'] = prepare_cannbot(inputs)
            for name, call in calls.items():
                wait_exclusive()
                report['precision'].setdefault(
                    name, {})[f'{case}/{family}/{seed}'] = [precision(call, expected) for _ in range(3)]
                save()
                print('PRECISION_PASS', name, case, family, seed, flush=True)
            del calls, inputs, expected
    if not args.check_only:
        inputs = make_inputs(CASES['full'], 'normal', 1103)
        expected = reference(inputs)
        calls = {'tla': module.prepare_call(*inputs)}
        if 'ascendc' in baselines:
            calls['ascendc'] = prepare_ascendc(inputs)
        if 'cannbot' in baselines:
            calls['cannbot'] = prepare_cannbot(inputs)
        report['options'] = {name: call.metadata for name, call in calls.items()}
        report['performance_precheck'] = {name: precision(call, expected) for name, call in calls.items()}
        del expected
        names = list(calls)
        report['phase'] = 'performance'
        for round_id in range(args.rounds):
            offset = round_id % len(names)
            for name in names[offset:] + names[:offset]:
                path = args.output / f'round-{round_id}-{name}.trace.json'
                result = checked_profile(calls[name], path, args.samples)
                report['formal'].setdefault(name, []).append(result)
                save()
        report['medians'] = {
            name:
            dict(
                device_us=statistics.median([v
                                             for r in runs
                                             for v in r['samples_us']]),
                wall_us=statistics.median([v
                                           for r in runs
                                           for v in r['wall_samples_us']]),
            )
            for name, runs in report['formal'].items()
        }
        if 'ascendc' in calls:
            baseline = report['medians']['ascendc']['device_us']
            report['pct_of_ascendc'] = {
                name: 100 * baseline / value['device_us']
                for name, value in report['medians'].items()
            }
    report['phase'] = 'complete'
    save()
    print(json.dumps(report.get('medians', {}), indent=2), flush=True)


if __name__ == '__main__':
    main()
