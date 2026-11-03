"""Shared NPU profiling, evidence and lazy-loading helpers for kernel tests."""
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def query_device():
    import torch

    logical = torch.npu.current_device()
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    physical = visible.split(',')[logical].strip() if visible else str(logical)
    return subprocess.run(
        ['npu-smi', 'info', '-t', 'proc-mem', '-i', physical],
        text=True,
        capture_output=True,
        timeout=20,
    )


def device_snapshot():
    proc = query_device()
    return dict(
        time=time.time(),
        returncode=proc.returncode,
        pids=re.findall(r'Process id:(\d+)', proc.stdout),
        text=proc.stdout + proc.stderr,
    )


def profile(call, path, samples):
    import torch
    import torch_npu

    for _ in range(5):
        call()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )
    snapshots = [device_snapshot()]
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=config,
    ) as profiler:
        for _ in range(samples):
            call()
        torch.npu.synchronize()
        snapshots.append(device_snapshot())
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
    assert tasks and len(tasks) % samples == 0, f'Missing/incomplete device events: {len(tasks)}'
    count = len(tasks) // samples
    names = [e['name'] for e in tasks[:count]]
    durations = []
    for i in range(0, len(tasks), count):
        group = tasks[i:i + count]
        assert [e['name'] for e in group] == names
        durations.append(sum(float(e['dur']) for e in group))
    # Also measure the complete public Python call, including allocation, runtime
    # queries, launch gaps and all device work. No profiler is active in this phase.
    wall_samples = []
    for _ in range(min(10, samples)):
        torch.npu.synchronize()
        started = time.perf_counter_ns()
        call()
        torch.npu.synchronize()
        wall_samples.append((time.perf_counter_ns() - started) / 1000)
    snapshots.append(device_snapshot())
    observed = [set(s['pids']) for s in snapshots]
    fairness_ok = (all(s['returncode'] == 0 for s in snapshots) and len(observed[0]) == 1
                   and all(p == observed[0] for p in observed))
    return dict(
        median_us=statistics.median(durations),
        samples_us=durations,
        wall_median_us=statistics.median(wall_samples),
        wall_samples_us=wall_samples,
        occupancy_consistent=fairness_ok,
        device_snapshots=snapshots,
        tasks_per_call=count,
        task_names=names,
        method='sum_of_all_device_tasks',
    )


def pipe_counters(call, folder):
    """Diagnostic overlapping pipe counters, separate from score timings."""
    import torch
    import torch_npu

    for _ in range(3):
        call()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(folder)),
    ):
        for _ in range(3):
            call()
        torch.npu.synchronize()
    for raw in folder.rglob('PROF_*'):
        if raw.is_dir():
            with (folder / 'msprof-export.log').open('a') as log:
                subprocess.run(
                    ['msprof', '--export=on', '--output=' + str(raw)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    files = list(folder.rglob('op_summary*.csv'))
    assert len(files) == 1, files
    with files[0].open() as stream:
        return list(csv.DictReader(stream))


def load_kernel(path):
    """Load a sibling operator without colliding with another kernel's module name."""
    path = Path(path).resolve()
    name = '_tla_kernel_' + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def save_environment(root, kernel, settings):
    import torch
    import torch_npu
    import triton

    distribution = importlib.metadata.distribution('triton-ascend')
    environment = dict(
        settings=settings,
        started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        physical_device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        core_counts=kernel.device_cores(torch.device(f'npu:{torch.npu.current_device()}')),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        wheel=json.loads(distribution.read_text('direct_url.json') or 'null'),
        operator_sha256=hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
    )
    save_json(root / 'environment.json', environment)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))


def passed(checks):
    return all(metric['passed'] for outputs in checks.values() for metric in outputs.values())
