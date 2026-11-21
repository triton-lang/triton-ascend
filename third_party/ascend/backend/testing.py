# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import builtins
import math
import multiprocessing
import os
from datetime import datetime, timezone
from typing import Optional
import fnmatch

import triton.runtime as runtime
from triton import knobs
from triton.knobs import cache


class ProfilerResultMismatchError(RuntimeError):

    def __init__(self, target_kernel_name: str, expected_rows: int, actual_rows: int):
        self.target_kernel_name = target_kernel_name
        self.expected_rows = expected_rows
        self.actual_rows = actual_rows
        super().__init__(
            "Profiler rows filtered by target kernel name do not match the expected count. "
            f"target_kernel_name={target_kernel_name!r}, expected_rows={expected_rows}, actual_rows={actual_rows}")


def _npu_bench_options(warmup=5, active=30):
    if active <= 0:
        raise ValueError("active must be positive")
    return (max(1, math.ceil(active / 10)), active, 4, 0.015, 0.25, 0.75, 3.0, 5)


def do_bench_npu_profiler(
        funcs,
        warmup=5,
        active=30,
        clear_l2_cache=False,
        prof_dir=None,
        keep_res=False,
        target_kernel_name: Optional[str] = None,
        _raise_on_mismatch: bool = False,
        quantiles=(0, 1),
):
    import torch
    import torch_npu
    if not isinstance(funcs, list):
        funcs = [funcs]

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = cache.get_triton_dir("profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")

    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()  # to avoid type cast
        buffer.sum()
        torch.npu.synchronize()  # shake out of any npu error

    total = warmup + active
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
    ) as prof:
        for fn in funcs:
            for _ in builtins.range(total):
                if clear_l2_cache:
                    buffer.sum()  # use buffer read to clear l2 cache
                    torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
    if clear_l2_cache:
        del buffer

    try:
        return _collect_prof_result(
            torch_path,
            funcs,
            warmup,
            active,
            target_kernel_name=target_kernel_name,
            clear_l2_cache=clear_l2_cache,
            _raise_on_mismatch=_raise_on_mismatch,
            quantiles=quantiles,
        )
    finally:
        _rm_dic(keep_res, torch_path)


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _collect_prof_result(
        base_dir: str,
        funcs,
        num_warmup: int,
        num_active: int,
        target_kernel_name: Optional[str] = None,
        clear_l2_cache: bool = False,
        _raise_on_mismatch: bool = False,
        quantiles=(0, 1),
):
    """
    Collect kernel performance from task_time*.csv or kernel_details.csv, returned in millisecond.
    Uses task_time*.csv by default. If target_kernel_name is provided, scans for kernel_details.csv as a fallback for accuracy.
    The first `num_warmup` rows of each function are warmup data and will be ignored, the next `num_active` rows will be averaged.

    :param base_dir: the profiler path
    :type base_dir: str
    :param funcs: a list of Callable being profiled
    :type funcs: List[Callable]
    :param num_warmup: warmup count in task_time*.csv or kernel_details.csv of each fn
    :type num_warmup: int
    :param num_active: active count in task_time*.csv or kernel_details.csv of each fn
    :type num_active: int
    :param target_kernel_name: target triton kernel name reported by profiler
    :type target_kernel_name: Optional[str]
    """

    import numpy as np
    import pandas as pd
    use_task_time = (target_kernel_name is None)
    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if use_task_time and fnmatch.fnmatch(file, "task_time*.csv"):
                kernel_details_file = os.path.join(root, file)
                break
            elif not use_task_time and file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)

    def _error(msg: str):
        print(f"[Error] {msg}")
        if num_funcs == 1:
            return float("inf")
        return [float("inf")] * num_funcs

    if kernel_details_file is None:
        return _error(
            f"No profiling data found under {base_dir}. The profiler may have failed to collect device tasks or stopped abnormally."
        )

    df = pd.read_csv(kernel_details_file, keep_default_na=False)
    if use_task_time:
        # The first and last lines of the task_time*.csv file are PROFILING_DISABLE, which should be deleted.
        df = df[1:-1]

    if df.empty:
        print("[WARNING] No profiling data on the device side.")
        if num_funcs == 1:
            return float("0")
        return [float("0")] * num_funcs

    if use_task_time:
        col_time = "task_time(us)"
        filter_cond = (not clear_l2_cache) | ~df["kernel_name"].str.contains(r"^ReduceSum", case=False, na=False)
    else:
        col_time = "Duration(us)"
        filter_cond = (not clear_l2_cache) | ~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)
    # filter out l2 cache clearing operation
    filter_df = df[filter_cond]
    if target_kernel_name is not None:
        filter_df = filter_df[filter_df["Name"] == target_kernel_name]

    expected_rows = num_funcs * (num_warmup + num_active)
    actual_rows = len(filter_df)

    if (target_kernel_name is not None and _raise_on_mismatch and actual_rows != expected_rows
            and not (all(fn is funcs[0] for fn in funcs) and actual_rows % expected_rows == 0)):
        raise ProfilerResultMismatchError(target_kernel_name, expected_rows, actual_rows)
    mul = 1
    if num_funcs == 1 or all(fn is funcs[0] for fn in funcs):
        if actual_rows % expected_rows != 0:
            if target_kernel_name is not None and _raise_on_mismatch:
                raise ProfilerResultMismatchError(target_kernel_name, expected_rows, actual_rows)
            return _error(
                f"Expected the actual row count to be a multiple of {expected_rows}, but got {actual_rows}. The expected row count and the actual row count do not match."
            )
        mul = actual_rows // expected_rows
    else:
        if actual_rows != expected_rows:
            print(
                "[WARNING] Passing a list of functions where a function may contain multiple kernels or launch no kernel is not supported and may lead to incorrect results."
            )
            return [float("inf")] * num_funcs

    if mul < 1:
        return float("inf") if num_funcs == 1 else [float("inf")] * num_funcs
    samples = filter_df[col_time].to_numpy(dtype=float).reshape(num_funcs, num_warmup + num_active, mul)[:, num_warmup:]
    valid = np.isfinite(samples).all(axis=(1, 2)) & (samples > 0).all(axis=(1, 2))
    samples = samples.sum(axis=2) / 1e3
    if num_active >= 10 and quantiles != (0, 1):
        start, end = (int(num_active * quantile) for quantile in quantiles)
        samples = np.sort(samples, axis=1)[:, start:max(start + 1, end)]
    time_cost = np.where(valid, samples.mean(axis=1), np.inf).tolist()

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost


try:
    from mspti import KernelMonitor
except ImportError:
    KernelMonitor = None

try:
    from triton.backends.ascend.utils import is_cann_version_at_least
    CANN_VERSION_AVAILABLE = True
except ImportError:
    CANN_VERSION_AVAILABLE = False
    print("[WARNING] triton.backends.ascend.utils not found. CANN version check skipped.")


# If the CANN version is earlier than 9.1.0, it needs to set libmspti.so in LD_PRELOAD to use mspti.
def do_bench_npu_mspti(
        funcs,
        warmup=5,
        active=30,
        clear_l2_cache=False,
        target_kernel_name: Optional[str] = None,
        quantiles=(0, 1),
):
    import torch
    import torch_npu
    if not isinstance(funcs, list):
        funcs = [funcs]

    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()  # to avoid type cast
        buffer.sum()
        torch.npu.synchronize()  # shake out of any npu error
    else:
        buffer = None

    all_kernel_durations = []

    def callback(data):
        if clear_l2_cache and ('reducesum' in data.name.lower()):
            return
        if target_kernel_name is not None and target_kernel_name not in data.name:
            return
        all_kernel_durations.append((data.start, data.end - data.start))

    monitor = KernelMonitor()
    torch.npu.synchronize()

    monitor.start(callback)

    try:
        total = warmup + active
        for fn in funcs:
            for _ in builtins.range(total):
                if clear_l2_cache:
                    buffer.sum()
                    torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
    finally:
        torch.npu.synchronize()
        monitor.stop()

    all_kernel_durations.sort(key=lambda row: row[0])
    all_kernel_durations = [duration for _, duration in all_kernel_durations]

    num_funcs = len(funcs)
    duration_per_kernel = []

    expected_rows = num_funcs * total
    actual_rows = len(all_kernel_durations)
    if actual_rows < expected_rows:
        if num_funcs == 1:
            return float("inf")
        return [float("inf")] * num_funcs

    mul = 1
    if num_funcs == 1 or all(fn is funcs[0] for fn in funcs):
        if actual_rows % expected_rows != 0:
            return float("inf") if num_funcs == 1 else [float("inf")] * num_funcs
        mul = actual_rows // expected_rows
        warmup = warmup * mul
        total = actual_rows // num_funcs
    else:
        if actual_rows != expected_rows:
            print(
                "[WARNING] Passing a list of functions containing multiple kernels is not fully supported and may lead to inaccurate results."
            )
            return [float("inf")] * num_funcs

    current_idx = 0
    for i in range(num_funcs):
        current_func_records = all_kernel_durations[current_idx:current_idx + total]
        current_idx += total
        current_active_records = current_func_records[warmup:total]
        if mul > 1:
            current_active_records = [
                sum(current_active_records[index:index + mul]) for index in range(0, len(current_active_records), mul)
            ]
        if active >= 10 and quantiles != (0, 1):
            start, end = (int(active * quantile) for quantile in quantiles)
            current_active_records = sorted(current_active_records)[start:max(start + 1, end)]
        avg_time = sum(current_active_records) / len(current_active_records)
        avg_time_ms = avg_time / 1000000.0
        duration_per_kernel.append(avg_time_ms)

    if num_funcs == 1:
        return duration_per_kernel[0]
    else:
        return duration_per_kernel


def _check_mspti_env():
    if CANN_VERSION_AVAILABLE:
        if not is_cann_version_at_least(9, 1, 0):
            if 'libmspti.so' not in os.getenv('LD_PRELOAD', ''):
                print("[WARNING] libmspti.so not set in LD_PRELOAD, please set libmspti.so in LD_PRELOAD to use mspti.")


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
    _active_limit=None,
):
    """
    Benchmark the runtime of the provided function on NPU.
    this function utilizes NPU profiling tools (mspti or torch_npu.profiler) to capture pure Device-side kernel execution time.
    By default, it returns the mean runtime in milliseconds of the provided function(s) based on `active` iterations.

    :param funcs: Function (or list of functions) to benchmark. If a list is provided, returns a list of mean runtimes.
    :type funcs: Callable or List[Callable]
    :param warmup: Warmup iterations. Runs the function before actual timing to stabilize performance.
        The argument is retained for compatibility; the effective count is max(ceil(active / 10), 1).
    :type warmup: int
    :param active: Active iterations to record for timing. The final result is the average over these iterations. Defaults to 30.
    :type active: int
    :param clear_l2_cache: Whether to clear the L2 cache before each function execution. Defaults to False.
    :type clear_l2_cache: bool, optional
    :param prof_dir: Directory to save profiler results. If None, defaults to a temporary directory under triton cache. If specified, forces fallback to torch_npu.profiler.
    :type prof_dir: str, optional
    :param keep_res: Whether to keep the raw profiler result files (e.g., CSVs) after parsing. Defaults to False. If True, forces fallback to torch_npu.profiler.
    :type keep_res: bool, optional
    :param target_kernel_name: Specific NPU kernel name to filter and benchmark. If None, benchmarks the entire function. If specified, returns the execution time of only that kernel. Defaults to None.
        Both profilers support filtering; specifying a target outside the NPU autotune route forces torch_npu.profiler.
    :type target_kernel_name: str, optional
    """
    import math
    import os
    import torch
    mspti_available = True
    if KernelMonitor is None:
        mspti_available = False
        print(f"[WARNING] mspti package not found. Falling back to torch_npu.profiler.")
    _check_mspti_env()

    warmup, active, confirmations, tolerance, lower, upper, budget, batches = _npu_bench_options(warmup, active)
    if _active_limit is not None:
        active = min(active, _active_limit)
        budget = 0
    if not isinstance(funcs, list):
        funcs = [funcs]
        use_autotune = False
    else:
        use_autotune = knobs.ascend.bench_method.lower() == "npu"

    for fn in funcs:
        fn()
        torch.npu.synchronize()

    force_fallback = (prof_dir is not None) or keep_res or (target_kernel_name is not None and not use_autotune)
    base_dir = prof_dir
    if batches > 1 and base_dir is None:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(cache.get_triton_dir("profile_results"), f"prof_{timestamp}_{os.getpid()}")
    histories = [[] for _ in funcs]
    pending = list(builtins.range(len(funcs)))
    for batch in builtins.range(max(1, batches)):
        warmup = max(1, math.ceil(active / 10))
        selected = [funcs[index] for index in pending] * confirmations
        quantiles = (lower, upper)
        results, need_fallback = None, True
        if mspti_available and not force_fallback:
            try:
                results = do_bench_npu_mspti(selected, warmup, active, clear_l2_cache, target_kernel_name, quantiles)
                values = results if isinstance(results, list) else [results]
                need_fallback = not all(math.isfinite(value) and value > 0 for value in values)
            except Exception:
                pass
        if need_fallback:
            batch_dir = os.path.join(base_dir, f"batch_{batch}") if batches > 1 else prof_dir
            results = do_bench_npu_profiler(selected, warmup, active, clear_l2_cache, batch_dir, keep_res,
                                            target_kernel_name, use_autotune, quantiles)
        if len(selected) == 1:
            results = [results]
        for position, index in enumerate(pending):
            histories[index] = sorted(histories[index] + results[position::len(pending)])[:confirmations]
        pending = [
            index for index in pending
            if confirmations > 1 and (len(histories[index]) < confirmations or not math.isfinite(histories[index][-1])
                                      or histories[index][0] <= 0 or histories[index][-1] > histories[index][0] *
                                      (1 + tolerance))
        ]
        if not pending:
            break
        if batch == 0 and batches > 1 and budget and 0 < min(results) < math.inf:
            previous_active = active
            active = max(active, math.ceil(budget / min(results)))
            if active != previous_active:
                for index in pending:
                    histories[index].clear()
    if batches > 1:
        _rm_dic(keep_res, base_dir)
    if pending:
        print(f"NPU benchmark: minimum not confirmed for configurations {[index + 1 for index in pending]}")
    scores = [history[0] for history in histories]
    return scores[0] if len(funcs) == 1 else scores
