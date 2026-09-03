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
import multiprocessing
import os
import warnings
from datetime import datetime, timezone
from typing import Optional

import triton.runtime as runtime
from triton.knobs import cache


class ProfilerResultMismatchError(RuntimeError):

    def __init__(self, target_kernel_name: str, expected_rows: int, actual_rows: int):
        self.target_kernel_name = target_kernel_name
        self.expected_rows = expected_rows
        self.actual_rows = actual_rows
        super().__init__(
            "Profiler rows filtered by target kernel name do not match the expected count. "
            f"target_kernel_name={target_kernel_name!r}, expected_rows={expected_rows}, actual_rows={actual_rows}")


class _MstxUnavailableError(RuntimeError):
    pass


_EVENT_BATCH_SIZE = 1024
_BENCH_MODE_ENV = "TRITON_NPU_BENCH_MODE"
_LIGHTWEIGHT_BENCH_MODES = ("event", "lightweight")
_PROFILER_BENCH_MODES = ("profiler", "level1")


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
    """Benchmark NPU callables while preserving the public API contract.

    MSTX device-range profiling is preferred by default, with Level1 as its
    compatibility fallback. Set ``TRITON_NPU_BENCH_MODE=event`` to explicitly
    request lightweight device-event timing, or ``level1`` to force the legacy
    profiler. Profiler-specific options take precedence over lightweight mode;
    incomplete profiler data falls back to event timing for the whole batch.
    """
    bench_mode = os.getenv(_BENCH_MODE_ENV, "profiler").strip().lower()
    valid_modes = _LIGHTWEIGHT_BENCH_MODES + _PROFILER_BENCH_MODES
    if bench_mode not in valid_modes:
        raise ValueError(f"{_BENCH_MODE_ENV} must be one of {valid_modes}, got {bench_mode!r}")

    profiler_options_supplied = prof_dir is not None or keep_res or target_kernel_name is not None
    if bench_mode in _LIGHTWEIGHT_BENCH_MODES and not profiler_options_supplied:
        return _do_bench_npu_event(funcs, warmup=warmup, active=active, clear_l2_cache=clear_l2_cache)

    try:
        precise_bench = _do_bench_npu_profiler if bench_mode == "level1" else _do_bench_npu_precise
        result = precise_bench(
            funcs,
            warmup=warmup,
            active=active,
            clear_l2_cache=clear_l2_cache,
            prof_dir=prof_dir,
            keep_res=keep_res,
            target_kernel_name=target_kernel_name,
        )
        values = result if isinstance(result, list) else [result]
        if all(value != float("inf") for value in values):
            return result
        warnings.warn(
            "NPU profiler did not produce usable kernel timing data; the operation may be a non-compute task "
            "such as MemcpyAsync. Falling back to device-event timing.",
            RuntimeWarning,
            stacklevel=2,
        )
    except ProfilerResultMismatchError as exc:
        warnings.warn(
            "NPU profiler did not produce a complete set of target kernel records; "
            "the operation may be a non-compute task such as MemcpyAsync. "
            f"Falling back to device-event timing. Details: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    return _do_bench_npu_event(funcs, warmup=warmup, active=active, clear_l2_cache=clear_l2_cache)


def _do_bench_npu_precise(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
    if target_kernel_name is not None:
        return _do_bench_npu_profiler(
            funcs,
            warmup=warmup,
            active=active,
            clear_l2_cache=clear_l2_cache,
            prof_dir=prof_dir,
            keep_res=keep_res,
            target_kernel_name=target_kernel_name,
        )

    try:
        return _do_bench_npu_mstx(
            funcs,
            warmup=warmup,
            active=active,
            clear_l2_cache=clear_l2_cache,
            prof_dir=prof_dir,
            keep_res=keep_res,
        )
    except (_MstxUnavailableError, ProfilerResultMismatchError) as exc:
        warnings.warn(
            f"MSTX device-range profiling is unavailable or incomplete; falling back to Level1 profiler. "
            f"Details: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _do_bench_npu_profiler(
            funcs,
            warmup=warmup,
            active=active,
            clear_l2_cache=clear_l2_cache,
            prof_dir=prof_dir,
            keep_res=keep_res,
        )


def _do_bench_npu_mstx(funcs, warmup=5, active=30, clear_l2_cache=False, prof_dir=None, keep_res=False):
    import torch
    import torch_npu

    if not isinstance(funcs, list):
        funcs = [funcs]
    if not funcs:
        raise ValueError("funcs must contain at least one callable")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if active <= 0:
        raise ValueError("active must be positive")

    try:
        mstx = torch_npu.npu.mstx
        stream = torch.npu.current_stream()
        export_type = torch_npu.profiler.ExportType.Db
        profiler_level = torch_npu.profiler.ProfilerLevel.Level_none
        schedule = torch_npu.profiler.schedule(wait=0, warmup=0, active=len(funcs) * active, repeat=1)
    except AttributeError as exc:
        raise _MstxUnavailableError("the installed torch_npu does not expose MSTX DB profiling") from exc

    config_kwargs = {
        "profiler_level": profiler_level,
        "data_simplification": False,
    }
    experimental_config = None
    last_config_error = None
    for marker_option in ("mstx", "msprof_tx"):
        for output_type in (export_type, [export_type]):
            try:
                experimental_config = torch_npu.profiler._ExperimentalConfig(
                    **config_kwargs,
                    export_type=output_type,
                    **{marker_option: True},
                )
                break
            except (TypeError, ValueError) as exc:
                last_config_error = exc
        if experimental_config is not None:
            break
    if experimental_config is None:
        raise _MstxUnavailableError(
            "the installed torch_npu cannot enable MSTX DB profiling"
        ) from last_config_error

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = cache.get_triton_dir("profile_results")
        torch_path = os.path.join(base_path, f"mstx_{timestamp}_{process.name}-{process.pid}")

    marker_prefix = f"triton_do_bench_{os.getpid()}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    driver = runtime.driver.active
    cache_buffer = driver.get_empty_cache_for_benchmark() if clear_l2_cache else None

    def run_once(fn):
        if cache_buffer is not None:
            driver.clear_cache(cache_buffer)
        fn()

    for fn in funcs:
        fn()
        torch.npu.synchronize()
        for _ in builtins.range(warmup):
            run_once(fn)
        torch.npu.synchronize()

    try:
        with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=schedule,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
        ) as prof:
            for func_idx, fn in enumerate(funcs):
                for active_idx in builtins.range(active):
                    if cache_buffer is not None:
                        driver.clear_cache(cache_buffer)
                    marker = f"{marker_prefix}/{func_idx}/{active_idx}"
                    range_id = mstx.range_start(marker, stream)
                    if not range_id:
                        raise _MstxUnavailableError("mstx.range_start failed")
                    try:
                        fn()
                    finally:
                        mstx.range_end(range_id)
                    torch.npu.synchronize()
                    prof.step()

        return _collect_mstx_result(torch_path, marker_prefix, len(funcs), active)
    finally:
        _rm_dic(keep_res, torch_path)


def _collect_mstx_result(base_dir: str, marker_prefix: str, num_funcs: int, num_active: int):
    import sqlite3
    from contextlib import closing

    duration_by_marker = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".db"):
                continue
            db_path = os.path.join(root, file)
            with closing(sqlite3.connect(db_path)) as connection:
                tables = {
                    row[0].upper()
                    for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                }
                if not {"MSTX_EVENTS", "STRING_IDS", "TASK"}.issubset(tables):
                    continue
                try:
                    rows = list(
                        connection.execute(
                            """
                            SELECT strings.value, MIN(tasks.startNs), MAX(tasks.endNs)
                            FROM MSTX_EVENTS AS events
                            JOIN STRING_IDS AS strings ON strings.id = events.message
                            JOIN TASK AS tasks ON tasks.connectionId = events.connectionId
                            GROUP BY strings.value, events.rangeId
                            """
                        )
                    )
                except sqlite3.DatabaseError:
                    continue
                for marker, start_ns, end_ns in rows:
                    if marker.startswith(f"{marker_prefix}/") and start_ns is not None and end_ns is not None:
                        duration_by_marker[marker] = (end_ns - start_ns) / 1e6

    expected_rows = num_funcs * num_active
    if len(duration_by_marker) != expected_rows:
        raise ProfilerResultMismatchError(marker_prefix, expected_rows, len(duration_by_marker))

    results = []
    for func_idx in builtins.range(num_funcs):
        durations = [
            duration_by_marker[f"{marker_prefix}/{func_idx}/{active_idx}"]
            for active_idx in builtins.range(num_active)
        ]
        results.append(sum(durations) / num_active)

    if num_funcs == 1:
        return results[0]
    return results


def _do_bench_npu_event(funcs, warmup=5, active=30, clear_l2_cache=False):
    if not isinstance(funcs, list):
        funcs = [funcs]
    if not funcs:
        raise ValueError("funcs must contain at least one callable")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if active <= 0:
        raise ValueError("active must be positive")

    driver = runtime.driver.active
    device_interface = driver.get_device_interface()
    cache_buffer = driver.get_empty_cache_for_benchmark() if clear_l2_cache else None
    results = []

    def run_once(fn):
        if cache_buffer is not None:
            driver.clear_cache(cache_buffer)
        fn()

    for fn in funcs:
        # Compile and initialize runtime state before applying the requested
        # warmup count, matching the historical do_bench_npu behaviour.
        fn()
        device_interface.synchronize()

        for _ in builtins.range(warmup):
            run_once(fn)
        device_interface.synchronize()

        elapsed_ms = 0.0
        measured = 0
        while measured < active:
            batch_size = min(_EVENT_BATCH_SIZE, active - measured)
            start_events = [device_interface.Event(enable_timing=True) for _ in builtins.range(batch_size)]
            end_events = [device_interface.Event(enable_timing=True) for _ in builtins.range(batch_size)]

            for start_event, end_event in zip(start_events, end_events):
                if cache_buffer is not None:
                    driver.clear_cache(cache_buffer)
                start_event.record()
                fn()
                end_event.record()

            device_interface.synchronize()
            elapsed_ms += sum(start.elapsed_time(end) for start, end in zip(start_events, end_events))
            measured += batch_size

        results.append(elapsed_ms / active)

    if len(results) == 1:
        return results[0]
    return results


def _do_bench_npu_profiler(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
    """Internal profiler benchmark used by autotune and profiling flows."""
    import torch
    import torch_npu

    if not isinstance(funcs, list):
        funcs = [funcs]

    # warmup kernel
    for fn in funcs:
        fn()
        torch.npu.synchronize()

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
):
    """
    Collect kernel performance from kernel_details.csv, returned in millisecond.
    The first `num_warmup` rows of each function are warmup data and will be ignored, the next `num_active` rows will be averaged.

    :param base_dir: the profiler path
    :type base_dir: str
    :param funcs: a list of Callable being profiled
    :type funcs: List[Callable]
    :param num_warmup: warmup count in kernel_details.csv of each fn
    :type num_warmup: int
    :param num_active: active count in kernel_details.csv of each fn
    :type num_active: int
    :param target_kernel_name: target triton kernel name reported by profiler
    :type target_kernel_name: Optional[str]
    """

    import numpy as np
    import pandas as pd

    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)
    if kernel_details_file is None:
        if num_funcs == 1:
            return float("inf")
        else:
            return [float("inf")] * num_funcs

    df = pd.read_csv(kernel_details_file)
    # filter out l2 cache clearing operation
    filter_cond = (not clear_l2_cache) | ~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)
    filter_df = df[filter_cond]
    if target_kernel_name is not None:
        filter_df = filter_df[filter_df["Name"] == target_kernel_name]

    expected_rows = num_funcs * (num_warmup + num_active)
    actual_rows = len(filter_df)
    if target_kernel_name is not None and actual_rows != expected_rows:
        raise ProfilerResultMismatchError(target_kernel_name, expected_rows, actual_rows)

    time_cost = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, num_active):
            row_index = func_idx * (num_warmup + num_active) + num_warmup + active_index
            time_cost[func_idx] += filter_df.iloc[row_index]["Duration(us)"]
    time_cost = [x / num_active / 1e3 for x in time_cost]

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost
