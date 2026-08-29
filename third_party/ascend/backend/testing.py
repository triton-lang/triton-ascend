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
import time
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


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
    diagnostic_callback=None,
    diagnostic_labels=None,
):
    import torch
    import torch_npu

    if not isinstance(funcs, list):
        funcs = [funcs]
    if diagnostic_labels is not None and len(diagnostic_labels) != len(funcs):
        raise ValueError(f"Expected one diagnostic label per benchmark function, got {len(diagnostic_labels)} labels "
                         f"for {len(funcs)} functions")

    collect_diagnostics = diagnostic_callback is not None
    benchmark_start = time.perf_counter() if collect_diagnostics else None
    config_metrics = None
    if collect_diagnostics:
        config_metrics = [{
            "prewarm_enqueue_ms": 0.0,
            "prewarm_synchronize_ms": 0.0,
            "profile_enqueue_ms": 0.0,
            "profile_synchronize_ms": 0.0,
            "profile_l2_clear_ms": 0.0,
            "profile_wall_ms": 0.0,
        } for _ in funcs]

    # warmup kernel
    if collect_diagnostics:
        for func_idx, fn in enumerate(funcs):
            enqueue_start = time.perf_counter()
            fn()
            enqueue_end = time.perf_counter()
            torch.npu.synchronize()
            synchronize_end = time.perf_counter()
            config_metrics[func_idx]["prewarm_enqueue_ms"] = (enqueue_end - enqueue_start) * 1e3
            config_metrics[func_idx]["prewarm_synchronize_ms"] = (synchronize_end - enqueue_end) * 1e3
    else:
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
    profile_create_start = time.perf_counter() if collect_diagnostics else None
    npu_profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )
    if collect_diagnostics:
        profile_create_ms = (time.perf_counter() - profile_create_start) * 1e3
        profile_enter_start = time.perf_counter()
        with npu_profiler:
            profile_enter_ms = (time.perf_counter() - profile_enter_start) * 1e3
            profile_loop_start = time.perf_counter()
            for func_idx, fn in enumerate(funcs):
                config_start = time.perf_counter()
                for _ in builtins.range(total):
                    if clear_l2_cache:
                        l2_clear_start = time.perf_counter()
                        buffer.sum()  # use buffer read to clear l2 cache
                        torch.npu.synchronize()
                        config_metrics[func_idx]["profile_l2_clear_ms"] += (time.perf_counter() - l2_clear_start) * 1e3
                    enqueue_start = time.perf_counter()
                    fn()
                    enqueue_end = time.perf_counter()
                    torch.npu.synchronize()
                    synchronize_end = time.perf_counter()
                    config_metrics[func_idx]["profile_enqueue_ms"] += (enqueue_end - enqueue_start) * 1e3
                    config_metrics[func_idx]["profile_synchronize_ms"] += (synchronize_end - enqueue_end) * 1e3
                config_metrics[func_idx]["profile_wall_ms"] = (time.perf_counter() - config_start) * 1e3
            profile_loop_ms = (time.perf_counter() - profile_loop_start) * 1e3
            profile_finalize_start = time.perf_counter()
        profile_finalize_ms = (time.perf_counter() - profile_finalize_start) * 1e3
    else:
        with npu_profiler:
            for fn in funcs:
                for _ in builtins.range(total):
                    if clear_l2_cache:
                        buffer.sum()  # use buffer read to clear l2 cache
                        torch.npu.synchronize()
                    fn()
                    torch.npu.synchronize()
    if clear_l2_cache:
        del buffer

    collect_start = time.perf_counter() if collect_diagnostics else None
    time_cost = None
    profiled_device_total_ms = None
    collect_error = None
    try:
        if collect_diagnostics:
            time_cost, profiled_device_total_ms = _collect_prof_result(
                torch_path,
                funcs,
                warmup,
                active,
                target_kernel_name=target_kernel_name,
                clear_l2_cache=clear_l2_cache,
                return_diagnostics=True,
            )
        else:
            time_cost = _collect_prof_result(
                torch_path,
                funcs,
                warmup,
                active,
                target_kernel_name=target_kernel_name,
                clear_l2_cache=clear_l2_cache,
            )
    except Exception as exc:
        collect_error = type(exc).__name__
        raise
    finally:
        collect_results_ms = (time.perf_counter() - collect_start) * 1e3 if collect_diagnostics else None
        cleanup_start = time.perf_counter() if collect_diagnostics else None
        try:
            _rm_dic(keep_res, torch_path)
        finally:
            if collect_diagnostics:
                cleanup_ms = (time.perf_counter() - cleanup_start) * 1e3
                benchmark_total_ms = (time.perf_counter() - benchmark_start) * 1e3
                active_time_cost = time_cost if isinstance(time_cost, list) else [time_cost]
                prewarm_wall_ms = sum(metrics["prewarm_enqueue_ms"] + metrics["prewarm_synchronize_ms"]
                                      for metrics in config_metrics)
                measured_phase_ms = (prewarm_wall_ms + profile_create_ms + profile_enter_ms + profile_loop_ms +
                                     profile_finalize_ms + collect_results_ms + cleanup_ms)

                for func_idx, metrics in enumerate(config_metrics):
                    device_total_ms = (profiled_device_total_ms[func_idx]
                                       if profiled_device_total_ms is not None else None)
                    device_active_mean_ms = (active_time_cost[func_idx] if func_idx < len(active_time_cost)
                                             and active_time_cost[func_idx] is not None else None)
                    details = {
                        "config_index":
                        func_idx,
                        "profile_calls":
                        total,
                        "prewarm_enqueue_ms":
                        f'{metrics["prewarm_enqueue_ms"]:.3f}',
                        "prewarm_synchronize_ms":
                        f'{metrics["prewarm_synchronize_ms"]:.3f}',
                        "profile_enqueue_ms":
                        f'{metrics["profile_enqueue_ms"]:.3f}',
                        "profile_synchronize_ms":
                        f'{metrics["profile_synchronize_ms"]:.3f}',
                        "profile_l2_clear_ms":
                        f'{metrics["profile_l2_clear_ms"]:.3f}',
                        "profile_wall_ms":
                        f'{metrics["profile_wall_ms"]:.3f}',
                        "target_device_total_ms":
                        (f"{device_total_ms:.3f}" if device_total_ms is not None else "unavailable"),
                        "target_device_active_mean_ms":
                        (f"{device_active_mean_ms:.3f}" if device_active_mean_ms is not None else "unavailable"),
                        "unaccounted_profile_ms": (f'{metrics["profile_wall_ms"] - device_total_ms:.3f}'
                                                   if device_total_ms is not None else "unavailable"),
                        "config": (diagnostic_labels[func_idx] if diagnostic_labels is not None else str(func_idx)),
                    }
                    diagnostic_callback("benchmark_config", details)

                diagnostic_callback(
                    "benchmark_profiler", {
                        "result": collect_error or "success",
                        "configs": len(funcs),
                        "prewarm_calls": len(funcs),
                        "profile_calls": len(funcs) * total,
                        "profile_create_ms": f"{profile_create_ms:.3f}",
                        "profile_enter_ms": f"{profile_enter_ms:.3f}",
                        "profile_loop_ms": f"{profile_loop_ms:.3f}",
                        "profile_finalize_ms": f"{profile_finalize_ms:.3f}",
                        "collect_results_ms": f"{collect_results_ms:.3f}",
                        "cleanup_ms": f"{cleanup_ms:.3f}",
                        "other_ms": f"{max(0.0, benchmark_total_ms - measured_phase_ms):.3f}",
                        "total_ms": f"{benchmark_total_ms:.3f}",
                        "target_kernel_name": target_kernel_name,
                        "profile_results_retained": keep_res,
                    })

    return time_cost


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
    return_diagnostics: bool = False,
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
    :param return_diagnostics: also return the total profiled target-kernel duration for each function
    :type return_diagnostics: bool
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
        time_cost = float("inf") if num_funcs == 1 else [float("inf")] * num_funcs
        if return_diagnostics:
            return time_cost, [float("inf")] * num_funcs
        return time_cost

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
    profiled_device_total = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for profile_index in np.arange(0, num_warmup + num_active):
            row_index = func_idx * (num_warmup + num_active) + profile_index
            duration_us = filter_df.iloc[row_index]["Duration(us)"]
            profiled_device_total[func_idx] += duration_us
            if profile_index >= num_warmup:
                time_cost[func_idx] += duration_us
    time_cost = [x / num_active / 1e3 for x in time_cost]
    profiled_device_total = [x / 1e3 for x in profiled_device_total]

    result = time_cost[0] if num_funcs == 1 else time_cost
    if return_diagnostics:
        return result, profiled_device_total
    return result
