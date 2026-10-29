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
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import triton.runtime as runtime


class ProfilerResultMismatchError(RuntimeError):
    def __init__(self, target_kernel_name: str, expected_rows: int, actual_rows: int):
        self.target_kernel_name = target_kernel_name
        self.expected_rows = expected_rows
        self.actual_rows = actual_rows
        super().__init__(
            "Profiler rows filtered by target kernel name do not match the expected count. "
            f"target_kernel_name={target_kernel_name!r}, expected_rows={expected_rows}, actual_rows={actual_rows}"
        )


def _npu_bench_options(clear_l2_cache=False):
    values = tuple(
        type(default)(os.getenv("TRITON_NPU_BENCH_" + name, default)) for name, default in (
            ("STABILITY_RUNS", 1),
            ("STABILITY_TOLERANCE", 0.01),
            ("LOWER_QUANTILE", 0.0),
            ("UPPER_QUANTILE", 1.0),
            ("MAX_RETRIES", 0),
            ("CACHE_MODE", "cold" if clear_l2_cache else "hot"),
        ))
    confirmations, tolerance, lower, upper, batches, mode = values
    if not (confirmations > 0 and batches >= 0 and math.isfinite(tolerance) and tolerance >= 0
            and 0 <= lower < upper <= 1 and mode in ("hot", "cold")):
        raise ValueError("Invalid TRITON_NPU_BENCH settings")
    return values


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
    import torch
    import torch_npu

    if not isinstance(funcs, list):
        funcs = [funcs]
    confirmations, tolerance, lower, upper, batches, mode = _npu_bench_options(clear_l2_cache)
    clear_l2_cache = mode == "cold"

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
        base_path = os.path.join(
            runtime.cache.get_home_dir(), ".triton", "profile_results"
        )
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")

    buffer = None
    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()  # to avoid type cast
        buffer.sum()
        torch.npu.synchronize()  # shake out of any npu error

    histories = [[] for _ in funcs]
    pending = list(builtins.range(len(funcs)))
    for batch in builtins.range(max(1, batches)):
        selected = [funcs[index] for index in pending] * confirmations
        batch_path = os.path.join(torch_path, f"batch_{batch}") if batches > 1 else torch_path
        _profile_npu_batch(selected, warmup, active, buffer, batch_path, experimental_config)
        try:
            scores = _collect_prof_result(batch_path, selected, warmup, active, target_kernel_name, clear_l2_cache,
                                          (lower, upper) if active >= 10 else (0, 1))
        finally:
            _rm_dic(keep_res, torch_path)
        if len(selected) == 1:
            scores = [scores]
        for position, index in enumerate(pending):
            histories[index] = sorted(histories[index] + scores[position::len(pending)])[:confirmations]
        pending = [
            index for index in pending
            if confirmations > 1 and (len(histories[index]) < confirmations or not math.isfinite(histories[index][-1])
                                      or histories[index][0] <= 0 or histories[index][-1] > histories[index][0] *
                                      (1 + tolerance))
        ]
        if not pending:
            break
    if clear_l2_cache:
        del buffer
    if pending:
        print(f"NPU benchmark: minimum not confirmed for configurations {[index + 1 for index in pending]}")
    time_cost = [history[0] for history in histories]
    return time_cost[0] if len(funcs) == 1 else time_cost


def _profile_npu_batch(funcs, warmup, active, buffer, torch_path, experimental_config):
    import torch
    import torch_npu

    mode = os.getenv("TRITON_NPU_BENCH_PROFILER_MODE", "torch_npu").strip().lower()
    if mode not in ("torch_npu", "mspti"):
        raise ValueError("Invalid TRITON_NPU_BENCH_PROFILER_MODE")
    total = warmup + active
    profile = _mspti_npu_profile(torch_path, buffer) if mode == "mspti" else torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )
    with profile as prof:
        for fn in funcs:
            for _ in builtins.range(total):
                if buffer is not None:
                    buffer.sum()  # use buffer read to clear l2 cache
                    torch.npu.synchronize()
                fn()
                torch.npu.synchronize()


@contextmanager
def _mspti_npu_profile(torch_path, buffer):
    import csv
    import torch
    from mspti import KernelMonitor, MsptiResult

    monitor, rows = KernelMonitor(), []

    def capture(data):
        rows.append((data.name, data.start, data.end, data.type))

    @contextmanager
    def record():
        result = monitor.start(capture)
        if result != MsptiResult.MSPTI_SUCCESS:
            monitor.stop()
            raise RuntimeError(f"msPTI start failed: {result}")
        try:
            yield
        except BaseException:
            monitor.stop()
            raise
        else:
            result = monitor.stop()
            if result != MsptiResult.MSPTI_SUCCESS:
                raise RuntimeError(f"msPTI stop failed: {result}")

    cache_names = set()
    if buffer is not None:
        # msPTI reports kernel types, not operator types; identify the eviction kernel once.
        with record():
            buffer.sum()
            torch.npu.synchronize()
        if not rows:
            raise RuntimeError("msPTI did not record the L2 eviction kernel; check LD_PRELOAD and CANN support")
        cache_names = {row[0] for row in rows}
        rows.clear()
    with record():
        yield
    if not rows or any(start <= 0 or end <= start for _, start, end, _ in rows):
        raise RuntimeError("msPTI returned missing or invalid kernel timestamps; check LD_PRELOAD and CANN support")
    rows.sort(key=lambda row: row[1])
    os.makedirs(torch_path, exist_ok=True)
    with open(os.path.join(torch_path, "kernel_details.csv"), "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("Name", "Type", "Start Time(us)", "Duration(us)", "Accelerator Core"))
        writer.writerows((name, "ReduceSum" if name in cache_names else kind, start / 1e3, (end - start) / 1e3, kind)
                         for name, start, end, kind in rows)


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
    quantiles=(0, 1),
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
    :param quantiles: lower and upper sorted-sample boundaries; retain at least one duration
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

    durations = filter_df["Duration(us)"].to_numpy(dtype=float)[:expected_rows].reshape(
        num_funcs, num_warmup + num_active)
    samples = durations[:, num_warmup:] / 1e3
    valid = np.isfinite(samples).all(axis=1) & (samples > 0).all(axis=1)
    if quantiles != (0, 1):
        start, end = (int(num_active * quantile) for quantile in quantiles)
        samples = np.sort(samples, axis=1)[:, start:max(start + 1, end)]
    time_cost = np.where(valid, samples.mean(axis=1), np.inf).tolist()

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost
