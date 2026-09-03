from __future__ import annotations

import csv
import logging
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

WARMUP_MATMUL_SHAPE = os.environ.get("CANN_BENCH_WARMUP_MATMUL_SHAPE", '"10240,10240;10240,10240"')
WARMUP_REDUCE_SHAPE = os.environ.get("CANN_BENCH_WARMUP_REDUCE_SHAPE", '"96,1024,1024;3"')
DEFAULT_PROFILE_ROOT = Path("logs") / "profiler_raw"

_WARMUP_TENSORS: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


@dataclass
class ProfileResult:
    latency_us: float | None
    active_window_us: float | None
    kernel_sum_us: float | None
    window_gap_us: float | None
    kernel_count: int | None
    step_count: int | None
    device_kernels: dict[str, float] = field(default_factory=dict)
    device_timeline: dict[str, Any] = field(default_factory=dict)
    csv_path: str | None = None
    trace_view_path: str | None = None
    error: str | None = None
    perf_metric_strategy: str = "kernel_details"
    measurement_scope: str = "timing_matrix.active_window_score"
    elapsed_us_source: str = "kernel_details.csv.active_window_median"


def profile_kernel_details(
    label: str,
    case_id: str,
    func: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    profile_root: Path | str = DEFAULT_PROFILE_ROOT,
    freq_boost: bool = True,
) -> tuple[Any | None, ProfileResult]:
    """Run a callable with the OpForge/CANN-Bench kernel-details timing contract."""

    try:
        import torch_npu
    except Exception as exc:  # pragma: no cover - runtime dependent.
        return None, _failed(f"torch_npu import failed: {exc}")

    warmup = max(int(warmup), 0)
    repeat = max(int(repeat), 1)
    root = Path(profile_root)
    prof_dir = root / _safe_path(case_id) / _safe_path(label)
    if prof_dir.exists():
        shutil.rmtree(prof_dir, ignore_errors=True)
    prof_dir.mkdir(parents=True, exist_ok=True)

    output = None
    try:
        with torch.inference_mode():
            if freq_boost:
                _prepare_warmup_tensors()
                _boost_freq_and_clear_cache()
            output = func()
            _sync_npu()
            _run_profiler(torch_npu, func, prof_dir, warmup, repeat, freq_boost=freq_boost)
    except Exception as exc:  # pragma: no cover - runtime dependent.
        return output, _failed(f"{type(exc).__name__}: {exc}", prof_dir=prof_dir)

    wait_info = _wait_profiler_files_ready(prof_dir)
    csv_path = _locate_kernel_details(prof_dir)
    trace_path = _locate_trace_view(csv_path, prof_dir)
    if not csv_path:
        return output, _failed("kernel_details.csv not found", prof_dir=prof_dir, wait_info=wait_info)

    timing = _parse_kernel_timing_csv(csv_path)
    if timing.get("error_msg"):
        return output, _failed(str(timing["error_msg"]), prof_dir=prof_dir, csv_path=csv_path, trace_path=trace_path)
    metrics = timing.get("metrics")
    if not isinstance(metrics, dict):
        return output, _failed("kernel_details.csv produced no positive timed kernels", prof_dir=prof_dir,
                               csv_path=csv_path, trace_path=trace_path)

    active = float(metrics["active_window_us"])
    kernel = float(metrics["kernel_sum_us"])
    gap = float(metrics["window_gap_us"])
    return output, ProfileResult(
        latency_us=active,
        active_window_us=active,
        kernel_sum_us=kernel,
        window_gap_us=gap,
        kernel_count=int(metrics["kernel_count"]),
        step_count=int(metrics["measured_step_count"]),
        device_kernels=dict(metrics["device_kernels"]),
        device_timeline={
            "device_active_window_us": active,
            "device_kernel_duration_sum_us": kernel,
            "median_step_kernel_duration_sum_us": float(metrics["median_step_kernel_sum_us"]),
            "device_window_gap_us": gap,
            "measured_step_count": int(metrics["measured_step_count"]),
            "blank_step_rows": int(metrics.get("blank_step_rows", 0)),
            "step_windows": metrics["step_windows"],
            "profiler_file_wait": wait_info,
        },
        csv_path=str(csv_path),
        trace_view_path=str(trace_path) if trace_path else None,
    )


def _failed(
    error: str,
    *,
    prof_dir: Path | None = None,
    csv_path: Path | None = None,
    trace_path: Path | None = None,
    wait_info: dict[str, Any] | None = None,
) -> ProfileResult:
    timeline: dict[str, Any] = {}
    if wait_info is not None:
        timeline["profiler_file_wait"] = wait_info
    return ProfileResult(
        None,
        None,
        None,
        None,
        None,
        None,
        {},
        timeline,
        str(csv_path) if csv_path else None,
        str(trace_path) if trace_path else None,
        error if prof_dir is None else f"{error}; prof_dir={prof_dir}",
    )


def _run_profiler(torch_npu: Any, func: Callable[[], Any], prof_dir: Path, warmup: int, repeat: int, *,
                  freq_boost: bool) -> None:
    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
    experimental_config = _experimental_config(torch_npu)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    sink = tempfile.NamedTemporaryFile(mode="w+", prefix="tutorial_profiler_", suffix=".log", delete=False)
    original_basic_config = logging.basicConfig

    def _silent_basic_config(**kwargs: Any) -> Any:
        kwargs["level"] = logging.ERROR
        kwargs["force"] = True
        return original_basic_config(**kwargs)

    logging.basicConfig = _silent_basic_config
    try:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=repeat, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(prof_dir)),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                experimental_config=experimental_config,
        ) as prof:
            for step in range(warmup + repeat):
                if freq_boost and step >= warmup:
                    _clear_cache()
                func()
                prof.step()
        time.sleep(0.1)
    finally:
        try:
            from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
            MultiProcessPool().close_pool(wait=True)
        except Exception:
            pass
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sink.close()
        logging.basicConfig = original_basic_config
        _remove_if_quiet(sink.name)


def _experimental_config(torch_npu: Any) -> Any:
    export_types = []
    for name in ("Text", "Db"):
        try:
            export_types.append(getattr(torch_npu.profiler.ExportType, name))
        except AttributeError:
            pass
    if not export_types:
        export_types = [torch_npu.profiler.ExportType.Text]
    return torch_npu.profiler._ExperimentalConfig(
        export_type=export_types,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )


def _prepare_warmup_tensors() -> None:
    global _WARMUP_TENSORS
    if _WARMUP_TENSORS is None:
        device = torch.device("npu")
        mm1 = torch.rand((10240, 10240), dtype=torch.float16, device=device)
        mm2 = torch.rand((10240, 10240), dtype=torch.float16, device=device)
        reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16, device=device)
        _WARMUP_TENSORS = (mm1, mm2, reduce_input)


def _boost_freq_and_clear_cache() -> None:
    if _WARMUP_TENSORS is None:
        return
    mm1, mm2, reduce_input = _WARMUP_TENSORS
    try:
        torch.matmul(mm1, mm2)
        _sync_npu(mm1.device)
        torch.max(reduce_input)
        _sync_npu(reduce_input.device)
    except RuntimeError:
        _sync_npu(mm1.device)


def _clear_cache() -> None:
    if _WARMUP_TENSORS is None:
        return
    reduce_input = _WARMUP_TENSORS[2]
    try:
        torch.max(reduce_input)
        _sync_npu(reduce_input.device)
    except RuntimeError:
        _sync_npu(reduce_input.device)


def _sync_npu(device: torch.device | None = None) -> None:
    if device is None:
        torch.npu.synchronize()
    else:
        torch.npu.synchronize(device)


def _parse_kernel_timing_csv(csv_path: Path) -> dict[str, Any]:
    step_kernel_times: dict[str, dict[str, list[float]]] = {}
    step_windows: dict[str, dict[str, Any]] = {}
    blank_step_rows = 0
    blank_examples: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Step Id", "Name", "Type", "Start Time(us)", "Duration(us)"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            return {"error_msg": "kernel_details.csv missing required fields: " + ", ".join(missing)}
        for row in reader:
            try:
                duration = float(str(row.get("Duration(us)", "0")).strip())
            except (TypeError, ValueError):
                continue
            if duration <= 0:
                continue
            try:
                start_us = float(str(row.get("Start Time(us)", "")).strip())
            except (TypeError, ValueError):
                return {
                    "error_msg": f"kernel_details.csv has malformed Start Time(us): {row.get('Start Time(us)', '')!r}"
                }
            op_type = row.get("Type", "")
            input_shapes = row.get("Input Shapes", "")
            name = row.get("Name", op_type) or op_type
            if _is_warmup_kernel(op_type, input_shapes):
                continue
            step_id = str(row.get("Step Id", "")).strip()
            if not step_id:
                blank_step_rows += 1
                if len(blank_examples) < 3:
                    blank_examples.append({
                        "name": str(name),
                        "type": str(op_type),
                        "start_us": str(row.get("Start Time(us)", "")),
                        "duration_us": str(row.get("Duration(us)", "")),
                    })
                continue
            step_kernel_times.setdefault(step_id, {}).setdefault(name, []).append(duration)
            window = step_windows.setdefault(step_id, {
                "start_us": start_us,
                "end_us": start_us + duration,
                "kernel_duration_sum_us": 0.0,
                "kernel_count": 0,
            })
            window["start_us"] = min(float(window["start_us"]), start_us)
            window["end_us"] = max(float(window["end_us"]), start_us + duration)
            window["kernel_duration_sum_us"] = float(window["kernel_duration_sum_us"]) + duration
            window["kernel_count"] = int(window["kernel_count"]) + 1
    if blank_step_rows:
        return {
            "error_msg":
            f"kernel_details.csv contains non-warmup kernels without Step Id; blank_step_rows={blank_step_rows}; examples={blank_examples}"
        }
    if not step_kernel_times:
        return {}

    for window in step_windows.values():
        kernel_sum = float(window["kernel_duration_sum_us"])
        active_window = max(float(window["end_us"]) - float(window["start_us"]), 0.0)
        window["start_us"] = round(float(window["start_us"]), 3)
        window["end_us"] = round(float(window["end_us"]), 3)
        window["kernel_duration_sum_us"] = round(kernel_sum, 2)
        window["device_active_window_us"] = round(active_window, 2)
        window["device_window_gap_us"] = round(max(active_window - kernel_sum, 0.0), 2)
        window["device_window_minus_kernel_sum_us"] = round(active_window - kernel_sum, 2)

    all_kernel_times: dict[str, list[float]] = {}
    for kernels in step_kernel_times.values():
        for name, times in kernels.items():
            all_kernel_times.setdefault(name, []).append(sum(times))
    device_kernels = {name: round(_median(times), 2) for name, times in all_kernel_times.items()}
    total_kernel_us = sum(device_kernels.values())
    active_window_us = _median([float(window["device_active_window_us"]) for window in step_windows.values()])
    median_step_kernel_sum_us = _median([float(window["kernel_duration_sum_us"]) for window in step_windows.values()])
    kernel_counts = [int(window["kernel_count"]) for window in step_windows.values()]
    return {
        "metrics": {
            "device_kernels": device_kernels,
            "kernel_sum_us": round(total_kernel_us, 2),
            "active_window_us": round(active_window_us, 2),
            "median_step_kernel_sum_us": round(median_step_kernel_sum_us, 2),
            "window_gap_us": round(max(active_window_us - median_step_kernel_sum_us, 0.0), 2),
            "measured_step_count": len(step_windows),
            "kernel_count": int(round(statistics.median(kernel_counts))) if kernel_counts else 0,
            "step_windows": dict(sorted(step_windows.items(), key=lambda item: _step_sort_key(item[0]))),
            "blank_step_rows": blank_step_rows,
        }
    }


def _is_warmup_kernel(op_type: str, input_shapes: str) -> bool:
    if not op_type or not input_shapes:
        return False
    if op_type == "MatMulV3" and WARMUP_MATMUL_SHAPE in input_shapes:
        return True
    if op_type == "ReduceMax" and WARMUP_REDUCE_SHAPE in input_shapes:
        return True
    return False


def _wait_profiler_files_ready(prof_dir: Path, *, min_wait_s: float = 0.2, timeout_s: float = 5.0,
                               poll_s: float = 0.2) -> dict[str, Any]:
    start = time.monotonic()
    last_snapshot: tuple[tuple[str, int, int], ...] | None = None
    stable_since: float | None = None
    polls = 0
    while True:
        polls += 1
        snapshot = _prof_file_snapshot(prof_dir)
        now = time.monotonic()
        if snapshot and snapshot == last_snapshot:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= min_wait_s:
                return {"ready": True, "wait_s": now - start, "polls": polls, "file_count": len(snapshot)}
        else:
            last_snapshot = snapshot
            stable_since = now if snapshot else None
        if now - start >= timeout_s:
            return {"ready": False, "wait_s": now - start, "polls": polls, "file_count": len(snapshot)}
        time.sleep(poll_s)


def _prof_file_snapshot(prof_dir: Path) -> tuple[tuple[str, int, int], ...]:
    rows: list[tuple[str, int, int]] = []
    if not prof_dir.is_dir():
        return tuple()
    for path in prof_dir.rglob("*"):
        if not path.is_file() or path.name.endswith(".done"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append((str(path.relative_to(prof_dir)), stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(rows))


def _locate_kernel_details(prof_dir: Path) -> Path | None:
    direct = prof_dir / "kernel_details.csv"
    if direct.is_file():
        return direct
    for path in prof_dir.rglob("kernel_details.csv"):
        if path.is_file():
            return path
    return None


def _locate_trace_view(csv_path: Path | None, prof_dir: Path) -> Path | None:
    candidates = []
    if csv_path is not None:
        candidates.append(csv_path.parent / "trace_view.json")
    candidates.extend(prof_dir.rglob("trace_view.json"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _step_sort_key(step_id: str) -> tuple[int, str]:
    try:
        return (int(step_id), str(step_id))
    except (TypeError, ValueError):
        return (10**9, str(step_id))


def _safe_path(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "unknown"


def _remove_if_quiet(path: str) -> None:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return
    lowered = text.lower()
    if any(token in lowered for token in ("error", "exception", "traceback", "failed", "fail ")):
        return
    try:
        Path(path).unlink()
    except OSError:
        pass
