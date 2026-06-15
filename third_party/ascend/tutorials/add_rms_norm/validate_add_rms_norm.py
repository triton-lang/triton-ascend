# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CANN-Bench-style validation for the AddRmsNorm tutorial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

try:
    import torch_npu
except Exception:  # pragma: no cover - depends on Ascend runtime.
    torch_npu = None

from add_rms_norm import add_rms_norm, add_rms_norm_reference

PUBLIC_BH_SHAPES = [
    (1, 3584),
    (1, 4096),
    (1, 5120),
    (1, 8192),
    (8, 3584),
    (8, 4096),
    (8, 5120),
    (8, 8192),
    (16, 3584),
    (16, 4096),
    (16, 5120),
    (16, 8192),
    (32, 3584),
    (32, 4096),
    (32, 5120),
    (32, 8192),
    (64, 3584),
    (64, 4096),
    (64, 5120),
    (64, 8192),
]

PUBLIC_SEQUENCE_LENGTHS = [1, 8, 32, 128]
DEFAULT_VALUE_RANGES = ((-1.0, 1.0), (-1.0, 1.0), (0.5, 1.5))
RANDOM_GENERALIZATION_POLICY = "seeded_non_public_bsh_v1"
RANDOM_MAX_ELEMENTS = 8 * 1024 * 1024
RANDOM_B_CANDIDATES = [1, 2, 3, 4, 5, 7, 8, 12, 16, 24, 32, 48, 64]
RANDOM_S_CANDIDATES = [1, 2, 3, 5, 7, 8, 16, 31, 32, 33, 64, 96, 127, 128]
BF16_THRESHOLD = 2**-7
BF16_SMALL_VALUE_THRESHOLD = 2**-8
BF16_SMALL_VALUE_ERROR = 2**-16
BF16_CANCEL_BOUNDARY = 2**-3
BF16_CANCEL_ZERO_THRESHOLD = 2**-3
WARMUP_MATMUL_SHAPE = os.environ.get("CANN_BENCH_WARMUP_MATMUL_SHAPE", '"10240,10240;10240,10240"')
WARMUP_REDUCE_SHAPE = os.environ.get("CANN_BENCH_WARMUP_REDUCE_SHAPE", '"96,1024,1024;3"')


@dataclass(frozen=True)
class Case:
    bsz: int
    seq: int
    hidden: int
    kind: str = "public"
    case_id: str | None = None
    case_seed: int | None = None
    audit_seed: int | None = None
    shape_policy: str = ""
    random_category: str = ""
    note: str = ""
    value_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_VALUE_RANGES

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.bsz, self.seq, self.hidden)


@dataclass
class ProfileResult:
    latency_us: float | None
    active_window_us: float | None
    kernel_sum_us: float | None
    window_gap_us: float | None
    kernel_count: int | None
    step_count: int | None
    device_kernels: dict[str, float]
    device_timeline: dict[str, object]
    csv_path: str | None
    trace_view_path: str | None
    error: str | None = None
    perf_metric_strategy: str = ""
    measurement_scope: str = ""
    elapsed_us_source: str = ""


_WARMUP_TENSORS: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def public_cases() -> list[Case]:
    return [Case(bsz, seq, hidden) for seq in PUBLIC_SEQUENCE_LENGTHS for bsz, hidden in PUBLIC_BH_SHAPES]


def random_generalization_cases(
    count: int,
    seed: int,
    *,
    policy: str = RANDOM_GENERALIZATION_POLICY,
) -> list[Case]:
    if count < 0:
        raise ValueError("random generalization count must be non-negative")
    if policy != RANDOM_GENERALIZATION_POLICY:
        raise ValueError(f"unsupported random shape policy: {policy}")

    rng = random.Random(int(seed))
    public_shapes = {case.shape for case in public_cases()}
    seen: set[tuple[int, int, int]] = set()
    cases: list[Case] = []
    attempts = 0
    max_attempts = max(200, count * 200)

    while len(cases) < count and attempts < max_attempts:
        attempts += 1
        category = rng.choice(["near_hidden", "contract_shape", "wide_hidden", "small_tail"])
        if category == "near_hidden":
            base_bsz, base_hidden = rng.choice(PUBLIC_BH_SHAPES)
            base_seq = rng.choice(PUBLIC_SEQUENCE_LENGTHS)
            hidden_delta = rng.choice([-384, -257, -128, -64, 64, 127, 192, 256, 384])
            bsz, seq, hidden = base_bsz, base_seq, max(1, base_hidden + hidden_delta)
        elif category == "wide_hidden":
            bsz = rng.choice([1, 2, 4, 8, 16])
            seq = rng.choice([1, 2, 4, 8, 16, 32])
            hidden = rng.randint(8193, 12288)
        elif category == "small_tail":
            bsz = rng.choice([1, 2, 3, 5, 8, 13, 16])
            seq = rng.choice([1, 2, 3, 5, 7, 11, 17, 31, 33])
            hidden = rng.randint(1, 1024)
        else:
            bsz = rng.choice(RANDOM_B_CANDIDATES)
            seq = rng.choice(RANDOM_S_CANDIDATES)
            hidden = rng.randint(64, 8192)

        shape = (bsz, seq, hidden)
        if shape in public_shapes or shape in seen:
            continue
        if bsz * seq * hidden > RANDOM_MAX_ELEMENTS:
            continue

        seen.add(shape)
        random_index = len(cases) + 1
        case_id = f"custom/add_rms_norm_random_{random_index:03d}"
        cases.append(
            Case(
                bsz,
                seq,
                hidden,
                kind="random_generalization",
                case_id=case_id,
                case_seed=_seed_from_case_id(case_id, int(seed)),
                audit_seed=int(seed),
                shape_policy=policy,
                random_category=category,
                note="seeded non-public random shape sample",
            ))

    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} random cases after {attempts} attempts, expected {count}")
    return cases


def _cannbench_case_id(index: int) -> str:
    return f"custom/add_rms_norm_{index}"


def _seed_from_case_id(case_id: str, eval_seed: int = 0) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    deterministic_hash = int.from_bytes(digest[:8], byteorder="big") % (2**31)
    return (int(eval_seed) + deterministic_hash) % (2**31)


def _cannbench_case_seed(index: int, eval_seed: int = 0) -> int:
    return _seed_from_case_id(_cannbench_case_id(index), eval_seed)


def _gen_bf16_uniform(
    shape: tuple[int, int, int],
    min_val: float,
    max_val: float,
    gen: torch.Generator,
) -> torch.Tensor:
    tensor_f64 = torch.rand(shape, dtype=torch.float64, generator=gen) * (max_val - min_val) + min_val
    return tensor_f64.to(torch.bfloat16)


def make_inputs(
    shape: tuple[int, int, int],
    case_seed: int,
    device: str,
    value_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_VALUE_RANGES,
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator()
    gen.manual_seed(int(case_seed))
    x1_range, x2_range, gamma_range = value_ranges
    x1_cpu = _gen_bf16_uniform(shape, x1_range[0], x1_range[1], gen)
    x2_cpu = _gen_bf16_uniform(shape, x2_range[0], x2_range[1], gen)
    gamma_cpu = _gen_bf16_uniform(shape, gamma_range[0], gamma_range[1], gen)
    return (
        x1_cpu.to(device=device).contiguous(),
        x2_cpu.to(device=device).contiguous(),
        gamma_cpu.to(device=device).contiguous(),
    )


def add_rms_norm_torch_npu(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    op = getattr(torch_npu, "npu_add_rms_norm", None)
    if not callable(op):
        raise RuntimeError("torch_npu.npu_add_rms_norm is not available")
    out = op(x1, x2, gamma, float(epsilon))
    if isinstance(out, (tuple, list)):
        if not out:
            raise RuntimeError("torch_npu.npu_add_rms_norm returned no outputs")
        return out[0]
    return out


def add_rms_norm_cannbench_golden(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Semantic golden matching the custom CANN-Bench golden.py implementation."""

    z = x1.to(torch.float32) + x2.to(torch.float32)
    variance = torch.mean(z * z, dim=-1, keepdim=True)
    y = z * torch.rsqrt(variance + float(epsilon)) * gamma.to(torch.float32)
    return y.to(dtype=x1.dtype)


def cannbench_bf16_compare(
    actual: torch.Tensor,
    golden: torch.Tensor,
    native_output: torch.Tensor | None = None,
) -> dict[str, object]:
    """Replicate the CANN-Bench BF16 relative-error acceptance path."""

    if actual.device.type != "cpu":
        actual = actual.cpu()
    if golden.device.type != "cpu":
        golden = golden.cpu()
    if native_output is not None and native_output.device.type != "cpu":
        native_output = native_output.cpu()
    if actual.shape != golden.shape:
        return {
            "passed": False,
            "threshold": BF16_THRESHOLD,
            "mere": 0.0,
            "mare": 0.0,
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "mismatch_count": int(actual.numel()),
            "total_count": int(actual.numel()),
            "error_msg": f"shape mismatch: actual={tuple(actual.shape)} golden={tuple(golden.shape)}",
        }

    target_dtype = actual.dtype
    actual64 = actual.to(torch.float64)
    golden_truncated = golden.to(target_dtype).to(torch.float64)

    if torch.any(torch.isnan(actual64)) or torch.any(torch.isnan(golden_truncated)):
        if not torch.all(torch.isnan(actual64) == torch.isnan(golden_truncated)):
            return {
                "passed": False,
                "threshold": BF16_THRESHOLD,
                "mere": 0.0,
                "mare": 0.0,
                "max_diff": 0.0,
                "mean_diff": 0.0,
                "mismatch_count": int(actual.numel()),
                "total_count": int(actual.numel()),
                "small_value_error_count": 0,
                "small_value_cpu_error_count": 0,
                "small_value_total_count": 0,
                "cancel_error_count": 0,
                "cancel_cpu_error_count": 0,
                "cancel_total_count": 0,
                "error_msg": "NaN position mismatch",
            }

    inf_match_mask = torch.zeros_like(actual64, dtype=torch.bool)
    if torch.any(torch.isinf(actual64)) or torch.any(torch.isinf(golden_truncated)):
        inf_out = torch.isinf(actual64)
        inf_gold = torch.isinf(golden_truncated)
        inf_mismatch = inf_out != inf_gold
        if torch.any(inf_mismatch):
            max_finite = float(torch.finfo(target_dtype).max)
            if torch.any(inf_out & ~inf_gold):
                mask = inf_out & ~inf_gold
                actual64[mask] = torch.sign(actual64[mask]) * max_finite
            if torch.any(inf_gold & ~inf_out):
                mask = inf_gold & ~inf_out
                golden_truncated[mask] = torch.sign(golden_truncated[mask]) * max_finite
        both_inf = inf_out & inf_gold
        if torch.any(both_inf):
            if not torch.all(torch.sign(actual64[both_inf]) == torch.sign(golden_truncated[both_inf])):
                return {
                    "passed": False,
                    "threshold": BF16_THRESHOLD,
                    "mere": 0.0,
                    "mare": 0.0,
                    "max_diff": 0.0,
                    "mean_diff": 0.0,
                    "mismatch_count": int(both_inf.sum().item()),
                    "total_count": int(actual.numel()),
                    "small_value_error_count": 0,
                    "small_value_cpu_error_count": 0,
                    "small_value_total_count": 0,
                    "cancel_error_count": 0,
                    "cancel_cpu_error_count": 0,
                    "cancel_total_count": 0,
                    "error_msg": "Inf sign mismatch",
                }
            inf_match_mask[both_inf] = True

    diff = torch.abs(actual64 - golden_truncated)
    golden_abs = torch.abs(golden_truncated)
    rel = diff / (golden_abs + 1e-7)
    valid = ~(torch.isnan(rel) | torch.isinf(rel) | inf_match_mask)
    total_count = int(actual.numel())
    if valid.any().item():
        valid_rel = rel[valid]
        valid_diff = diff[valid]
        mere = float(valid_rel.mean().item())
        mare = float(valid_rel.max().item())
        max_diff = float(valid_diff.max().item())
        mean_diff = float(valid_diff.mean().item())
    else:
        mere = 0.0
        mare = 0.0
        max_diff = 0.0
        mean_diff = 0.0

    mare_threshold = 10 * BF16_THRESHOLD
    if mere < BF16_THRESHOLD and mare < mare_threshold:
        return {
            "passed": True,
            "threshold": BF16_THRESHOLD,
            "mere": mere,
            "mare": mare,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "mismatch_count": 0,
            "total_count": total_count,
            "small_value_error_count": 0,
            "small_value_cpu_error_count": 0,
            "small_value_total_count": 0,
            "cancel_error_count": 0,
            "cancel_cpu_error_count": 0,
            "cancel_total_count": 0,
            "error_msg": "",
        }

    mismatch_mask = (rel > mare_threshold) & valid
    mismatch_count = int(mismatch_mask.sum().item())

    small_value_mask = (golden_abs < BF16_SMALL_VALUE_THRESHOLD) & valid
    small_value_total_count = int(small_value_mask.sum().item())
    small_value_error_mask = small_value_mask & (diff > BF16_SMALL_VALUE_ERROR)
    small_value_error_count = int(small_value_error_mask.sum().item())

    native64 = (native_output.to(torch.float64)
                if native_output is not None else golden.to(target_dtype).to(torch.float64))
    cpu_diff = torch.abs(native64 - golden_truncated)
    cpu_small_value_error_mask = small_value_mask & (cpu_diff > BF16_SMALL_VALUE_ERROR)
    small_value_cpu_error_count = int(cpu_small_value_error_mask.sum().item())
    if small_value_total_count > 0:
        if small_value_cpu_error_count == 0:
            small_value_passed = small_value_error_count == 0
        else:
            small_value_passed = (small_value_error_count / max(small_value_cpu_error_count, 1)) <= 2
    else:
        small_value_passed = True

    actual_abs = torch.abs(actual64)
    cancel_mask = ((actual_abs < BF16_CANCEL_ZERO_THRESHOLD)
                   & (golden_abs < BF16_CANCEL_BOUNDARY)
                   & (golden_abs >= BF16_SMALL_VALUE_THRESHOLD)
                   & valid)
    cancel_total_count = int(cancel_mask.sum().item())
    cancel_error_mask = cancel_mask & (rel > mare_threshold)
    cancel_error_count = int(cancel_error_mask.sum().item())
    cpu_relative_error = cpu_diff / (golden_abs + 1e-7)
    cancel_cpu_error_mask = cancel_mask & (cpu_relative_error > mare_threshold)
    cancel_cpu_error_count = int(cancel_cpu_error_mask.sum().item())
    if cancel_total_count > 0:
        if cancel_cpu_error_count == 0:
            cancel_passed = cancel_error_count == 0
        else:
            cancel_passed = (cancel_error_count / max(cancel_cpu_error_count, 1)) <= 2
    else:
        cancel_passed = True

    mismatch_in_normal = mismatch_mask & ~small_value_mask & ~cancel_mask
    normal_mismatch_count = int(mismatch_in_normal.sum().item())
    passed = bool(normal_mismatch_count == 0 and small_value_passed and cancel_passed)
    if normal_mismatch_count > 0:
        normal_mask = ~small_value_mask & ~cancel_mask & valid
        normal_rel = rel[normal_mask]
        display_mere = float(normal_rel.mean().item()) if normal_rel.numel() else 0.0
        display_mare = float(normal_rel.max().item()) if normal_rel.numel() else 0.0
    else:
        display_mere = mere
        display_mare = mare

    return {
        "passed": passed,
        "threshold": BF16_THRESHOLD,
        "mere": display_mere,
        "mare": display_mare,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "mismatch_count": mismatch_count,
        "total_count": total_count,
        "small_value_error_count": small_value_error_count,
        "small_value_cpu_error_count": small_value_cpu_error_count,
        "small_value_total_count": small_value_total_count,
        "cancel_error_count": cancel_error_count,
        "cancel_cpu_error_count": cancel_cpu_error_count,
        "cancel_total_count": cancel_total_count,
        "error_msg": "" if passed else "MERE/MARE threshold exceeded",
    }


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def format_latency_us(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f} us"


def format_speedup(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}x"


def speedup(numerator_us: object, denominator_us: object) -> float | None:
    try:
        numerator = float(numerator_us) if numerator_us is not None else 0.0
        denominator = float(denominator_us) if denominator_us is not None else 0.0
    except (TypeError, ValueError):
        return None
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return numerator / denominator


def make_speedup_matrix(
    candidate_active_us: object,
    candidate_kernel_us: object,
    baseline_active_us: object,
    baseline_kernel_us: object,
) -> dict[str, float | None | str]:
    matrix: dict[str, float | None | str] = {
        "active_vs_active": speedup(baseline_active_us, candidate_active_us),
        "kernel_vs_kernel": speedup(baseline_kernel_us, candidate_kernel_us),
        "baseline_active_vs_candidate_kernel": speedup(baseline_active_us, candidate_kernel_us),
        "baseline_kernel_vs_candidate_active": speedup(baseline_kernel_us, candidate_active_us),
    }
    for key, value in list(matrix.items()):
        matrix[f"{key}_text"] = format_speedup(value if isinstance(value, float) else None)
    return matrix


def _profiler_enum(enum_owner, enum_name: str, enum_type: str):
    try:
        return getattr(enum_owner, enum_name)
    except AttributeError as exc:
        available = ", ".join(name for name in dir(enum_owner) if not name.startswith("_"))
        raise ValueError(f"unsupported profiler {enum_type}: {enum_name}; available: {available}") from exc


def _profiler_export_types(export_type: str):
    names = [part.strip() for part in str(export_type or "Text").split(",") if part.strip()]
    if not names:
        raise ValueError("profiler export type must not be empty")
    return [_profiler_enum(torch_npu.profiler.ExportType, name, "export type") for name in names]


def _experimental_config(profiler_level: str, profiler_aic_metrics: str, profiler_export_type: str):
    return torch_npu.profiler._ExperimentalConfig(
        export_type=_profiler_export_types(profiler_export_type),
        profiler_level=_profiler_enum(torch_npu.profiler.ProfilerLevel, profiler_level, "level"),
        aic_metrics=_profiler_enum(torch_npu.profiler.AiCMetrics, profiler_aic_metrics, "aic metrics"),
    )


def _prepare_warmup_tensors(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _WARMUP_TENSORS
    if _WARMUP_TENSORS is None:
        mm1 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
        mm2 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
        reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).to(device)
        _WARMUP_TENSORS = (mm1, mm2, reduce_input)
    return _WARMUP_TENSORS


def _boost_freq_and_clear_cache(device: str) -> None:
    mm1, mm2, reduce_input = _prepare_warmup_tensors(device)
    try:
        torch.matmul(mm1, mm2)
        torch_npu.npu.synchronize()
        torch.max(reduce_input)
        torch_npu.npu.synchronize()
    except RuntimeError:
        torch_npu.npu.synchronize()


def _clear_cache(device: str) -> None:
    _, _, reduce_input = _prepare_warmup_tensors(device)
    try:
        torch.max(reduce_input)
        torch_npu.npu.synchronize()
    except RuntimeError:
        torch_npu.npu.synchronize()


def _is_warmup_kernel(op_type: str, input_shapes: str) -> bool:
    if not op_type or not input_shapes:
        return False
    if op_type == "MatMulV3" and WARMUP_MATMUL_SHAPE in input_shapes:
        return True
    if op_type == "ReduceMax" and WARMUP_REDUCE_SHAPE in input_shapes:
        return True
    return False


def _step_sort_key(step_id: str) -> tuple[int, str]:
    try:
        return (int(step_id), step_id)
    except (TypeError, ValueError):
        return (10**9, str(step_id))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2


def parse_visible_device_timing_csv(csv_path: Path) -> dict[str, object]:
    """Parse visible NPU kernel time and active-window time from kernel_details.csv."""

    step_kernel_times: dict[str, dict[str, list[float]]] = {}
    step_windows: dict[str, dict[str, object]] = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"Step Id", "Name", "Type", "Start Time(us)", "Duration(us)"}
        missing = sorted(required_fields - set(reader.fieldnames or []))
        if missing:
            raise ValueError("kernel_details.csv missing required fields: " + ", ".join(missing))

        for row_number, row in enumerate(reader, start=2):
            step_id = row.get("Step Id", "").strip()
            if not step_id:
                raise ValueError(f"kernel_details.csv has blank Step Id at row {row_number}")
            try:
                duration = float(row.get("Duration(us)", "0"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"malformed Duration(us) for step {step_id}: {row.get('Duration(us)', '')!r}") from exc
            if duration <= 0:
                continue
            try:
                start_time = float(str(row.get("Start Time(us)", "")).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"malformed Start Time(us) for step {step_id}: {row.get('Start Time(us)', '')!r}") from exc

            op_type = row.get("Type", "")
            input_shapes = row.get("Input Shapes", "")
            name = row.get("Name", op_type)
            if _is_warmup_kernel(op_type, input_shapes):
                continue

            step_kernel_times.setdefault(step_id, {}).setdefault(name, []).append(duration)
            window = step_windows.setdefault(
                step_id,
                {
                    "start_us": start_time,
                    "end_us": start_time + duration,
                    "kernel_duration_sum_us": 0.0,
                    "kernel_count": 0,
                },
            )
            window["start_us"] = min(float(window["start_us"]), start_time)
            window["end_us"] = max(float(window["end_us"]), start_time + duration)
            window["kernel_duration_sum_us"] = float(window["kernel_duration_sum_us"]) + duration
            window["kernel_count"] = int(window["kernel_count"]) + 1

    if not step_kernel_times:
        raise ValueError("kernel_details.csv contains no measured visible NPU kernels")

    for window in step_windows.values():
        active_window = max(float(window["end_us"]) - float(window["start_us"]), 0.0)
        kernel_sum = float(window["kernel_duration_sum_us"])
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

    device_kernels: dict[str, float] = {}
    kernel_duration_sum_us = 0.0
    for name, times in all_kernel_times.items():
        median_time = _median(times)
        device_kernels[name] = round(median_time, 2)
        kernel_duration_sum_us += median_time

    active_window_us = _median([float(window["device_active_window_us"]) for window in step_windows.values()])
    median_step_kernel_sum_us = _median([float(window["kernel_duration_sum_us"]) for window in step_windows.values()])
    kernel_counts = sorted({int(window["kernel_count"]) for window in step_windows.values()})

    return {
        "device_kernels": device_kernels,
        "device_kernel_duration_sum_us": round(kernel_duration_sum_us, 2),
        "median_step_kernel_duration_sum_us": round(median_step_kernel_sum_us, 2),
        "device_active_window_us": round(active_window_us, 2),
        "device_window_gap_us": round(max(active_window_us - median_step_kernel_sum_us, 0.0), 2),
        "step_windows": dict(sorted(step_windows.items(), key=lambda item: _step_sort_key(item[0]))),
        "measured_step_count": len(step_windows),
        "kernel_count_pattern": kernel_counts,
    }


def parse_cannbench_timing_csv(csv_path: Path, strategy: str) -> dict[str, object]:
    if strategy == "candidate_kernel_details":
        parsed = parse_visible_device_timing_csv(csv_path)
        return {
            "latency_us": parsed["device_active_window_us"],
            "active_window_us": parsed["device_active_window_us"],
            "kernel_sum_us": parsed["device_kernel_duration_sum_us"],
            "window_gap_us": parsed["device_window_gap_us"],
            "kernel_count": max(parsed["kernel_count_pattern"]) if parsed["kernel_count_pattern"] else None,
            "step_count": parsed["measured_step_count"],
            "device_kernels": parsed["device_kernels"],
            "device_timeline": {
                "device_active_window_us": parsed["device_active_window_us"],
                "device_kernel_duration_sum_us": parsed["device_kernel_duration_sum_us"],
                "median_step_kernel_duration_sum_us": parsed["median_step_kernel_duration_sum_us"],
                "device_window_gap_us": parsed["device_window_gap_us"],
                "measured_step_count": parsed["measured_step_count"],
                "kernel_count_pattern": parsed["kernel_count_pattern"],
                "step_windows": parsed["step_windows"],
            },
            "perf_metric_strategy": "kernel_details",
            "measurement_scope": "visible_device_active_window",
            "elapsed_us_source": "kernel_details.active_window_us",
            "kernel_sum_elapsed_us_source": "kernel_details.kernel_sum_us",
        }
    if strategy == "baseline_active_window":
        parsed = parse_visible_device_timing_csv(csv_path)
        return {
            "latency_us": parsed["device_active_window_us"],
            "active_window_us": parsed["device_active_window_us"],
            "kernel_sum_us": parsed["device_kernel_duration_sum_us"],
            "window_gap_us": parsed["device_window_gap_us"],
            "kernel_count": max(parsed["kernel_count_pattern"]) if parsed["kernel_count_pattern"] else None,
            "step_count": parsed["measured_step_count"],
            "device_kernels": parsed["device_kernels"],
            "device_timeline": {
                "device_active_window_us": parsed["device_active_window_us"],
                "device_kernel_duration_sum_us": parsed["device_kernel_duration_sum_us"],
                "median_step_kernel_duration_sum_us": parsed["median_step_kernel_duration_sum_us"],
                "device_window_gap_us": parsed["device_window_gap_us"],
                "measured_step_count": parsed["measured_step_count"],
                "kernel_count_pattern": parsed["kernel_count_pattern"],
                "step_windows": parsed["step_windows"],
            },
            "perf_metric_strategy": "baseline_active_window",
            "measurement_scope": "visible_device_active_window",
            "elapsed_us_source": "baseline_active_window.device_active_window_us",
            "kernel_sum_elapsed_us_source": "baseline_active_window.device_kernel_duration_sum_us",
        }
    raise ValueError(f"unsupported timing strategy: {strategy}")


def _file_snapshot(root: Path) -> tuple[tuple[str, int, int], ...]:
    snapshot = []
    if not root.is_dir():
        return tuple()
    for path in root.rglob("*"):
        if not path.is_file() or path.name.endswith(".done"):
            continue
        stat = path.stat()
        snapshot.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(snapshot))


def _wait_profiler_files_ready(root: Path, timeout_s: float = 10.0) -> None:
    start = time.monotonic()
    last_snapshot = None
    stable_since = None
    while True:
        snapshot = _file_snapshot(root)
        now = time.monotonic()
        if snapshot and snapshot == last_snapshot:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= 0.2:
                return
        else:
            last_snapshot = snapshot
            stable_since = now if snapshot else None
        if now - start >= timeout_s:
            return
        time.sleep(0.2)


def _locate_profiler_files(root: Path) -> tuple[Path | None, Path | None]:
    csv_path = None
    trace_view_path = None
    for path in root.rglob("kernel_details.csv"):
        csv_path = path
        break
    for path in root.rglob("trace_view.json"):
        trace_view_path = path
        break
    return csv_path, trace_view_path


def _close_profiler_pool() -> None:
    try:
        from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool

        MultiProcessPool().close_pool(wait=True)
    except Exception:
        pass


def profile_once(
    label: str,
    case_id: str,
    func: Callable[[], torch.Tensor],
    *,
    timing_strategy: str,
    device: str,
    warmup: int,
    repeat: int,
    profiler_root: Path,
    profiler_level: str,
    profiler_aic_metrics: str,
    profiler_export_type: str,
    freq_boost: bool,
) -> tuple[torch.Tensor | None, ProfileResult]:
    """Profile one callable and parse it using the matching CANN-Bench strategy."""

    prof_dir = profiler_root / case_id / label
    if prof_dir.exists():
        shutil.rmtree(prof_dir)
    prof_dir.mkdir(parents=True, exist_ok=True)

    last_output: torch.Tensor | None = None
    try:
        if freq_boost:
            _boost_freq_and_clear_cache(device)

        # CANN-Bench pre-flight: fail before opening profiler if the op cannot run.
        last_output = func()
        torch_npu.npu.synchronize()

        experimental_config = _experimental_config(profiler_level, profiler_aic_metrics, profiler_export_type)
        original_basic_config = logging.basicConfig
        logging.basicConfig = lambda **kw: original_basic_config(**{**kw, "level": logging.ERROR, "force": True})
        for logger_name in ["", "torch", "torch_npu", "torch_npu.profiler", "ascend", "profiler"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            logger.handlers = []
            logger.addHandler(logging.NullHandler())

        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        sink = tempfile.NamedTemporaryFile(mode="w+", prefix="add_rms_norm_profiler_", suffix=".log", delete=False)
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
                pending_exc: BaseException | None = None
                for i in range(warmup + repeat):
                    if freq_boost and i >= warmup:
                        _clear_cache(device)
                    try:
                        last_output = func()
                    except BaseException as exc:
                        pending_exc = exc
                        prof.step()
                        break
                    prof.step()
                if pending_exc is not None:
                    raise pending_exc
            time.sleep(0.1)
            _close_profiler_pool()
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            logging.basicConfig = original_basic_config
            sink.close()
            try:
                os.unlink(sink.name)
            except OSError:
                pass

        _wait_profiler_files_ready(prof_dir)
        csv_path, trace_view_path = _locate_profiler_files(prof_dir)
        if csv_path is None:
            return last_output, ProfileResult(
                None,
                None,
                None,
                None,
                None,
                None,
                {},
                {},
                None,
                str(trace_view_path) if trace_view_path else None,
                "kernel_details.csv not found",
                timing_strategy,
            )
        timing_data = parse_cannbench_timing_csv(csv_path, timing_strategy)
        latency = float(timing_data.get("latency_us") or 0.0)
        if latency <= 0:
            return last_output, ProfileResult(
                None,
                None,
                None,
                None,
                None,
                None,
                timing_data.get("device_kernels", {}),
                timing_data.get("device_timeline", {}),
                str(csv_path),
                str(trace_view_path) if trace_view_path else None,
                "parsed latency_us is non-positive",
                timing_data.get("perf_metric_strategy", timing_strategy),
                timing_data.get("measurement_scope", ""),
                timing_data.get("elapsed_us_source", ""),
            )
        return last_output, ProfileResult(
            latency,
            float(timing_data["active_window_us"]) if timing_data.get("active_window_us") is not None else None,
            float(timing_data["kernel_sum_us"]) if timing_data.get("kernel_sum_us") is not None else None,
            float(timing_data["window_gap_us"]) if timing_data.get("window_gap_us") is not None else None,
            int(timing_data["kernel_count"]) if timing_data.get("kernel_count") is not None else None,
            int(timing_data["step_count"]) if timing_data.get("step_count") is not None else None,
            timing_data.get("device_kernels", {}),
            timing_data.get("device_timeline", {}),
            str(csv_path),
            str(trace_view_path) if trace_view_path else None,
            None,
            str(timing_data.get("perf_metric_strategy", timing_strategy)),
            str(timing_data.get("measurement_scope", "")),
            str(timing_data.get("elapsed_us_source", "")),
        )
    except Exception as exc:
        return last_output, ProfileResult(None, None, None, None, None, None, {}, {}, None, None,
                                          f"{type(exc).__name__}: {exc}", timing_strategy)


def _path_key(path: str) -> str:
    return path.replace("/", "_").replace(" ", "_")


def run_case(
    case: Case,
    index: int,
    device: str,
    benchmark: bool,
    warmup: int,
    repeat: int,
    profiler_root: Path,
    profiler_level: str,
    profiler_aic_metrics: str,
    profiler_export_type: str,
    freq_boost: bool,
) -> dict[str, object]:
    case_id = case.case_id or _cannbench_case_id(index)
    case_seed = case.case_seed if case.case_seed is not None else _cannbench_case_seed(index)
    x1, x2, gamma = make_inputs(case.shape, case_seed, device, case.value_ranges)
    cpu_x1 = x1.cpu()
    cpu_x2 = x2.cpu()
    cpu_gamma = gamma.cpu()
    expected = add_rms_norm_cannbench_golden(
        cpu_x1.to(torch.float64),
        cpu_x2.to(torch.float64),
        cpu_gamma.to(torch.float64),
    )
    native_expected = add_rms_norm_cannbench_golden(cpu_x1, cpu_x2, cpu_gamma)

    implementations: dict[str, dict[str, object]] = {
        "triton": {
            "fn": lambda: add_rms_norm(x1, x2, gamma),
            "timing_strategy": "candidate_kernel_details",
            "role": "candidate",
        },
        "torch": {
            "fn": lambda: add_rms_norm_reference(x1, x2, gamma),
            "timing_strategy": "baseline_active_window",
            "role": "pytorch_semantic_baseline",
        },
        "torch_npu": {
            "fn": lambda: add_rms_norm_torch_npu(x1, x2, gamma),
            "timing_strategy": "baseline_active_window",
            "role": "task_npu_baseline_probe",
        },
    }

    result: dict[str, object] = {
        "case":
        index,
        "case_id":
        case_id,
        "case_seed":
        case_seed,
        "kind":
        case.kind,
        "shape":
        list(case.shape),
        "dtype":
        "bfloat16",
        "threshold":
        BF16_THRESHOLD,
        "value_range": [list(item) for item in case.value_ranges],
        "timing_policy": ("Triton candidate uses CANN-Bench KernelDetailsStrategy; "
                          "Torch semantic and torch_npu baselines use custom-baseline BaselineActiveWindowStrategy."),
        "implementations": {},
    }
    if case.audit_seed is not None:
        result["audit_seed"] = case.audit_seed
    if case.shape_policy:
        result["shape_policy"] = case.shape_policy
    if case.random_category:
        result["random_category"] = case.random_category
    if case.note:
        result["note"] = case.note

    for name, spec in implementations.items():
        func = spec["fn"]
        assert callable(func)
        output = None
        profile = ProfileResult(None, None, None, None, None, None, {}, {}, None, None, None)
        if benchmark:
            output, profile = profile_once(
                name,
                f"case_{index:03d}",
                func,
                timing_strategy=str(spec["timing_strategy"]),
                device=device,
                warmup=warmup,
                repeat=repeat,
                profiler_root=profiler_root,
                profiler_level=profiler_level,
                profiler_aic_metrics=profiler_aic_metrics,
                profiler_export_type=profiler_export_type,
                freq_boost=freq_boost,
            )
        else:
            try:
                output = func()
                torch_npu.npu.synchronize()
            except Exception as exc:
                profile.error = f"{type(exc).__name__}: {exc}"

        if output is not None:
            accuracy = cannbench_bf16_compare(output, expected, native_expected)
        else:
            accuracy = {
                "passed": False,
                "threshold": BF16_THRESHOLD,
                "mere": 0.0,
                "mare": 0.0,
                "max_diff": 0.0,
                "mean_diff": 0.0,
                "mismatch_count": int(x1.numel()),
                "total_count": int(x1.numel()),
                "error_msg": profile.error or "no output",
            }

        result["implementations"][name] = {
            "accuracy": accuracy,
            "role": spec["role"],
            "timing_strategy": spec["timing_strategy"],
            "perf_metric_strategy": profile.perf_metric_strategy,
            "measurement_scope": profile.measurement_scope,
            "elapsed_us_source": profile.elapsed_us_source,
            "primary_latency_us": profile.latency_us,
            "primary_latency": format_latency_us(profile.latency_us),
            "latency_us": profile.latency_us,
            "latency": format_latency_us(profile.latency_us),
            "active_window_us": profile.active_window_us,
            "active_window": format_latency_us(profile.active_window_us),
            "kernel_sum_us": profile.kernel_sum_us,
            "kernel_sum": format_latency_us(profile.kernel_sum_us),
            "window_gap_us": profile.window_gap_us,
            "window_gap": format_latency_us(profile.window_gap_us),
            "kernel_count": profile.kernel_count,
            "step_count": profile.step_count,
            "device_kernels": profile.device_kernels,
            "device_timeline": profile.device_timeline,
            "profiler_csv_path": profile.csv_path,
            "profiler_trace_view_path": profile.trace_view_path,
            "profile_error": profile.error,
        }

    impls = result["implementations"]
    triton_active = impls["triton"]["active_window_us"]
    triton_kernel = impls["triton"]["kernel_sum_us"]
    for base_name in ["torch", "torch_npu"]:
        base_active = impls[base_name]["active_window_us"]
        base_kernel = impls[base_name]["kernel_sum_us"]
        impls[base_name]["speedups_vs_triton"] = make_speedup_matrix(
            triton_active,
            triton_kernel,
            base_active,
            base_kernel,
        )
        impls[base_name]["speedup_vs_triton"] = impls[base_name]["speedups_vs_triton"]["active_vs_active"]
        impls[base_name]["speedup_vs_triton_text"] = impls[base_name]["speedups_vs_triton"]["active_vs_active_text"]
    torch_npu_available = bool(impls["torch_npu"]["accuracy"]["passed"] and impls["torch_npu"]["active_window_us"])
    selected_name = "torch_npu" if torch_npu_available else "torch"
    selected_speedups = impls[selected_name]["speedups_vs_triton"]
    result["selected_baseline"] = {
        "implementation":
        selected_name,
        "source":
        "task_npu_baseline_probe" if selected_name == "torch_npu" else "pytorch_semantic_baseline",
        "selection_rule": ("Use torch_npu.npu_add_rms_norm only when its output passes the same "
                           "BF16 CANN-Bench precision check; otherwise use the Torch semantic baseline."),
        "active_window_us":
        impls[selected_name]["active_window_us"],
        "active_window":
        impls[selected_name]["active_window"],
        "kernel_sum_us":
        impls[selected_name]["kernel_sum_us"],
        "kernel_sum":
        impls[selected_name]["kernel_sum"],
        "latency_us":
        impls[selected_name]["primary_latency_us"],
        "latency":
        impls[selected_name]["primary_latency"],
        "speedup_vs_triton":
        selected_speedups["active_vs_active"],
        "speedup_vs_triton_text":
        selected_speedups["active_vs_active_text"],
        "speedups_vs_triton":
        selected_speedups,
        "perf_metric_strategy":
        impls[selected_name]["perf_metric_strategy"],
        "measurement_scope":
        impls[selected_name]["measurement_scope"],
        "elapsed_us_source":
        impls[selected_name]["elapsed_us_source"],
    }
    return result


def print_case(record: dict[str, object], total: int) -> None:
    impls = record["implementations"]
    triton = impls["triton"]
    torch_impl = impls["torch"]
    torch_npu_impl = impls["torch_npu"]
    triton_acc = triton["accuracy"]
    status = "PASS" if triton_acc["passed"] else "FAIL"
    print(
        f"[{status}] {record['case']:03d}/{total:03d} {record['kind']} shape={tuple(record['shape'])} "
        f"triton_active={triton['active_window']} triton_kernel={triton['kernel_sum']} "
        f"torch_active={torch_impl['active_window']} torch_kernel={torch_impl['kernel_sum']} "
        f"torch_npu_active={torch_npu_impl['active_window']} torch_npu_kernel={torch_npu_impl['kernel_sum']} "
        f"speedup_vs_torch_active={torch_impl['speedups_vs_triton']['active_vs_active_text']} "
        f"speedup_vs_torch_kernel={torch_impl['speedups_vs_triton']['kernel_vs_kernel_text']} "
        f"speedup_vs_torch_baseline_active_candidate_kernel={torch_impl['speedups_vs_triton']['baseline_active_vs_candidate_kernel_text']} "
        f"speedup_vs_torch_baseline_kernel_candidate_active={torch_impl['speedups_vs_triton']['baseline_kernel_vs_candidate_active_text']} "
        f"speedup_vs_torch_npu_active={torch_npu_impl['speedups_vs_triton']['active_vs_active_text']} "
        f"speedup_vs_torch_npu_kernel={torch_npu_impl['speedups_vs_triton']['kernel_vs_kernel_text']} "
        f"triton_accuracy={'PASS' if triton_acc['passed'] else 'FAIL'} "
        f"torch_accuracy={'PASS' if torch_impl['accuracy']['passed'] else 'FAIL'} "
        f"torch_npu_accuracy={'PASS' if torch_npu_impl['accuracy']['passed'] else 'FAIL'} "
        f"MERE={triton_acc['mere']:.3e} MARE={triton_acc['mare']:.3e} "
        f"max_diff={triton_acc['max_diff']:.3e} "
        f"triton_source={triton['elapsed_us_source']} "
        f"triton_kernel_source=kernel_details.kernel_sum_us "
        f"torch_source={torch_impl['elapsed_us_source']} "
        f"torch_kernel_source=baseline_active_window.device_kernel_duration_sum_us "
        f"torch_npu_source={torch_npu_impl['elapsed_us_source']} "
        f"torch_npu_kernel_source=baseline_active_window.device_kernel_duration_sum_us")


def summarize(records: list[dict[str, object]]) -> None:
    passed = sum(1 for r in records if r["implementations"]["triton"]["accuracy"]["passed"])
    torch_npu_acc_pass = sum(1 for r in records if r["implementations"]["torch_npu"]["accuracy"]["passed"])
    torch_acc_pass = sum(1 for r in records if r["implementations"]["torch"]["accuracy"]["passed"])
    print(
        f"SUMMARY passed={passed}/{len(records)} threshold={BF16_THRESHOLD:.8f} "
        f"torch_accuracy_pass={torch_acc_pass}/{len(records)} "
        f"torch_npu_accuracy_pass={torch_npu_acc_pass}/{len(records)} "
        f"selected_task_npu_baseline={torch_npu_acc_pass} selected_torch_fallback={len(records) - torch_npu_acc_pass} "
        f"triton_strategy=kernel_details triton_scope=visible_device_active_window "
        f"baseline_strategy=baseline_active_window baseline_scope=visible_device_active_window")
    kinds = sorted({str(r["kind"]) for r in records})
    for kind in kinds:
        kind_records = [r for r in records if r["kind"] == kind]
        kind_passed = sum(1 for r in kind_records if r["implementations"]["triton"]["accuracy"]["passed"])
        print(f"SUMMARY_KIND kind={kind} passed={kind_passed}/{len(kind_records)}")

    def collect_impl(name: str, key: str, *, require_pass: bool = False) -> list[float]:
        values: list[float] = []
        for record in records:
            impl = record["implementations"][name]
            if require_pass and not impl["accuracy"]["passed"]:
                continue
            value = impl.get(key)
            if value is not None:
                values.append(float(value))
        return values

    def collect_speedups(base_name: str, key: str, *, require_pass: bool = False) -> list[float]:
        values: list[float] = []
        for record in records:
            impl = record["implementations"][base_name]
            if require_pass and not impl["accuracy"]["passed"]:
                continue
            value = impl["speedups_vs_triton"].get(key)
            if value is not None:
                values.append(float(value))
        return values

    def collect_selected(key: str) -> list[float]:
        values: list[float] = []
        for record in records:
            selected = record.get("selected_baseline")
            if selected:
                value = selected["speedups_vs_triton"].get(key)
                if value is not None:
                    values.append(float(value))
        return values

    triton_active = collect_impl("triton", "active_window_us")
    triton_kernel = collect_impl("triton", "kernel_sum_us")
    torch_active = collect_impl("torch", "active_window_us")
    torch_kernel = collect_impl("torch", "kernel_sum_us")
    torch_npu_active = collect_impl("torch_npu", "active_window_us")
    torch_npu_kernel = collect_impl("torch_npu", "kernel_sum_us")
    if triton_active:
        print(f"LATENCY_ACTIVE triton_geomean={geomean(triton_active):.3f} us "
              f"triton_min={min(triton_active):.3f} us triton_max={max(triton_active):.3f} us "
              f"torch_geomean={geomean(torch_active):.3f} us "
              f"torch_npu_geomean={geomean(torch_npu_active):.3f} us")
        print(f"LATENCY_KERNEL triton_geomean={geomean(triton_kernel):.3f} us "
              f"triton_min={min(triton_kernel):.3f} us triton_max={max(triton_kernel):.3f} us "
              f"torch_geomean={geomean(torch_kernel):.3f} us "
              f"torch_npu_geomean={geomean(torch_npu_kernel):.3f} us")
        print(
            "SPEEDUP_MATRIX "
            f"torch_active_vs_active={format_speedup(geomean(collect_speedups('torch', 'active_vs_active')))} "
            f"torch_kernel_vs_kernel={format_speedup(geomean(collect_speedups('torch', 'kernel_vs_kernel')))} "
            f"torch_baseline_active_vs_candidate_kernel={format_speedup(geomean(collect_speedups('torch', 'baseline_active_vs_candidate_kernel')))} "
            f"torch_baseline_kernel_vs_candidate_active={format_speedup(geomean(collect_speedups('torch', 'baseline_kernel_vs_candidate_active')))} "
            f"torch_npu_pass_active_vs_active={format_speedup(geomean(collect_speedups('torch_npu', 'active_vs_active', require_pass=True)))} "
            f"torch_npu_pass_kernel_vs_kernel={format_speedup(geomean(collect_speedups('torch_npu', 'kernel_vs_kernel', require_pass=True)))} "
            f"selected_active_vs_active={format_speedup(geomean(collect_selected('active_vs_active')))} "
            f"selected_kernel_vs_kernel={format_speedup(geomean(collect_selected('kernel_vs_kernel')))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AddRmsNorm Triton-Ascend implementation.")
    parser.add_argument("--device", default="npu", help="Torch device, default: npu")
    parser.add_argument("--public", action="store_true", help="Run all 80 public B/S/H cases.")
    parser.add_argument("--random-generalization", type=int, default=0,
                        help="Run N seeded random non-public shape cases.")
    parser.add_argument("--random-seed", type=int, default=20260613,
                        help="Seed for random generalization shapes and values.")
    parser.add_argument("--random-shape-policy", default=RANDOM_GENERALIZATION_POLICY,
                        help="Random generalization shape policy.")
    parser.add_argument("--benchmark", action="store_true", help="Use CANN-Bench-style profiler timing.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases for smoke checks.")
    parser.add_argument("--jsonl", type=Path, default=None, help="Write per-case machine-readable results.")
    parser.add_argument("--profiler-data-dir", type=Path, default=Path("logs/prof_data"))
    parser.add_argument("--profiler-level", default="Level1")
    parser.add_argument("--profiler-aic-metrics", default="PipeUtilization")
    parser.add_argument("--profiler-export-type", default="Text")
    parser.add_argument("--no-freq-boost", action="store_true", help="Disable CANN-Bench-style freq/cache warmup.")
    args = parser.parse_args()

    if torch_npu is None:
        raise RuntimeError("torch_npu is required for NPU validation")
    if args.random_generalization < 0:
        parser.error("--random-generalization must be non-negative")

    selected: list[Case] = []
    if args.public or args.random_generalization == 0:
        selected.extend(public_cases())
    if args.random_generalization:
        selected.extend(
            random_generalization_cases(
                args.random_generalization,
                args.random_seed,
                policy=args.random_shape_policy,
            ))
    if args.max_cases is not None:
        selected = selected[:args.max_cases]

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("")
    if args.benchmark:
        args.profiler_data_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, case in enumerate(selected, start=1):
        record = run_case(
            case,
            index,
            args.device,
            args.benchmark,
            args.warmup,
            args.repeat,
            args.profiler_data_dir,
            args.profiler_level,
            args.profiler_aic_metrics,
            args.profiler_export_type,
            not args.no_freq_boost,
        )
        records.append(record)
        print_case(record, len(selected))
        if args.jsonl:
            with args.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summarize(records)


if __name__ == "__main__":
    main()
