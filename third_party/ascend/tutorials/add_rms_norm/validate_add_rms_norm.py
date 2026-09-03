# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-contained validation for the AddRmsNorm tutorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

try:
    import torch_npu
except Exception:  # pragma: no cover - depends on Ascend runtime.
    torch_npu = None

from add_rms_norm import add_rms_norm, add_rms_norm_reference
from profiler_timing import ProfileResult, profile_kernel_details

PUBLIC_TEST_STANDARD_SHAPES = [
    (1, 3584, 28, 128),
    (1, 4096, 32, 128),
    (1, 5120, 40, 128),
    (1, 8192, 64, 128),
    (8, 3584, 28, 128),
    (8, 4096, 32, 128),
    (8, 5120, 40, 128),
    (8, 8192, 64, 128),
    (16, 3584, 28, 128),
    (16, 4096, 32, 128),
    (16, 5120, 40, 128),
    (16, 8192, 64, 128),
    (32, 3584, 28, 128),
    (32, 4096, 32, 128),
    (32, 5120, 40, 128),
    (32, 8192, 64, 128),
    (64, 3584, 28, 128),
    (64, 4096, 32, 128),
    (64, 5120, 40, 128),
    (64, 8192, 64, 128),
]

DEFAULT_VALUE_RANGES = ((-1.0, 1.0), (-1.0, 1.0), (0.5, 1.5))
RANDOM_GENERALIZATION_POLICY = "seeded_non_public_docs_standard_shapes_v3"
RANDOM_MAX_ELEMENTS = 8 * 1024 * 1024
RANDOM_VALUE_RANGE_CHOICES = (
    ((-0.01, 0.01), (-0.01, 0.01), (0.5, 1.5)),
    ((-1.0, 1.0), (-1.0, 1.0), (0.5, 1.5)),
    ((-8.0, 8.0), (-8.0, 8.0), (0.25, 1.75)),
    ((-64.0, 64.0), (-16.0, 16.0), (0.125, 2.0)),
)
BF16_THRESHOLD = 2**-7
BF16_SMALL_VALUE_THRESHOLD = 2**-8
BF16_SMALL_VALUE_ERROR = 2**-16
BF16_CANCEL_BOUNDARY = 2**-3
BF16_CANCEL_ZERO_THRESHOLD = 2**-3


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
    head_num: int = 0
    head_dim: int = 128

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.bsz, self.seq, self.hidden)


def public_cases() -> list[Case]:
    return [
        Case(bsz, 1, hidden, head_num=head_num, head_dim=head_dim)
        for bsz, hidden, head_num, head_dim in PUBLIC_TEST_STANDARD_SHAPES
    ]


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
    seen: set[tuple[tuple[int, int, int], tuple[tuple[float, float], ...]]] = set()
    cases: list[Case] = []
    attempts = 0
    max_attempts = max(200, count * 200)

    while len(cases) < count and attempts < max_attempts:
        attempts += 1
        bsz, hidden, head_num, head_dim = PUBLIC_TEST_STANDARD_SHAPES[(rng.randrange(len(PUBLIC_TEST_STANDARD_SHAPES)) +
                                                                       attempts) % len(PUBLIC_TEST_STANDARD_SHAPES)]
        seq = 1
        value_ranges = RANDOM_VALUE_RANGE_CHOICES[(rng.randrange(len(RANDOM_VALUE_RANGE_CHOICES)) + attempts) %
                                                  len(RANDOM_VALUE_RANGE_CHOICES)]

        shape = (bsz, seq, hidden)
        key = (shape, value_ranges)
        if key in seen:
            continue
        if bsz * seq * hidden > RANDOM_MAX_ELEMENTS:
            continue

        seen.add(key)
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
                random_category="docs_standard_value_variation",
                note="seeded docs-standard shape with non-public value sample",
                value_ranges=value_ranges,
                head_num=head_num,
                head_dim=head_dim,
            ))

    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} random cases after {attempts} attempts, expected {count}")
    return cases


def _local_case_id(index: int) -> str:
    return f"custom/add_rms_norm_{index}"


def _seed_from_case_id(case_id: str, eval_seed: int = 0) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    deterministic_hash = int.from_bytes(digest[:8], byteorder="big") % (2**31)
    return (int(eval_seed) + deterministic_hash) % (2**31)


def _local_case_seed(index: int, eval_seed: int = 0) -> int:
    return _seed_from_case_id(_local_case_id(index), eval_seed)


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


def add_rms_norm_torch_reference(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Local Torch semantic reference for AddRmsNorm."""

    z = x1.to(torch.float32) + x2.to(torch.float32)
    variance = torch.mean(z * z, dim=-1, keepdim=True)
    y = z * torch.rsqrt(variance + float(epsilon)) * gamma.to(torch.float32)
    return y.to(dtype=x1.dtype)


def local_bf16_compare(
    actual: torch.Tensor,
    reference: torch.Tensor,
    native_output: torch.Tensor | None = None,
) -> dict[str, object]:
    """Local BF16 precision checker with MERE, MARE, RMSE, small-value, and cancellation evidence."""

    if actual.device.type != "cpu":
        actual = actual.cpu()
    if reference.device.type != "cpu":
        reference = reference.cpu()
    if native_output is not None and native_output.device.type != "cpu":
        native_output = native_output.cpu()
    if actual.shape != reference.shape:
        return {
            "passed": False,
            "threshold": BF16_THRESHOLD,
            "mere": 0.0,
            "mare": 0.0,
            "rmse": 0.0,
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "mismatch_count": int(actual.numel()),
            "total_count": int(actual.numel()),
            "error_msg": f"shape mismatch: actual={tuple(actual.shape)} reference={tuple(reference.shape)}",
        }

    target_dtype = actual.dtype
    actual64 = actual.to(torch.float64)
    reference_truncated = reference.to(target_dtype).to(torch.float64)

    if torch.any(torch.isnan(actual64)) or torch.any(torch.isnan(reference_truncated)):
        if not torch.all(torch.isnan(actual64) == torch.isnan(reference_truncated)):
            return {
                "passed": False,
                "threshold": BF16_THRESHOLD,
                "mere": 0.0,
                "mare": 0.0,
                "rmse": 0.0,
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
    if torch.any(torch.isinf(actual64)) or torch.any(torch.isinf(reference_truncated)):
        inf_out = torch.isinf(actual64)
        inf_gold = torch.isinf(reference_truncated)
        inf_mismatch = inf_out != inf_gold
        if torch.any(inf_mismatch):
            max_finite = float(torch.finfo(target_dtype).max)
            if torch.any(inf_out & ~inf_gold):
                mask = inf_out & ~inf_gold
                actual64[mask] = torch.sign(actual64[mask]) * max_finite
            if torch.any(inf_gold & ~inf_out):
                mask = inf_gold & ~inf_out
                reference_truncated[mask] = torch.sign(reference_truncated[mask]) * max_finite
        both_inf = inf_out & inf_gold
        if torch.any(both_inf):
            if not torch.all(torch.sign(actual64[both_inf]) == torch.sign(reference_truncated[both_inf])):
                return {
                    "passed": False,
                    "threshold": BF16_THRESHOLD,
                    "mere": 0.0,
                    "mare": 0.0,
                    "rmse": 0.0,
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

    diff = torch.abs(actual64 - reference_truncated)
    reference_abs = torch.abs(reference_truncated)
    rel = diff / (reference_abs + 1e-7)
    valid = ~(torch.isnan(rel) | torch.isinf(rel) | inf_match_mask)
    total_count = int(actual.numel())
    if valid.any().item():
        valid_rel = rel[valid]
        valid_diff = diff[valid]
        mere = float(valid_rel.mean().item())
        mare = float(valid_rel.max().item())
        max_diff = float(valid_diff.max().item())
        mean_diff = float(valid_diff.mean().item())
        rmse = float(torch.sqrt(torch.mean(valid_diff * valid_diff)).item())
    else:
        mere = 0.0
        mare = 0.0
        max_diff = 0.0
        mean_diff = 0.0
        rmse = 0.0

    mare_threshold = 10 * BF16_THRESHOLD
    if mere < BF16_THRESHOLD and mare < mare_threshold:
        return {
            "passed": True,
            "threshold": BF16_THRESHOLD,
            "mere": mere,
            "mare": mare,
            "rmse": rmse,
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

    small_value_mask = (reference_abs < BF16_SMALL_VALUE_THRESHOLD) & valid
    small_value_total_count = int(small_value_mask.sum().item())
    small_value_error_mask = small_value_mask & (diff > BF16_SMALL_VALUE_ERROR)
    small_value_error_count = int(small_value_error_mask.sum().item())

    native64 = (native_output.to(torch.float64)
                if native_output is not None else reference.to(target_dtype).to(torch.float64))
    cpu_diff = torch.abs(native64 - reference_truncated)
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
                   & (reference_abs < BF16_CANCEL_BOUNDARY)
                   & (reference_abs >= BF16_SMALL_VALUE_THRESHOLD)
                   & valid)
    cancel_total_count = int(cancel_mask.sum().item())
    cancel_error_mask = cancel_mask & (rel > mare_threshold)
    cancel_error_count = int(cancel_error_mask.sum().item())
    cpu_relative_error = cpu_diff / (reference_abs + 1e-7)
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
        "rmse": rmse,
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


def geomean_or_none(values: list[float]) -> float | None:
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values)) if values else None


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


def run_case(
    case: Case,
    index: int,
    device: str,
    benchmark: bool,
    benchmark_torch: bool,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    case_id = case.case_id or _local_case_id(index)
    case_seed = case.case_seed if case.case_seed is not None else _local_case_seed(index)
    x1, x2, gamma = make_inputs(case.shape, case_seed, device, case.value_ranges)
    cpu_x1 = x1.cpu()
    cpu_x2 = x2.cpu()
    cpu_gamma = gamma.cpu()
    expected = add_rms_norm_torch_reference(
        cpu_x1.to(torch.float64),
        cpu_x2.to(torch.float64),
        cpu_gamma.to(torch.float64),
    )
    native_expected = add_rms_norm_torch_reference(cpu_x1, cpu_x2, cpu_gamma)

    implementations: dict[str, dict[str, object]] = {
        "triton": {
            "fn": lambda: add_rms_norm(x1, x2, gamma),
            "timing_strategy": "kernel_details",
            "role": "candidate",
            "profile_for_speed": True,
        },
        "torch": {
            "fn": lambda: add_rms_norm_reference(x1, x2, gamma),
            "timing_strategy": "kernel_details" if benchmark_torch else "correctness_only",
            "role": "pytorch_semantic_baseline",
            "profile_for_speed": benchmark_torch,
        },
        "torch_npu": {
            "fn": lambda: add_rms_norm_torch_npu(x1, x2, gamma),
            "timing_strategy": "kernel_details",
            "role": "task_npu_baseline_probe",
            "profile_for_speed": True,
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
        "attrs": {
            "Batch": case.bsz,
            "SequenceLength": case.seq,
            "HiddenSize": case.hidden,
            "HeadNum": case.head_num,
            "HeadDim": case.head_dim,
        },
        "case_detail": {
            "Batch": case.bsz,
            "SequenceLength": case.seq,
            "HiddenSize": case.hidden,
            "HeadNum": case.head_num,
            "HeadDim": case.head_dim,
        },
        "dtype":
        "bfloat16",
        "threshold":
        BF16_THRESHOLD,
        "value_range": [list(item) for item in case.value_ranges],
        "timing_policy": ("Candidate and torch_npu use the OpForge/CANN-Bench kernel-details active-window "
                          "contract when --benchmark is enabled; Torch semantic reference is timed only with "
                          "--benchmark-torch as an auxiliary comparison and is not the main speed gate."),
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
        should_profile = bool(benchmark and spec.get("profile_for_speed"))
        if should_profile:
            output, profile = profile_kernel_details(
                name,
                case_id,
                func,
                warmup=warmup,
                repeat=repeat,
            )
        else:
            try:
                output = func()
                torch_npu.npu.synchronize()
            except Exception as exc:
                profile.error = f"{type(exc).__name__}: {exc}"

        if output is not None:
            accuracy = local_bf16_compare(output, expected, native_expected)
        else:
            accuracy = {
                "passed": False,
                "threshold": BF16_THRESHOLD,
                "mere": 0.0,
                "mare": 0.0,
                "rmse": 0.0,
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
            "timing_csv_path": profile.csv_path,
            "timing_trace_path": profile.trace_view_path,
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
    torch_npu_available = bool(impls["torch_npu"]["accuracy"]["passed"])
    selected_name = "torch_npu" if torch_npu_available else "torch"
    selected_speedups = impls[selected_name]["speedups_vs_triton"]
    selected_timing = selected_name if selected_speedups.get("active_vs_active") is not None else "torch_npu"
    result["selected_baseline"] = {
        "implementation":
        selected_name,
        "source":
        "task_npu_baseline_probe" if selected_name == "torch_npu" else "pytorch_semantic_baseline",
        "selection_rule": ("Use torch_npu.npu_add_rms_norm only when its output passes the same "
                           "local BF16 precision check; otherwise use the Torch semantic baseline."),
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
        "main_speed_timing_implementation":
        selected_timing,
    }
    return result


def print_case(record: dict[str, object], total: int) -> None:
    impls = record["implementations"]
    triton = impls["triton"]
    torch_impl = impls["torch"]
    torch_npu_impl = impls["torch_npu"]
    triton_acc = triton["accuracy"]
    status = "PASS" if triton_acc["passed"] else "FAIL"
    print(f"[{status}] {record['case']:03d}/{total:03d} {record['kind']} shape={tuple(record['shape'])} "
          f"triton_active={triton['active_window']} triton_kernel={triton['kernel_sum']} "
          f"torch_active={torch_impl['active_window']} torch_kernel={torch_impl['kernel_sum']} "
          f"torch_npu_active={torch_npu_impl['active_window']} torch_npu_kernel={torch_npu_impl['kernel_sum']} "
          f"speedup_vs_torch_active={torch_impl['speedups_vs_triton']['active_vs_active_text']} "
          f"speedup_vs_torch_kernel={torch_impl['speedups_vs_triton']['kernel_vs_kernel_text']} "
          f"speedup_vs_torch_npu_active={torch_npu_impl['speedups_vs_triton']['active_vs_active_text']} "
          f"speedup_vs_torch_npu_kernel={torch_npu_impl['speedups_vs_triton']['kernel_vs_kernel_text']} "
          f"triton_accuracy={'PASS' if triton_acc['passed'] else 'FAIL'} "
          f"torch_accuracy={'PASS' if torch_impl['accuracy']['passed'] else 'FAIL'} "
          f"torch_npu_accuracy={'PASS' if torch_npu_impl['accuracy']['passed'] else 'FAIL'} "
          f"MERE={triton_acc['mere']:.3e} MARE={triton_acc['mare']:.3e} RMSE={triton_acc['rmse']:.3e} "
          f"max_diff={triton_acc['max_diff']:.3e} "
          f"triton_source={triton['elapsed_us_source']} "
          f"triton_kernel_source={triton['elapsed_us_source']} "
          f"torch_source={torch_impl['elapsed_us_source']} "
          f"torch_kernel_source={torch_impl['elapsed_us_source']} "
          f"torch_npu_source={torch_npu_impl['elapsed_us_source']} "
          f"torch_npu_kernel_source={torch_npu_impl['elapsed_us_source']}")


def summarize(records: list[dict[str, object]]) -> None:
    passed = sum(1 for r in records if r["implementations"]["triton"]["accuracy"]["passed"])
    torch_npu_acc_pass = sum(1 for r in records if r["implementations"]["torch_npu"]["accuracy"]["passed"])
    torch_acc_pass = sum(1 for r in records if r["implementations"]["torch"]["accuracy"]["passed"])
    max_rmse = max(float(r["implementations"]["triton"]["accuracy"].get("rmse") or 0.0)
                   for r in records) if records else 0.0
    print(
        f"SUMMARY passed={passed}/{len(records)} threshold={BF16_THRESHOLD:.8f} "
        f"max_rmse={max_rmse:.3e} "
        f"torch_accuracy_pass={torch_acc_pass}/{len(records)} "
        f"torch_npu_accuracy_pass={torch_npu_acc_pass}/{len(records)} "
        f"selected_task_npu_baseline={torch_npu_acc_pass} selected_torch_baseline={len(records) - torch_npu_acc_pass} "
        f"triton_strategy=kernel_details triton_scope=timing_matrix.active_window_score "
        f"baseline_strategy=kernel_details baseline_scope=timing_matrix.active_window_score")
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
    torch_npu_active = collect_impl("torch_npu", "active_window_us")
    torch_npu_kernel = collect_impl("torch_npu", "kernel_sum_us")
    torch_active = collect_impl("torch", "active_window_us")
    torch_speedups = collect_speedups("torch", "active_vs_active")
    torch_npu_runnable_all_speedup = geomean_or_none(collect_speedups("torch_npu", "active_vs_active"))
    torch_npu_runnable_all_gate = "N/A" if torch_npu_runnable_all_speedup is None else (
        "PASS" if torch_npu_runnable_all_speedup >= 1.2 else "FAIL")
    if triton_active:
        print(f"LATENCY_ACTIVE triton_geomean={geomean(triton_active):.3f} us "
              f"triton_min={min(triton_active):.3f} us triton_max={max(triton_active):.3f} us "
              f"torch_npu_geomean={format_latency_us(geomean_or_none(torch_npu_active))}")
        print(f"LATENCY_KERNEL triton_geomean={geomean(triton_kernel):.3f} us "
              f"triton_min={min(triton_kernel):.3f} us triton_max={max(triton_kernel):.3f} us "
              f"torch_npu_geomean={format_latency_us(geomean_or_none(torch_npu_kernel))}")
        print(
            "SPEEDUP_MATRIX "
            f"torch_npu_pass_active_vs_active={format_speedup(geomean(collect_speedups('torch_npu', 'active_vs_active', require_pass=True)))} "
            f"torch_npu_pass_kernel_vs_kernel={format_speedup(geomean(collect_speedups('torch_npu', 'kernel_vs_kernel', require_pass=True)))} "
            f"selected_active_vs_active={format_speedup(geomean_or_none(collect_selected('active_vs_active')))} "
            f"selected_kernel_vs_kernel={format_speedup(geomean_or_none(collect_selected('kernel_vs_kernel')))}")
        print("TORCH_SEMANTIC_AUX "
              f"torch_timed={len(torch_active)} "
              f"torch_speedup_sample={len(torch_speedups)} "
              f"active_vs_active={format_speedup(geomean_or_none(torch_speedups))}")
        print("TORCH_NPU_RUNNABLE_ALL_GATE "
              f"active_vs_active={format_speedup(torch_npu_runnable_all_speedup)} "
              f"gate={torch_npu_runnable_all_gate} threshold=1.200000x")


def build_summary(records: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:

    def impl_acc(record: dict[str, object], name: str) -> dict[str, object]:
        return record["implementations"][name]["accuracy"]

    def max_metric(name: str, metric: str, subset: list[dict[str, object]]) -> float:
        if not subset:
            return 0.0
        return max(float(impl_acc(record, name).get(metric) or 0.0) for record in subset)

    torch_npu_runnable = [
        record for record in records
        if record["implementations"]["torch_npu"].get("speedups_vs_triton", {}).get("active_vs_active") is not None
    ]
    torch_active_timed = [
        record for record in records if record["implementations"]["torch"].get("active_window_us") is not None
    ]
    torch_speedup_sample = [
        record for record in records
        if record["implementations"]["torch"].get("speedups_vs_triton", {}).get("active_vs_active") is not None
    ]
    torch_timed_candidate_active_geomean = geomean_or_none([
        float(record["implementations"]["triton"]["active_window_us"])
        for record in torch_active_timed
        if record["implementations"]["triton"].get("active_window_us") is not None
    ])
    torch_timed_baseline_active_geomean = geomean_or_none([
        float(record["implementations"]["torch"]["active_window_us"])
        for record in torch_active_timed
        if record["implementations"]["torch"].get("active_window_us") is not None
    ])
    torch_semantic_active_speedup_geomean = geomean_or_none([
        float(record["implementations"]["torch"]["speedups_vs_triton"]["active_vs_active"])
        for record in torch_speedup_sample
    ])
    torch_npu_runnable_candidate_active_geomean = geomean_or_none([
        float(record["implementations"]["triton"]["active_window_us"])
        for record in torch_npu_runnable
        if record["implementations"]["triton"].get("active_window_us") is not None
    ])
    torch_npu_runnable_baseline_active_geomean = geomean_or_none([
        float(record["implementations"]["torch_npu"]["active_window_us"])
        for record in torch_npu_runnable
        if record["implementations"]["torch_npu"].get("active_window_us") is not None
    ])
    torch_npu_runnable_all_speedup_geomean = geomean_or_none([
        float(record["implementations"]["torch_npu"]["speedups_vs_triton"]["active_vs_active"])
        for record in torch_npu_runnable
    ])
    groups: dict[str, dict[str, object]] = {}
    for kind in sorted({str(record["kind"]) for record in records}):
        subset = [record for record in records if record["kind"] == kind]
        groups[kind] = {
            "total": len(subset),
            "triton_passed": sum(1 for record in subset if impl_acc(record, "triton").get("passed")),
            "torch_passed": sum(1 for record in subset if impl_acc(record, "torch").get("passed")),
            "torch_npu_passed": sum(1 for record in subset if impl_acc(record, "torch_npu").get("passed")),
            "max_mere": max_metric("triton", "mere", subset),
            "max_mare": max_metric("triton", "mare", subset),
            "max_rmse": max_metric("triton", "rmse", subset),
            "max_diff": max_metric("triton", "max_diff", subset),
            "total_mismatch_count":
            sum(int(impl_acc(record, "triton").get("mismatch_count") or 0) for record in subset),
        }

    return {
        "schema_version":
        4,
        "source_jsonl":
        str(args.jsonl) if args.jsonl else "",
        "random_seed":
        args.random_seed,
        "shape_policy":
        args.random_shape_policy,
        "total_cases":
        len(records),
        "public_cases":
        sum(1 for record in records if record["kind"] == "public"),
        "random_generalization_cases":
        sum(1 for record in records if record["kind"] == "random_generalization"),
        "benchmark":
        bool(args.benchmark),
        "benchmark_torch":
        bool(args.benchmark_torch),
        "timing_source":
        "kernel_details.csv.active_window_median" if args.benchmark else "",
        "torch_timed":
        len(torch_active_timed),
        "torch_speedup_sample":
        len(torch_speedup_sample),
        "torch_semantic_candidate_active_geomean_us":
        torch_timed_candidate_active_geomean,
        "torch_semantic_baseline_active_geomean_us":
        torch_timed_baseline_active_geomean,
        "torch_semantic_active_speedup_geomean":
        torch_semantic_active_speedup_geomean,
        "torch_npu_runnable":
        len(torch_npu_runnable),
        "torch_npu_accuracy_passed":
        sum(1 for record in records if impl_acc(record, "torch_npu").get("passed")),
        "main_speed_sample":
        "torch_npu_runnable_all",
        "main_speed_case_count":
        len(torch_npu_runnable),
        "main_speed_candidate_active_geomean_us":
        torch_npu_runnable_candidate_active_geomean,
        "main_speed_torch_npu_active_geomean_us":
        torch_npu_runnable_baseline_active_geomean,
        "main_speed_active_speedup_geomean":
        torch_npu_runnable_all_speedup_geomean,
        "main_speed_gate":
        "N/A" if torch_npu_runnable_all_speedup_geomean is None else
        ("PASS" if torch_npu_runnable_all_speedup_geomean >= 1.2 else "FAIL"),
        "torch_npu_runnable_all_active_speedup_geomean":
        torch_npu_runnable_all_speedup_geomean,
        "torch_npu_runnable_all_speed_gate":
        "N/A" if torch_npu_runnable_all_speedup_geomean is None else
        ("PASS" if torch_npu_runnable_all_speedup_geomean >= 1.2 else "FAIL"),
        "accuracy":
        groups,
        "random_category_counts":
        dict(
            Counter(
                str(record.get("random_category") or "")
                for record in records
                if record["kind"] == "random_generalization")),
        "notes": [
            "Each implementation accuracy record includes RMSE computed by the local checker.",
            "Torch semantic timing is collected only when --benchmark-torch is set and remains an auxiliary comparison.",
            "Random generalization cases are fixed-seed reproducible non-public shape samples.",
        ],
    }


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
    parser.add_argument("--benchmark", action="store_true",
                        help="Use OpForge-compatible kernel_details.csv active-window timing.")
    parser.add_argument(
        "--benchmark-torch", action="store_true",
        help="Also profile Torch semantic reference as auxiliary timing; not used for the main speed gate.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases for smoke checks.")
    parser.add_argument("--jsonl", type=Path, default=None, help="Write per-case machine-readable results.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Write validation summary JSON.")
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
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(selected, start=1):
        record = run_case(
            case,
            index,
            args.device,
            args.benchmark,
            args.benchmark_torch,
            args.warmup,
            args.repeat,
        )
        records.append(record)
        print_case(record, len(selected))
        if args.jsonl:
            with args.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summarize(records)
    if args.summary_json:
        args.summary_json.write_text(
            json.dumps(build_summary(records, args), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
