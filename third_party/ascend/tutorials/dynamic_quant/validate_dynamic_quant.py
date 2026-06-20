#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-contained baseline, precision, and timing flow for DynamicQuant."""

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
except Exception as exc:  # pragma: no cover - depends on Ascend runtime.
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None

from dynamic_quant import dynamic_quant
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
PUBLIC_VALUE_RANGES = [
    (0.0, 0.0),
    (-0.01, 0.01),
    (-1.0, 1.0),
    (-8.0, 8.0),
    (-64.0, 64.0),
    (-256.0, 256.0),
    (-1024.0, 1024.0),
]
RANDOM_GENERALIZATION_POLICY = "seeded_dynamic_quant_docs_standard_shapes_v3"
SCALE_THRESHOLD = 1.0e-3
INT_THRESHOLD = 1


@dataclass(frozen=True)
class Case:
    case_id: str
    bsz: int
    seq: int
    hidden: int
    dst_type: str
    value_range: tuple[float, float]
    kind: str = "public"
    seed: int = 0
    note: str = ""
    head_num: int = 0
    head_dim: int = 128

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.bsz, self.seq, self.hidden)


def _case_seed(case_id: str, seed: int = 0) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    deterministic_hash = int.from_bytes(digest[:8], byteorder="big") % (2**31)
    return (int(seed) + deterministic_hash) % (2**31)


def public_cases() -> list[Case]:
    cases: list[Case] = []
    for index, (bsz, hidden, head_num, head_dim) in enumerate(PUBLIC_TEST_STANDARD_SHAPES, start=1):
        value_range = PUBLIC_VALUE_RANGES[(index - 1) % len(PUBLIC_VALUE_RANGES)]
        case_id = f"custom/dynamic_quant_{index}"
        cases.append(
            Case(case_id, bsz, 1, hidden, "int8", value_range, seed=_case_seed(case_id), head_num=head_num,
                 head_dim=head_dim))
    for offset, (bsz, hidden, head_num, head_dim) in enumerate(PUBLIC_TEST_STANDARD_SHAPES, start=21):
        value_range = PUBLIC_VALUE_RANGES[(offset - 21) % len(PUBLIC_VALUE_RANGES)]
        case_id = f"custom/dynamic_quant_{offset}"
        cases.append(
            Case(case_id, bsz, 1, hidden, "int4", value_range, seed=_case_seed(case_id), head_num=head_num,
                 head_dim=head_dim))
    return cases


def random_generalization_cases(count: int, seed: int) -> list[Case]:
    if count < 0:
        raise ValueError("random generalization count must be non-negative")
    rng = random.Random(int(seed))
    seen: set[tuple[tuple[int, int, int], str, tuple[float, float]]] = set()
    cases: list[Case] = []
    attempts = 0
    max_attempts = max(200, count * 200)

    while len(cases) < count and attempts < max_attempts:
        attempts += 1
        bsz, hidden, head_num, head_dim = PUBLIC_TEST_STANDARD_SHAPES[(rng.randrange(len(PUBLIC_TEST_STANDARD_SHAPES)) +
                                                                       attempts) % len(PUBLIC_TEST_STANDARD_SHAPES)]
        seq = 1
        dst_type = "int4" if len(cases) % 2 else "int8"
        value_range = rng.choice(PUBLIC_VALUE_RANGES + [(-512.0, 512.0), (-8.0, 2.0)])
        key = ((bsz, seq, hidden), dst_type, value_range)
        if key in seen:
            continue
        seen.add(key)
        case_index = len(cases) + 1
        case_id = f"custom/dynamic_quant_random_{case_index:03d}"
        cases.append(
            Case(
                case_id,
                bsz,
                seq,
                hidden,
                dst_type,
                value_range,
                kind="random_generalization",
                seed=_case_seed(case_id, seed),
                note=f"{RANDOM_GENERALIZATION_POLICY}:docs_standard_value_variation",
                head_num=head_num,
                head_dim=head_dim,
            ))

    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} random cases after {attempts} attempts, expected {count}")
    return cases


def _require_npu(device: str) -> None:
    if _TORCH_NPU_IMPORT_ERROR is not None:
        raise RuntimeError(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if device != "npu":
        raise ValueError("this tutorial validates the Triton-Ascend path on device='npu' only")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")


def _make_input(case: Case, device: str) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(int(case.seed))
    low, high = case.value_range
    if low == high:
        x_cpu = torch.full(case.shape, float(low), dtype=torch.float32)
    else:
        x_cpu = torch.rand(case.shape, dtype=torch.float32, generator=gen) * (high - low) + low
    return x_cpu.to(torch.bfloat16).to(device=device).contiguous()


def _reference_dynamic_quant(x: torch.Tensor, dst_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    if dst_type == "int4":
        qmin, qmax, qabs = -8, 7, 7
    elif dst_type == "int8":
        qmin, qmax, qabs = -128, 127, 127
    else:
        raise ValueError(f"unsupported dst_type {dst_type!r}")
    x_compute = x.to(torch.float32)
    abs_max = torch.max(torch.abs(x_compute), dim=-1, keepdim=True)[0]
    scale_out = abs_max.clamp(min=1.0e-12) / float(qabs)
    output = torch.clamp(torch.round(x_compute / scale_out), qmin, qmax).to(torch.int8)
    return output, scale_out.squeeze(-1).to(torch.float32)


def _unpack_quint4x2(packed: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    if packed.shape[-1] * 8 != target_shape[-1]:
        raise ValueError(f"packed int4 shape {tuple(packed.shape)} cannot expand to {target_shape}")
    values = packed.detach().to(torch.int32).cpu()
    chunks = []
    for shift in range(0, 32, 4):
        nibble = torch.bitwise_and(torch.bitwise_right_shift(values, shift), 0xF)
        signed = torch.where(nibble >= 8, nibble - 16, nibble)
        chunks.append(signed.to(torch.int8))
    stacked = torch.stack(chunks, dim=-1).reshape(target_shape)
    return stacked.to(device=packed.device)


def _torch_npu_dynamic_quant(x: torch.Tensor, dst_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    if torch_npu is None:
        raise RuntimeError(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if dst_type == "int8":
        return torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    if dst_type == "int4":
        quant_dtype = getattr(torch, "quint4x2", None)
        if quant_dtype is None:
            raise RuntimeError("torch.quint4x2 is not available in this PyTorch build")
        packed, scale = torch_npu.npu_dynamic_quant(x, dst_type=quant_dtype)
        return _unpack_quint4x2(packed, tuple(x.shape)), scale
    raise ValueError(f"unsupported dst_type {dst_type!r}")


def _compare_outputs(
    output: torch.Tensor,
    scale: torch.Tensor,
    ref_output: torch.Tensor,
    ref_scale: torch.Tensor,
) -> dict[str, object]:
    output_cpu = output.detach().cpu()
    scale_cpu = scale.detach().cpu()
    ref_output_cpu = ref_output.detach().cpu()
    ref_scale_cpu = ref_scale.detach().cpu()
    int_diff = (output_cpu.to(torch.int16) - ref_output_cpu.to(torch.int16)).abs()
    scale_diff = (scale_cpu - ref_scale_cpu).abs()
    max_int_diff = int(int_diff.max().item()) if int_diff.numel() else 0
    output_mean_abs_diff = float(int_diff.float().mean().item()) if int_diff.numel() else 0.0
    output_rmse = float(torch.sqrt(torch.mean(int_diff.float() * int_diff.float())).item()) if int_diff.numel() else 0.0
    mismatch_count = int((int_diff > INT_THRESHOLD).sum().item()) if int_diff.numel() else 0
    scale_max_abs_diff = float(scale_diff.max().item()) if scale_diff.numel() else 0.0
    scale_mean_abs_diff = float(scale_diff.mean().item()) if scale_diff.numel() else 0.0
    scale_rmse = float(torch.sqrt(torch.mean(scale_diff * scale_diff)).item()) if scale_diff.numel() else 0.0
    scale_rel = scale_diff / torch.clamp(ref_scale_cpu.abs(), min=1.0e-6)
    scale_mere = float(scale_rel.mean().item()) if scale_rel.numel() else 0.0
    scale_mare = float(scale_rel.max().item()) if scale_rel.numel() else 0.0
    scale_mismatch_count = int((scale_diff > SCALE_THRESHOLD).sum().item()) if scale_diff.numel() else 0
    range_ok = True
    passed = mismatch_count == 0 and scale_mismatch_count == 0 and range_ok
    return {
        "passed":
        passed,
        "threshold":
        SCALE_THRESHOLD,
        "mere":
        scale_mere,
        "mare":
        scale_mare,
        "rmse":
        max(output_rmse, scale_rmse),
        "output_rmse":
        output_rmse,
        "scale_rmse":
        scale_rmse,
        "max_diff":
        max(scale_max_abs_diff, float(max_int_diff)),
        "mean_diff":
        max(scale_mean_abs_diff, output_mean_abs_diff),
        "mismatch_count":
        mismatch_count + scale_mismatch_count,
        "total_count":
        int(int_diff.numel() + scale_diff.numel()),
        "error_msg":
        "" if passed else "integer or scale threshold exceeded",
        "output_results": [
            {
                "name": "output",
                "dtype": str(output_cpu.dtype),
                "shape": list(output_cpu.shape),
                "criterion": "quantized integer AE <= 1",
                "passed": mismatch_count == 0,
                "mismatch_count": mismatch_count,
                "total_count": int(int_diff.numel()),
                "max_diff": max_int_diff,
                "mean_diff": output_mean_abs_diff,
                "rmse": output_rmse,
                "mere": 0.0,
                "mare": 0.0,
            },
            {
                "name": "scale",
                "dtype": str(scale_cpu.dtype),
                "shape": list(scale_cpu.shape),
                "criterion": "floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence",
                "passed": scale_mismatch_count == 0,
                "mismatch_count": scale_mismatch_count,
                "total_count": int(scale_diff.numel()),
                "max_diff": scale_max_abs_diff,
                "mean_diff": scale_mean_abs_diff,
                "rmse": scale_rmse,
                "mere": scale_mere,
                "mare": scale_mare,
            },
        ],
    }


def _zero_accuracy(outputs: tuple[torch.Tensor, torch.Tensor]) -> dict[str, object]:
    out, scale = outputs
    return {
        "passed":
        True,
        "threshold":
        0.0,
        "mere":
        0.0,
        "mare":
        0.0,
        "rmse":
        0.0,
        "output_rmse":
        0.0,
        "scale_rmse":
        0.0,
        "max_diff":
        0.0,
        "mean_diff":
        0.0,
        "mismatch_count":
        0,
        "total_count":
        int(out.numel() + scale.numel()),
        "error_msg":
        "",
        "output_results": [
            {
                "name": "output",
                "dtype": str(out.dtype),
                "shape": list(out.shape),
                "criterion": "quantized integer AE <= 1",
                "passed": True,
                "mismatch_count": 0,
                "total_count": int(out.numel()),
                "max_diff": 0.0,
                "mean_diff": 0.0,
                "rmse": 0.0,
                "mere": 0.0,
                "mare": 0.0,
            },
            {
                "name": "scale",
                "dtype": str(scale.dtype),
                "shape": list(scale.shape),
                "criterion": "floating scale AE <= 1e-3 plus MARE/MERE/RMSE evidence",
                "passed": True,
                "mismatch_count": 0,
                "total_count": int(scale.numel()),
                "max_diff": 0.0,
                "mean_diff": 0.0,
                "rmse": 0.0,
                "mere": 0.0,
                "mare": 0.0,
            },
        ],
    }


def _failed_accuracy(error: str) -> dict[str, object]:
    return {
        "passed": False,
        "threshold": SCALE_THRESHOLD,
        "mere": 0.0,
        "mare": 0.0,
        "rmse": 0.0,
        "output_rmse": 0.0,
        "scale_rmse": 0.0,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "mismatch_count": 1,
        "total_count": 1,
        "error_msg": error,
        "output_results": [],
    }


def _fmt_us(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f} us"


def _fmt_x(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6f}x"


def _speedup(base_us: object, cand_us: object) -> float | None:
    try:
        b = float(base_us) if base_us is not None else 0.0
        c = float(cand_us) if cand_us is not None else 0.0
    except (TypeError, ValueError):
        return None
    return b / c if b > 0.0 and c > 0.0 else None


def _impl_record(
    name: str,
    role: str,
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    ref_outputs: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    case_id: str,
    *,
    profile_for_speed: bool,
) -> dict[str, object]:
    error = ""
    outputs: tuple[torch.Tensor, torch.Tensor] | None = None
    profile = ProfileResult(None, None, None, None, None, None)
    try:
        with torch.inference_mode():
            outputs = fn()
            torch.npu.synchronize()
        accuracy = (_zero_accuracy(outputs) if role == "pytorch_semantic_baseline" else _compare_outputs(
            outputs[0], outputs[1], ref_outputs[0], ref_outputs[1]))
    except Exception as exc:  # pragma: no cover - runtime/API dependent.
        error = f"{type(exc).__name__}: {exc}"
        accuracy = _failed_accuracy(error)

    if args.benchmark and profile_for_speed and outputs is not None:
        prof_outputs, profile = profile_kernel_details(name, case_id, fn, warmup=args.warmup, repeat=args.repeat)
        if prof_outputs is not None:
            outputs = prof_outputs
        if profile.error:
            error = profile.error
    elif error:
        profile.error = error

    return {
        "role": role,
        "accuracy": accuracy,
        "timing_strategy": "kernel_details" if profile_for_speed else "correctness_only",
        "perf_metric_strategy": profile.perf_metric_strategy,
        "measurement_scope": profile.measurement_scope,
        "elapsed_us_source": profile.elapsed_us_source,
        "primary_latency_us": profile.latency_us,
        "primary_latency": _fmt_us(profile.latency_us),
        "latency_us": profile.latency_us,
        "latency": _fmt_us(profile.latency_us),
        "active_window_us": profile.active_window_us,
        "active_window": _fmt_us(profile.active_window_us),
        "kernel_sum_us": profile.kernel_sum_us,
        "kernel_sum": _fmt_us(profile.kernel_sum_us),
        "window_gap_us": profile.window_gap_us,
        "window_gap": _fmt_us(profile.window_gap_us),
        "kernel_count": profile.kernel_count,
        "step_count": profile.step_count,
        "device_kernels": profile.device_kernels,
        "device_timeline": profile.device_timeline,
        "timing_csv_path": profile.csv_path,
        "timing_trace_path": profile.trace_view_path,
        "profile_error": profile.error or error or None,
    }


def run_case(case: Case, args: argparse.Namespace, index: int) -> dict[str, object]:
    x = _make_input(case, args.device)
    with torch.inference_mode():
        ref_outputs = _reference_dynamic_quant(x, case.dst_type)
        torch.npu.synchronize()

    impls = {
        "triton":
        _impl_record(
            "triton",
            "candidate",
            lambda: dynamic_quant(x, case.dst_type),
            ref_outputs,
            args,
            case.case_id,
            profile_for_speed=True,
        ),
        "torch":
        _impl_record(
            "torch",
            "pytorch_semantic_baseline",
            lambda: _reference_dynamic_quant(x, case.dst_type),
            ref_outputs,
            args,
            case.case_id,
            profile_for_speed=bool(args.benchmark_torch),
        ),
        "torch_npu":
        _impl_record(
            "torch_npu",
            "task_npu_baseline_probe",
            lambda: _torch_npu_dynamic_quant(x, case.dst_type),
            ref_outputs,
            args,
            case.case_id,
            profile_for_speed=True,
        ),
    }
    cand_active_us = impls["triton"]["active_window_us"]
    cand_kernel_us = impls["triton"]["kernel_sum_us"]
    for name in ["torch", "torch_npu"]:
        active_vs_active = _speedup(impls[name]["active_window_us"], cand_active_us)
        kernel_vs_kernel = _speedup(impls[name]["kernel_sum_us"], cand_kernel_us)
        baseline_active_vs_candidate_kernel = _speedup(impls[name]["active_window_us"], cand_kernel_us)
        baseline_kernel_vs_candidate_active = _speedup(impls[name]["kernel_sum_us"], cand_active_us)
        s = active_vs_active
        impls[name]["speedup_vs_triton"] = s
        impls[name]["speedup_vs_triton_text"] = _fmt_x(s)
        impls[name]["speedups_vs_triton"] = {
            "active_vs_active": active_vs_active,
            "active_vs_active_text": _fmt_x(active_vs_active),
            "kernel_vs_kernel": kernel_vs_kernel,
            "kernel_vs_kernel_text": _fmt_x(kernel_vs_kernel),
            "baseline_active_vs_candidate_kernel": baseline_active_vs_candidate_kernel,
            "baseline_active_vs_candidate_kernel_text": _fmt_x(baseline_active_vs_candidate_kernel),
            "baseline_kernel_vs_candidate_active": baseline_kernel_vs_candidate_active,
            "baseline_kernel_vs_candidate_active_text": _fmt_x(baseline_kernel_vs_candidate_active),
        }
    selected_name = "torch_npu" if impls["torch_npu"]["accuracy"]["passed"] else "torch"
    selected_speedup = impls[selected_name].get("speedup_vs_triton")
    return {
        "case": index,
        "case_id": case.case_id,
        "kind": case.kind,
        "shape": list(case.shape),
        "dtype": "bfloat16",
        "dst_type": case.dst_type,
        "attrs": {
            "Batch": case.bsz,
            "SequenceLength": case.seq,
            "HiddenSize": case.hidden,
            "HeadNum": case.head_num,
            "HeadDim": case.head_dim,
            "dst_type": case.dst_type,
        },
        "case_detail": {
            "Batch": case.bsz,
            "SequenceLength": case.seq,
            "HiddenSize": case.hidden,
            "HeadNum": case.head_num,
            "HeadDim": case.head_dim,
        },
        "value_range": list(case.value_range),
        "seed": case.seed,
        "note": case.note,
        "timing_policy":
        "Candidate and torch_npu benchmark paths are timed by torch_npu.profiler kernel_details.csv active-window, matching OpForge/CANN-Bench; Torch semantic reference is timed only with --benchmark-torch as an auxiliary comparison and is not the main speed gate.",
        "implementations": impls,
        "selected_baseline": {
            "implementation": selected_name,
            "source": "task_npu_baseline_probe" if selected_name == "torch_npu" else "pytorch_semantic_baseline",
            "selection_rule":
            "Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline.",
            "active_window_us": impls[selected_name]["active_window_us"],
            "active_window": impls[selected_name]["active_window"],
            "kernel_sum_us": impls[selected_name]["kernel_sum_us"],
            "kernel_sum": impls[selected_name]["kernel_sum"],
            "latency_us": impls[selected_name]["latency_us"],
            "latency": impls[selected_name]["latency"],
            "speedup_vs_triton": selected_speedup,
            "speedup_vs_triton_text": _fmt_x(selected_speedup),
            "speedups_vs_triton": impls[selected_name].get("speedups_vs_triton", {}),
            "perf_metric_strategy": impls[selected_name]["perf_metric_strategy"],
            "measurement_scope": impls[selected_name]["measurement_scope"],
            "elapsed_us_source": impls[selected_name]["elapsed_us_source"],
        },
    }


def _geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(max(v, 1.0e-9)) for v in values) / len(values)) if values else None


def build_summary(records: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:

    def acc(record: dict[str, object], name: str) -> dict[str, object]:
        return record["implementations"][name]["accuracy"]

    torch_npu_runnable = [r for r in records if r["implementations"]["torch_npu"].get("speedup_vs_triton") is not None]
    torch_active_timed = [r for r in records if r["implementations"]["torch"].get("active_window_us") is not None]
    torch_speedup_sample = [r for r in records if r["implementations"]["torch"].get("speedup_vs_triton") is not None]
    torch_timed_candidate_active_geomean = _geomean([
        float(r["implementations"]["triton"]["active_window_us"])
        for r in torch_active_timed
        if r["implementations"]["triton"].get("active_window_us") is not None
    ])
    torch_timed_baseline_active_geomean = _geomean([
        float(r["implementations"]["torch"]["active_window_us"])
        for r in torch_active_timed
        if r["implementations"]["torch"].get("active_window_us") is not None
    ])
    torch_semantic_active_speedup_geomean = _geomean([
        float(r["implementations"]["torch"]["speedup_vs_triton"])
        for r in torch_speedup_sample
        if r["implementations"]["torch"].get("speedup_vs_triton") is not None
    ])
    torch_npu_runnable_candidate_active_geomean = _geomean([
        float(r["implementations"]["triton"]["active_window_us"])
        for r in torch_npu_runnable
        if r["implementations"]["triton"].get("active_window_us") is not None
    ])
    torch_npu_runnable_baseline_active_geomean = _geomean([
        float(r["implementations"]["torch_npu"]["active_window_us"])
        for r in torch_npu_runnable
        if r["implementations"]["torch_npu"].get("active_window_us") is not None
    ])
    torch_npu_runnable_all_speedup_geomean = _geomean([
        float(r["implementations"]["torch_npu"]["speedup_vs_triton"])
        for r in torch_npu_runnable
        if r["implementations"]["torch_npu"].get("speedup_vs_triton") is not None
    ])
    return {
        "schema_version":
        4,
        "source_jsonl":
        str(args.jsonl) if args.jsonl else "",
        "total_cases":
        len(records),
        "public_cases":
        sum(1 for r in records if r["kind"] == "public"),
        "random_generalization_cases":
        sum(1 for r in records if r["kind"] == "random_generalization"),
        "passed":
        sum(1 for r in records if acc(r, "triton").get("passed")),
        "failed":
        sum(1 for r in records if not acc(r, "triton").get("passed")),
        "dst_type_counts":
        dict(Counter(str(r["dst_type"]) for r in records)),
        "selected_baseline_counts":
        dict(Counter(str(r["selected_baseline"]["implementation"]) for r in records)),
        "torch_npu_runnable":
        len(torch_npu_runnable),
        "torch_npu_accuracy_passed":
        sum(1 for r in records if acc(r, "torch_npu").get("passed")),
        "torch_npu_accuracy_failed":
        sum(1 for r in torch_npu_runnable if not acc(r, "torch_npu").get("passed")),
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
        "benchmark":
        bool(args.benchmark),
        "benchmark_torch":
        bool(args.benchmark_torch),
        "timing_source":
        "kernel_details.csv.active_window_median" if args.benchmark else "",
        "candidate_active_geomean_us":
        _geomean([
            float(r["implementations"]["triton"]["active_window_us"])
            for r in records
            if r["implementations"]["triton"].get("active_window_us") is not None
        ]),
        "selected_active_speedup_geomean":
        _geomean([
            float(r["selected_baseline"]["speedup_vs_triton"])
            for r in records
            if r["selected_baseline"].get("speedup_vs_triton") is not None
        ]),
        "max_rmse":
        max(float(acc(r, "triton").get("rmse") or 0.0) for r in records) if records else 0.0,
        "max_output_rmse":
        max(float(acc(r, "triton").get("output_rmse") or 0.0) for r in records) if records else 0.0,
        "max_scale_rmse":
        max(float(acc(r, "triton").get("scale_rmse") or 0.0) for r in records) if records else 0.0,
        "notes": [
            "Torch and torch_npu baselines are generated inside this operator directory.",
            "Torch semantic timing is collected only when --benchmark-torch is set and remains an auxiliary comparison.",
            "RMSE/MERE/MARE are emitted by validate_dynamic_quant.py for candidate and baselines.",
            "No external historical evaluation CSV/JSON is used by this self-validation flow.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true",
                        help="run the documented 20-shape test standard for both int8 and int4")
    parser.add_argument("--random-generalization", type=int, default=0, help="number of seeded random non-public cases")
    parser.add_argument("--random-seed", type=int, default=20260617)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--benchmark", action="store_true",
                        help="collect OpForge-compatible kernel_details.csv active-window timing for all paths")
    parser.add_argument(
        "--benchmark-torch", action="store_true",
        help="also profile Torch semantic reference as auxiliary timing; not used for the main speed gate")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    _require_npu(args.device)
    cases: list[Case] = []
    if args.public or args.random_generalization == 0:
        cases.extend(public_cases())
    if args.random_generalization:
        cases.extend(random_generalization_cases(args.random_generalization, args.random_seed))
    if args.max_cases is not None:
        cases = cases[:args.max_cases]

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with (args.jsonl.open("w", encoding="utf-8")
          if args.jsonl else open(os.devnull, "w", encoding="utf-8")) as out_file:
        for index, case in enumerate(cases, start=1):
            record = run_case(case, args, index)
            records.append(record)
            out_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            triton = record["implementations"]["triton"]
            torch_npu_impl = record["implementations"]["torch_npu"]
            status = "PASS" if triton["accuracy"]["passed"] else "FAIL"
            print(
                f"[{status}] {index:03d}/{len(cases):03d} {case.case_id} shape={case.shape} dst_type={case.dst_type} "
                f"triton_active={triton['active_window']} torch_npu_active={torch_npu_impl['active_window']} "
                f"selected={record['selected_baseline']['implementation']} "
                f"output_max_diff={triton['accuracy']['output_results'][0]['max_diff'] if triton['accuracy'].get('output_results') else 'N/A'} "
                f"rmse={triton['accuracy'].get('rmse', 0.0):.6g} "
                f"torch_npu_accuracy={'PASS' if torch_npu_impl['accuracy']['passed'] else 'FAIL'} "
                f"torch_npu_error={torch_npu_impl.get('profile_error') or torch_npu_impl['accuracy'].get('error_msg') or ''}"
            )

    summary = build_summary(records, args)
    print("SUMMARY "
          f"total={summary['total_cases']} passed={summary['passed']} failed={summary['failed']} "
          f"public={summary['public_cases']} random_generalization={summary['random_generalization_cases']} "
          f"torch_npu_runnable={summary['torch_npu_runnable']} "
          f"torch_npu_accuracy_passed={summary['torch_npu_accuracy_passed']} "
          f"torch_timed={summary['torch_timed']} "
          f"torch_speedup_sample={summary['torch_speedup_sample']} "
          f"torch_semantic_speedup={summary['torch_semantic_active_speedup_geomean']} "
          f"torch_npu_runnable_all_speedup={summary['torch_npu_runnable_all_active_speedup_geomean']} "
          f"torch_npu_runnable_all_gate={summary['torch_npu_runnable_all_speed_gate']} "
          f"max_rmse={summary['max_rmse']:.6g}")
    if args.summary_json:
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
