#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-contained baseline, precision, and timing flow for KvRmsNormRopeCache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

try:
    import torch_npu
except Exception as exc:  # pragma: no cover - requires Ascend runtime.
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None

from kv_rms_norm_rope_cache import kv_rms_norm_rope_cache
from profiler_timing import ProfileResult, profile_kernel_details

DOC_MODEL_ROWS = [
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
SPLIT_CHOICES = [
    (64, 64),
    (32, 96),
    (48, 80),
    (80, 48),
    (96, 32),
]
BF16_THRESHOLD = 0.02
DEFAULT_EPSILON = 1.0e-5


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    bsz: int
    hidden: int
    heads: int
    head_dim: int
    skv: int
    scache: int
    bcache: int
    d_rope: int
    d_value: int
    seed: int
    note: str

    @property
    def shape(self) -> dict[str, int]:
        return {
            "Bkv": self.bsz,
            "Bcache": self.bcache,
            "N": self.heads,
            "HiddenSize": self.hidden,
            "HeadDim": self.head_dim,
            "Skv": self.skv,
            "Scache": self.scache,
            "D": self.d_rope + self.d_value,
            "Dk": self.d_rope,
            "Dv": self.d_value,
        }


def public_cases() -> list[Case]:
    cases: list[Case] = []
    for idx, (bsz, hidden, heads, head_dim) in enumerate(DOC_MODEL_ROWS, start=1):
        cases.append(
            Case(
                case_id=f"custom/kv_rms_norm_rope_cache_{idx}",
                kind="public",
                bsz=bsz,
                hidden=hidden,
                heads=heads,
                head_dim=head_dim,
                skv=1,
                scache=8,
                bcache=bsz,
                d_rope=64,
                d_value=64,
                seed=_seed_from_case_id(f"custom/kv_rms_norm_rope_cache_{idx}", 20260617),
                note="documented model row, public 64/64 split",
            ))
    return cases


def random_generalization_cases(count: int, seed: int) -> list[Case]:
    if count < 0:
        raise ValueError("random generalization count must be non-negative")
    rng = random.Random(int(seed))
    cases: list[Case] = []
    seen: set[tuple[int, int, int, int, int, int, int]] = set()
    max_attempts = max(200, count * 200)
    attempts = 0
    while len(cases) < count and attempts < max_attempts:
        attempts += 1
        bsz, hidden, heads, head_dim = DOC_MODEL_ROWS[(rng.randrange(len(DOC_MODEL_ROWS)) + attempts) %
                                                      len(DOC_MODEL_ROWS)]
        d_rope, d_value = SPLIT_CHOICES[(rng.randrange(len(SPLIT_CHOICES)) + attempts) % len(SPLIT_CHOICES)]
        if d_rope + d_value != head_dim:
            continue
        skv = rng.choice([1, 2, 3, 4])
        scache = rng.choice([4, 5, 8, 9, 12])
        bcache = bsz + rng.choice([0, 0, 1])
        signature = (bsz, heads, skv, scache, bcache, d_rope, d_value)
        if signature in seen:
            continue
        seen.add(signature)
        idx = len(cases) + 1
        case_id = f"custom/kv_rms_norm_rope_cache_random_{idx:03d}"
        cases.append(
            Case(
                case_id=case_id,
                kind="random_generalization",
                bsz=bsz,
                hidden=hidden,
                heads=heads,
                head_dim=head_dim,
                skv=skv,
                scache=scache,
                bcache=bcache,
                d_rope=d_rope,
                d_value=d_value,
                seed=_seed_from_case_id(case_id, seed),
                note="seeded docs-row HeadDim=128 Dk/Dv split sample",
            ))
    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} random cases after {attempts} attempts, expected {count}")
    return cases


def _seed_from_case_id(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    stable = int.from_bytes(digest[:8], "big") % (2**31)
    return (stable + int(seed)) % (2**31)


def _bf16_uniform(shape: Iterable[int], min_val: float, max_val: float, gen: torch.Generator) -> torch.Tensor:
    value = torch.rand(tuple(shape), dtype=torch.float64, generator=gen)
    return (value * (max_val - min_val) + min_val).to(torch.bfloat16)


def make_inputs(case: Case, device: torch.device) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator()
    gen.manual_seed(int(case.seed))
    d_total = case.d_rope + case.d_value
    kv = _bf16_uniform((case.bsz, case.heads, case.skv, d_total), -1.0, 1.0, gen)
    gamma = _bf16_uniform((case.d_value, ), 0.5, 1.5, gen)
    cos = _bf16_uniform((case.bsz, 1, 1, case.d_rope), -1.0, 1.0, gen)
    sin = _bf16_uniform((case.bsz, 1, 1, case.d_rope), -1.0, 1.0, gen)
    k_cache = _bf16_uniform((case.bcache, case.heads, case.scache, case.d_rope), -0.25, 0.25, gen)
    ckv_cache = _bf16_uniform((case.bcache, case.heads, case.scache, case.d_value), -0.25, 0.25, gen)
    index = torch.arange(case.bsz * case.skv, dtype=torch.int64).reshape(case.bsz, case.skv) % case.scache
    if index.numel() >= 7:
        index.reshape(-1)[6::7] = -1
    return (
        kv.to(device),
        gamma.to(device),
        cos.to(device),
        sin.to(device),
        index.to(device),
        k_cache.to(device),
        ckv_cache.to(device),
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = int(x.shape[-1]) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference_impl(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_rope = int(cos.shape[-1])
    d_value = int(gamma.shape[0])
    rope = kv[..., :d_rope].to(torch.float32)
    value = kv[..., d_rope:d_rope + d_value].to(torch.float32)
    k_rope = rope * cos.to(torch.float32) + rotate_half(rope) * sin.to(torch.float32)
    variance = torch.mean(value * value, dim=-1, keepdim=True)
    ckv = value * torch.rsqrt(variance + float(epsilon)) * gamma.to(torch.float32)
    k_out = k_cache.clone()
    ckv_out = ckv_cache.clone()
    index_cpu = index.cpu()
    bsz, _, skv, _ = kv.shape
    scache = int(k_cache.shape[2])
    for b in range(int(bsz)):
        for s in range(int(skv)):
            pos = int(index_cpu[b, s].item())
            if pos == -1:
                continue
            if pos < 0 or pos >= scache:
                raise ValueError(f"cache index out of range: {pos}")
            k_out[b, :, pos, :] = k_rope[b, :, s, :].to(k_out.dtype)
            ckv_out[b, :, pos, :] = ckv[b, :, s, :].to(ckv_out.dtype)
    return k_out, ckv_out


def run_candidate(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    kv, gamma, cos, sin, index, k_cache, ckv_cache = inputs
    return kv_rms_norm_rope_cache(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache.clone(),
        ckv_cache.clone(),
        DEFAULT_EPSILON,
        "Norm",
    )


def run_torch_npu(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if torch_npu is None:
        raise RuntimeError(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    kv, gamma, cos, sin, index, k_cache, ckv_cache = inputs
    op = getattr(torch_npu, "npu_kv_rmsnorm_rope_cache_v2", None)
    if not callable(op):
        op = getattr(torch_npu, "npu_kv_rmsnorm_rope_cache", None)
    if not callable(op):
        raise RuntimeError("torch_npu npu_kv_rmsnorm_rope_cache API is not available")
    out = op(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache.clone(),
        ckv_cache.clone(),
        epsilon=DEFAULT_EPSILON,
        cache_mode="Norm",
    )
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError(f"torch_npu kv_rmsnorm_rope_cache returned unexpected output: {type(out)}")
    return out[0], out[1]


def compare_outputs(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    output_results: list[dict[str, object]] = []
    mismatch_count = 0
    total_count = 0
    max_diff = 0.0
    sum_diff = 0.0
    max_rel = 0.0
    sum_rel = 0.0
    sum_sq_diff = 0.0
    for name, out, ref in zip(["k_cache", "ckv_cache"], actual, expected):
        out_f = out.detach().to(torch.float32).cpu()
        ref_f = ref.detach().to(torch.float32).cpu()
        diff = torch.abs(out_f - ref_f)
        denom = torch.clamp(torch.abs(ref_f), min=1.0e-6)
        rel = diff / denom
        item_mismatch = int((diff > BF16_THRESHOLD).sum().item())
        item_total = int(diff.numel())
        item_max = float(diff.max().item()) if item_total else 0.0
        item_sum = float(diff.sum().item()) if item_total else 0.0
        item_sq = float((diff * diff).sum().item()) if item_total else 0.0
        item_mare = float(rel.max().item()) if item_total else 0.0
        item_mere = float(rel.mean().item()) if item_total else 0.0
        item_rmse = math.sqrt(item_sq / max(1, item_total))
        mismatch_count += item_mismatch
        total_count += item_total
        max_diff = max(max_diff, item_max)
        max_rel = max(max_rel, item_mare)
        sum_rel += float(rel.sum().item()) if item_total else 0.0
        sum_diff += item_sum
        sum_sq_diff += item_sq
        output_results.append({
            "name": name,
            "dtype": str(out.dtype),
            "shape": list(out.shape),
            "criterion": f"BF16 absolute error <= {BF16_THRESHOLD}",
            "passed": item_mismatch == 0,
            "mismatch_count": item_mismatch,
            "total_count": item_total,
            "max_diff": item_max,
            "mean_diff": item_sum / max(1, item_total),
            "rmse": item_rmse,
            "mere": item_mere,
            "mare": item_mare,
        })
    return {
        "passed": mismatch_count == 0,
        "threshold": BF16_THRESHOLD,
        "mismatch_count": mismatch_count,
        "total_count": total_count,
        "max_diff": max_diff,
        "mean_diff": sum_diff / max(1, total_count),
        "rmse": math.sqrt(sum_sq_diff / max(1, total_count)),
        "mere": sum_rel / max(1, total_count),
        "mare": max_rel,
        "error_msg": "" if mismatch_count == 0 else "BF16 absolute threshold exceeded",
        "output_results": output_results,
    }


def zero_accuracy(outputs: tuple[torch.Tensor, torch.Tensor]) -> dict[str, object]:
    output_results = []
    total = 0
    for name, tensor in zip(["k_cache", "ckv_cache"], outputs):
        total += int(tensor.numel())
        output_results.append({
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "criterion": f"BF16 absolute error <= {BF16_THRESHOLD}",
            "passed": True,
            "mismatch_count": 0,
            "total_count": int(tensor.numel()),
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "rmse": 0.0,
            "mere": 0.0,
            "mare": 0.0,
        })
    return {
        "passed": True,
        "threshold": BF16_THRESHOLD,
        "mismatch_count": 0,
        "total_count": total,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "rmse": 0.0,
        "mere": 0.0,
        "mare": 0.0,
        "error_msg": "",
        "output_results": output_results,
    }


def failed_accuracy(error: str) -> dict[str, object]:
    return {
        "passed": False,
        "threshold": BF16_THRESHOLD,
        "mismatch_count": 1,
        "total_count": 1,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "rmse": 0.0,
        "mere": 0.0,
        "mare": 0.0,
        "error_msg": error,
        "output_results": [],
    }


def fmt_us(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f} us"


def fmt_x(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.6f}x"


def speedup(base_us: object, cand_us: object) -> float | None:
    try:
        b = float(base_us) if base_us is not None else 0.0
        c = float(cand_us) if cand_us is not None else 0.0
    except (TypeError, ValueError):
        return None
    return b / c if b > 0.0 and c > 0.0 else None


def impl_record(
    name: str,
    role: str,
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    expected: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    case_id: str,
    *,
    profile_for_speed: bool,
) -> dict[str, object]:
    outputs = None
    error = ""
    profile = ProfileResult(None, None, None, None, None, None)
    try:
        with torch.inference_mode():
            outputs = fn()
            torch.npu.synchronize()
        accuracy = zero_accuracy(outputs) if role == "pytorch_semantic_baseline" else compare_outputs(outputs, expected)
    except Exception as exc:  # pragma: no cover - runtime/API dependent.
        error = f"{type(exc).__name__}: {exc}"
        accuracy = failed_accuracy(error)
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
        "primary_latency": fmt_us(profile.latency_us),
        "latency_us": profile.latency_us,
        "latency": fmt_us(profile.latency_us),
        "active_window_us": profile.active_window_us,
        "active_window": fmt_us(profile.active_window_us),
        "kernel_sum_us": profile.kernel_sum_us,
        "kernel_sum": fmt_us(profile.kernel_sum_us),
        "window_gap_us": profile.window_gap_us,
        "window_gap": fmt_us(profile.window_gap_us),
        "kernel_count": profile.kernel_count,
        "step_count": profile.step_count,
        "device_kernels": profile.device_kernels,
        "device_timeline": profile.device_timeline,
        "timing_csv_path": profile.csv_path,
        "timing_trace_path": profile.trace_view_path,
        "profile_error": profile.error or error or None,
    }


def run_case(case: Case, device: torch.device, args: argparse.Namespace, index: int) -> dict[str, object]:
    inputs = make_inputs(case, device)
    with torch.inference_mode():
        expected = reference_impl(*inputs, DEFAULT_EPSILON)
        torch.npu.synchronize()
    impls = {
        "triton":
        impl_record("triton", "candidate", lambda: run_candidate(inputs), expected, args, case.case_id,
                    profile_for_speed=True),
        "torch":
        impl_record("torch", "pytorch_semantic_baseline", lambda: reference_impl(*inputs, DEFAULT_EPSILON), expected,
                    args, case.case_id, profile_for_speed=bool(args.benchmark_torch)),
        "torch_npu":
        impl_record("torch_npu", "task_npu_baseline_probe", lambda: run_torch_npu(inputs), expected, args, case.case_id,
                    profile_for_speed=True),
    }
    cand_active_us = impls["triton"]["active_window_us"]
    cand_kernel_us = impls["triton"]["kernel_sum_us"]
    for name in ["torch", "torch_npu"]:
        active_vs_active = speedup(impls[name]["active_window_us"], cand_active_us)
        kernel_vs_kernel = speedup(impls[name]["kernel_sum_us"], cand_kernel_us)
        baseline_active_vs_candidate_kernel = speedup(impls[name]["active_window_us"], cand_kernel_us)
        baseline_kernel_vs_candidate_active = speedup(impls[name]["kernel_sum_us"], cand_active_us)
        s = active_vs_active
        impls[name]["speedup_vs_triton"] = s
        impls[name]["speedup_vs_triton_text"] = fmt_x(s)
        impls[name]["speedups_vs_triton"] = {
            "active_vs_active": active_vs_active,
            "active_vs_active_text": fmt_x(active_vs_active),
            "kernel_vs_kernel": kernel_vs_kernel,
            "kernel_vs_kernel_text": fmt_x(kernel_vs_kernel),
            "baseline_active_vs_candidate_kernel": baseline_active_vs_candidate_kernel,
            "baseline_active_vs_candidate_kernel_text": fmt_x(baseline_active_vs_candidate_kernel),
            "baseline_kernel_vs_candidate_active": baseline_kernel_vs_candidate_active,
            "baseline_kernel_vs_candidate_active_text": fmt_x(baseline_kernel_vs_candidate_active),
        }
    selected = "torch_npu" if impls["torch_npu"]["accuracy"]["passed"] else "torch"
    selected_speedup = impls[selected].get("speedup_vs_triton")
    return {
        "case": index,
        "case_id": case.case_id,
        "kind": case.kind,
        "shape": case.shape,
        "case_detail": asdict(case),
        "dtype": "bfloat16",
        "cache_mode": "Norm",
        "epsilon": DEFAULT_EPSILON,
        "timing_policy":
        "Candidate and torch_npu benchmark paths are timed by torch_npu.profiler kernel_details.csv active-window, matching OpForge/CANN-Bench; Torch semantic reference is timed only with --benchmark-torch as an auxiliary comparison and is not the main speed gate.",
        "implementations": impls,
        "selected_baseline": {
            "implementation": selected,
            "source": "task_npu_baseline_probe" if selected == "torch_npu" else "pytorch_semantic_baseline",
            "selection_rule":
            "Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline.",
            "active_window_us": impls[selected]["active_window_us"],
            "active_window": impls[selected]["active_window"],
            "kernel_sum_us": impls[selected]["kernel_sum_us"],
            "kernel_sum": impls[selected]["kernel_sum"],
            "latency_us": impls[selected]["latency_us"],
            "latency": impls[selected]["latency"],
            "speedup_vs_triton": selected_speedup,
            "speedup_vs_triton_text": fmt_x(selected_speedup),
            "speedups_vs_triton": impls[selected].get("speedups_vs_triton", {}),
            "perf_metric_strategy": impls[selected]["perf_metric_strategy"],
            "measurement_scope": impls[selected]["measurement_scope"],
            "elapsed_us_source": impls[selected]["elapsed_us_source"],
        },
    }


def print_record(index: int, total: int, record: dict[str, object]) -> None:
    shape = record["shape"]
    triton = record["implementations"]["triton"]
    torch_npu_impl = record["implementations"]["torch_npu"]
    acc = triton["accuracy"]
    status = "PASS" if acc["passed"] else "FAIL"
    print(
        f"[{status}] {index:03d}/{total:03d} {record['kind']} id={record['case_id']} "
        f"shape=B{shape['Bkv']},N{shape['N']},Skv{shape['Skv']},Scache{shape['Scache']},"
        f"Dk{shape['Dk']},Dv{shape['Dv']} triton_active={triton['active_window']} "
        f"torch_npu_active={torch_npu_impl['active_window']} selected={record['selected_baseline']['implementation']} "
        f"mismatches={acc['mismatch_count']} max_diff={acc['max_diff']:.6g} "
        f"MARE={acc['mare']:.6g} RMSE={acc['rmse']:.6g} "
        f"torch_npu_accuracy={'PASS' if torch_npu_impl['accuracy']['passed'] else 'FAIL'} "
        f"torch_npu_error={torch_npu_impl.get('profile_error') or torch_npu_impl['accuracy'].get('error_msg') or ''}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="run the 20 documented public-style cases")
    parser.add_argument("--random-generalization", type=int, default=0,
                        help="number of seeded random dynamic split cases")
    parser.add_argument("--random-seed", type=int, default=20260617)
    parser.add_argument("--benchmark", action="store_true",
                        help="collect OpForge-compatible kernel_details.csv active-window timing for all paths")
    parser.add_argument(
        "--benchmark-torch", action="store_true",
        help="also profile Torch semantic reference as auxiliary timing; not used for the main speed gate")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(max(v, 1.0e-9)) for v in values) / len(values)) if values else None


def main() -> None:
    args = parse_args()
    if not args.public and args.random_generalization == 0:
        args.public = True
    if _TORCH_NPU_IMPORT_ERROR is not None:
        raise RuntimeError(f"torch_npu import failed; source the Ascend runtime first: {_TORCH_NPU_IMPORT_ERROR}")
    npu_id = int(os.environ.get("NPU_ID", "0"))
    device = torch.device(f"npu:{npu_id}")
    cases: list[Case] = []
    if args.public:
        cases.extend(public_cases())
    cases.extend(random_generalization_cases(args.random_generalization, args.random_seed))
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with (args.jsonl.open("w", encoding="utf-8") if args.jsonl else open(os.devnull, "w", encoding="utf-8")) as jf:
        for idx, case in enumerate(cases, start=1):
            record = run_case(case, device, args, idx)
            records.append(record)
            jf.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            jf.flush()
            print_record(idx, len(cases), record)

    passed = sum(1 for record in records if record["implementations"]["triton"]["accuracy"]["passed"])
    triton_latencies = [
        float(record["implementations"]["triton"]["active_window_us"])
        for record in records
        if record["implementations"]["triton"].get("active_window_us") is not None
    ]
    selected_speedups = [
        float(record["selected_baseline"]["speedup_vs_triton"])
        for record in records
        if record["selected_baseline"].get("speedup_vs_triton") is not None
    ]
    torch_npu_runnable = [
        record for record in records if record["implementations"]["torch_npu"].get("speedup_vs_triton") is not None
    ]
    torch_active_timed = [
        record for record in records if record["implementations"]["torch"].get("active_window_us") is not None
    ]
    torch_speedup_sample = [
        record for record in records if record["implementations"]["torch"].get("speedup_vs_triton") is not None
    ]
    torch_timed_candidate_active_geomean = geomean([
        float(record["implementations"]["triton"]["active_window_us"])
        for record in torch_active_timed
        if record["implementations"]["triton"].get("active_window_us") is not None
    ])
    torch_timed_baseline_active_geomean = geomean([
        float(record["implementations"]["torch"]["active_window_us"])
        for record in torch_active_timed
        if record["implementations"]["torch"].get("active_window_us") is not None
    ])
    torch_semantic_active_speedup_geomean = geomean([
        float(record["implementations"]["torch"]["speedup_vs_triton"])
        for record in torch_speedup_sample
        if record["implementations"]["torch"].get("speedup_vs_triton") is not None
    ])
    torch_npu_runnable_candidate_active_geomean = geomean([
        float(record["implementations"]["triton"]["active_window_us"])
        for record in torch_npu_runnable
        if record["implementations"]["triton"].get("active_window_us") is not None
    ])
    torch_npu_runnable_baseline_active_geomean = geomean([
        float(record["implementations"]["torch_npu"]["active_window_us"])
        for record in torch_npu_runnable
        if record["implementations"]["torch_npu"].get("active_window_us") is not None
    ])
    torch_npu_runnable_all_speedup_geomean = geomean([
        float(record["implementations"]["torch_npu"]["speedup_vs_triton"])
        for record in torch_npu_runnable
        if record["implementations"]["torch_npu"].get("speedup_vs_triton") is not None
    ])
    summary: dict[str, object] = {
        "schema_version":
        4,
        "source_jsonl":
        str(args.jsonl) if args.jsonl else "",
        "total_cases":
        len(records),
        "passed":
        passed,
        "failed":
        len(records) - passed,
        "public_cases":
        sum(1 for record in records if record["kind"] == "public"),
        "random_generalization_cases":
        sum(1 for record in records if record["kind"] == "random_generalization"),
        "random_seed":
        args.random_seed,
        "benchmark":
        bool(args.benchmark),
        "benchmark_torch":
        bool(args.benchmark_torch),
        "timing_source":
        "kernel_details.csv.active_window_median" if args.benchmark else "",
        "candidate_active_geomean_us":
        geomean(triton_latencies),
        "selected_active_speedup_geomean":
        geomean(selected_speedups),
        "selected_baseline_counts": {
            name: sum(1
                      for record in records
                      if record["selected_baseline"]["implementation"] == name)
            for name in ["torch_npu", "torch"]
        },
        "torch_npu_runnable":
        len(torch_npu_runnable),
        "torch_npu_accuracy_passed":
        sum(1 for record in records if record["implementations"]["torch_npu"]["accuracy"]["passed"]),
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
        "max_rmse":
        max(float(record["implementations"]["triton"]["accuracy"].get("rmse") or 0.0)
            for record in records) if records else 0.0,
        "notes": [
            "Torch and torch_npu baselines are generated inside this operator directory.",
            "Torch semantic timing is collected only when --benchmark-torch is set and remains an auxiliary comparison.",
            "RMSE/MERE/MARE are emitted by validate_kv_rms_norm_rope_cache.py.",
            "No external historical evaluation CSV/JSON is used by this self-validation flow.",
        ],
    }
    if args.summary_json:
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SUMMARY total={summary['total_cases']} passed={summary['passed']} failed={summary['failed']} "
          f"public={summary['public_cases']} random={summary['random_generalization_cases']} "
          f"torch_npu_runnable={summary['torch_npu_runnable']} "
          f"torch_npu_accuracy_passed={summary['torch_npu_accuracy_passed']} "
          f"torch_timed={summary['torch_timed']} "
          f"torch_speedup_sample={summary['torch_speedup_sample']} "
          f"torch_semantic_speedup={summary['torch_semantic_active_speedup_geomean']} "
          f"torch_npu_runnable_all_speedup={summary['torch_npu_runnable_all_active_speedup_geomean']} "
          f"torch_npu_runnable_all_gate={summary['torch_npu_runnable_all_speed_gate']} "
          f"max_rmse={summary['max_rmse']:.6g}")
    if args.jsonl:
        print(f"CANONICAL_JSONL path={args.jsonl}")
    if args.summary_json:
        print(f"CANONICAL_JSONL_SUMMARY path={args.summary_json}")
    if passed != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
