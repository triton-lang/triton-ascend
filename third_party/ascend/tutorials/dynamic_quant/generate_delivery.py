#!/usr/bin/env python3
"""Generate self-contained tutorial reports from local validation logs only."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from docx import Document
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

COMMERCIAL_STANDARD_COMMIT = "c260c8ab7a9be4823ac8f8a07c60442de9bf141e"
TORCH_NPU_RUNNABLE_ALL_SPEED_GATE = 1.2
REGEN_COMMAND = ("UV_PROJECT_ENVIRONMENT=/tmp/uv-triton-ascend-delivery "
                 "uv run --no-project --with openpyxl --with python-docx python generate_delivery.py")

META_BY_DIR = {
    "add_rms_norm": {
        "title": "AddRmsNorm",
        "display": "AddRmsNorm",
        "source": "add_rms_norm.py",
        "validator_impl": "validate_add_rms_norm.py",
        "category": "浮点计算类",
        "public_total": 80,
        "random_seed": 20260613,
        "function": "z = x1 + x2; yOut = z * rsqrt(mean(z*z)+epsilon) * gamma",
        "inputs": "x1/x2/gamma: BF16 contiguous [B,S,H]; epsilon: positive float",
        "outputs": "yOut: BF16 contiguous [B,S,H]",
        "coverage": "80 public B/S/H cases plus fixed-seed random generalization cases.",
        "design":
        "Triton-Ascend fused row kernel for H<=8192; wider hidden uses same-backend chunked partial-sum/reduce/apply path.",
        "validation_jsonl": "logs/add_rms_norm_validation.jsonl",
        "validation_summary": "logs/add_rms_norm_validation.summary.json",
    },
    "dynamic_quant": {
        "title": "DynamicQuant",
        "display": "DynamicQuant",
        "source": "dynamic_quant.py",
        "validator_impl": "validate_dynamic_quant.py",
        "category": "量化计算类",
        "public_total": 40,
        "random_seed": 20260617,
        "function":
        "Per-token symmetric dynamic quantization on BF16 [B,S,H], output quantized tensor plus FP32 scale.",
        "inputs": "x: BF16 contiguous [B,S,H]; dst_type in {int8,int4}",
        "outputs": "output quantized tensor and FP32 scale [B,S]",
        "coverage": "40 public int8/int4 cases plus fixed-seed random generalization cases.",
        "design":
        "Triton row/row-stride kernel with chunked large-H path; quantized values are stored with the current Triton-Ascend int8 cast semantics.",
        "validation_jsonl": "logs/dynamic_quant_validation.jsonl",
        "validation_summary": "logs/dynamic_quant_validation.summary.json",
    },
    "kv_rms_norm_rope_cache": {
        "title": "KvRmsNormRopeCache",
        "display": "KvRmsNormRopeCache",
        "source": "kv_rms_norm_rope_cache.py",
        "validator_impl": "validate_kv_rms_norm_rope_cache.py",
        "category": "浮点计算类",
        "public_total": 20,
        "random_seed": 20260617,
        "function": "Decode KV split, RoPE, RMSNorm, and cache update; outputs updated K cache and CKV cache.",
        "inputs": "kv/gamma/cos/sin/index/cache tensors; BF16 data path with dynamic Dk/Dv validation",
        "outputs": "updated K cache and CKV cache tensors",
        "coverage": "20 public cases plus fixed-seed random dynamic-split cases.",
        "design": "Fast 64/64 split path plus generic dynamic Dk/Dv split path; unsupported metadata fails loudly.",
        "validation_jsonl": "logs/kv_rms_norm_rope_cache_validation.jsonl",
        "validation_summary": "logs/kv_rms_norm_rope_cache_validation.summary.json",
    },
    "mrope": {
        "title": "Mrope",
        "display": "MRoPE",
        "source": "mrope.py",
        "validator_impl": "validate_mrope.py",
        "category": "浮点计算类",
        "public_total": 20,
        "random_seed": 20260617,
        "function": "Apply RoPE/MRoPE rotation to BF16 query/key tensors and return query_out/key_out.",
        "inputs": "positions, query, key, cos_sin_cache plus head_size/mrope_section/rotary_mode/cache_mode attributes",
        "outputs": "query_out and key_out with same shape/dtype as query/key",
        "coverage": "20 embedded public cases plus fixed-seed random metadata generalization cases.",
        "design": "Optimized pair kernels for common RoPE/MRoPE modes plus generic Triton kernel for legal metadata.",
        "validation_jsonl": "logs/mrope_validation.jsonl",
        "validation_summary": "logs/mrope_validation.summary.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reject_obsolete_artifacts() -> None:
    obsolete = []
    obsolete.extend(Path(".").glob("OPFORGE_EVIDENCE.json"))
    if Path("logs").is_dir():
        obsolete.extend(Path("logs").glob("opforge_*"))
        obsolete.extend(Path("logs").glob("*20260615*"))
        obsolete.extend(Path("logs").glob("*20260617*"))
        obsolete.extend(Path("logs").glob("*20260618*"))
    if obsolete:
        names = ", ".join(str(path) for path in sorted(obsolete))
        raise SystemExit(f"obsolete historical evidence files must be removed: {names}")


def find_validation_files(meta: dict[str, Any]) -> tuple[Path, Path]:
    canonical = Path(meta["validation_jsonl"])
    summary = Path(meta["validation_summary"])
    if not canonical.is_file() or not summary.is_file():
        raise SystemExit(
            f"missing canonical validation files. Run: python run_inference.py --public --random-generalization 40 "
            f"--random-seed {meta['random_seed']} --benchmark --benchmark-torch --warmup 1 --repeat 3 "
            f"--jsonl {meta['validation_jsonl']} "
            f"--summary-json {meta['validation_summary']}")
    extra_jsonl = sorted(path for path in Path("logs").glob("*validation*.jsonl") if path != canonical)
    if extra_jsonl:
        names = ", ".join(str(path) for path in extra_jsonl)
        raise SystemExit(f"non-canonical validation JSONL files are not supported: {names}")
    return canonical, summary


def fnum(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_us(value: Any) -> str:
    v = fnum(value)
    return "N/A" if v is None else f"{v:.3f} us"


def fmt_x(value: Any) -> str:
    v = fnum(value)
    return "N/A" if v is None else f"{v:.6f}x"


def fmt_num(value: Any) -> str:
    v = fnum(value)
    if v is None:
        return str(value) if value not in (None, "") else "N/A"
    return f"{v:.6g}"


def geomean(values: list[float]) -> float | None:
    values = [float(v) for v in values if v and float(v) > 0.0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def acc(record: dict[str, Any], impl: str) -> dict[str, Any]:
    return ((record.get("implementations") or {}).get(impl) or {}).get("accuracy") or {}


def impl(record: dict[str, Any], name: str) -> dict[str, Any]:
    return (record.get("implementations") or {}).get(name) or {}


def output_result_metric(record: dict[str, Any], impl_name: str, output_name: str, metric: str) -> float | None:
    for item in acc(record, impl_name).get("output_results") or []:
        if item.get("name") == output_name:
            return fnum(item.get(metric))
    return None


def max_metric(records: list[dict[str, Any]], impl_name: str, metric: str) -> float:
    values = [fnum(acc(r, impl_name).get(metric)) for r in records]
    return max([v for v in values if v is not None], default=0.0)


def max_output_metric(records: list[dict[str, Any]], impl_name: str, output_name: str, metric: str) -> float:
    values = [output_result_metric(r, impl_name, output_name, metric) for r in records]
    return max([v for v in values if v is not None], default=0.0)


def shape_text(record: dict[str, Any]) -> str:
    if "shape" in record:
        return json.dumps(record["shape"], ensure_ascii=False)
    if "input_shape" in record:
        return json.dumps(record["input_shape"], ensure_ascii=False)
    return ""


def public_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == "public"]


def random_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == "random_generalization"]


def selected_impl(record: dict[str, Any]) -> str:
    return str((record.get("selected_baseline") or {}).get("implementation") or "torch")


def speedup(record: dict[str, Any], base: str) -> float | None:
    value = impl(record, base).get("speedup_vs_triton")
    if value is not None:
        return fnum(value)
    cand = fnum(impl(record, "triton").get("active_window_us"))
    base_us = fnum(impl(record, base).get("active_window_us"))
    if cand and base_us:
        return base_us / cand
    return None


def has_active_timing(record: dict[str, Any], name: str) -> bool:
    data = impl(record, name)
    return fnum(data.get("active_window_us")) is not None


def gate_text(value: float | None) -> str:
    if value is None:
        return "N/A (no torch_npu runnable timed case)"
    return f"{'PASS' if value >= TORCH_NPU_RUNNABLE_ALL_SPEED_GATE else 'FAIL'} >= {TORCH_NPU_RUNNABLE_ALL_SPEED_GATE:.1f}x"


def summary_items(meta: dict[str, Any], records: list[dict[str, Any]], jsonl_path: Path) -> list[tuple[str, Any]]:
    pubs = public_rows(records)
    selected_speedups = [
        fnum((r.get("selected_baseline") or {}).get("speedup_vs_triton"))
        for r in pubs
        if fnum((r.get("selected_baseline") or {}).get("speedup_vs_triton")) is not None
    ]
    candidate_us = [
        fnum(impl(r, "triton").get("active_window_us"))
        for r in pubs
        if fnum(impl(r, "triton").get("active_window_us"))
    ]
    selected_us = [
        fnum((r.get("selected_baseline") or {}).get("active_window_us"))
        for r in pubs
        if fnum((r.get("selected_baseline") or {}).get("active_window_us"))
    ]
    selected_counts = Counter(selected_impl(r) for r in pubs)
    overall_selected_counts = Counter(selected_impl(r) for r in records)
    tnpu_all = [r for r in records if has_active_timing(r, "torch_npu")]
    tnpu_all_pass = [r for r in tnpu_all if acc(r, "torch_npu").get("passed")]
    tnpu_all_fail = [r for r in tnpu_all if not acc(r, "torch_npu").get("passed")]
    tnpu_all_speedup = geomean([v for v in (speedup(r, "torch_npu") for r in tnpu_all) if v is not None])
    tnpu_all_candidate_us = geomean(
        [v for v in (fnum(impl(r, "triton").get("active_window_us")) for r in tnpu_all) if v is not None])
    tnpu_all_baseline_us = geomean(
        [v for v in (fnum(impl(r, "torch_npu").get("active_window_us")) for r in tnpu_all) if v is not None])
    torch_all = [r for r in records if has_active_timing(r, "torch")]
    torch_all_speedup = geomean([v for v in (speedup(r, "torch") for r in torch_all) if v is not None])
    torch_all_candidate_us = geomean(
        [v for v in (fnum(impl(r, "triton").get("active_window_us")) for r in torch_all) if v is not None])
    torch_all_baseline_us = geomean(
        [v for v in (fnum(impl(r, "torch").get("active_window_us")) for r in torch_all) if v is not None])
    tnpu = [r for r in pubs if has_active_timing(r, "torch_npu")]
    tnpu_pass = [r for r in tnpu if acc(r, "torch_npu").get("passed")]
    tnpu_fail = [r for r in tnpu if not acc(r, "torch_npu").get("passed")]
    torch_public = [r for r in pubs if has_active_timing(r, "torch")]
    torch_public_speedup = geomean([v for v in (speedup(r, "torch") for r in torch_public) if v is not None])
    torch_public_candidate_us = geomean(
        [v for v in (fnum(impl(r, "triton").get("active_window_us")) for r in torch_public) if v is not None])
    torch_public_baseline_us = geomean(
        [v for v in (fnum(impl(r, "torch").get("active_window_us")) for r in torch_public) if v is not None])
    return [
        ("evidence source", str(jsonl_path)),
        ("total cases", len(records)),
        ("public cases", len(pubs)),
        ("random/generalization cases", len(random_rows(records))),
        ("candidate pass", f"{sum(1 for r in records if acc(r, 'triton').get('passed'))}/{len(records)}"),
        ("main speed sample", f"torch_npu runnable all ({len(tnpu_all)} cases)"),
        ("main speed candidate active geomean", fmt_us(tnpu_all_candidate_us)),
        ("main speed torch_npu active geomean", fmt_us(tnpu_all_baseline_us)),
        ("main speed active/active geomean speedup", fmt_x(tnpu_all_speedup)),
        ("main speed gate", gate_text(tnpu_all_speedup)),
        ("overall selected baseline split", ", ".join(f"{k}={v}" for k, v in sorted(overall_selected_counts.items()))
         or "N/A"),
        ("public selected baseline split", ", ".join(f"{k}={v}" for k, v in sorted(selected_counts.items())) or "N/A"),
        ("overall torch_npu runnable all", len(tnpu_all)),
        ("overall torch_npu accuracy pass/fail", f"{len(tnpu_all_pass)}/{len(tnpu_all_fail)}"),
        ("torch_npu runnable-all active speedup geomean", fmt_x(tnpu_all_speedup)),
        ("torch_npu runnable-all speed gate", gate_text(tnpu_all_speedup)),
        ("public torch_npu runnable all", len(tnpu)),
        ("public torch_npu accuracy pass/fail", f"{len(tnpu_pass)}/{len(tnpu_fail)}"),
        ("aux torch semantic timed all", len(torch_all)),
        ("aux torch semantic candidate active geomean", fmt_us(torch_all_candidate_us)),
        ("aux torch semantic baseline active geomean", fmt_us(torch_all_baseline_us)),
        ("aux torch semantic active/active geomean speedup", fmt_x(torch_all_speedup)),
        ("aux public torch semantic timed", len(torch_public)),
        ("aux public torch semantic candidate active geomean", fmt_us(torch_public_candidate_us)),
        ("aux public torch semantic baseline active geomean", fmt_us(torch_public_baseline_us)),
        ("aux public torch semantic active/active geomean speedup", fmt_x(torch_public_speedup)),
        ("aux public candidate active geomean", fmt_us(geomean([v for v in candidate_us if v is not None]))),
        ("aux public selected baseline active geomean", fmt_us(geomean([v for v in selected_us if v is not None]))),
        ("aux public selected active/active geomean speedup",
         fmt_x(geomean([v for v in selected_speedups if v is not None]))),
        ("max candidate RMSE (all outputs)", fmt_num(max_metric(records, "triton", "rmse"))),
        ("max candidate output RMSE", fmt_num(max_output_metric(records, "triton", "output", "rmse"))),
        ("max candidate scale RMSE", fmt_num(max_output_metric(records, "triton", "scale", "rmse"))),
        ("commercial standard", f"references/commercial_standard.md @ {COMMERCIAL_STANDARD_COMMIT}"),
    ]


SPEED_HEADERS = [
    "Case",
    "Kind",
    "Shape",
    "DType",
    "Selected baseline",
    "Triton active",
    "Torch active",
    "torch_npu active",
    "Selected active speedup",
    "Torch active speedup",
    "torch_npu active speedup",
    "Triton precision",
    "Torch precision",
    "torch_npu precision",
    "MERE",
    "MARE",
    "RMSE",
    "Max diff",
    "torch_npu error/note",
]


def speed_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for r in public_rows(records):
        rows.append([
            r.get("case_id") or r.get("case"),
            r.get("kind", ""),
            shape_text(r),
            r.get("dtype") or "bfloat16",
            selected_impl(r),
            fmt_us(impl(r, "triton").get("active_window_us")),
            fmt_us(impl(r, "torch").get("active_window_us")),
            fmt_us(impl(r, "torch_npu").get("active_window_us")),
            fmt_x((r.get("selected_baseline") or {}).get("speedup_vs_triton")),
            fmt_x(speedup(r, "torch")),
            fmt_x(speedup(r, "torch_npu")),
            "PASS" if acc(r, "triton").get("passed") else "FAIL",
            "PASS" if acc(r, "torch").get("passed") else "FAIL",
            "PASS" if acc(r, "torch_npu").get("passed") else "FAIL",
            fmt_num(acc(r, "triton").get("mere")),
            fmt_num(acc(r, "triton").get("mare")),
            fmt_num(acc(r, "triton").get("rmse")),
            fmt_num(acc(r, "triton").get("max_diff")),
            impl(r, "torch_npu").get("profile_error") or acc(r, "torch_npu").get("error_msg") or "",
        ])
    return rows


PERF_HEADERS = [
    "Scope",
    "Cases",
    "Candidate active geomean",
    "Baseline active geomean",
    "Active/active geomean speedup",
    "Precision pass",
    "Note",
]


def perf_scope_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    pubs = public_rows(records)

    def row(name: str, subset: list[dict[str, Any]], base_name: str | None, note: str) -> list[Any]:
        cand = [fnum(impl(r, "triton").get("active_window_us")) for r in subset]
        if base_name == "selected":
            base = [fnum((r.get("selected_baseline") or {}).get("active_window_us")) for r in subset]
            su = [fnum((r.get("selected_baseline") or {}).get("speedup_vs_triton")) for r in subset]
            precision = sum(1 for r in subset if acc(r, "triton").get("passed"))
        elif base_name == "torch_correctness":
            base = []
            su = []
            precision = sum(1 for r in subset if acc(r, "torch").get("passed"))
        elif base_name:
            base = [fnum(impl(r, base_name).get("active_window_us")) for r in subset]
            su = [speedup(r, base_name) for r in subset]
            precision = sum(1 for r in subset if acc(r, base_name).get("passed"))
        else:
            base = []
            su = []
            precision = sum(1 for r in subset if acc(r, "triton").get("passed"))
        return [
            name,
            len(subset),
            fmt_us(geomean([v for v in cand if v])),
            fmt_us(geomean([v for v in base if v])),
            fmt_x(geomean([v for v in su if v])),
            f"{precision}/{len(subset)}" if subset else "0/0",
            note,
        ]

    tnpu_runnable = [r for r in records if has_active_timing(r, "torch_npu")]
    tnpu_pass = [r for r in tnpu_runnable if acc(r, "torch_npu").get("passed")]
    tnpu_fail = [r for r in tnpu_runnable if not acc(r, "torch_npu").get("passed")]
    torch_timed = [r for r in records if has_active_timing(r, "torch")]
    torch_public_timed = [r for r in pubs if has_active_timing(r, "torch")]
    return [
        row(
            "main torch_npu timed sample", tnpu_runnable, "torch_npu",
            f"主速度验收口径；candidate 和 torch_npu 均只在这同一批有 torch_npu active 计时的 case 上取几何平均；gate >= {TORCH_NPU_RUNNABLE_ALL_SPEED_GATE:.1f}x"
        ),
        row("torch_npu accuracy-pass", tnpu_pass, "torch_npu", "全量 torch_npu 有效计时且本地 checker PASS 子集"),
        row("torch_npu accuracy-fail", tnpu_fail, "torch_npu", "全量 torch_npu 有效计时但本地 checker FAIL 子集"),
        row("aux all torch semantic timed baseline", torch_timed, "torch",
            "补充 Torch 语义参考计时；由 --benchmark-torch 开启，不参与主速度 gate"),
        row("aux public torch semantic timed baseline", torch_public_timed, "torch",
            "补充 public Torch 语义参考计时；由 --benchmark-torch 开启，不参与主速度 gate"),
        row("aux public selected baseline", pubs, "selected", "补充语义标杆口径；torch_npu 仅在本地 checker 通过时选中，否则选 Torch"),
        row("aux public torch semantic correctness baseline", pubs, "torch_correctness",
            "Torch 语义参考覆盖全部 public case；latency/speedup 只看上面的 timed baseline 行"),
    ]


BASELINE_HEADERS = [
    "Case",
    "Selected implementation",
    "Selection rule",
    "Torch pass",
    "torch_npu runnable",
    "torch_npu pass",
    "torch_npu MERE",
    "torch_npu MARE",
    "torch_npu RMSE",
    "torch_npu max diff",
    "Reason",
    "Seed/attrs",
]


def baseline_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for r in public_rows(records):
        tnpu = impl(r, "torch_npu")
        tnpu_acc = acc(r, "torch_npu")
        rows.append([
            r.get("case_id") or r.get("case"),
            selected_impl(r),
            (r.get("selected_baseline") or {}).get("selection_rule", ""),
            "PASS" if acc(r, "torch").get("passed") else "FAIL",
            "NO" if tnpu.get("profile_error") else "YES",
            "PASS" if tnpu_acc.get("passed") else "FAIL",
            fmt_num(tnpu_acc.get("mere")),
            fmt_num(tnpu_acc.get("mare")),
            fmt_num(tnpu_acc.get("rmse")),
            fmt_num(tnpu_acc.get("max_diff")),
            tnpu.get("profile_error") or tnpu_acc.get("error_msg") or "",
            json.dumps(
                {
                    "seed": r.get("seed"),
                    "attrs": r.get("attrs"),
                    "dst_type": r.get("dst_type"),
                    "case_detail": r.get("case_detail"),
                }, ensure_ascii=False),
        ])
    return rows


L1_HEADERS = [
    "Case",
    "Output",
    "DType",
    "Shape",
    "Reference",
    "Criterion",
    "Candidate AE",
    "Candidate MARE",
    "Candidate MERE",
    "Candidate RMSE",
    "Baseline AE",
    "Baseline MARE",
    "Baseline MERE",
    "Baseline RMSE",
    "L1 metric status",
    "Checker status",
    "Note",
]


def l1_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for r in public_rows(records):
        cand_acc = acc(r, "triton")
        base = selected_impl(r)
        base_acc = acc(r, base)
        cand_outputs = cand_acc.get("output_results") or [{
            "name": "output",
            "dtype": r.get("dtype", ""),
            "shape": r.get("shape") or r.get("input_shape"),
            "criterion": f"operator checker threshold={cand_acc.get('threshold')}",
            "max_diff": cand_acc.get("max_diff"),
            "mare": cand_acc.get("mare"),
            "mere": cand_acc.get("mere"),
            "rmse": cand_acc.get("rmse"),
            "passed": cand_acc.get("passed"),
        }]
        base_by_name = {str(o.get("name")): o for o in (base_acc.get("output_results") or [])}
        for out in cand_outputs:
            b = base_by_name.get(str(out.get("name")), {})
            rows.append([
                r.get("case_id") or r.get("case"),
                out.get("name", "output"),
                out.get("dtype") or r.get("dtype") or "bfloat16",
                json.dumps(out.get("shape") or r.get("shape") or r.get("input_shape"), ensure_ascii=False),
                "Torch semantic reference generated by this directory",
                out.get("criterion") or f"operator checker threshold={cand_acc.get('threshold')}",
                fmt_num(out.get("max_diff")),
                fmt_num(out.get("mare")),
                fmt_num(out.get("mere")),
                fmt_num(out.get("rmse")),
                fmt_num(b.get("max_diff", 0.0 if base == "torch" else None)),
                fmt_num(b.get("mare", 0.0 if base == "torch" else None)),
                fmt_num(b.get("mere", 0.0 if base == "torch" else None)),
                fmt_num(b.get("rmse", 0.0 if base == "torch" else None)),
                "PASS" if cand_acc.get("passed") else "FAIL",
                "PASS" if out.get("passed", cand_acc.get("passed")) else "FAIL",
                f"baseline={base}; RMSE由本目录 checker 输出",
            ])
    return rows


def random_table_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for r in random_rows(records):
        a = acc(r, "triton")
        rows.append([
            r.get("case_id") or r.get("case"),
            shape_text(r),
            r.get("random_category") or r.get("dst_type") or "",
            r.get("seed", ""),
            "PASS" if a.get("passed") else "FAIL",
            a.get("mismatch_count", ""),
            fmt_num(a.get("max_diff")),
            fmt_num(a.get("mere")),
            fmt_num(a.get("mare")),
            fmt_num(a.get("rmse")),
        ])
    return rows


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return "\n".join(out)


def write_markdown(meta: dict[str, Any], records: list[dict[str, Any]], jsonl_path: Path,
                   summary: list[tuple[str, Any]]) -> None:
    display = meta["display"]
    validation_cmd = (
        "export NPU_ID=0 && export ASCEND_RT_VISIBLE_DEVICES=$NPU_ID && "
        "export ASCEND_VISIBLE_DEVICES=$NPU_ID && "
        "source /mnt/model/lcw/.local/Ascend-9.0.0/cann-9.0.0/set_env.sh && "
        f"python run_inference.py --public --random-generalization 40 --random-seed {meta['random_seed']} "
        "--benchmark --benchmark-torch --warmup 1 --repeat 3 "
        f"--jsonl {meta['validation_jsonl']} --summary-json {meta['validation_summary']}")
    readme = [
        f"# {display} Triton-Ascend Tutorial",
        "",
        "## 说明",
        "",
        "本目录是自包含交付目录。baseline、精度校验、性能统计和报告生成都由本目录脚本完成；不读取外部历史评测 CSV/JSON，也不保留历史对照文件。",
        "",
        "## 复现命令",
        "",
        "前提：当前 Python 环境已安装 torch/torch_npu，并可导入包含 `triton._C` 编译扩展的 Triton-Ascend；`run_inference.py` 会优先使用本仓库的 `python/triton`。",
        "",
        "```bash",
        validation_cmd,
        REGEN_COMMAND,
        "```",
        "",
        "## 当前证据",
        "",
        table(["指标", "数值"], [[k, v] for k, v in summary]),
        "",
        "## 文件",
        "",
        f"- `{meta['source']}`: Triton-Ascend candidate 实现",
        "- `run_inference.py`: 统一推理入口",
        f"- `{meta['validator_impl']}`: 本地 Torch / torch_npu / candidate baseline 与 checker",
        "- `generate_delivery.py`: 从本目录 logs 重新生成 README/DESIGN/验收报告/DOCX/XLSX",
        "- `references/commercial_standard.md`: 商业精度标准本地副本",
        "",
    ]
    Path("README.md").write_text("\n".join(readme), encoding="utf-8")

    design = [
        f"# {display} 算子设计方案",
        "",
        "## 1. 需求分析",
        "",
        f"- 功能：{meta['function']}",
        f"- 输入：{meta['inputs']}",
        f"- 输出：{meta['outputs']}",
        f"- 覆盖：{meta['coverage']}",
        "",
        "## 2. 实现策略",
        "",
        meta["design"],
        "",
        "被测 candidate 路径只调用本目录 Triton-Ascend 实现，不调用 Torch、torch_npu 高阶等价算子、外部 golden 或历史 baseline 文件。Torch/torch_npu 仅在 `run_inference.py` 校验流程中作为本地 baseline。",
        "",
        "## 3. 精度和 baseline 策略",
        "",
        "本目录生成 Torch 语义参考和 torch_npu baseline probe。torch_npu 只有在运行成功且通过同一 checker 时才作为 selected baseline；否则 selected baseline 为 Torch semantic。candidate 最终精度始终对 Torch 语义参考判定。",
        "",
        "## 4. 性能统计",
        "",
        f"性能统计来自本目录 `run_inference.py --benchmark --benchmark-torch` 的 torch_npu.profiler kernel_details.csv active-window 计时。主速度验收口径为 `torch_npu runnable all` active/active 几何平均 >= {TORCH_NPU_RUNNABLE_ALL_SPEED_GATE:.1f}x；所有 torch_npu 可计时 case 都纳入，精度通过和失败都计入。Torch semantic 计时只作为辅助 Torch 对比，不参与主速度 gate；selected baseline 仅用于语义标杆选择说明。",
        "",
        "## 5. 统计汇总",
        "",
        table(["指标", "数值"], [[k, v] for k, v in summary]),
        "",
        "## 6. 无 fallback / 无 hacking 声明",
        "",
        "实现调度只依赖 dtype、rank、shape、属性、contiguity 等合法运行时元数据，不依赖 case id、workload 文件名、输入取值、输出模式或 timing signature。unsupported contract fail loudly。",
        "",
    ]
    Path("DESIGN.md").write_text("\n".join(design), encoding="utf-8")

    report = [
        f"# {display} 算子自验证报告",
        "",
        "## 1. 报告说明",
        "",
        f"- 单一数值证据源：`{jsonl_path}`",
        "- 本报告由当前目录 `generate_delivery.py` 生成，只读取本目录 logs/templates/references。",
        "- L1 是商业精度等级，不是 L1 norm；本表对齐商业标准中的 MARE/MERE/RMSE 指标口径。",
        "- 本报告是本目录自验证，不等同于完整商业 L1 认证；完整商业认证还要求标准规定的用例规模和执行轮次。",
        "- RMSE/MERE/MARE 由本目录 checker 输出。",
        f"- 速度门槛按 `torch_npu runnable all` active/active 几何平均 >= {TORCH_NPU_RUNNABLE_ALL_SPEED_GATE:.1f}x；无 torch_npu 可计时 case 标记为 N/A，不用 selected baseline 或 Torch semantic timing 代替。",
        "- Torch semantic timing 由 `--benchmark-torch` 显式开启，只作为辅助 Torch 速度对比。",
        "- 截图证据未外置，日志内容嵌入 XLSX `日志证据` 工作表。",
        "- 不保留、不读取外部历史对照文件。",
        "",
        "## 2. 性能总体对比",
        "",
        table(["指标", "数值"], [[k, v] for k, v in summary]),
        "",
        "## 3. 性能口径汇总",
        "",
        table(PERF_HEADERS, perf_scope_rows(records)),
        "",
        "## 4. Baseline 校验明细",
        "",
        table(BASELINE_HEADERS, baseline_rows(records)),
        "",
        "## 5. Public 逐Case速度",
        "",
        table(SPEED_HEADERS, speed_rows(records)),
        "",
        "## 6. 商业L1精度对比",
        "",
        table(L1_HEADERS, l1_rows(records)),
        "",
        "## 7. 随机泛化明细",
        "",
        table(["Case", "Shape", "Category/dst", "Seed", "Status", "Mismatch", "Max diff", "MERE", "MARE", "RMSE"],
              random_table_rows(records)),
        "",
    ]
    Path("SELF_VALIDATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def style(ws, wrap: bool = True) -> None:
    fill = PatternFill("solid", fgColor="D9EAF7")
    side = Side(style="thin", color="CCCCCC")
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    for row in ws.iter_rows():
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=wrap)
            c.border = Border(bottom=side)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = fill
    for col in range(1, ws.max_column + 1):
        width = max(len(str(ws.cell(r, col).value or "")) for r in range(1, min(ws.max_row, 80) + 1)) + 2
        ws.column_dimensions[get_column_letter(col)].width = min(70, max(10, width))
    ws.freeze_panes = "A2"


def add_sheet(wb, name: str, rows: list[list[Any]], wrap: bool = True):
    ws = wb.create_sheet(name)
    for row in rows:
        ws.append(row)
    style(ws, wrap=wrap)
    return ws


def write_xlsx(meta: dict[str, Any], records: list[dict[str, Any]], jsonl_path: Path,
               summary: list[tuple[str, Any]]) -> None:
    wb = load_workbook("templates/XXX_operator_self_validation_report.xlsx")
    ws = wb.active
    ws.title = "自验证结果"
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for r in public_rows(records):
        ws.append([
            shape_text(r),
            r.get("dtype") or "bfloat16",
            "通过" if acc(r, "triton").get("passed") else "失败",
            "见“日志证据”工作表",
            "见“日志证据”工作表",
            fmt_us((r.get("selected_baseline") or {}).get("active_window_us")),
            fmt_us(impl(r, "triton").get("active_window_us")),
            f"selected={selected_impl(r)}; speedup={fmt_x((r.get('selected_baseline') or {}).get('speedup_vs_triton'))}; RMSE={fmt_num(acc(r, 'triton').get('rmse'))}; output_RMSE={fmt_num(acc(r, 'triton').get('output_rmse'))}; scale_RMSE={fmt_num(acc(r, 'triton').get('scale_rmse'))}",
        ])
    style(ws)
    add_sheet(wb, "说明", [
        ["项目", "说明"], ["验证日志", str(jsonl_path)], ["生成命令", REGEN_COMMAND],
        ["baseline流程", "本目录生成 Torch 与 torch_npu baseline；不读取外部历史评测文件"],
        ["L1含义", "商业精度等级，不是 L1 norm；本报告仅做本目录自验证指标对齐，不等同完整商业认证"],
        [
            "速度门槛",
            f"torch_npu runnable all active/active 几何平均 >= {TORCH_NPU_RUNNABLE_ALL_SPEED_GATE:.1f}x；无可计时 case 为 N/A；Torch semantic 计时仅作辅助对比"
        ], ["RMSE", "顶层 RMSE 为所有输出项 RMSE 最大值；同时保留 output_RMSE 与 scale_RMSE"]
    ])
    add_sheet(wb, "性能总体对比", [["指标", "数值"]] + [[k, v] for k, v in summary])
    add_sheet(wb, "性能口径汇总", [PERF_HEADERS] + perf_scope_rows(records))
    add_sheet(wb, "Baseline校验明细", [BASELINE_HEADERS] + baseline_rows(records))
    add_sheet(wb, "逐Case速度", [SPEED_HEADERS] + speed_rows(records))
    add_sheet(wb, "商业L1精度对比", [L1_HEADERS] + l1_rows(records))
    add_sheet(wb, "随机泛化明细",
              [["Case", "Shape", "Category/dst", "Seed", "Status", "Mismatch", "Max diff", "MERE", "MARE", "RMSE"]] +
              random_table_rows(records))
    log_rows = [["证据类型", "日志文件", "行号", "日志内容"]]
    for f in sorted(Path("logs").glob("*")):
        if f.is_file():
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            log_rows.append(["文件摘要", f.name, "", f"{f.stat().st_size} bytes; {len(lines)} lines"])
            for i, line in enumerate(lines[:120], 1):
                log_rows.append(["日志摘录", f.name, i, line[:3000]])
    add_sheet(wb, "日志证据", log_rows, wrap=False)
    wb.save(f"{meta['title']}算子自验证报告.xlsx")


def write_docx(meta: dict[str, Any], summary: list[tuple[str, Any]], records: list[dict[str, Any]]) -> None:
    doc = Document("templates/XXX_operator_design.docx")
    body = doc._element.body
    for child in list(body):
        if child is body.sectPr:
            continue
        body.remove(child)
    doc.add_paragraph(f"{meta['display']}算子设计方案")
    sections = [
        ("需求分析", [meta["function"], f"输入：{meta['inputs']}。", f"输出：{meta['outputs']}。"]),
        ("当前实现分析、算子整体流程", [meta["design"]]),
        ("算子原型", [f"源码 `{meta['source']}`，统一推理入口 `run_inference.py`，校验脚本 `{meta['validator_impl']}`。"]),
        ("相关约束", ["不支持组合 fail loudly；被测路径无 fallback。"]),
        ("需求详细设计", ["调度仅依赖合法运行时元数据，不依赖 case id、workload、输入值或 timing signature。"]),
        ("精度标准/性能标准", [
            f"L1 指标口径参考 references/commercial_standard.md，RMSE/MERE/MARE 由本目录 checker 输出；本报告是教程自验证，不等同完整商业 L1 认证；主速度门槛按 torch_npu runnable all active/active 几何平均 >= {TORCH_NPU_RUNNABLE_ALL_SPEED_GATE:.1f}x，Torch semantic 计时只作辅助对比，selected baseline 只作语义标杆说明。"
        ]),
        ("Baseline 自包含流程", [
            "Torch 和 torch_npu baseline 均由本目录 run_inference.py 生成；torch_npu 只有运行成功且精度通过时才被选为 selected baseline；Torch semantic 计时由 --benchmark-torch 显式开启且不参与主速度 gate；不读取或保留外部历史对照文件。"
        ]),
    ]
    for heading, paragraphs in sections:
        doc.add_paragraph(heading)
        for paragraph in paragraphs:
            doc.add_paragraph(paragraph)
    table1 = doc.add_table(rows=1, cols=2)
    table1.style = "Table Grid"
    table1.rows[0].cells[0].text = "指标"
    table1.rows[0].cells[1].text = "数值"
    for k, v in summary:
        cells = table1.add_row().cells
        cells[0].text = str(k)
        cells[1].text = str(v)
    doc.add_paragraph("性能口径汇总")
    table2 = doc.add_table(rows=1, cols=len(PERF_HEADERS))
    table2.style = "Table Grid"
    for c, h in zip(table2.rows[0].cells, PERF_HEADERS):
        c.text = h
    for row in perf_scope_rows(records):
        cells = table2.add_row().cells
        for c, value in zip(cells, row):
            c.text = str(value)
    doc.save(f"{meta['title']}算子设计方案.docx")


def main() -> None:
    op = Path.cwd().name
    if op not in META_BY_DIR:
        raise SystemExit(f"unsupported operator dir: {op}")
    meta = META_BY_DIR[op]
    reject_obsolete_artifacts()
    jsonl_path, summary_path = find_validation_files(meta)
    records = read_jsonl(jsonl_path)
    read_json(summary_path)
    summary = summary_items(meta, records, jsonl_path)
    write_markdown(meta, records, jsonl_path, summary)
    write_xlsx(meta, records, jsonl_path, summary)
    write_docx(meta, summary, records)
    print(
        f"generated {meta['title']}: public={len(public_rows(records))} random={len(random_rows(records))} source={jsonl_path}"
    )


if __name__ == "__main__":
    main()
