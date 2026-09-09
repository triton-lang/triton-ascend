"""Run task_0005 correctness and profiler-artifact baselines.

The runner intentionally has no default artifact or cache path. Its caller
must choose a task-owned output tree and an explicit clean TRITON_CACHE_DIR.
This makes a profile result attributable to the exact launch, rather than to a
randomly reused cache entry or a pytest wall-clock duration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import triton

from .contracts import (
    COMPILE_MODE,
    LOGITS_PRIMARY,
    MERGE_PRIMARY_CASES,
    NORM_PRIMARY_TOKENS,
    PHYSICAL_NPU,
    TARGET_ARCH,
    LogitsCase,
    MergeCase,
    NormRopeCase,
    artifact_schema,
    case_payload,
    expected_program_count,
)
from .reference import (
    bf16_peak_ulp,
    max_abs_error,
    ref_indexer_logits,
    ref_indexer_norm_rope,
    ref_merge_split_states,
)
from .runtime import (
    LaunchResult,
    collect_cache_records,
    compiler_identity,
    copy_cache_artifacts,
    launch_indexer_logits,
    launch_merge_split,
    launch_norm_rope,
    make_logits_inputs,
    make_merge_inputs,
    make_norm_rope_inputs,
    npu_available,
    npu_identity,
    validate_primary_metadata,
)
from .grid_specialization import (
    expected_grid_coverage,
    run_unchanged_dsl_grid_baseline,
)

_PERFORMANCE_COLUMNS = tuple(artifact_schema()["performance_columns"])


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_load(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_cache_root() -> Path:
    raw = os.environ.get("TRITON_CACHE_DIR")
    if not raw:
        raise RuntimeError("TRITON_CACHE_DIR must be set to a task-owned dedicated cache")
    root = Path(raw).expanduser().resolve()
    rejected = {
        Path("/"),
        Path("/home/w00609825"),
        Path("/home/w00609825/triton_workspace"),
    }
    if root in rejected:
        raise RuntimeError(f"refusing non-dedicated TRITON_CACHE_DIR: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _environment_payload() -> dict[str, Any]:
    package_version = getattr(triton, "__version__", "unknown")
    return {
        "schema_version": 1,
        "captured_at_utc": _timestamp(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
        },
        "torch_version": torch.__version__,
        "triton_version": package_version,
        "npu": npu_identity(),
        "compiler": compiler_identity(),
        "target_arch": TARGET_ARCH,
        "compile_mode": COMPILE_MODE,
    }


def _write_environment(root: Path, payload: dict[str, Any]) -> None:
    npu = payload["npu"]
    compiler = payload["compiler"]
    lines = (
        "# task_0005 baseline environment",
        "",
        f"- captured_at_utc: {payload['captured_at_utc']}",
        f"- python: {payload['python']['executable']}",
        f"- torch: {payload['torch_version']}",
        f"- triton: {payload['triton_version']}",
        f"- physical NPU requested: {npu.get('physical_npu_env')} (task requires {PHYSICAL_NPU})",
        f"- runtime NPU: {npu.get('runtime_name')}",
        f"- target arch: {payload['target_arch']}",
        f"- compile mode: {payload['compile_mode']}",
        f"- compiler resolved by _get_npucompiler_path(): {compiler.get('resolved_path')}",
        f"- compiler SHA-256: {compiler.get('sha256')}",
        f"- compiler identity matched: {compiler.get('matches_expected')}",
        "",
        "The compiler identity is queried from this Python process; shell PATH alone is not evidence.",
        "",
    )
    (root / "environment.md").write_text("\n".join(lines))


def _program_rows(operator: str, case: MergeCase | NormRopeCase | LogitsCase,
                  launch: LaunchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for specialization, grid in launch.grids.items():
        expected = expected_program_count(operator, case,
                                          specialization=specialization if operator == "norm_rope" else None)
        actual = math.prod(grid)
        rows.append({
            "operator": operator,
            "specialization": specialization,
            "grid": list(grid),
            "expected_program_count": expected,
            "launch_program_count": actual,
            "matches_before_contract": actual == expected,
        })
    return rows


def _merge_correctness(inputs: dict[str, torch.Tensor], result: LaunchResult) -> tuple[bool, dict[str, Any]]:
    expected_out, expected_lse = ref_merge_split_states(inputs["partial_out"], inputs["partial_lse"])
    actual_out, actual_lse = result.outputs
    out_ok = torch.allclose(actual_out, expected_out, rtol=1.0e-2, atol=4.0e-3)
    lse_ok = torch.allclose(actual_lse, expected_lse, rtol=1.0e-3, atol=1.0e-3)
    finite = bool(torch.isfinite(actual_out).all() and torch.isfinite(actual_lse).all())
    return bool(out_ok and lse_ok and finite), {
        "output": {
            "rtol": 1.0e-2,
            "atol": 4.0e-3,
            "max_abs_error": max_abs_error(actual_out, expected_out),
        },
        "lse": {
            "rtol": 1.0e-3,
            "atol": 1.0e-3,
            "max_abs_error": max_abs_error(actual_lse, expected_lse),
        },
        "outputs_finite": finite,
    }


def _norm_correctness(inputs: dict[str, torch.Tensor], result: LaunchResult) -> tuple[bool, dict[str, Any]]:
    expected = ref_indexer_norm_rope(
        inputs["q"],
        inputs["k"],
        inputs["z"],
        inputs["q_weight"],
        inputs["k_weight"],
        inputs["k_bias"],
        inputs["cos"],
        inputs["sin"],
        inputs["positions"],
    )
    labels = ("q", "k")
    checks: dict[str, Any] = {}
    passed = True
    for label, actual, reference in zip(labels, result.outputs[:2], expected[:2]):
        error = max_abs_error(actual, reference)
        limit = 2.0 * bf16_peak_ulp(reference)
        finite = bool(torch.isfinite(actual).all())
        checks[label] = {
            "max_abs_error": error,
            "limit": limit,
            "finite": finite,
        }
        passed = passed and finite and error <= limit
    z_equal = bool(torch.equal(result.outputs[2], expected[2]))
    checks["z"] = {"bitwise_passthrough": z_equal}
    return passed and z_equal, checks


def _logits_correctness(inputs: dict[str, torch.Tensor], result: LaunchResult) -> tuple[bool, dict[str, Any]]:
    expected = ref_indexer_logits(inputs["q"], inputs["weights"], inputs["k"])
    actual = result.outputs[0]
    # This baseline oracle is frozen before StaticProgramAxisFusionRule. It
    # permits only normal dot-accumulation variation, not a relaxed policy.
    rtol, atol = 1.0e-2, 5.0e-2
    finite = bool(torch.isfinite(actual).all())
    passed = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)) and finite
    return passed, {
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs_error(actual, expected),
        "outputs_finite": finite,
    }


def _case_id(operator: str, case: MergeCase | NormRopeCase | LogitsCase) -> str:
    if operator == "merge_split":
        assert isinstance(case, MergeCase)
        return f"merge-s{case.num_splits}-t{case.tokens}-h{case.heads}-d{case.head_dim}"
    if operator == "norm_rope":
        assert isinstance(case, NormRopeCase)
        return f"norm-t{case.tokens}-q{case.q_heads}-k{case.k_heads}-d{case.head_dim}"
    assert isinstance(case, LogitsCase)
    return f"logits-q{case.seq_q}-k{case.seq_k}-g{case.groups}-d{case.proxy_dim}"


def _run_case(
    operator: str,
    case: MergeCase | NormRopeCase | LogitsCase,
    *,
    graph_optimize: bool,
    warmup: int,
    active_samples: int,
    cache_root: Path,
    golden_root: Path,
) -> dict[str, Any]:
    device = torch.device("npu")
    if operator == "merge_split":
        assert isinstance(case, MergeCase)
        inputs = make_merge_inputs(case, device=device)
        launch = lambda: launch_merge_split(inputs, case, graph_optimize=graph_optimize)
        check = _merge_correctness
    elif operator == "norm_rope":
        assert isinstance(case, NormRopeCase)
        inputs = make_norm_rope_inputs(case, device=device)
        launch = lambda: launch_norm_rope(inputs, case, graph_optimize=graph_optimize)
        check = _norm_correctness
    elif operator == "indexer_logits":
        assert isinstance(case, LogitsCase)
        inputs = make_logits_inputs(case, device=device)
        launch = lambda: launch_indexer_logits(inputs, case, graph_optimize=graph_optimize)
        check = _logits_correctness
    else:
        raise ValueError(f"unknown operator {operator}")

    initial = launch()
    torch.npu.synchronize()
    passed, checks = check(inputs, initial)

    # Warm-ups and active launches stay in one profiler process. The profile
    # parser selects every final Name-matched invocation, including both Q and
    # K launches of the shared norm+RoPE kernel name.
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()
    for _ in range(active_samples):
        launch()
    torch.npu.synchronize()

    records = collect_cache_records(cache_root, initial.kernel_names)
    metadata_errors = validate_primary_metadata(records)
    case_id = _case_id(operator, case)
    copied_records = copy_cache_artifacts(
        records,
        golden_root / operator / case_id / ("graph_on" if graph_optimize else "graph_off"),
    )
    return {
        "operator": operator,
        "case": case_id,
        "case_parameters": case_payload(case),
        "graph_optimize": graph_optimize,
        "compile_mode": COMPILE_MODE,
        "correctness_passed": passed,
        "checks": checks,
        "program_count": _program_rows(operator, case, initial),
        "kernel_names": list(initial.kernel_names),
        "kernel_invocations_per_launch": initial.kernel_invocations,
        "warmup_samples": warmup,
        "active_samples": active_samples,
        "cache_records": copied_records,
        "metadata_errors": metadata_errors,
    }


def _iter_cases(args: argparse.Namespace) -> Iterable[tuple[str, Any]]:
    selected = ("merge_split", "norm_rope", "indexer_logits")
    if args.operator != "all":
        selected = (args.operator, )
    if "merge_split" in selected:
        wanted = set(args.splits) if args.splits else None
        for case in MERGE_PRIMARY_CASES:
            if wanted is None or case.num_splits in wanted:
                yield "merge_split", case
    if "norm_rope" in selected:
        tokens = tuple(args.norm_tokens) if args.norm_tokens else NORM_PRIMARY_TOKENS
        for token_count in tokens:
            yield "norm_rope", NormRopeCase(tokens=token_count)
    if "indexer_logits" in selected:
        yield "indexer_logits", LOGITS_PRIMARY


def run(args: argparse.Namespace) -> int:
    if not npu_available():
        raise RuntimeError("NPU unavailable; full-runtime baseline cannot be claimed")
    cache_root = _validate_cache_root()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = _environment_payload()
    _write_environment(root, environment)
    if environment["npu"].get("physical_npu_env") != str(PHYSICAL_NPU):
        raise RuntimeError(f"ASCEND_RT_VISIBLE_DEVICES must be {PHYSICAL_NPU}, got "
                           f"{environment['npu'].get('physical_npu_env')!r}")
    if not environment["compiler"].get("matches_expected"):
        raise RuntimeError("external BiSheng compiler identity did not match task_0005 requirement")

    label = args.run_label or f"{args.operator}-{args.variant}-{_timestamp()}"
    run_dir = root / "runs" / label
    if run_dir.exists():
        raise RuntimeError(f"run label already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    graph_optimize = args.variant == "baseline"
    cases: list[dict[str, Any]] = []
    for operator, case in _iter_cases(args):
        cases.append(
            _run_case(
                operator,
                case,
                graph_optimize=graph_optimize,
                warmup=args.warmup,
                active_samples=args.active_samples,
                cache_root=cache_root,
                golden_root=root / "golden",
            ))

    all_correct = all(case["correctness_passed"] for case in cases)
    all_metadata = all(not case["metadata_errors"] for case in cases)
    record = {
        "schema_version": 1,
        "run_label": label,
        "variant": args.variant,
        "graph_optimize": graph_optimize,
        "cache_root": str(cache_root),
        "environment": environment,
        "cases": cases,
        "passed": all_correct and all_metadata,
    }
    _write_json(run_dir / "run_record.json", record)

    correctness_path = root / "correctness.json"
    correctness = _json_load(correctness_path, {"schema_version": 1, "runs": [], "schema": artifact_schema()})
    correctness["runs"].append(record)
    _write_json(correctness_path, correctness)

    program_path = root / "program_count.json"
    programs = _json_load(program_path, {"schema_version": 1, "runs": [], "schema": artifact_schema()})
    programs["runs"].append({
        "run_label":
        label,
        "variant":
        args.variant,
        "graph_optimize":
        graph_optimize,
        "cases": [{
            "operator": case["operator"],
            "case": case["case"],
            "program_count": case["program_count"],
        } for case in cases],
    })
    _write_json(program_path, programs)

    performance_path = root / "performance.csv"
    if not performance_path.exists():
        with performance_path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=_PERFORMANCE_COLUMNS).writeheader()

    print(json.dumps({"run_record": str(run_dir / "run_record.json"), "passed": record["passed"]}))
    return 0 if record["passed"] else 1


def run_grid_observation(args: argparse.Namespace) -> int:
    """Emit the task_0002-v2-ready unchanged-DSL default-off observation."""

    if not npu_available():
        raise RuntimeError("NPU unavailable; grid observation cannot be claimed")
    cache_root = _validate_cache_root()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = _environment_payload()
    _write_environment(root, environment)
    if environment["npu"].get("physical_npu_env") != str(PHYSICAL_NPU):
        raise RuntimeError(f"ASCEND_RT_VISIBLE_DEVICES must be {PHYSICAL_NPU}, got "
                           f"{environment['npu'].get('physical_npu_env')!r}")
    if not environment["compiler"].get("matches_expected"):
        raise RuntimeError("external BiSheng compiler identity did not match task_0005 requirement")

    label = args.label or f"grid-specialization-off-{_timestamp()}"
    output = root / "grid_specialization" / f"{label}.json"
    if output.exists():
        raise RuntimeError(f"grid observation label already exists: {output}")
    report = run_unchanged_dsl_grid_baseline()
    target_names = tuple(expected_grid_coverage())
    records = collect_cache_records(cache_root, target_names)
    metadata_errors = validate_primary_metadata(records)
    report["cache_root"] = str(cache_root)
    report["environment"] = environment
    report["cache_records"] = copy_cache_artifacts(records, root / "golden" / "grid_specialization" / label)
    report["metadata_errors"] = metadata_errors
    report["passed"] = bool(report["passed"] and not metadata_errors)
    _write_json(output, report)
    print(json.dumps({"grid_observation": str(output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


def _csv_name_column(fieldnames: list[str] | None) -> str:
    for candidate in ("Name", "Op Name", "Kernel Name", "op_name"):
        if fieldnames and candidate in fieldnames:
            return candidate
    raise RuntimeError(f"no target-name column in profiler CSV: {fieldnames}")


def _csv_duration_column(fieldnames: list[str] | None) -> str:
    for candidate in (
            "Duration(us)",
            "Task Duration(us)",
            "Duration",
            "duration_us",
            "aicore_time(us)",
    ):
        if fieldnames and candidate in fieldnames:
            return candidate
    raise RuntimeError(f"no duration column in profiler CSV: {fieldnames}")


def _parse_duration(value: str) -> float:
    return float(value.strip().replace("\t", "").replace(",", ""))


def _target_samples(profiler_root: Path, kernel_name: str) -> list[tuple[Path, int, float]]:
    # CANN may export either timestamped CSVs or the canonical names
    # ``kernel_details.csv`` / ``op_summary.csv``.  Both are explicit schema
    # matches; no selection is made based on newest-file timestamps.
    patterns = (
        "kernel_details_*.csv",
        "kernel_details.csv",
        "op_summary_*.csv",
        "op_summary.csv",
    )
    candidates = sorted({path for pattern in patterns for path in profiler_root.rglob(pattern)})
    if not candidates:
        raise RuntimeError(f"no kernel_details/op_summary CSV under {profiler_root}")
    samples: list[tuple[Path, int, float]] = []
    for path in candidates:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            name_column = _csv_name_column(reader.fieldnames)
            duration_column = _csv_duration_column(reader.fieldnames)
            for row_index, row in enumerate(reader, start=2):
                if row.get(name_column) == kernel_name:
                    samples.append((path, row_index, _parse_duration(row[duration_column])))
    return samples


def ingest_profile(args: argparse.Namespace) -> int:
    root = args.artifact_root.resolve()
    run_record = _json_load(args.run_record, None)
    if not run_record:
        raise RuntimeError(f"missing run record: {args.run_record}")
    profiler_root = args.profiler_root.resolve()
    all_rows: list[dict[str, Any]] = []
    summary_cases: list[dict[str, Any]] = []
    for case in run_record["cases"]:
        for kernel_name in case["kernel_names"]:
            samples = _target_samples(profiler_root, kernel_name)
            invocations = int(case.get("kernel_invocations_per_launch", {}).get(kernel_name, 1))
            exact_active_samples = case["active_samples"] * invocations
            exact_warmup_samples = case["warmup_samples"] * invocations
            if len(samples) < exact_active_samples:
                raise RuntimeError(f"{kernel_name}: found {len(samples)} exact samples, need "
                                   f"{exact_active_samples}")
            # Warmup target launches occur first, active target launches last.
            # ``invocations`` is 2 for norm+RoPE because Q and K intentionally
            # share the frozen kernel symbol.
            active = samples[-exact_active_samples:]
            durations = [value for _, _, value in active]
            parallel_mode = next(
                (record.get("parallel_mode")
                 for record in case["cache_records"]
                 if record.get("kernel_name") == kernel_name),
                None,
            )
            for path, row_index, duration in active:
                all_rows.append({
                    "round": args.round,
                    "variant": run_record["variant"],
                    "operator": case["operator"],
                    "case": case["case"],
                    "kernel_name": kernel_name,
                    "source_csv": str(path.relative_to(profiler_root)),
                    "source_row": row_index,
                    "duration_us": f"{duration:.6f}",
                    "warmup_samples": exact_warmup_samples,
                    "active_samples": exact_active_samples,
                    "physical_npu": PHYSICAL_NPU,
                    "compile_mode": COMPILE_MODE,
                    "parallel_mode": parallel_mode,
                })
            ordered = sorted(durations)
            p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
            summary_cases.append({
                "operator": case["operator"],
                "case": case["case"],
                "kernel_name": kernel_name,
                "sample_count": len(durations),
                "active_launches": case["active_samples"],
                "kernel_invocations_per_launch": invocations,
                "median_us": statistics.median(durations),
                "p90_us": ordered[p90_index],
                "min_us": min(durations),
                "max_us": max(durations),
                "stdev_us": statistics.pstdev(durations),
            })

    performance_path = root / "performance.csv"
    write_header = not performance_path.exists()
    with performance_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PERFORMANCE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "schema_version": 1,
        "round": args.round,
        "run_label": run_record["run_label"],
        "variant": run_record["variant"],
        "profiler_root": str(profiler_root),
        "cases": summary_cases,
    }
    _write_json(root / "runs" / run_record["run_label"] / "performance_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--operator",
        choices=("all", "merge_split", "norm_rope", "indexer_logits"),
        default="all",
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "control"),
        required=True,
        help="baseline enables GraphOptimize; control disables it",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active-samples", type=int, default=20)
    parser.add_argument("--splits", type=int, nargs="*")
    parser.add_argument("--norm-tokens", type=int, nargs="*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="compile, check, and emit golden artifacts")
    _common_run_arguments(run_parser)
    grid_parser = subparsers.add_parser(
        "grid-observation",
        help="run original wrappers across raw grids and emit the default-off contract",
    )
    grid_parser.add_argument("--artifact-root", type=Path, required=True)
    grid_parser.add_argument("--label")
    profile_parser = subparsers.add_parser("ingest-profile",
                                           help="filter an explicit profiler tree by exact kernel name")
    profile_parser.add_argument("--artifact-root", type=Path, required=True)
    profile_parser.add_argument("--run-record", type=Path, required=True)
    profile_parser.add_argument("--profiler-root", type=Path, required=True)
    profile_parser.add_argument("--round", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        if args.warmup < 0 or args.active_samples < 20:
            raise RuntimeError("use non-negative warmup and at least 20 active samples")
        return run(args)
    if args.command == "grid-observation":
        return run_grid_observation(args)
    return ingest_profile(args)


if __name__ == "__main__":
    raise SystemExit(main())
