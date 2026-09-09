"""NPU correctness and launcher-contract smoke for PTSM integration.

The runner invokes the unchanged ``_indexer_norm_rope_kernel`` DSL through
the task_0005 runtime helper.  Q uses IAT followed by PTSM, while K exercises
the standalone PTSM path.  It records the exact cache manifests rather than
inferring a transform from a wall-clock measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from triton.backends.ascend.driver import NPUUtils
from triton.backends.ascend.program_grid import apply_program_grid_transforms

UNITTEST_ROOT = Path(__file__).resolve().parent
if str(UNITTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(UNITTEST_ROOT))

from task_0005_harness.contracts import (  # noqa: E402
    COMPILE_MODE, NormRopeCase, assert_fixture_integrity, expected_grid,
)
from task_0005_harness.reference import (  # noqa: E402
    bf16_peak_ulp, max_abs_error, ref_indexer_norm_rope,
)
from task_0005_harness.runtime import (  # noqa: E402
    collect_cache_records, compiler_identity, launch_norm_rope, make_norm_rope_inputs, npu_available, npu_identity,
)

IAT_RULE_MASK = 512
PTSM_RULE_MASK = 2048
Q_RULE_MASK = IAT_RULE_MASK | PTSM_RULE_MASK
K_RULE_MASK = PTSM_RULE_MASK
_KERNEL = "_indexer_norm_rope_kernel"


def _expected_physical_npu() -> str:
    """Use task_0201's card 0 by default, while permitting an explicit handoff."""
    expected = os.environ.get("TASK_0201_EXPECTED_PHYSICAL_NPU", "0")
    if not expected:
        raise RuntimeError("TASK_0201_EXPECTED_PHYSICAL_NPU must not be empty")
    return expected


def _require_dedicated_cache() -> Path:
    raw = os.environ.get("TRITON_CACHE_DIR")
    if not raw:
        raise RuntimeError("TRITON_CACHE_DIR must name a dedicated clean cache")
    cache = Path(raw).resolve()
    forbidden = {
        Path("/"),
        Path("/home/w00609825"),
        Path("/home/w00609825/triton_workspace"),
    }
    if cache in forbidden:
        raise RuntimeError(f"refusing non-dedicated TRITON_CACHE_DIR: {cache}")
    return cache


def _manifest_data(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _rule_mask(data: dict[str, Any]) -> int | None:
    value = data.get("program_mapping_rule_mask")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    specialization = data.get("program_grid_specialization")
    if isinstance(specialization, dict):
        value = specialization.get("rule_mask")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _transforms(data: dict[str, Any]) -> list[dict[str, Any]]:
    contract = data.get("program_grid_transforms")
    values = contract.get("transforms") if isinstance(contract, dict) else None
    return [value for value in values if isinstance(value, dict)] if isinstance(values, list) else []


def _transform_matches(
    transform: dict[str, Any],
    *,
    axis: int,
    factor: int,
    persistent: bool,
    grid_stride_abi_verified: bool,
) -> bool:
    return (transform.get("axis") == axis and transform.get("factor") == factor
            and transform.get("persistent_coverage") is persistent
            and transform.get("grid_stride_abi_verified") is grid_stride_abi_verified)


def _exact_transform_sequence(
    record: dict[str, Any],
    expected: tuple[tuple[int, int, bool, bool], ...],
) -> bool:
    transforms = record.get("program_grid_transforms")
    if not isinstance(transforms, list) or len(transforms) != len(expected):
        return False
    return all(
        _transform_matches(
            transform,
            axis=axis,
            factor=factor,
            persistent=persistent,
            grid_stride_abi_verified=grid_stride_abi_verified,
        ) for transform, (axis, factor, persistent, grid_stride_abi_verified) in zip(transforms, expected))


def _launch_projection(data: dict[str, Any]) -> dict[str, Any]:
    """Apply the same ordered transform/cap arithmetic as the generated driver."""
    specialization = data.get("program_grid_specialization")
    contract = data.get("program_grid_transforms")
    if not isinstance(specialization, dict) or not isinstance(contract, dict):
        return {}
    original_grid = specialization.get("grid")
    if not isinstance(original_grid, list) or len(original_grid) != 3:
        return {}
    try:
        # A practically unbounded core count yields the transformed logical
        # launch; the real vector-core count yields the generated driver's
        # persistent physical cap.
        logical_grid = apply_program_grid_transforms(original_grid, contract, physical_core_count=1 << 60)
        vector_cores = int(NPUUtils().get_aivector_core_num())
        physical_grid = apply_program_grid_transforms(original_grid, contract, physical_core_count=vector_cores)
    except Exception as error:  # pragma: no cover - diagnostic boundary
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "original_grid": list(original_grid),
        "logical_grid": list(logical_grid),
        "physical_grid": list(physical_grid),
        "vector_core_count": vector_cores,
    }


def _cache_evidence(cache: Path) -> list[dict[str, Any]]:
    records = collect_cache_records(cache, (_KERNEL, ))
    for record in records:
        data = _manifest_data(Path(record["manifest"]))
        transforms = _transforms(data)
        record["program_mapping_rule_mask"] = _rule_mask(data)
        record["program_grid_transforms"] = transforms
        record["launch_projection"] = _launch_projection(data)
        record["debug"] = data.get("debug") is True
        ttir_path = Path(record["manifest"]).with_suffix(".ttir")
        try:
            ttir = ttir_path.read_text(errors="replace")
        except OSError:
            ttir = ""
        record["ttir_assert_count"] = ttir.count("tt.assert")
        record["ttir_auto_overflow_assert_count"] = ttir.count("tt.auto_overflow_assert")
        record["iat_axis_one_seen"] = any(
            transform.get("axis") == 1 and transform.get("persistent_coverage") is False for transform in transforms)
        record["ptsm_axis_zero_seen"] = any(
            transform.get("axis") == 0 and transform.get("persistent_coverage") is True
            and transform.get("grid_stride_abi_verified") is True for transform in transforms)
    return records


def _debug_assertion_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify the deliberate debug-mode fail-closed boundary.

    A debug compile retains device assertions in the dependence closure.  The
    mapping rules must therefore leave this norm+RoPE specialization intact
    instead of dropping or bypassing assertions merely to obtain IAT/PTSM.
    The frozen before DSL has automatic overflow assertions, so their TTIR
    marker is a direct, cache-local witness of that behavior.
    """
    debug_enabled = os.environ.get("TRITON_DEBUG", "").lower() not in {"", "0", "false", "off", "no"}
    contract: dict[str, Any] = {
        "debug_enabled": debug_enabled,
        "mapping_records": [],
        "passed": None,
    }
    if not debug_enabled:
        return contract

    mapping_records = [
        record for record in records if record.get("program_mapping_rule_mask") in (Q_RULE_MASK, K_RULE_MASK)
    ]
    contract["mapping_records"] = [{
        "rule_mask": record.get("program_mapping_rule_mask"),
        "debug": record.get("debug"),
        "transforms": record.get("program_grid_transforms"),
        "ttir_assert_count": record.get("ttir_assert_count"),
        "ttir_auto_overflow_assert_count": record.get("ttir_auto_overflow_assert_count"),
    } for record in mapping_records]
    contract["passed"] = bool(
        len(mapping_records) == 2 and all(
            record.get("debug") and not record.get("program_grid_transforms") and record.get("ttir_assert_count", 0) > 0
            and record.get("ttir_auto_overflow_assert_count", 0) == record.get("ttir_assert_count", 0)
            for record in mapping_records))
    return contract


def _record_matches_primary_contract(
    record: dict[str, Any],
    *,
    rule_mask: int,
    transforms: tuple[tuple[int, int, bool, bool], ...],
    original_grid: list[int],
    logical_grid: list[int],
    physical_grid: list[int],
) -> bool:
    projection = record.get("launch_projection")
    return bool(
        record.get("program_mapping_rule_mask") == rule_mask and _exact_transform_sequence(record, transforms)
        and isinstance(projection, dict) and projection.get("original_grid") == original_grid
        and projection.get("logical_grid") == logical_grid and projection.get("physical_grid") == physical_grid)


def _path_evidence(records: list[dict[str, Any]], *, primary_tokens: bool) -> dict[str, bool]:
    evidence = {
        "q_iat_then_ptsm":
        any(
            record.get("program_mapping_rule_mask") == Q_RULE_MASK and record.get("iat_axis_one_seen")
            and record.get("ptsm_axis_zero_seen") for record in records),
        "k_ptsm_only":
        any(
            record.get("program_mapping_rule_mask") == K_RULE_MASK and not record.get("iat_axis_one_seen")
            and record.get("ptsm_axis_zero_seen") for record in records),
    }
    if primary_tokens:
        evidence.update({
            "q_iat16_ptsm4_launcher_projection":
            any(
                _record_matches_primary_contract(
                    record,
                    rule_mask=Q_RULE_MASK,
                    transforms=((1, 16, False, False), (0, 4, True, True)),
                    original_grid=[4096, 16, 1],
                    logical_grid=[1024, 1, 1],
                    physical_grid=[56, 1, 1],
                ) for record in records),
            "k_ptsm64_launcher_projection":
            any(
                _record_matches_primary_contract(
                    record,
                    rule_mask=K_RULE_MASK,
                    transforms=((0, 64, True, True), ),
                    original_grid=[4096, 1, 1],
                    logical_grid=[64, 1, 1],
                    physical_grid=[56, 1, 1],
                ) for record in records),
        })
    return evidence


def run(case: NormRopeCase, *, output: Path, require_transform: bool) -> int:
    cache = _require_dedicated_cache()
    if not npu_available():
        raise RuntimeError("NPU runtime is unavailable")
    npu = npu_identity()
    expected_npu = _expected_physical_npu()
    if npu.get("physical_npu_env") != expected_npu:
        raise RuntimeError(f"ASCEND_RT_VISIBLE_DEVICES must be {expected_npu} for task_0201, got "
                           f"{npu.get('physical_npu_env')!r}")

    assert_fixture_integrity()
    device = torch.device("npu")
    inputs = make_norm_rope_inputs(case, device=device)
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

    baseline = launch_norm_rope(
        inputs,
        case,
        graph_optimize=True,
        q_program_mapping_rule_mask=0,
        k_program_mapping_rule_mask=0,
    )
    torch.npu.synchronize()
    mapped = launch_norm_rope(
        inputs,
        case,
        graph_optimize=True,
        q_program_mapping_rule_mask=Q_RULE_MASK,
        k_program_mapping_rule_mask=K_RULE_MASK,
    )
    torch.npu.synchronize()

    q_error = max_abs_error(mapped.outputs[0], expected[0])
    k_error = max_abs_error(mapped.outputs[1], expected[1])
    q_limit = 2.0 * bf16_peak_ulp(expected[0])
    k_limit = 2.0 * bf16_peak_ulp(expected[1])
    correctness = {
        "q_max_abs_error": q_error,
        "q_limit": q_limit,
        "k_max_abs_error": k_error,
        "k_limit": k_limit,
        "z_bitwise_passthrough": bool(torch.equal(mapped.outputs[2], expected[2])),
        "q_matches_baseline": bool(torch.equal(mapped.outputs[0], baseline.outputs[0])),
        "k_matches_baseline": bool(torch.equal(mapped.outputs[1], baseline.outputs[1])),
    }
    correct = bool(q_error <= q_limit and k_error <= k_limit and correctness["z_bitwise_passthrough"])
    records = _cache_evidence(cache)
    paths = _path_evidence(records, primary_tokens=case.tokens == 4096)
    debug_assertions = _debug_assertion_contract(records)
    evidence_ok = (bool(debug_assertions["passed"]) if debug_assertions["debug_enabled"] else
                   (all(paths.values()) if require_transform else True))
    compiler = compiler_identity()
    payload = {
        "schema_version":
        2,
        "rule":
        "PersistentTaskStripMiningRule",
        "rule_masks": {"q": Q_RULE_MASK, "k": K_RULE_MASK},
        "fixture":
        "example2_indexer_norm_rope_before.py",
        "compile_mode":
        COMPILE_MODE,
        "case":
        asdict(case),
        "original_grids": {
            "q": list(expected_grid("norm_rope", case, specialization="q")),
            "k": list(expected_grid("norm_rope", case, specialization="k")),
        },
        "primary_contract": ({
            "q": {
                "transforms": [
                    {"axis": 1, "factor": 16, "persistent": False},
                    {"axis": 0, "factor": 4, "persistent": True},
                ],
                "logical_grid": [1024, 1, 1],
                "physical_grid": [56, 1, 1],
            },
            "k": {
                "transforms": [
                    {"axis": 0, "factor": 64, "persistent": True},
                ],
                "logical_grid": [64, 1, 1],
                "physical_grid": [56, 1, 1],
            },
        } if case.tokens == 4096 else None),
        "baseline_grid_arguments": {key: list(value)
                                    for key, value in baseline.grids.items()},
        "mapped_grid_arguments": {key: list(value)
                                  for key, value in mapped.grids.items()},
        "correct":
        correct,
        "correctness":
        correctness,
        "require_transform":
        require_transform,
        "path_evidence":
        paths,
        "debug_assertion_contract":
        debug_assertions,
        "evidence_ok":
        evidence_ok,
        "npu":
        npu,
        "compiler":
        compiler,
        "cache_dir":
        str(cache),
        "cache_records":
        records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if correct and evidence_ok and compiler.get("matches_expected") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--allow-no-persistent-transform",
        action="store_true",
        help="record a small-grid specialization even if PTSM is correctly rejected",
    )
    args = parser.parse_args()
    return run(
        NormRopeCase(tokens=args.tokens),
        output=args.output,
        require_transform=not args.allow_no_persistent_transform,
    )


if __name__ == "__main__":
    raise SystemExit(main())
