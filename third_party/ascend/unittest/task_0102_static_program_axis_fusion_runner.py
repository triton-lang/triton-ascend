"""Standalone correctness/compile smoke for StaticProgramAxisFusionRule.

The runner imports the frozen *before* logits DSL through task_0005's launch
helper.  It never adds a GROUPS constexpr argument, a num_stages override, or
another kernel definition: the only opt-in is the compiler rule bit 1024.

Callers must provide a fresh, task-owned TRITON_CACHE_DIR.  The cache records
written by the two paired launches are included in the JSON result so a smoke
run can prove the actual target and parallel mode instead of treating process
wall time as a performance result.
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

UNITTEST_ROOT = Path(__file__).resolve().parent
if str(UNITTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(UNITTEST_ROOT))

from task_0005_harness.contracts import (  # noqa: E402
    COMPILE_MODE, LogitsCase, assert_fixture_integrity, expected_grid,
)
from task_0005_harness.runtime import (  # noqa: E402
    collect_cache_records, compiler_identity, launch_indexer_logits, make_logits_inputs, npu_available, npu_identity,
)

SPAF_RULE_MASK = 1024


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


def _cache_evidence(cache: Path, kernel_name: str) -> list[dict[str, Any]]:
    records = collect_cache_records(cache, (kernel_name, ))
    evidence: list[dict[str, Any]] = []
    for record in records:
        manifest = Path(record["manifest"])
        manifest_data: dict[str, Any] = {}
        if manifest.is_file():
            try:
                parsed = json.loads(manifest.read_text(errors="replace"))
                if isinstance(parsed, dict):
                    manifest_data = parsed
            except json.JSONDecodeError:
                pass
        transforms = manifest_data.get("program_grid_transforms")
        transform_items = (transforms.get("transforms", []) if isinstance(transforms, dict) else [])
        specialization = manifest_data.get("program_grid_specialization")
        rule_mask = manifest_data.get("program_mapping_rule_mask")
        if rule_mask is None and isinstance(specialization, dict):
            rule_mask = specialization.get("rule_mask")
        copied = dict(record)
        copied["program_grid_transform_seen"] = bool(transform_items)
        copied["spaf_rule_mask_seen"] = rule_mask == SPAF_RULE_MASK
        evidence.append(copied)
    return evidence


def run(case: LogitsCase, *, output: Path) -> int:
    if case.groups not in (1, 2, 4):
        raise ValueError("groups must be one of 1, 2, or 4")
    cache = _require_dedicated_cache()
    if not npu_available():
        raise RuntimeError("NPU runtime is unavailable")

    assert_fixture_integrity()
    device = torch.device("npu")
    inputs = make_logits_inputs(case, device=device)

    # Both launches use the unchanged before DSL.  Separate compiler options
    # produce separate cache identities; no manually supplied GROUPS state is
    # passed to the kernel signature.
    baseline = launch_indexer_logits(inputs, case, graph_optimize=True, program_mapping_rule_mask=0)
    torch.npu.synchronize()
    spaf = launch_indexer_logits(
        inputs,
        case,
        graph_optimize=True,
        program_mapping_rule_mask=SPAF_RULE_MASK,
    )
    torch.npu.synchronize()

    baseline_output = baseline.outputs[0]
    spaf_output = spaf.outputs[0]
    correct = bool(
        torch.allclose(spaf_output, baseline_output, rtol=1.0e-2, atol=5.0e-2) and torch.isfinite(spaf_output).all())
    records = _cache_evidence(cache, "_indexer_logits_kernel")
    payload = {
        "schema_version": 1,
        "rule": "StaticProgramAxisFusionRule",
        "rule_mask": SPAF_RULE_MASK,
        "fixture": "example2_indexer_logits_before.py",
        "compile_mode": COMPILE_MODE,
        "case": asdict(case),
        "original_grid": list(expected_grid("indexer_logits", case)),
        "baseline_grid_argument": list(baseline.grids["logits"]),
        "spaf_grid_argument": list(spaf.grids["logits"]),
        "correct": correct,
        "max_abs_difference": float((spaf_output.float() - baseline_output.float()).abs().max().item()),
        "npu": npu_identity(),
        "compiler": compiler_identity(),
        "cache_dir": str(cache),
        "cache_records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if correct else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-q", type=int, default=2048)
    parser.add_argument("--seq-k", type=int, default=16)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--heads-per-group", type=int, default=4)
    parser.add_argument("--proxy-dim", type=int, default=16)
    parser.add_argument("--block-q", type=int, default=16)
    parser.add_argument("--block-k", type=int, default=16)
    args = parser.parse_args()
    case = LogitsCase(
        seq_q=args.seq_q,
        seq_k=args.seq_k,
        groups=args.groups,
        heads_per_group=args.heads_per_group,
        proxy_dim=args.proxy_dim,
        block_q=args.block_q,
        block_k=args.block_k,
    )
    return run(case, output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
