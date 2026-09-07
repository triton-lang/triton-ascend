#!/usr/bin/env python3
"""Source-contract checks for the analysis-only cross-core pipeline model."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "include" / "CVSplitScheduling" / "CrossCorePipelinePlan.h"
SOURCE = ROOT / "lib" / "CVSplitScheduling" / "CrossCorePipelinePlan.cpp"
PASS = ROOT / "lib" / "CVSplitScheduling" / "CVSplitScheduling.cpp"


def test_boundary_model_has_required_general_fields() -> None:
    text = HEADER.read_text()
    required = {
        "originId",
        "lane",
        "direction",
        "value",
        "producer",
        "earliestPublishAnchor",
        "consumers",
        "lastReader",
        "footprintBytes",
        "memorySpace",
        "elementType",
        "logicalShape",
    }
    missing = sorted(name for name in required if name not in text)
    assert not missing, f"missing boundary fields: {missing}"


def test_analysis_is_lane_generic_and_kernel_agnostic() -> None:
    text = HEADER.read_text() + SOURCE.read_text()
    forbidden = [
        r"_attn_fwd",
        r"flash[_ -]?attention",
        r"head[_ -]?dim",
        r"\bHD(?:64|128)\b",
        r"\blane\s*==\s*[0-9]+",
        r"\bunrollFactor\s*==\s*[0-9]+",
    ]
    matches = [pattern for pattern in forbidden if re.search(pattern, text, re.I)]
    assert not matches, f"kernel-specific pipeline-plan logic: {matches}"


def test_analysis_source_does_not_mutate_ir() -> None:
    text = SOURCE.read_text()
    forbidden = [
        "builder.create",
        "moveBefore",
        "moveAfter",
        "replaceAllUsesWith",
        "replaceUsesOfWith",
        "setAttr(",
        "erase()",
    ]
    present = [token for token in forbidden if token in text]
    assert not present, f"analysis-only source mutates IR: {present}"


def test_analysis_runs_before_dependency_scheduler() -> None:
    text = PASS.read_text()
    analysis = text.index("buildCrossCorePipelinePlan")
    scheduler = text.index("cv_split::DependencyScheduler scheduler")
    assert analysis < scheduler


if __name__ == "__main__":
    test_boundary_model_has_required_general_fields()
    test_analysis_is_lane_generic_and_kernel_agnostic()
    test_analysis_source_does_not_mutate_ir()
    test_analysis_runs_before_dependency_scheduler()
    print("pipeline plan source contract: PASS")
