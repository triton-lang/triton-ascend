#!/usr/bin/env python3
"""Source-contract checks for Projected logical boundaries."""

from pathlib import Path
import re


TEST_DIR = Path(__file__).resolve().parent
CVSPLIT_LIB = TEST_DIR.parent
ASCEND_ROOT = TEST_DIR.parents[2]
PIPELINE_HEADER = (
    ASCEND_ROOT / "include" / "CVSplitScheduling" / "CrossCorePipelinePlan.h"
)
PREDICATE_HEADER = (
    ASCEND_ROOT / "include" / "CVSplitScheduling" / "VectorAccumulatorMatmul.h"
)
PIPELINE_SOURCE = CVSPLIT_LIB / "CrossCorePipelinePlan.cpp"
PREDICATE_SOURCE = CVSPLIT_LIB / "VectorAccumulatorMatmul.cpp"
UNFUSE_SOURCE = CVSPLIT_LIB / "UnfusePVMatmuls.cpp"
RESOURCE_HEADER = (
    ASCEND_ROOT / "include" / "CVSplitScheduling" / "CrossCoreResourcePlan.h"
)
RESOURCE_SOURCE = CVSPLIT_LIB / "CrossCoreResourcePlan.cpp"
SCHEDULER_SOURCE = CVSPLIT_LIB / "DependencyScheduler.cpp"
CMAKE = CVSPLIT_LIB / "CMakeLists.txt"


def test_projected_boundary_has_stable_key_and_optional_anchors() -> None:
    text = PIPELINE_HEADER.read_text()
    required = {
        "CrossCoreBoundaryKey",
        "originId",
        "lane",
        "direction",
        "resultNumber",
        "BoundaryMaterialization",
        "Observed",
        "PostUnfuseDpsJoin",
        "std::optional<unsigned> lastReaderOrder",
    }
    missing = sorted(token for token in required if token not in text)
    assert not missing, f"missing projected-boundary API: {missing}"


def test_analysis_and_rewrite_share_one_structural_predicate() -> None:
    declaration = PREDICATE_HEADER.read_text()
    predicate = PREDICATE_SOURCE.read_text()
    pipeline = PIPELINE_SOURCE.read_text()
    unfuse = UNFUSE_SOURCE.read_text()
    name = "isVectorAccumulatorMatmul"
    assert name in declaration
    assert predicate.count(name) == 1
    assert pipeline.count(name) == 1
    assert unfuse.count(name) == 1
    assert "getDpsInitOperand(0)" in predicate
    assert "EngineType::VECTOR" in predicate
    assert "getDpsInitOperand(0)" not in pipeline


def test_projection_is_lane_generic_and_does_not_mutate_ir() -> None:
    text = PIPELINE_HEADER.read_text() + PIPELINE_SOURCE.read_text()
    forbidden_patterns = [
        r"_attn_fwd",
        r"flash[_ -]?attention",
        r"head[_ -]?dim",
        r"\bHD(?:64|128)\b",
        r"lane\s*==\s*[0-9]+",
        r"unrollFactor\s*==\s*[0-9]+",
        r"projectsAccumulatorJoin\s*&&\s*.*Yield",
    ]
    matches = [
        pattern for pattern in forbidden_patterns if re.search(pattern, text, re.I)
    ]
    assert not matches, f"specialized projected-boundary policy: {matches}"
    forbidden_mutations = [
        "builder.create",
        "moveBefore",
        "moveAfter",
        "replaceAllUsesWith",
        "replaceUsesOfWith",
        "setAttr(",
        "erase()",
        "unfusePVMatmuls(",
    ]
    present = [token for token in forbidden_mutations if token in PIPELINE_SOURCE.read_text()]
    assert not present, f"projected-boundary analysis mutates IR: {present}"


def test_unbound_anchors_are_explicitly_non_selectable() -> None:
    header = RESOURCE_HEADER.read_text()
    source = RESOURCE_SOURCE.read_text()
    assert "PendingMaterialization" in header
    assert "anchorsComplete" in header
    assert "else if (!plan.anchorsComplete)" in source
    pending = source.index("ResourcePlanStatus::PendingMaterialization")
    selection = source.index("plan.selectionEligible")
    assert pending < selection


def test_existing_scheduler_uses_the_stable_boundary_key() -> None:
    text = SCHEDULER_SOURCE.read_text()
    assert "earlyBoundary->key.originId" in text
    assert "earlyBoundary->key.lane" in text
    assert not re.search(r"earlyBoundary->(?:originId|lane|direction)", text)


def test_shared_predicate_is_built() -> None:
    assert "VectorAccumulatorMatmul.cpp" in CMAKE.read_text()


if __name__ == "__main__":
    test_projected_boundary_has_stable_key_and_optional_anchors()
    test_analysis_and_rewrite_share_one_structural_predicate()
    test_projection_is_lane_generic_and_does_not_mutate_ir()
    test_unbound_anchors_are_explicitly_non_selectable()
    test_existing_scheduler_uses_the_stable_boundary_key()
    test_shared_predicate_is_built()
    print("Projected-boundary source contract: PASS")
