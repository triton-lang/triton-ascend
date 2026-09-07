#!/usr/bin/env python3
"""Source-contract checks for Resource feasibility analysis."""

from pathlib import Path
import re


TEST_DIR = Path(__file__).resolve().parent
CVSPLIT_LIB = TEST_DIR.parent
ASCEND_ROOT = TEST_DIR.parents[2]
HEADER = ASCEND_ROOT / "include" / "CVSplitScheduling" / "CrossCoreResourcePlan.h"
SOURCE = CVSPLIT_LIB / "CrossCoreResourcePlan.cpp"
PASS = CVSPLIT_LIB / "CVSplitScheduling.cpp"
CMAKE = CVSPLIT_LIB / "CMakeLists.txt"


def test_resource_plan_has_required_neutral_records() -> None:
    text = HEADER.read_text()
    required = {
        "CrossCoreResourceLimits",
        "ResourceLineagePlan",
        "ResourcePhysicalGroup",
        "ResourceSlotAssignment",
        "ResourceOwnershipEdge",
        "CrossCoreResourcePlan",
        "physicalGroup",
        "slotCount",
        "allocatedUbBytes",
        "allocatedL1Bytes",
        "incrementalUbBytes",
        "requiredFlags",
        "completeLaneCoverage",
        "anchorsComplete",
        "selectionEligible",
    }
    missing = sorted(name for name in required if name not in text)
    assert not missing, f"missing resource-plan fields: {missing}"


def test_resource_policy_is_lane_generic_and_kernel_agnostic() -> None:
    text = HEADER.read_text() + SOURCE.read_text()
    forbidden = [
        r"_attn_fwd",
        r"flash[_ -]?attention",
        r"head[_ -]?dim",
        r"\bHD(?:64|128)\b",
        r"lane\s*==\s*[0-9]+",
        r"unrollFactor\s*==\s*[0-9]+",
        r"laneCount\s*==\s*[0-9]+",
    ]
    matches = [pattern for pattern in forbidden if re.search(pattern, text, re.I)]
    assert not matches, f"kernel-specific resource-plan logic: {matches}"
    assert "std::min(lanes, interCoreBufferDepth)" in SOURCE.read_text()
    assert "left.key.lane != right.key.lane" in SOURCE.read_text()
    assert "boundary.key.lane % lineage.slotCount" in SOURCE.read_text()


def test_resource_analysis_does_not_mutate_ir() -> None:
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
    assert not present, f"resource analysis mutates IR: {present}"


def test_resource_plan_does_not_change_scheduler_policy() -> None:
    text = PASS.read_text()
    resource = text.index("buildCrossCoreResourcePlan")
    scheduler = text.index("cv_split::DependencyScheduler scheduler")
    transfer = text.index("cv_split::insertCrossScopeTransfers")
    assert resource < scheduler < transfer
    scheduler_call = text[text.index("scheduler.run"):text.index("return failure();", scheduler)]
    transfer_call = text[transfer:text.index("if (failed(transferInfo))", transfer)]
    assert "resourcePlan" not in scheduler_call
    assert "materializedPlan" not in scheduler_call
    assert "materializedPlan" in transfer_call
    assert "materializedResources" in transfer_call
    assert "Unresolved" in SOURCE.read_text()
    assert "IncompleteBoundarySet" in SOURCE.read_text()
    assert "PendingMaterialization" in SOURCE.read_text()
    assert "selectionEligible" in SOURCE.read_text()


def test_resource_source_is_built() -> None:
    assert "CrossCoreResourcePlan.cpp" in CMAKE.read_text()


if __name__ == "__main__":
    test_resource_plan_has_required_neutral_records()
    test_resource_policy_is_lane_generic_and_kernel_agnostic()
    test_resource_analysis_does_not_mutate_ir()
    test_resource_plan_does_not_change_scheduler_policy()
    test_resource_source_is_built()
    print("Resource-plan source contract: PASS")
