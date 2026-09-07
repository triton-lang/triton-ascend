#!/usr/bin/env python3
"""Ensure only qualified materialization controls remain."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"
PYBIND = ROOT / "third_party/ascend/triton_ascend.cc"


def test_rejected_paths_are_absent() -> None:
    removed_files = (
        INCLUDE / "PurePrerequisiteHoisting.h",
        INCLUDE / "SoftmaxRegroup.h",
        LIB / "PurePrerequisiteHoisting.cpp",
        LIB / "SoftmaxRegroup.cpp",
    )
    assert all(not path.exists() for path in removed_files)
    text = "\n".join(
        path.read_text()
        for path in (
            INCLUDE / "Passes.td",
            LIB / "CVSplitScheduling.cpp",
            LIB / "CrossScopeTransfers.cpp",
            LIB / "CrossCoreResourcePlan.cpp",
            BACKEND,
            PYBIND,
        )
    )
    for removed in (
        "enablePurePrerequisiteHoisting",
        "purePrerequisiteHoistBudgetBytes",
        "promotePrivateBufferPools",
        "promotePrivatePools",
        "sinkScaleIntoFixpipe",
        "regroupSoftmaxMax",
        "l0cPipelineDistance",
        "pipelineDistance",
        "enableCostModelDiagnostics",
        "PostSplitScheduleMode::Analyze",
    ):
        assert removed not in text


def test_qualified_controls_and_safety_remain() -> None:
    text = (LIB / "CVSplitScheduling.cpp").read_text()
    for retained in (
        "enablePlanDrivenEarlyPublish",
        "scheduleCandidateId",
        "privateBufferUbBudgetBytes",
        "PostSplitScheduleMode::Materialize",
        "extractPostCVSplitRequests",
        "buildPostCVSplitSchedulePlan",
        "bindPostCVSplitScheduleAnchors",
        "insertCrossScopeTransfers",
        "createScopeSeparation",
        "kPreserveExplicitScheduleAttr",
        "restoreFunction",
    ):
        assert retained in text


if __name__ == "__main__":
    test_rejected_paths_are_absent()
    test_qualified_controls_and_safety_remain()
    print("Qualified materialization surface contract: PASS")
