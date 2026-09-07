#!/usr/bin/env python3
"""Contracts for the typed facts consumed by qualified materialization."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"


def read(path: Path) -> str:
    return path.read_text()


def test_request_facts_are_owned_read_only_and_active() -> None:
    header = read(INCLUDE / "PostCVSplitRequestExtraction.h")
    source = read(LIB / "PostCVSplitRequestExtraction.cpp")
    passed = read(LIB / "CVSplitScheduling.cpp")
    for token in (
        "PostCVSplitOwnedVectorRegionRequest",
        "PostCVSplitRequestSet",
        "CVSplitCubeRequest",
        "CVSplitTransferRequest",
        "CVSplitSynchronizationRequest",
        "PostCVSplitCandidateSummary",
        "extractPostCVSplitRequests",
    ):
        assert token in header
    assert "extractPostCVSplitRequests(" in passed
    assert passed.index("extractPostCVSplitRequests(") < passed.index(
        "buildPostCVSplitSchedulePlan("
    ) < passed.index("insertCrossScopeTransfers(")
    lowered = (header + source).lower()
    for forbidden in (
        "_attn_fwd",
        "flash_attention",
        "native_0316",
        "replacealluseswith",
        "builder.create",
        "estimatecube(",
        "estimateschedule(",
    ):
        assert forbidden not in lowered


def test_diagnostic_cost_model_is_absent() -> None:
    removed = (
        "CVSplitCostModel.h",
        "CVSplitCostModel.cpp",
        "CVSplitExperimentalPrimitiveCosts.cpp",
        "CVSplitScheduleEstimator.cpp",
        "CostModelCandidateGraph.cpp",
        "CostModelCandidateRanking.cpp",
        "CostModelDiagnostics.cpp",
        "CVSplitCalibrationA5Experimental.inc",
        "CVSplitRankingA5Experimental.inc",
    )
    for name in removed:
        assert not (INCLUDE / name).exists()
        assert not (LIB / name).exists()
    cmake = read(LIB / "CMakeLists.txt")
    passed = read(LIB / "CVSplitScheduling.cpp")
    assert "PostCVSplitRequestExtraction.cpp" in cmake
    assert "enableCostModelDiagnostics" not in passed
    assert "rankCostModelCandidates" not in passed
    assert "logCandidateScheduleEstimates" not in passed


if __name__ == "__main__":
    test_request_facts_are_owned_read_only_and_active()
    test_diagnostic_cost_model_is_absent()
    print("Post-CVSplit request extraction source contract: PASS")
