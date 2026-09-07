from pathlib import Path
import re

TEST_DIR = Path(__file__).resolve().parent
LIB = TEST_DIR.parent
ASCEND_ROOT = TEST_DIR.parents[2]
HEADER = ASCEND_ROOT / "include" / "CVSplitScheduling" / "CrossCoreScheduleCandidate.h"
SOURCE = LIB / "CrossCoreScheduleCandidate.cpp"
PASS = LIB / "CVSplitScheduling.cpp"
CMAKE = LIB / "CMakeLists.txt"


def test_candidate_has_neutral_immutable_records() -> None:
    header = HEADER.read_text()
    for token in (
            "ScheduleMatrixLineageLimit",
            "CrossCoreScheduleCandidate",
            "CrossCoreScheduleCandidateSet",
            "logicalLaneCount",
            "waveWidth",
            "maximumLiveMatrixResultsPerLineage",
            "prefetchLimit",
            "inFlightLimit",
            "transferSlotCount",
            "diagnosticOnly",
            "selectionEligible",
    ):
        assert token in header


def test_candidate_family_is_lane_and_lineage_generic() -> None:
    text = HEADER.read_text() + SOURCE.read_text()
    assert "CrossCoreDirection::CubeToVector" in text
    assert "matrixLineages.size() + 1" in text
    assert "phaseOrdinal < widenedPrefix" in text
    assert "std::min(2u, pipelinePlan.laneCount)" in text
    forbidden = (
        r"native_0316",
        r"_attn_fwd",
        r"HEAD_DIM",
        r"head.?dim",
        r"HD64|HD128",
        r"lane\s*==\s*3",
        r"unrollFactor\s*==\s*4",
        r"flash.?attention",
    )
    assert not [pattern for pattern in forbidden if re.search(pattern, text, re.I)]


def test_schedule_candidates_are_diagnostic_only() -> None:
    source = SOURCE.read_text()
    for mutation in (
            "moveBefore",
            "moveAfter",
            "replaceAllUsesWith",
            "create<",
            ".erase(",
            "setAttr",
    ):
        assert mutation not in source
    assert "candidate.selectionEligible = true" not in source
    assert "bool selectionEligible = false" in HEADER.read_text()


def test_candidate_builder_precedes_unchanged_scheduler() -> None:
    text = PASS.read_text()
    resource = text.index("buildCrossCoreResourcePlan")
    candidate = text.index("buildCrossCoreScheduleCandidates")
    scheduler = text.index("cv_split::DependencyScheduler scheduler")
    assert resource < candidate < scheduler
    scheduler_call = text[text.index("scheduler.run", scheduler):text.index("return failure();", scheduler)]
    assert "scheduleCandidates" not in scheduler_call
    assert "resourcePlan" not in scheduler_call


def test_candidate_source_is_built() -> None:
    assert "CrossCoreScheduleCandidate.cpp" in CMAKE.read_text()


if __name__ == "__main__":
    test_candidate_has_neutral_immutable_records()
    test_candidate_family_is_lane_and_lineage_generic()
    test_schedule_candidates_are_diagnostic_only()
    test_candidate_builder_precedes_unchanged_scheduler()
    test_candidate_source_is_built()
    print("Schedule-candidate source contract: PASS")
