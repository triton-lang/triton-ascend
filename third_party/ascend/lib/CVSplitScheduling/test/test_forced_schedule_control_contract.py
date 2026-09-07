from pathlib import Path
import re

TEST_DIR = Path(__file__).resolve().parent
LIB = TEST_DIR.parent
ASCEND_ROOT = TEST_DIR.parents[2]
PASSES = ASCEND_ROOT / "include" / "CVSplitScheduling" / "Passes.td"
HEADER = ASCEND_ROOT / "include" / "CVSplitScheduling" / "DependencyScheduler.h"
SCHEDULER = LIB / "DependencyScheduler.cpp"
PASS = LIB / "CVSplitScheduling.cpp"
PYBIND = ASCEND_ROOT / "triton_ascend.cc"
BACKEND = ASCEND_ROOT / "backend" / "compiler.py"


def test_development_option_is_complete_and_default_off() -> None:
    passes = PASSES.read_text()
    cpp = PASS.read_text()
    pybind = PYBIND.read_text()
    backend = BACKEND.read_text()
    assert 'Option<"scheduleCandidateId", "schedule-candidate-id"' in passes
    assert '"int", /*default*/"-1"' in passes
    assert "this->scheduleCandidateId = options.scheduleCandidateId;" in cpp
    assert "opts.scheduleCandidateId = scheduleCandidateId;" in pybind
    assert 'py::arg("schedule_candidate_id") = -1' in pybind
    assert "cv_split_schedule_candidate_id: int = -1" in backend
    assert '"cv_split_schedule_candidate_id"' in backend


def test_selected_candidate_reaches_scheduler() -> None:
    header = HEADER.read_text()
    cpp = PASS.read_text()
    assert "const CrossCoreScheduleCandidate *scheduleCandidate" in header
    call = cpp[cpp.index("scheduler.run"):cpp.index("return failure();", cpp.index("scheduler.run"))]
    assert "forcedScheduleCandidate" in call
    assert "scheduleCandidateSet->candidates[requested]" in cpp
    assert "requested >= scheduleCandidateSet->candidates.size()" in cpp


def test_control_is_structural_not_candidate_numbered() -> None:
    source = SCHEDULER.read_text()
    validator = source[source.index("validateForcedScheduleCandidate"):source.index("// Dependency-level scheduler")]
    for required in (
            "candidate->logicalLaneCount != pipelinePlan->laneCount",
            "candidate->waveWidth != candidate->logicalLaneCount",
            "candidate->maximumLiveMatrixResultsPerLineage != expectedDepth",
            "candidate->prefetchLimit != expectedDepth",
            "limit.inFlightLimit == 0",
            "limit.direction != CrossCoreDirection::CubeToVector",
            "lineage.originId != limit.originId",
            "lineage.direction != limit.direction",
            '" behavior="',
            '"generic"',
    ):
        assert required in validator
    assert not re.search(r"candidate(Id|->candidateId)\s*[!=]=\s*0", validator)


def test_validation_precedes_unchanged_reorder() -> None:
    source = SCHEDULER.read_text()
    run = source.index("DependencyScheduler::run")
    validate = source.index("validateForcedScheduleCandidate", run)
    graph = source.index("buildDependencyGraph", run)
    reorder = source.index("reorderForCrossScopeProducerPhases", run)
    assert run < validate < graph < reorder
    reorder_call = source[reorder:source.index("return success();", reorder)]
    assert "scheduleCandidate" not in reorder_call


def test_unavailable_candidate_rejects_transactionally() -> None:
    cpp = PASS.read_text()
    assert "requested schedule candidate unavailable" in cpp
    assert "return failure();" in cpp


def test_policy_has_no_kernel_shape_or_lane_rule() -> None:
    text = SCHEDULER.read_text() + PASS.read_text()
    forbidden = (
        r"native_0316",
        r"_attn_fwd",
        r"HEAD_DIM",
        r"HD64|HD128",
        r"lane\s*==\s*3",
        r"unrollFactor\s*==\s*4",
    )
    assert not [pattern for pattern in forbidden if re.search(pattern, text, re.I)]


if __name__ == "__main__":
    test_development_option_is_complete_and_default_off()
    test_selected_candidate_reaches_scheduler()
    test_control_is_structural_not_candidate_numbered()
    test_validation_precedes_unchanged_reorder()
    test_unavailable_candidate_rejects_transactionally()
    test_policy_has_no_kernel_shape_or_lane_rule()
    print("Forced schedule-control source contract: PASS")
