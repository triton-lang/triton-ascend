#!/usr/bin/env python3
"""Source and reference-policy contracts for Post-CVSplit schedule planning."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"
PYBIND = ROOT / "third_party/ascend/triton_ascend.cc"


def read(path: Path) -> str:
    return path.read_text()


def reference_plan(lanes: int, score_width: int, score_bytes: int,
                   probability_bytes: int, product_bytes: int) -> dict:
    depth = min(2, lanes)
    return {
        "score_depth": depth,
        "product_depth": depth,
        "probability_slots": lanes,
        "chunks": score_width // 64,
        "ub_bytes": depth * (score_bytes + product_bytes),
        "l1_bytes": lanes * probability_bytes,
        "events": 3 * lanes + 2 * depth,
        "reductions": lanes - 1,
    }


def test_schedule_mode_is_default_disabled_and_fully_propagated() -> None:
    passes = read(INCLUDE / "Passes.td")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    backend = read(BACKEND)
    pybind = read(PYBIND)
    assert 'Option<"postSplitScheduleMode"' in passes
    option = passes.split('Option<"postSplitScheduleMode"', 1)[1]
    option = option.split('>,', 1)[0]
    assert '"std::string"' in option
    assert r'\"disabled\"' in option
    assert "options.postSplitScheduleMode" in cpp
    assert "opts.postSplitScheduleMode" in pybind
    assert 'py::arg("post_split_schedule_mode") = "disabled"' in pybind
    assert 'cv_split_post_split_schedule_mode: str = "disabled"' in backend
    assert '"cv_split_post_split_schedule_mode"' in backend
    assert 'post_split_schedule_mode=metadata[' in backend
    for legacy in (
            "enablePostSplitPlanDiagnostics",
            "enableDetachedScheduleDiagnostics",
            "enableScheduleBindingDiagnostics",
            "enableScheduleMaterialization",
    ):
        assert legacy not in passes + cpp + pybind


def test_schedule_mode_values_and_invalid_input_are_explicit() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    for value in ("disabled", "materialize"):
        assert f'value == "{value}"' in cpp
    assert 'value == "analyze"' not in cpp
    assert "parsePostSplitScheduleMode(postSplitScheduleMode)" in cpp
    assert "invalid post-split-schedule-mode" in cpp
    assert "signalPassFailure()" in cpp


def test_plan_is_parameterized_verified_and_analysis_only() -> None:
    header = read(INCLUDE / "PostCVSplitSchedulePlan.h")
    source = read(LIB / "PostCVSplitSchedulePlan.cpp")
    for token in (
            "AttentionRecurrenceDescriptor",
            "PostCVSplitSchedulePlan",
            "PostCVSplitSlotAssignment",
            "PostCVSplitEventPlan",
            "PostCVSplitVectorLanePlan",
            "PostCVSplitReductionStep",
            "PostCVSplitBackendRequirements",
            "enableGraphSync",
            "buildPostCVSplitSchedulePlan",
    ):
        assert token in header
    for token in (
            "candidate.logicalLaneCount",
            "expectedKinds",
            "operationCounts",
            "kVectorChunkElements",
            "lane % plan.scoreLiveDepth",
            "lane % plan.productLiveDepth",
            "checkedAdd",
            "checkedMul",
            "verifyPlan",
            "activeValues.size() > 1",
            "pathStartsInUb",
            "cubeLineages.size() != 2",
            "PostCVSplitSchedulePlanStatus::FlagOverflow",
            "PostCVSplitSchedulePlanStatus::MemoryBudgetExceeded",
            "graph-sync=on",
    ):
        assert token in source
    recurrence_gate = source.split("if (!reductionGeometryFound", 1)[1]
    recurrence_gate = recurrence_gate.split(") {", 1)[0]
    assert "!plan.recurrence.hasPermute" not in recurrence_gate
    assert "probabilityTransfer.destinationLayout != CVSplitLayout::NZ" in source
    assert "for (const CVSplitCubeRequest &request : requests.cubeRequests)" in source
    assert "lineage * lanes" not in source
    lowered = source.lower()
    for forbidden in (
            "_attn_fwd",
            "flash_attention",
            "native_0316",
            "head_dim ==",
            "logicalunrollfactor == 4",
            "candidateid ==",
            "setattr(",
            "replacealluseswith",
            "builder.create",
            "erase()",
    ):
        assert forbidden not in lowered
    assert "disableGraphSync" not in header + source


def test_reference_parameterization_and_golden_fixture() -> None:
    u2 = reference_plan(2, 128, 32768, 32768, 32768)
    u4_hd64 = reference_plan(4, 128, 32768, 32768, 16384)
    u4_hd128 = reference_plan(4, 128, 32768, 32768, 32768)
    u8 = reference_plan(8, 128, 32768, 32768, 32768)
    assert u2["events"] == 10
    assert u4_hd64["ub_bytes"] == 98304
    assert u4_hd128 == {
        "score_depth": 2,
        "product_depth": 2,
        "probability_slots": 4,
        "chunks": 2,
        "ub_bytes": 131072,
        "l1_bytes": 131072,
        "events": 16,
        "reductions": 3,
    }
    assert u8["events"] == 28
    assert u4_hd128["events"] <= 16 < u8["events"]


def test_release_event_resources_follow_the_last_real_consumer() -> None:
    source = read(LIB / "PostCVSplitSchedulePlan.cpp")
    releases = source.split(
        "for (unsigned slot = 0; slot < plan.scoreLiveDepth; ++slot)", 1)[1]
    score, product = releases.split(
        "for (unsigned slot = 0; slot < plan.productLiveDepth; ++slot)", 1)
    product = product.split("plan.forwardEventCount", 1)[0]
    assert "PrincipalResource::Mte3" in score
    assert "PrincipalResource::Fixpipe" in score
    assert "PrincipalResource::Vector" not in score
    assert "PrincipalResource::Vector" in product
    assert "PrincipalResource::Fixpipe" in product


def test_integration_is_after_materialization_before_transfer_mutation() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    extract = cpp.index("extractPostCVSplitRequests(")
    planPosition = cpp.index("buildPostCVSplitSchedulePlan(")
    transfer = cpp.index("insertCrossScopeTransfers(", planPosition)
    assert extract < planPosition < transfer
    assert "if (materializePostSplitSchedule)" in cpp
    assert "mutation=no" in read(LIB / "PostCVSplitSchedulePlan.cpp")
    before_atomic = cpp.split("if (materializePostSplitSchedule) {", 1)[0]
    assert "kPreserveExplicitScheduleAttr" not in before_atomic
    assert "PostCVSplitSchedulePlan.cpp" in read(LIB / "CMakeLists.txt")


if __name__ == "__main__":
    test_schedule_mode_is_default_disabled_and_fully_propagated()
    test_schedule_mode_values_and_invalid_input_are_explicit()
    test_plan_is_parameterized_verified_and_analysis_only()
    test_reference_parameterization_and_golden_fixture()
    test_release_event_resources_follow_the_last_real_consumer()
    test_integration_is_after_materialization_before_transfer_mutation()
    print("Post-CVSplit schedule plan source contract: PASS")
