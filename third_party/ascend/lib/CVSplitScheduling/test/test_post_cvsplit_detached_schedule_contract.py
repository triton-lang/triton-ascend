#!/usr/bin/env python3
"""Source and reference-policy contracts for Detached-schedule construction."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"
PYBIND = ROOT / "third_party/ascend/triton_ascend.cc"


def read(path: Path) -> str:
    return path.read_text()


def reference_cube_stream(lanes: int, score_depth: int,
                          product_depth: int) -> list[tuple[str, int]]:
    commands: list[tuple[str, int]] = []
    for lane in range(min(lanes, score_depth)):
        commands.append(("score-matmul", lane))
    for lane in range(lanes):
        commands.extend((("score-release-wait", lane),
                         ("score-publish", lane)))
        refill = lane + score_depth
        if refill < lanes:
            commands.append(("score-matmul", refill))
    for lane in range(lanes):
        commands.extend((("probability-wait", lane),
                         ("product-matmul", lane)))
        if lane + 1 >= product_depth:
            publish = lane + 1 - product_depth
            commands.extend((("product-release-wait", publish),
                             ("product-publish", publish)))
    first_deferred = lanes - min(lanes, product_depth - 1)
    for lane in range(first_deferred, lanes):
        commands.extend((("product-release-wait", lane),
                         ("product-publish", lane)))
    return commands


def maximum_live(commands: list[tuple[str, int]], compute: str,
                 publish: str) -> int:
    live = 0
    maximum = 0
    for kind, _ in commands:
        if kind == compute:
            live += 1
            maximum = max(maximum, live)
        elif kind == publish:
            live -= 1
            assert live >= 0
    assert live == 0
    return maximum


def test_materialize_mode_builds_verified_detached_schedule() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    assert "analyzePostSplitSchedule" not in cpp
    assert "buildPostCVSplitDetachedSchedule(postSplitPlan)" in cpp
    assert "logPostCVSplitDetachedSchedule(detachedSchedule)" in cpp
    materialize = cpp.index("if (materializePostSplitSchedule)")
    detached = cpp.index("buildPostCVSplitDetachedSchedule(", materialize)
    transfer = cpp.index("insertCrossScopeTransfers(", detached)
    assert materialize < detached < transfer


def test_builder_is_paired_parameterized_and_non_mutating() -> None:
    header = read(INCLUDE / "PostCVSplitDetachedSchedule.h")
    source = read(LIB / "PostCVSplitDetachedSchedule.cpp")
    cmake = read(LIB / "CMakeLists.txt")
    for token in (
            "PostCVSplitDetachedSchedule",
            "PostCVSplitDetachedCommand",
            "PostCVSplitDetachedEventUse",
            "buildPostCVSplitDetachedSchedule",
            "ScoreReleaseWait",
            "ProbabilityPublish",
            "ProductReleaseWait",
            "RowwiseSoftmax",
            "DirectNzPack",
            "AffineReduce",
    ):
        assert token in header
    for token in (
            "plan.scoreLiveDepth",
            "plan.productLiveDepth",
            "findForwardEvent",
            "findReleaseEvent",
            "expectedEventUses",
            "verifyEventContracts",
            "verifyCommandOrder",
            "verifyLiveDepths",
            "verifyGeometryAndReduction",
            "publicationEligible = false",
            "mutationPerformed = false",
            "plan.backend.enableGraphSync",
            "graph-sync=on",
    ):
        assert token in source
    assert "disableGraphSync" not in source
    assert "PostCVSplitDetachedSchedule.cpp" in cmake
    lowered = source.lower()
    for forbidden in (
            "_attn_fwd",
            "flash_attention",
            "native_0316",
            "logicalunrollfactor == 4",
            "head_dim ==",
            "operation *",
            "opbuilder",
            "builder.create",
            "replacealluseswith",
            "takebody",
            "erase()",
            "kpreserveexplicitscheduleattr",
    ):
        assert forbidden not in lowered


def test_reference_streams_are_depth_bounded() -> None:
    for lanes in (2, 4, 8):
        commands = reference_cube_stream(lanes, min(2, lanes), min(2, lanes))
        assert len(commands) == 7 * lanes
        assert maximum_live(commands, "score-matmul", "score-publish") == 2
        assert maximum_live(commands, "product-matmul", "product-publish") == 2
        for lane in range(lanes):
            assert commands.count(("score-matmul", lane)) == 1
            assert commands.count(("score-publish", lane)) == 1
            assert commands.count(("product-matmul", lane)) == 1
            assert commands.count(("product-publish", lane)) == 1
    assert len(reference_cube_stream(4, 2, 2)) == 28
    assert 8 * 4 + (4 - 1) == 35
    assert 8 * 2 + (2 - 1) == 17


def test_integration_precedes_live_transfer_mutation() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    planBuildPosition = cpp.index("buildPostCVSplitSchedulePlan(")
    detachedBuildPosition = cpp.index("buildPostCVSplitDetachedSchedule(")
    transfer = cpp.index("insertCrossScopeTransfers(", detachedBuildPosition)
    assert planBuildPosition < detachedBuildPosition < transfer
    assert "if (materializePostSplitSchedule)" in cpp
    assert "logPostCVSplitDetachedSchedule(detachedSchedule)" in cpp
    assert "mutation=no" in read(LIB / "PostCVSplitDetachedSchedule.cpp")
    before_atomic = cpp.split("if (materializePostSplitSchedule) {", 1)[0]
    assert "kPreserveExplicitScheduleAttr" not in before_atomic


if __name__ == "__main__":
    test_materialize_mode_builds_verified_detached_schedule()
    test_builder_is_paired_parameterized_and_non_mutating()
    test_reference_streams_are_depth_bounded()
    test_integration_precedes_live_transfer_mutation()
    print("Detached atomic schedule source contract: PASS")
