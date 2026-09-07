from pathlib import Path
import re

TEST_DIR = Path(__file__).resolve().parent
LIB = TEST_DIR.parent
ASCEND_ROOT = TEST_DIR.parents[2]
SCHEDULER = LIB / "DependencyScheduler.cpp"
TRANSFERS_H = ASCEND_ROOT / "include" / "CVSplitScheduling" / "CrossScopeTransfers.h"
TRANSFERS = LIB / "CrossScopeTransfers.cpp"
PASS = LIB / "CVSplitScheduling.cpp"


def test_scheduler_accepts_control_and_monotone_widened_prefix() -> None:
    source = SCHEDULER.read_text()
    validator = source[source.index("validateForcedScheduleCandidate"):source.index("// Dependency-level scheduler")]
    for required in (
            "limit.inFlightLimit > 2",
            "sawDepthOne",
            "widenedMatrixLineages",
            "widenedMatrixLineages == 0 ? 1 : 2",
            '"first-widened"',
    ):
        assert required in validator
    assert not re.search(r"candidate(Id|->candidateId)\s*[!=]=\s*[012]", validator)


def test_candidate_reaches_verified_transfer_emitter() -> None:
    header = TRANSFERS_H.read_text()
    caller = PASS.read_text()
    assert "const CrossCoreScheduleCandidate *scheduleCandidate" in header
    call = caller[caller.index("cv_split::insertCrossScopeTransfers"):]
    assert "forcedScheduleCandidate" in call


def test_drain_lag_is_derived_per_lineage() -> None:
    source = TRANSFERS.read_text()
    for required in (
            "candidateDrainLagByOrigin",
            "limit.inFlightLimit - 1",
            "candidateDrainLagByOrigin.lookup(entry.first)",
            "i + drainLag",
            "forced schedule drain origin=",
    ):
        assert required in source
    assert "l0cPipelineDistance" not in source
    assert "byOrigin.size() != candidateDrainLagByOrigin.size()" in source


def test_candidate_validation_precedes_transfer_mutation() -> None:
    source = TRANSFERS.read_text()
    candidate = source.index("candidateDrainLagByOrigin")
    allocation = source.index("BufferPool bufferPool", candidate)
    assert candidate < allocation


def test_prefetch_is_not_moved_by_first_widened_schedule() -> None:
    source = TRANSFERS.read_text()
    candidate_region = source[source.index("candidateDrainLagByOrigin"):source.index("BufferPool bufferPool")]
    assert "prefetchLimit" not in candidate_region
    assert "moveBefore" not in candidate_region
    assert "moveAfter" not in candidate_region


def test_policy_has_no_kernel_shape_lane_or_candidate_id_rule() -> None:
    scheduler = SCHEDULER.read_text()
    transfers = TRANSFERS.read_text()
    text = scheduler[scheduler.index("validateForcedScheduleCandidate"):scheduler.
                     index("// Dependency-level scheduler")] + transfers[
                         transfers.index("candidateDrainLagByOrigin"):transfers.index("BufferPool bufferPool")]
    forbidden = (
        r"native_0316",
        r"_attn_fwd",
        r"HEAD_DIM",
        r"HD64|HD128",
        r"lane\s*==\s*3",
        r"unrollFactor\s*==\s*4",
        r"candidate(Id|->candidateId)\s*[!=]=\s*[012]",
    )
    assert not [pattern for pattern in forbidden if re.search(pattern, text, re.I)]


if __name__ == "__main__":
    test_scheduler_accepts_control_and_monotone_widened_prefix()
    test_candidate_reaches_verified_transfer_emitter()
    test_drain_lag_is_derived_per_lineage()
    test_candidate_validation_precedes_transfer_mutation()
    test_prefetch_is_not_moved_by_first_widened_schedule()
    test_policy_has_no_kernel_shape_lane_or_candidate_id_rule()
    print("First-widened schedule source contract: PASS")
