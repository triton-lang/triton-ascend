from pathlib import Path
import re

TEST_DIR = Path(__file__).resolve().parent
LIB = TEST_DIR.parent
SCHEDULER = LIB / "DependencyScheduler.cpp"
TRANSFERS = LIB / "CrossScopeTransfers.cpp"


def validator_source() -> str:
    source = SCHEDULER.read_text()
    return source[source.index("validateForcedScheduleCandidate"):source.index("// Dependency-level scheduler")]


def test_any_monotone_widened_prefix_is_structurally_supported() -> None:
    validator = validator_source()
    assert "sawDepthOne" in validator
    assert "if (sawDepthOne)" in validator
    assert "++widenedMatrixLineages" in validator
    assert "widenedMatrixLineages > 1" not in validator
    assert "widenedMatrixLineages == expectedMatrixLineages" in validator
    assert '"all-widened"' in validator
    assert '"prefix-widened"' in validator


def test_depths_remain_bounded_and_consistent() -> None:
    validator = validator_source()
    for required in (
            "limit.inFlightLimit == 0",
            "limit.inFlightLimit > 2",
            "widenedMatrixLineages == 0 ? 1 : 2",
            "candidate->maximumLiveMatrixResultsPerLineage != expectedDepth",
            "candidate->prefetchLimit != expectedDepth",
    ):
        assert required in validator


def test_existing_per_lineage_emitter_handles_all_widened() -> None:
    source = TRANSFERS.read_text()
    assert "limit.inFlightLimit - 1" in source
    assert "candidateDrainLagByOrigin.lookup(entry.first)" in source
    assert "i + drainLag" in source


def test_policy_has_no_candidate_number_kernel_shape_or_lane_rule() -> None:
    validator = validator_source()
    forbidden = (
        r"candidate(Id|->candidateId)\s*[!=]=\s*[012]",
        r"native_0316",
        r"_attn_fwd",
        r"HEAD_DIM",
        r"HD64|HD128",
        r"lane\s*==\s*3",
        r"unrollFactor\s*==\s*4",
    )
    assert not [pattern for pattern in forbidden if re.search(pattern, validator, re.I)]


if __name__ == "__main__":
    test_any_monotone_widened_prefix_is_structurally_supported()
    test_depths_remain_bounded_and_consistent()
    test_existing_per_lineage_emitter_handles_all_widened()
    test_policy_has_no_candidate_number_kernel_shape_or_lane_rule()
    print("All-widened schedule source contract: PASS")
