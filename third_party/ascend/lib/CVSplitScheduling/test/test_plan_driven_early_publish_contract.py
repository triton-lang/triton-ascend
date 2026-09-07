#!/usr/bin/env python3
"""Source-contract checks for Plan-driven early publication."""

from pathlib import Path
import re


TEST_DIR = Path(__file__).resolve().parent
CVSPLIT_LIB = TEST_DIR.parent
ASCEND_ROOT = TEST_DIR.parents[2]
PASSES_TD = ASCEND_ROOT / "include" / "CVSplitScheduling" / "Passes.td"
SCHEDULER_H = (
    ASCEND_ROOT / "include" / "CVSplitScheduling" / "DependencyScheduler.h"
)
SCHEDULER_CPP = CVSPLIT_LIB / "DependencyScheduler.cpp"
PASS_CPP = CVSPLIT_LIB / "CVSplitScheduling.cpp"
TRANSFER_CPP = CVSPLIT_LIB / "CrossScopeTransfers.cpp"
PYBIND_CPP = ASCEND_ROOT / "triton_ascend.cc"
COMPILER_PY = ASCEND_ROOT / "backend" / "compiler.py"


def test_early_publish_option_is_explicit_and_default_off() -> None:
    td = PASSES_TD.read_text()
    option_start = td.index('Option<"enablePlanDrivenEarlyPublish"')
    option_end = td.index(">,", option_start)
    option = td[option_start:option_end]
    assert option
    assert '"enable-plan-driven-early-publish"' in option
    assert '"bool", /*default*/"false"' in option

    pass_cpp = PASS_CPP.read_text()
    assert (
        "this->enablePlanDrivenEarlyPublish = options.enablePlanDrivenEarlyPublish;"
        in pass_cpp
    )

    pybind = PYBIND_CPP.read_text()
    assert (
        "opts.enablePlanDrivenEarlyPublish = enablePlanDrivenEarlyPublish;"
        in pybind
    )
    assert 'py::arg("enable_plan_driven_early_publish") = false' in pybind

    compiler = COMPILER_PY.read_text()
    assert "cv_split_enable_plan_driven_early_publish: bool = False" in compiler
    assert '"cv_split_enable_plan_driven_early_publish"' in compiler


def test_policy_selects_terminal_v2c_boundary_without_fixed_lane() -> None:
    source = SCHEDULER_CPP.read_text()
    policy = source.split(
        "collectTerminalEarlyPublishBoundaries(", 1
    )[1].split("unsigned countVectorToCubeBoundaries", 1)[0]

    assert "CrossCoreDirection::VectorToCube" in policy
    assert "boundary.producerOrder > terminal->producerOrder" in policy
    assert "selected[terminal->producer] = terminal" in policy

    forbidden = [
        r"lane\s*==",
        r"unrollFactor\s*==",
        r"HEAD_DIM",
        r"_attn_fwd",
        r"flash[_ -]?attention",
    ]
    matches = [pattern for pattern in forbidden if re.search(pattern, policy)]
    assert not matches, f"kernel-specific early-publication policy: {matches}"


def test_scheduler_validates_anchor_and_preserves_fallback() -> None:
    header = SCHEDULER_H.read_text()
    source = SCHEDULER_CPP.read_text()

    assert "const CrossCorePipelinePlan *pipelinePlan" in header
    assert "bool enablePlanDrivenEarlyPublish" in header
    assert "Operation *transferAnchor = phaseEnd;" in source
    assert "anchorInPhase" in source
    assert "anchorNoLaterThanPhaseEnd" in source
    assert "transferPhaseEnds[producer] = transferAnchor;" in source


def test_transfer_commit_uses_selected_anchor_and_preserves_packing() -> None:
    source = TRANSFER_CPP.read_text()
    assert "Operation *lateAnchor = xfer.transferInsertionAnchor;" in source
    assert "lateAnchor == xfer.producer && !packingOps.empty()" in source
    assert "lateAnchor = packingOps.back();" in source


if __name__ == "__main__":
    test_early_publish_option_is_explicit_and_default_off()
    test_policy_selects_terminal_v2c_boundary_without_fixed_lane()
    test_scheduler_validates_anchor_and_preserves_fallback()
    test_transfer_commit_uses_selected_anchor_and_preserves_packing()
    print("Plan-driven early-publish source contract: PASS")
