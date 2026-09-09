"""Static contracts for the task_0005 unchanged-DSL grid observation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_0005_harness.grid_specialization import (  # noqa: E402
    GRID_SPECIALIZATION_RULE_BITS, GRID_SPECIALIZATION_RULE_MASK, LEGACY_DEFAULT_GRAPH_RULE_MASK,
    assert_grid_specialization_enabled, assert_grid_specialization_off, expected_enabled_assertion_contract,
    expected_grid_coverage, grid_rule_mode, original_dsl_contract,
)

pytestmark = pytest.mark.backend("none")


def _event(target, grid, *, enabled=False):
    return {
        "jit_function": target,
        "original_grid": list(grid),
        "grid_evaluation": {
            "cache_lookup_order":
            ("grid_evaluation_before_cache_lookup" if enabled else "cache_lookup_before_grid_evaluation"),
        },
        "specialization": {
            "binder_specialization": {"sha256": f"specialization-{target}"},
            "legacy_jit_cache_key": {"sha256": f"cache-{target}-{grid}" if enabled else f"cache-{target}"},
            "grid_in_specialization": enabled,
        },
        "original_extent_attr": {
            "present": enabled,
            "canonical_grid": list(grid) if enabled else None,
        },
        "transform_metadata": {"present": False},
        "actual_launch_grid": list(grid),
        "actual_launch_grid_observation": "generated_launcher_intercept",
    }


def _report(*, enabled=False):
    events = [
        _event(target, grid, enabled=enabled) for target, grids in expected_grid_coverage().items() for grid in grids
    ]
    mode = grid_rule_mode(512 if enabled else LEGACY_DEFAULT_GRAPH_RULE_MASK)
    return {
        "mode": mode,
        "events": events,
        "correctness": [{"passed": True}],
    }


def test_original_kernel_ori_source_and_wrapper_signatures_are_unchanged():
    contract = original_dsl_contract()
    assert contract["path"].endswith("example1/kernel_ori.py")
    assert set(contract["targets"]) == set(expected_grid_coverage())
    for target in contract["targets"].values():
        assert not {"num_heads", "tokens", "groups"} & set(target["jit_signature"])


def test_grid_observation_coverage_and_rule_masks_are_frozen():
    assert GRID_SPECIALIZATION_RULE_BITS == {
        "IndependentAxisTensorizeRule": 512,
        "StaticProgramAxisFusionRule": 1024,
        "PersistentTaskStripMiningRule": 2048,
    }
    assert GRID_SPECIALIZATION_RULE_MASK == 3584
    assert LEGACY_DEFAULT_GRAPH_RULE_MASK & GRID_SPECIALIZATION_RULE_MASK == 0
    assert expected_grid_coverage() == {
        "_merge_split_states_kernel": [[8, 64, 1], [8, 65, 1]],
        "_indexer_norm_rope_kernel": [
            [1, 16, 1],
            [1, 1, 1],
            [16, 16, 1],
            [16, 1, 1],
            [512, 16, 1],
            [512, 1, 1],
        ],
        "_indexer_logits_kernel": [[1, 1, 1], [1, 1, 2], [1, 1, 4]],
    }


def test_default_off_assertion_rejects_extent_or_grid_transforms():
    report = _report()
    assert_grid_specialization_off(report)

    report["events"][0]["original_extent_attr"] = {"present": True}
    with pytest.raises(AssertionError, match="extent attr"):
        assert_grid_specialization_off(report)


def test_enabled_assertion_contract_requires_pre_cache_extent_and_cache_isolation():
    contract = expected_enabled_assertion_contract()
    assert set(contract["required_event_fields"]) == {
        "grid_evaluation",
        "specialization",
        "original_extent_attr",
        "transform_metadata",
        "actual_launch_grid",
    }
    report = _report(enabled=True)
    # The prime pass creates a launcher before the harness can wrap it.  A
    # matching observe event is therefore sufficient to prove the generated
    # launcher received this grid.
    prime_event = _event("_merge_split_states_kernel", [8, 64, 1], enabled=True)
    prime_event["actual_launch_grid"] = None
    prime_event["actual_launch_grid_observation"] = "not_intercepted"
    report["events"].append(prime_event)
    assert_grid_specialization_enabled(report)

    # A different raw grid must not reuse the first grid's specialization.
    report["events"][1]["specialization"]["legacy_jit_cache_key"] = (
        report["events"][0]["specialization"]["legacy_jit_cache_key"])
    with pytest.raises(AssertionError, match="distinct grids share"):
        assert_grid_specialization_enabled(report)
