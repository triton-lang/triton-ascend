from copy import deepcopy

import pytest

from cv_candidate_plan import coverage, merge_kind, plan_candidates
from cv_binary_selector import choose_factor
from test_cv_binary_selector import feature, trace
from run_cv_spec_experiment import check_merge_metadata


def test_single_spatial_tiles_keep_batch_axis_candidates():
    plan = plan_candidates([8, 1, 1])
    assert {key
            for key, value in plan.items()
            if value["status"] == "NEEDS_COMPILATION"} == {"unmerged", "a0_f2", "a0_f4"}
    assert all(plan[key]["status"] == "NO_MERGE_OPPORTUNITY" for key in ("a1_f2", "a1_f4", "a2_f2", "a2_f4"))


def test_large_grid_uses_exactly_the_same_rule_on_every_axis():
    assert all(v["status"] == "NEEDS_COMPILATION" for v in plan_candidates([128, 4, 4]).values())
    plan = plan_candidates([1, 8, 1])
    assert plan["a1_f2"]["status"] == "NEEDS_COMPILATION"
    assert plan["a0_f2"]["status"] == "NO_MERGE_OPPORTUNITY"
    assert plan_candidates([8, 3, 1])["a1_f2"]["status"] == "NO_MERGE_OPPORTUNITY"


def test_merge_meaning_comes_from_compiler_not_axis_number():
    metadata = {"block_batching": {"selected_axis": 2, "cv_mode": "independent_batch_v1"}}
    assert merge_kind(metadata, 2) == "independent_batch"
    metadata["block_batching"].update(cv_mode="spatial_concat_v1", cv_lowering="spatial_m_v1")
    assert merge_kind(metadata, 2) == "M"
    metadata["block_batching"]["cv_lowering"] = "unknown"
    with pytest.raises(ValueError, match="unrecognized"):
        merge_kind(metadata, 2)


def test_omitting_batch_axis_is_incomplete_coverage_not_optimal_zero():
    provided = {"unmerged": {"status": "STATIC_FEATURES_READY"}}
    result = coverage([8, 1, 1], provided)
    assert result["missing_candidates"] == ["a0_f2", "a0_f4"]
    assert not result["all_candidates_resolved"]
    provided["a0_f2"] = {"status": "STATIC_FEATURES_READY"}
    provided["a0_f4"] = {"status": "UB_REJECTED"}
    assert coverage([8, 1, 1], provided)["all_candidates_resolved"]
    provided["a0_f4"] = {"status": "RESOURCE_UNCONFIRMED"}
    assert coverage([8, 1, 1], provided)["unresolved_candidates"] == ["a0_f4"]


def test_spatial_and_independent_candidates_compete_by_one_score():
    base = feature(trace([(120, 20)] * 8))
    batch = feature(trace([(60, 40)] * 4, factor=2), 2)
    spatial = deepcopy(batch)
    spatial["work_score"] -= 20
    batch["merge_kind"], spatial["merge_kind"] = "independent_batch", "M"
    candidates = {"a0_f2": batch, "other": spatial}
    assert choose_factor(base, candidates, (2, 4))["selected_candidate"] == "other"
    spatial["resource_checked"] = False
    assert choose_factor(base, candidates, (2, 4))["selected_candidate"] == "a0_f2"


def test_masked_batch_requires_backend_memory_metadata_before_ranking():
    metadata = dict(
        enable_hivm_batch_matmul=True,
        block_batching=dict(compile_status='REWRITTEN', requested_factor=2, selected_axis=0,
                            cv_lowering='rank3_masked_batch_v1', guard=dict(require_same_fractal_bounds=True)),
        program_grid_transforms=dict(transforms=[dict(kind='ceil_div', factor=2, axis=0)]))
    with pytest.raises(ValueError, match='memory contract'):
        check_merge_metadata(metadata, 2, 0)
    metadata['block_batching']['cv_memory_lowering'] = 'full_reduction_guard_v3'
    check_merge_metadata(metadata, 2, 0)
    metadata['block_batching']['guard']['require_same_fractal_bounds'] = False
    with pytest.raises(ValueError, match='memory contract'):
        check_merge_metadata(metadata, 2, 0)


@pytest.mark.parametrize("grid", [[0, 1, 1], [1, 2], [True, 2, 1], [1.5, 2, 1]])
def test_invalid_grid_is_not_a_no_merge_proof(grid):
    with pytest.raises(ValueError):
        plan_candidates(grid)
