"""Public entry contracts and native sequential-group grid regressions."""
import pytest

from run_cv_spec_experiment import validate_shortlist
from v38_runtime_policy import preserves_grid, rejection


def loop_metadata(factor=2):
    loop = dict(id=0, bits=32, lower=0, upper=64, step=1)
    return dict(
        block_batching=dict(
            vv_lowering='sequential_iterations_v1', lane_width=factor, cv_mode='disabled', guard=dict(
                loop_schedule=dict(version=1, kind='sequential_iterations_v1', grid_policy='preserve', factor=factor,
                                   loop=loop), accesses=[dict(loops=[loop])], layout_checks=[])),
        program_grid_mapping_applied=False, row_coalescing_applied=False, ptsm_cap_authorized=False, coalesce_factor=1)


@pytest.mark.parametrize('factor', [2, 4])
def test_loop_grouping_preserves_core_occupancy(factor):
    metadata = loop_metadata(factor)
    assert preserves_grid(metadata, factor)
    assert rejection([56, 1, 1], 0, factor, 56, metadata) is None
    assert rejection([56, 1, 1], 0, factor, 56, {}) is not None


@pytest.mark.parametrize('field,value', [
    ('program_grid_mapping_applied', True),
    ('row_coalescing_applied', True),
    ('ptsm_cap_authorized', True),
    ('coalesce_factor', 2),
])
def test_inconsistent_loop_grid_contract_is_rejected(field, value):
    metadata = loop_metadata()
    metadata[field] = value
    with pytest.raises(ValueError):
        preserves_grid(metadata, 2)


def test_loop_schedule_must_bind_an_actual_access_loop():
    metadata = loop_metadata()
    metadata['block_batching']['guard']['accesses'] = []
    with pytest.raises(ValueError):
        preserves_grid(metadata, 2)


def test_two_directions_can_use_the_same_factor():
    validate_shortlist(
        dict(unmerged=dict(axis=-1, factor=1), a1_f4=dict(axis=1, factor=4), a2_f4=dict(axis=2, factor=4)), (2, 4))


@pytest.mark.parametrize('bad', [
    {},
    {'a0_f2': dict(axis=0, factor=2)},
    dict(unmerged=dict(axis=0, factor=1)),
    dict(unmerged=dict(axis=-1, factor=1), a0_f2=dict(axis=1, factor=2)),
    dict(unmerged=dict(axis=-1, factor=1), a0_f8=dict(axis=0, factor=8)),
    dict(unmerged=dict(axis=-1, factor=1), a0_f2=dict(axis=0, factor=2), a1_f2=dict(axis=1, factor=2), a2_f2=dict(
        axis=2, factor=2)),
])
def test_invalid_or_oversized_shortlist_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_shortlist(bad, (2, 4))


def test_cv_short_batch_preference_is_explicit_and_dtype_scoped():
    from cv_empirical_rule import select
    from test_cv_latency_selector import latency_pair
    baseline, merged = latency_pair(5)
    merged['merge_kind'] = 'independent_batch'
    assert select(baseline, {'a0_f2': merged}, False)['compiler_factor'] == 1
    decision = select(baseline, {'a0_f2': merged}, True)
    assert decision['compiler_factor'] == 2
    assert decision['reason_code'] == 'EMPIRICAL_SINGLE_TILE_SHORT_BATCH_X2'
    assert decision['prior_latency_choice'] == 'unmerged'


def test_cv_short_batch_cannot_bypass_resource_check():
    from cv_empirical_rule import select
    from test_cv_latency_selector import latency_pair
    baseline, merged = latency_pair(5)
    merged.update(merge_kind='independent_batch', resource_checked=False)
    assert select(baseline, {'a0_f2': merged})['compiler_factor'] == 1


def test_legacy_ub_proof_compatibility_preserves_allocation_and_bound_checks():
    from copy import deepcopy
    from run_cv_spec_experiment import ub_proof_matches
    legacy = dict(capacity_bits=2031616, address_extent_bits=98304,
                  buffers=[dict(function='aiv', offset_bytes=0, size_bits=98304,
                                end_bits=98304)], evidence='backend final HIVM constant UB addresses')
    observed = deepcopy(legacy)
    observed['conservative_upper_bound'] = False
    observed['buffers'][0]['conservative_i1'] = False
    assert ub_proof_matches(observed, legacy)
    assert 'conservative_upper_bound' not in legacy
    assert ub_proof_matches(observed, deepcopy(observed))
    for change in ('address', 'buffer', 'upper_bound', 'buffer_bound'):
        invalid = deepcopy(legacy)
        if change == 'address':
            invalid['address_extent_bits'] += 32
        elif change == 'buffer':
            invalid['buffers'][0]['offset_bytes'] += 32
        elif change == 'upper_bound':
            invalid['conservative_upper_bound'] = True
        else:
            invalid['buffers'][0]['conservative_i1'] = True
        assert not ub_proof_matches(observed, invalid)


def test_aic_only_proof_keeps_verified_zero_ub_usage():
    from run_cv_spec_experiment import ub_proof_matches
    proof = dict(capacity_bits=2031616, address_extent_bits=0, buffers=[],
                 evidence='backend final module has only explicitly AIC functions and no UB values')
    assert ub_proof_matches(proof, proof)
