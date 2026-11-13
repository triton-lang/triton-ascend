from collections import Counter
from copy import deepcopy

import pytest

from cv_binary_selector import VERSION, choose_factor, choose_zero_or_two, features

HARDWARE = {'compute_units': {'cube': {'fractal_sizes': {'fp32': {'m': 16, 'k': 8, 'n': 16}}}}}
PROOF = dict(resource_checked=True, ub_bits=1024, ub_capacity_bits=8192, binary_sha256='verified-binary')


def trace(groups, cores=28, auto=False, factor=1, repeats=None):
    """Synthetic compiled-work ledger, with explicit work per block."""
    records, total = [], Counter()
    repeats = repeats or [1] * len(groups)
    for block, (control, body) in enumerate(groups):
        count = repeats[block]
        counts = {
            'scalar_steps': control, 'mma_calls': count, f'mma_shape:(16, {8 * factor}, 16)': count,
            'llvm.hivm.vector_op': body - factor * count, 'useful_flops': 4096 * factor * count, 'padded_flops':
            4096 * factor * count, 'l1_to_l0_bytes': 128 * factor * count
        }
        total.update(counts)
        records.append(
            dict(block=block, counts=counts, entry_block_visits={'aic': {'entry': 1, 'body': control}},
                 dma_read_bytes=128 * factor * count, dma_write_bytes=64 * factor * count))
    return dict(status='STATIC_COST_ESTIMATED', launched_groups=len(groups), group_traces=records, counts=dict(total),
                logical_grid=[sum(repeats), 1, 1], mode='Cube', dma_read_bytes=128 * factor * sum(repeats),
                dma_write_bytes=64 * factor * sum(repeats), auto_blockify=auto, cube_core_count=cores)


def feature(t, factor=1, grid=None):
    return features(t, PROOF, factor, -1 if factor == 1 else 0, grid or [8, 1, 1], 'same-input-and-source',
                    'same-compiler', HARDWARE)


def pair(base_body=20, merged_body=40, cores=28):
    return (feature(trace([(120, base_body)] * 8, cores,
                          auto=False)), feature(trace([(60, merged_body)] * 4, cores, auto=False, factor=2), 2))


def test_control_heavy_pair_selects_two_from_structure():
    base, merged = pair()
    result = choose_zero_or_two(base, merged)
    assert result['choice'] == result['compiler_factor'] == 2
    assert result['instruction_proxy_margin'] == 40
    assert result['enabled'] and result['axis'] == 0


def test_compute_heavy_pair_can_choose_zero_with_the_same_rule():
    base, merged = pair(200, 400)
    result = choose_zero_or_two(base, merged)
    assert result['choice'] == 0 and result['compiler_factor'] == 1
    assert not result['enabled']
    assert result['reason_code'] == 'ORIGINAL_CRITICAL_CORE_WORK_NOT_HIGHER'


@pytest.mark.parametrize('field,value,reason', [
    ('dma_read_bytes', 1025, 'EXTRA_WORK_OR_TRAFFIC'),
    ('padded_flops', 32769, 'EXTRA_WORK_OR_TRAFFIC'),
    ('useful_flops', 32769, 'WORK_MISMATCH'),
    ('specification_sha256', 'different-input', 'SPECIFICATION_MISMATCH'),
    ('batch_grid', [8, 1, 1], 'GRID_MISMATCH'),
    ('ub_bits', 8193, 'MERGED_RESOURCE_UNAVAILABLE'),
    ('resource_checked', False, 'MERGED_RESOURCE_UNAVAILABLE'),
    ('analysis_available', False, 'STRUCTURE_UNCONFIRMED'),
    ('scoring_model', 'old-model', 'WORK_SCORE_UNCONFIRMED'),
    ('work_score', float('nan'), 'WORK_SCORE_UNCONFIRMED'),
])
def test_bad_candidate_never_selected_despite_apparent_gain(field, value, reason):
    base, merged = pair()
    merged[field] = value
    result = choose_zero_or_two(base, merged)
    assert result['choice'] == 0
    assert result['reason_code'] == reason


def test_unknown_baseline_does_not_misreport_safe_zero():
    base, merged = pair()
    base['resource_checked'] = False
    assert choose_zero_or_two(base, merged)['choice'] is None
    assert choose_zero_or_two(None, None)['choice'] is None


def test_small_per_group_cost_can_lose_after_all_queued_groups_are_counted():
    # 6 logical jobs on 2 cores: persistent baseline totals 110/core.
    # 3 merged jobs of 65 each require two waves; worst core totals 130.
    base_trace = trace([(90, 20)] * 2, cores=2, auto=True, repeats=[3, 3])
    base = feature(base_trace, grid=[6, 1, 1])
    merged = feature(trace([(25, 40)] * 3, cores=2, factor=2), 2, [6, 1, 1])
    assert merged['useful_flops'] == base['useful_flops'] == 6 * 4096
    assert merged['control_total_per_group'] + merged['body_per_group'] == 65
    assert base['work_score'] == 110
    assert merged['work_score'] == 130
    assert choose_zero_or_two(base, merged)['choice'] == 0


def test_uneven_loop_groups_use_busiest_core_not_average():
    result = feature(trace([(10, 10), (190, 10)], cores=2, auto=True))
    assert not result['uniform_execution_groups']
    assert result['work_score'] == 200  # Average would incorrectly be 110.
    assert result['critical_core'] == 1
    assert [c['total'] for c in result['core_workloads']] == [20, 200]


def test_queued_uneven_groups_accumulate_in_block_order():
    result = feature(trace([(90, 10), (50, 10), (40, 10)], cores=2))
    assert result['work_score'] == 110
    assert [c['groups'] for c in result['core_workloads']] == [[0], [1, 2]]


def test_one_rule_changes_choice_with_core_count_not_shape_label():
    assert choose_zero_or_two(*pair(20, 90, cores=28))['choice'] == 0
    assert choose_zero_or_two(*pair(20, 90, cores=2))['choice'] == 2


def test_equal_score_prefers_original():
    base, merged = pair(20, 80)
    assert base['work_score'] == merged['work_score']
    assert choose_zero_or_two(base, merged)['choice'] == 0


def test_names_size_categories_and_timing_fields_do_not_affect_selection():
    base, merged = pair()
    expected = choose_zero_or_two(base, merged)
    base.update(kernel='unknown_A', median_us=0.01, size_category='large')
    merged.update(kernel='unknown_B', median_us=100000, size_category='small')
    assert choose_zero_or_two(base, merged) == expected


def test_missing_or_inconsistent_group_counters_are_not_averaged_away():
    original = trace([(10, 10), (20, 10)], cores=2, auto=True)
    missing = deepcopy(original)
    del missing['group_traces'][1]['counts']
    with pytest.raises(ValueError, match='per-group counters missing'):
        feature(missing)
    changed = deepcopy(original)
    changed['group_traces'][1]['counts']['scalar_steps'] += 1
    with pytest.raises(ValueError, match='does not sum'):
        feature(changed)


def test_feature_accounting_uses_target_fractals_and_all_vector_intrinsics():
    value = feature(trace([(20, 12)] * 2, factor=2), 2, [4, 1, 1])
    assert value['control_total_per_group'] == 20
    assert value['cube_microtiles_per_group'] == 2
    assert value['vector_intrinsics_per_group'] == 10
    assert value['body_per_group'] == 12
    assert value['scoring_model'] == VERSION


@pytest.mark.parametrize('status', ['REWRITTEN', 'COMPILED_GUARDED'])
def test_frontend_and_finalized_metadata_describe_the_same_merge(status):
    from run_cv_spec_experiment import check_merge_metadata
    metadata = {
        'block_batching': {'compile_status': status, 'requested_factor': 2, 'selected_axis': 0},
        'program_grid_transforms': {'transforms': [{'kind': 'ceil_div', 'axis': 0, 'factor': 2}]}
    }
    check_merge_metadata(metadata, 2, 0)
    with pytest.raises(ValueError, match='factor/axis mismatch'):
        check_merge_metadata(metadata, 2, 1)
    metadata['program_grid_transforms']['transforms'][0]['factor'] = 4
    with pytest.raises(ValueError, match='grid transform mismatch'):
        check_merge_metadata(metadata, 2, 0)
    metadata['block_batching']['compile_status'] = 'REJECTED'
    with pytest.raises(ValueError, match='factor/axis mismatch'):
        check_merge_metadata(metadata, 2, 0)


def multi_pair(work=None):
    # Twelve programs allow 2/4/6 full groups; all candidates perform the
    # same useful matrix work. The proof is a synthetic unit-test fixture.
    work = work or {2: (60, 40), 4: (50, 40), 6: (40, 30)}
    base = feature(trace([(120, 20)] * 12), grid=[12, 1, 1])
    candidates = {
        f'a0_f{factor}': feature(trace([cost] * (12 // factor), factor=factor), factor, [12, 1, 1])
        for factor, cost in work.items()
    }
    return base, candidates


@pytest.mark.parametrize('expected,work', [
    (6, {2: (60, 40), 4: (50, 40), 6: (40, 30)}),
    (4, {2: (60, 40), 4: (50, 40), 6: (100, 30)}),
    (2, {2: (60, 40), 4: (80, 40), 6: (100, 30)}),
    (0, {2: (100, 40), 4: (110, 40), 6: (120, 30)}),
])
def test_one_scoring_rule_can_choose_each_requested_factor(expected, work):
    original, candidates = multi_pair(work)
    result = choose_factor(original, candidates)
    assert result['choice'] == expected
    assert result['requested_choices'] == [0, 2, 4, 6]
    assert result['rankings'][0]['factor'] == expected
    assert not result['excluded_candidates']


def test_unsafe_six_cannot_win_even_with_lowest_model_score():
    original, candidates = multi_pair()
    candidates['a0_f6']['ub_bits'] = 8193
    result = choose_factor(original, candidates)
    assert result['choice'] == 4
    assert result['excluded_candidates']['a0_f6']['reason_code'] == 'MERGED_RESOURCE_UNAVAILABLE'
    assert all(r['factor'] != 6 for r in result['rankings'])


def test_compile_rejection_is_preserved_without_manufacturing_a_score():
    original, candidates = multi_pair()
    candidates['a0_f6'] = dict(factor=6, axis=0, resource_checked=False, status='FRONTEND_REJECTED',
                               error='CV_FACTOR_REQUIRES_POWER_OF_TWO')
    result = choose_factor(original, candidates)
    assert result['choice'] == 4
    assert result['excluded_candidates']['a0_f6']['compiler_or_analysis_error'] == 'CV_FACTOR_REQUIRES_POWER_OF_TWO'
    assert len(result['rankings']) == 3


def test_tied_scores_choose_lower_factor_independent_of_input_order():
    original, candidates = multi_pair({2: (60, 40), 4: (50, 50), 6: (40, 60)})
    assert choose_factor(original, dict(reversed(list(candidates.items()))))['choice'] == 2
    original['work_score'] = 100
    assert choose_factor(original, candidates)['choice'] == 0


def test_grid_divisibility_uses_each_requested_factor():
    original, candidates = multi_pair()
    original['original_grid'] = [8, 1, 1]
    for c in candidates.values():
        c['original_grid'] = [8, 1, 1]
        c['batch_grid'] = [8 // c['factor'], 1, 1]
    result = choose_factor(original, candidates)
    assert result['choice'] == 4
    assert result['excluded_candidates']['a0_f6']['reason_code'] == 'TAIL_OR_NO_MERGE_OPPORTUNITY'


def test_request_subset_does_not_select_other_factors():
    original, candidates = multi_pair()
    assert choose_factor(original, candidates, factors=(2, ))['choice'] == 2
    with pytest.raises(ValueError, match='subset of 2/4/6'):
        choose_factor(original, candidates, factors=(2, 8))


def test_multi_selector_still_requires_verified_baseline():
    original, candidates = multi_pair()
    original['resource_checked'] = False
    result = choose_factor(original, candidates)
    assert result['choice'] is None and not result['rankings']
