"""CPU-only tests of configurable ranking, dependency costs and admission."""
from copy import deepcopy

import pytest

from backend_cost_model import Unsupported
from cv_binary_selector import choose_factor, choose_zero_or_two, features
from cv_cost_model import CVSchedule, EventGraph, validated_assumptions
from test_cv_binary_selector import HARDWARE, PROOF, trace


def latency_pair(vector_scale=1):
    """Equal total work: fewer groups trade scalar overhead for longer vectors."""
    params = validated_assumptions({'vector_cycle_scale': vector_scale})
    values = []
    for factor, groups, scalar, vector in ((1, 8, 120, 20), (2, 4, 60, 40)):
        graph = EventGraph()
        schedule = CVSchedule(graph, 'aiv0', params)
        for _ in range(scalar):
            schedule.scalar_step()
        schedule.vf({'cycles': vector, 'counts': {}})
        cycles = graph.finish()['cycles']
        t = trace([(scalar, vector)] * groups, factor=factor)
        t.update(score=cycles, critical_group_cycles=cycles, global_memory_bound=1, group_cycles=[cycles] * groups)
        values.append(
            features(t, PROOF, factor, -1 if factor == 1 else 0, [8, 1, 1], 'same-input', 'same-compiler', HARDWARE,
                     'latency', f'shared-profile-vector-scale-{vector_scale}'))
    return values


def test_changing_costs_can_change_selection_without_changing_instruction_counts():
    original, merged = latency_pair(1)
    assert (original['estimated_cycles'], merged['estimated_cycles']) == (148, 108)
    assert choose_zero_or_two(original, merged, 'latency')['choice'] == 2
    other_original, other_merged = latency_pair(5)
    assert other_original['work_score'] == original['work_score']
    assert other_merged['work_score'] == merged['work_score']
    assert (other_original['estimated_cycles'], other_merged['estimated_cycles']) == (228, 268)
    assert choose_zero_or_two(other_original, other_merged, 'latency')['choice'] == 0


def test_overlap_uses_dependencies_instead_of_adding_every_duration():
    graph = EventGraph()
    transfer = graph.add('aic', 4, 'independent_transfer', 100)
    graph.add('aic', 2, 'independent_compute', 200)
    assert graph.finish()['cycles'] == 200
    graph = EventGraph()
    transfer = graph.add('aic', 4, 'input_transfer', 100)
    graph.add('aic', 2, 'dependent_compute', 200, (transfer, ))
    assert graph.finish()['cycles'] == 300


@pytest.mark.parametrize('field,value,reason', [
    ('ub_bits', 8193, 'MERGED_RESOURCE_UNAVAILABLE'),
    ('resource_checked', False, 'MERGED_RESOURCE_UNAVAILABLE'),
    ('binary_sha256', '', 'MERGED_RESOURCE_UNAVAILABLE'),
    ('useful_flops', 1, 'WORK_MISMATCH'),
    ('specification_sha256', 'other-specification', 'SPECIFICATION_MISMATCH'),
    ('ranking_context', 'other-parameters', 'COST_PROFILE_MISMATCH'),
    ('estimated_cycles', float('nan'), 'LATENCY_SCORE_UNCONFIRMED'),
    ('estimated_cycles', -1, 'LATENCY_SCORE_UNCONFIRMED'),
])
def test_latency_cannot_bypass_resource_or_identity_checks(field, value, reason):
    original, merged = latency_pair()
    merged[field] = value
    result = choose_factor(original, {'a0_f2': merged}, (2, 4), 'latency')
    assert result['choice'] == 0
    assert result['excluded_candidates']['a0_f2']['reason_code'] == reason


def test_missing_baseline_resource_never_returns_executable_selection():
    original, merged = latency_pair()
    original['resource_checked'] = False
    assert choose_factor(original, {'a0_f2': merged}, (2, 4), 'latency')['choice'] is None


def test_extra_traffic_is_priced_by_latency_not_automatically_rejected():
    original, merged = latency_pair()
    for name in ('padded_flops', 'dma_read_bytes', 'dma_write_bytes', 'l1_to_l0_bytes'):
        merged[name] = 2 * original[name]
    # Supplied duration already includes the added traffic. Resource/semantic
    # checks remain independent, while the final total duration decides.
    assert choose_zero_or_two(original, merged, 'latency')['choice'] == 2
    merged['estimated_cycles'] = 2 * original['estimated_cycles']
    assert choose_zero_or_two(original, merged, 'latency')['choice'] == 0


def test_ties_and_historical_measurements_do_not_change_latency_selection():
    original, merged = latency_pair()
    merged['estimated_cycles'] = original['estimated_cycles']
    expected = choose_factor(original, {'a0_f2': merged}, (2, 4), 'latency')
    assert expected['selected_candidate'] == 'unmerged'
    changed = deepcopy(merged)
    changed.update(median_us=0.01, kernel='perfect_candidate', size_category='large')
    assert choose_factor(original, {'a0_f2': changed}, (2, 4), 'latency') == expected


@pytest.mark.parametrize('overrides', [
    {'cube_flops_per_cyle': 1},
    {'cube_flops_per_cycle': 0},
    {'dma_bytes_per_cycle': float('inf')},
    {'cross_flag_cycles': float('nan')},
    {'vector_cycle_scale': -1},
    {'scalar_instruction_cycles': True},
    {'issue_queue_length': 1.5},
    ['parameters'],
])
def test_invalid_cost_parameters_fail_before_scoring(overrides):
    with pytest.raises(Unsupported):
        validated_assumptions(overrides)


def test_transfer_parameters_affect_event_duration():
    graph = EventGraph()
    schedule = CVSchedule(graph, 'aic', validated_assumptions({'dma_startup_cycles': 10, 'dma_bytes_per_cycle': 32}))
    schedule.dma('load', 320)
    assert graph.finish()['cycles'] == 20
    graph = EventGraph()
    schedule = CVSchedule(graph, 'aic', validated_assumptions({'dma_startup_cycles': 10, 'dma_bytes_per_cycle': 16}))
    schedule.dma('load', 320)
    assert graph.finish()['cycles'] == 30
