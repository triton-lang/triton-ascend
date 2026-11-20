"""Keep legacy route completions exclusive with fitted scan completion."""
import json

import pytest

from test_prefixscan_profile_migration import evaluate, local_profile, scan_costs, scan_module

CAPABILITIES = {'physical_vector_core_count_hint': 56, 'physical_aicore_count_hint': 28}


@pytest.mark.parametrize('stride,legacy_completion', [(7, 0), (9, 150)])
def test_non_unit_simd_scan_keeps_one_completion_owner(tmp_path, stride, legacy_completion):
    profile, _, _ = local_profile(tmp_path)
    shape = (3, 5, stride)
    module = scan_module(shape, 1).replace('arith.addf', 'arith.minimumf')
    report = evaluate(tmp_path, shape, axis=1, profile=profile, module_text=module, capabilities=CAPABILITIES)
    route = report['stage_model']['routes']['all_simd']
    assert route.get('non_unit_simd_scan_kernel_completion_system_cycles', 0) == legacy_completion
    if stride == 7:
        assert scan_costs(report, 'simd') == pytest.approx(200.857818)


@pytest.mark.parametrize('fitted_modes', [(), ('simd', ), ('simt', ), ('simd', 'simt')])
@pytest.mark.parametrize('programs', [1, 2])
def test_minmax_completion_ownership_is_per_implementation(tmp_path, fitted_modes, programs):
    profile_path, asset_path, asset = local_profile(tmp_path)
    for mode in ('simd', 'simt'):
        asset[mode]['formulas'] = []
        if mode in fitted_modes:
            asset[mode]['formulas'].append({
                'id':
                'test_' + mode,
                'projection':
                'legacy',
                'domains': [{
                    'operation': 'cummin',
                    'data_type': 'i32',
                    'rank': 3,
                    'axis': 0,
                    'branch': 'chain' if mode == 'simd' else 'multi',
                    'warps': 0 if mode == 'simd' else 4,
                }],
                'terms': {'one': 1000 if mode == 'simd' else 1200},
            })
    asset_path.write_text(json.dumps(asset))
    shape = (16, 8, 32)
    module = scan_module(shape, 0).replace('arith.addf', 'arith.minsi').replace('f32', 'i32')
    report = evaluate(tmp_path, shape, profile=profile_path, module_text=module, capabilities=CAPABILITIES,
                      programs=programs)
    for mode, route, flag, legacy, cycles in (
        ('simd', 'all_simd', 'simd_large_sequence_minmax_scan_completion_applied',
         'non_unit_simd_scan_kernel_completion_system_cycles', 150),
        ('simt', 'all_simt_only', 'simt_multi_sequence_minmax_scan_completion_applied',
         'straight_line_simt_scan_kernel_completion_system_cycles', 370),
    ):
        plan = report['stage_model']['routes'][route]
        assert plan[flag] == (mode not in fitted_modes and programs == 1)
        assert plan.get(legacy, 0) == (0 if mode in fitted_modes else cycles)
        scans = [
            impl['resource_system_cycles']
            for stage in report['stage_model']['logical_stages']
            for impl in stage['implementations']
            if stage['workload'].get('prefix_scan_instances_per_iteration') and impl['implementation']['mode'] == mode
        ]
        assert len(scans) == 1
        assert bool(scans[0]['scan_completion_per_iteration']) == (mode in fitted_modes)
