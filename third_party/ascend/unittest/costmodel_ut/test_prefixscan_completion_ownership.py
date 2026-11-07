"""Keep legacy route completions exclusive with fitted scan completion."""
import json

import pytest

from test_prefixscan_profile_migration import local_profile, scan_module


@pytest.mark.parametrize('fitted_modes', [(), ('simd',), ('simt',), ('simd', 'simt')])
def test_minmax_completion_ownership_is_per_implementation(tmp_path, fitted_modes):
    profile_path, asset_path, asset = local_profile(tmp_path)
    for mode in ('simd', 'simt'):
        asset[mode]['formulas'] = []
        if mode in fitted_modes:
            asset[mode]['formulas'].append({
                'id': 'test_' + mode,
                'projection': 'legacy',
                'domains': [{
                    'operation': 'cummin', 'data_type': 'i32',
                    'rank': 3, 'axis': 0,
                    'branch': 'chain' if mode == 'simd' else 'multi',
                    'warps': 0 if mode == 'simd' else 4,
                }],
                'terms': {'one': 1000 if mode == 'simd' else 1200},
            })
    asset_path.write_text(json.dumps(asset))
    module = scan_module((16, 8, 32), 0).replace('arith.addf', 'arith.minsi').replace('f32', 'i32')
    from triton._C.libtriton import ir, ascend
    context = ir.context()
    ir.load_dialects(context)
    ascend.load_dialects(context)
    module_path = tmp_path / 'scan.mlir'
    module_path.write_text(module)
    parsed = ir.parse_mlir_module(str(module_path), context)
    report_path = tmp_path / 'report.jsonl'
    capabilities = json.dumps({'physical_vector_core_count_hint': 56,
                               'physical_aicore_count_hint': 28})
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_select_simd_simt_costmodel(
        pm, 'report', str(profile_path), 'Ascend950PR_9579', 4, True,
        False, False, 1, 1, 1, '', capabilities, str(report_path))
    pm.run(parsed, 'scan_completion_ownership')
    report = json.loads(report_path.read_text().splitlines()[-1])
    for mode, route, flag in (
        ('simd', 'all_simd', 'simd_large_sequence_minmax_scan_completion_applied'),
        ('simt', 'all_simt_only', 'simt_multi_sequence_minmax_scan_completion_applied'),
    ):
        assert report['stage_model']['routes'][route][flag] == (mode not in fitted_modes)
        scans = [impl['resource_system_cycles']
                 for stage in report['stage_model']['logical_stages']
                 for impl in stage['implementations']
                 if stage['workload'].get('prefix_scan_instances_per_iteration')
                 and impl['implementation']['mode'] == mode]
        assert len(scans) == 1
        assert bool(scans[0]['scan_completion_per_iteration']) == (mode in fitted_modes)
