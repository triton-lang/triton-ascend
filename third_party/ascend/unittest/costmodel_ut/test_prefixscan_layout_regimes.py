"""Validate calibrated scan costs and layout-specific domain precedence."""
import json

import pytest

from test_prefixscan_profile_migration import evaluate, local_profile, scan_costs, scan_module


@pytest.mark.parametrize('shape,axis,dtype,combine,expected', [
    ((256, 64), 0, 'f32', 'arith.mulf', 23243.141377323245),
    ((256, 16), 0, 'i32', 'arith.muli', 1889.6383382706615),
    ((2, 128, 32), 1, 'f32', 'arith.minnumf', 3218.47715048342),
    ((2, 128, 64), 1, 'i32', 'arith.maxsi', 23208.15683588404),
    ((2048, 2), 0, 'f32', 'arith.mulf', 1586.7646430801924),
])
def test_calibrated_holdout_scan_cost(tmp_path, shape, axis, dtype, combine, expected):
    module = scan_module(shape, axis).replace('f32', dtype).replace('arith.addf', combine)
    result = evaluate(tmp_path, shape, axis=axis, module_text=module)
    assert scan_costs(result, 'simt') == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize('qualified_first', [False, True])
@pytest.mark.parametrize('qualified', [False, True])
@pytest.mark.parametrize('shape,warps,full_lane', [((1024, 4), 4, True), ((256, 16), 4, False), ((1024, 4), 8, False)])
def test_full_lane_domain_precedence(tmp_path, qualified_first, qualified, shape, warps, full_lane):
    profile, asset, calibration = local_profile(tmp_path)
    domain = dict(operation='cumprod', data_type='f32', rank=2, axis=0, warps=0, branch='layout_cross_warp')
    formulas = [dict(id='general', projection='row_major', domains=[domain], terms={'one': 700.0})]
    if qualified:
        full = dict(domain, branch='layout_cross_warp_full_lane', warps=4)
        formulas.append(dict(id='full', projection='row_major', domains=[full], terms={'one': 900.0}))
    calibration['simt']['formulas'] = formulas[::-1] if qualified_first else formulas
    asset.write_text(json.dumps(calibration))
    module = scan_module(shape, 0).replace('arith.addf', 'arith.mulf')
    result = evaluate(tmp_path, shape, warps=warps, profile=profile, module_text=module)
    assert scan_costs(result, 'simt') == pytest.approx(900 if qualified and full_lane else 700)


@pytest.mark.parametrize('qualified_first', [False, True])
@pytest.mark.parametrize('shape,expected', [((8, 64, 16), 700), ((8, 128, 2), 300), ((8, 32, 512), 300)])
def test_subwarp_domain_preserves_other_layouts(tmp_path, qualified_first, shape, expected):
    profile, asset, calibration = local_profile(tmp_path)
    domain = dict(operation='cummax', data_type='f32', rank=3, axis=1, warps=4, branch='multi')
    formulas = [
        dict(id='coarse', projection='row_major', domains=[domain], terms={'one': 300.0}),
        dict(id='subwarp', projection='row_major', domains=[dict(domain, branch='layout_cross_warp_subwarp')],
             terms={'one': 700.0})
    ]
    calibration['simt']['formulas'] = formulas[::-1] if qualified_first else formulas
    asset.write_text(json.dumps(calibration))
    module = scan_module(shape, 1).replace('arith.addf', 'arith.maxnumf')
    result = evaluate(tmp_path, shape, axis=1, profile=profile, module_text=module)
    assert scan_costs(result, 'simt') == pytest.approx(expected)
