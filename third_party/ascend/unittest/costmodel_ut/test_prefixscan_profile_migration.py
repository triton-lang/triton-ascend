"""Exercise the compiled C++ scan model through its MLIR pass, without GoogleTest."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
PROFILE = ROOT / 'third_party/ascend/costmodel/profiles/simd_simt/ascend_950pr_simd_simt_v1.json'


def scan_module(shape, axis, copies=1):
    ty = 'tensor<' + 'x'.join(map(str, shape)) + 'xf32>'
    body = []
    for i in range(copies):
        operand = '%input' if i == 0 else f'%s{i-1}'
        body.append(f'''%s{i} = "tt.scan"({operand}) ({{
        ^bb0(%a: f32, %b: f32):
          %sum = arith.addf %a, %b : f32
          "tt.scan.return"(%sum) : (f32) -> ()
        }}) {{axis = {axis} : i32, reverse = false}} : ({ty}) -> {ty}''')
    return f'module {{ tt.func public @main(%input: {ty}) -> {ty} {{' + '\n'.join(
        body) + f'\ntt.return %s{copies-1} : {ty}\n}} }}'


def evaluate(tmp_path, shape, axis=0, copies=1, warps=4, profile=PROFILE, module_text=None):
    report = tmp_path / 'report.jsonl'
    if report.exists():
        report.unlink()
    from triton._C.libtriton import ir, ascend
    context = ir.context()
    ir.load_dialects(context)
    ascend.load_dialects(context)
    module_path = tmp_path / 'scan.mlir'
    module_path.write_text(module_text or scan_module(shape, axis, copies))
    module = ir.parse_mlir_module(str(module_path), context)
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_select_simd_simt_costmodel(pm, 'report', str(profile), 'Ascend950PR_9579', warps, True,
                                                      False, False, 1, 1, 1, '', '{}', str(report))
    pm.run(module, "prefixscan_migration")
    assert report.is_file()
    return json.loads(report.read_text().splitlines()[-1])


def scan_costs(report, mode):
    costs = []
    for stage in report['stage_model']['logical_stages']:
        if not stage['workload'].get('prefix_scan_instances_per_iteration'):
            continue
        impls = [i for i in stage['implementations'] if i['implementation']['mode'] == mode]
        assert impls
        costs.append(impls[0]['resource_system_cycles']['prefix_scan_critical_path_per_iteration'])
    assert costs
    return sum(costs)


@pytest.mark.parametrize('n,expected', [(32, 26.4 + 227.3 / 64 * 32), (64, 26.4 + 227.3),
                                        (128, 252.4 + 4.6015 / 64 * 128 * 7), (2048, 252.4 + 5.5717 / 64 * 2048 * 11)])
def test_simd_segments(tmp_path, n, expected):
    assert scan_costs(evaluate(tmp_path, (n, )), 'simd') == pytest.approx(expected)


@pytest.mark.parametrize('n,expected', [(32, 96.8), (64, 144.8 + 25.5), (128, 144.8 + 25.5 * 2),
                                        (256, 181.427 + 0.1425 * 256)])
def test_simt_segments(tmp_path, n, expected):
    assert scan_costs(evaluate(tmp_path, (n, )), 'simt') == pytest.approx(expected, abs=0.02)


def test_multiple_scans_keep_local_segment(tmp_path):
    report = evaluate(tmp_path, (32, ), copies=2)
    assert scan_costs(report, 'simt') == pytest.approx(2 * 96.8)


def test_nonleading_axis_transpose(tmp_path):
    leading = scan_costs(evaluate(tmp_path, (64, 64), axis=0), 'simd')
    transposed = scan_costs(evaluate(tmp_path, (64, 64), axis=1), 'simd')
    assert transposed - leading == pytest.approx(323 + 2 * 64 * 64 * 4 / 4.2)


def test_prefixscan_asset_changes_binary_cache_key(tmp_path):
    import ast
    import hashlib

    source = (ROOT / 'third_party/ascend/backend/compiler.py').read_text()
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_auto_simt_asset_hash')
    namespace = {'Path': Path, 'json': json, 'hashlib': hashlib}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<asset-hash>', 'exec'), namespace)
    profile = tmp_path / 'profile.json'
    asset = tmp_path / 'scan.json'
    profile.write_text(json.dumps({'prefixscan_profile': 'scan.json'}))
    asset.write_text('{"coefficient":1}')
    before = namespace['_auto_simt_asset_hash'](str(profile), '')
    asset.write_text('{"coefficient":2}')
    after = namespace['_auto_simt_asset_hash'](str(profile), '')
    assert before != after


@pytest.mark.parametrize('shape,axis,expected', [
    ((16, 64), 0, 190.4707 + 0.04753 * 1024),
    ((64, 64), 0, 412.9181 + 10.4793 * 4096 * 2 / 128),
    ((256, 64), 0, 412.9181 + 10.4793 * 16384 * 4 / 128 + 40.6469 * 64),
])
def test_simt_multidimensional_segments(tmp_path, shape, axis, expected):
    assert scan_costs(evaluate(tmp_path, shape, axis=axis), 'simt') == pytest.approx(expected)


def test_simd_multidimensional_hinge(tmp_path):
    report = evaluate(tmp_path, (512, 64))
    expected = 202.5 + (573.6 * 512 + 1347.7 * 256) / 64
    assert scan_costs(report, 'simd') == pytest.approx(expected)


def local_profile(tmp_path):
    profile = json.loads(PROFILE.read_text())
    # Keep unrelated assets valid when relocating the selection profile.
    for key in ('microbenchmark_profile', 'indirect_memory_profile', 'reduction_profile'):
        if key in profile:
            profile[key] = str((PROFILE.parent / profile[key]).resolve())
    calibration = json.loads((PROFILE.parent / profile['prefixscan_profile']).read_text())
    asset = tmp_path / 'calibration.json'
    asset.write_text(json.dumps(calibration))
    profile['prefixscan_profile'] = str(asset)
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(profile))
    return path, asset, calibration


def test_external_asset_invalidates_in_process_profile_cache(tmp_path):
    profile, asset, calibration = local_profile(tmp_path)
    first = scan_costs(evaluate(tmp_path, (32, ), profile=profile), 'simd')
    calibration['simd']['dependency_factor_1d_small'] *= 2
    asset.write_text(json.dumps(calibration))
    second = scan_costs(evaluate(tmp_path, (32, ), profile=profile), 'simd')
    assert second - first == pytest.approx(227.3 / 64 * 32)


def test_negative_scan_coefficient_is_rejected(tmp_path):
    profile, asset, calibration = local_profile(tmp_path)
    calibration['simt']['simt_1d_warp_segments']['4']['c_local'] = -1
    asset.write_text(json.dumps(calibration))
    with pytest.raises(RuntimeError):
        evaluate(tmp_path, (32, ), profile=profile)


def test_missing_external_asset_is_rejected(tmp_path):
    profile, asset, calibration = local_profile(tmp_path)
    asset.unlink()
    with pytest.raises(RuntimeError):
        evaluate(tmp_path, (32, ), profile=profile)


def test_loop_scan_uses_same_calibration(tmp_path):
    module = """module {
      tt.func public @main(%input: tensor<128xf32>) -> tensor<128xf32> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %input) -> tensor<128xf32> {
          %scan = "tt.scan"(%acc) ({
          ^bb0(%a: f32, %b: f32):
            %sum = arith.addf %a, %b : f32
            "tt.scan.return"(%sum) : (f32) -> ()
          }) {axis = 0 : i32, reverse = false} : (tensor<128xf32>) -> tensor<128xf32>
          scf.yield %scan : tensor<128xf32>
        }
        tt.return %out : tensor<128xf32>
      }
    }"""
    report = evaluate(tmp_path, (128, ), module_text=module)
    assert scan_costs(report, 'simd') == pytest.approx(316.821)
    assert scan_costs(report, 'simt') == pytest.approx(195.8)


@pytest.mark.parametrize('replacement', [('reverse = false', 'reverse = true'), ('arith.addf', 'arith.mulf'),
                                         ('f32', 'f16')])
def test_scan_calibration_ignores_direction_combine_and_dtype(tmp_path, replacement):
    profile, _, _ = local_profile(tmp_path)
    module = scan_module((128, ), 0).replace(*replacement)
    calibrated = evaluate(tmp_path, (128, ), profile=profile, module_text=module)
    assert scan_costs(calibrated, 'simd') == pytest.approx(316.821)
    assert scan_costs(calibrated, 'simt') == pytest.approx(195.8)


@pytest.mark.parametrize('dtype,bits', [('f16', 16), ('f32', 32), ('i32', 32), ('i64', 64)])
@pytest.mark.parametrize('axis', [0, 1])
def test_multidimensional_scan_uses_operand_width(tmp_path, dtype, bits, axis):
    shape = (16, 128) if axis == 0 else (128, 16)
    module = scan_module(shape, axis).replace('f32', dtype)
    if dtype.startswith('i'):
        module = module.replace('arith.addf', 'arith.addi')
    report = evaluate(tmp_path, shape, axis=axis, module_text=module)
    steps = {16: 16, 32: 32, 64: 64}[bits]
    expected = 202.5 + steps * 573.6 / 64
    if axis == 1:
        expected += 323 + 2 * 2048 * (bits / 8) / 4.2
    assert scan_costs(report, 'simd') == pytest.approx(expected)
    instances = [
        entry for stage in report['stage_model']['logical_stages']
        for entry in stage['workload']['prefix_scan_instances_per_iteration']
    ]
    assert instances and all(entry['element_bit_width'] == bits for entry in instances)


def test_mixed_scan_combines_keep_calibrated_cost(tmp_path):
    module = scan_module((32, ), 0, copies=2).replace('arith.addf', 'arith.mulf', 1)
    report = evaluate(tmp_path, (32, ), module_text=module)
    assert scan_costs(report, 'simd') == pytest.approx(2 * (26.4 + 227.3 / 2))
    assert scan_costs(report, 'simt') == pytest.approx(2 * 96.8)


def test_multi_result_scan_uses_first_operand_geometry(tmp_path):
    module = """module {
      tt.func public @main(%input: tensor<32xf32>, %other: tensor<32xf32>)
          -> (tensor<32xf32>, tensor<32xf32>) {
        %scan:2 = "tt.scan"(%input, %other) ({
        ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
          %sum0 = arith.addf %a, %c : f32
          %sum1 = arith.addf %b, %d : f32
          "tt.scan.return"(%sum0, %sum1) : (f32, f32) -> ()
        }) {axis = 0 : i32, reverse = false} :
          (tensor<32xf32>, tensor<32xf32>) -> (tensor<32xf32>, tensor<32xf32>)
        tt.return %scan#0, %scan#1 : tensor<32xf32>, tensor<32xf32>
      }
    }"""
    report = evaluate(tmp_path, (32, ), module_text=module)
    assert scan_costs(report, 'simd') == pytest.approx(26.4 + 227.3 / 2)
    assert scan_costs(report, 'simt') == pytest.approx(96.8)
