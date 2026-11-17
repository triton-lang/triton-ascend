"""CPU checks for bitwise scan completion ownership and profile dependencies."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from test_prefixscan_profile_migration import PROFILE, ROOT, evaluate


def kernel(operation='ori', reverse=False):
    direction = 'true' if reverse else 'false'
    return f'''module {{
      tt.func public @main(%x: !tt.ptr<i32> {{tt.divisibility = 16 : i32}},
                           %y: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
        %r = tt.make_range {{start = 0 : i32, end = 128 : i32}} : tensor<128xi32>
        %px = tt.splat %x : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
        %xp = tt.addptr %px, %r : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
        %v = tt.load %xp : tensor<128x!tt.ptr<i32>>
        %s = "tt.scan"(%v) ({{
        ^bb0(%a: i32, %b: i32):
          %c = arith.{operation} %a, %b : i32
          "tt.scan.return"(%c) : (i32) -> ()
        }}) {{axis = 0 : i32, reverse = {direction}}} : (tensor<128xi32>) -> tensor<128xi32>
        %py = tt.splat %y : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
        %yp = tt.addptr %py, %r : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
        tt.store %yp, %s : tensor<128x!tt.ptr<i32>>
        tt.return
      }}
    }}'''


def scan_implementations(report):
    for stage in report['stage_model']['logical_stages']:
        if stage['workload'].get('prefix_scan_instances_per_iteration'):
            yield from stage['implementations']


@pytest.mark.parametrize('operation', ['ori', 'andi', 'xori'])
@pytest.mark.parametrize('reverse', [False, True])
def test_both_routes_use_fitted_completion(tmp_path, operation, reverse):
    report = evaluate(tmp_path, (128, ), module_text=kernel(operation, reverse))
    implementations = list(scan_implementations(report))
    assert {i['implementation']['mode'] for i in implementations} == {'simd', 'simt'}
    for implementation in implementations:
        resources = implementation['resource_system_cycles']
        assert resources['scan_completion_per_iteration'] > 0
        assert len(resources['scan_formula_instances_per_iteration']) == 1
        assert resources['prefix_scan_critical_path_per_iteration'] == 0
        assert resources['prefix_scan_issue_per_iteration'] == 0


def local_profile(tmp_path):
    profile = json.loads(PROFILE.read_text())
    for key in ('microbenchmark_profile', 'indirect_memory_profile', 'reduction_profile', 'prefixscan_profile',
                'bitwise_scan_profile'):
        if key in profile:
            profile[key] = str((PROFILE.parent / profile[key]).resolve())
    asset = tmp_path / 'bitwise.json'
    asset.write_text(Path(profile['bitwise_scan_profile']).read_text())
    profile['bitwise_scan_profile'] = str(asset)
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(profile))
    return path, asset


def test_cpp_profile_cache_tracks_bitwise_asset(tmp_path):
    profile, asset = local_profile(tmp_path)
    first = list(scan_implementations(evaluate(tmp_path, (128, ), profile=profile, module_text=kernel())))
    data = json.loads(asset.read_text())
    for formula in data['formulas']:
        formula['terms'] = {k: 1.5 * v for k, v in formula['terms'].items()}
    asset.write_text(json.dumps(data))
    second = list(scan_implementations(evaluate(tmp_path, (128, ), profile=profile, module_text=kernel())))
    for before, after in zip(first, second):
        assert after['resource_system_cycles']['scan_completion_per_iteration'] == pytest.approx(
            1.5 * before['resource_system_cycles']['scan_completion_per_iteration'])


def test_python_cache_tracks_bitwise_asset(tmp_path):
    source = (ROOT / 'third_party/ascend/backend/compiler.py').read_text()
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_auto_simt_asset_hash')
    namespace = {'Path': Path, 'json': json, 'hashlib': hashlib}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<asset-hash>', 'exec'), namespace)
    profile, asset = local_profile(tmp_path)
    before = namespace['_auto_simt_asset_hash'](str(profile), '')
    asset.write_text(asset.read_text().replace('system_cycle', 'different_unit'))
    assert namespace['_auto_simt_asset_hash'](str(profile), '') != before


def test_invalid_equation_is_rejected(tmp_path):
    profile, asset = local_profile(tmp_path)
    data = json.loads(asset.read_text())
    data['formulas'][0]['terms']['unknown_work'] = 1
    asset.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        evaluate(tmp_path, (128, ), profile=profile, module_text=kernel())


def test_reverse_scan_reuses_available_forward_tree(tmp_path):
    profile, asset = local_profile(tmp_path)
    data = json.loads(asset.read_text())
    data['dispatch'] = [root for root in data['dispatch'] if not root['reverse']]
    asset.write_text(json.dumps(data))
    report = evaluate(tmp_path, (128, ), profile=profile, module_text=kernel(reverse=True))
    for implementation in scan_implementations(report):
        resources = implementation['resource_system_cycles']
        assert resources['scan_completion_per_iteration'] > 0
        assert resources['scan_formula_instances_per_iteration']


@pytest.mark.parametrize('reverse', [False, True])
def test_shared_tree_accepts_both_directions(tmp_path, reverse):
    profile, asset = local_profile(tmp_path)
    data = json.loads(asset.read_text())
    data['dispatch'] = [root for root in data['dispatch'] if not root.pop('reverse')]
    asset.write_text(json.dumps(data))
    report = evaluate(tmp_path, (128, ), profile=profile, module_text=kernel(reverse=reverse))
    for implementation in scan_implementations(report):
        assert implementation['resource_system_cycles']['scan_formula_instances_per_iteration']


@pytest.mark.parametrize('shared_first', [False, True])
def test_explicit_direction_has_priority_over_shared_tree(tmp_path, shared_first):
    profile, asset = local_profile(tmp_path)
    baseline = evaluate(tmp_path, (128, ), profile=profile, module_text=kernel(reverse=True))
    data = json.loads(asset.read_text())
    shared = []
    for root in data['dispatch']:
        if not root['reverse']:
            item = dict(root)
            item.pop('reverse')
            item['tree'] = {'formula': 'shared_test_formula'}
            shared.append(item)
    data['formulas'].append({'id': 'shared_test_formula', 'terms': {'one': 10000, 'scan_instruction_work': 1}})
    data['dispatch'] = shared + data['dispatch'] if shared_first else data['dispatch'] + shared
    asset.write_text(json.dumps(data))
    report = evaluate(tmp_path, (128, ), profile=profile, module_text=kernel(reverse=True))
    for before, after in zip(scan_implementations(baseline), scan_implementations(report)):
        assert after['resource_system_cycles']['scan_completion_per_iteration'] == pytest.approx(
            before['resource_system_cycles']['scan_completion_per_iteration'])
        assert 'shared_test_formula' not in after['resource_system_cycles']['scan_formula_instances_per_iteration']
    data['dispatch'] = [root for root in data['dispatch'] if root.get('reverse') is not True]
    asset.write_text(json.dumps(data))
    shared_report = evaluate(tmp_path, (128, ), profile=profile, module_text=kernel(reverse=True))
    for implementation in scan_implementations(shared_report):
        assert implementation['resource_system_cycles']['scan_formula_instances_per_iteration'] == {
            'shared_test_formula': 1
        }


def test_invalid_direction_value_is_rejected(tmp_path):
    profile, asset = local_profile(tmp_path)
    data = json.loads(asset.read_text())
    data['dispatch'][0]['reverse'] = 'both'
    asset.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        evaluate(tmp_path, (128, ), profile=profile, module_text=kernel())


@pytest.mark.parametrize('length,expected', [(128, 483), (256, 967)])
def test_piecewise_formula_threshold_is_preserved(tmp_path, length, expected):
    profile, asset = local_profile(tmp_path)
    data = json.loads(asset.read_text())
    data['formulas'] = [
        {'id': 'low', 'terms': {'one': 100, 'scan_instruction_work': 1}},
        {'id': 'high', 'terms': {'one': 200, 'scan_instruction_work': 1}},
    ]
    data['dispatch'] = [{
        'mode': 'simd',
        'element_bits': 32,
        'algorithm': 'S1',
        'operation_group': 'all_bitwise',
        'tree': {'feature': 'elements', 'threshold': 128, 'left': 'low', 'right': 'high'},
    }]
    asset.write_text(json.dumps(data))
    report = evaluate(tmp_path, (length, ), profile=profile, module_text=kernel().replace('128', str(length)))
    simd = next(i for i in scan_implementations(report) if i['implementation']['mode'] == 'simd')
    assert simd['resource_system_cycles']['scan_completion_per_iteration'] == pytest.approx(expected)


def test_scan_cost_uses_geometry_without_memory_layout_analysis(tmp_path):
    shape = (1024, )
    loaded = evaluate(tmp_path, shape, module_text=kernel().replace('128', '1024'))
    argument = '''module {
      tt.func public @main(%v: tensor<1024xi32>) -> tensor<1024xi32> {
        %s = "tt.scan"(%v) ({
        ^bb0(%a: i32, %b: i32):
          %c = arith.ori %a, %b : i32
          "tt.scan.return"(%c) : (i32) -> ()
        }) {axis = 0 : i32, reverse = false} : (tensor<1024xi32>) -> tensor<1024xi32>
        tt.return %s : tensor<1024xi32>
      }
    }'''
    bare = evaluate(tmp_path, shape, module_text=argument)
    before = {i['implementation']['mode']: i['resource_system_cycles'] for i in scan_implementations(loaded)}
    after = {i['implementation']['mode']: i['resource_system_cycles'] for i in scan_implementations(bare)}
    for mode in ('simd', 'simt'):
        assert after[mode]['scan_formula_instances_per_iteration'] == before[mode][
            'scan_formula_instances_per_iteration']
        assert after[mode]['scan_completion_per_iteration'] == pytest.approx(
            before[mode]['scan_completion_per_iteration'])


def test_complex_combine_uses_residual_fallback(tmp_path):
    text = kernel().replace('%c = arith.ori %a, %b : i32',
                            '%t = arith.ori %a, %b : i32\n          %c = arith.xori %t, %a : i32')
    report = evaluate(tmp_path, (128, ), module_text=text)
    for implementation in scan_implementations(report):
        assert not implementation['resource_system_cycles']['scan_formula_instances_per_iteration']


def test_fitted_scan_uses_base_warps_for_every_factor(tmp_path):
    from triton._C.libtriton import ir, ascend
    context = ir.context()
    ir.load_dialects(context)
    ascend.load_dialects(context)
    source = tmp_path / 'factor.mlir'
    source.write_text(kernel())
    module = ir.parse_mlir_module(str(source), context)
    report = tmp_path / 'factor.jsonl'
    manager = ir.pass_manager(context)
    ascend.passes.ttir.add_select_simd_simt_costmodel(manager, 'report', str(PROFILE), 'Ascend950PR_9579', 4, True,
                                                      True, False, 4, 1, 1024, '', '{}', str(report))
    manager.run(module, 'bitwise_factor_invariance')
    output = json.loads(report.read_text().splitlines()[-1])
    candidates = [i for i in scan_implementations(output) if i['implementation']['mode'] == 'simt']
    assert {i['implementation']['superblock_factor'] for i in candidates} == {1, 2, 4}
    costs = [i['resource_system_cycles']['scan_completion_per_iteration'] for i in candidates]
    assert min(costs) > 0
    assert costs == pytest.approx([costs[0]] * len(costs))


@pytest.mark.parametrize('shape,axis', [((1, ), 0), ((1, 128), 0), ((8, 1), 1), ((2, 1, 8), 1)])
@pytest.mark.parametrize('operation', ['ori', 'andi', 'xori', 'addi'])
@pytest.mark.parametrize('reverse', [False, True])
def test_singleton_axis_keeps_only_ordinary_work(tmp_path, shape, axis, operation, reverse):
    text = shaped_kernel(shape, axis, operation, reverse)
    report = evaluate(tmp_path, shape, axis=axis, module_text=text)
    start = text.index('%s = "tt.scan"')
    end = text.index('%flat = tt.reshape', start)
    copy_text = text[:start] + text[end:].replace('%s :', '%vshape :')
    copy = evaluate(tmp_path, shape, axis=axis, module_text=copy_text)
    stages = report['stage_model']['logical_stages']
    assert all(not stage['workload']['prefix_scan_instances_per_iteration'] for stage in stages)
    assert sum(s['workload']['scalar_operations_per_iteration'] for s in stages) == pytest.approx(
        sum(s['workload']['scalar_operations_per_iteration'] for s in copy['stage_model']['logical_stages']))
    totals = {mode: {'load': 0, 'store': 0} for mode in ('simd', 'simt')}
    for stage in stages:
        for implementation in stage['implementations']:
            mode = implementation['implementation']['mode']
            resources = implementation['resource_system_cycles']
            assert resources['scan_completion_per_iteration'] == 0
            assert resources['prefix_scan_critical_path_per_iteration'] == 0
            assert resources['prefix_scan_issue_per_iteration'] == 0
            assert not resources['scan_formula_instances_per_iteration']
            for operation in ('load', 'store'):
                totals[mode][operation] += resources[f'{operation}_per_iteration']
    assert all(value > 0 for counts in totals.values() for value in counts.values())


@pytest.mark.parametrize('leaf', ['shared_modes', {'formula': 'shared_modes'}])
def test_shared_formula_is_loaded_for_both_modes(tmp_path, leaf):
    profile, asset = local_profile(tmp_path)
    data = json.loads(asset.read_text())
    data['formulas'] = [{'id': 'shared_modes', 'terms': {'one': 1000, 'scan_instruction_work': 1}}]
    for root in data['dispatch']:
        root['tree'] = leaf
    asset.write_text(json.dumps(data))
    report = evaluate(tmp_path, (128, ), profile=profile, module_text=kernel())
    implementations = list(scan_implementations(report))
    assert {i['implementation']['mode'] for i in implementations} == {'simd', 'simt'}
    assert all(i['resource_system_cycles']['scan_formula_instances_per_iteration'] == {'shared_modes': 1}
               for i in implementations)


@pytest.mark.parametrize('leaf', ['', 'unknown_formula', 128, None, []])
def test_invalid_formula_leaf_is_rejected(tmp_path, leaf):
    profile, asset = local_profile(tmp_path)
    data = json.loads(asset.read_text())
    data['dispatch'][0]['tree'] = leaf
    asset.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        evaluate(tmp_path, (128, ), profile=profile, module_text=kernel())


@pytest.mark.parametrize('length', [13, 24, 25])
def test_non_power_of_two_uses_simd_formula_and_safe_simt_fallback(tmp_path, length):
    report = evaluate(tmp_path, (length, ), module_text=kernel().replace('128', str(length)))
    candidates = {i['implementation']['mode']: i['resource_system_cycles'] for i in scan_implementations(report)}
    assert candidates['simd']['scan_completion_per_iteration'] > 0
    assert candidates['simd']['scan_formula_instances_per_iteration']
    assert not candidates['simt']['scan_formula_instances_per_iteration']


def shaped_kernel(shape, axis, operation, reverse):
    import math
    n = math.prod(shape)
    ty = 'tensor<' + 'x'.join(map(str, shape)) + 'xi32>'
    text = kernel(operation, reverse).replace('128', str(n))
    text = text.replace('%s = "tt.scan"(%v)',
                        f'%vshape = tt.reshape %v : tensor<{n}xi32> -> {ty}\n        %s = "tt.scan"(%vshape)')
    text = text.replace('axis = 0 : i32', f'axis = {axis} : i32')
    text = text.replace(f': (tensor<{n}xi32>) -> tensor<{n}xi32>', f': ({ty}) -> {ty}')
    text = text.replace('%py = tt.splat', f'%flat = tt.reshape %s : {ty} -> tensor<{n}xi32>\n        %py = tt.splat')
    return text.replace('tt.store %yp, %s', 'tt.store %yp, %flat')


@pytest.mark.parametrize('shape,axis',
                         [((2, 4, 8, 16), axis) for axis in range(4)] + [((2, 4, 8, 2, 4), axis) for axis in range(5)])
@pytest.mark.parametrize('operation', ['ori', 'andi', 'xori'])
@pytest.mark.parametrize('reverse', [False, True])
def test_high_rank_scan_reuses_folded_geometry(tmp_path, shape, axis, operation, reverse):
    import math
    pre, post = math.prod(shape[:axis]), math.prod(shape[axis + 1:])
    if axis == 0:
        folded, folded_axis = (shape[axis], post), 0
    elif axis == len(shape) - 1:
        folded, folded_axis = (pre, shape[axis]), 1
    else:
        folded, folded_axis = (pre, shape[axis], post), 1
    high = evaluate(tmp_path, shape, axis=axis, module_text=shaped_kernel(shape, axis, operation, reverse))
    low = evaluate(tmp_path, folded, axis=folded_axis, module_text=shaped_kernel(folded, folded_axis, operation,
                                                                                 reverse))

    def costs(report):
        return {i['implementation']['mode']: i['resource_system_cycles'] for i in scan_implementations(report)}

    high_costs, low_costs = costs(high), costs(low)
    assert high_costs['simd']['scan_completion_per_iteration'] == pytest.approx(
        low_costs['simd']['scan_completion_per_iteration'])
    assert high_costs['simd']['scan_formula_instances_per_iteration'] == low_costs['simd'][
        'scan_formula_instances_per_iteration']
    assert high_costs['simt']['scan_formula_instances_per_iteration']
    assert high_costs['simt']['scan_completion_per_iteration'] > 0
