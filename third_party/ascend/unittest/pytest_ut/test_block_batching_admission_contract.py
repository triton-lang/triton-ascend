# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Admission evidence comes from the normalized kernel, not its name."""
import json
from copy import deepcopy

import pytest
from triton._C.libtriton import ascend
from triton.backends.ascend.block_batching import vv_computation_refusal
from block_batching_contract_ir import parse, pointwise
from test_block_batching_rectangular_mask import tiled_scan_ir
from block_batching_contract_ir import reduction_ir

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def rewrite(tmp_path, text, factor=3, axis=0):
    module = parse(tmp_path, text)
    report = dict(ascend.passes.ttir.rewrite_block_batching(module, factor, axis))
    report['computation'] = json.loads(report['computation_json'])
    report['owner'] = 'block_batching'
    return module, report


@pytest.mark.parametrize('kind', ['sum', 'scan', 'reverse_scan'])
@pytest.mark.parametrize('factor', [2, 3, 8])
def test_original_aggregation_axis_is_not_batch_axis(tmp_path, kind, factor):
    module, report = rewrite(tmp_path, reduction_ir(kind), factor)
    assert report['compile_status'] == 'REWRITTEN', report
    assert vv_computation_refusal(report) is None
    aggregation, = report['computation']['aggregations']
    assert aggregation['original_shape'] == [4]
    assert aggregation['effective_shape'] == [4]
    assert aggregation['original_axis'] == 0
    assert aggregation['lifted_axis'] == 1
    assert aggregation['effective_axis'] == 0
    assert aggregation['axis_extent'] == 4
    assert 'axis = 1 : i32' in str(module)


@pytest.mark.parametrize('shape,axis', [('1x4', 1), ('4x1', 0)])
def test_equivalent_scan_shapes_have_same_effective_domain(tmp_path, shape, axis):
    text = tiled_scan_ir().replace('4x4xf32', f'{shape}xf32').replace('axis = 0', f'axis = {axis}')
    _, report = rewrite(tmp_path, text, axis=1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert vv_computation_refusal(report) is None
    aggregation, = report['computation']['aggregations']
    assert aggregation['effective_shape'] == [4]
    assert aggregation['effective_axis'] == 0
    assert aggregation['domain'] == 'independent_row_scan'


def test_unsupported_scan_preserves_structural_reason_without_mutation(tmp_path):
    source = tiled_scan_ir()
    module = parse(tmp_path, source)
    before = str(module)
    report = dict(ascend.passes.ttir.rewrite_block_batching(module, 3, 1))
    assert report['reason'] == 'VV_SCAN_DOMAIN_NOT_SUPPORTED'
    aggregation, = json.loads(report['computation_json'])['aggregations']
    assert aggregation['effective_shape'] == [4, 4]
    assert aggregation['domain'] == 'unsupported_multidimensional_scan'
    assert str(module) == before


@pytest.mark.parametrize('mutation', ['missing', 'axis', 'domain', 'count'])
def test_incomplete_or_inconsistent_domain_cannot_pass_as_pointwise(tmp_path, mutation):
    _, report = rewrite(tmp_path, reduction_ir('sum'))
    report = deepcopy(report)
    if mutation == 'missing':
        report.pop('computation')
    elif mutation == 'axis':
        report['computation']['aggregations'][0]['lifted_axis'] = 0
    elif mutation == 'domain':
        report['computation']['aggregations'][0]['domain'] = 'unknown'
    else:
        report['computation']['aggregations'] = []
    assert vv_computation_refusal(report) == 'VV_COMPUTATION_DOMAIN_NOT_PROVEN'


def test_pointwise_contract_does_not_require_aggregation(tmp_path):
    _, report = rewrite(tmp_path, pointwise(0))
    assert report['compile_status'] == 'REWRITTEN'
    assert report['computation']['aggregations'] == []
    assert vv_computation_refusal(report) is None


@pytest.mark.parametrize('policy,expected_axis', [('auto', 0), ('semantic_only', 1)])
@pytest.mark.parametrize('inside', [False, True])
@pytest.mark.parametrize('kind', ['reduce', 'scan', 'reverse_scan'])
@pytest.mark.parametrize('factor', [3, 6])
def test_actual_aggregation_mapping_for_both_schedules(tmp_path, policy, expected_axis, inside, kind, factor):
    from block_batching_contract_ir import recurrent_ir
    scan = kind != 'reduce'
    op = 'scan' if scan else 'reduce'
    source = '%v' if inside else '%sum'
    result_type = 'tensor<4xf32>' if scan else 'f32'
    reverse = ', reverse = ' + ('true' if kind == 'reverse_scan' else 'false') if scan else ''
    aggregation = f'''    %aggregation = "tt.{op}"({source}) <{{axis = 0 : i32{reverse}}}> ({{
      ^bb0(%a: f32, %b: f32):
        %s = arith.addf %a, %b : f32
        tt.{op}.return %s : f32
    }}) : (tensor<4xf32>) -> {result_type}
'''
    output = '%aggregation'
    if not scan:
        aggregation += '    %repeated = tt.splat %aggregation : f32 -> tensor<4xf32>\n'
        output = '%repeated'
    text = recurrent_ir()
    if inside:
        text = text.replace('      %next = arith.addf %acc, %v',
                            aggregation + f'      %next = arith.addf %acc, {output}')
    else:
        text = text.replace('    tt.store %op, %sum', aggregation + f'    tt.store %op, {output}')
    module = parse(tmp_path, text)
    report = dict(ascend.passes.ttir.rewrite_block_batching(module, factor, 0, policy))
    assert report['compile_status'] == 'REWRITTEN', report
    report['owner'] = 'block_batching'
    report['computation'] = json.loads(report['computation_json'])
    assert vv_computation_refusal(report) is None
    record, = report['computation']['aggregations']
    targets = record['mapped_operations']
    assert record['lifted_axis'] == expected_axis
    assert all(t['axis'] == expected_axis for t in targets)
    assert all(t['shape'] == ([4] if policy == 'auto' else [factor, 4]) for t in targets)
    assert len(targets) == (factor + 1 if inside and policy == 'auto' else 1)
    assert '__vv_aggregation_id' not in str(module)
    assert module.verify()


@pytest.mark.parametrize('field,value', [('axis', 0), ('shape', [4]), ('batch_prefix', False), ('id', 99)])
def test_generated_mapping_tampering_is_rejected(tmp_path, field, value):
    _, report = rewrite(tmp_path, reduction_ir('sum'))
    report['computation']['aggregations'][0]['mapped_operations'][0][field] = value
    assert vv_computation_refusal(report) == 'VV_COMPUTATION_DOMAIN_NOT_PROVEN'


@pytest.mark.parametrize('field,value', [('version', 1), ('schedule', 'sequential_iterations'), ('stage', 'device_ir')])
def test_mapping_stage_and_schedule_cannot_be_substituted(tmp_path, field, value):
    _, report = rewrite(tmp_path, reduction_ir('sum'))
    report['computation'][field] = value
    assert vv_computation_refusal(report) == 'VV_COMPUTATION_DOMAIN_NOT_PROVEN'


@pytest.mark.parametrize('change', [dict(owner='none'), dict(lane_width=1), dict(cv_mode='independent_batch_v1')])
def test_domain_policy_does_not_change_other_owners(change):
    report = dict(owner='block_batching', cv_mode='disabled', lane_width=3)
    report.update(change)
    assert vv_computation_refusal(report) is None


def tile_reduction_ir(form):
    import re
    source = re.sub(r'%r\b', '%row_offset', tiled_scan_ir()).replace('%column', '%column_pid')
    begin, end = source.index('    %out ='), source.index('    tt.store')

    def reduce(name, value, shape, axis, result):
        return f'''    %{name} = "tt.reduce"(%{value}) <{{axis = {axis} : i32}}> ({{
          ^bb0(%a: f32, %b: f32):
            %s = arith.addf %a, %b : f32
            tt.reduce.return %s : f32
        }}) : (tensor<{shape}xf32>) -> {result}
'''

    body = reduce('r', 'v', '4x4', 1, 'tensor<4xf32>')
    value = 'r'
    if form == 'elementwise':
        body += '    %r2 = arith.addf %r, %r : tensor<4xf32>\n'
        value = 'r2'
    elif form in ('loop', 'loop_row'):
        body += '''    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %r2 = scf.for %i = %zero to %n step %one iter_args(%acc = %r) -> (tensor<4xf32>) : i32 {
      %next = arith.addf %acc, %r : tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
'''
        value = 'r2'
    elif form == 'branch':
        body += '''    %cond = arith.cmpi slt, %row, %n : i32
    %r2 = scf.if %cond -> (tensor<4xf32>) {
      scf.yield %r : tensor<4xf32>
    } else {
      %next = arith.addf %r, %r : tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
'''
        value = 'r2'
    if form in ('row', 'loop_row'):
        body += f'''    %expanded = tt.expand_dims %{value} {{axis = 1 : i32}} : tensor<4xf32> -> tensor<4x1xf32>
    %broadcast = tt.broadcast %expanded : tensor<4x1xf32> -> tensor<4x4xf32>
    %biased = arith.addf %v, %broadcast : tensor<4x4xf32>
'''
        body += reduce('again', 'biased', '4x4', 1, 'tensor<4xf32>')
        body += '''    %e = tt.expand_dims %again {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
    %out = tt.broadcast %e : tensor<4x1xf32> -> tensor<4x4xf32>
'''
    elif form in ('singleton', 'transpose'):
        if form == 'singleton':
            body += '    %e = tt.reshape %r : tensor<4xf32> -> tensor<1x4xf32>\n'
        else:
            body += '''    %column = tt.expand_dims %r {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
    %e = tt.trans %column {order = array<i32: 1, 0>} : tensor<4x1xf32> -> tensor<1x4xf32>
'''
        body += reduce('last', 'e', '1x4', 1, 'tensor<1xf32>')
        body += '''    %e2 = tt.expand_dims %last {axis = 1 : i32} : tensor<1xf32> -> tensor<1x1xf32>
    %out = tt.broadcast %e2 : tensor<1x1xf32> -> tensor<4x4xf32>
'''
    else:
        body += reduce('last', value, '4', 0, 'f32')
        body += '    %out = tt.splat %last : f32 -> tensor<4x4xf32>\n'
    return source[:begin] + body + source[end:]


@pytest.mark.parametrize('form', ['direct', 'elementwise', 'singleton', 'transpose', 'loop', 'branch'])
@pytest.mark.parametrize('factor', [2, 3, 8])
def test_multiaxis_tile_reduction_is_not_admitted_as_independent_rows(tmp_path, form, factor):
    module = parse(tmp_path, tile_reduction_ir(form))
    before = str(module)
    report = dict(ascend.passes.ttir.rewrite_block_batching(module, factor, 1))
    assert report['reason'] == 'VV_MULTI_AXIS_REDUCTION_NOT_SUPPORTED', report
    assert report['operation'] == 'tt.reduce'
    assert str(module) == before


@pytest.mark.parametrize('form', ['row', 'loop_row'])
@pytest.mark.parametrize('factor', [2, 3])
def test_repeated_row_reduction_preserves_domain_even_through_loop(tmp_path, form, factor):
    module, report = rewrite(tmp_path, tile_reduction_ir(form), factor, 1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert vv_computation_refusal(report) is None
    assert module.verify()


def test_singleton_source_tile_does_not_become_a_multiaxis_refusal(tmp_path):
    source = tile_reduction_ir('direct').replace('4x4xf32', '1x4xf32').replace('tensor<4xf32>', 'tensor<1xf32>')
    module, report = rewrite(tmp_path, source, 3, 1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert vv_computation_refusal(report) is None
    assert module.verify()


def test_unproven_flattened_row_coordinates_cannot_enter_next_reduction(tmp_path):
    source = tile_reduction_ir('singleton').replace(
        '%e = tt.reshape %r : tensor<4xf32> -> tensor<1x4xf32>',
        '''%column = tt.expand_dims %r {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
    %matrix = tt.broadcast %column : tensor<4x1xf32> -> tensor<4x4xf32>
    %e = tt.reshape %matrix : tensor<4x4xf32> -> tensor<1x16xf32>''').replace(
            '}) : (tensor<1x4xf32>) -> tensor<1xf32>', '}) : (tensor<1x16xf32>) -> tensor<1xf32>')
    module = parse(tmp_path, source)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 1)
    assert report['reason'] == 'VV_AGGREGATION_DOMAIN_NOT_PROVEN', report
    assert str(module) == before


@pytest.mark.parametrize('marker', ['__vv_effect_id', '__vv_aggregation_id'])
def test_source_cannot_supply_private_vv_provenance(tmp_path, marker):
    from block_batching_contract_ir import constant_recurrence
    source = constant_recurrence(0, 8, 1, 32).replace('tt.load %xp :', 'tt.load %xp {' + marker + ' = 0 : i64} :')
    module = parse(tmp_path, source)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert str(module) == before


@pytest.mark.parametrize('semantic_only', [False, True])
@pytest.mark.parametrize('factor', [3, 6])
def test_finalized_vv_effect_markers_never_escape_sandbox(tmp_path, semantic_only, factor):
    from block_batching_contract_ir import constant_recurrence
    module = parse(tmp_path, constant_recurrence(0, 14, 1, 32))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, "semantic_only" if semantic_only else "auto")
    assert report['compile_status'] == 'REWRITTEN', report
    assert (report['vv_lowering'] == 'sequential_iterations_v1') == (not semantic_only)
    assert '__vv_effect_id' not in str(module)
    assert '__vv_aggregation_id' not in str(module)
    assert module.verify()
