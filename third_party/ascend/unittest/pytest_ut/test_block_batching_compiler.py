# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Source admission, lowering contracts and frontend/native IR invariants; CPU-only."""
import json
import tempfile
import unittest
from pathlib import Path
import pytest
from copy import deepcopy
from triton._C.libtriton import ascend, ir
from triton.backends.ascend.block_batching_report import vv_computation_refusal
from block_batching_test_utils import parse, pointwise, tiled_scan_ir, reduction_ir, report_module as reports, assign_npu  # noqa: F401 - pytest fixtures
from types import SimpleNamespace
from triton.backends.ascend import block_batching_report as lowering_reports

pytestmark = pytest.mark.backend('native')

# Normalized computation domains


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
    from block_batching_test_utils import recurrent_ir
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
    from block_batching_test_utils import constant_recurrence
    source = constant_recurrence(0, 8, 1, 32).replace('tt.load %xp :', 'tt.load %xp {' + marker + ' = 0 : i64} :')
    module = parse(tmp_path, source)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert str(module) == before


@pytest.mark.parametrize('semantic_only', [False, True])
@pytest.mark.parametrize('factor', [3, 6])
def test_finalized_vv_effect_markers_never_escape_sandbox(tmp_path, semantic_only, factor):
    from block_batching_test_utils import constant_recurrence
    module = parse(tmp_path, constant_recurrence(0, 14, 1, 32))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, "semantic_only" if semantic_only else "auto")
    assert report['compile_status'] == 'REWRITTEN', report
    assert (report['vv_lowering'] == 'sequential_iterations_v1') == (not semantic_only)
    assert '__vv_effect_id' not in str(module)
    assert '__vv_aggregation_id' not in str(module)
    assert module.verify()


# Independent-row admission


def report():
    return dict(
        owner="block_batching", cv_mode="disabled", lane_width=2,
        operations={"tt.load": 1, "tt.store": 1, "math.exp": 1}, layout=dict(accesses=[
            dict(action="Keep", rewritten=False, mask_class="independent_axes", original_shape=[4096]) for _ in range(2)
        ]))


@pytest.mark.parametrize("operation", ["math.exp", "arith.mulf", "arith.addf"])
@pytest.mark.parametrize("width", [64, 1024, 4096, 8192])
def test_independent_row_pointwise_requires_a_memory_view(operation, width):
    value = report()
    value["operations"] = {"tt.load": 1, "tt.store": 1, operation: 1}
    for access in value["layout"]["accesses"]:
        access["original_shape"] = [width]
    before = deepcopy(value)
    assert reports.vv_layout_refusal(value) == "VV_POINTWISE_MEMORY_VIEW_REQUIRED"
    assert value == before


@pytest.mark.parametrize("shape", [[], [64], [4096]])
@pytest.mark.parametrize("width", [128, 8192])
def test_shared_parameters_do_not_form_an_independent_row_memory_view(shape, width):
    value = report()
    for access in value["layout"]["accesses"]:
        access["original_shape"] = [width]
    value["layout"]["accesses"].append(dict(action="Keep", rewritten=False, mask_class="shared", original_shape=shape))
    before = deepcopy(value)
    assert reports.vv_layout_refusal(value) == "VV_POINTWISE_MEMORY_VIEW_REQUIRED"
    assert value == before


@pytest.mark.parametrize('operation', ['arith.addf', 'tt.reduce', 'tt.scan'])
def test_unsupported_symbolic_bridge_requires_complete_fallback(operation):
    value = report()
    value['operations'][operation] = 1
    value['layout']['accesses'][0].update(original_shape=[10, 4, 4], symbolic_stride=True, mask_class='linear_prefix',
                                          reason='SYMBOLIC_LAYOUT_BRIDGE_UNSUPPORTED')
    before = deepcopy(value)
    assert reports.vv_layout_refusal(value) == 'VV_SYMBOLIC_LAYOUT_NOT_PROVEN'
    assert value == before
    value['layout']['accesses'][0].update(action='MemoryView', rewritten=True, reason='PROVEN_SYMBOLIC_STRIDE_CHAIN')
    assert reports.vv_layout_refusal(value) is None
    value['layout']['accesses'][0]['reason'] = 'SYMBOLIC_LAYOUT_BRIDGE_UNSUPPORTED'
    value['cv_mode'] = 'spatial_concat_v1'
    assert reports.vv_layout_refusal(value) is None


@pytest.mark.parametrize("kind", [
    "baseline", "no_owner", "cv", "reduction", "scan", "dot", "loop", "if", "shared_only", "shared_high_rank", "view",
    "higher_rank", "empty", "scalar", "unknown_mask"
])
def test_other_structures_keep_their_existing_admission(kind):
    value = report()
    if kind == "baseline":
        value["lane_width"] = 1
    elif kind == "no_owner":
        value["owner"] = "none"
    elif kind == "cv":
        value["cv_mode"] = "independent_batch_v1"
    elif kind in ("reduction", "scan", "dot", "loop", "if"):
        value["operations"][dict(reduction="tt.reduce", scan="tt.scan", dot="tt.dot", loop="scf.for",
                                 **{"if": "scf.if"})[kind]] = 1
    elif kind == "shared_only":
        for access in value["layout"]["accesses"]:
            access["mask_class"] = "shared"
    elif kind == "shared_high_rank":
        value["layout"]["accesses"].append(
            dict(action="Keep", rewritten=False, mask_class="shared", original_shape=[32, 128]))
    elif kind == "view":
        value["layout"]["accesses"][0].update(action="MemoryView", rewritten=True, mask_class="linear_prefix")
    elif kind == "higher_rank":
        value["layout"]["accesses"][0]["original_shape"] = [32, 128]
    elif kind == "empty":
        value["layout"]["accesses"] = []
    elif kind == "scalar":
        value["layout"]["accesses"][0]["original_shape"] = []
    elif kind == "unknown_mask":
        value["layout"]["accesses"][0]["mask_class"] = "unknown"
    assert reports.vv_layout_refusal(value) is None


@pytest.mark.parametrize('shape', [[1, 256], [256, 1], [1, 256, 1]])
def test_singleton_dimensions_preserve_independent_row_decision(shape):
    value = report()
    for access in value['layout']['accesses']:
        access['original_shape'] = shape
    assert reports.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'


@pytest.mark.parametrize('op', ['scf.if', 'scf.for'])
def test_value_only_control_flow_keeps_memory_view_requirement(op):
    value = report()
    value['operations'][op] = 1
    for access in value['layout']['accesses']:
        access['control_flow_independent'] = True
    assert reports.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'
    value['layout']['accesses'][0]['control_flow_independent'] = False
    assert reports.vv_layout_refusal(value) is None


def structured_rows():
    value = report()
    for access in value['layout']['accesses']:
        access.update(reason='ALREADY_STRUCTURED', control_flow_independent=True, proof_scope='existing_guard_no_wrap')
    return value


@pytest.mark.parametrize('shape', [[128], [1, 128], [128, 1], [1]])
def test_proved_top_level_rows_reach_backend_memory_validation(shape):
    value = structured_rows()
    for access in value['layout']['accesses']:
        access['original_shape'] = shape
    before = deepcopy(value)
    assert reports.vv_layout_refusal(value) is None
    # Source layout evidence cannot replace backend memory validation.
    assert reports.vv_memory_lowering_refusal(value) == 'VV_MEMORY_LOWERING_NOT_PROVEN'
    assert value == before


@pytest.mark.parametrize('field,bad', [('reason', 'UNKNOWN'), ('control_flow_independent', False),
                                       ('proof_scope', None)])
def test_partial_row_evidence_retains_structural_refusal(field, bad):
    value = structured_rows()
    value['layout']['accesses'][0][field] = bad
    assert reports.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'


@pytest.mark.parametrize('operation', ['scf.for', 'scf.if'])
def test_row_exception_does_not_admit_control_flow(operation):
    value = structured_rows()
    value['operations'][operation] = 1
    assert reports.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'


def test_row_exception_does_not_admit_shared_parameters_or_indirect_memory():
    value = structured_rows()
    value['layout']['accesses'][0].update(reason='SHARED_ACCESS', mask_class='shared')
    assert reports.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'
    value = structured_rows()
    value['layout']['lowering_observation'] = dict(stage='ttadapter', scope='module', indirect_load_ops=1,
                                                   indirect_store_ops=0)
    assert reports.vv_memory_lowering_refusal(value) == 'VV_INDIRECT_MEMORY_NOT_SUPPORTED'


def metadata():
    return dict(
        block_batching_factor=2, target=SimpleNamespace(arch="Ascend950PR_9579"), mix_mode="aiv", parallel_mode="simd",
        block_batching=dict(owner="block_batching", cv_mode="disabled",
                            memory_lowering=dict(dynamic_contiguous_copy=True)))


@pytest.mark.parametrize("change",
                         ["none", "baseline", "no_report", "wrong_owner", "cv", "no_copy", "arch", "mix", "simt"])
def test_compile_policy_is_candidate_only(change):
    value = metadata()
    if change == "baseline":
        value["block_batching_factor"] = 1
    if change == "no_report":
        del value["block_batching"]
    if change == "wrong_owner":
        value["block_batching"]["owner"] = "legacy"
    if change == "cv":
        value["block_batching"]["cv_mode"] = "batch"
    if change == "no_copy":
        value["block_batching"]["memory_lowering"]["dynamic_contiguous_copy"] = False
    if change == "arch":
        value["target"].arch = "Ascend910B4"
    if change == "mix":
        value["mix_mode"] = "mix"
    if change == "simt":
        value["parallel_mode"] = "simt"
    calls = []

    def supported(path):
        calls.append(path)
        return True

    options = reports.memory_compile_options(value, "/compiler", supported)
    assert options == (["--enable-lib-call-no-inline=false"] if change == "none" else [])
    assert calls == (["/compiler"] if change == "none" else [])


def test_unsupported_compiler_keeps_candidate_policy_default():
    value = metadata()
    assert reports.memory_compile_options(value, "/compiler", lambda _: False) == []
    assert value["block_batching_factor"] == 2
    assert value["block_batching"]["memory_lowering"]["libcall_inlining"] is False


@pytest.mark.parametrize(
    'change', ['none', 'missing_proof', 'indirect', 'missing_observation', 'no_copy', 'cv', 'baseline', 'unsupported'])
def test_structured_row_inlining_requires_source_and_backend_evidence(change):
    value = metadata()
    report = value['block_batching']
    report['memory_lowering']['dynamic_contiguous_copy'] = False
    report['layout'] = dict(
        accesses=[
            dict(action='Keep', rewritten=False, original_shape=[128], reason='ALREADY_STRUCTURED',
                 control_flow_independent=True, proof_scope='existing_guard_no_wrap', mask_class='independent_axes')
        ], lowering_observation=dict(stage='ttadapter', scope='module', indirect_load_ops=0, indirect_store_ops=0,
                                     memref_copy_ops=1))
    if change == 'missing_proof':
        del report['layout']['accesses'][0]['proof_scope']
    if change == 'indirect':
        report['layout']['lowering_observation']['indirect_load_ops'] = 1
    if change == 'missing_observation':
        del report['layout']['lowering_observation']
    if change == 'no_copy':
        report['layout']['lowering_observation']['memref_copy_ops'] = 0
    if change == 'cv':
        report['cv_mode'] = 'spatial_concat_v1'
    if change == 'baseline':
        value['block_batching_factor'] = 1
    calls = []

    def supported(path):
        calls.append(path)
        return change != 'unsupported'

    options = reports.memory_compile_options(value, '/compiler', supported)
    assert options == (['--enable-lib-call-no-inline=false'] if change == 'none' else [])
    assert calls == (['/compiler'] if change in ('none', 'unsupported') else [])
    assert report['memory_lowering']['dynamic_contiguous_copy'] is False


# Lowered-memory evidence


def lowered_report(operation, loads=0, stores=0):
    return dict(
        owner='block_batching', cv_mode='disabled', lane_width=2,
        operations={'tt.load': 1, 'tt.store': 1, operation: 1}, layout={
            'lowering_observation':
            dict(scope='module', stage='ttadapter', indirect_load_ops=loads, indirect_store_ops=stores,
                 memref_copy_ops=3)
        })


@pytest.mark.parametrize('operation', ['arith.mulf', 'tt.reduce', 'tt.scan'])
@pytest.mark.parametrize('loads,stores', [(1, 0), (0, 1), (2, 1)])
def test_unsupported_lowered_memory_cannot_bypass_admission(operation, loads, stores):
    report = lowered_report(operation, loads, stores)
    before = deepcopy(report)
    assert lowering_reports.vv_memory_lowering_refusal(report) == 'VV_INDIRECT_MEMORY_NOT_SUPPORTED'
    assert report == before


@pytest.mark.parametrize('operation', ['tt.reduce', 'tt.scan', 'scf.for', 'arith.mulf'])
def test_structured_memory_keeps_existing_compute_support(operation):
    report = lowered_report(operation)
    report['lane_scans_lowered'] = 2 if operation == 'tt.scan' else 0
    assert lowering_reports.vv_memory_lowering_refusal(report) is None


@pytest.mark.parametrize('change', [dict(owner='none'), dict(lane_width=1), dict(cv_mode='independent_batch_v1')])
def test_lowering_refusal_is_owned_by_vv_batching(change):
    report = lowered_report('tt.reduce', loads=1, stores=1)
    report.update(change)
    assert lowering_reports.vv_memory_lowering_refusal(report) is None


@pytest.mark.parametrize('observation', [
    None, {}, {'stage': 'ttadapter'}, {'stage': 'ttadapter', 'scope': 'module', 'indirect_load_ops': 0}, {
        'stage': 'ttadapter', 'scope': 'module', 'indirect_load_ops': -1, 'indirect_store_ops': 0
    }, {'stage': 'ttir', 'scope': 'module', 'indirect_load_ops': 0, 'indirect_store_ops': 0},
    {'stage': 'ttadapter', 'scope': 'module', 'indirect_load_ops': False, 'indirect_store_ops': 0}
])
def test_missing_lowering_proof_is_not_zero_indirect_memory(observation):
    report = lowered_report('tt.reduce')
    report['layout']['lowering_observation'] = observation
    assert lowering_reports.vv_memory_lowering_refusal(report) == 'VV_MEMORY_LOWERING_NOT_PROVEN'


# Frontend snapshots and native proof invariants


class TestFrontendSnapshot(unittest.TestCase):

    def test_frontend_snapshot_preserves_llvm_assume_properties(self):
        context = ir.context()
        ir.load_dialects(context)
        ascend.load_dialects(context)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.mlir'
            source.write_text('''module {
              tt.func @kernel(%cond: i1) {
                "llvm.intr.assume"(%cond) <{op_bundle_sizes = array<i32>}> : (i1) -> ()
                tt.return
              }
            }''')
            module = ir.parse_mlir_module(str(source), context)
            bytecode = ascend.serialize_block_input(module)
            self.assertTrue(bytecode.startswith(b'ML\xefR'))
            source.write_bytes(bytecode)
            restored = ir.parse_mlir_module(str(source), context)
            self.assertIn('llvm.intr.assume', str(restored))


@pytest.mark.parametrize('test_name,implementations', [
    ('EffectProjectionTest', ['BatchExpr.cpp', 'BatchFactsJSON.cpp']),
    ('AffineProfilesTest',
     ['BatchExpr.cpp', 'BatchAffine.cpp', 'BatchRangeAnalysis.cpp', 'BatchLoopSchedule.cpp', 'BatchLoopLegality.cpp']),
])
def test_checked_batching_values_under_ubsan(tmp_path, test_name, implementations):
    import os
    from pathlib import Path
    import re
    import shlex
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[2]
    llvm = shutil.which('llvm-config')
    if os.getenv('LLVM_SYSPATH'):
        llvm = str(Path(os.environ['LLVM_SYSPATH']) / 'bin/llvm-config')
    if not llvm:
        for cache in (root.parents[1] / 'build').glob('*/CMakeCache.txt'):
            match = re.search(r'^LLVM_DIR:[^=]+=(.+)/lib/cmake/llvm$', cache.read_text(), re.M)
            if match:
                llvm = str(Path(match[1]) / 'bin/llvm-config')
                break
    compiler = os.getenv('CXX', 'g++')
    if not llvm or not Path(llvm).is_file() or not shutil.which(compiler):
        pytest.skip('owned-fact UBSan test needs a C++ compiler and LLVM development installation')
    include = subprocess.check_output([llvm, '--includedir'], text=True).strip()
    libs = shlex.split(subprocess.check_output([llvm, '--ldflags', '--libs', 'support', '--system-libs'], text=True))
    sources = [root / 'unittest/BlockBatching' / (test_name + '.cpp')]
    sources += [root / 'lib/TritonToGraph/BlockBatching' / name for name in implementations]
    exe = tmp_path / test_name
    command = [
        compiler, '-std=c++17', '-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=all',
        '-I' + str(root / 'include'), '-I' + include, *map(str, sources), *libs, '-o',
        str(exe)
    ]
    # Match the linked LLVM build; UBSan vptr instrumentation requires RTTI
    # symbols which a no-RTTI LLVMSupport intentionally does not export.
    if subprocess.check_output([llvm, '--has-rtti'], text=True).strip() == 'NO':
        command.insert(1, '-fno-rtti')
    subprocess.run(command, capture_output=True, text=True, check=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    if test_name == 'EffectProjectionTest':
        assert 'owned effect projection regressions passed' in result.stdout
    assert not result.stderr
