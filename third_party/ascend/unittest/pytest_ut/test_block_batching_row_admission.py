# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Structural VV admission: no names, dimensions, timing or factor selection."""
from copy import deepcopy

import pytest
from block_batching_contract_host import batching
from types import SimpleNamespace

pytestmark = pytest.mark.backend('native')


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
    assert batching.vv_layout_refusal(value) == "VV_POINTWISE_MEMORY_VIEW_REQUIRED"
    assert value == before


@pytest.mark.parametrize("shape", [[], [64], [4096]])
@pytest.mark.parametrize("width", [128, 8192])
def test_shared_parameters_do_not_form_an_independent_row_memory_view(shape, width):
    value = report()
    for access in value["layout"]["accesses"]:
        access["original_shape"] = [width]
    value["layout"]["accesses"].append(dict(action="Keep", rewritten=False, mask_class="shared", original_shape=shape))
    before = deepcopy(value)
    assert batching.vv_layout_refusal(value) == "VV_POINTWISE_MEMORY_VIEW_REQUIRED"
    assert value == before


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
    assert batching.vv_layout_refusal(value) is None


@pytest.mark.parametrize('shape', [[1, 256], [256, 1], [1, 256, 1]])
def test_singleton_dimensions_preserve_independent_row_decision(shape):
    value = report()
    for access in value['layout']['accesses']:
        access['original_shape'] = shape
    assert batching.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'


@pytest.mark.parametrize('op', ['scf.if', 'scf.for'])
def test_value_only_control_flow_keeps_memory_view_requirement(op):
    value = report()
    value['operations'][op] = 1
    for access in value['layout']['accesses']:
        access['control_flow_independent'] = True
    assert batching.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'
    value['layout']['accesses'][0]['control_flow_independent'] = False
    assert batching.vv_layout_refusal(value) is None


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
    assert batching.vv_layout_refusal(value) is None
    # Source layout evidence cannot replace backend memory validation.
    assert batching.vv_memory_lowering_refusal(value) == 'VV_MEMORY_LOWERING_NOT_PROVEN'
    assert value == before


@pytest.mark.parametrize('field,bad', [('reason', 'UNKNOWN'), ('control_flow_independent', False),
                                       ('proof_scope', None)])
def test_partial_row_evidence_retains_structural_refusal(field, bad):
    value = structured_rows()
    value['layout']['accesses'][0][field] = bad
    assert batching.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'


@pytest.mark.parametrize('operation', ['scf.for', 'scf.if'])
def test_row_exception_does_not_admit_control_flow(operation):
    value = structured_rows()
    value['operations'][operation] = 1
    assert batching.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'


def test_row_exception_does_not_admit_shared_parameters_or_indirect_memory():
    value = structured_rows()
    value['layout']['accesses'][0].update(reason='SHARED_ACCESS', mask_class='shared')
    assert batching.vv_layout_refusal(value) == 'VV_POINTWISE_MEMORY_VIEW_REQUIRED'
    value = structured_rows()
    value['layout']['lowering_observation'] = dict(stage='ttadapter', scope='module', indirect_load_ops=1,
                                                   indirect_store_ops=0)
    assert batching.vv_memory_lowering_refusal(value) == 'VV_INDIRECT_MEMORY_NOT_SUPPORTED'


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

    options = batching.memory_compile_options(value, "/compiler", supported)
    assert options == (["--enable-lib-call-no-inline=false"] if change == "none" else [])
    assert calls == (["/compiler"] if change == "none" else [])


def test_unsupported_compiler_keeps_candidate_policy_default():
    value = metadata()
    assert batching.memory_compile_options(value, "/compiler", lambda _: False) == []
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

    options = batching.memory_compile_options(value, '/compiler', supported)
    assert options == (['--enable-lib-call-no-inline=false'] if change == 'none' else [])
    assert calls == (['/compiler'] if change in ('none', 'unsupported') else [])
    assert report['memory_lowering']['dynamic_contiguous_copy'] is False
