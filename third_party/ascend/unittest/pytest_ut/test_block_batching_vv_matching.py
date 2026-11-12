"""Structural VV admission: no names, dimensions, timing or factor selection."""
from copy import deepcopy

import pytest
from triton.backends.ascend import block_batching as batching


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
