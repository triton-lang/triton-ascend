# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from triton._C.libtriton import ascend
from test_block_batching_rewrite import parse, pointwise
from test_block_batching_layout_multidim import decomposed_source
from triton.backends.ascend.block_batching import _layout_resource_refusal, _internal_backend_error

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def test_layout_boundary_0(tmp_path):
    text = decomposed_source().replace(
        '    %v2 = tt.load %x2, %m2, %z2',
        '    %col = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>\n    %col2 = tt.expand_dims %col {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>\n    %cols = tt.broadcast %col2 : tensor<1x2xi32> -> tensor<2x2xi32>\n    %one = arith.constant dense<1> : tensor<2x2xi32>\n    %valid_col = arith.cmpi slt, %cols, %one : tensor<2x2xi32>\n    %holes = arith.andi %m2, %valid_col : tensor<2x2xi1>\n    %v2 = tt.load %x2, %holes, %z2'
    )
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    row = json.loads(report['layout_json'])['accesses'][0]
    assert not row['rewritten'] and row['reason'] == 'MASK_NOT_INTERVAL_OR_RECTANGLE', row
    assert module.verify()


def test_layout_boundary_1(tmp_path):
    text = pointwise(0).replace(
        '    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>',
        '\n    %local = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>\n    %row1 = tt.expand_dims %local {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>\n    %col1 = tt.expand_dims %local {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>\n    %row = tt.broadcast %row1 : tensor<2x1xi32> -> tensor<2x2xi32>\n    %col = tt.broadcast %col1 : tensor<1x2xi32> -> tensor<2x2xi32>\n    %two = arith.constant dense<2> : tensor<2x2xi32>\n    %r2 = arith.muli %row, %two : tensor<2x2xi32>\n    %range = arith.addi %r2, %col : tensor<2x2xi32>'
    ).replace('tensor<4x', 'tensor<2x2x')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    row = json.loads(report['layout_json'])['accesses'][0]
    assert not row['rewritten'] and row['reason'] == 'MULTI_AXIS_PROVENANCE_UNPROVEN', row
    assert module.verify()


def test_layout_boundary_2(tmp_path):
    text = pointwise(0).replace(
        '    %xp = tt.splat %x',
        '    %huge = arith.constant 2305843009213693940 : i64\n    %shifted = tt.addptr %x, %huge : !tt.ptr<f32>, i64\n    %xp = tt.splat %shifted'
    )
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    row = json.loads(report['layout_json'])['accesses'][0]
    assert not row['rewritten'] and row['reason'] == 'NEW_BYTE_ENDPOINT_UNPROVEN', row
    assert module.verify()


@pytest.mark.parametrize("code,message,resource,crash", [
    (1, "error: ub overflow, requires 3637760 bits while 1769472 bits available!", True, False),
    (1, "UndefinedBehaviorSanitizer: signed integer overflow", False, True),
    (1, "runtime error: signed integer overflow", False, True),
    (-6, "ub overflow, requires 3 bits while 2 bits available", False, True),
    (1, "failed to verify generated IR", False, True),
    (1, "UB diagnostic: unspecified overflow", False, False),
])
def test_resource_diagnostic_classification(code, message, resource, crash):
    assert _layout_resource_refusal(code, message) == resource
    assert _internal_backend_error(code, message) == crash
