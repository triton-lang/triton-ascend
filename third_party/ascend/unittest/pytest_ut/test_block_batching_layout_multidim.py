# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from triton._C.libtriton import ascend
from test_block_batching_rewrite import parse, pointwise
from block_batching_memory_cases import memory_ir

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize('factor', [2, 3, 6])
@pytest.mark.parametrize('dtype', ['f16', 'f32'])
def test_row_view_retains_column_and_semantic_rank(tmp_path, factor, dtype):
    text = memory_ir('nd_reused_range').replace(
        '%row_offset = arith.muli %rows, %kv : tensor<4x4xi32>', '%stride = arith.constant dense<8> : tensor<4x4xi32>\n'
        '    %row_offset = arith.muli %rows, %stride : tensor<4x4xi32>').replace('f32', dtype)
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert module.verify()
    assert all(r['rewritten'] for r in json.loads(report['layout_json'])['accesses'])
    assert f'tensor<{factor*4}x4x!tt.ptr<{dtype}>>' in str(module)
    assert f'tensor<{factor}x4x4x{dtype}>' in str(module)
    assert 'tt.trans' not in str(module)


def decomposed_source():
    # Move the ordered reshape to each actual memory projection. The bound
    # and the address must both inherit the same original linear range.
    text = pointwise(0)
    text = text.replace(
        '    %v = tt.load %xa, %mask, %zero : tensor<4x!tt.ptr<f32>>', '''
    %x2 = tt.reshape %xa : tensor<4x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %m2 = tt.reshape %mask : tensor<4xi1> -> tensor<2x2xi1>
    %z2 = tt.reshape %zero : tensor<4xf32> -> tensor<2x2xf32>
    %v2 = tt.load %x2, %m2, %z2 : tensor<2x2x!tt.ptr<f32>>
    %v = tt.reshape %v2 : tensor<2x2xf32> -> tensor<4xf32>''')
    return text


def test_ordered_range_decomposition_is_proven(tmp_path):
    module = parse(tmp_path, decomposed_source())
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    rows = json.loads(report['layout_json'])['accesses']
    assert rows[0]['local_axes'] == [0, 1]
    assert rows[0]['rewritten']
    assert module.verify()
    assert 'tensor<12x!tt.ptr<f32>>' in str(module)
    assert 'tensor<3x2x2xf32>' in str(module)


def test_cross_lane_user_reshape_is_still_rejected(tmp_path):
    source = decomposed_source().replace('tt.reshape %xa :', 'tt.reshape %xa {allow_reorder} :')
    module = parse(tmp_path, source)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REJECTED'
    assert report['reason'] == 'CROSS_LANE_REORDER'
    assert str(module) == before
