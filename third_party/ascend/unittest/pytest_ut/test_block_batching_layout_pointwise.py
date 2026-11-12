# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from triton._C.libtriton import ascend
from test_block_batching_rewrite import parse
from block_batching_memory_cases import memory_ir

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def column_source():
    text = memory_ir('nd_reused_range').replace(
        '%row_offset = arith.muli %rows, %kv : tensor<4x4xi32>', '%stride = arith.constant dense<8> : tensor<4x4xi32>\n'
        '    %row_offset = arith.muli %rows, %stride : tensor<4x4xi32>')
    text = text.replace('%xa = tt.addptr', '%raw_xa = tt.addptr').replace('%ya = tt.addptr', '%raw_ya = tt.addptr')
    text = text.replace(
        '    %zero =',
        '''    %xa = tt.trans %raw_xa {order = array<i32: 1, 0>} : tensor<4x4x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %ya = tt.trans %raw_ya {order = array<i32: 1, 0>} : tensor<4x4x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %tm = tt.trans %mask {order = array<i32: 1, 0>} : tensor<4x4xi1> -> tensor<4x4xi1>
    %zero =''')
    return text.replace('tt.load %xa, %mask', 'tt.load %xa, %tm').replace('%result, %mask', '%result, %tm')


@pytest.mark.parametrize('factor', [2, 3, 6])
def test_column_pointwise_region_is_generated_in_target_shape(tmp_path, factor):
    module = parse(tmp_path, column_source())
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    rows = json.loads(report['layout_json'])['accesses']
    assert all(r['action'] == 'PointwiseRegion' and r['rewritten'] for r in rows)
    assert module.verify()
    text = str(module)
    lines = [line for line in text.splitlines() if 'arith.addf' in line]
    assert len(lines) == 1
    assert f'tensor<4x{factor*4}xf32>' in lines[0]
    # The source has two pointer transposes and one predicate transpose. No
    # new value transpose is introduced at memory/pointwise boundaries.
    assert text.count('tt.trans') == 3


def test_per_lane_computation_is_kept_at_original_domain(tmp_path):
    text = column_source().replace(
        '    %result = arith.addf %value, %ones : tensor<4x4xf32>', '''
    %pf = arith.sitofp %p : i32 to f32
    %exp = math.exp %pf : f32
    %per_block = tt.splat %exp : f32 -> tensor<4x4xf32>
    %result = arith.addf %value, %per_block : tensor<4x4xf32>''')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert all(not r['rewritten'] and r['reason'] == 'EXPENSIVE_LAYOUT_BRIDGE'
               for r in json.loads(report['layout_json'])['accesses'])
    lines = [line for line in str(module).splitlines() if 'math.exp' in line]
    assert len(lines) == 1 and 'tensor<3xf32>' in lines[0]


def test_reduce_boundary_keeps_semantic_layout(tmp_path):
    text = column_source().replace(
        '    %result = arith.addf %value, %ones : tensor<4x4xf32>', '''
    %sum = "tt.reduce"(%value) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %combined = arith.addf %a, %b : f32
        tt.reduce.return %combined : f32
    }) : (tensor<4x4xf32>) -> tensor<4xf32>
    %expanded = tt.expand_dims %sum {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
    %result = tt.broadcast %expanded : tensor<4x1xf32> -> tensor<4x4xf32>''')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert all(not r['rewritten'] for r in json.loads(report['layout_json'])['accesses'])
    assert 'axis = 2' in str(module)
    assert module.verify()
