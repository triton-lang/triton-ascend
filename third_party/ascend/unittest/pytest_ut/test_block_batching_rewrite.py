# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from triton._C.libtriton import ascend, ir
from triton.backends.ascend.program_grid import normalize_program_grid_transforms

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


def parse(tmp_path, text):
    context = ir.context()
    ir.load_dialects(context)
    ascend.ir.load_dialects(context)
    path = tmp_path / 'batch.ttir'
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def pointwise(axis):
    return '''module {
  tt.func public @arbitrary_name(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %p = tt.get_program_id AXIS : i32
    %g = tt.get_num_programs AXIS : i32
    %gf = arith.sitofp %g : i32 to f32
    %gv = tt.splat %gf : f32 -> tensor<4xf32>
    %b = arith.constant 4 : i32
    %base = arith.muli %p, %b : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %basev = tt.splat %base : i32 -> tensor<4xi32>
    %idx = arith.addi %basev, %range : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %idx, %nv : tensor<4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ya = tt.addptr %yp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    %v = tt.load %xa, %mask, %zero : tensor<4x!tt.ptr<f32>>
    %result = arith.addf %v, %gv : tensor<4xf32>
    tt.store %ya, %result, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}'''.replace('AXIS', 'xyz'[axis])


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('factor', [2, 3, 6])
def test_generic_pointwise_v3(tmp_path, axis, factor):
    module = parse(tmp_path, pointwise(axis))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, axis)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['lane_width'] == factor
    assert module.verify()
    text = str(module)
    assert 'tt.get_num_programs' not in text
    assert text.count('tt.get_program_id') == 1
    assert 'hacc.block_mapping_owner = "block_batching"' in text
    contract = normalize_program_grid_transforms(ascend.ir.get_program_grid_transforms(module))
    assert contract['version'] == 3 and contract['hidden_extent_axes'] == [0, 1, 2]
    assert contract['transforms'][0]['axis'] == axis
    assert contract['transforms'][0]['factor'] == factor
    assert module.get_function('arbitrary_name').get_num_args() == 6
    guard = json.loads(report['guard_json'])
    assert len(guard['accesses']) == 2
    before = str(module)
    assert ascend.passes.ttir.rewrite_block_batching(module, factor, axis)['reason'] == 'ALREADY_MAPPED'
    assert str(module) == before


def test_auto_axis_uses_mask_structure(tmp_path):
    module = parse(tmp_path, pointwise(2))
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, -1)
    assert report['selected_axis'] == 2 and report['compile_status'] == 'REWRITTEN'


@pytest.mark.parametrize('factor,axis', [(True, 0), (2, False), (0, 0), (-2, 0), (2, 3)])
def test_rewriter_checks_python_configuration(tmp_path, factor, axis):
    module = parse(tmp_path, pointwise(0))
    before = str(module)
    with pytest.raises(ValueError):
        ascend.passes.ttir.rewrite_block_batching(module, factor, axis)
    assert str(module) == before


def test_rejection_is_transactional(tmp_path):
    module = parse(
        tmp_path,
        pointwise(0).replace(
            '%result = arith.addf %v, %gv : tensor<4xf32>',
            '%result = "arith.addf"(%v, %gv) {unknown_batch_semantics = true} '
            ': (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>'))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REJECTED'
    assert report['reason'] == 'UNSUPPORTED_OPERATION_ATTRIBUTE'
    assert report['location'] and report['operation'] == 'arith.addf'
    assert str(module) == before and module.verify()


def test_sum_reduce_and_shared_weight(tmp_path):
    module = parse(
        tmp_path, '''module {
  tt.func public @row(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %r: !tt.ptr<f32>,
                      %w: !tt.ptr<f32>, %n: i32, %stride: i32) {
    %p = tt.get_program_id x : i32
    %base = arith.muli %p, %stride : i32
    %xp = tt.addptr %x, %base : !tt.ptr<f32>, i32
    %yp = tt.addptr %y, %base : !tt.ptr<f32>, i32
    %rp = tt.addptr %r, %p : !tt.ptr<f32>, i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %nv : tensor<4xi32>
    %xpv = tt.splat %xp : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ypv = tt.splat %yp : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %wpv = tt.splat %w : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xpv, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ya = tt.addptr %ypv, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %wa = tt.addptr %wpv, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    %v = tt.load %xa, %mask, %zero : tensor<4x!tt.ptr<f32>>
    %weight = tt.load %wa, %mask, %zero : tensor<4x!tt.ptr<f32>>
    %sum = "tt.reduce"(%v) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %c = arith.addf %a, %b : f32
        tt.reduce.return %c : f32
    }) : (tensor<4xf32>) -> f32
    tt.store %rp, %sum : !tt.ptr<f32>
    %sv = tt.splat %sum : f32 -> tensor<4xf32>
    %product = arith.mulf %sv, %weight : tensor<4xf32>
    tt.store %ya, %product, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}''')
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert module.verify()
    text = str(module)
    assert 'tensor<3x4xf32>' in text
    assert 'axis = 1 : i32' in text
    assert text.count('tt.load') == 2
    assert 'originalGridZ' in text
    assert module.get_function('row').get_num_args() == 9


@pytest.mark.parametrize('rows', [2, 4, 5, 7])
def test_auto_axis_accepts_partial_groups_and_skips_singletons(tmp_path, rows):
    text = pointwise(1).replace(
        'module {', f'module attributes {{hacc.block_batching_grid = [1 : i64, {rows} : i64, 1 : i64]}} {{', 1)
    text = text.replace('%p = tt.get_program_id y', '%unused = tt.get_program_id x : i32\n    %p = tt.get_program_id y')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, -1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == 1
    assert report['tail_policy'] == 'masked'
    assert module.verify()
