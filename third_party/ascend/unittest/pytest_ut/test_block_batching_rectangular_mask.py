# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Block-pointer nonnegative checks must not hide a proven rectangular view."""
import json
import pytest
from triton._C.libtriton import ascend
from block_batching_contract_ir import parse, pointwise
from block_batching_contract_ir import analyze

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def with_lower_bound(text, value='%idx', bound='0'):
    extra = f'''%lower = arith.constant dense<{bound}> : tensor<4xi32>
    %nonnegative = arith.cmpi sge, {value}, %lower : tensor<4xi32>
    %rectangle = arith.andi %mask, %nonnegative : tensor<4xi1>
    %xp ='''
    return text.replace('%xp =', extra).replace('tt.load %xa, %mask',
                                                'tt.load %xa, %rectangle').replace('%result, %mask',
                                                                                   '%result, %rectangle')


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('factor', [2, 3, 6])
def test_nonnegative_program_coordinate_keeps_memory_view(tmp_path, axis, factor):
    source = with_lower_bound(pointwise(axis))
    assert all(row['action'] == 'MemoryView' for row in analyze(tmp_path, source, factor, axis))
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, axis)
    assert report['compile_status'] == 'REWRITTEN', report
    assert all(row['rewritten'] for row in json.loads(report['layout_json'])['accesses'])
    # The original nonnegative check remains in the guard proof.
    assert '["const",-1,32]' in report['guard_json'].replace(' ', '')
    assert module.verify()


@pytest.mark.parametrize('case', ['negative_origin', 'negative_coefficient', 'dynamic_origin', 'positive_lower'])
def test_unproven_lower_bound_does_not_disappear(tmp_path, case):
    text = pointwise(0)
    if case == 'negative_origin':
        text = text.replace('%idx = arith.addi %basev, %range', '%raw = arith.addi %basev, %range').replace(
            '%nv = tt.splat',
            '%one = arith.constant dense<1> : tensor<4xi32>\n    %idx = arith.subi %raw, %one : tensor<4xi32>\n    %nv = tt.splat'
        )
    elif case == 'dynamic_origin':
        text = text.replace('%idx = arith.addi %basev, %range', '%raw = arith.addi %basev, %range').replace(
            '%nv = tt.splat',
            '%shift = tt.splat %n : i32 -> tensor<4xi32>\n    %idx = arith.addi %raw, %shift : tensor<4xi32>\n    %nv = tt.splat'
        )
    elif case == 'negative_coefficient':
        text = text.replace('%idx = arith.addi %basev, %range', '%idx = arith.subi %range, %basev')
    rows = analyze(tmp_path, with_lower_bound(text, bound='1' if case == 'positive_lower' else '0'))
    assert all(row['action'] == 'Keep' for row in rows)


def test_nonnegative_simplification_retains_runtime_overflow_refusal(tmp_path):
    from triton.backends.ascend.block_batching_guard import check_guard
    module = parse(tmp_path, with_lower_bound(pointwise(0)))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    guard = dict(json.loads(report['guard_json']), axis=0)
    pointers = [1 << 32, 1 << 45]
    assert check_guard(guard, [*pointers, 64], (16, 1, 1), 2) == (True, 'PROVEN_AT_RUNTIME')
    assert check_guard(guard, [*pointers, 2**31 - 1], (2**30, 1, 1), 2) == (False, 'INDEX_OVERFLOW')
    assert check_guard(guard, [*pointers, 64], (16, 1, 1), 2) == (True, 'PROVEN_AT_RUNTIME')


def divided_origin(axis):
    text = with_lower_bound(pointwise(0))
    extra = f'''%other = tt.get_program_id {axis} : i32
    %four = arith.constant 4 : i32
    %eight = arith.constant 8 : i32
    %q = arith.divsi %other, %four : i32
    %q8 = arith.divsi %other, %eight : i32
    %rem = arith.remsi %other, %four : i32
    %scaled = arith.muli %q, %n : i32
    %base1 = arith.addi %scaled, %rem : i32
    %shared = arith.addi %base1, %q8 : i32
    %rootx = tt.addptr %x, %shared : !tt.ptr<f32>, i32
    %rooty = tt.addptr %y, %shared : !tt.ptr<f32>, i32
    %xp ='''
    return text.replace('%xp =', extra).replace('tt.splat %x :',
                                                'tt.splat %rootx :').replace('tt.splat %y :', 'tt.splat %rooty :')


def test_other_axis_quotient_remainder_preserves_divisors_and_guard(tmp_path):
    module = parse(tmp_path, divided_origin('y'))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert all(row['rewritten'] for row in json.loads(report['layout_json'])['accesses'])
    guard = json.loads(report['guard_json'])
    checks = json.dumps(guard['layout_checks'])
    assert '["pid_div", 1, 4, 32]' in checks
    assert '["pid_div", 1, 8, 32]' in checks
    assert '["pid_rem", 1, 4, 32]' in checks
    assert 'arith.divsi' in str(module) and 'arith.remsi' in str(module)
    assert module.verify()


def test_selected_axis_quotient_cannot_be_a_shared_origin(tmp_path):
    rows = analyze(tmp_path, divided_origin('x'), 2, 0)
    assert all(row['action'] == 'Keep' for row in rows)


def tiled_scan_ir():
    return '''module {
  tt.func public @tiled_scan(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %column = tt.get_program_id x : i32
    %row = tt.get_program_id y : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i64
    %c1 = arith.constant 1 : i64
    %rows = arith.extsi %n : i32 to i64
    %r = arith.muli %row, %c4 : i32
    %c = arith.muli %column, %c4 : i32
    %px = tt.make_tensor_ptr %x, [%rows, %c8], [%c8, %c1], [%r, %c] {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    %py = tt.make_tensor_ptr %y, [%rows, %c8], [%c8, %c1], [%r, %c] {order = array<i32: 1, 0>} : <tensor<4x4xf32>>
    %v = tt.load %px {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<4x4xf32>>
    %out = "tt.scan"(%v) <{axis = 0 : i32, reverse = true}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.scan.return %sum : f32
    }) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tt.store %py, %out {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<4x4xf32>>
    tt.return
  }
}'''


def test_shared_program_split_layout_guard_rejects_overflow_and_recovers(tmp_path):
    from triton.backends.ascend.block_batching_guard import check_guard
    text = tiled_scan_ir().replace(
        '%column =', '''%bh = tt.get_program_id z : i32
    %heads = arith.constant 2 : i32
    %width = arith.constant 8 : i32
    %pitch = arith.constant 16 : i32
    %batch = arith.divsi %bh, %heads : i32
    %head = arith.remsi %bh, %heads : i32
    %batch_row = arith.muli %batch, %n : i32
    %batch_base = arith.muli %batch_row, %pitch : i32
    %head_base = arith.muli %head, %width : i32
    %base = arith.addi %batch_base, %head_base : i32
    %rootx = tt.addptr %x, %base : !tt.ptr<f32>, i32
    %rooty = tt.addptr %y, %base : !tt.ptr<f32>, i32
    %stride = arith.constant 16 : i64
    %column =''').replace('tt.make_tensor_ptr %x,', 'tt.make_tensor_ptr %rootx,').replace(
            'tt.make_tensor_ptr %y,', 'tt.make_tensor_ptr %rooty,').replace('[%c8, %c1]', '[%stride, %c1]')
    begin, end = text.index('    %out ='), text.index('    tt.store')
    text = text[:begin] + '    %out = arith.addf %v, %v : tensor<4x4xf32>\n' + text[end:]
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert all(row['rewritten'] for row in json.loads(report['layout_json'])['accesses'])
    guard = dict(json.loads(report['guard_json']), axis=1)
    pointers = [1 << 32, 1 << 45]
    assert check_guard(guard, [*pointers, 64], (2, 16, 4), 2) == (True, 'PROVEN_AT_RUNTIME')
    assert check_guard(guard, [*pointers, 2**31 - 1], (2, 16, 4), 2) == (False, 'INDEX_OVERFLOW')
    assert check_guard(guard, [*pointers, 64], (2, 16, 4), 2) == (True, 'PROVEN_AT_RUNTIME')


@pytest.mark.parametrize('axis', [-1, 0, 1])
@pytest.mark.parametrize('scan_axis', [0, 1])
def test_multidimensional_scan_is_refused_before_layout_rewrite(tmp_path, axis, scan_axis):
    text = tiled_scan_ir().replace('axis = 0', f'axis = {scan_axis}')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, axis)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == 'VV_SCAN_DOMAIN_NOT_SUPPORTED', report
    assert 'block_mapping_owner' not in str(module)
    assert module.verify()


@pytest.mark.parametrize('shape,scan_axis', [('1x4', 1), ('4x1', 0)])
@pytest.mark.parametrize('factor', [2, 3])
def test_singleton_scan_shapes_keep_the_independent_row_contract(tmp_path, shape, scan_axis, factor):
    source = tiled_scan_ir().replace('4x4xf32', f'{shape}xf32').replace('axis = 0', f'axis = {scan_axis}')
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert module.verify()
