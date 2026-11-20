# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pointer, mask and row/column proofs; CPU-only."""
import json
import pytest
from triton._C.libtriton import ascend
from block_batching_test_utils import tiled_scan_ir, parse, pointwise, analyze, symbolic_columns_ir, assign_npu  # noqa: F401 - pytest fixtures
from triton.backends.ascend.block_batching_guard import RuntimeGuard

pytestmark = pytest.mark.backend('native')

# Rectangular masks and shared origins


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


# Automatic axis preference


@pytest.mark.parametrize('axis,policy,selected', [(-1, 'auto', 1), (0, 'auto', 0), (1, 'auto', 1),
                                                  (-1, 'semantic_only', 0)])
def test_auto_axis_uses_proven_column_region_and_preserves_explicit_axis(tmp_path, axis, policy, selected):
    module = parse(tmp_path, symbolic_columns_ir())
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, axis, policy)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == selected
    assert module.verify()


def test_auto_column_selection_respects_singleton_grid_hint(tmp_path):
    source = symbolic_columns_ir().replace(
        'module {', 'module attributes {hacc.block_batching_grid = [128 : i32, 1 : i32, 1 : i32]} {')
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, -1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == 0


@pytest.mark.parametrize('grid,factor,axis,selected',
                         [((128, 4, 1), 8, -1, 0),  # Avoid an inactive half-group in every column tile.
                          ((128, 4, 1), 6, -1, 0),  # The complete-row preference also covers nonpowers of two.
                          ((128, 4, 1), 4, -1, 1),  # Keep the established contiguous column path.
                          ((2, 4, 1), 8, -1, 1),  # Partial groups remain supported when neither axis fits F.
                          ((128, 4, 1), 8, 1, 1),  # Explicit requests are unchanged.
                          ])
def test_auto_column_preference_preserves_available_full_groups(tmp_path, grid, factor, axis, selected):
    hint = ', '.join(f'{extent} : i32' for extent in grid)
    source = symbolic_columns_ir().replace('module {', f'module attributes {{hacc.block_batching_grid = [{hint}]}} {{')
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, axis)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == selected
    assert module.verify()


@pytest.mark.parametrize('calculation', [
    '%pf = arith.sitofp %column_p : i32 to f32\n'
    '    %pv = tt.splat %pf : f32 -> tensor<2x4xf32>\n'
    '    %out = arith.addf %v, %pv : tensor<2x4xf32>',
    '%cast = arith.fptosi %v : tensor<2x4xf32> to tensor<2x4xi32>\n'
    '    %out = arith.sitofp %cast : tensor<2x4xi32> to tensor<2x4xf32>',
])
def test_auto_column_selection_retains_old_axis_for_unbridged_dataflow(tmp_path, calculation):
    source = symbolic_columns_ir().replace('%out = arith.addf %v, %one : tensor<2x4xf32>', calculation)
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, -1)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == 0
    assert module.verify()


# Row-relative vectors


def row_vector_ir(axis=1, calculation=None):
    other = (axis + 1) % 3
    calculation = calculation or '%out = arith.mulf %v, %sv : tensor<4xf32>'
    return f'''module {{
  tt.func public @row_vector(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32, %sx: i32, %sy: i32, %scale: f32) {{
    %row = tt.get_program_id {"xyz"[other]} : i32
    %rw = arith.extsi %row : i32 to i64
    %p = tt.get_program_id {"xyz"[axis]} : i32
    %pw = arith.extsi %p : i32 to i64
    %four = arith.constant 4 : i64
    %base = arith.muli %pw, %four : i64
    %bv = tt.splat %base : i64 -> tensor<4xi64>
    %r = tt.make_range {{start = 0 : i32, end = 4 : i32}} : tensor<4xi32>
    %r64 = arith.extsi %r : tensor<4xi32> to tensor<4xi64>
    %idx = arith.addi %bv, %r64 : tensor<4xi64>
    %nw = arith.extsi %n : i32 to i64
    %nv = tt.splat %nw : i64 -> tensor<4xi64>
    %mask = arith.cmpi slt, %idx, %nv : tensor<4xi64>
    %sxw = arith.extsi %sx : i32 to i64
    %syw = arith.extsi %sy : i32 to i64
    %xo = arith.muli %rw, %sxw : i64
    %yo = arith.muli %rw, %syw : i64
    %xr = tt.addptr %x, %xo : !tt.ptr<f32>, i64
    %yr = tt.addptr %y, %yo : !tt.ptr<f32>, i64
    %xp = tt.splat %xr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %yp = tt.splat %yr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %xa = tt.addptr %xp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %ya = tt.addptr %yp, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    %sv = tt.splat %scale : f32 -> tensor<4xf32>
    %v = tt.load %xa, %mask, %zero : tensor<4x!tt.ptr<f32>>
    {calculation}
    tt.store %ya, %out, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }}
}}'''


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("factor", [2, 3, 8])
@pytest.mark.parametrize("requested", [-1, None])
def test_row_relative_vector_uses_contiguous_region(tmp_path, axis, factor, requested):
    module = parse(tmp_path, row_vector_ir(axis))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, axis if requested is None else requested)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['selected_axis'] == axis
    assert module.verify()
    plans = json.loads(report['layout_json'])['accesses']
    assert all(p['rewritten'] and p['action'] == 'PointwiseRegion' for p in plans), plans
    assert f'tensor<{4 * factor}xf32>' in str(module)
    assert 'tt.trans' not in str(module)
    guard = json.loads(report['guard_json'])
    guard['axis'] = axis
    assert len(guard['layout_checks']) == 2
    runtime = RuntimeGuard(guard, factor)
    grid = [1, 1, 1]
    grid[axis], grid[(axis + 1) % 3] = 5, 3
    args = [1 << 32, 1 << 45, 17, 31, 37, .5]
    assert runtime(args, grid)[0]
    # Overlapping output rows must still be rejected, then recover with the
    # same compiled guard and a safe pitch.
    assert not runtime(args[:4] + [2, .5], grid)[0]
    assert runtime(args, grid)[0]


@pytest.mark.parametrize('opcode', ['divsi', 'remsi'])
@pytest.mark.parametrize('split_pid', ['%p', '%row'], ids=['tile-axis', 'row-axis'])
@pytest.mark.parametrize('requested', [2, -1], ids=['rewrite', 'auto-probe'])
def test_residual_program_split_requires_uniform_axis(tmp_path, opcode, split_pid, requested):
    # Axis z tiles the vector; x selects a row. Auto prefers x unless the
    # column-region probe proves z admits a contiguous pointwise rewrite.
    text = row_vector_ir(2).replace(
        '%mask = arith.cmpi slt, %idx, %nv : tensor<4xi64>', f'''%prefix = arith.cmpi slt, %idx, %nv : tensor<4xi64>
    %three = arith.constant 3 : i32
    %split = arith.{opcode} {split_pid}, %three : i32
    %enabled = arith.cmpi slt, %split, %n : i32
    %residual = tt.splat %enabled : i1 -> tensor<4xi1>
    %mask = arith.andi %prefix, %residual : tensor<4xi1>''')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, requested)
    assert module.verify()
    assert report['compile_status'] == 'REWRITTEN', report
    layouts = json.loads(report['layout_json'])['accesses']
    assert len(layouts) == 2
    if split_pid == '%row':
        assert report['selected_axis'] == 2
        assert all(p['rewritten'] and p['action'] == 'PointwiseRegion' for p in layouts), layouts
        assert 'tensor<8xf32>' in str(module)
    else:
        assert report['selected_axis'] == (0 if requested == -1 else 2)
        assert all(not p['rewritten'] for p in layouts), layouts
        assert 'tensor<8xf32>' not in str(module)


def test_row_relative_vector_preserves_explicit_row_axis(tmp_path):
    module = parse(tmp_path, row_vector_ir(1))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 2)
    assert report['selected_axis'] == 2
    assert all(not p['rewritten'] for p in json.loads(report['layout_json'])['accesses'])
    assert module.verify()


@pytest.mark.parametrize('calculation', [
    '%pf = arith.sitofp %p : i32 to f32\n'
    '    %pv = tt.splat %pf : f32 -> tensor<4xf32>\n'
    '    %out = arith.addf %v, %pv : tensor<4xf32>',
    '%iv = arith.fptosi %v : tensor<4xf32> to tensor<4xi32>\n'
    '    %out = arith.sitofp %iv : tensor<4xi32> to tensor<4xf32>',
])
def test_row_relative_vector_does_not_expand_unbridged_dataflow(tmp_path, calculation):
    module = parse(tmp_path, row_vector_ir(1, calculation))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, -1)
    # Axis 1 is encountered first by ordinary selection as well. Validate the
    # actual layout decision rather than merely observing the chosen axis.
    assert not any(p['rewritten'] and p['action'] == 'PointwiseRegion'
                   for p in json.loads(report['layout_json'])['accesses'])
    assert module.verify()
