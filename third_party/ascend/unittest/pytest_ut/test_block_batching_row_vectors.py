# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""A row-relative vector is the rank-one form of a contiguous pointwise tile."""
import json
import pytest
from triton._C.libtriton import ascend
from triton.backends.ascend.block_batching_guard import RuntimeGuard
from block_batching_contract_ir import parse

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


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
