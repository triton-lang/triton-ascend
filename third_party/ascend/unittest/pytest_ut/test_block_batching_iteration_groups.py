"""Sequential iteration groups reuse the proved program-axis loop schedule."""
import json
import pytest
from triton._C.libtriton import ascend
from test_block_batching_rewrite import parse

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def recurrent_ir():
    return '''module {
  tt.func public @recurrence(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %p = tt.get_program_id x : i32
    %g = tt.get_num_programs x : i32
    %c = arith.constant 4 : i32
    %lo = arith.muli %p, %c : i32
    %step = arith.muli %g, %c : i32
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %nv = tt.splat %n : i32 -> tensor<4xi32>
    %z = arith.constant dense<0.0> : tensor<4xf32>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %sum = scf.for %iv = %lo to %n step %step iter_args(%acc = %z) -> (tensor<4xf32>) : i32 {
      %ivv = tt.splat %iv : i32 -> tensor<4xi32>
      %ix = arith.addi %ivv, %r : tensor<4xi32>
      %mask = arith.cmpi slt, %ix, %nv : tensor<4xi32>
      %xp = tt.addptr %xs, %ix : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %v = tt.load %xp, %mask, %z : tensor<4x!tt.ptr<f32>>
      %next = arith.addf %acc, %v : tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    %pv = tt.splat %lo : i32 -> tensor<4xi32>
    %oi = arith.addi %pv, %r : tensor<4xi32>
    %os = tt.splat %o : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %op = tt.addptr %os, %oi : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %op, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}'''


@pytest.mark.parametrize('factor', [2, 3, 6])
def test_recurrence_groups_preserve_grid_and_original_tile(tmp_path, factor):
    module = parse(tmp_path, recurrent_ir())
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report.get('vv_lowering') == 'sequential_iterations_v1'
    spec = json.loads(report['guard_json'])
    assert spec['loop_schedule']['kind'] == 'sequential_iterations_v1'
    assert spec['loop_schedule']['factor'] == factor
    assert spec['loop_schedule']['grid_policy'] == 'preserve'
    assert 'hacc.program_grid_transforms' not in str(module)
    assert f'tensor<{factor}x4xf32>' not in str(module)
    assert 'tensor<4xf32>' in str(module)
    assert module.verify()


def test_semantic_only_keeps_original_program_folding(tmp_path):
    module = parse(tmp_path, recurrent_ir())
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'semantic_only')
    assert report.get('vv_lowering') != 'sequential_iterations_v1'


def generated_guard(tmp_path, factor=2):
    module = parse(tmp_path, recurrent_ir())
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    return json.loads(report['guard_json'])


def test_original_byte_proof_and_loop_arithmetic_both_remain_required(tmp_path):
    from test_block_batching_guard_v2 import module as guard
    spec = generated_guard(tmp_path)
    # G=64, R=4, N=513: the last program also has two iterations.
    assert guard.check_guard(spec, [1 << 32, 1 << 40, 513], (64, 1, 1), 2)[0]
    assert guard.check_guard(spec, [1 << 32, 1 << 32, 513], (64, 1, 1), 2) == (False, 'WRITE_ALIAS')
    assert guard.check_guard(spec, [1 << 32, 1 << 40, 256], (64, 1, 1), 2) == (False, 'LOOP_GROUP_NOT_FORMED')
    assert guard.check_guard(spec, [1 << 32, 1 << 40, 0], (64, 1, 1), 2) == (False, 'LOOP_GROUP_NOT_FORMED')
    assert guard.check_guard(spec, [1 << 32, 1 << 40, (1 << 31) - 1], (64, 1, 1), 2) == (False, 'INDEX_OVERFLOW')


def test_iteration_guard_cache_tracks_bounds_and_program_count(tmp_path):
    from test_block_batching_guard_v2 import module as guard
    cached = guard.RuntimeGuard(generated_guard(tmp_path), 2)
    args = [1 << 32, 1 << 40, 513]
    assert cached(args, (64, 1, 1))[0]
    args[2] = 256
    assert cached(args, (64, 1, 1)) == (False, 'LOOP_GROUP_NOT_FORMED')
    args[2] = 513
    assert cached(args, (64, 1, 1))[0]
    args[2] = (1 << 31) - 1
    assert cached(args, (64, 1, 1)) == (False, 'INDEX_OVERFLOW')
    args[2] = 513
    assert cached(args, (64, 1, 1))[0]
    assert cached(args, (256, 1, 1)) == (False, 'LOOP_GROUP_NOT_FORMED')


@pytest.mark.parametrize('mutation', ['factor', 'grid', 'loop'])
def test_schedule_cannot_bypass_original_fact_binding(tmp_path, mutation):
    from test_block_batching_guard_v2 import module as guard
    spec = generated_guard(tmp_path)
    schedule = spec['loop_schedule']
    if mutation == 'factor':
        schedule['factor'] = 3
    elif mutation == 'grid':
        schedule['grid_policy'] = 'ceil_div'
    else:
        schedule['loop']['upper'] = ['const', 513, 32]
    assert guard.check_guard(spec, [1 << 32, 1 << 40, 513], (64, 1, 1), 2) == (False, 'INVALID_LOOP_SCHEDULE')


@pytest.mark.parametrize('mutation', ['mapping', 'mode', 'factor'])
def test_final_compilation_checks_grid_and_schedule_together(tmp_path, mutation):
    from triton.backends.ascend.block_batching import BatchingRejected
    from triton.backends.ascend.block_batching_report import validate_vv_loop_contract
    report = dict(vv_lowering='sequential_iterations_v1', cv_mode='disabled', lane_width=2,
                  guard=generated_guard(tmp_path))
    metadata = dict(block_batching=report, program_grid_transforms=None, program_grid_mapping_applied=False,
                    row_coalescing_applied=False, ptsm_cap_authorized=False, coalesce_factor=1)
    assert validate_vv_loop_contract(metadata)
    if mutation == 'mapping':
        metadata['program_grid_mapping_applied'] = True
    elif mutation == 'mode':
        report['vv_lowering'] = 'vector_lanes_v1'
    else:
        report['lane_width'] = 3
    with pytest.raises(BatchingRejected):
        validate_vv_loop_contract(metadata)


def row_stream_ir(strided=False, conditional=False):
    body = '''
      %ivv = tt.splat %iv : i32 -> tensor<8xi32>
      %col = arith.addi %ivv, %range : tensor<8xi32>
      %mask = arith.cmpi slt, %col, %nv : tensor<8xi32>
      %xi = arith.addi %xbv, %col : tensor<8xi32>
      %oi = arith.addi %obv, %col : tensor<8xi32>
      %xp = tt.addptr %xs, %xi : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      %op = tt.addptr %os, %oi : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      %v = tt.load %xp, %mask, %zeros : tensor<8x!tt.ptr<f32>>
      %result = arith.subf %zeros, %v : tensor<8xf32>
      tt.store %op, %result, %mask : tensor<8x!tt.ptr<f32>>
'''
    if strided:
        body = body.replace('%xi = arith.addi %xbv, %col',
                            '%twocol = arith.addi %col, %col : tensor<8xi32>\n      %xi = arith.addi %xbv, %twocol')
    if conditional:
        body = '      scf.if %positive {\n' + body + '      }\n'
    return '''module {
  tt.func public @row_stream(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32, %sx: i32, %so: i32) {
    %p = tt.get_program_id x : i32
    %zero = arith.constant 0 : i32
    %step = arith.constant 8 : i32
    %positive = arith.cmpi sgt, %n, %zero : i32
    %zeros = arith.constant dense<0.0> : tensor<8xf32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %xb = arith.muli %p, %sx : i32
    %ob = arith.muli %p, %so : i32
    %xbv = tt.splat %xb : i32 -> tensor<8xi32>
    %obv = tt.splat %ob : i32 -> tensor<8xi32>
    %nv = tt.splat %n : i32 -> tensor<8xi32>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %os = tt.splat %o : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    scf.for %iv = %zero to %n step %step : i32 {
''' + body + '''    }
    tt.return
  }
}'''


@pytest.mark.parametrize('factor', [2, 3, 6])
def test_unbridged_uniform_pointwise_loop_requires_memory_view(tmp_path, factor):
    module = parse(tmp_path, row_stream_ir())
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == 'LOOP_POINTWISE_MEMORY_VIEW_REQUIRED'
    assert str(module) == before


def test_uniform_recurrence_reuses_coordinate_and_carry_proof(tmp_path):
    source = recurrent_ir().replace('%sum = scf.for %iv = %lo to %n step %step',
                                    '%zero = arith.constant 0 : i32\n    %sum = scf.for %iv = %zero to %n step %c')
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['vv_lowering'] == 'sequential_iterations_v1'
    assert module.verify()


def test_loop_presence_alone_does_not_trigger_memory_view_refusal(tmp_path):
    module = parse(tmp_path, row_stream_ir(conditional=True))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['vv_lowering'] == 'vector_lanes_v1'
    assert module.verify()
