# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Uniform regions: spatial recurrences, independent batch MMA and rollback."""
import json

import pytest
from triton._C.libtriton import ascend
from triton.backends.ascend.block_batching_guard import check_guard
from test_block_batching_cv_spatial import spatial_module
from test_block_batching_cv_full_tile import transposed_load_dot
from test_block_batching_rewrite import parse

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def spatial_loop(mode='m', axis=0, branch=False):
    text = spatial_module(mode, axis).replace('%limit: i32)', '%limit: i32, %trips: i32, %flag: i32)')
    begin = text.index('    %a = tt.load')
    end = text.index('    tt.store')
    body = text[begin:end].replace('%dot = tt.dot', '%next = tt.dot').replace('%b, %zero,', '%b, %carry,')
    if branch:
        body = '''
      %choose = arith.cmpi sgt, %flag, %loop_zero : i32
      %picked = scf.if %choose -> (tensor<4x4xf32>) {
        ''' + body + '''
        scf.yield %next : tensor<4x4xf32>
      } else {
        scf.yield %carry : tensor<4x4xf32>
      }
'''
    result = '%picked' if branch else '%next'
    return text[:begin] + '''
    %loop_zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %dot = scf.for %iv = %loop_zero to %trips step %one iter_args(%carry = %zero) -> (tensor<4x4xf32>) : i32 {
''' + body + f'    scf.yield {result} : tensor<4x4xf32>\n    }}\n' + text[end:]


@pytest.mark.parametrize('mode', ['m', 'n'])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('factor', [2, 4])
@pytest.mark.parametrize('branch', [False, True])
def test_loop_spatial_fusion_preserves_one_dot_and_one_recurrence(tmp_path, mode, axis, factor, branch):
    module = parse(tmp_path, spatial_loop(mode, axis, branch))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, axis, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == f'spatial_{mode}_uniform_control_v1', report
    text = str(module)
    assert text.count('tt.dot ') == 1 and text.count('scf.for') == 1
    assert text.count('scf.if') == int(branch)  # no new accumulator-transfer boundary
    shape = f'{4*factor}x4' if mode == 'm' else f'4x{4*factor}'
    dot, = [line for line in text.splitlines() if 'tt.dot ' in line]
    assert f'-> tensor<{shape}xf32>' in dot
    assert text.count('tt.get_program_id') == 1
    assert module.verify()
    guard = json.loads(report['guard_json'])
    contract = guard['cv_control_flow_contract']
    assert contract['lane_semantics'] == 'spatial_concatenation'
    assert contract['for_count'] == 1 and contract['if_count'] == int(branch)
    grid = [1, 1, 1]
    grid[axis] = factor
    for trips in [0, 1, 3]:
        args = [1 << 30, 1 << 40, 1 << 45, factor * 4, trips, 1]
        checked = check_guard(guard, args, grid, factor)
        assert checked[0], checked
    args[2] = args[0]
    args[4] = 3
    assert not check_guard(guard, args, grid, factor)[0]


def independent_loop(masked=False):
    text = transposed_load_dot().replace('%c: !tt.ptr<f32>)', '%c: !tt.ptr<f32>, %trips: i32, %limit: i32)')
    begin = text.index('    %av = tt.load')
    end = text.index('    %row = tt.broadcast')
    body = text[begin:end]
    body = body.replace('    %zero = arith.constant dense<0.0> : tensor<4x4xf32>\n', '')
    body = body.replace('%dot = tt.dot', '%next = tt.dot').replace('%bv, %zero :', '%bv, %carry :')
    if masked:
        body = body.replace(
            '%av = tt.load %ptr :', '''
      %bound = tt.splat %limit : i32 -> tensor<4x8xi32>
      %mask = arith.cmpi slt, %kk, %bound : tensor<4x8xi32>
      %pad = arith.constant dense<0.0> : tensor<4x8xf32>
      %av = tt.load %ptr, %mask, %pad :''')
    return text[:begin] + '''
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %z = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %dot = scf.for %iv = %z to %trips step %one iter_args(%carry = %zero) -> (tensor<4x4xf32>) : i32 {
''' + body + '    scf.yield %next : tensor<4x4xf32>\n    }\n' + text[end:]


def periodic_head_loop(period, axis):
    text = independent_loop().replace('tt.get_program_id x', f'tt.get_program_id {"xyz"[axis]}')
    return text.replace(
        '%base = arith.muli %p, %c32 : i32', f'''
    %heads = arith.constant {period} : i32
    %batch_stride = arith.constant {period * 64} : i32
    %batch = arith.divsi %p, %heads : i32
    %head = arith.remsi %p, %heads : i32
    %bb = arith.muli %batch, %batch_stride : i32
    %hh = arith.muli %head, %c32 : i32
    %base = arith.addi %bb, %hh : i32''')


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('period,factor', [(4, 2), (4, 4), (6, 2), (8, 4)])
def test_aligned_periodic_program_addresses_keep_scalar_quotient(tmp_path, axis, period, factor):
    module = parse(tmp_path, periodic_head_loop(period, axis))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, axis, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'rank3_uniform_control_batch_v1', report
    text = str(module)
    divisions = [line for line in text.splitlines() if 'arith.divsi ' in line or 'arith.remsi ' in line]
    assert len(divisions) == 2 and all('tensor<' not in line for line in divisions)
    assert text.count('tt.dot ') == 1 and module.verify()
    guard = json.loads(report['guard_json'])
    guard['require_same_fractal_bounds'] = True
    grid = [1, 1, 1]
    grid[axis] = period * 2
    assert check_guard(guard, [1 << 30, 1 << 40, 1 << 45, 3, 8], grid, factor)[0]


@pytest.mark.parametrize('period,factor', [(3, 2), (6, 4), (2, 4)])
def test_period_crossing_group_falls_back_to_independent_regions(tmp_path, period, factor):
    module = parse(tmp_path, periodic_head_loop(period, 0))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'static_uniform_control_lanes_v1', report
    assert json.loads(report['guard_json'])['cv_loop_fusion_rejection'] == 'CV_BATCH_QUOTIENT_GROUP_CROSSES_PERIOD'
    assert str(module).count('tt.dot ') == factor and module.verify()


def test_period_alignment_does_not_bypass_cv_factor_capability(tmp_path):
    module = parse(tmp_path, periodic_head_loop(6, 0))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0, 'auto', True, True)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == 'CV_FACTOR_REQUIRES_POWER_OF_TWO'
    assert str(module) == before


def independent_pointer_loop(delta=32):
    text = independent_loop()
    # The access proof accepts invariant deltas defined before the region,
    # matching canonical TTIR. An in-region delta is deliberately fail-closed.
    text = text.replace('%dot = scf.for', f'''%advance = arith.constant dense<{delta}> : tensor<4x8xi32>
    %loop:2 = scf.for''')
    text = text.replace('iter_args(%carry = %zero) -> (tensor<4x4xf32>)',
                        'iter_args(%carry = %zero, %input = %ptr) -> (tensor<4x4xf32>, tensor<4x8x!tt.ptr<f32>>)')
    text = text.replace('%av = tt.load %ptr', '%av = tt.load %input')
    text = text.replace(
        'scf.yield %next : tensor<4x4xf32>', f'''
      %nextptr = tt.addptr %input, %advance : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
      scf.yield %next, %nextptr : tensor<4x4xf32>, tensor<4x8x!tt.ptr<f32>>''')
    return text.replace('tt.store %cptr, %dot', 'tt.store %cptr, %loop#0')


@pytest.mark.parametrize('delta', [32, -32, 1073741824])
def test_batch_pointer_recurrence_rebuilds_addresses_in_i64(tmp_path, delta):
    module = parse(tmp_path, independent_pointer_loop(delta))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'rank3_uniform_control_batch_v1', report
    text = str(module)
    assert 'arith.divui ' in text
    assert 'tensor<2x4x8xi64>' in text
    assert text.count('tt.dot ') == 1 and module.verify()
    # There is no extra all-true mask or pointer-provenance runtime guard.
    guard = json.loads(report['guard_json'])
    assert 'require_full_masks' not in guard


@pytest.mark.parametrize('factor', [2, 4])
@pytest.mark.parametrize('masked', [False, True])
def test_loop_batch_preserves_independent_accumulators_and_transposed_load(tmp_path, factor, masked):
    module = parse(tmp_path, independent_loop(masked))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'rank3_uniform_control_batch_v1', report
    text = str(module)
    assert text.count('scf.for') == 1 and text.count('tt.dot ') == 1
    assert 'scf.if' not in text
    assert f'tensor<{factor}x4x8xf32> * tensor<{factor}x8x4xf32>' in text
    assert 'tt.cv_batch_full_mkn' in text
    assert module.verify()
    guard = json.loads(report['guard_json'])
    guard['require_same_fractal_bounds'] = True
    args = [1 << 30, 1 << 40, 1 << 45, 3, 8]
    checked = check_guard(guard, args, [factor, 1, 1], factor)
    assert checked[0], checked
    if masked:
        args[-1] = 7
        assert check_guard(guard, args, [factor, 1, 1], factor) == (False, 'CV_BATCH_REDUCTION_MASK_NOT_PROVEN')


def test_unprovable_spatial_local_bias_rolls_back_to_original_lane_regions(tmp_path):
    text = spatial_loop().replace(
        'tt.store %cp, %dot,', '''
    %local = arith.sitofp %range : tensor<4xi32> to tensor<4xf32>
    %localcol = tt.expand_dims %local {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
    %localmatrix = tt.broadcast %localcol : tensor<4x1xf32> -> tensor<4x4xf32>
    %biased = arith.addf %dot, %localmatrix : tensor<4x4xf32>
    tt.store %cp, %biased,''')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'static_uniform_control_lanes_v1'
    guard = json.loads(report['guard_json'])
    assert guard['cv_loop_fusion_rejection'] == 'CV_SPATIAL_UNIFORM_VALUE_NOT_PROVEN'
    assert str(module).count('scf.for') == 2 and str(module).count('tt.dot ') == 2
    assert 'tensor<8x4xf32>' not in str(module)
    assert module.verify()


def test_shared_load_keeps_both_operands_reduction_axis_obligations(tmp_path):
    # A row tail is harmless to the original A @ A.T reduction (axis 1).
    text = independent_loop(masked=True).replace('slt, %kk, %bound', 'slt, %mm, %bound')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    guard = json.loads(report['guard_json'])
    guard['require_same_fractal_bounds'] = True
    args = [1 << 30, 1 << 40, 1 << 45, 3, 3]
    checked = check_guard(guard, args, [2, 1, 1], 2)
    assert checked[0], checked
    # Exercise the guard contract for a shared source whose second use also
    # reduces original axis 0: both obligations must be retained, in any order.
    inputs = guard['cv_mask_contract']['inputs']
    for item in inputs:
        if item['operand'] == 1:
            item['axes'] = [0, 1]
    expected = (False, 'CV_BATCH_REDUCTION_MASK_NOT_PROVEN')
    assert check_guard(guard, args, [2, 1, 1], 2) == expected
    inputs.reverse()
    assert check_guard(guard, args, [2, 1, 1], 2) == expected


@pytest.mark.parametrize('kind', ['spatial', 'batch'])
def test_invariant_dot_yield_retains_explicit_zero_trip_result(tmp_path, kind):
    text = spatial_loop() if kind == 'spatial' else independent_loop()
    text = text.replace('%b, %carry,', '%b, %zero,').replace('%bv, %carry :', '%bv, %zero :')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert str(module).count('scf.if') == 1
    assert str(module).count('scf.for') == 1
    assert module.verify()


@pytest.mark.parametrize('negative', [False, True])
def test_canonical_nonnegative_mask_keeps_full_domain_proof(tmp_path, negative):
    text = independent_loop(masked=True)
    coord = '%kk'
    extra = ''
    if negative:
        coord = '%shifted'
        extra = '''
      %negative_one = arith.constant dense<-1> : tensor<4x8xi32>
      %shifted = arith.addi %kk, %negative_one : tensor<4x8xi32>
'''
    text = text.replace(
        '%mask = arith.cmpi slt, %kk, %bound : tensor<4x8xi32>', extra + f'''
      %integer_zero = arith.constant dense<0> : tensor<4x8xi32>
      %nonnegative = arith.cmpi sge, {coord}, %integer_zero : tensor<4x8xi32>
      %upper = arith.cmpi slt, %kk, %bound : tensor<4x8xi32>
      %mask = arith.andi %nonnegative, %upper : tensor<4x8xi1>
''')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'rank3_uniform_control_batch_v1', report
    assert 'arith.cmpi sge' in str(module)  # Never erase the device predicate.
    guard = json.loads(report['guard_json'])
    guard['require_same_fractal_bounds'] = True
    checked = check_guard(guard, [1 << 30, 1 << 40, 1 << 45, 3, 8], [2, 1, 1], 2)
    if negative:
        assert checked == (False, 'CV_BATCH_PADDING_NOT_PROVEN')
    else:
        assert checked[0], checked
    assert module.verify()


def test_batch_loop_allows_proved_auxiliary_vector_access(tmp_path):
    text = independent_loop().replace(
        '    %av = tt.load', '''
    %aux_base = tt.splat %b : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %aux_ptr = tt.addptr %aux_base, %r8 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %aux = tt.load %aux_ptr : tensor<8x!tt.ptr<f32>>
    %av = tt.load''')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'rank3_uniform_control_batch_v1', report
    guard = json.loads(report['guard_json'])
    guard['require_same_fractal_bounds'] = True
    checked = check_guard(guard, [1 << 30, 1 << 40, 1 << 45, 3, 8], [2, 1, 1], 2)
    assert checked[0], checked


@pytest.mark.parametrize('lowering,mode,semantics,rank', [
    ('spatial_m_uniform_control_v1', 'spatial_concat_v1', 'spatial_concatenation', 2),
    ('spatial_n_uniform_control_v1', 'spatial_concat_v1', 'spatial_concatenation', 2),
    ('rank3_uniform_control_batch_v1', 'independent_batch_v1', 'independent_batch', 3),
])
def test_loop_lowering_contract_requires_matching_region_and_memory_proofs(lowering, mode, semantics, rank):
    from triton.backends.ascend.block_batching import BatchingRejected
    from triton.backends.ascend.block_batching_report import validate_compilation_result
    from test_block_batching_guard import cv_metadata
    metadata = cv_metadata()
    report = metadata['block_batching']
    report.update(cv_mode=mode, cv_lowering=lowering, guard={})
    if rank == 3:
        metadata['enable_hivm_batch_matmul'] = True
        report['cv_memory_lowering'] = 'full_reduction_guard_v3'
        report['guard'].update(require_same_fractal_bounds=True,
                               cv_mask_contract=dict(version=1, inputs=[{}, {}], outputs=[{}]))
    report['guard']['cv_control_flow_contract'] = dict(version=1, lane_semantics=semantics, uniform_bounds=True,
                                                       dot_rank=rank, for_count=1, if_count=0)
    metadata.update(auto_blockify_enabled=True, has_auto_blockify_blacklist_op=False)
    validate_compilation_result(metadata)
    report['guard']['cv_control_flow_contract']['uniform_bounds'] = False
    with pytest.raises(BatchingRejected, match='CV_UNIFORM_CONTROL_CONTRACT_MISSING'):
        validate_compilation_result(metadata)
