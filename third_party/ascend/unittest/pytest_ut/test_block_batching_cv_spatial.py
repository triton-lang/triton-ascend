# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Generic M/N concatenation: axes, transpose, CSE, masks and fail-closed proof."""
import pytest
from triton._C.libtriton import ascend
from test_block_batching_rewrite import parse

pytestmark = pytest.mark.backend('native')


def spatial_module(mode='m', axis=0, transpose=False):
    # A deliberately shared range feeds M, N and K: changing that definition
    # globally would change the reduction and the shared matrix, not just M/N.
    r, n = ('%idx', '%range') if mode == 'm' else ('%range', '%idx')
    lhs = '%rcol' if not transpose else '%kcol'
    rhs = '%krow' if not transpose else '%rrow'
    aload = '%a' if not transpose else '%at'
    trans = '' if not transpose else '''
    %a = tt.trans %at {order = array<i32: 1, 0>} : tensor<4x4xf32> -> tensor<4x4xf32>
'''
    return '''module {
  tt.func public @not_a_kernel_name(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %z: !tt.ptr<f32>, %limit: i32) {
    %p = tt.get_program_id AXIS : i32
    %four = arith.constant 4 : i32
    %base = arith.muli %p, %four : i32
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %basev = tt.splat %base : i32 -> tensor<4xi32>
    %idx = arith.addi %basev, %range : tensor<4xi32>
    %rcol = tt.expand_dims ROW {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %rrow = tt.expand_dims ROW {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %nrow = tt.expand_dims COL {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %krow = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %kcol = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %stride = arith.constant dense<64> : tensor<4x1xi32>
    %aoff0 = arith.muli LHS, %stride : tensor<4x1xi32>
    %aoff1 = tt.broadcast %aoff0 : tensor<4x1xi32> -> tensor<4x4xi32>
    %aoff2 = tt.broadcast RHS : tensor<1x4xi32> -> tensor<4x4xi32>
    %aoff = arith.addi %aoff1, %aoff2 : tensor<4x4xi32>
    %boff0 = arith.muli %kcol, %stride : tensor<4x1xi32>
    %boff1 = tt.broadcast %boff0 : tensor<4x1xi32> -> tensor<4x4xi32>
    %boff2 = tt.broadcast %nrow : tensor<1x4xi32> -> tensor<4x4xi32>
    %boff = arith.addi %boff1, %boff2 : tensor<4x4xi32>
    %coff0 = arith.muli %rcol, %stride : tensor<4x1xi32>
    %coff1 = tt.broadcast %coff0 : tensor<4x1xi32> -> tensor<4x4xi32>
    %coff = arith.addi %coff1, %boff2 : tensor<4x4xi32>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %yp = tt.splat %y : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %zp = tt.splat %z : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %ap = tt.addptr %xp, %aoff : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %bp = tt.addptr %yp, %boff : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %cp = tt.addptr %zp, %coff : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %lim = tt.splat %limit : i32 -> tensor<4xi32>
    %mv = arith.cmpi slt, %idx, %lim : tensor<4xi32>
    %me = tt.expand_dims %mv {axis = MASK_AXIS : i32} : tensor<4xi1> -> MASK_TYPE
    %mask = tt.broadcast %me : MASK_TYPE -> tensor<4x4xi1>
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    ALOAD = tt.load %ap : tensor<4x4x!tt.ptr<f32>>
    %b = tt.load %bp : tensor<4x4x!tt.ptr<f32>>
    TRANS
    %dot = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
    tt.store %cp, %dot, %mask : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}'''.replace('AXIS',
             'xyz'[axis], 1).replace('ROW', r).replace('COL', n).replace('LHS', lhs).replace('RHS', rhs).replace(
                 'ALOAD', aload).replace('TRANS', trans).replace('MASK_AXIS', '1' if mode == 'm' else '0').replace(
                     'MASK_TYPE', 'tensor<4x1xi1>' if mode == 'm' else 'tensor<1x4xi1>')


@pytest.mark.parametrize('mode', ['m', 'n'])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('factor', [2, 4])
@pytest.mark.parametrize('transpose', [False, True])
def test_spatial_dot_and_shared_ranges(tmp_path, mode, axis, factor, transpose):
    mod = parse(tmp_path, spatial_module(mode, axis, transpose))
    report = ascend.passes.ttir.rewrite_block_batching(mod, factor, axis, 'auto', True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_mode'] == 'spatial_concat_v1'
    assert report['cv_lowering'] == f'spatial_{mode}_v1'
    assert report['tail_policy'] == 'full_groups_only'
    assert report['selected_axis'] == axis
    assert report['structured_masks'] == 1
    assert mod.verify()
    text = str(mod)
    assert text.count('tt.dot ') == 1
    assert 'end = 4 : i32' in text and f'end = {4 * factor} : i32' in text
    dot = next(line for line in text.splitlines() if 'tt.dot ' in line)
    shape = f'{4 * factor}x4' if mode == 'm' else f'4x{4 * factor}'
    assert f'-> tensor<{shape}xf32>' in dot
    assert '* tensor<4x4xf32>' in dot if mode == 'm' else ': tensor<4x4xf32> *' in dot


@pytest.mark.parametrize('kind', ['stride', 'pid_escape', 'k_axis', 'local_bias'])
def test_unproved_spatial_merge_keeps_complete_original(tmp_path, kind):
    text = spatial_module()
    if kind == 'stride':
        text = text.replace('%base = arith.muli %p, %four',
                            '%eight = arith.constant 8 : i32\n    %base = arith.muli %p, %eight')
        reason = 'CV_SPATIAL_COORDINATE_NOT_PROVEN'
    elif kind == 'pid_escape':
        text = text.replace('%lim = tt.splat %limit',
                            '%limit2 = arith.addi %limit, %p : i32\n    %lim = tt.splat %limit2')
        reason = 'CV_SPATIAL_PROGRAM_USE_NOT_PROVEN'
    elif kind == 'k_axis':
        text = text.replace(
            '%dot = tt.dot %a,', '''%a_k = tt.trans %a {order = array<i32: 1, 0>} : tensor<4x4xf32> -> tensor<4x4xf32>
    %dot = tt.dot %a_k,''')
        reason = 'CV_SPATIAL_DOT_AXIS_NOT_PROVEN'
    else:
        text = text.replace(
            'tt.store %cp, %dot,', '''%local = arith.sitofp %range : tensor<4xi32> to tensor<4xf32>
    %localcol = tt.expand_dims %local {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
    %localmatrix = tt.broadcast %localcol : tensor<4x1xf32> -> tensor<4x4xf32>
    %biased = arith.addf %dot, %localmatrix : tensor<4x4xf32>
    tt.store %cp, %biased,''')
        reason = 'CV_SPATIAL_UNIFORM_VALUE_NOT_PROVEN'
    mod = parse(tmp_path, text)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'auto', True)
    assert report['reason'] == reason, report
    assert str(mod) == before


@pytest.mark.parametrize('mode', ['m', 'n'])
def test_spatial_final_metadata_contract(mode):
    from triton.backends.ascend.block_batching_report import validate_compilation_result, CV_VALIDATION_POLICY
    from triton.backends.ascend.block_batching import BatchingRejected
    from test_block_batching_guard import cv_metadata
    metadata = cv_metadata()
    metadata['block_batching'].update(cv_mode='spatial_concat_v1', cv_lowering=f'spatial_{mode}_v1')
    validate_compilation_result(metadata)
    assert metadata['block_batching']['validation_policy'] == CV_VALIDATION_POLICY
    metadata['block_batching']['cv_lowering'] = 'rank3_batch_v1'
    with pytest.raises(BatchingRejected, match='CV_LOWERING_CONTRACT_MISSING'):
        validate_compilation_result(metadata)
