# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Native CV legality, lane representation, and unchanged rejection tests."""
import json
import pytest
from triton._C.libtriton import ascend
from test_block_batching_rewrite import parse, pointwise

pytestmark = pytest.mark.backend("native")


def dot_module():
    return pointwise(0).replace('tt.load %xa, %mask, %zero',
                                'tt.load %xa').replace('tt.store %ya, %result, %mask', 'tt.store %ya, %result').replace(
                                    '%result = arith.addf %v, %gv : tensor<4xf32>', '''
    %col = tt.expand_dims %v {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
    %a = tt.broadcast %col : tensor<4x1xf32> -> tensor<4x4xf32>
    %bmat = arith.addf %a, %a : tensor<4x4xf32>
    %acc = arith.constant dense<0.0> : tensor<4x4xf32>
    %d = tt.dot %a, %bmat, %acc, inputPrecision = ieee : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
    %result = "tt.reduce"(%d) <{axis = 1 : i32}> ({
      ^bb0(%left: f32, %right: f32):
        %s = arith.addf %left, %right : f32
        tt.reduce.return %s : f32
    }) : (tensor<4x4xf32>) -> tensor<4xf32>
''')


@pytest.mark.parametrize('factor', [2, 4, 8])
@pytest.mark.parametrize('precision', ['ieee', 'tf32x3'])
def test_dot_preserves_independent_lane_and_precision(tmp_path, factor, precision):
    mod = parse(tmp_path, dot_module().replace('inputPrecision = ieee', f'inputPrecision = {precision}'))
    report = ascend.passes.ttir.rewrite_block_batching(mod, factor, 0, 'semantic_only', True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_mode'] == 'independent_batch_v1'
    assert report['tail_policy'] == 'full_groups_only'
    assert report['operations']['tt.dot'] == 1
    assert mod.verify()
    text = str(mod)
    assert f'tensor<{factor}x4x4xf32>' in text
    dot = next(line for line in text.splitlines() if 'tt.dot ' in line)
    if precision == 'ieee':
        # The assembly printer elides the default IEEE enum.
        assert 'inputPrecision' not in dot
    else:
        assert f'inputPrecision = {precision}' in dot
    assert 'axis = 2 : i32' in text  # original row reduction, never reduce F


@pytest.mark.parametrize('cv,factor,reason', [(False, 2, 'CV_OPT_IN_REQUIRED'),
                                              (True, 3, 'CV_FACTOR_REQUIRES_POWER_OF_TWO')])
def test_cv_rejection_is_transactional(tmp_path, cv, factor, reason):
    mod = parse(tmp_path, dot_module())
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, factor, 0, 'auto', cv)
    assert report['reason'] == reason, report
    assert str(mod) == before


def test_cv_requires_dot_and_strict_opt_in(tmp_path):
    mod = parse(tmp_path, pointwise(0))
    assert ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'auto', True)['reason'] == 'CV_DOT_REQUIRED'
    with pytest.raises(ValueError, match='cv must be bool'):
        ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'auto', 1)


def test_cv_analysis_does_not_mutate(tmp_path):
    mod = parse(tmp_path, dot_module())
    before = str(mod)
    report = ascend.passes.ttir.analyze_block_batching(mod, 2, 0, True)
    assert report['compile_status'] == 'ANALYZED', report
    assert str(mod) == before


@pytest.mark.parametrize('mode,lowering', [
    ('independent_batch_v1', 'rank3_masked_batch_v1'),
    ('spatial_concat_v1', 'spatial_m_v1'),
    ('spatial_concat_v1', 'spatial_n_v1'),
    ('independent_batch_v1', 'static_uniform_control_lanes_v1'),
    ('spatial_concat_v1', 'spatial_m_uniform_control_v1'),
    ('spatial_concat_v1', 'spatial_n_uniform_control_v1'),
    ('independent_batch_v1', 'rank3_uniform_control_batch_v1'),
])
def test_cv_can_share_grid_coverage_with_auto_blockify(monkeypatch, mode, lowering):
    from types import SimpleNamespace
    from triton.backends.ascend import compiler
    from test_block_batching_guard import cv_metadata
    monkeypatch.setattr(compiler, '_is_auto_map_parallel_blocks_enabled', lambda: True)
    metadata = cv_metadata()
    metadata['block_batching'].update(cv_mode=mode, cv_lowering=lowering)
    metadata['has_auto_blockify_blacklist_op'] = False
    compiler._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))
    assert metadata['auto_blockify_enabled'] is True
    metadata['has_auto_blockify_blacklist_op'] = True
    compiler._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))
    assert metadata['auto_blockify_enabled'] is False
    metadata['has_auto_blockify_blacklist_op'] = False
    metadata['mix_mode'] = 'aic'
    compiler._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))
    assert metadata['auto_blockify_enabled'] is False
    metadata['mix_mode'] = 'mix'
    metadata['block_batching']['cv_lowering'] = 'rank3_batch_v1'
    compiler._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))
    assert metadata['auto_blockify_enabled'] is False
    metadata['block_batching']['cv_mode'] = 'disabled'
    compiler._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))
    assert metadata['auto_blockify_enabled'] is True  # VV policy is unchanged


def test_program_quotient_remainder_have_owned_guard_expressions(tmp_path):
    text = dot_module().replace(
        '%base = arith.muli %p, %b : i32', '''
    %two = arith.constant 2 : i32
    %eight = arith.constant 8 : i32
    %q = arith.divsi %p, %two : i32
    %r = arith.remsi %p, %two : i32
    %hi = arith.muli %q, %eight : i32
    %lo = arith.muli %r, %b : i32
    %base = arith.addi %hi, %lo : i32
''')
    mod = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'semantic_only', True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert 'pid_div' in report['guard_json'] and 'pid_rem' in report['guard_json']
    assert mod.verify()


def test_existing_batched_dot_is_rejected_without_mutation(tmp_path):
    text = dot_module().replace(
        '    %result = "tt.reduce"', '''
    %batch = tt.expand_dims %d {axis = 0 : i32} : tensor<4x4xf32> -> tensor<1x4x4xf32>
    %extra = tt.dot %batch, %batch, %batch : tensor<1x4x4xf32> * tensor<1x4x4xf32> -> tensor<1x4x4xf32>
    %result = "tt.reduce"''')
    mod = parse(tmp_path, text)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'semantic_only', True)
    assert report['reason'] == 'CV_REQUIRES_RANK_TWO_DOT'
    assert str(mod) == before


def test_cv_uniform_control_flow_retains_independent_rank_two_lanes(tmp_path):
    # Uniform control must not reach the unvalidated rank-three loop lowering.
    text = dot_module().replace(
        '    %result = "tt.reduce"', '''
    %zero_i = arith.constant 0 : i32
    %condition = arith.cmpi sgt, %n, %zero_i : i32
    scf.if %condition {
      scf.yield
    }
    %result = "tt.reduce"''')
    mod = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'semantic_only', True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'static_uniform_control_lanes_v1'
    assert str(mod).count('scf.if') == 2
    assert str(mod).count('tt.dot ') == 2
    assert 'tensor<2x4x4xf32>' not in str(mod)
    assert mod.verify()


def test_cv_shared_matrix_requires_a_separately_validated_lowering(tmp_path):
    text = dot_module().replace('%bmat = arith.addf %a, %a : tensor<4x4xf32>',
                                '%bmat = arith.constant dense<1.0> : tensor<4x4xf32>')
    mod = parse(tmp_path, text)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'semantic_only', True)
    assert report['reason'] == 'CV_SPATIAL_OPERATION_NOT_SUPPORTED'
    assert str(mod) == before


@pytest.mark.parametrize('factor', [2, 4, 8])
@pytest.mark.parametrize('effect', ['load', 'store'])
def test_cv_structured_mask_preserves_rank_two_lanes(tmp_path, effect, factor):
    text = dot_module()
    if effect == 'load':
        text = text.replace('tt.load %xa :', 'tt.load %xa, %mask, %zero :')
    else:
        text = text.replace('tt.store %ya, %result :', 'tt.store %ya, %result, %mask :')
    mod = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(mod, factor, 0, 'semantic_only', True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'static_masked_lanes_v1'
    assert report['structured_masks'] == 1
    assert isinstance(json.loads(report['layout_json']), dict)
    assert report['lane_width'] == factor
    assert mod.verify()
    text = str(mod)
    assert text.count('tt.dot ') == factor
    assert text.count('tt.load ') == factor and text.count('tt.store ') == factor
    assert text.count('tt.get_program_id') == 1
    assert 'tt.get_num_programs' not in text
    assert f'tensor<{factor}x4x4xf32>' not in text
    assert text.count('tt.return') == 1
    assert mod.get_function('arbitrary_name').get_num_args() == 6


@pytest.mark.parametrize('kind', ['data', 'nonzero', 'undefined'])
def test_cv_unsafe_mask_contract_is_transactional(tmp_path, kind):
    text = dot_module()
    if kind == 'data':
        text = text.replace(
            'tt.store %ya, %result :', '''
    %datamask = arith.cmpf ogt, %v, %zero : tensor<4xf32>
    tt.store %ya, %result, %datamask :''')
        reason = 'CV_UNSTRUCTURED_MEMORY_MASK'
    else:
        text = text.replace('tt.load %xa :', 'tt.load %xa, %mask' + (', %zero' if kind == 'nonzero' else '') + ' :')
        if kind == 'nonzero':
            text = text.replace('%zero = arith.constant dense<0.0>', '%zero = arith.constant dense<1.0>')
        # The common memory verifier rejects undefined masked values earlier
        # than the CV-specific explicit-zero check.
        reason = 'MASKED_LOAD_REQUIRES_OTHER' if kind == 'undefined' else 'CV_MASKED_LOAD_REQUIRES_ZERO_OTHER'
    mod = parse(tmp_path, text)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'semantic_only', True)
    assert report['reason'] == reason, report
    assert str(mod) == before


def test_cv_constant_full_tile_mask_is_allowed(tmp_path):
    text = dot_module().replace(
        '    %v = tt.load %xa :', '''
    %all_true = arith.constant dense<true> : tensor<4xi1>
    %v = tt.load %xa, %all_true, %zero :''')
    mod = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'semantic_only', True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert mod.verify()


def test_cv_shared_nonzero_accumulator_selects_unchanged_baseline(tmp_path):
    text = dot_module().replace('%acc = arith.constant dense<0.0>', '%acc = arith.constant dense<1.0>')
    mod = parse(tmp_path, text)
    before = str(mod)
    report = ascend.passes.ttir.rewrite_block_batching(mod, 2, 0, 'semantic_only', True)
    assert report['reason'] == 'CV_SHARED_ACCUMULATOR_NOT_VALIDATED'
    assert str(mod) == before
