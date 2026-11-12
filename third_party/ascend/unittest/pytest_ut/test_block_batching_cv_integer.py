# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Integer dot preserves K, exact i32 accumulation and per-lane byte unpacking."""
import re
import json

import pytest
from triton._C.libtriton import ascend
from test_block_batching_cv_spatial import spatial_module
from test_block_batching_cv_loop_fusion import spatial_loop
from test_block_batching_rewrite import parse

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def integer_module(mode='m', loop=False, unpack=False):
    text = (spatial_loop(mode) if loop else spatial_module(mode))
    text = text.replace('f32', 'i32').replace('dense<0.0>', 'dense<0>')
    text = text.replace('%x: !tt.ptr<i32>', '%x: !tt.ptr<i8>')
    text = text.replace('%y: !tt.ptr<i32>', '%y: !tt.ptr<i8>')
    lines = []
    for line in text.splitlines():
        if re.match(r'\s*%(xp|yp|ap|bp|a|b) = ', line):
            line = line.replace('!tt.ptr<i32>', '!tt.ptr<i8>')
            # addptr offsets remain i32; only loaded matrix values change.
            if 'tt.load' in line:
                line = line.replace('tensor<4x4xi32>', 'tensor<4x4xi8>')
        if 'tt.dot ' in line:
            line = line.replace('tensor<4x4xi32> * tensor<4x4xi32>', 'tensor<4x4xi8> * tensor<4x4xi8>')
        lines.append(line)
    text = '\n'.join(lines)
    if unpack:
        text = text.replace(
            '%b = tt.load %bp : tensor<4x4x!tt.ptr<i8>>', '''
      %packed = tt.load %bp : tensor<4x4x!tt.ptr<i8>>
      %wide = arith.extui %packed : tensor<4x4xi8> to tensor<4x4xi32>
      %bits = arith.constant dense<3> : tensor<4x4xi32>
      %low = arith.andi %wide, %bits : tensor<4x4xi32>
      %byte = arith.trunci %low : tensor<4x4xi32> to tensor<4x4xi8>
      %one_byte = arith.constant dense<1> : tensor<4x4xi8>
      %b = arith.subi %byte, %one_byte : tensor<4x4xi8>''')
    return text


@pytest.mark.parametrize('mode', ['m', 'n'])
@pytest.mark.parametrize('factor', [2, 4])
@pytest.mark.parametrize('loop,unpack', [(False, False), (True, False), (True, True)])
def test_integer_spatial_dot_keeps_original_k_and_i32_accumulator(tmp_path, mode, factor, loop, unpack):
    module = parse(tmp_path, integer_module(mode, loop, unpack))
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, 'auto', True, True)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == f'spatial_{mode}_' + ('uniform_control_v1' if loop else 'v1')
    result = str(module)
    assert result.count('tt.dot ') == 1
    dot, = [line for line in result.splitlines() if 'tt.dot ' in line]
    m, n = (4 * factor, 4) if mode == 'm' else (4, 4 * factor)
    assert f'tensor<{m}x4xi8> * tensor<4x{n}xi8> -> tensor<{m}x{n}xi32>' in dot
    assert result.count('scf.for') == int(loop)
    if unpack:
        assert all(op in result for op in ('arith.extui', 'arith.trunci', 'arith.subi'))
    assert module.verify()


@pytest.mark.parametrize('change,reason', [
    (('dense<0> : tensor<4x4xi32>', 'dense<1> : tensor<4x4xi32>'), 'CV_SHARED_ACCUMULATOR_NOT_VALIDATED'),
    (('i8', 'i32'), 'CV_UNSUPPORTED_DOT_TYPE'),
])
def test_integer_rejection_does_not_mutate_input(tmp_path, change, reason):
    module = parse(tmp_path, integer_module().replace(*change))
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True, True)
    assert report['compile_status'] == 'REJECTED', report
    assert report['reason'] == reason, report
    assert str(module) == before


def test_integer_cv_does_not_bypass_overflow_attributes(tmp_path):
    text = integer_module(loop=True, unpack=True).replace('arith.subi %byte, %one_byte :',
                                                          'arith.subi %byte, %one_byte overflow<nsw> :')
    module = parse(tmp_path, text)
    before = str(module)
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0, 'auto', True, True)
    assert report['reason'] == 'UNSUPPORTED_OPERATION_ATTRIBUTE', report
    assert str(module) == before


@pytest.mark.parametrize('factor', [2, 4])
@pytest.mark.parametrize('batch_matmul', [False, True])
def test_integer_independent_pairs_retain_rank_two_dots(tmp_path, factor, batch_matmul):
    text = integer_module().replace(
        '%boff = arith.addi %boff1, %boff2 : tensor<4x4xi32>', '''
    %lane_offset = tt.splat %base : i32 -> tensor<4x4xi32>
    %boff_local = arith.addi %boff1, %boff2 : tensor<4x4xi32>
    %boff = arith.addi %boff_local, %lane_offset : tensor<4x4xi32>''')
    module = parse(tmp_path, text)
    report = ascend.passes.ttir.rewrite_block_batching(module, factor, 0, 'auto', True, batch_matmul)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['cv_lowering'] == 'static_integer_lanes_v1', report
    dots = [line for line in str(module).splitlines() if 'tt.dot ' in line]
    assert len(dots) == factor
    assert all('tensor<4x4xi8> * tensor<4x4xi8> -> tensor<4x4xi32>' in line for line in dots)
    assert json.loads(report['guard_json'])['cv_integer_contract']['dot_rank'] == 2
    assert module.verify()


@pytest.mark.parametrize('proof', [
    None, {},
    dict(version=True, input_bits=8, accumulator_bits=32, lane_semantics='original_program_clone', dot_rank=2)
])
def test_integer_lane_report_requires_owned_contract(proof):
    from test_block_batching_guard import cv_metadata
    from triton.backends.ascend.block_batching_report import validate_compilation_result
    from triton.backends.ascend.block_batching import BatchingRejected
    metadata = cv_metadata()
    metadata['block_batching']['cv_lowering'] = 'static_integer_lanes_v1'
    metadata['block_batching']['guard'] = dict(cv_integer_contract=proof)
    with pytest.raises(BatchingRejected, match='CV_INTEGER_LANE_CONTRACT_MISSING'):
        validate_compilation_result(metadata)


def test_integer_lane_contract_allows_owned_auto_blockify_grid():
    from test_block_batching_guard import cv_metadata
    from triton.backends.ascend.block_batching_report import validate_compilation_result
    metadata = cv_metadata()
    metadata['block_batching'].update(
        cv_lowering='static_integer_lanes_v1', guard=dict(cv_integer_contract=dict(
            version=1, input_bits=8, accumulator_bits=32, lane_semantics='original_program_clone', dot_rank=2)))
    metadata.update(auto_blockify_enabled=True, has_auto_blockify_blacklist_op=False)
    validate_compilation_result(metadata)
    assert metadata['block_batching']['compile_status'] == 'COMPILED_GUARDED'
