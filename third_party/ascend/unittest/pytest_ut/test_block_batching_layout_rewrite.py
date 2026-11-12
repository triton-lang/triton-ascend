# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import json
import pytest
from triton._C.libtriton import ascend
from triton.backends.ascend.compiler import NPUOptions
from triton.backends.ascend.block_batching import (BatchRequest, LayoutTargetRejected, _compile_layout_candidate)
from test_block_batching_rewrite import parse, pointwise
from block_batching_byte_extern_cases import case_ir

pytestmark = pytest.mark.backend("native")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize("factor", [2, 3, 6])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_linear_memory_view_and_semantic_control(tmp_path, factor, axis):
    source = pointwise(axis)
    auto = parse(tmp_path, source)
    semantic = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(auto, factor, axis, "auto")
    control = ascend.passes.ttir.rewrite_block_batching(semantic, factor, axis, "semantic_only")
    assert report['compile_status'] == control['compile_status'] == 'REWRITTEN'
    automatic, semantic_guard = json.loads(report['guard_json']), json.loads(control['guard_json'])
    assert automatic['accesses'] == semantic_guard['accesses']
    assert automatic['obligations'][0]['proof'] == 'byte_domain'
    assert semantic_guard['obligations'][0]['proof'] == 'lane_correspondence'
    assert auto.verify() and semantic.verify()
    assert 'arith.minsi' in str(auto)
    assert f'tensor<{factor * 4}x!tt.ptr<f32>>' in str(auto)
    assert f'tensor<{factor}x4xf32>' in str(auto)  # semantic computation retained
    assert str(auto) != str(semantic)
    assert all(r['rewritten'] for r in json.loads(report['layout_json'])['accesses'])
    assert not any(r['rewritten'] for r in json.loads(control['layout_json'])['accesses'])


def test_bool_marker_and_extern_protection_survive(tmp_path):
    for kind in ('bool', 'byte', 'tanh', 'pow'):
        module = parse(tmp_path, case_ir(kind))
        report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0, 'auto')
        assert report['compile_status'] == 'REWRITTEN', report
        assert module.verify()
        assert all(r['rewritten'] for r in json.loads(report['layout_json'])['accesses'])
        if kind == 'bool':
            assert 'was_bool_to_int8 = true' in str(module)
            assert 'tensor<24x!tt.ptr<i8>>' in str(module)
        if kind in ('tanh', 'pow'):
            assert 'tt.extern_elementwise' in str(module)
            # tanh uses total floating arithmetic; only the unknown pow
            # exponent in this fixture requires inactive-lane sanitization.
            assert ('hacc.block_batching_safe_input' in str(module)) == (kind == 'pow')


def test_layout_options_identity_and_disabled_normalization():

    def options(factor, policy):
        return NPUOptions(arch='Ascend950', block_batching_factor=factor, block_batching_layout=policy)

    assert options(3, 'auto').hash() != options(3, 'semantic_only').hash()
    assert options(1, 'auto').hash() == options(1, 'semantic_only').hash()
    assert options(1, 'auto').block_batching_layout == 'semantic_only'
    with pytest.raises(ValueError, match='block_batching_layout'):
        options(3, 'unsafe')


def test_resource_retry_uses_frozen_source_and_exact_request():
    source = object()
    request = BatchRequest(3, 0)
    calls = []

    def compile_variant(options, frozen):
        assert frozen is source
        assert options['block_batching_factor'] == 3
        assert options['block_batching_axis'] == 0
        calls.append(options['block_batching_layout'])
        if len(calls) == 1:
            raise LayoutTargetRejected(3, 0, 'UB size exceeds target capacity')
        return 'complete semantic artifact'

    candidate, reason = _compile_layout_candidate(
        compile_variant, dict(block_batching_factor=3, block_batching_axis=0, block_batching_layout='auto'), source,
        request)
    assert calls == ['auto', 'semantic_only']
    assert candidate == 'complete semantic artifact'
    assert reason['stage'] == 'npubin'


@pytest.mark.parametrize(
    'error',
    [RuntimeError('invalid IR'),
     AssertionError('proof mismatch'),
     LayoutTargetRejected(6, 0, 'wrong attempt')])
def test_internal_or_mismatched_failure_is_not_retried(error):
    calls = []

    def compile_variant(options, source):
        calls.append(options)
        raise error

    with pytest.raises(type(error)):
        _compile_layout_candidate(compile_variant, {}, object(), BatchRequest(3, 0))
    assert len(calls) == 1


def test_retry_is_bounded():
    calls = []

    def compile_variant(options, source):
        calls.append(options)
        raise LayoutTargetRejected(3, 0, 'UB size exceeds target capacity')

    with pytest.raises(LayoutTargetRejected):
        _compile_layout_candidate(compile_variant, {}, object(), BatchRequest(3, 0))
    assert len(calls) == 2


def test_scalar_clamp_bound_is_exact_and_address_min_stays_rejected(tmp_path):
    source = pointwise(0).replace(
        '%nv = tt.splat %n', """%zero_i = arith.constant 0 : i32
    %clamped = arith.maxsi %n, %zero_i : i32
    %bounded = arith.minsi %clamped, %b : i32
    %nv = tt.splat %bounded""")
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert all(r['rewritten'] for r in json.loads(report['layout_json'])['accesses'])
    assert 'minsi' in report['guard_json'] and 'maxsi' in report['guard_json']
    assert module.verify()
    source = pointwise(0).replace('%base = arith.muli %p, %b', '%base = arith.minsi %p, %b')
    module = parse(tmp_path, source)
    report = ascend.passes.ttir.rewrite_block_batching(module, 3, 0)
    assert report['reason'] == 'UNSUPPORTED_ADDRESS_EXPRESSION'


def test_guard_scalar_bounds_reject_nonuniform_and_overflow():
    from triton.backends.ascend.block_batching_guard_v2 import ByteDomain
    from triton.backends.ascend.block_batching_guard import GuardFailure
    domain = ByteDomain([-4], (3, 1, 1), {'shape': [8]})
    clamp = ['minsi', 32, ['maxsi', 32, ['arg', 0, 32], ['const', 0, 32]], ['const', 19, 32]]
    assert domain.expression(clamp).offset == 0
    with pytest.raises(GuardFailure, match='NON_UNIFORM_SCALAR_BOUND'):
        domain.expression(['minsi', 32, ['pid', 0, 32], ['const', 19, 32]])
    with pytest.raises(GuardFailure):
        domain.expression(['minsi', 32, ['const', 2**31, 32], ['const', 19, 32]])
