# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Compatibility contracts for arithmetic reuse and lazy launch policy."""
from dataclasses import FrozenInstanceError
import importlib
import random

import pytest
from block_batching_contract_host import batching, bundle_environment, guard_module, PACKAGE

pytestmark = pytest.mark.backend('native')


@pytest.mark.parametrize('rank', [0, 1, 4, 8, 19])
def test_affine_arithmetic_preserves_values_and_dimensions(rank):
    cls = guard_module.Affine
    rng = random.Random(9741 + rank)
    for _ in range(100):
        a, b = (cls(rng.randrange(-2**63, 2**63), tuple(rng.randrange(-8, 9) for _ in range(rank))) for _ in range(2))
        for sign in [-1, 1]:
            assert a.add(b, sign=sign) == cls(a.offset + sign * b.offset,
                                              tuple(x + sign * y for x, y in zip(a.coefficients, b.coefficients)))
        scaled = a.scale(scale=-3)
        assert scaled.offset == -3 * a.offset
        assert scaled.coefficients == tuple(-3 * x for x in a.coefficients)
        assert a.uniform == all(x == 0 for x in a.coefficients)
        counts = tuple(rng.randrange(1, 8) for _ in range(rank))
        deltas = [k * (n - 1) for k, n in zip(a.coefficients, counts)]
        assert a.bounds(counts) == (a.offset + sum(min(0, v)
                                                   for v in deltas), a.offset + sum(max(0, v) for v in deltas))
    assert cls(3).coefficients == (0, 0, 0, 0)
    with pytest.raises(FrozenInstanceError):
        a.offset = 0


def test_affine_refuses_mismatched_dimensions_and_keeps_concrete_result_type():

    class Derived(guard_module.Affine):
        pass

    assert type(Derived(2).scale(scale=3)) is guard_module.Affine
    assert Derived(1, (1, 2)).add(guard_module.Affine(2, (3, 4))).coefficients == (4, 6)
    # Zipping down to the shorter operand used to drop the remaining
    # coefficients silently, which under-estimates the proved address range
    # and is therefore unsafe in the permissive direction. Refuse instead.
    with pytest.raises(guard_module.GuardFailure, match='AFFINE_DIMENSION_MISMATCH'):
        Derived(1, (1, 2)).add(guard_module.Affine(2, (3, )))
    with pytest.raises(guard_module.GuardFailure, match='AFFINE_DIMENSION_MISMATCH'):
        guard_module.Affine(1, (1, 2)).bounds((4, ))


def test_affine_dimension_mismatch_selects_baseline_instead_of_raising(monkeypatch):
    specification = {
        'version':
        1,
        'axis':
        0,
        'accesses': [{
            'root': 0,
            'bytes': 4,
            'write': False,
            'mask': None,
            'offset': ['addi', 64, ['pid', 0, 32], ['const', 4, 64]],
        }],
    }
    arguments, grid = [4096], (4, 1, 1)
    assert guard_module.check_guard(specification, arguments, grid, 2) == (True, 'PROVEN_AT_RUNTIME')

    # Model a call site drifting to another width: scalars become one-wide
    # while program ids stay four-wide. The refusal has to reach the caller as
    # a baseline selection; an escaping exception would abort the launch.
    affine = guard_module.Affine
    monkeypatch.setattr(guard_module, 'Affine', lambda offset, coefficients=(0, ): affine(offset, coefficients))
    refused = (False, 'AFFINE_DIMENSION_MISMATCH')
    assert guard_module.check_guard(specification, arguments, grid, 2) == refused
    assert guard_module.RuntimeGuard(specification, 2)(arguments, grid) == refused


@pytest.mark.parametrize('grid,cv,preserve,capture,floor,expected,queries', [
    ((0, 1, 1), False, False, True, 99, 'EMPTY_GRID', []),
    ((2, 1, 1), True, False, True, 99, 'CV_GRID_NOT_DIVISIBLE', ['capture']),
    ((1, 1, 1), False, False, True, 99, 'GRID_NOT_REDUCED', ['capture']),
    ((6, 1, 1), False, False, True, 3, 'GRID_BELOW_CORE_FLOOR', ['capture', 'floor']),
    ((6, 1, 1), False, False, True, 2, 'CAPTURE_BASELINE', ['capture', 'floor']),
    ((6, 1, 1), False, False, None, 2, 'CAPTURE_QUERY_UNAVAILABLE', ['capture', 'floor']),
    ((6, 1, 1), False, False, False, 2, 'PROVEN_AT_RUNTIME', ['capture', 'floor', 'guard']),
    ((1, 1, 1), False, True, False, 1, 'PROVEN_AT_RUNTIME', ['capture', 'floor', 'guard']),
])
def test_dispatch_keeps_gate_order_and_schedule_grid(bundle_environment, monkeypatch, grid, cv, preserve, capture,
                                                     floor, expected, queries):
    env = bundle_environment
    env.report["cv_mode"] = "independent_batch_v1" if cv else "disabled"
    calls = []

    def query_capture():
        calls.append('capture')
        return capture

    def query_floor():
        calls.append('floor')
        return floor

    def guard(args, original_grid):
        calls.append('guard')
        assert original_grid == grid
        return True, 'PROVEN_AT_RUNTIME'

    report_module = importlib.import_module(PACKAGE + '.block_batching_report')
    monkeypatch.setattr(report_module, 'validate_vv_loop_contract', lambda metadata: preserve)
    monkeypatch.setattr(guard_module, 'RuntimeGuard', lambda specification, factor: guard)
    env.wrapper = batching.BatchKernel(env.base, env.candidate, batching.BatchRequest(3, 0), env.report, [0, 1, 2, 3])
    env.wrapper._init_handles()
    monkeypatch.setattr(batching, '_capture_state', query_capture)
    monkeypatch.setattr(batching, '_core_floor', query_floor)
    env.wrapper._dispatch(*grid, *env.launch[3:])
    report = env.wrapper.last_launch
    assert calls == queries
    assert report.get('reason', report['runtime_status']) == expected
    if expected == 'PROVEN_AT_RUNTIME':
        assert env.launches[-1][0] == 'candidate'
        assert report['batch_grid'] == (grid if preserve else ((grid[0] + 2) // 3, *grid[1:]))
        assert report['grouping'] == ('loop_iterations' if preserve else 'programs')
    elif expected == 'EMPTY_GRID':
        assert not env.launches
    else:
        assert env.launches[-1][0] == 'baseline'


def test_no_candidate_does_not_query_capture_floor_or_guard(bundle_environment, monkeypatch):
    env = bundle_environment
    env.report['reason'] = 'STATIC_REFUSAL'
    env.wrapper = batching.BatchKernel(env.base, None, batching.BatchRequest(3, 0), env.report, [0, 1, 2, 3])

    def forbidden(*args):
        raise AssertionError('unavailable candidate must not query launch prerequisites')

    monkeypatch.setattr(batching, '_capture_state', forbidden)
    monkeypatch.setattr(batching, '_core_floor', forbidden)
    env.wrapper._guard = forbidden
    env.wrapper._dispatch(*env.launch)
    assert env.wrapper.last_launch['reason'] == 'STATIC_REFUSAL'
    assert env.launches[-1][0] == 'baseline'


def test_core_query_failure_still_precedes_capture_refusal(bundle_environment, monkeypatch):
    env = bundle_environment
    monkeypatch.setattr(batching, '_capture_state', lambda: True)

    def invalid_floor():
        raise ValueError('invalid core configuration')

    monkeypatch.setattr(batching, '_core_floor', invalid_floor)
    with pytest.raises(ValueError, match='invalid core configuration'):
        env.wrapper._dispatch(*env.launch)
    assert not env.launches


def test_vv_diagnostic_mutations_cannot_change_execution(bundle_environment):
    env = bundle_environment
    env.wrapper._dispatch(*env.launch)
    assert env.wrapper.last_launch['runtime_status'] == 'APPLIED'
    env.report['selected_axis'] = 1
    env.candidate.metadata.block_batching['guard'].clear()
    env.wrapper.block_batching.update(selected_axis=2, lane_width=99, cv_mode='independent_batch_v1', owner='none')
    env.wrapper.block_batching['guard'].clear()
    env.wrapper.request = batching.BatchRequest(99, 2)
    env.wrapper.positions[:] = [999]
    snapshot = env.wrapper.last_launch
    snapshot['applied_factor'] = 99
    env.wrapper._dispatch(*env.launch)
    assert env.wrapper.last_launch['runtime_status'] == 'APPLIED'
    assert env.wrapper.last_launch['applied_factor'] == 3
    assert env.wrapper.last_launch['selected_axis'] == 0
    assert env.wrapper.last_launch['batch_grid'] == (1, 1, 1)
    alias = list(env.launch)
    alias[10] = alias[9]
    env.wrapper._dispatch(*alias)
    assert env.wrapper.last_launch['reason'] == 'WRITE_ALIAS'
    assert [name for name, _ in env.launches] == ['candidate', 'candidate', 'baseline']


def test_cv_keeps_existing_report_and_dispatch_behavior(bundle_environment):
    env = bundle_environment
    env.report['cv_mode'] = 'independent_batch_v1'
    env.wrapper = batching.BatchKernel(env.base, env.candidate, batching.BatchRequest(3, 0), env.report, [0, 1, 2, 3])
    env.wrapper._dispatch(6, *env.launch[1:])
    assert env.wrapper.last_launch['runtime_status'] == 'APPLIED'
    env.report['selected_axis'] = 1
    env.wrapper._dispatch(6, *env.launch[1:])
    assert env.wrapper.last_launch['reason'] == 'CV_GRID_NOT_DIVISIBLE'


def test_vv_candidates_own_factor_and_guard_state(bundle_environment):
    from copy import deepcopy
    env = bundle_environment
    wrappers = []
    for factor in (2, 4):
        report = deepcopy(env.report)
        report['lane_width'] = factor
        candidate = env.Kernel('candidate_' + str(factor), report)
        wrappers.append(
            batching.BatchKernel(env.base, candidate, batching.BatchRequest(factor, 0), report, [0, 1, 2, 3]))
        report['guard'].clear()
        report['selected_axis'] = 2
    for wrapper, factor in zip(wrappers, (2, 4)):
        wrapper._dispatch(8, *env.launch[1:])
        assert wrapper.last_launch['applied_factor'] == factor
        assert wrapper.last_launch['batch_grid'] == (8 // factor, 1, 1)
        assert env.launches[-1][0] == 'candidate_' + str(factor)


def test_candidate_execution_error_is_not_replayed_on_baseline(bundle_environment):
    env = bundle_environment

    def fail(*launch):
        raise RuntimeError('candidate launch failed')

    env.candidate.run = fail
    with pytest.raises(RuntimeError, match='candidate launch failed'):
        env.wrapper._dispatch(*env.launch)
    assert not env.launches
    assert env.wrapper.last_launch['runtime_status'] == 'NOT_LAUNCHED'


@pytest.mark.parametrize('test_name,implementations', [
    ('EffectProjectionTest', ['BatchExpr.cpp', 'BatchFactsJSON.cpp']),
    ('AffineProfilesTest',
     ['BatchExpr.cpp', 'BatchAffine.cpp', 'BatchRangeAnalysis.cpp', 'BatchLoopSchedule.cpp', 'BatchLoopLegality.cpp']),
])
def test_checked_batching_values_under_ubsan(tmp_path, test_name, implementations):
    import os
    from pathlib import Path
    import re
    import shlex
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[2]
    llvm = shutil.which('llvm-config')
    if os.getenv('LLVM_SYSPATH'):
        llvm = str(Path(os.environ['LLVM_SYSPATH']) / 'bin/llvm-config')
    if not llvm:
        for cache in (root.parents[1] / 'build').glob('*/CMakeCache.txt'):
            match = re.search(r'^LLVM_DIR:[^=]+=(.+)/lib/cmake/llvm$', cache.read_text(), re.M)
            if match:
                llvm = str(Path(match[1]) / 'bin/llvm-config')
                break
    compiler = os.getenv('CXX', 'g++')
    if not llvm or not Path(llvm).is_file() or not shutil.which(compiler):
        pytest.skip('owned-fact UBSan test needs a C++ compiler and LLVM development installation')
    include = subprocess.check_output([llvm, '--includedir'], text=True).strip()
    libs = shlex.split(subprocess.check_output([llvm, '--ldflags', '--libs', 'support', '--system-libs'], text=True))
    sources = [root / 'unittest/BlockBatching' / (test_name + '.cpp')]
    sources += [root / 'lib/TritonToGraph/BlockBatching' / name for name in implementations]
    exe = tmp_path / test_name
    command = [
        compiler, '-std=c++17', '-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=all',
        '-I' + str(root / 'include'), '-I' + include, *map(str, sources), *libs, '-o',
        str(exe)
    ]
    # Match the linked LLVM build; UBSan vptr instrumentation requires RTTI
    # symbols which a no-RTTI LLVMSupport intentionally does not export.
    if subprocess.check_output([llvm, '--has-rtti'], text=True).strip() == 'NO':
        command.insert(1, '-fno-rtti')
    subprocess.run(command, capture_output=True, text=True, check=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    if test_name == 'EffectProjectionTest':
        assert 'owned effect projection regressions passed' in result.stdout
    assert not result.stderr


def test_cv_binding_reads_live_mode_and_an_initially_bound_empty_report(bundle_environment):
    env = bundle_environment
    env.report['cv_mode'] = 'independent_batch_v1'
    wrapper = batching.BatchKernel(env.base, env.candidate, batching.BatchRequest(3, 0), env.report, [0, 1, 2, 3])
    wrapper._dispatch(5, *env.launch[1:])
    assert wrapper.last_launch['reason'] == 'CV_GRID_NOT_DIVISIBLE'
    env.report['cv_mode'] = 'disabled'
    wrapper._dispatch(5, *env.launch[1:])
    assert wrapper.last_launch['runtime_status'] == 'APPLIED'
    env.report.clear()
    with pytest.raises(KeyError, match='selected_axis'):
        wrapper._dispatch(5, *env.launch[1:])


def test_element_span_reports_negative_stride_before_reading_missing_write_policy():
    domain = guard_module.Domain([], (2, 1, 1))
    access = {'offset': ['muli', 64, ['pid', 0, 32], ['const', -1, 64]], 'mask': None}
    with pytest.raises(guard_module.GuardFailure, match='NEGATIVE_STRIDE'):
        domain.access(access)
