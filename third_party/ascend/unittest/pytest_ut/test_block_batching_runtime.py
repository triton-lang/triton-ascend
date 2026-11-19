# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Runtime guards, dispatch and public compile/JIT binding; CPU-only."""
from dataclasses import FrozenInstanceError
import importlib
import random
import pytest
import ast
import hashlib
import sys
import shutil
import subprocess
import unittest
from block_batching_test_utils import batching, bundle_environment, guard_module, PACKAGE, worker_module, assign_npu  # noqa: F401 - pytest fixtures
from collections import namedtuple
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch

pytestmark = pytest.mark.backend('native')

# Guard arithmetic and dispatch


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


def _guard_spec(version, accesses):
    guard = dict(version=version, axis=0, accesses=accesses)
    if version == 3:
        guard.update(
            plan_kind='unified', semantic_contract_version=4, memory_policy='disjoint_or_terminal_store', factor=2,
            terminal_read_ids=[],
            obligations=[dict(access_id=i, rule='SupportedEffect', proof='byte_domain') for i in range(len(accesses))])
    return guard


@pytest.mark.parametrize('version', [1, 2, 3])
def test_guard_schemas_preserve_alias_overflow_and_cache_invalidation(version):
    lane = ['range', 0, 8, 32] if version == 1 else ['coord', 0, 8, 32]
    offset = ['addi', 32, ['muli', 32, ['pid', 0, 32], ['const', 8, 32]], lane]
    if version != 1:
        offset = ['muli', 64, offset, ['const', 4, 64]]
    accesses = [dict(root=i, bytes=4, write=bool(i), mask=None, offset=offset) for i in range(2)]
    if version != 1:
        for access in accesses:
            access.update(shape=[8], pointer_checks=[], loops=[], volatile=False, device_address=False)
    spec = _guard_spec(version, accesses)
    cached = guard_module.RuntimeGuard(spec, 2)
    for arguments, grid, expected in [
        ([4096, 8192], (2, 1, 1), (True, 'PROVEN_AT_RUNTIME')),
        ([4096, 4096], (2, 1, 1), (False, 'WRITE_ALIAS')),
        ([4096, 8192], (2**29, 1, 1), (False, 'INDEX_OVERFLOW')),
        ([4096, 8192], (0, 1, 1), (False, 'EMPTY_GRID')),
        ([0, 8192], (2, 1, 1), (False, 'INVALID_MEMORY_ARGUMENT')),
        ([4096, 8192], (2, 1, 1), (True, 'PROVEN_AT_RUNTIME')),
    ]:
        assert guard_module.check_guard(spec, arguments, grid, 2) == expected
        assert cached(arguments, grid) == expected


@pytest.mark.parametrize('version', [2, 3])
def test_cv_effect_proof_keeps_both_reduction_axes_and_padding(version):
    row, column = ['coord', 0, 32, 32], ['coord', 1, 32, 32]
    offset = [
        'addi', 64, ['muli', 64, ['pid', 0, 32], ['const', 4096, 64]],
        ['addi', 64, ['muli', 64, row, ['const', 128, 64]], ['muli', 64, column, ['const', 4, 64]]]
    ]
    accesses = [
        dict(root=i, bytes=4, write=bool(i), mask=None, offset=offset, shape=[32, 32], pointer_checks=[], loops=[],
             volatile=False, device_address=False) for i in range(2)
    ]
    accesses[0]['mask'] = ['slt', row, ['arg', 2, 32], 32]
    spec = _guard_spec(version, accesses)
    spec.update(
        require_same_fractal_bounds=True,
        cv_mask_contract=dict(version=1, inputs=[dict(access=0, operand=i, axes=[0, 1]) for i in range(2)],
                              outputs=[1]))
    for limit, expected in [(32, (True, 'PROVEN_AT_RUNTIME')), (31, (False, 'CV_BATCH_REDUCTION_MASK_NOT_PROVEN')),
                            (24, (False, 'CV_BATCH_PADDING_NOT_PROVEN'))]:
        assert guard_module.check_guard(spec, [1 << 20, 1 << 30, limit], (2, 1, 1), 2) == expected
    if version == 3:
        spec['obligations'][1]['rule'] = 'ShareInvariantLoad'
        assert guard_module.check_guard(spec, [1 << 20, 1 << 30, 32], (2, 1, 1), 2) == (False, 'INVALID_EFFECT_PROOF')


# Public compile and deferred JIT hooks


@pytest.fixture(scope='module')
def packaged_entries(tmp_path_factory):
    root = Path(__file__).resolve().parents[4]
    staging = tmp_path_factory.mktemp('bundle-hook')
    files = ('python/triton/compiler/compiler.py', 'python/triton/runtime/jit.py')
    for name in files:
        out = staging / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, out)
    # CI can run from the already-patched worktree. Otherwise apply the same
    # distribution patch as the package build, restricted to the two inputs.
    if 'def _compile(' not in (staging / files[0]).read_text():
        patch = root / 'third_party/ascend/patch/triton-ascend-3.6.0.patch'
        subprocess.run(['git', 'apply', *('--include=' + f for f in files), str(patch)], cwd=staging, check=True)
    compiler, jit = [ast.parse((staging / name).read_text()) for name in files]
    entry = next(n for n in compiler.body if isinstance(n, ast.FunctionDef) and n.name == 'compile')
    jit_cls = next(n for n in jit.body if isinstance(n, ast.ClassDef) and n.name == 'JITFunction')
    do_compile = next(n for n in jit_cls.body if isinstance(n, ast.FunctionDef) and n.name == '_do_compile')
    cache_path = root / 'python/triton/runtime/cache.py'
    cache_key = next(n for n in ast.parse(cache_path.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'get_cache_key')

    def bind(node, path, namespace):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace[node.name]

    return (lambda scope: bind(entry, staging / files[0], scope),
            lambda scope: bind(do_compile, staging / files[1], scope),
            bind(cache_key, cache_path, {'triton_key': lambda: 'host-test-installation'}))


@pytest.fixture
def boundary(packaged_entries, bundle_environment, monkeypatch):  # noqa: F811
    public, jit, cache_key = packaged_entries
    host = bundle_environment
    host.knobs.compilation = SimpleNamespace(override=False)
    host.knobs.cache = SimpleNamespace(manager_class=None, remote_manager_class=None)
    cache = sys.modules['triton.runtime.cache']
    monkeypatch.setattr(cache, 'get_cache_key', cache_key, raising=False)
    worker = importlib.import_module(PACKAGE + '.block_batching_worker')
    target = SimpleNamespace(arch='Ascend950')

    class Options(SimpleNamespace):

        def hash(self):
            return str(sorted(vars(self).items()))

    def parse_options(opts):
        normalized = dict(is_pure_simt=False, compile_mode='simd', block_batching_backend_revision='test')
        normalized.update(opts)
        return Options(**normalized)

    backend = SimpleNamespace(target=target, hash=lambda: 'ascend', parse_options=parse_options)
    source = SimpleNamespace(fn=SimpleNamespace(arg_names=[]), signature={}, hash=lambda: 'source',
                             parse_options=lambda: {})
    backend.compile_bundle = lambda src, options, base: worker_module.compile_bundle(backend, src, options, base)
    current = {'COMPILER_ENV': 'current'}
    groups = {}
    candidates = []

    def single(src, selected_target, options, env):
        assert selected_target is target
        environment = current if env is None else env
        normalized = backend.parse_options(dict(options))
        key = cache_key(src, backend, normalized, environment)
        fingerprint = hashlib.sha256(key.encode()).hexdigest()
        if fingerprint not in groups:
            fields = dict(hash=fingerprint, **vars(normalized), **environment)
            groups[fingerprint] = namedtuple('Metadata', fields)(**fields)
        return SimpleNamespace(hash=fingerprint, metadata=groups[fingerprint])

    def observe_candidate(compiler, base, options, request, positions):
        candidates.append((base.hash, dict(compiler.env_vars)))
        return base

    monkeypatch.setattr(worker.CandidateCompiler, 'compile', observe_candidate)
    compile_fn = public(
        dict(make_backend=lambda _: backend, GPUTarget=object, _compile=single,
             get_cache_invalidating_env_vars=lambda: dict(current),
             driver=SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target))))
    return SimpleNamespace(compile=compile_fn, jit=jit, cache_key=cache_key, backend=backend, target=target, src=source,
                           current=current, groups=groups, candidates=candidates, worker=worker)


@pytest.mark.parametrize('submitted', [{}, {'COMPILER_ENV': 'submitted', 'CUSTOM_ENV': 'caller-only'}])
def test_cold_and_cached_baseline_prove_exact_environment(boundary, submitted):
    b = boundary
    standard = b.compile(b.src, options={'block_batching_factor': 1, 'block_batching_axis': -1}, _env_vars=submitted)
    assert b.candidates == []
    b.groups.clear()  # Exercise a cold callback first, then a metadata cache hit.
    for _ in range(2):
        b.current['COMPILER_ENV'] = 'after submission'
        result = b.compile(b.src, options={'block_batching_factor': 4}, _env_vars=submitted)
        assert b.candidates[-1] == (standard.hash, submitted)
        assert result.metadata == standard.metadata
    assert len(b.candidates) == 2 and len(b.groups) == 1


@pytest.mark.parametrize('submitted', [None, {}, {'TRITON_ENABLE_ASAN': '1', 'CUSTOM_ENV': 'caller'}])
def test_auto_hook_freezes_the_same_environment_for_selection_and_f1(boundary, submitted):
    b = boundary
    opts = {'block_batching_factor': 1, 'block_batching_axis': -1}
    expected = dict(b.current if submitted is None else submitted)
    standard = b.compile(b.src, options=opts, _env_vars=expected)

    def select(src, options, compile_variant, *, env_vars):
        b.current['COMPILER_ENV'] = 'changed after submission'
        if submitted is not None:
            submitted['CUSTOM_ENV'] = 'changed after submission'
        assert env_vars == expected
        return compile_variant(opts, src)

    b.backend.compile_bundle = select
    actual = b.compile(b.src, options={'block_batching_factor': 'auto'}, _env_vars=submitted)
    assert actual.hash == standard.hash and actual.metadata == standard.metadata


def test_unprovable_environment_falls_back_without_compiling_candidate(boundary):
    b = boundary
    result = b.compile(b.src, options={'block_batching_factor': 4}, _env_vars={'mixedCase': 'unprovable'})
    assert result.candidate is None and result.baseline.metadata.mixedCase == 'unprovable'
    assert result.block_batching['reason'] == 'BASELINE_ENVIRONMENT_NOT_PROVEN'
    assert b.candidates == []


def test_mismatched_baseline_identity_is_an_error(boundary):
    b = boundary
    opts = {'block_batching_factor': 1, 'block_batching_axis': -1}
    baseline = b.compile(b.src, options=opts, _env_vars={})
    baseline.metadata = baseline.metadata._replace(hash='different-artifact')
    with pytest.raises(b.worker.BatchCompileError) as caught:
        b.worker._baseline_environment(b.src, b.backend, opts, baseline)
    assert (caught.value.kind, caught.value.stage) == ('InvalidArtifact', 'bind')


def test_deferred_jit_uses_submission_environment_for_baseline_and_candidate(boundary):
    b = boundary
    submitted = dict(b.current)
    submit = Mock(return_value='pending')
    cache = {}
    kernel = SimpleNamespace(device_caches={0: (cache, {}, b.target, b.backend, None)}, _call_hook=lambda *args: False,
                             ASTSource=lambda *args: b.src, compile=b.compile)
    options = b.backend.parse_options({'block_batching_factor': 4})
    active = ContextVar('async_compile_mode', default=SimpleNamespace(submit=submit))
    run = b.jit(
        dict(knobs=SimpleNamespace(runtime=SimpleNamespace(jit_cache_hook=None, jit_post_compile_hook=None)),
             _async_compile=SimpleNamespace(active_mode=active),
             get_cache_invalidating_env_vars=lambda: dict(b.current), get_cache_key=b.cache_key))
    assert run(kernel, 'jit-key', {}, 0, {}, options, {}, False) == 'pending'
    key, deferred, finalize = submit.call_args.args
    assert key == b.cache_key(b.src, b.backend, options, submitted)
    b.current.clear()
    result = deferred()
    finalize(result)
    assert result.metadata.COMPILER_ENV == 'current'
    assert b.candidates == [(result.hash, submitted)] and cache == {'jit-key': result}


# Selected-CV dispatch


class TestSelectedDispatch(unittest.TestCase):

    def setUp(self):
        package = ModuleType('_selected_dispatch_test')
        package.__path__ = [str(Path(__file__).resolve().parents[2] / 'backend')]
        self.patcher = patch.dict(sys.modules, {package.__name__: package})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.batching = importlib.import_module(package.__name__ + '.block_batching')
        self.events = []
        self.guard = Mock(return_value=(True, 'PROVEN_AT_RUNTIME'))

        def kernel(role):
            return SimpleNamespace(hash=role, metadata=SimpleNamespace(_asdict=lambda: {}), function=role,
                                   packed_metadata=role, launch_metadata=lambda *args: None,
                                   run=lambda *args: self.events.append(args[4]))

        self.baseline, self.candidate = kernel('baseline'), kernel('candidate')
        guard = importlib.import_module(package.__name__ + '.block_batching_guard')
        make_guard = patch.object(guard, 'RuntimeGuard', return_value=self.guard)
        make_guard.start()
        self.addCleanup(make_guard.stop)
        self.bundle = self.make_bundle('independent_batch_v1')
        self.knobs = SimpleNamespace(ascend=SimpleNamespace(compile_only=False))

    def make_bundle(self, cv_mode):
        report = dict(cv_mode=cv_mode, selected_axis=0, lane_width=2, guard={}, owner='block_batching')
        bundle = self.batching.BatchKernel(self.baseline, self.candidate, self.batching.BatchRequest(2, 0), report, [0],
                                           publish_manifest=False)
        bundle.block_auto_selection = dict(model_complete=True, selected_factor=2)
        return bundle

    def dispatch(self, grid=4, capture=False):
        triton = ModuleType('triton')
        triton.knobs = self.knobs
        with patch.dict(sys.modules, {'triton': triton}), \
             patch.object(self.batching, '_capture_state', return_value=capture), \
             patch.object(self.batching, '_core_floor', return_value=56):
            self.bundle._dispatch(grid, 1, 1, 0, None, None, None, None, None, 1234)
        return self.bundle.last_launch

    def test_automatic_cv_uses_the_scored_small_grid_candidate(self):
        launch = self.dispatch()
        self.assertEqual(self.events, ['candidate'])
        self.assertEqual(launch['applied_factor'], 2)

    def test_manual_cv_and_automatic_vv_retain_the_existing_floor(self):
        del self.bundle.block_auto_selection
        self.assertEqual(self.dispatch()['reason'], 'GRID_BELOW_CORE_FLOOR')
        self.bundle = self.make_bundle('disabled')
        self.assertEqual(self.dispatch()['reason'], 'GRID_BELOW_CORE_FLOOR')
        self.assertEqual(self.events, ['baseline', 'baseline'])

    def test_selected_cv_still_checks_aliases(self):
        self.guard.return_value = (False, 'WRITE_ALIAS')
        self.assertEqual(self.dispatch()['reason'], 'WRITE_ALIAS')
        self.assertEqual(self.events, ['baseline'])

    def test_selected_cv_still_checks_divisibility_and_capture(self):
        self.assertEqual(self.dispatch(grid=3)['reason'], 'CV_GRID_NOT_DIVISIBLE')
        self.assertEqual(self.dispatch(capture=True)['reason'], 'CAPTURE_BASELINE')
        self.assertEqual(self.events, ['baseline', 'baseline'])


class TestPreparedFallback(unittest.TestCase):

    def setUp(self):
        self.events = []

        def kernel(role):
            return SimpleNamespace(hash=role, metadata=SimpleNamespace(_asdict=lambda: {}), function=role,
                                   packed_metadata=role, launch_metadata=lambda *args: None, _init_handles=Mock(),
                                   _run=True, run=lambda *args: self.events.append((args[4], args[5])))

        self.original = kernel("f1")
        self.candidate = kernel("f2")
        self.guard = Mock(return_value=(True, "PROVEN_AT_RUNTIME"))
        with patch.object(guard_module, "RuntimeGuard", return_value=self.guard):
            self.bundle = batching.BatchKernel(
                self.original, self.candidate, batching.BatchRequest(2, 0),
                dict(cv_mode="independent_batch_v1", selected_axis=0, lane_width=2, guard={}, owner="block_batching"),
                [0], publish_manifest=False)
        self.bundle.block_auto_selection = dict(selected_factor=2)

    def dispatch(self, *, capture=False):
        with patch.object(batching, "_capture_state", return_value=capture), \
             patch.object(batching, "_core_floor", return_value=56):
            self.bundle._dispatch(4, 1, 1, 0, None, None, None, None, None, 1234)

    def test_eager_initialization_keeps_capture_fallback_ready(self):
        with patch.object(batching, "_capture_state", return_value=False):
            self.bundle._init_handles()
        self.original._init_handles.assert_called_once()
        self.candidate._init_handles.assert_called_once()
        self.dispatch()
        self.dispatch(capture=True)
        self.assertEqual(self.events, [("f2", "f2"), ("f1", "f1")])

    def test_alias_fallback_uses_the_complete_prepared_baseline(self):
        self.guard.return_value = False, "WRITE_ALIAS"
        self.dispatch()
        self.dispatch()
        self.assertEqual(self.events, [("f1", "f1"), ("f1", "f1")])
        self.assertEqual(self.bundle.last_launch["applied_factor"], 1)

    def test_candidate_load_resource_failure_uses_complete_f1(self):
        from triton.runtime.errors import OutOfResources
        self.candidate._init_handles.side_effect = OutOfResources(4096, 1024, "shared memory")
        cache = {}
        with patch.object(batching, "_capture_state", return_value=False):
            self.assertIs(batching.initialize_auto_kernel(cache, self.bundle), self.bundle)
            self.assertIs(batching.initialize_auto_kernel(cache, self.bundle), self.bundle)
        self.candidate._init_handles.assert_called_once()
        self.original._init_handles.assert_called_once()
        self.dispatch()
        self.assertIsNone(self.bundle.candidate)
        self.assertEqual(self.events, [("f1", "f1")])
