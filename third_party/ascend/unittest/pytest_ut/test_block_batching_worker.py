# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Worker isolation, artifact/refusal caches and backend failure classification; CPU-only."""
import ast
import importlib
import os
import sys
import pytest
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from block_batching_test_utils import PACKAGE, batching, report_module, bundle_environment, assign_npu  # noqa: F401 - pytest fixtures

pytestmark = pytest.mark.backend('native')

# Isolated workers and atomic publication

worker = importlib.import_module(PACKAGE + '.block_batching_worker')


def test_frozen_worker_source_keeps_function_optional(tmp_path):
    import triton
    from triton.compiler.compiler import ASTSource
    from triton.runtime.jit import JITFunction
    FrozenSource = importlib.import_module(PACKAGE + '.block_batching_source').FrozenSource

    @triton.jit
    def kernel(x):
        pass

    path = tmp_path / 'source.ttir'
    path.write_text('module {}')
    source = ASTSource(kernel, {'x': '*fp32'})
    parent = FrozenSource(source, path, source.hash(), worker._file_digest(path))
    restored = FrozenSource.from_wire(parent.to_wire(), tmp_path)
    assert parent.fn is kernel
    assert isinstance(parent, ASTSource) and isinstance(parent.fn, JITFunction)
    assert restored.fn is None
    # Compile-time instrumentation may inspect ASTSource.fn before logging a signature.
    assert not (isinstance(restored, ASTSource) and isinstance(restored.fn, JITFunction))
    assert restored.to_wire() == dict(parent.to_wire(), path=restored.path)


def test_cold_compiler_fingerprint_does_not_import_device_frameworks():
    import subprocess
    subprocess.run([
        sys.executable, '-I', '-c', '''
import importlib.abc, sys
class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'torch_npu'):
            raise AssertionError('fingerprinting imported ' + name)
sys.meta_path.insert(0, NoFramework())
from triton.runtime.cache import triton_key
assert triton_key() == triton_key()
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
'''
    ], check=True, timeout=30)


@pytest.fixture
def compile_errors(monkeypatch):
    errors = ModuleType('triton.compiler.errors')
    errors.MLIRCompilationError = type('MLIRCompilationError', (RuntimeError, ), {})
    monkeypatch.setitem(sys.modules, 'triton.compiler.errors', errors)
    return errors


def test_worker_bootstrap_report_roundtrip(tmp_path, monkeypatch):
    # Exercise the real interpreter command, private files, core limit, reply
    # identity and wait. Only the native analyzer is substituted in the child.
    fake = tmp_path / 'triton' / 'backends' / 'ascend'
    fake.mkdir(parents=True)
    for package in (fake, fake.parent, fake.parent.parent):
        (package / '__init__.py').touch()
    (fake / 'block_batching_worker.py').write_text(Path(worker.__file__).read_text())
    (fake / '__init__.py').write_text('''
from . import block_batching_worker as worker
import os, resource
assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
def native(operation, body, directory):
    return 'Success', dict(report=dict(report_version=1, requested_factor=body['factor'], requested_axis=body['axis'],
        compile_status='ANALYZED' if operation == 'analyze' else 'REWRITTEN', pid=os.getpid(), text=body['ttir']))
worker._native_report = native
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    for operation in ('analyze', 'probe'):
        report = worker.isolated_report(operation, 'text\n', 4, 2)
        assert report['pid'] != os.getpid() and report['text'] == 'text\n'


@pytest.mark.parametrize('ending,kind', [('raise SystemExit(1)', 'InternalError'),
                                         ('os.kill(os.getpid(), signal.SIGTERM)', 'Signal'),
                                         ('os.kill(os.getpid(), signal.SIGABRT)', 'Signal'),
                                         ('os.kill(os.getpid(), signal.SIGSEGV)', 'Signal'),
                                         ('time.sleep(10)', 'Timeout'), ('pass', 'ProtocolError')])
def test_termination_dominates_any_control_reply(monkeypatch, ending, kind):
    bootstrap = '''
import json, os, resource, signal, sys, time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
d = Path(sys.argv[1]); r = json.loads((d / 'request').read_text()); (d / 'stage').write_text('ttir')
r.update(status='NotApplicable', payload={'reason':'NO_PROGRAM_AXIS'}, operation='wrong' if 'PASS' == 'pass' else 'analyze')
(d / 'result').write_text(json.dumps(r))
'''.replace('PASS', 'pass' if ending == 'pass' else 'no') + ending
    monkeypatch.setattr(worker, '_BOOTSTRAP', bootstrap)
    monkeypatch.setattr(worker, '_TIMEOUT', 0.5 if ending == 'time.sleep(10)' else 10)
    with pytest.raises(worker.BatchCompileError) as caught:
        worker._attempt('analyze', {})
    assert caught.value.kind == kind
    if kind != 'ProtocolError':
        assert caught.value.stage == 'ttir'


def test_normal_worker_exit_preserves_mlir_diagnostic(monkeypatch):
    monkeypatch.setattr(
        worker, '_BOOTSTRAP', '''
import json, sys
from pathlib import Path
directory = Path(sys.argv[1])
request = json.loads((directory / 'request').read_text())
print('first discarded line', 'x' * 17000, file=sys.stderr)
print("loc(kernel.mlir:8:4): error: generated grid contract is invalid", file=sys.stderr)
request.update(status='InternalError', payload=dict(kind='InternalError', stage='ttir', message='rewrite failed'))
(directory / 'result').write_text(json.dumps(request))
''')
    with pytest.raises(worker.BatchCompileError) as caught:
        worker._attempt('compile', {})
    message = str(caught.value)
    assert (caught.value.kind, caught.value.stage) == ('InternalError', 'ttir')
    assert 'rewrite failed' in message and 'loc(kernel.mlir:8:4): error: generated grid contract is invalid' in message
    assert 'first discarded line' not in message


def test_failure_classification(compile_errors):
    import subprocess
    assert worker.BatchDecline('NO_PROGRAM_AXIS', 'ttir').payload() == dict(reason='NO_PROGRAM_AXIS', stage='ttir')
    for reason in ('INVALID_VV_PLAN', 'CANDIDATE_BINARY_CHANGED'):
        with pytest.raises(worker.BatchCompileError, match=reason):
            worker.BatchDecline(reason, 'ttir')
    unknown = subprocess.CalledProcessError(1, ['tool'], output=b'LLVM ERROR', stderr=b'')
    state, payload = worker._classify(unknown, 'npubin')
    assert state == 'InternalError' and 'LLVM ERROR' in payload['message']


def test_checked_generation_rejects_lost_or_changed_binary(tmp_path):
    metadata, binary = tmp_path / 'f.json', tmp_path / 'f.npubin'
    metadata.write_text('{}')
    binary.write_bytes(b'\x7fELFartifact')
    files = {path.name: str(path) for path in (metadata, binary)}
    manifest = worker._artifact_manifest(files, tmp_path, 'v')
    assert worker._checked_group(manifest, tmp_path, 'v', 'f') == files
    binary.write_bytes(b'\x7fELFcorrupt')
    with pytest.raises(worker.BatchCompileError, match='InvalidArtifact'):
        worker._checked_group(manifest, tmp_path, 'v', 'f')
    binary.unlink()
    with pytest.raises(worker.BatchCompileError, match='InvalidArtifact'):
        worker._checked_group(manifest, tmp_path, 'v', 'f')


@pytest.fixture
def decline_compiler(tmp_path, monkeypatch):
    import triton
    from triton.runtime.cache import get_cache_manager
    monkeypatch.setattr(worker, '_variant_key', lambda src, backend, opts, env: worker._digest([vars(opts), env]))
    monkeypatch.setattr(worker, '_worker_request', lambda backend, src, opts, env, key: dict(variant=key))
    caches = {}

    def cache(key):
        return caches.setdefault(key, get_cache_manager(key))

    monkeypatch.setattr(triton.runtime.cache, 'get_cache_manager', cache)
    compiler = worker.CandidateCompiler(SimpleNamespace(parse_options=lambda opts: SimpleNamespace(**opts)),
                                        SimpleNamespace(name='kernel'), {})
    with triton.knobs.cache.scope(), triton.knobs.compilation.scope(), triton.knobs.ascend.scope():
        triton.knobs.cache.dir = str(tmp_path)
        triton.knobs.compilation.listener = None
        triton.knobs.compilation.always_compile = False
        triton.knobs.ascend.compile_only = True
        yield compiler, caches


def test_decline_cache_reuses_exact_attempt_and_retries_stale_entries(decline_compiler, monkeypatch):
    import json
    from contextlib import ExitStack
    from triton import knobs
    compiler, caches = decline_compiler
    calls = []
    refusal = dict(reason='UNSUPPORTED_ADDRESS_EXPRESSION', stage='ttir', analysis={'stage': 'source'})

    def attempt(operation, body):
        calls.append(body['variant'])
        return SimpleNamespace(status='NotApplicable', payload=refusal)

    monkeypatch.setattr(worker, '_attempt', attempt)

    def compile_factor(factor=2, **options):
        with ExitStack() as pending:
            assert compiler._variant(dict(block_batching_factor=factor, **options), pending) == (refusal, None)

    compile_factor()
    compile_factor()
    assert len(calls) == 1
    first = next(cache for cache in caches.values() if cache.get_file(worker._DECLINE))
    assert not list((Path(first.cache_dir).parent / '.block-staging').iterdir())
    compile_factor(4)
    compiler.env_vars['test'] = 'changed'
    compile_factor()
    assert len(set(calls)) == 3
    compiler.env_vars.clear()
    knobs.compilation.always_compile = True
    compile_factor()
    knobs.compilation.always_compile = False
    path = Path(first.get_file(worker._DECLINE))
    stored = json.loads(path.read_text())
    for corrupt in ('{', json.dumps(dict(stored, variant='wrong')), json.dumps(dict(stored, digest='wrong')),
                    json.dumps(dict(stored, version=True))):
        path.write_text(corrupt)
        compile_factor()
    assert len(calls) == 8
    compile_factor()
    assert len(calls) == 8
    for context in [('Ascend950', 24, (False, 24, 48), 1152), ('Ascend950', 12, (True, 12, 24), 1152),
                    ('Ascend950', 12, (True, 12, 24), 2048)]:
        count = len(calls)
        compile_factor(block_compile_context=context)
        compile_factor(block_compile_context=context)
        assert len(calls) == count + 1


@pytest.mark.parametrize('failure', ['resource', 'budget', 'wrong_stage', 'Timeout', 'Signal', 'InternalError'])
def test_non_static_failures_are_never_cached(decline_compiler, monkeypatch, failure):
    from contextlib import ExitStack
    compiler, caches = decline_compiler
    calls = []

    def attempt(*args):
        calls.append(1)
        if failure in ('Timeout', 'Signal', 'InternalError'):
            raise worker.BatchCompileError('failed', kind=failure)
        payload = dict(reason='ANALYSIS_BUDGET_EXCEEDED', stage='ttir')
        if failure == 'resource':
            payload = dict(reason='VV_TARGET_RESOURCE_REJECTED', stage='npubin', diagnostic='UB overflow')
        elif failure == 'wrong_stage':
            payload = dict(reason='UNSUPPORTED_ADDRESS_EXPRESSION', stage='ttadapter')
        return SimpleNamespace(status='NotApplicable', payload=payload)

    monkeypatch.setattr(worker, '_attempt', attempt)
    for _ in range(2):
        with ExitStack() as pending:
            if failure in ('Timeout', 'Signal', 'InternalError'):
                with pytest.raises(worker.BatchCompileError):
                    compiler._variant({}, pending)
            else:
                assert compiler._variant({}, pending)[1] is None
    assert len(calls) == 2
    assert all(cache.get_file(worker._DECLINE) is None for cache in caches.values())


def test_failed_binding_does_not_publish(bundle_environment, monkeypatch):  # noqa: F811
    env = bundle_environment
    compiler_api = ModuleType('triton.compiler.compiler')
    compiler_api.CompiledKernel = lambda *args: env.candidate
    compiler_api._notify_compilation_listener = lambda *args, **kwargs: pytest.fail('unexpected listener')
    monkeypatch.setitem(sys.modules, 'triton.compiler.compiler', compiler_api)
    published = []
    cache = SimpleNamespace(put=lambda *args, **kwargs: published.append(args))
    candidate_compiler = worker.CandidateCompiler(SimpleNamespace(target='npu'), object(), {})
    monkeypatch.setattr(candidate_compiler, '_variant', lambda opts, pending:
                        (None, (cache, {'generation': 'test'}, {}, 'v', None)))
    with pytest.raises(worker.BatchCompileError, match='InvalidArtifact'):
        candidate_compiler.compile(env.base, {}, batching.BatchRequest(4, 0), (0, ))
    assert not published


def test_fallback_bundle_keeps_the_executable_cache_identity(bundle_environment):  # noqa: F811
    env = bundle_environment
    result = batching.BatchKernel(env.base, None, batching.BatchRequest(4, 0),
                                  dict(compile_status='REJECTED', cv_mode='disabled'), ())
    assert result.hash == env.base.hash and result._primary is env.base
    result._set_primary(env.candidate)
    assert result.hash == env.candidate.hash and result._primary is env.candidate


def test_valid_unsupported_lowering_and_corrupt_evidence_are_distinct(compile_errors):
    report = dict(
        owner='block_batching', cv_mode='disabled', lane_width=4, layout={
            'lowering_observation': dict(stage='ttadapter', scope='module', indirect_load_ops=1, indirect_store_ops=0)
        })
    refusal = report_module.vv_memory_lowering_refusal(report)
    assert worker.BatchDecline(refusal, 'ttadapter').reason == 'VV_INDIRECT_MEMORY_NOT_SUPPORTED'
    assert worker.BatchDecline('VV_SYMBOLIC_LAYOUT_NOT_PROVEN', 'ttir').reason == 'VV_SYMBOLIC_LAYOUT_NOT_PROVEN'
    del report['layout']['lowering_observation']['indirect_load_ops']
    invalid = report_module.vv_memory_lowering_refusal(report)
    with pytest.raises(worker.BatchCompileError, match='VV_MEMORY_LOWERING_NOT_PROVEN'):
        worker.BatchDecline(invalid, 'ttadapter')
    source_decline = dict(compile_status='REJECTED', reason='VV_SCAN_DOMAIN_NOT_SUPPORTED')
    assert worker._report_status('probe', source_decline)[0] == 'NotApplicable'
    malformed = report_module.vv_computation_refusal(report)
    with pytest.raises(worker.BatchCompileError, match='VV_COMPUTATION_DOMAIN_NOT_PROVEN'):
        worker.BatchDecline(malformed, 'ttir')


def test_unrecognized_or_cross_operation_data_is_a_protocol_error():
    request = dict(protocol=1, id='request', identity='hash', operation='analyze')
    for payload in (dict(report={}, manifest={}), dict(manifest={})):
        with pytest.raises(worker.BatchCompileError, match='ProtocolError'):
            worker.CompileOutcome(request, dict(request, status='Success', payload=payload))
    for reason in ('UNEXPECTED', [], 'CV_TARGET_RESOURCE_REJECTED'):
        with pytest.raises(worker.BatchCompileError, match='ProtocolError'):
            worker.CompileOutcome(request, dict(request, status='NotApplicable', payload=dict(reason=reason)))
    with pytest.raises(worker.BatchCompileError, match='ProtocolError'):
        worker.CompileOutcome(
            request,
            dict(request, status='InternalError', payload=dict(message='fake signal', kind='Signal', stage='npubin')))


def test_binding_before_index_publish_and_listener_afterwards(bundle_environment, monkeypatch):  # noqa: F811
    from collections import namedtuple
    env = bundle_environment
    events = []
    candidate = env.candidate
    metadata = candidate.metadata._asdict()
    context = ('Ascend950', 24, (False, 24, 48), None)
    metadata.update(hash='variant', target='npu', block_batching_backend_revision='r', block_compile_context=context)
    candidate.metadata = namedtuple('Metadata', metadata)(**metadata)
    fake_api = ModuleType('triton.compiler.compiler')
    fake_api.CompiledKernel = lambda *args: candidate
    fake_api._notify_compilation_listener = lambda listener, **kwargs: listener(**kwargs)
    monkeypatch.setitem(sys.modules, 'triton.compiler.compiler', fake_api)
    report_module = importlib.import_module(PACKAGE + '.block_batching_report')
    monkeypatch.setattr(report_module, 'validated_candidate_report', lambda *args: dict(env.report))
    env.knobs.compilation = SimpleNamespace(listener=lambda **kwargs: events.append(('listener', kwargs['cache_hit'])))
    compiler = worker.CandidateCompiler(
        SimpleNamespace(
            target='npu', parse_options=lambda opts: SimpleNamespace(block_batching_backend_revision='r',
                                                                     block_compile_context=context)), object(), {})
    index = object()
    monkeypatch.setattr(worker, '_publish_artifacts', lambda *args: events.append('index') or {})
    monkeypatch.setattr(batching.BatchKernel, 'publish_manifest', lambda self: events.append(
        ('bundle', self._contract is not None)))
    record = {'stage': Path('/private'), 'manifest': {}}

    def variant(opts, pending):
        if record is None:
            pending.callback(lambda: events.append('read released'))
        return None, (index, record, {}, 'variant', None)

    monkeypatch.setattr(compiler, '_variant', variant)
    bundle = compiler.compile(env.base, dict(block_batching_factor=3), batching.BatchRequest(3, 0), [0, 1, 2, 3])
    assert bundle.candidate is candidate and candidate.initializations == 0
    assert events == ['index', ('bundle', True), ('listener', False)]
    events.clear()
    record = None
    compiler.compile(env.base, dict(block_batching_factor=3), batching.BatchRequest(3, 0), [0, 1, 2, 3])
    assert events == ['read released', ('bundle', True), ('listener', True)]
    events.clear()
    candidate.metadata = candidate.metadata._replace(block_compile_context=('Ascend950', 12, (True, 12, 24), None))
    with pytest.raises(worker.BatchCompileError, match='InvalidArtifact'):
        compiler.compile(env.base, dict(block_batching_factor=3), batching.BatchRequest(3, 0), [0, 1, 2, 3])
    assert events == ['read released']


def test_worker_spawn_failure_is_structured(tmp_path):
    with pytest.raises(worker.BatchCompileError) as failure:
        worker._run_isolated([str(tmp_path / 'missing-worker')], tmp_path, 1)
    assert (failure.value.kind, failure.value.stage) == ('InternalError', 'startup')


def test_failed_attempt_removes_its_generation_only(tmp_path, monkeypatch):
    compiler = worker.CandidateCompiler(None, None, {})
    old = tmp_path / 'known-good'
    old.write_text('previous generation')
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(knobs=object()))
    monkeypatch.setitem(sys.modules, 'triton.compiler.compiler',
                        SimpleNamespace(CompiledKernel=object, _notify_compilation_listener=object))

    def fail(options, pending):
        import shutil
        staging = tmp_path / 'unique'
        staging.mkdir()
        pending.callback(shutil.rmtree, staging)
        (staging / 'kernel.npubin').write_bytes(b'half')
        raise worker.BatchCompileError('worker timed out', kind='Timeout')

    monkeypatch.setattr(compiler, '_variant', fail)
    with pytest.raises(worker.BatchCompileError, match='Timeout'):
        compiler.compile(None, {}, batching.BatchRequest(2, 0), ())
    assert list(tmp_path.iterdir()) == [old]


@pytest.mark.parametrize('abi_mismatch', [True, False])
def test_only_marked_launcher_abi_mismatch_selects_baseline(bundle_environment, abi_mismatch):  # noqa: F811
    env = bundle_environment
    error = NotImplementedError('NATIVE_ABI_MISMATCH')
    if abi_mismatch:
        error.ascend_launcher_abi_mismatch = True

    def fail():
        raise error

    env.candidate._init_handles = fail
    if abi_mismatch:
        env.wrapper._init_handles()
        assert env.wrapper.candidate is None
        assert env.wrapper.block_batching['reason'] == 'NATIVE_ABI_MISMATCH'
    else:
        with pytest.raises(NotImplementedError) as caught:
            env.wrapper._init_handles()
        assert caught.value is error
        assert env.wrapper.candidate is env.candidate


def test_report_validation_returns_capability_value_or_raises_internal_error():
    report = importlib.import_module(PACKAGE + '.block_batching_report')
    unsupported = dict(block_batching={'cv_mode': 'disabled'}, mix_mode='aic', workspace_size=0)
    assert report.validate_compilation_result(unsupported).reason == 'UNSUPPORTED_FINAL_EXECUTION_MODE'
    assert 'compile_status' not in unsupported['block_batching']
    with pytest.raises(worker.BatchCompileError, match='INVALID_WORKSPACE_SIZE'):
        report.validate_compilation_result(dict(unsupported, workspace_size=-1))


@pytest.mark.parametrize('raw', [None, '0', '1'])
def test_worker_knobs_preserve_native_line_info_environment(monkeypatch, raw):

    class Group:
        knob_descriptors = {'disable_line_info': None}

        def __setattr__(self, name, value):
            if name == 'disable_line_info':
                os.environ['TRITON_DISABLE_LINE_INFO'] = str(int(value))
            super().__setattr__(name, value)

    if raw is None:
        monkeypatch.delenv('TRITON_DISABLE_LINE_INFO', raising=False)
    else:
        monkeypatch.setenv('TRITON_DISABLE_LINE_INFO', raw)
    parent = Group()
    assert worker._knob_overrides(parent) == {}
    # The caller may also override the Python option with propagation disabled.
    vars(parent)['disable_line_info'] = True
    child = SimpleNamespace(compilation=Group(), ascend=Group(), cache=Group(), autotuning=Group())
    worker._restore_knobs(
        child,
        dict(knobs=worker._knob_overrides(parent), ascend_knobs={}, autotuning_knobs={}, instrumentation_mode='',
             cache_dir='/tmp/cache', dump_dir='/tmp/dump', propagate_env=False))
    assert child.compilation.disable_line_info is True
    assert os.environ.get('TRITON_DISABLE_LINE_INFO') == raw


def _stage_group(cache, variant, tag):
    import json
    stage = Path(worker._new_stage(cache).cache_dir)
    (stage / 'kernel.json').write_text(json.dumps({'generation': tag}))
    (stage / 'kernel.npubin').write_bytes(b'\x7fELF' + tag.encode())
    group = {p.name: str(p) for p in stage.iterdir()}
    return stage, group, worker._artifact_manifest(group, stage, variant)


def _race_candidate_index(directory, generation, barrier):
    cache = SimpleNamespace(cache_dir=str(directory))
    for _ in range(12):
        stage, group, manifest = _stage_group(cache, 'same-variant', generation)
        barrier.wait(timeout=30)
        worker._publish_artifacts(cache, stage, group, manifest)
        barrier.wait(timeout=30)


def test_concurrent_publishers_preserve_flat_group_identity(tmp_path):
    import json
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(3)
    directory = tmp_path / 'standard-key'
    generations = ['a' * 32, 'b' * 32]
    writers = [
        context.Process(target=_race_candidate_index, args=(directory, generation, barrier))
        for generation in generations
    ]
    for process in writers:
        process.start()
    try:
        for i in range(12):
            barrier.wait(timeout=30)
            if i:
                with worker._cache_lock(directory):
                    manifest = json.loads((directory / worker._MANIFEST).read_bytes())
                    worker._checked_group(manifest, directory, 'same-variant', 'kernel')
            barrier.wait(timeout=30)
            with worker._cache_lock(directory):
                manifest = json.loads((directory / worker._MANIFEST).read_bytes())
                group = worker._checked_group(manifest, directory, 'same-variant', 'kernel')
                generation = json.loads(Path(group['kernel.json']).read_text())['generation']
                assert generation in generations
                assert Path(group['kernel.npubin']).read_bytes() == b'\x7fELF' + generation.encode()
                assert all(Path(p).parent == directory for p in group.values())
    finally:
        for process in writers:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join()
    assert [p.exitcode for p in writers] == [0, 0]


@pytest.mark.parametrize('committed', [True, False])
def test_publication_interruption_preserves_a_complete_flat_group(tmp_path, monkeypatch, committed):
    import json
    cache = SimpleNamespace(cache_dir=str(tmp_path / 'standard-key'))
    previous, group, manifest = _stage_group(cache, 'variant', 'old')
    worker._publish_artifacts(cache, previous, group, manifest)
    stage, group, manifest = _stage_group(cache, 'variant', 'new')
    inode = (stage / 'kernel.npubin').stat().st_ino
    replace = Path.replace

    def interrupted(path, target):
        if path == stage:
            if committed:
                replace(path, target)
                raise KeyboardInterrupt
            raise PermissionError('rename denied')
        return replace(path, target)

    monkeypatch.setattr(Path, 'replace', interrupted)
    with pytest.raises(KeyboardInterrupt if committed else PermissionError):
        worker._publish_artifacts(cache, stage, group, manifest)
    worker._remove_stage(stage)
    directory = Path(cache.cache_dir)
    with worker._cache_lock(directory):
        manifest = json.loads((directory / worker._MANIFEST).read_bytes())
        group = worker._checked_group(manifest, directory, 'variant', 'kernel')
        assert json.loads(Path(group['kernel.json']).read_text())['generation'] == ('new' if committed else 'old')
        if committed:
            assert Path(group['kernel.npubin']).stat().st_ino == inode
        assert json.loads((directory / '__grp__kernel.json').read_text())['child_paths'] == group


def test_directory_recovery_after_writer_disappears(tmp_path):
    cache = SimpleNamespace(cache_dir=str(tmp_path / 'standard-key'))
    stage, group, manifest = _stage_group(cache, 'variant', 'old')
    worker._publish_artifacts(cache, stage, group, manifest)
    directory = Path(cache.cache_dir)
    previous = directory.parent / '.block-staging' / (directory.name + '.previous')
    directory.replace(previous)
    # No exception handler or finally block ran in the vanished publisher.
    with worker._cache_lock(directory):
        assert (directory / 'kernel.npubin').read_bytes() == b'\x7fELFold'
        assert not previous.exists()


def test_flat_candidate_cache_hit_force_compile_and_corruption(decline_compiler, monkeypatch):
    import json
    from contextlib import ExitStack
    from triton import knobs
    compiler, _ = decline_compiler
    calls = []

    def attempt(operation, body):
        calls.append(body['variant'])
        stage = Path(body['stage_key'])
        (stage / 'kernel.json').write_text('{}')
        (stage / 'kernel.npubin').write_bytes(b'\x7fELFbinary')
        group = {p.name: str(p) for p in stage.iterdir()}
        return SimpleNamespace(
            status='Success', payload=dict(manifest=worker._artifact_manifest(group, stage, body['variant']),
                                           timings={}))

    monkeypatch.setattr(worker, '_attempt', attempt)
    for force in (False, False, True):
        knobs.compilation.always_compile = force
        with ExitStack() as pending:
            refusal, result = compiler._variant(dict(block_batching_factor=2), pending)
            assert refusal is None
            cache, record, group, _, _ = result
            if record is not None:
                group = worker._publish_artifacts(cache, record['stage'], group, record['manifest'])
            assert all(Path(p).parent == Path(cache.cache_dir) for p in group.values())
            assert json.loads(Path(cache.get_file('__grp__kernel.json')).read_text())['child_paths'] == group
    assert len(calls) == 2
    knobs.compilation.always_compile = False
    with ExitStack() as pending:
        assert compiler._variant(dict(block_batching_factor=2), pending, cached_only=True)[1] is not None
    Path(group['kernel.npubin']).write_bytes(b'corrupt')
    with ExitStack() as pending, pytest.raises(worker.BatchCompileError, match='InvalidArtifact'):
        compiler._variant(dict(block_batching_factor=2), pending)
    assert len(calls) == 2


def test_baseline_writer_publishes_standard_paths_and_cleans_failure(tmp_path, monkeypatch):
    from triton.runtime.cache import FileCacheManager
    compiler_api = ModuleType('triton.compiler.compiler')
    compiler_api.CompiledKernel = lambda src, group, key: SimpleNamespace(
        metadata=SimpleNamespace(hash=key), metadata_group=group, kernel=Path(group['kernel.npubin']).read_bytes())
    monkeypatch.setitem(sys.modules, 'triton.compiler.compiler', compiler_api)
    cache = FileCacheManager(str(tmp_path / 'standard-key'))
    old_ir = cache.put('old IR', 'kernel.ttadapter')
    with worker.baseline_writer(cache) as writer:
        group = {
            'kernel.ttadapter': old_ir, 'kernel.json': writer.put('{}', 'kernel.json'), 'kernel.npubin':
            writer.put(b'\x7fELFbaseline', 'kernel.npubin')
        }
        stage = Path(writer.cache_dir)
        inode = Path(group['kernel.npubin']).stat().st_ino
        result = writer.commit(object(), group, 'variant')
    assert not stage.exists()
    assert result.kernel == b'\x7fELFbaseline'
    assert Path(result.metadata_group['kernel.npubin']).stat().st_ino == inode
    assert result.metadata_group == cache.get_group('kernel.json')
    assert 'kernel.ttadapter' not in result.metadata_group and not Path(old_ir).exists()
    with pytest.raises(RuntimeError), worker.baseline_writer(cache) as failed:
        failed.put(b'partial', 'kernel.npubin')
        raise RuntimeError('compile failed')
    assert not Path(failed.cache_dir).exists()
    assert Path(result.metadata_group['kernel.npubin']).read_bytes() == result.kernel


def test_analysis_binding_alias_and_isolated_report_listener(monkeypatch):
    import math

    # Stand in for the canonical PyCFunction without importing MLIR on the host.
    native = SimpleNamespace(analyze_block_batching=math.sin)
    monkeypatch.setitem(sys.modules, 'triton._C.libtriton',
                        SimpleNamespace(ascend=SimpleNamespace(passes=SimpleNamespace(ttir=native))))
    request = batching.BatchRequest(4, 0)
    reports = []

    def report(*args):
        reports.append(args)
        return dict(compile_status='ANALYZED', details={'axis': 0})

    monkeypatch.setattr(worker, 'isolated_report', report)
    observed = []

    def listener(value):
        observed.append(value)
        value['details']['axis'] = 2

    for binding in (None, native, SimpleNamespace(analyze_block_batching=native.analyze_block_batching)):
        result = worker.analyze_request('frozen TTIR', request, binding, report_listener=listener)
        assert result['details']['axis'] == 0
    assert len(observed) == 3 and reports == [('analyze', 'frozen TTIR', 4, 0)] * 3

    def listener_failure(value):
        raise LookupError('listener failed')

    with pytest.raises(LookupError, match='listener failed'):
        worker.analyze_request('frozen TTIR', request, report_listener=listener_failure)
    assert len(reports) == 4


def test_custom_analysis_binding_is_explicit_and_never_executes(monkeypatch):
    import math
    native = SimpleNamespace(analyze_block_batching=math.sin)
    monkeypatch.setitem(sys.modules, 'triton._C.libtriton',
                        SimpleNamespace(ascend=SimpleNamespace(passes=SimpleNamespace(ttir=native))))
    monkeypatch.setattr(worker, 'isolated_report', lambda *args: pytest.fail('unexpected native attempt'))
    request = batching.BatchRequest(4, 0)
    for fn in (lambda *args: pytest.fail('unexpected injected binding'), math.cos):
        with pytest.raises(worker.BatchCompileError, match='report_listener') as err:
            worker.analyze_request('module', request, SimpleNamespace(analyze_block_batching=fn))
        assert (err.value.kind, err.value.stage) == ('UnsupportedBinding', 'analyze')
    with pytest.raises(RuntimeError, match='matching rebuilt Ascend'):
        worker.analyze_request('module', request, SimpleNamespace())
    assert worker.analyze_request('module', batching.BatchRequest(1, 0), object())['compile_status'] == 'DISABLED'


@pytest.mark.parametrize('termination', ['SIGSEGV', 'SIGABRT'])
def test_analysis_listener_requires_normal_worker_exit(tmp_path, monkeypatch, termination):
    fake = tmp_path / 'triton' / 'backends' / 'ascend'
    fake.mkdir(parents=True)
    for package in (fake, fake.parent, fake.parent.parent):
        (package / '__init__.py').touch()
    (fake / 'block_batching_worker.py').write_text(Path(worker.__file__).read_text())
    (fake / '__init__.py').write_text(f"""
from . import block_batching_worker as worker
import os, resource, signal
def terminate(operation, body, directory):
    assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
    os.kill(os.getpid(), signal.{termination})
worker._native_report = terminate
""")
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    with pytest.raises(worker.BatchCompileError) as err:
        worker.analyze_request('module', batching.BatchRequest(2, 0),
                               report_listener=lambda report: pytest.fail('invalid completion notification'))
    assert (err.value.kind, err.value.stage) == ('Signal', 'analyze')


@pytest.mark.parametrize('outcome', ['success', 'decline', 'unknown', 'wrong_stage'])
@pytest.mark.parametrize('pipeline', ['simd', 'simt', 'tla'])
def test_worker_private_group_and_stage_outcome(tmp_path, monkeypatch, compile_errors, outcome, pipeline):
    triton = ModuleType('triton')
    knobs = SimpleNamespace(cache=SimpleNamespace(dir=str(tmp_path / 'ordinary'), dump_dir=str(tmp_path / 'dump')))
    triton.knobs = knobs
    monkeypatch.setitem(sys.modules, 'triton', triton)
    source, target, env = object(), object(), {'explicit': 'caller'}
    writer = SimpleNamespace(cache_dir=str(tmp_path / 'generation'))
    Path(writer.cache_dir).mkdir()
    stages = worker._CandidateStages(source, tmp_path, writer)
    options = SimpleNamespace(block_batching_factor=2, enable_tla_mode=pipeline == 'tla',
                              is_pure_simt=pipeline == 'simt', compile_on_910_95=True)
    events = []

    def compile_single(src, compiled_target, opts, *, _env_vars):
        assert src is source and compiled_target is target and _env_vars is env
        index = Path(knobs.cache.dir) / 'compiler-key'
        index.mkdir()
        assert not any(index.iterdir())
        assert stages.bind_writer(src, SimpleNamespace(cache_dir=str(index)), opts) is writer
        assert knobs.cache.dir == str(tmp_path / 'ordinary') and knobs.cache.dump_dir == str(tmp_path / 'dump')

        def lower(module, metadata):
            assert (tmp_path / 'stage').read_text() == 'ttir'
            if outcome == 'unknown':
                raise ValueError('unrecognized compiler error')
            if outcome != 'success':
                return worker.BatchDecline('NO_PROGRAM_AXIS', 'ttir' if outcome == 'decline' else 'ttadapter')
            return 'lowered'

        compiler_path = Path(worker.__file__).with_name('compiler.py')
        backend = next(n for n in ast.parse(compiler_path.read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == 'AscendBackend')
        add_stages = next(n for n in backend.body if isinstance(n, ast.FunctionDef) and n.name == 'add_stages')
        namespace = dict(_active_attempt=worker._active_attempt, _with_debug_line=lambda stage, opt: stage,
                         make_ttir=lambda module, metadata, opt: lower(module, metadata),
                         make_tla_ttir=lambda module, metadata, opt: lower(module, metadata))
        exec(compile(ast.Module(body=[add_stages], type_ignores=[]), str(compiler_path), 'exec'), namespace)
        registered = {}
        namespace['add_stages'](SimpleNamespace(target=SimpleNamespace(backend='npu')), registered, opts, None)
        assert set(registered) == ({'ttir', 'npubin'} if pipeline == 'simt' else {'ttir', 'ttadapter', 'npubin'})
        try:
            result = registered['ttir'](object(), {})
        except Exception as error:
            raise compile_errors.MLIRCompilationError('stage failed') from error
        (Path(writer.cache_dir) / 'k.ttir').write_text(result)
        (index / '__grp__k.json').write_text('private index')
        events.append('child listener')
        return result

    if outcome in ('unknown', 'wrong_stage'):
        with pytest.raises(compile_errors.MLIRCompilationError) as error:
            stages.compile(compile_single, target, options, env)
        assert isinstance(error.value.__cause__, ValueError if outcome == 'unknown' else worker.BatchCompileError)
    else:
        result = stages.compile(compile_single, target, options, env)
        assert result == ('lowered' if outcome == 'success' else worker.BatchDecline('NO_PROGRAM_AXIS', 'ttir'))
    assert events == (['child listener'] if outcome == 'success' else [])
    assert bool(list(stages.index_root.rglob('__grp__*'))) == (outcome == 'success')
    assert not list(Path(writer.cache_dir).glob('__grp__*'))
    assert not (tmp_path / 'ordinary').exists()
    assert knobs.cache.dir == str(tmp_path / 'ordinary') and worker._active_attempt.get() is None


# Vendor resource failures versus internal errors

vendor = importlib.import_module(PACKAGE + '.block_batching_vendor')

OVERFLOW = b'ub overflow, requires 3072256 bits while 1572864 bits available!'

PIPELINE = b'loc("kernel.mlir":2:1): error: Failed to run BishengHIR pipeline\n'

OPTIONS = SimpleNamespace(block_batching_layout='auto', block_batching_factor=4, block_batching_axis=-1)

REWRITTEN = {'layout': {'accesses': [{'rewritten': True}]}}

A5_LAYOUT = b'''[INFO] Using merged native A5 regbase pipeline
[WARNING] --limit-auto-multi-buffer-only-for-local-buffer, --limit-auto-multi-buffer-of-local-buffer and --limit-auto-multi-buffer-buffer are deprecated and will be removed in BiShengIR 1.4.0. Equivalent setting: --multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]". A3 default gm=4, A5 default gm=2; set a level to 1 to disable it.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
warning: warning: --enable-preload=true overrides --enablecv-pipeline-mode; forcing 'Skew'.
loc("/tmp/build/kernel.mlir":1:1): error: Failed to run buildFinalHIVMPipelines pipeline

loc("/tmp/build/kernel.mlir":2:3): error: ub overflow, requires 2097152 bits while 2031616 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
[ERROR] Failed to run BiShengIR regbase pipeline'''

A5_EXTERNAL_COMPILER = A5_LAYOUT.replace(
    b"warning: warning:",
    b'[INFO] libbisheng-llc.so not found at /usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/../lib/libbisheng-llc.so; '
    b'falling back to external `bisheng` process.\nwarning: warning:', 1).replace(b'2097152', b'2097408')

A3_LAYOUT = b'''[NOTE] Ub overflow detected; automatically set enable-code-motion to false and retrying compilation.
[NOTE] Ub overflow detected; automatically set enable-auto-multi-buffer to false and retrying compilation.
[NOTE] Due to Ub overflow, automatically set enable-code-motion to false, set enable-auto-multi-buffer to false; compilation still failed.
loc("/tmp/build/kernel.ttadapter.mlir":1:1): error: Failed to run BiShengHIR pipeline

loc("/tmp/build/kernel.ttadapter.mlir":2:3): error: ub overflow, requires 2621440 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
[ERROR] Failed to run BiShengIR pipeline'''

A3_LAYOUT_SINGLE_RETRY = b'''[NOTE] Ub overflow detected; automatically set enable-code-motion to false and retrying compilation.
[NOTE] Due to Ub overflow, automatically set enable-code-motion to false; compilation still failed.
loc("/tmp/build/kernel.ttadapter.mlir":1:1): error: Failed to run BiShengHIR pipeline

loc("/tmp/build/kernel.ttadapter.mlir":2:3): error: ub overflow, requires 2621440 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
[ERROR] Failed to run BiShengIR pipeline'''

A3_LOCATIONS = b'loc("/tmp/build/kernel.ttadapter.mlir":1:1): loc("/tmp/build/kernel.ttadapter.mlir":2:3): '

A3_INFO = b'[INFO] libbisheng-llc.so not found in LD_LIBRARY_PATH; using the configured compiler path.\n'

A3_PIPELINE_WARNING = (
    b'loc("/tmp/build/kernel.ttadapter.mlir":2:1): '
    b'loc("/tmp/build/kernel.ttadapter.mlir":3:3): warning: cv-pipelining is disabled for this kernel\n')


def test_a3_resource_retry_requires_complete_diagnostics():
    report = dict(cv_mode='disabled')
    for diagnostic in (A3_LAYOUT, A3_LAYOUT_SINGLE_RETRY):
        for stdout in (b'', A3_LOCATIONS):
            err = subprocess.CalledProcessError(1, ['vendor'], output=stdout, stderr=diagnostic)
            refusal = vendor.classify_vendor_failure(err, report, OPTIONS, cv_resources=True)
            assert refusal.reason == 'VV_TARGET_RESOURCE_REJECTED'
            assert diagnostic.decode() in refusal.diagnostic
    for rc, stdout, stderr in ((-11, A3_LOCATIONS,
                                A3_LAYOUT), (2, A3_LOCATIONS, A3_LAYOUT), (1, A3_LOCATIONS + b'LLVM ERROR', A3_LAYOUT),
                               (1, A3_LOCATIONS.replace(b'kernel.', b'other.'),
                                A3_LAYOUT), (1, A3_LOCATIONS, A3_LAYOUT + b'\nerror: verifier failed'),
                               (1, A3_LOCATIONS, A3_LAYOUT.replace(b'2621440', b'1572864'))):
        err = subprocess.CalledProcessError(rc, ['vendor'], output=stdout, stderr=stderr)
        assert vendor.classify_vendor_failure(err, report, OPTIONS, cv_resources=True) is None


def test_a3_resource_retry_accepts_captured_preamble_and_pipeline_warning():
    # A3 emits the loader note on stderr and a multi-location cv-pipelining
    # warning on stdout before the complete retry transcript.
    err = subprocess.CalledProcessError(1, ['vendor'], output=A3_PIPELINE_WARNING, stderr=A3_INFO + A3_LAYOUT)
    refusal = vendor.classify_vendor_failure(err, dict(cv_mode='disabled'), OPTIONS, cv_resources=True)
    assert refusal.reason == 'VV_TARGET_RESOURCE_REJECTED'
    assert A3_INFO.decode() in refusal.diagnostic
    assert A3_PIPELINE_WARNING.decode() in refusal.diagnostic


def test_a3_resource_retry_rejects_unknown_prefixed_diagnostic():
    err = subprocess.CalledProcessError(1, ['vendor'], output=A3_PIPELINE_WARNING,
                                        stderr=b'[ERROR] unexpected compiler failure\n' + A3_LAYOUT)
    assert vendor.classify_vendor_failure(err, dict(cv_mode='disabled'), OPTIONS, cv_resources=True) is None


@pytest.mark.parametrize('filename', [b'kernel.mlir', b'kernel.ttadapter.mlir'])
@pytest.mark.parametrize('diagnostic', [A5_LAYOUT, A5_EXTERNAL_COMPILER])
def test_a5_vv_layout_budget_exhaustion(filename, diagnostic):
    diagnostic = diagnostic.replace(b'kernel.mlir', filename)
    err = subprocess.CalledProcessError(1, ['vendor'], stderr=diagnostic)
    refusal = vendor.classify_vendor_failure(err, dict(REWRITTEN, cv_mode='disabled'), OPTIONS, cv_resources=True)
    assert (refusal.reason, refusal.stage, refusal.factor, refusal.axis) == ('LAYOUT_TARGET_RESOURCE_REJECTED',
                                                                             'npubin', 4, -1)
    assert refusal.diagnostic == diagnostic.decode()


@pytest.mark.parametrize('rc, stdout, stderr', [
    (-11, b'', A5_LAYOUT),
    (134, b'', A5_LAYOUT),
    (139, b'', A5_LAYOUT),
    (2, b'', A5_LAYOUT),
    (1, b'LLVM ERROR: verifier failed', A5_LAYOUT),
    (1, b'', A5_LAYOUT + b'\n[WARNING] unexpected subpass failure'),
    (1, b'', A5_LAYOUT.replace(b'[INFO]', b'[ERROR] LLVM ERROR:')),
    (1, b'', A5_LAYOUT.replace(b'2097152', b'2031616')),
    (1, b'', A5_LAYOUT.replace(b'ub overflow', b'ca overflow')),
    (1, b'', A5_LAYOUT.replace(b'kernel.mlir":2', b'other.mlir":2')),
    (1, b'', A5_EXTERNAL_COMPILER.replace(b'[INFO] libbisheng', b'[ERROR] libbisheng')),
    (1, b'', A5_EXTERNAL_COMPILER + b'\nLLVM ERROR: verifier failed'),
    (1, b'', A5_EXTERNAL_COMPILER.replace(b'2097408', b'2031616')),
    (2, b'', A5_EXTERNAL_COMPILER),
])
def test_a5_mixed_or_unproven_failure_is_internal(rc, stdout, stderr):
    err = subprocess.CalledProcessError(rc, ['vendor'], output=stdout, stderr=stderr)
    assert vendor.classify_vendor_failure(err, dict(REWRITTEN, cv_mode='disabled'), OPTIONS, cv_resources=True) is None


@pytest.mark.parametrize('mode, rewritten, policy', [
    ('independent_batch_v1', True, 'auto'),
    ('spatial_concat_v1', True, 'auto'),
    ('disabled', False, 'auto'),
    ('disabled', True, 'semantic_only'),
])
def test_a5_resource_failure_without_layout_retry_selects_baseline(mode, rewritten, policy):
    for diagnostic in (A5_LAYOUT, A5_EXTERNAL_COMPILER):
        err = subprocess.CalledProcessError(1, ['vendor'], stderr=diagnostic)
        report = {'cv_mode': mode, 'layout': {'accesses': [{'rewritten': rewritten}]}}
        options = SimpleNamespace(block_batching_layout=policy, block_batching_factor=4, block_batching_axis=-1)
        for cv_resources in (True, False):
            refusal = vendor.classify_vendor_failure(err, report, options, cv_resources=cv_resources)
            if mode == 'disabled':
                assert refusal.reason == 'VV_TARGET_RESOURCE_REJECTED'
            elif cv_resources:
                assert refusal.reason == 'CV_TARGET_RESOURCE_REJECTED'
            else:
                assert refusal.reason == 'LAYOUT_TARGET_RESOURCE_REJECTED'


@pytest.mark.parametrize('rc, stdout, stderr', [
    (-11, b'', OVERFLOW),
    (134, OVERFLOW, b''),
    (139, b'', OVERFLOW),
    (0, b'', OVERFLOW),
    (1, b'LLVM ERROR: verifier failed', OVERFLOW),
    (1, OVERFLOW, b'unexpected failure'),
    (1, b'unsupported pass: insufficient UB', b''),
    (1, b'', b'ub overflow, requires 1 bits while 4 bits available!'),
    (1, b'', OVERFLOW + b' extra diagnostics'),
])
def test_unmodeled_vendor_exit_never_becomes_resource_decline(rc, stdout, stderr):
    err = subprocess.CalledProcessError(rc, ['vendor'], output=stdout, stderr=stderr)
    vendor.classify_vendor_failure(err, dict(REWRITTEN, cv_mode='independent_batch_v1'), OPTIONS, cv_resources=True)
    assert err.stage == 'npubin'
    assert vendor.resource_kind(rc, stdout, stderr) is None


@pytest.mark.parametrize('buf', [b'ub', b'cbuf', b'ca', b'cb', b'cc'])
def test_cv_takes_precedence_and_preserves_both_vendor_streams(buf):
    # The compiler can place its pipeline line and capacity error on different streams.
    stdout = PIPELINE
    stderr = b'loc("kernel.mlir":3:3): error: ' + OVERFLOW.replace(b'ub', buf)
    err = subprocess.CalledProcessError(1, ['vendor'], output=stdout, stderr=stderr)
    refusal = vendor.classify_vendor_failure(err, dict(REWRITTEN, cv_mode='independent_batch_v1'), OPTIONS,
                                             cv_resources=True)
    assert refusal.reason == 'CV_TARGET_RESOURCE_REJECTED' and refusal.stage == 'npubin'
    assert stdout.rstrip().decode() in refusal.diagnostic
    assert stderr.decode() in refusal.diagnostic


def test_only_rewritten_auto_layout_gets_ub_retry():
    err = subprocess.CalledProcessError(1, ['vendor'], output=OVERFLOW, stderr=b'')
    assert vendor.classify_vendor_failure(err, {}, OPTIONS, cv_resources=False) is None
    semantic = SimpleNamespace(block_batching_layout='semantic_only', block_batching_factor=4, block_batching_axis=-1)
    assert vendor.classify_vendor_failure(err, REWRITTEN, semantic, cv_resources=False) is None
    for cv_resources in (True, False):
        rejected = vendor.classify_vendor_failure(err, REWRITTEN, OPTIONS, cv_resources=cv_resources)
        assert (rejected.reason, rejected.stage, rejected.factor, rejected.axis) == ('LAYOUT_TARGET_RESOURCE_REJECTED',
                                                                                     'npubin', 4, -1)


def test_multiline_bisheng_capacity_message():
    diagnostic = PIPELINE + OVERFLOW + (
        b' (possible reason\nlarge or block number is more than what user expect '
        b'due to multi-buffer feature is enabled and some ops need extra local buffer. )')
    assert vendor.resource_kind(1, b'', diagnostic) == 'ub'


@pytest.mark.parametrize('empty', [True, False])
def test_success_without_candidate_object_is_internal(tmp_path, empty):
    error_type = importlib.import_module(PACKAGE + '.block_batching_worker').BatchCompileError
    output = tmp_path / 'kernel.o'
    if empty:
        output.touch()
    result = subprocess.CompletedProcess(['vendor'], 0, b'vendor stdout', OVERFLOW)
    with pytest.raises(error_type) as failed:
        vendor.require_candidate_binary(output, result)
    assert (failed.value.kind, failed.value.stage) == ('InvalidArtifact', 'npubin')
    assert (failed.value.stdout, failed.value.stderr) == (result.stdout, result.stderr)
    output.write_bytes(b'vendor payload')
    vendor.require_candidate_binary(output, result)
    assert output.read_bytes() == b'vendor payload'


@pytest.mark.parametrize("changed", [None, "variant", "plan_id", "candidate_factor", "requested_axis"])
def test_selected_ttir_binds_to_the_exact_candidate(changed):
    prepared = dict(
        variant='candidate', digest='bytes', path='private.ttir',
        report=dict(plan_id='candidate', owner='block_batching', compile_status='REWRITTEN', candidate_factor=2,
                    requested_axis=-1, selected_axis=0))
    options = SimpleNamespace(block_batching_factor=2, block_batching_axis=-1)
    if changed is None:
        worker._check_ttir(prepared, 'candidate', options)
    else:
        target = prepared if changed == 'variant' else prepared['report']
        target[changed] = 'wrong'
        with pytest.raises(worker.BatchCompileError) as error:
            worker._check_ttir(prepared, 'candidate', options)
        assert (error.value.kind, error.value.stage) == ('InvalidArtifact', 'ttir')


def test_selected_ttir_rejects_changed_bytes_before_parsing(tmp_path):
    path = tmp_path / 'selected.ttir'
    path.write_bytes(b'replaced')
    prepared = dict(variant='candidate', digest='original', path=str(path), report={})
    with pytest.raises(worker.BatchCompileError) as error:
        worker._restore_ttir(prepared, dict(hash='candidate'), None, tmp_path)
    assert (error.value.kind, error.value.stage) == ('InvalidArtifact', 'ttir')
