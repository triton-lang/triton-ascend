# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Candidate subprocess and generation contracts, independent of the NPU driver."""
import ast
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from block_batching_contract_host import PACKAGE, batching, bundle_environment  # noqa: F401

worker = importlib.import_module(PACKAGE + '.block_batching_worker')


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


def test_valid_unsupported_lowering_and_corrupt_evidence_are_distinct(compile_errors):
    report = dict(
        owner='block_batching', cv_mode='disabled', lane_width=4, layout={
            'lowering_observation': dict(stage='ttadapter', scope='module', indirect_load_ops=1, indirect_store_ops=0)
        })
    refusal = batching.vv_memory_lowering_refusal(report)
    assert worker.BatchDecline(refusal, 'ttadapter').reason == 'VV_INDIRECT_MEMORY_NOT_SUPPORTED'
    del report['layout']['lowering_observation']['indirect_load_ops']
    invalid = batching.vv_memory_lowering_refusal(report)
    with pytest.raises(worker.BatchCompileError, match='VV_MEMORY_LOWERING_NOT_PROVEN'):
        worker.BatchDecline(invalid, 'ttadapter')
    source_decline = dict(compile_status='REJECTED', reason='VV_SCAN_DOMAIN_NOT_SUPPORTED')
    assert worker._report_status('probe', source_decline)[0] == 'NotApplicable'
    malformed = batching.vv_computation_refusal(report)
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
    metadata.update(hash='variant', target='npu', block_batching_backend_revision='r')
    candidate.metadata = namedtuple('Metadata', metadata)(**metadata)
    fake_api = ModuleType('triton.compiler.compiler')
    fake_api.CompiledKernel = lambda *args: candidate
    fake_api._notify_compilation_listener = lambda listener, **kwargs: listener(**kwargs)
    monkeypatch.setitem(sys.modules, 'triton.compiler.compiler', fake_api)
    report_module = importlib.import_module(PACKAGE + '.block_batching_report')
    monkeypatch.setattr(report_module, 'validated_candidate_report', lambda *args: dict(env.report))
    env.knobs.compilation = SimpleNamespace(listener=lambda **kwargs: events.append(('listener', kwargs['cache_hit'])))
    compiler = worker.CandidateCompiler(
        SimpleNamespace(target='npu', parse_options=lambda opts: SimpleNamespace(block_batching_backend_revision='r')),
        object(), {})
    index = object()
    monkeypatch.setattr(worker, '_publish_generation', lambda *args: events.append('index'))
    monkeypatch.setattr(batching.BatchKernel, 'publish_manifest', lambda self: events.append(
        ('bundle', self._contract is not None)))
    record = {'generation': 'g'}
    monkeypatch.setattr(compiler, '_variant', lambda opts, pending: (None, (index, record, {}, 'variant', None)))
    bundle = compiler.compile(env.base, dict(block_batching_factor=3), batching.BatchRequest(3, 0), [0, 1, 2, 3])
    assert bundle.candidate is candidate and candidate.initializations == 0
    assert events == ['index', ('bundle', True), ('listener', False)]
    events.clear()
    record = None
    compiler.compile(env.base, dict(block_batching_factor=3), batching.BatchRequest(3, 0), [0, 1, 2, 3])
    assert events == [('bundle', True), ('listener', True)]


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


def _race_candidate_index(directory, generation, barrier):
    import json
    from contextlib import ExitStack

    variant = 'same-variant'
    attempt = directory / 'attempts' / generation
    attempt.mkdir(parents=True)
    metadata = attempt / 'kernel.json'
    binary = attempt / 'kernel.npubin'
    metadata.write_text(json.dumps({'generation': generation}))
    binary.write_bytes(b'\x7fELF' + generation.encode())
    files = {p.name: str(p) for p in (metadata, binary)}
    manifest = worker._artifact_manifest(files, attempt, variant)
    record = dict(generation=generation, manifest=manifest)
    for _ in range(12):
        barrier.wait(timeout=30)
        with ExitStack() as pending:
            worker._publish_generation(SimpleNamespace(cache_dir=directory), record, pending)
        barrier.wait(timeout=30)


def test_concurrent_publishers_preserve_generation_identity(tmp_path):
    import json
    import multiprocessing

    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(3)
    generations = ['a' * 32, 'b' * 32]
    writers = [
        context.Process(target=_race_candidate_index, args=(tmp_path, generation, barrier))
        for generation in generations
    ]
    for process in writers:
        process.start()
    index = tmp_path / worker._MANIFEST
    try:
        for i in range(12):
            barrier.wait(timeout=30)
            # From round 2 a complete prior result is always visible, including
            # during any concurrent replacement of the public index.
            if i:
                previous = json.loads(index.read_bytes())
                assert previous['generation'] in generations
                worker._checked_group(previous['manifest'], tmp_path / 'attempts' / previous['generation'],
                                      'same-variant', 'kernel')
            barrier.wait(timeout=30)
            record = json.loads(index.read_bytes())
            generation = record['generation']
            assert generation in generations
            group = worker._checked_group(record['manifest'], tmp_path / 'attempts' / generation, 'same-variant',
                                          'kernel')
            assert json.loads(Path(group['kernel.json']).read_text())['generation'] == generation
            assert Path(group['kernel.npubin']).read_bytes() == b'\x7fELF' + generation.encode()
    finally:
        for process in writers:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join()
    assert [p.exitcode for p in writers] == [0, 0]


@pytest.mark.parametrize("committed", [True, False])
def test_index_commit_interruption_never_removes_visible_generation(tmp_path, monkeypatch, committed):
    from contextlib import ExitStack
    import json
    import shutil

    stage = tmp_path / 'private'
    stage.mkdir()
    manifest = tmp_path / worker._MANIFEST
    manifest.write_text('{"generation": "previous"}')
    replace = Path.replace

    def interrupted(path, target):
        if committed:
            replace(path, target)
            raise KeyboardInterrupt
        raise PermissionError('rename denied')

    monkeypatch.setattr(Path, "replace", interrupted)
    expected = KeyboardInterrupt if committed else PermissionError
    with pytest.raises(expected), ExitStack() as pending:
        pending.callback(shutil.rmtree, stage)
        worker._publish_generation(SimpleNamespace(cache_dir=tmp_path), {"generation": stage.name}, pending)
    assert json.loads(manifest.read_text())['generation'] == (stage.name if committed else 'previous')
    assert stage.exists() == committed


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
        result = batching.analyze_request('frozen TTIR', request, binding, report_listener=listener)
        assert result['details']['axis'] == 0
    assert len(observed) == 3 and reports == [('analyze', 'frozen TTIR', 4, 0)] * 3

    def listener_failure(value):
        raise LookupError('listener failed')

    with pytest.raises(LookupError, match='listener failed'):
        batching.analyze_request('frozen TTIR', request, report_listener=listener_failure)
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
            batching.analyze_request('module', request, SimpleNamespace(analyze_block_batching=fn))
        assert (err.value.kind, err.value.stage) == ('UnsupportedBinding', 'analyze')
    with pytest.raises(RuntimeError, match='matching rebuilt Ascend'):
        batching.analyze_request('module', request, SimpleNamespace())
    assert batching.analyze_request('module', batching.BatchRequest(1, 0), object())['compile_status'] == 'DISABLED'


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
        batching.analyze_request('module', batching.BatchRequest(2, 0),
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
