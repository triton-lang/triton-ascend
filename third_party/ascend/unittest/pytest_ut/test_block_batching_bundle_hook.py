# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Exercise the packaged public compiler hook, including callbacks deferred by JIT."""
import ast
import hashlib
import importlib
import sys
import shutil
import subprocess
from collections import namedtuple
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from block_batching_contract_host import PACKAGE, batching, bundle_environment  # noqa: F401

pytestmark = pytest.mark.backend('native')


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
    backend.compile_bundle = lambda src, options, base: batching.compile_bundle(backend, src, options, base)
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
