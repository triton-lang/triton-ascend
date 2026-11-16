# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

import triton
import triton.language as tl
import triton.knobs

# ``triton.autotune`` is resolved when a test builds its kernel, so this
# side-effect import must run first to install the Ascend factory.
import triton.backends.ascend.runtime  # noqa: F401
from triton._internal_testing import _fresh_knobs_impl


@pytest.fixture
def fresh_knobs():
    """Copy/reset every knob set, mirroring ``python/test/conftest.py``."""
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def _clear_in_process_cache():
    """Drop JITFunction's in-memory kernels so the next call re-enters compile()."""
    device = torch.npu.current_device()
    _add_kernel.device_caches[device][0].clear()


def _fresh_cache_dir(prefix):
    return tempfile.mkdtemp(prefix=prefix)


# ---------------------------------------------------------------------------
# store-binary-only patch
# ---------------------------------------------------------------------------


def test_store_binary_only():
    """T25: store_binary_only keeps the final npubin and skips intermediate IR."""
    cache_dir = _fresh_cache_dir("knob-store-binary-")
    try:
        with triton.knobs.cache.scope(), triton.knobs.compilation.scope():
            triton.knobs.cache.dir = cache_dir
            triton.knobs.compilation.store_binary_only = True
            _clear_in_process_cache()
            compiled = _add_kernel.warmup(torch.float32, torch.float32, torch.float32, 1024, BLOCK=256, grid=(4, ))

        suffixes = {path.suffix for path in Path(cache_dir).rglob("*") if path.is_file()}
        # The final binary must survive, otherwise CompiledKernel cannot build asm.
        assert ".npubin" in suffixes
        assert ".json" in suffixes
        assert compiled.asm["npubin"]
        for ext in (".ttir", ".ttadapter"):
            assert ext not in suffixes

        # The knob is not part of the cache key: a run with the default value
        # must hit the same entry and load the stored npubin.
        with triton.knobs.cache.scope(), triton.knobs.compilation.scope():
            triton.knobs.cache.dir = cache_dir
            triton.knobs.compilation.store_binary_only = False
            _clear_in_process_cache()
            cached = _add_kernel.warmup(torch.float32, torch.float32, torch.float32, 1024, BLOCK=256, grid=(4, ))
        assert cached.metadata.hash == compiled.metadata.hash
        assert cached.asm["npubin"] == compiled.asm["npubin"]
    finally:
        _clear_in_process_cache()
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# autotuning.print
# ---------------------------------------------------------------------------


def _make_tunable_add(cache_results=False):
    configs = [triton.Config({"BLOCK": 128}), triton.Config({"BLOCK": 256})]

    @triton.autotune(configs=configs, key=["n"], cache_results=cache_results)
    @triton.jit
    def tunable_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    return tunable_add


def test_autotuning_print(capsys, monkeypatch):
    """T28: autotuning.print is read at run time and accepts any truthy value."""
    n = 1024
    cache_dir = _fresh_cache_dir("knob-autotune-print-")
    x = torch.randn(n, device="npu", dtype=torch.float32)
    y = torch.randn(n, device="npu", dtype=torch.float32)
    out = torch.empty(n, device="npu", dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]), )
    try:
        with triton.knobs.cache.scope(), triton.knobs.autotuning.scope():
            triton.knobs.cache.dir = cache_dir
            triton.knobs.autotuning.print = True
            tunable_add = _make_tunable_add()

            tunable_add[grid](x, y, out, n)
            torch.npu.synchronize()
            first = capsys.readouterr().out
            assert "Triton autotuning for function" in first
            assert "best config selected" in first

            tunable_add[grid](x, y, out, n)
            torch.npu.synchronize()
            second = capsys.readouterr().out
            assert "Triton autotuning for function" not in second
            assert torch.allclose(out, x + y)

        with triton.knobs.cache.scope(), triton.knobs.autotuning.scope():
            triton.knobs.cache.dir = cache_dir
            triton.knobs.autotuning.print = False
            tunable_late = _make_tunable_add()
            triton.knobs.autotuning.print = True
            tunable_late[grid](x, y, out, n)
            torch.npu.synchronize()
            late = capsys.readouterr().out
            assert "Triton autotuning for function" in late

        with triton.knobs.cache.scope(), triton.knobs.autotuning.scope():
            triton.knobs.cache.dir = cache_dir
            triton.knobs.autotuning.reset()
            monkeypatch.setenv("TRITON_PRINT_AUTOTUNING", "true")
            tunable_env = _make_tunable_add()
            tunable_env[grid](x, y, out, n)
            torch.npu.synchronize()
            env_out = capsys.readouterr().out
            assert "Triton autotuning for function" in env_out
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Ascend knob group (T31-T35)
#
# ``triton.knobs.ascend`` is attached at backend import time.  It follows the
# upstream descriptor semantics, so the tests check the Ascend-specific names,
# consumers and cache keys rather than re-testing the generic knob machinery.
# ---------------------------------------------------------------------------

_ASCEND_KNOB_FIELDS = {
    "enable_sanitizer": "TRITON_ENABLE_SANITIZER",
    "print_ub_bits": "ENABLE_PRINT_UB_BITS",
    "memory_display": "TRITON_MEMORY_DISPLAY",
    "npu_compiler_path": "TRITON_NPU_COMPILER_PATH",
    "enable_ubtuner": "TRITON_ENABLE_UBTUNER",
    "autotune_parallel_compile": "TRITON_AUTOTUNE_PARALLEL_COMPILE",
    "bench_method": "TRITON_BENCH_METHOD",
    "compile_only": "TRITON_COMPILE_ONLY",
    "enable_taskqueue": "TRITON_ENABLE_TASKQUEUE",
    "device_print": "TRITON_DEVICE_PRINT",
    "grid_warn_print": "TRITON_GRID_WARN_PRINT",
}

_COMPILE_ENV_KEYS = (
    "TRITON_ENABLE_SANITIZER",
    "ENABLE_PRINT_UB_BITS",
    "TRITON_MEMORY_DISPLAY",
    "TRITON_NPU_COMPILER_PATH",
    "NPU_DEVICE_LIMIT",
    "TRITON_ENABLE_UBTUNER",
)

_AUTOTUNE_ENV_KEYS = ("TRITON_ENABLE_UBTUNER", "TRITON_BENCH_METHOD")


def test_ascend_knob_group_exposed():
    """T31: knobs.ascend exists and covers the documented fields only."""
    from triton.backends.ascend.knobs import install_ascend_knobs

    assert isinstance(triton.knobs.ascend, triton.knobs.base_knobs)
    descriptors = triton.knobs.ascend.knob_descriptors
    assert {name: knob.key for name, knob in descriptors.items()} == _ASCEND_KNOB_FIELDS
    assert set(triton.knobs.ascend.knobs) == set(_ASCEND_KNOB_FIELDS)

    env_keys = {knob.key for knob in descriptors.values()}
    assert "ASCEND_HOME_PATH" not in env_keys
    assert "NPU_DEVICE_LIMIT" not in env_keys

    # ``install_ascend_knobs`` is idempotent and keeps the same group object,
    # so module-level ``from triton import knobs`` consumers stay valid.
    group = triton.knobs.ascend
    install_ascend_knobs()
    assert triton.knobs.ascend is group


def test_ascend_knob_defaults_and_env(fresh_knobs, monkeypatch):
    """T32: defaults match the historical getenv() defaults; env is read live."""
    knobs = fresh_knobs.ascend
    assert knobs.knobs == {
        "enable_sanitizer": False,
        "print_ub_bits": False,
        "memory_display": False,
        "npu_compiler_path": None,
        "enable_ubtuner": "",
        "autotune_parallel_compile": True,
        "bench_method": "default",
        "compile_only": False,
        "enable_taskqueue": True,
        "device_print": False,
        "grid_warn_print": False,
    }

    for field, key in _ASCEND_KNOB_FIELDS.items():
        monkeypatch.setenv(
            key, {
                "enable_sanitizer": "1",
                "print_ub_bits": "true",
                "memory_display": "ON",
                "npu_compiler_path": "/opt/npuir",
                "enable_ubtuner": "compile",
                "autotune_parallel_compile": "0",
                "bench_method": "npu",
                "compile_only": "yes",
                "enable_taskqueue": "0",
                "device_print": "1",
                "grid_warn_print": "y",
            }[field])

    assert knobs.knobs == {
        "enable_sanitizer": True,
        "print_ub_bits": True,
        "memory_display": True,
        "npu_compiler_path": "/opt/npuir",
        "enable_ubtuner": "compile",
        "autotune_parallel_compile": False,
        "bench_method": "npu",
        "compile_only": True,
        "enable_taskqueue": False,
        "device_print": True,
        "grid_warn_print": True,
    }

    for key in _ASCEND_KNOB_FIELDS.values():
        monkeypatch.delenv(key)
    assert knobs.knobs == {
        "enable_sanitizer": False,
        "print_ub_bits": False,
        "memory_display": False,
        "npu_compiler_path": None,
        "enable_ubtuner": "",
        "autotune_parallel_compile": True,
        "bench_method": "default",
        "compile_only": False,
        "enable_taskqueue": True,
        "device_print": False,
        "grid_warn_print": False,
    }


def test_ascend_knob_assignment_scope_and_copy(fresh_knobs, monkeypatch):
    """T33: assignment writes env; scope/copy/reset follow upstream semantics."""
    knobs = fresh_knobs.ascend

    knobs.enable_sanitizer = True
    assert os.environ["TRITON_ENABLE_SANITIZER"] == "1"
    assert knobs.enable_sanitizer

    # ``= knobs.env`` only drops the in-process override; the environment
    # variable written by the previous assignment stays (upstream semantics).
    knobs.enable_sanitizer = fresh_knobs.env
    assert os.environ["TRITON_ENABLE_SANITIZER"] == "1"
    assert knobs.enable_sanitizer
    monkeypatch.delenv("TRITON_ENABLE_SANITIZER")
    assert not knobs.enable_sanitizer

    with knobs.scope():
        knobs.device_print = True
        assert knobs.device_print
        assert os.environ["TRITON_DEVICE_PRINT"] == "1"
    assert not knobs.device_print
    assert "TRITON_DEVICE_PRINT" not in os.environ

    # ``copy()`` snapshots instance overrides; propagate_env=False keeps the
    # copy from leaking into the process environment.
    fresh_knobs.propagate_env = False
    knobs.enable_taskqueue = False
    copied = knobs.copy()
    assert not copied.enable_taskqueue
    copied.reset()
    assert copied.enable_taskqueue
    assert not knobs.enable_taskqueue
    assert "TRITON_ENABLE_TASKQUEUE" not in os.environ


def test_ascend_knob_consumers(fresh_knobs):
    """T34: Ascend compile helpers read knobs.ascend, not os.environ."""
    from triton.backends.ascend import utils

    knobs = fresh_knobs.ascend
    assert not utils._is_ascend_sanitizer_enabled()
    assert not utils._enable_print_ub_bits()
    assert not utils._enable_dump_memory_info()

    knobs.enable_sanitizer = True
    knobs.print_ub_bits = True
    knobs.memory_display = True
    assert utils._is_ascend_sanitizer_enabled()
    assert utils._enable_print_ub_bits()
    assert utils._enable_dump_memory_info()

    os.environ["TRITON_ENABLE_SANITIZER"] = "0"
    assert knobs.enable_sanitizer
    assert utils._is_ascend_sanitizer_enabled()


def test_ascend_cache_invalidating_env_vars(monkeypatch):
    """T35: compile/autotune-affecting Ascend env vars enter the cache key."""
    import triton.compiler.compiler as compiler_module
    import triton.runtime.autotuner as upstream_autotuner
    import triton.runtime.jit as jit_module
    import triton.backends.ascend.runtime.autotuner as ascend_autotuner

    modules = (jit_module, compiler_module, upstream_autotuner, ascend_autotuner)
    for module in modules:
        assert getattr(module.get_cache_invalidating_env_vars, "_ascend_extended", False)

    for key in _COMPILE_ENV_KEYS + ("TRITON_BENCH_METHOD", ):
        monkeypatch.delenv(key, raising=False)

    values = {
        "TRITON_ENABLE_SANITIZER": "1",
        "ENABLE_PRINT_UB_BITS": "true",
        "TRITON_MEMORY_DISPLAY": "1",
        "TRITON_NPU_COMPILER_PATH": "/opt/npuir",
        "NPU_DEVICE_LIMIT": "cube,vector",
        "TRITON_ENABLE_UBTUNER": "run",
        "TRITON_BENCH_METHOD": "npu",
    }
    # Bool knobs are stored in the canonical form written by assignment ("1"),
    # so "true" and "1" share one cache entry.
    expected = dict(values)
    expected["ENABLE_PRINT_UB_BITS"] = "1"
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    for module in (jit_module, compiler_module):
        env_vars = module.get_cache_invalidating_env_vars()
        for key in _COMPILE_ENV_KEYS:
            assert env_vars[key] == expected[key], (module.__name__, key)
        assert "TRITON_BENCH_METHOD" not in env_vars

    # Autotuners additionally hash the benchmark method.
    for module in (upstream_autotuner, ascend_autotuner):
        env_vars = module.get_cache_invalidating_env_vars()
        for key in _COMPILE_ENV_KEYS + _AUTOTUNE_ENV_KEYS:
            assert env_vars[key] == expected[key], (module.__name__, key)


def test_ascend_cache_key_follows_effective_knob(monkeypatch):
    """A knob override changes the cache key even when it does not write the environment."""
    import triton.compiler.compiler as compiler_module
    import triton.runtime.jit as jit_module
    import triton.runtime.autotuner as upstream_autotuner

    for key in ("TRITON_ENABLE_SANITIZER", "TRITON_BENCH_METHOD", "NPU_DEVICE_LIMIT"):
        monkeypatch.delenv(key, raising=False)

    triton.knobs.propagate_env = False
    try:
        triton.knobs.ascend.enable_sanitizer = True
        triton.knobs.ascend.bench_method = "npu"
        assert os.environ.get("TRITON_ENABLE_SANITIZER") is None
        assert os.environ.get("TRITON_BENCH_METHOD") is None

        compile_vars = compiler_module.get_cache_invalidating_env_vars()
        assert compile_vars["TRITON_ENABLE_SANITIZER"] == "1"
        assert "TRITON_BENCH_METHOD" not in compile_vars
        assert "NPU_DEVICE_LIMIT" not in compile_vars
        assert jit_module.get_cache_invalidating_env_vars()["TRITON_ENABLE_SANITIZER"] == "1"

        autotune_vars = upstream_autotuner.get_cache_invalidating_env_vars()
        assert autotune_vars["TRITON_ENABLE_SANITIZER"] == "1"
        assert autotune_vars["TRITON_BENCH_METHOD"] == "npu"

        # The instance value wins over a conflicting environment variable.
        monkeypatch.setenv("TRITON_ENABLE_SANITIZER", "0")
        assert compiler_module.get_cache_invalidating_env_vars()["TRITON_ENABLE_SANITIZER"] == "1"

        # NPU_DEVICE_LIMIT is not a knob and still comes from the environment.
        monkeypatch.setenv("NPU_DEVICE_LIMIT", "cube,vector")
        assert compiler_module.get_cache_invalidating_env_vars()["NPU_DEVICE_LIMIT"] == "cube,vector"
    finally:
        triton.knobs.ascend.enable_sanitizer = triton.knobs.env
        triton.knobs.ascend.bench_method = triton.knobs.env
        triton.knobs.propagate_env = True


def test_effective_knob_misses_shared_compile_cache(fresh_knobs):
    """propagate_env=False must not reuse a binary compiled under another knob value."""
    knobs = fresh_knobs.ascend
    triton.knobs.propagate_env = False
    cache_dir = _fresh_cache_dir("knob-effective-cache-")
    try:
        assert os.environ.get("TRITON_ENABLE_SANITIZER") is None
        with triton.knobs.cache.scope():
            triton.knobs.cache.dir = cache_dir
            _clear_in_process_cache()
            disabled = _add_kernel.warmup(torch.float32, torch.float32, torch.float32, 1024, BLOCK=256, grid=(4, ))

        knobs.enable_sanitizer = True
        assert os.environ.get("TRITON_ENABLE_SANITIZER") is None
        with triton.knobs.cache.scope():
            triton.knobs.cache.dir = cache_dir
            _clear_in_process_cache()
            enabled = _add_kernel.warmup(torch.float32, torch.float32, torch.float32, 1024, BLOCK=256, grid=(4, ))

        assert enabled.metadata.hash != disabled.metadata.hash
        assert enabled.asm["npubin"] != disabled.asm["npubin"]
    finally:
        _clear_in_process_cache()
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Ascend knob effect tests (T36-T38)
#
# T31-T35 prove the group semantics and that consumers read the knob.  These
# tests additionally assert the observable effect: the real bishengir command
# line, autotuner parameters, and the launcher's compile-only branch.
# ---------------------------------------------------------------------------

_COMPILE_FLAGS = {
    "enable_sanitizer": "--enable-sanitizer=true",
    "print_ub_bits": "--enable-print-memory-allocated-size",
    "memory_display": "--enable-memory-display=true",
}

# A5 targets take bishengir's regbase path, whose PlanMemoryRegBase pass has no
# memory-display option (only the membased A2/A3 PlanMemory emits
# memory_info_{aic,aiv}.json). The knob stays readable on A5, but the backend
# does not forward the flag there.
_A5_UNSUPPORTED_FLAGS = {"memory_display"}


def _supported_compile_flags():
    from triton.backends.ascend.utils import is_compile_on_910_95

    arch = triton.runtime.driver.active.get_current_target().arch
    if is_compile_on_910_95(arch):
        return {name: flag for name, flag in _COMPILE_FLAGS.items() if name not in _A5_UNSUPPORTED_FLAGS}
    return dict(_COMPILE_FLAGS)


def test_ascend_compile_flags_follow_knobs(fresh_knobs, monkeypatch, tmp_path):
    """T36: knob-backed flags reach the bishengir command line and the artifact."""
    from triton.backends.ascend import compiler as ascend_compiler

    knobs = fresh_knobs.ascend
    recorded = []
    real_run = ascend_compiler.subprocess.run

    def recording_run(cmd, *args, **kwargs):
        recorded.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(ascend_compiler.subprocess, "run", recording_run)
    # --enable-memory-display makes bishengir write ./memory_info_{aic,aiv}.json
    # relative to the compiler's cwd, so compile from a temp directory to keep
    # the repository clean.
    workdir = tmp_path / "compile-cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    def compile_add(tag):
        cache_dir = _fresh_cache_dir(f"knob-flags-{tag}-")
        recorded.clear()
        try:
            with triton.knobs.cache.scope():
                triton.knobs.cache.dir = cache_dir
                _clear_in_process_cache()
                compiled = _add_kernel.warmup(torch.float32, torch.float32, torch.float32, 1024, BLOCK=256, grid=(4, ))
            return compiled, [list(cmd) for cmd in recorded]
        finally:
            _clear_in_process_cache()
            shutil.rmtree(cache_dir, ignore_errors=True)

    supported = _supported_compile_flags()
    baseline, baseline_cmds = compile_add("off")
    for flag in _COMPILE_FLAGS.values():
        assert all(flag not in cmd for cmd in baseline_cmds)

    for field in _COMPILE_FLAGS:
        setattr(knobs, field, True)
    enabled, enabled_cmds = compile_add("on")
    for flag in supported.values():
        assert any(flag in cmd for cmd in enabled_cmds), flag
    # The flags change code generation, not just the printed command line.
    assert enabled.metadata.hash != baseline.metadata.hash
    assert len(enabled.asm["npubin"]) > len(baseline.asm["npubin"])


def test_ascend_autotune_knobs(fresh_knobs):
    """T37: UB tuner mode and parallel compilation follow the knobs."""
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner
    from triton.backends.ascend.runtime.ubtuner import get_mode_choice

    knobs = fresh_knobs.ascend

    # Unset falls back to the tuner's default mode; only "compile"/"run" are
    # accepted, anything else falls back to the default as well.
    assert get_mode_choice() == "run"
    knobs.enable_ubtuner = "compile"
    assert get_mode_choice() == "compile"
    knobs.enable_ubtuner = "run"
    assert get_mode_choice() == "run"
    knobs.enable_ubtuner = "bogus"
    assert get_mode_choice() == "run"

    def make_tuner():
        return AutoTilingTuner(_add_kernel, _add_kernel.arg_names, [], ["n"], [], [])

    knobs.enable_ubtuner = "compile"
    knobs.autotune_parallel_compile = True
    tuner = make_tuner()
    assert tuner.enable_ubtuner
    assert tuner.compile_parallel

    knobs.enable_ubtuner = ""
    knobs.autotune_parallel_compile = False
    tuner = make_tuner()
    assert not tuner.enable_ubtuner
    assert not tuner.compile_parallel

    knobs.autotune_parallel_compile = True
    assert make_tuner().compile_parallel


def test_ascend_compile_only_skips_launch(fresh_knobs, monkeypatch):
    """T38: compile_only is read at launcher construction and skips the device call."""
    from types import SimpleNamespace

    from triton.backends.ascend import driver

    # Keep launcher construction away from the native build steps; only the
    # compile_only branch __call__ is under test here.
    fake_runtime = SimpleNamespace(create_launcher=lambda *_args, **_kwargs: (lambda *args, **kwargs: 0))

    def fake_npu_utils():
        return SimpleNamespace(get_so_path=lambda: "/tmp/fake-npu-utils.so", get_aivector_core_num=lambda: 1,
                               get_aicore_num=lambda: 1)

    monkeypatch.setattr(driver, "NPUUtils", fake_npu_utils)
    monkeypatch.setattr(driver, "make_launch_spec", lambda _metadata, _npu_utils: SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(driver, "argument_types", lambda _signature, _constants: [])
    monkeypatch.setattr(driver, "get_runtime", lambda _path, _debug=False: (fake_runtime, "/tmp/fake-runtime.so"))
    monkeypatch.setattr(driver, "wrap_handle_tensordesc", lambda fn, _signature: fn)
    monkeypatch.setattr(driver, "get_cache_manager",
                        lambda _key: SimpleNamespace(cache_dir="/tmp/knob-compile-only-cache"))

    src = SimpleNamespace(fn=SimpleNamespace(arg_names=["x"]), signature={0: "*fp32"}, constants={})
    metadata = SimpleNamespace(mix_mode="aiv", shared=0, debug=False)
    knobs = fresh_knobs.ascend

    def make_launcher(compile_only):
        knobs.compile_only = compile_only
        launcher = driver.NPULauncher(src, metadata)
        launched = []
        launcher.launch = lambda *args, **kwargs: launched.append(args) or 1
        launcher(0, 1, 2, 3, 4, {"hash": "knob-test"})
        return launcher, launched

    launcher, launched = make_launcher(True)
    assert launcher.compile_only
    assert launched == []

    launcher, launched = make_launcher(False)
    assert not launcher.compile_only
    assert len(launched) == 1
