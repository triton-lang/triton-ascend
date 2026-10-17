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

"""Import/patch contracts with isolated stub dependencies and optional native smoke tests.

The host tests apply the shipped Ascend patch in a temporary source tree, then
execute the patched package initializer and real Config implementation.
Run with --confcutdir pointing here to avoid the NPU allocation fixture.
"""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _definition(path, name):
    source = path.read_text()
    node = next(node for node in ast.parse(source).body if getattr(node, "name", None) == name)
    return ast.get_source_segment(source, node) + "\n"


def _write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


def _patched_triton_init(tmp_path):
    source_root = tmp_path / "source"
    source_path = "python/triton/__init__.py"
    _write(source_root, source_path, (ROOT / source_path).read_text())
    patch_path = ROOT / "third_party/ascend/patch/triton-ascend-3.6.0.patch"
    command = ["git", "apply", "--include=" + source_path, str(patch_path)]
    # A local build may already have applied the patch. Normalize that copy
    # back to the baseline, then always exercise the shipped forward patch.
    reverse_check = subprocess.run(command + ["--reverse", "--check"], cwd=source_root,
                                   capture_output=True, text=True, timeout=30)
    if reverse_check.returncode == 0:
        subprocess.run(command + ["--reverse"], cwd=source_root, check=True, capture_output=True, timeout=30)
    result = subprocess.run(command, cwd=source_root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    source = (source_root / source_path).read_text()
    assert "def _initialize_backend_runtime():" in source
    return source


def test_patch_restore_manifest_contains_initializer():
    namespace = {}
    exec(_definition(ROOT / "third_party/ascend/build/setup_patch.py", "get_triton_ascend_patch_file"), namespace)
    patch_files, dev_patch_files = namespace["get_triton_ascend_patch_file"]()
    assert "python/triton/__init__.py" in patch_files
    assert "python/triton/__init__.py" not in dev_patch_files


@pytest.fixture
def package(tmp_path):
    _write(tmp_path, "triton/__init__.py", _patched_triton_init(tmp_path))
    _write(tmp_path, "triton/backends/ascend/runtime/__init__.py",
           (ROOT / "third_party/ascend/backend/runtime/__init__.py").read_text())
    _write(tmp_path, "triton/backends/ascend/__init__.py", "")
    _write(tmp_path, "triton/backends/__init__.py", """
        import os
        from . import ascend
        backends = {"ascend": object()} if os.environ.get("TEST_ASCEND", "1") == "1" else {}
    """)
    config_source = _definition(ROOT / "python/triton/runtime/autotuner.py", "Config")
    _write(tmp_path, "triton/runtime/autotuner.py", config_source + textwrap.dedent("""
        from triton.backends import backends
        class Autotuner:
            pass
        class Heuristics:
            pass
        def autotune(*args, **kwargs):
            return "upstream"
        heuristics = autotune
        original_autotune = autotune
        original_config_init = Config.__init__
        native_env_vars = {"EXISTING_OPTION": "1"}
        env_calls = 0
        def get_cache_invalidating_env_vars():
            global env_calls
            env_calls += 1
            return native_env_vars
    """))
    _write(tmp_path, "triton/runtime/__init__.py", """
        from .autotuner import Autotuner, Config, autotune, heuristics
        from .jit import JITFunction, KernelInterface, reinterpret, TensorWrapper, MockTensor
        from .errors import OutOfResources, InterpreterError
        class Driver:
            @property
            def active(self):
                raise AssertionError("Import must not initialize the active driver")
        driver = Driver()
    """)
    _write(tmp_path, "triton/runtime/jit.py", """
        def constexpr_function(fn):
            return fn
        jit = constexpr_function
        reinterpret = constexpr_function
        JITFunction = KernelInterface = TensorWrapper = MockTensor = object
    """)
    _write(tmp_path, "triton/runtime/_async_compile.py", "AsyncCompileMode = FutureKernel = object\n")
    _write(tmp_path, "triton/runtime/_allocation.py", "set_allocator = object\n")
    _write(tmp_path, "triton/runtime/errors.py", "OutOfResources = InterpreterError = RuntimeError\n")
    _write(tmp_path, "triton/errors.py", "TritonError = RuntimeError\n")
    _write(tmp_path, "triton/compiler/__init__.py", "compile = object\nCompilationError = RuntimeError\n")
    _write(tmp_path, "triton/language/__init__.py", "from types import SimpleNamespace\ncore = SimpleNamespace(must_use_result=object)\n")
    _write(tmp_path, "triton/testing.py", "")
    _write(tmp_path, "triton/tools/__init__.py", "")
    _write(tmp_path, "triton/backends/ascend/utils.py",
           _definition(ROOT / "third_party/ascend/backend/utils.py", "_multibuffer_mode_to_tuple"))
    _write(tmp_path, "triton/backends/ascend/runtime/autotuner.py", """
        # These imports require the complete top-level Triton API.
        from triton import Config, jit, cdiv, next_power_of_2
        from triton.runtime.autotuner import Autotuner, get_cache_invalidating_env_vars
        class AutoTilingTuner(Autotuner):
            pass
        def autotune(*args, **kwargs):
            return AutoTilingTuner()
        get_max_configs = max_autotune = autotune
    """)
    return tmp_path


def _run(package, source, *, ascend=True, succeeds=True):
    env = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE="1", TEST_ASCEND=str(int(ascend)))
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], cwd=package,
                            env=env, capture_output=True, text=True, timeout=30)
    if succeeds:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


@pytest.mark.parametrize("first_import", [
    "import triton", "from triton import autotune as saved",
    "from triton.runtime import autotune as saved",
    "from triton.runtime.autotuner import autotune as saved",
    "import triton.backends.ascend.runtime",
])
def test_import_installs_all_aliases(package, first_import):
    _run(package, first_import + "\n" + textwrap.dedent("""
        import triton
        import triton.runtime.autotuner as upstream
        from triton.backends.ascend.runtime import autotuner as ascend
        assert triton.autotune is triton.runtime.autotune is upstream.autotune is ascend.autotune
        assert globals().get("saved", ascend.autotune) is ascend.autotune
        assert upstream.Autotuner is triton.runtime.Autotuner is ascend.Autotuner
        assert isinstance(triton.autotune(), ascend.AutoTilingTuner)
        assert triton.Config is upstream.Config is ascend.Config
    """))


def test_reinstall_and_reload_do_not_stack_wrappers(package):
    _run(package, """
        import importlib
        import triton
        import triton.runtime.autotuner as upstream
        from triton.backends.ascend import runtime as ascend_runtime
        init = triton.Config.__init__
        env_fn = upstream.get_cache_invalidating_env_vars
        for _ in range(3):
            ascend_runtime.install_runtime_patches()
            importlib.reload(ascend_runtime)
            triton.autotune = upstream.original_autotune
            importlib.reload(triton)
            assert triton.autotune is ascend_runtime.autotuner.autotune
            assert triton.Config.__init__ is init
            assert upstream.get_cache_invalidating_env_vars is env_fn
        env_fn()
        assert upstream.env_calls == 1
    """)


def test_config_mode_and_cache_round_trip(package):
    _run(package, """
        import json
        import triton
        assert triton.Config({}).num_stages == 3
        assert triton.Config({}, num_stages=1).num_stages == 1
        kwargs = {"multibuffer_mode": {"ub": 2}}
        config = triton.Config(kwargs, ubtune_cfg={"multibuffer": False})
        assert config.num_stages is None
        assert config.kwargs["multibuffer_mode"] == (("ub", 2),)
        assert kwargs["multibuffer_mode"] == {"ub": 2}
        restored = triton.Config(**json.loads(json.dumps(config.__dict__)))
        assert restored.kwargs == config.kwargs
        assert restored.num_stages is None
        assert restored.ubtune_cfg == config.ubtune_cfg
        try:
            triton.Config(kwargs, num_stages=2)
        except ValueError:
            pass
        else:
            raise AssertionError("Conflicting mode/stages must fail")
    """)


def test_cache_environment_tracks_mode_without_mutating_native_result(package):
    _run(package, """
        import os
        import triton.runtime.autotuner as upstream
        from triton.backends.ascend.runtime import autotuner as ascend
        assert upstream.get_cache_invalidating_env_vars is ascend.get_cache_invalidating_env_vars
        for mode in ("compile", "run", ""):
            os.environ["TRITON_ENABLE_UBTUNER"] = mode
            expected = {"EXISTING_OPTION": "1"}
            if mode:
                expected["TRITON_ENABLE_UBTUNER"] = mode
            assert upstream.get_cache_invalidating_env_vars() == expected
            assert ascend.get_cache_invalidating_env_vars() == expected
            assert upstream.native_env_vars == {"EXISTING_OPTION": "1"}
    """)


def test_without_ascend_keeps_upstream(package):
    _run(package, """
        import sys
        import triton
        import triton.runtime.autotuner as upstream
        assert triton.autotune is upstream.original_autotune
        assert triton.Config.__init__ is upstream.original_config_init
        assert "triton.backends.ascend.runtime" not in sys.modules
    """, ascend=False)


def test_dependency_errors_are_not_silenced(package):
    _write(package, "triton/backends/ascend/runtime/autotuner.py", "raise ImportError('missing-autotuner-dependency')\n")
    result = _run(package, "import triton", succeeds=False)
    assert "missing-autotuner-dependency" in result.stderr


@pytest.mark.parametrize("imports", [
    "import triton", "import torch; import triton", "import torch_npu; import triton",
])
def test_installed_ascend_import_order(imports):
    # Real integration smoke tests are skipped on source-only host machines.
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Requires an installed Triton-Ascend and torch_npu environment")
    source = imports + "\n" + textwrap.dedent("""
        from triton import autotune as saved
        import triton.runtime.autotuner as upstream
        from triton.backends.ascend.runtime import autotuner as ascend
        assert saved is triton.autotune is triton.runtime.autotune is upstream.autotune is ascend.autotune
        assert issubclass(ascend.AutoTilingTuner, upstream.Autotuner)
        assert triton.Config.__init__._ascend_config_patch_applied
    """)
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
