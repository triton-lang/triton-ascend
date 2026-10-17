import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.backend("none")


@pytest.fixture
def compiler(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "backend" / "compiler.py"
    spec = importlib.util.spec_from_file_location("triton.backends.ascend._test_metaop_compiler", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_metaop_option_matches_selected_compiler(compiler, monkeypatch, enabled, supported):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["check"] is True
        help_text = "  --enable-optimized-metaop=<bool> - Enable optimized metaop\n" if supported else "  --target=<string>\n"
        return SimpleNamespace(stdout=help_text)

    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/compiler/bin/bishengir-compile", {}))
    monkeypatch.setattr(compiler.subprocess, "run", run)
    expected = [f"--enable-optimized-metaop={str(enabled).lower()}"] if supported else []
    assert compiler._optimized_metaop_compile_options(enabled) == expected
    assert compiler._optimized_metaop_compile_options(enabled) == expected
    assert calls == [["/compiler/bin/bishengir-compile", "--help-hidden"]]


def test_probe_is_cached_per_compiler_path(compiler, monkeypatch):

    def run(argv, **kwargs):
        return SimpleNamespace(stdout="  --enable-optimized-metaop" if argv[0] == "/new/compiler" else "")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    assert not compiler._bishengir_supports_optimized_metaop("/old/compiler")
    assert compiler._bishengir_supports_optimized_metaop("/new/compiler")


def test_probe_does_not_hide_compiler_errors(compiler, monkeypatch):

    def run(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="compiler failed to start")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        compiler._bishengir_supports_optimized_metaop("/broken/compiler")
