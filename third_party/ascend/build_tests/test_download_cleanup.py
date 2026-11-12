import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("mode", ["retry-success", "retry-failure", "missing-curl", "consumer-error"])
def test_download_stream_cleans_temporary_files_in_all_exit_paths(tmp_path, monkeypatch, mode):
    build = REPO / "third_party/ascend/build"
    package = ModuleType("ascend_download_test")
    package.__path__ = [str(build)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.setup_patch", build / "setup_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    attempts = []

    def curl(command, **kwargs):
        attempts.append(command)
        if mode == "missing-curl":
            raise FileNotFoundError("curl")
        path = Path(command[command.index("-o") + 1])
        path.write_bytes(b"downloaded archive")
        return SimpleNamespace(
            returncode=0 if mode in {"retry-success", "consumer-error"} and len(attempts) == 2 else 28,
            stderr=b"timeout")

    monkeypatch.setattr(subprocess, "run", curl)
    if mode == "retry-success":
        with module._robust_open_url("https://example.invalid/archive", retries=2) as stream:
            assert stream.read() == b"downloaded archive"
        assert stream.closed
        assert len(attempts) == 2
    elif mode == "consumer-error":
        with pytest.raises(ValueError, match="extraction failed"):
            with module._robust_open_url("https://example.invalid/archive", retries=2) as stream:
                raise ValueError("extraction failed")
        assert stream.closed
    else:
        error = FileNotFoundError if mode == "missing-curl" else RuntimeError
        with pytest.raises(error):
            with module._robust_open_url("https://example.invalid/archive", retries=2):
                pytest.fail("A failed download must not yield a stream")
    assert not list(tmp_path.iterdir())
