import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def write_patch(path, name, old, new):
    path.write_text(
        f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -1,2 +1,2 @@\n-{old}\n+{new}\n context\n",
        newline="\n")


@pytest.fixture
def sources(monkeypatch, tmp_path):
    build_path = Path(__file__).resolve().parents[1] / "build"
    package = ModuleType("ascend_build_under_test")
    package.__path__ = [str(build_path)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.setup_patch", build_path / "setup_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "core.autocrlf", "false")
    git(tmp_path, "config", "core.hooksPath", str(tmp_path / "no-hooks"))
    source = tmp_path / "test.cpp"
    source.write_text("original\ncontext\n", newline="\n")
    (tmp_path / "dev.py").write_text("original\ncontext\n", newline="\n")
    git(tmp_path, "add", "test.cpp", "dev.py")
    git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
    patch_dir = tmp_path / "third_party" / "ascend" / "patch"
    patch_dir.mkdir(parents=True)
    write_patch(patch_dir / "triton-ascend-3.6.0.patch", "test.cpp", "original", "patched")
    write_patch(patch_dir / "triton-ascend-dev-3.6.0.patch", "dev.py", "original", "development")
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_is_dev_mode", lambda: True)
    monkeypatch.setattr(module, "get_triton_ascend_patch_file", lambda: (["test.cpp"], ["dev.py"]))
    monkeypatch.chdir(tmp_path)
    return module, source, patch_dir


def test_identical_patches_preserve_timestamps(sources):
    module, source, _ = sources
    module._apply_triton_ascend_patch()
    paths = [source, source.with_name("dev.py")]
    for path in paths:
        os.utime(path, ns=(1000000000000000000, 1000000000000000000))
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]

    module._apply_triton_ascend_patch()

    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before


def test_changed_patch_updates_content_and_timestamp(sources):
    module, source, patch_dir = sources
    module._apply_triton_ascend_patch()
    os.utime(source, ns=(1000000000000000000, 1000000000000000000))
    write_patch(patch_dir / "triton-ascend-3.6.0.patch", "test.cpp", "original", "new patch")

    module._apply_triton_ascend_patch()

    assert source.read_text() == "new patch\ncontext\n"
    assert source.stat().st_mtime_ns != 1000000000000000000


def test_changed_head_is_not_hidden(sources):
    module, source, patch_dir = sources
    source.write_text("new base\ncontext\n", newline="\n")
    write_patch(patch_dir / "triton-ascend-3.6.0.patch", "test.cpp", "new base", "patched")
    git(source.parent, "add", "test.cpp")
    git(source.parent, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "upstream")
    os.utime(source, ns=(1000000000000000000, 1000000000000000000))

    module._apply_triton_ascend_patch()

    assert source.read_text() == "patched\ncontext\n"
    assert source.stat().st_mtime_ns != 1000000000000000000


def test_patch_does_not_dirty_noop_ninja_build(sources):
    if shutil.which("ninja") is None:
        pytest.skip("Ninja is required to check incremental build scope")
    module, source, _ = sources
    module._apply_triton_ascend_patch()
    os.utime(source, ns=(1000000000000000000, 1000000000000000000))
    python = Path(sys.executable).as_posix().replace("$", "$$")
    source.with_name("build.ninja").write_text(
        f'rule copy\n  command = "{python}" -c "import shutil; shutil.copyfile(\'test.cpp\', \'test.o\')"\n'
        "build test.o: copy test.cpp\n")
    subprocess.run(["ninja"], check=True, capture_output=True)

    module._apply_triton_ascend_patch()

    result = subprocess.run(["ninja", "-n", "-d", "explain"], check=True, capture_output=True, text=True)
    assert "no work to do" in result.stdout
    assert source.with_name("test.o").read_bytes() == source.read_bytes()
