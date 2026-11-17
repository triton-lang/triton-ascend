"""Exercise source selection using local Git repositories, without Gitcode."""
import importlib.util
import json
from pathlib import Path
import subprocess
import os
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
READER = REPO / "third_party/ascend/build/npuir_source.py"


def load_reader():
    spec = importlib.util.spec_from_file_location("npuir_source_test", READER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_root(tmp_path, ref="stable"):
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake/npuir-info.json").write_text(
        json.dumps({
            "schema_version": 1,
            "submodule": "third_party/ascend/AscendNPU-IR",
            "ci_ref": ref,
        }))
    (tmp_path / ".gitmodules").write_text('[submodule "third_party/ascend/AscendNPU-IR"]\n'
                                          '\tpath = third_party/ascend/AscendNPU-IR\n'
                                          '\turl = https://example.invalid/npuir.git\n')
    return tmp_path


def test_npuir_repository_comes_from_gitmodules_and_ref_from_metadata(tmp_path):
    module = load_reader()
    root = fixture_root(tmp_path, "release/test")
    assert module.read_source(root) == ("https://example.invalid/npuir.git", "release/test")
    assert module.read_source(root, "b" * 40) == ("https://example.invalid/npuir.git", "b" * 40)


@pytest.mark.parametrize("ref", ["", "--upload-pack=bad", "stable\nother", "refs/heads/stable", "../stable"])
def test_invalid_source_ref_fails_before_network(tmp_path, ref):
    module = load_reader()
    root = fixture_root(tmp_path, ref)
    with pytest.raises(ValueError):
        module.read_source(root)


def test_exact_sha_is_resolved_without_network():
    module = load_reader()
    assert module.resolve_revision("https://example.invalid/unreachable.git", "b" * 40) == "b" * 40


def test_branch_resolution_observes_a_real_git_remote(tmp_path):
    module = load_reader()
    remote = tmp_path / "remote"
    subprocess.run(["git", "init", "-b", "stable", str(remote)], check=True, capture_output=True)
    subprocess.run([
        "git", "-C",
        str(remote), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m",
        "fixture"
    ], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(remote), "rev-parse", "HEAD"], text=True).strip()
    assert module.resolve_revision(str(remote), "stable") == sha


def test_fetch_script_uses_the_resolved_sha_even_after_branch_moves(tmp_path):
    remote = tmp_path / "remote"
    subprocess.run(["git", "init", "-b", "stable", str(remote)], check=True, capture_output=True)
    (remote / "CMakeLists.txt").write_text("first\n")
    subprocess.run(["git", "-C", str(remote), "add", "."], check=True)
    commit = [
        "git", "-C",
        str(remote), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-am", "fixture"
    ]
    subprocess.run(commit, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(remote), "rev-parse", "HEAD"], text=True).strip()
    (remote / "CMakeLists.txt").write_text("second\n")
    subprocess.run(commit, check=True, capture_output=True)
    root = tmp_path / "checkout"
    root.mkdir()
    fixture_root(root)
    reader = root / "third_party/ascend/build/npuir_source.py"
    reader.parent.mkdir(parents=True)
    shutil.copyfile(READER, reader)
    script = root / ".github/workflows/build/fetch-npuir-stable.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(REPO / ".github/workflows/build/fetch-npuir-stable.sh", script)
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if bash is None or shutil.which("python3") is None:
        pytest.skip("Bash and python3 are required")
    env = {
        **os.environ, "NPUIR_REPO_BRANCH": sha, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0":
        f"url.{remote.as_uri()}.insteadOf", "GIT_CONFIG_VALUE_0": "https://example.invalid/npuir.git"
    }
    # Windows may expose a Microsoft Store alias rather than a real python3.
    command = 'python3() { "$TEST_PYTHON" "$@"; }; export -f python3; bash "$1" npuir --skip-nested'
    env["TEST_PYTHON"] = Path(sys.executable).as_posix()
    result = subprocess.run([bash, "-c", command, "test", script.as_posix()], cwd=root, env=env, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == sha
    assert (root / "npuir/CMakeLists.txt").read_text() == "first\n"
