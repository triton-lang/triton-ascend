"""Regression coverage for vendored sources and real Git checkouts."""
import json
import subprocess

import pytest


def configure(module, root, monkeypatch, repository="https://example.invalid/distributed.git", revision="a" * 40):
    metadata = root / "cmake/triton-distributed-info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"schema_version": 1, "repository": repository, "revision": revision}))
    destination = root / "third_party/ascend/Triton-distributed-ascend"
    destination.mkdir(parents=True)
    monkeypatch.setattr(module, "_REPO_ROOT", root)
    monkeypatch.setenv("TRITON_BUILD_TD", "ON")
    return destination


def test_vendored_distributed_sources_need_neither_git_nor_network(ascend_setup, tmp_path, monkeypatch):
    destination = configure(ascend_setup, tmp_path, monkeypatch)
    (destination / "CMakeLists.txt").write_text("# vendored source\n")
    # Any attempted Git/network operation is a regression for a source archive.
    monkeypatch.setattr(ascend_setup, "_add_git_safe_dir", lambda *args: pytest.fail("Git on vendored sources"))
    ascend_setup._ensure_distributed_submodule()
    assert (destination / "CMakeLists.txt").read_text() == "# vendored source\n"


def test_existing_distributed_checkout_is_offline_when_already_pinned(ascend_setup, tmp_path, monkeypatch):
    destination = configure(ascend_setup, tmp_path, monkeypatch)
    subprocess.run(["git", "init", str(destination)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(destination), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(destination), "config", "user.name", "Test"], check=True)
    (destination / "CMakeLists.txt").write_text("# pinned source\n")
    subprocess.run(["git", "-C", str(destination), "add", "."], check=True)
    subprocess.run(["git", "-C", str(destination), "commit", "-m", "fixture"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    path = tmp_path / "cmake/triton-distributed-info.json"
    data = json.loads(path.read_text())
    data["revision"] = sha
    path.write_text(json.dumps(data))
    subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", data["repository"]], check=True)
    monkeypatch.setattr(ascend_setup, "_git_check_call_with_retry", lambda *args, **kwargs: pytest.fail("network"))
    ascend_setup._ensure_distributed_submodule()
    assert (destination / "CMakeLists.txt").read_text() == "# pinned source\n"
