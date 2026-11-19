"""Regression coverage for vendored sources and real Git checkouts."""
import json

import pytest


def configure(module, root, monkeypatch, revision="a" * 40):
    metadata = root / "cmake/triton-distributed-info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({
            "schema_version": 1, "repository": "https://gitcode.com/Ascend/Triton-distributed-ascend.git", "revision":
            revision
        }))
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


def test_invalid_distributed_revision_fails_before_git(ascend_setup, tmp_path, monkeypatch):
    configure(ascend_setup, tmp_path, monkeypatch, revision="master")
    with pytest.raises(ValueError, match="triton-distributed-info.json.*full lowercase commit SHA"):
        ascend_setup._ensure_distributed_submodule()
