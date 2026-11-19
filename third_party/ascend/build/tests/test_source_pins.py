import subprocess

import pytest

from third_party.ascend.build import setup_patch


def git(path, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
        cwd=path,
        text=True,
    ).strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "commit", "--allow-empty", "-qm", "initial")
    return root


def test_worktree_enforces_distributed_pin(repo, tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    git(repo, "worktree", "add", "-q", "--detach", str(worktree))
    monkeypatch.setattr(setup_patch, "_REPO_ROOT", worktree)
    monkeypatch.setenv("TRITON_BUILD_TD", "ON")
    destination = worktree / "third_party/ascend/Triton-distributed-ascend"
    destination.mkdir(parents=True)
    revision = "8c1dae1acbb4bcf99c3e473c8ed876f2cd42ba35"
    metadata = worktree / "cmake/triton-distributed-info.json"
    metadata.parent.mkdir()
    metadata.write_text('{"revision": "' + revision + '"}', encoding="utf-8")
    monkeypatch.setattr(setup_patch, "_add_git_safe_dir", lambda path: None)
    monkeypatch.setattr(setup_patch, "_git_check_call_with_retry", lambda *args, **kwargs: None)
    operations = []
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, **kwargs: operations.append((cmd, kwargs["cwd"])))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, revision))
    setup_patch._ensure_distributed_submodule()
    assert operations == [(["git", "checkout", revision], destination)]


@pytest.mark.parametrize("wrong_revision", [False, True])
def test_default_npuir_build_checks_gitlink(repo, tmp_path, monkeypatch, wrong_revision):
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    git(dependency, "init", "-q")
    git(dependency, "commit", "--allow-empty", "-qm", "pinned")
    path = "third_party/ascend/AscendNPU-IR"
    git(repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(dependency), path)
    git(repo, "commit", "-qm", "pin submodule")
    if wrong_revision:
        git(repo / path, "commit", "--allow-empty", "-qm", "different revision")
    monkeypatch.setattr(setup_patch, "_REPO_ROOT", repo)
    monkeypatch.chdir(repo)
    patched = []
    monkeypatch.setattr(setup_patch, "_apply_npuir_patch", lambda: patched.append(True))
    if wrong_revision:
        with pytest.raises(RuntimeError, match="AscendNPU-IR HEAD"):
            setup_patch._apply_triton_ascend_patch()
        assert not patched
    else:
        setup_patch._apply_triton_ascend_patch()
        assert patched == [True]
