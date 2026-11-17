import json
import subprocess

import pytest


@pytest.mark.parametrize("source_checkout", [True, False])
def test_distributed_clone_uses_metadata_with_or_without_parent_git(ascend_setup, tmp_path, monkeypatch,
                                                                  source_checkout):
    remote = tmp_path / "remote"
    subprocess.run(["git", "init", str(remote)], check=True, capture_output=True)
    (remote / "CMakeLists.txt").write_text("# distributed fixture\n")
    subprocess.run(["git", "-C", str(remote), "add", "."], check=True)
    subprocess.run(["git", "-C", str(remote), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "fixture"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(remote), "rev-parse", "HEAD"], text=True).strip()
    root = tmp_path / "source"
    (root / "cmake").mkdir(parents=True)
    (root / "third_party/ascend").mkdir(parents=True)
    if source_checkout:
        subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    url = "https://example.invalid/distributed.git"
    (root / "cmake/triton-distributed-info.json").write_text(json.dumps({
        "schema_version": 1, "repository": url, "revision": sha,
    }))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{remote.as_uri()}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", url)
    monkeypatch.setenv("TRITON_BUILD_TD", "ON")
    monkeypatch.setattr(ascend_setup, "_REPO_ROOT", root)
    ascend_setup._ensure_distributed_submodule()
    destination = root / "third_party/ascend/Triton-distributed-ascend"
    assert (destination / "CMakeLists.txt").read_text() == "# distributed fixture\n"
    assert subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip() == sha
