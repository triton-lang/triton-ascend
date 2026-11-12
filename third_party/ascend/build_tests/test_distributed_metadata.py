import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("source_checkout", [True, False])
def test_distributed_preparation_uses_metadata_even_for_source_distributions(tmp_path, monkeypatch, source_checkout):
    package = ModuleType("ascend_distributed_test")
    package.__path__ = [str(REPO / "third_party/ascend/build")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.setup_patch",
                                                  REPO / "third_party/ascend/build/setup_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = tmp_path / "cmake/triton-distributed-info.json"
    metadata.parent.mkdir()
    metadata.write_text(
        json.dumps({"schema_version": 1, "repository": "https://example.invalid/distributed.git", "revision":
                    "a" * 40}))
    parent = tmp_path / "third_party/ascend"
    parent.mkdir(parents=True)
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("TRITON_BUILD_TD", "ON")
    monkeypatch.setattr(module, "_is_git_repo", lambda: source_checkout)
    monkeypatch.setattr(module, "_add_git_safe_dir", lambda *args: None)
    calls = []

    def call(command, **kwargs):
        calls.append(command)
        if command[1] == "clone":
            (parent / "Triton-distributed-ascend").mkdir()

    monkeypatch.setattr(module, "_git_check_call_with_retry", call)
    monkeypatch.setattr(module.subprocess, "check_call", call)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="a" * 40))
    module._ensure_distributed_submodule()
    assert any("https://example.invalid/distributed.git" in command for command in calls)
    assert ["git", "checkout", "a" * 40] in calls
