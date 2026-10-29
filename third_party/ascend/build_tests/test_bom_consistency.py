import importlib.util
import json
from pathlib import Path
import shutil

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def checker(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("bom_checker_test", REPO / "scripts/check_bom_consistency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for relative in ["requirements.txt", "docker/Dockerfile", "docs/en/community/release_policy.md"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    shutil.copytree(REPO / "cmake", tmp_path / "cmake")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "DEPS_DIR", tmp_path / "cmake")
    return module


def test_unknown_checkout_prefix_stays_invalid(checker):
    assert not checker._glob_exists("wrong/requirements.txt")
    assert checker._glob_exists("pr/requirements.txt")


@pytest.mark.parametrize("mutation", ["copied-json", "invalid-checksum", "unsafe-file", "missing-arch"])
def test_toolchain_rejects_invalid_inputs(checker, mutation):
    path = checker.DEPS_DIR / "bisheng-info.json"
    data = json.loads(path.read_text())
    if mutation == "copied-json":
        data["json"] = "v3.11.3"
    elif mutation == "invalid-checksum":
        data["bisheng"]["x86_64"]["sha256"] = "bad"
    elif mutation == "unsafe-file":
        data["bisheng"]["x86_64"]["file"] = "../outside.run"
    else:
        del data["bisheng"]["aarch64"]
    path.write_text(json.dumps(data))
    checker.check_toolchain()
    assert checker.errors


@pytest.mark.parametrize("version", ["v3.11.3", "main", "v3.11.3\nv3.12.0"])
def test_audit_validates_the_upstream_json_text_pin(checker, version):
    (checker.DEPS_DIR / "json-version.txt").write_text(version)
    checker.check_toolchain()
    assert bool(checker.errors) is (version != "v3.11.3")


@pytest.mark.parametrize("field", ["llvm", "patch"])
def test_release_policy_cannot_silently_drift_llvm_or_patch(checker, field):
    path = checker.REPO_ROOT / "docs/en/community/release_policy.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("f6ded0b", "abcdef0") if field == "llvm" else text.replace("llvm_patch_f6ded0b.patch",
                                                                                   "llvm_patch_bad.patch")
    path.write_text(text, encoding="utf-8")
    checker.check_release_llvm()
    assert checker.errors
