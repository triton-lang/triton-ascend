import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def helpers(monkeypatch):
    spec = importlib.util.spec_from_file_location("build_helpers_test", REPO / "python/build_helpers.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", ["v3.11.3", "v3.12.0"])
def test_json_package_follows_upstream_text_pin(helpers, tmp_path, monkeypatch, version):
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "json-version.txt").write_text(version + "\n")
    monkeypatch.setattr(helpers, "get_base_dir", lambda: str(tmp_path))
    package = helpers.get_json_package_info()
    assert package.url == f"https://github.com/nlohmann/json/releases/download/{version}/include.zip"
    assert package.syspath_var_name == "JSON_SYSPATH"


@pytest.mark.parametrize("revision", ["a" * 40, "b" * 40])
def test_llvm_package_does_not_require_ascend_patch_checkout(helpers, tmp_path, monkeypatch, revision):
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "llvm-info.json").write_text(json.dumps({"llvm_hash": revision, "patches": ["absent.patch"]}))
    monkeypatch.setattr(helpers, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setenv("TRITON_LLVM_SYSTEM_SUFFIX", "fixture-x64")
    package = helpers.get_llvm_package_info()
    assert package.name == f"llvm-{revision[:8]}-fixture-x64"
    assert package.url.endswith(f"/{package.name}.tar.gz")


def test_ascend_artifact_retains_patch_content_identity(tmp_path, monkeypatch):
    build = REPO / "third_party/ascend/build"
    package = ModuleType("ascend_build_helpers_test")
    package.__path__ = [str(build)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.setup_patch", build / "setup_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "llvm-info.json").write_text(json.dumps({"llvm_hash": "a" * 40}))
    patch_dir = tmp_path / "third_party/ascend/patch"
    patch_dir.mkdir(parents=True)
    patch = b"fixture Ascend patch\n"
    (patch_dir / "llvm_patch_fixture.patch").write_bytes(patch)
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("TRITON_LLVM_SYSTEM_SUFFIX", "fixture-x64")
    info = module._get_ascend_llvm_package_info(tmp_path)
    name = f"llvm-aaaaaaaa-{hashlib.sha256(patch).hexdigest()[:8]}-fixture-x64"
    assert info["name"] == name
    assert info["url"] == f"https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz"


@pytest.mark.parametrize("value, expected", [("ON", True), ("1", True), ("yes", True), ("OFF", False), ("0", False)])
def test_shared_environment_flags_preserve_build_options(helpers, monkeypatch, value, expected):
    monkeypatch.setenv("TRITON_BUILD_WITH_CCACHE", value)
    assert helpers.check_env_flag("TRITON_BUILD_WITH_CCACHE") is expected
