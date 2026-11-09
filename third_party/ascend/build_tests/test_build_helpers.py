import hashlib

import pytest


@pytest.mark.parametrize("version", ["v3.11.3", "v3.12.0"])
def test_json_package_follows_upstream_text_pin(ascend_setup, setup_globals, tmp_path, version):
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "json-version.txt").write_text(version + "\n")
    ascend_setup.patch_module(setup_globals)
    package = setup_globals.get_json_package_info()
    assert package.url == f"https://github.com/nlohmann/json/releases/download/{version}/include.zip"
    assert package.syspath_var_name == "JSON_SYSPATH"


@pytest.mark.parametrize("revision", ["a" * 40, "b" * 40])
def test_llvm_package_does_not_require_ascend_patch_checkout(ascend_setup, tmp_path, monkeypatch, revision):
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "llvm-hash.txt").write_text(revision)
    monkeypatch.setattr(ascend_setup, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("TRITON_LLVM_SYSTEM_SUFFIX", "fixture-x64")
    package = ascend_setup._get_ascend_llvm_package_info(tmp_path)
    assert package["name"] == f"llvm-{revision[:8]}-00000000-fixture-x64"
    assert package["url"].endswith(f"/{package['name']}.tar.gz")


def test_ascend_artifact_retains_patch_content_identity(ascend_setup, tmp_path, monkeypatch):
    module = ascend_setup
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "llvm-hash.txt").write_text("a" * 40)
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
def test_environment_flags_preserve_build_options(ascend_setup, setup_globals, monkeypatch, value, expected):
    monkeypatch.setenv("TRITON_BUILD_WITH_CCACHE", value)
    assert ascend_setup.check_env_flag("TRITON_BUILD_WITH_CCACHE") is expected
    assert setup_globals.check_env_flag("TRITON_BUILD_WITH_CCACHE") is expected
