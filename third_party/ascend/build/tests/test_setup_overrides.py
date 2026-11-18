import hashlib

import pytest


@pytest.mark.parametrize("version", ["v3.11.3", "v3.12.0"])
def test_ascend_hook_reads_json_tag_without_replacing_upstream_definitions(ascend_setup, setup_globals, tmp_path,
                                                                           version):
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "json-version.txt").write_text(version + "\n")
    original = setup_globals.get_json_package_info
    ascend_setup.patch_module(setup_globals)
    package = setup_globals.get_json_package_info()
    assert isinstance(package, setup_globals.Package)
    assert package.url == f"https://github.com/nlohmann/json/releases/download/{version}/include.zip"
    assert package.syspath_var_name == "JSON_SYSPATH"
    assert original().url.endswith("/v3.11.3/include.zip")


def test_ascend_hook_selects_patched_llvm_from_shared_text_pin(ascend_setup, setup_globals, tmp_path, monkeypatch):
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "llvm-hash.txt").write_text("a" * 40)
    patch_dir = tmp_path / "third_party/ascend/patch"
    patch_dir.mkdir(parents=True)
    patch = b"Ascend patch\n"
    (patch_dir / "llvm_patch_fixture.patch").write_bytes(patch)
    monkeypatch.setattr(ascend_setup, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("TRITON_LLVM_SYSTEM_SUFFIX", "fixture-x64")
    ascend_setup.patch_module(setup_globals)
    package = setup_globals.get_llvm_package_info()
    assert isinstance(package, setup_globals.Package)
    assert package.name == f"llvm-aaaaaaaa-{hashlib.sha256(patch).hexdigest()[:8]}-fixture-x64"
    assert package.url.endswith(f"/{package.name}.tar.gz")
    assert not (metadata / "llvm-info.json").exists()


def test_ascend_hook_retains_unsupported_platform_fallback(ascend_setup, setup_globals, monkeypatch):
    monkeypatch.delenv("TRITON_LLVM_SYSTEM_SUFFIX", raising=False)
    monkeypatch.setattr(ascend_setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ascend_setup.platform, "machine", lambda: "riscv64")
    ascend_setup.patch_module(setup_globals)
    package = setup_globals.get_llvm_package_info()
    assert package.name == "LLVM-C.lib"
    assert package.url == ""
