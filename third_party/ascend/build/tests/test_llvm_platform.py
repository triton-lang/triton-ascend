import pytest


@pytest.mark.parametrize("system, arch, distro, glibc, override, common, ascend", [
    ("Linux", "x86_64", "ubuntu", "2.35", "", "ubuntu-x64", "ubuntu-x64"),
    ("Linux", "x86_64", "ubuntu", "2.28", "", "almalinux-x64", "almalinux-x64"),
    ("Linux", "x86_64", "almalinux", "2.34", "", "ubuntu-x64", "almalinux-x64"),
    ("Linux", "aarch64", "ubuntu", "2.35", "", "ubuntu-arm64", "ubuntu-arm64"),
    ("Linux", "aarch64", "almalinux", "2.28", "", "almalinux-arm64", "almalinux-arm64"),
    ("Darwin", "arm64", "", "", "", "macos-arm64", "macos-arm64"),
    ("Linux", "riscv64", "ubuntu", "2.35", "", None, None),
    ("Windows", "x86_64", "", "", "", None, None),
    ("Linux", "x86_64", "almalinux", "2.34", "custom-x64", "custom-x64", "custom-x64"),
])
def test_platform_selection_preserves_ascend_abi_exception(ascend_setup, setup_globals, tmp_path, monkeypatch, system,
                                                           arch, distro, glibc, override, common, ascend):
    monkeypatch.setattr(ascend_setup.platform, "system", lambda: system)
    monkeypatch.setattr(ascend_setup.platform, "machine", lambda: arch)
    monkeypatch.setattr(ascend_setup.platform, "libc_ver", lambda: ("glibc", glibc))
    monkeypatch.setattr(ascend_setup, "_is_linux_os", lambda name: name == distro)
    monkeypatch.setattr(setup_globals, "is_linux_os", lambda name: name == distro)
    monkeypatch.setenv("TRITON_LLVM_SYSTEM_SUFFIX", override)
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    # Upstream and Ascend read the same text pin.
    (metadata / "llvm-hash.txt").write_text("a" * 40)
    upstream = setup_globals.get_llvm_package_info()
    assert upstream.sym_name == (f"llvm-{common}" if common else None)
    monkeypatch.setattr(ascend_setup, "_get_llvm_patch_hash", lambda: "12345678")
    info = ascend_setup._get_ascend_llvm_package_info(tmp_path)
    if ascend is None:
        assert info is None
    else:
        assert info["name"] == f"llvm-aaaaaaaa-12345678-{ascend}"
