import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]


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
def test_common_platform_selection_preserves_ascend_abi_exception(tmp_path, monkeypatch, system, arch, distro, glibc,
                                                                  override, common, ascend):
    helper = importlib.import_module("python.build_helpers")
    build = REPO / "third_party/ascend/build"
    package = ModuleType("ascend_platform_test")
    package.__path__ = [str(build)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.setup_patch", build / "setup_patch.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(helper.platform, "system", lambda: system)
    monkeypatch.setattr(helper.platform, "machine", lambda: arch)
    monkeypatch.setattr(helper.platform, "libc_ver", lambda: ("glibc", glibc))
    monkeypatch.setattr(helper, "is_linux_os", lambda name: name == distro)
    monkeypatch.setenv("TRITON_LLVM_SYSTEM_SUFFIX", override)
    spec.loader.exec_module(module)
    assert helper.get_llvm_system_suffix() == common
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "llvm-info.json").write_text(json.dumps({"llvm_hash": "a" * 40}))
    monkeypatch.setattr(module, "_get_llvm_patch_hash", lambda: "12345678")
    info = module._get_ascend_llvm_package_info(tmp_path)
    if ascend is None:
        assert info is None
    else:
        assert info["name"] == f"llvm-aaaaaaaa-12345678-{ascend}"
