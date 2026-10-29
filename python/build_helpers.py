import json
import os
import platform
from dataclasses import dataclass
from typing import Optional
import sysconfig
import sys
from pathlib import Path


def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    return Path(get_base_dir()) / "build" / dir_name


def get_cmake_dir():
    cmake_dir = os.getenv("TRITON_BUILD_DIR", default=_get_cmake_dir())
    cmake_dir = Path(cmake_dir)
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


# Keep common build helpers here, following Triton and Intel. Ascend-specific
# artifact selection stays in third_party/ascend/build/setup_patch.py.
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


@dataclass
class Package:
    package: str
    name: str
    url: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str
    sym_name: Optional[str] = None


def get_json_package_info():
    json_version_path = os.path.join(get_base_dir(), "cmake", "json-version.txt")
    with open(json_version_path, "r") as json_version_file:
        version = json_version_file.read().strip()
    url = f"https://github.com/nlohmann/json/releases/download/{version}/include.zip"
    return Package("json", "", url, "JSON_INCLUDE_DIR", "", "JSON_SYSPATH")


def is_linux_os(id):
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r") as f:
            os_release_content = f.read()
            return f'ID="{id}"' in os_release_content
    return False


def get_llvm_system_suffix():
    if env_system_suffix := os.environ.get("TRITON_LLVM_SYSTEM_SUFFIX"):
        return env_system_suffix
    system = platform.system()
    machine = platform.machine()
    arch = {"x86_64": "x64", "aarch64": "arm64"}.get(machine, machine)
    if system == "Darwin":
        return f"macos-{arch}"
    if system == "Linux":
        if arch == "arm64":
            return "almalinux-arm64" if is_linux_os("almalinux") else "ubuntu-arm64"
        if arch == "x64":
            major, minor, *_ = map(int, platform.libc_ver()[1].split("."))
            # Ubuntu artifacts need glibc > 2.28; otherwise use AlmaLinux.
            return "ubuntu-x64" if major * 100 + minor > 228 else "almalinux-x64"
    print(
        f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
    )
    return None


def get_llvm_package_info():
    system_suffix = get_llvm_system_suffix()
    if system_suffix is None:
        return Package("llvm", "LLVM-C.lib", "", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")
    llvm_info_path = os.path.join(get_base_dir(), "cmake", "llvm-info.json")
    with open(llvm_info_path, "r") as llvm_info_file:
        llvm_info = json.load(llvm_info_file)
    rev = llvm_info["llvm_hash"][:8]
    name = f"llvm-{rev}-{system_suffix}"
    # Create a stable symlink that doesn't include revision
    sym_name = f"llvm-{system_suffix}"
    url = f"https://oaitriton.blob.core.windows.net/public/llvm-builds/{name}.tar.gz"
    return Package("llvm", name, url, "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH", sym_name=sym_name)
