import glob
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel

_THIS_DIR = Path(__file__).resolve().parent
_TRITON_SETUP = _THIS_DIR / "setup.py"


def _set_default_env_vars():
    os.environ.setdefault("TRITON_BUILD_WITH_CCACHE", "true")
    os.environ.setdefault("TRITON_BUILD_WITH_CLANG_LLD", "true")
    os.environ.setdefault("TRITON_BUILD_PROTON", "OFF")
    os.environ.setdefault("TRITON_BUILD_TD", "OFF")
    os.environ.setdefault("TRITON_WHEEL_NAME", "triton_ascend")
    os.environ.setdefault("TRITON_APPEND_CMAKE_ARGS", "-DTRITON_BUILD_UT=OFF")


def _is_git_repo():
    return (_THIS_DIR / ".git").is_dir()


def _is_linux_os(os_id):
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r") as f:
            return f'ID="{os_id}"' in f.read()
    return False


def _get_llvm_patch_hash():
    patch_dir = _THIS_DIR / "third_party" / "ascend" / "patch"
    if patch_dir.is_dir():
        patch_files = sorted(f for f in os.listdir(patch_dir)
                             if f.startswith("llvm_patch_") and f.endswith(".patch") and (patch_dir / f).is_file())
    else:
        patch_files = []
    if not patch_files:
        return "00000000"
    import hashlib
    h = hashlib.sha256()
    for pf in patch_files:
        h.update((patch_dir / pf).read_bytes())
    return h.hexdigest()[:8]


def _get_ascend_llvm_package_info(base_dir):
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine()]
    except KeyError:
        arch = platform.machine()

    env_system_suffix = os.environ.get("TRITON_LLVM_SYSTEM_SUFFIX")
    if env_system_suffix:
        system_suffix = env_system_suffix
    elif system == "Darwin":
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == "arm64" and _is_linux_os("almalinux"):
            system_suffix = "almalinux-arm64"
        elif arch == "arm64":
            system_suffix = "ubuntu-arm64"
        elif arch == "x64":
            vglibc = tuple(map(int, platform.libc_ver()[1].split(".")))
            vglibc = vglibc[0] * 100 + vglibc[1]
            system_suffix = "ubuntu-x64" if vglibc > 228 else "almalinux-x64"
        else:
            return None
    else:
        return None

    llvm_hash_path = base_dir / "cmake" / "llvm-hash.txt"
    rev = llvm_hash_path.read_text()[:8]
    patch_hash = _get_llvm_patch_hash()
    name = f"llvm-{rev}-{patch_hash}-{system_suffix}"
    sym_name = f"llvm-{system_suffix}"
    url = f"https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz"
    return {"name": name, "sym_name": sym_name, "url": url}


def _apply_patch(patch_path):
    try:
        subprocess.run(["git", "apply", patch_path], check=True, stdout=subprocess.DEVNULL, cwd=str(_THIS_DIR))
    except subprocess.CalledProcessError:
        raise RuntimeError(f"patch({patch_path}) failed")
    except FileNotFoundError:
        raise RuntimeError(f"patch({patch_path}) not found.")


def _checkout_file(files):
    try:
        subprocess.run(["git", "checkout", "--"] + files, check=True, stdout=subprocess.DEVNULL, cwd=str(_THIS_DIR))
    except subprocess.CalledProcessError:
        raise RuntimeError(f"init code failed, list:{files}")


def _is_dev_mode():
    if os.getenv("IS_MANYLINUX", "FALSE").upper() not in ["ON", "1", "YES", "TRUE", "Y"]:
        return True
    if os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", ""):
        return True
    if "dev" in _get_default_version():
        return True


def _get_triton_ascend_patch_file():
    patch_files = [
        "CMakeLists.txt",
        "include/triton/Dialect/Triton/IR/TritonAttrDefs.td",
        "lib/Dialect/Triton/IR/Traits.cpp",
        "python/src/ir.cc",
        "python/triton/_utils.py",
        "python/triton/compiler/code_generator.py",
        "python/triton/compiler/compiler.py",
        "python/triton/compiler/errors.py",
        "python/triton/language/math.py",
        "python/triton/language/semantic.py",
        "python/triton/language/standard.py",
        "python/triton/runtime/interpreter.py",
        "python/triton/runtime/jit.py",
        "bin/RegisterTritonDialects.h",
        "bin/triton-opt.cpp",
        "bin/CMakeLists.txt",
    ]
    dev_patch_files = ["python/triton/runtime/autotuner.py"]
    return patch_files, dev_patch_files


def _apply_triton_ascend_patch():
    patch_path = os.path.join("third_party", "ascend", "patch")
    dev_patch = os.path.join(patch_path, "triton-ascend-dev-3.6.0.patch")
    patch = os.path.join(patch_path, "triton-ascend-3.6.0.patch")
    patch_files, dev_patch_files = _get_triton_ascend_patch_file()
    if _is_dev_mode() and os.path.isfile(dev_patch):
        _checkout_file(dev_patch_files)
        _apply_patch(str(dev_patch))
    if os.path.isfile(patch):
        _checkout_file(patch_files)
        _apply_patch(str(patch))


def _get_default_version():
    version_file = _THIS_DIR / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    return "3.6.0-dev"


def _get_version(is_manylinux, get_git_commit_hash):
    version = os.environ.get("TRITON_VERSION", _get_default_version()) + \
              os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", "")
    if not is_manylinux:
        version += get_git_commit_hash()
    return version


def _setup_coverage_env():
    hitest_home = os.getenv("HITEST_HOME", "/opt/hitest/linux_avatar_x86_64")
    hitest_user_account = os.getenv("HITEST_USER_ACCOUNT", "a00000000")
    lltcov_rootpath = os.getenv("LLTCOV_ROOTPATH", "/opt/covdata")

    coverage_env_vars = {
        "HitestHome": hitest_home,
        "isOverlappedCompile": "0",
        "PlatformToken": "BOARD",
        "gcovmode": "0",
        "TimerPolicy": "1",
        "TimeInterval": "60",
        "SignalPolicy": "1",
        "SignalNUM": "34",
        "lltwrapper_cfg": "0",
        "HITEST_AGENT_INSIDE": "1",
        "USE_HLLT_COVERAGE": "1",
        "USE_HLLT_TESTCASE": "0",
        "simplemode": "0",
        "ncs_coverage_stub_mold": "1",
        "HITEST_ENABLE_SOKCET": "0",
        "hitest_disable_cfg": "0",
        "hitest_disable_dfg": "1",
        "hitest_disable_ir": "1",
        "HITEST_DISABLE_MACRO": "0",
        "HITEST_REMOVE_INCLUDE_DIR": "0",
        "HITEST_AGENT_SET_THREADNAME_PRCTL": "1",
        "HITEST_INST_HEADER_FILE": "0",
        "HITEST_USER_ACCOUNT": hitest_user_account,
        "lltcovRootpath": lltcov_rootpath,
        "HITEST_COVSTUB_ROOT_DIR": f"{hitest_home}/apache-tomcat-8.0.39/webapps/datasource/Container_Default/base",
        "HITEST_EXEC_CMD_WITH_FILE": "1",
        "HITEST_PRINT_LOG_ENABLE": "1",
    }
    os.environ.update(coverage_env_vars)
    os.environ["PATH"] = f"{hitest_home}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{hitest_home}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    print("The environment variables for the hitest coverage tool have been read.")
    print(f"  HitestHome: {hitest_home} (environment variables HITEST_HOME)")
    print(f"  HITEST_USER_ACCOUNT: {hitest_user_account} (environment variables HITEST_USER_ACCOUNT)")
    print(f"  lltcovRootpath: {lltcov_rootpath} (environment variables LLTCOV_ROOTPATH)")


def _clean_hitest_env():
    for key in list(os.environ.keys()):
        if key.startswith("HITEST_") or key in ("HitestHome", "lltcovRootpath"):
            del os.environ[key]


def add_git_safe_dir(path: str):
    safe_dirs = subprocess.run([
        "git",
        "config",
        "--global",
        "--get-all",
        "safe.directory",
    ], capture_output=True, text=True, cwd=_THIS_DIR).stdout.strip().splitlines()

    if path not in safe_dirs:
        subprocess.check_call([
            "git",
            "config",
            "--global",
            "--add",
            "safe.directory",
            path,
        ], cwd=_THIS_DIR)


def _ensure_distributed_submodule():
    if os.getenv("TRITON_BUILD_TD", "OFF").upper() not in ["ON", "1", "YES", "TRUE", "Y"]:
        return
    distributed_dir = _THIS_DIR / "third_party" / "ascend" / "Triton-distributed-ascend"
    commit_id = "63e07743167ab7cbe7902b19214a9c56b24e9777"
    if not distributed_dir.is_dir():
        subprocess.check_call([
            "git",
            "clone",
            "https://gitcode.com/Ascend/Triton-distributed-ascend.git",
            "-b",
            "master",
        ], cwd=_THIS_DIR / "third_party" / "ascend")
    if _is_git_repo():
        add_git_safe_dir(str(distributed_dir))
        subprocess.check_call([
            "git",
            "fetch",
            "origin",
        ], cwd=distributed_dir)
        subprocess.check_call([
            "git",
            "checkout",
            commit_id,
        ], cwd=distributed_dir)

        result = subprocess.run([
            "git",
            "rev-parse",
            "HEAD",
        ], capture_output=True, text=True, cwd=distributed_dir)
        current_id = result.stdout.strip()
        if current_id != commit_id:
            raise RuntimeError(f"Triton-Distributed submodule is not {commit_id}")


def _copy_ascend_tools(extdir, cmake_dir):
    for rel_src, name in [
        ("third_party/ascend/bin/triton-mlir-opt", "triton-mlir-opt"),
        ("bin/triton-opt", "triton-opt"),
    ]:
        src = Path(cmake_dir) / rel_src
        if src.exists():
            dst = Path(extdir) / name
            shutil.copy2(src, dst)
            if platform.system() != "Windows":
                os.chmod(dst, 0o755)
                try:
                    subprocess.check_call(["strip", "--strip-all", str(dst)])
                    print(f"Stripped {name} to reduce size")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
            print(f"Copied {name} to {dst}")


def _get_ascend_cmake_args():
    cmake_args = []
    ascendnpu_ir_tag = os.getenv("ASCENDNPU_IR_TAG")
    if ascendnpu_ir_tag is not None:
        cmake_args.append(f"-DASCENDNPU_IR_TAG={ascendnpu_ir_tag}")
    return cmake_args


def _get_install_requirements():
    install_requires = [
        "attrs==24.2.0",
        "numpy==1.26.4",
        "scipy==1.13.1;python_version<'3.13'",
        "scipy==1.15.1;python_version>='3.13'",
        "decorator==5.1.1",
        "psutil==6.0.0",
        "pytest==8.3.2",
        "pytest-xdist==3.6.1",
        "pyyaml",
        "pybind11",
        "pandas",
        "triton==3.6.0",
    ]
    return [*install_requires]


def _patch_module(mod):
    """Apply all Ascend-specific overrides to the imported setup_triton module."""

    # 1. Add "ascend" to the in-tree backends list.
    ascend_backend = mod.BackendInstaller.prepare("ascend")
    mod.backends = [ascend_backend, *mod.backends]

    # 2. Replace LLVM package info with Ascend build.
    _orig_get_llvm_package_info = mod.get_llvm_package_info

    def get_llvm_package_info():
        info = _get_ascend_llvm_package_info(Path(mod.get_base_dir()))
        if info is not None:
            return mod.Package(
                "llvm",
                info["name"],
                info["url"],
                "LLVM_INCLUDE_DIRS",
                "LLVM_LIBRARY_DIR",
                "LLVM_SYSPATH",
                sym_name=info["sym_name"],
            )
        return _orig_get_llvm_package_info()

    mod.get_llvm_package_info = get_llvm_package_info

    # 3. Patch CMakeBuild to apply Ascend patch / coverage / tools.
    _OrigCMakeBuild = mod.CMakeBuild

    class CMakeBuild(_OrigCMakeBuild):

        def run(self):
            _apply_triton_ascend_patch()

            try:
                out = subprocess.check_output(["cmake", "--version"])
            except OSError:
                raise RuntimeError("CMake must be installed to build the following extensions: " +
                                   ", ".join(e.name for e in self.extensions))

            match = re.search(r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode())
            cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
            if (cmake_major, cmake_minor) < (3, 20):
                raise RuntimeError("CMake >= 3.20 is required")

            enable_hitest = os.getenv("TRITON_ENABLE_COVERAGE_HITEST", "0").lower() \
                            in ("1", "on", "true")
            if enable_hitest:
                _setup_coverage_env()
                current_append = os.environ.get("TRITON_APPEND_CMAKE_ARGS", "")
                if current_append:
                    os.environ["TRITON_APPEND_CMAKE_ARGS"] = \
                        current_append + " -DTRITON_ENABLE_COVERAGE_HITEST=ON"
                else:
                    os.environ["TRITON_APPEND_CMAKE_ARGS"] = \
                        "-DTRITON_ENABLE_COVERAGE_HITEST=ON"
            else:
                _clean_hitest_env()

            for ext in self.extensions:
                self.build_extension(ext)

        def build_extension(self, ext):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))

            orig_check_call = subprocess.check_call
            asc_extra_args = list(_get_ascend_cmake_args())
            asc_extra_args.append("-DLLVM_MAJOR_VERSION_22_COMPATIBLE=ON")
            if mod.check_env_flag("TRITON_BUILD_TD", "OFF"):
                asc_extra_args.append("-DTRITON_BUILD_TD=ON")
            else:
                asc_extra_args.append("-DTRITON_BUILD_TD=OFF")

            def patched_check_call(cmd, *args, **kwargs):
                if (isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "cmake" and "--build" not in cmd):
                    cmd = list(cmd) + asc_extra_args
                return orig_check_call(cmd, *args, **kwargs)

            subprocess.check_call = patched_check_call
            try:
                super().build_extension(ext)
            finally:
                subprocess.check_call = orig_check_call

            _copy_ascend_tools(extdir, mod.get_cmake_dir())

    mod.CMakeBuild = CMakeBuild

    # 4. Replace BuildWheel (bdist_wheel) with Ascend auditwheel variant.
    is_manylinux = mod.check_env_flag("IS_MANYLINUX", "FALSE")

    class BuildWheel(bdist_wheel):

        def run(self):
            mod.add_links(external_only=True)
            bdist_wheel.run(self)

            if is_manylinux:
                file = glob.glob(os.path.join(self.dist_dir, "*-linux_*.whl"))[0]
                auditwheel_cmd = [
                    "auditwheel",
                    "-v",
                    "repair",
                    "--plat",
                    f"manylinux_2_27_{platform.machine()}",
                    "--plat",
                    f"manylinux_2_28_{platform.machine()}",
                    "-w",
                    self.dist_dir,
                    file,
                ]
                try:
                    subprocess.run(auditwheel_cmd, check=True, stdout=subprocess.PIPE)
                except subprocess.CalledProcessError:
                    raise RuntimeError("Auditwheel failed")
                finally:
                    os.remove(file)

    mod.BuildWheel = BuildWheel

    # 5. Patch get_package_dirs to include distributed package.
    _orig_get_package_dirs = mod.get_package_dirs

    def get_package_dirs():
        yield from _orig_get_package_dirs()
        if mod.check_env_flag("TRITON_BUILD_TD", "OFF"):
            yield ("triton_dist",
                   os.path.join("third_party", "ascend", "Triton-distributed-ascend", "python", "triton_dist"))

    mod.get_package_dirs = get_package_dirs

    # 6. Patch get_packages to include distributed subpackages.
    _orig_get_packages = mod.get_packages

    def get_packages():
        yield from _orig_get_packages()
        if mod.check_env_flag("TRITON_BUILD_TD", "OFF"):
            distributed_pkg_root = os.path.join("third_party", "ascend", "Triton-distributed-ascend", "python",
                                                "triton_dist")
            if os.path.isdir(distributed_pkg_root):
                for dirpath, _dirnames, filenames in os.walk(distributed_pkg_root):
                    if "__init__.py" in filenames or \
                            any(f.endswith(".py") for f in filenames):
                        rel = os.path.relpath(dirpath, distributed_pkg_root) \
                            .replace(os.sep, ".")
                        yield "triton_dist" if rel == "." \
                            else f"triton_dist.{rel}"

    mod.get_packages = get_packages

    # 7. Patch add_links to include distributed symlink.
    _orig_add_links = mod.add_links

    def add_links(external_only):
        _orig_add_links(external_only)
        if not external_only and \
                mod.check_env_flag("TRITON_BUILD_TD", "OFF"):
            distributed_dir = (_THIS_DIR / "third_party" / "ascend" / "Triton-distributed-ascend" / "python" /
                               "triton_dist").resolve()
            distributed_install_dir = _THIS_DIR / "python" / "triton_dist"
            mod.update_symlink(distributed_install_dir, distributed_dir)

    mod.add_links = add_links

    # Expose helpers needed by the setup() interceptor.
    mod._ascend_is_manylinux = is_manylinux


def _build_setup_kwargs(mod, kwargs):
    """Modify kwargs passed to setup() for Ascend."""
    is_manylinux = mod._ascend_is_manylinux

    kwargs["name"] = os.environ.get("TRITON_WHEEL_NAME", "triton_ascend")
    kwargs["version"] = _get_version(is_manylinux, mod.get_git_commit_hash)
    kwargs["url"] = "https://gitcode.com/Ascend/triton-ascend/"

    # README as long_description
    readme = _THIS_DIR / "README.md"
    if readme.exists():
        kwargs["long_description"] = readme.read_text(encoding="utf-8")

    # install_requires
    kwargs["install_requires"] = _get_install_requirements()

    # package_data for distributed
    package_data = dict(kwargs.get("package_data") or {})
    if mod.check_env_flag("TRITON_BUILD_TD", "OFF"):
        package_data["triton_dist"] = ["*.py", "*.pyi"]
    if package_data:
        kwargs["package_data"] = package_data

    # cmdclass: replace bdist_wheel with BuildWheel, build_ext with CMakeBuild
    cmdclass = dict(kwargs.get("cmdclass") or {})
    cmdclass["bdist_wheel"] = mod.BuildWheel
    cmdclass["build_ext"] = mod.CMakeBuild
    kwargs["cmdclass"] = cmdclass

    # packages / package_dir must be re-evaluated (they were computed with
    # the original backends list before we patched it). Re-call the patched
    # functions so ascend backend + distributed are included.
    kwargs["packages"] = list(mod.get_packages())
    kwargs["package_dir"] = dict(mod.get_package_dirs())

    # Recompute entry_points so that the ascend backend entry is present.
    kwargs["entry_points"] = mod.get_entry_points()

    return kwargs


def main():
    _set_default_env_vars()
    _ensure_distributed_submodule()

    # Import the community setup_triton module without executing its setup()
    # call. We do this by temporarily replacing setuptools.setup.
    import setuptools
    _real_setup = setuptools.setup
    captured = {}

    def _capture_setup(**kwargs):
        captured["kwargs"] = kwargs

    setuptools.setup = _capture_setup
    try:
        spec = importlib.util.spec_from_file_location("setup", str(_TRITON_SETUP))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["setup"] = mod
        spec.loader.exec_module(mod)
    finally:
        setuptools.setup = _real_setup

    # Apply Ascend overrides to the module before invoking setup().
    _patch_module(mod)

    kwargs = _build_setup_kwargs(mod, captured["kwargs"])
    _real_setup(**kwargs)


if __name__ == "__main__":
    main()
