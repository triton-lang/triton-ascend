"""Ascend-specific overrides for the in-tree ``setup.py``.

Unlike the previous ``setup_ascend.py`` (which imported ``setup.py`` out of
process via importlib and re-ran ``setup()``), these hooks run inside
``setup.py`` itself:

* :func:`prepare` is called once near the top of ``setup.py`` and sets up
  default build environment variables / fetches optional submodules.
* :func:`activate` is called right before the community ``setup(...)`` call.
  It rebinds the ``setup`` name in ``setup.py``'s globals to a wrapper which
  patches the setup module's globals and rewrites the keyword arguments, then
  delegates to the real ``setuptools.setup()``.

All Ascend-specific behavior stays isolated under ``third_party/ascend/build``;
``setup.py`` only carries two tiny hook calls.
"""

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import python.build_helpers as build_helpers

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel

from .build_npuir import build_npuir

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]

_BISHENGIR_PAYLOAD_ENV = "TRITON_ASCEND_BISHENGIR_PATH"


def _set_default_env_vars():
    os.environ.setdefault("TRITON_BUILD_WITH_CCACHE", "true")
    os.environ.setdefault("TRITON_BUILD_WITH_CLANG_LLD", "true")
    os.environ.setdefault("TRITON_BUILD_PROTON", "OFF")
    os.environ.setdefault("TRITON_BUILD_TD", "OFF")
    os.environ.setdefault("TRITON_BUILD_NPUIR", "OFF")
    os.environ.setdefault("TRITON_WHEEL_NAME", "triton_ascend")
    os.environ.setdefault("TRITON_APPEND_CMAKE_ARGS", "-DTRITON_BUILD_UT=OFF")


def _is_git_repo():
    return (_REPO_ROOT / ".git").is_dir()


def _is_linux_os(os_id):
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r") as f:
            return f'ID="{os_id}"' in f.read()
    return False


def _get_llvm_patch_hash():
    patch_dir = _REPO_ROOT / "third_party" / "ascend" / "patch"
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

    # Upstream 3.8 replaced cmake/llvm-hash.txt with cmake/llvm-info.json.
    llvm_info_path = Path(base_dir) / "cmake" / "llvm-info.json"
    with open(llvm_info_path, "r") as llvm_info_file:
        llvm_info = json.load(llvm_info_file)
    rev = llvm_info["llvm_hash"][:8]
    patch_hash = _get_llvm_patch_hash()
    name = f"llvm-{rev}-{patch_hash}-{system_suffix}"
    sym_name = f"llvm-{system_suffix}"
    url = f"https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz"
    return {"name": name, "sym_name": sym_name, "url": url}


def _resolve_ascend_llvm_syspath(mod):
    """Resolve the Ascend LLVM package dir to hand to CMake as LLVM_SYSPATH.

    Upstream now resolves third-party packages inside CMake by running
    python/build_helpers.py as a subprocess.  That subprocess imports
    build_helpers fresh from disk, so the in-memory get_llvm_package_info
    override patched onto the build_helpers module has no effect there.
    Resolve the Ascend LLVM package in-process instead (downloading and
    extracting it if needed) and return its directory; the caller passes it
    as -DLLVM_SYSPATH so the CMake-side resolution reuses it.
    """
    helper_args = build_helpers.BuildHelperArgs(
        cache_path=mod.get_triton_cache_path(),
        offline_build=mod.is_offline_build(),
        llvm_system_suffix=os.environ.get("TRITON_LLVM_SYSTEM_SUFFIX") or None,
        llvm_syspath=os.environ.get("LLVM_SYSPATH") or None,
        json_syspath=os.environ.get("JSON_SYSPATH") or None,
        ptxas_path=None,
        ptxas_blackwell_path=None,
        cuobjdump_path=None,
        nvdisasm_path=None,
        cudacrt_path=None,
        cudart_path=None,
        cupti_include_path=None,
        cupti_lib_path=None,
        cupti_lib_blackwell_path=None,
    )
    cmake_vars = build_helpers.get_thirdparty_cmake_vars(["llvm"], helper_args)
    return cmake_vars.get("LLVM_SYSPATH")


def _provision_ascend_llvm(mod):
    """Resolve the LLVM install the Ascend build compiles against."""
    return _resolve_ascend_llvm_syspath(mod)


def _apply_patch(patch_path, *, directory=None, cwd=None):
    cmd = ["git", "apply"]
    if directory:
        cmd.extend(["--directory", directory])
    cmd.append(patch_path)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, cwd=str(cwd or _REPO_ROOT))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"patch({patch_path}) failed,cmd={e.cmd}, retcode={e.returncode}") from e
    except FileNotFoundError:
        raise RuntimeError(f"patch({patch_path}) not found.")


def checkout_file(files, *, cwd=None):
    try:
        subprocess.run(["git", "checkout", "--"] + files, check=True, stdout=subprocess.DEVNULL, cwd=str(cwd
                                                                                                         or _REPO_ROOT))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"restore sources failed, list:{files}, cmd={e.cmd}, retcode={e.returncode}") from e


def _is_dev_mode():
    if os.getenv("IS_MANYLINUX", "FALSE").upper() not in ["ON", "1", "YES", "TRUE", "Y"]:
        return True
    if os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", ""):
        return True
    if "dev" in _get_default_version():
        return True
    return False


def _get_patch_files(patch_path):
    """Return repo-relative paths listed in a unified diff."""
    path = Path(patch_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    files = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("diff --git a/"):
                target = line.split(" b/", 1)[-1].rstrip("\n")
                if target != "/dev/null":
                    files.append(target)
    return files


def _normalize_crlf(files, *, cwd=None):
    """Convert CRLF line endings to LF in *files* (in-place).

    The sync workflow regenerates the npuir adapter patch via a text-mode
    round-trip, so the patch always carries LF line endings, while some
    npuir sources (e.g. the root CMakeLists.txt) are CRLF at HEAD.
    `git apply` matches context lines literally, so a CRLF working tree
    would reject the LF patch. CMake and mlir-tblgen parse LF fine.
    """
    base = Path(str(cwd or _THIS_DIR))
    for f in files:
        path = base / f
        data = path.read_bytes()
        if b"\r\n" in data:
            path.write_bytes(data.replace(b"\r\n", b"\n"))


def get_triton_ascend_patch_file():
    patch_dir = _REPO_ROOT / "third_party" / "ascend" / "patch"
    dev_patch = patch_dir / "triton-ascend-dev-3.8.0.patch"
    patch = patch_dir / "triton-ascend-3.8.0.patch"
    dev_patch_files = _get_patch_files(dev_patch) if dev_patch.is_file() else []
    patch_files = _get_patch_files(patch) if patch.is_file() else []
    return patch_files, dev_patch_files


def _apply_npuir_patch():
    """Apply AscendNPU-IR adaptations for LLVM 23 (Triton Ascend 3.7)."""
    patch_path = os.path.join("third_party", "ascend", "patch", "npuir_adapter_to_llvm_23.patch")
    npuir_dir = os.path.join("third_party", "ascend", "AscendNPU-IR")
    if not os.path.isfile(patch_path):
        raise RuntimeError(f"patch({patch_path}) not found.")
    if not os.path.isdir(npuir_dir):
        raise RuntimeError(f"AscendNPU-IR not found at {npuir_dir}")
    patch_files = _get_patch_files(patch_path)
    if not patch_files:
        raise RuntimeError(f"patch({patch_path}) has no file sections.")
    # Restore only files tracked at HEAD; new-file sections are created
    # fresh by `git apply` and have nothing to check out.
    tracked = subprocess.run(
        ["git", "ls-files", "--"] + patch_files,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=npuir_dir,
        text=True,
    ).stdout.splitlines()
    checkout_file(tracked, cwd=npuir_dir)
    _normalize_crlf(tracked, cwd=npuir_dir)
    _apply_patch(patch_path, directory=npuir_dir)


def _apply_triton_ascend_patch():
    patch_path = os.path.join("third_party", "ascend", "patch")
    dev_patch = os.path.join(patch_path, "triton-ascend-dev-3.8.0.patch")
    patch = os.path.join(patch_path, "triton-ascend-3.8.0.patch")
    if _is_dev_mode() and os.path.isfile(dev_patch):
        dev_patch_files = _get_patch_files(dev_patch)
        if dev_patch_files:
            checkout_file(dev_patch_files)
        _apply_patch(str(dev_patch))
    if os.path.isfile(patch):
        patch_files = _get_patch_files(patch)
        if not patch_files:
            raise RuntimeError(f"patch({patch}) has no file sections.")
        checkout_file(patch_files)
        _apply_patch(str(patch))
    _apply_npuir_patch()


def _print_patch_restore_warning():
    """Warn that the build left patched (dirty) source files in the worktree.

    ``apply_triton_ascend_patch`` modifies in-tree Triton sources, so a
    subsequent ``git pull`` would fail with local changes. Users can restore
    those files with ``python3 restore_sources.py`` at the repository root
    (which runs ``git checkout --`` on the patched file list).
    """
    if not _is_git_repo():
        return
    if sys.stdout.isatty():
        highlight = "\033[1;93m"
        reset = "\033[0m"
    else:
        highlight = ""
        reset = ""
    print("")
    print("=" * 72)
    print("WARNING: Ascend patches were applied to the in-tree Triton sources")
    print("         during this build. Your working tree is now dirty, which")
    print("         will cause `git pull` to fail with local changes.")
    print("")
    print("         To restore the source files, run:")
    print(f"            >>> {highlight}python3 restore_sources.py{reset} <<<")
    print("=" * 72)
    print("")


def _get_default_version():
    version_file = _REPO_ROOT / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    # Fallback tracks the upstream Triton version (3.8.0 since this sync);
    # version.txt is authoritative when present.
    return "3.8.0-dev"


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


def _add_git_safe_dir(path: str):
    safe_dirs = subprocess.run([
        "git",
        "config",
        "--global",
        "--get-all",
        "safe.directory",
    ], capture_output=True, text=True, cwd=str(_REPO_ROOT)).stdout.strip().splitlines()

    if path not in safe_dirs:
        subprocess.check_call([
            "git",
            "config",
            "--global",
            "--add",
            "safe.directory",
            path,
        ], cwd=str(_REPO_ROOT))


def _git_check_call_with_retry(cmd, cwd=None, retries=3, interval=5):
    """Run a git network command (clone/fetch) with retries.

    Network operations against the remote may fail intermittently; retry up to
    ``retries`` times, waiting ``interval`` seconds between attempts.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.check_call(cmd, cwd=cwd)
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            if attempt < retries:
                print(f"Command '{' '.join(cmd)}' failed (attempt {attempt}/{retries}), "
                      f"retrying in {interval}s...")
                time.sleep(interval)
            else:
                print(f"Command '{' '.join(cmd)}' failed after {retries} attempts.")
    raise last_error


def _ensure_npuir_submodule():
    if os.getenv("TRITON_BUILD_NPUIR", "OFF").upper() not in ["ON", "1", "YES", "TRUE", "Y"]:
        return
    build_npuir()


def _ensure_distributed_submodule():
    if os.getenv("TRITON_BUILD_TD", "OFF").upper() not in ["ON", "1", "YES", "TRUE", "Y"]:
        return
    distributed_dir = _REPO_ROOT / "third_party" / "ascend" / "Triton-distributed-ascend"
    commit_id = "8c1dae1acbb4bcf99c3e473c8ed876f2cd42ba35"
    if not distributed_dir.is_dir():
        try:
            _git_check_call_with_retry([
                "git",
                "clone",
                "https://gitcode.com/Ascend/Triton-distributed-ascend.git",
                "-b",
                "master",
            ], cwd=_REPO_ROOT / "third_party" / "ascend")
        except Exception:
            # A clone interrupted by a network failure leaves a partially
            # populated directory; remove it so the next build retries cleanly.
            if distributed_dir.is_dir():
                shutil.rmtree(distributed_dir, ignore_errors=True)
            raise
    if _is_git_repo():
        _add_git_safe_dir(str(distributed_dir))
        _git_check_call_with_retry([
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
    # triton-mlir-opt is deprecated for Triton Ascend 3.7 / LLVM 23.
    for rel_src, name in [
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


def _get_bishengir_payload_source():
    raw_path = os.getenv(_BISHENGIR_PAYLOAD_ENV)
    if not raw_path:
        return None

    source = Path(raw_path).expanduser().resolve()
    required_paths = [
        source / "bin" / "bishengir-compile",
        source / "bin" / "bishengir-opt",
        source / "lib",
    ]
    if not source.is_dir() or any(not path.exists() for path in required_paths):
        raise RuntimeError(f"{_BISHENGIR_PAYLOAD_ENV} must name a BishengIR directory containing "
                           "bin/bishengir-compile, bin/bishengir-opt, and lib")
    return source


def _copy_bishengir_payload(build_lib):
    source = _get_bishengir_payload_source()
    if source is None:
        return

    destination = Path(build_lib) / "triton" / "backends" / "ascend" / "bishengir"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=True)
    print(f"Bundled BishengIR payload from {source} into {destination}")


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
        "pytest>=8.3.2,<9.0.0",
        "pytest-xdist==3.6.1",
        "pyyaml",
        "pybind11",
        "pandas",
        "pyelftools>=0.29",
        "triton==3.8.0",
    ]
    return [*install_requires]


def patch_module(mod):
    """Apply all Ascend-specific overrides to the setup.py globals adapter."""

    # 1. Add "ascend" to the in-tree backends list. This also initializes the
    #    in-tree backend layout (asserts third_party/ascend/backend exists).
    ascend_backend = mod.BackendInstaller.prepare("ascend")
    mod.backends = [ascend_backend, *mod.backends]

    # 2. Replace LLVM package info with the Ascend pre-built LLVM.
    #    Upstream 3.8 moved Package/get_llvm_package_info into
    #    python/build_helpers.py (signature: get_llvm_package_info(helper_args)),
    #    so patch that module instead of the setup module.
    _orig_get_llvm_package_info = build_helpers.get_llvm_package_info

    def get_llvm_package_info(helper_args):
        info = _get_ascend_llvm_package_info(Path(mod.get_base_dir()))
        if info is not None:
            return build_helpers.Package(
                "llvm",
                info["name"],
                info["url"],
                "LLVM_INCLUDE_DIRS",
                "LLVM_LIBRARY_DIR",
                "LLVM_SYSPATH",
                sym_name=info["sym_name"],
            )
        return _orig_get_llvm_package_info(helper_args)

    build_helpers.get_llvm_package_info = get_llvm_package_info

    # 3. Skip downloading NVIDIA proprietary toolchain dependencies
    #    (ptxas/cuobjdump/...); they are irrelevant for the Ascend wheel.
    mod.download_and_copy_dependencies = lambda *args, **kwargs: None

    # 4. Patch CMakeBuild to apply the Ascend patch / coverage / tools.
    _OrigCMakeBuild = mod.CMakeBuild

    class CMakeBuild(_OrigCMakeBuild):

        def run(self):
            _apply_triton_ascend_patch()

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

            super().run()

        def build_extension(self, ext):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))

            orig_check_call = subprocess.check_call
            asc_extra_args = list(_get_ascend_cmake_args())
            asc_extra_args.append("-DLLVM_MAJOR_VERSION_23_COMPATIBLE=ON")
            if mod.check_env_flag("TRITON_BUILD_TD", "OFF"):
                asc_extra_args.append("-DTRITON_BUILD_TD=ON")
            else:
                asc_extra_args.append("-DTRITON_BUILD_TD=OFF")

            # Upstream no longer passes LLVM paths from setup.py; CMake resolves
            # them by invoking python/build_helpers.py as a subprocess, which
            # cannot see the in-memory get_llvm_package_info override.  Resolve
            # the Ascend LLVM package here and pass its path via LLVM_SYSPATH.
            ascend_llvm_syspath = _provision_ascend_llvm(mod)
            if ascend_llvm_syspath:
                asc_extra_args.append("-DLLVM_SYSPATH=" + ascend_llvm_syspath)
            # Resolve clang++ for the NVIDIA GSan runtime build (upstream's
            # find_program is REQUIRED). Prefer the LLVM package's clang++
            # (upstream's search order), then fall back to PATH. The user can
            # still override via TRITON_APPEND_CMAKE_ARGS.
            if "TRITON_GSAN_CLANGXX" not in os.environ.get("TRITON_APPEND_CMAKE_ARGS", ""):
                gsan_clangxx = None
                if ascend_llvm_syspath:
                    candidate = os.path.join(ascend_llvm_syspath, "bin", "clang++")
                    if os.path.isfile(candidate):
                        gsan_clangxx = candidate
                if not gsan_clangxx:
                    gsan_clangxx = shutil.which("clang++")
                if gsan_clangxx:
                    asc_extra_args.append("-DTRITON_GSAN_CLANGXX=" + gsan_clangxx)
            # Upstream now passes its TRITON_VERSION to CMake (Version.h);
            # pass the Ascend wheel version instead so the compiled version
            # matches the wheel.
            asc_extra_args.append("-DTRITON_VERSION=" + _get_version(mod._ascend_is_manylinux, mod.get_git_commit_hash))

            def patched_check_call(cmd, *args, **kwargs):
                # Only the cmake configure invocation takes extra -D args;
                # leave `cmake --build` and the new `cmake --install`
                # (wheel_headers component) untouched.
                if (isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "cmake" and "--build" not in cmd
                        and "--install" not in cmd):
                    cmd = list(cmd) + asc_extra_args
                return orig_check_call(cmd, *args, **kwargs)

            subprocess.check_call = patched_check_call
            try:
                super().build_extension(ext)
            finally:
                subprocess.check_call = orig_check_call

            _copy_ascend_tools(extdir, mod.get_cmake_dir())

    mod.CMakeBuild = CMakeBuild

    _OrigCMakeBuildPy = mod.CMakeBuildPy

    class AscendBuildPy(_OrigCMakeBuildPy):

        def run(self):
            super().run()
            _copy_bishengir_payload(self.build_lib)

    mod.AscendBuildPy = AscendBuildPy

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

    _orig_get_package_dirs = mod.get_package_dirs

    def get_package_dirs():
        yield from _orig_get_package_dirs()
        if mod.check_env_flag("TRITON_BUILD_TD", "OFF"):
            yield ("triton_dist",
                   os.path.join("third_party", "ascend", "Triton-distributed-ascend", "python", "triton_dist"))

    mod.get_package_dirs = get_package_dirs

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

    _orig_add_links = mod.add_links

    def add_links(external_only):
        _orig_add_links(external_only)
        if not external_only and \
                mod.check_env_flag("TRITON_BUILD_TD", "OFF"):
            distributed_dir = (_REPO_ROOT / "third_party" / "ascend" / "Triton-distributed-ascend" / "python" /
                               "triton_dist").resolve()
            distributed_install_dir = _REPO_ROOT / "python" / "triton_dist"
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
    readme = _REPO_ROOT / "README.md"
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

    cmdclass = dict(kwargs.get("cmdclass") or {})
    cmdclass["bdist_wheel"] = mod.BuildWheel
    cmdclass["build_ext"] = mod.CMakeBuild
    cmdclass["build_py"] = mod.AscendBuildPy
    kwargs["cmdclass"] = cmdclass

    # packages / package_dir / entry_points were computed at setup() argument
    # evaluation time using the original backends list; re-evaluate with the
    # patched functions so the ascend backend (+ distributed) is included.
    kwargs["packages"] = list(mod.get_packages())
    kwargs["package_dir"] = dict(mod.get_package_dirs())
    kwargs["entry_points"] = mod.get_entry_points()

    return kwargs


def prepare():
    """Top-of-setup.py hook: defaults and optional submodules."""
    _set_default_env_vars()
    _ensure_npuir_submodule()
    _ensure_distributed_submodule()


def activate():
    """Install the Ascend interceptor as the ``setup`` name in the calling
    ``setup.py`` module globals.

    Must be called at module level in ``setup.py`` immediately before its
    ``setup(...)`` call. ``from setuptools import setup`` binds the name at
    import time, so patching ``setuptools.setup`` alone would not intercept
    the call; we therefore rebind the caller's global ``setup`` directly.
    This works both for ``python setup.py`` and for PEP 517 builds that exec
    ``setup.py`` with a fresh globals dict.
    """
    import setuptools

    caller_globals = sys._getframe(1).f_globals
    current_setup = caller_globals.get("setup")
    if getattr(current_setup, "_triton_ascend_wrapped", False):
        return

    real_setup = setuptools.setup

    def _ascend_setup(**kwargs):
        setup_globals = sys._getframe(1).f_globals

        class _ModuleGlobals:

            def __getattr__(self, name):
                try:
                    return setup_globals[name]
                except KeyError as e:
                    raise AttributeError(name) from e

            def __setattr__(self, name, value):
                setup_globals[name] = value

        mod = _ModuleGlobals()
        patch_module(mod)
        kwargs = _build_setup_kwargs(mod, kwargs)
        dist = real_setup(**kwargs)
        # distutils.core.run_setup() retrieves the distribution from the
        # exec()'d globals under the name "dist".
        setup_globals["dist"] = dist
        _print_patch_restore_warning()
        return dist

    _ascend_setup._triton_ascend_wrapped = True
    setuptools.setup = _ascend_setup
    caller_globals["setup"] = _ascend_setup
