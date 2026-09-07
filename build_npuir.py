import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_NPUIR_DIR = _THIS_DIR / "third_party" / "ascend" / "AscendNPU-IR"

_MIN_FREE_DISK_GB = 30
_GIT_RETRY_TIMES = 3
_GIT_RETRY_INTERVAL = 5


def _log(msg):
    print(f"[build_npuir] {msg}", flush=True)


def _is_git_repo(dir_path):
    return (Path(dir_path) / ".git").is_dir()


def _check_disk_space(min_free_gb=_MIN_FREE_DISK_GB):
    """Check that the disk hosting the repo has at least ``min_free_gb`` free.

    Building AscendNPU-IR recursively fetches LLVM / Torch-MLIR sources and
    produces a large build tree, which requires substantial disk space.
    """
    usage = shutil.disk_usage(str(_THIS_DIR))
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    _log(f"Disk space on {_THIS_DIR.drive or _THIS_DIR.anchor}: "
         f"free {free_gb:.1f} GiB / total {total_gb:.1f} GiB "
         f"(required >= {min_free_gb} GiB)")
    if free_gb < min_free_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f} GiB free, but building "
                           f"AscendNPU-IR requires at least {min_free_gb} GiB. "
                           f"Please free up disk space and retry.")


def _get_ascend_path() -> Path:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError("ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first")
    return Path(path)


def _run_with_retry(cmd, cwd=None, retries=_GIT_RETRY_TIMES, interval=_GIT_RETRY_INTERVAL):
    """Run a network command (git clone/fetch/submodule) with retries."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            if attempt < retries:
                _log(f"Command '{' '.join(map(str, cmd))}' failed (attempt "
                     f"{attempt}/{retries}), retrying in {interval}s...")
                time.sleep(interval)
            else:
                _log(f"Command '{' '.join(map(str, cmd))}' failed after "
                     f"{retries} attempts.")
    raise last_error


def _is_submodule_initialized(dir_path):
    """A submodule is considered initialized when its source tree is present."""
    dir_path = Path(dir_path)
    return dir_path.is_dir() and (dir_path / "CMakeLists.txt").exists()


def _init_npuir_repo():
    """Initialize the AscendNPU-IR submodule and its nested submodules.

    AscendNPU-IR depends on LLVM and Torch-MLIR, which are pulled in as
    nested submodules, hence the recursive update.
    """
    _log("Initializing AscendNPU-IR repository ...")
    if not _is_git_repo(_THIS_DIR):
        raise RuntimeError(f"{_THIS_DIR} is not a git repository; cannot initialize the "
                           f"Triton-Ascend submodule.")

    # First initialize the AscendNPU-IR submodule itself from the root repo.
    _run_with_retry([
        "git",
        "submodule",
        "update",
        "--init",
        "--depth",
        "1",
        "--",
        "third_party/ascend/AscendNPU-IR",
    ], cwd=_THIS_DIR)

    # Then recursively initialize its nested submodules (LLVM, Torch-MLIR).
    if _is_submodule_initialized(_NPUIR_DIR):
        _run_with_retry([
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ], cwd=_NPUIR_DIR)
    else:
        raise RuntimeError(f"AscendNPU-IR submodule initialization failed: {_NPUIR_DIR} is not git repository.")

    if not _is_submodule_initialized(_NPUIR_DIR):
        raise RuntimeError(f"AscendNPU-IR initialization failed: {_NPUIR_DIR} is still empty.")
    _log("AscendNPU-IR repository initialized.")


def _build_and_package_bisheng(repo_dir, bisheng_compiler_path, build_type="Release", rebuild=True):
    """Configure, build and install AscendNPU-IR via its build.sh script."""
    repo_dir = Path(repo_dir)
    build_script = repo_dir / "build-tools" / "build.sh"
    if not build_script.exists():
        raise RuntimeError(f"Build script not found: {build_script}")
    max_jobs = min(os.cpu_count() // 8, 32)
    build_path = repo_dir / "build"
    if build_path.exists():
        shutil.rmtree(str(build_path))
    build_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(build_script), f"--build-type={str(build_type)}", "-o",
        str(build_path), "-t", "-j",
        str(max_jobs), f"--bisheng-compiler={str(bisheng_compiler_path)}", "--add-cmake-options",
        "-DLLVM_ENABLE_LLD=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "--build-triton", "--build-torch-mlir", "--build-shmem-template", "--bishengir-publish", "ON",
        "--collect-binary"
    ]
    if rebuild:
        cmd.append("-r")
    _log(f"Building AscendNPU-IR (bisheng): {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(repo_dir))
    _log(f"AscendNPU-IR build finished. Artifacts under: {build_path}")
    install_dir = build_path / "install"
    if install_dir.is_dir():
        _log(f"Installed artifacts under: {install_dir}")


def _copy_artifacts():
    """Collect built binaries/bitcode into third_party/ascend/bishengir."""
    ascend_bishengir_path = _THIS_DIR / "third_party" / "ascend" / "backend" / "bishengir"
    if ascend_bishengir_path.exists():
        shutil.rmtree(ascend_bishengir_path)
    bin_dir = ascend_bishengir_path / "bin"
    lib_dir = ascend_bishengir_path / "lib"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    file_copies = [
        (_NPUIR_DIR / "bishengir-output" / "bin" / "bishengir-compile", bin_dir / "bishengir-compile"),
        (_NPUIR_DIR / "bishengir-output" / "bin" / "bishengir-opt", bin_dir / "bishengir-opt"),
        (_NPUIR_DIR / "bishengir-output" / "bin" / "hivmc", bin_dir / "hivmc"),
        (_NPUIR_DIR / "bishengir-output" / "bin" / "hivmc-a5", bin_dir / "hivmc-a5"),
    ]
    for src, dst in file_copies:
        if src.is_file():
            shutil.copy(src, dst)
            if not os.path.exists(dst):
                raise RuntimeError(f"Copy {src} to {dst} failed.")
            _log(f"Copied {src} -> {dst}")
        else:
            raise RuntimeError(f"Copy {src} to {dst} failed.")

    bc_src_dir = _NPUIR_DIR / "bishengir-output" / "lib"
    if bc_src_dir.is_dir():
        for bc in bc_src_dir.glob("*.bc"):
            shutil.copy(bc, lib_dir / bc.name)
            if not os.path.exists(lib_dir / bc.name):
                raise RuntimeError(f"Copy {bc} to {lib_dir} failed.")
            _log(f"Copied {bc} -> {lib_dir / bc.name}")
    else:
        _log(f"warning: bitcode dir not found: {bc_src_dir}")


def build_npuir():
    _log("Step 1/5: checking disk space ...")
    _check_disk_space()

    _log("Step 2/5: locating bisheng compiler ...")
    bisheng_compiler_path = (_get_ascend_path() / "tools" / "bisheng_compiler" / "bin")
    _log(f"bisheng compiler: {bisheng_compiler_path}")

    _log("Step 3/5: initializing code repositories ...")
    _init_npuir_repo()

    _log("Step 4/5: building and packaging ...")
    _build_and_package_bisheng(
        _NPUIR_DIR,
        bisheng_compiler_path=bisheng_compiler_path,
        build_type="Release",
        rebuild=True,
    )

    _log("Step 5/5: copying artifacts ...")
    _copy_artifacts()
    _log("All done.")
