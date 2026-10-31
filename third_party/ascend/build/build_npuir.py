import os
import shutil
import subprocess
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
_NPUIR_DIR = _REPO_ROOT / "third_party" / "ascend" / "AscendNPU-IR"

_MIN_FREE_DISK_GB = 30
_GIT_RETRY_TIMES = 3
_GIT_RETRY_INTERVAL = 5


def _log(msg):
    print(f"[build_npuir] {msg}", flush=True)


def _is_git_repo(dir_path):
    return (Path(dir_path) / ".git").exists()


def _check_disk_space(min_free_gb=_MIN_FREE_DISK_GB):
    """Check that the disk hosting the repo has at least ``min_free_gb`` free.

    Building AscendNPU-IR recursively fetches LLVM / Torch-MLIR sources and
    produces a large build tree, which requires substantial disk space.
    """
    usage = shutil.disk_usage(str(_REPO_ROOT))
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    _log(f"Disk space on {_REPO_ROOT.drive or _REPO_ROOT.anchor}: "
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

    Local builds use the version pinned by the submodule gitlink (the
    relatively stable, tested revision), NOT the upstream stable branch:
    tracking stable HEAD is the wheel CI's job, which pre-fetches the
    sources before running setup.py (see wheels.Dockerfile) — an existing
    source tree is therefore left untouched here.

    AscendNPU-IR depends on LLVM and Torch-MLIR, which are pulled in as
    nested submodules, hence the recursive update.
    """
    if _is_submodule_initialized(_NPUIR_DIR):
        _log(f"AscendNPU-IR sources already present at {_NPUIR_DIR}, skipping initialization.")
        return

    _log("Initializing AscendNPU-IR repository ...")
    if not _is_git_repo(_REPO_ROOT):
        raise RuntimeError(f"{_REPO_ROOT} is not a git repository; cannot initialize the "
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
    ], cwd=_REPO_ROOT)

    # Then initialize the nested submodules recorded by the pinned revision
    # (third-party/llvm-project, torch-mlir, shmem) — single level, depth-1,
    # mirroring what the wheel CI fetches (fetch-npuir-stable.sh). NOT
    # --recursive: the torch-mlir fork's own .gitmodules points at github.com
    # (externals/llvm-project — the full upstream LLVM monorepo — and
    # stablehlo), which recursive init would try to clone with no depth
    # limit; build-tools/build.sh consumes third-party/* directly and never
    # uses those externals (the CI payload builds prove it daily).
    if _is_submodule_initialized(_NPUIR_DIR):
        _run_with_retry([
            "git",
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
        ], cwd=_NPUIR_DIR)
    else:
        raise RuntimeError(f"AscendNPU-IR submodule initialization failed: {_NPUIR_DIR} is not git repository.")

    if not _is_submodule_initialized(_NPUIR_DIR):
        raise RuntimeError(f"AscendNPU-IR initialization failed: {_NPUIR_DIR} is still empty.")
    _log("AscendNPU-IR repository initialized.")


def _cgroup_limits():
    """Effective CPU quota and memory limit, cgroup-aware (v1 and v2).

    nproc/os.cpu_count() report the host's cores (320 on the CI pods),
    which over-subscribes memory when scheduling parallel compiles.
    """
    quota, period = None, 100000
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()
            quota, period = (None if q == "max" else int(q)), int(p)
    except (FileNotFoundError, ValueError):
        try:
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota == -1:
                quota = None
        except (FileNotFoundError, ValueError):
            pass
    cpus = os.cpu_count() or 1
    if quota is not None:
        cpus = max(1, min(cpus, (quota + period - 1) // period))
    mem_bytes = None
    try:
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
    except FileNotFoundError:
        try:
            lim = open("/sys/fs/cgroup/memory/memory.limit_in_bytes").read().strip()
        except FileNotFoundError:
            lim = None
    if lim and lim != "max":
        try:
            mem_bytes = int(lim)
        except ValueError:
            pass
    if mem_bytes:
        # v1's unlimited sentinel (9223372036854771712) means "no limit";
        # clamp to the host's memory like the shell helper does.
        host = _host_memory_bytes()
        if host and mem_bytes > host:
            mem_bytes = host
    elif mem_bytes is None:
        # No cgroup memory limit (plain desktop/macOS): budget against the
        # physical memory so local builds don't oversubscribe and OOM.
        mem_bytes = _host_memory_bytes()
    return cpus, mem_bytes


def _host_memory_bytes():
    """Physical memory of the host, or None when it cannot be determined."""
    import subprocess
    import sys

    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        except Exception:
            return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return None


def _build_and_package_bisheng(repo_dir, bisheng_compiler_path, build_type="Release", rebuild=True):
    """Configure, build and install AscendNPU-IR via its build.sh script."""
    repo_dir = Path(repo_dir)
    build_script = repo_dir / "build-tools" / "build.sh"
    if not build_script.exists():
        raise RuntimeError(f"Build script not found: {build_script}")
    # Respect MAX_JOBS when set (the wheel build passes it as a build arg);
    # otherwise fall back to the standalone heuristic tuned for the CI hosts.
    # Default parallelism from the effective cgroup quota: ~2.5 GiB per job
    # (heavy MLIR TUs peak at 3-5 GB), capped by the CPU quota. MAX_JOBS
    # overrides explicitly.
    cpus, mem_bytes = _cgroup_limits()
    default_jobs = cpus
    if mem_bytes:
        default_jobs = max(1, min(cpus, mem_bytes // (2500 * 1024 * 1024)))
    # (v1 unlimited sentinel shows up as the host MemTotal after clamping)
    max_jobs = min(max(1, int(os.getenv("MAX_JOBS") or default_jobs)), 32)
    _log(f"cgroup: {cpus} CPUs, mem limit "
         f"{round(mem_bytes / (1024**3), 1) if mem_bytes else 'unset'} GiB -> -j {max_jobs}")
    build_path = repo_dir / "build"
    if build_path.exists():
        shutil.rmtree(str(build_path))
    build_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(build_script), f"--build-type={str(build_type)}", "-o",
        str(build_path), "-t", "-j",
        str(max_jobs), f"--bisheng-compiler={str(bisheng_compiler_path)}", "--add-cmake-options",
        "-DLLVM_ENABLE_LLD=ON", "--build-triton", "--build-torch-mlir", "--build-shmem-template", "--bishengir-publish",
        "ON", "--collect-binary"
    ]
    # CI-only override: the manylinux wheel images need a clang++ wrapper
    # (-include cstdint -Wno-error) for the Ascend LLVM fork; they create it
    # themselves and point NPUIR_CXX_COMPILER at it. Local builds use the
    # toolchain's default C++ compiler (no --cxx-compiler).
    cxx_compiler = os.getenv("NPUIR_CXX_COMPILER", "")
    if cxx_compiler:
        _log(f"Using C++ compiler override: {cxx_compiler}")
        cmd.append(f"--cxx-compiler={cxx_compiler}")
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
    ascend_bishengir_path = _REPO_ROOT / "third_party" / "ascend" / "backend" / "bishengir"
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
    # Default ccache dir into the repo: container CI often mounts $HOME
    # read-only (/root/.cache is not writable). An explicit CCACHE_DIR from
    # the caller (e.g. a workflow cache dir) takes precedence.
    os.environ.setdefault("CCACHE_DIR", str(_REPO_ROOT / ".ccache"))

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
