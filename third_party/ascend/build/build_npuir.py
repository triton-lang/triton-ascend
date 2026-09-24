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
    return (Path(dir_path) / ".git").is_dir()


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


def _nested_submodule_gitlinks(npuir_dir):
    """Map {relative_path: sha} for the nested submodules the npuir pin records."""
    out = subprocess.check_output(
        ["git", "ls-tree", "HEAD", "third-party/"],
        cwd=str(npuir_dir),
        text=True,
    )
    links = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 4:
            links[fields[3]] = fields[2]
    return links


def _fetch_nested_via_sha(npuir_dir, path, sha):
    """Fetch one nested repo at its exact SHA.

    Direct depth-1 fetches of an exact commit SHA work where branch clones do
    not (gitcode serves `want <sha>` requests but branch clones fail).
    """
    url = subprocess.check_output(
        ["git", "-C", str(npuir_dir), "config", "-f", ".gitmodules", "--get", f"submodule.{path}.url"],
        text=True,
    ).strip()
    dest = npuir_dir / path
    shutil.rmtree(dest, ignore_errors=True)
    _run_with_retry(["git", "init", "-q", str(dest)])
    _run_with_retry(["git", "-C", str(dest), "remote", "add", "origin", url])
    # gitcode intermittently answers 503 (server-side throttling) on large
    # depth-1 fetches; retry with a longer window than the default.
    _run_with_retry(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", sha], retries=5, interval=10)
    _run_with_retry(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"])


def _init_nested_submodule_sources(npuir_dir):
    """Fetch the nested submodule sources (llvm-project, torch-mlir, shmem)
    via direct depth-1 SHA fetches from the gitcode remotes.
    """
    links = _nested_submodule_gitlinks(npuir_dir)
    if not links:
        raise RuntimeError("No nested submodules recorded in the AscendNPU-IR pin.")
    for path, sha in links.items():
        dest = npuir_dir / path
        if _is_submodule_initialized(dest):
            _log(f"{path} already initialized, skipping")
            continue
        _log(f"Fetching {path} @ {sha}")
        _fetch_nested_via_sha(npuir_dir, path, sha)


def _init_npuir_repo():
    """Initialize the AscendNPU-IR submodule and its nested submodule sources.

    AscendNPU-IR depends on LLVM and Torch-MLIR. The recursive gitcode clone
    is not reliable from the CI network, so the nested sources are fetched
    with direct depth-1 SHA pulls from the gitcode remotes.
    """
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

    if not _is_submodule_initialized(_NPUIR_DIR):
        raise RuntimeError(f"AscendNPU-IR submodule initialization failed: {_NPUIR_DIR} is not git repository.")
    _log("AscendNPU-IR repository initialized.")

    _init_nested_submodule_sources(_NPUIR_DIR)
    _log("Nested submodule sources ready.")


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


def _create_clang_wrapper():
    """Wrap clang++ with the workarounds the Ascend LLVM fork needs.

    The Ascend LLVM fork excludes <cstdint> in Affine
    ValueBoundsOpInterfaceImpl.h under BSPUB_DAVINCI_BISHENGIR_A5, and the
    clang in the manylinux CI images resolves a libstdc++ that no longer
    provides stdint.h transitively — so force-include cstdint. clang 21's
    new -Wall warnings (e.g. -Wunnecessary-virtual-specifier) also trip
    -Werror on LLVM 19 code, so demote warnings to non-fatal (clang
    last-wins semantics; warnings still print).
    """
    wrapper = Path("/tmp/clang++-cstdint")
    wrapper.write_text('#!/bin/bash\nexec /usr/bin/clang++ -include cstdint "$@" -Wno-error\n')
    wrapper.chmod(0o755)
    return wrapper


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
        str(max_jobs), f"--bisheng-compiler={str(bisheng_compiler_path)}",
        f"--cxx-compiler={str(_create_clang_wrapper())}", "--add-cmake-options", "-DLLVM_ENABLE_LLD=ON",
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


if __name__ == "__main__":
    build_npuir()
