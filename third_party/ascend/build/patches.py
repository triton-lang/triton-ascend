import os
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]


def is_git_repo():
    return (_REPO_ROOT / ".git").is_dir()


def apply_patch(patch_path):
    try:
        subprocess.run(["git", "apply", patch_path], check=True, stdout=subprocess.DEVNULL, cwd=str(_REPO_ROOT))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"patch({patch_path}) failed,cmd={e.cmd}, retcode={e.returncode}") from e
    except FileNotFoundError:
        raise RuntimeError(f"patch({patch_path}) not found.")


def checkout_file(files):
    try:
        subprocess.run(["git", "checkout", "--"] + files, check=True, stdout=subprocess.DEVNULL, cwd=str(_REPO_ROOT))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"restore sources failed, list:{files}, cmd={e.cmd}, retcode={e.returncode}") from e


def get_default_version():
    version_file = _REPO_ROOT / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    return "3.6.0-dev"


def is_dev_mode():
    if os.getenv("IS_MANYLINUX", "FALSE").upper() not in ["ON", "1", "YES", "TRUE", "Y"]:
        return True
    if os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", ""):
        return True
    if "dev" in get_default_version():
        return True
    return False


def get_triton_ascend_patch_file():
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


def apply_triton_ascend_patch():
    patch_path = os.path.join("third_party", "ascend", "patch")
    dev_patch = os.path.join(patch_path, "triton-ascend-dev-3.6.0.patch")
    patch = os.path.join(patch_path, "triton-ascend-3.6.0.patch")
    patch_files, dev_patch_files = get_triton_ascend_patch_file()
    if is_dev_mode() and os.path.isfile(dev_patch):
        checkout_file(dev_patch_files)
        apply_patch(str(dev_patch))
    if os.path.isfile(patch):
        checkout_file(patch_files)
        apply_patch(str(patch))


def print_patch_restore_warning():
    """Warn that the build left patched (dirty) source files in the worktree.

    ``apply_triton_ascend_patch`` modifies in-tree Triton sources, so a
    subsequent ``git pull`` would fail with local changes. Users can restore
    those files with ``python3 restore_sources.py`` at the repository root
    (which runs ``git checkout --`` on the patched file list).
    """
    if not is_git_repo():
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
