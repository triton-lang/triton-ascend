"""Audit the explicitly migrated LLVM, JSON-library and Bisheng inputs.

Production consumers read their metadata directly. This script is not a reader
API, a runtime compatibility matrix or a release artifact verifier.
"""
import glob
import json
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPS_DIR = REPO_ROOT / "cmake"
WORKFLOWS_DIR = REPO_ROOT / ".github/workflows"
SCHEMA_VERSION = 1
CHECKOUT_ROOTS = ("", "pr/")
DOC_SITES = ("README.md", "README_zh.md", "docs/en/installation_guide.md", "docs/zh/installation_guide.md")
errors = []


def _fail(check, message):
    errors.append(f"[{check}] {message}")


def _load_json(path):

    def unique_keys(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(f"{path}: duplicate key {key}")
            data[key] = value
        return data

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=unique_keys)
    if not isinstance(data, dict) or type(
            data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path}: expected an object with integer schema_version {SCHEMA_VERSION}")
    return data


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _line_no(text, pos):
    return text.count("\n", 0, pos) + 1


def _load_llvm_info():
    root = REPO_ROOT
    revision = _read(DEPS_DIR / "llvm-hash.txt").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("llvm-hash.txt must contain exactly 40 hexadecimal characters")
    patch_dir = root / "third_party/ascend/patch"
    patches = sorted(path.name for path in patch_dir.glob("llvm_patch_*.patch") if path.is_file())
    if any(re.fullmatch(r"llvm_patch_[A-Za-z0-9_.-]+\.patch", name) is None for name in patches):
        raise ValueError("LLVM patches must have safe basenames")
    info = {"llvm_hash": revision, "patches": patches}
    return info


def check_llvm_info():
    try:
        _load_llvm_info()
    except (OSError, ValueError) as exc:
        _fail("llvm-info", str(exc))


def _load_bisheng_info():
    entries = {}
    for line in _read(DEPS_DIR / "bisheng-version.txt").splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(": ")
        if not separator or key in entries:
            raise ValueError("bisheng-version.txt requires unique key: value entries")
        entries[key] = value.strip()
    expected = {"x86_64", "x86_64_sha256", "aarch64", "aarch64_sha256"}
    if set(entries) != expected:
        raise ValueError("bisheng-version.txt requires a file and sha256 for both architectures")
    for arch in ("x86_64", "aarch64"):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:run|tar\.gz)", entries[arch]) is None:
            raise ValueError(f"bisheng {arch} needs a safe .run or .tar.gz basename")
        if re.fullmatch(r"[0-9a-fA-F]{64}", entries[arch + "_sha256"]) is None:
            raise ValueError(f"bisheng {arch} needs a full SHA-256")
    return entries


def check_toolchain():
    try:
        _load_bisheng_info()
        version = _read(DEPS_DIR / "json-version.txt").strip()
        if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", version) is None:
            raise ValueError("json-version.txt must contain a release tag such as v3.11.3")
    except (OSError, ValueError) as exc:
        _fail("toolchain", str(exc))


def check_optional_sources():
    try:
        info = _load_json(DEPS_DIR / "triton-distributed-info.json")
        if (set(info) != {"schema_version", "repository", "revision"}
                or not isinstance(info["repository"], str)
                or re.fullmatch(r"https://[^/\s]+/[A-Za-z0-9_./-]+\.git", info["repository"]) is None
                or not isinstance(info["revision"], str)
                or re.fullmatch(r"[0-9a-f]{40}", info["revision"]) is None):
            raise ValueError("Invalid Triton-Distributed source metadata")
        # Load only this Ascend-specific reader, without importing setup hooks.
        import importlib.util
        spec = importlib.util.spec_from_file_location("bom_npuir_source", Path(__file__).with_name("npuir_source.py"))
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        reader.read_source(REPO_ROOT)
        if (DEPS_DIR / "bisheng-info.json").exists():
            raise ValueError("Duplicate Bisheng pin; use bisheng-version.txt only")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        _fail("optional-sources", str(exc))


def _glob_exists(relative):
    """Check a (possibly checkout-prefixed) hashFiles() argument against the repo.

    Some workflows check the PR out under a ``pr/`` directory, so arguments
    like ``pr/cmake/llvm-hash.txt`` must match after stripping the prefix.
    Only the declared :data:`CHECKOUT_ROOTS` prefixes may be stripped; an
    arbitrary leading component (a typo like ``wrng/requirements.txt``) must
    keep failing even when the remaining path exists.
    """
    if relative.startswith(("/", "\\")) or "\\" in relative or ":" in relative or ".." in relative.split("/"):
        return False
    for root in CHECKOUT_ROOTS:
        if not relative.startswith(root):
            continue
        candidate = relative[len(root):]
        if glob.glob(os.path.join(REPO_ROOT, candidate), recursive=True):
            return True
    return False


def check_docs_llvm_hash():
    """Check reintroduced literals against the local pin and patch files."""
    info = _load_llvm_info()
    llvm_hash = info["llvm_hash"]
    checkout_re = re.compile(r"git checkout ([0-9a-f]{40})\b")
    patch_re = re.compile(r"llvm_patch_([0-9a-f]{7,40})\.patch")
    for rel in DOC_SITES:
        text = _read(REPO_ROOT / rel)
        for match in checkout_re.finditer(text):
            if match.group(1) != llvm_hash:
                _fail(
                    "docs-llvm-hash",
                    f"{rel}:{_line_no(text, match.start())} hardcodes 'git checkout {match.group(1)}', "
                    f"but the BOM pins {llvm_hash}; read cmake/llvm-hash.txt instead")
        for match in patch_re.finditer(text):
            if match.group(0) not in info["patches"]:
                _fail(
                    "docs-llvm-hash",
                    f"{rel}:{_line_no(text, match.start())} references patch {match.group(0)}, which does "
                    f"not appear in the local LLVM patch set")


def check_workflow_cache_paths():
    for path in sorted(WORKFLOWS_DIR.rglob("*")):
        if not path.is_file():
            continue
        for match in re.finditer(r"hashFiles\(([^)]*)\)", _read(path)):
            for arg in match.group(1).split(","):
                arg = arg.strip().strip("\"'")
                if arg and not _glob_exists(arg):
                    _fail("workflows-hashfiles", f"{path.relative_to(REPO_ROOT)}: {arg!r} matches no file")


def check_release_llvm():
    info = _load_llvm_info()
    text = _read(REPO_ROOT / "docs/en/community/release_policy.md")
    row = next((line for line in text.splitlines() if re.match(r"\|\s*`main`\s*\|", line)), "")
    commits = re.findall(r"\[([0-9a-f]{7,40})\]\(https://github.com/llvm/llvm-project/commit/([0-9a-f]{7,40})\)", row)
    if len(commits) != 1 or any(not info["llvm_hash"].startswith(value) for value in commits[0]):
        _fail("release-policy", "main LLVM differs from cmake/llvm-hash.txt")
    patches = set(re.findall(r"llvm_patch_[A-Za-z0-9_.-]+\.patch", row))
    if patches != set(info["patches"]):
        _fail("release-policy", "main patch list differs from cmake/llvm-hash.txt")


def main():
    errors.clear()
    try:
        check_llvm_info()
        check_toolchain()
        check_optional_sources()
        check_docs_llvm_hash()
        check_workflow_cache_paths()
        check_release_llvm()
    except (OSError, ValueError) as exc:
        _fail("metadata", str(exc))
    if errors:
        print("Build dependency consistency check failed:\n  - " + "\n  - ".join(errors))
        raise SystemExit(1)
    print("Build dependency consistency check passed (LLVM, JSON library, Bisheng, optional sources, docs and cache paths).")


if __name__ == "__main__":
    main()
