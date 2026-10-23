"""Consistency checks between the dependency BOM and its consumers.

The single source of truth for pinned dependencies is ``third_party/ascend/deps/``
(see ``third_party/ascend/deps/README.md``). This script verifies that every
consumer -- the ``cmake/`` mirrors, ``requirements.txt``, ``docker/Dockerfile``
build args, CI workflow literals and the release policy -- agrees with the
BOM, so a version bump that forgets one of the copies fails loudly instead of
silently desynchronizing.

Run locally:  python scripts/check_bom_consistency.py
"""

import glob
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPS_DIR = REPO_ROOT / "third_party" / "ascend" / "deps"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Beta CANN builder images referenced by CI before the corresponding runtime
# is official enough to be listed in runtime.json:
#   - 9.0.0-beta.2 / 9.2.0-beta.1: build-docker-image.yml matrix
#   - 9.2.0-beta.2: manylinux builder image (workflows/build/wheels.Dockerfile)
BETA_CANN_WHITELIST = {"9.0.0-beta.2", "9.2.0-beta.1", "9.2.0-beta.2"}

# Matches container tags ("cann:9.0.0-a3-ubuntu22.04-py3.12"), install paths
# ("cann-9.2.0-beta.2") and cache-key literals ("cann8.5.0").
CANN_VERSION_RE = re.compile(r"cann[:-]?(\d+(?:\.\d+)*(?:-beta\.\d+)?)")

errors = []


def _fail(check, message):
    errors.append(f"[{check}] {message}")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _line_no(text, pos):
    return text.count("\n", 0, pos) + 1


def check_llvm_info():
    llvm_info = _load_json(DEPS_DIR / "llvm-info.json")
    mirrored = _read(REPO_ROOT / "cmake" / "llvm-hash.txt").strip()
    if llvm_info["llvm_hash"] != mirrored:
        _fail("llvm-info", f"llvm_hash={llvm_info['llvm_hash']} != cmake/llvm-hash.txt={mirrored}")
    patch_dir = REPO_ROOT / "third_party" / "ascend" / "patch"
    on_disk = sorted(p.name for p in patch_dir.glob("llvm_patch_*.patch") if p.is_file())
    declared = sorted(llvm_info.get("patches", []))
    if declared != on_disk:
        _fail("llvm-info", f"patches manifest={declared} != third_party/ascend/patch contents={on_disk}")


def check_toolchain():
    toolchain = _load_json(DEPS_DIR / "toolchain.json")
    mirrored = _read(REPO_ROOT / "cmake" / "json-version.txt").strip()
    if toolchain["json"] != mirrored:
        _fail("toolchain", f"toolchain.json json={toolchain['json']} != cmake/json-version.txt={mirrored}")


def check_static_pins(runtime):
    torch_npu_versions = {r["torch_npu"] for r in runtime["runtimes"]}
    cann_versions = {r["cann"] for r in runtime["runtimes"]}

    pin = re.search(r"^torch-npu==(\S+)", _read(REPO_ROOT / "requirements.txt"), re.MULTILINE)
    if pin is None:
        _fail("static-pins", "requirements.txt has no 'torch-npu==<version>' pin")
    elif pin.group(1) not in torch_npu_versions:
        _fail(
            "static-pins", f"requirements.txt torch-npu=={pin.group(1)} is not a torch_npu value in runtime.json "
            f"(known: {sorted(torch_npu_versions)})")

    dockerfile = _read(REPO_ROOT / "docker" / "Dockerfile")
    pin = re.search(r"^ARG\s+TORCH_NPU_VERSION=(\S+)", dockerfile, re.MULTILINE)
    if pin is None:
        _fail("static-pins", "docker/Dockerfile has no 'ARG TORCH_NPU_VERSION=' default")
    elif pin.group(1) not in torch_npu_versions:
        _fail(
            "static-pins", f"docker/Dockerfile TORCH_NPU_VERSION={pin.group(1)} is not a torch_npu value in "
            f"runtime.json (known: {sorted(torch_npu_versions)})")

    pin = re.search(r"^ARG\s+CANN_BASE_IMAGE=(\S+)", dockerfile, re.MULTILINE)
    if pin is None:
        _fail("static-pins", "docker/Dockerfile has no 'ARG CANN_BASE_IMAGE=' default")
    else:
        cann = CANN_VERSION_RE.search(pin.group(1))
        if cann is not None and cann.group(1) not in cann_versions:
            _fail(
                "static-pins", f"docker/Dockerfile CANN_BASE_IMAGE pins CANN {cann.group(1)}, not a cann value in "
                f"runtime.json (known: {sorted(cann_versions)})")


def _workflow_files():
    for path in sorted(WORKFLOWS_DIR.rglob("*")):
        if path.is_file():
            yield path


def _glob_exists(relative):
    """Check a (possibly checkout-prefixed) hashFiles() argument against the repo.

    Some workflows check the PR out under a ``pr/`` directory, so arguments
    like ``pr/cmake/llvm-hash.txt`` must match after stripping the prefix.
    """
    parts = Path(relative).parts
    for i in range(len(parts)):
        candidate = "/".join(parts[i:])
        if glob.glob(os.path.join(REPO_ROOT, candidate), recursive=True):
            return True
    return False


def check_workflows(runtime):
    cann_versions = {r["cann"] for r in runtime["runtimes"]}
    known_cann = cann_versions | {runtime["default_cann"]} | BETA_CANN_WHITELIST
    list_re = re.compile(r"cann_version[\"']?\s*:\s*\[([^\]]*)\]")
    scalar_re = re.compile(r"cann_version[\"']?\s*:\s*[\"']?([\w.\-]+)")
    hash_files_re = re.compile(r"hashFiles\(([^)]*)\)")

    for path in _workflow_files():
        text = _read(path)
        rel = path.relative_to(REPO_ROOT).as_posix()

        for match in CANN_VERSION_RE.finditer(text):
            version = match.group(1)
            if version not in known_cann:
                _fail(
                    "workflows-cann", f"{rel}:{_line_no(text, match.start())} references CANN {version}, which is "
                    f"not declared in runtime.json or the beta whitelist")

        for line_no, line in enumerate(text.splitlines(), start=1):
            list_match = list_re.search(line)
            if list_match is not None:
                for item in list_match.group(1).split(","):
                    item = item.strip().strip("\"'")
                    if item and item not in known_cann:
                        _fail("workflows-cann",
                              f"{rel}:{line_no} cann_version entry '{item}' not declared in runtime.json")
                continue
            scalar_match = scalar_re.search(line)
            if scalar_match is not None and scalar_match.group(1) not in known_cann:
                _fail("workflows-cann", f"{rel}:{line_no} cann_version '{scalar_match.group(1)}' not in runtime.json")

        for match in hash_files_re.finditer(text):
            for arg in match.group(1).split(","):
                arg = arg.strip().strip("\"'")
                if arg and not _glob_exists(arg):
                    _fail(
                        "workflows-hashfiles", f"{rel}:{_line_no(text, match.start())} hashFiles argument "
                        f"'{arg}' matches no file in the repo")


def check_release_policy(runtime):
    torch_versions = {r["torch"] for r in runtime["runtimes"]}
    lines = _read(REPO_ROOT / "docs" / "en" / "community" / "release_policy.md").splitlines()

    # The "main Branch Mapping" header uniquely contains "Triton commit hash";
    # the later compatibility matrix table does not.
    header_idx = next(
        (i for i, line in enumerate(lines) if "Triton commit hash" in line and "CANN" in line and "PyTorch" in line),
        None)
    if header_idx is None:
        _fail("release-policy",
              "cannot find the 'main Branch Mapping' table header in docs/en/community/release_policy.md")
        return
    columns = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
    try:
        cann_col = columns.index("CANN")
        torch_col = columns.index("PyTorch")
    except ValueError:
        _fail("release-policy", "'main Branch Mapping' table header lacks a CANN or PyTorch column")
        return

    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        if set(stripped) <= set("|-: "):
            continue
        cells = [c.strip().strip("`") for c in stripped.strip("|").split("|")]
        if not cells or cells[0] != "main":
            continue
        cann = cells[cann_col] if cann_col < len(cells) else ""
        torch = cells[torch_col] if torch_col < len(cells) else ""
        if cann != runtime["default_cann"]:
            _fail(
                "release-policy", f"release_policy.md maps 'main' to CANN {cann}, but runtime.json "
                f"default_cann={runtime['default_cann']}")
        if torch not in torch_versions:
            _fail(
                "release-policy", f"release_policy.md maps 'main' to PyTorch {torch}, which is not a torch value "
                f"in runtime.json (known: {sorted(torch_versions)})")
        return
    _fail("release-policy", "no 'main' row found in the 'main Branch Mapping' table")


def main():
    check_llvm_info()
    check_toolchain()
    runtime = _load_json(DEPS_DIR / "runtime.json")
    check_static_pins(runtime)
    check_workflows(runtime)
    check_release_policy(runtime)

    if errors:
        print("BOM consistency check FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("BOM consistency check passed (deps/ BOM, cmake mirrors, static pins, workflows, release policy).")


if __name__ == "__main__":
    main()
