"""AscendNPU-IR CI source policy; local builds keep the native gitlink pin."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def read_source(root: Path, ref_override=None):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate NPU-IR metadata key: {key}")
            result[key] = value
        return result

    info = json.loads((root / "cmake/npuir-info.json").read_text(encoding="utf-8"),
                      object_pairs_hook=unique_keys)
    if (not isinstance(info, dict) or set(info) != {"schema_version", "submodule", "ci_ref"}
            or type(info["schema_version"]) is not int or info["schema_version"] != 1):
        raise ValueError("Invalid NPU-IR source metadata schema")
    path = info["submodule"]
    if not isinstance(path, str) or re.fullmatch(r"third_party/ascend/[A-Za-z0-9_-]+", path) is None:
        raise ValueError("Invalid NPU-IR submodule path")
    repository = subprocess.check_output([
        "git", "config", "--file", str(root / ".gitmodules"), "--get", f"submodule.{path}.url",
    ], text=True).strip()
    repository = os.environ.get("NPUIR_REPO_URL") or repository
    if re.fullmatch(r"https://[^/\s]+/[A-Za-z0-9_./-]+\.git", repository) is None:
        raise ValueError("NPU-IR requires an HTTPS repository URL")
    ref = ref_override if ref_override is not None else info["ci_ref"]
    if (not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", ref)
            or ref.startswith("refs/") or ".." in ref
            or subprocess.run(["git", "check-ref-format", "--branch", ref], capture_output=True).returncode):
        raise ValueError("NPU-IR ref must be a branch name or a full commit SHA")
    return repository, ref


def resolve_revision(repository: str, ref: str):
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return ref.lower()
    for attempt in range(3):
        result = subprocess.run(["git", "ls-remote", "--exit-code", repository, f"refs/heads/{ref}"],
                                capture_output=True, text=True)
        lines = result.stdout.splitlines()
        if result.returncode == 0 and len(lines) == 1:
            sha, _, resolved_ref = lines[0].partition("\t")
            if re.fullmatch(r"[0-9a-fA-F]{40}", sha) and resolved_ref == f"refs/heads/{ref}":
                return sha.lower()
        if attempt < 2:
            print(f"[npuir-source] resolve attempt {attempt + 1}/3 failed; retrying", file=sys.stderr)
            time.sleep(10)
    raise RuntimeError(f"Cannot resolve NPU-IR branch {ref!r} at {repository}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--ref", default=os.environ.get("NPUIR_REPO_BRANCH") or None)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--repository", action="store_true")
    output.add_argument("--resolve", action="store_true")
    args = parser.parse_args()
    repository, ref = read_source(args.root, args.ref)
    print(repository if args.repository else resolve_revision(repository, ref))


if __name__ == "__main__":
    main()
