"""Build-time Proton patches and synchronization with the source-review branch.

Keep Proton changes as source commits on codex/proton-upstream-boundaries-20261103.
On proton_patch, first restore the generated sources, synchronize any Ascend-side
changes, then run this script with `export --source <source-commit>` and `check`.
Commit the patch and manifest, not the generated third_party/proton changes.
Use --base when updating the upstream Proton baseline for a Triton upgrade.

Normal builds call set_proton_patch before CMake. ON applies the patch; OFF
restores it. A failed build leaves the patch applied so it can be inspected or
retried. `restore` removes only the patch and refuses conflicting local edits.
Do not build ON and OFF concurrently in the same source tree.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH_DIR = Path("third_party/ascend/patch/proton")
SOURCE_BRANCH = "codex/proton-upstream-boundaries-20261103"


def _git(root, *args, check=True, env=None):
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to apply the Ascend Proton patch") from exc
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or "Git command failed")
    return result


def _read_patch(root):
    directory = Path(root) / PATCH_DIR
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    patch = directory / "proton.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise RuntimeError("Proton patch checksum mismatch; regenerate it with proton_patch.py export")
    return patch, manifest


def set_proton_patch(enabled, root=REPO_ROOT):
    """Apply or reverse the complete patch without resetting user changes."""
    root = Path(root).resolve()
    patch, _ = _read_patch(root)
    # A reverse check succeeds for an already-patched tree, and vice versa.
    desired = ["--reverse"] if enabled else []
    current = [] if enabled else ["--reverse"]
    if _git(root, "apply", "--check", "--whitespace=nowarn", *desired, str(patch), check=False).returncode == 0:
        return False
    check = _git(root, "apply", "--check", "--whitespace=nowarn", *current, str(patch), check=False)
    if check.returncode:
        raise RuntimeError(
            "Cannot change the Ascend Proton patch state: sources are partially patched or conflict with local "
            "edits. No files were changed. Preserve your edits and resolve the conflict before rebuilding.\n" +
            check.stderr.decode(errors="replace"))
    _git(root, "apply", "--whitespace=nowarn", *current, str(patch))
    print("Ascend Proton patch " + ("applied" if enabled else "restored"))
    return True


def export_patch(root=REPO_ROOT, *, source=SOURCE_BRANCH, base=None):
    """Generate the entire Proton delta from committed source-review changes."""
    root = Path(root).resolve()
    directory = root / PATCH_DIR
    manifest_path = directory / "manifest.json"
    if base is None:
        base = json.loads(manifest_path.read_text(encoding="utf-8"))["base_commit"]
    base = _git(root, "rev-parse", "--verify", f"{base}^{{commit}}").stdout.decode().strip()
    source = _git(root, "rev-parse", "--verify", f"{source}^{{commit}}").stdout.decode().strip()
    if _git(root, "diff", "--quiet", base, "--", "third_party/proton", check=False).returncode:
        raise RuntimeError("Restore third_party/proton to the upstream baseline before exporting a new patch")
    patch = _git(root, "diff", "--binary", "--full-index", "--no-renames", base, source, "--",
                 "third_party/proton").stdout
    if not patch:
        raise RuntimeError("The source revision contains no Proton changes")
    manifest = {
        "base_commit": base,
        "source_commit": source,
        "source_branch": SOURCE_BRANCH,
        "base_tree": _git(root, "rev-parse", f"{base}:third_party/proton").stdout.decode().strip(),
        "source_tree": _git(root, "rev-parse", f"{source}:third_party/proton").stdout.decode().strip(),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "proton.patch").write_bytes(patch)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Exported Proton patch from {source}")


def check_patch(root=REPO_ROOT, *, source=None):
    """Check exact tree equality using a temporary index, leaving sources alone."""
    root = Path(root).resolve()
    patch, manifest = _read_patch(root)
    base_tree = _git(root, "rev-parse", f"{manifest['base_commit']}:third_party/proton").stdout.decode().strip()
    if base_tree != manifest["base_tree"]:
        raise RuntimeError("Proton baseline tree does not match the manifest")
    if _git(root, "diff", "--cached", "--quiet", manifest["base_commit"], "--", "third_party/proton",
            check=False).returncode:
        raise RuntimeError("The staged Proton sources differ from the baseline; do not commit generated patch changes")
    source = source or manifest["source_commit"]
    expected = _git(root, "rev-parse", f"{source}:third_party/proton").stdout.decode().strip()
    if expected != manifest["source_tree"]:
        raise RuntimeError("Proton source branch has changed; regenerate its patch before committing")
    with tempfile.TemporaryDirectory(prefix="proton-patch-") as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        _git(root, "read-tree", manifest["base_commit"], env=env)
        _git(root, "apply", "--cached", "--whitespace=nowarn", str(patch), env=env)
        actual = _git(root, "write-tree", "--prefix=third_party/proton/", env=env).stdout.decode().strip()
    if actual != expected:
        raise RuntimeError("Applying the Proton patch does not reproduce the source-review tree")
    print(f"Proton patch matches source tree {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("apply", help="Apply the patch without enabling a build")
    commands.add_parser("restore", help="Safely restore upstream Proton sources")
    export = commands.add_parser("export", help="Regenerate the patch from the source-review branch")
    export.add_argument("--source", default=SOURCE_BRANCH)
    export.add_argument("--base")
    check = commands.add_parser("check", help="Verify the patch reproduces the source-review tree")
    check.add_argument("--source")
    args = parser.parse_args()
    if args.command == "export":
        export_patch(args.root, source=args.source, base=args.base)
    elif args.command == "check":
        check_patch(args.root, source=args.source)
    else:
        set_proton_patch(args.command == "apply", args.root)


if __name__ == "__main__":
    main()
