import glob
import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

REPO = Path(__file__).resolve().parents[3]


def fingerprint(root, patterns):
    digest = hashlib.sha256()
    for pattern in patterns:
        for path in sorted(glob.glob(str(root / pattern))):
            digest.update(Path(path).read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize("workflow", [
    "Ascend950-ci.yml", "DynamicCVPipeline-ci.yml", "documentation.yml", "integration-tests-ascend.yml",
    "sync-upstream.yml"
])
def test_compiler_cache_inputs_follow_the_canonical_pin_and_patch_bytes(workflow, tmp_path):
    text = (REPO / ".github/workflows" / workflow).read_text(encoding="utf-8")
    keys = [key for key in re.findall(r"hashFiles\(([^)]+)\)", text) if "llvm-info.json" in key]
    assert keys
    for key in keys:
        patterns = re.findall(r"'([^']+)'", key)
        patterns = [pattern.removeprefix("pr/") for pattern in patterns]
        for pattern in patterns:
            for source in REPO.glob(pattern):
                if source.is_file():
                    target = tmp_path / source.relative_to(REPO)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        pin = tmp_path / "cmake/llvm-info.json"
        pin.parent.mkdir(parents=True, exist_ok=True)
        before = fingerprint(tmp_path, patterns)
        data = json.loads(pin.read_text())
        data["llvm_hash"] = "b" * 40
        pin.write_text(json.dumps(data))
        assert fingerprint(tmp_path, patterns) != before, f"{workflow}: canonical pin missing from cache inputs"

        json_pin = tmp_path / "cmake/json-version.txt"
        before = fingerprint(tmp_path, patterns)
        json_pin.write_text("v3.12.0\n")
        assert fingerprint(tmp_path, patterns) != before, f"{workflow}: JSON version missing from cache inputs"

        patch = next((tmp_path / "third_party/ascend/patch").glob("llvm_patch_*.patch"))
        before = fingerprint(tmp_path, patterns)
        patch.write_bytes(patch.read_bytes() + b"\nchanged patch\n")
        assert fingerprint(tmp_path, patterns) != before

        before = fingerprint(tmp_path, patterns)
        (tmp_path / "cmake/ascend-runtime-info.json").write_text('{"runtime": "changed"}')
        (tmp_path / "cmake/README.ascend-dependencies.md").write_text("documentation-only update")
        assert fingerprint(tmp_path, patterns) == before
