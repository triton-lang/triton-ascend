import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "document", ["README.md", "README_zh.md", "docs/en/installation_guide.md", "docs/zh/installation_guide.md"])
def test_documented_checkout_uses_the_local_manifest(document, tmp_path):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required to execute the documented shell example")
    text = (REPO / document).read_text(encoding="utf-8")
    assert "llvm_patch_${LLVM_COMMIT:0:8}" not in text
    assert "raw.githubusercontent.com/triton-lang/triton-ascend/main/third_party/ascend/patch" not in text
    start = text.rfind("\n", 0, text.index('TRITON_ROOT="$(pwd)"')) + 1
    ends = [text.find(marker, start) for marker in ("export LLVM_INSTALL_PREFIX", "\n```", "\n    ```")]
    end = min(value for value in ends if value >= 0)
    command = textwrap.dedent(text[start:end])

    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=upstream, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("config", "core.hooksPath", str(tmp_path / "no-hooks"))
    (upstream / "example.txt").write_text("original\n", newline="\n")
    git("add", "example.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD").stdout.strip()

    root = tmp_path / "ascend"
    build = root / "cmake"
    build.mkdir(parents=True)
    deps = build
    patch_dir = root / "third_party/ascend/patch"
    patch_dir.mkdir(parents=True)
    patch_name = "llvm_patch_fixture.patch"
    (deps / "llvm-info.json").write_text(
        json.dumps({"schema_version": 1, "llvm_hash": revision, "patches": [patch_name]}))
    (patch_dir / patch_name).write_text(
        "diff --git a/example.txt b/example.txt\n--- a/example.txt\n+++ b/example.txt\n"
        "@@ -1 +1 @@\n-original\n+patched\n", newline="\n")
    command = command.replace("https://github.com/llvm/llvm-project.git", f'"{upstream.as_posix()}" llvm-project')
    wrapper = 'set -eu\npython3() { "$TEST_PYTHON" "$@"; }\n' + command
    subprocess.run([bash, "-c", wrapper], cwd=root, env={**os.environ, "TEST_PYTHON": Path(sys.executable).as_posix()},
                   check=True, capture_output=True, text=True)
    assert (root / "llvm-project/example.txt").read_text() == "patched\n"
