import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("mode", ["metadata", "override", "invalid-json"])
def test_wrapper_reads_its_checkout_and_preserves_explicit_override(tmp_path, mode):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required to execute the LLVM wrapper")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    if mode != "override":
        (metadata /
         "llvm-info.json").write_text("{bad" if mode == "invalid-json" else json.dumps({"llvm_hash": "b" * 40}))
    text = (REPO / "scripts/build-llvm-project.sh").read_text(encoding="utf-8")
    script = text[:text.index("LLVM_PROJECT_PATH=")] + '\nprintf "%s\\n" "$LLVM_COMMIT_HASH"\n'
    env = {**os.environ, "PYTHON": Path(sys.executable).as_posix()}
    env.pop("LLVM_COMMIT_HASH", None)
    if mode == "override":
        env["LLVM_COMMIT_HASH"] = "explicit-ref"
    result = subprocess.run([bash, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == (1 if mode == "invalid-json" else 0), result.stderr
    expected = "" if mode == "invalid-json" else ("explicit-ref\n" if mode == "override" else "b" * 40 + "\n")
    assert result.stdout == expected
