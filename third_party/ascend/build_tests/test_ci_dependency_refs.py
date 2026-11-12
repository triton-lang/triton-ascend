import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("branch", ["main", "release/3.6.0"])
def test_runner_query_uses_the_current_branch_and_workflow_file(tmp_path, branch):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required")
    text = (REPO / ".github/workflows/ascend-runner-preparation.yml").read_text(encoding="utf-8")
    step = text.split("      - name: Detect if enough time has passed since last post-submit run", 1)[1]
    match = re.search(r"        run: \|\n((?:          .*\n|\n)+)", step)
    script = textwrap.dedent(match.group(1))
    # Supply stand-ins only for the transport/parser; execute the actual query construction.
    prefix = 'curl() { printf "%s\\n" "$@" >&2; }; jq() { printf null; }; '
    result = subprocess.run(
        [bash, "-c", prefix + script], cwd=tmp_path, env={
            **os.environ, "GITHUB_TOKEN": "test-token", "GITHUB_REPOSITORY": "example/triton", "GITHUB_REF_NAME":
            branch, "GITHUB_API_URL": "https://api.github.com", "GITHUB_ENV": (tmp_path / "env").as_posix()
        }, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "/actions/workflows/ascend-ci.yml/runs" in result.stderr
    assert "branch=" + branch in result.stderr
