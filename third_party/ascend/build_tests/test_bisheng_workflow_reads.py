import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "workflow", ["integration-tests-ascend.yml", "documentation.yml", "sync-upstream.yml", "DynamicCVPipeline-ci.yml"])
@pytest.mark.parametrize("mode", ["valid", "missing-file", "missing-checksum", "unsupported-arch"])
def test_actual_bisheng_install_step_requires_metadata_before_download(workflow, mode, tmp_path):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None or shutil.which("jq") is None:
        pytest.skip("Bash and jq are required to execute the workflow installer")
    text = (REPO / ".github/workflows" / workflow).read_text(encoding="utf-8")
    name = "Install Bisheng compiler" if workflow == "DynamicCVPipeline-ci.yml" else "Download and install Bisheng compiler"
    start = text.index(f"      - name: {name}")
    end = text.index("\n      - name:", start + 1)
    block = text[start:end]
    command = textwrap.dedent(block.split("        run: |\n", 1)[1])
    root = tmp_path / "pr" if workflow == "DynamicCVPipeline-ci.yml" else tmp_path
    metadata = root / "cmake"
    metadata.mkdir(parents=True)
    installer = b'#!/usr/bin/env bash\nprintf installed > installed.txt\n'
    data = {"bisheng": {"x86_64": {"file": "fixture.run", "sha256": hashlib.sha256(installer).hexdigest()}}}
    if mode == "missing-checksum":
        del data["bisheng"]["x86_64"]["sha256"]
    elif mode == "unsupported-arch":
        data["bisheng"] = {}
    if mode != "missing-file":
        (metadata / "bisheng-info.json").write_text(json.dumps(data))
    (tmp_path / "installer.sh").write_bytes(installer)
    harness = '''set -euo pipefail
uname() { printf 'x86_64\\n'; }
curl() { touch downloaded.txt; cp "$TEST_INSTALLER" bisheng.run; }
'''
    result = subprocess.run(
        [bash, "-c", harness + command], cwd=tmp_path, env={
            **os.environ, "TEST_INSTALLER": (tmp_path / "installer.sh").as_posix(), "GITHUB_PATH":
            (tmp_path / "github-path").as_posix()
        }, capture_output=True, text=True)
    if mode == "valid":
        assert result.returncode == 0, result.stderr
        assert (root / "installed.txt").read_text() == "installed"
        assert (tmp_path / "github-path").read_text().strip() == "/usr/local/bisheng/tools/bishengir/bin"
    else:
        assert result.returncode != 0
        assert not (root / "downloaded.txt").exists()
        assert not (root / "installed.txt").exists()
