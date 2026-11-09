import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "workflow",
    ["integration-tests-ascend.yml", "ascend-documentation.yml", "sync-upstream.yml", "DynamicCVPipeline-ci.yml"])
@pytest.mark.parametrize("mode", ["valid", "missing-file", "missing-checksum", "unsupported-arch", "checksum-mismatch"])
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
    prefix = tmp_path / "bisheng"
    command = command.replace("/usr/local/bisheng", prefix.as_posix())
    installer = (b'#!/usr/bin/env bash\nprintf installed > installed.txt\n'
                 b'mkdir -p "$TEST_BISHENG_PREFIX/tools/bishengir/bin"\n'
                 b'printf "#!/bin/sh\\nexit 0\\n" > "$TEST_BISHENG_PREFIX/tools/bishengir/bin/bishengir-compile"\n'
                 b'chmod +x "$TEST_BISHENG_PREFIX/tools/bishengir/bin/bishengir-compile"\n')
    data = {"bisheng": {"x86_64": {"file": "fixture.run", "sha256": hashlib.sha256(installer).hexdigest()}}}
    if mode == "missing-checksum":
        del data["bisheng"]["x86_64"]["sha256"]
    elif mode == "unsupported-arch":
        data["bisheng"] = {}
    elif mode == "checksum-mismatch":
        data["bisheng"]["x86_64"]["sha256"] = "0" * 64
    if mode != "missing-file":
        (metadata / "bisheng-info.json").write_text(json.dumps(data))
    (tmp_path / "installer.sh").write_bytes(installer)
    harness = '''set -euo pipefail
uname() { printf 'x86_64\\n'; }
curl() { touch downloaded.txt; cp "$TEST_INSTALLER" "${@: -1}"; }
'''
    result = subprocess.run(
        [bash, "-c", harness + command], cwd=tmp_path, env={
            **os.environ, "TEST_INSTALLER": (tmp_path / "installer.sh").as_posix(), "TEST_BISHENG_PREFIX":
            prefix.as_posix(), "GITHUB_PATH": (tmp_path / "github-path").as_posix()
        }, capture_output=True, text=True)
    if mode == "valid":
        assert result.returncode == 0, result.stderr
        assert (root / "installed.txt").read_text() == "installed"
        assert (tmp_path / "github-path").read_text().strip() == f"{prefix.as_posix()}/tools/bishengir/bin"
    else:
        assert result.returncode != 0
        assert (root / "downloaded.txt").exists() is (mode == "checksum-mismatch")
        assert not (root / "installed.txt").exists()


def test_integration_installer_retains_tar_payload_support(tmp_path):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None or shutil.which("jq") is None:
        pytest.skip("Bash and jq are required")
    text = (REPO / ".github/workflows/integration-tests-ascend.yml").read_text(encoding="utf-8")
    block = text.split("      - name: Download and install Bisheng compiler\n", 1)[1].split("\n      - name:", 1)[0]
    prefix = tmp_path / "bisheng"
    command = textwrap.dedent(block.split("        run: |\n", 1)[1]).replace("/usr/local/bisheng", prefix.as_posix())
    payload = tmp_path / "fixture.tar.gz"
    with tarfile.open(payload, "w:gz") as archive:
        content = b"#!/bin/sh\nexit 0\n"
        info = tarfile.TarInfo("bin/bishengir-compile")
        info.size = len(content)
        info.mode = 0o755
        archive.addfile(info, io.BytesIO(content))
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    (metadata / "bisheng-info.json").write_text(
        json.dumps(
            {"bisheng": {"x86_64": {"file": payload.name, "sha256":
                                    hashlib.sha256(payload.read_bytes()).hexdigest()}}}))
    harness = 'uname() { echo x86_64; }\ncurl() { cp "$TEST_PAYLOAD" "${@: -1}"; }\n'
    script = tmp_path / "install.sh"
    script.write_text(harness + command, encoding="utf-8", newline="\n")
    result = subprocess.run(
        [bash, "-e", "-o", "pipefail", script.as_posix()], cwd=tmp_path,
        env={**os.environ, "TEST_PAYLOAD": payload.as_posix(), "GITHUB_PATH":
             (tmp_path / "github-path").as_posix()}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (prefix / "tools/bishengir/bin/bishengir-compile").read_bytes() == content
