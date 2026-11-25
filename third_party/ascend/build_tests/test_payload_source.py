import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("mode", ["matching", "foreign", "missing-sha", "missing-payload", "cache-hit"])
def test_real_payload_step_checks_source_before_using_binaries(tmp_path, mode):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required")
    text = (REPO / ".github/workflows/ascend-wheels.yml").read_text(encoding="utf-8")
    step = text.split("      - name: Prepare npuir payload\n", 1)[1].split("\n      - name:", 1)[0]
    command = textwrap.dedent(step.split("        run: |\n", 1)[1])
    cache = tmp_path / "cache/payload"
    command = command.replace("${{ steps.npuir-cache.outputs.cache_hit }}", "true" if mode == "cache-hit" else "false")
    command = command.replace("${{ steps.npuir-cache.outputs.npuir_sha }}", "a" * 40)
    command = command.replace("${{ steps.npuir-cache.outputs.cache_dir }}", cache.as_posix())
    command = command.replace("${{ matrix.config.arch }}", "x86_64")
    if mode not in {"missing-payload", "cache-hit"}:
        payload = tmp_path / "npuir-artifacts/payload"
        (payload / "bin").mkdir(parents=True)
        (payload / "lib").mkdir()
        (payload / "bin/bishengir-compile").write_bytes(b"fixture compiler")
        if mode != "missing-sha":
            (payload / "npuir-sha.txt").write_text(("a" if mode == "matching" else "b") * 40)
    output = tmp_path / "outputs"
    script = tmp_path / "prepare.sh"
    script.write_text(command, encoding="utf-8", newline="\n")
    result = subprocess.run([bash, "-e", "-o", "pipefail", script.as_posix()], cwd=tmp_path,
                            env={**os.environ, "GITHUB_OUTPUT": output.as_posix()}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    build = "OFF" if mode in {"matching", "cache-hit"} else "ON"
    assert output.read_text().strip() == f"build_npuir={build}"
    if mode == "matching":
        assert (tmp_path / ".npuir-payload/bin/bishengir-compile").read_bytes() == b"fixture compiler"
        assert (cache / "npuir-sha.txt").read_text() == "a" * 40
    else:
        assert not (tmp_path / ".npuir-payload").exists()
        assert not cache.exists()
