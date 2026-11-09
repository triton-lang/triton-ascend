from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[3]


def test_build_changes_preserve_the_actual_ascend_source_patch(tmp_path):
    version = (REPO / "version.txt").read_text().strip().split(".post", 1)[0]
    patch = REPO / f"third_party/ascend/patch/triton-ascend-{version}.patch"
    for line in patch.read_text(encoding="utf-8").splitlines():
        if not line.startswith("diff --git "):
            continue
        relative = shlex.split(line)[2].removeprefix("a/")
        source = (REPO / relative).resolve()
        target = (tmp_path / relative).resolve()
        assert source.is_relative_to(REPO) and target.is_relative_to(tmp_path)
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    local_patch = tmp_path / "ascend.patch"
    local_patch.write_bytes(patch.read_bytes().replace(b"\r\n", b"\n"))
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "apply", str(local_patch)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cmake = (tmp_path / "CMakeLists.txt").read_text()
    assert "safe_compile.cmake" in cmake
