import json
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap
import urllib.request

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("mutation", [
    "valid", "valid-patch", "changed-release", "post-release", "bad-sha", "bad-llvm", "bad-cann", "bad-python",
    "schema", "extra", "missing-field", "null", "bad-patches", "bad-patch-path", "bad-checksum", "duplicate-patch",
    "unsorted-patches", "missing", "invalid-json", "multiple", "empty", "trailing-newline"
])
def test_release_workflow_reads_bound_image_inputs(tmp_path, mutation):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None or shutil.which("jq") is None:
        pytest.skip("Bash and jq are required")
    text = (REPO / ".github/workflows/release-docker-images-nightly.yml").read_text(encoding="utf-8")
    step = text.split("      - name: Resolve release image inputs\n", 1)[1].split("\n  build:", 1)[0]
    match = re.search(r"        run: \|\n((?:          .*\n|\n)+)", step)
    assert match is not None, "Release workflow needs a real metadata resolver"
    script = textwrap.dedent(match.group(1))
    metadata = tmp_path / "cmake"
    metadata.mkdir()
    data = {
        "schema_version": 1,
        "triton_version": "3.6.0",
        "source_hash": "a" * 40,
        "llvm_hash": "b" * 40,
        "patches": [],
        "cann": "9.2.0",
        "python_version": "3.11.15",
        "torch": "2.10.0",
        "torch_npu": "2.10.0.post7",
    }
    if mutation == "changed-release":
        data.update(triton_version="4.0.0", source_hash="c" * 40, python_version="3.12.1")
        data.update(torch="2.11.0", torch_npu="2.11.0.post1")
    elif mutation == "bad-sha":
        data["source_hash"] = "main"
    elif mutation == "post-release":
        data["triton_version"] = "3.6.0.post1"
    elif mutation == "bad-cann":
        data["cann"] = "not-a-version"
    elif mutation == "trailing-newline":
        data["cann"] += "\n"
    elif mutation == "bad-llvm":
        data["llvm_hash"] = "main"
    elif mutation == "bad-python":
        data["python_version"] = "3.11"
    elif mutation == "schema":
        data["schema_version"] = True
    elif mutation == "extra":
        data["unknown"] = "value"
    elif mutation == "missing-field":
        del data["torch"]
    elif mutation == "null":
        data["torch_npu"] = None
    elif mutation == "bad-patches":
        data["patches"] = {}
    elif mutation in {"valid-patch", "bad-patch-path", "bad-checksum", "duplicate-patch", "unsorted-patches"}:
        patch = {"file": "llvm_patch_release.patch", "sha256": "c" * 64}
        data["patches"] = [patch]
        if mutation == "bad-patch-path":
            patch["file"] = "../llvm_patch_escape.patch"
        elif mutation == "bad-checksum":
            patch["sha256"] = "invalid"
        elif mutation == "duplicate-patch":
            data["patches"].append(patch.copy())
        elif mutation == "unsorted-patches":
            data["patches"].append({**patch, "file": "llvm_patch_first.patch"})
    if mutation != "missing":
        contents = json.dumps(data)
        if mutation == "invalid-json":
            contents = "{bad"
        elif mutation == "multiple":
            contents += "\n" + contents
        elif mutation == "empty":
            contents = ""
        (metadata / "release-image-info.json").write_text(contents)
    output = tmp_path / "output"
    script_path = tmp_path / "resolve.sh"
    script_path.write_text(script, encoding="utf-8", newline="\n")
    result = subprocess.run([bash, "-e", "-o", "pipefail", script_path.as_posix()], cwd=tmp_path,
                            env={**os.environ, "GITHUB_OUTPUT":
                                 output.as_posix()}, capture_output=True, text=True, timeout=10)
    if mutation in {"valid", "valid-patch", "changed-release", "post-release"}:
        assert result.returncode == 0, result.stderr
        image = json.loads(output.read_text().removeprefix("image="))
        assert image["triton_version"] == data["triton_version"]
        assert image["source_hash"] == data["source_hash"]
        assert image["llvm_hash"] == data["llvm_hash"]
        assert image["python_series"] == ".".join(data["python_version"].split(".")[:2])
        assert image["torch_npu"] == data["torch_npu"]
    else:
        assert result.returncode != 0
        assert not output.exists()


def test_release_recipe_uses_a_bound_manifest_instead_of_an_independent_llvm_argument():
    workflow = (REPO / ".github/workflows/release-docker-images-nightly.yml").read_text(encoding="utf-8")
    path = re.search(r"(?m)^          file: (.+)$", workflow).group(1).strip()
    assert (REPO / path).is_file(), "Nightly must reference an existing Dockerfile"
    recipe = (REPO / path).read_text()
    assert not re.search(r"(?m)^ARG LLVM_COMMIT(?:=|$)", recipe)
    assert "release-image-info.json" in recipe
    assert 'git checkout "${LLVM_COMMIT}"' in recipe
    assert 'git apply "$patch"' in recipe
    assert "TRITON_ASCEND_SOURCE_SHA" in recipe


@pytest.mark.parametrize("mismatch", [False, True])
def test_actual_docker_release_arguments_cannot_drift(tmp_path, monkeypatch, mismatch):
    text = (REPO / "docker/triton_ascend-cann-torch_npu.Dockerfile").read_text()
    scripts = re.findall(r"RUN python3 - <<'PY'\n(.*?)\nPY", text, re.DOTALL)
    assert len(scripts) == 2
    info = json.loads((REPO / "cmake/release-image-info.json").read_text())
    (tmp_path / "release-image-info.json").write_text(json.dumps(info))
    values = {
        "CANN_VERSION": info["cann"],
        "PYTHON_VERSION": info["python_version"],
        "PYTHON_SERIES": ".".join(info["python_version"].split(".")[:2]),
        "PYTORCH_VERSION": info["torch"],
        "TORCH_NPU_VERSION": info["torch_npu"],
        "TRITON_ASCEND_VERSION": info["triton_version"],
        "TRITON_ASCEND_SOURCE_SHA": info["source_hash"],
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    if mismatch:
        monkeypatch.setenv("TRITON_ASCEND_SOURCE_SHA", "b" * 40)
    script = scripts[0].replace("/opt/triton-metadata", tmp_path.as_posix())
    if mismatch:
        with pytest.raises(SystemExit, match="TRITON_ASCEND_SOURCE_SHA"):
            exec(script, {})
    else:
        exec(script, {})


@pytest.mark.parametrize("valid_checksum", [False, True])
def test_actual_docker_patch_download_uses_release_sha_and_checks_contents(tmp_path, monkeypatch, valid_checksum):
    text = (REPO / "docker/triton_ascend-cann-torch_npu.Dockerfile").read_text()
    scripts = re.findall(r"RUN python3 - <<'PY'\n(.*?)\nPY", text, re.DOTALL)
    assert len(scripts) == 2
    contents = b"release patch contents"
    info = {
        "source_hash":
        "a" * 40, "patches": [{
            "file": "llvm_patch_release.patch", "sha256":
            hashlib.sha256(contents).hexdigest() if valid_checksum else "0" * 64
        }]
    }
    (tmp_path / "release-image-info.json").write_text(json.dumps(info))
    urls = []

    def download(url, **kwargs):
        urls.append(url)
        return io.BytesIO(contents)

    monkeypatch.setattr(urllib.request, "urlopen", download)
    script = scripts[1].replace("/opt/triton-metadata", tmp_path.as_posix())
    script = script.replace("/tmp/llvm-patches", (tmp_path / "patches").as_posix())
    if valid_checksum:
        exec(script, {})
        assert (tmp_path / "patches/llvm_patch_release.patch").read_bytes() == contents
    else:
        with pytest.raises(SystemExit, match="checksum mismatch"):
            exec(script, {})
        assert not list((tmp_path / "patches").iterdir())
    assert urls == [
        "https://raw.githubusercontent.com/triton-lang/triton-ascend/" + "a" * 40 +
        "/third_party/ascend/patch/llvm_patch_release.patch"
    ]


@pytest.mark.parametrize("mode", ["create", "fallback", "missing-arm"])
@pytest.mark.parametrize("suffix", ["", "nightly"])
def test_manifest_step_derives_tags_without_an_intermediate_output(tmp_path, mode, suffix):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required")
    workflow = (REPO / ".github/workflows/release-docker-images-nightly.yml").read_text(encoding="utf-8")
    step = workflow.split("      - name: Create multi-arch manifest\n", 1)[1]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    wrapper = r'''
docker() {
  printf '%s\n' "$*" >> "$CALLS"
  case "$*" in
    "buildx imagetools inspect "*"-arm64") [ "$MODE" != missing-arm ] ;;
    "buildx imagetools create "*) [ "$MODE" != fallback ] ;;
  esac
}
'''
    script_path = tmp_path / "manifest.sh"
    script_path.write_text(wrapper + script, encoding="utf-8", newline="\n")
    calls = tmp_path / "calls"
    env = {
        **os.environ, "CALLS": calls.as_posix(), "MODE": mode, "IMAGE_REPOSITORY": "registry.example/triton",
        "TRITON_ASCEND_VERSION": "3.6.0", "CANN_VERSION": "9.2.0", "TORCH_NPU_VERSION": "2.10.0.post7", "PYTHON_SERIES":
        "3.11", "DEVICE_TYPE": "a3", "OS_NAME": "ubuntu24.04", "TAG_SUFFIX": suffix
    }
    env.pop("BASE_TAG", None)
    result = subprocess.run([bash, script_path.as_posix()], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert calls.is_file(), result.stderr
    commands = calls.read_text().splitlines()
    base = "registry.example/triton:3.6.0-cann9.2.0-torch_npu2.10.0.post7-a3-ubuntu24.04-py3.11"
    assert commands[:2] == [f"buildx imagetools inspect {base}-amd64", f"buildx imagetools inspect {base}-arm64"]
    if mode == "missing-arm":
        assert result.returncode != 0
        assert len(commands) == 2, "Do not publish if an architecture is missing"
    else:
        assert result.returncode == 0, result.stderr
        destinations = [base, base + "-" + suffix] if suffix else [base]
        tags = " ".join("-t " + destination for destination in destinations)
        assert commands[2] == f"buildx imagetools create {tags} {base}-amd64 {base}-arm64"
        expected_fallback = []
        if mode == "fallback":
            for destination in destinations:
                expected_fallback += [
                    f"manifest create --amend {destination} {base}-amd64 {base}-arm64", f"manifest push {destination}"
                ]
        assert commands[3:] == expected_fallback
