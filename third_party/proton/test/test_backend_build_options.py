from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def configure(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is required")
    policy = Path(__file__).resolve().parents[1] / "cmake" / "ProtonBackends.cmake"

    def run(*options):
        return subprocess.run([cmake, *options, "-P", str(policy)], cwd=tmp_path, text=True, capture_output=True)

    return run


def test_missing_gpu_dependencies_fail_by_default(configure):
    result = configure()
    assert result.returncode != 0
    assert "PROTON_BUILD_CUDA is enabled" in result.stderr


def test_gpu_dependencies_preserve_default_backends(configure, tmp_path):
    for header in ("cupti.h", "cuda.h", "roctracer/roctracer.h", "hip/hip_runtime_api.h"):
        path = tmp_path / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    result = configure(f"-DCUPTI_INCLUDE_DIR={tmp_path}", f"-DROCTRACER_INCLUDE_DIR={tmp_path}")
    assert result.returncode == 0, result.stderr
    assert "CUDA=ON; HIP=ON; MSPTI=OFF; CANN=OFF" in result.stdout


def test_explicit_ascend_only_build(configure, tmp_path):
    (tmp_path / "acl").mkdir()
    (tmp_path / "acl" / "acl.h").touch()
    (tmp_path / "mspti.h").touch()
    result = configure("-DPROTON_BUILD_CUDA=OFF", "-DPROTON_BUILD_HIP=OFF", "-DPROTON_BUILD_CANN=ON",
                       "-DPROTON_BUILD_MSPTI=ON", f"-DCANN_INCLUDE_DIR={tmp_path}", f"-DMSPTI_INCLUDE_DIR={tmp_path}")
    assert result.returncode == 0, result.stderr
    assert "CUDA=OFF; HIP=OFF; MSPTI=ON; CANN=ON" in result.stdout


def test_requested_mspti_cannot_be_silently_disabled(configure):
    result = configure("-DPROTON_BUILD_CUDA=OFF", "-DPROTON_BUILD_HIP=OFF", "-DPROTON_BUILD_MSPTI=ON")
    assert result.returncode != 0
    assert "PROTON_BUILD_MSPTI requires PROTON_BUILD_CANN" in result.stderr
