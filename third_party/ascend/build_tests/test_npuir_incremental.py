import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def npuir(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "build" / "build_npuir.py"
    spec = importlib.util.spec_from_file_location("build_npuir_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.os, "cpu_count", lambda: 64)
    return module


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("existing_build", [False, True])
def test_build_directory_respects_rebuild(npuir, monkeypatch, tmp_path, rebuild, existing_build):
    script = tmp_path / "build-tools" / "build.sh"
    script.parent.mkdir()
    script.touch()
    marker = tmp_path / "build" / "CMakeCache.txt"
    if existing_build:
        marker.parent.mkdir()
        marker.write_text("existing configuration")
    calls = []
    monkeypatch.setattr(npuir.subprocess, "check_call", lambda cmd, **kwargs: calls.append(cmd))

    npuir._build_and_package_bisheng(tmp_path, "/fake/bisheng", rebuild=rebuild)

    assert marker.parent.is_dir()
    assert marker.exists() == (existing_build and not rebuild)
    if marker.exists():
        assert marker.read_text() == "existing configuration"
    assert ("-r" in calls[0]) == rebuild


@pytest.mark.parametrize("setting, expected", [(None, True), ("ON", True), ("OFF", False)])
def test_install_rebuild_setting(npuir, monkeypatch, tmp_path, setting, expected):
    monkeypatch.delenv("TRITON_NPUIR_REBUILD", raising=False)
    if setting is not None:
        monkeypatch.setenv("TRITON_NPUIR_REBUILD", setting)
    monkeypatch.setattr(npuir, "_check_disk_space", lambda: None)
    monkeypatch.setattr(npuir, "_get_ascend_path", lambda: tmp_path)
    monkeypatch.setattr(npuir, "_init_npuir_repo", lambda: None)
    monkeypatch.setattr(npuir, "_copy_artifacts", lambda: None)
    calls = []
    monkeypatch.setattr(npuir, "_build_and_package_bisheng", lambda *args, **kwargs: calls.append(kwargs))

    npuir.build_npuir()

    assert calls[0]["rebuild"] is expected
