import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def build(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "build" / "build_npuir.py"
    spec = importlib.util.spec_from_file_location("build_npuir_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    script = tmp_path / "build-tools" / "build.sh"
    script.parent.mkdir()
    script.touch()
    monkeypatch.delenv("MAX_JOBS", raising=False)
    calls = []
    monkeypatch.setattr(module.subprocess, "check_call", lambda cmd, **kwargs: calls.append(cmd))
    return module, tmp_path, calls


@pytest.mark.parametrize("cpus, expected", [(None, 1), (0, 1), (1, 1), (4, 1), (7, 1), (8, 1), (64, 8), (512, 32)])
def test_default_jobs_are_positive_and_bounded(build, monkeypatch, cpus, expected):
    module, repo, calls = build
    monkeypatch.setattr(module.os, "cpu_count", lambda: cpus)

    module._build_and_package_bisheng(repo, "/fake/bisheng")

    assert calls[0][calls[0].index("-j") + 1] == str(expected)


@pytest.mark.parametrize("jobs", [1, 3, 64])
def test_max_jobs_overrides_default(build, monkeypatch, jobs):
    module, repo, calls = build
    monkeypatch.setattr(module.os, "cpu_count", lambda: 64)
    monkeypatch.setenv("MAX_JOBS", str(jobs))

    module._build_and_package_bisheng(repo, "/fake/bisheng")

    assert calls[0][calls[0].index("-j") + 1] == str(jobs)


@pytest.mark.parametrize("jobs", ["0", "-1", "", "invalid", "1.5"])
def test_invalid_job_override_fails_before_cleanup(build, monkeypatch, jobs):
    module, repo, calls = build
    cache = repo / "build" / "CMakeCache.txt"
    cache.parent.mkdir()
    cache.write_text("existing configuration")
    monkeypatch.setenv("MAX_JOBS", jobs)

    with pytest.raises(ValueError, match="MAX_JOBS must be a positive integer"):
        module._build_and_package_bisheng(repo, "/fake/bisheng")

    assert calls == []
    assert cache.read_text() == "existing configuration"
