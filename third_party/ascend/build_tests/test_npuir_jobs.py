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
    monkeypatch.setattr(module, "_cgroup_limits", lambda: (8, None))
    calls = []
    monkeypatch.setattr(module.subprocess, "check_call", lambda cmd, **kwargs: calls.append(cmd))
    return module, tmp_path, calls


@pytest.mark.parametrize("cpus, memory_gib, expected", [(1, None, 1), (4, None, 4), (64, None, 32), (64, 5, 2),
                                                        (64, 1, 1), (8, 64, 8)])
def test_cgroup_cpu_and_memory_defaults_are_preserved(build, monkeypatch, cpus, memory_gib, expected):
    module, repo, calls = build
    memory = None if memory_gib is None else memory_gib * 1024**3
    monkeypatch.setattr(module, "_cgroup_limits", lambda: (cpus, memory))
    module._build_and_package_bisheng(repo, "/fake/bisheng")
    assert calls[0][calls[0].index("-j") + 1] == str(expected)


@pytest.mark.parametrize("jobs", [1, 3, 64])
def test_max_jobs_overrides_default(build, monkeypatch, jobs):
    module, repo, calls = build
    monkeypatch.setattr(module.os, "cpu_count", lambda: 64)
    monkeypatch.setenv("MAX_JOBS", str(jobs))

    module._build_and_package_bisheng(repo, "/fake/bisheng")

    assert calls[0][calls[0].index("-j") + 1] == str(min(jobs, 32))


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
