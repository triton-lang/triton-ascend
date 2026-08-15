import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.backend("none")

DEFAULT_UTILS_PATH = (Path(__file__).resolve().parents[2] / "backend" / "utils.py")


def _get_utils_path():
    override = os.environ.get("TRITON_ASCEND_UTILS_UNDER_TEST")
    if override:
        return Path(override)
    return DEFAULT_UTILS_PATH


def _load_utils_module():
    utils_path = _get_utils_path()
    spec = importlib.util.spec_from_file_location("repo_backend_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_npu_utils_uses_special_flags(utils, monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "_get_cxx", lambda: "c++")
    monkeypatch.setattr(utils, "_get_ascend_path", lambda: str(tmp_path / "ascend"))
    monkeypatch.setattr(utils.pybind11, "get_include", lambda: "/pybind11")
    monkeypatch.setattr(utils.sysconfig, "get_config_var", lambda name: ".so")
    monkeypatch.setattr(utils.sysconfig, "get_default_scheme", lambda: "posix_prefix", raising=False)
    monkeypatch.setattr(utils.sysconfig, "get_paths", lambda scheme=None: {"include": "/pyinclude"})

    calls = []

    def fake_get_backend_func(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        if name == "get_cc_cmd_npu_utils":
            return ["-DUSE_TORCH_NPU"]
        if name == "get_cc_cmd":
            return ["-ldl"]
        return []

    monkeypatch.setattr(utils, "get_backend_func", fake_get_backend_func)
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    src_path = tmp_path / "npu_utils.cpp"
    src_path.write_text("int main() { return 0; }\n")

    so_path = utils._build_npu_ext("npu_utils", str(src_path), kernel_launcher="torch")

    assert so_path.endswith(".so")
    assert any(name == "get_cc_cmd_npu_utils" for name, _, _ in calls)
    assert not any(name == "get_cc_cmd" for name, _, _ in calls)


def test_npu_utils_build_uses_special_flags(monkeypatch, tmp_path):
    utils = _load_utils_module()
    _assert_npu_utils_uses_special_flags(utils, monkeypatch, tmp_path)


@pytest.mark.parametrize(
    ("arch", "raw_ub_kib", "graph_budget_bytes"),
    (
        ("Ascend910B1", 192, 96 * 1024),
        ("Ascend910_9581", 256, 128 * 1024),
        ("Ascend950A3", 256, 128 * 1024),
        ("", 0, 0),
        ("unknown-arch", 0, 0),
        (None, 0, 0),
    ),
)
def test_graph_ub_budget_resolves_from_explicit_arch(arch, raw_ub_kib, graph_budget_bytes):
    """The compiler-side resolver must not depend on the active NPU device."""
    utils = _load_utils_module()

    assert utils.ub_size_in_kbytes_for_arch(arch) == raw_ub_kib
    assert utils.graph_ub_budget_bytes_for_arch(arch) == graph_budget_bytes
