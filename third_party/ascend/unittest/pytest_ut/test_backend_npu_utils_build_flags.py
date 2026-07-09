import builtins
import importlib.util
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace


DEFAULT_UTILS_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "utils.py"
)
DEFAULT_BACKEND_REGISTER_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "backend_register.py"
)


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


def _load_backend_register_module():
    spec = importlib.util.spec_from_file_location(
        "repo_backend_register", DEFAULT_BACKEND_REGISTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _guard_torch_npu_import(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_npu" or name.startswith("torch_npu."):
            raise AssertionError(f"unexpected import of {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _assert_npu_utils_uses_special_flags(utils, monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "_get_cxx", lambda: "c++")
    monkeypatch.setattr(utils, "_get_ascend_path", lambda: str(tmp_path / "ascend"))
    monkeypatch.setattr(utils.pybind11, "get_include", lambda: "/pybind11")
    monkeypatch.setattr(utils.sysconfig, "get_config_var", lambda name: ".so")
    monkeypatch.setattr(
        utils.sysconfig, "get_default_scheme", lambda: "posix_prefix", raising=False
    )
    monkeypatch.setattr(
        utils.sysconfig, "get_paths", lambda scheme=None: {"include": "/pyinclude"}
    )

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


def test_get_backend_func_detects_torch_npu_without_import(monkeypatch):
    utils = _load_utils_module()
    monkeypatch.delenv("TRITON_BACKEND", raising=False)
    monkeypatch.setattr(utils, "backend_policy", None)
    _guard_torch_npu_import(monkeypatch)

    monkeypatch.setattr(
        utils.importlib.util,
        "find_spec",
        lambda name: object() if name == "torch_npu" else None,
    )

    calls = []

    def fake_execute_func(category, method, *args, **kwargs):
        calls.append((category, method, args, kwargs))
        return "ok"

    monkeypatch.setattr(
        utils.backend_strategy_registry,
        "execute_func",
        fake_execute_func,
    )

    assert utils.get_backend_func("version_hash") == "ok"
    assert calls == [("torch_npu", "version_hash", (), {})]
    assert "torch_npu" not in sys.modules


def test_version_hash_reads_torch_npu_metadata_without_import(monkeypatch, tmp_path):
    backend_register = _load_backend_register_module()
    _guard_torch_npu_import(monkeypatch)

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.8.0"
    fake_torch.version = SimpleNamespace(git_version="torch-git-version")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    torch_npu_pkg = tmp_path / "torch_npu"
    torch_npu_pkg.mkdir()
    (torch_npu_pkg / "version.py").write_text(
        "__version__ = '2.8.0.post2'\n"
        "git_version = 'torch-npu-git-version'\n"
    )
    monkeypatch.setattr(
        backend_register,
        "_get_package_dir",
        lambda package_name: str(torch_npu_pkg),
    )

    version_hash = backend_register.backend_strategy_registry.execute_func(
        "torch_npu", "version_hash"
    )

    assert version_hash == ["torch-git-version", "torch-npu-git-version"]
    assert "torch_npu" not in sys.modules


def test_get_cc_cmd_npu_utils_resolves_paths_without_import(monkeypatch, tmp_path):
    backend_register = _load_backend_register_module()
    _guard_torch_npu_import(monkeypatch)

    torch_pkg = tmp_path / "torch"
    torch_pkg.mkdir()
    fake_torch = types.ModuleType("torch")
    fake_torch.__file__ = str(torch_pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    torch_npu_pkg = tmp_path / "torch_npu"
    torch_npu_pkg.mkdir()
    monkeypatch.setattr(
        backend_register,
        "_get_package_dir",
        lambda package_name: str(torch_npu_pkg),
    )
    monkeypatch.setattr(backend_register, "get_torch_cxx_abi", lambda: 1)

    cc_cmd = backend_register.backend_strategy_registry.execute_func(
        "torch_npu", "get_cc_cmd_npu_utils"
    )

    assert f"-I{torch_pkg / 'include'}" in cc_cmd
    assert f"-I{torch_npu_pkg / 'include'}" in cc_cmd
    assert f"-L{torch_npu_pkg / 'lib'}" in cc_cmd
    assert "-ltorch_npu" in cc_cmd
    assert "torch_npu" not in sys.modules
