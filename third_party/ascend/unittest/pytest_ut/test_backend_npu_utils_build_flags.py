import builtins
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


DEFAULT_UTILS_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "utils.py"
)
DEFAULT_REGISTER_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "backend_register.py"
)
DEFAULT_DRIVER_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "driver.py"
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


def _load_register_module():
    spec = importlib.util.spec_from_file_location(
        "repo_backend_register", DEFAULT_REGISTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_driver_module():
    spec = importlib.util.spec_from_file_location(
        "repo_backend_driver", DEFAULT_DRIVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _guard_torch_npu_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
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

    assert utils.get_backend_func("get_cc_cmd_npu_utils") == "ok"
    assert calls == [("torch_npu", "get_cc_cmd_npu_utils", (), {})]
    assert "torch_npu" not in sys.modules


def test_get_cc_cmd_npu_utils_resolves_torch_npu_path_without_import(
    monkeypatch, tmp_path
):
    register = _load_register_module()
    _guard_torch_npu_import(monkeypatch)
    monkeypatch.setattr(register, "get_torch_cxx_abi", lambda: 1)

    torch_path = tmp_path / "torch_pkg"
    torch_npu_path = tmp_path / "torch_npu_pkg"

    def fake_find_spec(name):
        if name == "torch":
            return SimpleNamespace(
                origin=str(torch_path / "__init__.py"),
                submodule_search_locations=[str(torch_path)],
            )
        if name == "torch_npu":
            return SimpleNamespace(
                origin=str(torch_npu_path / "__init__.py"),
                submodule_search_locations=[str(torch_npu_path)],
            )
        return None

    monkeypatch.setattr(register.importlib.util, "find_spec", fake_find_spec)

    cc_cmd = register.get_cc_cmd_npu_utils()

    assert f"-I{torch_path / 'include'}" in cc_cmd
    assert f"-I{torch_npu_path / 'include'}" in cc_cmd
    assert f"-L{torch_npu_path / 'lib'}" in cc_cmd
    assert "torch_npu" not in sys.modules


def test_npu_utils_build_rechecks_cache_after_lock(monkeypatch, tmp_path):
    driver = _load_driver_module()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_so = cache_dir / "npu_utils.so"
    cached_so.write_bytes(b"cached")

    class FakeCache:
        lock_path = str(cache_dir / "lock")

        def __init__(self):
            self.get_file_calls = 0

        def get_file(self, filename):
            self.get_file_calls += 1
            if self.get_file_calls == 1:
                return None
            return str(cached_so)

        def put(self, data, filename, binary=True):
            raise AssertionError("unexpected cache put")

    fake_cache = FakeCache()
    monkeypatch.setattr(driver, "get_cache_manager", lambda key: fake_cache)
    monkeypatch.setattr(
        driver,
        "_build_npu_ext",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected npu_utils build")
        ),
    )

    assert driver.NPUUtils()._build_or_get_cached_so() == str(cached_so)
    assert fake_cache.get_file_calls == 2
