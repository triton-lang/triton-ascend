import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


DEFAULT_UTILS_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "utils.py"
)
BACKEND_REGISTER_PATH = (
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
        "repo_backend_register", BACKEND_REGISTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_npu_utils_uses_debug_compilation_flags(monkeypatch):
    backend_register = _load_backend_register_module()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__file__="/torch/__init__.py"))
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(__file__="/torch_npu/__init__.py")
    )
    monkeypatch.setattr(backend_register, "get_torch_cxx_abi", lambda: 1)

    flags = backend_register.get_cc_cmd_npu_utils()

    assert "-g" in flags
    assert "-O0" in flags
    assert "-fno-omit-frame-pointer" in flags


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
    compile_commands = []

    def fake_get_backend_func(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        if name == "get_cc_cmd_npu_utils":
            return ["-DUSE_TORCH_NPU", "-g", "-O0", "-fno-omit-frame-pointer"]
        if name == "get_cc_cmd":
            return ["-ldl"]
        return []

    monkeypatch.setattr(utils, "get_backend_func", fake_get_backend_func)
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: (
            compile_commands.append(args[0]) or SimpleNamespace(returncode=0, stderr="")
        ),
    )

    src_path = tmp_path / "npu_utils.cpp"
    src_path.write_text("int main() { return 0; }\n")

    so_path = utils._build_npu_ext("npu_utils", str(src_path), kernel_launcher="torch")

    assert so_path.endswith(".so")
    assert any(name == "get_cc_cmd_npu_utils" for name, _, _ in calls)
    assert not any(name == "get_cc_cmd" for name, _, _ in calls)
    assert "-g" in compile_commands[0]
    assert "-O0" in compile_commands[0]
    assert "-fno-omit-frame-pointer" in compile_commands[0]


def test_npu_utils_build_uses_special_flags(monkeypatch, tmp_path):
    utils = _load_utils_module()
    _assert_npu_utils_uses_special_flags(utils, monkeypatch, tmp_path)
