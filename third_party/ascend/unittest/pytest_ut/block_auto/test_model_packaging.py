"""Exercise the real Ascend setup hook and Python-only installation payload.

The native build and backend initialization are deliberately outside this test:
it validates package discovery, setuptools data copying, and standalone model
loading without requiring an extension build, a source tools/ tree, or a device.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
from types import ModuleType

import pytest


def test_real_setup_hook_installs_complete_model_payload(tmp_path, monkeypatch):
    setuptools = pytest.importorskip("setuptools")
    pytest.importorskip("pybind11")
    from setuptools.command.build_py import build_py

    root = Path(__file__).resolve().parents[5]
    backend = root / "third_party/ascend/backend"
    setup_sha256 = hashlib.sha256((root / "setup.py").read_bytes()).hexdigest()
    captured, commands = {}, []

    def capture_setup(**kwargs):
        captured.update(kwargs)
        return None

    original_run = subprocess.run

    def guarded_run(command, *args, **kwargs):
        words = list(command) if not isinstance(command, str) else command.split()
        commands.append(words)
        if words[:3] == ["git", "submodule", "update"]:
            # BackendInstaller treats an unavailable submodule command as an
            # optional preparation failure. Keep that path offline and read-only.
            raise subprocess.CalledProcessError(1, words)
        if words[:2] == ["git", "rev-parse"] or words[:2] == ["git", "branch"]:
            return original_run(command, *args, **kwargs)
        raise AssertionError("setup metadata attempted an external command: " + repr(words))

    def no_network(*args, **kwargs):
        raise AssertionError("package verification attempted network access")

    with monkeypatch.context() as patch:
        for name, value in {
                "TRITON_BUILD_NPUIR": "OFF", "TRITON_BUILD_TD": "OFF", "TRITON_BUILD_PROTON": "OFF",
                "TRITON_OFFLINE_BUILD": "1", "IS_MANYLINUX": "TRUE"
        }.items():
            patch.setenv(name, value)
        patch.delenv("TRITON_ASCEND_BISHENGIR_PATH", raising=False)
        patch.delenv("TRITON_PLUGIN_DIRS", raising=False)
        patch.setattr(setuptools, "setup", capture_setup)
        patch.setattr(subprocess, "run", guarded_run)
        patch.setattr(socket.socket, "connect", no_network)
        patch.chdir(root)
        original_path = list(sys.path)
        original_environment = dict(os.environ)
        try:
            runpy.run_path(str(root / "setup.py"), run_name="__main__")
        finally:
            sys.path[:] = original_path
            for variable in set(os.environ) - set(original_environment):
                del os.environ[variable]
            os.environ.update(original_environment)

    package = "triton.backends.ascend.block_auto_models"
    assert package in captured["packages"]
    assert captured["cmdclass"]["build_py"].__name__ == "AscendBuildPy"
    patterns = captured["package_data"][package]
    assert {"data/*", "vv_skill_data/*", "vv_library_data/*", "vv_mixed_data/*"}.issubset(patterns)

    # Use the standard setuptools Python copier with the *actual captured*
    # Ascend package mapping/data rules, avoiding CMakeBuildPy.build_ext.
    distribution = setuptools.Distribution({
        "name": "ascend-model-installation-check",
        "version": "0",
        "packages": ["triton.backends.ascend", package],
        "package_dir": captured["package_dir"],
        "package_data": captured["package_data"],
        "include_package_data": True,
    })
    distribution.script_name = str(root / "setup.py")
    command = build_py(distribution)
    command.build_lib = str(tmp_path / "installed")
    command.compile = False
    command.ensure_finalized()
    egg_info = tmp_path / "metadata.egg-info"
    egg_info.mkdir()
    (egg_info / "SOURCES.txt").write_text("")
    command.existing_egg_info_dir = str(egg_info)
    with monkeypatch.context() as patch:
        patch.chdir(root)
        command.run()
    installed = Path(command.build_lib) / "triton/backends/ascend"
    assert not (Path(command.build_lib) / "tools").exists()
    expected = {
        path.relative_to(backend)
        for path in (backend / "block_auto_models").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    expected.update(Path(name) for name in ("block_auto_model.py", "block_auto_cv.py", "block_auto_artifacts.py"))
    hashes = {}
    for relative in sorted(expected):
        destination = installed / relative
        assert destination.is_file(), relative
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        assert actual == hashlib.sha256((backend / relative).read_bytes()).hexdigest(), relative
        hashes[relative.as_posix()] = actual

    # Loading under an isolated backend namespace bypasses only the backend's
    # unrelated native-extension initializer, never a model import/data lookup.
    # Sibling selector tests may have imported CLI runners during collection.
    # Remove them temporarily so the installed payload must load independently.
    for module_name in tuple(sys.modules):
        if module_name.startswith(("run_vv_", "run_cv_")):
            monkeypatch.delitem(sys.modules, module_name)
    name = "_installed_ascend_block_auto"
    namespace = ModuleType(name)
    namespace.__path__ = [str(installed)]
    monkeypatch.setitem(sys.modules, name, namespace)
    model = importlib.import_module(name + ".block_auto_model")
    assert Path(model.__file__).is_relative_to(installed)
    table = model._cost_table()
    assert table
    model._instruction_model()
    resources = installed / "block_auto_models"
    library = json.loads((resources / "vv_library_data/provenance.json").read_text())
    assert hashlib.sha256(
        (resources / "vv_library_data/pow_fp32.ll").read_bytes()).hexdigest() == library["llvm_sha256"]
    mixed = json.loads((resources / "vv_mixed_data/provenance.json").read_text())
    for filename, expected_hash in mixed["asset_sha256"].items():
        assert hashlib.sha256((resources / "vv_mixed_data" / filename).read_bytes()).hexdigest() == expected_hash
    for relative in expected:
        if relative.suffix == ".py" and relative.parts[0] == "block_auto_models" and len(relative.parts) == 2:
            importlib.import_module(name + ".block_auto_models." + relative.stem)
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith(name + "."):
            assert Path(module.__file__).is_relative_to(installed), module_name
    assert all(not module_name.startswith(("run_vv_", "run_cv_")) for module_name in sys.modules)
    assert hashlib.sha256((root / "setup.py").read_bytes()).hexdigest() == setup_sha256
    report = dict(status="PASS", setup_hook="prepare + activate + patch_module + _build_setup_kwargs",
                  source_setup_modified=False, native_build_executed=False, device_executed=False,
                  model_package=package, model_files=len(hashes), cost_entries=len(table), hashes=hashes,
                  setup_commands=commands, installed=str(installed))
    (tmp_path / "packaging-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "hashes"}, sort_keys=True))
