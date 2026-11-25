import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("version", ["v3.11.3", "v3.12.0", None])
def test_ascend_json_reader_uses_existing_text_pin(tmp_path, monkeypatch, version):
    build = Path(__file__).resolve().parents[1] / "build"
    package = ModuleType("ascend_json_pin_test")
    package.__path__ = [str(build)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.setup_patch", build / "setup_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mod = Mock(CMakeBuild=object, CMakeBuildPy=object, backends=[])
    mod.check_env_flag.return_value = False
    mod.get_base_dir.return_value = str(tmp_path)
    (tmp_path / "cmake").mkdir()
    if version is not None:
        (tmp_path / "cmake/json-version.txt").write_text(version + "\n")
    module.patch_module(mod)
    if version is None:
        with pytest.raises(FileNotFoundError):
            mod.get_json_package_info()
        mod.Package.assert_not_called()
    else:
        mod.get_json_package_info()
        mod.Package.assert_called_once_with(
            "json", "", f"https://github.com/nlohmann/json/releases/download/{version}/include.zip", "JSON_INCLUDE_DIR",
            "", "JSON_SYSPATH")
