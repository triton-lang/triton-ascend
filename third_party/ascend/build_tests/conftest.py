import ast
import importlib.util
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def ascend_setup(monkeypatch):
    build = REPO / "third_party/ascend/build"
    package = ModuleType("ascend_setup_review_test")
    package.__path__ = [str(build)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.setup_patch", build / "setup_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_globals(monkeypatch, tmp_path):
    # Execute the actual upstream package definitions without running setup(),
    # fetching submodules or applying patches to the developer's checkout.
    names = {"Package", "check_env_flag", "is_linux_os", "get_json_package_info", "get_llvm_package_info"}
    tree = ast.parse((REPO / "setup.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ModuleType("upstream_setup_review_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(os=os, platform=platform, dataclass=dataclass, Optional=Optional,
                           get_base_dir=lambda: str(tmp_path))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "setup.py", "exec"), module.__dict__)
    module.BackendInstaller = SimpleNamespace(prepare=lambda name: name)
    module.backends = []
    module.CMakeBuild = type("CMakeBuild", (), {})
    module.CMakeBuildPy = type("CMakeBuildPy", (), {})
    module.get_package_dirs = lambda: iter(())
    module.get_packages = lambda: iter(())
    module.add_links = lambda external_only: None
    return module


@pytest.fixture
def checker(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("bom_checker_test",
                                                  REPO / "third_party/ascend/build/check_bom_consistency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for relative in ["requirements.txt", "docker/Dockerfile", "docs/en/community/release_policy.md"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    shutil.copytree(REPO / "cmake", tmp_path / "cmake")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "DEPS_DIR", tmp_path / "cmake")
    return module
