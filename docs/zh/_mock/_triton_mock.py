# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Inject mock stubs for triton C extensions so Sphinx can import triton
source without compiling the native extension (e.g. in CI/doc builds
where CANN is unavailable).

Usage – call ``install()`` before any ``import triton`` statement:

    from docs.zh._mock._triton_mock import install as _install_mock
    _install_mock()
    import triton  # works without compiled _C extensions
"""
import importlib.machinery
import os
import sys
import types
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Minimal stubs for triton.backends.compiler / driver classes
# (keeping them as real Python classes avoids isinstance issues in autodoc)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GPUTarget:
    backend: str
    arch: Union[int, str]
    warp_size: int


class _Language(Enum):
    TRITON = 0
    GLUON = 1


class _BaseBackend(metaclass=ABCMeta):

    def __init__(self, target) -> None:
        self.target = target

    @staticmethod
    @abstractmethod
    def supports_target(target):
        raise NotImplementedError


class _DriverBase(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def is_active(cls):
        raise NotImplementedError

    @abstractmethod
    def get_current_target(self):
        raise NotImplementedError

    @abstractmethod
    def get_active_torch_device(self):
        raise NotImplementedError

    @abstractmethod
    def map_python_to_cpp_type(self, ty: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_benchmarker(self):
        raise NotImplementedError


def _make_module(name: str, parent=None) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []
    # A real (non-None) spec keeps importlib.util.find_spec(name) working for
    # third-party code probing optional triton submodules.
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = mod
    if parent is not None:
        setattr(parent, name.rsplit(".", 1)[-1], mod)
    return mod


def install() -> None:
    """Populate sys.modules with lightweight stubs for all C extensions."""
    if getattr(sys.modules.get("triton._C"), "__triton_doc_mock__", False):
        return  # our stubs are already installed

    # A failed real `import triton` leaves partially-initialized triton
    # modules behind (e.g. a genuine triton._C without CANN backends); purge
    # them so the stubs and the re-import start from a clean slate.
    for _name in [n for n in sys.modules if n == "triton" or n.startswith("triton.")]:
        del sys.modules[_name]

    # ------------------------------------------------------------------ #
    # triton._C.libtriton  (must come first – imported at triton load)    #
    # ------------------------------------------------------------------ #
    _c = _make_module("triton._C")
    _c.__triton_doc_mock__ = True

    libtriton = _make_module("triton._C.libtriton", parent=_c)
    libtriton.getenv = lambda key, default="": os.environ.get(key, default)
    libtriton.getenv_bool = (lambda key, default=False: os.environ.get(key, "1" if default else "0").lower() in
                             ("1", "true", "yes"))
    libtriton.get_cache_invalidating_env_vars = lambda: []
    libtriton.ir = MagicMock(name="triton._C.libtriton.ir")
    libtriton.buffer_ir = MagicMock(name="triton._C.libtriton.buffer_ir")

    # triton._C.libtriton.ascend – use a proper enum stub so that
    # isinstance(v, ascend_ir.AddressSpace) works in cann/extension/core.py
    import importlib.util as _ilu
    _stub_spec = _ilu.spec_from_file_location(
        "_ascend_ir_stub",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ascend_ir_stub.py"),
    )
    _ascend_ir_stub = _ilu.module_from_spec(_stub_spec)
    _stub_spec.loader.exec_module(_ascend_ir_stub)

    ascend_ext = _make_module("triton._C.libtriton.ascend", parent=libtriton)
    ascend_ext.ir = _ascend_ir_stub

    # ------------------------------------------------------------------ #
    # triton.backends – stub out the whole package so that               #
    # _discover_backends() is never called (it would try to load gpu      #
    # backends registered by a system-wide triton install).               #
    # ------------------------------------------------------------------ #
    _bk_compiler_mod = _make_module("triton.backends.compiler")
    _bk_compiler_mod.GPUTarget = _GPUTarget
    _bk_compiler_mod.Language = _Language
    _bk_compiler_mod.BaseBackend = _BaseBackend

    _bk_driver_mod = _make_module("triton.backends.driver")
    _bk_driver_mod.DriverBase = _DriverBase

    _bk = _make_module("triton.backends")
    _bk.backends = {}
    _bk.DriverBase = _DriverBase
    _bk.compiler = _bk_compiler_mod
    _bk.driver = _bk_driver_mod

    # ------------------------------------------------------------------ #
    # triton.backends.ascend  (CANN backend plugin)                       #
    # ------------------------------------------------------------------ #
    for _name in [
            "triton.backends.ascend",
            "triton.backends.ascend.driver",
            "triton.backends.ascend.utils",
            "triton.backends.ascend.compiler",
            "triton.backends.ascend.backend_register",
            "triton.backends.ascend.cpu_driver",
    ]:
        sys.modules[_name] = MagicMock(name=_name)

    # ------------------------------------------------------------------ #
    # Optional runtime deps                                               #
    # ------------------------------------------------------------------ #
    try:
        import pybind11  # noqa: F401
    except ImportError:
        sys.modules["pybind11"] = MagicMock(name="pybind11")

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = MagicMock(name="torch")
        sys.modules["torch.nn"] = MagicMock(name="torch.nn")
        sys.modules["torch._C"] = MagicMock(name="torch._C")
