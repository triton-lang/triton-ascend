# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Lazy import hooks so old import paths redirect to new locations
without modifying existing code.

  import triton.extension.buffer.language as bl
      -> triton.language.extra.extension.buffer.language

  from triton.runtime.libentry import libentry
      -> triton.backends.ascend.runtime.libentry
"""

import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

# (old_name, new_name, is_package)
_REDIRECTS = [
    ("triton.extension.buffer.language", "triton.language.extra.extension.buffer.language", True),
    ("triton.runtime.libentry", "triton.backends.ascend.runtime.libentry", False),
]

# Synthetic parent packages (deleted from python/triton/extension/).
_SYNTHETIC = {"triton.extension", "triton.extension.buffer"}


def _make_synth(name):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


class _Loader(Loader):

    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        import importlib
        return importlib.import_module(self._target)

    def exec_module(self, module):
        pass


class _Finder(MetaPathFinder):
    _map = {old: (new, pkg) for old, new, pkg in _REDIRECTS}

    def find_spec(self, fullname, path, target=None):
        if fullname in _SYNTHETIC:
            _make_synth(fullname)
            return ModuleSpec(fullname, None, is_package=True)
        pair = self._map.get(fullname)
        if pair is None:
            return None
        return ModuleSpec(fullname, _Loader(pair[0]), is_package=pair[1])


def install():
    """Register the backcompat import hook (idempotent)."""
    for f in sys.meta_path:
        if isinstance(f, _Finder):
            return
    for pkg in _SYNTHETIC:
        _make_synth(pkg)
    sys.meta_path.insert(0, _Finder())
