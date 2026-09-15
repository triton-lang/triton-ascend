"""Ascend-specific build support, invoked from the in-tree ``setup.py``.

The package is intentionally kept outside the Triton source tree so that the
community ``setup.py`` only needs two small hook calls:

* ``prepare()``  - set default build env vars and fetch optional submodules
* ``activate()`` - intercept ``setuptools.setup()`` to inject Ascend packages,
                   the Ascend LLVM toolchain and custom build commands
"""

from .setup_patch import (
    prepare,
    activate,
    get_triton_ascend_patch_file,
    checkout_file,
)
from .build_npuir import build_npuir

__all__ = [
    "prepare",
    "activate",
    "get_triton_ascend_patch_file",
    "checkout_file",
    "build_npuir",
]
