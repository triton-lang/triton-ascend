# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Ascend-specific knobs exposed as ``triton.knobs.ascend``.

The group is attached to the upstream ``triton.knobs`` module at runtime, so
the vendored ``python/triton/knobs.py`` stays untouched.  Every field is a
regular ``env_base`` descriptor:

* reading a field reads the same environment variable as before;
* assigning a field writes the environment variable (unless
  ``triton.knobs.propagate_env`` is ``False``);
* ``scope()`` / ``reset()`` / ``copy()`` and ``_fresh_knobs_impl`` work exactly
  as they do for the upstream groups.

``ASCEND_HOME_PATH`` and ``NPU_DEVICE_LIMIT`` intentionally stay environment
variables: the first is the CANN toolchain location (must be set before
import), the second is a device core cap read once per process.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Dict, Iterable

import triton.knobs as _knobs


class ascend_knobs(_knobs.base_knobs):
    """Knobs for the Ascend backend.

    Field names are Python-friendly; every field keeps its original
    environment variable name so existing scripts keep working.
    """

    # Compilation-affecting switches.
    enable_sanitizer: bool = _knobs.env_bool("TRITON_ENABLE_SANITIZER", False)
    print_ub_bits: bool = _knobs.env_bool("ENABLE_PRINT_UB_BITS", False)
    memory_display: bool = _knobs.env_bool("TRITON_MEMORY_DISPLAY", False)
    npu_compiler_path: str = _knobs.env_opt_str("TRITON_NPU_COMPILER_PATH")

    # Autotuning switches.
    enable_ubtuner: str = _knobs.env_str("TRITON_ENABLE_UBTUNER", "")
    autotune_parallel_compile: bool = _knobs.env_bool("TRITON_AUTOTUNE_PARALLEL_COMPILE", True)
    bench_method: str = _knobs.env_str("TRITON_BENCH_METHOD", "default")

    # Launcher / runtime switches.
    compile_only: bool = _knobs.env_bool("TRITON_COMPILE_ONLY", False)
    enable_taskqueue: bool = _knobs.env_bool("TRITON_ENABLE_TASKQUEUE", True)
    device_print: bool = _knobs.env_bool("TRITON_DEVICE_PRINT", False)
    grid_warn_print: bool = _knobs.env_bool("TRITON_GRID_WARN_PRINT", False)


_ASCEND_KNOBS = ascend_knobs()

# Environment variables that change the compiled artifact.  They are not in
# the C++ CACHE_INVALIDATING_ENV_VARS whitelist, so a switched knob would
# otherwise reuse artifacts produced under the previous value.
_COMPILE_ENV_VARS = (
    "TRITON_ENABLE_SANITIZER",
    "ENABLE_PRINT_UB_BITS",
    "TRITON_MEMORY_DISPLAY",
    "TRITON_NPU_COMPILER_PATH",
    "NPU_DEVICE_LIMIT",
    "TRITON_ENABLE_UBTUNER",
)

# Environment variables that change which autotune result is selected.
_AUTOTUNE_ENV_VARS = (
    "TRITON_ENABLE_UBTUNER",
    "TRITON_BENCH_METHOD",
)

# Modules that imported ``get_cache_invalidating_env_vars`` by value.  Only
# modules already imported are patched, so ``install_ascend_knobs`` runs from
# ``triton.backends.ascend`` and again from ``triton.backends.ascend.runtime``:
# the latter imports ``runtime.autotuner``, which is not in ``sys.modules`` yet
# when the package-level install runs.
_CACHE_KEY_MODULES: Dict[str, Iterable[str]] = {
    "triton.runtime.jit": _COMPILE_ENV_VARS,
    "triton.compiler.compiler": _COMPILE_ENV_VARS,
    "triton.runtime.autotuner": _COMPILE_ENV_VARS + _AUTOTUNE_ENV_VARS,
    "triton.backends.ascend.runtime.autotuner": _COMPILE_ENV_VARS + _AUTOTUNE_ENV_VARS,
}


def _extend_cache_invalidating_env_vars(module: ModuleType, keys: Iterable[str]) -> None:
    original = getattr(module, "get_cache_invalidating_env_vars", None)
    if original is None or getattr(original, "_ascend_extended", False):
        return

    def _patched(original=original, keys=tuple(keys)):
        env_vars = original()
        fields = {knob.key: name for name, knob in _knobs.ascend.knob_descriptors.items()}
        for key in keys:
            env_vars.pop(key, None)
            field = fields.get(key)
            if field is None:
                value = os.environ.get(key)
                if value is not None:
                    env_vars[key] = value
                continue
            knob = _knobs.ascend.knob_descriptors[field]
            current = getattr(_knobs.ascend, field)
            if current == getattr(knob, "default", None):
                continue
            token = _knobs.toenv(current)
            if token is None or token[0] is None:
                continue
            env_vars[key] = token[0]
        return env_vars

    _patched._ascend_extended = True
    module.get_cache_invalidating_env_vars = _patched


def _patch_ascend_cache_keys() -> None:
    for module_name, keys in _CACHE_KEY_MODULES.items():
        module = sys.modules.get(module_name)
        if module is not None:
            _extend_cache_invalidating_env_vars(module, keys)


def install_ascend_knobs() -> None:
    """Attach ``knobs.ascend`` and extend Ascend-aware cache keys.

    Idempotent: repeated calls keep the same group object so references held
    by callers stay valid, and cache wrappers are installed at most once.
    """
    if "ascend" not in vars(_knobs):
        _knobs.ascend = _ASCEND_KNOBS
    _patch_ascend_cache_keys()
