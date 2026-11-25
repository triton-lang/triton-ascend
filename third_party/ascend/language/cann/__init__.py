# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from triton.language import math

from . import libdevice
from . import extension

extension.parallel = extension.aux_ops.parallel
math.tanh = libdevice.tanh

# Auto-patch ``triton.autotune`` (and ``triton.runtime.autotune``) to the Ascend
# ``AutoTilingTuner`` the first time it is *called*, without importing the
# runtime -- and therefore torch -- during ``import triton``.
#
# The lazy-initialization guard in test_import_time.py requires that
# ``import triton`` leaves {acl, torch, torch_npu, mindspore} out of
# sys.modules. The Ascend runtime's autotuner package imports torch at module
# load (runtime/autotuner.py, runtime/utils.py, ...), so importing the runtime
# here would eagerly pull torch into ``import triton`` and break that invariant.
# Instead we install a thin proxy: the first call to ``triton.autotune(...)``
# imports the runtime (whose ``_patch_autotune`` rebinds both names to the real
# Ascend autotune) and then delegates to it.
#
# This module is imported by ``triton.language.extra`` during ``import triton``
# (triton/__init__.py -> ``from . import language``), by which point
# ``triton.autotune`` has already been bound from ``triton.runtime`` (L8-19) and
# ``triton.runtime`` is fully initialized. Nothing later in triton/__init__.py
# rebinds ``triton.autotune``, so replacing both names here is safe and is not
# overwritten by triton's own initialization. Failures are non-fatal: they keep
# ``import triton`` usable and fall back to the stock autotuner.
import triton as _triton

_original_autotune = _triton.autotune
_lazy_autotune_installed = False


def _install_ascend_autotune():
    """Import the Ascend runtime once so ``_patch_autotune`` runs for real."""
    global _lazy_autotune_installed
    if _lazy_autotune_installed:
        return
    try:
        import triton.backends.ascend.runtime  # noqa: F401  (side effect: patches triton.autotune)
    except Exception as _asc_rt_exc:  # pragma: no cover - defensive
        import warnings

        warnings.warn(f"triton-ascend: failed to auto-load ascend runtime autotuner: {_asc_rt_exc!r}")
        # Fall back to the stock autotune so callers keep working and the proxy
        # below does not recurse into itself.
        _triton.autotune = _original_autotune
        try:
            _triton.runtime.autotune = _original_autotune
        except Exception:  # pragma: no cover - defensive
            pass
    finally:
        _lazy_autotune_installed = True


class _LazyAutotune:
    """Proxy that swaps in AutoTilingTuner on first call, then delegates."""

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        _install_ascend_autotune()
        # After the import above, runtime's ``_patch_autotune`` has rebound
        # triton.autotune (and triton.runtime.autotune) to the real Ascend
        # autotune; on failure we restored the stock autotune above. Either way
        # ``_triton.autotune`` is now a concrete callable, not this proxy.
        return _triton.autotune(*args, **kwargs)


_lazy_autotune_proxy = _LazyAutotune()
_triton.autotune = _lazy_autotune_proxy
try:
    _triton.runtime.autotune = _lazy_autotune_proxy
except Exception:  # pragma: no cover - defensive
    pass

__all__ = ["libdevice", "extension"]
