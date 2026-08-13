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


# ---------------------------------------------------------------------------
# Install backward-compat import hooks for relocated modules BEFORE any
# sub-imports.  ``cann/extension/core.py`` imports
# ``triton.extension.buffer.language``, so the hook must already be in
# ``sys.meta_path`` when the extension subpackage is loaded below.
# ---------------------------------------------------------------------------
def _install_compat_hooks():
    import importlib.util
    import os
    _ext_hook = os.path.join(os.path.dirname(__file__), '..', 'extension', '__init__.py')
    _ext_hook = os.path.normpath(_ext_hook)
    if not os.path.exists(_ext_hook):
        return
    _spec = importlib.util.spec_from_file_location('triton._ext_compat', _ext_hook)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)


_install_compat_hooks()
del _install_compat_hooks

from triton.language import math
from triton.backends.ascend.utils import triton_enable_libdevice_simt

from . import libdevice
from . import extension

extension.parallel = extension.aux_ops.parallel
if not triton_enable_libdevice_simt():
    libdevice.atan2 = extension.math_ops.atan2
math.tanh = libdevice.tanh

__all__ = ["libdevice", "extension"]
