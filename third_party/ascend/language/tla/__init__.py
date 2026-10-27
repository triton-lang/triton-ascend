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
"""
TLA (Triton Language for Ascend) namespace.

Independent ``tla.*`` programming namespace for Ascend NPU kernels, migrated
from the TLA repo's SIMD VF additions.  It does **not** reuse the existing
``al``/``bl`` general interfaces (different programming paradigm).

Exports: ``alloc_buffer``/``alloc`` (on-chip buffer + versions),
``copy`` (GM<->buffer), ``view`` (GM view), ``scope`` (core-mode scopes:
``tla.scope.simd()`` / ``tla.scope.cube()``),
``simd`` (vector-register ops), ``annotate_buffer_versions``.
"""

from .core import buffer, buffer_type, vector, vector_type
from .scope import scope
from .alloc import alloc_buffer, alloc
from .copy import copy
from .view import view, reshape
from .aux_ops import annotate_buffer_versions
from . import math
from .math import compensated_rstd
from .device_assert import device_assert

# SIMD vector-register ops.  ``simd`` is exported as a module so its ``cast``
# (ave.hir.vextf) does not shadow unrelated top-level names.
from . import simd

# Cube-core ops (``mmad``/``tla.copy`` L1/L0C routes).
from . import cube

__all__ = [
    "buffer",
    "buffer_type",
    "vector",
    "vector_type",

    # buffer alloc / copy / view
    "alloc_buffer",
    "alloc",
    "copy",
    "view",
    "reshape",

    # core-mode scopes (scope.simd / scope.cube)
    "scope",

    # math helpers / diagnostics
    "math",
    "compensated_rstd",
    "device_assert",

    # aux ops
    "annotate_buffer_versions",

    # simd ops
    "simd",

    # cube ops
    "cube",
]
