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
``fill`` (scalar fill of a contiguous UB buffer),
``copy`` (GM<->buffer), ``view`` (GM view), ``scope`` (core-mode scopes:
``tla.scope.simd()`` / ``tla.scope.cube()``),
``simd`` (vector-register ops), ``where`` (eager scalar/SIMD select),
``annotate_buffer_versions``, and scalar dtypes.

Scalar types share Triton's dtype protocol, so ``tla.float32`` can be passed
directly to buffer, vector, and cast APIs.  The re-exported names map to the
following Triton dtypes:

``bfloat16`` -> ``bf16``, ``float16`` -> ``fp16``, ``float32`` -> ``fp32``,
``int8``/``int16``/``int32``/``int64`` keep their names, and
``uint8``/``uint16``/``uint32``/``uint64`` keep their names.  The ``dtype``
constructor uses Triton's canonical names (for example, ``dtype("fp32")``).

.. note::
    The scalar dtype objects are re-exported unchanged from
    :mod:`triton.language.core`; they are the same objects as
    ``triton.language.float32`` and the other public dtype names.
"""

# Re-export Triton's scalar dtype objects unchanged.  Using the original
# objects keeps ``tla.buffer_type``, ``tla.vector_type``, casts and ``@tla.jit``
# interoperable with ``triton.language`` without wrappers or conversions.
# These are dtype objects, not functions; their public spellings are the
# Triton names above, while ``tla.dtype(...)`` accepts canonical names such as
# ``"fp32"``.
from triton.language.core import (
    bfloat16,
    dtype,
    float16,
    float32,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

from .core import AddressSpace, buffer, buffer_type, vector, vector_type
from .scope import scope
from .alloc import alloc_buffer, alloc
from .fill import fill
from .copy import copy
from .dma import (
    PadMode,
    EvictionPolicy,
    AtomicKind,
    UnitFlagMode,
    FixpipeDMAMode,
    FixpipeDualDstMode,
    FixpipeSubBlock,
    FixpipePreQuantMode,
    FixpipePreReluMode,
    LoadGM2UBParams,
    StoreUB2GMParams,
    LoadGM2L1Params,
    CopyUB2L1Params,
    CopyL12UBParams,
    CopyL12BTParams,
    CopyL12FBParams,
    CopyL1ToL0Params,
    FixpipeL0C2GMParams,
    FixpipeL0C2L1Params,
    FixpipeL0C2UBParams,
    load_gm_to_ub,
    store_ub_to_gm,
    load_gm_to_l1,
    load_gm_to_l1_mx_scale,
    copy_ub_to_l1,
    copy_l1_to_ub,
    copy_l1_to_bt,
    copy_l1_to_fb,
    copy_l1_to_l0,
    store_l0c_to_gm,
    copy_l0c_to_ub,
    copy_l0c_to_l1,
    set_ctrl,
    set_atomic,
)

from .view import view, subview, reshape
from .scalar import select, abs, maximum, minimum, cast, max_value, cdiv, infinity, clamp
from .aux_ops import annotate_buffer_versions, assume

from . import math
from .math import compensated_rstd, truncmod, sqrt
from .debug import device_assert, device_print

# SIMD vector-register ops.  ``simd`` is exported as a module so its ``cast``
# (ave.hir.vextf) does not shadow unrelated top-level names.
from . import simd

# Scalar / SIMD-register selection (eager, unlike TileLang T.if_then_else).
from .selection import where

# Cube-core ops (``mmad``/``mmad_mx``/``tla.copy`` L1/L0C routes).
from . import cube
from .cube import MxFormat

# Explicit pipeline / global synchronization.
from . import sync
from .sync import sub_block_id


def jit(fn=None, **kwargs):
    """``@tla.jit``: like ``triton.jit``, but lowered via the TLA frontend."""
    from .jit import TLAJITFunction

    def decorator(f):
        return TLAJITFunction(f, **kwargs)

    return decorator(fn) if fn is not None else decorator


__all__ = [
    # scalar dtypes (shared with triton.language)
    "bfloat16",
    "dtype",
    "float16",
    "float32",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "AddressSpace",
    "PadMode",
    "EvictionPolicy",
    "AtomicKind",
    "UnitFlagMode",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipeSubBlock",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
    "LoadGM2UBParams",
    "StoreUB2GMParams",
    "LoadGM2L1Params",
    "CopyUB2L1Params",
    "CopyL12UBParams",
    "CopyL12BTParams",
    "CopyL12FBParams",
    "CopyL1ToL0Params",
    "FixpipeL0C2GMParams",
    "FixpipeL0C2L1Params",
    "FixpipeL0C2UBParams",
    "load_gm_to_ub",
    "store_ub_to_gm",
    "load_gm_to_l1",
    "load_gm_to_l1_mx_scale",
    "copy_ub_to_l1",
    "copy_l1_to_ub",
    "copy_l1_to_bt",
    "copy_l1_to_fb",
    "copy_l1_to_l0",
    "store_l0c_to_gm",
    "copy_l0c_to_ub",
    "copy_l0c_to_l1",
    "set_ctrl",
    "set_atomic",
    "buffer",
    "buffer_type",
    "vector",
    "vector_type",

    # buffer alloc / copy / view
    "alloc_buffer",
    "alloc",
    "fill",
    "copy",
    "view",
    "subview",
    "reshape",

    # scalar expressions
    "select",
    "abs",
    "maximum",
    "minimum",
    "cast",
    "max_value",
    "cdiv",
    "infinity",
    "clamp",

    # core-mode scopes (scope.simd / scope.cube)
    "scope",

    # math helpers / diagnostics
    "math",
    "compensated_rstd",
    "truncmod",
    "sqrt",
    "device_assert",
    "device_print",

    # aux ops
    "annotate_buffer_versions",
    "assume",

    # simd ops
    "simd",

    # scalar / SIMD selection
    "where",

    # cube ops
    "cube",
    "MxFormat",

    # explicit synchronization
    "sync",
    "sub_block_id",

    # kernel decorator
    "jit",
]
