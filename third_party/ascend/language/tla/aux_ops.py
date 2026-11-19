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
Compiler-hint helpers for the TLA namespace.

Migrated from TLA ``cann/extension/aux_ops.py`` (SIMD VF related subset):
``annotate_buffer_versions`` is the frontend declaration consumed by the
NPU-IR multi-buffer pass (``annotation.mark {hivm.multi_buffer = N : i32}``).
"""

from triton.language import core
from triton.language.core import (
    _unwrap_if_constexpr,
    builtin,
)

from triton._C.libtriton import ir

__all__ = ["annotate_buffer_versions", "assume"]


@builtin
def assume(cond, _semantic=None):
    """
    Tell the compiler that a scalar boolean condition is true.

    Emits ``llvm.intr.assume`` as an optimization hint, not a runtime check.
    Unlike TileLang's host-checkable ``T.assume`` constraints, no launch-time
    assertion is generated. If the condition is false, behavior is undefined;
    use ``tla.device_assert`` when a runtime diagnostic is required.

    :param cond: Python/constexpr bool or scalar ``int1`` tensor; integer,
        block tensor, and SIMD vector conditions are not supported
    :return: None
    """
    cond = _unwrap_if_constexpr(cond)
    if isinstance(cond, core.tensor):
        if cond.type.is_block() or cond.dtype != core.int1:
            raise TypeError("assume requires a scalar int1 condition")
    elif not isinstance(cond, bool):
        raise TypeError("assume requires a bool or scalar int1 condition")
    _semantic.assume(_semantic.to_tensor(cond))


def compile_hint_impl(ptr, hint_name: str, hint_val, builder: ir.builder):
    # Check isinstance(hint_val, bool) first to handle False explicitly
    if isinstance(hint_val, bool):
        hint_val = builder.get_bool_attr(hint_val)
    elif not hint_val:
        hint_val = builder.get_unit_attr()
    elif isinstance(hint_val, int):
        hint_val = builder.get_int32_attr(hint_val)
    elif isinstance(hint_val, core.constexpr):
        hint_val = builder.get_string_attr(hint_val.value)
    elif isinstance(hint_val, (list, core.tuple)):
        # only support i64 array attr for now
        hint_val = builder.get_i64_array_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    builder.create_annotation_mark(ptr.handle, hint_name, hint_val)


@builtin
def annotate_buffer_versions(buf, versions, _semantic=None):
    """
    Declare the multi-buffer version count for one storage buffer.

    ``versions`` must be a positive int; ``1`` opts the buffer out of
    multi-buffering.  The count is attached as an annotation hint consumed by
    the NPU-IR multi-buffer pass.

    :param buf: the ``tla.buffer`` to annotate
    :param versions: the version count (positive int)
    """
    v = _unwrap_if_constexpr(versions)
    if type(v) is not int or isinstance(v, bool) or v < 1:
        raise ValueError(f"buffer versions must be a positive int, got {v!r}")
    if v == 1:
        return
    compile_hint_impl(buf, "hivm.multi_buffer", v, _semantic.builder)
