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
"""Explicit on-chip TLA memref allocation."""
from triton.language.core import builtin, _unwrap_if_constexpr
from .core import buffer, buffer_type
from .aux_ops import annotate_buffer_versions


@builtin
def alloc(shape, dtype, space="ub", versions=1, _semantic=None):
    """Allocates an uninitialized, contiguous on-chip buffer.

    :param shape: Non-empty sequence of positive compile-time extents in
        elements, for example ``(16, 32)``. Strides are row-major.
    :type shape: tuple[int, ...] or list[int]
    :param dtype: Compile-time element type, for example ``tl.float16``.
    :type dtype: triton.language.dtype
    :param space: Compile-time address-space name. Use ``"ub"`` (default),
        ``"l1"`` or ``"l0c"`` for backend-supported local allocations.
    :type space: str
    :param versions: Positive compile-time multi-buffer version count.
        Defaults to ``1``, which disables multi-buffering for this buffer.
    :type versions: int
    :return: The allocated buffer. Initialize its contents before reading them.
    :rtype: tla.buffer

    .. note::
        GM memory must be supplied as a kernel argument and wrapped with
        ``tla.view``. Although the frontend recognizes ``"l0a"``, ``"l0b"``,
        ``"bt"`` and ``"fb"``, this does not imply allocation support.
        The current memory planner supports UB, L1 and L0C allocations.
        Allocate Cube-local L1/L0C buffers inside ``tla.scope.cube()``.
    """
    space = _unwrap_if_constexpr(space)
    versions = _unwrap_if_constexpr(versions)
    if space == "gm":
        raise ValueError("GM buffers must be supplied as kernel arguments")
    if type(versions) is not int or versions < 1:
        raise ValueError("buffer versions must be a positive integer")
    ty = buffer_type(shape, dtype, space)
    buf = buffer(_semantic.builder.create_tla_alloc(ty.to_ir(_semantic.builder)), ty)
    annotate_buffer_versions(buf, versions, _semantic=_semantic)
    return buf


@builtin
def alloc_buffer(shape, dtype, space="ub", versions=1, _semantic=None):
    """Allocates an on-chip buffer; alias of :func:`alloc`.

    :param shape: Non-empty sequence of positive compile-time extents in elements.
    :type shape: tuple[int, ...] or list[int]
    :param dtype: Compile-time element type, for example ``tl.float16``.
    :type dtype: triton.language.dtype
    :param space: Compile-time address-space name. Defaults to ``"ub"``.
        The allocation restrictions of :func:`alloc` also apply here.
    :type space: str
    :param versions: Positive compile-time multi-buffer count. Defaults to ``1``.
    :type versions: int
    :return: An uninitialized, contiguous buffer.
    :rtype: tla.buffer
    """
    return alloc(shape, dtype, space, versions, _semantic=_semantic)
