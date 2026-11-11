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
from .core import buffer, buffer_type, _normalize_space
from .aux_ops import annotate_buffer_versions


@builtin
def alloc(shape, dtype, space=None, versions=1, _semantic=None, *, scope=None):
    """Allocates an uninitialized, contiguous on-chip buffer.

    :param shape: Non-empty sequence of positive compile-time extents in
        elements, for example ``(16, 32)``. Strides are row-major.
    :type shape: tuple[int, ...] or list[int]
    :param dtype: Compile-time element type, for example ``tl.float16``.
    :type dtype: triton.language.dtype
    :param space: Compile-time address space, for example ``tla.AddressSpace.L0C``
        or the legacy string ``"l0c"``. If both ``space`` and ``scope`` are
        omitted, allocates in UB. Specify at most one of them.
    :type space: tla.AddressSpace, str or None
    :param versions: Positive compile-time multi-buffer version count.
        Defaults to ``1``, which disables multi-buffering for this buffer.
    :type versions: int
    :param scope: Keyword-only spelling of ``space``, for example
        ``scope=tla.AddressSpace.L1``. Defaults to ``None``.
    :type scope: tla.AddressSpace, str or None
    :return: The allocated buffer. Initialize its contents before reading them.
    :rtype: tla.buffer

    .. note::
        Allocations do not impose unique physical storage. Backend memory
        planning decides whether storage can be reused; use views to express
        explicit sharing of a logical buffer.

        GM memory must be supplied as a kernel argument and wrapped with
        ``tla.view``. Current backend allocation support covers UB, L1 and L0C.
        L0A/L0B and SSBUF have native type mappings, but explicit allocations
        require additional backend memory-planning support. BT/FB allocation
        is likewise backend-dependent. L0A_SF/L0B_SF have no native mapping
        and raise ``NotImplementedError`` before emitting an allocation.
        Allocate Cube-local buffers inside ``tla.scope.cube()``.

    Example inside a ``@tla.jit`` kernel::

        with tla.scope.cube():
            acc = tla.alloc((4, 1, 16, 16), tl.float32, scope=tla.AddressSpace.L0C)
    """
    space = _unwrap_if_constexpr(space)
    scope = _unwrap_if_constexpr(scope)
    if space is not None and scope is not None:
        raise ValueError("specify only one of space and scope")
    space = _normalize_space(scope if scope is not None else space if space is not None else "ub")
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
def alloc_buffer(shape, dtype, space=None, versions=1, _semantic=None, *, scope=None):
    """Allocates an on-chip buffer; alias of :func:`alloc`.

    :param shape: Non-empty sequence of positive compile-time extents in elements.
    :type shape: tuple[int, ...] or list[int]
    :param dtype: Compile-time element type, for example ``tl.float16``.
    :type dtype: triton.language.dtype
    :param space: Compile-time address space. Defaults to UB when ``scope``
        is also omitted. The restrictions of :func:`alloc` also apply here.
    :type space: tla.AddressSpace, str or None
    :param versions: Positive compile-time multi-buffer count. Defaults to ``1``.
    :type versions: int
    :param scope: Keyword-only spelling of ``space``. Defaults to ``None``;
        specify at most one of ``space`` and ``scope``.
    :type scope: tla.AddressSpace, str or None
    :return: An uninitialized, contiguous buffer.
    :rtype: tla.buffer
    """
    return alloc(shape, dtype, space, versions, _semantic=_semantic, scope=scope)
