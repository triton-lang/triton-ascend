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
from enum import Enum, auto

from triton.language import core as tl
from triton.language.core import builtin, _unwrap_if_constexpr
from .core import AddressSpace, buffer, buffer_type, _require_space, _shape
from .aux_ops import annotate_buffer_versions


class Layout(Enum):
    """Compile-time allocation constructors, exposed as ``tla.layout``.

    ``nd`` uses the supplied physical shape unchanged (the default).
    ``nz`` converts an aligned logical matrix shape to a physical NZ shape;
    see :func:`alloc` for supported spaces and element types.
    Neither constructor adds layout metadata to the returned buffer.
    """

    nd = auto()
    nz = auto()


def _allocation_shape(shape, dtype, space, layout):
    layout = _unwrap_if_constexpr(layout)
    if not isinstance(layout, Layout):
        raise TypeError(f"expected a tla.layout member, got {type(layout).__name__}")
    shape = _shape(_unwrap_if_constexpr(shape))
    if layout is Layout.nd:
        return shape
    if len(shape) != 2:
        raise ValueError("NZ allocation requires a two-dimensional logical shape (M, N)")
    dtype = _unwrap_if_constexpr(dtype)
    if space is AddressSpace.l1 and dtype in (tl.float16, tl.bfloat16, tl.float32):
        n0 = 256 // dtype.primitive_bitwidth
    elif space is AddressSpace.l0c and dtype in (tl.float32, tl.int32):
        n0 = 16
    else:
        raise ValueError("NZ allocation supports L1 float16/bfloat16/float32 and L0C float32/int32; "
                         f"got {space.name} {dtype}")
    m, n = shape
    if m % 16 or n % n0:
        raise ValueError(f"NZ allocation requires M divisible by 16 and N divisible by {n0}; got {shape}")
    return (n // n0, m // 16, 16, n0)


@builtin
def alloc(shape, dtype, space=AddressSpace.ub, versions=1, *, layout=Layout.nd, _semantic=None):
    """Allocates an uninitialized, contiguous on-chip buffer.

    :param shape: Non-empty sequence of positive compile-time extents in
        elements. Physical extents by default; logical ``(M, N)`` for NZ.
    :type shape: tuple[int, ...] or list[int]
    :param dtype: Compile-time element type, for example ``tl.float16``.
    :type dtype: triton.language.dtype
    :param space: Compile-time address space. Defaults to ``tla.space.ub``.
    :type space: tla.space
    :param versions: Positive compile-time multi-buffer version count.
        Defaults to ``1``, which disables multi-buffering for this buffer.
    :type versions: int
    :param layout: Compile-time allocation layout. ``tla.layout.nd`` (default)
        leaves the shape unchanged. ``tla.layout.nz`` converts ``(M, N)`` to
        ``(N // N0, M // 16, 16, N0)``. L1 supports float16/bfloat16 with
        ``N0=16`` and float32 with ``N0=8``; L0C supports float32/int32 with
        ``N0=16``. Other NZ combinations are rejected. M and N must be exact
        multiples of 16 and N0 respectively; allocation never adds padding.
    :type layout: tla.layout
    :return: An uninitialized buffer with physical shape and row-major strides.
        Initialize its contents before reading them.
    :rtype: tla.buffer

    .. note::
        Every independent allocation carries ``annotation.mark {mem_unique}``
        so the HIVM memory planner cannot reuse its storage for another buffer,
        including during planning retries. Use views to explicitly share storage.
        The TLA backend pipeline also disables tightly-coupled buffer reuse.

        GM memory must be supplied as a kernel argument and wrapped with
        ``tla.view``. Current backend allocation support covers UB, L1 and L0C.
        L0A/L0B and SSBUF have native type mappings, but explicit allocations
        require additional backend memory-planning support. BT/FB allocation
        is likewise backend-dependent. L0A_SF/L0B_SF have no native mapping
        and raise ``NotImplementedError`` before emitting an allocation.
        Allocate Cube-local buffers inside ``tla.scope.cube()``.

        Layout is only an allocation constructor: the returned buffer has no
        layout metadata. Indexing, ``subview`` and ``reshape`` keep their existing
        physical-coordinate semantics. Allocation neither converts data nor
        initializes padding; DMA and instruction-specific packing remain explicit.

    Example inside a ``@tla.jit`` kernel::

        with tla.scope.cube():
            acc = tla.alloc((16, 64), tl.float32,
                            space=tla.space.l0c, layout=tla.layout.nz)
            # acc.shape == (4, 1, 16, 16)
    """
    space = _require_space(space)
    versions = _unwrap_if_constexpr(versions)
    if space is AddressSpace.gm:
        raise ValueError("GM buffers must be supplied as kernel arguments")
    if type(versions) is not int or versions < 1:
        raise ValueError("buffer versions must be a positive integer")
    ty = buffer_type(_allocation_shape(shape, dtype, space, layout), dtype, space)
    buf = buffer(_semantic.builder.create_tla_alloc(ty.to_ir(_semantic.builder)), ty)
    _semantic.builder.create_annotation_mark(buf.handle, "mem_unique", _semantic.builder.get_unit_attr())
    annotate_buffer_versions(buf, versions, _semantic=_semantic)
    return buf


@builtin
def alloc_buffer(shape, dtype, space=AddressSpace.ub, versions=1, *, layout=Layout.nd, _semantic=None):
    """Allocates an on-chip buffer; alias of :func:`alloc`.

    :param shape: Physical extents, or logical ``(M, N)`` for NZ; see :func:`alloc`.
    :type shape: tuple[int, ...] or list[int]
    :param dtype: Compile-time element type, for example ``tl.float16``.
    :type dtype: triton.language.dtype
    :param space: Compile-time address space. Defaults to ``tla.space.ub``.
        The restrictions of :func:`alloc` also apply here.
    :type space: tla.space
    :param versions: Positive compile-time multi-buffer count. Defaults to ``1``.
    :type versions: int
    :param layout: Compile-time allocation layout; see :func:`alloc`.
        Defaults to ``tla.layout.nd``.
    :type layout: tla.layout
    :return: An uninitialized, contiguous buffer.
    :rtype: tla.buffer
    """
    return alloc(shape, dtype, space, versions, layout=layout, _semantic=_semantic)
