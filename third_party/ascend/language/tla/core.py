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
"""TLA memory and vector-register values, independent of Triton tensors."""
from enum import Enum, auto

from triton.language import core as tl


def _shape(values):
    values = tuple(tl._unwrap_if_constexpr(v) for v in values)
    if not values or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("buffer shape must contain positive compile-time integers")
    return values


def _row_major_strides(sizes):
    strides, stride = [], 1
    for size in reversed(sizes):
        strides.insert(0, stride)
        stride *= size
    return tuple(strides)


class AddressSpace(Enum):
    """Compile-time address spaces for TLA buffers.

    ``gm`` and ``ub`` map directly to HIVM. ``l1``, ``l0a``, ``l0b``, and
    ``l0c`` map to ``cbuf``, ``ca``, ``cb``, and ``cc``. ``ssbuf`` maps to
    ``ssbuf``, while ``bt`` and ``fb`` map to ``biasbuf`` and ``fixbuf``.
    See :func:`alloc` for allocation restrictions.

    ``l0a_sf`` and ``l0b_sf`` are reserved and raise ``NotImplementedError``
    until HIVM defines their representation.
    """

    gm = auto()
    ub = auto()
    l1 = auto()
    l0a = auto()
    l0b = auto()
    l0c = auto()
    ssbuf = auto()
    bt = auto()
    fb = auto()
    l0a_sf = auto()
    l0b_sf = auto()


def _require_space(space) -> AddressSpace:
    space = tl._unwrap_if_constexpr(space)
    if not isinstance(space, AddressSpace):
        raise TypeError(f"expected a tla.space member, got {type(space).__name__}")
    if space in (AddressSpace.l0a_sf, AddressSpace.l0b_sf):
        raise NotImplementedError(f"{space.name} requires a backend-defined scale-factor buffer mapping")
    return space


class buffer_type(tl.base_type):
    """Structural type of a TLA buffer.

    :param shape: Non-empty sequence of positive compile-time extents in
        elements.
    :type shape: tuple[int, ...] or list[int]
    :param dtype: Compile-time element type, for example ``tl.float16``.
    :type dtype: triton.language.dtype
    :param space: Compile-time address space name or :class:`AddressSpace`
        member. Defaults to ``"gm"``.
    :type space: tla.space
    :param strides: Element strides. Defaults to ``None``, which selects
        row-major strides. Must have the same rank as ``shape``.
    :type strides: tuple[int, ...] or None
    :param offset: Linear offset in elements. Defaults to ``0``; ``None``
        denotes a dynamic subview offset.
    :type offset: int or None

    The type is structural: register-width, layout and operation-specific
    restrictions belong to the DMA and SIMD ops that consume the buffer.
    ``L0A_SF`` and ``L0B_SF`` are reserved and raise ``NotImplementedError``.
    """

    def __init__(self, shape, dtype, space=AddressSpace.gm, strides=None, offset=0):
        self.shape = _shape(shape)
        self.element_ty = tl._unwrap_if_constexpr(dtype)
        self.space = _require_space(space)
        if strides is None:
            strides = _row_major_strides(self.shape)
        self.strides = _shape(strides)
        if len(self.strides) != len(self.shape):
            raise ValueError("strides must match buffer rank")
        self.offset = offset  # None denotes a dynamic subview offset.

    def to_ir(self, builder):
        return builder.create_tla_buffer_type(list(self.shape), self.element_ty.to_ir(builder), self.space,
                                              list(self.strides), self.offset)

    def _flatten_ir_types(self, builder, out):
        out.append(self.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        return buffer(handles[cursor], self), cursor + 1

    def mangle(self):
        return f"TLA_{self.space.name}_{self.element_ty.mangle()}_{self.shape}_{self.strides}_{self.offset}"

    def __eq__(self, other):
        return isinstance(other, buffer_type) and vars(self) == vars(other)

    def __str__(self):
        return self.mangle()


class buffer_access:
    """Concrete buffer element range addressed by ``buf[i0, i1, ...]``.

    Produced by :meth:`buffer.__getitem__`; carries the buffer plus the index
    tuple until a ``vlds``/``vsts`` op unpacks it.
    """

    __slots__ = ("buffer", "indices")

    def __init__(self, buffer, indices):
        self.buffer = buffer
        self.indices = indices


class buffer(tl.base_value):
    """A typed TLA buffer value.

    A buffer aliases global memory obtained from a kernel pointer via
    :func:`view`, or on-chip storage obtained via :func:`alloc`, and is
    consumed by the DMA, cube and SIMD memory interfaces. ``buf[i0, i1, ...]``
    returns a :class:`buffer_access` carrying the indices for ``simd.vlds`` /
    ``simd.vsts``.

    GM buffers must be supplied as kernel arguments and wrapped with
    :func:`view`; they cannot be produced by :func:`alloc`.
    """

    def __init__(self, handle, ty):
        self.handle, self.type = handle, ty
        self.shape, self.dtype, self.space, self.strides = ty.shape, ty.element_ty, ty.space, ty.strides
        self.offset = ty.offset

    def _flatten_ir(self, handles):
        handles.append(self.handle)

    def __getitem__(self, *indices):
        if len(indices) == 1 and isinstance(indices[0], tl.tuple):
            indices = tuple(indices[0].values)
        return buffer_access(self, indices)


class vector_type(tl.base_type):
    """A one-dimensional MLIR vector, including i1 predicate vectors.

    Register-width and operation-specific restrictions belong to SIMD ops,
    not to the structural type.
    """

    def __init__(self, lanes, element_ty):
        lanes = tl._unwrap_if_constexpr(lanes)
        element_ty = tl._unwrap_if_constexpr(element_ty)
        if type(lanes) is not int or lanes <= 0:
            raise ValueError("vector lanes must be a positive compile-time integer")
        if type(element_ty) is not tl.dtype or not (element_ty.is_int() or element_ty.is_floating()):
            raise TypeError("vector element type must be a scalar integer or floating-point dtype")
        self.lanes = lanes
        self.element_ty = element_ty

    @property
    def shape(self):
        return (self.lanes, )

    def to_ir(self, builder):
        return builder.create_tla_vector_type(self.lanes, self.element_ty.to_ir(builder))

    def _flatten_ir_types(self, builder, out):
        out.append(self.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        return vector(handles[cursor], self), cursor + 1

    def mangle(self):
        return f"TLA_vec_{self.lanes}_{self.element_ty.mangle()}"

    def __eq__(self, other):
        return isinstance(other, vector_type) and self.lanes == other.lanes and self.element_ty == other.element_ty

    def __str__(self):
        return f"vector<{self.lanes}x{self.element_ty}>"


class vector(tl.base_value):
    """A SIMD register value; tensor operations and implicit conversion are unsupported."""

    def __init__(self, handle, ty):
        if not isinstance(ty, vector_type):
            raise TypeError("tla.vector requires a tla.vector_type")
        self.handle, self.type = handle, ty

    @property
    def shape(self):
        return self.type.shape

    @property
    def dtype(self):
        return self.type.element_ty

    def _flatten_ir(self, handles):
        handles.append(self.handle)

    def __bool__(self):
        raise TypeError("a tla.vector cannot be used as a scalar condition")
