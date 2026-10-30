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


# Frontend space names; "gm"/"ub" map 1:1, the rest as in tla_builder.cc.
SPACES = ("gm", "ub", "l1", "l0a", "l0b", "l0c", "bt", "fb")
_HIVM_SCOPE = {
    "gm": "gm", "ub": "ub", "l1": "cbuf", "l0a": "ca", "l0b": "cb", "l0c": "cc", "bt": "biasbuf", "fb": "fixbuf"
}


def space_of(value):
    """Frontend space of a TLA value, or None for scalars.

    Resolves the enum on buffer/buffer_access and the printed
    ``#hivm.address_space<...>`` type on IR values.
    """
    buf = getattr(value, "buffer", value)
    space = getattr(buf, "space", None)
    if space is not None:
        return space
    # IR values expose get_type(); host values carry .type.
    ty = str(getattr(value, "type", None) or (value.get_type() if hasattr(value, "get_type") else None) or "")
    return next((name for name, scope in _HIVM_SCOPE.items() if f"#hivm.address_space<{scope}>" in ty), None)


def is_ub(value):
    """True for scalars or UB-backed values."""
    return space_of(value) in (None, "ub")


class buffer_type(tl.base_type):

    def __init__(self, shape, dtype, space="gm", strides=None, offset=0):
        self.shape = _shape(shape)
        self.element_ty = tl._unwrap_if_constexpr(dtype)
        self.space = tl._unwrap_if_constexpr(space)
        if self.space not in SPACES:
            raise ValueError(f"unsupported buffer space: {self.space}")
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
        return f"TLA_{self.space}_{self.element_ty.mangle()}_{self.shape}_{self.strides}_{self.offset}"

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
