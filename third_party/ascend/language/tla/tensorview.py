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
"""TensorView metadata and GM/UB accesses backed by the native tv dialect."""
from triton.language import core as tl
from .core import buffer


def _builder(builder):
    if not hasattr(builder, "create_tla_tensor_view_type"):
        from triton._C.libtriton.ascend import ir as ascend_ir
        from .builder import setup_unified_builder
        setup_unified_builder(builder, ascend_ir.tla_builder(builder))
    return builder


class tensor_view_type(tl.base_type):
    """Type of an N-dimensional logical TensorView."""

    def __init__(self, element_ty: tl.dtype, rank: int, view_kind=None, tile=(), traversal_strides=(), sparse_dims=(),
                 padding_value="zero", strides=()):
        self.element_ty = element_ty
        self.rank = rank
        self.view_kind = view_kind
        self.tile = tuple(tile)
        self.traversal_strides = tuple(traversal_strides)
        self.sparse_dims = tuple(sparse_dims)
        self.padding_value = padding_value
        self.strides = tuple(strides)

    def _unflatten_ir(self, handles, cursor):
        return tensor_view(handles[cursor], self), cursor + 1

    def to_ir(self, builder):
        return _builder(builder).create_tla_tensor_view_type(self.element_ty.to_ir(builder), self.rank, self.view_kind
                                                             or "", self.tile, self.traversal_strides, self.sparse_dims,
                                                             self.padding_value, self.strides)

    def _flatten_ir_types(self, builder, out):
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        suffix = (""
                  if self.view_kind is None else f", {self.view_kind}, tile={self.tile}, padding={self.padding_value}")
        return f"tensor_view<{self.element_ty}>[{self.rank}{suffix}]"

    def __eq__(self, other) -> bool:
        return isinstance(other, tensor_view_type) and vars(self) == vars(other)

    def mangle(self) -> str:
        encoding = "B" if self.view_kind is None else self.view_kind.upper()
        tile = "x".join(str(value) for value in self.tile)
        traversal = "x".join(str(value) for value in self.traversal_strides)
        sparse = "x".join(str(value) for value in self.sparse_dims)
        padding = self.padding_value.upper().replace("-", "N")
        strides = "x".join(str(value).replace("-", "N") for value in self.strides)
        return f"TV{self.element_ty.mangle()}R{self.rank}K{encoding}T{tile}V{traversal}S{sparse}P{padding}D{strides}"


class tensor_view(tl.base_value):
    """A logical tensor accessed by tiled `tla.load` and `tla.store` operations."""

    def __init__(self, handle, ty):
        self.handle, self.type = handle, ty

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty

    @property
    def rank(self):
        return self.type.rank

    @property
    def tile(self):
        return self.type.tile

    def __str__(self) -> str:
        return str(self.type)


def _make_view(base, shape, strides, tile, kind, traversal, sparse, padding, semantic):
    shape = tuple(shape)
    strides = tuple(strides)
    rank = len(shape)
    if not rank or len(strides) != rank:
        raise ValueError("shape and strides must have the same nonzero rank")
    if isinstance(base, buffer):
        if base.space != "gm":
            raise ValueError("TensorView requires a GM source buffer")
        dtype = base.dtype
    elif isinstance(base, tl.tensor) and base.type.is_ptr() and not base.type.element_ty.is_block():
        dtype = base.dtype.element_ty
        if dtype == tl.int1:
            base = semantic.cast(base, tl.pointer_type(tl.int8, base.type.address_space))
            dtype = tl.int8
    else:
        raise TypeError("TensorView requires a scalar pointer or GM buffer")
    tile = tuple(tl._unwrap_if_constexpr(v) for v in tile)
    traversal = tuple(tl._unwrap_if_constexpr(v) for v in traversal)
    sparse = tuple(tl._unwrap_if_constexpr(v) for v in sparse)
    if len(tile) != rank or any(type(v) is not int or v <= 0 for v in tile):
        raise ValueError("tile must contain one positive compile-time integer per dimension")
    if kind == "strided" and (len(traversal) != rank or any(type(v) is not int or v <= 0 for v in traversal)):
        raise ValueError("traversal_strides must contain one positive compile-time integer per dimension")
    if kind == "gather_scatter" and (not sparse or len(set(sparse)) != len(sparse)
                                     or any(type(v) is not int or not 0 <= v < rank for v in sparse)):
        raise ValueError("sparse_dim must contain unique dimensions within the view rank")
    padding = tl._unwrap_if_constexpr(padding)
    if padding not in ("zero", "nan", "inf", "-inf") or (padding != "zero" and not dtype.is_floating()):
        raise ValueError(
            "padding_value must be zero, nan, inf or -inf; nonzero padding requires a floating-point dtype")
    static_strides = tuple(v if type(v := tl._unwrap_if_constexpr(s)) is int else None for s in strides)
    base_type = tensor_view_type(dtype, rank, strides=static_strides)
    builder = _builder(semantic.builder)
    source = builder.create_tla_tensor_view(base.handle, semantic._convert_to_ir_values(shape),
                                            semantic._convert_to_ir_values(strides), base_type.to_ir(builder))
    ty = tensor_view_type(dtype, rank, kind, tile, traversal, sparse, padding, static_strides)
    handle = builder.create_tla_encoded_view(source, ty.to_ir(builder), kind)
    return tensor_view(handle, ty)


@tl.builtin
def make_partition_view(base, shape, strides, tile, padding_value="zero", _semantic=None):
    return _make_view(base, shape, strides, tile, "partition", (), (), padding_value, _semantic)


@tl.builtin
def make_strided_view(base, shape, strides, tile, traversal_strides, padding_value="zero", _semantic=None):
    return _make_view(base, shape, strides, tile, "strided", traversal_strides, (), padding_value, _semantic)


@tl.builtin
def make_gather_scatter_view(base, shape, strides, tile, sparse_dim, padding_value="zero", _semantic=None):
    sparse_dim = tl._unwrap_if_constexpr(sparse_dim)
    if type(sparse_dim) is int:
        sparse_dim = (sparse_dim, )
    return _make_view(base, shape, strides, tile, "gather_scatter", (), sparse_dim, padding_value, _semantic)


def _indices(view, index, semantic):
    if not isinstance(view, tensor_view) or view.type.view_kind is None:
        raise TypeError("expected an encoded tla.tensor_view")
    if len(index) != view.rank:
        raise ValueError("index must contain one entry per view dimension")
    _builder(semantic.builder)
    values = []
    for dim, idx in enumerate(index):
        if dim in view.type.sparse_dims:
            idx = idx if isinstance(idx, buffer) else semantic.to_tensor(idx)
        else:
            idx = semantic.to_tensor(idx)
            if idx.type.is_block() or not idx.dtype.is_int():
                raise TypeError("dense view indices must be scalar integers")
        values.append(idx.handle)
    return values


@tl.builtin
def load(view, index, dst=None, _semantic=None):
    """Load a tile, optionally into an explicit UB buffer."""
    indices = _indices(view, index, _semantic)
    if dst is not None:
        if not isinstance(dst, buffer) or dst.space != "ub" or dst.shape != view.tile or dst.dtype != view.dtype:
            raise ValueError("dst must be a UB buffer matching the view tile and dtype")
        _semantic.builder.create_tla_view_load(view.handle, indices, dst.handle, dst.type.to_ir(_semantic.builder))
        return dst
    ty = tl.block_type(view.dtype, list(view.tile))
    handle = _semantic.builder.create_tla_view_load(view.handle, indices, None, ty.to_ir(_semantic.builder))
    return tl.tensor(handle, ty)


@tl.builtin
def store(view, value, index, _semantic=None):
    indices = _indices(view, index, _semantic)
    if isinstance(value, buffer):
        if value.space != "ub" or value.shape != view.tile or value.dtype != view.dtype:
            raise ValueError("value must be a UB buffer matching the view tile and dtype")
    else:
        value = _semantic.to_tensor(value)
        if not value.type.is_block():
            value = _semantic.broadcast_impl_shape(value, list(view.tile))
        elif tuple(value.type.get_block_shapes()) != view.tile:
            raise ValueError("value shape must match the view tile")
        value = _semantic.cast(value, view.dtype)
    _semantic.builder.create_tla_view_store(view.handle, value.handle, indices)
