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
"""Create buffer views: a kernel pointer entry or an in-buffer tile.

``tla.view`` aliases memory as a buffer for subsequent DMA/simd ops:

- Pointer entry: ``view(ptr, shape, strides, offsets, space="gm")``.  Lowers to
  ``ptr_to_int`` + a 1-D ``hivm.hir.pointer_cast`` + one
  ``memref.reinterpret_cast`` carrying the linear offset ``sum(off_i*stride_i)``
  and the full sizes/strides -- npuir's form, valid for any stride/offset.
  ``strides`` are the final memref strides, defaulting to row-major.
- Buffer tile: ``view(buf, sizes, strides, offsets)``.  Lowers to a
  ``memref.subview``; strides are relative to the source view.

``space`` only applies to a pointer entry; a buffer's space is its own.
"""
from triton.language.core import builtin, tensor, int64, _unwrap_if_constexpr
from .core import buffer, buffer_type, _shape, _row_major_strides


def _numel(sizes):
    """Total element count of a static shape (positive ints)."""
    n = 1
    for size in sizes:
        n *= size
    return n


def _check_scalar_int(offset):
    """Return a scalar-integer offset as an int or tensor; reject everything else."""
    offset = _unwrap_if_constexpr(offset)
    if isinstance(offset, tensor):
        if offset.shape or not offset.dtype.is_int():
            raise TypeError("view offsets must be scalar integers")
        return offset
    if type(offset) is not int:
        raise TypeError("view offsets must be scalar integers")
    return offset


def _subview(src, sizes, steps, offsets, semantic):
    """Rank-preserving memref subview with dynamic offsets and static sizes."""
    handles = []
    for offset, size, step, extent in zip(offsets, sizes, steps, src.shape):
        offset = _check_scalar_int(offset)
        if not isinstance(offset, tensor) and (offset < 0 or offset + (size - 1) * step >= extent):
            raise ValueError("static view is outside the source buffer")
        handles.append(semantic.to_tensor(offset).handle)
    ty = buffer_type(sizes, src.dtype, src.space, tuple(a * b for a, b in zip(src.strides, steps)), offset=None)
    handle = semantic.builder.create_tla_subview(src.handle, handles, list(sizes), list(steps))
    return buffer(handle, ty)


def _entry(src, sizes, strides, offsets, space, semantic):
    """Pointer view: 1-D pointer_cast + one memref.reinterpret_cast.

    Lowers to::

      %cast = hivm.hir.pointer_cast %ptr : memref<1xelem, space>
      %view = memref.reinterpret_cast %cast to
        offset: [sum(off_i * stride_i)], sizes: [sizes], strides: [strides]
        : memref<1xelem, space> to memref<shape, strided<strides, offset: ?>, space>

    The 1-D entry is identity (PointerCastOp's LLVM lowering requires it);
    the strided final view is npuir's form, valid for any stride/offset.
    """
    linear = 0
    for off, stride in zip(offsets, strides):
        off = _check_scalar_int(off)
        linear = semantic.add(linear, semantic.mul(off, stride, False), False)
    ty = buffer_type(sizes, src.dtype.element_ty, space, strides, offset=None)
    handle = semantic.builder.create_tla_reinterpret_cast(
        _bind(src, buffer_type((1, ), src.dtype.element_ty, space), semantic).handle,
        semantic.to_tensor(linear).handle, list(sizes), list(strides), ty.to_ir(semantic.builder))
    return buffer(handle, ty)


def _bind(src, ty, semantic):
    """Pointer memory as a memref via one hivm.hir.pointer_cast."""
    handle = semantic.builder.create_tla_bind(semantic.cast(src, int64).handle, ty.to_ir(semantic.builder))
    return buffer(handle, ty)


@builtin
def view(src, shape, strides=None, offsets=None, space="gm", _semantic=None):
    if isinstance(src, buffer):
        if space != "gm":
            raise ValueError("view space is ignored for a buffer source")
        sizes = _shape(shape)
        steps = (1, ) * len(sizes) if strides is None else _shape(strides)
        offsets = (0, ) * len(sizes) if offsets is None else tuple(offsets)
        if not (len(sizes) == len(src.shape) == len(steps) == len(offsets)):
            raise ValueError("view sizes, offsets and strides must match source rank")
        return _subview(src, sizes, steps, offsets, _semantic)
    if isinstance(src, tensor) and src.dtype.is_ptr():
        sizes = _shape(shape)
        strides = _row_major_strides(sizes) if strides is None else _shape(strides)
        offsets = (0, ) * len(sizes) if offsets is None else tuple(offsets)
        if len(strides) != len(sizes) or len(offsets) != len(sizes):
            raise ValueError("view strides and offsets must match buffer rank")
        return _entry(src, sizes, strides, offsets, space, _semantic)
    raise TypeError("tla.view expects a tla.buffer or a pointer tensor, "
                    f"got {type(src).__name__}")


@builtin
def reshape(src, shape, _semantic=None):
    """Rank-only reshape of a contiguous row-major buffer.

    Lowers to static ``memref.collapse_shape`` / ``memref.expand_shape`` ops
    with explicit reassociation indices (mirrors npuir's
    CanonicalizeTensorReshape): first flatten the source to 1-D when its rank
    is > 1, then unflatten to the target shape when its rank is > 1, so a pure
    flatten/unflatten emits a single op.  The source must be an
    identity-layout buffer (alloc or reshape result), so strided tiles need
    ``tla.view`` first and are rejected here.  The element count is preserved
    exactly.
    """
    if not isinstance(src, buffer):
        raise TypeError("tla.reshape expects a tla.buffer, "
                        f"got {type(src).__name__}")
    sizes = _shape(shape)
    numel = _numel(sizes)
    if numel != _numel(src.shape):
        raise ValueError(f"tla.reshape element count mismatch: "
                         f"{src.shape} -> {sizes}")
    if src.strides != _row_major_strides(src.shape) or src.offset != 0:
        raise ValueError("tla.reshape requires a contiguous row-major buffer; "
                         "use tla.view for strided tiles")
    if sizes == src.shape:
        return src  # no-op reshape
    b = _semantic.builder
    ty = buffer_type(sizes, src.dtype, src.space)
    dst_ty = ty.to_ir(b)
    handle = src.handle
    if len(src.shape) > 1:  # flatten to 1-D
        flat_ty = buffer_type((numel, ), src.dtype, src.space).to_ir(b)
        handle = b.create_tla_collapse_shape(handle, flat_ty, [list(range(len(src.shape)))])
    if len(sizes) > 1:  # unflatten from 1-D to the target shape
        handle = b.create_tla_expand_shape(handle, dst_ty, [list(range(len(sizes)))])
    return buffer(handle, ty)
