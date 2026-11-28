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

- GM pointer entry: ``view(ptr, shape, strides, offsets, space="gm")`` preserves
  the pointer descriptor through ``tt.make_tensor_ptr``. Standard pointer
  lowering produces a ``memref.reinterpret_cast`` with the source offset plus
  ``sum(off_i*stride_i)``.
  ``strides`` are the final memref strides, defaulting to row-major.
- Buffer tile: ``subview(buf, offsets, sizes, strides)`` or the compatible
  ``view(buf, sizes, strides, offsets)``. Lowers to a ``memref.subview``;
  offsets and strides are relative to the source view.

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
    """GM views preserve pointer provenance; other spaces use address binding."""
    if src.shape:
        raise TypeError("view expects a scalar pointer")
    if space == "gm":
        elem_ty = src.dtype.element_ty
        if not (elem_ty.is_int() or elem_ty.is_floating()) or elem_ty.primitive_bitwidth % 8:
            raise TypeError("GM view requires a byte-addressable integer or float element type")
    linear = 0
    for off, stride in zip(offsets, strides):
        off = _check_scalar_int(off)
        # Widen before multiplication: a later index cast cannot undo i32 overflow.
        if space == "gm":
            off = semantic.cast(semantic.to_tensor(off), int64)
        linear = semantic.add(linear, semantic.mul(off, stride, False), False)
    ty = buffer_type(sizes, src.dtype.element_ty, space, strides, offset=None)
    if space == "gm":
        handle = semantic.builder.create_tla_pointer_view(src.handle,
                                                          semantic.to_tensor(linear).handle, ty.to_ir(semantic.builder))
        return buffer(handle, ty)
    handle = semantic.builder.create_tla_reinterpret_cast(
        _bind(src, buffer_type((1, ), src.dtype.element_ty, space), semantic).handle,
        semantic.to_tensor(linear).handle, list(sizes), list(strides), ty.to_ir(semantic.builder))
    return buffer(handle, ty)


def _bind(src, ty, semantic):
    """Pointer memory as a memref via one hivm.hir.pointer_cast."""
    handle = semantic.builder.create_tla_bind(semantic.cast(src, int64).handle, ty.to_ir(semantic.builder))
    return buffer(handle, ty)


@builtin
def subview(src, offsets, sizes, strides=None, _semantic=None):
    """Alias a rank-preserving slice of a buffer without copying data.

    ``sizes`` and ``strides`` contain positive compile-time integers; strides
    default to one and multiply the source strides. ``offsets`` may contain
    scalar integers or integer tensors, relative to the source view. Static
    out-of-bounds slices are rejected; dynamic offsets must be in bounds at
    runtime. The source dtype and memory space are preserved.

    This maps directly to ``memref.subview``. It preserves the source pitch:
    a compact prefix with a different pitch can instead be expressed by
    flattening, taking a one-dimensional subview, and reshaping that subview.
    """
    if not isinstance(src, buffer):
        raise TypeError("tla.subview expects a tla.buffer, "
                        f"got {type(src).__name__}")
    sizes = _shape(sizes)
    steps = (1, ) * len(sizes) if strides is None else _shape(strides)
    offsets = tuple(offsets)
    if not (len(sizes) == len(src.shape) == len(steps) == len(offsets)):
        raise ValueError("subview sizes, offsets and strides must match source rank")
    return _subview(src, sizes, steps, offsets, _semantic)


@builtin
def view(src, shape, strides=None, offsets=None, space="gm", _semantic=None):
    """Creates a buffer view from a kernel pointer or an existing buffer.

    :param src: Pointer tensor (kernel argument) or ``tla.buffer`` source.
    :type src: tl.tensor or tla.buffer
    :param shape: Result extents in elements.
    :type shape: tuple[int, ...] or list[int]
    :param strides: Element strides. For a pointer source these are the final
        memref strides; for a buffer source they are relative to the source
        view. Defaults to ``None``, which selects row-major strides for a
        pointer source and unit steps for a buffer source.
    :type strides: tuple[int, ...] or None
    :param offsets: Element offsets. For a pointer source these are relative to
        the pointer; for a buffer source they are relative to the source view.
        Defaults to ``None``, which selects zero offsets.
    :type offsets: tuple[int or tl.tensor, ...] or None
    :param space: Address space for a pointer source. Defaults to ``"gm"``.
        Ignored for a buffer source, whose space is preserved; a non-default
        value raises ``ValueError`` in that case.
    :type space: str
    :return: A buffer view aliasing the source memory.
    :rtype: tla.buffer
    :raises ValueError: If the rank of ``strides``/``offsets`` does not match
        the result shape, or a static offset is outside the source buffer.
    :raises TypeError: If ``src`` is neither a pointer tensor nor a
        ``tla.buffer``.

    A pointer source lowers to a one-dimensional ``hivm.hir.pointer_cast`` plus
    a single ``memref.reinterpret_cast`` carrying the linear offset and the
    full sizes/strides, so arbitrary strides and offsets are supported. A
    buffer source lowers to ``memref.subview`` and preserves the source pitch;
    use :func:`reshape` for a contiguous rank change.

    .. note::
        Only static positive extents are accepted. GM buffers must be created
        from kernel pointers with this interface rather than :func:`alloc`.
    """
    if isinstance(src, buffer):
        if space != "gm":
            raise ValueError("view space is ignored for a buffer source")
        offsets = (0, ) * len(src.shape) if offsets is None else offsets
        return subview(src, offsets, shape, strides, _semantic=_semantic)
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
    flatten/unflatten emits a single op. A contiguous subview is supported,
    including a nonzero or dynamic offset: the offset is preserved in the
    result type and is never reset to the allocation origin. Non-contiguous
    slices are rejected. The element count is preserved exactly.
    """
    if not isinstance(src, buffer):
        raise TypeError("tla.reshape expects a tla.buffer, "
                        f"got {type(src).__name__}")
    sizes = _shape(shape)
    numel = _numel(sizes)
    if numel != _numel(src.shape):
        raise ValueError(f"tla.reshape element count mismatch: "
                         f"{src.shape} -> {sizes}")
    if src.strides != _row_major_strides(src.shape):
        raise ValueError("tla.reshape requires a contiguous row-major buffer; "
                         "use tla.view for strided tiles")
    if sizes == src.shape:
        return src  # no-op reshape
    b = _semantic.builder
    ty = buffer_type(sizes, src.dtype, src.space, offset=src.offset)
    dst_ty = ty.to_ir(b)
    handle = src.handle
    if len(src.shape) > 1:  # flatten to 1-D
        flat_ty = buffer_type((numel, ), src.dtype, src.space, offset=src.offset).to_ir(b)
        handle = b.create_tla_collapse_shape(handle, flat_ty, [list(range(len(src.shape)))])
    if len(sizes) > 1:  # unflatten from 1-D to the target shape
        handle = b.create_tla_expand_shape(handle, dst_ty, [list(range(len(sizes)))])
    return buffer(handle, ty)
