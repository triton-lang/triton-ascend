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
"""Create buffer views: a GM pointer entry or an in-buffer tile.

``tla.view`` wraps a GM kernel pointer as a buffer for subsequent DMA/simd
ops: ``view(ptr, shape, strides, offsets)`` preserves the pointer descriptor
through ``tt.make_tensor_ptr``. Standard pointer lowering produces a
``memref.reinterpret_cast`` with the source offset plus
``sum(off_i*stride_i)``. ``strides`` are the final memref strides,
defaulting to row-major.

Slicing an existing buffer — in GM or any on-chip space — is :func:`subview`:
``subview(buf, offsets, sizes, strides)`` lowers to a ``memref.subview`` with
offsets and strides relative to the source view.
"""
from triton.language.core import builtin, tensor, int64, _unwrap_if_constexpr
from .core import buffer, buffer_type, _shape, _row_major_strides


def _numel(sizes):
    """Total element count of a static shape (positive ints)."""
    n = 1
    for size in sizes:
        n *= size
    return n


def _check_scalar_int(value):
    """Return a scalar-integer argument as an int or tensor; reject the rest."""
    value = _unwrap_if_constexpr(value)
    if isinstance(value, tensor):
        if value.shape or not value.dtype.is_int():
            raise TypeError("view sizes, strides and offsets must be scalar integers")
        return value
    if type(value) is not int:
        raise TypeError("view sizes, strides and offsets must be scalar integers")
    return value


def _check_dim(value, what):
    """Validate a size/stride entry: positive int or runtime scalar tensor."""
    value = _check_scalar_int(value)
    if type(value) is int and value <= 0:
        raise ValueError(f"view {what} must contain positive compile-time integers")
    return value


def _is_dynamic(value):
    return isinstance(value, tensor)


def _to_ir_dim(semantic, value):
    """A dim entry for the builder: runtime handles stay values, ints stay ints."""
    return semantic.to_tensor(value).handle if _is_dynamic(value) else value


def _runtime_row_major(sizes, semantic):
    """Row-major strides; dynamic sizes contribute runtime products."""
    strides = [1] * len(sizes)
    stride = 1
    for i in reversed(range(len(sizes))):
        strides[i] = stride
        if _is_dynamic(stride) or _is_dynamic(sizes[i]):
            stride = semantic.mul(stride, sizes[i], False)
        else:
            stride = stride * sizes[i]
    return strides


def _subview(src, sizes, steps, offsets, semantic):
    """Rank-preserving memref subview with mixed static/dynamic parameters."""
    # A single dynamic entry makes the view runtime-shaped, deferring the
    # bounds decision for the whole view to runtime; only fully static
    # slices are rejected up front.
    static_view = not any(_is_dynamic(v) for v in offsets + sizes + steps)
    offset_handles, size_dims, step_dims = [], [], []
    result_shape, result_strides = [], []
    for offset, size, step, extent, src_stride in zip(offsets, sizes, steps, src.shape, src.strides):
        offset = _check_scalar_int(offset)
        size = _check_dim(size, "sizes")
        step = _check_dim(step, "strides")
        if (static_view and extent is not None and (offset < 0 or offset + (size - 1) * step >= extent)):
            raise ValueError("static view is outside the source buffer")
        index = semantic.to_tensor(offset)
        if isinstance(index, tensor) and index.dtype.is_int_unsigned():
            # The native index cast widens signless integers as signed.
            index = semantic.cast(index, int64)
        offset_handles.append(index.handle)
        size_dims.append(_to_ir_dim(semantic, size))
        step_dims.append(_to_ir_dim(semantic, step))
        result_shape.append(None if _is_dynamic(size) else size)
        result_strides.append(None if (src_stride is None or _is_dynamic(step)) else src_stride * step)
    ty = buffer_type(tuple(result_shape), src.dtype, src.space, tuple(result_strides), offset=None)
    handle = semantic.builder.create_tla_subview(src.handle, offset_handles, size_dims, step_dims)
    return buffer(handle, ty)


def _entry(src, sizes, strides, offsets, semantic):
    """GM views preserve pointer provenance through tt.make_tensor_ptr."""
    if src.shape:
        raise TypeError("view expects a scalar pointer")
    if src.dtype.address_space != 1:
        raise TypeError("tla.view expects a GM pointer in address space 1")
    elem_ty = src.dtype.element_ty
    if not (elem_ty.is_int() or elem_ty.is_floating()) or elem_ty.primitive_bitwidth % 8:
        raise TypeError("GM view requires a byte-addressable integer or float element type")
    if strides is None:
        strides = _runtime_row_major(sizes, semantic)
    linear = 0
    for off, stride in zip(offsets, strides):
        off = _check_scalar_int(off)
        stride = _check_scalar_int(stride)
        # Widen before multiplication: a later index cast cannot undo i32 overflow.
        off = semantic.cast(semantic.to_tensor(off), int64)
        if _is_dynamic(stride):
            stride = semantic.cast(stride, int64)
        linear = semantic.add(linear, semantic.mul(off, stride, False), False)
    shape_ty = tuple(None if _is_dynamic(s) else s for s in sizes)
    strides_ty = tuple(None if _is_dynamic(s) else s for s in strides)
    ty = buffer_type(shape_ty, src.dtype.element_ty, "gm", strides_ty, offset=None)
    size_dims = [semantic.cast(semantic.to_tensor(s), int64).handle if _is_dynamic(s) else s for s in sizes]
    stride_dims = [semantic.cast(semantic.to_tensor(s), int64).handle if _is_dynamic(s) else s for s in strides]
    handle = semantic.builder.create_tla_pointer_view(src.handle,
                                                      semantic.to_tensor(linear).handle, ty.to_ir(semantic.builder),
                                                      size_dims, stride_dims)
    return buffer(handle, ty)


@builtin
def subview(src, offsets, sizes, strides=None, _semantic=None):
    """Alias a rank-preserving slice of a buffer without copying data.

    ``sizes``, ``strides`` and ``offsets`` each contain, per dimension, a
    positive compile-time integer or a scalar integer tensor (runtime value);
    strides default to one and multiply the source strides. Fully static
    out-of-bounds slices are rejected; a single dynamic entry defers the
    bounds decision for the whole view to runtime. The source dtype and
    memory space are preserved.

    This maps directly to ``memref.subview``. It preserves the source pitch:
    a compact prefix with a different pitch can instead be expressed by
    flattening, taking a one-dimensional subview, and reshaping that subview.
    """
    if not isinstance(src, buffer):
        raise TypeError("tla.subview expects a tla.buffer, "
                        f"got {type(src).__name__}")
    sizes = tuple(_check_dim(s, "sizes") for s in sizes)
    steps = (1, ) * len(sizes) if strides is None else tuple(_check_dim(s, "strides") for s in strides)
    offsets = tuple(_check_scalar_int(o) for o in offsets)
    if not (len(sizes) == len(src.shape) == len(steps) == len(offsets)):
        raise ValueError("subview sizes, offsets and strides must match source rank")
    return _subview(src, sizes, steps, offsets, _semantic)


@builtin
def view(src, shape, strides=None, offsets=None, _semantic=None):
    """Creates a GM buffer view from a kernel pointer.

    :param src: Scalar pointer tensor (kernel argument) in global memory.
    :type src: tl.tensor
    :param shape: Result extents in elements; each entry is a positive
        compile-time integer or a scalar integer tensor (runtime value).
    :type shape: tuple[int or tl.tensor, ...] or list[int or tl.tensor]
    :param strides: Element strides of the final memref. Each entry is a
        positive compile-time integer or a scalar integer tensor. Defaults to
        ``None``, which selects row-major strides (with runtime products for
        dynamic extents).
    :type strides: tuple[int or tl.tensor, ...] or None
    :param offsets: Element offsets relative to the pointer. Each entry is a
        positive compile-time integer or a scalar integer tensor. Defaults to
        ``None``, which selects zero offsets.
    :type offsets: tuple[int or tl.tensor, ...] or None
    :return: A GM buffer view aliasing the source memory.
    :rtype: tla.buffer
    :raises ValueError: If the rank of ``strides``/``offsets`` does not match
        the result shape.
    :raises TypeError: If ``src`` is not a scalar pointer tensor. Use
        :func:`subview` to slice an existing buffer in any space.

    The pointer descriptor is preserved through ``tt.make_tensor_ptr``
    carrying the linear offset and the full sizes/strides, so arbitrary
    strides and offsets are supported. Standard pointer lowering then
    produces a single ``memref.reinterpret_cast``.

    .. note::
        Static extents must be positive; dynamic (tensor) extents, strides and
        offsets are checked for bounds at runtime. GM buffers must be created
        from kernel pointers with this interface rather than :func:`alloc`.
    """
    if isinstance(src, buffer):
        raise TypeError("tla.view expects a GM pointer; "
                        "use tla.subview to slice an existing buffer")
    if isinstance(src, tensor) and src.dtype.is_ptr():
        sizes = tuple(_check_dim(s, "shape") for s in shape)
        offsets = (0, ) * len(sizes) if offsets is None else tuple(_check_scalar_int(o) for o in offsets)
        if strides is not None:
            strides = tuple(_check_dim(s, "strides") for s in strides)
        if (strides is not None and len(strides) != len(sizes)) or len(offsets) != len(sizes):
            raise ValueError("view strides and offsets must match buffer rank")
        return _entry(src, sizes, strides, offsets, _semantic)
    raise TypeError("tla.view expects a scalar pointer tensor, "
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
    if any(dim is None or stride is None for dim, stride in zip(src.shape, src.strides)):
        raise ValueError("tla.reshape requires a statically shaped buffer; "
                         "use tla.view/subview for dynamic tiles")
    sizes = _shape(shape)
    numel = _numel(sizes)
    if numel != _numel(src.shape):
        raise ValueError(f"tla.reshape element count mismatch: "
                         f"{src.shape} -> {sizes}")
    if src.strides != _row_major_strides(src.shape):
        raise ValueError("tla.reshape requires a contiguous row-major buffer; "
                         "use tla.subview for strided tiles")
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
