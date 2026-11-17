"""Structured GM/UB transfers through the HIVM SIMT stride operations."""
from triton.language.core import builtin, tensor, tuple as tl_tuple, int32, int64, _unwrap_if_constexpr

from .core import buffer


_ELEMENT_TYPES = frozenset({
    "fp16", "bf16", "fp32", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
})


def _check_buffers(src, dst, *, load):
    name = "stride_load" if load else "stride_store"
    if not isinstance(src, buffer) or not isinstance(dst, buffer):
        raise TypeError(f"tla.{name} expects two tla.buffer operands")
    gm, tile = (src, dst) if load else (dst, src)
    if gm.space != "gm" or tile.space != "ub":
        route = "GM->UB" if load else "UB->GM"
        raise ValueError(f"tla.{name} requires {route} buffers")
    if src.dtype != dst.dtype:
        raise TypeError(f"tla.{name} requires matching source and destination element types")
    if src.dtype.name not in _ELEMENT_TYPES:
        raise TypeError(f"tla.{name} does not support element type {src.dtype}")
    if not 1 <= len(tile.shape) <= 3:
        raise ValueError(f"tla.{name} requires a UB tile of rank 1, 2 or 3")
    elements = 1
    for size in tile.shape:
        elements *= size
    span = 1 + sum((size - 1) * stride for size, stride in zip(tile.shape, tile.strides))
    if elements >= 2**31 or span >= 2**31 or any(stride >= 2**31 for stride in tile.strides):
        raise ValueError(f"tla.{name} requires UB sizes, strides and index arithmetic to fit signed i32")
    return tile


def _integer(value, name):
    value = _unwrap_if_constexpr(value)
    if isinstance(value, tensor):
        if value.shape or not value.dtype.is_int() or value.dtype.is_bool():
            raise TypeError(f"{name} must be a scalar integer")
    elif type(value) is not int:
        raise TypeError(f"{name} must be a scalar integer")
    elif not -(2**31) <= value < 2**31:
        raise ValueError(f"{name} must fit signed i32 in the current NPUIR stride templates")
    return value


def _sequence(values, name, rank):
    values = _unwrap_if_constexpr(values)
    if not isinstance(values, (tuple, list, tl_tuple)):
        raise TypeError(f"{name} must be a tuple or list of scalar integers")
    if len(values) != rank:
        raise ValueError(f"{name} must contain {rank} entries, one per UB tile dimension")
    return tuple(_integer(value, f"{name}[{i}]") for i, value in enumerate(values))


def _indices(offset, strides, numels, shape, semantic):
    offset = _integer(offset, "offset")
    strides = _sequence(strides, "strides", len(shape))
    numels = _sequence(numels, "numels", len(shape))
    for i, (stride, size) in enumerate(zip(strides, shape)):
        if type(stride) is int and (stride < 0 or (stride == 0 and size != 1)):
            raise ValueError(f"strides[{i}] must be positive, or zero for a singleton UB dimension")
    values = [semantic.to_tensor(value) for value in (offset, *strides, *numels)]
    # HIVM requires a single i32/i64 type for all indices. Widen unsigned i32
    # and 64-bit operands rather than truncating runtime expressions to i32.
    index_type = int64 if any(value.dtype.primitive_bitwidth > 32 or value.dtype.is_uint32()
                             for value in values) else int32
    handles = [semantic.cast(value, index_type).handle for value in values]
    rank = len(shape)
    return handles[0], handles[1:1 + rank], handles[1 + rank:]


def _padding(other, dtype, semantic):
    other = _unwrap_if_constexpr(other)
    if not isinstance(other, (int, float, tensor)):
        raise TypeError("other must be a scalar number or scalar tensor")
    other = semantic.to_tensor(other)
    if other.shape or not (other.dtype.is_int() or other.dtype.is_floating()):
        raise TypeError("other must be a scalar number or scalar tensor")
    return semantic.cast(other, dtype).handle


@builtin
def stride_load(src, dst, offset, other, strides, numels, _semantic=None):
    """Loads a strided GM region into a rank 1--3 UB tile, padding its tail.

    For every logical index ``i`` in ``dst.shape``, load from
    ``src_origin + offset + sum(i[d] * strides[d])`` if all
    ``i[d] < numels[d]``; otherwise write ``other``. Addresses and strides
    are in elements. ``src_origin`` includes the source view's descriptor
    offset. The explicit strides describe GM addressing; they are not
    multiplied by ``src.strides``. The destination uses its own layout.

    :param src: GM buffer. Its rank and shape need not match the UB tile.
    :type src: tla.buffer
    :param dst: UB buffer of rank 1, 2 or 3, with the same dtype as ``src``.
        Strided and offset UB views are supported.
    :type dst: tla.buffer
    :param offset: Linear element offset relative to the GM view origin.
    :type offset: int or scalar integer tensor
    :param other: Padding value, cast to ``dst.dtype``.
    :type other: int, float or scalar tensor
    :param strides: One physical GM element stride per UB dimension.
        Entries may be runtime scalar integers.
    :type strides: tuple of scalar integers
    :param numels: One valid element count per UB dimension, not repeat
        counts. Counts are clipped to the tile size; nonpositive counts
        produce an empty valid region. Entries may be runtime scalars.
    :type numels: tuple of scalar integers
    :return: None. Updates ``dst`` in place.
    :rtype: None

    Emits ``hivm.hir.stride_load`` in SIMT template mode. Use on AIV,
    outside ``tla.scope.simd()`` (for example in ``tla.scope.vector()``).
    Requires the NPUIR register-based SIMT templates (Ascend 950/960).
    Supported element types are f16, bf16, f32 and signed/unsigned
    8/16/32/64-bit integers.

    The current templates require positive strides, except that a singleton
    UB dimension may have stride zero. All index values, UB sizes/strides,
    and intermediate index arithmetic must fit signed i32, even when the
    IR operands are i64. Runtime values must satisfy these preconditions.
    ``numels`` masks logical indices; it does not check GM allocation bounds.
    Every unmasked address must lie in the underlying allocation.

    Example inside a ``@tla.jit`` kernel::

        tile = tla.alloc((4, 16), tla.float32)
        tla.stride_load(src, tile, offset, 0.0,
                        strides=(row_stride, 1), numels=(rows, cols))
    """
    tile = _check_buffers(src, dst, load=True)
    offset, strides, numels = _indices(offset, strides, numels, tile.shape, _semantic)
    other = _padding(other, dst.dtype, _semantic)
    _semantic.builder.create_tla_stride_load(src.handle, dst.handle, offset, other, strides, numels)


@builtin
def stride_store(src, dst, offset, strides, numels, _semantic=None):
    """Stores a rank 1--3 UB tile into a strided GM region.

    For every logical index ``i`` in ``src.shape`` with all
    ``i[d] < numels[d]``, write ``src[i]`` to
    ``dst_origin + offset + sum(i[d] * strides[d])``. Masked positions
    leave GM unchanged. Offsets and strides are in elements, relative to
    the destination view origin, including its descriptor offset. Explicit
    strides are not multiplied by ``dst.strides``; the UB source uses its
    own layout.

    :param src: UB tile of rank 1, 2 or 3. Strided and offset views are
        supported.
    :type src: tla.buffer
    :param dst: GM buffer with the same dtype as ``src``. Its rank and shape
        need not match the UB tile.
    :type dst: tla.buffer
    :param offset: Linear element offset relative to the GM view origin.
    :type offset: int or scalar integer tensor
    :param strides: One physical GM element stride per UB dimension.
    :type strides: tuple of scalar integers
    :param numels: One valid element count per UB dimension, not repeat
        counts. Counts are clipped to the tile size; nonpositive counts
        suppress all stores. Entries may be runtime scalar integers.
    :type numels: tuple of scalar integers
    :return: None. Updates ``dst`` in place.
    :rtype: None

    Emits ``hivm.hir.stride_store`` in SIMT template mode. The AIV scope,
    target, dtype, stride and signed-i32 arithmetic requirements are the
    same as :func:`stride_load`. Use outside ``tla.scope.simd()``.
    ``offset`` and all entries of ``strides``/``numels`` may be runtime
    scalar integers.
    All unmasked GM addresses must be in bounds and distinct; overlapping
    writes have no ordering guarantee. There is no atomic or post-update
    behavior.

    Example inside a ``@tla.jit`` kernel::

        tla.stride_store(tile, dst, offset,
                         strides=(row_stride, 1), numels=(rows, cols))
    """
    tile = _check_buffers(src, dst, load=False)
    offset, strides, numels = _indices(offset, strides, numels, tile.shape, _semantic)
    _semantic.builder.create_tla_stride_store(src.handle, dst.handle, offset, strides, numels)
