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
"""In-place scalar fill for contiguous UB buffers.

The first version is deliberately narrow: it broadcasts one scalar into a UB
buffer through ``hivm.hir.vbrc`` and is only enabled for element types and
layouts that have been reviewed against the backend.  Unsupported spaces are
rejected with the portable alternative instead of being silently lowered:

* GM  -> ``tla.fill`` a UB buffer, then ``tla.store_ub_to_gm``.
* L0C -> ``tla.cube.mmad(..., clear_accum=True)``.
* L1/L0A/L0B/BT/FB -> not part of this version.

Non-zero or dynamic subview offsets are also rejected: ``tla.subview`` records
its result as ``offset=None`` even for a static slice, so ``tla.fill`` cannot
tell a zero-offset alias from a shifted one.  Filling in that state could
target the allocation origin instead of the requested region.  These cases
need the offset carried through the buffer type and a lowering review before
they can be enabled.
"""
from triton.language.core import builtin, tensor, _unwrap_if_constexpr
from .core import buffer, _row_major_strides

# Element types reviewed for the UB scalar fill.  i1 is excluded on purpose:
# its 16-element tail alignment still needs dedicated device evidence.
_FILL_DTYPES = frozenset({"fp32", "bf16", "int32"})


def _check_dst(dst):
    """Reject non-UB, non-contiguous or offset buffers before emitting IR."""
    if not isinstance(dst, buffer):
        raise TypeError(f"tla.fill expects a tla.buffer destination, got {type(dst).__name__}")
    if dst.space != "ub":
        raise ValueError(f"tla.fill supports only UB buffers, got space {dst.space!r}; "
                         "for GM use tla.fill(ub, value) followed by a UB->GM store, "
                         "and for L0C use tla.cube.mmad(..., clear_accum=True)")
    if dst.dtype.is_bool():
        raise ValueError("tla.fill does not support i1 buffers yet: the 16-element tail "
                         "alignment still needs device validation")
    if dst.dtype.name not in _FILL_DTYPES:
        supported = ", ".join(sorted(_FILL_DTYPES))
        raise ValueError(f"tla.fill does not support element type {dst.dtype}; supported: {supported}")
    if dst.strides != _row_major_strides(dst.shape) or dst.offset != 0:
        raise ValueError("tla.fill requires a contiguous row-major buffer at offset 0 "
                         "(the direct tla.alloc result or a legal reshape of it); "
                         "buffers carrying a subview offset or pitch are not supported yet")


def _check_value(value, dtype):
    """Reject scalar values that would lose information or change category."""
    value = _unwrap_if_constexpr(value)
    if isinstance(value, bool):
        raise TypeError("tla.fill value must be a number, got bool")
    if isinstance(value, int):
        if dtype.is_int():
            bits = dtype.primitive_bitwidth
            if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
                raise ValueError(f"tla.fill value {value} does not fit {dtype}")
        return value
    if isinstance(value, float):
        if dtype.is_int():
            raise TypeError(f"tla.fill value {value!r} is a float but the destination "
                            f"element type is {dtype}; pass an int instead")
        return value
    if isinstance(value, tensor):
        if value.type.is_block():
            raise TypeError(f"tla.fill value must be a scalar, got {value.type}")
        if value.dtype.is_floating() and dtype.is_int():
            raise TypeError(f"tla.fill value has floating type {value.dtype} but the "
                            f"destination element type is {dtype}")
        return value
    raise TypeError(f"tla.fill value must be a scalar number or a scalar tensor, "
                    f"got {type(value).__name__}")


@builtin
def fill(dst, value, _semantic=None):
    """Fills ``dst`` in place with one scalar value.

    Emits a single ``hivm.hir.vbrc`` with a scalar input; the scalar keeps an
    empty ``broadcast_dims`` attribute.  ``value`` is converted to the
    destination element type by the normal Triton scalar cast, and the buffer
    is not reallocated or copied.

    :param dst: UB destination buffer with a contiguous row-major layout and
        offset 0, as produced by :func:`tla.alloc` or a legal
        :func:`tla.reshape` of it.
    :type dst: tla.buffer
    :param value: Compile-time scalar or scalar tensor, converted to
        ``dst.dtype``.
    :type value: int, float or scalar tensor
    :return: None. ``dst`` is updated in place.
    :rtype: None
    :raises ValueError: if ``dst`` is not a supported UB buffer or ``value``
        does not fit the destination element type.
    :raises TypeError: if ``dst`` is not a buffer or ``value`` is not scalar.
    """
    _check_dst(dst)
    value = _check_value(value, dst.dtype)
    scalar = _semantic.to_tensor(value)
    scalar = _semantic.cast(scalar, dst.dtype)
    _semantic.builder.create_tla_fill(dst.handle, scalar.handle)
