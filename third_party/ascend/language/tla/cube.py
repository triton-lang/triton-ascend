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
"""Cube-core primitives for TLA (hivm.hir.mmadL1 / hivm.hir.mmadmxL1).

``mmad`` computes ``C = A x B (+C)`` with A=(M,K)/B=(K,N) in L1 and C in L0C.
Inputs may use logical 2-D shapes or physical 4-D NZ shapes. Transposition
is applied to the logical matrix dimensions represented by each L1 buffer.

``mmad_mx`` is the Ascend950PR microscaling variant: A/B are FP8/FP4 in L1,
per-32-element E8M0 scale factors live in L1, and C is an L0C accumulator.
"""
from enum import Enum

from triton.language.core import builtin, float8e4nv, float8e5, int8, uint8, _unwrap_if_constexpr
from .core import buffer
from .dma import _unit_flag_args

__all__ = ["MxFormat", "mmad", "mmad_mx"]


class MxFormat(Enum):
    """Microscaling element format stored by an ``mmad_mx`` operand.

    ``E5M2`` and ``E4M3`` are the FP8 formats; ``E2M1`` selects packed FP4,
    whose L1 buffer holds two logical K values per byte.
    """
    E5M2 = "e5m2"
    E4M3 = "e4m3"
    E2M1 = "e2m1"


_MX_FORMAT_CODES = {MxFormat.E5M2: 1, MxFormat.E4M3: 2, MxFormat.E2M1: 3}
_MX_SCALE_GROUP = 32


def _inferred_mx_format(dtype):
    if dtype == float8e5:
        return MxFormat.E5M2
    if dtype == float8e4nv:
        return MxFormat.E4M3
    return None


def _mx_format(format, dtype, name):
    """Resolve an operand's MxFormat and its native ``lhsFormat``/``rhsFormat``.

    The native attribute is only meaningful for packed integer operands; FP8
    element types already encode the format in the library-call name.
    """
    format = _unwrap_if_constexpr(format)
    if format is not None and not isinstance(format, MxFormat):
        raise TypeError(f"tla.mmad_mx format_{name} must be a tla.MxFormat or None")
    inferred = _inferred_mx_format(dtype)
    if format is None:
        if inferred is None:
            raise ValueError(f"tla.mmad_mx format_{name} is required for integer operands")
        return inferred, None
    if inferred is not None:
        if inferred is not format:
            raise ValueError(f"tla.mmad_mx format_{name}={format.value} does not match operand dtype")
        return format, None
    return format, _MX_FORMAT_CODES[format]


def _mx_k_factor(format):
    """Logical K values represented by one stored element."""
    return 2 if format is MxFormat.E2M1 else 1


def _scale_logical_shape(shape):
    """Logical (rows, K//32) of a 2-D ND or 4-D zZ/nN E8M0 scale buffer.

    The 4-D fractal form is ``(rows // 16, (K32 + 1) // 2, 16, 2)``: 16 rows by
    2 scale bytes per tile, padded on the K-group axis when ``K32`` is odd.
    """
    if len(shape) == 2:
        return shape
    if len(shape) == 4:
        return (shape[0] * shape[2], shape[1] * shape[3])
    raise ValueError("tla.mmad_mx scale buffers must be 2-D ND or 4-D zZ/nN")


def _logical_shape(shape):
    """Logical (rows, cols) of a 2-D ND or 4-D zN operand.

    A 4-D zN shape (cols/b1, rows/b0, b0, b1) maps to (rows=b0*shape[1],
    cols=b1*shape[0]), the same convention npuir derives from a 4-D operand's
    trailing two dims; 2-D ND shapes pass through unchanged.
    """
    if len(shape) == 2:
        return shape
    if len(shape) == 4:
        return (shape[1] * shape[2], shape[0] * shape[3])
    raise ValueError("tla.mmad buffers must be 2-D ND or 4-D zN")


@builtin
def mmad(a, b, acc, transpose_a=False, transpose_b=False, clear_accum=False, unit_flag_ctrl=None, *,
         unit_flag_group_id=None, real_m=None, real_k=None, real_n=None, _semantic=None):
    """Multiplies L1 matrices into L0C using ``hivm.hir.mmadL1``.

    Computes ``acc = a @ b`` when ``clear_accum=True`` and adds the product to
    the existing accumulator otherwise.

    :param a: L1 left operand with logical shape ``(M, K)``, or ``(K, M)``
        when ``transpose_a=True``.
    :type a: tla.buffer
    :param b: L1 right operand with logical shape ``(K, N)``, or ``(N, K)``
        when ``transpose_b=True``, and the same element type as ``a``.
    :type b: tla.buffer
    :param acc: L0C accumulator with logical shape ``(real_m, real_n)``. Its
        element type must be a backend-supported accumulator type for ``a``
        and ``b``.
    :type acc: tla.buffer
    :param transpose_a: Compile-time native A-transpose flag. Defaults to
        ``False``. The transpose is carried by MMAD rather than a separate
        data movement, and the frontend swaps the two logical dimensions of
        ``a`` when validating the product.
    :type transpose_a: bool
    :param transpose_b: Compile-time native B-transpose flag. Defaults to
        ``False``. Like ``transpose_a`` it is carried by MMAD, and the
        frontend swaps the two logical dimensions of ``b``.
    :type transpose_b: bool
    :param clear_accum: Whether to initialize the accumulator before this
        multiplication. Defaults to ``False``; in that case ``acc`` must
        already contain valid values.
    :type clear_accum: bool
    :param unit_flag_ctrl: Compile-time UnitFlag mode, or ``None`` to leave
        the attribute unset (default). ``DISABLED`` (0),
        ``ENABLED_WITHOUT_UPDATE`` (2) and ``ENABLED_WITH_UPDATE`` (3) select
        explicit modes. Value 1 is reserved.
    :type unit_flag_ctrl: tla.UnitFlagMode, int or None
    :param unit_flag_group_id: Keyword-only compile-time signed 64-bit group
        ID, or ``None`` to leave it unset. Match the related Fixpipe protocol;
        valid hardware IDs are backend-dependent.
    :type unit_flag_group_id: int or None
    :param real_m: Actual M extent to compute, or ``None`` (default) for the
        full ``M``. Must be a positive compile-time multiple of 16 that does
        not exceed ``M``.
    :type real_m: int or None
    :param real_k: Actual K extent to compute, or ``None`` (default) for the
        full ``K``. Must be a positive compile-time multiple of 16 that does
        not exceed ``K``.
    :type real_k: int or None
    :param real_n: Actual N extent to compute, or ``None`` (default) for the
        full ``N``. Must be a positive compile-time multiple of 16 that does
        not exceed ``N``.
    :type real_n: int or None
    :return: None. The result is stored in ``acc``.
    :rtype: None

    .. note::
        Use inside ``tla.scope.cube()``. Each operand may independently use
        a logical 2-D shape or a physical 4-D NZ shape. A 4-D buffer has logical
        shape ``(shape[1] * shape[2], shape[0] * shape[3])``. After the
        transposes are applied the dimensions must satisfy ``a=(M,K)``,
        ``b=(K,N)`` and ``acc=(real_m, real_n)``. Explicit ``real_m``,
        ``real_k`` and ``real_n`` restrict the multiplication to a leading
        submatrix, and ``acc`` is a packed L0C accumulator that must match
        those actual extents exactly. Dtypes, block sizes and alignment are
        checked by the backend.

        To preserve a manually configured UnitFlag protocol, launch with
        ``unit_flag=False`` to disable automatic UnitFlag inference. Leaving
        ``unit_flag_ctrl=None`` does not explicitly disable the protocol.
    """
    if not (isinstance(a, buffer) and isinstance(b, buffer) and isinstance(acc, buffer)):
        raise TypeError("tla.mmad expects tla.buffer operands for a, b, acc")
    if a.space != "l1" or b.space != "l1":
        raise ValueError("tla.mmad a/b must be L1 buffers (fill via tla.copy(gm->l1))")
    if acc.space != "l0c":
        raise ValueError("tla.mmad acc must be an L0C buffer")
    if a.dtype != b.dtype:
        raise ValueError("tla.mmad a and b must have the same element type")
    # Each operand is checked independently: npuir accepts 2-D ND and 4-D NZ
    # operands in any mix, and only the logical (M, K) / (K, N) / (M, N)
    # dims must agree.
    m, k = _logical_shape(a.shape)
    k2, n = _logical_shape(b.shape)
    transpose_a = bool(_unwrap_if_constexpr(transpose_a))
    transpose_b = bool(_unwrap_if_constexpr(transpose_b))
    if transpose_a:
        m, k = k, m
    if transpose_b:
        k2, n = n, k2
    if k != k2:
        raise ValueError("tla.mmad reduction dimensions must match after transposition")
    actual = []
    for name, bound, extent in zip(("real_m", "real_k", "real_n"), (m, k, n), (real_m, real_k, real_n)):
        extent = _unwrap_if_constexpr(extent)
        if extent is None:
            extent = bound
        elif type(extent) is not int or extent <= 0 or extent > bound or extent % 16:
            raise ValueError(f"tla.mmad {name} must be a positive multiple of 16 within {bound}")
        actual.append(extent)
    m, k, n = actual
    if _logical_shape(acc.shape) != (m, n):
        raise ValueError("tla.mmad acc must have the actual (real_m, real_n) shape")
    unit_flag_ctrl, unit_flag_group_id = _unit_flag_args(unit_flag_ctrl, unit_flag_group_id)
    _semantic.builder.create_hivm_mmadl1(a.handle, b.handle, acc.handle,
                                         init_condition=bool(_unwrap_if_constexpr(clear_accum)), real_m=m, real_k=k,
                                         real_n=n, transpose_b=transpose_b, unit_flag_ctrl=unit_flag_ctrl,
                                         unit_flag_group_id=unit_flag_group_id, transpose_a=transpose_a)


def _mx_operand(operand, transpose, format, name):
    """Stored (rows, cols), logical-K factor and native format code.

    The L1 buffer shape is the *stored* shape: packed FP4 keeps two logical K
    values per byte, so its logical reduction extent is twice the stored one.
    ``rows``/``cols`` are the stored dims after the native transpose flag.
    """
    if not isinstance(operand, buffer):
        raise TypeError(f"tla.mmad_mx expects tla.buffer operands for {name}")
    if operand.space != "l1":
        raise ValueError(f"tla.mmad_mx {name} must be an L1 buffer")
    resolved, code = _mx_format(format, operand.dtype, name)
    rows, cols = _logical_shape(operand.shape)
    if transpose:
        rows, cols = cols, rows
    return rows, cols, _mx_k_factor(resolved), code


def _mx_scale(scale, rows, k32, name):
    if not isinstance(scale, buffer):
        raise TypeError(f"tla.mmad_mx expects tla.buffer operands for {name}")
    if scale.space != "l1":
        raise ValueError(f"tla.mmad_mx {name} must be an L1 buffer")
    if scale.dtype not in (int8, uint8):
        raise ValueError(f"tla.mmad_mx {name} must be an int8/uint8 E8M0 buffer")
    actual_rows, actual_k32 = _scale_logical_shape(scale.shape)
    # A 4-D zZ/nN scale buffer pads an odd K-group count to a full 16x2 tile;
    # a 2-D ND buffer must match the logical extent exactly.
    expected_k32 = ((k32 + 1) // 2) * 2 if len(scale.shape) == 4 else k32
    if actual_rows != rows or actual_k32 != expected_k32:
        raise ValueError(f"tla.mmad_mx {name} must have logical shape ({rows}, {k32})")


@builtin
def mmad_mx(a, b, scale_a, scale_b, acc, *, format_a=None, format_b=None, transpose_a=False, transpose_b=False,
            clear_accum=False, bias=None, real_m=None, real_k=None, real_n=None, _semantic=None):
    """Multiplies microscaled L1 matrices into L0C using ``hivm.hir.mmadmxL1``.

    Computes ``acc = MX(a, scale_a) @ MX(b, scale_b) (+bias)``, where each
    group of 32 logical K elements of ``a``/``b`` is scaled by one E8M0 byte.
    This is the Ascend950PR (A5) microscaling Cube instruction.

    :param a: L1 left operand. Its element type is ``tl.float8e5``,
        ``tl.float8e4nv`` or a packed 8-bit integer type. For packed FP4 the
        K axis stores two logical values per byte.
    :type a: tla.buffer
    :param b: L1 right operand with the same element type rules as ``a``.
    :type b: tla.buffer
    :param scale_a: L1 E8M0 (``int8``/``uint8``) scale buffer with logical
        shape ``(M, K // 32)``, or the equivalent 4-D zZ shape
        ``(M // 16, (K // 32 + 1) // 2, 16, 2)``.
    :type scale_a: tla.buffer
    :param scale_b: L1 E8M0 scale buffer with logical shape ``(N, K // 32)``,
        or the equivalent 4-D nN shape.
    :type scale_b: tla.buffer
    :param acc: L0C accumulator with logical shape ``(real_m, real_n)``.
    :type acc: tla.buffer
    :param format_a: Compile-time :class:`tla.MxFormat` of ``a``. Required for
        packed integer operands and inferred from FP8 element types. Defaults
        to ``None``.
    :type format_a: tla.MxFormat or None
    :param format_b: Compile-time :class:`tla.MxFormat` of ``b``, with the same
        rules as ``format_a``.
    :type format_b: tla.MxFormat or None
    :param transpose_a: Compile-time native A-transpose flag. Defaults to
        ``False``.
    :type transpose_a: bool
    :param transpose_b: Compile-time native B-transpose flag. Defaults to
        ``False``.
    :type transpose_b: bool
    :param clear_accum: Whether to initialize the accumulator before this
        multiplication. Defaults to ``False``.
    :type clear_accum: bool
    :param bias: Optional L1 per-channel bias with logical shape ``(N,)``,
        ``(1, N)`` or a 4-D zN shape. Defaults to ``None``.
    :type bias: tla.buffer or None
    :param real_m: Actual M extent, or ``None`` (default) for the full ``M``.
    :type real_m: int or None
    :param real_k: Actual reduction extent along the operands' K axis, or
        ``None`` (default) for the full extent. For packed FP4 this is the
        stored byte count (half the logical ``K``).
    :type real_k: int or None
    :param real_n: Actual N extent, or ``None`` (default) for the full ``N``.
    :type real_n: int or None
    :return: None. The result is stored in ``acc``.
    :rtype: None

    .. note::
        Use inside ``tla.scope.cube()``. Load scales with
        :func:`tla.load_gm_to_l1_mx_scale` so the backend emits the MX scale
        DMA instead of an ND-to-NZ conversion. FP4 requires both operands to
        use ``MxFormat.E2M1``; mixed FP8/FP4 is not supported by the backend.
        ``acc`` must match the actual ``(real_m, real_n)`` extents exactly.
    """
    transpose_a = bool(_unwrap_if_constexpr(transpose_a))
    transpose_b = bool(_unwrap_if_constexpr(transpose_b))
    m, k_stored, factor_a, code_a = _mx_operand(a, transpose_a, format_a, "a")
    k_b_stored, n, factor_b, code_b = _mx_operand(b, transpose_b, format_b, "b")
    if factor_a != factor_b:
        raise ValueError("tla.mmad_mx does not support mixed FP8/FP4 operands")
    if k_stored != k_b_stored:
        raise ValueError("tla.mmad_mx reduction dimensions must match after transposition")
    k_logical = k_stored * factor_a
    if k_logical % _MX_SCALE_GROUP:
        raise ValueError("tla.mmad_mx reduction extent must be a multiple of 32")
    if not isinstance(acc, buffer):
        raise TypeError("tla.mmad_mx expects a tla.buffer for acc")
    if acc.space != "l0c":
        raise ValueError("tla.mmad_mx acc must be an L0C buffer")
    # Scales describe the full L1 tile, independent of any real_* sub-extent.
    full_m, full_n = m, n
    actual = []
    for name, bound, extent in zip(("real_m", "real_k", "real_n"), (m, k_stored, n), (real_m, real_k, real_n)):
        extent = _unwrap_if_constexpr(extent)
        if extent is None:
            extent = bound
        elif type(extent) is not int or extent <= 0 or extent > bound or extent % 16:
            raise ValueError(f"tla.mmad_mx {name} must be a positive multiple of 16 within {bound}")
        actual.append(extent)
    m, k_stored, n = actual
    if _logical_shape(acc.shape) != (m, n):
        raise ValueError("tla.mmad_mx acc must have the actual (real_m, real_n) shape")
    k32 = k_logical // _MX_SCALE_GROUP
    _mx_scale(scale_a, full_m, k32, "scale_a")
    _mx_scale(scale_b, full_n, k32, "scale_b")
    if bias is not None:
        if not isinstance(bias, buffer) or bias.space != "l1":
            raise ValueError("tla.mmad_mx bias must be an L1 buffer")
        if bias.dtype != acc.dtype:
            raise ValueError("tla.mmad_mx bias must have the accumulator element type")
    # The native op's real_k is the *stored* reduction extent: the template
    # multiplies it by the FP4 packing factor internally.
    _semantic.builder.create_hivm_mmadmx(a.handle, b.handle, scale_a.handle, scale_b.handle, acc.handle,
                                         init_condition=bool(_unwrap_if_constexpr(clear_accum)), real_m=m,
                                         real_k=k_stored, real_n=n, lhs_format=code_a, rhs_format=code_b,
                                         transpose_a=bool(_unwrap_if_constexpr(transpose_a)),
                                         transpose_b=bool(_unwrap_if_constexpr(transpose_b)),
                                         per_channel_bias=bias.handle if bias is not None else None)
