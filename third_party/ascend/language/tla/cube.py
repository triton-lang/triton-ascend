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
"""Cube-core primitives for TLA (hivm.hir.mmadL1 / hivm.hir.mmadL0).

``mmad`` computes ``C = A x B (+C)`` with A=(M,K)/B=(K,N) in L1 and C in L0C.
Inputs may use logical 2-D shapes or physical 4-D NZ shapes. Transposition
is applied to the logical matrix dimensions represented by each L1 buffer.
``mmad_l0`` consumes explicit physical L0A/L0B tiles and writes L0C.
"""
from triton.language.core import builtin, float16, float32, tensor, _unwrap_if_constexpr
from .core import buffer, _row_major_strides
from .dma import _boolean, _unit_flag_args

__all__ = ["mmad", "mmad_l0"]


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


def _check_mmad_l0_buffer(value, name, space, dtype, shape, capacity, alignment):
    if not isinstance(value, buffer):
        raise TypeError(f"tla.cube.mmad_l0 {name} must be a tla.buffer")
    if value.space != space:
        raise ValueError(f"tla.cube.mmad_l0 {name} must be in {space}")
    if value.dtype != dtype:
        raise ValueError(f"tla.cube.mmad_l0 {name} must have dtype {dtype}")
    if value.shape != shape:
        raise ValueError(f"tla.cube.mmad_l0 {name} must have exact physical shape {shape}")
    if value.strides != _row_major_strides(shape):
        raise ValueError(f"tla.cube.mmad_l0 {name} must have compact physical strides")
    elements = 1
    for extent in shape:
        elements *= extent
    size = elements * dtype.itemsize
    if size > capacity:
        raise ValueError(f"tla.cube.mmad_l0 {name} exceeds {capacity}-byte capacity")
    # None means the Python view metadata does not know the offset. Preserve
    # its SSA value; npuir resolves BT offsets statically and checks runtime
    # L0 addresses. A Python-unknown offset may still fold to a constant IR offset.
    if value.offset is not None:
        offset = value.offset
        if (type(offset) is not int or offset < 0 or offset * dtype.itemsize + size > capacity
                or offset * dtype.itemsize % alignment):
            raise ValueError(f"tla.cube.mmad_l0 {name} offset exceeds capacity or violates {alignment}-byte alignment")


@builtin
def mmad_l0(a, b, acc, *, m, n, k, bias=None, clear_accum=False, disable_gemv=True, unit_flag_ctrl=None,
            _semantic=None):
    """Multiplies physical L0 tiles into L0C using A5 ``hivm.hir.mmadL0``.

    :param a: FP16 L0A zN buffer with shape ``(k//16, m//16, 16, 16)``.
    :type a: tla.buffer
    :param b: FP16 L0B nZ buffer with shape ``(k//16, n//16, 16, 16)``.
        This differs from the physical L1 B layout used by :func:`mmad`.
    :type b: tla.buffer
    :param acc: FP32 L0C buffer with shape ``(n//16, m//16, 16, 16)``.
    :type acc: tla.buffer
    :param m: Compile-time M extent, a multiple of 16 in ``[16, 2048]``.
    :type m: int
    :param n: Compile-time N extent, a multiple of 16 in ``[16, 2048]``.
    :type n: int
    :param k: Compile-time K extent, a multiple of 16 in ``[16, 2048]``.
    :type k: int
    :param bias: Optional FP32 BT buffer of shape ``(n,)``. With bias,
        computes ``acc = a @ b + broadcast(bias)`` and replaces the old
        accumulator; ``clear_accum`` must be the compile-time value False.
    :type bias: tla.buffer or None
    :param clear_accum: Without bias, True computes ``acc = a @ b`` and
        False computes ``acc += a @ b``. A runtime scalar i1 is also accepted.
        Defaults to False; the caller must initialize acc before accumulating.
    :type clear_accum: bool or scalar tl.tensor of dtype tl.int1
    :param disable_gemv: Compile-time bool. The A5 implementation requires True.
    :type disable_gemv: bool
    :param unit_flag_ctrl: Compile-time UnitFlag mode. Only None (attribute
        absent) or ``tla.UnitFlagMode.DISABLED`` (0) is supported on A5.
    :type unit_flag_ctrl: tla.UnitFlagMode, int or None
    :return: None. The result is stored in ``acc``.
    :rtype: None

    .. note::
        Use inside ``tla.scope.cube()`` with an A5 backend that supports
        standalone MMAD L0. All buffers must have the exact physical shape
        and compact row-major strides, including size-one dimensions. Tile
        footprints must fit L0A/L0B (64 KiB each), L0C (256 KiB), and BT
        (4 KiB). A/B, acc and bias require 512, 1024 and 64 byte alignment,
        respectively. Compact subviews and nonzero offsets are supported;
        npuir must resolve the BT offset statically. Runtime L0 addresses
        and offsets are checked by the backend.

        This operation emits no copies, K loop or cross-pipe synchronization.
        Fill A/B with ``copy_l1_to_l0`` and bias with ``copy_l1_to_bt``;
        the caller owns buffer reuse and pipeline events. The CATLASS template
        owns the small-tile PIPE_M barrier. Standalone MMAD cannot share a
        function with L1 matmul/conv templates that own implicit L0 storage.
        Launch with ``unit_flag=False`` when using explicit UnitFlag settings.
    """
    m, n, k = (_unwrap_if_constexpr(value) for value in (m, n, k))
    for name, extent in (("m", m), ("n", n), ("k", k)):
        if type(extent) is not int or not 16 <= extent <= 2048 or extent % 16:
            raise ValueError(f"tla.cube.mmad_l0 {name} must be a compile-time multiple of 16 in [16, 2048]")
    _check_mmad_l0_buffer(a, "a", "l0a", float16, (k // 16, m // 16, 16, 16), 64 * 1024, 512)
    _check_mmad_l0_buffer(b, "b", "l0b", float16, (k // 16, n // 16, 16, 16), 64 * 1024, 512)
    _check_mmad_l0_buffer(acc, "acc", "l0c", float32, (n // 16, m // 16, 16, 16), 256 * 1024, 1024)
    bias = _unwrap_if_constexpr(bias)
    if bias is not None:
        _check_mmad_l0_buffer(bias, "bias", "bt", float32, (n,), 4 * 1024, 64)
    clear_accum = _unwrap_if_constexpr(clear_accum)
    if bias is not None and clear_accum is not False:
        raise ValueError("tla.cube.mmad_l0 with bias requires compile-time clear_accum=False")
    if isinstance(clear_accum, tensor):
        if clear_accum.shape or not clear_accum.dtype.is_bool():
            raise TypeError("tla.cube.mmad_l0 clear_accum must be a bool or scalar i1 tensor")
        init_condition = clear_accum.handle
    else:
        init_condition = _boolean(clear_accum, "tla.cube.mmad_l0 clear_accum")
    disable_gemv = _boolean(disable_gemv, "tla.cube.mmad_l0 disable_gemv")
    if not disable_gemv:
        raise ValueError("tla.cube.mmad_l0 on A5 requires disable_gemv=True")
    unit_flag_ctrl, _ = _unit_flag_args(unit_flag_ctrl, None)
    if unit_flag_ctrl not in (None, 0):
        raise ValueError("tla.cube.mmad_l0 on A5 supports only DISABLED unit_flag_ctrl")
    _semantic.builder.create_hivm_mmadl0(a.handle, b.handle, acc.handle, init_condition=init_condition,
                                         real_m=m, real_k=k, real_n=n, bias=None if bias is None else bias.handle,
                                         disable_gemv=disable_gemv, unit_flag_ctrl=unit_flag_ctrl)
