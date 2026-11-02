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
"""Cube-core primitives for TLA (hivm.hir.mmadL1).

``mmad`` computes ``C = A x B (+C)`` with A=(M,K)/B=(K,N) in L1 and C in L0C.
Inputs may use logical 2-D shapes or physical 4-D NZ shapes. Transposition
is applied to the logical matrix dimensions represented by each L1 buffer.
"""
from triton.language.core import builtin, _unwrap_if_constexpr
from .core import buffer
from .dma import _unit_flag_args

__all__ = ["mmad"]


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
    """Multiply L1 buffers into L0C, optionally clearing the accumulator first.

    ``transpose_a`` and ``transpose_b`` transpose the logical matrices stored
    in the input buffers. A=(K,M), B=(K,N), transpose_a=True computes C=(M,N);
    the transpose is carried by MMAD rather than a separate data movement.

    ``real_m``, ``real_k`` and ``real_n`` restrict the computation to a leading
    submatrix after transposition. Explicit extents must be positive
    compile-time multiples of 16 within the corresponding input dimensions.
    Omitted extents use the full dimensions. The L0C buffer must have exactly
    the actual (real_m, real_n) logical shape, matching its packed output layout.

    ``unit_flag_ctrl`` and ``unit_flag_group_id`` pass through to HIVM's
    UnitFlag interface. None leaves each attribute unset; modes 0/2/3 mean
    disabled, enabled without update, and enabled with update respectively.
    To test manual modes without backend automatic UnitFlag inference
    rewriting them, launch with ``unit_flag=False``.
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
