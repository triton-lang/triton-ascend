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
Inputs may use logical 2-D shapes or physical 4-D NZ shapes. Only
``transpose_b`` is exposed; ``transpose_a`` folds into the GM->L1 copy.
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
def mmad(a, b, acc, transpose_b=False, clear_accum=False, unit_flag_ctrl=None, *, unit_flag_group_id=None,
         _semantic=None):
    """Multiply L1 buffers into L0C, optionally clearing the accumulator first.

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
    if k != k2 or _logical_shape(acc.shape) != (m, n):
        raise ValueError("tla.mmad shape mismatch: a=(M,K), b=(K,N), acc=(M,N)")
    unit_flag_ctrl, unit_flag_group_id = _unit_flag_args(unit_flag_ctrl, unit_flag_group_id)
    _semantic.builder.create_hivm_mmadl1(a.handle, b.handle, acc.handle,
                                         init_condition=bool(_unwrap_if_constexpr(clear_accum)), real_m=m, real_k=k,
                                         real_n=n, transpose_b=bool(_unwrap_if_constexpr(transpose_b)),
                                         unit_flag_ctrl=unit_flag_ctrl, unit_flag_group_id=unit_flag_group_id)
