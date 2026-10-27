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

``mmad`` computes ``C = A x B (+C)`` with A=(M,K)/B=(K,N) in L1 and C in L0C,
mirroring tilelang ascend's L1-path ``gemm`` (only ``transpose_b`` is exposed;
``transpose_a`` folds into the GM->L1 copy and is out of v1 scope).
"""
from triton.language.core import builtin, _unwrap_if_constexpr
from .core import buffer

__all__ = ["mmad"]


@builtin
def mmad(a, b, acc, transpose_b=False, clear_accum=False, _semantic=None):
    if not (isinstance(a, buffer) and isinstance(b, buffer) and isinstance(acc, buffer)):
        raise TypeError("tla.mmad expects tla.buffer operands for a, b, acc")
    if a.space != "l1" or b.space != "l1":
        raise ValueError("tla.mmad a/b must be L1 buffers (fill via tla.copy(gm->l1))")
    if acc.space != "l0c":
        raise ValueError("tla.mmad acc must be an L0C buffer")
    if a.dtype != b.dtype:
        raise ValueError("tla.mmad a and b must have the same element type")
    if len(a.shape) != 2 or len(b.shape) != 2 or len(acc.shape) != 2:
        raise ValueError("tla.mmad expects 2-D buffers (M,K)/(K,N)/(M,N)")
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or acc.shape != (m, n):
        raise ValueError("tla.mmad shape mismatch: a=(M,K), b=(K,N), acc=(M,N)")
    _semantic.builder.create_hivm_mmadl1(a.handle, b.handle, acc.handle, bool(_unwrap_if_constexpr(clear_accum)), m, k,
                                         n, bool(_unwrap_if_constexpr(transpose_b)))
