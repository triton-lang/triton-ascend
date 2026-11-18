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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Scalar square root and integer remainder for TLA."""

import triton.language as tl
from triton.language.core import builtin, tensor, _unwrap_if_constexpr

__all__ = ["truncmod", "sqrt"]


@builtin
def truncmod(a, b, _semantic=None):
    """
    Integer remainder with division truncated toward zero.

    Like TileLang's ``T.truncmod``, a nonzero remainder has the sign of ``a``:
    ``truncmod(-7, 3)`` is -1 and ``truncmod(7, -3)`` is 1, unlike Python's
    floor-based ``%``. Integer type promotion follows Triton; mixed signed
    and unsigned tensor operands are rejected. Floating-point operands,
    block tensors, and SIMD vectors are not supported.

    :param a: integer scalar tensor or Python/constexpr integer dividend
    :param b: integer scalar tensor or Python/constexpr integer divisor;
        must be nonzero. Signed minimum divided by -1 is also undefined
    :return: scalar integer ``tl.tensor`` with the promoted operand dtype
    """
    a = _unwrap_if_constexpr(a)
    b = _unwrap_if_constexpr(b)
    for name, value in (("a", a), ("b", b)):
        if isinstance(value, tensor):
            if value.type.is_block() or not value.dtype.is_int() or value.dtype == tl.int1:
                raise TypeError(f"truncmod {name} must be a non-boolean scalar integer tensor")
        elif not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"truncmod {name} must be a scalar integer or integer tensor")
    if isinstance(b, int) and b == 0:
        raise ValueError("truncmod divisor must be nonzero")
    # Preserve Python scalar promotion rules; do not pre-convert to tensors.
    return _semantic.mod(a, b)


@builtin
def sqrt(x, _semantic=None):
    """Return the square root of a rank-0 FP32 tensor or an FP32-inferred Python float."""
    x = _unwrap_if_constexpr(x)
    if isinstance(x, tensor):
        if x.type.is_block():
            raise TypeError("tla.sqrt expects a scalar, got a block tensor")
        if x.dtype != tl.float32:
            raise TypeError(f"sqrt supports float32 scalars, got {x.dtype}")
    elif not isinstance(x, float):
        raise TypeError("tla.sqrt expects a float32 scalar")
    x = _semantic.to_tensor(x)
    if x.dtype != tl.float32:
        raise TypeError(f"sqrt supports float32 scalars, got {x.dtype}")
    return tl.sqrt(x, _semantic=_semantic)
