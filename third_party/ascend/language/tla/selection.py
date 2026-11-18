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
"""Eager scalar selection for TLA kernels."""

import triton.language as tl
from triton.language.core import builtin, tensor, _unwrap_if_constexpr

__all__ = ["where"]


def _scalar_operand(value, name):
    value = _unwrap_if_constexpr(value)
    if isinstance(value, tensor):
        if value.type.is_block():
            raise TypeError(f"tla.where {name} must be a scalar value, got a block tensor")
        if value.dtype.is_int() or value.dtype.is_floating():
            return value
    elif isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"tla.where {name} must be a scalar boolean, integer, or floating-point value")


@builtin
def where(condition, x, y, _semantic=None):
    """Select one of two scalar values under a bool or rank-0 int1 condition.

    The two value expressions are evaluated before the call and promoted
    using Triton's scalar rules.
    """
    condition = _unwrap_if_constexpr(condition)
    if isinstance(condition, tensor):
        if condition.type.is_block():
            raise TypeError("tla.where condition must be a scalar, got a block tensor")
        if condition.dtype != tl.int1:
            raise TypeError(f"tla.where condition must have int1 dtype, got {condition.dtype}")
    elif not isinstance(condition, bool):
        raise TypeError("tla.where condition must be a bool or a scalar int1 tl.tensor")

    x = _scalar_operand(x, "x")
    y = _scalar_operand(y, "y")
    return tl.where(condition, x, y, _semantic=_semantic)
